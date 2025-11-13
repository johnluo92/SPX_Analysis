"""Probabilistic VIX Forecasting System - Multi-output XGBoost models"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from config import TARGET_CONFIG, XGBOOST_CONFIG

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilisticVIXForecaster:
    def __init__(self):
        self.horizon = TARGET_CONFIG["horizon_days"]
        self.quantiles = TARGET_CONFIG["quantiles"]["levels"]
        self.regime_boundaries = TARGET_CONFIG["regimes"]["boundaries"]
        self.regime_labels = TARGET_CONFIG["regimes"]["labels"]
        self.models = {}
        self.calibrators = {}
        self.feature_names = None

    def train(self, df: pd.DataFrame, save_dir: str = "models"):
        """
        Train all cohort models and save to disk.

        Args:
            df: Feature dataframe with calendar_cohort column
            save_dir: Directory to save trained models
        """
        logger.info("=" * 80)
        logger.info("Training Probabilistic VIX Forecaster")
        logger.info("=" * 80)

        # Store feature names (exclude target and metadata columns)
        exclude_cols = [
            "calendar_cohort",
            "cohort_weight",
            "feature_quality",
            "target_point",
            "target_regime",
            "target_confidence",
        ]
        exclude_cols += [col for col in df.columns if col.startswith("target_")]

        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"Feature count: {len(self.feature_names)}")
        logger.info(f"Training samples: {len(df)}")

        # Create all target variables
        df_with_targets = self._create_targets(df)

        # Drop rows with missing targets
        df_clean = df_with_targets.dropna(subset=["target_point"])
        logger.info(f"Clean samples (after removing NaN targets): {len(df_clean)}")

        # Get unique cohorts
        cohorts = sorted(df_clean["calendar_cohort"].unique())
        logger.info(f"\nCohorts to train: {cohorts}")

        # Train each cohort
        all_metrics = {}

        for cohort in cohorts:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Training cohort: {cohort}")
            logger.info(f"{'=' * 80}")

            cohort_df = df_clean[df_clean["calendar_cohort"] == cohort].copy()
            logger.info(f"Cohort samples: {len(cohort_df)}")

            try:
                metrics = self._train_cohort_models(cohort, cohort_df)
                all_metrics[cohort] = metrics

                # Save models for this cohort
                self._save_cohort_models(cohort, save_dir)
                logger.info(f"✅ Trained: {cohort}")

            except Exception as e:
                logger.error(f"❌ Failed to train cohort {cohort}: {e}")
                raise

        # Generate diagnostics
        logger.info(f"\n{'=' * 80}")
        logger.info("Generating diagnostics...")
        self._generate_diagnostics(all_metrics, save_dir)

        logger.info(f"\n📊 Total cohorts loaded: {len(self.models)}")
        logger.info(f"✅ Training complete!")

        return self

    def _get_adaptive_cv_config(self, n_samples: int) -> Tuple[int, int]:
        if n_samples < 200:
            n_splits = 2
        elif n_samples < 400:
            n_splits = 3
        elif n_samples < 800:
            n_splits = 4
        else:
            n_splits = 5

        max_test_size = n_samples // (n_splits + 1)
        test_size = max(int(max_test_size * 0.8), 30)

        while (n_samples - test_size) < n_splits * test_size and n_splits > 2:
            n_splits -= 1
            max_test_size = n_samples // (n_splits + 1)
            test_size = int(max_test_size * 0.8)

        return n_splits, test_size

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        future_vix = df["vix"].shift(-self.horizon)
        df["target_point"] = ((future_vix / df["vix"]) - 1) * 100
        point_min, point_max = TARGET_CONFIG["point_estimate"]["range"]
        df["target_point"] = df["target_point"].clip(point_min, point_max)

        for q in self.quantiles:
            df[f"target_q{int(q * 100)}"] = df["target_point"]

        regime_bins = [-np.inf] + self.regime_boundaries + [np.inf]
        df["target_regime"] = pd.cut(
            future_vix,
            bins=regime_bins,
            labels=list(range(len(self.regime_labels))),
            include_lowest=True,
        ).astype(float)

        regime_volatility = df["vix"].rolling(21, min_periods=10).std()
        regime_stability = 1 / (
            1 + regime_volatility / df["vix"].rolling(21, min_periods=10).mean()
        )
        regime_stability = regime_stability.fillna(0.5)

        df["target_confidence"] = (
            0.5 * df["feature_quality"].fillna(0.5) + 0.5 * regime_stability
        ).clip(0, 1)

        return df

    def predict(self, X: pd.DataFrame, cohort: str) -> Dict:
        """
        Generate probabilistic forecast using point + uncertainty + direction models.

        Args:
            X: Feature dataframe (single row)
            cohort: Calendar cohort to use

        Returns:
            Dictionary with point_estimate, quantiles, direction_probability, etc.
        """
        if cohort not in self.models:
            raise ValueError(
                f"Cohort {cohort} not trained. Available: {list(self.models.keys())}"
            )

        X_features = X[self.feature_names]

        # Point estimate (this is also q50)
        point = self.models[cohort]["point"].predict(X_features)[0]

        # Uncertainty (standard deviation of prediction error)
        uncertainty = max(
            self.models[cohort]["uncertainty"].predict(X_features)[0],
            1.0,  # Minimum uncertainty of 1%
        )

        # Generate quantiles using normal distribution z-scores
        z_scores = {
            "q10": -1.28,
            "q25": -0.67,
            "q50": 0.00,  # By definition, equals point estimate
            "q75": 0.67,
            "q90": 1.28,
        }
        quantiles = {q: point + z * uncertainty for q, z in z_scores.items()}

        # Direction probability (probability VIX goes up)
        prob_up = float(
            self.models[cohort]["direction"].predict_proba(X_features)[0][1]
        )

        # Confidence score
        confidence = np.clip(
            self.models[cohort]["confidence"].predict(X_features)[0], 0, 1
        )

        return {
            "point_estimate": float(point),
            "quantiles": {k: float(v) for k, v in quantiles.items()},
            "direction_probability": prob_up,
            "confidence_score": float(confidence),
            "cohort": cohort,
        }

    def _train_cohort_models(self, cohort: str, df: pd.DataFrame) -> Dict:
        """
        Train 3 models for a single cohort: point, uncertainty, direction

        Args:
            cohort: Cohort name
            df: Dataframe filtered to this cohort

        Returns:
            Dictionary of metrics for each model
        """
        X = df[self.feature_names]

        if len(df) < 100:
            raise ValueError(f"Insufficient samples for cohort {cohort}: {len(df)}")

        # Initialize storage
        self.models[cohort] = {}
        self.calibrators[cohort] = {}
        metrics = {}

        # -------------------------------------------------------------------------
        # Model 1: Point Estimate (this becomes the median q50)
        # -------------------------------------------------------------------------
        logger.info("  Training point estimate model...")
        y_point = df["target_point"]
        model_point, metric_point = self._train_regressor(
            X, y_point, objective="reg:squarederror", eval_metric="rmse"
        )
        self.models[cohort]["point"] = model_point
        metrics["point"] = metric_point
        logger.info(f"    RMSE: {metric_point['rmse']:.3f}")

        # -------------------------------------------------------------------------
        # Model 2: Uncertainty (predicts absolute error for confidence intervals)
        # -------------------------------------------------------------------------
        logger.info("  Training uncertainty model...")
        train_predictions = model_point.predict(X)
        y_uncertainty = np.abs(df["target_point"] - train_predictions)

        model_uncertainty, metric_uncertainty = self._train_regressor(
            X, y_uncertainty, objective="reg:squarederror", eval_metric="rmse"
        )
        self.models[cohort]["uncertainty"] = model_uncertainty
        metrics["uncertainty"] = metric_uncertainty
        logger.info(f"    RMSE: {metric_uncertainty['rmse']:.3f}")

        # -------------------------------------------------------------------------
        # Model 3: Direction (binary: up or down)
        # -------------------------------------------------------------------------
        logger.info("  Training direction classifier...")
        y_direction = (df["target_point"] > 0).astype(int)

        model_direction, metric_direction = self._train_classifier(
            X, y_direction, num_classes=2
        )
        self.models[cohort]["direction"] = model_direction
        metrics["direction"] = metric_direction
        logger.info(f"    Accuracy: {metric_direction.get('accuracy', 0):.3f}")

        # Calibrate direction probabilities
        logger.info("  Calibrating direction probabilities...")
        calibrators = self._calibrate_probabilities(model_direction, X, y_direction)
        self.calibrators[cohort]["direction"] = calibrators

        # -------------------------------------------------------------------------
        # Model 4: Confidence Score (kept for compatibility)
        # -------------------------------------------------------------------------
        logger.info("  Training confidence model...")
        y_confidence = df["target_confidence"]
        model_conf, metric_conf = self._train_regressor(
            X, y_confidence, objective="reg:squarederror", eval_metric="rmse"
        )
        self.models[cohort]["confidence"] = model_conf
        metrics["confidence"] = metric_conf
        logger.info(f"    RMSE: {metric_conf['rmse']:.3f}")

        return metrics

    def _train_regressor(self, X, y, objective, eval_metric, quantile_alpha=None):
        params = XGBOOST_CONFIG["shared_params"].copy()
        params["objective"] = objective

        if quantile_alpha:
            params["quantile_alpha"] = quantile_alpha

        n_splits, test_size = self._get_adaptive_cv_config(len(X))

        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=XGBOOST_CONFIG["cv_config"]["gap"],
        )

        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            cv_params = params.copy()
            cv_params["early_stopping_rounds"] = 50

            model = XGBRegressor(**cv_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = model.predict(X_val)

            if eval_metric == "rmse":
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            elif eval_metric == "mae":
                score = mean_absolute_error(y_val, y_pred)

            cv_scores.append(score)

        final_params = params.copy()
        if "early_stopping_rounds" in final_params:
            del final_params["early_stopping_rounds"]

        final_model = XGBRegressor(**final_params)
        final_model.fit(X, y, verbose=False)

        metrics = {
            eval_metric: np.mean(cv_scores),
            f"{eval_metric}_std": np.std(cv_scores),
            "n_splits": n_splits,
            "test_size": test_size,
        }

        return final_model, metrics

    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {
                self._convert_to_json_serializable(
                    k
                ): self._convert_to_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _train_classifier(self, X, y, num_classes):
        class_counts = pd.Series(y).value_counts().sort_index()
        n_samples = len(X)

        # For binary classification, just use the classes as-is
        unique_classes = np.sort(y.unique())
        class_mapping = {int(old): int(new) for new, old in enumerate(unique_classes)}
        y_relabeled = y.map(class_mapping)
        expected_classes = np.arange(len(unique_classes))
        inverse_mapping = {int(new): int(old) for old, new in class_mapping.items()}

        params = XGBOOST_CONFIG["shared_params"].copy()

        # For binary classification
        if len(unique_classes) == 2:
            params["objective"] = "binary:logistic"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = len(unique_classes)

        n_splits, test_size = self._get_adaptive_cv_config(n_samples)
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=XGBOOST_CONFIG["cv_config"]["gap"],
        )

        cv_accuracy = []
        cv_logloss = []
        valid_folds = 0
        skipped_folds = 0

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_relabeled.iloc[train_idx], y_relabeled.iloc[val_idx]

            train_classes = np.sort(y_train.unique())

            if not np.array_equal(train_classes, expected_classes):
                skipped_folds += 1
                continue

            cv_params = params.copy()
            cv_params["early_stopping_rounds"] = 50

            model = XGBClassifier(**cv_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            accuracy = (y_pred == y_val).mean()
            logloss = log_loss(y_val, y_proba, labels=expected_classes)

            cv_accuracy.append(accuracy)
            cv_logloss.append(logloss)
            valid_folds += 1

        final_params = params.copy()
        if "early_stopping_rounds" in final_params:
            del final_params["early_stopping_rounds"]

        final_model = XGBClassifier(**final_params)
        final_model.fit(X, y_relabeled, verbose=False)

        final_model.class_mapping_ = class_mapping
        final_model.inverse_mapping_ = inverse_mapping
        final_model.effective_classes_ = len(unique_classes)

        if valid_folds == 0:
            metrics = {
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "log_loss": 999.0,
                "log_loss_std": 0.0,
                "num_classes": len(unique_classes),
                "class_mapping": class_mapping,
                "valid_folds": 0,
                "skipped_folds": skipped_folds,
            }
        else:
            metrics = {
                "accuracy": np.mean(cv_accuracy),
                "accuracy_std": np.std(cv_accuracy),
                "log_loss": np.mean(cv_logloss),
                "log_loss_std": np.std(cv_logloss),
                "num_classes": len(unique_classes),
                "class_mapping": class_mapping,
                "valid_folds": valid_folds,
                "skipped_folds": skipped_folds,
                "n_splits": n_splits,
                "test_size": test_size,
            }

        return final_model, metrics

    def _calibrate_probabilities(self, model, X, y):
        """Calibrate binary direction classifier probabilities."""
        y_proba = model.predict_proba(X)

        # Only 2 classes now (down=0, up=1)
        calibrators = []

        for class_idx in range(2):
            y_binary = (y == class_idx).astype(int)

            if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(y_proba[:, class_idx], y_binary)
                calibrators.append(calibrator)
            else:
                calibrators.append(None)

        return calibrators

    def _save_cohort_models(self, cohort: str, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        cohort_file = save_path / f"probabilistic_forecaster_{cohort}.pkl"

        with open(cohort_file, "wb") as f:
            pickle.dump(
                {
                    "models": self.models[cohort],
                    "calibrators": self.calibrators.get(cohort, {}),
                    "feature_names": self.feature_names,
                    "config": {
                        "horizon": self.horizon,
                        "quantiles": self.quantiles,
                        "regime_boundaries": self.regime_boundaries,
                        "regime_labels": self.regime_labels,
                    },
                },
                f,
            )

    def load(self, cohort: str, load_dir: str = "models"):
        load_path = Path(load_dir) / f"probabilistic_forecaster_{cohort}.pkl"

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.models[cohort] = data["models"]
        self.calibrators[cohort] = data["calibrators"]
        self.feature_names = data["feature_names"]

        config = data["config"]
        self.horizon = config["horizon"]
        self.quantiles = config["quantiles"]
        self.regime_boundaries = config["regime_boundaries"]
        self.regime_labels = config["regime_labels"]

    def _generate_diagnostics(self, cohort_metrics: Dict, save_dir: str):
        """Generate diagnostic plots and save metrics."""
        save_path = Path(save_dir)

        # Save metrics as JSON
        metrics_file = save_path / "probabilistic_model_metrics.json"
        with open(metrics_file, "w") as f:
            json_safe_metrics = self._convert_to_json_serializable(cohort_metrics)
            json.dump(json_safe_metrics, f, indent=2)

        logger.info(f"  Saved metrics: {metrics_file}")

        # Generate plots
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Model Performance by Cohort", fontsize=16, fontweight="bold")

            cohorts = list(cohort_metrics.keys())

            # Plot 1: Point Estimate RMSE
            point_rmse = [cohort_metrics[c]["point"]["rmse"] for c in cohorts]
            axes[0, 0].bar(range(len(cohorts)), point_rmse, color="steelblue")
            axes[0, 0].set_xticks(range(len(cohorts)))
            axes[0, 0].set_xticklabels(cohorts, rotation=45, ha="right")
            axes[0, 0].set_ylabel("RMSE (%)")
            axes[0, 0].set_title("Point Estimate RMSE")
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Uncertainty RMSE
            uncertainty_rmse = [
                cohort_metrics[c]["uncertainty"]["rmse"] for c in cohorts
            ]
            axes[0, 1].bar(range(len(cohorts)), uncertainty_rmse, color="coral")
            axes[0, 1].set_xticks(range(len(cohorts)))
            axes[0, 1].set_xticklabels(cohorts, rotation=45, ha="right")
            axes[0, 1].set_ylabel("RMSE (%)")
            axes[0, 1].set_title("Uncertainty Model RMSE")
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Direction Accuracy
            dir_accuracy = [cohort_metrics[c]["direction"]["accuracy"] for c in cohorts]
            axes[1, 0].bar(range(len(cohorts)), dir_accuracy, color="forestgreen")
            axes[1, 0].axhline(
                0.5, color="red", linestyle="--", label="Random", linewidth=2
            )
            axes[1, 0].set_xticks(range(len(cohorts)))
            axes[1, 0].set_xticklabels(cohorts, rotation=45, ha="right")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].set_title("Direction Classification Accuracy")
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Confidence RMSE
            conf_rmse = [cohort_metrics[c]["confidence"]["rmse"] for c in cohorts]
            axes[1, 1].bar(range(len(cohorts)), conf_rmse, color="purple")
            axes[1, 1].set_xticks(range(len(cohorts)))
            axes[1, 1].set_xticklabels(cohorts, rotation=45, ha="right")
            axes[1, 1].set_ylabel("RMSE")
            axes[1, 1].set_title("Confidence Score RMSE")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = save_path / "model_performance.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"  Saved plots: {plot_file}")

        except Exception as e:
            logger.warning(f"  Could not generate plots: {e}")


def train_probabilistic_forecaster(
    df: pd.DataFrame, save_dir: str = "models"
) -> ProbabilisticVIXForecaster:
    """
    Convenience function to train and return a ProbabilisticVIXForecaster.

    Args:
        df: Feature dataframe with calendar_cohort column
        save_dir: Directory to save trained models

    Returns:
        Trained ProbabilisticVIXForecaster instance
    """
    forecaster = ProbabilisticVIXForecaster()
    forecaster.train(df, save_dir=save_dir)
    return forecaster
