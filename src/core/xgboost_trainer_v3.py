"""
Simplified VIX Forecasting Trainer - V4.0
Trains exactly 2 models:
1. Direction Classifier (binary: up/down)
2. Magnitude Regressor (continuous: log-space VIX change)

Cohorts are now features, not separate model splits.
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from config import TARGET_CONFIG, XGBOOST_CONFIG, TRAINING_END_DATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedVIXForecaster:
    """
    Simplified VIX forecaster with exactly 2 models:
    - Direction classifier (binary: up/down)
    - Magnitude regressor (log-space VIX change)

    Cohorts are encoded as binary features rather than separate models.
    """

    def __init__(self):
        self.horizon = TARGET_CONFIG["horizon_days"]
        self.direction_model = None
        self.magnitude_model = None
        self.feature_names = None
        self.metrics = {}

    def train(self, df: pd.DataFrame, save_dir: str = "models") -> 'SimplifiedVIXForecaster':
        """
        Train direction and magnitude models on the full dataset.

        Args:
            df: DataFrame with features, vix, spx, and calendar_cohort columns
            save_dir: Directory to save trained models

        Returns:
            self
        """
        logger.info("=" * 80)
        logger.info("SIMPLIFIED VIX FORECASTER - TRAINING")
        logger.info("=" * 80)
        logger.info(f"\nTraining Strategy: 2 GLOBAL models")
        logger.info(f"  1. Direction Classifier (binary)")
        logger.info(f"  2. Magnitude Regressor (continuous)")
        logger.info(f"Cohorts: Encoded as binary features")
        logger.info(f"Target: Log-space VIX change (5-day)")

        # Step 1: Create cohort binary features
        logger.info("\n[1/6] Creating cohort features...")
        df = self._create_cohort_features(df)

        # Step 2: Create targets
        logger.info("\n[2/6] Creating targets...")
        df = self._create_targets(df)

        # Step 3: Prepare feature matrix
        logger.info("\n[3/6] Preparing feature matrix...")
        X, feature_names = self._prepare_features(df)
        self.feature_names = feature_names

        # Step 4: Split train/test
        logger.info("\n[4/6] Splitting train/test...")
        train_mask = df.index <= pd.Timestamp(TRAINING_END_DATE)
        X_train = X[train_mask]
        X_test = X[~train_mask]

        y_direction_train = df.loc[train_mask, "target_direction"]
        y_direction_test = df.loc[~train_mask, "target_direction"]

        y_magnitude_train = df.loc[train_mask, "target_log_vix_change"]
        y_magnitude_test = df.loc[~train_mask, "target_log_vix_change"]

        # Remove NaN targets
        valid_train_mask = ~(y_direction_train.isna() | y_magnitude_train.isna())
        valid_test_mask = ~(y_direction_test.isna() | y_magnitude_test.isna())

        X_train = X_train[valid_train_mask]
        y_direction_train = y_direction_train[valid_train_mask]
        y_magnitude_train = y_magnitude_train[valid_train_mask]

        X_test = X_test[valid_test_mask]
        y_direction_test = y_direction_test[valid_test_mask]
        y_magnitude_test = y_magnitude_test[valid_test_mask]

        logger.info(f"  Train samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Features: {len(self.feature_names)}")

        # Step 5: Train direction model
        logger.info("\n[5/6] Training DIRECTION classifier...")
        self.direction_model, direction_metrics = self._train_direction_model(
            X_train, y_direction_train, X_test, y_direction_test
        )
        self.metrics["direction"] = direction_metrics

        # Step 6: Train magnitude model
        logger.info("\n[6/6] Training MAGNITUDE regressor...")
        self.magnitude_model, magnitude_metrics = self._train_magnitude_model(
            X_train, y_magnitude_train, X_test, y_magnitude_test
        )
        self.metrics["magnitude"] = magnitude_metrics

        # Save models
        self._save_models(save_dir)

        # Generate diagnostic plots
        self._generate_diagnostics(X_test, y_direction_test, y_magnitude_test, save_dir)

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)
        self._print_summary()

        return self

    def _create_cohort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary cohort indicator features.
        """
        df = df.copy()

        # Create binary flags for each cohort
        df["is_fomc_period"] = (df["calendar_cohort"] == "fomc_period").astype(int)
        df["is_opex_week"] = (df["calendar_cohort"] == "opex_week").astype(int)
        df["is_earnings_heavy"] = (df["calendar_cohort"] == "earnings_heavy").astype(int)
        # mid_cycle is the baseline (all flags = 0)

        logger.info(f"  Cohort distribution:")
        cohort_counts = df["calendar_cohort"].value_counts()
        for cohort, count in cohort_counts.items():
            logger.info(f"    {cohort}: {count} samples")

        return df

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create direction and magnitude targets.
        """
        df = df.copy()

        # Create 5-day forward VIX
        future_vix = df["vix"].shift(-self.horizon)
        df["future_vix"] = future_vix

        # Log-space change
        df["target_log_vix_change"] = np.log(df["future_vix"]) - np.log(df["vix"])

        # Direction (binary: 1 if up, 0 if down)
        df["target_direction"] = (df["target_log_vix_change"] > 0).astype(int)

        # Also create percentage change for reporting
        df["target_vix_pct_change"] = ((df["future_vix"] - df["vix"]) / df["vix"]) * 100

        valid_targets = (~df["target_log_vix_change"].isna()).sum()
        logger.info(f"  Valid targets: {valid_targets}")
        logger.info(f"  Direction distribution:")
        logger.info(f"    UP (1): {df['target_direction'].sum()} ({df['target_direction'].mean():.1%})")
        logger.info(f"    DOWN (0): {len(df) - df['target_direction'].sum()} ({1 - df['target_direction'].mean():.1%})")

        return df

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Prepare feature matrix, excluding target and metadata columns.
        """
        # Exclude columns that shouldn't be features
        exclude_cols = [
            "vix", "spx", "calendar_cohort", "cohort_weight", "feature_quality",
            "future_vix", "target_vix_pct_change", "target_log_vix_change",
            "target_direction"
        ]

        # Include cohort binary features
        cohort_features = ["is_fomc_period", "is_opex_week", "is_earnings_heavy"]

        # Get all feature columns
        all_cols = df.columns.tolist()
        feature_cols = [c for c in all_cols if c not in exclude_cols]

        # Ensure cohort features are included
        for cf in cohort_features:
            if cf not in feature_cols and cf in df.columns:
                feature_cols.append(cf)

        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))

        # Ensure cohort features exist in dataframe
        for cf in cohort_features:
            if cf not in df.columns:
                logger.warning(f"  Missing cohort feature: {cf}, setting to 0")
                df[cf] = 0

        X = df[feature_cols].copy()

        logger.info(f"  Total features: {len(feature_cols)}")
        logger.info(f"  Cohort features: {[cf for cf in cohort_features if cf in feature_cols]}")

        return X, feature_cols

    def _train_direction_model(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[XGBClassifier, Dict]:
        """
        Train binary direction classifier.
        """
        params = XGBOOST_CONFIG["shared_params"].copy()
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        })

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "train": {
                "accuracy": float(accuracy_score(y_train, y_train_pred)),
                "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
                "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
                "logloss": float(log_loss(y_train, y_train_prob))
            },
            "test": {
                "accuracy": float(accuracy_score(y_test, y_test_pred)),
                "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
                "logloss": float(log_loss(y_test, y_test_prob))
            }
        }

        logger.info(f"  Train Accuracy: {metrics['train']['accuracy']:.3f}")
        logger.info(f"  Test Accuracy: {metrics['test']['accuracy']:.3f}")
        logger.info(f"  Test Precision: {metrics['test']['precision']:.3f}")
        logger.info(f"  Test Recall: {metrics['test']['recall']:.3f}")

        return model, metrics

    def _train_magnitude_model(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[XGBRegressor, Dict]:
        """
        Train continuous magnitude regressor (log-space).
        """
        params = XGBOOST_CONFIG["shared_params"].copy()
        params.update({
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        })

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Convert to percentage for reporting
        train_pct_actual = (np.exp(y_train) - 1) * 100
        train_pct_pred = (np.exp(y_train_pred) - 1) * 100

        test_pct_actual = (np.exp(y_test) - 1) * 100
        test_pct_pred = (np.exp(y_test_pred) - 1) * 100

        # Metrics
        metrics = {
            "train": {
                "mae_log": float(mean_absolute_error(y_train, y_train_pred)),
                "rmse_log": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                "mae_pct": float(mean_absolute_error(train_pct_actual, train_pct_pred)),
                "bias_pct": float(np.mean(train_pct_pred - train_pct_actual))
            },
            "test": {
                "mae_log": float(mean_absolute_error(y_test, y_test_pred)),
                "rmse_log": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                "mae_pct": float(mean_absolute_error(test_pct_actual, test_pct_pred)),
                "bias_pct": float(np.mean(test_pct_pred - test_pct_actual))
            }
        }

        logger.info(f"  Train MAE (log): {metrics['train']['mae_log']:.4f}")
        logger.info(f"  Test MAE (log): {metrics['test']['mae_log']:.4f}")
        logger.info(f"  Test MAE (%): {metrics['test']['mae_pct']:.2f}%")
        logger.info(f"  Test Bias (%): {metrics['test']['bias_pct']:+.2f}%")

        return model, metrics

    def predict(self, X: pd.DataFrame, current_vix: float) -> Dict:
        """
        Generate prediction for a single observation.

        Args:
            X: Feature DataFrame (single row)
            current_vix: Current VIX level

        Returns:
            Dictionary with direction probabilities and magnitude forecast
        """
        # Ensure features are in correct order
        X_features = X[self.feature_names]

        # Direction prediction
        prob_up = float(self.direction_model.predict_proba(X_features)[0, 1])
        prob_down = 1.0 - prob_up

        # Magnitude prediction (log-space)
        magnitude_log = float(self.magnitude_model.predict(X_features)[0])

        # Convert to percentage
        magnitude_pct = (np.exp(magnitude_log) - 1) * 100

        # Clip to reasonable bounds
        magnitude_pct = np.clip(magnitude_pct, -50, 100)

        # Expected VIX level
        expected_vix = current_vix * (1 + magnitude_pct / 100)

        return {
            "prob_up": prob_up,
            "prob_down": prob_down,
            "magnitude_pct": float(magnitude_pct),
            "magnitude_log": float(magnitude_log),
            "expected_vix": float(expected_vix),
            "current_vix": float(current_vix)
        }

    def _save_models(self, save_dir: str):
        """
        Save both models and metadata.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save direction model
        direction_file = save_path / "direction_5d_model.pkl"
        with open(direction_file, "wb") as f:
            pickle.dump(self.direction_model, f)
        logger.info(f"\n  ✅ Saved direction model: {direction_file}")

        # Save magnitude model
        magnitude_file = save_path / "magnitude_5d_model.pkl"
        with open(magnitude_file, "wb") as f:
            pickle.dump(self.magnitude_model, f)
        logger.info(f"  ✅ Saved magnitude model: {magnitude_file}")

        # Save feature names
        features_file = save_path / "feature_names.json"
        with open(features_file, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"  ✅ Saved feature names: {features_file}")

        # Save metrics
        metrics_file = save_path / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"  ✅ Saved metrics: {metrics_file}")

    def load(self, models_dir: str = "models"):
        """
        Load trained models.
        """
        models_path = Path(models_dir)

        # Load direction model
        direction_file = models_path / "direction_5d_model.pkl"
        with open(direction_file, "rb") as f:
            self.direction_model = pickle.load(f)

        # Load magnitude model
        magnitude_file = models_path / "magnitude_5d_model.pkl"
        with open(magnitude_file, "rb") as f:
            self.magnitude_model = pickle.load(f)

        # Load feature names
        features_file = models_path / "feature_names.json"
        with open(features_file, "r") as f:
            self.feature_names = json.load(f)

        logger.info(f"✅ Loaded models from: {models_dir}")
        logger.info(f"  Features: {len(self.feature_names)}")

    def _generate_diagnostics(
        self, X_test: pd.DataFrame, y_direction_test: pd.Series,
        y_magnitude_test: pd.Series, save_dir: str
    ):
        """
        Generate diagnostic plots.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Model Performance Diagnostics", fontsize=16, fontweight="bold")

            # Direction calibration
            ax = axes[0, 0]
            y_prob = self.direction_model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_direction_test, y_prob, n_bins=10)
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.plot(prob_pred, prob_true, "o-", label="Model", linewidth=2)
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Frequency")
            ax.set_title("Direction Probability Calibration")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Direction accuracy by bin
            ax = axes[0, 1]
            bins = [0, 0.4, 0.6, 1.0]
            labels = ["<40%", "40-60%", ">60%"]
            X_test_copy = X_test.copy()
            X_test_copy["prob"] = y_prob
            X_test_copy["actual"] = y_direction_test.values
            X_test_copy["prob_bin"] = pd.cut(X_test_copy["prob"], bins=bins, labels=labels)

            accs = []
            for label in labels:
                mask = X_test_copy["prob_bin"] == label
                if mask.sum() > 0:
                    acc = (X_test_copy.loc[mask, "actual"] == 1).mean()
                    accs.append(acc)
                else:
                    accs.append(0)

            ax.bar(range(len(labels)), accs, alpha=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Actual UP Frequency")
            ax.set_title("Direction Accuracy by Confidence")
            ax.grid(True, alpha=0.3, axis="y")

            # Magnitude predictions
            ax = axes[1, 0]
            y_mag_pred = self.magnitude_model.predict(X_test)
            test_pct_actual = (np.exp(y_magnitude_test) - 1) * 100
            test_pct_pred = (np.exp(y_mag_pred) - 1) * 100

            ax.scatter(test_pct_pred, test_pct_actual, alpha=0.5, s=30)
            lims = [
                min(test_pct_pred.min(), test_pct_actual.min()),
                max(test_pct_pred.max(), test_pct_actual.max())
            ]
            ax.plot(lims, lims, "k--", alpha=0.5)
            ax.set_xlabel("Predicted VIX Change (%)")
            ax.set_ylabel("Actual VIX Change (%)")
            ax.set_title("Magnitude Forecast Accuracy")
            ax.grid(True, alpha=0.3)

            # Magnitude error distribution
            ax = axes[1, 1]
            errors = test_pct_pred - test_pct_actual
            ax.hist(errors, bins=30, alpha=0.7, edgecolor="black")
            ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
            ax.axvline(errors.mean(), color="blue", linestyle="--", linewidth=2,
                      label=f"Mean: {errors.mean():.2f}%")
            ax.set_xlabel("Prediction Error (%)")
            ax.set_ylabel("Frequency")
            ax.set_title("Magnitude Error Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = save_path / "model_diagnostics.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"  ✅ Saved diagnostics: {plot_file}")

        except Exception as e:
            logger.warning(f"  Could not generate plots: {e}")

    def _print_summary(self):
        """
        Print training summary.
        """
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"\nModels Trained: 2")
        print(f"  1. Direction Classifier")
        print(f"  2. Magnitude Regressor")
        print(f"\nFeatures: {len(self.feature_names)}")
        print(f"  Including cohort flags: is_fomc_period, is_opex_week, is_earnings_heavy")

        print(f"\nDirection Performance:")
        print(f"  Train Accuracy: {self.metrics['direction']['train']['accuracy']:.1%}")
        print(f"  Test Accuracy: {self.metrics['direction']['test']['accuracy']:.1%}")
        print(f"  Test Precision: {self.metrics['direction']['test']['precision']:.1%}")
        print(f"  Test Recall: {self.metrics['direction']['test']['recall']:.1%}")

        print(f"\nMagnitude Performance:")
        print(f"  Test MAE: {self.metrics['magnitude']['test']['mae_pct']:.2f}%")
        print(f"  Test Bias: {self.metrics['magnitude']['test']['bias_pct']:+.2f}%")
        print(f"  Test RMSE (log): {self.metrics['magnitude']['test']['rmse_log']:.4f}")

        print("=" * 80)


def train_simplified_forecaster(df: pd.DataFrame, save_dir: str = "models") -> SimplifiedVIXForecaster:
    """
    Convenience function to train the simplified forecaster.
    """
    forecaster = SimplifiedVIXForecaster()
    forecaster.train(df, save_dir=save_dir)
    return forecaster
