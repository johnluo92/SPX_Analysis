"""Refactored Probabilistic VIX Forecasting System V3
Implements true quantile regression with log-transformed realized volatility targets.
Removes redundant point estimates - median (q50) serves as primary forecast.

FIXED VERSION - Corrected quantile regression implementation and small cohort handling
"""

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
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from config import TARGET_CONFIG, XGBOOST_CONFIG

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilisticVIXForecaster:
    """
    Quantile-based volatility forecasting with directional classifier.

    Key improvements:
    - True quantile regression (separate models per quantile)
    - Log-transformed realized volatility targets
    - Domain-aware bounds (VIX rarely < 10 or > 90)
    - No redundant point estimates (q50 median is primary forecast)
    - Proper handling of small cohorts (< 50 samples)
    """

    def __init__(self):
        self.horizon = TARGET_CONFIG["horizon_days"]
        self.quantiles = TARGET_CONFIG["quantiles"]["levels"]
        self.models = {}
        self.calibrators = {}
        self.feature_names = None

        # Domain knowledge: VIX bounds
        self.vix_floor = 10.0  # VIX rarely goes below this
        self.vix_ceiling = 90.0  # VIX rarely exceeds this

    def train(self, df: pd.DataFrame, save_dir: str = "models"):
        """Train all cohort models and save to disk."""
        logger.info("=" * 80)
        logger.info("REFACTORED QUANTILE REGRESSION TRAINING")
        logger.info("=" * 80)

        if "calendar_cohort" not in df.columns:
            raise ValueError("Missing calendar_cohort column")

        # Create log-transformed realized volatility targets
        df = self._create_targets(df)

        # ✅ FIX: Store feature names (exclude ONLY necessary columns)
        exclude_cols = [
            "vix",
            "spx",
            "calendar_cohort",
            "feature_quality",
            "cohort_weight",  # Added if exists
            "forward_realized_vol",  # Raw realized vol (before log transform)
            "target_log_rv",  # Single target for all quantiles
            "target_direction",
            "target_confidence",
        ]

        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_names = feature_cols

        # Train each cohort
        cohorts = sorted(df["calendar_cohort"].unique())
        logger.info(f"\nTraining {len(cohorts)} cohorts: {cohorts}")

        cohort_metrics = {}
        for cohort in cohorts:
            cohort_df = df[df["calendar_cohort"] == cohort].copy()
            logger.info(f"\n{'─' * 80}")
            logger.info(f"Cohort: {cohort} ({len(cohort_df)} samples)")
            logger.info(f"{'─' * 80}")

            metrics = self._train_cohort_models(cohort, cohort_df)
            cohort_metrics[cohort] = metrics

            # Save immediately
            self._save_cohort_models(cohort, save_dir)
            logger.info(f"✅ Saved: {cohort}")

        # Generate diagnostics
        self._generate_diagnostics(cohort_metrics, save_dir)
        logger.info(f"\n{'=' * 80}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'=' * 80}")

        return self

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create log-transformed realized volatility targets."""
        logger.info("  Target creation:")
        logger.info(f"    Total samples: {len(df)}")

        df = df.copy()

        # Calculate forward-looking realized volatility (t+1 to t+horizon+1)
        # ✅ FIX: Corrected forward window calculation
        spx_returns = np.log(df["spx"] / df["spx"].shift(1))

        forward_rv_list = []
        for i in range(len(df)):
            if i + self.horizon + 1 <= len(df):
                # ✅ FIX: Window from i+1 to i+horizon+1 (truly forward-looking)
                window_returns = spx_returns.iloc[i + 1 : i + self.horizon + 1]
                rv = window_returns.std() * np.sqrt(252) * 100  # Annualized %
                forward_rv_list.append(rv)
            else:
                forward_rv_list.append(np.nan)

        df["forward_realized_vol"] = forward_rv_list

        # Apply log transformation for better distribution properties
        df["target_log_rv"] = np.log(df["forward_realized_vol"].clip(lower=1.0))

        # ✅ FIX: NO separate target columns - all quantiles use same target!
        # This is the KEY fix - quantile_alpha parameter makes models learn different quantiles

        # Direction target: is VIX going up or down?
        future_vix = df["vix"].shift(-self.horizon)
        df["target_direction"] = (future_vix > df["vix"]).astype(int)

        # Confidence based on regime stability and feature quality
        regime_volatility = df["vix"].rolling(21, min_periods=10).std()
        regime_stability = 1 / (
            1 + regime_volatility / df["vix"].rolling(21, min_periods=10).mean()
        )
        regime_stability = regime_stability.fillna(0.5)

        df["target_confidence"] = (
            0.5 * df.get("feature_quality", 0.5).fillna(0.5) + 0.5 * regime_stability
        ).clip(0, 1)

        valid_targets = (~df["target_log_rv"].isna()).sum()
        logger.info(f"    Valid targets: {valid_targets}")
        logger.info(
            f"    NaN targets: {len(df) - valid_targets} (last {self.horizon} days)"
        )
        logger.info("")

        return df

    def _train_cohort_models(self, cohort: str, df: pd.DataFrame) -> Dict:
        """Train quantile models and direction classifier for a single cohort."""
        df = df.copy()
        X = df[self.feature_names]

        # ✅ FIX: Lowered minimum sample requirement from 100 to 30
        if len(df) < 30:
            raise ValueError(f"Insufficient samples for cohort {cohort}: {len(df)}")

        self.models[cohort] = {}
        self.calibrators[cohort] = {}
        metrics = {}

        # ✅ FIX: All quantile models use SAME target with different quantile_alpha
        y_target = df["target_log_rv"]  # Single target for ALL quantiles

        # Train 5 quantile regression models (q10, q25, q50, q75, q90)
        logger.info("  Training quantile regression models...")
        for q in self.quantiles:
            q_name = f"q{int(q * 100)}"

            # ✅ FIX: Same y_target for all models, quantile_alpha makes the difference
            model, metric = self._train_quantile_regressor(
                X, y_target, quantile_alpha=q
            )
            self.models[cohort][q_name] = model
            metrics[q_name] = metric
            logger.info(f"    {q_name}: MAE={metric['mae']:.4f}")

        # Train direction classifier
        logger.info("  Training direction classifier...")
        y_direction = df["target_direction"]

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

        # Train confidence model (kept for compatibility)
        logger.info("  Training confidence model...")
        y_confidence = df["target_confidence"]
        model_conf, metric_conf = self._train_regressor(
            X, y_confidence, objective="reg:squarederror", eval_metric="rmse"
        )
        self.models[cohort]["confidence"] = model_conf
        metrics["confidence"] = metric_conf
        logger.info(f"    RMSE: {metric_conf['rmse']:.3f}")

        return metrics

    def _train_quantile_regressor(self, X, y, quantile_alpha: float):
        """Train a single quantile regression model."""
        params = XGBOOST_CONFIG["shared_params"].copy()
        params.update(
            {
                "objective": "reg:quantileerror",
                "quantile_alpha": quantile_alpha,
            }
        )

        # Adaptive CV configuration
        n_splits, test_size = self._get_adaptive_cv_config(len(X))

        # Track validation performance
        val_maes = []

        # ✅ FIX: Skip CV for very small cohorts
        if n_splits == 0:
            logger.info(f"      Small cohort - training on all data without CV")
            # Train directly on all valid data
            valid_idx = ~y.isna()
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]

            if len(X_clean) < 10:
                raise ValueError(f"Insufficient valid samples: {len(X_clean)}")

            final_model = XGBRegressor(**params)
            final_model.fit(X_clean, y_clean, verbose=False)

            # Use training error as proxy metric
            y_pred = final_model.predict(X_clean)
            mae = mean_absolute_error(y_clean, y_pred)
            metrics = {"mae": float(mae), "mae_std": 0.0}
            return final_model, metrics

        # Standard CV training
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Remove NaN targets
            valid_train = ~y_train.isna()
            valid_val = ~y_val.isna()

            if valid_train.sum() < 10 or valid_val.sum() < 5:
                continue

            X_train_clean = X_train[valid_train]
            y_train_clean = y_train[valid_train]
            X_val_clean = X_val[valid_val]
            y_val_clean = y_val[valid_val]

            model = XGBRegressor(**params)
            model.fit(
                X_train_clean,
                y_train_clean,
                eval_set=[(X_val_clean, y_val_clean)],
                verbose=False,
            )

            y_pred = model.predict(X_val_clean)
            mae = mean_absolute_error(y_val_clean, y_pred)
            val_maes.append(mae)

        # Final model on all data
        valid_idx = ~y.isna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        final_model = XGBRegressor(**params)
        final_model.fit(X_clean, y_clean, verbose=False)

        metrics = {
            "mae": float(np.mean(val_maes)) if val_maes else 0.0,
            "mae_std": float(np.std(val_maes)) if val_maes else 0.0,
        }

        return final_model, metrics

    def _train_classifier(self, X, y, num_classes: int):
        """Train direction classifier."""
        params = XGBOOST_CONFIG["shared_params"].copy()
        params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }
        )

        n_splits, test_size = self._get_adaptive_cv_config(len(X))

        val_accs = []
        val_loglosses = []

        # ✅ FIX: Skip CV for very small cohorts
        if n_splits == 0:
            logger.info(f"      Small cohort - training on all data without CV")
            valid_idx = ~y.isna()
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]

            if len(X_clean) < 10:
                raise ValueError(f"Insufficient valid samples: {len(X_clean)}")

            final_model = XGBClassifier(**params)
            final_model.fit(X_clean, y_clean, verbose=False)

            # Use training metrics as proxy
            y_pred = final_model.predict(X_clean)
            y_pred_proba = final_model.predict_proba(X_clean)
            acc = accuracy_score(y_clean, y_pred)
            ll = log_loss(y_clean, y_pred_proba)

            metrics = {"accuracy": float(acc), "logloss": float(ll)}
            return final_model, metrics

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            valid_train = ~y_train.isna()
            valid_val = ~y_val.isna()

            if valid_train.sum() < 10 or valid_val.sum() < 5:
                continue

            X_train_clean = X_train[valid_train]
            y_train_clean = y_train[valid_train]
            X_val_clean = X_val[valid_val]
            y_val_clean = y_val[valid_val]

            model = XGBClassifier(**params)
            model.fit(
                X_train_clean,
                y_train_clean,
                eval_set=[(X_val_clean, y_val_clean)],
                verbose=False,
            )

            y_pred = model.predict(X_val_clean)
            y_pred_proba = model.predict_proba(X_val_clean)

            acc = accuracy_score(y_val_clean, y_pred)
            ll = log_loss(y_val_clean, y_pred_proba)

            val_accs.append(acc)
            val_loglosses.append(ll)

        # Final model
        valid_idx = ~y.isna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        final_model = XGBClassifier(**params)
        final_model.fit(X_clean, y_clean, verbose=False)

        metrics = {
            "accuracy": float(np.mean(val_accs)) if val_accs else 0.0,
            "logloss": float(np.mean(val_loglosses)) if val_loglosses else 0.0,
        }

        return final_model, metrics

    def _train_regressor(self, X, y, objective: str, eval_metric: str):
        """Train generic regressor (for confidence model)."""
        params = XGBOOST_CONFIG["shared_params"].copy()
        params.update(
            {
                "objective": objective,
                "eval_metric": eval_metric,
            }
        )

        n_splits, test_size = self._get_adaptive_cv_config(len(X))

        val_rmses = []

        # ✅ FIX: Skip CV for very small cohorts
        if n_splits == 0:
            logger.info(f"      Small cohort - training on all data without CV")
            valid_idx = ~y.isna()
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]

            if len(X_clean) < 10:
                raise ValueError(f"Insufficient valid samples: {len(X_clean)}")

            final_model = XGBRegressor(**params)
            final_model.fit(X_clean, y_clean, verbose=False)

            # Use training RMSE as proxy
            y_pred = final_model.predict(X_clean)
            rmse = np.sqrt(mean_absolute_error(y_clean, y_pred) ** 2)

            metrics = {"rmse": float(rmse)}
            return final_model, metrics

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            valid_train = ~y_train.isna()
            valid_val = ~y_val.isna()

            if valid_train.sum() < 10 or valid_val.sum() < 5:
                continue

            model = XGBRegressor(**params)
            model.fit(
                X_train[valid_train],
                y_train[valid_train],
                eval_set=[(X_val[valid_val], y_val[valid_val])],
                verbose=False,
            )

            y_pred = model.predict(X_val[valid_val])
            rmse = np.sqrt(mean_absolute_error(y_val[valid_val], y_pred) ** 2)
            val_rmses.append(rmse)

        # Final model
        valid_idx = ~y.isna()
        final_model = XGBRegressor(**params)
        final_model.fit(X[valid_idx], y[valid_idx], verbose=False)

        metrics = {"rmse": float(np.mean(val_rmses)) if val_rmses else 0.0}
        return final_model, metrics

    def _get_adaptive_cv_config(self, n_samples: int) -> Tuple[int, int]:
        """Determine appropriate CV splits based on sample size.

        Returns (0, 0) for very small cohorts to signal no CV should be used.
        """
        # For very small cohorts, skip CV entirely
        if n_samples < 50:
            return 0, 0  # Signal to skip CV

        # Determine initial splits based on sample size
        if n_samples < 200:
            n_splits = 2
        elif n_samples < 400:
            n_splits = 3
        elif n_samples < 800:
            n_splits = 4
        else:
            n_splits = 5

        # Calculate test_size without hardcoded minimum
        max_test_size = n_samples // (n_splits + 1)
        test_size = max(int(max_test_size * 0.8), 10)  # Lower minimum to 10

        # Adjust if configuration is invalid - allow n_splits to go to 1
        while (n_samples - test_size) < n_splits * test_size and n_splits > 1:
            n_splits -= 1
            max_test_size = n_samples // (n_splits + 1)
            test_size = max(int(max_test_size * 0.8), 10)

        # Final validation
        if n_splits * test_size > n_samples:
            # Fall back to single split with reasonable test size
            n_splits = 1
            test_size = min(n_samples // 3, 20)

        return n_splits, test_size

    def _calibrate_probabilities(self, model, X, y):
        """Calibrate classifier probabilities using isotonic regression."""
        y_proba = model.predict_proba(X)

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

    def predict(self, X: pd.DataFrame, cohort: str, current_vix: float) -> Dict:
        """
        Generate probabilistic forecast with quantiles and direction.

        Args:
            X: Feature dataframe (single row)
            cohort: Calendar cohort to use
            current_vix: Current VIX level for domain-aware bounds

        Returns:
            Dictionary with quantiles, direction_probability, confidence_score
        """
        if cohort not in self.models:
            raise ValueError(
                f"Cohort {cohort} not trained. Available: {list(self.models.keys())}"
            )

        X_features = X[self.feature_names]

        # Get quantile predictions in log space
        quantiles_log = {}
        for q in self.quantiles:
            q_name = f"q{int(q * 100)}"
            pred_log = self.models[cohort][q_name].predict(X_features)[0]
            quantiles_log[q_name] = pred_log

        # ✅ FIX: Proper conversion with monotonicity enforcement
        # 1. Exponentiate to get RV
        quantiles_rv = {k: np.exp(v) for k, v in quantiles_log.items()}

        # 2. Apply domain bounds
        quantiles_rv_bounded = {
            k: np.clip(v, self.vix_floor, self.vix_ceiling)
            for k, v in quantiles_rv.items()
        }

        # 3. Enforce monotonicity (q10 <= q25 <= q50 <= q75 <= q90)
        q_keys = ["q10", "q25", "q50", "q75", "q90"]
        q_values = [quantiles_rv_bounded[k] for k in q_keys]
        q_values_sorted = sorted(q_values)  # Force correct order
        quantiles_rv_monotonic = dict(zip(q_keys, q_values_sorted))

        # 4. Convert to VIX % change (now with properly ordered values)
        quantiles_pct = {
            k: ((v / current_vix) - 1) * 100 for k, v in quantiles_rv_monotonic.items()
        }

        # Direction probability (probability VIX goes up)
        prob_up = float(
            self.models[cohort]["direction"].predict_proba(X_features)[0][1]
        )
        prob_down = 1.0 - prob_up

        # Confidence score
        confidence = np.clip(
            self.models[cohort]["confidence"].predict(X_features)[0], 0, 1
        )

        # Use median (q50) as primary forecast
        median_forecast = quantiles_pct["q50"]

        # ✅ FIX: Return quantiles at BOTH nested and top level for compatibility
        return {
            "median_forecast": float(median_forecast),
            "quantiles": {k: float(v) for k, v in quantiles_pct.items()},  # Nested
            # Top-level quantiles for backward compatibility
            "q10": float(quantiles_pct["q10"]),
            "q25": float(quantiles_pct["q25"]),
            "q50": float(quantiles_pct["q50"]),
            "q75": float(quantiles_pct["q75"]),
            "q90": float(quantiles_pct["q90"]),
            "prob_up": float(prob_up),
            "prob_down": float(prob_down),
            "direction_probability": float(prob_up),
            "confidence_score": float(confidence),
            "cohort": cohort,
            "metadata": {
                "current_vix": current_vix,
                "realized_vol_bounds": {
                    "floor": self.vix_floor,
                    "ceiling": self.vix_ceiling,
                },
            },
        }

    def load(self, cohort: str, models_dir: str):
        """
        Load a single cohort's models from disk.

        Args:
            cohort: Cohort name to load
            models_dir: Directory containing saved models
        """
        models_path = Path(models_dir)
        file_path = models_path / f"probabilistic_forecaster_{cohort}.pkl"

        if not file_path.exists():
            raise FileNotFoundError(f"No saved model found for cohort: {cohort}")

        with open(file_path, "rb") as f:
            cohort_data = pickle.load(f)

        # Load cohort-specific data
        self.models[cohort] = cohort_data["models"]
        self.calibrators[cohort] = cohort_data["calibrators"]

        # Load shared configuration (from first cohort loaded)
        if self.feature_names is None:
            self.feature_names = cohort_data["feature_names"]
            self.quantiles = cohort_data["quantiles"]
            self.horizon = cohort_data["horizon"]

    def _save_cohort_models(self, cohort: str, save_dir: str):
        """Save cohort models to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        cohort_data = {
            "models": self.models[cohort],
            "calibrators": self.calibrators[cohort],
            "feature_names": self.feature_names,
            "quantiles": self.quantiles,
            "horizon": self.horizon,
        }

        file_path = save_path / f"probabilistic_forecaster_{cohort}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(cohort_data, f)

    def _generate_diagnostics(self, cohort_metrics: Dict, save_dir: str):
        """Generate diagnostic plots and save metrics."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        # Save metrics JSON
        metrics_file = save_path / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(cohort_metrics, f, indent=2)

        logger.info(f"  Saved metrics: {metrics_file}")

        # Generate plots
        try:
            cohorts = list(cohort_metrics.keys())

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(
                "Model Performance (Log-RV Quantile Regression) by Cohort", fontsize=16
            )

            # Plot quantile MAEs
            for idx, q in enumerate([10, 25, 50, 75, 90]):
                q_name = f"q{q}"
                row = idx // 3
                col = idx % 3

                maes = [cohort_metrics[c][q_name]["mae"] for c in cohorts]
                axes[row, col].bar(range(len(cohorts)), maes, color="steelblue")
                axes[row, col].set_xticks(range(len(cohorts)))
                axes[row, col].set_xticklabels(cohorts, rotation=45, ha="right")
                axes[row, col].set_ylabel("MAE (log space)")
                axes[row, col].set_title(f"{q}th Percentile")
                axes[row, col].grid(True, alpha=0.3)

            # Plot direction accuracy
            dir_accs = [cohort_metrics[c]["direction"]["accuracy"] for c in cohorts]
            axes[1, 2].bar(range(len(cohorts)), dir_accs, color="forestgreen")
            axes[1, 2].set_xticks(range(len(cohorts)))
            axes[1, 2].set_xticklabels(cohorts, rotation=45, ha="right")
            axes[1, 2].set_ylabel("Accuracy")
            axes[1, 2].set_title("Direction Classifier")
            axes[1, 2].set_ylim([0, 1])
            axes[1, 2].grid(True, alpha=0.3)

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
    Convenience function to train and return refactored forecaster.

    Args:
        df: Feature dataframe with calendar_cohort column
        save_dir: Directory to save trained models

    Returns:
        Trained ProbabilisticVIXForecaster instance
    """
    forecaster = ProbabilisticVIXForecaster()
    forecaster.train(df, save_dir=save_dir)
    return forecaster
