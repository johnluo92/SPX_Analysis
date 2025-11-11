"""
Probabilistic VIX Forecasting System
Trains multi-output XGBoost models for distribution forecasting
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from config import CALENDAR_COHORTS, TARGET_CONFIG, XGBOOST_CONFIG

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilisticVIXForecaster:
    """
    Multi-output forecaster producing full VIX distribution.

    Trains 8 models per calendar cohort:
      - 1 point estimate (mean VIX % change)
      - 5 quantiles (10th, 25th, 50th, 75th, 90th percentiles)
      - 1 regime classifier (4 classes: Low/Normal/Elevated/Crisis)
      - 1 confidence scorer (forecast quality)
    """

    def __init__(self):
        self.horizon = TARGET_CONFIG["horizon_days"]
        self.quantiles = TARGET_CONFIG["quantiles"]["levels"]
        self.regime_boundaries = TARGET_CONFIG["regimes"]["boundaries"]
        self.regime_labels = TARGET_CONFIG["regimes"]["labels"]

        self.models = {}  # {cohort: {model_type: model}}
        self.calibrators = {}  # {cohort: {model_type: IsotonicRegression}}
        self.feature_names = None

        logger.info("🎯 Probabilistic VIX Forecaster initialized")
        logger.info(f"   Horizon: {self.horizon} days")
        logger.info(f"   Quantiles: {self.quantiles}")
        logger.info(f"   Regimes: {self.regime_labels}")

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all target variables from raw VIX data.
        No data leakage - all targets use future VIX values only.
        """
        df = df.copy()

        # 1. Point Estimate: Future VIX % change (no leakage)
        future_vix = df["vix"].shift(-self.horizon)
        df["target_point"] = ((future_vix / df["vix"]) - 1) * 100

        # Clip extremes
        point_min, point_max = TARGET_CONFIG["point_estimate"]["range"]
        df["target_point"] = df["target_point"].clip(point_min, point_max)

        logger.info(
            f"   Point target range: {df['target_point'].min():.1f}% to {df['target_point'].max():.1f}%"
        )

        # 2. Quantiles: Use same target_point for all quantile models
        # XGBoost will learn different quantiles via pinball loss (no leakage)
        for q in self.quantiles:
            col_name = f"target_q{int(q * 100)}"
            df[col_name] = df["target_point"]  # Same target, different loss function

        logger.info(f"   Quantile targets created (XGBoost learns via pinball loss)")

        # 3. Regime: Classify future VIX level (no leakage)
        regime_bins = [-np.inf] + self.regime_boundaries + [np.inf]
        df["target_regime"] = pd.cut(
            future_vix,
            bins=regime_bins,
            labels=list(range(len(self.regime_labels))),
            include_lowest=True,
        )
        df["target_regime"] = df["target_regime"].astype(float)

        regime_counts = df["target_regime"].value_counts().sort_index()
        logger.info("   Regime distribution:")
        for regime_id, label in enumerate(self.regime_labels):
            count = regime_counts.get(regime_id, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            logger.info(
                f"      {label:10s} (class {regime_id}): {count:4d} ({pct:5.1f}%)"
            )

        # 4. Confidence: Combine feature quality + regime stability
        regime_volatility = df["vix"].rolling(21, min_periods=10).std()
        regime_stability = 1 / (
            1 + regime_volatility / df["vix"].rolling(21, min_periods=10).mean()
        )
        regime_stability = regime_stability.fillna(0.5)

        # Combine: 50% feature quality + 50% regime stability
        df["target_confidence"] = (
            0.5 * df["feature_quality"].fillna(0.5) + 0.5 * regime_stability
        ).clip(0, 1)

        logger.info(
            f"   Confidence labels: mean={df['target_confidence'].mean():.2f}, "
            f"std={df['target_confidence'].std():.2f}"
        )

        return df

    def train(self, df: pd.DataFrame, save_dir: str = "models") -> Dict:
        """
        Train separate model sets for each calendar cohort.
        """
        logger.info("=" * 80)
        logger.info("PROBABILISTIC VIX FORECASTER - TRAINING")
        logger.info("=" * 80)

        # Validate required columns
        required = ["vix", "calendar_cohort", "cohort_weight", "feature_quality"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create targets
        df = self._create_targets(df)

        # Remove rows where targets are NaN (edge effects from shifts)
        target_cols = ["target_point", "target_regime", "target_confidence"]
        df_clean = df.dropna(subset=target_cols)
        logger.info(
            f"Training samples: {len(df_clean)} (dropped {len(df) - len(df_clean)} edge rows)"
        )

        # Extract feature names (exclude metadata and targets)
        exclude_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        exclude_cols += [col for col in df_clean.columns if col.startswith("target_")]
        self.feature_names = [
            col for col in df_clean.columns if col not in exclude_cols
        ]

        logger.info(f"Features used: {len(self.feature_names)}")

        # Train per cohort
        cohort_metrics = {}
        cohorts = df_clean["calendar_cohort"].unique()

        for cohort in cohorts:
            logger.info(f"\n{'─' * 80}")
            logger.info(f"TRAINING COHORT: {cohort}")
            logger.info(f"{'─' * 80}")

            cohort_df = df_clean[df_clean["calendar_cohort"] == cohort].copy()
            logger.info(f"Cohort samples: {len(cohort_df)}")

            if len(cohort_df) < 200:
                logger.warning(
                    f"⚠️  Too few samples ({len(cohort_df)}) for {cohort}, skipping"
                )
                continue

            # Train all model types for this cohort
            metrics = self._train_cohort_models(cohort, cohort_df)
            cohort_metrics[cohort] = metrics

            # Save models immediately after training each cohort
            self._save_cohort_models(cohort, save_dir)

        # Generate diagnostics
        self._generate_diagnostics(cohort_metrics, save_dir)

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)

        return cohort_metrics

    def _train_cohort_models(self, cohort: str, df: pd.DataFrame) -> Dict:
        """Train all 8 models for a single cohort."""
        X = df[self.feature_names]

        # Check minimum sample size
        min_samples = 100  # Absolute minimum
        if len(df) < min_samples:
            logger.warning(
                f"⚠️  Skipping {cohort}: only {len(df)} samples (need >{min_samples})"
            )
            raise ValueError(f"Insufficient samples for cohort {cohort}")

        # Initialize model dictionary for this cohort
        self.models[cohort] = {}
        self.calibrators[cohort] = {}

        metrics = {}

        # 1. Point Estimate
        logger.info("\n[1/4] Training point estimate model...")
        y_point = df["target_point"]
        model_point, metric_point = self._train_regressor(
            X, y_point, objective="reg:squarederror", eval_metric="rmse"
        )
        self.models[cohort]["point"] = model_point
        metrics["point"] = metric_point
        logger.info(f"   ✅ Point RMSE: {metric_point['rmse']:.2f}%")

        # 2. Quantiles (5 models)
        logger.info("\n[2/4] Training quantile models...")
        self.models[cohort]["quantiles"] = {}
        metrics["quantiles"] = {}

        for q in self.quantiles:
            q_label = f"q{int(q * 100)}"
            y_quantile = df[f"target_{q_label}"]

            model_q, metric_q = self._train_regressor(
                X,
                y_quantile,
                objective="reg:quantileerror",
                quantile_alpha=q,
                eval_metric="mae",
            )
            self.models[cohort]["quantiles"][q] = model_q
            metrics["quantiles"][q_label] = metric_q
            logger.info(f"   ✅ {q_label:3s} MAE: {metric_q['mae']:.2f}%")

        # 3. Regime Classifier
        # if training is to be skipped because of too few cohorts
        # if len(df) < 500:
        #     logger.warning(f"\n[3/4] Skipping regime classifier (insufficient samples)")
        #     # Use a dummy classifier that always predicts proportional to training distribution
        #     self.models[cohort]["regime"] = None
        #     metrics["regime"] = {"accuracy": 0.0, "log_loss": 999.0, "skipped": True}
        # else:
        #     logger.info("\n[3/4] Training regime classifier...")

        logger.info("\n[3/4] Training regime classifier...")
        y_regime = df["target_regime"]
        model_regime, metric_regime = self._train_classifier(
            X, y_regime, num_classes=len(self.regime_labels)
        )
        self.models[cohort]["regime"] = model_regime
        self.calibrators[cohort]["regime"] = self._calibrate_probabilities(
            model_regime, X, y_regime
        )
        metrics["regime"] = metric_regime
        logger.info(f"   ✅ Regime Accuracy: {metric_regime['accuracy']:.3f}")
        logger.info(f"   ✅ Log Loss: {metric_regime['log_loss']:.3f}")

        # 4. Confidence Scorer
        logger.info("\n[4/4] Training confidence model...")
        y_confidence = df["target_confidence"]
        model_conf, metric_conf = self._train_regressor(
            X, y_confidence, objective="reg:squarederror", eval_metric="rmse"
        )
        self.models[cohort]["confidence"] = model_conf
        metrics["confidence"] = metric_conf
        logger.info(f"   ✅ Confidence RMSE: {metric_conf['rmse']:.3f}")

        return metrics

    def _train_regressor(self, X, y, objective, eval_metric, quantile_alpha=None):
        """Train single XGBoost regressor with adaptive CV."""
        params = XGBOOST_CONFIG["shared_params"].copy()
        params["objective"] = objective

        if quantile_alpha:
            params["quantile_alpha"] = quantile_alpha

        # Adaptive CV splits based on sample size
        n_samples = len(X)
        if n_samples < 200:
            n_splits = 2
            test_size = int(n_samples * 0.25)
        elif n_samples < 500:
            n_splits = 3
            test_size = int(n_samples * 0.20)
        else:
            n_splits = 5
            test_size = int(n_samples * 0.20)

        logger.info(f"   Using {n_splits} CV splits (n={n_samples})")

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create params with early stopping for CV
            cv_params = params.copy()
            cv_params["early_stopping_rounds"] = 50

            model = XGBRegressor(**cv_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)

            if eval_metric == "rmse":
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            elif eval_metric == "mae":
                score = mean_absolute_error(y_val, y_pred)

            cv_scores.append(score)

        # Train final model on full data WITHOUT early stopping
        # (no validation set available when using all data)
        final_params = params.copy()
        # Remove early_stopping_rounds for final training
        if "early_stopping_rounds" in final_params:
            del final_params["early_stopping_rounds"]

        final_model = XGBRegressor(**final_params)
        final_model.fit(X, y, verbose=False)

        metrics = {
            eval_metric: np.mean(cv_scores),
            f"{eval_metric}_std": np.std(cv_scores),
        }

        return final_model, metrics

    def _train_classifier(self, X, y, num_classes):
        """Train XGBoost classifier with adaptive CV and regime collapsing for small samples."""

        # Check class distribution
        class_counts = pd.Series(y).value_counts().sort_index()
        logger.info(f"   Class distribution: {dict(class_counts)}")

        # For small cohorts, collapse rare regimes
        n_samples = len(X)
        if n_samples < 500:
            # Collapse Crisis (3) into Elevated (2)
            y_collapsed = y.copy()
            y_collapsed[y == 3] = 2

            # If still too few samples in class 2, collapse into Normal (1)
            if (y_collapsed == 2).sum() < 20:
                y_collapsed[y_collapsed == 2] = 1
                effective_classes = 2
                logger.warning(
                    f"   ⚠️  Collapsed to 2 classes (Low/Normal) due to sample size"
                )
            else:
                effective_classes = 3
                logger.warning(f"   ⚠️  Collapsed Crisis→Elevated (now 3 classes)")

            y = y_collapsed
            num_classes = effective_classes

        # CRITICAL FIX: Relabel classes to ensure sequential 0,1,2... with no gaps
        unique_classes = np.sort(y.unique())
        class_mapping = {old: new for new, old in enumerate(unique_classes)}
        y_relabeled = y.map(class_mapping)

        # Verify relabeling worked correctly
        relabeled_unique = np.sort(y_relabeled.unique())
        expected_classes = np.arange(len(unique_classes))
        assert np.array_equal(relabeled_unique, expected_classes), (
            f"Relabeling failed: got {relabeled_unique}, expected {expected_classes}"
        )

        # Store inverse mapping for predictions
        inverse_mapping = {new: old for old, new in class_mapping.items()}

        logger.info(f"   Relabeled classes: {class_mapping}")
        logger.info(f"   Unique relabeled values: {relabeled_unique}")

        params = XGBOOST_CONFIG["shared_params"].copy()
        params["objective"] = "multi:softprob"
        params["num_class"] = len(unique_classes)

        # Adaptive CV splits based on sample size
        if n_samples < 200:
            n_splits = 2
            test_size = int(n_samples * 0.25)
        elif n_samples < 500:
            n_splits = 3
            test_size = int(n_samples * 0.20)
        else:
            n_splits = 5
            test_size = int(n_samples * 0.20)

        logger.info(
            f"   Using {n_splits} CV splits (n={n_samples}, {len(unique_classes)} classes)"
        )

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        cv_accuracy = []
        cv_logloss = []
        valid_folds = 0
        skipped_folds = 0

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_relabeled.iloc[train_idx], y_relabeled.iloc[val_idx]

            # Check class distribution in this fold
            train_classes = np.sort(y_train.unique())
            val_classes = np.sort(y_val.unique())

            logger.debug(
                f"   Fold {fold_idx + 1}: train_classes={train_classes}, "
                f"val_classes={val_classes}"
            )

            # CRITICAL FIX: Skip folds where training set doesn't have all classes
            # XGBoost requires all classes to be present in training data
            if not np.array_equal(train_classes, expected_classes):
                logger.warning(
                    f"   ⚠️  Skipping Fold {fold_idx + 1}: train set missing classes "
                    f"(has {train_classes}, needs {expected_classes})"
                )
                skipped_folds += 1
                continue

            # Create params with early stopping for CV
            cv_params = params.copy()
            cv_params["early_stopping_rounds"] = 50

            model = XGBClassifier(**cv_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            accuracy = (y_pred == y_val).mean()

            # Specify labels parameter to handle missing classes in validation fold
            logloss = log_loss(y_val, y_proba, labels=expected_classes)

            cv_accuracy.append(accuracy)
            cv_logloss.append(logloss)
            valid_folds += 1

        # Check if we have enough valid folds
        if valid_folds == 0:
            logger.error(
                f"   ❌ No valid CV folds! All {n_splits} folds had incomplete class coverage."
            )
            logger.warning(
                f"   → Training final model without CV validation (high variance risk!)"
            )

            # Train final model anyway but warn user
            final_params = params.copy()
            if "early_stopping_rounds" in final_params:
                del final_params["early_stopping_rounds"]

            final_model = XGBClassifier(**final_params)
            final_model.fit(X, y_relabeled, verbose=False)

            # Store mapping info
            final_model.class_mapping_ = class_mapping
            final_model.inverse_mapping_ = inverse_mapping
            final_model.effective_classes_ = len(unique_classes)

            # Return dummy metrics with warning flag
            metrics = {
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "log_loss": 999.0,
                "log_loss_std": 0.0,
                "num_classes": len(unique_classes),
                "class_mapping": class_mapping,
                "cv_warning": "No valid CV folds - temporal class imbalance",
                "valid_folds": 0,
                "skipped_folds": skipped_folds,
            }

            return final_model, metrics

        elif valid_folds < n_splits:
            logger.warning(
                f"   ⚠️  Used {valid_folds}/{n_splits} CV folds "
                f"({skipped_folds} skipped due to missing classes)"
            )

        # Train final model on full data WITHOUT early stopping
        final_params = params.copy()
        if "early_stopping_rounds" in final_params:
            del final_params["early_stopping_rounds"]

        final_model = XGBClassifier(**final_params)
        final_model.fit(X, y_relabeled, verbose=False)

        # Store mapping info in the model for prediction time
        final_model.class_mapping_ = class_mapping
        final_model.inverse_mapping_ = inverse_mapping
        final_model.effective_classes_ = len(unique_classes)

        metrics = {
            "accuracy": np.mean(cv_accuracy),
            "accuracy_std": np.std(cv_accuracy),
            "log_loss": np.mean(cv_logloss),
            "log_loss_std": np.std(cv_logloss),
            "num_classes": len(unique_classes),
            "class_mapping": class_mapping,
            "valid_folds": valid_folds,
            "skipped_folds": skipped_folds,
        }

        return final_model, metrics

    def _calibrate_probabilities(self, model, X, y):
        """
        Calibrate classifier probabilities using isotonic regression.

        FIXED: Creates calibrators for ORIGINAL classes, not relabeled classes.
        """
        y_proba = model.predict_proba(X)

        # Get the number of classes this model actually predicts
        n_predicted_classes = y_proba.shape[1]

        # Create calibrators for ALL 4 original regime classes
        # (not just the classes this specific model was trained on)
        calibrators = []

        for original_class_idx in range(4):  # Always 4 regimes in original taxonomy
            # Check if this model predicts this class
            if hasattr(model, "inverse_mapping_"):
                # Model has collapsed classes - find which predicted class maps to this original class
                predicted_class_idx = None
                for pred_idx, orig_idx in model.inverse_mapping_.items():
                    if orig_idx == original_class_idx:
                        predicted_class_idx = pred_idx
                        break

                if predicted_class_idx is None:
                    # This original class was collapsed away - no calibrator needed
                    calibrators.append(None)
                    continue

                # Use the mapped predicted class for calibration
                class_idx_for_calibration = predicted_class_idx
                y_binary = (y == original_class_idx).astype(int)

            else:
                # Model has all 4 classes - direct mapping
                class_idx_for_calibration = original_class_idx
                y_binary = (y == original_class_idx).astype(int)

            # Only calibrate if we have both positive and negative examples
            if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(y_proba[:, class_idx_for_calibration], y_binary)
                calibrators.append(calibrator)
            else:
                # No calibration possible
                calibrators.append(None)

        return calibrators

    def predict(self, X: pd.DataFrame, cohort: str) -> Dict:
        """
        Generate probabilistic forecast for new data.

        FIXED: Handles models trained with collapsed regime classes.
        """
        if cohort not in self.models:
            raise ValueError(
                f"Cohort {cohort} not trained. Available: {list(self.models.keys())}"
            )

        X_features = X[self.feature_names]

        # Get predictions from all models
        point = self.models[cohort]["point"].predict(X_features)[0]

        quantiles = {}
        for q in self.quantiles:
            q_label = f"q{int(q * 100)}"
            quantiles[q_label] = self.models[cohort]["quantiles"][q].predict(
                X_features
            )[0]

        # Enforce quantile monotonicity
        quantiles = self._enforce_quantile_order(quantiles)

        # Get regime predictions with proper handling of collapsed classes
        regime_model = self.models[cohort]["regime"]
        regime_probs_raw = regime_model.predict_proba(X_features)[0]

        # Initialize probabilities for all 4 original regime classes
        regime_probs_full = np.zeros(4)

        # CRITICAL FIX: Check if model was trained with collapsed classes
        if hasattr(regime_model, "inverse_mapping_"):
            # Model has fewer than 4 classes - map back to original indices
            for new_idx, prob in enumerate(regime_probs_raw):
                original_idx = int(regime_model.inverse_mapping_[new_idx])
                regime_probs_full[original_idx] = prob

            # Log for diagnostics
            logger.debug(
                f"Cohort {cohort}: Mapped {len(regime_probs_raw)} predictions "
                f"to {len(regime_probs_full)} original classes"
            )
        else:
            # Model has all 4 classes, use directly
            regime_probs_full = regime_probs_raw

        # Calibrate regime probabilities if calibrators exist
        if cohort in self.calibrators and "regime" in self.calibrators[cohort]:
            calibrators = self.calibrators[cohort]["regime"]
            regime_probs_calibrated = []

            for class_idx in range(len(regime_probs_full)):
                # Only calibrate if:
                # 1. We have a calibrator for this class
                # 2. The probability is non-zero
                if (
                    class_idx < len(calibrators)
                    and calibrators[class_idx] is not None
                    and regime_probs_full[class_idx] > 0
                ):
                    prob_calibrated = calibrators[class_idx].predict(
                        [regime_probs_full[class_idx]]
                    )[0]
                    regime_probs_calibrated.append(prob_calibrated)
                else:
                    # No calibrator or zero probability - keep original
                    regime_probs_calibrated.append(regime_probs_full[class_idx])

            regime_probs = np.array(regime_probs_calibrated)

            # Renormalize to sum to 1
            if regime_probs.sum() > 0:
                regime_probs = regime_probs / regime_probs.sum()
            else:
                regime_probs = regime_probs_full
        else:
            regime_probs = regime_probs_full

        # Confidence prediction
        confidence = self.models[cohort]["confidence"].predict(X_features)[0]
        confidence = np.clip(confidence, 0, 1)

        return {
            "point_estimate": float(point),
            "quantiles": {k: float(v) for k, v in quantiles.items()},
            "regime_probabilities": {
                self.regime_labels[i].lower(): float(regime_probs[i])
                for i in range(len(self.regime_labels))
            },
            "confidence_score": float(confidence),
            "cohort": cohort,
        }

    def _enforce_quantile_order(self, quantiles: Dict[str, float]) -> Dict[str, float]:
        """Ensure q10 <= q25 <= q50 <= q75 <= q90."""
        sorted_q = sorted(quantiles.items(), key=lambda x: int(x[0][1:]))

        # Forward pass: ensure increasing
        for i in range(1, len(sorted_q)):
            prev_val = sorted_q[i - 1][1]
            curr_val = sorted_q[i][1]
            if curr_val < prev_val:
                sorted_q[i] = (sorted_q[i][0], prev_val)

        return dict(sorted_q)

    def _save_cohort_models(self, cohort: str, save_dir: str):
        """Save models for one cohort to disk."""
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

        logger.info(f"💾 Saved: {cohort_file}")

    def load(self, cohort: str, load_dir: str = "models"):
        """Load trained models for a specific cohort."""
        load_path = Path(load_dir) / f"probabilistic_forecaster_{cohort}.pkl"

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.models[cohort] = data["models"]
        self.calibrators[cohort] = data["calibrators"]
        self.feature_names = data["feature_names"]

        # Restore config
        config = data["config"]
        self.horizon = config["horizon"]
        self.quantiles = config["quantiles"]
        self.regime_boundaries = config["regime_boundaries"]
        self.regime_labels = config["regime_labels"]

        logger.info(f"✅ Loaded cohort: {cohort}")

    def _generate_diagnostics(self, cohort_metrics: Dict, save_dir: str):
        """Generate diagnostic plots and JSON summaries."""
        save_path = Path(save_dir)

        # 1. Export metrics as JSON
        metrics_file = save_path / "probabilistic_model_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(cohort_metrics, f, indent=2)
        logger.info(f"📊 Metrics saved: {metrics_file}")

        # 2. Plot regime classification performance
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            cohorts = list(cohort_metrics.keys())
            accuracies = [cohort_metrics[c]["regime"]["accuracy"] for c in cohorts]
            log_losses = [cohort_metrics[c]["regime"]["log_loss"] for c in cohorts]

            axes[0].bar(range(len(cohorts)), accuracies)
            axes[0].set_xticks(range(len(cohorts)))
            axes[0].set_xticklabels(cohorts, rotation=45, ha="right")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Regime Classification Accuracy by Cohort")
            axes[0].axhline(0.25, color="r", linestyle="--", label="Random (4 classes)")
            axes[0].legend()

            axes[1].bar(range(len(cohorts)), log_losses)
            axes[1].set_xticks(range(len(cohorts)))
            axes[1].set_xticklabels(cohorts, rotation=45, ha="right")
            axes[1].set_ylabel("Log Loss")
            axes[1].set_title("Regime Classification Log Loss by Cohort")

            plt.tight_layout()
            plot_file = save_path / "regime_performance.png"
            plt.savefig(plot_file, dpi=150)
            plt.close()
            logger.info(f"📈 Plot saved: {plot_file}")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

        # 3. Summary table
        logger.info("\n" + "=" * 80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"{'Cohort':<30} | {'Point RMSE':>10} | {'Regime Acc':>10} | {'Conf RMSE':>10}"
        )
        logger.info("-" * 80)

        for cohort in sorted(cohorts):
            m = cohort_metrics[cohort]
            logger.info(
                f"{cohort:<30} | "
                f"{m['point']['rmse']:>9.2f}% | "
                f"{m['regime']['accuracy']:>9.3f} | "
                f"{m['confidence']['rmse']:>9.3f}"
            )

        logger.info("=" * 80)


def train_probabilistic_forecaster(
    df: pd.DataFrame, save_dir: str = "models"
) -> ProbabilisticVIXForecaster:
    """
    Convenience function to train forecaster.

    Args:
        df: DataFrame from feature_engine with calendar_cohort column
        save_dir: Where to save models

    Returns:
        Trained ProbabilisticVIXForecaster instance
    """
    forecaster = ProbabilisticVIXForecaster()
    forecaster.train(df, save_dir=save_dir)
    return forecaster
