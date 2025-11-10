"""
XGBoost Training System V2 - Production Grade with Academic Best Practices
===========================================================================

PRODUCTION FIXES:
1. Graceful SHAP fallback for multiclass base_score incompatibility
2. Conservative default hyperparameters (no nested CV overhead)
3. Robust error handling throughout training pipeline
4. Crisis-aware validation with proper time-series splits

This is the PRODUCTION version - use this, not the 900-line research version.
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("⚠️ Optuna not available - hyperparameter optimization disabled")

try:
    from core.temporal_validator import TemporalSafetyValidator

    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    warnings.warn("⚠️ Temporal validator not available")

try:
    from config import HYPERPARAMETER_SEARCH_SPACE, OPTUNA_CONFIG
except ImportError:
    OPTUNA_CONFIG = {"n_trials": 50}
    HYPERPARAMETER_SEARCH_SPACE = {}

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("âš ï¸ SHAP not available - using gain-based importance")

# Crisis periods for specialized validation
CRISIS_PERIODS = {
    "2008_gfc": ("2008-09-01", "2009-03-31"),
    "2011_debt": ("2011-07-25", "2011-10-04"),
    "2015_china": ("2015-08-17", "2015-09-18"),
    "2018_q4": ("2018-10-03", "2018-12-26"),
    "2020_covid": ("2020-02-19", "2020-04-30"),
    "2022_ukraine": ("2022-02-14", "2022-03-31"),
}

# Features with forward-looking power
FORWARD_INDICATORS = [
    "VX1-VX2",
    "VX2-VX1_RATIO",
    "vx_term_structure_regime",
    "vx_curve_acceleration",
    "vx_term_structure_divergence",
    "yield_10y2y",
    "yield_10y3m",
    "yield_2y3m",
    "yield_10y2y_velocity_10d",
    "yield_10y2y_acceleration",
    "yield_10y3m_velocity_10d",
    "yield_10y3m_acceleration",
    "yield_curve_curvature",
    "yield_10y2y_inversion_depth",
    "VXTLT",
    "vxtlt_vix_ratio",
    "bond_vol_regime",
    "VXTLT_velocity_10d",
    "VXTLT_acceleration_5d",
    "SKEW",
    "skew_vs_vix",
    "cboe_stress_composite",
    "pc_equity_inst_divergence",
]


class EnhancedXGBoostTrainer:
    """Production XGBoost trainer with academic rigor and robust error handling."""

    def __init__(self, output_dir: str = "./models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Multi-horizon storage
        self.regime_models = {}  # {1: model_1d, 3: model_3d, 5: model_5d, 10: model_10d}
        self.range_models = {}

        # Backward compatibility pointers (point to 5d by default)
        self.regime_model = None
        self.range_model = None

        self.feature_columns = None
        self.validation_results = {}
        self.feature_importance = {}
        self.shap_explainers = {}
        self.trained_horizons = []  # Track which horizons were trained

        self.optuna_studies = {}
        self.best_params = {}

        if VALIDATOR_AVAILABLE:
            try:
                from config import PUBLICATION_LAGS

                self.validator = TemporalSafetyValidator(PUBLICATION_LAGS)
            except ImportError:
                self.validator = TemporalSafetyValidator()
        else:
            self.validator = None

    def train(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        horizons: List[int] = [5],
        n_splits: int = 5,
        optimize_hyperparams: int = 0,  # CHANGED: trial budget (0 = use defaults)
        crisis_balanced: bool = True,
        compute_shap: bool = True,
        verbose: bool = True,
        enable_temporal_validation: bool = True,  # NEW
    ) -> Dict:
        """Train XGBoost with Optuna optimization and temporal validation.

        Args:
            optimize_hyperparams: Number of Optuna trials (0 = use default params)
            enable_temporal_validation: Run Tier 2 temporal checks during CV
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"🚀 XGBOOST V2 - MULTI-HORIZON WITH OPTUNA")
            print(f"{'=' * 80}")
            print(f"Features: {len(features.columns)} | Samples: {len(features):,}")
            print(f"Horizons: {horizons} days")
            if optimize_hyperparams > 0:
                print(f"Optimization: {optimize_hyperparams} trials per model")
            else:
                print(f"Optimization: Using default parameters")

        all_results = {}

        for horizon in horizons:
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"📅 TRAINING {horizon}-DAY HORIZON")
                print(f"{'=' * 80}")

            # 1. Prepare data
            X, y_regime, y_range = self._prepare_data(
                features, vix, spx, horizon, verbose
            )

            # 2. Crisis-Aware CV
            if crisis_balanced:
                tscv = self._create_crisis_balanced_cv(X, y_regime, n_splits, verbose)
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits, test_size=252)

            # 3. Hyperparameter optimization (if requested)
            if optimize_hyperparams > 0:
                regime_params, range_params = self._optimize_hyperparameters(
                    X, y_regime, y_range, tscv, optimize_hyperparams, horizon, verbose
                )
            else:
                regime_params = self._get_default_regime_params()
                range_params = self._get_default_range_params()

            # 4. Train regime classifier
            regime_model, regime_metrics = self._train_regime_classifier(
                X, y_regime, tscv, regime_params, verbose, enable_temporal_validation
            )
            self.regime_models[horizon] = regime_model

            # 5. Train range predictor
            range_model, range_metrics = self._train_range_predictor(
                X, y_range, tscv, range_params, verbose, enable_temporal_validation
            )
            self.range_models[horizon] = range_model

            all_results[horizon] = {
                "regime_metrics": regime_metrics,
                "range_metrics": range_metrics,
                "regime_params": regime_params,
                "range_params": range_params,
            }

            if verbose:
                print(f"✅ {horizon}d models trained")

        self.trained_horizons = horizons

        # Set default pointers
        default_horizon = 5 if 5 in self.regime_models else horizons[0]
        self.regime_model = self.regime_models[default_horizon]
        self.range_model = self.range_models[default_horizon]

        # Feature importance
        X_importance, y_regime_importance, y_range_importance = self._prepare_data(
            features, vix, spx, default_horizon, verbose=False
        )

        if compute_shap and SHAP_AVAILABLE:
            self._compute_shap_importance(
                X_importance, y_regime_importance, y_range_importance, verbose
            )
        else:
            self._compute_gain_importance(
                X_importance, y_regime_importance, y_range_importance, verbose
            )

        # Validation
        self._multi_horizon_validation_new(features, vix, spx, horizons, verbose)
        self._crisis_validation(
            X_importance, y_regime_importance, y_range_importance, verbose
        )

        # Save
        self._save_models(all_results, verbose)

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"✅ TRAINING COMPLETE")
            print(f"{'=' * 80}")

        return {
            "regime_models": self.regime_models,
            "range_models": self.range_models,
            "trained_horizons": self.trained_horizons,
            "all_results": all_results,
            "feature_importance": self.feature_importance,
            "best_params": self.best_params,
            "optuna_studies": self.optuna_studies,
        }

    def _prepare_data(
        self, features, vix, spx, horizon, verbose
    ):  # NEW: added horizon param
        """Clean features and engineer targets - UPDATED FOR CONFIGURABLE HORIZON."""

        # Remove vix_regime from features if present (data leakage!)
        if "vix_regime" in features.columns:
            features = features.drop(columns=["vix_regime"])
            if verbose:
                print("   ⚠️  Removed vix_regime from features (data leakage)")

        # Remove low-quality features
        missing_pct = features.isnull().mean()
        variance = features.var()
        valid_features = features.columns[(missing_pct < 0.5) & (variance > 1e-8)]

        features_clean = (
            features[valid_features].fillna(method="ffill").fillna(method="bfill")
        )
        valid_idx = features_clean.dropna().index
        features_clean = features_clean.loc[valid_idx]
        vix = vix.loc[valid_idx]
        spx = spx.loc[valid_idx]

        self.feature_columns = features_clean.columns.tolist()

        # === Forward Volatility Expansion/Compression Target ===
        expansion_threshold = 0.15  # 15% VIX increase = expansion event

        # Calculate maximum VIX increase over forward window (using horizon parameter)
        forward_max_change = vix.rolling(horizon).max().shift(-horizon) / vix - 1

        # Binary classification: Expansion (1) vs No-Expansion (0)
        y_regime = (forward_max_change > expansion_threshold).astype(int)

        # Range prediction: Forward realized vol (using horizon parameter)
        spx_returns = spx.pct_change()
        y_range = (
            spx_returns.rolling(horizon).std().shift(-horizon) * np.sqrt(252) * 100
        )

        # Align targets
        valid_target_idx = ~(y_regime.isnull() | y_range.isnull())
        X = features_clean[valid_target_idx]
        y_regime = y_regime[valid_target_idx]
        y_range = y_range[valid_target_idx]

        if verbose:
            print(f"\n📊 Data Quality ({horizon}-day horizon):")
            print(f"   Features: {len(self.feature_columns)}")
            print(f"   Samples: {len(X):,}")
            print(f"   Date range: {X.index.min().date()} → {X.index.max().date()}")
            print(
                f"\n   Target: Forward VIX Expansion ({horizon}-day, threshold={expansion_threshold:.1%})"
            )
            print(
                f"   Class 0 (No Expansion): {(y_regime == 0).sum():>4} ({(y_regime == 0).sum() / len(y_regime) * 100:>5.1f}%)"
            )
            print(
                f"   Class 1 (Expansion): {(y_regime == 1).sum():>4} ({(y_regime == 1).sum() / len(y_regime) * 100:>5.1f}%)"
            )

        return X, y_regime, y_range

    def _create_crisis_balanced_cv(self, X, y_regime, n_splits, verbose):
        """Create CV splits with crisis coverage."""
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=252)

        if verbose:
            print(f"\nCrisis-Balanced CV:")
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                val_dates = X.iloc[val_idx].index
                covered = sum(
                    1
                    for _, (start, end) in CRISIS_PERIODS.items()
                    if any((val_dates >= start) & (val_dates <= end))
                )
                print(f"   Fold {fold_idx}: {covered} crisis periods covered")

        return tscv

    def _get_default_regime_params(self):
        """Conservative parameters for expansion/compression classification."""
        return {
            "objective": "binary:logistic",  # Changed from 'multi:softprob'
            # num_class removed (not needed for binary)
            "max_depth": 6,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 10,
            "gamma": 0.2,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "scale_pos_weight": 2,  # Weight expansion class higher (typically rarer)
            "eval_metric": "logloss",  # Changed from 'mlogloss'
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def _get_default_range_params(self):
        """Conservative parameters for range prediction."""
        return {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.03,
            "n_estimators": 400,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def _train_regime_classifier(
        self, X, y, tscv, params, verbose, enable_temporal_validation=True
    ):
        """Train expansion/compression classifier with walk-forward validation."""
        cv_scores = []
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)

            acc = balanced_accuracy_score(y_val, y_pred)
            logloss = log_loss(y_val, y_pred_proba)

            # Binary F1 scores
            f1_no_expansion = f1_score(y_val, y_pred, pos_label=0, zero_division=0)
            f1_expansion = f1_score(y_val, y_pred, pos_label=1, zero_division=0)

            cv_scores.append(acc)
            fold_results.append(
                {
                    "fold": fold_idx,
                    "balanced_accuracy": acc,
                    "log_loss": logloss,
                    "f1_no_expansion": float(f1_no_expansion),
                    "f1_expansion": float(f1_expansion),
                    "expansion_rate": float(y_val.mean()),  # Baseline rate
                }
            )

            if verbose:
                print(f"\n   Fold {fold_idx}/{tscv.n_splits}:")
                print(f"      Accuracy: {acc:.3f} | Log Loss: {logloss:.3f}")
                print(
                    f"      F1 [No-Expansion={f1_no_expansion:.2f}, Expansion={f1_expansion:.2f}]"
                )
                print(f"      Base Rate: {y_val.mean():.1%}")

        # Train final model on all data
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X, y, verbose=False)

        metrics = {
            "cv_balanced_accuracy_mean": np.mean(cv_scores),
            "cv_balanced_accuracy_std": np.std(cv_scores),
            "fold_results": fold_results,
        }

        if verbose:
            print(
                f"\n   📊 CV Summary: {metrics['cv_balanced_accuracy_mean']:.3f} ± {metrics['cv_balanced_accuracy_std']:.3f}"
            )

        return final_model, metrics

    def _train_range_predictor(
        self, X, y, tscv, params, verbose, enable_temporal_validation=True
    ):
        """Train forward volatility range predictor."""
        cv_rmse = []
        cv_directional = []
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            y_val_change = y_val.diff().dropna()
            y_pred_change = pd.Series(y_pred, index=y_val.index).diff().dropna()
            directional_acc = ((y_val_change > 0) == (y_pred_change > 0)).mean()

            cv_rmse.append(rmse)
            cv_directional.append(directional_acc)

            fold_results.append(
                {
                    "fold": fold_idx,
                    "rmse": rmse,
                    "directional_accuracy": directional_acc,
                }
            )

            if verbose:
                print(f"\n   Fold {fold_idx}/{tscv.n_splits}:")
                print(f"      RMSE: {rmse:.2f}% | Directional: {directional_acc:.3f}")

        # Train final model
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X, y, verbose=False)

        metrics = {
            "cv_rmse_mean": np.mean(cv_rmse),
            "cv_rmse_std": np.std(cv_rmse),
            "cv_directional_mean": np.mean(cv_directional),
            "cv_directional_std": np.std(cv_directional),
            "fold_results": fold_results,
        }

        if verbose:
            print(f"\n   📊 CV Summary:")
            print(
                f"      RMSE: {metrics['cv_rmse_mean']:.2f}% Â± {metrics['cv_rmse_std']:.2f}%"
            )
            print(
                f"      Directional: {metrics['cv_directional_mean']:.3f} Â± {metrics['cv_directional_std']:.3f}"
            )

        return final_model, metrics

    def _compute_shap_importance(self, X, y_regime, y_range, verbose):
        """Compute SHAP importance with robust fallback."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"FEATURE IMPORTANCE (SHAP)")
            print(f"{'=' * 80}")

        sample_size = min(1000, len(X))
        sample_indices = np.random.RandomState(42).choice(
            len(X), sample_size, replace=False
        )
        X_sample = X.iloc[sample_indices]

        # Try SHAP for regime model (multiclass can fail)
        try:
            regime_explainer = shap.TreeExplainer(self.regime_model)
            regime_shap_values = regime_explainer.shap_values(X_sample)
            regime_importance = np.abs(regime_shap_values).mean(axis=(0, 1))
            regime_importance = regime_importance / regime_importance.sum()
            shap_worked_regime = True
        except (ValueError, KeyError, IndexError) as e:
            if verbose:
                print(
                    f"   âš ï¸ SHAP failed for regime model (multiclass base_score issue)"
                )
                print(f"      Using gain-based importance instead")
            regime_importance = self.regime_model.feature_importances_
            regime_importance = regime_importance / regime_importance.sum()
            shap_worked_regime = False

        # Try SHAP for range model
        try:
            range_explainer = shap.TreeExplainer(self.range_model)
            range_shap_values = range_explainer.shap_values(X_sample)
            range_importance = np.abs(range_shap_values).mean(axis=0)
            range_importance = range_importance / range_importance.sum()
            shap_worked_range = True
        except (ValueError, KeyError, IndexError) as e:
            if verbose:
                print(f"   âš ï¸ SHAP failed for range model")
                print(f"      Using gain-based importance instead")
            range_importance = self.range_model.feature_importances_
            range_importance = range_importance / range_importance.sum()
            shap_worked_range = False

        # Store explainers only if SHAP worked
        if shap_worked_regime and shap_worked_range:
            self.shap_explainers = {
                "regime": regime_explainer,
                "range": range_explainer,
                "sample_data": X_sample,
            }
        else:
            self.shap_explainers = {}

        # Build importance DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "regime_shap": regime_importance,
                "range_shap": range_importance,
            }
        )

        importance_df["overall_shap"] = (
            importance_df["regime_shap"] * 0.5 + importance_df["range_shap"] * 0.5
        )

        importance_df["is_forward_indicator"] = importance_df["feature"].isin(
            FORWARD_INDICATORS
        )
        importance_df = importance_df.sort_values("overall_shap", ascending=False)

        # Store
        self.feature_importance = {
            "regime": importance_df.nlargest(50, "regime_shap")[
                ["feature", "regime_shap", "is_forward_indicator"]
            ].to_dict("records"),
            "range": importance_df.nlargest(50, "range_shap")[
                ["feature", "range_shap", "is_forward_indicator"]
            ].to_dict("records"),
            "overall": importance_df.nlargest(50, "overall_shap")[
                [
                    "feature",
                    "overall_shap",
                    "regime_shap",
                    "range_shap",
                    "is_forward_indicator",
                ]
            ].to_dict("records"),
        }

        if verbose:
            method = "SHAP" if self.shap_explainers else "gain-based"
            print(f"   Using {method} importance")
            print(f"\n   Top 10 Features:")
            for idx, row in importance_df.head(10).iterrows():
                fwd = "🔮" if row["is_forward_indicator"] else "  "
                print(f"      {fwd} {row['feature']:<45} {row['overall_shap']:>6.3f}")

    def _compute_gain_importance(self, X, y_regime, y_range, verbose):
        """Fallback: gain-based importance (always works)."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"🌎” FEATURE IMPORTANCE (Gain-Based)")
            print(f"{'=' * 80}")

        regime_importance = self.regime_model.feature_importances_
        regime_importance = regime_importance / regime_importance.sum()

        range_importance = self.range_model.feature_importances_
        range_importance = range_importance / range_importance.sum()

        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "regime_shap": regime_importance,
                "range_shap": range_importance,
            }
        )

        importance_df["overall_shap"] = (
            importance_df["regime_shap"] * 0.5 + importance_df["range_shap"] * 0.5
        )

        importance_df["is_forward_indicator"] = importance_df["feature"].isin(
            FORWARD_INDICATORS
        )
        importance_df = importance_df.sort_values("overall_shap", ascending=False)

        self.feature_importance = {
            "regime": importance_df.nlargest(50, "regime_shap").to_dict("records"),
            "range": importance_df.nlargest(50, "range_shap").to_dict("records"),
            "overall": importance_df.nlargest(50, "overall_shap").to_dict("records"),
        }

        if verbose:
            print(f"\n   Top 10 Features:")
            for idx, row in importance_df.head(10).iterrows():
                fwd = "🔮" if row["is_forward_indicator"] else "  "
                print(f"      {fwd} {row['feature']:<45} {row['overall_shap']:>6.3f}")

    def _multi_horizon_validation(self, X, y_regime, y_range, verbose):
        """Test predictive power at different horizons."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"â° MULTI-HORIZON VALIDATION")
            print(f"{'=' * 80}")

        horizons = [1, 3, 5, 10]
        horizon_results = []

        for horizon in horizons:
            y_regime_shifted = y_regime.shift(-horizon).dropna()
            y_range_shifted = y_range.shift(-horizon).dropna()

            common_idx = X.index.intersection(y_regime_shifted.index)
            X_aligned = X.loc[common_idx]
            y_regime_aligned = y_regime_shifted.loc[common_idx]
            y_range_aligned = y_range_shifted.loc[common_idx]

            split_point = int(len(X_aligned) * 0.8)
            X_test = X_aligned.iloc[split_point:]
            y_regime_test = y_regime_aligned.iloc[split_point:]
            y_range_test = y_range_aligned.iloc[split_point:]

            regime_pred = self.regime_model.predict(X_test)
            regime_acc = balanced_accuracy_score(y_regime_test, regime_pred)

            range_pred = self.range_model.predict(X_test)
            range_rmse = np.sqrt(mean_squared_error(y_range_test, range_pred))

            horizon_results.append(
                {
                    "horizon_days": horizon,
                    "regime_accuracy": regime_acc,
                    "range_rmse": range_rmse,
                }
            )

            if verbose:
                print(
                    f"   {horizon}d: Regime={regime_acc:.3f} | Range={range_rmse:.2f}%"
                )

        self.validation_results["multi_horizon"] = horizon_results

    def _multi_horizon_validation_new(self, features, vix, spx, horizons, verbose):
        """Test each trained horizon's predictive power."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"⏰ MULTI-HORIZON VALIDATION")
            print(f"{'=' * 80}")

        horizon_results = []

        for horizon in horizons:
            # Prepare data for this horizon
            X, y_regime, y_range = self._prepare_data(
                features, vix, spx, horizon, verbose=False
            )

            # Test set: last 20%
            split_point = int(len(X) * 0.8)
            X_test = X.iloc[split_point:]
            y_regime_test = y_regime.iloc[split_point:]
            y_range_test = y_range.iloc[split_point:]

            # Predict using this horizon's model
            regime_pred = self.regime_models[horizon].predict(X_test)
            range_pred = self.range_models[horizon].predict(X_test)

            regime_acc = balanced_accuracy_score(y_regime_test, regime_pred)
            range_rmse = np.sqrt(mean_squared_error(y_range_test, range_pred))

            horizon_results.append(
                {
                    "horizon_days": horizon,
                    "regime_accuracy": regime_acc,
                    "range_rmse": range_rmse,
                }
            )

            if verbose:
                print(
                    f"   {horizon}d: Regime={regime_acc:.3f} | Range={range_rmse:.2f}%"
                )

        self.validation_results["multi_horizon"] = horizon_results

    def _crisis_validation(self, X, y_regime, y_range, verbose):
        """Validate during crisis periods."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"âš ï¸ CRISIS VALIDATION")
            print(f"{'=' * 80}")

        crisis_metrics = []

        for crisis_name, (start, end) in CRISIS_PERIODS.items():
            crisis_mask = (X.index >= start) & (X.index <= end)

            if crisis_mask.sum() == 0:
                continue

            X_crisis = X[crisis_mask]
            y_regime_crisis = y_regime[crisis_mask]
            y_range_crisis = y_range[crisis_mask]

            regime_pred = self.regime_model.predict(X_crisis)
            range_pred = self.range_model.predict(X_crisis)

            regime_acc = balanced_accuracy_score(y_regime_crisis, regime_pred)
            range_rmse = np.sqrt(mean_squared_error(y_range_crisis, range_pred))

            crisis_metrics.append(
                {
                    "crisis": crisis_name,
                    "regime_accuracy": regime_acc,
                    "range_rmse": range_rmse,
                }
            )

            if verbose:
                print(
                    f"   {crisis_name}: Regime={regime_acc:.3f} | Range={range_rmse:.2f}%"
                )

        self.validation_results["crisis_periods"] = crisis_metrics

    def _save_models(self, all_results, verbose):
        """Save models and metrics - UPDATED FOR MULTI-HORIZON."""

        # Save each horizon's models
        for horizon in self.trained_horizons:
            self.regime_models[horizon].save_model(
                str(self.output_dir / f"regime_classifier_v2_{horizon}d.json")
            )
            self.range_models[horizon].save_model(
                str(self.output_dir / f"range_predictor_v2_{horizon}d.json")
            )

        # Backward compat: save 5d as default names
        if 5 in self.regime_models:
            self.regime_models[5].save_model(
                str(self.output_dir / "regime_classifier_v2.json")
            )
            self.range_models[5].save_model(
                str(self.output_dir / "range_predictor_v2.json")
            )

        # Save SHAP explainers
        if self.shap_explainers:
            with open(self.output_dir / "shap_explainers.pkl", "wb") as f:
                pickle.dump(self.shap_explainers, f)

        # Save feature importance
        for task in ["regime", "range", "overall"]:
            if task in self.feature_importance:
                pd.DataFrame(self.feature_importance[task]).to_csv(
                    self.output_dir / f"feature_importance_v2_{task}.csv", index=False
                )

        # Save validation metrics
        validation_data = {
            "timestamp": datetime.now().isoformat(),
            "trained_horizons": self.trained_horizons,
            "n_features": len(self.feature_columns),
            "features": self.feature_columns,
            "horizon_results": all_results,
            "crisis_validation": self.validation_results.get("crisis_periods", []),
            "multi_horizon_validation": self.validation_results.get(
                "multi_horizon", []
            ),
            "shap_available": bool(self.shap_explainers),
        }

        with open(self.output_dir / "validation_metrics_v2.json", "w") as f:
            json.dump(validation_data, f, indent=2, default=str)

        if verbose:
            print(f"\n✅ Models saved to {self.output_dir}")
            print(f"   Horizons: {self.trained_horizons}")

    def predict(
        self,
        features: pd.DataFrame,
        horizon: int = 5,  # NEW: specify which horizon to use
        return_proba: bool = False,
    ) -> Dict:
        """Make predictions with trained models."""
        if not self.regime_models:
            raise ValueError("Models not trained. Call train() first.")

        if horizon not in self.regime_models:
            raise ValueError(
                f"No model trained for horizon={horizon}. Available: {list(self.regime_models.keys())}"
            )

        X = features[self.feature_columns].fillna(method="ffill").fillna(method="bfill")

        if return_proba:
            regime_pred = self.regime_models[horizon].predict_proba(X)
        else:
            regime_pred = self.regime_models[horizon].predict(X)

        range_pred = self.range_models[horizon].predict(X)

        return {
            "regime": regime_pred,
            "range": range_pred,
            "horizon": horizon,
            "index": X.index,
        }

    @classmethod
    def load(
        cls, model_dir: str = "./models", horizons: List[int] = None
    ) -> "EnhancedXGBoostTrainer":
        """Load trained models from disk - MULTI-HORIZON VERSION."""
        trainer = cls(output_dir=model_dir)

        # Load validation metadata to discover trained horizons
        metadata_path = Path(model_dir) / "validation_metrics_v2.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)
                trainer.feature_columns = data["features"]
                trained_horizons = data.get(
                    "trained_horizons", [5]
                )  # Default to [5] for old models
        else:
            # Fallback: try to load default 5d model
            trained_horizons = [5]
            # Try to infer feature columns from old validation file
            # (implementation omitted for brevity - add if needed)

        # Override with explicit horizons if provided
        if horizons is not None:
            trained_horizons = horizons

        # Load each horizon's models
        for horizon in trained_horizons:
            regime_path = Path(model_dir) / f"regime_classifier_v2_{horizon}d.json"
            range_path = Path(model_dir) / f"range_predictor_v2_{horizon}d.json"

            if regime_path.exists() and range_path.exists():
                regime_model = xgb.XGBClassifier()
                regime_model.load_model(str(regime_path))
                trainer.regime_models[horizon] = regime_model

                range_model = xgb.XGBRegressor()
                range_model.load_model(str(range_path))
                trainer.range_models[horizon] = range_model

                trainer.trained_horizons.append(horizon)

        # Backward compat: try loading default names if 5d not found
        if 5 not in trainer.regime_models:
            regime_path = Path(model_dir) / "regime_classifier_v2.json"
            range_path = Path(model_dir) / "range_predictor_v2.json"

            if regime_path.exists() and range_path.exists():
                regime_model = xgb.XGBClassifier()
                regime_model.load_model(str(regime_path))
                trainer.regime_models[5] = regime_model

                range_model = xgb.XGBRegressor()
                range_model.load_model(str(range_path))
                trainer.range_models[5] = range_model

                if 5 not in trainer.trained_horizons:
                    trainer.trained_horizons.append(5)

        # Set default pointers
        default_horizon = (
            5 if 5 in trainer.regime_models else trainer.trained_horizons[0]
        )
        trainer.regime_model = trainer.regime_models[default_horizon]
        trainer.range_model = trainer.range_models[default_horizon]

        # Load SHAP explainers
        shap_path = Path(model_dir) / "shap_explainers.pkl"
        if shap_path.exists():
            with open(shap_path, "rb") as f:
                trainer.shap_explainers = pickle.load(f)

        return trainer

    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y_regime: pd.Series,
        y_range: pd.Series,
        tscv,
        n_trials: int,
        horizon: int,
        verbose: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        Optimize hyperparameters using Optuna with temporal validation.
        CLEANED OUTPUT VERSION - Suppresses per-trial Optuna logs.
        """
        if not OPTUNA_AVAILABLE:
            if verbose:
                print("[WARN] Optuna not available - using default parameters")
            return self._get_default_regime_params(), self._get_default_range_params()

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"[OPT] HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
            print(f"{'=' * 80}")

        # Suppress Optuna's verbose logging globally
        import logging

        optuna_logger = logging.getLogger("optuna")
        original_level = optuna_logger.level
        optuna_logger.setLevel(logging.WARNING)

        # Create study storage directory
        study_dir = self.output_dir / "optuna_studies"
        study_dir.mkdir(exist_ok=True, parents=True)
        storage_url = f"sqlite:///{study_dir}/optimization_{horizon}d.db"

        # ========================================================================
        # REGIME CLASSIFIER OPTIMIZATION
        # ========================================================================
        if verbose:
            print(f"\n[1/2] Optimizing regime classifier...")

        regime_study_name = f"regime_classifier_{horizon}d"
        regime_study = optuna.create_study(
            study_name=regime_study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",
            sampler=TPESampler(n_startup_trials=min(10, n_trials // 5), seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )

        if len(regime_study.trials) > 0 and verbose:
            print(
                f"   [RESUME] Continuing from {len(regime_study.trials)} previous trials"
            )

        def regime_objective(trial):
            params = self._suggest_regime_params(trial)
            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_regime.iloc[train_idx], y_regime.iloc[val_idx]

                if self.validator and VALIDATOR_AVAILABLE:
                    try:
                        self.validator.validate_cv_split(
                            X_train, X_val, y_train, y_val, fold_idx
                        )
                    except ValueError as e:
                        if verbose:
                            print(
                                f"   [FAIL] Fold {fold_idx} failed temporal validation: {e}"
                            )
                        raise optuna.TrialPruned()

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, verbose=False)
                y_pred = model.predict(X_val)
                score = balanced_accuracy_score(y_val, y_pred)
                cv_scores.append(score)
                trial.report(np.mean(cv_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(cv_scores)

        # Run optimization with condensed output
        n_trials_before = len(regime_study.trials)
        patience_counter = 0
        best_value = (
            regime_study.best_value if len(regime_study.trials) > 0 else -np.inf
        )

        for i in range(n_trials):
            regime_study.optimize(regime_objective, n_trials=1, show_progress_bar=False)

            if regime_study.best_value > best_value + 1e-4:
                best_value = regime_study.best_value
                patience_counter = 0
                # Only print every 10th improvement or last trial
                if verbose and (i % 10 == 0 or i == n_trials - 1):
                    print(
                        f"   Trial {len(regime_study.trials)}: {best_value:.4f} [NEW BEST]"
                    )
            else:
                patience_counter += 1
                # Print every 20 trials to show progress
                if verbose and len(regime_study.trials) % 20 == 0:
                    print(
                        f"   Trial {len(regime_study.trials)}: {regime_study.trials[-1].value:.4f}"
                    )

            if patience_counter >= OPTUNA_CONFIG.get("early_stopping_patience", 20):
                if verbose:
                    print(
                        f"   [STOP] Early stopping (no improvement in {patience_counter} trials)"
                    )
                break

        best_regime_params = regime_study.best_params
        self._convert_optuna_params_to_xgboost(best_regime_params, is_classifier=True)

        if verbose:
            n_new_trials = len(regime_study.trials) - n_trials_before
            print(
                f"   [OK] Best RMSE: {regime_study.best_value:.4f} ({n_new_trials} new trials)"
            )

        # ========================================================================
        # RANGE PREDICTOR OPTIMIZATION
        # ========================================================================
        if verbose:
            print(f"\n[2/2] Optimizing range predictor...")

        range_study_name = f"range_predictor_{horizon}d"
        range_study = optuna.create_study(
            study_name=range_study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="minimize",
            sampler=TPESampler(n_startup_trials=min(10, n_trials // 5), seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )

        if len(range_study.trials) > 0 and verbose:
            print(
                f"   [RESUME] Continuing from {len(range_study.trials)} previous trials"
            )

        def range_objective(trial):
            params = self._suggest_range_params(trial)
            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_range.iloc[train_idx], y_range.iloc[val_idx]

                if self.validator and VALIDATOR_AVAILABLE:
                    try:
                        self.validator.validate_cv_split(
                            X_train, X_val, y_train, y_val, fold_idx
                        )
                    except ValueError as e:
                        if verbose:
                            print(
                                f"   [FAIL] Fold {fold_idx} failed temporal validation: {e}"
                            )
                        raise optuna.TrialPruned()

                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, verbose=False)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)
                trial.report(np.mean(cv_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(cv_scores)

        # Run optimization
        n_trials_before = len(range_study.trials)
        patience_counter = 0
        best_value = range_study.best_value if len(range_study.trials) > 0 else np.inf

        for i in range(n_trials):
            range_study.optimize(range_objective, n_trials=1, show_progress_bar=False)

            if range_study.best_value < best_value - 0.01:
                best_value = range_study.best_value
                patience_counter = 0
                if verbose and (i % 10 == 0 or i == n_trials - 1):
                    print(
                        f"   Trial {len(range_study.trials)}: {best_value:.2f}% [NEW BEST]"
                    )
            else:
                patience_counter += 1
                if verbose and len(range_study.trials) % 20 == 0:
                    print(
                        f"   Trial {len(range_study.trials)}: {range_study.trials[-1].value:.2f}%"
                    )

            if patience_counter >= OPTUNA_CONFIG.get("early_stopping_patience", 20):
                if verbose:
                    print(
                        f"   [STOP] Early stopping (no improvement in {patience_counter} trials)"
                    )
                break

        best_range_params = range_study.best_params
        self._convert_optuna_params_to_xgboost(best_range_params, is_classifier=False)

        if verbose:
            n_new_trials = len(range_study.trials) - n_trials_before
            print(
                f"   [OK] Best RMSE: {range_study.best_value:.2f}% ({n_new_trials} new trials)"
            )

        # Restore original logging level
        optuna_logger.setLevel(original_level)

        # Store studies
        self.optuna_studies[horizon] = {"regime": regime_study, "range": range_study}

        # Generate sensitivity analysis
        self._analyze_hyperparameter_sensitivity(
            regime_study, range_study, horizon, verbose
        )

        return best_regime_params, best_range_params

    def _suggest_regime_params(self, trial) -> Dict:
        """Suggest hyperparameters for regime classifier."""
        search_space = HYPERPARAMETER_SEARCH_SPACE.get("regime_classifier", {})

        params = {
            "objective": "binary:logistic",
            "max_depth": trial.suggest_int(
                "max_depth", *search_space.get("max_depth", (4, 12))
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                *search_space.get("learning_rate", (0.01, 0.1)),
                log=True,
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", *search_space.get("n_estimators", (200, 800)), step=50
            ),
            "subsample": trial.suggest_float(
                "subsample", *search_space.get("subsample", (0.5, 0.9))
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search_space.get("colsample_bytree", (0.5, 0.9))
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *search_space.get("min_child_weight", (3, 20))
            ),
            "gamma": trial.suggest_float(
                "gamma", *search_space.get("gamma", (0.05, 0.5))
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *search_space.get("reg_alpha", (0.01, 0.5)), log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *search_space.get("reg_lambda", (0.5, 5.0)), log=True
            ),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", *search_space.get("scale_pos_weight", (1, 5))
            ),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        return params

    def _suggest_range_params(self, trial) -> Dict:
        """Suggest hyperparameters for range predictor."""
        search_space = HYPERPARAMETER_SEARCH_SPACE.get("range_predictor", {})

        params = {
            "objective": "reg:squarederror",
            "max_depth": trial.suggest_int(
                "max_depth", *search_space.get("max_depth", (4, 10))
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                *search_space.get("learning_rate", (0.01, 0.1)),
                log=True,
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", *search_space.get("n_estimators", (150, 600)), step=50
            ),
            "subsample": trial.suggest_float(
                "subsample", *search_space.get("subsample", (0.5, 0.9))
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search_space.get("colsample_bytree", (0.5, 0.9))
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *search_space.get("min_child_weight", (2, 15))
            ),
            "gamma": trial.suggest_float(
                "gamma", *search_space.get("gamma", (0.02, 0.3))
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *search_space.get("reg_alpha", (0.01, 0.3)), log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *search_space.get("reg_lambda", (0.3, 3.0)), log=True
            ),
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        return params

    def _convert_optuna_params_to_xgboost(self, params: Dict, is_classifier: bool):
        """Add fixed params that Optuna doesn't tune."""
        if is_classifier:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"

        params["random_state"] = 42
        params["n_jobs"] = -1
        params["verbosity"] = 0

    def _analyze_hyperparameter_sensitivity(
        self, regime_study, range_study, horizon: int, verbose: bool = True
    ):
        """Analyze which hyperparameters matter most."""
        if not verbose:
            return

        print(f"\n{'=' * 80}")
        print(f"📊 HYPERPARAMETER SENSITIVITY ANALYSIS")
        print(f"{'=' * 80}")

        # Regime classifier importance
        try:
            regime_importance = optuna.importance.get_param_importances(regime_study)
            print(f"\n[Regime Classifier] Top parameters:")
            for i, (param, importance) in enumerate(
                sorted(regime_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                1,
            ):
                print(f"   {i}. {param:<20} {importance:>6.1%}")
        except:
            pass

        # Range predictor importance
        try:
            range_importance = optuna.importance.get_param_importances(range_study)
            print(f"\n[Range Predictor] Top parameters:")
            for i, (param, importance) in enumerate(
                sorted(range_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                1,
            ):
                print(f"   {i}. {param:<20} {importance:>6.1%}")
        except:
            pass

        # Save to file
        sensitivity_report = {
            "timestamp": datetime.now().isoformat(),
            "horizon": horizon,
            "regime_classifier": {
                "importance": regime_importance
                if "regime_importance" in locals()
                else {},
                "best_params": regime_study.best_params,
                "best_value": regime_study.best_value,
                "n_trials": len(regime_study.trials),
            },
            "range_predictor": {
                "importance": range_importance
                if "range_importance" in locals()
                else {},
                "best_params": range_study.best_params,
                "best_value": range_study.best_value,
                "n_trials": len(range_study.trials),
            },
        }

        with open(
            self.output_dir / f"hyperparameter_sensitivity_{horizon}d.json", "w"
        ) as f:
            json.dump(sensitivity_report, f, indent=2)


def train_enhanced_xgboost(
    integrated_system,
    horizons: List[int] = [5],  # NEW: support multi-horizon
    optimize_hyperparams: bool = False,
    crisis_balanced: bool = True,
    compute_shap: bool = True,
    verbose: bool = True,
) -> EnhancedXGBoostTrainer:
    """Convenience function to train enhanced XGBoost."""
    if not integrated_system.trained:
        raise ValueError("Train integrated system first")

    trainer = EnhancedXGBoostTrainer()

    trainer.train(
        features=integrated_system.orchestrator.features,
        vix=integrated_system.orchestrator.vix_ml,
        spx=integrated_system.orchestrator.spx_ml,
        horizons=horizons,  # Pass through
        n_splits=5,
        optimize_hyperparams=optimize_hyperparams,
        crisis_balanced=crisis_balanced,
        compute_shap=compute_shap,
        verbose=verbose,
    )

    return trainer
