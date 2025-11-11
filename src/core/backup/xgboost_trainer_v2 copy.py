"""
XGBoost Training System V2 - VIX Expansion Forecasting
========================================================

Production-grade XGBoost trainer for predicting forward VIX expansion/compression.
Target: Binary classification of whether VIX expands >threshold% over forecast horizon.

FEATURES:
- Multi-horizon training (1d, 3d, 5d, 10d forward predictions)
- Optuna hyperparameter optimization with early stopping
- Crisis-aware cross-validation
- Temporal safety validation
- SHAP explainability (with fallback to gain-based importance)
- Probability calibration for improved reliability
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from config import CRISIS_PERIODS

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn(
        "WARNING: Optuna not available - hyperparameter optimization disabled"
    )

try:
    from core.temporal_validator import TemporalSafetyValidator

    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    warnings.warn("WARNING: Temporal validator not available")

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
    warnings.warn("WARNING: SHAP not available - using gain-based importance")


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression


@dataclass
class VIXDistribution:
    """Container for a complete VIX forecast distribution"""

    timestamp: pd.Timestamp
    point_estimate: float  # Expected % change
    quantiles: Dict[float, float]  # {0.10: val, 0.25: val, ...}
    regime_probs: Dict[str, float]  # {'low_vol': 0.2, 'normal': 0.6, ...}
    confidence_score: float  # 0-1 based on feature quality
    feature_provenance: Dict[str, bool]  # Which features were available
    calendar_context: str  # Which context was active
    actual_outcome: Optional[float] = None  # Populated post-facto


class VIXExpansionTrainer:
    """Production XGBoost trainer for VIX expansion forecasting."""

    def __init__(self, output_dir: str = "./models", expansion_threshold: float = 0.15):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.expansion_threshold = expansion_threshold
        self.models = {}
        self.feature_columns = None
        self.validation_results = {}
        self.feature_importance = {}
        self.shap_explainers = {}
        self.trained_horizons = []
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
        optimize_hyperparams: int = 0,
        crisis_balanced: bool = True,
        compute_shap: bool = True,
        verbose: bool = True,
        enable_temporal_validation: bool = True,
    ) -> Dict:
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"XGBOOST V2 - VIX EXPANSION FORECASTING")
            print(f"{'=' * 80}")
            print(f"Features: {len(features.columns)} | Samples: {len(features):,}")
            print(f"Horizons: {horizons} days")
            print(f"Expansion threshold: {self.expansion_threshold:.1%}")
            if optimize_hyperparams > 0:
                print(f"Optimization: {optimize_hyperparams} trials per horizon")
            else:
                print(f"Optimization: Using default parameters")

        all_results = {}

        for horizon in horizons:
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"TRAINING {horizon}-DAY HORIZON")
                print(f"{'=' * 80}")

            X, y_expansion = self._prepare_data(features, vix, spx, horizon, verbose)

            if crisis_balanced:
                tscv = self._create_crisis_balanced_cv(
                    X, y_expansion, n_splits, verbose
                )
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits, test_size=252)

            if optimize_hyperparams > 0:
                should_skip = self._should_skip_optimization(
                    horizon, force_optimize=True
                )
                if should_skip:
                    if verbose:
                        print(
                            f"\n[CACHE] Using cached hyperparameters (study < 7 days old)"
                        )
                    db_path = (
                        self.output_dir
                        / "optuna_studies"
                        / f"optimization_{horizon}d.db"
                    )
                    storage = f"sqlite:///{db_path}"
                    study = optuna.load_study(
                        study_name=f"vix_expansion_{horizon}d", storage=storage
                    )
                    best_params = study.best_params
                    self._convert_optuna_params(best_params)
                else:
                    best_params = self._optimize_hyperparameters(
                        X, y_expansion, tscv, optimize_hyperparams, horizon, verbose
                    )
            else:
                best_params = self._get_default_params()

            model, metrics = self._train_expansion_model(
                X, y_expansion, tscv, best_params, verbose, enable_temporal_validation
            )
            self.models[horizon] = model
            self.validation_results[horizon] = metrics

            all_results[horizon] = {"metrics": metrics, "params": best_params}

            if verbose:
                print(f"OK: {horizon}d model trained")

        self.trained_horizons = horizons

        default_horizon = 5 if 5 in self.models else horizons[0]
        X_importance, y_importance = self._prepare_data(
            features, vix, spx, default_horizon, verbose=False
        )

        if compute_shap and SHAP_AVAILABLE:
            self._compute_shap_importance(default_horizon, X_importance, verbose)
        else:
            self._compute_gain_importance(default_horizon, verbose)

        if len(horizons) > 1 and verbose:
            self._validate_multi_horizon(features, vix, spx, horizons, verbose)

        if verbose:
            self._validate_on_crises(features, vix, spx, default_horizon, verbose)

        try:
            self._validate_model_performance(default_horizon, verbose)
        except AssertionError as e:
            if verbose:
                print(f"\n{e}")
                print("\n💡 Suggestions:")
                print(
                    f"  1. Lower expansion threshold (current: {self.expansion_threshold:.1%})"
                )
                print(
                    "  2. Increase cost_ratio in _calculate_optimal_scale_pos_weight()"
                )
                print("  3. Run hyperparameter optimization (--optimize 50)")
            raise

        self._save_models(verbose)

        return {
            "models": self.models,
            "trained_horizons": self.trained_horizons,
            "all_results": all_results,
            "feature_importance": self.feature_importance,
            "best_params": self.best_params,
            "optuna_studies": self.optuna_studies,
        }

    def _calculate_optimal_scale_pos_weight(
        self, y: pd.Series, cost_ratio: float = 2.0
    ) -> float:
        n_negative = (y == 0).sum()
        n_positive = (y == 1).sum()

        if n_positive == 0:
            return 2.0

        base_scale = n_negative / n_positive
        adjusted_scale = base_scale * cost_ratio
        return np.clip(adjusted_scale, 2.0, 20.0)

    def _prepare_data(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        horizon: int,
        verbose: bool,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if "vix_regime" in features.columns:
            features = features.drop(columns=["vix_regime"])
            if verbose:
                print("   WARNING: Removed vix_regime from features (data leakage)")

        aligned_idx = features.index.intersection(vix.index).intersection(spx.index)
        X = features.loc[aligned_idx].copy()
        vix_aligned = vix.loc[aligned_idx]

        vix_future = vix_aligned.shift(-horizon)
        vix_pct_change = (vix_future - vix_aligned) / vix_aligned
        y_expansion = (vix_pct_change > self.expansion_threshold).astype(int)

        valid_idx = ~(vix_future.isna() | y_expansion.isna())
        X = X[valid_idx]
        y_expansion = y_expansion[valid_idx]

        self.feature_columns = X.columns.tolist()

        if verbose:
            print(f"   Raw data: {len(features)} rows")
            print(f"   After alignment: {len(X)} samples")
            print(f"   Date range: {X.index[0].date()} → {X.index[-1].date()}")
            print(
                f"   Target: Forward VIX Expansion ({horizon}-day, threshold={self.expansion_threshold:.1%})"
            )
            n_no_expansion = (y_expansion == 0).sum()
            n_expansion = (y_expansion == 1).sum()
            print(
                f"      Class 0 (No Expansion): {n_no_expansion} ({n_no_expansion / len(y_expansion) * 100:5.1f}%)"
            )
            print(
                f"      Class 1 (Expansion): {n_expansion:4} ({n_expansion / len(y_expansion) * 100:5.1f}%)"
            )

        return X, y_expansion

    def _create_crisis_balanced_cv(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int, verbose: bool
    ) -> TimeSeriesSplit:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=252)

        if verbose:
            print(f"\nCrisis-Balanced CV (with regime transition validation):")

            try:
                vix_series = X["vix"] if "vix" in X.columns else None
            except:
                vix_series = None
                print(
                    "   [INFO] VIX not directly available, checking fold coverage only"
                )

            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                test_dates = X.index[test_idx]

                n_crisis = sum(
                    any(
                        pd.Timestamp(start) <= date <= pd.Timestamp(end)
                        for date in test_dates
                    )
                    for start, end in CRISIS_PERIODS.values()
                )

                regime_transition = False
                if vix_series is not None:
                    fold_vix = vix_series.iloc[test_idx].dropna()
                    if len(fold_vix) > 0:
                        vix_min = fold_vix.min()
                        vix_max = fold_vix.max()
                        regime_transition = vix_min < 24.40 and vix_max > 24.40

                transition_marker = "✓" if regime_transition else "○"
                print(
                    f"   Fold {fold_idx}: {n_crisis} crisis periods | Regime transition: {transition_marker}"
                )

        return tscv

    def _get_default_params(self) -> dict:
        return {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "gamma": 0.05,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def _should_skip_optimization(
        self, horizon: int, force_optimize: bool = False
    ) -> bool:
        if force_optimize or not OPTUNA_AVAILABLE:
            return False

        db_path = self.output_dir / "optuna_studies" / f"optimization_{horizon}d.db"
        if not db_path.exists():
            return False

        try:
            storage = f"sqlite:///{db_path}"
            study = optuna.load_study(
                study_name=f"vix_expansion_{horizon}d", storage=storage
            )

            if len(study.trials) == 0:
                return False

            last_trial_time = study.trials[-1].datetime_complete
            if last_trial_time is None:
                return False

            days_since_last = (datetime.now() - last_trial_time).days

            patience = OPTUNA_CONFIG.get("early_stopping_patience", 15)
            if len(study.trials) >= patience:
                recent_trials = study.trials[-patience:]
                best_in_recent = min(
                    t.value for t in recent_trials if t.value is not None
                )
                converged = best_in_recent >= study.best_value
            else:
                converged = False

            return days_since_last <= 7 and converged

        except Exception:
            return False

    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tscv: TimeSeriesSplit,
        n_trials: int,
        horizon: int,
        verbose: bool,
    ) -> Dict:
        if not OPTUNA_AVAILABLE:
            if verbose:
                print("   WARNING: Optuna not available, using defaults")
            return self._get_default_params()

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"[OPT] HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
            print(f"{'=' * 80}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        db_path = self.output_dir / "optuna_studies" / f"optimization_{horizon}d.db"
        db_path.parent.mkdir(exist_ok=True, parents=True)
        storage = f"sqlite:///{db_path}"

        study = optuna.create_study(
            study_name=f"vix_expansion_{horizon}d",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        n_trials_before = len(study.trials)
        if n_trials_before > 0 and verbose:
            print(f"   [RESUME] Continuing from {n_trials_before} previous trials")

        def objective(trial):
            params = self._suggest_params(trial)
            fold_metrics = {"log_loss": [], "f1_expansion": []}

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                try:
                    ll = log_loss(y_test, y_pred_proba, labels=[0, 1])
                    fold_metrics["log_loss"].append(ll)
                except ValueError:
                    fold_metrics["log_loss"].append(10.0)

                try:
                    f1_scores = f1_score(y_test, y_pred, average=None, labels=[0, 1])
                    f1_exp = f1_scores[1] if len(f1_scores) > 1 else 0.0
                    fold_metrics["f1_expansion"].append(f1_exp)
                except:
                    fold_metrics["f1_expansion"].append(0.0)

            trial.set_user_attr(
                "mean_f1_expansion", np.mean(fold_metrics["f1_expansion"])
            )
            return np.mean(fold_metrics["log_loss"])

        best_value = study.best_value if study.trials else float("inf")
        patience_counter = 0

        for _ in range(n_trials):
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            if study.best_value < best_value:
                best_value = study.best_value
                patience_counter = 0
                if verbose and len(study.trials) % 10 == 0:
                    print(
                        f"   Trial {len(study.trials)}: {study.best_value:.4f} [NEW BEST]"
                    )
            else:
                patience_counter += 1
                if verbose and len(study.trials) % 20 == 0:
                    print(f"   Trial {len(study.trials)}: {study.trials[-1].value:.4f}")

            if patience_counter >= OPTUNA_CONFIG.get("early_stopping_patience", 15):
                if verbose:
                    print(
                        f"   [STOP] Early stopping (no improvement in {patience_counter} trials)"
                    )
                break

        best_params = study.best_params
        self._convert_optuna_params(best_params)

        if verbose:
            n_new_trials = len(study.trials) - n_trials_before
            best_trial = study.best_trial
            best_f1 = best_trial.user_attrs.get("mean_f1_expansion", "N/A")

            print(f"   [OK] Best metrics:")
            print(f"        Log Loss: {study.best_value:.4f}")
            if isinstance(best_f1, float):
                print(f"        F1 (Expansion): {best_f1:.4f}")
            print(f"        Trials: {n_new_trials} new")

        self.optuna_studies[horizon] = study
        self.best_params[horizon] = best_params

        if verbose:
            self._analyze_hyperparameter_sensitivity(study, horizon)

        return best_params

    def _suggest_params(self, trial) -> Dict:
        search_space = HYPERPARAMETER_SEARCH_SPACE.get("vix_expansion", {})

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

    def _convert_optuna_params(self, params: Dict):
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        params["random_state"] = 42
        params["n_jobs"] = -1
        params["verbosity"] = 0

    def _analyze_hyperparameter_sensitivity(self, study, horizon: int):
        print(f"\n{'=' * 80}")
        print(f"HYPERPARAMETER SENSITIVITY ANALYSIS")
        print(f"{'=' * 80}")

        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\nTop parameters by importance:")
            for i, (param, imp) in enumerate(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5], 1
            ):
                print(f"   {i}. {param:<20} {imp:>6.1%}")

            report = {
                "timestamp": datetime.now().isoformat(),
                "horizon": horizon,
                "importance": importance,
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
            }

            with open(
                self.output_dir / f"hyperparameter_sensitivity_{horizon}d.json", "w"
            ) as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"   WARNING: Could not compute parameter importance: {e}")

    def _train_expansion_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tscv,
        params: dict,
        verbose: bool,
        enable_temporal_validation: bool,
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        optimal_scale = self._calculate_optimal_scale_pos_weight(y, cost_ratio=2.0)
        params["scale_pos_weight"] = optimal_scale

        if verbose:
            class_dist = y.value_counts(normalize=True)
            print(
                f"\n   Class Distribution: {class_dist[0]:.1%} negative, {class_dist[1]:.1%} positive"
            )
            print(f"   Using scale_pos_weight: {optimal_scale:.1f}")

        cv_metrics = []
        fold_predictions = []
        fold_targets = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if enable_temporal_validation and self.validator:
                try:
                    violations = self.validator.validate_split(
                        X_train, X_test, X.columns.tolist()
                    )
                    if violations and verbose:
                        print(f"   ⚠️  Fold {fold_idx} temporal issue: {violations[0]}")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Temporal validation failed: {e}")

            model = xgb.XGBClassifier(**params, enable_categorical=False)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fold_predictions.append(y_pred_proba)
            fold_targets.append(y_test.values)

            precisions, recalls, thresholds = precision_recall_curve(
                y_test, y_pred_proba
            )
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = (
                thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            )

            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

            accuracy = balanced_accuracy_score(y_test, y_pred_optimal)

            try:
                logloss = log_loss(y_test, y_pred_proba, labels=[0, 1])
            except ValueError:
                logloss = np.nan

            try:
                f1_scores_class = f1_score(
                    y_test, y_pred_optimal, average=None, labels=[0, 1]
                )
                f1_no_exp = f1_scores_class[0] if len(f1_scores_class) > 0 else 0.0
                f1_exp = f1_scores_class[1] if len(f1_scores_class) > 1 else 0.0
            except:
                f1_no_exp, f1_exp = 0.0, 0.0

            recall_exp = recall_score(
                y_test, y_pred_optimal, pos_label=1, zero_division=0
            )
            precision_exp = precision_score(
                y_test, y_pred_optimal, pos_label=1, zero_division=0
            )

            cm = confusion_matrix(y_test, y_pred_optimal, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            base_rate = y_test.mean()

            if verbose:
                logloss_str = f"{logloss:.3f}" if not np.isnan(logloss) else "N/A"
                print(f"\n   Fold {fold_idx}/{tscv.n_splits}:")
                print(
                    f"      Optimal Threshold: {optimal_threshold:.3f} (default=0.500)"
                )
                print(f"      Accuracy: {accuracy:.3f} | Log Loss: {logloss_str}")
                print(
                    f"      F1 [No-Expansion={f1_no_exp:.2f}, Expansion={f1_exp:.2f}]"
                )
                print(
                    f"      Expansion Metrics: Recall={recall_exp:.2f} | Precision={precision_exp:.2f}"
                )
                print(f"      Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
                print(f"      Base Rate: {base_rate:.1%}")

            cv_metrics.append(
                {
                    "fold": fold_idx,
                    "optimal_threshold": optimal_threshold,
                    "accuracy": accuracy,
                    "log_loss": logloss,
                    "f1_no_expansion": f1_no_exp,
                    "f1_expansion": f1_exp,
                    "recall_expansion": recall_exp,
                    "precision_expansion": precision_exp,
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "base_rate": base_rate,
                }
            )

        all_pred = np.concatenate(fold_predictions)
        all_true = np.concatenate(fold_targets)

        precisions, recalls, thresholds = precision_recall_curve(all_true, all_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        global_optimal_idx = np.argmax(f1_scores)
        global_optimal_threshold = (
            thresholds[global_optimal_idx]
            if global_optimal_idx < len(thresholds)
            else 0.5
        )

        if verbose:
            print(f"\n   📊 Global Optimal Threshold: {global_optimal_threshold:.3f}")
            print(f"      (Used for all future predictions)")

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"TRAINING FINAL CALIBRATED MODEL")
            print(f"{'=' * 80}")

        X_train_final, y_train_final, X_cal, y_cal = self._get_calibration_split(
            X, y, calibration_fraction=0.2
        )

        if verbose:
            print(f"   Training set: {len(X_train_final)} samples")
            print(
                f"   Calibration set: {len(X_cal)} samples ({len(X_cal) / len(X):.1%})"
            )

        final_model_base = xgb.XGBClassifier(**params)
        final_model_base.fit(X_train_final, y_train_final, verbose=False)

        final_model = self._calibrate_model(
            final_model_base, X_cal, y_cal, method="isotonic", verbose=verbose
        )

        if verbose:
            y_prob_uncal = final_model_base.predict_proba(X_cal)[:, 1]
            y_prob_cal = final_model.predict_proba(X_cal)[:, 1]
            self._plot_calibration_curve(
                y_cal,
                y_prob_uncal,
                y_prob_cal,
                horizon=5,
                output_dir=self.output_dir,
                verbose=verbose,
            )

        final_model._optimal_threshold = global_optimal_threshold
        final_model._feature_names = X.columns.tolist()

        avg_accuracy = np.mean([m["accuracy"] for m in cv_metrics])
        std_accuracy = np.std([m["accuracy"] for m in cv_metrics])
        avg_f1_expansion = np.mean([m["f1_expansion"] for m in cv_metrics])
        avg_recall_expansion = np.mean([m["recall_expansion"] for m in cv_metrics])
        avg_precision_expansion = np.mean(
            [m["precision_expansion"] for m in cv_metrics]
        )

        valid_logloss = [
            m["log_loss"] for m in cv_metrics if not np.isnan(m["log_loss"])
        ]
        avg_logloss = np.mean(valid_logloss) if valid_logloss else np.nan

        if verbose:
            print(f"\n   ✅ CV Summary:")
            print(f"      Accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
            if not np.isnan(avg_logloss):
                print(f"      Log Loss: {avg_logloss:.3f}")
            print(f"      Expansion F1: {avg_f1_expansion:.3f}")
            print(f"      Expansion Recall: {avg_recall_expansion:.3f}")
            print(f"      Expansion Precision: {avg_precision_expansion:.3f}")

        return final_model, {
            "cv_metrics": cv_metrics,
            "mean_accuracy": avg_accuracy,
            "mean_log_loss": avg_logloss,
            "mean_f1_expansion": avg_f1_expansion,
            "mean_recall_expansion": avg_recall_expansion,
            "mean_precision_expansion": avg_precision_expansion,
            "global_optimal_threshold": global_optimal_threshold,
            "calibration_metadata": final_model._calibration_metadata,
        }

    def _validate_model_performance(self, horizon: int, verbose: bool):
        if horizon not in self.validation_results:
            return

        metrics = self.validation_results[horizon]
        mean_f1 = metrics.get("mean_f1_expansion", 0.0)
        mean_recall = metrics.get("mean_recall_expansion", 0.0)

        MIN_F1 = 0.15
        MIN_RECALL = 0.20

        if mean_f1 < MIN_F1:
            raise AssertionError(
                f"❌ CRITICAL: Model {horizon}d has F1={mean_f1:.3f} for expansion class "
                f"(threshold: {MIN_F1:.2f}). Model is not learning to predict expansions!"
            )

        if mean_recall < MIN_RECALL:
            raise AssertionError(
                f"❌ CRITICAL: Model {horizon}d has Recall={mean_recall:.3f} for expansion class "
                f"(threshold: {MIN_RECALL:.2f}). Model misses too many expansions!"
            )

        if verbose:
            print(
                f"\n   ✅ Validation PASSED: F1={mean_f1:.3f}, Recall={mean_recall:.3f}"
            )

    def _compute_shap_importance(self, horizon: int, X: pd.DataFrame, verbose: bool):
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"FEATURE IMPORTANCE (SHAP)")
            print(f"{'=' * 80}")

        try:
            model = self.models[horizon]

            if isinstance(model, CalibratedClassifierCV):
                base_model = model.calibrated_classifiers_[0].estimator
            else:
                base_model = model

            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": np.abs(shap_values).mean(axis=0)}
            ).sort_values("importance", ascending=False)

            self.feature_importance[horizon] = feature_importance
            self.shap_explainers[horizon] = explainer

            if verbose:
                print("   Using SHAP values\n")
                print("   Top 10 Features:")
                for idx, row in feature_importance.head(10).iterrows():
                    print(f"      {row['feature']:<45} {row['importance']:.3f}")

        except Exception as e:
            if verbose:
                print(f"   WARNING: SHAP failed ({e})")
                print("      Using gain-based importance instead")
            self._compute_gain_importance(horizon, verbose)

    def _compute_gain_importance(self, horizon: int, verbose: bool):
        model = self.models[horizon]

        if isinstance(model, CalibratedClassifierCV):
            base_model = model.calibrated_classifiers_[0].estimator
        else:
            base_model = model

        importance_dict = base_model.get_booster().get_score(importance_type="gain")

        feature_importance = pd.DataFrame(
            [
                {"feature": k, "importance": v}
                for k, v in sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )

        feature_importance["importance"] /= feature_importance["importance"].sum()

        self.feature_importance[horizon] = feature_importance

        if verbose:
            print("   Using gain-based importance\n")
            print("   Top 10 Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"      {row['feature']:<45} {row['importance']:.3f}")

    def _validate_multi_horizon(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        horizons: List[int],
        verbose: bool,
    ):
        print(f"\n{'=' * 80}")
        print(f"MULTI-HORIZON VALIDATION")
        print(f"{'=' * 80}")

        for horizon in horizons:
            X, y = self._prepare_data(features, vix, spx, horizon, verbose=False)
            model = self.models[horizon]

            y_pred = model.predict(X)
            accuracy = balanced_accuracy_score(y, y_pred)

            print(f"   {horizon}d: Accuracy = {accuracy:.3f}")

    def _validate_on_crises(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        horizon: int,
        verbose: bool,
    ):
        print(f"\n{'=' * 80}")
        print(f"CRISIS VALIDATION")
        print(f"{'=' * 80}")

        X, y = self._prepare_data(features, vix, spx, horizon, verbose=False)
        model = self.models[horizon]

        for crisis_name, (start, end) in CRISIS_PERIODS.items():
            crisis_mask = (X.index >= start) & (X.index <= end)
            if crisis_mask.sum() == 0:
                continue

            X_crisis = X[crisis_mask]
            y_crisis = y[crisis_mask]

            y_pred = model.predict(X_crisis)
            accuracy = balanced_accuracy_score(y_crisis, y_pred)

            print(f"   {crisis_name}: Accuracy = {accuracy:.3f}")

    def _calibrate_model(
        self,
        model,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = "isotonic",
        verbose: bool = True,
    ) -> CalibratedClassifierCV:
        if verbose:
            print(f"\n   📊 Calibrating probabilities using {method} method...")

        calibrated_model = CalibratedClassifierCV(
            estimator=model, method=method, cv="prefit"
        )
        calibrated_model.fit(X_cal, y_cal)

        uncalibrated_proba = model.predict_proba(X_cal)[:, 1]
        calibrated_proba = calibrated_model.predict_proba(X_cal)[:, 1]

        brier_before = brier_score_loss(y_cal, uncalibrated_proba)
        brier_after = brier_score_loss(y_cal, calibrated_proba)

        improvement_pct = (1 - brier_after / brier_before) * 100

        if verbose:
            print(
                f"      Brier Score: {brier_before:.4f} → {brier_after:.4f} (lower is better)"
            )
            print(f"      Calibration improvement: {improvement_pct:.1f}%")

        calibrated_model._calibration_metadata = {
            "method": method,
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
            "improvement_pct": float(improvement_pct),
            "calibration_size": len(X_cal),
        }

        return calibrated_model

    def _plot_calibration_curve(
        self,
        y_true,
        y_prob_uncal,
        y_prob_cal,
        horizon: int,
        output_dir: Path,
        verbose: bool = True,
    ):
        try:
            prob_true_uncal, prob_pred_uncal = calibration_curve(
                y_true, y_prob_uncal, n_bins=10, strategy="quantile"
            )
            prob_true_cal, prob_pred_cal = calibration_curve(
                y_true, y_prob_cal, n_bins=10, strategy="quantile"
            )

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")
            ax.plot(
                prob_pred_uncal,
                prob_true_uncal,
                "ro-",
                linewidth=2,
                markersize=8,
                label="Uncalibrated XGBoost",
            )
            ax.plot(
                prob_pred_cal,
                prob_true_cal,
                "go-",
                linewidth=2,
                markersize=8,
                label="Calibrated",
            )

            ax.set_xlabel("Predicted Probability", fontsize=12)
            ax.set_ylabel("Actual Frequency (Fraction of Positives)", fontsize=12)
            ax.set_title(
                f"Probability Calibration - {horizon}d VIX Expansion", fontsize=14
            )
            ax.legend(loc="upper left", fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            output_path = output_dir / f"calibration_curve_{horizon}d.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            if verbose:
                print(f"      📈 Calibration curve saved: {output_path}")

        except Exception as e:
            if verbose:
                print(f"      ⚠️  Could not generate calibration plot: {e}")

    def _get_calibration_split(
        self, X: pd.DataFrame, y: pd.Series, calibration_fraction: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        n_total = len(X)
        n_cal = int(n_total * calibration_fraction)
        n_train = n_total - n_cal

        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_cal = X.iloc[n_train:]
        y_cal = y.iloc[n_train:]

        return X_train, y_train, X_cal, y_cal

    def _save_models(self, verbose: bool):
        import pickle

        for horizon, model in self.models.items():
            if isinstance(model, CalibratedClassifierCV):
                model_path = self.output_dir / f"vix_expansion_{horizon}d.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                if verbose:
                    print(
                        f"   💾 Saved calibrated model: vix_expansion_{horizon}d.pkl (use pickle)"
                    )
            else:
                model_path = self.output_dir / f"vix_expansion_{horizon}d.json"
                model.save_model(model_path)

                if verbose:
                    print(
                        f"   💾 Saved XGBoost model: vix_expansion_{horizon}d.json (use xgb.XGBClassifier)"
                    )

        for horizon, importance_df in self.feature_importance.items():
            importance_path = self.output_dir / f"feature_importance_{horizon}d.csv"
            importance_df.to_csv(importance_path, index=False)

        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj

        metrics_path = self.output_dir / "validation_metrics_v2.json"
        with open(metrics_path, "w") as f:
            serializable_results = convert_to_json_serializable(self.validation_results)
            json.dump(serializable_results, f, indent=2)

        if verbose:
            print(f"\n✅ Models saved to {self.output_dir}")
            print(f"   Horizons: {self.trained_horizons}")


def train_vix_expansion_model(
    integrated_system,
    horizons: List[int] = [5],
    optimize_hyperparams: int = 0,
    expansion_threshold: float = 0.15,
    crisis_balanced: bool = True,
    compute_shap: bool = True,
    verbose: bool = True,
) -> VIXExpansionTrainer:
    if not integrated_system.trained:
        raise ValueError("Train integrated system first")

    trainer = VIXExpansionTrainer(expansion_threshold=expansion_threshold)

    trainer.train(
        features=integrated_system.orchestrator.features,
        vix=integrated_system.orchestrator.vix_ml,
        spx=integrated_system.orchestrator.spx_ml,
        horizons=horizons,
        n_splits=5,
        optimize_hyperparams=optimize_hyperparams,
        crisis_balanced=crisis_balanced,
        compute_shap=compute_shap,
        verbose=verbose,
    )

    return trainer
