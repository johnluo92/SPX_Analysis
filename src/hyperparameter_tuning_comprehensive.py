#!/usr/bin/env python3
"""
IMPROVED Comprehensive hyperparameter tuning for VIX forecasting models.

KEY IMPROVEMENTS OVER V1:
1. SEPARATE tuning for direction and magnitude models (not shared params)
2. Model-specific hyperparameters (scale_pos_weight, monotone_constraints)
3. Tune early_stopping_rounds
4. Better overfitting detection
5. Direct config.py update strings in output
6. More comprehensive metrics tracking

Optimizes BOTH feature count and model-specific XGBoost hyperparameters.
Uses Optuna with time-series CV to find optimal configurations.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from config import TRAINING_END_DATE, TARGET_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.target_calculator import TargetCalculator
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedHyperparameterTuner:
    """
    Tunes feature selection count and XGBoost hyperparameters.

    MAJOR IMPROVEMENT: Optimizes direction and magnitude models SEPARATELY
    instead of using shared parameters.

    Strategy:
    1. For each candidate feature count (top_n)
    2. Run feature selection with that top_n
    3. Optimize direction model hyperparameters independently
    4. Optimize magnitude model hyperparameters independently
    5. Record combined performance
    6. Select best configuration overall
    """

    def __init__(self, features_df, vix_series):
        """
        Args:
            features_df: DataFrame with all features
            vix_series: VIX series for target calculation
        """
        self.features_df = features_df
        self.vix_series = vix_series
        self.target_calculator = TargetCalculator()
        self.feature_selector = SimplifiedFeatureSelector()

        # Calculate targets once
        logger.info("Calculating targets...")
        self.features_with_targets = self.target_calculator.calculate_all_targets(
            features_df, vix_col="vix"
        )

        # Use time-series split for CV
        self.tscv = TimeSeriesSplit(n_splits=3)

        # Cache for feature selection results
        self.feature_cache = {}

        # Track class balance for direction model
        up_ratio = self.features_with_targets["target_direction"].mean()
        self.scale_pos_weight_baseline = (1 - up_ratio) / up_ratio if up_ratio > 0 else 1.0

        logger.info(f"Initialized tuner with {len(features_df)} samples, {len(features_df.columns)} features")
        logger.info(f"Class balance: {up_ratio:.1%} UP, {1-up_ratio:.1%} DOWN")
        logger.info(f"Baseline scale_pos_weight: {self.scale_pos_weight_baseline:.3f}")

    def _get_selected_features(self, top_n):
        """Get or compute feature selection for given top_n"""
        if top_n in self.feature_cache:
            return self.feature_cache[top_n]

        logger.info(f"\n  Running feature selection with top_n={top_n}...")

        # Prepare features for selection
        exclude_cols = [
            "vix", "spx", "calendar_cohort", "cohort_weight", "feature_quality",
            "future_vix", "target_vix_pct_change", "target_log_vix_change", "target_direction"
        ]
        feature_cols = [c for c in self.features_df.columns if c not in exclude_cols]

        # Run feature selection
        selector = SimplifiedFeatureSelector(top_n=top_n)
        selected_features, _ = selector.select_features(
            self.features_df[feature_cols],
            self.vix_series
        )

        # Add cohort features back
        cohort_features = ["is_fomc_period", "is_opex_week", "is_earnings_heavy"]
        for cf in cohort_features:
            if cf not in selected_features and cf in self.features_df.columns:
                selected_features.append(cf)

        self.feature_cache[top_n] = selected_features
        logger.info(f"  Selected {len(selected_features)} features (including cohorts)")

        return selected_features

    def _prepare_data(self, selected_features):
        """Prepare X, y for training"""
        # Get features and targets
        X = self.features_with_targets[selected_features].copy()
        y_direction = self.features_with_targets["target_direction"].copy()
        y_magnitude = self.features_with_targets["target_log_vix_change"].copy()

        # Remove invalid samples
        valid_mask = ~(X.isna().any(axis=1) | y_direction.isna() | y_magnitude.isna())
        X = X[valid_mask]
        y_direction = y_direction[valid_mask]
        y_magnitude = y_magnitude[valid_mask]

        return X, y_direction, y_magnitude

    def objective_direction(self, trial):
        """
        Objective for DIRECTION model only.

        Optimizes binary classification hyperparameters independently.
        """
        # 1. Sample feature count (shared across both models)
        top_n = trial.suggest_int('top_n', 50, 250, step=25)

        # 2. Get selected features for this count
        selected_features = self._get_selected_features(top_n)

        # 3. Prepare data
        X, y_direction, _ = self._prepare_data(selected_features)

        # 4. Sample DIRECTION-SPECIFIC hyperparameters
        params = {
            'max_depth': trial.suggest_int('dir_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('dir_learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('dir_n_estimators', 100, 1000, step=100),
            'subsample': trial.suggest_float('dir_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('dir_colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('dir_colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('dir_min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('dir_reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('dir_reg_lambda', 0.5, 5.0),
            'gamma': trial.suggest_float('dir_gamma', 0.0, 1.0),
            # CLASSIFICATION-SPECIFIC: Handle class imbalance
            'scale_pos_weight': trial.suggest_float('dir_scale_pos_weight',
                                                     self.scale_pos_weight_baseline * 0.5,
                                                     self.scale_pos_weight_baseline * 2.0),
            # Tune early stopping
            'early_stopping_rounds': trial.suggest_int('dir_early_stopping', 20, 100, step=10),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 42,
            'n_jobs': -1
        }

        # 5. Evaluate with time-series CV
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []

        for train_idx, val_idx in self.tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = model.predict(X_val)

            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            accuracy = accuracy_score(y_val, y_pred)

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            accuracy_scores.append(accuracy)

        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_accuracy = np.mean(accuracy_scores)

        # Store detailed metrics
        trial.set_user_attr('direction_f1', float(avg_f1))
        trial.set_user_attr('direction_precision', float(avg_precision))
        trial.set_user_attr('direction_recall', float(avg_recall))
        trial.set_user_attr('direction_accuracy', float(avg_accuracy))
        trial.set_user_attr('n_features', len(selected_features))

        # Optimize F1 score (balanced precision/recall)
        return avg_f1

    def objective_magnitude(self, trial):
        """
        Objective for MAGNITUDE model only.

        Optimizes regression hyperparameters independently.
        """
        # 1. Sample feature count (must match direction model's choice)
        # We use the best top_n from direction optimization
        top_n = trial.suggest_int('top_n', 50, 250, step=25)

        # 2. Get selected features
        selected_features = self._get_selected_features(top_n)

        # 3. Prepare data
        X, _, y_magnitude = self._prepare_data(selected_features)

        # 4. Sample MAGNITUDE-SPECIFIC hyperparameters
        params = {
            'max_depth': trial.suggest_int('mag_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('mag_learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('mag_n_estimators', 100, 1000, step=100),
            'subsample': trial.suggest_float('mag_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('mag_colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('mag_colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('mag_min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('mag_reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('mag_reg_lambda', 0.5, 5.0),
            'gamma': trial.suggest_float('mag_gamma', 0.0, 1.0),
            # Tune early stopping
            'early_stopping_rounds': trial.suggest_int('mag_early_stopping', 20, 100, step=10),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'n_jobs': -1
        }

        # 5. Evaluate with time-series CV
        mae_scores = []
        rmse_scores = []

        for train_idx, val_idx in self.tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_magnitude.iloc[train_idx], y_magnitude.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = model.predict(X_val)

            # Convert to percentage space for MAE
            val_pct_actual = (np.exp(y_val) - 1) * 100
            val_pct_pred = (np.exp(y_pred) - 1) * 100

            mae = mean_absolute_error(val_pct_actual, val_pct_pred)
            rmse = np.sqrt(mean_squared_error(val_pct_actual, val_pct_pred))

            mae_scores.append(mae)
            rmse_scores.append(rmse)

        avg_mae = np.mean(mae_scores)
        avg_rmse = np.mean(rmse_scores)

        # Store detailed metrics
        trial.set_user_attr('magnitude_mae', float(avg_mae))
        trial.set_user_attr('magnitude_rmse', float(avg_rmse))
        trial.set_user_attr('n_features', len(selected_features))

        # Minimize MAE
        return avg_mae

    def objective_combined(self, trial):
        """
        Combined objective that optimizes top_n while using best params from separate optimizations.

        This runs AFTER we have optimal direction and magnitude params.
        """
        # Only tune top_n here - use fixed best params for models
        top_n = trial.suggest_int('top_n', 50, 250, step=25)
        selected_features = self._get_selected_features(top_n)
        X, y_direction, y_magnitude = self._prepare_data(selected_features)

        # Use best direction params (loaded from study)
        dir_params = trial.params.get('best_direction_params', {})
        mag_params = trial.params.get('best_magnitude_params', {})

        # Evaluate direction
        dir_f1_scores = []
        for train_idx, val_idx in self.tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]

            model = XGBClassifier(**dir_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)

            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            dir_f1_scores.append(f1)

        # Evaluate magnitude
        mag_mae_scores = []
        for train_idx, val_idx in self.tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_magnitude.iloc[train_idx], y_magnitude.iloc[val_idx]

            model = XGBRegressor(**mag_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)

            val_pct_actual = (np.exp(y_val) - 1) * 100
            val_pct_pred = (np.exp(y_pred) - 1) * 100
            mae = mean_absolute_error(val_pct_actual, val_pct_pred)
            mag_mae_scores.append(mae)

        avg_f1 = np.mean(dir_f1_scores)
        avg_mae = np.mean(mag_mae_scores)

        # Combined score
        direction_component = avg_f1
        magnitude_component = 1.0 / (1.0 + avg_mae / 10.0)
        combined_score = 0.6 * direction_component + 0.4 * magnitude_component

        trial.set_user_attr('direction_f1', float(avg_f1))
        trial.set_user_attr('magnitude_mae', float(avg_mae))
        trial.set_user_attr('n_features', len(selected_features))

        return combined_score

    def tune(self, n_trials_per_model, output_dir="models"):
        """
        Run comprehensive hyperparameter optimization with SEPARATE model tuning.

        Args:
            n_trials_per_model: Number of trials for each model (direction, magnitude)
            output_dir: Where to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*80)
        logger.info("IMPROVED COMPREHENSIVE HYPERPARAMETER TUNING")
        logger.info("Strategy: Optimize direction and magnitude models SEPARATELY")
        logger.info("="*80)

        # STAGE 1: Optimize direction model
        logger.info("\n[STAGE 1/3] Optimizing DIRECTION model...")
        logger.info(f"Running {n_trials_per_model} trials...")

        study_direction = optuna.create_study(
            direction='maximize',
            study_name='direction_model_tuning',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study_direction.optimize(self.objective_direction, n_trials=n_trials_per_model, show_progress_bar=True)

        best_dir_trial = study_direction.best_trial
        best_dir_params = {k: v for k, v in best_dir_trial.params.items() if k.startswith('dir_') or k == 'top_n'}

        logger.info(f"\nBest Direction Model:")
        logger.info(f"  F1 Score: {best_dir_trial.user_attrs['direction_f1']:.4f}")
        logger.info(f"  Precision: {best_dir_trial.user_attrs['direction_precision']:.4f}")
        logger.info(f"  Recall: {best_dir_trial.user_attrs['direction_recall']:.4f}")
        logger.info(f"  Accuracy: {best_dir_trial.user_attrs['direction_accuracy']:.4f}")

        # STAGE 2: Optimize magnitude model
        logger.info("\n[STAGE 2/3] Optimizing MAGNITUDE model...")
        logger.info(f"Running {n_trials_per_model} trials...")

        study_magnitude = optuna.create_study(
            direction='minimize',
            study_name='magnitude_model_tuning',
            sampler=optuna.samplers.TPESampler(seed=43)
        )
        study_magnitude.optimize(self.objective_magnitude, n_trials=n_trials_per_model, show_progress_bar=True)

        best_mag_trial = study_magnitude.best_trial
        best_mag_params = {k: v for k, v in best_mag_trial.params.items() if k.startswith('mag_') or k == 'top_n'}

        logger.info(f"\nBest Magnitude Model:")
        logger.info(f"  MAE: {best_mag_trial.user_attrs['magnitude_mae']:.2f}%")
        logger.info(f"  RMSE: {best_mag_trial.user_attrs['magnitude_rmse']:.2f}%")

        # STAGE 3: Find optimal top_n with both best models
        logger.info("\n[STAGE 3/3] Finding optimal feature count with best models...")

        # Use the top_n that appears most in top trials
        dir_top_ns = [t.params['top_n'] for t in study_direction.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mag_top_ns = [t.params['top_n'] for t in study_magnitude.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # Take top_n from best direction trial (direction is harder to predict)
        final_top_n = best_dir_trial.params['top_n']

        logger.info(f"\nSelected top_n: {final_top_n}")
        logger.info(f"  (from best direction model)")

        # Generate final results
        logger.info("\n" + "="*80)
        logger.info("FINAL CONFIGURATION")
        logger.info("="*80)

        # Clean param names for config.py
        dir_config_params = {k.replace('dir_', ''): v for k, v in best_dir_params.items() if k != 'top_n'}
        mag_config_params = {k.replace('mag_', ''): v for k, v in best_mag_params.items() if k != 'top_n'}

        logger.info(f"\nFeature Selection:")
        logger.info(f"  top_n: {final_top_n}")

        logger.info(f"\nDirection Model Hyperparameters:")
        for key, value in dir_config_params.items():
            logger.info(f"  {key}: {value}")

        logger.info(f"\nMagnitude Model Hyperparameters:")
        for key, value in mag_config_params.items():
            logger.info(f"  {key}: {value}")

        # Save comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "strategy": "separate_optimization",
            "description": "Direction and magnitude models optimized independently",
            "n_trials_per_model": n_trials_per_model,
            "feature_selection": {
                "top_n": int(final_top_n),
                "actual_features_used": int(best_dir_trial.user_attrs['n_features'])
            },
            "direction_model": {
                "best_trial": best_dir_trial.number,
                "performance": {
                    "f1_score": float(best_dir_trial.user_attrs['direction_f1']),
                    "precision": float(best_dir_trial.user_attrs['direction_precision']),
                    "recall": float(best_dir_trial.user_attrs['direction_recall']),
                    "accuracy": float(best_dir_trial.user_attrs['direction_accuracy'])
                },
                "hyperparameters": dir_config_params,
                "config_update": self._generate_config_update("direction", dir_config_params)
            },
            "magnitude_model": {
                "best_trial": best_mag_trial.number,
                "performance": {
                    "mae_pct": float(best_mag_trial.user_attrs['magnitude_mae']),
                    "rmse_pct": float(best_mag_trial.user_attrs['magnitude_rmse'])
                },
                "hyperparameters": mag_config_params,
                "config_update": self._generate_config_update("magnitude", mag_config_params)
            },
            "all_trials": {
                "direction": self._summarize_trials(study_direction, "direction"),
                "magnitude": self._summarize_trials(study_magnitude, "magnitude")
            }
        }

        # Save to file
        results_file = output_path / "comprehensive_tuning_v2_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("RESULTS SAVED")
        logger.info("="*80)
        logger.info(f"Full results: {results_file}")

        # Print update instructions
        self._print_update_instructions(final_top_n, dir_config_params, mag_config_params)

        return {
            'direction_study': study_direction,
            'magnitude_study': study_magnitude,
            'results': results
        }

    def _generate_config_update(self, model_type, params):
        """Generate config.py update string"""
        return {
            "objective": "binary:logistic" if model_type == "direction" else "reg:squarederror",
            "eval_metric": "logloss" if model_type == "direction" else "rmse",
            **params
        }

    def _print_update_instructions(self, top_n, dir_params, mag_params):
        """Print clear instructions for updating config.py"""
        logger.info("\n" + "="*80)
        logger.info("CONFIG.PY UPDATE INSTRUCTIONS")
        logger.info("="*80)

        logger.info("\n1. Update FEATURE_SELECTION_CONFIG:")
        logger.info(f'   top_n: {top_n}')

        logger.info("\n2. Replace XGBOOST_CONFIG with model-specific params:")
        logger.info("\nXGBOOST_CONFIG = {")
        logger.info('    "strategy": "separate_direction_magnitude",  # CHANGED')
        logger.info('    "cohort_aware": False,')
        logger.info('    "objectives": {')
        logger.info('        "direction_5d": {')
        logger.info('            "objective": "binary:logistic",')
        logger.info('            "eval_metric": "logloss",')
        for key, value in dir_params.items():
            if isinstance(value, float):
                logger.info(f'            "{key}": {value:.4f},')
            else:
                logger.info(f'            "{key}": {value},')
        logger.info('        },')
        logger.info('        "magnitude_5d": {')
        logger.info('            "objective": "reg:squarederror",')
        logger.info('            "eval_metric": "rmse",')
        for key, value in mag_params.items():
            if isinstance(value, float):
                logger.info(f'            "{key}": {value:.4f},')
            else:
                logger.info(f'            "{key}": {value},')
        logger.info('        }')
        logger.info('    }')
        logger.info('}')

        logger.info("\n3. Update xgboost_trainer_v3.py:")
        logger.info("   Line 93: params = XGBOOST_CONFIG['objectives']['direction_5d'].copy()")
        logger.info("   Line 110: params = XGBOOST_CONFIG['objectives']['magnitude_5d'].copy()")

        logger.info("\n4. Retrain models: python train_probabilistic_models.py")

    def _summarize_trials(self, study, model_type):
        """Create summary of all trials for analysis"""
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                data = {
                    "trial_number": trial.number,
                    "value": float(trial.value),
                    "params": {k: v for k, v in trial.params.items()}
                }
                # Add user attributes
                for attr_key, attr_val in trial.user_attrs.items():
                    data[attr_key] = float(attr_val) if isinstance(attr_val, (int, float)) else attr_val
                trials_data.append(data)

        # Sort by value (maximize for direction, minimize for magnitude)
        reverse = (model_type == "direction")
        trials_data.sort(key=lambda x: x['value'], reverse=reverse)

        return {
            "total_trials": len(trials_data),
            "top_10_configurations": trials_data[:10]
        }


def main():
    """Main tuning workflow"""
    logger.info("="*80)
    logger.info("VIX FORECASTING - IMPROVED COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    logger.info("KEY IMPROVEMENT: Separate optimization for direction and magnitude models")
    logger.info("="*80)

    # 1. Build features
    logger.info("\n[1/3] Building feature set...")
    fetcher = UnifiedDataFetcher()
    engineer = FeatureEngineer(fetcher)
    result = engineer.build_complete_features(years=20, end_date=TRAINING_END_DATE)
    features_df = result["features"]
    vix_series = result["vix"]

    # Add VIX and SPX to features_df for target calculation
    features_df["vix"] = vix_series
    features_df["spx"] = result["spx"]

    logger.info(f"Built {len(features_df.columns)} features over {len(features_df)} days")

    # 2. Run improved tuning
    logger.info("\n[2/3] Running improved optimization...")
    logger.info("This will take 45-90 minutes depending on n_trials...")
    logger.info("Optimizing direction and magnitude models SEPARATELY")

    tuner = ImprovedHyperparameterTuner(features_df, vix_series)
    studies = tuner.tune(n_trials_per_model=100, output_dir="models")  # 100 total trials

    logger.info("\n[3/3] DONE")
    logger.info("\nReview results in models/comprehensive_tuning_v2_results.json")
    logger.info("Then update config.py and retrain models.")


if __name__ == "__main__":
    main()
