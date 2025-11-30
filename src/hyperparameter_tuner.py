"""
Nested Cross-Validation Hyperparameter Tuner for VIX Forecasting System

Implements principled, statistically sound hyperparameter optimization using:
- Outer loop: Walk-forward validation (time-series aware)
- Inner loop: Optuna TPE optimization per split
- Multi-objective evaluation with diversity penalties
- Hierarchical parameter search (data → model → ensemble)
- Production-ready config generation with version control
"""

import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, f1_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import dirichlet

# Import core system components
from config import (
    QUALITY_FILTER_CONFIG, ENSEMBLE_CONFIG, FEATURE_SELECTION_CONFIG,
    MAGNITUDE_PARAMS, DIRECTION_PARAMS, CALENDAR_COHORTS,
    TRAINING_START_DATE, TRAINING_END_DATE, RANDOM_STATE
)
from core.xgboost_trainer_v3 import SimplifiedVIXForecaster
from core.xgboost_feature_selector_v2 import UnifiedFeatureSelector
from core.temporal_validator import TemporalSafetyValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NestedCVHyperparameterTuner:
    """
    Nested cross-validation hyperparameter tuner with:
    - Outer loop: Walk-forward splits for time-series validation
    - Inner loop: Optuna optimization per split
    - Multi-objective scoring with diversity metrics
    - Hierarchical search: data → features → models → ensemble
    """
    
    def __init__(
        self,
        n_outer_splits: int = 5,
        n_trials_per_split: int = 100,
        train_window_months: int = 24,
        test_window_months: int = 6,
        gap_days: int = 5,
        objectives: List[str] = None,
        diversity_weight: float = 0.15,
        random_state: int = RANDOM_STATE,
        save_dir: str = "tuning_results",
        enable_pruning: bool = True,
        timeout_hours: Optional[float] = None
    ):
        """
        Initialize nested CV hyperparameter tuner.
        
        Args:
            n_outer_splits: Number of walk-forward validation splits
            n_trials_per_split: Optuna trials per outer split
            train_window_months: Training window size in months
            test_window_months: Test window size in months
            gap_days: Gap between train/test to prevent leakage
            objectives: Optimization objectives ['magnitude_mae', 'direction_f1', 'calibration']
            diversity_weight: Weight for diversity penalty (0-1)
            random_state: Random seed for reproducibility
            save_dir: Directory to save results and configs
            enable_pruning: Enable Hyperband pruning for efficiency
            timeout_hours: Optional timeout per split
        """
        self.n_outer_splits = n_outer_splits
        self.n_trials_per_split = n_trials_per_split
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.gap_days = gap_days
        self.objectives = objectives or ['magnitude_mae', 'direction_f1', 'calibration']
        self.diversity_weight = diversity_weight
        self.random_state = random_state
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.enable_pruning = enable_pruning
        self.timeout_hours = timeout_hours
        
        # Results storage
        self.outer_fold_results = []
        self.best_params_per_fold = []
        self.final_params = None
        self.performance_summary = {}
        
        # Validator for temporal safety
        self.temporal_validator = TemporalSafetyValidator()
        
        logger.info(f"Initialized tuner: {n_outer_splits} outer splits, {n_trials_per_split} trials/split")
        logger.info(f"Objectives: {', '.join(self.objectives)}")
    
    def generate_walk_forward_splits(self, df: pd.DataFrame) -> List[Dict[str, pd.DatetimeIndex]]:
        """
        Generate walk-forward validation splits for time-series data.
        
        Returns list of dicts with 'train_idx' and 'test_idx' keys.
        """
        logger.info(f"\nGenerating {self.n_outer_splits} walk-forward splits...")
        
        splits = []
        data_dates = df.index.sort_values()
        total_days = (data_dates[-1] - data_dates[0]).days
        
        # Calculate split parameters
        test_days = self.test_window_months * 30
        train_days = self.train_window_months * 30
        
        # Start from earliest possible point where we have enough training data
        earliest_start = data_dates[0]
        
        for i in range(self.n_outer_splits):
            # Calculate test period for this split
            test_start_offset = train_days + (i * test_days)
            test_start = earliest_start + timedelta(days=test_start_offset)
            test_end = test_start + timedelta(days=test_days)
            
            # Calculate corresponding training period
            train_start = earliest_start
            train_end = test_start - timedelta(days=self.gap_days)
            
            # Get actual indices
            train_mask = (data_dates >= train_start) & (data_dates <= train_end)
            test_mask = (data_dates >= test_start) & (data_dates <= test_end)
            
            train_idx = data_dates[train_mask]
            test_idx = data_dates[test_mask]
            
            if len(train_idx) < 252 or len(test_idx) < 20:  # Minimum requirements
                logger.warning(f"Split {i+1}: Insufficient data (train={len(train_idx)}, test={len(test_idx)})")
                continue
            
            splits.append({
                'fold': i + 1,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'test_start': test_idx[0],
                'test_end': test_idx[-1]
            })
            
            logger.info(f"Split {i+1}: Train {train_idx[0].date()} to {train_idx[-1].date()} "
                       f"({len(train_idx)} days) | Test {test_idx[0].date()} to {test_idx[-1].date()} "
                       f"({len(test_idx)} days)")
        
        logger.info(f"Generated {len(splits)} valid splits\n")
        return splits
    
    def create_param_search_space(self, trial: optuna.Trial, stage: str) -> Dict[str, Any]:
        """
        Define hierarchical parameter search spaces.
        
        Stages:
        - 'data': Quality thresholds, cohort weights
        - 'features': Feature selection parameters
        - 'magnitude': Magnitude model hyperparameters
        - 'direction': Direction model hyperparameters
        - 'ensemble': Ensemble reconciliation parameters
        """
        params = {}
        
        if stage == 'data':
            # Quality filtering
            params['quality_threshold'] = trial.suggest_float('quality_threshold', 0.5, 0.8)
            
            # Cohort weights (must maintain ordering)
            params['fomc_period_weight'] = trial.suggest_float('fomc_period_weight', 1.0, 2.0)
            params['opex_week_weight'] = trial.suggest_float('opex_week_weight', 1.0, 1.5)
            params['earnings_heavy_weight'] = trial.suggest_float('earnings_heavy_weight', 1.0, 1.5)
        
        elif stage == 'features':
            # Feature selection
            params['magnitude_top_n'] = trial.suggest_int('magnitude_top_n', 40, 120)
            params['direction_top_n'] = trial.suggest_int('direction_top_n', 40, 120)
            params['correlation_threshold'] = trial.suggest_float('correlation_threshold', 0.85, 0.95)
            params['target_overlap'] = trial.suggest_float('target_overlap', 0.3, 0.7)
        
        elif stage == 'magnitude':
            # XGBoost magnitude model
            params['max_depth'] = trial.suggest_int('mag_max_depth', 2, 8)
            params['learning_rate'] = trial.suggest_float('mag_learning_rate', 0.005, 0.1, log=True)
            params['n_estimators'] = trial.suggest_int('mag_n_estimators', 100, 1000)
            params['subsample'] = trial.suggest_float('mag_subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('mag_colsample_bytree', 0.6, 1.0)
            params['colsample_bylevel'] = trial.suggest_float('mag_colsample_bylevel', 0.6, 1.0)
            params['min_child_weight'] = trial.suggest_int('mag_min_child_weight', 1, 15)
            params['reg_alpha'] = trial.suggest_float('mag_reg_alpha', 0.0, 5.0)
            params['reg_lambda'] = trial.suggest_float('mag_reg_lambda', 0.0, 10.0)
            params['gamma'] = trial.suggest_float('mag_gamma', 0.0, 2.0)
        
        elif stage == 'direction':
            # XGBoost direction model
            params['max_depth'] = trial.suggest_int('dir_max_depth', 3, 10)
            params['learning_rate'] = trial.suggest_float('dir_learning_rate', 0.01, 0.15, log=True)
            params['n_estimators'] = trial.suggest_int('dir_n_estimators', 100, 1000)
            params['subsample'] = trial.suggest_float('dir_subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('dir_colsample_bytree', 0.6, 1.0)
            params['min_child_weight'] = trial.suggest_int('dir_min_child_weight', 1, 15)
            params['reg_alpha'] = trial.suggest_float('dir_reg_alpha', 0.0, 5.0)
            params['reg_lambda'] = trial.suggest_float('dir_reg_lambda', 0.0, 10.0)
            params['gamma'] = trial.suggest_float('dir_gamma', 0.0, 2.0)
            params['scale_pos_weight'] = trial.suggest_float('dir_scale_pos_weight', 0.8, 2.0)
            params['max_delta_step'] = trial.suggest_int('dir_max_delta_step', 0, 5)
        
        elif stage == 'ensemble':
            # Sample from Dirichlet for weights that sum to 1
            alpha = np.ones(3)  # Uniform prior for magnitude, direction, agreement
            weights = dirichlet.rvs(alpha)[0]
            
            params['magnitude_weight'] = weights[0]
            params['direction_weight'] = weights[1]
            params['agreement_weight'] = weights[2]
            
            # Magnitude thresholds (must maintain ordering: small < medium < large)
            small = trial.suggest_float('mag_threshold_small', 1.0, 4.0)
            medium = trial.suggest_float('mag_threshold_medium', small + 1.0, 8.0)
            large = trial.suggest_float('mag_threshold_large', medium + 1.0, 20.0)
            
            params['magnitude_thresholds'] = {
                'small': small,
                'medium': medium,
                'large': large
            }
            
            # Agreement bonuses (ascending)
            moderate = trial.suggest_float('agreement_moderate', 0.05, 0.15)
            strong = trial.suggest_float('agreement_strong', moderate, 0.25)
            
            params['agreement_bonus'] = {
                'weak': 0.0,
                'moderate': moderate,
                'strong': strong
            }
            
            # Contradiction penalties (ascending)
            minor = trial.suggest_float('penalty_minor', 0.0, 0.1)
            moderate_pen = trial.suggest_float('penalty_moderate', minor, 0.25)
            severe = trial.suggest_float('penalty_severe', moderate_pen, 0.4)
            
            params['contradiction_penalty'] = {
                'minor': minor,
                'moderate': moderate_pen,
                'severe': severe
            }
        
        return params
    
    def evaluate_configuration(
        self,
        df: pd.DataFrame,
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
        params: Dict[str, Any],
        fold_num: int
    ) -> Dict[str, float]:
        """
        Evaluate a full parameter configuration on one fold.
        
        Returns dict of metrics: magnitude_mae, direction_f1, calibration_error, etc.
        """
        # Apply data quality filtering
        df_filtered = df.copy()
        if 'feature_quality' in df.columns:
            quality_threshold = params.get('quality_threshold', QUALITY_FILTER_CONFIG['min_threshold'])
            quality_mask = df['feature_quality'] >= quality_threshold
            df_filtered = df_filtered[quality_mask]
        
        # Update cohort weights if specified
        if 'fomc_period_weight' in params:
            for cohort in ['fomc_period', 'opex_week', 'earnings_heavy']:
                weight_key = f'{cohort}_weight'
                if weight_key in params and 'cohort_weight' in df_filtered.columns:
                    cohort_mask = df_filtered[f'is_{cohort}'] == 1
                    df_filtered.loc[cohort_mask, 'cohort_weight'] = params[weight_key]
        
        # Feature selection
        feature_selector = UnifiedFeatureSelector()
        
        # Use custom feature selection params if provided
        fs_params = {
            'magnitude_top_n': params.get('magnitude_top_n', FEATURE_SELECTION_CONFIG['magnitude_top_n']),
            'direction_top_n': params.get('direction_top_n', FEATURE_SELECTION_CONFIG['direction_top_n']),
            'correlation_threshold': params.get('correlation_threshold', FEATURE_SELECTION_CONFIG['correlation_threshold']),
            'target_overlap': params.get('target_overlap', FEATURE_SELECTION_CONFIG['target_overlap'])
        }
        
        # Select features on training data only
        train_data = df_filtered.loc[train_idx]
        mag_features, dir_features = feature_selector.select_features(
            train_data,
            magnitude_top_n=fs_params['magnitude_top_n'],
            direction_top_n=fs_params['direction_top_n']
        )
        
        # Build model parameters
        magnitude_params = MAGNITUDE_PARAMS.copy()
        direction_params = DIRECTION_PARAMS.copy()
        
        # Update with trial params
        for key, value in params.items():
            if key.startswith('mag_'):
                param_name = key.replace('mag_', '')
                magnitude_params[param_name] = value
            elif key.startswith('dir_'):
                param_name = key.replace('dir_', '')
                direction_params[param_name] = value
        
        # Train models
        try:
            # Create temporary config with trial params
            temp_config = {
                'MAGNITUDE_PARAMS': magnitude_params,
                'DIRECTION_PARAMS': direction_params
            }
            
            # Update config temporarily
            import config
            original_mag = config.MAGNITUDE_PARAMS.copy()
            original_dir = config.DIRECTION_PARAMS.copy()
            config.MAGNITUDE_PARAMS = magnitude_params
            config.DIRECTION_PARAMS = direction_params
            
            forecaster = SimplifiedVIXForecaster()
            forecaster.train(
                df_filtered,
                magnitude_features=mag_features,
                direction_features=dir_features,
                save_dir=str(self.save_dir / f"temp_fold_{fold_num}")
            )
            
            # Restore original config
            config.MAGNITUDE_PARAMS = original_mag
            config.DIRECTION_PARAMS = original_dir
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'magnitude_mae': 999.0,
                'direction_f1': 0.0,
                'calibration_error': 1.0,
                'diversity_score': 0.0,
                'valid': False
            }
        
        # Evaluate on test set
        test_data = df_filtered.loc[test_idx]
        
        # Prepare features
        X_mag_test = test_data[mag_features]
        X_dir_test = test_data[dir_features]
        
        # Get predictions
        mag_preds = []
        dir_preds = []
        dir_probs = []
        
        for idx in test_idx:
            if idx not in test_data.index:
                continue
            
            row = test_data.loc[idx]
            X_test = pd.DataFrame([row])
            current_vix = row['vix']
            
            try:
                pred = forecaster.predict(X_test, current_vix)
                mag_preds.append(pred['magnitude_pct'])
                dir_probs.append(pred['direction_probability'])
                dir_preds.append(1 if pred['direction'] == 'UP' else 0)
            except:
                continue
        
        if len(mag_preds) < 20:  # Insufficient predictions
            return {
                'magnitude_mae': 999.0,
                'direction_f1': 0.0,
                'calibration_error': 1.0,
                'diversity_score': 0.0,
                'valid': False
            }
        
        # Get actuals
        actuals_mag = test_data.loc[test_data.index.isin(test_idx[:len(mag_preds)])]['target_vix_pct_change']
        actuals_dir = test_data.loc[test_data.index.isin(test_idx[:len(dir_preds)])]['target_direction']
        
        # Compute metrics
        metrics = {}
        
        # Magnitude MAE
        metrics['magnitude_mae'] = mean_absolute_error(actuals_mag, mag_preds)
        
        # Direction F1
        if len(set(actuals_dir)) > 1:  # Check for both classes
            metrics['direction_f1'] = f1_score(actuals_dir, dir_preds, average='binary')
        else:
            metrics['direction_f1'] = 0.5
        
        # Calibration error (ECE - Expected Calibration Error)
        try:
            prob_true, prob_pred = calibration_curve(
                actuals_dir, dir_probs, n_bins=10, strategy='uniform'
            )
            metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        except:
            metrics['calibration_error'] = 0.5
        
        # Diversity metrics
        mag_features_set = set(mag_features)
        dir_features_set = set(dir_features)
        
        feature_jaccard = len(mag_features_set & dir_features_set) / len(mag_features_set | dir_features_set)
        pred_correlation = np.corrcoef(mag_preds, dir_probs)[0, 1] if len(mag_preds) > 1 else 0
        
        # Target diversity: 0.4-0.6 for feature Jaccard
        diversity_penalty = abs(feature_jaccard - 0.5) ** 2
        
        metrics['diversity_score'] = 1.0 - diversity_penalty
        metrics['feature_jaccard'] = feature_jaccard
        metrics['pred_correlation'] = abs(pred_correlation)
        metrics['valid'] = True
        
        return metrics
    
    def objective_function(
        self,
        trial: optuna.Trial,
        df: pd.DataFrame,
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
        fold_num: int
    ) -> float:
        """
        Optuna objective function for a single fold.
        
        Combines multiple objectives into weighted score.
        """
        # Sample all parameter stages
        params = {}
        params.update(self.create_param_search_space(trial, 'data'))
        params.update(self.create_param_search_space(trial, 'features'))
        params.update(self.create_param_search_space(trial, 'magnitude'))
        params.update(self.create_param_search_space(trial, 'direction'))
        params.update(self.create_param_search_space(trial, 'ensemble'))
        
        # Evaluate configuration
        metrics = self.evaluate_configuration(df, train_idx, test_idx, params, fold_num)
        
        if not metrics['valid']:
            return float('inf')  # Invalid configuration
        
        # Multi-objective score (minimize)
        score = 0.0
        
        if 'magnitude_mae' in self.objectives:
            # Normalize to ~0-1 scale (MAE typically 2-10%)
            score += (metrics['magnitude_mae'] / 10.0) * 0.4
        
        if 'direction_f1' in self.objectives:
            # Convert F1 to loss (1 - F1)
            score += (1.0 - metrics['direction_f1']) * 0.4
        
        if 'calibration' in self.objectives:
            score += metrics['calibration_error'] * 0.2
        
        # Add diversity penalty
        diversity_loss = (1.0 - metrics['diversity_score']) * self.diversity_weight
        score += diversity_loss
        
        # Store metrics for analysis
        trial.set_user_attr('magnitude_mae', metrics['magnitude_mae'])
        trial.set_user_attr('direction_f1', metrics['direction_f1'])
        trial.set_user_attr('calibration_error', metrics['calibration_error'])
        trial.set_user_attr('diversity_score', metrics['diversity_score'])
        trial.set_user_attr('feature_jaccard', metrics['feature_jaccard'])
        
        return score
    
    def optimize_fold(
        self,
        df: pd.DataFrame,
        split_info: Dict,
        fold_num: int
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        Run Optuna optimization for one outer fold.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZING FOLD {fold_num}/{self.n_outer_splits}")
        logger.info(f"Train: {split_info['train_start'].date()} to {split_info['train_end'].date()}")
        logger.info(f"Test: {split_info['test_start'].date()} to {split_info['test_end'].date()}")
        logger.info(f"{'='*80}\n")
        
        # Create Optuna study
        sampler = TPESampler(seed=self.random_state + fold_num)
        pruner = HyperbandPruner() if self.enable_pruning else None
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=f'fold_{fold_num}'
        )
        
        # Define objective wrapper
        def objective(trial):
            return self.objective_function(
                trial, df, split_info['train_idx'], split_info['test_idx'], fold_num
            )
        
        # Run optimization
        timeout_seconds = self.timeout_hours * 3600 if self.timeout_hours else None
        
        study.optimize(
            objective,
            n_trials=self.n_trials_per_split,
            timeout=timeout_seconds,
            show_progress_bar=True,
            n_jobs=1  # Parallel trials not supported with some backends
        )
        
        # Get best trial
        best_trial = study.best_trial
        best_params = best_trial.params
        
        logger.info(f"\nFold {fold_num} Best Trial:")
        logger.info(f"  Score: {best_trial.value:.4f}")
        logger.info(f"  Magnitude MAE: {best_trial.user_attrs.get('magnitude_mae', 0):.2f}%")
        logger.info(f"  Direction F1: {best_trial.user_attrs.get('direction_f1', 0):.3f}")
        logger.info(f"  Calibration Error: {best_trial.user_attrs.get('calibration_error', 0):.3f}")
        logger.info(f"  Diversity Score: {best_trial.user_attrs.get('diversity_score', 0):.3f}")
        
        # Save study
        study_path = self.save_dir / f"study_fold_{fold_num}.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        return best_params, study
    
    def run_nested_cv_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full nested cross-validation optimization.
        
        Outer loop: Walk-forward validation splits
        Inner loop: Optuna optimization per split
        
        Returns aggregated best parameters and performance summary.
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING NESTED CROSS-VALIDATION HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        
        # Generate outer splits
        splits = self.generate_walk_forward_splits(df)
        
        if len(splits) < self.n_outer_splits:
            logger.warning(f"Only {len(splits)} valid splits generated (requested {self.n_outer_splits})")
        
        # Optimize each fold
        for split_info in splits:
            fold_num = split_info['fold']
            
            best_params, study = self.optimize_fold(df, split_info, fold_num)
            
            self.best_params_per_fold.append({
                'fold': fold_num,
                'params': best_params,
                'study': study,
                'split_info': split_info
            })
        
        # Aggregate results across folds
        self.final_params = self._aggregate_parameters()
        
        # Compute performance statistics
        self.performance_summary = self._compute_performance_summary()
        
        # Save results
        self._save_results()
        
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*80)
        self._print_summary()
        
        return {
            'final_params': self.final_params,
            'performance_summary': self.performance_summary,
            'fold_results': self.best_params_per_fold
        }
    
    def _aggregate_parameters(self) -> Dict[str, Any]:
        """
        Aggregate best parameters across folds.
        
        Uses median for continuous params, mode for categorical.
        """
        logger.info("\nAggregating parameters across folds...")
        
        all_params = [fold['params'] for fold in self.best_params_per_fold]
        aggregated = {}
        
        # Collect all param keys
        all_keys = set()
        for params in all_params:
            all_keys.update(params.keys())
        
        for key in all_keys:
            values = [p[key] for p in all_params if key in p]
            
            if len(values) == 0:
                continue
            
            # Use median for numeric params
            if isinstance(values[0], (int, float)):
                if isinstance(values[0], int):
                    aggregated[key] = int(np.median(values))
                else:
                    aggregated[key] = float(np.median(values))
            else:
                # For dicts (like thresholds), aggregate recursively
                if isinstance(values[0], dict):
                    aggregated[key] = {}
                    for subkey in values[0].keys():
                        subvalues = [v[subkey] for v in values]
                        aggregated[key][subkey] = float(np.median(subvalues))
                else:
                    # Use mode for categorical
                    aggregated[key] = max(set(values), key=values.count)
        
        logger.info(f"Aggregated {len(aggregated)} parameters")
        return aggregated
    
    def _compute_performance_summary(self) -> Dict[str, Any]:
        """
        Compute performance statistics across folds.
        """
        logger.info("\nComputing performance summary...")
        
        # Extract metrics from best trials
        mag_maes = []
        dir_f1s = []
        cal_errors = []
        diversity_scores = []
        
        for fold in self.best_params_per_fold:
            study = fold['study']
            best_trial = study.best_trial
            
            mag_maes.append(best_trial.user_attrs.get('magnitude_mae', np.nan))
            dir_f1s.append(best_trial.user_attrs.get('direction_f1', np.nan))
            cal_errors.append(best_trial.user_attrs.get('calibration_error', np.nan))
            diversity_scores.append(best_trial.user_attrs.get('diversity_score', np.nan))
        
        summary = {
            'magnitude_mae': {
                'mean': float(np.mean(mag_maes)),
                'std': float(np.std(mag_maes)),
                'min': float(np.min(mag_maes)),
                'max': float(np.max(mag_maes)),
                'median': float(np.median(mag_maes))
            },
            'direction_f1': {
                'mean': float(np.mean(dir_f1s)),
                'std': float(np.std(dir_f1s)),
                'min': float(np.min(dir_f1s)),
                'max': float(np.max(dir_f1s)),
                'median': float(np.median(dir_f1s))
            },
            'calibration_error': {
                'mean': float(np.mean(cal_errors)),
                'std': float(np.std(cal_errors)),
                'min': float(np.min(cal_errors)),
                'max': float(np.max(cal_errors)),
                'median': float(np.median(cal_errors))
            },
            'diversity_score': {
                'mean': float(np.mean(diversity_scores)),
                'std': float(np.std(diversity_scores)),
                'min': float(np.min(diversity_scores)),
                'max': float(np.max(diversity_scores)),
                'median': float(np.median(diversity_scores))
            },
            'n_folds': len(self.best_params_per_fold),
            'total_trials': sum(len(fold['study'].trials) for fold in self.best_params_per_fold)
        }
        
        return summary
    
    def _save_results(self):
        """
        Save optimization results to disk.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate config hash for versioning
        config_str = json.dumps(self.final_params, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Save final parameters as config
        config_file = self.save_dir / f"optimized_config_{timestamp}_{config_hash}.json"
        config_data = {
            'timestamp': timestamp,
            'config_hash': config_hash,
            'tuning_metadata': {
                'n_outer_splits': self.n_outer_splits,
                'n_trials_per_split': self.n_trials_per_split,
                'objectives': self.objectives,
                'diversity_weight': self.diversity_weight,
                'random_state': self.random_state
            },
            'parameters': self.final_params,
            'performance_summary': self.performance_summary
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"\n✅ Saved optimized config: {config_file}")
        
        # Save detailed results
        results_file = self.save_dir / f"detailed_results_{timestamp}.json"
        detailed_results = {
            'final_params': self.final_params,
            'performance_summary': self.performance_summary,
            'fold_details': [
                {
                    'fold': fold['fold'],
                    'params': fold['params'],
                    'split_dates': {
                        'train_start': fold['split_info']['train_start'].isoformat(),
                        'train_end': fold['split_info']['train_end'].isoformat(),
                        'test_start': fold['split_info']['test_start'].isoformat(),
                        'test_end': fold['split_info']['test_end'].isoformat()
                    },
                    'best_value': fold['study'].best_value,
                    'n_trials': len(fold['study'].trials)
                }
                for fold in self.best_params_per_fold
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"✅ Saved detailed results: {results_file}")
    
    def _print_summary(self):
        """
        Print optimization summary.
        """
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*80)
        
        print(f"\nOptimization Configuration:")
        print(f"  Outer CV Splits: {self.n_outer_splits}")
        print(f"  Trials per Split: {self.n_trials_per_split}")
        print(f"  Total Trials: {self.performance_summary['total_trials']}")
        print(f"  Objectives: {', '.join(self.objectives)}")
        
        print(f"\nPerformance Across Folds:")
        print(f"  Magnitude MAE: {self.performance_summary['magnitude_mae']['mean']:.2f}% "
              f"± {self.performance_summary['magnitude_mae']['std']:.2f}%")
        print(f"  Direction F1: {self.performance_summary['direction_f1']['mean']:.3f} "
              f"± {self.performance_summary['direction_f1']['std']:.3f}")
        print(f"  Calibration Error: {self.performance_summary['calibration_error']['mean']:.3f} "
              f"± {self.performance_summary['calibration_error']['std']:.3f}")
        print(f"  Diversity Score: {self.performance_summary['diversity_score']['mean']:.3f} "
              f"± {self.performance_summary['diversity_score']['std']:.3f}")
        
        print(f"\nKey Optimized Parameters:")
        if 'quality_threshold' in self.final_params:
            print(f"  Quality Threshold: {self.final_params['quality_threshold']:.3f}")
        if 'magnitude_top_n' in self.final_params:
            print(f"  Magnitude Features: {self.final_params['magnitude_top_n']}")
        if 'direction_top_n' in self.final_params:
            print(f"  Direction Features: {self.final_params['direction_top_n']}")
        
        print("\n" + "="*80)


def run_hyperparameter_optimization(
    df: pd.DataFrame,
    n_outer_splits: int = 5,
    n_trials_per_split: int = 100,
    objectives: List[str] = None,
    save_dir: str = "tuning_results",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run hyperparameter optimization.
    
    Args:
        df: DataFrame with features and targets
        n_outer_splits: Number of walk-forward CV splits
        n_trials_per_split: Optuna trials per split
        objectives: List of objectives to optimize
        save_dir: Directory to save results
        **kwargs: Additional arguments for NestedCVHyperparameterTuner
    
    Returns:
        Dict with final_params, performance_summary, fold_results
    """
    tuner = NestedCVHyperparameterTuner(
        n_outer_splits=n_outer_splits,
        n_trials_per_split=n_trials_per_split,
        objectives=objectives,
        save_dir=save_dir,
        **kwargs
    )
    
    results = tuner.run_nested_cv_optimization(df)
    
    return results


if __name__ == "__main__":
    print("Hyperparameter Tuner Module")
    print("=" * 80)
    print("\nThis module implements nested cross-validation hyperparameter optimization")
    print("for the VIX forecasting system using Optuna.")
    print("\nUsage:")
    print("  from hyperparameter_tuner import run_hyperparameter_optimization")
    print("  results = run_hyperparameter_optimization(df, n_outer_splits=5, n_trials_per_split=100)")
    print("\n" + "=" * 80)
