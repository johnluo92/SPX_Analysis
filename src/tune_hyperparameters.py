#!/usr/bin/env python3
"""
Hyperparameter tuning for VIX forecasting models.
Uses Optuna with time-series CV to find optimal params for current feature set.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from config import TRAINING_END_DATE, TARGET_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.target_calculator import TargetCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, features_df, selected_features=None):
        """
        Args:
            features_df: DataFrame with all features
            selected_features: List of feature names to use (or None for all)
        """
        self.target_calculator = TargetCalculator()
        
        # Calculate targets
        logger.info("Calculating targets...")
        self.features_df = self.target_calculator.calculate_all_targets(features_df, vix_col="vix")
        
        # Prepare features
        exclude_cols = [
            "vix", "spx", "calendar_cohort", "cohort_weight", "feature_quality",
            "future_vix", "target_vix_pct_change", "target_log_vix_change", "target_direction"
        ]
        cohort_features = ["is_fomc_period", "is_opex_week", "is_earnings_heavy"]
        
        if selected_features is not None:
            feature_cols = [f for f in selected_features if f in self.features_df.columns and f not in exclude_cols]
        else:
            feature_cols = [c for c in self.features_df.columns if c not in exclude_cols]
        
        # Ensure cohort features included
        for cf in cohort_features:
            if cf not in self.features_df.columns:
                self.features_df[cf] = 0
        
        self.X = self.features_df[feature_cols].copy()
        self.y_direction = self.features_df["target_direction"].copy()
        self.y_magnitude = self.features_df["target_log_vix_change"].copy()
        
        # Remove invalid samples
        valid_mask = ~(self.X.isna().any(axis=1) | self.y_direction.isna() | self.y_magnitude.isna())
        self.X = self.X[valid_mask]
        self.y_direction = self.y_direction[valid_mask]
        self.y_magnitude = self.y_magnitude[valid_mask]
        
        logger.info(f"Prepared dataset: {len(self.X)} samples, {len(self.X.columns)} features")
        
        # Use time-series split for CV
        self.tscv = TimeSeriesSplit(n_splits=3)
        
    def objective_direction(self, trial):
        """Objective function for direction classifier"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 42,
            'n_jobs': -1
        }
        
        # Cross-validation scores
        cv_scores = []
        
        for train_idx, val_idx in self.tscv.split(self.X):
            X_train = self.X.iloc[train_idx]
            y_train = self.y_direction.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y_direction.iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Predict on validation
            y_pred = model.predict(X_val)
            
            # Use F1-like metric (harmonic mean of precision and recall)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            cv_scores.append(f1)
        
        return np.mean(cv_scores)
    
    def objective_magnitude(self, trial):
        """Objective function for magnitude regressor"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'n_jobs': -1
        }
        
        # Cross-validation scores
        cv_scores = []
        
        for train_idx, val_idx in self.tscv.split(self.X):
            X_train = self.X.iloc[train_idx]
            y_train = self.y_magnitude.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y_magnitude.iloc[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Predict on validation
            y_pred = model.predict(X_val)
            
            # Convert to percentage space for MAE
            val_pct_actual = (np.exp(y_val) - 1) * 100
            val_pct_pred = (np.exp(y_pred) - 1) * 100
            
            mae = mean_absolute_error(val_pct_actual, val_pct_pred)
            cv_scores.append(mae)
        
        # Return negative MAE (Optuna maximizes)
        return -np.mean(cv_scores)
    
    def tune(self, n_trials=100, output_dir="models"):
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of trials to run
            output_dir: Where to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER TUNING - DIRECTION CLASSIFIER")
        logger.info("="*80)
        
        # Tune direction model
        study_direction = optuna.create_study(
            direction='maximize',
            study_name='direction_classifier',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study_direction.optimize(self.objective_direction, n_trials=n_trials, show_progress_bar=True)
        
        logger.info("\nBest Direction Model Params:")
        for key, value in study_direction.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Best CV F1: {study_direction.best_value:.4f}")
        
        # Save direction results
        direction_results = {
            "best_params": study_direction.best_params,
            "best_value": study_direction.best_value,
            "n_trials": len(study_direction.trials)
        }
        
        with open(output_path / "tuning_direction.json", "w") as f:
            json.dump(direction_results, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER TUNING - MAGNITUDE REGRESSOR")
        logger.info("="*80)
        
        # Tune magnitude model
        study_magnitude = optuna.create_study(
            direction='maximize',  # We negate MAE, so maximize
            study_name='magnitude_regressor',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study_magnitude.optimize(self.objective_magnitude, n_trials=n_trials, show_progress_bar=True)
        
        logger.info("\nBest Magnitude Model Params:")
        for key, value in study_magnitude.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Best CV MAE: {-study_magnitude.best_value:.2f}%")
        
        # Save magnitude results
        magnitude_results = {
            "best_params": study_magnitude.best_params,
            "best_value": -study_magnitude.best_value,  # Convert back to MAE
            "n_trials": len(study_magnitude.trials)
        }
        
        with open(output_path / "tuning_magnitude.json", "w") as f:
            json.dump(magnitude_results, f, indent=2)
        
        # Save combined config for easy use
        combined_config = {
            "timestamp": datetime.now().isoformat(),
            "n_features": len(self.X.columns),
            "n_samples": len(self.X),
            "direction": direction_results,
            "magnitude": magnitude_results
        }
        
        with open(output_path / "tuning_results.json", "w") as f:
            json.dump(combined_config, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("TUNING COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: {output_path}/")
        logger.info(f"  - tuning_direction.json")
        logger.info(f"  - tuning_magnitude.json")
        logger.info(f"  - tuning_results.json")
        logger.info("\nTo use these params, update config.py XGBOOST_CONFIG['shared_params']")
        
        return study_direction, study_magnitude


def main():
    """Main tuning workflow"""
    logger.info("="*80)
    logger.info("VIX FORECASTING - HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    
    # 1. Build features
    logger.info("\n[1/3] Building feature set...")
    fetcher = UnifiedDataFetcher()
    engineer = FeatureEngineer(fetcher)
    result = engineer.build_complete_features(years=20, end_date=TRAINING_END_DATE)
    features_df = result["features"]
    
    logger.info(f"Built {len(features_df.columns)} features over {len(features_df)} days")
    
    # 2. Load selected features (if available)
    selected_features = None
    selection_file = Path("data_cache/selected_features.json")
    if selection_file.exists():
        with open(selection_file) as f:
            selected_features = json.load(f)
        logger.info(f"Using {len(selected_features)} selected features")
    else:
        logger.info("No feature selection file found, using all features")
    
    # 3. Run tuning
    logger.info("\n[2/3] Running hyperparameter optimization...")
    logger.info("This will take 10-30 minutes depending on n_trials...")
    
    tuner = HyperparameterTuner(features_df, selected_features=selected_features)
    study_dir, study_mag = tuner.tune(n_trials=50, output_dir="models")  # 50 trials = ~15 min
    
    logger.info("\n[3/3] DONE")
    logger.info("\nNext steps:")
    logger.info("1. Review tuning results in models/tuning_results.json")
    logger.info("2. Update config.py with best params")
    logger.info("3. Retrain models with: python train_probabilistic_models.py")


if __name__ == "__main__":
    main()
