import os
import random
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)

import numpy as np
np.random.seed(42)

import logging
import sys
from pathlib import Path
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/classifier_tuning.log")
    ]
)
logger = logging.getLogger(__name__)

class ClassifierTuner:
    """Optimize classifiers for 80%+ accuracy by reducing overfitting"""

    def __init__(self, df, features, target_col, name, train_end_date, val_end_date):
        self.df = df.sort_index()
        self.features = sorted(features)  # Ensure deterministic ordering
        self.target_col = target_col
        self.name = name
        self.train_end_date = train_end_date
        self.val_end_date = val_end_date

        # Split data
        train_end_idx = df[df.index <= pd.Timestamp(train_end_date)].index[-1]
        train_end_idx = df.index.get_loc(train_end_idx)
        val_end_idx = df[df.index <= pd.Timestamp(val_end_date)].index[-1]
        val_end_idx = df.index.get_loc(val_end_idx)

        self.train_df = df.iloc[:train_end_idx+1].sort_index()
        self.val_df = df.iloc[train_end_idx+1:val_end_idx+1].sort_index()
        self.test_df = df.iloc[val_end_idx+1:].sort_index()

        self.X_train = self.train_df[self.features]
        self.y_train = self.train_df[target_col]
        self.X_val = self.val_df[self.features]
        self.y_val = self.val_df[target_col]
        self.X_test = self.test_df[self.features]
        self.y_test = self.test_df[target_col]

        # Weights if available
        self.train_weights = self.train_df['cohort_weight'].values if 'cohort_weight' in self.train_df.columns else None
        self.val_weights = self.val_df['cohort_weight'].values if 'cohort_weight' in self.val_df.columns else None

        logger.info(f"\n{'='*80}")
        logger.info(f"{name} CLASSIFIER TUNING SETUP")
        logger.info(f"Train: {len(self.train_df)} samples | Val: {len(self.val_df)} samples | Test: {len(self.test_df)} samples")
        logger.info(f"Features: {len(self.features)}")
        logger.info(f"Train class balance: {self.y_train.mean():.2%} positive")
        logger.info(f"Val class balance: {self.y_val.mean():.2%} positive")

    def objective(self, trial):
        """Objective function optimized for validation accuracy"""

        # CONSTRAINED SEARCH SPACE TO REDUCE OVERFITTING
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',  # Better for probability calibration
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': 1,

            # Tree structure - CONSTRAIN depth to prevent overfitting
            'max_depth': trial.suggest_int('max_depth', 3, 6),  # Much lower than before (was 10-11)
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),  # Higher minimum

            # Boosting
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),

            # Sampling - prevent overfitting
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),  # Don't use 100% of data
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),  # Don't use all features
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.9),

            # Regularization - AGGRESSIVE
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 10.0),  # L1 - feature selection
            'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 15.0),  # L2 - weight regularization
            'gamma': trial.suggest_float('gamma', 0.5, 3.0),  # Min loss reduction

            # Class balance
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.7, 1.5),

            'early_stopping_rounds': 50
        }

        # Train model
        model = XGBClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            sample_weight=self.train_weights,
            eval_set=[(self.X_val, self.y_val)],
            sample_weight_eval_set=[self.val_weights] if self.val_weights is not None else None,
            verbose=False
        )

        # Predict on validation set
        y_val_pred = model.predict(self.X_val)

        # PRIMARY METRIC: Validation Accuracy
        val_acc = accuracy_score(self.y_val, y_val_pred)

        # SECONDARY METRICS for logging
        train_pred = model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_pred)
        val_f1 = f1_score(self.y_val, y_val_pred)

        # PENALTY for overfitting (train >> val)
        overfitting_penalty = max(0, (train_acc - val_acc - 0.10) * 0.5)  # Penalty if gap > 10%

        # Adjusted score
        score = val_acc - overfitting_penalty

        # Log progress
        trial.set_user_attr('train_acc', train_acc)
        trial.set_user_attr('val_acc', val_acc)
        trial.set_user_attr('val_f1', val_f1)
        trial.set_user_attr('overfitting_gap', train_acc - val_acc)

        return score  # Maximize validation accuracy with overfitting penalty

    def tune(self, n_trials=100):
        """Run optimization"""

        logger.info(f"\nStarting Optuna optimization for {self.name}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Target: 80%+ validation accuracy with minimal overfitting\n")

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.name}_classifier_tuning',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        # Best trial results
        best_trial = study.best_trial
        best_params = best_trial.params

        logger.info(f"\n{'='*80}")
        logger.info(f"{self.name} OPTIMIZATION COMPLETE")
        logger.info(f"Best validation accuracy: {best_trial.value:.4f}")
        logger.info(f"Train accuracy: {best_trial.user_attrs['train_acc']:.4f}")
        logger.info(f"Val accuracy: {best_trial.user_attrs['val_acc']:.4f}")
        logger.info(f"Val F1: {best_trial.user_attrs['val_f1']:.4f}")
        logger.info(f"Overfitting gap: {best_trial.user_attrs['overfitting_gap']:.4f}")

        # Train final model with best params
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': 1,
            'early_stopping_rounds': 50,
            **best_params
        }

        final_model = XGBClassifier(**final_params)
        final_model.fit(
            self.X_train, self.y_train,
            sample_weight=self.train_weights,
            eval_set=[(self.X_val, self.y_val)],
            sample_weight_eval_set=[self.val_weights] if self.val_weights is not None else None,
            verbose=False
        )

        # Evaluate on all splits
        train_pred = final_model.predict(self.X_train)
        val_pred = final_model.predict(self.X_val)
        test_pred = final_model.predict(self.X_test)

        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)

        test_prec = precision_score(self.y_test, test_pred, zero_division=0)
        test_rec = recall_score(self.y_test, test_pred, zero_division=0)
        test_f1 = f1_score(self.y_test, test_pred, zero_division=0)

        logger.info(f"\nFINAL MODEL PERFORMANCE:")
        logger.info(f"Train Acc: {train_acc:.1%}")
        logger.info(f"Val Acc: {val_acc:.1%}")
        logger.info(f"Test Acc: {test_acc:.1%} | Prec: {test_prec:.1%} | Rec: {test_rec:.1%} | F1: {test_f1:.1%}")

        results = {
            'name': self.name,
            'best_params': final_params,
            'metrics': {
                'train_acc': float(train_acc),
                'val_acc': float(val_acc),
                'test_acc': float(test_acc),
                'test_prec': float(test_prec),
                'test_rec': float(test_rec),
                'test_f1': float(test_f1)
            },
            'study': study
        }

        return results


def main():
    """Tune both UP and DOWN classifiers to achieve 80%+ accuracy"""

    logger.info("="*80)
    logger.info("CLASSIFIER TUNER - Target: 80%+ Validation & Test Accuracy")
    logger.info("="*80)

    # Import config (from src directory)
    from config import (
        DATA_SPLIT_CONFIG, TRAINING_YEARS,
        get_last_complete_month_end, FEATURE_SELECTION_CONFIG
    )
    from core.data_fetcher import UnifiedDataFetcher
    from core.feature_engineer import FeatureEngineer
    from core.target_calculator import TargetCalculator
    from core.xgboost_feature_selector_v2 import FeatureSelector

    # Prepare data
    training_end = get_last_complete_month_end()
    data_fetcher = UnifiedDataFetcher()
    feature_engineer = FeatureEngineer(data_fetcher)

    logger.info(f"Training through: {training_end}")
    result = feature_engineer.build_complete_features(years=TRAINING_YEARS, end_date=training_end)
    features_df = result["features"]
    vix = result["vix"]

    # Add targets
    target_calculator = TargetCalculator()
    complete_df = features_df.copy()
    complete_df["vix"] = vix
    complete_df = target_calculator.calculate_all_targets(complete_df, vix_col="vix")

    logger.info(f"Total samples: {len(complete_df)}")
    logger.info(f"UP samples: {complete_df['target_direction'].sum()}")
    logger.info(f"DOWN samples: {len(complete_df) - complete_df['target_direction'].sum()}")

    # Feature selection (using existing best features from config)
    feature_cols = [c for c in complete_df.columns if c not in [
        "vix", "spx", "calendar_cohort", "cohort_weight", "feature_quality",
        "future_vix", "target_vix_pct_change", "target_log_vix_change", "target_direction"
    ]]

    # Run feature selection for classifiers
    logger.info("\n" + "="*80)
    logger.info("FEATURE SELECTION FOR CLASSIFIERS")

    feature_selection_split_date = DATA_SPLIT_CONFIG["feature_selection_split_date"]
    split_date_idx = complete_df[complete_df.index <= pd.Timestamp(feature_selection_split_date)].index[-1]
    test_start_idx = complete_df.index.get_loc(split_date_idx) + 1

    # UP classifier features
    logger.info("\n📊 Selecting UP classifier features...")
    up_selector = FeatureSelector(
        target_type='up',
        top_n=FEATURE_SELECTION_CONFIG['up_top_n'],
        correlation_threshold=FEATURE_SELECTION_CONFIG.get('correlation_threshold', 0.90)
    )
    up_features, _ = up_selector.select_features(
        complete_df[feature_cols],
        vix,
        test_start_idx=test_start_idx
    )
    logger.info(f"UP features selected: {len(up_features)}")

    # DOWN classifier features
    logger.info("\n📊 Selecting DOWN classifier features...")
    down_selector = FeatureSelector(
        target_type='down',
        top_n=FEATURE_SELECTION_CONFIG['down_top_n'],
        correlation_threshold=FEATURE_SELECTION_CONFIG.get('correlation_threshold', 0.90)
    )
    down_features, _ = down_selector.select_features(
        complete_df[feature_cols],
        vix,
        test_start_idx=test_start_idx
    )
    logger.info(f"DOWN features selected: {len(down_features)}")

    # Tune UP classifier
    logger.info("\n" + "="*80)
    logger.info("TUNING UP CLASSIFIER")
    logger.info("="*80)

    up_tuner = ClassifierTuner(
        df=complete_df,
        features=up_features,
        target_col='target_direction',
        name='UP',
        train_end_date=DATA_SPLIT_CONFIG['train_end_date'],
        val_end_date=DATA_SPLIT_CONFIG['val_end_date']
    )

    up_results = up_tuner.tune(n_trials=100)

    # Tune DOWN classifier
    logger.info("\n" + "="*80)
    logger.info("TUNING DOWN CLASSIFIER")
    logger.info("="*80)

    # Create inverted target for DOWN classifier
    complete_df['target_direction_down'] = 1 - complete_df['target_direction']

    down_tuner = ClassifierTuner(
        df=complete_df,
        features=down_features,
        target_col='target_direction_down',
        name='DOWN',
        train_end_date=DATA_SPLIT_CONFIG['train_end_date'],
        val_end_date=DATA_SPLIT_CONFIG['val_end_date']
    )

    down_results = down_tuner.tune(n_trials=100)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TUNING COMPLETE - SUMMARY")
    logger.info("="*80)

    logger.info(f"\nUP CLASSIFIER:")
    logger.info(f"  Train: {up_results['metrics']['train_acc']:.1%}")
    logger.info(f"  Val: {up_results['metrics']['val_acc']:.1%}")
    logger.info(f"  Test: {up_results['metrics']['test_acc']:.1%} | F1: {up_results['metrics']['test_f1']:.1%}")
    logger.info(f"  Target: 80%+ ({'✓ PASS' if up_results['metrics']['val_acc'] >= 0.80 else '✗ BELOW TARGET'})")

    logger.info(f"\nDOWN CLASSIFIER:")
    logger.info(f"  Train: {down_results['metrics']['train_acc']:.1%}")
    logger.info(f"  Val: {down_results['metrics']['val_acc']:.1%}")
    logger.info(f"  Test: {down_results['metrics']['test_acc']:.1%} | F1: {down_results['metrics']['test_f1']:.1%}")
    logger.info(f"  Target: 80%+ ({'✓ PASS' if down_results['metrics']['val_acc'] >= 0.80 else '✗ BELOW TARGET'})")

    # Save updated config
    logger.info("\n" + "="*80)
    logger.info("SAVING OPTIMIZED PARAMETERS TO CONFIG")
    logger.info("="*80)

    logger.info("\nUP_CLASSIFIER_PARAMS = {")
    for key, value in up_results['best_params'].items():
        if isinstance(value, (int, float)):
            logger.info(f"    '{key}': {value},")
        else:
            logger.info(f"    '{key}': '{value}',")
    logger.info("}")

    logger.info("\nDOWN_CLASSIFIER_PARAMS = {")
    for key, value in down_results['best_params'].items():
        if isinstance(value, (int, float)):
            logger.info(f"    '{key}': {value},")
        else:
            logger.info(f"    '{key}': '{value}',")
    logger.info("}")

    logger.info("\n✅ Copy the above parameters to config.py and retrain!")

    return up_results, down_results


if __name__ == "__main__":
    main()
