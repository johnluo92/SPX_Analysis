#!/usr/bin/env python3
"""
ENHANCED COMPREHENSIVE HYPERPARAMETER TUNER
============================================
Major Improvements:
1. ENSEMBLE DIVERSITY METRICS - Addresses Priority #1 deficiency
2. Feature overlap penalties
3. Prediction diversity tracking
4. Additional tunable parameters (max_delta_step, differentiated ranges)
5. Enhanced robustness (NaN/inf handling, division by zero protection)
6. Better logging and diagnostics
"""

import argparse
import json
import logging
import sys
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor, XGBClassifier
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

from config import TRAINING_YEARS, XGBOOST_CONFIG, get_last_complete_month_end
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.target_calculator import TargetCalculator
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
import config as cfg

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class DiversityMetrics:
    """Utility class for computing ensemble diversity metrics"""

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """
        Compute Jaccard similarity between two sets
        J(A,B) = |A ∩ B| / |A ∪ B|
        Returns 0 for no overlap, 1 for complete overlap
        """
        if len(set1) == 0 and len(set2) == 0:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / max(union, 1)

    @staticmethod
    def feature_overlap_ratio(features1: List[str], features2: List[str]) -> float:
        """
        Compute feature overlap as percentage of smaller feature set
        """
        if not features1 or not features2:
            return 0.0
        set1, set2 = set(features1), set(features2)
        intersection = len(set1 & set2)
        min_size = min(len(set1), len(set2))
        return intersection / max(min_size, 1)

    @staticmethod
    def prediction_correlation(pred1: np.ndarray, pred2: np.ndarray) -> float:
        """
        Compute Spearman correlation between two prediction arrays
        Returns correlation coefficient (-1 to 1)
        Handles NaN/inf gracefully
        """
        # Remove any NaN or inf values
        mask = np.isfinite(pred1) & np.isfinite(pred2)
        if mask.sum() < 5:  # Need at least 5 valid pairs
            return 0.0

        pred1_clean = pred1[mask]
        pred2_clean = pred2[mask]

        try:
            corr, _ = spearmanr(pred1_clean, pred2_clean)
            if np.isnan(corr) or np.isinf(corr):
                return 0.0
            return float(corr)
        except Exception:
            return 0.0

    @staticmethod
    def compute_diversity_score(
        feature_jaccard: float,
        feature_overlap: float,
        pred_corr: float,
        target_jaccard: float = 0.4,
        target_overlap: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute diversity score with multiple components

        Lower feature overlap = higher diversity = better
        Lower prediction correlation (in absolute value) = higher diversity = better

        Returns dict with component scores and overall diversity
        """
        # Feature diversity (lower overlap = better, so invert)
        feature_div_jaccard = 1.0 - feature_jaccard
        feature_div_overlap = 1.0 - feature_overlap

        # Prediction diversity (lower absolute correlation = better)
        pred_diversity = 1.0 - abs(pred_corr)

        # Penalty for excessive feature overlap
        # Target ~40% Jaccard similarity, penalize deviation
        jaccard_penalty = abs(feature_jaccard - target_jaccard) * 2.0
        overlap_penalty = abs(feature_overlap - target_overlap) * 2.0

        # Overall diversity score (higher = more diverse)
        overall_diversity = (
            0.35 * feature_div_jaccard +
            0.35 * feature_div_overlap +
            0.30 * pred_diversity
        )

        return {
            'feature_div_jaccard': feature_div_jaccard,
            'feature_div_overlap': feature_div_overlap,
            'pred_diversity': pred_diversity,
            'jaccard_penalty': jaccard_penalty,
            'overlap_penalty': overlap_penalty,
            'overall_diversity': overall_diversity
        }


class EnhancedComprehensiveTuner:
    """
    Enhanced hyperparameter tuner with ensemble diversity optimization
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vix: pd.Series,
        n_trials: int = 100,
        output_dir: str = "tuning_results",
        diversity_weight: float = 1.5
    ):
        self.df = df
        self.vix = vix
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_calculator = TargetCalculator()
        self.diversity_weight = diversity_weight

        # Compute split indices
        total = len(df)
        test_size = XGBOOST_CONFIG["cv_config"]["test_size"]
        val_size = XGBOOST_CONFIG["cv_config"]["val_size"]

        self.train_end = int(total * (1 - test_size - val_size))
        self.val_end = int(total * (1 - test_size))
        self.test_start = self.val_end

        # Base columns (exclude targets and metadata)
        self.base_cols = [
            c for c in df.columns
            if c not in ["vix", "spx", "calendar_cohort", "cohort_weight", "feature_quality"]
        ]

        logger.info(f"Data splits: Train={self.train_end} | Val={self.val_end-self.train_end} | Test={total-self.val_end}")
        logger.info(f"Diversity weight: {self.diversity_weight:.2f}")

    def run_feature_selection(
        self,
        cv_params: Dict,
        target_type: str,
        top_n: int,
        correlation_threshold: float
    ) -> List[str]:
        """
        Run feature selection with given parameters
        Temporarily updates global config
        """
        # Backup original config
        original_cv = cfg.FEATURE_SELECTION_CV_PARAMS.copy()
        original_corr = cfg.FEATURE_SELECTION_CONFIG.get("correlation_threshold", 1.0)

        try:
            # Update config temporarily
            cfg.FEATURE_SELECTION_CV_PARAMS.update(cv_params)
            cfg.FEATURE_SELECTION_CONFIG["correlation_threshold"] = correlation_threshold

            # Run selection
            selector = SimplifiedFeatureSelector(
                target_type=target_type,
                top_n=top_n
            )
            selected, _ = selector.select_features(
                self.df[self.base_cols],
                self.vix,
                test_start_idx=self.test_start
            )

            return selected

        except Exception as e:
            logger.error(f"Feature selection error: {e}")
            return []
        finally:
            # Restore original config
            cfg.FEATURE_SELECTION_CV_PARAMS.update(original_cv)
            cfg.FEATURE_SELECTION_CONFIG["correlation_threshold"] = original_corr

    def compute_ensemble_confidence(
        self,
        magnitude_pct: float,
        direction_prob: float,
        mag_weight: float,
        dir_weight: float,
        agree_weight: float,
        thresholds: Dict,
        bonuses: Dict,
        penalties: Dict
    ) -> float:
        """
        Compute ensemble confidence with robust NaN/inf handling
        """
        # Validate inputs
        if not np.isfinite(magnitude_pct):
            magnitude_pct = 0.0
        if not np.isfinite(direction_prob):
            direction_prob = 0.5

        # Clip to valid ranges
        magnitude_pct = np.clip(magnitude_pct, -100, 100)
        direction_prob = np.clip(direction_prob, 0.0, 1.0)

        abs_mag = abs(magnitude_pct)

        # Categorize magnitude
        if abs_mag < thresholds["small"]:
            mag_category = "small"
        elif abs_mag < thresholds["medium"]:
            mag_category = "medium"
        else:
            mag_category = "large"

        # Magnitude confidence (0.5 to 1.0 scale)
        mag_conf = 0.5 + min(abs_mag / max(thresholds["large"], 1.0), 0.5) * 0.5

        # Direction confidence
        dir_conf = max(direction_prob, 1 - direction_prob)

        # Check agreement
        predicted_up = direction_prob > 0.5
        magnitude_up = magnitude_pct > 0
        models_agree = predicted_up == magnitude_up

        # Compute agreement score
        if models_agree:
            if abs_mag > thresholds["medium"] and dir_conf > 0.75:
                agreement_score = bonuses["strong"]
            elif abs_mag > thresholds["small"] and dir_conf > 0.65:
                agreement_score = bonuses["moderate"]
            else:
                agreement_score = bonuses["weak"]
        else:
            if abs_mag > thresholds["medium"] and dir_conf > 0.75:
                agreement_score = -penalties["severe"]
            elif abs_mag > thresholds["small"] and dir_conf > 0.65:
                agreement_score = -penalties["moderate"]
            else:
                agreement_score = -penalties["minor"]

        # Compute ensemble confidence
        ensemble_conf = (
            mag_weight * mag_conf +
            dir_weight * dir_conf +
            agree_weight * (0.5 + agreement_score)
        )

        # Clip to valid range
        ensemble_conf = np.clip(ensemble_conf, 0.5, 1.0)

        return float(ensemble_conf)

    def objective_complete(self, trial: optuna.Trial) -> float:
        """
        Enhanced objective function with diversity metrics
        """
        try:
            # ========================================
            # 1. SAMPLE HYPERPARAMETERS
            # ========================================

            # CV parameters (shared for feature selection)
            cv_params = {
                'n_estimators': trial.suggest_int('cv_n_est', 80, 200),
                'max_depth': trial.suggest_int('cv_depth', 3, 5),
                'learning_rate': trial.suggest_float('cv_lr', 0.03, 0.1, log=True),
                'subsample': trial.suggest_float('cv_sub', 0.75, 0.95),
                'colsample_bytree': trial.suggest_float('cv_col', 0.75, 0.95)
            }

            # Feature selection parameters
            mag_top_n = trial.suggest_int('mag_top_n', 60, 120)
            dir_top_n = trial.suggest_int('dir_top_n', 80, 150)
            corr_threshold = trial.suggest_float('corr_threshold', 0.85, 0.98)
            quality_threshold = trial.suggest_float('quality_threshold', 0.60, 0.80)

            # NEW: Diversity target parameter
            target_feature_overlap = trial.suggest_float('target_overlap', 0.35, 0.55)

            # Cohort weights
            cohort_weights = {
                'fomc_period': trial.suggest_float('cohort_fomc', 1.1, 1.5),
                'opex_week': trial.suggest_float('cohort_opex', 1.1, 1.4),
                'earnings_heavy': trial.suggest_float('cohort_earnings', 1.0, 1.3),
                'mid_cycle': 1.0
            }

            # Ensemble parameters
            ens_mag_weight = trial.suggest_float('ens_mag_weight', 0.25, 0.45)
            ens_dir_weight = trial.suggest_float('ens_dir_weight', 0.35, 0.55)
            ens_agree_weight = trial.suggest_float('ens_agree_weight', 0.15, 0.30)

            # Normalize ensemble weights
            weight_sum = ens_mag_weight + ens_dir_weight + ens_agree_weight
            ens_mag_weight /= weight_sum
            ens_dir_weight /= weight_sum
            ens_agree_weight /= weight_sum

            ensemble_params = {
                'mag_weight': ens_mag_weight,
                'dir_weight': ens_dir_weight,
                'agree_weight': ens_agree_weight,
                'thresholds': {
                    'small': trial.suggest_float('ens_small_thresh', 1.5, 3.0),
                    'medium': trial.suggest_float('ens_med_thresh', 4.0, 7.0),
                    'large': trial.suggest_float('ens_large_thresh', 8.0, 15.0)
                },
                'bonuses': {
                    'strong': trial.suggest_float('ens_bonus_strong', 0.10, 0.20),
                    'moderate': trial.suggest_float('ens_bonus_mod', 0.05, 0.12),
                    'weak': 0.0
                },
                'penalties': {
                    'severe': trial.suggest_float('ens_penalty_sev', 0.20, 0.35),
                    'moderate': trial.suggest_float('ens_penalty_mod', 0.10, 0.20),
                    'minor': trial.suggest_float('ens_penalty_min', 0.03, 0.08)
                }
            }

            # Magnitude model parameters (regression-optimized ranges)
            mag_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('mag_depth', 2, 4),  # Shallower for regression
                'learning_rate': trial.suggest_float('mag_lr', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('mag_n_est', 200, 600),
                'subsample': trial.suggest_float('mag_sub', 0.70, 0.95),
                'colsample_bytree': trial.suggest_float('mag_col_tree', 0.70, 0.95),
                'colsample_bylevel': trial.suggest_float('mag_col_lvl', 0.70, 0.95),
                'min_child_weight': trial.suggest_int('mag_mcw', 4, 10),  # Lower for regression
                'reg_alpha': trial.suggest_float('mag_alpha', 0.8, 4.0),
                'reg_lambda': trial.suggest_float('mag_lambda', 2.0, 6.0),
                'gamma': trial.suggest_float('mag_gamma', 0.0, 0.4),  # NEW: Added for noisy targets
                'early_stopping_rounds': 50,
                'seed': 42,
                'n_jobs': -1
            }

            # Direction model parameters (classification-optimized ranges)
            dir_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('dir_depth', 4, 8),  # Deeper for classification
                'learning_rate': trial.suggest_float('dir_lr', 0.02, 0.08, log=True),
                'n_estimators': trial.suggest_int('dir_n_est', 200, 600),
                'subsample': trial.suggest_float('dir_sub', 0.70, 0.92),
                'colsample_bytree': trial.suggest_float('dir_col_tree', 0.65, 0.90),
                'min_child_weight': trial.suggest_int('dir_mcw', 8, 15),  # Higher for classification
                'reg_alpha': trial.suggest_float('dir_alpha', 1.0, 3.5),
                'reg_lambda': trial.suggest_float('dir_lambda', 2.0, 5.0),
                'gamma': trial.suggest_float('dir_gamma', 0.2, 0.6),
                'scale_pos_weight': trial.suggest_float('dir_scale', 0.9, 1.4),
                'max_delta_step': trial.suggest_int('dir_max_delta', 0, 3),  # NEW: For imbalanced classification
                'early_stopping_rounds': 50,
                'seed': 42,
                'n_jobs': -1
            }

            # ========================================
            # 2. FEATURE SELECTION
            # ========================================

            mag_features = self.run_feature_selection(
                cv_params, 'magnitude', mag_top_n, corr_threshold
            )
            dir_features = self.run_feature_selection(
                cv_params, 'direction', dir_top_n, corr_threshold
            )

            if len(mag_features) < 20 or len(dir_features) < 20:
                logger.warning(f"Trial {trial.number}: Insufficient features selected")
                return 999.0

            # ========================================
            # 3. COMPUTE DIVERSITY METRICS
            # ========================================

            feature_jaccard = DiversityMetrics.jaccard_similarity(
                set(mag_features), set(dir_features)
            )
            feature_overlap = DiversityMetrics.feature_overlap_ratio(
                mag_features, dir_features
            )

            # ========================================
            # 4. PREPARE DATA
            # ========================================

            df_targets = self.target_calculator.calculate_all_targets(
                self.df.copy(), vix_col="vix"
            )

            # Apply quality filter
            if 'feature_quality' in df_targets.columns:
                quality_mask = df_targets['feature_quality'] >= quality_threshold
                df_filtered = df_targets[quality_mask].copy()

                if len(df_filtered) < len(df_targets) * 0.5:
                    logger.warning(f"Trial {trial.number}: Too much data filtered out")
                    return 999.0
            else:
                df_filtered = df_targets.copy()

            # Apply cohort weights
            df_filtered['cohort_weight'] = df_filtered['calendar_cohort'].map(
                cohort_weights
            ).fillna(1.0)

            # ========================================
            # 5. TRAIN MAGNITUDE MODEL
            # ========================================

            X_mag = df_filtered[mag_features].copy()
            y_mag = df_filtered["target_log_vix_change"].copy()

            # Remove invalid rows
            valid_mag = ~(X_mag.isna().any(axis=1) | y_mag.isna())
            X_mag = X_mag[valid_mag]
            y_mag = y_mag[valid_mag]
            weights_mag = df_filtered.loc[X_mag.index, 'cohort_weight'].values

            # Reset index
            X_mag = X_mag.reset_index(drop=True)
            y_mag = y_mag.reset_index(drop=True)

            # Split data
            train_end = int(len(X_mag) * 0.70)
            val_end = int(len(X_mag) * 0.85)

            X_mag_tr = X_mag.iloc[:train_end]
            y_mag_tr = y_mag.iloc[:train_end]
            w_mag_tr = weights_mag[:train_end]

            X_mag_val = X_mag.iloc[train_end:val_end]
            y_mag_val = y_mag.iloc[train_end:val_end]
            w_mag_val = weights_mag[train_end:val_end]

            X_mag_test = X_mag.iloc[val_end:]
            y_mag_test = y_mag.iloc[val_end:]

            if len(X_mag_val) < 20 or len(X_mag_test) < 20:
                logger.warning(f"Trial {trial.number}: Insufficient validation/test data")
                return 999.0

            # Train magnitude model
            mag_model = XGBRegressor(**mag_params)
            mag_model.fit(
                X_mag_tr, y_mag_tr,
                sample_weight=w_mag_tr,
                eval_set=[(X_mag_val, y_mag_val)],
                sample_weight_eval_set=[w_mag_val],
                verbose=False
            )

            # Predict and clip
            y_mag_pred_raw = mag_model.predict(X_mag_test)
            y_mag_pred = np.clip(y_mag_pred_raw, -2, 2)

            # Convert to percentage
            y_mag_test_np = y_mag_test.values
            test_pct_actual = (np.exp(y_mag_test_np) - 1) * 100
            test_pct_pred = (np.exp(y_mag_pred) - 1) * 100

            # Validate predictions
            if np.isnan(test_pct_pred).any() or np.isinf(test_pct_pred).any():
                logger.warning(f"Trial {trial.number}: Invalid magnitude predictions")
                return 999.0

            # Compute magnitude metrics
            mag_mae = np.mean(np.abs(test_pct_pred - test_pct_actual))
            mag_bias = np.mean(test_pct_pred - test_pct_actual)

            if np.isnan(mag_mae) or mag_mae > 20:
                logger.warning(f"Trial {trial.number}: Invalid magnitude MAE")
                return 999.0

            # ========================================
            # 6. TRAIN DIRECTION MODEL
            # ========================================

            X_dir = df_filtered[dir_features].copy()
            y_dir = df_filtered["target_direction"].copy()

            # Remove invalid rows
            valid_dir = ~(X_dir.isna().any(axis=1) | y_dir.isna())
            X_dir = X_dir[valid_dir]
            y_dir = y_dir[valid_dir]
            weights_dir = df_filtered.loc[X_dir.index, 'cohort_weight'].values

            # Reset index
            X_dir = X_dir.reset_index(drop=True)
            y_dir = y_dir.reset_index(drop=True)

            # Split data
            train_end = int(len(X_dir) * 0.70)
            val_end = int(len(X_dir) * 0.85)

            X_dir_tr = X_dir.iloc[:train_end]
            y_dir_tr = y_dir.iloc[:train_end]
            w_dir_tr = weights_dir[:train_end]

            X_dir_val = X_dir.iloc[train_end:val_end]
            y_dir_val = y_dir.iloc[train_end:val_end]
            w_dir_val = weights_dir[train_end:val_end]

            X_dir_test = X_dir.iloc[val_end:]
            y_dir_test = y_dir.iloc[val_end:]

            if len(X_dir_val) < 20 or len(X_dir_test) < 20:
                logger.warning(f"Trial {trial.number}: Insufficient direction data")
                return 999.0

            # Train direction model
            dir_model = XGBClassifier(**dir_params)
            dir_model.fit(
                X_dir_tr, y_dir_tr,
                sample_weight=w_dir_tr,
                eval_set=[(X_dir_val, y_dir_val)],
                sample_weight_eval_set=[w_dir_val],
                verbose=False
            )

            # Predict probabilities
            y_dir_proba_val = dir_model.predict_proba(X_dir_val)[:, 1]
            y_dir_proba_test = dir_model.predict_proba(X_dir_test)[:, 1]

            # Calibrate probabilities
            calibrator = IsotonicRegression(out_of_bounds='clip')
            y_dir_val_np = y_dir_val.values
            calibrator.fit(y_dir_proba_val, y_dir_val_np)
            y_dir_proba_calibrated = calibrator.transform(y_dir_proba_test)

            # Make predictions
            y_dir_pred_cal = (y_dir_proba_calibrated > 0.5).astype(int)
            y_dir_test_np = y_dir_test.values

            # Compute direction metrics
            dir_acc = (y_dir_pred_cal == y_dir_test_np).mean()

            # Precision and recall with zero division protection
            tp = np.sum((y_dir_pred_cal == 1) & (y_dir_test_np == 1))
            fp = np.sum((y_dir_pred_cal == 1) & (y_dir_test_np == 0))
            fn = np.sum((y_dir_pred_cal == 0) & (y_dir_test_np == 1))

            dir_prec = tp / max(tp + fp, 1)
            dir_rec = tp / max(tp + fn, 1)
            dir_f1 = 2 * (dir_prec * dir_rec) / max(dir_prec + dir_rec, 1e-8)

            if np.isnan(dir_f1):
                logger.warning(f"Trial {trial.number}: Invalid direction F1")
                return 999.0

            # ========================================
            # 7. COMPUTE PREDICTION DIVERSITY
            # ========================================

            # Convert direction probabilities to a scale similar to magnitude
            # Direction: 0.5 (down) to 1.0 (up) -> -1 (down) to +1 (up)
            dir_pred_scaled = (y_dir_proba_calibrated - 0.5) * 2.0

            # Magnitude predictions are already in percentage change
            # Scale to similar range for correlation: -1 to +1
            mag_pred_scaled = np.clip(test_pct_pred / 20.0, -1, 1)  # Normalize by typical range

            # Compute prediction correlation
            min_len = min(len(mag_pred_scaled), len(dir_pred_scaled))
            pred_correlation = DiversityMetrics.prediction_correlation(
                mag_pred_scaled[:min_len],
                dir_pred_scaled[:min_len]
            )

            # ========================================
            # 8. COMPUTE DIVERSITY SCORE
            # ========================================

            diversity_scores = DiversityMetrics.compute_diversity_score(
                feature_jaccard=feature_jaccard,
                feature_overlap=feature_overlap,
                pred_corr=pred_correlation,
                target_jaccard=0.40,
                target_overlap=target_feature_overlap
            )

            # ========================================
            # 9. COMPUTE ENSEMBLE CONFIDENCE
            # ========================================

            ensemble_confs = []
            min_len = min(len(test_pct_pred), len(y_dir_proba_calibrated))

            for i in range(min_len):
                mag_pct = test_pct_pred[i]
                dir_prob = y_dir_proba_calibrated[i]

                ens_conf = self.compute_ensemble_confidence(
                    mag_pct, dir_prob,
                    ensemble_params['mag_weight'],
                    ensemble_params['dir_weight'],
                    ensemble_params['agree_weight'],
                    ensemble_params['thresholds'],
                    ensemble_params['bonuses'],
                    ensemble_params['penalties']
                )
                ensemble_confs.append(ens_conf)

            avg_ensemble_conf = np.mean(ensemble_confs)

            # ========================================
            # 10. COMPUTE OBJECTIVE SCORE
            # ========================================

            # Base performance score
            base_score = (
                mag_mae +
                abs(mag_bias) * 0.3 +
                (1 - dir_acc) * 15 +
                abs(avg_ensemble_conf - 0.70) * 5
            )

            # Diversity penalty (penalize high overlap, reward diversity)
            diversity_penalty = (
                diversity_scores['jaccard_penalty'] * 3.0 +
                diversity_scores['overlap_penalty'] * 3.0 -
                diversity_scores['overall_diversity'] * 2.0  # Reward diversity
            )

            # Combined score (lower is better)
            combined_score = base_score + (diversity_penalty * self.diversity_weight)

            # ========================================
            # 11. STORE METRICS AS USER ATTRIBUTES
            # ========================================

            trial.set_user_attr('mag_mae', float(mag_mae))
            trial.set_user_attr('mag_bias', float(mag_bias))
            trial.set_user_attr('dir_acc', float(dir_acc))
            trial.set_user_attr('dir_f1', float(dir_f1))
            trial.set_user_attr('dir_precision', float(dir_prec))
            trial.set_user_attr('dir_recall', float(dir_rec))
            trial.set_user_attr('ensemble_conf', float(avg_ensemble_conf))
            trial.set_user_attr('n_mag_features', len(mag_features))
            trial.set_user_attr('n_dir_features', len(dir_features))
            trial.set_user_attr('quality_filtered_pct', float((1 - len(df_filtered) / len(df_targets)) * 100))

            # NEW: Diversity metrics
            trial.set_user_attr('feature_jaccard', float(feature_jaccard))
            trial.set_user_attr('feature_overlap', float(feature_overlap))
            trial.set_user_attr('pred_correlation', float(pred_correlation))
            trial.set_user_attr('diversity_overall', float(diversity_scores['overall_diversity']))
            trial.set_user_attr('diversity_penalty', float(diversity_penalty))
            trial.set_user_attr('base_score', float(base_score))

            # Common features for analysis
            common_features = set(mag_features) & set(dir_features)
            trial.set_user_attr('n_common_features', len(common_features))
            trial.set_user_attr('common_feature_pct', float(len(common_features) / max(len(mag_features), 1) * 100))

            return combined_score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            return 999.0

    def run_tuning(self):
        """Run hyperparameter optimization"""
        logger.info("=" * 80)
        logger.info(f"ENHANCED HYPERPARAMETER TUNING ({self.n_trials} trials)")
        logger.info("NEW: Ensemble diversity optimization enabled")
        logger.info("Tuning: CV params, feature selection, quality filter, cohort weights,")
        logger.info("        ensemble config, model params, diversity penalties")
        logger.info("=" * 80)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
        )

        study.optimize(
            self.objective_complete,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        return study

    def save_results(self, study: optuna.Study):
        """Save tuning results with diversity analysis"""
        best = study.best_trial
        attrs = best.user_attrs

        # Extract parameter groups
        cv_params = {k: v for k, v in best.params.items() if k.startswith('cv_')}
        mag_params = {
            k.replace('mag_', ''): v for k, v in best.params.items()
            if k.startswith('mag_') and k != 'mag_top_n'
        }
        dir_params = {
            k.replace('dir_', ''): v for k, v in best.params.items()
            if k.startswith('dir_') and k != 'dir_top_n'
        }
        ens_params = {
            k.replace('ens_', ''): v for k, v in best.params.items()
            if k.startswith('ens_')
        }

        results = {
            'timestamp': datetime.now().isoformat(),
            'trial_number': best.number,
            'diversity_weight': self.diversity_weight,
            'metrics': {
                'magnitude_mae': attrs.get('mag_mae', 0),
                'magnitude_bias': attrs.get('mag_bias', 0),
                'direction_accuracy': attrs.get('dir_acc', 0),
                'direction_f1': attrs.get('dir_f1', 0),
                'direction_precision': attrs.get('dir_precision', 0),
                'direction_recall': attrs.get('dir_recall', 0),
                'ensemble_confidence': attrs.get('ensemble_conf', 0),
                'n_magnitude_features': attrs.get('n_mag_features', 0),
                'n_direction_features': attrs.get('n_dir_features', 0),
                'n_common_features': attrs.get('n_common_features', 0),
                'common_feature_pct': attrs.get('common_feature_pct', 0),
                'quality_filtered_pct': attrs.get('quality_filtered_pct', 0),
                # NEW: Diversity metrics
                'feature_jaccard': attrs.get('feature_jaccard', 0),
                'feature_overlap': attrs.get('feature_overlap', 0),
                'pred_correlation': attrs.get('pred_correlation', 0),
                'diversity_overall': attrs.get('diversity_overall', 0),
                'diversity_penalty': attrs.get('diversity_penalty', 0),
                'base_score': attrs.get('base_score', 0)
            },
            'parameters': {
                'cv': cv_params,
                'magnitude': {
                    'top_n': best.params['mag_top_n'],
                    'model': mag_params
                },
                'direction': {
                    'top_n': best.params['dir_top_n'],
                    'model': dir_params
                },
                'ensemble': ens_params,
                'quality_threshold': best.params['quality_threshold'],
                'correlation_threshold': best.params['corr_threshold'],
                'target_overlap': best.params.get('target_overlap', 0.45),
                'cohort_weights': {
                    'fomc_period': best.params['cohort_fomc'],
                    'opex_week': best.params['cohort_opex'],
                    'earnings_heavy': best.params['cohort_earnings'],
                    'mid_cycle': 1.0
                }
            }
        }

        # Save JSON results
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Generate config file
        config = f"""# ENHANCED TUNED CONFIG WITH ENSEMBLE DIVERSITY
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Trial #{best.number} - Score: {best.value:.4f}
#
# PERFORMANCE METRICS:
# Magnitude MAE: {attrs.get('mag_mae', 0):.2f}% | Bias: {attrs.get('mag_bias', 0):+.2f}%
# Direction Acc: {attrs.get('dir_acc', 0):.1%} | F1: {attrs.get('dir_f1', 0):.4f}
# Ensemble Conf: {attrs.get('ensemble_conf', 0):.1%}
# Features: Mag={attrs.get('n_mag_features', 0)}, Dir={attrs.get('n_dir_features', 0)}, Common={attrs.get('n_common_features', 0)} ({attrs.get('common_feature_pct', 0):.1f}%)
# Quality Filtered: {attrs.get('quality_filtered_pct', 0):.1f}%
#
# DIVERSITY METRICS:
# Feature Jaccard: {attrs.get('feature_jaccard', 0):.3f}
# Feature Overlap: {attrs.get('feature_overlap', 0):.3f}
# Prediction Correlation: {attrs.get('pred_correlation', 0):.3f}
# Overall Diversity: {attrs.get('diversity_overall', 0):.3f}
# Diversity Penalty: {attrs.get('diversity_penalty', 0):.3f}

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {{
    'n_estimators': {best.params['cv_n_est']},
    'max_depth': {best.params['cv_depth']},
    'learning_rate': {best.params['cv_lr']:.4f},
    'subsample': {best.params['cv_sub']:.4f},
    'colsample_bytree': {best.params['cv_col']:.4f}
}}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {{
    'magnitude_top_n': {best.params['mag_top_n']},
    'direction_top_n': {best.params['dir_top_n']},
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': {best.params['corr_threshold']:.4f},
    'target_overlap': {best.params.get('target_overlap', 0.45):.4f}  # NEW: Diversity target
}}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {{
    'enabled': True,
    'min_threshold': {best.params['quality_threshold']:.4f},
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}}

# Cohort Weights
CALENDAR_COHORTS = {{
    'fomc_period': {{'condition': 'macro_event_period', 'range': (-7, 2), 'weight': {best.params['cohort_fomc']:.4f}}},
    'opex_week': {{'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': {best.params['cohort_opex']:.4f}}},
    'earnings_heavy': {{'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': {best.params['cohort_earnings']:.4f}}},
    'mid_cycle': {{'condition': 'default', 'range': None, 'weight': 1.0}}
}}

# Ensemble Configuration
ENSEMBLE_CONFIG = {{
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {{
        'magnitude': {ens_params['mag_weight']:.4f},
        'direction': {ens_params['dir_weight']:.4f},
        'agreement': {ens_params['agree_weight']:.4f}
    }},
    'magnitude_thresholds': {{
        'small': {ens_params['small_thresh']:.4f},
        'medium': {ens_params['med_thresh']:.4f},
        'large': {ens_params['large_thresh']:.4f}
    }},
    'agreement_bonus': {{
        'strong': {ens_params['bonus_strong']:.4f},
        'moderate': {ens_params['bonus_mod']:.4f},
        'weak': 0.0
    }},
    'contradiction_penalty': {{
        'severe': {ens_params['penalty_sev']:.4f},
        'moderate': {ens_params['penalty_mod']:.4f},
        'minor': {ens_params['penalty_min']:.4f}
    }},
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}}

# Magnitude Model Parameters (Regression-Optimized)
XGBOOST_CONFIG['magnitude_params'].update({{
    'max_depth': {mag_params['depth']},
    'learning_rate': {mag_params['lr']:.4f},
    'n_estimators': {mag_params['n_est']},
    'subsample': {mag_params['sub']:.4f},
    'colsample_bytree': {mag_params['col_tree']:.4f},
    'colsample_bylevel': {mag_params['col_lvl']:.4f},
    'min_child_weight': {mag_params['mcw']},
    'reg_alpha': {mag_params['alpha']:.4f},
    'reg_lambda': {mag_params['lambda']:.4f},
    'gamma': {mag_params['gamma']:.4f}  # NEW: Tuned for noisy targets
}})

# Direction Model Parameters (Classification-Optimized)
XGBOOST_CONFIG['direction_params'].update({{
    'max_depth': {dir_params['depth']},
    'learning_rate': {dir_params['lr']:.4f},
    'n_estimators': {dir_params['n_est']},
    'subsample': {dir_params['sub']:.4f},
    'colsample_bytree': {dir_params['col_tree']:.4f},
    'min_child_weight': {dir_params['mcw']},
    'reg_alpha': {dir_params['alpha']:.4f},
    'reg_lambda': {dir_params['lambda']:.4f},
    'gamma': {dir_params['gamma']:.4f},
    'scale_pos_weight': {dir_params['scale']:.4f},
    'max_delta_step': {dir_params.get('max_delta', 0)}  # NEW: For imbalanced classification
}})

# Diversity Configuration (NEW)
DIVERSITY_CONFIG = {{
    'enabled': True,
    'target_feature_jaccard': 0.40,
    'target_feature_overlap': {best.params.get('target_overlap', 0.45):.4f},
    'diversity_weight': {self.diversity_weight:.4f},
    'metrics': {{
        'feature_jaccard': {attrs.get('feature_jaccard', 0):.3f},
        'feature_overlap': {attrs.get('feature_overlap', 0):.3f},
        'pred_correlation': {attrs.get('pred_correlation', 0):.3f},
        'overall_diversity': {attrs.get('diversity_overall', 0):.3f}
    }}
}}
"""

        with open(self.output_dir / "tuned_config.py", 'w') as f:
            f.write(config)

        # Print summary
        logger.info(f"\n✅ Results: {self.output_dir}/results.json")
        logger.info(f"✅ Config: {self.output_dir}/tuned_config.py")
        logger.info(f"\n📊 BEST TRIAL #{best.number}")
        logger.info(f"   Magnitude MAE: {attrs.get('mag_mae', 0):.2f}% | Bias: {attrs.get('mag_bias', 0):+.2f}%")
        logger.info(f"   Direction Acc: {attrs.get('dir_acc', 0):.1%} | F1: {attrs.get('dir_f1', 0):.4f} | Prec: {attrs.get('dir_precision', 0):.1%} | Rec: {attrs.get('dir_recall', 0):.1%}")
        logger.info(f"   Ensemble Conf: {attrs.get('ensemble_conf', 0):.1%}")
        logger.info(f"   Features: Mag={attrs.get('n_mag_features', 0)}, Dir={attrs.get('n_dir_features', 0)}, Common={attrs.get('n_common_features', 0)} ({attrs.get('common_feature_pct', 0):.1f}%)")
        logger.info(f"\n🔀 DIVERSITY METRICS:")
        logger.info(f"   Feature Jaccard: {attrs.get('feature_jaccard', 0):.3f}")
        logger.info(f"   Feature Overlap: {attrs.get('feature_overlap', 0):.3f}")
        logger.info(f"   Prediction Correlation: {attrs.get('pred_correlation', 0):.3f}")
        logger.info(f"   Overall Diversity: {attrs.get('diversity_overall', 0):.3f}")
        logger.info(f"   Diversity Penalty: {attrs.get('diversity_penalty', 0):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="ENHANCED: Comprehensive hyperparameter tuning with ensemble diversity optimization"
    )
    parser.add_argument(
        '--trials', type=int, default=100,
        help="Number of Optuna trials (default: 100)"
    )
    parser.add_argument(
        '--output-dir', type=str, default='tuning_results',
        help="Output directory (default: tuning_results)"
    )
    parser.add_argument(
        '--diversity-weight', type=float, default=1.5,
        help="Weight for diversity penalty in objective (default: 1.5)"
    )
    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    training_end = get_last_complete_month_end()
    fetcher = UnifiedDataFetcher()
    engineer = FeatureEngineer(fetcher)
    result = engineer.build_complete_features(
        years=TRAINING_YEARS,
        end_date=training_end
    )

    df = result["features"].copy()
    df["vix"] = result["vix"]
    df["spx"] = result["spx"]

    logger.info(f"Dataset: {len(df)} samples, {len(df.columns)} columns\n")

    # Run tuning
    tuner = EnhancedComprehensiveTuner(
        df,
        result["vix"],
        n_trials=args.trials,
        output_dir=args.output_dir,
        diversity_weight=args.diversity_weight
    )

    study = tuner.run_tuning()
    tuner.save_results(study)

    logger.info("\n" + "=" * 80)
    logger.info("TUNING COMPLETE - Apply tuned_config.py to config.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
