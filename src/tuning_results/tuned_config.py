# ENHANCED TUNED CONFIG WITH ENSEMBLE DIVERSITY
# Generated: 2025-11-27 14:39:53
# Trial #88 - Score: 16.1718
#
# PERFORMANCE METRICS:
# Magnitude MAE: 11.12% | Bias: +0.05%
# Direction Acc: 63.2% | F1: 0.4739
# Ensemble Conf: 61.9%
# Features: Mag=68, Dir=83, Common=39 (57.4%)
# Quality Filtered: 0.4%
#
# DIVERSITY METRICS:
# Feature Jaccard: 0.348
# Feature Overlap: 0.574
# Prediction Correlation: 0.485
# Overall Diversity: 0.532
# Diversity Penalty: -0.594

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 137,
    'max_depth': 3,
    'learning_rate': 0.0523,
    'subsample': 0.8200,
    'colsample_bytree': 0.8948
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 65,
    'direction_top_n': 80,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.9338,
    'target_overlap': 0.5471
}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {
    'enabled': True,
    'min_threshold': 0.6525,
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}

# Cohort Weights
CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2), 'weight': 1.3550},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': 1.3017},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': 1.2624},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0}
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {
        'magnitude': 0.3943,
        'direction': 0.3541,
        'agreement': 0.1558
    },
    'magnitude_thresholds': {
        'small': 2.0480,
        'medium': 5.5824,
        'large': 10.2614
    },
    'agreement_bonus': {
        'strong': 0.1253,
        'moderate': 0.0847,
        'weak': 0.0
    },
    'contradiction_penalty': {
        'severe': 0.3009,
        'moderate': 0.1249,
        'minor': 0.0618
    },
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}

# Magnitude Model Parameters (Regression-Optimized)
XGBOOST_CONFIG['magnitude_params'].update({
    'max_depth': 4,
    'learning_rate': 0.0847,
    'n_estimators': 439,
    'subsample': 0.9159,
    'colsample_bytree': 0.7031,
    'colsample_bylevel': 0.7252,
    'min_child_weight': 10,
    'reg_alpha': 1.0159,
    'reg_lambda': 2.9359,
    'gamma': 0.2440  # NEW: Tuned for noisy targets
})

# Direction Model Parameters (Classification-Optimized)
XGBOOST_CONFIG['direction_params'].update({
    'max_depth': 7,
    'learning_rate': 0.0261,
    'n_estimators': 489,
    'subsample': 0.7665,
    'colsample_bytree': 0.6678,
    'min_child_weight': 15,
    'reg_alpha': 1.5528,
    'reg_lambda': 3.7720,
    'gamma': 0.4013,
    'scale_pos_weight': 1.1662,
    'max_delta_step': 3  # NEW: For imbalanced classification
})

# Diversity Configuration (NEW)
DIVERSITY_CONFIG = {
    'enabled': True,
    'target_feature_jaccard': 0.40,
    'target_feature_overlap': 0.5471,
    'diversity_weight': 1.5000,
    'metrics': {
        'feature_jaccard': 0.348,
        'feature_overlap': 0.574,
        'pred_correlation': 0.485,
        'overall_diversity': 0.532
    }
}
