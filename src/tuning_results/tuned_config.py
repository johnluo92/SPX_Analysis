# COMPREHENSIVE TUNED CONFIG
# Generated: 2025-11-25 00:33:42
# Trial #53 - Score: 16.8042
#
# PERFORMANCE METRICS:
# Magnitude MAE: 10.95% | Bias: +0.08%
# Direction Acc: 65.1% | F1: 0.6117
# Ensemble Conf: 58.0%
# Features: Mag=64, Dir=85
# Quality Filtered: 0.4%

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 93,
    'max_depth': 4,
    'learning_rate': 0.0630,
    'subsample': 0.8680,
    'colsample_bytree': 0.8891
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 61,
    'direction_top_n': 82,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.8584
}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {
    'enabled': True,
    'min_threshold': 0.7028,
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}

# Cohort Weights
CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2), 'weight': 1.1303},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': 1.2019},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': 1.0353},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0}
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {
        'magnitude': 0.3725,
        'direction': 0.5146,
        'agreement': 0.2344
    },
    'magnitude_thresholds': {
        'small': 1.5105,
        'medium': 5.5140,
        'large': 14.1987
    },
    'agreement_bonus': {
        'strong': 0.1865,
        'moderate': 0.0846,
        'weak': 0.0
    },
    'contradiction_penalty': {
        'severe': 0.2095,
        'moderate': 0.1252,
        'minor': 0.0370
    },
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}

# Magnitude Model Parameters
MAGNITUDE_PARAMS={
    'max_depth': 5,
    'learning_rate': 0.0202,
    'n_estimators': 210,
    'subsample': 0.8073,
    'colsample_bytree': 0.9373,
    'colsample_bylevel': 0.8225,
    'min_child_weight': 5,
    'reg_alpha': 3.3510,
    'reg_lambda': 3.8305,
    'gamma': 0.5412
}

# Direction Model Parameters
DIRECTION_PARAMS={
    'max_depth': 4,
    'learning_rate': 0.0401,
    'n_estimators': 383,
    'subsample': 0.7931,
    'colsample_bytree': 0.8123,
    'min_child_weight': 9,
    'reg_alpha': 2.8558,
    'reg_lambda': 3.8780,
    'gamma': 0.5362,
    'scale_pos_weight': 1.3161
}
