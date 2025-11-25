# COMPREHENSIVE TUNED CONFIG
# Generated: 2025-11-24 23:42:20
# Trial #0 - Score: 999.0000
#
# PERFORMANCE METRICS:
# Magnitude MAE: 0.00% | Bias: +0.00%
# Direction Acc: 0.0% | F1: 0.0000
# Ensemble Conf: 0.0%
# Features: Mag=0, Dir=0
# Quality Filtered: 0.0%

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 125,
    'max_depth': 5,
    'learning_rate': 0.0724,
    'subsample': 0.8697,
    'colsample_bytree': 0.7812
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 69,
    'direction_top_n': 84,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.9626
}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {
    'enabled': True,
    'min_threshold': 0.7202,
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}

# Cohort Weights
CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2), 'weight': 1.3832},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': 1.1062},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': 1.2910},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0}
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {
        'magnitude': 0.4165,
        'direction': 0.3925,
        'agreement': 0.1773
    },
    'magnitude_thresholds': {
        'small': 1.7751,
        'medium': 4.9127,
        'large': 11.6733
    },
    'agreement_bonus': {
        'strong': 0.1432,
        'moderate': 0.0704,
        'weak': 0.0
    },
    'contradiction_penalty': {
        'severe': 0.2918,
        'moderate': 0.1139,
        'minor': 0.0446
    },
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}

# Magnitude Model Parameters
XGBOOST_CONFIG['magnitude_params'].update({
    'max_depth': 3,
    'learning_rate': 0.0286,
    'n_estimators': 514,
    'subsample': 0.7499,
    'colsample_bytree': 0.8286,
    'colsample_bylevel': 0.8481,
    'min_child_weight': 4,
    'reg_alpha': 2.7441,
    'reg_lambda': 2.6821,
    'gamma': 0.1793
})

# Direction Model Parameters
XGBOOST_CONFIG['direction_params'].update({
    'max_depth': 6,
    'learning_rate': 0.0763,
    'n_estimators': 524,
    'subsample': 0.7670,
    'colsample_bytree': 0.6744,
    'min_child_weight': 12,
    'reg_alpha': 2.1004,
    'reg_lambda': 2.3661,
    'gamma': 0.3981,
    'scale_pos_weight': 0.9172
})
