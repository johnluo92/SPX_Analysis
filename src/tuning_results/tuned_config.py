# ENHANCED TUNED CONFIG WITH ENSEMBLE DIVERSITY
# Generated: 2025-11-28 14:21:11
# Trial #92 - Score: 16.5502
#
# PERFORMANCE METRICS:
# Magnitude MAE: 12.17% | Bias: +0.44%
# Direction Acc: 65.1% | F1: 0.5458
# Ensemble Conf: 63.3%
# Features: Mag=81, Dir=83, Common=42 (51.9%)
# Quality Filtered: 0.4%
#
# DIVERSITY METRICS:
# Feature Jaccard: 0.344
# Feature Overlap: 0.519
# Prediction Correlation: 0.238
# Overall Diversity: 0.626
# Diversity Penalty: -0.884

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 139,
    'max_depth': 4,
    'learning_rate': 0.0714,
    'subsample': 0.7990,
    'colsample_bytree': 0.9004
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 78,
    'direction_top_n': 80,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.9157,
    'target_overlap': 0.5242
}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {
    'enabled': True,
    'min_threshold': 0.6142,
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}

# Cohort Weights
CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2), 'weight': 1.3333},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': 1.1350},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': 1.0464},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0}
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {
        'magnitude': 0.3710,
        'direction': 0.4064,
        'agreement': 0.1641
    },
    'magnitude_thresholds': {
        'small': 2.5314,
        'medium': 6.6030,
        'large': 12.2186
    },
    'agreement_bonus': {
        'strong': 0.1666,
        'moderate': 0.1147,
        'weak': 0.0
    },
    'contradiction_penalty': {
        'severe': 0.3042,
        'moderate': 0.1871,
        'minor': 0.0334
    },
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}

# Magnitude Model Parameters (Regression-Optimized)
XGBOOST_CONFIG['magnitude_params'].update({
    'max_depth': 3,
    'learning_rate': 0.0627,
    'n_estimators': 201,
    'subsample': 0.7605,
    'colsample_bytree': 0.8673,
    'colsample_bylevel': 0.7007,
    'min_child_weight': 8,
    'reg_alpha': 1.1951,
    'reg_lambda': 5.7459,
    'gamma': 0.1265
})

# Direction Model Parameters (Classification-Optimized)
XGBOOST_CONFIG['direction_params'].update({
    'max_depth': 8,
    'learning_rate': 0.0303,
    'n_estimators': 578,
    'subsample': 0.8532,
    'colsample_bytree': 0.6581,
    'min_child_weight': 15,
    'reg_alpha': 1.3042,
    'reg_lambda': 4.6933,
    'gamma': 0.5869,
    'scale_pos_weight': 1.2959,
    'max_delta_step': 3
})

# Diversity Configuration (NEW)
DIVERSITY_CONFIG = {
    'enabled': True,
    'target_feature_jaccard': 0.40,
    'target_feature_overlap': 0.5242,
    'diversity_weight': 1.5000,
    'metrics': {
        'feature_jaccard': 0.344,
        'feature_overlap': 0.519,
        'pred_correlation': 0.238,
        'overall_diversity': 0.626
    }
}
