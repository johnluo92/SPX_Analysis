# ENHANCED TUNED CONFIG WITH ENSEMBLE DIVERSITY
# Generated: 2025-11-27 15:18:37
# Trial #94 - Score: 16.8558
#
# PERFORMANCE METRICS:
# Magnitude MAE: 12.15% | Bias: +0.62%
# Direction Acc: 63.7% | F1: 0.6614
# Ensemble Conf: 60.8%
# Features: Mag=74, Dir=84, Common=45 (60.8%)
# Quality Filtered: 0.4%
#
# DIVERSITY METRICS:
# Feature Jaccard: 0.398
# Feature Overlap: 0.608
# Prediction Correlation: -0.002
# Overall Diversity: 0.647
# Diversity Penalty: -0.926

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 80,
    'max_depth': 5,
    'learning_rate': 0.0865,
    'subsample': 0.8907,
    'colsample_bytree': 0.8346
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 71,
    'direction_top_n': 81,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.8949,
    'target_overlap': 0.5485
}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {
    'enabled': True,
    'min_threshold': 0.6173,
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}

# Cohort Weights
CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2), 'weight': 1.1670},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': 1.3929},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': 1.0212},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0}
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {
        'magnitude': 0.4390,
        'direction': 0.3737,
        'agreement': 0.2862
    },
    'magnitude_thresholds': {
        'small': 1.6346,
        'medium': 4.9595,
        'large': 10.8076
    },
    'agreement_bonus': {
        'strong': 0.1313,
        'moderate': 0.1067,
        'weak': 0.0
    },
    'contradiction_penalty': {
        'severe': 0.2287,
        'moderate': 0.1708,
        'minor': 0.0643
    },
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}

# Magnitude Model Parameters (Regression-Optimized)
XGBOOST_CONFIG['magnitude_params'].update({
    'max_depth': 2,
    'learning_rate': 0.0635,
    'n_estimators': 583,
    'subsample': 0.8753,
    'colsample_bytree': 0.9377,
    'colsample_bylevel': 0.7689,
    'min_child_weight': 4,
    'reg_alpha': 1.2117,
    'reg_lambda': 5.5989,
    'gamma': 0.1546
})

# Direction Model Parameters (Classification-Optimized)
XGBOOST_CONFIG['direction_params'].update({
    'max_depth': 6,
    'learning_rate': 0.0529,
    'n_estimators': 333,
    'subsample': 0.7765,
    'colsample_bytree': 0.8859,
    'min_child_weight': 14,
    'reg_alpha': 3.4405,
    'reg_lambda': 3.6146,
    'gamma': 0.4620,
    'scale_pos_weight': 0.9274,
    'max_delta_step': 1
})

# Diversity Configuration (NEW)
DIVERSITY_CONFIG = {
    'enabled': True,
    'target_feature_jaccard': 0.40,
    'target_feature_overlap': 0.5485,
    'diversity_weight': 1.5000,
    'metrics': {
        'feature_jaccard': 0.398,
        'feature_overlap': 0.608,
        'pred_correlation': -0.002,
        'overall_diversity': 0.647
    }
}
