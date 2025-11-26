# ENHANCED TUNED CONFIG WITH ENSEMBLE DIVERSITY
# Generated: 2025-11-26 12:26:23
# Trial #82 - Score: 17.1886
#
# PERFORMANCE METRICS:
# Magnitude MAE: 11.21% | Bias: +1.28%
# Direction Acc: 63.7% | F1: 0.4777
# Ensemble Conf: 59.0%
# Features: Mag=71, Dir=91, Common=45 (63.4%)
# Quality Filtered: 0.4%
#
# DIVERSITY METRICS:
# Feature Jaccard: 0.385
# Feature Overlap: 0.634
# Prediction Correlation: 0.478
# Overall Diversity: 0.500
# Diversity Penalty: -0.264

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 102,
    'max_depth': 3,
    'learning_rate': 0.0504,
    'subsample': 0.7612,
    'colsample_bytree': 0.7678
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 68,
    'direction_top_n': 88,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.8630,
    'target_overlap': 0.5266
}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {
    'enabled': True,
    'min_threshold': 0.6916,
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}

# Cohort Weights
CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2), 'weight': 1.3089},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': 1.3800},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': 1.0759},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0}
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {
        'magnitude': 0.2906,
        'direction': 0.4101,
        'agreement': 0.2740
    },
    'magnitude_thresholds': {
        'small': 1.5908,
        'medium': 4.4550,
        'large': 12.5307
    },
    'agreement_bonus': {
        'strong': 0.1341,
        'moderate': 0.0626,
        'weak': 0.0
    },
    'contradiction_penalty': {
        'severe': 0.2374,
        'moderate': 0.1654,
        'minor': 0.0419
    },
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}

# Magnitude Model Parameters (Regression-Optimized)
XGBOOST_CONFIG['magnitude_params'].update({
    'max_depth': 3,
    'learning_rate': 0.0189,
    'n_estimators': 485,
    'subsample': 0.7570,
    'colsample_bytree': 0.8536,
    'colsample_bylevel': 0.7561,
    'min_child_weight': 8,
    'reg_alpha': 1.0803,
    'reg_lambda': 3.0049,
    'gamma': 0.3370  # NEW: Tuned for noisy targets
})

# Direction Model Parameters (Classification-Optimized)
XGBOOST_CONFIG['direction_params'].update({
    'max_depth': 7,
    'learning_rate': 0.0246,
    'n_estimators': 475,
    'subsample': 0.7935,
    'colsample_bytree': 0.8729,
    'min_child_weight': 12,
    'reg_alpha': 1.9422,
    'reg_lambda': 2.4126,
    'gamma': 0.4799,
    'scale_pos_weight': 1.3267,
    'max_delta_step': 2  # NEW: For imbalanced classification
})

# Diversity Configuration (NEW)
DIVERSITY_CONFIG = {
    'enabled': True,
    'target_feature_jaccard': 0.40,
    'target_feature_overlap': 0.5266,
    'diversity_weight': 1.5000,
    'metrics': {
        'feature_jaccard': 0.385,
        'feature_overlap': 0.634,
        'pred_correlation': 0.478,
        'overall_diversity': 0.500
    }
}
