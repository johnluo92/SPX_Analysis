# TUNED CONFIGURATION - v2.3
# Copy these sections to your config.py, replacing existing definitions

QUALITY_FILTER_CONFIG = {
    'enabled': True,
    'min_threshold': 0.5647,
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}

CALENDAR_COHORTS = {
    'fomc_period': {
        'condition': 'macro_event_period',
        'range': (-7, 2),
        'weight': 1.2266,
        'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'
    },
    'opex_week': {
        'condition': 'days_to_monthly_opex',
        'range': (-7, 0),
        'weight': 1.4008,
        'description': 'Options expiration week + VIX futures rollover'
    },
    'earnings_heavy': {
        'condition': 'spx_earnings_pct',
        'range': (0.15, 1.0),
        'weight': 1.3661,
        'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'
    },
    'mid_cycle': {
        'condition': 'default',
        'range': None,
        'weight': 1.0,
        'description': 'Regular market conditions'
    }
}

FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 141,
    'max_depth': 6,
    'learning_rate': 0.0372,
    'subsample': 0.9229,
    'colsample_bytree': 0.8298
}

FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 87,
    'direction_top_n': 99,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.9136,
    'target_overlap': 0.5409,
    'description': 'Optimized via walk-forward CV with calibrated metrics (v2.3)'
}

MAGNITUDE_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 2,
    'learning_rate': 0.0292,
    'n_estimators': 348,
    'subsample': 0.8236,
    'colsample_bytree': 0.7590,
    'colsample_bylevel': 0.7446,
    'min_child_weight': 8,
    'reg_alpha': 0.9556,
    'reg_lambda': 8.1382,
    'gamma': 0.3737,
    'early_stopping_rounds': 50,
    'seed': 42,
    'n_jobs': -1
}

DIRECTION_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'learning_rate': 0.0248,
    'n_estimators': 809,
    'subsample': 0.6528,
    'colsample_bytree': 0.5953,
    'min_child_weight': 17,
    'reg_alpha': 1.9281,
    'reg_lambda': 3.5674,
    'gamma': 0.3692,
    'scale_pos_weight': 0.8491,
    'max_delta_step': 5,
    'early_stopping_rounds': 50,
    'seed': 42,
    'n_jobs': -1
}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {
        'magnitude': 0.2618,
        'direction': 0.5458,
        'agreement': 0.1924
    },
    'magnitude_thresholds': {
        'small': 2.2484,
        'medium': 4.5383,
        'large': 7.3915
    },
    'agreement_bonus': {
        'strong': 0.1632,
        'moderate': 0.1484,
        'weak': 0.0
    },
    'contradiction_penalty': {
        'severe': 0.3774,
        'moderate': 0.2433,
        'minor': 0.0026
    },
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.6817,
    'description': 'Ensemble combines magnitude + direction with agreement-based confidence'
}

DIVERSITY_CONFIG = {
    'enabled': True,
    'target_feature_jaccard': 0.40,
    'target_feature_overlap': 0.5409,
    'diversity_weight': 1.5000,
    'metrics': {
        'feature_jaccard': 0.396,
        'feature_overlap': 0.604,
        'pred_correlation': 0.724,
        'overall_diversity': 0.433
    },
    'description': 'Ensures complementary models without excessive overlap'
}
