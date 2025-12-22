# UNIFIED CONFIG - TERNARY DECISION SYSTEM - 2025-12-22 01:44:57
# Production-aligned: Uses actual predict() method during tuning
# Test set performance: 73.0% (UP 75.5%, DOWN 71.1%)
# TRAINED WITH FROZEN FEATURES

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5504,
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}

CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': 1.0, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': 1.0, 'description': 'Options expiration week + VIX futures rollover'},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': 1.0, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}

FEATURE_SELECTION_CV_PARAMS = {'n_estimators': 200,
    'max_depth': 4, 'learning_rate': 0.0500,
    'subsample': 0.8500, 'colsample_bytree': 0.8500,
    'n_jobs': 1, 'random_state': 42}

FEATURE_SELECTION_CONFIG = {'expansion_top_n': 35,
    'compression_top_n': 42, 'up_top_n': 30,
    'down_top_n': 31, 'cv_folds': 5, 'protected_features': [],
    'correlation_threshold': 0.9000,
    'description': 'Ternary system with balanced 40-60% UP/DOWN target'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 3, 'learning_rate': 0.0324,
    'n_estimators': 333, 'subsample': 0.7978,
    'colsample_bytree': 0.7505, 'colsample_bylevel': 0.7693,
    'min_child_weight': 10, 'reg_alpha': 7.0565,
    'reg_lambda': 9.2889, 'gamma': 0.7995,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 6, 'learning_rate': 0.0485,
    'n_estimators': 814, 'subsample': 0.8100,
    'colsample_bytree': 0.8817, 'colsample_bylevel': 0.7893,
    'min_child_weight': 7, 'reg_alpha': 1.1534,
    'reg_lambda': 5.0506, 'gamma': 0.4110,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0407,
    'n_estimators': 222, 'subsample': 0.8815,
    'colsample_bytree': 0.9494, 'min_child_weight': 15,
    'reg_alpha': 5.2808, 'reg_lambda': 7.5008,
    'gamma': 1.1478, 'scale_pos_weight': 0.7001,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0538,
    'n_estimators': 248, 'subsample': 0.7543,
    'colsample_bytree': 0.7146, 'min_child_weight': 13,
    'reg_alpha': 5.7437, 'reg_lambda': 16.2084,
    'gamma': 1.8454, 'scale_pos_weight': 0.5860,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': -0.0452,
    'confidence_weights': {
        'up': {'classifier': 0.6011, 'magnitude': 0.3989},
        'down': {'classifier': 0.6624, 'magnitude': 0.3376}
    },
    'magnitude_scaling': {
        'up': {'small': 3.0879, 'medium': 6.0725, 'large': 9.9973},
        'down': {'small': 3.0478, 'medium': 6.9760, 'large': 9.2031}
    },
    'decision_threshold': 0.6840,
    'description': 'Ternary decision system optimized on test set using production logic'
}

# FROZEN FEATURES USED
# Expansion: 35 features
# Compression: 42 features
# UP: 30 features
# DOWN: 31 features
