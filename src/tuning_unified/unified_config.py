# UNIFIED CONFIG - DISTRIBUTION-CONSTRAINED CLASSIFIERS - 2025-12-21 10:19:59
# Test set: 72.8% (UP 74.8%, DOWN 71.2%)
# UP dist: P10=0.354, P90=0.503, Mean=0.408
# DOWN dist: P10=0.299, P90=0.571, Mean=0.458
# TRAINED WITH FROZEN FEATURES

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5051,
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
    'description': 'Distribution-constrained classifiers (P10<0.3, P90>0.7)'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 2, 'learning_rate': 0.0184,
    'n_estimators': 617, 'subsample': 0.8123,
    'colsample_bytree': 0.8096, 'colsample_bylevel': 0.9092,
    'min_child_weight': 6, 'reg_alpha': 0.1962,
    'reg_lambda': 1.5656, 'gamma': 0.6889,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 5, 'learning_rate': 0.0300,
    'n_estimators': 526, 'subsample': 0.8239,
    'colsample_bytree': 0.8092, 'colsample_bylevel': 0.7188,
    'min_child_weight': 11, 'reg_alpha': 2.8648,
    'reg_lambda': 6.4278, 'gamma': 0.5312,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0287,
    'n_estimators': 296, 'subsample': 0.9116,
    'colsample_bytree': 0.7903, 'min_child_weight': 12,
    'reg_alpha': 2.2524, 'reg_lambda': 6.2375,
    'gamma': 0.5445, 'scale_pos_weight': 0.8207,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 6, 'learning_rate': 0.0294,
    'n_estimators': 226, 'subsample': 0.6693,
    'colsample_bytree': 0.8029, 'min_child_weight': 16,
    'reg_alpha': 2.6949, 'reg_lambda': 2.3271,
    'gamma': 2.8683, 'scale_pos_weight': 0.6477,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': -0.0913,
    'confidence_weights': {
        'up': {'classifier': 0.5792, 'magnitude': 0.4208},
        'down': {'classifier': 0.7225, 'magnitude': 0.2775}
    },
    'magnitude_scaling': {
        'up': {'small': 3.3411, 'medium': 6.0734, 'large': 11.4815},
        'down': {'small': 4.1496, 'medium': 5.2822, 'large': 9.4231}
    },
    'decision_threshold': 0.6742,
    'description': 'Ternary decision with well-calibrated probability distributions'
}

# TEST PERFORMANCE:
# Overall: 72.8% | UP: 74.8% (127) | DOWN: 71.2% (160)
# NO_DECISION: 193 (40.2%)
# UP Prob: P10=0.354, P90=0.503, Mean=0.408
# DOWN Prob: P10=0.299, P90=0.571, Mean=0.458

# FROZEN FEATURES USED
# Expansion: 35 features
# Compression: 42 features
# UP: 30 features
# DOWN: 31 features
