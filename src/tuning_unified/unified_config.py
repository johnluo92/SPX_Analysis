# UNIFIED CONFIG - TERNARY DECISION SYSTEM - 2025-12-19 10:35:05
# Direct UP/DOWN/NO_DECISION output based on single decision_threshold
# Simplified from 74 to 42 parameters (removed 12 complex threshold params)
# TRAINED WITH FROZEN FEATURES

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.6417,
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
    'description': 'Ternary system with balanced signal distribution and direction_bias tuning'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 3, 'learning_rate': 0.0687,
    'n_estimators': 805, 'subsample': 0.8212,
    'colsample_bytree': 0.7690, 'colsample_bylevel': 0.7013,
    'min_child_weight': 13, 'reg_alpha': 7.6960,
    'reg_lambda': 8.1959, 'gamma': 0.7496,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 5, 'learning_rate': 0.0409,
    'n_estimators': 325, 'subsample': 0.8589,
    'colsample_bytree': 0.7841, 'colsample_bylevel': 0.8395,
    'min_child_weight': 5, 'reg_alpha': 6.1738,
    'reg_lambda': 8.8805, 'gamma': 0.3002,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 3, 'learning_rate': 0.0163,
    'n_estimators': 219, 'subsample': 0.6786,
    'colsample_bytree': 0.8996, 'min_child_weight': 8,
    'reg_alpha': 4.7689, 'reg_lambda': 17.6359,
    'gamma': 1.5884, 'scale_pos_weight': 0.9115,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0426,
    'n_estimators': 348, 'subsample': 0.6902,
    'colsample_bytree': 0.8605, 'min_child_weight': 13,
    'reg_alpha': 5.4013, 'reg_lambda': 19.4028,
    'gamma': 1.6804, 'scale_pos_weight': 0.6828,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'direction_bias': -0.0282,
    'confidence_weights': {
        'up': {'classifier': 0.5500, 'magnitude': 0.4500},
        'down': {'classifier': 0.6843, 'magnitude': 0.3157}
    },
    'magnitude_scaling': {
        'up': {'small': 3.8182, 'medium': 5.0041, 'large': 9.8909},
        'down': {'small': 4.1024, 'medium': 5.7933, 'large': 9.9171}
    },
    'decision_threshold': 0.6714,
    'description': 'Ternary decision system with balanced 40-60% UP/DOWN target'
}

# DIRECTIONAL: 72.0% (UP 73.0%, DOWN 71.2%)
# NO_DECISION: 187 (39.0% of total)
# UP signals: 137 | DOWN signals: 156

# FROZEN FEATURES USED
# Expansion: 35 features
# Compression: 42 features
# UP: 30 features
# DOWN: 31 features
