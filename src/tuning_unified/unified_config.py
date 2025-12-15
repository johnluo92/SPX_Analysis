# UNIFIED CONFIG - 2025-12-15 00:27:56

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.6047,
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}

CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': 1.0, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': 1.0, 'description': 'Options expiration week + VIX futures rollover'},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': 1.0, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}

FEATURE_SELECTION_CV_PARAMS = {'n_estimators': 265,
    'max_depth': 6, 'learning_rate': 0.1311,
    'subsample': 0.8452, 'colsample_bytree': 0.7442,
    'n_jobs': 1, 'random_state': 42}

FEATURE_SELECTION_CONFIG = {'expansion_top_n': 76,
    'compression_top_n': 91, 'up_top_n': 97,
    'down_top_n': 128, 'cv_folds': 5, 'protected_features': [],
    'correlation_threshold': 0.9585,
    'description': 'Unified tuning with ensemble evaluation'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 6, 'learning_rate': 0.0524,
    'n_estimators': 406, 'subsample': 0.9060,
    'colsample_bytree': 0.8707, 'colsample_bylevel': 0.9485,
    'min_child_weight': 13, 'reg_alpha': 6.3692,
    'reg_lambda': 9.9966, 'gamma': 0.5903,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 7, 'learning_rate': 0.0328,
    'n_estimators': 750, 'subsample': 0.8689,
    'colsample_bytree': 0.8657, 'colsample_bylevel': 0.7930,
    'min_child_weight': 6, 'reg_alpha': 6.1072,
    'reg_lambda': 4.1598, 'gamma': 0.1893,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 4, 'learning_rate': 0.0492,
    'n_estimators': 347, 'subsample': 0.6782,
    'colsample_bytree': 0.9437, 'min_child_weight': 5,
    'reg_alpha': 6.0565, 'reg_lambda': 15.1840,
    'gamma': 1.7485, 'scale_pos_weight': 1.0063,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 7, 'learning_rate': 0.0484,
    'n_estimators': 404, 'subsample': 0.8578,
    'colsample_bytree': 0.7370, 'min_child_weight': 10,
    'reg_alpha': 3.4329, 'reg_lambda': 11.3462,
    'gamma': 1.3584, 'scale_pos_weight': 1.1525,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0898,
    'confidence_weights': {
        'up': {'classifier': 0.6899, 'magnitude': 0.3101},
        'down': {'classifier': 0.6786, 'magnitude': 0.3214}
    },
    'magnitude_scaling': {
        'up': {'small': 3.3474, 'medium': 5.8518, 'large': 10.3848},
        'down': {'small': 3.0771, 'medium': 7.0879, 'large': 12.5706}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6455,
            'medium_magnitude': 0.7421,
            'low_magnitude': 0.7457
        },
        'down': {
            'high_magnitude': 0.7038,
            'medium_magnitude': 0.7615,
            'low_magnitude': 0.8002
        }
    },
    'min_confidence_up': 0.6081,
    'min_confidence_down': 0.6817,
    'boost_threshold_up': 14.6153,
    'boost_threshold_down': 12.4955,
    'boost_amount_up': 0.0768,
    'boost_amount_down': 0.0730,
    'description': 'Unified tuning - base models + ensemble together'
}

# ACTIONABLE: 75.5% (UP 74.7%, DOWN 76.4%)
# Signals: 155 (32.3% actionable)
# UP signals: 83 | DOWN signals: 72
