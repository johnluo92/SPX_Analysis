# PHASE 1 OPTIMIZED CONFIG (RAW PREDICTIONS) - 2025-12-03 22:26:51

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.6286,
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}

CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': 1.2530, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': 1.4466, 'description': 'Options expiration week + VIX futures rollover'},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': 1.3473, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}

FEATURE_SELECTION_CV_PARAMS = {'n_estimators': 105,
    'max_depth': 6, 'learning_rate': 0.0639,
    'subsample': 0.8266, 'colsample_bytree': 0.7005}

FEATURE_SELECTION_CONFIG = {'expansion_top_n': 115,
    'compression_top_n': 116, 'up_top_n': 136,
    'down_top_n': 101, 'cv_folds': 5,
    'protected_features': [],
    'correlation_threshold': 0.8925,
    'description': 'Phase 1 optimized on RAW predictions (no ensemble filtering)'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 4, 'learning_rate': 0.0114,
    'n_estimators': 603, 'subsample': 0.7603,
    'colsample_bytree': 0.9497, 'colsample_bylevel': 0.8720,
    'min_child_weight': 13, 'reg_alpha': 3.0552,
    'reg_lambda': 7.9579, 'gamma': 0.0218,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 3, 'learning_rate': 0.0846,
    'n_estimators': 541, 'subsample': 0.7664,
    'colsample_bytree': 0.7703, 'colsample_bylevel': 0.9295,
    'min_child_weight': 11, 'reg_alpha': 7.0132,
    'reg_lambda': 3.6590, 'gamma': 0.5431,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 6, 'learning_rate': 0.1111,
    'n_estimators': 776, 'subsample': 0.7562,
    'colsample_bytree': 0.9223, 'min_child_weight': 13,
    'reg_alpha': 4.8640, 'reg_lambda': 9.2011,
    'gamma': 0.5018, 'scale_pos_weight': 1.0,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 8, 'learning_rate': 0.0883,
    'n_estimators': 766, 'subsample': 0.7406,
    'colsample_bytree': 0.8341, 'min_child_weight': 11,
    'reg_alpha': 2.6657, 'reg_lambda': 7.8673,
    'gamma': 0.1054, 'scale_pos_weight': 1.0,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

# TEST PERFORMANCE (RAW): 69.9% (UP 70.4%, DOWN 69.6%)
# MAE: 12.95% (UP: 11.17%, DOWN: 14.31%)
# Validation: Exp 8.80% Comp 4.94% UP 57.5% DOWN 59.3%
