# PHASE 1 OPTIMIZED CONFIG (RAW PREDICTIONS) - 2025-12-05 15:02:06

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5750,
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}

CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': 1.3272, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': 1.1126, 'description': 'Options expiration week + VIX futures rollover'},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': 1.3391, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}

FEATURE_SELECTION_CV_PARAMS = {'n_estimators': 100,
    'max_depth': 4, 'learning_rate': 0.0443,
    'subsample': 0.8943, 'colsample_bytree': 0.9286}

FEATURE_SELECTION_CONFIG = {'expansion_top_n': 73,
    'compression_top_n': 94, 'up_top_n': 96,
    'down_top_n': 99, 'cv_folds': 5,
    'protected_features': [],
    'correlation_threshold': 0.9246,
    'description': 'Phase 1 optimized on RAW predictions (no ensemble filtering)'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 3, 'learning_rate': 0.0205,
    'n_estimators': 709, 'subsample': 0.8139,
    'colsample_bytree': 0.7334, 'colsample_bylevel': 0.9399,
    'min_child_weight': 14, 'reg_alpha': 2.7827,
    'reg_lambda': 9.2059, 'gamma': 0.1368,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 4, 'learning_rate': 0.0850,
    'n_estimators': 440, 'subsample': 0.7546,
    'colsample_bytree': 0.7527, 'colsample_bylevel': 0.7170,
    'min_child_weight': 11, 'reg_alpha': 5.3725,
    'reg_lambda': 9.4165, 'gamma': 0.5212,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 10, 'learning_rate': 0.0971,
    'n_estimators': 680, 'subsample': 0.7700,
    'colsample_bytree': 0.8117, 'min_child_weight': 16,
    'reg_alpha': 5.8625, 'reg_lambda': 2.4112,
    'gamma': 0.6163, 'scale_pos_weight': 1.2817,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 11, 'learning_rate': 0.0301,
    'n_estimators': 445, 'subsample': 0.7961,
    'colsample_bytree': 0.7944, 'min_child_weight': 7,
    'reg_alpha': 4.2403, 'reg_lambda': 5.3672,
    'gamma': 1.0380, 'scale_pos_weight': 0.9259,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

# TEST PERFORMANCE (RAW): 69.4% (UP 67.8%, DOWN 70.8%)
# MAE: 13.12% (UP: 11.58%, DOWN: 14.50%)
# Validation: Exp 8.80% Comp 4.85% UP 58.7% DOWN 59.5%
