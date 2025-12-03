# PHASE 1 OPTIMIZED CONFIG - 2025-12-03 18:43:45

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5503,
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}

CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': 1.3828, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': 1.4475, 'description': 'Options expiration week + VIX futures rollover'},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': 1.3736, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}

FEATURE_SELECTION_CV_PARAMS = {'n_estimators': 268,
    'max_depth': 6, 'learning_rate': 0.0377,
    'subsample': 0.8647, 'colsample_bytree': 0.9171}

FEATURE_SELECTION_CONFIG = {'expansion_top_n': 121,
    'compression_top_n': 78, 'up_top_n': 124,
    'down_top_n': 143, 'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': 0.8801,
    'description': 'Phase 1 optimized on 2024-2025 test data'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 7, 'learning_rate': 0.0915,
    'n_estimators': 721, 'subsample': 0.9070,
    'colsample_bytree': 0.7224, 'colsample_bylevel': 0.7446,
    'min_child_weight': 15, 'reg_alpha': 7.0416,
    'reg_lambda': 2.1126, 'gamma': 0.4922,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 3, 'learning_rate': 0.0452,
    'n_estimators': 344, 'subsample': 0.9373,
    'colsample_bytree': 0.7107, 'colsample_bylevel': 0.7537,
    'min_child_weight': 11, 'reg_alpha': 5.2203,
    'reg_lambda': 2.4856, 'gamma': 0.4325,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 10, 'learning_rate': 0.0397,
    'n_estimators': 741, 'subsample': 0.7425,
    'colsample_bytree': 0.8691, 'min_child_weight': 10,
    'reg_alpha': 5.0048, 'reg_lambda': 6.3301,
    'gamma': 0.9399, 'scale_pos_weight': 1.0,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 5, 'learning_rate': 0.0331,
    'n_estimators': 554, 'subsample': 0.8662,
    'colsample_bytree': 0.8140, 'min_child_weight': 17,
    'reg_alpha': 5.3615, 'reg_lambda': 7.0824,
    'gamma': 1.1184, 'scale_pos_weight': 1.0,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

# TEST PERFORMANCE: Raw 66.6% (UP 65.6%, DOWN 67.4%)
# Actionable 67.7% (UP 65.6%, DOWN 69.8%) MAE 13.26%
# Validation: Exp 8.80% Comp 4.83% UP 56.9% DOWN 57.9%
