# UNIFIED CONFIG - 2025-12-09 07:58:21

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5669,
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}

CALENDAR_COHORTS = {
    'fomc_period': {'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': 1.1410, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},
    'opex_week': {'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': 1.1143, 'description': 'Options expiration week + VIX futures rollover'},
    'earnings_heavy': {'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': 1.3949, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'},
    'mid_cycle': {'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}

FEATURE_SELECTION_CV_PARAMS = {'n_estimators': 168,
    'max_depth': 6, 'learning_rate': 0.0320,
    'subsample': 0.9318, 'colsample_bytree': 0.9203,
    'n_jobs': 1, 'random_state': 42}

FEATURE_SELECTION_CONFIG = {'expansion_top_n': 115,
    'compression_top_n': 96, 'up_top_n': 112,
    'down_top_n': 149, 'cv_folds': 5, 'protected_features': [],
    'correlation_threshold': 0.8531,
    'description': 'Unified tuning with ensemble evaluation'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 4, 'learning_rate': 0.0119,
    'n_estimators': 699, 'subsample': 0.9268,
    'colsample_bytree': 0.7740, 'colsample_bylevel': 0.9367,
    'min_child_weight': 13, 'reg_alpha': 6.6587,
    'reg_lambda': 2.0007, 'gamma': 0.7695,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 3, 'learning_rate': 0.1138,
    'n_estimators': 506, 'subsample': 0.7987,
    'colsample_bytree': 0.7655, 'colsample_bylevel': 0.8903,
    'min_child_weight': 6, 'reg_alpha': 3.9449,
    'reg_lambda': 8.9603, 'gamma': 0.4900,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 4, 'learning_rate': 0.0202,
    'n_estimators': 297, 'subsample': 0.6315,
    'colsample_bytree': 0.9279, 'min_child_weight': 16,
    'reg_alpha': 1.7430, 'reg_lambda': 14.6926,
    'gamma': 2.1211, 'scale_pos_weight': 0.9606,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 7, 'learning_rate': 0.0228,
    'n_estimators': 231, 'subsample': 0.6327,
    'colsample_bytree': 0.7150, 'min_child_weight': 18,
    'reg_alpha': 3.3460, 'reg_lambda': 4.1332,
    'gamma': 1.2398, 'scale_pos_weight': 0.9195,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0398,
    'confidence_weights': {
        'up': {'classifier': 0.5506, 'magnitude': 0.4494},
        'down': {'classifier': 0.6777, 'magnitude': 0.3223}
    },
    'magnitude_scaling': {
        'up': {'small': 3.6145, 'medium': 6.2227, 'large': 10.5611},
        'down': {'small': 4.2454, 'medium': 7.1775, 'large': 13.3878}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6200,
            'medium_magnitude': 0.7418,
            'low_magnitude': 0.7783
        },
        'down': {
            'high_magnitude': 0.7424,
            'medium_magnitude': 0.8093,
            'low_magnitude': 0.8023
        }
    },
    'min_confidence_up': 0.6075,
    'min_confidence_down': 0.6775,
    'boost_threshold_up': 17.5297,
    'boost_threshold_down': 11.8896,
    'boost_amount_up': 0.0536,
    'boost_amount_down': 0.0564,
    'description': 'Unified tuning - base models + ensemble together'
}

# ACTIONABLE: 75.0% (UP 75.4%, DOWN 74.1%)
# Signals: 188 (39.2% actionable)
# UP signals: 130 | DOWN signals: 58
