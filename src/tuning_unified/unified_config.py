# UNIFIED CONFIG - TERNARY DECISION SYSTEM - 2025-12-21 12:44:20
# Production-aligned: Uses actual predict() method during tuning
# Test set performance: 72.3% (UP 74.0%, DOWN 71.1%)
# TRAINED WITH FROZEN FEATURES

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.6010,
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
    'max_depth': 3, 'learning_rate': 0.0568,
    'n_estimators': 525, 'subsample': 0.7429,
    'colsample_bytree': 0.7669, 'colsample_bylevel': 0.8084,
    'min_child_weight': 11, 'reg_alpha': 6.9131,
    'reg_lambda': 8.4442, 'gamma': 0.2447,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 5, 'learning_rate': 0.0518,
    'n_estimators': 425, 'subsample': 0.8733,
    'colsample_bytree': 0.8959, 'colsample_bylevel': 0.8160,
    'min_child_weight': 5, 'reg_alpha': 4.1977,
    'reg_lambda': 4.8079, 'gamma': 0.4272,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0557,
    'n_estimators': 209, 'subsample': 0.9302,
    'colsample_bytree': 0.7787, 'min_child_weight': 9,
    'reg_alpha': 3.8747, 'reg_lambda': 18.5912,
    'gamma': 1.6774, 'scale_pos_weight': 0.7311,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0191,
    'n_estimators': 293, 'subsample': 0.8437,
    'colsample_bytree': 0.9412, 'min_child_weight': 9,
    'reg_alpha': 3.8104, 'reg_lambda': 13.9756,
    'gamma': 1.9401, 'scale_pos_weight': 0.7199,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0484,
    'confidence_weights': {
        'up': {'classifier': 0.6533, 'magnitude': 0.3467},
        'down': {'classifier': 0.7371, 'magnitude': 0.2629}
    },
    'magnitude_scaling': {
        'up': {'small': 3.7312, 'medium': 7.2221, 'large': 11.5749},
        'down': {'small': 3.5946, 'medium': 7.7682, 'large': 9.5340}
    },
    'decision_threshold': 0.6778,
    'description': 'Ternary decision system optimized on test set using production logic'
}

# TEST SET PERFORMANCE (Production-aligned):
# Overall: 72.3% | UP: 74.0% (123 signals) | DOWN: 71.1% (166 signals)
# NO_DECISION: 191 (39.8% of total)

# FROZEN FEATURES USED
# Expansion: 35 features
# Compression: 42 features
# UP: 30 features
# DOWN: 31 features
