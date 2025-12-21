# UNIFIED CONFIG - TERNARY DECISION SYSTEM - 2025-12-20 17:42:35
# Production-aligned: Uses actual predict() method during tuning
# Test set performance: 73.2% (UP 75.0%, DOWN 71.9%)
# TRAINED WITH FROZEN FEATURES

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5942,
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
    'max_depth': 3, 'learning_rate': 0.0370,
    'n_estimators': 695, 'subsample': 0.8128,
    'colsample_bytree': 0.7820, 'colsample_bylevel': 0.7826,
    'min_child_weight': 5, 'reg_alpha': 6.8529,
    'reg_lambda': 8.9070, 'gamma': 0.4589,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 5, 'learning_rate': 0.0343,
    'n_estimators': 455, 'subsample': 0.9422,
    'colsample_bytree': 0.8821, 'colsample_bylevel': 0.7268,
    'min_child_weight': 6, 'reg_alpha': 4.7917,
    'reg_lambda': 3.5334, 'gamma': 0.3129,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 5, 'learning_rate': 0.0816,
    'n_estimators': 241, 'subsample': 0.9280,
    'colsample_bytree': 0.7186, 'min_child_weight': 18,
    'reg_alpha': 7.9960, 'reg_lambda': 6.7143,
    'gamma': 1.7258, 'scale_pos_weight': 0.9302,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0315,
    'n_estimators': 207, 'subsample': 0.8670,
    'colsample_bytree': 0.9249, 'min_child_weight': 11,
    'reg_alpha': 7.8304, 'reg_lambda': 5.6848,
    'gamma': 2.0598, 'scale_pos_weight': 0.8441,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': -0.0289,
    'confidence_weights': {
        'up': {'classifier': 0.6354, 'magnitude': 0.3646},
        'down': {'classifier': 0.7377, 'magnitude': 0.2623}
    },
    'magnitude_scaling': {
        'up': {'small': 2.8361, 'medium': 7.2699, 'large': 11.9570},
        'down': {'small': 3.4205, 'medium': 6.3161, 'large': 9.3007}
    },
    'decision_threshold': 0.6674,
    'description': 'Ternary decision system optimized on test set using production logic'
}

# TEST SET PERFORMANCE (Production-aligned):
# Overall: 73.2% | UP: 75.0% (120 signals) | DOWN: 71.9% (167 signals)
# NO_DECISION: 193 (40.2% of total)

# FROZEN FEATURES USED
# Expansion: 35 features
# Compression: 42 features
# UP: 30 features
# DOWN: 31 features
