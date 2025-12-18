# UNIFIED CONFIG (UPGRADED) - 2025-12-17 21:25:52
# Uses consistent eval_metric='logloss' for both classifiers
# TRAINED WITH FROZEN FEATURES

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5494,
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
    'description': 'Unified tuning with ensemble evaluation'}

EXPANSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 6, 'learning_rate': 0.0180,
    'n_estimators': 791, 'subsample': 0.7048,
    'colsample_bytree': 0.9217, 'colsample_bylevel': 0.8514,
    'min_child_weight': 9, 'reg_alpha': 7.5614,
    'reg_lambda': 9.3269, 'gamma': 0.3872,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 6, 'learning_rate': 0.0105,
    'n_estimators': 801, 'subsample': 0.8472,
    'colsample_bytree': 0.7356, 'colsample_bylevel': 0.9159,
    'min_child_weight': 3, 'reg_alpha': 7.2376,
    'reg_lambda': 3.8474, 'gamma': 0.0350,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0300,
    'n_estimators': 235, 'subsample': 0.8046,
    'colsample_bytree': 0.7546, 'min_child_weight': 6,
    'reg_alpha': 1.9392, 'reg_lambda': 16.4233,
    'gamma': 2.3502, 'scale_pos_weight': 0.7866,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0686,
    'n_estimators': 285, 'subsample': 0.6606,
    'colsample_bytree': 0.7336, 'min_child_weight': 18,
    'reg_alpha': 5.9200, 'reg_lambda': 18.1730,
    'gamma': 1.7357, 'scale_pos_weight': 0.7216,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0819,
    'confidence_weights': {
        'up': {'classifier': 0.5696, 'magnitude': 0.4304},
        'down': {'classifier': 0.6940, 'magnitude': 0.3060}
    },
    'magnitude_scaling': {
        'up': {'small': 3.9265, 'medium': 6.1406, 'large': 12.8440},
        'down': {'small': 3.2286, 'medium': 6.7560, 'large': 9.2251}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6152,
            'medium_magnitude': 0.7095,
            'low_magnitude': 0.7766
        },
        'down': {
            'high_magnitude': 0.7506,
            'medium_magnitude': 0.7371,
            'low_magnitude': 0.8056
        }
    },
    'min_confidence_up': 0.6959,
    'min_confidence_down': 0.6864,
    'boost_threshold_up': 14.4621,
    'boost_threshold_down': 14.1743,
    'boost_amount_up': 0.0549,
    'boost_amount_down': 0.0605,
    'description': 'Unified tuning - base models + ensemble together (UPGRADED)'
}

# ACTIONABLE: 75.2% (UP 74.4%, DOWN 76.6%)
# Signals: 137 (28.5% actionable)
# UP signals: 90 | DOWN signals: 47

# ============================================================================
# FROZEN FEATURES USED (loaded from pre-selection)
# ============================================================================
# These features were selected via feature stability testing and frozen
# for consistent, reproducible tuning across all trials.
#
# Expansion features (35):
#   VX1-VX2_zscore_63d, VXTH, VXTLT_vs_ma21, copper, credit_stress_composite, crude_oil_ret_63d, day_of_month, dxy_extreme_high, dxy_gold_corr_63d, dxy_ret_63d...
#
# Compression features (42):
#   bb_position_20d, dxy_gold_corr_21d, dxy_gold_corr_63d, gold_ret_21d, gold_ret_63d, gold_zscore_21d, gold_zscore_252d, joint_regime_correlation, joint_regime_expected_return, natgas...
#
# UP features (30):
#   crude_oil_ret_63d, day_of_month, dxy, dxy_gold_corr_21d, dxy_gold_corr_63d, dxy_vs_ma21, dxy_vs_ma63, dxy_zscore_63d, macd_signal, month...
#
# DOWN features (31):
#   crude_oil_ret_63d, day_of_month, dxy, dxy_gold_corr_21d, dxy_gold_corr_63d, dxy_vs_ma21, dxy_vs_ma63, dxy_zscore_63d, macd_signal, month...
#
# Full feature lists saved in: frozen_features.json
