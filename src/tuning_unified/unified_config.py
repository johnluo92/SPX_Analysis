# UNIFIED CONFIG (UPGRADED) - 2025-12-16 00:36:40
# Uses consistent eval_metric='logloss' for both classifiers
# TRAINED WITH FROZEN FEATURES

QUALITY_FILTER_CONFIG = {'enabled': True, 'min_threshold': 0.5000,
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
    'max_depth': 5, 'learning_rate': 0.0165,
    'n_estimators': 668, 'subsample': 0.9122,
    'colsample_bytree': 0.7868, 'colsample_bylevel': 0.7545,
    'min_child_weight': 14, 'reg_alpha': 6.1920,
    'reg_lambda': 6.2310, 'gamma': 0.1538,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

COMPRESSION_PARAMS = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 3, 'learning_rate': 0.0128,
    'n_estimators': 605, 'subsample': 0.9458,
    'colsample_bytree': 0.8038, 'colsample_bylevel': 0.9355,
    'min_child_weight': 4, 'reg_alpha': 6.0451,
    'reg_lambda': 9.0234, 'gamma': 0.1011,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

UP_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0282,
    'n_estimators': 297, 'subsample': 0.9009,
    'colsample_bytree': 0.7713, 'min_child_weight': 9,
    'reg_alpha': 6.6232, 'reg_lambda': 9.0507,
    'gamma': 1.8426, 'scale_pos_weight': 0.7410,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

DOWN_CLASSIFIER_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.0392,
    'n_estimators': 217, 'subsample': 0.6274,
    'colsample_bytree': 0.8678, 'min_child_weight': 16,
    'reg_alpha': 8.5402, 'reg_lambda': 19.4843,
    'gamma': 1.9034, 'scale_pos_weight': 0.7736,
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0357,
    'confidence_weights': {
        'up': {'classifier': 0.6990, 'magnitude': 0.3010},
        'down': {'classifier': 0.7367, 'magnitude': 0.2633}
    },
    'magnitude_scaling': {
        'up': {'small': 3.4269, 'medium': 7.1849, 'large': 11.1666},
        'down': {'small': 2.5002, 'medium': 7.6739, 'large': 9.8932}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6562,
            'medium_magnitude': 0.7383,
            'low_magnitude': 0.7310
        },
        'down': {
            'high_magnitude': 0.7544,
            'medium_magnitude': 0.7959,
            'low_magnitude': 0.8076
        }
    },
    'min_confidence_up': 0.6737,
    'min_confidence_down': 0.6750,
    'boost_threshold_up': 19.1231,
    'boost_threshold_down': 11.9596,
    'boost_amount_up': 0.0340,
    'boost_amount_down': 0.0528,
    'description': 'Unified tuning - base models + ensemble together (UPGRADED)'
}

# ACTIONABLE: 72.5% (UP 71.1%, DOWN 75.3%)
# Signals: 222 (46.2% actionable)
# UP signals: 149 | DOWN signals: 73

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
