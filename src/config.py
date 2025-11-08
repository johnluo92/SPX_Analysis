"""Enhanced Configuration V4 - Cleaned (119 features removed)"""
from pathlib import Path

# Data Paths
CACHE_DIR = './data_cache'
CBOE_DATA_DIR = './CBOE_Data_Archive'

# Training Control
ENABLE_TRAINING = True
TRAINING_YEARS = 15
RANDOM_STATE = 42

# VIX Regime Boundaries
REGIME_BOUNDARIES = [0, 16.77, 24.40, 39.67, 100]
REGIME_NAMES = {0: "Low Vol", 1: "Normal", 2: "Elevated", 3: "Crisis"}

# Thresholds
SKEW_ELEVATED_THRESHOLD = 145
CRISIS_VIX_THRESHOLD = 39.67
SPX_FORWARD_WINDOWS = [5, 13, 21]
SPX_RANGE_THRESHOLDS = [0.02, 0.03, 0.05]

# Model Parameters
DURATION_PREDICTOR_PARAMS = {
    'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 50,
    'min_samples_leaf': 20, 'random_state': RANDOM_STATE, 'n_jobs': -1,
    'max_duration_cap': 30
}

ANOMALY_DETECTOR_PARAMS = {
    'contamination': 0.01, 'n_estimators': 100, 'max_samples': 'auto',
    'random_state': RANDOM_STATE
}

XGBOOST_REGIME_PARAMS = {
    'objective': 'multi:softprob', 'num_class': 4, 'max_depth': 8,
    'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_weight': 5, 'gamma': 0.1,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'eval_metric': 'mlogloss', 'early_stopping_rounds': 50
}

XGBOOST_RANGE_PARAMS = {
    'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.05,
    'n_estimators': 250, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 3, 'gamma': 0.05, 'reg_alpha': 0.05,
    'reg_lambda': 0.5, 'random_state': RANDOM_STATE, 'n_jobs': -1,
    'eval_metric': 'rmse', 'early_stopping_rounds': 50
}

# SHAP Configuration
SHAP_CONFIG = {
    'method': 'shap', 'n_samples': 500, 'n_repeats': 1,
    'use_shap_if_available': True, 'timeout_warning': True
}

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    'anomaly': {'extreme': 88, 'high': 78, 'moderate': 70},
    'duration': {'long': 7, 'medium': 3}
}

# Anomaly Detection Thresholds
ANOMALY_THRESHOLDS = {
    'detector_coverage_min': 0.5,
    'feature_availability_min': 0.7,
    'ensemble_weight_min': 0.3
}

# Feature Definitions - VIX Base
VIX_BASE_FEATURES = {
    'mean_reversion': [
        'vix_vs_ma10', 'vix_vs_ma21', 'vix_vs_ma63', 'vix_vs_ma252',
        'vix_bb_position_20d', 'reversion_strength_21d', 'reversion_strength_63d',
        'vix_pull_ma21', 'vix_pull_ma63', 'vix_stretch_ma21', 'vix_stretch_ma63',
        'vix_extreme_low_21d', 'vix_mean_distance_21d'
    ],
    'dynamics': [
        'vix_ret_1d', 'vix_ret_5d', 'vix_ret_10d', 'vix_ret_21d',
        'vix_vol_10d', 'vix_vol_21d', 'vix_vol_63d',
        'vix_velocity_5d', 'vix_velocity_10d', 'vix_velocity_21d',
        'vix_zscore_21d', 'vix_zscore_63d', 'vix_zscore_252d',
        'vix_momentum_z_10d', 'vix_momentum_z_21d', 'vix_accel_5d'
    ],
    'regimes': [
        'vix_regime', 'vix_regime_duration', 'vix_displacement',
        'vix_term_structure'
    ]
}

# SPX Base Features (removed binary features, low-variance features, and redundant features)
SPX_BASE_FEATURES = {
    'price_action': [
        'spx_ret_1d', 'spx_ret_5d', 'spx_ret_10d', 'spx_ret_21d', 'spx_ret_63d',
        'spx_vs_ma20', 'spx_vs_ma50', 'spx_vs_ma200',
        'spx_momentum_z_10d', 'spx_momentum_z_21d'
    ],
    'volatility': [
        'spx_realized_vol_10d', 'spx_realized_vol_21d', 'spx_realized_vol_63d',
        'spx_vol_ratio_10_21', 'spx_vol_ratio_10_63',
        'spx_skew_21d', 'spx_kurt_21d', 'bb_position_20d', 'bb_width_20d'
    ],
    'technical': [
        'rsi_14', 'rsi_regime', 'rsi_divergence',
        'macd', 'macd_signal', 'macd_histogram', 'adx_14', 'trend_strength'
    ],
    'ohlc_microstructure': [
        'spx_body_size', 'spx_range', 'spx_range_pct',
        'spx_upper_shadow', 'spx_lower_shadow', 'spx_close_position',
        'spx_body_to_range', 'spx_gap',
        'spx_gap_magnitude',
        'spx_upper_rejection', 'spx_lower_rejection', 'spx_range_expansion'
    ]
}

# Cross-Asset Features
CROSS_ASSET_BASE_FEATURES = {
    'spx_vix_relationship': [
        'spx_vix_corr_21d', 'spx_vix_corr_63d', 'spx_vix_corr_126d',
        'vix_vs_rv_10d', 'vix_vs_rv_21d', 'vix_rv_ratio_10d', 'vix_rv_ratio_21d'
    ]
}

# CBOE Base Features (removed redundant _velocity_, _change_, and _momentum_regime features)
CBOE_BASE_FEATURES = {
    'skew_indicators': [
        'SKEW', 'skew_regime', 'skew_vs_vix', 'skew_vix_ratio',
        'skew_displacement'
    ],
    'put_call_ratios': [
        'pc_equity_inst_divergence',
        'pcc_accel_10d'
    ],
    'correlation_indices': [
        'COR1M', 'COR1M_change_21d', 'COR1M_zscore_63d',
        'COR3M', 'COR3M_change_21d', 'COR3M_zscore_63d',
        'cor_term_structure', 'cor_term_slope_change_21d'
    ],
    'other_cboe': [
        'VXTH', 'VXTH_change_21d', 'VXTH_zscore_63d',
        'vxth_vix_ratio',
        'cboe_stress_composite', 'cboe_stress_regime'
    ],
    'bond_volatility': [
        'VXTLT', 'VXTLT_change_21d', 'VXTLT_zscore_63d',
        'VXTLT_velocity_10d', 'VXTLT_acceleration_5d',
        'bond_vol_regime', 'vxtlt_vix_ratio', 'vxtlt_vix_spread',
        'VXTLT_percentile_63d', 'VXTLT_percentile_126d', 'VXTLT_percentile_252d',
        'VXTLT_vs_ma21', 'VXTLT_vs_ma63', 'bond_equity_vol_divergence'
    ]
}

# Futures Features (cleaned up redundant features)
FUTURES_FEATURES = {
    'vix_futures': [
        'VX1-VX2', 'VX1-VX2_change_21d',
        'VX1-VX2_zscore_63d', 'VX1-VX2_percentile_63d',
        'VX2-VX1_RATIO', 'VX2-VX1_RATIO_velocity_10d',
        'vx_term_structure_regime', 'vx_curve_acceleration',
        'vx_term_structure_divergence'
    ],
    'commodity_futures': [
        'CL1-CL2', 'CL1-CL2_velocity_5d',
        'CL1-CL2_zscore_63d', 'oil_term_regime',
        'crude_oil_ret_10d', 'crude_oil_ret_21d', 'crude_oil_ret_63d',
        'crude_oil_vol_21d', 'crude_oil_zscore_63d'
    ],
    'dollar_futures': [
        'DX1-DX2', 'DX1-DX2_velocity_5d',
        'DX1-DX2_zscore_63d', 'dxy_ret_10d', 'dxy_ret_21d', 'dxy_ret_63d',
        'dxy_vs_ma50', 'dxy_vs_ma200', 'dxy_vol_21d'
    ],
    'cross_futures': [
        'vx_crude_corr_21d', 'vx_crude_divergence',
        'vx_dollar_corr_21d', 'dollar_crude_corr_21d',
        'spx_vx_spread_corr_21d',
        'spx_dollar_corr_21d'
    ]
}

# Meta Features (removed redundant regime features)
META_FEATURES = {
    'regime_indicators': [
        'vix_regime_micro', 'regime_transition_risk', 'vol_regime', 'risk_premium_regime',
        'vol_term_regime', 'trend_regime', 'trend_strength',
        'liquidity_stress_composite', 'liquidity_regime',
        'correlation_regime'
    ],
    'cross_asset_relationships': [
        'equity_vol_divergence', 'equity_vol_corr_breakdown',
        'risk_premium_ma21', 'risk_premium_velocity',
        'risk_premium_zscore', 'gold_spx_divergence', 'dollar_spx_correlation'
    ],
    'rate_of_change': [
        'vix_velocity_3d_pct',
        'vix_jerk_5d', 'vix_momentum_regime',
        'SKEW_momentum_regime', 'spx_realized_vol_21d_velocity_3d',
        'spx_realized_vol_21d_acceleration_5d', 'vix_skew_momentum_divergence'
    ],
    'percentile_rankings': [
        'vix_percentile_21d', 'vix_percentile_63d', 'vix_percentile_126d',
        'vix_percentile_252d', 'vix_percentile_velocity',
        'vix_extreme_low_63d',
        'vix_extreme_low_252d',
        'SKEW_percentile_21d', 'SKEW_percentile_63d', 'SKEW_percentile_126d',
        'SKEW_percentile_velocity', 'risk_premium_percentile_21d',
        'risk_premium_percentile_63d', 'risk_premium_percentile_126d',
        'risk_premium_extreme_low_63d'
    ]
}

# Macro Features (cleaned - remove redundant features)
MACRO_FEATURES = {
    'currencies': [
        'Dollar_Index_lag1', 'Dollar_Index_zscore_63d'
    ],
    'volatility_indices': [
        'Bond_Vol_lag1', 'Bond_Vol_mom_10d', 'Bond_Vol_mom_21d',
        'Bond_Vol_mom_63d', 'Bond_Vol_zscore_63d'
    ]
}

# Calendar Features (removed quarter)
CALENDAR_FEATURES = [
    'month', 'day_of_week', 'day_of_month'
]

# Anomaly Detection Groups (cleaned)
ANOMALY_FEATURE_GROUPS = {
    'vix_mean_reversion': VIX_BASE_FEATURES['mean_reversion'] + ['vix'],
    'vix_momentum': VIX_BASE_FEATURES['dynamics'] + [
        'vix_velocity_3d', 'vix_jerk_5d'
    ],
    'vix_regime_structure': (
        VIX_BASE_FEATURES['mean_reversion'][:5] +
        VIX_BASE_FEATURES['dynamics'] + VIX_BASE_FEATURES['regimes']
    ),
    'bond_volatility_regime': [
        'VXTLT', 'VXTLT_change_21d', 'VXTLT_zscore_63d',
        'bond_vol_regime', 'vxtlt_vix_ratio', 'vxtlt_vix_spread',
        'VXTLT_percentile_63d', 'VXTLT_vs_ma21', 'bond_equity_vol_divergence',
        'vix', 'vix_regime', 'spx_realized_vol_21d'
    ],
    'cboe_options_flow': (
        CBOE_BASE_FEATURES['skew_indicators'] +
        CBOE_BASE_FEATURES['put_call_ratios'] +
        CBOE_BASE_FEATURES['correlation_indices']
    ),
    'cboe_cross_dynamics': CBOE_BASE_FEATURES['other_cboe'] + [
        'skew_vs_vix',
        'skew_vix_ratio', 'pc_equity_inst_divergence',
        'cor_term_slope_change_21d'
    ],
    'vix_spx_relationship': (
        CROSS_ASSET_BASE_FEATURES['spx_vix_relationship'] +
        ['vix', 'vix_vs_ma21', 'spx_realized_vol_21d', 'spx_ret_21d',
         'spx_momentum_z_21d']
    ),
    'spx_price_action': (
        SPX_BASE_FEATURES['price_action'] + SPX_BASE_FEATURES['technical'] +
        ['spx_body_size', 'spx_range_pct', 'spx_close_position',
         'spx_range_expansion', 'spx_gap']
    ),
    'spx_ohlc_microstructure': SPX_BASE_FEATURES['ohlc_microstructure'],
    'spx_volatility_regime': (
        SPX_BASE_FEATURES['volatility'] +
        ['vix_vs_rv_10d', 'vix_vs_rv_21d', 'vix_rv_ratio_21d', 'vix',
         'vix_vol_10d', 'vix_vol_21d', 'bb_width_20d', 'spx_ret_21d',
         'spx_ret_63d', 'spx_momentum_z_21d', 'vix_velocity_21d',
         'vix_zscore_63d', 'vix_regime',
         'spx_range_expansion']
    ),
    'cross_asset_divergence': [
        'spx_vix_corr_21d', 'spx_vix_corr_63d', 'vix_vs_rv_21d',
        'vix_rv_ratio_21d', 'spx_realized_vol_21d', 'spx_ret_21d',
        'spx_momentum_z_21d', 'vix_velocity_21d', 'vix_momentum_z_21d',
        'spx_vs_ma20', 'spx_vs_ma50', 'vix_vs_ma21', 'vix_vs_ma63',
        'spx_vol_ratio_10_63', 'rsi_14', 'spx_skew_21d', 'spx_kurt_21d',
        'vix', 'vix_regime', 'vix_displacement',
        'spx_upper_rejection', 'spx_lower_rejection', 'spx_gap_magnitude',
        'equity_vol_divergence', 'equity_vol_corr_breakdown'
    ],
    'tail_risk_complex': [
        'SKEW', 'skew_regime', 'skew_displacement', 'skew_vs_vix',
        'cboe_stress_composite', 'cboe_stress_regime',
        'VXTLT', 'vxtlt_vix_ratio', 'bond_vol_regime',  # NEW
        'vix', 'vix_regime', 'vix_zscore_63d', 'vix_velocity_21d',
        'spx_upper_rejection',
        'spx_skew_21d', 'spx_kurt_21d'
    ],
    'futures_term_structure': (
        FUTURES_FEATURES['vix_futures'] +
        FUTURES_FEATURES['commodity_futures'][:4] +
        FUTURES_FEATURES['dollar_futures'][:3]
    ),
    'macro_regime_shifts': (
        META_FEATURES['regime_indicators'] +
        META_FEATURES['cross_asset_relationships'][:5]
    ),
    'momentum_acceleration': (
        META_FEATURES['rate_of_change'] +
        ['vix_velocity_10d', 'vix_velocity_21d', 'vix_accel_5d',
         'spx_momentum_z_10d', 'spx_momentum_z_21d']
    ),
    'percentile_extremes': META_FEATURES['percentile_rankings']
}

# Regime Classification Groups (cleaned)
REGIME_CLASSIFICATION_FEATURE_GROUPS = {
    'all_vix': (
        VIX_BASE_FEATURES['mean_reversion'] +
        VIX_BASE_FEATURES['dynamics'] + VIX_BASE_FEATURES['regimes']
    ),
    'all_spx': (
        SPX_BASE_FEATURES['price_action'] +
        SPX_BASE_FEATURES['volatility'] + SPX_BASE_FEATURES['technical']
    ),
    'all_spx_ohlc': SPX_BASE_FEATURES['ohlc_microstructure'],
    'all_cross_asset': CROSS_ASSET_BASE_FEATURES['spx_vix_relationship'],
    'all_cboe': (
        CBOE_BASE_FEATURES['skew_indicators'] +
        CBOE_BASE_FEATURES['put_call_ratios'] +
        CBOE_BASE_FEATURES['correlation_indices'] +
        CBOE_BASE_FEATURES['other_cboe'] +
        CBOE_BASE_FEATURES['bond_volatility']  # NEW
    ),
    'all_futures': (
        FUTURES_FEATURES['vix_futures'] +
        FUTURES_FEATURES['commodity_futures'] +
        FUTURES_FEATURES['dollar_futures'] +
        FUTURES_FEATURES['cross_futures']
    ),
    'all_meta': (
        META_FEATURES['regime_indicators'] +
        META_FEATURES['cross_asset_relationships'] +
        META_FEATURES['rate_of_change'] +
        META_FEATURES['percentile_rankings']
    ),
    'all_macro': (
        list(MACRO_FEATURES['currencies']) +
        list(MACRO_FEATURES['volatility_indices'])
    ),
    'calendar': CALENDAR_FEATURES
}

# Range Prediction Groups (cleaned)
RANGE_PREDICTION_FEATURE_GROUPS = {
    'vix_dynamics': VIX_BASE_FEATURES['dynamics'] + ['vix'],
    'spx_price_vol': SPX_BASE_FEATURES['price_action'] + SPX_BASE_FEATURES['volatility'],
    'cboe_signals': (
        CBOE_BASE_FEATURES['skew_indicators'] +
        CBOE_BASE_FEATURES['put_call_ratios']
    ),
    'futures_structure': FUTURES_FEATURES['vix_futures'][:6],
    'meta_regimes': META_FEATURES['regime_indicators'][:7],
    'calendar': CALENDAR_FEATURES
}