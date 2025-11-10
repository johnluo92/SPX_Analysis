"""Enhanced Configuration V4 - Comprehensive Feature Groups and Model Settings"""
from pathlib import Path

# ==================== DATA PATHS ====================
FRED_API_KEY_PATH = Path(__file__).parent / 'json_data' / 'config.json'
CACHE_DIR = './data_cache'
CBOE_DATA_DIR = './CBOE_Data_Archive'

# ==================== TRAINING CONTROL ====================
ENABLE_TRAINING = True
TRAINING_YEARS = 15
RANDOM_STATE = 42

# ==================== VIX REGIME BOUNDARIES ====================
REGIME_BOUNDARIES = [0, 16.77, 24.40, 39.67, 100]
REGIME_NAMES = {0: "Low Vol", 1: "Normal", 2: "Elevated", 3: "Crisis"}

# ==================== THRESHOLDS ====================
SKEW_ELEVATED_THRESHOLD = 145
CRISIS_VIX_THRESHOLD = 39.67

# SPX Forward Prediction Windows
SPX_FORWARD_WINDOWS = [5, 13, 21]
SPX_RANGE_THRESHOLDS = [0.02, 0.03, 0.05]

# ==================== MODEL PARAMETERS ====================

# Random Forest for Anomaly Duration Prediction
DURATION_PREDICTOR_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'max_duration_cap': 30
}

# Isolation Forest for Anomaly Detection
ANOMALY_DETECTOR_PARAMS = {
    'contamination': 0.01,
    'n_estimators': 100,
    'max_samples': 'auto',
    'random_state': RANDOM_STATE
}

# XGBoost for Regime Classification
XGBOOST_REGIME_PARAMS = {
    'objective': 'multi:softprob',  # Multi-class probability
    'num_class': 4,  # 4 regime classes
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 50
}

# XGBoost for Forward SPX Range Prediction
XGBOOST_RANGE_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 250,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'rmse',
    'early_stopping_rounds': 50
}

# ==================== SHAP CONFIGURATION ====================
SHAP_CONFIG = {
    'method': 'shap',
    'n_samples': 500,
    'n_repeats': 1,
    'use_shap_if_available': True,
    'timeout_warning': True
}

# ==================== SIGNAL THRESHOLDS ====================
SIGNAL_THRESHOLDS = {
    'anomaly': {
        'extreme': 88,
        'high': 78,
        'moderate': 70
    },
    'duration': {
        'extended_multiplier': 1.5,
        'fresh_threshold': 'median'
    },
    'displacement': {
        'significant': 1.5
    },
    'regime_transition': {
        'high_confidence': 0.7,  # Regime prediction confidence threshold
        'extreme_confidence': 0.85
    }
}

# ==================== POSITION SIZING STRATEGIES ====================
POSITION_SIZING = {
    'AGGRESSIVE': {
        'description': 'Maximum position size',
        'iron_condor_notional': 1.0,
        'delta_hedge': 'Minimal',
        'stop_loss': 'Wide'
    },
    'MODERATE': {
        'description': 'Standard position size',
        'iron_condor_notional': 0.6,
        'delta_hedge': 'Standard',
        'stop_loss': 'Standard'
    },
    'LIGHT': {
        'description': 'Reduced position size',
        'iron_condor_notional': 0.3,
        'delta_hedge': 'Increased',
        'stop_loss': 'Tight'
    },
    'OPPORTUNISTIC': {
        'description': 'Scale in over multiple days',
        'iron_condor_notional': 0.5,
        'delta_hedge': 'Dynamic',
        'stop_loss': 'Trailing'
    }
}

# ==================== ENHANCED FEATURE GROUPS ====================
# Organized by modeling purpose and data source

# Base VIX Features (Core structural features)
VIX_BASE_FEATURES = {
    'mean_reversion': [
        'vix', 'vix_vs_ma10', 'vix_vs_ma21', 'vix_vs_ma63', 'vix_vs_ma126', 'vix_vs_ma252',
        'vix_vs_ma10_pct', 'vix_vs_ma21_pct', 'vix_vs_ma63_pct', 'vix_vs_ma126_pct', 'vix_vs_ma252_pct',
        'vix_zscore_63d', 'vix_zscore_126d', 'vix_zscore_252d',
        'vix_percentile_126d', 'vix_percentile_252d', 'reversion_strength_63d'
    ],
    'dynamics': [
        'vix_velocity_1d', 'vix_velocity_5d', 'vix_velocity_10d', 'vix_velocity_21d',
        'vix_velocity_1d_pct', 'vix_velocity_5d_pct', 'vix_velocity_10d_pct', 'vix_velocity_21d_pct',
        'vix_accel_5d', 'vix_vol_10d', 'vix_vol_21d',
        'vix_momentum_z_10d', 'vix_momentum_z_21d', 'vix_momentum_z_63d', 'vix_term_structure'
    ],
    'regimes': [
        'vix_regime', 'days_in_regime', 'vix_displacement', 'days_since_crisis', 'elevated_flag'
    ]
}

# SPX Base Features
SPX_BASE_FEATURES = {
    'price_action': [
        'spx_lag1', 'spx_lag5', 'spx_ret_5d', 'spx_ret_10d', 'spx_ret_13d', 'spx_ret_21d', 'spx_ret_63d',
        'spx_vs_ma20', 'spx_vs_ma50', 'spx_vs_ma200', 'ma20_vs_ma50'
    ],
    'volatility': [
        'spx_realized_vol_10d', 'spx_realized_vol_21d', 'spx_realized_vol_63d', 
        'spx_vol_ratio_10_63', 'spx_skew_21d', 'spx_kurt_21d'
    ],
    'technical': [
        'bb_width_20d', 'rsi_14', 'spx_momentum_z_10d', 'spx_momentum_z_21d'
    ],
    'ohlc_microstructure': [
        'spx_body_size', 'spx_range_pct', 'spx_upper_shadow', 'spx_lower_shadow',
        'spx_close_position', 'spx_body_to_range', 'spx_is_bullish',
        'spx_gap', 'spx_gap_filled', 'spx_gap_magnitude',
        'spx_range_expansion', 'spx_range_expansion_z',
        'spx_close_pos_ma_5d', 'spx_close_pos_ma_10d', 'spx_close_pos_ma_21d',
        'spx_doji', 'spx_long_body', 'spx_upper_wick_dominant', 'spx_lower_wick_dominant',
        'spx_hammer', 'spx_shooting_star',
        'spx_intraday_momentum', 'spx_intraday_mom_ma5',
        'spx_upper_rejection', 'spx_lower_rejection'
    ]
}

# SPX-VIX Relationship Features
CROSS_ASSET_BASE_FEATURES = {
    'spx_vix_relationship': [
        'spx_vix_corr_21d', 'spx_vix_corr_63d',
        'vix_vs_rv_10d', 'vix_vs_rv_21d', 'vix_vs_rv_30d', 'vix_vs_rv_63d',
        'vix_rv_ratio_10d', 'vix_rv_ratio_21d', 'vix_rv_ratio_30d', 'vix_rv_ratio_63d',
        'spx_trend_10d', 'spx_trend_21d'
    ]
}

# CBOE Features
CBOE_BASE_FEATURES = {
    'skew_indicators': [
        'SKEW', 'SKEW_change_21d', 'SKEW_zscore_63d',
        'skew_velocity_5d', 'skew_velocity_21d', 'skew_vs_vix', 'skew_vix_ratio',
        'skew_regime', 'skew_displacement', 'tail_risk_elevated'
    ],
    'put_call_ratios': [
        'PCC', 'PCC_change_21d', 'PCC_zscore_63d',
        'PCCE', 'PCCE_change_21d', 'PCCE_zscore_63d',
        'PCCI', 'PCCI_change_21d', 'PCCI_zscore_63d',
        'pc_divergence', 'pc_equity_inst_spread', 'pc_equity_inst_divergence',
        'pcce_extreme_high', 'pcci_extreme_high', 'pc_combined_extreme',
        'pcc_velocity_10d', 'pcci_velocity_10d', 'pcc_accel_10d'
    ],
    'correlation_indices': [
        'COR1M', 'COR1M_change_21d', 'COR1M_zscore_63d',
        'COR3M', 'COR3M_change_21d', 'COR3M_zscore_63d',
        'cor_term_structure', 'cor_term_slope', 'cor_term_slope_change_21d',
        'cor_avg', 'cor_regime', 'cor_spike'
    ],
    'other_cboe': [
        'VXTH', 'VXTH_change_21d', 'VXTH_zscore_63d',
        'vxth_vs_vix', 'vxth_vix_ratio', 'vxth_premium', 'high_beta_vol_regime',
        'cboe_stress_composite', 'cboe_stress_regime'
    ]
}

# Futures Features
FUTURES_FEATURES = {
    'vix_futures': [
        'vx_spread', 'vx_spread_ma10', 'vx_spread_ma21', 
        'vx_spread_velocity_5d', 'vx_spread_velocity_21d',
        'vx_spread_regime', 'vx_spread_zscore_63d', 'vx_spread_percentile_63d',
        'vx_ratio', 'vx_ratio_ma21', 'vx_ratio_velocity_10d',
        'vx_term_structure_regime', 'vx_steep_contango', 'vx_steep_backwardation',
        'vx_curve_acceleration', 'vx_term_structure_divergence'
    ],
    'commodity_futures': [
        'cl_spread', 'cl_spread_ma10', 'cl_spread_velocity_5d', 'cl_spread_zscore_63d',
        'oil_term_regime', 'oil_steep_backwardation', 'oil_steep_contango',
        'crude_oil_ret_10d', 'crude_oil_ret_21d', 'crude_oil_ret_63d',
        'crude_oil_vol_21d', 'crude_oil_zscore_63d'
    ],
    'dollar_futures': [
        'dx_spread', 'dx_spread_ma10', 'dx_spread_velocity_5d', 'dx_spread_zscore_63d',
        'dxy_ret_10d', 'dxy_ret_21d', 'dxy_ret_63d',
        'dxy_vs_ma50', 'dxy_vs_ma200', 'dxy_vol_21d', 'dxy_regime'
    ],
    'cross_futures': [
        'vx_crude_corr_21d', 'vx_crude_divergence',
        'vx_dollar_corr_21d', 'dollar_crude_corr_21d', 'dollar_crude_corr_breakdown',
        'spx_vx_spread_corr_21d', 'spx_dollar_corr_21d'
    ]
}

# Meta Features (Derived from base features)
META_FEATURES = {
    'regime_indicators': [
        'vix_regime_micro', 'vix_regime_macro', 'regime_stability', 'regime_transition_risk',
        'vol_regime', 'risk_premium_regime', 'vol_term_regime',
        'trend_regime', 'trend_strength',
        'liquidity_stress_composite', 'liquidity_regime',
        'correlation_regime', 'composite_market_regime', 'regime_consensus'
    ],
    'cross_asset_relationships': [
        'equity_vol_divergence', 'equity_vol_corr_breakdown',
        'vol_of_vol_10d', 'vol_of_vol_21d', 'vix_acceleration',
        'risk_premium', 'risk_premium_ma21', 'risk_premium_velocity', 'risk_premium_zscore',
        'gold_spx_divergence', 'dollar_spx_correlation'
    ],
    'rate_of_change': [
        # VIX ROC
        'vix_velocity_3d', 'vix_velocity_3d_pct', 'vix_acceleration_5d', 'vix_jerk_5d', 'vix_momentum_regime',
        # SKEW ROC
        'SKEW_velocity_3d', 'SKEW_velocity_5d', 'SKEW_velocity_10d', 'SKEW_velocity_21d',
        'SKEW_acceleration_5d', 'SKEW_jerk_5d', 'SKEW_momentum_regime',
        # Others
        'spx_realized_vol_21d_velocity_3d', 'spx_realized_vol_21d_acceleration_5d',
        'vix_skew_momentum_divergence'
    ],
    'percentile_rankings': [
        # VIX percentiles
        'vix_percentile_21d', 'vix_percentile_63d', 'vix_percentile_126d', 'vix_percentile_252d',
        'vix_percentile_velocity', 'vix_extreme_high_63d', 'vix_extreme_low_63d',
        'vix_extreme_high_252d', 'vix_extreme_low_252d',
        # SKEW percentiles
        'SKEW_percentile_21d', 'SKEW_percentile_63d', 'SKEW_percentile_126d', 'SKEW_percentile_252d',
        'SKEW_percentile_velocity', 'SKEW_extreme_high_63d', 'SKEW_extreme_low_63d',
        # Risk premium percentiles
        'risk_premium_percentile_21d', 'risk_premium_percentile_63d', 'risk_premium_percentile_126d',
        'risk_premium_extreme_high_63d', 'risk_premium_extreme_low_63d'
    ]
}

# Macro Features
MACRO_FEATURES = {
    'precious_metals': [
        'Gold_lag1', 'Gold_mom_10d', 'Gold_mom_21d', 'Gold_mom_63d', 'Gold_zscore_63d',
        'Silver_lag1', 'Silver_mom_10d', 'Silver_mom_21d', 'Silver_mom_63d', 'Silver_zscore_63d'
    ],
    'commodities': [
        'Crude_Oil_lag1', 'Crude_Oil_mom_10d', 'Crude_Oil_mom_21d', 'Crude_Oil_mom_63d', 'Crude_Oil_zscore_63d'
    ],
    'currencies': [
        'Dollar_Index_lag1', 'Dollar_Index_mom_10d', 'Dollar_Index_mom_21d', 
        'Dollar_Index_mom_63d', 'Dollar_Index_zscore_63d'
    ],
    'volatility_indices': [
        'Bond_Vol_lag1', 'Bond_Vol_mom_10d', 'Bond_Vol_mom_21d', 'Bond_Vol_mom_63d', 'Bond_Vol_zscore_63d'
    ]
}

# Calendar Features
CALENDAR_FEATURES = [
    'month', 'quarter', 'day_of_week', 'day_of_month', 'is_opex_week'
]

# ==================== FEATURE GROUPS FOR ANOMALY DETECTION ====================
# These groups are designed for the isolation forest anomaly detector
# Each group captures a specific aspect of market structure

ANOMALY_FEATURE_GROUPS = {
    'vix_mean_reversion': VIX_BASE_FEATURES['mean_reversion'] + ['vix'],
    
    'vix_momentum': VIX_BASE_FEATURES['dynamics'] + [
        'vix_velocity_3d', 'vix_acceleration_5d', 'vix_jerk_5d'
    ],
    
    'vix_regime_structure': (
        VIX_BASE_FEATURES['regimes'] + 
        ['vix', 'vix_vs_ma21', 'vix_vs_ma63', 'vix_velocity_5d', 'vix_velocity_21d',
         'vix_zscore_63d', 'vix_percentile_126d', 'vix_vol_10d', 'vix_vol_21d',
         'vix_momentum_z_21d', 'vix_accel_5d', 'vix_term_structure', 'reversion_strength_63d']
    ),
    
    'cboe_options_flow': (
        CBOE_BASE_FEATURES['skew_indicators'] + 
        CBOE_BASE_FEATURES['put_call_ratios'] + 
        CBOE_BASE_FEATURES['correlation_indices']
    ),
    
    'cboe_cross_dynamics': CBOE_BASE_FEATURES['other_cboe'] + [
        'skew_velocity_5d', 'skew_velocity_21d', 'skew_vs_vix', 'skew_vix_ratio',
        'pc_equity_inst_spread', 'pc_equity_inst_divergence',
        'cor_term_slope', 'cor_term_slope_change_21d'
    ],
    
    'vix_spx_relationship': (
        CROSS_ASSET_BASE_FEATURES['spx_vix_relationship'] + 
        ['vix', 'vix_vs_ma21', 'spx_realized_vol_21d', 'spx_ret_21d', 'spx_momentum_z_21d']
    ),
    
    'spx_price_action': (
        SPX_BASE_FEATURES['price_action'] + 
        SPX_BASE_FEATURES['technical'] +
        ['spx_body_size', 'spx_range_pct', 'spx_close_position', 'spx_range_expansion', 'spx_gap']
    ),
    
    'spx_ohlc_microstructure': SPX_BASE_FEATURES['ohlc_microstructure'],
    
    'spx_volatility_regime': (
        SPX_BASE_FEATURES['volatility'] + 
        ['vix_vs_rv_10d', 'vix_vs_rv_21d', 'vix_rv_ratio_21d',
         'vix', 'vix_vol_10d', 'vix_vol_21d', 'bb_width_20d',
         'spx_ret_21d', 'spx_ret_63d', 'spx_momentum_z_21d',
         'vix_velocity_21d', 'vix_zscore_63d', 'vix_regime', 'elevated_flag',
         'spx_range_expansion', 'spx_range_expansion_z']
    ),
    
    'cross_asset_divergence': [
        'spx_vix_corr_21d', 'spx_vix_corr_63d', 'vix_vs_rv_21d', 'vix_rv_ratio_21d',
        'spx_realized_vol_21d', 'spx_ret_21d', 'spx_momentum_z_21d',
        'vix_velocity_21d', 'vix_momentum_z_21d',
        'spx_vs_ma20', 'spx_vs_ma50', 'vix_vs_ma21', 'vix_vs_ma63',
        'spx_vol_ratio_10_63', 'rsi_14', 'spx_skew_21d', 'spx_kurt_21d',
        'vix', 'vix_regime', 'vix_displacement', 'elevated_flag',
        'spx_upper_rejection', 'spx_lower_rejection', 'spx_gap_magnitude',
        'equity_vol_divergence', 'equity_vol_corr_breakdown'
    ],
    
    'tail_risk_complex': [
        'SKEW', 'skew_regime', 'skew_displacement', 'skew_vs_vix', 'tail_risk_elevated',
        'cboe_stress_composite', 'cboe_stress_regime',
        'pc_combined_extreme', 'cor_spike',
        'vix', 'vix_regime', 'vix_zscore_63d', 'vix_velocity_21d',
        'spx_upper_rejection', 'spx_hammer', 'spx_shooting_star',
        'spx_skew_21d', 'spx_kurt_21d'
    ],
    
    'futures_term_structure': (
        FUTURES_FEATURES['vix_futures'] + 
        FUTURES_FEATURES['commodity_futures'][:7] +  # Term structure only
        FUTURES_FEATURES['dollar_futures'][:4]  # Spread features only
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

# ==================== FEATURE GROUPS FOR XGBOOST REGIME CLASSIFICATION ====================
# These are comprehensive feature sets for regime prediction
# XGBoost will handle feature selection internally

REGIME_CLASSIFICATION_FEATURE_GROUPS = {
    'all_vix': (
        VIX_BASE_FEATURES['mean_reversion'] + 
        VIX_BASE_FEATURES['dynamics'] + 
        VIX_BASE_FEATURES['regimes']
    ),
    
    'all_spx': (
        SPX_BASE_FEATURES['price_action'] + 
        SPX_BASE_FEATURES['volatility'] + 
        SPX_BASE_FEATURES['technical']
    ),
    
    'all_spx_ohlc': SPX_BASE_FEATURES['ohlc_microstructure'],
    
    'all_cross_asset': CROSS_ASSET_BASE_FEATURES['spx_vix_relationship'],
    
    'all_cboe': (
        CBOE_BASE_FEATURES['skew_indicators'] + 
        CBOE_BASE_FEATURES['put_call_ratios'] + 
        CBOE_BASE_FEATURES['correlation_indices'] + 
        CBOE_BASE_FEATURES['other_cboe']
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
        MACRO_FEATURES['precious_metals'] + 
        MACRO_FEATURES['commodities'] + 
        MACRO_FEATURES['currencies'] + 
        MACRO_FEATURES['volatility_indices']
    ),
    
    'calendar': CALENDAR_FEATURES
}

# ==================== UTILITY FUNCTIONS ====================

def get_all_feature_names():
    """Get list of all possible feature names across all groups"""
    all_features = set()
    
    # Add from VIX base
    for group in VIX_BASE_FEATURES.values():
        all_features.update(group)
    
    # Add from SPX base
    for group in SPX_BASE_FEATURES.values():
        all_features.update(group)
    
    # Add from cross-asset
    for group in CROSS_ASSET_BASE_FEATURES.values():
        all_features.update(group)
    
    # Add from CBOE
    for group in CBOE_BASE_FEATURES.values():
        all_features.update(group)
    
    # Add from futures
    for group in FUTURES_FEATURES.values():
        all_features.update(group)
    
    # Add from meta
    for group in META_FEATURES.values():
        all_features.update(group)
    
    # Add from macro
    for group in MACRO_FEATURES.values():
        all_features.update(group)
    
    # Add calendar
    all_features.update(CALENDAR_FEATURES)
    
    return sorted(list(all_features))


def get_feature_group(group_name: str):
    """Get features for a specific group name"""
    # Check anomaly groups first
    if group_name in ANOMALY_FEATURE_GROUPS:
        return ANOMALY_FEATURE_GROUPS[group_name]
    
    # Check regime classification groups
    if group_name in REGIME_CLASSIFICATION_FEATURE_GROUPS:
        return REGIME_CLASSIFICATION_FEATURE_GROUPS[group_name]
    
    # Check base groups
    for category in [VIX_BASE_FEATURES, SPX_BASE_FEATURES, CROSS_ASSET_BASE_FEATURES,
                     CBOE_BASE_FEATURES, FUTURES_FEATURES, META_FEATURES, MACRO_FEATURES]:
        if group_name in category:
            return category[group_name]
    
    return []


def print_feature_summary():
    """Print summary of all feature groups"""
    print("\n" + "="*80)
    print("FEATURE CONFIGURATION SUMMARY")
    print("="*80)
    
    print("\n📊 BASE FEATURES:")
    print(f"  VIX: {sum(len(g) for g in VIX_BASE_FEATURES.values())} features across {len(VIX_BASE_FEATURES)} groups")
    print(f"  SPX: {sum(len(g) for g in SPX_BASE_FEATURES.values())} features across {len(SPX_BASE_FEATURES)} groups")
    print(f"  Cross-Asset: {sum(len(g) for g in CROSS_ASSET_BASE_FEATURES.values())} features")
    print(f"  CBOE: {sum(len(g) for g in CBOE_BASE_FEATURES.values())} features across {len(CBOE_BASE_FEATURES)} groups")
    print(f"  Futures: {sum(len(g) for g in FUTURES_FEATURES.values())} features across {len(FUTURES_FEATURES)} groups")
    
    print("\n🎯 META FEATURES:")
    for name, group in META_FEATURES.items():
        print(f"  {name}: {len(group)} features")
    
    print("\n🌍 MACRO FEATURES:")
    print(f"  Total: {sum(len(g) for g in MACRO_FEATURES.values())} features across {len(MACRO_FEATURES)} groups")
    
    print(f"\n📅 CALENDAR: {len(CALENDAR_FEATURES)} features")
    
    print("\n🔍 ANOMALY DETECTION:")
    print(f"  {len(ANOMALY_FEATURE_GROUPS)} specialized feature groups")
    
    print("\n🎲 REGIME CLASSIFICATION:")
    print(f"  {len(REGIME_CLASSIFICATION_FEATURE_GROUPS)} comprehensive feature groups")
    
    total = len(get_all_feature_names())
    print(f"\n✅ TOTAL UNIQUE FEATURES: {total}")
    print("="*80)


if __name__ == "__main__":
    print_feature_summary()