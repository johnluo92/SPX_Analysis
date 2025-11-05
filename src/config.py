"""Configuration for VIX Predictor V4"""
import os
from pathlib import Path

FRED_API_KEY_PATH = Path(__file__).parent / 'json_data' / 'config.json'
CACHE_DIR = './data_cache'
CBOE_DATA_DIR = './CBOE_Data_Archive'

# Training Control
ENABLE_TRAINING = True  # Set to False to skip training in production

TRAINING_YEARS = 15
RANDOM_STATE = 42

MODEL_PARAMS = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 50, 'min_samples_leaf': 20,
                'random_state': RANDOM_STATE, 'n_jobs': -1}

SPX_FORWARD_WINDOWS = [5, 13, 21]
SPX_RANGE_THRESHOLDS = [0.02, 0.03, 0.05]

MACRO_TICKERS = {'^VIX': 'VIX', 'GLD': 'Gold', 'SLV': 'Silver', 'TLT': 'Treasuries', 'DXY': 'Dollar'}

FRED_SERIES = {'DGS10': 'Treasury_10Y', 'DGS2': 'Treasury_2Y', 'DGS5': 'Treasury_5Y', 'DGS30': 'Treasury_30Y',
               'T10Y2Y': 'Yield_Curve', 'T10Y3M': 'Yield_Curve_10Y3M', 'BAMLH0A0HYM2': 'High_Yield_Spread',
               'DEXUSEU': 'EUR_USD', 'DTWEXBGS': 'Dollar_Index', 'T5YIE': 'Inflation_Expectation_5Y',
               'T10YIE': 'Inflation_Expectation_10Y', 'DFII10': 'Real_Yield_10Y'}

COMMODITY_FRED_SERIES = {
    'DCOILWTICO': 'Crude Oil',
    'DCOILBRENTEU': 'Brent Crude',
    'DHHNGSP': 'Natural Gas',
    # 'PPIACO': 'Producer_Price_Index' # monthly data deprecated/removed
}


REGIME_BOUNDARIES = [0, 16.77, 24.40, 39.67, 100]
REGIME_NAMES = {0: "Low Vol", 1: "Normal", 2: "Elevated", 3: "Crisis"}

SKEW_ELEVATED_THRESHOLD = 145
CRISIS_VIX_THRESHOLD = 39.67

STRUCTURAL_FEATURE_GROUPS = {
    'mean_reversion': ['vix_vs_ma10', 'vix_vs_ma21', 'vix_vs_ma63', 'vix_vs_ma126', 'vix_vs_ma252',
                      'vix_vs_ma10_pct', 'vix_vs_ma21_pct', 'vix_vs_ma63_pct', 'vix_vs_ma126_pct', 'vix_vs_ma252_pct',
                      'vix_zscore_63d', 'vix_zscore_126d', 'vix_zscore_252d', 'vix_percentile_126d', 'vix_percentile_252d',
                      'reversion_strength_63d'],
    'dynamics': ['vix_velocity_1d', 'vix_velocity_5d', 'vix_velocity_10d', 'vix_velocity_21d', 'vix_velocity_1d_pct',
                'vix_velocity_5d_pct', 'vix_velocity_10d_pct', 'vix_velocity_21d_pct', 'vix_accel_5d', 'vix_vol_10d',
                'vix_vol_21d', 'vix_momentum_z_10d', 'vix_momentum_z_21d', 'vix_momentum_z_63d', 'vix_term_structure'],
    'spx_relationship': ['spx_vix_corr_21d', 'spx_vix_corr_63d', 'vix_vs_rv_10d', 'vix_vs_rv_21d', 'vix_vs_rv_30d',
                        'vix_rv_ratio_10d', 'vix_rv_ratio_21d', 'vix_rv_ratio_30d', 'spx_trend_10d', 'spx_trend_21d'],
    'cboe_institutional': ['SKEW', 'SKEW_change_21d', 'SKEW_zscore_63d', 'PCCI', 'PCCI_change_21d', 'PCCI_zscore_63d',
                          'PCCE', 'PCCE_change_21d', 'PCCE_zscore_63d', 'PCC', 'PCC_change_21d', 'PCC_zscore_63d',
                          'COR1M', 'COR1M_change_21d', 'COR1M_zscore_63d', 'COR3M', 'COR3M_change_21d', 'COR3M_zscore_63d',
                          'VXTH', 'VXTH_change_21d', 'VXTH_zscore_63d', 'pc_divergence', 'tail_risk_elevated', 'cor_term_structure'],
    'other': ['days_since_crisis']}

ANOMALY_FEATURE_GROUPS = {
    'vix_mean_reversion': ['vix', 'vix_vs_ma10', 'vix_vs_ma21', 'vix_vs_ma63', 'vix_vs_ma126', 'vix_vs_ma252',
                          'vix_vs_ma10_pct', 'vix_vs_ma21_pct', 'vix_vs_ma63_pct', 'vix_vs_ma126_pct', 'vix_vs_ma252_pct',
                          'vix_zscore_63d', 'vix_zscore_126d', 'vix_zscore_252d', 'vix_percentile_126d', 'vix_percentile_252d',
                          'reversion_strength_63d'],
    'vix_momentum': ['vix_velocity_1d', 'vix_velocity_5d', 'vix_velocity_10d', 'vix_velocity_21d', 'vix_velocity_5d_pct',
                    'vix_velocity_10d_pct', 'vix_velocity_21d_pct', 'vix_accel_5d', 'vix_vol_10d', 'vix_vol_21d',
                    'vix_momentum_z_10d', 'vix_momentum_z_21d', 'vix_momentum_z_63d', 'vix_term_structure'],
    'vix_regime_structure': ['vix', 'vix_regime', 'days_in_regime', 'vix_displacement', 'days_since_crisis', 'elevated_flag',
                            'vix_vs_ma21', 'vix_vs_ma63', 'vix_velocity_5d', 'vix_velocity_21d', 'vix_zscore_63d',
                            'vix_percentile_126d', 'vix_vol_10d', 'vix_vol_21d', 'vix_momentum_z_21d', 'vix_accel_5d',
                            'vix_term_structure', 'reversion_strength_63d'],
    'cboe_options_flow': ['SKEW', 'SKEW_change_21d', 'SKEW_zscore_63d', 'COR1M', 'COR1M_change_21d', 'COR1M_zscore_63d',
                         'COR3M', 'COR3M_change_21d', 'COR3M_zscore_63d', 'PCC', 'PCC_change_21d', 'PCC_zscore_63d',
                         'PCCE', 'PCCE_change_21d', 'PCCE_zscore_63d', 'PCCI', 'PCCI_change_21d', 'PCCI_zscore_63d',
                         'VXTH', 'VXTH_change_21d', 'VXTH_zscore_63d', 'pc_divergence', 'tail_risk_elevated', 'cor_term_structure'],
    'vix_spx_relationship': ['spx_vix_corr_21d', 'spx_vix_corr_63d', 'vix_vs_rv_10d', 'vix_vs_rv_21d', 'vix_vs_rv_30d',
                            'vix_rv_ratio_10d', 'vix_rv_ratio_21d', 'vix_rv_ratio_30d', 'spx_trend_10d', 'spx_trend_21d',
                            'vix', 'vix_vs_ma21', 'spx_realized_vol_21d', 'spx_ret_21d', 'spx_momentum_z_21d'],
    'spx_price_action': ['spx_lag1', 'spx_lag5', 'spx_ret_5d', 'spx_ret_10d', 'spx_ret_13d', 'spx_ret_21d', 'spx_ret_63d',
                        'spx_vs_ma20', 'spx_vs_ma50', 'spx_vs_ma200', 'ma20_vs_ma50', 'spx_momentum_z_10d', 'spx_momentum_z_21d',
                        'bb_width_20d', 'rsi_14', 'spx_skew_21d', 'spx_kurt_21d'],
    'spx_volatility_regime': ['spx_realized_vol_10d', 'spx_realized_vol_21d', 'spx_realized_vol_63d', 'spx_vol_ratio_10_63',
                             'vix_vs_rv_10d', 'vix_vs_rv_21d', 'vix_rv_ratio_21d', 'vix', 'vix_vol_10d', 'vix_vol_21d',
                             'bb_width_20d', 'spx_ret_21d', 'spx_ret_63d', 'spx_momentum_z_21d', 'vix_velocity_21d',
                             'vix_zscore_63d', 'vix_regime', 'elevated_flag'],
    'macro_rates': ['Treasury_10Y_level', 'Treasury_2Y_level', 'Treasury_5Y_level', 'Treasury_30Y_level',
                   'Treasury_10Y_change_10d', 'Treasury_10Y_change_21d', 'Treasury_10Y_change_63d', 'Treasury_2Y_change_10d',
                   'Treasury_2Y_change_21d', 'Treasury_5Y_change_10d', 'Treasury_5Y_change_21d', 'Treasury_30Y_change_10d',
                   'Treasury_30Y_change_21d', 'Treasury_10Y_zscore_63d', 'Treasury_10Y_zscore_252d', 'Treasury_2Y_zscore_63d',
                   'Treasury_5Y_zscore_63d', 'Yield_Curve_level', 'Yield_Curve_change_21d', 'Yield_Curve_zscore_63d',
                   'Yield_Curve_10Y3M_level', 'Yield_Curve_10Y3M_change_21d', 'Real_Yield_10Y_level', 'Real_Yield_10Y_change_21d',
                   'Inflation_Expectation_5Y_level', 'Inflation_Expectation_5Y_change_21d', 'Inflation_Expectation_10Y_level',
                   'Inflation_Expectation_10Y_change_21d', 'High_Yield_Spread_level', 'High_Yield_Spread_change_21d'],
    'commodities_stress': ['Gold_lag1', 'Gold_mom_10d', 'Gold_mom_21d', 'Gold_mom_63d', 'Gold_zscore_63d', 'Silver_mom_21d',
                          'Silver_zscore_63d', 'Crude Oil_mom_21d', 'Crude Oil_zscore_63d', 'Brent Crude_mom_21d',
                          'Brent Crude_zscore_63d', 'Natural Gas_mom_21d', 'Natural Gas_zscore_63d', 'Dollar_mom_21d',
                          'Dollar_zscore_63d', 'Producer_Price_Index_mom_21d', 'Producer_Price_Index_zscore_63d'],
    'cross_asset_divergence': ['spx_vix_corr_21d', 'spx_vix_corr_63d', 'vix_vs_rv_21d', 'vix_rv_ratio_21d',
                              'spx_realized_vol_21d', 'spx_ret_21d', 'spx_momentum_z_21d', 'vix_velocity_21d', 'vix_momentum_z_21d',
                              'spx_vs_ma20', 'spx_vs_ma50', 'vix_vs_ma21', 'vix_vs_ma63', 'spx_vol_ratio_10_63', 'rsi_14',
                              'spx_skew_21d', 'spx_kurt_21d', 'vix', 'vix_regime', 'vix_displacement', 'elevated_flag']}

ANOMALY_DETECTOR_PARAMS = {'contamination': 0.01, 'n_estimators': 100, 'max_samples': 'auto', 'random_state': RANDOM_STATE}

SHAP_CONFIG = {'method': 'shap', 'n_samples': 500, 'n_repeats': 1, 'use_shap_if_available': True, 'timeout_warning': True}

DURATION_PREDICTOR_PARAMS = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 50, 'min_samples_leaf': 20,
                             'random_state': RANDOM_STATE, 'n_jobs': -1, 'max_duration_cap': 30}

SIGNAL_THRESHOLDS = {'anomaly': {'extreme': 88, 'high': 78, 'moderate': 70},
                    'duration': {'extended_multiplier': 1.5, 'fresh_threshold': 'median'},
                    'displacement': {'significant': 1.5}}

# Backward compatibility - these should NOT be used in new code
ANOMALY_THRESHOLDS = {'high_anomaly': 0.78, 'severity_extreme': 0.88, 'severity_high': 0.78,
                     'severity_moderate': 0.70, 'detector_coverage_min': 0.8}

POSITION_SIZING = {
    'AGGRESSIVE': {'description': 'Maximum position size', 'iron_condor_notional': 1.0, 'delta_hedge': 'Minimal', 'stop_loss': 'Wide'},
    'MODERATE': {'description': 'Standard position size', 'iron_condor_notional': 0.6, 'delta_hedge': 'Standard', 'stop_loss': 'Standard'},
    'LIGHT': {'description': 'Reduced position size', 'iron_condor_notional': 0.3, 'delta_hedge': 'Increased', 'stop_loss': 'Tight'},
    'OPPORTUNISTIC': {'description': 'Scale in over multiple days', 'iron_condor_notional': 0.5, 'delta_hedge': 'Dynamic', 'stop_loss': 'Trailing'}}