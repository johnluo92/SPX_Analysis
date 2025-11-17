# Market Analysis System - Current Configuration

*Generated: 2025-11-07 00:23:03*

---

## 1. System Overview

This is a **market anomaly detection system** with 3 components:

1. **Feature Engine**: Builds 374 features from market data
2. **Anomaly Detector**: 15 Isolation Forest detectors find unusual market conditions
3. **Dashboard Exporter**: Outputs JSON for web visualization

## 2. Data Sources

### Primary Market Data
- **SPX**: S&P 500 index (Yahoo Finance)
- **VIX**: CBOE Volatility Index (Yahoo Finance)

### Alternative Data
- **CBOE**: 19 series (SKEW, Put/Call ratios, correlation indices)
- **Futures**: VIX futures, crude oil, dollar index
- **Macro**: Gold, silver, bonds (Yahoo)
- **FRED**: 17 economic series (rates, inflation, GDP)

### Data Freshness
- **Last Update**: 2025-11-04
- **Training Window**: 3769 days (15.0 years)

## 3. Feature Breakdown

**Total Features**: 696

| Category | Count | Examples |
|----------|-------|----------|
| VIX Base | 78 | vix, vix_vs_ma10, vix_vs_ma10_pct |
| SPX Base | 46 | spx_lag1, spx_lag5, spx_ret_5d |
| CBOE | 232 | SKEW, SKEW_change_21d, SKEW_zscore_63d |
| Futures | 45 | vx_spread, vx_spread_ma10, vx_spread_ma21 |
| Macro | 14 | GOLDSILVER, GOLDSILVER_change_21d, Gold_lag1 |
| Meta | 34 | days_in_regime, high_beta_vol_regime, cboe_stress_regime |
| FRED | 48 | 1M_Treasury_level, 1M_Treasury_change_10d, 1M_Treasury_change_21d |

## 4. Anomaly Detection System

### Architecture
- **Method**: Isolation Forest (unsupervised outlier detection)
- **Detectors**: 15 independent models
- **Ensemble**: Weighted average of all detectors

### Detector Domains
- ✅ **vix_mean_reversion**: 18/18 features (100%)
- ✅ **vix_momentum**: 18/18 features (100%)
- ✅ **vix_regime_structure**: 18/18 features (100%)
- ✅ **cboe_options_flow**: 39/39 features (100%)
- ✅ **cboe_cross_dynamics**: 17/17 features (100%)
- ✅ **vix_spx_relationship**: 17/17 features (100%)
- ✅ **spx_price_action**: 15/20 features (75%)
- ❌ **spx_ohlc_microstructure**: 0/25 features (0%)
- ✅ **spx_volatility_regime**: 20/22 features (91%)
- ✅ **cross_asset_divergence**: 23/26 features (88%)
- ✅ **tail_risk_complex**: 15/18 features (83%)
- ✅ **futures_term_structure**: 27/27 features (100%)
- ✅ **macro_regime_shifts**: 19/19 features (100%)
- ✅ **momentum_acceleration**: 20/20 features (100%)
- ✅ **percentile_extremes**: 19/19 features (100%)

### Thresholds
- **Moderate**: >0.73
- **High**: >0.81
- **Critical**: >0.93

## 5. Current Market State

### VIX Regime
- **Current VIX**: 19.00
- **Regime**: Normal
- **Range**: 16.8 - 24.4

### Anomaly Analysis
- **Ensemble Score**: 38.1%
- **Severity**: NORMAL
- **Active Detectors**: 15

### Top Anomalies
1. **Spx Price Action**: 81.1%
2. **Tail Risk Complex**: 79.9%
3. **Spx Volatility Regime**: 59.1%
4. **Vix Momentum**: 55.8%
5. **Vix Regime Structure**: 52.0%

## 6. System Outputs

### JSON Exports
- **live_state.json**: Current anomaly state (updates on refresh)
- **historical.json**: Full training history + regime stats
- **model_cache.pkl**: Serialized models for fast refresh

### Dashboard Integration
These files are consumed by a web dashboard for visualization.

## 7. Known Limitations

1. **Not predictive**: System detects current anomalies, doesn't forecast
2. **Feature quality varies**: Some detectors have <70% feature coverage
3. **No regime forecasting**: Removed due to weak signal
4. **Historical bias**: Trained on 2010-2025, may not generalize to unprecedented events

## 8. Planned Improvements

- [ ] Add VIX futures term structure (forward-looking data)
- [ ] Implement XGBoost feature importance ranking
- [ ] Build heuristic-based spike predictor
- [ ] Export anomaly time-series for lead-lag analysis
- [ ] Prune redundant/garbage features
