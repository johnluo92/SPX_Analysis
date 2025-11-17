# 🔧 Technical System Diagnostic Report

*Generated: 2025-11-07 22:06:43*
*Quick Mode: OFF*

---

## 1. 🏥 System Health Check

**Overall Status**: ✅ HEALTHY

### Component Status
| Component | Status | Details |
|-----------|--------|---------|
| data | ✅ | 3769 rows |
| model | ✅ | Trained |
| detector_coverage | ✅ | 0 detectors with <70% coverage |
| data_freshness | ✅ | 3 days old |
| data_completeness | ✅ | 79.1% complete |

## 2. 🎛️ Model Configuration Deep Dive

### Anomaly Detector Configuration
```python
contamination: 0.050
n_estimators: 100
max_samples: auto
random_state: 42
```

### Individual Detector Parameters
| Detector | Features Used | Coverage | Active |
|----------|--------------|----------|--------|
| vix_mean_reversion | 18 | 100.0% | ✅ |
| vix_momentum | 18 | 100.0% | ✅ |
| vix_regime_structure | 18 | 100.0% | ✅ |
| cboe_options_flow | 39 | 100.0% | ✅ |
| cboe_cross_dynamics | 17 | 100.0% | ✅ |
| vix_spx_relationship | 17 | 100.0% | ✅ |
| spx_price_action | 15 | 75.0% | ✅ |
| spx_volatility_regime | 20 | 90.9% | ✅ |
| cross_asset_divergence | 23 | 88.5% | ✅ |
| tail_risk_complex | 15 | 83.3% | ✅ |
| futures_term_structure | 27 | 100.0% | ✅ |
| macro_regime_shifts | 19 | 100.0% | ✅ |
| momentum_acceleration | 20 | 100.0% | ✅ |
| percentile_extremes | 19 | 100.0% | ✅ |
| random_4 | 0 | 0.0% | ✅ |

## 3. 🔄 Data Pipeline Flow Trace

### Data Sources → Features → Models
```
Data Fetching...........................       ✅ OK
  ↳ VIX: 3769 observations
  ↳ SPX: 3769 observations
Feature Engineering.....................       ✅ OK
  ↳ Generated 696 features
  ↳ Time period: 3769 days
Model Training..........................       ✅ OK
  ↳ 15/15 detectors trained
  ↳ Ensemble scores computed: 3769
```

### Feature Generation Summary
- **Raw Market Data Points**: 7538
- **Engineered Features**: 696
- **Final Feature Set**: 696
- **Data Reduction Ratio**: 0.1x

## 4. 📅 Data Freshness & Staleness

### Last Update Times
| Data Source | Last Update | Age (days) | Status |
|-------------|-------------|------------|--------|
| Main Features | 2025-11-04 | 3.9 | ❌ |

### ⚠️ Stale Features (>5% missing in recent data)
- **SKEW**: 100.0% missing
- **SKEW_change_21d**: 100.0% missing
- **SKEW_zscore_63d**: 100.0% missing
- **PCCI**: 100.0% missing
- **PCCI_change_21d**: 100.0% missing
- **PCCI_zscore_63d**: 100.0% missing
- **PCCE**: 100.0% missing
- **PCCE_change_21d**: 100.0% missing
- **PCCE_zscore_63d**: 100.0% missing
- **PCC**: 100.0% missing

## 5. 🎯 Current Anomaly Detection Breakdown

### Ensemble Score: 38.1%

### Detector Contributions
| Detector | Score | Weight | Weighted Score | Agreement |
|----------|-------|--------|----------------|-----------|
| spx_price_action | 81.1% | 1.00 | 81.1% | 🔴 |
| tail_risk_complex | 79.9% | 1.00 | 79.9% | 🔴 |
| spx_volatility_regime | 59.1% | 1.00 | 59.1% | 🔴 |
| vix_momentum | 55.8% | 1.00 | 55.8% | 🟢 |
| vix_regime_structure | 52.0% | 1.00 | 52.0% | 🟢 |
| percentile_extremes | 43.4% | 1.00 | 43.4% | 🟢 |
| cross_asset_divergence | 38.4% | 1.00 | 38.4% | 🟢 |
| vix_mean_reversion | 37.9% | 1.00 | 37.9% | 🟢 |
| vix_spx_relationship | 31.3% | 1.00 | 31.3% | 🟢 |
| macro_regime_shifts | 13.4% | 1.00 | 13.4% | 🔴 |
| momentum_acceleration | 8.2% | 1.00 | 8.2% | 🔴 |
| cboe_cross_dynamics | 7.7% | 1.00 | 7.7% | 🔴 |
| futures_term_structure | 5.4% | 1.00 | 5.4% | 🔴 |
| cboe_options_flow | 1.5% | 1.00 | 1.5% | 🔴 |

### Top Features Driving Current Anomaly
| Feature | Importance | Current Value | Z-Score |
|---------|-----------|---------------|---------|
| spx_vs_ma200 | 0.091 | 10.72 | 1.00 |
| spx_lag1 | 0.088 | 6851.97 | 2.70 |
| spx_ret_5d | 0.082 | -1.73 | -0.89 |
| spx_lag5 | 0.082 | 6890.89 | 2.74 |
| spx_ret_13d | 0.075 | 2.15 | 0.43 |
| spx_ret_63d | 0.070 | 6.72 | 0.54 |
| rsi_14 | 0.067 | 59.06 | 0.14 |
| spx_momentum_z_10d | 0.067 | -0.48 | -0.34 |
| spx_momentum_z_21d | 0.062 | -1.57 | -1.16 |
| bb_width_20d | 0.058 | 5.64 | -0.13 |

## 6. 🔗 Feature Correlation & Redundancy Analysis

### High Correlation Pairs (>95%)
| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| 1M_Treasury_zscore_252d | Breakeven_Inflation_10Y_zscore_252d | 1.000 |
| 1M_Treasury_zscore_252d | SOFR_90D_level | 1.000 |
| 1M_Treasury_zscore_252d | Corporate_Master_OAS_zscore_252d | 1.000 |
| 1M_Treasury_zscore_252d | High_Yield_OAS_level | 1.000 |
| 1M_Treasury_zscore_252d | CCC_High_Yield_OAS_level | 1.000 |
| 1M_Treasury_zscore_252d | 7Y_Treasury_level | 1.000 |
| 1M_Treasury_zscore_252d | BB_High_Yield_OAS_level | 1.000 |
| 1M_Treasury_zscore_252d | Yield_Curve_10Y3M_zscore_252d | 1.000 |
| 1M_Treasury_zscore_252d | 20Y_Treasury_zscore_63d | 1.000 |
| 1M_Treasury_zscore_252d | 3M_Treasury_zscore_252d | 1.000 |
| 1M_Treasury_zscore_252d | 3Y_Treasury_level | 1.000 |
| 1M_Treasury_zscore_252d | 20Y_Treasury_zscore_252d | 1.000 |
| 1M_Treasury_zscore_252d | Corporate_Master_OAS_zscore_63d | 1.000 |
| 1M_Treasury_zscore_252d | 2Y_Treasury_zscore_252d | 1.000 |
| 1M_Treasury_zscore_252d | Breakeven_Inflation_10Y_zscore_63d | 1.000 |

**Recommendation**: Consider removing 17552 redundant features to improve performance.

## 7. ⚡ Performance Profiling

### Execution Time Breakdown
| Operation | Time (ms) | % of Total |
|-----------|-----------|------------|
| batch_10_detections | 168.6 | 89.7% |
| single_detection | 19.3 | 10.3% |

## 8. 🔍 Common Failure Modes & Solutions

✅ No common failure modes detected.

## 9. 💡 What-If Scenario Analysis

### VIX Spike to 40
**Scenario**: If VIX suddenly spikes to 40 (crisis level)
**Expected Behavior**: Ensemble score would likely exceed 93% (CRITICAL threshold)
**Current System Response**: Multiple detectors would fire: vix_regime_structure, vix_momentum, cross_asset_divergence

### SKEW >150
**Scenario**: If SKEW index exceeds 150 (extreme tail risk)
**Expected Behavior**: tail_risk_complex detector triggers, ensemble score elevates
**Current System Response**: If SKEW features are available, system would classify as HIGH/CRITICAL

### All CBOE Data Missing
**Scenario**: If CBOE features become unavailable
**Expected Behavior**: System continues to function with reduced capability
**Current System Response**: 5/15 detectors would be disabled, ensemble relies on VIX/SPX/futures detectors

## 10. 📊 Data Quality Heatmap

### Feature Quality by Category
| Category | Total Features | Complete | Sparse | Missing | Quality Score |
|----------|---------------|----------|--------|---------|---------------|
| VIX | 78 | 73 | 0 | 5 | 93.6% |
| SPX | 47 | 44 | 0 | 3 | 93.6% |
| CBOE | 232 | 24 | 33 | 175 | 17.5% |
| Futures | 45 | 35 | 10 | 0 | 88.9% |
| Macro | 14 | 11 | 1 | 2 | 82.1% |
| Meta | 280 | 84 | 122 | 74 | 51.8% |

## 11. 🚀 System Optimization Recommendations

### 🟢 Optimization Opportunities
- Remove 17552 highly correlated features to reduce redundancy
- Investigate 197 stale features with high recent missing data
- Consider adding feature selection to reduce dimensionality
- Implement caching for expensive feature calculations
- Add monitoring alerts for data freshness

## 12. 📖 Quick Reference: Troubleshooting Guide


### Common Issues & Quick Fixes

**Issue**: System says "data too old"
- **Check**: `system.orchestrator.features.index[-1]`
- **Fix**: Run `system.refresh()` or retrain with fresh data

**Issue**: Ensemble score always near 0% or 100%
- **Check**: Are thresholds computed? `system.orchestrator.anomaly_detector.statistical_thresholds`
- **Fix**: Retrain system to recalculate thresholds

**Issue**: Many detectors show 0% coverage
- **Check**: CBOE files in `./CBOE_Data_Archive/`
- **Fix**: Download CBOE historical data or disable CBOE features in config

**Issue**: "Core data fetch failed" error
- **Check**: Internet connection, yfinance API status
- **Fix**: Run `system.orchestrator.fetcher.fetch_core_data(...)` separately to debug

**Issue**: High memory usage
- **Check**: Feature matrix size with `system.orchestrator.features.memory_usage(deep=True).sum()`
- **Fix**: Reduce training window in config.py (TRAINING_YEARS)

**Issue**: Slow detection speed (>1 second)
- **Check**: Number of features and detectors active
- **Fix**: Reduce features, disable low-value detectors, or enable quick_mode

**Issue**: NaN/Inf values in features
- **Check**: `system.orchestrator.features.isnull().sum()` and `np.isinf(system.orchestrator.features).sum()`
- **Fix**: Review feature_engine.py for division by zero or missing data handling


---
*Report generated in 2.34 seconds*