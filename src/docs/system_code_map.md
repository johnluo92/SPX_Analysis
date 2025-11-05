# VIX Anomaly Detection System - Code Navigation Map

**Version 4.0** | Last Updated: November 2025

---

## 🎯 Purpose of This Document

This is your **code GPS**. When you need to:
- Add a new feature → See [§3.2 Adding Features](#32-adding-features)
- Add raw data source → See [§2.1 Data Fetcher](#21-data-fetcher-coredatafetcherpy)
- Modify anomaly detector → See [§2.2 Anomaly Detector](#22-anomaly-detector-coreanomalydetectorpy)
- Change exports → See [§2.5 Unified Exporter](#25-unified-exporter-exportunifiedexporterpy)
- Debug live refresh → See [§4.2 Refresh Cycle](#42-refresh-cycle-live-updates)

**No need to re-paste the entire codebase.** Just reference this map.

---

## 📁 1. System Architecture Overview

```
src/
├── core/                          # Core ML/data modules
│   ├── data_fetcher.py           # Yahoo Finance + FRED API
│   ├── feature_engine.py         # Feature engineering
│   ├── anomaly_detector.py       # 15-detector system
│   └── predictor.py              # VIX predictor wrapper
├── export/
│   └── unified_exporter.py       # JSON export logic
├── config.py                      # All constants/thresholds
├── integrated_system_production.py  # Main orchestration
├── dashboard_orchestrator.py     # Web server + auto-refresh
├── data_service.js               # Frontend data layer
└── json_data/                    # Output directory
    ├── live_state.json           # Real-time state (15min)
    ├── historical.json           # Static training data
    └── model_cache.pkl           # Cached models
```

### Data Flow Diagram

```
Yahoo/FRED APIs
      ↓
data_fetcher.py (cache + fetch)
      ↓
feature_engine.py (200+ features)
      ↓
anomaly_detector.py (15 Isolation Forests)
      ↓
predictor.py (VIX + anomaly wrapper)
      ↓
unified_exporter.py (3 unified files)
      ↓
JSON files → data_service.js → Dashboard HTML
```

---

## 🔧 2. Core Module Reference

### 2.1 Data Fetcher (`core/data_fetcher.py`)

**Purpose**: Fetch historical/live data from Yahoo Finance and FRED  
**Class**: `UnifiedDataFetcher`

#### Key Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `fetch_yahoo(ticker, start, end)` | Get Yahoo Finance data | `fetch_yahoo('^VIX', '2020-01-01', '2025-11-03')` |
| `fetch_fred(series_id, start, end)` | Get FRED data | `fetch_fred('DGS10', '2020-01-01', '2025-11-03')` |
| `fetch_price(ticker)` | Live price | `fetch_price('^VIX')` |
| `fetch_fred_multiple()` | All FRED series | Returns DataFrame with all series |

#### Adding a New Data Source

**Example: Add Bitcoin data**

```python
# In config.py, add to MACRO_TICKERS:
MACRO_TICKERS = {
    '^VIX': 'VIX',
    'BTC-USD': 'Bitcoin',  # NEW
    ...
}

# Data fetcher will automatically pull it via fetch_macro()
# No code changes needed in data_fetcher.py!
```

**Example: Add new FRED series (unemployment rate)**

```python
# In config.py:
FRED_SERIES = {
    'DGS10': 'Treasury_10Y',
    'UNRATE': 'Unemployment_Rate',  # NEW
    ...
}

# System will auto-fetch via fetch_fred_multiple()
```

#### Cache Behavior

- **Location**: `data_cache/` directory
- **Format**: Parquet files (fast binary)
- **TTL**: 90 days for historical data, daily refresh for live
- **Invalidation**: Automatic on FRED revisions (ETag checking)

---

### 2.2 Anomaly Detector (`core/anomaly_detector.py`)

**Purpose**: 15 independent Isolation Forests across market domains  
**Class**: `MultiDimensionalAnomalyDetector`

#### Architecture

```
10 Domain Detectors (explicit features)
  ├─ vix_mean_reversion
  ├─ vix_momentum
  ├─ vix_regime_structure
  ├─ cboe_options_flow
  ├─ vix_spx_relationship
  ├─ spx_price_action
  ├─ spx_volatility_regime
  ├─ macro_rates
  ├─ commodities_stress
  └─ cross_asset_divergence

5 Random Subspace Detectors (25 random features each)
  └─ Catch patterns not explicitly modeled
```

#### Key Methods

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `train(features, verbose=True)` | Train all 15 detectors | Training mode only |
| `detect(features, verbose=False)` | Get anomaly scores | Every refresh cycle |
| `calculate_statistical_thresholds()` | Compute percentile thresholds | After training |
| `classify_anomaly(score, method='statistical')` | Map score → severity | Display logic |

#### Modifying Detectors

**Example: Change contamination rate**

```python
# In anomaly_detector.py, __init__():
self.contamination = contamination  # Default: 0.01 (1%)

# Higher contamination = more observations flagged as anomalies
# Lower contamination = only extreme outliers flagged
```

**Example: Add 11th domain detector**

```python
# Step 1: Define feature group in config.py
ANOMALY_FEATURE_GROUPS = {
    'vix_mean_reversion': [...],
    'crypto_volatility': [  # NEW
        'Bitcoin_lag1',
        'Bitcoin_mom_21d',
        'Bitcoin_zscore_63d',
        ...
    ]
}

# Step 2: No code changes needed!
# System auto-trains detector if features exist in data
```

#### Threshold Calculation

Thresholds are **statistically derived** (not hardcoded):

```python
# From historical ensemble scores:
moderate = 85th percentile  # ~0.70
high = 92nd percentile      # ~0.78
critical = 98th percentile  # ~0.88

# Bootstrap confidence intervals (1000 iterations)
# Stored in: statistical_thresholds dict
```

---

### 2.3 Feature Engine (`core/feature_engine.py`)

**Purpose**: Transform raw prices into 200+ engineered features  
**Class**: `UnifiedFeatureEngine`

#### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| VIX Mean Reversion | 16 | `vix_vs_ma21`, `vix_zscore_63d`, `vix_percentile_252d` |
| VIX Dynamics | 15 | `vix_velocity_5d`, `vix_accel_5d`, `vix_vol_21d` |
| VIX Regimes | 7 | `vix_regime`, `days_in_regime`, `days_since_crisis` |
| SPX Price Action | 17 | `spx_ret_21d`, `spx_vs_ma50`, `rsi_14` |
| SPX Volatility | 8 | `spx_realized_vol_21d`, `vix_rv_ratio_21d` |
| SPX-VIX Relationship | 10 | `spx_vix_corr_21d`, `vix_vs_rv_21d` |
| CBOE Indicators | 24 | `SKEW`, `PCCI`, `COR1M`, `VXTH` |
| FRED Macro | 60+ | `Treasury_10Y_zscore_63d`, `Yield_Curve_change_21d` |
| Commodities | 21 | `Gold_mom_21d`, `Crude Oil_zscore_63d` |
| Calendar | 5 | `month`, `is_opex_week` |

#### Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `build_complete_features(years=15)` | Full pipeline | Dict with `features`, `vix`, `spx`, `dates` |
| `_vix_mean_reversion(vix)` | VIX vs MAs | DataFrame |
| `_vix_dynamics(vix)` | Velocity/momentum | DataFrame |
| `_cboe_features(cboe_data)` | Options flow | DataFrame |
| `_fred_features(fred)` | Macro indicators | DataFrame |

#### Adding New Features

**Example: Add VIX implied correlation (VIX3M/VIX ratio)**

```python
# In feature_engine.py, add new method:
def _vix_term_structure_features(self, vix: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=vix.index)
    
    # Fetch VIX3M from Yahoo
    vix3m = self.fetcher.fetch_yahoo_series('^VIX3M', 'Close', 
                                            vix.index[0], vix.index[-1])
    vix3m = vix3m.reindex(vix.index, method='ffill')
    
    # Calculate features
    features['vix_vix3m_ratio'] = vix / vix3m
    features['vix_vix3m_spread'] = vix - vix3m
    features['vix_vix3m_ratio_zscore_63d'] = (
        (features['vix_vix3m_ratio'] - 
         features['vix_vix3m_ratio'].rolling(63).mean()) /
        features['vix_vix3m_ratio'].rolling(63).std()
    )
    
    return features

# In build_complete_features(), add to feature_groups list:
feature_groups = [
    self._vix_mean_reversion(vix),
    self._vix_term_structure_features(vix),  # NEW
    ...
]
```

**Example: Modify existing feature window**

```python
# Change: VIX z-score from 63-day to 30-day
# In _vix_mean_reversion():

# OLD:
for w in [63, 126, 252]:
    ma, std = vix.rolling(w).mean().shift(1), vix.rolling(w).std().shift(1)
    features[f'vix_zscore_{w}d'] = (vix - ma) / std

# NEW:
for w in [30, 63, 126, 252]:  # Added 30-day
    ma, std = vix.rolling(w).mean().shift(1), vix.rolling(w).std().shift(1)
    features[f'vix_zscore_{w}d'] = (vix - ma) / std
```

---

### 2.4 Predictor (`core/predictor.py`)

**Purpose**: High-level wrapper for anomaly detection + VIX prediction  
**Class**: `VIXPredictorV4`

#### Key Methods

| Method | Purpose | Use Case |
|--------|---------|----------|
| `train_with_features(features, vix, spx)` | Train all models | Training mode |
| `load_refresh_state(filepath)` | Load cached models | Cached mode (daily) |
| `export_refresh_state(filepath)` | Save models for caching | After training |
| `_compute_regime_statistics(vix)` | Regime transitions | Historical stats |

#### Operating Modes

**Training Mode** (weekly recommended):
```python
system = IntegratedMarketSystemV4()
system.train(years=15)  # ~3 minutes
```

**Cached Mode** (daily):
```python
system = IntegratedMarketSystemV4()
system.vix_predictor.load_refresh_state('./json_data/model_cache.pkl')  # ~5 seconds
```

---

### 2.5 Unified Exporter (`export/unified_exporter.py`)

**Purpose**: Single source of truth for all JSON exports  
**Class**: `UnifiedExporter`

#### Output Files (3 total)

| File | Size | Update Frequency | Contents |
|------|------|------------------|----------|
| `live_state.json` | 15 KB | Every 15min | Market snapshot + anomaly state |
| `historical.json` | 300 KB | Training only | Full historical scores + regime stats |
| `model_cache.pkl` | 15 MB | Training only | Trained detectors + scalers |

#### Key Methods

| Method | Purpose | When Called |
|--------|---------|-------------|
| `export_live_state()` | Real-time market state | Every refresh |
| `export_historical_context()` | Training data | Once during training |
| `export_model_cache()` | Pickle models | Once during training |

#### Data Contracts (Schemas)

**live_state.json structure:**
```json
{
  "schema_version": "2.0.0",
  "market": {
    "vix": 17.17,
    "spx": 6840.20,
    "regime": "Normal",
    "regime_days": 3
  },
  "anomaly": {
    "ensemble_score": 0.477,
    "classification": "NORMAL",
    "active_detectors": ["spx_price_action", "commodities_stress"],
    "detector_scores": {...}
  },
  "persistence": {
    "current_streak": 0,
    "mean_duration": 3.61,
    "max_duration": 56
  }
}
```

#### Adding New Export Fields

**Example: Add "market sentiment" to live_state.json**

```python
# In unified_exporter.py, export_live_state():

# After creating market snapshot:
market = MarketSnapshot(...)

# NEW: Add sentiment calculation
vix = vix_predictor.vix.iloc[-1]
skew = vix_predictor.features['SKEW'].iloc[-1] if 'SKEW' in vix_predictor.features else 135

sentiment = {
    'fear_greed_index': (vix - 15) / 50 * 100,  # Simple proxy
    'tail_risk': 'elevated' if skew > 145 else 'normal',
    'regime_stress': vix / regime_mean if regime_mean > 0 else 1.0
}

# Add to live_state dict:
live_state = {
    "market": asdict(market),
    "anomaly": asdict(anomaly),
    "sentiment": sentiment,  # NEW
    ...
}
```

---

## 🎛️ 3. Configuration & Constants

### 3.1 Config File (`config.py`)

**All magic numbers live here.** Never hardcode thresholds in modules.

#### Key Constants

| Constant | Type | Purpose |
|----------|------|---------|
| `TRAINING_YEARS` | int | Historical window (default: 15) |
| `RANDOM_STATE` | int | Reproducibility seed (42) |
| `REGIME_BOUNDARIES` | list | VIX regime cutoffs [0, 16.77, 24.40, 39.67, 100] |
| `ANOMALY_FEATURE_GROUPS` | dict | Feature lists per detector |
| `FRED_SERIES` | dict | FRED series IDs → names |
| `MACRO_TICKERS` | dict | Yahoo tickers → names |

#### Changing Regime Boundaries

```python
# OLD:
REGIME_BOUNDARIES = [0, 16.77, 24.40, 39.67, 100]
# Regimes: Low Vol | Normal | Elevated | Crisis

# NEW (more granular):
REGIME_BOUNDARIES = [0, 15, 18, 22, 30, 45, 100]
# Regimes: Low | Normal | Moderate | Elevated | High | Crisis

# ⚠️ WARNING: Retrain required after changing boundaries
```

---

### 3.2 Adding Features

**Full workflow example: Add "gamma exposure" feature**

**Step 1: Add data source**
```python
# config.py
MACRO_TICKERS = {
    '^VIX': 'VIX',
    'VOLGAMMA': 'GammaExposure',  # NEW (if available as ticker)
}
```

**Step 2: Create feature**
```python
# core/feature_engine.py
def _gamma_features(self, spx_vol: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=spx_vol.index)
    
    # Estimate gamma exposure (simplified)
    features['gamma_estimate'] = spx_vol.rolling(21).std() * 100
    features['gamma_regime'] = pd.cut(features['gamma_estimate'], 
                                      bins=[0, 5, 10, 20, 100], 
                                      labels=[0, 1, 2, 3])
    features['gamma_zscore_63d'] = (
        (features['gamma_estimate'] - 
         features['gamma_estimate'].rolling(63).mean()) /
        features['gamma_estimate'].rolling(63).std()
    )
    
    return features

# Add to build_complete_features():
feature_groups.append(self._gamma_features(spx_realized_vol))
```

**Step 3: Add to detector group**
```python
# config.py
ANOMALY_FEATURE_GROUPS = {
    'spx_volatility_regime': [
        'spx_realized_vol_21d',
        'gamma_estimate',        # NEW
        'gamma_zscore_63d',      # NEW
        ...
    ]
}
```

**Step 4: Retrain**
```python
# In config.py:
ENABLE_TRAINING = True

# Run:
python dashboard_orchestrator.py
```

---

## 🚀 4. Execution Modes

### 4.1 Training Mode (Weekly)

**When:** Monday morning, or after adding features  
**Duration:** ~3 minutes  
**Memory:** ~500 MB peak

```bash
# Set in config.py:
ENABLE_TRAINING = True

# Run:
python dashboard_orchestrator.py --years 15 --port 8000
```

**What happens:**
1. Fetch 15 years of data (Yahoo + FRED)
2. Engineer 200+ features
3. Train 15 Isolation Forests
4. Calculate statistical thresholds (bootstrap CIs)
5. Export 3 unified files (live/historical/cache)
6. Launch dashboard on port 8000

---

### 4.2 Refresh Cycle (Live Updates)

**Frequency:** Every 15 minutes during market hours  
**Duration:** ~5 seconds  
**Memory growth:** ~0.5 MB per cycle (normal)

**How it works:**

```python
# In dashboard_orchestrator.py, _attempt_refresh():

1. Fetch live VIX/SPX prices
   ├─ fetch_price('^VIX')  # e.g., 17.17
   └─ fetch_price('^GSPC') # e.g., 6840.20

2. Update last row of historical series
   ├─ vix_ml.iloc[-1] = live_vix
   └─ spx_ml.iloc[-1] = live_spx

3. Recalculate derived features (in-memory)
   ├─ vix_vs_ma21 = live_vix - ma21
   ├─ vix_zscore_63d = (live_vix - mu63) / std63
   └─ spx_momentum_z_21d = ...

4. Re-run anomaly detection (cached models)
   └─ detect(features.iloc[[-1]])

5. Export live_state.json only
   └─ historical.json unchanged
```

**Key optimization:**
- No retraining (uses cached models from model_cache.pkl)
- No data fetching (except live prices)
- Only last row recalculated

---

### 4.3 Cached Mode (Daily)

**When:** Tuesday-Friday (after Monday training)  
**Duration:** ~5 seconds startup  
**Memory:** ~150 MB baseline

```bash
# Set in config.py:
ENABLE_TRAINING = False

# Run:
python dashboard_orchestrator.py --skip-training
```

**Loads from cache:**
- `model_cache.pkl` → 15 trained detectors
- `historical.json` → Static regime stats
- `live_state.json` → Last known state

**Then starts auto-refresh** (see §4.2)

---

## 🐛 5. Debugging Guide

### 5.1 Common Issues

#### "FRED API key not found"

**Symptom:** `âš ï¸  FRED unavailable` during training

**Fix:**
```bash
# Create json_data/config.json:
{
  "fred_api_key": "your_key_here"
}

# Or set environment variable:
export FRED_API_KEY="your_key_here"

# Get key: https://fred.stlouisfed.org/docs/api/api_key.html
```

---

#### "Detector not trained"

**Symptom:** `ValueError: Detector not trained. Call init() first.`

**Cause:** Tried to run cached mode without prior training

**Fix:**
```python
# First time setup:
ENABLE_TRAINING = True
python dashboard_orchestrator.py  # Creates model_cache.pkl

# Subsequent runs:
ENABLE_TRAINING = False
python dashboard_orchestrator.py --skip-training
```

---

#### "Memory growth warning"

**Symptom:** `âš ï¸  Memory growth: +75 MB (current: 225 MB)`

**Cause:** Memory leak in refresh loop (unlikely but possible)

**Fix:**
```python
# In integrated_system_production.py:
if self.memory_monitoring_enabled:
    mem_report = self.get_memory_report()
    if mem_report['status'] == 'CRITICAL':
        # Restart system
        print("Critical memory growth - restarting...")
        sys.exit(1)  # Process manager will restart
```

---

### 5.2 Validation Tools

#### Check feature coverage

```python
# In Python console:
from integrated_system_production import IntegratedMarketSystemV4

system = IntegratedMarketSystemV4()
system.train(years=15)

# Automatic coverage report printed:
# âœ… vix_mean_reversion: 16/16 (100.0%)
# âš ï¸  cboe_options_flow: 18/24 (75.0%)
```

---

#### Validate exported JSONs

```bash
# Check schema versions:
cat json_data/live_state.json | jq '.schema_version'
# Output: "2.0.0"

cat json_data/historical.json | jq '.schema_version'
# Output: "2.0.0"

# Check file sizes:
ls -lh json_data/
# live_state.json: ~15 KB
# historical.json: ~300 KB
# model_cache.pkl: ~15 MB
```

---

#### Test anomaly detector

```python
# In Python console:
from core.anomaly_detector import MultiDimensionalAnomalyDetector
import pandas as pd
import numpy as np

# Create synthetic data
features = pd.DataFrame({
    'vix': np.random.uniform(15, 25, 1000),
    'vix_vs_ma21': np.random.normal(0, 2, 1000),
    'spx_ret_21d': np.random.normal(1, 3, 1000)
})

# Train detector
detector = MultiDimensionalAnomalyDetector(contamination=0.05)
detector.train(features, verbose=True)

# Test detection
result = detector.detect(features.iloc[[-1]], verbose=True)
print(f"Ensemble score: {result['ensemble']['score']:.3f}")
```

---

## 📊 6. Performance Characteristics

### 6.1 Timing Benchmarks

| Operation | Duration | Frequency |
|-----------|----------|-----------|
| Full training (15 years) | ~3 min | Weekly |
| Load cached models | ~5 sec | Daily |
| Live refresh (fetch + detect) | ~5 sec | Every 15 min |
| Feature engineering (1 row) | <1 ms | Every refresh |
| Anomaly detection (1 row) | ~10 ms | Every refresh |
| JSON export | ~50 ms | Every refresh |

---

### 6.2 Memory Profile

| Component | Memory (MB) |
|-----------|-------------|
| Baseline (Python + imports) | 100 |
| Loaded data (15 years) | 50 |
| Trained models (15 detectors) | 150 |
| Feature cache (last 252 days) | 5 |
| **Total steady-state** | **305** |

**Expected growth:** ~0.5 MB per refresh cycle  
**Warning threshold:** +50 MB from baseline  
**Critical threshold:** +200 MB from baseline

---

## 🔍 7. Advanced Topics

### 7.1 Custom Detectors

**Example: Add "crypto volatility" detector**

```python
# Step 1: Fetch Bitcoin data
# In data_fetcher.py, add to fetch_macro():
crypto = self.fetch_yahoo_series('BTC-USD', 'Close', start_date, end_date)
series_list.append(crypto)

# Step 2: Engineer features
# In feature_engine.py:
def _crypto_features(self, btc: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=btc.index)
    features['btc_ret_21d'] = btc.pct_change(21) * 100
    features['btc_vol_21d'] = btc.pct_change().rolling(21).std() * np.sqrt(252) * 100
    features['btc_vs_ma50'] = ((btc - btc.rolling(50).mean()) / btc.rolling(50).mean()) * 100
    return features

# Step 3: Define detector feature group
# In config.py:
ANOMALY_FEATURE_GROUPS = {
    'crypto_volatility': [
        'btc_ret_21d',
        'btc_vol_21d',
        'btc_vs_ma50'
    ]
}

# Step 4: Train (system auto-creates 16th detector)
```

---

### 7.2 Threshold Customization

**Bootstrap confidence intervals** (already implemented):

```python
# In anomaly_detector.py:
thresholds = self.calculate_statistical_thresholds_with_ci(
    n_bootstrap=1000,  # Increase for more precision
    confidence_level=0.95  # 95% CI
)

# Outputs:
# Moderate: 0.70 [0.68, 0.72]  (CI width: 0.04)
# High:     0.78 [0.76, 0.80]  (CI width: 0.04)
# Critical: 0.88 [0.86, 0.90]  (CI width: 0.04)
```

**To use different percentiles:**

```python
# In anomaly_detector.py, calculate_statistical_thresholds():
thresholds = {
    'moderate': float(np.percentile(scores, 80)),  # Changed from 85
    'high': float(np.percentile(scores, 90)),      # Changed from 92
    'critical': float(np.percentile(scores, 95))   # Changed from 98
}
```

---

### 7.3 Feature Importance

**SHAP values** (if installed):

```python
# Automatic if `pip install shap` installed
# In anomaly_detector.py, _calculate_feature_importance():

if SHAP_AVAILABLE:
    explainer = shap.TreeExplainer(detector)
    shap_values = explainer.shap_values(X_sample)
    # Returns directional feature importance
```

**Permutation importance** (fallback):

```python
# Always available (no dependencies)
# Shuffles each feature, measures score impact
importances = self._calculate_permutation_importance(detector, X_scaled, ...)
```

---

## 🎓 8. Quick Reference

### 8.1 File Locations Cheat Sheet

| Need to... | Edit this file |
|------------|----------------|
| Change VIX regime boundaries | `config.py` → `REGIME_BOUNDARIES` |
| Add new data source | `config.py` → `MACRO_TICKERS` or `FRED_SERIES` |
| Create new feature | `core/feature_engine.py` → Add method + call in `build_complete_features()` |
| Modify detector | `core/anomaly_detector.py` → Change `contamination`, `n_estimators`, etc. |
| Add export field | `export/unified_exporter.py` → Modify `export_live_state()` |
| Change refresh interval | `dashboard_orchestrator.py` → `REFRESH_INTERVAL` |
| Adjust thresholds | `core/anomaly_detector.py` → `calculate_statistical_thresholds()` |

---

### 8.2 Command Cheat Sheet

```bash
# Training mode (full retrain)
ENABLE_TRAINING=True python dashboard_orchestrator.py

# Cached mode (daily)
ENABLE_TRAINING=False python dashboard_orchestrator.py --skip-training

# Change port
python dashboard_orchestrator.py --port 9000

# Disable auto-refresh
python dashboard_orchestrator.py --no-refresh

# Custom refresh interval (30 min = 1800 sec)
python dashboard_orchestrator.py --refresh-interval 1800

# Debug mode
python integrated_system_production.py  # See console output
```

---

### 8.3 JSON Schema Quick Lookup

**live_state.json fields:**
```
.market.vix
.market.spx
.market.regime
.anomaly.ensemble_score
.anomaly.classification
.anomaly.active_detectors[]
.anomaly.detector_scores{}
.persistence.current_streak
.persistence.mean_duration
```

**historical.json fields:**
```
.historical.dates[]
.historical.ensemble_scores[]
.historical.spx_close[]
.historical.regime_stats.regimes[].statistics.mean_duration
.historical.thresholds.moderate / high / critical
.attribution{detector_name}[].feature / importance
```

---

## 🆘 9. Getting Unstuck

### "I want to add X but don't know where to start"

1. **Adding data?** → Start at `config.py` (add ticker/series ID)
2. **Adding feature?** → Start at `core/feature_engine.py` (create method)
3. **Changing detector?** → Start at `config.py` (modify `ANOMALY_FEATURE_GROUPS`)
4. **Changing output?** → Start at `export/unified_exporter.py`

---

### "System not behaving as expected"

**Check in this order:**

1. **Config flags**: Is `ENABLE_TRAINING` set correctly?
2. **Cache validity**: Delete `model_cache.pkl` and retrain
3. **Data freshness**: Check `data_cache/_cache_metadata.json` for stale entries
4. **Feature coverage**: Run training mode with `verbose=True` to see coverage %
5. **Memory**: Check `system.get_memory_report()` for leaks

---

### "Feature not appearing in detector"

**Diagnostic checklist:**

```python
# 1. Verify feature exists in DataFrame
print(system.vix_predictor.features.columns.tolist())
# Should contain your feature name

# 2. Check if feature group references it
from config import ANOMALY_FEATURE_GROUPS
print(ANOMALY_FEATURE_GROUPS['your_detector_name'])
# Should contain your feature

# 3. Verify feature has values (not all NaN)
print(system.vix_predictor.features['your_feature'].describe())

# 4. Check detector coverage
print(system.vix_predictor.anomaly_detector.detector_coverage)
# Should show >80% for your detector
```

---

### "Need to understand data flow for specific value"

**Example: Tracing VIX z-score calculation**

```
1. Raw VIX price
   └─ data_fetcher.py: fetch_vix() → pd.Series with daily closes

2. Feature engineering
   └─ feature_engine.py: _vix_mean_reversion()
      └─ vix_zscore_63d = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()

3. Anomaly detection input
   └─ anomaly_detector.py: detect(features)
      └─ vix_zscore_63d is in vix_mean_reversion feature group

4. Scaled for detector
   └─ RobustScaler.transform() → removes outliers, scales to [-1, 1] range

5. Isolation Forest score
   └─ detector.score_samples() → raw anomaly score

6. Percentile normalization
   └─ Compare to training distribution → final 0-1 score

7. Export
   └─ unified_exporter.py: export_live_state()
      └─ detector_scores['vix_mean_reversion'] in JSON
```

---

## 📐 10. Architecture Decisions (The "Why")

### 10.1 Why 15 detectors instead of 1?

**Reasoning:**
- Single detector can't capture multi-dimensional market stress
- Different regimes activate different detectors:
  - VIX spike → `vix_momentum` fires
  - Credit stress → `macro_rates` fires
  - Options positioning → `cboe_options_flow` fires
- Ensemble provides robustness (consensus of specialists)

**Trade-off:**
- **Pro**: Better coverage, interpretability (which domain is stressed)
- **Con**: More complex, harder to tune, requires more features

---

### 10.2 Why Isolation Forest over other algorithms?

**Reasoning:**
- No assumptions about data distribution (vs. Gaussian-based methods)
- Handles high-dimensional data well (200+ features)
- Fast training and inference
- Works with imbalanced data (anomalies are rare by definition)

**Alternatives considered:**
- **One-Class SVM**: Too slow for 15 detectors
- **Autoencoders**: Requires GPU, harder to interpret
- **DBSCAN**: Sensitive to hyperparameters
- **LOF**: Doesn't scale to 4000+ samples

---

### 10.3 Why cache models instead of retraining daily?

**Reasoning:**
- Training takes ~3 minutes (blocks dashboard startup)
- Market structure doesn't change daily
- Live price updates + feature recalculation is sufficient for intraday

**When to retrain:**
- Weekly (recommended): Capture recent market behavior
- After regime change: New VIX patterns emerge
- After adding features: Models need to learn new dimensions

---

### 10.4 Why 3 JSON files instead of 1?

**Reasoning:**

| File | Size | Update | Rationale |
|------|------|--------|-----------|
| `live_state.json` | 15 KB | Every 15min | Minimize frontend load (small, fast) |
| `historical.json` | 300 KB | Training only | Static reference, no need to reload |
| `model_cache.pkl` | 15 MB | Training only | Binary format (10x faster than JSON) |

**Alternative considered:**
- Single `dashboard_data.json` (400 KB) updated every 15min
- **Rejected**: Wastes bandwidth re-sending static historical data

---

### 10.5 Why bootstrap CIs for thresholds?

**Reasoning:**
- Provides uncertainty quantification (is 0.78 stable or noisy?)
- Helps detect overfitting (wide CIs = unstable threshold)
- Enables adaptive thresholds (use CI lower bound for conservative alerts)

**Example output:**
```
Moderate: 0.70 [0.68, 0.72]  # Narrow CI = stable
High:     0.78 [0.74, 0.82]  # Wide CI = less confident
```

---

## 🧪 11. Testing & Validation

### 11.1 Unit Test Examples

**Test data fetcher:**
```python
# tests/test_data_fetcher.py
from core.data_fetcher import UnifiedDataFetcher

def test_vix_fetch():
    fetcher = UnifiedDataFetcher()
    vix = fetcher.fetch_vix('2024-01-01', '2024-12-31')
    
    assert vix is not None
    assert len(vix) > 200  # ~252 trading days
    assert vix.min() > 0   # VIX can't be negative
    assert vix.max() < 100 # Sanity check

def test_cache_behavior():
    fetcher = UnifiedDataFetcher()
    
    # First fetch (network call)
    vix1 = fetcher.fetch_vix('2024-01-01', '2024-01-31')
    
    # Second fetch (should use cache)
    vix2 = fetcher.fetch_vix('2024-01-01', '2024-01-31')
    
    assert vix1.equals(vix2)
```

---

**Test anomaly detector:**
```python
# tests/test_anomaly_detector.py
from core.anomaly_detector import MultiDimensionalAnomalyDetector
import pandas as pd
import numpy as np

def test_detector_training():
    # Create normal data
    features = pd.DataFrame({
        'vix': np.random.uniform(15, 20, 1000),
        'vix_vs_ma21': np.random.normal(0, 1, 1000),
        'spx_ret_21d': np.random.normal(1, 2, 1000)
    })
    
    detector = MultiDimensionalAnomalyDetector(contamination=0.05)
    detector.train(features, verbose=False)
    
    assert detector.trained
    assert len(detector.detectors) >= 5  # At least some detectors trained

def test_anomaly_scoring():
    # Create extreme outlier
    features = pd.DataFrame({
        'vix': [60.0],  # VIX = 60 (crisis level)
        'vix_vs_ma21': [25.0],  # 25 points above MA
        'spx_ret_21d': [-15.0]  # -15% return
    })
    
    detector = MultiDimensionalAnomalyDetector()
    # ... train detector ...
    
    result = detector.detect(features)
    
    assert result['ensemble']['score'] > 0.80  # Should be high anomaly
```

---

### 11.2 Integration Test

**Full system test:**
```python
# tests/test_integration.py
from integrated_system_production import IntegratedMarketSystemV4

def test_full_pipeline():
    system = IntegratedMarketSystemV4()
    
    # Train on 2 years (faster)
    system.train(years=2, verbose=False)
    
    # Verify trained
    assert system.trained
    assert system.vix_predictor.anomaly_detector.trained
    
    # Get market state
    state = system.get_market_state()
    
    # Verify structure
    assert 'market_data' in state
    assert 'anomaly_analysis' in state
    assert 'model_diagnostics' in state
    
    # Verify values
    assert 0 <= state['anomaly_analysis']['ensemble']['score'] <= 1
    assert state['market_data']['vix'] > 0
```

---

### 11.3 Validation Metrics

**After training, check these:**

```python
# In Python console after system.train():

# 1. Feature coverage
system._verify_feature_coverage(system.vix_predictor.features, verbose=True)
# Target: All domains >80%

# 2. Detector count
assert len(system.vix_predictor.anomaly_detector.detectors) == 15

# 3. Threshold sanity
thresholds = system.vix_predictor.anomaly_detector.statistical_thresholds
assert 0.5 < thresholds['moderate'] < 0.75
assert 0.7 < thresholds['high'] < 0.85
assert 0.85 < thresholds['critical'] < 0.95

# 4. Historical scores distribution
scores = system.vix_predictor.historical_ensemble_scores
print(f"Mean: {scores.mean():.3f}")  # Should be ~0.50
print(f"Std:  {scores.std():.3f}")   # Should be ~0.15-0.25
```

---

## 🔄 12. Maintenance & Operations

### 12.1 Weekly Retrain Workflow

**Recommended schedule: Monday 8 AM (before market open)**

```bash
#!/bin/bash
# retrain.sh

# Set training mode
export ENABLE_TRAINING=True

# Backup old models
cp json_data/model_cache.pkl json_data/model_cache.pkl.bak
cp json_data/historical.json json_data/historical.json.bak

# Run training
python integrated_system_production.py > logs/training_$(date +%Y%m%d).log 2>&1

# Check exit code
if [ $? -eq 0 ]; then
    echo "âœ… Training successful"
    
    # Start dashboard
    export ENABLE_TRAINING=False
    python dashboard_orchestrator.py --port 8000 &
else
    echo "âŒ Training failed - restoring backup"
    cp json_data/model_cache.pkl.bak json_data/model_cache.pkl
    cp json_data/historical.json.bak json_data/historical.json
fi
```

---

### 12.2 Monitoring Checklist

**Daily checks:**

- [ ] Auto-refresh still running? (check dashboard)
- [ ] Memory growth normal? (<50 MB growth)
- [ ] Data cache fresh? (check `_cache_metadata.json` timestamps)
- [ ] FRED API responsive? (check logs for "FRED:* failed")

**Weekly checks:**

- [ ] Retrain completed successfully?
- [ ] Threshold changes? (compare to last week)
- [ ] Detector coverage maintained? (all >80%)
- [ ] Disk space sufficient? (`data_cache/` can grow to ~500 MB)

**Monthly checks:**

- [ ] Clear old cache files: `find data_cache/ -mtime +90 -delete`
- [ ] Review anomaly episodes: Are false positives decreasing?
- [ ] Check model drift: Compare detector scores to 3 months ago

---

### 12.3 Disaster Recovery

**Scenario 1: model_cache.pkl corrupted**

```bash
# Remove corrupted cache
rm json_data/model_cache.pkl

# Force retrain
export ENABLE_TRAINING=True
python integrated_system_production.py

# Verify outputs
ls -lh json_data/
# Should see model_cache.pkl (~15 MB)
```

---

**Scenario 2: Data cache corrupted**

```bash
# Nuclear option: delete cache
rm -rf data_cache/*

# System will rebuild cache on next run
python dashboard_orchestrator.py
# First run will be slow (~5 min to rebuild cache)
```

---

**Scenario 3: Dashboard not loading**

```bash
# Check if server running
lsof -i :8000
# If nothing, start manually:
python dashboard_orchestrator.py --port 8000

# Check JSON files exist
ls json_data/live_state.json json_data/historical.json
# If missing, run training mode once

# Check browser console
# Open dashboard, press F12, check for:
# - CORS errors (wrong port?)
# - JSON parse errors (corrupted file?)
# - 404 errors (wrong path?)
```

---

## 📚 13. Extension Recipes

### 13.1 Add Intraday VIX Updates

**Current**: Dashboard updates every 15 minutes with daily VIX close  
**Goal**: Update every 1 minute with live VIX price

**Changes needed:**

```python
# In dashboard_orchestrator.py, modify _refresh_loop():
self.refresh_interval = 60  # Change from 900 to 60 seconds

# In data_fetcher.py, add method:
def fetch_intraday_vix(self, interval='1m', period='1d'):
    """Fetch 1-minute VIX bars."""
    ticker = yf.Ticker('^VIX')
    data = ticker.history(interval=interval, period=period)
    return data['Close'].iloc[-1]  # Latest 1-min bar

# In dashboard_orchestrator.py, _attempt_refresh():
# Replace:
live_vix = self.system.vix_predictor.fetcher.fetch_price('^VIX')
# With:
live_vix = self.system.vix_predictor.fetcher.fetch_intraday_vix()
```

**Trade-offs:**
- **Pro**: Real-time updates (1-min latency)
- **Con**: Higher API rate limit usage (900 calls/day → 21,600 calls/day)

---

### 13.2 Add Options Skew Surface

**Goal**: Track VIX options skew (OTM puts vs calls) as anomaly indicator

**Implementation:**

```python
# Step 1: Fetch options data
# In data_fetcher.py:
def fetch_vix_options(self):
    """Get VIX option chain."""
    ticker = yf.Ticker('^VIX')
    options = ticker.options  # List of expiry dates
    
    # Get nearest expiry
    calls = ticker.option_chain(options[0]).calls
    puts = ticker.option_chain(options[0]).puts
    
    return calls, puts

# Step 2: Calculate skew
# In feature_engine.py:
def _options_skew_features(self) -> pd.DataFrame:
    calls, puts = self.fetcher.fetch_vix_options()
    
    features = pd.DataFrame()
    
    # Put/call open interest ratio
    features['vix_put_call_oi_ratio'] = puts['openInterest'].sum() / calls['openInterest'].sum()
    
    # OTM put premium (fear gauge)
    atm_strike = calls['strike'].iloc[len(calls)//2]
    otm_puts = puts[puts['strike'] < atm_strike * 0.8]
    features['vix_otm_put_premium'] = otm_puts['lastPrice'].mean()
    
    return features

# Step 3: Add to detector group
# In config.py:
ANOMALY_FEATURE_GROUPS = {
    'cboe_options_flow': [
        'SKEW',
        'vix_put_call_oi_ratio',  # NEW
        'vix_otm_put_premium',     # NEW
        ...
    ]
}
```

**Challenges:**
- Options data requires Yahoo Finance Pro or CBOE DataShop subscription
- Data availability: Only during market hours
- Latency: 15-minute delay on free feeds

---

### 13.3 Add Email Alerts

**Goal**: Email when anomaly score exceeds critical threshold

**Implementation:**

```python
# Step 1: Install dependency
# pip install sendgrid

# Step 2: Add to config.py
EMAIL_ALERTS = {
    'enabled': True,
    'api_key': 'your_sendgrid_key',
    'from_email': 'alerts@yourdomain.com',
    'to_emails': ['trader@example.com'],
    'threshold': 0.88  # Critical level
}

# Step 3: Create alert module
# alert_manager.py:
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from config import EMAIL_ALERTS

def send_alert(anomaly_score, active_detectors):
    if not EMAIL_ALERTS['enabled']:
        return
    
    if anomaly_score < EMAIL_ALERTS['threshold']:
        return
    
    message = Mail(
        from_email=EMAIL_ALERTS['from_email'],
        to_emails=EMAIL_ALERTS['to_emails'],
        subject=f'âš ï¸ Critical Anomaly Alert: {anomaly_score:.1%}',
        html_content=f"""
        <h2>Market Anomaly Detected</h2>
        <p><strong>Score:</strong> {anomaly_score:.1%}</p>
        <p><strong>Active Detectors:</strong></p>
        <ul>{''.join([f'<li>{d}</li>' for d in active_detectors])}</ul>
        """
    )
    
    sg = SendGridAPIClient(EMAIL_ALERTS['api_key'])
    sg.send(message)

# Step 4: Integrate into refresh cycle
# In dashboard_orchestrator.py, _attempt_refresh():
from alert_manager import send_alert

anomaly_result = self.system._get_cached_anomaly_result(force_refresh=True)
score = anomaly_result['ensemble']['score']

send_alert(score, anomaly_result['anomaly']['active_detectors'])
```

---

### 13.4 Add Machine Learning Ensemble

**Goal**: Use ML to weight detectors (instead of equal weighting)

**Current**: Ensemble score = mean of all 15 detector scores  
**Proposed**: Ensemble score = weighted sum (learn optimal weights)

**Implementation:**

```python
# In anomaly_detector.py, add method:
def train_ensemble_weights(self, features: pd.DataFrame, labels: pd.Series):
    """
    Learn optimal detector weights via logistic regression.
    
    Args:
        features: Historical features
        labels: Binary (1 = anomaly day, 0 = normal day)
    """
    from sklearn.linear_model import LogisticRegression
    
    # Get detector scores for all historical dates
    detector_scores = []
    for idx in range(len(features)):
        result = self.detect(features.iloc[[idx]], verbose=False)
        scores = [result['domain_anomalies'][name]['score'] 
                  for name in self.detectors.keys()]
        detector_scores.append(scores)
    
    X = np.array(detector_scores)  # Shape: (n_samples, 15)
    y = labels.values
    
    # Train logistic regression
    self.ensemble_model = LogisticRegression()
    self.ensemble_model.fit(X, y)
    
    # Extract weights
    self.detector_weights = self.ensemble_model.coef_[0]
    print(f"Learned weights: {self.detector_weights}")

# Modify detect() to use learned weights:
def detect(self, features: pd.DataFrame):
    # ... existing code to get detector scores ...
    
    if hasattr(self, 'detector_weights'):
        # Weighted ensemble
        weighted_score = np.dot(scores, self.detector_weights)
        ensemble_score = float(1 / (1 + np.exp(-weighted_score)))  # Sigmoid
    else:
        # Equal-weight ensemble (fallback)
        ensemble_score = float(np.mean(scores))
    
    return results
```

**Training labels:**
```python
# Create labels based on future drawdowns
spx_returns = spx.pct_change(5).shift(-5)  # 5-day forward return
labels = (spx_returns < -0.05).astype(int)  # 1 if >5% drawdown ahead

detector.train_ensemble_weights(features, labels)
```

---

## 🎯 14. Production Deployment

### 14.1 Systemd Service (Linux)

**Create service file:**
```bash
# /etc/systemd/system/vix-dashboard.service
[Unit]
Description=VIX Anomaly Detection Dashboard
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/SPX_Analysis/src
Environment="ENABLE_TRAINING=False"
ExecStart=/usr/bin/python3 dashboard_orchestrator.py --skip-training --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable vix-dashboard
sudo systemctl start vix-dashboard
sudo systemctl status vix-dashboard
```

---

### 14.2 Docker Container

**Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY json_data/ ./json_data/

# Expose port
EXPOSE 8000

# Run dashboard
CMD ["python", "src/dashboard_orchestrator.py", "--skip-training", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  vix-dashboard:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./json_data:/app/json_data
      - ./data_cache:/app/data_cache
    environment:
      - ENABLE_TRAINING=False
    restart: unless-stopped
```

---

### 14.3 AWS Deployment (EC2)

**Recommended instance**: `t3.medium` (2 vCPU, 4 GB RAM)

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key \
  --security-group-ids sg-xxxxx

# SSH into instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Install dependencies
sudo apt update
sudo apt install python3-pip git
git clone https://github.com/yourusername/SPX_Analysis.git
cd SPX_Analysis/src
pip3 install -r requirements.txt

# Configure FRED API key
echo '{"fred_api_key": "your_key"}' > json_data/config.json

# Run initial training
export ENABLE_TRAINING=True
python3 integrated_system_production.py

# Start dashboard
export ENABLE_TRAINING=False
nohup python3 dashboard_orchestrator.py --skip-training --port 8000 &

# Configure security group to allow port 8000
```

---

## 🔐 15. Security Considerations

### 15.1 API Key Management

**NEVER commit API keys to Git:**

```bash
# Add to .gitignore:
json_data/config.json
.env
*.key
```

**Use environment variables:**
```bash
export FRED_API_KEY="your_key_here"
# System will read from env if json_data/config.json not found
```

---

### 15.2 Dashboard Access Control

**Current**: Dashboard is publicly accessible on port 8000  
**Production**: Add authentication

**Option 1: Nginx reverse proxy with basic auth**
```nginx
# /etc/nginx/sites-available/vix-dashboard
server {
    listen 80;
    server_name vix.yourdomain.com;
    
    location / {
        auth_basic "Restricted";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://localhost:8000;
    }
}
```

**Option 2: VPN-only access**
- Only allow connections from VPN IP range
- Add to AWS security group or EC2 firewall

---

## 📖 16. Glossary

| Term | Definition |
|------|------------|
| **Ensemble score** | Weighted average of 15 detector anomaly scores (0-1 scale) |
| **Contamination** | Expected proportion of anomalies in training data (default: 1%) |
| **Isolation Forest** | ML algorithm that isolates anomalies via random partitioning |
| **Feature coverage** | % of detector's expected features present in data |
| **Regime** | VIX volatility state (Low/Normal/Elevated/Crisis) |
| **Z-score** | Standardized distance from mean (units of std dev) |
| **Persistence** | How long anomalies last (current streak vs. historical mean) |
| **RobustScaler** | Feature scaling robust to outliers (uses IQR instead of std dev) |
| **Bootstrap CI** | Confidence interval via resampling with replacement |
| **Lookback buffer** | Extra historical data for rolling calculations (e.g., 252 days for 1-year MA) |

---

## 🚦 17. Final Checklist

Before going live:

- [ ] FRED API key configured
- [ ] Trained on at least 5 years of data
- [ ] All 15 detectors active (coverage >80%)
- [ ] Statistical thresholds computed (bootstrap CIs)
- [ ] JSON files exported and validated
- [ ] Auto-refresh tested (run for 1 hour, check memory growth)
- [ ] Dashboard loads in browser
- [ ] Historical chart displays correctly
- [ ] Live updates working (scores change when VIX moves)
- [ ] Backup strategy in place (daily cron job)
- [ ] Monitoring alerts configured
- [ ] Documentation shared with team

---

## 📞 Quick Help

**"Where do I...?"**

| Task | Location |
|------|----------|
| Change a threshold | `config.py` → Constants |
| Add a ticker | `config.py` → `MACRO_TICKERS` |
| Create a feature | `core/feature_engine.py` → Add method |
| Modify a detector | `core/anomaly_detector.py` → Training params |
| Change export format | `export/unified_exporter.py` → Schema |
| Debug refresh | `dashboard_orchestrator.py` → `_attempt_refresh()` |
| Fix missing data | `core/data_fetcher.py` → Caching logic |

**"How do I...?"**

| Task | Command/Code |
|------|--------------|
| Retrain models | `ENABLE_TRAINING=True python dashboard_orchestrator.py` |
| Clear cache | `rm -rf data_cache/*` |
| Check memory | `system.get_memory_report()` |
| Validate exports | `cat json_data/live_state.json \| jq '.schema_version'` |
| Test detector | See §11.1 Unit Tests |

---

**End of Code Navigation Map** | Version 4.0 | Keep this handy! 🎯