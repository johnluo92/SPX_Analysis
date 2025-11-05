# SPX Analysis System

**Last Updated:** 2025-10-27  
**Location:** `~/Desktop/GitHub/SPX_Analysis/src`

---

## Quick Context for Claude
```bash
cd ~/Desktop/GitHub/SPX_Analysis/src
tree -L 2 -I '__pycache__|_archive|data_cache'
```

---

## 📊 JSON Data Layer
**Path:** `json_data/`

### Core Outputs (Updated Every Run)
- `anomaly_report.json` (1.8K) - Current anomaly scores, active alerts, domain breakdown
- `market_state.json` (5.2K) - Current VIX regime, transition probabilities
- `regime_statistics.json` (4.4K) - Regime duration stats, historical context
- `spx_predictions.json` (1.9K) - SPX forecasts (1d, 5d, 10d, 20d)
- `vix_history.json` (636K) - Full VIX time series with features

### Historical/Validation
- `historical_anomaly_scores.json` (969K) - 500+ days of anomaly scores + SPX for charts
- `anomaly_validation.json` (9.2K) - Backtest metrics, confusion matrices
- `validation_report.json` (701B) - Latest validation summary

### Legacy
- `spx_analysis.json` (1.6K) - Old SPX features (being phased out)
- `vix_history_documented.json` (40K) - Sample with documentation

**Need to Add:**
- `anomaly_metadata.json` - Domain definitions, methodology, thresholds
- `anomaly_feature_attribution.json` - Top features driving each domain score

---

## 🐍 Python Backend
**Path:** `.` (root of src/)

### Main Production Files
- `integrated_system_production.py` (22K) - Master orchestrator, runs everything
- `dashboard_orchestrator.py` (9.9K) - Generates all JSON outputs for dashboards
- `UnifiedDataFetcher.py` (23K) - Fetches VIX, SPX, FRED, Yahoo data with caching

### Model Components
- `vix_predictor_v2.py` (27K) - VIX forecasting (LSTM, XGBoost, regime-aware)
- `spx_predictor_v2.py` (12K) - SPX predictions using VIX + macro features
- `spx_features_v2.py` (14K) - SPX feature engineering
- `spx_model_v2.py` (11K) - SPX model training/inference
- `anomaly_system.py` (29K) - **10-domain ensemble anomaly detector**

### Utilities
- `config.py` (6.1K) - Shared configuration, API keys, paths
- `validation_simple.py` (5.7K) - Quick validation runner
- `dependency_checker.py` (7.8K) - Checks missing packages

---

## 🎨 Dashboards (HTML/JS)
**Path:** `.` (root) + `Chart Modules/`

### Main Dashboard
- `dashboard_unified.html` (11K) - **Main hub** (needs anomaly card added)

### Specialized Modules
- `Chart Modules/anomaly_diagnostic_embed.html` (39K) - **Deep dive anomaly dashboard** (the one we're improving)
- `Chart Modules/vix_distribution_embed.html` (11K) - VIX regime distribution
- `Chart Modules/transition_matrix_v2_embed.html` (15K) - Regime transition matrix

---

## 💾 Data Storage

### `data_cache/` (Parquet files)
- VIX, SPX, FRED, Yahoo data cached by date range
- Auto-refreshes when stale
- ~82 cached files, mostly parquet format

### `CBOE_Data_Archive/`
- Historical CBOE indicators (SKEW, Put/Call ratios, correlations)
- CSVs updated manually/semi-regularly

---

## 🗄️ Archive & Backups

### `_archive/`
- Old checkpoint files, backup from 2025-10-27
- Cache cleaning scripts
- Legacy dashboard prototypes

### `temp archive/`
- PNG exports, old analyzer scripts

---

## 🔄 How It All Works

1. **Data Collection** (UnifiedDataFetcher.py)
   - Fetches VIX, SPX, treasuries, commodities
   - Caches to `data_cache/` as parquet

2. **Model Inference** (integrated_system_production.py calls:)
   - `vix_predictor_v2.py` → VIX forecasts
   - `spx_predictor_v2.py` → SPX forecasts  
   - `anomaly_system.py` → 10-domain anomaly scores

3. **JSON Export** (dashboard_orchestrator.py)
   - Writes all results to `json_data/`
   - Includes historical time series for charts

4. **Dashboard Display**
   - `dashboard_unified.html` fetches JSONs, shows cards
   - Click card → opens detailed module (e.g., `anomaly_diagnostic_embed.html`)

---

## 🎯 Current Task

**Problem:** Anomaly dashboard shows scores but doesn't explain what's driving them.

**Solution:**
1. Add `anomaly_metadata.json` (domain descriptions, methodology)
2. Add `anomaly_feature_attribution.json` (top features per domain)
3. Update `anomaly_diagnostic_embed.html` with expandable feature sections
4. Add anomaly summary card to `dashboard_unified.html`

---

## 📦 Key Dependencies
- pandas, numpy, scikit-learn
- xgboost, tensorflow/keras
- yfinance, fredapi
- plotly (for charts in HTML)

---

## 🚀 Quick Start
```bash
# Run everything
python integrated_system_production.py

# Just generate JSONs for dashboards
python dashboard_orchestrator.py

# Open dashboards
open dashboard_unified.html
open "Chart Modules/anomaly_diagnostic_embed.html"
```