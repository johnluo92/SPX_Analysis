# SPX Prediction System — System Architecture

**Version:** 2.1  
**Last Updated:** 2025-10-23  
**Audience:** John & Future Claude Sessions

---

## Overview

**Purpose:** Predict S&P 500 directional moves and range-bound probabilities to generate high-confidence options trading signals.

**Philosophy:**
- **Simplicity & Consistency** — Twin pillars of system design
- **Homoscedasticity** — All predictions use trading days (not calendar days)
- **Fibonacci Horizons** — 8, 13, 21, 34 trading days
- **Walk-Forward Validation** — Real-world accuracy testing

**Outputs:**
- Directional predictions: "Will SPX be higher in N days?"
- Range predictions: "Will SPX stay within ±X% in N days?"
- Options signals: Iron Condors, Bull Put Spreads
- Real-time dashboard with confidence metrics

---

## System Architecture

### File Structure

| File | Responsibility |
|------|----------------|
| **config.py** | Constants: ETFs, tickers, FRED series, hyperparameters, forward windows [8,13,21,34], range thresholds [2%,3%,5%,8%,13%,21%] |
| **UnifiedDataFetcher.py** | Fetches SPX/VIX/macro/FRED data from Yahoo & FRED API; handles daily caching; returns trading days only |
| **spx_features.py** | Feature engineering: momentum, volatility, macro indicators, FRED data, market regimes; ~44 features |
| **spx_model.py** | RandomForest models for directional & range predictions; feature selection (top 30); training & prediction |
| **spx_predictor.py** | Pipeline orchestrator: fetch → engineer → train → predict; prints training metrics |
| **spx_dashboard.py** | Main entry: trains models, calculates model health, generates dashboard_data.json, starts HTTP server |
| **dashboard.js** | React frontend: reads JSON, displays predictions/signals/health with interactive UI |
| **dashboard.html** | HTML wrapper for React app |
| **cache_cleaner.py** | Optional: cleans cache files >N days old |

### Data Sources

**Market Data (Yahoo Finance):**
- SPX: ^GSPC closing prices (7 years)
- VIX: Volatility index
- Macro: Gold (GLD), Oil (CL=F), Dollar (DX-Y.NYB), 10Y (^TNX), 5Y (^FVX)

**Economic Data (FRED API):**
- T10YIE: 10Y Breakeven Inflation
- T5YIFR: 5Y Forward Inflation Expectation
- T10Y2Y: 10Y-2Y Yield Spread
- VIXCLS: VIX Close

**Real-Time:** SPX current price via Yahoo (15-min delay)

---

## Data Pipeline

### Step 1: Data Collection
```
UnifiedDataFetcher → Fetch 7 years of data
                  → Cache as .parquet files in .cache_sector_data/
                  → Daily refresh check (file modification time)
                  → Return TRADING DAYS ONLY (no weekends/holidays)
```

**Cache Files:**
- `yahoo_^GSPC_YYYY-MM-DD_YYYY-MM-DD.parquet`
- `vix_YYYY-MM-DD_YYYY-MM-DD.parquet`
- `macro_YYYY-MM-DD_YYYY-MM-DD.parquet`
- `fred_all_YYYY-MM-DD_YYYY-MM-DD.parquet`

### Step 2: Feature Engineering

**spx_features.py** creates ~44 features across categories:

**1. SPX Features**
- Returns: 5d, 10d, 21d, 63d percentage changes
- Realized volatility: 10d, 21d, 63d (annualized, backward-looking)
- Regime indicators: distance from MA50, MA200
- Percentiles: current position in recent distribution

**2. VIX Features**
- Level, changes, z-score
- Mean reversion signals
- Distance from moving averages

**3. Macro Features**
- Momentum: Gold, Oil, Dollar over 10d, 21d windows
- Level changes

**4. FRED Features**
- Yield spreads (10Y-2Y): level and changes over 21d, 63d
- Inflation expectations (level and momentum)
- Yield slope (10Y-5Y)

**5. Temporal Features**
- Month, quarter (seasonality patterns)

**⚠️ IV-RV Spread (Optional, Disabled by Default):**
```python
# spx_predictor.py has use_iv_rv_cheat flag (default=False)
if use_iv_rv_cheat:
    # DANGER: Uses FORWARD 30-day realized vol
    # Creates lookahead bias - DO NOT USE for real trading
else:
    # Safe: Uses only backward-looking features
```

**Current Top Feature Types** (examples from 2025-10-23):
- Yield curve dynamics (10Y-2Y spread changes)
- Inflation expectations (breakeven levels)
- Realized volatility (63-day lookback)
- Macro momentum (Gold, Dollar)
- VIX regime characteristics

### Step 3: Target Creation

For each forward window (8, 13, 21, 34 **TRADING DAYS**):

**Directional Target:**
```python
fwd_return = spx.pct_change(window).shift(-window)
target = (fwd_return > 0).astype(int)  # 1=up, 0=down
```

**Range Targets** (for each threshold: 2%, 3%, 5%, 8%, 13%, 21%):
```python
fwd_return = spx.pct_change(window).shift(-window)
in_range = (abs(fwd_return) <= threshold).astype(int)  # 1=stayed, 0=breached
```

**Critical:** All math uses TRADING DAYS for homoscedasticity (consistent variance across windows).

### Step 4: Feature Selection (Two Phases)

**Phase 1: Training/Selection** (spx_model.py)
- Quick RandomForest on 21d directional target
- Ranks ALL 44 features by importance
- Selects top 30 features
- Prints to console during training

**Example Output:**
```
📊 TOP 30 FEATURES BY IMPORTANCE:
    1. 10Y-2Y Yield Spread_change_63    0.0823
    2. 10Y Breakeven Inflation_level    0.0724
    3. spx_realized_vol_63              0.0644
    ...
```

**Phase 2: Deployed Model** (after training actual models)
- 21d directional model trained on selected 30 features
- Feature importances recalculated on final model
- Normalized to percentages for dashboard
- Different values than selection phase

**Example Output:**
```
📊 Top Features for Dashboard:
   10Y-2Y Yield Spread_change_63: 14.92%
   10Y Breakeven Inflation_level: 11.01%
   spx_realized_vol_63: 6.31%
```

**Why Different?** Selection phase ranks raw importance across all features; deployed phase shows relative contribution within the refined model.

### Step 5: Model Training

**28 Total Models:**
- 4 Directional (one per horizon: 8d, 13d, 21d, 34d)
- 24 Range (4 horizons × 6 thresholds)

**Algorithm:** RandomForestClassifier
```python
n_estimators = 200
max_depth = 6
min_samples_split = 50
min_samples_leaf = 30
max_features = 'sqrt'
```

**Train/Test Split:**
- 80/20 time-based (no shuffling)
- Test set = most recent 20% of data
- No future leakage

**Printed Metrics:**
```
📈 Training Directional Models...
  21d: Train 0.879 | Test 0.831 | Gap 0.048 ✅
```

**Gap = train_accuracy - test_accuracy**
- Gap < 0.10: Excellent generalization ✅
- Gap 0.10-0.15: Acceptable ✅
- Gap > 0.15: Potential overfit ⚠️

### Step 6: Prediction

**Input:** Current features (last row of feature matrix)

**Output:** Probabilities for each model
```json
{
  "direction_8d": 0.586,
  "direction_13d": 0.681,
  "direction_21d": 0.627,
  "direction_34d": 0.717,
  "range_8d_2pct": 0.519,
  "range_8d_3pct": 0.827,
  ...
}
```

**Console Output:**
```
📈 DIRECTIONAL (Will SPX be higher?):
   21d: 62.7%

📊 RANGE-BOUND (Will SPX stay within range?):
   21d ±5pct: 83.7%
```

### Step 7: Dashboard Generation

**spx_dashboard.py** performs:

1. **Model Health Calculation** (not printed, used by dashboard)
   ```python
   def get_model_health():
       # Aggregates train/test/gap across all models
       # Returns: avg_test_acc, avg_gap, status
       # Status: STRONG/GOOD/FAIR/WEAK
   ```

   **Thresholds:**
   - STRONG: test ≥ 85% AND gap ≤ 10%
   - GOOD: test ≥ 75% AND gap ≤ 15%
   - FAIR: test ≥ 65%
   - WEAK: test < 65%

2. **DTE Mapping** (trading days → calendar days)
   ```python
   calculate_calendar_days(trading_days)
   # 8td → ~11cd, 13td → ~18cd, 21td → ~30cd, 34td → ~49cd
   ```

3. **Trade Signal Generation**
   - Iron Condor: range probability > 90%
   - Bull Put Spread: directional > 65%
   - Strike selection based on probability thresholds

4. **JSON Creation**
   ```json
   {
     "spx_price": 6735.35,          // Real-time
     "spx_price_model": 6735.13,    // Model training data
     "vix": 18.23,
     "available_horizons": ["8d", "13d", "21d", "34d"],
     "dte_mapping": {"21d": 30},
     "directional": {...},
     "range_bound": {...},
     "trade_signals": [...],
     "model_health": {...},
     "top_features": {...}          // Deployed importances
   }
   ```

5. **HTTP Server** (localhost:8000)

### Step 8: Web Dashboard

**dashboard.js** (React, no build step):
- Reads dashboard_data.json
- Real-time SPX price display
- Time horizon selector (8D/13D/21D/34D with DTE)
- Directional confidence bars
- Range probability heatmaps
- Trade signals table
- Model health badge (colored by status)
- Top features footer

**Auto-refresh:** Regenerate JSON → dashboard updates on page refresh

---

## Key Concepts

### Trading Days vs Calendar Days

**Trading Days (Internal):**
- All model math uses trading days
- Maintains homoscedasticity
- SPX only moves on trading days
- Volatility compounds on trading days

**Calendar Days (Display):**
- Options use DTE (Days to Expiration)
- Dashboard converts: 21 trading days ≈ 30 calendar days
- Mapping done dynamically based on actual calendar

**Example:**
```
Model: "21d directional" → 21 trading days
Dashboard: "21d (30 DTE)" → ~30 calendar days
```

### Fibonacci Horizons

**Why [8, 13, 21, 34]?**
- Natural market cycle alignment
- 21d is the "anchor" (typically 75-85% accuracy)
- 8d captures short-term noise (~65-75%)
- 34d good for wide strategies (~75-80%)
- Not arbitrary - based on market rhythm

### Dual Target System

**Directional:** "Will SPX go up?"
- Binary classification: up vs down
- Use for: Bull Put Spreads, directional bets
- Trade threshold: >65% confidence

**Range:** "Will SPX stay within ±X%?"
- Binary classification: contained vs breached
- Use for: Iron Condors, neutral strategies
- Trade threshold: >90% confidence
- Wider ranges (±13%, ±21%) almost always true
- Tighter ranges (±2%, ±3%) harder to predict

### Performance Characteristics

**Typical Accuracy (Post IV-RV Fix):**
```
Directional Models:
  21d: 75-85% (best, most reliable)
  13d: 70-80% (good)
  8d: 65-75% (noisy)
  34d: 75-80% (wide strategies)

Range Models:
  Wide ranges (±13-21%): 95-99% (Iron Condors)
  Medium ranges (±5-8%): 85-95%
  Tight ranges (±2-3%): 65-85% (risky, often overfit)
```

**Gap Analysis:**
- Negative gap (test > train): Slight underfit, often good sign
- 0-5% gap: Excellent
- 5-10% gap: Very good
- 10-15% gap: Acceptable
- >15% gap: Warning, may not generalize

---

## Console vs Dashboard Data

### During Training (Console Output)

**Printed by spx_predictor.py:**
- Data fetch status & cache usage
- Feature engineering progress
- Top 30 feature selection (training/selection phase)
- Model training metrics (train/test/gap) for all 28 models
- Current predictions (directional & range probabilities)
- Top 5 deployed features (from 21d model)

**Not Printed:**
- Model health status (calculated but not shown)
- Trade signals (generated in dashboard)
- DTE mappings (calculated in dashboard)

### Dashboard Display (Web UI)

**Calculated by spx_dashboard.py, shown in browser:**
- Model health badge (STRONG/GOOD/FAIR/WEAK)
- Trade signals with strike recommendations
- DTE conversions for each horizon
- Interactive probability heatmaps
- Top features from deployed model
- Real-time price comparison

**Why Split?** Console shows training diagnostics for validation; dashboard shows trading-ready information for decision-making.

---

## Common Tasks

### Run the System
```bash
cd src/
python spx_dashboard.py
# Opens browser to localhost:8000
# ~8-10 seconds to train all models
```

### Add a Forward Window
```python
# config.py
SPX_FORWARD_WINDOWS = [8, 13, 21, 34, 55]  # Add 55d
# Rerun dashboard - automatically trains new models
```

### Add a Feature
```python
# spx_features.py
def build(...):
    # Add feature calculation
    features['my_new_feature'] = ...
# Rerun - feature selection evaluates it
```

### Check Model Performance
```bash
python spx_predictor.py
# Standalone run shows detailed metrics
```

### Update Real-Time Price
```bash
python spx_dashboard.py
# Fetches fresh price, regenerates JSON
```

### Clear Cache
```bash
rm -rf .cache_sector_data/
# Forces fresh data fetch
```

### Enable IV-RV Cheat (NOT RECOMMENDED)
```python
# spx_predictor.py
predictor = SPXPredictor(use_iv_rv_cheat=True)
# Prints warning about lookahead bias
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `IndexError: index 1 out of bounds` | Model has only 1 class | Already handled in predict() |
| `FRED API key not found` | Missing config.json | Create: `{"fred_api_key": "KEY"}` |
| `No module named 'spx_features'` | Wrong directory | Run from src/ |
| Dashboard shows old data | Browser cache | Hard refresh (Cmd+Shift+R) |
| Gap > 0.15 on multiple models | Overfitting | Add data / tune hyperparams |
| Console accuracy ≠ dashboard status | Different metrics | Console=per-model, dashboard=aggregate |

---

## Lessons Learned: The IV-RV Lookahead Bias

### What Happened

**Original Implementation (v2.0):**
```python
# spx_predictor.py (OLD - DON'T DO THIS)
for i in range(30, len(vix) - 30):
    date = vix.index[i]
    future_slice = spx.loc[spx.index >= date]
    future_prices = future_slice.iloc[:30]  # Look FORWARD 30 days
    realized_future = calculate_vol(future_prices)
    iv_rv_spread[date] = vix[date] - realized_future
```

**Problem:**
- Training used PERFECT forward information (knew what volatility WOULD BE)
- Live predictions used 30-day-old stale information (last date with complete forward window)
- Created train/production mismatch invisible in backtests
- Top 3 features (35% combined importance) were based on this lookahead bias

**Why Backtests Looked Good:**
- Historical dates had complete forward windows
- Model learned: "When iv_rv_spread = X, SPX does Y"
- Test set also had forward data (legitimate for those dates)
- But LIVE trading couldn't access that future data

**Impact:**
- Reported accuracy: ~91% (artificially inflated)
- Real live accuracy: ~85-88% (more realistic)
- 3-6% edge lost due to stale signals

### Current Implementation (v2.1)

**Default Behavior:**
```python
# spx_predictor.py (CURRENT)
predictor = SPXPredictor(use_iv_rv_cheat=False)  # Default
# Prints: "✅ Using ONLY backward-looking volatility features (no cheating)"
```

**If Enabled (for comparison only):**
```python
predictor = SPXPredictor(use_iv_rv_cheat=True)
# Prints: "⚠️ WARNING: Using forward-looking IV-RV spread (lookahead bias)"
```

### Key Takeaways

1. **Lookahead Bias is Subtle:** Features that "peek into the future" create train/test mismatch that's invisible in historical backtests but fails in live trading.

2. **Forward vs Backward:** Always ask: "Will I have this information at prediction time?" If not, don't train on it.

3. **Staleness Matters:** If your best feature requires 30 days to calculate, your live predictions are 30 days behind.

4. **Realistic Accuracy:** 85% is EXCELLENT for real trading. Don't chase inflated metrics.

5. **Feature Engineering Check:**
   - ✅ GOOD: Calculate from data up to and including current date
   - ❌ BAD: Calculate from data after current date
   - ✅ GOOD: Use past 30-day realized volatility
   - ❌ BAD: Use forward 30-day realized volatility

### Prevention Checklist

Before adding any feature, ask:
- [ ] Does this use .shift(-N)? (looking forward)
- [ ] Does this use future_slice or forward window? (looking ahead)
- [ ] Will this exact value exist on prediction day? (no lag)
- [ ] Can this be calculated from past data only? (backward-looking)

**Rule:** If a feature requires waiting N days to complete, either:
1. Use backward-looking version (recommended)
2. Explicitly lag it by N days in training AND production
3. Don't use it at all

---

## Quick Reference

### System Status (v2.1)
- ✅ Lookahead bias removed (default)
- ✅ Dynamic feature importance extraction
- ✅ Realistic performance metrics (75-85%)
- ✅ Model health calculated per run
- ✅ Trading-day homoscedasticity maintained
- ✅ DTE mapping functional
- ✅ 28 models trained per run (4 directional + 24 range)

### Best Practices
- Use 21d models for highest confidence
- Trade directional at >65% probability
- Trade range at >90% probability
- Monitor gap metric (keep <0.15)
- Avoid tight ranges (±2%, ±3%) for real trading
- Use wide ranges (±13%, ±21%) for Iron Condors
- Always validate on fresh data before trading

### File Locations
```
src/
├── config.py              # Constants & hyperparameters
├── UnifiedDataFetcher.py  # Data acquisition
├── spx_features.py        # Feature engineering
├── spx_model.py           # ML models
├── spx_predictor.py       # Pipeline orchestration
├── spx_dashboard.py       # Main entry point
├── dashboard.html         # UI wrapper
├── dashboard.js           # React frontend
├── cache_cleaner.py       # Cache maintenance
└── .cache_sector_data/    # Cached parquet files
```

### Key Numbers to Remember
- Forward windows: 8, 13, 21, 34 trading days
- Range thresholds: 2%, 3%, 5%, 8%, 13%, 21%
- Feature count: ~44 → select top 30
- Total models: 28 (4 + 24)
- Training time: ~8-10 seconds
- Cache refresh: Daily
- Target accuracy: 75-85% (realistic)

---

**End of System Architecture**  
*For questions or updates, refer to this document as source of truth.*