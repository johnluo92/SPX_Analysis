# Market Anomaly Detection System - Visual Architecture

*Comprehensive system architecture showing data flow, components, and relationships*

---

## System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MARKET ANOMALY DETECTION SYSTEM V4                    │
│                                                                           │
│  Purpose: Detect unusual market conditions using 15 ML detectors         │
│  Output: Real-time anomaly score (0-100%) with severity classification   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES LAYER                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │Yahoo Finance│  │  FRED API   │  │CBOE Archive │  │  Derived    │   │
│  │             │  │             │  │             │  │  Futures    │   │
│  │ • SPX       │  │ • 17 macro  │  │ • 19 CBOE   │  │ • VX spreads│   │
│  │ • VIX       │  │   series    │  │   indicators│  │ • CL spreads│   │
│  │ • Gold      │  │ • Treasuries│  │ • SKEW      │  │ • DX spreads│   │
│  │ • Silver    │  │ • Rates     │  │ • Put/Call  │  │             │   │
│  │ • Crude Oil │  │ • Inflation │  │ • COR1M/3M  │  │             │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                                  │                                       │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA FETCHER (data_fetcher.py)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Features:                                                                │
│  • Smart caching (avoid redundant API calls)                             │
│  • Incremental updates (only fetch new data)                             │
│  • Rate limit handling                                                   │
│  • Graceful degradation (continue if source unavailable)                │
│                                                                           │
│  Cache Directory: ./data_cache/                                          │
│  CBOE Directory: ./CBOE_Data_Archive/                                    │
│                                                                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  RAW TIME SERIES DATA    │
                    │  ~7,500 observations     │
                    │  (VIX + SPX + others)    │
                    └─────────┬────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINE (feature_engine.py)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Transforms raw prices into 696 engineered features:                     │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ VIX BASE FEATURES (78)                                            │  │
│  │ • Mean reversion: VIX vs moving averages (10,21,63,252 day)      │  │
│  │ • Dynamics: Returns, volatility, velocity, acceleration          │  │
│  │ • Regimes: Classification into Low/Normal/Elevated/Crisis        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ SPX BASE FEATURES (46)                                            │  │
│  │ • Price action: Returns, momentum, trend indicators               │  │
│  │ • Volatility: Realized vol, vol ratios, Bollinger Bands          │  │
│  │ • Technical: RSI, MACD, ADX, trend strength                      │  │
│  │ • Microstructure: Candlesticks, gaps, shadows (25 features)      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ CROSS-ASSET FEATURES (7)                                          │  │
│  │ • SPX-VIX correlation (21,63,126 day windows)                     │  │
│  │ • VIX vs Realized Volatility (risk premium)                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ CBOE FEATURES (232)                                               │  │
│  │ • SKEW indicators: Tail risk, velocity, regime                    │  │
│  │ • Put/Call ratios: PCCI, PCCE, PCC, divergences                  │  │
│  │ • Correlation indices: COR1M, COR3M, term structure               │  │
│  │ • Other: VXTH, stress composites                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ FUTURES FEATURES (45)                                             │  │
│  │ • VIX futures: Spreads, term structure, contango/backwardation   │  │
│  │ • Commodity: CL (crude oil) spreads and dynamics                  │  │
│  │ • Currency: DX (dollar index) spreads and trends                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ META FEATURES (34)                                                │  │
│  │ • Regime indicators: Micro/macro regimes, transitions             │  │
│  │ • Cross-asset relationships: Divergences, breakdowns              │  │
│  │ • Rate-of-change: Velocity, acceleration, jerk                    │  │
│  │ • Percentile rankings: Historical extremity measures              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ MACRO + CALENDAR (19)                                             │  │
│  │ • Macro: Gold/silver ratio, bond volatility, dollar strength      │  │
│  │ • Calendar: Month, quarter, day of week, OPEX flag               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  FEATURE MATRIX          │
                    │  3,769 days × 696 cols   │
                    │  (15 years of history)   │
                    └─────────┬────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ANOMALY DETECTION (anomaly_detector.py)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Method: 15 Independent Isolation Forest Detectors                       │
│  Algorithm: Unsupervised outlier detection via random tree isolation     │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  DOMAIN DETECTORS (Feature Subsets)                            │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │  1. vix_mean_reversion (18 features, 100% coverage)            │    │
│  │     Focus: VIX vs moving averages, Bollinger Bands             │    │
│  │                                                                 │    │
│  │  2. vix_momentum (18 features, 100%)                           │    │
│  │     Focus: VIX rate of change, velocity, acceleration          │    │
│  │                                                                 │    │
│  │  3. vix_regime_structure (18 features, 100%)                   │    │
│  │     Focus: Regime transitions, persistence, boundaries         │    │
│  │                                                                 │    │
│  │  4. cboe_options_flow (39 features, 100%)                      │    │
│  │     Focus: SKEW, Put/Call ratios, correlation indices          │    │
│  │                                                                 │    │
│  │  5. cboe_cross_dynamics (17 features, 100%)                    │    │
│  │     Focus: CBOE cross-relationships, divergences               │    │
│  │                                                                 │    │
│  │  6. vix_spx_relationship (17 features, 100%)                   │    │
│  │     Focus: SPX-VIX correlation, risk premium                   │    │
│  │                                                                 │    │
│  │  7. spx_price_action (20 features, 75%)                        │    │
│  │     Focus: Price momentum, technical indicators                │    │
│  │                                                                 │    │
│  │  8. spx_ohlc_microstructure (25 features, 0%)                  │    │
│  │     Focus: Candlestick patterns, gaps (DISABLED)               │    │
│  │                                                                 │    │
│  │  9. spx_volatility_regime (22 features, 91%)                   │    │
│  │     Focus: Realized vol vs implied vol, vol regimes            │    │
│  │                                                                 │    │
│  │  10. cross_asset_divergence (26 features, 88%)                 │    │
│  │      Focus: Multi-asset correlation breakdowns                 │    │
│  │                                                                 │    │
│  │  11. tail_risk_complex (18 features, 83%)                      │    │
│  │      Focus: SKEW extremes, VIX spikes, tail events             │    │
│  │                                                                 │    │
│  │  12. futures_term_structure (27 features, 100%)                │    │
│  │      Focus: Forward curves, spreads, contango                  │    │
│  │                                                                 │    │
│  │  13. macro_regime_shifts (19 features, 100%)                   │    │
│  │      Focus: Multi-regime transitions, consensus               │    │
│  │                                                                 │    │
│  │  14. momentum_acceleration (20 features, 100%)                 │    │
│  │      Focus: Second derivatives, rate-of-change                 │    │
│  │                                                                 │    │
│  │  15. percentile_extremes (19 features, 100%)                   │    │
│  │      Focus: Historical percentile rankings                     │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  Each detector:                                                           │
│  • Trained on domain-specific feature subset                             │
│  • Produces anomaly score (0-100%)                                       │
│  • Weighted by feature coverage × quality                                │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  ENSEMBLE VOTING                                                │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │  Weighted Average:                                              │    │
│  │                                                                 │    │
│  │  Ensemble = Σ (detector_score × weight)                        │    │
│  │             ─────────────────────────                           │    │
│  │                  Σ weight                                       │    │
│  │                                                                 │    │
│  │  Where weight = coverage × quality_penalty                      │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  STATISTICAL THRESHOLDS (learned from training data)           │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │  • NORMAL:    <70% (typical market conditions)                  │    │
│  │  • MODERATE:  70-78% (elevated but not critical)                │    │
│  │  • HIGH:      78-93% (significant anomaly, watch closely)       │    │
│  │  • CRITICAL:  >93% (extreme anomaly, major event likely)        │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   ANOMALY RESULT         │
                    │   • Ensemble: 38.1%      │
                    │   • Severity: NORMAL     │
                    │   • Top anomalies: [...]  │
                    └─────────┬────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR (integrated_system.py)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Coordinates:                                                             │
│  • Data fetching                                                         │
│  • Feature engineering                                                   │
│  • Anomaly detection                                                     │
│  • Regime statistics                                                     │
│  • State persistence                                                     │
│                                                                           │
│  Maintains:                                                               │
│  • Full VIX/SPX history                                                  │
│  • Feature matrix                                                        │
│  • Trained models                                                        │
│  • Historical ensemble scores                                            │
│  • Regime transition statistics                                          │
│                                                                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPORT LAYER (unified_exporter.py)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │
│  │  live_state.json   │  │  historical.json   │  │ model_cache.pkl  │  │
│  ├────────────────────┤  ├────────────────────┤  ├──────────────────┤  │
│  │ • Current ensemble │  │ • Full history of  │  │ • Serialized     │  │
│  │   score & severity │  │   ensemble scores  │  │   models         │  │
│  │ • Top 5 anomalies  │  │ • Regime stats     │  │ • Fast reload    │  │
│  │ • VIX regime       │  │ • Transition probs │  │   (<1 second)    │  │
│  │ • Persistence      │  │ • Feature ranks    │  │                  │  │
│  └────────────────────┘  └────────────────────┘  └──────────────────┘  │
│                                                                           │
└──────────────────┬────────────────────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   DASHBOARD / API    │
        │   Integration        │
        └──────────────────────┘

```

---

## Data Flow Diagram

```
TIME: Day 0 (Initial Training)
═══════════════════════════════════════════════════════════════

1. DATA COLLECTION (5-10 minutes)
   ├─ Fetch SPX/VIX from Yahoo Finance (3769 days)
   ├─ Fetch CBOE files from local archive
   ├─ Fetch FRED series via API
   └─ Fetch futures data
   
2. FEATURE ENGINEERING (2-3 minutes)
   ├─ Calculate VIX features (mean reversion, dynamics, regimes)
   ├─ Calculate SPX features (price, volatility, technical)
   ├─ Calculate cross-asset features (correlations, risk premium)
   ├─ Calculate CBOE features (SKEW, Put/Call, COR)
   ├─ Calculate futures features (spreads, term structure)
   ├─ Calculate meta features (regimes, percentiles, ROC)
   └─ Result: 3769 × 696 feature matrix
   
3. MODEL TRAINING (2-3 minutes)
   ├─ For each of 15 detectors:
   │  ├─ Extract feature subset
   │  ├─ Validate feature quality (remove constant/sparse)
   │  ├─ Scale features (RobustScaler)
   │  ├─ Train Isolation Forest (100 trees)
   │  └─ Calculate feature importance
   │
   ├─ Compute ensemble scores for all historical data
   ├─ Calculate statistical thresholds (MODERATE, HIGH, CRITICAL)
   └─ Compute regime transition statistics
   
4. PERSISTENCE (1-2 seconds)
   ├─ Save models to model_cache.pkl
   ├─ Export historical.json
   └─ Export live_state.json

Total: ~10-15 minutes


TIME: Day 1+ (Daily Refresh)
═══════════════════════════════════════════════════════════════

1. INCREMENTAL DATA FETCH (10-30 seconds)
   ├─ Check cache for last update date
   ├─ Fetch only new data since last update
   └─ Append to existing time series
   
2. INCREMENTAL FEATURE UPDATE (5-10 seconds)
   ├─ Calculate features for new days only
   └─ Append to feature matrix
   
3. DETECTION (1-2 seconds)
   ├─ Load cached models (fast)
   ├─ Run detection on latest day
   └─ Calculate ensemble score
   
4. EXPORT (1 second)
   ├─ Update live_state.json
   └─ Optionally update historical.json

Total: ~20-45 seconds
```

---

## Component Interaction Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTIONS                             │
└───────┬──────────────────────────────────────────────────────────────┘
        │
        ├─── Initialize System ──────► system.initialize()
        │                                │
        │                                ├─► Fetcher: Get all data
        │                                ├─► Engine: Build features
        │                                ├─► Detector: Train models
        │                                └─► Exporter: Save outputs
        │
        ├─── Refresh System ────────────► system.refresh()
        │                                │
        │                                ├─► Fetcher: Get new data only
        │                                ├─► Engine: Update features
        │                                ├─► Detector: Run detection
        │                                └─► Exporter: Update JSON
        │
        ├─── Get Market State ──────────► system.get_market_state()
        │                                │
        │                                └─► Returns: Current anomaly analysis
        │
        ├─── Run Diagnostics ───────────► introspector.generate_report()
        │                                │
        │                                ├─► Health check
        │                                ├─► Performance profiling
        │                                ├─► Data quality analysis
        │                                └─► Recommendations
        │
        └─── Access Raw Data ───────────► system.orchestrator.features
                                         │ system.orchestrator.vix_ml
                                         │ system.orchestrator.spx_ml
                                         └─► Direct DataFrame access
```

---

## Module Dependencies

```
config.py
  ├─ Defines: Regime boundaries, feature groups, thresholds
  └─ Used by: All modules

data_fetcher.py
  ├─ Imports: config (paths, series definitions)
  └─ Used by: integrated_system (orchestrator)

feature_engine.py
  ├─ Imports: config (regime boundaries, training years)
  └─ Used by: integrated_system (orchestrator)

anomaly_detector.py
  ├─ Imports: config (random state, thresholds, feature groups)
  └─ Used by: integrated_system (orchestrator)

unified_exporter.py
  ├─ Imports: config (regime names, boundaries)
  └─ Used by: integrated_system (orchestrator)

integrated_system.py
  ├─ Imports: ALL above modules
  ├─ Coordinates: Data flow between components
  └─ Exposes: User-facing API

enhanced_technical_introspector.py
  ├─ Imports: integrated_system (for inspection)
  └─ Used by: Users (for diagnostics)
```

---

## Feature Engineering Pipeline (Detailed)

```
RAW DATA (VIX, SPX prices)
          ↓
┌─────────────────────────────────────────────────────────┐
│ BASE TRANSFORMATIONS                                    │
├─────────────────────────────────────────────────────────┤
│ • Returns (1d, 5d, 10d, 21d, 63d windows)              │
│ • Moving averages (10, 21, 63, 252 day windows)        │
│ • Volatility (rolling std over various windows)        │
│ • Z-scores (standardized vs rolling mean/std)          │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ DERIVED FEATURES                                        │
├─────────────────────────────────────────────────────────┤
│ • Relative positioning (price vs MA, Bollinger Bands)  │
│ • Rate-of-change (velocity = 1st derivative)           │
│ • Acceleration (2nd derivative)                         │
│ • Jerk (3rd derivative)                                 │
│ • Percentile rankings (vs historical distribution)     │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ REGIME FEATURES                                         │
├─────────────────────────────────────────────────────────┤
│ • Regime classification (pd.cut on boundaries)          │
│ • Regime persistence (days in current regime)           │
│ • Regime transitions (upcoming regime changes)          │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ CROSS-ASSET FEATURES                                    │
├─────────────────────────────────────────────────────────┤
│ • Correlations (rolling windows, multiple pairs)        │
│ • Divergences (when correlations break)                 │
│ • Ratios (VIX/RV, SKEW/VIX, Put/Call)                   │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ META FEATURES                                           │
├─────────────────────────────────────────────────────────┤
│ • Composite regimes (multiple inputs → single regime)   │
│ • Stress indicators (weighted combination of signals)   │
│ • Confluence measures (how many signals agree?)         │
└────────────────────┬────────────────────────────────────┘
                     ↓
              FEATURE MATRIX
              (696 features)
```

---

## Isolation Forest Detection Process

```
INPUT: Feature vector for single day (696 values)
          ↓
┌─────────────────────────────────────────────────────────┐
│ DETECTOR 1: vix_mean_reversion                         │
├─────────────────────────────────────────────────────────┤
│ 1. Extract features: [vix, vix_vs_ma10, vix_vs_ma21...] │
│ 2. Scale using trained RobustScaler                     │
│ 3. Pass through 100 Isolation Trees                     │
│ 4. Calculate anomaly score (path length metric)         │
│ 5. Convert to percentile (vs training distribution)     │
│ Result: 35.3% anomaly                                   │
└────────────────────┬────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ DETECTOR 2-15: (repeat for each detector)              │
│ ...                                                      │
└────────────────────┬────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ ENSEMBLE AGGREGATION                                    │
├─────────────────────────────────────────────────────────┤
│ For each detector:                                       │
│   weight = coverage × quality_penalty                    │
│                                                          │
│ Ensemble = (score₁×w₁ + score₂×w₂ + ... + score₁₅×w₁₅) │
│            ─────────────────────────────────────────────  │
│                    (w₁ + w₂ + ... + w₁₅)                │
│                                                          │
│ Result: 38.1% ensemble anomaly                          │
└────────────────────┬────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ SEVERITY CLASSIFICATION                                 │
├─────────────────────────────────────────────────────────┤
│ IF ensemble < 70%:    NORMAL                            │
│ IF ensemble < 78%:    MODERATE                          │
│ IF ensemble < 93%:    HIGH                              │
│ ELSE:                 CRITICAL                          │
│                                                          │
│ Result: NORMAL (38.1%)                                  │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
project/
│
├── config.py                      # Configuration constants
│
├── data_cache/                    # Cached API responses
│   ├── spx_cache.pkl
│   ├── vix_cache.pkl
│   ├── fred_*.pkl
│   └── ...
│
├── CBOE_Data_Archive/             # Historical CBOE data
│   ├── SKEW_INDEX_CBOE.csv
│   ├── PCCI_INDX_CBOE.csv
│   └── ...
│
├── core/                          # Main system modules
│   ├── data_fetcher.py
│   ├── feature_engine.py
│   ├── anomaly_detector.py
│   └── __init__.py
│
├── export/                        # Export utilities
│   ├── unified_exporter.py
│   └── __init__.py
│
├── integrated_system_production.py  # Main orchestrator
│
├── outputs/                       # System outputs
│   ├── live_state.json           # Current state
│   ├── historical.json           # Full history
│   └── model_cache.pkl           # Trained models
│
├── docs/                          # Documentation
│   ├── TECHNICAL_DIAGNOSTIC.md   # Auto-generated diagnostics
│   ├── BUSINESS_TECHNICAL_OVERVIEW.md  # Business overview
│   └── SYSTEM_EXPLAINED.md       # Original self-doc
│
└── enhanced_technical_introspector.py  # Diagnostic tool
```

---

*This architecture document provides a visual overview of the complete system. For detailed explanations, see BUSINESS_TECHNICAL_OVERVIEW.md*
