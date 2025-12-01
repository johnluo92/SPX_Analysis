# ANOMALY DETECTION SYSTEM - BUSINESS SPECIFICATION
**Version:** 2.0  
**Created:** 2025-12-01  
**System:** VIX Forecasting Infrastructure  
**Purpose:** Spike Detection & Intraday Alert Generation

---

## EXECUTIVE SUMMARY

The anomaly detection system operates as a parallel intelligence layer within the VIX forecasting infrastructure. While the dual XGBoost model (magnitude regressor + direction classifier) excels at predicting normal regime transitions and VIX mean reversion, it systematically fails to detect VIX spikes (catching 0-7% of spikes vs 100% of crashes). The anomaly system fills this critical gap.

**Two-System Architecture:**
- **Intraday System:** Real-time anomaly detection generating live alerts for immediate trading
- **EOD Feature System:** Daily anomaly scores that become input features for the dual model's next-day predictions

**Critical Constraint:** The system must NOT interfere with the existing dual model's proven performance (70% accuracy, well-calibrated confidence). It operates in parallel, not as a replacement.

---

## CURRENT STATE ASSESSMENT

### Existing Anomaly Detector (Shelved)
Located in `anomaly_detector.py`, the current implementation:
- **Architecture:** Multi-dimensional IsolationForest with domain-specific feature groups + random subspaces
- **Feature Groups:** Defined in config but not integrated with production feature pipeline
- **Strengths:** 
  - Robust quality validation per feature group
  - Statistical threshold calculation from training distribution
  - Persistence tracking with timezone awareness
  - SHAP/permutation feature importance
  - Coverage-weighted ensemble scoring
- **Weaknesses:**
  - Disconnected from production infrastructure
  - No intraday capabilities
  - Feature groups don't leverage 331-feature production pipeline
  - No integration with PredictionDatabase
  - No spike-specific optimization
  - Operates in isolation (no feedback loop with dual model)

### Production Infrastructure (Current)
The system has enterprise-grade components ready for integration:

**Data Pipeline:**
- `UnifiedDataFetcher`: Multi-source data retrieval (FRED, Yahoo, CBOE, VX futures)
- `FeatureEngineer`: Generates 331 features across 6 domains with temporal safety
- `TemporalSafetyValidator`: Ensures no lookahead bias with publication lag handling

**Model Infrastructure:**
- `SimplifiedVIXForecaster`: Dual XGBoost model (magnitude + direction)
- `ForecastCalibrator`: Regime/cohort-specific bias correction using exponential decay
- `RegimeClassifier`: 4-regime classification with transition probabilities and persistence metrics

**Storage & Retrieval:**
- `PredictionDatabase`: SQLite backend with proper schema, indexing, and commit tracking
- `prediction_id` scheme: `pred_YYYYMMDD_h5` for 5-day horizon forecasts
- Automatic backfilling of actuals from Yahoo Finance
- Export capabilities for analysis

**Calendar System:**
- FOMC meetings from CSV with CPI/PCE/minutes tracking
- Monthly OpEx calendar (3rd Friday generator)
- VIX futures expiry calendar (Wednesday 30 days prior to OpEx)
- Earnings intensity scoring (peak seasons: Jan/Apr/Jul/Oct)
- Cohort priority hierarchy with weights

---

## PROBLEM DEFINITION

### The Spike Detection Gap
**Observed Performance (2024-2025 Live Data):**
```
Tuner 1: Missed 19/19 spikes (0.0% caught)
Tuner 2: Missed 26/28 spikes (7.1% caught)  
Tuner 3: Missed 30/31 spikes (3.2% caught)

All tuners: 100% crash capture (mean reversion)
```

**Root Cause Analysis:**
1. **Base Rate Bias:** Models trained on 80% DOWN days learn to predict DOWN
2. **Feature Informativeness Asymmetry:** 
   - VIX crashes = predictable from term structure (contango → mean reversion)
   - VIX spikes = tail events driven by shocks not captured in features
3. **Missing Precursor Signals:**
   - Rapid acceleration in velocity features
   - Positioning imbalances
   - Correlation breakdown
   - Liquidity stress accumulation

**Financial Impact:**
- Premium collection strategies (selling puts/calls) work beautifully in normal regimes
- One missed VIX spike during short volatility position = months of gains wiped out
- Cannot trade system profitably without spike protection

---

## SOLUTION ARCHITECTURE

### Two Independent Systems, One Infrastructure

```
INTRADAY SYSTEM (Real-Time Alerts)
  ↓
  Monitors: VIX, SPX, VVIX, Term Structure, Credit Spreads
  ↓
  Generates: Spike alerts for immediate trading decisions
  ↓
  Logs: Alert events to database

EOD SYSTEM (Feature Generation)
  ↓
  Aggregates: Intraday anomaly signals + EOD feature state
  ↓
  Calculates: Daily anomaly score (0.0-1.0)
  ↓
  Stores: Score as feature in next-day prediction row
  ↓
  Feeds: Dual XGBoost model for 5-day forecast
```

**Key Principle:** Two timeframes, two uses, one data pipeline
- Intraday: You trade the alerts
- EOD: Model learns from yesterday's anomaly state to predict next 5 days

---

## FUNCTIONAL REQUIREMENTS

### 1. INTRADAY ANOMALY DETECTION SYSTEM

**Purpose:** Generate real-time alerts for spike risk

**Input Sources (Live):**
- VIX current level + 5-minute velocity
- SPX price action + realized volatility
- VVIX (VIX of VIX)
- VX1-VX2 term structure
- HY credit spreads (if available intraday)
- Single-stock IV (AAPL, NVDA as SPX leaders)

**Detection Methodology:**
- IsolationForest trained on velocity/acceleration features
- Contamination rate: 5% (assume 5% of days have spike risk)
- Real-time scoring every 5-30 minutes during market hours (9:30-16:00 ET)

**Alert Thresholds:**
```
MODERATE:  Anomaly score > 75th percentile (consider reducing exposure)
HIGH:      Anomaly score > 90th percentile (hedge immediately)  
CRITICAL:  Anomaly score > 95th percentile (full spike protection)
```

**Alert Structure:**
```
{
  "timestamp": "2025-12-01T14:23:15-05:00",
  "alert_level": "HIGH",
  "anomaly_score": 0.87,
  "contributing_factors": [
    "vix_accel_5min: 3.2σ above normal",
    "vx_contango_collapse: -2.1 → -0.3 in 30min",
    "vvix_velocity: 95th percentile"
  ],
  "current_vix": 18.45,
  "action": "Consider long VIX calls or reduce short vol exposure"
}
```

**Database Schema (New Table):**
```sql
CREATE TABLE intraday_alerts (
    alert_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    alert_level TEXT,  -- MODERATE, HIGH, CRITICAL
    anomaly_score REAL,
    current_vix REAL,
    vix_regime TEXT,
    contributing_features TEXT,  -- JSON array
    resolution_time DATETIME,  -- When alert cleared
    outcome TEXT,  -- spike_occurred, false_alarm, no_resolution
    spike_magnitude REAL,  -- Actual VIX change if spike occurred
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alert_timestamp ON intraday_alerts(timestamp);
CREATE INDEX idx_alert_level ON intraday_alerts(alert_level);
CREATE INDEX idx_alert_outcome ON intraday_alerts(outcome);
```

**Performance Tracking:**
Track alert effectiveness:
- True positives: Alert → VIX spike within 1-3 hours
- False positives: Alert → No spike
- False negatives: Spike with no prior alert
- Lead time: Minutes between alert and spike onset

### 2. EOD ANOMALY SCORE CALCULATION

**Purpose:** Create daily anomaly feature for dual model consumption

**Calculation Logic:**
Aggregate intraday signals into single EOD score:

```
EOD_anomaly_score = weighted_average(
    max_intraday_score: 0.40,      # Peak anomaly during day
    mean_alert_level: 0.30,        # Average severity
    alert_persistence: 0.20,       # How long alerts persisted
    eod_feature_state: 0.10        # EOD velocity/acceleration state
)
```

**Normalization:**
- Scale to [0.0, 1.0] using training distribution percentiles
- Smooth extreme outliers (winsorize at 99th percentile)
- 7-day exponential moving average for stability

**Integration with Prediction Database:**
Extend `forecasts` table schema:
```sql
ALTER TABLE forecasts ADD COLUMN anomaly_score_prior_day REAL;
ALTER TABLE forecasts ADD COLUMN anomaly_alerts_prior_day INTEGER;
ALTER TABLE forecasts ADD COLUMN max_alert_level_prior_day TEXT;
```

When generating next-day forecast:
1. Retrieve yesterday's EOD anomaly score
2. Include as feature in XGBoost input (both magnitude and direction models)
3. Store in prediction row for analysis

**Expected Model Behavior:**
- High anomaly score → XGB learns to predict higher UP probability
- Persistent anomalies → XGB learns to increase magnitude forecast
- No architectural changes to dual model (just new input feature)

### 3. SPIKE GATE LOGIC

**Purpose:** Override dual model predictions when extreme anomaly detected

**Activation Conditions:**
```
IF anomaly_score > 0.90 AND dual_model predicts DOWN:
    OVERRIDE to NEUTRAL (no trade)
    Reason: Models don't handle spikes, high anomaly = uncertainty

IF anomaly_score > 0.95 AND dual_model predicts DOWN:
    OVERRIDE to UP
    Boost magnitude by 1.3x
    Confidence = 0.65 (moderate, not high)
    Reason: Extreme anomaly = likely spike incoming

IF anomaly_score > 0.85 AND dual_model predicts UP:
    CONFIRM prediction
    Boost magnitude by 1.2x  
    Increase confidence by 20% (cap at 0.95)
    Reason: Anomaly confirms dual model's UP call
```

**Implementation Location:**
- Add method to `SimplifiedVIXForecaster.predict()`
- Check anomaly score before returning final forecast
- Log all overrides to database for analysis

**Database Schema (Extend forecasts):**
```sql
ALTER TABLE forecasts ADD COLUMN spike_gate_triggered BOOLEAN;
ALTER TABLE forecasts ADD COLUMN spike_gate_action TEXT;  -- OVERRIDE_TO_UP, OVERRIDE_TO_NEUTRAL, BOOST_CONFIRM
ALTER TABLE forecasts ADD COLUMN spike_gate_reason TEXT;
```

### 4. FEATURE ENGINEERING FOR ANOMALY DETECTION

**Velocity Features (Add to FeatureEngineer):**
Already have `vix_velocity_5d`, expand with:
- `vix_accel_5d`: Second derivative (velocity of velocity)
- `vix_accel_21d`: Longer-term acceleration
- `vix_jerk_5d`: Third derivative (already exists for SKEW, add for VIX)

**VVIX Features (Expand existing):**
Currently have VVIX level, add:
- `vvix_velocity_5d`: VVIX percent change
- `vviv_accel_5d`: VVIX acceleration
- `vvix_spike_regime`: Binary flag when VVIX > 2.5x rolling median

**Term Structure Collapse Features:**
Currently have `VX1-VX2`, enhance:
- `vx_contango_velocity`: Rate of contango collapse
- `vx_contango_accel`: Acceleration of term structure movement
- `vx_backwardation_extreme`: Binary flag when VX1-VX2 > 1.5 (severe backwardation)

**Credit Spread Velocity:**
Currently have HY OAS level, add:
- `hy_oas_velocity_5d`: Credit spread widening rate
- `hy_oas_velocity_21d`: Medium-term credit stress velocity
- `credit_stress_acceleration`: Second derivative of HY spreads

**Realized-Implied Divergence:**
- `rv_iv_spread`: SPX realized vol - VIX (already have components)
- `rv_iv_spread_velocity`: Rate of divergence change
- `rv_iv_extreme_divergence`: Binary flag when spread > 90th percentile

**Regime Transition Features:**
- `regime_transitions_21d`: Count of regime changes in 21 days (already have components)
- `regime_volatility`: Frequency of regime bouncing (indicator of instability)

**Multi-Asset Correlation Breakdown:**
- `spx_vix_corr_breakdown`: Deviation from -0.7 normal correlation
- `dxy_crude_decorrelation`: Dollar-oil relationship breakdown (risk-off signal)

**Positioning/Flow Features (If Data Available):**
- CFTC VX futures positioning (net speculative)
- CBOE volume/OI ratios
- Single-stock IV percentile ranks (AAPL, NVDA)

### 5. TRAINING & CALIBRATION

**IsolationForest Training:**
- **Training Data:** Use ALL historical data (unsupervised, no need for spike labels)
- **Features:** 15-20 velocity/acceleration features + production 331 features (high dimensional)
- **Contamination:** 0.05 (assume 5% of days are anomalous)
- **n_estimators:** 200 trees
- **max_samples:** 256 samples per tree
- **Random state:** 42 (reproducibility)

**Statistical Threshold Calculation:**
After training, calculate percentile thresholds from training scores:
```
moderate_threshold = 75th percentile of anomaly scores
high_threshold = 90th percentile
critical_threshold = 95th percentile
```

Store thresholds in model metadata for consistent alert generation.

**Retraining Cadence:**
- Monthly retraining on expanding window
- Triggered retraining after regime change (VIX enters Elevated/Crisis for 20+ days)
- Emergency retraining if false alarm rate exceeds 70% over 30 days

**Validation Metrics:**
Track on holdout 2024-2025 data:
- Precision: % of alerts followed by actual spikes
- Recall: % of spikes with prior alerts  
- Lead time: Average minutes between alert and spike
- False alarm rate: % of alerts with no spike
- **Target:** 50% precision, 70% recall (better to over-alert than miss spikes)

### 6. INTEGRATION POINTS WITH EXISTING SYSTEM

**FeatureEngineer Integration:**
Add new method `build_anomaly_features()`:
- Generates all velocity/acceleration features
- Returns DataFrame aligned to production index
- Called during `build_complete_features()`
- Features automatically included in 331+ feature set

**PredictionDatabase Integration:**
Extend schema as specified in Section 2 and 3.
Add methods:
- `store_intraday_alert()`: Log alert events
- `get_eod_anomaly_score()`: Retrieve yesterday's score for forecast input
- `get_alert_performance()`: Calculate precision/recall metrics

**SimplifiedVIXForecaster Integration:**
Modify `predict()` method:
1. Check if anomaly_score feature exists in input
2. If anomaly_score > 0.85, activate spike gate logic
3. Log spike gate actions to database
4. Return modified forecast with spike gate metadata

**Integrated System Workflow:**
Modify `integrated_system.py`:
```
1. Build features (including anomaly features)
2. Load dual XGBoost models
3. Load anomaly detector (IsolationForest)
4. Generate dual model forecast
5. Get yesterday's EOD anomaly score from database
6. Apply spike gate if anomaly_score > 0.85
7. Store final prediction with anomaly metadata
8. Generate EOD anomaly score for next-day use
```

### 7. PRODUCTION DEPLOYMENT

**File Structure:**
```
src/core/
├── anomaly_detector_v2.py       # New implementation
├── anomaly_intraday.py          # Intraday alert system
├── anomaly_eod.py               # EOD score calculator
└── anomaly_spike_gate.py        # Override logic

models/
├── isolation_forest.pkl         # Trained IF model
├── anomaly_thresholds.json      # Statistical thresholds
└── anomaly_metadata.json        # Training info, feature importance

config.py additions:
- ANOMALY_DETECTION_CONFIG
- SPIKE_GATE_CONFIG
- INTRADAY_ALERT_CONFIG
```

**Configuration Schema:**
```python
ANOMALY_DETECTION_CONFIG = {
    'enabled': True,
    'contamination': 0.05,
    'n_estimators': 200,
    'random_state': 42,
    'feature_set': 'velocity_acceleration',  # vs 'full_331'
    'retraining_frequency': 'monthly',
    'alert_mode': 'intraday',  # vs 'eod_only'
}

SPIKE_GATE_CONFIG = {
    'enabled': True,
    'override_threshold': 0.90,
    'extreme_override_threshold': 0.95,
    'boost_factor': 1.3,
    'confidence_override': 0.65,
}

INTRADAY_ALERT_CONFIG = {
    'enabled': False,  # Start disabled for testing
    'check_interval_minutes': 15,
    'market_hours': {'start': '09:30', 'end': '16:00', 'timezone': 'US/Eastern'},
    'alert_thresholds': {
        'moderate': 0.75,
        'high': 0.90, 
        'critical': 0.95
    },
    'notification_channels': ['database', 'log'],  # Future: ['email', 'slack', 'sms']
}
```

**Testing Protocol:**
1. **Backtest on 2024-2025:** 
   - Verify spike gate catches missed spikes
   - Measure false alarm rate
   - Ensure no degradation of dual model performance on normal days

2. **Paper Trading:**
   - Run intraday system in shadow mode (alerts logged, not acted on)
   - Compare alerts to actual VIX moves
   - Tune thresholds based on precision/recall

3. **Gradual Rollout:**
   - Week 1: EOD scores only (no spike gate, just feature input)
   - Week 2: Spike gate enabled in conservative mode (log overrides, don't execute)
   - Week 3: Full spike gate enabled
   - Week 4: Intraday alerts enabled

---

## NON-FUNCTIONAL REQUIREMENTS

### Performance
- Intraday anomaly scoring: <1 second per check
- EOD aggregation: <5 seconds
- IsolationForest prediction: <50ms per sample
- No impact on dual model inference time

### Reliability
- Alert system must handle missing data gracefully
- Fallback to dual model if anomaly system fails
- Automatic retraining if performance degrades

### Maintainability  
- Clear separation between intraday and EOD systems
- Modular architecture (can disable spike gate independently)
- Comprehensive logging of all decisions
- Feature importance tracking for debugging

### Monitoring
Track daily:
- Alert count by level
- False alarm rate (rolling 30-day)
- Spike catch rate (when spikes occur)
- Spike gate override count
- Impact on dual model predictions (override %)

### Fail-Safe Modes
If anomaly system fails:
1. Log error to database
2. Continue with dual model predictions (no spike gate)
3. Send notification (email/Slack)
4. Automatic fallback to last known good model

---

## SUCCESS CRITERIA

### Phase 1: EOD Feature Integration (Month 1)
- [ ] EOD anomaly scores calculated for all historical dates
- [ ] Anomaly scores stored in prediction database
- [ ] Dual model retraining with anomaly feature included
- [ ] No degradation in dual model performance (70% accuracy maintained)
- [ ] Anomaly feature importance ranked in top 50 features

### Phase 2: Spike Gate Deployment (Month 2)
- [ ] Spike gate logic integrated into prediction workflow
- [ ] Catch rate on 2024-2025 spikes: >50% (vs 0-7% baseline)
- [ ] False override rate: <15% (don't override too often)
- [ ] Override decisions logged with full metadata
- [ ] Dual model performance on non-spike days: unchanged

### Phase 3: Intraday Alerts (Month 3)
- [ ] Intraday system running in production during market hours
- [ ] Alerts logged to database with timestamps
- [ ] Alert precision: >40% (alerts followed by spikes)
- [ ] Alert recall: >60% (spikes with prior alerts)
- [ ] Lead time: 30+ minutes average between alert and spike
- [ ] Integration with notification system (email/Slack)

### Phase 4: Live Trading (Month 4+)
- [ ] 3 months of paper trading validation
- [ ] Risk management rules defined (position sizing, max loss)
- [ ] Spike protection strategy tested (long VIX calls, short vol reduction)
- [ ] P&L tracking vs baseline (no anomaly system)
- [ ] Sharpe ratio improvement on volatility strategies

---

## DESIGN PRINCIPLES FOR IMPLEMENTATION

**For the Next Claude Instance:**

### 1. Infrastructure-First Approach
Start by understanding these existing systems:
- How FeatureEngineer generates features (temporal safety, publication lags)
- How PredictionDatabase stores and retrieves data (schema, indexing, commit protocol)
- How SimplifiedVIXForecaster makes predictions (dual model, ensemble scoring)
- How ForecastCalibrator adjusts forecasts (regime/cohort corrections)

Don't rebuild what exists. Extend it.

### 2. Minimal Disruption Philosophy
The dual XGBoost model works (70% accuracy, well-calibrated). Your job is NOT to fix it.
Your job is to catch what it misses (spikes) without breaking what it does well.

Rules:
- Anomaly system operates in parallel, not as replacement
- If anomaly system fails, dual model continues unaffected
- Test extensively before enabling spike gate overrides
- Always provide escape hatch to disable anomaly features

### 3. Separation of Concerns
Three independent modules:
1. **Intraday Alert System:** Real-time detection, no dependency on dual model
2. **EOD Score Calculator:** Aggregation logic, writes to database
3. **Spike Gate Logic:** Override rules, integrates with dual model

Each should be testable independently.

### 4. Data Quality Obsession
Before anomaly detection:
- Validate feature quality (use existing TemporalSafetyValidator)
- Check for missing data (VVIX, VX futures)
- Handle publication lags correctly
- Ensure no lookahead bias

Bad anomaly scores are worse than no anomaly scores.

### 5. Explainability Requirements
Every alert must explain itself:
- Which features contributed most?
- What were the z-scores of key indicators?
- Historical context (has this happened before?)

Use SHAP values or permutation importance (already implemented).
Traders need to understand WHY system is alerting.

### 6. Conservative Thresholds Initially
Better to miss spikes during testing than cause false alarms.
Start with high thresholds (95th percentile for alerts).
Lower gradually as confidence builds.

### 7. Comprehensive Logging
Log everything:
- Every anomaly score calculation
- Every alert generated
- Every spike gate activation
- Every override decision

Use PredictionDatabase for persistence.
Analysis of edge cases drives system improvement.

### 8. Feature Set Strategy
Don't use all 331 production features for anomaly detection.
Focus on:
- Velocity/acceleration features (15-20)
- Term structure features (5-10)  
- Credit spread features (5-10)
- Correlation features (5-10)

Total: 40-60 features optimized for spike precursors.
IsolationForest handles high dimensionality but focus is better.

### 9. Avoid Overfitting to 2024-2025
You have 20 years of VIX history in the system.
Use all of it for training.
2024-2025 is for validation only.

Spikes are rare. You need maximum data.

### 10. Production Readiness Checklist
Before deployment:
- [ ] Unit tests for all anomaly functions
- [ ] Integration tests with existing infrastructure
- [ ] Backtest on full historical data (2004-2025)
- [ ] Paper trading validation (3 months minimum)
- [ ] Documentation (docstrings, README, architecture diagrams)
- [ ] Error handling (graceful degradation)
- [ ] Performance profiling (no bottlenecks)
- [ ] Monitoring dashboard (alert metrics, system health)

---

## OPEN QUESTIONS FOR IMPLEMENTATION

These are design decisions the implementing engineer should make:

1. **Feature Set Composition:** Use 40-60 velocity features or full 331 production features?
   - Tradeoff: Specificity vs completeness
   - Test both, compare performance

2. **Ensemble Weighting:** How to combine multiple IsolationForest detectors?
   - Current system uses coverage-weighted average
   - Alternative: Max pooling (most anomalous detector wins)
   - Alternative: Hierarchical (domain → global)

3. **Spike Definition:** What constitutes a "spike" for validation?
   - Option A: VIX > 5% 1-day change
   - Option B: VIX > 30 absolute level
   - Option C: VIX enters Elevated/Crisis regime
   - Likely: Composite definition (any of above)

4. **Alert Persistence Logic:** How long should alerts last?
   - Auto-expire after 4 hours?
   - Remain until VIX resolves?
   - Require explicit cancellation?

5. **Intraday Data Frequency:** Every 5 minutes? 15 minutes? 30 minutes?
   - Tradeoff: Responsiveness vs noise
   - API rate limits (if using real-time feeds)

6. **Spike Gate Override Conservatism:** Should we override dual model or just add caution flag?
   - Safe: Flag only, let trader decide
   - Aggressive: Full override per specification
   - Middle: Override in Elevated/Crisis only

7. **Retraining Triggers:** When to retrain IsolationForest?
   - Monthly? Quarterly? Regime-change-triggered?
   - Performance degradation threshold?

8. **Historical Lookback for Thresholds:** Calculate percentiles on 1 year? 5 years? All data?
   - Longer = more stable, but misses regime shifts
   - Shorter = adaptive, but noisy

---

## REFERENCES TO EXISTING CODEBASE

**Key Files to Study:**
1. `/src/core/feature_engineer.py`: How features are built (lines 1-2000+, comprehensive)
2. `/src/core/xgboost_trainer_v3.py`: Dual model architecture and prediction flow
3. `/src/core/prediction_database.py`: Database schema, methods, commit protocol
4. `/src/core/forecast_calibrator.py`: How forecasts are adjusted post-hoc
5. `/src/core/regime_classifier.py`: Regime logic and transition probabilities
6. `/src/integrated_system.py`: System orchestration and workflow
7. `/src/config.py`: All configuration constants

**Existing Patterns to Follow:**
- Feature naming: `{indicator}_{transformation}_{window}d` (e.g., `vix_velocity_5d`)
- Database columns: snake_case, explicit data types
- Model saving: Pickle for models, JSON for metadata
- Logging: Use Python logging module, INFO level for key events
- Error handling: Try-except with informative error messages, never fail silently
- Documentation: Docstrings on all classes/methods, inline comments for complex logic

**Configuration Additions Needed:**
```python
# In config.py
ANOMALY_FEATURE_SET = [
    'vix_velocity_5d', 'vix_accel_5d', 'vix_accel_21d',
    'vvix_velocity_5d', 'vvix_accel_5d', 
    'vx_contango_velocity', 'vx_contango_accel',
    'hy_oas_velocity_5d', 'hy_oas_velocity_21d',
    'rv_iv_spread_velocity', 'skew_velocity_5d',
    'regime_transitions_21d',
    # ... expand based on testing
]

ANOMALY_THRESHOLDS = {
    'moderate': 0.75,  # 75th percentile
    'high': 0.90,
    'critical': 0.95
}
```

---

## FINAL NOTES

**What Makes This Different From Old Anomaly Detector:**
1. **Infrastructure Integration:** Not standalone, fully integrated with production pipeline
2. **Dual Purpose:** Both intraday alerts AND EOD features (old system: detection only)
3. **Feature Set:** Uses production features + spike-optimized velocity features (old system: custom groups)
4. **Database Integration:** Stores alerts, scores, and metadata in PredictionDatabase (old system: isolated)
5. **Spike Gate Logic:** Actively modifies dual model predictions (old system: detection only, no action)
6. **Production Ready:** Designed for live trading with fail-safes (old system: research prototype)

**What NOT to Change:**
- Dual XGBoost model architecture (magnitude + direction)
- Feature engineering pipeline (temporal safety, publication lags)
- Calibration system (regime/cohort corrections)
- Database schema for existing tables (extend, don't break)
- Configuration structure (add, don't replace)

**What Freedom You Have:**
- IsolationForest hyperparameters (tune for performance)
- Feature selection for anomaly detection (test different combinations)
- Alert threshold values (optimize for precision/recall tradeoff)
- Spike gate logic (conservative vs aggressive overrides)
- Intraday check frequency (balance responsiveness vs noise)
- Ensemble weighting schemes (coverage-weighted vs max-pooling vs hierarchical)

**Philosophy:**
Build the anomaly system like a co-pilot, not a replacement pilot.
The dual model flies the plane in normal conditions.
The anomaly system watches for storms and alerts the pilot.
In extreme weather (95th percentile anomaly), the co-pilot can take over.
But the pilot (dual model) is always in command unless override is absolutely necessary.

---

**APPENDIX: Previous Conversation Context**

User asked for two core problems to be solved:
1. **Ensemble scoring rebuild:** How to combine XGBRegressor + XGBClassifier + IsolationForest properly
2. **Daily anomaly score calculation:** Aggregate intraday signals into EOD feature

User's key insight:
> "the only thing that can predict spikes is the velocity of vix... calling upmoves on the vix is a death sentence... imagine buying protection every time you think there is spike but when there isn't you just bleed theta"

System already has velocity features. Need acceleration (velocity of velocity).
User understands asymmetric payoff: Better to miss 10 UPs and catch 26/28 than call 100 UPs and be wrong on 50.

Architecture confirmed:
- Intraday: Anomaly system runs live, generates alerts, user trades
- EOD: Anomaly score logged as feature
- Next day: Dual model ingests yesterday's anomaly score, forecasts next 5 days
- Two systems, two timeframes, one feeds the other

User's final quote:
> "What is dead may never die, but rises again, harder and stronger."

The old anomaly detector is dead. This specification is for its resurrection, battle-hardened and production-ready.

---

**END SPECIFICATION**
