# ANOMALY SYSTEM REWRITE - QUICK START GUIDE

**Read this first. Then read the full specification.**

---

## THE PROBLEM (One Sentence)
Your dual XGBoost model catches 100% of VIX crashes but misses 93-100% of VIX spikes, making it untradeble without spike protection.

## THE SOLUTION (One Paragraph)
Build a parallel IsolationForest-based anomaly detection system that operates on two timeframes: (1) Intraday alerts for immediate trading decisions, and (2) EOD anomaly scores that become input features for the dual model's next-day predictions. The system detects spikes the dual model can't see, without degrading the dual model's proven 70% accuracy on normal regime transitions.

---

## ARCHITECTURE IN 30 SECONDS

```
┌─────────────────────────────────────────────────┐
│   INTRADAY SYSTEM (Real-Time)                  │
│   Monitors: VIX velocity, term structure       │
│   Outputs: Spike alerts (MODERATE/HIGH/CRITICAL)│
│   Purpose: Trade these alerts                   │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│   EOD SYSTEM (Once Daily)                      │
│   Aggregates: Intraday alerts + EOD features   │
│   Outputs: Daily anomaly score (0.0-1.0)       │
│   Purpose: Feed to dual model as feature        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│   SPIKE GATE (Prediction Override)             │
│   IF anomaly_score > 0.90 AND model says DOWN: │
│   THEN override to NEUTRAL or UP               │
│   Purpose: Protect against missed spikes       │
└─────────────────────────────────────────────────┘
```

**Key Principle:** Two systems, two timeframes, one infrastructure. Anomaly system operates in parallel, never replaces dual model.

---

## YOUR MISSION (Three Deliverables)

### 1. Core Anomaly Detector (`anomaly_detector_v2.py`)
- IsolationForest trained on velocity/acceleration features
- Contamination=0.05, n_estimators=200
- Statistical threshold calculation (75th/90th/95th percentiles)
- Feature importance tracking (SHAP or permutation)
- Integration with FeatureEngineer (use production 331 features + spike-specific)

### 2. EOD Score Calculator (`anomaly_eod.py`)
- Aggregate intraday signals into daily score
- Store in PredictionDatabase as new column: `anomaly_score_prior_day`
- Generate for all historical dates (backfill)
- Integrate into dual model input pipeline

### 3. Spike Gate Logic (`anomaly_spike_gate.py`)
- Override logic in SimplifiedVIXForecaster.predict()
- Activates when anomaly_score > 0.85-0.95
- Three modes: OVERRIDE_TO_UP, OVERRIDE_TO_NEUTRAL, BOOST_CONFIRM
- Log all overrides to database for analysis

**Optional/Future:** Intraday alert system with live monitoring (can be delayed, focus on EOD first)

---

## WHAT YOU HAVE (Existing Infrastructure)

**Ready to Use:**
- `FeatureEngineer`: 331 features across 6 domains, temporal safety guaranteed
- `PredictionDatabase`: SQLite with proper schema, just extend columns
- `SimplifiedVIXForecaster`: Dual XGBoost (magnitude + direction), 70% accurate
- `ForecastCalibrator`: Regime/cohort bias correction with exponential decay
- `RegimeClassifier`: 4 regimes with transition probabilities
- `UnifiedDataFetcher`: FRED, Yahoo, CBOE data with caching

**Study These Files:**
1. `/src/core/feature_engineer.py` - How features are built
2. `/src/core/xgboost_trainer_v3.py` - Dual model prediction flow
3. `/src/core/prediction_database.py` - Database methods and schema
4. `/src/integrated_system.py` - System orchestration

---

## WHAT YOU DON'T HAVE (Build This)

**New Features Needed (Add to FeatureEngineer):**
```
Velocity/Acceleration (15-20 features):
- vix_accel_5d, vix_accel_21d (second derivative)
- vix_jerk_5d (third derivative)
- vvix_velocity_5d, vvix_accel_5d
- vx_contango_velocity, vx_contango_accel
- hy_oas_velocity_5d, hy_oas_velocity_21d
- rv_iv_spread_velocity
- skew_velocity_5d
- regime_transitions_21d
```

**Database Schema Extensions:**
```sql
-- EOD scores
ALTER TABLE forecasts ADD COLUMN anomaly_score_prior_day REAL;
ALTER TABLE forecasts ADD COLUMN anomaly_alerts_prior_day INTEGER;

-- Spike gate tracking  
ALTER TABLE forecasts ADD COLUMN spike_gate_triggered BOOLEAN;
ALTER TABLE forecasts ADD COLUMN spike_gate_action TEXT;

-- New table for intraday alerts (future)
CREATE TABLE intraday_alerts (
    alert_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    alert_level TEXT,
    anomaly_score REAL,
    current_vix REAL,
    contributing_features TEXT
);
```

---

## IMPLEMENTATION SEQUENCE (How to Not Get Overwhelmed)

**Week 1: Feature Engineering**
- [ ] Add velocity/acceleration features to FeatureEngineer
- [ ] Test on historical data (2004-2025)
- [ ] Verify no lookahead bias (use TemporalSafetyValidator)
- [ ] Confirm features generated correctly for all dates

**Week 2: Anomaly Detector Core**
- [ ] Build IsolationForest trainer (use existing `anomaly_detector.py` as reference)
- [ ] Train on full historical data
- [ ] Calculate statistical thresholds (75th/90th/95th percentiles)
- [ ] Save model + metadata (pickle + JSON)
- [ ] Test prediction speed (<50ms per sample)

**Week 3: EOD Score Integration**
- [ ] Build EOD score calculator
- [ ] Backfill anomaly scores for all historical forecasts
- [ ] Extend PredictionDatabase schema
- [ ] Integrate into forecast generation pipeline
- [ ] Verify scores stored correctly

**Week 4: Spike Gate Logic**
- [ ] Add spike gate to SimplifiedVIXForecaster.predict()
- [ ] Test override logic on 2024-2025 data
- [ ] Measure spike catch rate (target: >50% vs 0-7% baseline)
- [ ] Log all overrides with metadata
- [ ] Validate no degradation on non-spike days

**Week 5+: Validation & Tuning**
- [ ] Backtest full system on 2004-2025
- [ ] Tune thresholds for optimal precision/recall
- [ ] Paper trading validation (shadow mode)
- [ ] Document everything
- [ ] Deploy to production

---

## SUCCESS METRICS (How You Know It Works)

**Minimum Viable Product:**
- [ ] EOD anomaly scores calculated for all dates
- [ ] Spike catch rate on 2024-2025: >50% (vs 0-7% baseline)
- [ ] False override rate: <15%
- [ ] Dual model accuracy on non-spike days: 70% maintained
- [ ] System runs without errors for 30 days

**Production Ready:**
- [ ] Alert precision: >40%
- [ ] Alert recall: >60%
- [ ] Lead time: 30+ minutes average
- [ ] Feature importance documented
- [ ] Comprehensive logging implemented
- [ ] Error handling (graceful degradation)

---

## KEY CONSTRAINTS (Don't Break These)

### ❌ DO NOT:
- Replace the dual XGBoost model (it works, just enhance it)
- Modify existing database schema (extend only)
- Change FeatureEngineer's temporal safety logic
- Introduce lookahead bias
- Remove or rename existing config variables
- Deploy without 3 months paper trading

### ✅ DO:
- Use existing infrastructure (don't reinvent the wheel)
- Follow existing naming conventions (`feature_transformation_window`)
- Log everything to database
- Handle missing data gracefully
- Test extensively on historical data
- Provide escape hatch (can disable anomaly system)

---

## CRITICAL IMPLEMENTATION DETAILS

### Feature Set Strategy
**Option A:** Use 40-60 velocity features only (focused, interpretable)
**Option B:** Use full 331 production features (comprehensive, black box)
**Recommendation:** Start with Option A, test Option B if performance insufficient

### Threshold Tuning
**Conservative (Start Here):**
- Moderate: 85th percentile
- High: 93rd percentile  
- Critical: 97th percentile
- Lower risk of false alarms

**Aggressive (After Validation):**
- Moderate: 75th percentile
- High: 90th percentile
- Critical: 95th percentile
- Higher recall, more false alarms

### Override Philosophy
**Safe Mode:** Spike gate flags high anomaly but doesn't override (trader decides)
**Moderate Mode:** Override only in Elevated/Crisis regimes
**Aggressive Mode:** Full override per specification (test thoroughly first)

Start in Safe Mode. Graduate to Moderate after validation. Aggressive mode requires months of proof.

---

## DEBUGGING CHECKLIST (When Things Break)

**Anomaly Scores All 0.0 or 1.0:**
- Check normalization logic (should use training distribution percentiles)
- Verify features aren't all NaN
- Confirm IsolationForest trained properly

**Spike Gate Never Activates:**
- Check anomaly_score column exists in prediction input
- Verify threshold logic (should be 0.85-0.95, not 85-95)
- Print anomaly scores to console for debugging

**Dual Model Performance Degrades:**
- Disable spike gate immediately
- Check if anomaly feature has lookahead bias
- Verify temporal safety on new velocity features
- Retrain dual model without anomaly feature

**Database Commit Errors:**
- Check schema extensions applied correctly
- Verify no column name typos
- Use database.commit() after stores
- Check for duplicate key violations

**Feature Quality Failures:**
- Missing VX futures data (check VXFuturesEngineer)
- VVIX sparsity (handle gracefully, don't crash)
- Publication lag violations (use TemporalSafetyValidator)

---

## CODE SNIPPETS (Not Full Implementation, Just Direction)

**Adding Features to FeatureEngineer:**
```python
# In feature_engineer.py, add new method
def add_spike_features(self, df):
    """Velocity/acceleration features for spike detection"""
    # VIX acceleration
    df['vix_accel_5d'] = df['vix_velocity_5d'].diff(5)
    df['vix_accel_21d'] = df['vix_velocity_21d'].diff(21)
    
    # VVIX velocity (if VVIX exists)
    if 'vvix' in df.columns:
        df['vviv_velocity_5d'] = df['vvix'].pct_change(5)
        df['vvix_accel_5d'] = df['vviv_velocity_5d'].diff(5)
    
    # Term structure collapse
    if 'VX1-VX2' in df.columns:
        df['vx_contango_velocity'] = df['VX1-VX2'].diff(5)
        df['vx_contango_accel'] = df['vx_contango_velocity'].diff(5)
    
    return df

# In build_complete_features(), call this method
spike_features = self.add_spike_features(combined_df)
all_features = pd.concat([all_features, spike_features], axis=1)
```

**Spike Gate in SimplifiedVIXForecaster:**
```python
def predict(self, X, current_vix):
    # Existing prediction logic...
    base_forecast = {
        'magnitude_pct': magnitude_pct,
        'direction': direction,
        'direction_confidence': ensemble_confidence,
        # ... rest of forecast
    }
    
    # NEW: Check for anomaly score
    if 'anomaly_score_prior_day' in X.columns:
        anomaly_score = float(X['anomaly_score_prior_day'].iloc[0])
        
        if anomaly_score > 0.90 and direction == 'DOWN':
            # High anomaly + model says DOWN = danger
            if anomaly_score > 0.95:
                # Extreme: override to UP
                return {
                    **base_forecast,
                    'direction': 'UP',
                    'magnitude_pct': abs(magnitude_pct) * 1.3,
                    'direction_confidence': 0.65,
                    'spike_gate_triggered': True,
                    'spike_gate_action': 'OVERRIDE_TO_UP'
                }
            else:
                # Moderate: override to NEUTRAL
                return {
                    **base_forecast,
                    'direction': 'NEUTRAL',
                    'magnitude_pct': 0.0,
                    'direction_confidence': 0.50,
                    'spike_gate_triggered': True,
                    'spike_gate_action': 'OVERRIDE_TO_NEUTRAL'
                }
        
        elif anomaly_score > 0.85 and direction == 'UP':
            # Anomaly confirms UP prediction
            return {
                **base_forecast,
                'magnitude_pct': magnitude_pct * 1.2,
                'direction_confidence': min(ensemble_confidence * 1.2, 0.95),
                'spike_gate_triggered': True,
                'spike_gate_action': 'BOOST_CONFIRM'
            }
    
    # No spike gate activation
    return {**base_forecast, 'spike_gate_triggered': False}
```

**EOD Score Calculation:**
```python
def calculate_eod_anomaly_score(intraday_scores, eod_feature_state):
    """Aggregate intraday signals into daily score"""
    if len(intraday_scores) == 0:
        # No intraday data, use EOD features only
        return eod_feature_state['anomaly_score']
    
    # Weighted aggregation
    max_score = max(intraday_scores)  # Peak anomaly
    mean_score = np.mean(intraday_scores)  # Average
    persistence = len([s for s in intraday_scores if s > 0.75]) / len(intraday_scores)
    
    eod_score = (
        0.40 * max_score +
        0.30 * mean_score +
        0.20 * persistence +
        0.10 * eod_feature_state['anomaly_score']
    )
    
    # Normalize and smooth
    eod_score = np.clip(eod_score, 0.0, 0.95)
    return eod_score
```

---

## FINAL THOUGHTS

**You're Not Starting From Scratch:**
The user has built professional infrastructure. Your job is integration, not invention.

**Test Before Deploy:**
You have 20 years of VIX history. Use it. Backtest thoroughly. Paper trade for 3 months minimum.

**Fail Gracefully:**
If anomaly system breaks, dual model keeps working. Always provide escape hatch.

**Log Everything:**
You can't improve what you don't measure. Every prediction, every override, every alert → database.

**Be Conservative Initially:**
Better to miss spikes during testing than cause false alarms. Start with high thresholds (95th percentile). Lower gradually.

**Remember the Goal:**
Catch VIX spikes the dual model misses. Target: 50%+ catch rate (vs 0-7% baseline). Don't break the dual model's 70% accuracy on normal days.

---

**When in doubt, read the full specification: `ANOMALY_SYSTEM_SPECIFICATION.md`**

**When confused about infrastructure, study these files:**
- `feature_engineer.py`: How to add features
- `xgboost_trainer_v3.py`: How dual model works
- `prediction_database.py`: How to store data
- `integrated_system.py`: How everything connects

**When stuck, ask:**
"Am I extending existing infrastructure or rebuilding it?"
If rebuilding, you're probably doing it wrong.

---

Good luck. What is dead may never die, but rises again, harder and stronger.

**END QUICK START GUIDE**
