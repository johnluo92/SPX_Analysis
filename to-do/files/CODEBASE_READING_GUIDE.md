# CODEBASE READING GUIDE FOR ANOMALY SYSTEM IMPLEMENTATION

**Recommended reading order to understand the VIX forecasting infrastructure**

---

## START HERE (These 2 Documents)
1. **ANOMALY_QUICK_START.md** - 10 minute read, gives you the mission
2. **ANOMALY_SYSTEM_SPECIFICATION.md** - 30 minute read, comprehensive requirements

---

## PHASE 1: UNDERSTAND THE DATA PIPELINE (Read in Order)

### 1. Core Data Fetching
**File:** `src/core/data_fetcher.py` (250 lines)  
**Focus:** Lines 1-100 (UnifiedDataFetcher class initialization)  
**Key Concepts:**
- How FRED data is fetched with incremental caching
- Yahoo Finance data retrieval with business day logic
- CBOE data loading from local CSV files
- Publication lag handling and forward-fill limits
- How data is cached in parquet format

**Questions to Answer:**
- How does the system handle missing data?
- What are publication lags and why do they matter?
- How does incremental fetching work? (Saves time on retrain)

---

### 2. Feature Engineering Pipeline
**File:** `src/core/feature_engineer.py` (2000+ lines)  
**Read in Chunks:**

**Chunk 1 (Lines 1-200): Initialization & Calendar System**
- How FeatureEngineer is initialized with data_fetcher
- FOMC, OpEx, and VIX futures expiry calendar generation
- Cohort assignment logic (fomc_period, opex_week, earnings_heavy, mid_cycle)

**Chunk 2 (Lines 200-500): Base Features**
- VIX features (velocity, acceleration, percentiles, z-scores, regimes)
- SPX features (returns, realized vol, gaps, technical indicators)
- How publication lags are applied via `_apply_publication_lags()`

**Chunk 3 (Lines 500-1000): Domain-Specific Feature Engines**
- VXFuturesEngineer integration (term structure, roll yield, positioning)
- Credit spread features (HY OAS, IG-HY spreads)
- Treasury yield features (curve spreads, curvature, volatility)
- Commodity futures (crude oil, dollar index)

**Chunk 4 (Lines 1000-1500): Meta & Interaction Features**
- Regime indicators (volatility, trend, liquidity, correlation)
- Cross-asset relationships (equity-vol divergence, risk premium)
- Interaction features (VIX velocity × SPX realized vol)
- Regime-conditional features

**Chunk 5 (Lines 1500-end): System Integration**
- `build_complete_features()` orchestration
- Temporal safety validation
- Quality control and cohort assignment
- How all features are combined into final DataFrame

**Key Method to Understand:**
```python
def build_complete_features(self, years, end_date, force_historical)
```
This is the entry point. Follow its logic step by step.

**Questions to Answer:**
- How does the system prevent lookahead bias?
- Where would you add new velocity/acceleration features?
- How does calendar cohort assignment work?

---

### 3. Temporal Safety & Validation
**File:** `src/core/temporal_validator.py` (likely 100-200 lines)  
**Focus:** How the system ensures no future data leaks into past predictions  
**Key Concepts:**
- Publication lag enforcement
- Data quality scoring
- Feature staleness tracking

**Questions to Answer:**
- How do you validate a new feature for temporal safety?
- What happens if a feature is stale?

---

## PHASE 2: UNDERSTAND THE MODELS (Read in Order)

### 4. Target Calculation
**File:** `src/core/target_calculator.py` (likely 100-200 lines)  
**Focus:** How target variables are calculated  
**Key Concepts:**
- `target_log_vix_change`: Log(future_vix / current_vix)
- `target_direction`: 1 if UP, 0 if DOWN
- Horizon handling (5-day forward look)
- Why log-space vs percentage-space

**Questions to Answer:**
- Why use log returns instead of percentage returns?
- How are future dates calculated (business days)?

---

### 5. Regime Classification
**File:** `src/core/regime_classifier.py` (300 lines, **ALREADY PROVIDED**)  
**Focus:** How VIX is classified into 4 regimes  
**Key Concepts:**
- Regime boundaries: [0, 15.57, 23.36, 31.16, 100]
- Regime names: Low Vol, Normal, Elevated, Crisis
- Transition probabilities (e.g., Low Vol → Normal = 12.22%)
- Mean reversion statistics per regime
- Persistence metrics (median duration in each regime)

**Questions to Answer:**
- Which regime has highest spike probability? (Answer: Low Vol at 7%)
- What's the expected 5-day return in Crisis regime? (Answer: -6.49%)
- How long does each regime typically last?

---

### 6. Feature Selection
**File:** `src/core/xgboost_feature_selector_v2.py` (likely 300-500 lines)  
**Focus:** How 331 features are reduced to 81-99 for each model  
**Key Concepts:**
- Walk-forward cross-validation on train+val only (test excluded)
- Feature importance via XGBoost gain
- Correlation filtering (remove redundant features)
- Protected features (always included: is_fomc_period, is_opex_week, is_earnings_heavy)
- Separate selection for magnitude vs direction models

**Questions to Answer:**
- Why separate feature selection for magnitude vs direction?
- How does correlation filtering work?
- What are the top 20 features historically?

---

### 7. Dual XGBoost Model Training
**File:** `src/core/xgboost_trainer_v3.py` (500 lines, **ALREADY PROVIDED**)  
**Focus:** Magnitude regressor + direction classifier architecture  
**Key Concepts:**
- Train/Val/Test split (2021-12-31 / 2023-12-31 / 2024-01-01+)
- Quality filtering before training
- Cohort weighting (FOMC = 1.27x, OpEx = 1.40x)
- Magnitude model: Log-space prediction with clipping [-2, 2]
- Direction model: Binary classification with isotonic calibration
- Ensemble scoring: Weighted combination with agreement bonuses

**Key Method to Understand:**
```python
def predict(self, X, current_vix)
```
This is where anomaly score will integrate.

**Questions to Answer:**
- Why clip magnitude predictions at [-2, 2]?
- How does ensemble confidence work?
- What's the difference between raw direction probability and calibrated?

---

### 8. Forecast Calibration
**File:** `src/core/forecast_calibrator.py` (200 lines, **ALREADY PROVIDED**)  
**Focus:** Post-hoc bias correction using historical errors  
**Key Concepts:**
- 252-day rolling window with exponential decay
- Corrections fitted per regime+cohort, regime, cohort, and global
- Weighted average of errors (recent data weighted more)
- Adjusts magnitude forecast, not direction

**Questions to Answer:**
- Why use exponential decay instead of equal weights?
- What happens if no correction exists for a regime+cohort?
- How often should calibrator be refitted?

---

## PHASE 3: UNDERSTAND DATA STORAGE (Read in Order)

### 9. Prediction Database
**File:** `src/core/prediction_database.py` (300 lines, **ALREADY PROVIDED**)  
**Focus:** SQLite backend for storing forecasts and actuals  
**Key Concepts:**
- Schema with 25+ columns (prediction_id, forecast_date, magnitude_forecast, etc.)
- Commit tracking (ensures data isn't lost)
- Automatic backfilling of actuals from Yahoo Finance
- Performance summary calculation
- Duplicate prevention

**Key Method to Understand:**
```python
def store_prediction(self, record)
def get_predictions(self, with_actuals=False)
def backfill_actuals(self, fetcher)
```

**Questions to Answer:**
- How do you extend the schema to add new columns?
- What's the commit protocol? (Critical: must call .commit() after stores)
- How does backfilling work?

---

## PHASE 4: UNDERSTAND SYSTEM ORCHESTRATION (Read Last)

### 10. Integrated System
**File:** `src/integrated_system.py` (400 lines, **ALREADY PROVIDED**)  
**Focus:** End-to-end workflow from data fetch to forecast generation  
**Key Concepts:**
- Model staleness checking (retrains if model too old)
- Bootstrap calibration (backfills historical predictions)
- Feature caching (saves time on repeat predictions)
- Forecast generation workflow
- Enhanced forecast display

**Key Method to Understand:**
```python
def generate_forecast(self, date, df=None)
```
This is the production prediction pipeline.

**Questions to Answer:**
- When does the system retrain models?
- How does calibration bootstrap work?
- Where would you inject anomaly score into the prediction?

---

### 11. Training Script
**File:** `src/train_probabilistic_models.py` (likely 200-400 lines)  
**Focus:** Model training orchestration  
**Key Concepts:**
- Feature building with temporal safety
- Feature selection on train+val (test excluded)
- Model training with quality filtering
- Saving models and metadata

---

### 12. Configuration
**File:** `src/config.py` (500 lines, **ALREADY PROVIDED**)  
**Focus:** All system configuration constants  
**Read Sections:**
- `TARGET_CONFIG`: Horizon, log-space settings
- `CALENDAR_COHORTS`: Cohort definitions and weights
- `FEATURE_SELECTION_CONFIG`: Top N features for each model
- `MAGNITUDE_PARAMS` & `DIRECTION_PARAMS`: XGBoost hyperparameters (Tuner 1 values)
- `ENSEMBLE_CONFIG`: Confidence weighting
- `PUBLICATION_LAGS`: How many days each data source lags

**Questions to Answer:**
- Where would you add ANOMALY_DETECTION_CONFIG?
- What are the current cohort weights?
- What are the tuned XGBoost parameters?

---

## PHASE 5: UNDERSTAND THE OLD ANOMALY DETECTOR

### 13. Shelved Anomaly System
**File:** `anomaly_detector.py` (300 lines, **ALREADY PROVIDED**)  
**Focus:** What exists, what to keep, what to rewrite  
**Keep:**
- Quality validation logic (lines 21-46)
- Robust anomaly score calculation (lines 47-54)
- Coverage penalty system (lines 55-59)
- Feature importance calculation (lines 76-120)
- Statistical threshold calculation (lines 121-127)

**Rewrite:**
- Feature group definitions (hard-coded, should use production features)
- Training methodology (lines 128-185, needs production integration)
- Detection logic (lines 186-223, needs EOD score calculation)
- Persistence tracking (lines 231-254, keep but adapt)

**Questions to Answer:**
- What does `calculate_robust_anomaly_score()` do?
- How does coverage-weighted ensemble work?
- What's SHAP feature importance vs permutation?

---

## REFERENCE FILES (Consult When Needed)

### VX Futures Engineering
**Files:**
- `src/core/vx_continuous_contract_builder.py`
- `src/core/vx_futures_engineer.py`

**When to Read:** When you need to understand VX1-VX2 spread calculation, roll yield, or term structure features

---

### Supporting Utilities
**Files:**
- `src/core/calculations.py` (**ALREADY PROVIDED**) - Z-scores, percentiles, regime classification helpers
- `src/core/feature_correlation_analyzer.py` - Feature correlation analysis

**When to Read:** When implementing correlation filtering or feature validation

---

## READING STRATEGY FOR IMPLEMENTATION

### Day 1: System Understanding (4-6 hours)
Read in order:
1. Quick Start Guide (10 min)
2. Full Specification (30 min)
3. data_fetcher.py - Skim, understand caching
4. feature_engineer.py - Deep read, Chunks 1-2
5. xgboost_trainer_v3.py - Deep read, focus on predict()
6. integrated_system.py - Deep read, understand workflow

**Goal:** Can you explain the end-to-end prediction process?

---

### Day 2: Data & Features Deep Dive (4-6 hours)
Read in order:
1. feature_engineer.py - Deep read, Chunks 3-5
2. temporal_validator.py - Full read
3. target_calculator.py - Full read
4. regime_classifier.py - Review provided document
5. config.py - Skim all sections, note key values

**Goal:** Can you add a new feature without breaking temporal safety?

---

### Day 3: Model Architecture Deep Dive (3-4 hours)
Read in order:
1. xgboost_feature_selector_v2.py - Full read
2. xgboost_trainer_v3.py - Re-read, focus on training flow
3. forecast_calibrator.py - Full read
4. prediction_database.py - Full read

**Goal:** Can you explain how magnitude and direction models work together?

---

### Day 4: Existing Anomaly System Study (2-3 hours)
Read in order:
1. anomaly_detector.py - Full read, take notes
2. Re-read specification Section 1 (Current State Assessment)
3. Compare old system to new requirements

**Goal:** Can you identify what to keep vs what to rewrite?

---

### Day 5: Implementation Planning (2-3 hours)
Create your implementation plan:
1. List new features to add (velocity/acceleration)
2. Sketch database schema extensions
3. Outline IsolationForest training logic
4. Design EOD score calculation
5. Plan spike gate integration points

**Goal:** Can you write a 1-page implementation plan without re-reading docs?

---

## CRITICAL QUESTIONS TO ANSWER BEFORE CODING

Before you write a single line of code, ensure you can answer:

**Data Pipeline:**
- [ ] How does publication lag handling prevent lookahead bias?
- [ ] Where in feature_engineer.py would you add vix_accel_5d?
- [ ] How does the system handle missing VVIX data?

**Model Architecture:**
- [ ] Why separate magnitude and direction models?
- [ ] How does ensemble confidence combine magnitude + direction + agreement?
- [ ] What happens when models disagree (one says UP, one says DOWN)?

**Prediction Storage:**
- [ ] How do you add a new column to the forecasts table?
- [ ] What's the prediction_id scheme?
- [ ] How does backfilling of actuals work?

**Anomaly Integration:**
- [ ] Where exactly in the prediction pipeline would anomaly score be calculated?
- [ ] How would you store EOD anomaly scores for historical dates?
- [ ] Where in SimplifiedVIXForecaster.predict() would spike gate logic go?

**Testing:**
- [ ] How would you test a new feature for temporal safety?
- [ ] How would you validate spike gate doesn't break normal predictions?
- [ ] How would you backtest the system on 2004-2025 data?

If you can't answer these, read more before coding.

---

## COMMON PITFALLS (Avoid These)

### Pitfall 1: Not Understanding Temporal Safety
**Symptom:** Features that use future data to predict the past  
**Prevention:** Read temporal_validator.py, understand publication lags  
**Test:** Use TemporalSafetyValidator on every new feature

### Pitfall 2: Breaking the Dual Model
**Symptom:** Model accuracy drops from 70% to 60% after adding anomaly features  
**Prevention:** Test with and without anomaly features, compare performance  
**Solution:** Disable anomaly features if they degrade performance

### Pitfall 3: Database Schema Errors
**Symptom:** SQLite errors, duplicate keys, commit failures  
**Prevention:** Read prediction_database.py carefully, understand schema  
**Test:** Test schema extensions on copy of database first

### Pitfall 4: Overfitting to 2024-2025
**Symptom:** Works on test data, fails on new data  
**Prevention:** Train on all 20 years, validate on 2024-2025 only  
**Test:** Walk-forward validation on multiple time periods

### Pitfall 5: Ignoring Spike Gate Overrides
**Symptom:** System overrides dual model too often, trades poorly  
**Prevention:** Start with conservative thresholds (95th percentile)  
**Test:** Log override rate, should be <5% of predictions initially

---

## FILE COMPLEXITY RATINGS

**Easy (< 1 hour to understand):**
- calculations.py - Simple mathematical functions
- regime_classifier.py - Clear regime logic
- config.py - Just constants and dictionaries

**Moderate (1-3 hours to understand):**
- target_calculator.py - Forward-looking target logic
- forecast_calibrator.py - Bias correction methodology
- prediction_database.py - SQLite operations
- anomaly_detector.py - Existing system to study

**Complex (3-6 hours to understand):**
- data_fetcher.py - Multi-source data retrieval
- xgboost_trainer_v3.py - Dual model training
- integrated_system.py - System orchestration

**Very Complex (6+ hours to fully understand):**
- feature_engineer.py - 2000+ lines, 331 features across 6 domains
- xgboost_feature_selector_v2.py - Cross-validation logic

**Recommendation:** Start with Easy, progress to Moderate, tackle Complex when confident, save Very Complex for after you've implemented basic anomaly system.

---

## WHEN YOU'RE STUCK

**If confused about data:**
→ Re-read data_fetcher.py and feature_engineer.py initialization

**If confused about models:**
→ Re-read xgboost_trainer_v3.py predict() method

**If confused about storage:**
→ Re-read prediction_database.py store_prediction() method

**If confused about integration:**
→ Re-read integrated_system.py generate_forecast() method

**If confused about requirements:**
→ Re-read ANOMALY_SYSTEM_SPECIFICATION.md (the comprehensive doc)

**If completely lost:**
→ Go back to ANOMALY_QUICK_START.md and start over

---

## FINAL CHECKLIST BEFORE IMPLEMENTATION

- [ ] Read Quick Start Guide
- [ ] Read Full Specification
- [ ] Understand data fetching (data_fetcher.py)
- [ ] Understand feature engineering (feature_engineer.py, at least Chunks 1-2)
- [ ] Understand dual model (xgboost_trainer_v3.py)
- [ ] Understand database (prediction_database.py)
- [ ] Understand system integration (integrated_system.py)
- [ ] Studied old anomaly detector (anomaly_detector.py)
- [ ] Can answer all Critical Questions
- [ ] Have written 1-page implementation plan
- [ ] Excited to build this system!

---

**Remember:** You're not building from scratch. You're integrating with professional infrastructure. Read first, code second.

**Estimated Total Reading Time:** 15-20 hours across 5 days  
**Estimated Implementation Time:** 3-4 weeks for full system

Good luck. The codebase is your ally, not your enemy. Trust the infrastructure.

**END READING GUIDE**
