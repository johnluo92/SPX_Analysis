# UPGRADE GUIDE: integrated_system_production.py
## System Orchestration: Integrating Probabilistic Forecaster & Prediction Database

---

## SYSTEM CONTEXT

### What This File Does
**Orchestration hub** - Coordinates all components (feature engine, forecaster, anomaly detector, database) to generate and store probabilistic forecasts.

**Current Flow:**
```
User → IntegratedSystem.run() → Feature Engine → Binary Classifier → JSON output
```

**New Flow:**
```
User → IntegratedSystem.generate_forecast() → Feature Engine → Cohort Classifier →
  → Probabilistic Forecaster → Distribution Object → Prediction Database → 
  → Dashboard/API → Backtest Engine
```

---

## FILE ROLE: integrated_system_production.py

**Current Purpose:**
- Coordinates feature generation
- Runs anomaly detection
- Makes binary VIX expansion predictions
- Exports to JSON for dashboards

**What's Changing:**
- ADD: Prediction database integration
- ADD: Cohort-aware forecasting
- ADD: Distribution object handling
- ADD: Quality checking before predictions
- MODIFY: Main prediction method signature
- KEEP: Anomaly detector (runs in parallel)

---

## REQUIRED CONTEXT FROM OTHER FILES

### Imports Needed
```python
from config import (
    TARGET_CONFIG,
    CALENDAR_COHORTS,
    PREDICTION_DB_CONFIG,
    FEATURE_QUALITY_CONFIG
)
from feature_engine import FeatureEngineV5
from xgboost_trainer_v2 import ProbabilisticVIXForecaster
from temporal_validator import TemporalValidator
from prediction_database import PredictionDatabase  # NEW FILE
from anomaly_detector import AnomalyDetector  # EXISTING
```

### What You'll Receive from Other Files
```python
# From feature_engine.build_features():
df = pd.DataFrame({
    # 232 features
    'vix': [18.5, 19.2, ...],
    'spx': [4800, 4820, ...],
    # ... 230 more features
    
    # Metadata
    'calendar_cohort': ['mid_cycle', 'opex_minus_5', ...],
    'cohort_weight': [1.0, 1.2, ...],
    'feature_quality': [0.97, 0.85, ...]
})

# From forecaster.predict():
distribution = {
    'point_estimate': 8.5,
    'quantiles': {'q10': -2.3, 'q25': 3.1, 'q50': 7.9, 'q75': 14.2, 'q90': 23.8},
    'regime_probabilities': {'low': 0.02, 'normal': 0.38, 'elevated': 0.52, 'crisis': 0.08},
    'confidence_score': 0.87,
    'cohort': 'mid_cycle'
}
```

---

## DETAILED CHANGES

### CHANGE 1: Update Class Initialization

**FIND:**
```python
class IntegratedSystem:
    def __init__(self):
        self.feature_engine = FeatureEngineV5()
        self.anomaly_detector = AnomalyDetector()
        # ... old components
```

**REPLACE WITH:**
```python
class IntegratedSystem:
    """
    Orchestrates probabilistic VIX forecasting system.
    
    Components:
      - Feature Engine: Generate 232 features with calendar cohorts
      - Probabilistic Forecaster: Multi-output distribution model
      - Prediction Database: Store forecasts for backtesting
      - Anomaly Detector: Identify market regime anomalies (parallel)
      - Temporal Validator: Check data quality
    
    Example:
        >>> system = IntegratedSystem()
        >>> distribution = system.generate_forecast()
        >>> print(distribution['point_estimate'])  # 8.5% expected VIX change
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize integrated system.
        
        Args:
            models_dir: Directory containing trained models
        """
        logger.info("=" * 80)
        logger.info("INTEGRATED PROBABILISTIC FORECASTING SYSTEM")
        logger.info("=" * 80)
        
        # Core components
        self.feature_engine = FeatureEngineV5()
        self.forecaster = ProbabilisticVIXForecaster()
        self.validator = TemporalValidator()
        self.prediction_db = PredictionDatabase()
        
        # Anomaly detector (runs independently)
        self.anomaly_detector = AnomalyDetector()
        
        # Load trained models
        self.models_dir = Path(models_dir)
        self._load_models()
        
        # State tracking
        self.last_forecast = None
        self.forecast_history = []
        
        logger.info("✅ System initialized")
    
    
    def _load_models(self):
        """Load all trained cohort models."""
        logger.info("📂 Loading trained models...")
        
        model_files = list(self.models_dir.glob('probabilistic_forecaster_*.pkl'))
        
        if len(model_files) == 0:
            logger.warning("⚠️  No trained models found. Run training first.")
            return
        
        for model_file in model_files:
            cohort = model_file.stem.replace('probabilistic_forecaster_', '')
            self.forecaster.load(cohort, self.models_dir)
            logger.info(f"   ✅ Loaded: {cohort}")
        
        logger.info(f"📊 Total cohorts loaded: {len(self.forecaster.models)}")
```

---

### CHANGE 2: New Main Forecasting Method

**FIND:**
```python
def run(self):
    """Run system and return predictions."""
    # ... old binary prediction logic
```

**REPLACE WITH:**
```python
def generate_forecast(self, date=None, store_prediction=True):
    """
    Generate probabilistic VIX forecast for given date.
    
    Args:
        date: Date to forecast from (defaults to latest)
        store_prediction: If True, save to prediction database
        
    Returns:
        dict: Distribution object with keys:
            - point_estimate: Expected VIX % change
            - quantiles: dict of percentiles
            - regime_probabilities: dict of regime probs
            - confidence_score: Forecast quality [0, 1]
            - cohort: Which calendar cohort used
            - forecast_date: Target date (date + horizon)
            - feature_quality: Data quality score
            
    Raises:
        ValueError: If data quality too poor to forecast
        
    Example:
        >>> distribution = system.generate_forecast()
        >>> if distribution['confidence_score'] > 0.8:
        >>>     print(f"High confidence: VIX expected {distribution['point_estimate']:.1f}%")
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PROBABILISTIC FORECAST")
    logger.info("=" * 80)
    
    # 1. Build features
    logger.info("🔧 Building features...")
    df = self.feature_engine.build_features(window='15y')
    
    # 2. Select observation date
    if date is None:
        date = df.index[-1]
        logger.info(f"📅 Using latest date: {date.strftime('%Y-%m-%d')}")
    else:
        date = pd.Timestamp(date)
        logger.info(f"📅 Forecast date: {date.strftime('%Y-%m-%d')}")
    
    if date not in df.index:
        raise ValueError(f"Date {date} not in feature data")
    
    observation = df.loc[date]
    
    # 3. Check data quality
    logger.info("🔍 Checking data quality...")
    feature_dict = observation.to_dict()
    quality_score = self.validator.compute_feature_quality(feature_dict, date)
    usable, quality_msg = self.validator.check_quality_threshold(quality_score)
    
    logger.info(f"   Quality Score: {quality_score:.2f}")
    logger.info(f"   Status: {quality_msg}")
    
    if not usable:
        # Generate quality report for diagnostics
        report = self.validator.get_quality_report(feature_dict, date)
        logger.error("❌ Data quality insufficient:")
        for issue in report['issues']:
            logger.error(f"   • {issue}")
        raise ValueError(f"Cannot forecast: {quality_msg}")
    
    # 4. Get calendar cohort
    cohort = observation['calendar_cohort']
    cohort_weight = observation['cohort_weight']
    logger.info(f"📅 Calendar Cohort: {cohort} (weight: {cohort_weight:.2f})")
    
    # 5. Check if cohort model exists
    if cohort not in self.forecaster.models:
        logger.warning(f"⚠️  Cohort {cohort} not trained, falling back to mid_cycle")
        cohort = 'mid_cycle'
        
        if cohort not in self.forecaster.models:
            raise ValueError("No trained models available. Run training first.")
    
    # 6. Prepare features for prediction
    X = observation[self.forecaster.feature_names].values.reshape(1, -1)
    X_df = pd.DataFrame(X, columns=self.forecaster.feature_names)
    
    # 7. Generate distribution
    logger.info("🎯 Generating probabilistic forecast...")
    distribution = self.forecaster.predict(X_df, cohort)
    
    # Adjust confidence by cohort weight
    distribution['confidence_score'] *= (2 - cohort_weight)  # Higher weight = lower confidence
    distribution['confidence_score'] = np.clip(distribution['confidence_score'], 0, 1)
    
    # 8. Add metadata
    forecast_date = date + pd.Timedelta(days=TARGET_CONFIG['horizon_days'])
    distribution['metadata'] = {
        'observation_date': date.strftime('%Y-%m-%d'),
        'forecast_date': forecast_date.strftime('%Y-%m-%d'),
        'horizon_days': TARGET_CONFIG['horizon_days'],
        'feature_quality': float(quality_score),
        'cohort_weight': float(cohort_weight),
        'current_vix': float(observation['vix']),
        'features_used': len(self.forecaster.feature_names)
    }
    
    # 9. Log forecast summary
    self._log_forecast_summary(distribution)
    
    # 10. Store in database
    if store_prediction:
        prediction_id = self._store_prediction(distribution, observation)
        distribution['prediction_id'] = prediction_id
        logger.info(f"💾 Stored prediction: {prediction_id}")
    
    # 11. Update state
    self.last_forecast = distribution
    self.forecast_history.append({
        'date': date,
        'distribution': distribution
    })
    
    logger.info("=" * 80)
    logger.info("✅ FORECAST COMPLETE")
    logger.info("=" * 80)
    
    return distribution


def _log_forecast_summary(self, distribution):
    """Log human-readable forecast summary."""
    logger.info("\n📊 FORECAST SUMMARY")
    logger.info("─" * 60)
    
    # Point estimate
    point = distribution['point_estimate']
    logger.info(f"Point Estimate:     {point:+.1f}%")
    
    # Quantiles
    quantiles = distribution['quantiles']
    logger.info(f"Distribution:")
    logger.info(f"   10th percentile: {quantiles['q10']:+.1f}%")
    logger.info(f"   25th percentile: {quantiles['q25']:+.1f}%")
    logger.info(f"   Median (50th):   {quantiles['q50']:+.1f}%")
    logger.info(f"   75th percentile: {quantiles['q75']:+.1f}%")
    logger.info(f"   90th percentile: {quantiles['q90']:+.1f}%")
    
    # Regimes
    regimes = distribution['regime_probabilities']
    logger.info(f"Regime Probabilities:")
    for regime, prob in regimes.items():
        logger.info(f"   {regime.capitalize():10s}: {prob*100:5.1f}%")
    
    # Confidence
    conf = distribution['confidence_score']
    logger.info(f"Confidence Score:   {conf:.2f}")
    
    # Interpretation
    current_vix = distribution['metadata']['current_vix']
    expected_vix = current_vix * (1 + point/100)
    logger.info(f"\nInterpretation:")
    logger.info(f"   Current VIX: {current_vix:.2f}")
    logger.info(f"   Expected VIX in {TARGET_CONFIG['horizon_days']} days: {expected_vix:.2f}")
    logger.info(f"   90% confidence range: [{current_vix*(1+quantiles['q10']/100):.2f}, {current_vix*(1+quantiles['q90']/100):.2f}]")
```

---

### CHANGE 3: Add Prediction Storage Method

**LOCATION:** New method in class

```python
def _store_prediction(self, distribution, observation):
    """
    Store prediction in database for backtesting.
    
    Args:
        distribution: Forecast distribution object
        observation: Original feature row
        
    Returns:
        str: prediction_id (UUID)
    """
    import uuid
    
    prediction_id = str(uuid.uuid4())
    
    # Extract features used (for provenance)
    features_used = {
        feat: float(observation[feat]) 
        for feat in self.forecaster.feature_names
    }
    
    # Build database record
    record = {
        'prediction_id': prediction_id,
        'timestamp': pd.Timestamp.now(),
        'forecast_date': pd.Timestamp(distribution['metadata']['forecast_date']),
        'horizon': TARGET_CONFIG['horizon_days'],
        
        # Context
        'calendar_cohort': distribution['cohort'],
        'cohort_weight': distribution['metadata']['cohort_weight'],
        
        # Predictions
        'point_estimate': distribution['point_estimate'],
        'q10': distribution['quantiles']['q10'],
        'q25': distribution['quantiles']['q25'],
        'q50': distribution['quantiles']['q50'],
        'q75': distribution['quantiles']['q75'],
        'q90': distribution['quantiles']['q90'],
        'prob_low': distribution['regime_probabilities']['low'],
        'prob_normal': distribution['regime_probabilities']['normal'],
        'prob_elevated': distribution['regime_probabilities']['elevated'],
        'prob_crisis': distribution['regime_probabilities']['crisis'],
        'confidence_score': distribution['confidence_score'],
        
        # Metadata
        'feature_quality': distribution['metadata']['feature_quality'],
        'num_features_used': distribution['metadata']['features_used'],
        'current_vix': distribution['metadata']['current_vix'],
        
        # Provenance
        'features_used': json.dumps(features_used),
        'model_version': self._get_model_version()
    }
    
    # Store in database
    self.prediction_db.store_prediction(record)
    
    return prediction_id


def _get_model_version(self):
    """Get current model version (git hash or timestamp)."""
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        return f"git-{git_hash}"
    except:
        return f"v{pd.Timestamp.now().strftime('%Y%m%d')}"
```

---

### CHANGE 4: Add Batch Forecasting Method

**LOCATION:** New method for backtesting

```python
def generate_forecast_batch(self, start_date, end_date, frequency='daily'):
    """
    Generate forecasts for multiple dates (backtesting).
    
    Args:
        start_date: Start of forecast period
        end_date: End of forecast period
        frequency: 'daily' or 'weekly'
        
    Returns:
        list: List of distribution objects
        
    Example:
        >>> forecasts = system.generate_forecast_batch('2024-01-01', '2024-12-31')
        >>> print(f"Generated {len(forecasts)} forecasts")
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    logger.info(f"📅 Batch forecasting: {start_date.date()} to {end_date.date()}")
    
    # Get all business days in range
    date_range = pd.bdate_range(start_date, end_date)
    
    if frequency == 'weekly':
        date_range = date_range[date_range.weekday == 4]  # Fridays only
    
    logger.info(f"📊 Generating {len(date_range)} forecasts...")
    
    forecasts = []
    for date in date_range:
        try:
            distribution = self.generate_forecast(date, store_prediction=True)
            forecasts.append(distribution)
        except ValueError as e:
            logger.warning(f"⚠️  Skipped {date.date()}: {e}")
            continue
    
    logger.info(f"✅ Batch complete: {len(forecasts)}/{len(date_range)} successful")
    
    return forecasts
```

---

### CHANGE 5: Update Run Method (Backward Compatibility)

**LOCATION:** Keep old method for backward compatibility

```python
def run(self, date=None):
    """
    Legacy method - redirects to generate_forecast().
    
    Kept for backward compatibility with existing scripts.
    """
    logger.warning("⚠️  run() is deprecated, use generate_forecast()")
    return self.generate_forecast(date)
```

---

## INTEGRATION POINTS

### 1. Called By: Main Scripts
```python
# In main.py or dashboard_orchestrator.py
from integrated_system_production import IntegratedSystem

system = IntegratedSystem(models_dir='models')

# Real-time forecasting
distribution = system.generate_forecast()
print(f"Expected VIX change: {distribution['point_estimate']:.1f}%")

# Backtesting
forecasts = system.generate_forecast_batch('2024-01-01', '2024-12-31')
```

### 2. Uses From: All Components
- `feature_engine.py`: build_features()
- `xgboost_trainer_v2.py`: predict()
- `temporal_validator.py`: compute_feature_quality()
- `prediction_database.py`: store_prediction()

### 3. Provides To: Dashboard/API
```python
# In dashboard.py
distribution = system.generate_forecast()

# Extract for visualization
point = distribution['point_estimate']
quantiles = distribution['quantiles']
regimes = distribution['regime_probabilities']
confidence = distribution['confidence_score']
```

---

## TESTING

### Integration Test
```python
def test_integrated_system():
    """Test full system integration."""
    system = IntegratedSystem(models_dir='models')
    
    # Test single forecast
    distribution = system.generate_forecast()
    
    # Validate structure
    assert 'point_estimate' in distribution
    assert 'quantiles' in distribution
    assert 'regime_probabilities' in distribution
    assert 'confidence_score' in distribution
    assert 'metadata' in distribution
    
    # Validate quantile ordering
    q = distribution['quantiles']
    assert q['q10'] <= q['q25'] <= q['q50'] <= q['q75'] <= q['q90']
    
    # Validate regime probabilities sum to 1
    probs = list(distribution['regime_probabilities'].values())
    assert abs(sum(probs) - 1.0) < 0.01
    
    # Validate confidence range
    assert 0 <= distribution['confidence_score'] <= 1
    
    print("✅ Integration test passed")

test_integrated_system()
```

---

## COMMON PITFALLS

### 1. Not Checking Data Quality
```python
# WRONG: Generate forecast blindly
distribution = system.generate_forecast()  # Might fail silently

# CORRECT: Handle quality issues
try:
    distribution = system.generate_forecast()
except ValueError as e:
    logger.error(f"Cannot forecast: {e}")
    # Fallback to previous forecast or skip
```

### 2. Not Storing Predictions
```python
# WRONG: Generate but don't store
distribution = system.generate_forecast(store_prediction=False)
# Can't backtest without stored predictions!

# CORRECT: Always store for backtesting
distribution = system.generate_forecast(store_prediction=True)
```

### 3. Missing Cohort Models
```python
# WRONG: Assume all cohorts trained
distribution = forecaster.predict(X, cohort='fomc_week')  # KeyError if not trained

# CORRECT: Check availability
if cohort not in forecaster.models:
    cohort = 'mid_cycle'  # Fallback
```

---

## SUMMARY

**New Methods:**
- `generate_forecast()` - Main forecasting method (replaces run())
- `_store_prediction()` - Database integration
- `generate_forecast_batch()` - Batch forecasting
- `_log_forecast_summary()` - Human-readable output
- `_get_model_version()` - Version tracking

**Modified Methods:**
- `__init__()` - Add new components
- `run()` - Deprecation wrapper

**New Dependencies:**
- `PredictionDatabase` (new file)
- `ProbabilisticVIXForecaster` (rewritten file)

**Lines Changed:**
- ~300 new lines
- ~50 modified lines
- ~100 removed lines (old binary logic)

**Next Steps:**
1. Create prediction_database.py (new file)
2. Train probabilistic models
3. Test single forecast
4. Run batch backtest
5. Validate quantile coverage
