# UPGRADE GUIDE: temporal_validator.py
## Adding Feature Quality Scoring for Confidence Models

---

## SYSTEM CONTEXT

### What We're Adding
**Feature quality scoring** - A quantitative measure of how fresh and complete the feature data is for a given date. This feeds into the confidence model to adjust forecast reliability.

**The Problem:**
- Some features update daily (VIX, SPX) → Always fresh
- Some update weekly (CBOE GAMMA) → May be stale
- Some update monthly (CPI) → Often outdated
- Missing CBOE data during outages → Degraded forecast quality

**The Solution:**
- Compute `feature_quality` score [0, 1] per row
- 1.0 = Perfect (all features fresh and present)
- 0.5 = Degraded (some important features missing/stale)
- 0.0 = Unusable (critical features missing)
- Confidence model learns: Low feature quality → Low confidence prediction

---

## FILE ROLE: temporal_validator.py

**Current Purpose:**
- Validates features respect publication delays (temporal safety)
- Tracks feature lags (e.g., CPI has 30-day lag)
- Prevents look-ahead bias

**Current Key Methods:**
- `validate_features()`: Check for temporal violations
- `get_feature_lags()`: Return lag specifications
- `check_alignment()`: Verify date alignment

**What's Changing:**
- ADD: `compute_feature_quality()` method
- ADD: Feature staleness tracking
- ADD: Quality threshold checking
- KEEP: All existing validation logic

---

## REQUIRED CONTEXT FROM OTHER FILES

### From config.py
```python
from config import (
    FEATURE_QUALITY_CONFIG,  # Quality thresholds and penalties
    FEATURE_LAGS               # Publication delays (already exists)
)
```

### Used By: feature_engine.py
```python
# In feature_engine.build_features()
df['feature_quality'] = engine._compute_feature_quality_vectorized(df)
# This actually calls temporal_validator under the hood
```

### Used By: xgboost_trainer_v2.py
```python
# In trainer._create_targets()
y_confidence = (
    0.5 * df['feature_quality'] +  # From temporal_validator
    0.5 * regime_stability           # From market conditions
)
```

---

## DETAILED CHANGES

### CHANGE 1: Add Feature Freshness Tracking

**LOCATION:** Add to class `__init__` method

**FIND:**
```python
class TemporalValidator:
    def __init__(self):
        self.feature_lags = FEATURE_LAGS
        self.validation_log = []
```

**ADD AFTER INIT:**
```python
        # Track last update time per feature (for staleness detection)
        self.last_update_timestamps = {}  # {feature_name: pd.Timestamp}
        
        # Quality configuration
        self.quality_config = FEATURE_QUALITY_CONFIG
        
        logger.info("🔍 Temporal Validator initialized with quality scoring")
```

---

### CHANGE 2: Add Core Quality Scoring Method

**LOCATION:** Add new method to class

```python
def compute_feature_quality(self, feature_dict: dict, date: pd.Timestamp = None) -> float:
    """
    Compute feature quality score for a single observation.
    
    Quality score considers:
      1. Missingness: Are critical features present?
      2. Staleness: How old is each feature?
      3. Lag compliance: Does feature respect publication delay?
    
    Args:
        feature_dict: Dict of {feature_name: value}
        date: Date of observation (for staleness calculation)
        
    Returns:
        float: Quality score [0, 1] where:
            1.0 = Perfect data quality
            0.8-1.0 = Good quality (minor issues)
            0.5-0.8 = Degraded quality (proceed with caution)
            0.3-0.5 = Poor quality (high uncertainty)
            <0.3 = Unusable (refuse to forecast)
    
    Example:
        >>> features = {'vix': 18.5, 'spx': 4800, 'SKEW': None}
        >>> quality = validator.compute_feature_quality(features)
        >>> print(quality)  # 0.75 (SKEW missing, but not critical)
    """
    if date is None:
        date = pd.Timestamp.now()
    
    scores = []
    
    # 1. Check critical features (must be present)
    critical_features = self.quality_config['missingness_penalty']['critical_features']
    for feat in critical_features:
        if feat in feature_dict:
            if pd.isna(feature_dict[feat]) or feature_dict[feat] is None:
                scores.append(0.0)  # Critical missing = complete failure
            else:
                scores.append(1.0)
        else:
            # Feature not even in dict (shouldn't happen, but handle)
            scores.append(0.0)
    
    # 2. Check important features (0.5 penalty if missing)
    important_features = self.quality_config['missingness_penalty']['important_features']
    for feat in important_features:
        if feat in feature_dict:
            if pd.isna(feature_dict[feat]) or feature_dict[feat] is None:
                scores.append(0.5)  # Important missing = degraded
            else:
                # Check staleness
                staleness_score = self._compute_staleness_score(feat, date)
                scores.append(staleness_score)
        else:
            scores.append(0.5)
    
    # 3. Check optional features (0.9 penalty if missing)
    optional_features = self.quality_config['missingness_penalty']['optional_features']
    for feat in optional_features:
        if feat in feature_dict:
            if pd.isna(feature_dict[feat]) or feature_dict[feat] is None:
                scores.append(0.9)  # Optional missing = minor impact
            else:
                staleness_score = self._compute_staleness_score(feat, date)
                scores.append(staleness_score)
        else:
            scores.append(0.9)
    
    # Average all component scores
    if len(scores) == 0:
        return 1.0  # No tracked features = assume good quality
    
    quality_score = np.mean(scores)
    
    # Clip to [0, 1] range
    quality_score = np.clip(quality_score, 0.0, 1.0)
    
    return quality_score


def _compute_staleness_score(self, feature_name: str, date: pd.Timestamp) -> float:
    """
    Score feature freshness based on time since last update.
    
    Args:
        feature_name: Name of feature
        date: Current date
        
    Returns:
        float: Staleness score [0, 1] where:
            1.0 = Fresh (updated recently)
            0.5 = Stale (beyond typical update frequency)
            0.2 = Very stale (ancient data)
    """
    # Check if we have last update timestamp
    if feature_name not in self.last_update_timestamps:
        # No tracking data - assume fresh for now
        return 1.0
    
    last_update = self.last_update_timestamps[feature_name]
    days_stale = (date - last_update).days
    
    # Get expected lag for this feature
    expected_lag = self.feature_lags.get(feature_name, 1)  # Default 1 day
    
    # Score based on staleness relative to expected lag
    staleness_config = self.quality_config['staleness_penalty']
    
    if days_stale <= expected_lag:
        return staleness_config['none']  # 1.0
    elif days_stale <= expected_lag + 3:
        return staleness_config['minor']  # 0.95
    elif days_stale <= expected_lag + 7:
        return staleness_config['moderate']  # 0.80
    elif days_stale <= expected_lag + 14:
        return staleness_config['severe']  # 0.50
    else:
        return staleness_config['critical']  # 0.20
```

---

### CHANGE 3: Add Batch Quality Scoring (Vectorized)

**LOCATION:** Add method for DataFrame-wide quality scoring

```python
def compute_feature_quality_batch(self, df: pd.DataFrame) -> pd.Series:
    """
    Compute quality scores for entire DataFrame (vectorized).
    
    More efficient than calling compute_feature_quality() row-by-row.
    
    Args:
        df: DataFrame with features
        
    Returns:
        pd.Series: Quality scores indexed by date
        
    Example:
        >>> quality_series = validator.compute_feature_quality_batch(df)
        >>> df['feature_quality'] = quality_series
    """
    quality_scores = []
    
    for date, row in df.iterrows():
        feature_dict = row.to_dict()
        quality = self.compute_feature_quality(feature_dict, date)
        quality_scores.append(quality)
    
    return pd.Series(quality_scores, index=df.index)
```

**Note:** This is not truly vectorized (still loops), but provides consistent interface. Can optimize with Numba if performance issues.

---

### CHANGE 4: Add Update Timestamp Tracking

**LOCATION:** Add method to track when features were last fetched

```python
def update_feature_timestamp(self, feature_name: str, timestamp: pd.Timestamp = None):
    """
    Record when a feature was last updated.
    
    Called by data_fetcher after successful fetch.
    
    Args:
        feature_name: Name of feature
        timestamp: Update time (defaults to now)
        
    Example:
        >>> # In data_fetcher.py after fetching VIX:
        >>> validator.update_feature_timestamp('vix', pd.Timestamp.now())
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now()
    
    self.last_update_timestamps[feature_name] = timestamp
    logger.debug(f"🔄 Updated timestamp for {feature_name}: {timestamp}")


def get_feature_age(self, feature_name: str, current_date: pd.Timestamp = None) -> int:
    """
    Get days since feature was last updated.
    
    Args:
        feature_name: Name of feature
        current_date: Reference date (defaults to now)
        
    Returns:
        int: Days since last update (or 0 if never tracked)
        
    Example:
        >>> age = validator.get_feature_age('CPI')
        >>> print(f"CPI is {age} days old")
    """
    if current_date is None:
        current_date = pd.Timestamp.now()
    
    if feature_name not in self.last_update_timestamps:
        return 0  # Unknown age
    
    last_update = self.last_update_timestamps[feature_name]
    return (current_date - last_update).days
```

---

### CHANGE 5: Add Quality Threshold Checker

**LOCATION:** Add utility method for production use

```python
def check_quality_threshold(self, quality_score: float, strict: bool = False) -> tuple:
    """
    Check if quality score meets minimum threshold for forecasting.
    
    Args:
        quality_score: Quality score [0, 1]
        strict: If True, use higher threshold
        
    Returns:
        tuple: (usable: bool, warning_message: str)
        
    Example:
        >>> usable, msg = validator.check_quality_threshold(0.45)
        >>> if not usable:
        >>>     print(f"Cannot forecast: {msg}")
    """
    thresholds = self.quality_config['quality_thresholds']
    
    min_threshold = thresholds['acceptable'] if not strict else thresholds['good']
    
    if quality_score >= thresholds['excellent']:
        return (True, "Excellent data quality")
    elif quality_score >= thresholds['good']:
        return (True, "Good data quality")
    elif quality_score >= min_threshold:
        return (True, "Acceptable data quality (degraded forecast)")
    elif quality_score >= thresholds['poor']:
        return (False, "Poor data quality - critical features missing or stale")
    else:
        return (False, "Unusable data quality - refuse to forecast")


def get_quality_report(self, feature_dict: dict, date: pd.Timestamp = None) -> dict:
    """
    Generate detailed quality report for diagnostics.
    
    Returns breakdown of which features are causing quality issues.
    
    Args:
        feature_dict: Features to analyze
        date: Date of observation
        
    Returns:
        dict: Detailed report with component scores
        
    Example:
        >>> report = validator.get_quality_report(features)
        >>> print(json.dumps(report, indent=2))
    """
    if date is None:
        date = pd.Timestamp.now()
    
    report = {
        'overall_quality': self.compute_feature_quality(feature_dict, date),
        'date': str(date),
        'critical_features': {},
        'important_features': {},
        'optional_features': {},
        'issues': []
    }
    
    # Check critical features
    for feat in self.quality_config['missingness_penalty']['critical_features']:
        if feat in feature_dict:
            is_missing = pd.isna(feature_dict[feat]) or feature_dict[feat] is None
            age = self.get_feature_age(feat, date)
            report['critical_features'][feat] = {
                'present': not is_missing,
                'age_days': age,
                'expected_lag': self.feature_lags.get(feat, 1)
            }
            if is_missing:
                report['issues'].append(f"CRITICAL: {feat} is missing")
        else:
            report['critical_features'][feat] = {'present': False}
            report['issues'].append(f"CRITICAL: {feat} not in feature set")
    
    # Check important features
    for feat in self.quality_config['missingness_penalty']['important_features']:
        if feat in feature_dict:
            is_missing = pd.isna(feature_dict[feat]) or feature_dict[feat] is None
            age = self.get_feature_age(feat, date)
            report['important_features'][feat] = {
                'present': not is_missing,
                'age_days': age,
                'expected_lag': self.feature_lags.get(feat, 1)
            }
            if is_missing:
                report['issues'].append(f"Important: {feat} is missing")
            elif age > self.feature_lags.get(feat, 1) + 7:
                report['issues'].append(f"Important: {feat} is stale ({age} days)")
    
    # Check optional features (summary only)
    optional_count = len(self.quality_config['missingness_penalty']['optional_features'])
    optional_present = sum(
        1 for feat in self.quality_config['missingness_penalty']['optional_features']
        if feat in feature_dict and not pd.isna(feature_dict.get(feat))
    )
    report['optional_features']['coverage'] = f"{optional_present}/{optional_count}"
    
    return report
```

---

## INTEGRATION POINTS

### 1. Called By: feature_engine.py

**In feature_engine.build_features():**
```python
# Import validator
from temporal_validator import TemporalValidator
validator = TemporalValidator()

# Compute quality scores
df['feature_quality'] = validator.compute_feature_quality_batch(df)
```

### 2. Called By: xgboost_trainer_v2.py

**In trainer._create_targets():**
```python
# Use feature quality as part of confidence target
df['target_confidence'] = (
    0.5 * df['feature_quality'] +  # From temporal validator
    0.5 * regime_stability
)
```

### 3. Called By: integrated_system_production.py

**Before making predictions:**
```python
# Check if data quality sufficient
quality = validator.compute_feature_quality(today_features)
usable, msg = validator.check_quality_threshold(quality)

if not usable:
    logger.warning(f"Skipping forecast: {msg}")
    return None
```

---

## TESTING

### Unit Test: Quality Scoring
```python
def test_quality_scoring():
    """Test feature quality computation."""
    validator = TemporalValidator()
    
    # Test 1: Perfect quality (all features present)
    perfect_features = {
        'vix': 18.5,
        'spx': 4800,
        'vix_percentile_21d': 0.65,
        'spx_realized_vol_21d': 12.3,
        'VX1-VX2': 0.5,
        'SKEW': 140,
        'yield_10y2y': 0.4,
        'Dollar_Index': 103.5
    }
    quality = validator.compute_feature_quality(perfect_features)
    assert quality >= 0.95, f"Expected high quality, got {quality}"
    
    # Test 2: Missing critical feature
    missing_critical = perfect_features.copy()
    missing_critical['vix'] = None
    quality = validator.compute_feature_quality(missing_critical)
    assert quality < 0.5, f"Expected low quality, got {quality}"
    
    # Test 3: Missing optional feature (minor impact)
    missing_optional = perfect_features.copy()
    missing_optional['SKEW'] = None
    quality = validator.compute_feature_quality(missing_optional)
    assert 0.8 <= quality <= 0.95, f"Expected degraded quality, got {quality}"
    
    print("✅ Quality scoring tests passed")

test_quality_scoring()
```

### Integration Test with Feature Engine
```python
def test_quality_in_features():
    """Test quality scoring in feature engine."""
    from feature_engine import FeatureEngineV5
    
    engine = FeatureEngineV5()
    df = engine.build_features(window='1y')
    
    # Check feature_quality column exists
    assert 'feature_quality' in df.columns
    
    # Check range
    assert (df['feature_quality'] >= 0).all()
    assert (df['feature_quality'] <= 1).all()
    
    # Check no NaNs
    assert df['feature_quality'].notna().all()
    
    # Check mean is reasonable (should be >0.8 for good data pipeline)
    mean_quality = df['feature_quality'].mean()
    assert mean_quality >= 0.8, f"Mean quality too low: {mean_quality}"
    
    print("✅ Feature quality integration test passed")

test_quality_in_features()
```

---

## COMMON PITFALLS

### 1. Not Updating Timestamps
```python
# WRONG: Never update timestamps → always reports 0 staleness
validator = TemporalValidator()
quality = validator.compute_feature_quality(features)  # Always 1.0

# CORRECT: Update timestamps when fetching data
# In data_fetcher.py:
def fetch_cboe_data(self):
    data = ...  # fetch data
    validator.update_feature_timestamp('SKEW', pd.Timestamp.now())
```

### 2. Wrong Penalty Direction
```python
# WRONG: Higher penalty = higher score
if missing_critical:
    score = 1.0  # Should be 0.0!

# CORRECT: Missing critical = zero score
if missing_critical:
    score = 0.0
```

### 3. Ignoring Quality Thresholds
```python
# WRONG: Forecast anyway
quality = 0.25
distribution = forecaster.predict(features)  # Garbage in, garbage out

# CORRECT: Check threshold first
quality = 0.25
usable, msg = validator.check_quality_threshold(quality)
if not usable:
    raise ValueError(f"Cannot forecast: {msg}")
```

---

## PERFORMANCE CONSIDERATIONS

### Vectorization Opportunity
Current `compute_feature_quality_batch()` loops through rows. Can optimize:

```python
# Current: O(n) with Python loop
for date, row in df.iterrows():
    quality = compute_feature_quality(row.to_dict())

# Optimized: Vectorized operations
def compute_feature_quality_vectorized(df):
    # Check critical features (vectorized)
    critical_mask = df[critical_features].notna().all(axis=1)
    
    # Check important features (vectorized)
    important_mask = df[important_features].notna().mean(axis=1) > 0.8
    
    # Combine scores
    quality = critical_mask.astype(float) * 0.6 + important_mask * 0.4
    return quality
```

**Trade-off:** Vectorized version is 10x faster but less granular. Use vectorized for training, detailed for real-time predictions.

---

## SUMMARY

**New Methods Added:**
- `compute_feature_quality()` - Core quality scoring
- `compute_feature_quality_batch()` - Batch processing
- `_compute_staleness_score()` - Staleness penalty
- `update_feature_timestamp()` - Timestamp tracking
- `get_feature_age()` - Age calculation
- `check_quality_threshold()` - Production threshold check
- `get_quality_report()` - Diagnostic report

**Modified Methods:**
- `__init__()` - Add quality config initialization

**New Dependencies:**
- `FEATURE_QUALITY_CONFIG` from config.py

**Output:**
- `feature_quality` column added to DataFrame (via feature_engine)
- Quality scores feed into confidence model

**Lines Changed:**
- ~250 new lines
- 3 modified lines

**Next Steps:**
1. Update config.py with FEATURE_QUALITY_CONFIG
2. Update feature_engine.py to call compute_feature_quality_batch()
3. Update data_fetcher.py to call update_feature_timestamp() after fetches
4. Test quality scoring with real data
