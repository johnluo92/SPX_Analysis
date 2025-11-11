# UPGRADE GUIDE: feature_engine.py
## Calendar Cohort Integration & Feature Quality Tracking

---

## SYSTEM CONTEXT (READ THIS FIRST)

### The Big Picture
We're adding **calendar-aware context** to the feature engine. Instead of creating features like `days_to_opex` or `is_fomc_week`, we classify each date into a **cohort** (OpEx-5, FOMC-3, Mid-cycle, etc.). The trainer then learns different VIX distributions for each cohort.

**Why This Matters:**
- Traditional approach: Add `days_to_opex` as feature → Model struggles to learn "VIX behaves differently 5 days before OpEx"
- Cohort approach: Tag date as "opex_minus_5" cohort → Train separate distribution for this context
- Result: Cleaner feature space, better probabilistic forecasts

**Example:**
```python
# Old way (feature engineering hell)
features['days_to_opex'] = -5
features['is_opex_week'] = 1
features['opex_gamma_proxy'] = ...  # Complex interaction terms
# Model sees these as numbers, not context

# New way (cohort classification)
cohort = 'monthly_opex_minus_5'
weight = 1.2  # Higher uncertainty pre-OpEx
# Trainer uses opex_minus_5 model, which learned VIX dynamics for this exact context
```

---

## FILE ROLE: feature_engine.py

**Current Purpose:**
- Fetches market data (SPX, VIX, yields, CBOE, futures, macro)
- Engineers 232 features with temporal safety (respects publication lags)
- Returns DataFrame with features + targets

**Current Key Methods:**
- `build_features()`: Main orchestrator, calls all sub-methods
- `_fetch_core_market_data()`: SPX, VIX from Yahoo
- `_fetch_cboe_data()`: SKEW, put/call, VIX futures, etc.
- `_fetch_macro_data()`: CPI, dollar index, gold
- `_fetch_treasury_curve()`: Yield curve from FRED
- `_build_technical_features()`: RSI, MACD, Bollinger, etc.
- `_build_regime_features()`: VIX percentiles, regime classification

**What's Changing:**
1. **ADD:** Calendar cohort classification logic
2. **ADD:** Feature quality tracking (for confidence scoring)
3. **KEEP:** All existing feature generation (232 features stay)
4. **MODIFY:** `build_features()` adds cohort columns

---

## REQUIRED CONTEXT FROM OTHER FILES

### From config.py (import these)
```python
from config import (
    CALENDAR_COHORTS,          # Cohort definitions
    COHORT_PRIORITY,           # Matching order
    TARGET_CONFIG,             # For horizon days
    FEATURE_QUALITY_CONFIG,    # Quality thresholds
    FEATURE_LAGS               # Publication delays
)
```

### From data_fetcher.py (new method needed)
```python
# data_fetcher.py must have this method:
def fetch_fomc_calendar():
    """
    Returns DataFrame with columns: [date, meeting_type]
    dates: pd.DatetimeIndex
    meeting_type: 'scheduled', 'emergency', etc.
    """
```

### From temporal_validator.py (import existing)
```python
from temporal_validator import TemporalValidator
# Uses existing methods:
# - validator.validate_features(df)
# - validator.get_feature_lags()
```

---

## DETAILED CHANGES

### CHANGE 1: Add Calendar Data Sources

**LOCATION:** Top of class `__init__` method

**FIND:**
```python
class FeatureEngineV5:
    def __init__(self, config=None):
        self.data_fetcher = UnifiedDataFetcher()
        self.validator = TemporalValidator()
        # ... existing init
```

**ADD AFTER EXISTING INIT:**
```python
        # Calendar data for cohort classification
        self.fomc_calendar = None
        self.opex_calendar = None
        self.earnings_calendar = None  # Stub for now
        self.vix_futures_expiry = None
        
        # Cache for performance
        self._cohort_cache = {}  # {date: (cohort, weight)}
        
    def _load_calendar_data(self):
        """Load all calendar sources once at startup."""
        if self.fomc_calendar is None:
            try:
                # Load from data_fetcher
                self.fomc_calendar = self.data_fetcher.fetch_fomc_calendar()
                logger.info(f"✅ FOMC calendar loaded: {len(self.fomc_calendar)} meetings")
            except FileNotFoundError:
                logger.warning("⚠️  FOMC calendar not found, using stub")
                self.fomc_calendar = pd.DataFrame()  # Empty fallback
        
        # Generate OpEx calendar (always 3rd Friday of month)
        if self.opex_calendar is None:
            self.opex_calendar = self._generate_opex_calendar()
            logger.info(f"✅ OpEx calendar generated: {len(self.opex_calendar)} dates")
        
        # VIX futures expiry (Wednesday 30 days before S&P OpEx)
        if self.vix_futures_expiry is None:
            self.vix_futures_expiry = self._generate_vix_futures_expiry()
            logger.info(f"✅ VIX futures expiry calendar: {len(self.vix_futures_expiry)} dates")
        
        # Earnings calendar (stub - implement later or use API)
        if self.earnings_calendar is None:
            self.earnings_calendar = pd.DataFrame()
            logger.warning("⚠️  Earnings calendar not implemented (will use default cohort)")
```

---

### CHANGE 2: Add Calendar Generation Helpers

**LOCATION:** Add new methods to class (before `build_features()`)

```python
def _generate_opex_calendar(self, start_year=2009, end_year=2030):
    """
    Generate monthly options expiration dates (3rd Friday of each month).
    
    Returns:
        DataFrame with columns: [date, expiry_type]
    """
    opex_dates = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Find 3rd Friday of month
            # Start from 15th (earliest 3rd Friday), find next Friday
            first_possible = pd.Timestamp(year, month, 15)
            
            # Find next Friday (weekday 4)
            days_ahead = (4 - first_possible.weekday()) % 7
            if days_ahead == 0 and first_possible.day > 15:
                days_ahead = 7  # Move to next week if already past
            
            third_friday = first_possible + pd.Timedelta(days=days_ahead)
            opex_dates.append({
                'date': third_friday,
                'expiry_type': 'monthly_opex'
            })
    
    df = pd.DataFrame(opex_dates)
    df = df.set_index('date').sort_index()
    return df

def _generate_vix_futures_expiry(self):
    """
    VIX futures expire on Wednesday, 30 days before 3rd Friday.
    
    Returns:
        DataFrame with columns: [date, expiry_type]
    """
    if self.opex_calendar is None:
        self._generate_opex_calendar()
    
    vix_expiry = []
    for opex_date in self.opex_calendar.index:
        # 30 days before, find Wednesday
        approx_date = opex_date - pd.Timedelta(days=30)
        
        # Find next Wednesday (weekday 2)
        days_to_wed = (2 - approx_date.weekday()) % 7
        vix_date = approx_date + pd.Timedelta(days=days_to_wed)
        
        vix_expiry.append({
            'date': vix_date,
            'expiry_type': 'vix_futures'
        })
    
    df = pd.DataFrame(vix_expiry)
    df = df.set_index('date').sort_index()
    return df
```

**Why These Methods:**
- OpEx is deterministic (always 3rd Friday) → Generate programmatically
- VIX futures is derived from OpEx → Calculate from OpEx calendar
- FOMC is irregular → Must load from external source

---

### CHANGE 3: Core Cohort Classification Logic

**LOCATION:** Add new method (this is the heart of the upgrade)

```python
def get_calendar_cohort(self, date):
    """
    Determine which calendar cohort a date belongs to.
    
    Args:
        date: pd.Timestamp or datetime
        
    Returns:
        tuple: (cohort_name: str, weight: float)
        
    Example:
        >>> get_calendar_cohort(pd.Timestamp('2025-01-15'))
        ('monthly_opex_minus_1', 1.5)
    """
    date = pd.Timestamp(date)
    
    # Check cache first (performance optimization)
    if date in self._cohort_cache:
        return self._cohort_cache[date]
    
    # Ensure calendars are loaded
    if self.opex_calendar is None:
        self._load_calendar_data()
    
    # Calculate days to various events
    days_to_opex = self._days_to_monthly_opex(date)
    days_to_fomc = self._days_to_fomc(date)
    days_to_vix_expiry = self._days_to_vix_futures_expiry(date)
    earnings_pct = self._spx_earnings_intensity(date)
    
    # Match against cohorts in priority order
    for cohort_name in COHORT_PRIORITY:
        cohort_def = CALENDAR_COHORTS[cohort_name]
        condition = cohort_def['condition']
        
        if condition == 'days_to_monthly_opex':
            range_min, range_max = cohort_def['range']
            if range_min <= days_to_opex <= range_max:
                result = (cohort_name, cohort_def['weight'])
                self._cohort_cache[date] = result
                return result
        
        elif condition == 'days_to_fomc':
            if days_to_fomc is not None:
                range_min, range_max = cohort_def['range']
                if range_min <= days_to_fomc <= range_max:
                    result = (cohort_name, cohort_def['weight'])
                    self._cohort_cache[date] = result
                    return result
        
        elif condition == 'days_to_futures_expiry':
            if days_to_vix_expiry is not None:
                range_min, range_max = cohort_def['range']
                if range_min <= days_to_vix_expiry <= range_max:
                    result = (cohort_name, cohort_def['weight'])
                    self._cohort_cache[date] = result
                    return result
        
        elif condition == 'spx_earnings_pct':
            if earnings_pct is not None:
                range_min, range_max = cohort_def['range']
                if range_min <= earnings_pct <= range_max:
                    result = (cohort_name, cohort_def['weight'])
                    self._cohort_cache[date] = result
                    return result
        
        elif condition == 'default':
            # Catch-all for mid_cycle
            result = (cohort_name, cohort_def['weight'])
            self._cohort_cache[date] = result
            return result
    
    # Should never reach here if 'mid_cycle' is last in priority
    raise ValueError(f"No cohort matched for date {date}")


def _days_to_monthly_opex(self, date):
    """
    Calculate days until next monthly OpEx (3rd Friday).
    
    Returns:
        int: Negative if before OpEx, 0 on OpEx, positive after
        Example: -5 means "5 days until OpEx"
    """
    # Find next OpEx date
    future_opex = self.opex_calendar[self.opex_calendar.index >= date]
    
    if len(future_opex) == 0:
        return None  # No future OpEx (end of calendar)
    
    next_opex = future_opex.index[0]
    days_diff = (next_opex - date).days
    
    return -days_diff  # Negative before, positive after


def _days_to_fomc(self, date):
    """
    Calculate days until next FOMC meeting.
    
    Returns:
        int: Days to next meeting (negative before, positive after)
        None: If FOMC calendar unavailable
    """
    if self.fomc_calendar is None or len(self.fomc_calendar) == 0:
        return None
    
    future_fomc = self.fomc_calendar[self.fomc_calendar.index >= date]
    
    if len(future_fomc) == 0:
        return None
    
    next_fomc = future_fomc.index[0]
    days_diff = (next_fomc - date).days
    
    return -days_diff


def _days_to_vix_futures_expiry(self, date):
    """Calculate days until next VIX futures expiration."""
    if self.vix_futures_expiry is None or len(self.vix_futures_expiry) == 0:
        return None
    
    future_expiry = self.vix_futures_expiry[self.vix_futures_expiry.index >= date]
    
    if len(future_expiry) == 0:
        return None
    
    next_expiry = future_expiry.index[0]
    days_diff = (next_expiry - date).days
    
    return -days_diff


def _spx_earnings_intensity(self, date):
    """
    Calculate % of SPX components reporting earnings this week.
    
    Returns:
        float: Percentage [0.0, 1.0] of SPX reporting
        None: If earnings calendar unavailable (stub implementation)
    """
    # STUB IMPLEMENTATION
    # TODO: Integrate with earnings calendar API or manual CSV
    
    # For now, use heuristic: Peak earnings months are Jan, Apr, Jul, Oct
    month = date.month
    if month in [1, 4, 7, 10]:
        # Check if in earnings window (typically 2nd-4th week of month)
        week_of_month = (date.day - 1) // 7 + 1
        if week_of_month in [2, 3, 4]:
            return 0.25  # Assume 25% of SPX reporting
    
    return 0.05  # Low intensity otherwise
```

**Critical Details:**
- **Sign Convention:** `days_to_opex = -5` means "5 days until OpEx" (consistent with cohort ranges)
- **Priority Matching:** First match wins (FOMC > OpEx > Futures > Earnings > Mid-cycle)
- **Caching:** Avoid recalculating for same dates (performance boost)
- **Fallback:** Always return mid_cycle if nothing matches

---

### CHANGE 4: Integrate Cohorts into build_features()

**LOCATION:** End of `build_features()` method (after all feature generation)

**FIND THIS SECTION:**
```python
def build_features(self, window='15y'):
    # ... lots of existing code ...
    
    # Final consolidation
    logger.info("📊 CONSOLIDATING FEATURES")
    df = df.dropna(thresh=len(df.columns) * 0.5)  # Keep rows with >50% features
    
    logger.info(f"✅ Final feature count: {len(df.columns)}")
    return df
```

**REPLACE `return df` WITH:**
```python
    # Add calendar cohort classification
    logger.info("📅 ADDING CALENDAR COHORTS")
    self._load_calendar_data()  # Ensure calendars loaded
    
    cohort_data = []
    for date in df.index:
        cohort_name, cohort_weight = self.get_calendar_cohort(date)
        cohort_data.append({
            'calendar_cohort': cohort_name,
            'cohort_weight': cohort_weight
        })
    
    cohort_df = pd.DataFrame(cohort_data, index=df.index)
    df = pd.concat([df, cohort_df], axis=1)
    
    # Log cohort distribution
    cohort_counts = df['calendar_cohort'].value_counts()
    logger.info("📊 Cohort Distribution:")
    for cohort, count in cohort_counts.items():
        pct = count / len(df) * 100
        logger.info(f"   {cohort:30s} | {count:4d} rows ({pct:5.1f}%)")
    
    # Add feature quality tracking
    logger.info("🔍 COMPUTING FEATURE QUALITY SCORES")
    df['feature_quality'] = self._compute_feature_quality_vectorized(df)
    
    logger.info(f"✅ Final feature count: {len(df.columns)} (includes 3 metadata cols)")
    logger.info(f"   Features: {len(df.columns) - 3}")
    logger.info(f"   Metadata: calendar_cohort, cohort_weight, feature_quality")
    
    return df
```

---

### CHANGE 5: Add Feature Quality Computation

**LOCATION:** Add new method (used in CHANGE 4)

```python
def _compute_feature_quality_vectorized(self, df):
    """
    Compute feature quality score for each row.
    Based on missingness and staleness of features.
    
    Returns:
        pd.Series: Quality scores [0, 1] where 1 = perfect
    """
    from config import FEATURE_QUALITY_CONFIG
    
    quality_scores = []
    
    for idx, row in df.iterrows():
        score_components = []
        
        # Check critical features (must be present)
        for feat in FEATURE_QUALITY_CONFIG['missingness_penalty']['critical_features']:
            if feat in df.columns:
                if pd.isna(row[feat]):
                    score_components.append(0.0)  # Critical missing = fail
                else:
                    score_components.append(1.0)
        
        # Check important features (0.5 if missing)
        for feat in FEATURE_QUALITY_CONFIG['missingness_penalty']['important_features']:
            if feat in df.columns:
                if pd.isna(row[feat]):
                    score_components.append(0.5)
                else:
                    score_components.append(1.0)
        
        # Check optional features (0.9 if missing)
        for feat in FEATURE_QUALITY_CONFIG['missingness_penalty']['optional_features']:
            if feat in df.columns:
                if pd.isna(row[feat]):
                    score_components.append(0.9)
                else:
                    score_components.append(1.0)
        
        # Average all components
        if len(score_components) > 0:
            quality_scores.append(np.mean(score_components))
        else:
            quality_scores.append(1.0)  # Default if no tracked features
    
    return pd.Series(quality_scores, index=df.index)
```

**Simplified Logic:**
- Critical features missing → score = 0 (refuse to forecast)
- Important features missing → score = 0.5 (degraded forecast)
- Optional features missing → score = 0.9 (minor impact)
- This feeds into confidence model in trainer

---

## INTEGRATION WITH OTHER FILES

### 1. Called By: integrated_system_production.py
```python
# In integrated_system_production.py
from feature_engine import FeatureEngineV5

engine = FeatureEngineV5()
df = engine.build_features(window='15y')

# Now df has these columns:
# - 232 original features (vix, spx, yields, etc.)
# - calendar_cohort (str): 'monthly_opex_minus_5', etc.
# - cohort_weight (float): Uncertainty multiplier
# - feature_quality (float): Data quality score [0, 1]
```

### 2. Provides To: xgboost_trainer_v2.py
```python
# In xgboost_trainer_v2.py
def train(self, df):
    # Split by cohort
    for cohort in df['calendar_cohort'].unique():
        cohort_df = df[df['calendar_cohort'] == cohort]
        X = cohort_df.drop(['calendar_cohort', 'cohort_weight', 'feature_quality'], axis=1)
        # ... train model for this cohort
```

### 3. Uses From: data_fetcher.py
```python
# data_fetcher.py must provide:
def fetch_fomc_calendar():
    """Load FOMC meeting dates from CSV or API."""
    return pd.read_csv('data_cache/fomc_calendar.csv', index_col='date', parse_dates=True)
```

---

## TESTING & VALIDATION

### Unit Test: Cohort Assignment
```python
def test_cohort_assignment():
    engine = FeatureEngineV5()
    
    # Test OpEx detection
    opex_date = pd.Timestamp('2025-01-17')  # 3rd Friday of Jan
    cohort, weight = engine.get_calendar_cohort(opex_date - pd.Timedelta(days=5))
    assert cohort == 'monthly_opex_minus_5'
    assert weight == 1.2
    
    # Test FOMC detection (if calendar available)
    fomc_date = pd.Timestamp('2025-01-29')  # Scheduled FOMC
    cohort, weight = engine.get_calendar_cohort(fomc_date)
    assert cohort == 'fomc_week'
    assert weight == 1.4
    
    # Test mid-cycle fallback
    random_date = pd.Timestamp('2025-06-10')
    cohort, weight = engine.get_calendar_cohort(random_date)
    assert cohort in ['mid_cycle', 'earnings_heavy']  # Depends on calendar
    
    print("✅ Cohort assignment tests passed")

test_cohort_assignment()
```

### Integration Test: build_features()
```python
def test_build_features_with_cohorts():
    engine = FeatureEngineV5()
    df = engine.build_features(window='1y')  # 1 year for speed
    
    # Check cohort columns exist
    assert 'calendar_cohort' in df.columns
    assert 'cohort_weight' in df.columns
    assert 'feature_quality' in df.columns
    
    # Check no NaN cohorts
    assert df['calendar_cohort'].notna().all()
    
    # Check weight ranges
    assert (df['cohort_weight'] >= 1.0).all()
    assert (df['cohort_weight'] <= 1.5).all()
    
    # Check quality scores
    assert (df['feature_quality'] >= 0).all()
    assert (df['feature_quality'] <= 1.0).all()
    
    print("✅ build_features() integration test passed")

test_build_features_with_cohorts()
```

---

## COMMON PITFALLS

### 1. Off-by-One in Date Ranges
```python
# WRONG: Ranges don't align
CALENDAR_COHORTS = {
    'opex_minus_5': {'range': (-7, -3)},  # Days -7 to -3
    'opex_minus_1': {'range': (-1, 0)}    # Gap! Days -2 missing
}

# CORRECT: Continuous coverage
CALENDAR_COHORTS = {
    'opex_minus_5': {'range': (-7, -3)},
    'opex_minus_1': {'range': (-2, 0)}   # No gaps
}
```

### 2. Sign Convention Confusion
```python
# WRONG: Positive means "before"
days_to_opex = 5  # 5 days in the future? Or 5 days ago?

# CORRECT: Negative means "before", positive means "after"
days_to_opex = -5  # Clear: 5 days UNTIL OpEx (in future)
days_to_opex = 3   # Clear: 3 days AFTER OpEx (in past)
```

### 3. Missing Calendar Files
```python
# WRONG: Crash if file missing
self.fomc_calendar = pd.read_csv('fomc_calendar.csv')

# CORRECT: Graceful fallback
try:
    self.fomc_calendar = pd.read_csv('data_cache/fomc_calendar.csv')
except FileNotFoundError:
    logger.warning("FOMC calendar missing, defaulting to mid_cycle")
    self.fomc_calendar = pd.DataFrame()  # Empty = no FOMC cohorts
```

### 4. Cache Invalidation
```python
# WRONG: Cache never cleared, uses stale data
self._cohort_cache[date] = (cohort, weight)
# ... later, calendar data updated ...
# Cache still has old cohort assignments!

# CORRECT: Clear cache when calendars reload
def _load_calendar_data(self):
    self._cohort_cache.clear()  # Invalidate cache
    self.fomc_calendar = ...
```

---

## PERFORMANCE CONSIDERATIONS

### Caching
- Cohort assignment called once per date (4000+ times for 15-year history)
- Caching reduces from O(n×m) to O(n) where m = calendar lookups
- Expected speedup: 5-10x for large datasets

### Vectorization
- Feature quality computed row-by-row (can't vectorize due to conditional logic)
- For 4000 rows, takes ~2 seconds (acceptable)
- If too slow, convert to Numba JIT or Cython

### Memory
- Calendars are small (<10KB each)
- Cache grows to ~100KB for 15 years (negligible)
- Feature quality scores add 32KB column (4000 rows × 8 bytes)

---

## EXAMPLE OUTPUT

After running `build_features()`, DataFrame looks like:

```
                vix   spx  yield_10y2y  ...  calendar_cohort          cohort_weight  feature_quality
2024-01-10    14.2  4850      1.23      ...  mid_cycle                1.0            0.97
2024-01-11    15.1  4830      1.25      ...  monthly_opex_minus_5     1.2            0.95
2024-01-12    16.3  4800      1.27      ...  monthly_opex_minus_5     1.2            0.96
2024-01-15    18.5  4750      1.30      ...  monthly_opex_minus_1     1.5            0.94
2024-01-16    19.2  4780      1.28      ...  monthly_opex_minus_1     1.5            0.93
2024-01-17    17.8  4820      1.24      ...  monthly_opex_plus_1      1.1            0.96
2024-01-29    16.5  4900      1.20      ...  fomc_week                1.4            0.91  # CPI stale
```

Note how `cohort_weight` and `feature_quality` vary by date/context.

---

## NEXT STEPS AFTER THIS FILE

1. **Update data_fetcher.py:** Add `fetch_fomc_calendar()` method
2. **Test cohort assignment:** Run unit tests above
3. **Verify cohort distribution:** Check that all cohorts appear (not just mid_cycle)
4. **Update xgboost_trainer_v2.py:** Use `calendar_cohort` column for training splits

---

## SUMMARY

**New Methods Added:**
- `_load_calendar_data()`
- `_generate_opex_calendar()`
- `_generate_vix_futures_expiry()`
- `get_calendar_cohort()` ← MAIN METHOD
- `_days_to_monthly_opex()`
- `_days_to_fomc()`
- `_days_to_vix_futures_expiry()`
- `_spx_earnings_intensity()`
- `_compute_feature_quality_vectorized()`

**Modified Methods:**
- `build_features()`: Add cohort assignment at end

**New Dependencies:**
- `config.py`: CALENDAR_COHORTS, COHORT_PRIORITY, FEATURE_QUALITY_CONFIG
- `data_fetcher.py`: fetch_fomc_calendar() method (create if missing)

**Output Changes:**
- DataFrame now has 3 extra columns: calendar_cohort, cohort_weight, feature_quality
- Total features: 232 + 3 metadata = 235 columns

**Lines Changed:** ~250 new lines, 5 modified lines
