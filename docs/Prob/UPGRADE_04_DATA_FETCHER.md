# UPGRADE GUIDE: data_fetcher.py
## Adding FOMC Calendar Integration

---

## SYSTEM CONTEXT

### What We're Adding
A new data source: **FOMC meeting calendar**. This lets feature_engine.py classify dates into FOMC-related cohorts (fomc_week, fomc_minus_3).

**Why This Matters:**
- VIX behaves differently around FOMC meetings (typically compression before, volatility after)
- By treating FOMC as a training cohort, models learn these dynamics explicitly
- Without calendar: Model sees FOMC as random noise
- With calendar: Model learns "VIX distribution on FOMC day looks like X"

---

## FILE ROLE: data_fetcher.py

**Current Purpose:**
- Unified interface for fetching market data
- Sources: Yahoo Finance (SPX, VIX, commodities), CBOE (volatility indices), FRED (macro)
- Caching with parquet files
- Rate limiting and error handling

**What's Changing:**
- ADD: `fetch_fomc_calendar()` method
- ADD: FOMC calendar CSV management
- KEEP: All existing fetcher logic unchanged

---

## REQUIRED CONTEXT FROM OTHER FILES

### From config.py
```python
# No new imports needed - just file paths
# FOMC data will be stored in: data_cache/fomc_calendar.csv
```

### Used By: feature_engine.py
```python
# In feature_engine.py
from data_fetcher import UnifiedDataFetcher

fetcher = UnifiedDataFetcher()
fomc_df = fetcher.fetch_fomc_calendar()  # Returns DataFrame with FOMC dates
```

---

## DETAILED CHANGES

### CHANGE 1: Add FOMC Calendar Method

**LOCATION:** Add new method to `UnifiedDataFetcher` class

**INSERT AFTER EXISTING FETCH METHODS:**
```python
def fetch_fomc_calendar(self, start_year=2009, end_year=2030):
    """
    Fetch or generate FOMC meeting calendar.
    
    Strategy:
      1. Check for cached CSV (data_cache/fomc_calendar.csv)
      2. If missing, generate from hardcoded schedule
      3. In production, could scrape from federalreserve.gov or use API
    
    Args:
        start_year: Earliest year to include
        end_year: Latest year to include
        
    Returns:
        DataFrame with columns:
            - index: pd.DatetimeIndex (meeting date)
            - meeting_type: str ('scheduled', 'emergency', 'minutes_release')
            
    Example:
        >>> df = fetch_fomc_calendar()
        >>> df.loc['2025-01-29']
        meeting_type    scheduled
        Name: 2025-01-29, dtype: object
    """
    cache_path = Path('data_cache') / 'fomc_calendar.csv'
    
    # Try loading from cache
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
            
            # Filter to requested year range
            df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
            
            logger.info(f"✅ FOMC calendar loaded: {len(df)} meetings ({start_year}-{end_year})")
            return df
        except Exception as e:
            logger.warning(f"⚠️  Failed to load FOMC calendar: {e}")
            logger.info("   Regenerating from defaults...")
    
    # Generate default calendar
    df = self._generate_default_fomc_calendar(start_year, end_year)
    
    # Save to cache
    cache_path.parent.mkdir(exist_ok=True)
    df.to_csv(cache_path)
    logger.info(f"💾 FOMC calendar saved: {cache_path}")
    
    return df


def _generate_default_fomc_calendar(self, start_year, end_year):
    """
    Generate FOMC calendar from known schedule pattern.
    
    FOMC meets 8 times per year, typically:
      - End of January
      - Mid March
      - Early May
      - Mid June
      - End of July
      - Mid September
      - Early November
      - Mid December
    
    Pattern: Roughly every 6 weeks, avoiding major holidays.
    
    Returns:
        DataFrame with FOMC meeting dates
    """
    meetings = []
    
    # Historical and projected FOMC dates
    # Source: Federal Reserve official calendar
    fomc_schedule = {
        2024: [
            '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
            '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18'
        ],
        2025: [
            '2025-01-29', '2025-03-19', '2025-04-30', '2025-06-11',
            '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17'
        ],
        # Add more years as needed
        # Pattern repeats with slight variations
    }
    
    # Historical data (2009-2023)
    historical_meetings = [
        # 2009
        '2009-01-28', '2009-03-18', '2009-04-29', '2009-06-24',
        '2009-08-12', '2009-09-23', '2009-11-04', '2009-12-16',
        # 2010
        '2010-01-27', '2010-03-16', '2010-04-28', '2010-06-23',
        '2010-08-10', '2010-09-21', '2010-11-03', '2010-12-14',
        # 2011-2023 (add actual dates from Federal Reserve archives)
        # For brevity, showing pattern...
        # Full list at: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    ]
    
    # Combine historical and scheduled
    all_dates = historical_meetings.copy()
    for year in range(start_year, end_year + 1):
        if year in fomc_schedule:
            all_dates.extend(fomc_schedule[year])
    
    # Create DataFrame
    for date_str in all_dates:
        date = pd.Timestamp(date_str)
        if start_year <= date.year <= end_year:
            meetings.append({
                'date': date,
                'meeting_type': 'scheduled'
            })
    
    df = pd.DataFrame(meetings)
    df = df.set_index('date').sort_index()
    
    # Remove duplicates (if any)
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"📅 Generated FOMC calendar: {len(df)} meetings")
    
    return df
```

**Implementation Notes:**
- **Hardcoded dates:** For 2024-2025, use official Federal Reserve schedule
- **Historical dates:** For 2009-2023, manually add from Federal Reserve archives
- **Future dates:** For 2026+, use pattern (every 6 weeks, 8 times/year)
- **Cache first:** Always check CSV before regenerating

---

### CHANGE 2: Add Helper for Manual Calendar Updates

**LOCATION:** Add utility method (optional but recommended)

```python
def update_fomc_calendar_from_csv(self, csv_path: str):
    """
    Update FOMC calendar from user-provided CSV.
    
    Useful for:
      - Adding emergency meetings (COVID, 2008 crisis)
      - Updating with latest Fed announcements
      - Incorporating minutes release dates
    
    Args:
        csv_path: Path to CSV with columns [date, meeting_type]
        
    Example CSV format:
        date,meeting_type
        2024-01-31,scheduled
        2024-03-20,scheduled
        2024-08-05,emergency
    """
    import_df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # Merge with existing calendar
    existing_df = self.fetch_fomc_calendar()
    
    # Combine and deduplicate
    combined = pd.concat([existing_df.reset_index(), import_df])
    combined = combined.drop_duplicates(subset=['date'], keep='last')
    combined = combined.set_index('date').sort_index()
    
    # Save updated calendar
    cache_path = Path('data_cache') / 'fomc_calendar.csv'
    combined.to_csv(cache_path)
    
    logger.info(f"✅ FOMC calendar updated: {len(combined)} total meetings")
    logger.info(f"   Added {len(combined) - len(existing_df)} new meetings")
    
    return combined
```

**Usage:**
```python
# User creates fomc_custom.csv with emergency meetings
fetcher = UnifiedDataFetcher()
fetcher.update_fomc_calendar_from_csv('fomc_custom.csv')
```

---

### CHANGE 3: Create Default FOMC Calendar CSV

**LOCATION:** Create new file `data_cache/fomc_calendar.csv`

**Content (Example for 2024-2025):**
```csv
date,meeting_type
2024-01-31,scheduled
2024-03-20,scheduled
2024-05-01,scheduled
2024-06-12,scheduled
2024-07-31,scheduled
2024-09-18,scheduled
2024-11-07,scheduled
2024-12-18,scheduled
2025-01-29,scheduled
2025-03-19,scheduled
2025-04-30,scheduled
2025-06-11,scheduled
2025-07-30,scheduled
2025-09-17,scheduled
2025-11-05,scheduled
2025-12-17,scheduled
```

**How to Generate Full Historical Calendar:**
1. Visit: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
2. Scrape meeting dates for 2009-2023
3. Format as CSV
4. Place in `data_cache/`

**Or use this quick Python script:**
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_fomc_calendar(year):
    """Scrape FOMC meetings from Fed website."""
    url = f"https://www.federalreserve.gov/monetarypolicy/fomccalendars{year}.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Parse meeting dates (HTML structure varies by year)
    # This is a simplified example - real implementation needs robust parsing
    dates = []
    # ... parsing logic ...
    
    return dates

# Generate full calendar
all_meetings = []
for year in range(2009, 2026):
    meetings = scrape_fomc_calendar(year)
    all_meetings.extend(meetings)

df = pd.DataFrame(all_meetings, columns=['date', 'meeting_type'])
df.to_csv('data_cache/fomc_calendar.csv', index=False)
```

---

## INTEGRATION POINTS

### 1. Called By: feature_engine.py
```python
# In feature_engine._load_calendar_data()
self.fomc_calendar = self.data_fetcher.fetch_fomc_calendar()

# Later used in:
def _days_to_fomc(self, date):
    future_fomc = self.fomc_calendar[self.fomc_calendar.index >= date]
    next_fomc = future_fomc.index[0]
    return -(next_fomc - date).days
```

### 2. Optional: Used by diagnostic scripts
```python
# In diagnostics/calendar_coverage.py
from data_fetcher import UnifiedDataFetcher

fetcher = UnifiedDataFetcher()
fomc_df = fetcher.fetch_fomc_calendar(2009, 2025)

print(f"Total FOMC meetings: {len(fomc_df)}")
print(f"Years covered: {fomc_df.index.year.unique()}")
```

---

## TESTING

### Unit Test
```python
def test_fomc_calendar():
    """Test FOMC calendar fetching."""
    from data_fetcher import UnifiedDataFetcher
    
    fetcher = UnifiedDataFetcher()
    df = fetcher.fetch_fomc_calendar(2024, 2025)
    
    # Should have 8 meetings per year
    assert len(df) >= 16  # At least 16 for 2024-2025
    
    # Check index is DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    
    # Check column exists
    assert 'meeting_type' in df.columns
    
    # Check specific known dates
    assert '2024-01-31' in df.index.astype(str).tolist()
    assert '2025-01-29' in df.index.astype(str).tolist()
    
    # Check chronological order
    assert df.index.is_monotonic_increasing
    
    print("✅ FOMC calendar test passed")

test_fomc_calendar()
```

### Integration Test with Feature Engine
```python
def test_fomc_in_feature_engine():
    """Test that feature_engine can use FOMC calendar."""
    from feature_engine import FeatureEngineV5
    
    engine = FeatureEngineV5()
    engine._load_calendar_data()
    
    # Check FOMC calendar loaded
    assert engine.fomc_calendar is not None
    assert len(engine.fomc_calendar) > 0
    
    # Test cohort assignment on FOMC day
    fomc_date = pd.Timestamp('2025-01-29')
    cohort, weight = engine.get_calendar_cohort(fomc_date)
    assert cohort == 'fomc_week'
    assert weight == 1.4
    
    print("✅ FOMC integration test passed")

test_fomc_in_feature_engine()
```

---

## COMMON PITFALLS

### 1. Missing Historical Data
```python
# WRONG: Only include 2024-2025
fomc_schedule = {2024: [...], 2025: [...]}
# Problem: Training on 15 years needs 2009-2023 data!

# CORRECT: Include full historical range
fomc_schedule = {
    2009: [...], 2010: [...], ..., 2025: [...]
}
```

### 2. Date Format Inconsistencies
```python
# WRONG: Mix date formats
meetings = ['2024-01-31', '2024/03/20', 'March 19, 2025']

# CORRECT: Use ISO format consistently
meetings = ['2024-01-31', '2024-03-20', '2025-03-19']
```

### 3. Timezone Issues
```python
# WRONG: Create dates without timezone
date = pd.Timestamp('2025-01-29 14:00:00', tz='US/Eastern')
# Problem: Feature engine uses tz-naive dates

# CORRECT: Use date only (no time)
date = pd.Timestamp('2025-01-29')
```

### 4. Not Handling Missing File
```python
# WRONG: Crash if CSV missing
df = pd.read_csv('fomc_calendar.csv')  # FileNotFoundError!

# CORRECT: Fallback to generated calendar
try:
    df = pd.read_csv('fomc_calendar.csv')
except FileNotFoundError:
    df = self._generate_default_fomc_calendar()
```

---

## ALTERNATIVE APPROACHES

### Option 1: Live Scraping (Not Recommended)
```python
def fetch_fomc_calendar_live():
    """Scrape directly from Fed website on each run."""
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    # ... scraping logic ...
    # CONS: Slow, brittle (HTML changes), rate limits
```

### Option 2: Third-Party API
```python
def fetch_fomc_calendar_api():
    """Use financial data API."""
    # e.g., Alpha Vantage, Quandl, Bloomberg API
    # CONS: Requires API key, costs money, overkill for simple calendar
```

### Option 3: Manual CSV (Recommended)
```python
def fetch_fomc_calendar_csv():
    """Load from manually curated CSV."""
    return pd.read_csv('data_cache/fomc_calendar.csv')
    # PROS: Reliable, fast, no dependencies
    # CONS: Requires manual updates (but only 8 times/year)
```

**Recommendation:** Use Option 3 (manual CSV) for production. Update CSV quarterly with upcoming meetings.

---

## MAINTENANCE GUIDE

### Quarterly Update Process
1. Visit Federal Reserve website in January, April, July, October
2. Check for newly announced meetings
3. Add to `data_cache/fomc_calendar.csv`
4. Commit to git

**Example (January 2026 update):**
```bash
# Add 2026 meetings to CSV
echo "2026-01-28,scheduled" >> data_cache/fomc_calendar.csv
echo "2026-03-18,scheduled" >> data_cache/fomc_calendar.csv
# ... rest of 2026 meetings

git add data_cache/fomc_calendar.csv
git commit -m "Update FOMC calendar with 2026 meetings"
```

### Emergency Meeting Handling
If Fed announces emergency meeting (like March 2020 COVID response):
```bash
# Add to CSV immediately
echo "2026-03-15,emergency" >> data_cache/fomc_calendar.csv

# Or use Python:
from data_fetcher import UnifiedDataFetcher
fetcher = UnifiedDataFetcher()
emergency_df = pd.DataFrame({
    'date': ['2026-03-15'],
    'meeting_type': ['emergency']
})
fetcher.update_fomc_calendar_from_csv('emergency_fomc.csv')
```

---

## SUMMARY

**New Methods Added:**
- `fetch_fomc_calendar()` - Main method for calendar retrieval
- `_generate_default_fomc_calendar()` - Fallback generator
- `update_fomc_calendar_from_csv()` - Manual update utility

**New Files Required:**
- `data_cache/fomc_calendar.csv` - FOMC meeting dates (2009-2030)

**Dependencies:**
- None (uses standard library + pandas)

**Lines Changed:**
- ~150 new lines
- 0 modified lines (pure addition)

**Integration:**
- Used by: feature_engine.py (for cohort classification)
- Transparent to: Other files (encapsulated in data_fetcher)

**Next Steps:**
1. Create `data_cache/fomc_calendar.csv` with historical dates
2. Test `fetch_fomc_calendar()` method
3. Verify feature_engine.py can load calendar
4. Update calendar quarterly going forward
