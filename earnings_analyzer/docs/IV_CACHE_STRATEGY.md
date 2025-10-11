# IV Cache Strategy - Time-Gated Fetching

## Overview

The IV (Implied Volatility) caching system fetches current IV from Yahoo Finance options chains using a time-gated window strategy to balance data freshness with API efficiency.

**Key Design Principles:**
1. **Market state detection** - Only fetch when market is "REGULAR"
2. **Time-gated windows** - Fetch at most 4 times per trading day
3. **Multi-expiration averaging** - Use up to 3 expirations ≤45 DTE
4. **Robust fallbacks** - Always return cache on fetch failure

---

## Fetch Strategy

### Market Hours Detection

**Method:** Use yfinance `marketState` field
```python
stock = yf.Ticker(ticker)
market_state = stock.info.get("marketState")
if market_state != "REGULAR":
    return cache.get(ticker)  # Use cached data
```

**Market States:**
- `"REGULAR"` - Normal trading hours, OK to fetch
- `"CLOSED"` - Market closed (weekends, holidays, after hours)
- Other states - Treat as closed

**Advantages:**
- No need for holiday calendar
- No need for timezone logic
- yfinance handles all edge cases
- Simple and reliable

---

## Fetch Windows

### Time-Gated Caching Rules

**Fetch only during these windows:**
```
Window 1: 10:00 AM - 11:59 AM ET
Window 2: 12:00 PM - 1:59 PM ET
Window 3: 2:00 PM - 3:29 PM ET
Window 4: 3:30 PM - 5:00 PM ET
```

**Logic:**
- If current time is in a window AND last fetch was before this window → fetch new IV
- If current time is in same window as last fetch → use cached IV
- If market closed or outside fetch hours → use cached IV

**Examples:**

**Scenario 1: Multiple requests same window**
- 10:15 AM: Fetch new IV ✅
- 10:30 AM: Use cache (still in Window 1) ✅
- 11:45 AM: Use cache (still in Window 1) ✅

**Scenario 2: Cross window threshold**
- 11:45 AM: Fetch new IV (Window 1) ✅
- 12:15 PM: Fetch new IV (crossed into Window 2) ✅
- 12:30 PM: Use cache (still in Window 2) ✅

**Scenario 3: End of day**
- 3:45 PM: Fetch new IV (Window 4) ✅
- 4:30 PM: Use cache (still in Window 4) ✅
- Next day 10:15 AM: Fetch new IV (>24h old) ✅

---

## Expiration Selection

### Rule: Use Expirations ≤45 DTE

**Primary rule:** Use all expirations with DTE ≤45 days, up to 3 maximum.

**Rationale:**
- Near-term IV more relevant for earnings plays
- 45 DTE captures next 6 weeks of expected movement
- Matches typical earnings cycle timing
- Balances VIX-like measurement with data availability

### Examples

**Case 1: Multiple weeklies available**
```
Available: [7d, 14d, 21d, 30d, 45d]
Use: [7d, 14d, 21d]  (first 3 ≤45 DTE)
```

**Case 2: Mix of weeklies and monthlies**
```
Available: [14d, 30d, 46d, 60d]
Use: [14d, 30d]  (only these are ≤45 DTE)
```

**Case 3: Only monthlies available**
```
Available: [45d, 70d, 90d]
Use: [45d]  (only this is ≤45 DTE)
```

**Case 4: All expirations >45 DTE (edge case)**
```
Available: [60d, 90d, 120d]
Use: [60d]  (fallback to single nearest)
```

### Fallback Logic

If NO expirations ≤45 DTE exist:
- Use single nearest expiration
- This handles thinly traded stocks with only monthlies
- Better than returning no data

---

## IV Calculation

### Multi-Expiration Averaging

**Goal:** Get robust IV estimate by averaging multiple data points.

**Process:**
1. For each expiration (up to 3):
   - Find ATM strike (nearest to current price)
   - Get ATM put IV
   - Get ATM call IV
   - Add both to list

2. Average all IVs collected (up to 6 total)
```python
avg_iv = sum([put1, call1, put2, call2, put3, call3]) / 6
```

**Example:**
```
Expiration 1 (7 DTE):  Put IV = 26%, Call IV = 27%
Expiration 2 (14 DTE): Put IV = 25%, Call IV = 26%
Expiration 3 (21 DTE): Put IV = 24%, Call IV = 25%

Average IV = (26 + 27 + 25 + 26 + 24 + 25) / 6 = 25.5%
```

**Advantages:**
- Smooths out bid-ask noise
- Reduces impact of single mispriced option
- More stable than single-expiration IV
- Closer to "true" market expectation

---

## Cache Structure

### JSON Format

**Location:** `cache/iv_cache.json`

**Structure:**
```json
{
  "AAPL": {
    "iv": 26.5,
    "dte": 7,
    "expiration": "2025-10-18",
    "expirations_used": [
      "2025-10-18",
      "2025-10-25", 
      "2025-11-01"
    ],
    "fetched_at": "2025-10-11T14:23:15-04:00"
  },
  "MSFT": {
    "iv": 28.3,
    "dte": 14,
    "expiration": "2025-10-25",
    "expirations_used": [
      "2025-10-25",
      "2025-11-01"
    ],
    "fetched_at": "2025-10-11T14:23:20-04:00"
  }
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `iv` | float | Averaged IV across all expirations (percent) |
| `dte` | int | Days to expiration of nearest expiration used |
| `expiration` | string | Date of nearest expiration (ISO format) |
| `expirations_used` | list | All expiration dates used in average |
| `fetched_at` | string | Timestamp of fetch (ISO format with timezone) |

### Cache Staleness

**24-Hour Expiry:**
- If last fetch >24 hours ago → fetch new data
- Protects against using outdated IV
- Handles multi-day gaps (weekends, holidays)

**Same-Day Staleness:**
- If last fetch was today → check fetch window
- Only refetch if crossed into new window
- Prevents excessive API calls

---

## Implementation Details

### Key Functions

#### `_is_market_open(ticker: str) -> bool`
```python
stock = yf.Ticker(ticker)
market_state = stock.info.get("marketState")
return market_state == "REGULAR"
```

#### `_should_fetch_iv(ticker: str, cache: Dict) -> bool`
**Returns:** True if we should fetch new IV, False if cache is fresh

**Checks:**
1. Market open? (If no → False)
2. Time in valid range (10am-5pm)? (If no → False)
3. Cache exists? (If no → True)
4. Cache >24h old? (If yes → True)
5. Crossed fetch window? (If yes → True, else False)

#### `_get_nearest_expirations(expirations: List[str]) -> List[str]`
**Returns:** Up to 3 expirations ≤45 DTE (or single nearest if none ≤45)

**Logic:**
1. Filter to DTE ≤45
2. Sort by DTE ascending
3. Take first 3
4. If none ≤45, take single nearest

#### `get_current_iv(ticker: str) -> Optional[Dict]`
**Returns:** IV data dict or None

**Flow:**
1. Load cache
2. Check if should fetch (via `_should_fetch_iv`)
3. If no → return cached data
4. If yes → fetch new data:
   - Get current price
   - Get expirations (via `_get_nearest_expirations`)
   - For each expiration: get ATM put + call IV
   - Average all IVs
   - Save to cache
5. On error → return cached data (never corrupt cache)

---

## Display Integration

### Batch Analysis Header

**Shows most recent fetch time:**
```
EARNINGS CONTAINMENT ANALYZER - v2.6
Lookback: 24 quarters (~6 years)
Current IV (fetched: 2:15 PM ET)
===========================================================================
```

**Logic:**
- Scans all cache entries
- Finds most recent `fetched_at` timestamp
- Formats as "2:15 PM ET" or "10:30 AM ET"
- If no cache → shows generic message

**Note:** Batch of multiple tickers may have different fetch times. Header shows the most recent across all tickers.

---

## Error Handling

### Fetch Failures

**Causes:**
- yfinance API timeout
- No options data available
- Malformed response
- Network error

**Handling:**
```python
try:
    # Attempt to fetch new IV
    iv_data = fetch_iv_from_yfinance(ticker)
    cache[ticker] = iv_data
    save_cache(cache)
    return iv_data
except:
    # On ANY error, return cached data
    return cache.get(ticker)
```

**Principle:** Never corrupt cache. Better to use stale data than no data.

### Missing Cache

**If ticker not in cache AND fetch fails:**
- Return None
- Batch analysis shows "N/A" for IV columns
- Analysis continues without IV data

---

## Edge Cases

### 1. Thinly Traded Stocks
**Issue:** No weekly options, only monthlies >45 DTE

**Example:** [60d, 90d, 120d]

**Solution:** Use single nearest (60d)

**Impact:** IV may be less relevant for near-term, but better than no data.

---

### 2. Early Close Days
**Issue:** NYSE closes at 1pm on ~3 days per year

**Current behavior:** Will fetch until 1pm, then use cache

**Risk:** Low - only affects Window 2+ on those days

**Decision:** Not worth special handling. Accept this limitation.

---

### 3. After-Hours Fetching
**Issue:** What if someone runs analysis at 6pm?

**Behavior:** `marketState` will be "CLOSED" → use cache

**Result:** No fetch attempted, no corruption risk

---

### 4. Weekend/Holiday Runs
**Issue:** Running batch analysis on Saturday

**Behavior:** `marketState` will be "CLOSED" → use cache

**Result:** Shows IV from last trading day (Friday)

**Display:** "Current IV (fetched: Friday 3:45 PM ET)"

---

### 5. Parallel Processing
**Issue:** Multiple threads fetching IV for same ticker

**Behavior:** Last thread to finish wins (overwrites cache)

**Risk:** Low - threads will likely fetch same IV data

**Mitigation:** Time-gated caching reduces collision window

---

## Performance Characteristics

### Fetch Frequency

**Single ticker, single day:**
- Minimum: 1 fetch (if only run once)
- Maximum: 4 fetches (once per window)
- Typical: 1-2 fetches (morning + afternoon)

**Batch of 38 tickers, single day:**
- Minimum: 38 fetches (if all in same window)
- Maximum: 152 fetches (38 × 4 windows)
- Typical: 38-76 fetches (1-2 windows used)

### API Rate Limits

**yfinance has no explicit rate limits** but excessive requests may get throttled.

**Our strategy:**
- Maximum 4 fetches per ticker per day
- Minimum 2-hour gap between fetches for same ticker
- Parallel mode limited to 4 workers
- Natural throttling via REQUEST_DELAY (0.5s between tickers)

**Conclusion:** Very unlikely to hit any rate limits.

---

## Testing Strategy

### Test 1: Market Closed (Weekend)
```python
results = batch_analyze(["AAPL", "MSFT"], fetch_iv=True)
```

**Expected:**
- No fetch attempts
- Uses cached IV
- Header shows last fetch time (e.g., "Friday 3:45 PM ET")

### Test 2: Market Open (After 10am)
```python
results = batch_analyze(["AAPL", "MSFT"], fetch_iv=True)
```

**Expected:**
- Fetches new IV if crossed window threshold
- Updates cache
- Header shows current fetch time (e.g., "10:15 AM ET")

### Test 3: Multiple Runs Same Window
```python
# Run 1 at 10:15 AM
results1 = batch_analyze(["AAPL"], fetch_iv=True)

# Run 2 at 10:30 AM
results2 = batch_analyze(["AAPL"], fetch_iv=True)
```

**Expected:**
- Run 1: Fetches new IV
- Run 2: Uses cache (same window)
- Both show same fetch time

### Test 4: Cross Window Threshold
```python
# Run 1 at 11:45 AM (Window 1)
results1 = batch_analyze(["AAPL"], fetch_iv=True)

# Run 2 at 12:15 PM (Window 2)
results2 = batch_analyze(["AAPL"], fetch_iv=True)
```

**Expected:**
- Run 1: Fetches IV, shows "11:45 AM"
- Run 2: Fetches new IV (crossed window), shows "12:15 PM"

### Test 5: Cache Structure
```python
import json
with open("cache/iv_cache.json") as f:
    cache = json.load(f)
    print(json.dumps(cache["AAPL"], indent=2))
```

**Expected:**
```json
{
  "iv": 26.5,
  "dte": 7,
  "expiration": "2025-10-18",
  "expirations_used": [
    "2025-10-18",
    "2025-10-25",
    "2025-11-01"
  ],
  "fetched_at": "2025-10-11T14:23:15-04:00"
}
```

---

## Future Enhancements (Not Implemented)

### 1. Per-Ticker Fetch Times in Display
Currently header shows most recent fetch across all tickers.

**Possible enhancement:** Show fetch time per ticker
```
Ticker  HVol% CurIV%(2:15p) IVPrem | ...
  AAPL     27     28(2:15p)    +4%  | ...
  MSFT     24     26(10:15a)  +16%  | ...
```

**Decision:** Not worth the clutter. Global timestamp is sufficient.

---

### 2. IV Rank / IV Percentile
Calculate where current IV sits relative to historical range.

**Not implemented because:**
- Requires storing historical IV data (more complexity)
- Current IV premium vs HVol already serves this purpose
- Phase 2 priority, not urgent

---

### 3. Volatility Smile/Skew Detection
Analyze IV across strikes to detect skew.

**Not implemented because:**
- Most users don't need this granularity
- Adds significant complexity
- Phase 3 feature at best

---

## Conclusion

**Strengths:**
- Simple market detection (just check `marketState`)
- Time-gated caching prevents over-fetching
- Multi-expiration averaging improves accuracy
- Robust fallbacks handle all edge cases
- No external dependencies (uses yfinance only)

**Limitations:**
- No special handling for early close days (~3/year)
- Global fetch time in header (not per-ticker)
- No historical IV tracking (yet)

**Overall:** Solid, production-ready IV fetching with good balance of freshness, efficiency, and reliability.

---

**Version:** 2.6  
**Last Updated:** October 11, 2025  
**Status:** Production-ready, tested on weekends, ready for Monday market hours testing