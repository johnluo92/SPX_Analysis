# IV DATA CACHING STRATEGY

## Problem Statement
Yahoo Finance IV data becomes unreliable/empty after market hours. Running the analyzer after hours was:
1. Overwriting good cached IV with None/empty values
2. Breaking output tables (missing IV columns)
3. Breaking Plotly graphs (empty due to missing data)

## Solution: Smart IV Caching

### Cache Location
`cache/iv_cache.json`

### Caching Logic

#### 1. Cache Structure
```json
{
  "AAPL": {
    "iv": 26.5,
    "dte": 45,
    "strike": 150.0,
    "expiration": "2025-11-22",
    "fetched_at": "2025-10-09T14:30:22",
    "market_date": "2025-10-09"
  }
}
```

#### 2. Fetch Strategy
```
IF ticker in cache AND cache age < 24 hours:
    → Return cached data (fast, reliable)

ELSE IF ticker in cache AND cache is stale AND after market hours:
    → Return stale cache (better than None/empty)

ELSE IF ticker in cache AND cache is stale AND during market hours:
    → Fetch fresh data, update cache

ELSE (no cache):
    → Fetch fresh data if possible
    → If fetch fails, return None (no data corruption)
```

#### 3. Market Hours Check
Simple heuristic:
- Monday-Friday only
- Between 9 AM - 4 PM (rough US market hours)
- TODO: Add proper timezone handling with `pytz`

### Key Protections

#### Never Overwrite Good Data with Bad
```python
# Before caching, validate the fetched data
if iv_pct and iv_pct > 0:
    # Good data - cache it
    iv_cache[ticker] = {...}
else:
    # Bad/empty data - keep existing cache
    return cached_data_if_exists
```

#### Graceful Degradation
If NO valid IV data (cached or fresh):
1. Output table shows HVol only (no IV columns)
2. Pattern analysis continues (uses HVol)
3. User gets warning: "IV data unavailable (after hours)"
4. Plotly graphs still render (just without IV context)

### Benefits

1. **Run anytime** - after hours, weekends, doesn't matter
2. **No data corruption** - stale cache > no cache
3. **Fast** - 24-hour cache = fewer API calls
4. **Resilient** - handles fetch failures gracefully

### Cache Maintenance

#### Manual Refresh (if needed)
```bash
# Delete cache to force fresh fetch during market hours
rm cache/iv_cache.json
```

#### Automatic Expiry
Cache automatically expires after 24 hours during market hours.

### Future Enhancements

1. **Timezone awareness** - Use `pytz` for accurate market hours
2. **Extended hours** - Consider pre-market (4 AM - 9:30 AM ET)
3. **Cache by expiration** - Store multiple DTE options
4. **Backup API** - Fallback to alternative IV source if Yahoo fails

---

## Testing After Hours

To verify the fix works:

1. **Run during market hours** - should fetch fresh IV
2. **Run after hours** - should use cached IV (no corruption)
3. **Delete cache + run after hours** - should show "IV unavailable" warning but still produce results

Expected behavior:
- ✅ Output table shows results (with or without IV)
- ✅ Plotly graphs render (with or without IV context)
- ✅ No crashes, no empty results
- ✅ Cache never overwritten with None/empty values

---

**Last Updated:** October 2025  
**Status:** Implemented and tested