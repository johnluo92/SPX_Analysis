# SESSION HANDOFF - Next Coding Session

## CURRENT STATUS: Core Features Complete ✅

### What Works Now
1. **✅ Edge Count Showing** - Pattern column displays `[X edges]` correctly
2. **✅ 45D Columns Added** - Both `45D%` and `45Break` now visible in table
3. **✅ IV Caching Fixed** - 24-hour cache, no after-hours corruption
4. **✅ Pattern Detection** - All signals (IC45, IC90, BIAS↑/↓) working accurately
5. **✅ Clean Output** - Removed redundant "90d" labels from bias reasons
6. **✅ Parallel Processing** - 4 workers, fast batch analysis

### Current Output Format
```
Ticker  HVol% CurIV% IVPrem |  45D% 45Break  90D%  90Bias 90Break 90Drift  |  Pattern
  AAPL     27     28    +4% |    50    8:4↑    78      57     3:2   +5.4%   |  IC90 + BIAS↑ (3:2↑ breaks, +5.4% drift) [2 edges]
  MSFT     24     28   +16% |    75    5:1↑    78      65    5:0↑   +6.2%   |  IC90⚠↑ + IC45⚠↑ + BIAS↑ (5:0↑ breaks, +6.2% drift) [3 edges]
```

---

## Edge Count Logic (Confirmed Correct ✅)

**Each edge = one independent PATTERN** (not individual metrics)

Example: MSFT `[3 edges]`
1. **IC90⚠↑** = Edge #1 (mean reversion at 90d)
2. **IC45⚠↑** = Edge #2 (mean reversion at 45d)
3. **BIAS↑** = Edge #3 (directional pattern)

Breaks and drift are **reasons supporting the BIAS edge**, not separate edges.

**No fix needed** - edge counting is working as designed.

---

## REMAINING WORK - Phase 2: Execution Layer

### Priority 1: Position Sizing Calculator (HIGH VALUE)
**Goal:** Remove discretion from strategy selection, apply it only to sizing.

**What to build:**
```python
def calculate_position_size(
    account_size: float,
    risk_tolerance: float,  # e.g., 0.02 = 2% risk per trade
    ticker: str,
    strategy: str,  # "IC90", "IC45", "BIAS↑"
    below_55ma: bool,  # Is stock below 55-month MA?
    iv_premium: float,  # IV premium vs HVol (e.g., +15%)
    edge_count: int  # Number of independent edges
) -> Dict:
    """
    Calculate position size based on multiple conviction factors
    
    Returns:
        {
            'base_size': float,  # Base position size
            'adjusted_size': float,  # After conviction adjustments
            'contracts': int,  # Number of contracts to trade
            'capital_required': float,  # Estimated capital needed
            'conviction_multiplier': float  # 0.5x to 2.0x
        }
    """
```

**Conviction factors:**
- Below 55-month MA → +50% size (1.5x)
- IV ≥15% elevated → Full size (1.0x)
- IV ≤-15% depressed → Reduce by 50% (0.5x)
- Multi-edge (2-3 edges) → +25% size (1.25x)

**Example output:**
```
Account: $50,000
Base risk: 2% per trade = $1,000
Strategy: IC90 (containment play)

Conviction adjustments:
- Below 55-MA: +50% (bullish setup)
- IV elevated +18%: Full size
- 2 edges detected: +25%
→ Final multiplier: 1.75x

Position size: $1,750 (1.75% of account)
Estimated contracts: 3-4 (depending on strike width)
Capital required: ~$5,250 (3x margin)
```

---

### Priority 2: Trade Specification Tool
**Goal:** Eliminate execution hesitation - exact strikes, exact DTE, exact entry targets.

**What to build:**
```python
def generate_trade_spec(
    ticker: str,
    current_price: float,
    strategy: str,  # "IC90", "IC45", "BIAS↑"
    hvol: float,  # Historical volatility
    strike_width: float  # From backtest
) -> Dict:
    """
    Generate exact trade specifications
    
    Returns:
        {
            'strategy_type': str,  # "Iron Condor", "Bull Put Spread", etc.
            'dte_target': int,  # Days to expiration
            'strikes': {
                'put_sell': float,
                'put_buy': float,
                'call_sell': float,
                'call_buy': float
            },
            'entry_targets': {
                'min_credit': float,  # Minimum acceptable credit
                'ideal_credit': float  # Target credit
            },
            'position_delta': float,  # Estimated delta
            'max_loss': float,  # Max loss per contract
            'notes': str  # Special considerations
        }
    """
```

**Example output:**
```
Ticker: MSFT
Strategy: IC90 [3 edges]
Current Price: $420
HVol: 24% → Strike width: ±$50 (90d)

TRADE SPEC:
- Type: Iron Condor
- DTE: 90 days (targeting ~Dec 17 expiration)
- Strikes:
  * Sell 370 Put / Buy 360 Put (width: $10)
  * Sell 470 Call / Buy 480 Call (width: $10)
- Entry Target: $2.50-$3.00 credit per spread
- Max Loss: $7.00-$7.50 per spread
- Risk/Reward: ~3:1

Notes:
- ⚠↑ Upward break bias detected - consider widening call side
- IV elevated +16% - premium is rich, favorable entry
- 3 edges aligned - high conviction setup
```

---

### Priority 3: Execution Tracking & Feedback Loop
**Goal:** Build conviction through evidence, not emotion.

**What to track:**
```python
trade_log = {
    'ticker': str,
    'date_entered': datetime,
    'strategy_recommended': str,  # What the system said
    'strategy_executed': str,  # What you actually did
    'deviation_reason': str,  # Why you deviated (if at all)
    'size_recommended': float,
    'size_executed': float,
    'outcome': {
        'pnl': float,
        'roi': float,
        'days_held': int,
        'closed_reason': str  # "max profit", "stop loss", "expiration"
    },
    'system_accuracy': bool  # Did the pattern play out as expected?
}
```

**Metrics to calculate:**
- **Follow Rate**: % of signals you actually traded
- **Override Accuracy**: When you deviated, were you right?
- **P&L Attribution**: System trades vs discretionary trades
- **Pattern Win Rate**: IC90 win rate vs IC45 vs BIAS plays

**Goal:** After 20-30 trades, you'll have data showing:
- "I followed 80% of IC90 signals → 75% win rate"
- "I overrode 5 signals → 40% win rate"
- **→ Trust the system more, not less**

---

## FILES MODIFIED THIS SESSION

### 1. `earnings_analyzer/calculations/strategy.py` ✅
**Changes:**
- Added edge count tracking and display
- Removed "90d" labels from bias reasons
- Enhanced docstring explaining 90d-only bias logic

**Status:** Complete and working

---

### 2. `earnings_analyzer/analysis/batch.py` ✅
**Changes:**
- Added `45D%` and `45Break` columns to results table
- Widened table separators (110 → 145 characters)
- Preserved edge count in strategy display column
- Fixed IV column handling (graceful degradation if no data)

**Status:** Complete and working

---

### 3. `earnings_analyzer/data_sources/yahoo_finance.py` ✅
**Changes:** (from previous session)
- Complete IV caching rewrite
- 24-hour cache with staleness protection
- Market hours detection
- Backward compatibility

**Status:** Complete and working

---

## TESTING CHECKLIST

After any changes, run:
```python
from earnings_analyzer import batch_analyze
results = batch_analyze(["AAPL", "MSFT", "JPM", "GS", "AXP"], lookback_quarters=24)
```

**Verify:**
- ✅ Edge count shows: `[X edges]`
- ✅ 45D% and 45Break columns present
- ✅ 90D columns still showing
- ✅ IV columns display (if cached data exists)
- ✅ Pattern legend matches output
- ✅ No crashes, no missing data
- ✅ Table formatting readable (not too wide)

---

## OPEN QUESTIONS FOR DISCUSSION

### Strategy Refinement
1. **Conflicting signals:** What if IC90⚠↑ (upward break risk) but BIAS↓ (downward drift)?
2. **IC45 + IC90:** When both qualify, layer them or pick one?
3. **Asymmetric IC adjustment:** If IC90⚠↑, widen call side by how much?
4. **Below 55-MA threshold:** Should this increase position size for ALL strategies or just BIAS plays?

### Position Sizing Philosophy
5. **Base risk per trade:** 1% (conservative) vs 2% (moderate) vs 3% (aggressive)?
6. **Max single position:** Cap at 5% of account even if conviction is high?
7. **Portfolio heat:** Max total exposure across all positions (e.g., 20% total)?
8. **Correlation risk:** Multiple bank stocks = treat as one position for sizing?

### Execution Rules
9. **Entry discipline:** If IV isn't elevated, skip IC even if pattern qualifies?
10. **Exit rules:** Take profit at 50%? 75%? Hold to expiration?
11. **Adjustment triggers:** When to roll vs close vs hold?
12. **Stop loss:** Close at 2x credit received? 3x?

---

## NEXT SESSION PRIORITIES

### Immediate (Start Here)
1. **Position Sizing Calculator** - Build the core function
2. **Test with real account size** - $50k example
3. **Conviction multiplier logic** - Below 55-MA, IV premium, edge count

### Soon After
4. **Trade Specification Tool** - Generate exact strikes/DTE
5. **Output format** - Human-readable trade specs
6. **Integration** - Add to batch_analyze output (optional flag)

### Future (Phase 3)
7. **Execution Tracking** - Build trade log structure
8. **Performance Metrics** - Calculate follow rate, override accuracy
9. **Feedback Loop** - Use data to refine thresholds

---

## KEY DECISIONS MADE THIS SESSION

1. **✅ Edge count = independent patterns** - IC90 + IC45 + BIAS = 3 edges
2. **✅ Breaks/drift = supporting evidence** - Not counted as separate edges
3. **✅ 45D columns restored** - Users need to see both timeframes
4. **✅ Table width = 145 chars** - Fits all columns without wrapping
5. **✅ Strategy string unchanged** - Preserved exactly as `strategy.py` returns it

---

## ARTIFACTS READY FOR USE

### 1. `STRATEGY_GUIDE.md` (Complete Strategy Reference)
- 52KB comprehensive guide
- All patterns, thresholds, timeframes explained
- Position construction examples
- Common questions answered
- **Use in strategy-only discussion session** (no code context needed)

### 2. `IV_CACHE_STRATEGY.md` (Technical Doc)
- IV caching logic explained
- Cache structure and fetch strategy
- Protections against corruption
- **Reference for future debugging**

### 3. `PROJECT_MANIFESTO.md` (The Vision)
- Goal: Financial independence through systematic trading
- Problem: Discretion overriding backtests
- Solution: System picks WHAT, you pick HOW MUCH
- Execution rules and success metrics
- **Read before every session to stay aligned**

---

## NOTES

- **Core analysis complete** - Pattern detection is rock solid
- **Output is clean** - All metrics visible and formatted well
- **IV caching stable** - No corruption, smart fallbacks
- **Parallel processing working** - 10-14 tickers/sec
- **Ready for Phase 2** - Time to build execution tools

---

**Last Updated:** October 10, 2025 (Session 2 complete)  
**Status:** Foundation complete, execution layer next  
**Current Version:** v2.5 (HVol Backtest)  
**Next Goal:** Position sizing calculator (remove discretion from strategy selection)

---

## SESSION STARTUP CHECKLIST

When you start the next session:

1. **Read this handoff note** - Understand current state
2. **Review open questions** - Pick 2-3 to discuss before coding
3. **Pick one priority** - Don't try to build everything at once
4. **Test incrementally** - After each function, test it
5. **Update handoff note** - Document what you built

**Remember:** Slow is smooth, smooth is fast. One feature at a time.

---

## QUICK REFERENCE: Current Tool Capabilities

**What the tool does NOW:**
- ✅ Fetches earnings dates (Alpha Vantage, cached)
- ✅ Fetches price data (Yahoo Finance)
- ✅ Calculates historical volatility (30-day lookback)
- ✅ Determines strike width (volatility-adjusted)
- ✅ Backtests 45d and 90d containment
- ✅ Detects directional bias (90d only)
- ✅ Fetches current IV (45-day DTE target)
- ✅ Calculates IV premium vs HVol
- ✅ Recommends strategy (IC45/IC90/BIAS/SKIP)
- ✅ Counts independent edges
- ✅ Batch processes multiple tickers (serial or parallel)
- ✅ Exports results to CSV/JSON

**What the tool does NOT do yet:**
- ❌ Calculate position sizes
- ❌ Generate exact trade specifications
- ❌ Track trade execution vs recommendations
- ❌ Measure system follow rate and accuracy
- ❌ Adjust for 55-month MA (below = increase size)
- ❌ Provide entry/exit rules
- ❌ Suggest adjustments for asymmetric risk

**Next step:** Build the missing pieces (position sizing first).