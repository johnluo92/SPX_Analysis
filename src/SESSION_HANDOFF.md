# VIX-Based SPX Options Backtesting - Session Handoff

**Date:** October 17, 2025  
**Session Summary:** Built initial backtest engine, debugged pricing issues, discovered strategy needs professional parameters

---

## 🎯 Project Goal

Build a **surgical, repeatable VIX-based SPX options backtesting engine** for premium collection strategies with:
- No predictions, just historical probabilities
- VIX regime-based position sizing
- Mechanical entry/exit rules
- Visual decision support tools (cone charts)

---

## 📁 Current File Structure

```
SPX_Analysis/
└── src/
    ├── UnifiedDataFetcher.py          ✅ DONE - Fetches SPX/VIX data
    ├── config.py                       ⚠️  NEEDS UPDATE - See below
    ├── backtest_engine.py              ⚠️  NEEDS MAJOR REFACTOR
    ├── visualizer.py                   ❌ NOT STARTED
    ├── daily_decision.py               ❌ NOT STARTED
    │
    ├── data/                           📂 Cache directory
    └── results/                        📂 Output directory
```

---

## ✅ What's Working

### 1. **UnifiedDataFetcher.py**
- Fetches SPX OHLC from Yahoo Finance
- Fetches VIX from Yahoo/FRED
- Aligns data properly
- **Status:** Complete, no changes needed

### 2. **Basic Backtest Loop**
- Loads data correctly
- Iterates through trading days
- Tracks open positions
- Calculates equity curve
- **Status:** Core logic solid, but parameters need fixing

---

## ❌ Critical Issues Discovered

### **Issue #1: Credit Estimation is Broken**
**Problem:** Our simplified credit estimation produces unrealistic values:
- Sometimes $1.50 (hit minimum floor)
- Sometimes $237 (impossibly high)
- Can't reliably estimate option prices without proper Black-Scholes or historical data

**Current Hack:** Using fixed percentages (15-22% of spread width) with randomness

**Real Solution Needed:** 
- Implement proper Black-Scholes pricer with risk-free rate
- OR use historical options chain data (expensive: OptionMetrics, CBOE DataShop)
- OR accept rough approximation and focus on strategy testing

### **Issue #2: Strategy Loses Money with Current Parameters**
**3 DTE, 0.5-0.65 std dev:**
- Win Rate: 76-80% ✅
- Profit Factor: 0.18-0.41 ❌ (needs to be >1.0)
- Risk/Reward: 5.67:1 ❌ (terrible)

**Why it fails:**
- Credits too low ($1.50 avg) vs risk ($8.50 loss)
- 3 DTE gives no time for recovery
- 2020 crash destroys the strategy
- No trend filters = selling into falling knife

**What professionals do differently:**
- Trade 14-45 DTE (not 3)
- Collect 15-20% of spread width in credit
- Close at 50% max profit (don't hold to expiration)
- Use 2-3x stop loss
- Add trend filters (don't sell after big down days)
- Wider spreads ($25-50, not $10)

---

## 🔧 IMMEDIATE TODO - Next Session

### **Priority 1: Refactor to Professional Parameters**

Update `config.py`:
```python
POSITION_CONFIG = {
    'dte': 14,  # CHANGE from 3
    'close_at_profit_pct': 0.50,  # NEW: Close at 50% profit
    'std_dev_short': 0.80,  # CHANGE: Go wider
    'wing_width_dollars': 25,  # CHANGE: Wider spreads
}

ENTRY_RULES = {
    'vix_min': 15,
    'trend_filter': True,  # NEW: Enable trend filter
    'no_entry_after_down_day': True,  # NEW: Don't catch falling knives
}

EXIT_RULES = {
    'hold_to_expiration': False,  # CHANGE: Take profits early
    'profit_target_pct': 0.50,  # NEW: Close at 50% max profit
    'stop_loss_multiplier': 2.5,  # CHANGE: Wider stop
}
```

### **Priority 2: Add Early Exit Logic**

In `backtest_engine.py`, add to `check_exit()`:
```python
def check_exit(self, trade, current_date, spx_price):
    # Check profit target (NEW)
    if trade.unrealized_pnl >= trade.credit_received * 0.50:
        return 'PROFIT_TARGET'
    
    # Check days in trade (NEW) 
    if trade.days_held >= trade.dte - 1:
        return 'EXPIRATION'
    
    # Existing stop loss logic...
```

### **Priority 3: Add Trend Filter**

```python
def should_enter_trade(self, date, vix, spx_history):
    # NEW: Don't sell puts after down days
    if spx_history[-1] < spx_history[-2] * 0.98:  # 2% down
        return False
    
    # Existing checks...
```

### **Priority 4: Fix Credit Estimation**

Replace the broken `estimate_credit()` with simple fixed logic:
```python
def calculate_credit(self, wing_width, vix, dte):
    """Simple credit based on industry standards."""
    # Base: 15-20% of width
    if dte >= 14:
        base_pct = 0.18  # Better credit for longer DTE
    else:
        base_pct = 0.12
    
    # Scale by VIX
    vix_multiplier = min(vix / 20, 2.0)  # Cap at 2x
    
    credit = wing_width * base_pct * vix_multiplier
    return max(credit, wing_width * 0.10)
```

---

## 📊 Files to Build - Visualization Phase

### **1. visualizer.py**
**Purpose:** Create cone charts and performance plots

**Key Features:**
- Cone chart showing VIX-implied ranges
- Equity curve with drawdowns
- Win rate by VIX regime
- P&L distribution histogram

**Libraries:** matplotlib, seaborn

### **2. daily_decision.py**
**Purpose:** "Should I trade TODAY?" script

**Inputs:**
- Current SPX price
- Current VIX
- Recent market moves

**Outputs:**
- Trade recommendation (YES/NO)
- Suggested structure (BPS/IC)
- Strike recommendations
- Risk visualization (cone)

---

## 🧪 Testing Checklist - After Refactor

Run with new parameters and verify:
- [ ] Credits are 15-25% of spread width ($3.75-$6.25 for $25 spread)
- [ ] Win rate: 65-75%
- [ ] Profit factor: >1.0 (ideally 1.5-2.5)
- [ ] Sharpe ratio: >1.0
- [ ] Max drawdown: <20%
- [ ] Positive returns in both NORMAL and ELEVATED regimes

**Expected Results with Pro Parameters:**
- Total Return: 15-40% over 5 years (realistic)
- Sharpe: 1.5-2.5
- Max DD: 8-15%
- Both regimes should be profitable

---

## 💡 Key Learnings This Session

1. **3 DTE is too aggressive** - Professionals use 14-45 DTE
2. **Holding to expiration is suboptimal** - Take profits at 50%
3. **Fixed $10 spreads are too tight** - Use $25-50 for better credit
4. **Need trend filters** - Don't sell into crashes
5. **Credit estimation is hard** - Consider historical data or accept approximation
6. **0.5 std dev is too close** - Go to 0.8-1.0 for better probability

---

## 🔍 Research References

Key insights from web search:
- Industry standard: 15-20% of spread width in credit
- Optimal DTE: 14-45 days for theta decay sweet spot
- Close at 50% max profit for better risk-adjusted returns
- VIX regimes matter: <15 stand aside, 15-25 trade, >25 be selective
- Ultra-short DTE (0-5) requires very far OTM strikes

---

## 🚀 Long-Term Roadmap

**Phase 1 (CURRENT):** Core backtest engine with realistic parameters  
**Phase 2:** Visualization suite (cones, performance charts)  
**Phase 3:** Daily decision tool  
**Phase 4:** Walk-forward optimization  
**Phase 5:** Live trading integration (paper trading first)

---

## 📝 Notes for Next Claude

**What worked well:**
- Modular architecture (fetcher, engine, config separate)
- VIX regime detection
- Basic P&L calculation

**What needs work:**
- Credit estimation (currently using rough approximation)
- Parameter calibration (switching to pro parameters)
- Need visualizations
- Need trend filters

**Code Quality:**
- Clean, well-documented
- Good separation of concerns
- Easy to modify parameters

**User Preferences:**
- Keep it simple and surgical
- No over-engineering
- Focus on decision support, not prediction
- User wants to trade with "the big boys" using sound premises

---

## 🎯 Success Criteria

The backtest is ready when:
1. ✅ Profit factor > 1.0 in backtests
2. ✅ Both VIX regimes are profitable
3. ✅ Drawdowns are reasonable (<20%)
4. ✅ Cone visualization helps with daily decisions
5. ✅ User can run `python daily_decision.py` and get clear trade recommendations

---

**Last Updated:** Oct 17, 2025  
**Next Session:** Start with Priority 1 (refactor to 14 DTE with pro parameters)