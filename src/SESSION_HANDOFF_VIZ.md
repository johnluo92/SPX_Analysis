# VIX Cone Visualizer - Session Handoff

**Date:** October 17, 2025  
**Status:** Core cone chart working, adding context panels

---

## 🎯 What We Built This Session

### ✅ Working: `visualizer.py`
**Two-panel interactive chart using Plotly:**

**Panel 1: SPX VIX-Implied Probability Cone**
- Historical SPX (30 days lookback)
- 14-day forward cone (1σ, 2σ)
- VIX scenario cones (VIX +10, VIX -5)
- Suggested strike reference box
- Toggle for scenarios

**Panel 2: Volatility Landscape**
- Historical VIX (30 days)
- VIX regime lines (15, 25, 35)
- Projected VIX scenarios

**Dependencies:** plotly, scipy, pandas, numpy

**Usage:**
```bash
python visualizer.py
```

---

## 🎯 Next Session Goals

### **Priority 1: Add IV vs Realized Vol Chart**

**What user wants:**
- Time series showing: VIX vs what volatility ACTUALLY realized 30 days later
- See the "volatility risk premium" visually
- Understand: Does high IV typically fade or expand?

**Implementation:**
1. For each historical date, calculate:
   - VIX on that date
   - Realized vol 30 days LATER (using actual SPX moves)
2. Plot both on same chart with light shading between
3. Add context box showing:
   ```
   Current VIX: 25.3
   Historical pattern when VIX was 24-26:
     → Realized vol averaged 19.2%
     → Premium captured 78% of time
     → But expanded 22% of time
   ```
4. **CRITICAL:** Don't predict! Just show historical frequency

**Realized Vol Calculation:**
```python
def calculate_realized_vol(prices: pd.Series, window: int = 30) -> float:
    """Calculate realized volatility from price series."""
    returns = np.log(prices / prices.shift(1))
    realized_vol = returns.std() * np.sqrt(252) * 100  # Annualized %
    return realized_vol
```

**Add as Panel 3** (below volatility landscape)

---

## 📊 User Philosophy - IMPORTANT

**What user wants:**
- **Informative, not predictive** - Show data, let them decide
- **Context, not recommendations** - No "sell this strike"
- **Clean visuals** - Toggle complexity, default to simple
- **Signal, not noise** - Every element must answer a decision question

**What user is trading:**
- 1-2 ICs/spreads per expiration (14 DTE)
- LEAPS (400+ DTE) for long exposure during high vol
- Wants to see: volatility environment, strike safety, edge opportunity

**Key insight from research:**
- From 2010-2019, 86% of time IV > realized vol
- This is the "volatility risk premium" - the edge for sellers
- User wants to SEE this edge, not be told about it

---

## 🚫 What NOT to Build

- ❌ Explicit trade recommendations ("sell the 6295 put")
- ❌ Prediction statements ("vol will fade")
- ❌ Over-complicated models (Monte Carlo, ML)
- ❌ Features that duplicate ToS (strike calculators)
- ❌ Anything that adds noise without decision value

---

## 🔧 Current File Structure

```
SPX_Analysis/src/
├── visualizer.py              ✅ WORKING - Cone chart
├── UnifiedDataFetcher.py      ✅ DONE - Data fetching
├── config_PROFESSIONAL.py     ⚠️  Exists but not used by visualizer
├── backtest_engine.py         ⚠️  Exists but abandoned (over-optimized)
└── spx_cone_chart.html        📊 Output file
```

---

## 💻 Code to Add Next Session

### **In visualizer.py, add method:**

```python
def calculate_iv_rv_spread(self, lookback_days: int = 60) -> pd.DataFrame:
    """
    Calculate historical IV vs subsequent realized vol.
    
    Returns DataFrame with:
    - date: Historical date
    - vix: VIX on that date
    - realized_30d: Actual 30-day realized vol
    - spread: IV - RV
    """
    # Fetch extended data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 60)
    
    spx_data = self.fetcher.fetch_spx(...)
    vix_data = self.fetcher.fetch_vix(...)
    
    results = []
    
    for i in range(len(vix_data) - 30):
        date = vix_data.index[i]
        vix_value = vix_data.iloc[i]
        
        # Get SPX prices from this date + 30 days forward
        future_prices = spx_data['Close'].iloc[i:i+30]
        
        if len(future_prices) >= 30:
            # Calculate realized vol
            returns = np.log(future_prices / future_prices.shift(1))
            realized = returns.std() * np.sqrt(252) * 100
            
            results.append({
                'date': date,
                'vix': vix_value,
                'realized_30d': realized,
                'spread': vix_value - realized
            })
    
    return pd.DataFrame(results)
```

### **Add Panel 3 to plot_decision_chart():**

```python
# After Panel 2 (Volatility Landscape), add:

# Calculate IV vs RV
iv_rv_data = self.calculate_iv_rv_spread(lookback_days=60)

# Add trace for VIX
fig.add_trace(go.Scatter(
    x=iv_rv_data['date'],
    y=iv_rv_data['vix'],
    name='Implied Vol (VIX)',
    line=dict(color='blue', width=2),
), row=3, col=1)

# Add trace for realized
fig.add_trace(go.Scatter(
    x=iv_rv_data['date'],
    y=iv_rv_data['realized_30d'],
    name='Realized Vol (30d forward)',
    line=dict(color='orange', width=2),
    fill='tonexty',
    fillcolor='rgba(255, 200, 100, 0.2)',
), row=3, col=1)

# Add current context annotation
current_vix = vix_data.iloc[-1]
similar_periods = iv_rv_data[
    (iv_rv_data['vix'] >= current_vix - 1) & 
    (iv_rv_data['vix'] <= current_vix + 1)
]

avg_realized = similar_periods['realized_30d'].mean()
premium_pct = ((similar_periods['spread'] > 0).sum() / len(similar_periods)) * 100

fig.add_annotation(
    text=(
        f"Current VIX: {current_vix:.1f}<br>"
        f"When VIX was {current_vix-1:.0f}-{current_vix+1:.0f}:<br>"
        f"  Avg realized: {avg_realized:.1f}%<br>"
        f"  IV > RV: {premium_pct:.0f}% of time<br>"
        f"  IV < RV: {100-premium_pct:.0f}% of time"
    ),
    xref='x domain', yref='y domain',
    x=0.02, y=0.98,
    row=3, col=1
)
```

### **Update make_subplots() call:**

```python
fig = make_subplots(
    rows=3, cols=1,  # Changed from 2 to 3
    row_heights=[0.45, 0.25, 0.30],  # Adjusted heights
    subplot_titles=(...),  # Add title for Panel 3
)
```

---

## 🧪 Testing Checklist

After adding IV vs RV panel:
- [ ] Chart shows VIX and realized vol clearly
- [ ] Spread shading is subtle (not overwhelming)
- [ ] Context box shows current VIX regime stats
- [ ] Text is informative, not predictive
- [ ] No lag/performance issues with calculation
- [ ] Works with different date ranges

---

## 📚 Research References

Key findings from this session:
- IV-RV spread historically positive - IV tends to exceed realized vol
- 2010-2019: 86% of time IV > RV
- Volatility risk premium is one of most reliable market aspects
- Selling premium works when IV runs ahead of realized
- But calm periods can end suddenly - risk management critical

---

## 🎯 Future Enhancements (Not Next Session)

**If user requests:**
- Position tracker for open trades
- Alert system for vol spikes
- Multi-DTE comparison view
- Historical trough recovery analysis (only useful during crashes)

**Don't build unless asked:**
- Backtest improvements (abandoned that rabbit hole)
- Strike recommendations
- Monte Carlo simulations
- Anything predictive

---

## 💡 Key Learnings

1. **User doesn't want a perfect backtest** - wants decision support tools
2. **Show, don't tell** - visualizations over recommendations
3. **Keep it clean** - toggle complexity, default simple
4. **Research-backed** - but present as context, not prediction
5. **Informative, not prescriptive** - let user make decisions

---

**Next Claude:** Read this, then implement Panel 3 (IV vs RV chart). Test it, show user, iterate based on feedback. Keep it clean and informative!