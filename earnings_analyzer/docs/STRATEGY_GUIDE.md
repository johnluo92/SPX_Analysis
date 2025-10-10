# EARNINGS CONTAINMENT ANALYZER - STRATEGY GUIDE
**Version:** 2.5  
**Last Updated:** October 2025  
**Purpose:** Complete reference for understanding analysis output and making trading decisions

---

## TABLE OF CONTENTS
1. [Core Philosophy](#core-philosophy)
2. [Pattern Types Explained](#pattern-types-explained)
3. [Timeframe Logic](#timeframe-logic)
4. [Threshold Reference](#threshold-reference)
5. [Reading The Output](#reading-the-output)
6. [Position Construction Examples](#position-construction-examples)
7. [Common Questions](#common-questions)

---

## CORE PHILOSOPHY

### The Problem We're Solving
Most traders approach earnings plays with emotion and recency bias. This tool uses 6+ years of historical data (24 quarters) to identify **systematic edges** that persist across multiple earnings cycles.

### Two Types of Edges
1. **Mean Reversion (Containment):** Stock tends to stay within a predictable range post-earnings
2. **Directional Bias:** Stock consistently drifts in one direction over time

### The System's Role
- **Identifies patterns** (IC45, IC90, BIAS↑, BIAS↓)
- **Provides context** (IV elevation, break ratios, drift magnitude)
- **Does NOT decide position size** - that's discretionary based on conviction, account size, risk tolerance

---

## PATTERN TYPES EXPLAINED

### IC90 (Iron Condor - 90 Day)
**What it means:**
- Stock contained within calculated strike width in ≥70% of 90-day periods
- Post-earnings move is mean-reverting over ~3 months
- Break ratio is balanced (neither direction dominates breaks by >2:1)

**Trading implication:**
- Sell iron condor with ~90 DTE
- Expect price to stay range-bound
- Premium collection play

**Example:** `IC90 [1 edge]`

---

### IC45 (Iron Condor - 45 Day)
**What it means:**
- Stock contained within calculated strike width in ≥70% of 45-day periods
- Post-earnings move is mean-reverting over ~6 weeks
- Shorter timeframe = tighter containment needed

**Trading implication:**
- Sell iron condor with ~45 DTE
- Faster premium decay
- Less time for breakdown scenarios

**Example:** `IC45 [1 edge]`

---

### IC90 + IC45 (Dual Timeframe Containment)
**What it means:**
- Stock contained at BOTH 45-day and 90-day windows
- Strong mean-reversion tendency across multiple timeframes

**Trading implication:**
- High-probability IC candidate
- Can layer positions (sell 45d IC, manage to expiration, then sell 90d IC)
- Strongest containment signal

**Example:** `IC90 + IC45 [2 edges]`

---

### IC⚠↑ or IC⚠↓ (Asymmetric IC)
**What it means:**
- Stock meets containment threshold BUT breaks are skewed (>2:1 ratio)
- `⚠↑` = upside breaks dominate (watch upper strike)
- `⚠↓` = downside breaks dominate (watch lower strike)

**Trading implication:**
- Can still sell IC but structure it asymmetrically
- Widen the side with more breaks
- OR sell IC + add directional hedge on risky side
- NOT a balanced IC setup

**Example:** `IC90⚠↑ [1 edge]`

---

### BIAS↑ (Upward Directional Edge)
**What it means:**
- Stock shows consistent upward tendency over 90-day periods
- Triggered by one or more of:
  - **Directional bias:** ≥65% of moves are upward
  - **Break ratio:** Upside breaks ≥1.5x downside breaks
  - **Drift:** Average move is ≥+3.0% (consistent upward drift)

**Trading implication:**
- Bullish directional play (bull call spread, bull put spread, long call)
- NOT a mean-reversion candidate
- Trend-following opportunity

**Example:** `BIAS↑ (drift) [1 edge]`  
**Example:** `BIAS↑ (67% bias, 5:3↑ breaks, +4.7% drift) [1 edge]`

---

### BIAS↓ (Downward Directional Edge)
**What it means:**
- Stock shows consistent downward tendency over 90-day periods
- Triggered by one or more of:
  - **Directional bias:** ≤35% of moves are upward (65%+ downward)
  - **Break ratio:** Downside breaks ≥1.5x upside breaks
  - **Drift:** Average move is ≤-3.0% (consistent downward drift)

**Trading implication:**
- Bearish directional play (bear put spread, bear call spread, long put)
- NOT a mean-reversion candidate

**Example:** `BIAS↓ (2:4↓ breaks) [1 edge]`

---

### Combined Patterns (Multiple Edges)
**What it means:**
- Stock exhibits BOTH containment AND directional bias
- Multiple independent edges present

**Trading implication:**
- **Layer multiple position types**
- Example: `IC45 + BIAS↑ [2 edges]` → Sell IC45 + Bull Put Spread at 90d
- Example: `IC90 + IC45 + BIAS↑ [3 edges]` → Maximum opportunity for multi-leg construction

**Example:** `IC90 + BIAS↑ (drift) [2 edges]`

---

### SKIP [0 edges]
**What it means:**
- No systematic edge detected
- Containment <70% AND no directional bias

**Trading implication:**
- Pass on this ticker
- No historical pattern to exploit
- Risk/reward not favorable based on backtest

---

## TIMEFRAME LOGIC

### Why 45d and 90d?
- **45 days (~6 weeks):** Captures immediate post-earnings behavior
- **90 days (~3 months):** Full quarterly cycle, more stable patterns

### Why BIAS Uses Only 90d
**Question:** Why don't we check 45d bias?

**Answer:**
- **45d is too noisy** for reliable directional edge detection
- Need longer timeframe to distinguish trend from volatility
- 90d provides ~24 independent samples over 6 years (sufficient statistical significance)
- **IC45 = containment check, not directional check**

**Key principle:**
- Short timeframes (45d) → Containment detection
- Longer timeframes (90d) → Directional bias detection

---

## THRESHOLD REFERENCE

### Containment Threshold
- **≥70% (69.5% rounded)** = IC candidate
- Means: In 7+ out of 10 earnings, stock stayed within calculated strike width

### Strike Width Calculation
Based on historical volatility (HVol) + volatility tier multiplier:
- <25% HVol → 1.0x std deviation
- 25-35% HVol → 1.2x std deviation
- 35-45% HVol → 1.4x std deviation
- >45% HVol → 1.5x std deviation

**Formula:** `width = HVol × sqrt(DTE/365) × tier_multiplier × 100`

### Directional Bias Thresholds (90d only)
- **Upward bias:** ≥65% of moves are positive
- **Downward bias:** ≤35% of moves are positive
- **Break ratio:** ≥1.5:1 ratio + minimum 2 breaks
- **Drift:** ±3.0% average move vs reference price

### Asymmetric IC Threshold
- **Break ratio ≥2.0:1** → Triggers ⚠ warning
- **Break bias ≥70%** → Upside risk (⚠↑)
- **Break bias ≤30%** → Downside risk (⚠↓)

---

## READING THE OUTPUT

### Results Table Columns

```
Ticker  HVol% CurIV% IVPrem |  90D%  90Bias 90Break 90Drift  |  Pattern
```

#### Left Section: Volatility Context
- **HVol%:** Historical volatility (30-day lookback, annualized)
- **CurIV%:** Current implied volatility (~45 DTE ATM option)
- **IVPrem:** IV premium relative to HVol
  - **≥+15%** = Elevated (rich premium, favorable for selling)
  - **≤-15%** = Depressed (thin premium, consider skipping or reducing size)
  - **-15% to +15%** = Normal range

#### Middle Section: 90-Day Statistics
- **90D%:** Containment percentage (how often stock stayed within width)
- **90Bias:** Percentage of upward moves (50% = neutral, >65% = bullish, <35% = bearish)
- **90Break:** Break ratio with directional arrow if edge exists
  - `5:3↑` = 5 upside breaks, 3 downside breaks, upside dominant
  - `3:3` = Balanced breaks
- **90Drift:** Average percentage move from entry price
  - `+4.7%` = Consistent upward drift
  - `-2.1%` = Consistent downward drift

#### Right Section: Pattern
- **Strategy label(s)** with edge count
- **Parentheses:** Specific bias reasons (if applicable)
- **Edge count:** Number of independent 90-day patterns

### Pattern Legend (Printed After Table)
```
• IC45/IC90: Mean reversion containment windows (45-day/90-day)
• BIAS↑/BIAS↓: Directional edge (always based on 90-day thresholds)
• ⚠↑/⚠↓: Asymmetric break risk (watch for skewed movement)
• Edge count: Number of independent 90-day patterns detected
• All bias metrics (%, breaks, drift) reference 90-day analysis
```

---

## POSITION CONSTRUCTION EXAMPLES

### Example 1: Pure IC Candidate
**Output:** `PEP: IC90 + IC45 [2 edges] | 90D% = 74% | IVPrem = +32%`

**Analysis:**
- Strong containment at both timeframes
- IV elevated (+32%) = rich premium
- No directional bias

**Position:**
- Sell IC45 immediately post-earnings
- Manage to expiration or 50% profit
- Then sell IC90 for second premium collection
- Equal-width strikes (balanced structure)

---

### Example 2: IC + Directional Bias
**Output:** `FAST: IC90 + BIAS↑ (3:2↑ breaks, +4.0% drift) [2 edges]`

**Analysis:**
- 90d containment = 78% (IC candidate)
- Upward drift = +4.0% (directional edge)
- Breaks slightly favor upside

**Position:**
- Sell IC90 (capture containment edge)
- Add bull put spread at 90-120 DTE (capture drift edge)
- IC benefits from mean reversion, spread benefits from upward drift
- Two independent edges = layer two position types

---

### Example 3: Asymmetric IC
**Output:** `GS: IC90⚠↑ + IC45⚠↑ [2 edges] | 90Break = 5:2↑`

**Analysis:**
- Containment exists BUT upside breaks dominate 2.5:1
- Risk is on the upper strike

**Position:**
- Sell IC but widen call side (asymmetric structure)
- Example: If 1 std = 10%, use -8% put side / +12% call side
- OR sell IC + add long call as upside hedge
- NOT a balanced IC setup

---

### Example 4: Pure Directional
**Output:** `AXP: BIAS↑ (74% bias, 9:4↑ breaks, +5.8% drift) [1 edge]`

**Analysis:**
- No containment (43% only)
- Very strong upward bias across all metrics
- Not mean-reverting

**Position:**
- Bull call spread or bull put spread
- Do NOT sell IC (no containment edge)
- Trend-following play

---

### Example 5: Skip
**Output:** `BAC: SKIP [0 edges] | 90D% = 61%`

**Analysis:**
- Containment below 70%
- No directional bias detected

**Position:**
- Pass on this ticker
- No systematic edge to exploit
- Wait for next earnings cycle or different opportunity

---

## COMMON QUESTIONS

### Q: Why do some patterns have long descriptions in parentheses?
**A:** The parentheses show **which specific 90d thresholds triggered** the BIAS signal:
- `BIAS↑ (drift)` = Only drift threshold met
- `BIAS↑ (67% bias, 5:3↑ breaks, +4.7% drift)` = All three thresholds met (stronger signal)

This helps you understand if it's a single-factor edge or multi-factor confluence.

---

### Q: What if IC90 and BIAS↑ conflict?
**A:** They don't conflict - they're **independent edges**:
- IC90 = Mean reversion within 90 days
- BIAS↑ = Consistent upward drift over time

**Interpretation:** Stock tends to stay range-bound BUT has a slight upward drift bias. You can trade both edges with layered positions (IC for containment + directional spread for drift).

---

### Q: Should I always trade when edge count is high?
**A:** Edge count shows **opportunity**, not conviction. Consider:
- **IV context:** Is premium rich or thin?
- **Below 55-month MA?** Increase position size
- **Account risk limits:** Don't over-allocate
- **Market conditions:** Broad volatility environment

Edge count = "How many position types can I construct?" Not "How much size to put on."

---

### Q: What if 45d and 90d give different signals?
**A:** Look at the pattern output:
- `IC45 only` = Short-term containment, no longer-term edge
- `IC90 only` = Longer-term containment, but 45d is noisy
- `IC90 + IC45` = Both timeframes agree (strongest signal)

If only one appears, that's the timeframe with an edge. The other didn't meet threshold.

---

### Q: How do I interpret IVPrem?
**A:** IV Premium shows opportunity cost:
- **≥+15%** = Rich premium (favorable for selling ICs)
- **≤-15%** = Thin premium (consider reducing size or skipping)
- **-15% to +15%** = Normal (proceed based on pattern)

Example: `IC90 + IC45 [2 edges]` with IVPrem = -20%  
**Interpretation:** Pattern is strong, but premium is thin. Maybe reduce position size or skip.

---

### Q: Why does edge count matter for Plotly graphs?
**A:** When visualizing results, edge count helps you:
- Quickly identify multi-edge opportunities (IC + BIAS)
- Prioritize which tickers to analyze deeper
- Understand position construction complexity

On a scatter plot, color-coding by edge count makes high-opportunity setups stand out visually.

---

### Q: What's the difference between 90Bias and 90Drift?
**A:**
- **90Bias** = % of moves that were positive direction (binary: up or down?)
- **90Drift** = Average magnitude of moves (how far did it move?)

Example:
- 90Bias = 70% (7 out of 10 moves were up)
- 90Drift = +1.2% (average move was small)

**Interpretation:** Directionally bullish, but moves are small. Bias exists, but not large drift.

---

## NEXT STEPS

This guide explains **what the tool tells you**. The next phase is building:
1. **Position sizing calculator** - Input: pattern + IV context → Output: # of contracts
2. **Trade specification tool** - Input: ticker + strategy → Output: exact strikes, DTE, entry targets
3. **Execution tracking** - Log: system signal vs actual trade vs outcome

**Goal:** Systematic execution with discretionary sizing.

---

**Last Updated:** October 2025  
**Version:** 2.5 (Foundation - Pattern Detection Complete)