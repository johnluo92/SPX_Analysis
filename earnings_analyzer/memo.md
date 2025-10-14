# EARNINGS CONTAINMENT ANALYZER
## Technical Memorandum v3.0

**Date:** October 13, 2025  
**Lookback:** 24 Quarters (~6 Years)

---

## EXECUTIVE SUMMARY

This system identifies Iron Condor candidates and directional bias patterns by analyzing historical price behavior around earnings. It uses 30-day historical volatility (HVol) for consistent width sizing, then pattern-matches 24 quarters of containment data to find tradeable edges.

**Core Principle:** The edge is finding tickers that historically respect volatility-based boundaries with exploitable directional asymmetries—not predicting exact move magnitude.

---

## I. SYSTEM ARCHITECTURE

### Data Sources
- **Price Data:** Alpha Vantage API (4 rotating keys)
- **Implied Volatility:** Yahoo Finance (current snapshot)
- **Caching:** Local JSON (earnings dates, IV, price history)

### Core Calculations

**Historical Volatility (HVol30):**
- 30-day trailing window before earnings
- Annualized: `daily_vol × √252`

**Strike Width Formula:**
```
width = HVol30 × √(DTE/365) × 100
```
- DTE = Days to expiration (calendar days: 45 or 90)
- Multiplier = 1.0 (one standard deviation)
- Separate widths calculated for 45d and 90d horizons

**Entry Pricing:**
- BMO: Previous day's close
- AMC: Earnings day close

---

## II. STATISTICAL METHODOLOGY

### Containment Analysis
Each historical earnings event classified as:
- **Contained:** |Move| ≤ Width
- **Break Up:** Move > Width
- **Break Down:** Move < -Width

### Metrics (24-Quarter Aggregation)
- **Containment Rate:** % within width
- **Break Ratio:** Max(Up,Down) / Min(Up,Down)
- **Overall Bias:** % positive moves
- **Break Bias:** % breaks that went up
- **Average Drift:** Mean move / mean width

---

## III. STRATEGY CLASSIFICATION

### Iron Condor Signals

**IC90 / IC45** - Containment at 90/45 days
- Threshold: ≥69.5% containment
- Variants:
  - Standard: Symmetric breaks (ratio <2.0)
  - `⚠↑`: Upside asymmetry (break bias ≥70%)
  - `⚠↓`: Downside asymmetry (break bias ≤30%)
- Each horizon evaluated independently

### Directional Signals

**BIAS↑** - Upward Edge (ANY of):
- Overall bias ≥70%
- Break ratio ≥2.0 favoring upside (min 2 breaks)
- Average drift ≥+5.0%

**BIAS↓** - Downward Edge (ANY of):
- Overall bias ≤30%
- Break ratio ≥2.0 favoring downside (min 2 breaks)
- Average drift ≤-5.0%

### Edge Count
- Each pattern (IC90, IC45, BIAS) = 1 independent edge
- Max 3 edges per ticker
- High conviction = 2+ edges (stronger confirmation)
- Single-edge plays valid if other factors align

---

## IV. KNOWN BIASES & LIMITATIONS

### HVol30 Systematic Issues

**Underestimation:**
- Affects stable, liquid tickers
- HVol captures normal vol, not event-specific risk
- Widths slightly narrow; predictable patterns often compensate

**Overestimation:**
- Affects choppy, illiquid tickers
- HVol includes noise unrelated to earnings
- Widths too wide; mitigated by liquidity filters

### Data Constraints
- No historical IV (cannot backtest market-implied widths)
- Rate limits: 500 calls/day per API key
- Survivorship bias (delisted tickers excluded)

### Formula Assumptions
`HVol × √(DTE/365)` assumes log-normal distribution and stationary volatility—both violated during earnings.

**Why use it:** Consistent, requires no forward-looking data, systematic errors are manageable. The market is adversarial—if HVol worked perfectly, everyone would use it and the edge would vanish.

---

## V. VALIDATION

### Automated Checks
✅ Column integrity  
✅ Containment rates 0-100%  
✅ Strategy logic consistency  
✅ No data leakage (HVol pre-earnings only)

### Manual Review
- IV elevation context
- Liquidity (spreads, open interest)
- Break asymmetry for wing sizing
- Earnings timing

---

## VI. PHILOSOPHICAL FOUNDATIONS

**What This System Is:**
- Pattern recognition for containment behavior
- Filter for high-probability opportunities
- Detector for directional asymmetries

**What It Is Not:**
- Predictor of exact move size
- Guarantee of future performance
- Substitute for risk management

**Core Beliefs:**
- Edges exist in behavioral patterns that markets haven't yet arbitraged away
- Simplicity scales; complexity overfits and breaks when the market shifts
- The market is adversarial—it adapts, sniffs out patterns, eliminates inefficiencies
- **The insight IS the variance**—differentiation between tickers reveals the edge
- Continuous validation against live P&L is the only truth
- Imperfect data ≠ invalid system; perfection is the enemy of execution

---

## VII. THINKORSWIM INTEGRATION

### Division of Labor

**This System:**
- Historical containment (24 quarters)
- Directional bias classification
- HVol-based width baseline

**ThinkorSwim:**
- Real-time IV & IV Rank
- Greeks, liquidity, spreads
- Live options chain

### Synergy

**Width Determination:**
```
Live Width = MAX(TOS_IV × √(DTE/365), HVol30 × √(DTE/365))
```
Use larger to protect against underestimation.

**Pre-Trade Checklist:**
1. Backtest shows edge (1+ patterns)
2. TOS IV Rank >70% (rich premium)
3. Liquidity confirmed (tight spreads, sufficient OI)
4. Wings adjusted for break asymmetry

**TOS Filters:**
- IV Rank >70%
- IV - HVol differential
- Liquidity score
- Earnings countdown

---

## VIII. LESSONS LEARNED

### Evolution
- Tier multipliers → Artificial clustering (abandoned)
- RVol-based widths → Data leakage (abandoned)
- Leave-one-out RVol → Contamination persists (abandoned)
- Pure HVol → Limitations accepted, consistency achieved

**Key insight:** Markets punish overcomplicated systems. Simple, robust methods survive regime changes.

### Philosophy Shift
- From: Predict exact moves (hubris)
- To: Find predictable behavior patterns (humility)
- Result: Pattern-match history, trade the edges, respect the adversary

---

## IX. OPERATING PRINCIPLES

### Risk Management
- Max 5% per ticker
- Max 30% portfolio in earnings
- Exit IC at 2× credit (stop loss)
- Adjust wings for asymmetry

### Execution
- Prioritize 2+ edges (stronger confirmation)
- Single-edge plays valid with supporting factors
- Require IV Rank >70%
- Confirm liquidity
- Track live P&L vs expectations

### Iteration
- Review all trades post-expiration
- Update backtest quarterly
- Abandon if edge deteriorates
- Adapt to regime changes

---

## X. FINAL STATEMENT

This system uses an imperfect metric to estimate unknowable distributions. It makes convenient rather than empirically perfect assumptions. The market is adversarial and will eventually adapt.

**And yet.**

It provides systematic, repeatable, unbiased identification of opportunities **today**. It acknowledges limitations openly rather than hiding behind false precision. It generates signals validatable against reality, not theory.

The measure is not theoretical optimality but practical utility—transforming hunches into testable hypotheses, and hypotheses into executable trades.

Every edge has a half-life. The work never ends. The market will teach what the backtest cannot.

Trade what works. Abandon what fails. Adapt or perish.

---

**Status:** Production-Ready v3.0

*"The edge is finding tickers that behave predictably despite imperfect inputs."*

**END MEMORANDUM**