VIX Anomaly Detection System: Technical Overview
Version 4.0 | Multi-Dimensional Market Stress Detection

Executive Summary
This system monitors the VIX (CBOE Volatility Index) and broader market conditions to identify unusual patterns that may signal regime changes or elevated risk. It operates by comparing current market behavior against 15 years of historical data across multiple dimensions, producing daily anomaly scores and persistence metrics.
Primary Use Cases:

Daily monitoring during normal market conditions
Enhanced surveillance during elevated volatility periods
Regime transition detection (e.g., from "Normal" to "Elevated" volatility)


Core Functionality
What It Measures
The system analyzes market data through 10 specialized "detectors," each focused on a specific aspect of market behavior:

VIX Mean Reversion: How far VIX deviates from its historical averages (10-day to 252-day moving averages)
VIX Momentum: Rate of change in VIX (velocity, acceleration, volatility-of-volatility)
VIX Regime Structure: Stability of current volatility regime and days since last transition
CBOE Options Flow: Institutional hedging behavior (SKEW index, put/call ratios, correlation indices)
VIX-SPX Relationship: Correlation strength between VIX and S&P 500 movements
SPX Price Action: S&P 500 technical patterns (moving average crossovers, momentum, RSI)
SPX Volatility Regime: Realized volatility vs. implied volatility (VIX premium/discount)
Macro Rates: Treasury yields, yield curve shape, credit spreads
Commodities Stress: Gold, oil, silver momentum as flight-to-safety indicators
Cross-Asset Divergence: Conflicting signals across equities, bonds, and commodities

Each detector runs an Isolation Forest algorithm (a machine learning technique for anomaly detection) trained on 15 years of historical data. Five additional "random subspace" detectors analyze randomized feature combinations to capture patterns not explicitly modeled.
Output Metrics
Ensemble Anomaly Score (0.0 to 1.0)

Weighted average of all 15 detectors
Values above 0.70 = Moderate anomaly (85th percentile historically)
Values above 0.78 = High anomaly (92nd percentile)
Values above 0.88 = Critical anomaly (98th percentile)

These thresholds are statistically derived from the training data distribution using bootstrap confidence intervals (1,000 iterations), not arbitrary cutoffs.
Persistence Metrics

Current streak: Consecutive days above threshold
Mean duration: Average length of historical anomaly episodes (≈3.6 days)
Max duration: Longest historical episode (56 days, likely during 2008 or 2020 crises)
Anomaly rate: 8% of trading days historically exceed the "High" threshold

Regime Classification

Low Vol: VIX < 16.77
Normal: VIX 16.77-24.40
Elevated: VIX 24.40-39.67
Crisis: VIX > 39.67


Data Sources & Refresh Cycle
Training Data (15 years):

Yahoo Finance: SPX, VIX, gold, silver, dollar index, treasuries
FRED (Federal Reserve): Treasury yields, credit spreads, inflation expectations, currency indices
CBOE: SKEW, put/call ratios, correlation indices, tail hedge index

Live Updates:

VIX and SPX prices refreshed every 15 minutes during market hours
Derived features recalculated with live prices (z-scores, moving averages, percentiles)
Anomaly scores recomputed without retraining models

Operating Modes:

Training Mode (weekly recommended): Full retrain on 15 years of data (~3 minutes)
Cached Mode (daily): Uses pre-trained models, updates only live features (~5 seconds)


Interpretation Guidelines
Anomaly Score Context
Score = 0.40-0.50 (Normal)

System detects no unusual patterns
Markets behaving within historical norms
Routine monitoring sufficient

Score = 0.70-0.78 (Moderate)

2-3 detectors signaling elevated stress
Examples: VIX momentum surge, put/call ratio spike, correlation breakdown
Warrants closer observation but not necessarily actionable

Score = 0.78-0.88 (High)

4-5 detectors signaling stress
Multiple domains showing anomalous behavior simultaneously
Historical precedent: Occurs ~8% of trading days
May precede regime transitions

Score > 0.88 (Critical)

6+ detectors signaling extreme stress
Systemic patterns similar to 2008, 2020, or flash crash events
Historical precedent: Occurs <2% of trading days

Persistence Interpretation
Current Streak = 0 days

No consecutive anomaly days
Market returned to normal patterns

Current Streak = 3-5 days

Near historical mean duration
Anomaly likely resolving soon based on historical patterns

Current Streak > 10 days

Extended episode, well above mean
Suggests persistent structural stress rather than short-term volatility spike


Known Limitations
Data Availability

CBOE indicators (SKEW, put/call ratios): May lag 1-2 days after market close
FRED macro data: Updates with 1-2 day delay for most series
Weekend/holiday gaps filled by forward-filling last available values

Inference: If the system shows "9/15 detectors active" in diagnostics, it means 6 detectors lack real-time data. This reduces confidence but does not invalidate the ensemble score.
Statistical Assumptions

Isolation Forest assumes feature independence (not strictly true in financial markets)
Thresholds derived from 15-year window may not capture regime shifts lasting decades
Z-scores assume approximate normality of feature distributions (reasonable for most features, questionable for fat-tailed phenomena)

Model Refresh

Cached mode uses last trained model state
If market structure changes (e.g., post-pandemic volatility patterns), weekly retraining recommended
No online learning: Models do not adapt between training cycles


Practical Use Scenarios
Daily Monitoring (Cached Mode)
Use Case: Check morning anomaly score before market open
Workflow:

System loads cached models (~5 seconds)
Fetches live VIX/SPX prices
Recalculates derived features (z-scores, percentiles, momentum)
Runs anomaly detection
Updates live_state.json

Decision Rules:

Score < 0.70: No action, routine monitoring
Score 0.70-0.78: Review which detectors are elevated (feature attribution)
Score > 0.78: Consider increased hedging or position adjustment

Weekly Training
Use Case: Sunday evening full retrain to capture recent market behavior
Workflow:

Download 15 years of historical data (~2 minutes)
Engineer 200+ features
Train 15 anomaly detectors
Compute statistical thresholds
Generate persistence statistics
Cache models for daily use

Rationale: Ensures thresholds reflect current market regime (e.g., post-COVID volatility baseline differs from 2010-2019)
Crisis Mode (High Frequency)
Use Case: During VIX > 30 events, monitor every 15 minutes
Workflow:

Auto-refresh enabled (exponential backoff on failures)
Live VIX/SPX updates every 15 minutes
Anomaly scores recalculated in real-time
Persistence streak tracked intraday

Note: Current streak calculation attempts to exclude incomplete trading days (e.g., if market hasn't closed, don't count today in streak). Requires pytz library for timezone-aware logic.

Feature Attribution Example
When anomaly score = 0.82 (High), the system provides top contributing features per detector:
VIX Mean Reversion (Score: 0.91)

VIX vs 21-day MA: +8.2 (importance: 0.32)
VIX Z-Score (63d): +2.8 (importance: 0.28)
VIX Percentile (252d): 98th (importance: 0.21)

Interpretation: VIX is stretched 8.2 points above its 21-day average, 2.8 standard deviations above its 63-day mean, and in the 98th percentile of the past year—all mean reversion signals.

Memory Management (Technical Note)
The system includes optional memory profiling (requires psutil library). During live refresh cycles:

Baseline memory: ~150 MB
Per-refresh growth: ~0.5 MB expected
Warning threshold: +50 MB growth
Critical threshold: +200 MB growth

Inference: If memory growth exceeds thresholds, likely indicates memory leak in refresh loop. Recommended action: Restart system daily or use process monitoring.

Suggestions for Improvement
Short-Term Enhancements

Intraday VIX Term Structure: Current system uses daily VIX close. Adding VIX futures term structure (VX1-VX2 spread) would improve regime detection.
Implied Correlation Tracking: Monitor dispersion trades (index implied vol vs. single-stock implied vol) for early stress signals.
Gamma Exposure Proxy: Estimate market maker gamma positioning using SPX option open interest (requires additional data feed).

Medium-Term Enhancements

Regime Transition Forecasting: Current system detects anomalies but doesn't predict regime changes. Add Markov regime-switching model for transition probabilities.
Event Attribution: Link anomaly spikes to scheduled events (FOMC, earnings seasons, geopolitical events) to reduce false positives.
Portfolio Integration: Output position sizing recommendations based on anomaly score and persistence (currently outputs generic "Aggressive/Moderate/Light" labels).

Long-Term Considerations

Online Learning: Implement incremental learning to avoid weekly retraining (e.g., partial fit on rolling 252-day window).
Explainable AI: Current SHAP values provide feature importance but not directional causality. Add counterfactual analysis ("if VIX were 2 points lower, score would drop to X").
Multi-Asset Expansion: Extend to credit (HYG/LQD spreads), FX volatility (CVIX), commodities volatility (OVX) for holistic risk view.


Historical Calibration (Inference)
Based on threshold definitions (98th percentile = 0.88), the system likely identifies:

2008 Financial Crisis: Extended periods above 0.88 (Sept-Nov 2008)
2020 COVID Crash: Peak scores > 0.95 (March 2020)
2010 Flash Crash: Brief spike > 0.90 (May 6, 2010)
2015 ETF Flash Crash: Moderate spike 0.75-0.82 (Aug 24, 2015)

Caveat: Exact historical scores not provided in documentation. Above inferences based on percentile definitions and known market events.

When to Use This System
Ideal Scenarios:

Regime uncertainty: VIX near regime boundaries (16-17, 24-25, 39-40)
Correlation breakdowns: SPX rising while VIX rising (atypical pattern)
Options mispricing: SKEW index > 145 with low VIX (tail risk premium)
Macro stress: Yield curve inversion + high anomaly score

Not Suitable For:

Directional trading: System detects stress, not directionality (VIX can spike in either market direction)
High-frequency trading: 15-minute refresh insufficient for intraday strategies
Single-name equity: Designed for index-level volatility, not individual stocks


Final Notes
This system is a monitoring tool, not a trading signal generator. Anomaly scores indicate deviation from historical norms but do not prescribe specific actions. Users should integrate outputs with:

Portfolio risk limits
Existing volatility hedging strategies
Fundamental market views
Liquidity constraints

The 15-year training window captures multiple crisis events (2008, 2011, 2015, 2018, 2020), providing robust baseline for "normal" vs. "anomalous" behavior. However, unprecedented events (by definition) may not trigger high scores until sufficient intra-regime deviation occurs.
Recommended Review Frequency:

Daily: Cached mode (5 seconds)
Weekly: Full retrain (3 minutes)
Real-time: Crisis periods only (VIX > 30)