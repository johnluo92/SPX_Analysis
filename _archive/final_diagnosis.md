# Statistical Thresholds Analysis - Final Report

## 🚨 Critical Finding

**Statistical thresholds are 70% HIGHER than legacy → System would miss most anomalies**

---

## Test Results

### Threshold Comparison
| Level | Legacy | Statistical | Difference |
|-------|--------|------------|------------|
| Moderate | 0.50 | 0.85 | +70% ❌ |
| High | 0.70 | 0.93 | +32% ❌ |
| Critical | 0.85 | 0.99 | +16% ⚠️ |

### Classification Impact
- **7/8 test scores** would downgrade to NORMAL
- Current score (0.45): Both methods agree = NORMAL ✅
- Score 0.85: Legacy=CRITICAL, Statistical=NORMAL ❌

---

## Root Cause

**Training data has unusually high anomaly scores:**
- Mean: 0.50 (expected: 0.20-0.30)
- Range: 0.12 - 0.99 (very wide spread)
- 10% missing feature values

**Why:**
1. Contamination parameter (5%) too high → treats 100+ samples as anomalies
2. Missing data (10%) may inflate scores
3. Isolation Forest sees "normal" as high scores in this dataset

---

## The Math is Correct, But...

Statistical methods correctly say: *"90% of your training data has scores below 0.85"*

Problem: Your legacy thresholds assume "50% of training = anomaly" which is statistically invalid.

---

## Decision Matrix

### Option 1: Keep Legacy (Safe) ⭐
**Action:** None - use existing 0.50/0.70/0.85  
**Pro:** System works as-is, no changes  
**Con:** No statistical justification  
**Recommendation:** Use this if current alerts are valuable

### Option 2: Use Statistical (Strict)
**Action:** Replace with 0.85/0.93/0.99  
**Pro:** Statistically valid at 90/95/99% confidence  
**Con:** 70% fewer anomaly alerts  
**Recommendation:** Only if current system has too many false positives

### Option 3: Retrain System
**Action:** Lower contamination to 1-2%, fix missing data  
**Pro:** Better calibrated from start  
**Con:** Requires retraining, may change existing behavior  
**Recommendation:** Best long-term solution

### Option 4: Hybrid Approach
**Action:** Use statistical for ensemble, keep legacy for domains  
**Pro:** Balance between rigor and sensitivity  
**Con:** More complex to explain  
**Recommendation:** Good compromise

---

## Implementation Paths

### Path A: Status Quo (Recommended for now)
```python
# Keep current anomaly_system.py unchanged
# Add p-values as metadata only
# Document that thresholds are heuristic not statistical
```

### Path B: Statistical Migration (3-phase)
```python
# Phase 1: Add both methods to output
# Phase 2: Run parallel for 1 month
# Phase 3: Switch if statistical performs better
```

### Path C: Recalibration
```python
# 1. Lower contamination: 0.05 → 0.02
# 2. Fix missing data handling
# 3. Recalculate thresholds
# 4. Validate against historical alerts
```

---

## Files Generated

```
json_data/
├── threshold_analysis.json       # Full comparison
├── threshold_visualization.json  # Chart data
└── training_anomaly_scores.npy   # Raw scores (2009 samples)
```

---

## Recommendation

**DO NOT INTEGRATE** statistical thresholds yet.

**Instead:**
1. Review why training scores are so high
2. Investigate missing data impact (10% of features)
3. Consider lowering contamination parameter
4. Re-run test after fixes

**Statistical thresholds work correctly - your training data distribution is the issue.**

---

## Quick Commands

```bash
# Check training score distribution
python -c "import numpy as np; scores=np.load('json_data/training_anomaly_scores.npy'); print(f'Mean: {scores.mean():.3f}, Median: {np.median(scores):.3f}, P90: {np.percentile(scores, 90):.3f}')"

# View missing data by feature
python -c "from integrated_system_production import IntegratedMarketSystemV4; s=IntegratedMarketSystemV4(); s.train(7); print(s.vix_predictor.features.isna().sum().sort_values(ascending=False).head(20))"
```

---

## Questions for You

1. Are current anomaly alerts (with legacy thresholds) useful or noisy?
2. Would you accept 70% fewer alerts if they're more statistically valid?
3. Want to investigate why training scores are high?

**No action needed until you decide.**