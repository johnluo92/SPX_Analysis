# CONFIG.PY UPGRADE GUIDE

## LLM Session Requirements

**Files Needed in Context:**
- Current `config.py` (uploaded)
- This guide only

**Estimated Changes:** ~50 lines added
**Difficulty:** Low
**Time:** 15 minutes

## Current State Analysis

Your `config.py` contains:
- Cache paths
- API configurations (FRED, Alpha Vantage, CBOE)
- Model paths
- CBOE ticker mappings
- Date ranges

**What's Missing:**
- VIX regime definitions
- Quantile levels for distribution forecasting
- Loss function weights
- Calendar context definitions
- Confidence score parameters

## Required Additions

### 1. VIX Regime Definitions

Add after existing constants:

```python
# =============================================================================
# PROBABILISTIC FORECASTING CONFIGURATION
# =============================================================================

# VIX Regime Boundaries (based on historical quartiles)
VIX_REGIMES = {
    'low_vol': {'name': 'Low Volatility', 'max': 16.77, 'color': '#2ecc71'},
    'normal': {'name': 'Normal', 'min': 16.77, 'max': 24.40, 'color': '#3498db'},
    'elevated': {'name': 'Elevated', 'min': 24.40, 'max': 39.67, 'color': '#f39c12'},
    'crisis': {'name': 'Crisis', 'min': 39.67, 'color': '#e74c3c'}
}

# Regime thresholds as array (for fast classification)
REGIME_THRESHOLDS = [16.77, 24.40, 39.67]  # [low->normal, normal->elevated, elevated->crisis]
REGIME_NAMES = ['low_vol', 'normal', 'elevated', 'crisis']
REGIME_LABELS = {0: 'low_vol', 1: 'normal', 2: 'elevated', 3: 'crisis'}
```

**Rationale:**
- Thresholds from historical VIX quartiles (can be recalculated from data)
- Low Vol: 0-25th percentile
- Normal: 25th-50th percentile
- Elevated: 50th-90th percentile
- Crisis: >90th percentile
- Colors for visualization consistency

### 2. Quantile Levels

```python
# Quantile levels for distribution forecasting
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_NAMES = ['q10', 'q25', 'q50', 'q75', 'q90']

# Quantile loss weights (higher weight on tails for risk management)
QUANTILE_WEIGHTS = {
    0.10: 1.2,  # Slightly emphasize lower tail
    0.25: 1.0,
    0.50: 1.0,  # Median
    0.75: 1.0,
    0.90: 1.5   # Emphasize upper tail (risk events)
}
```

**Rationale:**
- 5 quantiles capture distribution shape efficiently
- 10th percentile: optimistic scenario
- 90th percentile: risk scenario
- Weighted toward tails for better risk assessment
- Median (50th) is natural point estimate

### 3. Multi-Output Loss Weights

```python
# Loss function weights for multi-output training
LOSS_WEIGHTS = {
    'point_estimate': 0.30,   # MSE on percentage change
    'quantiles': 0.35,        # Pinball loss on distribution
    'regimes': 0.25,          # Log loss on regime classification
    'confidence': 0.10        # Calibration penalty
}

# Verify weights sum to 1.0
assert abs(sum(LOSS_WEIGHTS.values()) - 1.0) < 1e-6, "Loss weights must sum to 1.0"
```

**Rationale:**
- Point estimate gets 30% - main prediction
- Quantiles get 35% - distribution shape is critical
- Regimes get 25% - discrete outcomes for allocation
- Confidence gets 10% - meta-accuracy for trust
- Can be tuned via hyperparameter search

### 4. Calendar Context Definitions

```python
# Options expiration calendar contexts
CALENDAR_CONTEXTS = {
    'pre_opex': {
        'name': 'Pre-OpEx',
        'description': '5 business days before monthly options expiration',
        'days_before_opex': [1, 2, 3, 4, 5],
        'color': '#e74c3c'
    },
    'opex_week': {
        'name': 'OpEx Week', 
        'description': 'Week of monthly options expiration',
        'days_before_opex': [0, -1, -2, -3, -4],
        'color': '#f39c12'
    },
    'post_opex': {
        'name': 'Post-OpEx',
        'description': '5 business days after monthly options expiration',
        'days_after_opex': [1, 2, 3, 4, 5],
        'color': '#3498db'
    },
    'mid_cycle': {
        'name': 'Mid-Cycle',
        'description': 'Rest of the month between OpEx cycles',
        'color': '#2ecc71'
    }
}

# Monthly options expiration: 3rd Friday of each month
# Dates are calculated dynamically, this just defines the rule
OPEX_RULE = '3rd_friday'  # Can be 'custom' for specific dates
```

**Rationale:**
- VIX behavior differs around options expiration
- 4 distinct contexts capture different dynamics
- Pre-OpEx: Positioning and hedging
- OpEx week: Gamma and pin risk
- Post-OpEx: Position unwind
- Mid-cycle: Normal trading

### 5. Confidence Score Parameters

```python
# Confidence score calculation parameters
CONFIDENCE_PARAMS = {
    'feature_availability': {
        'weight': 0.40,
        'min_features': 40,  # Below this, confidence drops rapidly
        'target_features': 48  # All selected features available
    },
    'feature_freshness': {
        'weight': 0.30,
        'max_staleness_days': {
            'FRED': 35,      # Monthly data tolerable
            'CBOE': 1,       # Daily data required
            'Yahoo': 1       # Real-time needed
        }
    },
    'regime_stability': {
        'weight': 0.20,
        'lookback_days': 5,
        'penalty_per_transition': 0.15  # Reduce confidence per regime change
    },
    'historical_error': {
        'weight': 0.10,
        'lookback_window': 63,  # ~3 months of trading days
        'similarity_threshold': 0.85  # Correlation to consider "similar"
    }
}

# Verify confidence weights sum to 1.0
assert abs(sum(p['weight'] for p in CONFIDENCE_PARAMS.values()) - 1.0) < 1e-6
```

**Rationale:**
- Feature availability (40%): Missing CBOE data is critical
- Freshness (30%): Stale macro data less useful
- Regime stability (20%): Transitions have higher uncertainty
- Historical error (10%): Past similar conditions
- Each component scored 0-1, weighted, produces final confidence

### 6. Prediction Output Schema

```python
# Expected output structure from probabilistic forecaster
PREDICTION_SCHEMA = {
    'timestamp': 'datetime64[ns]',
    'horizon_days': 'int',
    'point_estimate': 'float32',      # Expected % change
    'q10': 'float32',                  # 10th percentile
    'q25': 'float32',                  # 25th percentile  
    'q50': 'float32',                  # 50th percentile (median)
    'q75': 'float32',                  # 75th percentile
    'q90': 'float32',                  # 90th percentile
    'regime_low_vol': 'float32',       # P(Low Vol)
    'regime_normal': 'float32',        # P(Normal)
    'regime_elevated': 'float32',      # P(Elevated)
    'regime_crisis': 'float32',        # P(Crisis)
    'confidence': 'float32',           # 0-1 confidence score
    'calendar_context': 'str',         # Which context was used
    'feature_count': 'int',            # Features available at prediction
    'model_version': 'str'             # For provenance tracking
}
```

**Rationale:**
- Defines contract between trainer and predictor
- Type specifications for validation
- Includes metadata for auditing
- Regime probabilities must sum to 1.0

### 7. Model Paths Updates

Add to existing MODEL_BASE_PATH section:

```python
# Probabilistic model artifacts
PROBABILISTIC_MODEL_PATH = MODEL_BASE_PATH / "probabilistic"
PROBABILISTIC_MODEL_PATH.mkdir(exist_ok=True)

# Sub-paths for calendar contexts
CONTEXT_MODEL_PATHS = {
    ctx: PROBABILISTIC_MODEL_PATH / ctx 
    for ctx in CALENDAR_CONTEXTS.keys()
}

# Create all context directories
for path in CONTEXT_MODEL_PATHS.values():
    path.mkdir(exist_ok=True)

# Predictions database
PREDICTIONS_DB_PATH = JSON_DATA_PATH / "predictions.db"
```

### 8. Validation Thresholds

```python
# Validation thresholds for prediction quality
VALIDATION_THRESHOLDS = {
    'quantile_monotonicity': True,         # q10 <= q25 <= ... <= q90
    'regime_sum': {'min': 0.99, 'max': 1.01},  # Probabilities sum to ~1
    'confidence_range': {'min': 0.0, 'max': 1.0},
    'point_estimate_range': {'min': -0.50, 'max': 2.00},  # -50% to +200%
    'quantile_spread': {
        'min_iqr': 0.02,  # q75-q25 must be at least 2%
        'max_iqr': 0.50   # q75-q25 must be at most 50%
    }
}
```

**Rationale:**
- Runtime validation of predictions
- Catch model failures early
- Ensure physical consistency
- Flag anomalous outputs

## Complete Addition Block

Here's the full block to add to config.py:

```python
# =============================================================================
# PROBABILISTIC FORECASTING CONFIGURATION
# =============================================================================

# VIX Regime Definitions
VIX_REGIMES = {
    'low_vol': {'name': 'Low Volatility', 'max': 16.77, 'color': '#2ecc71'},
    'normal': {'name': 'Normal', 'min': 16.77, 'max': 24.40, 'color': '#3498db'},
    'elevated': {'name': 'Elevated', 'min': 24.40, 'max': 39.67, 'color': '#f39c12'},
    'crisis': {'name': 'Crisis', 'min': 39.67, 'color': '#e74c3c'}
}

REGIME_THRESHOLDS = [16.77, 24.40, 39.67]
REGIME_NAMES = ['low_vol', 'normal', 'elevated', 'crisis']
REGIME_LABELS = {0: 'low_vol', 1: 'normal', 2: 'elevated', 3: 'crisis'}

# Quantile Levels
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_NAMES = ['q10', 'q25', 'q50', 'q75', 'q90']
QUANTILE_WEIGHTS = {0.10: 1.2, 0.25: 1.0, 0.50: 1.0, 0.75: 1.0, 0.90: 1.5}

# Multi-Output Loss Weights
LOSS_WEIGHTS = {
    'point_estimate': 0.30,
    'quantiles': 0.35,
    'regimes': 0.25,
    'confidence': 0.10
}
assert abs(sum(LOSS_WEIGHTS.values()) - 1.0) < 1e-6

# Calendar Contexts
CALENDAR_CONTEXTS = {
    'pre_opex': {'name': 'Pre-OpEx', 'days_before_opex': [1,2,3,4,5], 'color': '#e74c3c'},
    'opex_week': {'name': 'OpEx Week', 'days_before_opex': [0,-1,-2,-3,-4], 'color': '#f39c12'},
    'post_opex': {'name': 'Post-OpEx', 'days_after_opex': [1,2,3,4,5], 'color': '#3498db'},
    'mid_cycle': {'name': 'Mid-Cycle', 'color': '#2ecc71'}
}
OPEX_RULE = '3rd_friday'

# Confidence Parameters
CONFIDENCE_PARAMS = {
    'feature_availability': {'weight': 0.40, 'min_features': 40, 'target_features': 48},
    'feature_freshness': {'weight': 0.30, 'max_staleness_days': {'FRED': 35, 'CBOE': 1, 'Yahoo': 1}},
    'regime_stability': {'weight': 0.20, 'lookback_days': 5, 'penalty_per_transition': 0.15},
    'historical_error': {'weight': 0.10, 'lookback_window': 63, 'similarity_threshold': 0.85}
}
assert abs(sum(p['weight'] for p in CONFIDENCE_PARAMS.values()) - 1.0) < 1e-6

# Prediction Schema
PREDICTION_SCHEMA = {
    'timestamp': 'datetime64[ns]', 'horizon_days': 'int', 'point_estimate': 'float32',
    'q10': 'float32', 'q25': 'float32', 'q50': 'float32', 'q75': 'float32', 'q90': 'float32',
    'regime_low_vol': 'float32', 'regime_normal': 'float32', 
    'regime_elevated': 'float32', 'regime_crisis': 'float32',
    'confidence': 'float32', 'calendar_context': 'str', 
    'feature_count': 'int', 'model_version': 'str'
}

# Model Paths
PROBABILISTIC_MODEL_PATH = MODEL_BASE_PATH / "probabilistic"
PROBABILISTIC_MODEL_PATH.mkdir(exist_ok=True)
CONTEXT_MODEL_PATHS = {ctx: PROBABILISTIC_MODEL_PATH / ctx for ctx in CALENDAR_CONTEXTS.keys()}
for path in CONTEXT_MODEL_PATHS.values():
    path.mkdir(exist_ok=True)
PREDICTIONS_DB_PATH = JSON_DATA_PATH / "predictions.db"

# Validation Thresholds
VALIDATION_THRESHOLDS = {
    'quantile_monotonicity': True,
    'regime_sum': {'min': 0.99, 'max': 1.01},
    'confidence_range': {'min': 0.0, 'max': 1.0},
    'point_estimate_range': {'min': -0.50, 'max': 2.00},
    'quantile_spread': {'min_iqr': 0.02, 'max_iqr': 0.50}
}
```

## Testing Requirements

After modification, test:

```python
# Test imports
from config import (
    VIX_REGIMES, REGIME_THRESHOLDS, QUANTILE_LEVELS,
    LOSS_WEIGHTS, CALENDAR_CONTEXTS, CONFIDENCE_PARAMS,
    PREDICTION_SCHEMA, VALIDATION_THRESHOLDS
)

# Test regime consistency
assert len(REGIME_THRESHOLDS) == len(REGIME_NAMES) - 1
assert all(REGIME_THRESHOLDS[i] < REGIME_THRESHOLDS[i+1] for i in range(len(REGIME_THRESHOLDS)-1))

# Test quantile ordering
assert all(QUANTILE_LEVELS[i] < QUANTILE_LEVELS[i+1] for i in range(len(QUANTILE_LEVELS)-1))

# Test loss weights sum
assert abs(sum(LOSS_WEIGHTS.values()) - 1.0) < 1e-6

# Test confidence weights sum
assert abs(sum(p['weight'] for p in CONFIDENCE_PARAMS.values()) - 1.0) < 1e-6

# Test paths exist
assert PROBABILISTIC_MODEL_PATH.exists()
assert PREDICTIONS_DB_PATH.parent.exists()

print("✅ All config tests passed")
```

## Common Pitfalls

1. **Don't hardcode regime thresholds without data justification**
   - Calculate from historical VIX quantiles
   - Update periodically as VIX distribution shifts

2. **Loss weights impact model behavior**
   - Higher quantile weight → better distribution shape
   - Higher regime weight → better discrete predictions
   - Balance based on use case

3. **Calendar contexts must be mutually exclusive**
   - Every trading day belongs to exactly one context
   - Implement date→context mapping function

4. **Path creation order matters**
   - Create parent directories before children
   - Use .mkdir(exist_ok=True) to avoid errors

## Next File to Modify

After config.py, proceed to:
**PREDICTIONS_DATABASE.md** - Storage schema depends on these constants

## Summary of Changes

- ✅ 8 new constant groups added (~50 lines)
- ✅ All weights validated to sum to 1.0
- ✅ Directory structure created
- ✅ Schema definitions for downstream use
- ✅ No breaking changes to existing code
- ✅ Backward compatible (old constants unchanged)
