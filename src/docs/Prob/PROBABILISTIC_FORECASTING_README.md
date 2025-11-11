# Probabilistic VIX Forecasting System V3

## Architecture Overview

This system transforms your binary VIX expansion classifier into a **full probabilistic distribution forecaster** that models the complete range of VIX outcomes over the next 5 days as continuous probability distributions.

### What Changed

**Before (Binary Classifier):**
- Single output: "Will VIX expand >5%?" (Yes/No)
- Problem: 39% precision - too many false positives
- Fixed threshold creates sharp boundary through continuous data

**After (Probabilistic Forecaster):**
- Complete distribution: Point estimate + 5 quantiles + 4 regime probabilities + confidence
- Outputs smooth probability curves instead of binary predictions
- Users can apply their own thresholds based on risk tolerance

---

## Core Components

### 1. Multi-Output Model Architecture

The system trains **three separate XGBoost models** that work together:

#### Point Estimate Model
- **Target**: VIX percentage change (-50% to +200%)
- **Objective**: Regression with MSE loss
- **Output**: Single point prediction of expected VIX movement

#### Quantile Models (5 models)
- **Targets**: 10th, 25th, 50th, 75th, 90th percentiles
- **Objective**: Quantile regression with pinball loss
- **Output**: Full uncertainty band around prediction

#### Regime Classification Model
- **Target**: Which regime VIX will occupy (Low/Normal/Elevated/Crisis)
- **Objective**: Multi-class classification
- **Output**: Probability for each of 4 regimes

### 2. VIXDistribution Object

Every prediction returns a `VIXDistribution` object containing:

```python
{
    'point_estimate': -0.08,          # -8% expected change
    'quantiles': {
        0.10: -0.25,                  # 10% chance of -25% or worse
        0.25: -0.15,                  # 25% chance of -15% or worse
        0.50: -0.08,                  # Median prediction
        0.75: 0.05,                   # 75% chance of +5% or worse
        0.90: 0.20                    # 90% chance of +20% or worse
    },
    'regime_probs': {
        'Low Vol': 0.15,              # 15% chance of Low Vol regime
        'Normal': 0.60,               # 60% chance Normal
        'Elevated': 0.23,             # 23% chance Elevated
        'Crisis': 0.02                # 2% chance Crisis
    },
    'confidence': 0.82,               # Overall confidence (0-1)
    'feature_quality': 0.91,          # Feature availability (0-1)
    'calendar_context': 'opex_week', # Calendar timing
    'timestamp': '2025-11-09',
    'iqr': 0.20,                      # Interquartile range (uncertainty)
    'tail_risk': 0.28                 # 90th - 50th (right tail)
}
```

### 3. Calendar-Aware Training

Instead of encoding OpEx cycles as features, the system uses **cohort-based learning**:

- **OpEx Week**: Data from 7 days before monthly expiration
- **Post-OpEx**: Week after expiration
- **Mid-Cycle**: Days 8-21 after expiration
- **Quarter End**: Last week of quarterly months

The same feature set produces context-aware distributions - the model learns "what does VIX look like during OpEx week?" vs "mid-cycle?"

### 4. Confidence Scoring

Each prediction includes a confidence score based on:

```python
confidence = 0.6 × feature_quality + 0.4 × regime_certainty

# feature_quality: proportion of features with valid data
# regime_certainty: 1 - entropy(regime_probs)
```

This degrades gracefully when:
- CBOE data is missing
- Macro indicators are stale
- Regime probabilities are highly uncertain

### 5. Predictions Database

Every prediction is stored with full provenance:

```json
{
  "timestamp": "2025-11-09T16:00:00",
  "point_estimate": -0.08,
  "quantiles": {...},
  "regime_probs": {...},
  "confidence": 0.82,
  "feature_quality": 0.91,
  "calendar_context": "opex_week"
}
```

This enables:
- Walk-forward backtesting
- Quantile calibration analysis
- Confidence score validation
- Feature quality monitoring

---

## Usage Examples

### Training the System

```bash
# Full pipeline with 50 Optuna trials
python integrated_system_production.py \
    --mode xgboost_full \
    --optimize 50 \
    --horizons 5
```

This will:
1. Build 232 features with temporal safety
2. Select optimal feature subset (typically ~48 features)
3. Train point estimate model
4. Train 5 quantile models
5. Train regime classifier
6. Save all models to `./models/`

### Making Predictions

```python
from integrated_system_production import IntegratedMarketSystemV4

# Initialize and train
system = IntegratedMarketSystemV4()
system.train(years=15, real_time_vix=False, enable_anomaly=False)

# Run feature selection
selection = system.run_feature_selection(horizons=[5])

# Train probabilistic models
system.train_xgboost_models(
    selected_features=selection['selected_features'],
    horizons=[5]
)

# Get current forecast
distribution = system.get_vix_distribution_forecast()

print(f"Point: {distribution['point_estimate']:+.2%}")
print(f"90th percentile: {distribution['quantiles']['0.9']:+.2%}")
print(f"Crisis probability: {distribution['regime_probs']['Crisis']:.1%}")
print(f"Confidence: {distribution['confidence']:.1%}")
```

### Using Distributions for Decisions

```python
# Risk Manager: Use 90th percentile for worst-case planning
worst_case = distribution.quantiles[0.90]
required_capital = position_size * (1 + worst_case) * risk_factor

# Options Trader: Identify asymmetric tail risk
tail_prob = distribution.get_tail_probability(threshold=0.30)
if tail_prob > 0.15 and distribution.regime_probs['Crisis'] < 0.05:
    # Cheap tail insurance opportunity
    buy_otm_calls()

# Regime Forecaster: Trigger allocation change
if distribution.regime_probs['Crisis'] > 0.20:
    reduce_equity_exposure()
```

---

## Technical Details

### Loss Function Components

The training optimizes a weighted multi-objective loss:

```python
total_loss = (
    0.25 × MSE(point_estimate) +           # Point accuracy
    0.35 × PinballLoss(quantiles) +        # Quantile calibration
    0.25 × LogLoss(regime_probs) +         # Regime classification
    0.15 × CalibrationPenalty()            # Overconfidence penalty
)
```

### Quantile Regression (Pinball Loss)

For quantile τ:

```python
pinball_loss(y_true, y_pred, τ) = {
    τ × (y_true - y_pred)         if y_true ≥ y_pred
    (1 - τ) × (y_pred - y_true)  otherwise
}
```

This asymmetric loss penalizes underestimation more for high quantiles (90th) and overestimation more for low quantiles (10th).

### Calendar Context Detection

```python
def identify_calendar_context(date):
    # Monthly OpEx is 3rd Friday of each month
    third_friday = calculate_third_friday(date.month, date.year)
    
    days_until_opex = (third_friday - date).days
    days_after_opex = (date - third_friday).days
    
    if date.month in [3,6,9,12] and date.day >= 23:
        return 'quarter_end'
    elif 0 < days_until_opex <= 7:
        return 'opex_week'
    elif 0 < days_after_opex <= 7:
        return 'post_opex'
    else:
        return 'mid_cycle'
```

---

## File Structure

### Updated Files

1. **config.py**
   - Added `FORECAST_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]`
   - Added `CALENDAR_CONTEXTS` definition
   - Added `PROBABILISTIC_LOSS_WEIGHTS`
   - Removed binary `expansion_threshold`

2. **xgboost_trainer_v2.py** (Complete rewrite)
   - New `VIXDistribution` class
   - New `ProbabilisticVIXForecaster` class
   - Methods: `train()`, `predict()`, `predict_batch()`
   - Calendar-aware CV splits
   - Multi-model training with quantile regression

3. **integrated_system_production.py**
   - Updated `train_xgboost_models()` API
   - New `get_vix_distribution_forecast()` method
   - Removed `expansion_threshold` parameter
   - Updated CLI to remove `--threshold` argument

### New Saved Models

After training, you'll find in `./models/`:

```
vix_point_estimate_5d.pkl          # Main regression model
vix_quantile_10_5d.pkl             # 10th percentile model
vix_quantile_25_5d.pkl             # 25th percentile model
vix_quantile_50_5d.pkl             # 50th percentile (median)
vix_quantile_75_5d.pkl             # 75th percentile model
vix_quantile_90_5d.pkl             # 90th percentile model
vix_regime_classifier_5d.pkl       # Regime classifier
probabilistic_validation_metrics.json  # CV performance
predictions_database.json          # All historical predictions
feature_importance_5d.csv          # Feature rankings
```

---

## Performance Expectations

### Typical CV Metrics

**Point Estimate:**
- RMSE: 0.08-0.12 (8-12% error on VIX % change)

**Quantiles:**
- Well-calibrated if empirical coverage ≈ theoretical
- Example: 10% of outcomes should be below 10th percentile

**Regime Classification:**
- Log Loss: 0.5-0.8
- Accuracy: 60-75% (better than base rate)

### Confidence Score Interpretation

- **0.9-1.0**: High confidence - all features present, clear regime
- **0.7-0.9**: Good confidence - most features, regime uncertainty
- **0.5-0.7**: Moderate confidence - missing features or regime unclear
- **<0.5**: Low confidence - significant data issues

---

## Advantages Over Binary Classification

1. **No Arbitrary Thresholds**: Users choose their own risk cutoffs
2. **Quantified Uncertainty**: IQR and tail risk metrics
3. **Calendar Awareness**: Built into training, not features
4. **Graceful Degradation**: Confidence scores track data quality
5. **Rich Information**: One model → multiple use cases
6. **Better Calibration**: Probabilistic training objectives
7. **Tail Risk Focus**: 90th percentile for worst-case planning

---

## Migration from Old System

### Old Code
```python
# Binary prediction
trainer = VIXExpansionTrainer(expansion_threshold=0.05)
result = trainer.predict(X)
# Output: 0 or 1
```

### New Code
```python
# Probabilistic prediction
forecaster = ProbabilisticVIXForecaster()
distribution = forecaster.predict(X, return_distribution=True)

# Apply your own threshold
will_expand_5pct = distribution.get_tail_probability(0.05) > 0.50

# Or use quantiles
worst_case = distribution.quantiles[0.90]
```

### Breaking Changes

1. `expansion_threshold` parameter removed from training
2. `compute_shap` parameter removed (uses gain importance now)
3. `predict()` returns `VIXDistribution` not binary label
4. Models saved as pickle files not JSON
5. Different validation metrics (RMSE vs accuracy)

---

## Future Enhancements

### Potential Additions

1. **Multi-horizon consistency**: Ensure 1d, 3d, 5d forecasts are coherent
2. **Conditional distributions**: P(VIX | SPX drops 5%)
3. **Path sampling**: Generate full 5-day VIX trajectories
4. **Online learning**: Update models with new data
5. **Ensemble methods**: Combine multiple probabilistic models

### Monitoring Recommendations

1. Track quantile coverage over time
2. Monitor confidence score vs actual error correlation  
3. Validate calendar context impact
4. Check for regime probability calibration
5. Compare point estimate vs median quantile

---

## Questions & Troubleshooting

### "Why is confidence low?"

Check `feature_quality` - if <0.8, you're missing >20% of features (likely CBOE data). The system can still predict but warns you.

### "How to interpret tail_risk?"

`tail_risk = q90 - q50` measures right-tail asymmetry. High values suggest fat tails (crisis potential).

### "Can I use just point estimate?"

Yes, but you lose uncertainty quantification. Point estimate is unaware of its own confidence.

### "What if quantiles cross?"

Shouldn't happen due to separate models per quantile, but if it does, use isotonic regression to enforce monotonicity.

---

## System Philosophy

This architecture solves the fundamental problem: **VIX movements are continuous, not binary**. By learning the full distribution, the system provides the raw material for any downstream decision - risk managers, traders, and strategists can all query the same model for their specific needs.

The 39% precision problem disappears because there's no longer a sharp threshold. Instead, you get smooth probabilities that degrade gracefully with data quality and provide honest uncertainty estimates.

**Key insight**: Options expiration cycles don't affect VIX through some feature interaction - they fundamentally change the distribution shape. Calendar-aware training captures this structural effect directly.
