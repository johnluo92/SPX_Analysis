# System Transformation: Binary Classifier → Probabilistic Distribution Forecaster

## Executive Summary

This document guides the transformation of a binary VIX expansion classifier (predicts: will VIX rise >5%?) into a sophisticated probabilistic distribution forecaster that outputs the full probability distribution of VIX outcomes over the next 5 days.

**Current System:**
- Binary classification: Expansion (1) vs No Expansion (0)
- Single threshold (5%)
- Output: probability of crossing threshold
- Metrics: Accuracy 58.8%, Recall 91.2%, Precision 36.1%
- Problem: 39% precision means 61% false positives

**Target System:**
- Probabilistic distribution forecasting
- Outputs: point estimate, 5 quantiles, 4 regime probabilities, confidence score
- No arbitrary thresholds - users apply their own
- Enables: risk management, options positioning, regime allocation
- Evaluation: quantile calibration, regime accuracy, confidence correlation

## System Architecture Overview

### Core Components (7 Files to Modify + 3 New Files)

**Modified Files:**
1. `config.py` - Add regime definitions, quantile levels, loss weights
2. `xgboost_trainer_v2.py` - Overhaul to multi-output architecture
3. `integrated_system_production.py` - Update orchestration for new outputs
4. `xgboost_feature_selector_v2.py` - Adapt for multi-target selection
5. `feature_engine.py` - Minor: add calendar context metadata
6. `temporal_validator.py` - Minor: validate quantile consistency
7. `data_fetcher.py` - Unchanged (already solid)

**New Files:**
1. `probabilistic_trainer.py` - Multi-output XGBoost with custom loss
2. `distribution_predictor.py` - Inference engine for probability distributions
3. `predictions_database.py` - Store predictions with full provenance

### Data Flow

```
Market Data (feature_engine.py)
    ↓ [232 features]
Feature Selection (xgboost_feature_selector_v2.py)
    ↓ [48 selected features]
Calendar Context Splitter
    ↓ [5d before OpEx | mid-cycle | post-OpEx]
Probabilistic Trainer (probabilistic_trainer.py)
    ↓ [Multi-output XGBoost with 5 objectives]
Distribution Predictor (distribution_predictor.py)
    ↓ [Point + Quantiles + Regimes + Confidence]
Predictions Database (predictions_database.py)
    ↓ [SQLite with full provenance]
Backtesting Engine
    ↓ [Walk-forward evaluation]
```

## Four Simultaneous Predictions

### 1. Point Estimate (Regression)
**Target:** Percentage change in VIX over 5 days
- Range: -50% (compression) to +200% (crisis)
- Loss: Mean Squared Error (MSE)
- Use: Expected value for planning

### 2. Quantile Predictions (Quantile Regression)
**Targets:** 5 quantiles at [10%, 25%, 50%, 75%, 90%]
- Captures full distribution shape
- Loss: Pinball loss per quantile
- Use: Risk assessment, tail events
- Constraint: Quantiles must be monotonic (q10 ≤ q25 ≤ ... ≤ q90)

### 3. Regime Classification (Multi-class)
**Targets:** 4 regime probabilities summing to 1.0
- Low Vol: VIX < 16.77
- Normal: 16.77 ≤ VIX < 24.40
- Elevated: 24.40 ≤ VIX < 39.67
- Crisis: VIX ≥ 39.67
- Loss: Multi-class log loss
- Use: Allocation decisions, risk budgeting

### 4. Confidence Score (Meta-prediction)
**Factors:**
- Feature availability (missing CBOE data reduces confidence)
- Feature staleness (outdated FRED data reduces confidence)
- Regime stability (transitions have lower confidence)
- Historical error in similar conditions
- Range: 0.0 (no confidence) to 1.0 (high confidence)

## Calendar Context (Not Features)

**Key Innovation:** Calendar effects are conditioning contexts, not features.

Instead of:
```python
features = [..., 'days_to_opex', 'is_opex_week', ...]
model.predict(features)
```

We use:
```python
contexts = ['pre_opex', 'mid_cycle', 'post_opex']
models = {ctx: train_model(data[data.context == ctx]) for ctx in contexts}
prediction = models[current_context].predict(features)
```

**Rationale:** 
- VIX dynamics differ fundamentally around OpEx
- Same features produce different distributions in different contexts
- Avoids complex interaction features
- Models learn "what does VIX do given these features in THIS context?"

## Multi-Output XGBoost Architecture

### Joint Training with Weighted Loss

```python
total_loss = (
    w1 * mse_loss(point_estimate) +           # Main prediction
    w2 * sum(pinball_loss(quantile_i)) +      # Distribution shape
    w3 * logloss(regime_probs) +              # Discrete outcomes
    w4 * calibration_penalty(confidence)       # Meta-accuracy
)
```

**Implementation Strategy:**
- Use XGBoost multi-output capability (train 9 targets simultaneously)
- Custom objective function combining losses
- Shared tree structure learns common patterns
- Task-specific heads for final predictions

### Targets Array Shape
```
[point_estimate, q10, q25, q50, q75, q90, regime_0, regime_1, regime_2, regime_3, confidence]
```
Shape: (n_samples, 11)

## Evaluation Framework

### Replacing Binary Metrics

**Old Metrics (binary):**
- Accuracy, Precision, Recall, F1
- Problem: Arbitrary threshold

**New Metrics (probabilistic):**
1. **Point Estimate:** MAE, RMSE, R²
2. **Quantiles:** Pinball loss, calibration plots
3. **Regimes:** Multi-class accuracy, confusion matrix
4. **Confidence:** Correlation with actual error
5. **Overall:** Continuous Ranked Probability Score (CRPS)

### Walk-Forward Backtesting

Store every prediction with:
- Timestamp
- Feature values used
- Full distribution (point + quantiles + regimes + confidence)
- Actual VIX outcome at horizon
- Calendar context
- Feature quality metrics

Enables queries like:
- "How accurate were 90th percentile predictions during crisis regimes?"
- "Did confidence scores predict actual error?"
- "Which features were most important when model was wrong?"

## Implementation Sequence

### Phase 1: Foundation (Sessions 1-3)
1. **Config upgrade** - Define regimes, quantiles, loss weights
2. **Predictions database** - Storage schema and API
3. **Feature selector adaptation** - Handle multi-output targets

### Phase 2: Core Training (Sessions 4-5)
4. **Probabilistic trainer** - Multi-output XGBoost with custom loss
5. **Distribution predictor** - Inference engine with validation

### Phase 3: Integration (Sessions 6-7)
6. **Orchestrator upgrade** - Update integrated_system_production.py
7. **Backtesting engine** - Walk-forward evaluation framework

### Phase 4: Refinement (Sessions 8-9)
8. **Calendar context splitter** - Separate models for OpEx cycles
9. **Confidence calibration** - Meta-model for confidence scores

## File Modification Complexity

| File | Lines Changed | Difficulty | Dependencies |
|------|---------------|------------|--------------|
| config.py | ~50 | Low | None |
| predictions_database.py | ~300 (new) | Medium | config.py |
| probabilistic_trainer.py | ~600 (new) | High | config.py, xgboost_trainer_v2.py |
| distribution_predictor.py | ~400 (new) | Medium | config.py, probabilistic_trainer.py |
| xgboost_feature_selector_v2.py | ~100 | Medium | config.py |
| integrated_system_production.py | ~200 | Medium | All above |
| feature_engine.py | ~20 | Low | config.py |
| temporal_validator.py | ~30 | Low | config.py |

## Success Criteria

### Quantitative
- Quantile calibration: 10th percentile should contain 10% of outcomes (±2%)
- Regime accuracy: >60% correct regime prediction
- Confidence correlation: r > 0.5 between confidence and inverse error
- CRPS: <15% (lower is better)

### Qualitative
- Single model run produces actionable distribution
- Users can impose their own risk thresholds
- Predictions traceable to feature provenance
- System runs end-to-end without intervention

## Key Design Principles

1. **No arbitrary thresholds** - Users decide what's risky
2. **Calendar as context** - Not features, but training splits
3. **Provenance tracking** - Every prediction fully auditable
4. **Probabilistic evaluation** - Calibration over accuracy
5. **Feature quality awareness** - Confidence degrades with missing data
6. **Regime-conditional** - Learn separate dynamics per volatility regime

## Next Steps

Read the individual file upgrade guides in this order:
1. `CONFIG_UPGRADE.md` - Foundation constants
2. `PREDICTIONS_DATABASE.md` - Storage layer
3. `PROBABILISTIC_TRAINER.md` - Core training logic
4. `DISTRIBUTION_PREDICTOR.md` - Inference engine
5. `FEATURE_SELECTOR_UPGRADE.md` - Multi-output adaptation
6. `ORCHESTRATOR_UPGRADE.md` - Integration
7. `BACKTESTING_ENGINE.md` - Evaluation framework

Each guide includes:
- Required context files for LLM session
- Detailed specifications
- Implementation examples
- Testing requirements
- Common pitfalls to avoid
