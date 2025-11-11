# UPGRADE GUIDE: config.py
## Configuration for Probabilistic Distribution Forecasting System

---

## SYSTEM CONTEXT (READ THIS FIRST)

### What We're Building
You're transforming a **binary VIX expansion classifier** into a **probabilistic distribution forecaster**. 

**Current System:**
- Predicts: "Will VIX expand >5% in 5 days?" (Yes/No)
- Problem: 39% precision - too many false positives
- Uses arbitrary threshold (5%) that doesn't reflect market reality

**New System:**
- Predicts: Complete probability distribution of VIX outcomes (-50% to +200%)
- Outputs: Point estimate, 5 quantiles, 4 regime probabilities, confidence score
- Calendar effects (OpEx, FOMC) become training contexts, not features
- Evaluated using proper probabilistic metrics (quantile coverage, Brier score)

### Architecture Overview
```
Input: 232 features (SPX, VIX, yields, macro, futures, CBOE)
       + Calendar cohort (OpEx-5, FOMC-3, Mid-cycle, etc.)
       
Process: Multi-output XGBoost trains separate models per cohort
         - Point estimate model (MSE loss)
         - 5 Quantile models (Pinball loss)  
         - Regime classifier (Log loss, 4 classes)
         - Confidence scorer (predicts forecast quality)
         
Output: Distribution object queryable by downstream apps
        - Risk manager: "Give me 90th percentile for worst-case"
        - Options trader: "What's P(VIX > 30)?"
        - Portfolio: "Regime probabilities for allocation"
```

---

## FILE ROLE: config.py

**Purpose:** Central configuration file containing all system constants, paths, and parameters.

**Current Contents:**
- CBOE ticker mappings
- FRED series codes
- Yahoo Finance symbols
- File paths (data_cache, models, json_data)
- Feature lag configurations (for temporal safety)
- XGBoost hyperparameters (for binary classifier)

**What Stays:**
- All data source configurations (CBOE, FRED, Yahoo)
- File paths
- Feature lag specifications
- Caching settings

**What Changes:**
- TARGET_CONFIG: From binary threshold → distribution parameters
- XGBOOST_CONFIG: From classifier → multi-output regressor
- Add CALENDAR_COHORTS: New concept for context-aware training
- Add PREDICTION_DB_SCHEMA: Storage format for backtesting

---

## REQUIRED CHANGES

### 1. REPLACE: Binary Target Configuration

**FIND THIS SECTION (around line 150-160):**
```python
# VIX Expansion Detection
EXPANSION_THRESHOLD = 0.05  # 5% expansion
HORIZONS = [5]  # 5-day forward looking
```

**REPLACE WITH:**
```python
# ============================================================================
# PROBABILISTIC FORECASTING TARGETS
# ============================================================================

TARGET_CONFIG = {
    # Point Estimate: VIX % change prediction
    'point_estimate': {
        'range': (-50, 200),  # Min: -50% (crash recovery), Max: +200% (black swan)
        'loss': 'reg:squarederror',
        'loss_weight': 1.0,
        'clip_extremes': True  # Cap predictions at range boundaries
    },
    
    # Quantile Predictions: Capture uncertainty and tail risk
    'quantiles': {
        'levels': [0.10, 0.25, 0.50, 0.75, 0.90],
        'loss': 'reg:quantileerror',  # Pinball loss
        'loss_weight': 1.0,
        'enforce_monotonicity': True  # q10 < q25 < q50 < q75 < q90
    },
    
    # Regime Classification: Which volatility regime at horizon?
    'regimes': {
        'boundaries': [16.77, 24.40, 39.67],  # Historical VIX quartiles
        'labels': ['Low', 'Normal', 'Elevated', 'Crisis'],
        'loss': 'multi:softprob',
        'loss_weight': 0.5,
        'num_classes': 4
    },
    
    # Confidence Scoring: How reliable is this forecast?
    'confidence': {
        'components': {
            'feature_quality': 0.5,  # Data freshness/completeness
            'regime_stability': 0.3,  # Market in transition?
            'historical_error': 0.2   # Performance in similar conditions
        },
        'loss': 'reg:squarederror',
        'loss_weight': 0.3,
        'calibration_method': 'isotonic'  # Post-hoc probability calibration
    },
    
    # Forecasting Horizon
    'horizon_days': 5,
    'horizon_label': '5d'
}
```

**Why This Design:**
- `range`: Based on historical VIX behavior (check 2008, 2020 for extremes)
- `quantiles`: Standard quintiles + median for distribution shape
- `regime_boundaries`: Derived from VIX's historical quartiles (check your data!)
- `loss_weights`: Point estimate most important, regime secondary, confidence tertiary

---

### 2. ADD: Calendar Cohort Definitions

**INSERT AFTER TARGET_CONFIG:**
```python
# ============================================================================
# CALENDAR COHORTS - Context-Aware Training
# ============================================================================
# Instead of encoding calendar as features, we train separate models for
# different market contexts (OpEx cycles, FOMC meetings, earnings seasons)

CALENDAR_COHORTS = {
    # Monthly Options Expiration (3rd Friday)
    'monthly_opex_minus_5': {
        'condition': 'days_to_monthly_opex',
        'range': (-7, -3),
        'weight': 1.2,  # Slightly higher uncertainty pre-OpEx
        'description': 'Week before monthly options expiration'
    },
    'monthly_opex_minus_1': {
        'condition': 'days_to_monthly_opex',
        'range': (-2, 0),
        'weight': 1.5,  # High gamma exposure
        'description': 'Immediate pre-expiration (Wed-Fri)'
    },
    'monthly_opex_plus_1': {
        'condition': 'days_to_monthly_opex',
        'range': (1, 3),
        'weight': 1.1,  # Post-expiration rebalancing
        'description': 'Days after monthly expiration'
    },
    
    # FOMC Meeting Cycles
    'fomc_minus_3': {
        'condition': 'days_to_fomc',
        'range': (-5, -1),
        'weight': 1.3,
        'description': 'Pre-FOMC positioning (Mon-Wed before meeting)'
    },
    'fomc_week': {
        'condition': 'days_to_fomc',
        'range': (0, 2),
        'weight': 1.4,  # Highest uncertainty
        'description': 'FOMC decision day + 2 days after'
    },
    
    # Earnings Season Intensity
    'earnings_heavy': {
        'condition': 'spx_earnings_pct',
        'range': (0.15, 1.0),  # >15% of SPX reporting this week
        'weight': 1.1,
        'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'
    },
    
    # Quarterly Futures Rollover (H, M, U, Z months)
    'futures_rollover': {
        'condition': 'days_to_futures_expiry',
        'range': (-5, 0),
        'weight': 1.15,
        'description': 'VIX futures expiration week'
    },
    
    # Default: No special calendar effects
    'mid_cycle': {
        'condition': 'default',
        'range': None,
        'weight': 1.0,
        'description': 'Regular market conditions (no major calendar events)'
    }
}

# Priority Order (checked top-to-bottom)
# If a date matches multiple cohorts, use the first match
COHORT_PRIORITY = [
    'fomc_week',           # Highest priority
    'fomc_minus_3',
    'monthly_opex_minus_1',
    'monthly_opex_minus_5',
    'futures_rollover',
    'monthly_opex_plus_1',
    'earnings_heavy',
    'mid_cycle'            # Catch-all
]
```

**Why Cohorts vs Features:**
- Traditional ML: Create `days_to_opex` as a feature → Model struggles to learn nonlinear calendar effects
- Cohort approach: Train separate distribution for "5 days before OpEx" → Model learns this context has different VIX dynamics
- Example: VIX typically compresses before OpEx, then spikes after (gamma unwind). Cohorts capture this without complex feature engineering.

---

### 3. REPLACE: XGBoost Configuration

**FIND THIS SECTION (around line 180-200):**
```python
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'scale_pos_weight': 4.5,  # For imbalanced binary classification
    ...
}
```

**REPLACE WITH:**
```python
# ============================================================================
# XGBOOST MULTI-OUTPUT CONFIGURATION
# ============================================================================

XGBOOST_CONFIG = {
    # Model Architecture
    'strategy': 'separate_models',  # Train separate model per output type
    'cohort_aware': True,  # Train per calendar cohort
    
    # Shared Hyperparameters (apply to all models)
    'shared_params': {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'gamma': 0.1,  # Minimum loss reduction for split
        'seed': 42,
        'n_jobs': -1
    },
    
    # Model-Specific Objectives
    'objectives': {
        'point': {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'early_stopping_rounds': 50
        },
        'quantile_10': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.10,
            'eval_metric': 'mae',
            'early_stopping_rounds': 50
        },
        'quantile_25': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.25,
            'eval_metric': 'mae',
            'early_stopping_rounds': 50
        },
        'quantile_50': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.50,
            'eval_metric': 'mae',
            'early_stopping_rounds': 50
        },
        'quantile_75': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.75,
            'eval_metric': 'mae',
            'early_stopping_rounds': 50
        },
        'quantile_90': {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.90,
            'eval_metric': 'mae',
            'early_stopping_rounds': 50
        },
        'regime': {
            'objective': 'multi:softprob',
            'num_class': 4,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50
        },
        'confidence': {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'early_stopping_rounds': 50
        }
    },
    
    # Cross-Validation Strategy
    'cv_config': {
        'method': 'time_series_split',  # Respects temporal ordering
        'n_splits': 5,
        'test_size': 0.2,
        'gap': 5  # Gap between train/test to prevent leakage
    }
}
```

**Key Changes:**
- Removed `scale_pos_weight` (no longer binary classification)
- Added quantile-specific objectives (one per quantile)
- Regime uses `multi:softprob` (4-class probabilities)
- All use proper time-series CV (no random shuffling!)

---

### 4. ADD: Prediction Storage Schema

**INSERT AFTER XGBOOST_CONFIG:**
```python
# ============================================================================
# PREDICTION DATABASE SCHEMA
# ============================================================================
# Every prediction stored for walk-forward backtesting

PREDICTION_DB_CONFIG = {
    'db_path': 'data_cache/predictions.db',
    'table_name': 'forecasts',
    
    'schema': {
        # Identifiers
        'prediction_id': 'TEXT PRIMARY KEY',  # UUID
        'timestamp': 'DATETIME',  # When prediction was made
        'forecast_date': 'DATE',  # Date being forecasted (timestamp + horizon)
        'horizon': 'INTEGER',  # Always 5 for now
        
        # Context
        'calendar_cohort': 'TEXT',  # Which cohort model was used
        'cohort_weight': 'REAL',  # Uncertainty weight from cohort
        
        # Predictions
        'point_estimate': 'REAL',  # Expected VIX % change
        'q10': 'REAL',  # 10th percentile
        'q25': 'REAL',  # 25th percentile
        'q50': 'REAL',  # Median (50th percentile)
        'q75': 'REAL',  # 75th percentile
        'q90': 'REAL',  # 90th percentile
        'prob_low': 'REAL',  # P(VIX < 16.77)
        'prob_normal': 'REAL',  # P(16.77 <= VIX < 24.40)
        'prob_elevated': 'REAL',  # P(24.40 <= VIX < 39.67)
        'prob_crisis': 'REAL',  # P(VIX >= 39.67)
        'confidence_score': 'REAL',  # [0, 1] forecast quality
        
        # Metadata
        'feature_quality': 'REAL',  # Data freshness score
        'regime_stability': 'REAL',  # Market transition indicator
        'num_features_used': 'INTEGER',  # How many features available
        'missing_features': 'TEXT',  # JSON list of missing features
        
        # Actuals (filled post-hoc for backtesting)
        'actual_vix_change': 'REAL',  # Realized % change
        'actual_regime': 'TEXT',  # Actual regime at horizon
        'point_error': 'REAL',  # |actual - point_estimate|
        'quantile_coverage': 'TEXT',  # JSON: which quantiles covered actual
        
        # Provenance
        'features_used': 'TEXT',  # JSON dict of feature values
        'model_version': 'TEXT',  # Git hash or version tag
        'created_at': 'DATETIME'  # Database insert time
    },
    
    'indexes': [
        'CREATE INDEX idx_timestamp ON forecasts(timestamp)',
        'CREATE INDEX idx_cohort ON forecasts(calendar_cohort)',
        'CREATE INDEX idx_forecast_date ON forecasts(forecast_date)'
    ]
}

# Backtesting Queries (for reference)
BACKTEST_QUERIES = {
    'quantile_coverage': '''
        SELECT 
            calendar_cohort,
            AVG(CASE WHEN actual_vix_change <= q10 THEN 1 ELSE 0 END) as coverage_10,
            AVG(CASE WHEN actual_vix_change <= q25 THEN 1 ELSE 0 END) as coverage_25,
            AVG(CASE WHEN actual_vix_change <= q50 THEN 1 ELSE 0 END) as coverage_50,
            AVG(CASE WHEN actual_vix_change <= q75 THEN 1 ELSE 0 END) as coverage_75,
            AVG(CASE WHEN actual_vix_change <= q90 THEN 1 ELSE 0 END) as coverage_90,
            COUNT(*) as n_predictions
        FROM forecasts
        WHERE actual_vix_change IS NOT NULL
        GROUP BY calendar_cohort
    ''',
    
    'regime_brier_score': '''
        SELECT
            calendar_cohort,
            AVG(
                POWER(prob_low - (actual_regime = 'Low'), 2) +
                POWER(prob_normal - (actual_regime = 'Normal'), 2) +
                POWER(prob_elevated - (actual_regime = 'Elevated'), 2) +
                POWER(prob_crisis - (actual_regime = 'Crisis'), 2)
            ) as brier_score
        FROM forecasts
        WHERE actual_regime IS NOT NULL
        GROUP BY calendar_cohort
    '''
}
```

**Why This Schema:**
- Stores full distribution (not just point estimate)
- Tracks feature quality (for confidence calibration)
- Includes actuals for probabilistic evaluation (quantile coverage, Brier score)
- Cohort-stratified backtesting (does FOMC model work better?)

---

### 5. ADD: Feature Metadata Updates

**FIND SECTION:**
```python
FEATURE_LAGS = {
    'CPI': 30,
    'GDP': 90,
    ...
}
```

**ADD BELOW IT:**
```python
# ============================================================================
# FEATURE QUALITY THRESHOLDS
# ============================================================================
# Used by temporal_validator to compute feature_quality score

FEATURE_QUALITY_CONFIG = {
    'staleness_penalty': {
        'none': 1.0,      # Updated today
        'minor': 0.95,    # 1-3 days stale
        'moderate': 0.80, # 4-7 days stale
        'severe': 0.50,   # 8-14 days stale
        'critical': 0.20  # >14 days stale
    },
    
    'missingness_penalty': {
        'critical_features': [  # Zero tolerance for missing
            'vix', 'spx', 'vix_percentile_21d', 'spx_realized_vol_21d'
        ],
        'important_features': [  # 0.5x weight if missing
            'VX1-VX2', 'SKEW', 'yield_10y2y', 'Dollar_Index'
        ],
        'optional_features': [  # 0.9x weight if missing
            'GAMMA', 'VPN', 'BFLY'
        ]
    },
    
    'quality_thresholds': {
        'excellent': 0.95,  # Forecast with high confidence
        'good': 0.85,       # Forecast normally
        'acceptable': 0.70, # Flag as degraded
        'poor': 0.50,       # Warn user
        'unusable': 0.30    # Refuse to forecast
    }
}
```

---

## INTEGRATION POINTS

**This config file is imported by:**

1. **feature_engine.py**
   - Uses `CALENDAR_COHORTS` to assign cohort per date
   - Uses `FEATURE_LAGS` for temporal safety
   - Needs: `TARGET_CONFIG['horizon_days']`

2. **xgboost_trainer_v2.py**
   - Uses `XGBOOST_CONFIG` for training
   - Uses `TARGET_CONFIG` to create targets
   - Needs: `CALENDAR_COHORTS` for cohort-wise training

3. **integrated_system_production.py**
   - Uses `PREDICTION_DB_CONFIG` to store forecasts
   - Needs: All configs for orchestration

4. **temporal_validator.py**
   - Uses `FEATURE_QUALITY_CONFIG` to score features
   - Uses `FEATURE_LAGS` to check staleness

5. **backtesting_engine.py** (new file)
   - Uses `PREDICTION_DB_CONFIG` to query predictions
   - Uses `BACKTEST_QUERIES` for evaluation

---

## VALIDATION CHECKLIST

After making changes, verify:

- [ ] `TARGET_CONFIG['regimes']['boundaries']` match your VIX historical quartiles
- [ ] `CALENDAR_COHORTS` cover all market conditions (no gaps)
- [ ] `COHORT_PRIORITY` list matches `CALENDAR_COHORTS.keys()`
- [ ] `XGBOOST_CONFIG['objectives']` includes all outputs (8 total: 1 point + 5 quantiles + 1 regime + 1 confidence)
- [ ] `PREDICTION_DB_CONFIG['schema']` includes all quantiles and regime probs
- [ ] `FEATURE_QUALITY_CONFIG['critical_features']` are actually always available
- [ ] File paths in `PREDICTION_DB_CONFIG` use existing `data_cache/` directory

---

## TESTING THE CHANGES

```python
# After updating config.py, test imports:
from config import (
    TARGET_CONFIG,
    CALENDAR_COHORTS,
    COHORT_PRIORITY,
    XGBOOST_CONFIG,
    PREDICTION_DB_CONFIG,
    FEATURE_QUALITY_CONFIG
)

# Validate structure
assert len(TARGET_CONFIG['quantiles']['levels']) == 5
assert len(TARGET_CONFIG['regimes']['labels']) == 4
assert 'mid_cycle' in CALENDAR_COHORTS
assert len(XGBOOST_CONFIG['objectives']) == 8
print("✅ Config validation passed")
```

---

## COMMON PITFALLS

1. **Quantile Ordering:** Ensure q10 < q25 < q50 < q75 < q90 in predictions (enforce in trainer)
2. **Regime Boundaries:** Must align with actual VIX distribution (check your data!)
3. **Cohort Gaps:** Every date must match exactly one cohort (test with edge cases)
4. **Loss Weights:** Sum doesn't need to equal 1.0, but ratios matter (point > quantiles > regime > confidence)
5. **Database Indexes:** Add indexes AFTER schema creation, not in schema definition

---

## EXAMPLE: How Cohorts Will Be Used

```python
# In feature_engine.py
date = pd.Timestamp('2025-01-15')
days_to_opex = 2  # Friday is OpEx
cohort, weight = get_calendar_cohort(date)
# Returns: ('monthly_opex_minus_1', 1.5)

# In xgboost_trainer_v2.py
df_opex = df[df['calendar_cohort'] == 'monthly_opex_minus_1']
model_opex = train_model(df_opex, config=XGBOOST_CONFIG)
# Trains separate model just for OpEx-1 dates

# In prediction
distribution = model_opex.predict(today_features)
distribution['confidence_score'] *= weight  # Adjust for cohort uncertainty
```

This is why cohorts live in config: They're used across feature_engine, trainer, and integrated_system.

---

## SUMMARY OF CHANGES

**Lines to Add:** ~200 lines (TARGET_CONFIG, CALENDAR_COHORTS, XGBOOST_CONFIG updates, PREDICTION_DB_CONFIG)

**Lines to Remove:** ~20 lines (EXPANSION_THRESHOLD, binary classifier params)

**Net Change:** +180 lines

**Backward Compatibility:** None (this is a breaking change). Old binary models will not work with new config.

**Next Steps After This File:**
1. Update `feature_engine.py` to use `CALENDAR_COHORTS`
2. Update `xgboost_trainer_v2.py` to use new `TARGET_CONFIG`
3. Create `prediction_database.py` using `PREDICTION_DB_CONFIG`
