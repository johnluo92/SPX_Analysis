# Configuration Documentation

## Directory Configuration
- **CACHE_DIR**: Location for cached data files (`./data_cache`)
- **CBOE_DATA_DIR**: Archive directory for CBOE historical data (`./CBOE_Data_Archive`)

## Training Configuration
- **ENABLE_TRAINING**: Toggle for model training mode
- **TRAINING_YEARS**: Historical data window (15 years)
- **RANDOM_STATE**: Seed for reproducibility (42)
- **TRAINING_END_DATE**: Last complete month (auto-calculated)
- **TRAINING_START_DATE**: Start of training period (auto-calculated with 450-day buffer)

## Data Split Configuration - Single Source of Truth
All data splits used throughout the system are defined here. Never hardcode split logic elsewhere.

### Split Dates
- **train_end_date**: 2021-12-31 - Used for model training
- **val_end_date**: 2023-12-31 - Used for hyperparameter tuning and early stopping
- **Test set**: Implicitly starts 2024-01-01 onwards (everything after val_end_date)
- **feature_selection_split_date**: 2023-12-31 (matches val_end_date)

### Split Descriptions
- **Train**: Training data up to 2021-12-31
- **Val**: Validation data from 2022-01-01 to 2023-12-31
- **Test**: Test data from 2024-01-01 onwards
- **Feature Selection**: Uses Train+Val (up to 2023-12-31), excludes Test

### Legacy Compatibility
- TRAIN_END_DATE and VAL_END_DATE maintained for backward compatibility (will be deprecated)

## Calibration Configuration
- **CALIBRATION_WINDOW_DAYS**: Rolling window for calibration (252 trading days)
- **CALIBRATION_DECAY_LAMBDA**: Exponential decay factor (0.125)
- **MIN_SAMPLES_FOR_CORRECTION**: Minimum samples required for bias correction (50)
- **MODEL_VALIDATION_MAE_THRESHOLD**: Maximum acceptable MAE (0.20)

## Target Configuration
Defines prediction target structure and transformations.

- **horizon_days**: Prediction horizon (5 business days)
- **horizon_label**: Human-readable label ("5d")
- **target_type**: Training space target type (log_vix_change)
- **output_type**: Output space type (vix_pct_change)

### Log Space Configuration
- **enabled**: Train on log(future_vix/current_vix), convert to percentage for output
- **description**: Ensures model learns multiplicative relationships

### Movement Bounds
- **floor**: Minimum percentage change (-50.0%)
- **ceiling**: Maximum percentage change (100.0%)
- **description**: Converted from log-space predictions

## Cohort & Event Configuration

### Cohort Priority
Order of cohort assignment: fomc_period → opex_week → earnings_heavy → mid_cycle

### Macro Event Configuration
Defines detection windows for major market events:

- **cpi_release**: Day 12 of month ±2 days
- **pce_release**: Day 28 of month ±3 days
- **fomc_minutes**: 21 days after FOMC meeting ±2 days
- **fomc_meeting**: 7 days before to 2 days after meeting

### Calendar Cohorts
Market condition cohorts with associated weights:

#### FOMC Period (Weight: 1.3333)
- **Condition**: macro_event_period
- **Range**: -7 to +2 days
- **Description**: FOMC meetings, CPI releases, PCE releases, FOMC minutes

#### OPEX Week (Weight: 1.1350)
- **Condition**: days_to_monthly_opex
- **Range**: -7 to 0 days
- **Description**: Options expiration week + VIX futures rollover

#### Earnings Heavy (Weight: 1.0464)
- **Condition**: spx_earnings_pct
- **Range**: 0.15 to 1.0 (15% to 100% of S&P 500 reporting)
- **Description**: Peak earnings season (Jan, Apr, Jul, Oct)

#### Mid Cycle (Weight: 1.0)
- **Condition**: default
- **Range**: None
- **Description**: Regular market conditions

## Model Configuration

### Direction Calibration
- **enabled**: Apply isotonic regression calibration
- **method**: isotonic
- **min_samples**: Minimum samples for calibration (100)
- **out_of_bounds**: Handling for OOB values (clip)
- **description**: Calibrate direction probabilities to match true accuracy rates

### XGBoost Strategy
- **strategy**: dual_model (separate magnitude and direction models)
- **cohort_aware**: Currently disabled
- **cv_config**: Time series split with 5 folds and 5-day gap

### Model Objectives
- magnitude_5d: Regression for magnitude prediction
- direction_5d: Classification for direction prediction

## Feature Selection Configuration

### CV Parameters (Tuned via Optimization)
- **n_estimators**: 139
- **max_depth**: 4
- **learning_rate**: 0.0714
- **subsample**: 0.7990
- **colsample_bytree**: 0.9004

### Selection Parameters
- **magnitude_top_n**: Top 78 features for magnitude model
- **direction_top_n**: Top 80 features for direction model
- **cv_folds**: 5-fold cross-validation
- **protected_features**: Always included (FOMC, OPEX, earnings indicators)
- **correlation_threshold**: Maximum feature correlation (0.9157)
- **target_overlap**: Target overlap between models (0.5242)

### Quality Filter
- **enabled**: Active
- **min_threshold**: Minimum quality score (0.6142)
- **warn_pct**: Warning threshold (20% poor quality)
- **error_pct**: Error threshold (50% poor quality)
- **strategy**: raise (fail on poor quality)

## Ensemble Configuration
Combines magnitude and direction models with agreement-based confidence.

### Core Settings
- **enabled**: Active
- **reconciliation_method**: weighted_agreement
- **min_ensemble_confidence**: Minimum confidence for predictions (0.50)
- **actionable_threshold**: High-confidence threshold (0.65)

### Confidence Weights
- **magnitude**: 0.3710
- **direction**: 0.4064
- **agreement**: 0.1641

### Magnitude Thresholds
- **small**: 2.5314% VIX change
- **medium**: 6.6030% VIX change
- **large**: 12.2186% VIX change

### Agreement Bonus
- **strong**: +0.1666 to confidence
- **moderate**: +0.1147 to confidence
- **weak**: No bonus

### Contradiction Penalty
- **severe**: -0.3042 to confidence
- **moderate**: -0.1871 to confidence
- **minor**: -0.0334 to confidence

## XGBoost Model Parameters

### Magnitude Model (Regression)
Optimized for predicting VIX percentage change magnitude.

- **objective**: reg:squarederror
- **eval_metric**: rmse
- **max_depth**: 3
- **learning_rate**: 0.0627
- **n_estimators**: 201
- **subsample**: 0.7605
- **colsample_bytree**: 0.8673
- **colsample_bylevel**: 0.7007
- **min_child_weight**: 8
- **reg_alpha**: 1.1951 (L1 regularization)
- **reg_lambda**: 5.7459 (L2 regularization)
- **gamma**: 0.1265
- **early_stopping_rounds**: 50
- **seed**: 42
- **n_jobs**: -1 (use all cores)

### Direction Model (Classification)
Optimized for predicting VIX movement direction (up/down).

- **objective**: binary:logistic
- **eval_metric**: logloss
- **max_depth**: 8
- **learning_rate**: 0.0303
- **n_estimators**: 578
- **subsample**: 0.8532
- **colsample_bytree**: 0.6581
- **min_child_weight**: 15
- **reg_alpha**: 1.3042 (L1 regularization)
- **reg_lambda**: 4.6933 (L2 regularization)
- **gamma**: 0.5869
- **scale_pos_weight**: 1.2959 (handles class imbalance)
- **max_delta_step**: 3
- **early_stopping_rounds**: 50
- **seed**: 42
- **n_jobs**: -1 (use all cores)

## Diversity Configuration
Ensures magnitude and direction models learn complementary patterns.

### Settings
- **enabled**: Active
- **target_feature_jaccard**: Target Jaccard similarity (0.40)
- **target_feature_overlap**: Target overlap ratio (0.5242)
- **diversity_weight**: Weight for diversity in optimization (1.5000)

### Achieved Metrics
- **feature_jaccard**: 0.344 (actual Jaccard similarity)
- **feature_overlap**: 0.519 (actual overlap)
- **pred_correlation**: 0.238 (prediction correlation)
- **overall_diversity**: 0.626 (composite diversity score)

## Prediction Database Configuration
SQLite database for storing and retrieving predictions.

### Database Settings
- **db_path**: data_cache/predictions.db
- **table_name**: forecasts
- **min_samples_for_calibration**: 50 (references MIN_SAMPLES_FOR_CORRECTION)

### Schema
Complete database schema with 24 fields:

**Identification & Timing**
- prediction_id (TEXT PRIMARY KEY)
- timestamp (DATETIME)
- observation_date (DATE)
- forecast_date (DATE)
- horizon (INTEGER)

**Cohort Information**
- calendar_cohort (TEXT)
- cohort_weight (REAL)

**Predictions**
- prob_up (REAL)
- prob_down (REAL)
- magnitude_forecast (REAL)
- expected_vix (REAL)
- direction_probability (REAL)
- direction_prediction (TEXT)

**Feature Quality**
- feature_quality (REAL)
- num_features_used (INTEGER)
- features_used (TEXT)

**Actuals & Errors**
- current_vix (REAL)
- actual_vix_change (REAL)
- actual_direction (INTEGER)
- direction_error (REAL)
- magnitude_error (REAL)
- direction_correct (INTEGER)

**Metadata**
- correction_type (TEXT)
- model_version (TEXT)
- created_at (DATETIME)

### Indexes
Optimized for common query patterns:
- idx_timestamp
- idx_observation_date
- idx_cohort
- idx_forecast_date
- idx_correction_type

## Temporal Safety Configuration
Prevents data leakage from future data.

- **ENABLE_TEMPORAL_SAFETY**: Enforces temporal constraints

### Forward Fill Limits
Maximum days to forward-fill missing data by frequency:
- **daily**: 5 days
- **weekly**: 7 days
- **monthly**: 35 days
- **quarterly**: 135 days

## Data Series Metadata

### FRED Series Metadata
Defines frequency and category for each FRED economic indicator.

**Labor Market**
- ICSA (weekly) - Initial jobless claims
- UNRATE (monthly) - Unemployment rate
- PAYEMS (monthly) - Non-farm payrolls

**Financial Stress**
- STLFSI4 (weekly) - St. Louis Fed Financial Stress Index

**Treasuries (all daily)**
- DGS1MO through DGS30 - Treasury yields (1 month to 30 years)

**Credit Spreads (all daily)**
- BAMLH0A0HYM2 - High yield master II
- BAMLH0A1HYBB through BAMLH0A3HYC - High yield by rating
- BAMLC0A0CM - Investment grade

**Funding (all daily)**
- SOFR - Secured overnight financing rate
- SOFR90DAYAVG - 90-day SOFR average

**Fed Rates**
- DFF (daily) - Federal funds rate

**Treasury Spreads (all daily)**
- T10Y2Y - 10-year minus 2-year
- T10Y3M - 10-year minus 3-month

**Inflation (all daily)**
- T5YIE - 5-year breakeven inflation
- T10YIE - 10-year breakeven inflation
- CPIAUCSL (monthly) - Consumer price index
- CPILFESL (monthly) - Core CPI
- PCEPI (monthly) - PCE price index
- PCEPILFE (monthly) - Core PCE

**Volatility**
- VIXCLS (daily) - VIX closing value

**Sentiment**
- UMCSENT (monthly) - University of Michigan consumer sentiment

**Production**
- INDPRO (monthly) - Industrial production index

**Monetary Aggregates**
- M1SL (monthly) - M1 money supply
- M2SL (monthly) - M2 money supply
- WALCL (weekly) - Fed balance sheet
- WTREGEN (weekly) - Fed treasury holdings

**Growth**
- GDP (quarterly) - Nominal GDP
- GDPC1 (quarterly) - Real GDP

### Publication Lags
Days after period end before data becomes available.

**Real-time (0 days)**
- Market data: ^GSPC, ^VIX, ^VVIX, CL=F, GC=F, DX-Y.NYB
- CBOE indices: SKEW, VIX3M, VX spreads, correlations
- Put/call ratios: PCCE, PCCI, PCC
- Funding: SOFR, SOFR90DAYAVG

**Next day (1 day)**
- Treasury yields: All DGS series
- Credit spreads: All BAML series
- Dollar index: DTWEXBGS
- Fed funds: DFF
- Treasury spreads: T10Y2Y, T10Y3M
- Inflation breakevens: T5YIE, T10YIE
- VIX closing: VIXCLS

**Weekly reports (4-7 days)**
- ICSA: 4 days (Thursday report)
- STLFSI4: 7 days
- WALCL: 4 days
- WTREGEN: 4 days
- M1SL, M2SL: 7 days

**Monthly reports (7-14 days)**
- UNRATE, PAYEMS: 7 days (first Friday)
- UMCSENT: 14 days (prelim/final)
- INDPRO: 14 days

**CPI/PCE releases**
- CPIAUCSL, CPILFESL: 14 days (mid-month)
- PCEPI, PCEPILFE: 28 days (month-end)

**Quarterly reports (90 days)**
- GDP, GDPC1: 90 days (advance/preliminary/final estimates)

## Feature Quality Configuration
Assesses data staleness and completeness.

### Staleness Penalty
Quality reduction for outdated data:
- **none**: 1.0 (no penalty)
- **minor**: 0.95
- **moderate**: 0.80
- **severe**: 0.50
- **critical**: 0.20

### Missingness Penalty
Feature importance tiers:
- **critical_features**: vix, spx, vix_percentile_21d, spx_realized_vol_21d
- **important_features**: VX1-VX2, SKEW, yield_10y2y, Dollar_Index
- **optional_features**: GAMMA, VPN, BFLY

### Quality Thresholds
- **excellent**: ≥0.95
- **good**: ≥0.85
- **acceptable**: ≥0.70
- **poor**: ≥0.50
- **unusable**: <0.30

## Regime Configuration
VIX-based market regime classification.

### Regime Boundaries
- **Low Vol**: 0 to 15.57
- **Normal**: 15.57 to 23.36
- **Elevated**: 23.36 to 31.16
- **Crisis**: 31.16 to 100

### Regime Names
- 0: Low Vol
- 1: Normal
- 2: Elevated
- 3: Crisis

## Hyperparameter Tuning Configuration
Optuna-based hyperparameter optimization (currently disabled).

### Settings
- **enabled**: False (use pre-optimized parameters)
- **method**: optuna
- **n_trials**: 500
- **cv_folds**: 5
- **timeout_hours**: 24

### Magnitude Parameter Space
- **max_depth**: 2 to 8
- **learning_rate**: 0.005 to 0.1
- **n_estimators**: 100 to 1000
- **subsample**: 0.6 to 1.0
- **colsample_bytree**: 0.6 to 1.0
- **colsample_bylevel**: 0.6 to 1.0
- **min_child_weight**: 1 to 15
- **reg_alpha**: 0.0 to 5.0
- **reg_lambda**: 0.0 to 10.0
- **gamma**: 0.0 to 2.0

### Direction Parameter Space
All magnitude parameters plus:
- **max_depth**: 3 to 10 (deeper for classification)
- **learning_rate**: 0.01 to 0.15
- **scale_pos_weight**: 0.8 to 2.0 (class imbalance handling)

### Usage
Run after ensemble implementation to fine-tune parameters for specific use cases.
