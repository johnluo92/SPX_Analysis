# System Architecture Documentation
*Generated: 2025-11-13 15:49:25*

## Overview
- Total modules analyzed: 19
- Total lines of code: 11765

## Module Structure

### core/

#### `anomaly_detector.py` (628 lines)
*Consolidated Anomaly Detection System*

**Classes:**
- `MultiDimensionalAnomalyDetector`
  - 15 independent Isolation Forests with feature importance and quality validation.
    - `__init__(self, contamination: float, random_state: int)`
    - `calculate_statistical_thresholds(self)`
    - `train(self, features: pd.DataFrame, verbose: bool)`
    - `detect(self, features: pd.DataFrame, verbose: bool)`
    - `calculate_historical_persistence_stats(self, ensemble_scores: np.ndarray, dates: Optional[pd.DatetimeIndex], ...)`

**Functions:**
- `validate_feature_quality(features: pd.DataFrame, feature_list: list, detector_name: str) -> Tuple[<ast.Tuple object at 0x100ab9e50>]`
- `calculate_robust_anomaly_score(raw_score: float, training_distribution: np.ndarray, min_percentile: float, ...) -> float`
- `calculate_coverage_penalty(coverage: float, min_coverage: float) -> float`
- `calculate_statistical_thresholds(self) -> dict`
- `train(self, features: pd.DataFrame, verbose: bool)`
- `detect(self, features: pd.DataFrame, verbose: bool) -> dict`
- `calculate_historical_persistence_stats(self, ensemble_scores: np.ndarray, dates: Optional[pd.DatetimeIndex], ...) -> Dict`
- `get_top_anomalies(self, result: dict, top_n: int) -> list`
- `get_feature_contributions(self, detector_name: str, top_n: int) -> List[Tuple[<ast.Tuple object at 0x100a8d710>]]`
- `classify_anomaly(self, score: float, method: str) -> tuple`
- `logger(self)`

**Key Dependencies:** numpy, pandas, sklearn.ensemble, sklearn.preprocessing


#### `backtesting_engine.py` (311 lines)
*Backtesting Engine for Probabilistic Forecasts*

**Classes:**
- `ProbabilisticBacktester`
  - Evaluate probabilistic forecasts stored in database.
    - `__init__(self, db_path)`
    - `run_full_evaluation(self, save_dir)`
    - `evaluate_quantile_coverage(self, df)`
    - `evaluate_brier_score(self, df)`
    - `evaluate_confidence(self, df)`

**Functions:**
- `run_full_evaluation(self, save_dir)`
- `evaluate_quantile_coverage(self, df)`
- `evaluate_brier_score(self, df)`
- `evaluate_confidence(self, df)`
- `evaluate_by_cohort(self, df)`
- `plot_diagnostics(self, df, save_dir)`
- `print_summary(self)`

**Key Dependencies:** core.prediction_database, matplotlib.pyplot, numpy, pandas


#### `data_fetcher.py` (569 lines)
**Classes:**
- `DataFetchLogger`
    - `__init__(self, name: str)`
    - `info(self, msg: str)`
    - `warning(self, msg: str)`
    - `error(self, msg: str)`
- `UnifiedDataFetcher`
    - `__init__(self, cache_dir: str, cboe_data_dir: str)`
    - `fetch_fred(self, series_id: str, start_date: str, ...)`
    - `fetch_fred_series(self, series_id: str, start_date: str, ...)`
    - `fetch_all_fred_series(self, start_date: str, end_date: str, ...)`
    - `fetch_yahoo(self, symbol: str, start_date: str, ...)`

**Functions:**
- `info(self, msg: str)`
- `warning(self, msg: str)`
- `error(self, msg: str)`
- `fetch_fred(self, series_id: str, start_date: str, ...) -> Optional[pd.Series]`
- `fetch_fred_series(self, series_id: str, start_date: str, ...) -> Optional[pd.Series]`
- `fetch_all_fred_series(self, start_date: str, end_date: str, ...) -> Dict[<ast.Tuple object at 0x10087b310>]`
- `fetch_yahoo(self, symbol: str, start_date: str, ...) -> Optional[pd.DataFrame]`
- `fetch_price(self, symbol: str) -> Optional[float]`
- `fetch_cboe_series(self, symbol: str) -> Optional[pd.Series]`
- `fetch_all_cboe(self) -> Dict[<ast.Tuple object at 0x100a6cc90>]`
- `fetch_fomc_calendar(self, start_year: int, end_year: int) -> Optional[pd.DataFrame]`
- `update_fomc_calendar_from_csv(self, csv_path: str) -> Optional[pd.DataFrame]`

**Key Dependencies:** numpy, pandas


#### `feature_engine.py` (1270 lines)
*Enhanced Feature Engine V5 - Streamlined, No Duplicates, WITH CALENDAR COHORT INTEGRATION*

**Classes:**
- `MetaFeatureEngine`
    - `extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series)`
    - `extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame)`
    - `extract_rate_of_change_features(df: pd.DataFrame)`
    - `extract_percentile_rankings(df: pd.DataFrame)`
- `FuturesFeatureEngine`
    - `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x100875dd0>])`
    - `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x1008aa590>])`
    - `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x1008b5790>])`
    - `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x1008e1450>], commodity_data: Dict[<ast.Tuple object at 0x1008e1090>], dollar_data: Dict[<ast.Tuple object at 0x1008e0cd0>], ...)`
- `TreasuryYieldFeatureEngine`
    - `extract_term_spreads(yields: pd.DataFrame)`
    - `extract_curve_shape(yields: pd.DataFrame)`
    - `extract_rate_volatility(yields: pd.DataFrame)`
- `UnifiedFeatureEngine`
    - `__init__(self, data_fetcher)`
    - `get_calendar_cohort(self, date)`
    - `apply_quality_control(self, features: pd.DataFrame)`
    - `build_complete_features(self, years: int, end_date: Optional[str])`

**Functions:**
- `calculate_robust_zscore(series, window, min_std)`
- `calculate_regime_with_validation(series, bins, labels, ...)`
- `calculate_percentile_with_validation(series, window, min_data_pct)`
- `safe_percentile_rank(x)`
- `extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series) -> pd.DataFrame`
- `extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame`
- `extract_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame`
- `extract_percentile_rankings(df: pd.DataFrame) -> pd.DataFrame`
- `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x100875dd0>]) -> pd.DataFrame`
- `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x1008aa590>]) -> pd.DataFrame`
- `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x1008b5790>]) -> pd.DataFrame`
- `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x1008e1450>], commodity_data: Dict[<ast.Tuple object at 0x1008e1090>], dollar_data: Dict[<ast.Tuple object at 0x1008e0cd0>], ...) -> pd.DataFrame`
- `extract_term_spreads(yields: pd.DataFrame) -> pd.DataFrame`
- `extract_curve_shape(yields: pd.DataFrame) -> pd.DataFrame`
- `extract_rate_volatility(yields: pd.DataFrame) -> pd.DataFrame`
- `get_calendar_cohort(self, date)`
- `apply_quality_control(self, features: pd.DataFrame)`
- `build_complete_features(self, years: int, end_date: Optional[str]) -> dict`

**Key Dependencies:** numpy, pandas


#### `forecast_calibrator.py` (420 lines)
*Forecast Calibrator - Post-Processing for Probabilistic Forecasts*

**Classes:**
- `ForecastCalibrator`
  - Calibrates probabilistic forecasts using historical forecast errors.
    - `__init__(self)`
    - `fit_from_database(self, db_path: str, min_samples: int, ...)`
    - `calibrate(self, forecast: Dict)`
    - `save(self, filepath: str)`
    - `load(cls, filepath: str)`

**Functions:**
- `fit_from_database(self, db_path: str, min_samples: int, ...) -> bool`
- `calibrate(self, forecast: Dict) -> Dict`
- `save(self, filepath: str)`
- `load(cls, filepath: str) -> Optional[ForecastCalibrator]`
- `get_diagnostics(self) -> Dict`

**Key Dependencies:** core.prediction_database, numpy, pandas


#### `prediction_database.py` (851 lines)
*Prediction Database for Probabilistic Forecasting System*

**Classes:**
- `CommitTracker`
  - Tracks uncommitted writes and screams if data isn't committed.
    - `__init__(self)`
    - `track_write(self, operation: str)`
    - `verify_clean_exit(self)`
- `PredictionDatabase`
  - SQLite database for storing and retrieving probabilistic forecasts.
    - `__init__(self, db_path)`
    - `store_prediction(self, record: Dict)`
    - `commit(self)`
    - `get_commit_status(self)`
    - `get_predictions(self, start_date: str, end_date: str, ...)`

**Functions:**
- `track_write(self, operation: str)`
- `verify_clean_exit(self)`
- `store_prediction(self, record: Dict) -> Optional[str]`
- `commit(self)`
- `get_commit_status(self) -> dict`
- `get_predictions(self, start_date: str, end_date: str, ...) -> pd.DataFrame`
- `migrate_schema(self)`
- `remove_all_duplicates(self)`
- `get_database_stats(self) -> Dict`
- `backfill_actuals(self, fetcher)`
- `compute_quantile_coverage(self, cohort: str) -> Dict`
- `compute_regime_brier_score(self, cohort: str) -> float`
- `get_performance_summary(self) -> Dict`
- `export_to_csv(self, filename: str)`
- `close(self)`

**Key Dependencies:** core.data_fetcher, numpy, pandas


#### `temporal_validator.py` (911 lines)
*Temporal Safety Validator - Prevent Look-Ahead Bias*

**Classes:**
- `TemporalSafetyValidator`
  - Comprehensive temporal safety validation with feature quality scoring.
    - `__init__(self, publication_lags: Dict[<ast.Tuple object at 0x100a6da90>])`
    - `audit_feature_code(self, feature_engine_path: str)`
    - `validate_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...)`
    - `validate_cv_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...)`
    - `verify_publication_lags(self, features: pd.DataFrame, prediction_date: datetime, ...)`

**Functions:**
- `audit_feature_code(self, feature_engine_path: str) -> Dict`
- `validate_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...) -> List[str]`
- `validate_cv_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...) -> bool`
- `verify_publication_lags(self, features: pd.DataFrame, prediction_date: datetime, ...) -> Tuple[<ast.Tuple object at 0x100885490>]`
- `validate_walk_forward_gap(self, features: pd.DataFrame, prediction_date: datetime, ...) -> Tuple[<ast.Tuple object at 0x1008794d0>]`
- `test_feature_availability_at_prediction_time(self, features: pd.DataFrame, feature_metadata: Dict[<ast.Tuple object at 0x10085bc90>], ...) -> Dict`
- `compute_feature_quality(self, feature_dict: dict, date: pd.Timestamp) -> float`
- `compute_feature_quality_batch(self, df: pd.DataFrame) -> pd.Series`
- `update_feature_timestamp(self, feature_name: str, timestamp: pd.Timestamp)`
- `get_feature_age(self, feature_name: str, current_date: pd.Timestamp) -> int`
- `check_quality_threshold(self, quality_score: float, strict: bool) -> tuple`
- `get_quality_report(self, feature_dict: dict, date: pd.Timestamp) -> dict`
- `generate_validation_report(self, output_path: str, include_metadata_check: bool, ...)`
- `run_full_validation(feature_engine_path: str, publication_lags: Dict[<ast.Tuple object at 0x100a85190>]) -> TemporalSafetyValidator`

**Key Dependencies:** numpy, pandas, sklearn.linear_model, sklearn.metrics


#### `xgboost_feature_selector_v2.py` (510 lines)
*XGBoost Feature Selector V2 - VIX % Change Forecasting (Regression)*

**Classes:**
- `IntelligentFeatureSelector`
  - Stability-based feature selection for VIX % change forecasting.
    - `__init__(self, output_dir: str)`
    - `run_full_pipeline(self, features: pd.DataFrame, vix: pd.Series, ...)`

**Functions:**
- `run_full_pipeline(self, features: pd.DataFrame, vix: pd.Series, ...) -> Dict`
- `run_intelligent_feature_selection(integrated_system, horizons: List[int], min_stability: float, ...) -> Dict`

**Key Dependencies:** numpy, pandas, sklearn.metrics, sklearn.model_selection, xgboost


#### `xgboost_trainer_v2.py` (605 lines)
*Probabilistic VIX Forecasting System - Multi-output XGBoost models*

**Classes:**
- `ProbabilisticVIXForecaster`
    - `__init__(self)`
    - `train(self, df: pd.DataFrame, save_dir: str)`
    - `predict(self, X: pd.DataFrame, cohort: str)`
    - `load(self, cohort: str, load_dir: str)`

**Functions:**
- `train(self, df: pd.DataFrame, save_dir: str)`
- `predict(self, X: pd.DataFrame, cohort: str) -> Dict`
- `load(self, cohort: str, load_dir: str)`
- `train_probabilistic_forecaster(df: pd.DataFrame, save_dir: str) -> ProbabilisticVIXForecaster`

**Key Dependencies:** matplotlib.pyplot, numpy, pandas, sklearn.isotonic, sklearn.metrics, sklearn.model_selection, xgboost


### core/.ipynb_checkpoints/

#### `feature_engine-checkpoint.py` (1186 lines)
*Enhanced Feature Engine V4 - Meta Features, Futures, and Maximum Feature Richness*

**Classes:**
- `MetaFeatureEngine`
  - Advanced meta-feature extraction from base features.
    - `extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series)`
    - `extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame)`
    - `extract_rate_of_change_features(df: pd.DataFrame)`
    - `extract_percentile_rankings(df: pd.DataFrame)`
- `FuturesFeatureEngine`
  - Specialized feature extraction for futures contracts.
    - `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x10086e350>])`
    - `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x100a27ed0>])`
    - `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x100875b10>])`
    - `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x100c25e90>], commodity_data: Dict[<ast.Tuple object at 0x100c27710>], dollar_data: Dict[<ast.Tuple object at 0x100c30fd0>], ...)`
- `UnifiedFeatureEngine`
  - Enhanced unified feature engine with meta-features and futures integration.
    - `__init__(self, data_fetcher)`
    - `build_complete_features(self, years: int)`

**Functions:**
- `extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series) -> pd.DataFrame`
- `extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame`
- `extract_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame`
- `extract_percentile_rankings(df: pd.DataFrame) -> pd.DataFrame`
- `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x10086e350>]) -> pd.DataFrame`
- `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x100a27ed0>]) -> pd.DataFrame`
- `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x100875b10>]) -> pd.DataFrame`
- `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x100c25e90>], commodity_data: Dict[<ast.Tuple object at 0x100c27710>], dollar_data: Dict[<ast.Tuple object at 0x100c30fd0>], ...) -> pd.DataFrame`
- `build_complete_features(self, years: int) -> dict`
- `test_enhanced_engine()`

**Key Dependencies:** numpy, pandas


### diagnostics/

#### `bias_diagnostic.py` (411 lines)
*BIAS DIAGNOSTIC - Find the source of the +17.59% systematic error*

**Functions:**
- `load_predictions()`
- `compute_bias_metrics(df)`
- `plot_error_timeseries(df)`
- `plot_bias_by_cohort(df)`
- `plot_bias_by_vix_regime(df)`
- `detect_regime_shift(df)`
- `generate_summary_report(df)`
- `main()`

**Key Dependencies:** matplotlib.pyplot, numpy, pandas


#### `database_investigator.py` (136 lines)
*Database Investigation - Find out why quantile_coverage is inconsistent*

**Key Dependencies:** pandas


#### `feature_quality_diagnostic.py` (275 lines)
*Feature Quality Diagnostics*

**Functions:**
- `analyze_feature_quality(features_df: pd.DataFrame, metadata_df: pd.DataFrame)`
- `load_features_from_system()`

**Key Dependencies:** core.data_fetcher, core.feature_engine, numpy, pandas


#### `production_diagnostics.py` (456 lines)
*Production-Grade Diagnostics for VIX Forecasting System*

**Classes:**
- `DataQualityReport`
  - Comprehensive data quality metrics.
    - `to_dict(self)`
    - `is_acceptable(self)`
    - `get_warnings(self)`
- `DataQualityMonitor`
  - Monitor data quality and detect anomalies.
    - `__init__(self, logger)`
    - `assess_quality(self, features_df: pd.DataFrame, forecast_date: pd.Timestamp)`
- `PredictionValidation`
  - Validate prediction outputs for sanity.
- `PredictionValidator`
  - Validate prediction outputs before storage.
    - `__init__(self, logger)`
    - `validate(self, prediction: Dict, current_vix: float)`
- `ForecastLogger`
  - Structured logging for forecasts with JSON export.
    - `__init__(self, log_dir)`
    - `log_forecast(self, forecast_date: str, prediction: Dict, ...)`
    - `get_session_summary(self)`
    - `export_summary(self)`

**Functions:**
- `to_dict(self)`
- `is_acceptable(self) -> bool`
- `get_warnings(self) -> List[str]`
- `assess_quality(self, features_df: pd.DataFrame, forecast_date: pd.Timestamp) -> DataQualityReport`
- `validate(self, prediction: Dict, current_vix: float) -> PredictionValidation`
- `log_forecast(self, forecast_date: str, prediction: Dict, ...)`
- `get_session_summary(self) -> Dict`
- `export_summary(self)`
- `production_forecast_with_diagnostics(features_df, forecast_date, model, ...)`

**Key Dependencies:** numpy, pandas


#### `walk_forward_validation.py` (641 lines)
*Fixed Walk-Forward Validation - Handles Missing quantile_coverage Data*

**Classes:**
- `EnhancedWalkForwardValidator`
  - Production-ready walk-forward validation with detailed diagnostics.
    - `__init__(self, db_path: str, horizon: int)`
    - `load_predictions_with_actuals(self)`
    - `compute_metrics(self, df: pd.DataFrame)`
    - `generate_diagnostic_report(self, output_dir: str)`

**Functions:**
- `load_predictions_with_actuals(self) -> pd.DataFrame`
- `compute_row_coverage(row)`
- `compute_metrics(self, df: pd.DataFrame) -> Dict`
- `parse_coverage(x)`
- `generate_diagnostic_report(self, output_dir: str)`
- `convert(obj)`

**Key Dependencies:** matplotlib.pyplot, numpy, pandas


### root/

#### `config.py` (911 lines)
*Configuration V5 - Probabilistic Distribution Forecasting*


#### `integrated_system_production.py` (1425 lines)
*Integrated Market Analysis System V5 - Probabilistic Forecasting*

**Classes:**
- `AnomalyOrchestrator`
  - Orchestrates anomaly detection workflow (preserved for backward compatibility).
    - `__init__(self)`
    - `train(self, features: pd.DataFrame, vix: pd.Series, ...)`
    - `detect_current(self, verbose: bool)`
    - `get_persistence_stats(self)`
    - `save_state(self, filepath: str)`
- `IntegratedSystem`
  - Main system integrating probabilistic forecasting with anomaly detection.
    - `__init__(self, models_dir)`
    - `generate_forecast(self, date, store_prediction)`
    - `run(self, date)`
    - `train(self, years: int, real_time_vix: bool, ...)`
    - `run_feature_selection(self, horizons: list, min_stability: float, ...)`

**Functions:**
- `train(self, features: pd.DataFrame, vix: pd.Series, ...)`
- `detect_current(self, verbose: bool) -> dict`
- `get_persistence_stats(self) -> dict`
- `save_state(self, filepath: str)`
- `load_state(self, filepath: str)`
- `generate_forecast(self, date, store_prediction)`
- `run(self, date)`
- `train(self, years: int, real_time_vix: bool, ...)`
- `run_feature_selection(self, horizons: list, min_stability: float, ...) -> dict`
- `get_market_state(self) -> dict`
- `print_anomaly_summary(self)`
- `get_memory_report(self) -> dict`
- `train_probabilistic_models(self, years: int, save_dir: str)`
- `generate_forecast_batch(self, start_date: str, end_date: str, ...)`
- `main()`

**Key Dependencies:** core.anomaly_detector, core.data_fetcher, core.feature_engine, core.forecast_calibrator, core.prediction_database, core.temporal_validator, core.xgboost_feature_selector_v2, core.xgboost_trainer_v2


#### `logging_config.py` (85 lines)
*Centralized logging configuration for the entire system*

**Functions:**
- `setup_logging(level, quiet_mode, log_file)`
- `get_logger(name: str) -> logging.Logger`


#### `train_probabilistic_models.py` (164 lines)
*Training Script for Probabilistic VIX Forecaster*

**Functions:**
- `filter_cohorts_by_min_samples(df: pd.DataFrame, min_samples: int) -> pd.DataFrame`
- `main()`

**Key Dependencies:** core.data_fetcher, core.feature_engine, core.xgboost_trainer_v2, pandas


## Module Dependencies

Key internal dependencies:

- `backtesting_engine.py` → core.prediction_database
- `feature_quality_diagnostic.py` → core.data_fetcher, core.feature_engine
- `forecast_calibrator.py` → core.prediction_database
- `integrated_system_production.py` → core.anomaly_detector, core.temporal_validator, core.feature_engine, diagnostics.walk_forward_validation, core.forecast_calibrator, core.prediction_database, core.data_fetcher, core.xgboost_trainer_v2, core.xgboost_feature_selector_v2
- `prediction_database.py` → core.data_fetcher
- `train_probabilistic_models.py` → core.data_fetcher, core.xgboost_trainer_v2, core.feature_engine