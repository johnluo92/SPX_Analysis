# System Architecture Documentation
*Generated: 2025-11-14 15:02:22*

## Overview
- Total modules analyzed: 19
- Total lines of code: 12658

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
- `validate_feature_quality(features: pd.DataFrame, feature_list: list, detector_name: str) -> Tuple[<ast.Tuple object at 0x101541150>]`
- `calculate_robust_anomaly_score(raw_score: float, training_distribution: np.ndarray, min_percentile: float, ...) -> float`
- `calculate_coverage_penalty(coverage: float, min_coverage: float) -> float`
- `calculate_statistical_thresholds(self) -> dict`
- `train(self, features: pd.DataFrame, verbose: bool)`
- `detect(self, features: pd.DataFrame, verbose: bool) -> dict`
- `calculate_historical_persistence_stats(self, ensemble_scores: np.ndarray, dates: Optional[pd.DatetimeIndex], ...) -> Dict`
- `get_top_anomalies(self, result: dict, top_n: int) -> list`
- `get_feature_contributions(self, detector_name: str, top_n: int) -> List[Tuple[<ast.Tuple object at 0x10150ed90>]]`
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


#### `data_fetcher.py` (690 lines)
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
- `fetch_all_fred_series(self, start_date: str, end_date: str, ...) -> Dict[<ast.Tuple object at 0x1014ac510>]`
- `fetch_yahoo(self, symbol: str, start_date: str, ...) -> Optional[pd.DataFrame]`
- `fetch_price(self, symbol: str) -> Optional[float]`
- `fetch_cboe_series(self, symbol: str) -> Optional[pd.Series]`
- `fetch_all_cboe(self) -> Dict[<ast.Tuple object at 0x101330610>]`
- `fetch_fomc_calendar(self, start_year: int, end_year: int) -> Optional[pd.DataFrame]`
- `update_fomc_calendar_from_csv(self, csv_path: str) -> Optional[pd.DataFrame]`

**Key Dependencies:** numpy, pandas


#### `feature_engineer.py` (1356 lines)
*Enhanced Feature Engine V5 - Streamlined, No Duplicates, WITH CALENDAR COHORT INTEGRATION*

**Classes:**
- `MetaFeatureEngine`
    - `extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series)`
    - `extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame)`
    - `extract_rate_of_change_features(df: pd.DataFrame)`
    - `extract_percentile_rankings(df: pd.DataFrame)`
- `FuturesFeatureEngine`
    - `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x1012e8450>])`
    - `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x101306d90>])`
    - `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x101481a90>])`
    - `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x101541590>], commodity_data: Dict[<ast.Tuple object at 0x101541950>], dollar_data: Dict[<ast.Tuple object at 0x101541d10>], ...)`
- `TreasuryYieldFeatureEngine`
    - `extract_term_spreads(yields: pd.DataFrame)`
    - `extract_curve_shape(yields: pd.DataFrame)`
    - `extract_rate_volatility(yields: pd.DataFrame)`
- `FeatureEngineer`
    - `__init__(self, data_fetcher)`
    - `get_calendar_cohort(self, date)`
    - `apply_quality_control(self, features: pd.DataFrame)`
    - `build_complete_features(self, years: int, end_date: Optional[str], ...)`

**Functions:**
- `calculate_robust_zscore(series, window, min_std)`
- `calculate_regime_with_validation(series, bins, labels, ...)`
- `calculate_percentile_with_validation(series, window, min_data_pct)`
- `safe_percentile_rank(x)`
- `extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series) -> pd.DataFrame`
- `extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame`
- `extract_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame`
- `extract_percentile_rankings(df: pd.DataFrame) -> pd.DataFrame`
- `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x1012e8450>]) -> pd.DataFrame`
- `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x101306d90>]) -> pd.DataFrame`
- `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x101481a90>]) -> pd.DataFrame`
- `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x101541590>], commodity_data: Dict[<ast.Tuple object at 0x101541950>], dollar_data: Dict[<ast.Tuple object at 0x101541d10>], ...) -> pd.DataFrame`
- `extract_term_spreads(yields: pd.DataFrame) -> pd.DataFrame`
- `extract_curve_shape(yields: pd.DataFrame) -> pd.DataFrame`
- `extract_rate_volatility(yields: pd.DataFrame) -> pd.DataFrame`
- `get_calendar_cohort(self, date)`
- `apply_quality_control(self, features: pd.DataFrame)`
- `build_complete_features(self, years: int, end_date: Optional[str], ...) -> dict`

**Key Dependencies:** numpy, pandas


#### `forecast_calibrator.py` (646 lines)
*Forecast Calibrator V3 - Bias Correction for Log-RV Forecasts*

**Classes:**
- `ForecastCalibrator`
    - `__init__(self, min_samples: int, use_robust: bool, ...)`
    - `fit_from_database(self, database, start_date: Optional[str], ...)`
    - `calibrate(self, raw_forecast: float, current_vix: float, ...)`
    - `get_diagnostics(self)`
    - `save_calibrator(self, output_dir: str)`
- `MockDatabase`
    - `get_predictions(self, with_actuals)`

**Functions:**
- `fit_from_database(self, database, start_date: Optional[str], ...) -> bool`
- `calibrate(self, raw_forecast: float, current_vix: float, ...) -> Dict`
- `get_diagnostics(self) -> Dict`
- `save_calibrator(self, output_dir: str)`
- `load_calibrator(self, input_dir: str)`
- `load(cls, input_dir: str) -> Optional[ForecastCalibrator]`
- `test_calibrator()`
- `get_predictions(self, with_actuals)`

**Key Dependencies:** numpy, pandas, sklearn.linear_model, sklearn.metrics


#### `prediction_database.py` (814 lines)
*Prediction Database Module - Log-Transformed Realized Volatility System V3*

**Classes:**
- `PredictionDatabase`
  - Manages storage and retrieval of VIX forecasts and actuals.
    - `__init__(self, db_path: str)`
    - `transaction(self)`
    - `store_prediction(self, record: Dict)`
    - `get_predictions(self, start_date: Optional[str], end_date: Optional[str], ...)`
    - `backfill_actuals(self, vix_series: pd.Series, horizon: int)`

**Functions:**
- `transaction(self)`
- `store_prediction(self, record: Dict) -> Optional[str]`
- `get_predictions(self, start_date: Optional[str], end_date: Optional[str], ...) -> pd.DataFrame`
- `backfill_actuals(self, vix_series: pd.Series, horizon: int)`
- `get_performance_summary(self) -> Dict`
- `close(self)`
- `validate_schema()`

**Key Dependencies:** numpy, pandas


#### `temporal_validator.py` (911 lines)
*Temporal Safety Validator - Prevent Look-Ahead Bias*

**Classes:**
- `TemporalSafetyValidator`
  - Comprehensive temporal safety validation with feature quality scoring.
    - `__init__(self, publication_lags: Dict[<ast.Tuple object at 0x101530910>])`
    - `audit_feature_code(self, feature_engine_path: str)`
    - `validate_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...)`
    - `validate_cv_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...)`
    - `verify_publication_lags(self, features: pd.DataFrame, prediction_date: datetime, ...)`

**Functions:**
- `audit_feature_code(self, feature_engine_path: str) -> Dict`
- `validate_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...) -> List[str]`
- `validate_cv_split(self, X_train: pd.DataFrame, X_val: pd.DataFrame, ...) -> bool`
- `verify_publication_lags(self, features: pd.DataFrame, prediction_date: datetime, ...) -> Tuple[<ast.Tuple object at 0x1012e1f90>]`
- `validate_walk_forward_gap(self, features: pd.DataFrame, prediction_date: datetime, ...) -> Tuple[<ast.Tuple object at 0x10134bb10>]`
- `test_feature_availability_at_prediction_time(self, features: pd.DataFrame, feature_metadata: Dict[<ast.Tuple object at 0x1013483d0>], ...) -> Dict`
- `compute_feature_quality(self, feature_dict: dict, date: pd.Timestamp) -> float`
- `compute_feature_quality_batch(self, df: pd.DataFrame) -> pd.Series`
- `update_feature_timestamp(self, feature_name: str, timestamp: pd.Timestamp)`
- `get_feature_age(self, feature_name: str, current_date: pd.Timestamp) -> int`
- `check_quality_threshold(self, quality_score: float, strict: bool) -> tuple`
- `get_quality_report(self, feature_dict: dict, date: pd.Timestamp) -> dict`
- `generate_validation_report(self, output_path: str, include_metadata_check: bool, ...)`
- `run_full_validation(feature_engine_path: str, publication_lags: Dict[<ast.Tuple object at 0x101510c10>]) -> TemporalSafetyValidator`

**Key Dependencies:** numpy, pandas, sklearn.linear_model, sklearn.metrics


#### `xgboost_feature_selector_v2.py` (566 lines)
*XGBoost Feature Selector V3 - Log-Transformed Realized Volatility System*

**Classes:**
- `XGBoostFeatureSelector`
  - Feature selection using XGBoost feature importance for log-RV forecasting.
    - `__init__(self, horizon: int, min_importance: float, ...)`
    - `select_features(self, features_df: pd.DataFrame, spx_returns: pd.Series, ...)`
    - `save_results(self, output_dir: str)`
    - `load_results(self, input_dir: str)`

**Functions:**
- `select_features(self, features_df: pd.DataFrame, spx_returns: pd.Series, ...) -> Tuple[<ast.Tuple object at 0x1012ea6d0>]`
- `save_results(self, output_dir: str)`
- `load_results(self, input_dir: str)`
- `test_feature_selector()`

**Key Dependencies:** numpy, pandas, sklearn.model_selection, xgboost


#### `xgboost_trainer_v3.py` (680 lines)
*Refactored Probabilistic VIX Forecasting System V3*

**Classes:**
- `ProbabilisticVIXForecaster`
  - Quantile-based volatility forecasting with directional classifier.
    - `__init__(self)`
    - `train(self, df: pd.DataFrame, save_dir: str)`
    - `predict(self, X: pd.DataFrame, cohort: str, ...)`
    - `load(self, cohort: str, load_dir: str)`

**Functions:**
- `train(self, df: pd.DataFrame, save_dir: str)`
- `predict(self, X: pd.DataFrame, cohort: str, ...) -> Dict`
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
    - `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x1012e9ed0>])`
    - `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x101388690>])`
    - `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x101348c10>])`
    - `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x1014afbd0>], commodity_data: Dict[<ast.Tuple object at 0x1014aff50>], dollar_data: Dict[<ast.Tuple object at 0x1014ac990>], ...)`
- `UnifiedFeatureEngine`
  - Enhanced unified feature engine with meta-features and futures integration.
    - `__init__(self, data_fetcher)`
    - `build_complete_features(self, years: int)`

**Functions:**
- `extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series) -> pd.DataFrame`
- `extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame`
- `extract_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame`
- `extract_percentile_rankings(df: pd.DataFrame) -> pd.DataFrame`
- `extract_vix_futures_features(vx_data: Dict[<ast.Tuple object at 0x1012e9ed0>]) -> pd.DataFrame`
- `extract_commodity_futures_features(futures_data: Dict[<ast.Tuple object at 0x101388690>]) -> pd.DataFrame`
- `extract_dollar_futures_features(dollar_data: Dict[<ast.Tuple object at 0x101348c10>]) -> pd.DataFrame`
- `extract_futures_cross_relationships(vx_data: Dict[<ast.Tuple object at 0x1014afbd0>], commodity_data: Dict[<ast.Tuple object at 0x1014aff50>], dollar_data: Dict[<ast.Tuple object at 0x1014ac990>], ...) -> pd.DataFrame`
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


#### `walk_forward_validation.py` (731 lines)
*Walk-Forward Validation - V3 Adapted for Log-RV Quantile Regression*

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

**Key Dependencies:** matplotlib.pyplot, numpy, pandas


### root/

#### `config.py` (1069 lines)
*Configuration V5 - Refactored for True Quantile Regression*


#### `integrated_system_production.py` (1175 lines)
*Integrated VIX Forecasting System - Production V3 with Complete Workflow*

**Classes:**
- `IntegratedForecastingSystem`
  - Production forecasting system using log-RV quantile regression.
    - `__init__(self, models_dir: str, db_path: str)`
    - `generate_forecast(self, date, store_prediction)`
    - `backfill_actuals(self)`
    - `generate_forecast_batch(self, start_date: str, end_date: str, ...)`
    - `train(self, years: int, real_time_vix: bool, ...)`

**Functions:**
- `generate_forecast(self, date, store_prediction)`
- `backfill_actuals(self)`
- `generate_forecast_batch(self, start_date: str, end_date: str, ...)`
- `train(self, years: int, real_time_vix: bool, ...)`
- `get_market_state(self) -> dict`
- `print_anomaly_summary(self)`
- `main()`

**Key Dependencies:** core.anomaly_detector, core.data_fetcher, core.feature_engineer, core.forecast_calibrator, core.prediction_database, core.temporal_validator, core.walk_forward_validation, core.xgboost_trainer_v3


#### `logging_config.py` (85 lines)
*Centralized logging configuration for the entire system*

**Functions:**
- `setup_logging(level, quiet_mode, log_file)`
- `get_logger(name: str) -> logging.Logger`


#### `train_probabilistic_models.py` (532 lines)
*Training Script for Probabilistic VIX Forecaster V3*

**Functions:**
- `prepare_training_data()`
- `validate_configuration() -> Tuple[<ast.Tuple object at 0x1012e6f10>]`
- `save_training_report(training_results: dict, output_dir: str)`
- `display_training_summary(forecaster: ProbabilisticVIXForecaster)`
- `main()`

**Key Dependencies:** core.data_fetcher, core.feature_engineer, core.xgboost_trainer_v3, pandas


## Module Dependencies

Key internal dependencies:

- `backtesting_engine.py` → core.prediction_database
- `feature_quality_diagnostic.py` → core.data_fetcher, core.feature_engine
- `integrated_system_production.py` → core.anomaly_detector, core.walk_forward_validation, core.feature_engineer, core.data_fetcher, core.xgboost_trainer_v3, core.forecast_calibrator, core.prediction_database, core.temporal_validator
- `train_probabilistic_models.py` → core.xgboost_trainer_v3, core.data_fetcher, core.feature_engineer