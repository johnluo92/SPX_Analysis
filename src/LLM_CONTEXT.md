# System Context for LLM
Generated: 2025-11-13 12:50

## Quick Stats
- Modules: 18 | Lines: 11098
- Classes: 25 | Functions: 238

## Core Architecture

### Entry Points
- **integrated_system_production.py**: 1237L
  Integrated Market Analysis System V5 - Probabilistic Forecasting
- **train_probabilistic_models.py**: 164L
  Training Script for Probabilistic VIX Forecaster
- **config.py**: 911L
  Configuration V5 - Probabilistic Distribution Forecasting
- **logging_config.py**: 85L
  Centralized logging configuration for the entire system

### Core Modules (core/)
- **feature_engine-checkpoint.py**: MetaFeatureEngine, FuturesFeatureEngine, UnifiedFeatureEngine
- **anomaly_detector.py**: MultiDimensionalAnomalyDetector
- **backtesting_engine.py**: ProbabilisticBacktester
- **data_fetcher.py**: DataFetchLogger, UnifiedDataFetcher
- **feature_engine.py**: MetaFeatureEngine, FuturesFeatureEngine, TreasuryYieldFeatureEngine, UnifiedFeatureEngine
- **forecast_calibrator.py**: ForecastCalibrator
- **prediction_database.py**: CommitTracker, PredictionDatabase
- **temporal_validator.py**: TemporalSafetyValidator
- **xgboost_feature_selector_v2.py**: IntelligentFeatureSelector
- **xgboost_trainer_v2.py**: ProbabilisticVIXForecaster

## Key Classes & Their Roles
- **ForecastCalibrator** (forecast_calibrator.py)
  Purpose: Calibrates probabilistic forecasts using historical forecast errors.
  Methods: fit_from_database, calibrate, save, load
- **MetaFeatureEngine** (feature_engine.py)
  Methods: extract_regime_indicators, extract_cross_asset_relationships, extract_rate_of_change_features, extract_percentile_rankings
- **FuturesFeatureEngine** (feature_engine.py)
  Methods: extract_vix_futures_features, extract_commodity_futures_features, extract_dollar_futures_features, extract_futures_cross_relationships
- **TreasuryYieldFeatureEngine** (feature_engine.py)
  Methods: extract_term_spreads, extract_curve_shape, extract_rate_volatility
- **UnifiedFeatureEngine** (feature_engine.py)
  Methods: get_calendar_cohort, build_complete_features
- **CommitTracker** (prediction_database.py)
  Purpose: Tracks uncommitted writes and screams if data isn't committed.
  Methods: track_write, verify_clean_exit
- **PredictionDatabase** (prediction_database.py)
  Purpose: SQLite database for storing and retrieving probabilistic forecasts.
  Methods: store_prediction, commit, get_commit_status, get_predictions
- **MultiDimensionalAnomalyDetector** (anomaly_detector.py)
  Purpose: 15 independent Isolation Forests with feature importance and quality validation.
  Methods: calculate_statistical_thresholds, train, detect, calculate_historical_persistence_stats
- **ProbabilisticBacktester** (backtesting_engine.py)
  Purpose: Evaluate probabilistic forecasts stored in database.
  Methods: run_full_evaluation, evaluate_quantile_coverage, evaluate_brier_score, evaluate_confidence
- **TemporalSafetyValidator** (temporal_validator.py)
  Purpose: Comprehensive temporal safety validation with feature quality scoring.
  Methods: audit_feature_code, validate_split, validate_cv_split, verify_publication_lags
- **ProbabilisticVIXForecaster** (xgboost_trainer_v2.py)
  Methods: train, predict, load
- **DataFetchLogger** (data_fetcher.py)
  Methods: info, warning, error
- **UnifiedDataFetcher** (data_fetcher.py)
  Methods: fetch_fred, fetch_all_fred_series, fetch_yahoo, fetch_price
- **IntelligentFeatureSelector** (xgboost_feature_selector_v2.py)
  Purpose: Stability-based feature selection for VIX % change forecasting.
  Methods: run_full_pipeline
- **MetaFeatureEngine** (feature_engine-checkpoint.py)
  Purpose: Advanced meta-feature extraction from base features.
  Methods: extract_regime_indicators, extract_cross_asset_relationships, extract_rate_of_change_features, extract_percentile_rankings
- **FuturesFeatureEngine** (feature_engine-checkpoint.py)
  Purpose: Specialized feature extraction for futures contracts.
  Methods: extract_vix_futures_features, extract_commodity_futures_features, extract_dollar_futures_features, extract_futures_cross_relationships
- **UnifiedFeatureEngine** (feature_engine-checkpoint.py)
  Purpose: Enhanced unified feature engine with meta-features and futures integration.
  Methods: build_complete_features
- **DataQualityReport** (production_diagnostics.py)
  Purpose: Comprehensive data quality metrics.
  Methods: to_dict, is_acceptable, get_warnings
- **ForecastLogger** (production_diagnostics.py)
  Purpose: Structured logging for forecasts with JSON export.
  Methods: log_forecast, get_session_summary, export_summary
- **EnhancedWalkForwardValidator** (walk_forward_validation.py)
  Purpose: Production-ready walk-forward validation with detailed diagnostics.
  Methods: load_predictions_with_actuals, compute_metrics, generate_diagnostic_report
- **AnomalyOrchestrator** (integrated_system_production.py)
  Methods: train, detect_current, get_persistence_stats, save_state
- **IntegratedSystem** (integrated_system_production.py)
  Methods: generate_forecast, run, train, run_feature_selection