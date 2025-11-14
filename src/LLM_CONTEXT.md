# System Context for LLM
Generated: 2025-11-14 15:02

## Quick Stats
- Modules: 19 | Lines: 12658
- Classes: 24 | Functions: 233

## Core Architecture

### Entry Points
- **integrated_system_production.py**: 1175L
  Integrated VIX Forecasting System - Production V3 with Complete Workflow
- **train_probabilistic_models.py**: 532L
  Training Script for Probabilistic VIX Forecaster V3
- **config.py**: 1069L
  Configuration V5 - Refactored for True Quantile Regression
- **logging_config.py**: 85L
  Centralized logging configuration for the entire system

### Core Modules (core/)
- **feature_engine-checkpoint.py**: MetaFeatureEngine, FuturesFeatureEngine, UnifiedFeatureEngine
- **anomaly_detector.py**: MultiDimensionalAnomalyDetector
- **backtesting_engine.py**: ProbabilisticBacktester
- **data_fetcher.py**: DataFetchLogger, UnifiedDataFetcher
- **feature_engineer.py**: MetaFeatureEngine, FuturesFeatureEngine, TreasuryYieldFeatureEngine, FeatureEngineer
- **forecast_calibrator.py**: ForecastCalibrator, MockDatabase
- **prediction_database.py**: PredictionDatabase
- **temporal_validator.py**: TemporalSafetyValidator
- **xgboost_feature_selector_v2.py**: XGBoostFeatureSelector
- **xgboost_trainer_v3.py**: ProbabilisticVIXForecaster

## Key Classes & Their Roles
- **ForecastCalibrator** (forecast_calibrator.py)
  Methods: fit_from_database, calibrate, get_diagnostics, save_calibrator
- **PredictionDatabase** (prediction_database.py)
  Purpose: Manages storage and retrieval of VIX forecasts and actuals.
  Methods: transaction, store_prediction, get_predictions, backfill_actuals
- **MetaFeatureEngine** (feature_engineer.py)
  Methods: extract_regime_indicators, extract_cross_asset_relationships, extract_rate_of_change_features, extract_percentile_rankings
- **FuturesFeatureEngine** (feature_engineer.py)
  Methods: extract_vix_futures_features, extract_commodity_futures_features, extract_dollar_futures_features, extract_futures_cross_relationships
- **TreasuryYieldFeatureEngine** (feature_engineer.py)
  Methods: extract_term_spreads, extract_curve_shape, extract_rate_volatility
- **FeatureEngineer** (feature_engineer.py)
  Methods: get_calendar_cohort, apply_quality_control, build_complete_features
- **MultiDimensionalAnomalyDetector** (anomaly_detector.py)
  Purpose: 15 independent Isolation Forests with feature importance and quality validation.
  Methods: calculate_statistical_thresholds, train, detect, calculate_historical_persistence_stats
- **ProbabilisticBacktester** (backtesting_engine.py)
  Purpose: Evaluate probabilistic forecasts stored in database.
  Methods: run_full_evaluation, evaluate_quantile_coverage, evaluate_brier_score, evaluate_confidence
- **TemporalSafetyValidator** (temporal_validator.py)
  Purpose: Comprehensive temporal safety validation with feature quality scoring.
  Methods: audit_feature_code, validate_split, validate_cv_split, verify_publication_lags
- **ProbabilisticVIXForecaster** (xgboost_trainer_v3.py)
  Purpose: Quantile-based volatility forecasting with directional classifier.
  Methods: train, predict, load
- **DataFetchLogger** (data_fetcher.py)
  Methods: info, warning, error
- **UnifiedDataFetcher** (data_fetcher.py)
  Methods: fetch_fred, fetch_fred_series, fetch_all_fred_series, fetch_yahoo
- **XGBoostFeatureSelector** (xgboost_feature_selector_v2.py)
  Purpose: Feature selection using XGBoost feature importance for log-RV forecasting.
  Methods: select_features, save_results, load_results
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
- **IntegratedForecastingSystem** (integrated_system_production.py)
  Purpose: Production forecasting system using log-RV quantile regression.
  Methods: generate_forecast, backfill_actuals, generate_forecast_batch, train