"""
Integrated VIX Forecasting System - Production V3 with Complete Workflow

CRITICAL CHANGES FROM V2:
1. Imports from xgboost_trainer_v3 (log-RV system)
2. Primary forecast is now median_forecast (from q50)
3. point_estimate maintained for backward compatibility
4. Enhanced logging shows quantile distribution
5. All predictions validated for quantile ordering

RESTORED IN V3 (5th Pass):
1. Complete mode - Full calibration/recalibration/walk-forward validation
2. Anomaly mode - Backward compatible with anomaly detection system
3. Config-driven workflow using CALIBRATION_PERIOD, VALIDATION_PERIOD
4. Walk-forward diagnostics integration
5. Batch forecasting with proper temporal hygiene

Author: VIX Forecasting System
Last Updated: 2025-11-14
Version: 3.1 (Log-RV with Complete Workflow)
"""

import argparse
import logging
import os
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Check for optional memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available - memory monitoring disabled")

# ============================================================
# V3 IMPORTS: Using new log-RV trainer
# ============================================================
# Anomaly system imports (for backward compatibility)
from config import (
    CALIBRATION_PERIOD,
    ENABLE_TRAINING,
    FEATURE_CONFIG,
    FORECASTING_CONFIG,
    PRODUCTION_START_DATE,
    TARGET_CONFIG,
    TRAINING_YEARS,
    VALIDATION_PERIOD,
    XGBOOST_CONFIG,
)
from core.anomaly_detector import MultiDimensionalAnomalyDetector
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer as UnifiedFeatureEngine
from core.forecast_calibrator import ForecastCalibrator
from core.prediction_database import PredictionDatabase
from core.temporal_validator import TemporalValidator
from core.xgboost_trainer_v3 import ProbabilisticVIXForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/integrated_system.log"),
    ],
)
logger = logging.getLogger(__name__)


class IntegratedForecastingSystem:
    """
    Production forecasting system using log-RV quantile regression.

    V3 ENHANCEMENTS:
    - Median forecast as primary output (more robust than mean)
    - Full quantile distribution (q10, q25, q50, q75, q90)
    - Direction classifier for complementary signal
    - Confidence scoring based on feature quality
    - Domain-aware bounds enforcement [10%, 90%]

    RESTORED CAPABILITIES (5th Pass):
    - Complete workflow mode for calibration/validation
    - Anomaly detection integration (backward compatible)
    - Config-driven operational modes
    """

    def __init__(
        self,
        models_dir: str = "models",
        db_path: str = "data_cache/predictions.db",
    ):
        """
        Initialize the integrated forecasting system.

        Args:
            models_dir: Directory containing trained models
            db_path: Path to predictions database
        """

        logger.info("=" * 80)
        logger.info("INTEGRATED PROBABILISTIC FORECASTING SYSTEM V3.1")
        logger.info("=" * 80)

        self.models_dir = Path(models_dir)
        self.db_path = db_path

        # Core forecasting components
        self.data_fetcher = UnifiedDataFetcher()
        self.feature_engine = UnifiedFeatureEngine(data_fetcher=self.data_fetcher)
        self.forecaster = ProbabilisticVIXForecaster()
        self.validator = TemporalValidator()
        self.prediction_db = PredictionDatabase(db_path=db_path)

        # Anomaly detector (runs independently when needed)
        self.orchestrator = MultiDimensionalAnomalyDetector()

        # Load trained models
        self._load_models()

        # Load calibrator (if available)
        self.calibrator = ForecastCalibrator.load()
        if self.calibrator:
            logger.info("📊 Forecast calibrator loaded")
        else:
            logger.info("ℹ️  No calibrator found - forecasts will not be calibrated")

        # State tracking
        self.last_forecast = None
        self.forecast_history = []
        self.trained = False
        self._cached_anomaly_result = None
        self._cache_timestamp = None

        # Feature caching for efficiency
        self._feature_cache = None
        self._feature_cache_date = None

        # Memory monitoring (if available)
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
            self.baseline_memory_mb = None
            self.memory_history = []
            self.memory_monitoring_enabled = True
        else:
            self.memory_monitoring_enabled = False

        logger.info("✅ System initialized")

    def _load_models(self):
        """Load all trained cohort models."""
        logger.info("📂 Loading trained models...")

        model_files = list(self.models_dir.glob("probabilistic_forecaster_*.pkl"))

        if len(model_files) == 0:
            logger.warning("⚠️ No trained models found. Run training first.")
            return

        for model_file in model_files:
            cohort = model_file.stem.replace("probabilistic_forecaster_", "")
            self.forecaster.load(cohort, self.models_dir)
            logger.info(f"   ✅ Loaded: {cohort}")

        logger.info(f"📊 Total cohorts loaded: {len(self.forecaster.models)}")

    def generate_forecast(
        self,
        forecast_date: Optional[pd.Timestamp] = None,
        store_prediction: bool = True,
    ) -> Optional[Dict]:
        """
        Generate probabilistic forecast using V3 quantile regression.

        FORECAST PIPELINE (6 steps):
        1. Prepare observation date and feature matrix
        2. Validate temporal hygiene (no data leakage)
        3. Determine cohort and feature quality
        4. Generate quantile predictions from trained models
        5. Apply calibration (if available)
        6. Store to database

        Args:
            forecast_date: Date to forecast for (default: today)
            store_prediction: Whether to store in database

        Returns:
            Dict containing forecast distribution or None if failed
        """

        logger.info("\n" + "=" * 80)
        logger.info("PROBABILISTIC FORECAST GENERATION (V3)")
        logger.info("=" * 80)

        # Step 1: Prepare date and features
        if forecast_date is None:
            forecast_date = pd.Timestamp.now().normalize()

        logger.info(f"\n📅 Forecast date: {forecast_date.date()}")
        logger.info(f"📈 Target horizon: {TARGET_CONFIG['horizon_days']} days")
        logger.info(f"🎯 Primary forecast: Median (q50) from quantile regression")

        # Get feature matrix
        features = self._get_features()

        if forecast_date not in features.index:
            logger.error(f"❌ Date {forecast_date.date()} not in feature matrix")
            return None

        # Step 2: Extract observation
        observation = features.loc[forecast_date]

        # Validate temporal hygiene
        logger.info("\n🔍 Validating temporal hygiene...")
        is_valid, violations = self.validator.validate_walk_forward_gap(
            features, forecast_date, strict=False
        )

        if violations:
            logger.warning(f"⚠️  Found {len(violations)} temporal violations")
            for v in violations[:3]:
                logger.warning(f"   → {v}")

        # Step 3: Determine cohort and quality
        cohort = observation["calendar_cohort"]
        cohort_weight = observation["cohort_weight"]
        quality_score = observation["feature_quality"]

        logger.info(f"\n📊 Cohort: {cohort} (weight: {cohort_weight:.3f})")
        logger.info(f"✨ Feature quality: {quality_score:.1%}")

        # Prepare feature vector (exclude metadata columns)
        metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        X = observation.drop(metadata_cols, errors="ignore")
        X_df = X.to_frame().T

        # Step 4: Generate distribution
        logger.info("\n🎯 Generating probabilistic forecast...")

        try:
            distribution = self.forecaster.predict(X_df, cohort)
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None

        # Step 5: Apply calibration if available
        if self.calibrator:
            distribution = self.calibrator.calibrate(distribution)
            logger.info("📊 Applied forecast calibration")

        # Adjust confidence by cohort weight
        distribution["confidence_score"] *= 2 - cohort_weight
        distribution["confidence_score"] = np.clip(
            distribution["confidence_score"], 0, 1
        )

        # Validate quantile ordering
        self._validate_quantile_ordering(distribution)

        # Step 6: Add metadata
        target_date = forecast_date + pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
        distribution["metadata"] = {
            "observation_date": forecast_date.strftime("%Y-%m-%d"),
            "forecast_date": target_date.strftime("%Y-%m-%d"),
            "horizon_days": TARGET_CONFIG["horizon_days"],
            "feature_quality": float(quality_score),
            "cohort_weight": float(cohort_weight),
            "current_vix": float(observation["vix"]),
            "features_used": len(self.forecaster.feature_names),
        }

        # Display forecast
        self._display_forecast_summary(distribution, observation)

        # Store in database
        if store_prediction:
            prediction_id = self._store_prediction(
                distribution, observation, forecast_date
            )
            distribution["prediction_id"] = prediction_id
            logger.info(f"💾 Stored prediction: {prediction_id}")

        # Update state
        self.last_forecast = distribution
        self.forecast_history.append(
            {"date": forecast_date, "distribution": distribution}
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ FORECAST COMPLETE")
        logger.info("=" * 80)

        return distribution

    def _validate_quantile_ordering(self, distribution: Dict):
        """Ensure quantiles are properly ordered."""
        quantiles = ["q10", "q25", "q50", "q75", "q90"]
        values = [distribution.get(q, 0) for q in quantiles]

        if values != sorted(values):
            logger.warning("⚠️  Quantiles not properly ordered - enforcing monotonicity")
            # Force ordering
            for i in range(1, len(quantiles)):
                if values[i] < values[i - 1]:
                    values[i] = values[i - 1]

            for q, v in zip(quantiles, values):
                distribution[q] = v

    def _display_forecast_summary(self, distribution: Dict, observation: pd.Series):
        """Display human-readable forecast summary."""

        print(f"\n{'=' * 80}")
        print("FORECAST SUMMARY")
        print(f"{'=' * 80}")

        # Primary forecast (median)
        median = distribution["median_forecast"]
        print(f"\nMedian Forecast (50th): {median:+.1f}%")
        print(f"                 (Primary forecast from quantile regression)")

        # Full distribution
        print(f"\nDistribution:")
        print(f"   10th percentile: {distribution['q10']:+.1f}%")
        print(f"   25th percentile: {distribution['q25']:+.1f}%")
        print(f"   Median (50th):   {distribution['q50']:+.1f}%")
        print(f"   75th percentile: {distribution['q75']:+.1f}%")
        print(f"   90th percentile: {distribution['q90']:+.1f}%")

        # Implied VIX levels
        current_vix = observation["vix"]
        print(f"\nImplied VIX Levels (from {current_vix:.2f}):")
        print(f"   10th: {current_vix * (1 + distribution['q10'] / 100):.2f}")
        print(f"   25th: {current_vix * (1 + distribution['q25'] / 100):.2f}")
        print(f"   50th: {current_vix * (1 + distribution['q50'] / 100):.2f}")
        print(f"   75th: {current_vix * (1 + distribution['q75'] / 100):.2f}")
        print(f"   90th: {current_vix * (1 + distribution['q90'] / 100):.2f}")

        # Direction and confidence
        print(f"\nDirectional Forecast:")
        print(f"   Probability UP:   {distribution['prob_up']:.1%}")
        print(f"   Probability DOWN: {distribution['prob_down']:.1%}")
        print(f"\nConfidence: {distribution['confidence_score']:.2f}")

        print(f"{'=' * 80}\n")

    def _store_prediction(
        self,
        distribution: Dict,
        observation: pd.Series,
        forecast_date: pd.Timestamp,
    ) -> int:
        """
        Store prediction to database using V3 schema.

        V3 CHANGES:
        - median_forecast is primary field
        - point_estimate populated for backward compatibility
        - All quantiles stored
        """

        target_date = forecast_date + pd.Timedelta(days=TARGET_CONFIG["horizon_days"])

        prediction = {
            "forecast_date": target_date,
            "observation_date": forecast_date,
            "horizon": TARGET_CONFIG["horizon_days"],
            "current_vix": float(observation["vix"]),
            "cohort": observation["calendar_cohort"],
            # V3: Median as primary
            "median_forecast": distribution["median_forecast"],
            # Backward compatibility
            "point_estimate": distribution["median_forecast"],
            # Full quantile distribution
            "q10": distribution["q10"],
            "q25": distribution["q25"],
            "q50": distribution["q50"],
            "q75": distribution["q75"],
            "q90": distribution["q90"],
            # Direction and confidence
            "prob_up": distribution["prob_up"],
            "prob_down": distribution["prob_down"],
            "confidence_score": distribution["confidence_score"],
            # Metadata
            "feature_quality": float(distribution["metadata"]["feature_quality"]),
            "cohort_weight": float(distribution["metadata"]["cohort_weight"]),
        }

        prediction_id = self.prediction_db.store_prediction(prediction)

        return prediction_id

    def backfill_actuals(self):
        """Backfill actual outcomes for all predictions."""

        logger.info("\n" + "=" * 80)
        logger.info("BACKFILLING ACTUALS")
        logger.info("=" * 80)

        self.prediction_db.backfill_actuals()

        logger.info("=" * 80 + "\n")

    def generate_forecast_batch(
        self,
        start_date: str,
        end_date: str,
    ):
        """
        Generate forecasts for a date range (for backtesting/calibration).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """

        logger.info("\n" + "=" * 80)
        logger.info("BATCH FORECAST GENERATION")
        logger.info("=" * 80)

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        logger.info(f"\nDate range: {start.date()} to {end.date()}")

        # Generate business days
        dates = pd.date_range(start, end, freq="B")

        logger.info(f"Forecasting {len(dates)} dates...\n")

        successful = 0
        failed = 0

        for date in dates:
            logger.info(f"\nProcessing {date.date()}...")

            result = self.generate_forecast(forecast_date=date)

            if result:
                successful += 1
            else:
                failed += 1

        logger.info("\n" + "=" * 80)
        logger.info("BATCH GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info("=" * 80 + "\n")

    # ========================================================================
    # ANOMALY DETECTION METHODS (Backward Compatibility)
    # ========================================================================

    def train(
        self,
        years: int = TRAINING_YEARS,
        real_time_vix: bool = True,
        verbose: bool = False,
        enable_anomaly: bool = False,
    ):
        """
        Train the complete system (backward compatible with anomaly system).

        Note: This is for anomaly detection only. Probabilistic models are
        trained separately via train_probabilistic_models.py
        """

        print(f"\n{'=' * 80}\nINTEGRATED SYSTEM V3.1 - TRAINING MODE\n{'=' * 80}")
        print(
            f"Config: {years}y training | Real-time VIX: {real_time_vix} | Anomaly: {enable_anomaly}"
        )

        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="pre-training")

        print("\n[1/2] Building features...")
        feature_data = self.feature_engine.build_complete_features(years=years)
        features = feature_data["features"]
        vix = feature_data["vix"]
        spx = feature_data["spx"]

        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-features")

        self.orchestrator.features = features
        self.orchestrator.vix_ml = vix
        self.orchestrator.spx_ml = spx

        if enable_anomaly:
            print("[2/2] Training anomaly system...")
            vix_history_all = self.orchestrator.fetcher.fetch_yahoo(
                "^VIX", "1990-01-02", datetime.now().strftime("%Y-%m-%d")
            )["Close"].squeeze()

            self.orchestrator.train(
                features=features,
                vix=vix,
                spx=spx,
                vix_history_all=vix_history_all,
                verbose=verbose,
            )

            if self.memory_monitoring_enabled:
                self._log_memory_stats(context="post-training")

            if real_time_vix:
                try:
                    live_vix = self.orchestrator.fetcher.fetch_price("^VIX")
                    if live_vix:
                        self.orchestrator.vix_ml.iloc[-1] = live_vix
                        self.orchestrator.features.iloc[
                            -1, self.orchestrator.features.columns.get_loc("vix")
                        ] = live_vix
                        if verbose:
                            print(f"✅ Updated live VIX: {live_vix:.2f}")
                except Exception as e:
                    warnings.warn(f"Live VIX fetch failed: {e}")
        else:
            print("[2/2] Anomaly training skipped (enable_anomaly=False)")

        self.trained = True
        print(f"\n{'=' * 80}\n✅ TRAINING COMPLETE\n{'=' * 80}")

    def get_market_state(self) -> dict:
        """Generate comprehensive market state snapshot (legacy anomaly method)."""

        if not self.trained:
            raise ValueError("Must train system first")

        anomaly_result = self._get_cached_anomaly_result()
        persistence_stats = self.orchestrator.get_persistence_stats()

        current_vix = float(self.orchestrator.vix_ml.iloc[-1])
        current_regime = self._classify_vix_regime(current_vix)

        ensemble = anomaly_result["ensemble"]
        ensemble_score = ensemble["score"]

        level, p_value, confidence = (
            self.orchestrator.anomaly_detector.classify_anomaly(
                ensemble_score, method="statistical"
            )
        )

        severity_messages = {
            "CRITICAL": f"Extreme anomaly ({ensemble_score:.1%}) - Markets in unprecedented configuration",
            "HIGH": f"Significant anomaly ({ensemble_score:.1%}) - Notable market stress detected",
            "MODERATE": f"Moderate anomaly ({ensemble_score:.1%}) - Elevated market uncertainty",
            "NORMAL": f"Normal conditions ({ensemble_score:.1%}) - Market within typical ranges",
        }

        ensemble["severity"] = level
        ensemble["severity_message"] = severity_messages[level]
        if p_value is not None:
            ensemble["p_value"] = float(p_value)
            ensemble["confidence"] = float(confidence)

        top_anomalies = self._get_top_anomalies_list(anomaly_result)

        regime_stats = self.orchestrator.regime_stats["regimes"][current_regime["id"]]
        persistence_prob = regime_stats["transitions_5d"]["persistence"]["probability"]
        persistence_prob_clamped = max(0.01, min(0.99, persistence_prob))
        expected_duration = 1.0 / (1.0 - persistence_prob_clamped)

        return {
            "timestamp": datetime.now().isoformat(),
            "vix": {
                "current": current_vix,
                "regime": current_regime,
                "regime_stats": regime_stats,
            },
            "anomaly_analysis": {
                "ensemble": ensemble,
                "top_anomalies": top_anomalies,
                "persistence": {
                    "probability": float(persistence_prob),
                    "expected_duration_days": float(expected_duration),
                },
            },
        }

    def print_anomaly_summary(self):
        """Print human-readable anomaly summary."""

        if not self.trained:
            raise ValueError("Must train system first")

        state = self.get_market_state()

        print(f"\n{'=' * 80}")
        print("ANOMALY DETECTION SUMMARY")
        print(f"{'=' * 80}")

        vix_info = state["vix"]
        anomaly = state["anomaly_analysis"]

        print(f"\nCurrent VIX: {vix_info['current']:.2f}")
        print(f"Regime: {vix_info['regime']['label']}")

        ensemble = anomaly["ensemble"]
        print(f"\nEnsemble Anomaly: {ensemble['score']:.1%}")
        print(f"Severity: {ensemble['severity']}")
        print(f"{ensemble['severity_message']}")

        print(f"\nTop Anomalies:")
        for i, anom in enumerate(anomaly["top_anomalies"][:5], 1):
            print(f"  {i}. {anom['feature']}: {anom['score']:.1%}")

        print(f"{'=' * 80}\n")

    def _get_cached_anomaly_result(self):
        """Get cached anomaly result (for efficiency)."""

        now = datetime.now()

        if self._cached_anomaly_result is not None:
            if (now - self._cache_timestamp).seconds < 300:  # 5 min cache
                return self._cached_anomaly_result

        result = self.orchestrator.predict()
        self._cached_anomaly_result = result
        self._cache_timestamp = now

        return result

    def _classify_vix_regime(self, vix: float) -> dict:
        """Classify VIX into regime (helper method)."""

        if vix < 15:
            return {"id": "low", "label": "Low Volatility", "range": "<15"}
        elif vix < 20:
            return {"id": "normal", "label": "Normal Volatility", "range": "15-20"}
        elif vix < 30:
            return {"id": "elevated", "label": "Elevated Volatility", "range": "20-30"}
        else:
            return {"id": "crisis", "label": "Crisis Volatility", "range": ">30"}

    def _get_top_anomalies_list(self, anomaly_result: dict) -> list:
        """Extract top anomalies from result."""

        top_anomalies = []

        for detection in anomaly_result.get("individual_detections", []):
            if detection.get("is_anomaly"):
                top_anomalies.append(
                    {
                        "feature": detection["feature"],
                        "score": detection["anomaly_score"],
                        "threshold": detection.get("threshold", None),
                    }
                )

        # Sort by score
        top_anomalies.sort(key=lambda x: x["score"], reverse=True)

        return top_anomalies

    def _log_memory_stats(self, context: str = ""):
        """Log memory usage (if psutil available)."""

        if not self.memory_monitoring_enabled:
            return

        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024

        if self.baseline_memory_mb is None:
            self.baseline_memory_mb = mem_mb

        delta_mb = mem_mb - self.baseline_memory_mb

        self.memory_history.append(
            {
                "context": context,
                "memory_mb": mem_mb,
                "delta_mb": delta_mb,
            }
        )

        logger.info(f"💾 Memory ({context}): {mem_mb:.1f} MB (Δ {delta_mb:+.1f} MB)")

    # ========================================================================
    # FEATURE CACHING
    # ========================================================================

    def _get_features(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get features with intelligent caching.

        Caching rules:
        - Features are built once per calendar day
        - Cache persists across multiple forecasts
        - force_refresh=True bypasses cache (for batch backtesting)

        Args:
            force_refresh: Force rebuild regardless of cache state

        Returns:
            pd.DataFrame: Feature matrix with all features + metadata
        """

        today = pd.Timestamp.now().normalize()

        # Check if cache is valid
        if (
            not force_refresh
            and self._feature_cache is not None
            and self._feature_cache_date == today
        ):
            logger.info("📦 Using cached features (already built today)")
            return self._feature_cache

        # Build fresh features
        logger.info("🔧 Building features...")
        feature_data = self.feature_engine.build_complete_features(years=TRAINING_YEARS)
        df = feature_data["features"]

        # Force numeric dtypes (safety check)
        metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        for col in df.columns:
            if col not in metadata_cols and df[col].dtype == object:
                logger.warning(f"⚠️ Converting object column to numeric: {col}")
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Verify no object columns remain
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        unexpected = [c for c in object_cols if c not in metadata_cols]
        if unexpected:
            logger.error(f"❌ Unexpected object columns: {unexpected[:5]}")
            raise ValueError(
                f"DataFrame contains {len(unexpected)} non-numeric columns"
            )

        # Cache it
        self._feature_cache = df
        self._feature_cache_date = today

        logger.info(f"✅ Features cached: {df.shape}, dtypes OK")
        return df


def main():
    """
    Main execution function with CLI argument support.

    OPERATIONAL MODES:
    - forecast: Generate single prediction for today
    - complete: Full calibration/validation workflow
    - batch: Generate predictions for date range
    - backfill: Backfill actual outcomes
    - anomaly: Run anomaly detection (legacy)
    """

    parser = argparse.ArgumentParser(
        description="Integrated VIX Forecasting System V3.1"
    )

    parser.add_argument(
        "--mode",
        choices=["forecast", "complete", "batch", "backfill", "anomaly"],
        default="forecast",
        help="Operation mode",
    )

    parser.add_argument(
        "--forecast-date", type=str, help="Forecast date (YYYY-MM-DD), default: today"
    )

    parser.add_argument(
        "--start-date", type=str, help="Start date for batch mode (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=str, help="End date for batch mode (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="data_cache/predictions.db",
        help="Path to predictions database",
    )

    args = parser.parse_args()

    # Check training is enabled for anomaly mode
    if args.mode == "anomaly" and not ENABLE_TRAINING:
        print(f"\n{'=' * 80}")
        print("⚠️ TRAINING DISABLED (config.ENABLE_TRAINING = False)")
        print("⚠️ Set ENABLE_TRAINING = True in config.py for anomaly mode")
        print(f"{'=' * 80}\n")
        return

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Initialize system
    system = IntegratedForecastingSystem(
        models_dir=args.models_dir,
        db_path=args.db_path,
    )

    # Execute requested operation
    if args.mode == "forecast":
        # Single forecast
        forecast_date = None
        if args.forecast_date:
            forecast_date = pd.Timestamp(args.forecast_date)

        result = system.generate_forecast(forecast_date=forecast_date)

        if result:
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.mode == "complete":
        # COMPLETE WORKFLOW MODE
        logger.info("🎯 MODE: Complete Workflow - Everything in one command")

        try:
            # Load config periods
            cal_start, cal_end = CALIBRATION_PERIOD
            val_start, val_end = VALIDATION_PERIOD

            logger.info("=" * 80)
            logger.info("COMPLETE WORKFLOW")
            logger.info("=" * 80)
            logger.info("This will:")
            logger.info(
                f"  1. Generate uncalibrated forecasts ({cal_start} to {cal_end})"
            )
            logger.info(f"  2. Backfill actuals for calibration period")
            logger.info(f"  3. Train calibrator on {cal_start[:4]} data")
            logger.info(f"  4. Regenerate {val_start[:4]} WITH calibration")
            logger.info(f"  5. Backfill actuals for validation period")
            logger.info(
                f"  6. Generate {PRODUCTION_START_DATE[:4]} forecasts (Jan 1 - today)"
            )
            logger.info(f"  7. Backfill actuals for {PRODUCTION_START_DATE[:4]}")
            logger.info(f"  8. Run validation diagnostics")
            logger.info("=" * 80)

            # STEP 1: Generate uncalibrated forecasts for CALIBRATION period only
            logger.info(
                f"\n[1/8] Generating uncalibrated forecasts ({cal_start} to {cal_end})..."
            )
            original_calibrator = system.calibrator
            system.calibrator = None  # Disable calibration
            system.generate_forecast_batch(cal_start, cal_end)

            # STEP 2: Backfill actuals for calibration period
            logger.info(f"\n[2/8] Backfilling actuals for {cal_start[:4]}...")
            system.prediction_db.backfill_actuals()

            # STEP 3: Train calibrator on calibration period ONLY
            logger.info(f"\n[3/8] Training calibrator on {cal_start[:4]} data...")
            calibrator = ForecastCalibrator()
            success = calibrator.fit_from_database(
                db=system.prediction_db,
                min_samples=50,
                start_date=cal_start,
                end_date=cal_end,
            )
            if not success:
                logger.error("❌ Calibration failed - insufficient data")
                system.calibrator = original_calibrator
                return
            calibrator.save()

            # STEP 4: Delete validation period forecasts (if any exist)
            logger.info(f"\n[4/8] Clearing old {val_start[:4]} forecasts...")
            conn = sqlite3.connect(system.prediction_db.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM forecasts WHERE forecast_date BETWEEN ? AND ?",
                (val_start, val_end),
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(f"   Deleted {deleted} old forecasts")

            # STEP 5: Generate validation period forecasts WITH calibration
            logger.info(
                f"\n[5/8] Generating {val_start[:4]} forecasts WITH calibration..."
            )
            system.calibrator = ForecastCalibrator.load()
            system.generate_forecast_batch(val_start, val_end)

            # STEP 6: Backfill actuals for validation period
            logger.info(f"\n[6/8] Backfilling actuals for {val_start[:4]}...")
            system.prediction_db.backfill_actuals()

            # STEP 7: Generate production year forecasts WITH calibration
            logger.info(f"\n[7/8] Generating {PRODUCTION_START_DATE[:4]} forecasts...")
            today = datetime.now().strftime("%Y-%m-%d")
            system.generate_forecast_batch(PRODUCTION_START_DATE, today)

            # STEP 8: Backfill actuals and run validation
            logger.info(
                f"\n[8/8] Backfilling {PRODUCTION_START_DATE[:4]} actuals and running validation..."
            )
            system.prediction_db.backfill_actuals()

            from diagnostics.walk_forward_validation import EnhancedWalkForwardValidator

            validator = EnhancedWalkForwardValidator(
                db_path=system.prediction_db.db_path
            )
            validator.generate_diagnostic_report()

            logger.info("\n" + "=" * 80)
            logger.info("✅ COMPLETE WORKFLOW FINISHED")
            logger.info("=" * 80)
            logger.info("\n📊 Results:")
            logger.info("  • diagnostics/walk_forward_metrics.json")
            logger.info("  • diagnostics/*.png")
            logger.info(f"\n📈 Calibrator trained on: {cal_start} to {cal_end}")
            logger.info(f"📈 Validation period: {val_start} to {val_end}")
            logger.info(f"📈 {PRODUCTION_START_DATE[:4]} forecasts: up to {today}")
            logger.info("\n🚀 System ready for production!")
            logger.info(
                "  Run daily: python integrated_system_production.py --mode forecast"
            )

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.mode == "batch":
        # Batch forecasts
        if not args.start_date or not args.end_date:
            logger.error("❌ Batch mode requires --start-date and --end-date")
            sys.exit(1)

        system.generate_forecast_batch(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        sys.exit(0)

    elif args.mode == "backfill":
        # Backfill actuals
        system.backfill_actuals()
        sys.exit(0)

    elif args.mode == "anomaly":
        # Legacy anomaly detection mode
        system.train(
            years=TRAINING_YEARS,
            real_time_vix=True,
            verbose=False,
            enable_anomaly=True,
        )
        system.print_anomaly_summary()

        anomaly_result = system._get_cached_anomaly_result()
        if anomaly_result and system.orchestrator.anomaly_detector:
            from export.unified_exporter import UnifiedExporter

            persistence_stats = system.orchestrator.get_persistence_stats()
            exporter = UnifiedExporter(output_dir="./json_data")

            exporter.export_live_state(
                orchestrator=system.orchestrator,
                anomaly_result=anomaly_result,
                spx=system.orchestrator.spx_ml,
                vix=system.orchestrator.vix_ml,
                persistence_stats=persistence_stats,
            )
            logger.info("✅ Anomaly data exported to ./json_data/")

        sys.exit(0)


if __name__ == "__main__":
    main()
