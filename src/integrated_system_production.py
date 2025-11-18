import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CALIBRATION_PERIOD,
    PRODUCTION_START_DATE,
    TARGET_CONFIG,
    TRAINING_YEARS,
    VALIDATION_PERIOD
)
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer as UnifiedFeatureEngine
from core.prediction_database import PredictionDatabase
from core.temporal_validator import TemporalSafetyValidator as TemporalValidator
from core.xgboost_trainer_v3 import SimplifiedVIXForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/integrated_system.log")
    ]
)
logger = logging.getLogger(__name__)


class IntegratedForecastingSystem:
    def __init__(self, models_dir: str = "models", db_path: str = "data_cache/predictions.db"):
        logger.info("=" * 80)
        logger.info("INTEGRATED FORECASTING SYSTEM V4.0")
        logger.info("=" * 80)

        self.models_dir = Path(models_dir)
        self.db_path = db_path

        self.data_fetcher = UnifiedDataFetcher()
        self.feature_engine = UnifiedFeatureEngine(data_fetcher=self.data_fetcher)
        self.forecaster = SimplifiedVIXForecaster()
        self.validator = TemporalValidator()
        self.prediction_db = PredictionDatabase(db_path=db_path)

        self._load_models()

        self.last_forecast = None
        self.forecast_history = []

        self._feature_cache = None
        self._feature_cache_date = None

        logger.info("✅ System initialized")

    def _load_models(self):
        logger.info("📂 Loading trained models...")

        direction_file = self.models_dir / "direction_5d_model.pkl"
        magnitude_file = self.models_dir / "magnitude_5d_model.pkl"

        if not direction_file.exists() or not magnitude_file.exists():
            logger.warning("⚠️ No trained models found. Run training first.")
            return

        self.forecaster.load(self.models_dir)
        logger.info(f"📊 Loaded 2 models with {len(self.forecaster.feature_names)} features")

    def generate_forecast(self, date=None, store_prediction=True):
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING FORECAST")
        logger.info("=" * 80)

        if date is not None:
            target_date = pd.Timestamp(date)
            logger.info(f"📅 Forecast date: {target_date.strftime('%Y-%m-%d')}")
            logger.info("🔧 Building features (historical mode)...")

            feature_data = self.feature_engine.build_complete_features(
                years=TRAINING_YEARS,
                end_date=target_date.strftime("%Y-%m-%d")
            )
            df = feature_data["features"]

            metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
            numeric_cols = [c for c in df.columns if c not in metadata_cols]

            for col in numeric_cols:
                if df[col].dtype == object:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                df[col] = df[col].astype(np.float64)

            logger.info("✅ Features validated")
        else:
            df = self._get_features()
            target_date = df.index[-1]
            logger.info(f"📅 Using latest date: {target_date.strftime('%Y-%m-%d')}")

        if target_date not in df.index:
            available_range = f"{df.index[0].date()} to {df.index[-1].date()}"
            raise ValueError(
                f"Date {target_date.date()} not in feature data. "
                f"Available range: {available_range}"
            )

        observation = df.loc[target_date]

        logger.info("🔍 Checking data quality...")
        feature_dict = observation.to_dict()
        quality_score = self.validator.compute_feature_quality(feature_dict, target_date)
        usable, quality_msg = self.validator.check_quality_threshold(quality_score)

        logger.info(f"   Quality Score: {quality_score:.2f}")
        logger.info(f"   Status: {quality_msg}")

        if not usable:
            report = self.validator.get_quality_report(feature_dict, target_date)
            logger.error("❌ Data quality insufficient:")
            for issue in report["issues"]:
                logger.error(f"   • {issue}")
            raise ValueError(f"Cannot forecast: {quality_msg}")

        cohort = observation.get("calendar_cohort", "mid_cycle")
        cohort_weight = observation.get("cohort_weight", 1.0)

        logger.info(f"📅 Calendar Cohort: {cohort} (weight: {cohort_weight:.2f})")

        logger.info("🎯 Preparing features for prediction...")
        feature_values = observation[self.forecaster.feature_names]
        feature_array = pd.to_numeric(feature_values, errors="coerce").values
        X_df = pd.DataFrame(
            feature_array.reshape(1, -1),
            columns=self.forecaster.feature_names,
            dtype=np.float64
        )
        X_df = X_df.fillna(0.0)

        current_vix = float(observation["vix"])

        logger.info("🎯 Generating forecast...")
        forecast = self.forecaster.predict(X_df, current_vix)

        forecast_date = target_date + pd.Timedelta(days=TARGET_CONFIG["horizon_days"])

        forecast["metadata"] = {
            "observation_date": target_date.strftime("%Y-%m-%d"),
            "forecast_date": forecast_date.strftime("%Y-%m-%d"),
            "horizon_days": TARGET_CONFIG["horizon_days"],
            "feature_quality": float(quality_score),
            "cohort_weight": float(cohort_weight),
            "calendar_cohort": cohort,
            "features_used": len(self.forecaster.feature_names)
        }

        self._log_forecast_summary(forecast)

        if store_prediction:
            prediction_id = self._store_prediction(forecast, observation, target_date)
            forecast["prediction_id"] = prediction_id
            logger.info(f"💾 Stored prediction: {prediction_id}")

        self.last_forecast = forecast
        self.forecast_history.append({"date": target_date, "forecast": forecast})

        logger.info("=" * 80)
        logger.info("✅ FORECAST COMPLETE")
        logger.info("=" * 80)

        return forecast

    def _log_forecast_summary(self, forecast):
        logger.info("\n" + "─" * 80)
        logger.info("FORECAST")
        logger.info("─" * 80)

        logger.info(f"\n5-Day VIX Forecast")
        logger.info(f"Current VIX: {forecast['current_vix']:.2f}")

        logger.info(f"\nDirection:")
        logger.info(f"  Probability UP:   {forecast['prob_up']:.1%}")
        logger.info(f"  Probability DOWN: {forecast['prob_down']:.1%}")

        logger.info(f"\nMagnitude:")
        logger.info(f"  Expected change: {forecast['magnitude_pct']:+.2f}%")
        logger.info(f"  Expected VIX: {forecast['expected_vix']:.2f}")

        if "metadata" in forecast:
            meta = forecast["metadata"]
            logger.info(f"\nContext:")
            logger.info(f"  Current cohort: {meta.get('calendar_cohort', 'N/A')}")
            logger.info(f"  Feature quality: {meta.get('feature_quality', 0):.2f}")
            logger.info(f"  Observation date: {meta.get('observation_date', 'N/A')}")

        logger.info("─" * 80 + "\n")

    def _store_prediction(self, forecast, observation, observation_date):
        forecast_date = observation_date + pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
        prediction_id = f"pred_{forecast_date.strftime('%Y%m%d')}_h{TARGET_CONFIG['horizon_days']}"

        prediction = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(),
            "forecast_date": forecast_date,
            "observation_date": observation_date,
            "horizon": TARGET_CONFIG["horizon_days"],
            "current_vix": float(observation["vix"]),
            "calendar_cohort": observation["calendar_cohort"],
            "cohort_weight": float(observation.get("cohort_weight", 1.0)),
            "prob_up": forecast["prob_up"],
            "prob_down": forecast["prob_down"],
            "magnitude_forecast": forecast["magnitude_pct"],
            "expected_vix": forecast["expected_vix"],
            "feature_quality": float(forecast["metadata"]["feature_quality"]),
            "num_features_used": len(self.forecaster.feature_names),
            "features_used": ",".join(self.forecaster.feature_names[:10]),
            "model_version": "v4.0_simplified"
        }

        stored_id = self.prediction_db.store_prediction(prediction)
        return stored_id if stored_id else prediction_id

    def backfill_actuals(self):
        logger.info("\n" + "=" * 80)
        logger.info("BACKFILLING ACTUALS")
        logger.info("=" * 80)

        self.prediction_db.backfill_actuals()

        logger.info("=" * 80 + "\n")

    def generate_forecast_batch(self, start_date: str, end_date: str):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BATCH FORECASTING: {start_date} to {end_date}")
        logger.info(f"{'=' * 80}")

        logger.info("🔧 Building features for batch...")
        feature_data = self.feature_engine.build_complete_features(
            years=TRAINING_YEARS,
            end_date=end_date
        )
        df = feature_data["features"]

        metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        numeric_cols = [c for c in df.columns if c not in metadata_cols]

        for col in numeric_cols:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].astype(np.float64)

        logger.info("✅ Features validated")

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        date_range = df[(df.index >= start) & (df.index <= end)].index

        logger.info(f"📅 Forecasting {len(date_range)} dates")
        logger.info(f"   Range: {date_range[0].date()} to {date_range[-1].date()}")

        forecasts = []
        commit_interval = 100

        for i, date in enumerate(date_range):
            try:
                observation = df.loc[date]
                feature_dict = observation.to_dict()
                quality_score = self.validator.compute_feature_quality(feature_dict, date)
                usable, _ = self.validator.check_quality_threshold(quality_score)

                if not usable:
                    logger.warning(f"   Skipped {date.date()}: low quality ({quality_score:.2f})")
                    continue

                feature_values = observation[self.forecaster.feature_names]
                feature_array = pd.to_numeric(feature_values, errors="coerce").values
                X_df = pd.DataFrame(
                    feature_array.reshape(1, -1),
                    columns=self.forecaster.feature_names,
                    dtype=np.float64
                ).fillna(0.0)

                current_vix = float(observation["vix"])
                forecast = self.forecaster.predict(X_df, current_vix)

                forecast_date = date + pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
                forecast["metadata"] = {
                    "observation_date": date.strftime("%Y-%m-%d"),
                    "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                    "horizon_days": TARGET_CONFIG["horizon_days"],
                    "feature_quality": float(quality_score),
                    "cohort_weight": float(observation.get("cohort_weight", 1.0)),
                    "calendar_cohort": observation.get("calendar_cohort", "mid_cycle"),
                    "features_used": len(self.forecaster.feature_names)
                }

                prediction_id = self._store_prediction(forecast, observation, date)
                forecast["prediction_id"] = prediction_id
                forecasts.append(forecast)

                if (i + 1) % commit_interval == 0:
                    pct = int(100 * (i + 1) / len(date_range))
                    logger.info(f"   Progress: {i + 1}/{len(date_range)} ({pct}%)")
                    self.prediction_db.commit()


            except Exception as e:
                logger.warning(f"   Failed {date.date()}: {e}")
                continue

        if len(forecasts) % commit_interval != 0:
            self.prediction_db.commit()

        logger.info(f"✅ Generated and stored {len(forecasts)} forecasts")

        status = self.prediction_db.get_commit_status()
        if status["pending_writes"] > 0:
            logger.error(f"🚨 WARNING: {status['pending_writes']} uncommitted writes!")
            raise RuntimeError(f"Lost {status['pending_writes']} forecasts - commit failed!")

        return forecasts

    def _get_features(self, force_refresh=False):
        now = pd.Timestamp.now()
        today = (now.year, now.month, now.day)

        if not force_refresh and self._feature_cache is not None and self._feature_cache_date == today:
            logger.info("📦 Using cached features (already built today)")
            return self._feature_cache

        logger.info("🔧 Building features...")
        feature_data = self.feature_engine.build_complete_features(years=TRAINING_YEARS)
        df = feature_data["features"]

        metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        numeric_cols = [c for c in df.columns if c not in metadata_cols]

        numeric_array = df[numeric_cols].values.astype(np.float64)
        df_clean = pd.DataFrame(numeric_array, columns=numeric_cols, index=df.index)

        for col in metadata_cols:
            if col in df.columns:
                df_clean[col] = df[col]

        df_clean = df_clean[df.columns]

        self._feature_cache = df_clean
        self._feature_cache_date = today

        logger.info(f"✅ Features validated and cached")
        logger.info(f"   Shape: {df_clean.shape}")

        return df_clean


def main():
    parser = argparse.ArgumentParser(description="Integrated VIX Forecasting System V4.0")
    parser.add_argument(
        "--mode",
        choices=["forecast", "complete", "batch", "backfill"],
        default="forecast",
        help="Operation mode"
    )
    parser.add_argument("--forecast-date", type=str, help="Forecast date (YYYY-MM-DD), default: today")
    parser.add_argument("--start-date", type=str, help="Start date for batch mode (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for batch mode (YYYY-MM-DD)")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data_cache/predictions.db",
        help="Path to predictions database"
    )

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    system = IntegratedForecastingSystem(models_dir=args.models_dir, db_path=args.db_path)

    if args.mode == "forecast":
        forecast_date = None
        if args.forecast_date:
            forecast_date = pd.Timestamp(args.forecast_date)

        result = system.generate_forecast(date=forecast_date)

        if result:
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.mode == "complete":
        logger.info("🎯 MODE: Complete Workflow")

        try:
            cal_start, cal_end = CALIBRATION_PERIOD
            val_start, val_end = VALIDATION_PERIOD

            logger.info("=" * 80)
            logger.info("COMPLETE WORKFLOW")
            logger.info("=" * 80)
            logger.info("This will:")
            logger.info(f"  1. Generate forecasts ({cal_start} to {cal_end})")
            logger.info(f"  2. Backfill actuals for calibration period")
            logger.info(f"  3. Generate {val_start[:4]} forecasts")
            logger.info(f"  4. Backfill actuals for validation period")
            logger.info(f"  5. Generate {PRODUCTION_START_DATE[:4]} forecasts (Jan 1 - today)")
            logger.info(f"  6. Backfill actuals for {PRODUCTION_START_DATE[:4]}")
            logger.info(f"  7. Run validation diagnostics")
            logger.info("=" * 80)

            logger.info(f"\n[1/7] Generating forecasts ({cal_start} to {cal_end})...")
            system.generate_forecast_batch(cal_start, cal_end)

            logger.info(f"\n[2/7] Backfilling actuals for {cal_start[:4]}...")
            system.prediction_db.backfill_actuals()

            logger.info(f"\n[3/7] Generating {val_start[:4]} forecasts...")
            system.generate_forecast_batch(val_start, val_end)

            logger.info(f"\n[4/7] Backfilling actuals for {val_start[:4]}...")
            system.prediction_db.backfill_actuals()

            logger.info(f"\n[5/7] Generating {PRODUCTION_START_DATE[:4]} forecasts...")
            today = datetime.now().strftime("%Y-%m-%d")
            system.generate_forecast_batch(PRODUCTION_START_DATE, today)

            logger.info(f"\n[6/7] Backfilling {PRODUCTION_START_DATE[:4]} actuals...")
            system.prediction_db.backfill_actuals()

            logger.info(f"\n[7/7] Running validation...")
            try:
                from core.walk_forward_validation import SimplifiedWalkForwardValidator
                validator = SimplifiedWalkForwardValidator(db_path=system.prediction_db.db_path)
                validator.generate_diagnostic_report()
            except ImportError:
                logger.warning("Walk-forward validation module not found - skipping diagnostic report")

            logger.info("\n" + "=" * 80)
            logger.info("✅ COMPLETE WORKFLOW FINISHED")
            logger.info("=" * 80)
            logger.info("\n📊 Results:")
            logger.info("  • diagnostics/walk_forward_metrics.json")
            logger.info("  • diagnostics/*.png")
            logger.info(f"\n📈 Validation period: {val_start} to {val_end}")
            logger.info(f"📈 {PRODUCTION_START_DATE[:4]} forecasts: up to {today}")
            logger.info("\n🚀 System ready for production!")
            logger.info("  Run daily: python integrated_system_production.py --mode forecast")

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.mode == "batch":
        if not args.start_date or not args.end_date:
            logger.error("❌ Batch mode requires --start-date and --end-date")
            sys.exit(1)

        system.generate_forecast_batch(start_date=args.start_date, end_date=args.end_date)
        sys.exit(0)

    elif args.mode == "backfill":
        system.prediction_db.backfill_actuals()
        sys.exit(0)


if __name__ == "__main__":
    main()
