"""
Training Script for Probabilistic VIX Forecaster
Trains all cohort models and saves to disk
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Setup clean logging FIRST
from logging_config import get_logger, setup_logging

setup_logging(level="DEBUG", quiet_mode=True)

from config import TRAINING_YEARS
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engine import UnifiedFeatureEngine
from core.xgboost_trainer_v2 import train_probabilistic_forecaster

logger = get_logger(__name__)


def filter_cohorts_by_min_samples(
    df: pd.DataFrame, min_samples: int = 200
) -> pd.DataFrame:
    """
    Merge small cohorts into mid_cycle to ensure sufficient training data.

    Args:
        df: Feature dataframe with calendar_cohort column
        min_samples: Minimum samples required for separate cohort training

    Returns:
        Modified dataframe with small cohorts merged to mid_cycle
    """
    cohort_counts = df["calendar_cohort"].value_counts()

    small_cohorts = cohort_counts[cohort_counts < min_samples].index.tolist()

    if small_cohorts:
        logger.warning(
            f"\n⚠️  Merging small cohorts into mid_cycle (< {min_samples} samples):"
        )
        for cohort in small_cohorts:
            count = cohort_counts[cohort]
            logger.warning(f"   {cohort}: {count} samples → mid_cycle")
            df.loc[df["calendar_cohort"] == cohort, "calendar_cohort"] = "mid_cycle"

        logger.info(f"\n📊 Updated cohort distribution:")
        updated_counts = df["calendar_cohort"].value_counts()
        for cohort, count in updated_counts.items():
            pct = count / len(df) * 100
            logger.info(f"   {cohort:30s}: {count:5d} samples ({pct:5.1f}%)")

    return df


def main():
    """
    Complete training pipeline:
    1. Build features with calendar cohorts
    2. Train probabilistic models for each cohort
    3. Save models to disk
    """

    logger.info("=" * 80)
    logger.info("PROBABILISTIC VIX FORECASTER - TRAINING PIPELINE")
    logger.info("=" * 80)

    # Check models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    logger.info(f"\n📂 Models will be saved to: {models_dir.absolute()}")
    logger.info(f"📅 Training window: {TRAINING_YEARS} years")

    # Step 1: Build Features
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: BUILDING FEATURES")
    logger.info("=" * 80)

    try:
        data_fetcher = UnifiedDataFetcher()
        feature_engine = UnifiedFeatureEngine(data_fetcher=data_fetcher)

        feature_data = feature_engine.build_complete_features(years=TRAINING_YEARS)
        df = feature_data["features"]

        logger.info(f"\n✅ Features built successfully")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
        logger.info(f"   Features: {len(df.columns)}")

        # Validate required columns
        required_cols = ["vix", "calendar_cohort", "cohort_weight", "feature_quality"]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            logger.error(f"❌ Missing required columns: {missing}")
            return False

        # Show cohort distribution
        cohort_dist = df["calendar_cohort"].value_counts()
        logger.info(f"\n📊 Calendar Cohort Distribution:")
        for cohort, count in cohort_dist.items():
            pct = count / len(df) * 100
            logger.info(f"   {cohort:30s}: {count:5d} samples ({pct:5.1f}%)")

        # Filter small cohorts to prevent CV errors
        df = filter_cohorts_by_min_samples(df, min_samples=200)

    except Exception as e:
        logger.error(f"❌ Feature building failed: {e}", exc_info=True)
        return False

    # Step 2: Train Probabilistic Forecaster
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TRAINING PROBABILISTIC MODELS")
    logger.info("=" * 80)

    try:
        forecaster = train_probabilistic_forecaster(df=df, save_dir=str(models_dir))

        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Cohorts trained: {len(forecaster.models)}")
        logger.info(f"   Models per cohort: 8 (point, 5 quantiles, regime, confidence)")

        # Verify saved files
        model_files = list(models_dir.glob("probabilistic_forecaster_*.pkl"))
        logger.info(f"\n💾 Saved model files:")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"   {model_file.name:50s} ({size_mb:.2f} MB)")

        # Show metrics
        logger.info(f"\n📊 Model diagnostics saved:")
        diagnostics_file = models_dir / "probabilistic_model_metrics.json"
        if diagnostics_file.exists():
            logger.info(f"   {diagnostics_file}")

        plot_file = models_dir / "regime_performance.png"
        if plot_file.exists():
            logger.info(f"   {plot_file}")

        return True

    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return False

    finally:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 80)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
