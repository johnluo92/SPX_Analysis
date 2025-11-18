import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import TARGET_CONFIG, TRAINING_END_DATE, TRAINING_YEARS
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.xgboost_trainer_v3 import train_simplified_forecaster

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log")
    ]
)
logger = logging.getLogger(__name__)


def prepare_training_data():
    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION")
    logger.info("=" * 80)
    
    logger.info("\n[1/3] Initializing feature engineering pipeline...")
    data_fetcher = UnifiedDataFetcher()
    feature_engineer = FeatureEngineer(data_fetcher)
    
    logger.info("\n[2/3] Building feature set...")
    logger.info(f"  Target period: {TRAINING_YEARS} years ending {TRAINING_END_DATE}")
    
    result = feature_engineer.build_complete_features(
        years=TRAINING_YEARS,
        end_date=TRAINING_END_DATE
    )
    
    features_df = result["features"]
    spx = result["spx"]
    vix = result["vix"]
    
    logger.info(f"\n  Features created: {len(features_df.columns)}")
    logger.info(f"  Date range: {features_df.index[0].date()} to {features_df.index[-1].date()}")
    logger.info(f"  Total samples: {len(features_df)}")
    
    logger.info("\n[3/3] Merging features with price data...")
    complete_df = features_df.copy()
    complete_df["vix"] = vix
    complete_df["spx"] = spx
    
    if "calendar_cohort" not in complete_df.columns:
        raise ValueError("❌ calendar_cohort column missing!")
    
    cohort_counts = complete_df["calendar_cohort"].value_counts()
    logger.info(f"\n  Cohort distribution:")
    for cohort, count in cohort_counts.items():
        logger.info(f"    {cohort}: {count} samples")
    
    logger.info(f"\n  Final dataframe shape: {complete_df.shape}")
    logger.info("\n✅ Data preparation complete")
    logger.info("=" * 80 + "\n")
    
    return complete_df


def save_training_report(forecaster, output_dir="models"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_version": "v4.0_simplified",
        "target_type": TARGET_CONFIG.get("target_type"),
        "training_summary": {
            "models_trained": 2,
            "model_types": ["direction_classifier", "magnitude_regressor"],
            "features": len(forecaster.feature_names)
        },
        "metrics": forecaster.metrics
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"✅ Training report saved: {report_file}")


def main():
    logger.info("\n" + "=" * 80)
    logger.info("SIMPLIFIED VIX FORECASTER - TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Version: 4.0 (Simplified 2-Model System)")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output: models/")
    logger.info("=" * 80 + "\n")
    
    try:
        complete_df = prepare_training_data()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("MODEL TRAINING")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Starting training...")
        logger.info("  Models: 2 (direction, magnitude)")
        logger.info("  Cohorts: Encoded as binary features")
        logger.info("  Target: Log-space VIX change (5-day)")
        logger.info("")
        
        forecaster = train_simplified_forecaster(
            df=complete_df,
            save_dir="models"
        )
        
        save_training_report(forecaster, output_dir="models")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Models saved to: models/")
        logger.info(f"  - direction_5d_model.pkl")
        logger.info(f"  - magnitude_5d_model.pkl")
        logger.info(f"  - feature_names.json")
        logger.info(f"  - training_metrics.json")
        logger.info(f"  Training report: models/training_report_*.json")
        logger.info("=" * 80 + "\n")
    
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
