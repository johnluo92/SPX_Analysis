import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
from typing import Tuple
import pandas as pd
from config import TARGET_CONFIG,TRAINING_END_DATE,TRAINING_YEARS,XGBOOST_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.xgboost_trainer_v3 import ProbabilisticVIXForecaster,train_probabilistic_forecaster
from logging_config import get_logger,setup_logging
Path("logs").mkdir(exist_ok=True)
setup_logging(level=logging.INFO,quiet_mode=False,log_file="logs/training.log")
logger=get_logger(__name__)
def prepare_training_data():
    logger.info("\n"+"="*80);logger.info("DATA PREPARATION");logger.info("="*80)
    logger.info("\n[1/4] Initializing feature engineering pipeline...")
    data_fetcher=UnifiedDataFetcher();feature_engineer=FeatureEngineer(data_fetcher)
    logger.info("\n[2/4] Building feature set for training period...")
    logger.info(f"  Target period: {TRAINING_YEARS} years ending {TRAINING_END_DATE}")
    logger.info(f"  This includes ~450 days warmup for technical indicators")
    result=feature_engineer.build_complete_features(years=TRAINING_YEARS,end_date=TRAINING_END_DATE)
    features_df=result["features"];spx=result["spx"];vix=result["vix"]
    logger.info(f"\n  Features created: {len(features_df.columns)}\n  Date range: {features_df.index[0].date()} to {features_df.index[-1].date()}\n  Total samples: {len(features_df)}")
    warmup_days=450;usable_samples=len(features_df)-warmup_days
    usable_start_date=features_df.index[warmup_days]if len(features_df)>warmup_days else features_df.index[0]
    logger.info(f"  Usable training samples (after warmup): {usable_samples}")
    logger.info(f"  Effective training period: {usable_start_date.date()} to {features_df.index[-1].date()}")
    logger.info("\n[3/4] Merging features with price data...")
    complete_df=features_df.copy();complete_df["vix"]=vix;complete_df["spx"]=spx
    if "calendar_cohort"not in complete_df.columns:
        raise ValueError("❌ calendar_cohort column missing from features! FeatureEngineer.build_complete_features() should have assigned it.")
    logger.info("  Assigning calendar cohorts using feature_engine logic...")
    logger.info("\n[4/4] Validating temporal hygiene...")
    cohort_counts=complete_df["calendar_cohort"].value_counts().sort_index()
    logger.info(f"  Cohort distribution:")
    for cohort,count in cohort_counts.items():logger.info(f"    {cohort}: {count} samples")
    logger.info(f"\n  Final dataframe shape: {complete_df.shape}")
    logger.info(f"  Columns: features ({len(features_df.columns)-2}) + vix + spx + calendar_cohort")
    logger.info("\n✅ Data preparation complete")
    max_date=complete_df.index.max()
    if max_date>pd.Timestamp(TRAINING_END_DATE):logger.warning(f"⚠️  Data extends beyond training end date: {max_date.date()}")
    else:logger.info(f"✅ Temporal hygiene validated: All data ≤ {TRAINING_END_DATE}")
    logger.info("="*80+"\n")
    return complete_df

def validate_configuration() -> Tuple[bool, str]:
    """
    Validate configuration for VIX percentage change target system.

    Returns:
        (is_valid, error_message)
    """
    errors = []

    logger.info("")
    logger.info("=" * 80)
    logger.info("CONFIGURATION VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # [1/4] Target configuration
    logger.info("[1/4] Validating target configuration...")

    # Check target type
    target_type = TARGET_CONFIG.get("target_type")
    if target_type != "vix_pct_change":  # ✅ CHANGED from "log_realized_volatility"
        errors.append(
            f"  ❌ Target type incorrect: {target_type} "
            "(expected 'vix_pct_change')"
        )
    else:
        logger.info("  ✅ Target type correct: vix_pct_change")

    # Check movement bounds (not volatility bounds)
    if "movement_bounds" in TARGET_CONFIG:  # ✅ CHANGED from "volatility_bounds"
        bounds = TARGET_CONFIG["movement_bounds"]
        floor = bounds.get("floor", 0)
        ceiling = bounds.get("ceiling", 0)
        logger.info(f"  ✅ Movement bounds: [{floor}, {ceiling}]")
    else:
        errors.append("  ❌ Movement bounds missing")

    logger.info("")

    # [2/4] Model objectives - check for removed models
    logger.info("[2/4] Validating model objectives...")

    objectives = XGBOOST_CONFIG.get("objectives", {})

    # These should be REMOVED in V3
    if "point" in objectives:
        errors.append("  ❌ Point estimate still present (should be removed)")
    else:
        logger.info("  ✅ Point estimate removed")

    if "uncertainty" in objectives:
        errors.append("  ❌ Uncertainty estimate still present (should be removed)")
    else:
        logger.info("  ✅ Uncertainty estimate removed")

    logger.info("")

    # [3/4] Quantile configuration - check for individual quantile models
    logger.info("[3/4] Validating quantile configuration...")

    expected_quantiles = [
        "quantile_10",
        "quantile_25",
        "quantile_50",
        "quantile_75",
        "quantile_90",
    ]

    missing_quantiles = []
    for q_name in expected_quantiles:
        if q_name not in objectives:
            missing_quantiles.append(q_name)

    if missing_quantiles:
        errors.append(
            f"  ❌ Missing quantile objectives: {', '.join(missing_quantiles)}"
        )
    else:
        logger.info("  ✅ All quantile objectives present")

    # Validate each quantile has correct objective
    for q_name in expected_quantiles:
        if q_name in objectives:
            q_config = objectives[q_name]
            if q_config.get("objective") != "reg:quantileerror":
                errors.append(
                    f"  ❌ {q_name} has wrong objective: {q_config.get('objective')}"
                )

            # Extract expected quantile_alpha from name (e.g., "quantile_50" -> 0.50)
            expected_alpha = int(q_name.split("_")[1]) / 100.0
            actual_alpha = q_config.get("quantile_alpha")

            if actual_alpha != expected_alpha:
                errors.append(
                    f"  ❌ {q_name} has wrong quantile_alpha: "
                    f"{actual_alpha} (expected {expected_alpha})"
                )

    if not missing_quantiles and all(
        objectives.get(q, {}).get("objective") == "reg:quantileerror"
        for q in expected_quantiles
    ):
        logger.info("  ✅ Quantile objectives correctly configured")

    logger.info("")

    # [4/4] Expected model count
    logger.info("[4/4] Validating expected model count...")
    logger.info("  Expected models per cohort: 7")
    logger.info("    - 5 quantile regressors (q10, q25, q50, q75, q90)")
    logger.info("    - 1 direction classifier")
    logger.info("    - 1 confidence scorer")

    # Verify direction and confidence are present
    if "direction" not in objectives:
        errors.append("  ❌ Direction classifier missing")
    else:
        if objectives["direction"].get("objective") != "binary:logistic":
            errors.append("  ❌ Direction classifier has wrong objective")
        else:
            logger.info("  ✅ Direction classifier configured")

    if "confidence" not in objectives:
        errors.append("  ❌ Confidence model missing")
    else:
        if objectives["confidence"].get("objective") != "reg:squarederror":
            errors.append("  ❌ Confidence model has wrong objective")
        else:
            logger.info("  ✅ Confidence model configured")

    logger.info("")

    # Summary
    if errors:
        logger.error("❌ Configuration validation FAILED")
        logger.error("   Please fix the following issues:")
        for error in errors:
            logger.error(error)
        logger.info("=" * 80)
        return False, "\n".join(errors)
    else:
        logger.info("✅ Configuration validation PASSED")
        logger.info("   All checks successful - ready to train")
        logger.info("=" * 80)
        return True, ""


def save_training_report(training_results:dict,output_dir:str="models"):
    output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True)
    report_file=output_path/f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report={"timestamp":datetime.now().isoformat(),"system_version":"v3.0_log_rv","target_type":TARGET_CONFIG.get("target_type"),"training_summary":{"cohorts_trained":list(training_results.models.keys())if hasattr(training_results,"models")else[],"models_per_cohort":7,"quantile_levels":TARGET_CONFIG.get("quantiles",{}).get("levels",[])},"configuration":{"xgboost":XGBOOST_CONFIG,"target":TARGET_CONFIG}}
    with open(report_file,"w")as f:json.dump(report,f,indent=2,default=str)
    logger.info(f"✅ Training report saved: {report_file}")
def display_training_summary(forecaster:ProbabilisticVIXForecaster):
    logger.info("\n"+"="*80);logger.info("TRAINING SUMMARY");logger.info("="*80)
    total_cohorts=len(forecaster.models)
    logger.info(f"  Cohorts trained: {total_cohorts}");logger.info(f"  Models per cohort: 7 (5 quantiles + direction + confidence)");logger.info(f"  Total models: {total_cohorts*7}");logger.info(f"  Feature count: {len(forecaster.feature_names)}")
    logger.info("\n  Cohorts:")
    for cohort in sorted(forecaster.models.keys()):logger.info(f"    - {cohort}")
    logger.info("="*80+"\n")
def main():
    logger.info("\n"+"="*80);logger.info("PROBABILISTIC VIX FORECASTER - TRAINING PIPELINE");logger.info("="*80)
    logger.info(f"Version: 3.0 (Log-Transformed Realized Volatility)");logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}");logger.info(f"Output: models");logger.info("="*80+"\n")
    try:
        logger.info("")
        is_valid,error_msg=validate_configuration()
        if not is_valid:
            logger.error(f"\n❌ Configuration validation failed:\n{error_msg}")
            sys.exit(1)
        logger.info("")
        complete_df=prepare_training_data()
        logger.info("");logger.info("="*80);logger.info("MODEL TRAINING");logger.info("="*80);logger.info("")
        logger.info("Starting training...");logger.info("  Models per cohort: 7 (5 quantiles, direction, confidence)");logger.info("  Primary forecast: Median (50th percentile)");logger.info("  Target: Log-transformed forward realized volatility");logger.info("")
        forecaster=train_probabilistic_forecaster(df=complete_df,save_dir="models")
        display_training_summary(forecaster)
        save_training_report(forecaster,output_dir="models")
        logger.info("\n"+"="*80);logger.info("✅ TRAINING COMPLETE");logger.info("="*80);logger.info(f"  Models saved to: models/");logger.info(f"  Training report: models/training_report_*.json");logger.info("="*80+"\n")
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__=="__main__":
    main()
