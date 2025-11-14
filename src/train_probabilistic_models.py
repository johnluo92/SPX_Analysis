"""
Training Script for Probabilistic VIX Forecaster V3

CRITICAL CHANGES FROM V2:
1. Imports from xgboost_trainer_v3 (log-RV system)
2. Target is log-transformed forward realized volatility
3. 7 models per cohort (removed point and uncertainty)
4. Models: 5 quantiles + direction + confidence
5. Enhanced validation and diagnostics

Author: VIX Forecasting System
Last Updated: 2025-11-13
Version: 3.0 (Log-RV Quantile Regression)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from config import (
    CALENDAR_COHORTS,
    TARGET_CONFIG,
    TRAINING_END_DATE,
    TRAINING_YEARS,
    XGBOOST_CONFIG,
)
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer

# ============================================================
# V3 IMPORT: Using new trainer with log-RV target
# ============================================================
from core.xgboost_trainer_v3 import (
    ProbabilisticVIXForecaster,
    train_probabilistic_forecaster,
)
from logging_config import get_logger, setup_logging

# Create logs directory before configuring logging
Path("logs").mkdir(exist_ok=True)

# Configure logging
setup_logging(level=logging.INFO, quiet_mode=False, log_file="logs/training.log")
logger = get_logger(__name__)


def prepare_training_data():
    """
    Fetch and prepare data for training using FeatureEngineer.build_complete_features().

    V3 CHANGE: Returns single merged dataframe with features, VIX, SPX, and calendar_cohort.
    The trainer calculates realized volatility targets internally.

    Returns:
        Complete dataframe ready for trainer (with calendar_cohort)
    """

    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION")
    logger.info("=" * 80)

    # Initialize data fetcher and feature engineer
    logger.info("\n[1/3] Initializing feature engineering pipeline...")
    data_fetcher = UnifiedDataFetcher()
    feature_engineer = FeatureEngineer(data_fetcher)

    # Build complete features (fetches data internally)
    logger.info("\n[2/3] Building complete feature set...")
    logger.info(f"  Training window: {TRAINING_YEARS} years")
    logger.info(f"  Training end date: {TRAINING_END_DATE}")

    result = feature_engineer.build_complete_features(
        years=TRAINING_YEARS,
        end_date=TRAINING_END_DATE,  # Use configured training end date
    )

    features_df = result["features"]
    spx = result["spx"]
    vix = result["vix"]

    logger.info(
        f"  Features created: {len(features_df.columns)}\n"
        f"  Date range: {features_df.index[0].date()} to {features_df.index[-1].date()}\n"
        f"  Total samples: {len(features_df)}"
    )

    # [3/3] Merge everything into single dataframe for trainer
    logger.info("\n[3/3] Merging features with price data...")

    # Create complete dataframe
    complete_df = features_df.copy()
    complete_df["vix"] = vix
    complete_df["spx"] = spx

    # Add calendar cohort assignment
    logger.info("  Assigning calendar cohorts...")
    complete_df["calendar_cohort"] = complete_df.index.to_series().apply(
        lambda d: _get_calendar_cohort(d)
    )

    cohort_counts = complete_df["calendar_cohort"].value_counts().sort_index()
    logger.info(f"  Cohort distribution:")
    for cohort, count in cohort_counts.items():
        logger.info(f"    {cohort}: {count} samples")

    logger.info(f"\n  Final dataframe shape: {complete_df.shape}")
    logger.info(
        f"  Columns: features ({len(features_df.columns)}) + vix + spx + calendar_cohort"
    )

    logger.info("\n✅ Data preparation complete")
    logger.info("=" * 80 + "\n")

    return complete_df


def _get_calendar_cohort(date: pd.Timestamp) -> str:
    """
    Assign calendar cohort based on date.

    Uses configuration from CALENDAR_COHORTS in config.py
    """
    month = date.month

    # Map from config structure to cohort names
    # CALENDAR_COHORTS = {"q1": [1,2,3], "q2": [4,5,6], ...}
    for cohort_name, months in CALENDAR_COHORTS.items():
        if month in months:
            return cohort_name

    # Fallback (shouldn't happen)
    return "q1"


def validate_configuration() -> Tuple[bool, str]:
    """
    Validate configuration for refactored quantile regression system.

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
    if target_type != "log_realized_volatility":
        errors.append(
            f"  ❌ Target type incorrect: {target_type} "
            "(expected 'log_realized_volatility')"
        )
    else:
        logger.info("  ✅ Target type correct: log_realized_volatility")

    # Check volatility bounds
    if "volatility_bounds" in TARGET_CONFIG:
        bounds = TARGET_CONFIG["volatility_bounds"]
        floor = bounds.get("floor", 0)
        ceiling = bounds.get("ceiling", 0)
        logger.info(f"  ✅ Volatility bounds: [{floor}, {ceiling}]")
    else:
        errors.append("  ❌ Volatility bounds missing")

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


def save_training_report(training_results: dict, output_dir: str = "models"):
    """
    Save detailed training report to JSON.

    Args:
        training_results: ProbabilisticVIXForecaster instance returned by trainer
        output_dir: Directory to save report
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = (
        output_path / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Prepare report
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_version": "v3.0_log_rv",
        "target_type": TARGET_CONFIG.get("target_type"),
        "training_summary": {
            "cohorts_trained": list(training_results.models.keys())
            if hasattr(training_results, "models")
            else [],
            "models_per_cohort": 7,
            "quantile_levels": TARGET_CONFIG.get("quantiles", {}).get("levels", []),
        },
        "configuration": {
            "xgboost": XGBOOST_CONFIG,
            "target": TARGET_CONFIG,
        },
    }

    # Save report
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"✅ Training report saved: {report_file}")


def display_training_summary(forecaster: ProbabilisticVIXForecaster):
    """Display training summary."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    if hasattr(forecaster, "models"):
        cohorts = list(forecaster.models.keys())
        logger.info(f"\nCohorts trained: {len(cohorts)}")
        for cohort in cohorts:
            model_names = list(forecaster.models[cohort].keys())
            logger.info(f"  {cohort}: {len(model_names)} models - {model_names}")

    logger.info("\n" + "=" * 80 + "\n")


def main():
    """Main training entry point."""

    parser = argparse.ArgumentParser(
        description="Train Probabilistic VIX Forecaster V3"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip configuration validation (not recommended)",
    )

    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if models exist",
    )

    args = parser.parse_args()

    # Create necessary directories (logs already created above)
    Path(args.output_dir).mkdir(exist_ok=True)

    # Display header
    print("\n" + "=" * 80)
    print("PROBABILISTIC VIX FORECASTER - TRAINING PIPELINE")
    print("=" * 80)
    print(f"Version: 3.0 (Log-Transformed Realized Volatility)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {args.output_dir}")
    print("=" * 80 + "\n")

    try:
        # Step 1: Validate configuration
        if not args.skip_validation:
            is_valid, error_msg = validate_configuration()
            if not is_valid:
                logger.error("❌ Configuration validation failed. Aborting.")
                sys.exit(1)
        else:
            logger.warning("⚠️  Skipping configuration validation (not recommended)")

        # Step 2: Prepare training data
        complete_df = prepare_training_data()

        # Step 3: Train models
        logger.info("\n" + "=" * 80)
        logger.info("MODEL TRAINING")
        logger.info("=" * 80)
        logger.info("\nStarting training...")
        logger.info(f"  Models per cohort: 7 (5 quantiles, direction, confidence)")
        logger.info(f"  Primary forecast: Median (50th percentile)")
        logger.info(f"  Target: Log-transformed forward realized volatility\n")

        # V3 CORRECTED CALL: Pass single dataframe
        forecaster = train_probabilistic_forecaster(
            df=complete_df,
            save_dir=args.output_dir,
        )

        if forecaster is None:
            logger.error("❌ Training failed")
            sys.exit(1)

        # Step 4: Display results
        display_training_summary(forecaster)

        # Step 5: Save report
        save_training_report(forecaster, output_dir=args.output_dir)

        # Step 6: Validate saved models
        logger.info("\n" + "=" * 80)
        logger.info("MODEL VALIDATION")
        logger.info("=" * 80)

        test_forecaster = ProbabilisticVIXForecaster()

        try:
            # Load first cohort as test
            cohorts = list(forecaster.models.keys())
            if cohorts:
                test_cohort = cohorts[0]
                test_forecaster.load(cohort=test_cohort, load_dir=args.output_dir)
                logger.info(
                    f"✅ Models loaded successfully (tested cohort: {test_cohort})"
                )

                # Test prediction capability
                logger.info("\nTesting prediction capability...")

                # Create test dataframe with only the feature columns
                test_df = complete_df[test_forecaster.feature_names].iloc[[-1]].copy()
                current_vix = float(complete_df["vix"].iloc[-1])

                predictions = test_forecaster.predict(
                    X=test_df,
                    cohort=test_cohort,
                    current_vix=current_vix,
                )

                if predictions:
                    logger.info("✅ Test prediction successful")
                    logger.info(
                        f"   Median forecast: {predictions['median_forecast']:+.2f}%"
                    )
                    logger.info(
                        f"   Quantiles: {list(predictions['quantiles'].keys())}"
                    )
                    logger.info(
                        f"   Direction prob: {predictions['direction_probability']:.2f}"
                    )
                else:
                    logger.error("❌ Test prediction failed")
                    sys.exit(1)
            else:
                logger.error("❌ No cohorts were trained")
                sys.exit(1)

        except Exception as e:
            logger.error(f"❌ Model loading/prediction failed: {e}", exc_info=True)
            sys.exit(1)

        # Success
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nModels saved to: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Review training report")
        logger.info(
            "  2. Run: python src/integrated_system_production.py --mode predict"
        )
        logger.info(
            "  3. Backfill actuals: python src/integrated_system_production.py --mode backfill"
        )
        logger.info("=" * 80 + "\n")

        sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
