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

from core.feature_engineer import FeatureEngineer

# ============================================================
# V3 IMPORT: Using new trainer with log-RV target
# ============================================================
from core.xgboost_trainer_v3 import (
    ProbabilisticVIXForecaster,
    train_probabilistic_forecaster,
)

from config import (
    COHORT_CONFIG,
    FEATURE_CONFIG,
    FORECASTING_CONFIG,
    TARGET_CONFIG,
    XGBOOST_CONFIG,
)
from core.data_fetcher import UnifiedDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger(__name__)


def prepare_training_data():
    """
    Fetch and prepare data for training.

    Returns:
        Tuple of (features_df, spx_returns_series, market_data)
    """

    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION")
    logger.info("=" * 80)

    # Initialize data fetcher
    logger.info("\n[1/3] Fetching market data...")
    data_fetcher = UnifiedDataFetcher()

    # Fetch comprehensive data (10 years for robust training)
    market_data = data_fetcher.fetch_all_data(years=10)

    # Validate data
    required_keys = ["vix", "spx", "treasury", "macro"]
    for key in required_keys:
        if key not in market_data or market_data[key] is None:
            raise ValueError(f"Missing required data: {key}")
        logger.info(f"  {key}: {len(market_data[key])} rows")

    # Engineer features
    logger.info("\n[2/3] Engineering features...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_features(market_data)

    logger.info(
        f"  Features created: {len(features_df.columns)}\n"
        f"  Date range: {features_df.index[0].date()} to {features_df.index[-1].date()}\n"
        f"  Total samples: {len(features_df)}"
    )

    # Extract SPX returns for realized volatility calculation
    logger.info("\n[3/3] Extracting SPX returns...")

    if "spx" in market_data and "close" in market_data["spx"].columns:
        spx_prices = market_data["spx"]["close"]
        spx_returns = spx_prices.pct_change()

        logger.info(
            f"  SPX returns: {len(spx_returns)} values\n"
            f"  Date range: {spx_returns.index[0].date()} to {spx_returns.index[-1].date()}"
        )
    else:
        raise ValueError("SPX price data not available")

    logger.info("\n✅ Data preparation complete")
    logger.info("=" * 80 + "\n")

    return features_df, spx_returns, market_data


def validate_configuration():
    """
    Validate that configuration is correct for V3 system.

    Checks:
    1. Target type is log_realized_volatility
    2. Volatility bounds are set
    3. Point estimate removed from objectives
    4. Quantile config is correct
    """

    logger.info("\n" + "=" * 80)
    logger.info("CONFIGURATION VALIDATION")
    logger.info("=" * 80)

    validation_passed = True

    # Check target config
    logger.info("\n[1/4] Validating target configuration...")

    if TARGET_CONFIG.get("target_type") != "log_realized_volatility":
        logger.error(
            f"❌ Incorrect target_type: {TARGET_CONFIG.get('target_type')}\n"
            f"   Expected: 'log_realized_volatility'"
        )
        validation_passed = False
    else:
        logger.info("  ✅ Target type correct: log_realized_volatility")

    if "volatility_bounds" not in TARGET_CONFIG:
        logger.error("❌ Missing volatility_bounds in TARGET_CONFIG")
        validation_passed = False
    else:
        bounds = TARGET_CONFIG["volatility_bounds"]
        logger.info(f"  ✅ Volatility bounds: [{bounds['floor']}, {bounds['ceiling']}]")

    # Check objectives
    logger.info("\n[2/4] Validating model objectives...")

    objectives = XGBOOST_CONFIG.get("objectives", {})

    if "point" in objectives:
        logger.error(
            "❌ 'point' objective still in config\n"
            "   Remove 'point' from XGBOOST_CONFIG['objectives']"
        )
        validation_passed = False
    else:
        logger.info("  ✅ Point estimate removed")

    if "uncertainty" in objectives:
        logger.error(
            "❌ 'uncertainty' objective still in config\n"
            "   Remove 'uncertainty' from XGBOOST_CONFIG['objectives']"
        )
        validation_passed = False
    else:
        logger.info("  ✅ Uncertainty estimate removed")

    # Validate quantile config
    logger.info("\n[3/4] Validating quantile configuration...")

    if "quantile" in objectives:
        quantile_config = objectives["quantile"]
        quantiles = quantile_config.get("quantiles", [])

        expected_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

        if quantiles == expected_quantiles:
            logger.info(f"  ✅ Quantiles correct: {quantiles}")
        else:
            logger.error(
                f"❌ Incorrect quantiles: {quantiles}\n"
                f"   Expected: {expected_quantiles}"
            )
            validation_passed = False
    else:
        logger.error("❌ Quantile objective missing")
        validation_passed = False

    # Validate model count
    logger.info("\n[4/4] Validating expected model count...")

    expected_models = 7  # 5 quantiles + direction + confidence
    logger.info(f"  Expected models per cohort: {expected_models}")
    logger.info("    - 5 quantile regressors (q10, q25, q50, q75, q90)")
    logger.info("    - 1 direction classifier")
    logger.info("    - 1 confidence scorer")

    if validation_passed:
        logger.info("\n✅ Configuration validation PASSED")
    else:
        logger.error("\n❌ Configuration validation FAILED")
        logger.error("   Please fix configuration before training")

    logger.info("=" * 80 + "\n")

    return validation_passed


def display_training_summary(training_results: dict):
    """
    Display comprehensive training summary.

    Args:
        training_results: Dictionary returned by train_probabilistic_forecaster
    """

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    # Overall statistics
    logger.info("\nOverall Statistics:")
    logger.info(f"  Total samples: {training_results.get('total_samples', 'N/A')}")
    logger.info(f"  Total features: {training_results.get('total_features', 'N/A')}")
    logger.info(f"  Training time: {training_results.get('training_time', 'N/A'):.1f}s")

    # Cohort information
    if "cohort_models" in training_results:
        cohorts = training_results["cohort_models"]
        logger.info(f"\nCohorts trained: {len(cohorts)}")

        for cohort_name, cohort_data in cohorts.items():
            logger.info(f"\n  {cohort_name}:")
            logger.info(f"    Samples: {cohort_data.get('samples', 'N/A')}")
            logger.info(f"    Models: {cohort_data.get('num_models', 'N/A')}")

            if "models" in cohort_data:
                logger.info("    Model performance:")
                for model_name, metrics in cohort_data["models"].items():
                    if "val_mae" in metrics:
                        logger.info(f"      {model_name}: MAE={metrics['val_mae']:.4f}")

    # Model counts validation
    logger.info("\nModel Count Validation:")
    if "cohort_models" in training_results:
        for cohort_name, cohort_data in training_results["cohort_models"].items():
            num_models = cohort_data.get("num_models", 0)
            status = "✅" if num_models == 7 else "❌"
            logger.info(f"  {status} {cohort_name}: {num_models} models")

    # Files saved
    if "model_files" in training_results:
        logger.info(f"\nModel Files Saved: {len(training_results['model_files'])}")
        for file_path in training_results["model_files"]:
            logger.info(f"  {Path(file_path).name}")

    logger.info("\n" + "=" * 80 + "\n")


def save_training_report(training_results: dict, output_dir: str = "models"):
    """
    Save detailed training report to JSON.

    Args:
        training_results: Dictionary returned by train_probabilistic_forecaster
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
        "horizon": FORECASTING_CONFIG.get("horizon"),
        "training_results": training_results,
        "configuration": {
            "xgboost": XGBOOST_CONFIG,
            "target": TARGET_CONFIG,
            "forecasting": FORECASTING_CONFIG,
        },
    }

    # Save report
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"✅ Training report saved: {report_file}")


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

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
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
            if not validate_configuration():
                logger.error("❌ Configuration validation failed. Aborting.")
                sys.exit(1)
        else:
            logger.warning("⚠️  Skipping configuration validation (not recommended)")

        # Step 2: Prepare training data
        features_df, spx_returns, market_data = prepare_training_data()

        # Step 3: Train models
        logger.info("\n" + "=" * 80)
        logger.info("MODEL TRAINING")
        logger.info("=" * 80)
        logger.info("\nStarting training...")
        logger.info(f"  Models per cohort: 7 (5 quantiles, direction, confidence)")
        logger.info(f"  Primary forecast: Median (50th percentile)")
        logger.info(f"  Target: Log-transformed forward realized volatility\n")

        training_results = train_probabilistic_forecaster(
            features_df=features_df,
            spx_returns=spx_returns,
            save_dir=args.output_dir,
        )

        if training_results is None:
            logger.error("❌ Training failed")
            sys.exit(1)

        # Step 4: Display results
        display_training_summary(training_results)

        # Step 5: Save report
        save_training_report(training_results, output_dir=args.output_dir)

        # Step 6: Validate saved models
        logger.info("\n" + "=" * 80)
        logger.info("MODEL VALIDATION")
        logger.info("=" * 80)

        forecaster = ProbabilisticVIXForecaster()

        try:
            forecaster.load_models(load_dir=args.output_dir)
            logger.info("✅ Models loaded successfully")

            # Test prediction
            logger.info("\nTesting prediction capability...")
            test_features = features_df.iloc[-1]
            test_date = features_df.index[-1]

            predictions = forecaster.predict(
                features=test_features,
                forecast_date=test_date,
            )

            if predictions:
                logger.info("✅ Test prediction successful")
                logger.info(
                    f"   Median forecast: {predictions['median_forecast']:+.2f}%"
                )
                logger.info(f"   Quantiles: {list(predictions['quantiles'].keys())}")
                logger.info(
                    f"   Direction prob: {predictions['direction_probability']:.2f}"
                )
            else:
                logger.error("❌ Test prediction failed")
                sys.exit(1)

        except Exception as e:
            logger.error(f"❌ Model loading/prediction failed: {e}")
            sys.exit(1)

        # Success
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nModels saved to: {args.output_dir}")
        logger.info(
            f"Total training time: {training_results.get('training_time', 'N/A'):.1f}s"
        )
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
