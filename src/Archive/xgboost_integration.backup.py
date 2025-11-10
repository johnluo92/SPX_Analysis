"""
XGBoost Integration - Streamlined
==================================
Direct integration with IntegratedMarketSystemV4 for feature selection and XGBoost training.

Usage:
    python xgboost_integration.py --mode features_only   # Just feature selection
    python xgboost_integration.py --mode full            # Feature selection + XGBoost training
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import ENABLE_TRAINING, TRAINING_YEARS
from core.xgboost_feature_selector_v2 import run_intelligent_feature_selection
from core.xgboost_trainer_v2 import train_enhanced_xgboost
from integrated_system_production import IntegratedMarketSystemV4


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="XGBoost Integration")

    parser.add_argument(
        "--mode",
        choices=["features_only", "full"],
        default="features_only",
        help="Execution mode",
    )

    parser.add_argument(
        "--optimize_CV",
        action="store_true",
        help="Enable nested CV for hyperparameter optimization (slower)",
    )

    parser.add_argument(
        "--optimize",
        type=int,
        default=0,
        metavar="N",
        help="Number of Optuna trials for hyperparameter optimization (0=use defaults, 50=standard, 200=thorough)",
    )

    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5],
        help="Horizons to train (e.g., --horizons 1 3 5 10)",
    )

    args = parser.parse_args()

    if not ENABLE_TRAINING:
        print(f"\n{'=' * 80}")
        print("⚠️ TRAINING DISABLED (config.ENABLE_TRAINING = False)")
        print("⚠️ Set ENABLE_TRAINING = True in config.py")
        print(f"{'=' * 80}\n")
        return

    # === STEP 1: Build features ONLY (no anomaly training) ===
    print(f"\n{'=' * 80}")
    print("STEP 1: BUILDING FEATURES")
    print(f"{'=' * 80}")

    system = IntegratedMarketSystemV4()

    # Build features without full system training
    print("\nBuilding feature set...")
    feature_data = system.feature_engine.build_complete_features(years=TRAINING_YEARS)
    features = feature_data["features"]
    vix = feature_data["vix"]
    spx = feature_data["spx"]

    # Store in orchestrator for compatibility
    system.orchestrator.features = features
    system.orchestrator.vix_ml = vix
    system.orchestrator.spx_ml = spx
    system.trained = True  # Mark as trained so selector doesn't complain

    print(
        f"\n✅ Features built: {len(features.columns)} features, {len(features)} samples"
    )

    # === STEP 2: Feature selection ===
    print(f"\n{'=' * 80}")
    print("STEP 2: FEATURE SELECTION")
    print(f"{'=' * 80}")

    selection_results = run_intelligent_feature_selection(
        system,
        horizons=args.horizons,
        min_stability=0.3,
        max_correlation=0.95,
        preserve_forward_indicators=True,
        verbose=True,
    )

    selected_features = selection_results["selected_features"]

    print(f"\n✅ Selected {len(selected_features)} features")
    print(f"   Saved to: ./models/selected_features_v2.txt")

    if args.mode == "features_only":
        print(f"\n{'=' * 80}")
        print("✅ FEATURE SELECTION COMPLETE")
        print(f"{'=' * 80}")
        return

    # === STEP 3: Train XGBoost with selected features ===
    if args.mode == "full":
        print(f"\n{'=' * 80}")
        print("STEP 3: TRAINING XGBOOST")
        print(f"{'=' * 80}")

        # Filter to selected features
        filtered_features = features[selected_features]
        system.orchestrator.features = filtered_features

        # Train XGBoost
        xgb_trainer = train_enhanced_xgboost(
            system,
            horizons=args.horizons,  # Pass horizons
            optimize_hyperparams=args.optimize,
            crisis_balanced=True,
            compute_shap=True,
            verbose=True,
        )

        print(f"\n✅ XGBoost training complete")
        print(f"   Horizons trained: {xgb_trainer.trained_horizons}")
        print(f"   Models saved to: ./models/")

        # Define output directory
        output_dir = Path("./json_data")
        output_dir.mkdir(exist_ok=True, parents=True)

        model_metadata = {
            "timestamp": datetime.now().isoformat(),
            "selected_features_count": len(selected_features),
            "trained_horizons": xgb_trainer.trained_horizons,
            "model_files": {
                f"{horizon}d": {
                    "regime": f"./models/regime_classifier_v2_{horizon}d.json",
                    "range": f"./models/range_predictor_v2_{horizon}d.json",
                }
                for horizon in xgb_trainer.trained_horizons
            },
        }

        with open(output_dir / "xgboost_models.json", "w") as f:
            json.dump(model_metadata, f, indent=2)

        print(f"\n✅ Exported metadata to ./json_data/xgboost_models.json")

    print(f"\n{'=' * 80}")
    print("✅ INTEGRATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
