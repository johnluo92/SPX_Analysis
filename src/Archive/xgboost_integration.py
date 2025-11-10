"""
XGBoost Integration - VIX Expansion Forecasting
================================================

Direct integration with IntegratedMarketSystemV4 for feature selection and VIX expansion model training.

Usage:
    python xgboost_integration.py --mode features_only   # Just feature selection
    python xgboost_integration.py --mode full            # Feature selection + model training
    python xgboost_integration.py --mode full --optimize 50 --horizons 1 3 5 10
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
from core.xgboost_trainer_v2 import train_vix_expansion_model
from integrated_system_production import IntegratedMarketSystemV4


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="XGBoost Integration - VIX Expansion Forecasting"
    )

    parser.add_argument(
        "--mode",
        choices=["features_only", "full"],
        default="features_only",
        help="Execution mode",
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

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="VIX expansion threshold (default 15%%)",
    )

    args = parser.parse_args()

    if not ENABLE_TRAINING:
        print(f"\n{'=' * 80}")
        print("WARNING: TRAINING DISABLED (config.ENABLE_TRAINING = False)")
        print("WARNING: Set ENABLE_TRAINING = True in config.py")
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
        f"\nOK: Features built: {len(features.columns)} features, {len(features)} samples"
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

    print(f"\nOK: Selected {len(selected_features)} features")
    print(f"   Saved to: ./models/selected_features_v2.txt")

    if args.mode == "features_only":
        print(f"\n{'=' * 80}")
        print("FEATURE SELECTION COMPLETE")
        print(f"{'=' * 80}")
        return

    # === STEP 3: Train XGBoost with selected features ===
    if args.mode == "full":
        print(f"\n{'=' * 80}")
        print("STEP 3: TRAINING VIX EXPANSION MODEL")
        print(f"{'=' * 80}")

        # Filter to selected features
        filtered_features = features[selected_features]
        system.orchestrator.features = filtered_features

        # Train model
        trainer = train_vix_expansion_model(
            system,
            horizons=args.horizons,
            optimize_hyperparams=args.optimize,
            expansion_threshold=args.threshold,
            crisis_balanced=True,
            compute_shap=True,
            verbose=True,
        )

        print(f"\nOK: VIX expansion model training complete")
        print(f"   Horizons trained: {trainer.trained_horizons}")
        print(f"   Expansion threshold: {args.threshold:.1%}")
        print(f"   Models saved to: ./models/")

        # Export metadata
        output_dir = Path("./json_data")
        output_dir.mkdir(exist_ok=True, parents=True)

        model_metadata = {
            "timestamp": datetime.now().isoformat(),
            "selected_features_count": len(selected_features),
            "trained_horizons": trainer.trained_horizons,
            "expansion_threshold": args.threshold,
            "model_files": {
                f"{horizon}d": f"./models/vix_expansion_{horizon}d.json"
                for horizon in trainer.trained_horizons
            },
        }

        with open(output_dir / "xgboost_models.json", "w") as f:
            json.dump(model_metadata, f, indent=2)

        print(f"\nOK: Exported metadata to ./json_data/xgboost_models.json")

    print(f"\n{'=' * 80}")
    print("INTEGRATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
