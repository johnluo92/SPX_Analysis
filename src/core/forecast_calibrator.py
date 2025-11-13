"""
Forecast Calibrator - Post-Processing for Probabilistic Forecasts

Fixes systematic biases in VIX forecasts:
1. Over-prediction bias (typically +10-15%)
2. Miscalibrated quantiles (intervals too narrow)
3. Backwards confidence scores (high conf = high error)

Usage:
    # One-time setup (after generating 100+ forecasts)
    calibrator = ForecastCalibrator()
    calibrator.fit_from_database()
    calibrator.save()

    # At runtime (in integrated_system_production.py)
    calibrator = ForecastCalibrator.load()
    calibrated = calibrator.calibrate(raw_forecast)
"""

import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForecastCalibrator:
    """
    Calibrates probabilistic forecasts using historical forecast errors.

    Three-step calibration:
    1. Bias Correction: Subtract systematic over/under-prediction
    2. Quantile Adjustment: Widen/narrow prediction intervals
    3. Confidence Recalibration: Fix correlation with forecast error
    """

    def __init__(self):
        self.fitted = False

        # Calibration parameters (learned from data)
        self.bias_correction = 0.0
        self.quantile_adjustments = {}
        self.confidence_mapping = None

        # Diagnostics
        self.n_training_samples = 0
        self.training_date_range = None
        self.metrics = {}

    def fit_from_database(
        self,
        db_path: str = "data_cache/predictions.db",
        min_samples: int = 50,
        start_date: str = None,
        end_date: str = None,
    ) -> bool:
        """
        Learn calibration parameters from prediction database.

        Args:
            db_path: Path to predictions database
            min_samples: Minimum forecasts needed to fit
            start_date: Only use forecasts >= this date (YYYY-MM-DD)
            end_date: Only use forecasts <= this date (YYYY-MM-DD)

        Returns:
            bool: True if calibration was fitted successfully
        """
        try:
            from core.prediction_database import PredictionDatabase
        except ModuleNotFoundError:
            # Running from within core/ directory
            from prediction_database import PredictionDatabase

        logger.info("=" * 80)
        logger.info("FITTING FORECAST CALIBRATOR")
        logger.info("=" * 80)

        # Load predictions with actuals
        db = PredictionDatabase(db_path)
        df = db.get_predictions(
            with_actuals=True, start_date=start_date, end_date=end_date
        )

        if len(df) < min_samples:
            logger.warning(
                f"❌ Insufficient data: {len(df)} forecasts (need >{min_samples})"
            )
            logger.info(f"   Generate more forecasts, then retry calibration")
            return False

        self.n_training_samples = len(df)
        self.training_date_range = (
            df["forecast_date"].min().strftime("%Y-%m-%d"),
            df["forecast_date"].max().strftime("%Y-%m-%d"),
        )

        logger.info(f"📊 Calibration data:")
        logger.info(f"   Samples: {len(df)}")
        logger.info(
            f"   Date range: {self.training_date_range[0]} to {self.training_date_range[1]}"
        )

        # 1. BIAS CORRECTION
        self._fit_bias_correction(df)

        # 2. QUANTILE ADJUSTMENTS
        self._fit_quantile_adjustments(df)

        # 3. CONFIDENCE RECALIBRATION
        self._fit_confidence_mapping(df)

        self.fitted = True

        logger.info("\n✅ Calibration fitted successfully")
        logger.info("=" * 80)

        return True

    def _fit_bias_correction(self, df: pd.DataFrame):
        """Learn systematic over/under-prediction bias."""
        errors = df["point_estimate"] - df["actual_vix_change"]
        self.bias_correction = errors.mean()

        # Diagnostics
        self.metrics["bias_before"] = float(self.bias_correction)
        self.metrics["mae_before"] = float(errors.abs().mean())

        logger.info(f"\n[1/3] Bias Correction")
        logger.info(f"   Systematic bias: {self.bias_correction:+.2f}%")

        if abs(self.bias_correction) > 5:
            logger.warning(
                f"   ⚠️  Large bias detected - model consistently "
                f"{'over' if self.bias_correction > 0 else 'under'}-predicts"
            )

    def _fit_quantile_adjustments(self, df: pd.DataFrame):
        """Learn how to widen/narrow prediction intervals."""
        logger.info(f"\n[2/3] Quantile Calibration")

        quantiles = ["q10", "q25", "q50", "q75", "q90"]
        targets = [0.10, 0.25, 0.50, 0.75, 0.90]

        for q_name, target in zip(quantiles, targets):
            # Compute empirical coverage
            empirical = (df["actual_vix_change"] <= df[q_name]).mean()
            error = empirical - target

            # Compute adjustment factor
            # If empirical > target: intervals too wide, need to narrow (factor < 1)
            # If empirical < target: intervals too narrow, need to widen (factor > 1)
            if abs(error) > 0.05:  # Only adjust if >5pp off
                # Dampen adjustment to avoid overcorrection
                adjustment = 1.0 + (
                    error * -1.5
                )  # Negative because inverse relationship
                adjustment = np.clip(adjustment, 0.5, 2.0)  # Reasonable bounds
            else:
                adjustment = 1.0

            self.quantile_adjustments[q_name] = {
                "empirical": float(empirical),
                "target": float(target),
                "adjustment": float(adjustment),
            }

            status = "✅" if abs(error) < 0.10 else "❌"
            logger.info(
                f"   {status} {q_name}: {empirical:.1%} → {target:.1%} "
                f"(adj={adjustment:.3f})"
            )

    def _fit_confidence_mapping(self, df: pd.DataFrame):
        """Learn relationship between confidence and forecast error."""
        logger.info(f"\n[3/3] Confidence Recalibration")

        corr = df[["confidence_score", "point_error"]].corr().iloc[0, 1]
        logger.info(f"   Raw correlation: {corr:+.3f}")

        if corr > 0.1:
            logger.warning(f"   ⚠️  Confidence is BACKWARDS - will invert")
            self.confidence_mapping = {"method": "invert"}

        elif abs(corr) < 0.1:
            logger.warning(f"   ⚠️  Confidence weakly predictive - will recalibrate")

            # Bin by confidence, compute mean error per bin
            df["conf_bin"], bin_edges = pd.qcut(
                df["confidence_score"], q=5, retbins=True, duplicates="drop"
            )
            error_by_bin = df.groupby("conf_bin", observed=True)["point_error"].mean()

            # Normalize to [0, 1] range (invert: high error = low confidence)
            max_error = error_by_bin.max()
            confidence_calibrated = 1.0 - (error_by_bin / max_error)

            self.confidence_mapping = {
                "method": "binned",
                "bin_edges": bin_edges[1:-1].tolist(),  # Interior edges only
                "calibrated_values": confidence_calibrated.tolist(),
            }

        else:
            logger.info(f"   ✅ Confidence well-calibrated (no adjustment)")
            self.confidence_mapping = {"method": "none"}

    def calibrate(self, forecast: Dict) -> Dict:
        """
        Apply calibration to a single forecast.

        Args:
            forecast: Raw forecast dict with keys:
                - point_estimate
                - quantiles: {q10, q25, q50, q75, q90}
                - confidence_score
                - direction_probability  # CHANGED from regime_probabilities

        Returns:
            Calibrated forecast dict (same structure)
        """
        if not self.fitted:
            warnings.warn("Calibrator not fitted - returning uncalibrated forecast")
            return forecast

        calibrated = forecast.copy()

        # 1. Apply bias correction to point estimate
        calibrated["point_estimate"] = forecast["point_estimate"] - self.bias_correction

        # 2. Adjust quantiles around corrected point estimate
        center = calibrated["point_estimate"]
        quantiles = forecast["quantiles"].copy()

        for q_name, params in self.quantile_adjustments.items():
            if q_name in quantiles:
                # Get distance from original center
                distance = quantiles[q_name] - forecast["point_estimate"]

                # Apply adjustment factor
                adjustment = params["adjustment"]
                new_distance = distance * adjustment

                # Recenter around corrected point estimate
                quantiles[q_name] = center + new_distance

        # Enforce monotonicity: q10 < q25 < q50 < q75 < q90
        q_values = [quantiles[q] for q in ["q10", "q25", "q50", "q75", "q90"]]
        q_values_sorted = sorted(q_values)

        for q_name, value in zip(["q10", "q25", "q50", "q75", "q90"], q_values_sorted):
            quantiles[q_name] = value

        calibrated["quantiles"] = quantiles

        # 3. Recalibrate confidence
        raw_conf = forecast["confidence_score"]

        if self.confidence_mapping["method"] == "invert":
            calibrated["confidence_score"] = 1.0 - raw_conf

        elif self.confidence_mapping["method"] == "binned":
            # Map to calibrated bin
            bin_edges = self.confidence_mapping["bin_edges"]
            calibrated_values = self.confidence_mapping["calibrated_values"]

            # Find which bin raw confidence falls into
            bin_idx = 0
            for i, edge in enumerate(bin_edges):
                if raw_conf > edge:
                    bin_idx = i + 1

            bin_idx = min(bin_idx, len(calibrated_values) - 1)
            calibrated["confidence_score"] = calibrated_values[bin_idx]

        # else: method='none', no adjustment

        # Ensure confidence in valid range
        calibrated["confidence_score"] = np.clip(
            calibrated["confidence_score"], 0.5, 0.99
        )

        # Copy unchanged fields - CHANGED: direction_probability instead of regime_probabilities
        calibrated["direction_probability"] = forecast["direction_probability"]
        calibrated["cohort"] = forecast["cohort"]

        return calibrated

    def save(self, filepath: str = "models/forecast_calibrator.pkl"):
        """Save fitted calibrator to disk."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted calibrator")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "bias_correction": self.bias_correction,
            "quantile_adjustments": self.quantile_adjustments,
            "confidence_mapping": self.confidence_mapping,
            "n_training_samples": self.n_training_samples,
            "training_date_range": self.training_date_range,
            "metrics": self.metrics,
            "fitted": self.fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"💾 Saved calibrator: {filepath}")

    @classmethod
    def load(
        cls, filepath: str = "models/forecast_calibrator.pkl"
    ) -> Optional["ForecastCalibrator"]:
        """
        Load calibrator from disk.

        Returns:
            ForecastCalibrator instance, or None if file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"⚠️  Calibrator not found: {filepath}")
            logger.info(f"   Forecasts will not be calibrated")
            return None

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        calibrator = cls()
        calibrator.bias_correction = state["bias_correction"]
        calibrator.quantile_adjustments = state["quantile_adjustments"]
        calibrator.confidence_mapping = state["confidence_mapping"]
        calibrator.n_training_samples = state["n_training_samples"]
        calibrator.training_date_range = state["training_date_range"]
        calibrator.metrics = state["metrics"]
        calibrator.fitted = state["fitted"]

        logger.info(f"✅ Loaded calibrator: {filepath}")
        logger.info(f"   Trained on {calibrator.n_training_samples} samples")
        logger.info(
            f"   Date range: {calibrator.training_date_range[0]} to {calibrator.training_date_range[1]}"
        )

        return calibrator

    def get_diagnostics(self) -> Dict:
        """Get calibration diagnostics for reporting."""
        if not self.fitted:
            return {"error": "Calibrator not fitted"}

        return {
            "fitted": self.fitted,
            "training_samples": self.n_training_samples,
            "date_range": self.training_date_range,
            "bias_correction": self.bias_correction,
            "quantile_adjustments": self.quantile_adjustments,
            "confidence_method": self.confidence_mapping["method"],
            "metrics": self.metrics,
        }


# Standalone script for fitting calibrator
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory to path so imports work
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n" + "=" * 80)
    print("FORECAST CALIBRATOR - TRAINING")
    print("=" * 80)

    calibrator = ForecastCalibrator()
    success = calibrator.fit_from_database()

    if success:
        calibrator.save()

        print("\n" + "=" * 80)
        print("CALIBRATION DIAGNOSTICS")
        print("=" * 80)

        diag = calibrator.get_diagnostics()
        print(f"\nBias Correction:     {diag['bias_correction']:+.2f}%")
        print(f"Confidence Method:   {diag['confidence_method']}")
        print(f"\nQuantile Adjustments:")
        for q_name, params in diag["quantile_adjustments"].items():
            print(
                f"  {q_name}: {params['empirical']:.1%} → {params['target']:.1%} "
                f"(factor={params['adjustment']:.3f})"
            )

        print("\n" + "=" * 80)
        print("✅ CALIBRATOR READY FOR PRODUCTION")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Integrate into integrated_system_production.py")
        print("  2. Generate new forecasts with calibration")
        print("  3. Run diagnostics/walk_forward_validation.py")

    else:
        print("\n" + "=" * 80)
        print("❌ CALIBRATION FAILED")
        print("=" * 80)
        print("\nGenerate more forecasts first:")
        print("  python integrated_system_production.py --mode batch \\")
        print("    --start-date 2020-01-01 --end-date 2023-12-31")
        sys.exit(1)
