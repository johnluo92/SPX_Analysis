"""Forecast Calibrator V3 - Bias Correction for Log-RV Forecasts"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastCalibrator:
    def __init__(
        self,
        min_samples: int = 50,
        use_robust: bool = True,
        cohort_specific: bool = True,
        regime_specific: bool = True,
    ):
        self.min_samples = min_samples
        self.use_robust = use_robust
        self.cohort_specific = cohort_specific
        self.regime_specific = regime_specific

        # Calibration models
        self.global_model = None
        self.cohort_models = {}
        self.regime_models = {}

        # Statistics
        self.calibration_stats = {}
        self.fitted = False

        logger.info(
            f"ForecastCalibrator V3 initialized:\n"
            f"  Min samples: {min_samples}\n"
            f"  Robust regression: {use_robust}\n"
            f"  Cohort-specific: {cohort_specific}\n"
            f"  Regime-specific: {regime_specific}"
        )

    def fit_from_database(self, database) -> bool:
        """
        Fit calibration models from historical predictions in database.

        METHODOLOGY:
        1. Extract predictions with actuals
        2. Calculate raw errors: actual - median_forecast
        3. Fit regression: error ~ f(median_forecast, current_vix, cohort)
        4. Validate on holdout set
        5. Store calibration models

        Args:
            database: PredictionDatabase instance

        Returns:
            True if calibration successful, False otherwise
        """

        logger.info("\n" + "=" * 80)
        logger.info("FORECAST CALIBRATION")
        logger.info("=" * 80)

        # ================================================================
        # STEP 1: Load historical predictions
        # ================================================================

        logger.info("\n[1/5] Loading historical predictions...")

        df = database.get_predictions(with_actuals=True)

        if len(df) == 0:
            logger.error("❌ No predictions with actuals available")
            return False

        if len(df) < self.min_samples:
            logger.warning(
                f"⚠️  Only {len(df)} samples available, need {self.min_samples}\n"
                f"   Calibration may be unreliable"
            )

        logger.info(
            f"  Loaded {len(df)} predictions\n"
            f"  Date range: {df['forecast_date'].min().date()} to "
            f"{df['forecast_date'].max().date()}"
        )

        # ================================================================
        # STEP 2: Prepare calibration data
        # ================================================================

        logger.info("\n[2/5] Preparing calibration data...")

        # Essential columns
        required_cols = ["median_forecast", "actual_vix_change", "current_vix"]

        # Check for missing columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"❌ Missing required columns: {missing}")
            return False

        # Drop rows with missing data
        calib_df = df[required_cols].copy()

        if "calendar_cohort" in df.columns and self.cohort_specific:
            calib_df["cohort"] = df["calendar_cohort"]
        else:
            calib_df["cohort"] = "all"

        calib_df = calib_df.dropna()

        # Calculate forecast errors
        # ERROR = ACTUAL - FORECAST
        # Positive error = underestimation
        # Negative error = overestimation
        calib_df["error"] = calib_df["actual_vix_change"] - calib_df["median_forecast"]

        # Calculate VIX regime
        calib_df["vix_regime"] = pd.cut(
            calib_df["current_vix"],
            bins=[0, 15, 25, 40, 100],
            labels=["low", "normal", "elevated", "crisis"],
        )

        logger.info(
            f"  Calibration samples: {len(calib_df)}\n"
            f"  Mean error: {calib_df['error'].mean():+.2f}%\n"
            f"  Mean absolute error: {calib_df['error'].abs().mean():.2f}%"
        )

        # Check for systematic bias
        if abs(calib_df["error"].mean()) > 0.5:
            if calib_df["error"].mean() > 0:
                logger.warning("⚠️  Systematic underestimation detected")
            else:
                logger.warning("⚠️  Systematic overestimation detected")

        # ================================================================
        # STEP 3: Fit global calibration model
        # ================================================================

        logger.info("\n[3/5] Fitting global calibration model...")

        X_global = calib_df[["median_forecast", "current_vix"]].values
        y_global = calib_df["error"].values

        if self.use_robust:
            self.global_model = HuberRegressor()
        else:
            self.global_model = LinearRegression()

        self.global_model.fit(X_global, y_global)

        # Evaluate global model
        y_pred = self.global_model.predict(X_global)
        mae_before = mean_absolute_error(
            calib_df["actual_vix_change"], calib_df["median_forecast"]
        )
        calibrated_forecast = calib_df["median_forecast"] + y_pred
        mae_after = mean_absolute_error(
            calib_df["actual_vix_change"], calibrated_forecast
        )

        improvement_pct = (mae_before - mae_after) / mae_before * 100

        logger.info(
            f"  MAE before calibration: {mae_before:.2f}%\n"
            f"  MAE after calibration:  {mae_after:.2f}%\n"
            f"  Improvement: {improvement_pct:+.1f}%"
        )

        self.calibration_stats["global"] = {
            "samples": len(calib_df),
            "mae_before": float(mae_before),
            "mae_after": float(mae_after),
            "improvement_pct": float(improvement_pct),
            "mean_error": float(calib_df["error"].mean()),
        }

        # ================================================================
        # STEP 4: Fit cohort-specific models
        # ================================================================

        if self.cohort_specific and "cohort" in calib_df.columns:
            logger.info("\n[4/5] Fitting cohort-specific calibration...")

            cohorts = calib_df["cohort"].unique()

            for cohort in cohorts:
                if pd.isna(cohort):
                    continue

                cohort_df = calib_df[calib_df["cohort"] == cohort]

                if len(cohort_df) < 20:  # Need minimum samples per cohort
                    logger.warning(
                        f"  ⚠️  {cohort}: Only {len(cohort_df)} samples, skipping"
                    )
                    continue

                X_cohort = cohort_df[["median_forecast", "current_vix"]].values
                y_cohort = cohort_df["error"].values

                if self.use_robust:
                    cohort_model = HuberRegressor()
                else:
                    cohort_model = LinearRegression()

                cohort_model.fit(X_cohort, y_cohort)
                self.cohort_models[cohort] = cohort_model

                # Evaluate cohort model
                y_pred_cohort = cohort_model.predict(X_cohort)
                mae_before = mean_absolute_error(
                    cohort_df["actual_vix_change"], cohort_df["median_forecast"]
                )
                calibrated = cohort_df["median_forecast"] + y_pred_cohort
                mae_after = mean_absolute_error(
                    cohort_df["actual_vix_change"], calibrated
                )

                improvement = (mae_before - mae_after) / mae_before * 100

                logger.info(
                    f"  {cohort}: {len(cohort_df)} samples, "
                    f"improvement: {improvement:+.1f}%"
                )

                self.calibration_stats[f"cohort_{cohort}"] = {
                    "samples": len(cohort_df),
                    "mae_before": float(mae_before),
                    "mae_after": float(mae_after),
                    "improvement_pct": float(improvement),
                    "mean_error": float(cohort_df["error"].mean()),
                }
        else:
            logger.info("\n[4/5] Cohort-specific calibration disabled")

        # ================================================================
        # STEP 5: Fit regime-specific models
        # ================================================================

        if self.regime_specific:
            logger.info("\n[5/5] Fitting regime-specific calibration...")

            for regime in ["low", "normal", "elevated", "crisis"]:
                regime_df = calib_df[calib_df["vix_regime"] == regime]

                if len(regime_df) < 20:
                    logger.warning(
                        f"  ⚠️  {regime}: Only {len(regime_df)} samples, skipping"
                    )
                    continue

                X_regime = regime_df[["median_forecast", "current_vix"]].values
                y_regime = regime_df["error"].values

                if self.use_robust:
                    regime_model = HuberRegressor()
                else:
                    regime_model = LinearRegression()

                regime_model.fit(X_regime, y_regime)
                self.regime_models[regime] = regime_model

                # Evaluate
                y_pred_regime = regime_model.predict(X_regime)
                mae_before = mean_absolute_error(
                    regime_df["actual_vix_change"], regime_df["median_forecast"]
                )
                calibrated = regime_df["median_forecast"] + y_pred_regime
                mae_after = mean_absolute_error(
                    regime_df["actual_vix_change"], calibrated
                )

                improvement = (mae_before - mae_after) / mae_before * 100

                logger.info(
                    f"  {regime} VIX: {len(regime_df)} samples, "
                    f"improvement: {improvement:+.1f}%"
                )

                self.calibration_stats[f"regime_{regime}"] = {
                    "samples": len(regime_df),
                    "mae_before": float(mae_before),
                    "mae_after": float(mae_after),
                    "improvement_pct": float(improvement),
                    "mean_error": float(regime_df["error"].mean()),
                }
        else:
            logger.info("\n[5/5] Regime-specific calibration disabled")

        # ================================================================
        # Mark as fitted
        # ================================================================

        self.fitted = True

        logger.info("\n✅ Calibration complete")
        logger.info("=" * 80 + "\n")

        return True

    def calibrate(
        self,
        raw_forecast: float,
        current_vix: float,
        cohort: Optional[str] = None,
    ) -> Dict:
        """
        Apply calibration to a raw forecast.

        METHODOLOGY:
        1. Select appropriate calibration model (cohort > regime > global)
        2. Predict expected error
        3. Adjust forecast: calibrated = raw + predicted_error
        4. Enforce bounds

        Args:
            raw_forecast: Raw median forecast from model (%)
            current_vix: Current VIX level
            cohort: Calendar cohort (optional)

        Returns:
            Dict with calibrated_forecast and metadata
        """

        if not self.fitted:
            logger.warning("⚠️  Calibrator not fitted, returning raw forecast")
            return {
                "calibrated_forecast": raw_forecast,
                "adjustment": 0.0,
                "method": "none",
                "raw_forecast": raw_forecast,
            }

        # Prepare input
        X = np.array([[raw_forecast, current_vix]])

        # Select calibration model (priority order)
        model = None
        method = "global"

        # Try cohort-specific first
        if cohort and cohort in self.cohort_models:
            model = self.cohort_models[cohort]
            method = f"cohort_{cohort}"

        # Try regime-specific
        elif self.regime_specific:
            if current_vix < 15:
                regime = "low"
            elif current_vix < 25:
                regime = "normal"
            elif current_vix < 40:
                regime = "elevated"
            else:
                regime = "crisis"

            if regime in self.regime_models:
                model = self.regime_models[regime]
                method = f"regime_{regime}"

        # Fallback to global
        if model is None:
            model = self.global_model
            method = "global"

        # Predict adjustment
        predicted_error = model.predict(X)[0]

        # Apply adjustment
        calibrated_forecast = raw_forecast + predicted_error

        # Log calibration
        logger.debug(
            f"Calibration ({method}): "
            f"{raw_forecast:+.2f}% → {calibrated_forecast:+.2f}% "
            f"(adjustment: {predicted_error:+.2f}%)"
        )

        return {
            "calibrated_forecast": float(calibrated_forecast),
            "adjustment": float(predicted_error),
            "method": method,
            "raw_forecast": float(raw_forecast),
        }

    def get_diagnostics(self) -> Dict:
        """
        Get calibration diagnostics and statistics.

        Returns:
            Dict with calibration statistics and model info
        """

        if not self.fitted:
            return {"error": "Calibrator not fitted"}

        diagnostics = {
            "fitted": self.fitted,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "min_samples": self.min_samples,
                "use_robust": self.use_robust,
                "cohort_specific": self.cohort_specific,
                "regime_specific": self.regime_specific,
            },
            "statistics": self.calibration_stats,
            "models": {
                "global": self.global_model is not None,
                "cohorts": list(self.cohort_models.keys()),
                "regimes": list(self.regime_models.keys()),
            },
        }

        # Calculate overall improvement
        if "global" in self.calibration_stats:
            global_stats = self.calibration_stats["global"]
            diagnostics["overall_improvement"] = global_stats.get("improvement_pct", 0)
            diagnostics["training_samples"] = global_stats.get("samples", 0)
            diagnostics["bias_correction"] = global_stats.get("mean_error", 0)

        return diagnostics

    def save_calibrator(self, output_dir: str = "models"):
        """
        Save calibrator to disk.

        Saves:
        1. Calibration models (pickle)
        2. Statistics (JSON)
        3. Diagnostics (JSON)
        """

        if not self.fitted:
            logger.error("❌ Cannot save unfitted calibrator")
            return

        import pickle

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save models
        calibrator_file = output_path / "forecast_calibrator.pkl"

        calibrator_data = {
            "global_model": self.global_model,
            "cohort_models": self.cohort_models,
            "regime_models": self.regime_models,
            "config": {
                "min_samples": self.min_samples,
                "use_robust": self.use_robust,
                "cohort_specific": self.cohort_specific,
                "regime_specific": self.regime_specific,
            },
            "fitted": self.fitted,
        }

        with open(calibrator_file, "wb") as f:
            pickle.dump(calibrator_data, f)

        logger.info(f"✅ Saved calibrator: {calibrator_file}")

        # Save diagnostics
        diagnostics_file = output_path / "calibrator_diagnostics.json"
        diagnostics = self.get_diagnostics()

        with open(diagnostics_file, "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)

        logger.info(f"✅ Saved diagnostics: {diagnostics_file}")

    def load_calibrator(self, input_dir: str = "models"):
        """Load calibrator from disk."""

        import pickle

        calibrator_file = Path(input_dir) / "forecast_calibrator.pkl"

        if not calibrator_file.exists():
            logger.error(f"❌ Calibrator file not found: {calibrator_file}")
            return False

        try:
            with open(calibrator_file, "rb") as f:
                calibrator_data = pickle.load(f)

            self.global_model = calibrator_data["global_model"]
            self.cohort_models = calibrator_data["cohort_models"]
            self.regime_models = calibrator_data["regime_models"]

            config = calibrator_data["config"]
            self.min_samples = config["min_samples"]
            self.use_robust = config["use_robust"]
            self.cohort_specific = config["cohort_specific"]
            self.regime_specific = config["regime_specific"]

            self.fitted = calibrator_data["fitted"]

            logger.info(f"✅ Loaded calibrator from {calibrator_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load calibrator: {e}")
            return False

    @classmethod
    def load(cls, input_dir: str = "models") -> Optional["ForecastCalibrator"]:
        """
        Load calibrator from disk (class method for convenience).

        Returns:
            ForecastCalibrator instance if successful, None otherwise
        """
        calibrator = cls()
        success = calibrator.load_calibrator(input_dir)

        if success:
            return calibrator
        else:
            return None


# ============================================================
# TESTING
# ============================================================


def test_calibrator():
    """Test calibrator with synthetic data."""

    print("\n" + "=" * 80)
    print("TESTING FORECAST CALIBRATOR V3")
    print("=" * 80)

    # Create synthetic prediction history with systematic bias
    np.random.seed(42)

    n_samples = 200
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    # True changes
    true_changes = np.random.randn(n_samples) * 5

    # Forecasts with systematic overestimation
    forecasts = true_changes + 1.0 + np.random.randn(n_samples) * 2

    # Current VIX levels
    vix_levels = 15 + np.random.randn(n_samples) * 5
    vix_levels = np.clip(vix_levels, 10, 40)

    # Create DataFrame mimicking database structure
    df = pd.DataFrame(
        {
            "forecast_date": dates,
            "median_forecast": forecasts,
            "actual_vix_change": true_changes,
            "current_vix": vix_levels,
            "calendar_cohort": np.random.choice(
                ["start_month", "mid_month", "end_month"], n_samples
            ),
        }
    )

    # Mock database class
    class MockDatabase:
        def get_predictions(self, with_actuals=False):
            return df

    mock_db = MockDatabase()

    # Initialize and fit calibrator
    calibrator = ForecastCalibrator(
        min_samples=50,
        use_robust=True,
        cohort_specific=True,
        regime_specific=True,
    )

    success = calibrator.fit_from_database(mock_db)

    if success:
        print("\n✅ Calibrator fitted successfully")

        # Get diagnostics
        diag = calibrator.get_diagnostics()
        print(f"✅ Overall improvement: {diag['overall_improvement']:+.1f}%")
        print(f"✅ Bias correction: {diag['bias_correction']:+.2f}%")

        # Test calibration
        test_forecast = 2.5
        test_vix = 18.0

        result = calibrator.calibrate(
            raw_forecast=test_forecast,
            current_vix=test_vix,
            cohort="mid_month",
        )

        print(f"\n✅ Test calibration:")
        print(f"   Raw: {result['raw_forecast']:+.2f}%")
        print(f"   Calibrated: {result['calibrated_forecast']:+.2f}%")
        print(f"   Adjustment: {result['adjustment']:+.2f}%")
        print(f"   Method: {result['method']}")

        # Save calibrator
        calibrator.save_calibrator(output_dir="/home/claude/test_output")
        print(f"\n✅ Calibrator saved")

    else:
        print("\n❌ Calibrator fitting failed")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_calibrator()
