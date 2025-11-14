#!/usr/bin/env python3
"""
Walk-Forward Validation - V3 Adapted for Log-RV Quantile Regression

CRITICAL CHANGES FROM V2:
1. Primary metric changed from point_estimate to median_forecast
2. point_error replaced with median_error as primary error metric
3. All analyses now use median_forecast (q50) as the central tendency
4. Backward compatible: still computes point_estimate metrics if available
5. Enhanced quantile calibration diagnostics

Author: VIX Forecasting System
Last Updated: 2025-11-14
Version: 3.0 (Log-RV Quantile Regression)
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class EnhancedWalkForwardValidator:
    """
    Production-ready walk-forward validation with detailed diagnostics.

    V3 ENHANCEMENTS:
    - Median forecast as primary metric (more robust than mean)
    - Quantile calibration analysis
    - Forecast uncertainty quantification
    - Regime-specific performance
    """

    def __init__(
        self,
        db_path: str = "data_cache/predictions.db",
        horizon: int = 5,
    ):
        self.db_path = Path(db_path)
        self.horizon = horizon
        self.results = None

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def load_predictions_with_actuals(self) -> pd.DataFrame:
        """
        Load all predictions that have actual outcomes.

        V3 CHANGES:
        - Loads median_forecast as primary field
        - Loads median_error as primary error metric
        - Maintains backward compatibility with point_estimate
        """

        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            forecast_date,
            observation_date,
            calendar_cohort as cohort,
            current_vix as vix_start,
            median_forecast,
            point_estimate,
            q10, q25, q50, q75, q90,
            prob_up,
            prob_down,
            confidence_score as confidence,
            feature_quality,
            actual_vix_change,
            actual_regime,
            median_error,
            point_error,
            quantile_coverage,
            horizon
        FROM forecasts
        WHERE actual_vix_change IS NOT NULL
        ORDER BY forecast_date
        """

        df = pd.read_sql_query(
            query, conn, parse_dates=["forecast_date", "observation_date"]
        )
        conn.close()

        logger.info(f"Loaded {len(df)} predictions with actuals")
        logger.info(
            f"Date range: {df['forecast_date'].min().date()} to {df['forecast_date'].max().date()}"
        )

        # Check for missing quantile_coverage
        missing_coverage = df["quantile_coverage"].isna().sum()
        if missing_coverage > 0:
            logger.warning(
                f"⚠️  {missing_coverage}/{len(df)} predictions have missing quantile_coverage - "
                f"computing from quantiles..."
            )

        return df

    def _compute_coverage_from_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute quantile coverage from actual vs predicted quantiles."""

        def compute_row_coverage(row):
            """Compute coverage for a single row."""
            if pd.isna(row["actual_vix_change"]):
                return {}

            coverage = {}
            actual = row["actual_vix_change"]

            # Check if actual is below each quantile
            for q in [10, 25, 50, 75, 90]:
                col = f"q{q}"
                if pd.notna(row[col]):
                    coverage[f"q{q}"] = 1 if actual <= row[col] else 0
                else:
                    coverage[f"q{q}"] = 0

            return coverage

        # Apply to all rows
        df["coverage"] = df.apply(compute_row_coverage, axis=1)
        return df

    def compute_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive evaluation metrics.

        V3 CHANGES:
        - Primary metrics based on median_forecast
        - Quantile calibration analysis
        - Enhanced uncertainty quantification
        """

        # Parse quantile coverage JSON - FIXED: Handle None/empty values
        def parse_coverage(x):
            if pd.isna(x) or x is None or x == "":
                return None  # Return None for now, we'll compute it
            if isinstance(x, str):
                try:
                    return json.loads(x.replace("'", '"'))  # Handle single quotes
                except:
                    try:
                        return eval(x)
                    except:
                        return None
            if isinstance(x, dict):
                return x
            return None

        df["coverage"] = df["quantile_coverage"].apply(parse_coverage)

        # Compute coverage for rows where it's missing
        missing_mask = df["coverage"].isna()
        if missing_mask.any():
            logger.info(
                f"Computing coverage for {missing_mask.sum()} rows from quantiles..."
            )
            df = self._compute_coverage_from_quantiles(df)

        metrics = {
            "overall": self._compute_overall_metrics(df),
            "quantile_calibration": self._compute_quantile_calibration(df),
            "by_cohort": self._compute_by_cohort(df),
            "by_regime": self._compute_by_regime(df),
            "confidence_analysis": self._analyze_confidence(df),
            "time_series": self._compute_time_series_metrics(df),
        }

        return metrics

    def _compute_overall_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Overall forecast accuracy.

        V3: Uses median_error as primary metric, with backward compat for point_error
        """

        # Use median_error as primary, fallback to point_error if needed
        if "median_error" in df.columns and df["median_error"].notna().any():
            primary_error = df["median_error"]
            logger.info("Using median_error as primary metric")
        elif "point_error" in df.columns:
            primary_error = df["point_error"]
            logger.warning("median_error not available, using point_error")
        else:
            # Compute from median_forecast and actual
            primary_error = df["actual_vix_change"] - df["median_forecast"]
            logger.info("Computing error from median_forecast and actual")

        return {
            "n_forecasts": int(len(df)),
            "mae": float(primary_error.abs().mean()),
            "rmse": float(np.sqrt((primary_error**2).mean())),
            "mape": float((primary_error.abs() / df["vix_start"]).mean() * 100),
            "median_abs_error": float(primary_error.abs().median()),
            "bias": float(primary_error.mean()),
            "forecast_width_mean": float((df["q90"] - df["q10"]).mean()),
            "forecast_iqr_mean": float((df["q75"] - df["q25"]).mean()),
        }

    def _compute_quantile_calibration(self, df: pd.DataFrame) -> Dict:
        """
        Check if quantiles are well-calibrated.

        V3: Enhanced analysis for log-RV quantile regression
        """

        calibration = {}

        # Check each quantile
        for q in [10, 25, 50, 75, 90]:
            expected = q / 100

            # Observed coverage
            observed = (
                df["coverage"]
                .apply(
                    lambda x: x.get(f"q{q}", 0) if (x and isinstance(x, dict)) else 0
                )
                .mean()
            )

            # Statistical test (binomial proportion test)
            n = len(df)
            stderr = np.sqrt(expected * (1 - expected) / n)
            z_score = (observed - expected) / stderr if stderr > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            # Is it calibrated? (95% confidence)
            calibrated = p_value > 0.05

            calibration[f"q{q}"] = {
                "expected": float(expected),
                "observed": float(observed),
                "diff": float(observed - expected),
                "stderr": float(stderr),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "calibrated": bool(calibrated),
            }

        # Interval coverage
        interval_80 = (
            df["coverage"]
            .apply(
                lambda x: (x.get("q10", 0) == 0 and x.get("q90", 0) == 1)
                if (x and isinstance(x, dict))
                else False
            )
            .mean()
        )

        interval_50 = (
            df["coverage"]
            .apply(
                lambda x: (x.get("q25", 0) == 0 and x.get("q75", 0) == 1)
                if (x and isinstance(x, dict))
                else False
            )
            .mean()
        )

        calibration["interval_80"] = {
            "coverage": float(interval_80),
            "expected": 0.80,
        }

        calibration["interval_50"] = {
            "coverage": float(interval_50),
            "expected": 0.50,
        }

        return calibration

    def _compute_by_cohort(self, df: pd.DataFrame) -> Dict:
        """Performance breakdown by cohort."""

        metrics_by_cohort = {}

        for cohort in df["cohort"].unique():
            cohort_df = df[df["cohort"] == cohort]

            if len(cohort_df) < 5:
                continue

            # Use median_error if available, else compute
            if (
                "median_error" in cohort_df.columns
                and cohort_df["median_error"].notna().any()
            ):
                error = cohort_df["median_error"]
            else:
                error = cohort_df["actual_vix_change"] - cohort_df["median_forecast"]

            metrics_by_cohort[cohort] = {
                "n": int(len(cohort_df)),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
            }

        return metrics_by_cohort

    def _compute_by_regime(self, df: pd.DataFrame) -> Dict:
        """Performance breakdown by VIX regime."""

        metrics_by_regime = {}

        if "actual_regime" not in df.columns:
            return metrics_by_regime

        for regime in df["actual_regime"].dropna().unique():
            regime_df = df[df["actual_regime"] == regime]

            if len(regime_df) < 5:
                continue

            # Use median_error if available
            if (
                "median_error" in regime_df.columns
                and regime_df["median_error"].notna().any()
            ):
                error = regime_df["median_error"]
            else:
                error = regime_df["actual_vix_change"] - regime_df["median_forecast"]

            metrics_by_regime[regime] = {
                "n": int(len(regime_df)),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
            }

        return metrics_by_regime

    def _analyze_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze relationship between confidence and forecast accuracy."""

        # Use median_error if available
        if "median_error" in df.columns and df["median_error"].notna().any():
            error = df["median_error"].abs()
        else:
            error = (df["actual_vix_change"] - df["median_forecast"]).abs()

        # Correlation between confidence and error (should be negative)
        correlation = df["confidence"].corr(-error)

        # Check if confidence is useful (statistically significant)
        _, p_value = stats.pearsonr(df["confidence"], -error)
        is_useful = p_value < 0.05

        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "is_useful": bool(is_useful),
            "interpretation": (
                "Higher confidence → Lower error ✅"
                if correlation > 0.1
                else "Confidence not predictive ⚠️"
            ),
        }

    def _compute_time_series_metrics(self, df: pd.DataFrame) -> Dict:
        """Time series analysis of forecast quality."""

        df = df.sort_values("forecast_date")

        # Use median_error
        if "median_error" in df.columns and df["median_error"].notna().any():
            error = df["median_error"].abs()
        else:
            error = (df["actual_vix_change"] - df["median_forecast"]).abs()

        # Rolling statistics
        window = min(20, len(df) // 4)
        rolling_mae = error.rolling(window, min_periods=5).mean()

        return {
            "rolling_mae_mean": float(rolling_mae.mean()),
            "rolling_mae_std": float(rolling_mae.std()),
            "trend": "improving"
            if rolling_mae.iloc[-1] < rolling_mae.iloc[0]
            else "worsening",
        }

    def generate_diagnostic_report(self, output_dir: str = "diagnostics"):
        """Generate comprehensive diagnostic report with plots."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("GENERATING WALK-FORWARD VALIDATION REPORT")
        logger.info("=" * 80)

        # Load data
        df = self.load_predictions_with_actuals()

        if len(df) < 10:
            logger.error(
                "❌ Insufficient data for validation (need at least 10 predictions)"
            )
            return

        # Compute metrics
        metrics = self.compute_metrics(df)

        # Save metrics
        with open(output_dir / "walk_forward_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✅ Saved metrics to: {output_dir / 'walk_forward_metrics.json'}")

        # Generate plots
        self._plot_calibration(df, output_dir)
        self._plot_forecast_vs_actual(df, output_dir)
        self._plot_confidence_analysis(df, output_dir)
        self._plot_time_series(df, output_dir)

        # Print summary
        self._print_summary(metrics)

        logger.info("=" * 80)
        logger.info("✅ VALIDATION REPORT COMPLETE")
        logger.info("=" * 80)

        self.results = metrics
        return metrics

    def _print_summary(self, metrics: Dict):
        """Print human-readable summary."""

        m = metrics["overall"]

        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION SUMMARY (V3)")
        print("=" * 80)

        print(f"\n📊 Overall Performance ({m['n_forecasts']} forecasts):")
        print(f"   MAE (median):  {m['mae']:.2f}%")
        print(f"   RMSE:          {m['rmse']:.2f}%")
        print(f"   Median Abs Error: {m['median_abs_error']:.2f}%")
        print(
            f"   Bias: {m['bias']:+.2f}% {'(over-predicting)' if m['bias'] > 0 else '(under-predicting)'}"
        )

        print(f"\n📏 Quantile Calibration:")
        for q in [10, 25, 50, 75, 90]:
            cal = metrics["quantile_calibration"][f"q{q}"]
            status = "✅" if cal["calibrated"] else "⚠️"
            print(
                f"   {status} q{q}: {cal['observed']:.1%} (expected {cal['expected']:.1%}, diff: {cal['diff']:+.1%})"
            )

        int80 = metrics["quantile_calibration"]["interval_80"]["coverage"]
        int50 = metrics["quantile_calibration"]["interval_50"]["coverage"]
        print(f"\n📦 Interval Coverage:")
        print(f"   80% interval (q10-q90): {int80:.1%} (expected 80%)")
        print(f"   50% interval (q25-q75): {int50:.1%} (expected 50%)")

        if metrics["by_cohort"]:
            print(f"\n📅 Performance by Cohort:")
            for cohort, stats in metrics["by_cohort"].items():
                print(f"   {cohort:30s}: MAE={stats['mae']:.2f}%, n={stats['n']}")

        conf = metrics["confidence_analysis"]
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Correlation (conf vs error): {conf['correlation']:.3f}")
        print(f"   Confidence useful: {'✅ Yes' if conf['is_useful'] else '⚠️ No'}")
        print(f"   {conf['interpretation']}")

    def _plot_calibration(self, df: pd.DataFrame, output_dir: Path):
        """Calibration plot."""

        fig, ax = plt.subplots(figsize=(10, 8))

        quantiles = [10, 25, 50, 75, 90]
        expected = [q / 100 for q in quantiles]
        observed = [
            df["coverage"]
            .apply(lambda x: x.get(f"q{q}", 0) if (x and isinstance(x, dict)) else 0)
            .mean()
            for q in quantiles
        ]

        # Perfect calibration line
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration", alpha=0.5
        )

        # Actual calibration
        ax.plot(
            expected,
            observed,
            "o-",
            linewidth=3,
            markersize=10,
            label="Observed",
            color="steelblue",
        )

        # Add labels
        for e, o, q in zip(expected, observed, quantiles):
            ax.annotate(
                f"q{q}", (e, o), xytext=(10, -5), textcoords="offset points", fontsize=9
            )

        # Confidence bands
        n = len(df)
        for e, o in zip(expected, observed):
            stderr = np.sqrt(e * (1 - e) / n)
            ax.fill_between(
                [e - 0.01, e + 0.01],
                [o - 1.96 * stderr] * 2,
                [o + 1.96 * stderr] * 2,
                alpha=0.2,
                color="steelblue",
            )

        ax.set_xlabel("Expected Quantile Coverage", fontsize=12, fontweight="bold")
        ax.set_ylabel("Observed Quantile Coverage", fontsize=12, fontweight="bold")
        ax.set_title(
            "Quantile Calibration (V3)\n(Points should lie on diagonal)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(output_dir / "calibration_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_forecast_vs_actual(self, df: pd.DataFrame, output_dir: Path):
        """Forecast vs actual scatter - V3 uses median_forecast."""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Median forecast (V3 primary)
        ax = axes[0]
        ax.scatter(
            df["median_forecast"],
            df["actual_vix_change"],
            alpha=0.6,
            s=50,
            c=df["confidence"],
            cmap="viridis",
        )

        # Perfect prediction line
        lims = [
            min(df["median_forecast"].min(), df["actual_vix_change"].min()),
            max(df["median_forecast"].max(), df["actual_vix_change"].max()),
        ]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=2)

        ax.set_xlabel(
            "Predicted VIX Change (%) - Median", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Actual VIX Change (%)", fontsize=11, fontweight="bold")
        ax.set_title("Median Forecast Accuracy (V3)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Error distribution
        ax = axes[1]
        if "median_error" in df.columns and df["median_error"].notna().any():
            errors = df["median_error"]
        else:
            errors = df["actual_vix_change"] - df["median_forecast"]

        ax.hist(errors, bins=30, alpha=0.7, edgecolor="black", color="coral")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
        ax.axvline(
            errors.mean(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean Error: {errors.mean():.2f}%",
        )

        ax.set_xlabel("Forecast Error (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax.set_title("Error Distribution (V3)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "forecast_vs_actual.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_confidence_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Confidence vs error relationship."""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Use median_error if available
        if "median_error" in df.columns and df["median_error"].notna().any():
            abs_error = df["median_error"].abs()
        else:
            abs_error = (df["actual_vix_change"] - df["median_forecast"]).abs()

        # Scatter plot
        ax = axes[0]
        ax.scatter(df["confidence"], abs_error, alpha=0.6, s=50, color="steelblue")

        # Trend line
        z = np.polyfit(df["confidence"], abs_error, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df["confidence"].min(), df["confidence"].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label="Trend")

        ax.set_xlabel("Confidence Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Absolute Error (%)", fontsize=11, fontweight="bold")
        ax.set_title("Confidence vs Error", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Binned analysis
        ax = axes[1]
        df["conf_bin"] = pd.cut(df["confidence"], bins=5)
        binned = df.groupby("conf_bin")[
            abs_error.name if hasattr(abs_error, "name") else "error"
        ].agg(["mean", "std", "count"])

        x_pos = range(len(binned))
        ax.bar(
            x_pos,
            binned["mean"],
            yerr=binned["std"],
            alpha=0.7,
            color="steelblue",
            capsize=5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{b.left:.2f}-{b.right:.2f}" for b in binned.index], rotation=45
        )

        ax.set_xlabel("Confidence Bins", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Absolute Error (%)", fontsize=11, fontweight="bold")
        ax.set_title("Error by Confidence Level", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            output_dir / "confidence_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_time_series(self, df: pd.DataFrame, output_dir: Path):
        """Time series of forecast quality."""

        df = df.sort_values("forecast_date")

        # Use median_error if available
        if "median_error" in df.columns and df["median_error"].notna().any():
            abs_error = df["median_error"].abs()
        else:
            abs_error = (df["actual_vix_change"] - df["median_forecast"]).abs()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Rolling MAE
        ax = axes[0]
        if len(df) >= 20:
            rolling_mae = abs_error.rolling(20, min_periods=5).mean()
            ax.plot(df["forecast_date"], rolling_mae, linewidth=2, color="darkblue")
            ax.fill_between(df["forecast_date"], 0, rolling_mae, alpha=0.3)
        else:
            ax.plot(df["forecast_date"], abs_error, "o-", linewidth=2)

        ax.set_ylabel(
            "Rolling MAE (20-forecast window)", fontsize=11, fontweight="bold"
        )
        ax.set_title("Forecast Quality Over Time (V3)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Confidence scores
        ax = axes[1]
        scatter = ax.scatter(
            df["forecast_date"],
            df["confidence"],
            c=abs_error,
            cmap="RdYlGn_r",
            s=50,
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label="Absolute Error (%)")

        ax.set_ylabel("Confidence Score", fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast Date", fontsize=11, fontweight="bold")
        ax.set_title(
            "Confidence Scores (colored by error)", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "time_series_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced walk-forward validation (V3)"
    )
    parser.add_argument(
        "--db", default="data_cache/predictions.db", help="Database path"
    )
    parser.add_argument("--output", default="diagnostics", help="Output directory")

    args = parser.parse_args()

    validator = EnhancedWalkForwardValidator(db_path=args.db)
    validator.generate_diagnostic_report(output_dir=args.output)
