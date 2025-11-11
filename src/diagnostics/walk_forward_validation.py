#!/usr/bin/env python3
"""
Enhanced Walk-Forward Validation with Comprehensive Diagnostics

Improvements:
- Quantile crossing detection
- Regime-based breakdown
- Cohort-specific analysis
- Time-series of forecast quality
- Confidence calibration curves
- Detailed plots
"""

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
    """Production-ready walk-forward validation with detailed diagnostics."""

    def __init__(self, db_path: str = "data_cache/predictions.db", horizon: int = 5):
        self.db_path = Path(db_path)
        self.horizon = horizon
        self.results = None

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def load_predictions_with_actuals(self) -> pd.DataFrame:
        """Load all predictions that have actual outcomes."""
        conn = sqlite3.connect(self.db_path)

        # First, check what columns exist
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(forecasts)")
        columns = [row[1] for row in cursor.fetchall()]
        logger.debug(f"Available columns: {columns}")

        # Build query based on available columns
        query = """
        SELECT
            forecast_date,
            calendar_cohort as cohort,
            current_vix as vix_start,
            point_estimate,
            q10, q25, q50, q75, q90,
            prob_low, prob_normal, prob_elevated, prob_crisis,
            confidence_score as confidence,
            feature_quality,
            actual_vix_change,
            actual_regime,
            point_error,
            quantile_coverage,
            horizon
        FROM forecasts
        WHERE actual_vix_change IS NOT NULL
        ORDER BY forecast_date
        """

        df = pd.read_sql_query(query, conn, parse_dates=["forecast_date"])
        conn.close()

        # Compute target_date from forecast_date + horizon
        df["target_date"] = df["forecast_date"] + pd.to_timedelta(
            df["horizon"], unit="D"
        )

        logger.info(f"Loaded {len(df)} predictions with actuals")
        logger.info(
            f"Date range: {df['forecast_date'].min().date()} to {df['forecast_date'].max().date()}"
        )

        return df

    def compute_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive evaluation metrics."""

        # Parse quantile coverage JSON
        df["coverage"] = df["quantile_coverage"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

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
        """Overall forecast accuracy."""
        return {
            "n_forecasts": int(len(df)),
            "mae": float(df["point_error"].abs().mean()),
            "rmse": float(np.sqrt((df["point_error"] ** 2).mean())),
            "mape": float((df["point_error"].abs() / df["vix_start"]).mean() * 100),
            "median_abs_error": float(df["point_error"].abs().median()),
            "bias": float(df["point_error"].mean()),
            "forecast_width_mean": float((df["q90"] - df["q10"]).mean()),
            "forecast_iqr_mean": float((df["q75"] - df["q25"]).mean()),
        }

    def _compute_quantile_calibration(self, df: pd.DataFrame) -> Dict:
        """Check if quantiles are well-calibrated."""
        calibration = {}

        for q in [10, 25, 50, 75, 90]:
            # Check if actual <= predicted quantile
            hits = df["coverage"].apply(lambda x: x.get(f"q{q}", 0)).mean()
            expected = q / 100.0

            calibration[f"q{q}"] = {
                "observed": float(hits),
                "expected": float(expected),
                "diff": float(hits - expected),
                "calibrated": bool(abs(hits - expected) < 0.10),
            }

        # Interval coverage
        calibration["interval_80"] = {
            "coverage": float(
                (
                    (df["actual_vix_change"] >= df["q10"])
                    & (df["actual_vix_change"] <= df["q90"])
                ).mean()
            )
        }

        calibration["interval_50"] = {
            "coverage": float(
                (
                    (df["actual_vix_change"] >= df["q25"])
                    & (df["actual_vix_change"] <= df["q75"])
                ).mean()
            )
        }

        return calibration

    def _compute_by_cohort(self, df: pd.DataFrame) -> Dict:
        """Breakdown by calendar cohort."""
        by_cohort = {}

        for cohort in df["cohort"].unique():
            cohort_df = df[df["cohort"] == cohort]

            if len(cohort_df) < 5:
                continue

            by_cohort[str(cohort)] = {
                "n": int(len(cohort_df)),
                "mae": float(cohort_df["point_error"].abs().mean()),
                "rmse": float(np.sqrt((cohort_df["point_error"] ** 2).mean())),
                "forecast_width": float((cohort_df["q90"] - cohort_df["q10"]).mean()),
                "q50_coverage": float(
                    cohort_df["coverage"].apply(lambda x: x.get("q50", 0)).mean()
                ),
            }

        return by_cohort

    def _compute_by_regime(self, df: pd.DataFrame) -> Dict:
        """Breakdown by actual VIX regime."""
        by_regime = {}

        for regime in df["actual_regime"].unique():
            regime_df = df[df["actual_regime"] == regime]

            by_regime[str(regime)] = {
                "n": int(len(regime_df)),
                "mae": float(regime_df["point_error"].abs().mean()),
                "mean_forecast_width": float(
                    (regime_df["q90"] - regime_df["q10"]).mean()
                ),
            }

        return by_regime

    def _analyze_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze if confidence scores are meaningful."""

        # Bin by confidence
        df["conf_bin"] = pd.cut(
            df["confidence"],
            bins=[0, 0.7, 0.85, 0.95, 1.0],
            labels=["Low", "Medium", "High", "Very High"],
        )

        # Aggregate by confidence bin
        conf_stats = (
            df.groupby("conf_bin", observed=True)
            .agg({"point_error": lambda x: x.abs().mean(), "forecast_date": "count"})
            .rename(columns={"forecast_date": "count"})
        )

        # Convert to JSON-serializable dictionary
        by_bin = {}
        for bin_name, row in conf_stats.iterrows():
            by_bin[str(bin_name)] = {
                "mean_abs_error": float(row["point_error"]),
                "count": int(row["count"]),
            }

        # Correlation
        corr = df[["confidence", "point_error"]].corr().iloc[0, 1]

        return {
            "by_bin": by_bin,
            "correlation": float(corr),
            "is_useful": bool(corr < -0.1),
        }

    def _compute_time_series_metrics(self, df: pd.DataFrame) -> Dict:
        """Track forecast quality over time."""
        df = df.sort_values("forecast_date").copy()

        window = 20
        result = {"has_trend": bool(len(df) >= window), "recent_mae": None}

        if len(df) >= 10:
            result["recent_mae"] = float(df["point_error"].abs().tail(10).mean())

        return result

    def generate_diagnostic_report(self, output_dir: str = "diagnostics"):
        """Generate comprehensive diagnostic report and plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        logger.info("=" * 80)
        logger.info("GENERATING DIAGNOSTIC REPORT")
        logger.info("=" * 80)

        # Load data
        df = self.load_predictions_with_actuals()

        if len(df) == 0:
            logger.error("No predictions with actuals found!")
            return

        # Compute metrics
        metrics = self.compute_metrics(df)

        # Print summary
        self._print_summary(metrics)

        # Generate plots
        self._plot_calibration(df, output_dir)
        self._plot_forecast_vs_actual(df, output_dir)
        self._plot_confidence_analysis(df, output_dir)
        self._plot_cohort_performance(df, output_dir)
        self._plot_time_series(df, output_dir)

        # Save metrics to JSON
        import json

        metrics_path = output_dir / "walk_forward_metrics.json"

        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=convert)

        logger.info(f"\n✅ Report saved to {output_dir}/")
        logger.info(f"   - walk_forward_metrics.json")
        logger.info(f"   - *.png (diagnostic plots)")

    def _print_summary(self, metrics: Dict):
        """Print human-readable summary."""
        m = metrics["overall"]

        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 80)

        print(f"\n📊 Overall Performance ({m['n_forecasts']} forecasts):")
        print(f"   MAE:  {m['mae']:.2f}%")
        print(f"   RMSE: {m['rmse']:.2f}%")
        print(f"   Median Abs Error: {m['median_abs_error']:.2f}%")
        print(
            f"   Bias: {m['bias']:+.2f}% {'(over-predicting)' if m['bias'] > 0 else '(under-predicting)'}"
        )

        print(f"\n📐 Quantile Calibration:")
        for q in [10, 25, 50, 75, 90]:
            cal = metrics["quantile_calibration"][f"q{q}"]
            status = "✅" if cal["calibrated"] else "❌"
            print(
                f"   {status} q{q}: {cal['observed']:.1%} (expected {cal['expected']:.1%}, diff: {cal['diff']:+.1%})"
            )

        int80 = metrics["quantile_calibration"]["interval_80"]["coverage"]
        int50 = metrics["quantile_calibration"]["interval_50"]["coverage"]
        print(f"\n📦 Interval Coverage:")
        print(f"   80% interval (q10-q90): {int80:.1%}")
        print(f"   50% interval (q25-q75): {int50:.1%}")

        if metrics["by_cohort"]:
            print(f"\n📅 Performance by Cohort:")
            for cohort, stats in metrics["by_cohort"].items():
                print(f"   {cohort:30s}: MAE={stats['mae']:.2f}%, n={stats['n']}")

        conf = metrics["confidence_analysis"]
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Correlation (conf vs error): {conf['correlation']:.3f}")
        print(f"   Confidence useful: {'✅ Yes' if conf['is_useful'] else '❌ No'}")

    def _plot_calibration(self, df: pd.DataFrame, output_dir: Path):
        """Calibration plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        quantiles = [10, 25, 50, 75, 90]
        expected = [q / 100 for q in quantiles]
        observed = [
            df["coverage"].apply(lambda x: x.get(f"q{q}", 0)).mean() for q in quantiles
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
            "Quantile Calibration\n(Points should lie on diagonal)",
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
        """Forecast vs actual scatter."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Point forecast
        ax = axes[0]
        ax.scatter(
            df["point_estimate"],
            df["actual_vix_change"],
            alpha=0.6,
            s=50,
            c=df["confidence"],
            cmap="viridis",
        )

        # Perfect prediction line
        lims = [
            min(df["point_estimate"].min(), df["actual_vix_change"].min()),
            max(df["point_estimate"].max(), df["actual_vix_change"].max()),
        ]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=2)

        ax.set_xlabel("Predicted VIX Change (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual VIX Change (%)", fontsize=11, fontweight="bold")
        ax.set_title("Point Forecast Accuracy", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Error distribution
        ax = axes[1]
        errors = df["actual_vix_change"] - df["point_estimate"]
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
        ax.set_title("Error Distribution", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "forecast_vs_actual.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_confidence_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Confidence vs error relationship."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter
        ax = axes[0]
        ax.scatter(df["confidence"], df["point_error"].abs(), alpha=0.5, s=50)

        # Add trend line
        z = np.polyfit(df["confidence"], df["point_error"].abs(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["confidence"].min(), df["confidence"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend")

        ax.set_xlabel("Confidence Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Absolute Forecast Error (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            "Confidence vs Error\n(Should be negative correlation)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Binned analysis
        ax = axes[1]
        df["conf_bin"] = pd.cut(df["confidence"], bins=5)
        binned = df.groupby("conf_bin", observed=True)["point_error"].agg(
            ["mean", "std", "count"]
        )
        binned["mean"].abs().plot(
            kind="bar",
            yerr=binned["std"],
            ax=ax,
            capsize=5,
            color="steelblue",
            alpha=0.7,
        )

        ax.set_xlabel("Confidence Bin", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Abs Error (%)", fontsize=11, fontweight="bold")
        ax.set_title("Error by Confidence Level", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            output_dir / "confidence_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_cohort_performance(self, df: pd.DataFrame, output_dir: Path):
        """Performance by cohort."""
        cohort_stats = (
            df.groupby("cohort")
            .agg({"point_error": lambda x: x.abs().mean(), "forecast_date": "count"})
            .rename(columns={"forecast_date": "count"})
        )

        cohort_stats = cohort_stats[cohort_stats["count"] >= 5].sort_values(
            "point_error"
        )

        if len(cohort_stats) == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        cohort_stats["point_error"].plot(kind="barh", ax=ax, color="teal", alpha=0.7)

        # Add count labels
        for i, (idx, row) in enumerate(cohort_stats.iterrows()):
            ax.text(
                row["point_error"],
                i,
                f"  n={int(row['count'])}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Mean Absolute Error (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Calendar Cohort", fontsize=11, fontweight="bold")
        ax.set_title("Forecast Accuracy by Cohort", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(output_dir / "cohort_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_time_series(self, df: pd.DataFrame, output_dir: Path):
        """Forecast quality over time."""
        df = df.sort_values("forecast_date")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Rolling MAE
        ax = axes[0]
        if len(df) >= 20:
            df["rolling_mae"] = (
                df["point_error"].abs().rolling(20, min_periods=5).mean()
            )
            ax.plot(
                df["forecast_date"], df["rolling_mae"], linewidth=2, color="darkblue"
            )
            ax.fill_between(df["forecast_date"], 0, df["rolling_mae"], alpha=0.3)
        else:
            ax.plot(df["forecast_date"], df["point_error"].abs(), "o-", linewidth=2)

        ax.set_ylabel(
            "Rolling MAE (20-forecast window)", fontsize=11, fontweight="bold"
        )
        ax.set_title("Forecast Quality Over Time", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Confidence scores
        ax = axes[1]
        ax.scatter(
            df["forecast_date"],
            df["confidence"],
            c=df["point_error"].abs(),
            cmap="RdYlGn_r",
            s=50,
            alpha=0.6,
        )
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

    parser = argparse.ArgumentParser(description="Enhanced walk-forward validation")
    parser.add_argument(
        "--db", default="data_cache/predictions.db", help="Database path"
    )
    parser.add_argument("--output", default="diagnostics", help="Output directory")

    args = parser.parse_args()

    validator = EnhancedWalkForwardValidator(db_path=args.db)
    validator.generate_diagnostic_report(output_dir=args.output)
