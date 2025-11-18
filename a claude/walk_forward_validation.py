#!/usr/bin/env python3
import json
import logging
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class SimplifiedWalkForwardValidator:
    def __init__(self, db_path="data_cache/predictions.db", horizon=5):
        self.db_path = Path(db_path)
        self.horizon = horizon
        self.results = None
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    def load_predictions_with_actuals(self):
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT
            forecast_date,
            observation_date,
            calendar_cohort as cohort,
            current_vix,
            prob_up,
            prob_down,
            magnitude_forecast,
            expected_vix,
            actual_vix_change,
            actual_direction,
            direction_correct,
            magnitude_error,
            feature_quality,
            horizon
        FROM forecasts
        WHERE actual_vix_change IS NOT NULL
        ORDER BY forecast_date
        """
        
        df = pd.read_sql_query(query, conn, parse_dates=["forecast_date", "observation_date"])
        conn.close()
        
        logger.info(f"Loaded {len(df)} predictions with actuals")
        logger.info(f"Date range: {df['forecast_date'].min().date()} to {df['forecast_date'].max().date()}")
        
        return df
    
    def compute_metrics(self, df):
        metrics = {
            "overall": self._compute_overall_metrics(df),
            "direction": self._compute_direction_metrics(df),
            "magnitude": self._compute_magnitude_metrics(df),
            "by_cohort": self._compute_by_cohort(df),
            "time_series": self._compute_time_series_metrics(df)
        }
        
        return metrics
    
    def _compute_overall_metrics(self, df):
        return {
            "n_forecasts": int(len(df)),
            "date_range": {
                "start": df["forecast_date"].min().isoformat(),
                "end": df["forecast_date"].max().isoformat()
            }
        }
    
    def _compute_direction_metrics(self, df):
        if "direction_correct" not in df.columns or df["direction_correct"].isna().all():
            df["predicted_direction"] = (df["prob_up"] > 0.5).astype(int)
            df["direction_correct"] = (df["predicted_direction"] == df["actual_direction"]).astype(int)
        
        accuracy = df["direction_correct"].mean()
        
        df_up = df[df["actual_direction"] == 1]
        df_down = df[df["actual_direction"] == 0]
        
        precision = (df_up["direction_correct"].sum() / len(df_up)) if len(df_up) > 0 else 0
        recall = (df_up["direction_correct"].sum() / len(df_up)) if len(df_up) > 0 else 0
        
        prob_bins = [0, 0.4, 0.6, 1.0]
        df["prob_bin"] = pd.cut(df["prob_up"], bins=prob_bins, labels=["Low", "Medium", "High"])
        
        calibration = {}
        for bin_label in ["Low", "Medium", "High"]:
            bin_df = df[df["prob_bin"] == bin_label]
            if len(bin_df) > 0:
                actual_freq = bin_df["actual_direction"].mean()
                mean_pred_prob = bin_df["prob_up"].mean()
                calibration[bin_label] = {
                    "predicted_prob": float(mean_pred_prob),
                    "actual_frequency": float(actual_freq),
                    "n_samples": int(len(bin_df))
                }
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "calibration_by_bin": calibration
        }
    
    def _compute_magnitude_metrics(self, df):
        if "magnitude_error" not in df.columns or df["magnitude_error"].isna().all():
            df["magnitude_error"] = np.abs(df["actual_vix_change"] - df["magnitude_forecast"])
        
        mae = df["magnitude_error"].mean()
        rmse = np.sqrt((df["magnitude_error"] ** 2).mean())
        bias = (df["magnitude_forecast"] - df["actual_vix_change"]).mean()
        median_error = df["magnitude_error"].median()
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "bias": float(bias),
            "median_error": float(median_error)
        }
    
    def _compute_by_cohort(self, df):
        metrics_by_cohort = {}
        
        for cohort in df["cohort"].unique():
            cohort_df = df[df["cohort"] == cohort]
            
            if len(cohort_df) < 5:
                continue
            
            if "direction_correct" not in cohort_df.columns or cohort_df["direction_correct"].isna().all():
                cohort_df["predicted_direction"] = (cohort_df["prob_up"] > 0.5).astype(int)
                cohort_df["direction_correct"] = (cohort_df["predicted_direction"] == cohort_df["actual_direction"]).astype(int)
            
            if "magnitude_error" not in cohort_df.columns or cohort_df["magnitude_error"].isna().all():
                cohort_df["magnitude_error"] = np.abs(cohort_df["actual_vix_change"] - cohort_df["magnitude_forecast"])
            
            metrics_by_cohort[cohort] = {
                "n": int(len(cohort_df)),
                "direction_accuracy": float(cohort_df["direction_correct"].mean()),
                "magnitude_mae": float(cohort_df["magnitude_error"].mean()),
                "magnitude_bias": float((cohort_df["magnitude_forecast"] - cohort_df["actual_vix_change"]).mean())
            }
        
        return metrics_by_cohort
    
    def _compute_time_series_metrics(self, df):
        df = df.sort_values("forecast_date")
        
        if "magnitude_error" not in df.columns or df["magnitude_error"].isna().all():
            df["magnitude_error"] = np.abs(df["actual_vix_change"] - df["magnitude_forecast"])
        
        window = min(20, len(df) // 4)
        rolling_mae = df["magnitude_error"].rolling(window, min_periods=5).mean()
        
        return {
            "rolling_mae_mean": float(rolling_mae.mean()),
            "rolling_mae_std": float(rolling_mae.std()),
            "trend": "improving" if rolling_mae.iloc[-1] < rolling_mae.iloc[0] else "worsening"
        }
    
    def generate_diagnostic_report(self, output_dir="diagnostics"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("GENERATING WALK-FORWARD VALIDATION REPORT")
        logger.info("=" * 80)
        
        df = self.load_predictions_with_actuals()
        
        if len(df) < 10:
            logger.error("❌ Insufficient data for validation (need at least 10 predictions)")
            return
        
        metrics = self.compute_metrics(df)
        
        with open(output_dir / "walk_forward_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ Saved metrics to: {output_dir / 'walk_forward_metrics.json'}")
        
        self._plot_direction_calibration(df, output_dir)
        self._plot_magnitude_accuracy(df, output_dir)
        self._plot_performance_by_cohort(df, output_dir)
        self._plot_time_series(df, output_dir)
        
        self._print_summary(metrics)
        
        logger.info("=" * 80)
        logger.info("✅ VALIDATION REPORT COMPLETE")
        logger.info("=" * 80)
        
        self.results = metrics
        return metrics
    
    def _print_summary(self, metrics):
        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION SUMMARY (V4.0)")
        print("=" * 80)
        
        m = metrics["overall"]
        print(f"\n📊 Overall Performance ({m['n_forecasts']} forecasts)")
        
        d = metrics["direction"]
        print(f"\n🎯 Direction Performance:")
        print(f"   Accuracy:  {d['accuracy']:.1%}")
        print(f"   Precision: {d['precision']:.1%}")
        print(f"   Recall:    {d['recall']:.1%}")
        
        print(f"\n📏 Direction Calibration by Confidence:")
        for bin_label, cal in d["calibration_by_bin"].items():
            print(f"   {bin_label:8s}: Pred={cal['predicted_prob']:.1%}, Actual={cal['actual_frequency']:.1%}, N={cal['n_samples']}")
        
        mag = metrics["magnitude"]
        print(f"\n📈 Magnitude Performance:")
        print(f"   MAE:    {mag['mae']:.2f}%")
        print(f"   RMSE:   {mag['rmse']:.2f}%")
        print(f"   Bias:   {mag['bias']:+.2f}%")
        print(f"   Median: {mag['median_error']:.2f}%")
        
        if metrics["by_cohort"]:
            print(f"\n📅 Performance by Cohort:")
            for cohort, stats in metrics["by_cohort"].items():
                print(f"   {cohort:20s}: Acc={stats['direction_accuracy']:.1%}, MAE={stats['magnitude_mae']:.2f}%, N={stats['n']}")
    
    def _plot_direction_calibration(self, df, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        prob_true, prob_pred = calibration_curve(
            df["actual_direction"], df["prob_up"], n_bins=10
        )
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "o-", label="Model", linewidth=2)
        ax.set_xlabel("Predicted Probability (UP)")
        ax.set_ylabel("Actual Frequency (UP)")
        ax.set_title("Direction Probability Calibration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        bins = [0, 0.4, 0.6, 1.0]
        labels = ["<40%", "40-60%", ">60%"]
        df["prob_bin"] = pd.cut(df["prob_up"], bins=bins, labels=labels)
        
        accs = []
        for label in labels:
            mask = df["prob_bin"] == label
            if mask.sum() > 0:
                acc = df.loc[mask, "actual_direction"].mean()
                accs.append(acc)
            else:
                accs.append(0)
        
        ax.bar(range(len(labels)), accs, alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Actual UP Frequency")
        ax.set_title("Direction Accuracy by Confidence")
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(output_dir / "direction_calibration.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_magnitude_accuracy(self, df, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        ax.scatter(df["magnitude_forecast"], df["actual_vix_change"], alpha=0.5, s=30)
        lims = [
            min(df["magnitude_forecast"].min(), df["actual_vix_change"].min()),
            max(df["magnitude_forecast"].max(), df["actual_vix_change"].max())
        ]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlabel("Predicted VIX Change (%)")
        ax.set_ylabel("Actual VIX Change (%)")
        ax.set_title("Magnitude Forecast Accuracy")
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        errors = df["magnitude_forecast"] - df["actual_vix_change"]
        ax.hist(errors, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
        ax.axvline(errors.mean(), color="blue", linestyle="--", linewidth=2,
                  label=f"Mean: {errors.mean():.2f}%")
        ax.set_xlabel("Prediction Error (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Magnitude Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "magnitude_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_performance_by_cohort(self, df, output_dir):
        cohorts = df["cohort"].unique()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        accs = []
        labels = []
        for cohort in cohorts:
            cohort_df = df[df["cohort"] == cohort]
            if len(cohort_df) >= 5:
                if "direction_correct" not in cohort_df.columns or cohort_df["direction_correct"].isna().all():
                    cohort_df["predicted_direction"] = (cohort_df["prob_up"] > 0.5).astype(int)
                    cohort_df["direction_correct"] = (cohort_df["predicted_direction"] == cohort_df["actual_direction"]).astype(int)
                acc = cohort_df["direction_correct"].mean()
                accs.append(acc)
                labels.append(cohort)
        
        ax.bar(range(len(labels)), accs, alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Direction Accuracy")
        ax.set_title("Direction Accuracy by Cohort")
        ax.grid(True, alpha=0.3, axis="y")
        
        ax = axes[1]
        maes = []
        labels = []
        for cohort in cohorts:
            cohort_df = df[df["cohort"] == cohort]
            if len(cohort_df) >= 5:
                if "magnitude_error" not in cohort_df.columns or cohort_df["magnitude_error"].isna().all():
                    cohort_df["magnitude_error"] = np.abs(cohort_df["actual_vix_change"] - cohort_df["magnitude_forecast"])
                mae = cohort_df["magnitude_error"].mean()
                maes.append(mae)
                labels.append(cohort)
        
        ax.bar(range(len(labels)), maes, alpha=0.7, color="coral")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Magnitude MAE (%)")
        ax.set_title("Magnitude MAE by Cohort")
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_by_cohort.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_time_series(self, df, output_dir):
        df = df.sort_values("forecast_date")
        
        if "magnitude_error" not in df.columns or df["magnitude_error"].isna().all():
            df["magnitude_error"] = np.abs(df["actual_vix_change"] - df["magnitude_forecast"])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        ax = axes[0]
        if len(df) >= 20:
            rolling_mae = df["magnitude_error"].rolling(20, min_periods=5).mean()
            ax.plot(df["forecast_date"], rolling_mae, linewidth=2, color="darkblue")
            ax.fill_between(df["forecast_date"], 0, rolling_mae, alpha=0.3)
        else:
            ax.plot(df["forecast_date"], df["magnitude_error"], "o-", linewidth=2)
        
        ax.set_ylabel("Rolling MAE (20-forecast window)")
        ax.set_title("Magnitude Forecast Quality Over Time")
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        if "direction_correct" not in df.columns or df["direction_correct"].isna().all():
            df["predicted_direction"] = (df["prob_up"] > 0.5).astype(int)
            df["direction_correct"] = (df["predicted_direction"] == df["actual_direction"]).astype(int)
        
        if len(df) >= 20:
            rolling_acc = df["direction_correct"].rolling(20, min_periods=5).mean()
            ax.plot(df["forecast_date"], rolling_acc, linewidth=2, color="darkgreen")
            ax.fill_between(df["forecast_date"], 0, rolling_acc, alpha=0.3)
        else:
            ax.plot(df["forecast_date"], df["direction_correct"], "o-", linewidth=2)
        
        ax.set_ylabel("Rolling Accuracy (20-forecast window)")
        ax.set_xlabel("Forecast Date")
        ax.set_title("Direction Forecast Accuracy Over Time")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "time_series_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified walk-forward validation")
    parser.add_argument("--db", default="data_cache/predictions.db", help="Database path")
    parser.add_argument("--output", default="diagnostics", help="Output directory")
    args = parser.parse_args()
    
    validator = SimplifiedWalkForwardValidator(db_path=args.db)
    validator.generate_diagnostic_report(output_dir=args.output)
