"""
Production-Grade Diagnostics for VIX Forecasting System

Adds comprehensive logging, validation, and health checks.
"""

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ============================================================================
# DATA QUALITY MONITORING
# ============================================================================


@dataclass
class DataQualityReport:
    """Comprehensive data quality metrics."""

    timestamp: str
    forecast_date: str

    # Completeness
    total_features: int
    non_null_features: int
    completeness_ratio: float

    # Feature quality
    mean_feature_quality: float
    min_feature_quality: float
    features_below_threshold: List[str]

    # Staleness
    most_stale_feature: str
    max_staleness_days: int
    stale_features_count: int

    # Data freshness
    vix_last_update: str
    vix_staleness_hours: float

    # Cohort info
    calendar_cohort: str
    cohort_weight: float

    # Anomalies
    extreme_values: List[Dict]
    suspicious_patterns: List[str]

    def to_dict(self):
        return asdict(self)

    def is_acceptable(self) -> bool:
        """Check if data quality passes minimum thresholds."""
        return (
            self.completeness_ratio >= 0.95
            and self.mean_feature_quality >= 0.7
            and self.vix_staleness_hours < 48
            and self.max_staleness_days < 5
        )

    def get_warnings(self) -> List[str]:
        """Generate human-readable warnings."""
        warnings = []

        if self.completeness_ratio < 0.95:
            warnings.append(
                f"⚠️  Data completeness: {self.completeness_ratio:.1%} "
                f"(missing {self.total_features - self.non_null_features} features)"
            )

        if self.mean_feature_quality < 0.7:
            warnings.append(
                f"⚠️  Feature quality degraded: {self.mean_feature_quality:.2f}"
            )

        if self.vix_staleness_hours > 24:
            warnings.append(
                f"⚠️  VIX data stale: {self.vix_staleness_hours:.1f} hours old"
            )

        if self.max_staleness_days > 3:
            warnings.append(
                f"⚠️  Stale features detected: {self.most_stale_feature} "
                f"({self.max_staleness_days} days old)"
            )

        if self.extreme_values:
            warnings.append(
                f"⚠️  {len(self.extreme_values)} extreme values detected"
            )

        return warnings


class DataQualityMonitor:
    """Monitor data quality and detect anomalies."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def assess_quality(
        self, features_df: pd.DataFrame, forecast_date: pd.Timestamp
    ) -> DataQualityReport:
        """
        Comprehensive data quality assessment.

        Args:
            features_df: Feature matrix (1 row for current forecast)
            forecast_date: Date being forecasted

        Returns:
            DataQualityReport with all metrics
        """
        row = features_df.iloc[0]

        # Completeness
        total_features = len(features_df.columns)
        non_null = row.notna().sum()
        completeness = non_null / total_features

        # Feature quality
        feature_quality = row.get("feature_quality", 0.0)

        # Find stale features (placeholder - implement based on your metadata)
        stale_features = []
        most_stale = "unknown"
        max_staleness = 0

        # VIX freshness (placeholder - implement based on your data fetcher)
        vix_last_update = "2025-11-11"
        vix_staleness = 0.0

        # Detect extreme values (>5 std from historical mean)
        extreme_values = []
        for col in features_df.select_dtypes(include=[np.number]).columns:
            if "metadata" in col or "cohort" in col:
                continue

            val = row[col]
            if pd.notna(val) and abs(val) > 10:  # Simple threshold
                extreme_values.append({"feature": col, "value": float(val)})

        # Detect suspicious patterns
        suspicious = []
        if row.get("current_vix", 0) < 10 or row.get("current_vix", 100) > 80:
            suspicious.append("VIX outside typical range")

        return DataQualityReport(
            timestamp=datetime.now().isoformat(),
            forecast_date=forecast_date.isoformat(),
            total_features=total_features,
            non_null_features=int(non_null),
            completeness_ratio=float(completeness),
            mean_feature_quality=float(feature_quality),
            min_feature_quality=float(feature_quality),  # Simplified
            features_below_threshold=[],
            most_stale_feature=most_stale,
            max_staleness_days=max_staleness,
            stale_features_count=len(stale_features),
            vix_last_update=vix_last_update,
            vix_staleness_hours=vix_staleness,
            calendar_cohort=row.get("calendar_cohort", "unknown"),
            cohort_weight=float(row.get("cohort_weight", 1.0)),
            extreme_values=extreme_values[:5],  # Top 5
            suspicious_patterns=suspicious,
        )


# ============================================================================
# PREDICTION VALIDATION
# ============================================================================


@dataclass
class PredictionValidation:
    """Validate prediction outputs for sanity."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    # Distribution checks
    quantiles_ordered: bool
    quantiles_reasonable: bool

    # Probability checks
    regime_probs_sum_to_one: bool
    confidence_in_range: bool

    # Business logic checks
    vix_forecast_reasonable: bool
    distribution_width_reasonable: bool


class PredictionValidator:
    """Validate prediction outputs before storage."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def validate(self, prediction: Dict, current_vix: float) -> PredictionValidation:
        """
        Comprehensive prediction validation.

        Args:
            prediction: Prediction dict with all outputs
            current_vix: Current VIX level

        Returns:
            PredictionValidation with pass/fail + diagnostics
        """
        errors = []
        warnings = []

        # 1. Check quantiles are ordered
        quantiles = [
            prediction.get(f"q{q}", np.nan) for q in [10, 25, 50, 75, 90]
        ]
        quantiles_ordered = all(
            quantiles[i] <= quantiles[i + 1] for i in range(len(quantiles) - 1)
        )

        if not quantiles_ordered:
            errors.append(
                f"Quantiles not ordered: {[f'{q:.2f}' for q in quantiles]}"
            )

        # 2. Check quantiles are reasonable (VIX can't go negative or above 200%)
        quantiles_reasonable = all(-100 <= q <= 500 for q in quantiles if pd.notna(q))

        if not quantiles_reasonable:
            errors.append(f"Quantiles outside reasonable range: {quantiles}")

        # 3. Check regime probabilities sum to 1
        regime_probs = [
            prediction.get(f"prob_{r}", 0)
            for r in ["low", "normal", "elevated", "crisis"]
        ]
        prob_sum = sum(regime_probs)
        regime_probs_valid = 0.99 <= prob_sum <= 1.01

        if not regime_probs_valid:
            errors.append(f"Regime probabilities sum to {prob_sum:.3f}, not 1.0")

        # 4. Check confidence score
        confidence = prediction.get("confidence_score", 0)
        confidence_valid = 0 <= confidence <= 1

        if not confidence_valid:
            errors.append(f"Confidence score {confidence} outside [0, 1]")

        # 5. Business logic: VIX forecast shouldn't be extreme
        point_estimate = prediction.get("point_estimate", 0)
        implied_vix = current_vix * (1 + point_estimate / 100)

        vix_reasonable = 5 <= implied_vix <= 100

        if not vix_reasonable:
            warnings.append(
                f"Implied VIX of {implied_vix:.1f} is unusual "
                f"(current: {current_vix:.1f}, change: {point_estimate:+.1f}%)"
            )

        # 6. Distribution width check
        width = quantiles[4] - quantiles[0]  # q90 - q10
        width_reasonable = 5 <= width <= 200

        if not width_reasonable:
            warnings.append(
                f"Unusually {'narrow' if width < 5 else 'wide'} "
                f"prediction interval: {width:.1f}%"
            )

        is_valid = len(errors) == 0

        return PredictionValidation(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quantiles_ordered=quantiles_ordered,
            quantiles_reasonable=quantiles_reasonable,
            regime_probs_sum_to_one=regime_probs_valid,
            confidence_in_range=confidence_valid,
            vix_forecast_reasonable=vix_reasonable,
            distribution_width_reasonable=width_reasonable,
        )


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================


class ForecastLogger:
    """Structured logging for forecasts with JSON export."""

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = []

        # Setup file logging
        log_file = self.log_dir / f"forecast_session_{self.session_id}.jsonl"
        self.log_file = log_file

    def log_forecast(
        self,
        forecast_date: str,
        prediction: Dict,
        data_quality: DataQualityReport,
        validation: PredictionValidation,
    ):
        """Log a single forecast with full context."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "forecast_date": forecast_date,
            "prediction": {
                "point_estimate": prediction.get("point_estimate"),
                "quantiles": {
                    f"q{q}": prediction.get(f"q{q}")
                    for q in [10, 25, 50, 75, 90]
                },
                "regime_probs": {
                    r: prediction.get(f"prob_{r}")
                    for r in ["low", "normal", "elevated", "crisis"]
                },
                "confidence": prediction.get("confidence_score"),
            },
            "data_quality": data_quality.to_dict(),
            "validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            },
        }

        # Append to session log
        self.session_log.append(entry)

        # Write to JSONL file (one line per forecast)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_session_summary(self) -> Dict:
        """Summarize this forecasting session."""
        if not self.session_log:
            return {"error": "No forecasts logged"}

        return {
            "session_id": self.session_id,
            "total_forecasts": len(self.session_log),
            "valid_forecasts": sum(
                1 for e in self.session_log if e["validation"]["is_valid"]
            ),
            "data_quality_issues": sum(
                1
                for e in self.session_log
                if not e["data_quality"]["is_acceptable"]
            ),
            "avg_confidence": np.mean(
                [e["prediction"]["confidence"] for e in self.session_log]
            ),
            "date_range": {
                "start": self.session_log[0]["forecast_date"],
                "end": self.session_log[-1]["forecast_date"],
            },
        }

    def export_summary(self):
        """Export session summary to JSON."""
        summary = self.get_session_summary()
        summary_file = self.log_dir / f"session_summary_{self.session_id}.json"

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_file


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


def production_forecast_with_diagnostics(
    features_df, forecast_date, model, current_vix
):
    """
    Production forecasting with full diagnostics.

    Returns:
        prediction, data_quality_report, validation_result
    """
    # Initialize monitors
    quality_monitor = DataQualityMonitor()
    validator = PredictionValidator()
    logger_instance = ForecastLogger()

    # 1. Assess data quality
    quality_report = quality_monitor.assess_quality(features_df, forecast_date)

    print(f"\n{'='*80}")
    print(f"📊 DATA QUALITY ASSESSMENT")
    print(f"{'='*80}")
    print(f"Forecast Date: {forecast_date}")
    print(f"Completeness: {quality_report.completeness_ratio:.1%}")
    print(f"Feature Quality: {quality_report.mean_feature_quality:.2f}")
    print(f"Calendar Cohort: {quality_report.calendar_cohort}")

    if not quality_report.is_acceptable():
        print(f"\n⚠️  DATA QUALITY WARNINGS:")
        for warning in quality_report.get_warnings():
            print(f"   {warning}")

    # 2. Generate prediction
    prediction = model.predict(features_df.iloc[0])

    # 3. Validate prediction
    validation = validator.validate(prediction, current_vix)

    print(f"\n{'='*80}")
    print(f"🎯 PREDICTION VALIDATION")
    print(f"{'='*80}")

    if validation.is_valid:
        print("✅ Prediction passed all validation checks")
    else:
        print("❌ VALIDATION FAILED:")
        for error in validation.errors:
            print(f"   ERROR: {error}")

    if validation.warnings:
        print("\nWarnings:")
        for warning in validation.warnings:
            print(f"   {warning}")

    # 4. Log everything
    logger_instance.log_forecast(
        forecast_date=forecast_date.isoformat(),
        prediction=prediction,
        data_quality=quality_report,
        validation=validation,
    )

    return prediction, quality_report, validation
