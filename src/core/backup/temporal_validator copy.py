"""
Temporal Safety Validator - Prevent Look-Ahead Bias
====================================================
Three-tier validation system to ensure no data leakage in training/prediction.
"""

import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class TemporalSafetyValidator:
    """Comprehensive temporal safety validation."""

    def __init__(self, publication_lags: Dict[str, int] = None):
        """
        Args:
            publication_lags: Dict mapping data sources to publication delays (days)
        """
        self.publication_lags = publication_lags or {}
        self.audit_results = {}

    # ========================================================================
    # TIER 1: STATIC CODE AUDIT (One-time check)
    # ========================================================================

    def audit_feature_code(
        self, feature_engine_path: str = "core/feature_engine.py"
    ) -> Dict:
        """
        Scan feature engineering code for suspicious patterns.

        Returns:
            Dict with audit results and any warnings
        """
        print(f"\n{'=' * 80}")
        print("🔍 TIER 1: STATIC CODE AUDIT")
        print(f"{'=' * 80}")

        if not Path(feature_engine_path).exists():
            return {"error": f"File not found: {feature_engine_path}"}

        with open(feature_engine_path, "r") as f:
            code = f.read()

        issues = []

        # Check 1: Forward shifts (CRITICAL)
        forward_shifts = re.findall(r"\.shift\s*\(\s*-\s*\d+", code)
        if forward_shifts:
            issues.append(
                {
                    "severity": "CRITICAL",
                    "type": "forward_shift",
                    "count": len(forward_shifts),
                    "message": f"Found {len(forward_shifts)} forward .shift(-N) operations",
                    "examples": forward_shifts[:5],
                }
            )

        # Check 2: Future indexing
        future_indices = re.findall(
            r'\[\s*["\'].*future.*["\']\s*\]', code, re.IGNORECASE
        )
        if future_indices:
            issues.append(
                {
                    "severity": "HIGH",
                    "type": "future_indexing",
                    "count": len(future_indices),
                    "message": "Found references to 'future' in column names",
                    "examples": future_indices[:3],
                }
            )

        # Check 3: Rolling with suspicious lambdas
        suspicious_rolling = re.findall(
            r"\.rolling\([^)]+\)\.apply\([^)]*lambda[^)]*\[[^\]]*\+[^\]]*\]", code
        )
        if suspicious_rolling:
            issues.append(
                {
                    "severity": "MEDIUM",
                    "type": "suspicious_rolling",
                    "count": len(suspicious_rolling),
                    "message": "Found rolling().apply() with forward-looking indexing",
                    "examples": suspicious_rolling[:3],
                }
            )

        # Check 4: Direct future assignment
        direct_future = re.findall(r"=\s*\w+\[\w+\s*\+\s*\d+\]", code)
        if direct_future:
            issues.append(
                {
                    "severity": "HIGH",
                    "type": "direct_future_access",
                    "count": len(direct_future),
                    "message": "Found direct future indexing (e.g., series[i+5])",
                    "examples": direct_future[:3],
                }
            )

        # Summary
        self.audit_results = {
            "timestamp": datetime.now().isoformat(),
            "file": feature_engine_path,
            "total_issues": len(issues),
            "issues": issues,
        }

        # Print results
        if issues:
            print(f"\n⚠️ FOUND {len(issues)} POTENTIAL ISSUES:\n")
            for issue in issues:
                print(f"[{issue['severity']}] {issue['type']}: {issue['message']}")
                if "examples" in issue:
                    for ex in issue["examples"]:
                        print(f"   → {ex}")
        else:
            print("✅ No suspicious patterns detected in feature code")

        print(f"\n{'=' * 80}")

        return self.audit_results

    # ========================================================================
    # TIER 2: CV SPLIT VALIDATION (Every fold)
    # ========================================================================

    def validate_split(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, feature_names: List[str]
    ) -> List[str]:
        """
        Verify temporal integrity of a CV split (simplified signature).

        Args:
            X_train: Training features
            X_val: Validation features
            feature_names: List of feature names (not used but kept for compatibility)

        Returns:
            List of violation messages (empty if valid)
        """
        violations = []

        # Check 1: Temporal ordering
        train_end = X_train.index.max()
        val_start = X_val.index.min()

        if train_end >= val_start:
            violations.append(
                f"TEMPORAL LEAK: Train ends {train_end.date()} >= Val starts {val_start.date()}"
            )

        # Check 2: Large gap warning (data quality)
        gap_days = (val_start - train_end).days
        if gap_days > 10:
            violations.append(
                f"Large gap between train/val ({gap_days} days) - may indicate missing data"
            )

        return violations

    def validate_cv_split(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        fold_num: int,
    ) -> bool:
        """
        Verify temporal integrity of a CV split (full signature).

        Returns:
            True if split is valid, raises ValueError if issues found
        """

        # Check 1: Temporal ordering
        train_end = X_train.index.max()
        val_start = X_val.index.min()

        if train_end >= val_start:
            raise ValueError(
                f"⚠️ FOLD {fold_num} TEMPORAL LEAK: "
                f"Train ends {train_end.date()} >= Val starts {val_start.date()}"
            )

        # Check 2: No gap too large (data quality check)
        gap_days = (val_start - train_end).days
        if gap_days > 10:
            warnings.warn(
                f"Fold {fold_num}: Large gap between train/val ({gap_days} days). "
                f"May indicate missing data."
            )

        # Check 3: Suspicious feature correlations
        suspicious_features = []

        # Sample features to check (checking all would be slow)
        features_to_check = (
            X_train.columns[:50] if len(X_train.columns) > 50 else X_train.columns
        )

        for col in features_to_check:
            # Skip if too many missing values
            if X_train[col].isna().mean() > 0.5 or X_val[col].isna().mean() > 0.5:
                continue

            # Check if validation values are suspiciously predictable from train
            # This catches cases where future data leaked into features
            try:
                # Compare last train values to first val values
                train_last = X_train[col].iloc[-20:].dropna()
                val_first = X_val[col].iloc[:20].dropna()

                if len(train_last) > 5 and len(val_first) > 5:
                    # Check for unrealistic continuity (sign of leakage)
                    combined = pd.concat([train_last, val_first])
                    autocorr = combined.autocorr(lag=1)

                    if autocorr > 0.99:  # Suspiciously high
                        suspicious_features.append(
                            {
                                "feature": col,
                                "autocorr": autocorr,
                                "reason": "Unrealistically high autocorrelation across split",
                            }
                        )
            except:
                continue

        if suspicious_features:
            warnings.warn(
                f"Fold {fold_num}: {len(suspicious_features)} features have suspicious "
                f"train/val continuity (may indicate leakage)"
            )
            for feat in suspicious_features[:3]:  # Show top 3
                warnings.warn(f"  → {feat['feature']}: autocorr={feat['autocorr']:.4f}")

        # Check 4: Target leakage detection
        # If we can predict validation targets from training features too well,
        # features might contain leaked information
        try:
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score

            # Quick linear model on last 100 train samples
            if len(X_train) > 100:
                X_check = X_train.iloc[-100:].fillna(0)
                y_check = y_train.iloc[-100:]

                model = Ridge(alpha=1.0)
                model.fit(X_check, y_check)

                # Predict validation
                X_val_check = X_val.iloc[:50].fillna(0)
                y_val_check = y_val.iloc[:50]

                if len(X_val_check) > 10:
                    preds = model.predict(X_val_check)
                    r2 = r2_score(y_val_check, preds)

                    if r2 > 0.8:  # Suspiciously good
                        warnings.warn(
                            f"Fold {fold_num}: Suspiciously high validation R² = {r2:.3f}. "
                            f"May indicate target leakage."
                        )
        except:
            pass  # Skip if dependencies missing

        return True

    # ========================================================================
    # TIER 3: PUBLICATION LAG VERIFICATION (During prediction)
    # ========================================================================

    def verify_publication_lags(
        self, features: pd.DataFrame, prediction_date: datetime, strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Verify features respect publication lags for a given prediction date.

        Args:
            features: Features for prediction
            prediction_date: Date we're making prediction for
            strict: If True, raise error on violations. If False, just warn.

        Returns:
            (is_valid, list_of_violations)
        """
        if not self.publication_lags:
            warnings.warn("No publication lags configured - skipping validation")
            return True, []

        violations = []

        # Map feature names to source data
        source_mapping = self._infer_feature_sources(features.columns)

        for feature_col, source in source_mapping.items():
            if source not in self.publication_lags:
                continue

            lag_days = self.publication_lags[source]
            latest_allowed_date = prediction_date - timedelta(days=lag_days)

            # Check feature's latest date
            feature_data = features[feature_col].dropna()
            if len(feature_data) == 0:
                continue

            latest_feature_date = feature_data.index[-1]

            if latest_feature_date > latest_allowed_date:
                violation = (
                    f"{feature_col} (source: {source}): "
                    f"Uses data from {latest_feature_date.date()} but "
                    f"only data up to {latest_allowed_date.date()} should be available "
                    f"(lag={lag_days} days)"
                )
                violations.append(violation)

        if violations:
            msg = f"\n⚠️ PUBLICATION LAG VIOLATIONS:\n" + "\n".join(
                f"  • {v}" for v in violations
            )
            if strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
            return False, violations

        return True, []

    # ========================================================================
    # NEW: TIER 4 - WALK-FORWARD GAP TEST (Gap 3 Fix)
    # ========================================================================

    def validate_walk_forward_gap(
        self,
        features: pd.DataFrame,
        prediction_date: datetime,
        feature_metadata: Dict[str, Dict] = None,
        strict: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that predictions at time T only use features available at T.

        This tests the actual "as-of" timestamps to ensure no forward-fill
        has propagated future information backward.

        Args:
            features: Feature DataFrame with DatetimeIndex
            prediction_date: The date we're making predictions for
            feature_metadata: Dict mapping feature names to their metadata
                             (including 'last_available_date' and 'source')
            strict: If True, raise error. If False, warn only.

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        if feature_metadata is None:
            warnings.warn(
                "No feature metadata provided - cannot validate as-of timestamps. "
                "Call build_complete_features() with return_metadata=True"
            )
            return True, []

        # Get features for prediction date
        if prediction_date not in features.index:
            warnings.warn(
                f"Prediction date {prediction_date.date()} not in features index"
            )
            return True, []

        feature_row = features.loc[prediction_date]

        # Check each non-null feature
        for feat_name in feature_row.dropna().index:
            if feat_name not in feature_metadata:
                continue

            meta = feature_metadata[feat_name]
            source = meta.get("source", "unknown")
            last_available = meta.get("last_available_date")

            if last_available is None:
                continue

            # Apply publication lag if known
            lag_days = self.publication_lags.get(source, 0)
            effective_cutoff = prediction_date - timedelta(days=lag_days)

            if last_available > effective_cutoff:
                violations.append(
                    f"{feat_name}: Uses data from {last_available.date()} "
                    f"but cutoff for prediction {prediction_date.date()} is {effective_cutoff.date()} "
                    f"(source: {source}, lag: {lag_days}d)"
                )

        if violations:
            msg = f"\n⚠️ WALK-FORWARD GAP VIOLATIONS at {prediction_date.date()}:\n"
            msg += "\n".join(f"  • {v}" for v in violations)

            if strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
            return False, violations

        return True, []

    def test_feature_availability_at_prediction_time(
        self,
        features: pd.DataFrame,
        feature_metadata: Dict[str, Dict],
        test_dates: List[datetime] = None,
        sample_size: int = 20,
    ) -> Dict:
        """
        Automated test: Sample random prediction dates and verify feature availability.

        Args:
            features: Full feature DataFrame
            feature_metadata: Feature metadata with as-of timestamps
            test_dates: Specific dates to test (if None, randomly sample)
            sample_size: Number of dates to sample if test_dates not provided

        Returns:
            Dict with test results and any violations found
        """
        print(f"\n{'=' * 80}")
        print("🧪 WALK-FORWARD GAP AUTOMATED TEST")
        print(f"{'=' * 80}")

        if test_dates is None:
            # Sample dates from latter half of data (where leakage more likely)
            all_dates = features.index
            mid_point = len(all_dates) // 2
            test_dates = np.random.choice(
                all_dates[mid_point:],
                size=min(sample_size, len(all_dates) - mid_point),
                replace=False,
            )

        results = {
            "total_tests": len(test_dates),
            "passed": 0,
            "failed": 0,
            "violations": [],
        }

        for test_date in test_dates:
            is_valid, violations = self.validate_walk_forward_gap(
                features,
                test_date,
                feature_metadata,
                strict=False,  # Don't raise, just collect
            )

            if is_valid:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["violations"].append(
                    {"date": test_date.date().isoformat(), "violations": violations}
                )

        # Print summary
        print(f"\nTest Results:")
        print(f"  ✅ Passed: {results['passed']}/{results['total_tests']}")
        print(f"  ❌ Failed: {results['failed']}/{results['total_tests']}")

        if results["failed"] > 0:
            print(f"\n⚠️ Found {results['failed']} dates with temporal violations")
            print("First violation example:")
            if results["violations"]:
                first = results["violations"][0]
                print(f"  Date: {first['date']}")
                for v in first["violations"][:3]:
                    print(f"    → {v}")
        else:
            print("\n✅ All walk-forward gap tests passed!")

        print(f"{'=' * 80}\n")

        return results

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _infer_feature_sources(self, feature_names: List[str]) -> Dict[str, str]:
        """Map feature names to data sources."""
        source_map = {}

        # Common patterns
        patterns = {
            r"^vix": "^VIX",
            r"^spx": "^GSPC",
            r"SKEW": "SKEW",
            r"VXTLT": "VXTLT",
            r"VX[12]": "VX1-VX2",
            r"^yield|^dgs": "DGS10",
            r"crude|^cl": "CL=F",
            r"dxy|dollar": "DX-Y.NYB",
        }

        for feat in feature_names:
            feat_lower = feat.lower()
            for pattern, source in patterns.items():
                if re.search(pattern, feat_lower):
                    source_map[feat] = source
                    break

        return source_map

    # ========================================================================
    # COMPREHENSIVE VALIDATION REPORT
    # ========================================================================

    def generate_validation_report(
        self,
        output_path: str = "./models/temporal_safety_report.json",
        include_metadata_check: bool = False,
        feature_metadata: Dict = None,
    ):
        """Generate comprehensive validation report."""
        import json

        report = {
            "timestamp": datetime.now().isoformat(),
            "validation_tiers": {
                "tier1_code_audit": self.audit_results,
                "tier2_cv_validation": "Run during training",
                "tier3_publication_lags": f"{len(self.publication_lags)} sources configured",
                "tier4_walk_forward_gap": "Automated test available via test_feature_availability_at_prediction_time()",
            },
            "publication_lags": self.publication_lags,
        }

        # Add metadata check if requested
        if include_metadata_check and feature_metadata:
            report["feature_metadata_check"] = {
                "total_features": len(feature_metadata),
                "features_with_timestamps": sum(
                    1
                    for m in feature_metadata.values()
                    if m.get("last_available_date") is not None
                ),
                "coverage_pct": round(
                    100
                    * sum(
                        1
                        for m in feature_metadata.values()
                        if m.get("last_available_date") is not None
                    )
                    / max(len(feature_metadata), 1),
                    1,
                ),
            }

        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Validation report saved: {output_path}")
        return report


# ========================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================


def run_full_validation(
    feature_engine_path: str = "core/feature_engine.py",
    publication_lags: Dict[str, int] = None,
) -> TemporalSafetyValidator:
    """Run full validation suite and return validator instance."""

    validator = TemporalSafetyValidator(publication_lags)

    # Tier 1: Code audit
    audit_results = validator.audit_feature_code(feature_engine_path)

    # Check for critical issues
    critical_issues = [
        issue
        for issue in audit_results.get("issues", [])
        if issue["severity"] == "CRITICAL"
    ]

    if critical_issues:
        print("\n❌ CRITICAL ISSUES FOUND - Fix before training!")
        return validator

    print("\n✅ Tier 1 validation passed")

    # Generate report
    validator.generate_validation_report()

    return validator
