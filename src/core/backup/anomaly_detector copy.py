"""Consolidated Anomaly Detection System

Robust anomaly scoring with feature quality validation, coverage penalties,
and outlier-resistant percentile calculation.
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

from config import ANOMALY_FEATURE_GROUPS, ANOMALY_THRESHOLDS, RANDOM_STATE

try:
    import pytz

    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    warnings.warn(
        "pytz not available - persistence streak timing will not be timezone-aware"
    )

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def validate_feature_quality(
    features: pd.DataFrame, feature_list: list, detector_name: str
) -> Tuple[list, dict]:
    """Filter out constant/zero features before detection."""
    quality_report = {}
    valid_features = []

    for feat in feature_list:
        if feat not in features.columns:
            quality_report[feat] = "missing"
            continue

        series = features[feat]

        if (series == 0).sum() / len(series) > 0.95:
            quality_report[feat] = "constant_zero"
            continue

        if series.isna().sum() / len(series) > 0.80:
            quality_report[feat] = "too_sparse"
            continue

        if series.std() < 1e-6:
            quality_report[feat] = "no_variance"
            continue

        unique_ratio = series.nunique() / len(series.dropna())
        if unique_ratio < 0.05 and series.nunique() < 3:
            quality_report[feat] = "too_few_unique"
            continue

        valid_features.append(feat)
        quality_report[feat] = "valid"

    removed_count = len(feature_list) - len(valid_features)
    if removed_count > 0:
        print(
            f"  {detector_name}: Removed {removed_count}/{len(feature_list)} low-quality features"
        )

    return valid_features, quality_report


def calculate_robust_anomaly_score(
    raw_score: float,
    training_distribution: np.ndarray,
    min_percentile: float = 0.05,
    max_percentile: float = 0.95,
) -> float:
    """Robust percentile calculation that handles outliers."""
    lower_bound = np.percentile(training_distribution, min_percentile * 100)
    upper_bound = np.percentile(training_distribution, max_percentile * 100)

    buffer = (upper_bound - lower_bound) * 0.1
    clipped_score = np.clip(raw_score, lower_bound - buffer, upper_bound + buffer)

    percentile = (training_distribution <= clipped_score).sum() / len(
        training_distribution
    )
    anomaly_score = 1.0 - percentile

    return float(np.clip(anomaly_score, 0.0, 0.95))


def calculate_coverage_penalty(coverage: float, min_coverage: float = 0.7) -> float:
    """Apply penalty to detectors with low feature coverage."""
    if coverage >= min_coverage:
        return 1.0
    else:
        penalty = (coverage / min_coverage) ** 2
        return max(penalty, 0.1)


class MultiDimensionalAnomalyDetector:
    """15 independent Isolation Forests with feature importance and quality validation."""

    def __init__(self, contamination: float = 0.01, random_state: int = RANDOM_STATE):
        self.contamination = contamination
        self.random_state = random_state
        self.detectors = {}
        self.scalers = {}
        self.feature_groups = ANOMALY_FEATURE_GROUPS.copy()
        self.random_subspaces = []
        self.training_distributions = {}
        self.feature_importances = {}
        self.detector_coverage = {}
        self.feature_quality_reports = {}
        self.trained = False
        self.training_ensemble_scores = []
        self.statistical_thresholds = None
        self.importance_config = {
            "method": "shap" if SHAP_AVAILABLE else "permutation",
            "n_samples": 500,
            "n_repeats": 1,
        }

    def _calculate_feature_importance(
        self, detector, X_scaled, baseline_scores, feature_names, verbose=False
    ):
        if SHAP_AVAILABLE and self.importance_config["method"] == "shap":
            try:
                return self._calculate_shap_importance(
                    detector, X_scaled, feature_names, verbose
                )
            except Exception as e:
                if verbose:
                    warnings.warn(f"SHAP failed, using permutation: {str(e)[:100]}")
        return self._calculate_permutation_importance(
            detector, X_scaled, baseline_scores, feature_names, verbose
        )

    def _calculate_shap_importance(
        self, detector, X_scaled, feature_names, verbose=False
    ):
        n_samples = min(self.importance_config["n_samples"], len(X_scaled))
        sample_indices = np.random.RandomState(self.random_state).choice(
            len(X_scaled), n_samples, replace=False
        )
        X_sample = X_scaled[sample_indices]
        explainer = shap.TreeExplainer(detector)
        shap_values = explainer.shap_values(X_sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total = mean_abs_shap.sum()
        normalized_importance = (
            mean_abs_shap / total
            if total > 0
            else np.ones(len(feature_names)) / len(feature_names)
        )
        return {
            name: float(imp) for name, imp in zip(feature_names, normalized_importance)
        }

    def _calculate_permutation_importance(
        self, detector, X_scaled, baseline_scores, feature_names, verbose=False
    ):
        try:
            if (
                len(X_scaled) == 0
                or len(feature_names) == 0
                or len(X_scaled) != len(baseline_scores)
            ):
                return {name: 1.0 / len(feature_names) for name in feature_names}

            n_samples = min(self.importance_config["n_samples"], len(X_scaled))
            if n_samples < len(X_scaled):
                sample_indices = np.random.RandomState(self.random_state).choice(
                    len(X_scaled), n_samples, replace=False
                )
                X_sample = X_scaled[sample_indices].copy()
                baseline_mean = np.mean(baseline_scores[sample_indices])
            else:
                X_sample = X_scaled.copy()
                baseline_mean = np.mean(baseline_scores)

            importances = {}
            for i, feature_name in enumerate(feature_names):
                try:
                    original_col = X_sample[:, i].copy()
                    importance_values = []
                    for repeat in range(self.importance_config["n_repeats"]):
                        np.random.RandomState(
                            self.random_state + i + repeat * 1000
                        ).shuffle(X_sample[:, i])
                        permuted_scores = detector.score_samples(X_sample)
                        importance_values.append(
                            abs(baseline_mean - np.mean(permuted_scores))
                        )
                        X_sample[:, i] = original_col
                    importances[feature_name] = np.mean(importance_values)
                except:
                    importances[feature_name] = 0.0

            total = sum(importances.values())
            if total > 0:
                importances = {k: v / total for k, v in importances.items()}
            else:
                importances = {name: 1.0 / len(feature_names) for name in feature_names}

            return {
                k: v if np.isfinite(v) else 1.0 / len(feature_names)
                for k, v in importances.items()
            }
        except:
            return {name: 1.0 / len(feature_names) for name in feature_names}

    def calculate_statistical_thresholds(self) -> dict:
        """Calculate thresholds from historical ensemble scores."""
        if len(self.training_ensemble_scores) == 0:
            return {"moderate": 0.70, "high": 0.78, "critical": 0.88}

        scores = np.array(self.training_ensemble_scores)
        thresholds = {
            "moderate": float(np.percentile(scores, 85)),
            "high": float(np.percentile(scores, 92)),
            "critical": float(np.percentile(scores, 98)),
        }

        self.logger.info(
            f"Statistical Thresholds: Moderate={thresholds['moderate']:.4f}, "
            f"High={thresholds['high']:.4f}, Critical={thresholds['critical']:.4f}"
        )

        return thresholds

    def train(self, features: pd.DataFrame, verbose: bool = True):
        """Train all detectors with feature quality validation."""
        if verbose:
            print("\n" + "=" * 60)
            print("🧠 TRAINING ANOMALY DETECTORS")
            print("=" * 60)
            print(f"Total available features: {len(features.columns)}")

        available_cols = set(features.columns)

        for name, feature_list in self.feature_groups.items():
            try:
                if verbose:
                    self.logger.debug(f"Training: {name}")

                available_features = [f for f in feature_list if f in features.columns]
                valid_features, quality_report = validate_feature_quality(
                    features, available_features, name
                )

                self.feature_quality_reports[name] = quality_report

                if len(valid_features) < 3:
                    if verbose:
                        self.logger.warning(
                            f"{name}: Only {len(valid_features)} valid features, skipping"
                        )
                    continue

                X = features[valid_features].fillna(0)
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)

                detector = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_estimators=100,
                    max_samples="auto",
                    n_jobs=-1,
                )
                detector.fit(X_scaled)

                training_scores = detector.score_samples(X_scaled)

                self.detectors[name] = detector
                self.scalers[name] = scaler
                self.training_distributions[name] = training_scores
                self.detector_coverage[name] = len(valid_features) / len(feature_list)

                self.feature_importances[name] = self._calculate_feature_importance(
                    detector, X_scaled, training_scores, valid_features, verbose=False
                )

            except Exception as e:
                if verbose:
                    self.logger.error(f"{name}: Training failed - {str(e)[:100]}")
                continue

        # Generate random subspaces
        if verbose:
            self.logger.debug("Generating 5 random subspace detectors...")

        all_features = list(available_cols)
        for i in range(5):
            try:
                subspace_size = np.random.randint(8, 15)
                subspace_features = (
                    np.random.RandomState(self.random_state + i)
                    .choice(
                        all_features,
                        size=min(subspace_size, len(all_features)),
                        replace=False,
                    )
                    .tolist()
                )

                name = f"random_{i + 1}"

                valid_features, quality_report = validate_feature_quality(
                    features, subspace_features, name
                )

                self.feature_quality_reports[name] = quality_report

                if len(valid_features) < 3:
                    continue

                X = features[valid_features].fillna(0)
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)

                detector = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state + i,
                    n_estimators=100,
                    max_samples="auto",
                    n_jobs=-1,
                )
                detector.fit(X_scaled)

                training_scores = detector.score_samples(X_scaled)

                self.detectors[name] = detector
                self.scalers[name] = scaler
                self.training_distributions[name] = training_scores
                self.random_subspaces.append(valid_features)
                self.detector_coverage[name] = 1.0

                self.feature_importances[name] = self._calculate_feature_importance(
                    detector, X_scaled, training_scores, valid_features, verbose=False
                )

            except Exception as e:
                if verbose:
                    self.logger.warning(
                        f"Random subspace {i + 1} failed: {str(e)[:50]}"
                    )

        self.trained = True

        if verbose:
            print(f"\n✅ Training complete: {len(self.detectors)} detectors active")

        self._compute_statistical_thresholds(features, verbose=verbose)
        self.statistical_thresholds = self.calculate_statistical_thresholds()

    def detect(self, features: pd.DataFrame, verbose: bool = True) -> dict:
        """Detect anomalies with robust scoring and coverage penalties."""
        if not self.trained:
            raise ValueError("Detector not trained")

        results = {
            "domain_anomalies": {},
            "random_anomalies": {},
            "ensemble": {},
            "data_quality": {},
        }

        scores = []
        detector_weights = []
        active_detectors = 0
        missing_features = []

        for name, detector in self.detectors.items():
            feature_list = (
                self.random_subspaces[int(name.split("_")[1]) - 1]
                if name.startswith("random_")
                else [f for f in self.feature_groups[name] if f in features.columns]
            )

            available = [f for f in feature_list if f in features.columns]
            coverage = len(available) / len(feature_list)

            if coverage < 0.5:
                missing_features.append({"detector": name, "coverage": coverage})
                continue

            try:
                X = features[available].fillna(0)
                X_scaled = self.scalers[name].transform(X)
                raw_score = detector.score_samples(X_scaled)[0]

                training_dist = self.training_distributions[name]
                anomaly_score = calculate_robust_anomaly_score(raw_score, training_dist)

                coverage_penalty = calculate_coverage_penalty(coverage)
                adjusted_score = anomaly_score * coverage_penalty

                scores.append(adjusted_score)
                detector_weights.append(coverage * coverage_penalty)
                active_detectors += 1

                level, _, _ = self.classify_anomaly(
                    adjusted_score, method="statistical"
                )

                result_dict = {
                    "score": float(adjusted_score),
                    "raw_anomaly_score": float(anomaly_score),
                    "percentile": float(adjusted_score * 100),
                    "level": level,
                    "coverage": float(coverage),
                    "coverage_penalty": float(coverage_penalty),
                    "weight": float(coverage * coverage_penalty),
                }

                if name.startswith("random_"):
                    results["random_anomalies"][name] = result_dict
                else:
                    results["domain_anomalies"][name] = result_dict
            except Exception as e:
                missing_features.append(
                    {"detector": name, "coverage": coverage, "error": str(e)}
                )

        if scores:
            weights = np.array(detector_weights)
            weighted_scores = np.array(scores) * weights
            ensemble_score = (
                float(weighted_scores.sum() / weights.sum())
                if weights.sum() > 0
                else float(np.mean(scores))
            )
            ensemble_score = min(ensemble_score, 0.95)

            results["ensemble"] = {
                "score": ensemble_score,
                "std": float(np.std(scores)),
                "max_anomaly": float(np.max(scores)),
                "min_anomaly": float(np.min(scores)),
                "n_detectors": active_detectors,
                "mean_weight": float(np.mean(weights)),
                "weighted": True,
            }

        results["data_quality"] = {
            "active_detectors": active_detectors,
            "total_detectors": len(self.detectors),
            "missing_features": missing_features,
            "weight_stats": {
                "mean": float(np.mean(weights)) if len(weights) > 0 else 0.0,
                "min": float(np.min(weights)) if len(weights) > 0 else 0.0,
                "max": float(np.max(weights)) if len(weights) > 0 else 0.0,
            },
            "feature_quality_summary": self._get_quality_summary(),
        }

        return results

    def _get_quality_summary(self) -> dict:
        """Summarize feature quality across all detectors."""
        summary = {}
        for detector_name, quality_report in self.feature_quality_reports.items():
            valid_count = sum(
                1 for status in quality_report.values() if status == "valid"
            )
            total_count = len(quality_report)
            summary[detector_name] = {
                "valid_ratio": valid_count / total_count if total_count > 0 else 0.0,
                "valid_count": valid_count,
                "total_count": total_count,
            }
        return summary

    def calculate_historical_persistence_stats(
        self,
        ensemble_scores: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        threshold: float = None,
    ) -> Dict:
        """Calculate persistence stats from complete historical ensemble scores."""
        if threshold is None:
            threshold = (
                self.statistical_thresholds.get("high", 0.78)
                if isinstance(self.statistical_thresholds, dict)
                else 0.78
            )

        if isinstance(ensemble_scores, list):
            ensemble_scores = np.array(ensemble_scores)

        is_anomalous = ensemble_scores >= threshold

        streaks, current_streak = [], 0
        for anomalous in is_anomalous:
            if anomalous:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0

        if dates is not None and current_streak > 0 and PYTZ_AVAILABLE:
            try:
                last_date = pd.Timestamp(dates[-1])
                et_tz = pytz.timezone("US/Eastern")
                now_et = datetime.now(et_tz)
                today_et = pd.Timestamp.now(tz="US/Eastern").normalize()

                last_date_normalized = last_date.normalize()
                if hasattr(last_date, "tz") and last_date.tz is not None:
                    last_date_normalized = last_date.tz_convert(
                        "US/Eastern"
                    ).normalize()

                if last_date_normalized == today_et:
                    market_closed = now_et.hour >= 16
                    if not market_closed and is_anomalous[-1]:
                        current_streak = max(0, current_streak - 1)
            except Exception as e:
                warnings.warn(f"Streak timing correction failed: {e}")

        total_anomaly_days = int(is_anomalous.sum())

        return {
            "current_streak": int(current_streak),
            "mean_duration": float(np.mean(streaks)) if streaks else 0.0,
            "max_duration": int(np.max(streaks)) if streaks else 0,
            "median_duration": float(np.median(streaks)) if streaks else 0.0,
            "total_anomaly_days": total_anomaly_days,
            "anomaly_rate": float(total_anomaly_days / len(ensemble_scores))
            if len(ensemble_scores) > 0
            else 0.0,
            "num_episodes": len(streaks),
            "threshold_used": float(threshold),
        }

    def get_top_anomalies(self, result: dict, top_n: int = 3) -> list:
        all_anomalies = []
        for name, data in result.get("domain_anomalies", {}).items():
            all_anomalies.append((name, data["score"]))
        for name, data in result.get("random_anomalies", {}).items():
            all_anomalies.append((name, data["score"]))
        all_anomalies.sort(key=lambda x: x[1], reverse=True)
        return all_anomalies[:top_n]

    def get_feature_contributions(
        self, detector_name: str, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        if detector_name not in self.feature_importances:
            return []
        importances = self.feature_importances[detector_name]
        return sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def _compute_statistical_thresholds(
        self, features: pd.DataFrame, verbose: bool = True
    ):
        """Compute ensemble scores for all training data to establish thresholds."""
        self.training_ensemble_scores = []
        batch_size = 100
        for i in range(0, len(features), batch_size):
            batch = features.iloc[i : i + batch_size]
            for idx in range(len(batch)):
                try:
                    result = self.detect(batch.iloc[[idx]], verbose=False)
                    self.training_ensemble_scores.append(result["ensemble"]["score"])
                except:
                    pass

        if verbose:
            print(f"\n✅ Computed {len(self.training_ensemble_scores)} ensemble scores")

    def classify_anomaly(self, score: float, method: str = "statistical") -> tuple:
        """Classify anomaly severity using statistical thresholds."""
        thresholds = (
            self.statistical_thresholds
            if self.statistical_thresholds
            else {"moderate": 0.70, "high": 0.78, "critical": 0.88}
        )

        if "moderate_ci" in thresholds:
            thresholds = {
                "moderate": thresholds["moderate"],
                "high": thresholds["high"],
                "critical": thresholds["critical"],
            }

        if len(self.training_ensemble_scores) > 0:
            p_value = (np.array(self.training_ensemble_scores) >= score).mean()
            confidence = 1 - p_value
        else:
            p_value = None
            confidence = None

        if score >= thresholds["critical"]:
            level = "CRITICAL"
        elif score >= thresholds["high"]:
            level = "HIGH"
        elif score >= thresholds["moderate"]:
            level = "MODERATE"
        else:
            level = "NORMAL"

        return level, p_value, confidence

    @property
    def logger(self):
        """Simple logger property for backwards compatibility."""
        import logging

        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
