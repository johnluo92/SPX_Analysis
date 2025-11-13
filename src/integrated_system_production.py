"""Integrated Market Analysis System V5 - Probabilistic Forecasting
Upgraded architecture with probabilistic distribution forecasting.
Anomaly detection preserved but probabilistic forecasting is primary focus.
"""

import argparse
import gc
import json
import os
import pickle
import subprocess
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    CALENDAR_COHORTS,
    CBOE_DATA_DIR,
    ENABLE_TRAINING,
    FEATURE_QUALITY_CONFIG,
    PREDICTION_DB_CONFIG,
    RANDOM_STATE,
    REGIME_BOUNDARIES,
    REGIME_NAMES,
    TARGET_CONFIG,
    TRAINING_YEARS,
)
from core.anomaly_detector import MultiDimensionalAnomalyDetector
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engine import UnifiedFeatureEngine
from core.forecast_calibrator import ForecastCalibrator
from core.prediction_database import PredictionDatabase
from core.temporal_validator import TemporalSafetyValidator as TemporalValidator
from core.xgboost_feature_selector_v2 import run_intelligent_feature_selection
from core.xgboost_trainer_v2 import (
    ProbabilisticVIXForecaster,
    train_probabilistic_forecaster,
)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyOrchestrator:
    def __init__(self):
        self.fetcher = UnifiedDataFetcher()
        self.anomaly_detector = None
        self.vix_history_all = None
        self.vix_ml = None
        self.spx_ml = None
        self.features = None
        self.regime_stats = None
        self.historical_ensemble_scores = None
        self.trained = False

    def train(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        vix_history_all: pd.Series = None,
        verbose: bool = True,
    ):
        if verbose:
            print("\n[Anomaly Orchestrator] Training...")

        self.features = features
        self.vix_ml = vix
        self.spx_ml = spx
        self.vix_history_all = vix_history_all if vix_history_all is not None else vix

        self.regime_stats = self._compute_regime_statistics(self.vix_history_all)

        self.anomaly_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05, random_state=RANDOM_STATE
        )
        self.anomaly_detector.train(features.fillna(0), verbose=verbose)

        self._generate_historical_scores(verbose)

        self.trained = True
        if verbose:
            print("✅ Anomaly orchestrator trained")

    def _generate_historical_scores(self, verbose: bool = True):
        """Generate complete historical anomaly scores."""
        if not self.anomaly_detector or not self.anomaly_detector.trained:
            warnings.warn("Anomaly detector not trained")
            return

        scores = []
        for i in range(len(self.features)):
            result = self.anomaly_detector.detect(
                self.features.iloc[[i]], verbose=False
            )
            scores.append(result["ensemble"]["score"])

        self.historical_ensemble_scores = np.array(scores)
        if verbose:
            print(f"Generated {len(scores)} historical anomaly scores")

    def detect_current(self, verbose: bool = False) -> dict:
        """Run anomaly detection on most recent feature row."""
        if not self.trained:
            raise ValueError("Must train before detecting")

        return self.anomaly_detector.detect(self.features.iloc[[-1]], verbose=verbose)

    def get_persistence_stats(self) -> dict:
        """Calculate anomaly persistence statistics."""
        if not self.trained or self.historical_ensemble_scores is None:
            return {
                "current_streak": 0,
                "mean_duration": 0.0,
                "max_duration": 0,
                "total_anomaly_days": 0,
                "anomaly_rate": 0.0,
                "num_episodes": 0,
            }

        return self.anomaly_detector.calculate_historical_persistence_stats(
            self.historical_ensemble_scores, dates=self.features.index
        )

    def _compute_regime_statistics(self, vix_series: pd.Series) -> dict:
        stats = {
            "observation_period": {
                "start_date": str(vix_series.index[0]),
                "end_date": str(vix_series.index[-1]),
                "total_days": len(vix_series),
            },
            "regimes": [],
        }

        for regime_id in range(len(REGIME_NAMES)):
            regime_name = REGIME_NAMES[regime_id]
            lower_bound = REGIME_BOUNDARIES[regime_id]
            upper_bound = (
                REGIME_BOUNDARIES[regime_id + 1]
                if regime_id < len(REGIME_BOUNDARIES) - 1
                else np.inf
            )

            regime_mask = (vix_series >= lower_bound) & (vix_series < upper_bound)
            regime_days = vix_series[regime_mask]

            vix_regimes = pd.Series(index=vix_series.index, dtype=int)
            for rid in range(len(REGIME_NAMES)):
                lb = REGIME_BOUNDARIES[rid]
                ub = (
                    REGIME_BOUNDARIES[rid + 1]
                    if rid < len(REGIME_BOUNDARIES) - 1
                    else np.inf
                )
                mask = (vix_series >= lb) & (vix_series < ub)
                vix_regimes[mask] = rid

            regime_transitions = vix_regimes[vix_regimes == regime_id]
            future_regimes_5d = vix_regimes.shift(-5)
            valid_mask = future_regimes_5d.notna()

            valid_indices = regime_transitions.index.intersection(
                valid_mask[valid_mask].index
            )
            transitions_5d = future_regimes_5d[valid_indices].value_counts()
            total_opp = len(valid_indices)

            regime_info = {
                "regime_id": int(regime_id),
                "regime_name": regime_name,
                "boundaries": [
                    float(lower_bound),
                    float(upper_bound) if upper_bound != np.inf else 100.0,
                ],
                "observations": {
                    "count": int(len(regime_days)),
                    "percentage": float(len(regime_days) / len(vix_series) * 100)
                    if len(vix_series) > 0
                    else 0.0,
                    "mean_vix": float(regime_days.mean())
                    if len(regime_days) > 0
                    else 0.0,
                    "std_vix": float(regime_days.std())
                    if len(regime_days) > 0
                    else 0.0,
                },
                "transitions_5d": {
                    "persistence": {
                        "probability": float(
                            transitions_5d.get(regime_id, 0) / total_opp
                        )
                        if total_opp > 0
                        else 0.0,
                        "observations": int(transitions_5d.get(regime_id, 0)),
                        "total_opportunities": int(total_opp),
                    },
                    "to_other_regimes": {
                        int(other): {
                            "probability": float(
                                transitions_5d.get(other, 0) / total_opp
                            )
                            if total_opp > 0
                            else 0.0,
                            "observations": int(transitions_5d.get(other, 0)),
                            "total_opportunities": int(total_opp),
                        }
                        for other in range(len(REGIME_NAMES))
                        if other != regime_id
                    },
                },
            }
            stats["regimes"].append(regime_info)

        return stats

    def save_state(self, filepath: str = "./json_data/model_cache.pkl"):
        """Save model state for quick refresh without retraining."""
        if not self.trained:
            raise ValueError("Must train before saving state")

        state = {
            "detectors": self.anomaly_detector.detectors,
            "scalers": self.anomaly_detector.scalers,
            "training_distributions": self.anomaly_detector.training_distributions,
            "feature_groups": self.anomaly_detector.feature_groups,
            "random_subspaces": self.anomaly_detector.random_subspaces,
            "statistical_thresholds": self.anomaly_detector.statistical_thresholds,
            "vix_history": self.vix_ml.tail(252).to_dict(),
            "spx_history": self.spx_ml.tail(252).to_dict(),
            "last_features": self.features.tail(1).to_dict(),
            "feature_columns": self.features.columns.tolist(),
            "export_timestamp": pd.Timestamp.now().isoformat(),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(
            f"✅ Saved state: {filepath} ({Path(filepath).stat().st_size / (1024 * 1024):.2f} MB)"
        )

    def load_state(self, filepath: str = "./json_data/model_cache.pkl"):
        """Load cached state for fast refresh."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.anomaly_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05, random_state=RANDOM_STATE
        )
        self.anomaly_detector.detectors = state["detectors"]
        self.anomaly_detector.scalers = state["scalers"]
        self.anomaly_detector.training_distributions = state["training_distributions"]
        self.anomaly_detector.feature_groups = state["feature_groups"]
        self.anomaly_detector.random_subspaces = state["random_subspaces"]
        self.anomaly_detector.statistical_thresholds = state["statistical_thresholds"]
        self.anomaly_detector.trained = True

        self.vix_ml = self._dict_to_series(state["vix_history"])
        self.spx_ml = self._dict_to_series(state["spx_history"])
        self.features = self._dict_to_dataframe(
            state["last_features"], state["feature_columns"]
        )

        self.trained = True
        print(f"✅ Loaded state from {filepath}")

    def _dict_to_series(self, d: dict) -> pd.Series:
        """Convert dict to pandas Series with DatetimeIndex."""
        dates = pd.to_datetime(list(d.keys()))
        return pd.Series(list(d.values()), index=dates)

    def _dict_to_dataframe(self, d: dict, columns: list) -> pd.DataFrame:
        """Convert dict to pandas DataFrame with DatetimeIndex."""
        dates = pd.to_datetime(list(next(iter(d.values())).keys()))
        data = {col: list(d[col].values()) for col in columns}
        return pd.DataFrame(data, index=dates)


class IntegratedSystem:
    def __init__(self, models_dir="models"):
        logger.info("=" * 80)
        logger.info("INTEGRATED PROBABILISTIC FORECASTING SYSTEM V5")
        logger.info("=" * 80)

        # Core components
        self.data_fetcher = UnifiedDataFetcher()
        self.feature_engine = UnifiedFeatureEngine(data_fetcher=self.data_fetcher)
        self.forecaster = ProbabilisticVIXForecaster()
        self.validator = TemporalValidator()
        self.prediction_db = PredictionDatabase()

        # Anomaly detector (runs independently)
        self.orchestrator = AnomalyOrchestrator()

        # Load trained models
        self.models_dir = Path(models_dir)
        self._load_models()
        # Load calibrator (if available)
        self.calibrator = ForecastCalibrator.load()
        if self.calibrator:
            logger.info("📊 Forecast calibrator loaded")
        else:
            logger.info("ℹ️  No calibrator found - forecasts will not be calibrated")

        # State tracking
        self.last_forecast = None
        self.forecast_history = []
        self.trained = False
        self._cached_anomaly_result = None
        self._cache_timestamp = None

        self._feature_cache = None
        self._feature_cache_date = None

        # Memory monitoring
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
            self.baseline_memory_mb = None
            self.memory_history = []
            self.memory_monitoring_enabled = True
        else:
            self.memory_monitoring_enabled = False

        logger.info("✅ System initialized")

    def _load_models(self):
        """Load all trained cohort models."""
        logger.info("📂 Loading trained models...")

        model_files = list(self.models_dir.glob("probabilistic_forecaster_*.pkl"))

        if len(model_files) == 0:
            logger.warning("⚠️ No trained models found. Run training first.")
            return

        for model_file in model_files:
            cohort = model_file.stem.replace("probabilistic_forecaster_", "")
            self.forecaster.load(cohort, self.models_dir)
            logger.info(f"   ✅ Loaded: {cohort}")

        logger.info(f"📊 Total cohorts loaded: {len(self.forecaster.models)}")

    def _log_forecast_summary(self, distribution):
        """Log human-readable forecast summary."""
        logger.info("\nFORECAST SUMMARY")
        logger.info("─" * 60)

        # Point estimate
        point = distribution["point_estimate"]
        logger.info(f"Point Estimate:     {point:+.1f}%")

        # Quantiles
        quantiles = distribution["quantiles"]
        logger.info(f"Distribution:")
        logger.info(f"   10th percentile: {quantiles['q10']:+.1f}%")
        logger.info(f"   25th percentile: {quantiles['q25']:+.1f}%")
        logger.info(f"   Median (50th):   {quantiles['q50']:+.1f}%")
        logger.info(f"   75th percentile: {quantiles['q75']:+.1f}%")
        logger.info(f"   90th percentile: {quantiles['q90']:+.1f}%")
        prob_up = distribution["direction_probability"]
        logger.info(f"Direction Forecast:")
        logger.info(f"   Probability UP:   {prob_up * 100:5.1f}%")
        logger.info(f"   Probability DOWN: {(1 - prob_up) * 100:5.1f}%")
        conf = distribution["confidence_score"]
        logger.info(f"Confidence Score:   {conf:.2f}")
        current_vix = distribution["metadata"]["current_vix"]
        expected_vix = current_vix * (1 + point / 100)
        logger.info(f"\nInterpretation:")
        logger.info(f"   Current VIX: {current_vix:.2f}")
        logger.info(
            f"   Expected VIX in {TARGET_CONFIG['horizon_days']} days: {expected_vix:.2f}"
        )
        logger.info(
            f"   80% confidence range: [{current_vix * (1 + quantiles['q10'] / 100):.2f}, {current_vix * (1 + quantiles['q90'] / 100):.2f}]"
        )

    def _store_prediction(self, distribution, observation):
        prediction_id = str(uuid.uuid4())

        features_used = {
            feat: float(observation[feat]) for feat in self.forecaster.feature_names
        }

        record = {
            "prediction_id": prediction_id,
            "timestamp": pd.Timestamp.now(),
            "forecast_date": pd.Timestamp(distribution["metadata"]["forecast_date"]),
            "horizon": TARGET_CONFIG["horizon_days"],
            "calendar_cohort": distribution["cohort"],
            "cohort_weight": distribution["metadata"]["cohort_weight"],
            "point_estimate": distribution["point_estimate"],
            "q10": distribution["quantiles"]["q10"],
            "q25": distribution["quantiles"]["q25"],
            "q50": distribution["quantiles"]["q50"],
            "q75": distribution["quantiles"]["q75"],
            "q90": distribution["quantiles"]["q90"],
            "direction_probability": distribution["direction_probability"],
            "confidence_score": distribution["confidence_score"],
            "feature_quality": distribution["metadata"]["feature_quality"],
            "num_features_used": distribution["metadata"]["features_used"],
            "current_vix": distribution["metadata"]["current_vix"],
            "features_used": json.dumps(features_used),
            "model_version": self._get_model_version(),
        }

        self.prediction_db.store_prediction(record)
        return prediction_id

    def _get_model_version(self):
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
            return f"git-{git_hash}"
        except:
            return f"v{pd.Timestamp.now().strftime('%Y%m%d')}"

    def generate_forecast(self, date=None, store_prediction=True):
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PROBABILISTIC FORECAST")
        logger.info("=" * 80)
        df = self._get_features()

        if date is None:
            date = df.index[-1]
            logger.info(f"📅 Using latest date: {date.strftime('%Y-%m-%d')}")
        else:
            date = pd.Timestamp(date)
            logger.info(f"📅 Forecast date: {date.strftime('%Y-%m-%d')}")

        if date not in df.index:
            raise ValueError(f"Date {date} not in feature data")

        observation = df.loc[date]
        logger.info("🔍 Checking data quality...")
        feature_dict = observation.to_dict()
        quality_score = self.validator.compute_feature_quality(feature_dict, date)
        usable, quality_msg = self.validator.check_quality_threshold(quality_score)

        logger.info(f"   Quality Score: {quality_score:.2f}")
        logger.info(f"   Status: {quality_msg}")

        if not usable:
            report = self.validator.get_quality_report(feature_dict, date)
            logger.error("❌ Data quality insufficient:")
            for issue in report["issues"]:
                logger.error(f"   • {issue}")
            raise ValueError(f"Cannot forecast: {quality_msg}")

        cohort = observation.get("calendar_cohort", "mid_cycle")
        cohort_weight = observation.get("cohort_weight", 1.0)
        logger.info(f"📅 Calendar Cohort: {cohort} (weight: {cohort_weight:.2f})")

        if cohort not in self.forecaster.models:
            logger.warning(f"⚠️  Cohort {cohort} not trained, falling back to mid_cycle")
            cohort = "mid_cycle"

            if cohort not in self.forecaster.models:
                raise ValueError("No trained models available. Run training first.")

        logger.info("🎯 Preparing features for prediction...")

        feature_values = observation[self.forecaster.feature_names]
        feature_array = pd.to_numeric(feature_values, errors="coerce").values

        X_df = pd.DataFrame(
            feature_array.reshape(1, -1),
            columns=self.forecaster.feature_names,
            dtype=np.float64,
        )
        X_df = X_df.fillna(0.0)

        non_numeric = X_df.select_dtypes(include=["object"]).columns.tolist()
        if non_numeric:
            logger.error(f"❌ Non-numeric columns detected: {non_numeric}")
            raise ValueError(
                f"Feature DataFrame contains {len(non_numeric)} object columns"
            )

        logger.info(
            f"✅ Features prepared: shape={X_df.shape}, dtype={X_df.dtypes.unique()[0]}"
        )

        logger.info("🎯 Generating probabilistic forecast...")

        try:
            distribution = self.forecaster.predict(X_df, cohort)
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

        if self.calibrator:
            distribution = self.calibrator.calibrate(distribution)
            logger.info("🎯 Applied forecast calibration")

        distribution["confidence_score"] *= 2 - cohort_weight
        distribution["confidence_score"] = np.clip(
            distribution["confidence_score"], 0, 1
        )

        forecast_date = date + pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
        distribution["metadata"] = {
            "observation_date": date.strftime("%Y-%m-%d"),
            "forecast_date": forecast_date.strftime("%Y-%m-%d"),
            "horizon_days": TARGET_CONFIG["horizon_days"],
            "feature_quality": float(quality_score),
            "cohort_weight": float(cohort_weight),
            "current_vix": float(observation["vix"]),
            "features_used": len(self.forecaster.feature_names),
        }

        self._log_forecast_summary(distribution)

        if store_prediction:
            prediction_id = self._store_prediction(distribution, observation)
            distribution["prediction_id"] = prediction_id
            logger.info(f"💾 Stored prediction: {prediction_id}")

        self.last_forecast = distribution
        self.forecast_history.append({"date": date, "distribution": distribution})

        logger.info("=" * 80)
        logger.info("✅ FORECAST COMPLETE")
        logger.info("=" * 80)

        return distribution

    def run(self, date=None):
        logger.warning("⚠️ run() is deprecated, use generate_forecast()")
        return self.generate_forecast(date)

    def train(
        self,
        years: int = TRAINING_YEARS,
        real_time_vix: bool = True,
        verbose: bool = False,
        enable_anomaly: bool = False,
    ):
        print(
            f"\n{'=' * 80}\nINTEGRATED SYSTEM V5 - PROBABILISTIC FORECASTING\n{'=' * 80}"
        )
        print(
            f"Config: {years}y training | Real-time VIX: {real_time_vix} | Anomaly: {enable_anomaly}"
        )

        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="pre-training")

        print("\n[1/2] Building features...")
        feature_data = self.feature_engine.build_complete_features(years=years)
        features = feature_data["features"]
        vix = feature_data["vix"]
        spx = feature_data["spx"]

        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-features")

        self.orchestrator.features = features
        self.orchestrator.vix_ml = vix
        self.orchestrator.spx_ml = spx

        if enable_anomaly:
            print("[2/2] Training anomaly system...")
            vix_history_all = self.orchestrator.fetcher.fetch_yahoo(
                "^VIX",
                "1990-01-02",
                datetime.now().strftime("%Y-%m-%d"),
            )["Close"].squeeze()

            self.orchestrator.train(
                features=features,
                vix=vix,
                spx=spx,
                vix_history_all=vix_history_all,
                verbose=verbose,
            )

            if self.memory_monitoring_enabled:
                self._log_memory_stats(context="post-training")

            if real_time_vix:
                try:
                    live_vix = self.orchestrator.fetcher.fetch_price("^VIX")
                    if live_vix:
                        self.orchestrator.vix_ml.iloc[-1] = live_vix
                        self.orchestrator.features.iloc[
                            -1, self.orchestrator.features.columns.get_loc("vix")
                        ] = live_vix
                        if verbose:
                            print(f"✅ Updated live VIX: {live_vix:.2f}")
                except Exception as e:
                    warnings.warn(f"Live VIX fetch failed: {e}")
        else:
            print("[2/2] Anomaly training skipped (enable_anomaly=False)")

        self.trained = True
        print(f"\n{'=' * 80}\n✅ TRAINING COMPLETE\n{'=' * 80}")

    def run_feature_selection(
        self,
        horizons: list = [5],
        min_stability: float = 0.3,
        max_correlation: float = 0.95,
        preserve_forward_indicators: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Run XGBoost feature selection."""
        if not self.trained:
            raise ValueError("Must train system first")

        print(f"\n{'=' * 80}\nFEATURE SELECTION\n{'=' * 80}")

        selection_results = run_intelligent_feature_selection(
            self,
            horizons=horizons,
            min_stability=min_stability,
            max_correlation=max_correlation,
            preserve_forward_indicators=preserve_forward_indicators,
            verbose=verbose,
        )

        selected_features = selection_results["selected_features"]
        print(f"\n✅ Selected {len(selected_features)} features")
        print(f"   Saved to: ./models/selected_features_v2.txt")

        return selection_results

    def get_market_state(self) -> dict:
        if not self.trained:
            raise ValueError("Must train system first")

        anomaly_result = self._get_cached_anomaly_result()
        persistence_stats = self.orchestrator.get_persistence_stats()

        current_vix = float(self.orchestrator.vix_ml.iloc[-1])
        current_regime = self._classify_vix_regime(current_vix)

        ensemble = anomaly_result["ensemble"]
        ensemble_score = ensemble["score"]

        level, p_value, confidence = (
            self.orchestrator.anomaly_detector.classify_anomaly(
                ensemble_score, method="statistical"
            )
        )

        severity_messages = {
            "CRITICAL": f"Extreme anomaly ({ensemble_score:.1%}) - Markets in unprecedented configuration",
            "HIGH": f"Significant anomaly ({ensemble_score:.1%}) - Notable market stress detected",
            "MODERATE": f"Moderate anomaly ({ensemble_score:.1%}) - Elevated market uncertainty",
            "NORMAL": f"Normal conditions ({ensemble_score:.1%}) - Market within typical ranges",
        }

        ensemble["severity"] = level
        ensemble["severity_message"] = severity_messages[level]
        if p_value is not None:
            ensemble["p_value"] = float(p_value)
            ensemble["confidence"] = float(confidence)

        top_anomalies = self._get_top_anomalies_list(anomaly_result)

        regime_stats = self.orchestrator.regime_stats["regimes"][current_regime["id"]]
        persistence_prob = regime_stats["transitions_5d"]["persistence"]["probability"]
        persistence_prob_clamped = max(0.01, min(0.99, persistence_prob))
        expected_duration = 1.0 / (1.0 - persistence_prob_clamped)

        return {
            "timestamp": datetime.now().isoformat(),
            "vix": {
                "current": current_vix,
                "regime": current_regime,
                "regime_stats": regime_stats,
            },
            "anomaly_analysis": {
                "ensemble": ensemble,
                "top_anomalies": top_anomalies,
                "persistence": persistence_stats,
                "domain_anomalies": anomaly_result.get("domain_anomalies", {}),
                "random_anomalies": anomaly_result.get("random_anomalies", {}),
            },
            "regime_forecast": {
                "persistence_probability": float(persistence_prob),
                "expected_duration_days": float(expected_duration),
                "transition_risk": "elevated" if ensemble_score > 0.7 else "normal",
            },
            "spx_state": self._get_spx_feature_state(),
            "system_health": {
                "trained": self.trained,
                "feature_count": len(self.orchestrator.features.columns),
                "detectors_active": anomaly_result.get("data_quality", {}).get(
                    "active_detectors", 0
                ),
                "last_update": self.orchestrator.features.index[-1].isoformat(),
            },
        }

    def _get_cached_anomaly_result(self) -> dict:
        now = datetime.now()
        if self._cached_anomaly_result is None or (
            self._cache_timestamp and (now - self._cache_timestamp).seconds > 60
        ):
            self._cached_anomaly_result = self.orchestrator.detect_current(
                verbose=False
            )
            self._cache_timestamp = now
        return self._cached_anomaly_result

    def _classify_vix_regime(self, vix: float) -> dict:
        """Classify current VIX regime."""
        for i, boundary in enumerate(REGIME_BOUNDARIES[1:]):
            if vix < boundary:
                return {
                    "id": i,
                    "name": REGIME_NAMES[i],
                    "range": [
                        float(REGIME_BOUNDARIES[i]),
                        float(REGIME_BOUNDARIES[i + 1]),
                    ],
                }
        return {
            "id": 3,
            "name": REGIME_NAMES[3],
            "range": [float(REGIME_BOUNDARIES[3]), 100.0],
        }

    def _get_spx_feature_state(self) -> dict:
        f = self.orchestrator.features.iloc[-1]
        return {
            "price_action": {
                "vs_ma50": float(f.get("spx_vs_ma50", 0)),
                "vs_ma200": float(f.get("spx_vs_ma200", 0)),
                "momentum_10d": float(f.get("spx_momentum_z_10d", 0)),
                "realized_vol_21d": float(f.get("spx_realized_vol_21d", 15)),
            },
            "vix_relationship": {
                "corr_21d": float(f.get("spx_vix_corr_21d", -0.7)),
                "vix_rv_ratio_21d": float(f.get("vix_rv_ratio_21d", 1.0)),
            },
        }

    def _get_top_anomalies_list(self, anomaly_results: dict) -> list:
        domain_scores = [
            {"name": name, "score": data["score"]}
            for name, data in anomaly_results.get("domain_anomalies", {}).items()
        ]
        return sorted(domain_scores, key=lambda x: x["score"], reverse=True)[:5]

    def print_anomaly_summary(self):
        if not self.trained:
            raise ValueError("Run train() first")

        state = self.get_market_state()
        anomaly = state["anomaly_analysis"]
        ensemble = anomaly["ensemble"]
        persistence = anomaly["persistence"]

        print(f"\n{'=' * 80}\n15-DIMENSIONAL ANOMALY SUMMARY\n{'=' * 80}")
        print(f"\n🎯 {ensemble['severity']}: {ensemble['score']:.1%}")
        print(f"   {ensemble['severity_message']}")
        print(
            f"\n⏱️ PERSISTENCE: {persistence['current_streak']}d streak | "
            f"Mean: {persistence['mean_duration']:.1f}d | Rate: {persistence['anomaly_rate']:.1%}"
        )
        print(f"\n🔍 TOP 3:")
        for i, anom in enumerate(anomaly["top_anomalies"][:3], 1):
            level = (
                "EXTREME"
                if anom["score"] > 0.9
                else ("HIGH" if anom["score"] > 0.75 else "MODERATE")
            )
            print(
                f"   {i}. {anom['name'].replace('_', ' ').title()}: {anom['score']:.0%} ({level})"
            )

        if self.memory_monitoring_enabled:
            mem_report = self.get_memory_report()
            if "error" not in mem_report:
                status_emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "NORMAL": "✅"}[
                    mem_report["status"]
                ]
                print(
                    f"\n📊 MEMORY: {status_emoji} {mem_report['status']} | "
                    f"{mem_report['current_mb']:.1f}MB (+{mem_report['growth_mb']:.1f}MB)"
                )

        print(f"\n{'=' * 80}")

    def train_probabilistic_models(
        self, years: int = TRAINING_YEARS, save_dir: str = "models"
    ):
        logger.info("=" * 80)
        logger.info("TRAINING PROBABILISTIC FORECASTING MODELS")
        logger.info("=" * 80)

        # Build features
        logger.info("\n[1/2] Building features...")
        feature_data = self.feature_engine.build_complete_features(years=years)
        df = feature_data["features"]

        logger.info(f"✅ Features: {df.shape}")
        logger.info(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")

        # Validate required columns
        required = ["vix", "calendar_cohort", "cohort_weight", "feature_quality"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Train models
        logger.info("\n[2/2] Training models per cohort...")
        self.forecaster = ProbabilisticVIXForecaster()
        metrics = self.forecaster.train(df, save_dir=save_dir)

        logger.info("\n" + "=" * 80)
        logger.info("✅ PROBABILISTIC TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Cohorts trained: {len(self.forecaster.models)}")
        logger.info(f"Models saved to: {Path(save_dir).absolute()}")

        # Reload models to verify
        self._load_models()

        return metrics

    def generate_forecast_batch(
        self, start_date: str, end_date: str, frequency: str = "daily"
    ):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BATCH FORECASTING: {start_date} to {end_date}")
        logger.info(f"{'=' * 80}")

        # Build features ONCE for entire batch
        logger.info("🔧 Building features for batch...")
        df = self._get_features(force_refresh=True)  # Force fresh build

        # Filter to date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        date_range = df[(df.index >= start) & (df.index <= end)].index

        logger.info(f"📅 Forecasting {len(date_range)} dates")
        logger.info(f"   Range: {date_range[0].date()} to {date_range[-1].date()}")

        forecasts = []
        commit_interval = 50  # Commit every 50 forecasts

        for i, date in enumerate(date_range):
            try:
                # Extract observation from EXISTING df (no rebuild!)
                observation = df.loc[date]

                # Quick quality check
                feature_dict = observation.to_dict()
                quality_score = self.validator.compute_feature_quality(
                    feature_dict, date
                )
                usable, _ = self.validator.check_quality_threshold(quality_score)

                if not usable:
                    logger.warning(
                        f"   Skipped {date.date()}: low quality ({quality_score:.2f})"
                    )
                    continue

                # Get cohort
                cohort = observation.get("calendar_cohort", "mid_cycle")
                cohort_weight = observation.get("cohort_weight", 1.0)

                if cohort not in self.forecaster.models:
                    cohort = "mid_cycle"

                # Prepare features
                feature_values = observation[self.forecaster.feature_names]
                feature_array = pd.to_numeric(feature_values, errors="coerce").values
                X_df = pd.DataFrame(
                    feature_array.reshape(1, -1),
                    columns=self.forecaster.feature_names,
                    dtype=np.float64,
                ).fillna(0.0)

                # Generate forecast
                distribution = self.forecaster.predict(X_df, cohort)

                # Apply calibration
                if self.calibrator:
                    distribution = self.calibrator.calibrate(distribution)

                distribution["confidence_score"] *= 2 - cohort_weight
                distribution["confidence_score"] = np.clip(
                    distribution["confidence_score"], 0, 1
                )

                # Add metadata
                forecast_date = date + pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
                distribution["metadata"] = {
                    "observation_date": date.strftime("%Y-%m-%d"),
                    "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                    "horizon_days": TARGET_CONFIG["horizon_days"],
                    "feature_quality": float(quality_score),
                    "cohort_weight": float(cohort_weight),
                    "current_vix": float(observation["vix"]),
                    "features_used": len(self.forecaster.feature_names),
                }

                # Store
                prediction_id = self._store_prediction(distribution, observation)
                distribution["prediction_id"] = prediction_id

                forecasts.append(distribution)

                # Progress + periodic commits
                if (i + 1) % commit_interval == 0:
                    logger.info(f"   Progress: {i + 1}/{len(date_range)} forecasts")
                    self.prediction_db.commit()
                    logger.info(f"   ✅ Committed batch of {commit_interval}")

            except Exception as e:
                logger.warning(f"   Failed {date.date()}: {e}")
                continue

        # Final commit
        if len(forecasts) % commit_interval != 0:
            self.prediction_db.commit()
            logger.info(f"   ✅ Committed final batch")

        logger.info("=" * 80)
        logger.info(f"✅ Generated {len(forecasts)} forecasts")
        logger.info("=" * 80)

        # Verify no uncommitted writes
        status = self.prediction_db.get_commit_status()
        if status["pending_writes"] > 0:
            logger.error(f"🚨 WARNING: {status['pending_writes']} uncommitted writes!")
            raise RuntimeError(
                f"Lost {status['pending_writes']} forecasts - commit failed!"
            )
        else:
            logger.info("✅ All forecasts committed to database")

        return forecasts

    def _get_features(self, force_refresh=False) -> pd.DataFrame:
        now = pd.Timestamp.now()
        cache_key = (now.year, now.month, now.day, now.hour)

        # Check if cache is valid
        if (
            not force_refresh
            and self._feature_cache is not None
            and self._feature_cache_date == today
        ):
            logger.info("📦 Using cached features (already built today)")
            return self._feature_cache

        # Build fresh features
        logger.info("🔧 Building features...")
        feature_data = self.feature_engine.build_complete_features(years=TRAINING_YEARS)
        df = feature_data["features"]

        # Force numeric dtypes (safety check)
        metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        for col in df.columns:
            if col not in metadata_cols and df[col].dtype == object:
                logger.warning(f"⚠️  Converting object column to numeric: {col}")
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Verify no object columns remain
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        unexpected = [c for c in object_cols if c not in metadata_cols]
        if unexpected:
            logger.error(f"❌ Unexpected object columns: {unexpected[:5]}")
            raise ValueError(
                f"DataFrame contains {len(unexpected)} non-numeric columns"
            )

        # Cache it
        self._feature_cache = df
        self._feature_cache_date = cache_key

        logger.info(f"✅ Features cached: {df.shape}, dtypes OK")
        return df


def main():
    parser = argparse.ArgumentParser(description="Integrated Market Analysis System V5")

    parser.add_argument(
        "--mode",
        choices=["forecast", "complete", "batch", "anomaly"],
        default="forecast",
        help="Execution mode",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for batch forecasting (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for batch forecasting (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    if not ENABLE_TRAINING:
        print(f"\n{'=' * 80}")
        print("⚠️ TRAINING DISABLED (config.ENABLE_TRAINING = False)")
        print("⚠️ Set ENABLE_TRAINING = True in config.py")
        print(f"{'=' * 80}\n")
        return

    system = IntegratedSystem()

    if args.mode == "complete":
        logger.info("🎯 MODE: Complete Workflow - Everything in one command")

        try:
            # This does EVERYTHING: validate 2023-2024, backfill 2025, ready for production
            from config import CALIBRATION_PERIOD, VALIDATION_PERIOD

            cal_start, cal_end = CALIBRATION_PERIOD
            val_start, val_end = VALIDATION_PERIOD

            logger.info("=" * 80)
            logger.info("COMPLETE WORKFLOW")
            logger.info("=" * 80)
            logger.info("This will:")
            logger.info(
                f"  1. Generate uncalibrated forecasts ({cal_start} to {cal_end})"
            )
            logger.info(f"  2. Backfill actuals for calibration period")
            logger.info(f"  3. Train calibrator on {cal_start[:4]} data")
            logger.info(f"  4. Regenerate {val_start[:4]} WITH calibration")
            logger.info(f"  5. Backfill actuals for validation period")
            logger.info(f"  6. Generate 2025 forecasts (Jan 1 - today)")
            logger.info(f"  7. Backfill actuals for 2025")
            logger.info(f"  8. Run validation diagnostics")
            logger.info("=" * 80)

            # STEP 1: Generate uncalibrated forecasts for CALIBRATION period only
            logger.info(
                f"\n[1/8] Generating uncalibrated forecasts ({cal_start} to {cal_end})..."
            )
            original_calibrator = system.calibrator
            system.calibrator = None  # Disable calibration
            system.generate_forecast_batch(cal_start, cal_end)

            # STEP 2: Backfill actuals for calibration period
            logger.info(f"\n[2/8] Backfilling actuals for {cal_start[:4]}...")
            system.prediction_db.backfill_actuals()

            # STEP 3: Train calibrator on calibration period ONLY
            logger.info(f"\n[3/8] Training calibrator on {cal_start[:4]} data...")
            from core.forecast_calibrator import ForecastCalibrator

            calibrator = ForecastCalibrator()
            success = calibrator.fit_from_database(
                db_path=system.prediction_db.db_path,
                min_samples=50,
                start_date=cal_start,
                end_date=cal_end,
            )
            if not success:
                logger.error("❌ Calibration failed - insufficient data")
                system.calibrator = original_calibrator
                return
            calibrator.save()

            # STEP 4: Delete validation period forecasts (if any exist)
            logger.info(f"\n[4/8] Clearing old {val_start[:4]} forecasts...")
            import sqlite3

            conn = sqlite3.connect(system.prediction_db.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM forecasts WHERE forecast_date BETWEEN ? AND ?",
                (val_start, val_end),
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(f"   Deleted {deleted} old forecasts")

            # STEP 5: Generate validation period forecasts WITH calibration
            logger.info(
                f"\n[5/8] Generating {val_start[:4]} forecasts WITH calibration..."
            )
            system.calibrator = ForecastCalibrator.load()
            system.generate_forecast_batch(val_start, val_end)

            # STEP 6: Backfill actuals for validation period
            logger.info(f"\n[6/8] Backfilling actuals for {val_start[:4]}...")
            system.prediction_db.backfill_actuals()

            # STEP 7: Generate 2025 forecasts WITH calibration
            logger.info(f"\n[7/8] Generating 2025 forecasts...")
            today = datetime.now().strftime("%Y-%m-%d")
            system.generate_forecast_batch("2025-01-01", today)

            # STEP 8: Backfill actuals for 2025 and run validation
            logger.info(f"\n[8/8] Backfilling 2025 actuals and running validation...")
            system.prediction_db.backfill_actuals()

            from diagnostics.walk_forward_validation import EnhancedWalkForwardValidator

            validator = EnhancedWalkForwardValidator(
                db_path=system.prediction_db.db_path
            )
            validator.generate_diagnostic_report()

            logger.info("\n" + "=" * 80)
            logger.info("✅ COMPLETE WORKFLOW FINISHED")
            logger.info("=" * 80)
            logger.info("\n📊 Results:")
            logger.info("  • diagnostics/walk_forward_metrics.json")
            logger.info("  • diagnostics/*.png")
            logger.info(f"\n📈 Calibrator trained on: {cal_start} to {cal_end}")
            logger.info(f"📈 Validation period: {val_start} to {val_end}")
            logger.info(f"📈 2025 forecasts: up to {today}")
            logger.info("\n🚀 System ready for production!")
            logger.info(
                "  Run daily: python integrated_system_production.py --mode forecast"
            )

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}", exc_info=True)
            return

    elif args.mode == "forecast":
        # Generate single probabilistic forecast
        try:
            distribution = system.generate_forecast()

            print(f"\n{'=' * 80}")
            print("PROBABILISTIC FORECAST GENERATED")
            print(f"{'=' * 80}")
            print(f"\nPoint Estimate: {distribution['point_estimate']:+.1f}%")
            print(f"Confidence: {distribution['confidence_score']:.2f}")
            print(f"Cohort: {distribution['cohort']}")
            print(f"\nFull distribution saved to database")
            print(f"{'=' * 80}\n")

        except ValueError as e:
            print(f"\n❌ Forecast failed: {e}\n")

    elif args.mode == "batch":
        # Batch backtesting
        if not args.start_date or not args.end_date:
            print("❌ Error: --start-date and --end-date required for batch mode")
            return

        forecasts = system.generate_forecast_batch(
            args.start_date, args.end_date, frequency="daily"
        )

        print(f"\n{'=' * 80}")
        print(f"BATCH FORECASTING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Generated {len(forecasts)} forecasts")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"{'=' * 80}\n")

    elif args.mode == "anomaly":
        system.train(
            years=TRAINING_YEARS, real_time_vix=True, verbose=False, enable_anomaly=True
        )
        system.print_anomaly_summary()

        anomaly_result = system._get_cached_anomaly_result()
        if anomaly_result and system.orchestrator.anomaly_detector:
            from export.unified_exporter import UnifiedExporter

            persistence_stats = system.orchestrator.get_persistence_stats()
            exporter = UnifiedExporter(output_dir="./json_data")

            exporter.export_live_state(
                orchestrator=system.orchestrator,
                anomaly_result=anomaly_result,
                spx=system.orchestrator.spx_ml,
                persistence_stats=persistence_stats,
            )

            exporter.export_historical_context(
                orchestrator=system.orchestrator,
                spx=system.orchestrator.spx_ml,
                historical_scores=system.orchestrator.historical_ensemble_scores,
            )

            system.orchestrator.save_state("./json_data/model_cache.pkl")

            print("\n✅ Exported unified dashboard files:")
            print("   • live_state.json    (15 KB, updates every refresh)")
            print("   • historical.json    (300 KB, static)")
            print("   • model_cache.pkl    (15 MB, static)")

    if system.memory_monitoring_enabled:
        mem_report = system.get_memory_report()
        if "error" not in mem_report:
            print(f"\n{'=' * 80}\nMEMORY REPORT\n{'=' * 80}")
            print(f"Status: {mem_report['status']}")
            print(
                f"Current: {mem_report['current_mb']:.1f}MB | Growth: {mem_report['growth_mb']:+.1f}MB"
            )

    print(f"\n{'=' * 80}\nANALYSIS COMPLETE\n{'=' * 80}")


if __name__ == "__main__":
    main()
