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
    """
    Orchestrates anomaly detection workflow (preserved for backward compatibility).
    1. Manages VIX/SPX history
    2. Maintains regime statistics
    3. Coordinates anomaly detector
    4. Handles state persistence
    """

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
        """Train anomaly detection system on historical features."""

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
        """Compute comprehensive regime statistics from VIX history."""
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
    """
    Main system integrating probabilistic forecasting with anomaly detection.

    Components:
      - Feature Engine: Generate 232 features with calendar cohorts
      - Probabilistic Forecaster: Multi-output distribution model
      - Prediction Database: Store forecasts for backtesting
      - Anomaly Detector: Identify market regime anomalies (parallel)
      - Temporal Validator: Check data quality

    Example:
        >>> system = IntegratedSystem()
        >>> distribution = system.generate_forecast()
        >>> print(distribution['point_estimate'])  # 8.5% expected VIX change
    """

    def __init__(self, models_dir="models"):
        """
        Initialize integrated system.

        Args:
            models_dir: Directory containing trained models
        """
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
        logger.info("\n📊 FORECAST SUMMARY")
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

        # Regimes
        regimes = distribution["regime_probabilities"]
        logger.info(f"Regime Probabilities:")
        for regime, prob in regimes.items():
            logger.info(f"   {regime.capitalize():10s}: {prob * 100:5.1f}%")

        # Confidence
        conf = distribution["confidence_score"]
        logger.info(f"Confidence Score:   {conf:.2f}")

        # Interpretation
        current_vix = distribution["metadata"]["current_vix"]
        expected_vix = current_vix * (1 + point / 100)
        logger.info(f"\nInterpretation:")
        logger.info(f"   Current VIX: {current_vix:.2f}")
        logger.info(
            f"   Expected VIX in {TARGET_CONFIG['horizon_days']} days: {expected_vix:.2f}"
        )
        logger.info(
            f"   90% confidence range: [{current_vix * (1 + quantiles['q10'] / 100):.2f}, {current_vix * (1 + quantiles['q90'] / 100):.2f}]"
        )

    def _store_prediction(self, distribution, observation):
        """
        Store prediction in database for backtesting.

        Args:
            distribution: Forecast distribution object
            observation: Original feature row

        Returns:
            str: prediction_id (UUID)
        """
        prediction_id = str(uuid.uuid4())

        # Extract features used (for provenance)
        features_used = {
            feat: float(observation[feat]) for feat in self.forecaster.feature_names
        }

        # Build database record
        record = {
            "prediction_id": prediction_id,
            "timestamp": pd.Timestamp.now(),
            "forecast_date": pd.Timestamp(distribution["metadata"]["forecast_date"]),
            "horizon": TARGET_CONFIG["horizon_days"],
            # Context
            "calendar_cohort": distribution["cohort"],
            "cohort_weight": distribution["metadata"]["cohort_weight"],
            # Predictions
            "point_estimate": distribution["point_estimate"],
            "q10": distribution["quantiles"]["q10"],
            "q25": distribution["quantiles"]["q25"],
            "q50": distribution["quantiles"]["q50"],
            "q75": distribution["quantiles"]["q75"],
            "q90": distribution["quantiles"]["q90"],
            "prob_low": distribution["regime_probabilities"]["low"],
            "prob_normal": distribution["regime_probabilities"]["normal"],
            "prob_elevated": distribution["regime_probabilities"]["elevated"],
            "prob_crisis": distribution["regime_probabilities"]["crisis"],
            "confidence_score": distribution["confidence_score"],
            # Metadata
            "feature_quality": distribution["metadata"]["feature_quality"],
            "num_features_used": distribution["metadata"]["features_used"],
            "current_vix": distribution["metadata"]["current_vix"],
            # Provenance
            "features_used": json.dumps(features_used),
            "model_version": self._get_model_version(),
        }

        # Store in database
        self.prediction_db.store_prediction(record)

        return prediction_id

    def _get_model_version(self):
        """Get current model version (git hash or timestamp)."""
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
        """
        Generate probabilistic VIX forecast for given date.

        **FIXED VERSION** - Handles object dtype bug in feature extraction.
        """
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PROBABILISTIC FORECAST")
        logger.info("=" * 80)

        # 1. Build features
        logger.info("🔧 Building features...")
        feature_data = self.feature_engine.build_complete_features(years=15)
        df = feature_data["features"]

        # **FIX 1: Force numeric dtypes immediately after loading**
        metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        for col in df.columns:
            if col not in metadata_cols and df[col].dtype == object:
                logger.warning(f"⚠️  Converting object column to numeric: {col}")
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Verify no object columns remain (except metadata)
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        unexpected = [c for c in object_cols if c not in metadata_cols]
        if unexpected:
            logger.error(f"❌ Unexpected object columns: {unexpected[:5]}")
            raise ValueError(
                f"DataFrame contains {len(unexpected)} non-numeric columns"
            )

        logger.info(f"✅ Features validated: {df.shape}, dtypes OK")

        # 2. Select observation date
        if date is None:
            date = df.index[-1]
            logger.info(f"📅 Using latest date: {date.strftime('%Y-%m-%d')}")
        else:
            date = pd.Timestamp(date)
            logger.info(f"📅 Forecast date: {date.strftime('%Y-%m-%d')}")

        if date not in df.index:
            raise ValueError(f"Date {date} not in feature data")

        observation = df.loc[date]

        # 3. Check data quality
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

        # 4. Get calendar cohort
        cohort = observation.get("calendar_cohort", "mid_cycle")
        cohort_weight = observation.get("cohort_weight", 1.0)
        logger.info(f"📅 Calendar Cohort: {cohort} (weight: {cohort_weight:.2f})")

        # 5. Check if cohort model exists
        if cohort not in self.forecaster.models:
            logger.warning(f"⚠️  Cohort {cohort} not trained, falling back to mid_cycle")
            cohort = "mid_cycle"

            if cohort not in self.forecaster.models:
                raise ValueError("No trained models available. Run training first.")

        # 6. Prepare features for prediction
        logger.info("🎯 Preparing features for prediction...")

        # **FIX 2: Robust feature extraction with explicit dtype handling**
        feature_values = observation[self.forecaster.feature_names]

        # Convert to numeric array (this handles any lingering string values)
        feature_array = pd.to_numeric(feature_values, errors="coerce").values

        # Create DataFrame with explicit float64 dtype
        X_df = pd.DataFrame(
            feature_array.reshape(1, -1),
            columns=self.forecaster.feature_names,
            dtype=np.float64,
        )

        # Fill any NaNs with 0 (consistent with training)
        X_df = X_df.fillna(0.0)

        # **FIX 3: Validation before prediction**
        non_numeric = X_df.select_dtypes(include=["object"]).columns.tolist()
        if non_numeric:
            logger.error(f"❌ Non-numeric columns detected: {non_numeric}")
            logger.error(f"   Sample values: {X_df[non_numeric].iloc[0].to_dict()}")
            raise ValueError(
                f"Feature DataFrame contains {len(non_numeric)} object columns"
            )

        logger.info(
            f"✅ Features prepared: shape={X_df.shape}, dtype={X_df.dtypes.unique()[0]}"
        )

        # 7. Generate distribution
        logger.info("🎯 Generating probabilistic forecast...")

        try:
            distribution = self.forecaster.predict(X_df, cohort)
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            logger.error(f"   X_df dtypes: {X_df.dtypes.value_counts().to_dict()}")
            logger.error(f"   X_df shape: {X_df.shape}")
            logger.error(f"   Sample values: {X_df.iloc[0, :5].to_dict()}")
            raise

        # 7.5 Apply calibration if available
        if self.calibrator:
            distribution = self.calibrator.calibrate(distribution)
            logger.info("🎯 Applied forecast calibration")

        # Adjust confidence by cohort weight
        distribution["confidence_score"] *= 2 - cohort_weight
        distribution["confidence_score"] = np.clip(
            distribution["confidence_score"], 0, 1
        )

        # 8. Add metadata
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

        # 9. Log forecast summary
        self._log_forecast_summary(distribution)

        # 10. Store in database
        if store_prediction:
            prediction_id = self._store_prediction(distribution, observation)
            distribution["prediction_id"] = prediction_id
            logger.info(f"💾 Stored prediction: {prediction_id}")

        # 11. Update state
        self.last_forecast = distribution
        self.forecast_history.append({"date": date, "distribution": distribution})

        logger.info("=" * 80)
        logger.info("✅ FORECAST COMPLETE")
        logger.info("=" * 80)

        return distribution

    def run(self, date=None):
        """
        Legacy method - redirects to generate_forecast().

        Kept for backward compatibility with existing scripts.
        """
        logger.warning("⚠️ run() is deprecated, use generate_forecast()")
        return self.generate_forecast(date)

    def train(
        self,
        years: int = TRAINING_YEARS,
        real_time_vix: bool = True,
        verbose: bool = False,
        enable_anomaly: bool = False,
    ):
        """Train the complete system (backward compatible with anomaly system)."""
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
        """Generate comprehensive market state snapshot (legacy anomaly method)."""
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
        """Get anomaly result with caching."""
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
        """Extract SPX feature state."""
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
        """Get top anomalies sorted by score."""
        domain_scores = [
            {"name": name, "score": data["score"]}
            for name, data in anomaly_results.get("domain_anomalies", {}).items()
        ]
        return sorted(domain_scores, key=lambda x: x["score"], reverse=True)[:5]

    def print_anomaly_summary(self):
        """Print comprehensive anomaly analysis summary."""
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

    def _initialize_memory_baseline(self):
        if not self.memory_monitoring_enabled:
            return
        try:
            gc.collect()
            mem_info = self.process.memory_info()
            self.baseline_memory_mb = mem_info.rss / (1024 * 1024)
            self.memory_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "memory_mb": self.baseline_memory_mb,
                    "type": "baseline",
                }
            )
        except Exception as e:
            warnings.warn(f"Memory baseline failed: {e}")
            self.memory_monitoring_enabled = False

    def _log_memory_stats(self, context: str = "refresh") -> dict:
        if not self.memory_monitoring_enabled:
            return {}
        try:
            mem_info = self.process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            if self.baseline_memory_mb is None:
                self._initialize_memory_baseline()
                return {}
            growth = current_mb - self.baseline_memory_mb
            self.memory_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "memory_mb": current_mb,
                    "type": context,
                }
            )
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-1000:]
            return {"current_mb": current_mb, "growth_mb": growth}
        except Exception as e:
            return {}

    def get_memory_report(self) -> dict:
        if not self.memory_monitoring_enabled:
            return {"error": "psutil not installed"}
        try:
            mem_info = self.process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            growth_mb = (
                current_mb - self.baseline_memory_mb if self.baseline_memory_mb else 0.0
            )
            status = (
                "CRITICAL"
                if growth_mb > 200
                else ("WARNING" if growth_mb > 50 else "NORMAL")
            )
            return {
                "current_mb": float(current_mb),
                "baseline_mb": float(self.baseline_memory_mb)
                if self.baseline_memory_mb
                else None,
                "growth_mb": float(growth_mb),
                "status": status,
            }
        except Exception as e:
            return {"error": str(e)}

    def train_probabilistic_models(
        self, years: int = TRAINING_YEARS, save_dir: str = "models"
    ):
        """
        Train probabilistic forecasting models.

        This is separate from the legacy anomaly training.

        Args:
            years: Training window in years
            save_dir: Where to save trained models

        Returns:
            Dict of training metrics per cohort
        """
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
        """Generate forecasts for date range and store in database."""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BATCH FORECASTING: {start_date} to {end_date}")
        logger.info(f"{'=' * 80}")

        # Build features once
        feature_data = self.feature_engine.build_complete_features(years=15)
        df = feature_data["features"]

        # Filter to date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        date_range = df[(df.index >= start) & (df.index <= end)].index

        forecasts = []
        for date in date_range:
            try:
                # Generate forecast (already stores in DB via generate_forecast)
                distribution = self.generate_forecast(date=date, store_prediction=True)
                forecasts.append(distribution)

                if len(forecasts) % 50 == 0:
                    logger.info(
                        f"   Progress: {len(forecasts)}/{len(date_range)} forecasts"
                    )
            except Exception as e:
                logger.warning(f"   Failed {date.date()}: {e}")
                continue

        logger.info(f"✅ Generated {len(forecasts)} forecasts")
        return forecasts


def main():
    """Main execution function with CLI argument support."""
    parser = argparse.ArgumentParser(description="Integrated Market Analysis System V5")

    parser.add_argument(
        "--mode",
        choices=["forecast", "batch", "anomaly"],  # Only keep working modes
        default="forecast",
        help="Execution mode: single forecast, batch backtest, or anomaly detection",
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

    if args.mode == "train":
        logger.info("🎯 MODE: Train probabilistic models")

        try:
            metrics = system.train_probabilistic_models(
                years=args.years, save_dir="models"
            )

            print(f"\n{'=' * 80}")
            print("✅ TRAINING SUCCESSFUL")
            print(f"{'=' * 80}")
            print(f"Cohorts trained: {len(metrics)}")
            print(f"\nRun forecasting with:")
            print(f"  python integrated_system_production.py --mode forecast")
            print(f"{'=' * 80}\n")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            return

    if args.mode == "forecast":
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
        print(f"\nBackfilling actuals...")

        system.prediction_db.backfill_actuals()

        print(f"\nComputing performance metrics...")
        summary = system.prediction_db.get_performance_summary()

        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 80)

        if "error" in summary:
            print(f"⚠️  {summary['error']}")
            print("\nNo actuals available yet. Predictions need time to mature.")
            print(f"Forecast horizon: {HORIZON} days")
            print("\nTo see metrics:")
            print("  1. Wait for forecasts to mature (5+ days)")
            print("  2. Run: python integrated_system_production.py --mode batch")
            print("  3. Then run: python diagnostics/walk_forward_validation.py")
        else:
            print(f"Predictions evaluated: {summary.get('n_predictions', 0)}")

            if summary.get("n_predictions", 0) > 0:
                if "point_estimate" in summary:
                    print(
                        f"Point Estimate MAE: {summary['point_estimate']['mae']:.2f}%"
                    )
                    print(
                        f"Point Estimate RMSE: {summary['point_estimate']['rmse']:.2f}%"
                    )

                if "quantile_coverage" in summary:
                    print(f"\nQuantile Coverage:")
                    for q, coverage in summary["quantile_coverage"].items():
                        expected = (
                            int(q[1:]) / 100
                        )  # Extract number from 'q10', 'q25', etc.
                        diff = coverage - expected
                        status = "✅" if abs(diff) < 0.10 else "⚠️"
                        print(
                            f"  {status} {q}: {coverage:.1%} (expected {expected:.1%}, diff: {diff:+.1%})"
                        )

                if "regime_brier_score" in summary:
                    brier = summary["regime_brier_score"]
                    if not pd.isna(brier):
                        print(f"\nRegime Classification Brier Score: {brier:.3f}")

                if "confidence_correlation" in summary:
                    corr = summary["confidence_correlation"]
                    if not pd.isna(corr):
                        print(f"Confidence vs Error Correlation: {corr:.3f}")
                        if corr < -0.1:
                            print("  ✅ Confidence scores are predictive of accuracy")
                        else:
                            print("  ⚠️ Confidence scores may need recalibration")

                if "by_cohort" in summary and summary["by_cohort"]:
                    print(f"\nPerformance by Cohort:")
                    for cohort, metrics in summary["by_cohort"].items():
                        print(
                            f"  {cohort}: MAE={metrics['mae']:.2f}% (n={metrics['n']})"
                        )

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
