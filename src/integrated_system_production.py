import argparse,logging,os,sqlite3,sys,warnings
from datetime import datetime,timedelta
from pathlib import Path
from typing import Dict,List,Optional
import numpy as np
import pandas as pd
try:
    import psutil
    PSUTIL_AVAILABLE=True
except ImportError:
    PSUTIL_AVAILABLE=False
    warnings.warn("psutil not available - memory monitoring disabled")
from config import CALIBRATION_PERIOD,ENABLE_TRAINING,FEATURE_CONFIG,FORECASTING_CONFIG,PRODUCTION_START_DATE,TARGET_CONFIG,TRAINING_YEARS,VALIDATION_PERIOD,XGBOOST_CONFIG
from core.anomaly_detector import MultiDimensionalAnomalyDetector
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer as UnifiedFeatureEngine
from core.forecast_calibrator import ForecastCalibrator
from core.prediction_database import PredictionDatabase
from core.temporal_validator import TemporalSafetyValidator as TemporalValidator
from core.xgboost_trainer_v3 import ProbabilisticVIXForecaster
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/integrated_system.log")])
logger=logging.getLogger(__name__)
class IntegratedForecastingSystem:
    def __init__(self,models_dir:str="models",db_path:str="data_cache/predictions.db"):
        logger.info("="*80);logger.info("INTEGRATED PROBABILISTIC FORECASTING SYSTEM V3.1");logger.info("="*80)
        self.models_dir=Path(models_dir);self.db_path=db_path
        self.data_fetcher=UnifiedDataFetcher();self.feature_engine=UnifiedFeatureEngine(data_fetcher=self.data_fetcher);self.forecaster=ProbabilisticVIXForecaster();self.validator=TemporalValidator();self.prediction_db=PredictionDatabase(db_path=db_path)
        self.orchestrator=MultiDimensionalAnomalyDetector()
        self._load_models()
        self.calibrator=ForecastCalibrator.load()
        if self.calibrator:logger.info("📊 Forecast calibrator loaded")
        else:logger.info("ℹ️  No calibrator found - forecasts will not be calibrated")
        self.last_forecast=None;self.forecast_history=[];self.trained=False;self._cached_anomaly_result=None;self._cache_timestamp=None
        self._feature_cache=None;self._feature_cache_date=None
        if PSUTIL_AVAILABLE:
            self.process=psutil.Process(os.getpid());self.baseline_memory_mb=None;self.memory_history=[];self.memory_monitoring_enabled=True
        else:self.memory_monitoring_enabled=False
        logger.info("✅ System initialized")
    def _load_models(self):
        logger.info("📂 Loading trained models...")
        model_files=list(self.models_dir.glob("probabilistic_forecaster_*.pkl"))
        if len(model_files)==0:
            logger.warning("⚠️ No trained models found. Run training first.")
            return
        for model_file in model_files:
            cohort=model_file.stem.replace("probabilistic_forecaster_","")
            self.forecaster.load(cohort,self.models_dir)
            logger.info(f"   ✅ Loaded: {cohort}")
        logger.info(f"📊 Total cohorts loaded: {len(self.forecaster.models)}")
    def generate_forecast(self,date=None,store_prediction=True):
        logger.info("\n"+"="*80);logger.info("GENERATING PROBABILISTIC FORECAST");logger.info("="*80)
        if date is not None:
            target_date=pd.Timestamp(date)
            logger.info(f"📅 Forecast date: {target_date.strftime('%Y-%m-%d')}")
            logger.info("🔧 Building features (historical mode)...")
            feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=target_date.strftime("%Y-%m-%d"))
            df=feature_data["features"]
            metadata_cols=["calendar_cohort","cohort_weight","feature_quality"]
            numeric_cols=[c for c in df.columns if c not in metadata_cols]
            logger.info(f"   Applying nuclear dtype fix to {len(numeric_cols)} columns...")
            for col in numeric_cols:
                if df[col].dtype==object:df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0.0)
                df[col]=df[col].astype(np.float64)
            logger.info("✅ Features validated and cached");logger.info(f"   Shape: {df.shape}");logger.info(f"   All {len(numeric_cols)} numeric cols are float64")
        else:
            df=self._get_features()
            target_date=df.index[-1]
            logger.info(f"📅 Using latest date: {target_date.strftime('%Y-%m-%d')}")
        if target_date not in df.index:
            available_range=f"{df.index[0].date()} to {df.index[-1].date()}"
            raise ValueError(f"Date {target_date.date()} not in feature data. Available range: {available_range}")
        observation=df.loc[target_date]
        logger.info("🔍 Checking data quality...")
        feature_dict=observation.to_dict()
        quality_score=self.validator.compute_feature_quality(feature_dict,target_date)
        usable,quality_msg=self.validator.check_quality_threshold(quality_score)
        logger.info(f"   Quality Score: {quality_score:.2f}");logger.info(f"   Status: {quality_msg}")
        if not usable:
            report=self.validator.get_quality_report(feature_dict,target_date)
            logger.error("❌ Data quality insufficient:")
            for issue in report["issues"]:logger.error(f"   • {issue}")
            raise ValueError(f"Cannot forecast: {quality_msg}")
        cohort=observation.get("calendar_cohort","mid_cycle")
        cohort_weight=observation.get("cohort_weight",1.0)
        logger.info(f"📅 Calendar Cohort: {cohort} (weight: {cohort_weight:.2f})")
        if cohort not in self.forecaster.models:
            logger.warning(f"⚠️  Cohort {cohort} not trained, falling back to mid_cycle");cohort="mid_cycle"
            if cohort not in self.forecaster.models:raise ValueError("No trained models available. Run training first.")
        logger.info("🎯 Preparing features for prediction...")
        feature_values=observation[self.forecaster.feature_names]
        feature_array=pd.to_numeric(feature_values,errors="coerce").values
        X_df=pd.DataFrame(feature_array.reshape(1,-1),columns=self.forecaster.feature_names,dtype=np.float64)
        X_df=X_df.fillna(0.0)
        current_vix=float(observation["vix"])
        non_numeric=X_df.select_dtypes(include=["object"]).columns.tolist()
        if non_numeric:
            logger.error(f"❌ Non-numeric columns detected: {non_numeric}")
            raise ValueError(f"Feature DataFrame contains {len(non_numeric)} object columns")
        logger.info(f"✅ Features prepared: shape={X_df.shape}, dtype={X_df.dtypes.unique()[0]}")
        logger.info("🎯 Generating probabilistic forecast...")
        try:distribution=self.forecaster.predict(X_df,cohort,current_vix)
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

        if self.calibrator:
            raw_median = distribution["median_forecast"]
            current_vix = float(observation["vix"])
            cohort = observation.get("calendar_cohort", None)

            calib_result = self.calibrator.calibrate(
                raw_forecast=raw_median,
                current_vix=current_vix,
                cohort=cohort
            )

            adjustment = calib_result["adjustment"]

            # Calculate adjustment ratio (not absolute adjustment)
            ratio = calib_result["calibrated_forecast"] / raw_median if raw_median != 0 else 1.0

            # Apply proportional scaling to preserve distribution shape
            distribution["median_forecast"] = calib_result["calibrated_forecast"]
            distribution["q50"] = calib_result["calibrated_forecast"]
            distribution["q10"] = distribution["q10"] * ratio
            distribution["q25"] = distribution["q25"] * ratio
            distribution["q75"] = distribution["q75"] * ratio
            distribution["q90"] = distribution["q90"] * ratio
            distribution["point_estimate"] = calib_result["calibrated_forecast"]

            # Ensure monotonicity after scaling
            quantile_values = [
                distribution["q10"],
                distribution["q25"],
                distribution["q50"],
                distribution["q75"],
                distribution["q90"]
            ]
            quantile_values_sorted = sorted(quantile_values)
            distribution["q10"] = quantile_values_sorted[0]
            distribution["q25"] = quantile_values_sorted[1]
            distribution["q50"] = quantile_values_sorted[2]
            distribution["q75"] = quantile_values_sorted[3]
            distribution["q90"] = quantile_values_sorted[4]

            logger.info(f"🎯 Applied calibration: {raw_median:+.2f}% → "
                        f"{calib_result['calibrated_forecast']:+.2f}% (ratio: {ratio:.3f})")
            logger.info(f"   Method: {calib_result['method']}")


        distribution["confidence_score"]*=2-cohort_weight
        distribution["confidence_score"]=np.clip(distribution["confidence_score"],0,1)
        forecast_date=target_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
        distribution["metadata"]={
            "observation_date":target_date.strftime("%Y-%m-%d"),
            "forecast_date":forecast_date.strftime("%Y-%m-%d"),
            "horizon_days":TARGET_CONFIG["horizon_days"],
            "feature_quality":float(quality_score),
            "cohort_weight":float(cohort_weight),
            "current_vix":float(observation["vix"]),
            "features_used":len(self.forecaster.feature_names)
        }
        self._log_forecast_summary(distribution)
        if store_prediction:
            prediction_id=self._store_prediction(distribution,observation,target_date)
            distribution["prediction_id"]=prediction_id
            logger.info(f"💾 Stored prediction: {prediction_id}")
        self.last_forecast=distribution
        self.forecast_history.append({"date":target_date,"distribution":distribution})
        logger.info("="*80);logger.info("✅ FORECAST COMPLETE");logger.info("="*80)
        return distribution
    def _log_forecast_summary(self,distribution:Dict):
        logger.info("\n"+"─"*80);logger.info("FORECAST DISTRIBUTION");logger.info("─"*80)
        logger.info(f"  Median Forecast (q50): {distribution['median_forecast']:+.2f}%")
        logger.info(f"\n  Quantile Distribution:")
        logger.info(f"    10th percentile: {distribution['quantiles']['q10']:+.2f}%");logger.info(f"    25th percentile: {distribution['quantiles']['q25']:+.2f}%");logger.info(f"    50th percentile: {distribution['quantiles']['q50']:+.2f}%");logger.info(f"    75th percentile: {distribution['quantiles']['q75']:+.2f}%");logger.info(f"    90th percentile: {distribution['quantiles']['q90']:+.2f}%")
        spread=distribution["quantiles"]["q90"]-distribution["quantiles"]["q10"]
        logger.info(f"\n  Uncertainty (q90-q10 spread): {spread:.2f}%")
        logger.info(f"\n  Direction:");logger.info(f"    Probability UP:   {distribution['prob_up']:.1%}");logger.info(f"    Probability DOWN: {distribution['prob_down']:.1%}")
        logger.info(f"\n  Confidence Score: {distribution['confidence_score']:.2f}")
        if "metadata"in distribution:
            meta=distribution["metadata"]
            logger.info(f"\n  Context:");logger.info(f"    Current VIX: {meta['current_vix']:.2f}");logger.info(f"    Feature Quality: {meta['feature_quality']:.2f}");logger.info(f"    Cohort Weight: {meta['cohort_weight']:.2f}")
        logger.info("─"*80+"\n")
    def _validate_quantile_ordering(self,distribution:Dict):
        quantiles=["q10","q25","q50","q75","q90"]
        values=[distribution.get(q,0)for q in quantiles]
        if values!=sorted(values):
            logger.warning("⚠️  Quantiles not properly ordered - enforcing monotonicity")
            for i in range(1,len(quantiles)):
                if values[i]<values[i-1]:values[i]=values[i-1]
            for q,v in zip(quantiles,values):distribution[q]=v
    def _display_forecast_summary(self,distribution:Dict,observation:pd.Series):
        print(f"\n{'='*80}");print("FORECAST SUMMARY");print(f"{'='*80}")
        median=distribution["median_forecast"]
        print(f"\nMedian Forecast (50th): {median:+.1f}%");print(f"                 (Primary forecast from quantile regression)")
        print(f"\nDistribution:");print(f"   10th percentile: {distribution['q10']:+.1f}%");print(f"   25th percentile: {distribution['q25']:+.1f}%");print(f"   Median (50th):   {distribution['q50']:+.1f}%");print(f"   75th percentile: {distribution['q75']:+.1f}%");print(f"   90th percentile: {distribution['q90']:+.1f}%")
        current_vix=observation["vix"]
        print(f"\nImplied VIX Levels (from {current_vix:.2f}):");print(f"   10th: {current_vix*(1+distribution['q10']/100):.2f}");print(f"   25th: {current_vix*(1+distribution['q25']/100):.2f}");print(f"   50th: {current_vix*(1+distribution['q50']/100):.2f}");print(f"   75th: {current_vix*(1+distribution['q75']/100):.2f}");print(f"   90th: {current_vix*(1+distribution['q90']/100):.2f}")
        print(f"\nDirectional Forecast:");print(f"   Probability UP:   {distribution['prob_up']:.1%}");print(f"   Probability DOWN: {distribution['prob_down']:.1%}")
        print(f"\nConfidence: {distribution['confidence_score']:.2f}")
        print(f"{'='*80}\n")
    def _store_prediction(self,distribution:Dict,observation:pd.Series,observation_date:pd.Timestamp)->str:
        target_date=observation_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
        prediction_id=f"pred_{target_date.strftime('%Y%m%d')}_h{TARGET_CONFIG['horizon_days']}"
        prediction={
            "prediction_id":prediction_id,
            "timestamp":datetime.now(),
            "forecast_date":target_date,
            "observation_date":observation_date,
            "horizon":TARGET_CONFIG["horizon_days"],
            "current_vix":float(observation["vix"]),
            "calendar_cohort":observation["calendar_cohort"],
            "cohort_weight":float(observation.get("cohort_weight",1.0)),
            "median_forecast":distribution["median_forecast"],
            "point_estimate":distribution["median_forecast"],
            "q10":distribution["q10"],
            "q25":distribution["q25"],
            "q50":distribution["q50"],
            "q75":distribution["q75"],
            "q90":distribution["q90"],
            "prob_up":distribution["prob_up"],
            "prob_down":distribution["prob_down"],
            "direction_probability":distribution.get("direction_probability",distribution["prob_up"]),
            "confidence_score":distribution["confidence_score"],
            "feature_quality":float(distribution["metadata"]["feature_quality"]),
            "num_features_used":len(self.forecaster.feature_names),
            "features_used":",".join(self.forecaster.feature_names[:10]),
            "model_version":"v3.1_log_rv"
        }
        stored_id=self.prediction_db.store_prediction(prediction)
        return stored_id if stored_id else prediction_id
    def backfill_actuals(self):
        logger.info("\n"+"="*80);logger.info("BACKFILLING ACTUALS");logger.info("="*80)
        feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS)
        vix_series=feature_data["vix"]
        self.prediction_db.backfill_actuals(vix_series)
        logger.info("="*80+"\n")
    def generate_forecast_batch(self,start_date:str,end_date:str,frequency:str="daily"):
        logger.info(f"\n{'='*80}");logger.info(f"BATCH FORECASTING: {start_date} to {end_date}");logger.info(f"{'='*80}")
        logger.info("🔧 Building features for batch...")
        feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=end_date)
        df=feature_data["features"]
        metadata_cols=["calendar_cohort","cohort_weight","feature_quality"]
        numeric_cols=[c for c in df.columns if c not in metadata_cols]
        logger.info(f"   Applying nuclear dtype fix to {len(numeric_cols)} columns...")
        for col in numeric_cols:
            if df[col].dtype==object:df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0.0)
            df[col]=df[col].astype(np.float64)
        object_cols=df.select_dtypes(include=["object"]).columns.tolist()
        unexpected=[c for c in object_cols if c not in metadata_cols]
        if unexpected:
            logger.error(f"❌ {len(unexpected)} non-numeric columns remain")
            raise ValueError(f"DataFrame still has object columns: {unexpected[:5]}")
        logger.info("✅ Features validated and cached");logger.info(f"   Shape: {df.shape}");logger.info(f"   All {len(numeric_cols)} numeric cols are float64")
        start=pd.Timestamp(start_date);end=pd.Timestamp(end_date)
        date_range=df[(df.index>=start)&(df.index<=end)].index
        logger.info(f"📅 Forecasting {len(date_range)} dates");logger.info(f"   Range: {date_range[0].date()} to {date_range[-1].date()}")
        forecasts=[];commit_interval=50
        for i,date in enumerate(date_range):
            try:
                observation=df.loc[date]
                feature_dict=observation.to_dict()
                quality_score=self.validator.compute_feature_quality(feature_dict,date)
                usable,_=self.validator.check_quality_threshold(quality_score)
                if not usable:
                    logger.warning(f"   Skipped {date.date()}: low quality ({quality_score:.2f})")
                    continue
                cohort=observation.get("calendar_cohort","mid_cycle")
                cohort_weight=observation.get("cohort_weight",1.0)
                if cohort not in self.forecaster.models:cohort="mid_cycle"
                feature_values=observation[self.forecaster.feature_names]
                feature_array=pd.to_numeric(feature_values,errors="coerce").values
                X_df=pd.DataFrame(feature_array.reshape(1,-1),columns=self.forecaster.feature_names,dtype=np.float64).fillna(0.0)
                current_vix=float(observation["vix"])
                distribution=self.forecaster.predict(X_df,cohort,current_vix)

                if self.calibrator:
                    raw_median = distribution["median_forecast"]
                    current_vix = float(observation["vix"])
                    cohort = observation.get("calendar_cohort", None)

                    calib_result = self.calibrator.calibrate(
                        raw_forecast=raw_median,
                        current_vix=current_vix,
                        cohort=cohort
                    )

                    adjustment = calib_result["adjustment"]

                    # Calculate adjustment ratio (not absolute adjustment)
                    ratio = calib_result["calibrated_forecast"] / raw_median if raw_median != 0 else 1.0

                    # Apply proportional scaling to preserve distribution shape
                    distribution["median_forecast"] = calib_result["calibrated_forecast"]
                    distribution["q50"] = calib_result["calibrated_forecast"]
                    distribution["q10"] = distribution["q10"] * ratio
                    distribution["q25"] = distribution["q25"] * ratio
                    distribution["q75"] = distribution["q75"] * ratio
                    distribution["q90"] = distribution["q90"] * ratio
                    distribution["point_estimate"] = calib_result["calibrated_forecast"]

                    # Ensure monotonicity after scaling
                    quantile_values = [
                        distribution["q10"],
                        distribution["q25"],
                        distribution["q50"],
                        distribution["q75"],
                        distribution["q90"]
                    ]
                    quantile_values_sorted = sorted(quantile_values)
                    distribution["q10"] = quantile_values_sorted[0]
                    distribution["q25"] = quantile_values_sorted[1]
                    distribution["q50"] = quantile_values_sorted[2]
                    distribution["q75"] = quantile_values_sorted[3]
                    distribution["q90"] = quantile_values_sorted[4]

                distribution["confidence_score"]*=2-cohort_weight
                distribution["confidence_score"]=np.clip(distribution["confidence_score"],0,1)
                forecast_date=date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
                distribution["metadata"]={
                    "observation_date":date.strftime("%Y-%m-%d"),
                    "forecast_date":forecast_date.strftime("%Y-%m-%d"),
                    "horizon_days":TARGET_CONFIG["horizon_days"],
                    "feature_quality":float(quality_score),
                    "cohort_weight":float(cohort_weight),
                    "current_vix":float(observation["vix"]),
                    "features_used":len(self.forecaster.feature_names)
                }
                forecast_date=date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
                prediction_id=self._store_prediction(distribution,observation,date)
                distribution["prediction_id"]=prediction_id
                forecasts.append(distribution)
                if(i+1)%commit_interval==0:
                    logger.info(f"   Progress: {i+1}/{len(date_range)} forecasts")
                    self.prediction_db.commit()
                    logger.info(f"   ✅ Committed batch of {commit_interval}")
            except Exception as e:
                logger.warning(f"   Failed {date.date()}: {e}")
                continue
        if len(forecasts)%commit_interval!=0:
            self.prediction_db.commit()
            logger.info(f"   ✅ Committed final batch")
        logger.info("="*80);logger.info(f"✅ Generated {len(forecasts)} forecasts");logger.info("="*80)
        status=self.prediction_db.get_commit_status()
        if status["pending_writes"]>0:
            logger.error(f"🚨 WARNING: {status['pending_writes']} uncommitted writes!")
            raise RuntimeError(f"Lost {status['pending_writes']} forecasts - commit failed!")
        else:logger.info("✅ All forecasts committed to database")
        return forecasts
    def train(self,years:int=TRAINING_YEARS,real_time_vix:bool=True,verbose:bool=False,enable_anomaly:bool=False):
        print(f"\n{'='*80}\nINTEGRATED SYSTEM V3.1 - TRAINING MODE\n{'='*80}")
        print(f"Config: {years}y training | Real-time VIX: {real_time_vix} | Anomaly: {enable_anomaly}")
        if self.memory_monitoring_enabled:self._log_memory_stats(context="pre-training")
        print("\n[1/2] Building features...")
        feature_data=self.feature_engine.build_complete_features(years=years)
        features=feature_data["features"];vix=feature_data["vix"];spx=feature_data["spx"]
        if self.memory_monitoring_enabled:self._log_memory_stats(context="post-features")
        self.orchestrator.features=features;self.orchestrator.vix_ml=vix;self.orchestrator.spx_ml=spx
        if enable_anomaly:
            print("[2/2] Training anomaly system...")
            vix_history_all=self.orchestrator.fetcher.fetch_yahoo("^VIX","1990-01-02",datetime.now().strftime("%Y-%m-%d"))["Close"].squeeze()
            self.orchestrator.train(features=features,vix=vix,spx=spx,vix_history_all=vix_history_all,verbose=verbose)
            if self.memory_monitoring_enabled:self._log_memory_stats(context="post-training")
            if real_time_vix:
                try:
                    live_vix=self.orchestrator.fetcher.fetch_price("^VIX")
                    if live_vix:
                        self.orchestrator.vix_ml.iloc[-1]=live_vix
                        self.orchestrator.features.iloc[-1,self.orchestrator.features.columns.get_loc("vix")]=live_vix
                        if verbose:print(f"✅ Updated live VIX: {live_vix:.2f}")
                except Exception as e:warnings.warn(f"Live VIX fetch failed: {e}")
        else:print("[2/2] Anomaly training skipped (enable_anomaly=False)")
        self.trained=True
        print(f"\n{'='*80}\n✅ TRAINING COMPLETE\n{'='*80}")
    def get_market_state(self)->dict:
        if not self.trained:raise ValueError("Must train system first")
        anomaly_result=self._get_cached_anomaly_result()
        persistence_stats=self.orchestrator.get_persistence_stats()
        current_vix=float(self.orchestrator.vix_ml.iloc[-1])
        current_regime=self._classify_vix_regime(current_vix)
        ensemble=anomaly_result["ensemble"]
        ensemble_score=ensemble["score"]
        level,p_value,confidence=self.orchestrator.anomaly_detector.classify_anomaly(ensemble_score,method="statistical")
        severity_messages={
            "CRITICAL":f"Extreme anomaly ({ensemble_score:.1%}) - Markets in unprecedented configuration",
            "HIGH":f"Significant anomaly ({ensemble_score:.1%}) - Notable market stress detected",
            "MODERATE":f"Moderate anomaly ({ensemble_score:.1%}) - Elevated market uncertainty",
            "NORMAL":f"Normal conditions ({ensemble_score:.1%}) - Market within typical ranges"
        }
        ensemble["severity"]=level;ensemble["severity_message"]=severity_messages[level]
        if p_value is not None:ensemble["p_value"]=float(p_value);ensemble["confidence"]=float(confidence)
        top_anomalies=self._get_top_anomalies_list(anomaly_result)
        regime_stats=self.orchestrator.regime_stats["regimes"][current_regime["id"]]
        persistence_prob=regime_stats["transitions_5d"]["persistence"]["probability"]
        persistence_prob_clamped=max(0.01,min(0.99,persistence_prob))
        expected_duration=1.0/(1.0-persistence_prob_clamped)
        return{
            "timestamp":datetime.now().isoformat(),
            "vix":{"current":current_vix,"regime":current_regime,"regime_stats":regime_stats},
            "anomaly_analysis":{"ensemble":ensemble,"top_anomalies":top_anomalies,"persistence":{"probability":float(persistence_prob),"expected_duration_days":float(expected_duration)}}
        }
    def print_anomaly_summary(self):
        if not self.trained:raise ValueError("Must train system first")
        state=self.get_market_state()
        print(f"\n{'='*80}");print("ANOMALY DETECTION SUMMARY");print(f"{'='*80}")
        vix_info=state["vix"];anomaly=state["anomaly_analysis"]
        print(f"\nCurrent VIX: {vix_info['current']:.2f}");print(f"Regime: {vix_info['regime']['label']}")
        ensemble=anomaly["ensemble"]
        print(f"\nEnsemble Anomaly: {ensemble['score']:.1%}");print(f"Severity: {ensemble['severity']}");print(f"{ensemble['severity_message']}")
        print(f"\nTop Anomalies:")
        for i,anom in enumerate(anomaly["top_anomalies"][:5],1):print(f"  {i}. {anom['feature']}: {anom['score']:.1%}")
        print(f"{'='*80}\n")
    def _get_cached_anomaly_result(self):
        now=datetime.now()
        if self._cached_anomaly_result is not None:
            if(now-self._cache_timestamp).seconds<300:return self._cached_anomaly_result
        result=self.orchestrator.predict()
        self._cached_anomaly_result=result;self._cache_timestamp=now
        return result
    def _classify_vix_regime(self,vix:float)->dict:
        if vix<15:return{"id":"low","label":"Low Volatility","range":"<15"}
        elif vix<20:return{"id":"normal","label":"Normal Volatility","range":"15-20"}
        elif vix<30:return{"id":"elevated","label":"Elevated Volatility","range":"20-30"}
        else:return{"id":"crisis","label":"Crisis Volatility","range":">30"}
    def _get_top_anomalies_list(self,anomaly_result:dict)->list:
        top_anomalies=[]
        for detection in anomaly_result.get("individual_detections",[]):
            if detection.get("is_anomaly"):
                top_anomalies.append({"feature":detection["feature"],"score":detection["anomaly_score"],"threshold":detection.get("threshold",None)})
        top_anomalies.sort(key=lambda x:x["score"],reverse=True)
        return top_anomalies
    def _log_memory_stats(self,context:str=""):
        if not self.memory_monitoring_enabled:return
        mem_info=self.process.memory_info();mem_mb=mem_info.rss/1024/1024
        if self.baseline_memory_mb is None:self.baseline_memory_mb=mem_mb
        delta_mb=mem_mb-self.baseline_memory_mb
        self.memory_history.append({"context":context,"memory_mb":mem_mb,"delta_mb":delta_mb})
        logger.info(f"💾 Memory ({context}): {mem_mb:.1f} MB (Δ {delta_mb:+.1f} MB)")
    def _get_features(self,force_refresh=False)->pd.DataFrame:
        now=pd.Timestamp.now();today=(now.year,now.month,now.day)
        if not force_refresh and self._feature_cache is not None and self._feature_cache_date==today:
            logger.info("📦 Using cached features (already built today)")
            return self._feature_cache
        logger.info("🔧 Building features...")
        feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS)
        df=feature_data["features"]
        metadata_cols=["calendar_cohort","cohort_weight","feature_quality"]
        numeric_cols=[c for c in df.columns if c not in metadata_cols]
        logger.info(f"   Applying nuclear dtype fix to {len(numeric_cols)} columns...")
        numeric_array=df[numeric_cols].values.astype(np.float64)
        df_clean=pd.DataFrame(numeric_array,columns=numeric_cols,index=df.index)
        for col in metadata_cols:
            if col in df.columns:df_clean[col]=df[col]
        df_clean=df_clean[df.columns]
        final_object_cols=df_clean[numeric_cols].select_dtypes(include=["object"]).columns.tolist()
        if final_object_cols:
            logger.error(f"❌ Nuclear fix failed: {len(final_object_cols)} object columns remain")
            raise ValueError(f"Dtype conversion failed on: {final_object_cols[:10]}")
        dtype_check=df_clean[numeric_cols].dtypes.value_counts()
        if len(dtype_check)!=1 or dtype_check.index[0]!=np.float64:
            logger.error(f"❌ Mixed dtypes: {dtype_check.to_dict()}")
            raise ValueError("Not all numeric columns are float64")
        self._feature_cache=df_clean;self._feature_cache_date=today
        logger.info(f"✅ Features validated and cached");logger.info(f"   Shape: {df_clean.shape}");logger.info(f"   All {len(numeric_cols)} numeric cols are float64")
        return df_clean
def main():
    parser=argparse.ArgumentParser(description="Integrated VIX Forecasting System V3.1")
    parser.add_argument("--mode",choices=["forecast","complete","batch","backfill","anomaly"],default="forecast",help="Operation mode")
    parser.add_argument("--forecast-date",type=str,help="Forecast date (YYYY-MM-DD), default: today")
    parser.add_argument("--start-date",type=str,help="Start date for batch mode (YYYY-MM-DD)")
    parser.add_argument("--end-date",type=str,help="End date for batch mode (YYYY-MM-DD)")
    parser.add_argument("--models-dir",type=str,default="models",help="Directory containing trained models")
    parser.add_argument("--db-path",type=str,default="data_cache/predictions.db",help="Path to predictions database")
    args=parser.parse_args()
    if args.mode=="anomaly"and not ENABLE_TRAINING:
        print(f"\n{'='*80}");print("⚠️ TRAINING DISABLED (config.ENABLE_TRAINING = False)");print("⚠️ Set ENABLE_TRAINING = True in config.py for anomaly mode");print(f"{'='*80}\n")
        return
    Path("logs").mkdir(exist_ok=True)
    system=IntegratedForecastingSystem(models_dir=args.models_dir,db_path=args.db_path)
    if args.mode=="forecast":
        forecast_date=None
        if args.forecast_date:forecast_date=pd.Timestamp(args.forecast_date)
        result=system.generate_forecast(date=forecast_date)
        if result:sys.exit(0)
        else:sys.exit(1)
    elif args.mode=="complete":
        logger.info("🎯 MODE: Complete Workflow - Everything in one command")
        try:
            cal_start,cal_end=CALIBRATION_PERIOD
            val_start,val_end=VALIDATION_PERIOD
            logger.info("="*80);logger.info("COMPLETE WORKFLOW");logger.info("="*80)
            logger.info("This will:");logger.info(f"  1. Generate uncalibrated forecasts ({cal_start} to {cal_end})");logger.info(f"  2. Backfill actuals for calibration period");logger.info(f"  3. Train calibrator on {cal_start[:4]} data");logger.info(f"  4. Regenerate {val_start[:4]} WITH calibration");logger.info(f"  5. Backfill actuals for validation period");logger.info(f"  6. Generate {PRODUCTION_START_DATE[:4]} forecasts (Jan 1 - today)");logger.info(f"  7. Backfill actuals for {PRODUCTION_START_DATE[:4]}");logger.info(f"  8. Run validation diagnostics");logger.info("="*80)
            logger.info(f"\n[1/8] Generating uncalibrated forecasts ({cal_start} to {cal_end})...")
            original_calibrator=system.calibrator;system.calibrator=None
            system.generate_forecast_batch(cal_start,cal_end)
            logger.info(f"\n[2/8] Backfilling actuals for {cal_start[:4]}...")
            system.prediction_db.backfill_actuals()
            logger.info(f"\n[3/8] Training calibrator on {cal_start[:4]} data...")
            calibrator=ForecastCalibrator()
            success=calibrator.fit_from_database(database=system.prediction_db,start_date=cal_start,end_date=cal_end)
            if not success:
                logger.error("❌ Calibration failed - insufficient data")
                system.calibrator=original_calibrator
                return
            calibrator.save_calibrator()
            logger.info(f"\n[4/8] Clearing old {val_start[:4]} forecasts...")
            conn=sqlite3.connect(system.prediction_db.db_path);cursor=conn.cursor()
            cursor.execute("DELETE FROM forecasts WHERE forecast_date BETWEEN ? AND ?",(val_start,val_end))
            deleted=cursor.rowcount;conn.commit();conn.close()
            logger.info(f"   Deleted {deleted} old forecasts")
            logger.info(f"\n[5/8] Generating {val_start[:4]} forecasts WITH calibration...")
            system.calibrator=ForecastCalibrator.load()
            system.generate_forecast_batch(val_start,val_end)
            logger.info(f"\n[6/8] Backfilling actuals for {val_start[:4]}...")
            system.prediction_db.backfill_actuals()
            logger.info(f"\n[7/8] Generating {PRODUCTION_START_DATE[:4]} forecasts...")
            today=datetime.now().strftime("%Y-%m-%d")
            system.generate_forecast_batch(PRODUCTION_START_DATE,today)
            logger.info(f"\n[8/8] Backfilling {PRODUCTION_START_DATE[:4]} actuals...")
            system.prediction_db.backfill_actuals()
            try:
                from core.walk_forward_validation import EnhancedWalkForwardValidator
            except ImportError:
                try:
                    from walk_forward_validation import EnhancedWalkForwardValidator
                except ImportError:
                    logger.warning("Walk-forward validation module not found - skipping diagnostic report")
                    EnhancedWalkForwardValidator=None
            if EnhancedWalkForwardValidator:
                validator=EnhancedWalkForwardValidator(db_path=system.prediction_db.db_path)
                validator.generate_diagnostic_report()
            logger.info("\n"+"="*80);logger.info("✅ COMPLETE WORKFLOW FINISHED");logger.info("="*80)
            logger.info("\n📊 Results:");logger.info("  • diagnostics/walk_forward_metrics.json");logger.info("  • diagnostics/*.png")
            logger.info(f"\n📈 Calibrator trained on: {cal_start} to {cal_end}");logger.info(f"📈 Validation period: {val_start} to {val_end}");logger.info(f"📈 {PRODUCTION_START_DATE[:4]} forecasts: up to {today}")
            logger.info("\n🚀 System ready for production!");logger.info("  Run daily: python integrated_system_production.py --mode forecast")
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}",exc_info=True)
            sys.exit(1)
    elif args.mode=="batch":
        if not args.start_date or not args.end_date:
            logger.error("❌ Batch mode requires --start-date and --end-date")
            sys.exit(1)
        system.generate_forecast_batch(start_date=args.start_date,end_date=args.end_date)
        sys.exit(0)
    elif args.mode=="backfill":
        system.prediction_db.backfill_actuals()
        sys.exit(0)
    elif args.mode=="anomaly":
        system.train(years=TRAINING_YEARS,real_time_vix=True,verbose=False,enable_anomaly=True)
        system.print_anomaly_summary()
        anomaly_result=system._get_cached_anomaly_result()
        if anomaly_result and system.orchestrator.anomaly_detector:
            from export.unified_exporter import UnifiedExporter
            persistence_stats=system.orchestrator.get_persistence_stats()
            exporter=UnifiedExporter(output_dir="./json_data")
            exporter.export_live_state(orchestrator=system.orchestrator,anomaly_result=anomaly_result,spx=system.orchestrator.spx_ml,vix=system.orchestrator.vix_ml,persistence_stats=persistence_stats)
            logger.info("✅ Anomaly data exported to ./json_data/")
        sys.exit(0)
if __name__=="__main__":
    main()
