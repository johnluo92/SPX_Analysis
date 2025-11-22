#!/usr/bin/env python3
import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from config import CALIBRATION_WINDOW_DAYS,CALIBRATION_DECAY_LAMBDA,MIN_SAMPLES_FOR_CORRECTION,MODEL_VALIDATION_MAE_THRESHOLD,TARGET_CONFIG,TRAINING_YEARS
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer as UnifiedFeatureEngine
from core.forecast_calibrator import ForecastCalibrator
from core.prediction_database import PredictionDatabase
from core.regime_classifier import classify_vix_regime
from core.temporal_validator import TemporalSafetyValidator as TemporalValidator
from core.xgboost_trainer_v3 import SimplifiedVIXForecaster
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/integrated_system.log")])
logger=logging.getLogger(__name__)
def get_last_complete_month_end():
    today=datetime.now();first_of_this_month=today.replace(day=1);last_month_end=first_of_this_month-pd.Timedelta(days=1);return last_month_end.strftime("%Y-%m-%d")
def check_model_stale():
    metadata_file=Path("models/training_metadata.json")
    if not metadata_file.exists():return True,"No model exists"
    with open(metadata_file)as f:meta=json.load(f)
    model_end=meta.get("training_end","2000-01-01");target_end=get_last_complete_month_end()
    if model_end<target_end:return True,f"Model trained through {model_end}, target {target_end}"
    return False,f"Model current (trained through {model_end})"
def retrain_model():
    logger.info("="*80);logger.info("RETRAINING MODEL");logger.info("="*80)
    from train_probabilistic_models import main as train_main
    try:
        train_main();logger.info("✅ Retraining complete\n");return True
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}");return False
def fit_calibrator(db):
    logger.info("="*80);logger.info("FITTING CALIBRATOR");logger.info("="*80)
    cal=ForecastCalibrator();cal.fit_from_database(db);cal.save("models")
    logger.info("✅ Calibrator saved\n");return cal
class VIXForecaster:
    def __init__(self):
        self.data_fetcher=UnifiedDataFetcher();self.feature_engine=UnifiedFeatureEngine(data_fetcher=self.data_fetcher);self.forecaster=SimplifiedVIXForecaster();self.calibrator=ForecastCalibrator();self.validator=TemporalValidator();self.db=PredictionDatabase();self._load_models()
    def _load_models(self):
        logger.info("📂 Loading models...")
        magnitude_file=Path("models/magnitude_5d_model.pkl")
        if not magnitude_file.exists():logger.error("❌ No model. Will retrain.");return False
        self.forecaster.load("models")
        logger.info(f"📊 Magnitude model: {len(self.forecaster.feature_names)} features")
        cal_file=Path("models/calibrator.pkl")
        if cal_file.exists():
            self.calibrator.load("models");logger.info("📊 Calibrator loaded")
        else:logger.warning("⚠️  No calibrator")
        return True
    def generate_forecast(self,date=None):
        logger.info("="*80);logger.info("GENERATING FORECAST");logger.info("="*80)
        if date:
            target_date=pd.Timestamp(date);logger.info(f"📅 Forecast date: {target_date.date()}")
            feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=target_date.strftime("%Y-%m-%d"),force_historical=True)
        else:
            now=pd.Timestamp.now().normalize()
            feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=now.strftime("%Y-%m-%d"),force_historical=True)
            target_date=feature_data["features"].index[-1]
            logger.info(f"📅 Using latest available: {target_date.date()}")
        df=feature_data["features"];metadata_cols=["calendar_cohort","cohort_weight","feature_quality"];numeric_cols=[c for c in df.columns if c not in metadata_cols]
        for col in numeric_cols:
            if df[col].dtype==object:df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0.0)
            df[col]=df[col].astype(np.float64)
        if target_date not in df.index:
            available=df.index[-5:]
            logger.error(f"❌ Date {target_date.date()} not available")
            logger.error(f"   Last 5 dates: {[d.date()for d in available]}")
            return None
        obs=df.loc[target_date];feature_dict=obs.to_dict();quality=self.validator.compute_feature_quality(feature_dict,target_date);usable,msg=self.validator.check_quality_threshold(quality)
        logger.info(f"📊 Data quality: {quality:.2f} - {msg}")
        if not usable:
            logger.error("❌ Quality insufficient");return None
        cohort=obs.get("calendar_cohort","mid_cycle");cohort_weight=obs.get("cohort_weight",1.0)
        logger.info(f"📅 Cohort: {cohort} (weight: {cohort_weight:.2f})")
        feature_vals=obs[self.forecaster.feature_names];feature_arr=pd.to_numeric(feature_vals,errors="coerce").values;X=pd.DataFrame(feature_arr.reshape(1,-1),columns=self.forecaster.feature_names,dtype=np.float64).fillna(0.0);current_vix=float(obs["vix"])
        forecast=self.forecaster.predict(X,current_vix)
        if self.calibrator.fitted:
            cal_result=self.calibrator.calibrate(forecast["magnitude_pct"],current_vix,cohort);forecast["magnitude_pct"]=cal_result["calibrated_forecast"];forecast["expected_vix"]=current_vix*(1+forecast["magnitude_pct"]/100);forecast["calibration"]=cal_result
        else:forecast["calibration"]={"correction_type":"not_fitted","adjustment":0.0}
        forecast_date=target_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
        forecast["metadata"]={"observation_date":target_date.strftime("%Y-%m-%d"),"forecast_date":forecast_date.strftime("%Y-%m-%d"),"horizon_days":TARGET_CONFIG["horizon_days"],"feature_quality":float(quality),"cohort_weight":float(cohort_weight),"calendar_cohort":cohort,"features_used":len(self.forecaster.feature_names)}
        self._log_forecast(forecast);self._store_forecast(forecast,obs,target_date)
        logger.info("="*80+"\n");return forecast
    def _log_forecast(self,f):
        logger.info(f"\n5-Day VIX Forecast")
        logger.info(f"Current VIX: {f['current_vix']:.2f}")
        logger.info(f"Direction: {f['direction']} (confidence: {f['confidence']:.1f}%)")
        logger.info(f"Magnitude: {f['magnitude_pct']:+.2f}%")
        logger.info(f"Expected VIX: {f['expected_vix']:.2f}")
        if "calibration"in f:
            cal=f["calibration"];logger.info(f"Calibration: {cal.get('adjustment',0):+.3f}% ({cal.get('correction_type','none')})")
        logger.info(f"Regime: {f['current_regime']} → {f['expected_regime']}")
        logger.info(f"Actionable: {'YES'if f['actionable']else'NO'}")
    def _store_forecast(self,forecast,obs,obs_date):
        forecast_date=obs_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"]);pred_id=f"pred_{forecast_date.strftime('%Y%m%d')}_h{TARGET_CONFIG['horizon_days']}"
        conf_pct=abs(forecast["magnitude_pct"]);scale=min(conf_pct/20.0,1.0)
        if forecast["direction"]=="UP":prob_up=0.5+(0.5*scale)
        else:prob_up=0.5-(0.5*scale)
        prob_down=1.0-prob_up;correction_type=forecast.get("calibration",{}).get("correction_type","none")
        pred={"prediction_id":pred_id,"timestamp":datetime.now(),"forecast_date":forecast_date,"observation_date":obs_date,"horizon":TARGET_CONFIG["horizon_days"],"current_vix":float(obs["vix"]),"calendar_cohort":obs["calendar_cohort"],"cohort_weight":float(obs.get("cohort_weight",1.0)),"prob_up":float(prob_up),"prob_down":float(prob_down),"magnitude_forecast":forecast["magnitude_pct"],"expected_vix":forecast["expected_vix"],"feature_quality":float(forecast["metadata"]["feature_quality"]),"num_features_used":len(self.forecaster.feature_names),"correction_type":correction_type,"features_used":(",".join(self.forecaster.feature_names[:10])),"model_version":"v5.1_unified"}
        self.db.store_prediction(pred);self.db.commit()
        logger.info(f"💾 Stored: {pred_id}")
def main():
    parser=argparse.ArgumentParser(description="VIX Forecasting System - Unified Interface")
    parser.add_argument("--force-retrain",action="store_true",help="Force retraining even if model current")
    parser.add_argument("--date",type=str,help="Specific forecast date (YYYY-MM-DD)")
    parser.add_argument("--skip-retrain-check",action="store_true",help="Skip model staleness check")
    args=parser.parse_args()
    Path("logs").mkdir(exist_ok=True);Path("models").mkdir(exist_ok=True)
    logger.info("\n"+"="*80)
    logger.info("VIX FORECASTING SYSTEM v5.1")
    logger.info("="*80)
    logger.info(f"Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if not args.skip_retrain_check or args.force_retrain:
        stale,reason=check_model_stale()
        logger.info(f"Model status: {reason}")
        if stale or args.force_retrain:
            if args.force_retrain:logger.info("🔄 Force retrain requested")
            if not retrain_model():logger.error("❌ Retrain failed");sys.exit(1)
        else:logger.info("✓ Model current, skipping retrain\n")
    db=PredictionDatabase();forecaster=VIXForecaster()
    if not Path("models/calibrator.pkl").exists():
        logger.info("📊 Calibrator missing, fitting...")
        fit_calibrator(db)
        forecaster.calibrator.load("models")
    forecast=forecaster.generate_forecast(date=args.date)
    if forecast:
        logger.info("✅ Forecast complete")
        sys.exit(0)
    else:
        logger.error("❌ Forecast failed")
        sys.exit(1)
if __name__=="__main__":main()
