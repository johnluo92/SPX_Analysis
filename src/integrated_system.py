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
    logger.info("\n🔄 RETRAINING MODEL")
    from train_probabilistic_models import main as train_main
    try:
        train_main();logger.info("✅ Retraining complete");return True
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}");return False
def backfill_predictions(db):
    logger.info("\n📊 BACKFILLING ACTUALS")
    fetcher=UnifiedDataFetcher()
    db.backfill_actuals(fetcher)
def fit_calibrator(db):
    logger.info("\n📊 FITTING CALIBRATOR")
    cal=ForecastCalibrator();result=cal.fit_from_database(db)
    if result:cal.save("models");logger.info("✅ Calibrator fitted");return cal
    else:logger.warning("⚠️  Calibrator not fitted");return None
class VIXForecaster:
    def __init__(self):
        self.data_fetcher=UnifiedDataFetcher();self.feature_engine=UnifiedFeatureEngine(data_fetcher=self.data_fetcher);self.forecaster=SimplifiedVIXForecaster();self.calibrator=ForecastCalibrator();self.validator=TemporalValidator();self.db=PredictionDatabase();self._load_models()
    def _load_models(self):
        magnitude_file=Path("models/magnitude_5d_model.pkl")
        if not magnitude_file.exists():logger.error("❌ No model found");return False
        self.forecaster.load("models")
        cal_file=Path("models/calibrator.pkl")
        if cal_file.exists():self.calibrator.load("models");logger.info("✅ Calibrator loaded")
        else:logger.warning("⚠️  No calibrator found")
        return True
    def generate_forecast(self,date,use_live_data=True):
        target_date=pd.Timestamp(date)
        # For production forecasts, use live data. For backtesting, use historical.
        feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=date,force_historical=not use_live_data)
        df=feature_data["features"]
        if target_date not in df.index:logger.error(f"❌ {date} not in data");return None
        metadata_cols=["calendar_cohort","cohort_weight","feature_quality"];numeric_cols=[c for c in df.columns if c not in metadata_cols]
        for col in numeric_cols:
            if df[col].dtype==object:df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0.0)
            df[col]=df[col].astype(np.float64)
        obs=df.loc[target_date];feature_dict=obs.to_dict();quality=self.validator.compute_feature_quality(feature_dict,target_date);usable,msg=self.validator.check_quality_threshold(quality)
        if not usable:logger.error(f"❌ Quality insufficient: {quality:.2f}");return None
        cohort=obs.get("calendar_cohort","mid_cycle");cohort_weight=obs.get("cohort_weight",1.0)
        feature_vals=obs[self.forecaster.feature_names];feature_arr=pd.to_numeric(feature_vals,errors="coerce").values;X=pd.DataFrame(feature_arr.reshape(1,-1),columns=self.forecaster.feature_names,dtype=np.float64).fillna(0.0);current_vix=float(obs["vix"])
        forecast=self.forecaster.predict(X,current_vix)
        if self.calibrator.fitted:
            cal_result=self.calibrator.calibrate(forecast["magnitude_pct"],current_vix,cohort);forecast["magnitude_pct"]=cal_result["calibrated_forecast"];forecast["expected_vix"]=current_vix*(1+forecast["magnitude_pct"]/100);forecast["calibration"]=cal_result
        else:forecast["calibration"]={"correction_type":"not_fitted","adjustment":0.0}
        forecast_date=target_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"])
        forecast["metadata"]={"observation_date":target_date.strftime("%Y-%m-%d"),"forecast_date":forecast_date.strftime("%Y-%m-%d"),"horizon_days":TARGET_CONFIG["horizon_days"],"feature_quality":float(quality),"cohort_weight":float(cohort_weight),"calendar_cohort":cohort,"features_used":len(self.forecaster.feature_names)}
        self._store_forecast(forecast,obs,target_date)
        return forecast
    def _store_forecast(self,forecast,obs,obs_date):
        forecast_date=obs_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"]);pred_id=f"pred_{forecast_date.strftime('%Y%m%d')}_h{TARGET_CONFIG['horizon_days']}"
        conf_pct=abs(forecast["magnitude_pct"]);scale=min(conf_pct/20.0,1.0)
        if forecast["direction"]=="UP":prob_up=0.5+(0.5*scale)
        else:prob_up=0.5-(0.5*scale)
        prob_down=1.0-prob_up;correction_type=forecast.get("calibration",{}).get("correction_type","none")
        pred={"prediction_id":pred_id,"timestamp":datetime.now(),"forecast_date":forecast_date,"observation_date":obs_date,"horizon":TARGET_CONFIG["horizon_days"],"current_vix":float(obs["vix"]),"calendar_cohort":obs["calendar_cohort"],"cohort_weight":float(obs.get("cohort_weight",1.0)),"prob_up":float(prob_up),"prob_down":float(prob_down),"magnitude_forecast":forecast["magnitude_pct"],"expected_vix":forecast["expected_vix"],"feature_quality":float(forecast["metadata"]["feature_quality"]),"num_features_used":len(self.forecaster.feature_names),"correction_type":correction_type,"features_used":(",".join(self.forecaster.feature_names[:10])),"model_version":"v5.1_unified"}
        self.db.store_prediction(pred);self.db.commit()
def main():
    parser=argparse.ArgumentParser(description="VIX Forecasting System")
    parser.add_argument("--force-retrain",action="store_true")
    parser.add_argument("--rebuild-calibration",action="store_true")
    args=parser.parse_args()
    Path("logs").mkdir(exist_ok=True);Path("models").mkdir(exist_ok=True)
    logger.info(f"VIX FORECASTING SYSTEM v5.1 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stale,reason=check_model_stale()
    logger.info(f"Model: {reason}")
    if stale or args.force_retrain:
        if not retrain_model():sys.exit(1)
    db=PredictionDatabase();forecaster=VIXForecaster()
    if not Path("models/calibrator.pkl").exists() or args.rebuild_calibration:
        logger.info("\n📊 FITTING CALIBRATOR FROM DATABASE")
        backfill_predictions(db)
        cal=fit_calibrator(db)
        if cal:
            forecaster.calibrator=cal
        else:
            logger.warning("⚠️  Proceeding without calibration")

    logger.info("\n📊 TODAY'S FORECAST")
    # Detect latest available trading day instead of blindly using yesterday
    yesterday=(datetime.now()-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    feature_data=forecaster.feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=yesterday,force_historical=False)
    df=feature_data["features"]
    latest_available=df.index[-1].strftime("%Y-%m-%d")
    logger.info(f"Latest available data: {latest_available}")

    forecast=forecaster.generate_forecast(latest_available,use_live_data=True)
    if forecast:
        logger.info(f"Current: {forecast['current_vix']:.2f} | {forecast['direction']} {forecast['magnitude_pct']:+.2f}% | Expected: {forecast['expected_vix']:.2f}")
        if forecast['calibration']['correction_type']!='not_fitted':logger.info(f"Calibration: {forecast['calibration']['adjustment']:+.3f}% ({forecast['calibration']['correction_type']})")
        logger.info(f"Regime: {forecast['current_regime']} → {forecast['expected_regime']} | Actionable: {'YES' if forecast['actionable'] else 'NO'}")
        sys.exit(0)
    else:sys.exit(1)
if __name__=="__main__":main()
