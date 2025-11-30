import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from config import CALIBRATION_WINDOW_DAYS,MIN_SAMPLES_FOR_CORRECTION,TARGET_CONFIG,TRAINING_YEARS,ENSEMBLE_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer as UnifiedFeatureEngine
from core.forecast_calibrator import ForecastCalibrator
from core.prediction_database import PredictionDatabase
from core.temporal_validator import TemporalSafetyValidator as TemporalValidator
from core.xgboost_trainer_v3 import SimplifiedVIXForecaster
Path("logs").mkdir(exist_ok=True)
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
    logger.info("🔄 RETRAINING MODELS")
    from train_probabilistic_models import main as train_main
    try:train_main();logger.info("✅ Retraining complete");return True
    except Exception as e:logger.error(f"❌ Retraining failed: {e}");return False

class VIXForecaster:
    def __init__(self):
        self.data_fetcher=UnifiedDataFetcher();self.feature_engine=UnifiedFeatureEngine(data_fetcher=self.data_fetcher);self.forecaster=SimplifiedVIXForecaster();self.calibrator=ForecastCalibrator();self.validator=TemporalValidator();self.db=PredictionDatabase();self._feature_cache=None;self._feature_cache_end_date=None;self._feature_cache_force_historical=None
        self._load_models()
    def _load_models(self):
        magnitude_file=Path("models/magnitude_5d_model.pkl");direction_file=Path("models/direction_5d_model.pkl")
        if not magnitude_file.exists()or not direction_file.exists():logger.error("❌ Models not found - run train_probabilistic_models.py first");return False
        self.forecaster.load("models");cal_file=Path("models/calibrator.pkl")
        if cal_file.exists():self.calibrator.load("models");logger.info("✅ Calibrator loaded")
        else:logger.warning("⚠️  No calibrator found - will bootstrap")
        return True
    def get_features(self,end_date,force_historical=False):
        end_ts=pd.Timestamp(end_date)
        if self._feature_cache is not None:
            cache_end_ts=pd.Timestamp(self._feature_cache_end_date)
            if self._feature_cache_end_date==end_date and self._feature_cache_force_historical==force_historical:return self._feature_cache
            if cache_end_ts>=end_ts:
                cached_df=self._feature_cache["features"]
                if end_ts<=cached_df.index[-1]:
                    filtered_features=cached_df[cached_df.index<=end_ts]
                    return{"features":filtered_features,"spx":self._feature_cache["spx"][self._feature_cache["spx"].index<=end_ts],"vix":self._feature_cache["vix"][self._feature_cache["vix"].index<=end_ts],"cboe_data":self._feature_cache.get("cboe_data"),"vvix":self._feature_cache.get("vvix")}
        feature_data=self.feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=end_date,force_historical=force_historical);self._feature_cache=feature_data;self._feature_cache_end_date=end_date;self._feature_cache_force_historical=force_historical
        return feature_data
    def needs_calibration_bootstrap(self):
        if self.calibrator.fitted:return False
        df=self.db.get_predictions(with_actuals=True)
        if len(df)>=MIN_SAMPLES_FOR_CORRECTION:return False
        return True
    def bootstrap_calibration(self):
        logger.info("🚀 BOOTSTRAPPING CALIBRATION DATA")
        cal_end=get_last_complete_month_end();cal_end_ts=pd.Timestamp(cal_end);cal_start_ts=cal_end_ts-pd.DateOffset(years=1);cal_start=cal_start_ts.strftime("%Y-%m-%d")
        logger.info(f"Calibration period: {cal_start} → {cal_end}")
        yesterday=(datetime.now()-pd.Timedelta(days=1)).strftime("%Y-%m-%d");feature_data=self.get_features(yesterday,force_historical=False);df=feature_data["features"];df=df[(df.index>=cal_start_ts)&(df.index<=cal_end_ts)]
        logger.info(f"Generating predictions for {len(df)} trading days...")
        generated=0;skipped=0
        for date in df.index:
            forecast_date=date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"]);forecast_date_str=forecast_date.strftime("%Y-%m-%d");existing=self.db.get_predictions(start_date=forecast_date_str,end_date=forecast_date_str)
            if len(existing)>0:skipped+=1;continue
            try:
                forecast=self._generate_forecast_from_df(date,feature_data["features"],calibrated=False)
                if forecast:generated+=1
                if generated%50==0:logger.info(f"  Progress: {generated} predictions...")
            except Exception:continue
        self.db.commit();logger.info(f"✅ Bootstrap: {generated} generated, {skipped} skipped")
        return generated
    def generate_current_month_forecasts(self):
        now=datetime.now();month_start=now.replace(day=1).strftime("%Y-%m-%d")
        logger.info(f"📅 GENERATING {now.strftime('%B %Y')} FORECASTS")
        yesterday=(datetime.now()-pd.Timedelta(days=1)).strftime("%Y-%m-%d");feature_data=self.get_features(yesterday,force_historical=False);df=feature_data["features"];month_start_ts=pd.Timestamp(month_start);df=df[df.index>=month_start_ts]
        logger.info(f"{now.strftime('%B')} period: {month_start} → {yesterday} ({len(df)} days)")
        generated=0;skipped=0
        for date in df.index:
            forecast_date=date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"]);forecast_date_str=forecast_date.strftime("%Y-%m-%d");existing=self.db.get_predictions(start_date=forecast_date_str,end_date=forecast_date_str)
            if len(existing)>0:skipped+=1;continue
            try:
                forecast=self._generate_forecast_from_df(date,feature_data["features"],calibrated=self.calibrator.fitted)
                if forecast:generated+=1
            except Exception:continue
        self.db.commit();logger.info(f"✅ {now.strftime('%B')}: {generated} generated, {skipped} skipped")
        return generated
    def _generate_forecast_from_df(self,date,df,calibrated=False):
        if date not in df.index:return None
        obs=df.loc[date];mag_missing=[f for f in self.forecaster.magnitude_feature_names if f not in df.columns];dir_missing=[f for f in self.forecaster.direction_feature_names if f not in df.columns]
        if mag_missing or dir_missing:logger.warning(f"Missing features");return None
        mag_vals=obs[self.forecaster.magnitude_feature_names];dir_vals=obs[self.forecaster.direction_feature_names]
        X_mag=pd.DataFrame(pd.to_numeric(mag_vals,errors="coerce").values.reshape(1,-1),columns=self.forecaster.magnitude_feature_names,dtype=np.float64).fillna(0.0)
        X_dir=pd.DataFrame(pd.to_numeric(dir_vals,errors="coerce").values.reshape(1,-1),columns=self.forecaster.direction_feature_names,dtype=np.float64).fillna(0.0)
        X=pd.DataFrame(index=[0]);[X.insert(len(X.columns),col,X_mag[col].values)for col in self.forecaster.magnitude_feature_names];[X.insert(len(X.columns),col,X_dir[col].values)for col in self.forecaster.direction_feature_names if col not in X.columns]
        current_vix=float(obs["vix"]);cohort=obs.get("calendar_cohort","mid_cycle");forecast=self.forecaster.predict(X,current_vix)
        if calibrated and self.calibrator.fitted:cal_result=self.calibrator.calibrate(forecast["magnitude_pct"],current_vix,cohort);forecast["magnitude_pct"]=cal_result["calibrated_forecast"];forecast["expected_vix"]=current_vix*(1+forecast["magnitude_pct"]/100);forecast["calibration"]=cal_result
        else:forecast["calibration"]={"correction_type":"not_fitted","adjustment":0.0}
        self._store_forecast(forecast,obs,date,calibrated=calibrated)
        return forecast
    def generate_forecast(self,date,df=None):
        target_date=pd.Timestamp(date)
        if df is None:feature_data=self.get_features(date,force_historical=False);df=feature_data["features"]
        if target_date not in df.index:logger.error(f"❌ {date} not in data");return None
        obs=df.loc[target_date];feature_dict=obs.to_dict();quality=self.validator.compute_feature_quality(feature_dict,target_date);usable,msg=self.validator.check_quality_threshold(quality)
        if not usable:logger.error(f"❌ Quality insufficient: {quality:.2f}");return None
        cohort=obs.get("calendar_cohort","mid_cycle");cohort_weight=obs.get("cohort_weight",1.0)
        mag_vals=obs[self.forecaster.magnitude_feature_names];dir_vals=obs[self.forecaster.direction_feature_names]
        X_mag=pd.DataFrame(pd.to_numeric(mag_vals,errors="coerce").values.reshape(1,-1),columns=self.forecaster.magnitude_feature_names,dtype=np.float64).fillna(0.0)
        X_dir=pd.DataFrame(pd.to_numeric(dir_vals,errors="coerce").values.reshape(1,-1),columns=self.forecaster.direction_feature_names,dtype=np.float64).fillna(0.0)
        X=pd.DataFrame(index=[0]);[X.insert(len(X.columns),col,X_mag[col].values)for col in self.forecaster.magnitude_feature_names];[X.insert(len(X.columns),col,X_dir[col].values)for col in self.forecaster.direction_feature_names if col not in X.columns]
        current_vix=float(obs["vix"]);forecast=self.forecaster.predict(X,current_vix)
        if self.calibrator.fitted:cal_result=self.calibrator.calibrate(forecast["magnitude_pct"],current_vix,cohort);forecast["magnitude_pct"]=cal_result["calibrated_forecast"];forecast["expected_vix"]=current_vix*(1+forecast["magnitude_pct"]/100);forecast["calibration"]=cal_result
        else:forecast["calibration"]={"correction_type":"not_fitted","adjustment":0.0}
        forecast_date=target_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"]);forecast["metadata"]={**{"observation_date":target_date.strftime("%Y-%m-%d"),"forecast_date":forecast_date.strftime("%Y-%m-%d"),"horizon_days":TARGET_CONFIG["horizon_days"],"feature_quality":float(quality),"cohort_weight":float(cohort_weight),"calendar_cohort":cohort,"features_used":len(self.forecaster.magnitude_feature_names)+len(self.forecaster.direction_feature_names)}};self._store_forecast(forecast,obs,target_date,calibrated=self.calibrator.fitted)
        return forecast
    def _store_forecast(self,forecast,obs,obs_date,calibrated=False):
        forecast_date=obs_date+pd.Timedelta(days=TARGET_CONFIG["horizon_days"]);pred_id=f"pred_{forecast_date.strftime('%Y%m%d')}_h{TARGET_CONFIG['horizon_days']}";conf_pct=abs(forecast["magnitude_pct"]);scale=min(conf_pct/20.0,1.0);prob_up=0.5+(0.5*scale)if forecast["direction"]=="UP"else 0.5-(0.5*scale);prob_down=1.0-prob_up;correction_type="calibrated"if calibrated else"not_fitted"
        pred={"prediction_id":pred_id,"timestamp":datetime.now(),"forecast_date":forecast_date,"observation_date":obs_date,"horizon":TARGET_CONFIG["horizon_days"],"current_vix":float(obs["vix"]),"calendar_cohort":obs.get("calendar_cohort","mid_cycle"),"cohort_weight":float(obs.get("cohort_weight",1.0)),"prob_up":float(prob_up),"prob_down":float(prob_down),"magnitude_forecast":forecast["magnitude_pct"],"expected_vix":forecast["expected_vix"],"feature_quality":float(forecast.get("metadata",{}).get("feature_quality",1.0)),"num_features_used":len(self.forecaster.magnitude_feature_names)+len(self.forecaster.direction_feature_names),"correction_type":correction_type,"features_used":(",".join(self.forecaster.magnitude_feature_names[:5])+","+",".join(self.forecaster.direction_feature_names[:5])),"model_version":"v5.3_dual","direction_probability":forecast.get("direction_probability",0.5),"direction_prediction":forecast.get("direction","UNKNOWN"),"direction_correct":None}
        self.db.store_prediction(pred);self.db.commit()

    def print_enhanced_forecast(self, forecast, cohort):
        """Enhanced forecast display showing full ensemble breakdown"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 DUAL-MODEL FORECAST BREAKDOWN")
        logger.info("=" * 80)

        # Current state
        logger.info(f"\n🎯 CURRENT STATE:")
        logger.info(f"   VIX Level: {forecast['current_vix']:.2f}")
        logger.info(f"   Regime: {forecast['current_regime']}")
        logger.info(f"   Market Cohort: {cohort}")
        logger.info(f"   Cohort Weight: {forecast.get('metadata', {}).get('cohort_weight', 1.0):.2f}x")
        logger.info(f"   Feature Quality: {forecast.get('metadata', {}).get('feature_quality', 1.0):.1%}")

        # Magnitude model - USE ACTUAL ENSEMBLE CONFIDENCE
        mag_pct = forecast['magnitude_pct']
        ens_conf = forecast['direction_confidence']  # Use actual ensemble confidence

        logger.info(f"\n📉 MAGNITUDE MODEL:")
        logger.info(f"   Forecast: {mag_pct:+.2f}%")
        logger.info(f"   Expected VIX: {forecast['expected_vix']:.2f}")
        logger.info(f"   Confidence: {ens_conf:.1%}")  # Same as ensemble, no separate calc
        if forecast['calibration']['correction_type'] != 'not_fitted':
            logger.info(f"   Calibration Adj: {forecast['calibration']['adjustment']:+.3f}% ({forecast['calibration']['correction_type']})")

        # Direction model
        dir_prob = forecast['direction_probability']
        dir_conf = max(dir_prob, 1 - dir_prob)
        logger.info(f"\n🎲 DIRECTION MODEL:")
        logger.info(f"   Prediction: {forecast['direction']}")
        logger.info(f"   Probability: {dir_prob:.1%} {'UP' if dir_prob > 0.5 else 'DOWN'}")
        logger.info(f"   Confidence: {dir_conf:.1%}")
        if self.forecaster.calibration_enabled:
            logger.info(f"   Calibration: {'✓ ENABLED' if self.forecaster.direction_calibrator is not None else '✗ NOT FITTED'}")

        # Model agreement
        predicted_up = dir_prob > 0.5
        magnitude_up = mag_pct > 0
        models_agree = predicted_up == magnitude_up

        agreement_symbol = "✓" if models_agree else "✗"
        agreement_text = "ALIGNED" if models_agree else "DIVERGENT"
        agreement_desc = "Both models agree" if models_agree else "Models disagree"

        logger.info(f"\n🔄 MODEL AGREEMENT:")
        logger.info(f"   Status: {agreement_symbol} {agreement_text}")
        logger.info(f"   {agreement_desc}")

        if not models_agree:
            logger.info(f"   → Magnitude says: {'UP' if magnitude_up else 'DOWN'}")
            logger.info(f"   → Direction says: {'UP' if predicted_up else 'DOWN'}")
            logger.info(f"   ⚠️  Confidence penalized for disagreement")

        # Ensemble confidence
        actionable = forecast['actionable']
        ens_threshold = ENSEMBLE_CONFIG.get("actionable_threshold", 0.65)

        logger.info(f"\n🎯 ENSEMBLE CONFIDENCE:")
        if self.forecaster.ensemble_enabled:
            logger.info(f"   Method: Weighted Reconciliation")
            logger.info(f"   Magnitude Weight: {ENSEMBLE_CONFIG['confidence_weights']['magnitude']:.1%}")
            logger.info(f"   Direction Weight: {ENSEMBLE_CONFIG['confidence_weights']['direction']:.1%}")
            logger.info(f"   Agreement Weight: {ENSEMBLE_CONFIG['confidence_weights']['agreement']:.1%}")
        else:
            logger.info(f"   Method: Raw Direction Confidence")
        logger.info(f"   Final Confidence: {ens_conf:.1%}")
        logger.info(f"   Actionable: {'✓ YES' if actionable else '✗ NO'} (threshold: {ens_threshold:.0%})")

        # Regime transition
        logger.info(f"\n🏛️  REGIME FORECAST:")
        logger.info(f"   Current: {forecast['current_regime']}")
        logger.info(f"   Expected: {forecast['expected_regime']}")
        if forecast['regime_change']:
            logger.info(f"   ⚠️  REGIME CHANGE EXPECTED")
        else:
            logger.info(f"   ✓ Staying in current regime")

        # Summary
        logger.info(f"\n{'=' * 80}")
        summary_color = "🟢" if actionable else "🟡"
        logger.info(f"{summary_color} SUMMARY: {forecast['direction']} {abs(mag_pct):.1f}% | Confidence: {ens_conf:.0%} | {'Actionable' if actionable else 'Not Actionable'}")
        logger.info("=" * 80 + "\n")

def main():
    parser=argparse.ArgumentParser(description="VIX Forecasting System v5.3 - Dual Model (Enhanced Display)");parser.add_argument("--force-retrain",action="store_true");parser.add_argument("--force-bootstrap",action="store_true");parser.add_argument("--rebuild-calibration",action="store_true");args=parser.parse_args();Path("logs").mkdir(exist_ok=True);Path("models").mkdir(exist_ok=True);logger.info(f"VIX FORECASTING SYSTEM v5.3 (Enhanced) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stale,reason=check_model_stale();logger.info(f"Model: {reason}")
    if stale or args.force_retrain:
        if not retrain_model():sys.exit(1)
    forecaster=VIXForecaster()
    if forecaster.needs_calibration_bootstrap()or args.force_bootstrap:
        logger.info("⚠️  Calibration data missing - bootstrapping...");forecaster.bootstrap_calibration();logger.info("📊 Backfilling actuals...");forecaster.db.backfill_actuals(forecaster.data_fetcher);logger.info("📊 Fitting calibrator...");cal=ForecastCalibrator();result=cal.fit_from_database(forecaster.db)
        if result:cal.save("models");forecaster.calibrator=cal;logger.info("✅ Calibrator fitted and saved")
        else:logger.warning("⚠️  Calibrator fitting failed")
        forecaster.generate_current_month_forecasts();logger.info("📊 Backfilling actuals...");forecaster.db.backfill_actuals(forecaster.data_fetcher)
    elif args.rebuild_calibration or not forecaster.calibrator.fitted:
        logger.info("📊 Refitting calibrator...");forecaster.db.backfill_actuals(forecaster.data_fetcher);cal=ForecastCalibrator();result=cal.fit_from_database(forecaster.db)
        if result:cal.save("models");forecaster.calibrator=cal;logger.info("✅ Calibrator refitted")
        else:logger.warning("⚠️  Calibrator fitting failed")
    yesterday=(datetime.now()-pd.Timedelta(days=1)).strftime("%Y-%m-%d");feature_data=forecaster.get_features(yesterday,force_historical=False);df=feature_data["features"];latest_available=df.index[-1].strftime("%Y-%m-%d");forecast=forecaster.generate_forecast(latest_available,df=df)
    if forecast:
        cohort=forecast.get('metadata',{}).get('calendar_cohort','unknown')
        forecaster.print_enhanced_forecast(forecast,cohort)
        total_preds=len(forecaster.db.get_predictions());with_actuals=len(forecaster.db.get_predictions(with_actuals=True));logger.info(f"📊 Database: {total_preds} predictions ({with_actuals} with actuals)");sys.exit(0)
    else:logger.error("❌ Forecast generation failed");sys.exit(1)
if __name__=="__main__":main()
