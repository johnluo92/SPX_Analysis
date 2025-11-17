import json,logging,pickle
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor,LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class QuantileCalibrator:
    def __init__(self,quantiles=[0.10,0.25,0.50,0.75,0.90]):
        self.quantiles=quantiles;self.calibrators={};self.fitted=False
    def fit(self,predictions,actuals):
        if len(actuals)<30:logger.warning(f"⚠️  Only {len(actuals)} samples for calibration - may be unreliable")
        logger.info(f"   Fitting quantile calibrator on {len(actuals)} samples...")
        for q_name,q_preds in predictions.items():
            if len(q_preds)!=len(actuals):raise ValueError(f"{q_name}: length mismatch ({len(q_preds)} vs {len(actuals)})")
            empirical_quantiles=[]
            for pred in q_preds:
                empirical_q=(actuals<=pred).mean()
                empirical_quantiles.append(empirical_q)
            empirical_quantiles=np.array(empirical_quantiles)
            calibrator=IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(q_preds,empirical_quantiles)
            self.calibrators[q_name]=calibrator
            target_quantile=float(q_name[1:])/100
            actual_coverage=(actuals<=q_preds).mean()
            logger.info(f"      {q_name}: {actual_coverage:.1%} coverage (target {target_quantile:.1%})")
        self.fitted=True
        logger.info("   ✅ Quantile calibrator fitted")
    def transform(self,predictions):
        if not self.fitted:logger.warning("⚠️  Calibrator not fitted, returning raw predictions");return predictions
        calibrated={}
        for q_name,q_value in predictions.items():
            if q_name not in self.calibrators:calibrated[q_name]=q_value;continue
            calibrator=self.calibrators[q_name]
            target_quantile=float(q_name[1:])/100
            try:pred_min=calibrator.X_min_;pred_max=calibrator.X_max_
            except AttributeError:pred_min=np.min(calibrator.X_thresholds_);pred_max=np.max(calibrator.X_thresholds_)
            def coverage_at_value(val):return calibrator.predict([val])[0]
            search_min=pred_min-abs(pred_min)*0.2;search_max=pred_max+abs(pred_max)*0.2;tol=0.001;max_iter=50
            for _ in range(max_iter):
                mid=(search_min+search_max)/2
                mid_coverage=coverage_at_value(mid)
                if abs(mid_coverage-target_quantile)<tol:break
                if mid_coverage<target_quantile:search_min=mid
                else:search_max=mid
            calibrated_value=mid
            calibrated[q_name]=calibrated_value
        return calibrated
class ForecastCalibrator:
    def __init__(self,min_samples=50,use_robust=True,cohort_specific=True,regime_specific=True):
        self.min_samples=min_samples;self.use_robust=use_robust;self.cohort_specific=cohort_specific;self.regime_specific=regime_specific;self.global_model=None;self.cohort_models={};self.regime_models={};self.calibration_stats={};self.fitted=False
        logger.info(f"ForecastCalibrator V3 initialized:\n  Min samples: {min_samples}\n  Robust regression: {use_robust}\n  Cohort-specific: {cohort_specific}\n  Regime-specific: {regime_specific}")
    def fit_from_database(self,database,start_date=None,end_date=None):
        logger.info("\n"+"="*80+"\nFORECAST CALIBRATION\n"+"="*80)
        logger.info("\n[1/5] Loading historical predictions...")
        df=database.get_predictions(with_actuals=True)
        if start_date:df=df[df["forecast_date"]>=pd.Timestamp(start_date)];logger.info(f"  Filtered to dates >= {start_date}")
        if end_date:df=df[df["forecast_date"]<=pd.Timestamp(end_date)];logger.info(f"  Filtered to dates <= {end_date}")
        if len(df)==0:logger.error("❌ No predictions with actuals available");return False
        if len(df)<self.min_samples:logger.warning(f"⚠️  Only {len(df)} samples available, need {self.min_samples}\n   Calibration may be unreliable")
        logger.info(f"  Loaded {len(df)} predictions\n  Date range: {df['forecast_date'].min().date()} to {df['forecast_date'].max().date()}")
        logger.info("\n[2/5] Preparing calibration data...")
        required_cols=["median_forecast","actual_vix_change","current_vix"]
        missing=[col for col in required_cols if col not in df.columns]
        if missing:logger.error(f"❌ Missing required columns: {missing}");return False
        calib_df=df[required_cols].copy()
        calib_df["cohort"]=df["calendar_cohort"]if"calendar_cohort"in df.columns and self.cohort_specific else"all"
        calib_df=calib_df.dropna();calib_df["error"]=calib_df["actual_vix_change"]-calib_df["median_forecast"];calib_df["vix_regime"]=pd.cut(calib_df["current_vix"],bins=[0,15,25,40,100],labels=["low","normal","elevated","crisis"])
        logger.info(f"  Calibration samples: {len(calib_df)}\n  Mean error: {calib_df['error'].mean():+.2f}%\n  Mean absolute error: {calib_df['error'].abs().mean():.2f}%")
        if abs(calib_df["error"].mean())>0.5:logger.warning("⚠️  Systematic underestimation detected"if calib_df["error"].mean()>0 else"⚠️  Systematic overestimation detected")
        logger.info("\n[3/5] Fitting global calibration model...")
        X_global=calib_df[["median_forecast","current_vix"]].values;y_global=calib_df["error"].values;self.global_model=HuberRegressor()if self.use_robust else LinearRegression();self.global_model.fit(X_global,y_global)
        y_pred=self.global_model.predict(X_global);mae_before=mean_absolute_error(calib_df["actual_vix_change"],calib_df["median_forecast"]);calibrated_forecast=calib_df["median_forecast"]+y_pred;mae_after=mean_absolute_error(calib_df["actual_vix_change"],calibrated_forecast);improvement_pct=(mae_before-mae_after)/mae_before*100
        logger.info(f"  MAE before calibration: {mae_before:.2f}%\n  MAE after calibration:  {mae_after:.2f}%\n  Improvement: {improvement_pct:+.1f}%")
        self.calibration_stats["global"]={"samples":len(calib_df),"mae_before":float(mae_before),"mae_after":float(mae_after),"improvement_pct":float(improvement_pct),"mean_error":float(calib_df["error"].mean())}
        if self.cohort_specific and"cohort"in calib_df.columns:
            logger.info("\n[4/5] Fitting cohort-specific calibration...")
            for cohort in calib_df["cohort"].unique():
                if pd.isna(cohort):continue
                cohort_df=calib_df[calib_df["cohort"]==cohort]
                if len(cohort_df)<20:logger.warning(f"  ⚠️  {cohort}: Only {len(cohort_df)} samples, skipping");continue
                X_cohort=cohort_df[["median_forecast","current_vix"]].values;y_cohort=cohort_df["error"].values;cohort_model=HuberRegressor()if self.use_robust else LinearRegression();cohort_model.fit(X_cohort,y_cohort);self.cohort_models[cohort]=cohort_model
                y_pred_cohort=cohort_model.predict(X_cohort);mae_before=mean_absolute_error(cohort_df["actual_vix_change"],cohort_df["median_forecast"]);calibrated=cohort_df["median_forecast"]+y_pred_cohort;mae_after=mean_absolute_error(cohort_df["actual_vix_change"],calibrated);improvement=(mae_before-mae_after)/mae_before*100
                logger.info(f"  {cohort}: {len(cohort_df)} samples, improvement: {improvement:+.1f}%")
                self.calibration_stats[f"cohort_{cohort}"]={"samples":len(cohort_df),"mae_before":float(mae_before),"mae_after":float(mae_after),"improvement_pct":float(improvement),"mean_error":float(cohort_df["error"].mean())}
        else:logger.info("\n[4/5] Cohort-specific calibration disabled")
        if self.regime_specific:
            logger.info("\n[5/5] Fitting regime-specific calibration...")
            for regime in["low","normal","elevated","crisis"]:
                regime_df=calib_df[calib_df["vix_regime"]==regime]
                if len(regime_df)<20:logger.warning(f"  ⚠️  {regime}: Only {len(regime_df)} samples, skipping");continue
                X_regime=regime_df[["median_forecast","current_vix"]].values;y_regime=regime_df["error"].values;regime_model=HuberRegressor()if self.use_robust else LinearRegression();regime_model.fit(X_regime,y_regime);self.regime_models[regime]=regime_model
                y_pred_regime=regime_model.predict(X_regime);mae_before=mean_absolute_error(regime_df["actual_vix_change"],regime_df["median_forecast"]);calibrated=regime_df["median_forecast"]+y_pred_regime;mae_after=mean_absolute_error(regime_df["actual_vix_change"],calibrated);improvement=(mae_before-mae_after)/mae_before*100
                logger.info(f"  {regime} VIX: {len(regime_df)} samples, improvement: {improvement:+.1f}%")
                self.calibration_stats[f"regime_{regime}"]={"samples":len(regime_df),"mae_before":float(mae_before),"mae_after":float(mae_after),"improvement_pct":float(improvement),"mean_error":float(regime_df["error"].mean())}
        else:logger.info("\n[5/5] Regime-specific calibration disabled")
        self.fitted=True
        logger.info("\n✅ Calibration complete\n"+"="*80+"\n")
        return True
    def calibrate(self,raw_forecast,current_vix,cohort=None):
        if not self.fitted:
            logger.debug("⚠️  Calibrator not fitted - using raw forecasts")
            return{"calibrated_forecast":raw_forecast,"adjustment":0.0,"method":"not_fitted","raw_forecast":raw_forecast}
        vix_regime=self._classify_vix_regime(current_vix)
        model=None;method="global"
        if self.cohort_specific and cohort and cohort in self.cohort_models:model=self.cohort_models[cohort];method=f"cohort_{cohort}"
        elif self.regime_specific and vix_regime in self.regime_models:model=self.regime_models[vix_regime];method=f"regime_{vix_regime}"
        else:model=self.global_model;method="global"
        if model is None:
            logger.warning("⚠️  No calibration model available - using raw forecast")
            return{"calibrated_forecast":raw_forecast,"adjustment":0.0,"method":"no_model","raw_forecast":raw_forecast}
        X=np.array([[raw_forecast,current_vix]])
        adjustment=float(model.predict(X)[0])
        calibrated_forecast=raw_forecast+adjustment
        logger.debug(f"🎯 Calibration applied: {raw_forecast:+.2f}% → {calibrated_forecast:+.2f}% (Δ{adjustment:+.2f}%) via {method}")
        return{"calibrated_forecast":calibrated_forecast,"adjustment":adjustment,"method":method,"raw_forecast":raw_forecast}
    def _classify_vix_regime(self,vix):
        if vix<16.77:return"low"
        elif vix<24.40:return"normal"
        elif vix<39.67:return"elevated"
        else:return"crisis"
    def get_diagnostics(self):
        if not self.fitted:return{"error":"Calibrator not fitted"}
        diagnostics={"fitted":self.fitted,"timestamp":datetime.now().isoformat(),"config":{"min_samples":self.min_samples,"use_robust":self.use_robust,"cohort_specific":self.cohort_specific,"regime_specific":self.regime_specific},"statistics":self.calibration_stats,"models":{"global":self.global_model is not None,"cohorts":list(self.cohort_models.keys()),"regimes":list(self.regime_models.keys())}}
        if"global"in self.calibration_stats:global_stats=self.calibration_stats["global"];diagnostics["overall_improvement"]=global_stats.get("improvement_pct",0);diagnostics["training_samples"]=global_stats.get("samples",0);diagnostics["bias_correction"]=global_stats.get("mean_error",0)
        return diagnostics
    def save_calibrator(self,output_dir="models"):
        if not self.fitted:logger.error("❌ Cannot save unfitted calibrator");return
        output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True);calibrator_file=output_path/"forecast_calibrator.pkl"
        calibrator_data={"global_model":self.global_model,"cohort_models":self.cohort_models,"regime_models":self.regime_models,"config":{"min_samples":self.min_samples,"use_robust":self.use_robust,"cohort_specific":self.cohort_specific,"regime_specific":self.regime_specific},"fitted":self.fitted}
        with open(calibrator_file,"wb")as f:pickle.dump(calibrator_data,f)
        logger.info(f"✅ Saved calibrator: {calibrator_file}")
        diagnostics_file=output_path/"calibrator_diagnostics.json";diagnostics=self.get_diagnostics()
        with open(diagnostics_file,"w")as f:json.dump(diagnostics,f,indent=2,default=str)
        logger.info(f"✅ Saved diagnostics: {diagnostics_file}")
    def load_calibrator(self,input_dir="models"):
        calibrator_file=Path(input_dir)/"forecast_calibrator.pkl"
        if not calibrator_file.exists():logger.error(f"❌ Calibrator file not found: {calibrator_file}");return False
        try:
            with open(calibrator_file,"rb")as f:calibrator_data=pickle.load(f)
            self.global_model=calibrator_data["global_model"];self.cohort_models=calibrator_data["cohort_models"];self.regime_models=calibrator_data["regime_models"];config=calibrator_data["config"];self.min_samples=config["min_samples"];self.use_robust=config["use_robust"];self.cohort_specific=config["cohort_specific"];self.regime_specific=config["regime_specific"];self.fitted=calibrator_data["fitted"]
            logger.info(f"✅ Loaded calibrator from {calibrator_file}")
            return True
        except Exception as e:logger.error(f"❌ Failed to load calibrator: {e}");return False
    @classmethod
    def load(cls,input_dir="models"):
        calibrator=cls();return calibrator if calibrator.load_calibrator(input_dir)else None
