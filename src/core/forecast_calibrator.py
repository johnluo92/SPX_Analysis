import json,logging,pickle
from datetime import datetime
from pathlib import Path
from typing import Dict,List,Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor,LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class ForecastCalibrator:
    def __init__(self,min_samples:int=50,use_robust:bool=True,cohort_specific:bool=True,regime_specific:bool=True):
        self.min_samples=min_samples;self.use_robust=use_robust;self.cohort_specific=cohort_specific;self.regime_specific=regime_specific;self.global_model=None;self.cohort_models={};self.regime_models={};self.calibration_stats={};self.fitted=False
        logger.info(f"ForecastCalibrator V3 initialized:\n  Min samples: {min_samples}\n  Robust regression: {use_robust}\n  Cohort-specific: {cohort_specific}\n  Regime-specific: {regime_specific}")
    def fit_from_database(self,database,start_date:Optional[str]=None,end_date:Optional[str]=None)->bool:
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

    def calibrate(
        self,
        raw_forecast: float,
        current_vix: float,
        cohort: Optional[str] = None,
    ) -> Dict:
        """
        Apply calibration - TEMPORARILY DISABLED pending retargeting.

        The bias correction was treating symptoms of the target mismatch problem.
        Once models are retrained with aligned targets (VIX % change), we'll
        reassess whether calibration provides meaningful improvement.

        Args:
            raw_forecast: Raw median forecast from model (%)
            current_vix: Current VIX level
            cohort: Calendar cohort (optional)

        Returns:
            Dict with raw forecast (no adjustment applied)
        """

        if not self.fitted:
            logger.debug("⚠️  Calibrator not fitted - using raw forecasts")
        else:
            logger.debug("⚠️  Calibration temporarily disabled - using raw forecasts until models retrained")

        return {
            "calibrated_forecast": raw_forecast,
            "adjustment": 0.0,
            "method": "disabled_pending_retrain",
            "raw_forecast": raw_forecast,
            "note": "Calibration disabled - target mismatch fix in progress",
        }


    def get_diagnostics(self)->Dict:
        if not self.fitted:return{"error":"Calibrator not fitted"}
        diagnostics={"fitted":self.fitted,"timestamp":datetime.now().isoformat(),"config":{"min_samples":self.min_samples,"use_robust":self.use_robust,"cohort_specific":self.cohort_specific,"regime_specific":self.regime_specific},"statistics":self.calibration_stats,"models":{"global":self.global_model is not None,"cohorts":list(self.cohort_models.keys()),"regimes":list(self.regime_models.keys())}}
        if"global"in self.calibration_stats:global_stats=self.calibration_stats["global"];diagnostics["overall_improvement"]=global_stats.get("improvement_pct",0);diagnostics["training_samples"]=global_stats.get("samples",0);diagnostics["bias_correction"]=global_stats.get("mean_error",0)
        return diagnostics
    def save_calibrator(self,output_dir:str="models"):
        if not self.fitted:logger.error("❌ Cannot save unfitted calibrator");return
        output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True);calibrator_file=output_path/"forecast_calibrator.pkl"
        calibrator_data={"global_model":self.global_model,"cohort_models":self.cohort_models,"regime_models":self.regime_models,"config":{"min_samples":self.min_samples,"use_robust":self.use_robust,"cohort_specific":self.cohort_specific,"regime_specific":self.regime_specific},"fitted":self.fitted}
        with open(calibrator_file,"wb")as f:pickle.dump(calibrator_data,f)
        logger.info(f"✅ Saved calibrator: {calibrator_file}")
        diagnostics_file=output_path/"calibrator_diagnostics.json";diagnostics=self.get_diagnostics()
        with open(diagnostics_file,"w")as f:json.dump(diagnostics,f,indent=2,default=str)
        logger.info(f"✅ Saved diagnostics: {diagnostics_file}")
    def load_calibrator(self,input_dir:str="models"):
        calibrator_file=Path(input_dir)/"forecast_calibrator.pkl"
        if not calibrator_file.exists():logger.error(f"❌ Calibrator file not found: {calibrator_file}");return False
        try:
            with open(calibrator_file,"rb")as f:calibrator_data=pickle.load(f)
            self.global_model=calibrator_data["global_model"];self.cohort_models=calibrator_data["cohort_models"];self.regime_models=calibrator_data["regime_models"];config=calibrator_data["config"];self.min_samples=config["min_samples"];self.use_robust=config["use_robust"];self.cohort_specific=config["cohort_specific"];self.regime_specific=config["regime_specific"];self.fitted=calibrator_data["fitted"]
            logger.info(f"✅ Loaded calibrator from {calibrator_file}")
            return True
        except Exception as e:logger.error(f"❌ Failed to load calibrator: {e}");return False
    @classmethod
    def load(cls,input_dir:str="models")->Optional["ForecastCalibrator"]:
        calibrator=cls();return calibrator if calibrator.load_calibrator(input_dir)else None
def test_calibrator():
    print("\n"+"="*80+"\nTESTING FORECAST CALIBRATOR V3\n"+"="*80)
    np.random.seed(42);n_samples=200;dates=pd.date_range("2020-01-01",periods=n_samples,freq="D");true_changes=np.random.randn(n_samples)*5;forecasts=true_changes+1.0+np.random.randn(n_samples)*2;vix_levels=15+np.random.randn(n_samples)*5;vix_levels=np.clip(vix_levels,10,40)
    df=pd.DataFrame({"forecast_date":dates,"median_forecast":forecasts,"actual_vix_change":true_changes,"current_vix":vix_levels,"calendar_cohort":np.random.choice(["start_month","mid_month","end_month"],n_samples)})
    class MockDatabase:
        def get_predictions(self,with_actuals=False):return df
    mock_db=MockDatabase();calibrator=ForecastCalibrator(min_samples=50,use_robust=True,cohort_specific=True,regime_specific=True);success=calibrator.fit_from_database(mock_db)
    if success:
        print("\n✅ Calibrator fitted successfully");diag=calibrator.get_diagnostics();print(f"✅ Overall improvement: {diag['overall_improvement']:+.1f}%");print(f"✅ Bias correction: {diag['bias_correction']:+.2f}%")
        result=calibrator.calibrate(raw_forecast=2.5,current_vix=18.0,cohort="mid_month");print(f"\n✅ Test calibration:");print(f"   Raw: {result['raw_forecast']:+.2f}%");print(f"   Calibrated: {result['calibrated_forecast']:+.2f}%");print(f"   Adjustment: {result['adjustment']:+.2f}%");print(f"   Method: {result['method']}")
        calibrator.save_calibrator(output_dir="/home/claude/test_output");print(f"\n✅ Calibrator saved")
    else:print("\n❌ Calibrator fitting failed")
    print("\n"+"="*80+"\nTEST COMPLETE\n"+"="*80)
if __name__=="__main__":test_calibrator()
