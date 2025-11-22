import json,logging,pickle
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import HuberRegressor
from config import CALIBRATION_WINDOW_DAYS,CALIBRATION_DECAY_LAMBDA,MIN_SAMPLES_FOR_CORRECTION
from core.regime_classifier import classify_vix_regime
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class ForecastCalibrator:
    def __init__(self):
        self.window_days=CALIBRATION_WINDOW_DAYS;self.decay_lambda=CALIBRATION_DECAY_LAMBDA;self.min_samples=MIN_SAMPLES_FOR_CORRECTION;self.corrections={};self.fitted=False;self.calibration_date=None
    def fit_from_database(self,database):
        logger.info("="*80);logger.info("FORECAST CALIBRATION");logger.info("="*80);logger.info("\n[1/5] Loading predictions with actuals...")
        df=database.get_predictions(with_actuals=True)
        if len(df)==0:logger.error("❌ No predictions with actuals");return False
        df["forecast_date"]=pd.to_datetime(df["forecast_date"]);df=df.sort_values("forecast_date")
        logger.info(f"  Total predictions: {len(df)}")
        logger.info(f"  Date range: {df['forecast_date'].min().date()} to {df['forecast_date'].max().date()}")
        logger.info("\n[2/5] Applying 252 trading day window...")
        latest_date=df["forecast_date"].max();trading_days=pd.bdate_range(end=latest_date,periods=self.window_days)
        if len(trading_days)==0:logger.error("❌ No trading days in window");return False
        window_start=trading_days[0];df=df[df["forecast_date"]>=window_start];self.calibration_date=latest_date
        logger.info(f"  Window: {window_start.date()} to {latest_date.date()}")
        logger.info(f"  Predictions in window: {len(df)}")
        if len(df)<self.min_samples:logger.warning(f"⚠️  Only {len(df)} samples (need {self.min_samples})");return False
        logger.info("\n[3/5] Computing exponential weights...")
        df["months_ago"]=((latest_date-df["forecast_date"]).dt.days/30.0)
        df["weight"]=np.exp(-self.decay_lambda*df["months_ago"])
        logger.info(f"  Weight range: {df['weight'].min():.3f} to {df['weight'].max():.3f}")
        logger.info(f"  Mean weight: {df['weight'].mean():.3f}")
        logger.info("\n[4/5] Classifying regimes and cohorts...")
        df["vix_regime"]=df["current_vix"].apply(lambda x:classify_vix_regime(x,numeric=False))
        df["error"]=df["actual_vix_change"]-df["magnitude_forecast"]
        regime_counts=df["vix_regime"].value_counts();cohort_counts=df["calendar_cohort"].value_counts()
        logger.info("  Regime distribution:");[logger.info(f"    {r}: {c}")for r,c in regime_counts.items()]
        logger.info("  Cohort distribution:");[logger.info(f"    {c}: {n}")for c,n in cohort_counts.items()]
        logger.info("\n[5/5] Fitting corrections...")
        self._fit_regime_cohort_corrections(df);self._fit_regime_corrections(df);self._fit_cohort_corrections(df);self._fit_global_correction(df)
        self.fitted=True;logger.info("\n✅ Calibration complete");logger.info("="*80+"\n");return True
    def _fit_regime_cohort_corrections(self,df):
        logger.info("  [A] Regime+Cohort corrections:")
        regimes=df["vix_regime"].unique();cohorts=df["calendar_cohort"].unique();self.corrections["regime_cohort"]={}
        for regime in regimes:
            for cohort in cohorts:
                key=f"{regime}_{cohort}";subset=df[(df["vix_regime"]==regime)&(df["calendar_cohort"]==cohort)]
                if len(subset)<self.min_samples:continue
                bias=np.average(subset["error"],weights=subset["weight"]);self.corrections["regime_cohort"][key]=float(bias)
                logger.info(f"      {key}: N={len(subset)}, bias={bias:+.3f}%")
        logger.info(f"    Total: {len(self.corrections['regime_cohort'])} combinations")
    def _fit_regime_corrections(self,df):
        logger.info("  [B] Regime-only corrections:")
        regimes=df["vix_regime"].unique();self.corrections["regime"]={}
        for regime in regimes:
            subset=df[df["vix_regime"]==regime]
            if len(subset)<self.min_samples:continue
            bias=np.average(subset["error"],weights=subset["weight"]);self.corrections["regime"][regime]=float(bias)
            logger.info(f"      {regime}: N={len(subset)}, bias={bias:+.3f}%")
    def _fit_cohort_corrections(self,df):
        logger.info("  [C] Cohort-only corrections:")
        cohorts=df["calendar_cohort"].unique();self.corrections["cohort"]={}
        for cohort in cohorts:
            subset=df[df["calendar_cohort"]==cohort]
            if len(subset)<self.min_samples:continue
            bias=np.average(subset["error"],weights=subset["weight"]);self.corrections["cohort"][cohort]=float(bias)
            logger.info(f"      {cohort}: N={len(subset)}, bias={bias:+.3f}%")
    def _fit_global_correction(self,df):
        logger.info("  [D] Global correction:")
        bias=np.average(df["error"],weights=df["weight"]);self.corrections["global"]=float(bias)
        logger.info(f"      All: N={len(df)}, bias={bias:+.3f}%")
    def calibrate(self,raw_forecast,current_vix,cohort):
        if not self.fitted:logger.warning("⚠️  Calibrator not fitted");return{"calibrated_forecast":raw_forecast,"adjustment":0.0,"correction_type":"not_fitted","raw_forecast":raw_forecast}
        regime=classify_vix_regime(current_vix,numeric=False);key_rc=f"{regime}_{cohort}"
        if key_rc in self.corrections.get("regime_cohort",{}):adj=self.corrections["regime_cohort"][key_rc];ctype="regime_cohort"
        elif regime in self.corrections.get("regime",{}):adj=self.corrections["regime"][regime];ctype="regime"
        elif cohort in self.corrections.get("cohort",{}):adj=self.corrections["cohort"][cohort];ctype="cohort"
        else:adj=self.corrections.get("global",0.0);ctype="global"
        calibrated=raw_forecast+adj
        return{"calibrated_forecast":calibrated,"adjustment":adj,"correction_type":ctype,"raw_forecast":raw_forecast}
    def save(self,output_dir="models"):
        if not self.fitted:logger.error("❌ Cannot save unfitted calibrator");return False
        output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True);cal_file=output_path/"calibrator.pkl";data={"corrections":self.corrections,"window_days":self.window_days,"decay_lambda":self.decay_lambda,"min_samples":self.min_samples,"calibration_date":self.calibration_date.isoformat()if self.calibration_date else None,"fitted":self.fitted}
        with open(cal_file,"wb")as f:pickle.dump(data,f)
        logger.info(f"✅ Saved: {cal_file}")
        diag_file=output_path/"calibrator_diagnostics.json";diag={"fitted":self.fitted,"calibration_date":self.calibration_date.isoformat()if self.calibration_date else None,"window_days":self.window_days,"decay_lambda":self.decay_lambda,"min_samples":self.min_samples,"corrections":self.corrections}
        with open(diag_file,"w")as f:json.dump(diag,f,indent=2)
        logger.info(f"✅ Saved: {diag_file}");return True
    def load(self,input_dir="models"):
        cal_file=Path(input_dir)/"calibrator.pkl"
        if not cal_file.exists():logger.error(f"❌ Not found: {cal_file}");return False
        with open(cal_file,"rb")as f:data=pickle.load(f)
        self.corrections=data["corrections"];self.window_days=data["window_days"];self.decay_lambda=data["decay_lambda"];self.min_samples=data["min_samples"];self.calibration_date=pd.Timestamp(data["calibration_date"])if data["calibration_date"]else None;self.fitted=data["fitted"]
        logger.info(f"✅ Loaded: {cal_file}");return True
    @classmethod
    def load_from_file(cls,input_dir="models"):
        cal=cls()
        return cal if cal.load(input_dir)else None
