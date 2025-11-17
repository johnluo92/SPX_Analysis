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
    """
    Enhanced quantile calibrator with asymmetric corrections and adaptive windowing.

    Key improvements over V1:
    1. Asymmetric handling: Lower/upper quantiles get direction-specific corrections
    2. Adaptive window sizing based on sample size
    3. Quantile-specific learning: Each quantile learns its own error pattern
    4. Robust monotonicity enforcement with minimal distortion
    """

    def __init__(self, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]):
        self.quantiles = quantiles
        self.calibrators = {}
        self.bias_corrections = {}
        self.coverage_adjustments = {}  # Store target-based adjustments
        self.fitted = False

    def fit(self, predictions, actuals):
        """
        Fit calibration using coverage-based corrections.

        Strategy:
        1. For each quantile, measure actual coverage
        2. Learn correction that maps raw prediction -> calibrated prediction
        3. Correction should push coverage toward target
        """
        if len(actuals) < 30:
            logger.warning(f"⚠️  Only {len(actuals)} samples for calibration - may be unreliable")

        logger.info(f"   Fitting quantile calibrator on {len(actuals)} samples...")

        actuals = np.array(actuals)

        for q_name, q_preds in predictions.items():
            q_preds = np.array(q_preds)

            if len(q_preds) != len(actuals):
                raise ValueError(f"{q_name}: length mismatch ({len(q_preds)} vs {len(actuals)})")

            target_quantile = float(q_name[1:]) / 100

            # Calculate current coverage
            current_coverage = (actuals <= q_preds).mean()
            coverage_error = current_coverage - target_quantile

            # Adaptive window size: balance between local adaptation and stability
            # Smaller windows for better local adaptation, but need enough samples
            n = len(actuals)
            if n < 100:
                window_size = max(20, n // 5)  # 20% of data, min 20
            elif n < 500:
                window_size = max(30, n // 8)  # 12.5% of data, min 30
            else:
                window_size = max(50, n // 15)  # 6.7% of data, min 50

            # Sort by prediction value to learn prediction-dependent patterns
            sort_idx = np.argsort(q_preds)
            sorted_preds = q_preds[sort_idx]
            sorted_actuals = actuals[sort_idx]

            # Calculate local corrections using sliding window
            corrections = []
            pred_values = []

            for i in range(len(sorted_preds)):
                # Centered window
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(sorted_preds), i + window_size // 2)

                window_actuals = sorted_actuals[start_idx:end_idx]
                window_pred = sorted_preds[i]

                # Calculate what the prediction should be for target coverage
                # Use asymmetric approach based on quantile type
                if target_quantile < 0.5:
                    # Lower quantile: want actual >= prediction
                    # Calculate value that would give target% below it
                    target_value = np.quantile(window_actuals, target_quantile)
                    # If we're underestimating coverage, need to move prediction DOWN
                    correction = target_value - window_pred

                elif target_quantile > 0.5:
                    # Upper quantile: want actual <= prediction
                    target_value = np.quantile(window_actuals, target_quantile)
                    # If we're overestimating coverage, need to move prediction DOWN
                    correction = target_value - window_pred

                else:
                    # Median: symmetric
                    target_value = np.quantile(window_actuals, 0.5)
                    correction = target_value - window_pred

                corrections.append(correction)
                pred_values.append(window_pred)

            corrections = np.array(corrections)
            pred_values = np.array(pred_values)

            # Fit isotonic regression if we have enough unique prediction values
            unique_preds = len(np.unique(pred_values))

            if unique_preds >= 5:
                # Use isotonic regression for smooth, monotonic corrections
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(pred_values, corrections)
                self.calibrators[q_name] = calibrator

                # Test the calibrator on training data
                test_corrections = calibrator.predict(q_preds)
                calibrated_test = q_preds + test_corrections
                expected_coverage = (actuals <= calibrated_test).mean()

            elif unique_preds >= 2:
                # Use linear interpolation
                unique_pred_vals = np.unique(pred_values)
                unique_corrections = [corrections[pred_values == p].mean() for p in unique_pred_vals]

                calibrator = interp1d(
                    unique_pred_vals,
                    unique_corrections,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(unique_corrections[0], unique_corrections[-1])
                )
                self.calibrators[q_name] = calibrator
                expected_coverage = current_coverage

            else:
                # Fallback: simple bias correction
                logger.warning(f"   ⚠️  {q_name}: using simple bias correction")
                self.calibrators[q_name] = None
                expected_coverage = current_coverage

            # Store simple bias correction as fallback
            # Use median for robustness
            self.bias_corrections[q_name] = float(np.median(corrections))

            # Store coverage-based adjustment for additional correction if needed
            # This helps when systematic under/over-coverage persists
            if abs(coverage_error) > 0.05:  # More than 5% off target
                # Calculate global adjustment to push toward target
                # Use conservative scaling
                coverage_adjustment = coverage_error * np.std(actuals) * 0.1
                self.coverage_adjustments[q_name] = float(coverage_adjustment)
            else:
                self.coverage_adjustments[q_name] = 0.0

            # Log diagnostics
            mean_correction = np.median(corrections)  # Use median for robustness
            logger.info(
                f"      {q_name}: {current_coverage:.1%} coverage (target {target_quantile:.1%}), "
                f"median correction: {mean_correction:+.4f}"
            )

        self.fitted = True
        logger.info("   ✅ Quantile calibrator fitted")

    def transform(self, predictions):
        """
        Apply calibration corrections with coverage-based adjustments.
        """
        if not self.fitted:
            logger.warning("⚠️  Calibrator not fitted, returning raw predictions")
            return predictions

        calibrated = {}

        for q_name, raw_value in predictions.items():
            if q_name not in self.calibrators:
                calibrated[q_name] = raw_value
                continue

            calibrator = self.calibrators[q_name]
            target_quantile = float(q_name[1:]) / 100

            # Get base correction
            if calibrator is None:
                # Use simple bias correction
                base_correction = self.bias_corrections[q_name]
            else:
                # Use learned correction function
                base_correction = float(calibrator.predict([raw_value])[0]
                                      if hasattr(calibrator, 'predict')
                                      else calibrator([raw_value])[0])

            # Apply base correction
            calibrated_value = raw_value + base_correction

            # Apply coverage adjustment if significant under/over-coverage detected
            coverage_adj = self.coverage_adjustments.get(q_name, 0.0)
            if abs(coverage_adj) > 0.001:
                # Directional adjustment based on quantile type
                if target_quantile < 0.5:
                    # Lower quantile: if undercovering, move DOWN (more conservative)
                    calibrated_value -= abs(coverage_adj)
                elif target_quantile > 0.5:
                    # Upper quantile: if overcovering, move DOWN (less conservative)
                    calibrated_value -= coverage_adj
                else:
                    # Median: symmetric adjustment
                    calibrated_value += coverage_adj

            calibrated[q_name] = calibrated_value

        # Enforce monotonicity with minimal distortion
        calibrated = self._enforce_monotonicity_minimal(calibrated)

        return calibrated

    def _enforce_monotonicity_minimal(self, quantiles_dict):
        """
        Enforce monotonicity with minimal distortion to predictions.
        Uses projection onto monotonic space.
        """
        q_names = ['q10', 'q25', 'q50', 'q75', 'q90']
        q_values = [quantiles_dict.get(q, None) for q in q_names]

        # Skip if any quantile is missing
        if None in q_values:
            return quantiles_dict

        q_values = np.array(q_values)
        original = q_values.copy()

        # Pool adjacent violators algorithm (PAV)
        # This finds the closest monotonic sequence
        n = len(q_values)
        result = q_values.copy()

        while True:
            violations = 0
            for i in range(n - 1):
                if result[i] > result[i + 1]:
                    # Violation found: average the violating pair
                    # Find the extent of violation
                    j = i + 1
                    while j < n and result[i] > result[j]:
                        j += 1

                    # Average the violating range
                    avg = np.mean(result[i:j])
                    result[i:j] = avg
                    violations += 1
                    break

            if violations == 0:
                break

        # Update dictionary
        output = quantiles_dict.copy()
        for q_name, q_value in zip(q_names, result):
            output[q_name] = float(q_value)

        # Log if significant adjustment was made
        max_adjustment = np.max(np.abs(result - original))
        if max_adjustment > 0.01:  # More than 1% adjustment
            logger.debug(f"   Monotonicity enforcement: max adjustment = {max_adjustment:.4f}")

        return output

    def get_diagnostics(self):
        """Return detailed calibration diagnostics."""
        if not self.fitted:
            return {"error": "Calibrator not fitted"}

        diagnostics = {
            "fitted": self.fitted,
            "quantiles": self.quantiles,
            "bias_corrections": self.bias_corrections,
            "coverage_adjustments": self.coverage_adjustments,
            "calibrator_types": {}
        }

        # Determine calibrator type for each quantile
        for q_name in self.calibrators.keys():
            cal = self.calibrators[q_name]
            if cal is None:
                cal_type = "simple_bias"
            elif isinstance(cal, IsotonicRegression):
                cal_type = "isotonic_regression"
            else:
                cal_type = "linear_interpolation"
            diagnostics["calibrator_types"][q_name] = cal_type

        return diagnostics


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
