import json,logging,pickle,warnings
from pathlib import Path
from typing import Dict,Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score,log_loss,mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier,XGBRegressor
from config import TARGET_CONFIG,XGBOOST_CONFIG
warnings.filterwarnings("ignore");logging.basicConfig(level=logging.INFO);logger=logging.getLogger(__name__)
class ProbabilisticVIXForecaster:
    def __init__(self):
        self.horizon=TARGET_CONFIG["horizon_days"];self.quantiles=TARGET_CONFIG["quantiles"]["levels"];self.models={};self.calibrators={};self.feature_names=None;self.vix_floor=10.0;self.vix_ceiling=90.0
    def train(self,df:pd.DataFrame,save_dir:str="models"):
        logger.info("="*80);logger.info("REFACTORED QUANTILE REGRESSION TRAINING");logger.info("="*80)
        if "calendar_cohort" not in df.columns:raise ValueError("Missing calendar_cohort column")
        df=self._create_targets(df)


        exclude_cols = [
            "vix",
            "spx",
            "calendar_cohort",
            "feature_quality",
            "cohort_weight",  # Added if exists
            "future_vix",  # Future VIX level
            "target_vix_pct_change",  # VIX % change target
            "target_direction",
            "target_confidence",
        ]

        feature_cols=[c for c in df.columns if c not in exclude_cols];self.feature_names=feature_cols;cohorts=sorted(df["calendar_cohort"].unique());logger.info(f"\nTraining {len(cohorts)} cohorts: {cohorts}");cohort_metrics={}
        for cohort in cohorts:
            cohort_df=df[df["calendar_cohort"]==cohort].copy();logger.info(f"\n{'─'*80}");logger.info(f"Cohort: {cohort} ({len(cohort_df)} samples)");logger.info(f"{'─'*80}");metrics=self._train_cohort_models(cohort,cohort_df);cohort_metrics[cohort]=metrics;self._save_cohort_models(cohort,save_dir);logger.info(f"✅ Saved: {cohort}")
        self._generate_diagnostics(cohort_metrics,save_dir);logger.info(f"\n{'='*80}");logger.info("TRAINING COMPLETE");logger.info(f"{'='*80}");return self

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create VIX percentage change targets."""
        logger.info("  Target creation:")
        logger.info(f"    Total samples: {len(df)}")

        df = df.copy()

        # ✅ NEW: Calculate forward VIX percentage change directly
        # This is exactly what we evaluate against!

        future_vix_list = []
        for i in range(len(df)):
            if i + self.horizon < len(df):
                # Get VIX at horizon days ahead
                future_vix = df["vix"].iloc[i + self.horizon]
                future_vix_list.append(future_vix)
            else:
                future_vix_list.append(np.nan)

        df["future_vix"] = future_vix_list

        # ✅ PRIMARY TARGET: VIX percentage change
        df["target_vix_pct_change"] = ((df["future_vix"] - df["vix"]) / df["vix"]) * 100

        # ✅ SINGLE target for ALL quantile models
        # No log transformation needed - we're predicting % directly

        # Direction target: is VIX going up or down? (unchanged)
        future_vix_series = df["vix"].shift(-self.horizon)
        df["target_direction"] = (future_vix_series > df["vix"]).astype(int)

        # Confidence based on regime stability (unchanged logic)
        regime_volatility = df["vix"].rolling(21, min_periods=10).std()
        regime_stability = 1 / (
            1 + regime_volatility / df["vix"].rolling(21, min_periods=10).mean()
        )
        regime_stability = regime_stability.fillna(0.5)

        df["target_confidence"] = (
            0.5 * df.get("feature_quality", 0.5).fillna(0.5) + 0.5 * regime_stability
        ).clip(0, 1)

        valid_targets = (~df["target_vix_pct_change"].isna()).sum()
        logger.info(f"    Valid targets: {valid_targets}")
        logger.info(f"    NaN targets: {len(df) - valid_targets} (last {self.horizon} days)")
        logger.info("")

        return df

    def _train_cohort_models(self, cohort: str, df: pd.DataFrame) -> Dict:
        """Train quantile models and direction classifier for a single cohort."""
        df = df.copy()
        X = df[self.feature_names]

        if len(df) < 30:
            raise ValueError(f"Insufficient samples for cohort {cohort}: {len(df)}")

        self.models[cohort] = {}
        self.calibrators[cohort] = {}
        metrics = {}

        # ✅ NEW: All quantile models use VIX % change target
        y_target = df["target_vix_pct_change"]  # Direct % prediction

        # Train 5 quantile regression models
        logger.info("  Training quantile regression models...")
        for q in self.quantiles:
            q_name = f"q{int(q * 100)}"

            # ✅ Same target, different quantile_alpha
            model, metric = self._train_quantile_regressor(
                X, y_target, quantile_alpha=q
            )
            self.models[cohort][q_name] = model
            metrics[q_name] = metric
            logger.info(f"    {q_name}: MAE={metric['mae']:.4f}")

        # Train direction classifier
        logger.info("  Training direction classifier...")
        y_direction = df["target_direction"]

        model_direction, metric_direction = self._train_classifier(
            X, y_direction, num_classes=2
        )
        self.models[cohort]["direction"] = model_direction
        metrics["direction"] = metric_direction
        logger.info(f"    Accuracy: {metric_direction.get('accuracy', 0):.3f}")

        # Calibrate direction probabilities
        logger.info("  Calibrating direction probabilities...")
        calibrators = self._calibrate_probabilities(model_direction, X, y_direction)
        self.calibrators[cohort]["direction"] = calibrators

        # Train confidence model (kept for compatibility)
        logger.info("  Training confidence model...")
        y_confidence = df["target_confidence"]
        model_conf, metric_conf = self._train_regressor(
            X, y_confidence, objective="reg:squarederror", eval_metric="rmse"
        )
        self.models[cohort]["confidence"] = model_conf
        metrics["confidence"] = metric_conf
        logger.info(f"    RMSE: {metric_conf['rmse']:.3f}")

        return metrics


    def _train_quantile_regressor(self,X,y,quantile_alpha:float):
        params=XGBOOST_CONFIG["shared_params"].copy();params.update({"objective":"reg:quantileerror","quantile_alpha":quantile_alpha});n_splits,test_size=self._get_adaptive_cv_config(len(X));val_maes=[]
        if n_splits==0:
            logger.info(f"      Small cohort - training on all data without CV");valid_idx=~y.isna();X_clean=X[valid_idx];y_clean=y[valid_idx]
            if len(X_clean)<10:raise ValueError(f"Insufficient valid samples: {len(X_clean)}")
            final_model=XGBRegressor(**params);final_model.fit(X_clean,y_clean,verbose=False);y_pred=final_model.predict(X_clean);mae=mean_absolute_error(y_clean,y_pred);metrics={"mae":float(mae),"mae_std":0.0};return final_model,metrics
        tscv=TimeSeriesSplit(n_splits=n_splits,test_size=test_size)
        for train_idx,val_idx in tscv.split(X):
            X_train,X_val=X.iloc[train_idx],X.iloc[val_idx];y_train,y_val=y.iloc[train_idx],y.iloc[val_idx];valid_train=~y_train.isna();valid_val=~y_val.isna()
            if valid_train.sum()<10 or valid_val.sum()<5:continue
            X_train_clean=X_train[valid_train];y_train_clean=y_train[valid_train];X_val_clean=X_val[valid_val];y_val_clean=y_val[valid_val];model=XGBRegressor(**params);model.fit(X_train_clean,y_train_clean,eval_set=[(X_val_clean,y_val_clean)],verbose=False);y_pred=model.predict(X_val_clean);mae=mean_absolute_error(y_val_clean,y_pred);val_maes.append(mae)
        valid_idx=~y.isna();X_clean=X[valid_idx];y_clean=y[valid_idx];final_model=XGBRegressor(**params);final_model.fit(X_clean,y_clean,verbose=False);metrics={"mae":float(np.mean(val_maes))if val_maes else 0.0,"mae_std":float(np.std(val_maes))if val_maes else 0.0};return final_model,metrics
    def _train_classifier(self,X,y,num_classes:int):
        params=XGBOOST_CONFIG["shared_params"].copy();params.update({"objective":"binary:logistic","eval_metric":"logloss"});n_splits,test_size=self._get_adaptive_cv_config(len(X));val_accs=[];val_loglosses=[]
        if n_splits==0:
            logger.info(f"      Small cohort - training on all data without CV");valid_idx=~y.isna();X_clean=X[valid_idx];y_clean=y[valid_idx]
            if len(X_clean)<10:raise ValueError(f"Insufficient valid samples: {len(X_clean)}")
            final_model=XGBClassifier(**params);final_model.fit(X_clean,y_clean,verbose=False);y_pred=final_model.predict(X_clean);y_pred_proba=final_model.predict_proba(X_clean);acc=accuracy_score(y_clean,y_pred);ll=log_loss(y_clean,y_pred_proba);metrics={"accuracy":float(acc),"logloss":float(ll)};return final_model,metrics
        tscv=TimeSeriesSplit(n_splits=n_splits,test_size=test_size)
        for train_idx,val_idx in tscv.split(X):
            X_train,X_val=X.iloc[train_idx],X.iloc[val_idx];y_train,y_val=y.iloc[train_idx],y.iloc[val_idx];valid_train=~y_train.isna();valid_val=~y_val.isna()
            if valid_train.sum()<10 or valid_val.sum()<5:continue
            X_train_clean=X_train[valid_train];y_train_clean=y_train[valid_train];X_val_clean=X_val[valid_val];y_val_clean=y_val[valid_val];model=XGBClassifier(**params);model.fit(X_train_clean,y_train_clean,eval_set=[(X_val_clean,y_val_clean)],verbose=False);y_pred=model.predict(X_val_clean);y_pred_proba=model.predict_proba(X_val_clean);acc=accuracy_score(y_val_clean,y_pred);ll=log_loss(y_val_clean,y_pred_proba);val_accs.append(acc);val_loglosses.append(ll)
        valid_idx=~y.isna();X_clean=X[valid_idx];y_clean=y[valid_idx];final_model=XGBClassifier(**params);final_model.fit(X_clean,y_clean,verbose=False);metrics={"accuracy":float(np.mean(val_accs))if val_accs else 0.0,"logloss":float(np.mean(val_loglosses))if val_loglosses else 0.0};return final_model,metrics
    def _train_regressor(self,X,y,objective:str,eval_metric:str):
        params=XGBOOST_CONFIG["shared_params"].copy();params.update({"objective":objective,"eval_metric":eval_metric});n_splits,test_size=self._get_adaptive_cv_config(len(X));val_rmses=[]
        if n_splits==0:
            logger.info(f"      Small cohort - training on all data without CV");valid_idx=~y.isna();X_clean=X[valid_idx];y_clean=y[valid_idx]
            if len(X_clean)<10:raise ValueError(f"Insufficient valid samples: {len(X_clean)}")
            final_model=XGBRegressor(**params);final_model.fit(X_clean,y_clean,verbose=False);y_pred=final_model.predict(X_clean);rmse=np.sqrt(mean_absolute_error(y_clean,y_pred)**2);metrics={"rmse":float(rmse)};return final_model,metrics
        tscv=TimeSeriesSplit(n_splits=n_splits,test_size=test_size)
        for train_idx,val_idx in tscv.split(X):
            X_train,X_val=X.iloc[train_idx],X.iloc[val_idx];y_train,y_val=y.iloc[train_idx],y.iloc[val_idx];valid_train=~y_train.isna();valid_val=~y_val.isna()
            if valid_train.sum()<10 or valid_val.sum()<5:continue
            model=XGBRegressor(**params);model.fit(X_train[valid_train],y_train[valid_train],eval_set=[(X_val[valid_val],y_val[valid_val])],verbose=False);y_pred=model.predict(X_val[valid_val]);rmse=np.sqrt(mean_absolute_error(y_val[valid_val],y_pred)**2);val_rmses.append(rmse)
        valid_idx=~y.isna();final_model=XGBRegressor(**params);final_model.fit(X[valid_idx],y[valid_idx],verbose=False);metrics={"rmse":float(np.mean(val_rmses))if val_rmses else 0.0};return final_model,metrics
    def _get_adaptive_cv_config(self,n_samples:int)->Tuple[int,int]:
        if n_samples<50:return 0,0
        if n_samples<200:n_splits=2
        elif n_samples<400:n_splits=3
        elif n_samples<800:n_splits=4
        else:n_splits=5
        max_test_size=n_samples//(n_splits+1);test_size=max(int(max_test_size*0.8),10)
        while(n_samples-test_size)<n_splits*test_size and n_splits>1:n_splits-=1;max_test_size=n_samples//(n_splits+1);test_size=max(int(max_test_size*0.8),10)
        if n_splits*test_size>n_samples:n_splits=1;test_size=min(n_samples//3,20)
        return n_splits,test_size
    def _calibrate_probabilities(self,model,X,y):
        y_proba=model.predict_proba(X);calibrators=[]
        for class_idx in range(2):
            y_binary=(y==class_idx).astype(int)
            if y_binary.sum()>0 and y_binary.sum()<len(y_binary):calibrator=IsotonicRegression(out_of_bounds="clip");calibrator.fit(y_proba[:,class_idx],y_binary);calibrators.append(calibrator)
            else:calibrators.append(None)
        return calibrators

    def predict(self, X: pd.DataFrame, cohort: str, current_vix: float) -> Dict:
        """
        Generate probabilistic forecast with quantiles and direction.

        Args:
            X: Feature dataframe (single row)
            cohort: Calendar cohort to use
            current_vix: Current VIX level (for metadata only)

        Returns:
            Dictionary with quantiles, direction_probability, confidence_score
        """
        if cohort not in self.models:
            raise ValueError(
                f"Cohort {cohort} not trained. Available: {list(self.models.keys())}"
            )

        X_features = X[self.feature_names]

        # ✅ NEW: Get quantile predictions DIRECTLY as VIX % changes
        quantiles_pct = {}
        for q in self.quantiles:
            q_name = f"q{int(q * 100)}"
            # Models now output % change directly!
            pred_pct = self.models[cohort][q_name].predict(X_features)[0]
            quantiles_pct[q_name] = pred_pct

        # ✅ NEW: Enforce monotonicity in % space (not RV space)
        q_keys = ["q10", "q25", "q50", "q75", "q90"]
        q_values = [quantiles_pct[k] for k in q_keys]
        q_values_sorted = sorted(q_values)
        quantiles_pct_monotonic = dict(zip(q_keys, q_values_sorted))

        # ✅ OPTIONAL: Apply domain bounds if desired
        # VIX rarely moves more than ±50% in 5 days
        for k in q_keys:
            quantiles_pct_monotonic[k] = np.clip(
                quantiles_pct_monotonic[k], -50, 100
            )

        # Direction probability (unchanged)
        prob_up = float(
            self.models[cohort]["direction"].predict_proba(X_features)[0][1]
        )
        prob_down = 1.0 - prob_up

        # Confidence score (unchanged)
        confidence = np.clip(
            self.models[cohort]["confidence"].predict(X_features)[0], 0, 1
        )

        # Use median (q50) as primary forecast
        median_forecast = quantiles_pct_monotonic["q50"]

        return {
            "median_forecast": float(median_forecast),
            "quantiles": {k: float(v) for k, v in quantiles_pct_monotonic.items()},
            "q10": float(quantiles_pct_monotonic["q10"]),
            "q25": float(quantiles_pct_monotonic["q25"]),
            "q50": float(quantiles_pct_monotonic["q50"]),
            "q75": float(quantiles_pct_monotonic["q75"]),
            "q90": float(quantiles_pct_monotonic["q90"]),
            "prob_up": float(prob_up),
            "prob_down": float(prob_down),
            "direction_probability": float(prob_up),
            "confidence_score": float(confidence),
            "cohort": cohort,
            "metadata": {
                "current_vix": current_vix,
                "prediction_type": "vix_pct_change",
                "target_aligned": True,
            },
        }


    def load(self,cohort:str,models_dir:str):
        models_path=Path(models_dir);file_path=models_path/f"probabilistic_forecaster_{cohort}.pkl"
        if not file_path.exists():raise FileNotFoundError(f"No saved model found for cohort: {cohort}")
        with open(file_path,"rb")as f:cohort_data=pickle.load(f)
        self.models[cohort]=cohort_data["models"];self.calibrators[cohort]=cohort_data["calibrators"]
        if self.feature_names is None:self.feature_names=cohort_data["feature_names"];self.quantiles=cohort_data["quantiles"];self.horizon=cohort_data["horizon"]
    def _save_cohort_models(self,cohort:str,save_dir:str):
        save_path=Path(save_dir);save_path.mkdir(exist_ok=True,parents=True);cohort_data={"models":self.models[cohort],"calibrators":self.calibrators[cohort],"feature_names":self.feature_names,"quantiles":self.quantiles,"horizon":self.horizon};file_path=save_path/f"probabilistic_forecaster_{cohort}.pkl"
        with open(file_path,"wb")as f:pickle.dump(cohort_data,f)
    def _generate_diagnostics(self,cohort_metrics:Dict,save_dir:str):
        save_path=Path(save_dir);save_path.mkdir(exist_ok=True,parents=True);metrics_file=save_path/"training_metrics.json"
        with open(metrics_file,"w")as f:json.dump(cohort_metrics,f,indent=2)
        logger.info(f"  Saved metrics: {metrics_file}")
        try:
            cohorts=list(cohort_metrics.keys());fig,axes=plt.subplots(2,3,figsize=(15,10));fig.suptitle("Model Performance (Log-RV Quantile Regression) by Cohort",fontsize=16)
            for idx,q in enumerate([10,25,50,75,90]):
                q_name=f"q{q}";row=idx//3;col=idx%3;maes=[cohort_metrics[c][q_name]["mae"]for c in cohorts];axes[row,col].bar(range(len(cohorts)),maes,color="steelblue");axes[row,col].set_xticks(range(len(cohorts)));axes[row,col].set_xticklabels(cohorts,rotation=45,ha="right");axes[row,col].set_ylabel("MAE (log space)");axes[row,col].set_title(f"{q}th Percentile");axes[row,col].grid(True,alpha=0.3)
            dir_accs=[cohort_metrics[c]["direction"]["accuracy"]for c in cohorts];axes[1,2].bar(range(len(cohorts)),dir_accs,color="forestgreen");axes[1,2].set_xticks(range(len(cohorts)));axes[1,2].set_xticklabels(cohorts,rotation=45,ha="right");axes[1,2].set_ylabel("Accuracy");axes[1,2].set_title("Direction Classifier");axes[1,2].set_ylim([0,1]);axes[1,2].grid(True,alpha=0.3);plt.tight_layout();plot_file=save_path/"model_performance.png";plt.savefig(plot_file,dpi=150,bbox_inches="tight");plt.close();logger.info(f"  Saved plots: {plot_file}")
        except Exception as e:logger.warning(f"  Could not generate plots: {e}")
def train_probabilistic_forecaster(df:pd.DataFrame,save_dir:str="models")->ProbabilisticVIXForecaster:
    forecaster=ProbabilisticVIXForecaster();forecaster.train(df,save_dir=save_dir);return forecaster
