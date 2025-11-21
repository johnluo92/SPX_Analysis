import json,logging,pickle
from pathlib import Path
from typing import Dict,List,Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score,log_loss,mean_absolute_error,mean_squared_error,precision_score,recall_score
from xgboost import XGBRegressor
from config import TARGET_CONFIG,XGBOOST_CONFIG,TRAINING_END_DATE,REGIME_BOUNDARIES,REGIME_NAMES
from core.target_calculator import TargetCalculator
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class SimplifiedVIXForecaster:
    def __init__(self):
        self.horizon=TARGET_CONFIG["horizon_days"];self.magnitude_model=None;self.feature_names=None;self.metrics={};self.target_calculator=TargetCalculator()
    def _get_regime(self,vix_level):
        for i,boundary in enumerate(REGIME_BOUNDARIES[1:]):
            if vix_level<boundary:return REGIME_NAMES[i]
        return REGIME_NAMES[len(REGIME_NAMES)-1]
    def train(self,df,selected_features=None,save_dir="models"):
        df=self.target_calculator.calculate_all_targets(df,vix_col="vix")
        validation=self.target_calculator.validate_targets(df)
        if not validation["valid"]:logger.error(f"❌ Target validation failed: {validation['warnings']}");raise ValueError("Invalid targets")
        for warning in validation["warnings"]:logger.warning(f"  ⚠️  {warning}")
        stats=validation["stats"];logger.info(f"  Valid targets: {stats['count']} | UP: {df['target_direction'].sum()} ({df['target_direction'].mean():.1%}) | DOWN: {len(df)-df['target_direction'].sum()}")
        X,feature_names=self._prepare_features(df,selected_features);self.feature_names=feature_names
        training_end=pd.Timestamp(TRAINING_END_DATE)
        if df.index.max()>training_end:
            train_mask=df.index<=training_end;X_train_val=X[train_mask];X_test=X[~train_mask]
            y_direction_train_val=df.loc[train_mask,"target_direction"];y_direction_test=df.loc[~train_mask,"target_direction"]
            y_magnitude_train_val=df.loc[train_mask,"target_log_vix_change"];y_magnitude_test=df.loc[~train_mask,"target_log_vix_change"]
        else:
            test_idx=int(len(df)*0.85);train_val_mask=pd.Series(False,index=df.index);train_val_mask.iloc[:test_idx]=True
            X_train_val=X[train_val_mask];X_test=X[~train_val_mask]
            y_direction_train_val=df.loc[train_val_mask,"target_direction"];y_direction_test=df.loc[~train_val_mask,"target_direction"]
            y_magnitude_train_val=df.loc[train_val_mask,"target_log_vix_change"];y_magnitude_test=df.loc[~train_val_mask,"target_log_vix_change"]
        val_split=int(len(X_train_val)*0.825);X_train=X_train_val[:val_split];X_val=X_train_val[val_split:]
        y_direction_train=y_direction_train_val.iloc[:val_split];y_direction_val=y_direction_train_val.iloc[val_split:]
        y_magnitude_train=y_magnitude_train_val.iloc[:val_split];y_magnitude_val=y_magnitude_train_val.iloc[val_split:]
        valid_train_mask=~(y_magnitude_train.isna());valid_val_mask=~(y_magnitude_val.isna());valid_test_mask=~(y_magnitude_test.isna())
        X_train=X_train[valid_train_mask];y_magnitude_train=y_magnitude_train[valid_train_mask]
        X_val=X_val[valid_val_mask];y_magnitude_val=y_magnitude_val[valid_val_mask]
        X_test=X_test[valid_test_mask];y_magnitude_test=y_magnitude_test[valid_test_mask];y_direction_test=y_direction_test[valid_test_mask]
        logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} | Features: {len(self.feature_names)}")
        self.magnitude_model,magnitude_metrics=self._train_magnitude_model(X_train,y_magnitude_train,X_val,y_magnitude_val,X_test,y_magnitude_test,y_direction_test);self.metrics["magnitude"]=magnitude_metrics
        self._save_models(save_dir);self._generate_diagnostics(X_test,y_direction_test,y_magnitude_test,save_dir,df.loc[X_test.index,"vix"])
        logger.info("✅ Training complete");self._print_summary()
        return self
    def _prepare_features(self,df,selected_features=None):
        exclude_cols=["vix","spx","calendar_cohort","cohort_weight","feature_quality","future_vix","target_vix_pct_change","target_log_vix_change","target_direction"]
        cohort_features=["is_fomc_period","is_opex_week","is_earnings_heavy"]
        if selected_features is not None:
            feature_cols=[f for f in selected_features if f in df.columns and f not in exclude_cols]
            missing_cohorts=[cf for cf in cohort_features if cf not in feature_cols and cf in df.columns]
            if missing_cohorts:logger.warning(f"  Cohort features missing from selection: {missing_cohorts}")
        else:
            logger.info("  Using all available features")
            all_cols=df.columns.tolist();feature_cols=[c for c in all_cols if c not in exclude_cols]
        feature_cols=list(dict.fromkeys(feature_cols))
        for cf in cohort_features:
            if cf not in df.columns:logger.warning(f"  Missing cohort feature: {cf}, setting to 0");df[cf]=0
        X=df[feature_cols].copy();cohort_present=[cf for cf in cohort_features if cf in feature_cols]
        logger.info(f"  Total features: {len(feature_cols)} | Cohort features: {cohort_present}")
        return X,feature_cols
    def _train_magnitude_model(self,X_train,y_train,X_val,y_val,X_test,y_test,y_direction_test):
        params=XGBOOST_CONFIG["magnitude_params"].copy()
        model=XGBRegressor(**params);model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
        y_train_pred=model.predict(X_train);y_val_pred=model.predict(X_val);y_test_pred=model.predict(X_test)
        train_pct_actual=(np.exp(y_train)-1)*100;train_pct_pred=(np.exp(y_train_pred)-1)*100
        val_pct_actual=(np.exp(y_val)-1)*100;val_pct_pred=(np.exp(y_val_pred)-1)*100
        test_pct_actual=(np.exp(y_test)-1)*100;test_pct_pred=(np.exp(y_test_pred)-1)*100
        direction_from_mag=(y_test_pred>0).astype(int);direction_acc=accuracy_score(y_direction_test,direction_from_mag);direction_prec=precision_score(y_direction_test,direction_from_mag,zero_division=0);direction_rec=recall_score(y_direction_test,direction_from_mag,zero_division=0)
        metrics={"train":{"mae_log":float(mean_absolute_error(y_train,y_train_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_train,y_train_pred))),"mae_pct":float(mean_absolute_error(train_pct_actual,train_pct_pred)),"bias_pct":float(np.mean(train_pct_pred-train_pct_actual))},"val":{"mae_log":float(mean_absolute_error(y_val,y_val_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_val,y_val_pred))),"mae_pct":float(mean_absolute_error(val_pct_actual,val_pct_pred)),"bias_pct":float(np.mean(val_pct_pred-val_pct_actual))},"test":{"mae_log":float(mean_absolute_error(y_test,y_test_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_test,y_test_pred))),"mae_pct":float(mean_absolute_error(test_pct_actual,test_pct_pred)),"bias_pct":float(np.mean(test_pct_pred-test_pct_actual)),"direction_accuracy":float(direction_acc),"direction_precision":float(direction_prec),"direction_recall":float(direction_rec)}}
        logger.info(f"  Train MAE: {metrics['train']['mae_pct']:.2f}% | Val MAE: {metrics['val']['mae_pct']:.2f}% | Test MAE: {metrics['test']['mae_pct']:.2f}%")
        logger.info(f"  Direction from magnitude - Test Acc: {direction_acc:.1%} | Prec: {direction_prec:.1%} | Rec: {direction_rec:.1%}")
        return model,metrics
    def predict(self,X,current_vix):
        X_features=X[self.feature_names]
        magnitude_log=float(self.magnitude_model.predict(X_features)[0]);magnitude_pct=(np.exp(magnitude_log)-1)*100;magnitude_pct=np.clip(magnitude_pct,-50,100)
        expected_vix=current_vix*(1+magnitude_pct/100);direction="UP"if magnitude_pct>0 else"DOWN";confidence=abs(magnitude_pct)
        current_regime=self._get_regime(current_vix);expected_regime=self._get_regime(expected_vix);regime_change=current_regime!=expected_regime
        regime_threshold=XGBOOST_CONFIG["regime_thresholds"].get(current_regime,{}).get("threshold",8.0);actionable=abs(magnitude_pct)>regime_threshold
        return{"magnitude_pct":float(magnitude_pct),"magnitude_log":float(magnitude_log),"expected_vix":float(expected_vix),"current_vix":float(current_vix),"direction":direction,"confidence":float(confidence),"current_regime":current_regime,"expected_regime":expected_regime,"regime_change":regime_change,"actionable":actionable}
    def _save_models(self,save_dir):
        save_path=Path(save_dir);save_path.mkdir(parents=True,exist_ok=True)
        magnitude_file=save_path/"magnitude_5d_model.pkl"
        with open(magnitude_file,"wb")as f:pickle.dump(self.magnitude_model,f)
        features_file=save_path/"feature_names.json"
        with open(features_file,"w")as f:json.dump(self.feature_names,f,indent=2)
        metrics_file=save_path/"training_metrics.json"
        with open(metrics_file,"w")as f:json.dump(self.metrics,f,indent=2)
        logger.info(f"  Saved models to: {save_dir}/")
    def load(self,models_dir="models"):
        models_path=Path(models_dir);magnitude_file=models_path/"magnitude_5d_model.pkl"
        with open(magnitude_file,"rb")as f:self.magnitude_model=pickle.load(f)
        features_file=models_path/"feature_names.json"
        with open(features_file,"r")as f:self.feature_names=json.load(f)
        logger.info(f"✅ Loaded magnitude model from: {models_dir} ({len(self.feature_names)} features)")
    def _generate_diagnostics(self,X_test,y_direction_test,y_magnitude_test,save_dir,vix_test):
        save_path=Path(save_dir);save_path.mkdir(parents=True,exist_ok=True)
        try:
            fig,axes=plt.subplots(2,2,figsize=(14,10));fig.suptitle("Magnitude Model Performance Diagnostics",fontsize=16,fontweight="bold")
            ax=axes[0,0];y_mag_pred=self.magnitude_model.predict(X_test);test_pct_actual=(np.exp(y_magnitude_test)-1)*100;test_pct_pred=(np.exp(y_mag_pred)-1)*100
            ax.scatter(test_pct_pred,test_pct_actual,alpha=0.5,s=30);lims=[min(test_pct_pred.min(),test_pct_actual.min()),max(test_pct_pred.max(),test_pct_actual.max())]
            ax.plot(lims,lims,"k--",alpha=0.5);ax.set_xlabel("Predicted VIX Change (%)");ax.set_ylabel("Actual VIX Change (%)");ax.set_title("Magnitude Forecast Accuracy");ax.grid(True,alpha=0.3)
            ax=axes[0,1];errors=test_pct_pred-test_pct_actual;ax.hist(errors,bins=30,alpha=0.7,edgecolor="black")
            ax.axvline(0,color="red",linestyle="--",linewidth=2,label="Zero error");ax.axvline(errors.mean(),color="blue",linestyle="--",linewidth=2,label=f"Mean: {errors.mean():.2f}%")
            ax.set_xlabel("Prediction Error (%)");ax.set_ylabel("Frequency");ax.set_title("Magnitude Error Distribution");ax.legend();ax.grid(True,alpha=0.3)
            ax=axes[1,0];direction_from_mag=(y_mag_pred>0).astype(int);bins=[0,0.4,0.6,1.0];labels=["<40%","40-60%",">60%"];X_test_copy=X_test.copy()
            conf_bins=pd.cut(np.abs(test_pct_pred)/100,bins=[0,0.05,0.10,1.0],labels=["Low<5%","Med 5-10%","High>10%"])
            accs=[]
            for label in["Low<5%","Med 5-10%","High>10%"]:
                mask=conf_bins==label
                if mask.sum()>0:acc=accuracy_score(y_direction_test[mask],direction_from_mag[mask]);accs.append(acc)
                else:accs.append(0)
            ax.bar(range(3),accs,alpha=0.7);ax.set_xticks(range(3));ax.set_xticklabels(["Low<5%","Med 5-10%","High>10%"])
            ax.set_ylabel("Direction Accuracy");ax.set_title("Direction Accuracy by Magnitude Confidence");ax.grid(True,alpha=0.3,axis="y")
            ax=axes[1,1];regimes=vix_test.apply(self._get_regime);regime_names=["Low Vol","Normal","Elevated","Crisis"];regime_maes=[]
            for regime in regime_names:
                mask=regimes==regime
                if mask.sum()>5:mae=mean_absolute_error(test_pct_actual[mask],test_pct_pred[mask]);regime_maes.append(mae)
                else:regime_maes.append(0)
            ax.bar(range(len(regime_names)),regime_maes,alpha=0.7);ax.set_xticks(range(len(regime_names)));ax.set_xticklabels(regime_names,rotation=45)
            ax.set_ylabel("MAE (%)");ax.set_title("Performance by VIX Regime");ax.grid(True,alpha=0.3,axis="y")
            plt.tight_layout();plot_file=save_path/"model_diagnostics.png";plt.savefig(plot_file,dpi=150,bbox_inches="tight");plt.close()
            logger.info(f"  Saved diagnostics: {plot_file}")
        except Exception as e:logger.warning(f"  Could not generate plots: {e}")
    def _print_summary(self):
        print("\n"+"="*60);print("TRAINING SUMMARY");print(f"Model: Magnitude Regressor (Direction Derived)");print(f"Features: {len(self.feature_names)} (including cohort flags)")
        print(f"\nMagnitude Performance:");print(f"  Train MAE: {self.metrics['magnitude']['train']['mae_pct']:.2f}%");print(f"  Val MAE:   {self.metrics['magnitude']['val']['mae_pct']:.2f}%");print(f"  Test MAE:  {self.metrics['magnitude']['test']['mae_pct']:.2f}%");print(f"  Test Bias: {self.metrics['magnitude']['test']['bias_pct']:+.2f}%")
        print(f"\nDirection (Derived from Magnitude):");print(f"  Test Acc:  {self.metrics['magnitude']['test']['direction_accuracy']:.1%}");print(f"  Test Prec: {self.metrics['magnitude']['test']['direction_precision']:.1%}");print(f"  Test Rec:  {self.metrics['magnitude']['test']['direction_recall']:.1%}")
        print("="*60+"\n")
def train_simplified_forecaster(df,selected_features=None,save_dir="models"):
    forecaster=SimplifiedVIXForecaster();forecaster.train(df,selected_features=selected_features,save_dir=save_dir)
    return forecaster
