import json
import logging
import pickle
from pathlib import Path
from typing import Dict,List,Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score,log_loss,mean_absolute_error,mean_squared_error,precision_score,recall_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier,XGBRegressor
from config import TARGET_CONFIG,XGBOOST_CONFIG,TRAINING_END_DATE
from core.target_calculator import TargetCalculator
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class SimplifiedVIXForecaster:
    def __init__(self):
        self.horizon=TARGET_CONFIG["horizon_days"];self.direction_model=None;self.magnitude_model=None;self.feature_names=None;self.metrics={};self.target_calculator=TargetCalculator()
    def train(self,df,selected_features=None,save_dir="models"):
        logger.info("[1/5] Creating targets (Cohort features ready)")
        df=self.target_calculator.calculate_all_targets(df,vix_col="vix")
        validation=self.target_calculator.validate_targets(df)
        if not validation["valid"]:logger.error(f"❌ Target validation failed: {validation['warnings']}");raise ValueError("Invalid targets")
        for warning in validation["warnings"]:logger.warning(f"  ⚠️  {warning}")
        stats=validation["stats"];logger.info(f"  Valid targets: {stats['count']} | UP: {df['target_direction'].sum()} ({df['target_direction'].mean():.1%}) | DOWN: {len(df)-df['target_direction'].sum()}")
        logger.info("[2/5] Preparing feature matrix...")
        X,feature_names=self._prepare_features(df,selected_features);self.feature_names=feature_names
        logger.info("[3/5] Splitting train/test...")
        training_end=pd.Timestamp(TRAINING_END_DATE)
        if df.index.max()>training_end:train_mask=df.index<=training_end
        else:split_idx=int(len(df)*0.8);train_mask=pd.Series(False,index=df.index);train_mask.iloc[:split_idx]=True
        X_train=X[train_mask];X_test=X[~train_mask]
        y_direction_train=df.loc[train_mask,"target_direction"];y_direction_test=df.loc[~train_mask,"target_direction"]
        y_magnitude_train=df.loc[train_mask,"target_log_vix_change"];y_magnitude_test=df.loc[~train_mask,"target_log_vix_change"]
        valid_train_mask=~(y_direction_train.isna()|y_magnitude_train.isna());valid_test_mask=~(y_direction_test.isna()|y_magnitude_test.isna())
        X_train=X_train[valid_train_mask];y_direction_train=y_direction_train[valid_train_mask];y_magnitude_train=y_magnitude_train[valid_train_mask]
        X_test=X_test[valid_test_mask];y_direction_test=y_direction_test[valid_test_mask];y_magnitude_test=y_magnitude_test[valid_test_mask]
        logger.info(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples | Features: {len(self.feature_names)}")
        logger.info("[4/5] Training direction classifier...")
        self.direction_model,direction_metrics=self._train_direction_model(X_train,y_direction_train,X_test,y_direction_test);self.metrics["direction"]=direction_metrics
        logger.info("[5/5] Training magnitude regressor...")
        self.magnitude_model,magnitude_metrics=self._train_magnitude_model(X_train,y_magnitude_train,X_test,y_magnitude_test);self.metrics["magnitude"]=magnitude_metrics
        self._save_models(save_dir);self._generate_diagnostics(X_test,y_direction_test,y_magnitude_test,save_dir)
        logger.info("✅ Training complete");self._print_summary()
        return self
    def _prepare_features(self,df,selected_features=None):
        exclude_cols=["vix","spx","calendar_cohort","cohort_weight","feature_quality","future_vix","target_vix_pct_change","target_log_vix_change","target_direction"]
        cohort_features=["is_fomc_period","is_opex_week","is_earnings_heavy"]
        if selected_features is not None:
            logger.info(f"  Using {len(selected_features)} selected features")
            feature_cols=[f for f in selected_features if f in df.columns and f not in exclude_cols]
            for cf in cohort_features:
                if cf in df.columns and cf not in feature_cols:feature_cols.append(cf);logger.info(f"  Added cohort feature: {cf}")
        else:
            logger.info("  Using all available features")
            all_cols=df.columns.tolist();feature_cols=[c for c in all_cols if c not in exclude_cols]
            for cf in cohort_features:
                if cf not in feature_cols and cf in df.columns:feature_cols.append(cf)
        feature_cols=list(dict.fromkeys(feature_cols))
        for cf in cohort_features:
            if cf not in df.columns:logger.warning(f"  Missing cohort feature: {cf}, setting to 0");df[cf]=0
        X=df[feature_cols].copy();cohort_present=[cf for cf in cohort_features if cf in feature_cols]
        logger.info(f"  Total features: {len(feature_cols)} | Cohort features: {cohort_present}")
        return X,feature_cols
    def _train_direction_model(self,X_train,y_train,X_test,y_test):
        params=XGBOOST_CONFIG["shared_params"].copy();params.update({"objective":"binary:logistic","eval_metric":"logloss"})
        model=XGBClassifier(**params);model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)
        y_train_pred=model.predict(X_train);y_train_prob=model.predict_proba(X_train)[:,1]
        y_test_pred=model.predict(X_test);y_test_prob=model.predict_proba(X_test)[:,1]
        metrics={"train":{"accuracy":float(accuracy_score(y_train,y_train_pred)),"precision":float(precision_score(y_train,y_train_pred,zero_division=0)),"recall":float(recall_score(y_train,y_train_pred,zero_division=0)),"logloss":float(log_loss(y_train,y_train_prob))},"test":{"accuracy":float(accuracy_score(y_test,y_test_pred)),"precision":float(precision_score(y_test,y_test_pred,zero_division=0)),"recall":float(recall_score(y_test,y_test_pred,zero_division=0)),"logloss":float(log_loss(y_test,y_test_prob))}}
        logger.info(f"  Train Accuracy: {metrics['train']['accuracy']:.1%} | Test Accuracy: {metrics['test']['accuracy']:.1%} | Precision: {metrics['test']['precision']:.1%} | Recall: {metrics['test']['recall']:.1%}")
        return model,metrics
    def _train_magnitude_model(self,X_train,y_train,X_test,y_test):
        params=XGBOOST_CONFIG["shared_params"].copy();params.update({"objective":"reg:squarederror","eval_metric":"rmse"})
        model=XGBRegressor(**params);model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)
        y_train_pred=model.predict(X_train);y_test_pred=model.predict(X_test)
        train_pct_actual=(np.exp(y_train)-1)*100;train_pct_pred=(np.exp(y_train_pred)-1)*100
        test_pct_actual=(np.exp(y_test)-1)*100;test_pct_pred=(np.exp(y_test_pred)-1)*100
        metrics={"train":{"mae_log":float(mean_absolute_error(y_train,y_train_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_train,y_train_pred))),"mae_pct":float(mean_absolute_error(train_pct_actual,train_pct_pred)),"bias_pct":float(np.mean(train_pct_pred-train_pct_actual))},"test":{"mae_log":float(mean_absolute_error(y_test,y_test_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_test,y_test_pred))),"mae_pct":float(mean_absolute_error(test_pct_actual,test_pct_pred)),"bias_pct":float(np.mean(test_pct_pred-test_pct_actual))}}
        return model,metrics
    def predict(self,X,current_vix):
        X_features=X[self.feature_names];prob_up=float(self.direction_model.predict_proba(X_features)[0,1]);prob_down=1.0-prob_up
        magnitude_log=float(self.magnitude_model.predict(X_features)[0]);magnitude_pct=(np.exp(magnitude_log)-1)*100;magnitude_pct=np.clip(magnitude_pct,-50,100)
        expected_vix=current_vix*(1+magnitude_pct/100)
        return {"prob_up":prob_up,"prob_down":prob_down,"magnitude_pct":float(magnitude_pct),"magnitude_log":float(magnitude_log),"expected_vix":float(expected_vix),"current_vix":float(current_vix)}
    def _save_models(self,save_dir):
        save_path=Path(save_dir);save_path.mkdir(parents=True,exist_ok=True)
        direction_file=save_path/"direction_5d_model.pkl"
        with open(direction_file,"wb")as f:pickle.dump(self.direction_model,f)
        magnitude_file=save_path/"magnitude_5d_model.pkl"
        with open(magnitude_file,"wb")as f:pickle.dump(self.magnitude_model,f)
        features_file=save_path/"feature_names.json"
        with open(features_file,"w")as f:json.dump(self.feature_names,f,indent=2)
        metrics_file=save_path/"training_metrics.json"
        with open(metrics_file,"w")as f:json.dump(self.metrics,f,indent=2)
        logger.info(f"  Saved models to: {save_dir}/")
    def load(self,models_dir="models"):
        models_path=Path(models_dir);direction_file=models_path/"direction_5d_model.pkl"
        with open(direction_file,"rb")as f:self.direction_model=pickle.load(f)
        magnitude_file=models_path/"magnitude_5d_model.pkl"
        with open(magnitude_file,"rb")as f:self.magnitude_model=pickle.load(f)
        features_file=models_path/"feature_names.json"
        with open(features_file,"r")as f:self.feature_names=json.load(f)
        logger.info(f"✅ Loaded models from: {models_dir} ({len(self.feature_names)} features)")
    def _generate_diagnostics(self,X_test,y_direction_test,y_magnitude_test,save_dir):
        save_path=Path(save_dir);save_path.mkdir(parents=True,exist_ok=True)
        try:
            fig,axes=plt.subplots(2,2,figsize=(14,10));fig.suptitle("Model Performance Diagnostics",fontsize=16,fontweight="bold")
            ax=axes[0,0];y_prob=self.direction_model.predict_proba(X_test)[:,1];prob_true,prob_pred=calibration_curve(y_direction_test,y_prob,n_bins=10)
            ax.plot([0,1],[0,1],"k--",label="Perfect calibration");ax.plot(prob_pred,prob_true,"o-",label="Model",linewidth=2)
            ax.set_xlabel("Predicted Probability");ax.set_ylabel("Actual Frequency");ax.set_title("Direction Probability Calibration");ax.legend();ax.grid(True,alpha=0.3)
            ax=axes[0,1];bins=[0,0.4,0.6,1.0];labels=["<40%","40-60%",">60%"];X_test_copy=X_test.copy()
            X_test_copy["prob"]=y_prob;X_test_copy["actual"]=y_direction_test.values;X_test_copy["prob_bin"]=pd.cut(X_test_copy["prob"],bins=bins,labels=labels)
            accs=[]
            for label in labels:
                mask=X_test_copy["prob_bin"]==label
                if mask.sum()>0:acc=(X_test_copy.loc[mask,"actual"]==1).mean();accs.append(acc)
                else:accs.append(0)
            ax.bar(range(len(labels)),accs,alpha=0.7);ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels)
            ax.set_ylabel("Actual UP Frequency");ax.set_title("Direction Accuracy by Confidence");ax.grid(True,alpha=0.3,axis="y")
            ax=axes[1,0];y_mag_pred=self.magnitude_model.predict(X_test);test_pct_actual=(np.exp(y_magnitude_test)-1)*100;test_pct_pred=(np.exp(y_mag_pred)-1)*100
            ax.scatter(test_pct_pred,test_pct_actual,alpha=0.5,s=30);lims=[min(test_pct_pred.min(),test_pct_actual.min()),max(test_pct_pred.max(),test_pct_actual.max())]
            ax.plot(lims,lims,"k--",alpha=0.5);ax.set_xlabel("Predicted VIX Change (%)");ax.set_ylabel("Actual VIX Change (%)");ax.set_title("Magnitude Forecast Accuracy");ax.grid(True,alpha=0.3)
            ax=axes[1,1];errors=test_pct_pred-test_pct_actual;ax.hist(errors,bins=30,alpha=0.7,edgecolor="black")
            ax.axvline(0,color="red",linestyle="--",linewidth=2,label="Zero error");ax.axvline(errors.mean(),color="blue",linestyle="--",linewidth=2,label=f"Mean: {errors.mean():.2f}%")
            ax.set_xlabel("Prediction Error (%)");ax.set_ylabel("Frequency");ax.set_title("Magnitude Error Distribution");ax.legend();ax.grid(True,alpha=0.3)
            plt.tight_layout();plot_file=save_path/"model_diagnostics.png";plt.savefig(plot_file,dpi=150,bbox_inches="tight");plt.close()
            logger.info(f"  Saved diagnostics: {plot_file}")
        except Exception as e:logger.warning(f"  Could not generate plots: {e}")
    def _print_summary(self):
        print("\n"+"="*60);print("TRAINING SUMMARY");print("="*60);print(f"Models: Direction Classifier + Magnitude Regressor");print(f"Features: {len(self.feature_names)} (including cohort flags)")
        print(f"\nDirection Performance:");print(f"  Test Accuracy:  {self.metrics['direction']['test']['accuracy']:.1%}");print(f"  Test Precision: {self.metrics['direction']['test']['precision']:.1%}");print(f"  Test Recall:    {self.metrics['direction']['test']['recall']:.1%}")
        print(f"\nMagnitude Performance:");print(f"  Test MAE:  {self.metrics['magnitude']['test']['mae_pct']:.2f}%");print(f"  Test Bias: {self.metrics['magnitude']['test']['bias_pct']:+.2f}%");print(f"  Test RMSE: {self.metrics['magnitude']['test']['rmse_log']:.4f} (log)")
        print("="*60+"\n")
def train_simplified_forecaster(df,selected_features=None,save_dir="models"):
    forecaster=SimplifiedVIXForecaster();forecaster.train(df,selected_features=selected_features,save_dir=save_dir)
    return forecaster
