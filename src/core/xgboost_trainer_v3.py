import json,logging,pickle
from pathlib import Path
from typing import Dict,List,Tuple
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,precision_score,recall_score
from xgboost import XGBRegressor,XGBClassifier
from config import TARGET_CONFIG,XGBOOST_CONFIG,TRAINING_END_DATE,REGIME_BOUNDARIES,REGIME_NAMES,QUALITY_FILTER_CONFIG,DIRECTION_CALIBRATION_CONFIG,ENSEMBLE_CONFIG, MAGNITUDE_PARAMS, DIRECTION_PARAMS
from core.target_calculator import TargetCalculator
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class SimplifiedVIXForecaster:
    def __init__(self):
        self.horizon=TARGET_CONFIG["horizon_days"];self.magnitude_model=None;self.direction_model=None;self.direction_calibrator=None;self.magnitude_feature_names=None;self.direction_feature_names=None;self.metrics={};self.target_calculator=TargetCalculator();self.calibration_enabled=DIRECTION_CALIBRATION_CONFIG["enabled"];self.ensemble_enabled=ENSEMBLE_CONFIG["enabled"]
    def _get_regime(self,vix_level):
        for i,boundary in enumerate(REGIME_BOUNDARIES[1:]):
            if vix_level<boundary:return REGIME_NAMES[i]
        return REGIME_NAMES[len(REGIME_NAMES)-1]
    def _apply_quality_filter(self,df):
        cfg=QUALITY_FILTER_CONFIG
        if not cfg["enabled"]:logger.info("Quality filtering disabled");return df
        if 'feature_quality' not in df.columns:logger.warning("⚠️  No feature_quality column - skipping filter");return df
        threshold=cfg["min_threshold"];initial_len=len(df);quality_mask=df['feature_quality']>=threshold;filtered_df=df[quality_mask].copy();filtered_pct=(1-len(filtered_df)/initial_len)*100
        if filtered_pct>cfg["error_pct"]:
            msg=f"Critical: {filtered_pct:.1f}% data below threshold {threshold}"
            if cfg["strategy"]=="raise":logger.error(f"❌ {msg}");raise ValueError(msg)
            else:logger.warning(f"⚠️  {msg} - continuing per strategy={cfg['strategy']}")
        elif filtered_pct>cfg["warn_pct"]:logger.warning(f"⚠️  Filtered {filtered_pct:.1f}% low-quality data")
        logger.info(f"Quality: {len(filtered_df)}/{initial_len} rows retained (threshold={threshold:.2f})")
        return filtered_df
    def _calibrate_direction_probabilities(self,y_val,y_val_proba,y_test_proba):
        if not self.calibration_enabled:logger.info("Direction calibration disabled");return y_test_proba
        cfg=DIRECTION_CALIBRATION_CONFIG;min_samples=cfg["min_samples"]
        if len(y_val)<min_samples:logger.warning(f"⚠️  Insufficient samples ({len(y_val)}<{min_samples}) - skipping calibration");return y_test_proba
        logger.info(f"Calibrating direction probabilities ({cfg['method']})...")
        if cfg["method"]=="isotonic":
            calibrator=IsotonicRegression(out_of_bounds=cfg["out_of_bounds"]);calibrator.fit(y_val_proba,y_val);self.direction_calibrator=calibrator;y_test_calibrated=calibrator.transform(y_test_proba)
        else:raise ValueError(f"Unknown calibration method: {cfg['method']}")
        before_conf=np.mean(np.maximum(y_test_proba,1-y_test_proba));after_conf=np.mean(np.maximum(y_test_calibrated,1-y_test_calibrated))
        logger.info(f"  Calibrated: confidence {before_conf:.1%} → {after_conf:.1%}")
        return y_test_calibrated
    def _compute_ensemble_confidence(self,magnitude_pct,direction_prob,current_vix):
        if not self.ensemble_enabled:return float(max(direction_prob,1-direction_prob))
        cfg=ENSEMBLE_CONFIG;weights=cfg["confidence_weights"];thresholds=cfg["magnitude_thresholds"];agreement_bonus=cfg["agreement_bonus"];contradiction_penalty=cfg["contradiction_penalty"]
        abs_mag=abs(magnitude_pct);mag_category="small"if abs_mag<thresholds["small"]else("medium"if abs_mag<thresholds["medium"]else"large")
        mag_conf=0.5+min(abs_mag/thresholds["large"],0.5)*0.5;dir_conf=max(direction_prob,1-direction_prob);predicted_up=direction_prob>0.5;magnitude_up=magnitude_pct>0;models_agree=predicted_up==magnitude_up
        if models_agree:
            if abs_mag>thresholds["medium"]and dir_conf>0.75:agreement_score=agreement_bonus["strong"]
            elif abs_mag>thresholds["small"]and dir_conf>0.65:agreement_score=agreement_bonus["moderate"]
            else:agreement_score=agreement_bonus["weak"]
        else:
            if abs_mag>thresholds["medium"]and dir_conf>0.75:agreement_score=-contradiction_penalty["severe"]
            elif abs_mag>thresholds["small"]and dir_conf>0.65:agreement_score=-contradiction_penalty["moderate"]
            else:agreement_score=-contradiction_penalty["minor"]
        ensemble_conf=weights["magnitude"]*mag_conf+weights["direction"]*dir_conf+weights["agreement"]*(0.5+agreement_score);ensemble_conf=np.clip(ensemble_conf,cfg["min_ensemble_confidence"],1.0)
        return float(ensemble_conf)
    def train(self,df,magnitude_features=None,direction_features=None,save_dir="models"):
        df=self.target_calculator.calculate_all_targets(df,vix_col="vix");df=self._apply_quality_filter(df);validation=self.target_calculator.validate_targets(df)
        if not validation["valid"]:logger.error(f"❌ Target validation failed: {validation['warnings']}");raise ValueError("Invalid targets")
        for warning in validation["warnings"]:logger.warning(f"  ⚠️  {warning}")
        stats=validation["stats"];logger.info(f"Valid targets: {stats['count']} | UP: {df['target_direction'].sum()} ({df['target_direction'].mean():.1%}) | DOWN: {len(df)-df['target_direction'].sum()}")
        X_mag,mag_names=self._prepare_features(df,magnitude_features);X_dir,dir_names=self._prepare_features(df,direction_features);self.magnitude_feature_names=mag_names;self.direction_feature_names=dir_names
        total_samples=len(X_mag);test_size=XGBOOST_CONFIG["cv_config"]["test_size"];val_size=XGBOOST_CONFIG["cv_config"]["val_size"];train_end_idx=int(total_samples*(1-test_size-val_size));val_end_idx=int(total_samples*(1-test_size))
        X_mag_train=X_mag.iloc[:train_end_idx];X_mag_val=X_mag.iloc[train_end_idx:val_end_idx];X_mag_test=X_mag.iloc[val_end_idx:];X_dir_train=X_dir.iloc[:train_end_idx];X_dir_val=X_dir.iloc[train_end_idx:val_end_idx];X_dir_test=X_dir.iloc[val_end_idx:]
        y_direction_train=df.iloc[:train_end_idx]["target_direction"];y_direction_val=df.iloc[train_end_idx:val_end_idx]["target_direction"];y_direction_test=df.iloc[val_end_idx:]["target_direction"];y_magnitude_train=df.iloc[:train_end_idx]["target_log_vix_change"];y_magnitude_val=df.iloc[train_end_idx:val_end_idx]["target_log_vix_change"];y_magnitude_test=df.iloc[val_end_idx:]["target_log_vix_change"]
        valid_train_mask=~(y_magnitude_train.isna());valid_val_mask=~(y_magnitude_val.isna());valid_test_mask=~(y_magnitude_test.isna())
        X_mag_train=X_mag_train[valid_train_mask];X_dir_train=X_dir_train[valid_train_mask];y_magnitude_train=y_magnitude_train[valid_train_mask];y_direction_train=y_direction_train[valid_train_mask];X_mag_val=X_mag_val[valid_val_mask];X_dir_val=X_dir_val[valid_val_mask];y_magnitude_val=y_magnitude_val[valid_val_mask];y_direction_val=y_direction_val[valid_val_mask];X_mag_test=X_mag_test[valid_test_mask];X_dir_test=X_dir_test[valid_test_mask];y_magnitude_test=y_magnitude_test[valid_test_mask];y_direction_test=y_direction_test[valid_test_mask]
        actual_train_end=df.iloc[:train_end_idx].index[-1];logger.info(f"Data: {df.index[0].date()} → {df.index[-1].date()} | Training through: {actual_train_end.date()}");logger.info(f"Split: Train={len(X_mag_train)} ({len(X_mag_train)/total_samples:.1%}) | Val={len(X_mag_val)} ({len(X_mag_val)/total_samples:.1%}) | Test={len(X_mag_test)} ({len(X_mag_test)/total_samples:.1%})");logger.info(f"Magnitude features: {len(self.magnitude_feature_names)} | Direction features: {len(self.direction_feature_names)}")
        self.magnitude_model,magnitude_metrics=self._train_magnitude_model(X_mag_train,y_magnitude_train,X_mag_val,y_magnitude_val,X_mag_test,y_magnitude_test,df);self.metrics["magnitude"]=magnitude_metrics;self.direction_model,direction_metrics,y_val_proba,y_test_proba=self._train_direction_model(X_dir_train,y_direction_train,X_dir_val,y_direction_val,X_dir_test,y_direction_test,df);self.metrics["direction"]=direction_metrics
        if self.calibration_enabled:
            y_test_proba_cal=self._calibrate_direction_probabilities(y_direction_val.values,y_val_proba,y_test_proba);y_test_pred_cal=(y_test_proba_cal>0.5).astype(int);test_acc_cal=accuracy_score(y_direction_test,y_test_pred_cal);test_prec_cal=precision_score(y_direction_test,y_test_pred_cal,zero_division=0);test_rec_cal=recall_score(y_direction_test,y_test_pred_cal,zero_division=0);avg_conf_cal=float(np.mean(np.maximum(y_test_proba_cal,1-y_test_proba_cal)));self.metrics["direction_calibrated"]={"test_accuracy":float(test_acc_cal),"test_precision":float(test_prec_cal),"test_recall":float(test_rec_cal),"avg_confidence":avg_conf_cal};logger.info(f"Direction (Calibrated): Test Acc={test_acc_cal:.1%} | Prec={test_prec_cal:.1%} | Rec={test_rec_cal:.1%} | Conf={avg_conf_cal:.1%}")
        self._save_models(save_dir);self._generate_diagnostics(X_mag_test,X_dir_test,y_direction_test,y_magnitude_test,save_dir,df.loc[X_mag_test.index,"vix"]);logger.info("✅ Training complete");self._print_summary()
        return self
    def _prepare_features(self,df,selected_features=None):
        exclude_cols=["vix","spx","calendar_cohort","cohort_weight","feature_quality","future_vix","target_vix_pct_change","target_log_vix_change","target_direction"];cohort_features=["is_fomc_period","is_opex_week","is_earnings_heavy"]
        if selected_features is not None:
            feature_cols=[f for f in selected_features if f in df.columns and f not in exclude_cols];missing_cohorts=[cf for cf in cohort_features if cf not in feature_cols and cf in df.columns]
            if missing_cohorts:logger.warning(f"  Cohort features missing: {missing_cohorts}")
        else:logger.info("  Using all available features");all_cols=df.columns.tolist();feature_cols=[c for c in all_cols if c not in exclude_cols]
        feature_cols=list(dict.fromkeys(feature_cols))
        for cf in cohort_features:
            if cf not in df.columns:logger.warning(f"  Missing cohort feature: {cf}, setting to 0");df[cf]=0
        X=df[feature_cols].copy()
        return X,feature_cols
    def _train_magnitude_model(self,X_train,y_train,X_val,y_val,X_test,y_test,df):
        params=MAGNITUDE_PARAMS.copy();model=XGBRegressor(**params)
        train_weights=None;val_weights=None
        if 'cohort_weight' in df.columns:
            train_weights=df.loc[X_train.index,'cohort_weight'].values;val_weights=df.loc[X_val.index,'cohort_weight'].values;logger.info(f"Using cohort weights: mean={train_weights.mean():.3f}")
        model.fit(X_train,y_train,sample_weight=train_weights,eval_set=[(X_val,y_val)],sample_weight_eval_set=[val_weights] if val_weights is not None else None,verbose=False)
        y_train_pred_raw=model.predict(X_train);y_val_pred_raw=model.predict(X_val);y_test_pred_raw=model.predict(X_test);y_train_pred=np.clip(y_train_pred_raw,-2,2);y_val_pred=np.clip(y_val_pred_raw,-2,2);y_test_pred=np.clip(y_test_pred_raw,-2,2)
        n_clipped_train=np.sum(np.abs(y_train_pred_raw)>2);n_clipped_val=np.sum(np.abs(y_val_pred_raw)>2);n_clipped_test=np.sum(np.abs(y_test_pred_raw)>2)
        if n_clipped_train>0:logger.warning(f"  ⚠️  Clipped {n_clipped_train}/{len(y_train_pred)} train predictions (max: {np.abs(y_train_pred_raw).max():.3f})")
        if n_clipped_val>0:logger.warning(f"  ⚠️  Clipped {n_clipped_val}/{len(y_val_pred)} val predictions (max: {np.abs(y_val_pred_raw).max():.3f})")
        if n_clipped_test>0:logger.warning(f"  ⚠️  Clipped {n_clipped_test}/{len(y_test_pred)} test predictions (max: {np.abs(y_test_pred_raw).max():.3f})")
        train_pct_actual=(np.exp(y_train)-1)*100;train_pct_pred=(np.exp(y_train_pred)-1)*100;val_pct_actual=(np.exp(y_val)-1)*100;val_pct_pred=(np.exp(y_val_pred)-1)*100;test_pct_actual=(np.exp(y_test)-1)*100;test_pct_pred=(np.exp(y_test_pred)-1)*100
        if np.isnan(train_pct_pred).any()or np.isinf(train_pct_pred).any():logger.error("❌ NaN/Inf detected in train predictions after clipping");raise ValueError("Model produced invalid predictions")
        metrics={"train":{"mae_log":float(mean_absolute_error(y_train,y_train_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_train,y_train_pred))),"mae_pct":float(mean_absolute_error(train_pct_actual,train_pct_pred)),"bias_pct":float(np.mean(train_pct_pred-train_pct_actual)),"clipped_pct":float(n_clipped_train/len(y_train_pred)*100)},"val":{"mae_log":float(mean_absolute_error(y_val,y_val_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_val,y_val_pred))),"mae_pct":float(mean_absolute_error(val_pct_actual,val_pct_pred)),"bias_pct":float(np.mean(val_pct_pred-val_pct_actual)),"clipped_pct":float(n_clipped_val/len(y_val_pred)*100)},"test":{"mae_log":float(mean_absolute_error(y_test,y_test_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_test,y_test_pred))),"mae_pct":float(mean_absolute_error(test_pct_actual,test_pct_pred)),"bias_pct":float(np.mean(test_pct_pred-test_pct_actual)),"clipped_pct":float(n_clipped_test/len(y_test_pred)*100)}}
        logger.info(f"Magnitude: Train MAE={metrics['train']['mae_pct']:.2f}% | Val MAE={metrics['val']['mae_pct']:.2f}% | Test MAE={metrics['test']['mae_pct']:.2f}% | Bias={metrics['test']['bias_pct']:+.2f}%")
        if metrics['test']['clipped_pct']>5.0:logger.warning(f"  ⚠️  High clipping rate: {metrics['test']['clipped_pct']:.1f}% of test predictions - consider retraining with tighter regularization")
        return model,metrics
    def _train_direction_model(self,X_train,y_train,X_val,y_val,X_test,y_test,df):
        params=DIRECTION_PARAMS.copy().copy();model=XGBClassifier(**params)
        train_weights=None;val_weights=None
        if 'cohort_weight' in df.columns:
            train_weights=df.loc[X_train.index,'cohort_weight'].values;val_weights=df.loc[X_val.index,'cohort_weight'].values;logger.info(f"Using cohort weights: mean={train_weights.mean():.3f}")
        model.fit(X_train,y_train,sample_weight=train_weights,eval_set=[(X_val,y_val)],sample_weight_eval_set=[val_weights] if val_weights is not None else None,verbose=False)
        y_train_pred=model.predict(X_train);y_val_pred=model.predict(X_val);y_test_pred=model.predict(X_test);y_train_proba=model.predict_proba(X_train)[:,1];y_val_proba=model.predict_proba(X_val)[:,1];y_test_proba=model.predict_proba(X_test)[:,1]
        train_acc=accuracy_score(y_train,y_train_pred);val_acc=accuracy_score(y_val,y_val_pred);test_acc=accuracy_score(y_test,y_test_pred);test_prec=precision_score(y_test,y_test_pred,zero_division=0);test_rec=recall_score(y_test,y_test_pred,zero_division=0)
        metrics={"train":{"accuracy":float(train_acc),"avg_confidence":float(np.mean(np.maximum(y_train_proba,1-y_train_proba)))},"val":{"accuracy":float(val_acc),"avg_confidence":float(np.mean(np.maximum(y_val_proba,1-y_val_proba)))},"test":{"accuracy":float(test_acc),"precision":float(test_prec),"recall":float(test_rec),"avg_confidence":float(np.mean(np.maximum(y_test_proba,1-y_test_proba)))}}
        logger.info(f"Direction: Train Acc={train_acc:.1%} | Val Acc={val_acc:.1%} | Test Acc={test_acc:.1%} | Prec={test_prec:.1%} | Rec={test_rec:.1%}")
        return model,metrics,y_val_proba,y_test_proba
    def predict(self,X,current_vix):
        X_mag=X[self.magnitude_feature_names];X_dir=X[self.direction_feature_names];magnitude_log_raw=float(self.magnitude_model.predict(X_mag)[0]);magnitude_log=np.clip(magnitude_log_raw,-2,2)
        if abs(magnitude_log_raw)>2:logger.warning(f"⚠️  Clipped prediction: {magnitude_log_raw:.3f} → {magnitude_log:.3f}")
        magnitude_pct=(np.exp(magnitude_log)-1)*100;magnitude_pct=np.clip(magnitude_pct,-50,100);dir_proba_raw=self.direction_model.predict_proba(X_dir)[0][1]
        if self.direction_calibrator is not None:
            dir_proba=float(self.direction_calibrator.transform([dir_proba_raw])[0])
        else:dir_proba=float(dir_proba_raw)
        direction="UP"if dir_proba>0.5 else"DOWN"
        ensemble_confidence=self._compute_ensemble_confidence(magnitude_pct,dir_proba,current_vix);actionable=ensemble_confidence>ENSEMBLE_CONFIG["actionable_threshold"]
        expected_vix=current_vix*(1+magnitude_pct/100);current_regime=self._get_regime(current_vix);expected_regime=self._get_regime(expected_vix);regime_change=current_regime!=expected_regime
        return {"magnitude_pct":float(magnitude_pct),"magnitude_log":float(magnitude_log),"expected_vix":float(expected_vix),"current_vix":float(current_vix),"direction":direction,"direction_probability":dir_proba,"direction_confidence":ensemble_confidence,"current_regime":current_regime,"expected_regime":expected_regime,"regime_change":regime_change,"actionable":actionable,"ensemble_enabled":self.ensemble_enabled,"calibration_enabled":self.calibration_enabled}
    def _save_models(self,save_dir):
        save_path=Path(save_dir);save_path.mkdir(parents=True,exist_ok=True);magnitude_file=save_path/"magnitude_5d_model.pkl"
        with open(magnitude_file,"wb")as f:pickle.dump(self.magnitude_model,f)
        direction_file=save_path/"direction_5d_model.pkl"
        with open(direction_file,"wb")as f:pickle.dump(self.direction_model,f)
        if self.direction_calibrator is not None:
            calibrator_file=save_path/"direction_calibrator.pkl"
            with open(calibrator_file,"wb")as f:pickle.dump(self.direction_calibrator,f)
        mag_features_file=save_path/"feature_names_magnitude.json"
        with open(mag_features_file,"w")as f:json.dump(self.magnitude_feature_names,f,indent=2)
        dir_features_file=save_path/"feature_names_direction.json"
        with open(dir_features_file,"w")as f:json.dump(self.direction_feature_names,f,indent=2)
        metrics_file=save_path/"training_metrics.json"
        with open(metrics_file,"w")as f:json.dump(self.metrics,f,indent=2)
    def load(self,models_dir="models"):
        models_path=Path(models_dir);magnitude_file=models_path/"magnitude_5d_model.pkl"
        with open(magnitude_file,"rb")as f:self.magnitude_model=pickle.load(f)
        direction_file=models_path/"direction_5d_model.pkl"
        with open(direction_file,"rb")as f:self.direction_model=pickle.load(f)
        calibrator_file=models_path/"direction_calibrator.pkl"
        if calibrator_file.exists():
            with open(calibrator_file,"rb")as f:self.direction_calibrator=pickle.load(f)
            logger.info("✅ Loaded direction calibrator")
        mag_features_file=models_path/"feature_names_magnitude.json"
        with open(mag_features_file,"r")as f:self.magnitude_feature_names=json.load(f)
        dir_features_file=models_path/"feature_names_direction.json"
        with open(dir_features_file,"r")as f:self.direction_feature_names=json.load(f)
        logger.info(f"✅ Loaded models: {len(self.magnitude_feature_names)} mag features, {len(self.direction_feature_names)} dir features")
    def _generate_diagnostics(self,X_mag_test,X_dir_test,y_direction_test,y_magnitude_test,save_dir,vix_test):
        save_path=Path(save_dir);save_path.mkdir(parents=True,exist_ok=True)
        try:
            fig,axes=plt.subplots(2,3,figsize=(18,10));fig.suptitle("Model Performance (Magnitude + Direction)",fontsize=16,fontweight="bold");ax=axes[0,0];y_mag_pred=np.clip(self.magnitude_model.predict(X_mag_test),-2,2);test_pct_actual=(np.exp(y_magnitude_test)-1)*100;test_pct_pred=(np.exp(y_mag_pred)-1)*100;ax.scatter(test_pct_pred,test_pct_actual,alpha=0.5,s=30);lims=[min(test_pct_pred.min(),test_pct_actual.min()),max(test_pct_pred.max(),test_pct_actual.max())];ax.plot(lims,lims,"k--",alpha=0.5);ax.set_xlabel("Predicted (%)");ax.set_ylabel("Actual (%)");ax.set_title("Magnitude Accuracy");ax.grid(True,alpha=0.3)
            ax=axes[0,1];errors=test_pct_pred-test_pct_actual;ax.hist(errors,bins=30,alpha=0.7,edgecolor="black");ax.axvline(0,color="red",linestyle="--",linewidth=2,label="Zero");ax.axvline(errors.mean(),color="blue",linestyle="--",linewidth=2,label=f"Mean: {errors.mean():.2f}%");ax.set_xlabel("Error (%)");ax.set_ylabel("Frequency");ax.set_title("Magnitude Error Distribution");ax.legend();ax.grid(True,alpha=0.3)
            ax=axes[0,2];dir_proba_raw=self.direction_model.predict_proba(X_dir_test)[:,1]
            if self.direction_calibrator is not None:
                dir_proba=self.direction_calibrator.transform(dir_proba_raw)
            else:dir_proba=dir_proba_raw
            conf_bins=pd.cut(np.maximum(dir_proba,1-dir_proba),bins=[0.5,0.6,0.7,0.8,1.0],labels=["50-60%","60-70%","70-80%","80-100%"]);accs=[]
            for label in["50-60%","60-70%","70-80%","80-100%"]:
                mask=conf_bins==label
                if mask.sum()>0:y_pred=(dir_proba[mask]>0.5).astype(int);acc=accuracy_score(y_direction_test[mask],y_pred);accs.append(acc)
                else:accs.append(0)
            ax.bar(range(4),accs,alpha=0.7);ax.set_xticks(range(4));ax.set_xticklabels(["50-60%","60-70%","70-80%","80-100%"]);ax.set_ylabel("Accuracy");ax.set_title("Direction by Confidence");ax.grid(True,alpha=0.3,axis="y");ax.axhline(0.5,color='red',linestyle='--',alpha=0.5)
            ax=axes[1,0];y_dir_pred=(dir_proba>0.5).astype(int);correct_mask=(y_dir_pred==y_direction_test);ax.hist([dir_proba[correct_mask],dir_proba[~correct_mask]],bins=20,alpha=0.7,label=['Correct','Wrong'],edgecolor='black');ax.set_xlabel("Probability (UP)");ax.set_ylabel("Frequency");ax.set_title("Direction Probability Distribution");ax.legend();ax.grid(True,alpha=0.3);ax.axvline(0.5,color='red',linestyle='--',alpha=0.5)
            ax=axes[1,1];regimes=vix_test.apply(self._get_regime);regime_names=["Low Vol","Normal","Elevated","Crisis"];regime_accs=[]
            for regime in regime_names:
                mask=regimes==regime
                if mask.sum()>5:y_pred=(dir_proba[mask]>0.5).astype(int);acc=accuracy_score(y_direction_test[mask],y_pred);regime_accs.append(acc)
                else:regime_accs.append(0)
            ax.bar(range(len(regime_names)),regime_accs,alpha=0.7);ax.set_xticks(range(len(regime_names)));ax.set_xticklabels(regime_names,rotation=45);ax.set_ylabel("Accuracy");ax.set_title("Direction by Regime");ax.grid(True,alpha=0.3,axis="y");ax.axhline(0.5,color='red',linestyle='--',alpha=0.5)
            ax=axes[1,2];regime_maes=[]
            for regime in regime_names:
                mask=regimes==regime
                if mask.sum()>5:mae=mean_absolute_error(test_pct_actual[mask],test_pct_pred[mask]);regime_maes.append(mae)
                else:regime_maes.append(0)
            ax.bar(range(len(regime_names)),regime_maes,alpha=0.7);ax.set_xticks(range(len(regime_names)));ax.set_xticklabels(regime_names,rotation=45);ax.set_ylabel("MAE (%)");ax.set_title("Magnitude by Regime");ax.grid(True,alpha=0.3,axis="y");plt.tight_layout();plot_file=save_path/"model_diagnostics.png";plt.savefig(plot_file,dpi=150,bbox_inches="tight");plt.close()
        except Exception as e:logger.warning(f"  Could not generate plots: {e}")
    def _print_summary(self):
        print("\nTRAINING SUMMARY");print(f"Models: Magnitude Regressor + Direction Classifier | Training through: {TRAINING_END_DATE}")
        if self.calibration_enabled:print("✓ Direction probability calibration: ENABLED")
        if self.ensemble_enabled:print("✓ Ensemble reconciliation: ENABLED")
        print(f"Magnitude: Train={self.metrics['magnitude']['train']['mae_pct']:.2f}% | Val={self.metrics['magnitude']['val']['mae_pct']:.2f}% | Test={self.metrics['magnitude']['test']['mae_pct']:.2f}% | Bias={self.metrics['magnitude']['test']['bias_pct']:+.2f}%")
        if self.metrics['magnitude']['test']['clipped_pct']>0:print(f"  ⚠️  Clipped predictions: {self.metrics['magnitude']['test']['clipped_pct']:.1f}% (consider tighter regularization)")
        print(f"Direction: Train={self.metrics['direction']['train']['accuracy']:.1%} | Val={self.metrics['direction']['val']['accuracy']:.1%} | Test={self.metrics['direction']['test']['accuracy']:.1%} | Prec={self.metrics['direction']['test']['precision']:.1%} | Rec={self.metrics['direction']['test']['recall']:.1%}")
        if "direction_calibrated"in self.metrics:
            dcal=self.metrics["direction_calibrated"];print(f"Direction (Cal): Test={dcal['test_accuracy']:.1%} | Prec={dcal['test_precision']:.1%} | Rec={dcal['test_recall']:.1%} | Conf={dcal['avg_confidence']:.1%}")
        print()
def train_simplified_forecaster(df,magnitude_features=None,direction_features=None,save_dir="models"):
    forecaster=SimplifiedVIXForecaster();forecaster.train(df,magnitude_features=magnitude_features,direction_features=direction_features,save_dir=save_dir)
    return forecaster
