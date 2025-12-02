import json,logging,pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,precision_score,recall_score,f1_score
from xgboost import XGBRegressor,XGBClassifier
from config import TARGET_CONFIG,XGBOOST_CONFIG,TRAINING_END_DATE,REGIME_BOUNDARIES,REGIME_NAMES,QUALITY_FILTER_CONFIG,DIRECTION_CALIBRATION_CONFIG,ENSEMBLE_CONFIG,EXPANSION_PARAMS,COMPRESSION_PARAMS,UP_CLASSIFIER_PARAMS,DOWN_CLASSIFIER_PARAMS,TRAIN_END_DATE,VAL_END_DATE

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class AsymmetricVIXForecaster:
    def __init__(self):
        self.horizon=TARGET_CONFIG["horizon_days"]
        self.expansion_model=None; self.compression_model=None
        self.up_classifier=None; self.down_classifier=None
        self.up_calibrator=None; self.down_calibrator=None
        self.expansion_features=None; self.compression_features=None
        self.up_features=None; self.down_features=None
        self.metrics={}
        self.calibration_enabled=DIRECTION_CALIBRATION_CONFIG["enabled"]
        self.skip_up_calibration=DIRECTION_CALIBRATION_CONFIG.get("skip_up_calibration",False)
        self.ensemble_enabled=ENSEMBLE_CONFIG["enabled"]
        from core.target_calculator import TargetCalculator
        self.target_calculator=TargetCalculator()

    def _get_regime(self,vix_level):
        for i,boundary in enumerate(REGIME_BOUNDARIES[1:]):
            if vix_level<boundary: return REGIME_NAMES[i]
        return REGIME_NAMES[len(REGIME_NAMES)-1]

    def _apply_quality_filter(self,df):
        cfg=QUALITY_FILTER_CONFIG
        if not cfg["enabled"]: logger.info("Quality filtering disabled"); return df
        if 'feature_quality' not in df.columns: logger.warning("⚠️  No feature_quality column - skipping filter"); return df
        threshold=cfg["min_threshold"]; initial_len=len(df)
        quality_mask=df['feature_quality']>=threshold; filtered_df=df[quality_mask].copy()
        filtered_pct=(1-len(filtered_df)/initial_len)*100
        if filtered_pct>cfg["error_pct"]:
            msg=f"Critical: {filtered_pct:.1f}% data below threshold {threshold}"
            if cfg["strategy"]=="raise": logger.error(f"❌ {msg}"); raise ValueError(msg)
            else: logger.warning(f"⚠️  {msg} - continuing per strategy={cfg['strategy']}")
        elif filtered_pct>cfg["warn_pct"]: logger.warning(f"⚠️  Filtered {filtered_pct:.1f}% low-quality data")
        logger.info(f"Quality: {len(filtered_df)}/{initial_len} rows retained (threshold={threshold:.2f})")
        return filtered_df

    def _calibrate_classifier_probabilities(self,y_val,y_val_proba,y_test_proba,name):
        if not self.calibration_enabled: logger.info(f"{name} calibration disabled"); return y_test_proba,None
        if self.skip_up_calibration and name=="UP": logger.info(f"{name} calibration skipped to preserve recall"); return y_test_proba,None
        cfg=DIRECTION_CALIBRATION_CONFIG; min_samples=cfg["min_samples"]
        if len(y_val)<min_samples: logger.warning(f"⚠️  {name}: Insufficient samples ({len(y_val)}<{min_samples}) - skipping"); return y_test_proba,None
        logger.info(f"Calibrating {name} probabilities ({cfg['method']})...")
        calibrator=IsotonicRegression(out_of_bounds=cfg["out_of_bounds"])
        calibrator.fit(y_val_proba,y_val); y_test_calibrated=calibrator.transform(y_test_proba)
        before_conf=np.mean(np.maximum(y_test_proba,1-y_test_proba))
        after_conf=np.mean(np.maximum(y_test_calibrated,1-y_test_calibrated))
        logger.info(f"  {name}: confidence {before_conf:.1%} → {after_conf:.1%}")
        return y_test_calibrated,calibrator

    def _compute_ensemble_confidence(self,classifier_prob,magnitude_pct,direction):
        if not self.ensemble_enabled: return float(classifier_prob)
        cfg=ENSEMBLE_CONFIG; weights=cfg["confidence_weights"]; scaling=cfg["magnitude_scaling"]
        abs_mag=abs(magnitude_pct)
        mag_category="high"if abs_mag>scaling["large"]else("medium"if abs_mag>scaling["medium"]else"low")
        mag_strength=min(abs_mag/scaling["large"],1.0)
        ensemble_conf=weights["classifier"]*classifier_prob+weights["magnitude"]*mag_strength
        ensemble_conf=np.clip(ensemble_conf,cfg["min_ensemble_confidence"],1.0)
        return float(ensemble_conf)

    def _get_dynamic_threshold(self,magnitude_pct,direction):
        cfg=ENSEMBLE_CONFIG; thresholds=cfg["dynamic_thresholds"][direction.lower()]
        scaling=cfg["magnitude_scaling"]; abs_mag=abs(magnitude_pct)
        if abs_mag>scaling["large"]: return thresholds["high_magnitude"]
        elif abs_mag>scaling["medium"]: return thresholds["medium_magnitude"]
        else: return thresholds["low_magnitude"]

    def train(self,df,expansion_features=None,compression_features=None,up_features=None,down_features=None,save_dir="models"):
        df=self.target_calculator.calculate_all_targets(df,vix_col="vix")
        df=self._apply_quality_filter(df)
        validation=self.target_calculator.validate_targets(df)
        if not validation["valid"]: logger.error(f"❌ Target validation failed: {validation['warnings']}"); raise ValueError("Invalid targets")
        for warning in validation["warnings"]: logger.warning(f"  ⚠️  {warning}")

        stats=validation["stats"]
        logger.info(f"Valid targets: {stats['count']} | UP: {df['target_direction'].sum()} ({df['target_direction'].mean():.1%}) | DOWN: {len(df)-df['target_direction'].sum()}")

        # Prepare feature sets
        if expansion_features: self.expansion_features=expansion_features
        else: self.expansion_features=self._prepare_features(df)[1]
        if compression_features: self.compression_features=compression_features
        else: self.compression_features=self._prepare_features(df)[1]
        if up_features: self.up_features=up_features
        else: self.up_features=self._prepare_features(df)[1]
        if down_features: self.down_features=down_features
        else: self.down_features=self._prepare_features(df)[1]

        # Split by date
        total_samples=len(df)
        train_end_idx=df[df.index<=pd.Timestamp(TRAIN_END_DATE)].index[-1]
        train_end_idx=df.index.get_loc(train_end_idx)
        val_end_idx=df[df.index<=pd.Timestamp(VAL_END_DATE)].index[-1]
        val_end_idx=df.index.get_loc(val_end_idx)

        train_df=df.iloc[:train_end_idx+1]; val_df=df.iloc[train_end_idx+1:val_end_idx+1]; test_df=df.iloc[val_end_idx+1:]

        actual_train_end=train_df.index[-1]; actual_val_end=val_df.index[-1]
        logger.info(f"Data: {df.index[0].date()} → {df.index[-1].date()} | Train through: {actual_train_end.date()} | Val through: {actual_val_end.date()}")
        logger.info(f"Split: Train={len(train_df)} ({len(train_df)/total_samples:.1%}) | Val={len(val_df)} ({len(val_df)/total_samples:.1%}) | Test={len(test_df)} ({len(test_df)/total_samples:.1%})")

        # Domain-specific splits for regressors
        train_up_mask=train_df['target_direction']==1; train_down_mask=train_df['target_direction']==0
        val_up_mask=val_df['target_direction']==1; val_down_mask=val_df['target_direction']==0
        test_up_mask=test_df['target_direction']==1; test_down_mask=test_df['target_direction']==0

        # Filter out NaN targets after domain split
        train_valid_target=train_df['target_log_vix_change'].notna()
        val_valid_target=val_df['target_log_vix_change'].notna()
        test_valid_target=test_df['target_log_vix_change'].notna()

        train_up_mask=train_up_mask&train_valid_target; train_down_mask=train_down_mask&train_valid_target
        val_up_mask=val_up_mask&val_valid_target; val_down_mask=val_down_mask&val_valid_target
        test_up_mask=test_up_mask&test_valid_target; test_down_mask=test_down_mask&test_valid_target

        logger.info(f"\n{'='*80}")
        logger.info("🎯 ASYMMETRIC 4-MODEL TRAINING")
        logger.info(f"Train UP: {train_up_mask.sum()} | DOWN: {train_down_mask.sum()}")
        logger.info(f"Val UP: {val_up_mask.sum()} | DOWN: {val_down_mask.sum()}")
        logger.info(f"Test UP: {test_up_mask.sum()} | DOWN: {test_down_mask.sum()}")

        # Train expansion regressor (UP samples only)
        logger.info(f"\n{'='*60}")
        logger.info("📈 EXPANSION REGRESSOR (UP samples)")
        X_exp_train=train_df[train_up_mask][self.expansion_features]
        y_exp_train=train_df[train_up_mask]['target_log_vix_change']
        X_exp_val=val_df[val_up_mask][self.expansion_features]
        y_exp_val=val_df[val_up_mask]['target_log_vix_change']
        X_exp_test=test_df[test_up_mask][self.expansion_features]
        y_exp_test=test_df[test_up_mask]['target_log_vix_change']

        self.expansion_model,exp_metrics=self._train_regressor_model(X_exp_train,y_exp_train,X_exp_val,y_exp_val,X_exp_test,y_exp_test,EXPANSION_PARAMS,"Expansion",train_df[train_up_mask],val_df[val_up_mask])
        self.metrics["expansion"]=exp_metrics

        # Train compression regressor (DOWN samples only)
        logger.info(f"\n{'='*60}")
        logger.info("📉 COMPRESSION REGRESSOR (DOWN samples)")
        X_comp_train=train_df[train_down_mask][self.compression_features]
        y_comp_train=train_df[train_down_mask]['target_log_vix_change']
        X_comp_val=val_df[val_down_mask][self.compression_features]
        y_comp_val=val_df[val_down_mask]['target_log_vix_change']
        X_comp_test=test_df[test_down_mask][self.compression_features]
        y_comp_test=test_df[test_down_mask]['target_log_vix_change']

        self.compression_model,comp_metrics=self._train_regressor_model(X_comp_train,y_comp_train,X_comp_val,y_comp_val,X_comp_test,y_comp_test,COMPRESSION_PARAMS,"Compression",train_df[train_down_mask],val_df[val_down_mask])
        self.metrics["compression"]=comp_metrics

        # Train UP classifier (optimized for UP F1)
        logger.info(f"\n{'='*60}")
        logger.info("🔺 UP CLASSIFIER (F1-optimized)")
        X_up_train=train_df[self.up_features]; y_up_train=train_df['target_direction']
        X_up_val=val_df[self.up_features]; y_up_val=val_df['target_direction']
        X_up_test=test_df[self.up_features]; y_up_test=test_df['target_direction']

        self.up_classifier,up_metrics,up_val_proba,up_test_proba=self._train_classifier_model(X_up_train,y_up_train,X_up_val,y_up_val,X_up_test,y_up_test,UP_CLASSIFIER_PARAMS,"UP",train_df,val_df,invert=False)
        self.metrics["up_classifier"]=up_metrics

        if self.calibration_enabled:
            up_test_proba_cal,self.up_calibrator=self._calibrate_classifier_probabilities(y_up_val.values,up_val_proba,up_test_proba,"UP")
            if self.up_calibrator is not None:
                up_test_pred_cal=(up_test_proba_cal>0.5).astype(int)
                test_acc_cal=accuracy_score(y_up_test,up_test_pred_cal)
                test_prec_cal=precision_score(y_up_test,up_test_pred_cal,zero_division=0)
                test_rec_cal=recall_score(y_up_test,up_test_pred_cal,zero_division=0)
                test_f1_cal=f1_score(y_up_test,up_test_pred_cal,zero_division=0)
                avg_conf_cal=float(np.mean(np.maximum(up_test_proba_cal,1-up_test_proba_cal)))
                self.metrics["up_classifier_calibrated"]={"test_accuracy":float(test_acc_cal),"test_precision":float(test_prec_cal),"test_recall":float(test_rec_cal),"test_f1":float(test_f1_cal),"avg_confidence":avg_conf_cal}
                logger.info(f"UP (Calibrated): Test Acc={test_acc_cal:.1%} | Prec={test_prec_cal:.1%} | Rec={test_rec_cal:.1%} | F1={test_f1_cal:.1%} | Conf={avg_conf_cal:.1%}")

        # Train DOWN classifier (optimized for DOWN F1)
        logger.info(f"\n{'='*60}")
        logger.info("🔻 DOWN CLASSIFIER (F1-optimized)")
        X_down_train=train_df[self.down_features]; y_down_train=1-train_df['target_direction']
        X_down_val=val_df[self.down_features]; y_down_val=1-val_df['target_direction']
        X_down_test=test_df[self.down_features]; y_down_test=1-test_df['target_direction']

        self.down_classifier,down_metrics,down_val_proba,down_test_proba=self._train_classifier_model(X_down_train,y_down_train,X_down_val,y_down_val,X_down_test,y_down_test,DOWN_CLASSIFIER_PARAMS,"DOWN",train_df,val_df,invert=True)
        self.metrics["down_classifier"]=down_metrics

        if self.calibration_enabled:
            down_test_proba_cal,self.down_calibrator=self._calibrate_classifier_probabilities(y_down_val.values,down_val_proba,down_test_proba,"DOWN")
            if self.down_calibrator is not None:
                down_test_pred_cal=(down_test_proba_cal>0.5).astype(int)
                test_acc_cal=accuracy_score(y_down_test,down_test_pred_cal)
                test_prec_cal=precision_score(y_down_test,down_test_pred_cal,zero_division=0)
                test_rec_cal=recall_score(y_down_test,down_test_pred_cal,zero_division=0)
                test_f1_cal=f1_score(y_down_test,down_test_pred_cal,zero_division=0)
                avg_conf_cal=float(np.mean(np.maximum(down_test_proba_cal,1-down_test_proba_cal)))
                self.metrics["down_classifier_calibrated"]={"test_accuracy":float(test_acc_cal),"test_precision":float(test_prec_cal),"test_recall":float(test_rec_cal),"test_f1":float(test_f1_cal),"avg_confidence":avg_conf_cal}
                logger.info(f"DOWN (Calibrated): Test Acc={test_acc_cal:.1%} | Prec={test_prec_cal:.1%} | Rec={test_rec_cal:.1%} | F1={test_f1_cal:.1%} | Conf={avg_conf_cal:.1%}")

        self._save_models(save_dir)
        self._generate_diagnostics(test_df,save_dir)
        logger.info("✅ Training complete")
        self._print_summary()
        return self

    def _prepare_features(self,df):
        exclude_cols=["vix","spx","calendar_cohort","cohort_weight","feature_quality","future_vix","target_vix_pct_change","target_log_vix_change","target_direction"]
        cohort_features=["is_fomc_period","is_opex_week","is_earnings_heavy"]
        all_cols=df.columns.tolist(); feature_cols=[c for c in all_cols if c not in exclude_cols]
        feature_cols=list(dict.fromkeys(feature_cols))
        for cf in cohort_features:
            if cf not in df.columns: logger.warning(f"  Missing cohort feature: {cf}, setting to 0"); df[cf]=0
        X=df[feature_cols].copy()
        return X,feature_cols

    def _train_regressor_model(self,X_train,y_train,X_val,y_val,X_test,y_test,params,name,train_df,val_df):
        model=XGBRegressor(**params)
        train_weights=None; val_weights=None
        if 'cohort_weight' in train_df.columns:
            train_weights=train_df['cohort_weight'].values
            val_weights=val_df['cohort_weight'].values
            logger.info(f"Using cohort weights: mean={train_weights.mean():.3f}")
        model.fit(X_train,y_train,sample_weight=train_weights,eval_set=[(X_val,y_val)],sample_weight_eval_set=[val_weights]if val_weights is not None else None,verbose=False)

        y_train_pred_raw=model.predict(X_train); y_val_pred_raw=model.predict(X_val); y_test_pred_raw=model.predict(X_test)
        y_train_pred=np.clip(y_train_pred_raw,-2,2); y_val_pred=np.clip(y_val_pred_raw,-2,2); y_test_pred=np.clip(y_test_pred_raw,-2,2)

        n_clipped_train=np.sum(np.abs(y_train_pred_raw)>2); n_clipped_val=np.sum(np.abs(y_val_pred_raw)>2); n_clipped_test=np.sum(np.abs(y_test_pred_raw)>2)
        if n_clipped_train>0: logger.warning(f"  ⚠️  Clipped {n_clipped_train}/{len(y_train_pred)} train predictions")
        if n_clipped_val>0: logger.warning(f"  ⚠️  Clipped {n_clipped_val}/{len(y_val_pred)} val predictions")
        if n_clipped_test>0: logger.warning(f"  ⚠️  Clipped {n_clipped_test}/{len(y_test_pred)} test predictions")

        train_pct_actual=(np.exp(y_train)-1)*100; train_pct_pred=(np.exp(y_train_pred)-1)*100
        val_pct_actual=(np.exp(y_val)-1)*100; val_pct_pred=(np.exp(y_val_pred)-1)*100
        test_pct_actual=(np.exp(y_test)-1)*100; test_pct_pred=(np.exp(y_test_pred)-1)*100

        metrics={"train":{"mae_log":float(mean_absolute_error(y_train,y_train_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_train,y_train_pred))),"mae_pct":float(mean_absolute_error(train_pct_actual,train_pct_pred)),"bias_pct":float(np.mean(train_pct_pred-train_pct_actual)),"clipped_pct":float(n_clipped_train/len(y_train_pred)*100)},"val":{"mae_log":float(mean_absolute_error(y_val,y_val_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_val,y_val_pred))),"mae_pct":float(mean_absolute_error(val_pct_actual,val_pct_pred)),"bias_pct":float(np.mean(val_pct_pred-val_pct_actual)),"clipped_pct":float(n_clipped_val/len(y_val_pred)*100)},"test":{"mae_log":float(mean_absolute_error(y_test,y_test_pred)),"rmse_log":float(np.sqrt(mean_squared_error(y_test,y_test_pred))),"mae_pct":float(mean_absolute_error(test_pct_actual,test_pct_pred)),"bias_pct":float(np.mean(test_pct_pred-test_pct_actual)),"clipped_pct":float(n_clipped_test/len(y_test_pred)*100)}}

        logger.info(f"{name}: Train MAE={metrics['train']['mae_pct']:.2f}% | Val MAE={metrics['val']['mae_pct']:.2f}% | Test MAE={metrics['test']['mae_pct']:.2f}% | Bias={metrics['test']['bias_pct']:+.2f}%")
        return model,metrics

    def _train_classifier_model(self,X_train,y_train,X_val,y_val,X_test,y_test,params,name,train_df,val_df,invert=False):
        model=XGBClassifier(**params)
        train_weights=None; val_weights=None
        if 'cohort_weight' in train_df.columns:
            train_weights=train_df['cohort_weight'].values
            val_weights=val_df['cohort_weight'].values
            logger.info(f"Using cohort weights: mean={train_weights.mean():.3f}")

        model.fit(X_train,y_train,sample_weight=train_weights,eval_set=[(X_val,y_val)],sample_weight_eval_set=[val_weights]if val_weights is not None else None,verbose=False)

        y_train_pred=model.predict(X_train); y_val_pred=model.predict(X_val); y_test_pred=model.predict(X_test)
        y_train_proba=model.predict_proba(X_train)[:,1]; y_val_proba=model.predict_proba(X_val)[:,1]; y_test_proba=model.predict_proba(X_test)[:,1]

        train_acc=accuracy_score(y_train,y_train_pred); val_acc=accuracy_score(y_val,y_val_pred); test_acc=accuracy_score(y_test,y_test_pred)
        test_prec=precision_score(y_test,y_test_pred,zero_division=0); test_rec=recall_score(y_test,y_test_pred,zero_division=0); test_f1=f1_score(y_test,y_test_pred,zero_division=0)

        metrics={"train":{"accuracy":float(train_acc),"avg_confidence":float(np.mean(np.maximum(y_train_proba,1-y_train_proba)))},"val":{"accuracy":float(val_acc),"avg_confidence":float(np.mean(np.maximum(y_val_proba,1-y_val_proba)))},"test":{"accuracy":float(test_acc),"precision":float(test_prec),"recall":float(test_rec),"f1":float(test_f1),"avg_confidence":float(np.mean(np.maximum(y_test_proba,1-y_test_proba)))}}

        logger.info(f"{name}: Train Acc={train_acc:.1%} | Val Acc={val_acc:.1%} | Test Acc={test_acc:.1%} | Prec={test_prec:.1%} | Rec={test_rec:.1%} | F1={test_f1:.1%}")
        return model,metrics,y_val_proba,y_test_proba

    def predict(self,X,current_vix):
        # Get classifier probabilities
        X_up=X[self.up_features]; X_down=X[self.down_features]
        p_up_raw=float(self.up_classifier.predict_proba(X_up)[0][1])
        p_down_raw=float(self.down_classifier.predict_proba(X_down)[0][1])

        # Apply calibration if available
        if self.up_calibrator is not None: p_up=float(self.up_calibrator.transform([p_up_raw])[0])
        else: p_up=p_up_raw
        if self.down_calibrator is not None: p_down=float(self.down_calibrator.transform([p_down_raw])[0])
        else: p_down=p_down_raw

        # Normalize probabilities
        total=p_up+p_down; p_up_norm=p_up/total; p_down_norm=p_down/total

        # Get domain-specific magnitude predictions
        X_exp=X[self.expansion_features]; X_comp=X[self.compression_features]
        expansion_log=float(self.expansion_model.predict(X_exp)[0])
        compression_log=float(self.compression_model.predict(X_comp)[0])
        expansion_log=np.clip(expansion_log,-2,2); compression_log=np.clip(compression_log,-2,2)

        expansion_pct=(np.exp(expansion_log)-1)*100; compression_pct=(np.exp(compression_log)-1)*100
        expansion_pct=np.clip(expansion_pct,-50,100); compression_pct=np.clip(compression_pct,-50,100)

        # WINNER-TAKES-ALL: Use winning classifier's magnitude prediction only
        if p_up>p_down:
            direction="UP"
            magnitude_pct=expansion_pct
            magnitude_log=expansion_log
            classifier_prob=p_up_norm
            ensemble_confidence=self._compute_ensemble_confidence(p_up_norm,expansion_pct,"UP")
            actionable_threshold=self._get_dynamic_threshold(expansion_pct,"UP")
        else:
            direction="DOWN"
            magnitude_pct=compression_pct
            magnitude_log=compression_log
            classifier_prob=p_down_norm
            ensemble_confidence=self._compute_ensemble_confidence(p_down_norm,compression_pct,"DOWN")
            actionable_threshold=self._get_dynamic_threshold(compression_pct,"DOWN")

        actionable=ensemble_confidence>actionable_threshold
        expected_vix=current_vix*(1+magnitude_pct/100)
        current_regime=self._get_regime(current_vix); expected_regime=self._get_regime(expected_vix)
        regime_change=current_regime!=expected_regime

        return {"magnitude_pct":float(magnitude_pct),"magnitude_log":float(magnitude_log),"expected_vix":float(expected_vix),"current_vix":float(current_vix),"direction":direction,"p_up":float(p_up_norm),"p_down":float(p_down_norm),"expansion_magnitude":float(expansion_pct),"compression_magnitude":float(compression_pct),"direction_probability":classifier_prob,"direction_confidence":ensemble_confidence,"current_regime":current_regime,"expected_regime":expected_regime,"regime_change":regime_change,"actionable":actionable,"actionable_threshold":actionable_threshold,"ensemble_enabled":self.ensemble_enabled,"calibration_enabled":self.calibration_enabled}

    def _save_models(self,save_dir):
        save_path=Path(save_dir); save_path.mkdir(parents=True,exist_ok=True)

        with open(save_path/"expansion_model.pkl","wb")as f: pickle.dump(self.expansion_model,f)
        with open(save_path/"compression_model.pkl","wb")as f: pickle.dump(self.compression_model,f)
        with open(save_path/"up_classifier.pkl","wb")as f: pickle.dump(self.up_classifier,f)
        with open(save_path/"down_classifier.pkl","wb")as f: pickle.dump(self.down_classifier,f)

        if self.up_calibrator is not None:
            with open(save_path/"up_calibrator.pkl","wb")as f: pickle.dump(self.up_calibrator,f)
        if self.down_calibrator is not None:
            with open(save_path/"down_calibrator.pkl","wb")as f: pickle.dump(self.down_calibrator,f)

        with open(save_path/"feature_names_expansion.json","w")as f: json.dump(self.expansion_features,f,indent=2)
        with open(save_path/"feature_names_compression.json","w")as f: json.dump(self.compression_features,f,indent=2)
        with open(save_path/"feature_names_up.json","w")as f: json.dump(self.up_features,f,indent=2)
        with open(save_path/"feature_names_down.json","w")as f: json.dump(self.down_features,f,indent=2)
        with open(save_path/"training_metrics.json","w")as f: json.dump(self.metrics,f,indent=2)

    def load(self,models_dir="models"):
        models_path=Path(models_dir)
        with open(models_path/"expansion_model.pkl","rb")as f: self.expansion_model=pickle.load(f)
        with open(models_path/"compression_model.pkl","rb")as f: self.compression_model=pickle.load(f)
        with open(models_path/"up_classifier.pkl","rb")as f: self.up_classifier=pickle.load(f)
        with open(models_path/"down_classifier.pkl","rb")as f: self.down_classifier=pickle.load(f)

        up_cal_file=models_path/"up_calibrator.pkl"
        if up_cal_file.exists():
            with open(up_cal_file,"rb")as f: self.up_calibrator=pickle.load(f)
            logger.info("✅ Loaded UP calibrator")
        down_cal_file=models_path/"down_calibrator.pkl"
        if down_cal_file.exists():
            with open(down_cal_file,"rb")as f: self.down_calibrator=pickle.load(f)
            logger.info("✅ Loaded DOWN calibrator")

        with open(models_path/"feature_names_expansion.json","r")as f: self.expansion_features=json.load(f)
        with open(models_path/"feature_names_compression.json","r")as f: self.compression_features=json.load(f)
        with open(models_path/"feature_names_up.json","r")as f: self.up_features=json.load(f)
        with open(models_path/"feature_names_down.json","r")as f: self.down_features=json.load(f)

        logger.info(f"✅ Loaded 4 models: {len(self.expansion_features)} exp, {len(self.compression_features)} comp, {len(self.up_features)} up, {len(self.down_features)} down features")

    def _generate_diagnostics(self,test_df,save_dir):
        save_path=Path(save_dir); save_path.mkdir(parents=True,exist_ok=True)
        try:
            fig,axes=plt.subplots(2,2,figsize=(16,12))
            fig.suptitle("Asymmetric 4-Model Performance",fontsize=16,fontweight="bold")

            # Expansion regressor accuracy
            ax=axes[0,0]; test_up_mask=test_df['target_direction']==1
            X_exp_test=test_df[test_up_mask][self.expansion_features]
            y_exp_test=test_df[test_up_mask]['target_log_vix_change']
            y_exp_pred=np.clip(self.expansion_model.predict(X_exp_test),-2,2)
            test_pct_actual=(np.exp(y_exp_test)-1)*100; test_pct_pred=(np.exp(y_exp_pred)-1)*100
            ax.scatter(test_pct_pred,test_pct_actual,alpha=0.5,s=30)
            lims=[min(test_pct_pred.min(),test_pct_actual.min()),max(test_pct_pred.max(),test_pct_actual.max())]
            ax.plot(lims,lims,"k--",alpha=0.5); ax.set_xlabel("Predicted (%)"); ax.set_ylabel("Actual (%)")
            ax.set_title("Expansion Regressor (UP samples)"); ax.grid(True,alpha=0.3)

            # Compression regressor accuracy
            ax=axes[0,1]; test_down_mask=test_df['target_direction']==0
            X_comp_test=test_df[test_down_mask][self.compression_features]
            y_comp_test=test_df[test_down_mask]['target_log_vix_change']
            y_comp_pred=np.clip(self.compression_model.predict(X_comp_test),-2,2)
            test_pct_actual=(np.exp(y_comp_test)-1)*100; test_pct_pred=(np.exp(y_comp_pred)-1)*100
            ax.scatter(test_pct_pred,test_pct_actual,alpha=0.5,s=30)
            lims=[min(test_pct_pred.min(),test_pct_actual.min()),max(test_pct_pred.max(),test_pct_actual.max())]
            ax.plot(lims,lims,"k--",alpha=0.5); ax.set_xlabel("Predicted (%)"); ax.set_ylabel("Actual (%)")
            ax.set_title("Compression Regressor (DOWN samples)"); ax.grid(True,alpha=0.3)

            # UP classifier probability distribution
            ax=axes[1,0]; X_up_test=test_df[self.up_features]; y_up_test=test_df['target_direction']
            up_proba_raw=self.up_classifier.predict_proba(X_up_test)[:,1]
            if self.up_calibrator is not None: up_proba=self.up_calibrator.transform(up_proba_raw)
            else: up_proba=up_proba_raw
            correct_mask=(up_proba>0.5)==y_up_test
            ax.hist([up_proba[correct_mask],up_proba[~correct_mask]],bins=20,alpha=0.7,label=['Correct','Wrong'],edgecolor='black')
            ax.set_xlabel("P(UP)"); ax.set_ylabel("Frequency"); ax.set_title("UP Classifier Distribution")
            ax.legend(); ax.grid(True,alpha=0.3); ax.axvline(0.5,color='red',linestyle='--',alpha=0.5)

            # DOWN classifier probability distribution
            ax=axes[1,1]; X_down_test=test_df[self.down_features]; y_down_test=1-test_df['target_direction']
            down_proba_raw=self.down_classifier.predict_proba(X_down_test)[:,1]
            if self.down_calibrator is not None: down_proba=self.down_calibrator.transform(down_proba_raw)
            else: down_proba=down_proba_raw
            correct_mask=(down_proba>0.5)==y_down_test
            ax.hist([down_proba[correct_mask],down_proba[~correct_mask]],bins=20,alpha=0.7,label=['Correct','Wrong'],edgecolor='black')
            ax.set_xlabel("P(DOWN)"); ax.set_ylabel("Frequency"); ax.set_title("DOWN Classifier Distribution")
            ax.legend(); ax.grid(True,alpha=0.3); ax.axvline(0.5,color='red',linestyle='--',alpha=0.5)

            plt.tight_layout(); plot_file=save_path/"model_diagnostics.png"
            plt.savefig(plot_file,dpi=150,bbox_inches="tight"); plt.close()
        except Exception as e: logger.warning(f"  Could not generate plots: {e}")

    def _print_summary(self):
        print("\nASYMMETRIC 4-MODEL TRAINING SUMMARY")
        print(f"Models: Expansion + Compression Regressors | UP + DOWN Classifiers | Training through: {TRAINING_END_DATE}")
        if self.calibration_enabled: print("✓ Classifier probability calibration: ENABLED")
        if self.skip_up_calibration: print("✓ UP calibration: SKIPPED (preserving recall)")
        if self.ensemble_enabled: print("✓ Asymmetric ensemble: WINNER-TAKES-ALL")
        print(f"Expansion: Train={self.metrics['expansion']['train']['mae_pct']:.2f}% | Val={self.metrics['expansion']['val']['mae_pct']:.2f}% | Test={self.metrics['expansion']['test']['mae_pct']:.2f}% | Bias={self.metrics['expansion']['test']['bias_pct']:+.2f}%")
        print(f"Compression: Train={self.metrics['compression']['train']['mae_pct']:.2f}% | Val={self.metrics['compression']['val']['mae_pct']:.2f}% | Test={self.metrics['compression']['test']['mae_pct']:.2f}% | Bias={self.metrics['compression']['test']['bias_pct']:+.2f}%")
        print(f"UP: Train={self.metrics['up_classifier']['train']['accuracy']:.1%} | Val={self.metrics['up_classifier']['val']['accuracy']:.1%} | Test={self.metrics['up_classifier']['test']['accuracy']:.1%} | Prec={self.metrics['up_classifier']['test']['precision']:.1%} | Rec={self.metrics['up_classifier']['test']['recall']:.1%} | F1={self.metrics['up_classifier']['test']['f1']:.1%}")
        if "up_classifier_calibrated"in self.metrics:
            ucal=self.metrics["up_classifier_calibrated"]
            print(f"UP (Cal): Test={ucal['test_accuracy']:.1%} | Prec={ucal['test_precision']:.1%} | Rec={ucal['test_recall']:.1%} | F1={ucal['test_f1']:.1%} | Conf={ucal['avg_confidence']:.1%}")
        print(f"DOWN: Train={self.metrics['down_classifier']['train']['accuracy']:.1%} | Val={self.metrics['down_classifier']['val']['accuracy']:.1%} | Test={self.metrics['down_classifier']['test']['accuracy']:.1%} | Prec={self.metrics['down_classifier']['test']['precision']:.1%} | Rec={self.metrics['down_classifier']['test']['recall']:.1%} | F1={self.metrics['down_classifier']['test']['f1']:.1%}")
        if "down_classifier_calibrated"in self.metrics:
            dcal=self.metrics["down_classifier_calibrated"]
            print(f"DOWN (Cal): Test={dcal['test_accuracy']:.1%} | Prec={dcal['test_precision']:.1%} | Rec={dcal['test_recall']:.1%} | F1={dcal['test_f1']:.1%} | Conf={dcal['avg_confidence']:.1%}")
        print()

def train_asymmetric_forecaster(df,expansion_features=None,compression_features=None,up_features=None,down_features=None,save_dir="models"):
    forecaster=AsymmetricVIXForecaster()
    forecaster.train(df,expansion_features=expansion_features,compression_features=compression_features,up_features=up_features,down_features=down_features,save_dir=save_dir)
    return forecaster
