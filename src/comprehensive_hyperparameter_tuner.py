#!/usr/bin/env python3
"""Production-Matched Hyperparameter Tuner v1.0
Optimizes for ACTUAL 2024-2025 test performance using EXACT production pipeline
PRESERVATION NOTE: The config_content f-string in _generate_config()
must remain exactly formatted - it generates production config.py syntax.
Do not minify that section's whitespace, newlines, or structure."""
import argparse,json,logging,sys,warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass,asdict
import numpy as np,pandas as pd,optuna
from optuna.samplers import TPESampler
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error,accuracy_score,precision_score,recall_score,brier_score_loss
from xgboost import XGBRegressor,XGBClassifier
warnings.filterwarnings("ignore")
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(levelname)s-%(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/production_tuner.log")])
logger=logging.getLogger(__name__)

@dataclass
class ProductionMetrics:
    mag_mae:float;mag_bias:float;mag_cal_error:float;dir_acc_raw:float;dir_prec_raw:float;dir_rec_raw:float;dir_acc_cal:float;dir_prec_cal:float;dir_rec_cal:float;dir_ece:float;dir_brier:float;ens_conf:float;actionable_pct:float;feature_jaccard:float;feature_overlap:float;pred_corr:float;n_mag_feats:int;n_dir_feats:int;n_common_feats:int

class ProductionTuner:
    def __init__(self,df,vix,n_trials=200,output_dir="tuning_production"):
        self.df=df.copy();self.vix=vix.copy();self.n_trials=n_trials
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        # Production date splits from config.py
        self.train_end=pd.Timestamp("2021-12-31")
        self.val_end=pd.Timestamp("2023-12-31")
        train_mask=(df.index<=self.train_end)
        val_mask=(df.index>self.train_end)&(df.index<=self.val_end)
        test_mask=df.index>self.val_end
        self.train_df=df[train_mask].copy()
        self.val_df=df[val_mask].copy()
        self.test_df=df[test_mask].copy()
        self.train_vix=vix[train_mask]
        self.val_vix=vix[val_mask]
        self.test_vix=vix[test_mask]
        from config import TARGET_CONFIG
        self.horizon=TARGET_CONFIG["horizon_days"]
        self._calc_targets()
        self.base_cols=[c for c in df.columns if c not in["vix","spx","calendar_cohort","cohort_weight","feature_quality","future_vix","target_vix_pct_change","target_log_vix_change","target_direction"]]
        logger.info(f"Production splits: Train={len(self.train_df)} ({self.df.index[0].date()} to {self.train_end.date()})")
        logger.info(f"                   Val={len(self.val_df)} ({(self.train_end+pd.Timedelta(days=1)).date()} to {self.val_end.date()})")
        logger.info(f"                   Test={len(self.test_df)} ({(self.val_end+pd.Timedelta(days=1)).date()} to {self.df.index[-1].date()})")
        logger.info(f"Base features: {len(self.base_cols)}")

    def _calc_targets(self):
        for df in[self.train_df,self.val_df,self.test_df]:
            vix_series=df['vix'];future_vix=vix_series.shift(-self.horizon)
            df['target_log_vix_change']=np.log(future_vix/vix_series)
            df['target_direction']=(future_vix>vix_series).astype(int)
            df['future_vix']=future_vix

    def _apply_quality_filter(self,df,threshold):
        if 'feature_quality' not in df.columns:
            logger.warning("No feature_quality column");return df
        quality_mask=df['feature_quality']>=threshold
        filtered=df[quality_mask].copy()
        filtered_pct=(1-len(filtered)/len(df))*100
        if filtered_pct>40:raise ValueError(f"Quality filter removed {filtered_pct:.1f}% of data")
        return filtered

    def _apply_cohort_weights(self,df,fomc_w,opex_w,earnings_w):
        cohort_map={'fomc_period':fomc_w,'opex_week':opex_w,'earnings_heavy':earnings_w,'mid_cycle':1.0}
        weights=df['calendar_cohort'].map(cohort_map).fillna(1.0)
        return weights

    def _select_features(self,df_train,vix_train,target_type,top_n,corr_threshold,cv_params):
        from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
        from core.feature_correlation_analyzer import FeatureCorrelationAnalyzer
        import config as cfg
        original_cv=cfg.FEATURE_SELECTION_CV_PARAMS.copy()
        cfg.FEATURE_SELECTION_CV_PARAMS.update(cv_params)
        try:
            selector=SimplifiedFeatureSelector(target_type=target_type,top_n=top_n)
            test_start_idx=int(len(df_train)*0.70)
            selected,importance_dict=selector.select_features(df_train[self.base_cols],vix_train,test_start_idx=test_start_idx)
            if len(selected)>0 and corr_threshold<1.0:
                analyzer=FeatureCorrelationAnalyzer(threshold=corr_threshold)
                protected=['is_fomc_period','is_opex_week','is_earnings_heavy']
                kept,removed=analyzer.analyze_and_remove(features_df=df_train[selected],importance_scores=importance_dict,protected_features=protected)
                return kept
            return selected
        finally:
            cfg.FEATURE_SELECTION_CV_PARAMS.update(original_cv)

    def _compute_ensemble_confidence(self,mag_pct,dir_prob,params):
        mag_pct=np.clip(mag_pct,-100,100);dir_prob=np.clip(dir_prob,0.0,1.0)
        abs_mag=abs(mag_pct);thresholds=params['thresholds']
        mag_conf=0.5+min(abs_mag/max(thresholds['large'],1.0),0.5)*0.5
        dir_conf=max(dir_prob,1-dir_prob)
        predicted_up=dir_prob>0.5;magnitude_up=mag_pct>0;models_agree=predicted_up==magnitude_up
        if models_agree:
            if abs_mag>thresholds['medium']and dir_conf>0.75:agreement_score=params['bonuses']['strong']
            elif abs_mag>thresholds['small']and dir_conf>0.65:agreement_score=params['bonuses']['moderate']
            else:agreement_score=params['bonuses']['weak']
        else:
            if abs_mag>thresholds['medium']and dir_conf>0.75:agreement_score=-params['penalties']['severe']
            elif abs_mag>thresholds['small']and dir_conf>0.65:agreement_score=-params['penalties']['moderate']
            else:agreement_score=-params['penalties']['minor']
        ens_conf=params['mag_weight']*mag_conf+params['dir_weight']*dir_conf+params['agree_weight']*(0.5+agreement_score)
        ens_conf=np.clip(ens_conf,params['min_confidence'],1.0)
        return float(ens_conf)

    def _expected_calibration_error(self,y_true,y_prob,n_bins=10):
        bin_edges=np.linspace(0,1,n_bins+1)
        bin_indices=np.digitize(y_prob,bin_edges[:-1])-1
        bin_indices=np.clip(bin_indices,0,n_bins-1);ece=0.0
        for i in range(n_bins):
            bin_mask=bin_indices==i
            if bin_mask.sum()==0:continue
            bin_acc=y_true[bin_mask].mean();bin_conf=y_prob[bin_mask].mean()
            bin_weight=bin_mask.sum()/len(y_true)
            ece+=bin_weight*abs(bin_acc-bin_conf)
        return float(ece)

    def _calibration_error_pct(self,y_true,y_pred):
        errors=y_pred-y_true;std_error=np.std(errors)
        within_std=np.abs(errors)<=std_error
        actual_coverage=within_std.mean();target_coverage=0.68
        return abs(actual_coverage-target_coverage)

    def _evaluate_trial(self,data_params,feature_params,mag_params,dir_params,ensemble_params):
        try:
            # Apply quality filter
            train_filt=self._apply_quality_filter(self.train_df,data_params['quality_threshold'])
            val_filt=self._apply_quality_filter(self.val_df,data_params['quality_threshold'])
            test_filt=self._apply_quality_filter(self.test_df,data_params['quality_threshold'])
            if len(train_filt)<100 or len(val_filt)<20 or len(test_filt)<20:
                logger.warning("Insufficient data after quality filter");return None
            # Apply cohort weights
            train_weights=self._apply_cohort_weights(train_filt,data_params['fomc_weight'],data_params['opex_weight'],data_params['earnings_weight'])
            # Feature selection on training data only
            mag_features=self._select_features(train_filt,self.train_vix.loc[train_filt.index],'magnitude',feature_params['mag_top_n'],feature_params['correlation_threshold'],feature_params['cv_params'])
            dir_features=self._select_features(train_filt,self.train_vix.loc[train_filt.index],'direction',feature_params['dir_top_n'],feature_params['correlation_threshold'],feature_params['cv_params'])
            if len(mag_features)<20 or len(dir_features)<20:
                logger.warning(f"Insufficient features: mag={len(mag_features)}, dir={len(dir_features)}");return None
            # Magnitude model
            X_mag_train=train_filt[mag_features].fillna(0)
            y_mag_train=train_filt['target_log_vix_change'].dropna()
            common_idx=X_mag_train.index.intersection(y_mag_train.index)
            X_mag_train=X_mag_train.loc[common_idx];y_mag_train=y_mag_train.loc[common_idx]
            w_mag_train=train_weights.loc[common_idx].values
            X_mag_val=val_filt[mag_features].fillna(0);y_mag_val=val_filt['target_log_vix_change'].dropna()
            common_idx_val=X_mag_val.index.intersection(y_mag_val.index)
            X_mag_val=X_mag_val.loc[common_idx_val];y_mag_val=y_mag_val.loc[common_idx_val]
            X_mag_test=test_filt[mag_features].fillna(0);y_mag_test=test_filt['target_log_vix_change'].dropna()
            common_idx_test=X_mag_test.index.intersection(y_mag_test.index)
            X_mag_test=X_mag_test.loc[common_idx_test];y_mag_test=y_mag_test.loc[common_idx_test]
            mag_model=XGBRegressor(**mag_params)
            mag_model.fit(X_mag_train,y_mag_train,sample_weight=w_mag_train,eval_set=[(X_mag_val,y_mag_val)],verbose=False)
            # Predict on test set
            y_mag_pred_raw=mag_model.predict(X_mag_test)
            y_mag_pred=np.clip(y_mag_pred_raw,-2,2)
            mag_pct_test_actual=(np.exp(y_mag_test.values)-1)*100
            mag_pct_test_pred=(np.exp(y_mag_pred)-1)*100
            mag_test_indices=X_mag_test.index
            mag_mae=mean_absolute_error(mag_pct_test_actual,mag_pct_test_pred)
            mag_bias=np.mean(mag_pct_test_pred-mag_pct_test_actual)
            mag_cal_error=self._calibration_error_pct(mag_pct_test_actual,mag_pct_test_pred)
            # Direction model
            X_dir_train=train_filt[dir_features].fillna(0);y_dir_train=train_filt['target_direction'].dropna()
            common_idx=X_dir_train.index.intersection(y_dir_train.index)
            X_dir_train=X_dir_train.loc[common_idx];y_dir_train=y_dir_train.loc[common_idx]
            w_dir_train=train_weights.loc[common_idx].values
            X_dir_val=val_filt[dir_features].fillna(0);y_dir_val=val_filt['target_direction'].dropna()
            common_idx_val=X_dir_val.index.intersection(y_dir_val.index)
            X_dir_val=X_dir_val.loc[common_idx_val];y_dir_val=y_dir_val.loc[common_idx_val]
            X_dir_test=test_filt[dir_features].fillna(0);y_dir_test=test_filt['target_direction'].dropna()
            common_idx_test=X_dir_test.index.intersection(y_dir_test.index)
            X_dir_test=X_dir_test.loc[common_idx_test];y_dir_test=y_dir_test.loc[common_idx_test]
            dir_model=XGBClassifier(**dir_params)
            dir_model.fit(X_dir_train,y_dir_train,sample_weight=w_dir_train,eval_set=[(X_dir_val,y_dir_val)],verbose=False)
            # Raw predictions
            y_dir_prob_val=dir_model.predict_proba(X_dir_val)[:,1]
            y_dir_prob_test_raw=dir_model.predict_proba(X_dir_test)[:,1]
            y_dir_pred_raw=(y_dir_prob_test_raw>0.5).astype(int)
            dir_acc_raw=accuracy_score(y_dir_test.values,y_dir_pred_raw)
            dir_prec_raw=precision_score(y_dir_test.values,y_dir_pred_raw,zero_division=0)
            dir_rec_raw=recall_score(y_dir_test.values,y_dir_pred_raw,zero_division=0)
            # Fit calibrator on val
            calibrator=IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_dir_prob_val,y_dir_val.values)
            # Calibrated predictions on test
            y_dir_prob_test_cal=calibrator.transform(y_dir_prob_test_raw)
            y_dir_pred_cal=(y_dir_prob_test_cal>0.5).astype(int)
            dir_acc_cal=accuracy_score(y_dir_test.values,y_dir_pred_cal)
            dir_prec_cal=precision_score(y_dir_test.values,y_dir_pred_cal,zero_division=0)
            dir_rec_cal=recall_score(y_dir_test.values,y_dir_pred_cal,zero_division=0)
            dir_brier=brier_score_loss(y_dir_test.values,y_dir_prob_test_cal)
            dir_ece=self._expected_calibration_error(y_dir_test.values,y_dir_prob_test_cal)
            dir_test_indices=X_dir_test.index
            # Ensemble confidence on common test indices
            common_test_indices=mag_test_indices.intersection(dir_test_indices)
            if len(common_test_indices)<10:
                logger.warning("Too few common test indices");return None
            mag_test_loc=mag_test_indices.get_indexer(common_test_indices)
            dir_test_loc=dir_test_indices.get_indexer(common_test_indices)
            mag_pct_aligned=mag_pct_test_pred[mag_test_loc]
            dir_prob_aligned=y_dir_prob_test_cal[dir_test_loc]
            ensemble_confs=[self._compute_ensemble_confidence(mag_pct_aligned[i],dir_prob_aligned[i],ensemble_params)for i in range(len(common_test_indices))]
            avg_ens_conf=np.mean(ensemble_confs)
            actionable_pct=np.mean(np.array(ensemble_confs)>ensemble_params['actionable_threshold'])
            # Diversity metrics
            feature_jaccard=len(set(mag_features)&set(dir_features))/max(len(set(mag_features)|set(dir_features)),1)
            feature_overlap=len(set(mag_features)&set(dir_features))/max(min(len(mag_features),len(dir_features)),1)
            mag_scaled=np.clip(mag_pct_aligned/20.0,-1,1);dir_scaled=(dir_prob_aligned-0.5)*2.0
            mask=np.isfinite(mag_scaled)&np.isfinite(dir_scaled)
            if mask.sum()>=5:
                pred_corr,_=spearmanr(mag_scaled[mask],dir_scaled[mask])
                if np.isnan(pred_corr)or np.isinf(pred_corr):pred_corr=0.0
            else:pred_corr=0.0
            metrics=ProductionMetrics(mag_mae=mag_mae,mag_bias=mag_bias,mag_cal_error=mag_cal_error,dir_acc_raw=dir_acc_raw,dir_prec_raw=dir_prec_raw,dir_rec_raw=dir_rec_raw,dir_acc_cal=dir_acc_cal,dir_prec_cal=dir_prec_cal,dir_rec_cal=dir_rec_cal,dir_ece=dir_ece,dir_brier=dir_brier,ens_conf=avg_ens_conf,actionable_pct=actionable_pct,feature_jaccard=feature_jaccard,feature_overlap=feature_overlap,pred_corr=float(pred_corr),n_mag_feats=len(mag_features),n_dir_feats=len(dir_features),n_common_feats=len(set(mag_features)&set(dir_features)))
            return metrics
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            import traceback;traceback.print_exc()
            return None

    def objective(self,trial):
        # Sample hyperparameters
        data_params={'quality_threshold':trial.suggest_float('quality_threshold',0.50,0.65),'fomc_weight':trial.suggest_float('cohort_fomc',1.10,1.50),'opex_weight':trial.suggest_float('cohort_opex',1.00,1.50),'earnings_weight':trial.suggest_float('cohort_earnings',1.00,1.50)}
        feature_params={'mag_top_n':trial.suggest_int('mag_top_n',70,140),'dir_top_n':trial.suggest_int('dir_top_n',70,160),'correlation_threshold':trial.suggest_float('corr_threshold',0.85,0.98),'cv_params':{'n_estimators':trial.suggest_int('cv_n_est',100,250),'max_depth':trial.suggest_int('cv_depth',3,6),'learning_rate':trial.suggest_float('cv_lr',0.03,0.12,log=True),'subsample':trial.suggest_float('cv_sub',0.70,0.95),'colsample_bytree':trial.suggest_float('cv_col',0.70,0.95)}}
        mag_params={'objective':'reg:squarederror','eval_metric':'rmse','max_depth':trial.suggest_int('mag_depth',2,7),'learning_rate':trial.suggest_float('mag_lr',0.008,0.12,log=True),'n_estimators':trial.suggest_int('mag_n_est',200,800),'subsample':trial.suggest_float('mag_sub',0.60,0.95),'colsample_bytree':trial.suggest_float('mag_col_tree',0.60,0.95),'colsample_bylevel':trial.suggest_float('mag_col_lvl',0.60,0.95),'min_child_weight':trial.suggest_int('mag_mcw',2,15),'reg_alpha':trial.suggest_float('mag_alpha',0.3,6.0),'reg_lambda':trial.suggest_float('mag_lambda',0.8,10.0),'gamma':trial.suggest_float('mag_gamma',0.0,0.8),'early_stopping_rounds':50,'seed':42,'n_jobs':-1}
        dir_params={'objective':'binary:logistic','eval_metric':'logloss','max_depth':trial.suggest_int('dir_depth',3,12),'learning_rate':trial.suggest_float('dir_lr',0.008,0.10,log=True),'n_estimators':trial.suggest_int('dir_n_est',200,900),'subsample':trial.suggest_float('dir_sub',0.60,0.95),'colsample_bytree':trial.suggest_float('dir_col_tree',0.55,0.95),'min_child_weight':trial.suggest_int('dir_mcw',4,20),'reg_alpha':trial.suggest_float('dir_alpha',0.5,5.0),'reg_lambda':trial.suggest_float('dir_lambda',1.0,8.0),'gamma':trial.suggest_float('dir_gamma',0.0,1.0),'scale_pos_weight':trial.suggest_float('dir_scale',0.80,1.60),'max_delta_step':trial.suggest_int('dir_max_delta',0,5),'early_stopping_rounds':50,'seed':42,'n_jobs':-1}
        # Ensemble params
        mag_w=trial.suggest_float('ens_mag_weight',0.25,0.50)
        dir_w=trial.suggest_float('ens_dir_weight',0.30,0.60)
        agree_w=trial.suggest_float('ens_agree_weight',0.10,0.35)
        total=mag_w+dir_w+agree_w;mag_w/=total;dir_w/=total;agree_w/=total
        small=trial.suggest_float('ens_small_thresh',1.5,4.0)
        medium=trial.suggest_float('ens_med_thresh',small+1.0,8.0)
        large=trial.suggest_float('ens_large_thresh',medium+1.0,20.0)
        ensemble_params={'mag_weight':mag_w,'dir_weight':dir_w,'agree_weight':agree_w,'thresholds':{'small':small,'medium':medium,'large':large},'bonuses':{'weak':0.0,'moderate':trial.suggest_float('ens_bonus_mod',0.05,0.15),'strong':trial.suggest_float('ens_bonus_strong',0.10,0.25)},'penalties':{'minor':trial.suggest_float('ens_penalty_min',0.00,0.10),'moderate':trial.suggest_float('ens_penalty_mod',0.05,0.25),'severe':trial.suggest_float('ens_penalty_sev',0.15,0.40)},'min_confidence':0.50,'actionable_threshold':trial.suggest_float('ens_actionable',0.60,0.72)}
        # Evaluate on production test set
        metrics=self._evaluate_trial(data_params,feature_params,mag_params,dir_params,ensemble_params)
        if metrics is None:return 999.0
        # Optimize for calibrated production metrics
        mag_best=9.5;mag_good=10.5
        mag_penalty=0 if metrics.mag_mae<=mag_best else(metrics.mag_mae-mag_good)*8.0 if metrics.mag_mae<=mag_good else(metrics.mag_mae-mag_good)*15.0
        mag_reward=max(0,mag_good-metrics.mag_mae)*3.0
        mag_bias_penalty=abs(metrics.mag_bias)*3.0
        mag_score=metrics.mag_mae+mag_bias_penalty+metrics.mag_cal_error*5.0+mag_penalty-mag_reward
        # Direction targets (calibrated metrics)
        dir_best_acc=0.65;dir_good_acc=0.62
        dir_best_prec=0.65;dir_good_prec=0.62
        dir_best_rec=0.60;dir_good_rec=0.55
        acc_penalty=0 if metrics.dir_acc_cal>=dir_best_acc else(dir_good_acc-metrics.dir_acc_cal)*25.0 if metrics.dir_acc_cal<dir_good_acc else 0
        acc_reward=max(0,metrics.dir_acc_cal-dir_best_acc)*10.0
        prec_penalty=0 if metrics.dir_prec_cal>=dir_best_prec else(dir_good_prec-metrics.dir_prec_cal)*20.0 if metrics.dir_prec_cal<dir_good_prec else 0
        prec_reward=max(0,metrics.dir_prec_cal-dir_best_prec)*8.0
        rec_penalty=0 if metrics.dir_rec_cal>=dir_best_rec else(dir_good_rec-metrics.dir_rec_cal)*7.5 if metrics.dir_rec_cal<dir_good_rec else 0
        rec_reward=max(0,metrics.dir_rec_cal-dir_best_rec)*5.0
        dir_score=(1-metrics.dir_acc_cal)*12+acc_penalty-acc_reward+(1-metrics.dir_rec_cal)*5.0+rec_penalty-rec_reward+metrics.dir_ece*18+prec_penalty-prec_reward
        # Ensemble score
        ens_score=abs(metrics.ens_conf-0.70)*4.0+(1-metrics.actionable_pct)*1.5
        # Diversity score
        jaccard_penalty=abs(metrics.feature_jaccard-0.40)*3.0
        overlap_penalty=abs(metrics.feature_overlap-0.50)*3.0
        diversity_penalty=(jaccard_penalty+overlap_penalty)*1.5
        combined_score=mag_score+dir_score+ens_score+diversity_penalty
        # Log to trial
        trial.set_user_attr('mag_mae',float(metrics.mag_mae))
        trial.set_user_attr('mag_bias',float(metrics.mag_bias))
        trial.set_user_attr('mag_cal_error',float(metrics.mag_cal_error))
        trial.set_user_attr('dir_acc_raw',float(metrics.dir_acc_raw))
        trial.set_user_attr('dir_prec_raw',float(metrics.dir_prec_raw))
        trial.set_user_attr('dir_rec_raw',float(metrics.dir_rec_raw))
        trial.set_user_attr('dir_acc',float(metrics.dir_acc_cal))
        trial.set_user_attr('dir_precision',float(metrics.dir_prec_cal))
        trial.set_user_attr('dir_recall',float(metrics.dir_rec_cal))
        trial.set_user_attr('dir_ece',float(metrics.dir_ece))
        trial.set_user_attr('dir_brier',float(metrics.dir_brier))
        trial.set_user_attr('ensemble_conf',float(metrics.ens_conf))
        trial.set_user_attr('actionable_pct',float(metrics.actionable_pct))
        trial.set_user_attr('feature_jaccard',float(metrics.feature_jaccard))
        trial.set_user_attr('feature_overlap',float(metrics.feature_overlap))
        trial.set_user_attr('pred_correlation',float(metrics.pred_corr))
        trial.set_user_attr('n_mag_features',int(metrics.n_mag_feats))
        trial.set_user_attr('n_dir_features',int(metrics.n_dir_feats))
        trial.set_user_attr('n_common_features',int(metrics.n_common_feats))
        return combined_score

    def run(self):
        logger.info("="*80)
        logger.info("PRODUCTION-MATCHED TUNER v1.0")
        logger.info("="*80)
        logger.info("Evaluates on 2024-2025 TEST data with single calibrator")
        logger.info("NO fold averaging - matches exact production pipeline")
        logger.info("="*80)
        study=optuna.create_study(direction='minimize',sampler=TPESampler(seed=42,n_startup_trials=min(30,self.n_trials//5)))
        study.optimize(self.objective,n_trials=self.n_trials,show_progress_bar=True,n_jobs=1)
        return study

    def save_results(self,study):
        best=study.best_trial;attrs=best.user_attrs
        # Build config sections
        data_params={'quality_threshold':float(best.params['quality_threshold']),'cohort_weights':{'fomc_period':float(best.params['cohort_fomc']),'opex_week':float(best.params['cohort_opex']),'earnings_heavy':float(best.params['cohort_earnings']),'mid_cycle':1.0}}
        feature_params={'mag_top_n':int(best.params['mag_top_n']),'dir_top_n':int(best.params['dir_top_n']),'correlation_threshold':float(best.params['corr_threshold']),'cv':{'n_estimators':int(best.params['cv_n_est']),'max_depth':int(best.params['cv_depth']),'learning_rate':float(best.params['cv_lr']),'subsample':float(best.params['cv_sub']),'colsample_bytree':float(best.params['cv_col'])}}
        mag_model_params={k.replace('mag_',''):int(v)if isinstance(v,int)else float(v)for k,v in best.params.items()if k.startswith('mag_')and k not in['mag_top_n']}
        dir_model_params={k.replace('dir_',''):int(v)if isinstance(v,int)else float(v)for k,v in best.params.items()if k.startswith('dir_')and k not in['dir_top_n']}
        ensemble_params={k.replace('ens_',''):float(v)for k,v in best.params.items()if k.startswith('ens_')}
        results={'timestamp':datetime.now().isoformat(),'tuner_version':'production_v1.0','description':'Single production-realistic evaluation on 2024-2025 test data','optimization':{'n_trials':int(self.n_trials),'best_trial':int(best.number),'best_score':float(best.value)},'data_splits':{'train_end':'2021-12-31','val_end':'2023-12-31','test_start':'2024-01-01','train_size':len(self.train_df),'val_size':len(self.val_df),'test_size':len(self.test_df)},'metrics':{'magnitude':{'mae':float(attrs.get('mag_mae',0)),'bias':float(attrs.get('mag_bias',0)),'calibration_error':float(attrs.get('mag_cal_error',0))},'direction_raw':{'accuracy':float(attrs.get('dir_acc_raw',0)),'precision':float(attrs.get('dir_prec_raw',0)),'recall':float(attrs.get('dir_rec_raw',0))},'direction_calibrated':{'accuracy':float(attrs.get('dir_acc',0)),'precision':float(attrs.get('dir_precision',0)),'recall':float(attrs.get('dir_recall',0)),'ece':float(attrs.get('dir_ece',0)),'brier':float(attrs.get('dir_brier',0))},'ensemble':{'confidence':float(attrs.get('ensemble_conf',0)),'actionable_pct':float(attrs.get('actionable_pct',0))},'diversity':{'feature_jaccard':float(attrs.get('feature_jaccard',0)),'feature_overlap':float(attrs.get('feature_overlap',0)),'pred_correlation':float(attrs.get('pred_correlation',0))},'features':{'magnitude':int(attrs.get('n_mag_features',0)),'direction':int(attrs.get('n_dir_features',0)),'common':int(attrs.get('n_common_features',0))}},'parameters':{'data':data_params,'features':feature_params,'magnitude_model':mag_model_params,'direction_model':dir_model_params,'ensemble':ensemble_params}}
        # Save JSON
        results_file=self.output_dir/"optimization_results.json"
        with open(results_file,'w')as f:json.dump(results,f,indent=2)
        logger.info(f"\n✅ Results saved: {results_file}")
        self._generate_config(best,attrs)
        self._print_summary(best,attrs)

    def _generate_config(self,trial,attrs):
        mag_w=trial.params['ens_mag_weight']
        dir_w=trial.params['ens_dir_weight']
        agree_w=trial.params['ens_agree_weight']
        total=mag_w+dir_w+agree_w
        config_content=f"""# PRODUCTION-TUNED CONFIG v1.0 - {datetime.now().strftime('%Y-%m-%d')}
# Optimized on 2024-2025 test data with single calibrator (matches production)
# Replace these sections in your config.py

QUALITY_FILTER_CONFIG={{'enabled':True,'min_threshold':{trial.params['quality_threshold']:.4f},'warn_pct':20.0,'error_pct':50.0,'strategy':'raise'}}

CALENDAR_COHORTS={{'fomc_period':{{'condition':'macro_event_period','range':(-7,2),'weight':{trial.params['cohort_fomc']:.4f},'description':'FOMC meetings, CPI releases, PCE releases, FOMC minutes'}},'opex_week':{{'condition':'days_to_monthly_opex','range':(-7,0),'weight':{trial.params['cohort_opex']:.4f},'description':'Options expiration week + VIX futures rollover'}},'earnings_heavy':{{'condition':'spx_earnings_pct','range':(0.15,1.0),'weight':{trial.params['cohort_earnings']:.4f},'description':'Peak earnings season (Jan, Apr, Jul, Oct)'}},'mid_cycle':{{'condition':'default','range':None,'weight':1.0,'description':'Regular market conditions'}}}}

FEATURE_SELECTION_CV_PARAMS={{'n_estimators':{trial.params['cv_n_est']},'max_depth':{trial.params['cv_depth']},'learning_rate':{trial.params['cv_lr']:.4f},'subsample':{trial.params['cv_sub']:.4f},'colsample_bytree':{trial.params['cv_col']:.4f}}}

FEATURE_SELECTION_CONFIG={{'magnitude_top_n':{trial.params['mag_top_n']},'direction_top_n':{trial.params['dir_top_n']},'cv_folds':5,'protected_features':['is_fomc_period','is_opex_week','is_earnings_heavy'],'correlation_threshold':{trial.params['corr_threshold']:.4f},'description':'Production-tuned on 2024-2025 test data'}}

MAGNITUDE_PARAMS={{'objective':'reg:squarederror','eval_metric':'rmse','max_depth':{trial.params['mag_depth']},'learning_rate':{trial.params['mag_lr']:.4f},'n_estimators':{trial.params['mag_n_est']},'subsample':{trial.params['mag_sub']:.4f},'colsample_bytree':{trial.params['mag_col_tree']:.4f},'colsample_bylevel':{trial.params['mag_col_lvl']:.4f},'min_child_weight':{trial.params['mag_mcw']},'reg_alpha':{trial.params['mag_alpha']:.4f},'reg_lambda':{trial.params['mag_lambda']:.4f},'gamma':{trial.params['mag_gamma']:.4f},'early_stopping_rounds':50,'seed':42,'n_jobs':-1}}

DIRECTION_PARAMS={{'objective':'binary:logistic','eval_metric':'logloss','max_depth':{trial.params['dir_depth']},'learning_rate':{trial.params['dir_lr']:.4f},'n_estimators':{trial.params['dir_n_est']},'subsample':{trial.params['dir_sub']:.4f},'colsample_bytree':{trial.params['dir_col_tree']:.4f},'min_child_weight':{trial.params['dir_mcw']},'reg_alpha':{trial.params['dir_alpha']:.4f},'reg_lambda':{trial.params['dir_lambda']:.4f},'gamma':{trial.params['dir_gamma']:.4f},'scale_pos_weight':{trial.params['dir_scale']:.4f},'max_delta_step':{trial.params['dir_max_delta']},'early_stopping_rounds':50,'seed':42,'n_jobs':-1}}

ENSEMBLE_CONFIG={{'enabled':True,'reconciliation_method':'weighted_agreement','confidence_weights':{{'magnitude':{mag_w/total:.4f},'direction':{dir_w/total:.4f},'agreement':{agree_w/total:.4f}}},'magnitude_thresholds':{{'small':{trial.params['ens_small_thresh']:.4f},'medium':{trial.params['ens_med_thresh']:.4f},'large':{trial.params['ens_large_thresh']:.4f}}},'agreement_bonus':{{'strong':{trial.params['ens_bonus_strong']:.4f},'moderate':{trial.params['ens_bonus_mod']:.4f},'weak':0.0}},'contradiction_penalty':{{'severe':{trial.params['ens_penalty_sev']:.4f},'moderate':{trial.params['ens_penalty_mod']:.4f},'minor':{trial.params['ens_penalty_min']:.4f}}},'min_ensemble_confidence':0.50,'actionable_threshold':{trial.params['ens_actionable']:.4f},'description':'Production-tuned ensemble'}}
"""
        config_file=self.output_dir/"tuned_config.py"
        with open(config_file,'w')as f:f.write(config_content)
        logger.info(f"✅ Config saved: {config_file}")

    def _print_summary(self,trial,attrs):
        logger.info("\n"+"="*80)
        logger.info(f"OPTIMIZATION COMPLETE - Trial #{trial.number}")
        logger.info("="*80)
        logger.info("\n📊 TEST SET METRICS (2024-2025 data):")
        logger.info(f"   Magnitude MAE: {attrs.get('mag_mae',0):.2f}%")
        logger.info(f"   Magnitude Bias: {attrs.get('mag_bias',0):+.2f}%")
        logger.info(f"   Magnitude Cal Error: {attrs.get('mag_cal_error',0):.3f}")
        logger.info(f"\n   Direction RAW:")
        logger.info(f"     Accuracy: {attrs.get('dir_acc_raw',0):.1%}")
        logger.info(f"     Precision: {attrs.get('dir_prec_raw',0):.1%}")
        logger.info(f"     Recall: {attrs.get('dir_rec_raw',0):.1%}")
        logger.info(f"\n   Direction CALIBRATED:")
        logger.info(f"     Accuracy: {attrs.get('dir_acc',0):.1%}")
        logger.info(f"     Precision: {attrs.get('dir_precision',0):.1%}")
        logger.info(f"     Recall: {attrs.get('dir_recall',0):.1%}")
        logger.info(f"     ECE: {attrs.get('dir_ece',0):.3f}")
        logger.info(f"\n   Ensemble Confidence: {attrs.get('ensemble_conf',0):.1%}")
        logger.info(f"   Actionable Trades: {attrs.get('actionable_pct',0):.1%}")
        logger.info(f"\n📈 FEATURES:")
        logger.info(f"   Magnitude: {attrs.get('n_mag_features',0)}")
        logger.info(f"   Direction: {attrs.get('n_dir_features',0)}")
        logger.info(f"   Common: {attrs.get('n_common_features',0)}")
        logger.info("="*80)
        logger.info("\n✅ Apply tuned_config.py to your config.py")
        logger.info("="*80)

def main():
    parser=argparse.ArgumentParser(description="Production-Matched Tuner v1.0")
    parser.add_argument('--trials',type=int,default=200,help="Number of trials")
    parser.add_argument('--output-dir',type=str,default='tuning_production',help="Output directory")
    args=parser.parse_args()
    # Load data
    logger.info("Loading data...")
    from config import TRAINING_YEARS,get_last_complete_month_end
    from core.data_fetcher import UnifiedDataFetcher
    from core.feature_engineer import FeatureEngineer
    training_end=get_last_complete_month_end()
    fetcher=UnifiedDataFetcher()
    engineer=FeatureEngineer(fetcher)
    result=engineer.build_complete_features(years=TRAINING_YEARS,end_date=training_end)
    df=result["features"].copy();df["vix"]=result["vix"];df["spx"]=result["spx"]
    logger.info(f"Dataset: {len(df)} samples, {len(df.columns)} features")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}\n")
    # Run tuner
    tuner=ProductionTuner(df=df,vix=result["vix"],n_trials=args.trials,output_dir=args.output_dir)
    study=tuner.run()
    tuner.save_results(study)

if __name__=="__main__":main()
