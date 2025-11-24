#!/usr/bin/env python3
import argparse,json,logging,sys,warnings
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
import optuna
from xgboost import XGBRegressor,XGBClassifier
from config import TRAINING_YEARS,XGBOOST_CONFIG,get_last_complete_month_end
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.target_calculator import TargetCalculator
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
import config as cfg
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(message)s")
logger=logging.getLogger(__name__)
class FixedDualModelTuner:
    def __init__(self,df,vix,n_trials=50,output_dir="tuning_results"):
        self.df=df;self.vix=vix;self.n_trials=n_trials;self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True);self.target_calculator=TargetCalculator()
        total=len(df);test_size=XGBOOST_CONFIG["cv_config"]["test_size"];val_size=XGBOOST_CONFIG["cv_config"]["val_size"]
        self.train_end=int(total*(1-test_size-val_size));self.val_end=int(total*(1-test_size));self.test_start=self.val_end
        self.base_cols=[c for c in df.columns if c not in["vix","spx","calendar_cohort","cohort_weight","feature_quality"]]
        logger.info(f"Data: Train={self.train_end} Val={self.val_end-self.train_end} Test={total-self.val_end}")
    def run_feature_selection(self,cv_params,target_type,top_n):
        original=cfg.FEATURE_SELECTION_CV_PARAMS.copy()
        try:
            cfg.FEATURE_SELECTION_CV_PARAMS.update(cv_params);selector=SimplifiedFeatureSelector(target_type=target_type,top_n=top_n);selected,_=selector.select_features(self.df[self.base_cols],self.vix,test_start_idx=self.test_start);return selected
        except Exception as e:
            logger.error(f"Feature selection error: {e}");return[]
        finally:
            cfg.FEATURE_SELECTION_CV_PARAMS.update(original)
    def objective_magnitude(self,trial,shared_cv_params):
        top_n=trial.suggest_int('mag_top_n',50,150);selected=self.run_feature_selection(shared_cv_params,'magnitude',top_n)
        if len(selected)<10:return 999.0
        params={'objective':'reg:squarederror','eval_metric':'rmse','max_depth':trial.suggest_int('mag_max_depth',2,5),'learning_rate':trial.suggest_float('mag_lr',0.01,0.12,log=True),'n_estimators':trial.suggest_int('mag_n_est',100,500),'subsample':trial.suggest_float('mag_sub',0.7,0.95),'colsample_bytree':trial.suggest_float('mag_col_tree',0.7,0.95),'colsample_bylevel':trial.suggest_float('mag_col_lvl',0.7,0.95),'min_child_weight':trial.suggest_int('mag_mcw',3,15),'reg_alpha':trial.suggest_float('mag_alpha',1.0,4.0),'reg_lambda':trial.suggest_float('mag_lambda',1.5,5.0),'gamma':trial.suggest_float('mag_gamma',0.1,0.8),'early_stopping_rounds':50,'seed':42,'n_jobs':-1}
        try:
            df_targets=self.target_calculator.calculate_all_targets(self.df.copy(),vix_col="vix");X,y=df_targets[selected].copy(),df_targets["target_log_vix_change"];valid=~(X.isna().any(axis=1)|y.isna());X,y=X[valid],y[valid]
            if len(X)<100:return 999.0
            X_tr,y_tr=X.iloc[:self.train_end],y.iloc[:self.train_end];X_val,y_val=X.iloc[self.train_end:self.val_end],y.iloc[self.train_end:self.val_end]
            if len(X_val)<10:return 999.0
            model=XGBRegressor(**params);model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False);y_pred=model.predict(X_val)
            if np.isnan(y_pred).any()or np.isinf(y_pred).any():logger.warning(f"Trial {trial.number}: NaN/Inf in predictions");return 999.0
            if np.abs(y_pred).max()>2.5:logger.warning(f"Trial {trial.number}: Extreme prediction {np.abs(y_pred).max():.3f}");return 999.0
            y_pred_clipped=np.clip(y_pred,-2,2);val_pct=(np.exp(y_val)-1)*100;pred_pct=(np.exp(y_pred_clipped)-1)*100
            if np.isnan(pred_pct).any()or np.isinf(pred_pct).any():logger.warning(f"Trial {trial.number}: NaN/Inf after exp");return 999.0
            mae=np.mean(np.abs(pred_pct-val_pct));bias=np.mean(pred_pct-val_pct)
            if np.isnan(mae)or np.isnan(bias):return 999.0
            trial.set_user_attr('val_mae',float(mae));trial.set_user_attr('val_bias',float(bias));trial.set_user_attr('n_features',len(selected));trial.set_user_attr('max_pred',float(np.abs(y_pred).max()));return mae+0.5*np.abs(bias)
        except Exception as e:
            logger.warning(f"Magnitude trial failed: {e}");return 999.0
    def objective_direction(self,trial,shared_cv_params):
        top_n=trial.suggest_int('dir_top_n',50,200);selected=self.run_feature_selection(shared_cv_params,'direction',top_n)
        if len(selected)<10:return -0.01
        params={'objective':'binary:logistic','eval_metric':'logloss','max_depth':trial.suggest_int('dir_max_depth',2,5),'learning_rate':trial.suggest_float('dir_lr',0.005,0.08,log=True),'n_estimators':trial.suggest_int('dir_n_est',100,500),'subsample':trial.suggest_float('dir_sub',0.65,0.9),'colsample_bytree':trial.suggest_float('dir_col_tree',0.65,0.9),'min_child_weight':trial.suggest_int('dir_mcw',5,15),'reg_alpha':trial.suggest_float('dir_alpha',1.0,3.0),'reg_lambda':trial.suggest_float('dir_lambda',1.5,4.0),'gamma':trial.suggest_float('dir_gamma',0.1,0.5),'scale_pos_weight':trial.suggest_float('dir_scale',0.8,1.2),'early_stopping_rounds':50,'seed':42,'n_jobs':-1}
        try:
            df_targets=self.target_calculator.calculate_all_targets(self.df.copy(),vix_col="vix");X,y=df_targets[selected].copy(),df_targets["target_direction"];valid=~(X.isna().any(axis=1)|y.isna());X,y=X[valid],y[valid]
            if len(X)<100:return -0.01
            X_tr,y_tr=X.iloc[:self.train_end],y.iloc[:self.train_end];X_val,y_val=X.iloc[self.train_end:self.val_end],y.iloc[self.train_end:self.val_end]
            if len(X_val)<10:return -0.01
            model=XGBClassifier(**params);model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False);y_pred=model.predict(X_val)
            acc=(y_pred==y_val).mean();prec=((y_pred==1)&(y_val==1)).sum()/max((y_pred==1).sum(),1);rec=((y_pred==1)&(y_val==1)).sum()/max((y_val==1).sum(),1);f1=2*(prec*rec)/max(prec+rec,1e-8)
            if np.isnan(f1)or f1<0.01:return -0.01
            trial.set_user_attr('val_acc',float(acc));trial.set_user_attr('val_prec',float(prec));trial.set_user_attr('val_rec',float(rec));trial.set_user_attr('val_f1',float(f1));trial.set_user_attr('n_features',len(selected));return -f1
        except Exception as e:
            logger.warning(f"Direction trial failed: {e}");return -0.01
    def tune_cv_params(self):
        logger.info("="*80);logger.info("TUNING SHARED CV PARAMS (20 trials)");logger.info("="*80)
        def cv_objective(trial):
            cv_params={'n_estimators':trial.suggest_int('cv_n_est',50,200),'max_depth':trial.suggest_int('cv_depth',3,6),'learning_rate':trial.suggest_float('cv_lr',0.02,0.12,log=True),'subsample':trial.suggest_float('cv_sub',0.7,0.95),'colsample_bytree':trial.suggest_float('cv_col',0.7,0.95)}
            try:
                sel_mag=self.run_feature_selection(cv_params,'magnitude',100);sel_dir=self.run_feature_selection(cv_params,'direction',100)
                if len(sel_mag)<20 or len(sel_dir)<20:return 999.0
                return -(len(sel_mag)+len(sel_dir))/2
            except:
                return 999.0
        study=optuna.create_study(direction='minimize');study.optimize(cv_objective,n_trials=20,show_progress_bar=True)
        best={'n_estimators':study.best_params['cv_n_est'],'max_depth':study.best_params['cv_depth'],'learning_rate':study.best_params['cv_lr'],'subsample':study.best_params['cv_sub'],'colsample_bytree':study.best_params['cv_col']}
        logger.info(f"\nBest CV params: {best}\n");return best
    def tune_models(self,cv_params):
        logger.info("="*80);logger.info(f"TUNING MAGNITUDE ({self.n_trials} trials)");logger.info("="*80)
        mag_study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42));mag_study.optimize(lambda t:self.objective_magnitude(t,cv_params),n_trials=self.n_trials,show_progress_bar=True)
        logger.info("\n"+"="*80);logger.info(f"TUNING DIRECTION ({self.n_trials} trials)");logger.info("="*80)
        dir_study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42));dir_study.optimize(lambda t:self.objective_direction(t,cv_params),n_trials=self.n_trials,show_progress_bar=True)
        return mag_study,dir_study
    def save_results(self,cv_params,mag_study,dir_study):
        mag_best=mag_study.best_trial;dir_best=dir_study.best_trial
        results={'timestamp':datetime.now().isoformat(),'shared_cv_params':cv_params,'magnitude':{'top_n':mag_best.params['mag_top_n'],'params':{k:v for k,v in mag_best.params.items()if k!='mag_top_n'},'metrics':mag_best.user_attrs},'direction':{'top_n':dir_best.params['dir_top_n'],'params':{k:v for k,v in dir_best.params.items()if k!='dir_top_n'},'metrics':dir_best.user_attrs}}
        with open(self.output_dir/"results.json",'w')as f:json.dump(results,f,indent=2)
        config=f"""# TUNED CONFIG - {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Mag MAE: {mag_best.user_attrs.get('val_mae',0):.2f}% | Bias: {mag_best.user_attrs.get('val_bias',0):+.2f}% | Max Pred: {mag_best.user_attrs.get('max_pred',0):.3f}
# Dir F1: {dir_best.user_attrs.get('val_f1',0):.4f} | Acc: {dir_best.user_attrs.get('val_acc',0):.1%}

FEATURE_SELECTION_CV_PARAMS = {{
    'n_estimators': {cv_params['n_estimators']},
    'max_depth': {cv_params['max_depth']},
    'learning_rate': {cv_params['learning_rate']:.4f},
    'subsample': {cv_params['subsample']:.4f},
    'colsample_bytree': {cv_params['colsample_bytree']:.4f}
}}

FEATURE_SELECTION_CONFIG = {{
    'magnitude_top_n': {mag_best.params['mag_top_n']},
    'direction_top_n': {dir_best.params['dir_top_n']},
    'cv_folds': 5,
    'protected_features': ['is_fomc_period','is_opex_week','is_earnings_heavy'],
    'correlation_threshold': 1
}}

XGBOOST_CONFIG['magnitude_params'].update({{
    'max_depth': {mag_best.params['mag_max_depth']},
    'learning_rate': {mag_best.params['mag_lr']:.4f},
    'n_estimators': {mag_best.params['mag_n_est']},
    'subsample': {mag_best.params['mag_sub']:.4f},
    'colsample_bytree': {mag_best.params['mag_col_tree']:.4f},
    'colsample_bylevel': {mag_best.params['mag_col_lvl']:.4f},
    'min_child_weight': {mag_best.params['mag_mcw']},
    'reg_alpha': {mag_best.params['mag_alpha']:.4f},
    'reg_lambda': {mag_best.params['mag_lambda']:.4f},
    'gamma': {mag_best.params['mag_gamma']:.4f}
}})

XGBOOST_CONFIG['direction_params'].update({{
    'max_depth': {dir_best.params['dir_max_depth']},
    'learning_rate': {dir_best.params['dir_lr']:.4f},
    'n_estimators': {dir_best.params['dir_n_est']},
    'subsample': {dir_best.params['dir_sub']:.4f},
    'colsample_bytree': {dir_best.params['dir_col_tree']:.4f},
    'min_child_weight': {dir_best.params['dir_mcw']},
    'reg_alpha': {dir_best.params['dir_alpha']:.4f},
    'reg_lambda': {dir_best.params['dir_lambda']:.4f},
    'gamma': {dir_best.params['dir_gamma']:.4f},
    'scale_pos_weight': {dir_best.params['dir_scale']:.4f}
}})
"""
        with open(self.output_dir/"config_updates.py",'w')as f:f.write(config)
        logger.info(f"\n✅ Saved: {self.output_dir}/results.json");logger.info(f"✅ Saved: {self.output_dir}/config_updates.py")
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trials',type=int,default=50);parser.add_argument('--output-dir',type=str,default='tuning_results');args=parser.parse_args()
    logger.info("Loading data...");training_end=get_last_complete_month_end();fetcher=UnifiedDataFetcher();engineer=FeatureEngineer(fetcher);result=engineer.build_complete_features(years=TRAINING_YEARS,end_date=training_end)
    df=result["features"].copy();df["vix"]=result["vix"];df["spx"]=result["spx"]
    logger.info(f"Starting tuner: {len(df)} samples, {args.trials} trials per model\n")
    tuner=FixedDualModelTuner(df,result["vix"],n_trials=args.trials,output_dir=args.output_dir);cv_params=tuner.tune_cv_params();mag_study,dir_study=tuner.tune_models(cv_params);tuner.save_results(cv_params,mag_study,dir_study)
    logger.info("\n"+"="*80);logger.info("COMPLETE");logger.info("="*80)
if __name__=="__main__":main()
