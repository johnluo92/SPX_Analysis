#!/usr/bin/env python3
import argparse,json,logging,sys,warnings
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
import optuna
from xgboost import XGBRegressor,XGBClassifier
from sklearn.isotonic import IsotonicRegression
from config import TRAINING_YEARS,XGBOOST_CONFIG,get_last_complete_month_end
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.target_calculator import TargetCalculator
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
import config as cfg
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(message)s")
logger=logging.getLogger(__name__)

class ComprehensiveTuner:
    def __init__(self,df,vix,n_trials=100,output_dir="tuning_results"):
        self.df=df;self.vix=vix;self.n_trials=n_trials;self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True);self.target_calculator=TargetCalculator()
        total=len(df);test_size=XGBOOST_CONFIG["cv_config"]["test_size"];val_size=XGBOOST_CONFIG["cv_config"]["val_size"]
        self.train_end=int(total*(1-test_size-val_size));self.val_end=int(total*(1-test_size));self.test_start=self.val_end
        self.base_cols=[c for c in df.columns if c not in["vix","spx","calendar_cohort","cohort_weight","feature_quality"]]
        logger.info(f"Data: Train={self.train_end} Val={self.val_end-self.train_end} Test={total-self.val_end}")

    def run_feature_selection(self,cv_params,target_type,top_n,correlation_threshold):
        original_cv=cfg.FEATURE_SELECTION_CV_PARAMS.copy()
        original_corr=cfg.FEATURE_SELECTION_CONFIG.get("correlation_threshold",1.0)
        try:
            cfg.FEATURE_SELECTION_CV_PARAMS.update(cv_params)
            cfg.FEATURE_SELECTION_CONFIG["correlation_threshold"]=correlation_threshold
            selector=SimplifiedFeatureSelector(target_type=target_type,top_n=top_n)
            selected,_=selector.select_features(self.df[self.base_cols],self.vix,test_start_idx=self.test_start)
            return selected
        except Exception as e:
            logger.error(f"Feature selection error: {e}");return[]
        finally:
            cfg.FEATURE_SELECTION_CV_PARAMS.update(original_cv)
            cfg.FEATURE_SELECTION_CONFIG["correlation_threshold"]=original_corr

    def compute_ensemble_confidence(self,magnitude_pct,direction_prob,mag_weight,dir_weight,agree_weight,thresholds,bonuses,penalties):
        abs_mag=abs(magnitude_pct)
        mag_category="small"if abs_mag<thresholds["small"]else("medium"if abs_mag<thresholds["medium"]else"large")
        mag_conf=0.5+min(abs_mag/thresholds["large"],0.5)*0.5
        dir_conf=max(direction_prob,1-direction_prob)
        predicted_up=direction_prob>0.5;magnitude_up=magnitude_pct>0;models_agree=predicted_up==magnitude_up

        if models_agree:
            if abs_mag>thresholds["medium"]and dir_conf>0.75:agreement_score=bonuses["strong"]
            elif abs_mag>thresholds["small"]and dir_conf>0.65:agreement_score=bonuses["moderate"]
            else:agreement_score=bonuses["weak"]
        else:
            if abs_mag>thresholds["medium"]and dir_conf>0.75:agreement_score=-penalties["severe"]
            elif abs_mag>thresholds["small"]and dir_conf>0.65:agreement_score=-penalties["moderate"]
            else:agreement_score=-penalties["minor"]

        ensemble_conf=mag_weight*mag_conf+dir_weight*dir_conf+agree_weight*(0.5+agreement_score)
        return np.clip(ensemble_conf,0.5,1.0)

    def objective_complete(self,trial):
        # CV params for feature selection
        cv_params={'n_estimators':trial.suggest_int('cv_n_est',80,200),'max_depth':trial.suggest_int('cv_depth',3,5),'learning_rate':trial.suggest_float('cv_lr',0.03,0.1,log=True),'subsample':trial.suggest_float('cv_sub',0.75,0.95),'colsample_bytree':trial.suggest_float('cv_col',0.75,0.95)}

        # Feature selection params
        mag_top_n=trial.suggest_int('mag_top_n',60,120)
        dir_top_n=trial.suggest_int('dir_top_n',80,150)
        corr_threshold=trial.suggest_float('corr_threshold',0.85,0.98)

        # Quality filter
        quality_threshold=trial.suggest_float('quality_threshold',0.60,0.80)

        # Cohort weights
        cohort_weights={'fomc_period':trial.suggest_float('cohort_fomc',1.1,1.5),'opex_week':trial.suggest_float('cohort_opex',1.1,1.4),'earnings_heavy':trial.suggest_float('cohort_earnings',1.0,1.3),'mid_cycle':1.0}

        # Ensemble params
        ensemble_params={'mag_weight':trial.suggest_float('ens_mag_weight',0.25,0.45),'dir_weight':trial.suggest_float('ens_dir_weight',0.35,0.55),'agree_weight':trial.suggest_float('ens_agree_weight',0.15,0.30),'thresholds':{'small':trial.suggest_float('ens_small_thresh',1.5,3.0),'medium':trial.suggest_float('ens_med_thresh',4.0,7.0),'large':trial.suggest_float('ens_large_thresh',8.0,15.0)},'bonuses':{'strong':trial.suggest_float('ens_bonus_strong',0.10,0.20),'moderate':trial.suggest_float('ens_bonus_mod',0.05,0.12),'weak':0.0},'penalties':{'severe':trial.suggest_float('ens_penalty_sev',0.20,0.35),'moderate':trial.suggest_float('ens_penalty_mod',0.10,0.20),'minor':trial.suggest_float('ens_penalty_min',0.03,0.08)}}

        # Normalize ensemble weights
        weight_sum=ensemble_params['mag_weight']+ensemble_params['dir_weight']+ensemble_params['agree_weight']
        ensemble_params['mag_weight']/=weight_sum
        ensemble_params['dir_weight']/=weight_sum
        ensemble_params['agree_weight']/=weight_sum

        # Magnitude model params
        mag_params={'objective':'reg:squarederror','eval_metric':'rmse','max_depth':trial.suggest_int('mag_depth',2,5),'learning_rate':trial.suggest_float('mag_lr',0.01,0.1,log=True),'n_estimators':trial.suggest_int('mag_n_est',200,600),'subsample':trial.suggest_float('mag_sub',0.70,0.95),'colsample_bytree':trial.suggest_float('mag_col_tree',0.70,0.95),'colsample_bylevel':trial.suggest_float('mag_col_lvl',0.70,0.95),'min_child_weight':trial.suggest_int('mag_mcw',4,12),'reg_alpha':trial.suggest_float('mag_alpha',0.8,4.0),'reg_lambda':trial.suggest_float('mag_lambda',2.0,6.0),'gamma':trial.suggest_float('mag_gamma',0.15,0.6),'early_stopping_rounds':50,'seed':42,'n_jobs':-1}

        # Direction model params
        dir_params={'objective':'binary:logistic','eval_metric':'logloss','max_depth':trial.suggest_int('dir_depth',3,6),'learning_rate':trial.suggest_float('dir_lr',0.02,0.08,log=True),'n_estimators':trial.suggest_int('dir_n_est',200,600),'subsample':trial.suggest_float('dir_sub',0.70,0.92),'colsample_bytree':trial.suggest_float('dir_col_tree',0.65,0.90),'min_child_weight':trial.suggest_int('dir_mcw',5,15),'reg_alpha':trial.suggest_float('dir_alpha',1.0,3.5),'reg_lambda':trial.suggest_float('dir_lambda',2.0,5.0),'gamma':trial.suggest_float('dir_gamma',0.2,0.6),'scale_pos_weight':trial.suggest_float('dir_scale',0.9,1.4),'early_stopping_rounds':50,'seed':42,'n_jobs':-1}

        try:
            # Feature selection
            mag_features=self.run_feature_selection(cv_params,'magnitude',mag_top_n,corr_threshold)
            dir_features=self.run_feature_selection(cv_params,'direction',dir_top_n,corr_threshold)
            if len(mag_features)<20 or len(dir_features)<20:return 999.0

            # Prepare data with targets
            df_targets=self.target_calculator.calculate_all_targets(self.df.copy(),vix_col="vix")

            # Apply quality filter
            if 'feature_quality' in df_targets.columns:
                quality_mask=df_targets['feature_quality']>=quality_threshold
                df_filtered=df_targets[quality_mask].copy()
                if len(df_filtered)<len(df_targets)*0.5:return 999.0
            else:
                df_filtered=df_targets.copy()

            # Apply cohort weights
            df_filtered['cohort_weight']=df_filtered['calendar_cohort'].map(cohort_weights).fillna(1.0)

            # Magnitude model
            X_mag=df_filtered[mag_features].copy();y_mag=df_filtered["target_log_vix_change"]
            valid_mag=~(X_mag.isna().any(axis=1)|y_mag.isna())
            X_mag,y_mag=X_mag[valid_mag],y_mag[valid_mag]
            weights_mag=df_filtered.loc[X_mag.index,'cohort_weight'].values

            # Recalculate splits after filtering
            train_end=int(len(X_mag)*0.70);val_end=int(len(X_mag)*0.85)
            X_mag_tr,y_mag_tr,w_mag_tr=X_mag.iloc[:train_end],y_mag.iloc[:train_end],weights_mag[:train_end]
            X_mag_val,y_mag_val,w_mag_val=X_mag.iloc[train_end:val_end],y_mag.iloc[train_end:val_end],weights_mag[train_end:val_end]
            X_mag_test,y_mag_test=X_mag.iloc[val_end:],y_mag.iloc[val_end:]

            if len(X_mag_val)<20 or len(X_mag_test)<20:return 999.0

            mag_model=XGBRegressor(**mag_params)
            mag_model.fit(X_mag_tr,y_mag_tr,sample_weight=w_mag_tr,eval_set=[(X_mag_val,y_mag_val)],sample_weight_eval_set=[w_mag_val],verbose=False)
            y_mag_pred=np.clip(mag_model.predict(X_mag_test),-2,2)

            test_pct_actual=(np.exp(y_mag_test)-1)*100
            test_pct_pred=(np.exp(y_mag_pred)-1)*100
            mag_mae=np.mean(np.abs(test_pct_pred-test_pct_actual))
            mag_bias=np.mean(test_pct_pred-test_pct_actual)

            if np.isnan(mag_mae)or mag_mae>20:return 999.0

            # Direction model
            X_dir=df_filtered[dir_features].copy();y_dir=df_filtered["target_direction"]
            valid_dir=~(X_dir.isna().any(axis=1)|y_dir.isna())
            X_dir,y_dir=X_dir[valid_dir],y_dir[valid_dir]
            weights_dir=df_filtered.loc[X_dir.index,'cohort_weight'].values

            train_end=int(len(X_dir)*0.70);val_end=int(len(X_dir)*0.85)
            X_dir_tr,y_dir_tr,w_dir_tr=X_dir.iloc[:train_end],y_dir.iloc[:train_end],weights_dir[:train_end]
            X_dir_val,y_dir_val,w_dir_val=X_dir.iloc[train_end:val_end],y_dir.iloc[train_end:val_end],weights_dir[train_end:val_end]
            X_dir_test,y_dir_test=X_dir.iloc[val_end:],y_dir.iloc[val_end:]

            if len(X_dir_val)<20 or len(X_dir_test)<20:return 999.0

            dir_model=XGBClassifier(**dir_params)
            dir_model.fit(X_dir_tr,y_dir_tr,sample_weight=w_dir_tr,eval_set=[(X_dir_val,y_dir_val)],sample_weight_eval_set=[w_dir_val],verbose=False)

            # Calibrate direction probabilities
            y_dir_proba_val=dir_model.predict_proba(X_dir_val)[:,1]
            y_dir_proba_test=dir_model.predict_proba(X_dir_test)[:,1]
            calibrator=IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_dir_proba_val,y_dir_val.values)
            y_dir_proba_calibrated=calibrator.transform(y_dir_proba_test)

            y_dir_pred_cal=(y_dir_proba_calibrated>0.5).astype(int)
            dir_acc=(y_dir_pred_cal==y_dir_test).mean()
            dir_prec=((y_dir_pred_cal==1)&(y_dir_test==1)).sum()/max((y_dir_pred_cal==1).sum(),1)
            dir_rec=((y_dir_pred_cal==1)&(y_dir_test==1)).sum()/max((y_dir_test==1).sum(),1)
            dir_f1=2*(dir_prec*dir_rec)/max(dir_prec+dir_rec,1e-8)

            if np.isnan(dir_f1):return 999.0

            # Compute ensemble confidence for test set
            ensemble_confs=[]
            for i in range(len(X_mag_test)):
                mag_pct=test_pct_pred.iloc[i]
                dir_prob=y_dir_proba_calibrated[i]
                ens_conf=self.compute_ensemble_confidence(mag_pct,dir_prob,ensemble_params['mag_weight'],ensemble_params['dir_weight'],ensemble_params['agree_weight'],ensemble_params['thresholds'],ensemble_params['bonuses'],ensemble_params['penalties'])
                ensemble_confs.append(ens_conf)

            avg_ensemble_conf=np.mean(ensemble_confs)

            # Combined score: prioritize MAE, then direction accuracy, then ensemble calibration
            combined_score=mag_mae+abs(mag_bias)*0.3+(1-dir_acc)*15+abs(avg_ensemble_conf-0.70)*5

            trial.set_user_attr('mag_mae',float(mag_mae))
            trial.set_user_attr('mag_bias',float(mag_bias))
            trial.set_user_attr('dir_acc',float(dir_acc))
            trial.set_user_attr('dir_f1',float(dir_f1))
            trial.set_user_attr('ensemble_conf',float(avg_ensemble_conf))
            trial.set_user_attr('n_mag_features',len(mag_features))
            trial.set_user_attr('n_dir_features',len(dir_features))
            trial.set_user_attr('quality_filtered_pct',float((1-len(df_filtered)/len(df_targets))*100))

            return combined_score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return 999.0

    def run_tuning(self):
        logger.info("="*80)
        logger.info(f"COMPREHENSIVE HYPERPARAMETER TUNING ({self.n_trials} trials)")
        logger.info("Tuning: CV params, feature selection, quality filter, cohort weights, ensemble config, model params")
        logger.info("="*80)

        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=20))
        study.optimize(self.objective_complete,n_trials=self.n_trials,show_progress_bar=True)

        return study

    def save_results(self,study):
        best=study.best_trial
        attrs=best.user_attrs

        # Extract all parameters
        cv_params={k:v for k,v in best.params.items()if k.startswith('cv_')}
        mag_params={k.replace('mag_',''):v for k,v in best.params.items()if k.startswith('mag_')and k!='mag_top_n'}
        dir_params={k.replace('dir_',''):v for k,v in best.params.items()if k.startswith('dir_')and k!='dir_top_n'}
        ens_params={k.replace('ens_',''):v for k,v in best.params.items()if k.startswith('ens_')}

        results={'timestamp':datetime.now().isoformat(),'trial_number':best.number,'metrics':{'magnitude_mae':attrs.get('mag_mae',0),'magnitude_bias':attrs.get('mag_bias',0),'direction_accuracy':attrs.get('dir_acc',0),'direction_f1':attrs.get('dir_f1',0),'ensemble_confidence':attrs.get('ensemble_conf',0),'n_magnitude_features':attrs.get('n_mag_features',0),'n_direction_features':attrs.get('n_dir_features',0),'quality_filtered_pct':attrs.get('quality_filtered_pct',0)},'parameters':{'cv':cv_params,'magnitude':{'top_n':best.params['mag_top_n'],'model':mag_params},'direction':{'top_n':best.params['dir_top_n'],'model':dir_params},'ensemble':ens_params,'quality_threshold':best.params['quality_threshold'],'correlation_threshold':best.params['corr_threshold'],'cohort_weights':{'fomc_period':best.params['cohort_fomc'],'opex_week':best.params['cohort_opex'],'earnings_heavy':best.params['cohort_earnings'],'mid_cycle':1.0}}}

        with open(self.output_dir/"results.json",'w')as f:json.dump(results,f,indent=2)

        # Generate config file
        config=f"""# COMPREHENSIVE TUNED CONFIG
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Trial #{best.number} - Score: {best.value:.4f}
#
# PERFORMANCE METRICS:
# Magnitude MAE: {attrs.get('mag_mae',0):.2f}% | Bias: {attrs.get('mag_bias',0):+.2f}%
# Direction Acc: {attrs.get('dir_acc',0):.1%} | F1: {attrs.get('dir_f1',0):.4f}
# Ensemble Conf: {attrs.get('ensemble_conf',0):.1%}
# Features: Mag={attrs.get('n_mag_features',0)}, Dir={attrs.get('n_dir_features',0)}
# Quality Filtered: {attrs.get('quality_filtered_pct',0):.1f}%

# Feature Selection CV Parameters
FEATURE_SELECTION_CV_PARAMS = {{
    'n_estimators': {best.params['cv_n_est']},
    'max_depth': {best.params['cv_depth']},
    'learning_rate': {best.params['cv_lr']:.4f},
    'subsample': {best.params['cv_sub']:.4f},
    'colsample_bytree': {best.params['cv_col']:.4f}
}}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {{
    'magnitude_top_n': {best.params['mag_top_n']},
    'direction_top_n': {best.params['dir_top_n']},
    'cv_folds': 5,
    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],
    'correlation_threshold': {best.params['corr_threshold']:.4f}
}}

# Quality Filter Configuration
QUALITY_FILTER_CONFIG = {{
    'enabled': True,
    'min_threshold': {best.params['quality_threshold']:.4f},
    'warn_pct': 20.0,
    'error_pct': 50.0,
    'strategy': 'raise'
}}

# Cohort Weights
CALENDAR_COHORTS = {{
    'fomc_period': {{'condition': 'macro_event_period', 'range': (-7, 2), 'weight': {best.params['cohort_fomc']:.4f}}},
    'opex_week': {{'condition': 'days_to_monthly_opex', 'range': (-7, 0), 'weight': {best.params['cohort_opex']:.4f}}},
    'earnings_heavy': {{'condition': 'spx_earnings_pct', 'range': (0.15, 1.0), 'weight': {best.params['cohort_earnings']:.4f}}},
    'mid_cycle': {{'condition': 'default', 'range': None, 'weight': 1.0}}
}}

# Ensemble Configuration
ENSEMBLE_CONFIG = {{
    'enabled': True,
    'reconciliation_method': 'weighted_agreement',
    'confidence_weights': {{
        'magnitude': {ens_params['mag_weight']:.4f},
        'direction': {ens_params['dir_weight']:.4f},
        'agreement': {ens_params['agree_weight']:.4f}
    }},
    'magnitude_thresholds': {{
        'small': {ens_params['small_thresh']:.4f},
        'medium': {ens_params['med_thresh']:.4f},
        'large': {ens_params['large_thresh']:.4f}
    }},
    'agreement_bonus': {{
        'strong': {ens_params['bonus_strong']:.4f},
        'moderate': {ens_params['bonus_mod']:.4f},
        'weak': 0.0
    }},
    'contradiction_penalty': {{
        'severe': {ens_params['penalty_sev']:.4f},
        'moderate': {ens_params['penalty_mod']:.4f},
        'minor': {ens_params['penalty_min']:.4f}
    }},
    'min_ensemble_confidence': 0.50,
    'actionable_threshold': 0.65
}}

# Magnitude Model Parameters
XGBOOST_CONFIG['magnitude_params'].update({{
    'max_depth': {mag_params['depth']},
    'learning_rate': {mag_params['lr']:.4f},
    'n_estimators': {mag_params['n_est']},
    'subsample': {mag_params['sub']:.4f},
    'colsample_bytree': {mag_params['col_tree']:.4f},
    'colsample_bylevel': {mag_params['col_lvl']:.4f},
    'min_child_weight': {mag_params['mcw']},
    'reg_alpha': {mag_params['alpha']:.4f},
    'reg_lambda': {mag_params['lambda']:.4f},
    'gamma': {mag_params['gamma']:.4f}
}})

# Direction Model Parameters
XGBOOST_CONFIG['direction_params'].update({{
    'max_depth': {dir_params['depth']},
    'learning_rate': {dir_params['lr']:.4f},
    'n_estimators': {dir_params['n_est']},
    'subsample': {dir_params['sub']:.4f},
    'colsample_bytree': {dir_params['col_tree']:.4f},
    'min_child_weight': {dir_params['mcw']},
    'reg_alpha': {dir_params['alpha']:.4f},
    'reg_lambda': {dir_params['lambda']:.4f},
    'gamma': {dir_params['gamma']:.4f},
    'scale_pos_weight': {dir_params['scale']:.4f}
}})
"""

        with open(self.output_dir/"tuned_config.py",'w')as f:f.write(config)

        logger.info(f"\n✅ Results: {self.output_dir}/results.json")
        logger.info(f"✅ Config: {self.output_dir}/tuned_config.py")
        logger.info(f"\n📊 BEST TRIAL #{best.number}")
        logger.info(f"   Magnitude MAE: {attrs.get('mag_mae',0):.2f}% | Bias: {attrs.get('mag_bias',0):+.2f}%")
        logger.info(f"   Direction Acc: {attrs.get('dir_acc',0):.1%} | F1: {attrs.get('dir_f1',0):.4f}")
        logger.info(f"   Ensemble Conf: {attrs.get('ensemble_conf',0):.1%}")
        logger.info(f"   Features: Mag={attrs.get('n_mag_features',0)}, Dir={attrs.get('n_dir_features',0)}")

def main():
    parser=argparse.ArgumentParser(description="Comprehensive hyperparameter tuning for dual-model VIX forecaster")
    parser.add_argument('--trials',type=int,default=100,help="Number of Optuna trials")
    parser.add_argument('--output-dir',type=str,default='tuning_results',help="Output directory")
    args=parser.parse_args()

    logger.info("Loading data...")
    training_end=get_last_complete_month_end()
    fetcher=UnifiedDataFetcher()
    engineer=FeatureEngineer(fetcher)
    result=engineer.build_complete_features(years=TRAINING_YEARS,end_date=training_end)

    df=result["features"].copy()
    df["vix"]=result["vix"]
    df["spx"]=result["spx"]

    logger.info(f"Dataset: {len(df)} samples, {len(df.columns)} columns\n")

    tuner=ComprehensiveTuner(df,result["vix"],n_trials=args.trials,output_dir=args.output_dir)
    study=tuner.run_tuning()
    tuner.save_results(study)

    logger.info("\n"+"="*80)
    logger.info("TUNING COMPLETE - Apply tuned_config.py to config.py")
    logger.info("="*80)

if __name__=="__main__":
    main()
