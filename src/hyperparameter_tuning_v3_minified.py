#!/usr/bin/env python3
import json,logging,warnings
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error,accuracy_score
from xgboost import XGBRegressor
from config import TRAINING_END_DATE,TARGET_CONFIG,XGBOOST_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.target_calculator import TargetCalculator
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
warnings.filterwarnings("ignore");logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(message)s");logger=logging.getLogger(__name__)
class RegimeAwareHyperparameterTuner:
    def __init__(self,features_df,vix_series):
        self.features_df=features_df;self.vix_series=vix_series;self.target_calculator=TargetCalculator();logger.info("Calculating targets...")
        self.features_with_targets=self.target_calculator.calculate_all_targets(features_df,vix_col="vix")
        valid_mask=~self.features_with_targets["target_log_vix_change"].isna();self.features_with_targets=self.features_with_targets[valid_mask];self.feature_cache={};n_samples=len(self.features_with_targets)
        self.train_end_idx=int(n_samples*0.80);self.val_end_idx=int(n_samples*0.90)
        logger.info(f"Initialized: {n_samples} samples, {len(features_df.columns)} features");logger.info(f"Split: Train={self.train_end_idx} (80%), Val={self.val_end_idx-self.train_end_idx} (10%), Test={n_samples-self.val_end_idx} (10%)")
    def _get_selected_features(self,top_n,protected_regime_features):
        cache_key=(top_n,tuple(protected_regime_features))
        if cache_key in self.feature_cache:return self.feature_cache[cache_key]
        exclude_cols=["vix","spx","calendar_cohort","cohort_weight","feature_quality","future_vix","target_vix_pct_change","target_log_vix_change","target_direction"]
        feature_cols=[c for c in self.features_df.columns if c not in exclude_cols];selector=SimplifiedFeatureSelector(top_n=top_n,protected_features=["is_fomc_period","is_opex_week","is_earnings_heavy"]+list(protected_regime_features))
        selected_features,_=selector.select_features(self.features_df[feature_cols],self.vix_series);cohort_features=["is_fomc_period","is_opex_week","is_earnings_heavy"]
        for cf in cohort_features:
            if cf not in selected_features and cf in self.features_df.columns:selected_features.append(cf)
        self.feature_cache[cache_key]=selected_features;return selected_features
    def objective(self,trial):
        top_n=trial.suggest_int('top_n',50,150,step=25)
        regime_feature_flags={'use_regime_expected_return':trial.suggest_categorical('use_regime_expected_return',[True,False]),'use_regime_prob_stay':trial.suggest_categorical('use_regime_prob_stay',[True,False]),'use_regime_within_percentile':trial.suggest_categorical('use_regime_within_percentile',[True,False])}
        protected_regime_features=[]
        if regime_feature_flags['use_regime_expected_return']:protected_regime_features.append('regime_expected_return_5d')
        if regime_feature_flags['use_regime_prob_stay']:protected_regime_features.append('regime_prob_stay')
        if regime_feature_flags['use_regime_within_percentile']:protected_regime_features.append('regime_within_regime_percentile')
        selected_features=self._get_selected_features(top_n,protected_regime_features)
        params={'max_depth':trial.suggest_int('max_depth',4,10),'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),'n_estimators':trial.suggest_int('n_estimators',100,500,step=50),'subsample':trial.suggest_float('subsample',0.7,1.0),'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),'colsample_bylevel':trial.suggest_float('colsample_bylevel',0.6,1.0),'min_child_weight':trial.suggest_int('min_child_weight',1,15),'reg_alpha':trial.suggest_float('reg_alpha',0.0,1.0),'reg_lambda':trial.suggest_float('reg_lambda',0.3,3.0),'gamma':trial.suggest_float('gamma',0.0,0.5),'early_stopping_rounds':50,'objective':'reg:squarederror','eval_metric':'rmse','seed':42,'n_jobs':-1}
        X=self.features_with_targets[selected_features].copy();y_magnitude=self.features_with_targets["target_log_vix_change"].copy();y_direction=self.features_with_targets["target_direction"].copy()
        X_train=X.iloc[:self.train_end_idx];X_val=X.iloc[self.train_end_idx:self.val_end_idx];X_test=X.iloc[self.val_end_idx:]
        y_mag_train=y_magnitude.iloc[:self.train_end_idx];y_mag_val=y_magnitude.iloc[self.train_end_idx:self.val_end_idx];y_mag_test=y_magnitude.iloc[self.val_end_idx:]
        y_dir_train=y_direction.iloc[:self.train_end_idx];y_dir_val=y_direction.iloc[self.train_end_idx:self.val_end_idx];y_dir_test=y_direction.iloc[self.val_end_idx:]
        model=XGBRegressor(**params);model.fit(X_train,y_mag_train,eval_set=[(X_val,y_mag_val)],verbose=False);y_pred_log=model.predict(X_test)
        test_pct_actual=(np.exp(y_mag_test)-1)*100;test_pct_pred=(np.exp(y_pred_log)-1)*100;mae=mean_absolute_error(test_pct_actual,test_pct_pred)
        direction_pred=(y_pred_log>0).astype(int);direction_acc=accuracy_score(y_dir_test,direction_pred);mae_score=1.0/(1.0+mae/10.0);combined_score=0.6*mae_score+0.4*direction_acc
        trial.set_user_attr('test_mae',float(mae));trial.set_user_attr('test_direction_acc',float(direction_acc));trial.set_user_attr('n_features',len(selected_features));trial.set_user_attr('regime_features_used',protected_regime_features)
        return combined_score
    def tune(self,n_trials,output_dir="models"):
        output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True)
        logger.info("\n"+"="*80);logger.info("REGIME-AWARE HYPERPARAMETER TUNING v3");logger.info(f"Strategy: Joint optimization of feature selection + XGBoost params");logger.info(f"Trials: {n_trials}");logger.info("="*80)
        study=optuna.create_study(direction='maximize',study_name='regime_aware_tuning_v3',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=10))
        study.optimize(self.objective,n_trials=n_trials,show_progress_bar=True);best_trial=study.best_trial
        logger.info("\n"+"="*80);logger.info("BEST CONFIGURATION FOUND");logger.info("="*80)
        logger.info(f"\nPerformance:");logger.info(f"  Test MAE: {best_trial.user_attrs['test_mae']:.2f}%");logger.info(f"  Test Direction Accuracy: {best_trial.user_attrs['test_direction_acc']:.1%}");logger.info(f"  Combined Score: {best_trial.value:.4f}")
        logger.info(f"\nFeature Selection:");logger.info(f"  top_n: {best_trial.params['top_n']}");logger.info(f"  Actual features used: {best_trial.user_attrs['n_features']}");logger.info(f"  Protected regime features: {best_trial.user_attrs['regime_features_used']}")
        logger.info(f"\nXGBoost Hyperparameters:");xgb_params={k:v for k,v in best_trial.params.items() if k not in ['top_n','use_regime_expected_return','use_regime_prob_stay','use_regime_within_percentile']}
        for key,value in sorted(xgb_params.items()):
            if isinstance(value,float):logger.info(f"  {key}: {value:.4f}")
            else:logger.info(f"  {key}: {value}")
        top_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE];top_trials.sort(key=lambda t:t.value,reverse=True)
        results={"timestamp":datetime.now().isoformat(),"strategy":"regime_aware_magnitude_only","description":"Joint optimization with regime feature testing","n_trials":n_trials,"best_trial":{"trial_number":best_trial.number,"combined_score":float(best_trial.value),"test_mae_pct":float(best_trial.user_attrs['test_mae']),"test_direction_acc":float(best_trial.user_attrs['test_direction_acc']),"n_features":int(best_trial.user_attrs['n_features']),"regime_features":best_trial.user_attrs['regime_features_used'],"feature_selection":{"top_n":int(best_trial.params['top_n'])},"xgboost_params":xgb_params},"top_10_trials":[{"trial":t.number,"score":float(t.value),"mae":float(t.user_attrs['test_mae']),"dir_acc":float(t.user_attrs['test_direction_acc']),"top_n":int(t.params['top_n']),"n_features":int(t.user_attrs['n_features'])}for t in top_trials[:10]]}
        results_file=output_path/"hyperparameter_tuning_v3_results.json"
        with open(results_file,"w")as f:json.dump(results,f,indent=2)
        logger.info(f"\n✅ Results saved: {results_file}");self._print_config_updates(best_trial);return study,results
    def _print_config_updates(self,trial):
        logger.info("\n"+"="*80);logger.info("CONFIG.PY UPDATE INSTRUCTIONS");logger.info("="*80);regime_features=trial.user_attrs['regime_features_used']
        logger.info("\n1. Update FEATURE_SELECTION_CONFIG:");logger.info(f"   top_n: {trial.params['top_n']}");logger.info(f"   protected_features: ['is_fomc_period','is_opex_week','is_earnings_heavy',")
        for rf in regime_features:logger.info(f"                        '{rf}',")
        logger.info("                       ]");logger.info("\n2. Update XGBOOST_CONFIG['magnitude_params']:");xgb_params={k:v for k,v in trial.params.items() if k not in ['top_n','use_regime_expected_return','use_regime_prob_stay','use_regime_within_percentile']}
        logger.info("   magnitude_params = {");logger.info("       'objective': 'reg:squarederror',");logger.info("       'eval_metric': 'rmse',")
        for key,value in sorted(xgb_params.items()):
            if isinstance(value,float):logger.info(f"       '{key}': {value:.4f},")
            else:logger.info(f"       '{key}': {value},")
        logger.info("       'early_stopping_rounds': 50,");logger.info("       'seed': 42,");logger.info("       'n_jobs': -1");logger.info("   }")
        logger.info("\n3. Retrain: python train_probabilistic_models.py");logger.info("="*80)
def main():
    logger.info("="*80);logger.info("VIX FORECASTING - REGIME-AWARE HYPERPARAMETER TUNING v3");logger.info("="*80);logger.info("\n[1/3] Building features...")
    fetcher=UnifiedDataFetcher();engineer=FeatureEngineer(fetcher);result=engineer.build_complete_features(years=20,end_date=TRAINING_END_DATE)
    features_df=result["features"];vix_series=result["vix"];features_df["vix"]=vix_series;features_df["spx"]=result["spx"]
    logger.info(f"Built {len(features_df.columns)} features, {len(features_df)} samples");logger.info("\n[2/3] Running optimization (50 trials ~ 30 minutes)...")
    tuner=RegimeAwareHyperparameterTuner(features_df,vix_series);study,results=tuner.tune(n_trials=50,output_dir="models");logger.info("\n[3/3] COMPLETE")
    logger.info(f"\nBest MAE: {results['best_trial']['test_mae_pct']:.2f}%");logger.info(f"Best Direction Acc: {results['best_trial']['test_direction_acc']:.1%}");logger.info("\nNext: Update config.py and retrain models")
if __name__=="__main__":main()
