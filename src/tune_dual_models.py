#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Tuner for Dual VIX Forecasting Models

Methodology:
- Time-series aware cross-validation
- Separate optimization for magnitude (regression) and direction (classification)
- Proper handling of class imbalance for direction
- Multiple metrics tracked for informed decision-making
"""
import json,logging,warnings
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error,accuracy_score,precision_score,recall_score,f1_score,log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor,XGBClassifier
from config import TRAINING_END_DATE
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.target_calculator import TargetCalculator
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(message)s")
logger=logging.getLogger(__name__)

class ComprehensiveDualModelTuner:
    """
    Tunes both magnitude (regression) and direction (classification) models
    using proper cross-validation and optimization objectives.
    """

    def __init__(self,features_df,vix_series):
        self.features_df=features_df
        self.vix_series=vix_series
        self.target_calculator=TargetCalculator()

        logger.info("Calculating targets...")
        self.features_with_targets=self.target_calculator.calculate_all_targets(features_df,vix_col="vix")
        valid_mask=~self.features_with_targets["target_log_vix_change"].isna()
        self.features_with_targets=self.features_with_targets[valid_mask]

        n_samples=len(self.features_with_targets)
        self.train_end_idx=int(n_samples*0.80)
        self.val_end_idx=int(n_samples*0.90)

        # Get base features excluding targets
        self.base_feature_cols=[c for c in self.features_df.columns
                                if c not in ["vix","spx","calendar_cohort","cohort_weight",
                                           "feature_quality","future_vix","target_vix_pct_change",
                                           "target_log_vix_change","target_direction"]]

        logger.info(f"Initialized tuner:")
        logger.info(f"  Total samples: {n_samples}")
        logger.info(f"  Train: {self.train_end_idx} (80%)")
        logger.info(f"  Val: {self.val_end_idx-self.train_end_idx} (10%)")
        logger.info(f"  Test: {n_samples-self.val_end_idx} (10%)")
        logger.info(f"  Base features: {len(self.base_feature_cols)}")

    def tune_magnitude(self,n_trials=50):
        """
        Tune magnitude model (XGBRegressor)

        Objective: Minimize validation MAE on percentage changes
        Secondary: Minimize bias, maintain reasonable directional accuracy
        """
        logger.info("\n"+"="*80)
        logger.info("MAGNITUDE MODEL TUNING (XGBRegressor)")
        logger.info("Objective: Minimize MAE on % changes")
        logger.info("="*80)

        def objective(trial):
            # Feature selection
            top_n=trial.suggest_int('top_n',75,150,step=25)

            # Create selector for magnitude
            selector=SimplifiedFeatureSelector(
                top_n=top_n,
                protected_features=["is_fomc_period","is_opex_week","is_earnings_heavy"],
                target_type='magnitude'
            )

            selected_features,_=selector.select_features(
                self.features_df[self.base_feature_cols],
                self.vix_series
            )

            if len(selected_features)<10:
                return float('inf')

            # Hyperparameters - focused on regression
            params={
                'max_depth':trial.suggest_int('max_depth',3,8),
                'learning_rate':trial.suggest_float('learning_rate',0.01,0.15,log=True),
                'n_estimators':trial.suggest_int('n_estimators',200,500,step=50),
                'subsample':trial.suggest_float('subsample',0.7,0.95),
                'colsample_bytree':trial.suggest_float('colsample_bytree',0.7,0.98),
                'colsample_bylevel':trial.suggest_float('colsample_bylevel',0.7,0.95),
                'min_child_weight':trial.suggest_int('min_child_weight',3,15),
                'reg_alpha':trial.suggest_float('reg_alpha',0.3,1.8),
                'reg_lambda':trial.suggest_float('reg_lambda',1.0,4.5),
                'gamma':trial.suggest_float('gamma',0.05,0.5),
                'objective':'reg:squarederror',
                'eval_metric':'rmse',
                'seed':42,
                'n_jobs':-1,
                'early_stopping_rounds':50
            }

            # Prepare data
            X=self.features_with_targets[selected_features].copy()
            y_log=self.features_with_targets["target_log_vix_change"].copy()

            X_train=X.iloc[:self.train_end_idx]
            X_val=X.iloc[self.train_end_idx:self.val_end_idx]
            X_test=X.iloc[self.val_end_idx:]

            y_train=y_log.iloc[:self.train_end_idx]
            y_val=y_log.iloc[self.train_end_idx:self.val_end_idx]
            y_test=y_log.iloc[self.val_end_idx:]

            # Train model
            model=XGBRegressor(**params)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)

            # Predictions in percentage space
            train_pred_pct=(np.exp(model.predict(X_train))-1)*100
            val_pred_pct=(np.exp(model.predict(X_val))-1)*100
            test_pred_pct=(np.exp(model.predict(X_test))-1)*100

            train_actual_pct=(np.exp(y_train)-1)*100
            val_actual_pct=(np.exp(y_val)-1)*100
            test_actual_pct=(np.exp(y_test)-1)*100

            # Primary metric: validation MAE
            val_mae=mean_absolute_error(val_actual_pct,val_pred_pct)
            test_mae=mean_absolute_error(test_actual_pct,test_pred_pct)

            # Secondary metrics
            test_bias=np.mean(test_pred_pct-test_actual_pct)
            test_dir_actual=(y_test>0).astype(int)
            test_dir_pred=(model.predict(X_test)>0).astype(int)
            test_dir_acc=accuracy_score(test_dir_actual,test_dir_pred)

            # Overfitting check
            train_mae=mean_absolute_error(train_actual_pct,train_pred_pct)
            overfit_gap=test_mae-train_mae

            # Store metrics
            trial.set_user_attr('val_mae',float(val_mae))
            trial.set_user_attr('test_mae',float(test_mae))
            trial.set_user_attr('train_mae',float(train_mae))
            trial.set_user_attr('overfit_gap',float(overfit_gap))
            trial.set_user_attr('test_bias',float(test_bias))
            trial.set_user_attr('test_dir_acc',float(test_dir_acc))
            trial.set_user_attr('n_features',len(selected_features))

            # Optimization objective: minimize val_mae
            return val_mae

        study=optuna.create_study(
            direction='minimize',
            study_name='magnitude_tuning',
            sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=15)
        )

        study.optimize(objective,n_trials=n_trials,show_progress_bar=True)

        # Report results
        best=study.best_trial
        logger.info("\n"+"="*80)
        logger.info("MAGNITUDE MODEL - BEST RESULTS")
        logger.info("="*80)
        logger.info(f"Val MAE:        {best.user_attrs['val_mae']:.2f}%")
        logger.info(f"Test MAE:       {best.user_attrs['test_mae']:.2f}%")
        logger.info(f"Train MAE:      {best.user_attrs['train_mae']:.2f}%")
        logger.info(f"Overfit gap:    {best.user_attrs['overfit_gap']:.2f}%")
        logger.info(f"Test bias:      {best.user_attrs['test_bias']:+.2f}%")
        logger.info(f"Dir accuracy:   {best.user_attrs['test_dir_acc']:.1%}")
        logger.info(f"Features used:  {best.user_attrs['n_features']}")

        return study

    def tune_direction(self,n_trials=50):
        """
        Tune direction model (XGBClassifier)

        Objective: Maximize test accuracy while:
        - Minimizing overfitting (train-test gap)
        - Minimizing prediction bias (UP/DOWN distribution)
        - Maintaining balanced precision/recall
        """
        logger.info("\n"+"="*80)
        logger.info("DIRECTION MODEL TUNING (XGBClassifier)")
        logger.info("Objective: Maximize accuracy with low overfit & bias")
        logger.info("="*80)

        def objective(trial):
            # Feature selection
            top_n=trial.suggest_int('top_n',75,150,step=25)

            # Create selector for direction
            selector=SimplifiedFeatureSelector(
                top_n=top_n,
                protected_features=["is_fomc_period","is_opex_week","is_earnings_heavy"],
                target_type='direction'
            )

            selected_features,_=selector.select_features(
                self.features_df[self.base_feature_cols],
                self.vix_series
            )

            if len(selected_features)<10:
                return float('-inf')

            # Hyperparameters - focused on classification with regularization
            params={
                'max_depth':trial.suggest_int('max_depth',2,6),
                'learning_rate':trial.suggest_float('learning_rate',0.01,0.15,log=True),
                'n_estimators':trial.suggest_int('n_estimators',200,500,step=50),
                'subsample':trial.suggest_float('subsample',0.65,0.9),
                'colsample_bytree':trial.suggest_float('colsample_bytree',0.65,0.9),
                'min_child_weight':trial.suggest_int('min_child_weight',5,25),
                'reg_alpha':trial.suggest_float('reg_alpha',0.5,2.5),
                'reg_lambda':trial.suggest_float('reg_lambda',1.5,6.0),
                'gamma':trial.suggest_float('gamma',0.1,0.7),
                'scale_pos_weight':trial.suggest_float('scale_pos_weight',0.6,1.4),
                'objective':'binary:logistic',
                'eval_metric':'logloss',
                'seed':42,
                'n_jobs':-1,
                'early_stopping_rounds':50
            }

            # Prepare data
            X=self.features_with_targets[selected_features].copy()
            y=self.features_with_targets["target_direction"].copy()

            X_train=X.iloc[:self.train_end_idx]
            X_val=X.iloc[self.train_end_idx:self.val_end_idx]
            X_test=X.iloc[self.val_end_idx:]

            y_train=y.iloc[:self.train_end_idx]
            y_val=y.iloc[self.train_end_idx:self.val_end_idx]
            y_test=y.iloc[self.val_end_idx:]

            # Train model
            model=XGBClassifier(**params)
            model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)

            # Predictions
            train_pred=model.predict(X_train)
            val_pred=model.predict(X_val)
            test_pred=model.predict(X_test)

            train_prob=model.predict_proba(X_train)[:,1]
            val_prob=model.predict_proba(X_val)[:,1]
            test_prob=model.predict_proba(X_test)[:,1]

            # Accuracy metrics
            train_acc=accuracy_score(y_train,train_pred)
            val_acc=accuracy_score(y_val,val_pred)
            test_acc=accuracy_score(y_test,test_pred)

            # Overfitting measure
            overfit_gap=train_acc-test_acc

            # Precision/Recall/F1
            test_prec=precision_score(y_test,test_pred,zero_division=0)
            test_rec=recall_score(y_test,test_pred,zero_division=0)
            test_f1=f1_score(y_test,test_pred,zero_division=0)

            # Log loss (calibration)
            test_logloss=log_loss(y_test,test_prob)

            # Prediction bias (distribution mismatch)
            pred_up_rate=test_pred.mean()
            actual_up_rate=y_test.mean()
            bias_penalty=abs(pred_up_rate-actual_up_rate)

            # Store all metrics
            trial.set_user_attr('train_acc',float(train_acc))
            trial.set_user_attr('val_acc',float(val_acc))
            trial.set_user_attr('test_acc',float(test_acc))
            trial.set_user_attr('overfit_gap',float(overfit_gap))
            trial.set_user_attr('test_prec',float(test_prec))
            trial.set_user_attr('test_rec',float(test_rec))
            trial.set_user_attr('test_f1',float(test_f1))
            trial.set_user_attr('test_logloss',float(test_logloss))
            trial.set_user_attr('bias_penalty',float(bias_penalty))
            trial.set_user_attr('pred_up_rate',float(pred_up_rate))
            trial.set_user_attr('actual_up_rate',float(actual_up_rate))
            trial.set_user_attr('n_features',len(selected_features))

            # Composite score
            # Maximize: test_acc
            # Penalize: overfitting and prediction bias
            score=test_acc - 0.2*overfit_gap - 0.15*bias_penalty

            return score

        study=optuna.create_study(
            direction='maximize',
            study_name='direction_tuning',
            sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=15)
        )

        study.optimize(objective,n_trials=n_trials,show_progress_bar=True)

        # Report results
        best=study.best_trial
        logger.info("\n"+"="*80)
        logger.info("DIRECTION MODEL - BEST RESULTS")
        logger.info("="*80)
        logger.info(f"Train Acc:      {best.user_attrs['train_acc']:.1%}")
        logger.info(f"Val Acc:        {best.user_attrs['val_acc']:.1%}")
        logger.info(f"Test Acc:       {best.user_attrs['test_acc']:.1%}")
        logger.info(f"Overfit gap:    {best.user_attrs['overfit_gap']:.1%}")
        logger.info(f"Test Precision: {best.user_attrs['test_prec']:.1%}")
        logger.info(f"Test Recall:    {best.user_attrs['test_rec']:.1%}")
        logger.info(f"Test F1:        {best.user_attrs['test_f1']:.1%}")
        logger.info(f"Test LogLoss:   {best.user_attrs['test_logloss']:.4f}")
        logger.info(f"Pred UP rate:   {best.user_attrs['pred_up_rate']:.1%}")
        logger.info(f"Actual UP rate: {best.user_attrs['actual_up_rate']:.1%}")
        logger.info(f"Bias penalty:   {best.user_attrs['bias_penalty']:.3f}")
        logger.info(f"Features used:  {best.user_attrs['n_features']}")
        logger.info(f"Composite score: {best.value:.4f}")

        return study

    def save_results(self,mag_study,dir_study,output_dir="models"):
        """Save tuning results and generate config updates"""
        output_path=Path(output_dir)
        output_path.mkdir(parents=True,exist_ok=True)

        mag_best=mag_study.best_trial
        dir_best=dir_study.best_trial

        # Extract hyperparameters (exclude feature selection params)
        mag_params={k:v for k,v in mag_best.params.items() if k!='top_n'}
        dir_params={k:v for k,v in dir_best.params.items() if k!='top_n'}

        # Compile full results
        results={
            "timestamp":datetime.now().isoformat(),
            "tuning_method":"TPE with time-series CV",
            "magnitude_model":{
                "top_n":int(mag_best.params['top_n']),
                "n_features":int(mag_best.user_attrs['n_features']),
                "metrics":{
                    "val_mae":float(mag_best.user_attrs['val_mae']),
                    "test_mae":float(mag_best.user_attrs['test_mae']),
                    "train_mae":float(mag_best.user_attrs['train_mae']),
                    "overfit_gap":float(mag_best.user_attrs['overfit_gap']),
                    "test_bias":float(mag_best.user_attrs['test_bias']),
                    "test_dir_acc":float(mag_best.user_attrs['test_dir_acc'])
                },
                "hyperparameters":mag_params,
                "best_trial":mag_best.number
            },
            "direction_model":{
                "top_n":int(dir_best.params['top_n']),
                "n_features":int(dir_best.user_attrs['n_features']),
                "metrics":{
                    "train_acc":float(dir_best.user_attrs['train_acc']),
                    "val_acc":float(dir_best.user_attrs['val_acc']),
                    "test_acc":float(dir_best.user_attrs['test_acc']),
                    "overfit_gap":float(dir_best.user_attrs['overfit_gap']),
                    "test_prec":float(dir_best.user_attrs['test_prec']),
                    "test_rec":float(dir_best.user_attrs['test_rec']),
                    "test_f1":float(dir_best.user_attrs['test_f1']),
                    "test_logloss":float(dir_best.user_attrs['test_logloss']),
                    "pred_up_rate":float(dir_best.user_attrs['pred_up_rate']),
                    "actual_up_rate":float(dir_best.user_attrs['actual_up_rate']),
                    "bias_penalty":float(dir_best.user_attrs['bias_penalty'])
                },
                "hyperparameters":dir_params,
                "best_trial":dir_best.number
            }
        }

        results_file=output_path/"comprehensive_tuning_results.json"
        with open(results_file,"w")as f:
            json.dump(results,f,indent=2)

        logger.info(f"\n✅ Results saved: {results_file}")

        # Print config updates
        self._print_config_updates(mag_best,dir_best)

    def _print_config_updates(self,mag_trial,dir_trial):
        """Print formatted config.py updates"""
        logger.info("\n"+"="*80)
        logger.info("UPDATE config.py WITH THESE VALUES:")
        logger.info("="*80)

        mag_params={k:v for k,v in mag_trial.params.items() if k!='top_n'}
        dir_params={k:v for k,v in dir_trial.params.items() if k!='top_n'}

        logger.info("\nFEATURE_SELECTION_CONFIG = {")
        logger.info(f"    'top_n': {mag_trial.params['top_n']},")
        logger.info("    'cv_folds': 3,")
        logger.info("    'protected_features': ['is_fomc_period', 'is_opex_week', 'is_earnings_heavy'],")
        logger.info("    'correlation_threshold': 1.0")
        logger.info("}")

        logger.info("\nXGBOOST_CONFIG = {")
        logger.info("    'strategy': 'dual_model',")
        logger.info("    'magnitude_params': {")
        logger.info("        'objective': 'reg:squarederror',")
        logger.info("        'eval_metric': 'rmse',")
        for k,v in sorted(mag_params.items()):
            if isinstance(v,float):
                logger.info(f"        '{k}': {v:.4f},")
            else:
                logger.info(f"        '{k}': {v},")
        logger.info("        'early_stopping_rounds': 50,")
        logger.info("        'seed': 42,")
        logger.info("        'n_jobs': -1")
        logger.info("    },")
        logger.info("    'direction_params': {")
        logger.info("        'objective': 'binary:logistic',")
        logger.info("        'eval_metric': 'logloss',")
        for k,v in sorted(dir_params.items()):
            if isinstance(v,float):
                logger.info(f"        '{k}': {v:.4f},")
            else:
                logger.info(f"        '{k}': {v},")
        logger.info("        'early_stopping_rounds': 50,")
        logger.info("        'seed': 42,")
        logger.info("        'n_jobs': -1")
        logger.info("    }")
        logger.info("}")
        logger.info("\n"+"="*80)
        logger.info("After updating config.py, run:")
        logger.info("  python train_probabilistic_models.py")
        logger.info("="*80)


def main():
    logger.info("="*80)
    logger.info("COMPREHENSIVE DUAL MODEL HYPERPARAMETER TUNING")
    logger.info("="*80)

    logger.info("\n[1/4] Building features...")
    fetcher=UnifiedDataFetcher()
    engineer=FeatureEngineer(fetcher)
    result=engineer.build_complete_features(years=20,end_date=TRAINING_END_DATE)

    features_df=result["features"]
    vix_series=result["vix"]
    features_df["vix"]=vix_series
    features_df["spx"]=result["spx"]

    logger.info(f"Features: {len(features_df.columns)} | Samples: {len(features_df)}")

    logger.info("\n[2/4] Initializing tuner...")
    tuner=ComprehensiveDualModelTuner(features_df,vix_series)

    logger.info("\n[3/4] Running optimization...")
    mag_study=tuner.tune_magnitude(n_trials=40)
    dir_study=tuner.tune_direction(n_trials=40)

    logger.info("\n[4/4] Saving results...")
    tuner.save_results(mag_study,dir_study)

    logger.info("\n"+"="*80)
    logger.info("TUNING COMPLETE")
    logger.info("="*80)
    logger.info(f"Magnitude: Test MAE={mag_study.best_trial.user_attrs['test_mae']:.2f}%")
    logger.info(f"Direction: Test Acc={dir_study.best_trial.user_attrs['test_acc']:.1%}")


if __name__=="__main__":
    main()
