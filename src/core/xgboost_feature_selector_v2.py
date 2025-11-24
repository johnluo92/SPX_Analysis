import json,logging
from datetime import datetime
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np,pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from config import FEATURE_SELECTION_CONFIG,TARGET_CONFIG, FEATURE_SELECTION_CV_PARAMS
from core.target_calculator import TargetCalculator
logger=logging.getLogger(__name__)
class SimplifiedFeatureSelector:
    def __init__(self,horizon:int=None,top_n:int=None,cv_folds:int=None,protected_features:List[str]=None,target_type:str='magnitude'):
        self.horizon=horizon if horizon is not None else TARGET_CONFIG["horizon_days"];self.top_n=top_n if top_n is not None else FEATURE_SELECTION_CONFIG["top_n"];self.cv_folds=cv_folds if cv_folds is not None else FEATURE_SELECTION_CONFIG["cv_folds"];self.protected_features=protected_features if protected_features is not None else FEATURE_SELECTION_CONFIG["protected_features"];self.target_type=target_type;self.selected_features=None;self.importance_scores=None;self.metadata=None;self.target_calculator=TargetCalculator()
        target_label="Direction (UP/DOWN)"if target_type=='direction'else"Log VIX change"
        logger.info(f"Initialized Feature Selector:");logger.info(f"  Horizon: {self.horizon} days");logger.info(f"  Target: {target_label}");logger.info(f"  Select Top N: {self.top_n}");logger.info(f"  CV Folds: {self.cv_folds}")
    def select_features(self,features_df:pd.DataFrame,vix_series:pd.Series)->Tuple[List[str],Dict]:
        if self.target_type=='direction':logger.info("\n[1/4] Calculating direction target...")
        else:logger.info("\n[1/4] Calculating forward log VIX change...")
        if self.target_type=='direction':target_log=self.target_calculator.calculate_log_vix_change(vix_series,dates=features_df.index);target=(target_log>0).astype(int);stats={"count":target.notna().sum(),"up":target.sum(),"down":(~target.astype(bool)).sum()};logger.info(f"  Valid targets: {stats['count']}");logger.info(f"  UP: {stats['up']} ({stats['up']/stats['count']:.1%}) | DOWN: {stats['down']} ({stats['down']/stats['count']:.1%})")
        else:target=self.target_calculator.calculate_log_vix_change(vix_series,dates=features_df.index);stats=self.target_calculator.get_target_stats(target);logger.info(f"  Valid targets: {stats['count']}");logger.info(f"  Target range: [{stats['min']:.4f}, {stats['max']:.4f}]");logger.info(f"  Target mean: {stats['mean']:.4f}");logger.info(f"  Target std: {stats['std']:.4f}")
        if len(target)==0:logger.error("❌ No valid targets calculated");return[],{}
        logger.info("\n[2/4] Aligning features with targets...")
        common_dates=features_df.index.intersection(target.index)
        if len(common_dates)<100:logger.error(f"❌ Insufficient aligned data: {len(common_dates)} samples");return[],{}
        X=features_df.loc[common_dates].copy();y=target.loc[common_dates].copy();X=X.ffill().bfill();X=X.dropna(axis=1,how="all")
        valid_mask=~(X.isna().any(axis=1)|y.isna());X=X[valid_mask];y=y[valid_mask]
        if self.target_type=='direction':logger.info(f"  Aligned dataset");logger.info(f"  Samples: {len(X)}");logger.info(f"  Features: {len(X.columns)}");logger.info(f"  UP: {y.sum()} ({y.mean():.1%}) | DOWN: {(~y.astype(bool)).sum()} ({(1-y.mean()):.1%})")
        else:logger.info(f"  Aligned dataset");logger.info(f"  Samples: {len(X)}");logger.info(f"  Features: {len(X.columns)}");logger.info(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
        logger.info(f"\n[3/4] Computing feature importance via {self.cv_folds}-fold CV...")
        importance_scores=self._compute_importance(X,y)
        logger.info("\n[4/4] Selecting features...")
        selected=self._select_top_features(importance_scores,X.columns);self.selected_features=selected;self.importance_scores=importance_scores
        self.metadata={"timestamp":datetime.now().isoformat(),"target":"direction"if self.target_type=='direction'else"log_vix_change","target_type":self.target_type,"horizon":self.horizon,"samples":len(X),"total_features":len(X.columns),"selected_features":len(selected),"top_n":self.top_n,"cv_folds":self.cv_folds,"protected_features":self.protected_features,"target_statistics":{"mean":float(y.mean()),"std":float(y.std()),"min":float(y.min()),"max":float(y.max())}if self.target_type!='direction'else{"up_pct":float(y.mean()),"down_pct":float(1-y.mean())},"top_20_features":[{"feature":f,"importance":float(importance_scores[f])}for f in selected[:20]]}
        self._print_summary(selected,importance_scores)
        return selected,self.metadata
    def _compute_importance(self,X:pd.DataFrame,y:pd.Series)->Dict[str,float]:
        tscv=TimeSeriesSplit(n_splits=self.cv_folds, gap=5);importance_accumulator=np.zeros(len(X.columns))
        for fold_idx,(train_idx,val_idx)in enumerate(tscv.split(X)):
            X_train=X.iloc[train_idx];y_train=y.iloc[train_idx];X_val=X.iloc[val_idx];y_val=y.iloc[val_idx]
            if self.target_type=='direction':
                model=xgb.XGBClassifier(objective="binary:logistic",eval_metric="logloss",random_state=42+fold_idx,n_jobs=-1,**FEATURE_SELECTION_CV_PARAMS)
                model.fit(X_train,y_train,verbose=False);train_pred=model.predict(X_train);val_pred=model.predict(X_val);train_acc=(train_pred==y_train).mean();val_acc=(val_pred==y_val).mean();logger.info(f"  Fold {fold_idx+1}/{self.cv_folds}: Train Acc={train_acc:.1%}, Val Acc={val_acc:.1%}")
            else:
                model=xgb.XGBRegressor(objective="reg:squarederror",random_state=42+fold_idx,n_jobs=-1,**FEATURE_SELECTION_CV_PARAMS)
                model.fit(X_train,y_train,verbose=False);train_pred=model.predict(X_train);val_pred=model.predict(X_val);train_mae=np.mean(np.abs(train_pred-y_train));val_mae=np.mean(np.abs(val_pred-y_val));logger.info(f"  Fold {fold_idx+1}/{self.cv_folds}: Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")
            importance=model.feature_importances_;importance_accumulator+=importance
        avg_importance=importance_accumulator/self.cv_folds;importance_dict=dict(zip(X.columns,avg_importance))
        return importance_dict
    def _select_top_features(self,importance_scores:Dict,all_features:pd.Index)->List[str]:
        sorted_features=sorted(importance_scores.items(),key=lambda x:x[1],reverse=True);selected=[]
        for pf in self.protected_features:
            if pf in importance_scores:selected.append(pf)
        for feature,score in sorted_features:
            if feature in selected:continue
            if len(selected)>=self.top_n+len(self.protected_features):break
            selected.append(feature)
        return selected
    def _print_summary(self,selected_features:List[str],importance_scores:Dict):
        logger.info("\n"+"="*80);logger.info("TOP 20 SELECTED FEATURES")
        for rank,feature in enumerate(selected_features[:20],1):
            score=importance_scores[feature];protected_flag=" [PROTECTED]"if feature in self.protected_features else""
            logger.info(f"{rank:2d}. {feature:50s} {score:.6f}{protected_flag}")
        logger.info("\n"+"="*80);logger.info("FEATURE SELECTION COMPLETE");logger.info("="*80)
    def save_results(self,output_dir:str="data_cache"):
        output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True)
        if self.selected_features is None:logger.error("❌ No features selected yet");return
        suffix="_direction"if self.target_type=='direction'else"_magnitude";features_file=output_path/f"selected_features{suffix}.json"
        with open(features_file,"w")as f:json.dump(self.selected_features,f,indent=2)
        logger.info(f"✅ Saved selected features: {features_file}")
        importance_file=output_path/f"feature_importance{suffix}.json"
        with open(importance_file,"w")as f:json.dump(self.importance_scores,f,indent=2)
        logger.info(f"✅ Saved importance scores: {importance_file}")
        metadata_file=output_path/f"selection_metadata{suffix}.json"
        with open(metadata_file,"w")as f:json.dump(self.metadata,f,indent=2)
        logger.info(f"✅ Saved metadata: {metadata_file}")
    def load_results(self,input_dir:str="data_cache"):
        input_path=Path(input_dir);suffix="_direction"if self.target_type=='direction'else"_magnitude";features_file=input_path/f"selected_features{suffix}.json"
        if features_file.exists():
            with open(features_file,"r")as f:self.selected_features=json.load(f)
            logger.info(f"✅ Loaded {len(self.selected_features)} selected features")
        importance_file=input_path/f"feature_importance{suffix}.json"
        if importance_file.exists():
            with open(importance_file,"r")as f:self.importance_scores=json.load(f)
            logger.info("✅ Loaded importance scores")
        metadata_file=input_path/f"selection_metadata{suffix}.json"
        if metadata_file.exists():
            with open(metadata_file,"r")as f:self.metadata=json.load(f)
            logger.info("✅ Loaded selection metadata")
