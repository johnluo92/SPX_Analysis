import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class SimplifiedFeatureSelector:
    def __init__(self,horizon:int=5,top_n:int=40,cv_folds:int=3):
        self.horizon=horizon;self.top_n=top_n;self.cv_folds=cv_folds;self.selected_features=None;self.importance_scores=None;self.metadata=None
        logger.info(f"Initialized Feature Selector:");logger.info(f"  Horizon: {horizon} days");logger.info(f"  Target: Log VIX change");logger.info(f"  Top N: {top_n}");logger.info(f"  CV Folds: {cv_folds}")
    def select_features(self,features_df:pd.DataFrame,vix_series:pd.Series)->Tuple[List[str],Dict]:
        logger.info("\n"+"="*80);logger.info("FEATURE SELECTION - LOG VIX CHANGE");logger.info("="*80);logger.info("\n[1/4] Calculating forward log VIX change...")
        target=self._calculate_target(vix_series,features_df.index)
        if len(target)==0:logger.error("❌ No valid targets calculated");return[],{}
        logger.info("\n[2/4] Aligning features with targets...")
        common_dates=features_df.index.intersection(target.index)
        if len(common_dates)<100:logger.error(f"❌ Insufficient aligned data: {len(common_dates)} samples");return[],{}
        X=features_df.loc[common_dates].copy();y=target.loc[common_dates].copy();X=X.ffill().bfill();X=X.dropna(axis=1,how="all")
        valid_mask=~(X.isna().any(axis=1)|y.isna());X=X[valid_mask];y=y[valid_mask]
        logger.info(f"  Aligned dataset:");logger.info(f"    Samples: {len(X)}");logger.info(f"    Features: {len(X.columns)}");logger.info(f"    Target range: [{y.min():.4f}, {y.max():.4f}]");logger.info(f"    Date range: {X.index[0].date()} to {X.index[-1].date()}")
        logger.info(f"\n[3/4] Computing feature importance via {self.cv_folds}-fold CV...")
        importance_scores=self._compute_importance(X,y)
        logger.info("\n[4/4] Selecting features...")
        selected=self._select_top_features(importance_scores,X.columns);self.selected_features=selected;self.importance_scores=importance_scores
        self.metadata={"timestamp":datetime.now().isoformat(),"target":"log_vix_change","horizon":self.horizon,"samples":len(X),"total_features":len(X.columns),"selected_features":len(selected),"top_n":self.top_n,"cv_folds":self.cv_folds,"target_statistics":{"mean":float(y.mean()),"std":float(y.std()),"min":float(y.min()),"max":float(y.max())},"top_20_features":[{"feature":f,"importance":float(importance_scores[f])}for f in selected[:20]]}
        self._print_summary(selected,importance_scores)
        return selected,self.metadata
    def _calculate_target(self,vix_series:pd.Series,dates:pd.DatetimeIndex)->pd.Series:
        target=pd.Series(index=dates,dtype=float);vix_series=vix_series.sort_index();valid_count=0;insufficient_data=0
        for date in dates:
            if date not in vix_series.index:continue
            try:
                date_pos=vix_series.index.get_loc(date)
                if not isinstance(date_pos,int):continue
            except KeyError:continue
            end_pos=date_pos+self.horizon
            if end_pos>=len(vix_series):insufficient_data+=1;continue
            current_vix=vix_series.iloc[date_pos];future_vix=vix_series.iloc[end_pos]
            if pd.isna(current_vix)or pd.isna(future_vix):insufficient_data+=1;continue
            if current_vix<=0 or future_vix<=0:insufficient_data+=1;continue
            log_change=np.log(future_vix/current_vix);target[date]=log_change;valid_count+=1
        target=target.dropna()
        logger.info(f"  Target calculation:");logger.info(f"    Valid: {valid_count}");logger.info(f"    Insufficient data: {insufficient_data}");logger.info(f"    Target range: [{target.min():.4f}, {target.max():.4f}]");logger.info(f"    Target mean: {target.mean():.4f}");logger.info(f"    Target std: {target.std():.4f}")
        return target
    def _compute_importance(self,X:pd.DataFrame,y:pd.Series)->Dict[str,float]:
        tscv=TimeSeriesSplit(n_splits=self.cv_folds);importance_accumulator=np.zeros(len(X.columns))
        for fold_idx,(train_idx,val_idx)in enumerate(tscv.split(X)):
            X_train=X.iloc[train_idx];y_train=y.iloc[train_idx];X_val=X.iloc[val_idx];y_val=y.iloc[val_idx]
            model=xgb.XGBRegressor(objective="reg:squarederror",n_estimators=100,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42+fold_idx,n_jobs=-1)
            model.fit(X_train,y_train,verbose=False);importance=model.feature_importances_;importance_accumulator+=importance
            train_pred=model.predict(X_train);val_pred=model.predict(X_val);train_mae=np.mean(np.abs(train_pred-y_train));val_mae=np.mean(np.abs(val_pred-y_val))
            logger.info(f"  Fold {fold_idx+1}/{self.cv_folds}: Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")
        avg_importance=importance_accumulator/self.cv_folds;importance_dict=dict(zip(X.columns,avg_importance))
        return importance_dict
    def _select_top_features(self,importance_scores:Dict,all_features:pd.Index)->List[str]:
        sorted_features=sorted(importance_scores.items(),key=lambda x:x[1],reverse=True);cohort_features=["is_fomc_period","is_opex_week","is_earnings_heavy"];selected=[]
        for feature,score in sorted_features:
            if len(selected)>=self.top_n:break
            selected.append(feature)
        for cf in cohort_features:
            if cf in all_features and cf not in selected:selected.append(cf);logger.info(f"  Added cohort feature: {cf}")
        logger.info(f"\n  Selected {len(selected)} features:");logger.info(f"    Top importance: {importance_scores[selected[0]]:.6f}");logger.info(f"    Min importance: {importance_scores[selected[self.top_n-1]if self.top_n<=len(selected)else selected[-1]]:.6f}");logger.info(f"    Cohort features included: {sum(1 for cf in cohort_features if cf in selected)}")
        return selected
    def _print_summary(self,selected_features:List[str],importance_scores:Dict):
        logger.info("\n"+"="*80);logger.info("TOP 20 SELECTED FEATURES");logger.info("="*80)
        for rank,feature in enumerate(selected_features[:20],1):
            score=importance_scores[feature];cohort_flag=" [COHORT]"if feature.startswith("is_")else""
            logger.info(f"{rank:2d}. {feature:50s} {score:.6f}{cohort_flag}")
        logger.info("\n"+"="*80);logger.info("FEATURE SELECTION COMPLETE");logger.info("="*80)
    def save_results(self,output_dir:str="data_cache"):
        output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True)
        if self.selected_features is None:logger.error("❌ No features selected yet");return
        features_file=output_path/"selected_features.json"
        with open(features_file,"w")as f:json.dump(self.selected_features,f,indent=2)
        logger.info(f"✅ Saved selected features: {features_file}")
        importance_file=output_path/"feature_importance.json"
        with open(importance_file,"w")as f:json.dump(self.importance_scores,f,indent=2)
        logger.info(f"✅ Saved importance scores: {importance_file}")
        metadata_file=output_path/"selection_metadata.json"
        with open(metadata_file,"w")as f:json.dump(self.metadata,f,indent=2)
        logger.info(f"✅ Saved metadata: {metadata_file}")
    def load_results(self,input_dir:str="data_cache"):
        input_path=Path(input_dir);features_file=input_path/"selected_features.json"
        if features_file.exists():
            with open(features_file,"r")as f:self.selected_features=json.load(f)
            logger.info(f"✅ Loaded {len(self.selected_features)} selected features")
        importance_file=input_path/"feature_importance.json"
        if importance_file.exists():
            with open(importance_file,"r")as f:self.importance_scores=json.load(f)
            logger.info("✅ Loaded importance scores")
        metadata_file=input_path/"selection_metadata.json"
        if metadata_file.exists():
            with open(metadata_file,"r")as f:self.metadata=json.load(f)
            logger.info("✅ Loaded selection metadata")
