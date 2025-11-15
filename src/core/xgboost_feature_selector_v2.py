import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
class XGBoostFeatureSelector:
    def __init__(self,horizon:int=21,min_importance:float=0.001,top_n:int=50,cv_folds:int=3):
        self.horizon=horizon;self.min_importance=min_importance;self.top_n=top_n;self.cv_folds=cv_folds;self.selected_features=None;self.importance_scores=None;self.selection_metadata=None
        logger.info(f"Initialized XGBoost Feature Selector V3:\n  Horizon: {horizon} days\n  Target: Log-transformed forward realized volatility\n  Min importance: {min_importance}\n  Top N: {top_n}")
    def _calculate_forward_realized_volatility(self,spx_returns:pd.Series,dates:pd.DatetimeIndex)->pd.Series:
        logger.info(f"Calculating forward-looking realized volatility:\n  Window: {self.horizon} trading days\n  Method: Annualized std of forward returns\n  Transform: Natural log")
        realized_vols=pd.Series(index=dates,dtype=float);spx_returns=spx_returns.sort_index();valid_count=0;insufficient_data=0
        for date in dates:
            if date not in spx_returns.index:continue
            try:date_pos=spx_returns.index.get_loc(date)
            except KeyError:continue
            end_pos=date_pos+self.horizon+1
            if end_pos>len(spx_returns):insufficient_data+=1;continue
            forward_returns=spx_returns.iloc[date_pos:end_pos]
            if forward_returns.notna().sum()<self.horizon/2:insufficient_data+=1;continue
            valid_returns=forward_returns.dropna()
            if len(valid_returns)<10:insufficient_data+=1;continue
            realized_vol=valid_returns.std()*np.sqrt(252);log_rv=np.log(realized_vol*100);realized_vols[date]=log_rv;valid_count+=1
        realized_vols=realized_vols.dropna()
        logger.info(f"Forward RV calculation complete:\n  Valid calculations: {valid_count}\n  Insufficient data: {insufficient_data}\n  Log(RV) range: [{realized_vols.min():.2f}, {realized_vols.max():.2f}]\n  Log(RV) mean: {realized_vols.mean():.2f}\n  Log(RV) std: {realized_vols.std():.2f}")
        return realized_vols
    def select_features(self,features_df:pd.DataFrame,spx_returns:pd.Series,feature_categories:Optional[Dict[str,List[str]]]=None)->Tuple[List[str],Dict]:
        logger.info("\n"+"="*80+"\nFEATURE SELECTION - LOG-RV TARGET\n"+"="*80)
        logger.info("\nStep 1: Calculating forward-looking realized volatility...")
        target=self._calculate_forward_realized_volatility(spx_returns=spx_returns,dates=features_df.index)
        if len(target)==0:logger.error("❌ No valid targets calculated");return [],{}
        logger.info("\nStep 2: Aligning features with targets...")
        common_dates=features_df.index.intersection(target.index)
        if len(common_dates)<100:logger.error(f"❌ Insufficient aligned data: {len(common_dates)} samples\n   Need at least 100 samples for reliable feature selection");return [],{}
        X=features_df.loc[common_dates].copy();y=target.loc[common_dates].copy();X=X.ffill().bfill();X=X.dropna(axis=1,how="all");valid_mask=~(X.isna().any(axis=1)|y.isna());X=X[valid_mask];y=y[valid_mask]
        logger.info(f"Aligned dataset:\n  Samples: {len(X)}\n  Features: {len(X.columns)}\n  Target range: [{y.min():.2f}, {y.max():.2f}]\n  Date range: {X.index[0].date()} to {X.index[-1].date()}")
        logger.info(f"\nStep 3: Computing feature importance via {self.cv_folds}-fold CV...")
        tscv=TimeSeriesSplit(n_splits=self.cv_folds);importance_accumulator=np.zeros(len(X.columns));fold_performances=[]
        for fold_idx,(train_idx,val_idx)in enumerate(tscv.split(X)):
            X_train=X.iloc[train_idx];y_train=y.iloc[train_idx];X_val=X.iloc[val_idx];y_val=y.iloc[val_idx]
            model=xgb.XGBRegressor(objective="reg:squarederror",n_estimators=100,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42+fold_idx,n_jobs=-1)
            model.fit(X_train,y_train,verbose=False);importance=model.feature_importances_;importance_accumulator+=importance;train_pred=model.predict(X_train);val_pred=model.predict(X_val);train_mae=np.mean(np.abs(train_pred-y_train));val_mae=np.mean(np.abs(val_pred-y_val))
            fold_performances.append({"fold":fold_idx+1,"train_mae":train_mae,"val_mae":val_mae,"train_samples":len(X_train),"val_samples":len(X_val)})
            logger.info(f"  Fold {fold_idx+1}/{self.cv_folds}: Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")
        avg_importance=importance_accumulator/self.cv_folds;self.importance_scores=dict(zip(X.columns,avg_importance))
        logger.info("\nStep 4: Selecting features...")
        sorted_features=sorted(self.importance_scores.items(),key=lambda x:x[1],reverse=True);selected=[]
        for feature,score in sorted_features:
            if len(selected)>=self.top_n:break
            if score<self.min_importance:continue
            selected.append(feature)
        self.selected_features=selected
        logger.info(f"\nSelected {len(selected)} features:\n  Top importance: {sorted_features[0][1]:.4f}\n  Min importance: {sorted_features[len(selected)-1][1]:.4f}\n  Threshold: {self.min_importance}")
        category_analysis={}
        if feature_categories:
            logger.info("\nFeature selection by category:")
            for category,cat_features in feature_categories.items():
                selected_in_cat=[f for f in selected if f in cat_features];total_in_cat=len([f for f in cat_features if f in X.columns])
                if total_in_cat>0:pct=100*len(selected_in_cat)/total_in_cat;logger.info(f"  {category}: {len(selected_in_cat)}/{total_in_cat} ({pct:.1f}%)");category_analysis[category]={"selected":len(selected_in_cat),"total":total_in_cat,"features":selected_in_cat}
        self.selection_metadata={"timestamp":datetime.now().isoformat(),"target":"log_realized_volatility","horizon":self.horizon,"samples":len(X),"total_features":len(X.columns),"selected_features":len(selected),"min_importance":self.min_importance,"top_n":self.top_n,"cv_folds":self.cv_folds,"fold_performances":fold_performances,"avg_train_mae":np.mean([f["train_mae"]for f in fold_performances]),"avg_val_mae":np.mean([f["val_mae"]for f in fold_performances]),"target_statistics":{"mean":float(y.mean()),"std":float(y.std()),"min":float(y.min()),"max":float(y.max())},"category_analysis":category_analysis,"top_20_features":[{"feature":f,"importance":float(s)}for f,s in sorted_features[:20]]}
        logger.info("\n"+"="*80+"\nTOP 20 SELECTED FEATURES\n"+"="*80)
        for rank,(feature,score)in enumerate(sorted_features[:20],1):logger.info(f"{rank:2d}. {feature:50s} {score:.6f}")
        logger.info("\n"+"="*80+"\nFEATURE SELECTION COMPLETE\n"+"="*80)
        return self.selected_features,self.selection_metadata
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
        with open(metadata_file,"w")as f:json.dump(self.selection_metadata,f,indent=2)
        logger.info(f"✅ Saved metadata: {metadata_file}")
    def load_results(self,input_dir:str="data_cache"):
        input_path=Path(input_dir);features_file=input_path/"selected_features.json"
        if features_file.exists():
            with open(features_file,"r")as f:self.selected_features=json.load(f)
            logger.info(f"✅ Loaded {len(self.selected_features)} selected features")
        importance_file=input_path/"feature_importance.json"
        if importance_file.exists():
            with open(importance_file,"r")as f:self.importance_scores=json.load(f)
            logger.info(f"✅ Loaded importance scores")
        metadata_file=input_path/"selection_metadata.json"
        if metadata_file.exists():
            with open(metadata_file,"r")as f:self.selection_metadata=json.load(f)
            logger.info(f"✅ Loaded selection metadata")
def test_feature_selector():
    print("\n"+"="*80+"\nTESTING FEATURE SELECTOR V3\n"+"="*80);np.random.seed(42);dates=pd.date_range("2020-01-01","2023-12-31",freq="D");spx_returns=pd.Series(np.random.randn(len(dates))*0.01,index=dates);n_features=100;features_df=pd.DataFrame(np.random.randn(len(dates),n_features),index=dates,columns=[f"feature_{i}"for i in range(n_features)])
    for i in range(5):future_vol=spx_returns.rolling(21).std().shift(-21);features_df[f"predictive_{i}"]=future_vol+np.random.randn(len(dates))*0.001
    selector=XGBoostFeatureSelector(horizon=21,min_importance=0.001,top_n=50);selected,metadata=selector.select_features(features_df=features_df,spx_returns=spx_returns)
    print(f"\n✅ Selected {len(selected)} features");print(f"✅ Avg validation MAE: {metadata['avg_val_mae']:.4f}");predictive_selected=[f for f in selected if "predictive"in f];print(f"✅ Predictive features selected: {len(predictive_selected)}/5");selector.save_results(output_dir="/home/claude/test_output");print(f"✅ Results saved to /home/claude/test_output");print("\n"+"="*80+"\nTEST COMPLETE\n"+"="*80)
if __name__=="__main__":test_feature_selector()
