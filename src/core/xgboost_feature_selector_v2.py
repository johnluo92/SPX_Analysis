import json,logging
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor,XGBClassifier
from config import FEATURE_SELECTION_CV_PARAMS,TARGET_CONFIG

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class SimplifiedFeatureSelector:
    def __init__(self,target_type='expansion',top_n=100):
        self.target_type=target_type; self.top_n=top_n
        self.feature_importance={}; self.selected_features=[]
        self.metadata={}
        from core.target_calculator import TargetCalculator
        self.target_calculator=TargetCalculator()

    def select_features(self,features_df,vix,test_start_idx):
        logger.info(f"Feature selection: {self.target_type} | top_n={self.top_n}")

        # Calculate targets
        full_df=features_df.copy(); full_df['vix']=vix
        full_df=self.target_calculator.calculate_all_targets(full_df,vix_col='vix')

        # Split train+val vs test
        train_val_df=full_df.iloc[:test_start_idx]; feature_cols=[c for c in features_df.columns]

        # Domain-specific data filtering
        if self.target_type=='expansion':
            logger.info(f"  Using UP samples only (expansion domain)")
            train_val_df=train_val_df[train_val_df['target_direction']==1]
            target_col='target_log_vix_change'; is_classifier=False
        elif self.target_type=='compression':
            logger.info(f"  Using DOWN samples only (compression domain)")
            train_val_df=train_val_df[train_val_df['target_direction']==0]
            target_col='target_log_vix_change'; is_classifier=False
        elif self.target_type=='up':
            logger.info(f"  Using all samples (UP classification)")
            target_col='target_direction'; is_classifier=True
        elif self.target_type=='down':
            logger.info(f"  Using all samples (DOWN classification)")
            train_val_df['target_direction_down']=1-train_val_df['target_direction']
            target_col='target_direction_down'; is_classifier=True
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        # Drop NaN targets
        train_val_df=train_val_df.dropna(subset=[target_col])
        logger.info(f"  Training samples: {len(train_val_df)}")

        X_train_val=train_val_df[feature_cols]; y_train_val=train_val_df[target_col]

        # Train model for feature importance
        if is_classifier:
            model=XGBClassifier(objective='binary:logistic',eval_metric='aucpr',**FEATURE_SELECTION_CV_PARAMS)
        else:
            model=XGBRegressor(objective='reg:squarederror',eval_metric='rmse',**FEATURE_SELECTION_CV_PARAMS)

        # Time series CV for robust importance
        tscv=TimeSeriesSplit(n_splits=5)
        importance_scores=np.zeros(len(feature_cols))

        for fold_idx,(train_idx,val_idx) in enumerate(tscv.split(X_train_val)):
            X_fold_train=X_train_val.iloc[train_idx]; y_fold_train=y_train_val.iloc[train_idx]
            X_fold_val=X_train_val.iloc[val_idx]; y_fold_val=y_train_val.iloc[val_idx]

            model.fit(X_fold_train,y_fold_train,eval_set=[(X_fold_val,y_fold_val)],verbose=False)
            importance_scores+=model.feature_importances_

        importance_scores/=5  # Average across folds

        # Create feature importance dictionary
        self.feature_importance={feat:float(imp)for feat,imp in zip(feature_cols,importance_scores)}

        # Select top N features
        sorted_features=sorted(self.feature_importance.items(),key=lambda x:x[1],reverse=True)
        self.selected_features=[f for f,_ in sorted_features[:self.top_n]]

        logger.info(f"  Selected {len(self.selected_features)} features")
        logger.info(f"  Top 5: {self.selected_features[:5]}")

        self.metadata={"target_type":self.target_type,"top_n":self.top_n,"n_selected":len(self.selected_features),"n_samples":len(train_val_df),"top_10_features":self.selected_features[:10],"top_10_importance":[self.feature_importance[f]for f in self.selected_features[:10]]}

        return self.selected_features,self.metadata

    def save_results(self,output_dir="data_cache",suffix=""):
        output_path=Path(output_dir); output_path.mkdir(exist_ok=True)

        importance_file=output_path/f"feature_importance{suffix}.json"
        with open(importance_file,"w")as f: json.dump(self.feature_importance,f,indent=2)

        selected_file=output_path/f"selected_features{suffix}.json"
        with open(selected_file,"w")as f: json.dump(self.selected_features,f,indent=2)

        metadata_file=output_path/f"feature_selection_metadata{suffix}.json"
        with open(metadata_file,"w")as f: json.dump(self.metadata,f,indent=2)

        logger.info(f"  Saved feature selection results to {output_dir}")
