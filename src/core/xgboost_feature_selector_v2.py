import json,logging
from pathlib import Path
import numpy as np,pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor,XGBClassifier
from config import FEATURE_SELECTION_CV_PARAMS,TARGET_CONFIG,FEATURE_SELECTION_CONFIG

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class FeatureSelector:
    """Unified feature selection: importance ranking + correlation filtering"""

    def __init__(self,target_type='expansion',top_n=100,correlation_threshold=0.90,random_state=42):
        self.target_type=target_type
        self.top_n=top_n
        self.correlation_threshold=correlation_threshold
        self.random_state=random_state
        self.feature_importance={}
        self.selected_features=[]
        self.removed_features=[]
        self.correlation_matrix=None
        self.metadata={}
        from core.target_calculator import TargetCalculator
        self.target_calculator=TargetCalculator()

    def select_features(self,features_df,vix,test_start_idx):
        logger.info(f"Feature selection: {self.target_type} | top_n={self.top_n} | corr_threshold={self.correlation_threshold}")

        # Step 1: Calculate targets
        full_df=features_df.copy()
        full_df['vix']=vix
        full_df=self.target_calculator.calculate_all_targets(full_df,vix_col='vix')

        # CRITICAL: Sort by index BEFORE splitting
        full_df = full_df.sort_index()

        # Step 2: Split train+val vs test
        train_val_df=full_df.iloc[:test_start_idx]

        # CRITICAL: Sort feature columns for determinism
        feature_cols=sorted([c for c in features_df.columns])

        # Step 3: Domain-specific data filtering
        if self.target_type=='expansion':
            logger.info(f"  Using UP samples only (expansion domain)")
            train_val_df=train_val_df[train_val_df['target_direction']==1]
            target_col='target_log_vix_change'
            is_classifier=False
        elif self.target_type=='compression':
            logger.info(f"  Using DOWN samples only (compression domain)")
            train_val_df=train_val_df[train_val_df['target_direction']==0]
            target_col='target_log_vix_change'
            is_classifier=False
        elif self.target_type=='up':
            logger.info(f"  Using all samples (UP classification)")
            target_col='target_direction'
            is_classifier=True
        elif self.target_type=='down':
            logger.info(f"  Using all samples (DOWN classification)")
            train_val_df['target_direction_down']=1-train_val_df['target_direction']
            target_col='target_direction_down'
            is_classifier=True
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        # Step 4: Drop NaN targets and sort index for determinism
        train_val_df=train_val_df.dropna(subset=[target_col]).sort_index()
        logger.info(f"  Training samples: {len(train_val_df)}")

        X_train_val=train_val_df[feature_cols]
        y_train_val=train_val_df[target_col]

        # Step 5: Train model for feature importance with explicit random_state
        cv_params = FEATURE_SELECTION_CV_PARAMS.copy()
        cv_params['random_state'] = self.random_state
        cv_params['seed'] = self.random_state
        cv_params['n_jobs'] = 1  # CRITICAL: Force single-threaded for determinism

        if is_classifier:
            model=XGBClassifier(objective='binary:logistic',eval_metric='aucpr',**cv_params)
        else:
            model=XGBRegressor(objective='reg:squarederror',eval_metric='rmse',**cv_params)

        # Step 6: Time series CV for robust importance
        tscv=TimeSeriesSplit(n_splits=5)
        importance_scores=np.zeros(len(feature_cols))

        for fold_idx,(train_idx,val_idx) in enumerate(tscv.split(X_train_val)):
            X_fold_train=X_train_val.iloc[train_idx]
            y_fold_train=y_train_val.iloc[train_idx]
            X_fold_val=X_train_val.iloc[val_idx]
            y_fold_val=y_train_val.iloc[val_idx]

            model.fit(X_fold_train,y_fold_train,eval_set=[(X_fold_val,y_fold_val)],verbose=False)
            importance_scores+=model.feature_importances_

        importance_scores/=5

        # Step 7: Create importance dictionary (sorted for determinism)
        self.feature_importance={feat:float(imp) for feat,imp in sorted(zip(feature_cols,importance_scores))}

        # Step 8: Select top N*1.5 features (will reduce after correlation filtering)
        # CRITICAL: Sort by (-importance, name) for deterministic tie-breaking
        sorted_features=sorted(self.feature_importance.items(),key=lambda x:(-x[1],x[0]))
        candidate_features=[f for f,_ in sorted_features[:int(self.top_n*1.5)]]

        logger.info(f"  Candidate features: {len(candidate_features)}")

        # Step 9: Correlation filtering - process in deterministic order
        candidate_df=features_df[sorted(candidate_features)]
        self.correlation_matrix=candidate_df.corr().abs()

        to_remove=set()
        # Process correlation pairs in deterministic order
        corr_items = []
        for i in range(len(self.correlation_matrix)):
            for j in range(i+1,len(self.correlation_matrix)):
                if self.correlation_matrix.iloc[i,j]>self.correlation_threshold:
                    feat_i=self.correlation_matrix.index[i]
                    feat_j=self.correlation_matrix.columns[j]
                    imp_i=self.feature_importance[feat_i]
                    imp_j=self.feature_importance[feat_j]
                    corr_items.append((self.correlation_matrix.iloc[i,j], feat_i, feat_j, imp_i, imp_j))

        # CRITICAL: Sort by (correlation desc, feat_i, feat_j) for determinism
        corr_items.sort(key=lambda x: (-x[0], x[1], x[2]))

        for _, feat_i, feat_j, imp_i, imp_j in corr_items:
            # Skip if either feature already removed
            if feat_i in to_remove or feat_j in to_remove:
                continue
            # Remove lower importance feature (break ties by name for determinism)
            if imp_i < imp_j or (imp_i == imp_j and feat_i > feat_j):
                to_remove.add(feat_i)
            else:
                to_remove.add(feat_j)

        self.removed_features=sorted(list(to_remove))
        remaining=[f for f in candidate_features if f not in to_remove]

        # Step 10: Final selection (top_n after correlation filtering)
        # CRITICAL: Sort remaining features by importance for deterministic selection
        remaining_with_importance = [(f, self.feature_importance[f]) for f in remaining]
        remaining_with_importance.sort(key=lambda x: (-x[1], x[0]))  # Sort by (-importance, name)
        self.selected_features = [f for f, _ in remaining_with_importance[:self.top_n]]

        logger.info(f"  Removed {len(self.removed_features)} correlated features")
        logger.info(f"  Final selection: {len(self.selected_features)} features")
        logger.info(f"  Top 5: {self.selected_features[:5]}")

        self.metadata={
            "target_type":self.target_type,
            "top_n":self.top_n,
            "correlation_threshold":self.correlation_threshold,
            "random_state":self.random_state,
            "n_selected":len(self.selected_features),
            "n_removed_correlation":len(self.removed_features),
            "n_samples":len(train_val_df),
            "top_10_features":self.selected_features[:10],
            "top_10_importance":[self.feature_importance[f] for f in self.selected_features[:10]]
        }

        return sorted(self.selected_features),self.metadata

    def save_results(self,output_dir="data_cache",suffix=""):
        output_path=Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save importance scores (sorted for determinism)
        importance_file=output_path/f"feature_importance{suffix}.json"
        with open(importance_file,"w") as f:
            json.dump(dict(sorted(self.feature_importance.items())),f,indent=2,sort_keys=True)

        # Save selected features (sorted)
        selected_file=output_path/f"selected_features{suffix}.json"
        with open(selected_file,"w") as f:
            json.dump(sorted(self.selected_features),f,indent=2)

        # Save metadata
        metadata_file=output_path/f"feature_selection_metadata{suffix}.json"
        with open(metadata_file,"w") as f:
            json.dump(self.metadata,f,indent=2,sort_keys=True)

        # Save correlation report
        if self.correlation_matrix is not None:
            report={
                "threshold":self.correlation_threshold,
                "random_state":self.random_state,
                "total_features_before":len(self.correlation_matrix),
                "features_removed":len(self.removed_features),
                "features_kept":len(self.selected_features),
                "removed_features":sorted(self.removed_features)
            }

            report_file=output_path/f"correlation_report{suffix}.json"
            with open(report_file,"w") as f:
                json.dump(report,f,indent=2,sort_keys=True)

        logger.info(f"  Saved results to {output_dir}")

    def generate_diagnostics(self,output_dir="diagnostics",suffix=""):
        """Generate correlation heatmap"""
        if self.correlation_matrix is None:
            return

        output_path=Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Heatmap for top 50 features (sorted for determinism)
        top_50=sorted(self.selected_features[:50]) if len(self.selected_features)>50 else sorted(self.selected_features)

        plt.figure(figsize=(16,14))
        sns.heatmap(
            self.correlation_matrix.loc[top_50,top_50],
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5
        )

        title_suffix=suffix.replace("_"," ").title() if suffix else ""
        plt.title(f"Feature Correlation Matrix{title_suffix} (Top 50)")
        plt.tight_layout()

        heatmap_file=output_path/f"correlation_heatmap{suffix}.png"
        plt.savefig(heatmap_file,dpi=150)
        plt.close()

# Backward compatibility alias
SimplifiedFeatureSelector=FeatureSelector
