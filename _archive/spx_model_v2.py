"""
SPX Model V2 - Dashboard Edition
Clean output, no verbosity unless requested
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE, MODEL_PARAMS, SPX_FORWARD_WINDOWS


class SPXModelV2:
    """SPX prediction models with silent training by default."""
    
    def __init__(self):
        self.directional_models = {}
        self.range_models = {}
        self.selected_features = None
        self.feature_importances = None
        self.validation_results = None
    
    def _print(self, msg: str, verbose: bool = False):
        """Conditional print."""
        if verbose:
            print(msg)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 30, verbose: bool = False):
        """Select top k features using mutual information."""
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X.fillna(0), y)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self._print(f"✅ Selected {len(selected_features)} features", verbose)
        
        return selected_features
    
    def train_with_time_series_cv(self, features: pd.DataFrame, spx: pd.Series,
                                  n_splits: int = 5, use_feature_selection: bool = True,
                                  verbose: bool = False):
        """Train with time-series cross-validation."""
        
        # Create targets
        targets = self._create_targets(spx)
        
        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        features_aligned = features.loc[common_idx]
        targets_aligned = targets.loc[common_idx]
        
        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(features_aligned), 1):
            X_train = features_aligned.iloc[train_idx]
            X_test = features_aligned.iloc[test_idx]
            y_train_dir = targets_aligned['direction_5d'].iloc[train_idx]
            y_test_dir = targets_aligned['direction_5d'].iloc[test_idx]
            
            # Feature selection on train only
            if use_feature_selection:
                selected = self.select_features(X_train, y_train_dir, k=30, verbose=False)
                X_train = X_train[selected]
                X_test = X_test[selected]
            
            # Train simple RF
            rf = RandomForestClassifier(**MODEL_PARAMS)
            rf.fit(X_train.fillna(0), y_train_dir)
            
            # Evaluate
            train_pred = rf.predict(X_train.fillna(0))
            test_pred = rf.predict(X_test.fillna(0))
            
            train_acc = balanced_accuracy_score(y_train_dir, train_pred)
            test_acc = balanced_accuracy_score(y_test_dir, test_pred)
            
            # Naive baseline (always predict majority class in train)
            naive_pred = np.ones(len(y_test_dir)) if y_train_dir.mean() > 0.5 else np.zeros(len(y_test_dir))
            naive_acc = balanced_accuracy_score(y_test_dir, naive_pred)
            
            cv_results.append({
                'fold': fold_idx,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'naive_acc': naive_acc,
                'gap': train_acc - test_acc,
                'beat_naive': test_acc > naive_acc,
                'train_up_pct': y_train_dir.mean(),
                'test_up_pct': y_test_dir.mean()
            })
        
        self.validation_results = pd.DataFrame(cv_results)
        
        if verbose:
            print("\n" + "="*70)
            print("CROSS-VALIDATION SUMMARY")
            print("="*70)
            
            df = self.validation_results
            print(f"\n📊 Performance Across {n_splits} Folds:")
            print(f"   Avg Test Acc:   {df['test_acc'].mean():.3f} ± {df['test_acc'].std():.3f}")
            print(f"   Avg Naive Acc:  {df['naive_acc'].mean():.3f}")
            print(f"   Avg Gap:        {df['gap'].mean():+.3f}")
            print(f"   Beat Naive:     {df['beat_naive'].sum()}/{n_splits} folds")
            print(f"   Best Fold:      {df['test_acc'].max():.3f}")
            print(f"   Worst Fold:     {df['test_acc'].min():.3f}")
            print(f"   Range:          {df['test_acc'].max() - df['test_acc'].min():.3f}")
            
            stability = "STABLE" if df['test_acc'].std() < 0.05 else "UNSTABLE"
            print(f"\n   Stability:      {'✅' if stability == 'STABLE' else '⚠️'} {stability}")
        
        # Train final model on full data
        self._print("\n" + "="*70, verbose)
        self._print("FINAL MODEL TRAINING (Full Dataset)", verbose)
        self._print("="*70, verbose)
        
        if use_feature_selection:
            y_full = targets_aligned['direction_5d']
            self.selected_features = self.select_features(features_aligned, y_full, k=30, verbose=False)
            features_selected = features_aligned[self.selected_features]
        else:
            features_selected = features_aligned
            self.selected_features = features_aligned.columns.tolist()
        
        self._train_final_models(features_selected, targets_aligned, verbose)
        
        return self.validation_results
    
    def train_simple(self, features: pd.DataFrame, spx: pd.Series,
                    test_split: float = 0.2, use_feature_selection: bool = True,
                    verbose: bool = False):
        """Train with simple train/test split."""
        
        # Create targets
        targets = self._create_targets(spx)
        
        # Align
        common_idx = features.index.intersection(targets.index)
        features_aligned = features.loc[common_idx]
        targets_aligned = targets.loc[common_idx]
        
        # Split
        split_idx = int(len(features_aligned) * (1 - test_split))
        X_train = features_aligned.iloc[:split_idx]
        X_test = features_aligned.iloc[split_idx:]
        y_train = targets_aligned.iloc[:split_idx]
        y_test = targets_aligned.iloc[split_idx:]
        
        # Feature selection
        if use_feature_selection:
            self.selected_features = self.select_features(X_train, y_train['direction_5d'], k=30, verbose=verbose)
            X_train = X_train[self.selected_features]
            X_test = X_test[self.selected_features]
        else:
            self.selected_features = X_train.columns.tolist()
        
        # Evaluate
        rf = RandomForestClassifier(**MODEL_PARAMS)
        rf.fit(X_train.fillna(0), y_train['direction_5d'])
        
        train_pred = rf.predict(X_train.fillna(0))
        test_pred = rf.predict(X_test.fillna(0))
        
        train_acc = balanced_accuracy_score(y_train['direction_5d'], train_pred)
        test_acc = balanced_accuracy_score(y_test['direction_5d'], test_pred)
        
        if verbose:
            print(f"\n📊 Simple Split Results:")
            print(f"   Train Acc: {train_acc:.3f}")
            print(f"   Test Acc:  {test_acc:.3f}")
            print(f"   Gap:       {train_acc - test_acc:+.3f}")
        
        # Train final on full data
        features_selected = features_aligned[self.selected_features]
        self._train_final_models(features_selected, targets_aligned, verbose)
    
    def _train_final_models(self, features: pd.DataFrame, targets: pd.DataFrame, verbose: bool = False):
        """Train final models on full dataset."""
        
        X = features.fillna(0)
        
        self._print("\n📈 Training Directional Models (Final)...", verbose)
        for window in SPX_FORWARD_WINDOWS:
            target_col = f'direction_{window}d'
            if target_col not in targets.columns:
                continue
            
            y = targets[target_col].dropna()
            X_valid = X.loc[y.index]
            
            rf = RandomForestClassifier(**MODEL_PARAMS)
            rf.fit(X_valid, y)
            
            self.directional_models[window] = rf
            self._print(f"   ✓ {window}d model trained on {len(X_valid)} samples", verbose)
        
        self._print("\n📊 Training Range Models (Final)...", verbose)
        for window in SPX_FORWARD_WINDOWS:
            for threshold in [2, 3, 5]:
                target_col = f'range_{window}d_{threshold}pct'
                if target_col not in targets.columns:
                    continue
                
                y = targets[target_col].dropna()
                X_valid = X.loc[y.index]
                
                rf = RandomForestClassifier(**MODEL_PARAMS)
                rf.fit(X_valid, y)
                
                self.range_models[f'{window}d_{threshold}pct'] = rf
                self._print(f"   ✓ {window}d ±{threshold}% model trained on {len(X_valid)} samples", verbose)
        
        # Calculate feature importances from 5d directional model
        if 5 in self.directional_models:
            importances = self.directional_models[5].feature_importances_
            feature_importance_dict = dict(zip(features.columns, importances))
            self.feature_importances = dict(sorted(feature_importance_dict.items(), 
                                                  key=lambda x: x[1], reverse=True))
    
    def _create_targets(self, spx: pd.Series):
        """Create target variables."""
        targets = pd.DataFrame(index=spx.index)
        
        # Directional targets
        for window in SPX_FORWARD_WINDOWS:
            future_return = (spx.shift(-window) - spx) / spx
            targets[f'direction_{window}d'] = (future_return > 0).astype(int)
        
        # Range-bound targets
        for window in SPX_FORWARD_WINDOWS:
            for threshold_pct in [2, 3, 5]:
                threshold = threshold_pct / 100
                future_return = (spx.shift(-window) - spx) / spx
                in_range = (future_return.abs() <= threshold).astype(int)
                targets[f'range_{window}d_{threshold_pct}pct'] = in_range
        
        return targets
    
    def predict(self, features: pd.DataFrame):
        """Generate predictions."""
        predictions = {}
        
        X = features.fillna(0)
        
        # Directional predictions
        for window, model in self.directional_models.items():
            proba = model.predict_proba(X)[0][1]
            predictions[f'direction_{window}d'] = proba
        
        # Range predictions
        for key, model in self.range_models.items():
            proba = model.predict_proba(X)[0][1]
            predictions[f'range_{key}'] = proba
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20):
        """Get top N features by importance."""
        if self.feature_importances is None:
            return {}
        
        return dict(list(self.feature_importances.items())[:top_n])