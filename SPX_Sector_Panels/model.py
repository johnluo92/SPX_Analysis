"""
Sector Rotation Model - Clean Implementation
Twin Pillars: Simplicity & Consistency
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List

from config import (
    SECTOR_CATEGORIES, HYPERPARAMETERS, FORWARD_WINDOW,
    TEST_SPLIT, WALK_FORWARD_SPLITS, RANDOM_STATE,
    FEATURE_IMPORTANCE_THRESHOLD
)


class SectorModel:
    """Train and predict sector rotation."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.selected_features = None
    
    def create_targets(self, sectors: pd.DataFrame) -> pd.DataFrame:
        """Binary targets: sector outperforms SPY."""
        targets = pd.DataFrame(index=sectors.index)
        spy_fwd = sectors['SPY'].squeeze().pct_change(FORWARD_WINDOW).shift(-FORWARD_WINDOW)
        
        for ticker in sectors.columns:
            if ticker == 'SPY':
                continue
            sector_fwd = sectors[ticker].squeeze().pct_change(FORWARD_WINDOW).shift(-FORWARD_WINDOW)
            targets[f'{ticker}_target'] = (sector_fwd > spy_fwd).astype(int)
        
        return targets.dropna()
    
    def select_features(self, features: pd.DataFrame, targets: pd.DataFrame) -> List[str]:
        """Select features by importance."""
        X = features.loc[features.index.intersection(targets.index)]
        y = targets.iloc[:, 0].loc[X.index]
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=7, min_samples_split=20,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X, y)
        
        importance = pd.Series(model.feature_importances_, index=X.columns)
        selected = importance[importance >= FEATURE_IMPORTANCE_THRESHOLD].index.tolist()
        
        print(f"Selected {len(selected)}/{len(features.columns)} features (threshold={FEATURE_IMPORTANCE_THRESHOLD})")
        return selected
    
    def train(self, features: pd.DataFrame, targets: pd.DataFrame, 
              use_feature_selection: bool = True):
        """Train models for all sectors."""
        if use_feature_selection:
            self.selected_features = self.select_features(features, targets)
            features = features[self.selected_features]
        
        X = features.loc[features.index.intersection(targets.index)]
        y_all = targets.loc[X.index]
        
        split_idx = int(len(X) * (1 - TEST_SPLIT))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        
        print(f"\nTraining: {len(X_train)} train, {len(X_test)} test")
        
        for target_col in y_all.columns:
            ticker = target_col.replace('_target', '')
            category = SECTOR_CATEGORIES.get(ticker, 'MIXED')
            config = HYPERPARAMETERS.get(category, HYPERPARAMETERS['MIXED'])
            
            y = y_all[target_col]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model = RandomForestClassifier(**config, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            gap = train_acc - test_acc
            
            self.models[ticker] = model
            self.results[ticker] = {
                'category': category,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap
            }
            
            status = "✅" if gap < 0.20 else "🟡" if gap < 0.30 else "⚠️"
            print(f"{ticker:6} {category:20} Train:{train_acc:.3f} Test:{test_acc:.3f} Gap:{gap:.3f} {status}")
    
    def validate(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict:
        """Walk-forward validation."""
        if self.selected_features:
            features = features[self.selected_features]
        
        X = features.loc[features.index.intersection(targets.index)]
        y_all = targets.loc[X.index]
        
        tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
        validation = {}
        
        print(f"\nWalk-forward validation ({WALK_FORWARD_SPLITS} folds):")
        
        for target_col in y_all.columns:
            ticker = target_col.replace('_target', '')
            category = SECTOR_CATEGORIES.get(ticker, 'MIXED')
            config = HYPERPARAMETERS.get(category, HYPERPARAMETERS['MIXED'])
            
            y = y_all[target_col]
            fold_accs = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model = RandomForestClassifier(**config, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_train, y_train)
                fold_accs.append(model.score(X_test, y_test))
            
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            
            validation[ticker] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'fold_accs': fold_accs
            }
            
            status = "✅" if std_acc < 0.10 else "⚠️"
            print(f"{ticker:6} Mean:{mean_acc:.3f} Std:{std_acc:.3f} {status}")
        
        return validation
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict rotation probabilities."""
        if self.selected_features:
            features = features[self.selected_features]
        
        probs = pd.DataFrame(index=features.index)
        for ticker, model in self.models.items():
            prob_array = model.predict_proba(features)
            # Handle case where only one class exists
            if prob_array.shape[1] == 1:
                # Only one class, use 0.5 as probability
                probs[ticker] = 0.5
            else:
                probs[ticker] = prob_array[:, 1]
        
        return probs
    
    def summary(self) -> pd.DataFrame:
        """Results summary."""
        df = pd.DataFrame(self.results).T
        df = df.sort_values('gap')
        return df[['category', 'test_acc', 'gap']]