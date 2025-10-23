"""
SPX Prediction Model
Dual targets: Directional move + Range-bound probability
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple

from config import (
    SPX_FORWARD_WINDOWS, SPX_RANGE_THRESHOLDS,
    TEST_SPLIT, RANDOM_STATE, MODEL_PARAMS
)


class SPXModel:
    """Predict SPX directional moves and range-bound probability."""
    
    def __init__(self):
        self.directional_models = {}  # One model per forward window
        self.range_models = {}  # One model per threshold
        self.results = {}
        self.selected_features = None
        self.feature_importances = None  # Store feature importances from selection
    
    def create_targets(self, spx: pd.Series) -> Dict[str, pd.Series]:
        """
        Create prediction targets.
        
        Returns:
            Dictionary with directional and range targets
        """
        targets = {}
        
        # Directional targets: Will SPX be higher in N days?
        for window in SPX_FORWARD_WINDOWS:
            fwd_return = spx.pct_change(window).shift(-window)
            targets[f'direction_{window}d'] = (fwd_return > 0).astype(int)
        
        # Range targets: Will SPX stay within ±X% in N days?
        for window in SPX_FORWARD_WINDOWS:
            for threshold in SPX_RANGE_THRESHOLDS:
                fwd_return = spx.pct_change(window).shift(-window)
                in_range = (abs(fwd_return) <= threshold).astype(int)
                targets[f'range_{window}d_{int(threshold*100)}pct'] = in_range
        
        return targets
    
    def train_directional(self, features: pd.DataFrame, targets: Dict[str, pd.Series]):
        """Train directional prediction models."""
        print("\n📈 Training Directional Models...")
        
        for window in SPX_FORWARD_WINDOWS:
            target_name = f'direction_{window}d'
            y = targets[target_name]
            
            # Align features and target
            common_idx = features.index.intersection(y.index)
            X = features.loc[common_idx]
            y = y.loc[common_idx].dropna()
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            # Train/test split
            split_idx = int(len(X) * (1 - TEST_SPLIT))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model using centralized parameters
            model = RandomForestClassifier(**MODEL_PARAMS)
            
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            gap = train_acc - test_acc
            
            self.directional_models[f'{window}d'] = model
            self.results[f'directional_{window}d'] = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap,
                'type': 'directional'
            }
            
            status = "✅" if gap < 0.15 else "⚠️"
            print(f"  {window}d: Train {train_acc:.3f} | Test {test_acc:.3f} | Gap {gap:.3f} {status}")
    
    def train_range(self, features: pd.DataFrame, targets: Dict[str, pd.Series]):
        """Train range-bound prediction models."""
        print("\n📊 Training Range Models...")
        
        for window in SPX_FORWARD_WINDOWS:
            for threshold in SPX_RANGE_THRESHOLDS:
                target_name = f'range_{window}d_{int(threshold*100)}pct'
                y = targets[target_name]
                
                # Align
                common_idx = features.index.intersection(y.index)
                X = features.loc[common_idx]
                y = y.loc[common_idx].dropna()
                common_idx = X.index.intersection(y.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]
                
                # Train/test split
                split_idx = int(len(X) * (1 - TEST_SPLIT))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Train model using centralized parameters
                model = RandomForestClassifier(**MODEL_PARAMS)
                
                model.fit(X_train, y_train)
                
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                gap = train_acc - test_acc
                
                model_key = f'{window}d_{int(threshold*100)}pct'
                self.range_models[model_key] = model
                self.results[f'range_{model_key}'] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'gap': gap,
                    'type': 'range'
                }
                
                status = "✅" if gap < 0.15 else "⚠️"
                print(f"  {window}d ±{int(threshold*100)}%: Train {train_acc:.3f} | Test {test_acc:.3f} | Gap {gap:.3f} {status}")
    
    def train(self, features: pd.DataFrame, spx: pd.Series, use_feature_selection: bool = True):
        """Train all models."""
        # Create targets
        targets = self.create_targets(spx)
        
        # Optional: Feature selection (keep top features)
        if use_feature_selection:
            features = self._select_features(features, targets)
            self.selected_features = features.columns.tolist()
        else:
            self.selected_features = None
        
        # Train both model types
        self.train_directional(features, targets)
        self.train_range(features, targets)
        
        print("\n✅ All SPX models trained")
    
    def _select_features(self, features: pd.DataFrame, targets: Dict[str, pd.Series], 
                        top_n: int = 30) -> pd.DataFrame:
        """Select top N most important features."""
        print(f"\n🔍 Selecting top {top_n} features...")
        
        # Use 21d directional target for feature selection
        y = targets['direction_21d']
        common_idx = features.index.intersection(y.index)
        X = features.loc[common_idx]
        y = y.loc[common_idx].dropna()
        X = X.loc[y.index]
        
        # Quick model for importance
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Get top features and store importances
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance_sorted = importance.sort_values(ascending=False)
        top_features = importance_sorted.head(top_n).index.tolist()
        
        # Store feature importances as dict for easy access
        self.feature_importances = importance_sorted.head(top_n).to_dict()
        
        print(f"✅ Selected {len(top_features)} features")
        print(f"\n📊 TOP {top_n} FEATURES BY IMPORTANCE:")
        for i, (feat, imp) in enumerate(importance_sorted.head(top_n).items(), 1):
            print(f"   {i:2d}. {feat:50s} {imp:.4f}")
        
        return features[top_features]
    
    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Predict both directional and range probabilities.
        
        Returns:
            Dictionary with all predictions
        """
        # Use only selected features if feature selection was used
        if self.selected_features:
            features = features[self.selected_features]
        
        predictions = {}
        
        # Directional predictions
        for key, model in self.directional_models.items():
            proba = model.predict_proba(features)
            if proba.shape[1] == 1:
                # Only 1 class present (all same outcome)
                prob = proba[0, 0]
            else:
                # Normal case: get probability of positive class
                prob = proba[0, 1]
            predictions[f'direction_{key}'] = prob
        
        # Range predictions
        for key, model in self.range_models.items():
            proba = model.predict_proba(features)
            if proba.shape[1] == 1:
                # Only 1 class present
                prob = proba[0, 0]
            else:
                # Normal case
                prob = proba[0, 1]
            predictions[f'range_{key}'] = prob
        
        return predictions
    
    def summary(self) -> pd.DataFrame:
        """Results summary."""
        df = pd.DataFrame(self.results).T
        df = df.sort_values('gap')
        return df[['type', 'test_acc', 'gap']]
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of {feature_name: importance_score} sorted by importance
        """
        if self.feature_importances is None:
            return {}
        
        # Return top N from stored importances
        items = list(self.feature_importances.items())[:top_n]
        return dict(items)