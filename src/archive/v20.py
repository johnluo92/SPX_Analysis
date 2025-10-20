"""
Sector Rotation Random Forest Model v2.0
IMPROVEMENTS:
- Feature importance analysis
- Automatic feature selection (remove weak features)
- Simple hyperparameter comparison (3 configs only)
- Walk-forward validation for robustness testing

NO PREDICTIONS - ONLY PROBABILITIES
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SectorRotationFeatures:
    """Feature engineering for sector rotation analysis."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_relative_strength(self,
                                    sectors: pd.DataFrame,
                                    windows: List[int] = [21, 63, 126]) -> pd.DataFrame:
        """Calculate sector relative strength vs SPY."""
        print("🔧 Calculating relative strength...")
        
        features = pd.DataFrame(index=sectors.index)
        spy = sectors['SPY']
        
        for ticker in sectors.columns:
            if ticker == 'SPY':
                continue
            
            sector = sectors[ticker]
            
            for window in windows:
                sector_ret = sector.pct_change(window)
                spy_ret = spy.pct_change(window)
                rs = (sector_ret - spy_ret) * 100
                features[f'{ticker}_RS_{window}d'] = rs
        
        print(f"   ✅ Created {len(features.columns)} RS features")
        return features
    
    def calculate_macro_changes(self,
                                macro: pd.DataFrame,
                                windows: List[int] = [21, 63]) -> pd.DataFrame:
        """Calculate macro factor changes with acceleration."""
        print("🔧 Calculating macro changes...")
        
        features = pd.DataFrame(index=macro.index)
        
        for col in macro.columns:
            for window in windows:
                # Velocity (rate of change)
                change = macro[col].pct_change(window) * 100
                features[f'{col}_change_{window}d'] = change
                
                # Acceleration (change in rate of change)
                features[f'{col}_accel_{window}d'] = change.diff(window)
        
        # Yield curve slope (if we have both yields)
        if '10Y Treasury' in macro.columns and '2Y Treasury' in macro.columns:
            features['Yield_Curve_Slope'] = (
                macro['10Y Treasury'] - macro['2Y Treasury']
            )
            
            # Slope change
            for window in windows:
                features[f'Yield_Curve_Slope_change_{window}d'] = (
                    features['Yield_Curve_Slope'].diff(window)
                )
        
        # Interaction features (uncorrelated factors)
        if 'Gold' in macro.columns and 'Dollar' in macro.columns:
            # Gold/Dollar ratio (flight to safety + currency strength)
            features['Gold_Dollar_Ratio'] = macro['Gold'] / macro['Dollar']
            features['Gold_Dollar_Ratio_change_21d'] = features['Gold_Dollar_Ratio'].pct_change(21) * 100
        
        if 'Crude Oil' in macro.columns and 'Dollar' in macro.columns:
            # Oil-Dollar interaction (commodity-currency dynamics)
            oil_change = macro['Crude Oil'].pct_change(21)
            dollar_change = macro['Dollar'].pct_change(21)
            features['Oil_Dollar_Interaction'] = oil_change * dollar_change * 100
        
        print(f"   ✅ Created {len(features.columns)} macro features")
        return features
    
    def add_seasonality(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add seasonality features (simplified)."""
        print("🔧 Adding seasonality...")
        
        features = pd.DataFrame(index=index)
        
        # Month (1-12) - well-documented seasonal patterns
        features['Month'] = index.month
        
        # Quarter (1-4) - earnings seasons, fiscal year effects
        features['Quarter'] = index.quarter
        
        # Remove Day_of_Year - likely noise for 21-day predictions
        
        print(f"   ✅ Created {len(features.columns)} seasonality features")
        return features
    
    def add_vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """Add VIX-based features with volatility-of-volatility."""
        print("🔧 Adding VIX features...")
        
        features = pd.DataFrame(index=vix.index)
        
        # VIX level
        features['VIX'] = vix
        
        # VIX regime (categorical)
        features['VIX_Regime'] = pd.cut(
            vix,
            bins=[0, 15, 25, 35, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # VIX change
        for window in [5, 21]:
            features[f'VIX_change_{window}d'] = vix.diff(window)
        
        # NEW: VIX volatility (stability of VIX)
        vix_change = vix.diff(5)
        features['VIX_volatility_21d'] = vix_change.rolling(21).std()
        
        print(f"   ✅ Created {len(features.columns)} VIX features")
        return features
    
    def combine_features(self,
                        sectors: pd.DataFrame,
                        macro: pd.DataFrame,
                        vix: pd.Series) -> pd.DataFrame:
        """Combine all features into single DataFrame."""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
        
        rs_features = self.calculate_relative_strength(sectors)
        macro_features = self.calculate_macro_changes(macro)
        season_features = self.add_seasonality(sectors.index)
        vix_features = self.add_vix_features(vix)
        
        all_features = pd.concat([
            rs_features,
            macro_features,
            season_features,
            vix_features
        ], axis=1)
        
        initial_len = len(all_features)
        all_features = all_features.dropna()
        dropped = initial_len - len(all_features)
        
        print(f"\n📊 Feature Summary:")
        print(f"   Total features: {len(all_features.columns)}")
        print(f"   Total samples: {len(all_features)}")
        print(f"   Dropped (NaN): {dropped}")
        print(f"   Date range: {all_features.index.min().date()} to {all_features.index.max().date()}")
        
        return all_features


class SectorRotationModel:
    """Random Forest model with feature selection and validation."""
    
    # Three simple hyperparameter configs to test
    HYPERPARAMETER_CONFIGS = {
        'conservative': {
            'n_estimators': 200,
            'max_depth': 6,
            'min_samples_split': 25,
            'min_samples_leaf': 30,
            'max_features': 'sqrt',
        },
        'balanced': {
            'n_estimators': 250,
            'max_depth': 7,
            'min_samples_split': 20,
            'min_samples_leaf': 25,
            'max_features': 'sqrt',
        },
        'exploratory': {
            'n_estimators': 300,
            'max_depth': 8,
            'min_samples_split': 15,
            'min_samples_leaf': 20,
            'max_features': 'sqrt',
        }
    }
    
    def __init__(self, config_name: str = 'balanced', random_state: int = 42):
        """Initialize model with hyperparameter config."""
        self.config_name = config_name
        self.config = self.HYPERPARAMETER_CONFIGS[config_name]
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.selected_features = None
        self.scaler = StandardScaler()
    
    def create_targets(self,
                      sectors: pd.DataFrame,
                      forward_window: int = 21) -> pd.DataFrame:
        """Create target variables (forward returns)."""
        print(f"\n🎯 Creating targets ({forward_window}d forward)...")
        
        targets = pd.DataFrame(index=sectors.index)
        spy_forward = sectors['SPY'].pct_change(forward_window).shift(-forward_window)
        
        for ticker in sectors.columns:
            if ticker == 'SPY':
                continue
            
            sector_forward = sectors[ticker].pct_change(forward_window).shift(-forward_window)
            targets[f'{ticker}_outperform'] = (
                (sector_forward > spy_forward).astype(int)
            )
        
        targets = targets.dropna()
        
        print(f"   ✅ Created {len(targets.columns)} targets")
        print(f"   Samples: {len(targets)}")
        
        return targets
    
    def analyze_feature_importance(self,
                                   features: pd.DataFrame,
                                   targets: pd.DataFrame,
                                   min_importance: float = 0.01) -> List[str]:
        """
        Analyze feature importance and select top features.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            min_importance: Minimum importance threshold (default 1%)
            
        Returns:
            List of selected feature names
        """
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx]
        y_all = targets.loc[common_idx]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Train quick model on first target to get feature importance
        y_sample = y_all.iloc[:, 0]
        
        model = RandomForestClassifier(
            n_estimators=100,  # Quick run
            max_depth=7,
            min_samples_split=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_scaled, y_sample)
        
        # Get importance
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        print(f"\n📊 Top 15 Features:")
        for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
            print(f"   {i:2d}. {feat:40s} {imp:6.3f}")
        
        # Select features above threshold
        selected = importance[importance >= min_importance].index.tolist()
        removed = len(importance) - len(selected)
        
        print(f"\n🔧 Feature Selection:")
        print(f"   Threshold: {min_importance:.3f} ({min_importance*100:.1f}%)")
        print(f"   Selected: {len(selected)} features")
        print(f"   Removed: {removed} features")
        
        if removed > 0:
            weak_features = importance[importance < min_importance]
            print(f"\n   Removed features (importance < {min_importance:.3f}):")
            for feat, imp in weak_features.items():
                print(f"      {feat:40s} {imp:6.4f}")
        
        return selected
    
    def train_models(self,
                    features: pd.DataFrame,
                    targets: pd.DataFrame,
                    test_split: float = 0.2,
                    use_feature_selection: bool = True) -> Dict:
        """Train random forest models with optional feature selection."""
        print("\n" + "="*70)
        print(f"TRAINING RANDOM FOREST MODELS ({self.config_name.upper()} config)")
        print("="*70)
        
        # Feature selection
        if use_feature_selection and self.selected_features is None:
            self.selected_features = self.analyze_feature_importance(
                features, targets, min_importance=0.01
            )
        
        # Use selected features if available
        if self.selected_features is not None:
            features = features[self.selected_features]
            print(f"\n✅ Using {len(self.selected_features)} selected features")
        
        # Align
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx]
        y_all = targets.loc[common_idx]
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Train/test split
        split_idx = int(len(X_scaled) * (1 - test_split))
        X_train = X_scaled.iloc[:split_idx]
        X_test = X_scaled.iloc[split_idx:]
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train)} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
        print(f"   Test: {len(X_test)} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")
        
        print(f"\n📊 Hyperparameters:")
        for key, val in self.config.items():
            print(f"   {key}: {val}")
        
        results = {}
        
        # Train one model per sector
        for target_col in y_all.columns:
            ticker = target_col.replace('_outperform', '')
            
            print(f"\n🌲 Training model for {ticker}...")
            
            y = y_all[target_col]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Train
            model = RandomForestClassifier(
                **self.config,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            # Probabilities
            train_probs = model.predict_proba(X_train)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
            
            # Feature importance for this sector
            feature_imp = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            self.models[ticker] = model
            self.feature_importance[ticker] = feature_imp
            
            results[ticker] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_prob_mean': train_probs.mean(),
                'test_prob_mean': test_probs.mean(),
                'overfitting_gap': train_acc - test_acc,
                'top_features': feature_imp.head(5).to_dict()
            }
            
            print(f"   Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | Gap: {train_acc - test_acc:.3f}")
            print(f"   Avg Prob (test): {test_probs.mean():.3f}")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        
        return results
    
    def walk_forward_validate(self,
                             features: pd.DataFrame,
                             targets: pd.DataFrame,
                             n_splits: int = 5) -> Dict:
        """
        Walk-forward validation to test robustness.
        
        Returns:
            Dictionary with validation results per sector
        """
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION")
        print("="*70)
        
        if self.selected_features is not None:
            features = features[self.selected_features]
        
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx]
        y_all = targets.loc[common_idx]
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        validation_results = {}
        
        for target_col in y_all.columns:
            ticker = target_col.replace('_outperform', '')
            y = y_all[target_col]
            
            fold_accs = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
                X_train_fold = X_scaled.iloc[train_idx]
                X_test_fold = X_scaled.iloc[test_idx]
                y_train_fold = y.iloc[train_idx]
                y_test_fold = y.iloc[test_idx]
                
                model = RandomForestClassifier(
                    **self.config,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                model.fit(X_train_fold, y_train_fold)
                acc = model.score(X_test_fold, y_test_fold)
                fold_accs.append(acc)
            
            validation_results[ticker] = {
                'mean_accuracy': np.mean(fold_accs),
                'std_accuracy': np.std(fold_accs),
                'fold_accuracies': fold_accs
            }
        
        # Summary
        print(f"\n📊 Walk-Forward Validation Results ({n_splits} folds):")
        print(f"{'Sector':<6} {'Mean Acc':>10} {'Std Dev':>10} {'Stability':>12}")
        print("-" * 42)
        
        for ticker, results in validation_results.items():
            mean_acc = results['mean_accuracy']
            std_acc = results['std_accuracy']
            stability = "✅ Stable" if std_acc < 0.10 else "⚠️  Unstable"
            print(f"{ticker:<6} {mean_acc:>10.3f} {std_acc:>10.3f} {stability:>12}")
        
        return validation_results
    
    def predict_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict rotation probabilities for all sectors."""
        if self.selected_features is not None:
            features = features[self.selected_features]
        
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )
        
        probs = pd.DataFrame(index=features.index)
        
        for ticker, model in self.models.items():
            prob = model.predict_proba(features_scaled)[:, 1]
            probs[ticker] = prob
        
        return probs


def test_sector_rotation_model():
    """Test the full pipeline with feature selection."""
    print("\n" + "="*70)
    print("TESTING: Sector Rotation Model Pipeline v2.0")
    print("="*70)
    
    from sector_data_fetcher import SectorDataFetcher
    from UnifiedDataFetcher import UnifiedDataFetcher
    from datetime import timedelta
    
    # Fetch data
    print("\n📊 Step 1: Fetch Data")
    print("-"*70)
    
    sector_fetcher = SectorDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)  # 7 years
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    sectors = sector_fetcher.fetch_sector_etfs(start_str, end_str)
    macro = sector_fetcher.fetch_macro_factors(start_str, end_str)
    
    unified = UnifiedDataFetcher()
    vix = unified.fetch_vix(start_str, end_str)
    
    sectors_aligned, macro_aligned, vix_aligned = sector_fetcher.align_data(
        sectors, macro, vix
    )
    
    # Feature engineering
    print("\n📊 Step 2: Feature Engineering")
    print("-"*70)
    
    feat_eng = SectorRotationFeatures()
    features = feat_eng.combine_features(
        sectors_aligned,
        macro_aligned,
        vix_aligned
    )
    
    # Create targets
    print("\n📊 Step 3: Create Targets")
    print("-"*70)
    
    model = SectorRotationModel(config_name='balanced')
    targets = model.create_targets(sectors_aligned, forward_window=21)
    
    # Train with feature selection
    print("\n📊 Step 4: Train Models (with feature selection)")
    print("-"*70)
    
    results = model.train_models(features, targets, use_feature_selection=True)
    
    # Walk-forward validation
    print("\n📊 Step 5: Walk-Forward Validation")
    print("-"*70)
    
    validation_results = model.walk_forward_validate(features, targets, n_splits=5)
    
    # Display results
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results).T
    print(results_df[['train_accuracy', 'test_accuracy', 'overfitting_gap', 'test_prob_mean']])
    
    # Current predictions
    print("\n" + "="*70)
    print("CURRENT ROTATION PROBABILITIES")
    print("="*70)
    
    current_features = features.iloc[[-1]]
    current_probs = model.predict_probabilities(current_features)
    
    print("\nProbability of outperforming SPY (next 21 days):")
    print(current_probs.T.sort_values(by=current_probs.index[-1], ascending=False))
    
    print("\n✅ PIPELINE TEST COMPLETE")
    
    return model, features, results, validation_results


if __name__ == "__main__":
    model, features, results, validation = test_sector_rotation_model()