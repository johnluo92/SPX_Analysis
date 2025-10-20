"""
Sector Rotation Model v3.0 - Clean & Stable
IMPROVEMENTS:
- Removed ALL 21-day features (noise)
- Using only 63-day windows (one earnings cycle)
- Added FRED macro cycle indicators
- Fixed XLRE dividend distortion
- Reduced feature count for stability

Focus: Stability > Complexity
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
    """Feature engineering with focus on stability."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_relative_strength(self,
                                    sectors: pd.DataFrame,
                                    windows: List[int] = [63, 126]) -> pd.DataFrame:
        """
        Calculate sector relative strength vs SPY.
        
        REMOVED: 21-day windows (too noisy)
        KEPT: 63-day (3mo), 126-day (6mo)
        """
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
                                windows: List[int] = [63]) -> pd.DataFrame:
        """
        Calculate macro factor changes.
        
        CHANGE: Only 63-day windows (one earnings cycle)
        REMOVED: 21-day windows (noise), acceleration features (overfitting)
        """
        print("🔧 Calculating macro changes...")
        
        features = pd.DataFrame(index=macro.index)
        
        for col in macro.columns:
            for window in windows:
                # Velocity only (no acceleration)
                change = macro[col].pct_change(window) * 100
                features[f'{col}_change_{window}d'] = change
        
        # Yield curve slope
        if '10Y Treasury' in macro.columns and '2Y Treasury' in macro.columns:
            features['Yield_Curve_Slope'] = (
                macro['10Y Treasury'] - macro['2Y Treasury']
            )
            
            # Slope change (63-day only)
            features[f'Yield_Curve_Slope_change_63d'] = (
                features['Yield_Curve_Slope'].diff(63)
            )
        
        # Keep only the proven interaction features
        if 'Gold' in macro.columns and 'Dollar' in macro.columns:
            features['Gold_Dollar_Ratio'] = macro['Gold'] / macro['Dollar']
        
        # Remove Oil_Dollar_Interaction (weak signal per feature importance)
        
        print(f"   ✅ Created {len(features.columns)} macro features")
        return features
    
    def add_fred_macro_cycle(self, 
                            fred_data: Dict[str, pd.Series],
                            windows: List[int] = [63]) -> pd.DataFrame:
        """
        Add FRED macro cycle indicators.
        
        NEW: Economic cycle features for regime detection
        """
        print("🔧 Adding FRED macro cycle indicators...")
        
        if not fred_data:
            print("   ⚠️  No FRED data provided, skipping")
            return pd.DataFrame()
        
        # Use first series' index as base
        index = list(fred_data.values())[0].index
        features = pd.DataFrame(index=index)
        
        for name, series in fred_data.items():
            # Level
            features[name] = series
            
            # Change
            for window in windows:
                features[f'{name}_change_{window}d'] = series.diff(window)
        
        print(f"   ✅ Created {len(features.columns)} FRED features")
        return features
    
    def add_seasonality(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add only proven seasonality features."""
        print("🔧 Adding seasonality...")
        
        features = pd.DataFrame(index=index)
        
        # Month only (Quarter was noise)
        features['Month'] = index.month
        
        print(f"   ✅ Created {len(features.columns)} seasonality features")
        return features
    
    def add_vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """Add VIX features - keep only what works."""
        print("🔧 Adding VIX features...")
        
        features = pd.DataFrame(index=vix.index)
        
        # VIX volatility (proven top-3 feature)
        vix_change = vix.diff(5)
        features['VIX_volatility_21d'] = vix_change.rolling(21).std()
        
        # VIX change (63-day only, matches other windows)
        features['VIX_change_63d'] = vix.diff(63)
        
        # Remove: VIX level, VIX_Regime, VIX_change_5d, VIX_change_21d (all weak)
        
        print(f"   ✅ Created {len(features.columns)} VIX features")
        return features
    
    def combine_features(self,
                        sectors: pd.DataFrame,
                        macro: pd.DataFrame,
                        vix: pd.Series,
                        fred_data: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """Combine all features - clean and stable."""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING v3.0 - STABLE")
        print("="*70)
        
        rs_features = self.calculate_relative_strength(sectors)
        macro_features = self.calculate_macro_changes(macro)
        season_features = self.add_seasonality(sectors.index)
        vix_features = self.add_vix_features(vix)
        
        feature_sets = [rs_features, macro_features, season_features, vix_features]
        
        # Add FRED if available
        if fred_data:
            fred_features = self.add_fred_macro_cycle(fred_data)
            if not fred_features.empty:
                feature_sets.append(fred_features)
        
        all_features = pd.concat(feature_sets, axis=1)
        
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
    """Stable model with conservative hyperparameters."""
    
    def __init__(self, random_state: int = 42):
        """Initialize with proven conservative config."""
        self.config = {
            'n_estimators': 200,
            'max_depth': 6,
            'min_samples_split': 25,
            'min_samples_leaf': 30,
            'max_features': 'sqrt',
        }
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.selected_features = None
        self.scaler = StandardScaler()
    
    def create_targets(self,
                      sectors: pd.DataFrame,
                      forward_window: int = 21) -> pd.DataFrame:
        """Create target variables."""
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
                                   min_importance: float = 0.015) -> List[str]:
        """
        Feature selection with HIGHER threshold for stability.
        
        CHANGE: 1.5% threshold (up from 1.0%)
        """
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS (v3.0)")
        print("="*70)
        
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx]
        y_all = targets.loc[common_idx]
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        y_sample = y_all.iloc[:, 0]
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=25,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_scaled, y_sample)
        
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        print(f"\n📊 Top 15 Features:")
        for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
            print(f"   {i:2d}. {feat:40s} {imp:6.3f}")
        
        selected = importance[importance >= min_importance].index.tolist()
        removed = len(importance) - len(selected)
        
        print(f"\n🔧 Feature Selection:")
        print(f"   Threshold: {min_importance:.3f} ({min_importance*100:.1f}%)")
        print(f"   Selected: {len(selected)} features")
        print(f"   Removed: {removed} features")
        
        if removed > 0:
            print(f"\n   ⚠️  Removed {removed} weak features")
        
        return selected
    
    def train_models(self,
                    features: pd.DataFrame,
                    targets: pd.DataFrame,
                    test_split: float = 0.2,
                    use_feature_selection: bool = True) -> Dict:
        """Train models with stability focus."""
        print("\n" + "="*70)
        print("TRAINING MODELS - CONSERVATIVE & STABLE")
        print("="*70)
        
        if use_feature_selection and self.selected_features is None:
            self.selected_features = self.analyze_feature_importance(
                features, targets, min_importance=0.015
            )
        
        if self.selected_features is not None:
            features = features[self.selected_features]
            print(f"\n✅ Using {len(self.selected_features)} selected features")
        
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx]
        y_all = targets.loc[common_idx]
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        split_idx = int(len(X_scaled) * (1 - test_split))
        X_train = X_scaled.iloc[:split_idx]
        X_test = X_scaled.iloc[split_idx:]
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train)} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
        print(f"   Test: {len(X_test)} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")
        
        results = {}
        
        for target_col in y_all.columns:
            ticker = target_col.replace('_outperform', '')
            
            print(f"\n🌲 {ticker}...", end=" ")
            
            y = y_all[target_col]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            model = RandomForestClassifier(
                **self.config,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            test_probs = model.predict_proba(X_test)[:, 1]
            
            self.models[ticker] = model
            
            results[ticker] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting_gap': train_acc - test_acc,
                'test_prob_mean': test_probs.mean()
            }
            
            gap_status = "✅" if (train_acc - test_acc) < 0.30 else "⚠️"
            print(f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | Gap: {train_acc - test_acc:.3f} {gap_status}")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        
        return results
    
    def walk_forward_validate(self,
                             features: pd.DataFrame,
                             targets: pd.DataFrame,
                             n_splits: int = 5) -> Dict:
        """Walk-forward validation."""
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
        
        print(f"\n📊 Stability Results ({n_splits} folds):")
        print(f"{'Sector':<6} {'Mean':>8} {'±Std':>8} {'Status':>12}")
        print("-" * 38)
        
        for ticker, res in validation_results.items():
            mean = res['mean_accuracy']
            std = res['std_accuracy']
            status = "✅ Stable" if std < 0.10 else "⚠️ Unstable"
            print(f"{ticker:<6} {mean:>8.3f} {std:>8.3f} {status:>12}")
        
        return validation_results
    
    def predict_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict rotation probabilities."""
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


def test_clean_model():
    """Test v3.0 clean model."""
    print("\n" + "="*70)
    print("SECTOR ROTATION MODEL v3.0 - CLEAN & STABLE")
    print("="*70)
    
    from sector_data_fetcher import SectorDataFetcher
    from UnifiedDataFetcher import UnifiedDataFetcher
    from datetime import timedelta
    
    # Fetch data
    print("\n📊 Step 1: Fetch Data")
    print("-"*70)
    
    sector_fetcher = SectorDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)
    
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
    print("\n📊 Step 2: Feature Engineering (v3.0 - Clean)")
    print("-"*70)
    
    feat_eng = SectorRotationFeatures()
    features = feat_eng.combine_features(
        sectors_aligned,
        macro_aligned,
        vix_aligned,
        fred_data=None  # Add FRED in Phase 2
    )
    
    # Create targets
    print("\n📊 Step 3: Create Targets")
    print("-"*70)
    
    model = SectorRotationModel()
    targets = model.create_targets(sectors_aligned, forward_window=21)
    
    # Train
    print("\n📊 Step 4: Train Models")
    print("-"*70)
    
    results = model.train_models(features, targets, use_feature_selection=True)
    
    # Validate
    print("\n📊 Step 5: Walk-Forward Validation")
    print("-"*70)
    
    validation = model.walk_forward_validate(features, targets, n_splits=5)
    
    # Summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results).T
    print(results_df[['test_accuracy', 'overfitting_gap', 'test_prob_mean']].round(3))
    
    # Current predictions
    print("\n" + "="*70)
    print("CURRENT ROTATION PROBABILITIES")
    print("="*70)
    
    current_features = features.iloc[[-1]]
    current_probs = model.predict_probabilities(current_features)
    
    print("\nProbability of outperforming SPY (next 21 days):")
    sorted_probs = current_probs.T.sort_values(by=current_probs.index[-1], ascending=False)
    print(sorted_probs)
    
    print("\n✅ v3.0 TEST COMPLETE")
    
    return model, features, results, validation


if __name__ == "__main__":
    model, features, results, validation = test_clean_model()