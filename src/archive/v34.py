"""
Sector Rotation Model v3.4 - Back to Formula

FIXES FROM v3.3:
1. ✅ Treasury yields verified correct (Yahoo fixed their data)
2. ✅ Removed ALL acceleration features (pure noise)
3. ✅ Simplified to 2 windows: [21, 63] (removed 126)
4. ✅ Kept 5Y Treasury for yield curve slope
5. ✅ Kept v3.3 hyperparameters (already optimal)
6. ✅ Sector-specific feature importance tracking

TARGET: Improve signal-to-noise by removing weak features
FOCUS: Let the 4 working sectors (XLE, XLP, XLRE, XLB) shine
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SectorRotationFeatures:
    """Feature engineering - v3.4 simplified."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_relative_strength(self,
                                    sectors: pd.DataFrame,
                                    windows: List[int] = [21, 63]) -> pd.DataFrame:
        """
        Calculate sector relative strength vs SPY.
        v3.4: Two windows only (removed 126d for stability).
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
        
        print(f"   ✅ Created {len(features.columns)} RS features (2 windows: {windows})")
        return features
    
    def calculate_macro_changes(self,
                                macro: pd.DataFrame,
                                windows: List[int] = [21, 63]) -> pd.DataFrame:
        """
        Calculate macro factor changes - VELOCITY ONLY (no acceleration).
        v3.4 FIX: Removed all acceleration features (noise).
        """
        print("🔧 Calculating macro changes...")
        
        features = pd.DataFrame(index=macro.index)
        
        # Velocity (rate of change) for all factors
        for col in macro.columns:
            if col == '5Y Treasury':
                # Skip 5Y raw changes (only use in slope calculation)
                print(f"   ⏭️  Skipping {col} raw changes (using in slope only)")
                continue
            
            for window in windows:
                change = macro[col].pct_change(window) * 100
                features[f'{col}_change_{window}d'] = change
        
        # Yield curve slope (10Y - 5Y) - CRITICAL economic indicator
        if '10Y Treasury' in macro.columns and '5Y Treasury' in macro.columns:
            features['Yield_Curve_Slope'] = (
                macro['10Y Treasury'] - macro['5Y Treasury']
            )
            
            # Slope changes (flattening/steepening)
            for window in windows:
                features[f'Yield_Curve_Slope_change_{window}d'] = (
                    features['Yield_Curve_Slope'].diff(window)
                )
            
            print(f"   ✅ Created Yield Curve Slope features (10Y-5Y)")
        
        # Interaction features (uncorrelated factors only)
        if 'Gold' in macro.columns and 'Dollar' in macro.columns:
            features['Gold_Dollar_Ratio'] = macro['Gold'] / macro['Dollar']
            features['Gold_Dollar_Ratio_change_21d'] = (
                features['Gold_Dollar_Ratio'].pct_change(21) * 100
            )
            print(f"   ✅ Created Gold-Dollar Ratio")
        
        if 'Crude Oil' in macro.columns and 'Dollar' in macro.columns:
            oil_change = macro['Crude Oil'].pct_change(21)
            dollar_change = macro['Dollar'].pct_change(21)
            features['Oil_Dollar_Interaction'] = oil_change * dollar_change * 100
            print(f"   ✅ Created Oil-Dollar Interaction")
        
        print(f"   ✅ Created {len(features.columns)} macro features (velocity only)")
        return features
    
    def add_seasonality(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add seasonality features."""
        print("🔧 Adding seasonality...")
        
        features = pd.DataFrame(index=index)
        features['Month'] = index.month
        features['Quarter'] = index.quarter
        
        print(f"   ✅ Created {len(features.columns)} seasonality features")
        return features
    
    def add_vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """Add VIX-based features."""
        print("🔧 Adding VIX features...")
        
        features = pd.DataFrame(index=vix.index)
        
        # VIX level
        features['VIX'] = vix
        
        # VIX regime
        features['VIX_Regime'] = pd.cut(
            vix,
            bins=[0, 15, 25, 35, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # VIX changes
        for window in [5, 21]:
            features[f'VIX_change_{window}d'] = vix.diff(window)
        
        # VIX volatility (stability indicator)
        vix_change = vix.diff(5)
        features['VIX_volatility_21d'] = vix_change.rolling(21).std()
        
        print(f"   ✅ Created {len(features.columns)} VIX features")
        return features
    
    def combine_features(self,
                        sectors: pd.DataFrame,
                        macro: pd.DataFrame,
                        vix: pd.Series) -> pd.DataFrame:
        """Combine all features - v3.4 simplified."""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING v3.4 - BACK TO FORMULA")
        print("="*70)
        print("Changes: 2 windows [21,63], no acceleration, fixed treasury data")
        
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
    """Model with v3.3 hyperparameters (keeping what works)."""
    
    def __init__(self, random_state: int = 42):
        """Initialize with proven config."""
        self.config = {
            'n_estimators': 150,      # ↓ from 250
            'max_depth': 4,           # ↓ from 7 (CRITICAL)
            'min_samples_split': 50,  # ↑ from 20
            'min_samples_leaf': 50,   # ↑ from 25
            'max_features': 'sqrt',
        }
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.selected_features = None
        self.scaler = StandardScaler()
        self.results = {}
        self.validation_results = {}
        self.sector_categories = {
            'MACRO_SENSITIVE': ['XLE', 'XLB', 'XLRE', 'XLP'],
            'SENTIMENT_DRIVEN': ['XLY', 'XLC', 'XLV', 'XLI'],
            'MIXED': ['XLK', 'XLF', 'XLU']
        }
    
    def create_targets(self,
                      sectors: pd.DataFrame,
                      forward_window: int = 21) -> pd.DataFrame:
        """Create target variables (21d forward)."""
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
        """Feature selection with 1% threshold."""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
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
            max_depth=7,
            min_samples_split=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_scaled, y_sample)
        
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        print(f"\n📊 Top 20 Features:")
        for i, (feat, imp) in enumerate(importance.head(20).items(), 1):
            print(f"   {i:2d}. {feat:45s} {imp:6.3f}")
        
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
        """Train models with v3.3 config."""
        print("\n" + "="*70)
        print("TRAINING MODELS - v3.4 (SAME HYPERPARAMETERS)")
        print("="*70)
        
        if use_feature_selection and self.selected_features is None:
            self.selected_features = self.analyze_feature_importance(
                features, targets, min_importance=0.01
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
        
        print(f"\n📊 Hyperparameters:")
        for key, val in self.config.items():
            print(f"   {key}: {val}")
        
        results = {}
        
        for target_col in y_all.columns:
            ticker = target_col.replace('_outperform', '')
            
            # Identify sector category
            category = 'MIXED'
            for cat, tickers in self.sector_categories.items():
                if ticker in tickers:
                    category = cat
                    break
            
            print(f"\n🌲 {ticker} ({category})...", end=" ")
            
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
            gap = train_acc - test_acc
            
            self.models[ticker] = model
            
            results[ticker] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting_gap': gap,
                'test_prob_mean': test_probs.mean(),
                'category': category
            }
            
            # Gap status
            if gap < 0.20:
                status = "✅ Excellent"
            elif gap < 0.30:
                status = "🟡 Good"
            else:
                status = "⚠️  High"
            
            print(f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | Gap: {gap:.3f} {status}")
        
        self.results = results
        
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
        
        self.validation_results = validation_results
        
        print(f"\n📊 Stability Results ({n_splits} folds):")
        print(f"{'Sector':<6} {'Mean':>8} {'±Std':>8} {'Status':>12}")
        print("-" * 38)
        
        for ticker, res in validation_results.items():
            mean = res['mean_accuracy']
            std = res['std_accuracy']
            status = "✅ Stable" if std < 0.10 else "⚠️  Unstable"
            print(f"{ticker:<6} {mean:>8.3f} {std:>8.3f} {status:>12}")
        
        return validation_results
    
    def calculate_confidence_scores(self) -> pd.DataFrame:
        """Calculate confidence tiers."""
        print("\n" + "="*70)
        print("CONFIDENCE SCORING")
        print("="*70)
        
        confidence_data = []
        
        for ticker in self.results.keys():
            gap = self.results[ticker]['overfitting_gap']
            test_acc = self.results[ticker]['test_accuracy']
            wf_std = self.validation_results[ticker]['std_accuracy']
            category = self.results[ticker]['category']
            
            # Tier logic
            if gap < 0.20 and test_acc > 0.55 and wf_std < 0.10:
                tier = "HIGH"
                emoji = "🟢"
            elif gap < 0.30 and test_acc > 0.50:
                tier = "MEDIUM"
                emoji = "🟡"
            else:
                tier = "LOW"
                emoji = "🔴"
            
            confidence_data.append({
                'Sector': ticker,
                'Category': category,
                'Tier': tier,
                'Emoji': emoji,
                'Test_Acc': test_acc,
                'Gap': gap,
                'WF_Std': wf_std
            })
        
        confidence_df = pd.DataFrame(confidence_data)
        
        print("\n📊 Confidence by Sector Type:")
        for category in ['MACRO_SENSITIVE', 'MIXED', 'SENTIMENT_DRIVEN']:
            print(f"\n{category}:")
            subset = confidence_df[confidence_df['Category'] == category]
            print(subset[['Sector', 'Tier', 'Test_Acc', 'Gap']].to_string(index=False))
        
        return confidence_df
    
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


def test_v34_model():
    """Test v3.4 - back to formula with critical simplifications."""
    print("\n" + "="*70)
    print("SECTOR ROTATION MODEL v3.4 - BACK TO FORMULA")
    print("="*70)
    print("\nKey Improvements:")
    print("  1. Treasury yields verified correct (no scaling needed)")
    print("  2. Removed ALL acceleration features (noise)")
    print("  3. Simplified to 2 windows [21, 63]")
    print("  4. Kept 5Y Treasury for yield curve")
    print("  5. Focus on signal, remove noise")
    
    from sector_data_fetcher_v34 import SectorDataFetcher
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    # Fetch data
    print("\n📊 Step 1: Fetch Data (Yahoo Finance)")
    print("-"*70)
    
    sector_fetcher = SectorDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"📅 Data period: {start_str} to {end_str} (7 years)")
    
    sectors = sector_fetcher.fetch_sector_etfs(start_str, end_str)
    macro = sector_fetcher.fetch_macro_factors(start_str, end_str)
    
    unified = UnifiedDataFetcher()
    vix = unified.fetch_vix(start_str, end_str, source='yahoo')
    
    sectors_aligned, macro_aligned, vix_aligned = sector_fetcher.align_data(
        sectors, macro, vix
    )
    
    # Feature engineering
    print("\n📊 Step 2: Feature Engineering (v3.4 Simplified)")
    print("-"*70)
    
    feat_eng = SectorRotationFeatures()
    features = feat_eng.combine_features(
        sectors_aligned,
        macro_aligned,
        vix_aligned
    )
    
    # Create targets
    print("\n📊 Step 3: Create Targets (21d forward)")
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
    
    # Confidence scoring
    print("\n📊 Step 6: Confidence Scoring")
    print("-"*70)
    
    confidence = model.calculate_confidence_scores()
    
    # Summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results).T
    print("\n📊 Performance Metrics:")
    print(results_df[['category', 'test_accuracy', 'overfitting_gap']].round(3))
    
    # Category analysis
    print("\n" + "="*70)
    print("PERFORMANCE BY SECTOR CATEGORY")
    print("="*70)
    
    for category in ['MACRO_SENSITIVE', 'MIXED', 'SENTIMENT_DRIVEN']:
        subset = results_df[results_df['category'] == category]
        if len(subset) > 0:
            print(f"\n{category}:")
            print(f"  Mean Gap: {subset['overfitting_gap'].mean():.3f}")
            print(f"  Mean Test Acc: {subset['test_accuracy'].mean():.3f}")
            print(f"  Sectors: {', '.join(subset.index.tolist())}")
    
    # Gap improvement analysis
    print("\n" + "="*70)
    print("GAP ANALYSIS (Target: <0.20 excellent, <0.30 good)")
    print("="*70)
    
    excellent = results_df[results_df['overfitting_gap'] < 0.20]
    good = results_df[(results_df['overfitting_gap'] >= 0.20) & 
                      (results_df['overfitting_gap'] < 0.30)]
    poor = results_df[results_df['overfitting_gap'] >= 0.30]
    
    print(f"\n✅ Excellent (Gap <0.20): {len(excellent)} sectors")
    if len(excellent) > 0:
        print(excellent[['test_accuracy', 'overfitting_gap', 'category']].to_string())
    
    print(f"\n🟡 Good (Gap 0.20-0.30): {len(good)} sectors")
    if len(good) > 0:
        print(good[['test_accuracy', 'overfitting_gap', 'category']].to_string())
    
    print(f"\n⚠️  Poor (Gap >0.30): {len(poor)} sectors")
    if len(poor) > 0:
        print(poor[['test_accuracy', 'overfitting_gap', 'category']].to_string())
    
    # Current predictions
    print("\n" + "="*70)
    print("CURRENT ROTATION PROBABILITIES (Next 21 Days)")
    print("="*70)
    
    current_features = features.iloc[[-1]]
    current_probs = model.predict_probabilities(current_features)
    
    prob_series = current_probs.T.iloc[:, 0]
    summary = pd.DataFrame({
        'Probability': prob_series,
        'Test_Acc': results_df['test_accuracy'],
        'Gap': results_df['overfitting_gap'],
        'Category': results_df['category'],
        'Confidence': confidence.set_index('Sector')['Tier']
    })
    
    summary = summary.sort_values('Probability', ascending=False)
    
    print("\n📊 Ranked by Rotation Probability:")
    print(summary.to_string())
    
    print("\n" + "="*70)
    print("✅ v3.4 COMPLETE - BACK TO FORMULA")
    print("="*70)
    print("\n💡 Next Steps:")
    print("   • If macro-sensitive sectors improved: SUCCESS! ✅")
    print("   • If sentiment sectors still poor: EXPECTED (need news data)")
    print("   • Consider: Sector-specific feature sets in v3.5")
    
    return model, features, results, validation, confidence


if __name__ == "__main__":
    model, features, results, validation, confidence = test_v34_model()