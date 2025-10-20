"""
Sector Rotation Model v3.7 - "The v2.0 Phoenix"
PHILOSOPHY: Back to v2.0 simplicity with v3.0+ infrastructure
CHANGES:
- ✅ LESS regularization (leaf=25, depth=7 like v2.0)
- ✅ Feature threshold 1.0% (keep more features)
- ✅ Keep ALL proven features (no aggressive cutting)
- ✅ CORE 5 factors only (Gold, Oil, Dollar, 10Y, 2Y)
- ✅ 7 years of data (2018-2025)
- ✅ Keep v3.0 infrastructure (caching, diagnostics, calibration)

Goal: Get 5+ sectors with gap <0.25, XLF back to 55%+
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SectorRotationFeaturesV7:
    """Feature engineering - v2.0 simplicity."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_relative_strength(self,
                                    sectors: pd.DataFrame,
                                    windows: List[int] = [63, 126]) -> pd.DataFrame:
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
                                windows: List[int] = [63]) -> pd.DataFrame:
        """Calculate macro changes - simple and clean."""
        print("🔧 Calculating macro changes (CORE 5 factors)...")
        
        features = pd.DataFrame(index=macro.index)
        
        # Simple percentage changes
        for col in macro.columns:
            for window in windows:
                change = macro[col].pct_change(window) * 100
                features[f'{col}_change_{window}d'] = change
        
        # Yield curve slope (proven important)
        if '10Y Treasury' in macro.columns and '2Y Treasury' in macro.columns:
            features['Yield_Curve_Slope'] = (
                macro['10Y Treasury'] - macro['2Y Treasury']
            )
            features['Yield_Curve_Slope_change_63d'] = (
                features['Yield_Curve_Slope'].diff(63)
            )
        
        # Gold-Dollar ratio (proven in all versions)
        if 'Gold' in macro.columns and 'Dollar' in macro.columns:
            features['Gold_Dollar_Ratio'] = macro['Gold'] / macro['Dollar']
            features['Gold_Dollar_Ratio_change_63d'] = (
                features['Gold_Dollar_Ratio'].pct_change(63) * 100
            )
        
        print(f"   ✅ Created {len(features.columns)} macro features")
        return features
    
    def add_seasonality(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add seasonality - Month only."""
        print("🔧 Adding seasonality...")
        
        features = pd.DataFrame(index=index)
        features['Month'] = index.month
        
        print(f"   ✅ Created {len(features.columns)} seasonality features")
        return features
    
    def add_vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """Add VIX features - proven important."""
        print("🔧 Adding VIX features...")
        
        features = pd.DataFrame(index=vix.index)
        
        # VIX volatility (top-3 feature historically)
        vix_change = vix.diff(5)
        features['VIX_volatility_21d'] = vix_change.rolling(21).std()
        
        # VIX change
        features['VIX_change_63d'] = vix.diff(63)
        
        print(f"   ✅ Created {len(features.columns)} VIX features")
        return features
    
    def combine_features(self,
                        sectors: pd.DataFrame,
                        macro: pd.DataFrame,
                        vix: pd.Series) -> pd.DataFrame:
        """Combine all features - simple approach."""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING v3.7 - THE v2.0 PHOENIX")
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


class SectorRotationModelV7:
    """Model v3.7 - v2.0 Phoenix with v3.0 infrastructure."""
    
    def __init__(self, random_state: int = 42):
        """Initialize with v2.0 style hyperparameters."""
        
        # v2.0 HYPERPARAMETERS (less regularization)
        self.base_config = {
            'n_estimators': 200,
            'max_depth': 7,           # v2.0 setting
            'min_samples_split': 20,  # v2.0 setting
            'min_samples_leaf': 25,   # v2.0 setting (not 30-60!)
            'max_features': 'sqrt',
        }
        
        self.random_state = random_state
        self.models = {}
        self.calibrated_models = {}
        self.selected_features = None
        self.scaler = StandardScaler()
        self.results = {}
        self.validation_results = {}
    
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
                                   min_importance: float = 0.010) -> List[str]:
        """Feature selection with 1.0% threshold (keep more features)."""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS (v3.7 - Less Aggressive)")
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
            min_samples_leaf=25,
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
        print(f"   Threshold: {min_importance:.3f} (1.0% - less aggressive)")
        print(f"   Selected: {len(selected)} features")
        print(f"   Removed: {removed} features")
        
        return selected
    
    def train_models(self,
                    features: pd.DataFrame,
                    targets: pd.DataFrame,
                    test_split: float = 0.2,
                    use_feature_selection: bool = True,
                    use_calibration: bool = True) -> Dict:
        """Train models with v2.0 hyperparameters."""
        print("\n" + "="*70)
        print("TRAINING MODELS v3.7 - THE v2.0 PHOENIX")
        print("="*70)
        
        if use_feature_selection and self.selected_features is None:
            self.selected_features = self.analyze_feature_importance(
                features, targets, min_importance=0.010
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
        
        print(f"\n📊 Hyperparameters (v2.0 STYLE - Less Regularization):")
        for key, val in self.base_config.items():
            print(f"   {key}: {val}")
        
        results = {}
        
        for target_col in y_all.columns:
            ticker = target_col.replace('_outperform', '')
            
            print(f"\n🌲 {ticker}...", end=" ")
            
            y = y_all[target_col]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            model = RandomForestClassifier(
                **self.base_config,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Calibrate
            if use_calibration:
                calibrated = CalibratedClassifierCV(
                    model, 
                    method='isotonic',
                    cv='prefit'
                )
                calibrated.fit(X_train, y_train)
                self.calibrated_models[ticker] = calibrated
                eval_model = calibrated
            else:
                eval_model = model
            
            self.models[ticker] = model
            
            train_acc = eval_model.score(X_train, y_train)
            test_acc = eval_model.score(X_test, y_test)
            gap = train_acc - test_acc
            
            results[ticker] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting_gap': gap,
            }
            
            # Gap status
            if gap < 0.20:
                status = "✅ Excellent"
            elif gap < 0.25:
                status = "🟢 Good"
            elif gap < 0.30:
                status = "🟡 Fair"
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
                    **self.base_config,
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
            
            # Tier logic
            if gap < 0.20 and test_acc > 0.55 and wf_std < 0.10:
                tier = "HIGH"
                emoji = "🟢"
            elif gap < 0.25 and test_acc > 0.52 and wf_std < 0.10:
                tier = "GOOD"
                emoji = "🟢"
            elif gap < 0.30 and test_acc > 0.50:
                tier = "MEDIUM"
                emoji = "🟡"
            else:
                tier = "LOW"
                emoji = "🔴"
            
            confidence_data.append({
                'Sector': ticker,
                'Tier': tier,
                'Emoji': emoji,
                'Test_Acc': test_acc,
                'Gap': gap,
                'WF_Std': wf_std
            })
        
        confidence_df = pd.DataFrame(confidence_data)
        
        print("\n📊 Confidence Tiers:")
        print(confidence_df.to_string(index=False))
        
        return confidence_df
    
    def predict_probabilities(self, features: pd.DataFrame, use_calibrated: bool = True) -> pd.DataFrame:
        """Predict rotation probabilities."""
        if self.selected_features is not None:
            features = features[self.selected_features]
        
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )
        
        probs = pd.DataFrame(index=features.index)
        
        for ticker in self.models.keys():
            if use_calibrated and ticker in self.calibrated_models:
                model = self.calibrated_models[ticker]
            else:
                model = self.models[ticker]
            
            prob = model.predict_proba(features_scaled)[:, 1]
            probs[ticker] = prob
        
        return probs


def test_v37_phoenix():
    """Test v3.7 - The v2.0 Phoenix."""
    print("\n" + "="*70)
    print("SECTOR ROTATION MODEL v3.7 - THE v2.0 PHOENIX")
    print("="*70)
    print("🔥 Combining v2.0 simplicity with v3.0 infrastructure")
    
    from sector_data_fetcher import SectorDataFetcher
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    # Fetch data (7 years)
    print("\n📊 Step 1: Fetch Data (7 years)")
    print("-"*70)
    
    sector_fetcher = SectorDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"📅 Data period: {start_str} to {end_str} (7 years)")
    print(f"   Rationale: Proven stable regime")
    
    sectors = sector_fetcher.fetch_sector_etfs(start_str, end_str)
    macro = sector_fetcher.fetch_macro_factors(start_str, end_str)
    
    unified = UnifiedDataFetcher()
    vix = unified.fetch_vix(start_str, end_str)
    
    print("\n✅ Using CORE 5 factors only (no HYG, LQD, TLT, XME, XRT)")
    
    # Align
    sectors_aligned, macro_aligned, vix_aligned = sector_fetcher.align_data(
        sectors, macro, vix
    )
    
    # Feature engineering
    print("\n📊 Step 2: Feature Engineering (v3.7 - Phoenix)")
    print("-"*70)
    
    feat_eng = SectorRotationFeaturesV7()
    features = feat_eng.combine_features(
        sectors_aligned,
        macro_aligned,
        vix_aligned
    )
    
    # Targets
    print("\n📊 Step 3: Create Targets")
    print("-"*70)
    
    model = SectorRotationModelV7()
    targets = model.create_targets(sectors_aligned, forward_window=21)
    
    # Train
    print("\n📊 Step 4: Train Models (v2.0 Hyperparameters)")
    print("-"*70)
    
    results = model.train_models(
        features, targets, 
        use_feature_selection=True,
        use_calibration=True
    )
    
    # Validate
    print("\n📊 Step 5: Walk-Forward Validation")
    print("-"*70)
    
    validation = model.walk_forward_validate(features, targets, n_splits=5)
    
    # Confidence
    print("\n📊 Step 6: Confidence Scoring")
    print("-"*70)
    
    confidence = model.calculate_confidence_scores()
    
    # Summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results).T
    print("\n📊 Performance Metrics:")
    print(results_df[['test_accuracy', 'overfitting_gap']].round(3))
    
    # Gap analysis
    print("\n" + "="*70)
    print("GAP THRESHOLD ANALYSIS (Goal: 5+ sectors <0.25)")
    print("="*70)
    
    excellent = results_df[results_df['overfitting_gap'] < 0.20]
    good = results_df[(results_df['overfitting_gap'] >= 0.20) & 
                      (results_df['overfitting_gap'] < 0.25)]
    
    total_under_25 = len(excellent) + len(good)
    
    print(f"\n✅ Excellent (Gap <0.20): {len(excellent)} sectors")
    if len(excellent) > 0:
        print(excellent[['test_accuracy', 'overfitting_gap']].to_string())
    
    print(f"\n🟢 Good (Gap 0.20-0.25): {len(good)} sectors")
    if len(good) > 0:
        print(good[['test_accuracy', 'overfitting_gap']].to_string())
    
    print(f"\n🎯 TOTAL WITH GAP <0.25: {total_under_25} sectors")
    
    if total_under_25 >= 5:
        print(f"   ✅ GOAL ACHIEVED! ({total_under_25} ≥ 5)")
    else:
        print(f"   🟡 Close ({total_under_25}/5)")
    
    # vs v3.0 comparison
    print("\n" + "="*70)
    print("📊 vs v3.0 COMPARISON")
    print("="*70)
    
    v30_results = {
        'XLK': {'acc': 0.563, 'gap': 0.291},
        'XLF': {'acc': 0.526, 'gap': 0.321},
        'XLE': {'acc': 0.639, 'gap': 0.231},
        'XLP': {'acc': 0.621, 'gap': 0.226},
        'XLRE': {'acc': 0.688, 'gap': 0.178},
        'XLB': {'acc': 0.633, 'gap': 0.212},
    }
    
    print(f"\n{'Sector':<6} {'v3.0 Acc':>10} {'v3.7 Acc':>10} {'Δ':>8} {'v3.0 Gap':>10} {'v3.7 Gap':>10} {'Δ':>8}")
    print("-" * 66)
    
    for ticker in ['XLK', 'XLF', 'XLE', 'XLP', 'XLRE', 'XLB']:
        if ticker in results_df.index:
            v30_acc = v30_results[ticker]['acc']
            v37_acc = results_df.loc[ticker, 'test_accuracy']
            v30_gap = v30_results[ticker]['gap']
            v37_gap = results_df.loc[ticker, 'overfitting_gap']
            
            acc_delta = v37_acc - v30_acc
            gap_delta = v37_gap - v30_gap
            
            acc_emoji = "✅" if acc_delta > 0 else "🔴"
            gap_emoji = "✅" if gap_delta < 0 else "🔴"
            
            print(f"{ticker:<6} {v30_acc:>10.3f} {v37_acc:>10.3f} {acc_delta:>7.3f}{acc_emoji} {v30_gap:>10.3f} {v37_gap:>10.3f} {gap_delta:>7.3f}{gap_emoji}")
    
    # Current predictions
    print("\n" + "="*70)
    print("CURRENT ROTATION PROBABILITIES (Next 21 Days)")
    print("="*70)
    
    current_features = features.iloc[[-1]]
    current_probs = model.predict_probabilities(current_features, use_calibrated=True)
    
    prob_series = current_probs.T.iloc[:, 0]
    summary = pd.DataFrame({
        'Probability': prob_series,
        'Test_Acc': results_df['test_accuracy'],
        'Gap': results_df['overfitting_gap'],
        'Confidence': confidence.set_index('Sector')['Tier']
    })
    
    summary = summary.sort_values('Probability', ascending=False)
    
    print("\n📊 Ranked by Rotation Probability:")
    print(summary.to_string())
    
    print("\n" + "="*70)
    print("✅ v3.7 COMPLETE - THE v2.0 PHOENIX")
    print("="*70)
    print("\n🔥 v3.7 Changes:")
    print("   ✅ LESS regularization (leaf=25, depth=7 like v2.0)")
    print("   ✅ Feature threshold 1.0% (keep more features)")
    print("   ✅ CORE 5 factors only")
    print("   ✅ Keep v3.0 infrastructure (caching, diagnostics)")
    print("   🎯 Goal: Get 5+ sectors <0.25 gap\n")
    
    # FINAL DECISION
    print("\n" + "="*70)
    print("🚀 DEPLOYMENT DECISION")
    print("="*70)
    
    if total_under_25 >= 5:
        print("✅ v3.7 MEETS GOAL - DEPLOY THIS VERSION")
        print("   Next: Build visualizer for v3.7")
    else:
        print("🟡 v3.7 CLOSE - Use best of v3.0/v3.7")
        print("   Next: Build visualizer, START TRADING")
    
    print("\n🎖️ For God, Country, and Family. 🇺🇸")
    
    return model, features, results, validation, confidence


if __name__ == "__main__":
    model, features, results, validation, confidence = test_v37_phoenix()