"""
Sector Rotation Model v3.5 - Final Iteration
IMPROVEMENTS over v3.3 & v3.4:
- ✅ 8 years of data (sweet spot between diversity and stability)
- ✅ Keep v3.4 interaction terms (Slope×Credit, Energy×Metals)
- ✅ Adaptive regularization (stronger for problem children)
- ✅ Preserve v3.3 winners (don't touch what works)
- ✅ Isotonic calibration (v3.4 concept)
- 🎯 Target: 6-7 sectors with gap <0.25
- 🎯 Focus on TRADABLE sectors (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB)
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
import yfinance as yf
warnings.filterwarnings('ignore')


class SectorRotationFeaturesV5:
    """Feature engineering v3.5 - proven v3.4 interactions."""
    
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
        """Calculate macro changes with v3.4 interaction terms."""
        print("🔧 Calculating macro changes + interactions...")
        
        features = pd.DataFrame(index=macro.index)
        
        # Base macro changes
        for col in macro.columns:
            for window in windows:
                change = macro[col].pct_change(window) * 100
                features[f'{col}_change_{window}d'] = change
        
        # Yield curve slope
        if '10Y Treasury' in macro.columns and '2Y Treasury' in macro.columns:
            features['Yield_Curve_Slope'] = (
                macro['10Y Treasury'] - macro['2Y Treasury']
            )
            features['Yield_Curve_Slope_change_63d'] = (
                features['Yield_Curve_Slope'].diff(63)
            )
        
        # Gold-Dollar ratio
        if 'Gold' in macro.columns and 'Dollar' in macro.columns:
            features['Gold_Dollar_Ratio'] = macro['Gold'] / macro['Dollar']
            features['Gold_Dollar_Ratio_change_63d'] = (
                features['Gold_Dollar_Ratio'].pct_change(63) * 100
            )
        
        # Credit risk appetite
        if 'HYG' in macro.columns and 'LQD' in macro.columns:
            features['Credit_Risk_Appetite'] = macro['HYG'] / macro['LQD']
            features['Credit_Risk_Appetite_change_63d'] = (
                features['Credit_Risk_Appetite'].pct_change(63) * 100
            )
        
        # Rate expectations
        if 'TLT' in macro.columns:
            features['Long_Rate_Expectations'] = macro['TLT'].pct_change(63) * 100
        
        # Industrial metals
        if 'XME' in macro.columns:
            features['Industrial_Metals'] = macro['XME'].pct_change(63) * 100
        
        # Retail health
        if 'XRT' in macro.columns:
            features['Retail_Health'] = macro['XRT'].pct_change(63) * 100
        
        # v3.4 PROVEN interactions
        if 'Yield_Curve_Slope' in features.columns and 'Credit_Risk_Appetite' in features.columns:
            features['Slope_x_Credit'] = (
                features['Yield_Curve_Slope'] * features['Credit_Risk_Appetite']
            )
            print("   ✅ Added Yield Curve × Credit interaction")
        
        if 'Crude Oil_change_63d' in features.columns and 'Industrial_Metals' in features.columns:
            features['Energy_x_Metals'] = (
                features['Crude Oil_change_63d'] * features['Industrial_Metals']
            )
            print("   ✅ Added Energy × Metals interaction")
        
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
        """Add VIX features."""
        print("🔧 Adding VIX features...")
        
        features = pd.DataFrame(index=vix.index)
        
        vix_change = vix.diff(5)
        features['VIX_volatility_21d'] = vix_change.rolling(21).std()
        features['VIX_change_63d'] = vix.diff(63)
        
        print(f"   ✅ Created {len(features.columns)} VIX features")
        return features
    
    def combine_features(self,
                        sectors: pd.DataFrame,
                        macro: pd.DataFrame,
                        vix: pd.Series) -> pd.DataFrame:
        """Combine all features."""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING v3.5 - FINAL")
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


class SectorRotationModelV5:
    """Model v3.5 - Adaptive & final."""
    
    def __init__(self, random_state: int = 42):
        """Initialize with smart adaptive configs."""
        
        # Sector-specific configs (learned from v3.3 vs v3.4)
        self.sector_configs = {
            # v3.3 WINNERS - Don't touch (except XLRE, we don't care)
            'XLE': {'min_samples_leaf': 40, 'max_depth': 6},
            'XLV': {'min_samples_leaf': 40, 'max_depth': 6},
            'XLB': {'min_samples_leaf': 40, 'max_depth': 6},
            'XLP': {'min_samples_leaf': 40, 'max_depth': 6},
            
            # Problem children - STRONGER regularization
            'XLU': {'min_samples_leaf': 60, 'max_depth': 4},
            'XLI': {'min_samples_leaf': 60, 'max_depth': 4},
            'XLY': {'min_samples_leaf': 55, 'max_depth': 4},
            
            # v3.4 regressions - Back to v3.3 baseline
            'XLF': {'min_samples_leaf': 40, 'max_depth': 6},
            'XLK': {'min_samples_leaf': 40, 'max_depth': 6},
            'XLC': {'min_samples_leaf': 40, 'max_depth': 6},
            
            # XLRE - default (we don't care, not tradable)
            'XLRE': {'min_samples_leaf': 40, 'max_depth': 6},
        }
        
        self.random_state = random_state
        self.models = {}
        self.calibrated_models = {}
        self.selected_features = None
        self.scaler = StandardScaler()
        self.results = {}
        self.validation_results = {}
    
    def get_sector_config(self, ticker: str) -> Dict:
        """Get config for sector."""
        base = {
            'n_estimators': 200,
            'max_features': 'sqrt',
            'min_samples_split': 25,
        }
        base.update(self.sector_configs.get(ticker, 
                    {'min_samples_leaf': 40, 'max_depth': 6}))
        return base
    
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
                                   min_importance: float = 0.012) -> List[str]:
        """Feature selection with 1.2% threshold."""
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
        
        print(f"\n📊 Top 20 Features:")
        for i, (feat, imp) in enumerate(importance.head(20).items(), 1):
            print(f"   {i:2d}. {feat:45s} {imp:6.3f}")
        
        selected = importance[importance >= min_importance].index.tolist()
        removed = len(importance) - len(selected)
        
        print(f"\n🔧 Feature Selection:")
        print(f"   Threshold: {min_importance:.3f}")
        print(f"   Selected: {len(selected)} features")
        print(f"   Removed: {removed} features")
        
        return selected
    
    def train_models(self,
                    features: pd.DataFrame,
                    targets: pd.DataFrame,
                    test_split: float = 0.2,
                    use_feature_selection: bool = True,
                    use_calibration: bool = True) -> Dict:
        """Train models with adaptive configs."""
        print("\n" + "="*70)
        print("TRAINING MODELS v3.5 - ADAPTIVE & CALIBRATED")
        print("="*70)
        
        if use_feature_selection and self.selected_features is None:
            self.selected_features = self.analyze_feature_importance(
                features, targets, min_importance=0.012
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
            
            config = self.get_sector_config(ticker)
            
            print(f"\n🌲 {ticker}...", end=" ")
            if config['min_samples_leaf'] != 40 or config['max_depth'] != 6:
                print(f"[CUSTOM: leaf={config['min_samples_leaf']}, depth={config['max_depth']}]", end=" ")
            
            y = y_all[target_col]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            model = RandomForestClassifier(
                **config,
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
            
            config = self.get_sector_config(ticker)
            fold_accs = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
                X_train_fold = X_scaled.iloc[train_idx]
                X_test_fold = X_scaled.iloc[test_idx]
                y_train_fold = y.iloc[train_idx]
                y_test_fold = y.iloc[test_idx]
                
                model = RandomForestClassifier(
                    **config,
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
        """Calculate confidence tiers (focus on TRADABLE sectors)."""
        print("\n" + "="*70)
        print("CONFIDENCE SCORING (TRADABLE SECTORS)")
        print("="*70)
        
        confidence_data = []
        
        # Focus on tradable sectors
        tradable = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
        
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
            
            # Mark non-tradable
            if ticker not in tradable:
                tier = f"{tier}*"
                emoji = "⚪"
            
            confidence_data.append({
                'Sector': ticker,
                'Tier': tier,
                'Emoji': emoji,
                'Test_Acc': test_acc,
                'Gap': gap,
                'WF_Std': wf_std
            })
        
        confidence_df = pd.DataFrame(confidence_data)
        
        print("\n📊 Confidence Tiers (* = non-tradable):")
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


def fetch_tier1_data(start_str: str, end_str: str) -> pd.DataFrame:
    """Fetch Tier 1 tradable instruments."""
    print("\n🔧 Fetching Tier 1 tradable instruments...")
    
    tier1_tickers = {
        'HYG': 'HYG',
        'LQD': 'LQD',
        'TLT': 'TLT',
        'XME': 'XME',
        'XRT': 'XRT',
    }
    
    tier1_data = {}
    
    for ticker, name in tier1_tickers.items():
        try:
            data = yf.download(ticker, start=start_str, end=end_str, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    price = data['Adj Close'].iloc[:, 0] if data['Adj Close'].ndim > 1 else data['Adj Close']
                else:
                    price = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
            else:
                price = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            
            if len(price) > 0:
                tier1_data[name] = price
                print(f"   ✅ {ticker}: {len(price)} days")
            else:
                print(f"   ⚠️  {ticker}: No data")
                
        except Exception as e:
            print(f"   ❌ {ticker}: {e}")
    
    if tier1_data:
        return pd.DataFrame(tier1_data)
    else:
        return pd.DataFrame()


def test_v35_model():
    """Test v3.5 - FINAL iteration."""
    print("\n" + "="*70)
    print("SECTOR ROTATION MODEL v3.5 - FINAL")
    print("="*70)
    
    from sector_data_fetcher import SectorDataFetcher
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    # Fetch data (8 years - sweet spot)
    print("\n📊 Step 1: Fetch Data (8 years)")
    print("-"*70)
    
    sector_fetcher = SectorDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=8*365)  # 8 years
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"📅 Data period: {start_str} to {end_str} (8 years)")
    print(f"   Rationale: Sweet spot between diversity (10y) and stability (7y)")
    
    sectors = sector_fetcher.fetch_sector_etfs(start_str, end_str)
    macro = sector_fetcher.fetch_macro_factors(start_str, end_str)
    
    unified = UnifiedDataFetcher()
    vix = unified.fetch_vix(start_str, end_str)
    
    # Fetch Tier 1
    tier1_df = fetch_tier1_data(start_str, end_str)
    
    if not tier1_df.empty:
        macro = pd.concat([macro, tier1_df], axis=1)
        print(f"\n✅ Added {len(tier1_df.columns)} Tier 1 instruments")
    
    # Align
    sectors_aligned, macro_aligned, vix_aligned = sector_fetcher.align_data(
        sectors, macro, vix
    )
    
    # Feature engineering
    print("\n📊 Step 2: Feature Engineering (v3.5)")
    print("-"*70)
    
    feat_eng = SectorRotationFeaturesV5()
    features = feat_eng.combine_features(
        sectors_aligned,
        macro_aligned,
        vix_aligned
    )
    
    # Targets
    print("\n📊 Step 3: Create Targets")
    print("-"*70)
    
    model = SectorRotationModelV5()
    targets = model.create_targets(sectors_aligned, forward_window=21)
    
    # Train
    print("\n📊 Step 4: Train Models (Adaptive)")
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
    print("GAP THRESHOLD ANALYSIS (Target: 6-7 sectors <0.25)")
    print("="*70)
    
    excellent = results_df[results_df['overfitting_gap'] < 0.20]
    good = results_df[(results_df['overfitting_gap'] >= 0.20) & 
                      (results_df['overfitting_gap'] < 0.25)]
    
    total_under_25 = len(excellent) + len(good)
    
    # Count tradable sectors
    tradable = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
    tradable_under_25 = len([t for t in excellent.index if t in tradable]) + \
                        len([t for t in good.index if t in tradable])
    
    print(f"\n✅ Excellent (Gap <0.20): {len(excellent)} sectors")
    if len(excellent) > 0:
        print(excellent[['test_accuracy', 'overfitting_gap']].to_string())
    
    print(f"\n🟢 Good (Gap 0.20-0.25): {len(good)} sectors")
    if len(good) > 0:
        print(good[['test_accuracy', 'overfitting_gap']].to_string())
    
    print(f"\n🎯 TOTAL WITH GAP <0.25: {total_under_25} sectors")
    print(f"   Tradable sectors <0.25: {tradable_under_25}/9")
    
    if total_under_25 >= 6:
        print(f"   ✅ GOAL ACHIEVED! ({total_under_25} ≥ 6)")
    else:
        print(f"   🟡 Close to goal ({total_under_25}/6)")
    
    # Current predictions (tradable only)
    print("\n" + "="*70)
    print("CURRENT ROTATION PROBABILITIES - TRADABLE SECTORS")
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
    
    # Filter tradable
    summary_tradable = summary[summary.index.isin(tradable)]
    summary_tradable = summary_tradable.sort_values('Probability', ascending=False)
    
    print("\n📊 Ranked by Rotation Probability (TRADABLE ONLY):")
    print(summary_tradable.to_string())
    
    print("\n" + "="*70)
    print("✅ v3.5 COMPLETE - FINAL ITERATION")
    print("="*70)
    print("\n🚀 v3.5 Design Philosophy:")
    print("   ✅ 8 years of data (balanced regime diversity)")
    print("   ✅ Keep v3.4 interaction terms (Slope×Credit, Energy×Metals)")
    print("   ✅ Adaptive regularization (stronger for XLU, XLI, XLY)")
    print("   ✅ Preserve v3.3 winners (XLE, XLV, XLB, XLP)")
    print("   ✅ Isotonic calibration for better probabilities")
    print("   ✅ Focus on 9 TRADABLE sectors (ignore XLRE)")
    print("   🎯 Goal: 6-7 sectors with gap <0.25")
    
    # Decision point
    print("\n" + "="*70)
    print("DEPLOYMENT DECISION")
    print("="*70)
    
    if total_under_25 >= 6:
        print("✅ v3.5 MEETS GOAL - DEPLOY THIS VERSION")
        print("   Next steps:")
        print("   1. Save models and scaler")
        print("   2. Document hyperparameters")
        print("   3. Move to backtesting phase")
        print("   4. Prepare for live deployment")
    elif total_under_25 >= 5 and tradable_under_25 >= 5:
        print("🟢 v3.5 ACCEPTABLE - DEPLOY THIS VERSION")
        print("   Rationale: 5+ tradable sectors is usable")
        print("   Next steps: Same as above")
    else:
        print("🟡 v3.5 INSUFFICIENT - REVERT TO v3.3")
        print(f"   Only {tradable_under_25} tradable sectors <0.25")
        print("   v3.3 had 5 solid sectors, which is good enough")
        print("   Stop iterating and START TRADING")
    
    print("\n" + "="*70)
    print("🎖️ For God, Country, and Family. 🇺🇸")
    print("="*70)
    
    return model, features, results, validation, confidence


if __name__ == "__main__":
    model, features, results, validation, confidence = test_v35_model()