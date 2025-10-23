"""
SPX Feature Explorer - UNIFIED EDITION
Single file to test both legitimate vol features AND interaction features

Usage from Jupyter:
    from spx_feature_explorer import SPXFeatureExplorer
    
    explorer = SPXFeatureExplorer()
    
    # Option 1: Test everything
    results = explorer.run_complete_analysis(years=7)
    
    # Option 2: Just vol features
    results = explorer.run_vol_feature_test(years=7)
    
    # Option 3: Just interactions
    results = explorer.run_interaction_test(years=7)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE, TEST_SPLIT


class SPXFeatureExplorer:
    """
    Comprehensive feature testing pipeline.
    
    Tests TWO types of new features:
    1. Legitimate Vol Features (VIX vs Past RV - NO lookahead)
    2. Interaction Features (combinations of existing features)
    
    All data fetching and orchestration handled internally.
    """
    
    def __init__(self):
        self.baseline_results = None
        self.test_results = None
        self.spx = None
        self.vix = None
        self.predictor = None
    
    # ========================================
    # DATA FETCHING (Internal)
    # ========================================
    
    def _fetch_data(self, years: int = 7) -> Tuple[pd.Series, pd.Series]:
        """
        Fetch SPX and VIX data internally.
        
        Returns:
            (spx, vix) - both timezone-naive
        """
        print("\n" + "="*70)
        print("FETCHING DATA")
        print("="*70)
        
        from UnifiedDataFetcher import UnifiedDataFetcher
        from datetime import datetime, timedelta
        
        fetcher = UnifiedDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"\n📊 Date Range: {start_str} to {end_str}")
        
        # Fetch SPX
        print("   Fetching SPX...")
        spx_df = fetcher.fetch_spx(start_str, end_str)
        spx = spx_df['Close'].squeeze()
        
        if spx.index.tz is not None:
            spx.index = spx.index.tz_localize(None)
        
        # Fetch VIX
        print("   Fetching VIX...")
        vix = fetcher.fetch_vix(start_str, end_str)
        
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        
        print(f"\n✅ SPX: {len(spx)} days")
        print(f"✅ VIX: {len(vix)} days")
        
        self.spx = spx
        self.vix = vix
        
        return spx, vix
    
    def _train_baseline(self, years: int = 7) -> 'SPXPredictor':
        """
        Train baseline model internally.
        
        Returns:
            Trained SPXPredictor
        """
        print("\n" + "="*70)
        print("TRAINING BASELINE MODEL")
        print("="*70)
        
        from spx_predictor import SPXPredictor
        
        predictor = SPXPredictor(use_iv_rv_cheat=False)
        predictor.train(years=years)
        
        self.predictor = predictor
        
        return predictor
    
    # ========================================
    # VOL FEATURES (VIX vs Past RV)
    # ========================================
    
    def generate_vix_vs_rv_features(self, 
                                     spx: pd.Series, 
                                     vix: pd.Series) -> pd.DataFrame:
        """
        Generate VIX vs Historical Realized Volatility features.
        NO LOOKAHEAD BIAS - uses only backward-looking data.
        
        Args:
            spx: SPX closing prices (raw, not scaled)
            vix: VIX levels (raw, not scaled)
        
        Returns:
            DataFrame with legitimate vol features
        """
        print("\n" + "="*70)
        print("GENERATING LEGITIMATE VOL FEATURES")
        print("="*70)
        print("📊 VIX vs Past Realized Volatility (No Lookahead)")
        
        features = pd.DataFrame(index=vix.index)
        returns = spx.pct_change()
        
        # ========================================
        # 1. VIX vs Past Realized Vol (Multiple Horizons)
        # ========================================
        print("\n1️⃣ VIX Premium over Past Realized Vol")
        
        for window in [21, 30, 63]:
            # Past realized vol (annualized)
            past_rv = returns.rolling(window).std() * np.sqrt(252) * 100
            
            # Spread: VIX - Past RV
            features[f'vix_vs_rv_{window}d'] = vix - past_rv
            
            # Ratio: VIX / Past RV
            features[f'vix_rv_ratio_{window}d'] = vix / past_rv.replace(0, np.nan)
            
            print(f"   ✓ Created: vix_vs_rv_{window}d, vix_rv_ratio_{window}d")
        
        # ========================================
        # 2. VIX vs Weighted Average RV (Robust Measure)
        # ========================================
        print("\n2️⃣ VIX vs Weighted Average RV")
        
        rv_21 = returns.rolling(21).std() * np.sqrt(252) * 100
        rv_30 = returns.rolling(30).std() * np.sqrt(252) * 100
        rv_63 = returns.rolling(63).std() * np.sqrt(252) * 100
        
        avg_rv = (rv_21 * 0.4 + rv_30 * 0.35 + rv_63 * 0.25)
        
        features['vix_vs_avg_rv'] = vix - avg_rv
        features['vix_avg_rv_ratio'] = vix / avg_rv.replace(0, np.nan)
        
        print("   ✓ Created: vix_vs_avg_rv, vix_avg_rv_ratio")
        
        # ========================================
        # 3. Historical Context of Spread
        # ========================================
        print("\n3️⃣ Historical Context")
        
        spread_21 = vix - rv_21
        features['vix_rv_spread_percentile'] = spread_21.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
        
        # Z-score of spread
        features['vix_rv_spread_zscore'] = (
            (spread_21 - spread_21.rolling(252).mean()) / 
            spread_21.rolling(252).std()
        )
        
        print("   ✓ Created: vix_rv_spread_percentile, vix_rv_spread_zscore")
        
        # ========================================
        # 4. VIX Velocity & Regime Features
        # ========================================
        print("\n4️⃣ VIX Dynamics")
        
        # How fast is VIX moving?
        features['vix_velocity_5d'] = vix.diff(5)
        features['vix_velocity_21d'] = vix.diff(21)
        
        # VIX term structure proxy
        vix_ma_5 = vix.rolling(5).mean()
        vix_ma_21 = vix.rolling(21).mean()
        features['vix_term_structure'] = vix - vix_ma_21
        features['vix_term_structure_slope'] = vix_ma_5 - vix_ma_21
        
        # VIX stability (how volatile is volatility?)
        features['vix_stability_21d'] = vix.rolling(21).std()
        features['vix_stability_63d'] = vix.rolling(63).std()
        
        print("   ✓ Created: vix_velocity, vix_term_structure, vix_stability")
        
        # ========================================
        # 5. Mean Reversion Signals
        # ========================================
        print("\n5️⃣ Mean Reversion Signals")
        
        for window in [21, 63, 126, 252]:
            vix_ma = vix.rolling(window).mean()
            features[f'vix_reversion_{window}d'] = (
                (vix - vix_ma) / vix_ma.replace(0, np.nan) * 100
            )
        
        print("   ✓ Created: vix_reversion features (4 windows)")
        
        # ========================================
        # 6. Regime Classification (Enhanced)
        # ========================================
        print("\n6️⃣ Enhanced Regime Features")
        
        # Basic regime (same as production)
        features['vix_regime'] = pd.cut(
            vix, 
            bins=[0, 15, 20, 30, 100], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Days in regime
        regime_change = features['vix_regime'] != features['vix_regime'].shift(1)
        regime_id = regime_change.cumsum()
        features['days_in_regime'] = regime_id.groupby(regime_id).cumcount() + 1
        
        # Regime transition flag
        features['regime_transition'] = regime_change.astype(int)
        
        print("   ✓ Created: vix_regime, days_in_regime, regime_transition")
        
        # Drop NaNs
        features = features.dropna()
        
        print(f"\n✅ Generated {len(features.columns)} legitimate vol features")
        print(f"   Samples: {len(features)} (after dropna)")
        
        return features
    
    # ========================================
    # INTERACTION FEATURES
    # ========================================
    
    def generate_interactions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction terms for top features.
        
        Theory: Combinations may be more predictive than individuals.
        
        Args:
            features_df: Existing feature matrix (scaled)
        
        Returns:
            DataFrame with NEW interaction features only
        """
        print("\n" + "="*70)
        print("GENERATING INTERACTION FEATURES")
        print("="*70)
        
        df = features_df.copy()
        interactions = pd.DataFrame(index=df.index)
        
        # ========================================
        # GROUP 1: IV-RV Interactions
        # ========================================
        print("\n📊 Group 1: IV-RV Interactions")
        
        if 'iv_rv_spread' in df.columns:
            # Non-linear: extreme spreads matter more
            interactions['iv_rv_spread_sq'] = df['iv_rv_spread'] ** 2
            interactions['iv_rv_spread_abs'] = df['iv_rv_spread'].abs()
            print("  ✓ Created: iv_rv_spread_sq, iv_rv_spread_abs")
            
            # High spread + high VIX = mean reversion opportunity
            if 'vix_percentile' in df.columns:
                interactions['iv_rv_x_vix_pct'] = (
                    df['iv_rv_spread'] * df['vix_percentile']
                )
                print("  ✓ Created: iv_rv_x_vix_pct")
            
            # Spread + yield curve = directional bias
            if 'yield_slope' in df.columns:
                interactions['iv_rv_x_yield'] = (
                    df['iv_rv_spread'] * df['yield_slope']
                )
                print("  ✓ Created: iv_rv_x_yield")
        
        # ========================================
        # GROUP 2: Macro Regime Interactions
        # ========================================
        print("\n📈 Group 2: Macro Regime Interactions")
        
        # Yield curve + inflation = growth/recession signal
        if 'yield_slope' in df.columns and '10Y Breakeven Inflation_level' in df.columns:
            interactions['yield_x_inflation'] = (
                df['yield_slope'] * df['10Y Breakeven Inflation_level']
            )
            print("  ✓ Created: yield_x_inflation")
        
        # Steep curve + rising rates = Fed tightening
        if 'yield_slope' in df.columns and '10Y-2Y Yield Spread_change_63' in df.columns:
            interactions['yield_slope_momentum'] = (
                df['yield_slope'] * df['10Y-2Y Yield Spread_change_63']
            )
            print("  ✓ Created: yield_slope_momentum")
        
        # ========================================
        # GROUP 3: Vol Regime Interactions
        # ========================================
        print("\n💥 Group 3: Vol Regime Interactions")
        
        # VIX vs realized vol disconnect
        if 'vix' in df.columns and 'spx_realized_vol_63' in df.columns:
            interactions['vix_rv_ratio'] = (
                df['vix'] / (df['spx_realized_vol_63'] + 1e-6)
            )
            interactions['vix_rv_disconnect'] = (
                df['vix'] - df['spx_realized_vol_63']
            )
            print("  ✓ Created: vix_rv_ratio, vix_rv_disconnect")
        
        # VIX spike + falling vol = false alarm
        if 'vix_change_21' in df.columns and 'spx_realized_vol_21' in df.columns:
            interactions['vix_spike_x_rvol'] = (
                df['vix_change_21'] * (1 / (df['spx_realized_vol_21'] + 1e-6))
            )
            print("  ✓ Created: vix_spike_x_rvol")
        
        # ========================================
        # GROUP 4: Momentum Cross-Signals
        # ========================================
        print("\n🎯 Group 4: Momentum Cross-Signals")
        
        # SPX up + Gold down = risk-on
        if 'spx_ret_21' in df.columns and 'Gold_mom_21' in df.columns:
            interactions['risk_on_signal'] = (
                df['spx_ret_21'] - df['Gold_mom_21']
            )
            print("  ✓ Created: risk_on_signal")
        
        # SPX up + Dollar up = unusual (typically inverse)
        if 'spx_ret_21' in df.columns and 'Dollar_mom_21' in df.columns:
            interactions['spx_dollar_divergence'] = (
                df['spx_ret_21'] * df['Dollar_mom_21']
            )
            print("  ✓ Created: spx_dollar_divergence")
        
        # Strong momentum + low vol = trending market
        if 'spx_ret_63' in df.columns and 'spx_realized_vol_63' in df.columns:
            interactions['momentum_per_vol'] = (
                df['spx_ret_63'] / (df['spx_realized_vol_63'] + 1e-6)
            )
            print("  ✓ Created: momentum_per_vol")
        
        # ========================================
        # GROUP 5: Technical Regime Features
        # ========================================
        print("\n🔧 Group 5: Technical Regime Features")
        
        # Distance from MA + momentum = trend strength
        if 'spx_vs_ma200' in df.columns and 'spx_ret_63' in df.columns:
            interactions['trend_strength'] = (
                df['spx_vs_ma200'] * df['spx_ret_63']
            )
            print("  ✓ Created: trend_strength")
        
        # MA alignment = strong trend
        if all(col in df.columns for col in ['spx_vs_ma20', 'spx_vs_ma50', 'spx_vs_ma200']):
            interactions['ma_alignment'] = (
                (df['spx_vs_ma20'] + df['spx_vs_ma50'] + df['spx_vs_ma200']) / 3
            )
            print("  ✓ Created: ma_alignment")
        
        # Drop any NaN rows created by shifts/divisions
        interactions = interactions.dropna()
        
        print(f"\n✅ Generated {len(interactions.columns)} interaction features")
        print(f"   Samples: {len(interactions)} (after dropna)")
        
        return interactions
    
    # ========================================
    # TESTING FRAMEWORK
    # ========================================
    
    def test_features(self, 
                     base_features: pd.DataFrame,
                     new_features: pd.DataFrame,
                     spx: pd.Series,
                     target: str = 'direction_21d') -> Tuple[bool, Dict]:
        """
        A/B test: Do new features improve the model WITHOUT overfitting?
        
        Success criteria:
        1. Test accuracy improves by >0.5%
        2. Gap remains < 5% (ideally negative)
        3. At least one new feature shows >1% importance
        
        Args:
            base_features: Current production features
            new_features: New features to test
            spx: SPX series for creating targets
            target: Which target to use for testing
        
        Returns:
            (accept: bool, results: dict)
        """
        print("\n" + "="*70)
        print("FEATURE ADDITION TEST")
        print("="*70)
        
        # Align indices
        common_idx = base_features.index.intersection(new_features.index).intersection(spx.index)
        base_features = base_features.loc[common_idx]
        new_features = new_features.loc[common_idx]
        spx = spx.loc[common_idx]
        
        # Create target
        if target.startswith('direction_'):
            window = int(target.split('_')[1].replace('d', ''))
            fwd_return = spx.pct_change(window).shift(-window)
            y = (fwd_return > 0).astype(int)
        else:
            raise ValueError(f"Target {target} not supported yet")
        
        # Drop NaN targets
        valid_idx = y.dropna().index
        base_features = base_features.loc[valid_idx]
        new_features = new_features.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Train/test split
        split_idx = int(len(base_features) * (1 - TEST_SPLIT))
        
        # ========================================
        # BASELINE: Current features
        # ========================================
        print(f"\n1️⃣ BASELINE ({len(base_features.columns)} features)")
        
        X_train_base = base_features.iloc[:split_idx]
        X_test_base = base_features.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        model_base = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_base.fit(X_train_base, y_train)
        
        train_acc_base = model_base.score(X_train_base, y_train)
        test_acc_base = model_base.score(X_test_base, y_test)
        gap_base = train_acc_base - test_acc_base
        
        print(f"   Train Accuracy: {train_acc_base:.1%}")
        print(f"   Test Accuracy:  {test_acc_base:.1%}")
        print(f"   Gap:            {gap_base:+.1%}")
        
        self.baseline_results = {
            'train_acc': train_acc_base,
            'test_acc': test_acc_base,
            'gap': gap_base,
            'n_features': len(base_features.columns)
        }
        
        # ========================================
        # NEW: With new features
        # ========================================
        print(f"\n2️⃣ WITH NEW FEATURES (+{len(new_features.columns)} features)")
        
        combined = pd.concat([base_features, new_features], axis=1)
        X_train_new = combined.iloc[:split_idx]
        X_test_new = combined.iloc[split_idx:]
        
        model_new = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_new.fit(X_train_new, y_train)
        
        train_acc_new = model_new.score(X_train_new, y_train)
        test_acc_new = model_new.score(X_test_new, y_test)
        gap_new = train_acc_new - test_acc_new
        
        print(f"   Train Accuracy: {train_acc_new:.1%}")
        print(f"   Test Accuracy:  {test_acc_new:.1%}")
        print(f"   Gap:            {gap_new:+.1%}")
        
        # ========================================
        # NEW FEATURE IMPORTANCE
        # ========================================
        print(f"\n3️⃣ NEW FEATURE IMPORTANCE (Top 5)")
        
        importance = pd.Series(
            model_new.feature_importances_,
            index=combined.columns
        )
        
        new_feature_imp = importance[new_features.columns].sort_values(ascending=False)
        
        for i, (feat, imp) in enumerate(new_feature_imp.head(5).items(), 1):
            print(f"   {i}. {feat:35s}: {imp*100:5.2f}%")
        
        max_new_importance = new_feature_imp.max() * 100 if len(new_feature_imp) > 0 else 0
        
        self.test_results = {
            'train_acc': train_acc_new,
            'test_acc': test_acc_new,
            'gap': gap_new,
            'n_features': len(combined.columns),
            'new_feature_importance': new_feature_imp.to_dict(),
            'max_new_importance': max_new_importance
        }
        
        # ========================================
        # DECISION
        # ========================================
        print("\n4️⃣ VERDICT")
        
        improvement = test_acc_new - test_acc_base
        gap_increase = abs(gap_new) - abs(gap_base)
        
        print(f"   Test Accuracy Δ:  {improvement:+.1%}")
        print(f"   Gap Δ:            {gap_increase:+.1%}")
        print(f"   Max New Feature:  {max_new_importance:.2f}%")
        
        # Decision logic
        if improvement > 0.005 and gap_increase < 0.03 and max_new_importance > 1.0:
            print(f"\n   ✅ ACCEPT: Meaningful improvement, controlled gap, relevant features")
            accept = True
        elif improvement > 0.002 and gap_increase < 0.05:
            print(f"\n   ⚠️ MARGINAL: Slight improvement, but watch carefully")
            accept = False
        else:
            print(f"\n   ❌ REJECT: No improvement or overfitting risk")
            accept = False
        
        return accept, {
            'baseline': self.baseline_results,
            'with_new': self.test_results,
            'improvement': improvement,
            'gap_increase': gap_increase
        }
    
    def walk_forward_validation(self,
                               features: pd.DataFrame,
                               spx: pd.Series,
                               n_splits: int = 5,
                               target: str = 'direction_21d') -> pd.DataFrame:
        """
        Rolling window validation: Does accuracy hold as time progresses?
        
        Args:
            features: Feature matrix to test
            spx: SPX series for targets
            n_splits: Number of time-based splits
            target: Prediction target
        
        Returns:
            DataFrame with per-split results
        """
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION")
        print("="*70)
        
        # Align
        common_idx = features.index.intersection(spx.index)
        features = features.loc[common_idx]
        spx = spx.loc[common_idx]
        
        # Create target
        if target.startswith('direction_'):
            window = int(target.split('_')[1].replace('d', ''))
            fwd_return = spx.pct_change(window).shift(-window)
            y = (fwd_return > 0).astype(int).dropna()
        else:
            raise ValueError(f"Target {target} not supported")
        
        # Align again after target creation
        valid_idx = features.index.intersection(y.index)
        features = features.loc[valid_idx]
        y = y.loc[valid_idx]
        
        n_samples = len(features)
        test_size = n_samples // (n_splits + 1)
        
        results = []
        
        for i in range(n_splits):
            # Expanding window: train on all data up to test period
            train_end = test_size * (i + 2)
            test_start = test_size * (i + 1)
            test_end = test_size * (i + 2)
            
            # Ensure we don't go out of bounds
            if test_end > n_samples:
                test_end = n_samples
            
            # Train/test indices
            X_train = features.iloc[:train_end]
            X_test = features.iloc[test_start:test_end]
            y_train = y.iloc[:train_end]
            y_test = y.iloc[test_start:test_end]
            
            train_dates = features.index[:train_end]
            test_dates = features.index[test_start:test_end]
            
            print(f"\n📅 Split {i+1}/{n_splits}")
            print(f"   Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
            print(f"   Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_split=50,
                min_samples_leaf=30,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            gap = train_acc - test_acc
            
            # Predictions for analysis
            y_pred = model.predict(X_test)
            up_predictions = (y_pred == 1).sum()
            down_predictions = (y_pred == 0).sum()
            
            results.append({
                'split': i+1,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap,
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'n_test': len(test_dates),
                'up_preds': up_predictions,
                'down_preds': down_predictions
            })
            
            print(f"   Train Acc:  {train_acc:.1%}")
            print(f"   Test Acc:   {test_acc:.1%}")
            print(f"   Gap:        {gap:+.1%}")
            print(f"   Predictions: {up_predictions} UP, {down_predictions} DOWN")
        
        # ========================================
        # SUMMARY STATISTICS
        # ========================================
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        mean_test = df_results['test_acc'].mean()
        std_test = df_results['test_acc'].std()
        mean_gap = df_results['gap'].mean()
        worst = df_results['test_acc'].min()
        best = df_results['test_acc'].max()
        
        print(f"Test Accuracy:  {mean_test:.1%} ± {std_test:.1%}")
        print(f"Average Gap:    {mean_gap:+.1%}")
        print(f"Worst Split:    {worst:.1%}")
        print(f"Best Split:     {best:.1%}")
        print(f"Range:          {best - worst:.1%}")
        
        # ========================================
        # STABILITY ASSESSMENT
        # ========================================
        print("\n" + "="*70)
        print("STABILITY ASSESSMENT")
        print("="*70)
        
        if std_test < 0.05:
            print("✅ STABLE: Model is consistent across time periods")
            stability = "STABLE"
        elif std_test < 0.10:
            print("⚠️ MODERATE: Model shows some variation across time")
            stability = "MODERATE"
        else:
            print("❌ UNSTABLE: High variance across splits - use with caution")
            stability = "UNSTABLE"
        
        df_results['stability'] = stability
        
        return df_results
    
    # ========================================
    # HIGH-LEVEL API (Single Command)
    # ========================================
    
    def run_complete_analysis(self, 
                             years: int = 7,
                             target: str = 'direction_21d',
                             test_vol: bool = True,
                             test_interactions: bool = True) -> Dict:
        """
        🚀 ONE-COMMAND ANALYSIS
        
        Tests both vol features AND interactions.
        Handles all data fetching and model training internally.
        
        Usage from Jupyter:
            explorer = SPXFeatureExplorer()
            results = explorer.run_complete_analysis(years=7)
        
        Args:
            years: Years of data to use
            target: Prediction target (default: direction_21d)
            test_vol: Test legitimate vol features?
            test_interactions: Test interaction features?
        
        Returns:
            Dict with all results
        """
        print("\n" + "🎯"*35)
        print("COMPLETE FEATURE ANALYSIS PIPELINE")
        print("🎯"*35)
        
        # Step 1: Fetch data
        spx, vix = self._fetch_data(years)
        
        # Step 2: Train baseline
        predictor = self._train_baseline(years)
        
        results = {}
        
        # ========================================
        # TEST 1: Legitimate Vol Features
        # ========================================
        if test_vol:
            print("\n" + "="*70)
            print("TEST 1: LEGITIMATE VOL FEATURES (VIX vs Past RV)")
            print("="*70)
            
            # Generate vol features
            vol_features = self.generate_vix_vs_rv_features(spx, vix)
            
            # Align with base features
            vol_features = vol_features.reindex(predictor.features_scaled.index)
            
            # Scale vol features
            scaler = StandardScaler()
            vol_features_scaled = pd.DataFrame(
                scaler.fit_transform(vol_features.fillna(0)),
                index=vol_features.index,
                columns=vol_features.columns
            )
            
            # Test
            accept_vol, test_res_vol = self.test_features(
                predictor.features_scaled,
                vol_features_scaled,
                spx,
                target
            )
            
            results['vol_features'] = {
                'accept': accept_vol,
                'test_results': test_res_vol,
                'features': vol_features_scaled
            }
            
            # Walk-forward if promising
            if accept_vol or test_res_vol['improvement'] > 0:
                print("\n↪️ Running walk-forward validation...")
                combined = pd.concat([predictor.features_scaled, vol_features_scaled], axis=1)
                stability = self.walk_forward_validation(combined, spx, 5, target)
                results['vol_features']['stability'] = stability
        
        # ========================================
        # TEST 2: Interaction Features
        # ========================================
        if test_interactions:
            print("\n" + "="*70)
            print("TEST 2: INTERACTION FEATURES")
            print("="*70)
            
            # Generate interactions
            interaction_features = self.generate_interactions(predictor.features_scaled)
            
            # Test
            accept_int, test_res_int = self.test_features(
                predictor.features_scaled,
                interaction_features,
                spx,
                target
            )
            
            results['interactions'] = {
                'accept': accept_int,
                'test_results': test_res_int,
                'features': interaction_features
            }
            
            # Walk-forward if promising
            if accept_int or test_res_int['improvement'] > 0:
                print("\n↪️ Running walk-forward validation...")
                combined = pd.concat([predictor.features_scaled, interaction_features], axis=1)
                stability = self.walk_forward_validation(combined, spx, 5, target)
                results['interactions']['stability'] = stability
        
        # ========================================
        # FINAL RECOMMENDATIONS
        # ========================================
        print("\n" + "="*70)
        print("💡 FINAL RECOMMENDATIONS")
        print("="*70)
        
        if test_vol and 'vol_features' in results:
            vol_improvement = results['vol_features']['test_results']['improvement']
            vol_accept = results['vol_features']['accept']
            
            print(f"\n📊 VOL FEATURES:")
            print(f"   Improvement: {vol_improvement:+.1%}")
            print(f"   Decision: {'✅ ACCEPT' if vol_accept else '❌ REJECT'}")
            
            if vol_accept:
                print("\n   → Add vol features to spx_features.py")
                print(f"   → Expected 21d accuracy: ~{results['vol_features']['test_results']['with_new']['test_acc']:.1%}")
        
        if test_interactions and 'interactions' in results:
            int_improvement = results['interactions']['test_results']['improvement']
            int_accept = results['interactions']['accept']
            
            print(f"\n🔗 INTERACTION FEATURES:")
            print(f"   Improvement: {int_improvement:+.1%}")
            print(f"   Decision: {'✅ ACCEPT' if int_accept else '❌ REJECT'}")
            
            if int_accept:
                print("\n   → Add interaction features to spx_features.py")
                print(f"   → Expected 21d accuracy: ~{results['interactions']['test_results']['with_new']['test_acc']:.1%}")
        
        # Best combo?
        if test_vol and test_interactions and 'vol_features' in results and 'interactions' in results:
            if vol_accept and int_accept:
                print(f"\n🚀 BOTH ACCEPTED - Test combo:")
                print(f"   → Combine both feature sets")
                print(f"   → Re-run test to check for interaction effects")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        return results
    
    def run_vol_feature_test(self, years: int = 7, target: str = 'direction_21d') -> Dict:
        """
        🎯 Test ONLY legitimate vol features.
        
        Usage:
            explorer = SPXFeatureExplorer()
            results = explorer.run_vol_feature_test(years=7)
        """
        return self.run_complete_analysis(
            years=years,
            target=target,
            test_vol=True,
            test_interactions=False
        )
    
    def run_interaction_test(self, years: int = 7, target: str = 'direction_21d') -> Dict:
        """
        🎯 Test ONLY interaction features.
        
        Usage:
            explorer = SPXFeatureExplorer()
            results = explorer.run_interaction_test(years=7)
        """
        return self.run_complete_analysis(
            years=years,
            target=target,
            test_vol=False,
            test_interactions=True
        )
    
    # ========================================
    # BACKWARDS COMPATIBLE API
    # ========================================
    
    def run_full_analysis(self,
                         base_features: pd.DataFrame,
                         spx: pd.Series,
                         target: str = 'direction_21d') -> Dict:
        """
        LEGACY METHOD: For manual orchestration (backward compatible)
        
        Use run_complete_analysis() instead for automatic orchestration.
        
        Args:
            base_features: Production feature matrix
            spx: SPX series
            target: Prediction target
        
        Returns:
            Dict with all results
        """
        print("\n" + "="*70)
        print("FULL FEATURE DISCOVERY PIPELINE (LEGACY)")
        print("="*70)
        print("⚠️  Note: Consider using run_complete_analysis() for auto-orchestration")
        
        # Phase 1: Generate interactions
        new_features = self.generate_interactions(base_features)
        
        # Phase 2: Test
        accept, test_results = self.test_features(
            base_features, 
            new_features, 
            spx, 
            target
        )
        
        # Phase 3: Validate (only if accepted or marginal)
        if accept or test_results['improvement'] > 0:
            print("\n↪️ Proceeding to walk-forward validation...")
            combined = pd.concat([base_features, new_features], axis=1)
            stability_results = self.walk_forward_validation(
                combined, 
                spx, 
                n_splits=5, 
                target=target
            )
        else:
            print("\n⏸️ Skipping walk-forward (features rejected)")
            stability_results = None
        
        return {
            'new_features': new_features,
            'test_results': test_results,
            'stability_results': stability_results,
            'accept': accept
        }


# ========================================
# EXAMPLE USAGE & DOCUMENTATION
# ========================================

def main():
    """
    Quick start guide for SPXFeatureExplorer
    """
    print("\n" + "="*70)
    print("SPX FEATURE EXPLORER - Quick Start")
    print("="*70)
    print("""
🚀 RECOMMENDED: Single Command Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

From Jupyter/IPython:

    from spx_feature_explorer import SPXFeatureExplorer
    
    explorer = SPXFeatureExplorer()
    results = explorer.run_complete_analysis(years=7)
    
That's it! It will:
  ✓ Fetch SPX/VIX data
  ✓ Train baseline model
  ✓ Test vol features (VIX vs Past RV)
  ✓ Test interaction features
  ✓ Run walk-forward validation
  ✓ Give you clear recommendations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


📊 Test ONLY Vol Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    explorer = SPXFeatureExplorer()
    results = explorer.run_vol_feature_test(years=7)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🔗 Test ONLY Interactions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    explorer = SPXFeatureExplorer()
    results = explorer.run_interaction_test(years=7)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


📈 Check Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Vol features
    if 'vol_features' in results:
        print(results['vol_features']['test_results'])
        
        # Top vol features
        importance = results['vol_features']['test_results']['with_new']['new_feature_importance']
        top_5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\\nTop 5 Vol Features:")
        for feat, imp in top_5:
            print(f"  {feat}: {imp*100:.2f}%")
    
    # Interactions
    if 'interactions' in results:
        print(results['interactions']['test_results'])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


⏱️  Expected Runtime:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Data fetch:            ~10 seconds
  Baseline training:     ~30 seconds
  Vol feature test:      ~20 seconds
  Interaction test:      ~20 seconds
  Walk-forward (each):   ~60 seconds
  
  TOTAL: ~3-4 minutes for complete analysis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


💡 What You'll Get:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✓ Clear ACCEPT/REJECT decision for each feature set
  ✓ Test accuracy improvement (baseline vs new)
  ✓ Feature importance rankings
  ✓ Walk-forward stability metrics
  ✓ Actionable recommendations for production

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🔧 Advanced: Manual Orchestration (Legacy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you prefer manual control:

    from spx_predictor import SPXPredictor
    from spx_feature_explorer import SPXFeatureExplorer
    from UnifiedDataFetcher import UnifiedDataFetcher
    from datetime import datetime, timedelta
    
    # Manual setup
    predictor = SPXPredictor()
    predictor.train(years=7)
    
    fetcher = UnifiedDataFetcher()
    end = datetime.now()
    start = end - timedelta(days=7*365)
    
    spx_df = fetcher.fetch_spx(start.strftime('%Y-%m-%d'), 
                               end.strftime('%Y-%m-%d'))
    spx = spx_df['Close'].squeeze()
    spx.index = spx.index.tz_localize(None)
    
    # Test
    explorer = SPXFeatureExplorer()
    results = explorer.run_full_analysis(
        base_features=predictor.features_scaled,
        spx=spx,
        target='direction_21d'
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run: python -c "from spx_feature_explorer import main; main()"
    """)


if __name__ == "__main__":
    main()