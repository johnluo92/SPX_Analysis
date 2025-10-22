"""
SPX Feature Explorer - EXPERIMENTAL FEATURES
Test new feature interactions WITHOUT touching production code

Usage:
    explorer = SPXFeatureExplorer()
    
    # Load your existing trained predictor
    predictor = SPXPredictor()
    predictor.train(years=7)
    
    # Generate interaction features
    new_features = explorer.generate_interactions(predictor.features_scaled)
    
    # Test if they improve performance
    accept, results = explorer.test_features(
        predictor.features_scaled, 
        new_features, 
        predictor.model
    )
    
    # Walk-forward validation for stability
    stability = explorer.walk_forward_validation(
        predictor.features_scaled,
        new_features,
        spx_series
    )
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE, TEST_SPLIT


class SPXFeatureExplorer:
    """
    Experimental feature discovery WITHOUT touching production code.
    
    Philosophy:
    - Generate interaction features
    - Test rigorously (A/B test vs baseline)
    - Validate stability (walk-forward)
    - Only promote to production if proven
    """
    
    def __init__(self):
        self.baseline_results = None
        self.test_results = None
    
    # ========================================
    # PHASE 1: FEATURE GENERATION
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
            
            # Spread acceleration = regime shift
            # DISABLED: Creates 21-day data loss, misaligns train/test split
            # if 'iv_rv_momentum_21' in df.columns:
            #     interactions['iv_rv_acceleration'] = (
            #         df['iv_rv_momentum_21'] - 
            #         df['iv_rv_momentum_21'].shift(21)
            #     )
            #     print("  ✓ Created: iv_rv_acceleration")
        
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
        print("\n📐 Group 5: Technical Regime Features")
        
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
    # PHASE 2: RIGOROUS TESTING
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
            new_features: New interaction features to test
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
        # NEW: With interaction features
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
            print(f"\n   ⚠️  MARGINAL: Slight improvement, but watch carefully")
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
    
    # ========================================
    # PHASE 3: WALK-FORWARD VALIDATION
    # ========================================
    
    def walk_forward_validation(self,
                               features: pd.DataFrame,
                               spx: pd.Series,
                               n_splits: int = 5,
                               target: str = 'direction_21d') -> pd.DataFrame:
        """
        Rolling window validation: Does accuracy hold as time progresses?
        
        If accuracy varies wildly (70% one split, 95% another),
        your signal is unstable. If it's consistently 85-92%, you're golden.
        
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
            print("⚠️  MODERATE: Model shows some variation across time")
            stability = "MODERATE"
        else:
            print("❌ UNSTABLE: High variance across splits - use with caution")
            stability = "UNSTABLE"
        
        df_results['stability'] = stability
        
        return df_results
    
    # ========================================
    # CONVENIENCE METHODS
    # ========================================
    
    def run_full_analysis(self,
                         base_features: pd.DataFrame,
                         spx: pd.Series,
                         target: str = 'direction_21d') -> Dict:
        """
        Run complete feature discovery pipeline:
        1. Generate interactions
        2. A/B test vs baseline
        3. Walk-forward validation (if accepted)
        
        Args:
            base_features: Production feature matrix
            spx: SPX series
            target: Prediction target
        
        Returns:
            Dict with all results
        """
        print("\n" + "="*70)
        print("FULL FEATURE DISCOVERY PIPELINE")
        print("="*70)
        
        # Phase 1: Generate
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
            print("\n⏩ Proceeding to walk-forward validation...")
            combined = pd.concat([base_features, new_features], axis=1)
            stability_results = self.walk_forward_validation(
                combined, 
                spx, 
                n_splits=5, 
                target=target
            )
        else:
            print("\n⏸️  Skipping walk-forward (features rejected)")
            stability_results = None
        
        return {
            'new_features': new_features,
            'test_results': test_results,
            'stability_results': stability_results,
            'accept': accept
        }


def main():
    """
    Example usage - run after training your SPXPredictor
    """
    print("\n" + "="*70)
    print("SPX FEATURE EXPLORER - Example Usage")
    print("="*70)
    print("""
This is a standalone exploration tool. Use it like this:

    from spx_predictor import SPXPredictor
    from spx_feature_explorer import SPXFeatureExplorer
    
    # 1. Train your baseline model
    predictor = SPXPredictor()
    predictor.train(years=7)
    
    # 2. Explore new features
    explorer = SPXFeatureExplorer()
    results = explorer.run_full_analysis(
        base_features=predictor.features_scaled,
        spx=predictor.features.index.to_series(),  # Need actual SPX
        target='direction_21d'
    )
    
    # 3. Decide whether to promote to production
    if results['accept']:
        print("✅ Features are ready for production!")
        # Manually add to spx_features.py
    else:
        print("❌ Keep experimenting")
    
Run this file to see this message.
    """)


if __name__ == "__main__":
    main()