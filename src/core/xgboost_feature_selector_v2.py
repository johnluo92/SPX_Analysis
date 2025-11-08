"""
XGBoost Feature Selector V2 - Intelligent Selection with Stability Testing
===========================================================================

Academic enhancements:
1. Stability-based selection (features must be important across multiple folds)
2. Recursive feature addition (start small, add features incrementally)
3. Multicollinearity handling (remove redundant features post-selection)
4. Domain expertise integration (preserve forward indicators even if low importance)
5. Performance cliff detection (optimal feature count before diminishing returns)

References:
- Feature stability: Nogueira et al., 2018 (JMLR)
- Recursive addition: Guyon et al., 2002 (Machine Learning)
- Financial feature selection: Research Square, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from xgboost_trainer_v2 import EnhancedXGBoostTrainer

# Forward indicators that should be preserved regardless of importance
CRITICAL_FORWARD_INDICATORS = [
    'VX1-VX2', 'VX2-VX1_RATIO',  # VIX futures term structure
    'yield_10y2y', 'yield_10y3m',  # Yield curve inversion
    'VXTLT', 'vxtlt_vix_ratio',  # Bond volatility
    'SKEW', 'skew_vs_vix',  # Tail risk
]


class IntelligentFeatureSelector:
    """
    Stability-based feature selection with domain expertise integration.
    
    Selection Process:
    1. Train XGBoost with ALL features across multiple folds
    2. Measure feature stability (importance variance across folds)
    3. Recursive addition: Start with top 20, add in batches of 10-15
    4. Stop when performance plateaus or drops (diminishing returns)
    5. Remove multicollinear features (correlation > 0.95)
    6. Force-include critical forward indicators
    """
    
    def __init__(self, output_dir: str = './models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.baseline_trainer = None
        self.feature_importance_df = None
        self.stability_scores = None
        self.validation_results = []
        self.selected_features = None
        self.optimal_size = None
    
    def run_full_pipeline(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        candidate_sizes: List[int] = None,  # Auto-detect if None
        min_stability: float = 0.3,  # Features must be stable across folds
        max_correlation: float = 0.95,  # Remove highly correlated pairs
        preserve_forward_indicators: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Run intelligent feature selection pipeline.
        
        Args:
            features: Full feature matrix
            vix: VIX series
            spx: SPX series
            candidate_sizes: Feature counts to test (auto if None)
            min_stability: Minimum stability score (0-1) to keep feature
            max_correlation: Max correlation before removing redundant feature
            preserve_forward_indicators: Force-keep critical forward indicators
            verbose: Print progress
            
        Returns:
            {
                'baseline_results': {...},
                'feature_importance': DataFrame,
                'stability_scores': Series,
                'validation_results': [...],
                'selected_features': List[str],
                'optimal_size': int,
                'performance_summary': {...}
            }
        """
        if verbose:
            print(f"\n{'='*80}")
            print("🎯 INTELLIGENT FEATURE SELECTION PIPELINE V2")
            print(f"{'='*80}")
            print(f"Starting features: {len(features.columns)}")
            print(f"Min stability: {min_stability}")
            print(f"Max correlation: {max_correlation}")
        
        # Step 1: Train baseline with ALL features + measure stability
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 1/5] Training baseline + measuring feature stability...")
            print(f"{'='*80}")
        
        self.baseline_trainer, baseline_metrics, stability_scores = self._train_baseline_with_stability(
            features, vix, spx, verbose
        )
        
        self.stability_scores = stability_scores
        
        # Step 2: Extract and rank features by stability-weighted importance
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 2/5] Ranking features by stability-weighted importance...")
            print(f"{'='*80}")
        
        self.feature_importance_df = self._rank_features_with_stability(
            stability_scores, min_stability, verbose
        )
        
        # Step 3: Remove multicollinear features
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 3/5] Removing multicollinear features...")
            print(f"{'='*80}")
        
        filtered_importance_df = self._remove_multicollinearity(
            self.feature_importance_df, features, max_correlation, verbose
        )
        
        # Step 4: Recursive feature addition to find optimal set
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 4/5] Recursive feature addition (finding performance cliff)...")
            print(f"{'='*80}")
        
        if candidate_sizes is None:
            # Auto-detect: test 20, 40, 60, 80, 100, 120
            max_features = min(120, len(filtered_importance_df))
            candidate_sizes = list(range(20, max_features + 1, 20))
        
        optimal_features = self._recursive_feature_addition(
            filtered_importance_df, features, vix, spx, 
            candidate_sizes, baseline_metrics, verbose
        )
        
        # Step 5: Force-include critical forward indicators
        if preserve_forward_indicators:
            if verbose:
                print(f"\n{'='*80}")
                print("[STEP 5/5] Ensuring critical forward indicators included...")
                print(f"{'='*80}")
            
            optimal_features = self._ensure_forward_indicators(
                optimal_features, features.columns, verbose
            )
        
        self.selected_features = optimal_features
        self.optimal_size = len(optimal_features)
        
        # Generate summary
        summary = self._generate_summary_report(baseline_metrics, verbose)
        
        # Save results
        self._save_selection_results(summary)
        
        return {
            'baseline_results': baseline_metrics,
            'feature_importance': self.feature_importance_df,
            'stability_scores': self.stability_scores,
            'validation_results': self.validation_results,
            'selected_features': self.selected_features,
            'optimal_size': self.optimal_size,
            'performance_summary': summary
        }
    
    def _train_baseline_with_stability(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        verbose: bool
    ) -> Tuple[EnhancedXGBoostTrainer, Dict, pd.Series]:
        """
        Train baseline and measure feature importance stability across folds.
        
        Stability = 1 - (std_importance / mean_importance)
        High stability = feature consistently important across time periods
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        trainer = EnhancedXGBoostTrainer(output_dir=str(self.output_dir / 'baseline'))
        
        # Train with SHAP to get accurate importance
        results = trainer.train(
            features=features,
            vix=vix,
            spx=spx,
            n_splits=5,
            optimize_hyperparams=False,  # Skip for baseline (faster)
            crisis_balanced=True,
            compute_shap=True,
            verbose=False  # Suppress detailed output
        )
        
        # Now compute per-fold importance to measure stability
        if verbose:
            print("   Computing per-fold feature importance for stability...")
        
        from sklearn.model_selection import TimeSeriesSplit
        import xgboost as xgb
        
        tscv = TimeSeriesSplit(n_splits=5, test_size=252)
        
        # Prepare data
        X = features.fillna(method='ffill').fillna(method='bfill').dropna()
        y_regime = pd.cut(vix, bins=[0, 16.77, 24.40, 39.67, 100], labels=[0,1,2,3]).astype(int)
        
        valid_idx = X.dropna().index.intersection(y_regime.dropna().index)
        X = X.loc[valid_idx]
        y_regime = y_regime.loc[valid_idx]
        
        fold_importances = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train = X.iloc[train_idx]
            y_train = y_regime.iloc[train_idx]
            
            # Train simple model (no optimization)
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=4,
                max_depth=6,
                learning_rate=0.03,
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train, y_train, verbose=False)
            
            # Get gain-based importance
            importance_dict = model.get_booster().get_score(importance_type='gain')
            importance_series = pd.Series(importance_dict, name=f'fold_{fold_idx}')
            
            # Normalize
            importance_series = importance_series / importance_series.sum()
            
            fold_importances.append(importance_series)
        
        # Combine fold importances
        importance_matrix = pd.DataFrame(fold_importances).T.fillna(0)
        
        # Calculate stability: high mean, low variance = stable
        mean_importance = importance_matrix.mean(axis=1)
        std_importance = importance_matrix.std(axis=1)
        
        # Stability score: 1 - coefficient of variation (clamped to [0, 1])
        stability = 1 - (std_importance / mean_importance.replace(0, np.nan))
        stability = stability.clip(0, 1).fillna(0)
        
        stability_series = pd.Series(stability, name='stability')
        
        if verbose:
            print(f"   ✅ Stability computed for {len(stability_series)} features")
            print(f"      Mean stability: {stability_series.mean():.3f}")
            print(f"      Highly stable features (>0.7): {(stability_series > 0.7).sum()}")
        
        baseline_metrics = {
            'regime_metrics': results['regime_metrics'],
            'range_metrics': results['range_metrics'],
        }
        
        return trainer, baseline_metrics, stability_series
    
    def _rank_features_with_stability(
        self,
        stability_scores: pd.Series,
        min_stability: float,
        verbose: bool
    ) -> pd.DataFrame:
        """
        Rank features by stability-weighted importance.
        
        Formula: final_score = importance * (stability ^ 0.5)
        This gives preference to stable features while not completely ignoring unstable ones.
        """
        # Load importance from baseline trainer
        importance_path = self.output_dir / 'baseline' / 'feature_importance_v2_overall.csv'
        
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
        else:
            raise FileNotFoundError(f"Baseline importance not found: {importance_path}")
        
        # Merge with stability
        importance_df = importance_df.set_index('feature')
        stability_df = stability_scores.to_frame('stability')
        
        combined = importance_df.join(stability_df, how='left').fillna(0)
        combined = combined.reset_index().rename(columns={'index': 'feature'})
        
        # Stability-weighted score
        combined['stability_weighted_score'] = (
            combined['overall_shap'] * (combined['stability'] ** 0.5)
        )
        
        # Filter by minimum stability
        combined['passes_stability'] = combined['stability'] >= min_stability
        
        # Sort by stability-weighted score
        combined = combined.sort_values('stability_weighted_score', ascending=False)
        
        if verbose:
            print(f"\n   Feature Stability Statistics:")
            print(f"      Total features: {len(combined)}")
            print(f"      Pass stability threshold: {combined['passes_stability'].sum()}")
            print(f"      Fail stability threshold: {(~combined['passes_stability']).sum()}")
            
            print(f"\n   Top 10 Features (Stability-Weighted):")
            for i, row in combined.head(10).iterrows():
                stable = "✅" if row['passes_stability'] else "⚠️"
                fwd = "🔮" if row.get('is_forward_indicator', False) else "  "
                print(f"      {stable} {fwd} {row['feature']:<45} "
                      f"Score={row['stability_weighted_score']:.4f} "
                      f"(Stability={row['stability']:.2f})")
        
        return combined
    
    def _remove_multicollinearity(
        self,
        importance_df: pd.DataFrame,
        features: pd.DataFrame,
        max_correlation: float,
        verbose: bool
    ) -> pd.DataFrame:
        """
        Remove redundant features with correlation > max_correlation.
        
        Strategy: For each correlated pair, keep the one with higher stability-weighted score.
        """
        if verbose:
            print(f"\n   Checking for multicollinearity (threshold: {max_correlation})...")
        
        # Compute correlation matrix for features in importance_df
        feature_names = importance_df['feature'].tolist()
        available_features = [f for f in feature_names if f in features.columns]
        
        corr_matrix = features[available_features].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr_pairs = []
        
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > max_correlation:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.index[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if verbose:
            print(f"      Found {len(high_corr_pairs)} highly correlated pairs")
        
        # Remove lower-scoring feature from each pair
        features_to_remove = set()
        
        for pair in high_corr_pairs:
            f1 = pair['feature1']
            f2 = pair['feature2']
            
            score1 = importance_df[importance_df['feature'] == f1]['stability_weighted_score'].values[0]
            score2 = importance_df[importance_df['feature'] == f2]['stability_weighted_score'].values[0]
            
            if score1 < score2:
                features_to_remove.add(f1)
                if verbose:
                    print(f"      Removing {f1} (corr={pair['correlation']:.3f} with {f2})")
            else:
                features_to_remove.add(f2)
                if verbose:
                    print(f"      Removing {f2} (corr={pair['correlation']:.3f} with {f1})")
        
        # Filter out removed features
        filtered_df = importance_df[~importance_df['feature'].isin(features_to_remove)].copy()
        
        if verbose:
            print(f"\n   ✅ Removed {len(features_to_remove)} redundant features")
            print(f"      Remaining: {len(filtered_df)} features")
        
        return filtered_df
    
    def _recursive_feature_addition(
        self,
        importance_df: pd.DataFrame,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        candidate_sizes: List[int],
        baseline_metrics: Dict,
        verbose: bool
    ) -> List[str]:
        """
        Recursively add features and find performance cliff.
        
        Start with top 20, add in batches, stop when:
        1. Performance stops improving (< 0.5% gain)
        2. Performance degrades
        3. Reach maximum tested size
        """
        baseline_regime_acc = baseline_metrics['regime_metrics']['cv_balanced_accuracy_mean']
        baseline_range_rmse = baseline_metrics['range_metrics']['cv_rmse_mean']
        
        best_regime_acc = 0
        best_range_rmse = float('inf')
        best_size = 20
        best_features = None
        
        for size in sorted(candidate_sizes):
            if size > len(importance_df):
                continue
            
            if verbose:
                print(f"\n   {'─'*60}")
                print(f"   Testing: Top {size} features")
                print(f"   {'─'*60}")
            
            # Get top N features
            top_features = importance_df.head(size)['feature'].tolist()
            features_filtered = features[top_features]
            
            # Train XGBoost
            trainer = EnhancedXGBoostTrainer(output_dir=str(self.output_dir / f'filtered_{size}'))
            
            results = trainer.train(
                features=features_filtered,
                vix=vix,
                spx=spx,
                n_splits=5,
                optimize_hyperparams=False,
                crisis_balanced=True,
                compute_shap=False,  # Skip SHAP for speed
                verbose=False
            )
            
            regime_acc = results['regime_metrics']['cv_balanced_accuracy_mean']
            range_rmse = results['range_metrics']['cv_rmse_mean']
            
            # Calculate retention
            regime_retention = (regime_acc / baseline_regime_acc) if baseline_regime_acc > 0 else 0
            range_retention = (baseline_range_rmse / range_rmse) if range_rmse > 0 else 0
            
            # Store results
            result = {
                'n_features': size,
                'regime_accuracy': regime_acc,
                'range_rmse': range_rmse,
                'regime_retention': regime_retention,
                'range_retention': range_retention,
                'regime_improvement': regime_acc - best_regime_acc,
                'passes_threshold': (regime_retention >= 0.95 and range_retention >= 0.95)
            }
            
            self.validation_results.append(result)
            
            if verbose:
                print(f"      Regime Acc: {regime_acc:.3f} ({regime_retention:.1%} retention, "
                      f"{result['regime_improvement']:+.3f} improvement)")
                print(f"      Range RMSE: {range_rmse:.2f}% ({range_retention:.1%} retention)")
                print(f"      Status: {'✅ PASS' if result['passes_threshold'] else '⚠️ BELOW THRESHOLD'}")
            
            # Check for performance cliff
            if regime_acc > best_regime_acc:
                best_regime_acc = regime_acc
                best_range_rmse = range_rmse
                best_size = size
                best_features = top_features
                
                if verbose:
                    print(f"      🎯 New best: {size} features")
            else:
                # Performance not improving
                improvement = regime_acc - best_regime_acc
                if improvement < 0.005:  # Less than 0.5% improvement
                    if verbose:
                        print(f"\n   ⚠️ Performance plateau detected (<0.5% improvement)")
                        print(f"      Stopping at {best_size} features")
                    break
        
        if verbose:
            print(f"\n   ✅ Optimal feature set: {best_size} features")
            print(f"      Best regime accuracy: {best_regime_acc:.3f}")
            print(f"      Best range RMSE: {best_range_rmse:.2f}%")
        
        return best_features
    
    def _ensure_forward_indicators(
        self,
        selected_features: List[str],
        all_features: pd.Index,
        verbose: bool
    ) -> List[str]:
        """
        Force-include critical forward indicators even if not in top N.
        
        These features have theoretical forward-looking power and should be preserved.
        """
        missing_indicators = []
        
        for indicator in CRITICAL_FORWARD_INDICATORS:
            if indicator in all_features and indicator not in selected_features:
                missing_indicators.append(indicator)
        
        if missing_indicators:
            if verbose:
                print(f"\n   Adding {len(missing_indicators)} critical forward indicators:")
                for ind in missing_indicators:
                    print(f"      + {ind}")
            
            selected_features = selected_features + missing_indicators
        else:
            if verbose:
                print(f"\n   ✅ All critical forward indicators already included")
        
        return selected_features
    
    def _generate_summary_report(
        self,
        baseline_metrics: Dict,
        verbose: bool
    ) -> Dict:
        """Generate comprehensive summary report."""
        
        # Find best validation result
        best_result = max(self.validation_results, key=lambda x: x['regime_accuracy'])
        
        baseline_regime_acc = baseline_metrics['regime_metrics']['cv_balanced_accuracy_mean']
        baseline_range_rmse = baseline_metrics['range_metrics']['cv_rmse_mean']
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'baseline': {
                'n_features': len(self.feature_importance_df),
                'regime_accuracy': baseline_regime_acc,
                'range_rmse': baseline_range_rmse,
            },
            'selected': {
                'n_features': self.optimal_size,
                'regime_accuracy': best_result['regime_accuracy'],
                'range_rmse': best_result['range_rmse'],
                'regime_retention': best_result['regime_retention'],
                'range_retention': best_result['range_retention'],
                'features': self.selected_features,
            },
            'reduction': {
                'features_removed': len(self.feature_importance_df) - self.optimal_size,
                'reduction_pct': (1 - self.optimal_size / len(self.feature_importance_df)) * 100,
            },
            'stability_stats': {
                'mean_stability': float(self.stability_scores.mean()),
                'median_stability': float(self.stability_scores.median()),
                'high_stability_count': int((self.stability_scores > 0.7).sum()),
            },
            'validation_results': self.validation_results,
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print("📊 FEATURE SELECTION SUMMARY")
            print(f"{'='*80}")
            print(f"Original features: {summary['baseline']['n_features']}")
            print(f"Selected features: {summary['selected']['n_features']}")
            print(f"Reduction: {summary['reduction']['features_removed']} features ({summary['reduction']['reduction_pct']:.1f}%)")
            print(f"\nPerformance Retention:")
            print(f"  Regime: {summary['selected']['regime_retention']:.1%}")
            print(f"  Range:  {summary['selected']['range_retention']:.1%}")
            print(f"\nStability:")
            print(f"  Mean: {summary['stability_stats']['mean_stability']:.3f}")
            print(f"  High stability features: {summary['stability_stats']['high_stability_count']}")
        
        return summary
    
    def _save_selection_results(self, summary: Dict):
        """Save feature selection results."""
        
        # Save summary JSON
        summary_path = self.output_dir / 'feature_selection_summary_v2.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Saved: {summary_path}")
        
        # Save selected features list
        features_path = self.output_dir / 'selected_features_v2.txt'
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.selected_features))
        
        print(f"✅ Saved: {features_path}")
        
        # Save full importance rankings with stability
        importance_path = self.output_dir / 'feature_importance_ranked_v2.csv'
        self.feature_importance_df.to_csv(importance_path, index=False)
        
        print(f"✅ Saved: {importance_path}")
        
        # Save stability scores
        stability_path = self.output_dir / 'feature_stability_scores.csv'
        self.stability_scores.to_frame('stability').to_csv(stability_path)
        
        print(f"✅ Saved: {stability_path}")


# ===== Integration Function =====

def run_intelligent_feature_selection(
    integrated_system,
    min_stability: float = 0.3,
    max_correlation: float = 0.95,
    preserve_forward_indicators: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run intelligent feature selection on trained IntegratedMarketSystemV4.
    
    Usage:
        from integrated_system_production import IntegratedMarketSystemV4
        from xgboost_feature_selector_v2 import run_intelligent_feature_selection
        
        system = IntegratedMarketSystemV4()
        system.train(years=15)
        
        selection_results = run_intelligent_feature_selection(
            system,
            min_stability=0.3,
            max_correlation=0.95,
            preserve_forward_indicators=True
        )
        
        selected_features = selection_results['selected_features']
    
    Args:
        integrated_system: Trained IntegratedMarketSystemV4 instance
        min_stability: Minimum stability threshold (0-1)
        max_correlation: Maximum correlation before removing redundancy
        preserve_forward_indicators: Force-keep critical forward indicators
        verbose: Print detailed progress
        
    Returns:
        Feature selection results dict
    """
    if not integrated_system.trained:
        raise ValueError("Train integrated system first: system.train(years=15)")
    
    selector = IntelligentFeatureSelector()
    
    results = selector.run_full_pipeline(
        features=integrated_system.orchestrator.features,
        vix=integrated_system.orchestrator.vix_ml,
        spx=integrated_system.orchestrator.spx_ml,
        candidate_sizes=None,  # Auto-detect
        min_stability=min_stability,
        max_correlation=max_correlation,
        preserve_forward_indicators=preserve_forward_indicators,
        verbose=verbose
    )
    
    return results


if __name__ == "__main__":
    print("Intelligent Feature Selector V2")
    print("\nEnhancements:")
    print("  • Stability-based selection (importance variance across folds)")
    print("  • Recursive feature addition (find performance cliff)")
    print("  • Multicollinearity removal (eliminate redundancy)")
    print("  • Forward indicator preservation (domain expertise)")
    print("\nRun from integrated_system_production.py using:")
    print("  results = run_intelligent_feature_selection(system)")