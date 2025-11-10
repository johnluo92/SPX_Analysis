"""
XGBoost Feature Selector V2 - FIXED: Proper data alignment
===========================================================

FIX: The issue was improper data cleaning that resulted in 0 samples.
This version properly aligns features with targets before any operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from core.xgboost_trainer_v2 import EnhancedXGBoostTrainer

# Forward indicators to preserve
CRITICAL_FORWARD_INDICATORS = [
    'VX1-VX2', 'VX2-VX1_RATIO',
    'yield_10y2y', 'yield_10y3m',
    'VXTLT', 'vxtlt_vix_ratio',
    'SKEW', 'skew_vs_vix',
]


class IntelligentFeatureSelector:
    """Stability-based feature selection with proper data handling."""
    
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
        candidate_sizes: List[int] = None,
        min_stability: float = 0.3,
        max_correlation: float = 0.95,
        preserve_forward_indicators: bool = True,
        verbose: bool = True
    ) -> Dict:
        """Run intelligent feature selection pipeline."""
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
        
        # Step 2: Rank features by stability-weighted importance
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
        
        # Step 4: Recursive feature addition
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 4/5] Recursive feature addition (finding performance cliff)...")
            print(f"{'='*80}")
        
        if candidate_sizes is None:
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
        Train baseline and measure feature importance stability.
        
        CRITICAL FIX: Proper data alignment before any operations.
        """
        from sklearn.model_selection import TimeSeriesSplit
        import xgboost as xgb
        
        # === FIX: Clean and align data FIRST ===
        if verbose:
            print(f"   Preparing data (features: {len(features.columns)}, samples: {len(features)})...")
        
        # Remove low-quality features
        missing_pct = features.isnull().mean()
        variance = features.var()
        valid_features = features.columns[(missing_pct < 0.5) & (variance > 1e-8)]
        
        if verbose:
            print(f"   After quality filter: {len(valid_features)} features")
        
        # Clean features
        features_clean = features[valid_features].fillna(method='ffill').fillna(method='bfill')
        
        # Create targets
        regime_boundaries = [0, 16.77, 24.40, 39.67, 100]
        y_regime = pd.cut(vix, bins=regime_boundaries, labels=[0,1,2,3], include_lowest=True).astype(int)
        
        spx_returns = spx.pct_change()
        y_range = spx_returns.rolling(5).std().shift(-5) * np.sqrt(252) * 100
        
        # Align everything on common index
        common_idx = features_clean.index.intersection(y_regime.index).intersection(y_range.index)
        common_idx = common_idx[~y_regime[common_idx].isna() & ~y_range[common_idx].isna()]
        
        X = features_clean.loc[common_idx]
        y_regime_aligned = y_regime.loc[common_idx]
        y_range_aligned = y_range.loc[common_idx]
        
        if verbose:
            print(f"   After alignment: {len(X)} samples")
            print(f"   Date range: {X.index.min().date()} → {X.index.max().date()}")
        
        if len(X) < 1000:
            raise ValueError(f"Only {len(X)} samples after cleaning - check your data!")
        
        # === Train baseline trainer (fast, no SHAP) ===
        trainer = EnhancedXGBoostTrainer(output_dir=str(self.output_dir / 'baseline'))
        
        results = trainer.train(
            features=X,
            vix=vix.loc[X.index],
            spx=spx.loc[X.index],
            n_splits=5,
            optimize_hyperparams=False,
            crisis_balanced=True,
            compute_shap=False,  # Skip SHAP for speed
            verbose=False
        )
        
        # === Compute per-fold importance for stability ===
        if verbose:
            print("   Computing per-fold feature importance for stability...")
        
        tscv = TimeSeriesSplit(n_splits=5, test_size=252)
        fold_importances = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train = X.iloc[train_idx]
            y_train = y_regime_aligned.iloc[train_idx]
            
            # Train simple model
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
            importance_series = importance_series / importance_series.sum()
            
            fold_importances.append(importance_series)
        
        # Calculate stability
        importance_matrix = pd.DataFrame(fold_importances).T.fillna(0)
        mean_importance = importance_matrix.mean(axis=1)
        std_importance = importance_matrix.std(axis=1)
        
        stability = 1 - (std_importance / mean_importance.replace(0, np.nan))
        stability = stability.clip(0, 1).fillna(0)
        stability_series = pd.Series(stability, name='stability')
        
        if verbose:
            print(f"   ✅ Stability computed for {len(stability_series)} features")
            print(f"      Mean stability: {stability_series.mean():.3f}")
            print(f"      High stability features (>0.7): {(stability_series > 0.7).sum()}")
        
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
        """Rank features by stability-weighted importance."""
        importance_path = self.output_dir / 'baseline' / 'feature_importance_v2_overall.csv'
        
        if not importance_path.exists():
            raise FileNotFoundError(f"Baseline importance not found: {importance_path}")
        
        importance_df = pd.read_csv(importance_path)
        importance_df = importance_df.set_index('feature')
        stability_df = stability_scores.to_frame('stability')
        
        combined = importance_df.join(stability_df, how='left').fillna(0)
        combined = combined.reset_index().rename(columns={'index': 'feature'})
        
        # Stability-weighted score
        combined['stability_weighted_score'] = (
            combined['overall_shap'] * (combined['stability'] ** 0.5)
        )
        
        combined['passes_stability'] = combined['stability'] >= min_stability
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
        """Remove redundant features with correlation > threshold."""
        if verbose:
            print(f"\n   Checking for multicollinearity (threshold: {max_correlation})...")
        
        feature_names = importance_df['feature'].tolist()
        available_features = [f for f in feature_names if f in features.columns]
        
        corr_matrix = features[available_features].corr().abs()
        
        # Find highly correlated pairs
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
            f1, f2 = pair['feature1'], pair['feature2']
            
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
        """Recursively add features and find performance cliff."""
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
                compute_shap=False,
                verbose=False
            )
            
            regime_acc = results['regime_metrics']['cv_balanced_accuracy_mean']
            range_rmse = results['range_metrics']['cv_rmse_mean']
            
            regime_retention = (regime_acc / baseline_regime_acc) if baseline_regime_acc > 0 else 0
            range_retention = (baseline_range_rmse / range_rmse) if range_rmse > 0 else 0
            
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
                improvement = regime_acc - best_regime_acc
                if improvement < 0.005:
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
        """Force-include critical forward indicators."""
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
    
    def _generate_summary_report(self, baseline_metrics: Dict, verbose: bool) -> Dict:
        """Generate comprehensive summary report."""
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
        summary_path = self.output_dir / 'feature_selection_summary_v2.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Saved: {summary_path}")
        
        features_path = self.output_dir / 'selected_features_v2.txt'
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.selected_features))
        print(f"✅ Saved: {features_path}")
        
        importance_path = self.output_dir / 'feature_importance_ranked_v2.csv'
        self.feature_importance_df.to_csv(importance_path, index=False)
        print(f"✅ Saved: {importance_path}")
        
        stability_path = self.output_dir / 'feature_stability_scores.csv'
        self.stability_scores.to_frame('stability').to_csv(stability_path)
        print(f"✅ Saved: {stability_path}")


def run_intelligent_feature_selection(
    integrated_system,
    min_stability: float = 0.3,
    max_correlation: float = 0.95,
    preserve_forward_indicators: bool = True,
    verbose: bool = True
) -> Dict:
    """Run intelligent feature selection on trained system."""
    if not integrated_system.trained:
        raise ValueError("Train integrated system first: system.train(years=15)")
    
    selector = IntelligentFeatureSelector()
    
    results = selector.run_full_pipeline(
        features=integrated_system.orchestrator.features,
        vix=integrated_system.orchestrator.vix_ml,
        spx=integrated_system.orchestrator.spx_ml,
        candidate_sizes=None,
        min_stability=min_stability,
        max_correlation=max_correlation,
        preserve_forward_indicators=preserve_forward_indicators,
        verbose=verbose
    )
    
    return results


if __name__ == "__main__":
    print("Intelligent Feature Selector V2 - FIXED")
    print("\nEnhancements:")
    print("  • Proper data alignment (fixes 0 samples error)")
    print("  • Stability-based selection")
    print("  • Recursive feature addition")
    print("  • Multicollinearity removal")
    print("  • Forward indicator preservation")