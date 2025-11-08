"""
Phase 1: XGBoost Feature Selection Pipeline
============================================

Goal: Use XGBoost to identify the TOP 50-75 features that actually predict:
  1. VIX regime transitions (classification)
  2. 5-day forward realized volatility (regression)

Then validate that these filtered features:
  - Preserve >95% of predictive power
  - Remove multicollinearity and noise
  - Improve anomaly detection signal-to-noise ratio

Usage:
    from xgboost_feature_selector import XGBoostFeatureSelector
    
    selector = XGBoostFeatureSelector()
    results = selector.run_full_pipeline(
        features=system.orchestrator.features,
        vix=system.orchestrator.vix_ml,
        spx=system.orchestrator.spx_ml
    )
    
    # Get filtered features
    top_features = results['selected_features']
    features_filtered = features[top_features]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

from core.xgboost_trainer import XGBoostTrainer


class XGBoostFeatureSelector:
    """
    Systematic feature selection using XGBoost importance.
    
    Process:
    1. Train XGBoost on ALL features (baseline)
    2. Extract composite importance (0.6*gain + 0.4*permutation)
    3. Test filtered feature sets (top 50, 75, 100)
    4. Select optimal set that preserves >95% accuracy
    5. Export filtered features for anomaly retraining
    """
    
    def __init__(self, output_dir: str = './models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.trainer = XGBoostTrainer(output_dir=str(self.output_dir))
        
        self.baseline_results = None
        self.feature_importance_df = None
        self.validation_results = []
        self.selected_features = None
    
    def run_full_pipeline(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        candidate_sizes: List[int] = [50, 75, 100],
        verbose: bool = True
    ) -> Dict:
        """
        Run complete feature selection pipeline.
        
        Args:
            features: Full feature matrix (696 features)
            vix: VIX series
            spx: SPX series
            candidate_sizes: Feature set sizes to test [50, 75, 100]
            verbose: Print progress
            
        Returns:
            {
                'baseline_results': {...},
                'feature_importance': DataFrame,
                'validation_results': [...],
                'selected_features': List[str],
                'selected_size': int,
                'performance_summary': {...}
            }
        """
        if verbose:
            print(f"\n{'='*80}")
            print("🎯 XGBOOST FEATURE SELECTION PIPELINE")
            print(f"{'='*80}")
            print(f"Starting features: {len(features.columns)}")
            print(f"Testing sizes: {candidate_sizes}")
        
        # Step 1: Train baseline on ALL features
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 1/4] Training baseline XGBoost on ALL features...")
            print(f"{'='*80}")
        
        self.baseline_results = self.trainer.train(
            features=features,
            vix=vix,
            spx=spx,
            n_splits=5,
            optimize_hyperparams=False,
            verbose=verbose
        )
        
        baseline_regime_acc = self.baseline_results['regime_metrics']['cv_balanced_accuracy_mean']
        baseline_range_rmse = self.baseline_results['range_metrics']['cv_rmse_mean']
        
        if verbose:
            print(f"\n✅ Baseline Performance:")
            print(f"   Regime Accuracy: {baseline_regime_acc:.3f}")
            print(f"   Range RMSE: {baseline_range_rmse:.2f}%")
        
        # Step 2: Extract and rank features
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 2/4] Extracting feature importance...")
            print(f"{'='*80}")
        
        self.feature_importance_df = self._load_and_rank_features(verbose)
        
        # Step 3: Test filtered feature sets
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 3/4] Testing filtered feature sets...")
            print(f"{'='*80}")
        
        for size in candidate_sizes:
            self._test_feature_subset(
                features, vix, spx, size,
                baseline_regime_acc, baseline_range_rmse,
                verbose
            )
        
        # Step 4: Select optimal feature set
        if verbose:
            print(f"\n{'='*80}")
            print("[STEP 4/4] Selecting optimal feature set...")
            print(f"{'='*80}")
        
        optimal_result = self._select_optimal_features(
            baseline_regime_acc, baseline_range_rmse, verbose
        )
        
        self.selected_features = optimal_result['features']
        
        # Generate summary report
        summary = self._generate_summary_report(
            baseline_regime_acc, baseline_range_rmse, optimal_result, verbose
        )
        
        # Save results
        self._save_selection_results(summary)
        
        return {
            'baseline_results': self.baseline_results,
            'feature_importance': self.feature_importance_df,
            'validation_results': self.validation_results,
            'selected_features': self.selected_features,
            'selected_size': len(self.selected_features),
            'performance_summary': summary
        }
    
    def _load_and_rank_features(self, verbose: bool) -> pd.DataFrame:
        """Load feature importance and create composite ranking."""
        
        # Load both regime and range importance
        regime_importance = pd.read_csv(self.output_dir / 'feature_importance_regime.csv')
        range_importance = pd.read_csv(self.output_dir / 'feature_importance_range.csv')
        
        # Merge on feature name
        importance_df = regime_importance.merge(
            range_importance[['feature', 'range_composite']],
            on='feature',
            how='outer'
        ).fillna(0)
        
        # Create overall composite (average of regime and range)
        importance_df['overall_composite'] = (
            importance_df['regime_composite'] * 0.5 +
            importance_df['range_composite'] * 0.5
        )
        
        # Sort by overall composite
        importance_df = importance_df.sort_values('overall_composite', ascending=False)
        
        if verbose:
            print(f"\n✅ Feature importance extracted")
            print(f"\nTop 10 Features (Overall Composite):")
            for i, row in importance_df.head(10).iterrows():
                fwd = "🔮" if row.get('is_forward_indicator', False) else "  "
                print(f"  {fwd} {i+1:>2}. {row['feature']:<45} {row['overall_composite']:.4f}")
            
            # Forward indicator contribution
            if 'is_forward_indicator' in importance_df.columns:
                fwd_contribution = importance_df[
                    importance_df['is_forward_indicator']
                ]['overall_composite'].sum()
                print(f"\n🔮 Forward Indicator Total Contribution: {fwd_contribution:.1%}")
        
        return importance_df
    
    def _test_feature_subset(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        n_features: int,
        baseline_regime_acc: float,
        baseline_range_rmse: float,
        verbose: bool
    ):
        """Test a specific feature subset size."""
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Testing: Top {n_features} features")
            print(f"{'─'*60}")
        
        # Get top N features
        top_features = self.feature_importance_df.head(n_features)['feature'].tolist()
        
        # Filter features
        features_filtered = features[top_features]
        
        if verbose:
            print(f"Training XGBoost on {len(features_filtered.columns)} features...")
        
        # Train XGBoost on filtered features
        filtered_trainer = XGBoostTrainer(output_dir=str(self.output_dir / f'filtered_{n_features}'))
        
        filtered_results = filtered_trainer.train(
            features=features_filtered,
            vix=vix,
            spx=spx,
            n_splits=5,
            optimize_hyperparams=False,
            verbose=False  # Suppress detailed output
        )
        
        filtered_regime_acc = filtered_results['regime_metrics']['cv_balanced_accuracy_mean']
        filtered_range_rmse = filtered_results['range_metrics']['cv_rmse_mean']
        
        # Calculate performance retention
        regime_retention = (filtered_regime_acc / baseline_regime_acc) if baseline_regime_acc > 0 else 0
        range_retention = (baseline_range_rmse / filtered_range_rmse) if filtered_range_rmse > 0 else 0
        
        # Store results
        result = {
            'n_features': n_features,
            'features': top_features,
            'regime_accuracy': filtered_regime_acc,
            'range_rmse': filtered_range_rmse,
            'regime_retention': regime_retention,
            'range_retention': range_retention,
            'regime_accuracy_drop': baseline_regime_acc - filtered_regime_acc,
            'range_rmse_change': filtered_range_rmse - baseline_range_rmse,
            'passes_threshold': (regime_retention >= 0.95 and range_retention >= 0.95)
        }
        
        self.validation_results.append(result)
        
        if verbose:
            print(f"  Regime Accuracy: {filtered_regime_acc:.3f} "
                  f"(Retention: {regime_retention:.1%}, Drop: {result['regime_accuracy_drop']:+.3f})")
            print(f"  Range RMSE: {filtered_range_rmse:.2f}% "
                  f"(Retention: {range_retention:.1%}, Change: {result['range_rmse_change']:+.2f}%)")
            
            if result['passes_threshold']:
                print(f"  ✅ PASSES: >95% performance retained")
            else:
                print(f"  ⚠️  WARNING: <95% performance retained")
    
    def _select_optimal_features(
        self,
        baseline_regime_acc: float,
        baseline_range_rmse: float,
        verbose: bool
    ) -> Dict:
        """Select optimal feature set based on validation results."""
        
        # Filter to sets that pass 95% threshold
        passing_results = [r for r in self.validation_results if r['passes_threshold']]
        
        if not passing_results:
            if verbose:
                print("\n⚠️  No feature sets passed 95% threshold, selecting best available")
            
            # Select set with highest combined retention
            best_result = max(
                self.validation_results,
                key=lambda r: (r['regime_retention'] + r['range_retention']) / 2
            )
        else:
            # Select smallest set that passes threshold (most parsimonious)
            best_result = min(passing_results, key=lambda r: r['n_features'])
        
        if verbose:
            print(f"\n✅ SELECTED: Top {best_result['n_features']} features")
            print(f"   Regime Accuracy: {best_result['regime_accuracy']:.3f} "
                  f"({best_result['regime_retention']:.1%} retention)")
            print(f"   Range RMSE: {best_result['range_rmse']:.2f}% "
                  f"({best_result['range_retention']:.1%} retention)")
        
        return best_result
    
    def _generate_summary_report(
        self,
        baseline_regime_acc: float,
        baseline_range_rmse: float,
        optimal_result: Dict,
        verbose: bool
    ) -> Dict:
        """Generate comprehensive summary report."""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'baseline': {
                'n_features': len(self.feature_importance_df),
                'regime_accuracy': baseline_regime_acc,
                'range_rmse': baseline_range_rmse
            },
            'selected': {
                'n_features': optimal_result['n_features'],
                'regime_accuracy': optimal_result['regime_accuracy'],
                'range_rmse': optimal_result['range_rmse'],
                'regime_retention': optimal_result['regime_retention'],
                'range_retention': optimal_result['range_retention'],
                'features': optimal_result['features']
            },
            'reduction': {
                'features_removed': len(self.feature_importance_df) - optimal_result['n_features'],
                'reduction_pct': (1 - optimal_result['n_features'] / len(self.feature_importance_df)) * 100
            },
            'validation_results': self.validation_results
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print("📊 FEATURE SELECTION SUMMARY")
            print(f"{'='*80}")
            print(f"Original features: {summary['baseline']['n_features']}")
            print(f"Selected features: {summary['selected']['n_features']}")
            print(f"Reduction: {summary['reduction']['features_removed']} features "
                  f"({summary['reduction']['reduction_pct']:.1f}%)")
            print(f"\nPerformance Retention:")
            print(f"  Regime: {summary['selected']['regime_retention']:.1%}")
            print(f"  Range:  {summary['selected']['range_retention']:.1%}")
        
        return summary
    
    def _save_selection_results(self, summary: Dict):
        """Save feature selection results."""
        
        # Save summary JSON
        summary_path = self.output_dir / 'feature_selection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Saved: {summary_path}")
        
        # Save selected features list
        features_path = self.output_dir / 'selected_features.txt'
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.selected_features))
        
        print(f"✅ Saved: {features_path}")
        
        # Save full importance rankings
        importance_path = self.output_dir / 'feature_importance_ranked.csv'
        self.feature_importance_df.to_csv(importance_path, index=False)
        
        print(f"✅ Saved: {importance_path}")


# ===== Integration Function =====

def run_feature_selection(
    integrated_system,
    candidate_sizes: List[int] = [50, 75, 100],
    verbose: bool = True
) -> Dict:
    """
    Run XGBoost feature selection on trained IntegratedMarketSystemV4.
    
    Usage:
        from integrated_system_production import IntegratedMarketSystemV4
        from xgboost_feature_selector import run_feature_selection
        
        system = IntegratedMarketSystemV4()
        system.train(years=15)
        
        selection_results = run_feature_selection(system)
        selected_features = selection_results['selected_features']
    
    Args:
        integrated_system: Trained IntegratedMarketSystemV4 instance
        candidate_sizes: Feature set sizes to test
        verbose: Print progress
        
    Returns:
        Feature selection results dict
    """
    if not integrated_system.trained:
        raise ValueError("Train integrated system first")
    
    selector = XGBoostFeatureSelector()
    
    results = selector.run_full_pipeline(
        features=integrated_system.orchestrator.features,
        vix=integrated_system.orchestrator.vix_ml,
        spx=integrated_system.orchestrator.spx_ml,
        candidate_sizes=candidate_sizes,
        verbose=verbose
    )
    
    return results


if __name__ == "__main__":
    print("XGBoost Feature Selection Pipeline")
    print("\nPhase 1: Scientific feature selection using XGBoost importance")
    print("Run from integrated_system_production.py using:")
    print("  selection_results = run_feature_selection(system)")
