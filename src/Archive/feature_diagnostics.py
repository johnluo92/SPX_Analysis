"""Feature Diagnostic System - Identify What to Keep vs Remove

Run this to generate a comprehensive report on:
1. Which features are used by which anomaly detectors
2. Feature quality metrics (variance, sparsity, correlation)
3. Recommendations for removal
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class FeatureDiagnostics:
    """Analyze feature quality and detector usage."""
    
    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: Trained AnomalyOrchestrator instance
        """
        self.orch = orchestrator
        self.features = orchestrator.features
        self.detector = orchestrator.anomaly_detector
        self.report = {}
    
    def analyze_all(self) -> dict:
        """Run all diagnostic analyses."""
        print("\n" + "="*80)
        print("FEATURE DIAGNOSTIC ANALYSIS")
        print("="*80)
        
        self.report['timestamp'] = datetime.now().isoformat()
        self.report['total_features'] = len(self.features.columns)
        self.report['total_samples'] = len(self.features)
        
        # 1. Feature quality
        print("\n[1/5] Analyzing feature quality...")
        self.report['quality'] = self._analyze_feature_quality()
        
        # 2. Detector usage
        print("[2/5] Analyzing detector usage...")
        self.report['detector_usage'] = self._analyze_detector_usage()
        
        # 3. Correlation analysis
        print("[3/5] Analyzing feature correlations...")
        self.report['correlations'] = self._analyze_correlations()
        
        # 4. Removal recommendations
        print("[4/5] Generating removal recommendations...")
        self.report['recommendations'] = self._generate_recommendations()
        
        # 5. Category breakdown
        print("[5/5] Analyzing by category...")
        self.report['category_breakdown'] = self._analyze_by_category()
        
        return self.report
    
    def _analyze_feature_quality(self) -> dict:
        """Analyze basic quality metrics for all features."""
        quality = {}
        
        for col in self.features.columns:
            series = self.features[col]
            
            # Handle completely empty series
            non_na_count = len(series.dropna())
            unique_ratio = float(series.nunique() / non_na_count) if non_na_count > 0 else 0.0
            
            quality[col] = {
                'missing_pct': float(series.isna().sum() / len(series) * 100),
                'zero_pct': float((series == 0).sum() / len(series) * 100),
                'std': float(series.std()) if not series.isna().all() else 0.0,
                'unique_values': int(series.nunique()),
                'unique_ratio': unique_ratio,
                'min': float(series.min()) if not series.isna().all() else None,
                'max': float(series.max()) if not series.isna().all() else None,
            }
        
        return quality
    
    def _analyze_detector_usage(self) -> dict:
        """Analyze which features are used by which detectors."""
        usage = {}
        
        # Domain detectors
        for detector_name, feature_list in self.detector.feature_groups.items():
            available = [f for f in feature_list if f in self.features.columns]
            
            quality_report = self.detector.feature_quality_reports.get(detector_name, {})
            valid = [f for f in available if quality_report.get(f) == 'valid']
            rejected = [f for f in available if quality_report.get(f) != 'valid']
            
            usage[detector_name] = {
                'requested': len(feature_list),
                'available': len(available),
                'valid': len(valid),
                'rejected': len(rejected),
                'coverage': float(len(valid) / len(feature_list)) if feature_list else 0.0,
                'rejected_features': rejected,
                'rejection_reasons': {
                    f: quality_report.get(f, 'unknown') 
                    for f in rejected
                }
            }
        
        # Random subspaces
        for i, subspace in enumerate(self.detector.random_subspaces):
            detector_name = f'random_{i+1}'
            quality_report = self.detector.feature_quality_reports.get(detector_name, {})
            valid = [f for f in subspace if quality_report.get(f) == 'valid']
            
            usage[detector_name] = {
                'requested': len(subspace),
                'available': len(subspace),
                'valid': len(valid),
                'rejected': len(subspace) - len(valid),
                'coverage': float(len(valid) / len(subspace)) if subspace else 0.0,
            }
        
        return usage
    
    def _analyze_correlations(self) -> dict:
        """Find highly correlated feature pairs (redundancy detection)."""
        # Remove features that are all NaN before correlation
        valid_features = self.features.loc[:, self.features.notna().sum() > 10]
        
        if len(valid_features.columns) < 2:
            return {
                'high_correlation_pairs': [],
                'count': 0,
                'note': 'Not enough valid features for correlation analysis'
            }
        
        corr_matrix = valid_features.corr().abs()
        
        # Find pairs with correlation > 0.95
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and corr_val > 0.95:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        return {
            'high_correlation_pairs': high_corr_pairs,
            'count': len(high_corr_pairs)
        }
    
    def _generate_recommendations(self) -> dict:
        """Generate removal recommendations based on quality metrics."""
        recommendations = {
            'remove_high_priority': [],  # Clear garbage
            'remove_medium_priority': [],  # Likely redundant
            'review_manual': [],  # Needs human judgment
            'keep': []  # Definitely useful
        }
        
        quality = self.report['quality']
        usage = self.report['detector_usage']
        
        # Build reverse lookup: feature -> which detectors use it
        feature_to_detectors = {}
        for detector_name, feature_list in self.detector.feature_groups.items():
            for feat in feature_list:
                if feat not in feature_to_detectors:
                    feature_to_detectors[feat] = []
                feature_to_detectors[feat].append(detector_name)
        
        for feat, metrics in quality.items():
            reason = []
            
            # High priority removal
            if metrics['missing_pct'] > 80:
                reason.append(f"missing {metrics['missing_pct']:.1f}%")
            if metrics['zero_pct'] > 95:
                reason.append(f"zero {metrics['zero_pct']:.1f}%")
            if metrics['std'] < 1e-6:
                reason.append("no variance")
            if metrics['unique_ratio'] < 0.05 and metrics['unique_values'] < 3:
                reason.append("too few unique values")
            
            if reason:
                # Check if feature is actually used by any detector
                used_by = feature_to_detectors.get(feat, [])
                actually_used = False
                for detector in used_by:
                    # Check if detector exists in usage report
                    if detector not in usage:
                        continue
                    # Check if feature was rejected
                    if feat in usage[detector].get('rejected_features', []):
                        continue
                    actually_used = True
                    break
                
                if not actually_used:
                    recommendations['remove_high_priority'].append({
                        'feature': feat,
                        'reasons': reason,
                        'metrics': metrics
                    })
                else:
                    recommendations['review_manual'].append({
                        'feature': feat,
                        'reasons': reason,
                        'metrics': metrics,
                        'used_by': used_by
                    })
            else:
                # Feature looks okay
                recommendations['keep'].append(feat)
        
        # Check for redundant features from correlation analysis
        if 'correlations' in self.report:
            for pair in self.report['correlations']['high_correlation_pairs']:
                f1, f2 = pair['feature_1'], pair['feature_2']
                # Keep the one with better quality
                if f1 in quality and f2 in quality:
                    q1, q2 = quality[f1], quality[f2]
                    if q1['missing_pct'] > q2['missing_pct']:
                        worse = f1
                    else:
                        worse = f2
                    
                    recommendations['remove_medium_priority'].append({
                        'feature': worse,
                        'reasons': [f"redundant with {f1 if worse == f2 else f2}"],
                        'correlation': pair['correlation']
                    })
        
        return recommendations
    
    def _analyze_by_category(self) -> dict:
        """Break down features by category (Base, CBOE, Futures, etc.)."""
        categories = {
            'base': [],
            'cboe': [],
            'futures': [],
            'macro': [],
            'meta': [],
            'fred': [],
            'unknown': []
        }
        
        for col in self.features.columns:
            col_lower = col.lower()
            
            # Categorize by naming patterns
            if any(x in col_lower for x in ['vix', 'spx', 'rsi', 'macd', 'bb_', 'ret_', 'vol_', 'ma']):
                categories['base'].append(col)
            elif any(x in col_lower for x in ['skew', 'pcc', 'pcce', 'pcci', 'cor', 'vxth', 'cndr', 'bvol', 'bfly', 'dspx']):
                categories['cboe'].append(col)
            elif any(x in col_lower for x in ['vx_', 'cl_', 'dx_', 'crude', 'oil', 'dollar', 'dxy']):
                categories['futures'].append(col)
            elif any(x in col_lower for x in ['gold', 'silver', 'bond_vol', 'dollar_index']):
                categories['macro'].append(col)
            elif any(x in col_lower for x in ['regime', 'percentile', 'velocity', 'acceleration', 'jerk', 'divergence']):
                categories['meta'].append(col)
            elif any(x in col_lower for x in ['treasury', 'yield', 'dgs', 'dff', 'unrate', 'cpi', 'gdp', 'pce']):
                categories['fred'].append(col)
            else:
                categories['unknown'].append(col)
        
        # Compute stats per category
        breakdown = {}
        for cat, features in categories.items():
            if not features:
                continue
            
            breakdown[cat] = {
                'count': len(features),
                'avg_missing_pct': float(np.mean([
                    self.report['quality'][f]['missing_pct'] for f in features
                ])),
                'features': features[:10]  # Sample of features
            }
        
        return breakdown
    
    def save_report(self, output_path: str = './diagnostics/feature_report.json'):
        """Save report to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        print(f"\n✅ Report saved: {output_path}")
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "="*80)
        print("FEATURE DIAGNOSTIC SUMMARY")
        print("="*80)
        
        recs = self.report['recommendations']
        print(f"\n🎯 REMOVAL RECOMMENDATIONS:")
        print(f"   High Priority (clear garbage):  {len(recs['remove_high_priority'])} features")
        print(f"   Medium Priority (redundant):    {len(recs['remove_medium_priority'])} features")
        print(f"   Manual Review Needed:           {len(recs['review_manual'])} features")
        print(f"   Keep (good quality):            {len(recs['keep'])} features")
        
        print(f"\n📊 DETECTOR USAGE:")
        for detector, stats in self.report['detector_usage'].items():
            status = "✅" if stats['coverage'] > 0.7 else ("⚠️" if stats['coverage'] > 0.5 else "❌")
            print(f"   {status} {detector:30s} {stats['valid']:3d}/{stats['requested']:3d} valid ({stats['coverage']:.1%})")
        
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print(f"   Found {self.report['correlations']['count']} highly correlated pairs (>0.95)")
        
        if self.report['correlations']['count'] > 0:
            print("\n   Top 5 redundant pairs:")
            for pair in self.report['correlations']['high_correlation_pairs'][:5]:
                print(f"      {pair['feature_1']} <-> {pair['feature_2']} ({pair['correlation']:.3f})")
        
        print(f"\n📁 CATEGORY BREAKDOWN:")
        for cat, stats in self.report['category_breakdown'].items():
            print(f"   {cat.upper():10s}: {stats['count']:3d} features (avg {stats['avg_missing_pct']:.1f}% missing)")
        
        print("\n" + "="*80)
        
        # Print top 10 removal candidates
        if recs['remove_high_priority']:
            print("\n🗑️  TOP 10 REMOVAL CANDIDATES:")
            for i, item in enumerate(recs['remove_high_priority'][:10], 1):
                reasons_str = ', '.join(item['reasons'])
                print(f"   {i:2d}. {item['feature']:40s} → {reasons_str}")
        
        print("\n" + "="*80)


def run_diagnostics(system):
    """
    Run diagnostics on trained IntegratedMarketSystemV4.
    
    Usage:
        from integrated_system_production import IntegratedMarketSystemV4
        system = IntegratedMarketSystemV4()
        system.train(years=15)
        
        from feature_diagnostics import run_diagnostics
        report = run_diagnostics(system)
    """
    diagnostics = FeatureDiagnostics(system.orchestrator)
    report = diagnostics.analyze_all()
    diagnostics.print_summary()
    diagnostics.save_report()
    
    return report


if __name__ == "__main__":
    print("Import this module and call run_diagnostics(system)")
    print("\nExample:")
    print("  from integrated_system_production import IntegratedMarketSystemV4")
    print("  system = IntegratedMarketSystemV4()")
    print("  system.train(years=15)")
    print("  ")
    print("  from feature_diagnostics import run_diagnostics")
    print("  report = run_diagnostics(system)")