"""
Phase 1: Statistical Threshold Calculator
Add-on module that calculates scientific thresholds alongside existing hardcoded ones.

SAFETY: Does NOT modify anomaly_system.py yet - just provides new infrastructure.
"""

import numpy as np
from scipy import stats
from scipy.stats import genpareto
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class StatisticalThresholdCalculator:
    """
    Calculate statistically-driven anomaly thresholds using multiple methods.
    
    This is a NON-BREAKING addition - it calculates new thresholds but doesn't
    replace existing hardcoded ones until explicitly requested.
    """
    
    def __init__(self, training_scores: np.ndarray, confidence_levels: dict = None):
        """
        Args:
            training_scores: Array of anomaly scores from training data
            confidence_levels: Dict of {name: alpha} for significance levels
                             Default: {'moderate': 0.10, 'high': 0.05, 'critical': 0.01}
        """
        self.training_scores = np.array(training_scores)
        self.confidence_levels = confidence_levels or {
            'moderate': 0.10,  # 90th percentile
            'high': 0.05,      # 95th percentile
            'critical': 0.01   # 99th percentile
        }
        
        # Calculate basic statistics
        self.mean = np.mean(self.training_scores)
        self.std = np.std(self.training_scores)
        self.median = np.median(self.training_scores)
        
        # Fit EVT distribution
        self._fit_evt()
        
    def _fit_evt(self):
        """Fit Generalized Pareto Distribution (EVT Peak-Over-Threshold)."""
        try:
            # Use 90th percentile as threshold for POT
            pot_threshold = np.percentile(self.training_scores, 90)
            excesses = self.training_scores[self.training_scores > pot_threshold] - pot_threshold
            
            if len(excesses) > 10:
                # Fit GPD to excesses
                self.evt_params = genpareto.fit(excesses)
                self.pot_threshold = pot_threshold
                self.evt_fitted = True
            else:
                self.evt_fitted = False
        except:
            self.evt_fitted = False
    
    def compute_all_thresholds(self) -> dict:
        """
        Compute thresholds using all 4 methods.
        
        Returns:
            Dict with structure:
            {
                'method_name': {
                    'moderate': threshold_value,
                    'high': threshold_value,
                    'critical': threshold_value
                },
                'recommended': {...},  # Ensemble recommendation
                'legacy': {...}        # Your current hardcoded values for comparison
            }
        """
        results = {}
        
        # Method 1: Empirical p-values (distribution-free)
        results['empirical'] = self._compute_empirical_thresholds()
        
        # Method 2: Z-scores (parametric)
        results['zscore'] = self._compute_zscore_thresholds()
        
        # Method 3: Bootstrap (non-parametric)
        results['bootstrap'] = self._compute_bootstrap_thresholds()
        
        # Method 4: EVT (extreme events)
        if self.evt_fitted:
            results['evt'] = self._compute_evt_thresholds()
        
        # Ensemble recommendation (median across methods)
        results['recommended'] = self._ensemble_thresholds(results)
        
        # Add legacy thresholds for comparison
        results['legacy'] = {
            'moderate': 0.50,
            'high': 0.70,
            'critical': 0.85
        }
        
        return results
    
    def _compute_empirical_thresholds(self) -> dict:
        """
        Empirical p-value method: Use actual training distribution.
        Most robust, makes no distributional assumptions.
        """
        thresholds = {}
        for name, alpha in self.confidence_levels.items():
            # (1 - alpha) quantile of training scores
            thresholds[name] = float(np.percentile(self.training_scores, (1 - alpha) * 100))
        return thresholds
    
    def _compute_zscore_thresholds(self) -> dict:
        """
        Z-score method: Assumes normal distribution.
        
        Z-score thresholds:
        - Z > 2.0: 95th percentile
        - Z > 2.5: 98.76th percentile
        - Z > 3.0: 99.87th percentile
        """
        z_scores = {
            'moderate': 2.0,
            'high': 2.5,
            'critical': 3.0
        }
        
        thresholds = {}
        for name, z in z_scores.items():
            thresholds[name] = float(self.mean + z * self.std)
        
        return thresholds
    
    def _compute_bootstrap_thresholds(self, n_bootstrap: int = 1000) -> dict:
        """Bootstrap confidence intervals (non-parametric)."""
        bootstrap_quantiles = {level: [] for level in self.confidence_levels.keys()}
        
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(
                self.training_scores, 
                size=len(self.training_scores), 
                replace=True
            )
            
            for name, alpha in self.confidence_levels.items():
                quantile = np.percentile(boot_sample, (1 - alpha) * 100)
                bootstrap_quantiles[name].append(quantile)
        
        thresholds = {}
        for name in self.confidence_levels.keys():
            thresholds[name] = float(np.median(bootstrap_quantiles[name]))
        
        return thresholds
    
    def _compute_evt_thresholds(self) -> dict:
        """Extreme Value Theory using Generalized Pareto Distribution."""
        if not self.evt_fitted:
            return {}
        
        thresholds = {}
        c, loc, scale = self.evt_params
        
        for name, alpha in self.confidence_levels.items():
            if c != 0:
                return_level = self.pot_threshold + (scale / c) * ((1 / alpha) ** c - 1)
            else:
                return_level = self.pot_threshold - scale * np.log(alpha)
            
            thresholds[name] = float(return_level)
        
        return thresholds
    
    def _ensemble_thresholds(self, all_thresholds: dict) -> dict:
        """Combine thresholds using median (robust to outlier methods)."""
        ensemble = {}
        
        methods = [k for k in all_thresholds.keys() if k != 'recommended' and all_thresholds[k]]
        
        for level in self.confidence_levels.keys():
            values = [all_thresholds[method][level] for method in methods 
                     if level in all_thresholds[method]]
            if values:
                ensemble[level] = float(np.median(values))
        
        return ensemble
    
    def get_pvalue(self, score: float) -> float:
        """
        Calculate empirical p-value for a given score.
        
        Returns:
            p-value: Probability of observing this score by chance
        """
        return float(np.mean(self.training_scores >= score))
    
    def classify_score(self, score: float, method: str = 'recommended') -> Tuple[str, float, float]:
        """
        Classify a score using specified method.
        
        Args:
            score: Anomaly score to classify
            method: One of ['empirical', 'zscore', 'bootstrap', 'evt', 'recommended', 'legacy']
            
        Returns:
            (severity_level, p_value, confidence)
        """
        thresholds_all = self.compute_all_thresholds()
        thresholds = thresholds_all.get(method, thresholds_all['recommended'])
        
        p_value = self.get_pvalue(score)
        
        if score >= thresholds['critical']:
            return 'CRITICAL', p_value, 0.99
        elif score >= thresholds['high']:
            return 'HIGH', p_value, 0.95
        elif score >= thresholds['moderate']:
            return 'MODERATE', p_value, 0.90
        else:
            return 'NORMAL', p_value, 0.0
    
    def compare_with_legacy(self) -> dict:
        """
        Compare scientific thresholds with legacy hardcoded values.
        
        Returns diagnostic information showing differences.
        """
        all_thresholds = self.compute_all_thresholds()
        
        comparison = {
            'summary': {},
            'by_method': {}
        }
        
        for method_name, thresholds in all_thresholds.items():
            if method_name == 'legacy' or not thresholds:
                continue
            
            comparison['by_method'][method_name] = {}
            
            for level in ['moderate', 'high', 'critical']:
                scientific = thresholds.get(level, 0)
                legacy = all_thresholds['legacy'][level]
                diff = scientific - legacy
                pct_diff = (diff / legacy) * 100 if legacy > 0 else 0
                
                comparison['by_method'][method_name][level] = {
                    'scientific': scientific,
                    'legacy': legacy,
                    'difference': diff,
                    'pct_difference': pct_diff
                }
        
        # Overall recommendation vs legacy
        rec = all_thresholds['recommended']
        legacy = all_thresholds['legacy']
        
        comparison['summary'] = {
            'recommended_vs_legacy': {
                level: {
                    'recommended': rec[level],
                    'legacy': legacy[level],
                    'difference': rec[level] - legacy[level],
                    'pct_difference': ((rec[level] - legacy[level]) / legacy[level] * 100)
                }
                for level in ['moderate', 'high', 'critical']
            }
        }
        
        return comparison


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_threshold_calculator_on_training_data(training_scores: np.ndarray, 
                                               test_scores: np.ndarray = None,
                                               verbose: bool = True):
    """
    Test the statistical threshold calculator on your actual training data.
    
    Args:
        training_scores: Anomaly scores from training (ensemble scores)
        test_scores: Optional test scores to classify
        verbose: Print detailed output
    
    Returns:
        dict with threshold analysis
    """
    calc = StatisticalThresholdCalculator(training_scores)
    
    results = {
        'training_stats': {
            'n_samples': len(training_scores),
            'mean': float(np.mean(training_scores)),
            'std': float(np.std(training_scores)),
            'min': float(np.min(training_scores)),
            'max': float(np.max(training_scores)),
            'median': float(np.median(training_scores))
        },
        'thresholds': calc.compute_all_thresholds(),
        'comparison': calc.compare_with_legacy()
    }
    
    if verbose:
        print("\n" + "="*80)
        print("STATISTICAL THRESHOLD ANALYSIS")
        print("="*80)
        
        print(f"\nTraining Data Statistics:")
        print(f"  Samples: {results['training_stats']['n_samples']}")
        print(f"  Mean: {results['training_stats']['mean']:.4f}")
        print(f"  Std Dev: {results['training_stats']['std']:.4f}")
        print(f"  Range: [{results['training_stats']['min']:.4f}, {results['training_stats']['max']:.4f}]")
        
        print("\n" + "-"*80)
        print("THRESHOLD COMPARISON")
        print("-"*80)
        
        print(f"\n{'Method':<15} {'Moderate':>12} {'High':>12} {'Critical':>12}")
        print("-"*55)
        
        for method, thresholds in results['thresholds'].items():
            if thresholds:
                print(f"{method:<15} {thresholds.get('moderate', 0):>12.4f} "
                      f"{thresholds.get('high', 0):>12.4f} {thresholds.get('critical', 0):>12.4f}")
        
        print("\n" + "-"*80)
        print("RECOMMENDED VS LEGACY")
        print("-"*80)
        
        summary = results['comparison']['summary']['recommended_vs_legacy']
        for level in ['moderate', 'high', 'critical']:
            data = summary[level]
            print(f"\n{level.upper()}:")
            print(f"  Recommended: {data['recommended']:.4f}")
            print(f"  Legacy:      {data['legacy']:.4f}")
            print(f"  Difference:  {data['difference']:+.4f} ({data['pct_difference']:+.1f}%)")
    
    # Test classification if test scores provided
    if test_scores is not None:
        results['test_classifications'] = []
        
        if verbose:
            print("\n" + "-"*80)
            print("TEST SCORE CLASSIFICATIONS")
            print("-"*80)
            print(f"\n{'Score':<10} {'Legacy':<12} {'Recommended':<12} {'P-Value':<10}")
            print("-"*50)
        
        for score in test_scores:
            legacy_level, _, _ = calc.classify_score(score, method='legacy')
            rec_level, p_val, conf = calc.classify_score(score, method='recommended')
            
            results['test_classifications'].append({
                'score': float(score),
                'legacy_classification': legacy_level,
                'recommended_classification': rec_level,
                'p_value': p_val,
                'confidence': conf
            })
            
            if verbose:
                print(f"{score:<10.4f} {legacy_level:<12} {rec_level:<12} {p_val:<10.4f}")
    
    return results


def export_threshold_analysis(calc: StatisticalThresholdCalculator, 
                              filepath: str = './json_data/threshold_analysis.json'):
    """
    Export comprehensive threshold analysis to JSON for dashboard display.
    
    Args:
        calc: Trained StatisticalThresholdCalculator
        filepath: Output JSON path
    """
    import json
    from datetime import datetime
    
    analysis = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'training_samples': len(calc.training_scores),
            'methods_used': ['empirical', 'zscore', 'bootstrap', 'evt'],
            'confidence_levels': calc.confidence_levels
        },
        'training_distribution': {
            'mean': float(calc.mean),
            'std': float(calc.std),
            'median': float(calc.median),
            'min': float(np.min(calc.training_scores)),
            'max': float(np.max(calc.training_scores)),
            'percentiles': {
                '10': float(np.percentile(calc.training_scores, 10)),
                '25': float(np.percentile(calc.training_scores, 25)),
                '50': float(np.percentile(calc.training_scores, 50)),
                '75': float(np.percentile(calc.training_scores, 75)),
                '90': float(np.percentile(calc.training_scores, 90)),
                '95': float(np.percentile(calc.training_scores, 95)),
                '99': float(np.percentile(calc.training_scores, 99))
            }
        },
        'thresholds': calc.compute_all_thresholds(),
        'comparison': calc.compare_with_legacy(),
        'interpretation': {
            'empirical': 'Distribution-free, uses actual training data percentiles',
            'zscore': 'Assumes normal distribution, standard statistical approach',
            'bootstrap': 'Non-parametric resampling, robust to outliers',
            'evt': 'Extreme Value Theory, best for tail events',
            'recommended': 'Median across all methods (most robust)',
            'legacy': 'Current hardcoded thresholds (0.50, 0.70, 0.85)'
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Threshold analysis exported to: {filepath}")
    return analysis