"""
Anomaly System Validation - Simplified
Quick validation checks for production readiness.
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from vix_predictor_v2 import VIXPredictorV4


class QuickValidator:
    """Fast validation checks for anomaly system."""
    
    def __init__(self):
        self.predictor = VIXPredictorV4()
        self.results = {'tests': {}, 'patterns': {}}
    
    def run(self, export: bool = True):
        """Run all validation tests."""
        print("\n" + "="*80)
        print("ANOMALY SYSTEM VALIDATION")
        print("="*80)
        
        # Train
        print("\n[1/2] Training system...")
        start = time.time()
        self.predictor.train()
        train_time = time.time() - start
        print(f"   ✅ Trained in {train_time:.1f}s")
        
        # Run tests
        print("\n[2/2] Running tests...")
        tests = [
            ('data_leakage', self._test_data_leakage),
            ('contamination', self._test_contamination),
            ('consistency', self._test_consistency),
            ('coverage', self._test_coverage),
            ('performance', self._test_performance)
        ]
        
        passed = 0
        for name, test_func in tests:
            result = test_func()
            self.results['tests'][name] = result
            
            if result['status'] == 'PASS':
                passed += 1
                print(f"   ✅ {name:15s} {result['message']}")
            else:
                print(f"   ❌ {name:15s} {result['message']}")
        
        # Summary
        pass_rate = passed / len(tests)
        self.results['summary'] = {
            'passed': passed,
            'total': len(tests),
            'pass_rate': pass_rate,
            'train_time': train_time
        }
        
        # Verdict
        print("\n" + "="*80)
        if pass_rate >= 0.8:
            print("✅ READY FOR PRODUCTION")
        elif pass_rate >= 0.6:
            print("⚠️  NEEDS IMPROVEMENT")
        else:
            print("❌ DO NOT DEPLOY")
        print(f"Pass rate: {pass_rate:.0%} ({passed}/{len(tests)})")
        print("="*80)
        
        # Export
        if export:
            output_path = './json_data/validation_report.json'
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n📄 Report saved: {output_path}")
        
        return self.results
    
    def _test_data_leakage(self):
        """Test for data leakage in features."""
        features = self.predictor.features
        vix = self.predictor.vix_ml
        
        # Check MA alignment
        expected_ma = vix.rolling(21).mean().shift(1)
        recovered_ma = vix - features['vix_vs_ma21']
        corr = expected_ma.corr(recovered_ma)
        
        return {
            'status': 'PASS' if corr > 0.99 else 'FAIL',
            'message': f'corr={corr:.4f}',
            'value': float(corr)
        }
    
    def _test_contamination(self):
        """Test contamination rate."""
        features = self.predictor.features.fillna(0)
        sample_size = min(100, len(features))
        
        scores = []
        for i in np.random.choice(len(features), sample_size, replace=False):
            result = self.predictor.anomaly_detector.detect(
                features.iloc[[i]], verbose=False
            )
            scores.append(result['ensemble']['score'])
        
        contamination = np.mean(np.array(scores) > 0.95)
        
        return {
            'status': 'PASS' if 0.02 < contamination < 0.08 else 'FAIL',
            'message': f'rate={contamination:.1%}',
            'value': float(contamination)
        }
    
    def _test_consistency(self):
        """Test ensemble consistency."""
        current = self.predictor.features.iloc[[-1]]
        result = self.predictor.anomaly_detector.detect(current, verbose=False)
        std = result['ensemble']['std']
        
        return {
            'status': 'PASS' if 0.05 < std < 0.3 else 'FAIL',
            'message': f'std={std:.3f}',
            'value': float(std)
        }
    
    def _test_coverage(self):
        """Test feature coverage."""
        all_features = set(self.predictor.features.columns)
        covered = set()
        
        for features in self.predictor.anomaly_detector.feature_groups.values():
            covered.update(features)
        for subspace in self.predictor.anomaly_detector.random_subspaces:
            covered.update(subspace)
        
        coverage = len(covered) / len(all_features)
        
        return {
            'status': 'PASS' if coverage > 0.6 else 'FAIL',
            'message': f'coverage={coverage:.0%}',
            'value': float(coverage)
        }
    
    def _test_performance(self):
        """Test detection speed."""
        current = self.predictor.features.iloc[[-1]]
        times = []
        
        for _ in range(10):
            start = time.time()
            self.predictor.anomaly_detector.detect(current, verbose=False)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        
        return {
            'status': 'PASS' if avg_time < 1.0 else 'WARNING',
            'message': f'{avg_time*1000:.0f}ms',
            'value': float(avg_time * 1000)
        }


def main():
    """Run validation."""
    validator = QuickValidator()
    results = validator.run(export=True)
    
    # Print key metrics
    print("\n📊 Key Metrics:")
    for test_name, test_result in results['tests'].items():
        if 'value' in test_result:
            print(f"   {test_name:15s} {test_result['value']:.3f}")


if __name__ == "__main__":
    main()
