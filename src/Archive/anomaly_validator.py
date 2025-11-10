"""Anomaly System Validator - Does It Actually Work?

Validates anomaly detection by checking:
1. Do ensemble scores spike during known crises?
2. Are scores calibrated (distribution matches expectations)?
3. Do high anomalies predict VIX increases?
"""
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class AnomalyValidator:
    """Validate that anomaly detection system is working correctly."""
    
    # Known market crisis periods
    CRISIS_PERIODS = {
        '2011_debt_crisis': ('2011-07-01', '2011-10-31'),
        '2015_china_devaluation': ('2015-08-01', '2015-09-30'),
        '2018_q4_selloff': ('2018-09-01', '2018-12-31'),
        '2020_covid': ('2020-02-01', '2020-04-30'),
        '2022_ukraine': ('2022-02-01', '2022-03-31'),
    }
    
    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: Trained AnomalyOrchestrator instance
        """
        self.orch = orchestrator
        self.features = orchestrator.features
        self.vix = orchestrator.vix_ml
        self.spx = orchestrator.spx_ml
        self.historical_scores = orchestrator.historical_ensemble_scores
        self.dates = self.features.index
        self.report = {}
    
    def validate_all(self) -> dict:
        """Run all validation checks."""
        print("\n" + "="*80)
        print("ANOMALY SYSTEM VALIDATION")
        print("="*80)
        
        self.report['timestamp'] = datetime.now().isoformat()
        
        # 1. Crisis alignment
        print("\n[1/5] Checking crisis period alignment...")
        self.report['crisis_alignment'] = self._validate_crisis_alignment()
        
        # 2. Score distribution
        print("[2/5] Validating score distribution...")
        self.report['distribution'] = self._validate_distribution()
        
        # 3. Predictive power
        print("[3/5] Testing predictive power...")
        self.report['predictive_power'] = self._validate_predictive_power()
        
        # 4. Detector consistency
        print("[4/5] Checking detector consistency...")
        self.report['detector_consistency'] = self._validate_detector_consistency()
        
        # 5. Threshold calibration
        print("[5/5] Validating threshold calibration...")
        self.report['threshold_calibration'] = self._validate_thresholds()
        
        return self.report
    
    def _validate_crisis_alignment(self) -> dict:
        """Check if anomaly scores spike during known crises."""
        alignment = {}
        
        for crisis_name, (start, end) in self.CRISIS_PERIODS.items():
            try:
                crisis_mask = (self.dates >= start) & (self.dates <= end)
                if crisis_mask.sum() == 0:
                    continue
                
                crisis_scores = self.historical_scores[crisis_mask]
                baseline_scores = self.historical_scores[~crisis_mask]
                
                alignment[crisis_name] = {
                    'mean_score': float(crisis_scores.mean()),
                    'baseline_mean': float(baseline_scores.mean()),
                    'ratio': float(crisis_scores.mean() / baseline_scores.mean()),
                    'pct_above_80': float((crisis_scores > 0.80).sum() / len(crisis_scores) * 100),
                    'max_score': float(crisis_scores.max()),
                    'days': int(crisis_mask.sum())
                }
            except Exception as e:
                alignment[crisis_name] = {'error': str(e)}
        
        # Overall crisis vs normal
        all_crisis_dates = []
        for start, end in self.CRISIS_PERIODS.values():
            mask = (self.dates >= start) & (self.dates <= end)
            all_crisis_dates.extend(self.dates[mask].tolist())
        
        crisis_mask = self.dates.isin(all_crisis_dates)
        
        alignment['overall'] = {
            'crisis_mean': float(self.historical_scores[crisis_mask].mean()),
            'normal_mean': float(self.historical_scores[~crisis_mask].mean()),
            'ratio': float(self.historical_scores[crisis_mask].mean() / 
                          self.historical_scores[~crisis_mask].mean()),
        }
        
        return alignment
    
    def _validate_distribution(self) -> dict:
        """Check if score distribution is reasonable."""
        scores = self.historical_scores
        
        distribution = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'percentiles': {
                '50': float(np.percentile(scores, 50)),
                '75': float(np.percentile(scores, 75)),
                '90': float(np.percentile(scores, 90)),
                '95': float(np.percentile(scores, 95)),
                '99': float(np.percentile(scores, 99)),
            },
            'pct_above_moderate': float((scores > 0.70).sum() / len(scores) * 100),
            'pct_above_high': float((scores > 0.78).sum() / len(scores) * 100),
            'pct_above_critical': float((scores > 0.88).sum() / len(scores) * 100),
        }
        
        # Check for degenerate distributions
        warnings = []
        if distribution['std'] < 0.05:
            warnings.append("Low variance - scores too similar")
        if distribution['mean'] > 0.7:
            warnings.append("Mean too high - system oversensitive")
        if distribution['mean'] < 0.2:
            warnings.append("Mean too low - system undersensitive")
        if distribution['pct_above_critical'] > 10:
            warnings.append("Too many critical anomalies (>10%)")
        
        distribution['warnings'] = warnings
        
        return distribution
    
    def _validate_predictive_power(self) -> dict:
        """Test if high anomaly scores predict VIX increases."""
        results = {}
        
        # Forward VIX changes at different horizons
        for horizon in [5, 10, 21]:
            vix_fwd_change = self.vix.pct_change(horizon).shift(-horizon) * 100
            
            # Split into high vs low anomaly periods
            high_anomaly = self.historical_scores > 0.75
            
            high_vix_change = vix_fwd_change[high_anomaly].dropna()
            low_vix_change = vix_fwd_change[~high_anomaly].dropna()
            
            results[f'{horizon}d_forward'] = {
                'high_anomaly_mean_vix_change': float(high_vix_change.mean()),
                'low_anomaly_mean_vix_change': float(low_vix_change.mean()),
                'difference': float(high_vix_change.mean() - low_vix_change.mean()),
                'high_anomaly_positive_pct': float((high_vix_change > 0).sum() / len(high_vix_change) * 100),
            }
        
        # VIX spike prediction (VIX increase >20% in next 5 days)
        vix_spike = (self.vix.pct_change(5).shift(-5) > 0.20)
        high_anomaly = self.historical_scores > 0.75
        
        # Contingency table
        tp = (high_anomaly & vix_spike).sum()  # True positive
        fp = (high_anomaly & ~vix_spike).sum()  # False positive
        fn = (~high_anomaly & vix_spike).sum()  # False negative
        tn = (~high_anomaly & ~vix_spike).sum()  # True negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results['vix_spike_prediction'] = {
            'precision': float(precision),
            'recall': float(recall),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
        
        return results
    
    def _validate_detector_consistency(self) -> dict:
        """Check if individual detectors are consistent with ensemble."""
        consistency = {}
        
        # Get current detection for all detectors
        current_result = self.orch.detect_current(verbose=False)
        
        ensemble_score = current_result['ensemble']['score']
        
        # Domain detectors
        for name, data in current_result.get('domain_anomalies', {}).items():
            deviation = abs(data['score'] - ensemble_score)
            consistency[name] = {
                'score': data['score'],
                'deviation_from_ensemble': float(deviation),
                'coverage': data['coverage'],
                'weight': data['weight'],
            }
        
        # Check for outlier detectors
        deviations = [d['deviation_from_ensemble'] for d in consistency.values()]
        
        summary = {
            'mean_deviation': float(np.mean(deviations)),
            'max_deviation': float(np.max(deviations)),
            'outlier_detectors': [
                name for name, data in consistency.items() 
                if data['deviation_from_ensemble'] > 0.3
            ]
        }
        
        consistency['summary'] = summary
        
        return consistency
    
    def _validate_thresholds(self) -> dict:
        """Validate that statistical thresholds are sensible."""
        thresholds = self.orch.anomaly_detector.statistical_thresholds
        
        validation = {
            'thresholds': thresholds,
            'empirical_frequencies': {
                'moderate': float((self.historical_scores >= thresholds['moderate']).sum() / len(self.historical_scores) * 100),
                'high': float((self.historical_scores >= thresholds['high']).sum() / len(self.historical_scores) * 100),
                'critical': float((self.historical_scores >= thresholds['critical']).sum() / len(self.historical_scores) * 100),
            }
        }
        
        # Check if frequencies match expectations
        warnings = []
        if validation['empirical_frequencies']['moderate'] < 10:
            warnings.append("Moderate threshold too high (expect ~15%)")
        if validation['empirical_frequencies']['high'] < 5:
            warnings.append("High threshold too high (expect ~8%)")
        if validation['empirical_frequencies']['critical'] < 1:
            warnings.append("Critical threshold too high (expect ~2%)")
        
        validation['warnings'] = warnings
        
        return validation
    
    def print_summary(self):
        """Print human-readable validation summary."""
        print("\n" + "="*80)
        print("ANOMALY SYSTEM VALIDATION SUMMARY")
        print("="*80)
        
        # Crisis alignment
        print("\n📊 CRISIS ALIGNMENT:")
        ca = self.report['crisis_alignment']
        if 'overall' in ca:
            ratio = ca['overall']['ratio']
            status = "✅" if ratio > 1.5 else ("⚠️" if ratio > 1.2 else "❌")
            print(f"   {status} Crisis/Normal Ratio: {ratio:.2f}x")
            print(f"      Crisis mean: {ca['overall']['crisis_mean']:.3f}")
            print(f"      Normal mean: {ca['overall']['normal_mean']:.3f}")
        
        print("\n   Individual Crises:")
        for crisis, data in ca.items():
            if crisis == 'overall' or 'error' in data:
                continue
            status = "✅" if data['ratio'] > 1.3 else "⚠️"
            print(f"   {status} {crisis:25s} {data['mean_score']:.3f} ({data['ratio']:.2f}x)")
        
        # Distribution
        print("\n📈 SCORE DISTRIBUTION:")
        dist = self.report['distribution']
        print(f"   Mean: {dist['mean']:.3f} | Std: {dist['std']:.3f}")
        print(f"   P95: {dist['percentiles']['95']:.3f} | P99: {dist['percentiles']['99']:.3f}")
        print(f"   Above Moderate (70%): {dist['pct_above_moderate']:.1f}%")
        print(f"   Above High (78%):     {dist['pct_above_high']:.1f}%")
        print(f"   Above Critical (88%): {dist['pct_above_critical']:.1f}%")
        
        if dist['warnings']:
            print("\n   ⚠️  Warnings:")
            for warning in dist['warnings']:
                print(f"      - {warning}")
        
        # Predictive power
        print("\n🎯 PREDICTIVE POWER:")
        pred = self.report['predictive_power']
        
        for horizon, data in pred.items():
            if horizon == 'vix_spike_prediction':
                continue
            print(f"   {horizon}: High anomaly → VIX change {data['high_anomaly_mean_vix_change']:+.2f}% vs {data['low_anomaly_mean_vix_change']:+.2f}%")
        
        spike = pred['vix_spike_prediction']
        print(f"\n   VIX Spike Prediction (+20% in 5d):")
        print(f"      Precision: {spike['precision']:.1%}")
        print(f"      Recall:    {spike['recall']:.1%}")
        print(f"      TP: {spike['true_positives']} | FP: {spike['false_positives']} | FN: {spike['false_negatives']}")
        
        # Consistency
        print("\n🔍 DETECTOR CONSISTENCY:")
        cons = self.report['detector_consistency']['summary']
        print(f"   Mean deviation from ensemble: {cons['mean_deviation']:.3f}")
        print(f"   Max deviation: {cons['max_deviation']:.3f}")
        if cons['outlier_detectors']:
            print(f"   ⚠️  Outlier detectors: {', '.join(cons['outlier_detectors'])}")
        
        print("\n" + "="*80)
        
        # Overall verdict
        print("\n🎯 OVERALL VERDICT:")
        
        issues = []
        if ca.get('overall', {}).get('ratio', 0) < 1.3:
            issues.append("❌ Crisis detection weak")
        if dist['std'] < 0.05:
            issues.append("❌ Low score variance")
        if pred['5d_forward']['difference'] < 2:
            issues.append("❌ Weak predictive power")
        if cons['max_deviation'] > 0.4:
            issues.append("❌ Detector inconsistency")
        
        if not issues:
            print("   ✅ Anomaly system is working correctly")
        else:
            print("   ⚠️  Issues detected:")
            for issue in issues:
                print(f"      {issue}")
        
        print("\n" + "="*80)
    
    def save_report(self, output_path: str = './diagnostics/anomaly_validation.json'):
        """Save validation report."""
        import json
        from pathlib import Path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        print(f"\n✅ Validation report saved: {output_path}")


def validate_anomaly_system(system):
    """
    Validate anomaly detection system.
    
    Usage:
        from integrated_system_production import IntegratedMarketSystemV4
        system = IntegratedMarketSystemV4()
        system.train(years=15)
        
        from anomaly_validator import validate_anomaly_system
        report = validate_anomaly_system(system)
    """
    validator = AnomalyValidator(system.orchestrator)
    report = validator.validate_all()
    validator.print_summary()
    validator.save_report()
    
    return report


if __name__ == "__main__":
    print("Import this module and call validate_anomaly_system(system)")
