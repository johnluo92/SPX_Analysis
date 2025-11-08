"""
Data Quality Control System for SPX Anomaly Detection
======================================================

Implements comprehensive quality gates that run BEFORE training to ensure:
1. No constant/zero features pollute detector groups
2. No stale data triggers false anomalies
3. Detector coverage thresholds are enforced
4. Feature quality metrics are tracked

Author: Integrated into SPX_Analysis system
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureQualityMetrics:
    """Quality metrics for a single feature"""
    name: str
    missing_pct: float
    zero_pct: float
    constant_recent: bool
    stale_days: int
    has_inf: bool
    has_nan: bool
    std_recent: float
    unique_values: int
    passed: bool
    failed_checks: List[str]


@dataclass
class DetectorQualityReport:
    """Quality report for a detector group"""
    detector_name: str
    total_features: int
    available_features: int
    quality_features: int
    coverage_pct: float
    quality_coverage_pct: float
    missing_features: List[str]
    bad_features: List[str]
    passed: bool


class DataQualityController:
    """
    Main quality control system that validates features before training.
    
    Quality Gates:
    - Missing data threshold: 50%
    - Zero saturation threshold: 90%
    - Constant recent window: 20 days
    - Stale data threshold: 5 days
    - Inf/NaN detection
    - Detector coverage minimum: 50%
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.50,
        zero_threshold: float = 0.90,
        constant_window: int = 20,
        stale_window: int = 5,
        detector_coverage_min: float = 0.50,
        verbose: bool = True
    ):
        self.missing_threshold = missing_threshold
        self.zero_threshold = zero_threshold
        self.constant_window = constant_window
        self.stale_window = stale_window
        self.detector_coverage_min = detector_coverage_min
        self.verbose = verbose
        
        self.feature_metrics: Dict[str, FeatureQualityMetrics] = {}
        self.detector_reports: Dict[str, DetectorQualityReport] = {}
        self.removed_features: Set[str] = set()
        
    def validate_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main validation pipeline - run this BEFORE training.
        
        Returns:
            - Cleaned DataFrame with only quality features
            - Quality report dictionary
        """
        if self.verbose:
            print("\n" + "="*70)
            print("🔍 DATA QUALITY CONTROL - PRE-TRAINING VALIDATION")
            print("="*70)
            print(f"Input: {len(df)} rows, {len(df.columns)} features")
        
        # Step 1: Analyze individual feature quality
        self._analyze_feature_quality(df)
        
        # Step 2: Remove bad features
        clean_df = self._remove_bad_features(df)
        
        # Step 3: Generate quality report
        report = self._generate_quality_report(df, clean_df)
        
        if self.verbose:
            self._print_quality_summary(report)
        
        return clean_df, report
    
    def validate_detector_groups(
        self, 
        df: pd.DataFrame, 
        detector_groups: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], Dict]:
        """
        Validate detector feature groups and enforce coverage thresholds.
        
        Returns:
            - Updated detector groups (with bad features removed)
            - Detector quality report
        """
        if self.verbose:
            print("\n" + "="*70)
            print("🎯 DETECTOR GROUP VALIDATION")
            print("="*70)
        
        available_features = set(df.columns)
        updated_groups = {}
        
        for detector_name, feature_list in detector_groups.items():
            report = self._validate_detector_group(
                detector_name, 
                feature_list, 
                available_features
            )
            self.detector_reports[detector_name] = report
            
            if report.passed:
                # Keep only quality features in this group
                quality_features = [
                    f for f in feature_list 
                    if f in available_features and f not in self.removed_features
                ]
                updated_groups[detector_name] = quality_features
                
                if self.verbose:
                    print(f"✅ {detector_name:30s}: {report.quality_coverage_pct:5.1%} coverage "
                          f"({report.quality_features}/{report.total_features} features)")
            else:
                if self.verbose:
                    print(f"❌ {detector_name:30s}: {report.quality_coverage_pct:5.1%} coverage "
                          f"(BELOW MINIMUM {self.detector_coverage_min:.0%})")
        
        # Generate detector summary
        detector_summary = self._generate_detector_summary()
        
        return updated_groups, detector_summary
    
    def _analyze_feature_quality(self, df: pd.DataFrame) -> None:
        """Analyze quality metrics for each feature"""
        for col in df.columns:
            if col == 'date' or col.endswith('_target'):
                continue
            
            series = df[col]
            
            # Calculate metrics
            missing_pct = series.isna().sum() / len(series)
            zero_pct = (series.fillna(0) == 0).sum() / len(series)
            has_inf = np.isinf(series).any()
            has_nan = series.isna().any()
            unique_values = series.nunique()
            
            # Check recent data behavior
            recent = series.tail(self.constant_window)
            recent_clean = recent.dropna()
            constant_recent = len(recent_clean) > 0 and recent_clean.std() == 0
            std_recent = recent_clean.std() if len(recent_clean) > 0 else 0
            
            # Check staleness
            last_n = series.tail(self.stale_window).dropna()
            stale_days = 0
            if len(last_n) >= self.stale_window and last_n.std() == 0:
                stale_days = self.stale_window
            
            # Determine if feature passes quality gates
            failed_checks = []
            
            if missing_pct > self.missing_threshold:
                failed_checks.append(f"Missing: {missing_pct:.1%}")
            
            if zero_pct > self.zero_threshold:
                failed_checks.append(f"Zeros: {zero_pct:.1%}")
            
            if constant_recent:
                failed_checks.append(f"Constant (last {self.constant_window}d)")
            
            if stale_days > 0:
                failed_checks.append(f"Stale ({stale_days}d)")
            
            if has_inf:
                failed_checks.append("Contains inf")
            
            passed = len(failed_checks) == 0
            
            # Store metrics
            self.feature_metrics[col] = FeatureQualityMetrics(
                name=col,
                missing_pct=missing_pct,
                zero_pct=zero_pct,
                constant_recent=constant_recent,
                stale_days=stale_days,
                has_inf=has_inf,
                has_nan=has_nan,
                std_recent=std_recent,
                unique_values=unique_values,
                passed=passed,
                failed_checks=failed_checks
            )
            
            if not passed:
                self.removed_features.add(col)
    
    def _remove_bad_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features that failed quality checks"""
        cols_to_keep = [col for col in df.columns if col not in self.removed_features]
        clean_df = df[cols_to_keep].copy()
        
        if self.verbose and self.removed_features:
            print(f"\n⚠️  Removed {len(self.removed_features)} features with quality issues:")
            
            # Group by issue type
            issue_groups = {}
            for col in self.removed_features:
                if col in self.feature_metrics:
                    for issue in self.feature_metrics[col].failed_checks:
                        if issue not in issue_groups:
                            issue_groups[issue] = []
                        issue_groups[issue].append(col)
            
            # Print grouped issues
            for issue, features in sorted(issue_groups.items()):
                print(f"\n   {issue}:")
                for feat in sorted(features)[:10]:  # Show first 10
                    print(f"      • {feat}")
                if len(features) > 10:
                    print(f"      ... and {len(features)-10} more")
        
        return clean_df
    
    def _validate_detector_group(
        self, 
        detector_name: str, 
        feature_list: List[str],
        available_features: Set[str]
    ) -> DetectorQualityReport:
        """Validate a single detector group"""
        total_features = len(feature_list)
        
        # Check which features are available
        available = [f for f in feature_list if f in available_features]
        missing = [f for f in feature_list if f not in available_features]
        
        # Check which available features passed quality gates
        quality_features = [
            f for f in available 
            if f not in self.removed_features
        ]
        bad_features = [
            f for f in available 
            if f in self.removed_features
        ]
        
        # Calculate coverage
        coverage_pct = len(available) / total_features if total_features > 0 else 0
        quality_coverage_pct = len(quality_features) / total_features if total_features > 0 else 0
        
        # Check if detector passes minimum coverage threshold
        passed = quality_coverage_pct >= self.detector_coverage_min
        
        return DetectorQualityReport(
            detector_name=detector_name,
            total_features=total_features,
            available_features=len(available),
            quality_features=len(quality_features),
            coverage_pct=coverage_pct,
            quality_coverage_pct=quality_coverage_pct,
            missing_features=missing,
            bad_features=bad_features,
            passed=passed
        )
    
    def _generate_quality_report(self, original_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict:
        """Generate comprehensive quality report"""
        failed_features = [
            {
                'feature': metrics.name,
                'issues': metrics.failed_checks,
                'missing_pct': round(metrics.missing_pct * 100, 1),
                'zero_pct': round(metrics.zero_pct * 100, 1),
                'std_recent': round(metrics.std_recent, 4)
            }
            for metrics in self.feature_metrics.values()
            if not metrics.passed
        ]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'original_features': len(original_df.columns),
                'removed_features': len(self.removed_features),
                'final_features': len(clean_df.columns),
                'pass_rate': round((1 - len(self.removed_features) / len(original_df.columns)) * 100, 1)
            },
            'thresholds': {
                'missing_threshold': self.missing_threshold,
                'zero_threshold': self.zero_threshold,
                'constant_window': self.constant_window,
                'stale_window': self.stale_window,
                'detector_coverage_min': self.detector_coverage_min
            },
            'failed_features': failed_features,
            'removed_feature_names': sorted(list(self.removed_features))
        }
        
        return report
    
    def _generate_detector_summary(self) -> Dict:
        """Generate detector validation summary"""
        passed_detectors = [
            name for name, report in self.detector_reports.items() 
            if report.passed
        ]
        failed_detectors = [
            name for name, report in self.detector_reports.items() 
            if not report.passed
        ]
        
        summary = {
            'total_detectors': len(self.detector_reports),
            'passed_detectors': len(passed_detectors),
            'failed_detectors': len(failed_detectors),
            'pass_rate': round(len(passed_detectors) / len(self.detector_reports) * 100, 1) if self.detector_reports else 0,
            'detector_details': {
                name: {
                    'passed': report.passed,
                    'coverage': round(report.quality_coverage_pct * 100, 1),
                    'quality_features': report.quality_features,
                    'total_features': report.total_features,
                    'bad_features': report.bad_features[:5]  # Show first 5
                }
                for name, report in self.detector_reports.items()
            }
        }
        
        return summary
    
    def _print_quality_summary(self, report: Dict) -> None:
        """Print human-readable quality summary"""
        summary = report['summary']
        
        print(f"\n{'='*70}")
        print("📊 QUALITY SUMMARY")
        print(f"{'='*70}")
        print(f"Original features:  {summary['original_features']}")
        print(f"Removed features:   {summary['removed_features']}")
        print(f"Final features:     {summary['final_features']}")
        print(f"Pass rate:          {summary['pass_rate']}%")
        print(f"{'='*70}\n")
    
    def save_report(self, report: Dict, filepath: str) -> None:
        """Save quality report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose:
            print(f"💾 Quality report saved to: {filepath}")
    
    def get_cboe_quality_report(self) -> Dict:
        """Generate focused report on CBOE features"""
        cboe_features = {
            name: metrics 
            for name, metrics in self.feature_metrics.items()
            if any(prefix in name for prefix in ['SKEW', 'PCC', 'COR', 'VXTH', 'skew', 'pc_', 'cor_', 'vxth_'])
        }
        
        cboe_bad = [name for name, metrics in cboe_features.items() if not metrics.passed]
        cboe_good = [name for name, metrics in cboe_features.items() if metrics.passed]
        
        return {
            'total_cboe_features': len(cboe_features),
            'good_features': len(cboe_good),
            'bad_features': len(cboe_bad),
            'pass_rate': round(len(cboe_good) / len(cboe_features) * 100, 1) if cboe_features else 0,
            'bad_feature_names': sorted(cboe_bad),
            'common_issues': self._count_issue_types(cboe_features)
        }
    
    def _count_issue_types(self, features: Dict) -> Dict[str, int]:
        """Count frequency of each issue type"""
        issue_counts = {}
        for metrics in features.values():
            for issue in metrics.failed_checks:
                # Extract issue type (before the colon)
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))


class FeatureQualityMonitor:
    """
    Continuous monitoring system for feature quality during production.
    Detects when features degrade over time.
    """
    
    def __init__(self, baseline_report: Dict):
        self.baseline = baseline_report
        self.alerts: List[Dict] = []
    
    def check_feature_drift(self, current_df: pd.DataFrame) -> List[Dict]:
        """
        Compare current data against baseline quality metrics.
        Returns list of alerts for degraded features.
        """
        alerts = []
        
        for col in current_df.columns:
            if col == 'date':
                continue
            
            # Calculate current metrics
            recent = current_df[col].tail(20)
            recent_clean = recent.dropna()
            
            if len(recent_clean) == 0:
                alerts.append({
                    'feature': col,
                    'alert': 'NO_DATA',
                    'message': 'No recent data available'
                })
                continue
            
            # Check for staleness
            if recent_clean.std() == 0:
                alerts.append({
                    'feature': col,
                    'alert': 'STALE',
                    'message': 'No variation in last 20 days'
                })
            
            # Check for missing data spike
            missing_pct = recent.isna().sum() / len(recent)
            if missing_pct > 0.5:
                alerts.append({
                    'feature': col,
                    'alert': 'MISSING_SPIKE',
                    'message': f'Missing {missing_pct:.1%} of recent data'
                })
        
        self.alerts = alerts
        return alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of current alerts"""
        if not self.alerts:
            return {'status': 'healthy', 'alert_count': 0}
        
        alert_types = {}
        for alert in self.alerts:
            alert_type = alert['alert']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        return {
            'status': 'degraded',
            'alert_count': len(self.alerts),
            'alert_types': alert_types,
            'critical_features': [a['feature'] for a in self.alerts]
        }


def quick_quality_check(df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> None:
    """
    Quick quality check function for interactive use.
    
    Usage:
        from data_quality_control import quick_quality_check
        quick_quality_check(features_df, ANOMALY_FEATURE_GROUPS)
    """
    print("\n" + "="*70)
    print("🚀 QUICK QUALITY CHECK")
    print("="*70)
    
    controller = DataQualityController(verbose=False)
    clean_df, report = controller.validate_features(df)
    
    print(f"\nFeatures: {report['summary']['original_features']} → {report['summary']['final_features']}")
    print(f"Pass rate: {report['summary']['pass_rate']}%")
    
    if feature_groups:
        updated_groups, detector_summary = controller.validate_detector_groups(clean_df, feature_groups)
        print(f"\nDetectors: {detector_summary['passed_detectors']}/{detector_summary['total_detectors']} passed")
        
        if detector_summary['failed_detectors'] > 0:
            print("\n⚠️  Failed detectors:")
            for name, details in detector_summary['detector_details'].items():
                if not details['passed']:
                    print(f"   • {name}: {details['coverage']}% coverage")
    
    # CBOE-specific report
    cboe_report = controller.get_cboe_quality_report()
    if cboe_report['total_cboe_features'] > 0:
        print(f"\n📊 CBOE Features: {cboe_report['good_features']}/{cboe_report['total_cboe_features']} passed ({cboe_report['pass_rate']}%)")
        if cboe_report['bad_features'] > 0:
            print(f"   Common issues: {', '.join([f'{k} ({v})' for k, v in list(cboe_report['common_issues'].items())[:3]])}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("""
    Data Quality Control System
    ===========================
    
    Usage in your feature_engine.py:
    
    from data_quality_control import DataQualityController
    
    # In your FeatureEngine class, add:
    
    def build_features_with_quality_control(self, ...):
        # Build features as normal
        features = self.build_features(...)
        
        # Apply quality control BEFORE training
        qc = DataQualityController(verbose=True)
        clean_features, report = qc.validate_features(features)
        
        # Validate detector groups
        from config import ANOMALY_FEATURE_GROUPS
        updated_groups, detector_report = qc.validate_detector_groups(
            clean_features, 
            ANOMALY_FEATURE_GROUPS
        )
        
        # Save report
        qc.save_report(report, './data_cache/quality_report.json')
        
        return clean_features, updated_groups
    """)
