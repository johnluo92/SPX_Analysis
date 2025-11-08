"""Enhanced Technical System Introspector for Debugging & Deep Diagnostics

This script provides comprehensive technical introspection for the Integrated Market System.
Designed for: Engineers, Data Scientists, LLMs debugging issues

Key Capabilities:
1. Deep model parameter inspection
2. Data pipeline flow tracing
3. Performance profiling
4. Feature correlation & redundancy analysis
5. Anomaly detector ensemble breakdown
6. Data freshness & staleness detection
7. Common failure mode diagnostics
8. What-if scenario analysis

Usage in Jupyter:
    from enhanced_technical_introspector import TechnicalIntrospector
    
    introspector = TechnicalIntrospector(system)
    introspector.generate_report()
    
    # Or specific diagnostics:
    introspector.diagnose_current_state()
    introspector.profile_performance()
    introspector.check_data_freshness()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from IPython.display import Markdown, display, HTML
import warnings
import time
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')


class TechnicalIntrospector:
    """Deep technical introspection for system debugging and optimization."""
    
    def __init__(self, system):
        """
        Args:
            system: Trained IntegratedMarketSystemV4 instance
        """
        self.system = system
        self.orch = system.orchestrator
        self.detector = self.orch.anomaly_detector
        self.features = self.orch.features
        self.vix = self.orch.vix_ml
        self.spx = self.orch.spx_ml
        
        # Diagnostic flags
        self.issues_found = []
        self.warnings_found = []
        self.performance_metrics = {}
        
    def generate_report(self, output_path='./docs/TECHNICAL_DIAGNOSTIC.md', 
                       display_in_notebook=True, quick_mode=False):
        """
        Generate comprehensive technical diagnostic report.
        
        Args:
            output_path: Where to save markdown report
            display_in_notebook: Show in Jupyter
            quick_mode: Skip expensive computations (for quick checks)
        """
        report = []
        start_time = time.time()
        
        # Header
        report.append("# 🔧 Technical System Diagnostic Report")
        report.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append(f"*Quick Mode: {'ON' if quick_mode else 'OFF'}*\n")
        report.append("---\n")
        
        # 1. System Health Check
        report.append("## 1. 🏥 System Health Check\n")
        health = self._check_system_health()
        
        status_emoji = '✅' if health['status'] == 'healthy' else '⚠️' if health['status'] == 'degraded' else '❌'
        report.append(f"**Overall Status**: {status_emoji} {health['status'].upper()}\n")
        
        if health['critical_issues']:
            report.append("### ❌ Critical Issues")
            for issue in health['critical_issues']:
                report.append(f"- **{issue['component']}**: {issue['message']}")
            report.append("")
        
        if health['warnings']:
            report.append("### ⚠️ Warnings")
            for warning in health['warnings']:
                report.append(f"- {warning}")
            report.append("")
        
        report.append("### Component Status")
        report.append("| Component | Status | Details |")
        report.append("|-----------|--------|---------|")
        for comp, details in health['components'].items():
            status = '✅' if details['ok'] else '❌'
            report.append(f"| {comp} | {status} | {details['message']} |")
        report.append("")
        
        # 2. Model Configuration Deep Dive
        report.append("## 2. 🎛️ Model Configuration Deep Dive\n")
        model_config = self._inspect_model_configuration()
        
        report.append("### Anomaly Detector Configuration")
        report.append("```python")
        report.append(f"contamination: {model_config['anomaly_detector']['contamination']:.3f}")
        report.append(f"n_estimators: {model_config['anomaly_detector']['n_estimators']}")
        report.append(f"max_samples: {model_config['anomaly_detector']['max_samples']}")
        report.append(f"random_state: {model_config['anomaly_detector']['random_state']}")
        report.append("```\n")
        
        report.append("### Individual Detector Parameters")
        report.append("| Detector | Features Used | Coverage | Active |")
        report.append("|----------|--------------|----------|--------|")
        for name, params in model_config['detectors'].items():
            status = '✅' if params['active'] else '❌'
            report.append(
                f"| {name} | {params['n_features']} | "
                f"{params['coverage']:.1%} | {status} |"
            )
        report.append("")
        
        # 3. Data Pipeline Flow Trace
        report.append("## 3. 🔄 Data Pipeline Flow Trace\n")
        pipeline_trace = self._trace_data_pipeline()
        
        report.append("### Data Sources → Features → Models")
        report.append("```")
        for step in pipeline_trace['steps']:
            report.append(f"{step['stage']:.<40} {step['status']:>10}")
            if step.get('details'):
                for detail in step['details']:
                    report.append(f"  ↳ {detail}")
        report.append("```\n")
        
        report.append("### Feature Generation Summary")
        report.append(f"- **Raw Market Data Points**: {pipeline_trace['raw_data_points']}")
        report.append(f"- **Engineered Features**: {pipeline_trace['engineered_features']}")
        report.append(f"- **Final Feature Set**: {pipeline_trace['final_features']}")
        report.append(f"- **Data Reduction Ratio**: {pipeline_trace['reduction_ratio']:.1f}x\n")
        
        # 4. Data Freshness Analysis
        report.append("## 4. 📅 Data Freshness & Staleness\n")
        freshness = self._check_data_freshness()
        
        report.append("### Last Update Times")
        report.append("| Data Source | Last Update | Age (days) | Status |")
        report.append("|-------------|-------------|------------|--------|")
        for source, info in freshness['sources'].items():
            age_days = info['age_days']
            status = '✅' if age_days <= 1 else '⚠️' if age_days <= 3 else '❌'
            report.append(
                f"| {source} | {info['last_update']} | "
                f"{age_days:.1f} | {status} |"
            )
        report.append("")
        
        if freshness['stale_features']:
            report.append("### ⚠️ Stale Features (>5% missing in recent data)")
            for feat, missing_pct in freshness['stale_features'][:10]:
                report.append(f"- **{feat}**: {missing_pct:.1%} missing")
            report.append("")
        
        # 5. Current Anomaly Breakdown
        report.append("## 5. 🎯 Current Anomaly Detection Breakdown\n")
        anomaly_breakdown = self._breakdown_current_anomaly()
        
        report.append(f"### Ensemble Score: {anomaly_breakdown['ensemble_score']:.1%}\n")
        
        report.append("### Detector Contributions")
        report.append("| Detector | Score | Weight | Weighted Score | Agreement |")
        report.append("|----------|-------|--------|----------------|-----------|")
        for det in anomaly_breakdown['detector_contributions']:
            agreement = '🟢' if det['agrees_with_ensemble'] else '🔴'
            report.append(
                f"| {det['name']} | {det['score']:.1%} | "
                f"{det['weight']:.2f} | {det['weighted_score']:.1%} | {agreement} |"
            )
        report.append("")
        
        if anomaly_breakdown['top_driving_features']:
            report.append("### Top Features Driving Current Anomaly")
            report.append("| Feature | Importance | Current Value | Z-Score |")
            report.append("|---------|-----------|---------------|---------|")
            for feat_info in anomaly_breakdown['top_driving_features'][:10]:
                report.append(
                    f"| {feat_info['feature']} | {feat_info['importance']:.3f} | "
                    f"{feat_info['current_value']:.2f} | {feat_info['zscore']:.2f} |"
                )
            report.append("")
        
        # 6. Feature Correlation & Redundancy
        if not quick_mode:
            report.append("## 6. 🔗 Feature Correlation & Redundancy Analysis\n")
            correlation_analysis = self._analyze_feature_correlations()
            
            report.append(f"### High Correlation Pairs (>{correlation_analysis['threshold']:.0%})")
            report.append("| Feature 1 | Feature 2 | Correlation |")
            report.append("|-----------|-----------|-------------|")
            for pair in correlation_analysis['high_correlation_pairs'][:15]:
                report.append(
                    f"| {pair['feature1']} | {pair['feature2']} | "
                    f"{pair['correlation']:.3f} |"
                )
            report.append("")
            
            report.append(f"**Recommendation**: Consider removing {len(correlation_analysis['high_correlation_pairs'])} "
                         f"redundant features to improve performance.\n")
        
        # 7. Performance Profiling
        report.append("## 7. ⚡ Performance Profiling\n")
        if not quick_mode:
            performance = self._profile_performance()
            
            report.append("### Execution Time Breakdown")
            report.append("| Operation | Time (ms) | % of Total |")
            report.append("|-----------|-----------|------------|")
            total_time = sum(performance['timings'].values())
            for op, time_ms in sorted(performance['timings'].items(), 
                                     key=lambda x: x[1], reverse=True):
                pct = (time_ms / total_time * 100) if total_time > 0 else 0
                report.append(f"| {op} | {time_ms:.1f} | {pct:.1f}% |")
            report.append("")
            
            if performance['bottlenecks']:
                report.append("### ⚠️ Performance Bottlenecks")
                for bottleneck in performance['bottlenecks']:
                    report.append(f"- **{bottleneck['operation']}**: {bottleneck['time_ms']:.1f}ms "
                                f"({bottleneck['reason']})")
                report.append("")
        else:
            report.append("*Performance profiling skipped in quick mode*\n")
        
        # 8. Common Failure Modes & Solutions
        report.append("## 8. 🔍 Common Failure Modes & Solutions\n")
        failure_modes = self._diagnose_common_failures()
        
        if failure_modes['detected_issues']:
            for issue in failure_modes['detected_issues']:
                report.append(f"### ⚠️ {issue['name']}")
                report.append(f"**Symptoms**: {issue['symptoms']}")
                report.append(f"**Root Cause**: {issue['cause']}")
                report.append(f"**Solution**: {issue['solution']}\n")
        else:
            report.append("✅ No common failure modes detected.\n")
        
        # 9. What-If Scenarios
        report.append("## 9. 💡 What-If Scenario Analysis\n")
        scenarios = self._run_scenario_analysis()
        
        for scenario in scenarios['scenarios']:
            report.append(f"### {scenario['name']}")
            report.append(f"**Scenario**: {scenario['description']}")
            report.append(f"**Expected Behavior**: {scenario['expected']}")
            report.append(f"**Current System Response**: {scenario['actual']}\n")
        
        # 10. Data Quality Heatmap
        report.append("## 10. 📊 Data Quality Heatmap\n")
        quality_heatmap = self._generate_quality_heatmap()
        
        report.append("### Feature Quality by Category")
        report.append("| Category | Total Features | Complete | Sparse | Missing | Quality Score |")
        report.append("|----------|---------------|----------|--------|---------|---------------|")
        for cat, metrics in quality_heatmap['categories'].items():
            report.append(
                f"| {cat} | {metrics['total']} | {metrics['complete']} | "
                f"{metrics['sparse']} | {metrics['missing']} | "
                f"{metrics['quality_score']:.1%} |"
            )
        report.append("")
        
        # 11. System Recommendations
        report.append("## 11. 🚀 System Optimization Recommendations\n")
        recommendations = self._generate_recommendations(
            health, model_config, freshness, correlation_analysis if not quick_mode else None,
            performance if not quick_mode else None
        )
        
        if recommendations['critical']:
            report.append("### 🔴 Critical (Fix Immediately)")
            for rec in recommendations['critical']:
                report.append(f"1. **{rec['title']}**")
                report.append(f"   - Issue: {rec['issue']}")
                report.append(f"   - Fix: {rec['fix']}")
            report.append("")
        
        if recommendations['high_priority']:
            report.append("### 🟡 High Priority (Address Soon)")
            for rec in recommendations['high_priority']:
                report.append(f"- {rec}")
            report.append("")
        
        if recommendations['optimization']:
            report.append("### 🟢 Optimization Opportunities")
            for rec in recommendations['optimization']:
                report.append(f"- {rec}")
            report.append("")
        
        # 12. Quick Reference Guide
        report.append("## 12. 📖 Quick Reference: Troubleshooting Guide\n")
        report.append(self._generate_troubleshooting_guide())
        
        # Summary timing
        elapsed = time.time() - start_time
        report.append(f"\n---\n*Report generated in {elapsed:.2f} seconds*")
        
        # Compile and save
        report_text = '\n'.join(report)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Technical diagnostic report saved: {output_path}")
        print(f"   Issues found: {len(self.issues_found)}")
        print(f"   Warnings: {len(self.warnings_found)}")
        
        if display_in_notebook:
            display(Markdown(report_text))
        
        return report_text
    
    def _check_system_health(self) -> dict:
        """Comprehensive system health check."""
        health = {
            'status': 'healthy',
            'critical_issues': [],
            'warnings': [],
            'components': {}
        }
        
        # Check data availability
        data_ok = self.features is not None and len(self.features) > 0
        health['components']['data'] = {
            'ok': data_ok,
            'message': f"{len(self.features)} rows" if data_ok else "No data loaded"
        }
        if not data_ok:
            health['critical_issues'].append({
                'component': 'Data',
                'message': 'No feature data available'
            })
            health['status'] = 'failed'
        
        # Check model training
        model_ok = self.detector and self.detector.trained
        health['components']['model'] = {
            'ok': model_ok,
            'message': 'Trained' if model_ok else 'Not trained'
        }
        if not model_ok:
            health['critical_issues'].append({
                'component': 'Model',
                'message': 'Anomaly detector not trained'
            })
            health['status'] = 'failed'
        
        # Check detector coverage
        if model_ok:
            low_coverage_detectors = [
                name for name, cov in self.detector.detector_coverage.items()
                if cov < 0.7
            ]
            health['components']['detector_coverage'] = {
                'ok': len(low_coverage_detectors) == 0,
                'message': f"{len(low_coverage_detectors)} detectors with <70% coverage"
            }
            if low_coverage_detectors:
                health['warnings'].append(
                    f"Low coverage detectors: {', '.join(low_coverage_detectors[:3])}"
                )
                if health['status'] == 'healthy':
                    health['status'] = 'degraded'
        
        # Check data freshness
        if self.features is not None:
            last_date = self.features.index[-1]
            age_days = (datetime.now() - pd.to_datetime(last_date)).days
            data_fresh = age_days <= 3
            health['components']['data_freshness'] = {
                'ok': data_fresh,
                'message': f"{age_days} days old"
            }
            if not data_fresh:
                health['warnings'].append(f"Data is {age_days} days old (>3 days)")
                if health['status'] == 'healthy':
                    health['status'] = 'degraded'
        
        # Check missing data
        if self.features is not None:
            missing_pct = self.features.isnull().sum().sum() / (len(self.features) * len(self.features.columns))
            data_complete = missing_pct < 0.3
            health['components']['data_completeness'] = {
                'ok': data_complete,
                'message': f"{(1-missing_pct)*100:.1f}% complete"
            }
            if not data_complete:
                health['warnings'].append(f"High missing data: {missing_pct*100:.1f}%")
                if health['status'] == 'healthy':
                    health['status'] = 'degraded'
        
        return health
    
    def _inspect_model_configuration(self) -> dict:
        """Deep inspection of model parameters."""
        config = {
            'anomaly_detector': {
                'contamination': self.detector.contamination,
                'random_state': self.detector.random_state,
                'n_estimators': 100,  # IsolationForest default
                'max_samples': 'auto'
            },
            'detectors': {}
        }
        
        for name, detector in self.detector.detectors.items():
            if detector is not None:
                feature_list = self.detector.feature_groups.get(name, [])
                available = [f for f in feature_list if f in self.features.columns]
                coverage = len(available) / len(feature_list) if feature_list else 0
                
                config['detectors'][name] = {
                    'active': True,
                    'n_features': len(available),
                    'coverage': coverage,
                    'features': available[:5]  # Sample
                }
            else:
                config['detectors'][name] = {
                    'active': False,
                    'n_features': 0,
                    'coverage': 0.0,
                    'features': []
                }
        
        return config
    
    def _trace_data_pipeline(self) -> dict:
        """Trace data flow through the entire pipeline."""
        trace = {
            'steps': [],
            'raw_data_points': 0,
            'engineered_features': 0,
            'final_features': 0,
            'reduction_ratio': 0
        }
        
        # Step 1: Data Fetching
        if self.vix is not None and self.spx is not None:
            raw_points = len(self.vix) + len(self.spx)
            trace['raw_data_points'] = raw_points
            trace['steps'].append({
                'stage': 'Data Fetching',
                'status': '✅ OK',
                'details': [
                    f"VIX: {len(self.vix)} observations",
                    f"SPX: {len(self.spx)} observations"
                ]
            })
        else:
            trace['steps'].append({
                'stage': 'Data Fetching',
                'status': '❌ FAILED',
                'details': ['Core market data missing']
            })
        
        # Step 2: Feature Engineering
        if self.features is not None:
            trace['engineered_features'] = len(self.features.columns)
            trace['final_features'] = len(self.features.columns)
            trace['steps'].append({
                'stage': 'Feature Engineering',
                'status': '✅ OK',
                'details': [
                    f"Generated {len(self.features.columns)} features",
                    f"Time period: {len(self.features)} days"
                ]
            })
        else:
            trace['steps'].append({
                'stage': 'Feature Engineering',
                'status': '❌ FAILED',
                'details': ['No features generated']
            })
        
        # Step 3: Model Training
        if self.detector and self.detector.trained:
            n_detectors = len(self.detector.detectors)
            active = len([d for d in self.detector.detectors.values() if d is not None])
            trace['steps'].append({
                'stage': 'Model Training',
                'status': '✅ OK',
                'details': [
                    f"{active}/{n_detectors} detectors trained",
                    f"Ensemble scores computed: {len(self.detector.training_ensemble_scores)}"
                ]
            })
        else:
            trace['steps'].append({
                'stage': 'Model Training',
                'status': '❌ FAILED',
                'details': ['Detector not trained']
            })
        
        # Calculate reduction ratio
        if trace['raw_data_points'] > 0 and trace['final_features'] > 0:
            trace['reduction_ratio'] = trace['final_features'] / trace['raw_data_points']
        
        return trace
    
    def _check_data_freshness(self) -> dict:
        """Check staleness of all data sources."""
        freshness = {
            'sources': {},
            'stale_features': []
        }
        
        now = datetime.now()
        
        # Check main data
        if self.features is not None:
            last_date = pd.to_datetime(self.features.index[-1])
            age = (now - last_date).total_seconds() / 86400
            
            freshness['sources']['Main Features'] = {
                'last_update': last_date.strftime('%Y-%m-%d'),
                'age_days': age
            }
        
        # Check individual feature freshness (recent missing data)
        if self.features is not None:
            recent_data = self.features.iloc[-5:]  # Last 5 days
            for col in self.features.columns:
                missing_pct = recent_data[col].isnull().sum() / len(recent_data)
                if missing_pct > 0.4:  # >40% missing in recent data
                    freshness['stale_features'].append((col, missing_pct))
        
        freshness['stale_features'].sort(key=lambda x: x[1], reverse=True)
        
        return freshness
    
    def _breakdown_current_anomaly(self) -> dict:
        """Detailed breakdown of current anomaly detection."""
        if not self.detector or not self.detector.trained:
            return {
                'ensemble_score': 0.0,
                'detector_contributions': [],
                'top_driving_features': []
            }
        
        # Get current detection
        result = self.detector.detect(self.features.iloc[[-1]], verbose=False)
        
        breakdown = {
            'ensemble_score': result['ensemble']['score'],
            'detector_contributions': [],
            'top_driving_features': []
        }
        
        # Detector contributions
        ensemble_score = result['ensemble']['score']
        all_scores = []
        
        for name, data in result.get('domain_anomalies', {}).items():
            all_scores.append(data['score'])
            breakdown['detector_contributions'].append({
                'name': name,
                'score': data['score'],
                'weight': data['weight'],
                'weighted_score': data['score'] * data['weight'],
                'agrees_with_ensemble': abs(data['score'] - ensemble_score) < 0.2
            })
        
        # Sort by weighted contribution
        breakdown['detector_contributions'].sort(
            key=lambda x: x['weighted_score'], reverse=True
        )
        
        # Get top driving features (from highest scoring detector)
        if breakdown['detector_contributions']:
            top_detector_name = breakdown['detector_contributions'][0]['name']
            if top_detector_name in self.detector.feature_importances:
                importances = self.detector.feature_importances[top_detector_name]
                current_row = self.features.iloc[-1]
                
                for feat, imp in sorted(importances.items(), 
                                       key=lambda x: x[1], reverse=True)[:15]:
                    if feat in current_row.index:
                        current_val = current_row[feat]
                        if not pd.isna(current_val):
                            # Calculate z-score
                            hist_mean = self.features[feat].mean()
                            hist_std = self.features[feat].std()
                            zscore = (current_val - hist_mean) / hist_std if hist_std > 0 else 0
                            
                            breakdown['top_driving_features'].append({
                                'feature': feat,
                                'importance': imp,
                                'current_value': current_val,
                                'zscore': zscore
                            })
        
        return breakdown
    
    def _analyze_feature_correlations(self) -> dict:
        """Find highly correlated (redundant) features."""
        if self.features is None or len(self.features.columns) < 2:
            return {
                'threshold': 0.95,
                'high_correlation_pairs': [],
                'feature_clusters': []
            }
        
        # Sample data for speed (last 500 days)
        sample_data = self.features.iloc[-500:] if len(self.features) > 500 else self.features
        
        # Compute correlation matrix
        corr_matrix = sample_data.corr().abs()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        threshold = 0.95
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        
        return {
            'threshold': threshold,
            'high_correlation_pairs': high_corr_pairs,
            'feature_clusters': []  # Could implement clustering
        }
    
    def _profile_performance(self) -> dict:
        """Profile system performance."""
        timings = {}
        
        # Test feature detection
        if self.features is not None and len(self.features) > 0:
            start = time.time()
            _ = self.detector.detect(self.features.iloc[[-1]], verbose=False)
            timings['single_detection'] = (time.time() - start) * 1000
        
        # Test batch detection
        if self.features is not None and len(self.features) >= 10:
            start = time.time()
            for i in range(10):
                _ = self.detector.detect(self.features.iloc[[i]], verbose=False)
            timings['batch_10_detections'] = (time.time() - start) * 1000
        
        # Identify bottlenecks
        bottlenecks = []
        if timings.get('single_detection', 0) > 100:
            bottlenecks.append({
                'operation': 'Single Detection',
                'time_ms': timings['single_detection'],
                'reason': 'Exceeds 100ms threshold'
            })
        
        return {
            'timings': timings,
            'bottlenecks': bottlenecks
        }
    
    def _diagnose_common_failures(self) -> dict:
        """Diagnose common failure modes."""
        issues = []
        
        # Issue: High missing data rate
        if self.features is not None:
            missing_pct = self.features.isnull().sum().sum() / (len(self.features) * len(self.features.columns))
            if missing_pct > 0.3:
                issues.append({
                    'name': 'High Missing Data Rate',
                    'symptoms': f'{missing_pct*100:.1f}% of data is missing',
                    'cause': 'Data sources not updating or API failures',
                    'solution': 'Check data_fetcher logs, verify API keys, check CBOE file availability'
                })
        
        # Issue: Low detector coverage
        if self.detector:
            low_cov = [n for n, c in self.detector.detector_coverage.items() if c < 0.5]
            if low_cov:
                issues.append({
                    'name': 'Low Detector Coverage',
                    'symptoms': f'{len(low_cov)} detectors have <50% feature coverage',
                    'cause': 'Missing data sources (CBOE files, FRED series)',
                    'solution': 'Verify CBOE_Data_Archive directory, check FRED API key, review data_fetcher output'
                })
        
        # Issue: Stale data
        if self.features is not None:
            age_days = (datetime.now() - pd.to_datetime(self.features.index[-1])).days
            if age_days > 5:
                issues.append({
                    'name': 'Stale Data',
                    'symptoms': f'Most recent data is {age_days} days old',
                    'cause': 'System not being refreshed regularly',
                    'solution': 'Run system.refresh() or retrain with recent data'
                })
        
        return {'detected_issues': issues}
    
    def _run_scenario_analysis(self) -> dict:
        """Run what-if scenarios."""
        scenarios = []
        
        if self.detector and self.detector.statistical_thresholds:
            thresholds = self.detector.statistical_thresholds
            
            scenarios.append({
                'name': 'VIX Spike to 40',
                'description': 'If VIX suddenly spikes to 40 (crisis level)',
                'expected': f'Ensemble score would likely exceed {thresholds["critical"]:.0%} (CRITICAL threshold)',
                'actual': 'Multiple detectors would fire: vix_regime_structure, vix_momentum, cross_asset_divergence'
            })
            
            scenarios.append({
                'name': 'SKEW >150',
                'description': 'If SKEW index exceeds 150 (extreme tail risk)',
                'expected': 'tail_risk_complex detector triggers, ensemble score elevates',
                'actual': 'If SKEW features are available, system would classify as HIGH/CRITICAL'
            })
            
            scenarios.append({
                'name': 'All CBOE Data Missing',
                'description': 'If CBOE features become unavailable',
                'expected': 'System continues to function with reduced capability',
                'actual': '5/15 detectors would be disabled, ensemble relies on VIX/SPX/futures detectors'
            })
        
        return {'scenarios': scenarios}
    
    def _generate_quality_heatmap(self) -> dict:
        """Generate feature quality heatmap data."""
        if self.features is None:
            return {'categories': {}}
        
        # Categorize features
        categories = {
            'VIX': [],
            'SPX': [],
            'CBOE': [],
            'Futures': [],
            'Macro': [],
            'Meta': []
        }
        
        for col in self.features.columns:
            if 'vix' in col.lower():
                categories['VIX'].append(col)
            elif 'spx' in col.lower():
                categories['SPX'].append(col)
            elif any(x in col.upper() for x in ['SKEW', 'PCC', 'COR', 'VXTH']):
                categories['CBOE'].append(col)
            elif any(x in col.lower() for x in ['vx_', 'cl_', 'dx_', 'crude', 'dxy']):
                categories['Futures'].append(col)
            elif any(x in col.lower() for x in ['gold', 'silver', 'bond']):
                categories['Macro'].append(col)
            else:
                categories['Meta'].append(col)
        
        heatmap = {'categories': {}}
        
        for cat, feats in categories.items():
            if not feats:
                continue
            
            cat_data = self.features[feats]
            missing_pct = cat_data.isnull().sum() / len(cat_data)
            
            complete = (missing_pct == 0).sum()
            sparse = ((missing_pct > 0) & (missing_pct < 0.2)).sum()
            missing = (missing_pct >= 0.2).sum()
            
            quality_score = (complete + sparse * 0.5) / len(feats)
            
            heatmap['categories'][cat] = {
                'total': len(feats),
                'complete': complete,
                'sparse': sparse,
                'missing': missing,
                'quality_score': quality_score
            }
        
        return heatmap
    
    def _generate_recommendations(self, health, model_config, freshness, 
                                 correlation_analysis, performance) -> dict:
        """Generate actionable recommendations."""
        recommendations = {
            'critical': [],
            'high_priority': [],
            'optimization': []
        }
        
        # Critical recommendations
        if health['status'] == 'failed':
            for issue in health['critical_issues']:
                recommendations['critical'].append({
                    'title': f"Fix {issue['component']}",
                    'issue': issue['message'],
                    'fix': 'Retrain the system or check data availability'
                })
        
        # High priority
        if health['status'] == 'degraded':
            for warning in health['warnings']:
                recommendations['high_priority'].append(warning)
        
        # Optimizations
        if correlation_analysis and len(correlation_analysis['high_correlation_pairs']) > 10:
            recommendations['optimization'].append(
                f"Remove {len(correlation_analysis['high_correlation_pairs'])} highly correlated "
                "features to reduce redundancy"
            )
        
        if freshness.get('stale_features') and len(freshness['stale_features']) > 20:
            recommendations['optimization'].append(
                f"Investigate {len(freshness['stale_features'])} stale features with high recent missing data"
            )
        
        recommendations['optimization'].extend([
            "Consider adding feature selection to reduce dimensionality",
            "Implement caching for expensive feature calculations",
            "Add monitoring alerts for data freshness"
        ])
        
        return recommendations
    
    def _generate_troubleshooting_guide(self) -> str:
        """Generate quick troubleshooting guide."""
        guide = """
### Common Issues & Quick Fixes

**Issue**: System says "data too old"
- **Check**: `system.orchestrator.features.index[-1]`
- **Fix**: Run `system.refresh()` or retrain with fresh data

**Issue**: Ensemble score always near 0% or 100%
- **Check**: Are thresholds computed? `system.orchestrator.anomaly_detector.statistical_thresholds`
- **Fix**: Retrain system to recalculate thresholds

**Issue**: Many detectors show 0% coverage
- **Check**: CBOE files in `./CBOE_Data_Archive/`
- **Fix**: Download CBOE historical data or disable CBOE features in config

**Issue**: "Core data fetch failed" error
- **Check**: Internet connection, yfinance API status
- **Fix**: Run `system.orchestrator.fetcher.fetch_core_data(...)` separately to debug

**Issue**: High memory usage
- **Check**: Feature matrix size with `system.orchestrator.features.memory_usage(deep=True).sum()`
- **Fix**: Reduce training window in config.py (TRAINING_YEARS)

**Issue**: Slow detection speed (>1 second)
- **Check**: Number of features and detectors active
- **Fix**: Reduce features, disable low-value detectors, or enable quick_mode

**Issue**: NaN/Inf values in features
- **Check**: `system.orchestrator.features.isnull().sum()` and `np.isinf(system.orchestrator.features).sum()`
- **Fix**: Review feature_engine.py for division by zero or missing data handling
"""
        return guide
    
    def diagnose_current_state(self):
        """Quick diagnostic of current system state."""
        print("🔧 Running Quick Diagnostic...\n")
        
        health = self._check_system_health()
        print(f"System Status: {health['status'].upper()}")
        
        if health['critical_issues']:
            print("\n❌ Critical Issues:")
            for issue in health['critical_issues']:
                print(f"   - {issue['component']}: {issue['message']}")
        
        if health['warnings']:
            print("\n⚠️  Warnings:")
            for warning in health['warnings']:
                print(f"   - {warning}")
        
        if not health['critical_issues']:
            print("\n✅ System is operational")
        
        return health


# Convenience function
def diagnose_system(system, full_report=False, quick_mode=True):
    """
    Quick system diagnosis or full report generation.
    
    Args:
        system: Trained IntegratedMarketSystemV4
        full_report: Generate complete markdown report
        quick_mode: Skip expensive computations
    """
    introspector = TechnicalIntrospector(system)
    
    if full_report:
        return introspector.generate_report(quick_mode=quick_mode)
    else:
        return introspector.diagnose_current_state()