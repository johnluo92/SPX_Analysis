"""Consolidated Anomaly Detection System"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

from config import RANDOM_STATE, ANOMALY_THRESHOLDS, ANOMALY_FEATURE_GROUPS

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    warnings.warn("pytz not available - persistence streak timing will not be timezone-aware")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class MultiDimensionalAnomalyDetector:
    """15 independent Isolation Forests with feature importance."""
    
    def __init__(self, contamination: float = 0.01, random_state: int = RANDOM_STATE):
        self.contamination = contamination
        self.random_state = random_state
        self.detectors = {}
        self.scalers = {}
        self.feature_groups = {}
        self.random_subspaces = []
        self.training_distributions = {}
        self.feature_importances = {}
        self.detector_coverage = {}
        self.trained = False
        self.training_ensemble_scores = []
        self.statistical_thresholds = None
        self.importance_config = {
            'method': 'shap' if SHAP_AVAILABLE else 'permutation',
            'n_samples': 500,
            'n_repeats': 1,
        }
        
    def _define_feature_groups(self):
        self.feature_groups = ANOMALY_FEATURE_GROUPS.copy()
    
    def _calculate_feature_importance(self, detector, X_scaled, baseline_scores, feature_names, verbose=False):
        if SHAP_AVAILABLE and self.importance_config['method'] == 'shap':
            try:
                return self._calculate_shap_importance(detector, X_scaled, feature_names, verbose)
            except Exception as e:
                if verbose:
                    warnings.warn(f"SHAP failed, using permutation: {str(e)[:100]}")
        return self._calculate_permutation_importance(detector, X_scaled, baseline_scores, feature_names, verbose)
    
    def _calculate_shap_importance(self, detector, X_scaled, feature_names, verbose=False):
        n_samples = min(self.importance_config['n_samples'], len(X_scaled))
        sample_indices = np.random.RandomState(self.random_state).choice(len(X_scaled), n_samples, replace=False)
        X_sample = X_scaled[sample_indices]
        explainer = shap.TreeExplainer(detector)
        shap_values = explainer.shap_values(X_sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total = mean_abs_shap.sum()
        normalized_importance = mean_abs_shap / total if total > 0 else np.ones(len(feature_names)) / len(feature_names)
        return {name: float(imp) for name, imp in zip(feature_names, normalized_importance)}
    
    def _calculate_permutation_importance(self, detector, X_scaled, baseline_scores, feature_names, verbose=False):
        try:
            if len(X_scaled) == 0 or len(feature_names) == 0 or len(X_scaled) != len(baseline_scores):
                return {name: 1.0/len(feature_names) for name in feature_names}
            
            n_samples = min(self.importance_config['n_samples'], len(X_scaled))
            if n_samples < len(X_scaled):
                sample_indices = np.random.RandomState(self.random_state).choice(len(X_scaled), n_samples, replace=False)
                X_sample = X_scaled[sample_indices].copy()
                baseline_mean = np.mean(baseline_scores[sample_indices])
            else:
                X_sample = X_scaled.copy()
                baseline_mean = np.mean(baseline_scores)
            
            importances = {}
            for i, feature_name in enumerate(feature_names):
                try:
                    original_col = X_sample[:, i].copy()
                    importance_values = []
                    for repeat in range(self.importance_config['n_repeats']):
                        np.random.RandomState(self.random_state + i + repeat * 1000).shuffle(X_sample[:, i])
                        permuted_scores = detector.score_samples(X_sample)
                        importance_values.append(abs(baseline_mean - np.mean(permuted_scores)))
                        X_sample[:, i] = original_col
                    importances[feature_name] = np.mean(importance_values)
                except:
                    importances[feature_name] = 0.0
            
            total = sum(importances.values())
            if total > 0:
                importances = {k: v/total for k, v in importances.items()}
            else:
                importances = {name: 1.0/len(feature_names) for name in feature_names}
            
            return {k: v if np.isfinite(v) else 1.0/len(feature_names) for k, v in importances.items()}
        except:
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    def calculate_statistical_thresholds(self) -> dict:
        """Calculate thresholds from historical ensemble scores."""
        if len(self.training_ensemble_scores) == 0:
            return {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}
        
        scores = np.array(self.training_ensemble_scores)
        thresholds = {
            'moderate': float(np.percentile(scores, 85)),
            'high': float(np.percentile(scores, 92)),
            'critical': float(np.percentile(scores, 98))
        }
        
        print(f"\n📊 Statistical Thresholds:")
        print(f"   Moderate (85th): {thresholds['moderate']:.4f}")
        print(f"   High (92nd):     {thresholds['high']:.4f}")
        print(f"   Critical (98th): {thresholds['critical']:.4f}")
        
        return thresholds
    
    def calculate_statistical_thresholds_with_ci(self, n_bootstrap: int = 1000, confidence_level: float = 0.95) -> dict:
        """Calculate thresholds with bootstrap confidence intervals."""
        if len(self.training_ensemble_scores) == 0:
            return self.calculate_statistical_thresholds()
        
        scores = np.array(self.training_ensemble_scores)
        n_samples = len(scores)
        
        bootstrap_moderate, bootstrap_high, bootstrap_critical = [], [], []
        
        print(f"\n📊 Computing Bootstrap CIs...")
        print(f"   Samples: {n_samples} | Iterations: {n_bootstrap} | CI: {confidence_level*100:.0f}%")
        
        rng = np.random.RandomState(self.random_state)
        for i in range(n_bootstrap):
            bootstrap_sample = rng.choice(scores, size=n_samples, replace=True)
            bootstrap_moderate.append(np.percentile(bootstrap_sample, 85))
            bootstrap_high.append(np.percentile(bootstrap_sample, 92))
            bootstrap_critical.append(np.percentile(bootstrap_sample, 98))
            
            if (i + 1) % 250 == 0:
                print(f"   Progress: {i + 1}/{n_bootstrap}...")
        
        moderate_point = float(np.percentile(scores, 85))
        high_point = float(np.percentile(scores, 92))
        critical_point = float(np.percentile(scores, 98))
        
        alpha = 1 - confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100
        
        thresholds = {
            'moderate': moderate_point,
            'moderate_ci': {
                'lower': float(np.percentile(bootstrap_moderate, lower_pct)),
                'upper': float(np.percentile(bootstrap_moderate, upper_pct)),
                'std': float(np.std(bootstrap_moderate))
            },
            'high': high_point,
            'high_ci': {
                'lower': float(np.percentile(bootstrap_high, lower_pct)),
                'upper': float(np.percentile(bootstrap_high, upper_pct)),
                'std': float(np.std(bootstrap_high))
            },
            'critical': critical_point,
            'critical_ci': {
                'lower': float(np.percentile(bootstrap_critical, lower_pct)),
                'upper': float(np.percentile(bootstrap_critical, upper_pct)),
                'std': float(np.std(bootstrap_critical))
            },
            'bootstrap_config': {
                'n_iterations': n_bootstrap,
                'confidence_level': confidence_level,
                'n_samples': n_samples
            }
        }
        
        print(f"\n📊 Statistical Thresholds with {confidence_level*100:.0f}% CIs:")
        print(f"   Moderate: {moderate_point:.4f} [{thresholds['moderate_ci']['lower']:.4f}, {thresholds['moderate_ci']['upper']:.4f}]")
        print(f"   High:     {high_point:.4f} [{thresholds['high_ci']['lower']:.4f}, {thresholds['high_ci']['upper']:.4f}]")
        print(f"   Critical: {critical_point:.4f} [{thresholds['critical_ci']['lower']:.4f}, {thresholds['critical_ci']['upper']:.4f}]")
        
        moderate_cv = thresholds['moderate_ci']['std'] / moderate_point * 100
        high_cv = thresholds['high_ci']['std'] / high_point * 100
        critical_cv = thresholds['critical_ci']['std'] / critical_point * 100
        
        print(f"\n📈 Threshold Stability (CV%):")
        print(f"   Moderate: {moderate_cv:.2f}% | High: {high_cv:.2f}% | Critical: {critical_cv:.2f}%")
        
        if max(moderate_cv, high_cv, critical_cv) > 5.0:
            warnings.warn("⚠️ High threshold variability (CV > 5%). Consider increasing training data.")
        
        return thresholds
    
    def train(self, features: pd.DataFrame, verbose: bool = True):
        self._define_feature_groups()
        if verbose:
            print(f"\n{'='*80}\nMULTI-DIMENSIONAL ANOMALY DETECTOR\n{'='*80}")
            print(f"Features: {len(features.columns)} | Samples: {len(features)} | Contamination: {self.contamination:.1%}")
            print(f"Attribution: {'SHAP' if SHAP_AVAILABLE else 'Permutation'}")
            print(f"\n{'-'*80}\nPHASE 1: Training 10 Domain Detectors\n{'-'*80}")
        
        for name, feature_list in self.feature_groups.items():
            available_features = [f for f in feature_list if f in features.columns]
            if len(available_features) < 3:
                if verbose:
                    print(f"\n⚠️ {name}: Skipped ({len(available_features)} features)")
                continue
            
            try:
                X = features[available_features].fillna(0)
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                detector = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_estimators=100,
                    max_samples='auto',
                    n_jobs=-1
                )
                detector.fit(X_scaled)
                training_scores = detector.score_samples(X_scaled)
                self.training_distributions[name] = training_scores
                
                try:
                    self.feature_importances[name] = self._calculate_feature_importance(
                        detector, X_scaled, training_scores, available_features, verbose=False
                    )
                except:
                    self.feature_importances[name] = {f: 1.0/len(available_features) for f in available_features}
                
                self.detectors[name] = detector
                self.scalers[name] = scaler
                
                if verbose:
                    print(f"\n✅ {name}: {len(available_features)} features | Range: [{training_scores.min():.3f}, {training_scores.max():.3f}]")
            except Exception as e:
                if verbose:
                    print(f"\n❌ {name}: Failed - {e}")
                continue
        
        if verbose:
            print(f"\n{'-'*80}\nPHASE 2: Training 5 Random Subspace Detectors\n{'-'*80}")
        
        all_features = features.columns.tolist()
        for i in range(5):
            detector_name = f'random_{i+1}'
            try:
                random_features = np.random.choice(all_features, size=min(25, len(all_features)), replace=False).tolist()
                self.random_subspaces.append(random_features)
                X = features[random_features].fillna(0)
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                detector = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state + i,
                    n_estimators=100,
                    max_samples='auto',
                    n_jobs=-1
                )
                detector.fit(X_scaled)
                training_scores = detector.score_samples(X_scaled)
                self.training_distributions[detector_name] = training_scores
                
                try:
                    self.feature_importances[detector_name] = self._calculate_feature_importance(
                        detector, X_scaled, training_scores, random_features, verbose=False
                    )
                except:
                    self.feature_importances[detector_name] = {f: 1.0/len(random_features) for f in random_features}
                
                self.detectors[detector_name] = detector
                self.scalers[detector_name] = scaler
                
                if verbose:
                    print(f"\n✅ Random #{i+1}: {len(random_features)} features | Range: [{training_scores.min():.3f}, {training_scores.max():.3f}]")
            except Exception as e:
                if verbose:
                    print(f"\n❌ Random #{i+1}: Failed - {e}")
                continue
        
        self.detector_coverage = {}
        for name, detector in self.detectors.items():
            if name.startswith('random_'):
                idx = int(name.split('_')[1]) - 1
                required_features = self.random_subspaces[idx]
            else:
                required_features = self.feature_groups[name]
            
            available_features = [f for f in required_features if f in features.columns]
            coverage = len(available_features) / len(required_features) if len(required_features) > 0 else 0.0
            self.detector_coverage[name] = coverage
        
        if verbose:
            print(f"\n{'-'*80}\nDetector Coverage\n{'-'*80}")
            for name, coverage in sorted(self.detector_coverage.items(), key=lambda x: x[1], reverse=True):
                status = "✅" if coverage >= 0.8 else "⚠️" if coverage >= 0.5 else "❌"
                print(f"{status} {name:30s}: {coverage:5.1%}")
        
        if len(self.detectors) < 5:
            raise RuntimeError(f"Training failed: only {len(self.detectors)} detectors trained (min 5 required)")
        
        self.trained = True
        
        if verbose:
            print(f"\n{'-'*80}\nComputing statistical thresholds...\n{'-'*80}")
        
        self._compute_statistical_thresholds(features, verbose=verbose)
        
        if verbose:
            print(f"\n{'-'*80}\nCalculating Bootstrap CIs\n{'-'*80}")
        
        self.statistical_thresholds = self.calculate_statistical_thresholds_with_ci(n_bootstrap=1000, confidence_level=0.95)
        
        if verbose:
            print(f"\n{'='*80}\n✅ TRAINING COMPLETE - {len(self.detectors)} DETECTORS")
            print(f"   Domain: {len([k for k in self.detectors.keys() if not k.startswith('random_')])}")
            print(f"   Random: {len([k for k in self.detectors.keys() if k.startswith('random_')])}\n{'='*80}")
    
    def detect(self, features: pd.DataFrame, verbose: bool = False) -> dict:
        if not self.trained:
            raise ValueError("Detector not trained")
        
        results = {
            'ensemble': {},
            'domain_anomalies': {},
            'random_anomalies': {},
            'data_quality': {}
        }
        
        scores, detector_weights, active_detectors, missing_features = [], [], 0, []
        
        for name, detector in self.detectors.items():
            if name.startswith('random_'):
                idx = int(name.split('_')[1]) - 1
                required_features = self.random_subspaces[idx]
            else:
                required_features = [f for f in self.feature_groups[name] if f in features.columns]
            
            available = [f for f in required_features if f in features.columns]
            coverage = len(available) / len(required_features) if len(required_features) > 0 else 0.0
            
            if coverage < ANOMALY_THRESHOLDS['detector_coverage_min']:
                missing_features.append({'detector': name, 'coverage': coverage, 'reason': 'below_threshold'})
                continue
            
            try:
                X = features[available].fillna(0)
                X_scaled = self.scalers[name].transform(X)
                raw_score = detector.score_samples(X_scaled)[0]
                
                training_dist = self.training_distributions[name]
                percentile = (training_dist <= raw_score).sum() / len(training_dist)
                anomaly_score = 1.0 - percentile
                
                scores.append(anomaly_score)
                detector_weights.append(coverage)
                active_detectors += 1
                
                level, _, _ = self.classify_anomaly(anomaly_score, method='statistical')
                
                result_dict = {
                    'score': float(anomaly_score),
                    'percentile': float(anomaly_score * 100),
                    'level': level,
                    'coverage': float(coverage),
                    'weight': float(coverage)
                }
                
                if name.startswith('random_'):
                    results['random_anomalies'][name] = result_dict
                else:
                    results['domain_anomalies'][name] = result_dict
            except Exception as e:
                missing_features.append({'detector': name, 'coverage': coverage, 'error': str(e)})
        
        if scores:
            weights = np.array(detector_weights)
            weighted_scores = np.array(scores) * weights
            ensemble_score = float(weighted_scores.sum() / weights.sum()) if weights.sum() > 0 else float(np.mean(scores))
            
            results['ensemble'] = {
                'score': ensemble_score,
                'std': float(np.std(scores)),
                'max_anomaly': float(np.max(scores)),
                'min_anomaly': float(np.min(scores)),
                'n_detectors': active_detectors,
                'mean_weight': float(np.mean(weights)),
                'weighted': True
            }
        
        results['data_quality'] = {
            'active_detectors': active_detectors,
            'total_detectors': len(self.detectors),
            'missing_features': missing_features,
            'weight_stats': {
                'mean': float(np.mean(weights)) if len(weights) > 0 else 0.0,
                'min': float(np.min(weights)) if len(weights) > 0 else 0.0,
                'max': float(np.max(weights)) if len(weights) > 0 else 0.0
            }
        }
        
        return results
    
    def calculate_historical_persistence_stats(
        self, 
        ensemble_scores: np.ndarray, 
        dates: Optional[pd.DatetimeIndex] = None,
        threshold: float = None
    ) -> Dict:
        """Calculate persistence stats from complete historical ensemble scores."""
        if threshold is None:
            if hasattr(self, 'statistical_thresholds') and self.statistical_thresholds is not None:
                threshold = self.statistical_thresholds.get('high', 0.78) if isinstance(self.statistical_thresholds, dict) else 0.78
            else:
                threshold = 0.78
        
        if isinstance(ensemble_scores, list):
            ensemble_scores = np.array(ensemble_scores)
        
        is_anomalous = ensemble_scores >= threshold
        
        streaks, current_streak = [], 0
        for anomalous in is_anomalous:
            if anomalous:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if dates is not None and current_streak > 0 and PYTZ_AVAILABLE:
            try:
                last_date = pd.Timestamp(dates[-1])
                et_tz = pytz.timezone('US/Eastern')
                now_et = datetime.now(et_tz)
                today_et = pd.Timestamp.now(tz='US/Eastern').normalize()
                
                last_date_normalized = last_date.normalize()
                if hasattr(last_date, 'tz') and last_date.tz is not None:
                    last_date_normalized = last_date.tz_convert('US/Eastern').normalize()
                
                if last_date_normalized == today_et:
                    market_closed = now_et.hour >= 16
                    if not market_closed and is_anomalous[-1]:
                        current_streak = max(0, current_streak - 1)
            except Exception as e:
                warnings.warn(f"Streak timing correction failed: {e}")
        
        total_anomaly_days = int(is_anomalous.sum())
        
        return {
            'current_streak': int(current_streak),
            'mean_duration': float(np.mean(streaks)) if streaks else 0.0,
            'max_duration': int(np.max(streaks)) if streaks else 0,
            'median_duration': float(np.median(streaks)) if streaks else 0.0,
            'total_anomaly_days': total_anomaly_days,
            'anomaly_rate': float(total_anomaly_days / len(ensemble_scores)) if len(ensemble_scores) > 0 else 0.0,
            'num_episodes': len(streaks),
            'threshold_used': float(threshold)
        }
    
    def get_top_anomalies(self, result: dict, top_n: int = 3) -> list:
        all_anomalies = []
        for name, data in result.get('domain_anomalies', {}).items():
            all_anomalies.append((name, data['score']))
        for name, data in result.get('random_anomalies', {}).items():
            all_anomalies.append((name, data['score']))
        all_anomalies.sort(key=lambda x: x[1], reverse=True)
        return all_anomalies[:top_n]
    
    def get_feature_contributions(self, detector_name: str, top_n: int = 5) -> List[Tuple[str, float]]:
        if detector_name not in self.feature_importances:
            return []
        importances = self.feature_importances[detector_name]
        return sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def _compute_statistical_thresholds(self, features: pd.DataFrame, verbose: bool = True):
        self.training_ensemble_scores = []
        batch_size = 100
        for i in range(0, len(features), batch_size):
            batch = features.iloc[i:i+batch_size]
            for idx in range(len(batch)):
                try:
                    result = self.detect(batch.iloc[[idx]], verbose=False)
                    self.training_ensemble_scores.append(result['ensemble']['score'])
                except:
                    pass
        
        if verbose:
            print(f"\n✅ Computed {len(self.training_ensemble_scores)} ensemble scores for threshold calculation")
    
    def classify_anomaly(self, score: float, method: str = 'statistical') -> tuple:
        """Classify anomaly severity using statistical thresholds."""
        if self.statistical_thresholds is None:
            thresholds = {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}
        else:
            if 'moderate_ci' in self.statistical_thresholds:
                thresholds = {
                    'moderate': self.statistical_thresholds['moderate'],
                    'high': self.statistical_thresholds['high'],
                    'critical': self.statistical_thresholds['critical']
                }
            else:
                thresholds = self.statistical_thresholds
        
        if len(self.training_ensemble_scores) > 0:
            p_value = (np.array(self.training_ensemble_scores) >= score).mean()
            confidence = 1 - p_value
        else:
            p_value = None
            confidence = None
        
        if score >= thresholds['critical']:
            level = 'CRITICAL'
        elif score >= thresholds['high']:
            level = 'HIGH'
        elif score >= thresholds['moderate']:
            level = 'MODERATE'
        else:
            level = 'NORMAL'
        
        return level, p_value, confidence