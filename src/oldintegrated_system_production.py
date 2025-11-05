"""Integrated Market Prediction System V4 - Production + Memory Profiling"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json
import os
import gc

warnings.filterwarnings('ignore')

from core.feature_engine import UnifiedFeatureEngine
from core.predictor import VIXPredictorV4, REGIME_BOUNDARIES, REGIME_NAMES
from config import TRAINING_YEARS, ENABLE_TRAINING
from export.unified_exporter import UnifiedExporter
from generate_claude_package import generate_claude_package


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available - memory profiling disabled. Install: pip install psutil")


class IntegratedMarketSystemV4:
    def __init__(self, cboe_data_dir: str = "./CBOE_Data_Archive"):
        self.feature_engine = UnifiedFeatureEngine(cboe_data_dir)
        self.vix_predictor = VIXPredictorV4(cboe_data_dir)
        self.trained = False
        self.cv_results = {}
        self._cached_anomaly_result = None
        self._cache_timestamp = None
        # self.spx = None
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
            self.baseline_memory_mb = None
            self.memory_history = []
            self.memory_warning_threshold_mb = 50
            self.memory_critical_threshold_mb = 200
            self.memory_monitoring_enabled = True
        else:
            self.memory_monitoring_enabled = False
    
    def _initialize_memory_baseline(self):
        """Establish memory baseline after initialization."""
        if not self.memory_monitoring_enabled:
            return
        
        try:
            gc.collect()
            mem_info = self.process.memory_info()
            self.baseline_memory_mb = mem_info.rss / (1024 * 1024)
            self.memory_history.append({
                'timestamp': datetime.now().isoformat(),
                'memory_mb': self.baseline_memory_mb,
                'type': 'baseline'
            })
            
            print(f"\n📊 Memory Baseline: {self.baseline_memory_mb:.1f} MB")
            print(f"   RSS: {mem_info.rss / (1024*1024):.1f} MB | VMS: {mem_info.vms / (1024*1024):.1f} MB")
        except Exception as e:
            warnings.warn(f"Memory baseline initialization failed: {e}")
            self.memory_monitoring_enabled = False
    
    def _log_memory_stats(self, context: str = "refresh") -> dict:
        """Log current memory usage and detect growth."""
        if not self.memory_monitoring_enabled:
            return {}
        
        try:
            mem_info = self.process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
            
            if self.baseline_memory_mb is None:
                self._initialize_memory_baseline()
                return {}
            
            growth = current_mb - self.baseline_memory_mb
            
            recent_history = self.memory_history[-10:] if len(self.memory_history) > 0 else []
            if len(recent_history) > 0:
                recent_avg = sum([h['memory_mb'] for h in recent_history]) / len(recent_history)
                trend = current_mb - recent_avg
            else:
                trend = 0.0
            
            gc_collections = gc.get_count()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'current_mb': float(current_mb),
                'baseline_mb': float(self.baseline_memory_mb),
                'growth_mb': float(growth),
                'vms_mb': float(vms_mb),
                'recent_trend_mb': float(trend),
                'gc_collections': gc_collections,
                'context': context
            }
            
            self.memory_history.append({
                'timestamp': stats['timestamp'],
                'memory_mb': current_mb,
                'type': context
            })
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-1000:]
            
            if growth > self.memory_critical_threshold_mb:
                print(f"\n🚨 CRITICAL MEMORY GROWTH: +{growth:.1f} MB")
                print(f"   Current: {current_mb:.1f} MB | Baseline: {self.baseline_memory_mb:.1f} MB")
                print(f"   Recent trend: {trend:+.1f} MB | Context: {context}")
                print(f"   ⚠️ Consider restarting the system")
            elif growth > self.memory_warning_threshold_mb:
                print(f"\n⚠️ Memory growth: +{growth:.1f} MB (current: {current_mb:.1f} MB) | Context: {context}")
            
            return stats
        except Exception as e:
            warnings.warn(f"Memory stats logging failed: {e}")
            return {}
    
    def get_memory_report(self) -> dict:
        """Generate detailed memory report for diagnostics."""
        if not self.memory_monitoring_enabled:
            return {'error': 'Memory monitoring not available (psutil not installed)'}
        
        try:
            mem_info = self.process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            growth_mb = current_mb - self.baseline_memory_mb if self.baseline_memory_mb else 0.0
            
            gc.collect()
            objects = gc.get_objects()
            
            type_counts = {}
            for obj in objects:
                obj_type = type(obj).__name__
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if len(self.memory_history) > 1:
                first_mb = self.memory_history[0]['memory_mb']
                last_mb = self.memory_history[-1]['memory_mb']
                total_growth = last_mb - first_mb
                measurements = len(self.memory_history)
                avg_growth_per_measurement = total_growth / measurements if measurements > 1 else 0.0
            else:
                total_growth = 0.0
                measurements = len(self.memory_history)
                avg_growth_per_measurement = 0.0
            
            return {
                'current_mb': float(current_mb),
                'baseline_mb': float(self.baseline_memory_mb) if self.baseline_memory_mb else None,
                'growth_mb': float(growth_mb),
                'vms_mb': float(mem_info.vms / (1024 * 1024)),
                'status': self._get_memory_status(growth_mb),
                'history': {
                    'measurements': measurements,
                    'total_growth_mb': float(total_growth),
                    'avg_growth_per_cycle': float(avg_growth_per_measurement),
                    'recent_samples': self.memory_history[-10:] if len(self.memory_history) > 0 else []
                },
                'gc_stats': {
                    'collections': gc.get_count(),
                    'tracked_objects': len(objects),
                    'top_types': [{'type': t, 'count': c} for t, c in top_types]
                },
                'thresholds': {
                    'warning_mb': self.memory_warning_threshold_mb,
                    'critical_mb': self.memory_critical_threshold_mb
                }
            }
        except Exception as e:
            return {'error': f'Memory report generation failed: {e}'}
    
    def _get_memory_status(self, growth_mb: float) -> str:
        """Classify memory status based on growth."""
        if growth_mb > self.memory_critical_threshold_mb:
            return 'CRITICAL'
        elif growth_mb > self.memory_warning_threshold_mb:
            return 'WARNING'
        else:
            return 'NORMAL'
    
    def train(self, years: int = TRAINING_YEARS, real_time_vix: bool = True, verbose: bool = False):
        print(f"\n{'='*80}\nINTEGRATED MARKET SYSTEM V4\n{'='*80}")
        print(f"Training: {years}y | Real-Time: {real_time_vix}")
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="pre-training")
        
        print("\n[1/2] Building features...")
        feature_data = self.feature_engine.build_complete_features(years=years)
        features, vix, dates = feature_data['features'], feature_data['vix'], feature_data['dates']
        self.vix_predictor.spx_ml = feature_data['spx']

        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-feature-engineering")
        
        vix_history_all = self.vix_predictor.fetcher.fetch_vix('1990-01-02', datetime.now().strftime('%Y-%m-%d'), lookback_buffer_days=0)
        
        print("\n[2/2] Training VIX Predictor...")
        self.vix_predictor.train_with_features(features=features, vix=vix, spx=self.vix_predictor.spx_ml, vix_history_all=vix_history_all, verbose=verbose)

        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-training")
        
        if real_time_vix:
            try:
                live_vix = self.vix_predictor.fetcher.fetch_price('^VIX')
                if live_vix:
                    self.vix_predictor.vix.iloc[-1] = live_vix
                    print(f"✅ Live VIX: {live_vix:.2f}")
            except:
                print("⚠️ Live VIX unavailable")
        
        self._verify_feature_coverage(features, verbose)
        self.trained = True
        
        if self.memory_monitoring_enabled and self.baseline_memory_mb is None:
            self._initialize_memory_baseline()
        
        print(f"\n{'='*80}\n✅ TRAINING COMPLETE\n{'='*80}")
        return self.cv_results
    
    def _verify_feature_coverage(self, features: pd.DataFrame, verbose: bool = False):
        from config import ANOMALY_FEATURE_GROUPS
        print("\n📋 FEATURE COVERAGE:")
        all_ok = True
        for domain, expected in ANOMALY_FEATURE_GROUPS.items():
            available = [f for f in expected if f in features.columns]
            coverage = len(available) / len(expected) * 100
            status = "✅" if coverage > 80 else ("⚠️" if coverage > 50 else "❌")
            if coverage <= 80:
                all_ok = False
            print(f"   {status} {domain:30s} {len(available):3d}/{len(expected):3d} ({coverage:5.1f}%)")
            if verbose and coverage < 100:
                missing = [f for f in expected if f not in features.columns][:5]
                print(f"      Missing: {', '.join(missing)}")
        print("\n✅ All detectors operational" if all_ok else "\n⚠️ Some detectors limited")
    
    def _get_cached_anomaly_result(self, force_refresh: bool = False):
        """Get cached anomaly result with optional force refresh."""
        if not self.vix_predictor.anomaly_detector:
            return None
        
        if force_refresh or self._cached_anomaly_result is None:
            if self.memory_monitoring_enabled and force_refresh:
                self._log_memory_stats(context="pre-anomaly-detection")
            
            self._cached_anomaly_result = self.vix_predictor.anomaly_detector.detect(
                self.vix_predictor.features.iloc[[-1]], 
                verbose=False
            )
            self._cache_timestamp = datetime.now()
            
            if self.memory_monitoring_enabled and force_refresh:
                self._log_memory_stats(context="post-anomaly-detection")
        
        return self._cached_anomaly_result

    def _recalculate_live_features(self, live_vix: float, live_spx: float):
        """Recalculate derived features for the last row after live price updates."""
        
        # Update base prices
        self.vix_predictor.vix_ml.iloc[-1] = live_vix
        self.vix_predictor.vix.iloc[-1] = live_vix
        self.vix_predictor.spx_ml.iloc[-1] = live_spx
        
        # Get last row index
        idx = self.vix_predictor.features.index[-1]
        
        # Recalculate VIX mean reversion features
        for w in [10, 21, 63, 126, 252]:
            ma = self.vix_predictor.vix_ml.iloc[:-1].tail(w).mean()
            self.vix_predictor.features.loc[idx, f'vix_vs_ma{w}'] = live_vix - ma
            self.vix_predictor.features.loc[idx, f'vix_vs_ma{w}_pct'] = ((live_vix - ma) / ma * 100)
        
        # Recalculate VIX z-scores
        for w in [63, 126, 252]:
            window_data = self.vix_predictor.vix_ml.iloc[:-1].tail(w)
            ma, std = window_data.mean(), window_data.std()
            self.vix_predictor.features.loc[idx, f'vix_zscore_{w}d'] = (live_vix - ma) / std
        
        # Recalculate VIX percentiles
        for w in [126, 252]:
            window_data = self.vix_predictor.vix_ml.iloc[:-1].tail(w)
            percentile = (window_data < live_vix).sum() / len(window_data) * 100
            self.vix_predictor.features.loc[idx, f'vix_percentile_{w}d'] = percentile
        
        # Recalculate VIX dynamics (velocity, momentum)
        for w in [1, 5, 10, 21]:
            if len(self.vix_predictor.vix_ml) > w:
                self.vix_predictor.features.loc[idx, f'vix_velocity_{w}d'] = (
                    live_vix - self.vix_predictor.vix_ml.iloc[-(w+1)]
                )
        
        # Recalculate SPX features
        for w in [20, 50, 200]:
            ma = self.vix_predictor.spx_ml.iloc[:-1].tail(w).mean()
            self.vix_predictor.features.loc[idx, f'spx_vs_ma{w}'] = ((live_spx - ma) / ma) * 100
        
        # Recalculate SPX momentum
        for w in [10, 21]:
            if len(self.vix_predictor.spx_ml) > w:
                ret = (live_spx - self.vix_predictor.spx_ml.iloc[-(w+1)]) / self.vix_predictor.spx_ml.iloc[-(w+1)]
                window_rets = self.vix_predictor.spx_ml.pct_change(w).iloc[:-1].tail(63)
                ret_ma, ret_std = window_rets.mean(), window_rets.std()
                self.vix_predictor.features.loc[idx, f'spx_momentum_z_{w}d'] = (ret - ret_ma) / ret_std
        
        # Recalculate VIX/RV ratio
        spx_returns = self.vix_predictor.spx_ml.pct_change()
        for w in [10, 21, 30, 63]:
            rv = spx_returns.iloc[:-1].tail(w).std() * np.sqrt(252) * 100
            self.vix_predictor.features.loc[idx, f'vix_rv_ratio_{w}d'] = live_vix / rv if rv > 0 else 1.0
        
        # Update base price features
        self.vix_predictor.features.loc[idx, 'vix'] = live_vix
        self.vix_predictor.features.loc[idx, 'spx_lag1'] = live_spx
        
        # Invalidate anomaly cache
        self._cached_anomaly_result = None
        self._cache_timestamp = None
        
        return {
            'vix_updated': live_vix,
            'spx_updated': live_spx,
            'features_recalculated': True,
            'anomaly_cache_cleared': True
        }
    
    def get_market_state(self):
        if not self.trained:
            raise ValueError("Run train() first")
        
        if self.vix_predictor.vix_ml is None:
            raise ValueError("VIX predictor not properly initialized. Run train() with ENABLE_TRAINING=True")
        
        try:
            live_vix = self.vix_predictor.fetcher.fetch_price('^VIX')
            if live_vix and self.vix_predictor.vix is not None:
                self.vix_predictor.vix.iloc[-1] = live_vix
        except:
            live_vix = None
        
        try:
            live_spx = self.vix_predictor.fetcher.fetch_price('^GSPC')
        except:
            live_spx = None
        
        current_vix = float(self.vix_predictor.vix_ml.iloc[-1])
        model_spx = float(self.vix_predictor.spx_ml.iloc[-1]) if self.vix_predictor.spx_ml is not None else 0.0

        current_spx = float(live_spx if live_spx else model_spx)
        
        vix_features = self._get_vix_feature_state(current_vix)
        try:
            spx_features = self._get_spx_feature_state()
        except:
            spx_features = {'price_action': {}, 'vix_relationship': {}}
        
        anomaly_results = self._get_cached_anomaly_result()
        
        persistence_stats = {
            'current_streak': 0, 'mean_duration': 0.0, 'max_duration': 0,
            'total_anomaly_days': 0, 'anomaly_rate': 0.0, 'num_episodes': 0
        }
        
        regime_stats = self._calculate_regime_stats(current_vix)
        ensemble_score = anomaly_results['ensemble']['score'] if anomaly_results else 0.0
        severity = self._classify_severity(ensemble_score)
        domain_anomalies = anomaly_results.get('domain_anomalies', {}) if anomaly_results else {}
        
        has_cboe = 'cboe_options_flow' in domain_anomalies
        has_cross = 'cross_asset_divergence' in domain_anomalies
        overall_confidence = "HIGH" if (has_cboe and has_cross) else ("MODERATE" if (has_cboe or has_cross) else "LOW")
        confidence_msg = "All detectors operational" if overall_confidence == "HIGH" else \
                        ("Some detectors operational" if overall_confidence == "MODERATE" else "Limited detection data")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_data': {
                'spx': current_spx,
                'spx_model': model_spx,
                'spx_change_today': ((current_spx - model_spx) / model_spx) * 100 if live_spx and model_spx > 0 else 0,
                'vix': current_vix,
                'vix_regime': self._classify_vix_regime(current_vix)
            },
            'vix_structure': vix_features,
            'spx_structure': spx_features,
            'vix_predictions': self._format_vix_predictions(current_vix, regime_stats, ensemble_score),
            'anomaly_analysis': {
                'ensemble': {
                    'score': float(ensemble_score),
                    'std': float(anomaly_results['ensemble']['std']) if anomaly_results else 0.0,
                    'severity': severity,
                    'severity_message': self._get_severity_message(severity),
                    'interpretation': self._get_interpretation(ensemble_score)
                },
                'domain_anomalies': domain_anomalies,
                'top_anomalies': self._get_top_anomalies_list(anomaly_results) if anomaly_results else [],
                'persistence': persistence_stats,
                'data_availability': {
                    'cboe_indicators': has_cboe,
                    'cross_asset_data': has_cross,
                    'overall_confidence': overall_confidence,
                    'confidence_message': confidence_msg
                }
            },
            'model_diagnostics': {
                'vix_accuracy': 0.65,
                'anomaly_detectors_active': anomaly_results.get('data_quality', {}).get('active_detectors', 0) if anomaly_results else 0,
                'anomaly_detectors_total': 15,
                'memory_status': self.get_memory_report() if self.memory_monitoring_enabled else None
            }
        }
    
    def _classify_severity(self, score: float) -> str:
        if self.trained and hasattr(self, 'vix_predictor') and self.vix_predictor.anomaly_detector:
            level, _, _ = self.vix_predictor.anomaly_detector.classify_anomaly(score, method='statistical')
            return level
        from config import ANOMALY_THRESHOLDS
        if score >= ANOMALY_THRESHOLDS['severity_extreme']:
            return "CRITICAL"
        elif score >= ANOMALY_THRESHOLDS['severity_high']:
            return "HIGH"
        elif score >= ANOMALY_THRESHOLDS['severity_moderate']:
            return "MODERATE"
        return "NORMAL"
    
    def _get_severity_message(self, severity: str) -> str:
        msgs = {
            "CRITICAL": "Extreme market stress",
            "HIGH": "Elevated stress",
            "MODERATE": "Moderate stress",
            "NORMAL": "Normal bounds"
        }
        return msgs.get(severity, "Unknown")
    
    def _get_interpretation(self, score: float) -> str:
        if score >= 0.85:
            return "Multiple detectors signaling systemic stress"
        elif score >= 0.70:
            return "Several domains showing elevated anomalies"
        elif score >= 0.50:
            return "Some anomalous behavior detected"
        return "Behavior consistent with historical patterns"
    
    def _get_vix_feature_state(self, current_vix: float) -> dict:
        f = self.vix_predictor.features.iloc[-1]
        return {
            'current_level': float(current_vix),
            'vs_ma21': float(f.get('vix_vs_ma21', 0)),
            'vs_ma63': float(f.get('vix_vs_ma63', 0)),
            'zscore_63d': float(f.get('vix_zscore_63d', 0)),
            'percentile_252d': float(f.get('vix_percentile_252d', 50)),
            'regime': int(f.get('vix_regime', 1)),
            'days_in_regime': int(f.get('days_in_regime', 0)),
            'velocity_5d': float(f.get('vix_velocity_5d', 0))
        }
    
    def _get_spx_feature_state(self) -> dict:
        f = self.vix_predictor.features.iloc[-1]
        return {
            'price_action': {
                'vs_ma50': float(f.get('spx_vs_ma50', 0)),
                'vs_ma200': float(f.get('spx_vs_ma200', 0)),
                'momentum_10d': float(f.get('spx_momentum_z_10d', 0)),
                'realized_vol_21d': float(f.get('spx_realized_vol_21d', 15))
            },
            'vix_relationship': {
                'corr_21d': float(f.get('spx_vix_corr_21d', -0.7)),
                'vix_rv_ratio_21d': float(f.get('vix_rv_ratio_21d', 1.0))
            }
        }
    
    def _classify_vix_regime(self, vix: float) -> dict:
        for i, boundary in enumerate(REGIME_BOUNDARIES[1:]):
            if vix < boundary:
                return {
                    'id': i,
                    'name': REGIME_NAMES[i],
                    'range': [float(REGIME_BOUNDARIES[i]), float(REGIME_BOUNDARIES[i+1])]
                }
        return {
            'id': 3,
            'name': REGIME_NAMES[3],
            'range': [float(REGIME_BOUNDARIES[3]), 100.0]
        }
    
    def _calculate_regime_stats(self, current_vix: float) -> dict:
        regime_id = self._classify_vix_regime(current_vix)['id']
        regime_data = self.vix_predictor.regime_stats_historical['regimes'][regime_id]
        return {
            'current_regime': regime_data,
            'transition_probabilities': regime_data['transitions_5d']
        }
    
    def _format_vix_predictions(self, current_vix: float, regime_stats: dict, anomaly_score: float) -> dict:
        return {
            'regime_persistence': {
                'probability': regime_stats['transition_probabilities']['persistence']['probability'],
                'expected_duration': regime_stats['current_regime']['statistics']['mean_duration']
            },
            'transition_risk': {
                'elevated': anomaly_score > 0.7,
                'direction': 'higher' if current_vix < 20 else 'lower',
                'confidence': 'high' if anomaly_score > 0.8 else ('moderate' if anomaly_score > 0.6 else 'low')
            }
        }
    
    def _get_top_anomalies_list(self, anomaly_results: dict) -> list:
        domain_scores = [
            {'name': name, 'score': data['score']}
            for name, data in anomaly_results.get('domain_anomalies', {}).items()
        ]
        return sorted(domain_scores, key=lambda x: x['score'], reverse=True)[:5]
    
    def export_json(self, filepath: str = "./json_data/market_state.json"):
        state = self.get_market_state()
        
        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(clean_nans(state), f, indent=2)
        print(f"\n✅ Exported: {filepath}")
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-export")
        
        return state
    
    def print_anomaly_summary(self):
        if not self.trained:
            raise ValueError("Run train() first")
        
        if hasattr(self.vix_predictor, 'historical_ensemble_scores') and \
           self.vix_predictor.historical_ensemble_scores is not None and \
           self.vix_predictor.anomaly_detector:
            persistence_stats = self.vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
                self.vix_predictor.historical_ensemble_scores,
                dates=self.vix_predictor.features.index
            )
        else:
            persistence_stats = {
                'current_streak': 0, 'mean_duration': 0.0, 'max_duration': 0,
                'total_anomaly_days': 0, 'anomaly_rate': 0.0, 'num_episodes': 0
            }
        
        state = self.get_market_state()
        anomaly = state['anomaly_analysis']
        ensemble = anomaly['ensemble']
        
        print(f"\n{'='*80}\n15-DIMENSIONAL ANOMALY SUMMARY\n{'='*80}")
        print(f"\n🎯 ASSESSMENT: {ensemble['severity']}")
        print(f"   Score: {ensemble['score']:.1%}")
        print(f"   {ensemble['severity_message']}")
        print(f"\n⏱️ PERSISTENCE:")
        print(f"   Streak: {persistence_stats['current_streak']}d | Mean: {persistence_stats['mean_duration']:.1f}d | Rate: {persistence_stats['anomaly_rate']:.1%}")
        print(f"\n🔍 TOP 3:")
        for i, anom in enumerate(anomaly['top_anomalies'][:3], 1):
            level = "EXTREME" if anom['score'] > 0.9 else ("HIGH" if anom['score'] > 0.75 else "MODERATE")
            print(f"   {i}. {anom['name'].replace('_', ' ').title()}: {anom['score']:.0%} ({level})")
        
        if self.memory_monitoring_enabled:
            mem_report = self.get_memory_report()
            if 'error' not in mem_report:
                status_emoji = "🚨" if mem_report['status'] == 'CRITICAL' else ("⚠️" if mem_report['status'] == 'WARNING' else "✅")
                print(f"\n📊 MEMORY: {status_emoji} {mem_report['status']}")
                print(f"   Current: {mem_report['current_mb']:.1f} MB")
                if mem_report['baseline_mb']:
                    print(f"   Growth: {mem_report['growth_mb']:+.1f} MB from baseline")
                print(f"   Measurements: {mem_report['history']['measurements']}")
        
        print(f"\n{'='*80}")


def main():
    system = IntegratedMarketSystemV4()
    
    if not ENABLE_TRAINING:
        print(f"\n{'='*80}")
        print("⚠️ TRAINING DISABLED (config.ENABLE_TRAINING = False)")
        print("⚠️ System cannot function without training")
        print("⚠️ Please set ENABLE_TRAINING = True in config.py")
        print(f"{'='*80}\n")
        return
    
    system.train(years=TRAINING_YEARS, real_time_vix=True, verbose=False)
    
    # ============================================================================
    # NEW: Unified Export System (Step 1.1)
    # ============================================================================
    
    # Get cached anomaly result
    anomaly_result = system._get_cached_anomaly_result()
    
    if anomaly_result and system.vix_predictor.anomaly_detector:
        # Calculate persistence stats (needed for live state)
        persistence_stats = system.vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
            system.vix_predictor.historical_ensemble_scores,
            dates=system.vix_predictor.features.index
        )
        
        # Initialize unified exporter
        exporter = UnifiedExporter(output_dir='./json_data')
        
        # Export live state (updates every refresh)
        exporter.export_live_state(
            vix_predictor=system.vix_predictor,
            anomaly_result=anomaly_result,
            spx=system.vix_predictor.spx_ml,
            persistence_stats=persistence_stats
        )
        
        # Export historical context (training only)
        exporter.export_historical_context(
            vix_predictor=system.vix_predictor,
            spx=system.vix_predictor.spx_ml,
            historical_scores=system.vix_predictor.historical_ensemble_scores
        )
        
        # Export model cache (training only)
        exporter.export_model_cache(
            vix_predictor=system.vix_predictor
        )
        
        print("\n✅ Exported unified dashboard files:")
        print("   • live_state.json       (15 KB, updates every refresh)")
        print("   • historical.json       (300 KB, static)")
        print("   • model_cache.pkl       (15 MB, static)")
    
    
    # ============================================================================
    # Memory Reporting
    # ============================================================================
    if system.memory_monitoring_enabled:
        print(f"\n{'='*80}\nFINAL MEMORY REPORT\n{'='*80}")
        mem_report = system.get_memory_report()
        if 'error' not in mem_report:
            print(f"Status: {mem_report['status']}")
            print(f"Current: {mem_report['current_mb']:.1f} MB")
            print(f"Baseline: {mem_report['baseline_mb']:.1f} MB")
            print(f"Growth: {mem_report['growth_mb']:+.1f} MB")
            print(f"Total measurements: {mem_report['history']['measurements']}")
            print(f"Avg growth per cycle: {mem_report['history']['avg_growth_per_cycle']:.3f} MB")
            
            if mem_report['history']['measurements'] > 1:
                print(f"\nMemory Trend:")
                for sample in mem_report['history']['recent_samples'][-5:]:
                    print(f"  {sample['timestamp']}: {sample['memory_mb']:.1f} MB ({sample.get('type', 'unknown')})")
    
    print(f"\n{'='*80}\nANALYSIS COMPLETE\n{'='*80}")
    print("\n📊 New Files:")
    print("   • live_state.json")
    print("   • historical.json") 
    print("   • model_cache.pkl")
    print("\n📊 Legacy Files (for transition):")
    print("   • market_state.json")
    print("   • anomaly_report.json")

    # Generate Claude package
    # package = generate_claude_package(system)


if __name__ == "__main__":
    main()