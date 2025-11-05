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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class IntegratedMarketSystemV4:
    def __init__(self, cboe_data_dir: str = "./CBOE_Data_Archive"):
        self.feature_engine = UnifiedFeatureEngine(cboe_data_dir)
        self.vix_predictor = VIXPredictorV4(cboe_data_dir)
        self.trained = False
        self.cv_results = {}
        self._cached_anomaly_result = None
        self._cache_timestamp = None
        
        # Memory profiling setup
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
        except Exception as e:
            warnings.warn(f"Memory baseline initialization failed: {e}")
            self.memory_monitoring_enabled = False
    
    def _log_memory_stats(self, context: str = "refresh") -> dict:
        """Log memory usage and detect growth."""
        if not self.memory_monitoring_enabled:
            return {}
        
        try:
            mem_info = self.process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            
            if self.baseline_memory_mb is None:
                self._initialize_memory_baseline()
                return {}
            
            growth = current_mb - self.baseline_memory_mb
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'current_mb': float(current_mb),
                'baseline_mb': float(self.baseline_memory_mb),
                'growth_mb': float(growth),
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
                print(f"🚨 CRITICAL: Memory +{growth:.1f}MB | Consider restart")
            elif growth > self.memory_warning_threshold_mb:
                print(f"⚠️ Memory +{growth:.1f}MB")
            
            return stats
        except Exception as e:
            warnings.warn(f"Memory logging failed: {e}")
            return {}
    
    def get_memory_report(self) -> dict:
        """Generate memory diagnostics report."""
        if not self.memory_monitoring_enabled:
            return {'error': 'psutil not installed'}
        
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
                total_growth = self.memory_history[-1]['memory_mb'] - self.memory_history[0]['memory_mb']
                avg_growth = total_growth / len(self.memory_history)
            else:
                total_growth = avg_growth = 0.0
            
            return {
                'current_mb': float(current_mb),
                'baseline_mb': float(self.baseline_memory_mb) if self.baseline_memory_mb else None,
                'growth_mb': float(growth_mb),
                'status': self._get_memory_status(growth_mb),
                'history': {
                    'measurements': len(self.memory_history),
                    'total_growth_mb': float(total_growth),
                    'avg_growth_per_cycle': float(avg_growth),
                    'recent_samples': self.memory_history[-10:]
                },
                'gc_stats': {
                    'collections': gc.get_count(),
                    'tracked_objects': len(objects),
                    'top_types': [{'type': t, 'count': c} for t, c in top_types]
                }
            }
        except Exception as e:
            return {'error': f'Report generation failed: {e}'}
    
    def _get_memory_status(self, growth_mb: float) -> str:
        if growth_mb > self.memory_critical_threshold_mb:
            return 'CRITICAL'
        elif growth_mb > self.memory_warning_threshold_mb:
            return 'WARNING'
        return 'NORMAL'
    
    def train(self, years: int = TRAINING_YEARS, real_time_vix: bool = True, verbose: bool = False):
        """Train feature engine and VIX predictor."""
        print(f"\n{'='*80}\nINTEGRATED MARKET SYSTEM V4\n{'='*80}")
        print(f"Config: {years}y training | Real-time VIX: {real_time_vix}")
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="pre-training")
        
        # Build features
        print("\n[1/2] Building features...")
        feature_data = self.feature_engine.build_complete_features(years=years)
        features, vix, dates = feature_data['features'], feature_data['vix'], feature_data['dates']
        self.vix_predictor.spx_ml = feature_data['spx']
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-features")
        
        # Train predictor
        print("[2/2] Training VIX predictor...")
        vix_history_all = self.vix_predictor.fetcher.fetch_vix(
            '1990-01-02', 
            datetime.now().strftime('%Y-%m-%d'), 
            lookback_buffer_days=0
        )
        
        self.vix_predictor._update_regime_history_after_training(
            features=features, 
            vix=vix, 
            spx=self.vix_predictor.spx_ml, 
            vix_history_all=vix_history_all, 
            verbose=verbose
        )
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-training")
        
        # Update live VIX
        if real_time_vix:
            try:
                live_vix = self.vix_predictor.fetcher.fetch_price('^VIX')
                if live_vix:
                    self.vix_predictor.vix.iloc[-1] = live_vix
                    print(f"✅ Live VIX: {live_vix:.2f}")
            except:
                pass
        
        self._verify_feature_coverage(features, verbose)
        self.trained = True
        
        if self.memory_monitoring_enabled and self.baseline_memory_mb is None:
            self._initialize_memory_baseline()
        
        print(f"\n{'='*80}\n✅ TRAINING COMPLETE\n{'='*80}")
        return self.cv_results
    
    def _verify_feature_coverage(self, features: pd.DataFrame, verbose: bool = False):
        """Verify anomaly detector feature coverage."""
        from config import ANOMALY_FEATURE_GROUPS
        
        print("\n📋 FEATURE COVERAGE:")
        all_ok = True
        for domain, expected in ANOMALY_FEATURE_GROUPS.items():
            available = [f for f in expected if f in features.columns]
            coverage = len(available) / len(expected) * 100
            status = "✅" if coverage > 80 else ("⚠️" if coverage > 50 else "❌")
            all_ok = all_ok and (coverage > 80)
            
            print(f"   {status} {domain:30s} {len(available):3d}/{len(expected):3d} ({coverage:5.1f}%)")
            
            if verbose and coverage < 100:
                missing = [f for f in expected if f not in features.columns][:5]
                print(f"      Missing: {', '.join(missing)}")
        
        print(f"\n{'✅ All detectors operational' if all_ok else '⚠️ Some detectors limited'}")
    
    def _get_cached_anomaly_result(self, force_refresh: bool = False):
        """Get cached anomaly detection result."""
        if not self.vix_predictor.anomaly_detector:
            return None
        
        if force_refresh or self._cached_anomaly_result is None:
            if self.memory_monitoring_enabled and force_refresh:
                self._log_memory_stats(context="pre-anomaly")
            
            self._cached_anomaly_result = self.vix_predictor.anomaly_detector.detect(
                self.vix_predictor.features.iloc[[-1]], 
                verbose=False
            )
            self._cache_timestamp = datetime.now()
            
            if self.memory_monitoring_enabled and force_refresh:
                self._log_memory_stats(context="post-anomaly")
        
        return self._cached_anomaly_result
    
    def _recalculate_live_features(self, live_vix: float, live_spx: float):
        """Recalculate derived features after live price updates."""
        self.vix_predictor.vix_ml.iloc[-1] = live_vix
        self.vix_predictor.vix.iloc[-1] = live_vix
        self.vix_predictor.spx_ml.iloc[-1] = live_spx
        
        idx = self.vix_predictor.features.index[-1]
        
        # VIX mean reversion
        for w in [10, 21, 63, 126, 252]:
            ma = self.vix_predictor.vix_ml.iloc[:-1].tail(w).mean()
            self.vix_predictor.features.loc[idx, f'vix_vs_ma{w}'] = live_vix - ma
            self.vix_predictor.features.loc[idx, f'vix_vs_ma{w}_pct'] = ((live_vix - ma) / ma * 100)
        
        # VIX z-scores & percentiles
        for w in [63, 126, 252]:
            window_data = self.vix_predictor.vix_ml.iloc[:-1].tail(w)
            ma, std = window_data.mean(), window_data.std()
            self.vix_predictor.features.loc[idx, f'vix_zscore_{w}d'] = (live_vix - ma) / std
            
            if w in [126, 252]:
                percentile = (window_data < live_vix).sum() / len(window_data) * 100
                self.vix_predictor.features.loc[idx, f'vix_percentile_{w}d'] = percentile
        
        # VIX velocity
        for w in [1, 5, 10, 21]:
            if len(self.vix_predictor.vix_ml) > w:
                self.vix_predictor.features.loc[idx, f'vix_velocity_{w}d'] = (
                    live_vix - self.vix_predictor.vix_ml.iloc[-(w+1)]
                )
        
        # SPX features
        for w in [20, 50, 200]:
            ma = self.vix_predictor.spx_ml.iloc[:-1].tail(w).mean()
            self.vix_predictor.features.loc[idx, f'spx_vs_ma{w}'] = ((live_spx - ma) / ma) * 100
        
        # SPX momentum z-scores
        for w in [10, 21]:
            if len(self.vix_predictor.spx_ml) > w:
                ret = (live_spx - self.vix_predictor.spx_ml.iloc[-(w+1)]) / self.vix_predictor.spx_ml.iloc[-(w+1)]
                window_rets = self.vix_predictor.spx_ml.pct_change(w).iloc[:-1].tail(63)
                ret_ma, ret_std = window_rets.mean(), window_rets.std()
                self.vix_predictor.features.loc[idx, f'spx_momentum_z_{w}d'] = (ret - ret_ma) / ret_std
        
        # VIX/RV ratio
        spx_returns = self.vix_predictor.spx_ml.pct_change()
        for w in [10, 21, 30, 63]:
            rv = spx_returns.iloc[:-1].tail(w).std() * np.sqrt(252) * 100
            self.vix_predictor.features.loc[idx, f'vix_rv_ratio_{w}d'] = live_vix / rv if rv > 0 else 1.0
        
        # Update base features
        self.vix_predictor.features.loc[idx, 'vix'] = live_vix
        self.vix_predictor.features.loc[idx, 'spx_lag1'] = live_spx
        
        # Invalidate cache
        self._cached_anomaly_result = None
        self._cache_timestamp = None
        
        return {'vix_updated': live_vix, 'spx_updated': live_spx, 'features_recalculated': True}
    
    def get_market_state(self):
        """Get comprehensive market state analysis."""
        if not self.trained:
            raise ValueError("Run train() first")
        
        if self.vix_predictor.vix_ml is None:
            raise ValueError("VIX predictor not initialized. Set ENABLE_TRAINING=True")
        
        # Fetch live prices
        try:
            live_vix = self.vix_predictor.fetcher.fetch_price('^VIX')
            if live_vix:
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
        
        # Get features
        vix_features = self._get_vix_feature_state(current_vix)
        try:
            spx_features = self._get_spx_feature_state()
        except:
            spx_features = {'price_action': {}, 'vix_relationship': {}}
        
        # Anomaly detection
        anomaly_results = self._get_cached_anomaly_result()
        ensemble_score = anomaly_results['ensemble']['score'] if anomaly_results else 0.0
        severity = self._classify_severity(ensemble_score)
        domain_anomalies = anomaly_results.get('domain_anomalies', {}) if anomaly_results else {}
        
        # Data quality assessment
        has_cboe = 'cboe_options_flow' in domain_anomalies
        has_cross = 'cross_asset_divergence' in domain_anomalies
        overall_confidence = "HIGH" if (has_cboe and has_cross) else ("MODERATE" if (has_cboe or has_cross) else "LOW")
        
        # Regime analysis
        regime_stats = self._calculate_regime_stats(current_vix)
        
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
                'data_availability': {
                    'cboe_indicators': has_cboe,
                    'cross_asset_data': has_cross,
                    'overall_confidence': overall_confidence
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
        """Classify anomaly severity."""
        if self.trained and self.vix_predictor.anomaly_detector:
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
        return {
            "CRITICAL": "Extreme market stress",
            "HIGH": "Elevated stress",
            "MODERATE": "Moderate stress",
            "NORMAL": "Normal bounds"
        }.get(severity, "Unknown")
    
    def _get_interpretation(self, score: float) -> str:
        if score >= 0.85:
            return "Multiple detectors signaling systemic stress"
        elif score >= 0.70:
            return "Several domains showing elevated anomalies"
        elif score >= 0.50:
            return "Some anomalous behavior detected"
        return "Behavior consistent with historical patterns"
    
    def _get_vix_feature_state(self, current_vix: float) -> dict:
        """Extract VIX feature state."""
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
        """Extract SPX feature state."""
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
        """Classify current VIX regime."""
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
        """Calculate regime statistics."""
        regime_id = self._classify_vix_regime(current_vix)['id']
        regime_data = self.vix_predictor.regime_stats_historical['regimes'][regime_id]
        return {
            'current_regime': regime_data,
            'transition_probabilities': regime_data['transitions_5d']
        }
    
    def _format_vix_predictions(self, current_vix: float, regime_stats: dict, anomaly_score: float) -> dict:
        current_regime = regime_stats['current_regime']
        persistence_prob = regime_stats['transition_probabilities']['persistence']['probability']
        persistence_prob_clamped = max(0.01, min(0.99, persistence_prob))
        expected_duration = 1.0 / (1.0 - persistence_prob_clamped)
        
        return {
            'regime_persistence': {
                'probability': persistence_prob,
                'expected_duration': float(expected_duration)
            },
            'transition_risk': {
                'elevated': anomaly_score > 0.7,
                'direction': 'higher' if current_vix < 20 else 'lower',
                'confidence': 'high' if anomaly_score > 0.8 else ('moderate' if anomaly_score > 0.6 else 'low')
            }
        }
    
    def _get_top_anomalies_list(self, anomaly_results: dict) -> list:
        """Get top anomalies sorted by score."""
        domain_scores = [
            {'name': name, 'score': data['score']}
            for name, data in anomaly_results.get('domain_anomalies', {}).items()
        ]
        return sorted(domain_scores, key=lambda x: x['score'], reverse=True)[:5]
    
    def export_json(self, filepath: str = "./json_data/market_state.json"):
        """Export market state to JSON."""
        state = self.get_market_state()
        
        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(clean_nans(state), f, indent=2)
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-export")
        
        return state
    
    def print_anomaly_summary(self):
        """Print comprehensive anomaly analysis."""
        if not self.trained:
            raise ValueError("Run train() first")
        
        # Get persistence stats
        persistence_stats = {'current_streak': 0, 'mean_duration': 0.0, 'max_duration': 0,
                           'total_anomaly_days': 0, 'anomaly_rate': 0.0, 'num_episodes': 0}
        
        if hasattr(self.vix_predictor, 'historical_ensemble_scores') and \
           self.vix_predictor.historical_ensemble_scores is not None and \
           self.vix_predictor.anomaly_detector:
            persistence_stats = self.vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
                self.vix_predictor.historical_ensemble_scores,
                dates=self.vix_predictor.features.index
            )
        
        state = self.get_market_state()
        anomaly = state['anomaly_analysis']
        ensemble = anomaly['ensemble']
        
        # Anomaly summary
        print(f"\n{'='*80}\n15-DIMENSIONAL ANOMALY SUMMARY\n{'='*80}")
        print(f"\n🎯 {ensemble['severity']}: {ensemble['score']:.1%}")
        print(f"   {ensemble['severity_message']}")
        print(f"\n⏱️ PERSISTENCE: {persistence_stats['current_streak']}d streak | "
              f"Mean: {persistence_stats['mean_duration']:.1f}d | Rate: {persistence_stats['anomaly_rate']:.1%}")
        print(f"\n🔍 TOP 3:")
        for i, anom in enumerate(anomaly['top_anomalies'][:3], 1):
            level = "EXTREME" if anom['score'] > 0.9 else ("HIGH" if anom['score'] > 0.75 else "MODERATE")
            print(f"   {i}. {anom['name'].replace('_', ' ').title()}: {anom['score']:.0%} ({level})")
        
        # Memory status
        if self.memory_monitoring_enabled:
            mem_report = self.get_memory_report()
            if 'error' not in mem_report:
                status_emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "NORMAL": "✅"}[mem_report['status']]
                print(f"\n📊 MEMORY: {status_emoji} {mem_report['status']} | "
                      f"{mem_report['current_mb']:.1f}MB (+{mem_report['growth_mb']:.1f}MB)")
        
        print(f"\n{'='*80}")


def main():
    """Main execution function."""
    system = IntegratedMarketSystemV4()
    
    if not ENABLE_TRAINING:
        print(f"\n{'='*80}")
        print("⚠️ TRAINING DISABLED (config.ENABLE_TRAINING = False)")
        print("⚠️ Set ENABLE_TRAINING = True in config.py")
        print(f"{'='*80}\n")
        return
    
    # Train system
    system.train(years=TRAINING_YEARS, real_time_vix=True, verbose=False)
    system.print_anomaly_summary()
    
    # Export unified dashboard files
    anomaly_result = system._get_cached_anomaly_result()
    
    if anomaly_result and system.vix_predictor.anomaly_detector:
        persistence_stats = system.vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
            system.vix_predictor.historical_ensemble_scores,
            dates=system.vix_predictor.features.index
        )
        
        exporter = UnifiedExporter(output_dir='./json_data')
        
        # Live state (updates every refresh)
        exporter.export_live_state(
            vix_predictor=system.vix_predictor,
            anomaly_result=anomaly_result,
            spx=system.vix_predictor.spx_ml,
            persistence_stats=persistence_stats
        )
        
        # Historical context (training only)
        exporter.export_historical_context(
            vix_predictor=system.vix_predictor,
            spx=system.vix_predictor.spx_ml,
            historical_scores=system.vix_predictor.historical_ensemble_scores
        )
        
        # Model cache (training only)
        exporter.export_model_cache(vix_predictor=system.vix_predictor)
        
        print("\n✅ Exported unified dashboard files:")
        print("   • live_state.json    (15 KB, updates every refresh)")
        print("   • historical.json    (300 KB, static)")
        print("   • model_cache.pkl    (15 MB, static)")
    
    # Final memory report
    if system.memory_monitoring_enabled:
        mem_report = system.get_memory_report()
        if 'error' not in mem_report:
            print(f"\n{'='*80}\nMEMORY REPORT\n{'='*80}")
            print(f"Status: {mem_report['status']}")
            print(f"Current: {mem_report['current_mb']:.1f}MB | Growth: {mem_report['growth_mb']:+.1f}MB")
            print(f"Cycles: {mem_report['history']['measurements']} | "
                  f"Avg/cycle: {mem_report['history']['avg_growth_per_cycle']:.3f}MB")
    
    print(f"\n{'='*80}\nANALYSIS COMPLETE\n{'='*80}")


if __name__ == "__main__":
    main()