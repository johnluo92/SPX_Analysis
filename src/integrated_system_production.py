"""Integrated Market Analysis System V4 - Refactored
Simplified architecture with anomaly detection at the core.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json
import os
import gc
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

from core.feature_engine import UnifiedFeatureEngine
from core.anomaly_detector import MultiDimensionalAnomalyDetector
from core.data_fetcher import UnifiedDataFetcher
from core.xgboost_trainer import XGBoostTrainer
from config import (
    TRAINING_YEARS, ENABLE_TRAINING, RANDOM_STATE, 
    REGIME_BOUNDARIES, REGIME_NAMES, CBOE_DATA_DIR
)
from export.unified_exporter import UnifiedExporter

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AnomalyOrchestrator:
    """
    Orchestrates anomaly detection workflow:
    1. Manages VIX/SPX history
    2. Maintains regime statistics
    3. Coordinates anomaly detector
    4. Handles state persistence
    """
    
    def __init__(self):
        self.fetcher = UnifiedDataFetcher()
        self.anomaly_detector = None
        self.vix_history_all = None
        self.vix_ml = None
        self.spx_ml = None
        self.features = None
        self.regime_stats = None
        self.historical_ensemble_scores = None
        self.trained = False
    
    def train(self, features: pd.DataFrame, vix: pd.Series, spx: pd.Series, 
              vix_history_all: pd.Series = None, verbose: bool = True):
        """Train anomaly detection system on historical features."""
        
        if verbose:
            print("\n[Anomaly Orchestrator] Training...")
        
        # Store data
        self.features = features
        self.vix_ml = vix
        self.spx_ml = spx
        self.vix_history_all = vix_history_all if vix_history_all is not None else vix
        
        # Compute regime statistics
        self.regime_stats = self._compute_regime_statistics(self.vix_history_all)
        
        # Train anomaly detector
        self.anomaly_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05, random_state=RANDOM_STATE
        )
        self.anomaly_detector.train(features.fillna(0), verbose=verbose)
        
        # Generate historical ensemble scores
        self._generate_historical_scores(verbose)
        
        self.trained = True
        if verbose:
            print("✅ Anomaly orchestrator trained")
    
    def _generate_historical_scores(self, verbose: bool = True):
        """Generate complete historical anomaly scores."""
        if not self.anomaly_detector or not self.anomaly_detector.trained:
            warnings.warn("Anomaly detector not trained")
            return
        
        scores = []
        for i in range(len(self.features)):
            result = self.anomaly_detector.detect(
                self.features.iloc[[i]], verbose=False
            )
            scores.append(result['ensemble']['score'])
        
        self.historical_ensemble_scores = np.array(scores)
        if verbose:
            print(f"Generated {len(scores)} historical anomaly scores")
    
    def detect_current(self, verbose: bool = False) -> dict:
        """Run anomaly detection on most recent feature row."""
        if not self.trained:
            raise ValueError("Must train before detecting")
        
        return self.anomaly_detector.detect(
            self.features.iloc[[-1]], verbose=verbose
        )
    
    def get_persistence_stats(self) -> dict:
        """Calculate anomaly persistence statistics."""
        if not self.trained or self.historical_ensemble_scores is None:
            return {
                'current_streak': 0, 'mean_duration': 0.0, 'max_duration': 0,
                'total_anomaly_days': 0, 'anomaly_rate': 0.0, 'num_episodes': 0
            }
        
        return self.anomaly_detector.calculate_historical_persistence_stats(
            self.historical_ensemble_scores,
            dates=self.features.index
        )
    
    def _compute_regime_statistics(self, vix_series: pd.Series) -> dict:
        """Compute comprehensive regime statistics from VIX history."""
        stats = {
            'observation_period': {
                'start_date': str(vix_series.index[0]),
                'end_date': str(vix_series.index[-1]),
                'total_days': len(vix_series)
            },
            'regimes': []
        }
        
        for regime_id in range(len(REGIME_NAMES)):
            regime_name = REGIME_NAMES[regime_id]
            lower_bound = REGIME_BOUNDARIES[regime_id]
            upper_bound = REGIME_BOUNDARIES[regime_id + 1] if regime_id < len(REGIME_BOUNDARIES) - 1 else np.inf
            
            regime_mask = (vix_series >= lower_bound) & (vix_series < upper_bound)
            regime_days = vix_series[regime_mask]
            
            # Compute regime assignments
            vix_regimes = pd.Series(index=vix_series.index, dtype=int)
            for rid in range(len(REGIME_NAMES)):
                lb = REGIME_BOUNDARIES[rid]
                ub = REGIME_BOUNDARIES[rid + 1] if rid < len(REGIME_BOUNDARIES) - 1 else np.inf
                mask = (vix_series >= lb) & (vix_series < ub)
                vix_regimes[mask] = rid
            
            # Calculate transitions
            regime_transitions = vix_regimes[vix_regimes == regime_id]
            future_regimes_5d = vix_regimes.shift(-5)
            valid_mask = future_regimes_5d.notna()
            
            valid_indices = regime_transitions.index.intersection(valid_mask[valid_mask].index)
            transitions_5d = future_regimes_5d[valid_indices].value_counts()
            total_opp = len(valid_indices)
            
            regime_info = {
                'regime_id': int(regime_id),
                'regime_name': regime_name,
                'boundaries': [float(lower_bound), float(upper_bound) if upper_bound != np.inf else 100.0],
                'observations': {
                    'count': int(len(regime_days)),
                    'percentage': float(len(regime_days) / len(vix_series) * 100) if len(vix_series) > 0 else 0.0,
                    'mean_vix': float(regime_days.mean()) if len(regime_days) > 0 else 0.0,
                    'std_vix': float(regime_days.std()) if len(regime_days) > 0 else 0.0
                },
                'transitions_5d': {
                    'persistence': {
                        'probability': float(transitions_5d.get(regime_id, 0) / total_opp) if total_opp > 0 else 0.0,
                        'observations': int(transitions_5d.get(regime_id, 0)),
                        'total_opportunities': int(total_opp)
                    },
                    'to_other_regimes': {
                        int(other): {
                            'probability': float(transitions_5d.get(other, 0) / total_opp) if total_opp > 0 else 0.0,
                            'observations': int(transitions_5d.get(other, 0)),
                            'total_opportunities': int(total_opp)
                        } for other in range(len(REGIME_NAMES)) if other != regime_id
                    }
                }
            }
            stats['regimes'].append(regime_info)
        
        return stats
    
    def save_state(self, filepath: str = './json_data/model_cache.pkl'):
        """Save model state for quick refresh without retraining."""
        if not self.trained:
            raise ValueError("Must train before saving state")
        
        state = {
            'detectors': self.anomaly_detector.detectors,
            'scalers': self.anomaly_detector.scalers,
            'training_distributions': self.anomaly_detector.training_distributions,
            'feature_groups': self.anomaly_detector.feature_groups,
            'random_subspaces': self.anomaly_detector.random_subspaces,
            'statistical_thresholds': self.anomaly_detector.statistical_thresholds,
            'vix_history': self.vix_ml.tail(252).to_dict(),
            'spx_history': self.spx_ml.tail(252).to_dict(),
            'last_features': self.features.tail(1).to_dict(),
            'feature_columns': self.features.columns.tolist(),
            'export_timestamp': pd.Timestamp.now().isoformat(),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✅ Saved state: {filepath} ({Path(filepath).stat().st_size / (1024*1024):.2f} MB)")
    
    def load_state(self, filepath: str = './json_data/model_cache.pkl'):
        """Load cached state for fast refresh."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore anomaly detector
        self.anomaly_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05, random_state=RANDOM_STATE
        )
        self.anomaly_detector.detectors = state['detectors']
        self.anomaly_detector.scalers = state['scalers']
        self.anomaly_detector.training_distributions = state['training_distributions']
        self.anomaly_detector.feature_groups = state['feature_groups']
        self.anomaly_detector.random_subspaces = state['random_subspaces']
        self.anomaly_detector.statistical_thresholds = state['statistical_thresholds']
        self.anomaly_detector.trained = True
        
        # Restore historical data
        self.vix_ml = self._dict_to_series(state['vix_history'])
        self.spx_ml = self._dict_to_series(state['spx_history'])
        self.features = self._dict_to_dataframe(
            state['last_features'], state['feature_columns']
        )
        
        self.trained = True
        print(f"✅ Loaded state from {filepath}")
    
    def _dict_to_series(self, d: dict) -> pd.Series:
        """Convert dict to pandas Series with DatetimeIndex."""
        dates = pd.to_datetime(list(d.keys()))
        return pd.Series(list(d.values()), index=dates)
    
    def _dict_to_dataframe(self, d: dict, columns: list) -> pd.DataFrame:
        """Convert dict to pandas DataFrame with DatetimeIndex."""
        dates = pd.to_datetime(list(next(iter(d.values())).keys()))
        data = {col: list(d[col].values()) for col in columns}
        return pd.DataFrame(data, index=dates)


class IntegratedMarketSystemV4:
    """
    Main system integrating:
    - Feature engineering (UnifiedFeatureEngine)
    - Anomaly detection (AnomalyOrchestrator)
    - Market state reporting
    - Memory monitoring
    """
    
    def __init__(self, cboe_data_dir: str = CBOE_DATA_DIR):
        # Create data fetcher first
        self.data_fetcher = UnifiedDataFetcher()
        
        # Pass the fetcher object to feature engine
        self.feature_engine = UnifiedFeatureEngine(data_fetcher=self.data_fetcher)
        
        self.orchestrator = AnomalyOrchestrator()
        self.trained = False
        self._cached_anomaly_result = None
        self._cache_timestamp = None
        
        # Memory profiling (rest of __init__ stays the same)
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
            self.baseline_memory_mb = None
            self.memory_history = []
            self.memory_monitoring_enabled = True
        else:
            self.memory_monitoring_enabled = False
    
    def train(self, years: int = TRAINING_YEARS, real_time_vix: bool = True, verbose: bool = False):
        """Train the complete system."""
        print(f"\n{'='*80}\nINTEGRATED MARKET SYSTEM V4 - REFACTORED\n{'='*80}")
        print(f"Config: {years}y training | Real-time VIX: {real_time_vix}")
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="pre-training")
        
        # Build features
        print("\n[1/2] Building features...")
        feature_data = self.feature_engine.build_complete_features(years=years)
        features = feature_data['features']
        vix = feature_data['vix']
        spx = feature_data['spx']
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-features")
        
        # Fetch complete VIX history for regime stats
        print("[2/2] Training anomaly system...")
        vix_history_all = self.orchestrator.fetcher.fetch_yahoo(
            '^VIX',
            '1990-01-02',
            datetime.now().strftime('%Y-%m-%d'),
        )['Close'].squeeze()
        
        # Train orchestrator
        self.orchestrator.train(
            features=features,
            vix=vix,
            spx=spx,
            vix_history_all=vix_history_all,
            verbose=verbose
        )
        
        if self.memory_monitoring_enabled:
            self._log_memory_stats(context="post-training")
        
        # Update live VIX
        if real_time_vix:
            try:
                live_vix = self.orchestrator.fetcher.fetch_price('^VIX')
                if live_vix:
                    self.orchestrator.vix_ml.iloc[-1] = live_vix
                    self.orchestrator.features.iloc[-1, self.orchestrator.features.columns.get_loc('vix')] = live_vix
                    if verbose:
                        print(f"✅ Updated live VIX: {live_vix:.2f}")
            except Exception as e:
                warnings.warn(f"Live VIX fetch failed: {e}")
        
        self.trained = True
        print(f"\n{'='*80}\n✅ TRAINING COMPLETE\n{'='*80}")
    
    def get_market_state(self) -> dict:
        """Generate comprehensive market state snapshot."""
        if not self.trained:
            raise ValueError("Must train system first")
        
        # Get current anomaly detection
        anomaly_result = self._get_cached_anomaly_result()
        
        # Get persistence stats
        persistence_stats = self.orchestrator.get_persistence_stats()
        
        # Current VIX and regime
        current_vix = float(self.orchestrator.vix_ml.iloc[-1])
        current_regime = self._classify_vix_regime(current_vix)
        
        # Format anomaly analysis
        ensemble = anomaly_result['ensemble']
        ensemble_score = ensemble['score']
        
        level, p_value, confidence = self.orchestrator.anomaly_detector.classify_anomaly(
            ensemble_score, method='statistical'
        )
        
        severity_messages = {
            'CRITICAL': f'Extreme anomaly ({ensemble_score:.1%}) - Markets in unprecedented configuration',
            'HIGH': f'Significant anomaly ({ensemble_score:.1%}) - Notable market stress detected',
            'MODERATE': f'Moderate anomaly ({ensemble_score:.1%}) - Elevated market uncertainty',
            'NORMAL': f'Normal conditions ({ensemble_score:.1%}) - Market within typical ranges'
        }
        
        ensemble['severity'] = level
        ensemble['severity_message'] = severity_messages[level]
        if p_value is not None:
            ensemble['p_value'] = float(p_value)
            ensemble['confidence'] = float(confidence)
        
        # Get top anomalies
        top_anomalies = self._get_top_anomalies_list(anomaly_result)
        
        # Regime stats
        regime_stats = self.orchestrator.regime_stats['regimes'][current_regime['id']]
        persistence_prob = regime_stats['transitions_5d']['persistence']['probability']
        persistence_prob_clamped = max(0.01, min(0.99, persistence_prob))
        expected_duration = 1.0 / (1.0 - persistence_prob_clamped)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'vix': {
                'current': current_vix,
                'regime': current_regime,
                'regime_stats': regime_stats
            },
            'anomaly_analysis': {
                'ensemble': ensemble,
                'top_anomalies': top_anomalies,
                'persistence': persistence_stats,
                'domain_anomalies': anomaly_result.get('domain_anomalies', {}),
                'random_anomalies': anomaly_result.get('random_anomalies', {})
            },
            'regime_forecast': {
                'persistence_probability': float(persistence_prob),
                'expected_duration_days': float(expected_duration),
                'transition_risk': 'elevated' if ensemble_score > 0.7 else 'normal'
            },
            'spx_state': self._get_spx_feature_state(),
            'system_health': {
                'trained': self.trained,
                'feature_count': len(self.orchestrator.features.columns),
                'detectors_active': anomaly_result.get('data_quality', {}).get('active_detectors', 0),
                'last_update': self.orchestrator.features.index[-1].isoformat()
            }
        }
    
    def _get_cached_anomaly_result(self) -> dict:
        """Get anomaly result with caching."""
        now = datetime.now()
        if self._cached_anomaly_result is None or \
           (self._cache_timestamp and (now - self._cache_timestamp).seconds > 60):
            self._cached_anomaly_result = self.orchestrator.detect_current(verbose=False)
            self._cache_timestamp = now
        return self._cached_anomaly_result
    
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
    
    def _get_spx_feature_state(self) -> dict:
        """Extract SPX feature state."""
        f = self.orchestrator.features.iloc[-1]
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
    
    def _get_top_anomalies_list(self, anomaly_results: dict) -> list:
        """Get top anomalies sorted by score."""
        domain_scores = [
            {'name': name, 'score': data['score']}
            for name, data in anomaly_results.get('domain_anomalies', {}).items()
        ]
        return sorted(domain_scores, key=lambda x: x['score'], reverse=True)[:5]
    
    def print_anomaly_summary(self):
        """Print comprehensive anomaly analysis summary."""
        if not self.trained:
            raise ValueError("Run train() first")
        
        state = self.get_market_state()
        anomaly = state['anomaly_analysis']
        ensemble = anomaly['ensemble']
        persistence = anomaly['persistence']
        
        print(f"\n{'='*80}\n15-DIMENSIONAL ANOMALY SUMMARY\n{'='*80}")
        print(f"\n🎯 {ensemble['severity']}: {ensemble['score']:.1%}")
        print(f"   {ensemble['severity_message']}")
        print(f"\n⏱️ PERSISTENCE: {persistence['current_streak']}d streak | "
              f"Mean: {persistence['mean_duration']:.1f}d | Rate: {persistence['anomaly_rate']:.1%}")
        print(f"\n🔍 TOP 3:")
        for i, anom in enumerate(anomaly['top_anomalies'][:3], 1):
            level = "EXTREME" if anom['score'] > 0.9 else ("HIGH" if anom['score'] > 0.75 else "MODERATE")
            print(f"   {i}. {anom['name'].replace('_', ' ').title()}: {anom['score']:.0%} ({level})")
        
        if self.memory_monitoring_enabled:
            mem_report = self.get_memory_report()
            if 'error' not in mem_report:
                status_emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "NORMAL": "✅"}[mem_report['status']]
                print(f"\n📊 MEMORY: {status_emoji} {mem_report['status']} | "
                      f"{mem_report['current_mb']:.1f}MB (+{mem_report['growth_mb']:.1f}MB)")
        
        print(f"\n{'='*80}")
    
    # Memory monitoring methods
    def _initialize_memory_baseline(self):
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
            warnings.warn(f"Memory baseline failed: {e}")
            self.memory_monitoring_enabled = False
    
    def _log_memory_stats(self, context: str = "refresh") -> dict:
        if not self.memory_monitoring_enabled:
            return {}
        try:
            mem_info = self.process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            if self.baseline_memory_mb is None:
                self._initialize_memory_baseline()
                return {}
            growth = current_mb - self.baseline_memory_mb
            self.memory_history.append({
                'timestamp': datetime.now().isoformat(),
                'memory_mb': current_mb,
                'type': context
            })
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-1000:]
            return {'current_mb': current_mb, 'growth_mb': growth}
        except Exception as e:
            return {}
    
    def get_memory_report(self) -> dict:
        if not self.memory_monitoring_enabled:
            return {'error': 'psutil not installed'}
        try:
            mem_info = self.process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            growth_mb = current_mb - self.baseline_memory_mb if self.baseline_memory_mb else 0.0
            status = 'CRITICAL' if growth_mb > 200 else ('WARNING' if growth_mb > 50 else 'NORMAL')
            return {
                'current_mb': float(current_mb),
                'baseline_mb': float(self.baseline_memory_mb) if self.baseline_memory_mb else None,
                'growth_mb': float(growth_mb),
                'status': status
            }
        except Exception as e:
            return {'error': str(e)}


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
    from feature_diagnostics import run_diagnostics
    report = run_diagnostics(system)
    # Review ./diagnostics/feature_report.json
    from anomaly_validator import validate_anomaly_system
    val_report = validate_anomaly_system(system)
    # Check if crisis detection is working
    
    # Export unified dashboard files
    anomaly_result = system._get_cached_anomaly_result()
    
    if anomaly_result and system.orchestrator.anomaly_detector:
        persistence_stats = system.orchestrator.get_persistence_stats()
        
        exporter = UnifiedExporter(output_dir='./json_data')
        
        # Live state (updates every refresh)
        exporter.export_live_state(
            orchestrator=system.orchestrator,  # Pass orchestrator as predictor
            anomaly_result=anomaly_result,
            spx=system.orchestrator.spx_ml,
            persistence_stats=persistence_stats
        )
        
        # Historical context (training only)
        exporter.export_historical_context(
            orchestrator=system.orchestrator,
            spx=system.orchestrator.spx_ml,
            historical_scores=system.orchestrator.historical_ensemble_scores
        )
        
        # Model cache (training only)
        system.orchestrator.save_state('./json_data/model_cache.pkl')
        
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
    
    print(f"\n{'='*80}\nANALYSIS COMPLETE\n{'='*80}")


if __name__ == "__main__":
    main()