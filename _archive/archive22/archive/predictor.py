"""VIX Predictor V4 - Streamlined Production Version"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .anomaly_detector import MultiDimensionalAnomalyDetector
from config import RANDOM_STATE, REGIME_BOUNDARIES, REGIME_NAMES
from .data_fetcher import UnifiedDataFetcher


class VIXPredictorV4:
    """Production VIX predictor with anomaly detection."""
    
    def __init__(self, cboe_data_dir: str = "./CBOE_Data_Archive"):
        self.fetcher = UnifiedDataFetcher()
        self.vix_history_all = None
        self.regime_stats_historical = None
        self.vix_ml = None
        self.vix = None
        self.spx_ml = None
        self.features = None
        self.anomaly_detector = None
        self.duration_predictor = None
        self.scaler = StandardScaler()
        self.validation_metrics = {}
        self.trained = False
        self.historical_ensemble_scores = None
    
    # ============================================================================
    # LOAD/SAVE STATE
    # ============================================================================
    
    def load_refresh_state(self, filepath: str = './json_data/refresh_state.pkl'):
        """Load cached model state for live updates without retraining."""
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
        self.vix = self.vix_ml.copy()
        self.spx_ml = self._dict_to_series(state['spx_history'])
        
        # Restore features
        self.features = self._dict_to_dataframe(
            state['last_features'], state['feature_columns']
        )
        
        # Load regime statistics
        regime_stats_path = filepath.parent / 'regime_statistics.json'
        if regime_stats_path.exists():
            with open(regime_stats_path, 'r') as f:
                self.regime_stats_historical = json.load(f)
        else:
            self.regime_stats_historical = self._compute_regime_statistics(self.vix_ml)
        
        # Load historical anomaly scores
        hist_scores_path = filepath.parent / 'historical_anomaly_scores.json'
        if hist_scores_path.exists():
            with open(hist_scores_path, 'r') as f:
                hist_data = json.load(f)
                self.historical_ensemble_scores = np.array(hist_data['ensemble_scores'])
        
        # Validate detector
        test_result = self.anomaly_detector.detect(self.features.iloc[[-1]], verbose=False)
        if test_result is None or 'ensemble' not in test_result:
            raise RuntimeError("Loaded detector validation failed")
        
        self.trained = True
        print(f"✅ Loaded state from {filepath} - Ready for live updates")
    
    def export_refresh_state(self, filepath: str = './json_data/refresh_state.pkl'):
        """Export minimal state for live refresh without full retraining."""
        if not self.trained:
            raise ValueError("Model must be trained before exporting")
        
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
            'vix_last_date': self._to_iso(self.vix_ml.index[-1]),
            'spx_last_date': self._to_iso(self.spx_ml.index[-1]),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"✅ Exported state: {filepath} ({Path(filepath).stat().st_size / (1024*1024):.2f} MB)")
    
    # ============================================================================
    # TRAINING
    # ============================================================================
    
    def _update_regime_history_after_training(self, features: pd.DataFrame, vix: pd.Series, 
                                              spx: pd.Series, vix_history_all: pd.Series = None, 
                                              verbose: bool = True):
        """Main training entry point."""
        # Store data
        self.vix = vix
        self.vix_ml = vix
        self.spx_ml = spx
        self.features = features
        self.vix_history_all = vix_history_all if vix_history_all is not None else vix
        
        # Compute regime statistics
        self.regime_stats_historical = self._compute_regime_statistics(self.vix_history_all)
        
        # Train components
        self._train_anomaly_detector(verbose)
        self._generate_historical_ensemble_scores(verbose)
        
        if self.historical_ensemble_scores is None:
            raise RuntimeError("Historical scores generation failed")
        
        self._train_duration_predictor(verbose)
        
        self.trained = True
        if verbose:
            print("✅ Training complete")
    
    def train_with_features(self, features: pd.DataFrame, vix: pd.Series, spx: pd.Series, 
                      vix_history_all: pd.Series = None, verbose: bool = True):
        """Alias for backward compatibility."""
        self._update_regime_history_after_training(features, vix, spx, vix_history_all, verbose)
    
    def _train_anomaly_detector(self, verbose: bool = True):
        """Train multi-dimensional anomaly detection system."""
        self.anomaly_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05, random_state=RANDOM_STATE
        )
        self.anomaly_detector.train(self.features.fillna(0), verbose=verbose)
    
    def _train_duration_predictor(self, verbose: bool = True):
        """Train regime duration predictor."""
        features_clean = self.features.fillna(0)
        regime_series = self.features['vix_regime']
        regime_change_mask = regime_series != regime_series.shift(-1)
        days_until_change = pd.Series(index=self.features.index, dtype=float)
        
        for idx in self.features.index:
            if idx in regime_change_mask.index:
                future_changes = regime_change_mask.loc[idx:]
                if future_changes.sum() > 0:
                    days_until_change.loc[idx] = (future_changes[future_changes].index[0] - idx).days
        
        valid_mask = days_until_change.notna() & (days_until_change > 0) & (days_until_change <= 30)
        X_train = features_clean[valid_mask]
        y_train = days_until_change[valid_mask]
        
        if len(y_train) > 100:
            self.duration_predictor = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=50,
                min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
            )
            self.duration_predictor.fit(X_train, y_train)
            if verbose:
                print(f"Duration predictor trained on {len(X_train)} samples")
        else:
            self.duration_predictor = None
    
    def _generate_historical_ensemble_scores(self, verbose: bool = True):
        """Generate historical anomaly scores for all training data."""
        if not self.anomaly_detector or not self.anomaly_detector.trained:
            warnings.warn("Anomaly detector not trained, skipping score generation")
            return
        
        scores = []
        for i in range(len(self.features)):
            result = self.anomaly_detector.detect(
                self.features.iloc[[i]], verbose=False
            )
            scores.append(result['ensemble']['score'])
        
        self.historical_ensemble_scores = np.array(scores)
        if verbose:
            print(f"Generated {len(scores)} historical ensemble scores")
    
    # ============================================================================
    # REGIME STATISTICS
    # ============================================================================
    
    def _compute_regime_statistics(self, vix_series: pd.Series) -> dict:
        """Compute comprehensive regime statistics."""
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
            
            vix_regimes = pd.Series(index=vix_series.index, dtype=int)
            for rid in range(len(REGIME_NAMES)):
                lb = REGIME_BOUNDARIES[rid]
                ub = REGIME_BOUNDARIES[rid + 1] if rid < len(REGIME_BOUNDARIES) - 1 else np.inf
                mask = (vix_series >= lb) & (vix_series < ub)
                vix_regimes[mask] = rid
            
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
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _dict_to_series(self, d: dict) -> pd.Series:
        """Convert dict to pandas Series with DatetimeIndex."""
        dates = pd.to_datetime(list(d.keys()))
        return pd.Series(list(d.values()), index=dates)
    
    def _dict_to_dataframe(self, d: dict, columns: list) -> pd.DataFrame:
        """Convert dict to pandas DataFrame with DatetimeIndex."""
        dates = pd.to_datetime(list(next(iter(d.values())).keys()))
        data = {col: list(d[col].values()) for col in columns}
        return pd.DataFrame(data, index=dates)
    
    def _to_iso(self, timestamp):
        """Convert timestamp to ISO format string."""
        return timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)


def main():
    print("❌ Run: python integrated_system_production.py")


if __name__ == "__main__":
    main()