"""VIX Predictor V4 - With Load Capability"""
import pandas as pd
import numpy as np
from datetime import datetime
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
    
    def load_refresh_state(self, filepath: str = './json_data/refresh_state.pkl'):
        """
        Load previously exported refresh state to enable live updates without retraining.
        
        This allows the system to:
        - Use pre-trained anomaly detectors
        - Update prices in real-time
        - Recalculate anomaly scores without full retraining
        """
        print(f"\n{'='*80}")
        print("LOADING CACHED MODEL STATE")
        print(f"{'='*80}")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Refresh state not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            refresh_state = pickle.load(f)
        
        print(f"📦 Loading from: {filepath}")
        print(f"   Size: {filepath.stat().st_size / (1024*1024):.2f} MB")
        print(f"   Exported: {refresh_state['export_timestamp']}")
        
        # Restore anomaly detector
        self.anomaly_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05, 
            random_state=RANDOM_STATE
        )
        self.anomaly_detector.detectors = refresh_state['detectors']
        self.anomaly_detector.scalers = refresh_state['scalers']
        self.anomaly_detector.training_distributions = refresh_state['training_distributions']
        self.anomaly_detector.feature_groups = refresh_state['feature_groups']
        self.anomaly_detector.random_subspaces = refresh_state['random_subspaces']
        self.anomaly_detector.statistical_thresholds = refresh_state['statistical_thresholds']
        
        # CRITICAL FIX: Mark detector as trained
        self.anomaly_detector.trained = True
        
        print(f"   ✅ Loaded {len(self.anomaly_detector.detectors)} anomaly detectors")
        
        # Restore historical data (as pandas Series with DatetimeIndex)
        vix_dict = refresh_state['vix_history']
        spx_dict = refresh_state['spx_history']
        
        # Convert dict back to Series with proper index
        vix_dates = pd.to_datetime(list(vix_dict.keys()))
        spx_dates = pd.to_datetime(list(spx_dict.keys()))
        
        self.vix_ml = pd.Series(list(vix_dict.values()), index=vix_dates)
        self.vix = self.vix_ml.copy()
        self.spx_ml = pd.Series(list(spx_dict.values()), index=spx_dates)
        print(f"   ✅ Loaded {len(self.vix_ml)} days of VIX history")
        print(f"   ✅ Loaded {len(self.spx_ml)} days of SPX history")
        
        # Restore features (as DataFrame with proper index)
        features_dict = refresh_state['last_features']
        feature_columns = refresh_state['feature_columns']
        
        # Reconstruct DataFrame from dict
        feature_dates = pd.to_datetime(list(next(iter(features_dict.values())).keys()))
        feature_data = {col: list(features_dict[col].values()) for col in feature_columns}
        self.features = pd.DataFrame(feature_data, index=feature_dates)
        print(f"   ✅ Loaded {len(self.features.columns)} features")
        
        # Load regime statistics from separate JSON file
        regime_stats_path = Path(filepath).parent / 'regime_statistics.json'
        if regime_stats_path.exists():
            with open(regime_stats_path, 'r') as f:
                self.regime_stats_historical = json.load(f)
            print(f"   ✅ Loaded regime statistics")
        else:
            print(f"   ⚠️  Regime statistics not found, will use defaults")
            self.regime_stats_historical = self._compute_regime_statistics(self.vix_ml)
        
        # Load historical ensemble scores from separate JSON file
        hist_scores_path = Path(filepath).parent / 'historical_anomaly_scores.json'
        if hist_scores_path.exists():
            with open(hist_scores_path, 'r') as f:
                hist_data = json.load(f)
                self.historical_ensemble_scores = np.array(hist_data['ensemble_scores'])
            print(f"   ✅ Loaded {len(self.historical_ensemble_scores)} historical anomaly scores")
        else:
            print(f"   ⚠️  Historical scores not found")
            self.historical_ensemble_scores = None
        
        # DIAGNOSTIC: Verify detector is functional
        try:
            test_result = self.anomaly_detector.detect(self.features.iloc[[-1]], verbose=False)
            print(f"   ✅ Detector validation successful: score={test_result['ensemble']['score']:.3f}")
        except Exception as e:
            print(f"   ❌ Detector validation FAILED: {e}")
            raise RuntimeError(f"Loaded detector is not functional: {e}")
        
        self.trained = True
        print(f"\n✅ MODEL STATE LOADED - READY FOR LIVE UPDATES")
        print(f"{'='*80}\n")
    
    def train_with_features(self, features: pd.DataFrame, vix: pd.Series, spx: pd.Series, 
                          vix_history_all: pd.Series = None, verbose: bool = True):
        """Train predictor with pre-computed features."""
        self.vix, self.vix_ml, self.spx_ml, self.features = vix, vix, spx, features
        self.vix_history_all = vix_history_all if vix_history_all is not None else vix
        self.regime_stats_historical = self._compute_regime_statistics(self.vix_history_all)
        
        self._train_anomaly_detector(verbose)
        self._train_duration_predictor(verbose)
        self._generate_historical_ensemble_scores(verbose)
        
        if self.historical_ensemble_scores is None:
            raise RuntimeError("❌ CRITICAL: Historical scores generation failed!")
        
        self.trained = True
        if verbose: 
            print(f"✅ VIX predictor trained ({len(features.columns)} features)")

    def _generate_historical_ensemble_scores(self, verbose: bool = True):
        """Generate ensemble scores for all historical data."""
        if verbose:
            print(f"\n{'='*80}")
            print("GENERATING HISTORICAL ENSEMBLE SCORES")
            print(f"{'='*80}")
            print(f"Processing {len(self.features)} observations...")
        
        ensemble_scores = []
        batch_size = 100
        
        for i in range(0, len(self.features), batch_size):
            batch = self.features.iloc[i:i+batch_size]
            
            for idx in range(len(batch)):
                try:
                    result = self.anomaly_detector.detect(batch.iloc[[idx]], verbose=False)
                    ensemble_scores.append(result['ensemble']['score'])
                except:
                    ensemble_scores.append(0.0)
            
            if verbose and (i + batch_size) % 500 == 0:
                print(f"   Processed {min(i + batch_size, len(self.features))}/{len(self.features)}...")
        
        self.historical_ensemble_scores = np.array(ensemble_scores)
        
        if verbose:
            print(f"✅ Generated {len(self.historical_ensemble_scores)} historical scores")
            print(f"   Range: [{self.historical_ensemble_scores.min():.4f}, {self.historical_ensemble_scores.max():.4f}]")
            print(f"   Mean: {self.historical_ensemble_scores.mean():.4f}")
            print(f"{'='*80}\n")
    
    def _compute_regime_statistics(self, vix: pd.Series) -> dict:
        """Compute historical regime statistics."""
        stats = {
            'metadata': {
                'total_trading_days': len(vix),
                'start_date': str(vix.index[0].date()),
                'end_date': str(vix.index[-1].date()),
                'generated_at': datetime.now().isoformat()
            },
            'regime_boundaries': REGIME_BOUNDARIES,
            'regimes': []
        }
        
        regime_series = pd.cut(vix, bins=REGIME_BOUNDARIES, labels=[0,1,2,3], include_lowest=True).astype(int)
        
        for regime_id in range(4):
            mask = regime_series == regime_id
            regime_days = vix[mask]
            regime_change = regime_series != regime_series.shift(1)
            episode_id = regime_change.cumsum()
            regime_episodes = regime_series[mask].groupby(episode_id[mask]).size()
            future_regime = regime_series.shift(-5)
            transitions_5d = future_regime[mask].value_counts()
            total_opp = len(future_regime[mask].dropna())
            
            regime_info = {
                'id': regime_id,
                'name': REGIME_NAMES[regime_id],
                'vix_range': [float(REGIME_BOUNDARIES[regime_id]), float(REGIME_BOUNDARIES[regime_id+1])],
                'statistics': {
                    'total_days': int(mask.sum()),
                    'frequency': float(mask.sum() / len(vix)),
                    'mean_duration': float(regime_episodes.mean()),
                    'median_duration': float(regime_episodes.median()),
                    'max_duration': int(regime_episodes.max()) if len(regime_episodes) > 0 else 0,
                    'mean_vix': float(regime_days.mean()),
                    'std_vix': float(regime_days.std())
                },
                'transitions_5d': {
                    'persistence': {
                        'probability': float(transitions_5d.get(regime_id, 0) / total_opp) if total_opp > 0 else 0.0,
                        'observations': int(transitions_5d.get(regime_id, 0)),
                        'total_opportunities': int(total_opp)
                    },
                    'to_other_regimes': {
                        other: {
                            'probability': float(transitions_5d.get(other, 0) / total_opp) if total_opp > 0 else 0.0,
                            'observations': int(transitions_5d.get(other, 0)),
                            'total_opportunities': int(total_opp)
                        } for other in range(4) if other != regime_id
                    }
                }
            }
            stats['regimes'].append(regime_info)
        return stats
    
    def _train_anomaly_detector(self, verbose: bool = True):
        """Train 15-detector anomaly system."""
        self.anomaly_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05, 
            random_state=RANDOM_STATE
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
        X_train, y_train = features_clean[valid_mask], days_until_change[valid_mask]
        
        if len(y_train) > 100:
            self.duration_predictor = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=50,
                min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
            )
            self.duration_predictor.fit(X_train, y_train)
            if verbose:
                print(f"Duration predictor trained ({len(X_train)} samples)")
        else:
            self.duration_predictor = None
    
    def export_refresh_state(self, filepath: str = './json_data/refresh_state.pkl'):
        """
        Export minimal state needed for live refresh (no full retraining).
        
        Saves:
        - Trained anomaly detector components
        - Historical data (last 252 days for rolling calculations)
        - Last computed features
        - Feature metadata
        """
        if not self.trained:
            raise ValueError("Model must be trained before exporting refresh state")
        
        refresh_state = {
            # Anomaly detector components
            'detectors': self.anomaly_detector.detectors,
            'scalers': self.anomaly_detector.scalers,
            'training_distributions': self.anomaly_detector.training_distributions,
            'feature_groups': self.anomaly_detector.feature_groups,
            'random_subspaces': self.anomaly_detector.random_subspaces,
            'statistical_thresholds': self.anomaly_detector.statistical_thresholds,
            
            # Historical data for rolling calculations (last 252 trading days)
            'vix_history': self.vix_ml.tail(252).to_dict(),
            'spx_history': self.spx_ml.tail(252).to_dict(),
            
            # Pre-computed features
            'last_features': self.features.tail(1).to_dict(),
            'feature_columns': self.features.columns.tolist(),
            
            # Metadata
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'vix_last_date': self.vix_ml.index[-1].isoformat() if hasattr(self.vix_ml.index[-1], 'isoformat') else str(self.vix_ml.index[-1]),
            'spx_last_date': self.spx_ml.index[-1].isoformat() if hasattr(self.spx_ml.index[-1], 'isoformat') else str(self.spx_ml.index[-1]),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(refresh_state, f)
        
        file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        
        print(f"✅ Exported refresh state: {filepath}")
        print(f"   📦 Size: {file_size_mb:.2f} MB")
        print(f"   🔢 Detectors: {len(self.anomaly_detector.detectors)}")
        print(f"   📊 Features: {len(self.features.columns)}")
        print(f"   📅 History: {len(self.vix_ml.tail(252))} days")

def main():
    print("❌ Run: python integrated_system_production.py")


if __name__ == "__main__":
    main()