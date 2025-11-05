"""
SPX Prediction System V2 - Enhanced with ML-Based Confidence
Addresses: Weak prediction signals, meaningful confidence metrics, data contract compliance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import LOOKBACK_YEARS
from UnifiedDataFetcher import UnifiedDataFetcher
from spx_features_v2 import SPXFeatureEngineV2
from spx_model_v2 import SPXModelV2


class SPXPredictorV2:
    """SPX prediction with enhanced confidence and diagnostic output."""
    
    def __init__(self, verbose: bool = False):
        self.fetcher = UnifiedDataFetcher(log_level="WARNING" if not verbose else "INFO")
        self.feature_engine = SPXFeatureEngineV2()
        self.model = SPXModelV2()
        self.features = None
        self.features_scaled = None
        self.spx = None
        self.vix = None
        self.verbose = verbose
        
        # Track prediction quality metrics
        self.prediction_diagnostics = {
            'model_certainty': {},  # How confident is the ensemble?
            'signal_strength': {},  # How far from random?
            'feature_stability': None  # Are top features consistent?
        }
    
    def _print(self, msg: str, level: str = "info"):
        """Conditional printing based on verbose flag."""
        if self.verbose or level in ["critical", "warning"]:
            prefix = {"critical": "❌", "warning": "⚠️ ", "info": "  "}
            print(f"{prefix.get(level, '  ')}{msg}")
    
    def _align_to_daily(self, data, reference_index):
        """Align data to daily frequency."""
        if data is None:
            return None
        
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.copy()
        
        data = data.reindex(reference_index, method='ffill')
        return data
    
    def _calculate_ml_confidence(self, probability: float, prediction_type: str) -> dict:
        """
        Calculate ML-based confidence using Random Forest probability distributions.
        
        Returns both a categorical confidence level and quantitative metrics.
        
        Args:
            probability: Model's predicted probability
            prediction_type: 'directional' or 'range_bound'
        
        Returns:
            dict with 'level' (HIGH/MODERATE/LOW), 'score', and 'interpretation'
        """
        # Distance from neutral (0.5)
        distance_from_neutral = abs(probability - 0.5)
        
        # Convert to 0-1 confidence score
        confidence_score = distance_from_neutral * 2  # Maps [0, 0.5] to [0, 1]
        
        # Categorical levels with ML-appropriate thresholds
        # Based on Random Forest probability calibration research
        if prediction_type == 'directional':
            # Directional predictions: stricter thresholds
            if confidence_score >= 0.35:  # prob >= 0.675 or <= 0.325
                level = 'HIGH'
                interpretation = 'Strong ensemble consensus'
            elif confidence_score >= 0.15:  # prob >= 0.575 or <= 0.425
                level = 'MODERATE'
                interpretation = 'Moderate ensemble agreement'
            else:
                level = 'LOW'
                interpretation = 'Weak signal, close to random'
        else:
            # Range-bound predictions: slightly relaxed thresholds
            if confidence_score >= 0.30:
                level = 'HIGH'
                interpretation = 'Strong regime signal'
            elif confidence_score >= 0.10:
                level = 'MODERATE'
                interpretation = 'Detectable pattern'
            else:
                level = 'LOW'
                interpretation = 'Unclear regime signal'
        
        return {
            'level': level,
            'score': round(confidence_score, 3),
            'interpretation': interpretation,
            'probability': round(probability, 4)
        }
    
    def fetch_data(self, years: int = LOOKBACK_YEARS):
        """Fetch all data needed for SPX prediction."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self._print(f"📊 Fetching {years} years of data...", "info")
        
        # Core data
        spx_df = self.fetcher.fetch_spx(start_str, end_str, lookback_buffer_days=365)
        if spx_df is None:
            raise ValueError("SPX data fetch failed - cannot proceed")
        
        spx = spx_df['Close'].squeeze()
        
        vix = self.fetcher.fetch_vix(start_str, end_str, lookback_buffer_days=365)
        if vix is None:
            raise ValueError("VIX data fetch failed - cannot proceed")
        
        # Optional data
        macro = self.fetcher.fetch_macro(start_str, end_str, lookback_buffer_days=365)
        fred = self.fetcher.fetch_fred_multiple(start_str, end_str, lookback_buffer_days=365)
        commodities = self.fetcher.fetch_commodities_fred(start_str, end_str, lookback_buffer_days=365)
        
        # Align all to SPX index
        spx = self._align_to_daily(spx, spx.index)
        vix = self._align_to_daily(vix, spx.index)
        macro = self._align_to_daily(macro, spx.index) if macro is not None else None
        fred = self._align_to_daily(fred, spx.index) if fred is not None else None
        commodities = self._align_to_daily(commodities, spx.index) if commodities is not None else None
        
        self._print(f"✅ Loaded {len(spx)} trading days", "info")
        if commodities is not None:
            self._print(f"✅ Commodity data: {len(commodities.columns)} features", "info")
        
        self.spx = spx
        self.vix = vix
        
        return spx, vix, fred, macro, commodities
    
    def build_features(self, spx, vix, fred, macro, commodities):
        """Build feature matrix."""
        features = self.feature_engine.build(
            spx=spx,
            vix=vix,
            fred=fred,
            macro=macro,
            commodities=commodities
        )
        
        return features
    
    def train(self, years: int = LOOKBACK_YEARS, 
             use_cv: bool = True,
             n_folds: int = 5):
        """Train SPX prediction models with enhanced diagnostics."""
        print(f"\n🔬 Training SPX Predictor ({years}y, {'CV' if use_cv else 'simple'})...")
        
        # Step 1: Fetch data
        spx, vix, fred, macro, commodities = self.fetch_data(years)
        
        # Step 2: Build features
        features = self.build_features(spx, vix, fred, macro, commodities)
        
        # Step 3: Scale features
        features_scaled = self.feature_engine.scale(features, fit=True)
        
        # Step 4: Train with proper validation
        if use_cv:
            cv_results = self.model.train_with_time_series_cv(
                features_scaled, 
                spx, 
                n_splits=n_folds,
                use_feature_selection=True,
                verbose=self.verbose
            )
        else:
            self.model.train_simple(
                features_scaled,
                spx,
                test_split=0.2,
                use_feature_selection=True,
                verbose=self.verbose
            )
            cv_results = None
        
        # Store features
        if self.model.selected_features:
            features_scaled = features_scaled[self.model.selected_features]
        
        self.features = features
        self.features_scaled = features_scaled
        
        # Diagnostic summary
        if cv_results is not None:
            avg_acc = cv_results['test_acc'].mean()
            beat_naive = cv_results['beat_naive'].sum()
            avg_gap = cv_results['gap'].mean()
            acc_std = cv_results['test_acc'].std()
            
            print(f"✅ SPX Predictor trained:")
            print(f"   • Test accuracy: {avg_acc:.3f} (naive: 0.500)")
            print(f"   • Beat naive: {beat_naive}/{n_folds} folds")
            print(f"   • Avg gap: {avg_gap:+.3f}")
            print(f"   • Stability (std): {acc_std:.3f}")
            
            # Enhanced diagnostics
            if avg_acc < 0.510:
                print(f"   ⚠️  WEAK SIGNAL: Model barely beats random (accuracy < 51%)")
                print(f"   → Consider: Feature engineering, different horizons, regime-specific models")
            elif avg_acc < 0.530:
                print(f"   ⚠️  MODEST SIGNAL: Model shows weak edge (accuracy 51-53%)")
                print(f"   → Consider: Feature selection, ensemble refinement")
            else:
                print(f"   ✅ STRONG SIGNAL: Model shows meaningful edge (accuracy > 53%)")
            
            if abs(avg_gap) > 0.15:
                print(f"   ⚠️  OVERFITTING: Large train/test gap ({avg_gap:.3f})")
                print(f"   → Consider: Reduce model complexity, increase regularization")
            
            if acc_std > 0.10:
                print(f"   ⚠️  UNSTABLE: High variance across folds ({acc_std:.3f})")
                print(f"   → Consider: More data, simpler features, different validation strategy")
        else:
            print(f"✅ SPX Predictor trained (simple split)")
        
        return cv_results if use_cv else None
    
    def predict_current(self, export_json: bool = False):
        """Get predictions with ML-based confidence metrics."""
        if self.features_scaled is None:
            raise ValueError("Model not trained. Run train() first.")
        
        current_features = self.features_scaled.iloc[[-1]]
        raw_predictions = self.model.predict(current_features)
        
        # Build enriched predictions with ML-based confidence
        enriched_predictions = {
            'directional': {},
            'range_bound': {}
        }
        
        # Track diagnostics
        probabilities = []
        
        for key, prob in raw_predictions.items():
            if key.startswith('direction_'):
                horizon = key.replace('direction_', '')
                confidence_info = self._calculate_ml_confidence(prob, 'directional')
                
                enriched_predictions['directional'][horizon] = {
                    'probability_up': float(prob),
                    'direction': 'UP' if prob > 0.5 else 'DOWN',
                    'confidence': confidence_info['level'],
                    'confidence_score': confidence_info['score'],
                    'interpretation': confidence_info['interpretation']
                }
                probabilities.append(prob)
                
            elif key.startswith('range_'):
                parts = key.replace('range_', '').split('_')
                horizon = parts[0]
                threshold = parts[1]
                range_key = f"{horizon}_pm{threshold}"
                confidence_info = self._calculate_ml_confidence(prob, 'range_bound')
                
                enriched_predictions['range_bound'][range_key] = {
                    'probability_in_range': float(prob),
                    'expectation': 'RANGE-BOUND' if prob > 0.5 else 'BREAKOUT',
                    'confidence': confidence_info['level'],
                    'confidence_score': confidence_info['score'],
                    'interpretation': confidence_info['interpretation']
                }
                probabilities.append(prob)
        
        # Overall prediction quality diagnostic
        if probabilities:
            avg_distance_from_neutral = np.mean([abs(p - 0.5) for p in probabilities])
            self.prediction_diagnostics['signal_strength']['current'] = avg_distance_from_neutral
            
            if avg_distance_from_neutral < 0.05:
                signal_quality = "VERY WEAK - Models are uncertain"
            elif avg_distance_from_neutral < 0.10:
                signal_quality = "WEAK - Low conviction signals"
            elif avg_distance_from_neutral < 0.20:
                signal_quality = "MODERATE - Some conviction"
            else:
                signal_quality = "STRONG - High conviction signals"
            
            self.prediction_diagnostics['signal_strength']['quality'] = signal_quality
        
        if self.verbose:
            print("\n" + "="*70)
            print("CURRENT SPX PREDICTIONS (ML-Based Confidence)")
            print("="*70)
            
            print("\n📈 DIRECTIONAL (Will SPX be higher?):")
            for horizon, data in sorted(enriched_predictions['directional'].items()):
                prob = data['probability_up']
                direction = data['direction']
                confidence = data['confidence']
                conf_score = data['confidence_score']
                interp = data['interpretation']
                print(f"   {horizon:4s}: {prob:5.1%} → {direction:4s} | {confidence:8s} (score: {conf_score:.3f})")
                print(f"         {interp}")
            
            print("\n📊 RANGE-BOUND (Will SPX stay within range?):")
            for range_key, data in sorted(enriched_predictions['range_bound'].items()):
                prob = data['probability_in_range']
                expectation = data['expectation']
                confidence = data['confidence']
                conf_score = data['confidence_score']
                interp = data['interpretation']
                print(f"   {range_key:12s}: {prob:5.1%} → {expectation:11s} | {confidence:8s} (score: {conf_score:.3f})")
                print(f"                  {interp}")
            
            print(f"\n📊 OVERALL SIGNAL QUALITY: {self.prediction_diagnostics['signal_strength']['quality']}")
            print(f"   Avg distance from neutral: {self.prediction_diagnostics['signal_strength']['current']:.3f}")
        
        if export_json:
            spx_current = float(self.spx.iloc[-1])
            
            export_data = {
                'current_spx': spx_current,
                'timestamp': datetime.now().isoformat(),
                'predictions': enriched_predictions,
                'diagnostics': {
                    'signal_strength': self.prediction_diagnostics['signal_strength'],
                    'model_note': 'Confidence based on Random Forest probability calibration'
                }
            }
            
            import json
            with open('./json_data/spx_predictions.json', 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print("✅ SPX predictions exported to spx_predictions.json")
        
        return enriched_predictions
    
    def get_feature_importance(self, top_n: int = 20):
        """Get and display top features with stability analysis."""
        if self.model.feature_importances is None:
            return {}
        
        top_features = self.model.get_feature_importance(top_n)
        
        # Analyze feature stability (are top features consistently important?)
        if len(top_features) > 0:
            top_values = list(top_features.values())
            if len(top_values) >= 5:
                top5_sum = sum(top_values[:5])
                concentration = top5_sum / sum(top_values) if sum(top_values) > 0 else 0
                
                if concentration > 0.7:
                    stability = "HIGH - Top 5 features dominate"
                elif concentration > 0.5:
                    stability = "MODERATE - Balanced importance"
                else:
                    stability = "LOW - Diffuse importance"
                
                self.prediction_diagnostics['feature_stability'] = {
                    'concentration': concentration,
                    'stability': stability
                }
        
        if self.verbose:
            print("\n" + "="*70)
            print(f"TOP {top_n} FEATURES")
            print("="*70)
            
            for i, (feat, importance) in enumerate(top_features.items(), 1):
                print(f"   {i:2d}. {feat:45s} {importance:.4f}")
            
            if self.prediction_diagnostics['feature_stability']:
                print(f"\n   Feature Stability: {self.prediction_diagnostics['feature_stability']['stability']}")
                print(f"   Top-5 Concentration: {self.prediction_diagnostics['feature_stability']['concentration']:.2%}")
        
        return top_features
    
    def export_dashboard_data(self, output_path: str = './json_data/spx_analysis.json'):
        """Export comprehensive data for dashboard with diagnostics."""
        if self.features_scaled is None:
            raise ValueError("Model not trained. Run train() first.")
        
        predictions = self.predict_current(export_json=False)
        
        cv_summary = None
        if self.model.validation_results is not None:
            df = self.model.validation_results
            cv_summary = {
                'avg_test_acc': float(df['test_acc'].mean()),
                'avg_naive_acc': float(df['naive_acc'].mean()),
                'improvement': float(df['test_acc'].mean() - df['naive_acc'].mean()),
                'beat_naive_count': int(df['beat_naive'].sum()),
                'total_folds': len(df),
                'avg_gap': float(df['gap'].mean()),
                'stability_std': float(df['test_acc'].std()),
                'quality_assessment': self._assess_model_quality(df)
            }
        
        top_features = self.get_feature_importance(top_n=15)
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'spx_current': float(self.spx.iloc[-1]),
                'spx_date': str(self.spx.index[-1].date()),
                'training_days': len(self.spx),
                'n_features': len(self.features_scaled.columns)
            },
            'predictions': predictions,
            'validation': cv_summary,
            'top_features': {k: float(v) for k, v in top_features.items()},
            'diagnostics': self.prediction_diagnostics
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ SPX analysis exported to {output_path}")
        
        return export_data
    
    def _assess_model_quality(self, cv_results):
        """Assess overall model quality from CV results."""
        avg_acc = cv_results['test_acc'].mean()
        beat_naive = cv_results['beat_naive'].sum()
        total = len(cv_results)
        gap = abs(cv_results['gap'].mean())
        
        if avg_acc >= 0.530 and beat_naive >= total * 0.8 and gap < 0.10:
            return "STRONG - Reliable predictive signal"
        elif avg_acc >= 0.515 and beat_naive >= total * 0.6 and gap < 0.15:
            return "MODERATE - Detectable but weak signal"
        else:
            return "WEAK - Marginal or unreliable signal"


def main():
    """Run enhanced SPX Predictor V2."""
    
    print("\n" + "="*70)
    print("SPX PREDICTION SYSTEM V2 - Enhanced")
    print("="*70)
    
    predictor = SPXPredictorV2(verbose=True)
    
    cv_results = predictor.train(years=LOOKBACK_YEARS, use_cv=True, n_folds=5)
    
    predictor.get_feature_importance(top_n=20)
    
    predictor.predict_current(export_json=True)
    
    predictor.export_dashboard_data()


if __name__ == "__main__":
    main()