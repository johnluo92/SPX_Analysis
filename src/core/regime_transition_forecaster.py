"""
Regime Transition Forecaster - Hybrid XGBoost + Anomaly + Heuristics
=====================================================================

Combines multiple signals for probabilistic regime forecasts:

1. XGBoost Base Probabilities (data-driven ML predictions)
2. Anomaly Score Adjustment (stress indicator)
3. Sequential Features (momentum, streaks, acceleration)
4. Historical Transition Matrix (Bayesian prior)
5. Domain Heuristics (futures term structure, yield curve)

Output: 5-day regime transition probabilities with confidence intervals
        and human-readable reasoning

Academic Foundation:
- Bayesian model averaging (combines multiple predictors)
- Regime-switching models (Hamilton, 1989)
- Ensemble forecasting (reduces single-model bias)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RegimeTransitionForecaster:
    """
    Probabilistic regime transition forecaster using hybrid methodology.
    
    Architecture:
    - Base layer: XGBoost regime classifier (supervised learning)
    - Adjustment layer: Anomaly ensemble score (unsupervised stress)
    - Context layer: Sequential features (momentum, streaks)
    - Prior layer: Historical transition matrix (Bayesian prior)
    - Heuristic layer: Domain rules (futures, yield curve)
    
    Final output: Weighted combination with uncertainty quantification
    """
    
    def __init__(
        self,
        xgb_trainer,
        anomaly_detector,
        regime_stats: Dict,
        selected_features: List[str]
    ):
        """
        Initialize forecaster with trained components.
        
        Args:
            xgb_trainer: Trained EnhancedXGBoostTrainer
            anomaly_detector: Trained MultiDimensionalAnomalyDetector
            regime_stats: Historical regime statistics
            selected_features: List of selected feature names
        """
        self.xgb_trainer = xgb_trainer
        self.anomaly_detector = anomaly_detector
        self.regime_stats = regime_stats
        self.selected_features = selected_features
        
        # Extract historical transition matrix
        self.transition_matrix = self._build_transition_matrix()
        
        # Weights for combining signals (can be tuned)
        self.weights = {
            'xgboost_base': 0.45,
            'anomaly_adjustment': 0.25,
            'sequential_context': 0.15,
            'historical_prior': 0.10,
            'heuristic_rules': 0.05,
        }
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Extract historical transition probabilities from regime stats.
        
        Returns 4x4 matrix where [i,j] = P(transition from regime i to j in 5 days)
        """
        n_regimes = len(self.regime_stats['regimes'])
        matrix = np.zeros((n_regimes, n_regimes))
        
        for regime_data in self.regime_stats['regimes']:
            from_regime = regime_data['regime_id']
            
            # Persistence probability
            persist_prob = regime_data['transitions_5d']['persistence']['probability']
            matrix[from_regime, from_regime] = persist_prob
            
            # Transition probabilities to other regimes
            for to_regime, trans_data in regime_data['transitions_5d']['to_other_regimes'].items():
                matrix[from_regime, to_regime] = trans_data['probability']
        
        # Normalize rows (ensure probabilities sum to 1)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums > 0, out=matrix)
        
        return matrix
    
    def forecast_5d_transition(
        self,
        current_features: pd.DataFrame,
        return_reasoning: bool = True,
        return_confidence: bool = True
    ) -> Dict:
        """
        Forecast 5-day regime transition probabilities.
        
        Args:
            current_features: Current feature row (single observation)
            return_reasoning: Include human-readable reasoning
            return_confidence: Calculate confidence intervals
            
        Returns:
            {
                'current_regime': int,
                'current_vix': float,
                'transition_probabilities': {0: 0.05, 1: 0.70, 2: 0.20, 3: 0.05},
                'most_likely_regime': int,
                'confidence': float (0-1),
                'reasoning': List[str],  # If return_reasoning=True
                'confidence_intervals': {...}  # If return_confidence=True
            }
        """
        # 1. Get current regime
        current_vix = float(current_features['vix'].iloc[-1])
        current_regime = self._classify_vix_regime(current_vix)
        
        # 2. XGBoost base probabilities
        xgb_probs = self._get_xgboost_probabilities(current_features)
        
        # 3. Anomaly score adjustment
        anomaly_adjustment = self._get_anomaly_adjustment(current_features)
        
        # 4. Sequential context adjustment
        sequential_adjustment = self._get_sequential_adjustment(current_features, current_regime)
        
        # 5. Historical prior
        historical_prior = self.transition_matrix[current_regime, :]
        
        # 6. Heuristic rules
        heuristic_adjustment = self._get_heuristic_adjustment(current_features, current_regime)
        
        # 7. Combine all signals with weights
        final_probs = (
            self.weights['xgboost_base'] * xgb_probs +
            self.weights['anomaly_adjustment'] * anomaly_adjustment +
            self.weights['sequential_context'] * sequential_adjustment +
            self.weights['historical_prior'] * historical_prior +
            self.weights['heuristic_rules'] * heuristic_adjustment
        )
        
        # Normalize to ensure sum = 1
        final_probs = final_probs / final_probs.sum()
        
        # 8. Calculate confidence
        confidence = self._calculate_confidence(final_probs, xgb_probs, historical_prior)
        
        # 9. Build result
        result = {
            'timestamp': datetime.now().isoformat(),
            'current_regime': current_regime,
            'current_vix': current_vix,
            'transition_probabilities': {
                int(i): float(prob) for i, prob in enumerate(final_probs)
            },
            'most_likely_regime': int(np.argmax(final_probs)),
            'confidence': float(confidence),
        }
        
        # 10. Generate reasoning
        if return_reasoning:
            result['reasoning'] = self._generate_reasoning(
                current_features, current_regime, final_probs,
                xgb_probs, anomaly_adjustment, sequential_adjustment, heuristic_adjustment
            )
        
        # 11. Calculate confidence intervals
        if return_confidence:
            result['confidence_intervals'] = self._calculate_confidence_intervals(
                final_probs, xgb_probs, historical_prior
            )
        
        return result
    
    def _classify_vix_regime(self, vix: float) -> int:
        """Classify VIX into regime (0=Low, 1=Normal, 2=Elevated, 3=Crisis)."""
        if vix < 16.77:
            return 0
        elif vix < 24.40:
            return 1
        elif vix < 39.67:
            return 2
        else:
            return 3
    
    def _get_xgboost_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Get XGBoost regime prediction probabilities."""
        # Filter to selected features
        features_filtered = features[self.selected_features]
        
        # Predict probabilities
        predictions = self.xgb_trainer.predict(features_filtered, return_proba=True)
        
        # Extract probability vector (last row if multiple predictions)
        if predictions['regime'].ndim == 1:
            probs = predictions['regime']
        else:
            probs = predictions['regime'][-1, :]
        
        return probs
    
    def _get_anomaly_adjustment(self, features: pd.DataFrame) -> np.ndarray:
        """
        Adjust probabilities based on anomaly score.
        
        High anomaly → increase probability of transitioning to higher regimes
        """
        # Get current anomaly score
        anomaly_result = self.anomaly_detector.detect(features, verbose=False)
        ensemble_score = anomaly_result['ensemble']['score']
        
        # Create adjustment vector
        # Logic: High anomaly increases crisis probability, decreases low vol probability
        adjustment = np.array([
            max(0, 1 - ensemble_score * 2),  # Regime 0 (Low Vol)
            1 - ensemble_score * 0.5,         # Regime 1 (Normal)
            1 + ensemble_score * 0.5,         # Regime 2 (Elevated)
            1 + ensemble_score * 2,           # Regime 3 (Crisis)
        ])
        
        # Normalize
        adjustment = adjustment / adjustment.sum()
        
        return adjustment
    
    def _get_sequential_adjustment(
        self,
        features: pd.DataFrame,
        current_regime: int
    ) -> np.ndarray:
        """
        Adjust based on momentum, streaks, and acceleration.
        
        Strong upward momentum → increase transition to higher regimes
        Strong downward momentum → increase transition to lower regimes
        """
        adjustment = np.ones(4)
        
        # VIX momentum
        if 'vix_momentum_z_10d' in features.columns:
            momentum = float(features['vix_momentum_z_10d'].iloc[-1])
            
            if momentum > 1.0:  # Strong upward momentum
                adjustment[min(current_regime + 1, 3)] *= 1.5
                adjustment[max(current_regime - 1, 0)] *= 0.5
            elif momentum < -1.0:  # Strong downward momentum
                adjustment[max(current_regime - 1, 0)] *= 1.5
                adjustment[min(current_regime + 1, 3)] *= 0.5
        
        # VIX acceleration
        if 'vix_accel_5d' in features.columns:
            accel = float(features['vix_accel_5d'].iloc[-1])
            
            if abs(accel) > 2.0:  # High acceleration (either direction)
                # Increase uncertainty → spread probability
                adjustment = np.ones(4)
        
        # Anomaly streak
        if 'anomaly_streak_5d' in features.columns:
            streak = float(features['anomaly_streak_5d'].iloc[-1])
            
            if streak >= 3:  # Persistent anomaly
                # Likely to transition to higher regime
                adjustment[min(current_regime + 1, 3)] *= 1.3
        
        # Normalize
        adjustment = adjustment / adjustment.sum()
        
        return adjustment
    
    def _get_heuristic_adjustment(
        self,
        features: pd.DataFrame,
        current_regime: int
    ) -> np.ndarray:
        """
        Apply domain-specific heuristic rules.
        
        Rules:
        1. VIX futures backwardation → crisis signal
        2. Yield curve inversion → elevated stress
        3. SKEW > 150 → tail risk elevated
        4. Bond vol spike → systemic stress
        """
        adjustment = np.ones(4)
        
        # Rule 1: VIX Futures Term Structure
        if 'VX1-VX2' in features.columns:
            vx_spread = float(features['VX1-VX2'].iloc[-1])
            
            if vx_spread < -2:  # Strong backwardation
                # Increase crisis probability
                adjustment[3] *= 1.5
                adjustment[0] *= 0.5
        
        # Rule 2: Yield Curve Inversion
        if 'yield_10y2y' in features.columns:
            yield_spread = float(features['yield_10y2y'].iloc[-1])
            
            if yield_spread < -0.5:  # Deep inversion
                # Increase elevated/crisis probability
                adjustment[2] *= 1.3
                adjustment[3] *= 1.2
        
        # Rule 3: SKEW Tail Risk
        if 'SKEW' in features.columns:
            skew = float(features['SKEW'].iloc[-1])
            
            if skew > 150:  # High tail risk
                # Increase crisis probability
                adjustment[3] *= 1.3
        
        # Rule 4: Bond Volatility
        if 'VXTLT' in features.columns and 'vix' in features.columns:
            vxtlt = float(features['VXTLT'].iloc[-1])
            vix = float(features['vix'].iloc[-1])
            
            if vxtlt > vix:  # Bond vol > equity vol (rare, systemic)
                # Strong crisis signal
                adjustment[3] *= 1.8
                adjustment[0] *= 0.3
        
        # Normalize
        adjustment = adjustment / adjustment.sum()
        
        return adjustment
    
    def _calculate_confidence(
        self,
        final_probs: np.ndarray,
        xgb_probs: np.ndarray,
        historical_prior: np.ndarray
    ) -> float:
        """
        Calculate forecast confidence based on signal agreement.
        
        High confidence when:
        1. XGBoost and historical prior agree
        2. Probabilities are concentrated (not uniform)
        3. Sequential signals align
        
        Returns: confidence score 0-1
        """
        # Measure 1: Concentration (entropy-based)
        # Low entropy = high confidence
        entropy = -np.sum(final_probs * np.log(final_probs + 1e-10))
        max_entropy = np.log(4)  # Uniform distribution entropy
        concentration_score = 1 - (entropy / max_entropy)
        
        # Measure 2: Agreement between XGBoost and historical
        # Use KL divergence (lower = more agreement)
        kl_div = np.sum(xgb_probs * np.log((xgb_probs + 1e-10) / (historical_prior + 1e-10)))
        agreement_score = np.exp(-kl_div)  # Convert to 0-1 scale
        
        # Measure 3: Strength of most likely regime
        max_prob = np.max(final_probs)
        strength_score = max_prob
        
        # Combine measures
        confidence = (
            0.4 * concentration_score +
            0.3 * agreement_score +
            0.3 * strength_score
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def _calculate_confidence_intervals(
        self,
        final_probs: np.ndarray,
        xgb_probs: np.ndarray,
        historical_prior: np.ndarray
    ) -> Dict:
        """
        Calculate 90% confidence intervals for each regime probability.
        
        Uses bootstrap-style uncertainty estimation.
        """
        intervals = {}
        
        for regime in range(4):
            # Estimate variance from signal disagreement
            probs_list = [
                final_probs[regime],
                xgb_probs[regime],
                historical_prior[regime],
            ]
            
            mean_prob = np.mean(probs_list)
            std_prob = np.std(probs_list)
            
            # 90% CI (1.645 * std for normal distribution)
            lower = max(0, mean_prob - 1.645 * std_prob)
            upper = min(1, mean_prob + 1.645 * std_prob)
            
            intervals[regime] = {
                'lower': float(lower),
                'upper': float(upper),
                'width': float(upper - lower),
            }
        
        return intervals
    
    def _generate_reasoning(
        self,
        features: pd.DataFrame,
        current_regime: int,
        final_probs: np.ndarray,
        xgb_probs: np.ndarray,
        anomaly_adj: np.ndarray,
        sequential_adj: np.ndarray,
        heuristic_adj: np.ndarray
    ) -> List[str]:
        """
        Generate human-readable reasoning for forecast.
        
        Explains which signals contributed most to the forecast.
        """
        reasoning = []
        
        # Current state
        current_vix = float(features['vix'].iloc[-1])
        regime_names = {0: "Low Volatility", 1: "Normal", 2: "Elevated", 3: "Crisis"}
        reasoning.append(f"Current state: VIX={current_vix:.2f} ({regime_names[current_regime]})")
        
        # Most likely outcome
        most_likely = int(np.argmax(final_probs))
        prob = final_probs[most_likely] * 100
        
        if most_likely == current_regime:
            reasoning.append(f"Most likely: Persist in {regime_names[current_regime]} ({prob:.0f}% probability)")
        else:
            reasoning.append(f"Most likely: Transition to {regime_names[most_likely]} ({prob:.0f}% probability)")
        
        # XGBoost signal
        xgb_most_likely = int(np.argmax(xgb_probs))
        xgb_prob = xgb_probs[xgb_most_likely] * 100
        reasoning.append(f"ML model predicts: {regime_names[xgb_most_likely]} ({xgb_prob:.0f}%)")
        
        # Anomaly signal
        anomaly_result = self.anomaly_detector.detect(features, verbose=False)
        ensemble_score = anomaly_result['ensemble']['score']
        
        if ensemble_score > 0.78:
            reasoning.append(f"⚠️ HIGH anomaly detected ({ensemble_score:.1%}) - elevated transition risk")
        elif ensemble_score > 0.70:
            reasoning.append(f"MODERATE anomaly ({ensemble_score:.1%}) - watching closely")
        else:
            reasoning.append(f"Normal conditions ({ensemble_score:.1%} anomaly score)")
        
        # Momentum signal
        if 'vix_momentum_z_10d' in features.columns:
            momentum = float(features['vix_momentum_z_10d'].iloc[-1])
            
            if momentum > 1.0:
                reasoning.append(f"VIX momentum POSITIVE ({momentum:.1f}σ) - upward pressure")
            elif momentum < -1.0:
                reasoning.append(f"VIX momentum NEGATIVE ({momentum:.1f}σ) - downward pressure")
        
        # Heuristic signals
        if 'VX1-VX2' in features.columns:
            vx_spread = float(features['VX1-VX2'].iloc[-1])
            
            if vx_spread < -2:
                reasoning.append(f"⚠️ VIX futures in BACKWARDATION ({vx_spread:.2f}) - crisis signal")
            elif vx_spread > 2:
                reasoning.append(f"VIX futures in CONTANGO ({vx_spread:.2f}) - calm markets")
        
        if 'yield_10y2y' in features.columns:
            yield_spread = float(features['yield_10y2y'].iloc[-1])
            
            if yield_spread < 0:
                reasoning.append(f"⚠️ Yield curve INVERTED ({yield_spread:.2f}%) - recession risk")
        
        if 'SKEW' in features.columns:
            skew = float(features['SKEW'].iloc[-1])
            
            if skew > 150:
                reasoning.append(f"⚠️ SKEW elevated ({skew:.0f}) - tail risk premium high")
        
        return reasoning
    
    def backtest_forecasts(
        self,
        features_history: pd.DataFrame,
        vix_history: pd.Series,
        lookback_days: int = 252
    ) -> Dict:
        """
        Backtest forecaster on historical data.
        
        Tests:
        1. Regime persistence prediction accuracy
        2. Transition prediction accuracy
        3. Calibration (do 70% probabilities occur 70% of the time?)
        
        Args:
            features_history: Historical features
            vix_history: Historical VIX
            lookback_days: Days to test
            
        Returns:
            Backtest metrics
        """
        from sklearn.metrics import log_loss, brier_score_loss
        
        # Get last N days
        features_test = features_history.iloc[-lookback_days:]
        vix_test = vix_history.iloc[-lookback_days:]
        
        predictions = []
        actuals = []
        
        for i in range(len(features_test) - 5):
            # Forecast from day i
            current_features = features_test.iloc[[i]]
            forecast = self.forecast_5d_transition(current_features, return_reasoning=False)
            
            # Actual regime 5 days later
            actual_vix = vix_test.iloc[i + 5]
            actual_regime = self._classify_vix_regime(actual_vix)
            
            predictions.append(forecast['transition_probabilities'])
            actuals.append(actual_regime)
        
        # Convert to arrays
        pred_probs = np.array([[p[j] for j in range(4)] for p in predictions])
        actuals_array = np.array(actuals)
        
        # Metrics
        logloss = log_loss(actuals_array, pred_probs)
        
        # Accuracy (predicted most likely == actual)
        pred_regimes = pred_probs.argmax(axis=1)
        accuracy = (pred_regimes == actuals_array).mean()
        
        # Calibration (binned)
        calibration = self._calculate_calibration(pred_probs, actuals_array)
        
        return {
            'log_loss': float(logloss),
            'accuracy': float(accuracy),
            'calibration': calibration,
            'n_samples': len(predictions),
        }
    
    def _calculate_calibration(
        self,
        pred_probs: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Calculate probability calibration.
        
        Good calibration: predictions with 70% confidence should be correct 70% of time.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        calibration_curve = []
        
        for i in range(n_bins):
            bin_mask = (pred_probs.max(axis=1) >= bins[i]) & (pred_probs.max(axis=1) < bins[i + 1])
            
            if bin_mask.sum() > 0:
                bin_pred_regimes = pred_probs[bin_mask].argmax(axis=1)
                bin_actuals = actuals[bin_mask]
                bin_accuracy = (bin_pred_regimes == bin_actuals).mean()
                
                calibration_curve.append({
                    'confidence_bin': float(bin_centers[i]),
                    'actual_accuracy': float(bin_accuracy),
                    'count': int(bin_mask.sum()),
                })
        
        return calibration_curve


# ===== Convenience Functions =====

def create_forecaster(
    xgb_trainer,
    integrated_system
) -> RegimeTransitionForecaster:
    """
    Create forecaster from trained components.
    
    Usage:
        from xgboost_trainer_v2 import EnhancedXGBoostTrainer
        from integrated_system_production import IntegratedMarketSystemV4
        from regime_transition_forecaster import create_forecaster
        
        # Load models
        xgb_trainer = EnhancedXGBoostTrainer.load('./models')
        
        # Load selected features
        with open('./models/selected_features_v2.txt', 'r') as f:
            selected_features = [line.strip() for line in f]
        
        # Create forecaster
        forecaster = create_forecaster(xgb_trainer, system)
        
        # Forecast
        current_features = system.orchestrator.features.iloc[[-1]]
        forecast = forecaster.forecast_5d_transition(current_features)
    """
    # Load selected features
    from pathlib import Path
    
    selected_features_path = Path('./models/selected_features_v2.txt')
    if selected_features_path.exists():
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f]
    else:
        # Fallback to XGBoost feature columns
        selected_features = xgb_trainer.feature_columns
    
    forecaster = RegimeTransitionForecaster(
        xgb_trainer=xgb_trainer,
        anomaly_detector=integrated_system.orchestrator.anomaly_detector,
        regime_stats=integrated_system.orchestrator.regime_stats,
        selected_features=selected_features
    )
    
    return forecaster


if __name__ == "__main__":
    print("Regime Transition Forecaster")
    print("\nHybrid methodology combining:")
    print("  • XGBoost ML predictions")
    print("  • Anomaly detection scores")
    print("  • Sequential momentum features")
    print("  • Historical transition matrix")
    print("  • Domain heuristic rules")
    print("\nOutputs probabilistic 5-day forecasts with confidence intervals and reasoning.")
    print("\nCreate with: forecaster = create_forecaster(xgb_trainer, system)")