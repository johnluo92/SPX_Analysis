"""
SPX Prediction System
Train models to predict directional moves and range-bound probability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import LOOKBACK_YEARS, MACRO_TICKERS
from UnifiedDataFetcher import UnifiedDataFetcher
from spx_features import SPXFeatureEngine
from spx_model import SPXModel


class SPXPredictor:
    """SPX prediction orchestrator."""
    
    def __init__(self):
        self.fetcher = UnifiedDataFetcher()
        self.feature_engine = SPXFeatureEngine()
        self.model = SPXModel()
        self.features = None
        self.features_scaled = None
    
    def fetch_data(self, years: int = LOOKBACK_YEARS):
        """Fetch all data needed for SPX prediction."""
        print("\n" + "="*70)
        print("SPX PREDICTION SYSTEM")
        print("="*70)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"\n📊 Fetching data: {start_str} to {end_str}")
        
        # Core data
        spx_df = self.fetcher.fetch_spx(start_str, end_str)
        spx = spx_df['Close'].squeeze()
        spx.index = spx.index.tz_localize(None)  # Remove timezone
        
        vix = self.fetcher.fetch_vix(start_str, end_str)
        vix.index = vix.index.tz_localize(None) if vix.index.tz else vix.index
        
        # Macro data
        macro = self.fetcher.fetch_macro(start_str, end_str)
        
        # FRED data
        fred = self.fetcher.fetch_fred_multiple(start_str, end_str)
        
        # Calculate IV-RV spread for features
        print("📊 Calculating IV-RV spread...")
        iv_rv_results = []
        for i in range(30, len(vix) - 30):
            date = vix.index[i]
            vix_val = vix.iloc[i]
            
            future_slice = spx.loc[spx.index >= date]
            if len(future_slice) >= 30:
                future_prices = future_slice.iloc[:30]
                future_returns = np.log(future_prices / future_prices.shift(1))
                realized_future = future_returns.std() * np.sqrt(252) * 100
                
                iv_rv_results.append({
                    'date': date,
                    'spread': vix_val - realized_future
                })
        
        iv_rv_spread = pd.DataFrame(iv_rv_results).set_index('date')['spread']
        
        print(f"✅ Data loaded\n")
        
        return spx, vix, fred, macro, iv_rv_spread
    
    def build_features(self, spx, vix, fred, macro, iv_rv_spread):
        """Build feature matrix."""
        print("🔧 Building features...")
        
        features = self.feature_engine.build(
            spx=spx,
            vix=vix,
            fred=fred,
            macro=macro,
            iv_rv_spread=iv_rv_spread
        )
        
        print(f"✅ Features built: {features.shape[0]} samples, {features.shape[1]} features\n")
        
        return features
    
    def train(self, years: int = LOOKBACK_YEARS):
        """Train SPX prediction models."""
        # Fetch data
        spx, vix, fred, macro, iv_rv_spread = self.fetch_data(years)
        
        # Build features
        features = self.build_features(spx, vix, fred, macro, iv_rv_spread)
        
        # Scale features
        features_scaled = self.feature_engine.scale(features)
        
        # Train models (feature selection happens inside)
        self.model.train(features_scaled, spx, use_feature_selection=True)
        
        # Save only the selected features that were actually used
        if self.model.selected_features:
            features_scaled = features_scaled[self.model.selected_features]
        
        self.features = features
        self.features_scaled = features_scaled
        
        print("\n✅ SPX Predictor ready!")
        
        return self.model.summary()
    
    def predict_current(self):
        """Get predictions for current market conditions."""
        if self.features_scaled is None:
            raise ValueError("Model not trained. Run train() first.")
        
        current_features = self.features_scaled.iloc[[-1]]
        predictions = self.model.predict(current_features)
        
        print("\n" + "="*70)
        print("CURRENT SPX PREDICTIONS")
        print("="*70)
        
        print("\n📈 DIRECTIONAL (Will SPX be higher?):")
        for key, prob in predictions.items():
            if key.startswith('direction_'):
                horizon = key.replace('direction_', '')
                print(f"   {horizon}: {prob:.1%}")
        
        print("\n📊 RANGE-BOUND (Will SPX stay within range?):")
        for key, prob in predictions.items():
            if key.startswith('range_'):
                parts = key.replace('range_', '').split('_')
                horizon = parts[0]
                threshold = parts[1]
                print(f"   {horizon} ±{threshold}: {prob:.1%}")
        
        return predictions
    
    def backtest_signal(self, signal_type: str = 'direction_21d', threshold: float = 0.60):
        """
        Simple backtest: how often was the model right when confident?
        
        Args:
            signal_type: Which prediction to backtest
            threshold: Confidence threshold (e.g., 0.60 = 60% probability)
        """
        if self.features_scaled is None:
            raise ValueError("Model not trained. Run train() first.")
        
        print(f"\n🔍 Backtesting {signal_type} with {threshold:.0%} confidence threshold...")
        
        # Get model
        if signal_type.startswith('direction_'):
            horizon = signal_type.replace('direction_', '')
            model = self.model.directional_models[horizon]
        else:
            model_key = signal_type.replace('range_', '')
            model = self.model.range_models[model_key]
        
        # Get predictions on all data
        probs = model.predict_proba(self.features_scaled)[:, 1]
        
        # Filter for confident predictions
        confident_mask = probs >= threshold
        confident_count = confident_mask.sum()
        
        if confident_count == 0:
            print(f"   No predictions above {threshold:.0%} threshold")
            return
        
        # Get actual outcomes
        spx = self.features.index.to_series()  # Dummy, need actual SPX
        # TODO: Calculate actual win rate for confident signals
        
        print(f"   Found {confident_count} confident predictions ({confident_count/len(probs):.1%} of time)")
        print(f"   Average confidence when signaling: {probs[confident_mask].mean():.1%}")


def main():
    """Run SPX prediction system."""
    predictor = SPXPredictor()
    
    # Train models
    summary = predictor.train(years=7)
    
    print("\n📊 Model Summary:")
    print(summary)
    
    # Current predictions
    predictions = predictor.predict_current()
    
    # Simple backtest
    predictor.backtest_signal('direction_21d', threshold=0.65)


if __name__ == "__main__":
    main()