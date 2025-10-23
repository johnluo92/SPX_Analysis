"""
SPX Prediction System - Main Orchestrator
Train models to predict directional moves and range-bound probability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import LOOKBACK_YEARS
from UnifiedDataFetcher import UnifiedDataFetcher
from spx_features import SPXFeatureEngine
from spx_model import SPXModel


class SPXPredictor:
    """SPX prediction orchestrator - trains models and generates forecasts."""
    
    def __init__(self):
        """Initialize SPX Predictor with data fetcher, feature engine, and model."""
        self.fetcher = UnifiedDataFetcher()
        self.feature_engine = SPXFeatureEngine()
        self.model = SPXModel()
        self.features = None
        self.features_scaled = None
    
    def _normalize_timezone(self, data):
        """Remove timezone from any Series or DataFrame."""
        if isinstance(data, pd.Series):
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data = data.copy()
                data.index = data.index.tz_localize(None)
        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data = data.copy()
                data.index = data.index.tz_localize(None)
        return data
    
    def _align_to_daily(self, data, reference_index):
        """
        Align data to daily frequency matching reference index.
        Handles intraday data by taking last value of each day.
        """
        if isinstance(data, pd.Series):
            data = data.copy()
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        
        # Normalize to date only (remove time component)
        data.index = pd.to_datetime(data.index.date)
        
        # If duplicate dates (intraday data), take last value of each day
        if data.index.duplicated().any():
            data = data.groupby(data.index).last()
        
        # Reindex to match reference, forward fill missing days
        data = data.reindex(reference_index, method='ffill')
        
        return data
    
    def fetch_data(self, years: int = LOOKBACK_YEARS):
        """
        Fetch all data needed for SPX prediction.
        
        Args:
            years: Number of years of historical data to fetch
            
        Returns:
            Tuple of (spx, vix, fred, macro) data
        """
        print("\n" + "="*70)
        print("SPX PREDICTION SYSTEM - DATA COLLECTION")
        print("="*70)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"\n📊 Fetching data: {start_str} to {end_str}")
        
        # Fetch core SPX and VIX data
        spx_df = self.fetcher.fetch_spx(start_str, end_str)
        spx = spx_df['Close'].squeeze()
        spx = self._normalize_timezone(spx)
        
        vix = self.fetcher.fetch_vix(start_str, end_str)
        vix = self._normalize_timezone(vix)
        
        # Fetch macro factors (yields, commodities, dollar)
        macro = self.fetcher.fetch_macro(start_str, end_str)
        macro = self._normalize_timezone(macro)
        
        # Fetch FRED economic data
        fred = self.fetcher.fetch_fred_multiple(start_str, end_str)
        fred = self._normalize_timezone(fred)
        
        # Align all data to daily SPX trading calendar
        spx = self._align_to_daily(spx, spx.index)
        vix = self._align_to_daily(vix, spx.index)
        macro = self._align_to_daily(macro, spx.index) if macro is not None else None
        fred = self._align_to_daily(fred, spx.index) if fred is not None else None
        
        print(f"✅ Data loaded: {len(spx)} trading days\n")
        
        return spx, vix, fred, macro
    
    def build_features(self, spx, vix, fred, macro):
        """
        Build feature matrix from raw data.
        
        Args:
            spx: SPX closing prices
            vix: VIX levels
            fred: FRED economic data
            macro: Macro factors
            
        Returns:
            DataFrame of engineered features
        """
        print("🔧 Building features...")
        
        features = self.feature_engine.build(
            spx=spx,
            vix=vix,
            fred=fred,
            macro=macro,
            iv_rv_spread=None  # Using only backward-looking volatility features
        )
        
        print(f"✅ Features built: {features.shape[0]} samples, {features.shape[1]} features\n")
        
        return features
    
    def train(self, years: int = LOOKBACK_YEARS):
        """
        Train SPX prediction models.
        
        Args:
            years: Number of years of historical data to use
            
        Returns:
            DataFrame with model performance summary
        """
        # Step 1: Fetch data
        spx, vix, fred, macro = self.fetch_data(years)
        
        # Step 2: Build features
        features = self.build_features(spx, vix, fred, macro)
        
        # Step 3: Scale features
        features_scaled = self.feature_engine.scale(features)
        
        # Step 4: Train models (includes feature selection)
        self.model.train(features_scaled, spx, use_feature_selection=True)
        
        # Step 5: Store the features that were used
        if self.model.selected_features:
            features_scaled = features_scaled[self.model.selected_features]
        
        self.features = features
        self.features_scaled = features_scaled
        
        print("\n✅ SPX Predictor ready for predictions!")
        
        return self.model.summary()
    
    def predict_current(self):
        """
        Get predictions for current market conditions.
        
        Returns:
            Dictionary of predictions (directional and range probabilities)
        """
        if self.features_scaled is None:
            raise ValueError("Model not trained. Run train() first.")
        
        current_features = self.features_scaled.iloc[[-1]]
        predictions = self.model.predict(current_features)
        
        # Print formatted predictions
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
    
    def get_feature_importance(self, top_n: int = 10):
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of {feature_name: importance_score}
        """
        if self.model is None or not hasattr(self.model, 'feature_importances'):
            return {}
        
        return self.model.get_feature_importance(top_n)
    
    def backtest_signal(self, signal_type: str = 'direction_21d', threshold: float = 0.60):
        """
        Simple backtest: how often was the model right when confident?
        
        Args:
            signal_type: Which prediction to backtest (e.g., 'direction_21d')
            threshold: Confidence threshold (e.g., 0.60 = 60% probability)
        """
        if self.features_scaled is None:
            raise ValueError("Model not trained. Run train() first.")
        
        print(f"\n🔍 Backtesting {signal_type} with {threshold:.0%} confidence threshold...")
        
        # Get the appropriate model
        if signal_type.startswith('direction_'):
            horizon = signal_type.replace('direction_', '')
            model = self.model.directional_models[horizon]
        else:
            model_key = signal_type.replace('range_', '')
            model = self.model.range_models[model_key]
        
        # Get predictions on all historical data
        probs = model.predict_proba(self.features_scaled)[:, 1]
        
        # Filter for confident predictions
        confident_mask = probs >= threshold
        confident_count = confident_mask.sum()
        
        if confident_count == 0:
            print(f"   No predictions above {threshold:.0%} threshold")
            return
        
        print(f"   Found {confident_count} confident predictions ({confident_count/len(probs):.1%} of time)")
        print(f"   Average confidence when signaling: {probs[confident_mask].mean():.1%}")


def main():
    """
    Run SPX prediction system.
    Train models and display current predictions.
    """
    
    print("\n" + "="*70)
    print("SPX PREDICTION SYSTEM - TRAINING")
    print("="*70)
    
    # Initialize and train
    predictor = SPXPredictor()
    summary = predictor.train(years=7)
    
    print("\n📊 MODEL PERFORMANCE SUMMARY:")
    print(summary)
    
    # Get current predictions
    predictions = predictor.predict_current()
    
    # Show top features
    print("\n🔑 TOP FEATURES:")
    top_features = predictor.get_feature_importance(top_n=10)
    for i, (feat, importance) in enumerate(top_features.items(), 1):
        print(f"   {i:2d}. {feat:50s} {importance:.4f}")


if __name__ == "__main__":
    main()