"""
Feature Engineering - Declarative & Clean
Twin Pillars: Simplicity & Consistency
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

from config import (
    LONG_HORIZON_SECTORS, RS_WINDOWS_SHORT, RS_WINDOWS_LONG,
    MACRO_WINDOWS, VIX_WINDOWS
)


class FeatureEngine:
    """Build features from sector and macro data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def build(self, sectors: pd.DataFrame, macro: pd.DataFrame, 
              vix: pd.Series) -> pd.DataFrame:
        """Build all features."""
        features = pd.concat([
            self._relative_strength(sectors),
            self._macro_features(macro),
            self._vix_features(vix),
            self._seasonality(sectors.index)
        ], axis=1)
        
        return features.dropna()
    
    def _relative_strength(self, sectors: pd.DataFrame) -> pd.DataFrame:
        """Relative strength vs SPY."""
        features = pd.DataFrame(index=sectors.index)
        spy = sectors['SPY'].squeeze()
        
        for ticker in sectors.columns:
            if ticker == 'SPY':
                continue
            
            windows = RS_WINDOWS_LONG if ticker in LONG_HORIZON_SECTORS else RS_WINDOWS_SHORT
            sector = sectors[ticker].squeeze()
            
            for window in windows:
                sector_ret = sector.pct_change(window)
                spy_ret = spy.pct_change(window)
                features[f'{ticker}_rs_{window}'] = (sector_ret - spy_ret) * 100
        
        return features
    
    def _macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Macro velocity and interactions."""
        features = pd.DataFrame(index=macro.index)
        
        # Velocity
        for col in macro.columns:
            if col == '5Y Treasury':
                continue
            for window in MACRO_WINDOWS:
                features[f'{col}_vel_{window}'] = macro[col].squeeze().pct_change(window) * 100
        
        # Yield curve
        if '10Y Treasury' in macro.columns and '5Y Treasury' in macro.columns:
            ten_y = macro['10Y Treasury'].squeeze()
            five_y = macro['5Y Treasury'].squeeze()
            slope = ten_y - five_y
            features['yield_slope'] = slope
            for window in MACRO_WINDOWS:
                features[f'yield_slope_vel_{window}'] = slope.diff(window)
        
        # Interactions
        if 'Gold' in macro.columns and 'Dollar' in macro.columns:
            features['gold_dollar_ratio'] = macro['Gold'].squeeze() / macro['Dollar'].squeeze()
        
        if 'Crude Oil' in macro.columns and 'Dollar' in macro.columns:
            oil_vel = macro['Crude Oil'].squeeze().pct_change(21)
            dollar_vel = macro['Dollar'].squeeze().pct_change(21)
            features['oil_dollar_interact'] = oil_vel * dollar_vel * 100
        
        return features
    
    def _vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """VIX level and changes."""
        features = pd.DataFrame(index=vix.index)
        vix = vix.squeeze()
        features['vix'] = vix
        features['vix_regime'] = pd.cut(vix, bins=[0, 15, 25, 35, 100], labels=[0, 1, 2, 3]).astype(int)
        
        for window in VIX_WINDOWS:
            features[f'vix_vel_{window}'] = vix.diff(window)
        
        features['vix_vol_21'] = vix.diff(5).rolling(21).std()
        
        return features
    
    def _seasonality(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Month and quarter."""
        features = pd.DataFrame(index=index)
        features['month'] = index.month
        features['quarter'] = index.quarter
        return features
    
    def scale(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize features."""
        scaled = self.scaler.fit_transform(features)
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted scaler."""
        scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)