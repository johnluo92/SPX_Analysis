"""
SPX Feature Engineering - MINIMAL CORE FEATURES
Features specifically designed for SPX directional and range prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import VIX_WINDOWS, FRED_WINDOWS


class SPXFeatureEngine:
    """Build features for SPX prediction - CORE ONLY."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def build(self, 
              spx: pd.Series,
              vix: pd.Series,
              fred: pd.DataFrame = None,
              macro: pd.DataFrame = None,
              iv_rv_spread: pd.Series = None,
              use_iv_rv_cheat=False) -> pd.DataFrame:
        """
        Build core SPX prediction features.
        
        All inputs must have timezone-naive DatetimeIndex.
        
        Args:
            spx: SPX closing prices
            vix: VIX levels
            fred: FRED economic data (T10YIE, T5YIFR, T10Y2Y)
            macro: Macro factors (yields, oil, gold, dollar)
            iv_rv_spread: Pre-calculated IV-RV spread
        """
        feature_dfs = [
            self._price_features(spx),
            self._vix_features(vix),
            self._volatility_regime(vix),
            self._seasonality(spx.index)
        ]
        
        # Optional core features
        if fred is not None and not fred.empty:
            feature_dfs.append(self._fred_features(fred))
        
        if macro is not None and not macro.empty:
            feature_dfs.append(self._macro_features(macro))
        
        if iv_rv_spread is not None:
            feature_dfs.append(self._iv_rv_features(iv_rv_spread))

        features = pd.concat(feature_dfs, axis=1)
        return features.dropna()
    
    def _price_features(self, spx: pd.Series) -> pd.DataFrame:
        """SPX price momentum and trend."""
        features = pd.DataFrame(index=spx.index)
        
        # Returns
        for window in [5, 10, 21, 63]:
            features[f'spx_ret_{window}'] = spx.pct_change(window) * 100
        
        # Moving averages
        for window in [20, 50, 200]:
            ma = spx.rolling(window).mean()
            features[f'spx_vs_ma{window}'] = ((spx - ma) / ma) * 100
        
        # Realized volatility
        returns = spx.pct_change()
        for window in [10, 21, 63]:
            features[f'spx_realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252) * 100
        
        return features
    
    def _vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """VIX level, changes, and mean reversion."""
        features = pd.DataFrame(index=vix.index)
        
        features['vix'] = vix
        
        # VIX percentile (where is VIX vs 1Y history)
        features['vix_percentile'] = vix.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
        
        # Changes
        for window in [1, 5, 21]:
            features[f'vix_change_{window}'] = vix.diff(window)
        
        # Mean reversion signal
        vix_ma = vix.rolling(63).mean()
        features['vix_vs_ma63'] = vix - vix_ma
        
        # Z-score
        features['vix_zscore'] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        
        return features
    
    def _volatility_regime(self, vix: pd.Series) -> pd.DataFrame:
        """VIX regime classification."""
        features = pd.DataFrame(index=vix.index)
        
        # Regime: 0=Low, 1=Normal, 2=Elevated, 3=Crisis
        features['vix_regime'] = pd.cut(
            vix, 
            bins=[0, 15, 20, 30, 100], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Days in current regime
        regime_change = features['vix_regime'] != features['vix_regime'].shift(1)
        regime_id = regime_change.cumsum()
        features['days_in_regime'] = regime_id.groupby(regime_id).cumcount() + 1
        
        return features
    
    def _fred_features(self, fred: pd.DataFrame) -> pd.DataFrame:
        """FRED economic indicators - level and changes."""
        features = pd.DataFrame(index=fred.index)
        
        for col in fred.columns:
            # Level
            features[f'{col}_level'] = fred[col]
            
            # Changes
            for window in FRED_WINDOWS:
                features[f'{col}_change_{window}'] = fred[col].diff(window)
        
        return features
    
    def _macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Macro momentum."""
        features = pd.DataFrame(index=macro.index)
        
        for col in macro.columns:
            if col == '5Y Treasury':
                continue
            for window in [10, 21]:
                features[f'{col}_mom_{window}'] = macro[col].pct_change(window, fill_method=None) * 100
        
        # Yield curve slope (if available)
        if '10Y Treasury' in macro.columns and '5Y Treasury' in macro.columns:
            slope = macro['10Y Treasury'] - macro['5Y Treasury']
            features['yield_slope'] = slope
            features['yield_slope_change_21'] = slope.diff(21)
        
        return features
    
    def _iv_rv_features(self, iv_rv_spread: pd.Series) -> pd.DataFrame:
        """IV vs RV spread - premium selling signal."""
        features = pd.DataFrame(index=iv_rv_spread.index)
        
        features['iv_rv_spread'] = iv_rv_spread
        
        # Spread momentum
        for window in [5, 21]:
            features[f'iv_rv_momentum_{window}'] = iv_rv_spread.diff(window)
        
        # Premium rich/cheap signal
        spread_ma = iv_rv_spread.rolling(63).mean()
        features['iv_rv_vs_avg'] = iv_rv_spread - spread_ma
        
        return features
    
    def _seasonality(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Calendar effects."""
        features = pd.DataFrame(index=index)
        
        features['month'] = index.month
        features['quarter'] = index.quarter
        features['day_of_week'] = index.dayofweek
        
        return features
    
    def scale(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize features."""
        scaled = self.scaler.fit_transform(features)
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted scaler."""
        scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)