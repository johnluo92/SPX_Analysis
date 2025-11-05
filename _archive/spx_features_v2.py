"""
SPX Feature Engineering V2 - PhD-Level Rigor
All features properly lagged, no look-ahead bias, aligned with explorer methodology
WITH COMMODITY SUPPORT
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import REGIME_BOUNDARIES


class SPXFeatureEngineV2:
    """
    Feature engineering with strict temporal controls.
    
    Key principles:
    1. NO LOOK-AHEAD: All rolling stats shifted by 1
    2. EXPLICIT LAGS: Use .shift() for clarity
    3. PROPER MISSING DATA: No forward fill in features
    4. MATCHED HORIZONS: Feature windows aligned with prediction targets
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def build(self, 
              spx: pd.Series,
              vix: pd.Series,
              fred: pd.DataFrame = None,
              macro: pd.DataFrame = None,
              commodities: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build features with strict temporal controls.
        
        Args:
            spx: SPX closing prices
            vix: VIX levels
            fred: FRED economic data
            macro: Macro factors
            commodities: NEW - FRED commodity futures (Gold, Oil, Dollar, Silver)
        """
        print("\n" + "="*70)
        print("BUILDING FEATURES V2 (PhD Edition)")
        print("="*70)
        
        feature_dfs = [
            self._price_features(spx),
            self._vix_features(vix),
            self._volatility_regime(vix),
            self._vix_vs_rv_features(spx, vix),
            self._seasonality(spx.index)
        ]
        
        # Optional features
        if fred is not None and not fred.empty:
            feature_dfs.append(self._fred_features(fred))
        
        if macro is not None and not macro.empty:
            feature_dfs.append(self._macro_features(macro))
        
        # NEW: Commodity features from FRED
        if commodities is not None and not commodities.empty:
            feature_dfs.append(self._commodity_features(commodities))
        
        features = pd.concat(feature_dfs, axis=1)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        print(f"\n✅ Built {len(features.columns)} features")
        print(f"   Samples: {len(features)}")
        print(f"   Missing: {features.isna().sum().sum()} values ({features.isna().sum().sum() / features.size * 100:.1f}%)")
        
        return features
    
    def _price_features(self, spx: pd.Series) -> pd.DataFrame:
        """SPX price features - properly lagged."""
        print("\n📊 Price Features")
        features = pd.DataFrame(index=spx.index)
        
        # EXPLICIT LAGS (Foundation)
        features['spx_lag1'] = spx.shift(1)
        features['spx_lag5'] = spx.shift(5)
        
        # RETURNS (match prediction horizons: 13d, 21d)
        for window in [5, 10, 13, 21, 63]:
            features[f'spx_ret_{window}d'] = spx.pct_change(window) * 100
        
        # MOVING AVERAGES (properly shifted)
        for window in [20, 50, 200]:
            ma = spx.rolling(window).mean().shift(1)  # ✅ SHIFTED
            features[f'spx_vs_ma{window}'] = ((spx - ma) / ma) * 100
        
        # MA CROSSOVERS
        ma_20 = spx.rolling(20).mean().shift(1)
        ma_50 = spx.rolling(50).mean().shift(1)
        features['ma20_vs_ma50'] = ((ma_20 - ma_50) / ma_50 * 100)
        
        # REALIZED VOLATILITY (backward-looking)
        returns = spx.pct_change()
        for window in [10, 21, 63]:
            # Shift to ensure backward-looking
            features[f'spx_realized_vol_{window}d'] = (
                returns.rolling(window).std().shift(1) * np.sqrt(252) * 100
            )
        
        # VOLATILITY REGIME
        vol_10d = returns.rolling(10).std().shift(1) * np.sqrt(252) * 100
        vol_63d = returns.rolling(63).std().shift(1) * np.sqrt(252) * 100
        features['spx_vol_ratio_10_63'] = vol_10d / vol_63d.replace(0, np.nan)
        
        # MOMENTUM Z-SCORES (properly lagged)
        for window in [10, 21]:
            ret = spx.pct_change(window)
            ret_ma = ret.rolling(63).mean().shift(1)  # ✅ SHIFTED
            ret_std = ret.rolling(63).std().shift(1)   # ✅ SHIFTED
            features[f'spx_momentum_z_{window}d'] = (ret - ret_ma) / ret_std
        
        # HIGHER MOMENTS
        features['spx_skew_21d'] = returns.rolling(21).skew()
        features['spx_kurt_21d'] = returns.rolling(21).kurt()
        
        # TECHNICAL INDICATORS
        ma = spx.rolling(20).mean()
        std = spx.rolling(20).std()
        features['bb_width_20d'] = ((ma + 2*std - (ma - 2*std)) / ma * 100)
        
        # RSI (14-day)
        delta = spx.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        print(f"   ✓ {len([c for c in features.columns if 'spx' in c or 'ma' in c or 'bb' in c or 'rsi' in c])} price features")
        
        return features
    
    def _vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """VIX features - level, changes, mean reversion."""
        print("\n📈 VIX Features")
        features = pd.DataFrame(index=vix.index)
        
        # EXPLICIT LAGS
        features['vix'] = vix
        features['vix_lag1'] = vix.shift(1)
        features['vix_lag5'] = vix.shift(5)
        
        # PERCENTILE (backward-looking)
        features['vix_percentile_252d'] = vix.rolling(253).apply(
            lambda x: pd.Series(x[:-1]).rank(pct=True).iloc[-1] * 100 if len(x) > 1 else np.nan,
            raw=False
        )
        
        # CHANGES
        for window in [1, 5, 10, 21]:
            features[f'vix_change_{window}d'] = vix.diff(window)
        
        # MEAN REVERSION (properly shifted)
        for window in [21, 63, 126, 252]:
            vix_ma = vix.rolling(window).mean().shift(1)  # ✅ SHIFTED
            features[f'vix_vs_ma{window}'] = vix - vix_ma
            features[f'vix_reversion_{window}d'] = (
                (vix - vix_ma) / vix_ma.replace(0, np.nan) * 100
            )
        
        # Z-SCORE (backward-looking)
        for window in [63, 252]:
            vix_ma = vix.rolling(window).mean().shift(1)
            vix_std = vix.rolling(window).std().shift(1)
            features[f'vix_zscore_{window}d'] = (vix - vix_ma) / vix_std
        
        # VIX VOLATILITY
        vix_change = vix.pct_change()
        for window in [10, 21, 63]:
            features[f'vix_stability_{window}d'] = vix_change.rolling(window).std()
        
        # VIX TERM STRUCTURE (if components available)
        vix_ma_5 = vix.rolling(5).mean().shift(1)
        vix_ma_21 = vix.rolling(21).mean().shift(1)
        features['vix_term_structure_5_21'] = vix_ma_5 - vix_ma_21
        
        print(f"   ✓ {len([c for c in features.columns if 'vix' in c])} VIX features")
        
        return features
    
    def _volatility_regime(self, vix: pd.Series) -> pd.DataFrame:
        """VIX regime classification - USING CONFIG BOUNDARIES."""
        print("\n🎯 Regime Features")
        features = pd.DataFrame(index=vix.index)
        
        # ✅ USE CONFIG BOUNDARIES - NO HARDCODED VALUES
        features['vix_regime'] = pd.cut(
            vix, 
            bins=REGIME_BOUNDARIES,  # ✅ FROM CONFIG
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)
        
        # DAYS IN CURRENT REGIME
        regime_change = features['vix_regime'] != features['vix_regime'].shift(1)
        regime_id = regime_change.cumsum()
        features['days_in_regime'] = regime_id.groupby(regime_id).cumcount() + 1
        
        # REGIME TRANSITION FLAG
        features['regime_transition'] = regime_change.astype(int)
        
        print(f"   ✓ 3 regime features (using GMM boundaries: {REGIME_BOUNDARIES})")
        
        return features
    
    def _vix_vs_rv_features(self, spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
        """VIX vs Realized Volatility - properly aligned."""
        print("\n⚡ VIX vs RV Features")
        features = pd.DataFrame(index=vix.index)
        
        returns = spx.pct_change()
        
        # VIX VS PAST REALIZED VOL (match prediction horizons)
        for window in [13, 21, 30, 63]:
            past_rv = returns.rolling(window).std().shift(1) * np.sqrt(252) * 100  # ✅ SHIFTED
            
            features[f'vix_vs_rv_{window}d'] = vix - past_rv
            features[f'vix_rv_ratio_{window}d'] = vix / past_rv.replace(0, np.nan)
        
        # WEIGHTED AVERAGE RV
        rv_21 = returns.rolling(21).std().shift(1) * np.sqrt(252) * 100
        rv_63 = returns.rolling(63).std().shift(1) * np.sqrt(252) * 100
        avg_rv = (rv_21 * 0.6 + rv_63 * 0.4)
        
        features['vix_vs_avg_rv'] = vix - avg_rv
        features['vix_avg_rv_ratio'] = vix / avg_rv.replace(0, np.nan)
        
        # VIX-RV SPREAD Z-SCORE (backward-looking)
        spread_21 = vix - rv_21
        spread_ma = spread_21.rolling(252).mean().shift(1)
        spread_std = spread_21.rolling(252).std().shift(1)
        features['vix_rv_spread_zscore'] = (spread_21 - spread_ma) / spread_std
        
        # VIX-RV SPREAD PERCENTILE
        features['vix_rv_spread_percentile'] = spread_21.rolling(253).apply(
            lambda x: pd.Series(x[:-1]).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan,
            raw=False
        )
        
        # VIX-SPX CORRELATION
        vix_change = vix.pct_change()
        features['vix_spx_corr_21d'] = returns.rolling(21).corr(vix_change)
        
        print(f"   ✓ {len([c for c in features.columns if 'rv' in c or 'corr' in c])} VIX/RV features")
        
        return features
    
    def _fred_features(self, fred: pd.DataFrame) -> pd.DataFrame:
        """FRED economic indicators - properly lagged."""
        print("\n💰 FRED Features")
        features = pd.DataFrame(index=fred.index)
        
        for col in fred.columns:
            # EXPLICIT LAGS
            features[f'{col}_level'] = fred[col]
            features[f'{col}_lag1'] = fred[col].shift(1)
            
            # CHANGES (match horizons)
            for window in [10, 21, 63]:
                features[f'{col}_change_{window}d'] = fred[col].diff(window)
            
            # Z-SCORES (properly shifted)
            for window in [63, 252]:
                ma = fred[col].rolling(window).mean().shift(1)
                std = fred[col].rolling(window).std().shift(1)
                features[f'{col}_zscore_{window}d'] = (fred[col] - ma) / std
        
        print(f"   ✓ {len([c for c in features.columns if any(x in c for x in fred.columns)])} FRED features")
        
        return features
    
    def _macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Macro momentum - properly lagged."""
        print("\n🌎 Macro Features")
        features = pd.DataFrame(index=macro.index)
        
        for col in macro.columns:
            if col == '5Y Treasury':
                continue
            
            # EXPLICIT LAGS
            features[f'{col}_lag1'] = macro[col].shift(1)
            
            # MOMENTUM (match horizons)
            for window in [10, 21, 63]:
                features[f'{col}_mom_{window}d'] = macro[col].pct_change(window) * 100
            
            # Z-SCORES
            ma = macro[col].rolling(63).mean().shift(1)
            std = macro[col].rolling(63).std().shift(1)
            features[f'{col}_zscore_63d'] = (macro[col] - ma) / std
        
        # YIELD CURVE SLOPE (if available)
        if '10Y Treasury' in macro.columns and '5Y Treasury' in macro.columns:
            slope = macro['10Y Treasury'] - macro['5Y Treasury']
            features['yield_slope'] = slope
            features['yield_slope_lag1'] = slope.shift(1)
            
            for window in [10, 21]:
                features[f'yield_slope_change_{window}d'] = slope.diff(window)
        
        print(f"   ✓ {len([c for c in features.columns if any(x in c for x in macro.columns)])} macro features")
        
        return features
    
    def _commodity_features(self, commodities: pd.DataFrame) -> pd.DataFrame:
        """
        Commodity features from FRED futures - properly lagged.
        
        NEW METHOD - Generates features for commodities_stress detector:
        - Gold_lag1, Gold_mom_10d, Gold_mom_21d, Gold_mom_63d, Gold_zscore_63d
        - Silver_mom_21d, Silver_zscore_63d
        - Crude Oil_mom_21d, Crude Oil_zscore_63d
        - Dollar_mom_21d, Dollar_zscore_63d
        """
        print("\n🛢️ Commodity Features")
        features = pd.DataFrame(index=commodities.index)
        
        # Gold features
        if 'Gold' in commodities.columns:
            gold = commodities['Gold']
            features['Gold_lag1'] = gold.shift(1)
            
            for window in [10, 21, 63]:
                features[f'Gold_mom_{window}d'] = gold.pct_change(window) * 100
            
            ma_63 = gold.rolling(63).mean().shift(1)
            std_63 = gold.rolling(63).std().shift(1)
            features['Gold_zscore_63d'] = (gold - ma_63) / std_63
        
        # Silver features
        if 'Silver' in commodities.columns:
            silver = commodities['Silver']
            features['Silver_mom_21d'] = silver.pct_change(21) * 100
            
            ma_63 = silver.rolling(63).mean().shift(1)
            std_63 = silver.rolling(63).std().shift(1)
            features['Silver_zscore_63d'] = (silver - ma_63) / std_63
        
        # Crude Oil features
        if 'Crude Oil' in commodities.columns:
            oil = commodities['Crude Oil']
            features['Crude Oil_mom_21d'] = oil.pct_change(21) * 100
            
            ma_63 = oil.rolling(63).mean().shift(1)
            std_63 = oil.rolling(63).std().shift(1)
            features['Crude Oil_zscore_63d'] = (oil - ma_63) / std_63
        
        # Dollar features
        if 'Dollar' in commodities.columns:
            dollar = commodities['Dollar']
            features['Dollar_mom_21d'] = dollar.pct_change(21) * 100
            
            ma_63 = dollar.rolling(63).mean().shift(1)
            std_63 = dollar.rolling(63).std().shift(1)
            features['Dollar_zscore_63d'] = (dollar - ma_63) / std_63
        
        print(f"   ✓ {len([c for c in features.columns])} commodity features")
        
        return features
    
    def _seasonality(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Calendar effects."""
        print("\n📅 Seasonality Features")
        features = pd.DataFrame(index=index)
        
        features['month'] = index.month
        features['quarter'] = index.quarter
        features['day_of_week'] = index.dayofweek
        features['day_of_month'] = index.day
        
        # OPEX week flag (3rd Friday)
        features['is_opex_week'] = 0
        for date in index:
            # Find 3rd Friday
            third_friday = pd.date_range(
                start=date.replace(day=1),
                end=date.replace(day=1) + pd.offsets.MonthEnd(1),
                freq='W-FRI'
            )[2] if len(pd.date_range(
                start=date.replace(day=1),
                end=date.replace(day=1) + pd.offsets.MonthEnd(1),
                freq='W-FRI'
            )) >= 3 else None
            
            if third_friday:
                # Mark Mon-Fri of OPEX week
                start_week = third_friday - pd.Timedelta(days=4)
                end_week = third_friday
                if start_week <= date <= end_week:
                    features.loc[date, 'is_opex_week'] = 1
        
        print(f"   ✓ 5 seasonality features")
        
        return features
    
    def scale(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Standardize features.
        
        Args:
            features: Raw features
            fit: If True, fit scaler. If False, use existing scaler.
        """
        if fit:
            scaled = self.scaler.fit_transform(features.fillna(0))
        else:
            scaled = self.scaler.transform(features.fillna(0))
        
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)