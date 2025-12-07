"""
Commodity & Currency Volatility Feature Engineer
Standalone system for DXY (Dollar Index), OVX (Oil Vol), GVZ (Gold Vol), and related futures
Pattern matches: vx_futures_engineer.py and regime_classifier.py
"""
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
warnings.filterwarnings("ignore")


class CommodityVolatilityEngineer:
    """
    Generates features from commodity futures, currency indices, and their volatility indices.
    Covers: DXY, OVX, GVZ, RVX, VXEEM, Natural Gas, Copper
    """

    def __init__(self, data_fetcher):
        """
        Args:
            data_fetcher: UnifiedDataFetcher instance with fetch_yahoo and fetch_cboe_series methods
        """
        self.fetcher = data_fetcher

        # Define what to fetch and from where
        self.yahoo_symbols = {
            'DX-Y.NYB': 'Dollar_Index',      # US Dollar Index (DXY)
            'GC=F': 'Gold_Futures',          # Gold futures
            'CL=F': 'Crude_Oil',             # Already fetched but useful here
            'NG=F': 'Natural_Gas',           # Natural gas futures
            'HG=F': 'Copper',                # Copper futures (Dr. Copper - economic indicator)
        }

        self.cboe_vol_indices = {
            '^OVX': 'OVX',                   # Crude Oil ETF Volatility Index
            '^GVZ': 'GVZ',                   # Gold ETF Volatility Index
        }

    def build_all_commodity_vol_features(
        self,
        start_date: str,
        end_date: str,
        target_index: pd.DatetimeIndex,
        vix_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Main entry point - builds complete feature set

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD)
            target_index: DatetimeIndex to align features to
            vix_series: Optional VIX series for cross-asset comparisons

        Returns:
            DataFrame with all commodity/currency volatility features
        """
        # Fetch all data sources (silently)
        yahoo_data = self._fetch_yahoo_data(start_date, end_date, target_index)
        cboe_vol_data = self._fetch_cboe_volatility_indices(start_date, end_date, target_index)

        # Initialize feature container
        features = pd.DataFrame(index=target_index)

        # Build feature groups
        dollar_features = self._build_dollar_index_features(yahoo_data, target_index)
        features = pd.concat([features, dollar_features], axis=1)

        commodity_vol_features = self._build_commodity_vol_features(cboe_vol_data, vix_series, target_index)
        features = pd.concat([features, commodity_vol_features], axis=1)

        commodity_features = self._build_commodity_features(yahoo_data, target_index)
        features = pd.concat([features, commodity_features], axis=1)

        cross_asset_features = self._build_cross_asset_features(
            yahoo_data, cboe_vol_data, vix_series, target_index
        )
        features = pd.concat([features, cross_asset_features], axis=1)

        regime_features = self._build_regime_features(features, yahoo_data, cboe_vol_data)
        features = pd.concat([features, regime_features], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.loc[:, ~features.columns.duplicated()]

        # Convert all dtypes to numeric (fix categorical issue)
        for col in features.columns:
            if features[col].dtype == 'object' or features[col].dtype.name == 'category':
                features[col] = pd.to_numeric(features[col], errors='coerce')
            features[col] = features[col].astype(np.float64)

        print(f"   → Commodity/currency features: {len(features.columns)} features | {features.notna().mean().mean()*100:.1f}% coverage")

        return features

    def _fetch_yahoo_data(
        self,
        start_date: str,
        end_date: str,
        target_index: pd.DatetimeIndex
    ) -> Dict[str, pd.Series]:
        """Fetch futures & index data from Yahoo Finance"""
        data = {}
        ff_limit = 5  # Forward fill limit for daily data

        for symbol, name in self.yahoo_symbols.items():
            try:
                df = self.fetcher.fetch_yahoo(symbol, start_date, end_date)
                if df is not None and 'Close' in df.columns:
                    data[name] = df['Close'].reindex(target_index, method='ffill', limit=ff_limit)
            except:
                pass

        return data

    def _fetch_cboe_volatility_indices(
        self,
        start_date: str,
        end_date: str,
        target_index: pd.DatetimeIndex
    ) -> Dict[str, pd.Series]:
        """Fetch CBOE volatility indices (OVX, GVZ)"""
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)

        data = {}
        ff_limit = 5

        for symbol, name in self.cboe_vol_indices.items():
            try:
                df = self.fetcher.fetch_yahoo(symbol, start_date, end_date)
                if df is not None and 'Close' in df.columns:
                    data[name] = df['Close'].reindex(target_index, method='ffill', limit=ff_limit)
            except:
                pass

        return data

    def _build_dollar_index_features(
        self,
        yahoo_data: Dict[str, pd.Series],
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build comprehensive Dollar Index (DXY) features"""
        features = pd.DataFrame(index=target_index)

        if 'Dollar_Index' not in yahoo_data:
            return features

        dxy = yahoo_data['Dollar_Index']
        features['dxy'] = dxy

        # Returns across multiple horizons
        for window in [1, 5, 10, 21, 63]:
            features[f'dxy_ret_{window}d'] = dxy.pct_change(window) * 100

        # Realized volatility
        dxy_returns = dxy.pct_change()
        for window in [10, 21, 63]:
            features[f'dxy_vol_{window}d'] = dxy_returns.rolling(window).std() * np.sqrt(252) * 100

        # Momentum & velocity
        for window in [5, 10, 21]:
            features[f'dxy_velocity_{window}d'] = dxy.diff(window)
            mom = dxy.diff(window)
            features[f'dxy_momentum_z_{window}d'] = self._zscore(mom, 63)

        # Mean reversion & relative positioning
        for window in [21, 63, 252]:
            ma = dxy.rolling(window).mean()
            features[f'dxy_vs_ma{window}'] = ((dxy - ma) / ma.replace(0, np.nan)) * 100
            features[f'dxy_zscore_{window}d'] = self._zscore(dxy, window)
            features[f'dxy_percentile_{window}d'] = self._percentile(dxy, window)

        # Acceleration
        features['dxy_accel_5d'] = dxy.diff(5).diff(5)

        # Bollinger bands
        bb_window = 20
        bb_ma = dxy.rolling(bb_window).mean()
        bb_std = dxy.rolling(bb_window).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        features['dxy_bb_position_20d'] = (
            (dxy - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        ).clip(0, 1)
        features['dxy_bb_width_20d'] = ((bb_upper - bb_lower) / bb_ma.replace(0, np.nan)) * 100

        # Regime classification
        features['dxy_extreme_high'] = (dxy > dxy.rolling(252).quantile(0.90)).astype(int)
        features['dxy_extreme_low'] = (dxy < dxy.rolling(252).quantile(0.10)).astype(int)

        return features

    def _build_commodity_vol_features(
        self,
        cboe_vol_data: Dict[str, pd.Series],
        vix: Optional[pd.Series],
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build features from CBOE commodity volatility indices"""
        features = pd.DataFrame(index=target_index)

        # OVX (Oil Volatility Index)
        if 'OVX' in cboe_vol_data:
            ovx = cboe_vol_data['OVX']
            features['ovx'] = ovx

            # Returns & momentum
            for window in [5, 10, 21]:
                features[f'ovx_ret_{window}d'] = ovx.pct_change(window) * 100
                features[f'ovx_velocity_{window}d'] = ovx.diff(window)

            # Volatility of volatility
            ovx_returns = ovx.pct_change()
            for window in [10, 21]:
                features[f'ovx_vol_{window}d'] = ovx_returns.rolling(window).std() * np.sqrt(252) * 100

            # Z-scores & percentiles
            for window in [21, 63, 252]:
                features[f'ovx_zscore_{window}d'] = self._zscore(ovx, window)
                features[f'ovx_percentile_{window}d'] = self._percentile(ovx, window)

            # Relative to moving averages
            for window in [21, 63]:
                ma = ovx.rolling(window).mean()
                features[f'ovx_vs_ma{window}'] = ((ovx - ma) / ma.replace(0, np.nan)) * 100

            # OVX vs VIX comparison
            if vix is not None:
                vix_aligned = vix.reindex(target_index, method='ffill', limit=5)
                features['ovx_vix_ratio'] = ovx / vix_aligned.replace(0, np.nan)
                features['ovx_vix_spread'] = ovx - vix_aligned
                features['ovx_vix_spread_zscore'] = self._zscore(features['ovx_vix_spread'], 63)

            # Regime
            features['ovx_regime'] = self._regime_classify(
                ovx, bins=[0, 20, 30, 45, 150], labels=[0, 1, 2, 3]
            )

        # GVZ (Gold Volatility Index)
        if 'GVZ' in cboe_vol_data:
            gvz = cboe_vol_data['GVZ']
            features['gvz'] = gvz

            # Returns & momentum
            for window in [5, 10, 21]:
                features[f'gvz_ret_{window}d'] = gvz.pct_change(window) * 100
                features[f'gvz_velocity_{window}d'] = gvz.diff(window)

            # Z-scores & percentiles
            for window in [21, 63]:
                features[f'gvz_zscore_{window}d'] = self._zscore(gvz, window)
                features[f'gvz_percentile_{window}d'] = self._percentile(gvz, window)

            # GVZ vs VIX (flight to safety indicator)
            if vix is not None:
                vix_aligned = vix.reindex(target_index, method='ffill', limit=5)
                features['gvz_vix_ratio'] = gvz / vix_aligned.replace(0, np.nan)
                features['gvz_vix_spread'] = gvz - vix_aligned

        return features

    def _build_commodity_features(
        self,
        yahoo_data: Dict[str, pd.Series],
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build features from commodity futures (Gold, Nat Gas, Copper)"""
        features = pd.DataFrame(index=target_index)

        # Gold features (already in main system but extend here)
        if 'Gold_Futures' in yahoo_data:
            gold = yahoo_data['Gold_Futures']
            features['gold'] = gold

            for window in [5, 10, 21, 63]:
                features[f'gold_ret_{window}d'] = gold.pct_change(window) * 100

            gold_returns = gold.pct_change()
            for window in [10, 21, 63]:
                features[f'gold_vol_{window}d'] = gold_returns.rolling(window).std() * np.sqrt(252) * 100

            for window in [21, 63, 252]:
                features[f'gold_zscore_{window}d'] = self._zscore(gold, window)

            # Gold momentum regime
            features['gold_momentum_21d'] = gold.diff(21)
            features['gold_regime'] = (features['gold_momentum_21d'] > 0).astype(int)

        # Natural Gas features
        if 'Natural_Gas' in yahoo_data:
            ng = yahoo_data['Natural_Gas']
            features['natgas'] = ng

            for window in [5, 10, 21]:
                features[f'natgas_ret_{window}d'] = ng.pct_change(window) * 100

            ng_returns = ng.pct_change()
            features['natgas_vol_21d'] = ng_returns.rolling(21).std() * np.sqrt(252) * 100

            for window in [21, 63]:
                features[f'natgas_zscore_{window}d'] = self._zscore(ng, window)

        # Copper features (economic indicator - "Dr. Copper")
        if 'Copper' in yahoo_data:
            copper = yahoo_data['Copper']
            features['copper'] = copper

            for window in [5, 10, 21, 63]:
                features[f'copper_ret_{window}d'] = copper.pct_change(window) * 100

            for window in [21, 63, 252]:
                features[f'copper_zscore_{window}d'] = self._zscore(copper, window)

            # Copper as economic indicator
            features['copper_momentum_63d'] = copper.diff(63)
            features['copper_economic_signal'] = (features['copper_momentum_63d'] > 0).astype(int)

        return features

    def _build_cross_asset_features(
        self,
        yahoo_data: Dict[str, pd.Series],
        cboe_vol_data: Dict[str, pd.Series],
        vix: Optional[pd.Series],
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build cross-asset relationship features"""
        features = pd.DataFrame(index=target_index)

        # Dollar vs Gold (classic inverse relationship)
        if 'Dollar_Index' in yahoo_data and 'Gold_Futures' in yahoo_data:
            dxy = yahoo_data['Dollar_Index']
            gold = yahoo_data['Gold_Futures']

            # Correlation over different windows
            for window in [21, 63, 126]:
                features[f'dxy_gold_corr_{window}d'] = (
                    dxy.pct_change().rolling(window).corr(gold.pct_change())
                )

            # Relative strength
            dxy_rank = dxy.pct_change(21).rolling(63).rank(pct=True)
            gold_rank = gold.pct_change(21).rolling(63).rank(pct=True)
            features['dxy_gold_divergence'] = (dxy_rank - gold_rank).abs()

            # When both rally = crisis mode
            dxy_strong = dxy.pct_change(21) > 0
            gold_strong = gold.pct_change(21) > 0
            features['dxy_gold_both_rally'] = (dxy_strong & gold_strong).astype(int)

        # OVX vs Crude Oil (vol vs underlying)
        if 'OVX' in cboe_vol_data and 'Crude_Oil' in yahoo_data:
            ovx = cboe_vol_data['OVX']
            oil = yahoo_data['Crude_Oil']

            # OVX relative to oil realized vol
            oil_rv_21d = oil.pct_change().rolling(21).std() * np.sqrt(252) * 100
            features['ovx_oil_rv_spread'] = ovx - oil_rv_21d
            features['ovx_oil_rv_ratio'] = ovx / oil_rv_21d.replace(0, np.nan)

        # GVZ vs Gold (vol vs underlying)
        if 'GVZ' in cboe_vol_data and 'Gold_Futures' in yahoo_data:
            gvz = cboe_vol_data['GVZ']
            gold = yahoo_data['Gold_Futures']

            gold_rv_21d = gold.pct_change().rolling(21).std() * np.sqrt(252) * 100
            features['gvz_gold_rv_spread'] = gvz - gold_rv_21d
            features['gvz_gold_rv_ratio'] = gvz / gold_rv_21d.replace(0, np.nan)

        # Copper vs Dollar (economic growth indicator)
        if 'Copper' in yahoo_data and 'Dollar_Index' in yahoo_data:
            copper = yahoo_data['Copper']
            dxy = yahoo_data['Dollar_Index']

            # When copper falls + dollar rises = growth concerns
            copper_weak = copper.pct_change(21) < 0
            dxy_strong = dxy.pct_change(21) > 0
            features['growth_concern_signal'] = (copper_weak & dxy_strong).astype(int)

        # Multi-asset vol stress composite
        vol_components = []
        if vix is not None:
            vix_z = self._zscore(vix, 63)
            vol_components.append(vix_z.reindex(target_index))
        if 'OVX' in cboe_vol_data:
            vol_components.append(self._zscore(cboe_vol_data['OVX'], 63))
        if 'GVZ' in cboe_vol_data:
            vol_components.append(self._zscore(cboe_vol_data['GVZ'], 63))

        if len(vol_components) >= 2:
            vol_df = pd.DataFrame(vol_components).T
            features['multi_asset_vol_stress'] = vol_df.mean(axis=1)
            features['vol_stress_dispersion'] = vol_df.std(axis=1)

        return features

    def _build_regime_features(
        self,
        features: pd.DataFrame,
        yahoo_data: Dict[str, pd.Series],
        cboe_vol_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Build regime classification features"""
        regime_features = pd.DataFrame(index=features.index)

        # Dollar regime (strong/weak) - NUMERIC ONLY
        if 'dxy_percentile_252d' in features.columns:
            pct = features['dxy_percentile_252d']
            regime_features['dxy_regime_numeric'] = pd.cut(
                pct,
                bins=[0, 25, 75, 100],
                labels=[0, 1, 2]
            ).astype(float)

        # Commodity vol regime (low/medium/high/extreme) - NUMERIC ONLY
        if 'ovx' in features.columns:
            ovx = features['ovx']
            regime_features['commodity_vol_regime'] = self._regime_classify(
                ovx, bins=[0, 20, 30, 45, 150], labels=[0, 1, 2, 3]
            )

        # Flight to safety indicator (gold + dollar both rising)
        if 'dxy_gold_both_rally' in features.columns:
            regime_features['flight_to_safety_active'] = features['dxy_gold_both_rally']

        # Risk-on / risk-off composite
        risk_on_signals = []
        if 'dxy_ret_21d' in features.columns:
            risk_on_signals.append(features['dxy_ret_21d'] < 0)  # Dollar weakening
        if 'copper_ret_21d' in features.columns:
            risk_on_signals.append(features['copper_ret_21d'] > 0)  # Copper rising
        if 'gold_ret_21d' in features.columns:
            risk_on_signals.append(features['gold_ret_21d'] < 0)  # Gold falling

        if len(risk_on_signals) >= 2:
            risk_on_df = pd.DataFrame(risk_on_signals).T
            regime_features['risk_on_score'] = risk_on_df.sum(axis=1) / len(risk_on_signals)

        return regime_features

    # Utility functions
    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score"""
        mean = series.rolling(window, min_periods=window//2).mean()
        std = series.rolling(window, min_periods=window//2).std()
        return (series - mean) / std.replace(0, np.nan)

    def _percentile(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling percentile rank"""
        return series.rolling(window, min_periods=window//2).rank(pct=True) * 100

    def _regime_classify(
        self,
        series: pd.Series,
        bins: list,
        labels: list
    ) -> pd.Series:
        """Classify series into regimes"""
        return pd.cut(series, bins=bins, labels=labels).astype(float)


# Convenience function matching your existing pattern
def build_commodity_vol_features(
    data_fetcher,
    start_date: str,
    end_date: str,
    target_index: pd.DatetimeIndex,
    vix_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Convenience function to build commodity volatility features

    Usage:
        from core.commodity_volatility_engineer import build_commodity_vol_features

        features = build_commodity_vol_features(
            data_fetcher=fetcher,
            start_date='2020-01-01',
            end_date='2024-12-31',
            target_index=spx.index,
            vix_series=vix
        )
    """
    engineer = CommodityVolatilityEngineer(data_fetcher)
    return engineer.build_all_commodity_vol_features(
        start_date, end_date, target_index, vix_series
    )
