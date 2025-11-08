"""Enhanced Feature Engine V4 - Meta Features, Futures, and Maximum Feature Richness"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

from config_v3 import REGIME_BOUNDARIES, TRAINING_YEARS


class MetaFeatureEngine:
    """
    Advanced meta-feature extraction from base features.
    Focus: Regime detection, cross-asset dynamics, rate-of-change, percentile rankings.
    """
    
    @staticmethod
    def extract_regime_indicators(df: pd.DataFrame, vix: pd.Series, spx: pd.Series) -> pd.DataFrame:
        """
        Extract comprehensive regime indicators across multiple dimensions.
        Returns meta-features that classify current market state.
        """
        meta = pd.DataFrame(index=df.index)
        
        # === VIX Regime Classification ===
        if 'vix' in df.columns:
            v = df['vix']
            
            # Multi-timeframe regime
            meta['vix_regime_micro'] = pd.cut(v, bins=[0, 12, 16, 20, 100], labels=[0,1,2,3]).astype(float)
            meta['vix_regime_macro'] = pd.cut(v, bins=REGIME_BOUNDARIES, labels=[0,1,2,3]).astype(float)
            
            # Regime stability (how long in current regime)
            regime_changes = (meta['vix_regime_macro'] != meta['vix_regime_macro'].shift(1)).cumsum()
            meta['regime_stability'] = regime_changes.groupby(regime_changes).cumcount() + 1
            
            # Regime transition probability (based on velocity)
            if 'vix_velocity_5d' in df.columns:
                meta['regime_transition_risk'] = (
                    df['vix_velocity_5d'].abs() / v * 100
                ).clip(0, 100)
        
        # === Volatility Regime ===
        if all(col in df.columns for col in ['spx_realized_vol_21d', 'vix']):
            rv = df['spx_realized_vol_21d']
            v = df['vix']
            
            # Vol regime classification
            meta['vol_regime'] = pd.cut(rv, bins=[0, 10, 15, 25, 100], labels=[0,1,2,3]).astype(float)
            
            # Risk premium regime
            risk_prem = v - rv
            meta['risk_premium_regime'] = pd.cut(
                risk_prem, bins=[-100, 0, 5, 10, 100], labels=[0,1,2,3]
            ).astype(float)
            
            # Volatility term structure regime
            if 'vix_term_structure' in df.columns:
                ts = df['vix_term_structure']
                meta['vol_term_regime'] = pd.cut(
                    ts, bins=[-100, -2, 0, 2, 100], labels=[0,1,2,3]
                ).astype(float)
        
        # === Trend Regime ===
        if 'spx_vs_ma200' in df.columns:
            trend = df['spx_vs_ma200']
            meta['trend_regime'] = pd.cut(
                trend, bins=[-100, -5, 0, 5, 100], labels=[0,1,2,3]
            ).astype(float)
            
            # Trend strength
            if 'spx_vs_ma50' in df.columns:
                meta['trend_strength'] = (
                    df['spx_vs_ma200'].abs() + df['spx_vs_ma50'].abs()
                ) / 2
        
        # === Liquidity/Stress Regime ===
        stress_components = []
        
        if 'SKEW' in df.columns:
            skew_stress = ((df['SKEW'] - 130) / 30).clip(0, 1)
            stress_components.append(skew_stress)
        
        if 'vix' in df.columns:
            vix_stress = ((df['vix'] - 15) / 25).clip(0, 1)
            stress_components.append(vix_stress)
        
        if 'spx_realized_vol_21d' in df.columns:
            rv_stress = ((df['spx_realized_vol_21d'] - 15) / 20).clip(0, 1)
            stress_components.append(rv_stress)
        
        if stress_components:
            meta['liquidity_stress_composite'] = pd.DataFrame(stress_components).T.mean(axis=1)
            meta['liquidity_regime'] = pd.cut(
                meta['liquidity_stress_composite'],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=[0,1,2,3]
            ).astype(float)
        
        # === Correlation Regime ===
        if 'spx_vix_corr_21d' in df.columns:
            corr = df['spx_vix_corr_21d']
            # Normal: -0.8 to -0.5, Stress: < -0.8, Anomalous: > -0.5
            meta['correlation_regime'] = pd.cut(
                corr, bins=[-1, -0.8, -0.5, 0, 1], labels=[0,1,2,3]
            ).astype(float)
        
        # === Composite Market Regime ===
        # Combine multiple regime indicators into single composite
        regime_cols = [c for c in meta.columns if 'regime' in c and c != 'regime_stability']
        if regime_cols:
            meta['composite_market_regime'] = meta[regime_cols].mean(axis=1)
            meta['regime_consensus'] = meta[regime_cols].std(axis=1)  # Low = consensus, High = divergence
        
        return meta
    
    @staticmethod
    def extract_cross_asset_relationships(df: pd.DataFrame, macro: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract cross-asset correlation and divergence features.
        Captures when assets move out of sync (predictive of regime change).
        """
        meta = pd.DataFrame(index=df.index)
        
        # === Equity-Vol Relationship ===
        if all(col in df.columns for col in ['spx_ret_21d', 'vix_velocity_21d']):
            spx_ret = df['spx_ret_21d']
            vix_chg = df['vix_velocity_21d']
            
            # Expected: negative correlation
            meta['equity_vol_divergence'] = (
                (spx_ret.rank(pct=True) + vix_chg.rank(pct=True)) - 1
            ).abs()  # 0 = normal, 1 = max divergence
            
            # Rolling correlation breakdown
            if 'spx_vix_corr_21d' in df.columns:
                corr = df['spx_vix_corr_21d']
                corr_ma = corr.rolling(63).mean()
                meta['equity_vol_corr_breakdown'] = (corr - corr_ma).abs()
        
        # === Vol of Vol ===
        if 'vix' in df.columns:
            vix_returns = df['vix'].pct_change()
            meta['vol_of_vol_10d'] = vix_returns.rolling(10).std() * np.sqrt(252) * 100
            meta['vol_of_vol_21d'] = vix_returns.rolling(21).std() * np.sqrt(252) * 100
            
            # VIX acceleration (second derivative)
            if 'vix_velocity_5d' in df.columns:
                meta['vix_acceleration'] = df['vix_velocity_5d'].diff(5)
        
        # === Risk Premium Dynamics ===
        if all(col in df.columns for col in ['vix', 'spx_realized_vol_21d']):
            risk_prem = df['vix'] - df['spx_realized_vol_21d']
            meta['risk_premium'] = risk_prem
            meta['risk_premium_ma21'] = risk_prem.rolling(21).mean()
            meta['risk_premium_velocity'] = risk_prem.diff(10)
            meta['risk_premium_zscore'] = (
                (risk_prem - risk_prem.rolling(63).mean()) / 
                risk_prem.rolling(63).std()
            )
        
        # === Macro Asset Integration ===
        if macro is not None:
            # Gold-SPX relationship (risk-on/risk-off)
            if 'Gold' in macro.columns and 'spx_ret_21d' in df.columns:
                gold_ret = macro['Gold'].pct_change(21) * 100
                meta['gold_spx_divergence'] = (
                    gold_ret.rank(pct=True) - df['spx_ret_21d'].rank(pct=True)
                ).abs()
            
            # Dollar-SPX relationship
            if 'Dollar_Index' in macro.columns and 'spx_ret_21d' in df.columns:
                dxy_ret = macro['Dollar_Index'].pct_change(21) * 100
                meta['dollar_spx_correlation'] = (
                    dxy_ret.rolling(63).corr(df['spx_ret_21d'])
                )
        
        return meta
    
    @staticmethod
    def extract_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract velocity, acceleration, and jerk (3rd derivative) features.
        Captures momentum shifts that precede regime changes.
        """
        meta = pd.DataFrame(index=df.index)
        
        # Define key series for ROC analysis
        roc_series = {
            'vix': df.get('vix'),
            'SKEW': df.get('SKEW'),
            'spx_realized_vol_21d': df.get('spx_realized_vol_21d'),
            'rsi_14': df.get('rsi_14'),
        }
        
        # Add put/call ratios
        for pc_col in ['PCC', 'PCCE', 'PCCI']:
            if pc_col in df.columns:
                roc_series[pc_col] = df[pc_col]
        
        for name, series in roc_series.items():
            if series is None:
                continue
            
            # Multi-timeframe velocity
            for window in [3, 5, 10, 21]:
                meta[f'{name}_velocity_{window}d'] = series.diff(window)
                meta[f'{name}_velocity_{window}d_pct'] = series.pct_change(window) * 100
            
            # Acceleration (2nd derivative)
            vel_5d = series.diff(5)
            meta[f'{name}_acceleration_5d'] = vel_5d.diff(5)
            
            # Jerk (3rd derivative) - rate of acceleration change
            accel_5d = vel_5d.diff(5)
            meta[f'{name}_jerk_5d'] = accel_5d.diff(5)
            
            # Momentum regime (accelerating vs decelerating)
            meta[f'{name}_momentum_regime'] = np.sign(meta[f'{name}_acceleration_5d'])
        
        # === Cross-series momentum divergence ===
        if all(col in df.columns for col in ['vix', 'SKEW']):
            vix_mom = df['vix'].diff(10)
            skew_mom = df['SKEW'].diff(10)
            meta['vix_skew_momentum_divergence'] = (
                vix_mom.rank(pct=True) - skew_mom.rank(pct=True)
            ).abs()
        
        return meta
    
    @staticmethod
    def extract_percentile_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract percentile rankings across multiple lookback windows.
        Answers: "Where are we in historical distribution?"
        """
        meta = pd.DataFrame(index=df.index)
        
        # Key series for percentile ranking
        ranking_series = {
            'vix': df.get('vix'),
            'SKEW': df.get('SKEW'),
            'spx_realized_vol_21d': df.get('spx_realized_vol_21d'),
            'rsi_14': df.get('rsi_14'),
        }
        
        # Add vol spread if available
        if all(col in df.columns for col in ['vix', 'spx_realized_vol_21d']):
            ranking_series['risk_premium'] = df['vix'] - df['spx_realized_vol_21d']
        
        for name, series in ranking_series.items():
            if series is None:
                continue
            
            # Multi-timeframe percentile rankings
            for window in [21, 63, 126, 252]:
                meta[f'{name}_percentile_{window}d'] = series.rolling(window + 1).apply(
                    lambda x: pd.Series(x[:-1]).rank(pct=True).iloc[-1] * 100 if len(x) > 1 else 50,
                    raw=False
                )
            
            # Percentile velocity (how fast moving through distribution)
            if f'{name}_percentile_63d' in meta.columns:
                meta[f'{name}_percentile_velocity'] = (
                    meta[f'{name}_percentile_63d'].diff(10)
                )
            
            # Extreme percentile flags
            for window in [63, 252]:
                pct_col = f'{name}_percentile_{window}d'
                if pct_col in meta.columns:
                    meta[f'{name}_extreme_high_{window}d'] = (meta[pct_col] > 90).astype(int)
                    meta[f'{name}_extreme_low_{window}d'] = (meta[pct_col] < 10).astype(int)
        
        return meta


class FuturesFeatureEngine:
    """
    Specialized feature extraction for futures contracts.
    Focus: VX futures, crude oil, dollar index futures.
    """
    
    @staticmethod
    def extract_vix_futures_features(vx_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Extract features from VIX futures (VX1, VX2, term structure).
        Captures forward expectations and risk premium dynamics.
        """
        features = pd.DataFrame()
        
        # Extract VX1-VX2 spread if available
        if 'VX1-VX2' in vx_data:
            spread = vx_data['VX1-VX2']
            features['vx_spread'] = spread
            features['vx_spread_ma10'] = spread.rolling(10).mean()
            features['vx_spread_ma21'] = spread.rolling(21).mean()
            features['vx_spread_velocity_5d'] = spread.diff(5)
            features['vx_spread_velocity_21d'] = spread.diff(21)
            
            # Spread regime
            features['vx_spread_regime'] = pd.cut(
                spread, bins=[-10, -1, 0, 1, 10], labels=[0,1,2,3]
            ).astype(float)
            
            # Z-score
            spread_ma = spread.rolling(63).mean()
            spread_std = spread.rolling(63).std()
            features['vx_spread_zscore_63d'] = (spread - spread_ma) / spread_std
            
            # Percentile
            features['vx_spread_percentile_63d'] = spread.rolling(64).apply(
                lambda x: pd.Series(x[:-1]).rank(pct=True).iloc[-1] * 100 if len(x) > 1 else 50,
                raw=False
            )
        
        # Extract VX ratio if available
        if 'VX2-VX1_RATIO' in vx_data:
            ratio = vx_data['VX2-VX1_RATIO']
            features['vx_ratio'] = ratio
            features['vx_ratio_ma21'] = ratio.rolling(21).mean()
            features['vx_ratio_velocity_10d'] = ratio.diff(10)
            
            # Ratio regime (contango vs backwardation)
            features['vx_term_structure_regime'] = pd.cut(
                ratio, bins=[-1, -0.05, 0, 0.05, 1], labels=[0,1,2,3]
            ).astype(float)
            
            # Extreme term structure flags
            features['vx_steep_contango'] = (ratio > 0.15).astype(int)
            features['vx_steep_backwardation'] = (ratio < -0.05).astype(int)
        
        # VX curve dynamics (if we have both spread and ratio)
        if 'VX1-VX2' in vx_data and 'VX2-VX1_RATIO' in vx_data:
            spread = vx_data['VX1-VX2']
            ratio = vx_data['VX2-VX1_RATIO']
            
            # Curve steepness acceleration
            features['vx_curve_acceleration'] = ratio.diff(5).diff(5)
            
            # Spread-ratio divergence (unusual term structure shape)
            spread_rank = spread.rolling(63).rank(pct=True)
            ratio_rank = ratio.rolling(63).rank(pct=True)
            features['vx_term_structure_divergence'] = (spread_rank - ratio_rank).abs()
        
        return features
    
    @staticmethod
    def extract_commodity_futures_features(futures_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Extract features from commodity futures (crude oil, natural gas).
        Captures energy market dynamics and macro risk.
        """
        features = pd.DataFrame()
        
        # Crude oil futures spread (if available)
        if 'CL1-CL2' in futures_data:
            cl_spread = futures_data['CL1-CL2']
            features['cl_spread'] = cl_spread
            features['cl_spread_ma10'] = cl_spread.rolling(10).mean()
            features['cl_spread_velocity_5d'] = cl_spread.diff(5)
            features['cl_spread_zscore_63d'] = (
                (cl_spread - cl_spread.rolling(63).mean()) / 
                cl_spread.rolling(63).std()
            )
            
            # Oil term structure regime
            features['oil_term_regime'] = pd.cut(
                cl_spread, bins=[-10, -1, 0, 2, 20], labels=[0,1,2,3]
            ).astype(float)
            
            # Extreme oil term structure
            features['oil_steep_backwardation'] = (cl_spread < -2).astype(int)
            features['oil_steep_contango'] = (cl_spread > 5).astype(int)
        
        # Individual commodity price if available
        for commodity in ['Crude_Oil', 'Natural_Gas']:
            if commodity in futures_data:
                price = futures_data[commodity]
                prefix = commodity.lower().replace('_', '_')
                
                # Price momentum
                for window in [10, 21, 63]:
                    features[f'{prefix}_ret_{window}d'] = price.pct_change(window) * 100
                
                # Volatility
                features[f'{prefix}_vol_21d'] = (
                    price.pct_change().rolling(21).std() * np.sqrt(252) * 100
                )
                
                # Z-score
                price_ma = price.rolling(63).mean()
                price_std = price.rolling(63).std()
                features[f'{prefix}_zscore_63d'] = (price - price_ma) / price_std
        
        return features
    
    @staticmethod
    def extract_dollar_futures_features(dollar_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Extract features from dollar index futures.
        Captures FX risk and global liquidity dynamics.
        """
        features = pd.DataFrame()
        
        # Dollar futures spread (if available)
        if 'DX1-DX2' in dollar_data:
            dx_spread = dollar_data['DX1-DX2']
            features['dx_spread'] = dx_spread
            features['dx_spread_ma10'] = dx_spread.rolling(10).mean()
            features['dx_spread_velocity_5d'] = dx_spread.diff(5)
            features['dx_spread_zscore_63d'] = (
                (dx_spread - dx_spread.rolling(63).mean()) / 
                dx_spread.rolling(63).std()
            )
        
        # Dollar index level (if available)
        if 'Dollar_Index' in dollar_data:
            dxy = dollar_data['Dollar_Index']
            
            # Multi-timeframe momentum
            for window in [10, 21, 63]:
                features[f'dxy_ret_{window}d'] = dxy.pct_change(window) * 100
            
            # DXY vs moving averages
            for window in [50, 200]:
                ma = dxy.rolling(window).mean()
                features[f'dxy_vs_ma{window}'] = ((dxy - ma) / ma) * 100
            
            # DXY volatility
            features['dxy_vol_21d'] = (
                dxy.pct_change().rolling(21).std() * np.sqrt(252) * 100
            )
            
            # DXY regime
            dxy_ma = dxy.rolling(200).mean()
            features['dxy_regime'] = (dxy > dxy_ma).astype(int)  # 1 = strengthening, 0 = weakening
        
        return features
    
    @staticmethod
    def extract_futures_cross_relationships(
        vx_data: Dict[str, pd.Series],
        commodity_data: Dict[str, pd.Series],
        dollar_data: Dict[str, pd.Series],
        spx_ret: pd.Series = None
    ) -> pd.DataFrame:
        """
        Extract cross-futures relationships and correlations.
        Captures macro regime shifts through futures interactions.
        """
        features = pd.DataFrame()
        
        # VX vs Crude correlation (risk appetite indicator)
        if 'VX1-VX2' in vx_data and 'CL1-CL2' in commodity_data:
            vx_spread = vx_data['VX1-VX2']
            cl_spread = commodity_data['CL1-CL2']
            
            features['vx_crude_corr_21d'] = (
                vx_spread.rolling(21).corr(cl_spread)
            )
            
            # Spread divergence (normalized rank difference)
            vx_rank = vx_spread.rolling(63).rank(pct=True)
            cl_rank = cl_spread.rolling(63).rank(pct=True)
            features['vx_crude_divergence'] = (vx_rank - cl_rank).abs()
        
        # Dollar vs VX (global risk dynamics)
        if 'VX1-VX2' in vx_data and 'Dollar_Index' in dollar_data:
            vx_spread = vx_data['VX1-VX2']
            dxy = dollar_data['Dollar_Index']
            dxy_ret = dxy.pct_change(21) * 100
            
            features['vx_dollar_corr_21d'] = (
                vx_spread.rolling(21).corr(dxy_ret)
            )
        
        # Dollar vs Crude (commodity-currency link)
        if 'Dollar_Index' in dollar_data and 'Crude_Oil' in commodity_data:
            dxy = dollar_data['Dollar_Index']
            crude = commodity_data['Crude_Oil']
            
            features['dollar_crude_corr_21d'] = (
                dxy.pct_change().rolling(21).corr(crude.pct_change())
            )
            
            # Expected negative correlation - divergence indicates macro shift
            features['dollar_crude_corr_breakdown'] = (
                features['dollar_crude_corr_21d'] + 0.5
            ).abs()  # Larger value = more divergence from expected
        
        # SPX vs futures (if SPX provided)
        if spx_ret is not None:
            if 'VX1-VX2' in vx_data:
                vx_spread = vx_data['VX1-VX2']
                features['spx_vx_spread_corr_21d'] = (
                    spx_ret.rolling(21).corr(vx_spread)
                )
            
            if 'Dollar_Index' in dollar_data:
                dxy_ret = dollar_data['Dollar_Index'].pct_change(21) * 100
                features['spx_dollar_corr_21d'] = (
                    spx_ret.rolling(21).corr(dxy_ret)
                )
        
        return features


class UnifiedFeatureEngine:
    """
    Enhanced unified feature engine with meta-features and futures integration.
    Maximizes feature richness for XGBoost and anomaly detection.
    """
    
    def __init__(self, data_fetcher):
        self.fetcher = data_fetcher
        self.meta_engine = MetaFeatureEngine()
        self.futures_engine = FuturesFeatureEngine()
    
    def build_complete_features(self, years: int = TRAINING_YEARS) -> dict:
        """Build complete feature set with all enhancements"""
        print(f"\n{'='*80}\nENHANCED FEATURE ENGINE V4\n{'='*80}\nWindow: {years}y")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365 + 450)  # Extra buffer
        start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
        # === STAGE 1: Core Data ===
        print("\n[1/8] Core market data (SPX, VIX)...")
        spx_df = self.fetcher.fetch_spx(start_str, end_str)
        vix = self.fetcher.fetch_vix(start_str, end_str)
        spx_ohlc = self.fetcher.fetch_yahoo('^GSPC', start_str, end_str, 
                                             incremental=True, extract_ohlc_features=True)
        
        if spx_df is None or vix is None:
            raise ValueError("❌ Core data fetch failed")
        
        spx = spx_df['Close'].squeeze()
        vix = vix.reindex(spx.index, method='ffill')
        print(f"   ✅ SPX: {len(spx)} | VIX: {len(vix)} | OHLC features: {len([c for c in spx_ohlc.columns if c.startswith('GSPC_')]) if spx_ohlc is not None else 0}")
        
        # === STAGE 2: Base Features (from original engine) ===
        print("\n[2/8] Base features (mean reversion, dynamics, technical)...")
        base_features = self._build_base_features(spx, vix, spx_ohlc)
        print(f"   ✅ {len(base_features.columns)} base features")
        
        # === STAGE 3: CBOE Data ===
        print("\n[3/8] CBOE alternative data...")
        cboe_data = self.fetcher.load_cboe_data()
        if cboe_data is not None:
            cboe_data = cboe_data.reindex(spx.index, method='ffill')
            cboe_features = self._build_cboe_features(cboe_data, vix)
            print(f"   ✅ {len(cboe_features.columns)} CBOE features")
        else:
            cboe_features = pd.DataFrame(index=spx.index)
            print("   ⚠️ CBOE data not available")
        
        # === STAGE 4: Futures Data ===
        print("\n[4/8] Futures data (VX, CL, DX)...")
        futures_features = self._build_futures_features(start_str, end_str, spx.index, spx)
        print(f"   ✅ {len(futures_features.columns)} futures features")
        
        # === STAGE 5: Macro Data ===
        print("\n[5/8] Macro/multi-asset data...")
        macro_df = self._fetch_macro_data(start_str, end_str, spx.index)
        macro_features = self._build_macro_features(macro_df) if macro_df is not None else pd.DataFrame(index=spx.index)
        print(f"   ✅ {len(macro_features.columns)} macro features")
        
        # === STAGE 6: Meta Features ===
        print("\n[6/8] Meta features (regimes, cross-asset, ROC, percentiles)...")
        combined_base = pd.concat([base_features, cboe_features, futures_features, macro_features], axis=1)
        meta_features = self._build_meta_features(combined_base, spx, vix, macro_df)
        print(f"   ✅ {len(meta_features.columns)} meta features")
        
        # === STAGE 7: FRED Data (optional but valuable) ===
        print("\n[7/8] FRED macro/rates data...")
        fred_features = self._build_fred_features(start_str, end_str, spx.index)
        print(f"   ✅ {len(fred_features.columns)} FRED features")
        
        # === STAGE 8: Consolidation ===
        print("\n[8/8] Consolidating all features...")
        all_features = [
            base_features,
            cboe_features,
            futures_features,
            macro_features,
            meta_features,
            fred_features
        ]
        
        features = pd.concat(all_features, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
        
        # Trim to requested years
        final_start = end_date - timedelta(days=years * 365)
        features = features[features.index >= final_start]
        spx = spx[spx.index >= final_start]
        vix = vix[vix.index >= final_start]
        
        missing_pct = features.isna().sum().sum() / features.size * 100
        
        print(f"\n{'='*80}\n✅ BUILD COMPLETE")
        print(f"   Total Features: {len(features.columns)}")
        print(f"   Samples: {len(features)}")
        print(f"   Range: {features.index[0].date()} to {features.index[-1].date()}")
        print(f"   Missing: {features.isna().sum().sum()} ({missing_pct:.2f}%)")
        
        # Feature breakdown by category
        print(f"\n   Feature Breakdown:")
        print(f"   - Base: {len(base_features.columns)}")
        print(f"   - CBOE: {len(cboe_features.columns)}")
        print(f"   - Futures: {len(futures_features.columns)}")
        print(f"   - Macro: {len(macro_features.columns)}")
        print(f"   - Meta: {len(meta_features.columns)}")
        print(f"   - FRED: {len(fred_features.columns)}")
        print("="*80)
        
        return {
            'features': features,
            'spx': spx,
            'vix': vix,
            'dates': features.index,
            'feature_breakdown': {
                'base': base_features.columns.tolist(),
                'cboe': cboe_features.columns.tolist(),
                'futures': futures_features.columns.tolist(),
                'macro': macro_features.columns.tolist(),
                'meta': meta_features.columns.tolist(),
                'fred': fred_features.columns.tolist()
            }
        }
    
    # ==================== BASE FEATURES (From Original Engine) ====================
    
    def _build_base_features(self, spx: pd.Series, vix: pd.Series, spx_ohlc: pd.DataFrame = None) -> pd.DataFrame:
        """Build all base features from original engine"""
        feature_groups = [
            self._vix_mean_reversion(vix),
            self._vix_dynamics(vix),
            self._vix_regimes(vix),
            self._spx_price_action(spx),
            self._spx_technical_indicators(spx),
            self._spx_volatility_regime(spx, vix),
            self._spx_vix_relationship(spx, vix),
            self._calendar_features(spx.index)
        ]
        
        if spx_ohlc is not None:
            ohlc_features = self._extract_ohlc_features(spx_ohlc)
            feature_groups.append(ohlc_features)
        
        return pd.concat(feature_groups, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
    
    def _vix_mean_reversion(self, vix: pd.Series) -> pd.DataFrame:
        """VIX mean reversion features"""
        features = pd.DataFrame(index=vix.index)
        features['vix'] = vix
        
        for w in [10, 21, 63, 126, 252]:
            ma = vix.rolling(w).mean().shift(1)
            features[f'vix_vs_ma{w}'] = vix - ma
            features[f'vix_vs_ma{w}_pct'] = ((vix - ma) / ma * 100)
        
        for w in [63, 126, 252]:
            ma, std = vix.rolling(w).mean().shift(1), vix.rolling(w).std().shift(1)
            features[f'vix_zscore_{w}d'] = (vix - ma) / std
        
        for w in [126, 252]:
            features[f'vix_percentile_{w}d'] = vix.rolling(w+1).apply(
                lambda x: pd.Series(x[:-1]).rank(pct=True).iloc[-1]*100 if len(x)>1 else 50, raw=False
            )
        
        features['reversion_strength_63d'] = (vix - vix.rolling(63).mean().shift(1)).abs()
        return features
    
    def _vix_dynamics(self, vix: pd.Series) -> pd.DataFrame:
        """VIX momentum and acceleration features"""
        features = pd.DataFrame(index=vix.index)
        
        for w in [1, 5, 10, 21]:
            features[f'vix_velocity_{w}d'] = vix.diff(w)
            features[f'vix_velocity_{w}d_pct'] = vix.pct_change(w) * 100
        
        features['vix_accel_5d'] = features['vix_velocity_5d'].diff(5)
        
        for w in [10, 21]:
            features[f'vix_vol_{w}d'] = vix.pct_change().rolling(w).std() * np.sqrt(252) * 100
        
        for w in [10, 21, 63]:
            velocity = vix.diff(w)
            ma, std = velocity.rolling(63).mean().shift(1), velocity.rolling(63).std().shift(1)
            features[f'vix_momentum_z_{w}d'] = (velocity - ma) / std
        
        features['vix_term_structure'] = vix.rolling(5).mean().shift(1) - vix.rolling(21).mean().shift(1)
        return features
    
    def _vix_regimes(self, vix: pd.Series) -> pd.DataFrame:
        """VIX regime classification"""
        features = pd.DataFrame(index=vix.index)
        
        features['vix_regime'] = pd.cut(
            vix, bins=REGIME_BOUNDARIES, labels=[0,1,2,3], include_lowest=True
        ).astype(int)
        
        regime_change = features['vix_regime'] != features['vix_regime'].shift(1)
        regime_id = regime_change.cumsum()
        features['days_in_regime'] = regime_id.groupby(regime_id).cumcount() + 1
        
        for regime_num in range(4):
            mask = features['vix_regime'] == regime_num
            regime_vix = vix[mask]
            if len(regime_vix) > 0:
                features['vix_displacement'] = 0.0
                features.loc[mask, 'vix_displacement'] = (
                    (vix[mask] - regime_vix.mean()) / regime_vix.std()
                )
        
        crisis_dates = vix[vix > 40].index
        features['days_since_crisis'] = 0
        for idx in vix.index:
            past = crisis_dates[crisis_dates < idx]
            features.loc[idx, 'days_since_crisis'] = (
                (idx - past[-1]).days if len(past) > 0 else 9999
            )
        
        features['elevated_flag'] = (vix > 20).astype(int)
        return features
    
    def _spx_price_action(self, spx: pd.Series) -> pd.DataFrame:
        """SPX price momentum and trends"""
        features = pd.DataFrame(index=spx.index)
        
        features['spx_lag1'] = spx.shift(1)
        features['spx_lag5'] = spx.shift(5)
        
        for w in [5, 10, 13, 21, 63]:
            features[f'spx_ret_{w}d'] = spx.pct_change(w) * 100
        
        for w in [20, 50, 200]:
            ma = spx.rolling(w).mean().shift(1)
            features[f'spx_vs_ma{w}'] = ((spx - ma) / ma) * 100
        
        returns = spx.pct_change()
        for w in [10, 21, 63]:
            features[f'spx_realized_vol_{w}d'] = (
                returns.rolling(w).std().shift(1) * np.sqrt(252) * 100
            )
        
        vol_10 = returns.rolling(10).std().shift(1) * np.sqrt(252) * 100
        vol_63 = returns.rolling(63).std().shift(1) * np.sqrt(252) * 100
        features['spx_vol_ratio_10_63'] = vol_10 / vol_63.replace(0, np.nan)
        
        for w in [10, 21]:
            ret = spx.pct_change(w)
            ret_ma, ret_std = ret.rolling(63).mean().shift(1), ret.rolling(63).std().shift(1)
            features[f'spx_momentum_z_{w}d'] = (ret - ret_ma) / ret_std
        
        features['spx_skew_21d'] = returns.rolling(21).skew()
        features['spx_kurt_21d'] = returns.rolling(21).kurt()
        return features
    
    def _spx_technical_indicators(self, spx: pd.Series) -> pd.DataFrame:
        """Technical indicators"""
        features = pd.DataFrame(index=spx.index)
        
        ma_20 = spx.rolling(20).mean().shift(1)
        ma_50 = spx.rolling(50).mean().shift(1)
        features['ma20_vs_ma50'] = ((ma_20 - ma_50) / ma_50 * 100)
        
        ma, std = spx.rolling(20).mean(), spx.rolling(20).std()
        features['bb_width_20d'] = ((ma + 2*std - (ma - 2*std)) / ma * 100)
        
        delta = spx.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        return features
    
    def _extract_ohlc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract OHLC microstructure features"""
        features = pd.DataFrame(index=df.index)
        
        spx_ohlc_cols = [col for col in df.columns if col.startswith('GSPC_')]
        if not spx_ohlc_cols:
            return features
        
        rename_map = {col: col.replace('GSPC_', 'spx_') for col in spx_ohlc_cols}
        base_features = df[spx_ohlc_cols].rename(columns=rename_map)
        
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
            hl_range = (h - l).replace(0, np.nan)
            
            if 'spx_gap' in base_features.columns:
                base_features['spx_gap_magnitude'] = base_features['spx_gap'].abs()
            
            if 'spx_range_expansion' in base_features.columns:
                avg_range = hl_range.rolling(21).mean()
                base_features['spx_range_expansion_z'] = (
                    (hl_range - avg_range) / hl_range.rolling(21).std()
                )
            
            if all(f'spx_{col}' in base_features.columns for col in ['body_to_range', 'upper_shadow', 'lower_shadow']):
                base_features['spx_doji'] = (base_features['spx_body_to_range'] < 0.1).astype(int)
                base_features['spx_long_body'] = (base_features['spx_body_to_range'] > 0.7).astype(int)
                base_features['spx_upper_wick_dominant'] = (base_features['spx_upper_shadow'] > 0.5).astype(int)
                base_features['spx_lower_wick_dominant'] = (base_features['spx_lower_shadow'] > 0.5).astype(int)
                
                base_features['spx_hammer'] = (
                    (base_features['spx_lower_shadow'] > 0.6) & 
                    (base_features['spx_upper_shadow'] < 0.15) & 
                    (base_features['spx_body_to_range'] < 0.3)
                ).astype(int)
                base_features['spx_shooting_star'] = (
                    (base_features['spx_upper_shadow'] > 0.6) & 
                    (base_features['spx_lower_shadow'] < 0.15) & 
                    (base_features['spx_body_to_range'] < 0.3)
                ).astype(int)
            
            base_features['spx_intraday_momentum'] = (c - o) / hl_range
            base_features['spx_intraday_mom_ma5'] = base_features['spx_intraday_momentum'].rolling(5).mean()
            
            if all(f'spx_{col}' in base_features.columns for col in ['upper_shadow', 'lower_shadow', 'range_pct']):
                base_features['spx_upper_rejection'] = (
                    base_features['spx_upper_shadow'] * base_features['spx_range_pct']
                )
                base_features['spx_lower_rejection'] = (
                    base_features['spx_lower_shadow'] * base_features['spx_range_pct']
                )
        
        return base_features
    
    def _spx_volatility_regime(self, spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
        """VIX vs realized vol"""
        features = pd.DataFrame(index=spx.index)
        returns = spx.pct_change()
        
        for w in [10, 21, 30, 63]:
            rv = returns.rolling(w).std().shift(1) * np.sqrt(252) * 100
            features[f'vix_vs_rv_{w}d'] = vix - rv
            features[f'vix_rv_ratio_{w}d'] = vix / rv.replace(0, np.nan)
        
        return features
    
    def _spx_vix_relationship(self, spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
        """SPX-VIX correlation"""
        features = pd.DataFrame(index=spx.index)
        
        spx_ret, vix_ret = spx.pct_change(), vix.pct_change()
        for w in [21, 63]:
            features[f'spx_vix_corr_{w}d'] = spx_ret.rolling(w).corr(vix_ret)
        
        for w in [10, 21]:
            features[f'spx_trend_{w}d'] = spx.pct_change(w) * 100
        
        return features
    
    def _calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Calendar effects"""
        features = pd.DataFrame(index=index)
        
        features['month'] = index.month
        features['quarter'] = index.quarter
        features['day_of_week'] = index.dayofweek
        features['day_of_month'] = index.day
        features['is_opex_week'] = 0
        
        for date in index:
            try:
                third_fridays = pd.date_range(
                    start=date.replace(day=1),
                    end=date.replace(day=1)+pd.offsets.MonthEnd(1),
                    freq='W-FRI'
                )
                if len(third_fridays) >= 3:
                    third_friday = third_fridays[2]
                    if third_friday - pd.Timedelta(days=4) <= date <= third_friday:
                        features.loc[date, 'is_opex_week'] = 1
            except:
                pass
        
        return features
    
    # ==================== CBOE FEATURES ====================
    
    def _build_cboe_features(self, cboe_data: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
        """Build all CBOE features"""
        base_cboe = self._cboe_features(cboe_data)
        cboe_interactions = self._cboe_interactions(cboe_data, vix)
        return pd.concat([base_cboe, cboe_interactions], axis=1)
    
    def _cboe_features(self, cboe_data: pd.DataFrame) -> pd.DataFrame:
        """CBOE indicator features"""
        features = pd.DataFrame(index=cboe_data.index)
        
        for col in cboe_data.columns:
            features[col] = cboe_data[col]
            features[f'{col}_change_21d'] = cboe_data[col].diff(21)
            ma, std = cboe_data[col].rolling(63).mean().shift(1), cboe_data[col].rolling(63).std().shift(1)
            features[f'{col}_zscore_63d'] = (cboe_data[col] - ma) / std
        
        if 'PCCE' in cboe_data.columns and 'PCCI' in cboe_data.columns:
            features['pc_divergence'] = cboe_data['PCCE'] - cboe_data['PCCI']
        if 'SKEW' in cboe_data.columns:
            features['tail_risk_elevated'] = (cboe_data['SKEW'] > 135).astype(int)
        if 'COR1M' in cboe_data.columns and 'COR3M' in cboe_data.columns:
            features['cor_term_structure'] = cboe_data['COR1M'] - cboe_data['COR3M']
        
        return features
    
    def _cboe_interactions(self, cboe: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
        """Enhanced CBOE cross-indicator dynamics"""
        features = pd.DataFrame(index=cboe.index)
        
        if 'SKEW' in cboe.columns:
            skew = cboe['SKEW']
            features['skew_velocity_5d'] = skew.diff(5)
            features['skew_velocity_21d'] = skew.diff(21)
            features['skew_vs_vix'] = skew - vix
            features['skew_vix_ratio'] = skew / vix.replace(0, np.nan)
            
            skew_ma = skew.rolling(126).mean()
            features['skew_regime'] = pd.cut(skew, bins=[0, 135, 145, 160, 300], 
                                             labels=[0, 1, 2, 3]).astype(float)
            features['skew_displacement'] = (skew - skew_ma) / skew.rolling(126).std()
        
        if all(col in cboe.columns for col in ['PCC', 'PCCE', 'PCCI']):
            pcc, pcce, pcci = cboe['PCC'], cboe['PCCE'], cboe['PCCI']
            
            features['pc_equity_inst_spread'] = pcce - pcci
            features['pc_equity_inst_spread_ma21'] = features['pc_equity_inst_spread'].rolling(21).mean()
            features['pc_equity_inst_divergence'] = (
                features['pc_equity_inst_spread'] - features['pc_equity_inst_spread_ma21']
            )
            
            features['pcce_extreme_high'] = (pcce > pcce.rolling(252).quantile(0.90)).astype(int)
            features['pcci_extreme_high'] = (pcci > pcci.rolling(252).quantile(0.90)).astype(int)
            features['pc_combined_extreme'] = (
                features['pcce_extreme_high'] & features['pcci_extreme_high']
            ).astype(int)
            
            features['pcc_velocity_10d'] = pcc.diff(10)
            features['pcci_velocity_10d'] = pcci.diff(10)
            features['pcc_accel_10d'] = features['pcc_velocity_10d'].diff(10)
        
        if all(col in cboe.columns for col in ['COR1M', 'COR3M']):
            cor1m, cor3m = cboe['COR1M'], cboe['COR3M']
            
            features['cor_term_slope'] = cor1m - cor3m
            features['cor_term_slope_change_21d'] = features['cor_term_slope'].diff(21)
            
            cor_avg = (cor1m + cor3m) / 2
            features['cor_avg'] = cor_avg
            features['cor_regime'] = pd.cut(cor_avg, bins=[-1, 0.3, 0.5, 0.7, 1], 
                                           labels=[0, 1, 2, 3]).astype(float)
            features['cor_spike'] = (cor_avg > cor_avg.rolling(63).quantile(0.90)).astype(int)
        
        if 'VXTH' in cboe.columns:
            vxth = cboe['VXTH']
            features['vxth_vs_vix'] = vxth - vix
            features['vxth_vix_ratio'] = vxth / vix.replace(0, np.nan)
            features['vxth_premium'] = (vxth - vix) / vix.replace(0, np.nan) * 100
            
            features['high_beta_vol_regime'] = pd.cut(
                features['vxth_vix_ratio'], 
                bins=[0, 1.1, 1.3, 1.5, 10], 
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        stress_indicators = []
        if 'SKEW' in cboe.columns:
            skew_stress = ((cboe['SKEW'] - 130) / 30).clip(0, 1)
            stress_indicators.append(skew_stress)
        if 'PCC' in cboe.columns:
            pcc_stress = ((cboe['PCC'] - 0.5) / 0.5).clip(0, 1)
            stress_indicators.append(pcc_stress)
        if 'COR1M' in cboe.columns:
            cor_stress = ((cboe['COR1M'] - 0.3) / 0.5).clip(0, 1)
            stress_indicators.append(cor_stress)
        
        if stress_indicators:
            features['cboe_stress_composite'] = pd.DataFrame(stress_indicators).T.mean(axis=1)
            features['cboe_stress_regime'] = pd.cut(
                features['cboe_stress_composite'], 
                bins=[0, 0.3, 0.6, 0.8, 1], 
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        return features
    
    # ==================== FUTURES FEATURES ====================
    
    def _build_futures_features(self, start_str: str, end_str: str, 
                                target_index: pd.DatetimeIndex, spx: pd.Series) -> pd.DataFrame:
        """Build comprehensive futures-based features"""
        all_futures = pd.DataFrame(index=target_index)
        
        # Load CBOE futures data
        cboe_data = self.fetcher.load_cboe_data()
        if cboe_data is None:
            return all_futures
        
        cboe_data = cboe_data.reindex(target_index, method='ffill')
        
        # Extract VX futures features
        vx_data = {}
        for key in ['VX1-VX2', 'VX2-VX1_RATIO']:
            if key in cboe_data.columns:
                vx_data[key] = cboe_data[key]
        
        if vx_data:
            vx_features = self.futures_engine.extract_vix_futures_features(vx_data)
            all_futures = pd.concat([all_futures, vx_features], axis=1)
        
        # Extract commodity futures features
        commodity_data = {}
        for key in ['CL1-CL2']:
            if key in cboe_data.columns:
                commodity_data[key] = cboe_data[key]
        
        # Fetch crude oil price
        crude = self.fetcher.fetch_yahoo('CL=F', start_str, end_str, incremental=True)
        if crude is not None and 'Close' in crude.columns:
            commodity_data['Crude_Oil'] = crude['Close'].reindex(target_index, method='ffill')
        
        if commodity_data:
            commodity_features = self.futures_engine.extract_commodity_futures_features(commodity_data)
            all_futures = pd.concat([all_futures, commodity_features], axis=1)
        
        # Extract dollar futures features
        dollar_data = {}
        for key in ['DX1-DX2']:
            if key in cboe_data.columns:
                dollar_data[key] = cboe_data[key]
        
        # Fetch dollar index
        dxy = self.fetcher.fetch_yahoo('DX=F', start_str, end_str, incremental=True)
        if dxy is not None and 'Close' in dxy.columns:
            dollar_data['Dollar_Index'] = dxy['Close'].reindex(target_index, method='ffill')
        
        if dollar_data:
            dollar_features = self.futures_engine.extract_dollar_futures_features(dollar_data)
            all_futures = pd.concat([all_futures, dollar_features], axis=1)
        
        # Cross-futures relationships
        if vx_data and (commodity_data or dollar_data):
            spx_ret = spx.pct_change(21) * 100 if spx is not None else None
            cross_features = self.futures_engine.extract_futures_cross_relationships(
                vx_data, commodity_data, dollar_data, spx_ret
            )
            all_futures = pd.concat([all_futures, cross_features], axis=1)
        
        return all_futures
    
    # ==================== MACRO FEATURES ====================
    
    def _fetch_macro_data(self, start_str: str, end_str: str, target_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        """Fetch macro/multi-asset data"""
        try:
            key_symbols = {
                'GLD': 'Gold',
                'SLV': 'Silver', 
                'DX=F': 'Dollar_Index',
                'CL=F': 'Crude_Oil',
                '^MOVE': 'Bond_Vol',
            }
            
            macro_series = []
            for symbol, name in key_symbols.items():
                df = self.fetcher.fetch_yahoo(symbol, start_str, end_str, incremental=True)
                if df is not None and 'Close' in df.columns:
                    s = df['Close'].squeeze()
                    s.name = name
                    macro_series.append(s)
            
            if macro_series:
                macro = pd.DataFrame(macro_series).T.reindex(target_index, method='ffill')
                return macro
            return None
        except Exception as e:
            print(f"   ⚠️ Macro fetch error: {e}")
            return None
    
    def _build_macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Multi-asset macro features"""
        features = pd.DataFrame(index=macro.index)
        
        for col in macro.columns:
            features[f'{col}_lag1'] = macro[col].shift(1)
            
            for w in [10, 21, 63]:
                features[f'{col}_mom_{w}d'] = macro[col].pct_change(w) * 100
            
            ma, std = macro[col].rolling(63).mean().shift(1), macro[col].rolling(63).std().shift(1)
            features[f'{col}_zscore_63d'] = (macro[col] - ma) / std
        
        return features
    
    # ==================== META FEATURES ====================
    
    def _build_meta_features(self, base_df: pd.DataFrame, spx: pd.Series, 
                            vix: pd.Series, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        """Build all meta features"""
        meta_groups = [
            self.meta_engine.extract_regime_indicators(base_df, vix, spx),
            self.meta_engine.extract_cross_asset_relationships(base_df, macro_df),
            self.meta_engine.extract_rate_of_change_features(base_df),
            self.meta_engine.extract_percentile_rankings(base_df)
        ]
        
        return pd.concat(meta_groups, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
    
    # ==================== FRED FEATURES ====================
    
    def _build_fred_features(self, start_str: str, end_str: str, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Build FRED macro/rates features"""
        features = pd.DataFrame(index=target_index)
        
        try:
            fred_data = self.fetcher.fetch_all_fred_series(start_str, end_str, incremental=True)
            if not fred_data:
                return features
            
            fred_series = []
            for category, df in fred_data.items():
                if df is not None and not df.empty:
                    fred_series.append(df)
            
            if not fred_series:
                return features
            
            fred = pd.concat(fred_series, axis=1).reindex(target_index, method='ffill')
            
            for col in fred.columns:
                features[f'{col}_level'] = fred[col]
                
                for w in [10, 21, 63]:
                    features[f'{col}_change_{w}d'] = fred[col].diff(w)
                
                for w in [63, 252]:
                    ma, std = fred[col].rolling(w).mean().shift(1), fred[col].rolling(w).std().shift(1)
                    features[f'{col}_zscore_{w}d'] = (fred[col] - ma) / std
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ FRED features error: {e}")
            return features


# ==================== TEST FUNCTION ====================

def test_enhanced_engine():
    """Test enhanced feature engine"""
    from data_fetcher_v7 import UnifiedDataFetcher
    
    print("\n" + "="*80)
    print("ENHANCED FEATURE ENGINE V4 - TEST")
    print("="*80)
    
    fetcher = UnifiedDataFetcher(log_level="INFO")
    engine = UnifiedFeatureEngine(data_fetcher=fetcher)
    
    # Build 2 years of features for quick test
    result = engine.build_complete_features(years=2)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print(f"Features shape: {result['features'].shape}")
    print(f"SPX shape: {result['spx'].shape}")
    print(f"VIX shape: {result['vix'].shape}")
    
    # Show sample of meta features
    print("\n" + "="*80)
    print("SAMPLE META FEATURES:")
    meta_cols = result['feature_breakdown']['meta'][:10]
    if meta_cols:
        print(result['features'][meta_cols].tail())
    
    print("="*80)


if __name__ == "__main__":
    test_enhanced_engine()
        print(f"