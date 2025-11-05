"""Unified Feature Engine - Complete Feature Set"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from .data_fetcher import UnifiedDataFetcher
from config import REGIME_BOUNDARIES, TRAINING_YEARS


class UnifiedFeatureEngine:
    def __init__(self, cboe_data_dir: str = "./CBOE_Data_Archive"):
        self.fetcher = UnifiedDataFetcher()
        self.cboe_data_dir = Path(cboe_data_dir)
    
    def _validate_buffer_sufficiency(self, data: pd.Series, required_window: int, data_name: str) -> bool:
        """Validate lookback buffer is sufficient for feature calculations."""
        if len(data) < required_window:
            warnings.warn(
                f"⚠️ BUFFER INSUFFICIENT: {data_name} has {len(data)} obs, "
                f"needs {required_window}. Features may contain NaN."
            )
            return False
        return True
    
    def build_complete_features(self, years: int = TRAINING_YEARS) -> dict:
        print(f"\n{'='*80}\nUNIFIED FEATURE ENGINE\n{'='*80}\nWindow: {years}y")
        
        MAX_WINDOW = 365
        SAFETY_MARGIN = 50
        REQUIRED_BUFFER = MAX_WINDOW + SAFETY_MARGIN
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
        print(f"\n[VALIDATION] Required buffer: {REQUIRED_BUFFER} days for max window {MAX_WINDOW}")
        
        print("\n[1/7] Core market data...")
        spx_df = self.fetcher.fetch_spx(start_str, end_str, lookback_buffer_days=REQUIRED_BUFFER)
        vix = self.fetcher.fetch_vix(start_str, end_str, lookback_buffer_days=REQUIRED_BUFFER)
        if spx_df is None or vix is None:
            raise ValueError("Core data fetch failed")
        
        spx = spx_df['Close'].squeeze()
        vix = vix.reindex(spx.index, method='ffill')
        
        spx_valid = self._validate_buffer_sufficiency(spx, REQUIRED_BUFFER, "SPX")
        vix_valid = self._validate_buffer_sufficiency(vix, REQUIRED_BUFFER, "VIX")
        
        if not spx_valid or not vix_valid:
            raise ValueError(
                f"❌ CRITICAL: Core data buffer insufficient. "
                f"SPX: {len(spx)}, VIX: {len(vix)}, Required: {REQUIRED_BUFFER}"
            )
        
        print(f"   ✅ SPX: {len(spx)} | VIX: {len(vix)}")
        
        print("\n[2/7] Macro data...")
        macro = self.fetcher.fetch_macro(start_str, end_str, lookback_buffer_days=REQUIRED_BUFFER)
        if macro is not None:
            macro = macro.reindex(spx.index, method='ffill')
            print(f"   ✅ {len(macro.columns)} series")
        
        print("\n[3/7] FRED data...")
        fred = self.fetcher.fetch_fred_multiple(start_str, end_str, lookback_buffer_days=REQUIRED_BUFFER)
        if fred is not None:
            fred = fred.reindex(spx.index, method='ffill')
            print(f"   ✅ {len(fred.columns)} series")
        
        print("\n[4/7] Commodities...")
        commodities = self.fetcher.fetch_commodities_fred(start_str, end_str, lookback_buffer_days=REQUIRED_BUFFER)
        if commodities is not None:
            commodities = commodities.reindex(spx.index, method='ffill')
            print(f"   ✅ {len(commodities.columns)} series")
        
        print("\n[5/7] CBOE indicators...")
        cboe_data = self._load_cboe_data()
        if cboe_data is not None:
            cboe_data = cboe_data.reindex(spx.index, method='ffill')
            print(f"   ✅ {len(cboe_data.columns)} indicators")
        
        print(f"\n[BUFFER VALIDATION SUMMARY]")
        all_valid = True
        for source_name, source_data in [("Macro", macro), ("FRED", fred), ("Commodities", commodities), ("CBOE", cboe_data)]:
            if source_data is not None:
                min_len = min(len(source_data[col].dropna()) for col in source_data.columns)
                if min_len < 252:
                    print(f"   ⚠️ {source_name}: Shortest series = {min_len} obs (need 252)")
                    all_valid = False
                else:
                    print(f"   ✅ {source_name}: All series ≥252 obs")
        
        if not all_valid:
            warnings.warn("⚠️ Some data sources have insufficient lookback. Features may contain NaN.")
        
        print("\n[6/7] Engineering features...")
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
        if cboe_data is not None:
            feature_groups.append(self._cboe_features(cboe_data))
        if fred is not None:
            feature_groups.append(self._fred_features(fred))
        if macro is not None:
            feature_groups.append(self._macro_features(macro))
        if commodities is not None:
            feature_groups.append(self._commodity_features(commodities))
        
        print("\n[7/7] Consolidating...")
        features = pd.concat(feature_groups, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
        
        missing_pct = features.isna().sum().sum() / features.size * 100
        if missing_pct > 5.0:
            warnings.warn(
                f"⚠️ HIGH MISSING DATA: {missing_pct:.2f}% NaN. "
                f"Buffer may be insufficient or data sources incomplete."
            )
        
        print(f"\n{'='*80}\n✅ BUILD COMPLETE")
        print(f"   Features: {len(features.columns)} | Samples: {len(features)}")
        print(f"   Range: {features.index[0].date()} to {features.index[-1].date()}")
        print(f"   Missing: {features.isna().sum().sum()} ({missing_pct:.2f}%)\n{'='*80}")
        
        return {'features': features, 'spx': spx, 'vix': vix, 'dates': features.index}
    
    def _vix_mean_reversion(self, vix: pd.Series) -> pd.DataFrame:
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
        features = pd.DataFrame(index=vix.index)
        features['vix_regime'] = pd.cut(vix, bins=REGIME_BOUNDARIES, labels=[0,1,2,3], include_lowest=True).astype(int)
        regime_change = features['vix_regime'] != features['vix_regime'].shift(1)
        regime_id = regime_change.cumsum()
        features['days_in_regime'] = regime_id.groupby(regime_id).cumcount() + 1
        
        for regime_num in range(4):
            mask = features['vix_regime'] == regime_num
            regime_vix = vix[mask]
            if len(regime_vix) > 0:
                features['vix_displacement'] = 0.0
                features.loc[mask, 'vix_displacement'] = (vix[mask] - regime_vix.mean()) / regime_vix.std()
        
        crisis_dates = vix[vix > 40].index
        features['days_since_crisis'] = 0
        for idx in vix.index:
            past = crisis_dates[crisis_dates < idx]
            features.loc[idx, 'days_since_crisis'] = (idx - past[-1]).days if len(past) > 0 else 9999
        
        features['elevated_flag'] = (vix > 20).astype(int)
        return features
    
    def _spx_price_action(self, spx: pd.Series) -> pd.DataFrame:
        features = pd.DataFrame(index=spx.index)
        features['spx_lag1'], features['spx_lag5'] = spx.shift(1), spx.shift(5)
        for w in [5, 10, 13, 21, 63]:
            features[f'spx_ret_{w}d'] = spx.pct_change(w) * 100
        for w in [20, 50, 200]:
            ma = spx.rolling(w).mean().shift(1)
            features[f'spx_vs_ma{w}'] = ((spx - ma) / ma) * 100
        
        returns = spx.pct_change()
        for w in [10, 21, 63]:
            features[f'spx_realized_vol_{w}d'] = returns.rolling(w).std().shift(1) * np.sqrt(252) * 100
        
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
    
    def _spx_volatility_regime(self, spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
        features = pd.DataFrame(index=spx.index)
        returns = spx.pct_change()
        for w in [10, 21, 30, 63]:
            rv = returns.rolling(w).std().shift(1) * np.sqrt(252) * 100
            features[f'vix_vs_rv_{w}d'] = vix - rv
            features[f'vix_rv_ratio_{w}d'] = vix / rv.replace(0, np.nan)
        
        rv_21 = returns.rolling(21).std().shift(1) * np.sqrt(252) * 100
        rv_63 = returns.rolling(63).std().shift(1) * np.sqrt(252) * 100
        features['vix_vs_avg_rv'] = vix - (rv_21 * 0.6 + rv_63 * 0.4)
        return features
    
    def _spx_vix_relationship(self, spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
        features = pd.DataFrame(index=spx.index)
        spx_ret, vix_ret = spx.pct_change(), vix.pct_change()
        for w in [21, 63]:
            features[f'spx_vix_corr_{w}d'] = spx_ret.rolling(w).corr(vix_ret)
        for w in [10, 21]:
            features[f'spx_trend_{w}d'] = spx.pct_change(w) * 100
        return features
    
    def _calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
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
    
    def _cboe_features(self, cboe_data: pd.DataFrame) -> pd.DataFrame:
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
    
    def _fred_features(self, fred: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=fred.index)
        for col in fred.columns:
            features[f'{col}_level'] = fred[col]
            features[f'{col}_lag1'] = fred[col].shift(1)
            for w in [10, 21, 63]:
                features[f'{col}_change_{w}d'] = fred[col].diff(w)
            for w in [63, 252]:
                ma, std = fred[col].rolling(w).mean().shift(1), fred[col].rolling(w).std().shift(1)
                features[f'{col}_zscore_{w}d'] = (fred[col] - ma) / std
        return features
    
    def _macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=macro.index)
        for col in macro.columns:
            features[f'{col}_lag1'] = macro[col].shift(1)
            for w in [10, 21, 63]:
                features[f'{col}_mom_{w}d'] = macro[col].pct_change(w) * 100
            ma, std = macro[col].rolling(63).mean().shift(1), macro[col].rolling(63).std().shift(1)
            features[f'{col}_zscore_63d'] = (macro[col] - ma) / std
        return features
    
    def _commodity_features(self, commodities: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=commodities.index)
        for col in commodities.columns:
            features[f'{col}_lag1'] = commodities[col].shift(1)
            for w in [10, 21, 63]:
                features[f'{col}_mom_{w}d'] = commodities[col].pct_change(w) * 100
            ma, std = commodities[col].rolling(63).mean().shift(1), commodities[col].rolling(63).std().shift(1)
            features[f'{col}_zscore_63d'] = (commodities[col] - ma) / std
        return features
    
    def _load_cboe_data(self) -> pd.DataFrame:
        cboe_files = {
            'SKEW': 'SKEW_INDEX_CBOE.csv',
            'PCCI': 'PCCI_INDX_CBOE.csv',
            'PCCE': 'PCCE_EQUITIES_CBOE.csv',
            'PCC': 'PCC_INDX_EQ_TOTAL_CBOE.csv',
            'COR1M': 'COR1M_CBOE.csv',
            'COR3M': 'COR3M_CBOE.csv',
            'VXTH': 'VXTH_TAILHEDGE_CBOE.csv'
        }
        series_list = []
        for name, filename in cboe_files.items():
            filepath = self.cboe_data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
                    if 'Close' in df.columns:
                        s = df['Close'].squeeze()
                        s.name = name
                        series_list.append(s)
                except Exception as e:
                    print(f"   ⚠️ {filename}: {e}")
        return pd.concat(series_list, axis=1) if series_list else None