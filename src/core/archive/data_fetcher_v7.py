"""Unified Data Fetcher V7 - Bug Fix: Corrected gap calculation in OHLCFeatureExtractor"""
import os
import json
import logging
import requests
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import time

warnings.filterwarnings('ignore')


class DataFetchLogger:
    def __init__(self, name: str = "DataFetcher"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str, exc: Exception = None):
        self.logger.error(f"{msg}: {type(exc).__name__}: {str(exc)}" if exc else msg)
    def debug(self, msg: str): self.logger.debug(msg)


class DataValidator:
    @staticmethod
    def is_business_day_range(start: str, end: str) -> bool:
        try:
            start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
            return start_dt <= end_dt and end_dt <= pd.Timestamp.now() + pd.Timedelta(days=1) and \
                   (end_dt - start_dt).days <= 365 * 50
        except:
            return False
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str, min_rows: int = 10) -> Tuple[bool, str]:
        if df is None or df.empty:
            return False, f"{name}: Empty"
        if len(df) < min_rows:
            return False, f"{name}: Insufficient ({len(df)})"
        if not isinstance(df.index, pd.DatetimeIndex):
            return False, f"{name}: Invalid index"
        null_pct = df.isna().sum().sum() / df.size * 100
        return (True, "OK") if null_pct <= 50 else (False, f"{name}: Excessive nulls ({null_pct:.1f}%)")
    
    @staticmethod
    def is_historical_data(end_date: str) -> bool:
        try:
            return (pd.Timestamp.now() - pd.to_datetime(end_date)).days > 7
        except:
            return False


class OHLCFeatureExtractor:
    """Extract informative features from OHLC data"""
    
    @staticmethod
    def extract_price_action_features(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
        """
        Extract price action features from OHLC data
        Returns DataFrame with features like body size, range, shadows, etc.
        """
        features = pd.DataFrame(index=df.index)
        
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return features
        
        o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
        
        # Basic candle properties
        features[f'{prefix}body_size'] = (c - o).abs() / o  # Directional momentum
        features[f'{prefix}range'] = (h - l) / o  # Volatility/uncertainty
        features[f'{prefix}range_pct'] = (h - l) / l * 100
        
        # Shadow analysis (rejection levels)
        hl_range = (h - l).replace(0, np.nan)
        features[f'{prefix}upper_shadow'] = (h - np.maximum(o, c)) / hl_range  # Rejection at highs
        features[f'{prefix}lower_shadow'] = (np.minimum(o, c) - l) / hl_range  # Rejection at lows
        
        # Close position in range (0=low, 1=high)
        features[f'{prefix}close_position'] = (c - l) / hl_range
        
        # Directional indicators
        features[f'{prefix}is_bullish'] = (c > o).astype(int)
        features[f'{prefix}body_to_range'] = (c - o).abs() / hl_range  # Strong moves vs wicks
        
        # Gap analysis
        features[f'{prefix}gap'] = (o - c.shift(1)) / c.shift(1)  # Fixed: gap vs previous close
        features[f'{prefix}gap_filled'] = ((c >= c.shift(1)) & (o < c.shift(1))).astype(int)
        
        # Typical price & momentum
        features[f'{prefix}typical_price'] = (h + l + c) / 3
        features[f'{prefix}typical_price_chg'] = features[f'{prefix}typical_price'].pct_change()
        
        # Price velocity (momentum across timeframes)
        for w in [3, 5, 10, 21]:
            features[f'{prefix}close_velocity_{w}d'] = c.pct_change(w) * 100
            features[f'{prefix}typical_velocity_{w}d'] = features[f'{prefix}typical_price'].pct_change(w) * 100
        
        # Range expansion/contraction
        avg_range = (h - l).rolling(21).mean()
        features[f'{prefix}range_expansion'] = (h - l) / avg_range
        
        # Trend strength via close position consistency
        for w in [5, 10, 21]:
            features[f'{prefix}close_pos_ma_{w}d'] = features[f'{prefix}close_position'].rolling(w).mean()
        
        return features
    
    @staticmethod
    def extract_volume_features(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
        """Extract volume-related features if Volume column exists"""
        features = pd.DataFrame(index=df.index)
        
        if 'Volume' not in df.columns:
            return features
        
        vol = df['Volume']
        c = df['Close']
        
        # Volume trends
        features[f'{prefix}volume'] = vol
        features[f'{prefix}volume_ma20'] = vol.rolling(20).mean()
        features[f'{prefix}volume_ratio'] = vol / vol.rolling(20).mean()
        
        # Volume momentum
        for w in [5, 10, 21]:
            features[f'{prefix}volume_chg_{w}d'] = vol.pct_change(w) * 100
        
        # Price-volume relationship
        features[f'{prefix}price_volume_corr_21d'] = c.pct_change().rolling(21).corr(vol.pct_change())
        
        # Volume-weighted price
        features[f'{prefix}vwap_approx'] = (df['Close'] * vol).rolling(20).sum() / vol.rolling(20).sum()
        
        return features


class UnifiedDataFetcher:
    """
    Consolidated data fetcher supporting:
    - FRED (prioritized for macro/rates)
    - Yahoo Finance (market data, futures, ETFs)
    - CBOE alternative data files
    - Incremental updates (fetch only new data from last row)
    """
    
    # Complete instrument definitions
    YAHOO_SYMBOLS = {
        'VIX_FAMILY': ["^VIX", "^VIX3M", "^VIX6M", "^VIX9D", "^VVIX"],
        'VOL_INDICES': ["^MOVE", "^SKEW", "^GVZ", "^OVX", "^GAMMA"],
        'CBOE_INDICES': ["^DSPX"],
        'EQUITY_IDX': ["^SPX", "^GSPC", "^NDX", "^DJI", "^RUT"],
        'EQUITY_FUT': ["ES=F", "NQ=F", "YM=F", "RTY=F"],
        'PRECIOUS': ["GLD", "SLV", "GC=F", "SI=F"],
        'CURRENCY': ["DX=F", "DX-Y.NYB", "6E=F", "6J=F", "6B=F", "6C=F", "6A=F"],
        'COMMODITIES': ["CL=F", "BZ=F", "NG=F", "HG=F", "HO=F", "RB=F"],
        'TREAS_FUT': ["ZN=F", "ZB=F", "ZT=F", "ZF=F", "UB=F"],
    }
    
    FRED_SYMBOLS = {
        'TREASURY_YIELDS': {
            'DGS1MO': '1M_Treasury', 'DGS3MO': '3M_Treasury', 'DGS6MO': '6M_Treasury',
            'DGS1': '1Y_Treasury', 'DGS2': '2Y_Treasury', 'DGS3': '3Y_Treasury',
            'DGS5': '5Y_Treasury', 'DGS7': '7Y_Treasury', 'DGS10': '10Y_Treasury',
            'DGS20': '20Y_Treasury', 'DGS30': '30Y_Treasury',
        },
        'TREASURY_SPREADS': {
            'T10Y2Y': 'Yield_Curve_10Y2Y', 'T10Y3M': 'Yield_Curve_10Y3M',
            'T5YIE': 'Breakeven_Inflation_5Y', 'T10YIE': 'Breakeven_Inflation_10Y',
            'T5YIFR': 'Forward_Inflation_5Y',
        },
        'TIPS': {
            'DFII5': '5Y_TIPS', 'DFII7': '7Y_TIPS', 'DFII10': '10Y_TIPS',
            'DFII20': '20Y_TIPS', 'DFII30': '30Y_TIPS',
        },
        'FED_RATES': {
            'DFF': 'Fed_Funds_Effective', 'EFFR': 'Fed_Funds_Effective_Rate',
            'OBFR': 'Overnight_Bank_Funding', 'SOFR': 'SOFR',
            'SOFR30DAYAVG': 'SOFR_30D', 'SOFR90DAYAVG': 'SOFR_90D', 'SOFR180DAYAVG': 'SOFR_180D',
        },
        'FED_POLICY': {
            'DFEDTARU': 'Fed_Funds_Target_Upper', 'DFEDTARL': 'Fed_Funds_Target_Lower',
            'IORB': 'Interest_Reserve_Balances',
        },
        'CREDIT_SPREADS': {
            'BAMLH0A0HYM2': 'High_Yield_OAS', 'BAMLC0A0CM': 'Corporate_Master_OAS',
            'BAMLH0A1HYBB': 'BB_High_Yield_OAS', 'BAMLH0A2HYB': 'B_High_Yield_OAS',
            'BAMLH0A3HYC': 'CCC_High_Yield_OAS', 'BAMLC0A1CAAAEY': 'AAA_Corporate_OAS',
            'BAMLC0A2CAAY': 'AA_Corporate_OAS', 'BAMLC0A3CAY': 'A_Corporate_OAS',
            'BAMLC0A4CBBB': 'BBB_Corporate_OAS',
        },
        'DOLLAR_INDICES': {
            'DTWEXBGS': 'Dollar_Index_Broad', 'DTWEXM': 'Dollar_Index_Major',
            'DTWEXO': 'Dollar_Index_Other', 'DTWEXEMEGS': 'Dollar_Index_Emerging',
        },
        'VOLATILITY': {
            'VIXCLS': 'VIX_Close', 'VXVCLS': 'VXV_3M_VIX',
        },
        'ECONOMIC': {
            'UNRATE': 'Unemployment_Rate', 'CPIAUCSL': 'CPI', 'CPILFESL': 'Core_CPI',
            'PCEPI': 'PCE_Price_Index', 'PCEPILFE': 'Core_PCE',
            'GDP': 'GDP', 'GDPC1': 'Real_GDP',
            'UMCSENT': 'Consumer_Sentiment', 'INDPRO': 'Industrial_Production',
            'PAYEMS': 'Nonfarm_Payrolls',
        },
        'MONEY_SUPPLY': {
            'M1SL': 'M1_Money_Stock', 'M2SL': 'M2_Money_Stock',
            'WALCL': 'Fed_Balance_Sheet', 'WTREGEN': 'Treasury_General_Account',
            'RRPONTSYD': 'Reverse_Repo_Operations',
        },
        'COMMODITIES': {
            'DCOILWTICO': 'Crude_Oil_WTI', 'DCOILBRENTEU': 'Brent_Crude',
            'DHHNGSP': 'Natural_Gas',
        }
    }
    
    CBOE_FILES = {
        'SKEW': 'SKEW_INDEX_CBOE.csv',
        'PCCI': 'PCCI_INDX_CBOE.csv',
        'PCCE': 'PCCE_EQUITIES_CBOE.csv',
        'PCC': 'PCC_INDX_EQ_TOTAL_CBOE.csv',
        'COR1M': 'COR1M_CBOE.csv',
        'COR3M': 'COR3M_CBOE.csv',
        'VXTH': 'VXTH_TAILHEDGE_CBOE.csv',
        'CNDR': 'CNDR_SPX_IRON_CONDOR_CBOE.csv',
        'VX1-VX2': 'VX1-VX2.csv',
        'VX2-VX1_RATIO': '(VX2-VX1)OVER(VX1).csv',
        'CL1-CL2': 'CL1-CL2.csv',
        'DX1-DX2': 'DX1-DX2.csv',
        'BVOL': 'BVOL.csv',
        'BFLY': 'BFLY.csv',
        'DSPX': 'DSPX.csv',
        'VPN': 'VPN.csv',
        'GAMMA': 'GAMMA.csv',
        'VIX_VIX3M': 'VIXoverVIX3M.csv',
        'GOLDSILVER': 'GOLDSILVER_RATIO.csv'
    }
    
    def __init__(self, cache_dir: str = './data_cache', cboe_data_dir: str = './CBOE_Data_Archive',
                 log_level: str = "INFO"):
        self.fred_base_url = 'https://api.stlouisfed.org/fred/series/observations'
        self.fred_api_key = self._read_fred_api_key()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cboe_data_dir = Path(cboe_data_dir)
        
        self.logger = DataFetchLogger()
        if log_level == "DEBUG":
            self.logger.logger.setLevel(logging.DEBUG)
        elif log_level == "WARNING":
            self.logger.logger.setLevel(logging.WARNING)
        self.validator = DataValidator()
        self.ohlc_extractor = OHLCFeatureExtractor()
        
        self.cache_metadata_path = self.cache_dir / "_cache_metadata.json"
        self._load_cache_metadata()
        
        if self.fred_api_key is None:
            self.logger.warning("FRED API key missing - FRED unavailable")
        else:
            self.logger.debug(f"FRED key loaded")
    
    def _load_cache_metadata(self):
        if self.cache_metadata_path.exists():
            try:
                with open(self.cache_metadata_path, 'r') as f:
                    self.cache_metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Cache metadata load failed: {e}")
                self.cache_metadata = {}
        else:
            self.cache_metadata = {}
    
    def _save_cache_metadata(self):
        try:
            with open(self.cache_metadata_path, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Cache metadata save failed: {e}")
    
    def _update_cache_metadata(self, cache_key: str, etag: str = None):
        self.cache_metadata[cache_key] = {
            'created': datetime.now().isoformat(),
            'etag': etag
        }
        self._save_cache_metadata()
    
    def _is_cache_stale(self, cache_key: str, ttl_days: int = 90) -> bool:
        if cache_key not in self.cache_metadata:
            return True
        try:
            created = datetime.fromisoformat(self.cache_metadata[cache_key]['created'])
            age_days = (datetime.now() - created).days
            return age_days > ttl_days
        except:
            return True
    
    def _read_fred_api_key(self) -> Optional[str]:
        config_locations = [
            Path(__file__).parent / 'json_data' / 'config.json',
            Path(__file__).parent.parent / 'json_data' / 'config.json',
            Path.cwd() / 'json_data' / 'config.json',
        ]
        for config_path in config_locations:
            try:
                if config_path.resolve().exists():
                    with open(config_path.resolve(), 'r') as f:
                        key = json.load(f).get('fred_api_key')
                        if key:
                            return key
            except:
                continue
        return os.getenv('FRED_API_KEY')
    
    def _get_last_date_from_cache(self, cache_path: Path) -> Optional[str]:
        """Get the last date from cached data to enable incremental updates"""
        try:
            if not cache_path.exists():
                return None
            df = pd.read_parquet(cache_path)
            if df.empty:
                return None
            return df.index[-1].strftime('%Y-%m-%d')
        except Exception as e:
            self.logger.debug(f"Could not read last date from cache: {e}")
            return None
    
    def _merge_with_cache(self, new_data: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
        """Merge new data with existing cache, avoiding duplicates"""
        try:
            if not cache_path.exists():
                return new_data
            cached_df = pd.read_parquet(cache_path)
            if cached_df.empty:
                return new_data
            # Combine and remove duplicates
            combined = pd.concat([cached_df, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            return combined
        except Exception as e:
            self.logger.warning(f"Cache merge failed: {e}, using new data only")
            return new_data
    
    def _cache_path(self, name: str, start: str, end: str, cache_type: str = "daily") -> Path:
        return self.cache_dir / f"{name}_{start}_{end}_{cache_type}.parquet"
    
    def _should_use_cache(self, path: Path, is_historical: bool, cache_key: str = None) -> bool:
        if not path.exists():
            return False
        if is_historical:
            if cache_key and self._is_cache_stale(cache_key, ttl_days=90):
                return False
            return True
        return datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date()
    
    def _normalize_data(self, data, name: str) -> Optional[pd.DataFrame]:
        if data is None:
            return None
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        if data.empty:
            return None
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                self.logger.error(f"{name}: Index conversion failed", e)
                return None
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data.index = pd.DatetimeIndex(data.index.date)
        if data.index.duplicated().any():
            data = data.groupby(data.index).last()
        is_valid, msg = self.validator.validate_dataframe(data, name)
        return data if is_valid else None
    
    def _safe_cache_write(self, data: pd.DataFrame, path: Path, name: str):
        try:
            data.to_parquet(path)
            self.logger.debug(f"{name}: Cached")
        except Exception as e:
            self.logger.error(f"{name}: Cache write failed", e)
    
    def _safe_cache_read(self, path: Path, name: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_parquet(path)
            self.logger.debug(f"{name}: From cache")
            return self._normalize_data(df, f"{name}:cache")
        except Exception as e:
            self.logger.error(f"{name}: Cache read failed", e)
            return None
    
    # ==================== FRED FETCHING ====================
    
    def fetch_fred(self, series_id: str, start_date: str = None, end_date: str = None,
                   incremental: bool = True) -> Optional[pd.Series]:
        """Fetch FRED series with optional incremental update"""
        if not self.fred_api_key:
            return None
        
        # Try incremental update if enabled
        if incremental and start_date:
            cache_path = self.cache_dir / f"fred_{series_id}.parquet"
            last_date = self._get_last_date_from_cache(cache_path)
            if last_date:
                # Fetch only new data
                update_start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                if pd.to_datetime(update_start) <= pd.to_datetime(end_date or datetime.now().strftime('%Y-%m-%d')):
                    self.logger.info(f"FRED:{series_id}: Incremental from {update_start}")
                    start_date = update_start
        
        try:
            url = f'{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&file_type=json'
            if start_date:
                url += f'&observation_start={start_date}'
            if end_date:
                url += f'&observation_end={end_date}'
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            series_data = {item['date']: float(item['value'])
                          for item in response.json()['observations'] if item['value'] != '.'}
            if not series_data:
                return None
            
            series = pd.Series(series_data)
            series.index = pd.to_datetime(series.index)
            series.name = series_id
            df = self._normalize_data(pd.DataFrame(series), f'FRED:{series_id}')
            
            if df is not None and incremental:
                # Merge with cache
                cache_path = self.cache_dir / f"fred_{series_id}.parquet"
                df = self._merge_with_cache(df, cache_path)
                self._safe_cache_write(df, cache_path, f'FRED:{series_id}')
            
            return df[series_id] if df is not None else None
            
        except Exception as e:
            self.logger.error(f"FRED:{series_id}: Failed", e)
            return None
    
    def fetch_all_fred_series(self, start_date: str = None, end_date: str = None,
                             incremental: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch all FRED series organized by category"""
        if not self.fred_api_key:
            self.logger.warning("FRED API key not available")
            return {}
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*20)).strftime('%Y-%m-%d')
        
        all_data = {}
        total = sum(len(series_dict) for series_dict in self.FRED_SYMBOLS.values())
        count = 0
        
        self.logger.info(f"Fetching {total} FRED series...")
        
        for category, series_dict in self.FRED_SYMBOLS.items():
            series_list = []
            for series_id, name in series_dict.items():
                count += 1
                self.logger.info(f"[{count}/{total}] {series_id}: {name}")
                
                s = self.fetch_fred(series_id, start_date, end_date, incremental=incremental)
                if s is not None:
                    s.name = name
                    series_list.append(s)
                    self.logger.info(f"  ✅ {len(s)} rows")
                else:
                    self.logger.warning(f"  ❌ Failed")
                
                # Rate limit politeness
                time.sleep(0.15)
            
            if series_list:
                all_data[category] = pd.concat(series_list, axis=1, join='outer')
                self.logger.info(f"{category}: ✅ {len(series_list)} series")
        
        return all_data
    
    # ==================== YAHOO FINANCE FETCHING ====================
    
    def fetch_yahoo(self, ticker: str, start_date: str = None, end_date: str = None,
                   incremental: bool = True, extract_ohlc_features: bool = False) -> Optional[pd.DataFrame]:
        """Fetch Yahoo Finance data with optional incremental update and OHLC feature extraction"""
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*20)).strftime('%Y-%m-%d')
        
        # Try incremental update
        cache_path = self.cache_dir / f"yahoo_{ticker.replace('^','_').replace('=','_')}.parquet"
        if incremental:
            last_date = self._get_last_date_from_cache(cache_path)
            if last_date:
                update_start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                if pd.to_datetime(update_start) <= pd.to_datetime(end_date):
                    self.logger.info(f"Yahoo:{ticker}: Incremental from {update_start}")
                    start_date = update_start
        
        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d')
            if df.empty:
                return None
            
            df = self._normalize_data(df, f'Yahoo:{ticker}')
            if df is None:
                return None
            
            # Extract OHLC features if requested
            if extract_ohlc_features:
                ohlc_features = self.ohlc_extractor.extract_price_action_features(df, prefix=ticker.replace('^','').replace('=','').replace('-','_')+'_')
                vol_features = self.ohlc_extractor.extract_volume_features(df, prefix=ticker.replace('^','').replace('=','').replace('-','_')+'_')
                df = pd.concat([df, ohlc_features, vol_features], axis=1)
            
            if incremental:
                df = self._merge_with_cache(df, cache_path)
                self._safe_cache_write(df, cache_path, f'Yahoo:{ticker}')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Yahoo:{ticker}: Failed", e)
            return None
    
    def fetch_all_yahoo_symbols(self, start_date: str = None, end_date: str = None,
                                incremental: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch all Yahoo Finance symbols organized by category"""
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*20)).strftime('%Y-%m-%d')
        
        all_data = {}
        total = sum(len(symbols) for symbols in self.YAHOO_SYMBOLS.values())
        count = 0
        
        self.logger.info(f"Fetching {total} Yahoo Finance symbols...")
        
        for category, symbols in self.YAHOO_SYMBOLS.items():
            category_data = {}
            for symbol in symbols:
                count += 1
                self.logger.info(f"[{count}/{total}] {symbol}")
                
                df = self.fetch_yahoo(symbol, start_date, end_date, incremental=incremental)
                if df is not None:
                    category_data[symbol] = df
                    self.logger.info(f"  ✅ {len(df)} rows")
                else:
                    self.logger.warning(f"  ❌ Failed")
                
                time.sleep(0.1)  # Rate limit politeness
            
            if category_data:
                all_data[category] = category_data
                self.logger.info(f"{category}: ✅ {len(category_data)} symbols")
        
        return all_data
    
    # ==================== CBOE DATA LOADING ====================
    
    def load_cboe_data(self) -> Optional[pd.DataFrame]:
        """Load CBOE alternative data from CSV files"""
        series_list = []
        
        for name, filename in self.CBOE_FILES.items():
            filepath = self.cboe_data_dir / filename
            if not filepath.exists():
                self.logger.debug(f"CBOE:{name}: File not found")
                continue
            
            try:
                df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
                if 'Close' in df.columns:
                    s = df['Close'].squeeze()
                    s.name = name
                    series_list.append(s)
                    self.logger.info(f"CBOE:{name}: ✅ {len(s)} rows")
                else:
                    self.logger.warning(f"CBOE:{name}: No 'Close' column")
            except Exception as e:
                self.logger.error(f"CBOE:{name}: Failed", e)
        
        if not series_list:
            self.logger.warning("CBOE: No data loaded")
            return None
        
        df = pd.concat(series_list, axis=1, join='outer')
        self.logger.info(f"CBOE: ✅ {len(df.columns)} series loaded")
        return df
    
    # ==================== CONVENIENCE METHODS ====================
    
    def fetch_spx(self, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        return self.fetch_yahoo('^GSPC', start_date, end_date)
    
    def fetch_vix(self, start_date: str = None, end_date: str = None) -> Optional[pd.Series]:
        df = self.fetch_yahoo('^VIX', start_date, end_date)
        return df['Close'] if df is not None else None


def test_fetcher():
    """Test the consolidated fetcher"""
    print(f"\n{'='*80}\nUNIFIED DATA FETCHER V6 - TEST\n{'='*80}")
    
    fetcher = UnifiedDataFetcher(log_level="INFO")
    
    print(f"\n✅ FRED: {'Available' if fetcher.fred_api_key else 'NOT AVAILABLE'}")
    print(f"✅ Yahoo Finance: Available")
    print(f"✅ CBOE Data Dir: {fetcher.cboe_data_dir}")
    
    # Test incremental fetch
    print(f"\n{'='*80}\nTesting Incremental Fetch\n{'='*80}")
    vix = fetcher.fetch_vix(start_date='2020-01-01')
    if vix is not None:
        print(f"✅ VIX: {len(vix)} rows | Range: {vix.index[0]} to {vix.index[-1]}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    test_fetcher()
