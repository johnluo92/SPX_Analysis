"""Unified Data Fetcher V5 - With Cache TTL & Revision Detection"""
import os
import json
import logging
import requests
import warnings
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings('ignore')

from config import MACRO_TICKERS, FRED_SERIES, CACHE_DIR, COMMODITY_FRED_SERIES


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


class UnifiedDataFetcher:
    def __init__(self, cache_dir: str = CACHE_DIR, log_level: str = "INFO"):
        self.fred_base_url = 'https://api.stlouisfed.org/fred/series/observations'
        self.fred_api_key = self._read_fred_api_key()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = DataFetchLogger()
        if log_level == "DEBUG":
            self.logger.logger.setLevel(logging.DEBUG)
        elif log_level == "WARNING":
            self.logger.logger.setLevel(logging.WARNING)
        self.validator = DataValidator()
        
        self.cache_metadata_path = self.cache_dir / "_cache_metadata.json"
        self._load_cache_metadata()
        
        if self.fred_api_key is None:
            self.logger.warning("FRED API key missing - FRED unavailable")
            self.logger.info("Get key: https://fred.stlouisfed.org/docs/api/api_key.html")
        else:
            self.logger.debug(f"FRED key: {self.fred_api_key[:8]}...")
    
    def _load_cache_metadata(self):
        """Load cache metadata (creation times, ETags)."""
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
        """Persist cache metadata."""
        try:
            with open(self.cache_metadata_path, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Cache metadata save failed: {e}")
    
    def _update_cache_metadata(self, cache_key: str, etag: str = None):
        """Record cache creation/update."""
        self.cache_metadata[cache_key] = {
            'created': datetime.now().isoformat(),
            'etag': etag
        }
        self._save_cache_metadata()
    
    def _is_cache_stale(self, cache_key: str, ttl_days: int = 90) -> bool:
        """Check if cache exceeds TTL."""
        if cache_key not in self.cache_metadata:
            return True
        
        try:
            created = datetime.fromisoformat(self.cache_metadata[cache_key]['created'])
            age_days = (datetime.now() - created).days
            if age_days > ttl_days:
                self.logger.info(f"Cache stale: {cache_key} ({age_days}d old, TTL={ttl_days}d)")
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Cache age check failed: {e}")
            return True
    
    def _read_fred_api_key(self) -> Optional[str]:
        config_locations = [
            Path(__file__).parent / 'json_data' / 'config.json',
            Path(__file__).parent.parent / 'json_data' / 'config.json',
            Path.cwd() / 'json_data' / 'config.json',
            Path('./json_data/config.json'),
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
    
    def _cache_path(self, name: str, start: str, end: str, cache_type: str = "daily") -> Path:
        return self.cache_dir / f"{name}_{start}_{end}_{cache_type}.parquet"
    
    def _should_use_cache(self, path: Path, is_historical: bool, cache_key: str = None) -> bool:
        """Determine if cache should be used."""
        if not path.exists():
            return False
        
        if is_historical:
            if cache_key and self._is_cache_stale(cache_key, ttl_days=90):
                self.logger.debug(f"Cache expired (90d TTL): {path.name}")
                return False
            return True
        
        return datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date()
    
    def _normalize_data(self, data, name: str) -> Optional[pd.DataFrame]:
        if data is None:
            return None
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        if data.empty:
            self.logger.warning(f"{name}: Empty")
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
            self.logger.debug(f"{name}: Removing {data.index.duplicated().sum()} duplicates")
            data = data.groupby(data.index).last()
        is_valid, msg = self.validator.validate_dataframe(data, name)
        return data if is_valid else (self.logger.warning(msg), None)[1]
    
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
    
    def _apply_lookback_buffer(self, start_date: str, buffer_days: int, name: str) -> str:
        if buffer_days == 0:
            return start_date
        buffered = (pd.to_datetime(start_date) - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        self.logger.debug(f"{name}: Buffer {start_date} → {buffered} (+{buffer_days}d)")
        return buffered
    
    def _check_fred_revision(self, series_id: str, cache_key: str) -> bool:
        """Check if FRED series has been revised since cache."""
        if not self.fred_api_key or cache_key not in self.cache_metadata:
            return False
        
        try:
            url = f'{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&file_type=json&limit=1'
            response = requests.head(url, timeout=5)
            response.raise_for_status()
            
            current_modified = response.headers.get('Last-Modified')
            cached_modified = self.cache_metadata[cache_key].get('etag')
            
            if current_modified and cached_modified and current_modified != cached_modified:
                self.logger.info(f"FRED:{series_id}: Revision detected (Last-Modified changed)")
                return True
            return False
        except Exception as e:
            self.logger.debug(f"FRED:{series_id}: Revision check failed: {e}")
            return False
    
    def fetch_fred(self, series_id: str, start_date: str = None, end_date: str = None,
                   lookback_buffer_days: int = 0) -> Optional[pd.Series]:
        if not self.fred_api_key:
            self.logger.debug(f"FRED:{series_id}: No key")
            return None
        
        original_start = start_date
        if start_date and lookback_buffer_days > 0:
            start_date = self._apply_lookback_buffer(start_date, lookback_buffer_days, f"FRED:{series_id}")
        
        if start_date and end_date and not self.validator.is_business_day_range(start_date, end_date):
            self.logger.error(f"FRED:{series_id}: Invalid range")
            return None
        
        is_historical = self.validator.is_historical_data(end_date) if end_date else False
        cache_path = self._cache_path(f'fred_{series_id}', start_date or 'none', end_date or 'none',
                                      "permanent" if is_historical else "daily")
        cache_key = f"fred_{series_id}_{start_date}_{end_date}"
        
        use_cache = self._should_use_cache(cache_path, is_historical, cache_key)
        if use_cache and is_historical and self._check_fred_revision(series_id, cache_key):
            self.logger.info(f"FRED:{series_id}: Cache invalidated by revision")
            use_cache = False
        
        if use_cache:
            df = self._safe_cache_read(cache_path, f"FRED:{series_id}")
            if df is not None and series_id in df.columns:
                return df[series_id]
        
        try:
            url = f'{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&file_type=json'
            if start_date:
                url += f'&observation_start={start_date}'
            if end_date:
                url += f'&observation_end={end_date}'
            
            self.logger.debug(f"FRED:{series_id}: API fetch")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            last_modified = response.headers.get('Last-Modified')
            
            series_data = {item['date']: float(item['value'])
                          for item in response.json()['observations'] if item['value'] != '.'}
            if not series_data:
                self.logger.warning(f"FRED:{series_id}: No data")
                return None
            
            series = pd.Series(series_data)
            series.index = pd.to_datetime(series.index)
            series.name = series_id
            df = self._normalize_data(pd.DataFrame(series), f'FRED:{series_id}')
            if df is not None:
                self._safe_cache_write(df, cache_path, f"FRED:{series_id}")
                self._update_cache_metadata(cache_key, etag=last_modified)
                return df[series_id]
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"FRED:{series_id}: Network", e)
            return None
        except Exception as e:
            self.logger.error(f"FRED:{series_id}: Error", e)
            return None
    
    def fetch_fred_latest(self, series_id: str) -> Optional[float]:
        if not self.fred_api_key:
            return None
        try:
            url = f'{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&file_type=json&limit=1&sort_order=desc'
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()['observations']
            if data and data[0]['value'] != '.':
                value = float(data[0]['value'])
                self.logger.debug(f"FRED:{series_id}: Latest = {value}")
                return value
        except Exception as e:
            self.logger.error(f"FRED:{series_id}: Latest failed", e)
        return None
    
    def fetch_fred_multiple(self, start_date: str, end_date: str, lookback_buffer_days: int = 0) -> Optional[pd.DataFrame]:
        if not self.fred_api_key:
            self.logger.warning("FRED:multiple: No key")
            return None
        
        original_start = start_date
        if lookback_buffer_days > 0:
            start_date = self._apply_lookback_buffer(start_date, lookback_buffer_days, "FRED:multiple")
        
        is_historical = self.validator.is_historical_data(end_date)
        cache_path = self._cache_path('fred_all', start_date, end_date, "permanent" if is_historical else "daily")
        cache_key = f'fred_all_{start_date}_{end_date}'
        
        if self._should_use_cache(cache_path, is_historical, cache_key):
            df = self._safe_cache_read(cache_path, 'FRED:multiple')
            if df is not None:
                return df
        
        series_list, failed = [], []
        for series_id, name in FRED_SERIES.items():
            s = self.fetch_fred(series_id, start_date, end_date, lookback_buffer_days=0)
            if s is not None:
                s.name = name
                series_list.append(s)
            else:
                failed.append(series_id)
        
        if failed:
            self.logger.warning(f"FRED:multiple: Failed {len(failed)}: {failed}")
        if not series_list:
            self.logger.error("FRED:multiple: All failed")
            return None
        
        df = self._normalize_data(pd.concat(series_list, axis=1, join='outer'), 'FRED:multiple')
        if df is not None:
            self._safe_cache_write(df, cache_path, 'FRED:multiple')
            self._update_cache_metadata(cache_key)
        return df
    
    def fetch_commodities_fred(self, start_date: str, end_date: str, lookback_buffer_days: int = 0) -> Optional[pd.DataFrame]:
        if not self.fred_api_key:
            self.logger.warning("Commodities: No key")
            return None
        
        if lookback_buffer_days > 0:
            start_date = self._apply_lookback_buffer(start_date, lookback_buffer_days, "Commodities")
        
        is_historical = self.validator.is_historical_data(end_date)
        cache_path = self._cache_path('commodities_fred', start_date, end_date, "permanent" if is_historical else "daily")
        cache_key = f'commodities_fred_{start_date}_{end_date}'
        
        if self._should_use_cache(cache_path, is_historical, cache_key):
            df = self._safe_cache_read(cache_path, 'Commodities')
            if df is not None:
                return df
        
        series_list, failed = [], []
        for series_id, name in COMMODITY_FRED_SERIES.items():
            s = self.fetch_fred(series_id, start_date, end_date, lookback_buffer_days=0)
            if s is not None:
                s.name = name
                series_list.append(s)
                self.logger.info(f"Commodities: ✅ {name}")
            else:
                failed.append(series_id)
                self.logger.warning(f"Commodities: ❌ {name}")
        
        for ticker, name in [('GLD', 'Gold'), ('SLV', 'Silver')]:
            try:
                s = self.fetch_yahoo_series(ticker, 'Close', start_date, end_date, lookback_buffer_days=0)
                if s is not None:
                    s.name = name
                    series_list.append(s)
                    self.logger.info(f"Commodities: ✅ {name}")
                else:
                    failed.append(ticker)
            except Exception as e:
                self.logger.error(f"Commodities:{ticker}: Failed", e)
                failed.append(ticker)
        
        if failed:
            self.logger.warning(f"Commodities: Failed {failed}")
        if not series_list:
            self.logger.error("Commodities: All failed")
            return None
        
        df = self._normalize_data(pd.concat(series_list, axis=1, join='outer'), 'Commodities')
        if df is not None:
            self._safe_cache_write(df, cache_path, 'Commodities')
            self._update_cache_metadata(cache_key)
        return df
    
    def fetch_yahoo(self, ticker: str, start_date: str = None, end_date: str = None,
                    lookback_buffer_days: int = 0) -> Optional[pd.DataFrame]:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if lookback_buffer_days > 0:
            start_date = self._apply_lookback_buffer(start_date, lookback_buffer_days, f"Yahoo:{ticker}")
        
        if not self.validator.is_business_day_range(start_date, end_date):
            self.logger.error(f"Yahoo:{ticker}: Invalid range")
            return None
        
        is_historical = self.validator.is_historical_data(end_date)
        cache_path = self._cache_path(f'yahoo_{ticker}', start_date, end_date, "permanent" if is_historical else "daily")
        cache_key = f'yahoo_{ticker}_{start_date}_{end_date}'
        
        if self._should_use_cache(cache_path, is_historical, cache_key):
            df = self._safe_cache_read(cache_path, f'Yahoo:{ticker}')
            if df is not None:
                return df
        
        try:
            self.logger.debug(f"Yahoo:{ticker}: API fetch")
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d')
            if df.empty:
                self.logger.warning(f"Yahoo:{ticker}: Empty")
                return None
            df = self._normalize_data(df, f'Yahoo:{ticker}')
            if df is not None:
                self._safe_cache_write(df, cache_path, f'Yahoo:{ticker}')
                self._update_cache_metadata(cache_key)
            return df
        except Exception as e:
            self.logger.error(f"Yahoo:{ticker}: Failed", e)
            return None
    
    def fetch_yahoo_series(self, ticker: str, column: str, start_date: str = None, end_date: str = None,
                          lookback_buffer_days: int = 0) -> Optional[pd.Series]:
        df = self.fetch_yahoo(ticker, start_date, end_date, lookback_buffer_days)
        if df is None:
            return None
        if column not in df.columns:
            self.logger.error(f"Yahoo:{ticker}: '{column}' missing")
            return None
        series = df[column].squeeze()
        series.name = f"{ticker}_{column}"
        return series
    
    def fetch_price(self, ticker: str) -> Optional[float]:
        try:
            price = float(yf.Ticker(ticker).fast_info["last_price"])
            self.logger.debug(f"Yahoo:{ticker}: Live = {price}")
            return price
        except Exception as e:
            self.logger.error(f"Yahoo:{ticker}: Live failed", e)
            return None
    
    def fetch_spx(self, start_date: str = None, end_date: str = None, lookback_buffer_days: int = 0) -> Optional[pd.DataFrame]:
        return self.fetch_yahoo('^GSPC', start_date, end_date, lookback_buffer_days)
    
    def fetch_vix(self, start_date: str = None, end_date: str = None, lookback_buffer_days: int = 0) -> Optional[pd.Series]:
        vix = self.fetch_yahoo_series('^VIX', 'Close', start_date, end_date, lookback_buffer_days)
        if vix is not None:
            vix.name = 'VIX'
        return vix
    
    def fetch_vix_realtime(self, start_date: str = None, end_date: str = None, lookback_buffer_days: int = 0) -> Optional[pd.Series]:
        vix = self.fetch_vix(start_date, end_date, lookback_buffer_days)
        if vix is None:
            return None
        live_vix = self.fetch_price('^VIX')
        if live_vix is not None:
            vix_rt = vix.copy()
            vix_rt.iloc[-1] = live_vix
            self.logger.info(f"VIX: Historical={vix.iloc[-1]:.2f}, Live={live_vix:.2f}")
            return vix_rt
        return vix
    
    def fetch_macro(self, start_date: str, end_date: str, include_gold_silver_ratio: bool = True,
                    lookback_buffer_days: int = 0) -> Optional[pd.DataFrame]:
        if lookback_buffer_days > 0:
            start_date = self._apply_lookback_buffer(start_date, lookback_buffer_days, "Macro")
        
        cache_key = 'macro_with_ratio' if include_gold_silver_ratio else 'macro'
        is_historical = self.validator.is_historical_data(end_date)
        cache_path = self._cache_path(cache_key, start_date, end_date, "permanent" if is_historical else "daily")
        full_cache_key = f'{cache_key}_{start_date}_{end_date}'
        
        if self._should_use_cache(cache_path, is_historical, full_cache_key):
            df = self._safe_cache_read(cache_path, 'Macro')
            if df is not None:
                return df
        
        series_list, failed = [], []
        for ticker, name in MACRO_TICKERS.items():
            df = self.fetch_yahoo(ticker, start_date, end_date, lookback_buffer_days=0)
            if df is not None and 'Close' in df.columns:
                s = df['Close'].squeeze()
                s.name = name
                series_list.append(s)
            else:
                failed.append(ticker)
        
        if include_gold_silver_ratio:
            gld = self.fetch_yahoo_series('GLD', 'Close', start_date, end_date, lookback_buffer_days=0)
            slv = self.fetch_yahoo_series('SLV', 'Close', start_date, end_date, lookback_buffer_days=0)
            if gld is not None and slv is not None:
                ratio = gld / slv
                ratio.name = 'Gold/Silver Ratio'
                series_list.append(ratio)
            else:
                failed.append('GLD/SLV')
        
        if failed:
            self.logger.warning(f"Macro: Failed {failed}")
        if not series_list:
            self.logger.error("Macro: All failed")
            return None
        
        df = self._normalize_data(pd.concat(series_list, axis=1, join='outer'), 'Macro')
        if df is not None:
            self._safe_cache_write(df, cache_path, 'Macro')
            self._update_cache_metadata(full_cache_key)
        return df


def test_fetcher():
    print(f"\n{'='*70}\nUNIFIED DATA FETCHER V5\n{'='*70}")
    fetcher = UnifiedDataFetcher(log_level="INFO")
    print(f"\n✅ FRED: {'Loaded' if fetcher.fred_api_key else 'NOT FOUND'}")
    if fetcher.fred_api_key:
        print(f"   Key: {fetcher.fred_api_key[:8]}...{fetcher.fred_api_key[-4:]}")
        treasury_10y = fetcher.fetch_fred_latest('DGS10')
        print(f"   {'✅' if treasury_10y else '❌'} 10Y Treasury: {treasury_10y}%" if treasury_10y else "   ❌ FRED fetch failed")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    test_fetcher()