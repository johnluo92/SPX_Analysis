import os
import json
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, List

from config import SECTOR_ETFS, MACRO_TICKERS, FRED_SERIES, FRED_API_KEY_PATH, CACHE_DIR

class UnifiedDataFetcher:
    """
    Unified data fetcher for FRED economic data and Yahoo Finance market data.
    Supports VIX, SPX, sectors, macro, and FRED series with caching.
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.fred_base_url = 'https://api.stlouisfed.org/fred/series/observations'
        self.fred_api_key = self._read_fred_api_key()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _read_fred_api_key(self) -> str:
        """Read FRED API key from config.json."""
        try:
            with open(FRED_API_KEY_PATH, 'r') as f:
                config = json.load(f)
                return config.get('fred_api_key')
        except FileNotFoundError:
            print(f"⚠️  FRED API key not found at {FRED_API_KEY_PATH}")
            print("    FRED data fetching will be disabled.")
            return None
    
    def _cache_path(self, name: str, start: str, end: str) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{name}_{start}_{end}.parquet"
    
    def _is_cached_today(self, path: Path) -> bool:
        """Check if cache file was created today."""
        try:
            if not path.exists():
                return False
            file_date = datetime.fromtimestamp(path.stat().st_mtime).date()
            is_fresh = file_date == datetime.now().date()
            if is_fresh:
                print(f"✓ Using cached: {path.name}")
            return is_fresh
        except Exception as e:
            print(f"⚠️  Cache check failed for {path.name}: {e}")
            return False
    
    # ==================== FRED DATA METHODS ====================
    
    def fetch_fred(self, series_id: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch data from FRED API with caching.
        
        Args:
            series_id: FRED series ID (e.g., 'VIXCLS', 'DFF', 'T10Y2Y')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            pandas Series with date index and values
        """
        if not self.fred_api_key:
            raise ValueError("FRED API key not configured")
        
        # Check cache
        cache_path = self._cache_path(f'fred_{series_id}', start_date or 'none', end_date or 'none')
        if self._is_cached_today(cache_path):
            df = pd.read_parquet(cache_path)
            return df[series_id]
        
        url = f'{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&file_type=json'
        
        if start_date:
            url += f'&observation_start={start_date}'
        if end_date:
            url += f'&observation_end={end_date}'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()['observations']
            
            # Parse data into Series
            series_data = {}
            for item in data:
                try:
                    date = item['date']
                    value = item['value']
                    
                    # Handle missing values marked as '.'
                    if value == '.':
                        continue
                    
                    series_data[date] = float(value)
                except (KeyError, ValueError):
                    continue
            
            series = pd.Series(series_data)
            series.index = pd.to_datetime(series.index)
            series.name = series_id
            
            # Cache it
            df = pd.DataFrame(series)
            df.to_parquet(cache_path)
            
            return series
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch FRED data for {series_id}: {str(e)}")
    
    def fetch_fred_multiple(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch multiple FRED series defined in config."""
        cache_path = self._cache_path('fred_all', start_date, end_date)
        
        if self._is_cached_today(cache_path):
            return pd.read_parquet(cache_path)
        
        if not self.fred_api_key:
            print("⚠️  FRED API key missing, skipping FRED data")
            return pd.DataFrame()
        
        print(f"Fetching FRED: {start_date} to {end_date}")
        
        series_list = []
        
        for series_id, name in FRED_SERIES.items():
            try:
                s = self.fetch_fred(series_id, start_date, end_date)
                s.name = name
                series_list.append(s)
            except Exception as e:
                print(f"Failed {name}: {e}")
        
        if not series_list:
            return pd.DataFrame()
        
        df = pd.concat(series_list, axis=1, join='outer')
        df.index = pd.to_datetime(df.index).normalize()
        
        df.to_parquet(cache_path)
        return df
    
    def fetch_fred_latest(self, series_id: str) -> Optional[float]:
        """
        Fetch the most recent value from a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Latest value as float, or None if unavailable
        """
        if not self.fred_api_key:
            raise ValueError("FRED API key not configured")
        
        url = f'{self.fred_base_url}?series_id={series_id}&api_key={self.fred_api_key}&file_type=json&limit=1&sort_order=desc'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()['observations']
            
            if data and data[0]['value'] != '.':
                return float(data[0]['value'])
            return None
            
        except (requests.exceptions.RequestException, KeyError, ValueError, IndexError):
            return None
    
    # ==================== YAHOO FINANCE METHODS ====================
    
    def fetch_yahoo(self, ticker: str, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None, interval: str = '1d') -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance with caching.
        
        Args:
            ticker: Yahoo Finance ticker (e.g., '^GSPC' for SPX, '^VIX' for VIX)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', '1wk', '1mo')
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        # Check cache
        cache_path = self._cache_path(f'yahoo_{ticker}', start_date or 'none', end_date or 'none')
        if self._is_cached_today(cache_path):
            return pd.read_parquet(cache_path)
        
        try:
            # Set default dates if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch data using yfinance
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            # Clean up the DataFrame
            df.index.name = 'Date'
            df.index = pd.to_datetime(df.index).normalize()
            
            # Cache it
            df.to_parquet(cache_path)
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch Yahoo data for {ticker}: {str(e)}")
    
    def fetch_yahoo_series(self, ticker: str, column: str = 'Close', 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch a single column from Yahoo Finance as a Series.
        
        Args:
            ticker: Yahoo Finance ticker
            column: Column to extract ('Close', 'Open', 'High', 'Low', 'Volume')
            start_date: Start date
            end_date: End date
            
        Returns:
            pandas Series
        """
        df = self.fetch_yahoo(ticker, start_date, end_date)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
        
        series = df[column]
        series.name = f"{ticker}_{column}"
        return series
    
    # ==================== SECTOR & MACRO METHODS ====================
    
    def fetch_sectors(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch sector ETFs including SPY with caching."""
        cache_path = self._cache_path('sectors', start_date, end_date)
        
        if self._is_cached_today(cache_path):
            return pd.read_parquet(cache_path)
        
        print(f"Fetching sectors: {start_date} to {end_date}")
        
        tickers = list(SECTOR_ETFS.keys()) + ['SPY']
        series_list = []
        
        for ticker in tickers:
            try:
                df = self.fetch_yahoo(ticker, start_date, end_date)
                if not df.empty:
                    s = df['Close'].squeeze()
                    s.name = ticker
                    series_list.append(s)
            except Exception as e:
                print(f"Failed {ticker}: {e}")
        
        if not series_list:
            raise ValueError("No sector data")
        
        df = pd.concat(series_list, axis=1, join='outer')
        df.to_parquet(cache_path)
        return df
    
    def fetch_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch macro factors from Yahoo Finance with caching."""
        cache_path = self._cache_path('macro', start_date, end_date)
        
        if self._is_cached_today(cache_path):
            return pd.read_parquet(cache_path)
        
        print(f"Fetching macro: {start_date} to {end_date}")
        
        series_list = []
        
        for ticker, name in MACRO_TICKERS.items():
            try:
                df = self.fetch_yahoo(ticker, start_date, end_date)
                if not df.empty:
                    s = df['Close'].squeeze()
                    s.name = name
                    series_list.append(s)
            except Exception as e:
                print(f"Failed {name}: {e}")
        
        if not series_list:
            raise ValueError("No macro data")
        
        df = pd.concat(series_list, axis=1, join='outer')
        df.to_parquet(cache_path)
        return df
    
    # ==================== CONVENIENCE METHODS FOR OPTIONS TRADING ====================
    
    def fetch_vix(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, source: str = 'yahoo') -> pd.Series:
        """
        Fetch VIX data from either FRED or Yahoo Finance with caching.
        
        Args:
            start_date: Start date
            end_date: End date
            source: 'fred' or 'yahoo'
            
        Returns:
            pandas Series of VIX values
        """
        if source.lower() == 'fred':
            s = self.fetch_fred('VIXCLS', start_date, end_date)
            s.name = 'VIX'
            return s
        else:
            cache_path = self._cache_path('vix', start_date or 'none', end_date or 'none')
            if self._is_cached_today(cache_path):
                df = pd.read_parquet(cache_path)
                return df['VIX']
            
            s = self.fetch_yahoo_series('^VIX', 'Close', start_date, end_date)
            s.name = 'VIX'
            
            df = pd.DataFrame(s)
            df.to_parquet(cache_path)
            return s
    
    def fetch_spx(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch SPX price data (OHLCV) with caching.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        return self.fetch_yahoo('^GSPC', start_date, end_date)
    
    def fetch_spx_close(self, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch SPX closing prices only.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            pandas Series of SPX close prices
        """
        return self.fetch_yahoo_series('^GSPC', 'Close', start_date, end_date)
    
    def fetch_treasury_rate(self, maturity: str = '10Y', 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch Treasury rates from FRED.
        
        Args:
            maturity: '3M', '2Y', '10Y', '30Y'
            start_date: Start date
            end_date: End date
            
        Returns:
            pandas Series of rates
        """
        series_map = {
            '3M': 'DGS3MO',
            '2Y': 'DGS2',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }
        
        series_id = series_map.get(maturity.upper())
        if not series_id:
            raise ValueError(f"Invalid maturity. Choose from: {list(series_map.keys())}")
        
        return self.fetch_fred(series_id, start_date, end_date)
    
    def fetch_fed_funds_rate(self, start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch Federal Funds Rate (effective) from FRED.
        
        Returns:
            pandas Series of daily effective fed funds rate
        """
        return self.fetch_fred('DFF', start_date, end_date)
    
    # ==================== ALIGNMENT METHOD ====================
    
    def align(self, sectors: pd.DataFrame, macro: pd.DataFrame, 
              vix: pd.Series, fred: pd.DataFrame = None) -> tuple:
        """Align all data to common dates."""
        common_dates = sectors.index.intersection(macro.index).intersection(vix.index)
        
        if fred is not None and not fred.empty:
            common_dates = common_dates.intersection(fred.index)
        
        sectors_aligned = sectors.loc[common_dates].ffill().dropna()
        common_dates = sectors_aligned.index
        
        macro_aligned = macro.loc[common_dates].ffill()
        vix_aligned = vix.loc[common_dates].ffill()
        
        if fred is not None and not fred.empty:
            fred_aligned = fred.loc[common_dates].ffill()
            print(f"Aligned: {len(sectors_aligned)} days ({common_dates.min().date()} to {common_dates.max().date()})")
            return sectors_aligned, macro_aligned, vix_aligned, fred_aligned
        
        print(f"Aligned: {len(sectors_aligned)} days ({common_dates.min().date()} to {common_dates.max().date()})")
        return sectors_aligned, macro_aligned, vix_aligned
    
    # ==================== BATCH FETCHING ====================
    
    def fetch_multiple(self, specs: List[Dict]) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Fetch multiple series/tickers in one call.
        
        Args:
            specs: List of dicts with fetch specifications
                   Example: [
                       {'source': 'yahoo', 'ticker': '^GSPC', 'name': 'SPX'},
                       {'source': 'fred', 'series_id': 'VIXCLS', 'name': 'VIX'}
                   ]
        
        Returns:
            Dictionary mapping names to fetched data
        """
        results = {}
        
        for spec in specs:
            name = spec.get('name', spec.get('ticker', spec.get('series_id')))
            source = spec['source'].lower()
            
            try:
                if source == 'yahoo':
                    ticker = spec['ticker']
                    column = spec.get('column', None)
                    start = spec.get('start_date')
                    end = spec.get('end_date')
                    
                    if column:
                        results[name] = self.fetch_yahoo_series(ticker, column, start, end)
                    else:
                        results[name] = self.fetch_yahoo(ticker, start, end)
                        
                elif source == 'fred':
                    series_id = spec['series_id']
                    start = spec.get('start_date')
                    end = spec.get('end_date')
                    
                    results[name] = self.fetch_fred(series_id, start, end)
                    
                else:
                    print(f"⚠️  Unknown source '{source}' for {name}, skipping...")
                    
            except Exception as e:
                print(f"⚠️  Failed to fetch {name}: {str(e)}")
        
        return results
    
    # ==================== UTILITY METHODS ====================
    
    def align_series(self, *series: pd.Series, method: str = 'inner') -> pd.DataFrame:
        """
        Align multiple series to common dates.
        
        Args:
            *series: Variable number of pandas Series
            method: 'inner' (intersection) or 'outer' (union, forward fill)
            
        Returns:
            DataFrame with aligned series
        """
        df = pd.DataFrame({s.name or f'series_{i}': s for i, s in enumerate(series)})
        
        if method == 'inner':
            df = df.dropna()
        elif method == 'outer':
            df = df.fillna(method='ffill')
        
        return df