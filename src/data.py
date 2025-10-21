"""
Data Fetcher - Clean & Simple
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import SECTOR_ETFS, MACRO_TICKERS, CACHE_DIR


class DataFetcher:
    """Fetch and align sector and macro data."""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _cache_path(self, name: str, start: str, end: str) -> Path:
        """Simple cache filename."""
        return self.cache_dir / f"{name}_{start}_{end}.parquet"
    
    def _is_cached_today(self, path: Path) -> bool:
        """Check if cache file was created today."""
        if not path.exists():
            return False
        file_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        return file_date == datetime.now().date()
    
    def fetch_sectors(self, start: str, end: str) -> pd.DataFrame:
        """Fetch sector ETFs including SPY."""
        cache_path = self._cache_path('sectors', start, end)
        
        if self._is_cached_today(cache_path):
            return pd.read_parquet(cache_path)
        
        print(f"Fetching sectors: {start} to {end}")
        
        tickers = list(SECTOR_ETFS.keys()) + ['SPY']
        series_list = []
        
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if not df.empty:
                    if 'Close' in df.columns:
                        s = df['Close'].squeeze()
                    else:
                        s = df.iloc[:, 0].squeeze()
                    s.name = ticker
                    series_list.append(s)
            except Exception as e:
                print(f"Failed {ticker}: {e}")
        
        if not series_list:
            raise ValueError("No sector data")
        
        df = pd.concat(series_list, axis=1, join='outer')
        df.index = pd.to_datetime(df.index).normalize()
        
        df.to_parquet(cache_path)
        return df
    
    def fetch_macro(self, start: str, end: str) -> pd.DataFrame:
        """Fetch macro factors from Yahoo Finance."""
        cache_path = self._cache_path('macro', start, end)
        
        if self._is_cached_today(cache_path):
            return pd.read_parquet(cache_path)
        
        print(f"Fetching macro: {start} to {end}")
        
        series_list = []
        
        for ticker, name in MACRO_TICKERS.items():
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if not df.empty:
                    if 'Close' in df.columns:
                        s = df['Close'].squeeze()
                    else:
                        s = df.iloc[:, 0].squeeze()
                    s.name = name
                    series_list.append(s)
            except Exception as e:
                print(f"Failed {name}: {e}")
        
        if not series_list:
            raise ValueError("No macro data")
        
        df = pd.concat(series_list, axis=1, join='outer')
        df.index = pd.to_datetime(df.index).normalize()
        
        df.to_parquet(cache_path)
        return df
    
    def fetch_vix(self, start: str, end: str) -> pd.Series:
        """Fetch VIX from Yahoo Finance."""
        cache_path = self._cache_path('vix', start, end)
        
        if self._is_cached_today(cache_path):
            df = pd.read_parquet(cache_path)
            return df['VIX']
        
        print(f"Fetching VIX: {start} to {end}")
        
        df = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=True)
        vix = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 0].squeeze()
        vix.index = pd.to_datetime(vix.index).normalize()
        vix.name = 'VIX'
        
        # Save as DataFrame
        vix_df = pd.DataFrame(vix)
        vix_df.to_parquet(cache_path)
        return vix
    
    def align(self, sectors: pd.DataFrame, macro: pd.DataFrame, 
              vix: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Align all data to common dates."""
        common_dates = sectors.index.intersection(macro.index).intersection(vix.index)
        
        sectors_aligned = sectors.loc[common_dates].ffill().dropna()
        common_dates = sectors_aligned.index
        
        macro_aligned = macro.loc[common_dates].ffill()
        vix_aligned = vix.loc[common_dates].ffill()
        
        print(f"Aligned: {len(sectors_aligned)} days ({common_dates.min().date()} to {common_dates.max().date()})")
        
        return sectors_aligned, macro_aligned, vix_aligned