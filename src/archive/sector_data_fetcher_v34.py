"""
Sector Rotation Data Fetcher v3.4 - Simplified & Clean

CHANGES FROM v2.1:
✅ Yahoo Finance fixed treasury yields - no scaling needed
✅ All macro factors from Yahoo Finance (FRED still disabled)
✅ Clean validation and error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from UnifiedDataFetcher import UnifiedDataFetcher


class SectorDataFetcher:
    """
    Fetches sector ETF and macro factor data with strict alignment.
    
    MACRO FACTORS (ALL YAHOO FINANCE):
    - GLD: Gold (inflation hedge, safe haven)
    - CL=F: WTI Crude Oil Futures (energy costs, inflation)
    - DX-Y.NYB: Dollar Index (currency strength, trade)
    - ^TNX: 10-Year Treasury Yield (risk-free rate, growth expectations) [FIXED: /10]
    - ^FVX: 5-Year Treasury Yield (for yield curve calculation) [FIXED: /10]
    """
    
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Health Care',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services',
        'XLB': 'Materials'
    }
    
    MACRO_TICKERS = {
        'GLD': 'Gold',
        'CL=F': 'Crude Oil',
        'DX-Y.NYB': 'Dollar',
        '^TNX': '10Y Treasury',  # Will be divided by 10
        '^FVX': '5Y Treasury',   # Will be divided by 10
    }
    
    def __init__(self, cache_dir: str = '.cache_sector_data', config_path: str = 'config.json'):
        """Initialize fetcher with caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load FRED API key from config.json (kept for future use)
        fred_api_key_path = None
        config_file = Path(config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    fred_api_key = config.get('fred_api_key')
                    
                    if fred_api_key and fred_api_key != "YOUR_FRED_API_KEY_HERE":
                        temp_key_file = self.cache_dir / '.fred_api_key_temp.txt'
                        with open(temp_key_file, 'w') as key_file:
                            key_file.write(fred_api_key)
                        fred_api_key_path = str(temp_key_file)
                        print(f"✅ Loaded FRED API key from {config_path} (DISABLED)")
                    else:
                        print(f"⚠️  FRED API key not set in {config_path}")
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {e}")
        
        # Initialize UnifiedDataFetcher (kept for future use)
        if fred_api_key_path:
            self.unified = UnifiedDataFetcher(fred_api_key_path=fred_api_key_path)
        else:
            self.unified = UnifiedDataFetcher(fred_api_key_path='fred_api_key.txt')
        
        print("ℹ️  Using Yahoo Finance for all macro factors")
    
    def _get_cache_path(self, dataset_name: str) -> Path:
        """Get cache file path for a dataset."""
        return self.cache_dir / f"{dataset_name}.parquet"
    
    def _get_cache_metadata_path(self, dataset_name: str) -> Path:
        """Get cache metadata path."""
        return self.cache_dir / f"{dataset_name}_metadata.json"
    
    def _is_cache_valid(self, dataset_name: str) -> bool:
        """Check if cached data is valid (fetched today)."""
        metadata_path = self._get_cache_metadata_path(dataset_name)
        
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cache_date = datetime.fromisoformat(metadata['fetch_date']).date()
            today = datetime.now().date()
            
            return cache_date == today
        except Exception as e:
            print(f"⚠️  Cache metadata error for {dataset_name}: {e}")
            return False
    
    def _save_to_cache(self, data: pd.DataFrame, dataset_name: str):
        """Save data to cache with metadata."""
        cache_path = self._get_cache_path(dataset_name)
        metadata_path = self._get_cache_metadata_path(dataset_name)
        
        data.to_parquet(cache_path)
        
        ticker_hash = hash(frozenset(self.MACRO_TICKERS.items()))
        
        metadata = {
            'fetch_date': datetime.now().isoformat(),
            'rows': len(data),
            'start_date': str(data.index.min()),
            'end_date': str(data.index.max()),
            'columns': list(data.columns),
            'ticker_config_hash': ticker_hash
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Cached {dataset_name}: {len(data)} rows, {len(data.columns)} columns")
    
    def _load_from_cache(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        if not self._is_cache_valid(dataset_name):
            return None
        
        cache_path = self._get_cache_path(dataset_name)
        
        try:
            data = pd.read_parquet(cache_path)
            print(f"   ✅ Loaded {dataset_name} from cache ({len(data)} rows, {len(data.columns)} cols)")
            return data
        except Exception as e:
            print(f"   ⚠️  Cache load error for {dataset_name}: {e}")
            return None
    
    def fetch_sector_etfs(self, 
                          start_date: str, 
                          end_date: str,
                          include_spy: bool = True) -> pd.DataFrame:
        """Fetch all SPDR sector ETF data."""
        dataset_name = f"sectors_{start_date}_{end_date}"
        
        cached = self._load_from_cache(dataset_name)
        if cached is not None:
            return cached
        
        print(f"\n📊 Fetching sector ETF data...")
        print(f"   Period: {start_date} to {end_date}")
        
        tickers = list(self.SECTOR_ETFS.keys())
        if include_spy:
            tickers.append('SPY')
        
        all_data = {}
        
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    progress=False,
                    auto_adjust=True
                )
                
                if not data.empty:
                    if 'Close' in data.columns:
                        all_data[ticker] = data['Close']
                    elif isinstance(data, pd.Series):
                        all_data[ticker] = data
                    else:
                        all_data[ticker] = data.iloc[:, 0]
                    
                    print(f"   ✅ {ticker}: {len(data)} days")
                else:
                    print(f"   ❌ {ticker}: No data")
                    
            except Exception as e:
                print(f"   ❌ {ticker}: {e}")
        
        if not all_data:
            raise ValueError("No sector ETF data could be fetched")
        
        df = pd.concat(all_data, axis=1)
        df.columns = all_data.keys()
        
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        self._save_to_cache(df, dataset_name)
        
        print(f"\n   ✅ Fetched {len(df.columns)} sector ETFs")
        print(f"   📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        return df
    
    def fetch_macro_factors(self,
                           start_date: str,
                           end_date: str) -> pd.DataFrame:
        """
        Fetch macro factor data - ALL FROM YAHOO FINANCE.
        
        UPDATE v3.4: Yahoo Finance fixed treasury yields!
        ^TNX and ^FVX now return actual % (no /10 needed)
        """
        dataset_name = f"macro_{start_date}_{end_date}_v34"  # New cache name
        
        cached = self._load_from_cache(dataset_name)
        if cached is not None:
            print(f"\n📊 Macro factors loaded from cache (v3.4):")
            self._print_macro_summary(cached)
            return cached
        
        print(f"\n📊 Fetching macro factor data (Yahoo Finance)...")
        
        all_data = {}
        failed = []
        
        for ticker, name in self.MACRO_TICKERS.items():
            try:
                print(f"   Fetching {name} ({ticker})...", end=" ")
                
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                if not data.empty:
                    if 'Close' in data.columns:
                        series = data['Close']
                    elif isinstance(data, pd.Series):
                        series = data
                    else:
                        series = data.iloc[:, 0]
                    
                    # Yahoo Finance now returns treasury yields correctly
                    # No scaling needed!
                    print(f"✅ {len(data)} days")
                    
                    all_data[name] = series
                else:
                    print(f"❌ No data")
                    failed.append(f"{name} ({ticker})")
                    
            except Exception as e:
                print(f"❌ {e}")
                failed.append(f"{name} ({ticker})")
        
        if not all_data:
            raise ValueError("No macro factor data could be fetched")
        
        # Combine
        df = pd.concat(all_data, axis=1)
        df.columns = all_data.keys()
        
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Save to cache
        self._save_to_cache(df, dataset_name)
        
        print(f"\n   ✅ Fetched {len(df.columns)} macro factors")
        print(f"   📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        if failed:
            print(f"\n   ⚠️  FAILED TO FETCH: {', '.join(failed)}")
        
        # Print summary with validation
        self._print_macro_summary(df)
        
        return df
    
    def _print_macro_summary(self, macro_df: pd.DataFrame):
        """Print detailed summary with data validation."""
        print(f"\n📊 MACRO FACTOR SUMMARY:")
        print(f"   Columns: {list(macro_df.columns)}")
        print(f"   Date range: {macro_df.index.min().date()} to {macro_df.index.max().date()}")
        print(f"   Total observations: {len(macro_df)}")
        
        # VALIDATION: Check treasury yield ranges
        print(f"\n   🔍 TREASURY YIELD VALIDATION:")
        if '10Y Treasury' in macro_df.columns:
            ten_y = macro_df['10Y Treasury']
            print(f"      10Y Treasury - Current: {ten_y.iloc[-1]:.2f}%")
            print(f"      10Y Treasury - Mean: {ten_y.mean():.2f}%")
            print(f"      10Y Treasury - Range: {ten_y.min():.2f}% to {ten_y.max():.2f}%")
            
            if ten_y.mean() > 10:
                print(f"      ⚠️  WARNING: 10Y yields look wrong (expected 1-5%)")
            else:
                print(f"      ✅ 10Y yields in expected range")
        
        if '5Y Treasury' in macro_df.columns:
            five_y = macro_df['5Y Treasury']
            print(f"      5Y Treasury - Current: {five_y.iloc[-1]:.2f}%")
            print(f"      5Y Treasury - Mean: {five_y.mean():.2f}%")
            print(f"      5Y Treasury - Range: {five_y.min():.2f}% to {five_y.max():.2f}%")
            
            if five_y.mean() > 10:
                print(f"      ⚠️  WARNING: 5Y yields look wrong (expected 1-5%)")
            else:
                print(f"      ✅ 5Y yields in expected range")
        
        # Missing values
        missing = macro_df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n   ⚠️  Missing values:")
            for col, miss in missing[missing > 0].items():
                print(f"      {col}: {miss} ({miss/len(macro_df)*100:.1f}%)")
        
        # Correlation matrix
        print(f"\n   📊 21-Day Return Correlations:")
        returns = macro_df.pct_change(21).dropna()
        corr = returns.corr()
        print(corr.round(2).to_string())
        
        # Flag high correlations
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.6:
                    high_corr.append(
                        f"{corr.columns[i]} ↔ {corr.columns[j]}: {corr.iloc[i, j]:.2f}"
                    )
        
        if high_corr:
            print(f"\n   ⚠️  HIGH CORRELATIONS (>0.6):")
            for h in high_corr:
                print(f"      {h}")
    
    def align_data(self, 
                   sectors: pd.DataFrame,
                   macro: pd.DataFrame,
                   vix: pd.Series = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """Align all datasets to common dates."""
        print("\n🔧 Aligning data (enforcing homoscedasticity)...")
        
        # Normalize indices
        sectors.index = pd.to_datetime(sectors.index.date)
        macro.index = pd.to_datetime(macro.index.date)
        
        if vix is not None:
            vix.index = pd.to_datetime(vix.index.date)
        
        # Find common dates
        common_dates = sectors.index.intersection(macro.index)
        
        if vix is not None:
            common_dates = common_dates.intersection(vix.index)
        
        # Align
        sectors_aligned = sectors.loc[common_dates].sort_index()
        macro_aligned = macro.loc[common_dates].sort_index()
        
        if vix is not None:
            vix_aligned = vix.loc[common_dates].sort_index()
        else:
            vix_aligned = None
        
        print(f"   ✅ Aligned to {len(common_dates)} common dates")
        print(f"   📅 Range: {common_dates.min().date()} to {common_dates.max().date()}")
        
        # Check missing values
        sectors_missing = sectors_aligned.isnull().sum().sum()
        macro_missing = macro_aligned.isnull().sum().sum()
        
        if sectors_missing > 0:
            print(f"   ⚠️  Sectors: {sectors_missing} missing values")
        if macro_missing > 0:
            print(f"   ⚠️  Macro: {macro_missing} missing values")
        
        # Forward fill
        if sectors_missing > 0 or macro_missing > 0:
            print("   🔧 Forward-filling missing values...")
            sectors_aligned = sectors_aligned.ffill()
            macro_aligned = macro_aligned.ffill()
            
            initial_len = len(sectors_aligned)
            sectors_aligned = sectors_aligned.dropna()
            macro_aligned = macro_aligned.loc[sectors_aligned.index]
            
            if vix_aligned is not None:
                vix_aligned = vix_aligned.loc[sectors_aligned.index]
            
            dropped = initial_len - len(sectors_aligned)
            if dropped > 0:
                print(f"   🗑️  Dropped {dropped} rows with unrecoverable NaNs")
        
        print(f"   ✅ Final aligned dataset: {len(sectors_aligned)} rows")
        
        # Homoscedasticity check
        if len(sectors_aligned) != len(macro_aligned):
            raise ValueError("HOMOSCEDASTICITY VIOLATED: Mismatched data lengths")
        
        if not sectors_aligned.index.equals(macro_aligned.index):
            raise ValueError("HOMOSCEDASTICITY VIOLATED: Mismatched indices")
        
        if vix_aligned is not None:
            if len(vix_aligned) != len(sectors_aligned):
                raise ValueError("HOMOSCEDASTICITY VIOLATED: VIX length mismatch")
            if not vix_aligned.index.equals(sectors_aligned.index):
                raise ValueError("HOMOSCEDASTICITY VIOLATED: VIX index mismatch")
        
        print("   ✅ HOMOSCEDASTICITY VERIFIED")
        
        return sectors_aligned, macro_aligned, vix_aligned