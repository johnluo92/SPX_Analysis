"""
Sector Rotation Data Fetcher v2.0
IMPROVEMENTS:
- Replaced USO with CL=F (WTI futures)
- Replaced UUP with DX-Y.NYB (actual DXY)
- Added TLT (long bond proxy for duration)
- Better diagnostics for macro factor loading
- Correlation matrix output
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
    
    IMPROVED MACRO FACTORS:
    - GLD: Gold (inflation hedge, safe haven)
    - CL=F: WTI Crude Oil Futures (energy costs, inflation)
    - DX-Y.NYB: Dollar Index (currency strength, trade)
    - TLT: 20Y Treasury Bond ETF (duration, flight-to-quality)
    - DGS10, DGS2: Treasury yields via FRED (term structure)
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
    'GLD': 'Gold',           # Inflation hedge, safe haven
    'CL=F': 'Crude Oil',     # Energy costs, inflation
    'DX-Y.NYB': 'Dollar',    # Currency strength, trade
    }
    
    def __init__(self, cache_dir: str = '.cache_sector_data', config_path: str = 'config.json'):
        """Initialize fetcher with caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load FRED API key from config.json
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
                        print(f"✅ Loaded FRED API key from {config_path}")
                    else:
                        print(f"⚠️  FRED API key not set in {config_path}")
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {e}")
        else:
            print(f"⚠️  Config file not found: {config_path}")
        
        # Initialize UnifiedDataFetcher
        if fred_api_key_path:
            self.unified = UnifiedDataFetcher(fred_api_key_path=fred_api_key_path)
        else:
            self.unified = UnifiedDataFetcher(fred_api_key_path='fred_api_key.txt')
        
        if self.unified.fred_api_key:
            print("✅ FRED API connected - Treasury yield data available")
        else:
            print("⚠️  FRED API key not configured")
    
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
        
        # Include ticker config for cache validation
        ticker_hash = hash(frozenset(self.MACRO_TICKERS.items()))
        
        metadata = {
            'fetch_date': datetime.now().isoformat(),
            'rows': len(data),
            'start_date': str(data.index.min()),
            'end_date': str(data.index.max()),
            'columns': list(data.columns),
            'ticker_config_hash': ticker_hash  # Invalidate if tickers change
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
        Fetch macro factor data with IMPROVED DIAGNOSTICS.
        """
        dataset_name = f"macro_{start_date}_{end_date}"
        
        cached = self._load_from_cache(dataset_name)
        if cached is not None:
            print(f"\n📊 Macro factors loaded from cache:")
            self._print_macro_summary(cached)
            return cached
        
        print(f"\n📊 Fetching macro factor data...")
        
        all_data = {}
        failed = []
        
        # Fetch ETF-based factors
        for ticker, name in self.MACRO_TICKERS.items():
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
                        all_data[name] = data['Close']
                    elif isinstance(data, pd.Series):
                        all_data[name] = data
                    else:
                        all_data[name] = data.iloc[:, 0]
                    
                    print(f"   ✅ {name} ({ticker}): {len(data)} days")
                else:
                    print(f"   ❌ {name} ({ticker}): No data")
                    failed.append(f"{name} ({ticker})")
                    
            except Exception as e:
                print(f"   ❌ {name} ({ticker}): {e}")
                failed.append(f"{name} ({ticker})")
        
        # Fetch FRED data (Treasury yields)
        if self.unified.fred_api_key:
            treasury_series = {
                'DGS10': '10Y Treasury',
                'DGS2': '2Y Treasury',
            }
            
            for series_id, name in treasury_series.items():
                try:
                    data = self.unified.fetch_fred(
                        series_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if len(data) > 0:
                        all_data[name] = data
                        print(f"   ✅ {name}: {len(data)} days")
                    else:
                        print(f"   ❌ {name}: No data")
                        failed.append(name)
                        
                except Exception as e:
                    print(f"   ❌ {name}: {e}")
                    failed.append(name)
        
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
        
        # Print summary statistics
        self._print_macro_summary(df)
        
        return df
    
    def _print_macro_summary(self, macro_df: pd.DataFrame):
        """Print detailed summary of macro factors."""
        print(f"\n📊 MACRO FACTOR SUMMARY:")
        print(f"   Columns: {list(macro_df.columns)}")
        print(f"   Date range: {macro_df.index.min().date()} to {macro_df.index.max().date()}")
        print(f"   Total observations: {len(macro_df)}")
        
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
        
        # Print correlation matrix with formatting
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
        else:
            print(f"\n   ✅ All factors well-diversified (correlations <0.6)")
    
    def align_data(self, 
                   sectors: pd.DataFrame,
                   macro: pd.DataFrame,
                   vix: pd.Series = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """Align all datasets to common dates (CRITICAL FOR HOMOSCEDASTICITY)."""
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
    
    def generate_alignment_report(self,
                                  sectors: pd.DataFrame,
                                  macro: pd.DataFrame,
                                  vix: pd.Series = None) -> Dict:
        """Generate comprehensive alignment report."""
        report = {
            'sectors': {
                'count': len(sectors),
                'start': str(sectors.index.min().date()),
                'end': str(sectors.index.max().date()),
                'missing': int(sectors.isnull().sum().sum()),
                'columns': list(sectors.columns)
            },
            'macro': {
                'count': len(macro),
                'start': str(macro.index.min().date()),
                'end': str(macro.index.max().date()),
                'missing': int(macro.isnull().sum().sum()),
                'columns': list(macro.columns)
            }
        }
        
        if vix is not None:
            report['vix'] = {
                'count': len(vix),
                'start': str(vix.index.min().date()),
                'end': str(vix.index.max().date()),
                'missing': int(vix.isnull().sum())
            }
        
        report['alignment'] = {
            'homoscedastic': (
                len(sectors) == len(macro) and 
                sectors.index.equals(macro.index)
            ),
            'common_dates': len(sectors.index.intersection(macro.index)),
            'sectors_only': len(sectors.index.difference(macro.index)),
            'macro_only': len(macro.index.difference(sectors.index))
        }
        
        return report