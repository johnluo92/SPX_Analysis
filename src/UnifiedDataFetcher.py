import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, List

class UnifiedDataFetcher:
    """
    Unified data fetcher for FRED economic data and Yahoo Finance market data.
    Supports VIX, SPX, and any FRED series for options backtesting.
    """
    
    def __init__(self, fred_api_key_path='fred_api_key.txt'):
        """
        Initialize the unified fetcher.
        
        Args:
            fred_api_key_path: Path to FRED API key file
        """
        self.fred_base_url = 'https://api.stlouisfed.org/fred/series/observations'
        self.fred_api_key = self._read_fred_api_key(fred_api_key_path)
        
    def _read_fred_api_key(self, api_key_path: str) -> str:
        """Read FRED API key from file."""
        try:
            expanded_path = os.path.expanduser(api_key_path)
            with open(expanded_path, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"⚠️  FRED API key not found at {api_key_path}")
            print("    FRED data fetching will be disabled.")
            return None
    
    # ==================== FRED DATA METHODS ====================
    
    def fetch_fred(self, series_id: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch data from FRED API.
        
        Args:
            series_id: FRED series ID (e.g., 'VIXCLS', 'DFF', 'T10Y2Y')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            pandas Series with date index and values
        """
        if not self.fred_api_key:
            raise ValueError("FRED API key not configured")
        
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
            
            return series
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch FRED data for {series_id}: {str(e)}")
    
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
        Fetch data from Yahoo Finance.
        
        Args:
            ticker: Yahoo Finance ticker (e.g., '^GSPC' for SPX, '^VIX' for VIX)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', '1wk', '1mo')
            
        Returns:
            pandas DataFrame with OHLCV data
        """
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
    
    # ==================== CONVENIENCE METHODS FOR OPTIONS TRADING ====================
    
    def fetch_vix(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, source: str = 'yahoo') -> pd.Series:
        """
        Fetch VIX data from either FRED or Yahoo Finance.
        
        Args:
            start_date: Start date
            end_date: End date
            source: 'fred' or 'yahoo'
            
        Returns:
            pandas Series of VIX values
        """
        if source.lower() == 'fred':
            return self.fetch_fred('VIXCLS', start_date, end_date)
        else:
            return self.fetch_yahoo_series('^VIX', 'Close', start_date, end_date)
    
    def fetch_spx(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch SPX price data (OHLCV).
        
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


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    # Initialize fetcher
    fetcher = UnifiedDataFetcher()
    
    print("=" * 60)
    print("UNIFIED DATA FETCHER - EXAMPLES")
    print("=" * 60)
    
    # Example 1: Fetch SPX and VIX for backtesting
    print("\n📊 Example 1: Fetch SPX and VIX (last 2 years)")
    start = '2023-01-01'
    
    spx = fetcher.fetch_spx_close(start_date=start)
    vix = fetcher.fetch_vix(start_date=start, source='yahoo')
    
    print(f"SPX: {len(spx)} observations")
    print(f"VIX: {len(vix)} observations")
    print(f"Latest SPX: ${spx.iloc[-1]:.2f}")
    print(f"Latest VIX: {vix.iloc[-1]:.2f}")
    
    # Example 2: Fetch economic indicators
    print("\n📈 Example 2: Fetch Treasury rates and Fed Funds")
    
    treasury_10y = fetcher.fetch_treasury_rate('10Y', start_date=start)
    fed_funds = fetcher.fetch_fed_funds_rate(start_date=start)
    
    print(f"10Y Treasury: {len(treasury_10y)} observations")
    print(f"Fed Funds: {len(fed_funds)} observations")
    
    # Example 3: Batch fetch
    print("\n🔄 Example 3: Batch fetch multiple series")
    
    specs = [
        {'source': 'yahoo', 'ticker': '^GSPC', 'column': 'Close', 'name': 'SPX'},
        {'source': 'yahoo', 'ticker': '^VIX', 'column': 'Close', 'name': 'VIX'},
        {'source': 'fred', 'series_id': 'DFF', 'name': 'FedFunds'},
        {'source': 'fred', 'series_id': 'DGS10', 'name': 'Treasury10Y'},
    ]
    
    data = fetcher.fetch_multiple(specs)
    
    for name, series in data.items():
        print(f"{name}: {len(series)} observations")
    
    # Example 4: Align series for analysis
    print("\n🔗 Example 4: Align series to common dates")
    
    aligned = fetcher.align_series(
        data['SPX'],
        data['VIX'],
        data['FedFunds'],
        method='inner'
    )
    
    print(f"Aligned DataFrame: {len(aligned)} rows x {len(aligned.columns)} columns")
    print(f"Date range: {aligned.index[0].date()} to {aligned.index[-1].date()}")
    print("\nFirst 5 rows:")
    print(aligned.head())
    
    print("\n✅ All examples completed successfully!")