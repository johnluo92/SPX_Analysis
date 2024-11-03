import os
import platform
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class OHLCAVFetcher:
    def __init__(self, ticker):
        """
        Initialize the OHLCAV (Open, High, Low, Close, Adjusted Close, Volume) data fetcher
        
        Parameters:
        ticker (str): The stock ticker symbol (e.g., 'SPY' for S&P 500 ETF)
        """
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)
        self.default_fields = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        # Determine base path based on operating system for saving data if needed
        if platform.system() == 'Windows':
            self.base_path = r"C:\Users\jl078\OneDrive\Desktop\WorkSpace\SPX_Analysis\src"
        else:  # macOS
            self.base_path = os.path.expanduser("~/Desktop/GitHub/SPX_Analysis/src")

    def fetch_ohlcav(self, start_date=None, end_date=None, interval='1d'):
        """
        Fetch complete OHLCAV data
        
        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval ('1d', '1wk', '1mo', etc.)
        
        Returns:
        pandas.DataFrame: DataFrame containing all OHLCAV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            df = self.yf_ticker.history(period='max', interval=interval)
        else:
            df = self.yf_ticker.history(start=start_date, end=end_date, interval=interval)

        # Ensure all required columns are present
        for field in self.default_fields:
            if field not in df.columns:
                df[field] = np.nan

        # Clean and format the data
        df = df[self.default_fields]  # Reorder columns to standard format
        df.index = pd.to_datetime(df.index).date  # Convert index to date objects
        return df

    def get_latest_ohlcav(self):
        """
        Get the most recent OHLCAV data point
        
        Returns:
        pandas.Series: Latest OHLCAV values
        """
        start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        df = self.yf_ticker.history(start=start, interval='1d')
        if not df.empty:
            return df[self.default_fields].iloc[-1]
        return None

    def save_to_csv(self, df, filename=None):
        """
        Save OHLCAV data to CSV file
        
        Parameters:
        df (pandas.DataFrame): DataFrame to save
        filename (str): Optional filename, defaults to ticker_OHLCAV.csv
        """
        if filename is None:
            filename = f"{self.ticker}_OHLCAV.csv"
        
        filepath = os.path.join(self.base_path, filename)
        df.to_csv(filepath)
        return filepath

    def load_from_csv(self, filename=None):
        """
        Load OHLCAV data from CSV file
        
        Parameters:
        filename (str): Optional filename, defaults to ticker_OHLCAV.csv
        
        Returns:
        pandas.DataFrame: OHLCAV data
        """
        if filename is None:
            filename = f"{self.ticker}_OHLCAV.csv"
            
        filepath = os.path.join(self.base_path, filename)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).date
        return df

    def calculate_returns(self, df=None, method='close'):
        """
        Calculate returns based on specified price type
        
        Parameters:
        df (pandas.DataFrame): Optional DataFrame to use, otherwise fetches new data
        method (str): 'close' or 'adjusted' for return calculation type
        
        Returns:
        pandas.Series: Daily returns
        """
        if df is None:
            df = self.fetch_ohlcav()
            
        price_col = 'Adj Close' if method == 'adjusted' else 'Close'
        returns = df[price_col].pct_change()
        return returns

    def calculate_volatility(self, df=None, window=20):
        """
        Calculate rolling volatility
        
        Parameters:
        df (pandas.DataFrame): Optional DataFrame to use, otherwise fetches new data
        window (int): Rolling window for volatility calculation
        
        Returns:
        pandas.Series: Rolling volatility
        """
        if df is None:
            df = self.fetch_ohlcav()
            
        returns = self.calculate_returns(df, method='adjusted')
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility

    def get_trading_summaries(self, df=None):
        """
        Get summary statistics for the trading data
        
        Parameters:
        df (pandas.DataFrame): Optional DataFrame to use, otherwise fetches new data
        
        Returns:
        dict: Dictionary containing various trading summaries
        """
        if df is None:
            df = self.fetch_ohlcav()
            
        summaries = {
            'daily_ranges': df['High'] - df['Low'],
            'daily_returns': self.calculate_returns(df),
            'volume_ma': df['Volume'].rolling(window=20).mean(),
            'price_ma': df['Close'].rolling(window=20).mean(),
            'volatility': self.calculate_volatility(df)
        }
        return summaries

    def to_period(self, df=None, period='W'):
        """
        Convert daily data to other periods (weekly, monthly, etc.)
        
        Parameters:
        df (pandas.DataFrame): Optional DataFrame to use, otherwise fetches new data
        period (str): 'W' for weekly, 'M' for monthly, etc.
        
        Returns:
        pandas.DataFrame: Resampled OHLCAV data
        """
        if df is None:
            df = self.fetch_ohlcav()
            
        resampled = pd.DataFrame({
            'Open': df['Open'].resample(period).first(),
            'High': df['High'].resample(period).max(),
            'Low': df['Low'].resample(period).min(),
            'Close': df['Close'].resample(period).last(),
            'Adj Close': df['Adj Close'].resample(period).last(),
            'Volume': df['Volume'].resample(period).sum()
        })
        return resampled