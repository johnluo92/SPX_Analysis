"""Yahoo Finance price and IV data fetching"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from ..config import REQUEST_TIMEOUT, IV_TARGET_DTE
from ..cache import get_cached_iv, cache_iv_data, is_market_hours


class YahooFinanceClient:
    """Client for fetching price data and implied volatility"""
    
    @staticmethod
    def get_price_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical closing prices"""
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {'period1': start_ts, 'period2': end_ts, 'interval': '1d'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            data = response.json()
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            closes = result['indicators']['quote'][0]['close']
            
            df = pd.DataFrame({
                'date': [datetime.fromtimestamp(ts) for ts in timestamps],
                'close': closes
            })
            df.set_index('date', inplace=True)
            df.dropna(inplace=True)
            
            return df
        
        except Exception as e:
            print(f"❌ Error fetching prices for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_current_iv(ticker: str, dte_target: int = IV_TARGET_DTE, retry_count: int = 2) -> Optional[Dict]:
        """
        Fetch current implied volatility from options chain
        Returns ATM IV for expiration closest to target DTE
        
        Now with intelligent caching:
        - Checks cache first
        - Returns cached data if from same trading day
        - Fetches fresh and caches during market hours
        - Uses cache after hours to avoid stale Yahoo data
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        # Check cache first
        cached = get_cached_iv(ticker)
        if cached is not None:
            # Cache is valid (same trading day)
            return cached
        
        # No valid cache - need to fetch fresh
        # Only fetch during market hours to avoid stale data
        market_status = is_market_hours()

        if not market_status['is_open']:
            # After hours - no cache available, return None
            # This prevents using stale Yahoo data
            return None
        
        # Market is open - fetch fresh IV
        try:
            stock = yf.Ticker(ticker)
            
            hist = stock.history(period='1d')
            if hist.empty:
                return None
            current_price = hist['Close'].iloc[-1]
            
            expirations = stock.options
            if not expirations:
                return None
            
            today = datetime.now()
            target_exp = None
            min_diff = 999
            
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                dte = (exp_date - today).days
                if abs(dte - dte_target) < min_diff:
                    min_diff = abs(dte - dte_target)
                    target_exp = exp_str
            
            if not target_exp:
                return None
            
            chain = stock.option_chain(target_exp)
            calls = chain.calls
            
            if calls.empty or 'impliedVolatility' not in calls.columns:
                return None
            
            calls['strike_diff'] = abs(calls['strike'] - current_price)
            atm_idx = calls['strike_diff'].idxmin()
            atm_call = calls.loc[atm_idx]
            
            iv_pct = atm_call['impliedVolatility'] * 100
            actual_dte = (datetime.strptime(target_exp, '%Y-%m-%d') - today).days
            
            iv_data = {
                'iv': round(iv_pct, 1),
                'dte': actual_dte,
                'strike': atm_call['strike'],
                'expiration': target_exp
            }
            
            # Cache the fresh data
            cache_iv_data(ticker, iv_data)
            
            return iv_data
        
        except Exception:
            if retry_count > 0:
                time.sleep(1)
                return YahooFinanceClient.get_current_iv(ticker, dte_target, retry_count - 1)
            return None