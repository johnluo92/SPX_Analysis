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
        """
        if not YFINANCE_AVAILABLE:
            return None
        
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
            
            return {
                'iv': round(iv_pct, 1),
                'dte': actual_dte,
                'strike': atm_call['strike'],
                'expiration': target_exp
            }
        
        except Exception:
            if retry_count > 0:
                time.sleep(1)
                return YahooFinanceClient.get_current_iv(ticker, dte_target, retry_count - 1)
            return None