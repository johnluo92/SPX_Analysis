"""Yahoo Finance price and IV data fetching"""
import requests
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from zoneinfo import ZoneInfo

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from ..config import REQUEST_TIMEOUT

# IV cache settings
IV_CACHE_FILE = "cache/iv_cache.json"
FETCH_WINDOWS = [
    (10, 0),   # 10:00 AM ET
    (12, 0),   # 12:00 PM ET
    (14, 0),   # 2:00 PM ET
    (15, 30),  # 3:30 PM ET
]


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
    def _load_iv_cache() -> Dict:
        """Load IV cache from file"""
        if os.path.exists(IV_CACHE_FILE):
            try:
                with open(IV_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    @staticmethod
    def _save_iv_cache(cache: Dict) -> None:
        """Save IV cache to file"""
        os.makedirs(os.path.dirname(IV_CACHE_FILE), exist_ok=True)
        with open(IV_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    
    @staticmethod
    def _get_market_time() -> datetime:
        """Get current time in ET timezone"""
        return datetime.now(ZoneInfo("America/New_York"))
    
    @staticmethod
    def _is_market_open(ticker: str) -> bool:
        """Check if market is currently open using yfinance marketState"""
        try:
            stock = yf.Ticker(ticker)
            market_state = stock.info.get("marketState")
            return market_state == "REGULAR"
        except:
            return False
    
    @staticmethod
    def _should_fetch_iv(ticker: str, cache: Dict) -> bool:
        """
        Determine if we should fetch new IV data based on fetch windows
        
        Fetch windows: 10am, 12pm, 2pm, 3:30pm ET
        Only fetch if we've crossed a window threshold since last fetch
        
        Fetching allowed between 10 AM - 10 PM ET
        """
        now = YahooFinanceClient._get_market_time()
        
        # Don't fetch before 10 AM or after 10 PM ET
        if now.hour < 10 or now.hour >= 22:
            return False
        
        # If no cache entry, fetch
        if ticker not in cache:
            return True
        
        try:
            last_fetch = datetime.fromisoformat(cache[ticker]['fetched_at'])
            
            # If last fetch was more than 24 hours ago, fetch
            if (now - last_fetch).total_seconds() > 86400:
                return True
            
            # If last fetch was today, check if we've crossed a window
            if last_fetch.date() == now.date():
                last_hour = last_fetch.hour
                last_minute = last_fetch.minute
                
                for window_hour, window_minute in FETCH_WINDOWS:
                    # Current time is past this window
                    current_past_window = (now.hour > window_hour or 
                                          (now.hour == window_hour and now.minute >= window_minute))
                    
                    # Last fetch was before this window
                    last_before_window = (last_hour < window_hour or 
                                         (last_hour == window_hour and last_minute < window_minute))
                    
                    if current_past_window and last_before_window:
                        return True
                
                return False
            else:
                # Last fetch was a previous day, fetch new data
                return True
        
        except:
            # If any error parsing, fetch fresh data
            return True
    
    @staticmethod
    def _get_nearest_liquid_expiration(expirations: List[str]) -> Optional[str]:
        """
        Get the nearest liquid expiration (typically 15-60 DTE)
        
        Strategy: Find first expiration with reasonable DTE (positive days, preferably 15+)
        This maximizes success rate by not forcing specific DTE requirements
        
        Args:
            expirations: List of expiration date strings
        
        Returns:
            First reasonable expiration date string, or None if no expirations
        """
        if not expirations:
            return None
        
        today = datetime.now()
        
        # Look for expirations in the 15-60 DTE range (most liquid)
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                dte = (exp_date - today).days
                
                # Prefer 15-60 DTE range for liquidity
                if 15 <= dte <= 60:
                    return exp
            except:
                continue
        
        # Fallback: if nothing in 15-60 range, just get first valid positive DTE
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                dte = (exp_date - today).days
                
                if dte > 0:
                    return exp
            except:
                continue
        
        return None
    
    @staticmethod
    def get_current_iv(ticker: str, retry_count: int = 2) -> Optional[Dict]:
        """
        Fetch current implied volatility from nearest liquid expiration
        Uses time-gated caching (10am, 12pm, 2pm, 3:30pm ET windows)
        Returns ATM IV averaged from put and call
        
        Note: DTE will vary by ticker based on available expirations
        The actual DTE used is returned in the result for transparency
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        # Load cache
        cache = YahooFinanceClient._load_iv_cache()
        
        # Check if we should fetch
        if not YahooFinanceClient._should_fetch_iv(ticker, cache):
            # Return cached data
            return cache.get(ticker)
        
        # Fetch new IV data
        try:
            stock = yf.Ticker(ticker)
            
            # Get current price
            hist = stock.history(period='1d')
            if hist.empty:
                return cache.get(ticker)
            current_price = hist['Close'].iloc[-1]
            
            # Get all expirations
            expirations = stock.options
            if not expirations:
                return cache.get(ticker)
            
            # Get nearest liquid expiration (no specific DTE requirement)
            target_exp = YahooFinanceClient._get_nearest_liquid_expiration(expirations)
            
            if not target_exp:
                return cache.get(ticker)
            
            # Get option chain for target expiration
            try:
                chain = stock.option_chain(target_exp)
                calls = chain.calls
                puts = chain.puts
                
                if calls.empty or puts.empty:
                    return cache.get(ticker)
                
                if 'impliedVolatility' not in calls.columns or 'impliedVolatility' not in puts.columns:
                    return cache.get(ticker)
                
                # Find ATM strikes
                calls['strike_diff'] = abs(calls['strike'] - current_price)
                puts['strike_diff'] = abs(puts['strike'] - current_price)
                
                atm_call_idx = calls['strike_diff'].idxmin()
                atm_put_idx = puts['strike_diff'].idxmin()
                
                atm_call = calls.loc[atm_call_idx]
                atm_put = puts.loc[atm_put_idx]
                
                # Get IVs
                call_iv = atm_call['impliedVolatility'] * 100
                put_iv = atm_put['impliedVolatility'] * 100
                
                # Average call and put IV
                avg_iv = (call_iv + put_iv) / 2
                
                # Calculate actual DTE
                exp_date = datetime.strptime(target_exp, '%Y-%m-%d')
                dte = (exp_date - datetime.now()).days
                
                # Create cache entry
                iv_data = {
                    'iv': round(avg_iv, 1),
                    'dte': dte,
                    'expiration': target_exp,
                    'fetched_at': YahooFinanceClient._get_market_time().isoformat()
                }
                
                # Update cache
                cache[ticker] = iv_data
                YahooFinanceClient._save_iv_cache(cache)
                
                return iv_data
            
            except Exception:
                return cache.get(ticker)
        
        except Exception:
            if retry_count > 0:
                time.sleep(1)
                return YahooFinanceClient.get_current_iv(ticker, retry_count - 1)
            # Return cached data if fetch fails
            return cache.get(ticker)