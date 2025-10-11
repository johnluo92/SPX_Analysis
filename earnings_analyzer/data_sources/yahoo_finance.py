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
MAX_DTE_FOR_IV = 45  # Use expirations within 45 days


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
        """
        # Check if market is open first
        if not YahooFinanceClient._is_market_open(ticker):
            return False
        
        now = YahooFinanceClient._get_market_time()
        
        # Don't fetch before 10 AM or after 5 PM ET
        if now.hour < 10 or now.hour >= 17:
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
    def _get_nearest_expirations(expirations: List[str], max_count: int = 3) -> List[str]:
        """
        Get expirations ≤45 DTE, up to max_count
        If none ≤45 DTE exist, fall back to single nearest expiration
        
        Examples:
            [7d, 14d, 21d] → use all 3
            [14d, 30d, 46d] → use first 2 (14d, 30d)
            [45d, 70d, 90d] → use only 45d
            [60d, 90d, 120d] → use only 60d (fallback)
        
        Args:
            expirations: List of expiration date strings
            max_count: Maximum number of expirations to return
        
        Returns:
            List of up to max_count expiration dates (≤45 DTE preferred)
        """
        today = datetime.now()
        
        # Filter to expirations ≤45 DTE
        valid_exps = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_date - today).days
            if 0 < dte <= MAX_DTE_FOR_IV:
                valid_exps.append((exp, dte))
        
        # If we have expirations ≤45 DTE, use up to max_count
        if valid_exps:
            valid_exps.sort(key=lambda x: x[1])  # Sort by DTE
            return [exp for exp, _ in valid_exps[:max_count]]
        
        # Fallback: if NO expirations ≤45 DTE, use single nearest expiration
        all_exps = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            dte = abs((exp_date - today).days)
            all_exps.append((exp, dte))
        
        all_exps.sort(key=lambda x: x[1])
        return [all_exps[0][0]] if all_exps else []
    
    @staticmethod
    def get_current_iv(ticker: str, retry_count: int = 2) -> Optional[Dict]:
        """
        Fetch current implied volatility from expirations ≤45 DTE (up to 3)
        Uses time-gated caching (10am, 12pm, 2pm, 3:30pm ET windows)
        Returns ATM IV averaged across up to 6 options (up to 3 puts + 3 calls)
        
        If no expirations ≤45 DTE, falls back to single nearest expiration
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
            
            # Get up to 3 nearest expirations (prefer within 30 DTE)
            nearest_exps = YahooFinanceClient._get_nearest_expirations(expirations, max_count=3)
            
            if not nearest_exps:
                return cache.get(ticker)
            
            # Collect IVs from all expirations
            all_ivs = []
            exps_used = []
            
            for exp in nearest_exps:
                try:
                    chain = stock.option_chain(exp)
                    calls = chain.calls
                    puts = chain.puts
                    
                    if calls.empty or puts.empty:
                        continue
                    
                    if 'impliedVolatility' not in calls.columns or 'impliedVolatility' not in puts.columns:
                        continue
                    
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
                    
                    all_ivs.extend([call_iv, put_iv])
                    exps_used.append(exp)
                
                except:
                    continue
            
            # If we didn't get any valid IVs, return cache
            if not all_ivs:
                return cache.get(ticker)
            
            # Calculate average IV
            avg_iv = sum(all_ivs) / len(all_ivs)
            
            # Calculate DTE of nearest expiration
            nearest_exp = exps_used[0]
            exp_date = datetime.strptime(nearest_exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            
            # Create cache entry
            iv_data = {
                'iv': round(avg_iv, 1),
                'dte': dte,
                'expiration': nearest_exp,
                'expirations_used': exps_used,
                'fetched_at': YahooFinanceClient._get_market_time().isoformat()
            }
            
            # Update cache
            cache[ticker] = iv_data
            YahooFinanceClient._save_iv_cache(cache)
            
            return iv_data
        
        except Exception:
            if retry_count > 0:
                time.sleep(1)
                return YahooFinanceClient.get_current_iv(ticker, retry_count - 1)
            # Return cached data if fetch fails
            return cache.get(ticker)