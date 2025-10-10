"""Yahoo Finance price and IV data fetching"""
import requests
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from ..config import REQUEST_TIMEOUT, IV_TARGET_DTE

IV_CACHE_FILE = "cache/iv_cache.json"
IV_CACHE_HOURS = 24  # Cache IV for 24 hours


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
    def _is_market_hours() -> bool:
        """Check if currently during US market hours (rough check)"""
        now = datetime.now()
        # Monday-Friday, 9:30 AM - 4:00 PM ET (simplified check)
        if now.weekday() >= 5:  # Weekend
            return False
        hour = now.hour
        # Very rough: assuming you're in US timezone for now
        # Better: use pytz for proper timezone handling
        return 9 <= hour <= 16
    
    @staticmethod
    def get_current_iv(ticker: str, dte_target: int = IV_TARGET_DTE, retry_count: int = 2) -> Optional[Dict]:
        """
        Fetch current implied volatility from options chain with caching
        
        Strategy:
        1. Check cache first
        2. If cache is fresh (<24h old), return cached value
        3. If cache is stale AND during market hours, fetch new data
        4. If cache is stale AND after hours, keep using stale cache
        5. Never overwrite good cache with None/empty data
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        # Load cache
        iv_cache = YahooFinanceClient._load_iv_cache()
        
        # Check if we have cached data for this ticker
        if ticker in iv_cache:
            cached_data = iv_cache[ticker]
            
            # Backward compatibility: handle old cache format without fetched_at
            if 'fetched_at' not in cached_data:
                # Old cache format - treat as valid but will be updated with timestamp on next fetch
                # For now, use it if after hours, otherwise let it fetch fresh
                if not YahooFinanceClient._is_market_hours():
                    return {
                        'iv': cached_data['iv'],
                        'dte': cached_data['dte'],
                        'strike': cached_data['strike'],
                        'expiration': cached_data['expiration']
                    }
                # Market hours - fall through to fetch new data with proper timestamp
            else:
                # New cache format with timestamp
                cached_time = datetime.fromisoformat(cached_data['fetched_at'])
                age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                
                # If cache is fresh, use it
                if age_hours < IV_CACHE_HOURS:
                    return {
                        'iv': cached_data['iv'],
                        'dte': cached_data['dte'],
                        'strike': cached_data['strike'],
                        'expiration': cached_data['expiration']
                    }
                
                # Cache is stale - only fetch if during market hours
                if not YahooFinanceClient._is_market_hours():
                    # After hours: keep using stale cache rather than getting bad data
                    return {
                        'iv': cached_data['iv'],
                        'dte': cached_data['dte'],
                        'strike': cached_data['strike'],
                        'expiration': cached_data['expiration']
                    }
        
        # No cache or cache stale during market hours - fetch new data
        try:
            stock = yf.Ticker(ticker)
            
            hist = stock.history(period='1d')
            if hist.empty:
                # Failed to fetch - return cached data if available
                if ticker in iv_cache:
                    cached = iv_cache[ticker]
                    return {
                        'iv': cached['iv'],
                        'dte': cached['dte'],
                        'strike': cached['strike'],
                        'expiration': cached['expiration']
                    }
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            expirations = stock.options
            if not expirations:
                # No options data - return cached if available
                if ticker in iv_cache:
                    cached = iv_cache[ticker]
                    return {
                        'iv': cached['iv'],
                        'dte': cached['dte'],
                        'strike': cached['strike'],
                        'expiration': cached['expiration']
                    }
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
                # No suitable expiration - return cached if available
                if ticker in iv_cache:
                    cached = iv_cache[ticker]
                    return {
                        'iv': cached['iv'],
                        'dte': cached['dte'],
                        'strike': cached['strike'],
                        'expiration': cached['expiration']
                    }
                return None
            
            chain = stock.option_chain(target_exp)
            calls = chain.calls
            
            if calls.empty or 'impliedVolatility' not in calls.columns:
                # Empty chain - return cached if available
                if ticker in iv_cache:
                    cached = iv_cache[ticker]
                    return {
                        'iv': cached['iv'],
                        'dte': cached['dte'],
                        'strike': cached['strike'],
                        'expiration': cached['expiration']
                    }
                return None
            
            calls['strike_diff'] = abs(calls['strike'] - current_price)
            atm_idx = calls['strike_diff'].idxmin()
            atm_call = calls.loc[atm_idx]
            
            iv_pct = atm_call['impliedVolatility'] * 100
            actual_dte = (datetime.strptime(target_exp, '%Y-%m-%d') - today).days
            
            # Successfully fetched - update cache
            result = {
                'iv': round(iv_pct, 1),
                'dte': actual_dte,
                'strike': atm_call['strike'],
                'expiration': target_exp
            }
            
            iv_cache[ticker] = {
                **result,
                'fetched_at': datetime.now().isoformat(),
                'market_date': datetime.now().strftime('%Y-%m-%d')
            }
            YahooFinanceClient._save_iv_cache(iv_cache)
            
            return result
        
        except Exception:
            # Exception during fetch - return cached data if available
            if ticker in iv_cache:
                cached = iv_cache[ticker]
                return {
                    'iv': cached['iv'],
                    'dte': cached['dte'],
                    'strike': cached['strike'],
                    'expiration': cached['expiration']
                }
            
            # Retry if allowed
            if retry_count > 0:
                time.sleep(1)
                return YahooFinanceClient.get_current_iv(ticker, dte_target, retry_count - 1)
            
            return None