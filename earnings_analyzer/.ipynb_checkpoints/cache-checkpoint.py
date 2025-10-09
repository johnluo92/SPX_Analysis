"""Caching and rate limit persistence"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Set, Any, Optional

from .config import CACHE_FILE, RATE_LIMIT_FILE, IV_CACHE_FILE, RATE_LIMIT_HOURS


def _ensure_cache_directory():
    """Ensure cache directory exists"""
    cache_dir = os.path.dirname(CACHE_FILE)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


def load_cache() -> Dict[str, Any]:
    """Load cached earnings data"""
    _ensure_cache_directory()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    """Save earnings data to cache"""
    _ensure_cache_directory()
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


def load_rate_limits() -> Set[int]:
    """Load rate limit state with automatic expiry"""
    _ensure_cache_directory()
    if not os.path.exists(RATE_LIMIT_FILE):
        return set()
    
    try:
        with open(RATE_LIMIT_FILE, 'r') as f:
            data = json.load(f)
        
        now = datetime.now().timestamp()
        active_limits = {}
        
        for key_idx, info in data.items():
            reset_time = info.get('reset_time', 0)
            if reset_time > now:
                active_limits[int(key_idx)] = info
        
        return set(active_limits.keys())
    except:
        return set()


def save_rate_limits(rate_limited_keys: Set[int]) -> None:
    """Persist rate limit state with expiry"""
    _ensure_cache_directory()
    reset_time = (datetime.now() + timedelta(hours=RATE_LIMIT_HOURS)).timestamp()
    
    data = {
        str(k): {
            'reset_time': reset_time,
            'limited_at': datetime.now().isoformat()
        } 
        for k in rate_limited_keys
    }
    
    with open(RATE_LIMIT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_iv_cache() -> Dict[str, Any]:
    """Load IV cache with same-day validation"""
    _ensure_cache_directory()
    if not os.path.exists(IV_CACHE_FILE):
        return {}
    
    try:
        with open(IV_CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        today = datetime.now().date().isoformat()
        valid_cache = {}
        
        for ticker, data in cache.items():
            # Support both 'date' and 'market_date' field names
            cache_date = data.get('date') or data.get('market_date')
            if cache_date == today:
                valid_cache[ticker] = data
        
        return valid_cache
    except:
        return {}


def save_iv_cache(cache: Dict[str, Any]) -> None:
    """Save IV cache with timestamp"""
    _ensure_cache_directory()
    with open(IV_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_cached_iv(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get cached IV data for a ticker if valid (same trading day)
    Returns None if no valid cache exists
    """
    cache = load_iv_cache()
    return cache.get(ticker)


def cache_iv_data(ticker: str, iv_data: Dict[str, Any]) -> None:
    """
    Cache IV data for a ticker with current date
    """
    cache = load_iv_cache()
    
    # Add date field for validation
    iv_data['date'] = datetime.now().date().isoformat()
    iv_data['market_date'] = datetime.now().date().isoformat()  # Keep both for compatibility
    
    cache[ticker] = iv_data
    save_iv_cache(cache)


def is_market_hours() -> Dict[str, Any]:
    """
    Check if US stock market is currently open and return status info
    Returns dict with: is_open, time_str, day_name
    """
    import pytz
    
    try:
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        # Weekend check
        is_weekend = now_et.weekday() >= 5  # Saturday = 5, Sunday = 6
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_open = not is_weekend and (market_open <= now_et <= market_close)
        
        return {
            'is_open': is_open,
            'time_str': now_et.strftime('%H:%M'),
            'day_name': now_et.strftime('%A')
        }
    except:
        # If pytz not available, assume market closed (safe default)
        now = datetime.now()
        return {
            'is_open': False,
            'time_str': now.strftime('%H:%M'),
            'day_name': now.strftime('%A')
        }