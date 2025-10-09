"""Caching and rate limit persistence"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Set, Any

from .config import CACHE_FILE, RATE_LIMIT_FILE, RATE_LIMIT_HOURS


def load_cache() -> Dict[str, Any]:
    """Load cached earnings data"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    """Save earnings data to cache"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


def load_rate_limits() -> Set[int]:
    """Load rate limit state with automatic expiry"""
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