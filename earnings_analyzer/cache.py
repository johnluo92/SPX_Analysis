"""Caching and rate limit persistence with thread safety"""
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, Set, Any

from .config import CACHE_FILE, RATE_LIMIT_FILE, RATE_LIMIT_HOURS

# Thread locks for safe concurrent access
_cache_lock = threading.Lock()
_rate_limit_lock = threading.Lock()


def load_cache() -> Dict[str, Any]:
    """Load cached earnings data with thread safety"""
    with _cache_lock:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  Cache file corrupted, creating backup and starting fresh")
                # Backup corrupted file
                backup_name = f"{CACHE_FILE}.corrupted.{int(datetime.now().timestamp())}"
                os.rename(CACHE_FILE, backup_name)
                return {}
        return {}


def save_cache(cache: Dict[str, Any]) -> None:
    """Save earnings data to cache with thread safety and atomic write"""
    with _cache_lock:
        # Write to temporary file first (atomic operation)
        temp_file = f"{CACHE_FILE}.tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(cache, f, indent=2, default=str)
            
            # Atomic rename (overwrites existing file)
            os.replace(temp_file, CACHE_FILE)
        except Exception as e:
            # Clean up temp file if something went wrong
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e


def load_rate_limits() -> Set[int]:
    """Load rate limit state with automatic expiry and thread safety"""
    with _rate_limit_lock:
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
    """Persist rate limit state with expiry and thread safety"""
    with _rate_limit_lock:
        reset_time = (datetime.now() + timedelta(hours=RATE_LIMIT_HOURS)).timestamp()
        
        data = {
            str(k): {
                'reset_time': reset_time,
                'limited_at': datetime.now().isoformat()
            } 
            for k in rate_limited_keys
        }
        
        # Atomic write with temp file
        temp_file = f"{RATE_LIMIT_FILE}.tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            os.replace(temp_file, RATE_LIMIT_FILE)
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e