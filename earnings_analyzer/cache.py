"""Caching with thread safety"""
import json
import os
import threading
from typing import Dict, Any

from .config import CACHE_FILE

# Thread lock for safe concurrent access
_cache_lock = threading.Lock()


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
                from datetime import datetime
                backup_name = f"{CACHE_FILE}.corrupted.{int(datetime.now().timestamp())}"
                os.rename(CACHE_FILE, backup_name)
                return {}
        return {}


def save_cache(cache: Dict[str, Any]) -> None:
    """Save earnings data to cache with thread safety and atomic write"""
    with _cache_lock:
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        
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