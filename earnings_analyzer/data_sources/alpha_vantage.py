"""Alpha Vantage earnings data fetching"""
import requests
import time
import csv
import io
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from ..config import ALPHAVANTAGE_KEYS, REQUEST_TIMEOUT
from ..cache import load_cache, save_cache, load_rate_limits, save_rate_limits


class AlphaVantageClient:
    """Client for fetching earnings data from Alpha Vantage"""
    
    def __init__(self):
        self.current_key_index = 0
        self.rate_limited_keys = load_rate_limits()
        # Store upcoming cache in Cache/ folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.upcoming_cache_file = os.path.join(project_root, "Cache", "earnings_upcoming_cache.json")
    
    def get_earnings(self, ticker: str, use_cache: bool = True, debug: bool = False) -> Tuple[List[Dict], str]:
        """
        Get earnings announcement dates for a ticker
        Returns: (earnings_list, status)
        """
        cache = load_cache()
        if use_cache and ticker in cache:
            return [
                {'date': datetime.fromisoformat(e['date']), 'time': e['time']} 
                for e in cache[ticker]
            ], "cached"
        
        if len(self.rate_limited_keys) >= len(ALPHAVANTAGE_KEYS):
            if debug:
                print(f"  {ticker}: All API keys exhausted")
            return [], "rate_limited_all"
        
        max_attempts = len(ALPHAVANTAGE_KEYS) - len(self.rate_limited_keys)
        
        for attempt in range(max_attempts):
            while self.current_key_index in self.rate_limited_keys:
                self.current_key_index = (self.current_key_index + 1) % len(ALPHAVANTAGE_KEYS)
            
            current_key = ALPHAVANTAGE_KEYS[self.current_key_index]
            url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={current_key}"
            
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                data = response.json()
                
                error_msg = data.get('Note', data.get('Information', ''))
                if error_msg:
                    if any(phrase in error_msg.lower() for phrase in ['rate limit', 'call frequency']):
                        self.rate_limited_keys.add(self.current_key_index)
                        save_rate_limits(self.rate_limited_keys)
                        
                        self.current_key_index = (self.current_key_index + 1) % len(ALPHAVANTAGE_KEYS)
                        
                        if len(self.rate_limited_keys) >= len(ALPHAVANTAGE_KEYS):
                            return [], "rate_limited_all"
                        
                        time.sleep(1)
                        continue
                    
                    return [], "api_error"
                
                if 'quarterlyEarnings' not in data:
                    return [], "no_earnings"
                
                earnings_info = self._parse_earnings(data['quarterlyEarnings'])
                
                cache[ticker] = [
                    {'date': e['date'].isoformat(), 'time': e['time']} 
                    for e in earnings_info
                ]
                save_cache(cache)
                
                return sorted(earnings_info, key=lambda x: x['date'], reverse=True), "success"
            
            except Exception:
                if attempt < max_attempts - 1:
                    self.current_key_index = (self.current_key_index + 1) % len(ALPHAVANTAGE_KEYS)
                    continue
                return [], "exception"
        
        return [], "unknown_error"
    
    def get_upcoming_earnings(self, ticker: str, use_cache: bool = True, debug: bool = False) -> Tuple[Optional[Dict], str]:
        """
        Get next upcoming earnings date for a ticker
        Returns: ({'date': 'YYYY-MM-DD'}, status) or (None, status)
        """
        today = datetime.now().date()
        
        # Load upcoming cache
        upcoming_cache = {}
        if os.path.exists(self.upcoming_cache_file):
            try:
                with open(self.upcoming_cache_file, 'r') as f:
                    upcoming_cache = json.load(f)
            except:
                upcoming_cache = {}
        
        # Check cache validity
        if use_cache and ticker in upcoming_cache:
            cached_date_str = upcoming_cache[ticker].get('date')
            if cached_date_str:
                try:
                    cached_date = datetime.strptime(cached_date_str, '%Y-%m-%d').date()
                    # Only use cache if the date is still in the future
                    if cached_date >= today:
                        return upcoming_cache[ticker], "cached"
                except:
                    pass
        
        # Need to fetch fresh data
        if len(self.rate_limited_keys) >= len(ALPHAVANTAGE_KEYS):
            if debug:
                print(f"  {ticker}: All API keys exhausted for upcoming earnings")
            return None, "rate_limited_all"
        
        max_attempts = len(ALPHAVANTAGE_KEYS) - len(self.rate_limited_keys)
        
        for attempt in range(max_attempts):
            while self.current_key_index in self.rate_limited_keys:
                self.current_key_index = (self.current_key_index + 1) % len(ALPHAVANTAGE_KEYS)
            
            current_key = ALPHAVANTAGE_KEYS[self.current_key_index]
            url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={ticker}&horizon=3month&apikey={current_key}"
            
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                
                # Check for rate limit in response text
                if 'rate limit' in response.text.lower() or 'call frequency' in response.text.lower():
                    self.rate_limited_keys.add(self.current_key_index)
                    save_rate_limits(self.rate_limited_keys)
                    
                    self.current_key_index = (self.current_key_index + 1) % len(ALPHAVANTAGE_KEYS)
                    
                    if len(self.rate_limited_keys) >= len(ALPHAVANTAGE_KEYS):
                        return None, "rate_limited_all"
                    
                    time.sleep(1)
                    continue
                
                # Parse CSV
                csv_reader = csv.DictReader(io.StringIO(response.text))
                upcoming_date = None
                
                for row in csv_reader:
                    report_date_str = row.get('reportDate', '').strip()
                    if report_date_str:
                        try:
                            report_date = datetime.strptime(report_date_str, '%Y-%m-%d').date()
                            if report_date >= today:
                                upcoming_date = report_date_str
                                break
                        except:
                            continue
                
                if not upcoming_date:
                    return None, "no_upcoming"
                
                # Store only the date (no timing)
                result = {'date': upcoming_date}
                
                # Update cache
                upcoming_cache[ticker] = result
                os.makedirs(os.path.dirname(self.upcoming_cache_file), exist_ok=True)
                with open(self.upcoming_cache_file, 'w') as f:
                    json.dump(upcoming_cache, f, indent=2)
                
                return result, "success"
            
            except Exception as e:
                if debug:
                    print(f"  Error fetching upcoming earnings for {ticker}: {e}")
                if attempt < max_attempts - 1:
                    self.current_key_index = (self.current_key_index + 1) % len(ALPHAVANTAGE_KEYS)
                    continue
                return None, "exception"
        
        return None, "unknown_error"
    
    @staticmethod
    def _parse_earnings(quarterly_data: List[Dict]) -> List[Dict]:
        """Parse raw earnings data from API"""
        earnings_info = []
        
        for quarter in quarterly_data:
            reported_date = quarter.get('reportedDate')
            reported_time = quarter.get('reportTime') or quarter.get('reportedTime', 'amc')
            
            if reported_date:
                time_map = {'pre-market': 'bmo', 'post-market': 'amc'}
                normalized_time = time_map.get(reported_time.lower(), reported_time.lower())
                
                earnings_info.append({
                    'date': datetime.strptime(reported_date, '%Y-%m-%d'),
                    'time': normalized_time
                })
        
        return earnings_info