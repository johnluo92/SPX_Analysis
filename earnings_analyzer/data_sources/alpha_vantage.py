"""Alpha Vantage earnings data fetching"""
import requests
import time
from datetime import datetime
from typing import List, Dict, Tuple

from ..config import ALPHAVANTAGE_KEYS, REQUEST_TIMEOUT
from ..cache import load_cache, save_cache, load_rate_limits, save_rate_limits


class AlphaVantageClient:
    """Client for fetching earnings data from Alpha Vantage"""
    
    def __init__(self):
        self.current_key_index = 0
        self.rate_limited_keys = load_rate_limits()
    
    def get_earnings(self, ticker: str, use_cache: bool = True, debug: bool = False) -> Tuple[List[Dict], str]:
        """
        Get earnings announcement dates for a ticker
        Returns: (earnings_list, status)
        """
        # Always reload cache to get latest (important for parallel processing)
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
                
                # Reload cache again before saving (in case another thread updated it)
                cache = load_cache()
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