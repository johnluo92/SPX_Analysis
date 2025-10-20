"""YFinance earnings data fetching"""
from datetime import datetime, timezone
from typing import List, Dict, Tuple

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from ..cache import load_cache, save_cache


class YFinanceEarningsClient:
    """Client for fetching earnings data from Yahoo Finance"""

    def __init__(self):
        pass

    def get_earnings(self, ticker: str, use_cache: bool = True, debug: bool = False) -> Tuple[List[Dict], str]:
        """
        Get earnings announcement dates for a ticker from yfinance
        Returns: (earnings_list, status)
        """
        if not YFINANCE_AVAILABLE:
            if debug:
                print(f"  {ticker}: yfinance not available")
            return [], "yfinance_unavailable"

        # Always reload cache to get latest (important for parallel processing)
        cache = load_cache()

        if use_cache and ticker in cache:
            try:
                earnings = []
                for e in cache[ticker]:
                    d = datetime.fromisoformat(e['date'])
                    if d.tzinfo is not None:
                        d = d.astimezone(timezone.utc).replace(tzinfo=None)
                    earnings.append({'date': d, 'time': e['time']})
                return earnings, "cached"
            except Exception as e:
                if debug:
                    print(f"  {ticker}: Cache parsing failed: {e}")

        try:
            stock = yf.Ticker(ticker)
            earnings_dates = stock.get_earnings_dates(limit=100)

            if earnings_dates is None or earnings_dates.empty:
                if debug:
                    print(f"  {ticker}: No earnings data available")
                return [], "no_earnings"

            earnings_info = self._parse_earnings(earnings_dates)

            if not earnings_info:
                if debug:
                    print(f"  {ticker}: No valid earnings data after parsing")
                return [], "no_earnings"

            # Ensure timezone consistency before saving
            for e in earnings_info:
                if e['date'].tzinfo is not None:
                    e['date'] = e['date'].astimezone(timezone.utc).replace(tzinfo=None)

            cache = load_cache()
            cache[ticker] = [
                {'date': e['date'].isoformat(), 'time': e['time']}
                for e in earnings_info
            ]
            save_cache(cache)

            # Sort safely (normalize all just in case)
            sorted_info = sorted(
                earnings_info,
                key=lambda x: x['date'].astimezone(timezone.utc).replace(tzinfo=None)
                if x['date'].tzinfo is not None else x['date'],
                reverse=True
            )

            return sorted_info, "success"

        except Exception as e:
            if debug:
                print(f"  {ticker}: Exception during earnings fetch: {e}")
            return [], "exception"

    @staticmethod
    def _parse_earnings(earnings_df) -> List[Dict]:
        """Parse earnings dataframe from yfinance"""
        earnings_info = []

        for earnings_date, _ in earnings_df.iterrows():
            if hasattr(earnings_date, 'to_pydatetime'):
                date_obj = earnings_date.to_pydatetime()
            else:
                date_obj = earnings_date

            # Convert to UTC and remove tzinfo
            if date_obj.tzinfo is not None:
                date_obj = date_obj.astimezone(timezone.utc).replace(tzinfo=None)

            # Determine timing
            hour = date_obj.hour
            if hour < 10:
                timing = 'bmo'
            elif hour >= 16:
                timing = 'amc'
            else:
                timing = 'amc'

            # Normalize date
            normalized_date = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
            earnings_info.append({'date': normalized_date, 'time': timing})

        return earnings_info
