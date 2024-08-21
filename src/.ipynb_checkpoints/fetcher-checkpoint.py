import os
import requests
import pandas as pd

class SeriesFetcher:
    def __init__(self, series_id, api_key_path='~/Desktop/GitHub/SPX_Analysis/src/fred_api_key.txt'):
        self.series_id = series_id
        self.base_url = 'https://api.stlouisfed.org/fred/series/observations'
        self.api_key_file = os.path.expanduser(api_key_path)
        self.api_key = self._read_api_key(self.api_key_file)

    def _read_api_key(self, api_key_file):
        with open(api_key_file, 'r') as file:
            api_key = file.read().strip()
        return api_key

    def fetch_series(self, start_date=None, fetch_last=False):
        url = f'{self.base_url}?series_id={self.series_id}&api_key={self.api_key}&file_type=json'
        if fetch_last:
            url += '&limit=1&sort_order=desc'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['observations']

        if fetch_last:
            return float(data[0]['value']) if data else None

        series_data = {}
        for item in data:
            try:
                date = item['date']
                value = float(item['value'])
                if start_date and pd.to_datetime(date) < pd.to_datetime(start_date):
                    continue
                series_data[date] = value
            except (KeyError, ValueError):
                pass  # Skip over invalid values
            
        return pd.Series(series_data)

    def get_series_data(self, start_date=None):
        return self.fetch_series(start_date=start_date)

    def get_latest_value(self):
        return self.fetch_series(fetch_last=True)
