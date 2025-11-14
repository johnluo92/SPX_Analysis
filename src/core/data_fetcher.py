import json
import logging
import os
import warnings
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")


class DataFetchLogger:
    def __init__(self, name: str = "DataFetcher"):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()
        self.logger.propagate = False
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


class UnifiedDataFetcher:
    QUARTERLY_SERIES = {"GDP", "GDPC1"}
    MONTHLY_SERIES = {
        "UNRATE",
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "PCEPILFE",
        "UMCSENT",
        "INDPRO",
        "PAYEMS",
        "M1SL",
        "M2SL",
        "WALCL",
        "WTREGEN",
    }
    DAILY_SERIES = {
        "DGS1MO",
        "DGS3MO",
        "DGS6MO",
        "DGS1",
        "DGS2",
        "DGS5",
        "DGS10",
        "DGS30",
        "DFF",
        "SOFR",
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
        "VIXCLS",
        "T10Y2Y",
        "T10Y3M",
        "T5YIE",
        "T10YIE",
    }

    FRED_SERIES_GROUPS = {
        "TREASURIES": {
            "DGS1MO": "1M_Treasury",
            "DGS3MO": "3M_Treasury",
            "DGS6MO": "6M_Treasury",
            "DGS1": "1Y_Treasury",
            "DGS2": "2Y_Treasury",
            "DGS5": "5Y_Treasury",
            "DGS10": "10Y_Treasury",
            "DGS30": "30Y_Treasury",
        },
        "TREASURY_SPREADS": {
            "T10Y2Y": "Yield_Curve_10Y2Y",
            "T10Y3M": "Yield_Curve_10Y3M",
            "T5YIE": "Breakeven_Inflation_5Y",
            "T10YIE": "Breakeven_Inflation_10Y",
        },
        "FED_RATES": {"DFF": "Fed_Funds_Effective", "SOFR": "SOFR"},
        "CREDIT_SPREADS": {
            "BAMLH0A0HYM2": "High_Yield_OAS",
            "BAMLC0A0CM": "Corporate_Master_OAS",
        },
        "VOLATILITY": {"VIXCLS": "VIX_Close"},
        "ECONOMIC": {
            "UNRATE": "Unemployment_Rate",
            "CPIAUCSL": "CPI",
            "CPILFESL": "Core_CPI",
            "PCEPI": "PCE_Price_Index",
            "PCEPILFE": "Core_PCE",
            "GDP": "GDP",
            "GDPC1": "Real_GDP",
            "UMCSENT": "Consumer_Sentiment",
        },
    }

    CBOE_FILES = {
        "SKEW": "SKEW_INDEX_CBOE.csv",
        "PCCI": "PCCI_INDX_CBOE.csv",
        "PCCE": "PCCE_EQUITIES_CBOE.csv",
        "PCC": "PCC_INDX_EQ_TOTAL_CBOE.csv",
        "COR1M": "COR1M_CBOE.csv",
        "COR3M": "COR3M_CBOE.csv",
        "VXTH": "VXTH_TAILHEDGE_CBOE.csv",
        "CNDR": "CNDR_SPX_IRON_CONDOR_CBOE.csv",
        "VX1-VX2": "VX1-VX2.csv",
        "VX2-VX1_RATIO": "(VX2-VX1)OVER(VX1).csv",
        "CL1-CL2": "CL1-CL2.csv",
        "DX1-DX2": "DX1-DX2.csv",
        "BVOL": "BVOL.csv",
        "BFLY": "BFLY.csv",
        "DSPX": "DSPX.csv",
        "VPN": "VPN.csv",
        "GAMMA": "GAMMA.csv",
        "VIX_VIX3M": "VIXoverVIX3M.csv",
        "VXTLT": "VXTLT.csv",
        "GOLDSILVER": "GOLDSILVER_RATIO.csv",
    }

    def __init__(
        self,
        cache_dir: str = "./data_cache",
        cboe_data_dir: str = "./CBOE_Data_Archive",
    ):
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cboe_data_dir = Path(cboe_data_dir)
        self.logger = DataFetchLogger()
        if not self.fred_api_key:
            self.logger.warning("FRED API key missing")

    def _get_last_date_from_cache(self, cache_path: Path) -> Optional[str]:
        try:
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    return df.index[-1].strftime("%Y-%m-%d")
        except:
            pass
        return None

    def _merge_with_cache(
        self, new_data: pd.DataFrame, cache_path: Path
    ) -> pd.DataFrame:
        try:
            if cache_path.exists() and new_data is not None and not new_data.empty:
                cached_df = pd.read_parquet(cache_path)
                if not cached_df.empty:
                    combined = pd.concat([cached_df, new_data])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    return combined.sort_index()
        except Exception as e:
            self.logger.warning(f"Cache merge failed: {e}")
        return new_data

    def _normalize_data(self, data, name: str) -> Optional[pd.DataFrame]:
        if data is None or (hasattr(data, "empty") and data.empty):
            return None
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                self.logger.error(f"{name}: Index conversion failed - {e}")
                return None
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data.index = pd.DatetimeIndex(data.index.date)
        if data.index.duplicated().any():
            data = data[~data.index.duplicated(keep="last")]
        if len(data) < 10:
            return None
        return data

    def _should_update_fred_series(self, series_id: str, last_date: str) -> bool:
        last_dt = pd.to_datetime(last_date)
        now = datetime.now()

        if series_id in self.QUARTERLY_SERIES:
            quarter_end_months = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}
            last_quarter_month = max(
                [m for m in quarter_end_months.keys() if m <= last_dt.month]
            )
            if now.month > last_quarter_month:
                quarter_end = datetime(
                    last_dt.year if last_quarter_month < 12 else last_dt.year + 1,
                    last_quarter_month + 1 if last_quarter_month < 12 else 1,
                    1,
                )
                release_date = quarter_end + timedelta(days=35)
                return now >= release_date
            return False

        if series_id in self.MONTHLY_SERIES:
            if last_dt.month == now.month and last_dt.year == now.year:
                return False
            if now.month == last_dt.month + 1 or (
                now.month == 1 and last_dt.month == 12
            ):
                return now.day >= 12
            return True

        if series_id in self.DAILY_SERIES:
            return len(pd.bdate_range(last_dt, now)) - 1 > 1

        return (now - last_dt).days > 2

    def fetch_fred(
        self,
        series_id: str,
        start_date: str = None,
        end_date: str = None,
        incremental: bool = True,
    ) -> Optional[pd.Series]:
        if not self.fred_api_key:
            return None

        cache_path = self.cache_dir / f"fred_{series_id}.parquet"

        if incremental and cache_path.exists():
            last_date = self._get_last_date_from_cache(cache_path)
            if last_date:
                if not self._should_update_fred_series(series_id, last_date):
                    try:
                        cached_df = pd.read_parquet(cache_path)
                        if series_id in cached_df.columns:
                            return cached_df[series_id]
                    except:
                        pass
                start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
                self.logger.info(
                    f"FRED:{series_id}: Checking updates from {start_date}"
                )

        try:
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "observation_start": start_date or "1990-01-01",
                "observation_end": end_date or datetime.now().strftime("%Y-%m-%d"),
            }
            response = requests.get(self.fred_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data or not data["observations"]:
                if cache_path.exists():
                    try:
                        cached_df = pd.read_parquet(cache_path)
                        if series_id in cached_df.columns:
                            return cached_df[series_id]
                    except:
                        pass
                return None

            obs = data["observations"]
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["value"] != "."]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            df = df.set_index("date")[["value"]]
            df.columns = [series_id]
            df = self._normalize_data(df, f"FRED:{series_id}")

            if df is None:
                if cache_path.exists():
                    try:
                        cached_df = pd.read_parquet(cache_path)
                        if series_id in cached_df.columns:
                            return cached_df[series_id]
                    except:
                        pass
                return None

            if series_id in self.QUARTERLY_SERIES or series_id in self.MONTHLY_SERIES:
                date_range = pd.date_range(df.index[0], df.index[-1], freq="D")
                df = df.reindex(date_range, method="ffill")
                df.index = pd.DatetimeIndex(df.index.date)

            if incremental and cache_path.exists():
                df = self._merge_with_cache(df, cache_path)

            df.to_parquet(cache_path)
            self.logger.info(f"FRED:{series_id}: Updated ({len(df)} rows)")
            return df[series_id] if series_id in df.columns else None

        except Exception as e:
            if cache_path.exists():
                try:
                    cached_df = pd.read_parquet(cache_path)
                    if series_id in cached_df.columns:
                        return cached_df[series_id]
                except:
                    pass
            self.logger.warning(f"FRED:{series_id}: Fetch failed - {str(e)[:100]}")
            return None

    def fetch_fred_series(
        self,
        series_id: str,
        start_date: str = None,
        end_date: str = None,
        incremental: bool = True,
    ) -> Optional[pd.Series]:
        return self.fetch_fred(series_id, start_date, end_date, incremental)

    def fetch_all_fred_series(
        self, start_date: str = None, end_date: str = None, incremental: bool = True
    ) -> Dict[str, pd.Series]:
        all_series = {}
        fetched_count = 0

        for group_name, series_dict in self.FRED_SERIES_GROUPS.items():
            for series_id, readable_name in series_dict.items():
                cache_path = self.cache_dir / f"fred_{series_id}.parquet"
                was_cached = cache_path.exists()
                series = self.fetch_fred(
                    series_id, start_date, end_date, incremental=incremental
                )

                if series is not None:
                    all_series[readable_name] = series
                    if was_cached and cache_path.exists():
                        cache_age = (
                            datetime.now()
                            - datetime.fromtimestamp(cache_path.stat().st_mtime)
                        ).days
                        if cache_age < 1:
                            fetched_count += 1

        if all_series:
            msg = f"FRED: {len(all_series)} series"
            if fetched_count > 0:
                msg += f" ({fetched_count} updated)"
            self.logger.info(msg)
        return all_series

    def fetch_yahoo(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        incremental: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Yahoo Finance data with intelligent caching.

        Key behaviors:
        - Historical requests (end_date > 7 days ago): Only use cache if it covers full range
        - Live requests (end_date = None or recent): Incrementally update cache
        """
        cache_path = (
            self.cache_dir
            / f"yahoo_{symbol.replace('^', '').replace('=', '_')}.parquet"
        )

        # Parse end_date to determine request type
        if end_date:
            end_dt = pd.to_datetime(end_date)
            is_historical_request = end_dt < (datetime.now() - timedelta(days=7))
        else:
            end_dt = None
            is_historical_request = False

        # HISTORICAL REQUEST PATH: Use cache only if it covers the full range
        if is_historical_request and cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                if not cached_df.empty:
                    start_dt = (
                        pd.to_datetime(start_date) if start_date else cached_df.index[0]
                    )

                    # Check if cache covers the requested range
                    cache_start = cached_df.index[0]
                    cache_end = cached_df.index[-1]

                    # Cache is usable if it starts before/at requested start and ends after/at requested end
                    cache_covers_range = (cache_start <= start_dt) and (
                        cache_end >= end_dt
                    )

                    if cache_covers_range:
                        # Filter to requested date range
                        filtered_df = cached_df[
                            (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
                        ]

                        if not filtered_df.empty:
                            self.logger.info(
                                f"Yahoo:{symbol}: Historical data ({start_dt.date()} to {end_dt.date()})"
                            )
                            return filtered_df
                    else:
                        # Cache doesn't cover full range - need to fetch
                        self.logger.info(
                            f"Yahoo:{symbol}: Cache incomplete "
                            f"(has {cache_start.date()} to {cache_end.date()}, "
                            f"need {start_dt.date()} to {end_dt.date()})"
                        )
                        # Fall through to fetch logic

            except Exception as e:
                self.logger.warning(f"Yahoo:{symbol}: Cache read failed - {e}")

        # LIVE/UPDATE REQUEST PATH: Fetch new data
        fetch_start = start_date or "2000-01-01"

        # For historical requests with incomplete cache, disable incremental mode
        # We need to fetch the full range requested
        if is_historical_request:
            incremental = False
            self.logger.info(
                f"Yahoo:{symbol}: Fetching full historical range ({fetch_start} to {end_date})"
            )

        # For incremental updates, only fetch from last cached date
        if incremental and cache_path.exists():
            last_date = self._get_last_date_from_cache(cache_path)
            if last_date:
                last_dt = pd.to_datetime(last_date)
                now = datetime.now()
                business_days_diff = len(pd.bdate_range(last_dt, now)) - 1
                cache_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                hours_since_update = (now - cache_mtime).total_seconds() / 3600

                # Return cache if it's fresh enough
                if business_days_diff == 0 and hours_since_update < 1:
                    try:
                        cached_df = pd.read_parquet(cache_path)
                        if not cached_df.empty:
                            self.logger.info(
                                f"Yahoo:{symbol}: Using cache (updated {int(hours_since_update * 60)}m ago)"
                            )
                            return cached_df
                    except:
                        pass
                elif business_days_diff == 1 and now.hour < 16:
                    try:
                        cached_df = pd.read_parquet(cache_path)
                        if not cached_df.empty:
                            self.logger.info(
                                f"Yahoo:{symbol}: Using cache (1 bday behind, pre-market)"
                            )
                            return cached_df
                    except:
                        pass

                # Fetch from 2 days before last date to ensure no gaps
                fetch_start = (last_dt - timedelta(days=2)).strftime("%Y-%m-%d")
                self.logger.info(
                    f"Yahoo:{symbol}: Fetching from {fetch_start} ({business_days_diff} bdays behind)"
                )

        try:
            ticker = yf.Ticker(symbol)

            # For live requests, default to slightly in the future to catch today's data
            fetch_end = (
                end_date
                if end_date
                else (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
            )

            df = ticker.history(
                start=fetch_start,
                end=fetch_end,
                auto_adjust=True,
            )

            if df.empty:
                if cache_path.exists():
                    try:
                        cached_df = pd.read_parquet(cache_path)
                        if not cached_df.empty:
                            self.logger.info(
                                f"Yahoo:{symbol}: No new data, using cache"
                            )
                            return cached_df
                    except:
                        pass
                return None

            df = self._normalize_data(df, f"Yahoo:{symbol}")
            if df is None:
                if cache_path.exists():
                    try:
                        return pd.read_parquet(cache_path)
                    except:
                        pass
                return None

            # Merge with cache for incremental updates
            if incremental and cache_path.exists():
                df = self._merge_with_cache(df, cache_path)

            # Save updated cache
            df.to_parquet(cache_path)
            last_date_str = df.index[-1].strftime("%Y-%m-%d")
            self.logger.info(
                f"Yahoo:{symbol}: {len(df)} rows (latest: {last_date_str})"
            )

            # If this was a historical request, filter before returning
            if is_historical_request:
                start_dt = pd.to_datetime(start_date) if start_date else df.index[0]
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            return df

        except Exception as e:
            if cache_path.exists():
                try:
                    cached_df = pd.read_parquet(cache_path)
                    if not cached_df.empty:
                        self.logger.warning(
                            f"Yahoo:{symbol}: Fetch failed, using cache - {str(e)[:100]}"
                        )
                        # Filter to requested range if historical
                        if is_historical_request:
                            start_dt = (
                                pd.to_datetime(start_date)
                                if start_date
                                else cached_df.index[0]
                            )
                            cached_df = cached_df[
                                (cached_df.index >= start_dt)
                                & (cached_df.index <= end_dt)
                            ]
                        return cached_df
                except:
                    pass
            self.logger.error(f"Yahoo:{symbol}: {str(e)[:100]}")
            return None

    def fetch_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except:
            pass
        return None

    def fetch_cboe_series(self, symbol: str) -> Optional[pd.Series]:
        if symbol not in self.CBOE_FILES:
            return None
        file_path = self.cboe_data_dir / self.CBOE_FILES[symbol]
        if not file_path.exists():
            self.logger.warning(f"CBOE:{symbol}: File not found")
            return None
        try:
            df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
            df = self._normalize_data(df, f"CBOE:{symbol}")
            if df is None or df.empty:
                return None
            self.logger.info(f"CBOE:{symbol}: {len(df)} rows")
            return df.iloc[:, 0] if len(df.columns) > 0 else None
        except Exception as e:
            self.logger.error(f"CBOE:{symbol}: {str(e)[:100]}")
            return None

    def fetch_all_cboe(self) -> Dict[str, pd.Series]:
        series_dict = {}
        for symbol in self.CBOE_FILES.keys():
            series = self.fetch_cboe_series(symbol)
            if series is not None:
                series_dict[symbol] = series
        if series_dict:
            self.logger.info(f"CBOE: {len(series_dict)} series loaded")
        return series_dict

    def fetch_fomc_calendar(
        self, start_year: int, end_year: int
    ) -> Optional[pd.DataFrame]:
        cache_path = self.cache_dir / "fomc_calendar.csv"
        if not cache_path.exists():
            self.logger.error(f"FOMC: Calendar file not found at {cache_path}")
            return None
        try:
            df = pd.read_csv(cache_path, parse_dates=["date"])
            if "date" not in df.columns or "meeting_type" not in df.columns:
                self.logger.error(
                    "FOMC: Invalid CSV format. Required columns: date, meeting_type"
                )
                return None
            df = df.set_index("date")
            df = self._normalize_data(df, "FOMC:calendar")
            if df is None or df.empty:
                self.logger.error("FOMC: Calendar normalization failed")
                return None
            df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
            if df.empty:
                self.logger.warning(
                    f"FOMC: No meetings found in range {start_year}-{end_year}"
                )
                return None
            self.logger.info(f"FOMC: {len(df)} meetings ({start_year}-{end_year})")
            return df
        except Exception as e:
            self.logger.error(f"FOMC: Failed to load calendar - {str(e)[:100]}")
            return None

    def update_fomc_calendar_from_csv(self, csv_path: str) -> Optional[pd.DataFrame]:
        try:
            import_df = pd.read_csv(csv_path, parse_dates=["date"])
            if (
                "date" not in import_df.columns
                or "meeting_type" not in import_df.columns
            ):
                self.logger.error(
                    "FOMC: Invalid import CSV. Required columns: date, meeting_type"
                )
                return None

            existing_df = self.fetch_fomc_calendar(1990, 2100)
            if existing_df is None:
                self.logger.error("FOMC: Cannot update - existing calendar not found")
                return None

            existing_df = existing_df.reset_index()
            existing_df.rename(columns={existing_df.columns[0]: "date"}, inplace=True)
            combined = pd.concat([existing_df, import_df])
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date")

            cache_path = self.cache_dir / "fomc_calendar.csv"
            combined.to_csv(cache_path, index=False)

            added_count = len(combined) - len(existing_df)
            self.logger.info(f"FOMC: Calendar updated: {len(combined)} total meetings")
            if added_count > 0:
                self.logger.info(f"FOMC: Added {added_count} new meetings")

            combined = combined.set_index("date")
            return self._normalize_data(combined, "FOMC:calendar")
        except Exception as e:
            self.logger.error(f"FOMC: Update failed - {str(e)[:100]}")
            return None
