"""fetch_yahoo caches closes only: an in-progress session bar is never persisted, and a bar written
before its session's final print is refetched (2026-11-05: a 10:01 ET SPX print sat as the close).
Run: cd src && python3 -m pytest tests/test_fetch_yahoo_closes.py -q"""
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import data_fetcher as dfm

ET = ZoneInfo("America/New_York")
NOW_ET = datetime(2026, 11, 6, 14, 45)
YAHOO = {"2026-11-03": 7568.92, "2026-11-04": 7688.62, "2026-11-05": 7353.73, "2026-11-06": 7400.00}


def _frame(closes):
    return pd.DataFrame({"Close": list(closes.values())}, index=pd.DatetimeIndex(pd.to_datetime(list(closes))))


class _Ticker:
    calls = 0

    def __init__(self, symbol):
        pass

    def history(self, start, end, auto_adjust):
        _Ticker.calls += 1
        return _frame(YAHOO)


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    monkeypatch.setattr(dfm.UnifiedDataFetcher, "_auto_update_investing_com", lambda self: None)
    monkeypatch.setattr(dfm, "_et_now", lambda: NOW_ET)
    monkeypatch.setattr(dfm.yf, "Ticker", _Ticker)
    _Ticker.calls = 0
    return dfm.UnifiedDataFetcher(cache_dir=str(tmp_path))


def _seed(fetcher, closes, written_et):
    path = fetcher.cache_dir / "yahoo_GSPC.parquet"
    _frame(closes).to_parquet(path)
    ts = written_et.replace(tzinfo=ET).timestamp()
    os.utime(path, (ts, ts))
    return path


def test_in_progress_bar_not_persisted(fetcher):
    path = _seed(fetcher, {"2026-11-03": 7568.92, "2026-11-04": 7688.62}, datetime(2026, 11, 5, 9, 0))
    out = fetcher.fetch_yahoo("^GSPC")
    cached = pd.read_parquet(path)
    assert cached.index[-1] == pd.Timestamp("2026-11-05")
    assert pd.Timestamp("2026-11-06") not in out.index


def test_pre_close_bar_is_refetched(fetcher):
    path = _seed(fetcher, {"2026-11-04": 7688.62, "2026-11-05": 7563.80}, datetime(2026, 11, 5, 10, 1))
    fetcher.fetch_yahoo("^GSPC")
    assert _Ticker.calls == 1
    assert pd.read_parquet(path).loc["2026-11-05", "Close"] == 7353.73


def test_post_close_bar_served_from_cache(fetcher):
    _seed(fetcher, {"2026-11-04": 7688.62, "2026-11-05": 7353.73}, datetime(2026, 11, 5, 17, 0))
    fetcher.fetch_yahoo("^GSPC")
    assert _Ticker.calls == 0
