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


class _FailingTicker(_Ticker):
    def history(self, start, end, auto_adjust):
        _Ticker.calls += 1
        raise ConnectionError("yahoo down")


class _EmptyTicker(_Ticker):
    def history(self, start, end, auto_adjust):
        _Ticker.calls += 1
        return pd.DataFrame()


# Three failed-refetch paths: yfinance raises, returns an empty frame, or returns rows _normalize_data rejects
# (a full fetch needs 10 rows; _Ticker serves 4).
@pytest.mark.parametrize("ticker,kwargs", [(_FailingTicker, {}), (_EmptyTicker, {}), (_Ticker, {"incremental": False})],
                         ids=["raises", "empty", "normalize-none"])
def test_failed_refetch_drops_the_partial_bar_from_return_and_disk(fetcher, monkeypatch, ticker, kwargs):
    monkeypatch.setattr(dfm.yf, "Ticker", ticker)
    logged = []
    monkeypatch.setattr(fetcher.logger, "error", logged.append)
    path = _seed(fetcher, {"2026-11-03": 7568.92, "2026-11-04": 7688.62, "2026-11-05": 7563.80}, datetime(2026, 11, 5, 10, 1))
    out = fetcher.fetch_yahoo("^GSPC", **kwargs)
    assert _Ticker.calls == 1
    assert out.index[-1] == pd.Timestamp("2026-11-04")
    on_disk = pd.read_parquet(path)
    assert list(on_disk.index) == [pd.Timestamp("2026-11-03"), pd.Timestamp("2026-11-04")]
    assert on_disk.loc["2026-11-04", "Close"] == 7688.62
    assert os.listdir(fetcher.cache_dir) == ["yahoo_GSPC.parquet"]
    assert len(logged) == 1 and "^GSPC" in logged[0] and "2026-11-05" in logged[0] and "dropped it from the cache" in logged[0]


def test_failed_refetch_serves_a_final_cache_whole(fetcher, monkeypatch):
    monkeypatch.setattr(dfm.yf, "Ticker", _FailingTicker)
    _seed(fetcher, {"2026-11-03": 7568.92, "2026-11-04": 7688.62}, datetime(2026, 11, 4, 17, 0))
    out = fetcher.fetch_yahoo("^GSPC")
    assert _Ticker.calls == 1
    assert out.index[-1] == pd.Timestamp("2026-11-04")


def test_failed_refetch_removes_a_one_row_partial_cache(fetcher, monkeypatch):
    monkeypatch.setattr(dfm.yf, "Ticker", _FailingTicker)
    _seed(fetcher, {"2026-11-05": 7563.80}, datetime(2026, 11, 5, 10, 1))
    assert fetcher.fetch_yahoo("^GSPC") is None
    assert os.listdir(fetcher.cache_dir) == []


# A reader holding the old file open (layer1_gate, dn_tactical, byz) keeps reading the whole old file: the cache is
# replaced by a new inode, never truncated and rewritten in place. Covers the success path and the fallback rewrite.
@pytest.mark.parametrize("ticker,written", [(_Ticker, datetime(2026, 11, 5, 9, 0)), (_FailingTicker, datetime(2026, 11, 5, 10, 1))],
                         ids=["success", "fallback"])
def test_cache_write_replaces_never_truncates(fetcher, monkeypatch, ticker, written):
    monkeypatch.setattr(dfm.yf, "Ticker", ticker)
    path = _seed(fetcher, {"2026-11-03": 7568.92, "2026-11-04": 7688.62, "2026-11-05": 7563.80}, written)
    before = path.read_bytes()
    with open(path, "rb") as reader:
        fetcher.fetch_yahoo("^GSPC")
        assert _Ticker.calls == 1
        assert path.read_bytes() != before
        assert reader.read() == before


class _FredResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"observations": [{"date": "2026-11-04", "value": "3.10"}, {"date": "2026-11-05", "value": "3.20"}]}


# The FRED success path (C4/C5's OAS and STLFSI) is replaced the same way: a top-up never truncates the live file.
def test_fred_write_replaces_never_truncates(fetcher, monkeypatch):
    fetcher.fred_api_key = "test-key"
    gets = []
    monkeypatch.setattr(dfm.requests, "get", lambda url, params, timeout: gets.append(params) or _FredResponse())
    path = fetcher.cache_dir / "fred_BAMLH0A1HYBB.parquet"
    pd.DataFrame({"BAMLH0A1HYBB": [3.00, 3.05]}, index=pd.DatetimeIndex(["2026-10-30", "2026-11-02"])).to_parquet(path)
    before = path.read_bytes()
    with open(path, "rb") as reader:
        out = fetcher.fetch_fred("BAMLH0A1HYBB")
        assert len(gets) == 1 and gets[0]["observation_start"] == "2026-11-03"
        assert out.index[-1] == pd.Timestamp("2026-11-05") and out.iloc[-1] == 3.20
        assert pd.read_parquet(path).index[-1] == pd.Timestamp("2026-11-05")
        assert reader.read() == before
    assert os.listdir(fetcher.cache_dir) == [path.name]

