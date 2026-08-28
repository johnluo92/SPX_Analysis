"""Look-ahead-safe daily timing-signal panel for volatility-timing backtests.

Every column uses ONLY information available at the market close of day T.
No forward leakage: rolling percentiles exclude day T from their own reference
window; trailing realized vol ends at T; same-day-published indices (VIX, CBOE
correlation/skew/put-call) are used as-of T with no shift.

This module builds NO trading P&L. It emits a clean signal panel only.

Architecture
------------
1. `load_raw()`   -> dict[str, Series] on each source's native DatetimeIndex.
                     A missing/unreadable source is skipped and recorded as a drop.
2. `align_raw()`  -> reindex every raw series onto the GSPC master trading-day
                     index, forward-filling only across genuine holiday gaps
                     (limit=FFILL_LIMIT days). Trailing coverage gaps (e.g. a CBOE
                     feed that ends before GSPC) are LEFT as NaN — never fabricated.
3. SIGNALS registry: name -> callable(aligned_raw) -> Series on the GSPC index.
   Adding a signal is a few lines: write a callable, register it. If a callable's
   input is absent, that column is dropped and reported (not silently filled).

Regenerate the parquet:
    python src/harness/signal_panel.py
    # or, programmatically:
    from harness.signal_panel import build_panel
    panel = build_panel(write=True)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent                       # .../SPX_Analysis/src
DATA = SRC / "data_cache"
VX = DATA / "vx_continuous"
CBOE = SRC / "CBOE_Data_Archive"
OUT = HERE / "signals.parquet"

FFILL_LIMIT = 5      # bridge holiday-length gaps only; never a multi-week hole
RV_WINDOW = 21       # trailing trading days for realized-vol
PCT_WINDOW = 252     # rolling percentile lookback (trading days)
PCT_MINP = 63        # min prior obs before a percentile is emitted (~1 quarter)


# ── source loading ────────────────────────────────────────────────────────────

def _parquet_close(path: Path, col: str) -> pd.Series:
    df = pd.read_parquet(path)
    s = df[col].copy()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _cboe_close(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")   # PCCE/TOTAL dup dates
    s = pd.Series(df["Close"].values, index=pd.DatetimeIndex(df["Date"]).normalize())
    return s.sort_index()


def load_raw() -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Return (raw_series, drops). Each entry is a Series on its native index.

    A source that is missing or unreadable is skipped and its reason recorded in
    `drops`; downstream signals depending on it are then dropped too.
    """
    specs: dict[str, Callable[[], pd.Series]] = {
        "spx":   lambda: _parquet_close(DATA / "yahoo_GSPC.parquet", "Close"),
        "vix":   lambda: _parquet_close(DATA / "yahoo_VIX.parquet", "Close"),
        "vix3m": lambda: _parquet_close(DATA / "yahoo_VIX3M.parquet", "Close"),
        "vvix":  lambda: _parquet_close(DATA / "yahoo_VVIX.parquet", "Close"),
        "vix6m": lambda: _parquet_close(DATA / "yahoo_VIX6M.parquet", "Close"),
        "vx1":   lambda: _parquet_close(VX / "VX1.parquet", "settle"),
        "vx2":   lambda: _parquet_close(VX / "VX2.parquet", "settle"),
        "cor1m": lambda: _cboe_close(CBOE / "COR1M_CBOE.csv"),
        "cor3m": lambda: _cboe_close(CBOE / "COR3M_CBOE.csv"),
        "skew":  lambda: _cboe_close(CBOE / "SKEW_INDEX_CBOE.csv"),
        "dspx":  lambda: _cboe_close(CBOE / "DSPX.csv"),
        "pcc_index":  lambda: _cboe_close(CBOE / "PCCI_INDX_CBOE.csv"),
        "pcc_equity": lambda: _cboe_close(CBOE / "PCCE_EQUITIES_CBOE.csv"),
    }
    raw, drops = {}, {}
    for name, loader in specs.items():
        try:
            s = loader().dropna()
            if s.empty:
                drops[name] = "source loaded but empty after dropna"
            else:
                raw[name] = s
        except FileNotFoundError:
            drops[name] = "source file missing"
        except Exception as exc:  # malformed file, missing column, etc.
            drops[name] = f"load error: {type(exc).__name__}: {exc}"
    return raw, drops


def align_raw(raw: dict[str, pd.Series], master: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Reindex every raw series onto the master index.

    ffill(limit=FFILL_LIMIT) bridges INTERNAL holiday-scale gaps only. A carry
    past a source's own last real print is not a holiday bridge but fabrication,
    so every value with index > that source's last genuine observation is forced
    back to NaN. A source ending before the master (e.g. CBOE feeds ending 2025-10)
    therefore keeps honest trailing NaN.
    """
    out = {}
    for k, s in raw.items():
        last_real = s.index.max()
        aligned = s.reindex(master).ffill(limit=FFILL_LIMIT)
        aligned[aligned.index > last_real] = np.nan   # no fabricated tail carry
        out[k] = aligned
    return out


# ── look-ahead-safe helpers ───────────────────────────────────────────────────

def rolling_pct_rank(s: pd.Series, window: int = PCT_WINDOW,
                     min_periods: int = PCT_MINP) -> pd.Series:
    """Percentile of s[T] within the STRICTLY-PRIOR window s[T-window .. T-1].

    Day T is excluded from its own reference distribution (this is the `.shift(1)`
    semantics the spec asks for, done inside the window: the reference is the
    prior `window` obs, today's value is only the thing being ranked). Emits NaN
    until `min_periods` prior observations exist. Look-ahead-free by construction.
    """
    def _rank(w: np.ndarray) -> float:
        today = w[-1]
        ref = w[:-1]
        ref = ref[~np.isnan(ref)]
        if len(ref) < min_periods or np.isnan(today):
            return np.nan
        return float((ref < today).mean())
    return s.rolling(window + 1, min_periods=min_periods + 1).apply(_rank, raw=True)


def trailing_realized_vol(close: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    """Annualized realized vol (%) from log returns over the trailing `window`
    days ENDING at T. Uses returns up to and including T only — strictly trailing.
    """
    ret = np.log(close / close.shift(1))
    return ret.rolling(window, min_periods=window).std() * np.sqrt(252) * 100.0


# ── signal registry ───────────────────────────────────────────────────────────
# Each callable takes the aligned-raw dict and returns a Series on the master
# index. A KeyError (missing input source) causes the column to be dropped and
# reported. The trailing comment on each = its day-T timing assumption.

SIGNALS: dict[str, Callable[[dict[str, pd.Series]], pd.Series]] = {
    # VIX term structure — all same-day close, no shift.
    "vix":              lambda r: r["vix"],                                   # VIX close(T)
    "vix3m":            lambda r: r["vix3m"],                                 # VIX3M close(T)
    "vix_vix3m_ratio":  lambda r: r["vix"] / r["vix3m"],                      # <1 = contango/calm; close(T)/close(T)
    "vix3m_minus_vix":  lambda r: r["vix3m"] - r["vix"],                      # close(T) spread
    "vvix":             lambda r: r["vvix"],                                  # VVIX close(T)
    "vix6m":            lambda r: r["vix6m"],                                 # VIX6M close(T)

    # VX futures term structure — settle(T), published EOD.
    "vx_contango_pct":  lambda r: (r["vx2"] / r["vx1"] - 1.0) * 100.0,        # (VX2/VX1-1)*100, settle(T)

    # CBOE implied-correlation — index published same day at close.
    "cor1m":            lambda r: r["cor1m"],                                 # COR1M close(T)
    "cor3m":            lambda r: r["cor3m"],                                 # COR3M close(T)
    "cor_term_ratio":   lambda r: r["cor1m"] / r["cor3m"],                    # close(T)/close(T)
    "cor1m_chg_5d":     lambda r: r["cor1m"] - r["cor1m"].shift(5),          # T minus (T-5); trailing

    # Rolling percentiles — reference window EXCLUDES day T (look-ahead-free).
    "cor1m_pct_252d":          lambda r: rolling_pct_rank(r["cor1m"]),                       # rank of cor1m(T) vs T-252..T-1
    "vix_vix3m_ratio_pct_252d": lambda r: rolling_pct_rank(r["vix"] / r["vix3m"]),           # rank of ratio(T) vs T-252..T-1

    # Variance-risk-premium — VIX(T) minus STRICTLY-TRAILING 21d realized vol ending at T.
    "vrp_21d":          lambda r: r["vix"] - trailing_realized_vol(r["spx"]),  # both as-of T; RV uses returns <=T

    # Tail / skew / positioning — CBOE indices published same day.
    "skew":             lambda r: r["skew"],                                 # SKEW close(T)
    "pcc_index":        lambda r: r["pcc_index"],                            # PCCI index put/call, close(T)
    "pcc_equity":       lambda r: r["pcc_equity"],                           # PCCE equity put/call, close(T)
    "dspx":             lambda r: r["dspx"],                                 # DSPX dispersion, close(T)
}


# ── builder ───────────────────────────────────────────────────────────────────

def build_panel(write: bool = True, verbose: bool = True) -> pd.DataFrame:
    raw, load_drops = load_raw()

    if "spx" not in raw:
        raise RuntimeError("GSPC (yahoo_GSPC.parquet) is the master index and could "
                           f"not be loaded: {load_drops.get('spx', 'unknown')}")
    master = raw["spx"].index

    aligned = align_raw(raw, master)

    cols, drops = {}, dict(load_drops)
    for name, fn in SIGNALS.items():
        try:
            s = fn(aligned)
        except KeyError as exc:
            drops[name] = f"depends on missing source {exc}"
            continue
        cols[name] = s.reindex(master)

    panel = pd.DataFrame(cols, index=master)
    panel.index.name = "date"

    if write:
        panel.to_parquet(OUT)

    if verbose:
        _print_summary(panel, drops, master, wrote=write)

    return panel


def _print_summary(panel: pd.DataFrame, drops: dict[str, str],
                   master: pd.DatetimeIndex, wrote: bool) -> None:
    n = len(panel)
    print("=" * 78)
    print(f"SIGNAL PANEL  |  {n} GSPC trading days  |  {master[0].date()} .. {master[-1].date()}")
    print("=" * 78)
    print(f"{'column':<26}{'first':>12}{'last':>12}{'n_valid':>9}{'n_nan':>8}")
    print("-" * 78)
    for c in panel.columns:
        s = panel[c]
        valid = s.dropna()
        first = valid.index[0].date() if not valid.empty else "-"
        last = valid.index[-1].date() if not valid.empty else "-"
        print(f"{c:<26}{str(first):>12}{str(last):>12}{len(valid):>9}{s.isna().sum():>8}")
    print("-" * 78)

    full = panel.dropna(how="any")
    if full.empty:
        print("FULL-ROW SPAN (all columns present): none (coverage never fully overlaps)")
    else:
        print(f"FULL-ROW SPAN (every column non-NaN): {full.index[0].date()} .. {full.index[-1].date()}  ({len(full)} rows)")

    if drops:
        print("\nDROPPED COLUMNS / SOURCES:")
        for k, why in drops.items():
            print(f"  - {k}: {why}")
    else:
        print("\nDROPPED COLUMNS / SOURCES: none")

    if wrote:
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    build_panel(write=True, verbose=True)
