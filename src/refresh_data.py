#!/usr/bin/env python3
"""
Byzantium Capital — Data Refresh
Updates VX continuous contracts + Yahoo market data used by regime_dashboard.py.
Does NOT run any ML training.

Usage:
    cd /Users/johnluo/Desktop/GitHub/SPX_Analysis/src
    python3 refresh_data.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Symbols the dashboard actually uses
YAHOO_SYMBOLS = [
    "^VIX",    # front-month vol
    "^VIX3M",  # 3-month vol (term structure)
    "^VIX6M",  # 6-month vol
    "^GSPC",   # SPX price
    "^VVIX",   # vol-of-vol
    "^SKEW",   # tail risk
]

# FRED series required by layer1_gate.py conditions C4 (credit) and C5 (stress)
FRED_GATE_SERIES = [
    "STLFSI4",       # St. Louis Financial Stress Index — C5 stress regime
    "BAMLH0A1HYBB",  # HY BB OAS — C4 credit widening (quality spread numerator)
    "BAMLH0A3HYC",   # HY CCC OAS — C4 credit widening (quality spread denominator)
]


def _last_date(path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_parquet(path)
        return df.index[-1] if not df.empty else None
    except Exception:
        return None


def refresh_vx() -> None:
    from core.vx_continuous_contract_builder import VXContinuousContractBuilder

    builder = VXContinuousContractBuilder()
    cache_dir = SRC / "data_cache" / "vx_continuous"

    # Fetch any CBOE CSVs missing from CDN before rebuilding
    last = _last_date(cache_dir / "VX1.parquet")
    if last:
        start = last + timedelta(days=1)
        end   = pd.Timestamp.now().normalize()
        dates = pd.bdate_range(start, end)
        if len(dates):
            print(f"  Fetching CBOE CSVs for {len(dates)} missing trading days...")
            for d in dates:
                builder._fetch_missing_cboe_data(d)

    contracts = builder.get_all_continuous_contracts(force_rebuild=True)

    vx1_last = contracts["VX1"].index[-1] if "VX1" in contracts else None

    for name, df in sorted(contracts.items()):
        last_dt  = df.index[-1].strftime("%Y-%m-%d")
        settle   = df["settle"].iloc[-1]
        dte      = int(df["days_to_expiry"].iloc[-1]) if "days_to_expiry" in df.columns else "?"
        print(f"  {name}   {last_dt}   settle={settle:>7.4f}   DTE={dte}")
        if name != "VX1" and vx1_last is not None:
            vxn_last = df.index[-1]
            lag_bd = len(pd.bdate_range(vxn_last, vx1_last)) - 1
            if lag_bd > 3:
                print(f"  ⚠ {name} lag: last={vxn_last.strftime('%Y-%m-%d')}, {lag_bd}bd behind VX1 ({vx1_last.strftime('%Y-%m-%d')})")


def refresh_fred_gate() -> None:
    from core.data_fetcher import UnifiedDataFetcher

    fetcher = UnifiedDataFetcher()
    for series_id in FRED_GATE_SERIES:
        # STLFSI4 is weekly — incremental fetch has a gap-fill bug for low-frequency series
        result = fetcher.fetch_fred(series_id, incremental=(series_id != "STLFSI4"))
        if result is not None and not result.empty:
            last_dt = result.index[-1].strftime("%Y-%m-%d")
            val     = result.iloc[-1]
            print(f"  {series_id:<18}   {last_dt}   {val:.4f}")
        else:
            print(f"  {series_id:<18}   FAILED — check FRED_API_KEY or network")


def refresh_yahoo() -> None:
    from core.data_fetcher import UnifiedDataFetcher

    fetcher = UnifiedDataFetcher()
    for sym in YAHOO_SYMBOLS:
        df = fetcher.fetch_yahoo(sym, incremental=True)
        if df is not None and not df.empty:
            last_dt = df.index[-1].strftime("%Y-%m-%d")
            col     = "Close" if "Close" in df.columns else df.columns[0]
            val     = df[col].iloc[-1]
            print(f"  {sym:<10}   {last_dt}   {col}={val:.2f}")
        else:
            print(f"  {sym:<10}   FAILED — check network or yfinance")


def main():
    t0 = datetime.now()
    print(f"\n  Data Refresh — {t0.strftime('%Y-%m-%d %H:%M')}")
    print(f"  {'─'*46}")

    print("\n  VX Continuous Contracts")
    refresh_vx()

    print("\n  Yahoo Finance")
    refresh_yahoo()

    print("\n  FRED (gate conditions C4 + C5)")
    refresh_fred_gate()

    elapsed = (datetime.now() - t0).seconds
    print(f"\n  Done in {elapsed}s.  Run regime_dashboard.py for updated signals.\n")


if __name__ == "__main__":
    main()
