#!/usr/bin/env python3
"""
Gate sector backtest: forward returns by equity sector after GO→NO-GO transitions.

Usage:
  python gate_sector_backtest.py
"""
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from layer1_gate import build_gate_features, evaluate_gate

TICKERS = ['SPY', 'XLK', 'XLY', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLC']
HORIZONS = [5, 21]
WARMUP = 63  # rows dropped for rolling warmup in gate features


def identify_transitions(go: pd.Series) -> pd.DatetimeIndex:
    """Return dates where gate flipped GO→NO-GO (True→False)."""
    shifted = go.shift(1)
    transitions = go.index[(shifted == True) & (go == False)]
    return transitions


def classify_transitions(f: pd.DataFrame, go: pd.Series, transitions: pd.DatetimeIndex):
    """
    Split transitions into C4-triggered vs non-C4.

    C4-triggered: on the day BEFORE the transition (last GO day), C4 was passing
    AND on the transition date C4 is failing.
    Non-C4: gate closes but C4 was already failing on the last GO day
    (meaning something else (C1/C2/C3/C5) changed to close the gate).
    """
    c4_triggered = []
    non_c4 = []
    for t in transitions:
        t_idx = f.index.get_loc(t)
        if t_idx == 0:
            continue
        prev_date = f.index[t_idx - 1]
        c4_was_ok_yesterday = (f.loc[prev_date, "credit_widening_regime"] == 0)
        c4_fail_today = (f.loc[t, "credit_widening_regime"] != 0)
        if c4_was_ok_yesterday and c4_fail_today:
            c4_triggered.append(t)
        else:
            non_c4.append(t)
    return pd.DatetimeIndex(c4_triggered), pd.DatetimeIndex(non_c4)


def compute_forward_returns(prices: pd.DataFrame, events: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """
    For each event date and each ticker, compute the horizon-day forward return
    starting from the CLOSE on the event date.
    Returns DataFrame indexed by event date, columns = tickers, values = % return.
    """
    rows = {}
    for t in events:
        try:
            t_pos = prices.index.get_loc(t)
        except KeyError:
            # Event date not a trading day — find next available
            later = prices.index[prices.index >= t]
            if len(later) == 0:
                continue
            t_pos = prices.index.get_loc(later[0])

        fwd_pos = t_pos + horizon
        if fwd_pos >= len(prices):
            continue  # not enough forward data

        p0 = prices.iloc[t_pos]
        p1 = prices.iloc[fwd_pos]
        ret = (p1 / p0 - 1) * 100
        rows[prices.index[t_pos]] = ret

    return pd.DataFrame(rows).T


def compute_pre_returns(prices: pd.DataFrame, events: pd.DatetimeIndex, lookback: int) -> pd.DataFrame:
    """
    For each event date compute the lookback-day return ENDING on the event date close.
    Returns DataFrame indexed by event date, columns = tickers.
    """
    rows = {}
    for t in events:
        try:
            t_pos = prices.index.get_loc(t)
        except KeyError:
            later = prices.index[prices.index >= t]
            if len(later) == 0:
                continue
            t_pos = prices.index.get_loc(later[0])

        pre_pos = t_pos - lookback
        if pre_pos < 0:
            continue

        p0 = prices.iloc[pre_pos]
        p1 = prices.iloc[t_pos]
        ret = (p1 / p0 - 1) * 100
        rows[prices.index[t_pos]] = ret

    return pd.DataFrame(rows).T


def print_table(label: str, events: pd.DatetimeIndex, fwd_dict: dict, pre_spy_5d, pre_spy_1d):
    n = len(events)
    if n == 0:
        print(f"\n{label}: no events")
        return

    print(f"\n{label}  (n={n} events)")
    print(f"  Transition dates: {[str(d.date()) for d in events]}")

    for horizon, ret_df in fwd_dict.items():
        sub = ret_df[ret_df.index.isin(events)]
        print(f"\n  Forward +{horizon}d returns (mean across events):")
        header = f"  {'Ticker':<6}  {'Mean%':>7}  {'Median%':>8}  {'n':>4}"
        print(header)
        print(f"  {'-'*36}")
        for ticker in TICKERS:
            if ticker not in sub.columns:
                print(f"  {ticker:<6}  {'no data':>7}")
                continue
            col = sub[ticker].dropna()
            n_t = len(col)
            if n_t < 5:
                print(f"  {ticker:<6}  n too small (n={n_t})")
            else:
                print(f"  {ticker:<6}  {col.mean():>+7.2f}%  {col.median():>+8.2f}%  {n_t:>4}")

    # Pre-event SPX returns
    print(f"\n  SPX leading indicators (pre-event, SPY):")
    for lookback_label, pre_df in [("-5d", pre_spy_5d), ("-1d", pre_spy_1d)]:
        if "SPY" not in pre_df.columns:
            continue
        sub = pre_df[pre_df.index.isin(events)]["SPY"].dropna()
        n_t = len(sub)
        if n_t < 5:
            print(f"    SPY {lookback_label}: n too small (n={n_t})")
        else:
            print(f"    SPY {lookback_label}: mean={sub.mean():+.2f}%  median={sub.median():+.2f}%  (n={n_t})")


def main():
    print("Building gate features...")
    f = build_gate_features()
    go_raw = evaluate_gate(f)

    # Drop warmup rows
    f = f.iloc[WARMUP:]
    go = go_raw.iloc[WARMUP:]

    transitions = identify_transitions(go)
    print(f"Total GO→NO-GO transitions found: {len(transitions)}")
    if len(transitions) == 0:
        print("No transitions found. Exiting.")
        return

    c4_trans, non_c4_trans = classify_transitions(f, go, transitions)
    print(f"  C4-triggered closures: {len(c4_trans)}")
    print(f"  Non-C4 closures: {len(non_c4_trans)}")

    print("\nDownloading sector ETF data via yfinance (2007-01-01 to present)...")
    raw = yf.download(TICKERS, start='2007-01-01', auto_adjust=True, progress=False)
    prices = raw['Close'].ffill()
    print(f"  Price data: {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} trading days)")

    # Compute forward returns for all transitions
    fwd = {}
    for h in HORIZONS:
        fwd[h] = compute_forward_returns(prices, transitions, h)

    # Pre-event returns for SPY
    pre_5d = compute_pre_returns(prices, transitions, 5)
    pre_1d = compute_pre_returns(prices, transitions, 1)

    # For subgroup analysis
    fwd_c4 = {}
    fwd_non_c4 = {}
    pre_5d_c4 = compute_pre_returns(prices, c4_trans, 5)
    pre_1d_c4 = compute_pre_returns(prices, c4_trans, 1)
    pre_5d_non_c4 = compute_pre_returns(prices, non_c4_trans, 5)
    pre_1d_non_c4 = compute_pre_returns(prices, non_c4_trans, 1)
    for h in HORIZONS:
        fwd_c4[h] = compute_forward_returns(prices, c4_trans, h)
        fwd_non_c4[h] = compute_forward_returns(prices, non_c4_trans, h)

    print("\n" + "=" * 60)
    print("SECTION 1: ALL GO→NO-GO TRANSITIONS")
    print("=" * 60)
    print_table("All transitions", transitions, fwd, pre_5d, pre_1d)

    print("\n" + "=" * 60)
    print("SECTION 2: C4-TRIGGERED CLOSURES (C4 was passing, then flipped)")
    print("=" * 60)
    print_table("C4-triggered", c4_trans, fwd_c4, pre_5d_c4, pre_1d_c4)

    print("\n" + "=" * 60)
    print("SECTION 3: NON-C4 CLOSURES (gate closed via C1/C2/C3/C5, C4 already failing or unchanged)")
    print("=" * 60)
    print_table("Non-C4", non_c4_trans, fwd_non_c4, pre_5d_non_c4, pre_1d_non_c4)


if __name__ == "__main__":
    main()
