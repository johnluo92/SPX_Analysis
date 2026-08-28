#!/usr/bin/env python3
"""Reusable, realized, tail-inclusive SPX put-credit-spread P&L simulator.

Refactored from bt_gate_soft_voting.py (the reference geometry) and bt_dp_gate_pnl.py
(realized-outcome logic). RESEARCH ONLY — realized outcomes, no look-ahead, no fabricated
numbers. One spread opened per entry date, held to expiry, priced/strike-set on entry-date
data only; outcome read from the realized SPX close at expiry.

Geometry (matches bt_gate_soft_voting_results.json meta):
  R0  (VIX <  16.77): DTE=45, W=10, K=0.50
  R1+ (VIX >= 16.77): DTE=21, W=10, K=0.65
  EM       = S * (VIX/100) * sqrt(DTE/365)
  K_short  = round((S - K*EM) / GRID) * GRID
  K_long   = K_short - W
  credit   = BSM put(K_short) - BSM put(K_long)      (no rate, no div; VIX/100 as sigma)

Realized (tail-inclusive) P&L in index points, per contract, markup 1.0:
  survived (SPX_expiry >= K_short):  +credit
  breached:                          credit - min(W, max(0, K_short - SPX_expiry))
                                     (= full-width loss when SPX gaps through K_long)

Run from src/ (imports and data_cache paths assume cwd=src/, or use absolute via DATA_DIR).
"""
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")

VIX_R0 = 16.77
GRID = 5
R0_PARAMS = dict(dte=45, width=10, k=0.50)
R1_PARAMS = dict(dte=21, width=10, k=0.65)
CREDIT_CAP_FRAC = 0.99   # credit capped at 0.99*W (matches reference pnl_at_markup); ~never binds


def bsm_put(S, K, sigma, T):
    if T <= 0:
        return max(0.0, K - S)
    if sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def load_spx(data_dir=DATA_DIR):
    return pd.read_parquet(os.path.join(data_dir, "yahoo_GSPC.parquet"))["Close"].sort_index()


def load_vix(data_dir=DATA_DIR):
    return pd.read_parquet(os.path.join(data_dir, "yahoo_VIX.parquet"))["Close"].sort_index()


def _regime_params(vix, vix_r0, r0_params, r1_params):
    return dict(r0_params) if vix < vix_r0 else dict(r1_params)


def _expiry_close(spx, entry_loc, dte_cal):
    """Realized SPX close nearest to entry_date + dte calendar days; None if not a real
    forward expiry (must be strictly after entry). No look-ahead into the outcome — the strike
    is already fixed by the time this is read."""
    entry_date = spx.index[entry_loc]
    target = entry_date + timedelta(days=dte_cal)
    loc = spx.index.get_indexer([target], method="nearest")[0]
    if loc < 0 or loc >= len(spx) or loc <= entry_loc:
        return None, None
    return float(spx.iloc[loc]), spx.index[loc]


def simulate_spreads(
    entry_dates,
    spx=None,
    vix=None,
    vix_r0=VIX_R0,
    r0_params=None,
    r1_params=None,
    grid=GRID,
    credit_cap_frac=CREDIT_CAP_FRAC,
    data_dir=DATA_DIR,
):
    """Simulate one SPX put credit spread per entry date, held to expiry.

    entry_dates: iterable of timestamps (a subset of SPX trading days).
    spx, vix: optional pre-loaded Close series; loaded from data_cache if None. VIX is aligned
        to SPX trading days via ffill(limit=5) — the entry-date IV source (matches reference).
    r0_params, r1_params: {dte, width, k} geometry per regime (defaults = documented rule).

    Returns one row per simulable trade (non-simulable entries — no forward expiry, or no VIX —
    are dropped). Columns: entry_date, expiry_date, dte, regime, short_strike, long_strike,
    width, credit, spx_entry, spx_expiry, otm_survived, realized_pnl, vix, em.
    """
    if spx is None:
        spx = load_spx(data_dir)
    if vix is None:
        vix = load_vix(data_dir)
    r0_params = R0_PARAMS if r0_params is None else r0_params
    r1_params = R1_PARAMS if r1_params is None else r1_params

    vix_aligned = vix.reindex(spx.index, method="ffill", limit=5)

    rows = []
    for d in pd.DatetimeIndex(entry_dates):
        eloc = spx.index.get_indexer([d], method="nearest")[0]
        if eloc < 0:
            continue
        v = vix_aligned.iloc[eloc]
        if pd.isna(v):
            continue
        v = float(v)
        S = float(spx.iloc[eloc])
        p = _regime_params(v, vix_r0, r0_params, r1_params)
        dte, W, K = p["dte"], p["width"], p["k"]
        T = dte / 365.0
        sigma = v / 100.0
        em = S * sigma * math.sqrt(T)
        K_short = round((S - K * em) / grid) * grid
        K_long = K_short - W
        credit = bsm_put(S, K_short, sigma, T) - bsm_put(S, K_long, sigma, T)
        credit = min(credit, credit_cap_frac * W)
        S_exp, exp_date = _expiry_close(spx, eloc, dte)
        if S_exp is None:
            continue
        survived = S_exp >= K_short
        if survived:
            pnl = credit
        else:
            loss = min(W, max(0.0, K_short - S_exp))
            pnl = credit - loss
        rows.append(dict(
            entry_date=spx.index[eloc], expiry_date=exp_date, dte=dte,
            regime="R0" if v < vix_r0 else "R1+",
            short_strike=float(K_short), long_strike=float(K_long), width=float(W),
            credit=float(credit), spx_entry=S, spx_expiry=S_exp,
            otm_survived=bool(survived), realized_pnl=float(pnl), vix=v, em=float(em),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("entry_date").reset_index(drop=True)
    return df


def aggregate(df):
    """Aggregate a simulate_spreads DataFrame into headline stats.

    annual_roc: mean per-trade return-on-capital (pnl / max-loss, max-loss = width - credit)
        annualized by trades/year — assumes capital recycles each trade (memory
        tested_dp_dte_annual_roc framing). Percent.
    max_drawdown: peak-to-trough of the equity curve (cumulative realized_pnl by entry_date),
        in index points; sequential-by-entry (understates true concurrent overlap risk).
    """
    if df is None or len(df) == 0:
        return dict(n_trades=0, total_pnl=0.0, annual_roc=float("nan"), win_rate=float("nan"),
                    max_drawdown=0.0, mean_credit=float("nan"), mean_loss_on_breach=float("nan"))
    n = len(df)
    total_pnl = float(df["realized_pnl"].sum())
    win_rate = float(df["otm_survived"].mean() * 100)
    span_days = (df["entry_date"].iloc[-1] - df["entry_date"].iloc[0]).days
    span_years = span_days / 365.25 if span_days > 0 else float("nan")
    trades_yr = n / span_years if span_years and span_years > 0 else float("nan")
    maxloss = (df["width"] - df["credit"]).clip(lower=1e-9)
    mean_roc = float((df["realized_pnl"] / maxloss).mean())
    annual_roc = mean_roc * trades_yr * 100 if trades_yr == trades_yr else float("nan")
    cum = df["realized_pnl"].cumsum()
    max_dd = float((cum.cummax() - cum).max())
    breaches = df[~df["otm_survived"]]
    mean_loss_on_breach = float(breaches["realized_pnl"].mean()) if len(breaches) else 0.0
    return dict(
        n_trades=n, total_pnl=round(total_pnl, 3), annual_roc=round(annual_roc, 2),
        win_rate=round(win_rate, 2), max_drawdown=round(max_dd, 3),
        mean_credit=round(float(df["credit"].mean()), 4),
        mean_loss_on_breach=round(mean_loss_on_breach, 4), trades_yr=round(trades_yr, 2),
    )


def per_year_pnl(df):
    if df is None or len(df) == 0:
        return {}
    g = df.assign(year=df["entry_date"].dt.year).groupby("year")["realized_pnl"].sum()
    return {int(k): round(float(v), 3) for k, v in g.items()}


def gate_open_entry_dates(warmup=63):
    """The baseline_hardAND entry set: hard-AND(C1..C5) gate-open days, post-warmup.
    Reused verbatim from bt_gate_soft_voting.main for the fidelity check."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from layer1_gate import build_gate_features, evaluate_gate
    f = build_gate_features().iloc[warmup:]
    go = evaluate_gate(f).iloc[warmup:].reindex(f.index).astype("boolean").fillna(False)
    return f.index[go.astype(bool)]


if __name__ == "__main__":
    print("Fidelity check vs bt_gate_soft_voting_results.json baseline_hardAND ...")
    import json
    entries = gate_open_entry_dates()
    df = simulate_spreads(entries)
    agg = aggregate(df)
    py = per_year_pnl(df)
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bt_gate_soft_voting_results.json")
    with open(ref_path) as fh:
        ref = json.load(fh)["configs"]["baseline_hardAND"]
    ref_py = ref["markups"]["1.0"]["per_year_pnl"]
    print(f"  entries (gate-open, post-warmup): {len(entries)}")
    print(f"  simulable trades:                 {agg['n_trades']}")
    print(f"  TARGET  wr={ref['wr']:.2f}%  n={ref['n']}  wins={ref['wins']}")
    print(f"  MINE    wr={agg['win_rate']:.2f}%  n={agg['n_trades']}  wins={int(df['otm_survived'].sum())}")
    print(f"  DELTA   wr={agg['win_rate']-ref['wr']:+.2f}pp  n={agg['n_trades']-ref['n']:+d}")
    print(f"\n  per-year P&L (markup 1.0)  mine vs target:")
    for y in sorted(set(list(py) + [int(k) for k in ref_py])):
        m = py.get(y, 0.0)
        t = ref_py.get(str(y), 0.0)
        print(f"    {y}  mine={m:>+9.3f}  target={t:>+9.3f}  delta={m-t:>+8.3f}")
    print(f"\n  aggregate: {agg}")
    print(f"\n  crisis-year trade counts: "
          f"2018={len(df[df.entry_date.dt.year==2018])} "
          f"2020={len(df[df.entry_date.dt.year==2020])} "
          f"2022={len(df[df.entry_date.dt.year==2022])}")
