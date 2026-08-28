#!/usr/bin/env python3
"""BT-VOLPEAK rigor pass: (A) paired-Δ block-bootstrap CI + sign test on the SPX
vol-peak edge, and (POWER) the same edge restricted to the 2019-2026 equity-cache
window — the decisive gate for whether the equity transfer test (Test B) is powered.

Canonical config: DTE98, trailing-126d VIX percentile (incl. T, look-ahead-safe),
PEAK = pctile >= 0.80. Reproduces the results-doc headline first as a fidelity check.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from harness.spread_sim import simulate_spreads, load_spx, load_vix

DTE98 = dict(dte=98, width=10, k=0.65)
LOOKBACK = 126
PEAK_CUT = 0.80


def vix_trailing_pct(vix, lookback=LOOKBACK):
    """Rank of VIX(T) within [T-lookback+1 .. T] inclusive of T (look-ahead-safe:
    uses only data up to and including the entry day). Returns a Series aligned to vix."""
    def _rank(w):
        last = w[-1]
        if np.isnan(last):
            return np.nan
        v = w[~np.isnan(w)]
        if len(v) < int(lookback * 0.6):
            return np.nan
        return float((v < last).sum() / len(v))
    return vix.rolling(lookback, min_periods=int(lookback * 0.6)).apply(_rank, raw=True)


def _arm(dates, spx, vix):
    df = simulate_spreads(dates, spx=spx, vix=vix, r0_params=DTE98, r1_params=DTE98)
    return df


def _stats(df):
    return dict(n=len(df), survive=float(df["otm_survived"].mean() * 100),
                per_trade=float(df["realized_pnl"].mean()))


def paired_year_bootstrap(base_df, peak_df, n=5000, seed=0):
    """Resample calendar YEARS with replacement; within each resample compute
    Δ = mean(peak per-trade) − mean(base per-trade) using the trades whose entry-year
    was drawn. Respects within-year clustering/overlap. Returns (deltas array)."""
    rng = np.random.default_rng(seed)
    by_year_b = {y: g["realized_pnl"].values for y, g in base_df.groupby(base_df["entry_date"].dt.year)}
    by_year_p = {y: g["realized_pnl"].values for y, g in peak_df.groupby(peak_df["entry_date"].dt.year)}
    years = sorted(set(by_year_b) & set(by_year_p))
    deltas = []
    for _ in range(n):
        pick = rng.choice(years, size=len(years), replace=True)
        b = np.concatenate([by_year_b[y] for y in pick])
        p = np.concatenate([by_year_p[y] for y in pick])
        deltas.append(p.mean() - b.mean())
    return np.array(deltas)


def sign_test(base_df, peak_df):
    """Per-year paired wins on per-trade P&L and survival; two-sided binomial p."""
    from scipy.stats import binomtest
    yb = base_df.assign(y=base_df.entry_date.dt.year).groupby("y")
    yp = peak_df.assign(y=peak_df.entry_date.dt.year).groupby("y")
    b_pt = yb["realized_pnl"].mean(); p_pt = yp["realized_pnl"].mean()
    b_sv = yb["otm_survived"].mean(); p_sv = yp["otm_survived"].mean()
    common = sorted(set(b_pt.index) & set(p_pt.index))
    pt_wins = sum(p_pt[y] > b_pt[y] for y in common)
    sv_wins = sum(p_sv[y] > b_sv[y] for y in common)
    N = len(common)
    return dict(
        n_years=N,
        pt_wins=int(pt_wins), pt_p=float(binomtest(int(pt_wins), N).pvalue),
        sv_wins=int(sv_wins), sv_p=float(binomtest(int(sv_wins), N).pvalue),
    )


def run(spx, vix, pct, label, lo=None, hi=None):
    all_days = pd.DatetimeIndex(spx.index)
    if lo is not None:
        all_days = all_days[(all_days >= lo) & (all_days <= hi)]
    peak_mask = (pct >= PEAK_CUT).reindex(all_days).fillna(False)
    base_dates = all_days
    peak_dates = all_days[peak_mask.values]
    base = _arm(base_dates, spx, vix)
    peak = _arm(peak_dates, spx, vix)
    bs, ps = _stats(base), _stats(peak)
    deltas = paired_year_bootstrap(base, peak)
    d_lo, d_med, d_hi = np.percentile(deltas, [5, 50, 95])
    frac_pos = float((deltas > 0).mean())
    st = sign_test(base, peak)
    print(f"\n{'='*80}\n  {label}")
    print(f"  baseline  n={bs['n']:>5}  survive={bs['survive']:.1f}%  per_trade={bs['per_trade']:+.3f}")
    print(f"  PEAK      n={ps['n']:>5}  survive={ps['survive']:.1f}%  per_trade={ps['per_trade']:+.3f}")
    print(f"  Δ survive = {ps['survive']-bs['survive']:+.1f}pp   Δ per_trade = {ps['per_trade']-bs['per_trade']:+.3f}")
    print(f"  PAIRED year-block bootstrap Δ per_trade: median {d_med:+.3f}  90%CI [{d_lo:+.3f}, {d_hi:+.3f}]  P(Δ>0)={frac_pos:.3f}")
    print(f"  Sign test (n_years={st['n_years']}): per_trade {st['pt_wins']}/{st['n_years']} p={st['pt_p']:.4f}  |  survival {st['sv_wins']}/{st['n_years']} p={st['sv_p']:.4f}")
    return dict(base=bs, peak=ps, d_lo=d_lo, d_med=d_med, d_hi=d_hi, frac_pos=frac_pos, sign=st)


if __name__ == "__main__":
    spx, vix = load_spx(), load_vix()
    vix_on_spx = vix.reindex(spx.index, method="ffill", limit=5)
    pct = vix_trailing_pct(vix_on_spx)

    run(spx, vix, pct, "TEST A — FULL SAMPLE 2005-2026 (canonical config DTE98/126d/p80)")
    run(spx, vix, pct, "POWER CHECK — 2019-01-01 .. 2026-01-14 (equity-cache window)",
        lo=pd.Timestamp("2019-01-01"), hi=pd.Timestamp("2026-01-14"))
