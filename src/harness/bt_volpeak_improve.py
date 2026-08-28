"""Test C — can we improve EV beyond the binary VIX-peak, at the operative DTE98 tenor,
index-level (the equity sample can't support conditioning)?

Pre-registered (before seeing numbers):
  H1  Continuous response: does per-trade edge rise monotonically with the VIX-126d
      percentile? If yes, a percentile->size tilt (deploy more budget at higher pctile,
      within ES caps) dominates the binary PEAK cut.
  H2  VIX-peak x vrp_21d-HIGH (top tercile): richer VRP at a peak -> better.
  H3  VIX-peak x cor1m-LOW (bottom tercile): John flagged cor1m/cor3m. First CHECK
      overlap — cor1m tends HIGH at vol peaks, so cor1m-low & VIX-peak may be near-exclusive.

Guards against mining: fit 2005-2015 / validate 2016-2026; report both. A combo only
"improves" if it beats plain PEAK on per-trade AND holds sign out-of-sample.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.spread_sim import simulate_spreads, load_spx, load_vix
from harness.bt_volpeak_rigor import vix_trailing_pct, DTE98, PEAK_CUT

PANEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.parquet")
SPLIT = pd.Timestamp("2016-01-01")


def _pt(dates, spx, vix):
    if len(dates) == 0:
        return (0, float("nan"), float("nan"))
    df = simulate_spreads(dates, spx=spx, vix=vix, r0_params=DTE98, r1_params=DTE98)
    if len(df) == 0:
        return (0, float("nan"), float("nan"))
    return (len(df), float(df["otm_survived"].mean() * 100), float(df["realized_pnl"].mean()))


def _row(label, dates, spx, vix):
    for tag, lo, hi in [("FULL", None, None), ("FIT<16", None, SPLIT), ("VAL>=16", SPLIT, None)]:
        d = dates
        if lo is not None:
            d = d[d >= lo]
        if hi is not None:
            d = d[d < hi]
        n, sv, pt = _pt(d, spx, vix)
        print(f"    {label:<26} {tag:<9} n={n:>5}  survive={sv:>5.1f}%  per_trade={pt:>+6.3f}")


def main():
    spx, vix = load_spx(), load_vix()
    vix_on_spx = vix.reindex(spx.index, method="ffill", limit=5)
    pct = vix_trailing_pct(vix_on_spx)
    panel = pd.read_parquet(PANEL)

    days = pd.DatetimeIndex(spx.index)
    pct_d = pct.reindex(days)
    peak = pct_d >= PEAK_CUT

    # align panel signals to spx days (look-ahead-safe trailing-252d percentiles)
    def tpct(col, w=252, mp=126):
        s = panel[col].reindex(days).ffill()
        return s.rolling(w, min_periods=mp).apply(lambda x: (x[:-1][~np.isnan(x[:-1])] < x[-1]).mean(), raw=True)
    vrp_p = tpct("vrp_21d")
    cor1m_p = tpct("cor1m")

    print("=" * 84)
    print("  TEST C — improving on binary VIX-peak at DTE98 (index-level)")
    print("=" * 84)

    # --- H3 overlap check FIRST ---
    pk = peak.fillna(False)
    cor_lo = (cor1m_p < 0.33).reindex(days).fillna(False)
    vrp_hi = (vrp_p >= 0.67).reindex(days).fillna(False)
    both_cor = int((pk & cor_lo).sum())
    print(f"\n  OVERLAP check (H3): PEAK days={int(pk.sum())}  cor1m-LOW days={int(cor_lo.sum())}  "
          f"PEAK&cor1m-LOW={both_cor}")
    print(f"    mean cor1m-pctile on PEAK days = {float(cor1m_p.reindex(days)[pk].mean()):.3f} "
          f"(vs {float(cor1m_p.reindex(days).mean()):.3f} overall) -> "
          f"{'cor1m runs HIGH at peaks; cor1m-LOW x PEAK near-exclusive' if float(cor1m_p.reindex(days)[pk].mean())>0.5 else 'compatible'}")

    print("\n  BASELINE + PEAK (reference):")
    _row("all-days baseline", days, spx, vix)
    _row("PEAK (p126>=.80)", days[pk.values], spx, vix)

    # --- H1 continuous: edge by VIX-percentile band ---
    print("\n  H1 continuous — per-trade by VIX-126d percentile band:")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        band = ((pct_d >= lo) & (pct_d < hi)).fillna(False)
        _row(f"pctile[{lo:.1f},{hi:.1f})", days[band.values], spx, vix)

    # --- H2 / H3 combos ---
    print("\n  H2 — PEAK x vrp_21d-HIGH (top tercile):")
    _row("PEAK & vrp-HIGH", days[(pk & vrp_hi).values], spx, vix)
    print("\n  H3 — PEAK x cor1m-LOW (bottom tercile):")
    _row("PEAK & cor1m-LOW", days[(pk & cor_lo).values], spx, vix)


if __name__ == "__main__":
    main()
