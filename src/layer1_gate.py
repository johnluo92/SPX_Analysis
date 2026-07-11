#!/usr/bin/env python3
"""
Layer 1 Rule-Based Gate — Environmental Classification
GO/NO-GO gate for SPX spread selling.

Usage:
  python layer1_gate.py              # current gate status
  python layer1_gate.py --backtest   # + historical analysis
  python layer1_gate.py --as-of 2023-01-15   # gate status on a past date
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def _load(filename, col, ff_limit=5, target_index=None):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_parquet(path)
    s = df[col]
    if target_index is not None:
        s = s.reindex(target_index, method="ffill", limit=ff_limit)
    return s


def _percentile_63d(s: pd.Series) -> pd.Series:
    """Rolling 63d percentile rank of s (0–100). Matches feature_engineer logic."""
    def _pct_rank(window):
        valid = window.dropna()
        if len(valid) < 44:  # 63 * 0.7
            return np.nan
        last = window.iloc[-1]
        if pd.isna(last):
            return np.nan
        return (valid < last).sum() / len(valid) * 100
    return s.rolling(64).apply(_pct_rank, raw=False)


def build_gate_features(as_of: str = None) -> pd.DataFrame:
    """Compute all gate condition features from cached parquet data."""
    spx_raw = pd.read_parquet(os.path.join(DATA_DIR, "yahoo_GSPC.parquet"))
    idx = spx_raw.index
    if as_of:
        idx = idx[idx <= as_of]

    spx = spx_raw["Close"].reindex(idx, method="ffill", limit=5)
    vix  = _load("yahoo_VIX.parquet",   "Close",              target_index=idx)
    vvix = _load("yahoo_VVIX.parquet",  "Close",              target_index=idx)
    vx1  = _load("vx_continuous/VX1.parquet", "settle",       target_index=idx)
    vx2  = _load("vx_continuous/VX2.parquet", "settle",       target_index=idx)
    vx3  = _load("vx_continuous/VX3.parquet", "settle",       target_index=idx)
    vx4  = _load("vx_continuous/VX4.parquet", "settle",       target_index=idx)
    stress = _load("fred_STLFSI4.parquet",    "STLFSI4",     ff_limit=90, target_index=idx)

    try:
        hy_bb  = _load("fred_BAMLH0A1HYBB.parquet", "BAMLH0A1HYBB", ff_limit=45, target_index=idx)
        hy_ccc = _load("fred_BAMLH0A3HYC.parquet",  "BAMLH0A3HYC",  ff_limit=45, target_index=idx)
        credit_available = True
    except Exception:
        credit_available = False

    f = pd.DataFrame(index=idx)

    # C1: VX term structure in contango — (VX2/VX1 - 1) × 100
    f["vx_contango_pct"] = (vx2 / vx1.replace(0, np.nan) - 1) * 100

    # C2: VRP elevated — VIX minus 21d realized vol > +2 pts
    rv_21d = spx.pct_change().rolling(21).std() * np.sqrt(252) * 100
    f["vix_vs_rv_21d"] = vix - rv_21d

    # C3: VVIX not extreme — adaptive rolling 252-day trailing p85 of VVIX.
    # min_periods=126: require at least half a year of history before computing threshold.
    # .shift(1) ensures current day's VVIX is not in its own threshold computation (look-ahead-free).
    # Threshold auto-adjusts as VVIX regime shifts — no manual recalibration needed.
    vvix_rolling_p85 = vvix.rolling(252, min_periods=126).quantile(0.85).shift(1)
    f["vvix"] = vvix
    f["c3_vvix_threshold"] = vvix_rolling_p85

    # C4: Credit spreads not widening
    if credit_available:
        quality_spread = hy_ccc - hy_bb
        f["credit_widening_regime"] = (
            quality_spread > quality_spread.rolling(63).quantile(0.8)
        ).astype(float)
    else:
        f["credit_widening_regime"] = 0.0

    # C5: Financial stress benign (regime ≤ 1 means STL FSI < 0)
    f["stress_raw"] = stress
    f["stress_regime"] = pd.cut(
        stress,
        bins=[-10, -1, 0, 2, 20],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # VVIX/VIX ratio stored for display context only — not a gate condition.
    # Dropped as C6: ratio is regime-conditional (R0 mean=6.44, R1 mean=4.87) so a single
    # full-history threshold (p85=6.664) fires on 32.6% of R0 days and 0.9% of R1 days —
    # blocking precisely the low-VIX high-VRP environments we want to enter. VRP when ratio
    # fires (3.87 pts) exceeds VRP when it passes (3.40 pts). C3 covers the absolute VVIX
    # danger case. If a regime-conditional ratio threshold is added later, it belongs here.
    vvix_vix_ratio = vvix / vix.where(vix >= 5, other=np.nan)
    f["vvix_vix_ratio"] = vvix_vix_ratio

    # C7: Uniform VX contango through VX3 — extends C1 (VX2/VX1 only).
    # A kinked curve (VX1<VX2 but VX2≥VX3) passes C1 but signals unstable regime transition.
    # Treat missing VX3 as NO-GO (NaN → 0.0 in evaluate_gate).
    f["vx_curve_ok"] = np.where(
        vx3.isna(),
        np.nan,
        ((vx2 > vx1) & (vx3 > vx2)).astype(float),
    )

    # Continuous curve health score: weighted sum of normalized VX slopes.
    # Used for display context in print_current_status — not a gate condition.
    slope1 = (vx2 - vx1) / vx1.replace(0, np.nan)
    slope2 = (vx3 - vx2) / vx1.replace(0, np.nan)
    slope3 = (vx4 - vx3) / vx1.replace(0, np.nan)
    s_full    = 0.50 * slope1 + 0.30 * slope2 + 0.20 * slope3
    s_partial = (0.50 * slope1 + 0.30 * slope2) / 0.80  # rescaled when VX4 missing
    f["vx_curve_score"] = s_full.where(~vx4.isna(), s_partial)

    return f


def evaluate_gate(f: pd.DataFrame) -> pd.Series:
    """Apply C1–C5 → boolean GO series. VVIX/VIX ratio DP block removed 2026-06-01 (pct>85
    episodes empirically safer for DP). VX curve (vx_curve_ok) computed in build_gate_features()
    for display context only — never in this gate verdict.

    C3: VVIX < trailing 252-day p85 of VVIX (adaptive; recalibrates daily).
    """
    c1 = f["vx_contango_pct"] > 0
    c2 = f["vix_vs_rv_21d"] > 2
    c3 = f["vvix"] < f["c3_vvix_threshold"]
    c4 = f["credit_widening_regime"] == 0
    c5 = f["stress_regime"] <= 1
    return (c1 & c2 & c3 & c4 & c5).rename("gate_open")


def _vrp_rich_now(vix, vrp_window=252, vrp_cut=0.67):
    """VRP-rich flag = today's variance-risk-premium (VIX − 21d realized vol, the C2 definition,
    build_gate_features:76) in the top tercile of its trailing 252d. BT-VOLPEAK Test C
    (2026-07-10): PEAK ∧ VRP-rich is the strongest cell — survive 96.6%, +2.28/trade (vs PEAK
    alone 93.4% / +1.98), OOS-stable (fit 97.1% / val 96.1%). Returns (rich_bool, pctile) or
    (None, None) if SPX/history unavailable."""
    try:
        spx = pd.read_parquet(os.path.join(DATA_DIR, "yahoo_GSPC.parquet"))["Close"].sort_index()
        rv21 = spx.pct_change().rolling(21).std() * np.sqrt(252) * 100
        vrp = (vix - rv21.reindex(vix.index, method="ffill", limit=5)).dropna()
        w = vrp.tail(vrp_window)
        if len(w) < 126:
            return None, None
        cur = float(w.iloc[-1])
        prior = w.iloc[:-1]
        p = float((prior < cur).sum() / len(prior))
        return (p >= vrp_cut), round(p, 3)
    except Exception:
        return None, None


def vol_regime_state(lookback=126, peak_cut=0.80, trough_cut=0.20, swan_vix=30.0):
    """BT-VOLPEAK deployment-timing tilt (SSOT). Rank of today's VIX within its trailing
    `lookback` days (incl. today) — the look-ahead-safe percentile validated 2026-07-10:
    LONG-DTE (98-180d) put spreads sold at a local vol MAXIMUM survive better (88.9%→93.4%,
    +0.46/trade, 20/22 yrs, paired-Δ 90%CI [+0.19,+0.76] excludes 0). A PEAK that is ALSO
    VRP-rich is the strongest cell (96.6% / +2.28, Test C). At a TROUGH the lean-in window is
    shut. Tilts WHEN the ES budget deploys, never its size. Returns {} on failure (never blocks)."""
    try:
        vix = pd.read_parquet(os.path.join(DATA_DIR, "yahoo_VIX.parquet"))["Close"].dropna()
        w = vix.tail(lookback)
        if len(w) < int(lookback * 0.6):
            return {}
        cur = float(w.iloc[-1])
        prior = w.iloc[:-1]
        pct = float((prior < cur).sum() / len(prior))
        swan = cur >= swan_vix
        vrp_rich, vrp_pct = _vrp_rich_now(vix)
        if pct >= peak_cut:
            state = "PEAK"
            tilt = "SWAN-WATCH" if swan else ("LEAN-IN-STRONG" if vrp_rich else "LEAN-IN")
        elif pct <= trough_cut:
            state, tilt = "TROUGH", "HOLD-BACK"
        else:
            state, tilt = "NORMAL", "standard"
        return {"vix": round(cur, 2), "pct_126d": round(pct, 3), "state": state,
                "tilt": tilt, "swan": bool(swan), "vrp_rich": vrp_rich, "vrp_pct": vrp_pct,
                "asof": str(vix.index[-1].date())}
    except Exception:
        return {}


def print_current_status(f: pd.DataFrame, go: pd.Series):
    last = f.index[-1]
    row = f.loc[last]
    gate_open = bool(go.iloc[-1])
    status = "GO  ✓" if gate_open else "NO-GO ✗"

    print(f"\n{'='*58}")
    print(f"  Layer 1 Gate — {last.date()}")
    print(f"  Status: {status}")
    print(f"{'='*58}")

    vvix_ratio = row["vvix_vix_ratio"]
    vx_curve = row["vx_curve_ok"]
    curve_score = row["vx_curve_score"]

    c6_pass = (not pd.isna(vx_curve)) and (vx_curve == 1)
    c6_display = (
        f"score={curve_score:.3f}" if not pd.isna(curve_score) else "VX3 STALE — NO-GO"
    )
    ratio_display = f"ratio={vvix_ratio:.3f}" if not pd.isna(vvix_ratio) else "n/a"

    checks = [
        ("C1 VX Contango > 0%",
         row["vx_contango_pct"] > 0,
         f"{row['vx_contango_pct']:+.2f}%"),
        ("C2 VIX vs RV21d > +2 pts",
         row["vix_vs_rv_21d"] > 2,
         f"{row['vix_vs_rv_21d']:+.2f} pts"),
        ("C3 VVIX < rolling p85",
         row["vvix"] < row["c3_vvix_threshold"],
         f"VVIX={row['vvix']:.1f}  threshold={row['c3_vvix_threshold']:.1f} (252d p85)"),
        ("C4 Credit widening = 0",
         row["credit_widening_regime"] == 0,
         f"regime={row['credit_widening_regime']:.0f}"),
        ("C5 Stress regime ≤ 1",
         row["stress_regime"] <= 1,
         f"regime={row['stress_regime']:.0f}  (raw={row['stress_raw']:.3f})"),
        ("C6 VX1 < VX2 < VX3", c6_pass, c6_display),
    ]

    for label, passed, display in checks:
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] {label:<30} {display}")
    print(f"  [·] VVIX/VIX (context only)         {ratio_display}  [p85=6.664, R0-norm=6.44]")

    # display only — not in gate verdict; Q2/Q3 boundary at 8% (dTR +14pp)
    ctg = row["vx_contango_pct"]
    if pd.isna(ctg):
        ctg_label = "n/a"
    elif ctg >= 8:
        ctg_label = "[strong]"
    elif ctg >= 4:
        ctg_label = "[moderate]"
    else:
        ctg_label = "[weak / backwardation]"
    ctg_display = f"{ctg:+.1f}%" if not pd.isna(ctg) else "n/a"
    print(f"  [·] Contango magnitude (VX2/VX1−1)  {ctg_display}  {ctg_label:<22}  (flag < 8%)")
    print()


def print_backtest(f: pd.DataFrame, go: pd.Series):
    valid_mask = go.notna()
    valid_go = go[valid_mask]
    total = len(valid_go)
    open_days = int(valid_go.sum())
    pct = open_days / total * 100 if total else 0.0

    open_annual = go.resample("YE").sum()
    count_annual = go.resample("YE").count()
    annual_pct = (open_annual / count_annual * 100).rename("pct")

    print(f"{'='*58}")
    print(f"  Layer 1 Gate — Historical Backtest")
    print(f"  {valid_go.index[0].date()} → {valid_go.index[-1].date()}")
    print(f"{'='*58}")
    print(f"  Overall: {open_days} / {total} days open  ({pct:.1f}%)")
    print()
    print(f"  {'Year':<8} {'Open':>6} {'Total':>6} {'Pct':>7}")
    print(f"  {'-'*32}")
    for year_end in open_annual.index:
        n_open = open_annual[year_end]
        n_total = count_annual[year_end]
        if n_total < 5:
            continue
        yr_pct = n_open / n_total * 100
        print(f"  {year_end.year:<8} {n_open:>6.0f} {n_total:>6.0f} {yr_pct:>6.1f}%")
    print()

    cond_stats = {
        "C1 VX Contango > 0%":    (f["vx_contango_pct"] > 0).mean() * 100,
        "C2 VIX vs RV21d > +2":   (f["vix_vs_rv_21d"] > 2).mean() * 100,
        "C3 VVIX < rolling p85":   (f["vvix"] < f["c3_vvix_threshold"]).mean() * 100,
        "C4 Credit widening = 0":  (f["credit_widening_regime"] == 0).mean() * 100,
        "C5 Stress regime ≤ 1":    (f["stress_regime"] <= 1).mean() * 100,
        "C6 VX1 < VX2 < VX3 [display only, not in gate verdict]": (f["vx_curve_ok"].fillna(0) == 1).mean() * 100,
    }
    print("  Condition pass rates (% of valid days):")
    for name, rate in cond_stats.items():
        bar = "█" * int(rate / 5)
        print(f"    {name:<28} {rate:>5.1f}%  {bar}")
    print()

    # Consecutive dry-spell analysis
    go_int = go.astype(float)
    no_go_runs = []
    run = 0
    for v in go_int:
        if v == 0 or pd.isna(v):
            run += 1
        else:
            if run > 0:
                no_go_runs.append(run)
            run = 0
    if run > 0:
        no_go_runs.append(run)

    if no_go_runs:
        print(f"  NO-GO streaks: max={max(no_go_runs)}d  median={int(np.median(no_go_runs))}d  "
              f"p75={int(np.percentile(no_go_runs, 75))}d")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 rule-based GO/NO-GO gate for SPX spread selling"
    )
    parser.add_argument("--backtest", action="store_true",
                        help="Print historical gate open rates")
    parser.add_argument("--as-of", default=None, dest="as_of",
                        help="Evaluate gate as of this date (YYYY-MM-DD)")
    args = parser.parse_args()

    f = build_gate_features(as_of=args.as_of)
    go = evaluate_gate(f)

    # Drop warmup rows before output (63d for percentile)
    warmup = 63
    f_trim = f.iloc[warmup:]
    go_trim = go.iloc[warmup:]

    print_current_status(f_trim, go_trim)

    if args.backtest:
        print_backtest(f_trim, go_trim)


if __name__ == "__main__":
    main()
