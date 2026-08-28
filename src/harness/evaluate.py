"""Timing-signal evaluation harness.

For each candidate signal in signals.parquet, bucket entry days by the signal's
look-ahead-safe trailing-252d percentile (quintiles), simulate realized tail-inclusive
SPX put-credit-spreads per bucket, and report per-trade edge / win-rate / annual ROC
with year-block-bootstrap CIs. Answers: does firing only in a signal's rich bucket beat
firing every day (floatgen analog) or every gate-open day (spxtrader analog)?

Bucketing uses trailing-252d percentile (rank of day T vs T-252..T-1, shift(1)) so the
bucket a day lands in is exactly what would have been known live — no in-sample boundary.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from harness.spread_sim import simulate_spreads, aggregate, gate_open_entry_dates

PANEL = "harness/signals.parquet"
SIGNALS = [
    "vix", "vix_vix3m_ratio", "vix3m_minus_vix", "vx_contango_pct",
    "cor1m", "cor3m", "cor_term_ratio", "cor1m_chg_5d",
    "vrp_21d", "skew", "pcc_index", "pcc_equity", "dspx", "vvix",
]
NB = 5  # quintiles


def trailing_pct(s: pd.Series, window=252, min_periods=126) -> pd.Series:
    """Rank of day T within the prior `window` days (T excluded via shift(1))."""
    def _rank(w):
        last = w[-1]                  # current day
        if np.isnan(last):
            return np.nan
        v = w[:-1]                    # prior window only
        v = v[~np.isnan(v)]
        if len(v) < min_periods:
            return np.nan
        return (v < last).sum() / len(v)
    return s.rolling(window + 1, min_periods=min_periods + 1).apply(_rank, raw=True)


def block_bootstrap_ci(df_trades: pd.DataFrame, n=1000, lo=5, hi=95, seed=0):
    """Year-block bootstrap on per-trade realized_pnl mean. Resamples whole calendar
    years with replacement to respect within-year trade clustering/overlap."""
    if len(df_trades) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    by_year = {y: g["realized_pnl"].values for y, g in df_trades.groupby(df_trades["entry_date"].dt.year)}
    years = list(by_year.keys())
    means = []
    for _ in range(n):
        pick = rng.choice(years, size=len(years), replace=True)
        vals = np.concatenate([by_year[y] for y in pick])
        means.append(vals.mean())
    return (float(np.percentile(means, lo)), float(np.percentile(means, hi)))


def eval_signal(sig, panel, universe_dates, spx, vix, label):
    pct = trailing_pct(panel[sig])
    rows = []
    base_dates = universe_dates
    base = simulate_spreads(base_dates, spx=spx, vix=vix)
    base_agg = aggregate(base)
    for q in range(NB):
        lo, hi = q / NB, (q + 1) / NB
        mask = (pct >= lo) & (pct < hi) if q < NB - 1 else (pct >= lo) & (pct <= hi)
        qdates = panel.index[mask.fillna(False)]
        qdates = qdates.intersection(pd.DatetimeIndex(base_dates))
        if len(qdates) < 10:
            rows.append((q, len(qdates), np.nan, np.nan, np.nan, (np.nan, np.nan)))
            continue
        tr = simulate_spreads(qdates, spx=spx, vix=vix)
        a = aggregate(tr)
        ci = block_bootstrap_ci(tr)
        rows.append((q, a["n_trades"], tr["realized_pnl"].mean(), a["win_rate"], a["annual_roc"], ci))
    return base_agg, base, rows


def main():
    panel = pd.read_parquet(PANEL)
    spx = pd.read_parquet("data_cache/yahoo_GSPC.parquet")["Close"]
    vix = pd.read_parquet("data_cache/yahoo_VIX.parquet")["Close"]

    all_days = pd.DatetimeIndex(panel.index)
    gate_days = pd.DatetimeIndex(gate_open_entry_dates())

    for label, universe in [("ALL-DAYS (floatgen analog)", all_days),
                            ("GATE-OPEN (spxtrader analog)", gate_days)]:
        base = simulate_spreads(universe, spx=spx, vix=vix)
        ba = aggregate(base)
        print(f"\n{'='*96}\n  UNIVERSE: {label}")
        print(f"  BASELINE fire-every-day: n={ba['n_trades']} per_trade={base['realized_pnl'].mean():+.3f} "
              f"wr={ba['win_rate']:.1f}% ann_roc={ba['annual_roc']:.0f}% maxDD={ba['max_drawdown']:.1f}")
        print(f"{'='*96}")
        print(f"  {'signal':<18} {'Q':<3} {'n':>5} {'per_trade':>10} {'CI90':>18} {'wr%':>6} {'ann_roc%':>9}  mono")
        summary = []
        for sig in SIGNALS:
            _, _, rows = eval_signal(sig, panel, universe, spx, vix, label)
            pts = [r[2] for r in rows if not pd.isna(r[2])]
            mono = ""
            if len(pts) >= 4:
                inc = all(pts[i] <= pts[i+1] for i in range(len(pts)-1))
                dec = all(pts[i] >= pts[i+1] for i in range(len(pts)-1))
                mono = "↑" if inc else "↓" if dec else ""
            for (q, n, pt, wr, roc, ci) in rows:
                cistr = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if not pd.isna(ci[0]) else "n/a"
                ptstr = f"{pt:+.3f}" if not pd.isna(pt) else "  n/a"
                wrstr = f"{wr:.1f}" if not pd.isna(wr) else "n/a"
                rocstr = f"{roc:.0f}" if not pd.isna(roc) else "n/a"
                m = mono if q == NB - 1 else ""
                print(f"  {sig:<18} Q{q+1:<2} {n:>5} {ptstr:>10} {cistr:>18} {wrstr:>6} {rocstr:>9}  {m}")
            # edge of top vs bottom bucket
            if len(pts) >= 2:
                summary.append((sig, pts[-1] - pts[0], mono, base["realized_pnl"].mean()))
            print()
        print(f"  {'-'*70}\n  Q5−Q1 per-trade edge (top minus bottom bucket), {label}:")
        for sig, edge, mono, basept in sorted(summary, key=lambda x: -abs(x[1])):
            print(f"    {sig:<18} Q5−Q1={edge:+.3f}  base_per_trade={basept:+.3f}  mono={mono or '·'}")


if __name__ == "__main__":
    main()
