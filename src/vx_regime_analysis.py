#!/usr/bin/env python3
"""
VX1-VX2 Contango/Backwardation Regime Analysis
Reconstructs the TradingView VX1!-VX2! chart and quantifies regime durations.

Usage:
    cd /Users/johnluo/Desktop/GitHub/SPX_Analysis/src
    python3 vx_regime_analysis.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

SRC = Path(__file__).parent
CSV_PATH   = SRC / "CBOE_Data_Archive" / "VX1-VX2.csv"
VX1_PQRT   = SRC / "data_cache" / "vx_continuous" / "VX1.parquet"
VX2_PQRT   = SRC / "data_cache" / "vx_continuous" / "VX2.parquet"
OUT_PNG     = SRC / "vx_regime_analysis.png"


# ── 1. Load & merge data ─────────────────────────────────────────────────────

def load_spread() -> pd.Series:
    csv = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date")["Close"]
    csv.name = "spread"

    vx1 = pd.read_parquet(VX1_PQRT)["settle"]
    vx2 = pd.read_parquet(VX2_PQRT)["settle"]
    pq  = (vx1 - vx2).rename("spread")
    pq.index = pd.to_datetime(pq.index)

    # extend CSV with parquet where parquet is newer
    cutoff    = csv.index[-1]
    extension = pq[pq.index > cutoff]
    combined  = pd.concat([csv, extension]).sort_index()
    combined  = combined[~combined.index.duplicated(keep="last")]
    return combined.dropna()


def load_ratio() -> pd.Series:
    vx1 = pd.read_parquet(VX1_PQRT)["settle"]
    vx2 = pd.read_parquet(VX2_PQRT)["settle"]
    ratio = (vx1 / vx2).rename("ratio")
    ratio.index = pd.to_datetime(ratio.index)
    return ratio.dropna()


# ── 2. Regime segmentation ───────────────────────────────────────────────────

def segment_regimes(spread: pd.Series, min_days: int = 1):
    """
    Returns a DataFrame of contiguous regime episodes.
    regime: 'contango' (spread < 0) | 'backwardation' (spread >= 0)
    """
    s = spread.copy()
    s_regime = (s >= 0).astype(int)  # 1=backwardation, 0=contango
    changes  = s_regime != s_regime.shift(1)
    group_id = changes.cumsum()

    episodes = []
    for gid, grp in s.groupby(group_id):
        regime    = "backwardation" if (grp >= 0).all() else "contango"
        start     = grp.index[0]
        end       = grp.index[-1]
        duration  = len(grp)
        peak_abs  = grp.abs().max()
        mean_val  = grp.mean()
        episodes.append({
            "regime":    regime,
            "start":     start,
            "end":       end,
            "days":      duration,
            "peak_abs":  peak_abs,
            "mean":      mean_val,
        })
    return pd.DataFrame(episodes)


# ── 3. Stats printing ────────────────────────────────────────────────────────

def print_stats(spread: pd.Series, episodes: pd.DataFrame):
    today_val = spread.iloc[-1]
    today_dt  = spread.index[-1].strftime("%Y-%m-%d")

    print(f"\n{'═'*70}")
    print(f"  VX1-VX2 REGIME ANALYSIS  (data through {today_dt})")
    print(f"{'═'*70}")
    print(f"  Current spread : {today_val:+.3f}  "
          f"({'BACKWARDATION' if today_val >= 0 else 'CONTANGO'})")
    print()

    for regime in ["contango", "backwardation"]:
        ep = episodes[episodes["regime"] == regime]
        if ep.empty:
            continue
        label = regime.upper()
        print(f"  {label} episodes  ({len(ep)} total)")
        print(f"  {'─'*60}")
        pct_days = ep["days"].sum() / len(spread) * 100
        print(f"    % of all trading days : {pct_days:.1f}%")
        print(f"    Median duration       : {ep['days'].median():.0f} td")
        print(f"    Mean   duration       : {ep['days'].mean():.0f} td")
        print(f"    Longest episode       : {ep['days'].max():.0f} td  "
              f"(started {ep.loc[ep['days'].idxmax(),'start'].strftime('%Y-%m-%d')})")
        print(f"    Shortest episode      : {ep['days'].min():.0f} td")
        print(f"    Median peak |spread|  : {ep['peak_abs'].median():.2f}")
        print(f"    Max    peak |spread|  : {ep['peak_abs'].max():.2f}")
        print()

    # backwardation → resolution speed
    bk = episodes[episodes["regime"] == "backwardation"].copy()
    if not bk.empty:
        print(f"  BACKWARDATION RESOLUTION")
        print(f"  {'─'*60}")
        short_bk = bk[bk["days"] <= 10]
        medium_bk = bk[(bk["days"] > 10) & (bk["days"] <= 30)]
        long_bk   = bk[bk["days"] > 30]
        print(f"    ≤ 10 td (flash):  {len(short_bk)} episodes  ({len(short_bk)/len(bk)*100:.0f}%)")
        print(f"    11-30 td:         {len(medium_bk)} episodes  ({len(medium_bk)/len(bk)*100:.0f}%)")
        print(f"    > 30 td:          {len(long_bk)} episodes  ({len(long_bk)/len(bk)*100:.0f}%)")
        print()

    # current contango streak
    last_ep = episodes.iloc[-1]
    if last_ep["regime"] == "contango":
        since = last_ep["start"].strftime("%Y-%m-%d")
        print(f"  CURRENT CONTANGO STREAK: {last_ep['days']} td (since {since})")
        longer = episodes[(episodes["regime"] == "contango") & (episodes["days"] > last_ep["days"])]
        print(f"  Historical contango episodes longer than current: {len(longer)}")
        if len(longer):
            top = longer.nlargest(3, "days")[["start","end","days"]]
            for _, r in top.iterrows():
                print(f"    {r['start'].strftime('%Y-%m-%d')} → {r['end'].strftime('%Y-%m-%d')}  "
                      f"({r['days']} td)")
    print(f"{'═'*70}\n")


# ── 4. Plotting ──────────────────────────────────────────────────────────────

def plot(spread: pd.Series, ratio: pd.Series, episodes: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")

    # ── TOP: raw spread with colored bars ──
    dates = spread.index
    vals  = spread.values
    colors = np.where(vals >= 0, "#00d4a8", "#e84040")  # teal=backwardation, red=contango

    ax1.bar(dates, vals, color=colors, width=1.2, alpha=0.85)
    ax1.axhline(0, color="#555555", linewidth=0.8, linestyle="--")

    # shade deepest contango percentile
    p10 = np.percentile(vals[vals < 0], 10)
    ax1.axhline(p10, color="#ff6060", linewidth=0.6, linestyle=":")
    ax1.text(dates[-1], p10 - 0.08, f" 10th pct\n {p10:.2f}", color="#ff6060",
             fontsize=7, va="top", ha="right")

    # mark current
    cur_val = vals[-1]
    ax1.scatter([dates[-1]], [cur_val], color="white", s=40, zorder=5)
    ax1.annotate(f"Now: {cur_val:+.2f}", xy=(dates[-1], cur_val),
                 xytext=(-60, 20), textcoords="offset points",
                 color="white", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="white", lw=0.8))

    ax1.set_ylabel("VX1 − VX2  (points)", color="#aaaaaa")
    ax1.set_title("VX1! − VX2!  |  Contango (−) / Backwardation (+)",
                  color="white", fontsize=13, pad=10)

    legend_els = [
        Patch(facecolor="#e84040", label="Contango (normal / benign)"),
        Patch(facecolor="#00d4a8", label="Backwardation (stress)"),
    ]
    ax1.legend(handles=legend_els, facecolor="#1a1a2e", labelcolor="white",
               fontsize=9, loc="upper left")

    # ── BOTTOM: log(VX1/VX2) ratio — level-normalized ──
    ratio_aligned = ratio.reindex(spread.index).dropna()
    log_ratio = np.log(ratio_aligned)

    log_colors = np.where(log_ratio.values >= 0, "#00d4a8", "#e84040")
    ax2.bar(log_ratio.index, log_ratio.values, color=log_colors, width=1.2, alpha=0.85)
    ax2.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("ln(VX1/VX2)", color="#aaaaaa")
    ax2.set_title("Log-Ratio (level-normalized — equal weight across high/low VIX regimes)",
                  color="#aaaaaa", fontsize=9, pad=6)

    # x-axis formatting
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Chart saved → {OUT_PNG}")
    return fig


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading spread data...")
    spread = load_spread()
    ratio  = load_ratio()
    print(f"  {len(spread)} trading days  "
          f"({spread.index[0].strftime('%Y-%m-%d')} → {spread.index[-1].strftime('%Y-%m-%d')})")

    episodes = segment_regimes(spread)
    print_stats(spread, episodes)

    print("Generating chart...")
    plot(spread, ratio, episodes)
    print("Done.")


if __name__ == "__main__":
    main()
