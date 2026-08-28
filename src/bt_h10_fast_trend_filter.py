"""
BT-33 (H10): can a FASTER price trend filter beat the 200dma skip (BT-23) on the DP down-years?
(John 2026-06-19: price is the most timely leading indicator — BT-32 rejected macro leads — so the
only avenue left is a finer/faster PRICE filter. He also set the frame: the system is decision
SUPPORT, not auto-trading — this measures which trend-state flag gives the most informative
down-size signal.)

BT-23 found the 200dma skip cuts worst-YEAR 76% but did NOT reduce peak maxDD, because the 2022
trough is seeded by entries opened BEFORE price broke the 200dma — a slow filter lets them through.
Hypothesis: a faster filter (shorter MA, or drawdown-from-high, or short-horizon momentum) catches
those earlier and actually reduces maxDD, at acceptable EV cost.

Filters tested (skip the DP entry if, at entry, live-knowable from SPX close only):
  ma_W     SPX < its W-day SMA           W in {20,50,100,150,200}   (200 = BT-23 baseline)
  dd_X     SPX < (1-X) * trailing 252d max (drawdown-from-high)     X in {3%,5%,10%}
  mom_N    SPX N-day return < 0          N in {20,50}
DP book = BT-21 (build_dp, GO-day 2016+, 50-wide). Realized-by-expiry DD (BT-24 method), so this is
comparable to BT-24/27/28 and to the 200dma row.

PRE-REGISTERED DECISION RULE (written before running):
  A filter "beats 200dma" iff it reduces realized maxDD >=15% BELOW the 200dma's maxDD AND keeps
  total P&L within 10% of the 200dma's total. (BT-23 already showed 200dma fixes worst-YEAR but not
  maxDD; the open prize is maxDD. A faster filter that cuts maxDD without gutting EV is the win.) If
  no filter cuts maxDD materially below 200dma, peak drawdown is confirmed un-fixable by entry
  trend-timing -> it is a DEPLOYMENT-SIZE decision (BT-24), and the 200dma stays as the worst-year
  tool. Report skip%, EV/DD, and how each filter scores the 2018/2022 entries.
CAVEATS: 2016+ (n=2 loss years — same hard limit as BT-32; do not over-read a single maxDD number).
  Skip = full size-down to 0. Faster filters skip MORE entries (EV cost) and whipsaw MORE (skip a
  dip that recovers). Thresholds conventional, not optimized (optimizing on n=2 = overfit).
"""
import os
import numpy as np
import pandas as pd
from bt_h2_dp_sizing import build_dp
from bt_dp_gate_pnl import DATA


def filters_at(dates):
    spx = pd.read_parquet(os.path.join(DATA, "yahoo_GSPC.parquet"))["Close"].sort_index()
    out = {}
    for w in [20, 50, 100, 150, 200]:
        ma = spx.rolling(w).mean()
        out[f"ma_{w}"] = (spx < ma).reindex(dates).fillna(False).astype(bool)
    tmax = spx.rolling(252).max()
    for x in [0.03, 0.05, 0.10]:
        out[f"dd_{int(x*100)}"] = (spx < (1 - x) * tmax).reindex(dates).fillna(False).astype(bool)
    for n in [20, 50]:
        out[f"mom_{n}"] = (spx.pct_change(n) < 0).reindex(dates).fillna(False).astype(bool)
    return out


def realized_dd(df, col="pnl"):
    s = df.sort_values("expiry")
    cum = s[col].cumsum().to_numpy()
    return float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0


def report(markup):
    dp = build_dp(50, markup).sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(dp.date)
    flt = filters_at(dates)
    base_total = dp.pnl.sum()
    base_worst = dp.groupby("year").pnl.sum().min()
    base_dd = realized_dd(dp)

    # 200dma reference (BT-23)
    ref_keep = dp[~flt["ma_200"].values]
    ref_total = ref_keep.pnl.sum()
    ref_dd = realized_dd(ref_keep)

    print(f"\n{'='*102}")
    print(f"  BT-33 / H10 — faster price trend filters vs 200dma, DP book, markup {markup:.2f}")
    print(f"{'='*102}")
    print(f"  no-skip: total ${base_total:+,.0f}  worst-yr ${base_worst:+,.0f}  maxDD ${-base_dd:,.0f}")
    print(f"  200dma (BT-23 ref): total ${ref_total:+,.0f}  maxDD ${-ref_dd:,.0f}")
    print(f"  {'filter':>8} {'skip%':>6} {'total P&L':>12} {'%vs200':>7} {'worst-yr':>11} "
          f"{'2018':>10} {'2022':>10} {'maxDD':>11} {'EV/DD':>7} {'beats200':>9}")
    print(f"  {'-'*100}")
    for name, w in flt.items():
        keep = dp[~w.values]
        total = keep.pnl.sum()
        by = keep.groupby("year").pnl.sum()
        dd = realized_dd(keep)
        ratio = total / dd if dd else float("inf")
        beats = (dd <= ref_dd * 0.85 and total >= ref_total * 0.90)
        tag = "BEATS" if (name != "ma_200" and beats) else ("ref" if name == "ma_200" else "-")
        print(f"  {name:>8} {w.mean()*100:>5.0f}% ${total:>+10,.0f} {total/ref_total*100:>6.0f}% "
              f"${by.min():>+10,.0f} ${by.get(2018,0):>+9,.0f} ${by.get(2022,0):>+9,.0f} "
              f"${-dd:>+10,.0f} {ratio:>6.2f} {tag:>9}")
    print(f"\n  beats200 = maxDD >=15% below 200dma's AND total within 10% of 200dma's.")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
