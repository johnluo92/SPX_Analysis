"""
BT-29 (H8a): does gate CLOSURE (GO -> NO-GO) actually predict downside? This is the PREMISE gate
for any gate-triggered hedge (John 2026-06-19: "find ways to detect an onslaught incoming ... so
things don't completely fall apart when the chickens come home to roost"). BT-22 already killed
SHORT-premium DN-as-hedge on TIMING (DN wins land ~45d after the DP trough). The new idea is a
LONG-premium hedge (puts / put debit spreads) deployed AT gate-close, time-aligned to the fall.
But a hedge can only work if the trigger has forward-downside edge. If GO->NO-GO does not precede
worse-than-baseline drawdowns, no instrument bought on it can pay for itself. Test premise first
(same discipline that killed H4 before pricing anything).

Method: cleaned Layer-1 gate (build_gate_features / evaluate_gate), valid window 2013+. A CLOSURE
= NO-GO today, GO yesterday. The gate flickers (~12 closures/yr), so also test DEBOUNCED closures
(first closure after >=ARM consecutive GO days) to isolate "real" regime breaks from chatter.
For each closure and each baseline day, measure SPX forward over h trading days:
  - fwd return  = close[t+h]/close[t]-1
  - fwd max drawdown = min over k<=h of close[t+k]/close[t]-1   (the worst dip in the window —
    what a long put would monetize)
Compare closure distribution vs ALL-valid-day baseline. Edge exists iff closures show materially
more negative forward returns AND deeper forward drawdowns, with a closure-block-bootstrap CI on
the drawdown gap that excludes 0.

PRE-REGISTERED DECISION RULE (written before running):
  Gate-close has hedgeable downside edge iff at some horizon h in {5,10,21,44}: (a) mean fwd
  max-drawdown after closure is at least 1.5x the baseline mean fwd max-drawdown, AND (b) the
  closure-block-bootstrap 95% CI on (closure mean DD - baseline mean DD) excludes 0. If the edge
  is absent or marginal, a per-closure hedge cannot beat its whipsaw cost -> do not build it; the
  gate's value is throttling DP entry COUNT (BT-04), not signalling a tradable crash.
  DEBOUNCED closures are the fair test of John's "real onslaught" idea; raw closures show the
  whipsaw tax. Report both.
CAVEATS: forward returns use SPX close only (leak-free). 2013+ (C1 VX start). Overlapping windows
  autocorrelated -> block bootstrap by closure episode. This tests the SIGNAL; instrument P&L
  (put cost, spread payoff) is H8b, only if this premise passes.
"""
import numpy as np
import pandas as pd
import os
from layer1_gate import build_gate_features, evaluate_gate
from bt_dp_gate_pnl import DATA

HORIZONS = [5, 10, 21, 44]
ARM = 10                 # debounce: a "real" closure follows >=ARM consecutive GO days
RNG = np.random.default_rng(20260619)
N_BOOT = 3000


def gate_series():
    f = build_gate_features()
    valid = (f["vx_contango_pct"].notna() & f["vix_vs_rv_21d"].notna()
             & f["vvix"].notna() & f["c3_vvix_threshold"].notna()
             & f["credit_widening_regime"].notna() & f["stress_regime"].notna())
    g = evaluate_gate(f)[valid].astype(bool)
    return g


def closures(g, debounce):
    prev = g.shift(1).fillna(False).astype(bool)
    close = g.eq(False) & prev.eq(True)
    days = list(g.index[close])
    if not debounce:
        return days
    # keep only closures preceded by >=ARM consecutive GO days
    out = []
    gi = g.index
    for d in days:
        loc = gi.get_loc(d)
        run = 0
        for k in range(loc - 1, -1, -1):
            if g.iloc[k]:
                run += 1
            else:
                break
        if run >= ARM:
            out.append(d)
    return out


def fwd_stats(spx, dates, h):
    rets, dds = [], []
    sidx = spx.index
    for d in dates:
        if d not in sidx:
            continue
        loc = sidx.get_loc(d)
        if loc + h >= len(sidx):
            continue
        path = spx.iloc[loc:loc + h + 1].to_numpy()
        rel = path / path[0] - 1.0
        rets.append(rel[-1])
        dds.append(rel[1:].min())
    return np.array(rets), np.array(dds)


def boot_dd_gap(spx, close_dates, base_dates, h):
    c_r, c_dd = fwd_stats(spx, close_dates, h)
    b_r, b_dd = fwd_stats(spx, base_dates, h)
    if len(c_dd) == 0 or len(b_dd) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    obs = c_dd.mean() - b_dd.mean()
    diffs = []
    for _ in range(N_BOOT):
        cs = RNG.choice(c_dd, size=len(c_dd), replace=True)
        bs = RNG.choice(b_dd, size=len(b_dd), replace=True)
        diffs.append(cs.mean() - bs.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, lo, hi, c_dd.mean(), b_dd.mean()


def report():
    g = gate_series()
    spx = pd.read_parquet(os.path.join(DATA, "yahoo_GSPC.parquet"))["Close"].sort_index()
    base = list(g.index)
    for label, deb in [("RAW closures (every GO->NO-GO)", False),
                       (f"DEBOUNCED closures (>= {ARM} GO days first)", True)]:
        cl = closures(g, deb)
        print(f"\n{'='*96}")
        print(f"  BT-29 / H8a — gate-close forward downside: {label}")
        print(f"{'='*96}")
        print(f"  {len(cl)} closures, {g.index.min().date()}-{g.index.max().date()}; "
              f"baseline = {len(base)} valid days")
        print(f"  {'horizon':>8} {'cl mean ret':>12} {'base ret':>10} {'cl mean DD':>11} "
              f"{'base DD':>9} {'DD ratio':>9} {'DD gap 95% CI':>22} {'edge':>6}")
        print(f"  {'-'*92}")
        for h in HORIZONS:
            c_r, c_dd = fwd_stats(spx, cl, h)
            b_r, b_dd = fwd_stats(spx, base, h)
            obs, lo, hi, cm, bm = boot_dd_gap(spx, cl, base, h)
            ratio = cm / bm if bm else float("nan")
            edge = "YES" if (ratio >= 1.5 and not (lo <= 0 <= hi)) else "no"
            print(f"  {h:>7}d {c_r.mean()*100:>+11.2f}% {b_r.mean()*100:>+9.2f}% "
                  f"{cm*100:>+10.2f}% {bm*100:>+8.2f}% {ratio:>8.2f}x "
                  f"[{lo*100:>+5.2f}%,{hi*100:>+5.2f}%] {edge:>6}")
        print(f"\n  EDGE = closure mean fwd max-DD >= 1.5x baseline AND DD-gap CI excludes 0.")


if __name__ == "__main__":
    report()
