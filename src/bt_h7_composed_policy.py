"""
BT-27 (H7): do the two DP sizing levers found in Wave 6 COMPOSE into one book that beats the
flat every-GO-day book — or does one lever cannibalize the other?

The Wave-6 blueprint stacks two rules that touch the SAME regime in OPPOSITE directions:
  - BT-23 trend-skip: do NOT open DP while SPX < 200dma (a hard entry block, ~4% of GO-days).
  - BT-25 post-event up-size: size UP when 0-1y past a VIX>40 event (the strongest regime).
A VIX>40 spike usually breaks the 200dma, so "size up (fresh-post-event)" frequently fires
exactly when "skip (sub-200dma)" blocks. Neither was ever measured jointly. This test runs the
composed book end-to-end and asks the coherence question the registry left open: once the skip
takes its entries, is there any post-event up-size left to apply, and does the composite beat flat?

Reuses build_dp (BT-23: GO-day 2016+ entries, 50-wide, below200 flag, expiry) and vol_events
(BT-25: debounced VIX>40 up-crosses). Fresh-post-event = days-since-event < 365 at entry, live-
knowable (cross date, not peak). Drawdown is realized-by-expiry (BT-24 concurrency-honest method)
for EVERY book, so flat/skip/composed are apples-to-apples. Hold-to-expiry. Credits BSM (floor).

Books compared:
  FLAT      every GO entry, size 1 (= BT-21).
  SKIP      drop sub-200dma entries, size 1 (isolates BT-23's lever).
  COMPOSED  drop sub-200dma, size = UPMULT if fresh-post-event else 1 (the blueprint).
UPMULT swept {1.5, 2, 3} — BT-25 found the up-size direction, not a magnitude.

PRE-REGISTERED DECISION RULE (written before running):
  The composed book "wins" (blueprint confirmed) iff at BOTH markups it (a) beats FLAT's
  EV-per-drawdown ratio (total P&L / realized maxDD) AND (b) worst calendar-year is no worse
  than FLAT. If COMPOSED only ties SKIP (i.e. the up-size adds ~nothing because fresh-post-event
  entries are mostly blocked by the skip), DROP the up-size from the live blueprint and keep only
  the skip. If neither beats FLAT on EV/DD, the tilts are cosmetic and flat-to-BP stands.
COHERENCE DIAGNOSTIC: report what fraction of fresh-post-event GO entries SURVIVE the 200dma skip
  — the reachability of the up-size lever. If near 0, the up-size is unreachable by construction.
CAVEATS: credits BSM (floor). 2016+ gate era only. Realized-by-expiry DD (not comparable to
  BT-21/23 sequential figures). Fractional contracts via UPMULT are a comparison device.
"""
import os
import numpy as np
import pandas as pd
from bt_h2_dp_sizing import build_dp
from bt_h1_time_since_event import vol_events
from bt_dp_gate_pnl import DATA

T_EVENT = 40
FRESH_DAYS = 365
UPMULTS = [1.5, 2.0, 3.0]


def tag_fresh(df, events):
    ev = pd.DatetimeIndex(sorted(events))
    dse = []
    for dt in df.date:
        loc = ev.searchsorted(dt, side="right") - 1
        dse.append((dt - ev[loc]).days if loc >= 0 else np.nan)
    out = df.copy()
    out["dse"] = dse
    out["fresh"] = out.dse.notna() & (out.dse < FRESH_DAYS)
    return out


def realized_dd(led):
    """Drawdown of size-weighted realized cum-P&L ordered by expiry (BT-24 method)."""
    if len(led) == 0:
        return 0.0
    s = led.sort_values("expiry")
    cum = (s.pnl * s["size"]).cumsum().to_numpy()
    return float(np.max(np.maximum.accumulate(cum) - cum))


def peak_capital(led):
    """Peak simultaneous size-weighted capital-at-risk (sum of open max-losses)."""
    if len(led) == 0:
        return 0.0
    s = led.sort_values("date")
    open_pos, peak = [], 0.0
    for _, r in s.iterrows():
        open_pos = [p for p in open_pos if p[0] > r.date]
        open_pos.append((r.expiry, r.maxloss * r["size"]))
        peak = max(peak, sum(p[1] for p in open_pos))
    return peak


def metrics(led):
    total = (led.pnl * led["size"]).sum()
    by_yr = (led.pnl * led["size"]).groupby(led.year).sum()
    dd = realized_dd(led)
    ratio = total / dd if dd else float("inf")
    return dict(n=led["size"].sum(), total=total, worst=by_yr.min(),
                w2022=by_yr.get(2022, 0.0), dd=dd, ratio=ratio, peakcap=peak_capital(led))


def book(df, skip, upmult):
    b = df.copy()
    if skip:
        b = b[~b.below200]
    b["size"] = np.where(b.fresh, upmult, 1.0) if upmult else 1.0
    return b


def report(markup):
    df = tag_fresh(build_dp(50, markup),
                   vol_events(pd.read_parquet(os.path.join(DATA, "yahoo_VIX.parquet"))["Close"].sort_index(), T_EVENT))
    n_fresh = int(df.fresh.sum())
    n_fresh_survive = int((df.fresh & ~df.below200).sum())
    print(f"\n{'='*100}")
    print(f"  BT-27 / H7 — composed DP policy (skip <200dma × up-size fresh-post-event), "
          f"50-wide, markup {markup:.2f}")
    print(f"{'='*100}")
    print(f"  {len(df)} GO-day entries 2016+; sub-200dma {int(df.below200.sum())}; "
          f"fresh-post-event (<1y, VIX>{T_EVENT}) {n_fresh}; "
          f"fresh AND surviving skip {n_fresh_survive} "
          f"({(n_fresh_survive/n_fresh*100 if n_fresh else 0):.0f}% of fresh reachable)")
    flat = metrics(book(df, skip=False, upmult=None))
    print(f"  {'book':>16} {'eff n':>7} {'total P&L':>12} {'worst-yr':>11} {'2022':>11} "
          f"{'realized DD':>12} {'EV/DD':>7} {'peak cap':>11} {'verdict':>9}")
    print(f"  {'-'*108}")
    def line(name, m, verdict=""):
        print(f"  {name:>16} {m['n']:>7.0f} ${m['total']:>+10,.0f} ${m['worst']:>+10,.0f} "
              f"${m['w2022']:>+10,.0f} ${-m['dd']:>+11,.0f} {m['ratio']:>6.2f} "
              f"${m['peakcap']:>9,.0f} {verdict:>9}")
    line("FLAT (BT-21)", flat)
    skip_m = metrics(book(df, skip=True, upmult=None))
    line("SKIP only", skip_m, "WIN" if (skip_m['ratio'] >= flat['ratio'] and skip_m['worst'] >= flat['worst']) else "-")
    for um in UPMULTS:
        m = metrics(book(df, skip=True, upmult=um))
        win = m['ratio'] >= flat['ratio'] and m['worst'] >= flat['worst']
        line(f"COMPOSED x{um:g}", m, "WIN" if win else "-")
    print(f"\n  WIN = EV/DD ratio >= FLAT AND worst-yr no worse than FLAT (both markups required).")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
