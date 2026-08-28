"""
BT-28 (H6): does an active EXIT overlay (profit-take and/or stop) beat hold-to-expiry on the
gated DP book? Everything in Wave 6 assumed hold-to-expiry (John 2026-06-18) — never tested. This
is the single untested DECISION the operator faces daily, and the -$76,526 maxDD is an UNMANAGED
number. H6 marks each spread to BSM every trading day of its life and applies exit rules.

Method: same GO-day entry set as BT-21 (cleaned gate, 2016+, 25d put, 50-wide, 44 DTE). Entry
credit = BSM mid x markup (richer fill, as everywhere). Each subsequent trading day the spread is
re-priced at BSM MID (no markup — you do NOT get a markup discount buying back; conservative,
asymmetric, and it makes hold-to-expiry reproduce BT-21 exactly since expiry value = intrinsic).
  - PROFIT-TAKE X: close when captured >= X% of credit received  (value_mid*MULT <= (1-X)*credit)
  - STOP Y: close when loss >= Y x credit received               (value_mid*MULT >= (1+Y)*credit)
  - else hold to expiry (realized intrinsic = BT-21).
Realized P&L booked at the EXIT date (close or expiry); drawdown is realized-by-exit-date cum-P&L
(BT-24 concurrency-honest method) so it's comparable to the BT-24/BT-27 uncapped baseline.

PRE-REGISTERED DECISION RULE (written before running):
  An exit overlay "wins" iff at BOTH markups it cuts realized maxDD >= 20% AND keeps total P&L
  within 15% of hold-to-expiry. If no overlay clears both, hold-to-expiry is CONFIRMED (not merely
  assumed) and the no-management mandate stands. Report each overlay's total, WR, worst-yr, maxDD,
  EV/DD, and mean holding days (turnover cost proxy).
CAVEATS: intra-life marks are BSM MID — the SAME credit-realism floor as entry, now on exits too;
  a real desk pays bid/ask to close (overlay results are an upper bound on exit quality). Stops/PTs
  are checked at DAILY CLOSE only — true intraday stop pain is worse, PT slightly optimistic.
  2016+ gate era. Hold-to-expiry row must equal BT-21 (sanity anchor).
"""
import os
import math
import numpy as np
import pandas as pd
from layer1_gate import build_gate_features, evaluate_gate
from bt_dp_gate_pnl import bsm_put, short_strike_from_delta, DATA, DTE_CAL, TARGET_DELTA, START_YEAR, MULT, WIDTH

PT = [None, 0.50, 0.75]      # profit-take: close at X% of credit captured
STOP = [None, 2.0, 3.0]      # stop: close at Y x credit lost


def build_entries(markup):
    spx = pd.read_parquet(os.path.join(DATA, "yahoo_GSPC.parquet"))["Close"].sort_index()
    vix = pd.read_parquet(os.path.join(DATA, "yahoo_VIX.parquet"))["Close"].sort_index()
    f = build_gate_features()
    valid = (f["vx_contango_pct"].notna() & f["vix_vs_rv_21d"].notna()
             & f["vvix"].notna() & f["c3_vvix_threshold"].notna()
             & f["credit_widening_regime"].notna() & f["stress_regime"].notna())
    go = evaluate_gate(f)[valid]
    go = go[(go.index.year >= START_YEAR) & go]
    T = DTE_CAL / 365.0
    rows = []
    for dt in go.index:
        eloc = spx.index.get_indexer([dt], method="nearest")[0]
        S = float(spx.iloc[eloc])
        vloc = vix.index.get_indexer([dt], method="nearest")[0]
        sigma = float(vix.iloc[vloc]) / 100.0
        target = dt + pd.Timedelta(days=DTE_CAL)
        xloc = spx.index.get_indexer([target], method="nearest")[0]
        if (spx.index[xloc] - dt).days < DTE_CAL - 7:
            continue
        Ks = short_strike_from_delta(S, sigma, T, TARGET_DELTA)
        Kl = Ks - WIDTH
        credit = (bsm_put(S, Ks, sigma, T) - bsm_put(S, Kl, sigma, T)) * markup * MULT
        rows.append((dt, spx.index[xloc], dt.year, Ks, Kl, credit, eloc, xloc))
    return pd.DataFrame(rows, columns=["date", "expiry", "year", "Ks", "Kl", "credit", "eloc", "xloc"]), spx, vix


def simulate(entries, spx, vix, pt, stop):
    """Walk each entry day-by-day; apply pt/stop at daily close; book P&L at exit date."""
    out = []
    sidx = spx.index
    for _, r in entries.iterrows():
        credit, Ks, Kl = r.credit, r.Ks, r.Kl
        exited = False
        for loc in range(r.eloc + 1, r.xloc):
            t = sidx[loc]
            Trem = (r.expiry - t).days / 365.0
            if Trem <= 0:
                break
            S = float(spx.iloc[loc])
            vloc = vix.index.get_indexer([t], method="nearest")[0]
            sig = float(vix.iloc[vloc]) / 100.0
            val = (bsm_put(S, Ks, sig, Trem) - bsm_put(S, Kl, sig, Trem)) * MULT
            if pt is not None and val <= (1 - pt) * credit:
                out.append((t, r.year, credit - val)); exited = True; break
            if stop is not None and val >= (1 + stop) * credit:
                out.append((t, r.year, credit - val)); exited = True; break
        if not exited:
            ST = float(spx.iloc[r.xloc])
            intrinsic = (max(Ks - ST, 0.0) - max(Kl - ST, 0.0)) * MULT
            out.append((r.expiry, r.year, credit - intrinsic))
    led = pd.DataFrame(out, columns=["exit", "year", "pnl"])
    return led


def realized_dd(led):
    s = led.sort_values("exit")
    cum = s.pnl.cumsum().to_numpy()
    return float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0


def hold_days(entries, led):
    return (led.exit.reset_index(drop=True) - entries.date.reset_index(drop=True)).dt.days.mean()


def report(markup):
    entries, spx, vix = build_entries(markup)
    base = simulate(entries, spx, vix, None, None)
    b_total, b_dd = base.pnl.sum(), realized_dd(base)
    print(f"\n{'='*104}")
    print(f"  BT-28 / H6 — DP exit overlay vs hold-to-expiry, 50-wide, markup {markup:.2f} "
          f"({len(entries)} GO entries 2016+)")
    print(f"{'='*104}")
    print(f"  {'profit-take':>12} {'stop':>6} {'total P&L':>12} {'WR':>7} {'worst-yr':>11} "
          f"{'2022':>11} {'realized DD':>12} {'EV/DD':>7} {'hold d':>7} {'verdict':>8}")
    print(f"  {'-'*104}")
    for pt in PT:
        for stop in STOP:
            led = simulate(entries, spx, vix, pt, stop)
            total, dd = led.pnl.sum(), realized_dd(led)
            by_yr = led.groupby("year").pnl.sum()
            wr = (led.pnl > 0).mean() * 100
            ratio = total / dd if dd else float("inf")
            hd = hold_days(entries, led)
            is_base = pt is None and stop is None
            win = (not is_base) and dd <= b_dd * 0.80 and total >= b_total * 0.85
            v = "BASE" if is_base else ("WIN" if win else "-")
            ptl = "hold" if pt is None else f"{pt*100:.0f}%"
            stl = "none" if stop is None else f"{stop:.0f}x"
            print(f"  {ptl:>12} {stl:>6} ${total:>+10,.0f} {wr:>6.1f}% ${by_yr.min():>+10,.0f} "
                  f"${by_yr.get(2022,0):>+10,.0f} ${-dd:>+11,.0f} {ratio:>6.2f} {hd:>6.0f} {v:>8}")
    print(f"\n  WIN = realized maxDD >=20% lower AND total within 15% of hold-to-expiry (both markups).")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
