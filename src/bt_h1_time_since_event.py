"""
BT-25 (H1): is DP edge richest right AFTER a vol event and does it DECAY as time-since-event
grows? (John's thesis 2026-06-19: deploy en masse into/after a spike — "blood on the streets" —
then scale out monotonically as the market gets frothy/benign, operate lean when primed for the
next event; max size years 1-4 post-event, cautious after.)

Two separable claims, measured not assumed:
  (1) LEVEL: is mean DP P&L higher when days-since-event (DSE) is small?
  (2) DECAY: does mean DP P&L fall monotonically as DSE grows into the multi-year bins?

Design — measured UNCONDITIONAL on the FRED gate to maximize events (needs only SPX+VIX, full
2005-2026 cache), then the gate is applied live on top. DP = BT-21 pricing verbatim (25Δ put,
50-wide, 44 DTE, BSM w/ VIX, hold-to-expiry, leak-free outcome from SPX close only).

Event = VIX daily close up-crossing threshold T, debounced: re-crossings within MERGE days of an
active episode merge into one event (so one selloff = one event, not ten). Live-knowable: DSE
uses the threshold-cross date, NOT the (hindsight-only) VIX peak — a tradable signal. Entries
before the first event in-sample are dropped (no prior-event marker), noted as a limitation.

Honest tail handling: small-DSE entries are high-variance — selling into a STILL-RISING spike
(2008) can lose big even though premium is rich. So each bin reports mean, WR, OTM-survival AND
worst trade, not EV alone. Event-block bootstrap (resample whole episodes) gives CI because the
number of independent major events is small — the binding constraint on any multi-year curve.

PRE-REGISTERED DECISION RULE (written before running):
  Thesis SUPPORTED iff (a) the <365d-post-event bins have mean P&L >= the 4y+ bin, AND (b) that
  gap's event-block-bootstrap 95% CI excludes 0, AND (c) the decay is monotone (each older bin
  <= the prior). If mean P&L is flat across DSE bins, or the gap CI includes 0, the thesis is
  intuitive but NOT in the data -> do not build a time-since-event sizing taper.
CAVEATS: credits BSM (floor). DSE confounds with bull-market maturity (bigger DSE ~ later cycle)
— that is partly the point, but means the effect is regime, not pure clock. n_events single-digit
at T=40. Unconditional on gate (live gate removes some bad-DSE entries already).
"""
import numpy as np
import pandas as pd
import os
from bt_dp_gate_pnl import bsm_put, short_strike_from_delta, DATA, DTE_CAL, TARGET_DELTA, WIDTH, MULT

MERGE = 90          # calendar days: re-crossings within this of an active episode = same event
BIN_EDGES = [0, 90, 365, 730, 1095, 1460, 10**9]
BIN_LBL = ["0-90d", "90d-1y", "1-2y", "2-3y", "3-4y", "4y+"]
RNG = np.random.default_rng(20260619)
N_BOOT = 3000


def vol_events(vix, T):
    up = (vix >= T) & (vix.shift(1) < T)
    ev, last = [], None
    for dt in vix.index[up.fillna(False)]:
        if last is None or (dt - last).days >= MERGE:
            ev.append(dt)
            last = dt
    return ev


def dp_stream(markup):
    spx = pd.read_parquet(os.path.join(DATA, "yahoo_GSPC.parquet"))["Close"].sort_index()
    vix = pd.read_parquet(os.path.join(DATA, "yahoo_VIX.parquet"))["Close"].sort_index()
    T = DTE_CAL / 365.0
    rows = []
    for dt in spx.index:
        S = float(spx.loc[dt])
        sigma = float(vix.loc[dt]) / 100.0 if dt in vix.index else None
        if sigma is None or sigma <= 0:
            continue
        target = dt + pd.Timedelta(days=DTE_CAL)
        xloc = spx.index.get_indexer([target], method="nearest")[0]
        if (spx.index[xloc] - dt).days < DTE_CAL - 7:
            continue
        ST = float(spx.iloc[xloc])
        Ks = short_strike_from_delta(S, sigma, T, TARGET_DELTA)
        Kl = Ks - WIDTH
        credit = (bsm_put(S, Ks, sigma, T) - bsm_put(S, Kl, sigma, T)) * markup
        spread_val = max(Ks - ST, 0.0) - max(Kl - ST, 0.0)
        pnl = (credit - spread_val) * MULT
        rows.append((dt, pnl, ST >= Ks))
    return pd.DataFrame(rows, columns=["date", "pnl", "otm"]), vix


def tag_dse(df, events):
    ev = pd.DatetimeIndex(sorted(events))
    dse, epi = [], []
    for dt in df.date:
        loc = ev.searchsorted(dt, side="right") - 1
        if loc < 0:
            dse.append(np.nan); epi.append(-1)
        else:
            dse.append((dt - ev[loc]).days); epi.append(loc)
    out = df.copy()
    out["dse"] = dse
    out["epi"] = epi
    return out[out.dse.notna()].reset_index(drop=True)


def boot_gap(df):
    """Event-block bootstrap of (mean P&L in <365d bins) - (mean in 4y+ bin)."""
    near = df[df.dse < 365]
    far = df[df.dse >= 1460]
    if len(near) == 0 or len(far) == 0:
        return np.nan, np.nan, np.nan
    obs = near.pnl.mean() - far.pnl.mean()
    epis = df.epi.unique()
    by_epi = {e: df[df.epi == e] for e in epis}
    diffs = []
    for _ in range(N_BOOT):
        pick = RNG.choice(epis, size=len(epis), replace=True)
        pool = pd.concat([by_epi[e] for e in pick])
        n = pool[pool.dse < 365]; fr = pool[pool.dse >= 1460]
        if len(n) and len(fr):
            diffs.append(n.pnl.mean() - fr.pnl.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, lo, hi


def report(markup, T):
    df, _ = dp_stream(markup)
    events = vol_events(pd.read_parquet(os.path.join(DATA, "yahoo_VIX.parquet"))["Close"].sort_index(), T)
    tagged = tag_dse(df, events)
    print(f"\n{'='*92}")
    print(f"  BT-25 / H1 — DP P&L by days-since VIX>{T} event, 2005-2026, markup {markup:.2f} "
          f"(unconditional on gate)")
    print(f"{'='*92}")
    print(f"  {len(events)} debounced events; first {events[0].date()}, last {events[-1].date()}; "
          f"{len(tagged)}/{len(df)} entries have a prior event")
    print(f"  {'DSE bin':>8} {'n':>6} {'mean P&L':>10} {'WR':>7} {'OTM-surv':>9} "
          f"{'worst':>9} {'total':>12}")
    print(f"  {'-'*70}")
    tagged["bin"] = pd.cut(tagged.dse, BIN_EDGES, labels=BIN_LBL, right=False)
    for lbl in BIN_LBL:
        g = tagged[tagged.bin == lbl]
        if len(g) == 0:
            print(f"  {lbl:>8} {0:>6}")
            continue
        print(f"  {lbl:>8} {len(g):>6} ${g.pnl.mean():>+8,.0f} {(g.pnl>0).mean()*100:>6.1f}% "
              f"{g.otm.mean()*100:>8.1f}% ${g.pnl.min():>+7,.0f} ${g.pnl.sum():>+10,.0f}")
    obs, lo, hi = boot_gap(tagged)
    sig = ("CI undefined (far bin empty — no event old enough)" if np.isnan(lo)
           else "CI excludes 0" if not (lo <= 0 <= hi)
           else "CI INCLUDES 0 (not significant)")
    print(f"\n  near(<1y) - far(4y+) mean P&L gap: ${obs:+,.0f}/trade  "
          f"95% CI [${lo:+,.0f}, ${hi:+,.0f}]  -> {sig}")
    # monotonicity check across bins present
    means = [tagged[tagged.bin == lbl].pnl.mean() for lbl in BIN_LBL if len(tagged[tagged.bin == lbl])]
    mono = all(means[i] >= means[i+1] - 1e-9 for i in range(len(means)-1))
    print(f"  monotone decay across bins: {mono}")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        for T in (30, 40):
            report(m, T)
