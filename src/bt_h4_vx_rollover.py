"""
BT-26 (H4): is DP entry P&L systematically different by position in the monthly VX-futures
cycle? (John's roadmap item: "month-end VX-rollover vol spikes" — a datable CALENDAR signal,
distinct from the already-rejected VIX-velocity momentum signal, tested_signals_negative.)

Premise being tested: the front VX contract settles monthly; if the roll/settlement window
produces a datable vol spike, then DP entry timed against the VX cycle (days-to-front-settlement)
should show a systematic P&L pattern — e.g. richer credit entering INTO the spike, or worse
outcomes entering right before it.

Datable signal: VX1.parquet carries days_to_expiry (front-contract days to settlement) and
roll_indicator — both live-knowable from the futures calendar, no hindsight. We bin DP entries
by days_to_expiry and compare outcome.

Two measurements, honest about the a-priori:
  (1) PREMISE: is realized next-day VIX move elevated in any cycle window? (already spot-checked
      flat; reported here for the record.)
  (2) CONSEQUENCE: does the 44-DTE DP hold P&L differ by entry cycle-position? Note a 44-DTE
      hold spans ~2 VX cycles, so any intramonth spike largely washes out over the hold — the
      honest prior is a weak/no effect. Data decides.

DP = BT-21 pricing verbatim (25d put, 50-wide, 44 DTE, BSM w/ VIX, hold-to-expiry, leak-free
outcome from SPX close only). Unconditional on the FRED gate to maximize sample, 2013-2026
(VX continuous starts 2013-05). Block bootstrap by VX contract_id (one cycle = one block) for CI.

PRE-REGISTERED DECISION RULE (written before running the full P&L test):
  H4 SUPPORTED iff some contiguous days_to_expiry window has mean DP P&L differing from the rest
  by a margin whose contract-block-bootstrap 95% CI excludes 0, AND the effect is monotone or
  single-peaked in cycle position (a coherent calendar pattern, not one noisy bin). If the CI
  includes 0 or the pattern is incoherent, the rollover-timing signal is NOT in the data ->
  do not build a VX-cycle entry filter. (Distinct from VIX velocity, already rejected.)
CAVEATS: credits BSM (floor). 44-DTE hold spans ~2 cycles (dilutes entry-cycle effect by design).
Unconditional on gate. VX continuous 2013+ (shorter than the 2005+ DP history elsewhere).
"""
import os
import numpy as np
import pandas as pd
from bt_dp_gate_pnl import bsm_put, short_strike_from_delta, DATA, DTE_CAL, TARGET_DELTA, WIDTH, MULT

BIN_EDGES = [0, 4, 8, 15, 22, 99]
BIN_LBL = ["1-3d", "4-7d", "8-14d", "15-21d", "22d+"]
RNG = np.random.default_rng(20260619)
N_BOOT = 3000


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
        rows.append((dt, credit * MULT, pnl, ST >= Ks))
    return pd.DataFrame(rows, columns=["date", "credit", "pnl", "otm"])


def tag_cycle(df):
    vx = pd.read_parquet(os.path.join(DATA, "vx_continuous/VX1.parquet"))
    j = df.set_index("date").join(vx[["days_to_expiry", "contract_id"]], how="inner")
    return j.reset_index().rename(columns={"index": "date"})


def boot_gap(df, win_lo, win_hi):
    """Contract-block bootstrap of (mean P&L in window) - (mean P&L outside window)."""
    inw = df[(df.days_to_expiry >= win_lo) & (df.days_to_expiry <= win_hi)]
    out = df[(df.days_to_expiry < win_lo) | (df.days_to_expiry > win_hi)]
    if len(inw) == 0 or len(out) == 0:
        return np.nan, np.nan, np.nan
    obs = inw.pnl.mean() - out.pnl.mean()
    cids = df.contract_id.unique()
    by_c = {c: df[df.contract_id == c] for c in cids}
    diffs = []
    for _ in range(N_BOOT):
        pick = RNG.choice(cids, size=len(cids), replace=True)
        pool = pd.concat([by_c[c] for c in pick])
        i = pool[(pool.days_to_expiry >= win_lo) & (pool.days_to_expiry <= win_hi)]
        o = pool[(pool.days_to_expiry < win_lo) | (pool.days_to_expiry > win_hi)]
        if len(i) and len(o):
            diffs.append(i.pnl.mean() - o.pnl.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, lo, hi


def report(markup):
    df = tag_cycle(dp_stream(markup))
    df["bin"] = pd.cut(df.days_to_expiry, BIN_EDGES, labels=BIN_LBL, right=False)
    print(f"\n{'='*92}")
    print(f"  BT-26 / H4 — DP P&L by VX-cycle position (days to front-VX settlement), "
          f"markup {markup:.2f}")
    print(f"{'='*92}")
    print(f"  {len(df)} entries, {df.date.min().date()}-{df.date.max().date()}, "
          f"{df.contract_id.nunique()} VX cycles (bootstrap blocks)")
    print(f"  {'cycle pos':>10} {'n':>6} {'mean P&L':>10} {'WR':>7} {'OTM':>7} "
          f"{'avg credit':>11} {'worst':>9}")
    print(f"  {'-'*64}")
    means = []
    for lbl in BIN_LBL:
        g = df[df.bin == lbl]
        means.append(g.pnl.mean())
        print(f"  {lbl:>10} {len(g):>6} ${g.pnl.mean():>+8,.0f} {(g.pnl>0).mean()*100:>6.1f}% "
              f"{g.otm.mean()*100:>6.1f}% ${g.credit.mean():>+9,.0f} ${g.pnl.min():>+7,.0f}")
    # test the most-extreme single window (closest-to-settlement) vs rest
    obs, lo, hi = boot_gap(df, 0, 3)
    sig = "CI excludes 0" if not (lo <= 0 <= hi) else "CI INCLUDES 0 (not significant)"
    print(f"\n  near-settlement (1-3d) vs rest: ${obs:+,.0f}/trade  "
          f"95% CI [${lo:+,.0f}, ${hi:+,.0f}]  -> {sig}")
    mono = (all(means[i] <= means[i+1] for i in range(len(means)-1))
            or all(means[i] >= means[i+1] for i in range(len(means)-1)))
    rng_ = max(means) - min(means)
    print(f"  monotone across cycle: {mono}   |  spread high-low bin: ${rng_:,.0f}/trade")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
