"""
BT-22 (H3): does a TACTICAL DN sleeve buffer the DP book's down-year tail?

Question (John 2026-06-19): "if we just layer into delta positives, can the results
be buffered by the dn side." DP alone (BT-21) makes money 9 of 11 years but the entire
risk is 2022 (-$53k/contract) + 2018. DN tactical (BT-16) is +EV and positive every year,
and fires in the selloffs where DP bleeds. Combining them should be negatively-correlated
in the tail. This measures by how much, and at what good-year cost.

Reuses (no reinvention, both canonical):
  - DP stream  = bt_dp_gate_pnl.build_pnl (BT-21): 25Δ put, 50-wide, 44 DTE, every GO day, 2016+.
  - DN stream  = BT-16 firing union (bt_dn_conjunction) + BSM call pricing (bt_dn_triggers):
                 25Δ call, 10-wide, 45 DTE, 4 triggers, 45d one-at-a-time non-overlap.
  Each leg priced in its own registered convention so each reproduces its source backtest;
  the combination is pure arithmetic on the two trade streams.

DP runs 1 contract / GO-day (~90/yr, the BT-21 unit). DN fires ~5/yr; to be a meaningful
hedge its per-event size must scale up. We sweep DN contracts/event and read the dose-response.

PRE-REGISTERED DECISION RULE (written before running):
  Tactical DN "buffers" DP IF at some DN size whose annual EV contribution is <= DP's own
  annual EV (i.e. DN is a hedge, not the main bet):
    (a) combined worst-year loss shrinks >=25% vs DP-only, AND
    (b) combined max drawdown shrinks vs DP-only, AND
    (c) combined total EV stays positive.
  If DN only helps when sized so large it dominates EV, or cuts total EV without commensurate
  tail relief -> reject DN-as-buffer; DN stays a standalone tactical sleeve (BT-16) and the
  DP tail is addressed by sizing/throttling instead (H2).

CAVEATS: both credits BSM-modeled (live DN ran richer 32% vs ~21.7%, live DP per BT-16 note)
-> EV is a conservative floor on both sides. Drawdown is time-ordered across overlapping
trades at fixed contract counts. Annual-Sharpe n=11 years (noisy comparator, labeled).
SPX/VIX: DP from SPX_Analysis data_cache, DN from bt01_*.pkl yfinance caches (both daily SPX
close; annual aggregation immune to the tiny source diff). Survivorship: SPX index, minimal.
"""
import os
import sys
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from bt_dp_gate_pnl import build_pnl  # DP stream (BT-21), same dir

SIGVAL = "/Users/johnluo/Desktop/Byzantium_Knowledge/Trading/Schwab_API/equity_screener/signal_validation"
sys.path.insert(0, SIGVAL)
from bt_dn_triggers import (                       # noqa: E402
    bsm_call, find_25delta_call_strike, next_trading_day,
    load_spx, load_vix_series, load_gate_history,
    RISK_FREE, MIN_PRICE, BACKTEST_START, BACKTEST_END,
)
from bt_dn_conjunction import (                    # noqa: E402
    fire_gate_close, fire_drawdown, fire_vix_cross, fire_rally_exhaustion,
)

START_YEAR = 2016
DN_WIDTH = 10.0
DN_DTE = 45
DN_GAP = 45
DN_SCALES = [1, 5, 10, 20, 40]


def dn_stream(markup):
    """BT-16 disjunctive DN program -> per-trade ledger with dates, filtered to START_YEAR+."""
    spx = load_spx(BACKTEST_START, BACKTEST_END)
    vix = load_vix_series(BACKTEST_START, BACKTEST_END)
    gate = load_gate_history()
    sets = {
        "gate_close": fire_gate_close(spx, gate),
        "drawdown_3pct_10d": fire_drawdown(spx),
        "vix_cross_25": fire_vix_cross(spx, vix),
        "rally_exhaustion": fire_rally_exhaustion(spx),
    }
    union = sorted({d for ds in sets.values() for d in ds})
    accepted, last = [], None
    for d in union:
        if last is None or (d - last).days >= DN_GAP:
            accepted.append(d)
            last = d
    rows = []
    for ts in accepted:
        if ts not in spx.index or float(spx[ts]) < MIN_PRICE or ts not in vix.index:
            continue
        S = float(spx[ts])
        T = DN_DTE / 252.0
        sigma = max(float(vix[ts]) / 100.0, 0.08) * markup
        Ks = find_25delta_call_strike(S, T, RISK_FREE, sigma)
        Kl = Ks + DN_WIDTH
        credit = max(bsm_call(S, Ks, T, RISK_FREE, sigma) - bsm_call(S, Kl, T, RISK_FREE, sigma), 0.0)
        if credit <= 0:
            continue
        xts = next_trading_day(spx, ts.date() + timedelta(days=DN_DTE))
        if xts is None:
            continue
        intrinsic = min(max(float(spx[xts]) - Ks, 0.0), DN_WIDTH)
        rows.append((pd.Timestamp(ts), ts.year, (credit - intrinsic) * 100.0))
    df = pd.DataFrame(rows, columns=["date", "year", "pnl"])
    return df[df.year >= START_YEAR].reset_index(drop=True)


def max_drawdown(ledger):
    """ledger: DataFrame[date, pnl] (1 contract). Time-ordered cumulative peak-to-trough."""
    s = ledger.sort_values("date")
    cum = s.pnl.cumsum().to_numpy()
    if len(cum) == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(cum) - cum))


def annual(ledger):
    return ledger.groupby("year").pnl.sum()


def sharpe_annual(by_year):
    if by_year.std(ddof=1) == 0 or len(by_year) < 2:
        return float("nan")
    return by_year.mean() / by_year.std(ddof=1)


def report(markup):
    dp = build_pnl(markup)
    dp = dp[dp.year >= START_YEAR][["date", "year", "pnl"]].reset_index(drop=True)
    dn = dn_stream(markup)
    yrs = sorted(set(dp.year) | set(dn.year))

    dp_yr = annual(dp).reindex(yrs, fill_value=0.0)
    dn_yr = annual(dn).reindex(yrs, fill_value=0.0)
    dp_dd = max_drawdown(dp)
    dp_total = dp.pnl.sum()
    dp_worst_year = dp_yr.min()
    dp_sharpe = sharpe_annual(dp_yr)

    print(f"\n{'='*86}\n  BT-22 / H3 — DP book buffered by tactical DN, {min(yrs)}-{max(yrs)}, markup {markup:.2f}")
    print(f"{'='*86}")
    print(f"  DP-only (1 ctr/GO-day): total ${dp_total:+,.0f}  worst-yr ${dp_worst_year:+,.0f}  "
          f"maxDD ${-dp_dd:,.0f}  annual-Sharpe {dp_sharpe:.2f}  (n={len(dp)} trades)")
    print(f"  DN-only (1 ctr/event):  total ${dn.pnl.sum():+,.0f}  EV/yr ${dn.pnl.sum()/len(yrs):+,.0f}  "
          f"(n={len(dn)} trades over {len(yrs)} yrs)")
    print(f"  DN 2018 ${dn_yr.get(2018,0):+,.0f}   DN 2022 ${dn_yr.get(2022,0):+,.0f}   "
          f"(the two DP loss years)\n")

    hdr = f"  {'DNx':>4} {'DNev/yr':>9} {'combo total':>12} {'combo worst-yr':>15} " \
          f"{'2018':>9} {'2022':>9} {'combo maxDD':>12} {'Sharpe':>7} {'verdict':>9}"
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for scale in DN_SCALES:
        dn_s = dn.copy()
        dn_s.pnl *= scale
        combo = pd.concat([dp[["date", "pnl"]], dn_s[["date", "pnl"]]], ignore_index=True)
        combo_yr = (dp_yr + dn_yr * scale)
        c_total = combo.pnl.sum()
        c_worst = combo_yr.min()
        c_dd = max_drawdown(combo)
        c_sharpe = sharpe_annual(combo_yr)
        dn_ev_yr = dn.pnl.sum() * scale / len(yrs)
        # buffer verdict per pre-reg: hedge-sized (DN EV <= DP EV), worst-yr >=25% better, DD better, total>0
        hedge_sized = dn_ev_yr <= (dp_total / len(yrs))
        better_worst = c_worst >= dp_worst_year * 0.75  # less negative by >=25%
        better_dd = c_dd < dp_dd * 0.95                  # material (>=5%) DD reduction, not rounding noise
        verdict = "BUFFER" if (hedge_sized and better_worst and better_dd and c_total > 0) else "-"
        print(f"  {scale:>4} ${dn_ev_yr:>+8,.0f} ${c_total:>+11,.0f} ${c_worst:>+14,.0f} "
              f"${combo_yr.get(2018,0):>+8,.0f} ${combo_yr.get(2022,0):>+8,.0f} "
              f"${-c_dd:>+11,.0f} {c_sharpe:>7.2f} {verdict:>9}")
    print(f"\n  Read: 2022/2018 columns are the test. BUFFER = DN hedge-sized AND worst-yr >=25% less"
          f"\n  negative AND maxDD improved AND total still +. Credits BSM (floor). Sharpe annual, n={len(yrs)}.")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
