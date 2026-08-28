"""
BT-24 (H2-C): does an exposure / concurrency cap on the DP book kill the peak drawdown?

H2-B (BT-23) cut the worst YEAR 76% with a trend skip but left peak drawdown unchanged
(−$76,526) — because the 2022 trough is seeded by positions opened *before* the trend broke,
all in flight at once. The remaining lever is bounding how many DP spreads can be open
simultaneously. John's no-auto-close policy stands, so this is an ENTRY-side cap (skip new
entries when the book is full), never a forced close.

This sim is concurrency-honest, unlike the sequential-by-entry drawdown of BT-21/23:
  - walk GO days chronologically; on each, expire anything past its expiry, then open 1 new
    DP iff currently-open count < CAP, else SKIP.
  - realize P&L at expiry; the equity curve is realized cum-P&L ordered by EXPIRY date.
  - track peak concurrent open positions and peak simultaneous capital-at-risk (sum of open
    max-losses) = the true worst-case account exposure at an instant.

Tested standalone and combined with the H2-B trend skip (no new DP while SPX < 200dma).
50-wide for comparability with BT-23; a narrower width scales every dollar figure down
(BT-23: tail ∝ capital), so read caps as "N concurrent units of whatever width you run."

PRE-REGISTERED DECISION RULE (written before running):
  An exposure cap "works" if some CAP cuts realized-equity max drawdown >=25% AND keeps total
  P&L within 20% of uncapped. Report the cap that best trades drawdown for retained EV; if even
  a tight cap barely dents drawdown, the drawdown is intra-trade (single-entry depth), not a
  concurrency problem, and capping is the wrong tool.

CAVEATS: credits BSM-modeled -> conservative floor. Realized-equity drawdown (by expiry) is the
honest metric here and is NOT comparable to the sequential-by-entry drawdown quoted in BT-21/23;
the uncapped row is the apples-to-apples baseline. 1 contract per accepted entry. Outcome
leak-proof (expiry SPX close only).
"""
import numpy as np
import pandas as pd
from bt_h2_dp_sizing import build_dp

CAPS = [9999, 8, 6, 4, 2]


def simulate(df, cap, trend_skip):
    """Chronological entry-capped book. df: build_dp output. Returns accepted-trade ledger."""
    df = df.sort_values("date").reset_index(drop=True)
    open_pos = []   # list of (expiry, maxloss)
    accepted = []
    peak_n = 0
    peak_cap = 0.0
    for _, r in df.iterrows():
        if trend_skip and r.below200:
            continue
        open_pos = [p for p in open_pos if p[0] > r.date]   # expire matured positions
        if len(open_pos) >= cap:
            continue
        open_pos.append((r.expiry, r.maxloss))
        accepted.append((r.date, r.expiry, r.year, r.pnl, r.maxloss))
        peak_n = max(peak_n, len(open_pos))
        peak_cap = max(peak_cap, sum(p[1] for p in open_pos))
    led = pd.DataFrame(accepted, columns=["date", "expiry", "year", "pnl", "maxloss"])
    return led, peak_n, peak_cap


def realized_dd(led):
    """Drawdown of realized cum-P&L ordered by expiry date (concurrency-honest)."""
    if len(led) == 0:
        return 0.0
    s = led.sort_values("expiry")
    cum = s.pnl.cumsum().to_numpy()
    return float(np.max(np.maximum.accumulate(cum) - cum))


def block(df, markup, trend_skip):
    tag = "trend-skip + cap" if trend_skip else "cap only"
    base, base_n, base_cap = simulate(df, CAPS[0], trend_skip)
    base_total = base.pnl.sum()
    base_dd = realized_dd(base)
    print(f"\n{'='*94}")
    print(f"  BT-24 / H2-C — DP exposure cap ({tag}), 50-wide, markup {markup:.2f}")
    print(f"{'='*94}")
    print(f"  uncapped: {len(base)} trades, peak {base_n} concurrent, peak capital-at-risk "
          f"${base_cap:,.0f}, total ${base_total:+,.0f}, realized maxDD ${-base_dd:,.0f}")
    print(f"  {'cap':>5} {'trades':>7} {'peak#':>6} {'peak cap@risk':>14} {'total P&L':>12} "
          f"{'worst-yr':>11} {'2022':>11} {'realized maxDD':>15} {'verdict':>8}")
    print(f"  {'-'*98}")
    for cap in CAPS:
        led, pn, pc = simulate(df, cap, trend_skip)
        by_yr = led.groupby("year").pnl.sum()
        dd = realized_dd(led)
        total = led.pnl.sum()
        cut_dd = dd <= base_dd * 0.75
        keep_ev = total >= base_total * 0.80
        verdict = "CAP" if (cap < CAPS[0] and cut_dd and keep_ev) else "-"
        label = "none" if cap == CAPS[0] else str(cap)
        print(f"  {label:>5} {len(led):>7} {pn:>6} ${pc:>12,.0f} ${total:>+10,.0f} "
              f"${by_yr.min():>+10,.0f} ${by_yr.get(2022,0):>+10,.0f} ${-dd:>+13,.0f} {verdict:>8}")
    print(f"\n  CAP = realized maxDD >=25% lower AND total P&L within 20% of uncapped.")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        df = build_dp(50, m)
        block(df, m, trend_skip=False)
        block(df, m, trend_skip=True)
