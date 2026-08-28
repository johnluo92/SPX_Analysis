"""
BT-30 (H8b): can an ALWAYS-ON (signal-free) long-put tail hedge nullify the DP down-years without
giving up the franchise? H8a (BT-29) proved gate-close is coincident, not leading — so no
signal-triggered hedge works. The only honest hedge is a constant long-premium overlay sized so
good-year bleed < bad-year savings. This is John's "properly hedge the down years so things don't
fall apart when the chickens come home to roost," done the only way the data permits.

Method: same GO-day cadence and pricing as BT-21 (the hedge holds while DP is exposed). On each GO
day buy a long OTM put at delta HEDGE_DELTA, 44 DTE, hold to expiry; cost = BSM mid (buyer pays
mid, no markup — DP still receives mid x markup, as built). Per-GO-day hedge P&L =
(max(Kp-ST,0) - cost) * MULT. Net book = DP P&L + RATIO * hedge P&L, swept over RATIO (long puts
per short-put-spread) and HEDGE_DELTA {0.10 far-OTM cheap, 0.25 at-the-DP-strike}. Also test a put
DEBIT SPREAD hedge (buy HEDGE_DELTA put / sell a farther-OTM put) — much cheaper bleed, capped
payoff — as the realistic "affordable insurance" variant. Drawdown realized-by-expiry (BT-24).

PRE-REGISTERED DECISION RULE (written before running):
  A hedge "works" iff it (a) cuts the worst calendar-year loss >= 50%, AND (b) cuts realized maxDD
  >= 25%, AND (c) retains >= 75% of DP-alone total P&L. (a)+(b) = the down years are actually
  hedged; (c) = the insurance is affordable, not a tax that eats the franchise. If no (delta,ratio)
  clears all three, an always-on hedge cannot nullify the tail cheaply -> the realistic defenses
  remain the 200dma skip (BT-23) + 2x stop (BT-28), not a standing long-put book.
CAVEATS: BSM mid pricing both sides (floor); long-put cost is if anything UNDERSTATED at the mid
  (real far-OTM puts carry a skew premium -> hedge would bleed MORE live, so results are optimistic
  for the hedge). 2016+ gate era. Hold-to-expiry. Realized-by-expiry DD comparable to BT-24/27/28.
"""
import os
import numpy as np
import pandas as pd
from layer1_gate import build_gate_features, evaluate_gate
from bt_dp_gate_pnl import (
    bsm_put, short_strike_from_delta, DATA, DTE_CAL, TARGET_DELTA, START_YEAR, MULT, WIDTH,
)

HEDGE_DELTAS = [0.10, 0.25]
RATIOS = [0.25, 0.5, 1.0]
SPREAD_WIDTH = 50.0     # for the put-debit-spread hedge variant


def build(markup, hedge_delta):
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
        ST = float(spx.iloc[xloc])
        # DP short put spread (BT-21)
        Ks = short_strike_from_delta(S, sigma, T, TARGET_DELTA)
        Kl = Ks - WIDTH
        credit = (bsm_put(S, Ks, sigma, T) - bsm_put(S, Kl, sigma, T)) * markup
        dp_pnl = (credit - (max(Ks - ST, 0.0) - max(Kl - ST, 0.0))) * MULT
        # long put hedge at hedge_delta
        Kp = short_strike_from_delta(S, sigma, T, hedge_delta)
        cost = bsm_put(S, Kp, sigma, T)
        put_pnl = (max(Kp - ST, 0.0) - cost) * MULT
        # put DEBIT SPREAD hedge: long Kp / short (Kp - SPREAD_WIDTH)
        Kp2 = Kp - SPREAD_WIDTH
        dcost = bsm_put(S, Kp, sigma, T) - bsm_put(S, Kp2, sigma, T)
        dspread_pnl = ((max(Kp - ST, 0.0) - max(Kp2 - ST, 0.0)) - dcost) * MULT
        rows.append((dt, spx.index[xloc], dt.year, dp_pnl, put_pnl, dspread_pnl))
    return pd.DataFrame(rows, columns=["date", "expiry", "year", "dp", "put", "dspread"])


def realized_dd(df, col):
    s = df.sort_values("expiry")
    cum = s[col].cumsum().to_numpy()
    return float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0


def report(markup):
    print(f"\n{'='*104}")
    print(f"  BT-30 / H8b — always-on long-put tail hedge on the DP book, markup {markup:.2f}")
    print(f"{'='*104}")
    base = build(markup, 0.25)
    dp_total = base.dp.sum()
    dp_worst = base.groupby("year").dp.sum().min()
    dp_dd = realized_dd(base, "dp")
    print(f"  DP-alone: total ${dp_total:+,.0f}  worst-yr ${dp_worst:+,.0f}  maxDD ${-dp_dd:,.0f}  "
          f"(2022 ${base.groupby('year').dp.sum().get(2022,0):+,.0f}, "
          f"2018 ${base.groupby('year').dp.sum().get(2018,0):+,.0f})")
    print(f"  {'hedge':>16} {'ratio':>6} {'total P&L':>12} {'%kept':>6} {'worst-yr':>11} "
          f"{'2022':>11} {'2018':>11} {'maxDD':>11} {'works':>6}")
    print(f"  {'-'*104}")
    for hd in HEDGE_DELTAS:
        df = build(markup, hd)
        for kind in ["put", "dspread"]:
            for ratio in RATIOS:
                net = df.dp + ratio * df[kind]
                tmp = df.assign(net=net)
                total = net.sum()
                worst = tmp.groupby("year").net.sum().min()
                dd = realized_dd(tmp, "net")
                kept = total / dp_total * 100
                works = (worst >= dp_worst * 0.50 and dd <= dp_dd * 0.75 and total >= dp_total * 0.75)
                by = tmp.groupby("year").net.sum()
                name = f"{int(hd*100)}d {'put' if kind=='put' else 'putspr'}"
                print(f"  {name:>16} {ratio:>5.2f}x ${total:>+10,.0f} {kept:>5.0f}% ${worst:>+10,.0f} "
                      f"${by.get(2022,0):>+10,.0f} ${by.get(2018,0):>+10,.0f} ${-dd:>+10,.0f} "
                      f"{'YES' if works else 'no':>6}")
    print(f"\n  WORKS = worst-yr loss cut >=50% AND maxDD cut >=25% AND >=75% of DP total retained.")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
