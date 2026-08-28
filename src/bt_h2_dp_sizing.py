"""
BT-23 (H2): is the DP 2022/2018 tail a SIZING problem, and is 50-wide the right unit?

H3 (BT-22) proved DN cannot hedge the DP tail. So the tail has to be governed on the DP
side itself. John (2026-06-19) raised two concrete levers:
  1. "i don't know about selling a 50-point spread as DP ... the margin req would be too
     great ... the cost of the trade would need to be mapped out." -> measure DP per dollar
     of capital-at-risk (max loss), not per contract, and sweep the width.
  2. the tail is a DP-sizing problem -> test a regime/trend governor that down-weights
     entries in the regime that produced 2022.

Reuses BT-21 pricing verbatim (bt_dp_gate_pnl: bsm_put, short_strike_from_delta, cleaned
gate via layer1_gate) so each width reproduces the registered convention; only WIDTH and the
per-day contract weight change. Outcome leak-proof (expiry SPX close only). Hold-to-expiry.

Capital-at-risk per contract = (width - credit) * 100 = the Schwab max-loss / margin. This is
the denominator John cares about: $/ROC, not $/contract.

PRE-REGISTERED DECISION RULES (written before running):
  A. WIDTH: narrower beats 50-wide IF, at equal deployed capital, it delivers >= the same
     total P&L AND a worst-year loss no worse than 50-wide. (Capital efficiency = total P&L /
     total capital-at-risk; tie-break on worst-year-per-capital.)
  B. TREND GOVERNOR: down-weighting DP when SPX < its 200d SMA at entry "works" IF it shrinks
     the worst calendar-year loss >= 25% AND cuts max drawdown >= 10% AND keeps total EV per
     dollar of capital within 15% of flat. (2022 was ~entirely sub-200dma; this is the natural
     trend test. It is a SIZING taper, not a 6th gate condition.)
  Fractional contracts used for equal-capital and weighted books (removes lumpiness; a
  backtest comparison device, not a live order count). Drawdown sequential by entry date
  (understates concurrency, same caveat as BT-21). Credits BSM-modeled -> conservative floor.
"""
import os
import numpy as np
import pandas as pd
from layer1_gate import build_gate_features, evaluate_gate
from bt_dp_gate_pnl import (
    bsm_put, short_strike_from_delta, DATA, DTE_CAL, TARGET_DELTA, START_YEAR, MULT,
)

WIDTHS = [10, 25, 50]
TREND_K = [1.00, 0.50, 0.00]   # contract weight applied to sub-200dma entries
SMA_WIN = 200


def build_dp(width, markup):
    spx = pd.read_parquet(os.path.join(DATA, "yahoo_GSPC.parquet"))["Close"].sort_index()
    vix = pd.read_parquet(os.path.join(DATA, "yahoo_VIX.parquet"))["Close"].sort_index()
    sma = spx.rolling(SMA_WIN).mean()
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
        ma = float(sma.iloc[eloc])
        vloc = vix.index.get_indexer([dt], method="nearest")[0]
        sigma = float(vix.iloc[vloc]) / 100.0
        target = dt + pd.Timedelta(days=DTE_CAL)
        xloc = spx.index.get_indexer([target], method="nearest")[0]
        if (spx.index[xloc] - dt).days < DTE_CAL - 7:
            continue
        ST = float(spx.iloc[xloc])
        Ks = short_strike_from_delta(S, sigma, T, TARGET_DELTA)
        Kl = Ks - width
        credit = (bsm_put(S, Ks, sigma, T) - bsm_put(S, Kl, sigma, T)) * markup
        spread_val = max(Ks - ST, 0.0) - max(Kl - ST, 0.0)
        pnl = (credit - spread_val) * MULT
        maxloss = max((width - credit) * MULT, 1.0)
        below = (not np.isnan(ma)) and (S < ma)
        rows.append((dt, spx.index[xloc], dt.year, credit * MULT, pnl, maxloss, below))
    return pd.DataFrame(rows, columns=["date", "expiry", "year", "credit", "pnl", "maxloss", "below200"])


def max_dd(dates, pnls):
    order = np.argsort(dates.values)
    cum = np.cumsum(np.asarray(pnls)[order])
    return float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0


def block_width(markup):
    print(f"\n{'='*92}\n  BT-23 / H2-A — DP width & capital efficiency, markup {markup:.2f}  (1 contract / GO-day)")
    print(f"{'='*92}")
    print(f"  {'width':>5} {'avg credit':>11} {'avg maxloss':>12} {'total P&L':>12} "
          f"{'total capital':>14} {'ROC':>7} {'worst-yr':>10} {'wyr ROC':>8} {'maxDD':>11} {'WR':>6}")
    print(f"  {'-'*100}")
    books = {}
    for w in WIDTHS:
        df = build_dp(w, markup)
        books[w] = df
        by_yr = df.groupby("year").agg(pnl=("pnl", "sum"), cap=("maxloss", "sum"))
        worst = by_yr.pnl.min()
        worst_yr = by_yr.pnl.idxmin()
        wyr_roc = by_yr.loc[worst_yr, "pnl"] / by_yr.loc[worst_yr, "cap"] * 100
        total = df.pnl.sum()
        cap = df.maxloss.sum()
        dd = max_dd(df.date, df.pnl.values)
        wr = (df.pnl > 0).mean() * 100
        print(f"  {w:>5} ${df.credit.mean():>9,.0f} ${df.maxloss.mean():>10,.0f} "
              f"${total:>+10,.0f} ${cap:>12,.0f} {total/cap*100:>6.1f}% "
              f"${worst:>+9,.0f} {wyr_roc:>+7.1f}% ${-dd:>+10,.0f} {wr:>5.1f}%  ({worst_yr})")
    return books


def block_equal_capital(books, markup):
    base_cap = books[50].maxloss.sum()   # anchor: total capital of the live 50-wide book
    print(f"\n  --- H2-A equal-capital: every width scaled to the 50-wide book's deployed capital "
          f"(${base_cap:,.0f}) ---")
    print(f"  {'width':>5} {'scale x':>8} {'total P&L':>12} {'worst-yr':>11} {'2022':>11} "
          f"{'2018':>11} {'maxDD':>12}")
    print(f"  {'-'*78}")
    for w in WIDTHS:
        df = books[w]
        scale = base_cap / df.maxloss.sum()
        by_yr = (df.groupby("year").pnl.sum() * scale)
        dd = max_dd(df.date, df.pnl.values * scale)
        print(f"  {w:>5} {scale:>7.2f}x ${df.pnl.sum()*scale:>+10,.0f} "
              f"${by_yr.min():>+10,.0f} ${by_yr.get(2022,0):>+10,.0f} "
              f"${by_yr.get(2018,0):>+10,.0f} ${-dd:>+11,.0f}")


def block_trend(markup):
    df = build_dp(50, markup)   # governor tested on the live width
    base_total = df.pnl.sum()
    base_cap = df.maxloss.sum()
    print(f"\n{'='*92}\n  BT-23 / H2-B — trend governor on 50-wide DP: weight sub-200dma entries by k, "
          f"markup {markup:.2f}")
    print(f"{'='*92}")
    n_below = int(df.below200.sum())
    print(f"  sub-200dma GO-day entries: {n_below}/{len(df)} ({n_below/len(df)*100:.0f}%)   "
          f"flat book: total ${base_total:+,.0f}  cap ${base_cap:,.0f}")
    print(f"  {'k(sub)':>7} {'eff trades':>11} {'total P&L':>12} {'ROC':>7} {'worst-yr':>11} "
          f"{'2022':>11} {'2018':>11} {'maxDD':>12} {'verdict':>9}")
    print(f"  {'-'*96}")
    flat_dd = max_dd(df.date, df.pnl.values)
    flat_worst = df.groupby("year").pnl.sum().min()
    flat_roc = base_total / base_cap
    for k in TREND_K:
        wgt = np.where(df.below200.values, k, 1.0)
        pnl = df.pnl.values * wgt
        cap = df.maxloss.values * wgt
        by_yr = pd.Series(pnl).groupby(df.year.values).sum()
        dd = max_dd(df.date, pnl)
        total = pnl.sum()
        capsum = cap.sum()
        roc = total / capsum if capsum else 0.0
        worst = by_yr.min()
        eff = float(wgt.sum())
        better_worst = worst >= flat_worst * 0.75
        better_dd = dd <= flat_dd * 0.90
        keep_ev = roc >= flat_roc * 0.85
        verdict = "GOVERN" if (k < 1.0 and better_worst and better_dd and keep_ev) else "-"
        print(f"  {k:>6.2f} {eff:>10.0f} ${total:>+10,.0f} {roc*100:>6.1f}% "
              f"${worst:>+10,.0f} ${by_yr.get(2022,0):>+10,.0f} ${by_yr.get(2018,0):>+10,.0f} "
              f"${-dd:>+11,.0f} {verdict:>9}")
    print(f"\n  GOVERN = worst-yr >=25% less negative AND maxDD >=10% lower AND ROC within 15% of flat.")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        books = block_width(m)
        block_equal_capital(books, m)
        block_trend(m)
