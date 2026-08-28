"""
BT-31 (H8c): how much volatility-SKEW premium can the H8b 10Δ-put @0.5× tail hedge absorb before
it stops nullifying the DP down-years? BT-30 found the hedge WORKS at BSM-flat-VIX pricing but
flagged that real far-OTM SPX puts trade well above ATM vol (skew), which the model ignores —
making the result optimistic. Rather than guess, bracket it: re-price the long put with
sigma_put = sigma_ATM * (1 + UPLIFT) and find the breakeven UPLIFT where the hedge drops below the
pre-reg "works" bar. Then compare to the empirically plausible skew premium for SPX 10Δ puts.

This is the gating question for the only viable hedge. A flat IV uplift is a deliberate
simplification (true skew varies by date/regime, and is steepest in the stress where the hedge
pays — so a constant uplift is itself conservative-to-optimistic depending on timing), but it
correctly answers "how fat a skew tax can this survive."

PRE-REGISTERED DECISION RULE (written before running):
  Report the largest UPLIFT at which 10Δ-put @0.5× still clears BT-30's bar (worst-yr cut >=50%
  AND maxDD cut >=25% AND >=75% of DP total retained), at both markups. SPX 10Δ 1-month puts
  empirically trade ~30-60% above ATM IV in calm regimes (steeper in stress). If the breakeven
  uplift is BELOW ~30%, the hedge does not survive realistic skew -> reject as uneconomic, keep
  200dma-skip + 2x stop. If breakeven is ABOVE ~50-60%, the hedge is robust to skew -> promote to
  ADOPT. In between -> marginal, needs live chain validation.
CAVEATS: only the LONG PUT is re-priced (the DP short spread keeps BSM-flat — DP is near-ATM where
  skew is mild, and pricing DP richer would only help). Flat uplift, not a real surface. 2016+.
"""
import os
import numpy as np
import pandas as pd
from layer1_gate import build_gate_features, evaluate_gate
from bt_dp_gate_pnl import (
    bsm_put, short_strike_from_delta, DATA, DTE_CAL, TARGET_DELTA, START_YEAR, MULT, WIDTH,
)

HEDGE_DELTA = 0.10
RATIO = 0.5
UPLIFTS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00]


def build(markup, uplift):
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
        Ks = short_strike_from_delta(S, sigma, T, TARGET_DELTA)
        Kl = Ks - WIDTH
        credit = (bsm_put(S, Ks, sigma, T) - bsm_put(S, Kl, sigma, T)) * markup
        dp_pnl = (credit - (max(Ks - ST, 0.0) - max(Kl - ST, 0.0))) * MULT
        Kp = short_strike_from_delta(S, sigma, T, HEDGE_DELTA)   # strike fixed at ATM-vol delta
        cost = bsm_put(S, Kp, sigma * (1 + uplift), T)           # but PRICED at skewed IV
        put_pnl = (max(Kp - ST, 0.0) - cost) * MULT
        rows.append((dt, spx.index[xloc], dt.year, dp_pnl, put_pnl))
    return pd.DataFrame(rows, columns=["date", "expiry", "year", "dp", "put"])


def realized_dd(df, col):
    s = df.sort_values("expiry")
    cum = s[col].cumsum().to_numpy()
    return float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0


def report(markup):
    base = build(markup, 0.0)
    dp_total, dp_worst, dp_dd = base.dp.sum(), base.groupby("year").dp.sum().min(), realized_dd(base, "dp")
    print(f"\n{'='*92}")
    print(f"  BT-31 / H8c — 10Δ-put @0.5× hedge vs SKEW uplift, markup {markup:.2f}")
    print(f"{'='*92}")
    print(f"  DP-alone: total ${dp_total:+,.0f}  worst-yr ${dp_worst:+,.0f}  maxDD ${-dp_dd:,.0f}")
    print(f"  {'IV uplift':>10} {'total P&L':>12} {'%kept':>6} {'worst-yr':>11} {'2022':>11} "
          f"{'maxDD':>11} {'works':>6}")
    print(f"  {'-'*78}")
    for up in UPLIFTS:
        df = build(markup, up)
        net = df.dp + RATIO * df.put
        tmp = df.assign(net=net)
        total = net.sum()
        worst = tmp.groupby("year").net.sum().min()
        dd = realized_dd(tmp, "net")
        works = (worst >= dp_worst * 0.50 and dd <= dp_dd * 0.75 and total >= dp_total * 0.75)
        by = tmp.groupby("year").net.sum()
        print(f"  {up*100:>+9.0f}% ${total:>+10,.0f} {total/dp_total*100:>5.0f}% ${worst:>+10,.0f} "
              f"${by.get(2022,0):>+10,.0f} ${-dd:>+10,.0f} {'YES' if works else 'no':>6}")
    print(f"\n  works = worst-yr cut >=50% AND maxDD cut >=25% AND >=75% total retained.")
    print(f"  SPX 10Δ 1-mo puts empirically ~+30-60% over ATM IV (calm); steeper in stress.")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
