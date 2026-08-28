"""
BT-21 P&L: if we repeatedly sold SPX put spreads on every GO day (last 10y), what's the
economics? WR, EV/trade, avg win/loss, R:R, max drawdown — the plain-terms question.

Method (BACKTEST_REGISTRY standard): 25-delta short put via BSM with VIX as IV, 50-pt wide,
44 DTE, credit at markup {1.00, 1.15}. Outcome at expiry from SPX close only (leak-proof
outcome; cleaned gate decides GO). Hold-to-expiry, no closing (John 2026-06-18).

Drawdown caveat: trades overlap (44d windows, many concurrent); cumulative-P&L drawdown here
treats them sequentially by entry date — a SIMPLIFICATION (understates true concurrent risk).
Worst calendar-year P&L is the honest concurrency-free tail. Per-contract = points x $100.
Survivorship: SPX index, minimal. Credit is BSM-modeled (live SPX exec ran richer per BT-16).
"""
import math
import os
import numpy as np
import pandas as pd
from layer1_gate import build_gate_features, evaluate_gate

DATA = os.path.join(os.path.dirname(__file__), "data_cache")
DTE_CAL = 44
WIDTH = 50.0
TARGET_DELTA = 0.25
START_YEAR = 2016
MULT = 100.0


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bsm_put(S, K, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * ncdf(-d2) - S * ncdf(-d1)


def short_strike_from_delta(S, sigma, T, delta):
    d1 = -_inv_norm(delta)           # N(-d1)=delta  ->  -d1 = Phi^{-1}(delta)
    return S * math.exp(0.5 * sigma * sigma * T - d1 * sigma * math.sqrt(T))


def _inv_norm(p):
    # Acklam rational approximation, plenty for delta targeting
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def build_pnl(markup):
    spx = pd.read_parquet(os.path.join(DATA, "yahoo_GSPC.parquet"))["Close"].sort_index()
    vix = pd.read_parquet(os.path.join(DATA, "yahoo_VIX.parquet"))["Close"].sort_index()
    f = build_gate_features()
    valid = (f["vx_contango_pct"].notna() & f["vix_vs_rv_21d"].notna()
             & f["vvix"].notna() & f["c3_vvix_threshold"].notna()
             & f["credit_widening_regime"].notna() & f["stress_regime"].notna())
    go = evaluate_gate(f)[valid]
    go = go[(go.index.year >= START_YEAR) & go]   # GO days only, last 10y

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
        spread_val = max(Ks - ST, 0.0) - max(Kl - ST, 0.0)   # 0..WIDTH
        pnl = (credit - spread_val) * MULT
        rows.append((dt, dt.year, credit * MULT, pnl, ST >= Ks))
    return pd.DataFrame(rows, columns=["date", "year", "credit", "pnl", "otm"])


def report(markup):
    df = build_pnl(markup).sort_values("date").reset_index(drop=True)
    n = len(df)
    wr = (df.pnl > 0).mean() * 100
    otm = df.otm.mean() * 100
    ev = df.pnl.mean()
    wins = df[df.pnl > 0].pnl
    losses = df[df.pnl <= 0].pnl
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    worst = df.pnl.min()
    rr = abs(avg_loss) / avg_win if avg_win else float("nan")
    cum = df.pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    total = df.pnl.sum()
    yrs = df.year.nunique()

    print(f"\n  ===== SPX DP on every GO day, {df.year.min()}-{df.year.max()}, markup {markup:.2f} =====")
    print(f"  trades:        {n}  ({n/yrs:.0f}/yr over {yrs} yrs)")
    print(f"  avg credit:    ${df.credit.mean():.0f}/contract  (25d short, {WIDTH:.0f}-wide)")
    print(f"  win rate:      {wr:.1f}%   (OTM-survival {otm:.1f}%)")
    print(f"  EV / trade:    ${ev:+.0f}")
    print(f"  avg win:       ${avg_win:+.0f}    avg loss: ${avg_loss:+.0f}    R:R {rr:.2f}:1")
    print(f"  worst trade:   ${worst:+.0f}")
    print(f"  max drawdown:  ${dd:+.0f}  (sequential; understates concurrency)")
    print(f"  total P&L:     ${total:+,.0f}/contract over {yrs} yrs (${total/yrs:+,.0f}/yr)")
    print(f"  by year:")
    for yr, g in df.groupby("year"):
        print(f"    {yr}  n={len(g):>3}  WR {(g.pnl>0).mean()*100:>5.1f}%  "
              f"P&L ${g.pnl.sum():>+7,.0f}  worst ${g.pnl.min():>+6,.0f}")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
