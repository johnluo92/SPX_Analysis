"""
BT-21 crude: does opening SPX DP only on GO days keep more short strikes OTM?

Leak-proof by construction — the outcome (does SPX_expiry >= short strike) is a function of
SPX price ONLY. The cleaned gate (C4 publication-lagged 2026-06-18) decides GO/NO-GO; FRED
revisions can only move WHICH days are GO, never the survival outcome.

Crude DP proxy (no option chain, no BSM): on each trading day "sell" a put at a fixed OTM
distance d, expiring DTE_CAL calendar days later. "Survived" = SPX at expiry >= short strike
(put expires worthless). 5% OTM ~ 25-delta at VIX~20 / 44 DTE (1 sigma ~ 6.9%, 0.67 sigma).

Compares GO-day vs NO-GO-day entries, hold-to-expiry. Year-block (cluster) bootstrap for CI
because overlapping 44-DTE windows are autocorrelated. No closing logic — John 2026-06-18:
do not auto-close on gate flip; the only policy under test is don't-open-during-NO-GO.
"""
import os
import numpy as np
import pandas as pd
from layer1_gate import build_gate_features, evaluate_gate

DATA = os.path.join(os.path.dirname(__file__), "data_cache")
WARMUP = 63
DTE_CAL = 44
OTM_LIST = [0.03, 0.05, 0.07]
N_BOOT = 2000
RNG = np.random.default_rng(20260618)


def load_spx() -> pd.Series:
    return pd.read_parquet(os.path.join(DATA, "yahoo_GSPC.parquet"))["Close"].sort_index()


def build_trades() -> pd.DataFrame:
    spx = load_spx()
    f = build_gate_features()
    # evaluate_gate turns NaN inputs into False (NaN > 0 == False), so pre-2013 days (no VX data)
    # would silently masquerade as NO-GO and dump the 2008 GFC into that bucket — vacuous, the
    # BT-19 caveat. Keep only days where every gate input is actually computable.
    valid = (f["vx_contango_pct"].notna() & f["vix_vs_rv_21d"].notna()
             & f["vvix"].notna() & f["c3_vvix_threshold"].notna()
             & f["credit_widening_regime"].notna() & f["stress_regime"].notna())
    go = evaluate_gate(f)[valid]

    rows = []
    for dt in go.index:
        eloc = spx.index.get_indexer([dt], method="nearest")[0]
        spot = float(spx.iloc[eloc])
        target = dt + pd.Timedelta(days=DTE_CAL)
        xloc = spx.index.get_indexer([target], method="nearest")[0]
        xdate = spx.index[xloc]
        if (xdate - dt).days < DTE_CAL - 7:   # right-censored: no real expiry ahead
            continue
        xclose = float(spx.iloc[xloc])
        state = "GO" if bool(go.loc[dt]) else "NOGO"
        for d in OTM_LIST:
            short = spot * (1 - d)
            rows.append((dt.year, state, d, xclose >= short))
    return pd.DataFrame(rows, columns=["year", "state", "otm", "survived"])


def boot_diff(df: pd.DataFrame, otm: float) -> tuple[float, float, float, float, float]:
    """Year-block bootstrap of (GO survival - NOGO survival) at a given OTM distance."""
    sub = df[df["otm"] == otm]
    years = sub["year"].unique()
    by_year = {y: sub[sub["year"] == y] for y in years}
    go_rate = sub[sub.state == "GO"]["survived"].mean()
    nogo_rate = sub[sub.state == "NOGO"]["survived"].mean()
    diffs = []
    for _ in range(N_BOOT):
        pick = RNG.choice(years, size=len(years), replace=True)
        pool = pd.concat([by_year[y] for y in pick])
        g = pool[pool.state == "GO"]["survived"]
        n = pool[pool.state == "NOGO"]["survived"]
        if len(g) and len(n):
            diffs.append(g.mean() - n.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return go_rate * 100, nogo_rate * 100, (go_rate - nogo_rate) * 100, lo * 100, hi * 100


def main():
    df = build_trades()
    print(f"\nBT-21 crude OTM-survival — SPX DP, {DTE_CAL}d, "
          f"{df.year.min()}-{df.year.max()}, cleaned gate")
    print(f"trades: {len(df)//len(OTM_LIST)} entries x {len(OTM_LIST)} OTM bands\n")
    counts = df[df.otm == OTM_LIST[0]].groupby("state").size()
    print(f"entries: GO={counts.get('GO',0)}  NOGO={counts.get('NOGO',0)}\n")
    print(f"  {'OTM':>5} {'GO surv':>9} {'NOGO surv':>10} {'GO-NOGO':>9} {'95% CI (yr-block)':>22}")
    print(f"  {'-'*58}")
    for d in OTM_LIST:
        g, n, diff, lo, hi = boot_diff(df, d)
        sig = "" if (lo <= 0 <= hi) else "  *CI excludes 0"
        print(f"  {d*100:>4.0f}% {g:>8.1f}% {n:>9.1f}% {diff:>+8.1f}pp  [{lo:>+5.1f},{hi:>+5.1f}]pp{sig}")
    print()
    print("  Per-year GO survival @5% OTM:")
    y5 = df[(df.otm == 0.05) & (df.state == "GO")].groupby("year")["survived"].agg(["mean", "size"])
    for yr, r in y5.iterrows():
        print(f"    {yr}  {r['mean']*100:>5.1f}%  (n={int(r['size'])})")


if __name__ == "__main__":
    main()
