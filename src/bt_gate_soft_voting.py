#!/usr/bin/env python3
"""BT-19: Soft / weighted-voting gate vs hard-AND(C1..C5) — DP outcome backtest.

RESEARCH ONLY. Does not modify the live gate. Reuses layer1_gate.build_gate_features
and the spxtrader DP strike geometry (recommend.py). Pre-registered spec in
equity_screener/signal_validation/bt_gate_soft_voting_results.md.

Outputs JSON to bt_gate_soft_voting_results.json for the markdown to cite.
"""
import json, math, os, sys
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from layer1_gate import build_gate_features, evaluate_gate

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
WARMUP = 63
OUT = os.path.join(os.path.dirname(__file__), "bt_gate_soft_voting_results.json")

# DP geometry — sourced from spxtrader/recommend.py (_REGIME_PARAMS, _build_recommendation).
# R0 (VIX<16.77): DTE=45, W=10, K=0.50.  R1 (16.77<=VIX<24.41): DTE=21, W=10, K=0.65.
# K_short = S - K*EM_1sd ; EM_1sd = S*(VIX/100)*sqrt(DTE/365) ; credit = BSM put-diff.
VIX_R0, VIX_R1 = 16.77, 24.41
MARKUPS = [1.0, 1.15, 1.30]
GRID = 5

def regime_params(vix):
    if vix < VIX_R0:   return dict(dte=45, W=10, K=0.50)
    return dict(dte=21, W=10, K=0.65)   # R1; R2+ rarely GO, use R1 geometry

def bsm_put(S, K, sigma, T):
    if T <= 0: return max(0.0, K - S)
    if sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)

def load_spx():
    return pd.read_parquet(os.path.join(DATA_DIR, "yahoo_GSPC.parquet"))["Close"].sort_index()

def fwd_close(spx, date, dte_cal):
    """SPX close nearest to date + dte calendar days (expiry)."""
    target = date + timedelta(days=dte_cal)
    locs = spx.index.get_indexer([target], method="nearest")
    loc = locs[0]
    if loc < 0 or loc >= len(spx): return None
    # guard: expiry must be in the future relative to entry
    entry_loc = spx.index.get_indexer([date], method="nearest")[0]
    if loc <= entry_loc: return None
    return float(spx.iloc[loc])

def simulate_dp(spx, date, vix):
    """One DP spread per the live geometry. Returns dict with credit_pct (BSM, markup 1.0),
    win flag at expiry (binary OTM), max_loss_per_pt, or None if not simulable."""
    p = regime_params(vix)
    dte, W, K = p["dte"], p["W"], p["K"]
    locs = spx.index.get_indexer([date], method="nearest")[0]
    if locs < 0: return None
    S = float(spx.iloc[locs])
    T = dte / 365.0
    sigma = vix / 100.0
    em = S * sigma * math.sqrt(T)
    K_short = round((S - K * em) / GRID) * GRID
    K_long = K_short - W
    credit = bsm_put(S, K_short, sigma, T) - bsm_put(S, K_long, sigma, T)
    credit_pct = credit / W   # fraction of width, markup 1.0
    S_exp = fwd_close(spx, date, dte)
    if S_exp is None: return None
    # bull put: win if SPX stays above short strike at expiry (short put expires OTM)
    win = S_exp >= K_short
    if win:
        loss_pts = 0.0
    else:
        intrinsic = min(W, K_short - S_exp)   # capped at width
        loss_pts = intrinsic
    return dict(credit_pct=credit_pct, win=win, W=W, loss_pts=loss_pts,
                K_short=K_short, S=S, S_exp=S_exp, dte=dte, vix=vix)

def pnl_at_markup(sim, markup):
    """Net P&L in points for one spread at given credit markup. credit capped at width."""
    cpct = min(sim["credit_pct"] * markup, 0.99)
    credit_pts = cpct * sim["W"]
    if sim["win"]:
        return credit_pts
    return credit_pts - sim["loss_pts"]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d*100, (c+h)/d*100)

def year_block_bootstrap(df, value_col, B=2000, seed=7):
    """Stratified-by-year block bootstrap of the mean of value_col. Resample years with replacement."""
    if len(df) < 2: return (np.nan, np.nan)
    years = df.index.year if hasattr(df.index, "year") else df["year"]
    groups = {y: df[df["year"] == y][value_col].values for y in df["year"].unique()}
    yrs = list(groups.keys())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        pick = rng.choice(yrs, size=len(yrs), replace=True)
        vals = np.concatenate([groups[y] for y in pick])
        if len(vals): means.append(vals.mean())
    if not means: return (np.nan, np.nan)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

def wr_bootstrap(df, B=2000, seed=11):
    """Stratified-by-year bootstrap CI for win rate (%)."""
    if len(df) < 2: return (np.nan, np.nan)
    groups = {y: df[df["year"] == y]["win"].values for y in df["year"].unique()}
    yrs = list(groups.keys())
    rng = np.random.default_rng(seed)
    wrs = []
    for _ in range(B):
        pick = rng.choice(yrs, size=len(yrs), replace=True)
        vals = np.concatenate([groups[y] for y in pick])
        if len(vals): wrs.append(vals.mean()*100)
    return (float(np.percentile(wrs, 2.5)), float(np.percentile(wrs, 97.5)))

def evaluate_config(name, go_mask, f, sims_by_date, spx):
    """go_mask: boolean Series aligned to f.index. Returns metrics dict."""
    go_dates = f.index[go_mask.reindex(f.index).fillna(False)]
    rows = []
    for d in go_dates:
        sim = sims_by_date.get(d)
        if sim is None: continue
        rows.append(dict(date=d, year=d.year, win=int(sim["win"]),
                         credit_pct=sim["credit_pct"], loss_pts=sim["loss_pts"],
                         W=sim["W"], vix=sim["vix"]))
    if not rows:
        return dict(name=name, n=0, note="no simulable GO days")
    df = pd.DataFrame(rows).set_index("date")
    n = len(df)
    wins = int(df["win"].sum())
    wr = wins / n * 100
    wr_ci = wilson(wins, n)
    wr_bci = wr_bootstrap(df)

    span_years = (df.index[-1] - df.index[0]).days / 365.25
    trades_yr = n / span_years if span_years > 0 else float("nan")

    # regime split
    df["regime"] = np.where(df["vix"] < VIX_R0, "R0", "R1+")
    reg = {}
    for r in ["R0", "R1+"]:
        sub = df[df["regime"] == r]
        if len(sub):
            reg[r] = dict(n=int(len(sub)), wr=float(sub["win"].mean()*100))

    out = dict(name=name, n=n, wins=wins, wr=round(wr,2),
               wr_wilson=[round(wr_ci[0],2), round(wr_ci[1],2)],
               wr_yearboot=[round(wr_bci[0],2), round(wr_bci[1],2)],
               trades_yr=round(trades_yr,2), regime=reg, markups={})

    # per-markup EV, EV/maxloss, drawdown, per-year P&L
    for m in MARKUPS:
        df[f"pnl_{m}"] = df.apply(lambda row: pnl_at_markup(
            dict(credit_pct=row["credit_pct"], win=bool(row["win"]),
                 W=row["W"], loss_pts=row["loss_pts"]), m), axis=1)
        pnl = df[f"pnl_{m}"]
        ev = float(pnl.mean())
        # max loss per trade = width - min credit (worst single-trade outcome possible) — use realized worst
        maxloss = float(df["W"].mean())  # per-pt width; report EV/maxL as ev/avg_width
        cum = pnl.cumsum()
        running_max = cum.cummax()
        max_dd = float((running_max - cum).max())
        cr = df["credit_pct"].mean() * m
        wr_be = (1 - min(cr, 0.99)) * 100
        ev_ci = year_block_bootstrap(df.assign(year=df.index.year), f"pnl_{m}")
        per_year = (df.assign(year=df.index.year)
                      .groupby("year")[f"pnl_{m}"].sum().round(3).to_dict())
        out["markups"][str(m)] = dict(
            ev_per_trade=round(ev,4), ev_ci=[round(ev_ci[0],4), round(ev_ci[1],4)],
            ev_per_maxloss=round(ev/maxloss,5), mean_credit_pct=round(cr*100,2),
            wr_be=round(wr_be,2), wr_minus_be=round(wr - wr_be,2),
            max_drawdown_pts=round(max_dd,3),
            per_year_pnl={str(k): v for k,v in per_year.items()})
    # tail-year table
    tail = {}
    for ty in [2007, 2008, 2009, 2010, 2011, 2012, 2018, 2020, 2022]:
        sub = df[df.index.year == ty]
        tail[str(ty)] = dict(go_days=int(len(sub)),
                             wins=int(sub["win"].sum()) if len(sub) else 0,
                             wr=round(float(sub["win"].mean()*100),1) if len(sub) else None,
                             pnl_115=round(float(sub["pnl_1.15"].sum()),3) if len(sub) else 0.0)
    out["tail_years"] = tail
    return out

def soft_score(f, weights, squash="logistic", kappa=1.0):
    """Continuous soft-vote score over C1,C2,C3 margins (standardized).
    Margins (positive = clears threshold):
      C1: vx_contango_pct - 0
      C2: vix_vs_rv_21d - 2
      C3: (c3_vvix_threshold - vvix)   (positive when VVIX below threshold)
    Each margin standardized by its own rolling-sample std (full-sample std here, fit window applied by caller).
    g_i = logistic(kappa * z_i) in (0,1).  S = sum w_i * g_i.
    Returns S Series and the per-condition g for inspection."""
    m1 = f["vx_contango_pct"] - 0.0
    m2 = f["vix_vs_rv_21d"] - 2.0
    m3 = f["c3_vvix_threshold"] - f["vvix"]
    margins = {"C1": m1, "C2": m2, "C3": m3}
    g = {}
    for c, m in margins.items():
        sd = m.std()
        z = m / sd if sd and not np.isnan(sd) else m * 0.0
        if squash == "logistic":
            g[c] = 1.0 / (1.0 + np.exp(-kappa * z))
        elif squash == "tanh":
            g[c] = 0.5 * (np.tanh(kappa * z) + 1.0)
        else:
            raise ValueError(squash)
    S = sum(weights[c] * g[c] for c in margins)
    return S, g

def main():
    print("Building gate features (fresh, look-ahead-free)...")
    f = build_gate_features().iloc[WARMUP:]
    go_hard = evaluate_gate(f).iloc[WARMUP:].reindex(f.index)
    spx = load_spx()

    c1 = f["vx_contango_pct"] > 0
    c2 = f["vix_vs_rv_21d"] > 2
    c3 = f["vvix"] < f["c3_vvix_threshold"]
    c4 = f["credit_widening_regime"] == 0
    c5 = f["stress_regime"] <= 1
    conds = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}

    # Pre-simulate DP outcome on EVERY day that has VIX (so any config can index it).
    # Use VIX from features-aligned series.
    vix_s = pd.read_parquet(os.path.join(DATA_DIR, "yahoo_VIX.parquet"))["Close"].reindex(f.index, method="ffill", limit=5)
    print("Simulating DP outcomes per day...")
    sims = {}
    for d in f.index:
        v = vix_s.get(d)
        if v is None or pd.isna(v): continue
        s = simulate_dp(spx, d, float(v))
        if s is not None: sims[d] = s
    print(f"  simulable days: {len(sims)}")

    results = {"meta": {
        "feature_first": str(f.index[0].date()), "feature_last": str(f.index[-1].date()),
        "n_days": len(f), "hard_go_days": int(go_hard.fillna(False).sum()),
        "simulable_days": len(sims),
        "C1_first_valid": str(f["vx_contango_pct"].dropna().index[0].date()),
        "geometry": "R0: DTE45 W10 K0.50 ; R1+: DTE21 W10 K0.65 ; K_short=S-K*EM, EM=S*(VIX/100)*sqrt(DTE/365); BSM put-diff credit (recommend.py)",
    }, "configs": {}}

    # CONFIG 1: BASELINE hard AND(C1..C5)
    results["configs"]["baseline_hardAND"] = evaluate_config(
        "BASELINE hard AND(C1..C5)", go_hard.fillna(False), f, sims, spx)

    # CONFIG 2/3: leave-one-out for each condition
    for drop in ["C1","C2","C3","C4","C5"]:
        kept = [c for c in ["C1","C2","C3","C4","C5"] if c != drop]
        m = conds[kept[0]].copy()
        for c in kept[1:]: m = m & conds[c]
        results["configs"][f"drop_{drop}"] = evaluate_config(
            f"Leave-one-out drop {drop} (AND of remaining 4)", m, f, sims, spx)

    # CONFIG 4: k-of-n majority
    cond_sum = sum(conds[c].astype(int) for c in ["C1","C2","C3","C4","C5"])
    for k in [3, 4, 5]:
        results["configs"][f"kofn_ge{k}"] = evaluate_config(
            f"k-of-5 majority: open if >= {k} conditions pass", cond_sum >= k, f, sims, spx)

    # CONFIG 4b: k-of-n majority WITH C4&C5 as hard vetoes (tail guard)
    veto = c4 & c5
    soft3_sum = sum(conds[c].astype(int) for c in ["C1","C2","C3"])
    for k in [2, 3]:
        results["configs"][f"kofn_C123_ge{k}_veto_C45"] = evaluate_config(
            f"open if >= {k} of (C1,C2,C3) pass AND C4&C5 hard veto", (soft3_sum >= k) & veto, f, sims, spx)

    # CONFIG 5: WEIGHTED SOFT vote over C1,C2,C3, C4&C5 hard veto.
    # Fit threshold tau on IN-SAMPLE 2013-2019 to match hard-AND base rate region, lock, eval OOS.
    weights = {"C1": 1.0, "C2": 1.0, "C3": 1.0}
    for kappa in [0.75, 1.0, 1.5]:
        S, _ = soft_score(f, weights, "logistic", kappa)
        # in-sample window for tau calibration
        is_mask = (f.index.year >= 2013) & (f.index.year <= 2019)
        S_is = S[is_mask & veto.reindex(f.index).fillna(False)]
        # sweep tau over a grid; pick tau that, in-sample with veto, gives WR closest to >= hard-AND WR
        # but admits MORE trades (the soft-gate promise). Decision metric = in-sample EV@1.15.
        taus = np.round(np.arange(0.6*sum(weights.values()), 0.95*sum(weights.values())+1e-9, 0.03), 4)
        best = None
        for tau in taus:
            mask = (S >= tau) & veto
            m_is = mask & is_mask
            res = evaluate_config(f"soft k={kappa} tau={tau}", m_is, f, sims, spx)
            if res["n"] < 30: continue
            ev = res["markups"]["1.15"]["ev_per_trade"]
            wr = res["wr"]
            if best is None or ev > best[1]:
                best = (tau, ev, wr, res["n"])
        if best is None:
            results["configs"][f"soft_k{kappa}"] = dict(name=f"soft k={kappa}", note="no in-sample tau yielded n>=30")
            continue
        tau = best[0]
        mask_full = (S >= tau) & veto
        full = evaluate_config(f"SOFT logistic k={kappa} tau={tau} (C4&C5 veto)", mask_full.fillna(False), f, sims, spx)
        # explicit in-sample vs OOS split
        is_res  = evaluate_config(f"soft IS 2013-2019",  mask_full & ((f.index.year>=2013)&(f.index.year<=2019)), f, sims, spx)
        oos_res = evaluate_config(f"soft OOS 2020-2026", mask_full & (f.index.year>=2020), f, sims, spx)
        full["chosen_tau"] = float(tau)
        full["kappa"] = kappa
        full["in_sample_2013_2019"] = {"n": is_res["n"], "wr": is_res.get("wr"),
            "ev_115": is_res.get("markups",{}).get("1.15",{}).get("ev_per_trade")}
        full["oos_2020_2026"] = {"n": oos_res["n"], "wr": oos_res.get("wr"),
            "ev_115": oos_res.get("markups",{}).get("1.15",{}).get("ev_per_trade")}
        results["configs"][f"soft_k{kappa}"] = full

    # Also: hard-AND in-sample vs OOS split for the gap reference
    hb = go_hard.fillna(False)
    results["configs"]["baseline_IS_2013_2019"] = evaluate_config(
        "baseline IS 2013-2019", hb & ((f.index.year>=2013)&(f.index.year<=2019)), f, sims, spx)
    results["configs"]["baseline_OOS_2020_2026"] = evaluate_config(
        "baseline OOS 2020-2026", hb & (f.index.year>=2020), f, sims, spx)

    # CONFIG 6: a deliberately RELAXED soft config — admits MORE than baseline.
    # Soft over C1,C2,C3 with LOW tau (open easily) but keep C4&C5 hard veto.
    # This is the honest "loosening" arm John asked about.
    S10, _ = soft_score(f, {"C1":1.0,"C2":1.0,"C3":1.0}, "logistic", 1.0)
    for tau in [1.5, 1.8, 2.1]:
        mask = (S10 >= tau) & veto
        results["configs"][f"soft_relaxed_tau{tau}"] = evaluate_config(
            f"RELAXED soft logistic k=1.0 tau={tau} (C4&C5 veto, admits more)", mask.fillna(False), f, sims, spx)

    # ---- SIMPLIFIED-GATE 2007-2012 HOLDOUT (the only pre-2013 tail test possible) ----
    # C1 (VX contango) has NO data before 2013-05-01, so the full gate cannot be evaluated
    # pre-2013. The pre-2013 stress holdout is therefore run on the SIMPLIFIED gate C2&C3&C5
    # (all have 2007+ data) — this is the gate variant the prior stress tests used.
    simp = (c2 & c3 & c5)
    pre = (f.index.year >= 2007) & (f.index.year <= 2012)
    results["configs"]["simplified_C2C3C5_2007_2012"] = evaluate_config(
        "SIMPLIFIED gate C2&C3&C5 — 2007-2012 structural holdout", simp & pre, f, sims, spx)
    results["configs"]["simplified_C2C3C5_2013_2026"] = evaluate_config(
        "SIMPLIFIED gate C2&C3&C5 — 2013-2026 in-sample reference", simp & (f.index.year>=2013), f, sims, spx)
    # Same simplified gate, drop the stress veto C5 — does 2008 reopen?
    simp_noC5 = (c2 & c3)
    results["configs"]["simplified_C2C3_noC5_2007_2012"] = evaluate_config(
        "SIMPLIFIED no-C5 (C2&C3 only) — 2007-2012: does dropping stress reopen 2008?", simp_noC5 & pre, f, sims, spx)

    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Wrote {OUT}")
    # quick console summary
    for k, v in results["configs"].items():
        if v.get("n",0) == 0:
            print(f"{k:34s} n=0 ({v.get('note','')})"); continue
        m = v.get("markups",{}).get("1.15",{})
        print(f"{k:34s} n={v['n']:4d} WR={v['wr']:.1f}% [{v['wr_wilson'][0]:.1f},{v['wr_wilson'][1]:.1f}] "
              f"EV115={m.get('ev_per_trade')} edge={m.get('wr_minus_be')}pp tr/yr={v['trades_yr']}")

if __name__ == "__main__":
    main()
