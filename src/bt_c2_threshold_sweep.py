#!/usr/bin/env python3
"""BT-19b: C2 cutpoint sweep — de-magic the +2 VRP threshold.

RESEARCH ONLY. Reuses the BT-19 harness (bt_gate_soft_voting). Sweeps C2's
cutpoint c in (VIX - RV21d) > c over a grid, holding C1/C3/C4/C5 at live
thresholds and ANDing as the live gate does. c=2.0 reproduces the baseline
exactly (internal consistency check). Answers: is +2 on a flat plateau, or
does another cut dominate on full-history DP outcome (WR/EV) with CI separation?

Outputs bt_c2_threshold_sweep_results.json for the markdown to cite.
"""
import json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from layer1_gate import build_gate_features, evaluate_gate
from bt_gate_soft_voting import (
    build_gate_features as _bgf, WARMUP, DATA_DIR, simulate_dp, load_spx,
    evaluate_config, VIX_R0,
)

OUT = os.path.join(os.path.dirname(__file__), "bt_c2_threshold_sweep_results.json")
CUTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]   # 2.0 = live threshold


def main():
    print("Building gate features (look-ahead-free)...")
    f = build_gate_features().iloc[WARMUP:]
    spx = load_spx()

    c1 = f["vx_contango_pct"] > 0
    c3 = f["vvix"] < f["c3_vvix_threshold"]
    c4 = f["credit_widening_regime"] == 0
    c5 = f["stress_regime"] <= 1
    base_others = c1 & c3 & c4 & c5   # C2 swept separately

    vix_s = pd.read_parquet(os.path.join(DATA_DIR, "yahoo_VIX.parquet"))["Close"].reindex(
        f.index, method="ffill", limit=5)
    print("Simulating DP outcomes per day...")
    sims = {}
    for d in f.index:
        v = vix_s.get(d)
        if v is None or pd.isna(v):
            continue
        s = simulate_dp(spx, d, float(v))
        if s is not None:
            sims[d] = s
    print(f"  simulable days: {len(sims)}")

    results = {"meta": {
        "feature_first": str(f.index[0].date()), "feature_last": str(f.index[-1].date()),
        "live_threshold": 2.0,
        "C1_first_valid": str(f["vx_contango_pct"].dropna().index[0].date()),
        "note": "C2' = (VIX - RV21d) > cut, ANDed with live C1&C3&C4&C5. c=2.0 must reproduce baseline.",
    }, "sweep": {}}

    for cut in CUTS:
        c2p = f["vix_vs_rv_21d"] > cut
        mask = base_others & c2p
        results["sweep"][f"c2_gt_{cut}"] = evaluate_config(
            f"C2 cut (VIX-RV21d) > {cut}", mask.fillna(False), f, sims, spx)

    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Wrote {OUT}\n")

    print(f"{'C2 cut':>8}  {'n':>5}  {'WR%':>6}  {'Wilson95':>16}  {'EV@1.15':>8}  {'edge_pp':>7}  {'tr/yr':>6}  {'2022pnl':>8}")
    for cut in CUTS:
        v = results["sweep"][f"c2_gt_{cut}"]
        if v.get("n", 0) == 0:
            print(f"{cut:>8}  n=0"); continue
        m = v["markups"]["1.15"]
        t22 = v["tail_years"]["2022"]["pnl_115"]
        flag = "  <-- LIVE" if cut == 2.0 else ""
        print(f"{cut:>8}  {v['n']:>5}  {v['wr']:>6.2f}  "
              f"[{v['wr_wilson'][0]:>5.1f},{v['wr_wilson'][1]:>5.1f}]  "
              f"{m['ev_per_trade']:>8.4f}  {m['wr_minus_be']:>7.2f}  "
              f"{v['trades_yr']:>6.2f}  {t22:>8.1f}{flag}")


if __name__ == "__main__":
    main()
