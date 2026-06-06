#!/usr/bin/env python3
"""
GO->NO-GO gate transition backtest.
Computes forward SPX returns at 1d/5d/10d/21d horizons for:
  1. All GO->NO-GO transitions
  2. C4-specific transitions (C4 first fails while gate was open)
  3. Pre-transition SPX drift (lead/lag check) for C4 events
  4. Median days until gate reopens + full closed-period SPX return
  5. Baseline: all trading days; Control: random GO days (no transition)
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from layer1_gate import build_gate_features, evaluate_gate

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
HORIZONS = [1, 5, 10, 21]
WARMUP = 63  # rows dropped before any analysis (matches layer1_gate warmup)


def load_spx() -> pd.Series:
    spx_raw = pd.read_parquet(os.path.join(DATA_DIR, "yahoo_GSPC.parquet"))
    return spx_raw["Close"].sort_index()


def fwd_return(spx: pd.Series, date, h: int) -> float | None:
    """Log return from 'date' close to 'date+h trading days' close."""
    locs = spx.index.get_indexer([date], method="nearest")
    loc = locs[0]
    if loc < 0 or loc + h >= len(spx):
        return None
    return np.log(spx.iloc[loc + h] / spx.iloc[loc]) * 100


def bwd_return(spx: pd.Series, date, h: int) -> float | None:
    """Log return from 'date-h trading days' close to 'date' close."""
    locs = spx.index.get_indexer([date], method="nearest")
    loc = locs[0]
    if loc < 0 or loc - h < 0:
        return None
    return np.log(spx.iloc[loc] / spx.iloc[loc - h]) * 100


def collect_forward_returns(events: list, spx: pd.Series) -> pd.DataFrame:
    rows = []
    for date in events:
        row = {"date": date}
        for h in HORIZONS:
            row[f"fwd_{h}d"] = fwd_return(spx, date, h)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def print_table(label: str, df: pd.DataFrame, horizons=HORIZONS):
    print(f"\n{label}")
    print(f"  n = {len(df)}")
    header = "  " + "".join(f"{'fwd_'+str(h)+'d':>10}" for h in horizons)
    print(header)
    for stat_name, func in [("mean", np.mean), ("median", np.median),
                             ("pct_pos", lambda x: np.mean(np.array(x) > 0) * 100)]:
        row_str = f"  {stat_name:<9}"
        for h in horizons:
            col = f"fwd_{h}d"
            vals = df[col].dropna().tolist()
            n = len(vals)
            if n < 5:
                row_str += f"{'n<5':>10}"
            else:
                row_str += f"{func(vals):>9.2f} "
        print(row_str)


def main():
    print("Loading gate features...")
    f = build_gate_features()
    gate = evaluate_gate(f)

    # Drop warmup rows
    f = f.iloc[WARMUP:]
    gate = gate.iloc[WARMUP:]

    # Align gate to only dates with non-NaN gate value
    valid = gate.notna()
    gate = gate[valid]
    f = f[valid]

    spx = load_spx()

    print(f"Gate series: {gate.index[0].date()} to {gate.index[-1].date()}, n={len(gate)} days")
    print(f"Gate open: {gate.sum()} days ({gate.mean()*100:.1f}%)")
    print(f"SPX series: {spx.index[0].date()} to {spx.index[-1].date()}")

    # ---- Identify individual condition passes per day ----
    c1 = f["vx_contango_pct"] > 0
    c2 = f["vix_vs_rv_21d"] > 2
    c3 = f["vvix"] < f["c3_vvix_threshold"]
    c4 = f["credit_widening_regime"] == 0
    c5 = f["stress_regime"] <= 1

    # ---- GO->NO-GO transition events ----
    # gate[t-1]=True, gate[t]=False
    gate_arr = gate.astype(float)
    transitions_go_to_nogo = gate_arr.index[
        (gate_arr.shift(1) == 1) & (gate_arr == 0)
    ].tolist()

    print(f"\nGO->NO-GO transitions: n={len(transitions_go_to_nogo)}")

    # ---- C4-specific transitions ----
    # C4 first fails (c4[t-1]=True, c4[t]=False) while gate was previously open
    # "gate previously open" = gate[t-1]=True
    c4_arr = c4.astype(float)
    prev_gate_open = gate_arr.shift(1) == 1
    c4_first_fail = (c4_arr.shift(1) == 1) & (c4_arr == 0)
    c4_transitions = c4_arr.index[c4_first_fail & prev_gate_open].tolist()

    print(f"C4-specific transitions (C4 first fails while gate open): n={len(c4_transitions)}")

    # ---- Forward returns for GO->NO-GO events ----
    all_transitions_df = collect_forward_returns(transitions_go_to_nogo, spx)

    # ---- Forward returns for C4 transitions ----
    c4_transitions_df = collect_forward_returns(c4_transitions, spx)

    # ---- Baseline: all valid trading days ----
    all_days = gate.index.tolist()
    baseline_df = collect_forward_returns(all_days, spx)

    # ---- Control: random GO days (no transition) ----
    go_days_set = set(gate_arr.index[gate_arr == 1].tolist())
    transition_set = set(transitions_go_to_nogo)
    control_days = [d for d in go_days_set if d not in transition_set]
    control_df = collect_forward_returns(control_days, spx)

    # ===== PRINT RESULTS =====
    print("\n" + "=" * 65)
    print("  GATE BACKTEST: GO->NO-GO TRANSITION FORWARD RETURNS")
    print("=" * 65)
    print("  Metric: log return (%) from transition day close")
    print("  Horizons: +1d, +5d, +10d, +21d trading days")
    print()

    print_table("BASELINE — All valid trading days", baseline_df)
    print_table("CONTROL  — GO days (no transition)", control_df)
    print_table("ALL GO->NO-GO transitions", all_transitions_df)
    print_table("C4-specific transitions (C4 first fails, gate was open)", c4_transitions_df)

    # ---- C4 lead/lag check: pre-transition SPX drift ----
    print("\n" + "-" * 65)
    print("  C4 LEAD/LAG CHECK — SPX return BEFORE the transition event")
    print("  (negative = SPX was already falling before gate flipped)")
    print(f"  n = {len(c4_transitions)}")
    bwd_horizons = [1, 3, 5]
    header = "  " + "".join(f"{'bwd_'+str(h)+'d':>10}" for h in bwd_horizons)
    print(header)
    for stat_name, func in [("mean", np.mean), ("median", np.median)]:
        row_str = f"  {stat_name:<9}"
        for h in bwd_horizons:
            vals = [bwd_return(spx, d, h) for d in c4_transitions]
            vals = [v for v in vals if v is not None]
            if len(vals) < 5:
                row_str += f"{'n<5':>10}"
            else:
                row_str += f"{func(vals):>9.2f} "
        print(row_str)

    # ---- Closed-period analysis: days until gate reopens + SPX return ----
    print("\n" + "-" * 65)
    print("  CLOSED-PERIOD ANALYSIS — After each GO->NO-GO transition")
    print("  Days until gate reopens (first GO day after transition)")

    closure_lengths = []
    closure_spx_returns = []

    for t_date in transitions_go_to_nogo:
        # Find index of t_date in gate series
        t_loc = gate_arr.index.get_loc(t_date)
        # Scan forward for next GO day
        reopen_loc = None
        for i in range(t_loc + 1, len(gate_arr)):
            if gate_arr.iloc[i] == 1:
                reopen_loc = i
                break
        if reopen_loc is None:
            continue  # gate never reopened — still closed
        reopen_date = gate_arr.index[reopen_loc]
        days_closed = reopen_loc - t_loc  # trading days
        closure_lengths.append(days_closed)

        # SPX return over closed period: from t_date close to reopen_date close
        t_spx_loc = spx.index.get_indexer([t_date], method="nearest")[0]
        r_spx_loc = spx.index.get_indexer([reopen_date], method="nearest")[0]
        if t_spx_loc >= 0 and r_spx_loc >= 0 and t_spx_loc < len(spx) and r_spx_loc < len(spx):
            ret = np.log(spx.iloc[r_spx_loc] / spx.iloc[t_spx_loc]) * 100
            closure_spx_returns.append(ret)

    n_closed = len(closure_lengths)
    n_never_reopen = len(transitions_go_to_nogo) - n_closed - 0
    # count events where gate still hasn't reopened
    n_still_closed = len(transitions_go_to_nogo) - n_closed

    print(f"  Events with reopen found: n={n_closed}  (still closed at end of series: n={n_still_closed})")
    if n_closed >= 5:
        print(f"  Days closed (trading days):")
        print(f"    median = {np.median(closure_lengths):.0f}d")
        print(f"    mean   = {np.mean(closure_lengths):.1f}d")
        print(f"    p25    = {np.percentile(closure_lengths, 25):.0f}d")
        print(f"    p75    = {np.percentile(closure_lengths, 75):.0f}d")
        print(f"    max    = {max(closure_lengths)}d")
    else:
        print("  n<5 — insufficient events to compute closure duration stats")

    if len(closure_spx_returns) >= 5:
        print(f"  SPX log return over closed period (transition->reopen, n={len(closure_spx_returns)}):")
        print(f"    mean   = {np.mean(closure_spx_returns):.2f}%")
        print(f"    median = {np.median(closure_spx_returns):.2f}%")
        print(f"    pct_positive = {np.mean(np.array(closure_spx_returns) > 0)*100:.1f}%")
    else:
        print("  n<5 — insufficient events to compute closed-period SPX return")

    # ---- Condition breakdown at transition events ----
    print("\n" + "-" * 65)
    print("  CONDITION BREAKDOWN AT GO->NO-GO TRANSITION EVENTS")
    print(f"  (which conditions were failing on the transition day, n={len(transitions_go_to_nogo)})")
    for cname, cseries in [("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)]:
        fails = sum(1 for d in transitions_go_to_nogo if d in cseries.index and not cseries[d])
        print(f"    {cname} failing: {fails}/{len(transitions_go_to_nogo)}")

    # ---- C4-specific: condition breakdown at C4 transition events ----
    if len(c4_transitions) >= 1:
        print(f"\n  CONDITION BREAKDOWN AT C4-TRANSITION EVENTS (n={len(c4_transitions)})")
        print("  (shows which other conditions also failed alongside C4)")
        for cname, cseries in [("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)]:
            fails = sum(1 for d in c4_transitions if d in cseries.index and not cseries[d])
            print(f"    {cname} failing: {fails}/{len(c4_transitions)}")

    print("\n" + "=" * 65)
    print("  END OF BACKTEST")
    print("=" * 65)


if __name__ == "__main__":
    main()
