#!/usr/bin/env python3
"""
Byzantium Capital — Regime Dashboard
Single command morning glance: is today a good environment to sell premium?

Usage:
    cd /Users/johnluo/Desktop/GitHub/SPX_Analysis/src
    python3 regime_dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd

SRC = Path(__file__).parent
DC  = SRC / "data_cache"
VC  = DC  / "vx_continuous"


# ── loaders ──────────────────────────────────────────────────────────────────

def _close(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    col = "Close" if "Close" in df.columns else "settle"
    s = df[col].dropna()
    s.index = pd.to_datetime(s.index)
    return s


def load_vx_spread() -> pd.Series:
    csv = pd.read_csv(
        SRC / "CBOE_Data_Archive" / "VX1-VX2.csv",
        parse_dates=["Date"], index_col="Date",
    )["Close"]
    vx1 = _close(VC / "VX1.parquet")
    vx2 = _close(VC / "VX2.parquet")
    pq  = (vx1 - vx2).rename("spread")
    pq.index = pd.to_datetime(pq.index)
    ext = pq[pq.index > csv.index[-1]]
    combined = pd.concat([csv, ext]).sort_index()
    return combined[~combined.index.duplicated(keep="last")].dropna()


# ── regime helpers ────────────────────────────────────────────────────────────

def current_streak(spread: pd.Series) -> tuple[str, int, pd.Timestamp]:
    regime = "contango" if spread.iloc[-1] < 0 else "backwardation"
    in_regime = (spread < 0) if regime == "contango" else (spread >= 0)
    # walk backwards
    idx = len(spread) - 1
    while idx >= 0 and in_regime.iloc[idx]:
        idx -= 1
    streak = len(spread) - 1 - idx
    since  = spread.index[idx + 1]
    return regime, streak, since


def pct_rank(series: pd.Series, value: float, window: int = None) -> float:
    s = series.iloc[-window:] if window else series
    return float((s < value).mean() * 100)


def rolling_pct_rank(series: pd.Series, value: float, years: int) -> float:
    window = years * 252
    return pct_rank(series, value, window)


def last_backwardation_days_ago(spread: pd.Series) -> int:
    bk_dates = spread.index[spread >= 0]
    if bk_dates.empty:
        return 9999
    last_bk = bk_dates[-1]
    all_dates = spread.index
    return int((all_dates[-1] - last_bk).days)


# ── score engine ──────────────────────────────────────────────────────────────

def compute_score(
    spread_val, streak, spx_close, spx_20, spx_50, spx_200,
    vix, vix_1yr_pct, vix3m, vvix, days_since_bk,
) -> tuple[int, list[tuple]]:
    """Returns (total_score, [(points, max, label), ...])"""
    breakdown = []

    # 1. VX term structure (0–30)
    if spread_val < 0:
        base = 20
        if spread_val < -2.0:
            depth_bonus = 10
        elif spread_val < -1.0:
            depth_bonus = 5
        else:
            depth_bonus = 0
        if streak >= 30:
            streak_bonus = 0   # long streaks are fine but don't add more pts
        elif streak < 5:
            streak_bonus = -5  # fresh return, caution
        else:
            streak_bonus = 0
        pts = base + depth_bonus + streak_bonus
        label = f"VX1−VX2 contango ({spread_val:+.2f}, streak {streak}td)"
    else:
        pts = -10
        label = f"VX1−VX2 in BACKWARDATION ({spread_val:+.2f}) — avoid new spreads"
    breakdown.append((pts, 30, label))

    # 2. VIX level vs 1yr percentile (0–25)
    if 15 <= vix <= 25:
        base = 25
        extra = ""
    elif 12 <= vix < 15:
        base = 10
        extra = " (cheap premium)"
    elif 25 < vix <= 35:
        base = 15
        extra = " (higher but manageable)"
    elif vix > 35:
        base = 0
        extra = " (danger zone)"
    else:
        base = 5
        extra = " (very low premium)"
    pts = base
    label = f"VIX {vix:.1f} at {vix_1yr_pct:.0f}th pct (1yr){extra}"
    breakdown.append((pts, 25, label))

    # 3. VVIX — vol of vol (0–15)
    if vvix < 85:
        pts, note = 15, "calm"
    elif vvix < 100:
        pts, note = 10, "moderate"
    elif vvix < 120:
        pts, note = 5,  "elevated"
    else:
        pts, note = 0,  "EXTREME — vol is unstable"
    breakdown.append((pts, 15, f"VVIX {vvix:.1f} ({note})"))

    # 4. SPX trend (0–20)
    pts = 0
    flags = []
    if spx_close > spx_200:
        pts += 10; flags.append(">200MA")
    if spx_close > spx_50:
        pts += 5;  flags.append(">50MA")
    if spx_close > spx_20:
        pts += 5;  flags.append(">20MA")
    label = f"SPX {spx_close:,.0f}  {'  '.join(flags) if flags else 'BELOW all MAs'}"
    breakdown.append((pts, 20, label))

    # 5. VIX/VIX3M term structure (0–10)
    ratio = vix / vix3m if vix3m else 1.0
    if ratio < 0.90:
        pts = 10; note = "steep (very normal)"
    elif ratio < 1.00:
        pts = 7;  note = "mild"
    elif ratio < 1.05:
        pts = 3;  note = "flat (caution)"
    else:
        pts = 0;  note = "INVERTED"
    breakdown.append((pts, 10, f"VIX/VIX3M {ratio:.3f} ({note})"))

    total = sum(p for p, _, _ in breakdown)
    total = max(0, min(100, total))
    return total, breakdown


# ── display ───────────────────────────────────────────────────────────────────

def bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "●" * filled + "○" * (width - filled)


def rating(score: int) -> str:
    if score >= 80: return "EXCELLENT — strong premium environment"
    if score >= 65: return "FAVORABLE"
    if score >= 50: return "NEUTRAL — proceed selectively"
    if score >= 35: return "CAUTIOUS — raise selectivity, tighten strikes"
    return "AVOID NEW POSITIONS"


def color(score: int) -> str:
    if score >= 65: return "\033[92m"   # green
    if score >= 50: return "\033[93m"   # yellow
    return "\033[91m"                   # red

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
WHITE = "\033[97m"


def display(
    spread, streak, since, regime,
    vix, vix_1yr_pct, vix_5yr_pct, vix3m, vvix, skew,
    spx_close, spx_20, spx_50, spx_200,
    data_date, score, breakdown,
):
    today     = date.today().strftime("%Y-%m-%d")
    data_lag  = (datetime.today() - data_date).days
    lag_warn  = f"  {DIM}⚠ {data_lag}d stale — refresh cache for live data{RESET}" if data_lag > 2 else ""
    vx_ratio  = (vx1_val / vx2_val) if 'vx1_val' in dir() else None  # computed below in main

    W = 66
    print(f"\n{BOLD}{CYAN}{'═'*W}{RESET}")
    print(f"{BOLD}{CYAN}  REGIME DASHBOARD{RESET}   {today}   {DIM}data through {data_date.strftime('%Y-%m-%d')}{RESET}{lag_warn}")
    print(f"{BOLD}{CYAN}{'═'*W}{RESET}")

    # ── VX Term Structure ──
    regime_tag = (f"{BOLD}\033[92m● CONTANGO{RESET}" if regime == "contango"
                  else f"{BOLD}\033[91m● BACKWARDATION{RESET}")
    print(f"\n  {BOLD}VX TERM STRUCTURE{RESET}  {' '*22}{regime_tag}")
    print(f"  {'VX1 − VX2':<20} {spread:>+7.3f}   │  pct rank (all-time): {pct_rank_spread:.0f}th")
    print(f"  {'Streak':<20} {streak:>6} td   │  since {since.strftime('%Y-%m-%d')}")
    print(f"  {'VIX / VIX3M':<20} {vix/vix3m:>7.3f}   │  VIX3M: {vix3m:.1f}")

    # ── VIX Block ──
    vix_tag = (f"{BOLD}\033[92m● BENIGN{RESET}" if vix < 20 else
               f"{BOLD}\033[93m● ELEVATED{RESET}" if vix < 30 else
               f"{BOLD}\033[91m● HIGH{RESET}")
    print(f"\n  {BOLD}VIX{RESET}  {' '*35}{vix_tag}")
    print(f"  {'Level':<20} {vix:>7.2f}   │  1yr pct: {vix_1yr_pct:.0f}th   5yr pct: {vix_5yr_pct:.0f}th")
    print(f"  {'VVIX (vol-of-vol)':<20} {vvix:>7.2f}   │  {'elevated ⚠' if vvix > 100 else 'calm'}")
    print(f"  {'SKEW':<20} {skew:>7.2f}   │  {'elevated tail risk' if skew > 145 else 'normal'}")

    # ── SPX Trend ──
    pct_vs_20  = (spx_close / spx_20  - 1) * 100
    pct_vs_50  = (spx_close / spx_50  - 1) * 100
    pct_vs_200 = (spx_close / spx_200 - 1) * 100
    spx_tag    = (f"{BOLD}\033[92m● UPTREND{RESET}" if spx_close > spx_200
                  else f"{BOLD}\033[91m● DOWNTREND{RESET}")
    print(f"\n  {BOLD}SPX{RESET}  {' '*35}{spx_tag}")
    print(f"  {'Level':<20} {spx_close:>7,.1f}   │  vs 200MA: {pct_vs_200:+.1f}%")
    print(f"  {'vs 20MA':<20} {pct_vs_20:>+6.1f}%   │  vs 50MA:  {pct_vs_50:+.1f}%")

    # ── Composite Score ──
    clr = color(score)
    print(f"\n  {'─'*W}")
    print(f"  {BOLD}SELL-PREMIUM SCORE{RESET}   "
          f"{clr}{BOLD}{score:>3} / 100{RESET}   "
          f"{clr}{bar(score)}{RESET}")
    print(f"  {clr}{BOLD}{rating(score)}{RESET}")
    print(f"  {'─'*W}")
    for pts, mx, label in breakdown:
        sign  = "+" if pts >= 0 else ""
        clr2  = "\033[92m" if pts >= mx * 0.7 else ("\033[93m" if pts > 0 else "\033[91m")
        print(f"  {clr2}[{sign}{pts:>3}/{mx}]{RESET}  {label}")
    print(f"  {'─'*W}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    spread_s = load_vx_spread()
    vx1_s    = _close(VC / "VX1.parquet")
    vx2_s    = _close(VC / "VX2.parquet")
    vix_s    = _close(DC / "yahoo_VIX.parquet")
    vix3m_s  = _close(DC / "yahoo_VIX3M.parquet")
    vvix_s   = _close(DC / "yahoo_VVIX.parquet")
    skew_s   = _close(DC / "yahoo_SKEW.parquet")
    spx_s    = _close(DC / "yahoo_GSPC.parquet")

    spread_val = float(spread_s.iloc[-1])
    regime, streak, since = current_streak(spread_s)

    global pct_rank_spread
    pct_rank_spread = pct_rank(spread_s, spread_val)

    # align all to latest common date
    common_idx = spx_s.index  # SPX as reference
    vix_s   = vix_s.reindex(common_idx, method="ffill")
    vix3m_s = vix3m_s.reindex(common_idx, method="ffill")
    vvix_s  = vvix_s.reindex(common_idx, method="ffill")
    skew_s  = skew_s.reindex(common_idx, method="ffill")

    vix   = float(vix_s.iloc[-1])
    vix3m = float(vix3m_s.iloc[-1])
    vvix  = float(vvix_s.iloc[-1])
    skew  = float(skew_s.iloc[-1])

    spx_close = float(spx_s.iloc[-1])
    spx_20    = float(spx_s.iloc[-20:].mean()) if len(spx_s) >= 20 else spx_close
    spx_50    = float(spx_s.iloc[-50:].mean()) if len(spx_s) >= 50 else spx_close
    spx_200   = float(spx_s.iloc[-200:].mean()) if len(spx_s) >= 200 else spx_close

    vix_1yr_pct = rolling_pct_rank(vix_s.dropna(), vix, 1)
    vix_5yr_pct = rolling_pct_rank(vix_s.dropna(), vix, 5)

    data_date = spx_s.index[-1].to_pydatetime()

    score, breakdown = compute_score(
        spread_val, streak,
        spx_close, spx_20, spx_50, spx_200,
        vix, vix_1yr_pct, vix3m, vvix,
        days_since_bk=last_backwardation_days_ago(spread_s),
    )

    display(
        spread_val, streak, since, regime,
        vix, vix_1yr_pct, vix_5yr_pct, vix3m, vvix, skew,
        spx_close, spx_20, spx_50, spx_200,
        data_date, score, breakdown,
    )


if __name__ == "__main__":
    main()
