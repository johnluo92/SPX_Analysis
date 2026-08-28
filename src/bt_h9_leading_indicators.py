"""
BT-32 (H9): does ANY leading macro indicator flag the DP down-years (2018, 2022) BEFORE the loss,
without flagging the 9 good years? (John 2026-06-19: "I do not want to buy puts ... is there any
way to gate more effectively when shit hits the fan ... yield inversions? other leading indicators,
anything that signals 'bad times ahead, size down'.") The current gate (C1-C5) is coincident
(BT-29). This tests whether a LEADING overlay can size the DP book down ahead of the drawdown.

Quick look already showed yield inversion MISSES 2018 entirely (2y10y never inverted in 2018;
first inversion Aug-2019) and is coincident-to-lagging in 2022 (2y10y inverted Apr-2022 after SPX
already fell; 3m10y not until Oct-2022). This test makes that rigorous and adds a basket.

Indicators (warning = "bad times ahead", all live-knowable with stated lag):
  yc_2y10y   2y10y term spread < 0 (classic recession lead)            FRED DGS2/DGS10, lag 1d
  yc_3m10y   3m10y term spread < 0                                     FRED DGS3MO/DGS10, lag 1d
  claims_up  ICSA 4wk-MA rising YoY (labor turning)                    FRED ICSA, lag 1wk (pub)
  copper_gld HG/GC ratio < its 200d MA (Dr Copper weakening)           Yahoo HG_F/GC_F
  dollar_up  DXY > its 200d MA (risk-off dollar bid)                   Yahoo DX-Y
  hy_widen   HY OAS > its 63d MA (credit stress building)              FRED BAMLH0A0HYM2, lag 1d
DP book = BT-21 (build_dp, GO-day 2016+, 50-wide). Two outputs:
  (1) DIAGNOSTIC: warning-rate among GO-days BY YEAR — the real test is "lights up 2018 & 2022,
      dark in good years." A leading indicator that's on in good years too is just a tax.
  (2) OVERLAY: skip DP entries while in warning; worst-yr / maxDD / total vs baseline.

PRE-REGISTERED DECISION RULE (written before running):
  An indicator "helps" iff skipping its warning-state entries (a) cuts worst calendar-year loss
  >=40% AND (b) cuts realized maxDD >=25% AND (c) retains >=80% of DP total P&L. (c) is the
  selectivity bar — a true leading signal should be OFF most of the good years, so skipping it is
  nearly free. If an indicator cuts the tail only by also gutting total (warning-on in good years),
  it is a coincident/too-early tax, not a leading edge -> reject. Report the by-year diagnostic
  regardless; the numbers tell whether 2018/2022 were actually flagged ahead of time.
CAVEATS: 2016+ gate era (only 2 DP loss years in-sample -> any "hit" is n=2, the binding limit;
  a leading rule fit to 2 events is fragile by construction). Lags applied per above (live-honest).
  Skipping = full size-down to 0 (upper bound on the overlay's effect). Realized-by-expiry DD.
"""
import os
import numpy as np
import pandas as pd
from bt_h2_dp_sizing import build_dp
from bt_dp_gate_pnl import DATA


def _col(f):
    df = pd.read_parquet(os.path.join(DATA, f))
    return df[df.columns[0]].sort_index()


def warning_series():
    d2, d10, d3 = _col("fred_DGS2.parquet"), _col("fred_DGS10.parquet"), _col("fred_DGS3MO.parquet")
    icsa = _col("fred_ICSA.parquet")
    hg = pd.read_parquet(os.path.join(DATA, "yahoo_HG_F.parquet"))["Close"].sort_index()
    gc = pd.read_parquet(os.path.join(DATA, "yahoo_GC_F.parquet"))["Close"].sort_index()
    dxy = pd.read_parquet(os.path.join(DATA, "yahoo_DX-Y.NYB.parquet"))["Close"].sort_index()
    hy = _col("fred_BAMLH0A0HYM2.parquet")

    yc2 = (d10 - d2).shift(1)
    yc3 = (d10 - d3).shift(1)
    icsa_ma = icsa.rolling(4).mean()
    claims = (icsa_ma > icsa_ma.shift(52)).shift(1)        # YoY rising, lag 1 wk
    cg = (hg / gc); copper = (cg < cg.rolling(200).mean())
    dollar = (dxy > dxy.rolling(200).mean())
    hyw = (hy > hy.rolling(63).mean()).shift(1)

    return {
        "yc_2y10y":   (yc2 < 0),
        "yc_3m10y":   (yc3 < 0),
        "claims_up":  claims,
        "copper_gld": copper,
        "dollar_up":  dollar,
        "hy_widen":   hyw,
    }


def align(sig, dates):
    s = sig.reindex(sig.index.union(dates)).ffill().reindex(dates)
    return s.fillna(False).astype(bool)


def realized_dd(df, col):
    s = df.sort_values("expiry")
    cum = s[col].cumsum().to_numpy()
    return float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0


def report(markup):
    dp = build_dp(50, markup).sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(dp.date)
    sigs = warning_series()
    dp_total = dp.pnl.sum()
    dp_worst = dp.groupby("year").pnl.sum().min()
    dp_dd = realized_dd(dp, "pnl")
    loss_years = [2018, 2022]

    print(f"\n{'='*100}")
    print(f"  BT-32 / H9 — leading-indicator overlays on DP book, markup {markup:.2f}")
    print(f"{'='*100}")
    print(f"  DP-alone: total ${dp_total:+,.0f}  worst-yr ${dp_worst:+,.0f}  maxDD ${-dp_dd:,.0f}  "
          f"(loss yrs 2018/2022)")
    yrs = sorted(dp.year.unique())
    print(f"\n  DIAGNOSTIC — warning-rate among GO-days by year (want: HIGH in 2018/2022, LOW else):")
    print(f"  {'indicator':>12} " + " ".join(f"{y:>5}" for y in yrs))
    for name, sig in sigs.items():
        w = align(sig, dates)
        dpw = dp.assign(w=w.values)
        rates = dpw.groupby("year").w.mean() * 100
        cells = []
        for y in yrs:
            r = rates.get(y, 0.0)
            mark = "*" if y in loss_years and r >= 50 else ""
            cells.append(f"{r:>4.0f}{mark if mark else ' '}")
        print(f"  {name:>12} " + " ".join(cells))

    print(f"\n  OVERLAY — skip DP entries while in warning:")
    print(f"  {'indicator':>12} {'skip%':>6} {'total P&L':>12} {'%kept':>6} {'worst-yr':>11} "
          f"{'2018':>10} {'2022':>10} {'maxDD':>11} {'helps':>6}")
    print(f"  {'-'*94}")
    for name, sig in sigs.items():
        w = align(sig, dates).values
        kept = dp[~w]
        total = kept.pnl.sum()
        by = kept.groupby("year").pnl.sum()
        worst = by.min()
        dd = realized_dd(kept, "pnl")
        helps = (worst >= dp_worst * 0.60 and dd <= dp_dd * 0.75 and total >= dp_total * 0.80)
        print(f"  {name:>12} {w.mean()*100:>5.0f}% ${total:>+10,.0f} {total/dp_total*100:>5.0f}% "
              f"${worst:>+10,.0f} ${by.get(2018,0):>+9,.0f} ${by.get(2022,0):>+9,.0f} "
              f"${-dd:>+10,.0f} {'YES' if helps else 'no':>6}")
    print(f"\n  helps = worst-yr cut >=40% AND maxDD cut >=25% AND >=80% total retained.")


if __name__ == "__main__":
    for m in (1.00, 1.15):
        report(m)
