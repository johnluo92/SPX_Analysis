import yfinance as yf
from pathlib import Path

outdir = Path("yfinance_output")
outdir.mkdir(exist_ok=True)

symbols = ["^VIX", "^VIX3M", "^VIX6M", "^VIX1Y", "^VIX9D", "^VVIX", "^MOVE", "^SKEW",
           "^RVX", "^GVZ", "^SPX", "^NDX", "^DJI", "^RUT", "^SP500FTR", "^SP500FER",
           "^VWB", "^SPXA50R", "^SPXA200R", "^TNX", "^FVX", "^TYX", "^IRX", 
           "ZN=F", "ZB=F", "ZT=F", "ZF=F", "SR1=F", "SR3=F", "SOFR=F", "ED=F",
           "GE=F", "^COR3M", "ES=F", "NQ=F", "YM=F", "RTY=F"]

periods = ["max", "20y", "10y", "5y", "1y"]
summary = {}

for sym in symbols:
    print(f"\n--- Fetching {sym} ---")
    ticker = yf.Ticker(sym)
    # Try a tiny fetch first to see if symbol exists at all
    try:
        test = ticker.history(period="1d")
        if test.empty:
            print(f"  -> Symbol {sym} appears missing or delisted; skipping.")
            summary[sym] = "no data"
            continue
    except Exception as e:
        print(f"  -> Symbol {sym} appears missing or delisted; skipping ({e})")
        summary[sym] = "no data"
        continue

    data = None
    for period in periods:
        try:
            df = ticker.history(period=period)
            if not df.empty:
                data = df
                print(f"  -> Found data for period={period} ({len(df)} rows)")
                break
        except yf.shared._exceptions.YFInvalidPeriodError as e:
            print(f"  !! Invalid period for {sym}: {e}")
            continue
        except Exception as e:
            print(f"  !! Error for {sym} ({period}): {e}")
            break

    if data is not None:
        fname = outdir / f"{sym.replace('^','_caret_').replace('=','_eq_')}.csv"
        data.to_csv(fname)
        print(f"     ✅ Got {len(data)} rows x {data.shape[1]} cols; saved to {fname.name}")
        summary[sym] = len(data)
    else:
        print(f"  -> No historical data found for {sym}")
        summary[sym] = "no data"

print("\nSUMMARY")
for k, v in summary.items():
    print(f" * {k}: {v} rows" if isinstance(v,int) else f" * {k}: no data")
