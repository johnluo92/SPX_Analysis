# check_missing_trading_days_fixed.py
import pandas as pd
import pandas_market_calendars as mcal

FILE_PATH = "GOLDSILVER_RATIO.csv"  # adjust as needed

def load_dates_from_csv(path, date_col="Date"):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError("Some dates could not be parsed. Check your CSV.")
    # convert to plain python date (tz-naive)
    return pd.Index(df[date_col].dt.date.unique())

def expected_nyse_trading_dates(start_date, end_date, extra_closures=None):
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start_date, end_date=end_date)
    # schedule.index are trading days with tz info (usually America/New_York).
    # Convert to plain date objects for safe comparison.
    trading_dates = pd.Index(sched.index.date)
    if extra_closures:
        # make sure closures are date objects
        closures = pd.Index(pd.to_datetime(extra_closures).date)
        trading_dates = trading_dates.difference(closures)
    return trading_dates

def main():
    actual_days = load_dates_from_csv(FILE_PATH)
    start = min(actual_days)
    end = max(actual_days)

    # add any special closures you want excluded from expected trading days:
    special_closures = ["2025-01-09"]  # Jimmy Carter memorial (National Day of Mourning)

    expected_days = expected_nyse_trading_dates(start, end, extra_closures=special_closures)

    missing = expected_days.difference(actual_days)
    print('Checking file: ' + FILE_PATH)
    print(f"Data range: {start} → {end}")
    print(f"Expected NYSE trading days in range: {len(expected_days)}")
    print(f"Actual unique dates in CSV: {len(actual_days)}")
    if len(missing) == 0:
        print("✅ No missing trading days.")
    else:
        print(f"⚠️ Missing {len(missing)} trading day(s):")
        for d in missing:
            print(d.isoformat())

if __name__ == "__main__":
    main()
