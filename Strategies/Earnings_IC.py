import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import json
import os

ALPHAVANTAGE_KEY = ""
CACHE_FILE = "earnings_cache.json"

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

def load_cache():
    """Load cached earnings data"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save earnings data to cache"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)

# ============================================================================
# DATA FETCHING - ALPHA VANTAGE FOR EARNINGS
# ============================================================================

def get_earnings_details(ticker, use_cache=True):
    """Get exact earnings announcement dates from Alpha Vantage"""
    
    cache = load_cache()
    
    # Check cache first
    if use_cache and ticker in cache:
        print(f"✓ Using cached earnings for {ticker}")
        return [
            {'date': datetime.fromisoformat(e['date']), 'time': e['time']} 
            for e in cache[ticker]
        ]
    
    # Fetch from API
    print(f"⏳ Fetching earnings from Alpha Vantage for {ticker}...")
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={ALPHAVANTAGE_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'Note' in data or 'Information' in data:
            error_msg = data.get('Note', data.get('Information', ''))
            print(f"⚠️  API limit: {error_msg[:80]}...")
            return []
        
        if 'quarterlyEarnings' not in data:
            print(f"⚠️  No earnings data for {ticker}")
            return []
        
        earnings_info = []
        for quarter in data['quarterlyEarnings']:
            reported_date = quarter.get('reportedDate')
            reported_time = quarter.get('reportedTime', 'amc')
            
            if reported_date:
                earnings_info.append({
                    'date': datetime.strptime(reported_date, '%Y-%m-%d'),
                    'time': reported_time.lower()
                })
        
        # Save to cache
        cache[ticker] = [
            {'date': e['date'].isoformat(), 'time': e['time']} 
            for e in earnings_info
        ]
        save_cache(cache)
        print(f"✓ Cached {len(earnings_info)} earnings dates for {ticker}")
        
        return sorted(earnings_info, key=lambda x: x['date'], reverse=True)
    
    except Exception as e:
        print(f"❌ Error fetching earnings: {e}")
        return []

# ============================================================================
# DATA FETCHING - YAHOO FINANCE FOR PRICES
# ============================================================================

def get_yahoo_price_data(ticker, start_date, end_date):
    """Get historical closing prices from Yahoo Finance"""
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {'period1': start_ts, 'period2': end_ts, 'interval': '1d'}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        closes = result['indicators']['quote'][0]['close']
        
        df = pd.DataFrame({
            'date': [datetime.fromtimestamp(ts) for ts in timestamps],
            'close': closes
        })
        df.set_index('date', inplace=True)
        df.dropna(inplace=True)
        
        print(f"✓ Fetched {len(df)} days of price data from Yahoo Finance")
        return df
    
    except Exception as e:
        print(f"❌ Error fetching prices: {e}")
        return pd.DataFrame()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_nearest_price(price_data, target_date):
    """Find closing price on nearest trading day"""
    if price_data.empty:
        return None, None
    
    start = target_date - timedelta(days=7)
    end = target_date + timedelta(days=7)
    nearby = price_data[(price_data.index >= start) & (price_data.index <= end)]
    
    if nearby.empty:
        return None, None
    
    time_diffs = (nearby.index - target_date).to_series().abs()
    closest_idx = time_diffs.argmin()
    return nearby.iloc[closest_idx]['close'], nearby.index[closest_idx]

def get_reference_price(price_data, earnings_date, timing):
    """Get entry price based on earnings timing (BMO vs AMC)"""
    # BMO: Enter at close day before | AMC: Enter at close of earnings day
    target_date = earnings_date - timedelta(days=1) if timing == 'bmo' else earnings_date
    return find_nearest_price(price_data, target_date)

def calculate_historical_volatility(price_data, earnings_date, lookback_days=30):
    """Calculate 30-day historical volatility before earnings (annualized)"""
    end_date = earnings_date - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days + 10)
    
    window = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
    
    if len(window) < 20:
        return None
    
    returns = window['close'].pct_change().dropna()
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    
    return annual_vol

def get_volatility_tier(hvol):
    """Map historical volatility to standard deviation multiplier"""
    hvol_pct = hvol * 100
    
    if hvol_pct < 25:
        return 1.0
    elif hvol_pct < 35:
        return 1.2
    elif hvol_pct < 45:
        return 1.4
    else:
        return 1.5

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_earnings_movement(ticker, lookback_quarters=24, verbose=True):
    """
    Analyze post-earnings movements with volatility-adjusted strikes
    
    Args:
        ticker: Stock symbol
        lookback_quarters: Historical quarters to analyze (default 24 = 6 years)
        verbose: Print detailed output
    """
    
    if verbose:
        print(f"\n{'='*75}")
        print(f"📊 {ticker} - Post-Earnings Containment Analysis")
        print(f"{'='*75}")
    
    # Get earnings dates from Alpha Vantage
    earnings_info = get_earnings_details(ticker)
    if not earnings_info:
        return None
    
    today = datetime.now()
    past_earnings = [e for e in earnings_info if e['date'] < today][:lookback_quarters]
    
    if len(past_earnings) < 10:
        if verbose:
            print(f"⚠️  Insufficient data: only {len(past_earnings)} earnings periods")
        return None
    
    # Get price data from Yahoo Finance
    oldest = min([e['date'] for e in past_earnings]) - timedelta(days=120)
    price_data = get_yahoo_price_data(ticker, oldest, today)
    
    if price_data.empty:
        return None
    
    # Collect movement data for both timeframes
    data_45 = []
    data_90 = []
    hvol_list = []
    
    if verbose:
        print(f"\nAnalyzing {len(past_earnings)} earnings periods...")
    
    for earnings in past_earnings:
        # Calculate historical volatility
        hvol = calculate_historical_volatility(price_data, earnings['date'])
        if hvol is None:
            continue
        
        hvol_list.append(hvol * 100)
        strike_std = get_volatility_tier(hvol)
        
        # Get entry price (day before for BMO, same day for AMC)
        ref_price, ref_date = get_reference_price(price_data, earnings['date'], earnings['time'])
        if ref_price is None:
            continue
        
        # Calculate strike width based on volatility tier
        dte_45_factor = np.sqrt(45 / 365)
        dte_90_factor = np.sqrt(90 / 365)
        strike_width_45 = hvol * dte_45_factor * strike_std * 100
        strike_width_90 = hvol * dte_90_factor * strike_std * 100
        
        # Test 45-day outcome
        target_45 = earnings['date'] + timedelta(days=45)
        if target_45 <= today:
            price_45, date_45 = find_nearest_price(price_data, target_45)
            if price_45 is not None:
                move_45 = (price_45 - ref_price) / ref_price * 100
                data_45.append({
                    'move': move_45,
                    'width': strike_width_45,
                    'hvol': hvol * 100,
                    'date': earnings['date'].strftime('%Y-%m-%d')
                })
        
        # Test 90-day outcome
        target_90 = earnings['date'] + timedelta(days=90)
        if target_90 <= today:
            price_90, date_90 = find_nearest_price(price_data, target_90)
            if price_90 is not None:
                move_90 = (price_90 - ref_price) / ref_price * 100
                data_90.append({
                    'move': move_90,
                    'width': strike_width_90,
                    'hvol': hvol * 100,
                    'date': earnings['date'].strftime('%Y-%m-%d')
                })
    
    if len(data_45) < 10 or len(data_90) < 10:
        if verbose:
            print(f"⚠️  Insufficient valid data")
        return None
    
    # Calculate statistics for both timeframes
    def calc_stats(data):
        total = len(data)
        moves = np.array([d['move'] for d in data])
        widths = np.array([d['width'] for d in data])
        
        # Containment
        stays_within = sum(1 for i, m in enumerate(moves) if abs(m) <= widths[i])
        breaks_up = sum(1 for i, m in enumerate(moves) if m > widths[i])
        breaks_down = sum(1 for i, m in enumerate(moves) if m < -widths[i])
        
        # Directional bias
        up_moves = sum(1 for m in moves if m > 0)
        up_bias = (up_moves / total) * 100
        
        return {
            'total': total,
            'containment': (stays_within / total) * 100,
            'breaks_up': breaks_up,
            'breaks_down': breaks_down,
            'up_bias': up_bias,
            'avg_width': np.mean(widths)
        }
    
    stats_45 = calc_stats(data_45)
    stats_90 = calc_stats(data_90)
    avg_hvol = np.mean(hvol_list)
    avg_tier = get_volatility_tier(avg_hvol / 100)
    
    # Determine recommendation based on data
    rec_parts = []
    
    # Check 90-day containment first (preferred timeframe)
    if stats_90['containment'] >= 70:
        rec_parts.append("IC (90 DTE)")
    elif stats_45['containment'] >= 70:
        rec_parts.append("IC (45 DTE)")
    
    # Check directional edge
    if stats_90['up_bias'] >= 70 and stats_90['breaks_down'] <= stats_90['total'] * 0.15:
        rec_parts.append("Bull Put Spread")
    elif stats_90['up_bias'] <= 30 and stats_90['breaks_up'] <= stats_90['total'] * 0.15:
        rec_parts.append("Bear Call Spread")
    
    recommendation = " + ".join(rec_parts) if rec_parts else "SKIP - No Edge"
    
    # Print results
    if verbose:
        print(f"\n📊 {ticker} | {avg_hvol:.1f}% HVol | {avg_tier:.1f} std (±{stats_90['avg_width']:.1f}%)")
        print(f"\n  45-Day: {stats_45['total']}/{lookback_quarters} tested")
        print(f"    Containment: {stats_45['containment']:.0f}%")
        print(f"    Breaks: Up {stats_45['breaks_up']}, Down {stats_45['breaks_down']}")
        print(f"    Bias: {stats_45['up_bias']:.0f}% up")
        
        print(f"\n  90-Day: {stats_90['total']}/{lookback_quarters} tested")
        print(f"    Containment: {stats_90['containment']:.0f}%")
        print(f"    Breaks: Up {stats_90['breaks_up']}, Down {stats_90['breaks_down']}")
        print(f"    Bias: {stats_90['up_bias']:.0f}% up")
        
        print(f"\n  💡 Strategy: {recommendation}")
    
    summary = {
        'ticker': ticker,
        'hvol': round(avg_hvol, 1),
        'tier': round(avg_tier, 1),
        'strike_width': round(stats_90['avg_width'], 1),
        '45d_contain': round(stats_45['containment'], 0),
        '45d_breaks_up': stats_45['breaks_up'],
        '45d_breaks_dn': stats_45['breaks_down'],
        '45d_bias': round(stats_45['up_bias'], 0),
        '90d_contain': round(stats_90['containment'], 0),
        '90d_breaks_up': stats_90['breaks_up'],
        '90d_breaks_dn': stats_90['breaks_down'],
        '90d_bias': round(stats_90['up_bias'], 0),
        'strategy': recommendation
    }
    
    return summary

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_analyze(tickers, lookback_quarters=24):
    """Analyze multiple tickers"""
    
    print("\n" + "="*75)
    print(f"EARNINGS CONTAINMENT ANALYZER")
    print(f"Lookback: {lookback_quarters} quarters (~{lookback_quarters/4:.0f} years)")
    print(f"Volatility Tiers: <25%=1.0std | 25-35%=1.2std | 35-45%=1.4std | >45%=1.5std")
    print("="*75)
    
    results = []
    
    for ticker in tickers:
        summary = analyze_earnings_movement(ticker, lookback_quarters, verbose=True)
        
        if summary:
            results.append(summary)
        
        time.sleep(1)  # Be nice to APIs
    
    if not results:
        print("\n⚠️  No valid results")
        return None
    
    df = pd.DataFrame(results)
    
    # Summary table - split into sections for readability
    print("\n" + "="*75)
    print("📋 SUMMARY TABLE")
    print("="*75)
    
    print("\nVOLATILITY PROFILE:")
    print(df[['ticker', 'hvol', 'tier', 'strike_width']].to_string(index=False))
    
    print("\n45-DAY RESULTS:")
    print(df[['ticker', '45d_contain', '45d_breaks_up', '45d_breaks_dn', '45d_bias']].to_string(index=False))
    
    print("\n90-DAY RESULTS:")
    print(df[['ticker', '90d_contain', '90d_breaks_up', '90d_breaks_dn', '90d_bias']].to_string(index=False))
    
    print("\nRECOMMENDATIONS:")
    print(df[['ticker', 'strategy']].to_string(index=False))
    
    # Compact overview table
    print("\n" + "="*75)
    print("📊 COMPACT OVERVIEW (90-Day Focus)")
    print("="*75)
    compact = df[['ticker', 'hvol', 'tier', '90d_contain', '90d_bias', '90d_breaks_up', '90d_breaks_dn']].copy()
    compact.columns = ['Ticker', 'HVol%', 'Tier', '90D%', 'Bias%', 'Up', 'Down']
    print(compact.to_string(index=False))
    
    return df

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Test with 2 tickers
    tickers = ["AAPL", "MSFT"]
    
    results = batch_analyze(tickers, lookback_quarters=24)