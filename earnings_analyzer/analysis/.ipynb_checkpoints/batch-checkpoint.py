"""Batch processing with parallel execution"""
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from ..config import DEFAULT_LOOKBACK_QUARTERS, REQUEST_DELAY, ALPHAVANTAGE_KEYS
from ..cache import load_cache, load_rate_limits, is_market_hours
from ..data_sources import YahooFinanceClient
from .single import analyze_ticker


def batch_analyze(tickers: List[str], lookback_quarters: int = DEFAULT_LOOKBACK_QUARTERS,
                 debug: bool = False, fetch_iv: bool = True, parallel: bool = False,
                 max_workers: int = 4) -> Optional[pd.DataFrame]:
    """
    Analyze multiple tickers with optional parallel processing
    
    Args:
        tickers: List of ticker symbols
        lookback_quarters: Number of quarters to analyze
        debug: Print debug information
        fetch_iv: Fetch current IV data
        parallel: Use parallel processing
        max_workers: Number of parallel workers
    
    Returns:
        DataFrame with analysis results
    """
    start_time = time.time()
    
    print("\n" + "="*75)
    print(f"EARNINGS CONTAINMENT ANALYZER - v2.4")
    print(f"Lookback: {lookback_quarters} quarters (~{lookback_quarters/4:.0f} years)")
    
    if fetch_iv:
        market_status = is_market_hours()
        
        if market_status['is_open']:
            print(f"Current IV from Yahoo Finance (15-20min delayed)")
        else:
            print(f"⚠️  IV from cache (market closed)")
            print(f"   Last updated: {market_status['time_str']} on {market_status['day_name']}")
    
    if parallel:
        print(f"Parallel processing: {max_workers} workers")
    print("="*75)
    
    rate_limited_keys = load_rate_limits()
    if rate_limited_keys:
        available = len(ALPHAVANTAGE_KEYS) - len(rate_limited_keys)
        print(f"\n⚠️  Rate Limit: {available}/{len(ALPHAVANTAGE_KEYS)} API keys available")
    else:
        print(f"\n✓ All {len(ALPHAVANTAGE_KEYS)} API keys available")
    
    results = []
    fetch_summary = {'cached': [], 'api': [], 'failed': []}
    iv_summary = {'success': [], 'failed': [], 'cached_count': 0, 'fresh_count': 0}
    
    if parallel:
        results = _batch_analyze_parallel(
            tickers, lookback_quarters, debug, fetch_iv, 
            max_workers, fetch_summary, iv_summary
        )
    else:
        results = _batch_analyze_serial(
            tickers, lookback_quarters, debug, fetch_iv,
            fetch_summary, iv_summary
        )
    
    print("\r" + " " * 80 + "\r", end='')
    
    elapsed = time.time() - start_time
    _print_fetch_summary(fetch_summary, iv_summary, fetch_iv, elapsed, len(tickers))
    
    if not results:
        print("\n⚠️  No valid results")
        return None
    
    df = _create_results_dataframe(results)
    _print_results_table(df)
    _print_insights(df)
    
    return df


def _batch_analyze_serial(tickers, lookback_quarters, debug, fetch_iv, 
                          fetch_summary, iv_summary):
    """Process tickers serially"""
    results = []
    cache = load_cache()
    yf_client = YahooFinanceClient()
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\r[{i}/{len(tickers)}] Processing {ticker}...", end='', flush=True)
        
        from_cache = ticker in cache
        summary, status = analyze_ticker(ticker, lookback_quarters, verbose=False, debug=debug)
        
        if summary:
            if fetch_iv:
                _fetch_and_add_iv(ticker, summary, yf_client, iv_summary)
            
            results.append(summary)
            if from_cache:
                fetch_summary['cached'].append(ticker)
            else:
                fetch_summary['api'].append(ticker)
        else:
            fetch_summary['failed'].append(ticker)
        
        time.sleep(REQUEST_DELAY)
    
    return results


def _batch_analyze_parallel(tickers, lookback_quarters, debug, fetch_iv,
                           max_workers, fetch_summary, iv_summary):
    """Process tickers in parallel while preserving order"""
    results_dict = {}
    cache = load_cache()
    yf_client = YahooFinanceClient()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit with index to maintain order
        future_to_ticker = {
            executor.submit(analyze_ticker, ticker, lookback_quarters, False, debug): (i, ticker)
            for i, ticker in enumerate(tickers)
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            idx, ticker = future_to_ticker[future]
            completed += 1
            print(f"\r[{completed}/{len(tickers)}] Processing {ticker}...", end='', flush=True)
            
            try:
                summary, status = future.result()
                from_cache = ticker in cache
                
                if summary:
                    if fetch_iv:
                        _fetch_and_add_iv(ticker, summary, yf_client, iv_summary)
                    
                    # Store with original index
                    results_dict[idx] = summary
                    
                    if from_cache:
                        fetch_summary['cached'].append(ticker)
                    else:
                        fetch_summary['api'].append(ticker)
                else:
                    fetch_summary['failed'].append(ticker)
            except Exception as e:
                print(f"\n❌ Error processing {ticker}: {e}")
                fetch_summary['failed'].append(ticker)
    
    # Sort by original index and return as list
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    
    return results


def _fetch_and_add_iv(ticker, summary, yf_client, iv_summary):
    """Fetch and add IV data to summary"""
    iv_data = yf_client.get_current_iv(ticker)
    if iv_data:
        summary['current_iv'] = iv_data['iv']
        summary['iv_dte'] = iv_data['dte']
        iv_premium = ((iv_data['iv'] - summary['hvol']) / summary['hvol']) * 100
        summary['iv_premium'] = round(iv_premium, 1)
        iv_summary['success'].append(ticker)
        
        # Track if cached or fresh
        if iv_data.get('cached', False):
            iv_summary['cached_count'] += 1
        else:
            iv_summary['fresh_count'] += 1
    else:
        summary['current_iv'] = None
        summary['iv_dte'] = None
        summary['iv_premium'] = None
        iv_summary['failed'].append(ticker)


def _print_fetch_summary(fetch_summary, iv_summary, fetch_iv, elapsed, total_tickers):
    """Print fetch summary with timing"""
    print(f"\n📊 FETCH SUMMARY ({elapsed:.1f}s)")
    print(f"{'='*75}")
    if fetch_summary['cached']:
        cached_list = ', '.join(fetch_summary['cached'][:5])
        if len(fetch_summary['cached']) > 5:
            cached_list += '...'
        print(f"✓ Earnings Cached ({len(fetch_summary['cached'])}): {cached_list}")
    if fetch_summary['api']:
        print(f"✓ Earnings API ({len(fetch_summary['api'])}): {', '.join(fetch_summary['api'])}")
    if fetch_iv:
        total_iv_success = len(iv_summary['success'])
        if total_iv_success > 0:
            cache_info = ""
            if iv_summary['cached_count'] > 0 and iv_summary['fresh_count'] > 0:
                cache_info = f" ({iv_summary['cached_count']} cached, {iv_summary['fresh_count']} fresh)"
            elif iv_summary['cached_count'] > 0:
                cache_info = f" (all cached)"
            elif iv_summary['fresh_count'] > 0:
                cache_info = f" (all fresh)"
            
            iv_list = ', '.join(iv_summary['success'][:5])
            if len(iv_summary['success']) > 5:
                iv_list += '...'
            print(f"✓ IV Retrieved ({total_iv_success}){cache_info}: {iv_list}")
        
        if iv_summary['failed']:
            print(f"✗ IV Failed ({len(iv_summary['failed'])}): {', '.join(iv_summary['failed'])}")
    if fetch_summary['failed']:
        print(f"✗ Analysis Failed ({len(fetch_summary['failed'])}): {', '.join(fetch_summary['failed'])}")
    
    # Performance metrics
    tickers_per_sec = total_tickers / elapsed if elapsed > 0 else 0
    print(f"\n⚡ Performance: {tickers_per_sec:.1f} tickers/sec")


def _format_break_ratio(up_breaks, down_breaks, break_bias):
    """Format break ratio with directional arrow"""
    if up_breaks == 0 and down_breaks == 0:
        return "0:0"
    
    if break_bias >= 66.7:
        return f"{up_breaks}:{down_breaks}↑"
    elif break_bias <= 33.3:
        return f"{up_breaks}:{down_breaks}↓"
    else:
        return f"{up_breaks}:{down_breaks}"


def _create_results_dataframe(results):
    """Create formatted results dataframe"""
    df = pd.DataFrame(results)
    
    df['45d_width'] = df.apply(lambda x: round(x['strike_width'] * np.sqrt(45/90), 1), axis=1)
    
    df['45_break_fmt'] = df.apply(
        lambda x: _format_break_ratio(x['45d_breaks_up'], x['45d_breaks_dn'], x['45d_break_bias']), 
        axis=1
    )
    df['90_break_fmt'] = df.apply(
        lambda x: _format_break_ratio(x['90d_breaks_up'], x['90d_breaks_dn'], x['90d_break_bias']), 
        axis=1
    )
    
    df['strategy_display'] = df['strategy'].apply(lambda x: 
        x.replace('BIAS↑', 'BIAS↑').replace('BIAS↓', 'BIAS↓') if 'BIAS' in x else x
    )
    
    return df


def _print_results_table(df):
    """Print results table"""
    print(f"\n{'='*110}")
    print("BACKTEST RESULTS")
    print("="*110)
    
    display_cols = {
        'Ticker': df['ticker'],
        'HVol%': df['hvol'].astype(int),
    }
    
    if 'current_iv' in df.columns and df['current_iv'].notna().any():
        display_cols['CurIV%'] = df['current_iv'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        display_cols['IVPrem'] = df['iv_premium'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
        display_cols['|'] = '|'
    
    display_cols.update({
        '90D%': df['90d_contain'].astype(int),
        '90Bias': df['90d_overall_bias'].astype(int),
        '90Break': df['90_break_fmt'],
        '90Drift': df['90d_drift'].apply(lambda x: f"{x:+.1f}%"),
        ' | ': '|',
        'Pattern': df['strategy_display']
    })
    
    display_df = pd.DataFrame(display_cols)
    print(display_df.to_string(index=False))


def _print_insights(df):
    """Print key takeaways and insights"""
    print(f"\n{'='*110}")
    print("KEY TAKEAWAYS:")
    print("="*110)
    
    ic_count = len(df[df['strategy'].str.contains('IC', na=False)])
    bias_up_count = len(df[df['strategy'].str.contains('BIAS↑', na=False)])
    bias_down_count = len(df[df['strategy'].str.contains('BIAS↓', na=False)])
    skip_count = len(df[df['strategy'] == 'SKIP'])
    
    print(f"\n📊 Pattern Summary: {ic_count} IC candidates | {bias_up_count} Upward bias | {bias_down_count} Downward bias | {skip_count} No edge")
    
    if 'iv_premium' in df.columns and df['iv_premium'].notna().any():
        elevated = df[df['iv_premium'] >= 15].sort_values('iv_premium', ascending=False)
        depressed = df[df['iv_premium'] <= -15].sort_values('iv_premium')
        
        print(f"\n💰 IV Landscape:")
        if not elevated.empty:
            tickers_str = ', '.join([f"{row['ticker']}(+{row['iv_premium']:.0f}%)" for _, row in elevated.head(5).iterrows()])
            print(f"  Rich Premium (≥15%): {tickers_str}")
        if not depressed.empty:
            tickers_str = ', '.join([f"{row['ticker']}({row['iv_premium']:.0f}%)" for _, row in depressed.head(3).iterrows()])
            print(f"  Thin Premium (≤-15%): {tickers_str}")
        
        normal_count = len(df[(df['iv_premium'] > -15) & (df['iv_premium'] < 15)])
        if normal_count > 0:
            print(f"  Normal Range: {normal_count} tickers")
    
    ic_up_skew = df[(df['strategy'].str.contains('IC.*⚠↑', regex=True, na=False))]
    ic_down_skew = df[(df['strategy'].str.contains('IC.*⚠↓', regex=True, na=False))]
    
    if not ic_up_skew.empty or not ic_down_skew.empty:
        print(f"\n⚠️  Asymmetric ICs:")
        if not ic_up_skew.empty:
            print(f"  Upside risk: {', '.join(ic_up_skew['ticker'].tolist())}")
        if not ic_down_skew.empty:
            print(f"  Downside risk: {', '.join(ic_down_skew['ticker'].tolist())}")
    
    strong_bias = df[
        (df['strategy'].str.contains('BIAS', na=False)) &
        ((df['90d_overall_bias'] >= 70) | (df['90d_overall_bias'] <= 30))
    ]
    if not strong_bias.empty:
        print(f"\n📈 Strong Directional Signals:")
        for _, row in strong_bias.iterrows():
            direction = "↑" if row['90d_overall_bias'] >= 70 else "↓"
            print(f"  {row['ticker']}: {row['90d_overall_bias']:.0f}% bias {direction}, {row['90_break_fmt']} breaks, {row['90d_drift']:+.1f}% drift")
    
    print(f"\n💡 Remember: Past patterns ≠ Future results. IV context shows current opportunity cost.")