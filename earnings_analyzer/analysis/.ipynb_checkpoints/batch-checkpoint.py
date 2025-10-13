"""Batch processing with parallel execution"""
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import DEFAULT_LOOKBACK_QUARTERS, REQUEST_DELAY, ALPHAVANTAGE_KEYS
from ..cache import load_cache, load_rate_limits
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
    print("\n" + "="*75)
    print(f"EARNINGS CONTAINMENT ANALYZER - v2.9")
    print(f"Lookback: {lookback_quarters} quarters (~{lookback_quarters/4:.0f} years)")
    if fetch_iv:
        print(f"Current IV from Yahoo Finance (15-20min delayed)")
    if parallel:
        print(f"Parallel processing: {max_workers} workers")
    print("="*75)
    
    rate_limited_keys = load_rate_limits()
    if rate_limited_keys:
        available = len(ALPHAVANTAGE_KEYS) - len(rate_limited_keys)
        print(f"\n⚠️  Rate Limit: {available}/{len(ALPHAVANTAGE_KEYS)} API keys available")
    else:
        print(f"\n✓ All {len(ALPHAVANTAGE_KEYS)} API keys available")
    
    # ============================================================================
    # ⚠️  DO NOT TOUCH - TICKER ORDER PRESERVATION START
    # ============================================================================
    # This dictionary maintains the exact input order of tickers
    # Key: ticker symbol, Value: original index position
    ticker_order = {ticker: idx for idx, ticker in enumerate(tickers)}
    # ============================================================================
    # ⚠️  DO NOT TOUCH - TICKER ORDER PRESERVATION END
    # ============================================================================
    
    results = []
    fetch_summary = {'cached': [], 'api': [], 'failed': []}
    iv_summary = {'success': [], 'failed': []}
    
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
    
    _print_fetch_summary(fetch_summary, iv_summary, fetch_iv)
    
    if not results:
        print("\n⚠️  No valid results")
        return None
    
    # ============================================================================
    # ⚠️  DO NOT TOUCH - TICKER ORDER PRESERVATION START
    # ============================================================================
    # Sort results by original input order before creating DataFrame
    results.sort(key=lambda x: ticker_order.get(x['ticker'], 999))
    # ============================================================================
    # ⚠️  DO NOT TOUCH - TICKER ORDER PRESERVATION END
    # ============================================================================
    
    df = _create_results_dataframe(results)
    _print_results_table(df)
    # KEY TAKEAWAYS REMOVED
    
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
    """Process tickers in parallel"""
    results = []
    cache = load_cache()
    yf_client = YahooFinanceClient()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(analyze_ticker, ticker, lookback_quarters, False, debug): ticker
            for ticker in tickers
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            print(f"\r[{completed}/{len(tickers)}] Processing {ticker}...", end='', flush=True)
            
            try:
                summary, status = future.result()
                from_cache = ticker in cache
                
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
            except Exception as e:
                print(f"\n❌ Error processing {ticker}: {e}")
                fetch_summary['failed'].append(ticker)
    
    return results


def _fetch_and_add_iv(ticker, summary, yf_client, iv_summary):
    """Fetch and add IV data to summary"""
    iv_data = yf_client.get_current_iv(ticker)
    if iv_data:
        summary['current_iv'] = iv_data['iv']
        summary['iv_dte'] = iv_data['dte']
        
        # Compare IV to RVol45 (if available), otherwise fall back to hvol
        if 'rvol_45d' in summary and summary['rvol_45d'] is not None:
            benchmark = summary['rvol_45d']
        else:
            benchmark = summary.get('hvol', 0)
        
        if benchmark > 0:
            iv_premium = ((iv_data['iv'] - benchmark) / benchmark) * 100
            summary['iv_premium'] = round(iv_premium, 1)
        else:
            summary['iv_premium'] = None
            
        iv_summary['success'].append(ticker)
    else:
        summary['current_iv'] = None
        summary['iv_dte'] = None
        summary['iv_premium'] = None
        iv_summary['failed'].append(ticker)


def _print_fetch_summary(fetch_summary, iv_summary, fetch_iv):
    """Print fetch summary"""
    print(f"\n📊 FETCH SUMMARY")
    print(f"{'='*75}")
    if fetch_summary['cached']:
        cached_list = ', '.join(fetch_summary['cached'][:5])
        if len(fetch_summary['cached']) > 5:
            cached_list += '...'
        print(f"✓ Earnings Cached ({len(fetch_summary['cached'])}): {cached_list}")
    if fetch_summary['api']:
        print(f"✓ Earnings API ({len(fetch_summary['api'])}): {', '.join(fetch_summary['api'])}")
    if fetch_iv:
        if iv_summary['success']:
            iv_list = ', '.join(iv_summary['success'][:5])
            if len(iv_summary['success']) > 5:
                iv_list += '...'
            print(f"✓ IV Retrieved ({len(iv_summary['success'])}): {iv_list}")
        if iv_summary['failed']:
            print(f"✗ IV Failed ({len(iv_summary['failed'])}): {', '.join(iv_summary['failed'])}")
    if fetch_summary['failed']:
        print(f"✗ Analysis Failed ({len(fetch_summary['failed'])}): {', '.join(fetch_summary['failed'])}")


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
    """
    Create formatted results dataframe
    
    ⚠️  DO NOT TOUCH - ORDER PRESERVATION
    Results list is already sorted by input order before this function is called
    Do NOT sort, reorder, or modify the sequence of results here
    """
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
    """
    Print results table with RVol45 comparison
    
    ⚠️  DO NOT TOUCH - ORDER PRESERVATION
    DataFrame is already in correct order, just print it as-is
    """
    print(f"\n{'='*110}")
    print("BACKTEST RESULTS")
    print("="*110)
    
    display_cols = {
        'Ticker': df['ticker'],
    }
    
    # Add RVol45 if available
    if 'rvol_45d' in df.columns:
        display_cols['RVol45'] = df['rvol_45d'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
    
    # Add IV data with DTE
    if 'current_iv' in df.columns and df['current_iv'].notna().any():
        display_cols['IV45'] = df.apply(
            lambda x: f"{int(x['current_iv'])}" if pd.notna(x['current_iv']) else "N/A",
            axis=1
        )
        display_cols['DTE'] = df['iv_dte'].apply(lambda x: f"{int(x)}d" if pd.notna(x) else "N/A")
        
        # vs45RV = IV45 vs RVol45 comparison
        if 'rvol_45d' in df.columns:
            display_cols['vs45RV'] = df.apply(
                lambda x: f"{x['iv_premium']:+.0f}%" if pd.notna(x.get('iv_premium')) else "N/A",
                axis=1
            )
        
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
    
    # ============================================================================
    # ⚠️  DO NOT TOUCH - ORDER PRESERVATION START
    # ============================================================================
    # Print with index=False to show data in exact order received
    # Do NOT sort, reset_index, or reorder before printing
    print(display_df.to_string(index=False))
    # ============================================================================
    # ⚠️  DO NOT TOUCH - ORDER PRESERVATION END
    # ============================================================================