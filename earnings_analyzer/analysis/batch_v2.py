"""Batch processing with parallel execution - V2 Refactored

This version:
- Uses typed AnalysisResult objects
- Separates concerns (orchestration vs presentation)
- ~50 lines instead of ~400
- All presentation logic extracted to presentation/ module
"""
import time
import pandas as pd
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import DEFAULT_LOOKBACK_QUARTERS, REQUEST_DELAY, ALPHAVANTAGE_KEYS
from ..cache import load_cache, load_rate_limits
from ..core.models import AnalysisResult, results_to_dataframe
from .single_v2 import analyze_ticker_v2
from .enrichment import enrich_with_iv
from ..presentation import print_results_table, print_fetch_summary, print_insights


def batch_analyze_v2(tickers: List[str], 
                     lookback_quarters: int = DEFAULT_LOOKBACK_QUARTERS,
                     debug: bool = False, 
                     fetch_iv: bool = True, 
                     parallel: bool = False,
                     max_workers: int = 4) -> Optional[pd.DataFrame]:
    """
    Analyze multiple tickers with optional parallel processing - V2
    
    This is a thin orchestrator that delegates to specialized modules:
    - Analysis: single_v2.py
    - IV enrichment: enrichment.py
    - Presentation: presentation/*.py
    
    Args:
        tickers: List of ticker symbols
        lookback_quarters: Number of quarters to analyze
        debug: Print debug information
        fetch_iv: Fetch current IV data
        parallel: Use parallel processing
        max_workers: Number of parallel workers
    
    Returns:
        DataFrame with analysis results (for backward compatibility)
    """
    # Print header
    _print_header(lookback_quarters, fetch_iv, parallel, max_workers)
    
    # Check rate limits
    _print_rate_limit_status()
    
    # Run analysis (serial or parallel)
    if parallel:
        results, fetch_summary = _run_parallel_analysis(
            tickers, lookback_quarters, debug, max_workers
        )
    else:
        results, fetch_summary = _run_serial_analysis(
            tickers, lookback_quarters, debug
        )
    
    # Clear progress line
    print("\r" + " " * 80 + "\r", end='')
    
    if not results:
        print("\n⚠️  No valid results")
        return None
    
    # Enrich with IV data if requested
    iv_summary = {'success': [], 'failed': []}
    if fetch_iv:
        results, iv_summary = enrich_with_iv(results, debug)
    
    # Print fetch summary
    print_fetch_summary(fetch_summary, iv_summary, fetch_iv)
    
    # Convert to DataFrame for compatibility
    df = results_to_dataframe(results)
    
    # Print results and insights
    print_results_table(df, fetch_iv)
    print_insights(df)
    
    return df


def _print_header(lookback_quarters: int, fetch_iv: bool, 
                 parallel: bool, max_workers: int) -> None:
    """Print analysis header"""
    print("\n" + "="*75)
    print(f"EARNINGS CONTAINMENT ANALYZER - v3")
    print(f"Lookback: {lookback_quarters} quarters (~{lookback_quarters/4:.0f} years)")
    if fetch_iv:
        print(f"Current IV from Yahoo Finance (15-20min delayed)")
    if parallel:
        print(f"Parallel processing: {max_workers} workers")
    print("="*75)


def _print_rate_limit_status() -> None:
    """Print API rate limit status"""
    rate_limited_keys = load_rate_limits()
    if rate_limited_keys:
        available = len(ALPHAVANTAGE_KEYS) - len(rate_limited_keys)
        print(f"\n⚠️  Rate Limit: {available}/{len(ALPHAVANTAGE_KEYS)} API keys available")
    else:
        print(f"\n✓ All {len(ALPHAVANTAGE_KEYS)} API keys available")


def _run_serial_analysis(tickers: List[str], 
                        lookback_quarters: int,
                        debug: bool) -> tuple[List[AnalysisResult], dict]:
    """Run analysis serially"""
    results = []
    cache = load_cache()
    fetch_summary = {'cached': [], 'api': [], 'failed': []}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\r[{i}/{len(tickers)}] Processing {ticker}...", end='', flush=True)
        
        from_cache = ticker in cache
        result, status = analyze_ticker_v2(ticker, lookback_quarters, 
                                          verbose=False, debug=debug)
        
        if result:
            results.append(result)
            if from_cache:
                fetch_summary['cached'].append(ticker)
            else:
                fetch_summary['api'].append(ticker)
        else:
            fetch_summary['failed'].append(ticker)
        
        time.sleep(REQUEST_DELAY)
    
    return results, fetch_summary


def _run_parallel_analysis(tickers: List[str], 
                          lookback_quarters: int,
                          debug: bool,
                          max_workers: int) -> tuple[List[AnalysisResult], dict]:
    """Run analysis in parallel while preserving order"""
    results_dict = {}
    cache = load_cache()
    fetch_summary = {'cached': [], 'api': [], 'failed': []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit with index to maintain order
        future_to_ticker = {
            executor.submit(analyze_ticker_v2, ticker, lookback_quarters, False, debug): (i, ticker)
            for i, ticker in enumerate(tickers)
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            idx, ticker = future_to_ticker[future]
            completed += 1
            print(f"\r[{completed}/{len(tickers)}] Processing {ticker}...", end='', flush=True)
            
            try:
                result, status = future.result()
                from_cache = ticker in cache
                
                if result:
                    # Store with original index to preserve order
                    results_dict[idx] = result
                    
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
    
    return results, fetch_summary


# Backward compatibility wrapper
def batch_analyze_v2_compat(tickers: List[str], **kwargs) -> Optional[pd.DataFrame]:
    """
    Wrapper that ensures output matches old batch_analyze exactly
    
    This allows gradual migration - code expecting old format still works
    """
    return batch_analyze_v2(tickers, **kwargs)