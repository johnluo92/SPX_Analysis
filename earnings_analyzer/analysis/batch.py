"""Batch processing with parallel execution"""
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import DEFAULT_LOOKBACK_QUARTERS, REQUEST_DELAY, ALPHAVANTAGE_KEYS
from ..cache import load_cache, load_rate_limits
from ..data_sources import YahooFinanceClient, AlphaVantageClient
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
    print(f"EARNINGS CONTAINMENT ANALYZER - v3")
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
    
    results = []
    fetch_summary = {'cached': [], 'api': [], 'failed': []}
    iv_summary = {'success': [], 'failed': []}
    earnings_summary = {'success': [], 'failed': []}
    
    if parallel:
        results = _batch_analyze_parallel(
            tickers, lookback_quarters, debug, fetch_iv, 
            max_workers, fetch_summary, iv_summary, earnings_summary
        )
    else:
        results = _batch_analyze_serial(
            tickers, lookback_quarters, debug, fetch_iv,
            fetch_summary, iv_summary, earnings_summary
        )
    
    print("\r" + " " * 80 + "\r", end='')
    
    _print_fetch_summary(fetch_summary, iv_summary, earnings_summary, fetch_iv)
    
    if not results:
        print("\n⚠️  No valid results")
        return None
    
    df = _create_results_dataframe(results)
    _print_results_table(df)
    _print_insights(df)
    
    return df


def _batch_analyze_serial(tickers, lookback_quarters, debug, fetch_iv, 
                          fetch_summary, iv_summary, earnings_summary):
    """Process tickers serially"""
    results = []
    cache = load_cache()
    yf_client = YahooFinanceClient()
    av_client = AlphaVantageClient()
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\r[{i}/{len(tickers)}] Processing {ticker}...", end='', flush=True)
        
        from_cache = ticker in cache
        summary, status = analyze_ticker(ticker, lookback_quarters, verbose=False, debug=debug)
        
        if summary:
            if fetch_iv:
                _fetch_and_add_iv(ticker, summary, yf_client, iv_summary)
            
            # Add upcoming earnings countdown
            _fetch_and_add_earnings_countdown(ticker, summary, av_client, earnings_summary)
            
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
                           max_workers, fetch_summary, iv_summary, earnings_summary):
    """Process tickers in parallel while preserving order"""
    results_dict = {}
    cache = load_cache()
    yf_client = YahooFinanceClient()
    av_client = AlphaVantageClient()
    
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
                    
                    # Add upcoming earnings countdown
                    _fetch_and_add_earnings_countdown(ticker, summary, av_client, earnings_summary)
                    
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
        # IV elevation vs RVol45d (apples-to-apples comparison)
        if 'rvol_45d' in summary and summary['rvol_45d'] is not None:
            iv_elevation = ((iv_data['iv'] - summary['rvol_45d']) / summary['rvol_45d']) * 100
            summary['iv_elevation'] = round(iv_elevation, 1)
        else:
            summary['iv_elevation'] = None
        iv_summary['success'].append(ticker)
    else:
        summary['current_iv'] = None
        summary['iv_dte'] = None
        summary['iv_elevation'] = None
        iv_summary['failed'].append(ticker)


def _fetch_and_add_earnings_countdown(ticker, summary, av_client, earnings_summary):
    """Fetch and add earnings countdown to summary"""
    upcoming, status = av_client.get_upcoming_earnings(ticker, use_cache=True, debug=False)
    
    if upcoming and upcoming.get('date'):
        try:
            today = datetime.now().date()
            earnings_date = datetime.strptime(upcoming['date'], '%Y-%m-%d').date()
            days_until = (earnings_date - today).days
            
            summary['days_till_earnings'] = days_until
            summary['earnings_date'] = upcoming['date']
            earnings_summary['success'].append(ticker)
        except:
            summary['days_till_earnings'] = None
            summary['earnings_date'] = None
            earnings_summary['failed'].append(ticker)
    else:
        summary['days_till_earnings'] = None
        summary['earnings_date'] = None
        earnings_summary['failed'].append(ticker)


def _format_days_till(days):
    """Format days till earnings for display"""
    if days is None or (isinstance(days, float) and pd.isna(days)):
        return "N/A"
    return str(int(days))


def _print_fetch_summary(fetch_summary, iv_summary, earnings_summary, fetch_iv):
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
    if earnings_summary['success']:
        earnings_list = ', '.join(earnings_summary['success'][:5])
        if len(earnings_summary['success']) > 5:
            earnings_list += '...'
        print(f"✓ Upcoming Earnings ({len(earnings_summary['success'])}): {earnings_list}")
    if earnings_summary['failed']:
        print(f"✗ Upcoming Earnings Failed ({len(earnings_summary['failed'])}): {', '.join(earnings_summary['failed'])}")
    if fetch_summary['failed']:
        print(f"✗ Analysis Failed ({len(fetch_summary['failed'])}): {', '.join(fetch_summary['failed'])}")


def _format_break_ratio(up_breaks, down_breaks, break_bias):
    """DEPRECATED - kept for backward compatibility"""
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
    
    # Format break ratios (kept for backward compatibility with insights)
    df['45_break_fmt'] = df.apply(
        lambda x: _format_break_ratio(x['45d_breaks_up'], x['45d_breaks_dn'], x['45d_break_bias']), 
        axis=1
    )
    df['90_break_fmt'] = df.apply(
        lambda x: _format_break_ratio(x['90d_breaks_up'], x['90d_breaks_dn'], x['90d_break_bias']), 
        axis=1
    )
    
    # DO NOT SORT - preserve input order
    
    return df


def _print_results_table(df):
    """Print clean, aligned table using tabulate - perfect for Jupyter notebooks"""
    
    from tabulate import tabulate
    from ..calculations.strategy import determine_strategy_45, determine_strategy_90
    
    # Calculate 45d and 90d strategies separately for each row
    strategies_45d = []
    strategies_90d = []
    edges_45d = []
    edges_90d = []
    
    for _, row in df.iterrows():
        stats_45 = {
            'containment': row['45d_contain'],
            'breaks_up': row['45d_breaks_up'],
            'breaks_down': row['45d_breaks_dn'],
            'break_bias': row['45d_break_bias'],
            'overall_bias': row['45d_overall_bias'],
            'avg_move_pct': row['45d_drift']
        }
        stats_90 = {
            'containment': row['90d_contain'],
            'breaks_up': row['90d_breaks_up'],
            'breaks_down': row['90d_breaks_dn'],
            'break_bias': row['90d_break_bias'],
            'overall_bias': row['90d_overall_bias'],
            'avg_move_pct': row['90d_drift']
        }
        
        pattern_45, edge_45 = determine_strategy_45(stats_45)
        pattern_90, edge_90 = determine_strategy_90(stats_90)
        
        strategies_45d.append(pattern_45)
        strategies_90d.append(pattern_90)
        edges_45d.append(edge_45)
        edges_90d.append(edge_90)
    
    # Format helpers
    def format_bias(bias_pct):
        if bias_pct >= 66.7:
            return f"↑{bias_pct:.0f}%"
        elif bias_pct <= 33.3:
            down_pct = 100 - bias_pct
            return f"↓{down_pct:.0f}%"
        else:
            if bias_pct >= 50:
                return f"↑{bias_pct:.0f}%"
            else:
                down_pct = 100 - bias_pct
                return f"↓{down_pct:.0f}%"
    
    def format_breaks(up, down):
        if up == 0 and down == 0:
            return "0:0"
        total = up + down
        up_pct = (up / total) * 100
        
        if up > down and up_pct >= 60:
            return f"{up}:{down}↑({up_pct:.0f}%)"
        elif down > up and (100 - up_pct) >= 60:
            return f"{up}:{down}↓({100-up_pct:.0f}%)"
        else:
            return f"{up}:{down}({up_pct:.0f}%)"
    
    def clean_pattern(pattern):
        if pattern == "SKIP":
            return "-"
        if "(" in pattern:
            pattern = pattern.split("(")[0].strip()
        return pattern.replace("BIAS↑", "Bias↑").replace("BIAS↓", "Bias↓").replace("⚠ ", "⚠️")
    
    # Build table data
    table_data = []
    for i, row in df.iterrows():
        table_data.append([
            row['ticker'],
            _format_days_till(row.get('days_till_earnings')),
            int(row['hvol']),
            # 45d section
            int(row['45d_contain']),
            format_bias(row['45d_overall_bias']),
            format_breaks(row['45d_breaks_up'], row['45d_breaks_dn']),
            f"{row['45d_drift']:+.1f}%",
            clean_pattern(strategies_45d[i]),
            edges_45d[i],
            # 90d section
            int(row['90d_contain']),
            format_bias(row['90d_overall_bias']),
            format_breaks(row['90d_breaks_up'], row['90d_breaks_dn']),
            f"{row['90d_drift']:+.1f}%",
            clean_pattern(strategies_90d[i]),
            edges_90d[i]
        ])
    
    # Headers with section separators
    headers = [
        "Ticker",
        "Days Till",
        "HVol%",
        "45d%",
        "45Bias",
        "45Brk",
        "45Drift",
        "45Pattern",
        "45E",
        "90d%",
        "90Bias",
        "90Brk",
        "90Drift",
        "90Pattern",
        "90E"
    ]
    
    # Print with clean formatting
    print("\n" + "="*175)
    print("BACKTEST RESULTS")
    print("="*175)
    
    # Use 'grid' tablefmt for clean boxes
    table_str = tabulate(
        table_data, 
        headers=headers, 
        tablefmt='grid',
        numalign='right',
        stralign='left',
        disable_numparse=True
    )
    
    print(table_str)


def _print_insights(df):
    """Print condensed 3-line insights for quick batch overview"""
    
    # Line 1: Pattern counts
    ic_count = len(df[df['strategy'].str.contains('IC', na=False)])
    bias_up_count = len(df[df['strategy'].str.contains('BIAS↑', na=False)])
    bias_down_count = len(df[df['strategy'].str.contains('BIAS↓', na=False)])
    skip_count = len(df[df['strategy'] == 'SKIP'])
    
    # Line 2: High conviction tickers (show actual edge counts, not capped)
    high_conviction_dict = {}  # {ticker: edge_count}
    high_conviction = df[df['strategy'].str.contains(r'\[(?:\d+) edges?\]', regex=True, na=False)]
    if not high_conviction.empty:
        high_conviction = high_conviction.copy()
        high_conviction['edge_count'] = high_conviction['strategy'].str.extract(r'\[(\d+) edge').astype(int)
        high_conviction = high_conviction[high_conviction['edge_count'] >= 3].sort_values('edge_count', ascending=False)
        for _, row in high_conviction.iterrows():
            high_conviction_dict[row['ticker']] = int(row['edge_count'])
    
    # Line 3: Asymmetric IC warnings
    ic_up_skew = df[(df['strategy'].str.contains('IC.*⚠️↑', regex=True, na=False))]
    ic_down_skew = df[(df['strategy'].str.contains('IC.*⚠️↓', regex=True, na=False))]
    asymmetric_warnings = []
    if not ic_up_skew.empty:
        asymmetric_warnings.extend([f"{t}↑" for t in ic_up_skew['ticker'].tolist()])
    if not ic_down_skew.empty:
        asymmetric_warnings.extend([f"{t}↓" for t in ic_down_skew['ticker'].tolist()])
    
    # Print 3-line condensed format
    print(f"\n{'='*140}")
    print("📊 QUICK INTEL")
    print("="*140)
    
    # Line 1
    print(f"{ic_count} IC candidates | {bias_up_count} Bias↑ | {bias_down_count} Bias↓ | {skip_count} Skip")
    
    # Line 2 - Show edge counts with tickers
    if high_conviction_dict:
        conviction_str = ', '.join([f"{ticker}[{edges}]" for ticker, edges in high_conviction_dict.items()])
        print(f"High conviction (3+ edges): {conviction_str}")
    else:
        print(f"High conviction (3+ edges): None")
    
    # Line 3
    if asymmetric_warnings:
        print(f"⚠️ Asymmetric ICs: {', '.join(asymmetric_warnings)} (adjust wings for directional risk)")
    else:
        print(f"⚠️ Asymmetric ICs: None")
    
    print("="*140)
    
    # Strong directional signals with 45d/90d split
    _print_strong_directional_signals(df)
    
    print(f"\n{'─'*140}")
    print(f"NOTE: Past patterns do not guarantee future results. IV context shows current opportunity cost.")
    print(f"{'─'*140}")


def _print_strong_directional_signals(df):
    """Print strong directional signals split by timeframe"""
    
    # Collect signals for each timeframe
    signals_45d = []
    signals_90d = []
    
    for _, row in df.iterrows():
        if 'BIAS' not in str(row['strategy']):
            continue
        
        # Check 45d signal
        bias_45 = row['45d_overall_bias']
        if bias_45 >= 70 or bias_45 <= 30:
            direction = "↑" if bias_45 >= 70 else "↓"
            breaks_up = row['45d_breaks_up']
            breaks_dn = row['45d_breaks_dn']
            
            # Format break ratio with directional arrow if strong
            breaks_total = breaks_up + breaks_dn
            if breaks_total > 0:
                up_pct = (breaks_up / breaks_total) * 100
                if up_pct >= 60:
                    break_str = f"{breaks_up}:{breaks_dn}↑"
                elif up_pct <= 40:
                    break_str = f"{breaks_up}:{breaks_dn}↓"
                else:
                    break_str = f"{breaks_up}:{breaks_dn}"
            else:
                break_str = "0:0"
            
            signals_45d.append({
                'ticker': row['ticker'],
                'bias': bias_45,
                'direction': direction,
                'break_str': break_str,
                'drift': row['45d_drift']
            })
        
        # Check 90d signal
        bias_90 = row['90d_overall_bias']
        if bias_90 >= 70 or bias_90 <= 30:
            direction = "↑" if bias_90 >= 70 else "↓"
            breaks_up = row['90d_breaks_up']
            breaks_dn = row['90d_breaks_dn']
            
            # Format break ratio with directional arrow if strong
            breaks_total = breaks_up + breaks_dn
            if breaks_total > 0:
                up_pct = (breaks_up / breaks_total) * 100
                if up_pct >= 60:
                    break_str = f"{breaks_up}:{breaks_dn}↑"
                elif up_pct <= 40:
                    break_str = f"{breaks_up}:{breaks_dn}↓"
                else:
                    break_str = f"{breaks_up}:{breaks_dn}"
            else:
                break_str = "0:0"
            
            signals_90d.append({
                'ticker': row['ticker'],
                'bias': bias_90,
                'direction': direction,
                'break_str': break_str,
                'drift': row['90d_drift']
            })
    
    if signals_45d or signals_90d:
        print(f"\n┌─ STRONG DIRECTIONAL SIGNALS")
        
        if signals_45d:
            print(f"│  45d timeframe:")
            for signal in signals_45d:
                print(f"│    {signal['ticker']:6} {signal['bias']:.0f}% bias {signal['direction']}, "
                      f"{signal['break_str']} breaks, {signal['drift']:+.1f}% drift")
        
        if signals_90d:
            if signals_45d:  # Add separator if we had 45d signals
                print(f"│")
            print(f"│  90d timeframe:")
            for signal in signals_90d:
                print(f"│    {signal['ticker']:6} {signal['bias']:.0f}% bias {signal['direction']}, "
                      f"{signal['break_str']} breaks, {signal['drift']:+.1f}% drift")
        
        print(f"└─")