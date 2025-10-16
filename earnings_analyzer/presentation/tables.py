"""Table generation for analysis results

Extracted from batch.py - handles all table printing
"""
import pandas as pd
from typing import List, Union
from tabulate import tabulate

from ..core.models import AnalysisResult
from ..calculations.strategy import determine_strategy_45, determine_strategy_90
from .formatters import (
    format_trend, 
    format_breaks, 
    format_iv_with_dte,
    clean_pattern
)


def print_results_table(results: Union[pd.DataFrame, List[AnalysisResult]], 
                       fetch_iv: bool = False) -> None:
    """Print clean, aligned results table
    
    Args:
        results: Either DataFrame (legacy) or List[AnalysisResult] (new)
        fetch_iv: Whether IV column should be shown
    """
    # Convert to DataFrame if needed
    if isinstance(results, list):
        # Assume list of AnalysisResult objects
        df = pd.DataFrame([r.to_dict() for r in results])
    else:
        df = results
    
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
            'break_up_pct': row['45d_break_up_pct'],
            'trend_pct': row['45d_trend_pct'],
            'drift_pct': row['45d_drift']
        }
        stats_90 = {
            'containment': row['90d_contain'],
            'breaks_up': row['90d_breaks_up'],
            'breaks_down': row['90d_breaks_dn'],
            'break_up_pct': row['90d_break_up_pct'],
            'trend_pct': row['90d_trend_pct'],
            'drift_pct': row['90d_drift']
        }
        
        pattern_45, edge_45 = determine_strategy_45(stats_45)
        pattern_90, edge_90 = determine_strategy_90(stats_90)
        
        strategies_45d.append(pattern_45)
        strategies_90d.append(pattern_90)
        edges_45d.append(edge_45)
        edges_90d.append(edge_90)
    
    # Build table data
    table_data = []
    for i, row in df.iterrows():
        row_data = [
            row['ticker'],
            int(row['hvol']),
            # 45d section
            int(row['45d_contain']),
            format_trend(row['45d_trend_pct']),
            format_breaks(row['45d_breaks_up'], row['45d_breaks_dn']),
            f"{row['45d_drift']:+.1f}%",
            clean_pattern(strategies_45d[i]),
            edges_45d[i],
            # 90d section
            int(row['90d_contain']),
            format_trend(row['90d_trend_pct']),
            format_breaks(row['90d_breaks_up'], row['90d_breaks_dn']),
            f"{row['90d_drift']:+.1f}%",
            clean_pattern(strategies_90d[i]),
            edges_90d[i]
        ]
        
        # Add IV column if fetch_iv is enabled
        if fetch_iv:
            row_data.append(format_iv_with_dte(row.get('current_iv'), row.get('iv_dte')))
        
        table_data.append(row_data)
    
    # Headers
    headers = [
        "Ticker",
        "HVol%",
        "45d%",
        "45Trend",
        "45Brk",
        "45Drift",
        "45Pattern",
        "45E",
        "90d%",
        "90Trend",
        "90Brk",
        "90Drift",
        "90Pattern",
        "90E"
    ]
    
    if fetch_iv:
        headers.append("CurrIV")
    
    # Print with clean formatting
    print("\n" + "="*165)
    print("BACKTEST RESULTS")
    print("="*165)
    
    table_str = tabulate(
        table_data, 
        headers=headers, 
        tablefmt='grid',
        numalign='right',
        stralign='left',
        disable_numparse=True
    )
    
    print(table_str)


def print_fetch_summary(fetch_summary: dict, iv_summary: dict, fetch_iv: bool) -> None:
    """Print data fetch summary"""
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