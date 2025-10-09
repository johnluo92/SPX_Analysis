"""Single ticker earnings analysis with Leave-One-Out Cross-Validation"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

from ..config import DEFAULT_LOOKBACK_QUARTERS, MIN_QUARTERS_REQUIRED
from ..data_sources import AlphaVantageClient, YahooFinanceClient
from ..calculations import (
    get_reference_price,
    find_nearest_price,
    calculate_stats,
    determine_strategy
)


def calculate_percentile_strike_width_loo(moves: List[float], current_index: int, percentile: float = 75.0) -> float:
    """
    Calculate strike width using leave-one-out: exclude current move from calculation
    
    Args:
        moves: List of all historical moves
        current_index: Index of move being tested
        percentile: Percentile for strike width (default 75th)
    
    Returns:
        Strike width based on other moves only
    """
    # Exclude current move
    other_moves = [moves[i] for i in range(len(moves)) if i != current_index]
    
    if len(other_moves) < 2:
        # Not enough data for LOO, fallback to simple percentile
        return np.percentile(np.abs(moves), percentile)
    
    # Calculate percentile from absolute values of OTHER moves
    abs_other_moves = np.abs(other_moves)
    return np.percentile(abs_other_moves, percentile)


def analyze_ticker(ticker: str, lookback_quarters: int = DEFAULT_LOOKBACK_QUARTERS, 
                   verbose: bool = True, debug: bool = False) -> Tuple[Optional[Dict], str]:
    """
    Analyze post-earnings movements for a single ticker using Leave-One-Out validation
    
    Args:
        ticker: Stock ticker symbol
        lookback_quarters: Number of quarters to analyze
        verbose: Print detailed output
        debug: Print debug information
    
    Returns:
        (summary_dict, status_string)
    """
    if verbose:
        print(f"\n{'='*75}")
        print(f"📊 {ticker} - Post-Earnings Containment Analysis (LOO)")
        print(f"{'='*75}")
    
    av_client = AlphaVantageClient()
    yf_client = YahooFinanceClient()
    
    # Fetch earnings data
    earnings_info, status = av_client.get_earnings(ticker, debug=debug)
    if not earnings_info:
        return None, status
    
    # Filter past earnings
    today = datetime.now()
    past_earnings = [e for e in earnings_info if e['date'] < today][:lookback_quarters]
    
    if len(past_earnings) < MIN_QUARTERS_REQUIRED:
        if verbose:
            print(f"⚠️  Insufficient data: only {len(past_earnings)} earnings periods")
        return None, "insufficient_quarters"
    
    # Fetch price data
    oldest = min([e['date'] for e in past_earnings]) - timedelta(days=120)
    price_data = yf_client.get_price_data(ticker, oldest, today)
    
    if price_data.empty:
        return None, "no_price_data"
    
    # STEP 1: Collect all raw moves first (no width calculation yet)
    raw_moves_45 = []
    raw_moves_90 = []
    
    for earnings in past_earnings:
        ref_price, ref_date = get_reference_price(price_data, earnings['date'], earnings['time'])
        if ref_price is None:
            continue
        
        # 45-day move
        target_45 = earnings['date'] + timedelta(days=45)
        if target_45 <= today:
            price_45, date_45 = find_nearest_price(price_data, target_45)
            if price_45 is not None:
                move_45 = (price_45 - ref_price) / ref_price * 100
                raw_moves_45.append({
                    'move': move_45,
                    'date': earnings['date'].strftime('%Y-%m-%d')
                })
        
        # 90-day move
        target_90 = earnings['date'] + timedelta(days=90)
        if target_90 <= today:
            price_90, date_90 = find_nearest_price(price_data, target_90)
            if price_90 is not None:
                move_90 = (price_90 - ref_price) / ref_price * 100
                raw_moves_90.append({
                    'move': move_90,
                    'date': earnings['date'].strftime('%Y-%m-%d')
                })
    
    if len(raw_moves_45) < MIN_QUARTERS_REQUIRED or len(raw_moves_90) < MIN_QUARTERS_REQUIRED:
        if verbose:
            print(f"⚠️  Insufficient valid data")
        return None, "insufficient_valid_data"
    
    # STEP 2: Leave-One-Out - Calculate width for each move using OTHER moves
    data_45 = []
    moves_45_only = [m['move'] for m in raw_moves_45]
    
    for i, raw_move in enumerate(raw_moves_45):
        width = calculate_percentile_strike_width_loo(moves_45_only, i, percentile=75.0)
        data_45.append({
            'move': raw_move['move'],
            'width': width,
            'date': raw_move['date']
        })
    
    data_90 = []
    moves_90_only = [m['move'] for m in raw_moves_90]
    
    for i, raw_move in enumerate(raw_moves_90):
        width = calculate_percentile_strike_width_loo(moves_90_only, i, percentile=75.0)
        data_90.append({
            'move': raw_move['move'],
            'width': width,
            'date': raw_move['date']
        })
    
    # STEP 3: Calculate statistics
    stats_45 = calculate_stats(data_45)
    stats_90 = calculate_stats(data_90)
    
    # Calculate RVol (realized volatility = std dev of moves)
    rvol_45 = np.std([d['move'] for d in data_45])
    rvol_90 = np.std([d['move'] for d in data_90])
    
    # Average strike widths used in backtest
    avg_width_45 = np.mean([d['width'] for d in data_45])
    avg_width_90 = np.mean([d['width'] for d in data_90])
    
    # Determine strategy
    recommendation = determine_strategy(stats_45, stats_90)
    
    if verbose:
        print(f"\n📊 {ticker} | RVol45: {rvol_45:.1f}% | RVol90: {rvol_90:.1f}%")
        print(f"    Strike Widths: 45D={avg_width_45:.1f}% | 90D={avg_width_90:.1f}%")
        print(f"\n  45-Day: {stats_45['total']}/{lookback_quarters} tested (LOO)")
        print(f"    Containment: {stats_45['containment']:.0f}%")
        print(f"    Breaks: Up {stats_45['breaks_up']}, Down {stats_45['breaks_down']}")
        print(f"    Overall Bias: {stats_45['overall_bias']:.0f}% up")
        print(f"    Break Bias: {stats_45['break_bias']:.0f}% of breaks were upward")
        print(f"    Avg Drift: {stats_45['avg_move_pct']:+.1f}% ({stats_45['drift_vs_width']:+.0f}% of width)")
        
        print(f"\n  90-Day: {stats_90['total']}/{lookback_quarters} tested (LOO)")
        print(f"    Containment: {stats_90['containment']:.0f}%")
        print(f"    Breaks: Up {stats_90['breaks_up']}, Down {stats_90['breaks_down']}")
        print(f"    Overall Bias: {stats_90['overall_bias']:.0f}% up")
        print(f"    Break Bias: {stats_90['break_bias']:.0f}% of breaks were upward")
        print(f"    Avg Drift: {stats_90['avg_move_pct']:+.1f}% ({stats_90['drift_vs_width']:+.0f}% of width)")
        
        print(f"\n  💡 Strategy: {recommendation}")
    
    # Summary dictionary
    summary = {
        'ticker': ticker,
        'rvol_45': round(rvol_45, 1),
        'rvol_90': round(rvol_90, 1),
        'strike_width_45': round(avg_width_45, 1),
        'strike_width_90': round(avg_width_90, 1),
        '45d_contain': round(stats_45['containment'], 0),
        '45d_breaks_up': stats_45['breaks_up'],
        '45d_breaks_dn': stats_45['breaks_down'],
        '45d_overall_bias': round(stats_45['overall_bias'], 0),
        '45d_break_bias': round(stats_45['break_bias'], 0),
        '45d_drift': round(stats_45['avg_move_pct'], 1),
        '90d_contain': round(stats_90['containment'], 0),
        '90d_breaks_up': stats_90['breaks_up'],
        '90d_breaks_dn': stats_90['breaks_down'],
        '90d_overall_bias': round(stats_90['overall_bias'], 0),
        '90d_break_bias': round(stats_90['break_bias'], 0),
        '90d_drift': round(stats_90['avg_move_pct'], 1),
        'strategy': recommendation
    }
    
    return summary, "success"