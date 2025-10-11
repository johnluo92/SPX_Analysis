"""Single ticker earnings analysis"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

from ..config import DEFAULT_LOOKBACK_QUARTERS, MIN_QUARTERS_REQUIRED
from ..data_sources import AlphaVantageClient, YahooFinanceClient
from ..calculations import (
    calculate_historical_volatility,
    get_volatility_tier,
    calculate_strike_width,
    get_reference_price,
    find_nearest_price,
    calculate_stats,
    determine_strategy,
    calculate_rvol_tier
)


def analyze_ticker(ticker: str, lookback_quarters: int = DEFAULT_LOOKBACK_QUARTERS, 
                   verbose: bool = True, debug: bool = False) -> Tuple[Optional[Dict], str]:
    """
    Analyze post-earnings movements for a single ticker
    
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
        print(f"📊 {ticker} - Post-Earnings Containment Analysis")
        print(f"{'='*75}")
    
    av_client = AlphaVantageClient()
    yf_client = YahooFinanceClient()
    
    earnings_info, status = av_client.get_earnings(ticker, debug=debug)
    if not earnings_info:
        return None, status
    
    today = datetime.now()
    past_earnings = [e for e in earnings_info if e['date'] < today][:lookback_quarters]
    
    if len(past_earnings) < MIN_QUARTERS_REQUIRED:
        if verbose:
            print(f"⚠️  Insufficient data: only {len(past_earnings)} earnings periods")
        return None, "insufficient_quarters"
    
    oldest = min([e['date'] for e in past_earnings]) - timedelta(days=120)
    price_data = yf_client.get_price_data(ticker, oldest, today)
    
    if price_data.empty:
        return None, "no_price_data"
    
    data_45 = []
    data_90 = []
    hvol_list = []
    rvol_45d_list = []
    rvol_90d_list = []
    
    for i, earnings in enumerate(past_earnings):
        hvol = calculate_historical_volatility(price_data, earnings['date'])
        if hvol is None:
            continue
        
        hvol_list.append(hvol * 100)
        
        ref_price, ref_date = get_reference_price(price_data, earnings['date'], earnings['time'])
        if ref_price is None:
            continue
        
        # Calculate RVol-based tiers (leave-one-out for this earnings period)
        tier_45 = calculate_rvol_tier(data_45, 45, hvol) if i > 0 else get_volatility_tier(hvol)
        tier_90 = calculate_rvol_tier(data_90, 90, hvol) if i > 0 else get_volatility_tier(hvol)
        
        strike_width_45 = hvol * np.sqrt(45/365) * tier_45 * 100
        strike_width_90 = hvol * np.sqrt(90/365) * tier_90 * 100
        
        target_45 = earnings['date'] + timedelta(days=45)
        if target_45 <= today:
            price_45, date_45 = find_nearest_price(price_data, target_45)
            if price_45 is not None:
                move_45 = (price_45 - ref_price) / ref_price * 100
                realized_vol_45 = abs(move_45) / np.sqrt(45/365)
                rvol_45d_list.append(realized_vol_45)
                data_45.append({
                    'move': move_45,
                    'width': strike_width_45,
                    'hvol': hvol * 100,
                    'rvol': realized_vol_45,
                    'date': earnings['date'].strftime('%Y-%m-%d')
                })
        
        target_90 = earnings['date'] + timedelta(days=90)
        if target_90 <= today:
            price_90, date_90 = find_nearest_price(price_data, target_90)
            if price_90 is not None:
                move_90 = (price_90 - ref_price) / ref_price * 100
                realized_vol_90 = abs(move_90) / np.sqrt(90/365)
                rvol_90d_list.append(realized_vol_90)
                data_90.append({
                    'move': move_90,
                    'width': strike_width_90,
                    'hvol': hvol * 100,
                    'rvol': realized_vol_90,
                    'date': earnings['date'].strftime('%Y-%m-%d')
                })
    
    if len(data_45) < MIN_QUARTERS_REQUIRED or len(data_90) < MIN_QUARTERS_REQUIRED:
        if verbose:
            print(f"⚠️  Insufficient valid data")
        return None, "insufficient_valid_data"
    
    stats_45 = calculate_stats(data_45)
    stats_90 = calculate_stats(data_90)
    avg_hvol = np.mean(hvol_list)
    avg_rvol_45d = np.mean(rvol_45d_list)
    avg_rvol_90d = np.mean(rvol_90d_list)
    
    recommendation = determine_strategy(stats_45, stats_90)
    
    # SIMPLIFIED VERBOSE OUTPUT
    if verbose:
        print(f"\n📊 {ticker} | HVol: {avg_hvol:.1f}% | RVol: 45d={avg_rvol_45d:.1f}%, 90d={avg_rvol_90d:.1f}%")
        print(f"\n  45d: {stats_45['containment']:.0f}% contain | {stats_45['overall_bias']:.0f}% bias | {stats_45['breaks_up']}:{stats_45['breaks_down']} breaks | {stats_45['avg_move_pct']:+.1f}% drift")
        print(f"  90d: {stats_90['containment']:.0f}% contain | {stats_90['overall_bias']:.0f}% bias | {stats_90['breaks_up']}:{stats_90['breaks_down']} breaks | {stats_90['avg_move_pct']:+.1f}% drift")
        print(f"\n  💡 Strategy: {recommendation}")
    
    summary = {
        'ticker': ticker,
        'hvol': round(avg_hvol, 1),
        'rvol_45d': round(avg_rvol_45d, 1),
        'rvol_90d': round(avg_rvol_90d, 1),
        'strike_width': round(stats_90['avg_width'], 1),
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