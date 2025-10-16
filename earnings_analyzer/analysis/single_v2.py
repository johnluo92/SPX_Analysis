"""Single ticker earnings analysis - V2 using typed models

This is a refactored version that returns AnalysisResult objects instead of dicts.
The old single.py remains unchanged for backward compatibility.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional

from ..config import DEFAULT_LOOKBACK_QUARTERS, MIN_QUARTERS_REQUIRED
from ..data_sources import AlphaVantageClient, YahooFinanceClient
from ..calculations import (
    calculate_historical_volatility,
    calculate_strike_width,
    get_reference_price,
    find_nearest_price,
    calculate_stats
)
from ..core.models import AnalysisResult, TimeframeStats


def analyze_ticker_v2(ticker: str, 
                       lookback_quarters: int = DEFAULT_LOOKBACK_QUARTERS, 
                       verbose: bool = True, 
                       debug: bool = False) -> Tuple[Optional[AnalysisResult], str]:
    """
    Analyze post-earnings movements for a single ticker
    
    V2: Returns typed AnalysisResult instead of dictionary
    
    Args:
        ticker: Stock ticker symbol
        lookback_quarters: Number of quarters to analyze
        verbose: Print detailed output
        debug: Print debug information
    
    Returns:
        (AnalysisResult, status_string)
    """
    if verbose:
        print(f"\n{'='*75}")
        print(f"📊 {ticker} - Post-Earnings Containment Analysis")
        print(f"{'='*75}")
    
    # Fetch data (same as old version)
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
    
    # Calculate movements (same logic as old version)
    data_45 = []
    data_90 = []
    hvol_list = []
    realized_moves_45 = []
    
    for earnings in past_earnings:
        hvol = calculate_historical_volatility(price_data, earnings['date'])
        if hvol is None:
            continue
        
        hvol_list.append(hvol * 100)
        
        ref_price, ref_date = get_reference_price(price_data, earnings['date'], earnings['time'])
        if ref_price is None:
            continue
        
        strike_width_45 = calculate_strike_width(hvol, 45, multiplier=1.0)
        strike_width_90 = calculate_strike_width(hvol, 90, multiplier=1.0)
        
        # 45-day calculations
        target_45 = earnings['date'] + timedelta(days=45)
        if target_45 <= today:
            price_45, date_45 = find_nearest_price(price_data, target_45)
            if price_45 is not None:
                move_45 = (price_45 - ref_price) / ref_price * 100
                realized_moves_45.append(move_45)
                
                data_45.append({
                    'move': move_45,
                    'width': strike_width_45,
                    'hvol': hvol * 100,
                    'date': earnings['date'].strftime('%Y-%m-%d'),
                    'ref_date': ref_date.strftime('%Y-%m-%d') if ref_date else None
                })
        
        # 90-day calculations
        target_90 = earnings['date'] + timedelta(days=90)
        if target_90 <= today:
            price_90, date_90 = find_nearest_price(price_data, target_90)
            if price_90 is not None:
                move_90 = (price_90 - ref_price) / ref_price * 100
                data_90.append({
                    'move': move_90,
                    'width': strike_width_90,
                    'hvol': hvol * 100,
                    'date': earnings['date'].strftime('%Y-%m-%d'),
                    'ref_date': ref_date.strftime('%Y-%m-%d') if ref_date else None
                })
    
    if len(data_45) < MIN_QUARTERS_REQUIRED or len(data_90) < MIN_QUARTERS_REQUIRED:
        if verbose:
            print(f"⚠️  Insufficient valid data")
        return None, "insufficient_valid_data"
    
    # Calculate statistics (reuse existing function)
    stats_45_dict = calculate_stats(data_45)
    stats_90_dict = calculate_stats(data_90)
    
    # Convert to typed TimeframeStats
    stats_45 = TimeframeStats(
        total=stats_45_dict['total'],
        containment=stats_45_dict['containment'],
        breaks_up=stats_45_dict['breaks_up'],
        breaks_down=stats_45_dict['breaks_down'],
        trend_pct=stats_45_dict['trend_pct'],
        break_up_pct=stats_45_dict['break_up_pct'],
        drift_pct=stats_45_dict['drift_pct'],
        drift_vs_width=stats_45_dict['drift_vs_width'],
        avg_width=stats_45_dict['avg_width']
    )
    
    stats_90 = TimeframeStats(
        total=stats_90_dict['total'],
        containment=stats_90_dict['containment'],
        breaks_up=stats_90_dict['breaks_up'],
        breaks_down=stats_90_dict['breaks_down'],
        trend_pct=stats_90_dict['trend_pct'],
        break_up_pct=stats_90_dict['break_up_pct'],
        drift_pct=stats_90_dict['drift_pct'],
        drift_vs_width=stats_90_dict['drift_vs_width'],
        avg_width=stats_90_dict['avg_width']
    )
    
    avg_hvol = np.mean(hvol_list)
    
    # Calculate RVol45
    rvol_45d = None
    if len(realized_moves_45) >= MIN_QUARTERS_REQUIRED:
        rvol_45d = np.std(realized_moves_45, ddof=1)
    
    # Create typed result
    result = AnalysisResult(
        ticker=ticker,
        hvol=round(avg_hvol, 1),
        strike_width=round(stats_90.avg_width, 1),
        rvol_45d=round(rvol_45d, 1) if rvol_45d is not None else None,
        stats_45=stats_45,
        stats_90=stats_90,
        earnings_history=data_90  # Keep for audit trail
    )
    
    # Print output (same as old version)
    if verbose:
        rvol_display = f"{rvol_45d:.1f}%" if rvol_45d else "N/A"
        print(f"\n📊 {ticker} | {avg_hvol:.1f}% HVol | RVol45: {rvol_display}")
        print(f"    Average 45D width: ±{stats_45.avg_width:.1f}% | 90D width: ±{stats_90.avg_width:.1f}%")
        print(f"\n  45-Day: {stats_45.total}/{lookback_quarters} tested")
        print(f"    Containment: {stats_45.containment:.0f}%")
        print(f"    Breaks: Up {stats_45.breaks_up}, Down {stats_45.breaks_down}")
        print(f"    Trend: {stats_45.trend_pct:.0f}% up")
        print(f"    Break Direction: {stats_45.break_up_pct:.0f}% of breaks were upward")
        print(f"    Avg Drift: {stats_45.drift_pct:+.1f}% ({stats_45.drift_vs_width:+.0f}% of width)")
        
        print(f"\n  90-Day: {stats_90.total}/{lookback_quarters} tested")
        print(f"    Containment: {stats_90.containment:.0f}%")
        print(f"    Breaks: Up {stats_90.breaks_up}, Down {stats_90.breaks_down}")
        print(f"    Trend: {stats_90.trend_pct:.0f}% up")
        print(f"    Break Direction: {stats_90.break_up_pct:.0f}% of breaks were upward")
        print(f"    Avg Drift: {stats_90.drift_pct:+.1f}% ({stats_90.drift_vs_width:+.0f}% of width)")
        
        print(f"\n  💡 Strategy: {result.combined_strategy}")
    
    return result, "success"


# Backward compatibility wrapper
def analyze_ticker_v2_compat(ticker: str, 
                              lookback_quarters: int = DEFAULT_LOOKBACK_QUARTERS,
                              verbose: bool = True, 
                              debug: bool = False) -> Tuple[Optional[dict], str]:
    """
    Wrapper that returns dict for backward compatibility
    
    This lets us test the new version while keeping old code working
    """
    result, status = analyze_ticker_v2(ticker, lookback_quarters, verbose, debug)
    if result is None:
        return None, status
    return result.to_dict(), status