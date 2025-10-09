"""Volatility calculations - Percentile-based strike widths"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List


def calculate_percentile_strike_width(moves: List[float], percentile: float = 75.0) -> float:
    """
    Calculate strike width based on percentile of absolute historical moves
    
    Args:
        moves: List of percentage moves
        percentile: Target percentile (default 75th = 75% containment target)
    
    Returns:
        Strike width as percentage
    """
    if not moves or len(moves) < 2:
        return None
    
    abs_moves = [abs(move) for move in moves]
    return np.percentile(abs_moves, percentile)


def find_nearest_price(price_data: pd.DataFrame, target_date: datetime) -> Tuple[Optional[float], Optional[datetime]]:
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


def get_reference_price(price_data: pd.DataFrame, earnings_date: datetime, 
                       timing: str) -> Tuple[Optional[float], Optional[datetime]]:
    """Get entry price based on earnings timing"""
    target_date = earnings_date - timedelta(days=1) if timing == 'bmo' else earnings_date
    return find_nearest_price(price_data, target_date)