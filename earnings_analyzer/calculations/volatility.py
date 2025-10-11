"""Volatility calculations"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

from ..config import HVOL_LOOKBACK_DAYS, VOLATILITY_TIERS


def calculate_historical_volatility(price_data: pd.DataFrame, earnings_date: datetime, 
                                    lookback_days: int = HVOL_LOOKBACK_DAYS) -> Optional[float]:
    """Calculate annualized historical volatility"""
    end_date = earnings_date - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days + 10)
    
    window = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
    
    if len(window) < 20:
        return None
    
    returns = window['close'].pct_change().dropna()
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    
    return annual_vol


def get_volatility_tier(hvol: float) -> float:
    """Map historical volatility to strike width multiplier"""
    hvol_pct = hvol * 100
    
    for threshold, multiplier in VOLATILITY_TIERS:
        if hvol_pct < threshold:
            return multiplier
    
    return VOLATILITY_TIERS[-1][1]


def calculate_rvol_tier(historical_data: List[Dict], dte: int, current_hvol: float) -> float:
    """
    Calculate strike width tier based on realized volatility from past earnings
    
    Uses leave-one-out methodology: calculates tier from historical RVol
    without including the current period being evaluated.
    
    Args:
        historical_data: List of dicts with 'rvol' key from past earnings
        dte: Target days to expiration (45 or 90)
        current_hvol: Current period's HVol (fallback if no history)
    
    Returns:
        Tier multiplier for strike width calculation
    """
    if not historical_data:
        # First period: fallback to HVol-based tier
        return get_volatility_tier(current_hvol)
    
    # Extract RVol values
    rvols = [d['rvol'] for d in historical_data if 'rvol' in d]
    
    if not rvols:
        return get_volatility_tier(current_hvol)
    
    # Calculate average RVol from historical data
    avg_rvol = np.mean(rvols)
    
    # Map avg RVol to tier using same thresholds as HVol
    # This creates consistent tier logic while using realized data
    for threshold, multiplier in VOLATILITY_TIERS:
        if avg_rvol < threshold:
            return multiplier
    
    return VOLATILITY_TIERS[-1][1]


def calculate_strike_width(hvol: float, dte: int) -> float:
    """Calculate strike width for given DTE"""
    strike_std = get_volatility_tier(hvol)
    dte_factor = np.sqrt(dte / 365)
    return hvol * dte_factor * strike_std * 100


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