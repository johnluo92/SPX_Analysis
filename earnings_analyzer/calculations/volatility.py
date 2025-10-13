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


def calculate_rvol_tier(historical_data: List[Dict], dte: int, current_hvol: float, 
                       current_index: int = None) -> float:
    """
    Calculate strike width tier based on realized volatility from past earnings
    
    Uses leave-one-out methodology: calculates tier from historical RVol
    WITHOUT including the current period being evaluated.
    
    CRITICAL FIX: Now properly excludes current period via current_index parameter
    
    Args:
        historical_data: List of dicts with 'rvol' key from past earnings
        dte: Target days to expiration (45 or 90)
        current_hvol: Current period's HVol (fallback if no history)
        current_index: Index of current period to EXCLUDE (leave-one-out)
    
    Returns:
        Tier multiplier for strike width calculation
    """
    if not historical_data:
        # First period: fallback to HVol-based tier
        return get_volatility_tier(current_hvol)
    
    # Extract RVol values, EXCLUDING current period if specified
    if current_index is not None:
        # Leave-one-out: exclude the period we're evaluating
        rvols = [
            d['rvol'] for i, d in enumerate(historical_data) 
            if 'rvol' in d and i != current_index
        ]
    else:
        # If no index specified, use all historical data
        # This happens when calculating for future (not backtest)
        rvols = [d['rvol'] for d in historical_data if 'rvol' in d]
    
    if not rvols or len(rvols) < 3:
        # Insufficient historical data: fallback to HVol
        # Require at least 3 past periods for robust estimation
        return get_volatility_tier(current_hvol)
    
    # Calculate average RVol from historical data (excluding current)
    avg_rvol = np.mean(rvols)
    
    # Map avg RVol to tier using same thresholds as HVol
    # This creates consistent tier logic while using realized data
    for threshold, multiplier in VOLATILITY_TIERS:
        if avg_rvol < threshold:
            return multiplier
    
    return VOLATILITY_TIERS[-1][1]


def calculate_strike_width(hvol: float, dte: int, tier_multiplier: float = None) -> float:
    """
    Calculate strike width for given DTE
    
    Args:
        hvol: Historical volatility (annualized)
        dte: Days to expiration
        tier_multiplier: Optional tier override, otherwise uses hvol-based tier
    
    Returns:
        Strike width as percentage
    """
    if tier_multiplier is None:
        tier_multiplier = get_volatility_tier(hvol)
    
    dte_factor = np.sqrt(dte / 365)
    return hvol * dte_factor * tier_multiplier * 100


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
    """
    Get entry price based on earnings timing
    
    CRITICAL: This determines when we "enter" the position and calculate strike width
    - BMO (before market open): Use close price from day BEFORE earnings
    - AMC (after market close): Use close price from earnings day itself
    
    Args:
        price_data: DataFrame with price history
        earnings_date: Date of earnings announcement
        timing: 'bmo' or 'amc'
    
    Returns:
        Tuple of (reference_price, reference_date)
        reference_date is the actual date used for width calculation (for audit trail)
    """
    # Determine target date based on timing
    if timing.lower() == 'bmo':
        # BMO: We enter after close on day before earnings
        target_date = earnings_date - timedelta(days=1)
    else:
        # AMC: We enter after close on earnings day
        target_date = earnings_date
    
    # Find nearest trading day to target
    price, actual_date = find_nearest_price(price_data, target_date)
    
    # actual_date is now our reference_date for audit trail
    return price, actual_date


def calculate_realized_volatility(price_data: pd.DataFrame, start_date: datetime, 
                                  end_date: datetime) -> Optional[float]:
    """
    Calculate realized volatility over a specific period
    
    Args:
        price_data: DataFrame with price history
        start_date: Start of measurement period
        end_date: End of measurement period
    
    Returns:
        Annualized realized volatility as decimal (e.g., 0.25 = 25%)
    """
    window = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
    
    if len(window) < 10:
        return None
    
    returns = window['close'].pct_change().dropna()
    
    if len(returns) < 5:
        return None
    
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    
    return annual_vol