"""Volatility calculations - Simplified to use HVol directly without tier bucketing"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple

from ..config import HVOL_LOOKBACK_DAYS


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


def calculate_strike_width(hvol: float, dte: int, multiplier: float = 1.0) -> float:
    """
    Calculate strike width using direct HVol - NO TIER BUCKETING
    
    This creates natural variation across tickers based on their actual volatility.
    
    Formula: width = hvol * sqrt(dte/365) * multiplier * 100
    
    Args:
        hvol: Historical volatility (annualized, as decimal)
        dte: Days to expiration
        multiplier: Optional adjustment (default 1.0 = 1 standard deviation)
    
    Returns:
        Strike width as percentage
    """
    dte_factor = np.sqrt(dte / 365)
    return hvol * dte_factor * multiplier * 100


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
    
    - BMO (before market open): Use close price from day BEFORE earnings
    - AMC (after market close): Use close price from earnings day itself
    
    Args:
        price_data: DataFrame with price history
        earnings_date: Date of earnings announcement
        timing: 'bmo' or 'amc'
    
    Returns:
        Tuple of (reference_price, reference_date)
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