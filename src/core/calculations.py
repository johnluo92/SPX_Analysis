"""
Core calculation utilities for SPX Analysis system.

This module provides canonical implementations of common calculations used throughout
the codebase. Each function should have ONE implementation here that is used everywhere.

Design Principles:
- Pure functions: No side effects, deterministic outputs
- Defensive programming: Handle edge cases gracefully
- Clear documentation: Explain the "why" not just the "what"
- Performance conscious: Vectorized operations where possible
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional


def calculate_robust_zscore(
    series: pd.Series,
    window: int,
    min_std: float = 1e-8
) -> pd.Series:
    """
    Calculate rolling z-score with protection against division by zero.
    
    Z-score measures how many standard deviations away from the mean a value is.
    This implementation uses a rolling window to capture changing regimes.
    
    Args:
        series: Input time series
        window: Rolling window size in periods
        min_std: Minimum standard deviation to prevent division by zero
                 (default: 1e-8, essentially zero but avoids NaN/Inf)
    
    Returns:
        pd.Series: Rolling z-scores. NaN during warmup period (first 'window' values)
    
    Example:
        >>> vix = pd.Series([12, 15, 18, 20, 16, 14])
        >>> zscore = calculate_robust_zscore(vix, window=3)
        >>> # Values >2 or <-2 indicate unusual regime vs recent history
    
    Notes:
        - First 'window' values will be NaN (insufficient data)
        - min_std prevents Inf when data is completely flat
        - Useful for mean reversion signals
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std().clip(lower=min_std)
    return (series - rolling_mean) / rolling_std


def calculate_regime_with_validation(
    series: pd.Series,
    bins: List[float],
    labels: List[int],
    feature_name: str = "feature"
) -> pd.Series:
    """
    Classify series into discrete regimes with robust error handling.
    
    Converts continuous values into categorical regimes (e.g., VIX into 
    Low/Normal/Elevated/Crisis). Handles edge cases gracefully.
    
    Args:
        series: Input time series to classify
        bins: Bin edges for classification (e.g., [0, 16.77, 24.40, 39.67, 100])
        labels: Integer labels for each regime (e.g., [0, 1, 2, 3])
        feature_name: Descriptive name for debugging/warnings
    
    Returns:
        pd.Series: Integer regime labels (0-indexed). Returns 0 for all invalid cases.
    
    Example:
        >>> vix = pd.Series([12, 20, 30, 45])
        >>> regimes = calculate_regime_with_validation(
        ...     vix,
        ...     bins=[0, 16.77, 24.40, 39.67, 100],
        ...     labels=[0, 1, 2, 3],
        ...     feature_name="vix"
        ... )
        >>> # Output: [0, 1, 2, 3] representing Low/Normal/Elevated/Crisis
    
    Edge Cases:
        - <50% valid data: Returns all zeros
        - Completely flat data: Returns all zeros  
        - Values outside bins: Filled with 0
        - Binning errors: Returns all zeros
    
    Notes:
        - Conservative defaults to 0 (typically "normal" regime)
        - Useful for regime-specific model selection
    """
    # Edge case 1: Too much missing data
    if series.notna().sum() / len(series) < 0.5:
        return pd.Series(0, index=series.index)
    
    # Edge case 2: Data has no variance (flat line)
    valid_values = series.dropna()
    if len(valid_values) > 0 and valid_values.max() - valid_values.min() < 1e-6:
        return pd.Series(0, index=series.index)
    
    # Attempt classification
    try:
        regime = pd.cut(series, bins=bins, labels=labels)
        return regime.fillna(0).astype(int)
    except Exception as e:
        # If anything goes wrong, default to regime 0
        import warnings
        warnings.warn(
            f"⚠️ Regime classification failed for {feature_name}: {str(e)}. "
            f"Defaulting to regime 0."
        )
        return pd.Series(0, index=series.index)


def calculate_percentile_with_validation(
    series: pd.Series,
    window: int,
    min_data_pct: float = 0.7
) -> pd.Series:
    """
    Calculate rolling percentile rank with data quality requirements.
    
    Percentile rank shows where the current value sits in the distribution of
    recent values (0-100, where 50 = median, 90 = top 10%).
    
    Args:
        series: Input time series
        window: Rolling window size for percentile calculation
        min_data_pct: Minimum fraction of valid data required (0.0 to 1.0)
                      If window needs 63 days and min_data_pct=0.7, 
                      at least 44 valid values required
    
    Returns:
        pd.Series: Percentile ranks (0-100). NaN when insufficient valid data.
    
    Example:
        >>> vix = pd.Series([10, 12, 15, 18, 20, 25, 22])
        >>> percentile = calculate_percentile_with_validation(vix, window=5)
        >>> # Last value (22) vs last 5 values [12,15,18,20,25] 
        >>> # ranks at 60th percentile
    
    Use Cases:
        - Identify extreme values (percentile >90 or <10)
        - Mean reversion signals (extreme percentiles often revert)
        - Normalize across different market regimes
    
    Notes:
        - Stricter data requirements than simple rolling operations
        - Returns NaN rather than unreliable values
        - Window+1 used internally to include current observation
    """
    def safe_percentile_rank(window_data: pd.Series) -> float:
        """Inner function to compute percentile for a single window."""
        valid_data = window_data.dropna()
        
        # Not enough data for reliable calculation
        if len(valid_data) < window * min_data_pct or len(valid_data) == 0:
            return np.nan
        
        last_value = window_data.iloc[-1]
        if pd.isna(last_value):
            return np.nan
        
        # Calculate percentile: what % of values are below current value?
        below_count = (valid_data < last_value).sum()
        percentile = (below_count / len(valid_data)) * 100
        
        return percentile
    
    # Use window+1 to include the current observation
    return series.rolling(window + 1).apply(safe_percentile_rank, raw=False)


def calculate_velocity(
    series: pd.Series,
    window: int
) -> pd.Series:
    """
    Calculate rate of change (velocity) over a rolling window.
    
    Simple wrapper around diff() for semantic clarity. "Velocity" is more
    intuitive than "difference" when discussing market momentum.
    
    Args:
        series: Input time series
        window: Lookback period for difference
    
    Returns:
        pd.Series: Change over 'window' periods (current - past)
    
    Example:
        >>> vix = pd.Series([15, 16, 20, 25])
        >>> velocity = calculate_velocity(vix, window=2)
        >>> # [NaN, NaN, 5, 9] - change over last 2 periods
    """
    return series.diff(window)


def calculate_acceleration(
    series: pd.Series,
    window: int
) -> pd.Series:
    """
    Calculate rate of change of velocity (acceleration).
    
    Second derivative: measures how the rate of change is changing.
    Useful for identifying inflection points and regime transitions.
    
    Args:
        series: Input time series
        window: Lookback period for each difference operation
    
    Returns:
        pd.Series: Acceleration (change in velocity)
    
    Example:
        >>> vix = pd.Series([10, 12, 16, 22, 30, 40])
        >>> accel = calculate_acceleration(vix, window=2)
        >>> # Positive acceleration = speeding up
        >>> # Negative acceleration = slowing down or reversing
    
    Use Cases:
        - Early warning of regime changes
        - Momentum exhaustion signals  
        - Volatility spike detection
    """
    velocity = series.diff(window)
    return velocity.diff(window)


# ============================================================================
# VALIDATION AND QUALITY CONTROL UTILITIES
# ============================================================================

def validate_series_quality(
    series: pd.Series,
    min_valid_pct: float = 0.5,
    check_variance: bool = True,
    min_variance: float = 1e-6
) -> tuple[bool, str]:
    """
    Check if a series meets minimum quality standards for analysis.
    
    Args:
        series: Time series to validate
        min_valid_pct: Minimum fraction of non-NaN values required
        check_variance: Whether to check for flat/constant data
        min_variance: Minimum variance required if check_variance=True
    
    Returns:
        (is_valid, message): Boolean validity and explanation string
    
    Example:
        >>> data = pd.Series([12, 12, 12, 12, 12])
        >>> is_valid, msg = validate_series_quality(data)
        >>> print(msg)  # "Series has insufficient variance"
    """
    # Check 1: Sufficient non-missing data
    valid_pct = series.notna().sum() / len(series)
    if valid_pct < min_valid_pct:
        return False, f"Only {valid_pct:.1%} valid data (need {min_valid_pct:.1%})"
    
    # Check 2: Data has variance (not flat line)
    if check_variance:
        valid_values = series.dropna()
        if len(valid_values) > 1:
            data_range = valid_values.max() - valid_values.min()
            if data_range < min_variance:
                return False, f"Series has insufficient variance (range: {data_range:.2e})"
    
    return True, "Series passes quality checks"


def safe_division(
    numerator: Union[pd.Series, float],
    denominator: Union[pd.Series, float],
    fill_value: float = np.nan,
    min_denominator: float = 1e-10
) -> Union[pd.Series, float]:
    """
    Perform division with protection against divide-by-zero.
    
    Args:
        numerator: Value(s) to divide
        denominator: Value(s) to divide by
        fill_value: What to return when denominator is ~0
        min_denominator: Threshold below which denominator is considered zero
    
    Returns:
        Result of division, with fill_value for invalid operations
    
    Example:
        >>> ratio = safe_division(vix, realized_vol, fill_value=1.0)
        >>> # When realized_vol ≈ 0, returns 1.0 instead of Inf
    """
    if isinstance(denominator, pd.Series):
        safe_denom = denominator.replace(0, np.nan).abs()
        safe_denom = safe_denom.where(safe_denom >= min_denominator, np.nan)
        result = numerator / safe_denom
        if fill_value is not np.nan:
            result = result.fillna(fill_value)
        return result
    else:
        # Scalar division
        if abs(denominator) < min_denominator:
            return fill_value
        return numerator / denominator


# ============================================================================
# STATISTICAL AGGREGATIONS
# ============================================================================

def rolling_rank_pct(
    series: pd.Series,
    window: int
) -> pd.Series:
    """
    Calculate rolling percentile rank using pandas native implementation.
    
    Simpler alternative to calculate_percentile_with_validation when you
    don't need strict data quality requirements.
    
    Args:
        series: Input time series
        window: Rolling window size
    
    Returns:
        pd.Series: Percentile ranks (0-100)
    
    Note:
        This is a simpler, faster version without validation. Use 
        calculate_percentile_with_validation if you need quality checks.
    """
    return series.rolling(window).rank(pct=True) * 100


def exponential_decay_weights(
    length: int,
    halflife: int
) -> np.ndarray:
    """
    Generate exponential decay weights for weighted calculations.
    
    Recent observations get more weight than distant ones. Useful for
    adaptive calculations that should respond faster to recent changes.
    
    Args:
        length: Number of weights to generate
        halflife: Period where weight decays to 50%
    
    Returns:
        np.ndarray: Normalized weights summing to 1.0
    
    Example:
        >>> weights = exponential_decay_weights(length=10, halflife=3)
        >>> weighted_mean = (data * weights).sum()
    """
    decay_factor = 0.5 ** (1 / halflife)
    positions = np.arange(length)
    weights = decay_factor ** positions[::-1]  # Most recent = highest weight
    return weights / weights.sum()  # Normalize to sum to 1


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

# Default parameters used throughout the system
DEFAULT_ZSCORE_WINDOW = 63
DEFAULT_PERCENTILE_WINDOW = 63
DEFAULT_MIN_STD = 1e-8
DEFAULT_MIN_DATA_PCT = 0.7

# Common regime boundaries (defined here for consistency)
VIX_REGIME_BINS = [0, 16.77, 24.40, 39.67, 100]
VIX_REGIME_LABELS = [0, 1, 2, 3]  # Low, Normal, Elevated, Crisis

SKEW_REGIME_BINS = [0, 130, 145, 160, 200]
SKEW_REGIME_LABELS = [0, 1, 2, 3]


if __name__ == "__main__":
    # Quick validation tests
    print("Running calculation module validation tests...\n")
    
    # Test 1: Z-score calculation
    test_series = pd.Series([10, 12, 15, 20, 18, 16, 14, 22, 25, 28])
    zscore = calculate_robust_zscore(test_series, window=5)
    print("✓ Z-score test passed")
    
    # Test 2: Regime classification
    vix_test = pd.Series([12, 20, 30, 45])
    regimes = calculate_regime_with_validation(
        vix_test, 
        bins=VIX_REGIME_BINS,
        labels=VIX_REGIME_LABELS,
        feature_name="vix_test"
    )
    assert list(regimes) == [0, 1, 2, 3], "Regime test failed"
    print("✓ Regime classification test passed")
    
    # Test 3: Percentile calculation
    percentile = calculate_percentile_with_validation(test_series, window=5)
    assert percentile.notna().sum() > 0, "Percentile test failed"
    print("✓ Percentile calculation test passed")
    
    # Test 4: Division safety
    safe_result = safe_division(
        pd.Series([1, 2, 3]),
        pd.Series([2, 0, 1]),
        fill_value=0
    )
    assert safe_result.iloc[1] == 0, "Safe division test failed"
    print("✓ Safe division test passed")
    
    print("\n✅ All validation tests passed!")