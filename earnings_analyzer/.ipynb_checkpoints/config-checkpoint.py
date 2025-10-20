# ============================================================================
# FILE: config.py
# ============================================================================

"""Configuration and constants"""

# Removed ALPHAVANTAGE_KEYS - no longer needed

CACHE_FILE = "cache/earnings_cache.json"
IV_CACHE_FILE = "cache/iv_cache.json"

DEFAULT_LOOKBACK_QUARTERS = 24
MIN_QUARTERS_REQUIRED = 10
HVOL_LOOKBACK_DAYS = 30

VOLATILITY_TIERS = [
    (25, 1.0),
    (35, 1.2),
    (45, 1.4),
    (float('inf'), 1.5)
]

# Strategy thresholds
CONTAINMENT_THRESHOLD = 69.5
MIN_QUARTERS_REQUIRED = 12
UPWARD_BIAS_THRESHOLD = 70    # % of closes above entry for upward signal
DOWNWARD_BIAS_THRESHOLD = 30  # % of closes above entry for downward signal
DRIFT_THRESHOLD = 5.0
BREAK_BIAS_THRESHOLD = 70
BREAK_RATIO_THRESHOLD = 2.0
MIN_BREAKS_FOR_SIGNAL = 8     # Minimum total breaks needed for directional signal

IV_TARGET_DTE = 45
IV_PREMIUM_ELEVATED = 15
IV_PREMIUM_DEPRESSED = -15

REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.5