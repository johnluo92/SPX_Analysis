# ============================================================================
# FILE: config.py
# ============================================================================

"""Configuration and constants"""

ALPHAVANTAGE_KEYS = [
    "HPCFVLGHWHQU0QTY",
    "VL7Z4WRK8T5MJPK5",
    "DYU6F4AG3IL03321",
    "EXMUX4OSACRK51NZ"
]

CACHE_FILE = "cache/earnings_cache.json"
RATE_LIMIT_FILE = "cache/rate_limits.json"
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

CONTAINMENT_THRESHOLD = 69.5
BREAK_RATIO_THRESHOLD = 2.0
UPWARD_BIAS_THRESHOLD = 65
DOWNWARD_BIAS_THRESHOLD = 35
BREAK_BIAS_THRESHOLD = 70
DRIFT_THRESHOLD = 3.0

IV_TARGET_DTE = 45
IV_PREMIUM_ELEVATED = 15
IV_PREMIUM_DEPRESSED = -15

RATE_LIMIT_HOURS = 24
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.5