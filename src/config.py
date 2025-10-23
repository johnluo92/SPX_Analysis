"""
SPX Prediction System - Configuration
All constants and parameters for SPX forecasting
"""

# ============================================================================
# MACRO FACTORS (Yahoo Finance Tickers)
# ============================================================================

MACRO_TICKERS = {
    'GLD': 'Gold',
    'CL=F': 'Crude Oil',
    'DX-Y.NYB': 'Dollar',
    '^TNX': '10Y Treasury',
    '^FVX': '5Y Treasury',
}

# ============================================================================
# FRED ECONOMIC SERIES
# ============================================================================

FRED_SERIES = {
    'T10YIE': '10Y Breakeven Inflation',
    'T5YIFR': '5Y Forward Inflation',
    'T10Y2Y': '10Y-2Y Yield Spread',
    'VIXCLS': 'VIX Close'
}

FRED_API_KEY_PATH = 'config.json'

# ============================================================================
# SPX PREDICTION PARAMETERS
# ============================================================================

# Fibonacci-based prediction horizons (trading days)
SPX_FORWARD_WINDOWS = [8, 13, 21, 34]

# Range prediction thresholds (±2%, ±3%, ±5%, ±8%, ±13%)
SPX_RANGE_THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.13]

# ============================================================================
# FEATURE ENGINEERING WINDOWS
# ============================================================================

VIX_WINDOWS = [5, 21]           # VIX change windows
FRED_WINDOWS = [21, 63]         # Economic indicator windows
MACRO_WINDOWS = [21, 63]        # Macro momentum windows

# ============================================================================
# MODEL TRAINING PARAMETERS
# ============================================================================

TEST_SPLIT = 0.1                # 10% holdout for testing
RANDOM_STATE = 42               # Reproducibility seed

# RandomForest hyperparameters (optimized for SPX)
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'min_samples_split': 50,
    'min_samples_leaf': 30,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# ============================================================================
# DATA COLLECTION PARAMETERS
# ============================================================================

LOOKBACK_YEARS = 7              # Historical data window
CACHE_DIR = '.cache_sector_data'  # Directory for cached data