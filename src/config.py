"""
Configuration - All constants in one place.
Twin Pillars: Simplicity & Consistency
"""

# ============================================================================
# SECTOR DEFINITIONS
# ============================================================================

SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
    'XLB': 'Materials'
}

# ============================================================================
# MACRO FACTORS (Yahoo Finance)
# ============================================================================

MACRO_TICKERS = {
    'GLD': 'Gold',
    'CL=F': 'Crude Oil',
    'DX-Y.NYB': 'Dollar',
    '^TNX': '10Y Treasury',
    '^FVX': '5Y Treasury',
}

# ============================================================================
# FRED SERIES (Economic Data)
# ============================================================================

FRED_SERIES = {
    'T10YIE': '10Y Breakeven Inflation',
    'T5YIFR': '5Y Forward Inflation',
    'T10Y2Y': '10Y-2Y Yield Spread',
    'VIXCLS': 'VIX Close'
}

FRED_API_KEY_PATH = 'config.json'

# ============================================================================
# SPX PREDICTION DATA SOURCES (MINIMAL)
# ============================================================================

SPX_INDICATORS = {
    '^VIX': 'VIX',
}

SENTIMENT_LOOKBACK = [5, 10, 21]

# ============================================================================
# SECTOR CATEGORIZATION
# ============================================================================

SECTOR_CATEGORIES = {
    'XLF': 'FINANCIALS',
    'XLE': 'MACRO_SENSITIVE',
    'XLP': 'MACRO_SENSITIVE',
    'XLRE': 'MACRO_SENSITIVE',
    'XLB': 'MACRO_SENSITIVE',
    'XLY': 'SENTIMENT_DRIVEN',
    'XLC': 'SENTIMENT_DRIVEN',
    'XLV': 'SENTIMENT_DRIVEN',
    'XLI': 'SENTIMENT_DRIVEN',
    'XLK': 'MIXED',
    'XLU': 'MIXED',
}

# Sectors that benefit from longer lookback windows
LONG_HORIZON_SECTORS = ['XLF', 'XLE', 'XLRE', 'XLB', 'XLP', 'XLU']

# ============================================================================
# HYPERPARAMETERS BY CATEGORY
# ============================================================================

HYPERPARAMETERS = {
    'FINANCIALS': {
        'n_estimators': 150,
        'max_depth': 5,
        'min_samples_split': 40,
        'min_samples_leaf': 35,
        'max_features': 'sqrt',
    },
    'MACRO_SENSITIVE': {
        'n_estimators': 150,
        'max_depth': 5,
        'min_samples_split': 40,
        'min_samples_leaf': 30,
        'max_features': 'sqrt',
    },
    'SENTIMENT_DRIVEN': {
        'n_estimators': 150,
        'max_depth': 4,
        'min_samples_split': 50,
        'min_samples_leaf': 50,
        'max_features': 'sqrt',
    },
    'MIXED': {
        'n_estimators': 150,
        'max_depth': 4,
        'min_samples_split': 45,
        'min_samples_leaf': 40,
        'max_features': 'sqrt',
    }
}

# ============================================================================
# FEATURE ENGINEERING SPECS
# ============================================================================

# Windows for different feature types
RS_WINDOWS_SHORT = [21, 63]
RS_WINDOWS_LONG = [21, 63, 126]
MACRO_WINDOWS = [21, 63]
VIX_WINDOWS = [5, 21]
FRED_WINDOWS = [21, 63]

# Feature importance threshold
FEATURE_IMPORTANCE_THRESHOLD = 0.008

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

FORWARD_WINDOW = 21  # Days ahead to predict
TEST_SPLIT = 0.2
WALK_FORWARD_SPLITS = 5
RANDOM_STATE = 42

# SPX Prediction Model Parameters
SPX_FORWARD_WINDOWS = [7, 14, 21]  # Multiple prediction horizons
SPX_RANGE_THRESHOLDS = [0.02, 0.03, 0.05]  # ±2%, ±3%, ±5% range prediction

# ============================================================================
# DATA PARAMETERS
# ============================================================================

LOOKBACK_YEARS = 7
CACHE_DIR = '.cache_sector_data'