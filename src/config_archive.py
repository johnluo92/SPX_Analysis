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
    'XLF': 'Financials',          # Cyclical – tied to interest rates, lending, and credit cycle
    'XLE': 'Energy',              # Cyclical – sensitive to oil prices and global demand
    'XLB': 'Materials',           # Cyclical – driven by industrial and construction demand
    'XLI': 'Industrials',         # Cyclical – benefits from economic expansion and capex cycles
    'XLY': 'Consumer Discretionary',  # Cyclical – driven by consumer confidence and income growth

    'XLK': 'Information Technology',  # Growth – secular innovation, sentiment-driven
    'XLC': 'Communication Services',  # Growth – mix of tech and media, sentiment-driven

    'XLP': 'Consumer Staples',    # Defensive – stable demand, less economic sensitivity
    'XLV': 'Health Care',         # Defensive – steady earnings, resilient in downturns
    'XLU': 'Utilities',           # Defensive – rate-sensitive, essential services
    'XLRE': 'Real Estate',        # Defensive – income-oriented, interest-rate sensitive
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
TEST_SPLIT = 0.1
WALK_FORWARD_SPLITS = 5
RANDOM_STATE = 42

# SPX Prediction Model Parameters
SPX_FORWARD_WINDOWS = [8, 13, 21, 34]  # Multiple prediction horizons [these are trading days]
SPX_RANGE_THRESHOLDS = [0.02, 0.03, 0.05, .08, .13]  # ±2%, ±3%, ±5% range prediction

# ============================================================================
# DATA PARAMETERS
# ============================================================================

LOOKBACK_YEARS = 7
CACHE_DIR = '.cache_sector_data'