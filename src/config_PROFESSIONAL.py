"""
Configuration for VIX-based SPX options backtesting engine.
PROFESSIONAL PARAMETERS based on industry best practices.

Philosophy:
- Premium collection through systematic short options
- VIX regimes dictate position sizing and structure
- No predictions, just historical probabilities
- Surgical, repeatable, mechanical
"""

from typing import Dict, List
from datetime import datetime

# ==================== DATA CONFIGURATION ====================

DATA_CONFIG = {
    'spx_ticker': '^GSPC',
    'vix_ticker': '^VIX',
    'vix_source': 'yahoo',  # 'yahoo' or 'fred' (FRED = VIXCLS)
    
    # Default date range for backtesting
    'default_start': '2020-01-01',
    'default_end': datetime.now().strftime('%Y-%m-%d'),
    
    # Data alignment
    'alignment_method': 'inner',  # Drop non-matching dates
    'forward_fill_weekends': True,
}

# ==================== VIX REGIME DEFINITIONS ====================

VIX_REGIMES = {
    'LOW': {
        'range': (0, 15),
        'action': 'STAND_ASIDE',
        'description': 'Too cheap - don\'t sell premium',
        'position_size_multiplier': 0.0,  # No trades
    },
    'NORMAL': {
        'range': (15, 25),
        'action': 'TRADE',
        'description': 'Normal volatility - full position sizing',
        'position_size_multiplier': 1.0,
    },
    'ELEVATED': {
        'range': (25, 100),
        'action': 'SELECTIVE',  # CHANGED: Not "aggressive"
        'description': 'High premiums but dangerous - be selective',
        'position_size_multiplier': 0.8,  # CHANGED: Size DOWN in high vol
    }
}

def get_vix_regime(vix_level: float) -> str:
    """Determine VIX regime from current VIX level."""
    for regime, config in VIX_REGIMES.items():
        low, high = config['range']
        if low <= vix_level < high:
            return regime
    return 'ELEVATED'

# ==================== TRADE STRUCTURE DEFINITIONS ====================

TRADE_STRUCTURES = {
    'BULL_PUT_SPREAD': {
        'vix_range': (15, 30),
        'description': 'Single-sided put credit spread',
        'short_delta': 15,  # CHANGED: 15-delta for better credit
        'wing_width': 25,   # CHANGED: $25 wide
    },
    'IRON_CONDOR': {
        'vix_range': (30, 100),
        'description': 'Both sides - use only when vol extremely high',
        'short_delta': 10,
        'wing_width': 25,
    }
}

def get_trade_structure(vix_level: float) -> str:
    """Determine appropriate trade structure based on VIX."""
    if vix_level < 30:
        return 'BULL_PUT_SPREAD'
    else:
        return 'IRON_CONDOR'  # Only in extreme vol

# ==================== POSITION PARAMETERS ====================

POSITION_CONFIG = {
    # PROFESSIONAL PARAMETERS
    'dte': 14,  # CHANGED: 14 DTE for better credit and time decay
    'max_concurrent_positions': 3,
    
    # Strike placement - CONSERVATIVE
    'strike_method': 'std_dev',
    'std_dev_short': 0.85,  # CHANGED: ~80% probability of success
    'std_dev_elevated_multiplier': 1.2,  # Go wider in high vol
    'wing_width_dollars': 25,  # CHANGED: Wider spreads for better credit
    
    # Position sizing
    'account_size': 50000,
    'max_risk_per_trade': 0.02,  # 2% of account max risk
    
    # Credit targets
    'min_credit_pct': 0.15,  # Minimum 15% of spread width
    'target_credit_pct': 0.20,  # Target 20% of spread
}

# ==================== ENTRY RULES ====================

ENTRY_RULES = {
    'vix_min': 15,  # Don't trade if VIX below this
    'vix_max': 100,  # No upper limit
    
    # Day-of-week filter
    'avoid_entry_days': [4],  # 0=Mon, 4=Fri
    
    # Position limits
    'max_positions': 3,
    
    # TREND FILTERS - CRITICAL FOR SUCCESS
    'use_trend_filter': True,  # NEW: Enable trend protection
    'no_entry_after_down_day': True,  # NEW: Don't catch falling knives
    'down_day_threshold': 0.015,  # NEW: 1.5% down = skip entry
    'trend_lookback_days': 5,  # NEW: Look at 5-day trend
}

# ==================== EXIT RULES ====================

EXIT_RULES = {
    # PROFESSIONAL EXIT MANAGEMENT
    'hold_to_expiration': False,  # CHANGED: Don't hold to expiry
    
    # Profit taking - CRITICAL
    'profit_target_pct': 0.50,  # NEW: Close at 50% max profit
    'profit_target_enabled': True,  # NEW
    
    # Stop loss
    'stop_loss_multiplier': 2.5,  # CHANGED: Wider stop (2.5x credit)
    'stop_loss_enabled': True,
    
    # Time-based exit
    'close_days_before_expiry': 1,  # NEW: Close 1 day before expiry
    
    # Emergency exits
    'max_loss_per_trade': 0.05,  # 5% of account = hard stop
}

# ==================== RISK PARAMETERS ====================

RISK_CONFIG = {
    'risk_free_rate_source': 'fed_funds',
    'commissions_per_contract': 0.65,
    'slippage_bps': 5,
}

# ==================== VISUALIZATION SETTINGS ====================

VIZ_CONFIG = {
    # Cone visualization
    'cone_std_devs': [1, 2],
    'cone_days_forward': 14,  # CHANGED: Match DTE
    
    # Performance charts
    'show_drawdowns': True,
    'show_win_rate_by_regime': True,
    'show_pnl_distribution': True,
    'show_profit_target_analysis': True,  # NEW
    
    # Output
    'save_plots': True,
    'plot_dpi': 150,
    'plot_style': 'seaborn-v0_8-darkgrid',
}

# ==================== BACKTEST SETTINGS ====================

BACKTEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': None,
    
    # Reporting
    'report_frequency': 'monthly',
    'calculate_sharpe': True,
    'calculate_max_drawdown': True,
    'calculate_calmar_ratio': True,  # NEW
    'benchmark_ticker': '^GSPC',
}

# ==================== OUTPUT PATHS ====================

PATHS = {
    'data_cache': 'data/',
    'results_dir': 'results/',
    'plots_dir': 'results/plots/',
}

# ==================== VALIDATION ====================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # VIX regime ranges must be continuous
    regimes = sorted(VIX_REGIMES.items(), key=lambda x: x[1]['range'][0])
    for i in range(len(regimes) - 1):
        current_high = regimes[i][1]['range'][1]
        next_low = regimes[i + 1][1]['range'][0]
        if current_high != next_low:
            errors.append(f"VIX regime gap: {regimes[i][0]} ends at {current_high}, {regimes[i+1][0]} starts at {next_low}")
    
    # Position sizing
    if POSITION_CONFIG['max_risk_per_trade'] > 0.1:
        errors.append("max_risk_per_trade > 10% is dangerous")
    
    # DTE range
    if not 7 <= POSITION_CONFIG['dte'] <= 45:
        errors.append("DTE should be between 7-45 for professional trading")
    
    # Profit target
    if EXIT_RULES['profit_target_pct'] > 0.8:
        errors.append("profit_target_pct > 80% is too greedy")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

# ==================== QUICK ACCESS FUNCTIONS ====================

def get_config_summary() -> str:
    """Print human-readable config summary."""
    summary = []
    summary.append("=" * 60)
    summary.append("VIX-BASED SPX OPTIONS - PROFESSIONAL CONFIG")
    summary.append("=" * 60)
    summary.append(f"\nVIX Regimes:")
    for regime, config in VIX_REGIMES.items():
        summary.append(f"  {regime}: VIX {config['range'][0]}-{config['range'][1]} → {config['action']}")
    
    summary.append(f"\nPosition Parameters (PROFESSIONAL):")
    summary.append(f"  DTE: {POSITION_CONFIG['dte']} days (optimal for theta decay)")
    summary.append(f"  Strike Placement: {POSITION_CONFIG['std_dev_short']} std dev")
    summary.append(f"  Spread Width: ${POSITION_CONFIG['wing_width_dollars']}")
    summary.append(f"  Target Credit: {POSITION_CONFIG['target_credit_pct']*100}% of width")
    summary.append(f"  Account Size: ${POSITION_CONFIG['account_size']:,}")
    
    summary.append(f"\nEntry Rules:")
    summary.append(f"  Min VIX: {ENTRY_RULES['vix_min']}")
    summary.append(f"  Trend Filter: {'ENABLED' if ENTRY_RULES['use_trend_filter'] else 'DISABLED'}")
    summary.append(f"  Down Day Protection: {'ENABLED' if ENTRY_RULES['no_entry_after_down_day'] else 'DISABLED'}")
    
    summary.append(f"\nExit Rules (PROFESSIONAL):")
    summary.append(f"  Profit Target: {EXIT_RULES['profit_target_pct']*100}% of max profit")
    summary.append(f"  Stop Loss: {EXIT_RULES['stop_loss_multiplier']}x credit")
    summary.append(f"  Hold to Expiration: {EXIT_RULES['hold_to_expiration']}")
    summary.append(f"  Close Before Expiry: {EXIT_RULES['close_days_before_expiry']} day(s)")
    
    summary.append("=" * 60)
    summary.append("\n⚠️  IMPORTANT: This is a PROFESSIONAL configuration")
    summary.append("Based on industry best practices for credit spreads:")
    summary.append("- 14 DTE for optimal theta decay")
    summary.append("- Close at 50% profit (don't be greedy)")
    summary.append("- Wider spreads ($25) for better credit collection")
    summary.append("- Trend filters to avoid selling into crashes")
    summary.append("=" * 60)
    
    return "\n".join(summary)


# ==================== INITIALIZATION ====================

if __name__ == "__main__":
    try:
        validate_config()
        print("✅ Configuration validated successfully")
        print(get_config_summary())
    except ValueError as e:
        print(f"❌ Configuration validation failed:")
        print(str(e))