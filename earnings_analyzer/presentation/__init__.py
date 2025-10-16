"""Presentation layer - all display and formatting logic"""

from .formatters import (
    format_trend,
    format_breaks,
    format_break_ratio,
    format_iv_value,
    format_iv_with_dte,
    clean_pattern,
    format_drift
)

from .tables import (
    print_results_table,
    print_fetch_summary
)

from .insights import (
    print_insights
)

__all__ = [
    # Formatters
    'format_trend',
    'format_breaks',
    'format_break_ratio',
    'format_iv_value',
    'format_iv_with_dte',
    'clean_pattern',
    'format_drift',
    
    # Tables
    'print_results_table',
    'print_fetch_summary',
    
    # Insights
    'print_insights',
]