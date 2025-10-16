# ============================================================================
# FILE 1: earnings_analyzer/__init__.py
# REPLACE ENTIRE FILE WITH THIS
# ============================================================================
"""Earnings Containment Analyzer - Post-earnings movement analysis tool

Version 3.0 - Refactored with typed models and clean architecture
"""

__version__ = '3.0'

# Main analysis functions (V2 is now the default)
from .analysis.single_v2 import analyze_ticker_v2 as analyze_ticker
from .analysis.batch_v2 import batch_analyze_v2 as batch_analyze

# Core models
from .core.models import AnalysisResult, TimeframeStats, results_to_dataframe

# Output utilities
from .output import export_to_csv, export_to_json

__all__ = [
    # Analysis
    'analyze_ticker',
    'batch_analyze',
    
    # Models
    'AnalysisResult',
    'TimeframeStats',
    'results_to_dataframe',
    
    # Output
    'export_to_csv',
    'export_to_json',
]