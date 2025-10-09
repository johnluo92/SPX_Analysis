"""Display formatting helpers"""
import pandas as pd


def format_break_ratio(up_breaks: int, down_breaks: int, break_bias: float) -> str:
    """Format break ratio with directional arrow if edge exists"""
    if up_breaks == 0 and down_breaks == 0:
        return "0:0"
    
    if break_bias >= 66.7:
        return f"{up_breaks}:{down_breaks}↑"
    elif break_bias <= 33.3:
        return f"{up_breaks}:{down_breaks}↓"
    else:
        return f"{up_breaks}:{down_breaks}"


def format_strategy_display(strategy: str) -> str:
    """Shorten strategy string for display"""
    return strategy


def format_iv_value(iv_value) -> str:
    """Format IV value for display"""
    if pd.notna(iv_value):
        return f"{int(iv_value)}"
    return "N/A"


def format_iv_premium(iv_premium) -> str:
    """Format IV premium for display"""
    if pd.notna(iv_premium):
        return f"{iv_premium:+.0f}%"
    return "N/A"


def format_drift(drift: float) -> str:
    """Format drift percentage for display"""
    return f"{drift:+.1f}%"