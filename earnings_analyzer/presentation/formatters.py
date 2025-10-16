"""Display formatting utilities

Extracted from batch.py to separate presentation from business logic
"""
from typing import List, Optional
import pandas as pd


def format_trend(trend_pct: float) -> str:
    """Format trend direction with percentage"""
    if trend_pct >= 66.7:
        return f"↑{trend_pct:.0f}%"
    elif trend_pct <= 33.3:
        down_pct = 100 - trend_pct
        return f"↓{down_pct:.0f}%"
    else:
        # Moderate - show whichever direction is majority
        if trend_pct >= 50:
            return f"↑{trend_pct:.0f}%"
        else:
            down_pct = 100 - trend_pct
            return f"↓{down_pct:.0f}%"


def format_breaks(up: int, down: int) -> str:
    """Format break ratio with directional indicator if strong"""
    if up == 0 and down == 0:
        return "0:0"
    
    total = up + down
    up_pct = (up / total) * 100
    
    if up > down and up_pct >= 60:
        return f"{up}:{down}↑({up_pct:.0f}%)"
    elif down > up and (100 - up_pct) >= 60:
        return f"{up}:{down}↓({100-up_pct:.0f}%)"
    else:
        return f"{up}:{down}({up_pct:.0f}%)"


def format_break_ratio(up_breaks: int, down_breaks: int, break_bias: float) -> str:
    """Legacy format for break ratio (kept for backward compatibility)"""
    if up_breaks == 0 and down_breaks == 0:
        return "0:0"
    
    if break_bias >= 66.7:
        return f"{up_breaks}:{down_breaks}↑"
    elif break_bias <= 33.3:
        return f"{up_breaks}:{down_breaks}↓"
    else:
        return f"{up_breaks}:{down_breaks}"


def format_iv_value(value) -> str:
    """Format IV values, showing N/A for missing data"""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def format_iv_with_dte(iv, dte) -> str:
    """Format IV with DTE, showing N/A for missing data"""
    if iv is None or pd.isna(iv):
        return "N/A"
    if dte is None or pd.isna(dte):
        return f"{iv:.1f}%"
    return f"{iv:.1f}% ({int(dte)}DTE)"


def clean_pattern(pattern: str) -> str:
    """Clean pattern string for display"""
    if pattern == "SKIP":
        return "-"
    if "(" in pattern:
        pattern = pattern.split("(")[0].strip()
    return pattern.replace("BIAS↑", "Bias↑").replace("BIAS↓", "Bias↓").replace("⚠ ", "⚠️")


def format_drift(drift: float) -> str:
    """Format drift percentage"""
    return f"{drift:+.1f}%"