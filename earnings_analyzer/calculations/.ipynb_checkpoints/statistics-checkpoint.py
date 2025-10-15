"""Statistical calculations for containment and directional analysis"""
import numpy as np
from typing import List, Dict


def calculate_stats(data: List[Dict]) -> Dict:
    """
    Calculate containment and directional statistics
    
    Args:
        data: List of dicts with 'move' and 'width' keys
    
    Returns:
        Dictionary with containment, bias, and drift statistics
    """
    total = len(data)
    moves = np.array([d['move'] for d in data])
    widths = np.array([d['width'] for d in data])
    
    # Containment and breaks
    stays_within = sum(1 for i, m in enumerate(moves) if abs(m) <= widths[i])
    breaks_up = sum(1 for i, m in enumerate(moves) if m > widths[i])
    breaks_down = sum(1 for i, m in enumerate(moves) if m < -widths[i])
    
    # Trend direction: % of times closed ABOVE entry
    up_moves = sum(1 for m in moves if m > 0)
    trend_pct = (up_moves / total) * 100
    
    # Break direction: % of breaks that went UP
    total_breaks = breaks_up + breaks_down
    if total_breaks > 0:
        break_up_pct = (breaks_up / total_breaks) * 100
    else:
        break_up_pct = 50.0  # No breaks = neutral
    
    # Drift: average move magnitude
    avg_move = np.mean(moves)
    avg_move_vs_width = (avg_move / np.mean(widths)) * 100 if np.mean(widths) > 0 else 0
    
    return {
        'total': total,
        'containment': (stays_within / total) * 100,
        'breaks_up': breaks_up,
        'breaks_down': breaks_down,
        'trend_pct': trend_pct,           # RENAMED from 'overall_bias'
        'break_up_pct': break_up_pct,     # RENAMED from 'break_bias'
        'drift_pct': avg_move,             # RENAMED from 'avg_move_pct'
        'drift_vs_width': avg_move_vs_width,
        'avg_width': np.mean(widths)
    }