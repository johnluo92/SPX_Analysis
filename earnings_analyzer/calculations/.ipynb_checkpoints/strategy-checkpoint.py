"""Strategy determination logic"""
from typing import Dict

from ..config import (
    CONTAINMENT_THRESHOLD,
    BREAK_RATIO_THRESHOLD,
    UPWARD_BIAS_THRESHOLD,
    DOWNWARD_BIAS_THRESHOLD,
    BREAK_BIAS_THRESHOLD,
    DRIFT_THRESHOLD
)


def determine_strategy(stats_45: Dict, stats_90: Dict) -> str:
    """
    Determine trading strategy based on containment and directional patterns
    
    Args:
        stats_45: 45-day statistics
        stats_90: 90-day statistics
    
    Returns:
        Strategy recommendation string
    """
    ic_recommendations = []
    bias_reasons = []
    
    # Check 90-day containment
    if stats_90['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_90['breaks_up'], stats_90['breaks_down']) / (
            min(stats_90['breaks_up'], stats_90['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            ic_recommendations.append("IC90")
        elif stats_90['break_bias'] >= BREAK_BIAS_THRESHOLD:
            ic_recommendations.append("IC90⚠↑")
        elif stats_90['break_bias'] <= (100 - BREAK_BIAS_THRESHOLD):
            ic_recommendations.append("IC90⚠↓")
        else:
            ic_recommendations.append("IC90")
    
    # Check 45-day containment
    if stats_45['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_45['breaks_up'], stats_45['breaks_down']) / (
            min(stats_45['breaks_up'], stats_45['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            ic_recommendations.append("IC45")
        elif stats_45['break_bias'] >= BREAK_BIAS_THRESHOLD:
            ic_recommendations.append("IC45⚠↑")
        elif stats_45['break_bias'] <= (100 - BREAK_BIAS_THRESHOLD):
            ic_recommendations.append("IC45⚠↓")
        else:
            ic_recommendations.append("IC45")
    
    # If any IC pattern found, return it
    if ic_recommendations:
        return " + ".join(ic_recommendations)
    
    # No IC pattern - check for directional bias
    has_upward_edge = False
    has_downward_edge = False
    
    if stats_90['overall_bias'] >= UPWARD_BIAS_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    if stats_90['breaks_up'] >= stats_90['breaks_down'] * 1.5 and stats_90['breaks_up'] >= 2:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↑ breaks")
    
    # Use relative drift (adaptive to volatility)
    if stats_90['drift_vs_width'] >= 20:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    elif stats_90['avg_move_pct'] >= DRIFT_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    if stats_90['overall_bias'] <= DOWNWARD_BIAS_THRESHOLD:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    if stats_90['breaks_down'] >= stats_90['breaks_up'] * 1.5 and stats_90['breaks_down'] >= 2:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↓ breaks")
    
    if stats_90['drift_vs_width'] <= -20:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    elif stats_90['avg_move_pct'] <= -DRIFT_THRESHOLD:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    if has_upward_edge:
        reason_str = ", ".join(bias_reasons)
        return f"BIAS↑ ({reason_str})"
    elif has_downward_edge:
        reason_str = ", ".join(bias_reasons)
        return f"BIAS↓ ({reason_str})"
    
    return "SKIP"