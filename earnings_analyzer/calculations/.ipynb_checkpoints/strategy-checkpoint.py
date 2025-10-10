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
    All conditions are independent and can stack
    
    Args:
        stats_45: 45-day statistics
        stats_90: 90-day statistics
    
    Returns:
        Strategy recommendation string
    """
    recommendations = []
    
    # === CHECK 90-DAY CONTAINMENT ===
    if stats_90['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_90['breaks_up'], stats_90['breaks_down']) / (
            min(stats_90['breaks_up'], stats_90['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            recommendations.append("IC90")
        elif stats_90['break_bias'] >= BREAK_BIAS_THRESHOLD:
            recommendations.append("IC90⚠↑")
        elif stats_90['break_bias'] <= (100 - BREAK_BIAS_THRESHOLD):
            recommendations.append("IC90⚠↓")
        else:
            recommendations.append("IC90")
    
    # === CHECK 45-DAY CONTAINMENT ===
    if stats_45['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_45['breaks_up'], stats_45['breaks_down']) / (
            min(stats_45['breaks_up'], stats_45['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            recommendations.append("IC45")
        elif stats_45['break_bias'] >= BREAK_BIAS_THRESHOLD:
            recommendations.append("IC45⚠↑")
        elif stats_45['break_bias'] <= (100 - BREAK_BIAS_THRESHOLD):
            recommendations.append("IC45⚠↓")
        else:
            recommendations.append("IC45")
    
    # === CHECK DIRECTIONAL BIAS (ALWAYS) ===
    bias_reasons = []
    has_upward_edge = False
    has_downward_edge = False
    
    # UPWARD BIAS CHECKS
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
    
    # DOWNWARD BIAS CHECKS
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
    
    # ADD BIAS TO RECOMMENDATIONS
    if has_upward_edge:
        reason_str = ", ".join(bias_reasons)
        recommendations.append(f"BIAS↑ ({reason_str})")
    elif has_downward_edge:
        reason_str = ", ".join(bias_reasons)
        recommendations.append(f"BIAS↓ ({reason_str})")
    
    # === RETURN ALL RECOMMENDATIONS OR SKIP ===
    return " + ".join(recommendations) if recommendations else "SKIP"