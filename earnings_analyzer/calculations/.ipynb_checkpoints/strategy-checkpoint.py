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
    
    IMPORTANT: All BIAS signals (directional edge detection) use 90-day thresholds only.
    - IC45/IC90: Containment windows (mean reversion timeframes)
    - BIAS↑/BIAS↓: Directional trends (always 90-day analysis)
    - Edge count: Number of independent 90-day patterns detected
    
    Args:
        stats_45: 45-day statistics (used for IC45 containment only)
        stats_90: 90-day statistics (used for IC90 containment AND all bias detection)
    
    Returns:
        Strategy recommendation string with edge count
    """
    rec_parts = []
    bias_reasons = []
    
    # Check 90-day IC first
    if stats_90['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_90['breaks_up'], stats_90['breaks_down']) / (
            min(stats_90['breaks_up'], stats_90['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            rec_parts.append("IC90")
        elif stats_90['break_bias'] >= BREAK_BIAS_THRESHOLD:
            rec_parts.append("IC90⚠↑")
        else:
            rec_parts.append("IC90⚠↓")
    
    # Check 45-day IC
    if stats_45['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_45['breaks_up'], stats_45['breaks_down']) / (
            min(stats_45['breaks_up'], stats_45['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            rec_parts.append("IC45")
        elif stats_45['break_bias'] >= BREAK_BIAS_THRESHOLD:
            rec_parts.append("IC45⚠↑")
        else:
            rec_parts.append("IC45⚠↓")
    
    # Check directional bias (using 90-day stats)
    has_upward_edge = False
    has_downward_edge = False
    
    if stats_90['overall_bias'] >= UPWARD_BIAS_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% 90d bias")
    
    if stats_90['breaks_up'] >= stats_90['breaks_down'] * 1.5 and stats_90['breaks_up'] >= 2:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↑ 90d breaks")
    
    if stats_90['avg_move_pct'] >= DRIFT_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% 90d drift")
    
    if stats_90['overall_bias'] <= DOWNWARD_BIAS_THRESHOLD:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% 90d bias")
    
    if stats_90['breaks_down'] >= stats_90['breaks_up'] * 1.5 and stats_90['breaks_down'] >= 2:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↓ 90d breaks")
    
    if stats_90['avg_move_pct'] <= -DRIFT_THRESHOLD:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% 90d drift")
    
    if has_upward_edge:
        reason_str = ", ".join(bias_reasons)
        rec_parts.append(f"BIAS↑ ({reason_str})")
    elif has_downward_edge:
        reason_str = ", ".join(bias_reasons)
        rec_parts.append(f"BIAS↓ ({reason_str})")
    
    # Return combined strategy or SKIP
    if rec_parts:
        return " + ".join(rec_parts)
    else:
        return "SKIP"