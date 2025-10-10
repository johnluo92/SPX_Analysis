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
    
    All directional bias detection uses 90-day thresholds:
    - Overall bias: % of moves that were positive
    - Break bias: When breaks occur, which direction?
    - Drift: Average move size and direction
    
    45-day analysis is used ONLY for IC45 containment patterns.
    No directional bias is calculated from 45-day data.
    
    Args:
        stats_45: 45-day statistics
        stats_90: 90-day statistics
    
    Returns:
        Strategy recommendation string with edge count
    """
    rec_parts = []
    bias_reasons = []
    edge_count = 0  # Track independent patterns
    
    # Pattern 1: IC90 (mean reversion at 90 days)
    if stats_90['containment'] >= CONTAINMENT_THRESHOLD:
        edge_count += 1
        break_ratio = max(stats_90['breaks_up'], stats_90['breaks_down']) / (
            min(stats_90['breaks_up'], stats_90['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            rec_parts.append("IC90")
        elif stats_90['break_bias'] >= BREAK_BIAS_THRESHOLD:
            rec_parts.append("IC90⚠↑")
        else:
            rec_parts.append("IC90⚠↓")
    
    # Pattern 2: IC45 (mean reversion at 45 days)
    if stats_45['containment'] >= CONTAINMENT_THRESHOLD:
        edge_count += 1
        break_ratio = max(stats_45['breaks_up'], stats_45['breaks_down']) / (
            min(stats_45['breaks_up'], stats_45['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            rec_parts.append("IC45")
        elif stats_45['break_bias'] >= BREAK_BIAS_THRESHOLD:
            rec_parts.append("IC45⚠↑")
        else:
            rec_parts.append("IC45⚠↓")
    
    # Pattern 3: Directional bias (90-day only)
    has_upward_edge = False
    has_downward_edge = False
    
    # Check overall bias
    if stats_90['overall_bias'] >= UPWARD_BIAS_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    # Check break pattern
    if stats_90['breaks_up'] >= stats_90['breaks_down'] * 1.5 and stats_90['breaks_up'] >= 2:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↑ breaks")
    
    # Check drift
    if stats_90['avg_move_pct'] >= DRIFT_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    # Downward checks
    if stats_90['overall_bias'] <= DOWNWARD_BIAS_THRESHOLD:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    if stats_90['breaks_down'] >= stats_90['breaks_up'] * 1.5 and stats_90['breaks_down'] >= 2:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↓ breaks")
    
    if stats_90['avg_move_pct'] <= -DRIFT_THRESHOLD:
        has_downward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    # Add bias pattern if detected
    if has_upward_edge:
        edge_count += 1
        reason_str = ", ".join(bias_reasons)
        rec_parts.append(f"BIAS↑ ({reason_str})")
    elif has_downward_edge:
        edge_count += 1
        reason_str = ", ".join(bias_reasons)
        rec_parts.append(f"BIAS↓ ({reason_str})")
    
    # Build final string
    if not rec_parts:
        return "SKIP"
    
    strategy_str = " + ".join(rec_parts)
    
    # Add edge count
    edge_suffix = f" [{edge_count} edge{'s' if edge_count != 1 else ''}]"
    
    return strategy_str + edge_suffix