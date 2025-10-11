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
    
    Returns combined pattern string with edge counts embedded.
    
    Args:
        stats_45: 45-day statistics
        stats_90: 90-day statistics
    
    Returns:
        Strategy recommendation string with all viable patterns
    """
    patterns = []
    total_edges = 0
    
    # Check IC90 (always check, don't exclude IC45)
    if stats_90['containment'] >= CONTAINMENT_THRESHOLD:
        total_edges += 1
        break_ratio = max(stats_90['breaks_up'], stats_90['breaks_down']) / (
            min(stats_90['breaks_up'], stats_90['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            patterns.append("IC90")
        elif stats_90['break_bias'] >= BREAK_BIAS_THRESHOLD:
            patterns.append("IC90⚠↑")
        else:
            patterns.append("IC90⚠↓")
    
    # Check IC45 (independently, not mutually exclusive with IC90)
    if stats_45['containment'] >= CONTAINMENT_THRESHOLD:
        total_edges += 1
        break_ratio = max(stats_45['breaks_up'], stats_45['breaks_down']) / (
            min(stats_45['breaks_up'], stats_45['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            patterns.append("IC45")
        elif stats_45['break_bias'] >= BREAK_BIAS_THRESHOLD:
            patterns.append("IC45⚠↑")
        else:
            patterns.append("IC45⚠↓")
    
    # Check for BIAS (can combine with IC patterns)
    bias_reasons = []
    has_upward_edge = False
    has_downward_edge = False
    
    # Check upward bias signals - FIXED: Use actual ratio, not just >= 1.5x
    if stats_90['overall_bias'] >= UPWARD_BIAS_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    # FIXED: Calculate actual ratio and require >= 2.0x (matches BREAK_RATIO_THRESHOLD)
    if stats_90['breaks_down'] > 0:
        up_ratio = stats_90['breaks_up'] / stats_90['breaks_down']
    else:
        up_ratio = float('inf') if stats_90['breaks_up'] > 0 else 0
    
    if up_ratio >= BREAK_RATIO_THRESHOLD and stats_90['breaks_up'] >= 2:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↑ breaks")
    
    if stats_90['avg_move_pct'] >= DRIFT_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    # Check downward bias signals - FIXED: Use actual ratio
    if stats_90['overall_bias'] <= DOWNWARD_BIAS_THRESHOLD:
        has_downward_edge = True
        if not bias_reasons:
            bias_reasons.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    if stats_90['breaks_up'] > 0:
        down_ratio = stats_90['breaks_down'] / stats_90['breaks_up']
    else:
        down_ratio = float('inf') if stats_90['breaks_down'] > 0 else 0
    
    if down_ratio >= BREAK_RATIO_THRESHOLD and stats_90['breaks_down'] >= 2:
        has_downward_edge = True
        if not has_upward_edge:
            bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↓ breaks")
    
    if stats_90['avg_move_pct'] <= -DRIFT_THRESHOLD:
        has_downward_edge = True
        if not has_upward_edge:
            bias_reasons.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    # Add BIAS pattern if found (BIAS counts as 1 edge total, not per reason)
    if has_upward_edge and not has_downward_edge:
        reason_str = ", ".join(bias_reasons)
        total_edges += 1  # BIAS is 1 edge regardless of supporting reasons
        patterns.append(f"BIAS↑ ({reason_str})")
    elif has_downward_edge and not has_upward_edge:
        reason_str = ", ".join(bias_reasons)
        total_edges += 1  # BIAS is 1 edge regardless of supporting reasons
        patterns.append(f"BIAS↓ ({reason_str})")
    
    # Combine patterns with total edge count at the end
    if patterns:
        pattern_str = " + ".join(patterns)
        edge_text = "edges" if total_edges > 1 else "edge"
        return f"{pattern_str} [{total_edges} {edge_text}]"
    else:
        return "SKIP"