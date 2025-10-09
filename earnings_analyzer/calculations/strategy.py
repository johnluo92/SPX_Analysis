"""Strategy determination logic - Dual pattern detection"""
from typing import Dict, List

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
    Can identify BOTH containment and bias patterns simultaneously
    
    Args:
        stats_45: 45-day statistics
        stats_90: 90-day statistics
    
    Returns:
        Strategy recommendation string (can include multiple patterns)
    """
    patterns = []
    
    # ========================================
    # STEP 1: Check for Containment Patterns
    # ========================================
    containment_pattern = None
    
    if stats_90['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_90['breaks_up'], stats_90['breaks_down']) / (
            min(stats_90['breaks_up'], stats_90['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            containment_pattern = "IC90"
        elif stats_90['break_bias'] >= BREAK_BIAS_THRESHOLD:
            containment_pattern = "IC90⚠↑"
        else:
            containment_pattern = "IC90⚠↓"
            
    elif stats_45['containment'] >= CONTAINMENT_THRESHOLD:
        break_ratio = max(stats_45['breaks_up'], stats_45['breaks_down']) / (
            min(stats_45['breaks_up'], stats_45['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            containment_pattern = "IC45"
        elif stats_45['break_bias'] >= BREAK_BIAS_THRESHOLD:
            containment_pattern = "IC45⚠↑"
        else:
            containment_pattern = "IC45⚠↓"
    
    # ========================================
    # STEP 2: Check for Directional Bias
    # ========================================
    bias_signals = []
    has_upward_edge = False
    has_downward_edge = False
    
    # Upward bias signals
    if stats_90['overall_bias'] >= UPWARD_BIAS_THRESHOLD:
        has_upward_edge = True
        bias_signals.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    if stats_90['breaks_up'] >= stats_90['breaks_down'] * 1.5 and stats_90['breaks_up'] >= 2:
        has_upward_edge = True
        bias_signals.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↑ breaks")
    
    if stats_90['avg_move_pct'] >= DRIFT_THRESHOLD:
        has_upward_edge = True
        bias_signals.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    # Downward bias signals
    if stats_90['overall_bias'] <= DOWNWARD_BIAS_THRESHOLD:
        has_downward_edge = True
        bias_signals.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    if stats_90['breaks_down'] >= stats_90['breaks_up'] * 1.5 and stats_90['breaks_down'] >= 2:
        has_downward_edge = True
        bias_signals.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↓ breaks")
    
    if stats_90['avg_move_pct'] <= -DRIFT_THRESHOLD:
        has_downward_edge = True
        bias_signals.append(f"{stats_90['avg_move_pct']:+.1f}% drift")
    
    # ========================================
    # STEP 3: Combine Patterns
    # ========================================
    
    # Add containment pattern first (if exists)
    if containment_pattern:
        patterns.append(containment_pattern)
    
    # Add directional bias (if exists)
    if has_upward_edge:
        reason_str = ", ".join(bias_signals)
        patterns.append(f"BIAS↑ ({reason_str})")
    elif has_downward_edge:
        reason_str = ", ".join(bias_signals)
        patterns.append(f"BIAS↓ ({reason_str})")
    
    # Return combined pattern or SKIP
    if patterns:
        return " + ".join(patterns)
    else:
        return "SKIP"