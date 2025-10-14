"""Strategy determination logic"""
from typing import Dict, Tuple
from ..config import (
    CONTAINMENT_THRESHOLD,
    BREAK_RATIO_THRESHOLD,
    UPWARD_BIAS_THRESHOLD,
    DOWNWARD_BIAS_THRESHOLD,
    BREAK_BIAS_THRESHOLD,
    DRIFT_THRESHOLD
)


def determine_strategy_45(stats_45: Dict) -> Tuple[str, int]:
    """
    Determine 45-day strategy based on containment and directional patterns
    
    Args:
        stats_45: 45-day statistics
    
    Returns:
        Tuple of (pattern_string, edge_count)
    """
    patterns = []
    edge_count = 0
    
    # Check IC45
    if stats_45['containment'] >= CONTAINMENT_THRESHOLD:
        edge_count += 1
        break_ratio = max(stats_45['breaks_up'], stats_45['breaks_down']) / (
            min(stats_45['breaks_up'], stats_45['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            patterns.append("IC45")
        elif stats_45['break_bias'] >= BREAK_BIAS_THRESHOLD:
            patterns.append("IC45⚠↑")
        else:
            patterns.append("IC45⚠↓")
    
    # Check for BIAS45
    bias_reasons = []
    has_upward_edge = False
    has_downward_edge = False
    
    # Check upward bias signals
    if stats_45['overall_bias'] >= UPWARD_BIAS_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_45['overall_bias']:.0f}% bias")
    
    # Calculate break ratio
    if stats_45['breaks_down'] > 0:
        up_ratio = stats_45['breaks_up'] / stats_45['breaks_down']
    else:
        up_ratio = float('inf') if stats_45['breaks_up'] > 0 else 0
    
    if up_ratio >= BREAK_RATIO_THRESHOLD and stats_45['breaks_up'] >= 2:
        has_upward_edge = True
        bias_reasons.append(f"{stats_45['breaks_up']}:{stats_45['breaks_down']}↑ breaks")
    
    if stats_45['avg_move_pct'] >= DRIFT_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_45['avg_move_pct']:+.1f}% drift")
    
    # Check downward bias signals
    if stats_45['overall_bias'] <= DOWNWARD_BIAS_THRESHOLD:
        has_downward_edge = True
        if not bias_reasons:
            bias_reasons.append(f"{stats_45['overall_bias']:.0f}% bias")
    
    if stats_45['breaks_up'] > 0:
        down_ratio = stats_45['breaks_down'] / stats_45['breaks_up']
    else:
        down_ratio = float('inf') if stats_45['breaks_down'] > 0 else 0
    
    if down_ratio >= BREAK_RATIO_THRESHOLD and stats_45['breaks_down'] >= 2:
        has_downward_edge = True
        if not has_upward_edge:
            bias_reasons.append(f"{stats_45['breaks_up']}:{stats_45['breaks_down']}↓ breaks")
    
    if stats_45['avg_move_pct'] <= -DRIFT_THRESHOLD:
        has_downward_edge = True
        if not has_upward_edge:
            bias_reasons.append(f"{stats_45['avg_move_pct']:+.1f}% drift")
    
    # Add BIAS pattern if found
    if has_upward_edge and not has_downward_edge:
        reason_str = ", ".join(bias_reasons)
        edge_count += 1
        patterns.append(f"BIAS↑ ({reason_str})")
    elif has_downward_edge and not has_upward_edge:
        reason_str = ", ".join(bias_reasons)
        edge_count += 1
        patterns.append(f"BIAS↓ ({reason_str})")
    
    # Combine patterns
    if patterns:
        pattern_str = " + ".join(patterns)
        return (pattern_str, edge_count)
    else:
        return ("SKIP", 0)


def determine_strategy_90(stats_90: Dict) -> Tuple[str, int]:
    """
    Determine 90-day strategy based on containment and directional patterns
    
    Args:
        stats_90: 90-day statistics
    
    Returns:
        Tuple of (pattern_string, edge_count)
    """
    patterns = []
    edge_count = 0
    
    # Check IC90
    if stats_90['containment'] >= CONTAINMENT_THRESHOLD:
        edge_count += 1
        break_ratio = max(stats_90['breaks_up'], stats_90['breaks_down']) / (
            min(stats_90['breaks_up'], stats_90['breaks_down']) + 1
        )
        
        if break_ratio < BREAK_RATIO_THRESHOLD:
            patterns.append("IC90")
        elif stats_90['break_bias'] >= BREAK_BIAS_THRESHOLD:
            patterns.append("IC90⚠↑")
        else:
            patterns.append("IC90⚠↓")
    
    # Check for BIAS90
    bias_reasons = []
    has_upward_edge = False
    has_downward_edge = False
    
    # Check upward bias signals
    if stats_90['overall_bias'] >= UPWARD_BIAS_THRESHOLD:
        has_upward_edge = True
        bias_reasons.append(f"{stats_90['overall_bias']:.0f}% bias")
    
    # Calculate break ratio
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
    
    # Check downward bias signals
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
    
    # Add BIAS pattern if found
    if has_upward_edge and not has_downward_edge:
        reason_str = ", ".join(bias_reasons)
        edge_count += 1
        patterns.append(f"BIAS↑ ({reason_str})")
    elif has_downward_edge and not has_upward_edge:
        reason_str = ", ".join(bias_reasons)
        edge_count += 1
        patterns.append(f"BIAS↓ ({reason_str})")
    
    # Combine patterns
    if patterns:
        pattern_str = " + ".join(patterns)
        return (pattern_str, edge_count)
    else:
        return ("SKIP", 0)


def determine_strategy(stats_45: Dict, stats_90: Dict) -> str:
    """
    Determine combined trading strategy - BACKWARD COMPATIBLE WRAPPER
    
    This function maintains backward compatibility by combining 45d and 90d
    strategies into a single string with total edge count.
    
    Args:
        stats_45: 45-day statistics
        stats_90: 90-day statistics
    
    Returns:
        Combined strategy recommendation string with total edge count
    """
    pattern_45, edges_45 = determine_strategy_45(stats_45)
    pattern_90, edges_90 = determine_strategy_90(stats_90)
    
    # Combine unique patterns (avoid duplicates)
    all_patterns = []
    
    # Add 90d patterns first (IC90 preferred over IC45 for display)
    if pattern_90 != "SKIP":
        all_patterns.append(pattern_90)
    
    # Add 45d patterns if different from 90d
    if pattern_45 != "SKIP":
        # Extract pattern types from 45d
        if "IC45" in pattern_45:
            all_patterns.append(pattern_45)
        elif "BIAS" in pattern_45 and "BIAS" not in pattern_90:
            # Only add BIAS45 if no BIAS90
            all_patterns.append(pattern_45)
    
    total_edges = edges_45 + edges_90
    
    if all_patterns:
        combined_pattern = " + ".join(all_patterns)
        edge_text = "edges" if total_edges > 1 else "edge"
        return f"{combined_pattern} [{total_edges} {edge_text}]"
    else:
        return "SKIP"