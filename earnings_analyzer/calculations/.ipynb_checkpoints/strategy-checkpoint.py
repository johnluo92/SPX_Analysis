"""Strategy determination logic"""
from typing import Dict, Tuple
from ..config import (
    CONTAINMENT_THRESHOLD,
    BREAK_RATIO_THRESHOLD,
    UPWARD_BIAS_THRESHOLD,
    DOWNWARD_BIAS_THRESHOLD,
    BREAK_BIAS_THRESHOLD,
    DRIFT_THRESHOLD,
    MIN_BREAKS_FOR_SIGNAL
)


def determine_strategy_45(stats_45: Dict) -> Tuple[str, int]:
    """
    Determine 45-day strategy based on containment and directional patterns
    Requires 2/3 signals agreeing with NO conflicting signals
    
    Args:
        stats_45: 45-day statistics with keys:
            - containment: % contained within width
            - trend_pct: % of times closed above entry
            - breaks_up/breaks_down: break counts
            - drift_pct: average % move
    
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
        elif stats_45['break_up_pct'] >= BREAK_BIAS_THRESHOLD:
            patterns.append("IC45⚠️↑")
        else:
            patterns.append("IC45⚠️↓")
    
    # Check for directional BIAS45 - REQUIRES CONSENSUS
    up_signals = 0
    down_signals = 0
    bias_reasons = []
    
    # Signal 1: Trend direction (% closed above entry)
    if stats_45['trend_pct'] >= UPWARD_BIAS_THRESHOLD:
        up_signals += 1
        bias_reasons.append(f"{stats_45['trend_pct']:.0f}% up-trend")
    
    if stats_45['trend_pct'] <= DOWNWARD_BIAS_THRESHOLD:
        down_signals += 1
        down_pct = 100 - stats_45['trend_pct']
        bias_reasons.append(f"{down_pct:.0f}% down-trend")
    
    # Signal 2: Break ratio (ONLY if sufficient sample size)
    total_breaks = stats_45['breaks_up'] + stats_45['breaks_down']
    
    if total_breaks >= MIN_BREAKS_FOR_SIGNAL:
        # Check upward break dominance
        if stats_45['breaks_down'] > 0:
            up_ratio = stats_45['breaks_up'] / stats_45['breaks_down']
        else:
            up_ratio = float('inf') if stats_45['breaks_up'] > 0 else 0
        
        if up_ratio >= BREAK_RATIO_THRESHOLD and stats_45['breaks_up'] >= 2:
            up_signals += 1
            bias_reasons.append(f"{stats_45['breaks_up']}:{stats_45['breaks_down']}↑ breaks")
        
        # Check downward break dominance
        if stats_45['breaks_up'] > 0:
            down_ratio = stats_45['breaks_down'] / stats_45['breaks_up']
        else:
            down_ratio = float('inf') if stats_45['breaks_down'] > 0 else 0
        
        if down_ratio >= BREAK_RATIO_THRESHOLD and stats_45['breaks_down'] >= 2:
            down_signals += 1
            bias_reasons.append(f"{stats_45['breaks_up']}:{stats_45['breaks_down']}↓ breaks")
    
    # Signal 3: Drift magnitude
    if stats_45['drift_pct'] >= DRIFT_THRESHOLD:
        up_signals += 1
        bias_reasons.append(f"{stats_45['drift_pct']:+.1f}% drift")
    
    if stats_45['drift_pct'] <= -DRIFT_THRESHOLD:
        down_signals += 1
        bias_reasons.append(f"{stats_45['drift_pct']:+.1f}% drift")
    
    # Add BIAS pattern ONLY if clear consensus (2+ signals, 0 opposing)
    if up_signals >= 2 and down_signals == 0:
        reason_str = ", ".join(bias_reasons)
        edge_count += 1
        patterns.append(f"BIAS↑ ({reason_str})")
    elif down_signals >= 2 and up_signals == 0:
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
    Requires 2/3 signals agreeing with NO conflicting signals
    
    Args:
        stats_90: 90-day statistics with keys:
            - containment: % contained within width
            - trend_pct: % of times closed above entry
            - breaks_up/breaks_down: break counts
            - drift_pct: average % move
    
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
        elif stats_90['break_up_pct'] >= BREAK_BIAS_THRESHOLD:
            patterns.append("IC90⚠️↑")
        else:
            patterns.append("IC90⚠️↓")
    
    # Check for directional BIAS90 - REQUIRES CONSENSUS
    up_signals = 0
    down_signals = 0
    bias_reasons = []
    
    # Signal 1: Trend direction (% closed above entry)
    if stats_90['trend_pct'] >= UPWARD_BIAS_THRESHOLD:
        up_signals += 1
        bias_reasons.append(f"{stats_90['trend_pct']:.0f}% up-trend")
    
    if stats_90['trend_pct'] <= DOWNWARD_BIAS_THRESHOLD:
        down_signals += 1
        down_pct = 100 - stats_90['trend_pct']
        bias_reasons.append(f"{down_pct:.0f}% down-trend")
    
    # Signal 2: Break ratio (ONLY if sufficient sample size)
    total_breaks = stats_90['breaks_up'] + stats_90['breaks_down']
    
    if total_breaks >= MIN_BREAKS_FOR_SIGNAL:
        # Check upward break dominance
        if stats_90['breaks_down'] > 0:
            up_ratio = stats_90['breaks_up'] / stats_90['breaks_down']
        else:
            up_ratio = float('inf') if stats_90['breaks_up'] > 0 else 0
        
        if up_ratio >= BREAK_RATIO_THRESHOLD and stats_90['breaks_up'] >= 2:
            up_signals += 1
            bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↑ breaks")
        
        # Check downward break dominance
        if stats_90['breaks_up'] > 0:
            down_ratio = stats_90['breaks_down'] / stats_90['breaks_up']
        else:
            down_ratio = float('inf') if stats_90['breaks_down'] > 0 else 0
        
        if down_ratio >= BREAK_RATIO_THRESHOLD and stats_90['breaks_down'] >= 2:
            down_signals += 1
            bias_reasons.append(f"{stats_90['breaks_up']}:{stats_90['breaks_down']}↓ breaks")
    
    # Signal 3: Drift magnitude
    if stats_90['drift_pct'] >= DRIFT_THRESHOLD:
        up_signals += 1
        bias_reasons.append(f"{stats_90['drift_pct']:+.1f}% drift")
    
    if stats_90['drift_pct'] <= -DRIFT_THRESHOLD:
        down_signals += 1
        bias_reasons.append(f"{stats_90['drift_pct']:+.1f}% drift")
    
    # Add BIAS pattern ONLY if clear consensus (2+ signals, 0 opposing)
    if up_signals >= 2 and down_signals == 0:
        reason_str = ", ".join(bias_reasons)
        edge_count += 1
        patterns.append(f"BIAS↑ ({reason_str})")
    elif down_signals >= 2 and up_signals == 0:
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