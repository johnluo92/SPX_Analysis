"""Insight generation from analysis results

Extracted from batch.py - analyzes patterns and generates summaries
"""
import pandas as pd
from typing import Union, List

from ..core.models import AnalysisResult


def print_insights(results: Union[pd.DataFrame, List[AnalysisResult]]) -> None:
    """Print condensed insights for quick batch overview
    
    Args:
        results: Either DataFrame (legacy) or List[AnalysisResult] (new)
    """
    # Convert to DataFrame if needed
    if isinstance(results, list):
        df = pd.DataFrame([r.to_dict() for r in results])
    else:
        df = results
    
    # Line 1: Pattern counts
    ic_count = len(df[df['strategy'].str.contains('IC', na=False)])
    bias_up_count = len(df[df['strategy'].str.contains('BIAS↑', na=False)])
    bias_down_count = len(df[df['strategy'].str.contains('BIAS↓', na=False)])
    skip_count = len(df[df['strategy'] == 'SKIP'])
    
    # Line 2: High conviction tickers
    high_conviction_dict = {}
    high_conviction = df[df['strategy'].str.contains(r'\[(?:\d+) edges?\]', regex=True, na=False)]
    if not high_conviction.empty:
        high_conviction = high_conviction.copy()
        high_conviction['edge_count'] = high_conviction['strategy'].str.extract(r'\[(\d+) edge').astype(int)
        high_conviction = high_conviction[high_conviction['edge_count'] >= 3].sort_values('edge_count', ascending=False)
        for _, row in high_conviction.iterrows():
            high_conviction_dict[row['ticker']] = int(row['edge_count'])
    
    # Line 3: Asymmetric IC warnings
    ic_up_skew = df[(df['strategy'].str.contains('IC.*⚠️↑', regex=True, na=False))]
    ic_down_skew = df[(df['strategy'].str.contains('IC.*⚠️↓', regex=True, na=False))]
    asymmetric_warnings = []
    if not ic_up_skew.empty:
        asymmetric_warnings.extend([f"{t}↑" for t in ic_up_skew['ticker'].tolist()])
    if not ic_down_skew.empty:
        asymmetric_warnings.extend([f"{t}↓" for t in ic_down_skew['ticker'].tolist()])
    
    # Print condensed format
    print(f"\n{'='*140}")
    print("📊 QUICK INTEL")
    print("="*140)
    
    # Line 1
    print(f"{ic_count} IC candidates | {bias_up_count} Bias↑ | {bias_down_count} Bias↓ | {skip_count} Skip")
    
    # Line 2
    if high_conviction_dict:
        conviction_str = ', '.join([f"{ticker}[{edges}]" for ticker, edges in high_conviction_dict.items()])
        print(f"High conviction (3+ edges): {conviction_str}")
    else:
        print(f"High conviction (3+ edges): None")
    
    # Line 3
    if asymmetric_warnings:
        print(f"⚠️ Asymmetric ICs: {', '.join(asymmetric_warnings)} (adjust wings for directional risk)")
    else:
        print(f"⚠️ Asymmetric ICs: None")
    
    print("="*140)
    
    # Strong directional signals
    _print_strong_directional_signals(df)


def _print_strong_directional_signals(df: pd.DataFrame) -> None:
    """Print strong directional signals split by timeframe"""
    signals_45d = []
    signals_90d = []
    
    for _, row in df.iterrows():
        if 'BIAS' not in str(row['strategy']):
            continue
        
        # Check 45d signal
        trend_45 = row['45d_trend_pct']
        if trend_45 >= 70 or trend_45 <= 30:
            direction = "↑" if trend_45 >= 70 else "↓"
            breaks_up = row['45d_breaks_up']
            breaks_dn = row['45d_breaks_dn']
            
            # Format break ratio with directional arrow if strong
            breaks_total = breaks_up + breaks_dn
            if breaks_total > 0:
                up_pct = (breaks_up / breaks_total) * 100
                if up_pct >= 60:
                    break_str = f"{breaks_up}:{breaks_dn}↑"
                elif up_pct <= 40:
                    break_str = f"{breaks_up}:{breaks_dn}↓"
                else:
                    break_str = f"{breaks_up}:{breaks_dn}"
            else:
                break_str = "0:0"
            
            signals_45d.append({
                'ticker': row['ticker'],
                'trend': trend_45,
                'direction': direction,
                'break_str': break_str,
                'drift': row['45d_drift']
            })
        
        # Check 90d signal
        trend_90 = row['90d_trend_pct']
        if trend_90 >= 70 or trend_90 <= 30:
            direction = "↑" if trend_90 >= 70 else "↓"
            breaks_up = row['90d_breaks_up']
            breaks_dn = row['90d_breaks_dn']
            
            # Format break ratio
            breaks_total = breaks_up + breaks_dn
            if breaks_total > 0:
                up_pct = (breaks_up / breaks_total) * 100
                if up_pct >= 60:
                    break_str = f"{breaks_up}:{breaks_dn}↑"
                elif up_pct <= 40:
                    break_str = f"{breaks_up}:{breaks_dn}↓"
                else:
                    break_str = f"{breaks_up}:{breaks_dn}"
            else:
                break_str = "0:0"
            
            signals_90d.append({
                'ticker': row['ticker'],
                'trend': trend_90,
                'direction': direction,
                'break_str': break_str,
                'drift': row['90d_drift']
            })
    
    if signals_45d or signals_90d:
        print(f"\n┌─ STRONG DIRECTIONAL SIGNALS")
        
        if signals_45d:
            print(f"│  45d timeframe:")
            for signal in signals_45d:
                print(f"│    {signal['ticker']:6} {signal['trend']:.0f}% trend {signal['direction']}, "
                      f"{signal['break_str']} breaks, {signal['drift']:+.1f}% drift")
        
        if signals_90d:
            if signals_45d:
                print(f"│")
            print(f"│  90d timeframe:")
            for signal in signals_90d:
                print(f"│    {signal['ticker']:6} {signal['trend']:.0f}% trend {signal['direction']}, "
                      f"{signal['break_str']} breaks, {signal['drift']:+.1f}% drift")
        
        print(f"└─")
    
    print(f"\n{'─'*140}")
    print(f"NOTE: Past patterns do not guarantee future results. IV context shows current opportunity cost.")
    print(f"{'─'*140}")