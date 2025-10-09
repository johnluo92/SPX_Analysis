"""Report generation"""
import pandas as pd
from typing import Dict, List


def generate_summary_report(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics from results dataframe
    
    Args:
        df: Results dataframe
    
    Returns:
        Dictionary with summary statistics
    """
    ic_count = len(df[df['strategy'].str.contains('IC', na=False)])
    bias_up_count = len(df[df['strategy'].str.contains('BIAS↑', na=False)])
    bias_down_count = len(df[df['strategy'].str.contains('BIAS↓', na=False)])
    skip_count = len(df[df['strategy'] == 'SKIP'])
    
    report = {
        'total_analyzed': len(df),
        'ic_candidates': ic_count,
        'bias_up': bias_up_count,
        'bias_down': bias_down_count,
        'skip': skip_count,
    }
    
    if 'iv_premium' in df.columns and df['iv_premium'].notna().any():
        elevated = df[df['iv_premium'] >= 15]
        depressed = df[df['iv_premium'] <= -15]
        normal = df[(df['iv_premium'] > -15) & (df['iv_premium'] < 15)]
        
        report['iv_elevated'] = len(elevated)
        report['iv_depressed'] = len(depressed)
        report['iv_normal'] = len(normal)
        
        if not elevated.empty:
            report['most_elevated'] = elevated.nlargest(5, 'iv_premium')[['ticker', 'iv_premium']].to_dict('records')
        
        if not depressed.empty:
            report['most_depressed'] = depressed.nsmallest(5, 'iv_premium')[['ticker', 'iv_premium']].to_dict('records')
    
    ic_up_skew = df[(df['strategy'].str.contains('IC.*⚠↑', regex=True, na=False))]
    ic_down_skew = df[(df['strategy'].str.contains('IC.*⚠↓', regex=True, na=False))]
    
    if not ic_up_skew.empty:
        report['ic_upside_risk'] = ic_up_skew['ticker'].tolist()
    
    if not ic_down_skew.empty:
        report['ic_downside_risk'] = ic_down_skew['ticker'].tolist()
    
    strong_bias = df[
        (df['strategy'].str.contains('BIAS', na=False)) &
        ((df['90d_overall_bias'] >= 70) | (df['90d_overall_bias'] <= 30))
    ]
    
    if not strong_bias.empty:
        report['strong_directional'] = strong_bias[['ticker', '90d_overall_bias', '90d_drift']].to_dict('records')
    
    return report


def export_to_csv(df: pd.DataFrame, filename: str = 'earnings_analysis.csv') -> None:
    """
    Export results to CSV
    
    Args:
        df: Results dataframe
        filename: Output filename
    """
    df.to_csv(filename, index=False)
    print(f"\n✓ Results exported to {filename}")


def export_to_json(df: pd.DataFrame, filename: str = 'earnings_analysis.json') -> None:
    """
    Export results to JSON
    
    Args:
        df: Results dataframe
        filename: Output filename
    """
    df.to_json(filename, orient='records', indent=2)
    print(f"\n✓ Results exported to {filename}")