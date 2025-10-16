"""Interactive quality matrix visualization using Plotly - V2 Refactored

Key improvements:
- Uses AnalysisResult.strategy_45 and .strategy_90 properties (no recalculation)
- Cleaner separation of concerns
- Works with both DataFrame and List[AnalysisResult]
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, Union, List

from ..core.models import AnalysisResult


def plot_quality_matrix_v2(results: Union[pd.DataFrame, List[AnalysisResult]], 
                           save_path: str = 'quality_matrix.html', 
                           show: bool = True):
    """
    Create interactive quality matrix plot with side-by-side 45d and 90d views
    
    V2: Uses cached strategy calculations from AnalysisResult objects
    
    Args:
        results: Either DataFrame or List[AnalysisResult]
        save_path: Path to save HTML file
        show: Whether to open in browser
    """
    # Convert to DataFrame if needed
    if isinstance(results, list):
        df = pd.DataFrame([r.to_dict() for r in results])
        results_list = results  # Keep typed objects for strategy access
    else:
        df = results
        # Convert DataFrame rows back to AnalysisResult for strategy access
        results_list = [AnalysisResult.from_dict(row) for _, row in df.iterrows()]
    
    # CRITICAL: Filter out tickers with missing IV data
    missing_iv_tickers = []
    valid_indices = []
    
    for i, result in enumerate(results_list):
        if result.has_iv_data:
            valid_indices.append(i)
        else:
            missing_iv_tickers.append(result.ticker)
    
    if missing_iv_tickers:
        print(f"\n⚠️  Quality Matrix: Excluding {len(missing_iv_tickers)} ticker(s) without IV data: {', '.join(missing_iv_tickers)}")
        print(f"    These tickers still appear in the backtest results table above.")
    
    if not valid_indices:
        print("\n❌ Cannot create quality matrix: No tickers with valid IV data")
        return None
    
    # Filter to valid results
    df = df.iloc[valid_indices].copy()
    results_list = [results_list[i] for i in valid_indices]
    
    # Get strategies using cached properties (no recalculation!)
    strategies_45d = []
    strategies_90d = []
    
    for result in results_list:
        pattern_45, _ = result.strategy_45  # Uses cached property
        pattern_90, _ = result.strategy_90  # Uses cached property
        strategies_45d.append(pattern_45)
        strategies_90d.append(pattern_90)
    
    # Add to DataFrame for filtering
    df['strategy_45d'] = strategies_45d
    df['strategy_90d'] = strategies_90d
    
    # Separate by strategy type
    df_ic45 = df[df['strategy_45d'].str.contains('IC45', na=False)]
    df_bias45 = df[(df['strategy_45d'].str.contains('BIAS', na=False)) & 
                   (~df['strategy_45d'].str.contains('IC', na=False))]
    
    df_ic90 = df[df['strategy_90d'].str.contains('IC90', na=False)]
    df_bias90 = df[(df['strategy_90d'].str.contains('BIAS', na=False)) & 
                   (~df['strategy_90d'].str.contains('IC', na=False))]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('45-Day Containment', '90-Day Containment'),
        horizontal_spacing=0.08
    )
    
    # === LEFT PLOT: 45-Day View ===
    _add_45d_traces(fig, df_ic45, df_bias45)
    
    # === RIGHT PLOT: 90-Day View ===
    _add_90d_traces(fig, df_ic90, df_bias90)
    
    # Add threshold lines
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2)
    
    # Build title
    title_text = 'Trade Quality Matrix - Earnings Plays'
    if missing_iv_tickers:
        title_text += f'<br><sub>Excluded {len(missing_iv_tickers)} ticker(s) without IV data: {", ".join(missing_iv_tickers)}</sub>'
    
    # Layout
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center'
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=1600,
        height=800,
        legend=dict(
            x=0.5,
            y=-0.1,
            xanchor='center',
            yanchor='top',
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        annotations=[
            dict(
                text="IC Threshold",
                x=0.5,
                y=69.5,
                xref="paper",
                yref="y",
                showarrow=False,
                font=dict(size=11, color='red'),
                xanchor='center',
                yanchor='bottom',
                bgcolor='rgba(255,255,255,0.8)',
                borderpad=4
            )
        ]
    )
    
    # Update axes
    fig.update_xaxes(title_text="Current Implied Volatility (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="45-Day Containment Rate (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, range=[40, 95], row=1, col=1)
    
    fig.update_xaxes(title_text="Current Implied Volatility (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, row=1, col=2)
    fig.update_yaxes(title_text="90-Day Containment Rate (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, range=[40, 95], 
                     side='right', row=1, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"\n✓ Quality matrix saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig


def _add_45d_traces(fig, df_ic45, df_bias45):
    """Add 45-day traces to figure"""
    # IC45 candidates (green)
    if not df_ic45.empty:
        customdata_45_ic = []
        for _, row in df_ic45.iterrows():
            dte_str = f"{int(row['iv_dte'])} DTE" if pd.notna(row.get('iv_dte')) else "N/A"
            customdata_45_ic.append([row['strategy_45d'], dte_str])
        
        fig.add_trace(go.Scatter(
            x=df_ic45['current_iv'],
            y=df_ic45['45d_contain'],
            mode='markers+text',
            name='IC45',
            marker=dict(size=10, color='green', opacity=0.7, line=dict(width=1, color='darkgreen')),
            text=df_ic45['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            customdata=customdata_45_ic,
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}% (%{customdata[1]})<br>' +
                         '45d Containment: %{y:.1f}%<br>' +
                         '%{customdata[0]}<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=1)
    
    # BIAS45 patterns (blue)
    if not df_bias45.empty:
        customdata_45_bias = []
        for _, row in df_bias45.iterrows():
            pattern = row['strategy_45d']
            dte_str = f"{int(row['iv_dte'])} DTE" if pd.notna(row.get('iv_dte')) else "N/A"
            customdata_45_bias.append([pattern, dte_str])
        
        fig.add_trace(go.Scatter(
            x=df_bias45['current_iv'],
            y=df_bias45['45d_contain'],
            mode='markers+text',
            name='BIAS45',
            marker=dict(size=10, color='blue', opacity=0.7, line=dict(width=1, color='darkblue')),
            text=df_bias45['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            customdata=customdata_45_bias,
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}% (%{customdata[1]})<br>' +
                         '45d Containment: %{y:.1f}%<br>' +
                         '%{customdata[0]}<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=1)


def _add_90d_traces(fig, df_ic90, df_bias90):
    """Add 90-day traces to figure"""
    # IC90 candidates (green)
    if not df_ic90.empty:
        customdata_90_ic = []
        for _, row in df_ic90.iterrows():
            dte_str = f"{int(row['iv_dte'])} DTE" if pd.notna(row.get('iv_dte')) else "N/A"
            customdata_90_ic.append([row['strategy_90d'], dte_str])
        
        fig.add_trace(go.Scatter(
            x=df_ic90['current_iv'],
            y=df_ic90['90d_contain'],
            mode='markers+text',
            name='IC90',
            marker=dict(size=10, color='green', opacity=0.7, line=dict(width=1, color='darkgreen')),
            text=df_ic90['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            customdata=customdata_90_ic,
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}% (%{customdata[1]})<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '%{customdata[0]}<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=2)
    
    # BIAS90 patterns (blue)
    if not df_bias90.empty:
        customdata_90_bias = []
        for _, row in df_bias90.iterrows():
            pattern = row['strategy_90d']
            dte_str = f"{int(row['iv_dte'])} DTE" if pd.notna(row.get('iv_dte')) else "N/A"
            customdata_90_bias.append([pattern, dte_str])
        
        fig.add_trace(go.Scatter(
            x=df_bias90['current_iv'],
            y=df_bias90['90d_contain'],
            mode='markers+text',
            name='BIAS90',
            marker=dict(size=10, color='blue', opacity=0.7, line=dict(width=1, color='darkblue')),
            text=df_bias90['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            customdata=customdata_90_bias,
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}% (%{customdata[1]})<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '%{customdata[0]}<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=2)