"""Interactive quality matrix visualization using Plotly"""
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def plot_quality_matrix(df: pd.DataFrame, save_path: str = 'quality_matrix.html', show: bool = True):
    """
    Create interactive quality matrix plot
    
    Args:
        df: Results dataframe from batch_analyze
        save_path: Path to save HTML file
        show: Whether to open in browser
    """
    
    # Separate by strategy type
    df_ic = df[df['strategy'].str.contains('IC', na=False)]
    df_bias = df[df['strategy'].str.contains('BIAS', na=False) & ~df['strategy'].str.contains('IC', na=False)]
    df_skip = df[df['strategy'] == 'SKIP']
    
    # Create figure
    fig = go.Figure()
    
    # Add Iron Condor points (green)
    if not df_ic.empty:
        fig.add_trace(go.Scatter(
            x=df_ic['current_iv'],
            y=df_ic['90d_contain'],
            mode='markers+text',
            name='Iron Condor',
            marker=dict(size=10, color='green', opacity=0.7, line=dict(width=1, color='darkgreen')),
            text=df_ic['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    # Add Directional Bias points (blue)
    if not df_bias.empty:
        fig.add_trace(go.Scatter(
            x=df_bias['current_iv'],
            y=df_bias['90d_contain'],
            mode='markers+text',
            name='Directional Bias',
            marker=dict(size=10, color='blue', opacity=0.7, line=dict(width=1, color='darkblue')),
            text=df_bias['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    # Add Skip points (gray) - optional, can remove
    if not df_skip.empty:
        fig.add_trace(go.Scatter(
            x=df_skip['current_iv'],
            y=df_skip['90d_contain'],
            mode='markers+text',
            name='No Edge (Skip)',
            marker=dict(size=8, color='gray', opacity=0.5),
            text=df_skip['ticker'],
            textposition='top center',
            textfont=dict(size=8, color='gray'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    # Add IC threshold line
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="IC Threshold", annotation_position="right")
    
    # Layout
    fig.update_layout(
        title=dict(
            text='Trade Quality Matrix - Earnings Plays',
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Current Implied Volatility (%)',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='90-Day Containment Rate (%)',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
            range=[40, 95]  # Focus on relevant range
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=1400,
        height=800,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"\n✅ Quality matrix saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig