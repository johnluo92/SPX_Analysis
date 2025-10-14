"""Interactive quality matrix visualization using Plotly"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional


def plot_quality_matrix(df: pd.DataFrame, save_path: str = 'quality_matrix.html', show: bool = True):
    """
    Create interactive quality matrix plot with side-by-side 45d and 90d views
    
    Args:
        df: Results dataframe from batch_analyze
        save_path: Path to save HTML file
        show: Whether to open in browser
    """
    
    # Separate by strategy type
    df_ic45 = df[df['strategy'].str.contains('IC45', na=False)]
    df_ic90 = df[df['strategy'].str.contains('IC90', na=False)]
    df_bias = df[df['strategy'].str.contains('BIAS', na=False) & ~df['strategy'].str.contains('IC', na=False)]
    df_skip = df[df['strategy'] == 'SKIP']
    
    # Create side-by-side subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('45-Day Containment', '90-Day Containment'),
        horizontal_spacing=0.12
    )
    
    # === LEFT PLOT: 45-Day View ===
    
    # IC45 candidates (green)
    if not df_ic45.empty:
        fig.add_trace(go.Scatter(
            x=df_ic45['current_iv'],
            y=df_ic45['45d_contain'],
            mode='markers+text',
            name='IC45',
            marker=dict(size=10, color='green', opacity=0.7, line=dict(width=1, color='darkgreen')),
            text=df_ic45['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '45d Containment: %{y:.1f}%<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=1)
    
    # Skip points (gray) for 45d
    if not df_skip.empty:
        fig.add_trace(go.Scatter(
            x=df_skip['current_iv'],
            y=df_skip['45d_contain'],
            mode='markers+text',
            name='Skip',
            marker=dict(size=8, color='gray', opacity=0.5),
            text=df_skip['ticker'],
            textposition='top center',
            textfont=dict(size=8, color='gray'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '45d Containment: %{y:.1f}%<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=1)
    
    # === RIGHT PLOT: 90-Day View ===
    
    # IC90 candidates (green)
    if not df_ic90.empty:
        fig.add_trace(go.Scatter(
            x=df_ic90['current_iv'],
            y=df_ic90['90d_contain'],
            mode='markers+text',
            name='IC90',
            marker=dict(size=10, color='green', opacity=0.7, line=dict(width=1, color='darkgreen')),
            text=df_ic90['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=2)
    
    # Directional Bias (blue)
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
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=2)
    
    # Skip points (gray) for 90d
    if not df_skip.empty:
        fig.add_trace(go.Scatter(
            x=df_skip['current_iv'],
            y=df_skip['90d_contain'],
            mode='markers+text',
            name='Skip',
            marker=dict(size=8, color='gray', opacity=0.5),
            text=df_skip['ticker'],
            textposition='top center',
            textfont=dict(size=8, color='gray'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '<extra></extra>',
            showlegend=False  # Don't duplicate Skip in legend
        ), row=1, col=2)
    
    # Add threshold lines to both plots
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="IC Threshold", annotation_position="right",
                  row=1, col=1)
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="IC Threshold", annotation_position="right",
                  row=1, col=2)
    
    # Layout
    fig.update_layout(
        title=dict(
            text='Trade Quality Matrix - Earnings Plays',
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
        )
    )
    
    # Update axes - same range for both
    fig.update_xaxes(title_text="Current Implied Volatility (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="Current Implied Volatility (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, row=1, col=2)
    
    fig.update_yaxes(title_text="45-Day Containment Rate (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, range=[40, 95], row=1, col=1)
    fig.update_yaxes(title_text="90-Day Containment Rate (%)", gridcolor='lightgray', 
                     showgrid=True, zeroline=False, range=[40, 95], row=1, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"\n✅ Quality matrix saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig