"""Interactive quality matrix visualization using Plotly - Enhanced Version"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional


def plot_quality_matrix(df: pd.DataFrame, save_path: str = 'quality_matrix.html', show: bool = True):
    """
    Create interactive quality matrix plot with enhanced readability
    
    Args:
        df: Results dataframe from batch_analyze
        save_path: Path to save HTML file
        show: Whether to open in browser
    """
    
    # Separate by PRIMARY strategy (IC supersedes BIAS)
    df_ic = df[df['strategy'].str.contains('IC', na=False)]
    df_bias = df[(df['strategy'].str.contains('BIAS', na=False)) & (~df['strategy'].str.contains('IC', na=False))]
    df_skip = df[df['strategy'] == 'SKIP']
    
    # Create figure
    fig = go.Figure()
    
    # Add Iron Condor points with smart labeling
    if not df_ic.empty:
        fig.add_trace(go.Scatter(
            x=df_ic['iv_elevation'],
            y=df_ic['90d_contain'],
            mode='markers+text',
            name='Iron Condor',
            marker=dict(
                size=11,
                color='green',
                opacity=0.8,
                line=dict(width=1.5, color='darkgreen')
            ),
            text=df_ic['ticker'],
            textposition='top center',
            textfont=dict(size=8, color='darkgreen'),
            cliponaxis=False,  # Allow labels outside plot area
            hovertemplate='<b>%{text}</b><br>' +
                         'IV Premium: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         'Strategy: Iron Condor<br>' +
                         '<extra></extra>'
        ))
    
    # Add Directional Bias points with smart labeling
    if not df_bias.empty:
        fig.add_trace(go.Scatter(
            x=df_bias['iv_elevation'],
            y=df_bias['90d_contain'],
            mode='markers+text',
            name='Directional Bias',
            marker=dict(
                size=11,
                color='#4169E1',
                opacity=0.8,
                line=dict(width=1.5, color='darkblue')
            ),
            text=df_bias['ticker'],
            textposition='top center',
            textfont=dict(size=8, color='darkblue'),
            cliponaxis=False,
            hovertemplate='<b>%{text}</b><br>' +
                         'IV Premium: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         'Strategy: Directional Bias<br>' +
                         '<extra></extra>'
        ))
    
    # Add Skip points with labels
    if not df_skip.empty:
        fig.add_trace(go.Scatter(
            x=df_skip['iv_elevation'],
            y=df_skip['90d_contain'],
            mode='markers+text',
            name='No Edge (Skip)',
            marker=dict(
                size=10,
                color='lightgray',
                opacity=0.6,
                line=dict(width=1, color='gray')
            ),
            text=df_skip['ticker'],
            textposition='top center',
            textfont=dict(size=7, color='gray'),
            cliponaxis=False,
            hovertemplate='<b>%{text}</b><br>' +
                         'IV Premium: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         'No Edge Detected<br>' +
                         '<extra></extra>'
        ))
    
    # Simplified reference lines - fewer annotations
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="IC Threshold", 
                  annotation_position="right",
                  annotation_font_size=10)
    
    # Simpler vertical reference line
    fig.add_vline(x=15, line_dash="dot", line_color="green", line_width=1.5,
                  annotation_text="Rich Zone", 
                  annotation_position="top",
                  annotation_font_size=10)
    
    # Subtle shaded regions with less opacity
    fig.add_vrect(x0=15, x1=df['iv_elevation'].max() + 10, 
                  fillcolor="green", opacity=0.05, layer="below",
                  line_width=0)
    
    # Highlight the "sweet spot" zone more subtly
    fig.add_shape(type="rect",
                  x0=15, y0=69.5, x1=df['iv_elevation'].max() + 10, y1=100,
                  fillcolor="green", opacity=0.08,
                  line=dict(width=0), layer="below")
    
    # Update layout with cleaner appearance
    fig.update_layout(
        title={
            'text': 'Trade Quality Matrix - Earnings Plays',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='IV Premium (%) - Implied Vol vs 45d Realized Vol',
        yaxis_title='90-Day Containment Rate (%) - RVol90d',
        hovermode='closest',
        width=1400,  # Wider to accommodate labels
        height=800,  # Taller for better spacing
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            range=[df['iv_elevation'].min() - 5, df['iv_elevation'].max() + 10]
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            range=[35, 95]  # Focus on relevant range
        ),
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    # Add subtle annotation explaining the gold zone
    fig.add_annotation(
        x=df['iv_elevation'].max() - 20,
        y=85,
        text="Gold Zone = Top-right quadrant<br>(High Containment + Rich Premium)",
        showarrow=False,
        font=dict(size=9, color='darkgreen'),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="green",
        borderwidth=1,
        borderpad=4
    )
    
    # Save to HTML
    fig.write_html(save_path)
    print(f"\n✅ Quality matrix saved to: {save_path}")
    
    # Show in browser
    if show:
        fig.show()
    
    return fig


def plot_containment_comparison(df: pd.DataFrame, save_path: str = 'containment_comparison.html', show: bool = True):
    """
    Compare 45d vs 90d containment rates
    
    Args:
        df: Results dataframe
        save_path: Path to save HTML
        show: Whether to open in browser
    """
    
    fig = go.Figure()
    
    # Sort by 90d containment
    df_sorted = df.sort_values('90d_contain', ascending=False)
    
    # Add 45d containment bars
    fig.add_trace(go.Bar(
        x=df_sorted['ticker'],
        y=df_sorted['45d_contain'],
        name='45-Day',
        marker_color='lightblue',
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>45d: %{y:.1f}%<extra></extra>'
    ))
    
    # Add 90d containment bars
    fig.add_trace(go.Bar(
        x=df_sorted['ticker'],
        y=df_sorted['90d_contain'],
        name='90-Day',
        marker_color='darkblue',
        hovertemplate='<b>%{x}</b><br>90d: %{y:.1f}%<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="IC Threshold", annotation_position="right")
    
    fig.update_layout(
        title='Containment Rate Comparison (45d vs 90d)',
        xaxis_title='Ticker',
        yaxis_title='Containment Rate (%)',
        barmode='group',
        width=1400,
        height=600,
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif")
    )
    
    fig.write_html(save_path)
    print(f"✅ Containment comparison saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig


def plot_iv_elevation_distribution(df: pd.DataFrame, save_path: str = 'iv_elevation_dist.html', show: bool = True):
    """
    Show distribution of IV elevation across all tickers
    
    Args:
        df: Results dataframe
        save_path: Path to save HTML
        show: Whether to open in browser
    """
    
    # Sort by IV elevation
    df_sorted = df.sort_values('iv_elevation', ascending=False)
    
    fig = go.Figure()
    
    # Color code by strategy with better colors
    colors = []
    for strategy in df_sorted['strategy']:
        if 'IC' in strategy:
            colors.append('green')
        elif 'BIAS' in strategy:
            colors.append('#4169E1')  # Royal blue
        else:
            colors.append('lightgray')
    
    fig.add_trace(go.Bar(
        x=df_sorted['ticker'],
        y=df_sorted['iv_elevation'],
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=0.5,
        hovertemplate='<b>%{x}</b><br>' +
                     'IV Elevation: %{y:.1f}%<br>' +
                     'Strategy: ' + df_sorted['strategy'] + '<br>' +
                     '<extra></extra>'
    ))
    
    # Simplified reference lines
    fig.add_hline(y=15, line_dash="dash", line_color="green", line_width=2,
                  annotation_text="Rich (+15%)", annotation_position="right")
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1.5,
                  annotation_text="Fair Value", annotation_position="right")
    
    # Subtle shaded zones
    fig.add_hrect(y0=15, y1=df['iv_elevation'].max() + 10, 
                  fillcolor="green", opacity=0.08, layer="below")
    
    fig.update_layout(
        title='IV Elevation Distribution',
        xaxis_title='Ticker',
        yaxis_title='IV Elevation (%)',
        width=1400,
        height=600,
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif")
    )
    
    fig.write_html(save_path)
    print(f"✅ IV elevation distribution saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig