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
    
    # Import strategy functions to calculate separate 45d/90d patterns
    from ..calculations.strategy import determine_strategy_45, determine_strategy_90
    
    # Calculate 45d and 90d strategies separately for each row
    strategies_45d = []
    strategies_90d = []
    
    for _, row in df.iterrows():
        # FIXED: Updated to use new column names
        stats_45 = {
            'containment': row['45d_contain'],
            'breaks_up': row['45d_breaks_up'],
            'breaks_down': row['45d_breaks_dn'],
            'break_up_pct': row['45d_break_up_pct'],  # RENAMED from break_bias
            'trend_pct': row['45d_trend_pct'],         # RENAMED from overall_bias
            'drift_pct': row['45d_drift']              # RENAMED from avg_move_pct
        }
        stats_90 = {
            'containment': row['90d_contain'],
            'breaks_up': row['90d_breaks_up'],
            'breaks_down': row['90d_breaks_dn'],
            'break_up_pct': row['90d_break_up_pct'],  # RENAMED from break_bias
            'trend_pct': row['90d_trend_pct'],         # RENAMED from overall_bias
            'drift_pct': row['90d_drift']              # RENAMED from avg_move_pct
        }
        
        pattern_45, _ = determine_strategy_45(stats_45)
        pattern_90, _ = determine_strategy_90(stats_90)
        
        strategies_45d.append(pattern_45)
        strategies_90d.append(pattern_90)
    
    # Add as temporary columns
    df = df.copy()
    df['strategy_45d'] = strategies_45d
    df['strategy_90d'] = strategies_90d
    
    # Separate by strategy type using the new columns
    # Each ticker can only belong to ONE category per timeframe
    df_ic45 = df[df['strategy_45d'].str.contains('IC45', na=False)]
    df_bias45 = df[(df['strategy_45d'].str.contains('BIAS', na=False)) & 
                   (~df['strategy_45d'].str.contains('IC', na=False))]
    
    df_ic90 = df[df['strategy_90d'].str.contains('IC90', na=False)]
    df_bias90 = df[(df['strategy_90d'].str.contains('BIAS', na=False)) & 
                   (~df['strategy_90d'].str.contains('IC', na=False))]
    
    # Create side-by-side subplots with reasonable spacing
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('45-Day Containment', '90-Day Containment'),
        horizontal_spacing=0.08
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
            customdata=df_ic45['strategy_45d'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '45d Containment: %{y:.1f}%<br>' +
                         '%{customdata}<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=1)
    
    # BIAS45 patterns (blue)
    if not df_bias45.empty:
        bias_patterns_45 = []
        for _, row in df_bias45.iterrows():
            pattern = row['strategy_45d']
            if 'BIAS' in pattern:
                bias_part = pattern.split('BIAS')[1].split('[')[0].strip()
                bias_patterns_45.append(f"BIAS{bias_part}")
            else:
                bias_patterns_45.append("BIAS detected")
        
        fig.add_trace(go.Scatter(
            x=df_bias45['current_iv'],
            y=df_bias45['45d_contain'],
            mode='markers+text',
            name='BIAS45',
            marker=dict(size=10, color='blue', opacity=0.7, line=dict(width=1, color='darkblue')),
            text=df_bias45['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            customdata=bias_patterns_45,
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '45d Containment: %{y:.1f}%<br>' +
                         '%{customdata}<br>' +
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
            customdata=df_ic90['strategy_90d'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '%{customdata}<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=2)
    
    # BIAS90 patterns (blue)
    if not df_bias90.empty:
        bias_patterns_90 = []
        for _, row in df_bias90.iterrows():
            pattern = row['strategy_90d']
            if 'BIAS' in pattern:
                bias_part = pattern.split('BIAS')[1].split('[')[0].strip()
                bias_patterns_90.append(f"BIAS{bias_part}")
            else:
                bias_patterns_90.append("BIAS detected")
        
        fig.add_trace(go.Scatter(
            x=df_bias90['current_iv'],
            y=df_bias90['90d_contain'],
            mode='markers+text',
            name='BIAS90',
            marker=dict(size=10, color='blue', opacity=0.7, line=dict(width=1, color='darkblue')),
            text=df_bias90['ticker'],
            textposition='top center',
            textfont=dict(size=9),
            customdata=bias_patterns_90,
            hovertemplate='<b>%{text}</b><br>' +
                         'Current IV: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         '%{customdata}<br>' +
                         '<extra></extra>',
            showlegend=True
        ), row=1, col=2)
    
    # Add threshold lines
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", opacity=0.5,
                  row=1, col=1)
    fig.add_hline(y=69.5, line_dash="dash", line_color="red", opacity=0.5,
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
                     side='right',
                     row=1, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"\nQuality matrix saved to: {save_path}")
    
    if show:
        fig.show()
    
    return fig