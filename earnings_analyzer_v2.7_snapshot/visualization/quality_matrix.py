"""Trade Quality Matrix - Interactive visualization for earnings plays"""
import plotly.graph_objects as go
import pandas as pd


def plot_quality_matrix(df: pd.DataFrame, save_path: str = None, show: bool = True):
    """
    Create interactive scatter plot showing trade quality
    
    Args:
        df: Results DataFrame from batch_analyze
        save_path: Optional path to save figure (e.g. 'quality_matrix.html')
        show: Whether to display the plot
    """
    # Filter out SKIP strategies
    df_filtered = df[df['strategy'] != 'SKIP'].copy()
    
    # Separate IC and BIAS strategies
    ic_mask = df_filtered['strategy'].str.contains('IC', na=False)
    df_ic = df_filtered[ic_mask]
    df_bias = df_filtered[~ic_mask]
    
    fig = go.Figure()
    
    # Add Iron Condor points
    if not df_ic.empty:
        fig.add_trace(go.Scatter(
            x=df_ic['iv_premium'],
            y=df_ic['90d_contain'],
            mode='markers+text',
            name='Iron Condor',
            marker=dict(size=12, color='green', opacity=0.7, line=dict(width=1, color='darkgreen')),
            text=df_ic['ticker'],
            textposition='top center',
            textfont=dict(size=9, color='darkgreen'),
            hovertemplate='<b>%{text}</b><br>' +
                         'IV Premium: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         'Strategy: ' + df_ic['strategy'] + '<br>' +
                         '<extra></extra>'
        ))
    
    # Add Directional Bias points
    if not df_bias.empty:
        fig.add_trace(go.Scatter(
            x=df_bias['iv_premium'],
            y=df_bias['90d_contain'],
            mode='markers+text',
            name='Directional Bias',
            marker=dict(size=12, color='blue', opacity=0.7, line=dict(width=1, color='darkblue')),
            text=df_bias['ticker'],
            textposition='top center',
            textfont=dict(size=9, color='darkblue'),
            hovertemplate='<b>%{text}</b><br>' +
                         'IV Premium: %{x:.1f}%<br>' +
                         '90d Containment: %{y:.1f}%<br>' +
                         'Strategy: ' + df_bias['strategy'] + '<br>' +
                         '<extra></extra>'
        ))
    
    # Add quadrant lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3)
    fig.add_vline(x=15, line_dash="dash", line_color="red", opacity=0.3)
    
    # Layout
    fig.update_layout(
        title=dict(
            text='Trade Quality Matrix - Earnings Plays',
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='IV Premium (%)',
            showgrid=False,
            zeroline=True
        ),
        yaxis=dict(
            title='Containment Rate (%)',
            showgrid=False,
            zeroline=False
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=1200,
        height=700,
        legend=dict(
            x=1,
            y=0,
            xanchor='right',
            yanchor='bottom'
        ),
        annotations=[
            dict(
                text='Gold Zone = Top-right quadrant (High Containment + Rich Premium)',
                xref='paper',
                yref='paper',
                x=1,
                y=-0.1,
                xanchor='right',
                yanchor='top',
                showarrow=False,
                font=dict(size=10, color='gray', style='italic')
            )
        ]
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"\n✓ Quality matrix saved to {save_path}")
    
    if show:
        fig.show()
    
    return fig