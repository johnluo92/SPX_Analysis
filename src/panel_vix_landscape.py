"""
Panel 2: Volatility Landscape

Displays:
- Historical VIX levels
- Current VIX marker
- VIX regime threshold lines (15, 25, 35)
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict


class VolatilityLandscapePanel:
    """Generates the volatility landscape visualization panel."""
    
    def __init__(self):
        self.colors = {
            'vix_current': '#2E86AB',
            'vix_rising': '#E74C3C',
            'vix_falling': '#2ECC71',
        }
    
    def add_to_figure(self,
                      fig: go.Figure,
                      vix_data: pd.Series,
                      lookback_days: int = 180,
                      row: int = 2,
                      col: int = 1) -> go.Figure:
        """
        Add volatility landscape panel to figure.
        
        Args:
            fig: Plotly figure object
            vix_data: Series with VIX values
            lookback_days: Days of VIX history to show
            row: Subplot row
            col: Subplot column
        """
        # Get historical data
        vix_historical_dates = vix_data.index[-lookback_days:]
        vix_historical_values = vix_data.iloc[-lookback_days:]
        current_vix = vix_data.iloc[-1]
        
        # Calculate trend
        vix_5d_ago = vix_data.iloc[-6] if len(vix_data) >= 6 else current_vix
        vix_trend = current_vix - vix_5d_ago
        
        # Historical VIX line
        fig.add_trace(go.Scatter(
            x=vix_historical_dates,
            y=vix_historical_values,
            mode='lines',
            name='VIX Historical',
            line=dict(color=self.colors['vix_current'], width=2),
            hovertemplate='VIX: %{y:.2f}<extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)',
            showlegend=False,
        ), row=row, col=col)
        
        # Current VIX marker
        fig.add_trace(go.Scatter(
            x=[vix_historical_dates[-1]],
            y=[current_vix],
            mode='markers+text',
            name='Current VIX',
            marker=dict(
                size=12,
                color=self.colors['vix_rising'] if vix_trend > 0 else self.colors['vix_falling'],
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=[f"{current_vix:.1f}"],
            textposition="top center",
            textfont=dict(size=12, color='black'),
            hovertemplate=f'Current VIX: {current_vix:.2f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        # VIX regime threshold lines
        fig.add_hline(
            y=15,
            line=dict(color='gray', width=1, dash='dot'),
            annotation_text="Min Trading (15)",
            annotation_position="right",
            row=row, col=col
        )
        
        fig.add_hline(
            y=25,
            line=dict(color='orange', width=1, dash='dot'),
            annotation_text="Elevated (25)",
            annotation_position="right",
            row=row, col=col
        )
        
        fig.add_hline(
            y=35,
            line=dict(color='red', width=1, dash='dot'),
            annotation_text="High Risk (35)",
            annotation_position="right",
            row=row, col=col
        )
        
        return fig
    
    def create_legend(self,
                      fig: go.Figure,
                      current_vix: float,
                      vix_trend: float,
                      row: int = 2,
                      col: int = 2) -> go.Figure:
        """Create legend panel for this subplot."""
        legend_text = (
            f"<b>Legend - Panel 2</b><br><br>"
            f"<span style='color:#2E86AB'>━━━</span> VIX Historical<br>"
            f"<span style='color:{'#E74C3C' if vix_trend > 0 else '#2ECC71'}'>●</span> Current VIX ({current_vix:.1f})<br>"
            f"<span style='color:gray'>· · ·</span> Min Trading (15)<br>"
            f"<span style='color:orange'>· · ·</span> Elevated (25)<br>"
            f"<span style='color:red'>· · ·</span> High Risk (35)<br>"
        )
        
        fig.add_annotation(
            xref=f'x{row*2}',
            yref=f'y{row*2}',
            x=0.5, y=0.5,
            text=legend_text,
            showarrow=False,
            xanchor='center',
            yanchor='middle',
            align='left',
            font=dict(size=10),
            bordercolor='#2E86AB',
            borderwidth=2,
            borderpad=10,
            bgcolor='rgba(255, 255, 255, 0.95)',
        )
        
        return fig