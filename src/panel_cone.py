"""
Panel 1: SPX VIX-Implied Probability Cone

Displays:
- Historical SPX price action
- VIX-implied probability cones (1σ, 2σ)
- Volatility scenario cones (VIX expansion/contraction)
- Strike recommendations for spreads
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm


class ProbabilityConePanel:
    """Generates the probability cone visualization panel."""
    
    def __init__(self):
        self.colors = {
            'spx_historical': '#2E86AB',
            'spx_future': '#8B5A9B',
            'cone_1sigma': 'rgba(139, 90, 155, 0.15)',
            'cone_2sigma': 'rgba(139, 90, 155, 0.08)',
            'cone_expanded': 'rgba(231, 76, 60, 0.12)',
            'cone_contracted': 'rgba(46, 204, 113, 0.12)',
            'strike_short': '#F18F01',
            'strike_long': '#C73E1D',
        }
    
    def calculate_cone(self, 
                       current_price: float,
                       vix: float,
                       days_forward: int = 14,
                       std_devs: List[float] = [1, 2]) -> Dict[str, pd.Series]:
        """Calculate VIX-implied probability cone."""
        daily_vol = vix / 100 / np.sqrt(252)
        
        dates = pd.date_range(
            start=datetime.now(),
            periods=days_forward + 1,
            freq='D'
        )
        
        cone_bounds = {}
        
        for std in std_devs:
            upper_bounds = []
            lower_bounds = []
            
            for day in range(days_forward + 1):
                days_vol = daily_vol * np.sqrt(day) if day > 0 else 0
                move = current_price * days_vol * std
                
                upper_bounds.append(current_price + move)
                lower_bounds.append(current_price - move)
            
            cone_bounds[f'{std}sigma_upper'] = pd.Series(upper_bounds, index=dates)
            cone_bounds[f'{std}sigma_lower'] = pd.Series(lower_bounds, index=dates)
            cone_bounds[f'{std}sigma_expected'] = pd.Series(
                [current_price] * (days_forward + 1), 
                index=dates
            )
        
        return cone_bounds
    
    def calculate_strikes(self,
                          current_spx: float,
                          current_vix: float,
                          forward_days: int,
                          strike_std_dev: float,
                          wing_width: float) -> Dict:
        """Calculate suggested strike prices."""
        daily_vol = current_vix / 100 / np.sqrt(252)
        move_vol = daily_vol * np.sqrt(forward_days)
        expected_move = current_spx * move_vol
        
        short_strike = current_spx - (strike_std_dev * expected_move)
        short_strike = round(short_strike / 5) * 5
        long_strike = short_strike - wing_width
        
        # Calculate PoP
        z = (short_strike - current_spx) / (current_spx * move_vol)
        prob_itm = norm.cdf(z) * 100
        prob_otm = 100 - prob_itm
        
        # Estimate credit
        credit = self._estimate_credit(wing_width, current_vix, forward_days)
        max_loss = wing_width - credit
        
        return {
            'short_strike': short_strike,
            'long_strike': long_strike,
            'credit': credit,
            'max_loss': max_loss,
            'pop': prob_otm
        }
    
    def _estimate_credit(self, wing_width: float, vix: float, dte: int) -> float:
        """Estimate credit for the spread."""
        if dte >= 30:
            base_pct = 0.22
        elif dte >= 14:
            base_pct = 0.18
        elif dte >= 7:
            base_pct = 0.14
        else:
            base_pct = 0.12
        
        vix_multiplier = min(vix / 20, 2.0)
        credit = wing_width * base_pct * vix_multiplier
        credit = max(credit, wing_width * 0.10)
        
        return credit
    
    def add_to_figure(self,
                      fig: go.Figure,
                      spx_data: pd.DataFrame,
                      current_vix: float,
                      lookback_days: int = 30,
                      forward_days: int = 14,
                      show_strikes: bool = True,
                      strike_std_dev: float = 0.85,
                      wing_width: float = 25,
                      show_vol_scenarios: bool = True,
                      row: int = 1,
                      col: int = 1) -> go.Figure:
        """
        Add probability cone panel to figure.
        
        Args:
            fig: Plotly figure object
            spx_data: DataFrame with SPX prices
            current_vix: Current VIX level
            lookback_days: Days of historical data
            forward_days: Days to project forward
            show_strikes: Show suggested strikes
            strike_std_dev: Std devs for short strike
            wing_width: Spread width
            show_vol_scenarios: Show VIX expansion/contraction
            row: Subplot row
            col: Subplot column
        """
        # Get current values
        historical_dates = spx_data.index[-lookback_days:]
        historical_prices = spx_data['Close'].iloc[-lookback_days:]
        current_spx = historical_prices.iloc[-1]
        
        # Historical SPX
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_prices,
            mode='lines',
            name='SPX Historical',
            line=dict(color=self.colors['spx_historical'], width=2.5),
            hovertemplate='SPX: %{y:.2f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        # Today marker
        fig.add_trace(go.Scatter(
            x=[historical_dates[-1]],
            y=[current_spx],
            mode='markers',
            name='Today',
            marker=dict(
                size=14,
                color='black',
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            hovertemplate=f'<b>TODAY</b><br>SPX: ${current_spx:.2f}<br>VIX: {current_vix:.1f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        # Calculate and plot cone
        cone = self.calculate_cone(current_spx, current_vix, forward_days)
        cone_dates = cone['2sigma_upper'].index
        
        # 2-sigma cone
        fig.add_trace(go.Scatter(
            x=cone_dates,
            y=cone['2sigma_upper'],
            mode='lines',
            name='±2σ Range',
            line=dict(color=self.colors['spx_future'], width=1, dash='dot'),
            hovertemplate='Upper 2σ: %{y:.2f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=cone_dates,
            y=cone['2sigma_lower'],
            mode='lines',
            line=dict(color=self.colors['spx_future'], width=1, dash='dot'),
            fill='tonexty',
            fillcolor=self.colors['cone_2sigma'],
            hovertemplate='Lower 2σ: %{y:.2f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        # 1-sigma cone
        fig.add_trace(go.Scatter(
            x=cone_dates,
            y=cone['1sigma_upper'],
            mode='lines',
            name='±1σ Range (68%)',
            line=dict(color=self.colors['spx_future'], width=2),
            hovertemplate='Upper 1σ: %{y:.2f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=cone_dates,
            y=cone['1sigma_lower'],
            mode='lines',
            line=dict(color=self.colors['spx_future'], width=2),
            fill='tonexty',
            fillcolor=self.colors['cone_1sigma'],
            hovertemplate='Lower 1σ: %{y:.2f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        # Expected path
        fig.add_trace(go.Scatter(
            x=cone_dates,
            y=cone['1sigma_expected'],
            mode='lines',
            name='No Move',
            line=dict(color='gray', width=1.5, dash='dash'),
            hovertemplate='Expected: %{y:.2f}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        # Volatility scenarios
        vix_high = current_vix + 10
        vix_low = max(current_vix - 5, 12)
        
        if show_vol_scenarios:
            # VIX expansion
            cone_high = self.calculate_cone(current_spx, vix_high, forward_days, std_devs=[1])
            
            fig.add_trace(go.Scatter(
                x=cone_dates,
                y=cone_high['1sigma_upper'],
                mode='lines',
                name=f'If VIX→{vix_high:.0f}',
                line=dict(color='rgba(231, 76, 60, 0.7)', width=1.5, dash='dash'),
                hovertemplate=f'VIX {vix_high:.0f}: %{{y:.2f}}<extra></extra>',
                showlegend=False,
            ), row=row, col=col)
            
            fig.add_trace(go.Scatter(
                x=cone_dates,
                y=cone_high['1sigma_lower'],
                mode='lines',
                line=dict(color='rgba(231, 76, 60, 0.7)', width=1.5, dash='dash'),
                fill='tonexty',
                fillcolor=self.colors['cone_expanded'],
                hovertemplate=f'VIX {vix_high:.0f}: %{{y:.2f}}<extra></extra>',
                showlegend=False,
            ), row=row, col=col)
            
            # VIX contraction (if applicable)
            if current_vix > 20:
                cone_low = self.calculate_cone(current_spx, vix_low, forward_days, std_devs=[1])
                
                fig.add_trace(go.Scatter(
                    x=cone_dates,
                    y=cone_low['1sigma_upper'],
                    mode='lines',
                    name=f'If VIX→{vix_low:.0f}',
                    line=dict(color='rgba(46, 204, 113, 0.8)', width=1.5, dash='dash'),
                    hovertemplate=f'VIX {vix_low:.0f}: %{{y:.2f}}<extra></extra>',
                    showlegend=False,
                ), row=row, col=col)
                
                fig.add_trace(go.Scatter(
                    x=cone_dates,
                    y=cone_low['1sigma_lower'],
                    mode='lines',
                    line=dict(color='rgba(46, 204, 113, 0.8)', width=1.5, dash='dash'),
                    hovertemplate=f'VIX {vix_low:.0f}: %{{y:.2f}}<extra></extra>',
                    showlegend=False,
                ), row=row, col=col)
        
        # Add strikes
        if show_strikes:
            strikes = self.calculate_strikes(
                current_spx, current_vix, forward_days, strike_std_dev, wing_width
            )
            
            # Short strike line
            fig.add_hline(
                y=strikes['short_strike'],
                line=dict(color=self.colors['strike_short'], width=2, dash='dash'),
                annotation_text=f"Short ${strikes['short_strike']:.0f}",
                annotation_position="left",
                annotation=dict(font=dict(size=11)),
                row=row, col=col
            )
            
            # Long strike line
            fig.add_hline(
                y=strikes['long_strike'],
                line=dict(color=self.colors['strike_long'], width=1.5, dash='dot'),
                annotation_text=f"Long ${strikes['long_strike']:.0f}",
                annotation_position="left",
                annotation=dict(font=dict(size=10, color='gray')),
                row=row, col=col
            )
            
            # Trade details box
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref=f'x{row if row > 1 else ""} domain',
                yref=f'y{row if row > 1 else ""} domain',
                text=(
                    f"<b>Bull Put Spread ({forward_days} DTE)</b><br>"
                    f"<br>"
                    f"Short ${strikes['short_strike']:.0f} / Long ${strikes['long_strike']:.0f}<br>"
                    f"Width: ${wing_width:.0f}<br>"
                    f"<br>"
                    f"Est. Credit: <b>${strikes['credit']:.2f}</b><br>"
                    f"Max Risk: ${strikes['max_loss']:.2f}<br>"
                    f"<br>"
                    f"PoP: <b>{strikes['pop']:.0f}%</b>"
                ),
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#2E86AB',
                borderwidth=2,
                align='left',
                font=dict(size=11),
                xanchor='left',
                yanchor='top',
            )
        
        return fig
    
    def create_legend(self,
                      fig: go.Figure,
                      current_vix: float,
                      show_vol_scenarios: bool = True,
                      row: int = 1,
                      col: int = 2) -> go.Figure:
        """Create legend panel for this subplot."""
        vix_high = current_vix + 10
        vix_low = max(current_vix - 5, 12)
        
        legend_text = (
            f"<b>Legend - Panel 1</b><br><br>"
            f"<span style='color:#2E86AB'>━━━</span> SPX Historical<br>"
            f"<span style='color:black'>●</span> Today<br>"
            f"<span style='color:#8B5A9B'>━━━</span> ±1σ Range (68%)<br>"
            f"<span style='color:#8B5A9B'>· · ·</span> ±2σ Range<br>"
            f"<span style='color:gray'>- - -</span> No Move<br>"
        )
        
        if show_vol_scenarios:
            legend_text += f"<span style='color:rgba(231,76,60,0.7)'>- - -</span> If VIX→{vix_high:.0f}<br>"
            if current_vix > 20:
                legend_text += f"<span style='color:rgba(46,204,113,0.8)'>- - -</span> If VIX→{vix_low:.0f}<br>"
        
        fig.add_annotation(
            xref=f'x{2 if row == 1 else row*2}',
            yref=f'y{2 if row == 1 else row*2}',
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