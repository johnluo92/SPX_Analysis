"""
Panel 3: Implied Vol vs Realized Vol

Displays:
- Historical VIX (implied volatility)
- Realized volatility (30-day forward)
- The "volatility risk premium" spread
- Research findings and statistics
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict

from UnifiedDataFetcher import UnifiedDataFetcher


class IVvsRVPanel:
    """Generates the IV vs RV comparison panel."""
    
    def __init__(self):
        self.fetcher = UnifiedDataFetcher()
        self.colors = {
            'iv_line': '#2E86AB',
            'rv_line': '#F18F01',
        }
    
    def calculate_iv_rv_spread(self, lookback_years: int = 5) -> pd.DataFrame:
        """
        Calculate historical IV vs subsequent realized vol.
        
        Args:
            lookback_years: Number of years of history to analyze
        
        Returns:
            DataFrame with date, vix, realized_30d, spread columns
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365 + 60)
        
        print(f"📊 Calculating IV vs RV spread ({lookback_years} years)...")
        spx_data = self.fetcher.fetch_spx(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        vix_data = self.fetcher.fetch_vix(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        results = []
        
        for i in range(len(vix_data) - 30):
            date = vix_data.index[i]
            vix_value = vix_data.iloc[i]
            
            future_slice = spx_data['Close'].loc[spx_data.index >= date]
            if len(future_slice) >= 30:
                future_prices = future_slice.iloc[:30]
                
                # Calculate realized vol
                returns = np.log(future_prices / future_prices.shift(1))
                realized = returns.std() * np.sqrt(252) * 100
                
                results.append({
                    'date': date,
                    'vix': vix_value,
                    'realized_30d': realized,
                    'spread': vix_value - realized
                })
        
        print(f"   Calculated {len(results)} IV/RV data points")
        return pd.DataFrame(results)
    
    def calculate_statistics(self,
                            iv_rv_data: pd.DataFrame,
                            current_vix: float) -> Dict:
        """Calculate statistics for the research findings."""
        # Overall statistics
        overall_premium_count = (iv_rv_data['spread'] > 0).sum()
        overall_premium_pct = (overall_premium_count / len(iv_rv_data)) * 100
        overall_avg_spread = iv_rv_data['spread'].mean()
        
        # High VIX periods (VIX > 30)
        high_vix_periods = iv_rv_data[iv_rv_data['vix'] > 30]
        high_vix_premium_pct = ((high_vix_periods['spread'] > 0).sum() / len(high_vix_periods) * 100) if len(high_vix_periods) > 0 else 0
        
        # Normal VIX periods (VIX 15-25)
        normal_vix_periods = iv_rv_data[(iv_rv_data['vix'] >= 15) & (iv_rv_data['vix'] <= 25)]
        normal_vix_premium_pct = ((normal_vix_periods['spread'] > 0).sum() / len(normal_vix_periods) * 100) if len(normal_vix_periods) > 0 else 0
        
        # Current VIX regime (±2 points)
        similar_periods = iv_rv_data[
            (iv_rv_data['vix'] >= current_vix - 2) & 
            (iv_rv_data['vix'] <= current_vix + 2)
        ]
        
        if len(similar_periods) > 0:
            avg_realized = similar_periods['realized_30d'].mean()
            premium_count = (similar_periods['spread'] > 0).sum()
            premium_pct = (premium_count / len(similar_periods)) * 100
            expansion_pct = 100 - premium_pct
            avg_spread = similar_periods['spread'].mean()
        else:
            avg_realized = 0
            premium_pct = 0
            expansion_pct = 0
            avg_spread = 0
        
        return {
            'overall_premium_pct': overall_premium_pct,
            'overall_avg_spread': overall_avg_spread,
            'high_vix_premium_pct': high_vix_premium_pct,
            'normal_vix_premium_pct': normal_vix_premium_pct,
            'current_avg_realized': avg_realized,
            'current_premium_pct': premium_pct,
            'current_expansion_pct': expansion_pct,
            'current_avg_spread': avg_spread,
            'current_periods_analyzed': len(similar_periods)
        }
    
    def add_to_figure(self,
                      fig: go.Figure,
                      iv_rv_data: pd.DataFrame,
                      row: int = 3,
                      col: int = 1) -> go.Figure:
        """
        Add IV vs RV panel to figure.
        
        Args:
            fig: Plotly figure object
            iv_rv_data: DataFrame with IV/RV data
            row: Subplot row
            col: Subplot column
        """
        if len(iv_rv_data) == 0:
            return fig
        
        # Plot VIX (Implied Vol)
        fig.add_trace(go.Scatter(
            x=iv_rv_data['date'],
            y=iv_rv_data['vix'],
            mode='lines',
            name='Implied Vol (VIX)',
            line=dict(color=self.colors['iv_line'], width=2),
            hovertemplate='IV: %{y:.1f}%<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        # Plot Realized Vol with fill
        fig.add_trace(go.Scatter(
            x=iv_rv_data['date'],
            y=iv_rv_data['realized_30d'],
            mode='lines',
            name='Realized Vol (30d)',
            line=dict(color=self.colors['rv_line'], width=2),
            fill='tonexty',
            fillcolor='rgba(255, 200, 100, 0.12)',
            hovertemplate='RV: %{y:.1f}%<extra></extra>',
            showlegend=False,
        ), row=row, col=col)
        
        return fig