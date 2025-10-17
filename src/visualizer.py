"""
VIX Probability Cone Visualizer - ORCHESTRATOR

This file orchestrates the creation of the full visualization by combining
individual panel modules. Each panel can be modified independently.

Panels:
- Panel 1: Probability Cone (panel_cone.py)
- Panel 2: Volatility Landscape (panel_vix_landscape.py)  
- Panel 3: IV vs Realized Vol (panel_iv_rv.py)
"""

import pandas as pd
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

from UnifiedDataFetcher import UnifiedDataFetcher
from panel_cone import ProbabilityConePanel
from panel_vix_landscape import VolatilityLandscapePanel
from panel_iv_rv import IVvsRVPanel


class ConeVisualizer:
    """Orchestrates the creation of the multi-panel VIX visualization."""
    
    def __init__(self):
        self.fetcher = UnifiedDataFetcher()
        
        # Initialize panel modules
        self.cone_panel = ProbabilityConePanel()
        self.vix_panel = VolatilityLandscapePanel()
        self.iv_rv_panel = IVvsRVPanel()
    
    def plot_decision_chart(self,
                           lookback_days: int = 30,
                           forward_days: int = 14,
                           show_strikes: bool = True,
                           strike_std_dev: float = 0.85,
                           wing_width: float = 25,
                           show_vol_scenarios: bool = True,
                           iv_rv_years: int = 5,
                           vix_lookback_days: int = 180):
        """
        Create the complete decision chart with all panels.
        
        Args:
            lookback_days: Days of historical SPX data for Panel 1
            forward_days: Days to project forward in Panel 1
            show_strikes: Show strike recommendations in Panel 1
            strike_std_dev: Std devs for short strike placement
            wing_width: Spread width in dollars
            show_vol_scenarios: Show VIX expansion/contraction scenarios
            iv_rv_years: Years of IV/RV history for Panel 3
            vix_lookback_days: Days of VIX history for Panel 2
            
        Returns:
            Plotly figure with all panels
        """
        print("\n" + "="*60)
        print("BUILDING VIX VISUALIZATION")
        print("="*60)
        
        # ==================== DATA FETCHING ====================
        
        end_date = datetime.now()
        
        # Fetch SPX data for Panel 1
        spx_start = end_date - timedelta(days=lookback_days + 10)
        print(f"\n📊 Fetching SPX data...")
        spx_data = self.fetcher.fetch_spx(
            spx_start.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Fetch VIX data for Panels 1 & 2
        vix_start = end_date - timedelta(days=max(vix_lookback_days, lookback_days) + 10)
        print(f"📊 Fetching VIX data...")
        vix_data = self.fetcher.fetch_vix(
            vix_start.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Calculate IV vs RV for Panel 3
        iv_rv_data = self.iv_rv_panel.calculate_iv_rv_spread(lookback_years=iv_rv_years)
        
        # Get current values
        current_spx = spx_data['Close'].iloc[-1]
        current_vix = vix_data.iloc[-1]
        
        print(f"\n✅ Data loaded:")
        print(f"   Current SPX: ${current_spx:.2f}")
        print(f"   Current VIX: {current_vix:.2f}")
        
        # Calculate VIX trend for Panel 2
        vix_5d_ago = vix_data.iloc[-6] if len(vix_data) >= 6 else current_vix
        vix_trend = current_vix - vix_5d_ago
        vix_trend_pct = (vix_trend / vix_5d_ago) * 100
        
        # ==================== CREATE FIGURE LAYOUT ====================
        
        print(f"\n🎨 Creating visualization layout...")
        
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.45, 0.25, 0.30],
            column_widths=[0.82, 0.18],  # Main chart 82%, legend panel 18%
            subplot_titles=(
                f"<b>SPX VIX-Implied Probability Cone</b><br>"
                f"<sup>Current: SPX ${current_spx:.2f} | VIX {current_vix:.1f} | "
                f"{forward_days}-Day Projection</sup>",
                "",  # Legend panel 1
                f"<b>Volatility Landscape</b><br>"
                f"<sup>VIX Trend: {vix_trend:+.1f} pts ({vix_trend_pct:+.1f}%) over 5 days</sup>",
                "",  # Legend panel 2
                f"<b>Implied Vol vs Realized Vol (30-Day Forward)</b><br>"
                f"<sup>{iv_rv_years}-Year History: The Volatility Risk Premium</sup>",
                ""   # Legend panel 3
            ),
            vertical_spacing=0.10,
            horizontal_spacing=0.02,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # ==================== PANEL 1: PROBABILITY CONE ====================
        
        print("🔵 Adding Panel 1: Probability Cone...")
        fig = self.cone_panel.add_to_figure(
            fig=fig,
            spx_data=spx_data,
            current_vix=current_vix,
            lookback_days=lookback_days,
            forward_days=forward_days,
            show_strikes=show_strikes,
            strike_std_dev=strike_std_dev,
            wing_width=wing_width,
            show_vol_scenarios=show_vol_scenarios,
            row=1,
            col=1
        )
        
        fig = self.cone_panel.create_legend(
            fig=fig,
            current_vix=current_vix,
            show_vol_scenarios=show_vol_scenarios,
            row=1,
            col=2
        )
        
        # ==================== PANEL 2: VOLATILITY LANDSCAPE ====================
        
        print("🔵 Adding Panel 2: Volatility Landscape...")
        fig = self.vix_panel.add_to_figure(
            fig=fig,
            vix_data=vix_data,
            lookback_days=vix_lookback_days,
            row=2,
            col=1
        )
        
        fig = self.vix_panel.create_legend(
            fig=fig,
            current_vix=current_vix,
            vix_trend=vix_trend,
            row=2,
            col=2
        )
        
        # ==================== PANEL 3: IV vs RV ====================
        
        print("🔵 Adding Panel 3: IV vs Realized Vol...")
        fig = self.iv_rv_panel.add_to_figure(
            fig=fig,
            iv_rv_data=iv_rv_data,
            row=3,
            col=1
        )
        
        # Calculate stats for legend
        stats = self.iv_rv_panel.calculate_statistics(iv_rv_data, current_vix)
        
        fig = self.iv_rv_panel.create_legend(
            fig=fig,
            current_vix=current_vix,
            stats=stats,
            iv_rv_years=iv_rv_years,
            row=3,
            col=2
        )
        
        # ==================== FINAL LAYOUT CONFIGURATION ====================
        
        # Hide axes for legend panels
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_xaxes(visible=False, row=3, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=3, col=2)
        
        # Main chart axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_yaxes(title_text="SPX Price", tickformat=',.0f', row=1, col=1)
        fig.update_yaxes(title_text="VIX Level", tickformat='.1f', row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", tickformat='.1f', row=3, col=1)
        
        fig.update_layout(
            height=1200,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white',
        )
        
        print("✅ Visualization complete!\n")
        
        return fig
    
    def show_chart(self, **kwargs):
        """Create and display the chart."""
        fig = self.plot_decision_chart(**kwargs)
        fig.show()
    
    def save_chart(self, filename: str = 'spx_cone_chart.html', **kwargs):
        """Create and save the chart to HTML."""
        fig = self.plot_decision_chart(**kwargs)
        fig.write_html(filename)
        print(f"💾 Chart saved to {filename}")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VIX PROBABILITY CONE VISUALIZER")
    print("Modular Architecture - Each Panel is Independent")
    print("="*60)
    
    viz = ConeVisualizer()
    
    # Create and show the chart
    viz.show_chart(
        lookback_days=30,           # Panel 1: SPX history
        forward_days=14,            # Panel 1: Projection days
        show_strikes=True,          # Panel 1: Show strike recommendations
        strike_std_dev=0.85,        # Panel 1: Strike placement
        wing_width=25,              # Panel 1: Spread width
        show_vol_scenarios=True,    # Panel 1: Show VIX scenarios
        vix_lookback_days=180,      # Panel 2: VIX history (6 months)
        iv_rv_years=5,              # Panel 3: IV/RV years of data
    )
    
    # Also save to file
    viz.save_chart('spx_cone_chart.html', 
                   vix_lookback_days=180, 
                   iv_rv_years=5)
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print("\n📁 Files created:")
    print("   • visualizer.py (orchestrator)")
    print("   • panel_cone.py (Panel 1)")
    print("   • panel_vix_landscape.py (Panel 2)")
    print("   • panel_iv_rv.py (Panel 3)")
    print("\n💡 To modify a panel, edit its individual file!")
    print("📊 Open 'spx_cone_chart.html' in your browser")