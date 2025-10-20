"""
VIX Visualizer v3.0 - Regime Performance Edition

NEW IN v3.0:
- Panel 3 is now Regime Performance Matrix (actionable!)
- Shows SPX forward returns by regime
- Real edge calculations from 35 years of data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from UnifiedDataFetcher import UnifiedDataFetcher
from panel_cone import ProbabilityConePanel
from panel_regime_performance import RegimePerformancePanel
from panel_iv_rv import IVvsRVPanel
from regime_analyzer import RegimeAnalyzer


class VisualizationConfig:
    """Centralized configuration."""
    
    LAYOUT = {
        'width': 1600,
        'height': 1400,
        'row_heights': [0.15, 0.35, 0.20, 0.30],  # Summary, Cone, Performance, IV/RV
        'vertical_spacing': 0.08,
        'template': 'plotly_white',
        'hovermode': 'x unified',
    }
    
    DATA = {
        'lookback_days': 30,
        'forward_days': 14,
        'iv_rv_years': 5,
        'regime_history_years': 35,  # Full history for regime stats
    }


class ConeVisualizer:
    """Visualizer with regime performance matrix."""
    
    def __init__(self, config=None):
        self.config = config or VisualizationConfig()
        self.fetcher = UnifiedDataFetcher()
        self.cone_panel = ProbabilityConePanel()
        self.performance_panel = RegimePerformancePanel()
        self.iv_rv_panel = IVvsRVPanel()
        self.regime_analyzer = RegimeAnalyzer()
    
    def plot_decision_chart(self,
                           lookback_days: int = None,
                           forward_days: int = None,
                           iv_rv_years: int = None):
        """Create the complete decision chart."""
        
        lookback_days = lookback_days or self.config.DATA['lookback_days']
        forward_days = forward_days or self.config.DATA['forward_days']
        iv_rv_years = iv_rv_years or self.config.DATA['iv_rv_years']
        
        print("\n" + "="*60)
        print("BUILDING VIX VISUALIZATION v3.0")
        print("Regime Performance Matrix Edition")
        print("="*60)
        
        # ==================== FETCH DATA ====================
        
        end_date = datetime.now()
        regime_start = datetime(1990, 1, 1)  # 35 years for robust stats
        
        print(f"\n📊 Fetching data...")
        
        # SPX for cone (recent) and performance (full history)
        spx_recent = self.fetcher.fetch_spx(
            (end_date - timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        spx_full = self.fetcher.fetch_spx_close(
            regime_start.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # VIX full history
        vix_full = self.fetcher.fetch_vix(
            regime_start.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"   ✅ SPX: {len(spx_full)} days (1990-2025)")
        print(f"   ✅ VIX: {len(vix_full)} days (1990-2025)")
        
        # Current values
        current_spx = spx_recent['Close'].iloc[-1]
        current_vix = vix_full.iloc[-1]
        current_regime = self.regime_analyzer.classify_regime(current_vix)
        velocity = self.regime_analyzer.calculate_velocity(vix_full).iloc[-1]
        
        # Calculate regime statistics
        print(f"\n🔧 Calculating regime statistics...")
        regime_stats = self.regime_analyzer.analyze_transitions(vix_full)
        
        # Calculate regime performance (THIS IS THE NEW PART)
        print(f"🔧 Calculating regime performance matrix...")
        performance_stats = self.performance_panel.calculate_regime_performance(
            spx_full, vix_full
        )
        
        # IV/RV stats
        print(f"🔧 Calculating IV vs RV...")
        iv_rv_data = self.iv_rv_panel.calculate_iv_rv_spread(lookback_years=iv_rv_years)
        iv_stats = self.iv_rv_panel.calculate_statistics(iv_rv_data, current_vix)
        
        print(f"✅ All calculations complete")
        
        # ==================== CREATE FIGURE ====================
        
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=self.config.LAYOUT['row_heights'],
            vertical_spacing=self.config.LAYOUT['vertical_spacing'],
            subplot_titles=[
                "",  # Row 1: Summary table
                f"<b>SPX Probability Cone</b> (VIX {current_vix:.1f})",
                f"<b>SPX Performance by Regime</b> (35 Years)",
                f"<b>Implied vs Realized Volatility</b> ({iv_rv_years}Y History)"
            ],
            specs=[
                [{"type": "table"}],     # Row 1: Summary
                [{"type": "scatter"}],   # Row 2: Cone
                [{"type": "table"}],     # Row 3: Performance Matrix (NEW!)
                [{"type": "scatter"}]    # Row 4: IV/RV
            ]
        )
        
        # ==================== ROW 1: SUMMARY TABLE ====================
        
        print("📊 Building summary table...")
        self._add_summary_table(
            fig, current_spx, current_vix, current_regime, velocity, 
            regime_stats, iv_stats, performance_stats, forward_days
        )
        
        # ==================== ROW 2: PROBABILITY CONE ====================
        
        print("📊 Adding probability cone...")
        fig = self.cone_panel.add_to_figure(
            fig=fig,
            spx_data=spx_recent,
            current_vix=current_vix,
            lookback_days=lookback_days,
            forward_days=forward_days,
            show_strikes=False,
            show_vol_scenarios=True,
            row=2, col=1
        )
        
        # ==================== ROW 3: REGIME PERFORMANCE MATRIX ====================
        
        print("📊 Adding regime performance matrix...")
        fig = self.performance_panel.add_to_figure(
            fig=fig,
            performance_stats=performance_stats,
            current_regime=current_regime,
            row=3, col=1
        )
        
        # ==================== ROW 4: IV vs RV ====================
        
        print("📊 Adding IV vs RV...")
        fig = self.iv_rv_panel.add_to_figure(
            fig=fig,
            iv_rv_data=iv_rv_data,
            row=4, col=1
        )
        
        # ==================== FINAL LAYOUT ====================
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        fig.update_yaxes(title_text="SPX Price", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=4, col=1)
        
        fig.update_layout(
            width=self.config.LAYOUT['width'],
            height=self.config.LAYOUT['height'],
            showlegend=False,
            hovermode=self.config.LAYOUT['hovermode'],
            template=self.config.LAYOUT['template'],
            title=dict(
                text=f"<b>SPX/VIX Decision Dashboard</b><br><sup>{datetime.now().strftime('%Y-%m-%d %H:%M')}</sup>",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            )
        )
        
        print("✅ Visualization complete!\n")
        return fig
    
    def _add_summary_table(self, fig, current_spx, current_vix, current_regime, 
                          velocity, regime_stats, iv_stats, performance_stats, forward_days):
        """Add summary table at the top."""
        
        regime_info = regime_stats[current_regime]
        perf_info = performance_stats[current_regime]['21d']
        
        # Velocity label
        if abs(velocity) < 2:
            vel_label = "Stable"
        elif velocity >= 2:
            vel_label = "Rising Fast ⚠️"
        else:
            vel_label = "Falling Fast"
        
        # Top transitions
        top_trans = sorted(regime_info['transitions'].items(), key=lambda x: x[1], reverse=True)[:3]
        trans_text = " | ".join([f"{r}: {p*100:.0f}%" for r, p in top_trans if p > 0.01])
        
        # Build table
        fig.add_trace(go.Table(
            header=dict(
                values=[
                    "<b>CURRENT MARKET</b>",
                    "<b>VIX REGIME</b>",
                    "<b>EXPECTED RETURNS</b>",
                    "<b>TRADING OUTLOOK</b>"
                ],
                fill_color='#2E86AB',
                font=dict(color='white', size=12),
                align='left',
                height=30
            ),
            cells=dict(
                values=[
                    [
                        f"<b>SPX:</b> ${current_spx:,.0f}",
                        f"<b>VIX:</b> {current_vix:.2f}",
                        f"<b>Δ5d:</b> {velocity:+.2f} ({vel_label})",
                        f"<b>Cone:</b> {forward_days} days"
                    ],
                    [
                        f"<b>Regime:</b> {current_regime}",
                        f"<b>Duration:</b> {regime_info['avg_duration_days']:.0f}d avg",
                        f"<b>Frequency:</b> {regime_info['pct_of_time']:.0f}% of time",
                        f"<b>Next:</b> {trans_text}"
                    ],
                    [
                        f"<b>1-Month Avg:</b> {perf_info['avg_return']:+.2f}%",
                        f"<b>Win Rate:</b> {perf_info['win_rate']:.0f}%",
                        f"<b>Avg Win:</b> {perf_info['avg_win']:+.2f}%",
                        f"<b>Avg Loss:</b> {perf_info['avg_loss']:.2f}%"
                    ],
                    [
                        f"<b>Strategy:</b> {self.performance_panel._determine_action(current_regime, performance_stats[current_regime])}",
                        f"<b>IV>RV:</b> {iv_stats['current_premium_pct']:.0f}% likely",
                        f"<b>Vol Premium:</b> {iv_stats['current_avg_spread']:+.2f}%",
                        f"<b>Watch:</b> VIX {self.regime_analyzer.normal_threshold:.0f} (regime shift)"
                    ]
                ],
                fill_color=[['#E8F4F8', 'white', '#E8F4F8', 'white']],
                align='left',
                font=dict(size=11),
                height=25
            )
        ), row=1, col=1)
    
    def show_chart(self, **kwargs):
        """Create and display the chart."""
        fig = self.plot_decision_chart(**kwargs)
        fig.show()
    
    def save_chart(self, filename: str = 'spx_dashboard_v3.0.html', **kwargs):
        """Create and save the chart to HTML."""
        fig = self.plot_decision_chart(**kwargs)
        fig.write_html(filename)
        print(f"💾 Chart saved to {filename}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPX/VIX DECISION DASHBOARD v3.0")
    print("Now With Regime Performance Matrix")
    print("="*60)
    
    viz = ConeVisualizer()
    viz.show_chart()
    viz.save_chart()
    
    print("\n✅ DONE!")