"""
VIX Visualizer - Clean Dashboard
Twin Pillars: Simplicity & Consistency
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from config import LOOKBACK_YEARS
from data import DataFetcher
from features import FeatureEngine
from model import SectorModel
from panel_cone import ProbabilityConePanel
from panel_sector_rotation import SectorRotationPanel
from panel_iv_rv import IVvsRVPanel
from regime_analyzer import RegimeAnalyzer


class DashboardConfig:
    """Dashboard display configuration."""
    
    LAYOUT = {
        'width': 1600,
        'height': 1850,
        'template': 'plotly_white',
        'hovermode': 'x unified',
        'vertical_spacing': 0.06,
    }
    
    FONT = {
        'family': 'Arial, sans-serif',
        'size': 12,
        'color': '#2c3e50'
    }
    
    TITLE_FONT = {
        'family': 'Arial, sans-serif',
        'size': 14,
        'color': '#2c3e50'
    }
    
    DISPLAY = {
        'lookback_days': 30,
        'forward_days': 21,
        'iv_rv_years': 5,
    }
    
    COLORS = {
        'primary': '#2E86AB',
        'accent': '#F18F01',
        'success': '#27ae60',
        'warning': '#e74c3c',
        'neutral': '#95a5a6',
        'background': '#f8f9fa',
    }


class Dashboard:
    """Clean dashboard orchestrator."""
    
    def __init__(self, config=None):
        self.config = config or DashboardConfig()
        self.data_fetcher = DataFetcher()
        self.regime_analyzer = RegimeAnalyzer()
        self.cone_panel = ProbabilityConePanel()
        self.rotation_panel = SectorRotationPanel()
        self.iv_rv_panel = IVvsRVPanel()
        
        self.model = None
        self.features = None
    
    def _validate_freshness(self, data_date, max_age_days=3):
        """Check if data is recent."""
        today = datetime.now().date()
        data_date = pd.to_datetime(data_date).date()
        age = (today - data_date).days
        
        if age > max_age_days:
            print(f"⚠️  Data is {age} days old (from {data_date})")
            return False
        return True
    
    def train_model(self):
        """Train sector rotation model."""
        print("\n🤖 Training ML Model...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch training data
        sectors = self.data_fetcher.fetch_sectors(start_str, end_str)
        macro = self.data_fetcher.fetch_macro(start_str, end_str)
        vix = self.data_fetcher.fetch_vix(start_str, end_str)
        
        sectors, macro, vix = self.data_fetcher.align(sectors, macro, vix)
        
        # Build features
        engine = FeatureEngine()
        features = engine.build(sectors, macro, vix)
        features_scaled = engine.scale(features)
        
        # Train model
        model = SectorModel()
        targets = model.create_targets(sectors)
        model.train(features_scaled, targets, use_feature_selection=True)
        model.validate(features_scaled, targets)
        
        self.model = model
        self.features = features_scaled
        
        print("✅ Model trained\n")
    
    def _get_predictions(self):
        """Get current sector rotation predictions."""
        current_features = self.features.iloc[[-1]]
        probs = self.model.predict(current_features)
        
        results_df = pd.DataFrame(self.model.results).T
        
        predictions = pd.DataFrame({
            'Probability': probs.T.iloc[:, 0],
            'Test_Acc': results_df['test_acc'],
            'Gap': results_df['gap'],
            'Category': results_df['category']
        })
        
        # Calculate confidence tiers
        confidence_map = {}
        for sector in self.model.results.keys():
            gap = self.model.results[sector]['gap']
            test_acc = self.model.results[sector]['test_acc']
            
            if gap < 0.20 and test_acc > 0.55:
                tier = "HIGH"
            elif gap < 0.30 and test_acc > 0.50:
                tier = "MEDIUM"
            else:
                tier = "LOW"
            
            confidence_map[sector] = tier
        
        predictions['Confidence'] = predictions.index.map(confidence_map)
        predictions = predictions.sort_values('Probability', ascending=False)
        
        return predictions
    
    def create(self, 
               lookback_days: int = None,
               forward_days: int = None,
               iv_rv_years: int = None,
               skip_training: bool = False):
        """
        Create dashboard.
        
        Args:
            lookback_days: Historical days to show
            forward_days: Forward projection days
            iv_rv_years: Years of IV/RV history
            skip_training: Skip model training (use existing)
        """
        lookback_days = lookback_days or self.config.DISPLAY['lookback_days']
        forward_days = forward_days or self.config.DISPLAY['forward_days']
        iv_rv_years = iv_rv_years or self.config.DISPLAY['iv_rv_years']
        
        print("\n" + "="*70)
        print("SPX/VIX DECISION DASHBOARD")
        print("="*70)
        
        # Train model if needed
        if not skip_training or self.model is None:
            self.train_model()
        
        predictions = self._get_predictions()
        
        # Fetch display data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 10)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print("📊 Fetching display data...")
        
        spx_recent = self.data_fetcher.fetch_sectors(start_str, end_str)
        vix_recent = self.data_fetcher.fetch_vix(start_str, end_str)
        
        # Get current values
        last_date = spx_recent.index[-1]
        self._validate_freshness(last_date)
        
        current_spx = float(spx_recent['SPY'].iloc[-1])
        current_vix = float(vix_recent.iloc[-1])
        current_regime = self.regime_analyzer.classify_regime(current_vix)
        velocity = self.regime_analyzer.calculate_velocity(vix_recent).iloc[-1]
        
        # IV/RV analysis
        iv_rv_data = self.iv_rv_panel.calculate_iv_rv_spread(lookback_years=iv_rv_years)
        iv_stats = self.iv_rv_panel.calculate_statistics(iv_rv_data, current_vix)
        
        print(f"✅ Data ready (as of {last_date.strftime('%Y-%m-%d')})\n")
        
        # Premium signal
        premium_pct = iv_stats['current_premium_pct']
        premium_signal = "Strong Premium Selling" if premium_pct > 70 else \
                        "Premium Buying" if premium_pct < 40 else \
                        "Neutral Premium"
        
        # Create figure
        fig = make_subplots(
            rows=4, cols=1,
            vertical_spacing=0.06,
            subplot_titles=[
                f"<b>Market Summary</b> • {last_date.strftime('%b %d, %Y')}",
                f"<b>ML Sector Rotation</b> • 21-Day Outlook • High Confidence Signals",
                f"<b>SPX Probability Cone</b> • {forward_days}-Day Forward • VIX {current_vix:.1f}",
                f"<b>Volatility Premium (IV vs RV)</b> • {iv_rv_years}Y History • Signal: <b>{premium_signal}</b> ({premium_pct:.0f}% IV>RV)"
            ],
            specs=[
                [{"type": "table"}],
                [{"type": "table"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}],
            ],
            row_heights=[0.10, 0.18, 0.34, 0.38],
        )
        
        # Add panels
        print("📊 Building dashboard...")
        
        self._add_summary(
            fig, current_spx, current_vix, current_regime,
            velocity, iv_stats, predictions, forward_days, last_date
        )
        
        self._add_rotation(fig, predictions)
        
        self._add_cone(fig, spx_recent, current_vix, lookback_days, forward_days)
        
        self._add_iv_rv(fig, iv_rv_data)
        
        # Final layout
        self._apply_layout(fig)
        
        print("✅ Dashboard complete!\n")
        return fig
    
    def _add_summary(self, fig, current_spx, current_vix, current_regime,
                    velocity, iv_stats, predictions, forward_days, data_date):
        """Add summary table."""
        vel_label = "Stable" if abs(velocity) < 2 else \
                   "Rising Fast ⚠️" if velocity >= 2 else \
                   "Falling Fast ✓"
        
        vel_color = "white" if abs(velocity) < 2 else \
                   "#ffe5e5" if velocity >= 2 else \
                   "#e5f5e5"
        
        # Top picks
        top_3 = predictions.head(3)
        top_picks = [
            f"{ticker} ({top_3.loc[ticker, 'Probability']:.0%})"
            for ticker in top_3.index
            if top_3.loc[ticker, 'Probability'] > 0.55 and 
               top_3.loc[ticker, 'Confidence'] in ['HIGH', 'MEDIUM']
        ]
        rotation_text = ", ".join(top_picks) if top_picks else "No strong signals"
        
        # Premium signal
        premium_pct = iv_stats['current_premium_pct']
        premium_signal = "Sell Premium" if premium_pct > 70 else \
                        "Buy Premium" if premium_pct < 40 else \
                        "Neutral"
        
        fig.add_trace(go.Table(
            header=dict(
                values=[
                    "<b>CURRENT MARKET</b>",
                    "<b>VIX REGIME</b>",
                    "<b>VOLATILITY PREMIUM</b>",
                    "<b>ML ROTATION</b>"
                ],
                fill_color=self.config.COLORS['primary'],
                font=dict(
                    family=self.config.FONT['family'],
                    color='white',
                    size=13
                ),
                align='left',
                height=35
            ),
            cells=dict(
                values=[
                    [
                        f"<b>SPX:</b> ${current_spx:,.0f}",
                        f"<b>VIX:</b> {current_vix:.2f}",
                        f"<b>Δ5d:</b> {velocity:+.2f} ({vel_label})",
                        f"<b>Data:</b> {data_date.strftime('%b %d, %Y')}"
                    ],
                    [
                        f"<b>Regime:</b> {current_regime}",
                        f"<b>Range:</b> {self._regime_range(current_regime)}",
                        f"<b>Signal:</b> {self._regime_signal(current_regime)}",
                        f"<b>Outlook:</b> {forward_days} days"
                    ],
                    [
                        f"<b>IV>RV:</b> {premium_pct:.0f}% likely",
                        f"<b>Avg Spread:</b> {iv_stats['current_avg_spread']:+.1f}%",
                        f"<b>Periods:</b> {iv_stats['current_periods_analyzed']}",
                        f"<b>Signal:</b> {premium_signal}"
                    ],
                    [
                        f"<b>Top Picks:</b> {rotation_text}",
                        f"<b>High Conf:</b> {len(predictions[predictions['Confidence']=='HIGH'])}",
                        f"<b>Model:</b> v3.5 RF",
                        f"<b>Horizon:</b> 21 days"
                    ]
                ],
                fill_color=[
                    ['white', vel_color, 'white', 'white']
                ],
                align='left',
                font=dict(
                    family=self.config.FONT['family'],
                    size=11
                ),
                height=28
            )
        ), row=1, col=1)
    
    def _add_rotation(self, fig, predictions):
        """Add sector rotation table."""
        model_results = {
            ticker: {
                'test_accuracy': self.model.results[ticker]['test_acc'],
                'overfitting_gap': self.model.results[ticker]['gap'],
                'category': self.model.results[ticker]['category']
            }
            for ticker in predictions.index
        }
        
        confidence_df = pd.DataFrame({
            'Sector': predictions.index.tolist(),
            'Tier': predictions['Confidence'].values,
            'Category': predictions['Category'].values
        })
        
        fig = self.rotation_panel.add_to_figure(
            fig=fig,
            predictions=predictions,
            model_results=model_results,
            confidence_df=confidence_df,
            row=2, col=1
        )
    
    def _add_cone(self, fig, spx_data, current_vix, lookback_days, forward_days):
        """Add probability cone."""
        # Convert to DataFrame if Series
        if isinstance(spx_data, pd.Series):
            spx_df = pd.DataFrame({'Close': spx_data})
        elif 'SPY' in spx_data.columns:
            spx_df = pd.DataFrame({'Close': spx_data['SPY'].squeeze()})
        else:
            spx_df = spx_data
        
        fig = self.cone_panel.add_to_figure(
            fig=fig,
            spx_data=spx_df,
            current_vix=current_vix,
            lookback_days=lookback_days,
            forward_days=forward_days,
            show_strikes=False,
            show_vol_scenarios=True,
            row=3, col=1
        )
    
    def _add_iv_rv(self, fig, iv_rv_data):
        """Add IV vs RV panel."""
        fig = self.iv_rv_panel.add_to_figure(
            fig=fig,
            iv_rv_data=iv_rv_data,
            row=4, col=1
        )
    
    def _apply_layout(self, fig):
        """Apply final layout styling."""
        fig.update_xaxes(title_text="Date", row=3, col=1, title_font=self.config.TITLE_FONT)
        fig.update_xaxes(title_text="Date", row=4, col=1, title_font=self.config.TITLE_FONT)
        
        fig.update_yaxes(title_text="SPX Price", row=3, col=1, title_font=self.config.TITLE_FONT)
        fig.update_yaxes(title_text="Volatility (%)", row=4, col=1, title_font=self.config.TITLE_FONT)
        
        fig.update_layout(
            width=self.config.LAYOUT['width'],
            height=self.config.LAYOUT['height'],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=self.config.FONT
            ),
            hovermode=self.config.LAYOUT['hovermode'],
            template=self.config.LAYOUT['template'],
            font=self.config.FONT,
            title=dict(
                text=f"<b>SPX/VIX Decision Dashboard</b><br>"
                     f"<sup style='color:#7f8c8d'>Simplified Edition • "
                     f"{datetime.now().strftime('%B %d, %Y %H:%M')}</sup>",
                x=0.5,
                xanchor='center',
                font=dict(
                    family=self.config.FONT['family'],
                    size=20,
                    color=self.config.COLORS['primary']
                )
            ),
            margin=dict(t=100, b=60, l=80, r=80),
            plot_bgcolor=self.config.COLORS['background'],
            paper_bgcolor='white',
        )
        
        for annotation in fig.layout.annotations:
            annotation.update(font=self.config.TITLE_FONT)
    
    def _regime_range(self, regime: str) -> str:
        """Get regime range description."""
        ranges = {'Low': '<15', 'Normal': '15-25', 'Elevated': '25-35', 'Crisis': '>35'}
        return ranges.get(regime, 'Unknown')
    
    def _regime_signal(self, regime: str) -> str:
        """Get regime trading signal."""
        signals = {
            'Low': 'Buy Dips',
            'Normal': 'Selective',
            'Elevated': 'Hedged',
            'Crisis': 'Wait & Watch'
        }
        return signals.get(regime, 'Unknown')
    
    def save(self, filename: str = 'spx_dashboard.html', **kwargs):
        """Create and save dashboard."""
        fig = self.create(**kwargs)
        fig.write_html(filename)
        print(f"💾 Saved: {filename}")
        fig.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPX/VIX DECISION DASHBOARD - SIMPLIFIED")
    print("="*70)
    
    dashboard = Dashboard()
    dashboard.save('spx_dashboard_simplified.html')
    
    print("\n✅ COMPLETE!")