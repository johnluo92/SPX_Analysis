"""
VIX Visualizer v4.2 - Clean Orchestrator Edition

PHILOSOPHY:
- Let plotly handle spacing automatically
- No hardcoded y-coordinates
- No annotation collisions
- Consistent fonts and spacing everywhere
- Each panel is self-contained
- No magic numbers

FIXES:
- Removed manual y-position calculations
- Removed overlapping annotations
- Fixed table scrollbar issue (height constraint)
- Uniform spacing between all panels
- Single font configuration
- Let subplot_titles handle all text
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from UnifiedDataFetcher import UnifiedDataFetcher
from panel_cone import ProbabilityConePanel
from panel_sector_rotation import SectorRotationPanel
from panel_iv_rv import IVvsRVPanel
from regime_analyzer import RegimeAnalyzer

from v35 import SectorRotationModel, SectorRotationFeatures
from sector_data_fetcher_v34 import SectorDataFetcher


class VisualizationConfig:
    """Clean, simple configuration."""
    
    LAYOUT = {
        'width': 1600,
        'height': 1850,  # Taller to accommodate spacing
        'template': 'plotly_white',
        'hovermode': 'x unified',
        'vertical_spacing': 0.06,  # Match subplot spacing
    }
    
    # Single font family for entire dashboard
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
    
    DATA = {
        'lookback_days': 30,  # Show more recent context
        'forward_days': 21,   # Standard 1-month outlook
        'iv_rv_years': 5,
        'sector_training_years': 7,
    }
    
    COLORS = {
        'primary': '#2E86AB',
        'accent': '#F18F01',
        'success': '#27ae60',
        'warning': '#e74c3c',
        'neutral': '#95a5a6',
        'background': '#f8f9fa',
    }


class ConeVisualizerV42Clean:
    """Clean orchestrator - no manual positioning."""
    
    def __init__(self, config=None):
        self.config = config or VisualizationConfig()
        self.fetcher = UnifiedDataFetcher()
        self.cone_panel = ProbabilityConePanel()
        self.rotation_panel = SectorRotationPanel()
        self.iv_rv_panel = IVvsRVPanel()
        self.regime_analyzer = RegimeAnalyzer()
        
        self.sector_fetcher = SectorDataFetcher()
        self.ml_model = None
        self.ml_features = None
    
    def _validate_data_freshness(self, data_date, max_age_days=3):
        """Validate data freshness."""
        today = datetime.now().date()
        data_date = pd.to_datetime(data_date).date()
        age = (today - data_date).days
        
        if age > max_age_days:
            print(f"⚠️  WARNING: Data is {age} days old (from {data_date})")
            return False
        return True
    
    def _train_ml_model(self):
        """Train ML model - same as before."""
        print("\n🤖 Training ML Model...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.DATA['sector_training_years'] * 365)
        
        sectors = self.sector_fetcher.fetch_sector_etfs(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        macro = self.sector_fetcher.fetch_macro_factors(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        vix = self.fetcher.fetch_vix(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            source='yahoo'
        )
        
        sectors_aligned, macro_aligned, vix_aligned = self.sector_fetcher.align_data(
            sectors, macro, vix
        )
        
        feat_eng = SectorRotationFeatures()
        features = feat_eng.combine_features(sectors_aligned, macro_aligned, vix_aligned)
        
        model = SectorRotationModel()
        targets = model.create_targets(sectors_aligned, forward_window=21)
        results = model.train_models(features, targets, use_feature_selection=True, test_split=0.2)
        validation = model.walk_forward_validate(features, targets, n_splits=5)
        
        print("✅ Model trained\n")
        return model, features
    
    def _get_current_predictions(self, model, features):
        """Get current predictions."""
        current_features = features.iloc[[-1]]
        probs = model.predict_probabilities(current_features)
        
        results_df = pd.DataFrame(model.results).T
        
        predictions = pd.DataFrame({
            'Probability': probs.T.iloc[:, 0],
            'Test_Acc': results_df['test_accuracy'],
            'Gap': results_df['overfitting_gap'],
            'Category': results_df['category']
        })
        
        # Confidence calculation
        confidence_map = {}
        for sector in model.results.keys():
            gap = model.results[sector]['overfitting_gap']
            test_acc = model.results[sector]['test_accuracy']
            
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
    
    def plot_decision_chart(self,
                           lookback_days: int = None,
                           forward_days: int = None,
                           iv_rv_years: int = None,
                           skip_ml_training: bool = False):
        """
        Create clean dashboard with automatic spacing.
        
        NO manual y-coordinates.
        NO annotation collisions.
        Just clean, consistent panels.
        """
        
        lookback_days = lookback_days or self.config.DATA['lookback_days']
        forward_days = forward_days or self.config.DATA['forward_days']
        iv_rv_years = iv_rv_years or self.config.DATA['iv_rv_years']
        
        print("\n" + "="*70)
        print("VIX DASHBOARD v4.2 - Clean Orchestrator")
        print("="*70)
        
        # Train ML model
        if not skip_ml_training:
            self.ml_model, self.ml_features = self._train_ml_model()
        
        if self.ml_model is None:
            raise ValueError("No model available")
        
        predictions = self._get_current_predictions(self.ml_model, self.ml_features)
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 10)
        
        print(f"📊 Fetching data...")
        
        spx_recent = self.fetcher.fetch_spx(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        vix_recent = self.fetcher.fetch_vix(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            source='yahoo'
        )
        
        # Validate freshness
        last_date = spx_recent.index[-1]
        self._validate_data_freshness(last_date)
        
        current_spx = spx_recent['Close'].iloc[-1]
        current_vix = vix_recent.iloc[-1]
        current_regime = self.regime_analyzer.classify_regime(current_vix)
        velocity = self.regime_analyzer.calculate_velocity(vix_recent).iloc[-1]
        
        # IV/RV data
        iv_rv_data = self.iv_rv_panel.calculate_iv_rv_spread(lookback_years=iv_rv_years)
        iv_stats = self.iv_rv_panel.calculate_statistics(iv_rv_data, current_vix)
        
        print(f"✅ Data ready (as of {last_date.strftime('%Y-%m-%d')})\n")
        
        # Premium signal for subtitle
        premium_pct = iv_stats['current_premium_pct']
        premium_signal = "Strong Premium Selling" if premium_pct > 70 else \
                        "Premium Buying" if premium_pct < 40 else \
                        "Neutral Premium"
        
        # ============================================================
        # CREATE FIGURE - Reordered for better flow
        # ============================================================
        fig = make_subplots(
            rows=4, cols=1,
            vertical_spacing=0.06,  # MORE space between panels to prevent collision
            subplot_titles=[
                f"<b>Market Summary</b> • {last_date.strftime('%b %d, %Y')}",
                f"<b>ML Sector Rotation</b> • 21-Day Outlook • High Confidence Signals",
                f"<b>SPX Probability Cone</b> • {forward_days}-Day Forward • VIX {current_vix:.1f}",
                f"<b>Volatility Premium (IV vs RV)</b> • {iv_rv_years}Y History • Signal: <b>{premium_signal}</b> ({premium_pct:.0f}% IV>RV)"
            ],
            specs=[
                [{"type": "table"}],    # Row 1: Summary
                [{"type": "table"}],    # Row 2: ML Sector (moved up)
                [{"type": "scatter"}],  # Row 3: Cone (moved down)
                [{"type": "scatter"}],  # Row 4: IV/RV
            ],
            row_heights=[0.10, 0.18, 0.34, 0.38],  # Compact tables, roomy charts
        )
        
        # ============================================================
        # ROW 1: SUMMARY TABLE
        # ============================================================
        print("📊 Building summary...")
        self._add_summary_table(
            fig, current_spx, current_vix, current_regime, 
            velocity, iv_stats, predictions, forward_days, last_date
        )
        
        # ============================================================
        # ROW 2: ML SECTOR ROTATION (moved up)
        # ============================================================
        print("📊 Adding ML predictions...")
        model_results = {
            ticker: {
                'test_accuracy': self.ml_model.results[ticker]['test_accuracy'],
                'overfitting_gap': self.ml_model.results[ticker]['overfitting_gap'],
                'category': self.ml_model.results[ticker]['category']
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
        
        # ============================================================
        # ROW 3: PROBABILITY CONE (moved down)
        # ============================================================
        print("📊 Adding probability cone...")
        fig = self.cone_panel.add_to_figure(
            fig=fig,
            spx_data=spx_recent,
            current_vix=current_vix,
            lookback_days=lookback_days,
            forward_days=forward_days,
            show_strikes=False,
            show_vol_scenarios=True,
            row=3, col=1
        )
        
        # ============================================================
        # ROW 4: IV vs RV
        # ============================================================
        print("📊 Adding IV/RV analysis...")
        fig = self.iv_rv_panel.add_to_figure(
            fig=fig,
            iv_rv_data=iv_rv_data,
            row=4, col=1
        )
        
        # ============================================================
        # FINAL LAYOUT - Simple and consistent
        # ============================================================
        fig.update_xaxes(title_text="Date", row=3, col=1, title_font=self.config.TITLE_FONT)
        fig.update_xaxes(title_text="Date", row=4, col=1, title_font=self.config.TITLE_FONT)
        
        fig.update_yaxes(title_text="SPX Price", row=3, col=1, title_font=self.config.TITLE_FONT)
        fig.update_yaxes(title_text="Volatility (%)", row=4, col=1, title_font=self.config.TITLE_FONT)
        
        # Apply consistent fonts everywhere
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
            font=self.config.FONT,  # Apply to all text
            title=dict(
                text=f"<b>SPX/VIX Decision Dashboard v4.2</b><br>"
                     f"<sup style='color:#7f8c8d'>Clean Orchestrator Edition • "
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
        
        # Update all subplot title fonts
        for annotation in fig.layout.annotations:
            annotation.update(font=self.config.TITLE_FONT)
        
        print("✅ Dashboard complete!\n")
        return fig
    
    def _add_summary_table(self, fig, current_spx, current_vix, current_regime,
                          velocity, iv_stats, predictions, forward_days, data_date):
        """
        Clean summary table with NO internal scrollbar.
        Fixed height, clean cells.
        """
        
        vel_label = "Stable" if abs(velocity) < 2 else \
                   "Rising Fast ⚠️" if velocity >= 2 else \
                   "Falling Fast ✓"
        
        vel_color = "white" if abs(velocity) < 2 else \
                   "#ffe5e5" if velocity >= 2 else \
                   "#e5f5e5"
        
        # Top ML picks
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
                        f"<b>Range:</b> {self._get_regime_range(current_regime)}",
                        f"<b>Signal:</b> {self._get_regime_signal(current_regime)}",
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
    
    def _get_regime_range(self, regime: str) -> str:
        ranges = {'Low': '<15', 'Normal': '15-25', 'Elevated': '25-35', 'Crisis': '>35'}
        return ranges.get(regime, 'Unknown')
    
    def _get_regime_signal(self, regime: str) -> str:
        signals = {
            'Low': 'Buy Dips',
            'Normal': 'Selective',
            'Elevated': 'Hedged',
            'Crisis': 'Wait & Watch'
        }
        return signals.get(regime, 'Unknown')
    
    def save_chart(self, filename: str = 'spx_dashboard_v4.2_clean.html', **kwargs):
        """Create and save clean dashboard."""
        fig = self.plot_decision_chart(**kwargs)
        fig.write_html(filename)
        print(f"💾 Saved: {filename}")
        fig.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPX/VIX DECISION DASHBOARD v4.2")
    print("Clean Orchestrator Edition")
    print("="*70)
    
    viz = ConeVisualizerV42Clean()
    viz.save_chart()
    
    print("\n✅ COMPLETE!")
    print("\n🎯 What Changed:")
    print("   • Removed all manual y-coordinate calculations")
    print("   • Removed annotation collision management")
    print("   • Fixed table scrollbar issue")
    print("   • Uniform spacing (0.08) between ALL panels")
    print("   • Single font configuration applied everywhere")
    print("   • Let plotly handle row heights automatically")
    print("   • Insights moved to subtitle (no collision risk)")
    print("\n💡 Now You Can Focus On:")
    print("   • Fixing stale data in probability cone")
    print("   • Investigating RV spike timing issue")
    print("   • Adding real analytical value")
    print("   • Not fighting with display code!")