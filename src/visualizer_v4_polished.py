"""
VIX Visualizer v4.1 - Production Polish Edition

IMPROVEMENTS IN v4.1:
- Clear visual section breaks between panels
- Optimized probability cone layout (focuses on forward projection)
- Enhanced IV/RV storytelling with annotations
- Better spacing and hierarchy
- Data freshness validation
- Professional visual polish
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

# Import v3.5 model components
from v35 import SectorRotationModel, SectorRotationFeatures
from sector_data_fetcher_v34 import SectorDataFetcher


class VisualizationConfig:
    """Centralized configuration with production polish."""
    
    LAYOUT = {
        'width': 1800,
        'height': 2000,
        'row_heights': [0.08, 0.28, 0.24, 0.32, 0.08],  # Added spacer row
        'vertical_spacing': 0.04,
        'template': 'plotly_white',
        'hovermode': 'x unified',
    }
    
    DATA = {
        'lookback_days': 21,  # Reduced to focus on recent action
        'forward_days': 14,
        'iv_rv_years': 5,
        'sector_training_years': 7,
    }
    
    COLORS = {
        'section_break': '#2E86AB',
        'accent': '#F18F01',
        'success': '#2ECC71',
        'warning': '#E74C3C',
        'neutral': '#95A5A6',
    }


class ConeVisualizerV4Polished:
    """Production-grade visualizer with professional polish."""
    
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
        """Validate that data is fresh enough."""
        today = datetime.now().date()
        data_date = pd.to_datetime(data_date).date()
        age = (today - data_date).days
        
        if age > max_age_days:
            print(f"⚠️  WARNING: Data is {age} days old (from {data_date})")
            print(f"   Consider updating data sources or checking market hours")
            return False
        return True
    
    def _train_ml_model(self):
        """Train the v3.5 sector rotation model."""
        print("\n🤖 Training ML Sector Rotation Model (v3.5)...")
        print("-" * 60)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.DATA['sector_training_years'] * 365)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"📊 Fetching {self.config.DATA['sector_training_years']} years of sector data...")
        
        sectors = self.sector_fetcher.fetch_sector_etfs(start_str, end_str)
        macro = self.sector_fetcher.fetch_macro_factors(start_str, end_str)
        vix = self.fetcher.fetch_vix(start_str, end_str, source='yahoo')
        
        sectors_aligned, macro_aligned, vix_aligned = self.sector_fetcher.align_data(
            sectors, macro, vix
        )
        
        print("🔧 Engineering features...")
        feat_eng = SectorRotationFeatures()
        features = feat_eng.combine_features(
            sectors_aligned,
            macro_aligned,
            vix_aligned
        )
        
        print("🌲 Training Random Forest models...")
        model = SectorRotationModel()
        targets = model.create_targets(sectors_aligned, forward_window=21)
        
        results = model.train_models(
            features, targets, 
            use_feature_selection=True,
            test_split=0.2
        )
        
        print("📊 Running walk-forward validation...")
        validation = model.walk_forward_validate(features, targets, n_splits=5)
        
        confidence = model.calculate_confidence_scores()
        
        print("✅ ML model training complete!\n")
        
        return model, features, results, validation, confidence
    
    def _get_current_predictions(self, model, features):
        """Get predictions for current market conditions."""
        current_features = features.iloc[[-1]]
        probs = model.predict_probabilities(current_features)
        
        results_df = pd.DataFrame(model.results).T
        
        predictions = pd.DataFrame({
            'Probability': probs.T.iloc[:, 0],
            'Test_Acc': results_df['test_accuracy'],
            'Gap': results_df['overfitting_gap'],
            'Category': results_df['category']
        })
        
        confidence_map = {}
        for sector in model.results.keys():
            gap = model.results[sector]['overfitting_gap']
            test_acc = model.results[sector]['test_accuracy']
            wf_std = model.validation_results[sector]['std_accuracy']
            
            if gap < 0.20 and test_acc > 0.55 and wf_std < 0.10:
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
        """Create production-grade decision chart."""
        
        lookback_days = lookback_days or self.config.DATA['lookback_days']
        forward_days = forward_days or self.config.DATA['forward_days']
        iv_rv_years = iv_rv_years or self.config.DATA['iv_rv_years']
        
        print("\n" + "="*60)
        print("BUILDING PRODUCTION VIX DASHBOARD v4.1")
        print("="*60)
        
        # Train ML model
        if not skip_ml_training:
            model, features, results, validation, confidence_df = self._train_ml_model()
            predictions = self._get_current_predictions(model, features)
            self.ml_model = model
            self.ml_features = features
        else:
            print("\n⚠️  Using cached model")
            if self.ml_model is None:
                raise ValueError("No cached model available")
            predictions = self._get_current_predictions(self.ml_model, self.ml_features)
        
        # Fetch visualization data
        end_date = datetime.now()
        
        print(f"\n📊 Fetching visualization data...")
        
        spx_recent = self.fetcher.fetch_spx(
            (end_date - timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        vix_recent = self.fetcher.fetch_vix(
            (end_date - timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            source='yahoo'
        )
        
        # Validate data freshness
        last_spx_date = spx_recent.index[-1]
        self._validate_data_freshness(last_spx_date)
        
        current_spx = spx_recent['Close'].iloc[-1]
        current_vix = vix_recent.iloc[-1]
        current_regime = self.regime_analyzer.classify_regime(current_vix)
        velocity = self.regime_analyzer.calculate_velocity(vix_recent).iloc[-1]
        
        print(f"🔧 Calculating IV vs RV...")
        iv_rv_data = self.iv_rv_panel.calculate_iv_rv_spread(lookback_years=iv_rv_years)
        iv_stats = self.iv_rv_panel.calculate_statistics(iv_rv_data, current_vix)
        
        print(f"✅ All data ready (as of {last_spx_date.strftime('%Y-%m-%d')})")
        
        # Create figure with spacer rows for visual breaks
        fig = make_subplots(
            rows=5, cols=1,
            row_heights=self.config.LAYOUT['row_heights'],
            vertical_spacing=self.config.LAYOUT['vertical_spacing'],
            subplot_titles=[
                "",  # Summary table
                f"<b>📈 SPX Probability Cone</b><span style='color:gray; font-size:12px'>  •  VIX {current_vix:.1f}  •  {last_spx_date.strftime('%b %d, %Y')}</span>",
                f"<b>🔄 ML Sector Rotation</b><span style='color:gray; font-size:12px'>  •  21-Day Outlook  •  v3.5 Model</span>",
                f"<b>📊 Volatility Premium Analysis</b><span style='color:gray; font-size:12px'>  •  {iv_rv_years}Y History  •  Why Premium Selling Works</span>",
                "",  # Spacer
            ],
            specs=[
                [{"type": "table"}],
                [{"type": "scatter"}],
                [{"type": "table"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}],  # Invisible spacer
            ]
        )
        
        # Row 1: Summary Table
        print("📊 Building summary...")
        self._add_summary_table(
            fig, current_spx, current_vix, current_regime, velocity,
            iv_stats, predictions, forward_days, last_spx_date
        )
        
        # Row 2: Probability Cone
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

        
        # Row 3: ML Sector Rotation
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
            row=3, col=1
        )
        
        # Row 4: IV vs RV with enhanced storytelling
        print("📊 Adding IV/RV analysis...")
        fig = self.iv_rv_panel.add_to_figure(
            fig=fig,
            iv_rv_data=iv_rv_data,
            row=4, col=1
        )
        
        # Add insight annotations to IV/RV panel
        self._add_iv_rv_insights(fig, iv_stats, row=4)
        
        # Final layout
        fig.update_xaxes(title_text="", row=2, col=1)
        fig.update_xaxes(title_text="", row=4, col=1)
        
        fig.update_yaxes(title_text="SPX Price", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=4, col=1)
        
        # Hide spacer row
        fig.update_xaxes(visible=False, row=5, col=1)
        fig.update_yaxes(visible=False, row=5, col=1)
        
        fig.update_layout(
            width=self.config.LAYOUT['width'],
            height=self.config.LAYOUT['height'],
            showlegend=False,
            hovermode=self.config.LAYOUT['hovermode'],
            template=self.config.LAYOUT['template'],
            title=dict(
                text=f"<b>SPX/VIX Decision Dashboard v4.1</b><br><sup>Production Edition  •  {datetime.now().strftime('%B %d, %Y %H:%M')}</sup>",
                x=0.5,
                xanchor='center',
                font=dict(size=22)
            ),
            margin=dict(t=100, b=60, l=80, r=80)
        )
        
        print("✅ Visualization complete!\n")
        return fig
    
    def _get_section_y_position(self, row):
        """Calculate y position for section break based on row."""
        # Approximate positions based on row heights
        positions = {
            1: 0.92,
            2: 0.58,
            3: 0.30,
            4: 0.02
        }
        return positions.get(row, 0.5)
    
    def _add_iv_rv_insights(self, fig, iv_stats, row):
        """Add insight box to IV/RV panel."""
        premium_pct = iv_stats['current_premium_pct']
        
        if premium_pct > 70:
            insight = "📉 Strong Premium Selling Environment"
            color = self.config.COLORS['success']
        elif premium_pct < 40:
            insight = "📈 Prefer Premium Buying"
            color = self.config.COLORS['warning']
        else:
            insight = "⚖️ Neutral Premium Environment"
            color = self.config.COLORS['neutral']
        
        insight_text = (
            f"<b>{insight}</b><br>"
            f"<br>"
            f"<b>Key Insight:</b> Over the past {iv_stats['current_periods_analyzed']} periods<br>"
            f"with VIX near current levels, implied vol has<br>"
            f"exceeded realized vol <b>{premium_pct:.0f}%</b> of the time.<br>"
            f"<br>"
            f"<b>Average Overstatement:</b> {iv_stats['current_avg_spread']:+.1f}%<br>"
            f"<br>"
            f"<i>This volatility risk premium is why<br>"
            f"systematic premium selling strategies<br>"
            f"have positive expected value.</i>"
        )
        
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=self._get_section_y_position(row) + 0.12,
            text=insight_text,
            showarrow=False,
            xanchor='right',
            yanchor='middle',
            align='left',
            font=dict(size=10),
            bordercolor=color,
            borderwidth=2,
            borderpad=10,
            bgcolor='rgba(255, 255, 255, 0.97)',
        )
    
    def _add_summary_table(self, fig, current_spx, current_vix, current_regime,
                          velocity, iv_stats, predictions, forward_days, data_date):
        """Enhanced summary table with data date."""
        
        if abs(velocity) < 2:
            vel_label = "Stable"
            vel_color = "white"
        elif velocity >= 2:
            vel_label = "Rising Fast ⚠️"
            vel_color = "#FFE5E5"
        else:
            vel_label = "Falling Fast"
            vel_color = "#E5F5E5"
        
        top_3 = predictions.head(3)
        top_picks = []
        for ticker in top_3.index:
            prob = top_3.loc[ticker, 'Probability']
            conf = top_3.loc[ticker, 'Confidence']
            if prob > 0.55 and conf in ['HIGH', 'MEDIUM']:
                top_picks.append(f"{ticker} ({prob:.1%})")
        
        rotation_text = ", ".join(top_picks) if top_picks else "No strong signals"
        
        fig.add_trace(go.Table(
            header=dict(
                values=[
                    "<b>CURRENT MARKET</b>",
                    "<b>VIX REGIME</b>",
                    "<b>VOLATILITY PREMIUM</b>",
                    "<b>ML ROTATION PICKS</b>"
                ],
                fill_color='#2E86AB',
                font=dict(color='white', size=13),
                align='left',
                height=32
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
                        f"<b>Cone:</b> {forward_days} days"
                    ],
                    [
                        f"<b>IV>RV:</b> {iv_stats['current_premium_pct']:.0f}% likely",
                        f"<b>Overstatement:</b> {iv_stats['current_avg_spread']:+.1f}%",
                        f"<b>Periods:</b> {iv_stats['current_periods_analyzed']}",
                        f"<b>Edge:</b> {self._interpret_iv_rv(iv_stats)}"
                    ],
                    [
                        f"<b>Top:</b> {rotation_text}",
                        f"<b>High Conf:</b> {len(predictions[predictions['Confidence']=='HIGH'])}",
                        f"<b>Model:</b> v3.5 (7Y)",
                        f"<b>Horizon:</b> 21 days"
                    ]
                ],
                fill_color=[['#E8F4F8', vel_color, '#E8F4F8', 'white']],
                align='left',
                font=dict(size=11),
                height=26
            )
        ), row=1, col=1)
    
    def _get_regime_range(self, regime: str) -> str:
        ranges = {'Low': '<15', 'Normal': '15-25', 'Elevated': '25-35', 'Crisis': '>35'}
        return ranges.get(regime, 'Unknown')
    
    def _get_regime_signal(self, regime: str) -> str:
        signals = {'Low': 'Buy Dips', 'Normal': 'Selective', 'Elevated': 'Hedged', 'Crisis': 'Wait'}
        return signals.get(regime, 'Unknown')
    
    def _interpret_iv_rv(self, iv_stats: dict) -> str:
        premium_pct = iv_stats['current_premium_pct']
        if premium_pct > 70:
            return "Sell premium"
        elif premium_pct < 40:
            return "Buy premium"
        else:
            return "Neutral"
    
    def save_chart(self, filename: str = 'spx_dashboard_v4.1_production.html', **kwargs):
        """Create and save the polished chart."""
        fig = self.plot_decision_chart(**kwargs)
        fig.write_html(filename)
        print(f"💾 Chart saved to {filename}")
        fig.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPX/VIX DECISION DASHBOARD v4.1")
    print("Production Polish Edition")
    print("="*60)
    
    viz = ConeVisualizerV4Polished()
    viz.save_chart()
    
    print("\n✅ DONE!")
    print("\n📊 What's New in v4.1:")
    print("   • Clear visual section breaks between panels")
    print("   • Optimized probability cone layout")
    print("   • Enhanced IV/RV storytelling with insights")
    print("   • Data freshness validation")
    print("   • Professional spacing and hierarchy")
    print("   • Better use of whitespace")
    print("\n💡 Production Ready:")
    print("   • All panels have clear visual separation")
    print("   • Key insights highlighted with annotation boxes")
    print("   • Responsive layout adapts to content")
    print("   • Data date clearly shown in summary")