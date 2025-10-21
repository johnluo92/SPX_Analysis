"""
VIX Visualizer v4.0 - ML Sector Rotation Edition

NEW IN v4.0:
- Panel 3 is now ML-driven sector rotation predictions from v3.5 model
- Real machine learning probabilities for sector outperformance
- Integrates the full v3.5 training pipeline
- Shows which sectors to rotate into for next 21 days
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
    """Centralized configuration."""
    
    LAYOUT = {
        'width': 1600,
        'height': 1600,  # Increased more to give proper room
        'row_heights': [0.10, 0.30, 0.25, 0.35],  # More space for rotation table and IV/RV
        'vertical_spacing': 0.05,  # Even tighter
        'template': 'plotly_white',
        'hovermode': 'x unified',
    }
    
    DATA = {
        'lookback_days': 30,
        'forward_days': 14,
        'iv_rv_years': 5,
        'sector_training_years': 7,  # For ML model
    }


class ConeVisualizerV4:
    """Visualizer with ML sector rotation predictions."""
    
    def __init__(self, config=None):
        self.config = config or VisualizationConfig()
        self.fetcher = UnifiedDataFetcher()
        self.cone_panel = ProbabilityConePanel()
        self.rotation_panel = SectorRotationPanel()
        self.iv_rv_panel = IVvsRVPanel()
        self.regime_analyzer = RegimeAnalyzer()
        
        # ML model components
        self.sector_fetcher = SectorDataFetcher()
        self.ml_model = None
        self.ml_features = None
    
    def _train_ml_model(self):
        """Train the v3.5 sector rotation model."""
        print("\n🤖 Training ML Sector Rotation Model (v3.5)...")
        print("-" * 60)
        
        # Fetch training data
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
        
        # Feature engineering
        print("🔧 Engineering features...")
        feat_eng = SectorRotationFeatures()
        features = feat_eng.combine_features(
            sectors_aligned,
            macro_aligned,
            vix_aligned
        )
        
        # Train model
        print("🌲 Training Random Forest models...")
        model = SectorRotationModel()
        targets = model.create_targets(sectors_aligned, forward_window=21)
        
        results = model.train_models(
            features, targets, 
            use_feature_selection=True,
            test_split=0.2
        )
        
        # Walk-forward validation
        print("📊 Running walk-forward validation...")
        validation = model.walk_forward_validate(features, targets, n_splits=5)
        
        # Confidence scoring
        confidence = model.calculate_confidence_scores()
        
        print("✅ ML model training complete!\n")
        
        return model, features, results, validation, confidence
    
    def _get_current_predictions(self, model, features):
        """Get predictions for current market conditions."""
        # Use most recent features
        current_features = features.iloc[[-1]]
        probs = model.predict_probabilities(current_features)
        
        # Combine with model metadata
        results_df = pd.DataFrame(model.results).T
        
        predictions = pd.DataFrame({
            'Probability': probs.T.iloc[:, 0],
            'Test_Acc': results_df['test_accuracy'],
            'Gap': results_df['overfitting_gap'],
            'Category': results_df['category']
        })
        
        # Add confidence from confidence scoring
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
        
        # Sort by probability
        predictions = predictions.sort_values('Probability', ascending=False)
        
        return predictions
    
    def plot_decision_chart(self,
                           lookback_days: int = None,
                           forward_days: int = None,
                           iv_rv_years: int = None,
                           skip_ml_training: bool = False):
        """Create the complete decision chart with ML predictions."""
        
        lookback_days = lookback_days or self.config.DATA['lookback_days']
        forward_days = forward_days or self.config.DATA['forward_days']
        iv_rv_years = iv_rv_years or self.config.DATA['iv_rv_years']
        
        print("\n" + "="*60)
        print("BUILDING VIX VISUALIZATION v4.0")
        print("ML Sector Rotation Edition")
        print("="*60)
        
        # ==================== TRAIN ML MODEL ====================
        
        if not skip_ml_training:
            model, features, results, validation, confidence_df = self._train_ml_model()
            predictions = self._get_current_predictions(model, features)
            
            self.ml_model = model
            self.ml_features = features
        else:
            print("\n⚠️  Skipping ML training (using cached model)")
            if self.ml_model is None:
                raise ValueError("No cached model available. Set skip_ml_training=False")
            predictions = self._get_current_predictions(self.ml_model, self.ml_features)
        
        # ==================== FETCH VISUALIZATION DATA ====================
        
        end_date = datetime.now()
        
        print(f"\n📊 Fetching visualization data...")
        
        # SPX and VIX for cone
        spx_recent = self.fetcher.fetch_spx(
            (end_date - timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        vix_recent = self.fetcher.fetch_vix(
            (end_date - timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            source='yahoo'
        )
        
        # Current values
        current_spx = spx_recent['Close'].iloc[-1]
        current_vix = vix_recent.iloc[-1]
        current_regime = self.regime_analyzer.classify_regime(current_vix)
        velocity = self.regime_analyzer.calculate_velocity(vix_recent).iloc[-1]
        
        # IV/RV stats
        print(f"🔧 Calculating IV vs RV...")
        iv_rv_data = self.iv_rv_panel.calculate_iv_rv_spread(lookback_years=iv_rv_years)
        iv_stats = self.iv_rv_panel.calculate_statistics(iv_rv_data, current_vix)
        
        print(f"✅ All data ready")
        
        # ==================== CREATE FIGURE ====================
        
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=self.config.LAYOUT['row_heights'],
            vertical_spacing=self.config.LAYOUT['vertical_spacing'],
            subplot_titles=[
                "",  # Row 1: Summary table
                f"<b>SPX Probability Cone</b> (VIX {current_vix:.1f})",
                f"<b>ML Sector Rotation Predictions</b> (21-Day Horizon)",
                f"<b>Implied vs Realized Volatility</b> ({iv_rv_years}Y History)"
            ],
            specs=[
                [{"type": "table"}],     # Row 1: Summary
                [{"type": "scatter"}],   # Row 2: Cone
                [{"type": "table"}],     # Row 3: ML Rotation (NEW!)
                [{"type": "scatter"}]    # Row 4: IV/RV
            ]
        )
        
        # ==================== ROW 1: SUMMARY TABLE ====================
        
        print("📊 Building summary table...")
        self._add_summary_table(
            fig, current_spx, current_vix, current_regime, velocity,
            iv_stats, predictions, forward_days
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
        
        # ==================== ROW 3: ML SECTOR ROTATION ====================
        
        print("📊 Adding ML sector rotation predictions...")
        
        # Prepare data for panel
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
                text=f"<b>SPX/VIX Decision Dashboard v4.0</b><br><sup>ML Sector Rotation • {datetime.now().strftime('%Y-%m-%d %H:%M')}</sup>",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            )
        )
        
        print("✅ Visualization complete!\n")
        return fig
    
    def _add_summary_table(self, fig, current_spx, current_vix, current_regime,
                          velocity, iv_stats, predictions, forward_days):
        """Add summary table at the top with ML predictions."""
        
        # Velocity label
        if abs(velocity) < 2:
            vel_label = "Stable"
        elif velocity >= 2:
            vel_label = "Rising Fast ⚠️"
        else:
            vel_label = "Falling Fast"
        
        # Top 3 rotation picks
        top_3 = predictions.head(3)
        top_picks = []
        for ticker in top_3.index:
            prob = top_3.loc[ticker, 'Probability']
            conf = top_3.loc[ticker, 'Confidence']
            if prob > 0.55 and conf in ['HIGH', 'MEDIUM']:
                top_picks.append(f"{ticker} ({prob:.1%})")
        
        if top_picks:
            rotation_text = ", ".join(top_picks)
        else:
            rotation_text = "No strong signals"
        
        # Build table
        fig.add_trace(go.Table(
            header=dict(
                values=[
                    "<b>CURRENT MARKET</b>",
                    "<b>VIX REGIME</b>",
                    "<b>VOLATILITY PREMIUM</b>",
                    "<b>ML ROTATION PICKS</b>"
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
                        f"<b>VIX Range:</b> {self._get_regime_range(current_regime)}",
                        f"<b>Signal:</b> {self._get_regime_signal(current_regime)}",
                        ""
                    ],
                    [
                        f"<b>IV>RV:</b> {iv_stats['current_premium_pct']:.0f}% likely",
                        f"<b>Avg Spread:</b> {iv_stats['current_avg_spread']:+.1f}%",
                        f"<b>Periods:</b> {iv_stats['current_periods_analyzed']}",
                        f"<b>Edge:</b> {self._interpret_iv_rv(iv_stats)}"
                    ],
                    [
                        f"<b>Top Picks:</b> {rotation_text}",
                        f"<b>High Conf:</b> {len(predictions[predictions['Confidence']=='HIGH'])} sectors",
                        f"<b>Model:</b> v3.5 (7Y data)",
                        f"<b>Horizon:</b> 21 days"
                    ]
                ],
                fill_color=[['#E8F4F8', 'white', '#E8F4F8', 'white']],
                align='left',
                font=dict(size=11),
                height=25
            )
        ), row=1, col=1)
    
    def _get_regime_range(self, regime: str) -> str:
        """Get VIX range for regime."""
        ranges = {
            'Low': '<15',
            'Normal': '15-25',
            'Elevated': '25-35',
            'Crisis': '>35'
        }
        return ranges.get(regime, 'Unknown')
    
    def _get_regime_signal(self, regime: str) -> str:
        """Get trading signal for regime."""
        signals = {
            'Low': 'Buy Dips',
            'Normal': 'Selective',
            'Elevated': 'Hedged',
            'Crisis': 'Wait'
        }
        return signals.get(regime, 'Unknown')
    
    def _interpret_iv_rv(self, iv_stats: dict) -> str:
        """Interpret IV/RV stats for trading."""
        premium_pct = iv_stats['current_premium_pct']
        
        if premium_pct > 70:
            return "Premium selling favored"
        elif premium_pct < 40:
            return "Premium buying favored"
        else:
            return "Neutral edge"
        
        # Position on the RIGHT side, using x domain coordinates
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=0.82,  # Right side
            y=0.15,  # Bottom panel vertical center
            text=legend_text,
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            align='left',
            font=dict(size=9),
            bordercolor='#2E86AB',
            borderwidth=1.5,
            borderpad=8,
            bgcolor='rgba(255, 255, 255, 0.95)',
        )
    
    def save_chart(self, filename: str = 'spx_dashboard_v4.0.html', **kwargs):
        """Create and save the chart to HTML."""
        fig = self.plot_decision_chart(**kwargs)
        fig.write_html(filename)
        print(f"💾 Chart saved to {filename}")
        fig.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPX/VIX DECISION DASHBOARD v4.0")
    print("Now With ML Sector Rotation Predictions")
    print("="*60)
    
    viz = ConeVisualizerV4()
    viz.save_chart()
    
    print("\n✅ DONE!")
    print("\n📊 What's New in v4.0:")
    print("   • Panel 3: ML sector rotation predictions from v3.5")
    print("   • Real probabilities for 21-day outperformance")
    print("   • Confidence scoring (HIGH/MEDIUM/LOW)")
    print("   • Sector category analysis")
    print("   • Top picks highlighted in summary")
    print("\n💡 How to Use:")
    print("   1. Check summary for top ML rotation picks")
    print("   2. Focus on sectors with MEDIUM/HIGH confidence")
    print("   3. Rotation prob >55% = potential buy signal")
    print("   4. Consider macro-sensitive sectors (better model fit)")
    print("   5. Cross-reference with VIX regime and IV/RV edge")