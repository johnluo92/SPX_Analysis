"""
Panel 3: ML-Driven Sector Rotation Predictions

Displays v3.5 machine learning model predictions for sector rotation.
Shows which sectors are likely to outperform SPY in the next 21 days.

REPLACES: Regime Performance Matrix (moved to historical analysis)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class SectorRotationPanel:
    """Generates ML-driven sector rotation predictions panel."""
    
    def __init__(self):
        self.sector_names = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Health Care',
            'XLI': 'Industrials',
            'XLY': 'Consumer Disc.',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLC': 'Communication',
            'XLB': 'Materials'
        }
        
        self.category_colors = {
            'FINANCIALS': '#3498DB',
            'MACRO_SENSITIVE': '#2ECC71',
            'SENTIMENT_DRIVEN': '#E74C3C',
            'MIXED': '#F39C12'
        }
        
        self.confidence_colors = {
            'HIGH': '#2ECC71',
            'MEDIUM': '#F39C12',
            'LOW': '#E74C3C'
        }
    
    def add_to_figure(self,
                      fig: go.Figure,
                      predictions: pd.DataFrame,
                      model_results: Dict,
                      confidence_df: pd.DataFrame,
                      row: int = 3,
                      col: int = 1) -> go.Figure:
        """
        Add sector rotation predictions table to figure.
        
        Args:
            fig: Plotly figure object
            predictions: DataFrame with rotation probabilities (sorted)
            model_results: Dict with test accuracy, gap, etc.
            confidence_df: DataFrame with confidence tiers
            row: Subplot row
            col: Subplot column
        """
        # Prepare data for table
        tickers = predictions.index.tolist()
        
        sector_col = [self.sector_names.get(t, t) for t in tickers]
        ticker_col = tickers
        prob_col = [f"{p:.1%}" for p in predictions['Probability']]
        category_col = [predictions.loc[t, 'Category'] for t in tickers]
        confidence_col = [predictions.loc[t, 'Confidence'] for t in tickers]
        accuracy_col = [f"{predictions.loc[t, 'Test_Acc']:.1%}" for t in tickers]
        gap_col = [f"{predictions.loc[t, 'Gap']:.2f}" for t in tickers]
        
        # Color coding
        cell_colors = []
        for ticker in tickers:
            prob = predictions.loc[ticker, 'Probability']
            confidence = predictions.loc[ticker, 'Confidence']
            
            # High probability + medium/high confidence = green tint
            # Low confidence = red tint
            if prob > 0.55 and confidence in ['HIGH', 'MEDIUM']:
                color = 'rgba(46, 204, 113, 0.15)'  # Green tint
            elif confidence == 'LOW':
                color = 'rgba(231, 76, 60, 0.10)'   # Red tint
            else:
                color = 'white'
            
            cell_colors.append(color)
        
        # Create table with fixed height to show all rows without scrolling
        # 11 sectors × 24px per row + 32px header = ~296px total
        fig.add_trace(go.Table(
            header=dict(
                values=[
                    '<b>Sector</b>',
                    '<b>Ticker</b>',
                    '<b>Rotation Prob</b>',
                    '<b>Category</b>',
                    '<b>Confidence</b>',
                    '<b>Test Acc</b>',
                    '<b>Gap</b>'
                ],
                fill_color='#2E86AB',
                font=dict(color='white', size=11, family='Arial Black'),
                align=['left', 'center', 'center', 'left', 'center', 'center', 'center'],
                height=28
            ),
            cells=dict(
                values=[
                    sector_col,
                    ticker_col,
                    prob_col,
                    category_col,
                    confidence_col,
                    accuracy_col,
                    gap_col
                ],
                fill_color=[cell_colors] * 7,
                font=dict(size=10),
                align=['left', 'center', 'center', 'left', 'center', 'center', 'center'],
                height=24
            )
        ), row=row, col=col)
        
        # Add interpretation box
        top_3 = tickers[:3]
        top_3_names = [self.sector_names.get(t, t) for t in top_3]
        top_3_probs = [predictions.loc[t, 'Probability'] for t in top_3]
        top_3_conf = [predictions.loc[t, 'Confidence'] for t in top_3]
        
        # Build recommendation text
        strong_picks = []
        for name, prob, conf in zip(top_3_names, top_3_probs, top_3_conf):
            if prob > 0.55 and conf in ['HIGH', 'MEDIUM']:
                strong_picks.append(f"{name} ({prob:.1%})")
        
        if strong_picks:
            rec_text = f"<b>Top Picks:</b> {', '.join(strong_picks)}"
        else:
            rec_text = "<b>Caution:</b> No strong rotation signals"
        
        # Calculate average metrics
        avg_gap = predictions['Gap'].mean()
        high_conf_count = len(confidence_df[confidence_df['Tier'] == 'HIGH'])
        medium_conf_count = len(confidence_df[confidence_df['Tier'] == 'MEDIUM'])
        
        insight_text = (
            f"<b>ML Sector Rotation Predictions (21-Day Horizon)</b><br>"
            f"<br>"
            f"{rec_text}<br>"
            f"<br>"
            f"<b>Model Quality:</b><br>"
            f"• High Confidence: {high_conf_count} sectors<br>"
            f"• Medium Confidence: {medium_conf_count} sectors<br>"
            f"• Avg Overfitting Gap: {avg_gap:.2f}<br>"
            f"<br>"
            f"<b>How to Use:</b><br>"
            f"• Focus on HIGH/MEDIUM confidence<br>"
            f"• Rotation Prob >55% = potential buy<br>"
            f"• Consider category (macro-sensitive<br>"
            f"  sectors perform better in model)<br>"
            f"<br>"
            f"<b>Color Guide:</b><br>"
            f"• <span style='background-color:rgba(46,204,113,0.3)'>Green</span> = Strong signal (>55% + good confidence)<br>"
            f"• <span style='background-color:rgba(231,76,60,0.2)'>Red</span> = Low confidence (use caution)<br>"
            f"<br>"
            f"<i>Model trained on 7 years of data<br>"
            f"with sector-specific hyperparameters</i>"
        )
        
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.28,  # Position below Panel 3
            text=insight_text,
            showarrow=False,
            xanchor='center',
            yanchor='top',
            align='left',
            font=dict(size=10),
            bordercolor='#2E86AB',
            borderwidth=2,
            borderpad=10,
            bgcolor='rgba(255, 255, 255, 0.95)',
        )
        
        return fig
    
    def create_category_summary(self, predictions: pd.DataFrame) -> str:
        """Create a text summary by category."""
        categories = predictions.groupby('Category')
        
        summary_lines = []
        for cat_name, group in categories:
            avg_prob = group['Probability'].mean()
            avg_acc = group['Test_Acc'].mean()
            best_ticker = group['Probability'].idxmax()
            best_prob = group.loc[best_ticker, 'Probability']
            
            summary_lines.append(
                f"<b>{cat_name}:</b> Avg {avg_prob:.1%} | "
                f"Best: {best_ticker} ({best_prob:.1%})"
            )
        
        return "<br>".join(summary_lines)


# Standalone test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING: Sector Rotation Panel")
    print("="*60)
    
    # Create mock data
    tickers = ['XLF', 'XLY', 'XLK', 'XLU', 'XLB', 'XLC', 'XLI', 
               'XLRE', 'XLV', 'XLE', 'XLP']
    
    mock_predictions = pd.DataFrame({
        'Probability': [0.625, 0.603, 0.580, 0.550, 0.528, 0.486, 
                       0.462, 0.446, 0.394, 0.388, 0.260],
        'Test_Acc': [0.560, 0.480, 0.544, 0.459, 0.590, 0.477,
                    0.443, 0.661, 0.489, 0.584, 0.596],
        'Gap': [0.286, 0.313, 0.269, 0.400, 0.275, 0.334,
               0.388, 0.204, 0.302, 0.298, 0.244],
        'Category': ['FINANCIALS', 'SENTIMENT_DRIVEN', 'MIXED', 
                    'MIXED', 'MACRO_SENSITIVE', 'SENTIMENT_DRIVEN',
                    'SENTIMENT_DRIVEN', 'MACRO_SENSITIVE', 
                    'SENTIMENT_DRIVEN', 'MACRO_SENSITIVE', 
                    'MACRO_SENSITIVE'],
        'Confidence': ['MEDIUM', 'LOW', 'MEDIUM', 'LOW', 'MEDIUM',
                      'LOW', 'LOW', 'MEDIUM', 'LOW', 'MEDIUM', 'MEDIUM']
    }, index=tickers)
    
    mock_results = {
        ticker: {
            'test_accuracy': row['Test_Acc'],
            'overfitting_gap': row['Gap'],
            'category': row['Category']
        }
        for ticker, row in mock_predictions.iterrows()
    }
    
    mock_confidence = pd.DataFrame({
        'Sector': tickers,
        'Tier': mock_predictions['Confidence'].values,
        'Category': mock_predictions['Category'].values
    })
    
    # Create figure
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "table"}]],
        subplot_titles=["<b>ML Sector Rotation Predictions</b>"]
    )
    
    panel = SectorRotationPanel()
    fig = panel.add_to_figure(
        fig=fig,
        predictions=mock_predictions,
        model_results=mock_results,
        confidence_df=mock_confidence,
        row=1, col=1
    )
    
    fig.update_layout(
        width=1400,
        height=700,
        template='plotly_white'
    )
    
    fig.show()
    fig.write_html('test_sector_rotation_panel.html')
    
    print("\n✅ Test complete!")
    print("💾 Saved to test_sector_rotation_panel.html")