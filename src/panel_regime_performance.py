"""
Panel 3: Regime Performance Matrix

Shows EXACTLY what to expect in each VIX regime based on 35 years of data.
This is what TradingView can't give you - regime-specific performance stats.

Answers:
- What are SPY returns in this regime?
- What's my win rate?
- What should I do?
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict
from regime_analyzer import RegimeAnalyzer


class RegimePerformancePanel:
    """Generates regime-specific performance statistics."""
    
    def __init__(self, analyzer: RegimeAnalyzer = None):
        self.analyzer = analyzer or RegimeAnalyzer()
        
        # Color scheme
        self.regime_colors = {
            'Low': '#2ECC71',
            'Normal': '#F39C12',
            'Elevated': '#E67E22',
            'Crisis': '#E74C3C'
        }
    
    def calculate_regime_performance(self,
                                    spx_data: pd.Series,
                                    vix_data: pd.Series,
                                    forward_periods: list = [5, 21]) -> Dict:
        """
        Calculate forward returns by regime.
        
        Args:
            spx_data: SPX closing prices (full history)
            vix_data: VIX levels (full history)
            forward_periods: Days forward to calculate returns [1-week, 1-month]
        
        Returns:
            Dict with regime performance stats
        """
        print("\n🔧 Debug: Starting performance calculation...")
        
        # Normalize indices - remove timezone info and ensure same type
        spx_data = spx_data.copy()
        vix_data = vix_data.copy()
        
        # Convert to timezone-naive if needed
        if hasattr(spx_data.index, 'tz') and spx_data.index.tz is not None:
            spx_data.index = spx_data.index.tz_localize(None)
        if hasattr(vix_data.index, 'tz') and vix_data.index.tz is not None:
            vix_data.index = vix_data.index.tz_localize(None)
        
        # Normalize to date only (remove time component)
        spx_data.index = pd.to_datetime(spx_data.index.date)
        vix_data.index = pd.to_datetime(vix_data.index.date)
        
        # Align data
        common_dates = spx_data.index.intersection(vix_data.index)
        spx_aligned = spx_data.loc[common_dates].sort_index()
        vix_aligned = vix_data.loc[common_dates].sort_index()
        
        print(f"   ✅ Aligned {len(spx_aligned)} days of data")
        
        # Classify all days
        regimes = self.analyzer.classify_series(vix_aligned)
        
        results = {}
        
        for regime_name in ['Low', 'Normal', 'Elevated', 'Crisis']:
            regime_mask = (regimes == regime_name)
            regime_dates = regimes[regime_mask].index
            
            if len(regime_dates) == 0:
                results[regime_name] = self._empty_stats()
                continue
            
            # Calculate forward returns for each period
            period_stats = {}
            
            for days in forward_periods:
                returns = []
                
                for i, date in enumerate(regime_dates):
                    # Find position in aligned series
                    try:
                        idx = spx_aligned.index.get_loc(date)
                        
                        # Check if we have enough future data
                        if idx + days < len(spx_aligned):
                            start_price = spx_aligned.iloc[idx]
                            end_price = spx_aligned.iloc[idx + days]
                            
                            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                                ret = (end_price / start_price - 1) * 100
                                returns.append(ret)
                    except Exception as e:
                        continue
                
                if len(returns) > 10:  # Need meaningful sample
                    returns_array = np.array(returns)
                    win_rate = (returns_array > 0).sum() / len(returns_array) * 100
                    avg_return = returns_array.mean()
                    median_return = np.median(returns_array)
                    
                    wins = returns_array[returns_array > 0]
                    losses = returns_array[returns_array <= 0]
                    
                    avg_win = wins.mean() if len(wins) > 0 else 0
                    avg_loss = losses.mean() if len(losses) > 0 else 0
                    
                    # Risk-adjusted metric (like Sharpe)
                    std = returns_array.std()
                    sharpe = avg_return / std if std > 0 else 0
                    
                    period_stats[f'{days}d'] = {
                        'avg_return': avg_return,
                        'median_return': median_return,
                        'win_rate': win_rate,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'sharpe': sharpe,
                        'sample_size': len(returns)
                    }
                else:
                    period_stats[f'{days}d'] = {
                        'avg_return': 0,
                        'median_return': 0,
                        'win_rate': 0,
                        'avg_win': 0,
                        'avg_loss': 0,
                        'sharpe': 0,
                        'sample_size': len(returns)
                    }
            
            results[regime_name] = period_stats
        
        print(f"   ✅ Performance calculated for all regimes\n")
        return results
    
    def _empty_stats(self):
        """Return empty stats structure."""
        return {
            '5d': {'avg_return': 0, 'median_return': 0, 'win_rate': 0, 
                   'avg_win': 0, 'avg_loss': 0, 'sharpe': 0, 'sample_size': 0},
            '21d': {'avg_return': 0, 'median_return': 0, 'win_rate': 0, 
                    'avg_win': 0, 'avg_loss': 0, 'sharpe': 0, 'sample_size': 0}
        }
    
    def _determine_action(self, regime: str, stats: Dict) -> str:
        """Determine trading action based on regime and stats."""
        monthly = stats.get('21d', {})
        win_rate = monthly.get('win_rate', 0)
        
        if regime == 'Low':
            return "BUY DIPS"
        elif regime == 'Normal':
            if win_rate >= 60:
                return "HOLD"
            else:
                return "SELECTIVE"
        elif regime == 'Elevated':
            # High returns but high risk - be selective
            return "SELECTIVE BUY"
        else:  # Crisis
            # Highest returns but maximum drawdown risk
            return "DRY POWDER"
    
    def add_to_figure(self,
                      fig: go.Figure,
                      performance_stats: Dict,
                      current_regime: str,
                      row: int = 3,
                      col: int = 1) -> go.Figure:
        """
        Add performance matrix table to figure.
        
        Args:
            fig: Plotly figure object
            performance_stats: Performance statistics from calculate_regime_performance
            current_regime: Current regime name
            row: Subplot row
            col: Subplot column
        """
        # Build table data
        regimes = ['Low', 'Normal', 'Elevated', 'Crisis']
        
        # Headers
        headers = [
            '<b>Regime</b>',
            '<b>1-Week Return</b>',
            '<b>1-Month Return</b>',
            '<b>Win Rate</b>',
            '<b>Risk/Reward</b>',
            '<b>Action</b>'
        ]
        
        # Cells
        regime_col = []
        week_col = []
        month_col = []
        winrate_col = []
        rr_col = []
        action_col = []
        
        cell_colors = []
        
        for regime in regimes:
            stats = performance_stats.get(regime, self._empty_stats())
            week_stats = stats.get('5d', {})
            month_stats = stats.get('21d', {})
            
            # Highlight current regime
            is_current = (regime == current_regime)
            bg_color = 'rgba(46, 134, 171, 0.2)' if is_current else 'white'
            
            regime_col.append(f"<b>{regime}</b>" if is_current else regime)
            
            week_return = week_stats.get('avg_return', 0)
            week_col.append(f"<b>{week_return:+.2f}%</b>" if is_current else f"{week_return:+.2f}%")
            
            month_return = month_stats.get('avg_return', 0)
            month_col.append(f"<b>{month_return:+.2f}%</b>" if is_current else f"{month_return:+.2f}%")
            
            win_rate = month_stats.get('win_rate', 0)
            winrate_col.append(f"<b>{win_rate:.0f}%</b>" if is_current else f"{win_rate:.0f}%")
            
            avg_win = month_stats.get('avg_win', 0)
            avg_loss = month_stats.get('avg_loss', 0)
            rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            rr_col.append(f"<b>{rr_ratio:.2f}</b>" if is_current else f"{rr_ratio:.2f}")
            
            action = self._determine_action(regime, stats)
            action_col.append(f"<b>{action}</b>" if is_current else action)
            
            cell_colors.append(bg_color)
        
        # Create table
        fig.add_trace(go.Table(
            header=dict(
                values=headers,
                fill_color='#2E86AB',
                font=dict(color='white', size=13, family='Arial Black'),
                align='center',
                height=35
            ),
            cells=dict(
                values=[
                    regime_col,
                    week_col,
                    month_col,
                    winrate_col,
                    rr_col,
                    action_col
                ],
                fill_color=[cell_colors] * 6,
                font=dict(size=12),
                align=['center', 'center', 'center', 'center', 'center', 'center'],
                height=30
            )
        ), row=row, col=col)
        
        # Add insight annotation below table
        current_stats = performance_stats.get(current_regime, self._empty_stats())
        month_stats = current_stats.get('21d', {})
        
        insight_text = (
            f"<b>Current Regime Insight ({current_regime}):</b><br>"
            f"Based on {month_stats.get('sample_size', 0)} historical occurrences:<br>"
            f"• Expected 1-month return: <b>{month_stats.get('avg_return', 0):+.2f}%</b><br>"
            f"• Probability of profit: <b>{month_stats.get('win_rate', 0):.0f}%</b><br>"
            f"• Avg win: {month_stats.get('avg_win', 0):+.2f}% | Avg loss: {month_stats.get('avg_loss', 0):.2f}%<br>"
            f"• Sharpe-like ratio: {month_stats.get('sharpe', 0):.2f}<br>"
            f"<br>"
            f"<i>This is what happened AFTER entering {current_regime} regime in the past.<br>"
            f"Use this to calibrate position sizing and hedge ratios.</i>"
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
            font=dict(size=11),
            bordercolor='#2E86AB',
            borderwidth=2,
            borderpad=10,
            bgcolor='rgba(255, 255, 255, 0.95)',
        )
        
        return fig


# Standalone test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING: Regime Performance Matrix")
    print("="*60)
    
    from UnifiedDataFetcher import UnifiedDataFetcher
    from datetime import datetime
    from plotly.subplots import make_subplots
    
    # Fetch data
    print("\n📊 Fetching 35 years of data...")
    fetcher = UnifiedDataFetcher()
    
    spx_data = fetcher.fetch_spx_close(
        datetime(1990, 1, 1).strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d')
    )
    
    vix_data = fetcher.fetch_vix(
        datetime(1990, 1, 1).strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d')
    )
    
    print(f"✅ Loaded SPX: {len(spx_data)} days")
    print(f"✅ Loaded VIX: {len(vix_data)} days")
    
    # Calculate performance
    print("\n🔧 Calculating regime performance stats...")
    panel = RegimePerformancePanel()
    performance = panel.calculate_regime_performance(spx_data, vix_data)
    
    # Print summary
    print("\n" + "-"*60)
    print("PERFORMANCE BY REGIME")
    print("-"*60)
    
    for regime, stats in performance.items():
        month_stats = stats.get('21d', {})
        print(f"\n{regime.upper()}:")
        print(f"  1-Month Avg: {month_stats.get('avg_return', 0):+.2f}%")
        print(f"  Win Rate: {month_stats.get('win_rate', 0):.0f}%")
        print(f"  Sample Size: {month_stats.get('sample_size', 0)}")
    
    # Visualize
    print("\n🎨 Creating visualization...")
    
    analyzer = RegimeAnalyzer()
    current_regime = analyzer.classify_regime(vix_data.iloc[-1])
    
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "table"}]],
        subplot_titles=["<b>SPX Performance by VIX Regime</b><br><sub>35 Years of Data (1990-2025)</sub>"]
    )
    
    fig = panel.add_to_figure(
        fig=fig,
        performance_stats=performance,
        current_regime=current_regime,
        row=1, col=1
    )
    
    fig.update_layout(
        width=1400,
        height=600,
        template='plotly_white'
    )
    
    fig.show()
    fig.write_html('test_regime_performance.html')
    
    print("\n✅ Test complete!")
    print("💾 Saved to test_regime_performance.html")
    print("\n" + "="*60)
    print("READY TO INTEGRATE")
    print("="*60)