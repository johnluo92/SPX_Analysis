"""
Phase 1: VIX Regime Analyzer
Goal: Calculate regime statistics WITHOUT changing UI yet

This module provides regime classification and transition analysis
for VIX levels, enabling better trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime


class RegimeAnalyzer:
    """
    Analyzes VIX regimes and calculates transition probabilities.
    
    Regime Definitions (flexible, can be adjusted):
        Low:      VIX < 15
        Normal:   15 <= VIX < 25  
        Elevated: 25 <= VIX < 35
        Crisis:   VIX >= 35
    """
    
    def __init__(self, 
                 low_threshold: float = 15,
                 normal_threshold: float = 25,
                 elevated_threshold: float = 35):
        """
        Initialize with custom thresholds if desired.
        
        Args:
            low_threshold: Upper bound for "Low" regime
            normal_threshold: Upper bound for "Normal" regime
            elevated_threshold: Upper bound for "Elevated" regime
        """
        self.low_threshold = low_threshold
        self.normal_threshold = normal_threshold
        self.elevated_threshold = elevated_threshold
    
    def classify_regime(self, vix: float) -> str:
        """
        Classify a single VIX level into a regime.
        
        Args:
            vix: VIX level
            
        Returns:
            Regime name: 'Low', 'Normal', 'Elevated', or 'Crisis'
        """
        if vix < self.low_threshold:
            return 'Low'
        elif vix < self.normal_threshold:
            return 'Normal'
        elif vix < self.elevated_threshold:
            return 'Elevated'
        else:
            return 'Crisis'
    
    def classify_series(self, vix_series: pd.Series) -> pd.Series:
        """
        Classify an entire VIX time series into regimes.
        
        Args:
            vix_series: Series of VIX values
            
        Returns:
            Series of regime labels
        """
        return vix_series.apply(self.classify_regime)
    
    def calculate_velocity(self, 
                          vix_series: pd.Series, 
                          window: int = 5) -> pd.Series:
        """
        Calculate VIX velocity (rate of change).
        
        Args:
            vix_series: Series of VIX values
            window: Days to look back for change calculation
            
        Returns:
            Series of VIX changes over window
        """
        return vix_series - vix_series.shift(window)
    
    def analyze_transitions(self, vix_series: pd.Series) -> Dict:
        """
        Analyze regime transition probabilities and durations.
        
        Args:
            vix_series: Historical VIX data with DatetimeIndex
            
        Returns:
            Dictionary with regime statistics:
            {
                'Low': {
                    'count': 450,
                    'avg_duration_days': 47,
                    'median_duration_days': 32,
                    'transitions': {'Low': 0.70, 'Normal': 0.25, ...},
                    'avg_vix': 13.2
                },
                ...
            }
        """
        # Classify all points
        regimes = self.classify_series(vix_series)
        
        # Find regime runs (consecutive days in same regime)
        regime_changes = regimes != regimes.shift(1)
        regime_runs = regime_changes.cumsum()
        
        results = {}
        
        for regime_name in ['Low', 'Normal', 'Elevated', 'Crisis']:
            # Filter to this regime
            regime_mask = regimes == regime_name
            regime_points = vix_series[regime_mask]
            
            if len(regime_points) == 0:
                results[regime_name] = {
                    'count': 0,
                    'avg_duration_days': 0,
                    'median_duration_days': 0,
                    'transitions': {},
                    'avg_vix': 0,
                    'std_vix': 0
                }
                continue
            
            # Calculate durations (how long each regime run lasted)
            regime_run_ids = regime_runs[regime_mask]
            durations = regime_run_ids.value_counts()
            
            # Calculate transitions (what happens after this regime?)
            # Look at the regime that follows each occurrence of this regime
            transition_counts = {'Low': 0, 'Normal': 0, 'Elevated': 0, 'Crisis': 0}
            
            for idx in range(len(regimes) - 1):
                if regimes.iloc[idx] == regime_name:
                    next_regime = regimes.iloc[idx + 1]
                    transition_counts[next_regime] += 1
            
            total_transitions = sum(transition_counts.values())
            transition_probs = {
                k: v / total_transitions if total_transitions > 0 else 0
                for k, v in transition_counts.items()
            }
            
            results[regime_name] = {
                'count': len(regime_points),
                'pct_of_time': len(regime_points) / len(vix_series) * 100,
                'avg_duration_days': durations.mean(),
                'median_duration_days': durations.median(),
                'max_duration_days': durations.max(),
                'transitions': transition_probs,
                'avg_vix': regime_points.mean(),
                'std_vix': regime_points.std(),
                'min_vix': regime_points.min(),
                'max_vix': regime_points.max()
            }
        
        return results
    
    def get_current_regime_context(self, 
                                   vix_series: pd.Series,
                                   current_vix: float = None) -> Dict:
        """
        Get context for the current VIX level based on historical patterns.
        
        Args:
            vix_series: Historical VIX data
            current_vix: Current VIX level (uses last value if None)
            
        Returns:
            Dictionary with current regime context
        """
        if current_vix is None:
            current_vix = vix_series.iloc[-1]
        
        current_regime = self.classify_regime(current_vix)
        transitions = self.analyze_transitions(vix_series)
        regime_stats = transitions[current_regime]
        
        # Calculate velocity
        velocity = self.calculate_velocity(vix_series)
        current_velocity = velocity.iloc[-1]
        
        # Determine velocity classification
        if abs(current_velocity) < 2:
            velocity_label = "Stable"
        elif current_velocity >= 2:
            velocity_label = "Rising Fast"
        else:
            velocity_label = "Falling Fast"
        
        return {
            'regime': current_regime,
            'vix': current_vix,
            'velocity': current_velocity,
            'velocity_label': velocity_label,
            'typical_duration': regime_stats['avg_duration_days'],
            'transition_probs': regime_stats['transitions'],
            'regime_stats': regime_stats
        }
    
    def get_regime_color(self, regime: str) -> str:
        """Get color for regime visualization."""
        colors = {
            'Low': '#2ECC71',      # Green
            'Normal': '#F39C12',   # Orange/Yellow
            'Elevated': '#E67E22', # Dark Orange
            'Crisis': '#E74C3C'    # Red
        }
        return colors.get(regime, 'gray')
    
    def get_regime_background_color(self, regime: str, alpha: float = 0.15) -> str:
        """Get semi-transparent background color for regime zones."""
        colors = {
            'Low': f'rgba(46, 204, 113, {alpha})',
            'Normal': f'rgba(243, 156, 18, {alpha})',
            'Elevated': f'rgba(230, 126, 34, {alpha})',
            'Crisis': f'rgba(231, 76, 60, {alpha})'
        }
        return colors.get(regime, f'rgba(128, 128, 128, {alpha})')


def test_regime_analyzer():
    """Test the regime analyzer with sample data."""
    print("\n" + "="*60)
    print("PHASE 1 TEST: Regime Analyzer")
    print("="*60)
    
    # Try to load real VIX data
    try:
        from UnifiedDataFetcher import UnifiedDataFetcher
        
        print("\n📊 Fetching VIX data (1990-2025)...")
        fetcher = UnifiedDataFetcher()
        end_date = datetime.now()
        start_date = datetime(1990, 1, 1)  # 35 years of data for robust statistics
        
        vix_data = fetcher.fetch_vix(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"✅ Loaded {len(vix_data)} days of VIX data")
        print(f"   Range: {vix_data.index[0].date()} to {vix_data.index[-1].date()}")
        
    except Exception as e:
        print(f"⚠️  Could not fetch real data: {e}")
        print("   Creating synthetic data for testing...")
        
        # Create synthetic VIX data for testing
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        vix_data = pd.Series(
            np.random.normal(20, 8, len(dates)).clip(10, 80),
            index=dates
        )
    
    # Initialize analyzer
    print("\n🔧 Initializing RegimeAnalyzer...")
    analyzer = RegimeAnalyzer()
    
    # Test classification
    print("\n📋 Testing regime classification...")
    current_vix = vix_data.iloc[-1]
    print(f"   Current VIX: {current_vix:.2f}")
    print(f"   Regime: {analyzer.classify_regime(current_vix)}")
    
    # Test velocity
    print("\n📈 Testing velocity calculation...")
    velocity = analyzer.calculate_velocity(vix_data)
    print(f"   Current 5-day change: {velocity.iloc[-1]:+.2f}")
    print(f"   Average velocity: {velocity.mean():+.2f}")
    
    # Test transition analysis
    print("\n🔄 Analyzing regime transitions...")
    transitions = analyzer.analyze_transitions(vix_data)
    
    print("\n" + "-"*60)
    print("REGIME STATISTICS")
    print("-"*60)
    
    for regime, stats in transitions.items():
        if stats['count'] > 0:
            print(f"\n{regime.upper()} Regime:")
            print(f"   Occurred: {stats['count']} days ({stats['pct_of_time']:.1f}% of time)")
            print(f"   Avg Duration: {stats['avg_duration_days']:.1f} days")
            print(f"   Median Duration: {stats['median_duration_days']:.1f} days")
            print(f"   Longest Run: {stats['max_duration_days']:.0f} days")
            print(f"   Avg VIX: {stats['avg_vix']:.2f} (±{stats['std_vix']:.2f})")
            print(f"   Range: {stats['min_vix']:.2f} - {stats['max_vix']:.2f}")
            print(f"   Transitions to:")
            for next_regime, prob in stats['transitions'].items():
                if prob > 0.01:  # Only show meaningful probabilities
                    print(f"      → {next_regime}: {prob*100:.1f}%")
    
    # Test current regime context
    print("\n" + "-"*60)
    print("CURRENT REGIME CONTEXT")
    print("-"*60)
    
    context = analyzer.get_current_regime_context(vix_data)
    print(f"\nRegime: {context['regime']}")
    print(f"VIX: {context['vix']:.2f}")
    print(f"Velocity: {context['velocity']:+.2f} ({context['velocity_label']})")
    print(f"Typical Duration: {context['typical_duration']:.1f} days")
    print(f"\nMost Likely Next Moves:")
    for regime, prob in sorted(context['transition_probs'].items(), 
                               key=lambda x: x[1], reverse=True):
        if prob > 0.05:
            print(f"   → {regime}: {prob*100:.0f}%")
    
    print("\n" + "="*60)
    print("✅ Phase 1 Test Complete!")
    print("="*60)
    
    print("\n💡 Insights:")
    print("   • Regime classification works")
    print("   • Transition probabilities are calculated")
    print("   • Ready for Phase 2A (visualization)")
    
    return analyzer, vix_data, transitions


if __name__ == "__main__":
    analyzer, vix_data, transitions = test_regime_analyzer()