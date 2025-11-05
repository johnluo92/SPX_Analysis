"""
VIX Regime Discovery & Analysis Tool
Data-driven regime identification and transition analysis

This script discovers natural VIX regimes from historical data and analyzes:
1. Optimal regime boundaries (not arbitrary)
2. Transition probabilities between regimes
3. Regime duration statistics
4. Feature importance for regime changes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from UnifiedDataFetcher import UnifiedDataFetcher


class VIXRegimeAnalyzer:
    """
    Discover natural VIX regimes and transition patterns from historical data.
    """
    
    def __init__(self):
        self.fetcher = UnifiedDataFetcher()
        self.vix = None
        self.regimes = None
        self.regime_labels = None
        self.transition_matrix = None
        self.regime_stats = None
        
    def load_data(self, years: int = 15):
        """Load VIX historical data."""
        print("\n" + "="*80)
        print("VIX REGIME DISCOVERY - DATA-DRIVEN ANALYSIS")
        print("="*80)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        print(f"\n📊 Loading {years} years of VIX data...")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        
        vix = self.fetcher.fetch_vix(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        
        self.vix = vix.dropna()
        
        print(f"✅ Loaded {len(self.vix)} trading days")
        print(f"   Range: {self.vix.min():.2f} to {self.vix.max():.2f}")
        print(f"   Mean: {self.vix.mean():.2f} | Median: {self.vix.median():.2f}")
        
        return self.vix
    
    def discover_regimes(self, method: str = 'gmm', n_regimes: int = 4):
        """
        Discover natural VIX regimes using statistical methods.
        
        Methods:
        - 'gmm': Gaussian Mixture Model (best for VIX)
        - 'quantile': Quantile-based (simple, interpretable)
        - 'kde': Kernel Density Estimation (finds natural valleys)
        """
        print("\n" + "="*80)
        print(f"REGIME DISCOVERY: {method.upper()} METHOD")
        print("="*80)
        
        if method == 'gmm':
            regimes, boundaries = self._discover_gmm(n_regimes)
        elif method == 'quantile':
            regimes, boundaries = self._discover_quantile(n_regimes)
        elif method == 'kde':
            regimes, boundaries = self._discover_kde(n_regimes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.regimes = regimes
        self.regime_labels = [f"Regime {i}" for i in range(n_regimes)]
        
        # Print regime statistics
        print("\n" + "-"*80)
        print("DISCOVERED REGIME BOUNDARIES")
        print("-"*80)
        
        for i in range(n_regimes):
            regime_mask = regimes == i
            regime_vix = self.vix[regime_mask]
            
            if len(regime_vix) > 0:
                pct = len(regime_vix) / len(self.vix) * 100
                print(f"\nRegime {i}: {boundaries[i]:.2f} to {boundaries[i+1]:.2f}")
                print(f"   Frequency: {len(regime_vix)}/{len(self.vix)} days ({pct:.1f}%)")
                print(f"   Mean VIX: {regime_vix.mean():.2f}")
                print(f"   Std Dev: {regime_vix.std():.2f}")
        
        return regimes, boundaries
    
    def _discover_gmm(self, n_regimes: int):
        """Gaussian Mixture Model - finds natural clusters in VIX distribution."""
        print(f"\n🔬 Fitting {n_regimes}-component Gaussian Mixture Model...")
        
        X = self.vix.values.reshape(-1, 1)
        
        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        regimes = gmm.fit_predict(X)
        
        # Sort regimes by mean VIX level
        means = [self.vix[regimes == i].mean() for i in range(n_regimes)]
        regime_order = np.argsort(means)
        regime_mapping = {old: new for new, old in enumerate(regime_order)}
        regimes = pd.Series(regimes).map(regime_mapping).values
        
        # Compute boundaries between regimes
        boundaries = []
        for i in range(n_regimes + 1):
            if i == 0:
                boundaries.append(self.vix.min())
            elif i == n_regimes:
                boundaries.append(self.vix.max())
            else:
                # Boundary = midpoint between adjacent regime means
                lower_mean = self.vix[regimes == i-1].mean()
                upper_mean = self.vix[regimes == i].mean()
                boundaries.append((lower_mean + upper_mean) / 2)
        
        print(f"   ✓ BIC Score: {gmm.bic(X):.2f}")
        print(f"   ✓ Converged: {gmm.converged_}")
        
        return regimes, boundaries
    
    def _discover_quantile(self, n_regimes: int):
        """Simple quantile-based regime assignment."""
        print(f"\n📊 Using {n_regimes} quantile-based regimes...")
        
        quantiles = np.linspace(0, 1, n_regimes + 1)
        boundaries = [self.vix.quantile(q) for q in quantiles]
        
        regimes = pd.cut(
            self.vix,
            bins=boundaries,
            labels=False,
            include_lowest=True
        )
        
        return regimes, boundaries
    
    def _discover_kde(self, n_regimes: int):
        """Kernel Density Estimation - finds valleys in distribution."""
        print(f"\n📈 Finding {n_regimes} regimes using KDE valleys...")
        
        from scipy.signal import find_peaks
        
        # Fit KDE
        kde = stats.gaussian_kde(self.vix.values)
        x_range = np.linspace(self.vix.min(), self.vix.max(), 1000)
        density = kde(x_range)
        
        # Find valleys (negative peaks in density)
        valleys, _ = find_peaks(-density, distance=50)
        
        if len(valleys) >= n_regimes - 1:
            # Take the deepest valleys
            valley_depths = -density[valleys]
            deepest = np.argsort(valley_depths)[-(n_regimes-1):]
            valley_positions = sorted(x_range[valleys[deepest]])
        else:
            # Fall back to quantiles
            print("   ⚠️  Not enough valleys found, using quantiles")
            return self._discover_quantile(n_regimes)
        
        # Create boundaries
        boundaries = [self.vix.min()] + valley_positions + [self.vix.max()]
        
        regimes = pd.cut(
            self.vix,
            bins=boundaries,
            labels=False,
            include_lowest=True
        )
        
        return regimes, boundaries
    
    def compute_transition_matrix(self, lag: int = 1):
        """
        Compute regime transition probability matrix.
        
        P[i,j] = P(regime_t+lag = j | regime_t = i)
        """
        print("\n" + "="*80)
        print(f"REGIME TRANSITION ANALYSIS (lag={lag} day{'s' if lag > 1 else ''})")
        print("="*80)
        
        if self.regimes is None:
            raise ValueError("Run discover_regimes() first")
        
        n_regimes = len(np.unique(self.regimes))
        
        # Build transition counts
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(self.regimes) - lag):
            current_regime = self.regimes[i]
            future_regime = self.regimes[i + lag]
            transition_counts[current_regime, future_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = transition_counts / row_sums
        
        # Print transition matrix
        print("\n" + "-"*80)
        print("TRANSITION PROBABILITY MATRIX")
        print("-"*80)
        print(f"{'From \\ To':<15}", end="")
        for j in range(n_regimes):
            print(f"Regime {j:>7}", end="")
        print()
        print("-"*80)
        
        for i in range(n_regimes):
            print(f"Regime {i:<8}", end="")
            for j in range(n_regimes):
                prob = self.transition_matrix[i, j]
                print(f"{prob:>10.1%}", end="")
            print()
        
        # Identify key transitions
        print("\n" + "-"*80)
        print("KEY TRANSITION PATTERNS")
        print("-"*80)
        
        for i in range(n_regimes):
            stay_prob = self.transition_matrix[i, i]
            print(f"\nRegime {i} → Persistence: {stay_prob:.1%}")
            
            # Most likely transitions (excluding staying)
            trans_probs = self.transition_matrix[i].copy()
            trans_probs[i] = 0  # Exclude staying
            
            if trans_probs.max() > 0.05:
                most_likely = trans_probs.argmax()
                print(f"   Most likely move: → Regime {most_likely} ({trans_probs[most_likely]:.1%})")
        
        return self.transition_matrix
    
    def analyze_regime_durations(self):
        """Analyze how long VIX stays in each regime."""
        print("\n" + "="*80)
        print("REGIME DURATION ANALYSIS")
        print("="*80)
        
        if self.regimes is None:
            raise ValueError("Run discover_regimes() first")
        
        # Identify regime runs
        regime_series = pd.Series(self.regimes, index=self.vix.index)
        regime_change = regime_series != regime_series.shift(1)
        run_id = regime_change.cumsum()
        
        # Compute duration statistics
        durations = {}
        for regime in np.unique(self.regimes):
            regime_runs = run_id[regime_series == regime]
            regime_durations = regime_runs.value_counts().values
            durations[regime] = regime_durations
        
        self.regime_stats = {}
        
        print("\n" + "-"*80)
        print("DURATION STATISTICS (trading days)")
        print("-"*80)
        
        for regime in sorted(durations.keys()):
            durs = durations[regime]
            self.regime_stats[regime] = {
                'mean_duration': durs.mean(),
                'median_duration': np.median(durs),
                'max_duration': durs.max(),
                'min_duration': durs.min()
            }
            
            print(f"\nRegime {regime}:")
            print(f"   Mean duration:   {durs.mean():.1f} days")
            print(f"   Median duration: {np.median(durs):.1f} days")
            print(f"   Range:           {durs.min()} to {durs.max()} days")
            print(f"   Total episodes:  {len(durs)}")
        
        return self.regime_stats
    
    def analyze_regime_triggers(self):
        """Analyze what features correlate with regime changes."""
        print("\n" + "="*80)
        print("REGIME CHANGE TRIGGERS")
        print("="*80)
        
        if self.regimes is None:
            raise ValueError("Run discover_regimes() first")
        
        # Build basic features
        vix_ret = self.vix.pct_change()
        vix_vol = vix_ret.rolling(20).std() * np.sqrt(252)
        vix_momentum = self.vix.diff(5)
        
        # Identify regime changes
        regime_series = pd.Series(self.regimes, index=self.vix.index)
        regime_change = (regime_series != regime_series.shift(1)).astype(int)
        
        # Compute correlations
        features = pd.DataFrame({
            'vix_level': self.vix,
            'vix_1d_change': self.vix.diff(1),
            'vix_5d_change': self.vix.diff(5),
            'vix_5d_volatility': vix_vol,
            'vix_momentum': vix_momentum
        })
        
        print("\n" + "-"*80)
        print("FEATURE CORRELATION WITH REGIME CHANGES")
        print("-"*80)
        
        correlations = {}
        for col in features.columns:
            corr = features[col].corr(regime_change)
            correlations[col] = abs(corr)
            print(f"{col:25s}: {corr:+.3f}")
        
        print("\n" + "-"*80)
        print("INTERPRETATION")
        print("-"*80)
        
        top_feature = max(correlations, key=correlations.get)
        print(f"\nStrongest predictor: {top_feature}")
        print(f"Correlation: {correlations[top_feature]:.3f}")
        
        return correlations
    
    def recommend_prediction_targets(self):
        """Based on analysis, recommend what to predict."""
        print("\n" + "="*80)
        print("RECOMMENDED PREDICTION TARGETS")
        print("="*80)
        
        if self.transition_matrix is None or self.regime_stats is None:
            print("\n⚠️  Run full analysis first:")
            print("   1. discover_regimes()")
            print("   2. compute_transition_matrix()")
            print("   3. analyze_regime_durations()")
            return
        
        n_regimes = len(self.transition_matrix)
        
        print("\n" + "-"*80)
        print("1. REGIME TRANSITION PREDICTIONS")
        print("-"*80)
        
        print("\nTarget: P(future_regime | current_regime, features)")
        print("Type: Multi-class classification")
        print("Horizons: 5d, 13d, 21d")
        print("\nWhy: Directly actionable for position sizing and hedging")
        print("Expected Accuracy: ~55-70% (based on transition persistence)")
        
        print("\n" + "-"*80)
        print("2. REGIME STABILITY PREDICTIONS")
        print("-"*80)
        
        print("\nTarget: P(stay_in_current_regime)")
        print("Type: Binary classification")
        print("Horizons: 5d, 13d, 21d")
        
        # Calculate average persistence
        avg_persistence = np.diag(self.transition_matrix).mean()
        print(f"\nWhy: High baseline accuracy ({avg_persistence:.1%} historical persistence)")
        print("Expected Accuracy: ~65-75%")
        
        print("\n" + "-"*80)
        print("3. VOLATILITY EXPANSION PREDICTIONS")
        print("-"*80)
        
        print("\nTarget: P(transition_to_higher_regime)")
        print("Type: Binary classification")
        print("Horizons: 5d, 13d, 21d")
        
        # Calculate upward transition rates
        upward_rates = []
        for i in range(n_regimes - 1):
            upward_prob = self.transition_matrix[i, i+1:].sum()
            upward_rates.append(upward_prob)
        
        avg_upward = np.mean(upward_rates) if upward_rates else 0
        print(f"\nWhy: Asymmetric risk management (avg upward transition: {avg_upward:.1%})")
        print("Expected Accuracy: ~60-70%")
        
        print("\n" + "-"*80)
        print("4. EXPECTED REGIME DURATION")
        print("-"*80)
        
        print("\nTarget: How many days until regime change?")
        print("Type: Regression (days remaining)")
        print("Horizons: Current regime")
        
        for regime in range(n_regimes):
            if regime in self.regime_stats:
                mean_dur = self.regime_stats[regime]['mean_duration']
                print(f"   Regime {regime} avg duration: {mean_dur:.1f} days")
        
        print("\nWhy: Position duration planning")
        print("Expected Accuracy: Moderate (high variance)")
        
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        print("\nPRIORITY ORDER:")
        print("1. Regime Transition (multi-class) - Most actionable")
        print("2. Regime Stability (binary) - Highest accuracy")
        print("3. Volatility Expansion (binary) - Risk management")
        print("4. Duration Prediction (regression) - Lower priority")
        
    def visualize_regimes(self, save_path: str = None):
        """Create comprehensive regime visualization."""
        if self.regimes is None:
            print("Run discover_regimes() first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VIX Regime Analysis', fontsize=16, fontweight='bold')
        
        # 1. VIX time series with regime coloring
        ax = axes[0, 0]
        for regime in np.unique(self.regimes):
            mask = self.regimes == regime
            ax.scatter(self.vix.index[mask], self.vix.values[mask], 
                      label=f'Regime {regime}', alpha=0.6, s=1)
        ax.set_ylabel('VIX Level')
        ax.set_title('VIX History with Discovered Regimes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Regime distribution
        ax = axes[0, 1]
        for regime in np.unique(self.regimes):
            regime_vix = self.vix[self.regimes == regime]
            ax.hist(regime_vix, bins=50, alpha=0.6, label=f'Regime {regime}')
        ax.set_xlabel('VIX Level')
        ax.set_ylabel('Frequency')
        ax.set_title('VIX Distribution by Regime')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Transition matrix heatmap
        if self.transition_matrix is not None:
            ax = axes[1, 0]
            im = ax.imshow(self.transition_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(self.transition_matrix)))
            ax.set_yticks(range(len(self.transition_matrix)))
            ax.set_xlabel('To Regime')
            ax.set_ylabel('From Regime')
            ax.set_title('Regime Transition Probabilities')
            
            # Add text annotations
            for i in range(len(self.transition_matrix)):
                for j in range(len(self.transition_matrix)):
                    text = ax.text(j, i, f'{self.transition_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
            
            plt.colorbar(im, ax=ax)
        
        # 4. Regime duration box plot
        if self.regime_stats is not None:
            ax = axes[1, 1]
            regime_series = pd.Series(self.regimes, index=self.vix.index)
            regime_change = regime_series != regime_series.shift(1)
            run_id = regime_change.cumsum()
            
            duration_data = []
            duration_labels = []
            for regime in sorted(np.unique(self.regimes)):
                regime_runs = run_id[regime_series == regime]
                durations = regime_runs.value_counts().values
                duration_data.append(durations)
                duration_labels.append(f'Regime {regime}')
            
            ax.boxplot(duration_data, labels=duration_labels)
            ax.set_ylabel('Duration (days)')
            ax.set_title('Regime Duration Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run complete VIX regime analysis."""
    analyzer = VIXRegimeAnalyzer()
    
    # Load data
    analyzer.load_data(years=15)
    
    # Try different methods
    methods = ['gmm', 'quantile', 'kde']
    
    for method in methods:
        try:
            print("\n" + "="*80)
            print(f"TESTING METHOD: {method.upper()}")
            print("="*80)
            
            analyzer.discover_regimes(method=method, n_regimes=4)
            analyzer.compute_transition_matrix(lag=5)  # 5-day transitions
            analyzer.analyze_regime_durations()
            analyzer.analyze_regime_triggers()
            
            print("\n" + "="*80)
            print(f"COMPLETED: {method.upper()} METHOD")
            print("="*80)
            
        except Exception as e:
            print(f"\n⚠️  Method {method} failed: {e}")
            continue
    
    # Final recommendations
    analyzer.recommend_prediction_targets()
    
    # Visualizations
    try:
        analyzer.visualize_regimes(save_path='vix_regime_analysis.png')
    except Exception as e:
        print(f"\n⚠️  Visualization failed: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review recommended prediction targets")
    print("2. Choose regime boundaries to use in VIXModel")
    print("3. Implement multi-class regime prediction")
    print("4. Build visualizations for dashboard")


if __name__ == "__main__":
    main()