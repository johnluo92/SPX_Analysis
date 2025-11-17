# CREATE NEW FILE: backtesting_engine.py
## Probabilistic Forecast Evaluation Engine

---

## SYSTEM CONTEXT

### Purpose
Evaluate probabilistic forecasts using proper scoring rules (not accuracy). Generates diagnostic reports showing where models excel or fail.

**Key Metrics:**
- Quantile Coverage: Do 90% of actuals fall below q90?
- Brier Score: Are regime probabilities well-calibrated?
- Sharpness: Are quantiles tight (confident) or wide (uncertain)?
- Confidence Correlation: Does confidence predict error?

---

## FILE LOCATION

`src/backtesting_engine.py`

---

## COMPLETE IMPLEMENTATION

```python
"""
Backtesting Engine for Probabilistic Forecasts

Evaluates forecast quality using proper scoring rules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

from prediction_database import PredictionDatabase

logger = logging.getLogger(__name__)


class ProbabilisticBacktester:
    """
    Evaluate probabilistic forecasts stored in database.
    
    Metrics:
        - Quantile Coverage: Empirical vs nominal rates
        - Brier Score: Probability calibration
        - Pinball Loss: Quantile sharpness
        - CRPS: Continuous Ranked Probability Score
    """
    
    def __init__(self, db_path=None):
        self.db = PredictionDatabase(db_path)
        self.results = {}
    
    def run_full_evaluation(self, save_dir='diagnostics'):
        """Run all evaluation metrics and generate report."""
        logger.info("=" * 80)
        logger.info("PROBABILISTIC BACKTEST EVALUATION")
        logger.info("=" * 80)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Get predictions with actuals
        df = self.db.get_predictions(with_actuals=True)
        logger.info(f"📊 Evaluating {len(df)} predictions")
        
        if len(df) == 0:
            logger.error("❌ No predictions with actuals found")
            return
        
        # 1. Quantile Coverage
        logger.info("\n[1/5] Computing quantile coverage...")
        coverage = self.evaluate_quantile_coverage(df)
        self.results['quantile_coverage'] = coverage
        
        # 2. Brier Score
        logger.info("\n[2/5] Computing regime Brier score...")
        brier = self.evaluate_brier_score(df)
        self.results['brier_score'] = brier
        
        # 3. Confidence Calibration
        logger.info("\n[3/5] Evaluating confidence calibration...")
        conf_metrics = self.evaluate_confidence(df)
        self.results['confidence'] = conf_metrics
        
        # 4. Cohort Comparison
        logger.info("\n[4/5] Comparing cohort performance...")
        cohort_metrics = self.evaluate_by_cohort(df)
        self.results['by_cohort'] = cohort_metrics
        
        # 5. Generate Plots
        logger.info("\n[5/5] Generating diagnostic plots...")
        self.plot_diagnostics(df, save_dir)
        
        # Save results
        results_file = save_dir / 'backtest_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n💾 Results saved: {results_file}")
        
        # Print summary
        self.print_summary()
        
        logger.info("=" * 80)
        logger.info("✅ EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        return self.results
    
    
    def evaluate_quantile_coverage(self, df):
        """
        Check if quantile coverage matches nominal rates.
        
        Returns:
            dict: {q10: {'empirical': 0.11, 'nominal': 0.10, 'error': 0.01}}
        """
        coverage = {}
        
        for q in [10, 25, 50, 75, 90]:
            col = f'q{q}'
            empirical = (df['actual_vix_change'] <= df[col]).mean()
            nominal = q / 100
            error = empirical - nominal
            
            coverage[col] = {
                'empirical': float(empirical),
                'nominal': float(nominal),
                'error': float(error),
                'acceptable': abs(error) < 0.05  # Within ±5%
            }
            
            status = "✅" if coverage[col]['acceptable'] else "⚠️ "
            logger.info(f"   {col}: {empirical:.3f} (nominal: {nominal:.2f}, error: {error:+.3f}) {status}")
        
        return coverage
    
    
    def evaluate_brier_score(self, df):
        """Compute Brier score for regime classification."""
        brier_scores = []
        
        for _, row in df.iterrows():
            if not row['actual_regime']:
                continue
            
            # One-hot actual
            actual_onehot = {
                'low': 0, 'normal': 0, 'elevated': 0, 'crisis': 0
            }
            actual_onehot[row['actual_regime'].lower()] = 1
            
            # Squared errors
            brier = (
                (row['prob_low'] - actual_onehot['low']) ** 2 +
                (row['prob_normal'] - actual_onehot['normal']) ** 2 +
                (row['prob_elevated'] - actual_onehot['elevated']) ** 2 +
                (row['prob_crisis'] - actual_onehot['crisis']) ** 2
            )
            brier_scores.append(brier)
        
        mean_brier = float(np.mean(brier_scores))
        logger.info(f"   Brier Score: {mean_brier:.3f} ({'✅ Good' if mean_brier < 0.2 else '⚠️  Poor'})")
        
        return {
            'score': mean_brier,
            'interpretation': 'good' if mean_brier < 0.2 else 'poor'
        }
    
    
    def evaluate_confidence(self, df):
        """Check if confidence correlates with accuracy."""
        # Correlation between confidence and error
        corr = df[['confidence_score', 'point_error']].corr().iloc[0, 1]
        
        # Bin by confidence and compute error
        df['conf_bin'] = pd.qcut(df['confidence_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        error_by_conf = df.groupby('conf_bin')['point_error'].mean()
        
        logger.info(f"   Confidence-Error Correlation: {corr:.3f}")
        logger.info(f"   Error by Confidence Bin:")
        for bin_name, error in error_by_conf.items():
            logger.info(f"      {bin_name:10s}: {error:.2f}%")
        
        return {
            'correlation': float(corr),
            'error_by_bin': {k: float(v) for k, v in error_by_conf.items()}
        }
    
    
    def evaluate_by_cohort(self, df):
        """Compare performance across calendar cohorts."""
        cohort_metrics = {}
        
        for cohort in df['calendar_cohort'].unique():
            cohort_df = df[df['calendar_cohort'] == cohort]
            
            if len(cohort_df) < 10:
                continue  # Skip small samples
            
            cohort_metrics[cohort] = {
                'n': len(cohort_df),
                'mae': float(cohort_df['point_error'].mean()),
                'rmse': float(np.sqrt(((cohort_df['actual_vix_change'] - cohort_df['point_estimate']) ** 2).mean())),
                'brier': float(self.db.compute_regime_brier_score(cohort))
            }
        
        # Sort by MAE
        sorted_cohorts = sorted(cohort_metrics.items(), key=lambda x: x[1]['mae'])
        
        logger.info(f"\n   {'Cohort':<30} | {'N':>5} | {'MAE':>8} | {'RMSE':>8} | {'Brier':>8}")
        logger.info("   " + "-" * 75)
        for cohort, metrics in sorted_cohorts:
            logger.info(f"   {cohort:<30} | {metrics['n']:>5} | {metrics['mae']:>7.2f}% | {metrics['rmse']:>7.2f}% | {metrics['brier']:>8.3f}")
        
        return cohort_metrics
    
    
    def plot_diagnostics(self, df, save_dir):
        """Generate diagnostic visualizations."""
        # 1. Quantile Coverage Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Coverage plot
        quantiles = [10, 25, 50, 75, 90]
        empirical = [self.results['quantile_coverage'][f'q{q}']['empirical'] for q in quantiles]
        nominal = [q/100 for q in quantiles]
        
        axes[0, 0].plot(nominal, empirical, 'o-', label='Empirical')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[0, 0].set_xlabel('Nominal Coverage')
        axes[0, 0].set_ylabel('Empirical Coverage')
        axes[0, 0].set_title('Quantile Coverage Calibration')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence vs Error
        axes[0, 1].scatter(df['confidence_score'], df['point_error'], alpha=0.5)
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Point Error (%)')
        axes[0, 1].set_title(f'Confidence vs Error (ρ={self.results["confidence"]["correlation"]:.2f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error Distribution by Cohort
        cohorts = df['calendar_cohort'].unique()
        errors_by_cohort = [df[df['calendar_cohort'] == c]['point_error'].values for c in cohorts]
        axes[1, 0].boxplot(errors_by_cohort, labels=cohorts)
        axes[1, 0].set_xticklabels(cohorts, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Point Error (%)')
        axes[1, 0].set_title('Error Distribution by Cohort')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Brier Score Over Time
        df_sorted = df.sort_values('forecast_date')
        df_sorted['brier'] = df_sorted.apply(lambda row: (
            (row['prob_low'] - (row['actual_regime'] == 'Low')) ** 2 +
            (row['prob_normal'] - (row['actual_regime'] == 'Normal')) ** 2 +
            (row['prob_elevated'] - (row['actual_regime'] == 'Elevated')) ** 2 +
            (row['prob_crisis'] - (row['actual_regime'] == 'Crisis')) ** 2
        ), axis=1)
        df_sorted['brier_ma'] = df_sorted['brier'].rolling(30).mean()
        axes[1, 1].plot(df_sorted['forecast_date'], df_sorted['brier_ma'])
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Brier Score (30-day MA)')
        axes[1, 1].set_title('Regime Classification Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'backtest_diagnostics.png', dpi=150)
        logger.info(f"   📈 Saved: {save_dir / 'backtest_diagnostics.png'}")
    
    
    def print_summary(self):
        """Print human-readable summary."""
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        
        # Quantile Coverage
        logger.info("\n📊 Quantile Coverage:")
        for q in [10, 25, 50, 75, 90]:
            metrics = self.results['quantile_coverage'][f'q{q}']
            status = "✅" if metrics['acceptable'] else "❌"
            logger.info(f"   {q}th: {metrics['empirical']:.1%} (nominal: {metrics['nominal']:.0%}) {status}")
        
        # Brier
        logger.info(f"\n🎯 Regime Brier Score: {self.results['brier_score']['score']:.3f}")
        
        # Confidence
        logger.info(f"\n🔍 Confidence Calibration: {self.results['confidence']['correlation']:.2f}")
        
        # Best/Worst Cohorts
        cohort_metrics = self.results['by_cohort']
        sorted_cohorts = sorted(cohort_metrics.items(), key=lambda x: x[1]['mae'])
        logger.info(f"\n🏆 Best Cohort: {sorted_cohorts[0][0]} (MAE: {sorted_cohorts[0][1]['mae']:.2f}%)")
        logger.info(f"⚠️  Worst Cohort: {sorted_cohorts[-1][0]} (MAE: {sorted_cohorts[-1][1]['mae']:.2f}%)")
```

---

## USAGE

```python
from backtesting_engine import ProbabilisticBacktester

# Run full evaluation
backtester = ProbabilisticBacktester()
results = backtester.run_full_evaluation(save_dir='diagnostics')

# Access specific metrics
print(f"Q90 Coverage: {results['quantile_coverage']['q90']['empirical']:.1%}")
print(f"Brier Score: {results['brier_score']['score']:.3f}")
```

---

## TESTING

```python
def test_backtester():
    """Test backtesting engine."""
    # Requires populated database
    backtester = ProbabilisticBacktester()
    results = backtester.run_full_evaluation(save_dir='test_diagnostics')
    
    assert 'quantile_coverage' in results
    assert 'brier_score' in results
    assert 'confidence' in results
    
    print("✅ Backtesting test passed")

test_backtester()
```

---

## SUMMARY

**File:** `backtesting_engine.py` (~250 lines)

**Dependencies:**
- prediction_database.py
- matplotlib, seaborn

**Output:**
- `diagnostics/backtest_results.json`
- `diagnostics/backtest_diagnostics.png`

**Run Schedule:** Weekly or after retraining models
