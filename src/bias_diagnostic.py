"""
BIAS DIAGNOSTIC - Find the source of the +17.59% systematic error

Analyzes forecast errors across:
- Time periods (2023 vs 2024 vs 2025)
- Calendar cohorts
- VIX regimes (low/high VIX environments)
- Feature quality scores

Output: Identifies when and where the bias occurs
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Setup
DB_PATH = Path("data_cache/predictions.db")
OUTPUT_DIR = Path("diagnostics/bias_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions():
    """Load all predictions with actuals"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        forecast_date,
        target_date,
        point_estimate,
        actual_outcome,
        abs_error,
        signed_error,
        calendar_cohort,
        feature_quality
    FROM predictions
    WHERE actual_outcome IS NOT NULL
    ORDER BY forecast_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse dates
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['target_date'] = pd.to_datetime(df['target_date'])
    
    # Add year column
    df['year'] = df['target_date'].dt.year
    
    # Calculate VIX level at forecast time (implied from actual + error)
    df['vix_at_forecast'] = df['actual_outcome'] / (1 + df['point_estimate']/100)
    
    # VIX regime
    df['vix_regime'] = pd.cut(df['vix_at_forecast'], 
                               bins=[0, 15, 20, 30, 100],
                               labels=['Low (<15)', 'Normal (15-20)', 'Elevated (20-30)', 'Crisis (>30)'])
    
    print(f"✅ Loaded {len(df)} predictions with actuals")
    print(f"   Date range: {df['forecast_date'].min().date()} to {df['forecast_date'].max().date()}")
    
    return df


def compute_bias_metrics(df):
    """Compute bias statistics"""
    
    print("\n" + "="*80)
    print("BIAS ANALYSIS")
    print("="*80)
    
    # Overall
    print(f"\n📊 Overall (n={len(df)}):")
    print(f"   Mean Error: {df['signed_error'].mean():.2f}%")
    print(f"   Median Error: {df['signed_error'].median():.2f}%")
    print(f"   MAE: {df['abs_error'].mean():.2f}%")
    print(f"   Std Dev: {df['signed_error'].std():.2f}%")
    
    # By year
    print(f"\n📅 By Year:")
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        print(f"   {year} (n={len(year_data):3d}): "
              f"Bias={year_data['signed_error'].mean():+6.2f}%, "
              f"MAE={year_data['abs_error'].mean():5.2f}%")
    
    # By cohort
    print(f"\n📆 By Calendar Cohort:")
    for cohort in df['calendar_cohort'].value_counts().index[:8]:
        cohort_data = df[df['calendar_cohort'] == cohort]
        print(f"   {cohort:25s} (n={len(cohort_data):3d}): "
              f"Bias={cohort_data['signed_error'].mean():+6.2f}%, "
              f"MAE={cohort_data['abs_error'].mean():5.2f}%")
    
    # By VIX regime
    print(f"\n🎯 By VIX Regime:")
    for regime in ['Low (<15)', 'Normal (15-20)', 'Elevated (20-30)', 'Crisis (>30)']:
        regime_data = df[df['vix_regime'] == regime]
        if len(regime_data) > 0:
            print(f"   {regime:20s} (n={len(regime_data):3d}): "
                  f"Bias={regime_data['signed_error'].mean():+6.2f}%, "
                  f"MAE={regime_data['abs_error'].mean():5.2f}%")


def plot_error_timeseries(df):
    """Plot forecast error over time"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Signed error over time
    ax = axes[0]
    ax.scatter(df['forecast_date'], df['signed_error'], 
               alpha=0.3, s=20, c='steelblue')
    
    # Rolling mean
    rolling = df.set_index('forecast_date')['signed_error'].rolling('30D').mean()
    ax.plot(rolling.index, rolling.values, 'r-', linewidth=2, label='30-day MA')
    
    # Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Year boundaries
    for year in [2024, 2025]:
        ax.axvline(pd.Timestamp(f'{year}-01-01'), color='gray', linestyle=':', alpha=0.5)
        ax.text(pd.Timestamp(f'{year}-01-01'), ax.get_ylim()[1]*0.9, 
                str(year), fontsize=10, alpha=0.7)
    
    ax.set_ylabel('Forecast Error (%)', fontsize=12)
    ax.set_title('Forecast Error Over Time (Positive = Over-prediction)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Absolute error over time
    ax = axes[1]
    ax.scatter(df['forecast_date'], df['abs_error'], 
               alpha=0.3, s=20, c='orange')
    
    rolling_mae = df.set_index('forecast_date')['abs_error'].rolling('30D').mean()
    ax.plot(rolling_mae.index, rolling_mae.values, 'darkred', linewidth=2, label='30-day MA')
    
    ax.axhline(df['abs_error'].mean(), color='red', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'Overall MAE: {df["abs_error"].mean():.1f}%')
    
    for year in [2024, 2025]:
        ax.axvline(pd.Timestamp(f'{year}-01-01'), color='gray', linestyle=':', alpha=0.5)
    
    ax.set_ylabel('Absolute Error (%)', fontsize=12)
    ax.set_title('Forecast Accuracy Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error distribution by year
    ax = axes[2]
    
    years = sorted(df['year'].unique())
    positions = []
    for i, year in enumerate(years):
        year_errors = df[df['year'] == year]['signed_error']
        parts = ax.violinplot([year_errors], positions=[i], 
                               widths=0.7, showmeans=True, showmedians=True)
        positions.append(i)
        
        # Color code
        mean_error = year_errors.mean()
        color = 'red' if mean_error > 10 else 'orange' if mean_error > 5 else 'green'
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(years)
    ax.set_ylabel('Forecast Error (%)', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_title('Error Distribution by Year', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_timeseries.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR / 'error_timeseries.png'}")
    plt.close()


def plot_bias_by_cohort(df):
    """Plot bias by calendar cohort"""
    
    cohort_stats = df.groupby('calendar_cohort').agg({
        'signed_error': ['mean', 'count'],
        'abs_error': 'mean'
    }).reset_index()
    
    cohort_stats.columns = ['cohort', 'bias', 'count', 'mae']
    cohort_stats = cohort_stats[cohort_stats['count'] >= 20]  # Only cohorts with 20+ samples
    cohort_stats = cohort_stats.sort_values('bias')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' 
              for x in cohort_stats['bias']]
    
    bars = ax.barh(cohort_stats['cohort'], cohort_stats['bias'], color=colors, alpha=0.7)
    
    # Add count labels
    for i, (bias, count) in enumerate(zip(cohort_stats['bias'], cohort_stats['count'])):
        ax.text(bias + 1, i, f'n={int(count)}', 
                va='center', fontsize=9, alpha=0.7)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Forecast Error (%) [Positive = Over-prediction]', fontsize=12)
    ax.set_title('Forecast Bias by Calendar Cohort', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bias_by_cohort.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'bias_by_cohort.png'}")
    plt.close()


def plot_bias_by_vix_regime(df):
    """Plot bias by VIX regime"""
    
    regime_stats = df.groupby('vix_regime').agg({
        'signed_error': ['mean', 'std', 'count'],
        'abs_error': 'mean'
    }).reset_index()
    
    regime_stats.columns = ['regime', 'bias', 'std', 'count', 'mae']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bias by regime
    ax = axes[0]
    colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' 
              for x in regime_stats['bias']]
    
    bars = ax.bar(range(len(regime_stats)), regime_stats['bias'], 
                   color=colors, alpha=0.7)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(regime_stats)))
    ax.set_xticklabels(regime_stats['regime'], rotation=45, ha='right')
    ax.set_ylabel('Mean Forecast Error (%)', fontsize=12)
    ax.set_title('Forecast Bias by VIX Regime', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add counts
    for i, (bias, count) in enumerate(zip(regime_stats['bias'], regime_stats['count'])):
        ax.text(i, bias + 2, f'n={int(count)}', 
                ha='center', fontsize=9, alpha=0.7)
    
    # Plot 2: MAE by regime
    ax = axes[1]
    ax.bar(range(len(regime_stats)), regime_stats['mae'], 
           color='steelblue', alpha=0.7)
    
    ax.set_xticks(range(len(regime_stats)))
    ax.set_xticklabels(regime_stats['regime'], rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error (%)', fontsize=12)
    ax.set_title('Forecast Accuracy by VIX Regime', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bias_by_vix_regime.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'bias_by_vix_regime.png'}")
    plt.close()


def detect_regime_shift(df):
    """Statistical test for regime shift"""
    
    print("\n" + "="*80)
    print("REGIME SHIFT DETECTION")
    print("="*80)
    
    # Split into periods
    df_2023 = df[df['year'] == 2023]
    df_2024 = df[df['year'] == 2024]
    df_2025 = df[df['year'] == 2025]
    
    from scipy import stats
    
    # Test 2023 vs 2024
    if len(df_2023) > 0 and len(df_2024) > 0:
        t_stat, p_val = stats.ttest_ind(df_2023['signed_error'], 
                                         df_2024['signed_error'])
        print(f"\n2023 vs 2024:")
        print(f"   t-statistic: {t_stat:.3f}")
        print(f"   p-value: {p_val:.4f}")
        print(f"   {'⚠️  SIGNIFICANT SHIFT' if p_val < 0.05 else '✅ No significant shift'}")
    
    # Test 2024 vs 2025
    if len(df_2024) > 0 and len(df_2025) > 0:
        t_stat, p_val = stats.ttest_ind(df_2024['signed_error'], 
                                         df_2025['signed_error'])
        print(f"\n2024 vs 2025:")
        print(f"   t-statistic: {t_stat:.3f}")
        print(f"   p-value: {p_val:.4f}")
        print(f"   {'⚠️  SIGNIFICANT SHIFT' if p_val < 0.05 else '✅ No significant shift'}")
    
    # Test 2023+2024 vs 2025
    if len(df_2023) + len(df_2024) > 0 and len(df_2025) > 0:
        df_train = pd.concat([df_2023, df_2024])
        t_stat, p_val = stats.ttest_ind(df_train['signed_error'], 
                                         df_2025['signed_error'])
        print(f"\n2023+2024 (training) vs 2025 (validation):")
        print(f"   Training bias: {df_train['signed_error'].mean():+.2f}%")
        print(f"   Validation bias: {df_2025['signed_error'].mean():+.2f}%")
        print(f"   t-statistic: {t_stat:.3f}")
        print(f"   p-value: {p_val:.4f}")
        print(f"   {'⚠️  SIGNIFICANT REGIME SHIFT' if p_val < 0.05 else '✅ No significant shift'}")


def generate_summary_report(df):
    """Generate text summary"""
    
    report = []
    report.append("="*80)
    report.append("BIAS DIAGNOSTIC SUMMARY")
    report.append("="*80)
    
    # Overall
    overall_bias = df['signed_error'].mean()
    report.append(f"\n📊 Overall Performance:")
    report.append(f"   Systematic Bias: {overall_bias:+.2f}%")
    report.append(f"   MAE: {df['abs_error'].mean():.2f}%")
    report.append(f"   Samples: {len(df)}")
    
    # Diagnosis
    report.append(f"\n🔍 Diagnosis:")
    
    # Check by year
    year_biases = df.groupby('year')['signed_error'].mean()
    max_year_bias = year_biases.max()
    min_year_bias = year_biases.min()
    
    if max_year_bias - min_year_bias > 10:
        worst_year = year_biases.idxmax()
        report.append(f"   ⚠️  REGIME SHIFT DETECTED")
        report.append(f"   → Bias concentrated in {worst_year}: {year_biases[worst_year]:+.2f}%")
        report.append(f"   → Action: Retrain calibrator monthly, add regime detection")
    else:
        report.append(f"   ✅ Bias is consistent across years")
        report.append(f"   → Action: Check for systematic feature leakage")
    
    # Check by cohort
    cohort_biases = df.groupby('calendar_cohort')['signed_error'].mean()
    if cohort_biases.std() > 8:
        worst_cohort = cohort_biases.idxmax()
        report.append(f"   ⚠️  COHORT-SPECIFIC BIAS")
        report.append(f"   → Worst cohort: {worst_cohort} ({cohort_biases[worst_cohort]:+.2f}%)")
        report.append(f"   → Action: Retrain cohort-specific calibrators")
    
    # Check by VIX regime
    regime_biases = df.groupby('vix_regime')['signed_error'].mean()
    if len(regime_biases) > 1 and regime_biases.std() > 8:
        report.append(f"   ⚠️  VIX-REGIME-SPECIFIC BIAS")
        report.append(f"   → Bias varies by market regime")
        report.append(f"   → Action: Add VIX regime to calibrator features")
    
    report.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(OUTPUT_DIR / 'bias_summary.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Saved: {OUTPUT_DIR / 'bias_summary.txt'}")


def main():
    print("="*80)
    print("BIAS DIAGNOSTIC TOOL")
    print("="*80)
    print("\nAnalyzing forecast bias patterns...\n")
    
    # Load data
    df = load_predictions()
    
    # Compute metrics
    compute_bias_metrics(df)
    
    # Generate plots
    print("\n📊 Generating diagnostic plots...")
    plot_error_timeseries(df)
    plot_bias_by_cohort(df)
    plot_bias_by_vix_regime(df)
    
    # Statistical tests
    detect_regime_shift(df)
    
    # Summary
    generate_summary_report(df)
    
    print("\n" + "="*80)
    print("✅ BIAS DIAGNOSTIC COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("   - error_timeseries.png")
    print("   - bias_by_cohort.png")
    print("   - bias_by_vix_regime.png")
    print("   - bias_summary.txt")


if __name__ == "__main__":
    main()
