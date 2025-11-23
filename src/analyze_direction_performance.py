#!/usr/bin/env python3
"""
Diagnostic analyzer for direction classifier performance
Shows what data we're generating and where improvements are needed
"""
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_predictions_db():
    """Analyze the predictions database to understand model performance"""
    db_path = "data_cache/predictions.db"
    
    if not Path(db_path).exists():
        print("❌ No database found at", db_path)
        return
    
    conn = sqlite3.connect(db_path)
    
    # Load all predictions with actuals
    df = pd.read_sql_query("""
        SELECT 
            forecast_date,
            observation_date,
            calendar_cohort,
            current_vix,
            direction_prediction,
            direction_probability,
            direction_correct,
            magnitude_forecast,
            actual_vix_change,
            actual_direction,
            correction_type
        FROM forecasts
        WHERE actual_vix_change IS NOT NULL
        ORDER BY forecast_date
    """, conn)
    
    conn.close()
    
    if len(df) == 0:
        print("❌ No predictions with actuals found")
        return
    
    print("="*80)
    print("DIRECTION CLASSIFIER DIAGNOSTICS")
    print("="*80)
    print(f"\nTotal predictions with actuals: {len(df)}")
    print(f"Date range: {df['forecast_date'].min()} → {df['forecast_date'].max()}")
    
    # Overall accuracy
    correct = df['direction_correct'].sum()
    total = len(df)
    accuracy = 100 * correct / total if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    
    # Confidence distribution
    print(f"\nConfidence Distribution:")
    print(f"  Mean: {df['direction_probability'].mean():.1%}")
    print(f"  Median: {df['direction_probability'].median():.1%}")
    print(f"  Std: {df['direction_probability'].std():.1%}")
    print(f"  Min: {df['direction_probability'].min():.1%}")
    print(f"  Max: {df['direction_probability'].max():.1%}")
    
    # Confidence bins
    bins = [0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
    labels = ['<55%', '55-60%', '60-65%', '65-70%', '70-75%', '>75%']
    df['conf_bin'] = pd.cut(df['direction_probability'], bins=bins, labels=labels)
    
    print(f"\nAccuracy by Confidence Level:")
    for label in labels:
        subset = df[df['conf_bin'] == label]
        if len(subset) > 0:
            acc = 100 * subset['direction_correct'].sum() / len(subset)
            print(f"  {label}: {subset['direction_correct'].sum()}/{len(subset)} = {acc:.1f}%")
    
    # Performance by cohort
    print(f"\n{'='*80}")
    print(f"PERFORMANCE BY COHORT")
    print(f"{'='*80}")
    for cohort in df['calendar_cohort'].unique():
        subset = df[df['calendar_cohort'] == cohort]
        if len(subset) > 0:
            acc = 100 * subset['direction_correct'].sum() / len(subset)
            avg_conf = subset['direction_probability'].mean()
            print(f"{cohort:20s}: {subset['direction_correct'].sum():3d}/{len(subset):3d} = {acc:5.1f}% | Avg Conf: {avg_conf:.1%}")
    
    # Performance by VIX regime
    df['vix_regime'] = pd.cut(df['current_vix'], 
                               bins=[0, 15.57, 23.36, 31.16, 100],
                               labels=['Low Vol', 'Normal', 'Elevated', 'Crisis'])
    
    print(f"\n{'='*80}")
    print(f"PERFORMANCE BY VIX REGIME")
    print(f"{'='*80}")
    for regime in ['Low Vol', 'Normal', 'Elevated', 'Crisis']:
        subset = df[df['vix_regime'] == regime]
        if len(subset) > 0:
            acc = 100 * subset['direction_correct'].sum() / len(subset)
            avg_conf = subset['direction_probability'].mean()
            print(f"{regime:20s}: {subset['direction_correct'].sum():3d}/{len(subset):3d} = {acc:5.1f}% | Avg Conf: {avg_conf:.1%}")
    
    # Direction prediction distribution
    print(f"\n{'='*80}")
    print(f"PREDICTION DISTRIBUTION")
    print(f"{'='*80}")
    up_preds = (df['direction_prediction'] == 'UP').sum()
    down_preds = (df['direction_prediction'] == 'DOWN').sum()
    print(f"UP predictions:   {up_preds:3d} ({100*up_preds/total:.1f}%)")
    print(f"DOWN predictions: {down_preds:3d} ({100*down_preds/total:.1f}%)")
    
    up_correct = df[df['direction_prediction'] == 'UP']['direction_correct'].sum()
    down_correct = df[df['direction_prediction'] == 'DOWN']['direction_correct'].sum()
    
    if up_preds > 0:
        print(f"\nUP accuracy:   {up_correct}/{up_preds} = {100*up_correct/up_preds:.1f}%")
    if down_preds > 0:
        print(f"DOWN accuracy: {down_correct}/{down_preds} = {100*down_correct/down_preds:.1f}%")
    
    # Actual distribution
    actual_up = (df['actual_direction'] == 1).sum()
    actual_down = (df['actual_direction'] == 0).sum()
    print(f"\nActual UP:   {actual_up:3d} ({100*actual_up/total:.1f}%)")
    print(f"Actual DOWN: {actual_down:3d} ({100*actual_down/total:.1f}%)")
    
    # Time series performance (monthly)
    df['month'] = pd.to_datetime(df['forecast_date']).dt.to_period('M')
    print(f"\n{'='*80}")
    print(f"MONTHLY PERFORMANCE TREND")
    print(f"{'='*80}")
    for month in df['month'].unique()[-12:]:  # Last 12 months
        subset = df[df['month'] == month]
        if len(subset) > 0:
            acc = 100 * subset['direction_correct'].sum() / len(subset)
            avg_conf = subset['direction_probability'].mean()
            print(f"{str(month):10s}: {subset['direction_correct'].sum():2d}/{len(subset):2d} = {acc:5.1f}% | Conf: {avg_conf:.1%}")
    
    # Magnitude vs direction alignment
    print(f"\n{'='*80}")
    print(f"MAGNITUDE vs DIRECTION ALIGNMENT")
    print(f"{'='*80}")
    mag_up = (df['magnitude_forecast'] > 0).sum()
    mag_down = (df['magnitude_forecast'] < 0).sum()
    print(f"Magnitude predicts UP:   {mag_up:3d} ({100*mag_up/total:.1f}%)")
    print(f"Magnitude predicts DOWN: {mag_down:3d} ({100*mag_down/total:.1f}%)")
    
    # Agreement between magnitude and direction
    df['mag_direction'] = df['magnitude_forecast'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
    agreement = (df['mag_direction'] == df['direction_prediction']).sum()
    print(f"\nMagnitude-Direction agreement: {agreement}/{total} ({100*agreement/total:.1f}%)")
    
    # When they agree vs disagree
    agree_df = df[df['mag_direction'] == df['direction_prediction']]
    disagree_df = df[df['mag_direction'] != df['direction_prediction']]
    
    if len(agree_df) > 0:
        agree_acc = 100 * agree_df['direction_correct'].sum() / len(agree_df)
        print(f"Accuracy when models agree:    {agree_df['direction_correct'].sum()}/{len(agree_df)} = {agree_acc:.1f}%")
    
    if len(disagree_df) > 0:
        disagree_acc = 100 * disagree_df['direction_correct'].sum() / len(disagree_df)
        print(f"Accuracy when models disagree: {disagree_df['direction_correct'].sum()}/{len(disagree_df)} = {disagree_acc:.1f}%")
    
    return df


def analyze_feature_importance():
    """Compare feature importance between magnitude and direction models"""
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE COMPARISON")
    print(f"{'='*80}")
    
    mag_file = Path("data_cache/feature_importance_magnitude.json")
    dir_file = Path("data_cache/feature_importance_direction.json")
    
    if not mag_file.exists() or not dir_file.exists():
        print("❌ Feature importance files not found")
        return
    
    with open(mag_file) as f:
        mag_imp = json.load(f)
    with open(dir_file) as f:
        dir_imp = json.load(f)
    
    # Top features for each
    print("\nTop 10 Magnitude Features:")
    mag_sorted = sorted(mag_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feat, imp) in enumerate(mag_sorted, 1):
        print(f"{i:2d}. {feat:50s} {imp:.6f}")
    
    print("\nTop 10 Direction Features:")
    dir_sorted = sorted(dir_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feat, imp) in enumerate(dir_sorted, 1):
        print(f"{i:2d}. {feat:50s} {imp:.6f}")
    
    # Features unique to each
    mag_feats = set(mag_imp.keys())
    dir_feats = set(dir_imp.keys())
    common = mag_feats & dir_feats
    mag_only = mag_feats - dir_feats
    dir_only = dir_feats - mag_feats
    
    print(f"\nFeature Set Overlap:")
    print(f"  Common features: {len(common)}")
    print(f"  Magnitude only:  {len(mag_only)}")
    print(f"  Direction only:  {len(dir_only)}")


def analyze_hyperparameters():
    """Show current hyperparameters for both models"""
    print(f"\n{'='*80}")
    print(f"CURRENT HYPERPARAMETERS")
    print(f"{'='*80}")
    
    config_file = Path("config.py")
    if not config_file.exists():
        print("❌ config.py not found")
        return
    
    # Read the config file
    with open(config_file) as f:
        content = f.read()
    
    # Extract magnitude_params and direction_params
    import re
    
    mag_match = re.search(r'"magnitude_params":\{([^}]+)\}', content, re.DOTALL)
    dir_match = re.search(r'"direction_params":\{([^}]+)\}', content, re.DOTALL)
    
    if mag_match:
        print("\nMagnitude Model (XGBRegressor):")
        for line in mag_match.group(1).strip().split('\n'):
            if ':' in line:
                print(f"  {line.strip()}")
    
    if dir_match:
        print("\nDirection Model (XGBClassifier):")
        for line in dir_match.group(1).strip().split('\n'):
            if ':' in line:
                print(f"  {line.strip()}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VIX FORECASTING SYSTEM - DIAGNOSTIC ANALYSIS")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run analyses
    df = analyze_predictions_db()
    analyze_feature_importance()
    analyze_hyperparameters()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)