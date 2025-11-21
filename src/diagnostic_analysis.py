#!/usr/bin/env python3
"""
Diagnostic Script - Investigate Remaining Performance Issues
Run this to analyze MAE degradation and Precision==Recall mystery
"""
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def analyze_mae_by_period():
    """Check if MAE is degrading over time"""
    print("\n" + "="*80)
    print("ANALYSIS #1: MAE by Time Period")
    print("="*80)
    
    db_path = Path("data_cache/predictions.db")
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            forecast_date,
            magnitude_forecast,
            actual_vix_change,
            magnitude_error,
            calendar_cohort
        FROM forecasts 
        WHERE actual_vix_change IS NOT NULL
        ORDER BY forecast_date
    """, conn, parse_dates=['forecast_date'])
    conn.close()
    
    df['year'] = df['forecast_date'].dt.year
    df['quarter'] = df['forecast_date'].dt.to_period('Q')
    
    print("\nMAE by Year:")
    yearly = df.groupby('year').agg({
        'magnitude_error': ['mean', 'std', 'count']
    })
    print(yearly)
    
    print("\nMAE by Quarter:")
    quarterly = df.groupby('quarter').agg({
        'magnitude_error': ['mean', 'std', 'count']
    })
    print(quarterly)
    
    # Check if MAE is trending up
    quarterly_mae = quarterly[('magnitude_error', 'mean')].values
    if len(quarterly_mae) > 4:
        recent_4q = quarterly_mae[-4:].mean()
        older_4q = quarterly_mae[-8:-4].mean() if len(quarterly_mae) > 8 else quarterly_mae[:-4].mean()
        
        print(f"\nRecent 4Q average MAE: {recent_4q:.2f}%")
        print(f"Previous 4Q average MAE: {older_4q:.2f}%")
        
        if recent_4q > older_4q * 1.1:
            print("⚠️ WARNING: MAE is degrading over time (>10% increase)")
        else:
            print("✅ MAE is stable or improving")
    
    return df

def analyze_prediction_distribution():
    """Check UP/DOWN prediction rates vs actual rates"""
    print("\n" + "="*80)
    print("ANALYSIS #2: Prediction Distribution (Precision==Recall Investigation)")
    print("="*80)
    
    db_path = Path("data_cache/predictions.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            prob_up,
            actual_direction,
            magnitude_forecast
        FROM forecasts 
        WHERE actual_direction IS NOT NULL
    """, conn)
    conn.close()
    
    df['predicted_direction'] = (df['prob_up'] > 0.5).astype(int)
    
    total = len(df)
    actual_ups = df['actual_direction'].sum()
    predicted_ups = df['predicted_direction'].sum()
    
    print(f"\nTotal forecasts: {total}")
    print(f"Actual UP events: {actual_ups} ({actual_ups/total:.1%})")
    print(f"Predicted UP: {predicted_ups} ({predicted_ups/total:.1%})")
    print(f"Difference: {abs(predicted_ups - actual_ups)} forecasts ({abs(predicted_ups - actual_ups)/total:.1%})")
    
    # Confusion matrix
    cm = confusion_matrix(df['actual_direction'], df['predicted_direction'])
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (DOWN):  {tn:4d}")
    print(f"  False Positives:        {fp:4d} (predicted UP, actually DOWN)")
    print(f"  False Negatives:        {fn:4d} (predicted DOWN, actually UP)")
    print(f"  True Positives (UP):    {tp:4d}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nCalculated Metrics:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  Difference: {abs(precision - recall):.1%}")
    
    if abs(fp - fn) < 5:
        print(f"\n✅ Precision==Recall is CORRECT (FP≈FN: {fp} vs {fn})")
        print("   This is a quirk of the data distribution, not a bug")
    else:
        print(f"\n⚠️ WARNING: FP and FN should be similar but aren't ({fp} vs {fn})")
    
    return df

def analyze_residuals_by_regime():
    """Check if errors are regime-dependent"""
    print("\n" + "="*80)
    print("ANALYSIS #3: Residual Analysis by Regime")
    print("="*80)
    
    db_path = Path("data_cache/predictions.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            current_vix,
            magnitude_forecast,
            actual_vix_change,
            magnitude_error,
            calendar_cohort
        FROM forecasts 
        WHERE actual_vix_change IS NOT NULL
    """, conn)
    conn.close()
    
    # Define VIX regimes
    def get_regime(vix):
        if vix < 16.77:
            return "Low Vol"
        elif vix < 24.40:
            return "Normal"
        elif vix < 39.67:
            return "Elevated"
        else:
            return "Crisis"
    
    df['regime'] = df['current_vix'].apply(get_regime)
    df['residual'] = df['magnitude_forecast'] - df['actual_vix_change']
    
    print("\nPerformance by VIX Regime:")
    regime_stats = df.groupby('regime').agg({
        'magnitude_error': ['mean', 'std'],
        'residual': 'mean',
        'current_vix': 'count'
    })
    regime_stats.columns = ['MAE', 'Std', 'Bias', 'Count']
    print(regime_stats)
    
    print("\nPerformance by Calendar Cohort:")
    cohort_stats = df.groupby('calendar_cohort').agg({
        'magnitude_error': ['mean', 'std'],
        'residual': 'mean',
        'current_vix': 'count'
    })
    cohort_stats.columns = ['MAE', 'Std', 'Bias', 'Count']
    print(cohort_stats)
    
    # Check for systematic bias
    overall_bias = df['residual'].mean()
    print(f"\nOverall Bias: {overall_bias:+.2f}%")
    
    if abs(overall_bias) > 2.0:
        print("⚠️ WARNING: Significant bias detected")
        if overall_bias > 0:
            print("   Model is OVERESTIMATING VIX changes")
        else:
            print("   Model is UNDERESTIMATING VIX changes")
    else:
        print("✅ Bias is acceptable")
    
    return df

def analyze_confidence_distribution():
    """Check if confidence scaling is appropriate"""
    print("\n" + "="*80)
    print("ANALYSIS #4: Confidence Distribution")
    print("="*80)
    
    db_path = Path("data_cache/predictions.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            prob_up,
            magnitude_forecast,
            actual_direction,
            magnitude_error
        FROM forecasts 
        WHERE actual_direction IS NOT NULL
    """, conn)
    conn.close()
    
    df['predicted_direction'] = (df['prob_up'] > 0.5).astype(int)
    df['confidence'] = df['prob_up'].apply(lambda x: x if x > 0.5 else 1-x)
    df['correct'] = (df['predicted_direction'] == df['actual_direction']).astype(int)
    
    # Bin by confidence
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    df['confidence_bin'] = pd.cut(df['confidence'], bins=bins, labels=labels)
    
    print("\nAccuracy by Confidence Level:")
    conf_stats = df.groupby('confidence_bin').agg({
        'correct': 'mean',
        'prob_up': 'count',
        'confidence': 'mean'
    })
    conf_stats.columns = ['Accuracy', 'Count', 'Avg_Confidence']
    print(conf_stats)
    
    # Check calibration
    print("\nCalibration Quality:")
    for bin_label in labels:
        if bin_label in df['confidence_bin'].values:
            bin_df = df[df['confidence_bin'] == bin_label]
            pred_conf = bin_df['confidence'].mean()
            actual_acc = bin_df['correct'].mean()
            error = abs(pred_conf - actual_acc)
            
            print(f"  {bin_label:10s}: Pred={pred_conf:.1%}, Actual={actual_acc:.1%}, Error={error:.1%}")
            
            if error > 0.15:
                print(f"               ⚠️ POOR calibration")
            elif error > 0.10:
                print(f"               ⚠️ Fair calibration")
            else:
                print(f"               ✅ Good calibration")
    
    # Check if we're capping too many predictions
    very_high_conf = (df['confidence'] > 0.95).sum()
    print(f"\nPredictions with >95% confidence: {very_high_conf} ({very_high_conf/len(df):.1%})")
    
    if very_high_conf > len(df) * 0.2:
        print("⚠️ WARNING: Too many predictions at maximum confidence")
        print("   Consider increasing scaling factor threshold from 20% to 30%")
    
    return df

def generate_recommendations():
    """Generate recommendations based on analysis"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # This will be filled in based on the analysis results
    print("\nBased on the analysis above:")
    print("\n1. If MAE is degrading over time:")
    print("   → Implement quarterly retraining")
    print("   → Include recent data (2023-2024) in training set")
    
    print("\n2. If Precision==Recall is confirmed correct:")
    print("   → No action needed, it's a data distribution quirk")
    print("   → Continue monitoring in case it changes")
    
    print("\n3. If residuals show regime-specific bias:")
    print("   → Consider regime-specific models")
    print("   → Or add regime interaction terms")
    
    print("\n4. If confidence scaling needs improvement:")
    print("   → Test scaling_factor = confidence_pct / 30.0")
    print("   → Re-validate calibration")
    
    print("\n5. Production monitoring setup:")
    print("   → Track rolling 30-day MAE")
    print("   → Alert if MAE > 15% for 5 consecutive days")
    print("   → Alert if calibration error > 20%")

def main():
    """Run all diagnostics"""
    print("="*80)
    print("VIX FORECASTING SYSTEM - DIAGNOSTIC REPORT")
    print("="*80)
    
    try:
        df1 = analyze_mae_by_period()
        df2 = analyze_prediction_distribution()
        df3 = analyze_residuals_by_regime()
        df4 = analyze_confidence_distribution()
        generate_recommendations()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the analysis above")
        print("2. Discuss findings in next session")
        print("3. Implement recommended fixes")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
