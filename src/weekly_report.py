#!/usr/bin/env python3
import pandas as pd
from core.prediction_database import PredictionDatabase

db = PredictionDatabase()
df = db.get_predictions(with_actuals=True)

# Last 30 days performance
recent = df[df['forecast_date'] >= pd.Timestamp.now() - pd.Timedelta(days=30)]

print("\n" + "="*80)
print("LAST 30 DAYS PERFORMANCE")
print("="*80)

# Overall metrics
print(f"\nOverall:")
print(f"  Predictions: {len(recent)}")
print(f"  Direction Accuracy: {recent['direction_correct'].mean():.1%}")
print(f"  MAE: {recent['magnitude_error'].mean():.2f}%")
print(f"  Bias: {(recent['magnitude_forecast'] - recent['actual_vix_change']).mean():+.2f}%")

# By regime
print(f"\nBy Regime:")
for cohort in recent['calendar_cohort'].unique():
    subset = recent[recent['calendar_cohort'] == cohort]
    print(f"  {cohort}: {len(subset)} predictions, {subset['direction_correct'].mean():.1%} accuracy, {subset['magnitude_error'].mean():.2f}% MAE")

# Best/worst predictions
print(f"\nBest 3 Predictions (lowest error):")
best = recent.nsmallest(3, 'magnitude_error')[['forecast_date', 'magnitude_forecast', 'actual_vix_change', 'magnitude_error']]
for idx, row in best.iterrows():
    print(f"  {row['forecast_date'].date()}: Forecast {row['magnitude_forecast']:+.2f}%, Actual {row['actual_vix_change']:+.2f}%, Error {row['magnitude_error']:.2f}%")

print(f"\nWorst 3 Predictions (highest error):")
worst = recent.nlargest(3, 'magnitude_error')[['forecast_date', 'magnitude_forecast', 'actual_vix_change', 'magnitude_error']]
for idx, row in worst.iterrows():
    print(f"  {row['forecast_date'].date()}: Forecast {row['magnitude_forecast']:+.2f}%, Actual {row['actual_vix_change']:+.2f}%, Error {row['magnitude_error']:.2f}%")

print("="*80 + "\n")