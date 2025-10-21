"""
Sector Rotation v3.5 - Simplified & Consistent
Run this file to execute the full pipeline.
"""

from datetime import datetime, timedelta
import pandas as pd

from config import LOOKBACK_YEARS
from data import DataFetcher
from features import FeatureEngine
from model import SectorModel


def main():
    """Execute the full sector rotation pipeline."""
    
    print("="*70)
    print("SECTOR ROTATION MODEL v3.5 - SIMPLIFIED")
    print("="*70)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\nData period: {start_str} to {end_str} ({LOOKBACK_YEARS} years)")
    
    # ========================================================================
    # STEP 1: FETCH DATA
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: FETCH DATA")
    print("="*70)
    
    fetcher = DataFetcher()
    sectors = fetcher.fetch_sectors(start_str, end_str)
    macro = fetcher.fetch_macro(start_str, end_str)
    vix = fetcher.fetch_vix(start_str, end_str)
    
    sectors, macro, vix = fetcher.align(sectors, macro, vix)
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    engine = FeatureEngine()
    features = engine.build(sectors, macro, vix)
    features_scaled = engine.scale(features)
    
    print(f"Built {len(features.columns)} features from {len(features)} samples")
    
    # ========================================================================
    # STEP 3: CREATE TARGETS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: CREATE TARGETS")
    print("="*70)
    
    model = SectorModel()
    targets = model.create_targets(sectors)
    
    print(f"Created {len(targets.columns)} targets from {len(targets)} samples")
    
    # ========================================================================
    # STEP 4: TRAIN MODELS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: TRAIN MODELS")
    print("="*70)
    
    model.train(features_scaled, targets, use_feature_selection=True)
    
    # ========================================================================
    # STEP 5: VALIDATE
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: WALK-FORWARD VALIDATION")
    print("="*70)
    
    validation = model.validate(features_scaled, targets)
    
    # ========================================================================
    # STEP 6: CURRENT PREDICTIONS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: CURRENT PREDICTIONS")
    print("="*70)
    
    current_features = features_scaled.iloc[[-1]]
    current_probs = model.predict(current_features)
    
    # Combine with results
    summary = model.summary()
    summary['probability'] = current_probs.T.iloc[:, 0]
    summary = summary.sort_values('probability', ascending=False)
    
    print("\nCurrent Rotation Probabilities (Next 21 Days):")
    print(summary.to_string())
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(model.results).T
    
    print("\nBy Category:")
    for category in ['FINANCIALS', 'MACRO_SENSITIVE', 'MIXED', 'SENTIMENT_DRIVEN']:
        subset = results_df[results_df['category'] == category]
        if len(subset) > 0:
            print(f"\n{category}:")
            print(f"  Mean Test Acc: {subset['test_acc'].mean():.3f}")
            print(f"  Mean Gap: {subset['gap'].mean():.3f}")
            print(f"  Sectors: {', '.join(subset.index)}")
    
    print("\nGap Analysis:")
    excellent = results_df[results_df['gap'] < 0.20]
    good = results_df[(results_df['gap'] >= 0.20) & (results_df['gap'] < 0.30)]
    poor = results_df[results_df['gap'] >= 0.30]
    
    print(f"  ✅ Excellent (<0.20): {len(excellent)} sectors")
    print(f"  🟡 Good (0.20-0.30): {len(good)} sectors")
    print(f"  ⚠️  Poor (>0.30): {len(poor)} sectors")
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)
    
    return model, features_scaled, results_df, validation, summary


if __name__ == "__main__":
    model, features, results, validation, summary = main()