"""
Test Script: Analyze Statistical Thresholds on Your Training Data
This script DOES NOT modify anything - it just shows you what would change.

Run this BEFORE integrating to see the impact.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from config import TRAINING_YEARS

# Add the phase 1 calculator (save previous artifact as statistical_thresholds.py)
from statistical_thresholds import (
    StatisticalThresholdCalculator,
    test_threshold_calculator_on_training_data,
    export_threshold_analysis
)


def load_existing_system():
    """Load your existing trained system."""
    print("Loading existing system...")
    
    try:
        from integrated_system_production import IntegratedMarketSystemV4
        
        system = IntegratedMarketSystemV4()
        system.train(years=TRAINING_YEARS, real_time_vix=False, verbose=False)
        
        print("✅ System loaded successfully")
        return system
        
    except Exception as e:
        print(f"❌ Failed to load system: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_training_scores(system):
    """
    Extract ensemble anomaly scores from training data.
    This is what we'll use to calculate statistical thresholds.
    """
    print("\nExtracting training scores from anomaly detector...")
    
    # Get the trained detector
    detector = system.vix_predictor.anomaly_detector
    
    if not detector.trained:
        print("❌ Detector not trained")
        return None
    
    # Get training features
    features = system.vix_predictor.features
    
    # Calculate ensemble scores for ALL training samples
    print(f"Computing scores for {len(features)} training samples...")
    
    all_scores = []
    
    # Process in batches for performance
    batch_size = 100
    for i in range(0, len(features), batch_size):
        batch = features.iloc[i:i+batch_size]
        
        for idx in range(len(batch)):
            try:
                single_sample = batch.iloc[[idx]]
                result = detector.detect(single_sample, verbose=False)
                score = result['ensemble']['score']
                all_scores.append(score)
            except:
                pass
        
        if (i + batch_size) % 500 == 0:
            print(f"  Processed {i + batch_size}/{len(features)} samples...")
    
    training_scores = np.array(all_scores)
    
    print(f"\n✅ Extracted {len(training_scores)} training scores")
    print(f"   Range: [{training_scores.min():.4f}, {training_scores.max():.4f}]")
    print(f"   Mean: {training_scores.mean():.4f} ± {training_scores.std():.4f}")
    
    return training_scores


def analyze_current_vs_statistical(system, training_scores):
    """
    Compare current hardcoded thresholds with statistical ones.
    Show how many classifications would change.
    """
    print("\n" + "="*80)
    print("CLASSIFICATION IMPACT ANALYSIS")
    print("="*80)
    
    # Create calculator
    calc = StatisticalThresholdCalculator(training_scores)
    
    # Get current anomaly result
    current_features = system.vix_predictor.features.iloc[[-1]]
    current_result = system.vix_predictor.anomaly_detector.detect(
        current_features, verbose=False
    )
    current_score = current_result['ensemble']['score']
    
    print(f"\nCurrent Anomaly Score: {current_score:.4f}")
    
    # Classify using both methods
    legacy_level, _, _ = calc.classify_score(current_score, method='legacy')
    rec_level, p_value, confidence = calc.classify_score(current_score, method='recommended')
    
    print(f"\nLegacy Classification: {legacy_level}")
    print(f"Statistical Classification: {rec_level} (p={p_value:.4f}, {confidence:.0%} confidence)")
    
    if legacy_level != rec_level:
        print(f"\n⚠️  CLASSIFICATION WOULD CHANGE: {legacy_level} → {rec_level}")
    else:
        print(f"\n✅ Classifications match")
    
    # Test on a range of scores
    test_scores = np.array([0.30, 0.50, 0.65, 0.70, 0.75, 0.85, 0.90, 0.95])
    
    print("\n" + "-"*80)
    print("CLASSIFICATION COMPARISON ACROSS SCORE RANGE")
    print("-"*80)
    print(f"\n{'Score':<8} {'Legacy':<12} {'Statistical':<12} {'P-Value':<10} {'Changed':<8}")
    print("-"*60)
    
    changes = 0
    for score in test_scores:
        legacy, _, _ = calc.classify_score(score, method='legacy')
        stat, pval, _ = calc.classify_score(score, method='recommended')
        changed = "⚠️  YES" if legacy != stat else "✅ NO"
        
        if legacy != stat:
            changes += 1
        
        print(f"{score:<8.4f} {legacy:<12} {stat:<12} {pval:<10.4f} {changed:<8}")
    
    print(f"\n📊 Summary: {changes}/{len(test_scores)} scores would have different classifications")
    
    return calc


def generate_visual_data(calc: StatisticalThresholdCalculator, 
                        training_scores: np.ndarray):
    """
    Generate data for dashboard visualization.
    """
    print("\n" + "="*80)
    print("GENERATING DASHBOARD VISUALIZATION DATA")
    print("="*80)
    
    # Export threshold analysis JSON
    export_threshold_analysis(calc, './json_data/threshold_analysis.json')
    
    # Create visualization-ready data
    viz_data = {
        'score_distribution': {
            'bins': np.linspace(0, 1, 51).tolist(),
            'histogram': np.histogram(training_scores, bins=50, range=(0, 1))[0].tolist(),
            'density': True
        },
        'threshold_markers': calc.compute_all_thresholds(),
        'percentile_lines': {
            f'p{p}': float(np.percentile(training_scores, p))
            for p in [50, 75, 90, 95, 99]
        }
    }
    
    import json
    with open('./json_data/threshold_visualization.json', 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print("✅ threshold_visualization.json exported")
    
    return viz_data


def main():
    """Main test workflow."""
    print("\n" + "="*80)
    print("STATISTICAL THRESHOLD ANALYSIS - NON-BREAKING TEST")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load your existing system")
    print("  2. Extract training anomaly scores")
    print("  3. Calculate statistical thresholds")
    print("  4. Compare with current hardcoded thresholds")
    print("  5. Show how many classifications would change")
    print("  6. Export visualization data")
    print("\n⚠️  NO CHANGES will be made to your system")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Load system
    system = load_existing_system()
    if system is None:
        print("\n❌ Cannot proceed without system")
        return
    
    # Step 2: Extract training scores
    print("\n" + "="*80)
    print("STEP 2: EXTRACT TRAINING SCORES")
    print("="*80)
    
    training_scores = extract_training_scores(system)
    if training_scores is None:
        print("\n❌ Cannot proceed without training scores")
        return
    
    # Step 3: Run threshold analysis
    print("\n" + "="*80)
    print("STEP 3: CALCULATE STATISTICAL THRESHOLDS")
    print("="*80)
    
    results = test_threshold_calculator_on_training_data(
        training_scores, 
        verbose=True
    )
    
    # Step 4: Compare classifications
    calc = analyze_current_vs_statistical(system, training_scores)
    
    # Step 5: Generate visualization data
    viz_data = generate_visual_data(calc, training_scores)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n📁 Files Generated:")
    print("  ✅ ./json_data/threshold_analysis.json")
    print("  ✅ ./json_data/threshold_visualization.json")
    
    print("\n📊 Next Steps:")
    print("  1. Review the threshold comparison above")
    print("  2. Check the exported JSON files")
    print("  3. If satisfied, proceed to Phase 2 integration")
    print("  4. We'll then update the dashboard to visualize both methods")
    
    # Save training scores as JSON with metadata for dashboard use
    import json
    from datetime import datetime
    
    training_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'training_years': TRAINING_YEARS,
            'total_samples': len(training_scores),
            'date_range': {
                'start': str(system.vix_predictor.features.index[0].date()),
                'end': str(system.vix_predictor.features.index[-1].date())
            },
            'statistics': {
                'mean': float(training_scores.mean()),
                'std': float(training_scores.std()),
                'min': float(training_scores.min()),
                'max': float(training_scores.max()),
                'percentiles': {
                    'p50': float(np.percentile(training_scores, 50)),
                    'p75': float(np.percentile(training_scores, 75)),
                    'p90': float(np.percentile(training_scores, 90)),
                    'p95': float(np.percentile(training_scores, 95)),
                    'p99': float(np.percentile(training_scores, 99))
                }
            }
        },
        'thresholds': {
            'legacy': {
                'moderate': 0.50,
                'high': 0.70,
                'critical': 0.85
            },
            'statistical': calc.compute_all_thresholds()['recommended']
        },
        'time_series': [
            {
                'date': str(date.date()),
                'ensemble_score': float(score),
                'vix': float(vix_val)
            }
            for date, score, vix_val in zip(
                system.vix_predictor.features.index,
                training_scores,
                system.vix_predictor.vix_ml.values
            )
        ]
    }
    
    output_path = Path('./json_data/training_anomaly_scores.json')
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    size_kb = output_path.stat().st_size / 1024
    print(f"\n💾 Training scores saved to: training_anomaly_scores.json ({size_kb:.1f} KB)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()