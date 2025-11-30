"""
Hyperparameter Tuning Integration Example

Demonstrates how to integrate the nested CV hyperparameter tuner
with the existing VIX forecasting system.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Import hyperparameter tuner
from hyperparameter_tuner import (
    NestedCVHyperparameterTuner,
    run_hyperparameter_optimization
)

# Import existing system components
from integrated_system import IntegratedVIXForecastingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_for_tuning(cache_dir: str = "./data_cache") -> pd.DataFrame:
    """
    Load and prepare data using the integrated system.
    
    Returns DataFrame with features, targets, and metadata ready for tuning.
    """
    logger.info("Loading data via IntegratedVIXForecastingSystem...")
    
    # Initialize integrated system
    system = IntegratedVIXForecastingSystem(
        cache_dir=cache_dir,
        enable_training=True
    )
    
    # Build features
    df = system.build_complete_features()
    
    logger.info(f"Loaded {len(df)} rows spanning {df.index[0].date()} to {df.index[-1].date()}")
    logger.info(f"Features: {len(df.columns)} columns")
    
    # Validate data quality
    if 'target_vix_pct_change' not in df.columns:
        logger.error("Missing target column - run target calculation first")
        raise ValueError("Targets not calculated")
    
    # Check for sufficient data
    non_null_targets = df['target_vix_pct_change'].notna().sum()
    logger.info(f"Valid targets: {non_null_targets} ({non_null_targets/len(df):.1%})")
    
    if non_null_targets < 1000:
        logger.warning("Insufficient data for robust tuning - consider longer history")
    
    return df


def run_quick_tuning_test(df: pd.DataFrame, save_dir: str = "tuning_results_test"):
    """
    Run a quick tuning test with minimal trials for validation.
    
    Args:
        df: Prepared DataFrame
        save_dir: Directory to save test results
    """
    logger.info("\n" + "="*80)
    logger.info("RUNNING QUICK TUNING TEST (2 splits, 20 trials each)")
    logger.info("="*80)
    
    tuner = NestedCVHyperparameterTuner(
        n_outer_splits=2,
        n_trials_per_split=20,
        train_window_months=24,
        test_window_months=6,
        gap_days=5,
        objectives=['magnitude_mae', 'direction_f1', 'calibration'],
        diversity_weight=0.15,
        save_dir=save_dir,
        enable_pruning=True,
        timeout_hours=0.5  # 30 minute timeout
    )
    
    results = tuner.run_nested_cv_optimization(df)
    
    logger.info("\n✅ Quick test complete!")
    logger.info(f"Results saved to: {save_dir}")
    
    return results


def run_full_tuning(df: pd.DataFrame, save_dir: str = "tuning_results_full"):
    """
    Run full hyperparameter optimization with production settings.
    
    Args:
        df: Prepared DataFrame
        save_dir: Directory to save results
    """
    logger.info("\n" + "="*80)
    logger.info("RUNNING FULL HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    logger.info("⚠️  This will take several hours - consider running overnight")
    logger.info("="*80)
    
    tuner = NestedCVHyperparameterTuner(
        n_outer_splits=5,
        n_trials_per_split=200,  # As specified in design doc
        train_window_months=24,
        test_window_months=6,
        gap_days=5,
        objectives=['magnitude_mae', 'direction_f1', 'calibration'],
        diversity_weight=0.15,
        save_dir=save_dir,
        enable_pruning=True,
        timeout_hours=20  # Allow up to 20 hours
    )
    
    results = tuner.run_nested_cv_optimization(df)
    
    logger.info("\n✅ Full optimization complete!")
    logger.info(f"Results saved to: {save_dir}")
    
    return results


def apply_optimized_config(config_file: str):
    """
    Apply optimized configuration to config.py
    
    Args:
        config_file: Path to optimized config JSON file
    """
    import json
    from pathlib import Path
    
    logger.info(f"Loading optimized config from: {config_file}")
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    params = config_data['parameters']
    
    # Generate Python config code
    config_updates = []
    
    # Quality filter
    if 'quality_threshold' in params:
        config_updates.append(f"QUALITY_FILTER_CONFIG['min_threshold'] = {params['quality_threshold']:.4f}")
    
    # Cohort weights
    if 'fomc_period_weight' in params:
        config_updates.append(f"CALENDAR_COHORTS['fomc_period']['weight'] = {params['fomc_period_weight']:.4f}")
    if 'opex_week_weight' in params:
        config_updates.append(f"CALENDAR_COHORTS['opex_week']['weight'] = {params['opex_week_weight']:.4f}")
    if 'earnings_heavy_weight' in params:
        config_updates.append(f"CALENDAR_COHORTS['earnings_heavy']['weight'] = {params['earnings_heavy_weight']:.4f}")
    
    # Feature selection
    if 'magnitude_top_n' in params:
        config_updates.append(f"FEATURE_SELECTION_CONFIG['magnitude_top_n'] = {params['magnitude_top_n']}")
    if 'direction_top_n' in params:
        config_updates.append(f"FEATURE_SELECTION_CONFIG['direction_top_n'] = {params['direction_top_n']}")
    if 'correlation_threshold' in params:
        config_updates.append(f"FEATURE_SELECTION_CONFIG['correlation_threshold'] = {params['correlation_threshold']:.4f}")
    if 'target_overlap' in params:
        config_updates.append(f"FEATURE_SELECTION_CONFIG['target_overlap'] = {params['target_overlap']:.4f}")
    
    # Magnitude model params
    mag_params = {}
    for key, value in params.items():
        if key.startswith('mag_'):
            param_name = key.replace('mag_', '')
            mag_params[param_name] = value
    
    if mag_params:
        config_updates.append("\n# Magnitude Model Parameters")
        for key, value in mag_params.items():
            if isinstance(value, float):
                config_updates.append(f"MAGNITUDE_PARAMS['{key}'] = {value:.4f}")
            else:
                config_updates.append(f"MAGNITUDE_PARAMS['{key}'] = {value}")
    
    # Direction model params
    dir_params = {}
    for key, value in params.items():
        if key.startswith('dir_'):
            param_name = key.replace('dir_', '')
            dir_params[param_name] = value
    
    if dir_params:
        config_updates.append("\n# Direction Model Parameters")
        for key, value in dir_params.items():
            if isinstance(value, float):
                config_updates.append(f"DIRECTION_PARAMS['{key}'] = {value:.4f}")
            else:
                config_updates.append(f"DIRECTION_PARAMS['{key}'] = {value}")
    
    # Ensemble params
    if 'magnitude_weight' in params:
        config_updates.append("\n# Ensemble Configuration")
        config_updates.append(f"ENSEMBLE_CONFIG['confidence_weights']['magnitude'] = {params['magnitude_weight']:.4f}")
        config_updates.append(f"ENSEMBLE_CONFIG['confidence_weights']['direction'] = {params['direction_weight']:.4f}")
        config_updates.append(f"ENSEMBLE_CONFIG['confidence_weights']['agreement'] = {params['agreement_weight']:.4f}")
    
    if 'magnitude_thresholds' in params:
        thresholds = params['magnitude_thresholds']
        config_updates.append(f"ENSEMBLE_CONFIG['magnitude_thresholds']['small'] = {thresholds['small']:.4f}")
        config_updates.append(f"ENSEMBLE_CONFIG['magnitude_thresholds']['medium'] = {thresholds['medium']:.4f}")
        config_updates.append(f"ENSEMBLE_CONFIG['magnitude_thresholds']['large'] = {thresholds['large']:.4f}")
    
    if 'agreement_bonus' in params:
        bonus = params['agreement_bonus']
        config_updates.append(f"ENSEMBLE_CONFIG['agreement_bonus']['moderate'] = {bonus['moderate']:.4f}")
        config_updates.append(f"ENSEMBLE_CONFIG['agreement_bonus']['strong'] = {bonus['strong']:.4f}")
    
    if 'contradiction_penalty' in params:
        penalty = params['contradiction_penalty']
        config_updates.append(f"ENSEMBLE_CONFIG['contradiction_penalty']['minor'] = {penalty['minor']:.4f}")
        config_updates.append(f"ENSEMBLE_CONFIG['contradiction_penalty']['moderate'] = {penalty['moderate']:.4f}")
        config_updates.append(f"ENSEMBLE_CONFIG['contradiction_penalty']['severe'] = {penalty['severe']:.4f}")
    
    # Write update script
    update_script = Path("apply_optimized_config.py")
    
    with open(update_script, 'w') as f:
        f.write('"""\nApply optimized hyperparameters to config.py\n')
        f.write(f'Generated from: {config_file}\n')
        f.write(f'Timestamp: {config_data["timestamp"]}\n')
        f.write(f'Config Hash: {config_data["config_hash"]}\n')
        f.write('"""\n\n')
        f.write('import sys\n')
        f.write('sys.path.insert(0, "src")\n\n')
        f.write('from config import (\n')
        f.write('    QUALITY_FILTER_CONFIG, CALENDAR_COHORTS,\n')
        f.write('    FEATURE_SELECTION_CONFIG, MAGNITUDE_PARAMS,\n')
        f.write('    DIRECTION_PARAMS, ENSEMBLE_CONFIG\n')
        f.write(')\n\n')
        f.write('# Apply optimized parameters\n')
        for update in config_updates:
            f.write(update + '\n')
        f.write('\nprint("✅ Configuration updated successfully!")\n')
        f.write(f'print("Config hash: {config_data["config_hash"]}")\n')
    
    logger.info(f"\n✅ Generated config update script: {update_script}")
    logger.info(f"Run with: python {update_script}")
    logger.info(f"\nPerformance Summary:")
    logger.info(f"  Magnitude MAE: {config_data['performance_summary']['magnitude_mae']['mean']:.2f}% "
                f"± {config_data['performance_summary']['magnitude_mae']['std']:.2f}%")
    logger.info(f"  Direction F1: {config_data['performance_summary']['direction_f1']['mean']:.3f} "
                f"± {config_data['performance_summary']['direction_f1']['std']:.3f}")


def compare_configs(baseline_metrics: dict, tuned_config_file: str):
    """
    Compare baseline vs tuned configuration performance.
    
    Args:
        baseline_metrics: Dict with baseline performance metrics
        tuned_config_file: Path to tuned config JSON
    """
    import json
    
    with open(tuned_config_file, 'r') as f:
        tuned_data = json.load(f)
    
    tuned_perf = tuned_data['performance_summary']
    
    print("\n" + "="*80)
    print("BASELINE VS TUNED CONFIGURATION COMPARISON")
    print("="*80)
    
    print("\nMagnitude MAE:")
    print(f"  Baseline: {baseline_metrics.get('magnitude_mae', 0):.2f}%")
    print(f"  Tuned:    {tuned_perf['magnitude_mae']['mean']:.2f}% ± {tuned_perf['magnitude_mae']['std']:.2f}%")
    
    if 'magnitude_mae' in baseline_metrics:
        improvement = ((baseline_metrics['magnitude_mae'] - tuned_perf['magnitude_mae']['mean']) / 
                      baseline_metrics['magnitude_mae'] * 100)
        print(f"  Improvement: {improvement:+.1f}%")
    
    print("\nDirection F1:")
    print(f"  Baseline: {baseline_metrics.get('direction_f1', 0):.3f}")
    print(f"  Tuned:    {tuned_perf['direction_f1']['mean']:.3f} ± {tuned_perf['direction_f1']['std']:.3f}")
    
    if 'direction_f1' in baseline_metrics:
        improvement = ((tuned_perf['direction_f1']['mean'] - baseline_metrics['direction_f1']) / 
                      baseline_metrics['direction_f1'] * 100)
        print(f"  Improvement: {improvement:+.1f}%")
    
    print("\nCalibration Error:")
    print(f"  Baseline: {baseline_metrics.get('calibration_error', 0):.3f}")
    print(f"  Tuned:    {tuned_perf['calibration_error']['mean']:.3f} ± {tuned_perf['calibration_error']['std']:.3f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VIX Forecasting Hyperparameter Tuning")
    parser.add_argument(
        '--mode',
        choices=['test', 'full', 'apply'],
        default='test',
        help='Tuning mode: test (quick), full (production), apply (use existing config)'
    )
    parser.add_argument(
        '--cache-dir',
        default='./data_cache',
        help='Data cache directory'
    )
    parser.add_argument(
        '--config-file',
        help='Path to optimized config file (for apply mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'apply':
        if not args.config_file:
            logger.error("--config-file required for apply mode")
            exit(1)
        
        apply_optimized_config(args.config_file)
    
    else:
        # Load data
        logger.info("Preparing data...")
        df = prepare_data_for_tuning(args.cache_dir)
        
        if args.mode == 'test':
            results = run_quick_tuning_test(df)
        else:
            results = run_full_tuning(df)
        
        logger.info("\n✅ Tuning complete!")
        logger.info(f"Check {results} for detailed results")
