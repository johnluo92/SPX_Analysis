"""
Production Readiness Test Suite
================================
Tests real-world failure scenarios and edge cases that could break in production.

Run from src/:
    python tests/test_production_readiness.py
    python tests/test_production_readiness.py --critical-only
    python tests/test_production_readiness.py --category data
"""

import sys
import json
import traceback
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Test results tracking
test_results = {
    'critical': {'passed': [], 'failed': []},
    'important': {'passed': [], 'failed': []},
    'advisory': {'passed': [], 'failed': []},
    'warnings': []
}


def record(severity, test_name, passed, message=""):
    """Record test result by severity"""
    category = 'passed' if passed else 'failed'
    test_results[severity][category].append(f"{test_name}: {message}")


# =============================================================================
# CRITICAL: System-Breaking Issues
# =============================================================================

def test_parallel_array_integrity():
    """CRITICAL: JSON arrays must be parallel (dates, scores, SPX aligned)"""
    print("\n" + "="*70)
    print("CRITICAL: Parallel Array Integrity")
    print("="*70)
    
    try:
        json_path = Path('json_data/historical_anomaly_scores.json')
        
        if not json_path.exists():
            print("⚠️  File not found - run training first")
            test_results['warnings'].append("historical_anomaly_scores.json missing")
            return
        
        with open(json_path) as f:
            data = json.load(f)
        
        # Check required arrays
        required = ['dates', 'ensemble_scores', 'spx_close']
        missing = [k for k in required if k not in data]
        
        if missing:
            print(f"❌ Missing arrays: {missing}")
            record('critical', 'Parallel arrays', False, f"missing {missing}")
            return
        
        # Check lengths
        lengths = {k: len(data[k]) for k in required}
        
        if len(set(lengths.values())) != 1:
            print(f"❌ Length mismatch: {lengths}")
            record('critical', 'Parallel arrays', False, f"lengths {lengths}")
            return
        
        print(f"✅ All arrays aligned: {lengths['dates']} observations")
        
        # Check forward returns alignment
        if 'forward_returns' in data:
            if 'forward_1d' in data['forward_returns']:
                fwd_len = len(data['forward_returns']['forward_1d'])
                if fwd_len != lengths['dates']:
                    print(f"❌ Forward returns misaligned: {fwd_len} vs {lengths['dates']}")
                    record('critical', 'Forward returns alignment', False, f"{fwd_len} != {lengths['dates']}")
                else:
                    print(f"✅ Forward returns aligned: {fwd_len} observations")
                    record('critical', 'Forward returns alignment', True)
        
        record('critical', 'Parallel arrays', True, f"{lengths['dates']} aligned")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record('critical', 'Parallel arrays', False, str(e))


def test_refresh_state_loadable():
    """CRITICAL: Model state must be loadable for cached mode"""
    print("\n" + "="*70)
    print("CRITICAL: Refresh State Loadability")
    print("="*70)
    
    try:
        import pickle
        state_path = Path('json_data/refresh_state.pkl')
        
        if not state_path.exists():
            print("⚠️  refresh_state.pkl not found - run training first")
            test_results['warnings'].append("refresh_state.pkl missing")
            return
        
        # Try to load
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        # Validate structure
        required_keys = ['trained_detectors', 'feature_columns', 'statistical_thresholds']
        missing = [k for k in required_keys if k not in state]
        
        if missing:
            print(f"❌ Missing keys in state: {missing}")
            record('critical', 'Refresh state structure', False, f"missing {missing}")
            return
        
        print(f"✅ State loaded successfully")
        print(f"   Detectors: {len(state['trained_detectors'])}")
        print(f"   Features: {len(state['feature_columns'])}")
        
        # Check detector integrity
        for name, detector_dict in state['trained_detectors'].items():
            if 'model' not in detector_dict:
                print(f"❌ Detector {name} missing model")
                record('critical', f'Detector {name} integrity', False, "missing model")
                return
        
        print(f"✅ All {len(state['trained_detectors'])} detectors intact")
        record('critical', 'Refresh state loadable', True)
        
    except Exception as e:
        print(f"❌ Load failed: {e}")
        traceback.print_exc()
        record('critical', 'Refresh state loadable', False, str(e))


def test_feature_coverage_threshold():
    """CRITICAL: Must have >80% feature coverage or detection unreliable"""
    print("\n" + "="*70)
    print("CRITICAL: Feature Coverage Threshold")
    print("="*70)
    
    try:
        from config import ANOMALY_FEATURE_GROUPS
        from unified_feature_engine import UnifiedFeatureEngine
        
        engine = UnifiedFeatureEngine()
        
        # Get required features
        all_required = []
        for features in ANOMALY_FEATURE_GROUPS.values():
            all_required.extend(features)
        all_required = list(set(all_required))
        
        print(f"   Required features: {len(all_required)}")
        
        # Build features (use smaller window for speed)
        result = engine.build_complete_features(years=1)
        features = result['features']
        
        # Check coverage
        available = [f for f in all_required if f in features.columns]
        coverage = len(available) / len(all_required) * 100
        
        print(f"   Available features: {len(available)}")
        print(f"   Coverage: {coverage:.1f}%")
        
        if coverage < 80:
            print(f"❌ Coverage below 80% threshold")
            missing = [f for f in all_required if f not in features.columns]
            print(f"   Missing: {missing[:10]}..." if len(missing) > 10 else f"   Missing: {missing}")
            record('critical', 'Feature coverage', False, f"{coverage:.1f}% < 80%")
        else:
            print(f"✅ Coverage above threshold")
            record('critical', 'Feature coverage', True, f"{coverage:.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record('critical', 'Feature coverage', False, str(e))


def test_json_export_validity():
    """CRITICAL: All required JSON files must be valid and parseable"""
    print("\n" + "="*70)
    print("CRITICAL: JSON Export Validity")
    print("="*70)
    
    required_files = [
        'anomaly_report.json',
        'historical_anomaly_scores.json',
        'dashboard_data.json',
        'market_state.json'
    ]
    
    all_valid = True
    
    for filename in required_files:
        filepath = Path(f'json_data/{filename}')
        
        if not filepath.exists():
            print(f"❌ {filename}: Missing")
            record('critical', f'JSON {filename}', False, "missing")
            all_valid = False
            continue
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            if not data:
                print(f"❌ {filename}: Empty")
                record('critical', f'JSON {filename}', False, "empty")
                all_valid = False
            else:
                size_kb = filepath.stat().st_size / 1024
                print(f"✅ {filename}: Valid ({size_kb:.1f} KB)")
                record('critical', f'JSON {filename}', True)
        
        except json.JSONDecodeError as e:
            print(f"❌ {filename}: Invalid JSON - {e}")
            record('critical', f'JSON {filename}', False, f"parse error: {e}")
            all_valid = False
    
    if all_valid:
        print(f"\n✅ All {len(required_files)} files valid")


# =============================================================================
# IMPORTANT: Data Quality Issues
# =============================================================================

def test_nan_propagation():
    """IMPORTANT: NaN values in features should not exceed 15%"""
    print("\n" + "="*70)
    print("IMPORTANT: NaN Propagation Control")
    print("="*70)
    
    try:
        from unified_feature_engine import UnifiedFeatureEngine
        
        engine = UnifiedFeatureEngine()
        result = engine.build_complete_features(years=1)
        features = result['features']
        
        # Calculate NaN percentage
        total_cells = features.size
        nan_cells = features.isna().sum().sum()
        nan_pct = (nan_cells / total_cells) * 100
        
        print(f"   Total cells: {total_cells:,}")
        print(f"   NaN cells: {nan_cells:,}")
        print(f"   NaN percentage: {nan_pct:.2f}%")
        
        if nan_pct > 15:
            print(f"❌ Excessive NaN values (>{15}%)")
            record('important', 'NaN propagation', False, f"{nan_pct:.2f}%")
        elif nan_pct > 10:
            print(f"⚠️  High NaN values (10-15%)")
            record('important', 'NaN propagation', True, f"{nan_pct:.2f}% (warning)")
        else:
            print(f"✅ NaN levels acceptable")
            record('important', 'NaN propagation', True, f"{nan_pct:.2f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record('important', 'NaN propagation', False, str(e))


def test_cache_staleness():
    """IMPORTANT: No cache files should exceed 90-day TTL"""
    print("\n" + "="*70)
    print("IMPORTANT: Cache Staleness Check")
    print("="*70)
    
    try:
        cache_dir = Path('data_cache')
        
        if not cache_dir.exists():
            print("⚠️  Cache directory not found")
            test_results['warnings'].append("data_cache missing")
            return
        
        # Find all parquet files
        cache_files = list(cache_dir.glob('*.parquet'))
        
        if not cache_files:
            print("⚠️  No cache files found")
            test_results['warnings'].append("No cache files")
            return
        
        now = datetime.now()
        stale_files = []
        
        for filepath in cache_files:
            age_days = (now - datetime.fromtimestamp(filepath.stat().st_mtime)).days
            if age_days > 90:
                stale_files.append((filepath.name, age_days))
        
        if stale_files:
            print(f"❌ Found {len(stale_files)} stale cache files (>90d):")
            for name, age in stale_files[:5]:
                print(f"   • {name}: {age} days old")
            if len(stale_files) > 5:
                print(f"   ... and {len(stale_files) - 5} more")
            record('important', 'Cache staleness', False, f"{len(stale_files)} stale files")
        else:
            print(f"✅ All {len(cache_files)} cache files fresh (<90d)")
            record('important', 'Cache staleness', True)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record('important', 'Cache staleness', False, str(e))


def test_detector_training_stability():
    """IMPORTANT: All detectors must train without errors"""
    print("\n" + "="*70)
    print("IMPORTANT: Detector Training Stability")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        from config import ANOMALY_FEATURE_GROUPS
        
        detector = MultiDimensionalAnomalyDetector()
        
        # Create minimal test dataset
        all_features = []
        for features in ANOMALY_FEATURE_GROUPS.values():
            all_features.extend(features)
        all_features = list(set(all_features))
        
        test_df = pd.DataFrame(
            np.random.randn(500, len(all_features)),  # Sufficient samples
            columns=all_features
        )
        
        print(f"   Training with {len(all_features)} features, {len(test_df)} samples...")
        
        # Train
        detector.train(test_df, verbose=False)
        
        # Check all detectors trained
        trained_count = len(detector.trained_detectors)
        expected_count = len(ANOMALY_FEATURE_GROUPS) + 5  # Domain + random
        
        if trained_count < expected_count:
            print(f"❌ Only {trained_count}/{expected_count} detectors trained")
            record('important', 'Detector training', False, f"{trained_count}/{expected_count}")
        else:
            print(f"✅ All {trained_count} detectors trained successfully")
            record('important', 'Detector training', True)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        record('important', 'Detector training', False, str(e))


def test_threshold_sanity():
    """IMPORTANT: Statistical thresholds must be ordered and in [0, 1]"""
    print("\n" + "="*70)
    print("IMPORTANT: Threshold Sanity Check")
    print("="*70)
    
    try:
        json_path = Path('json_data/anomaly_report.json')
        
        if not json_path.exists():
            print("⚠️  anomaly_report.json not found")
            test_results['warnings'].append("anomaly_report.json missing")
            return
        
        with open(json_path) as f:
            data = json.load(f)
        
        if 'classification' not in data or 'thresholds' not in data['classification']:
            print("❌ Thresholds missing from report")
            record('important', 'Threshold structure', False, "missing")
            return
        
        thresholds = data['classification']['thresholds']
        
        # Extract values (handle both simple and CI formats)
        moderate = thresholds.get('moderate', 0)
        high = thresholds.get('high', 0)
        critical = thresholds.get('critical', 0)
        
        print(f"   Moderate: {moderate:.4f}")
        print(f"   High:     {high:.4f}")
        print(f"   Critical: {critical:.4f}")
        
        # Check ordering
        if not (moderate < high < critical):
            print(f"❌ Thresholds not properly ordered")
            record('important', 'Threshold ordering', False, f"{moderate} < {high} < {critical}")
            return
        
        # Check range
        if not (0 <= moderate <= 1 and 0 <= high <= 1 and 0 <= critical <= 1):
            print(f"❌ Thresholds outside [0, 1] range")
            record('important', 'Threshold range', False, "out of bounds")
            return
        
        # Check spacing (should be reasonable)
        spacing_low = high - moderate
        spacing_high = critical - high
        
        if spacing_low < 0.02 or spacing_high < 0.02:
            print(f"⚠️  Thresholds very close together (spacing: {spacing_low:.3f}, {spacing_high:.3f})")
            record('important', 'Threshold spacing', True, "tight spacing")
        else:
            print(f"✅ Thresholds properly ordered and spaced")
            record('important', 'Threshold sanity', True)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record('important', 'Threshold sanity', False, str(e))


# =============================================================================
# ADVISORY: Performance and Best Practices
# =============================================================================

def test_memory_baseline_established():
    """ADVISORY: Memory baseline should be set for leak detection"""
    print("\n" + "="*70)
    print("ADVISORY: Memory Baseline")
    print("="*70)
    
    try:
        from integrated_system_production import IntegratedMarketSystemV4
        
        system = IntegratedMarketSystemV4()
        
        if not system.memory_monitoring_enabled:
            print("⚠️  Memory monitoring disabled (psutil not installed)")
            test_results['warnings'].append("psutil not installed - no memory monitoring")
            return
        
        if system.baseline_memory_mb is None:
            print("⚠️  No baseline established yet (run training to set)")
            record('advisory', 'Memory baseline', False, "not set")
        else:
            print(f"✅ Baseline established: {system.baseline_memory_mb:.1f} MB")
            record('advisory', 'Memory baseline', True, f"{system.baseline_memory_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        record('advisory', 'Memory baseline', False, str(e))


def test_file_sizes_reasonable():
    """ADVISORY: Export files should be within expected size ranges"""
    print("\n" + "="*70)
    print("ADVISORY: File Size Sanity Check")
    print("="*70)
    
    expected_ranges = {
        'anomaly_report.json': (1, 50),  # KB
        'historical_anomaly_scores.json': (100, 500),
        'dashboard_data.json': (5, 50),
        'market_state.json': (1, 20),
        'refresh_state.pkl': (5000, 25000),  # ~5-25 MB
    }
    
    issues = []
    
    for filename, (min_kb, max_kb) in expected_ranges.items():
        filepath = Path(f'json_data/{filename}')
        
        if not filepath.exists():
            continue
        
        size_kb = filepath.stat().st_size / 1024
        
        if size_kb < min_kb:
            print(f"⚠️  {filename}: Too small ({size_kb:.1f} KB < {min_kb} KB)")
            issues.append(f"{filename} too small")
        elif size_kb > max_kb:
            print(f"⚠️  {filename}: Too large ({size_kb:.1f} KB > {max_kb} KB)")
            issues.append(f"{filename} too large")
        else:
            print(f"✅ {filename}: {size_kb:.1f} KB (expected {min_kb}-{max_kb})")
    
    if issues:
        record('advisory', 'File sizes', False, f"{len(issues)} anomalies")
    else:
        record('advisory', 'File sizes', True)


def test_bootstrap_ci_present():
    """ADVISORY: Bootstrap CIs should be present in thresholds"""
    print("\n" + "="*70)
    print("ADVISORY: Bootstrap CI Presence")
    print("="*70)
    
    try:
        json_path = Path('json_data/anomaly_report.json')
        
        if not json_path.exists():
            print("⚠️  anomaly_report.json not found")
            return
        
        with open(json_path) as f:
            data = json.load(f)
        
        thresholds = data.get('classification', {}).get('thresholds', {})
        
        # Check for CI keys
        has_cis = all(
            f'{level}_ci' in thresholds 
            for level in ['moderate', 'high', 'critical']
        )
        
        if has_cis:
            print("✅ Bootstrap CIs present in thresholds")
            record('advisory', 'Bootstrap CIs', True)
        else:
            print("⚠️  Bootstrap CIs missing (using old threshold format)")
            record('advisory', 'Bootstrap CIs', False, "not found")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        record('advisory', 'Bootstrap CIs', False, str(e))


def test_detector_coverage_weights():
    """ADVISORY: Detector coverage weights should be present"""
    print("\n" + "="*70)
    print("ADVISORY: Detector Coverage Weights")
    print("="*70)
    
    try:
        json_path = Path('json_data/anomaly_report.json')
        
        if not json_path.exists():
            print("⚠️  anomaly_report.json not found")
            return
        
        with open(json_path) as f:
            data = json.load(f)
        
        # Check for weight_stats in data_quality
        if 'data_quality' in data and 'weight_stats' in data['data_quality']:
            stats = data['data_quality']['weight_stats']
            print(f"✅ Coverage weights present:")
            print(f"   Mean: {stats['mean']:.3f}")
            print(f"   Min:  {stats['min']:.3f}")
            print(f"   Max:  {stats['max']:.3f}")
            record('advisory', 'Coverage weights', True)
        else:
            print("⚠️  Coverage weights missing")
            record('advisory', 'Coverage weights', False, "not found")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        record('advisory', 'Coverage weights', False, str(e))


# =============================================================================
# Main Runner
# =============================================================================

def print_summary():
    """Print test summary by severity"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    # Count by severity
    for severity in ['critical', 'important', 'advisory']:
        passed = len(test_results[severity]['passed'])
        failed = len(test_results[severity]['failed'])
        total = passed + failed
        
        if total == 0:
            continue
        
        status = "✅" if failed == 0 else "❌"
        print(f"\n{status} {severity.upper()}: {passed}/{total} passed")
        
        if failed > 0:
            for failure in test_results[severity]['failed']:
                print(f"   ❌ {failure}")
    
    # Show warnings
    if test_results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(test_results['warnings'])}):")
        for warning in test_results['warnings']:
            print(f"   • {warning}")
    
    print("\n" + "="*70)
    
    # Overall status
    critical_failed = len(test_results['critical']['failed'])
    important_failed = len(test_results['important']['failed'])
    
    if critical_failed > 0:
        print("❌ CRITICAL FAILURES - System not production-ready")
        print("\nAction: Fix critical issues before deployment")
        return 1
    elif important_failed > 0:
        print("⚠️  IMPORTANT FAILURES - System degraded")
        print("\nAction: Address important issues for optimal operation")
        return 1
    else:
        print("✅ ALL TESTS PASSED - Production ready")
        print("\nNext: Deploy with confidence")
        return 0


def main():
    """Run test suite"""
    parser = argparse.ArgumentParser(description='Production readiness test suite')
    parser.add_argument('--critical-only', action='store_true', help='Run only critical tests')
    parser.add_argument('--category', choices=['critical', 'important', 'advisory'], 
                       help='Run specific category only')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PRODUCTION READINESS TEST SUITE")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Working directory: {Path.cwd()}")
    
    # Define test categories
    critical_tests = [
        test_parallel_array_integrity,
        test_refresh_state_loadable,
        test_feature_coverage_threshold,
        test_json_export_validity,
    ]
    
    important_tests = [
        test_nan_propagation,
        test_cache_staleness,
        test_detector_training_stability,
        test_threshold_sanity,
    ]
    
    advisory_tests = [
        test_memory_baseline_established,
        test_file_sizes_reasonable,
        test_bootstrap_ci_present,
        test_detector_coverage_weights,
    ]
    
    # Run based on arguments
    if args.critical_only or args.category == 'critical':
        print("\n🔴 Running CRITICAL tests only...\n")
        for test in critical_tests:
            test()
    
    elif args.category == 'important':
        print("\n🟡 Running IMPORTANT tests only...\n")
        for test in important_tests:
            test()
    
    elif args.category == 'advisory':
        print("\n🟢 Running ADVISORY tests only...\n")
        for test in advisory_tests:
            test()
    
    else:
        print("\n🔍 Running ALL tests...\n")
        for test in critical_tests + important_tests + advisory_tests:
            test()
    
    # Print summary
    exit_code = print_summary()
    
    # Save results
    results_file = Path('tests/production_test_results.json')
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                severity: {
                    'passed': len(test_results[severity]['passed']),
                    'failed': len(test_results[severity]['failed'])
                }
                for severity in ['critical', 'important', 'advisory']
            },
            'details': test_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
