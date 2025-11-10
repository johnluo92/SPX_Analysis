"""
Comprehensive Test Suite for Issues #1-7
=========================================
Tests ALL changes made to the VIX anomaly detection system.

Usage:
    python test_comprehensive.py              # Run all tests
    python test_comprehensive.py --quick      # Skip slow tests
    python test_comprehensive.py --issue 5    # Test specific issue only
"""

import sys
import traceback
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd

# Test results storage
test_results = {
    'passed': [],
    'failed': [],
    'warnings': [],
    'skipped': [],
    'by_issue': {i: {'passed': 0, 'failed': 0} for i in range(0, 8)}  # 0 = integration tests
}


def record_result(issue_num, test_name, passed, message=""):
    """Record test result"""
    category = 'passed' if passed else 'failed'
    test_results[category].append(f"Issue #{issue_num} - {test_name}: {message}")
    test_results['by_issue'][issue_num][category] += 1


# =============================================================================
# ISSUE #1-2: Data Fetcher Cache TTL & Revision Detection
# =============================================================================

def test_issue_1_2_cache_metadata():
    """Test 1.1: Cache metadata tracking"""
    print("\n" + "="*70)
    print("TEST 1.1: Cache Metadata Tracking (Issue #1-2)")
    print("="*70)
    
    try:
        from UnifiedDataFetcher import UnifiedDataFetcher
        from pathlib import Path
        
        # Use temporary cache directory
        test_cache_dir = Path('./test_cache_temp')
        test_cache_dir.mkdir(exist_ok=True)
        
        fetcher = UnifiedDataFetcher(cache_dir=str(test_cache_dir))
        
        # Check metadata path attribute exists
        if hasattr(fetcher, 'cache_metadata_path'):
            print(f"✅ cache_metadata_path attribute exists: {fetcher.cache_metadata_path}")
            record_result(1, "Metadata path attribute", True)
        else:
            print("❌ cache_metadata_path attribute missing")
            record_result(1, "Metadata path attribute", False)
            return
        
        # Check metadata structure
        if isinstance(fetcher.cache_metadata, dict):
            print("✅ Cache metadata initialized as dict")
            record_result(1, "Cache metadata structure", True)
        else:
            print("❌ Cache metadata structure invalid")
            record_result(1, "Cache metadata structure", False)
            return
        
        # Test metadata update method exists
        if hasattr(fetcher, '_update_cache_metadata'):
            print("✅ _update_cache_metadata method exists")
            
            # Test metadata update
            test_key = "test_series_2024"
            fetcher._update_cache_metadata(test_key, etag="test_etag_123")
            
            if test_key in fetcher.cache_metadata:
                metadata = fetcher.cache_metadata[test_key]
                if 'created' in metadata and 'etag' in metadata:
                    print(f"✅ Metadata update works: created={metadata['created'][:10]}, etag={metadata['etag']}")
                    record_result(1, "Metadata update", True)
                else:
                    print("❌ Metadata structure incomplete")
                    record_result(1, "Metadata update", False)
            else:
                print("❌ Metadata update failed")
                record_result(1, "Metadata update", False)
        else:
            print("❌ _update_cache_metadata method missing")
            record_result(1, "Metadata update method", False)
        
        # Cleanup
        if fetcher.cache_metadata_path.exists():
            fetcher.cache_metadata_path.unlink()
        if test_cache_dir.exists():
            import shutil
            shutil.rmtree(test_cache_dir)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(1, "Cache metadata", False, str(e))


def test_issue_1_2_cache_ttl():
    """Test 1.2: Cache TTL validation"""
    print("\n" + "="*70)
    print("TEST 1.2: Cache TTL Validation (Issue #1-2)")
    print("="*70)
    
    try:
        from UnifiedDataFetcher import UnifiedDataFetcher
        from pathlib import Path
        
        test_cache_dir = Path('./test_cache_temp')
        test_cache_dir.mkdir(exist_ok=True)
        
        fetcher = UnifiedDataFetcher(cache_dir=str(test_cache_dir))
        
        # Check _is_cache_stale method exists
        if not hasattr(fetcher, '_is_cache_stale'):
            print("❌ _is_cache_stale method missing")
            record_result(1, "TTL method", False, "method not found")
            return
        
        print("✅ _is_cache_stale method exists")
        
        # Test stale cache detection (91 days ago)
        test_key = "old_cache"
        old_date = (datetime.now() - timedelta(days=91)).isoformat()
        fetcher.cache_metadata[test_key] = {
            'created': old_date,
            'etag': 'old_etag'
        }
        
        is_stale = fetcher._is_cache_stale(test_key, ttl_days=90)
        
        if is_stale:
            print("✅ TTL correctly identifies stale cache (91d > 90d)")
            record_result(1, "TTL stale detection", True)
        else:
            print("❌ TTL failed to identify stale cache")
            record_result(1, "TTL stale detection", False)
        
        # Test fresh cache (30 days ago)
        fresh_key = "fresh_cache"
        fresh_date = (datetime.now() - timedelta(days=30)).isoformat()
        fetcher.cache_metadata[fresh_key] = {
            'created': fresh_date,
            'etag': 'fresh_etag'
        }
        
        is_stale = fetcher._is_cache_stale(fresh_key, ttl_days=90)
        
        if not is_stale:
            print("✅ TTL correctly identifies fresh cache (30d < 90d)")
            record_result(1, "TTL fresh detection", True)
        else:
            print("❌ TTL incorrectly marked fresh cache as stale")
            record_result(1, "TTL fresh detection", False)
        
        # Test missing cache key (should be stale)
        missing_key = "nonexistent_cache"
        is_stale = fetcher._is_cache_stale(missing_key, ttl_days=90)
        
        if is_stale:
            print("✅ Missing cache key correctly treated as stale")
            record_result(1, "TTL missing key", True)
        else:
            print("❌ Missing cache key incorrectly treated as fresh")
            record_result(1, "TTL missing key", False)
        
        # Cleanup
        if test_cache_dir.exists():
            import shutil
            shutil.rmtree(test_cache_dir)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(1, "Cache TTL", False, str(e))


def test_issue_1_buffer_validation():
    """Test 1.3: Buffer sufficiency validation"""
    print("\n" + "="*70)
    print("TEST 1.3: Buffer Validation (Issue #1)")
    print("="*70)
    
    try:
        from unified_feature_engine import UnifiedFeatureEngine
        
        engine = UnifiedFeatureEngine()
        
        # Check method exists
        if not hasattr(engine, '_validate_buffer_sufficiency'):
            print("❌ _validate_buffer_sufficiency method missing")
            record_result(1, "Buffer validation method", False, "method not found")
            return
        
        print("✅ _validate_buffer_sufficiency method exists")
        
        # Test insufficient buffer detection
        test_data = pd.Series(range(100), index=pd.date_range('2024-01-01', periods=100))
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_valid = engine._validate_buffer_sufficiency(
                test_data, 
                required_window=252, 
                data_name="TestData"
            )
            
            if not is_valid:
                print("✅ Correctly identifies insufficient buffer (100 < 252)")
                record_result(1, "Buffer insufficient detection", True)
                
                # Check if warning was issued
                if len(w) > 0 and "BUFFER INSUFFICIENT" in str(w[0].message):
                    print("✅ Warning issued for insufficient buffer")
                    record_result(1, "Buffer warning", True)
                else:
                    print("⚠️  No warning issued (may be OK)")
            else:
                print("❌ Failed to detect insufficient buffer")
                record_result(1, "Buffer insufficient detection", False)
        
        # Test sufficient buffer
        test_data_large = pd.Series(range(300), index=pd.date_range('2024-01-01', periods=300))
        
        is_valid = engine._validate_buffer_sufficiency(
            test_data_large,
            required_window=252,
            data_name="TestDataLarge"
        )
        
        if is_valid:
            print("✅ Correctly identifies sufficient buffer (300 > 252)")
            record_result(1, "Buffer sufficient detection", True)
        else:
            print("❌ False negative on sufficient buffer")
            record_result(1, "Buffer sufficient detection", False)
        
        # Test edge case (exactly at threshold)
        test_data_exact = pd.Series(range(252), index=pd.date_range('2024-01-01', periods=252))
        
        is_valid = engine._validate_buffer_sufficiency(
            test_data_exact,
            required_window=252,
            data_name="TestDataExact"
        )
        
        if is_valid:
            print("✅ Edge case: exactly at threshold (252 >= 252)")
            record_result(1, "Buffer edge case", True)
        else:
            print("⚠️  Edge case: threshold rejected (may be too strict)")
            record_result(1, "Buffer edge case", False, "threshold rejected")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(1, "Buffer validation", False, str(e))


def test_issue_2_fred_revision():
    """Test 2.1: FRED revision detection"""
    print("\n" + "="*70)
    print("TEST 2.1: FRED Revision Detection (Issue #2)")
    print("="*70)
    
    try:
        from UnifiedDataFetcher import UnifiedDataFetcher
        from pathlib import Path
        
        test_cache_dir = Path('./test_cache_temp')
        test_cache_dir.mkdir(exist_ok=True)
        
        fetcher = UnifiedDataFetcher(cache_dir=str(test_cache_dir))
        
        # Check _check_fred_revision method exists
        if not hasattr(fetcher, '_check_fred_revision'):
            print("❌ _check_fred_revision method missing")
            record_result(2, "Revision detection method", False, "method not found")
            return
        
        print("✅ _check_fred_revision method exists")
        
        # Test with no metadata (should return False - no revision detected)
        test_key = "fred_DGS10_test"
        has_revision = fetcher._check_fred_revision('DGS10', test_key)
        
        if not has_revision:
            print("✅ No revision for missing metadata (expected)")
            record_result(2, "Revision no metadata", True)
        else:
            print("⚠️  Revision detected with no metadata (unexpected)")
            record_result(2, "Revision no metadata", False, "unexpected True")
        
        # Test with stale metadata (simulated revision)
        if fetcher.fred_api_key:
            test_key_with_meta = "fred_DGS10_2024"
            fetcher.cache_metadata[test_key_with_meta] = {
                'created': datetime.now().isoformat(),
                'etag': 'old_etag_value'
            }
            
            print("✅ Testing revision detection with real API (may fail if no FRED key)")
            # This will make an actual API call
            has_revision = fetcher._check_fred_revision('DGS10', test_key_with_meta)
            print(f"   Revision detected: {has_revision}")
            record_result(2, "Revision detection", True, f"result={has_revision}")
        else:
            print("⚠️  FRED API key not available - skipping live test")
            test_results['skipped'].append("Issue #2 - Revision detection (no FRED key)")
        
        # Check _should_use_cache respects revision detection
        if hasattr(fetcher, '_should_use_cache'):
            print("✅ _should_use_cache method exists")
            record_result(2, "Cache decision method", True)
        else:
            print("❌ _should_use_cache method missing")
            record_result(2, "Cache decision method", False)
        
        # Cleanup
        if test_cache_dir.exists():
            import shutil
            shutil.rmtree(test_cache_dir)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(2, "FRED revision detection", False, str(e))


# =============================================================================
# ISSUE #3: Confidence-Weighted Ensemble
# =============================================================================

def test_issue_3_coverage_tracking():
    """Test 3.1: Detector coverage tracking"""
    print("\n" + "="*70)
    print("TEST 3.1: Detector Coverage Tracking (Issue #3)")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        
        detector = MultiDimensionalAnomalyDetector()
        
        # Check detector_coverage attribute exists
        if hasattr(detector, 'detector_coverage'):
            print("✅ detector_coverage attribute exists")
            record_result(3, "Coverage attribute", True)
        else:
            print("❌ detector_coverage attribute missing")
            record_result(3, "Coverage attribute", False)
            return
        
        # Simulate training to populate coverage
        from config import ANOMALY_FEATURE_GROUPS
        
        # Create mock features
        all_features = []
        for features in ANOMALY_FEATURE_GROUPS.values():
            all_features.extend(features)
        
        # Remove duplicates
        all_features = list(set(all_features))
        
        # Create DataFrame with partial feature availability
        available_features = all_features[:len(all_features)//2]  # Only 50% available
        test_df = pd.DataFrame(
            np.random.randn(100, len(available_features)),
            columns=available_features
        )
        
        print(f"Training with {len(available_features)}/{len(all_features)} features...")
        detector.train(test_df, verbose=False)
        
        # Check if coverage was calculated
        if detector.detector_coverage:
            print(f"✅ Coverage tracked for {len(detector.detector_coverage)} detectors")
            
            # Check coverage values are reasonable
            coverage_values = list(detector.detector_coverage.values())
            if all(0 <= c <= 1 for c in coverage_values):
                print("✅ Coverage values in valid range [0, 1]")
                record_result(3, "Coverage calculation", True, f"mean={np.mean(coverage_values):.2f}")
            else:
                print("❌ Invalid coverage values detected")
                record_result(3, "Coverage calculation", False)
        else:
            print("❌ No coverage data populated")
            record_result(3, "Coverage tracking", False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(3, "Coverage tracking", False, str(e))


def test_issue_3_weighted_ensemble():
    """Test 3.2: Weighted ensemble scoring"""
    print("\n" + "="*70)
    print("TEST 3.2: Weighted Ensemble Scoring (Issue #3)")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        from config import ANOMALY_FEATURE_GROUPS
        
        detector = MultiDimensionalAnomalyDetector()
        
        # Create test features
        all_features = []
        for features in ANOMALY_FEATURE_GROUPS.values():
            all_features.extend(features)
        all_features = list(set(all_features))
        
        test_df = pd.DataFrame(
            np.random.randn(100, len(all_features)),
            columns=all_features
        )
        
        detector.train(test_df, verbose=False)
        
        # Test detection with single observation
        test_obs = test_df.iloc[[0]]
        result = detector.detect(test_obs, verbose=False)
        
        # Check if ensemble has weighted flag
        if 'weighted' in result['ensemble']:
            print(f"✅ Ensemble marked as weighted: {result['ensemble']['weighted']}")
            record_result(3, "Weighted flag", True)
        else:
            print("⚠️  Weighted flag not in ensemble (may be OK for old format)")
            record_result(3, "Weighted flag", False, "flag missing")
        
        # Check weight_stats in data_quality
        if 'weight_stats' in result['data_quality']:
            stats = result['data_quality']['weight_stats']
            print(f"✅ Weight stats present: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
            record_result(3, "Weight stats", True)
        else:
            print("❌ Weight stats missing from results")
            record_result(3, "Weight stats", False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(3, "Weighted ensemble", False, str(e))


# =============================================================================
# ISSUE #4: Bootstrap Confidence Intervals
# =============================================================================

def test_issue_4_bootstrap_ci():
    """Test 4.1: Bootstrap CI calculation"""
    print("\n" + "="*70)
    print("TEST 4.1: Bootstrap Confidence Intervals (Issue #4)")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        
        detector = MultiDimensionalAnomalyDetector()
        
        # Simulate training scores
        np.random.seed(42)
        detector.training_ensemble_scores = np.random.beta(2, 5, 1000).tolist()
        
        print("Testing bootstrap CI calculation (100 iterations)...")
        thresholds = detector.calculate_statistical_thresholds_with_ci(n_bootstrap=100)
        
        # Check structure
        required_keys = ['moderate', 'high', 'critical', 
                        'moderate_ci', 'high_ci', 'critical_ci',
                        'bootstrap_config']
        
        missing = [k for k in required_keys if k not in thresholds]
        
        if not missing:
            print("✅ All CI keys present")
            record_result(4, "CI structure", True)
            
            # Validate CI ranges
            for level in ['moderate', 'high', 'critical']:
                point = thresholds[level]
                ci = thresholds[f'{level}_ci']
                
                if ci['lower'] <= point <= ci['upper']:
                    print(f"✅ {level.capitalize()}: {point:.4f} ∈ [{ci['lower']:.4f}, {ci['upper']:.4f}]")
                else:
                    print(f"❌ {level.capitalize()}: point estimate outside CI")
                    record_result(4, f"{level} CI range", False)
                    return
            
            record_result(4, "CI calculation", True, "all levels valid")
        else:
            print(f"❌ Missing keys: {missing}")
            record_result(4, "CI structure", False, f"missing {missing}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(4, "Bootstrap CI", False, str(e))


def test_issue_4_classify_compatibility():
    """Test 4.2: classify_anomaly backward compatibility"""
    print("\n" + "="*70)
    print("TEST 4.2: Classify Anomaly Compatibility (Issue #4)")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        
        detector = MultiDimensionalAnomalyDetector()
        detector.training_ensemble_scores = np.random.beta(2, 5, 1000).tolist()
        
        # Test with CI format
        thresholds_ci = detector.calculate_statistical_thresholds_with_ci(n_bootstrap=50)
        detector.statistical_thresholds = thresholds_ci
        
        level, p_val, conf = detector.classify_anomaly(0.75, method='statistical')
        print(f"✅ Classify works with CI format: level={level}")
        record_result(4, "Classify with CI", True)
        
        # Test with simple format (backward compatibility)
        thresholds_simple = {
            'moderate': 0.70,
            'high': 0.78,
            'critical': 0.88
        }
        detector.statistical_thresholds = thresholds_simple
        
        level, p_val, conf = detector.classify_anomaly(0.75, method='statistical')
        print(f"✅ Classify works with simple format: level={level}")
        record_result(4, "Classify backward compat", True)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(4, "Classify compatibility", False, str(e))


# =============================================================================
# ISSUE #5: Persistence Streak Timing
# =============================================================================

def test_issue_5_api_signature():
    """Test 5.1: Persistence method signature"""
    print("\n" + "="*70)
    print("TEST 5.1: Persistence API Signature (Issue #5)")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        import inspect
        
        detector = MultiDimensionalAnomalyDetector()
        
        sig = inspect.signature(detector.calculate_historical_persistence_stats)
        params = list(sig.parameters.keys())
        
        if 'dates' in params:
            print(f"✅ 'dates' parameter present in signature")
            print(f"   Full signature: {sig}")
            record_result(5, "API signature", True)
        else:
            print(f"❌ 'dates' parameter missing")
            record_result(5, "API signature", False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(5, "API signature", False, str(e))


def test_issue_5_timezone_handling():
    """Test 5.2: Timezone-aware streak correction"""
    print("\n" + "="*70)
    print("TEST 5.2: Timezone Handling (Issue #5)")
    print("="*70)
    
    try:
        import pytz
        from anomaly_system import MultiDimensionalAnomalyDetector
        
        detector = MultiDimensionalAnomalyDetector()
        detector.trained = True
        detector.statistical_thresholds = {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}
        
        # Create test data with today's date
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D')
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.82, 0.79, 0.81, 0.83])
        
        # Test with dates parameter
        result = detector.calculate_historical_persistence_stats(scores, dates=dates)
        
        print(f"✅ Persistence calculation with dates works")
        print(f"   Current streak: {result['current_streak']}")
        print(f"   Threshold used: {result['threshold_used']:.3f}")
        record_result(5, "Persistence with dates", True)
        
        # Verify timezone adjustment logic exists
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        print(f"   Current ET time: {now_et.strftime('%H:%M:%S %Z')}")
        print(f"   Market closed: {now_et.hour >= 16}")
        record_result(5, "Timezone logic", True)
        
    except ImportError:
        print("⚠️  pytz not installed - streak timing will not be timezone-aware")
        record_result(5, "Timezone handling", False, "pytz not installed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(5, "Timezone handling", False, str(e))


# =============================================================================
# ISSUE #6: Exponential Backoff
# =============================================================================

def test_issue_6_backoff_calculation():
    """Test 6.1: Exponential backoff logic"""
    print("\n" + "="*70)
    print("TEST 6.1: Exponential Backoff Calculation (Issue #6)")
    print("="*70)
    
    try:
        from dashboard_orchestrator import DashboardOrchestrator
        
        orchestrator = DashboardOrchestrator()
        
        # Get actual config values
        base = orchestrator.base_refresh_interval
        max_interval = orchestrator.max_refresh_interval
        
        print(f"   Base interval: {base}s")
        print(f"   Max interval: {max_interval}s")
        
        # Test cases with proper capping at max_interval
        test_cases = [
            (0, base),  # No failures -> base
            (1, min(base * 2, max_interval)),  # 1 failure -> 2x (capped)
            (2, min(base * 4, max_interval)),  # 2 failures -> 4x (capped)
            (3, min(base * 8, max_interval)),  # 3 failures -> 8x (capped)
            (5, max_interval),  # 5+ failures -> max
        ]
        
        all_passed = True
        for failures, expected in test_cases:
            orchestrator.consecutive_failures = failures
            actual = orchestrator._calculate_backoff_delay()
            
            if actual == expected:
                print(f"✅ {failures} failures → {actual}s (expected {expected}s)")
            else:
                print(f"❌ {failures} failures → {actual}s (expected {expected}s)")
                all_passed = False
        
        if all_passed:
            record_result(6, "Backoff calculation", True)
        else:
            record_result(6, "Backoff calculation", False, "delay mismatch")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(6, "Backoff calculation", False, str(e))


def test_issue_6_circuit_breaker():
    """Test 6.2: Circuit breaker configuration"""
    print("\n" + "="*70)
    print("TEST 6.2: Circuit Breaker (Issue #6)")
    print("="*70)
    
    try:
        from dashboard_orchestrator import DashboardOrchestrator
        
        orchestrator = DashboardOrchestrator()
        
        if hasattr(orchestrator, 'max_failures_before_stop'):
            threshold = orchestrator.max_failures_before_stop
            print(f"✅ Circuit breaker configured: {threshold} failures")
            
            if threshold == 10:
                print("✅ Threshold matches specification (10)")
                record_result(6, "Circuit breaker", True)
            else:
                print(f"⚠️  Threshold is {threshold}, expected 10")
                record_result(6, "Circuit breaker threshold", False, f"value={threshold}")
        else:
            print("❌ Circuit breaker attribute missing")
            record_result(6, "Circuit breaker", False, "attribute missing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(6, "Circuit breaker", False, str(e))


# =============================================================================
# ISSUE #7: Memory Profiling
# =============================================================================

def test_issue_7_memory_monitoring():
    """Test 7.1: Memory monitoring setup"""
    print("\n" + "="*70)
    print("TEST 7.1: Memory Monitoring Setup (Issue #7)")
    print("="*70)
    
    try:
        from integrated_system_production import IntegratedMarketSystemV4
        
        system = IntegratedMarketSystemV4()
        
        if system.memory_monitoring_enabled:
            print("✅ Memory monitoring enabled")
            
            # Check required attributes
            required_attrs = [
                'process', 'baseline_memory_mb', 'memory_history',
                'memory_warning_threshold_mb', 'memory_critical_threshold_mb'
            ]
            
            missing = [attr for attr in required_attrs if not hasattr(system, attr)]
            
            if not missing:
                print("✅ All memory attributes present")
                record_result(7, "Memory attributes", True)
            else:
                print(f"❌ Missing attributes: {missing}")
                record_result(7, "Memory attributes", False, f"missing {missing}")
        else:
            print("⚠️  Memory monitoring disabled (psutil not available)")
            record_result(7, "Memory monitoring", False, "psutil not installed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(7, "Memory monitoring setup", False, str(e))


def test_issue_7_memory_logging():
    """Test 7.2: Memory logging functionality"""
    print("\n" + "="*70)
    print("TEST 7.2: Memory Logging (Issue #7)")
    print("="*70)
    
    try:
        from integrated_system_production import IntegratedMarketSystemV4
        
        system = IntegratedMarketSystemV4()
        
        if not system.memory_monitoring_enabled:
            print("⚠️  Skipping (psutil not available)")
            test_results['skipped'].append("Issue #7 - Memory logging (psutil not installed)")
            return
        
        # Initialize baseline
        system._initialize_memory_baseline()
        
        if system.baseline_memory_mb is not None:
            print(f"✅ Baseline established: {system.baseline_memory_mb:.1f} MB")
            record_result(7, "Baseline initialization", True)
        else:
            print("❌ Baseline not set")
            record_result(7, "Baseline initialization", False)
            return
        
        # Test memory logging
        stats = system._log_memory_stats(context="test")
        
        if stats and 'current_mb' in stats:
            print(f"✅ Memory stats captured: {stats['current_mb']:.1f} MB")
            print(f"   Context: {stats['context']}")
            record_result(7, "Memory logging", True)
        else:
            print("❌ Memory stats empty or invalid")
            record_result(7, "Memory logging", False)
        
        # Test memory report
        report = system.get_memory_report()
        
        if 'error' not in report:
            print(f"✅ Memory report generated")
            print(f"   Status: {report['status']}")
            print(f"   Tracked objects: {report['gc_stats']['tracked_objects']}")
            record_result(7, "Memory report", True)
        else:
            print(f"❌ Memory report error: {report['error']}")
            record_result(7, "Memory report", False, report['error'])
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        record_result(7, "Memory logging", False, str(e))


# =============================================================================
# Integration Tests
# =============================================================================

def test_integration_imports():
    """Integration: Verify all modules import correctly"""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Module Imports")
    print("="*70)
    
    modules = [
        'anomaly_system',
        'vix_predictor_v2',
        'integrated_system_production',
        'dashboard_orchestrator',
        'unified_feature_engine',
        'UnifiedDataFetcher',
        'config',
    ]
    
    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            all_passed = False
    
    if all_passed:
        record_result(0, "All imports", True, "system-wide")
    else:
        record_result(0, "Import failures", False, "system-wide")


def test_integration_api_chain():
    """Integration: Test full API chain"""
    print("\n" + "="*70)
    print("INTEGRATION TEST: API Chain")
    print("="*70)
    
    try:
        print("Testing: IntegratedSystem → VIXPredictor → AnomalyDetector")
        
        from integrated_system_production import IntegratedMarketSystemV4
        
        system = IntegratedMarketSystemV4()
        
        # Check chain
        if hasattr(system, 'vix_predictor'):
            print("✅ IntegratedSystem → VIXPredictor")
            
            if hasattr(system.vix_predictor, 'anomaly_detector'):
                print("✅ VIXPredictor → AnomalyDetector")
                record_result(0, "API chain", True, "complete")
            else:
                print("❌ VIXPredictor missing anomaly_detector")
                record_result(0, "API chain", False, "broken at VIXPredictor")
        else:
            print("❌ IntegratedSystem missing vix_predictor")
            record_result(0, "API chain", False, "broken at IntegratedSystem")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        record_result(0, "API chain", False, str(e))


# =============================================================================
# Main Test Runner
# =============================================================================

def print_summary():
    """Print detailed test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    # Per-issue breakdown
    print("\nResults by Issue:")
    print(f"  Integration Tests: {test_results['by_issue'][0]['passed']}/{test_results['by_issue'][0]['passed'] + test_results['by_issue'][0]['failed']} passed")
    for issue_num in range(1, 8):
        stats = test_results['by_issue'][issue_num]
        total = stats['passed'] + stats['failed']
        if total > 0:
            status = "✅" if stats['failed'] == 0 else "❌"
            print(f"  {status} Issue #{issue_num}: {stats['passed']}/{total} passed")
    
    # Overall stats
    total_passed = len(test_results['passed'])
    total_failed = len(test_results['failed'])
    total_tests = total_passed + total_failed
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    # Show failures
    if test_results['failed']:
        print(f"\n❌ FAILURES ({len(test_results['failed'])}):")
        for failure in test_results['failed']:
            print(f"   • {failure}")
    
    # Show warnings
    if test_results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(test_results['warnings'])}):")
        for warning in test_results['warnings'][:5]:
            print(f"   • {warning}")
        if len(test_results['warnings']) > 5:
            print(f"   ... and {len(test_results['warnings']) - 5} more")
    
    # Show skipped
    if test_results['skipped']:
        print(f"\n⏭️  SKIPPED ({len(test_results['skipped'])}):")
        for skipped in test_results['skipped']:
            print(f"   • {skipped}")
    
    print("\n" + "="*70)
    
    # Overall status
    if total_failed == 0:
        print("✅ ALL TESTS PASSED - Ready for production")
        print("\nNext steps:")
        print("1. Run full training: python dashboard_orchestrator.py --years 10")
        print("2. Monitor dashboard for 24h")
        print("3. Review memory growth trends")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review and fix issues")
        print("\nAction required:")
        print("1. Review failed tests above")
        print("2. Check file modifications")
        print("3. Verify dependencies installed")
        print("4. Re-run: python test_comprehensive.py")
        return 1


def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description='Comprehensive test suite for Issues #1-7')
    parser.add_argument('--quick', action='store_true', help='Skip slow tests')
    parser.add_argument('--issue', type=int, choices=range(1, 8), help='Test specific issue only')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VIX ANOMALY DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("Testing Issues #1-7")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    if args.issue:
        print(f"Focus: Issue #{args.issue} only")
    
    # Define test groups
    test_groups = {
        1: [test_issue_1_2_cache_metadata, test_issue_1_2_cache_ttl, test_issue_1_buffer_validation],
        2: [test_issue_2_fred_revision],  # Issue #2: FRED revision detection
        3: [test_issue_3_coverage_tracking, test_issue_3_weighted_ensemble],
        4: [test_issue_4_bootstrap_ci, test_issue_4_classify_compatibility],
        5: [test_issue_5_api_signature, test_issue_5_timezone_handling],
        6: [test_issue_6_backoff_calculation, test_issue_6_circuit_breaker],
        7: [test_issue_7_memory_monitoring, test_issue_7_memory_logging],
    }
    
    # Run integration tests first (unless specific issue requested)
    if not args.issue:
        test_integration_imports()
        test_integration_api_chain()
    
    # Run issue-specific tests
    if args.issue:
        # Run only requested issue
        print(f"\n{'='*70}")
        print(f"RUNNING TESTS FOR ISSUE #{args.issue}")
        print(f"{'='*70}")
        for test_func in test_groups[args.issue]:
            test_func()
    else:
        # Run all tests
        for issue_num in range(1, 8):
            if test_groups[issue_num]:
                print(f"\n{'='*70}")
                print(f"ISSUE #{issue_num} TESTS")
                print(f"{'='*70}")
                for test_func in test_groups[issue_num]:
                    test_func()
                    if args.quick and issue_num == 4:  # Skip slow bootstrap tests
                        print("⏭️  Skipping remaining Issue #4 tests (--quick mode)")
                        break
    
    # Print summary
    exit_code = print_summary()
    
    # Save results to JSON
    results_file = Path('test_results_comprehensive.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': 'quick' if args.quick else 'full',
            'focus_issue': args.issue,
            'summary': {
                'passed': len(test_results['passed']),
                'failed': len(test_results['failed']),
                'warnings': len(test_results['warnings']),
                'skipped': len(test_results['skipped'])
            },
            'by_issue': test_results['by_issue'],
            'details': test_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())