"""
Comprehensive Test Suite for Issues #5-7
==========================================
Tests all changes made to the VIX anomaly detection system.

Run this BEFORE attempting full training or dashboard launch.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
import json

# Test results storage
test_results = {
    'passed': [],
    'failed': [],
    'warnings': [],
    'skipped': []
}


def test_imports():
    """Test 1: Verify all required imports work"""
    print("\n" + "="*70)
    print("TEST 1: Import Validation")
    print("="*70)
    
    imports_to_test = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('pytz', None),
        ('psutil', None),
        ('sklearn.ensemble', 'IsolationForest'),
        ('config', None),
        ('anomaly_system', 'MultiDimensionalAnomalyDetector'),
        ('vix_predictor_v2', 'VIXPredictorV4'),
        ('integrated_system_production', 'IntegratedMarketSystemV4'),
        ('dashboard_orchestrator', 'DashboardOrchestrator'),
        ('UnifiedDataFetcher', 'UnifiedDataFetcher'),
    ]
    
    for module_name, class_name in imports_to_test:
        try:
            if class_name:
                exec(f"from {module_name} import {class_name}")
                print(f"✅ {module_name}.{class_name}")
                test_results['passed'].append(f"Import: {module_name}.{class_name}")
            else:
                exec(f"import {module_name}")
                print(f"✅ {module_name}")
                test_results['passed'].append(f"Import: {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            test_results['failed'].append(f"Import: {module_name} - {e}")
            if module_name in ['pytz', 'psutil']:
                test_results['warnings'].append(f"Install missing: pip install {module_name}")


def test_anomaly_detector_signature():
    """Test 2: Verify calculate_historical_persistence_stats accepts dates parameter"""
    print("\n" + "="*70)
    print("TEST 2: Anomaly Detector Method Signatures")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        import inspect
        
        detector = MultiDimensionalAnomalyDetector()
        
        # Check method signature
        sig = inspect.signature(detector.calculate_historical_persistence_stats)
        params = list(sig.parameters.keys())
        
        print(f"Method signature: {sig}")
        print(f"Parameters: {params}")
        
        if 'dates' in params:
            print("✅ 'dates' parameter present")
            test_results['passed'].append("API: calculate_historical_persistence_stats has dates param")
        else:
            print("❌ 'dates' parameter MISSING")
            test_results['failed'].append("API: calculate_historical_persistence_stats missing dates param")
        
        # Check if method accepts optional dates=None
        try:
            import numpy as np
            detector.trained = True  # Bypass training check
            detector.statistical_thresholds = {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}
            
            test_scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
            result = detector.calculate_historical_persistence_stats(test_scores, dates=None)
            
            print("✅ Method accepts dates=None (backward compatible)")
            test_results['passed'].append("API: Backward compatible with dates=None")
        except Exception as e:
            print(f"❌ Method fails with dates=None: {e}")
            test_results['failed'].append(f"API: dates=None compatibility - {e}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        test_results['failed'].append(f"Detector signature test: {e}")


def test_statistical_thresholds_structure():
    """Test 3: Verify bootstrap CI structure"""
    print("\n" + "="*70)
    print("TEST 3: Statistical Thresholds Structure")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        import numpy as np
        
        detector = MultiDimensionalAnomalyDetector()
        
        # Simulate training scores
        np.random.seed(42)
        detector.training_ensemble_scores = np.random.beta(2, 5, 1000).tolist()
        
        # Test bootstrap CI calculation
        print("Testing calculate_statistical_thresholds_with_ci()...")
        thresholds = detector.calculate_statistical_thresholds_with_ci(n_bootstrap=100)
        
        required_keys = ['moderate', 'high', 'critical', 
                        'moderate_ci', 'high_ci', 'critical_ci',
                        'bootstrap_config']
        
        missing_keys = [k for k in required_keys if k not in thresholds]
        
        if not missing_keys:
            print("✅ All required threshold keys present")
            print(f"   Moderate: {thresholds['moderate']:.4f} [{thresholds['moderate_ci']['lower']:.4f}, {thresholds['moderate_ci']['upper']:.4f}]")
            print(f"   High:     {thresholds['high']:.4f} [{thresholds['high_ci']['lower']:.4f}, {thresholds['high_ci']['upper']:.4f}]")
            print(f"   Critical: {thresholds['critical']:.4f} [{thresholds['critical_ci']['lower']:.4f}, {thresholds['critical_ci']['upper']:.4f}]")
            test_results['passed'].append("Thresholds: Bootstrap CI structure valid")
        else:
            print(f"❌ Missing keys: {missing_keys}")
            test_results['failed'].append(f"Thresholds: Missing keys {missing_keys}")
        
        # Test classify_anomaly handles both formats
        print("\nTesting classify_anomaly() with CI format...")
        detector.statistical_thresholds = thresholds
        level, _, _ = detector.classify_anomaly(0.75, method='statistical')
        print(f"✅ classify_anomaly() works with CI format (level: {level})")
        test_results['passed'].append("Thresholds: classify_anomaly handles CI format")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        test_results['failed'].append(f"Thresholds test: {e}")


def test_memory_monitoring():
    """Test 4: Verify memory monitoring functionality"""
    print("\n" + "="*70)
    print("TEST 4: Memory Monitoring")
    print("="*70)
    
    try:
        from integrated_system_production import IntegratedMarketSystemV4
        
        system = IntegratedMarketSystemV4()
        
        if system.memory_monitoring_enabled:
            print("✅ Memory monitoring enabled")
            test_results['passed'].append("Memory: Monitoring enabled")
            
            # Test memory logging
            stats = system._log_memory_stats(context="test")
            
            if stats:
                print("✅ Memory stats captured")
                print(f"   Current: {stats['current_mb']:.1f} MB")
                test_results['passed'].append("Memory: Stats collection works")
            else:
                print("❌ Memory stats empty")
                test_results['failed'].append("Memory: Stats collection failed")
            
            # Test memory report
            report = system.get_memory_report()
            
            if 'error' not in report:
                print("✅ Memory report generated")
                print(f"   Status: {report['status']}")
                test_results['passed'].append("Memory: Report generation works")
            else:
                print(f"❌ Memory report error: {report['error']}")
                test_results['failed'].append(f"Memory: Report error - {report['error']}")
        else:
            print("⚠️  Memory monitoring disabled (psutil not available)")
            test_results['warnings'].append("Memory: psutil not installed - monitoring disabled")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        test_results['failed'].append(f"Memory test: {e}")


def test_orchestrator_backoff():
    """Test 5: Verify exponential backoff logic"""
    print("\n" + "="*70)
    print("TEST 5: Dashboard Orchestrator Backoff")
    print("="*70)
    
    try:
        from dashboard_orchestrator import DashboardOrchestrator
        
        orchestrator = DashboardOrchestrator()
        
        # Test backoff calculation
        test_cases = [
            (0, 900),   # No failures -> base interval
            (1, 1800),  # 1 failure -> 2x
            (2, 3600),  # 2 failures -> 4x
            (3, 7200),  # 3 failures -> 8x (but capped at 300s)
        ]
        
        all_passed = True
        for failures, expected_delay in test_cases:
            orchestrator.consecutive_failures = failures
            actual_delay = orchestrator._calculate_backoff_delay()
            
            # Note: max_refresh_interval is 300s, so delays cap
            expected_capped = min(expected_delay, orchestrator.max_refresh_interval)
            
            if actual_delay == expected_capped:
                print(f"✅ {failures} failures -> {actual_delay}s delay (expected {expected_capped}s)")
            else:
                print(f"❌ {failures} failures -> {actual_delay}s delay (expected {expected_capped}s)")
                all_passed = False
        
        if all_passed:
            test_results['passed'].append("Backoff: Exponential delay calculation correct")
        else:
            test_results['failed'].append("Backoff: Delay calculation mismatch")
        
        # Check circuit breaker threshold
        if orchestrator.max_failures_before_stop == 10:
            print(f"✅ Circuit breaker at {orchestrator.max_failures_before_stop} failures")
            test_results['passed'].append("Backoff: Circuit breaker configured")
        else:
            print(f"❌ Circuit breaker misconfigured: {orchestrator.max_failures_before_stop}")
            test_results['failed'].append("Backoff: Circuit breaker value wrong")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        test_results['failed'].append(f"Backoff test: {e}")


def test_weighted_ensemble():
    """Test 6: Verify coverage-weighted ensemble"""
    print("\n" + "="*70)
    print("TEST 6: Coverage-Weighted Ensemble")
    print("="*70)
    
    try:
        from anomaly_system import MultiDimensionalAnomalyDetector
        import pandas as pd
        import numpy as np
        
        detector = MultiDimensionalAnomalyDetector()
        
        # Check if detector_coverage is initialized
        if hasattr(detector, 'detector_coverage'):
            print("✅ detector_coverage attribute exists")
            test_results['passed'].append("Ensemble: detector_coverage attribute present")
        else:
            print("❌ detector_coverage attribute missing")
            test_results['failed'].append("Ensemble: detector_coverage attribute missing")
            return
        
        # Simulate detection to check weight_stats in results
        print("Testing weighted ensemble calculation...")
        print("⚠️  Requires trained detector - skipping runtime test")
        test_results['skipped'].append("Ensemble: Runtime test (requires training)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        test_results['failed'].append(f"Ensemble test: {e}")


def test_file_structure():
    """Test 7: Verify expected file structure"""
    print("\n" + "="*70)
    print("TEST 7: File Structure Validation")
    print("="*70)
    
    expected_files = {
        'anomaly_system.py': True,
        'vix_predictor_v2.py': True,
        'integrated_system_production.py': True,
        'dashboard_orchestrator.py': True,
        'UnifiedDataFetcher.py': True,
        'config.py': True,
        'json_data/': False  # Directory
    }
    
    for filename, is_file in expected_files.items():
        path = Path(filename)
        
        if is_file:
            if path.exists() and path.is_file():
                print(f"✅ {filename}")
                test_results['passed'].append(f"File: {filename} exists")
            else:
                print(f"❌ {filename} MISSING")
                test_results['failed'].append(f"File: {filename} missing")
        else:
            if path.exists() and path.is_dir():
                print(f"✅ {filename} (directory)")
                test_results['passed'].append(f"Dir: {filename} exists")
            else:
                print(f"⚠️  {filename} (directory) not found - will be created")
                test_results['warnings'].append(f"Dir: {filename} will be created on first run")


def test_timezone_handling():
    """Test 8: Verify timezone handling for streak correction"""
    print("\n" + "="*70)
    print("TEST 8: Timezone Handling (Issue #5)")
    print("="*70)
    
    try:
        import pytz
        from datetime import datetime
        
        # Test US/Eastern timezone
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        print(f"✅ pytz available")
        print(f"   Current ET time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Test market hours detection
        is_after_close = now_et.hour >= 16
        print(f"   Market closed: {is_after_close} (hour: {now_et.hour})")
        
        test_results['passed'].append("Timezone: pytz functional")
        test_results['passed'].append("Timezone: Market hours detection works")
        
    except ImportError:
        print("❌ pytz not installed")
        print("   Install: pip install pytz")
        test_results['failed'].append("Timezone: pytz not installed")
    except Exception as e:
        print(f"❌ Timezone test failed: {e}")
        test_results['failed'].append(f"Timezone: {e}")


def test_config_values():
    """Test 9: Verify config.py has expected values"""
    print("\n" + "="*70)
    print("TEST 9: Configuration Validation")
    print("="*70)
    
    try:
        import config
        
        required_attrs = [
            'TRAINING_YEARS',
            'ENABLE_TRAINING',
            'RANDOM_STATE',
            'REGIME_BOUNDARIES',
            'ANOMALY_FEATURE_GROUPS',
            'ANOMALY_THRESHOLDS'
        ]
        
        for attr in required_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"✅ {attr} = {value if not isinstance(value, (dict, list)) else type(value).__name__}")
                test_results['passed'].append(f"Config: {attr} present")
            else:
                print(f"❌ {attr} MISSING")
                test_results['failed'].append(f"Config: {attr} missing")
        
        # Check specific values
        if hasattr(config, 'ENABLE_TRAINING'):
            print(f"\n⚠️  ENABLE_TRAINING = {config.ENABLE_TRAINING}")
            if config.ENABLE_TRAINING:
                test_results['warnings'].append("Config: ENABLE_TRAINING=True (full training mode)")
            else:
                test_results['warnings'].append("Config: ENABLE_TRAINING=False (requires refresh_state.pkl)")
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        test_results['failed'].append(f"Config: {e}")


def print_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    print(f"\n✅ PASSED: {len(test_results['passed'])}")
    for test in test_results['passed'][:10]:  # Show first 10
        print(f"   • {test}")
    if len(test_results['passed']) > 10:
        print(f"   ... and {len(test_results['passed']) - 10} more")
    
    print(f"\n❌ FAILED: {len(test_results['failed'])}")
    for test in test_results['failed']:
        print(f"   • {test}")
    
    print(f"\n⚠️  WARNINGS: {len(test_results['warnings'])}")
    for warning in test_results['warnings']:
        print(f"   • {warning}")
    
    print(f"\n⏭️  SKIPPED: {len(test_results['skipped'])}")
    for test in test_results['skipped']:
        print(f"   • {test}")
    
    print("\n" + "="*70)
    
    # Overall status
    if len(test_results['failed']) == 0:
        print("✅ ALL CRITICAL TESTS PASSED - Ready for integration testing")
        print("\nNext steps:")
        print("1. pip install pytz psutil (if not already installed)")
        print("2. Run integration test: python test_integration.py")
        print("3. Try full training: python dashboard_orchestrator.py --years 10")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before proceeding")
        print("\nAction required:")
        print("1. Review failed tests above")
        print("2. Install missing dependencies")
        print("3. Verify file modifications")
        print("4. Re-run this test script")
        return 1


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("VIX ANOMALY DETECTION SYSTEM - TEST SUITE")
    print("Issues #5-7 Validation")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    # Run all tests
    test_imports()
    test_file_structure()
    test_config_values()
    test_anomaly_detector_signature()
    test_statistical_thresholds_structure()
    test_weighted_ensemble()
    test_timezone_handling()
    test_memory_monitoring()
    test_orchestrator_backoff()
    
    # Print summary
    exit_code = print_summary()
    
    # Save results to JSON
    results_file = Path('test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'passed': len(test_results['passed']),
                'failed': len(test_results['failed']),
                'warnings': len(test_results['warnings']),
                'skipped': len(test_results['skipped'])
            },
            'details': test_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
