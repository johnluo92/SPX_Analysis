"""Quick Test - Verify Refactored System Works"""
import sys

print("\n" + "="*80)
print("TESTING REFACTORED SYSTEM")
print("="*80)

# Test 1: Imports
print("\n[1/4] Testing imports...")
try:
    from config import REGIME_BOUNDARIES, REGIME_NAMES, TRAINING_YEARS, ANOMALY_FEATURE_GROUPS
    from core.data_fetcher import UnifiedDataFetcher
    from core.feature_engine import UnifiedFeatureEngine
    from core.anomaly_detector import MultiDimensionalAnomalyDetector
    from integrated_system_production import AnomalyOrchestrator, IntegratedMarketSystemV4
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Config
print("\n[2/4] Testing config...")
try:
    assert len(REGIME_BOUNDARIES) == 5, "Wrong number of regime boundaries"
    assert len(REGIME_NAMES) == 4, "Wrong number of regime names"
    assert TRAINING_YEARS > 0, "Training years must be positive"
    assert len(ANOMALY_FEATURE_GROUPS) > 10, "Missing anomaly feature groups"
    print(f"✅ Config OK - {len(ANOMALY_FEATURE_GROUPS)} anomaly groups")
except Exception as e:
    print(f"❌ Config test failed: {e}")
    sys.exit(1)

# Test 3: Instantiation
print("\n[3/4] Testing instantiation...")
try:
    orchestrator = AnomalyOrchestrator()
    assert orchestrator.fetcher is not None, "Fetcher not initialized"
    assert orchestrator.trained == False, "Should not be trained yet"
    
    detector = MultiDimensionalAnomalyDetector(contamination=0.05)
    assert detector.contamination == 0.05, "Wrong contamination"
    
    system = IntegratedMarketSystemV4()
    assert system.feature_engine is not None, "Feature engine not initialized"
    assert system.orchestrator is not None, "Orchestrator not initialized"
    
    print("✅ All classes instantiate correctly")
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    sys.exit(1)

# Test 4: Interface
print("\n[4/4] Testing interface...")
try:
    required_attrs = ['fetcher', 'anomaly_detector', 'vix_ml', 'features', 'trained']
    for attr in required_attrs:
        assert hasattr(orchestrator, attr), f"Missing attribute: {attr}"
    
    required_methods = ['train', 'detect_current', 'save_state', 'load_state']
    for method in required_methods:
        assert hasattr(orchestrator, method), f"Missing method: {method}"
        assert callable(getattr(orchestrator, method)), f"{method} not callable"
    
    print("✅ Interface complete")
except Exception as e:
    print(f"❌ Interface test failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("🎉 ALL TESTS PASSED - System ready!")
print("="*80)
print("\nNext: Run 'python integrated_system_production.py'")
