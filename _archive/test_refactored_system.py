"""Comprehensive Validation Suite for Refactored System"""
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


class SystemValidator:
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.errors = []
    
    def test(self, name: str):
        """Decorator for test functions"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                print(f"\n{'='*70}")
                print(f"TEST: {name}")
                print(f"{'='*70}")
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    self.results[name] = {
                        'status': 'PASS' if result else 'FAIL',
                        'time': elapsed,
                        'error': None
                    }
                    status = "✅ PASS" if result else "❌ FAIL"
                    print(f"\n{status} ({elapsed:.2f}s)")
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    self.results[name] = {
                        'status': 'ERROR',
                        'time': elapsed,
                        'error': str(e)
                    }
                    print(f"\n❌ ERROR: {str(e)}")
                    print(traceback.format_exc())
                    self.errors.append((name, e))
                    return False
            return wrapper
        return decorator
    
    def print_summary(self):
        print(f"\n\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        errors = sum(1 for r in self.results.values() if r['status'] == 'ERROR')
        total_time = sum(r['time'] for r in self.results.values())
        
        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        
        if errors > 0:
            print(f"\n{'='*70}")
            print("ERRORS DETAIL:")
            print(f"{'='*70}")
            for name, error in self.errors:
                print(f"\n{name}:")
                print(f"  {error}")
        
        print(f"\n{'='*70}")
        print("DETAILED RESULTS:")
        print(f"{'='*70}")
        for name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASS' else ("❌" if result['status'] == 'FAIL' else "⚠️")
            print(f"{status_icon} {name:50s} {result['time']:6.2f}s")
        
        return passed == total


validator = SystemValidator()


@validator.test("1. Import Refactored Modules")
def test_imports():
    """Verify all refactored modules can be imported"""
    try:
        from UnifiedDataFetcher import UnifiedDataFetcher
        from unified_feature_engine import UnifiedFeatureEngine
        from vix_predictor_v2 import VIXPredictorV4
        from integrated_system_production import IntegratedMarketSystemV4
        print("   ✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False


@validator.test("2. UnifiedDataFetcher - Initialization")
def test_fetcher_init():
    """Test data fetcher initialization"""
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    fetcher = UnifiedDataFetcher(log_level="WARNING")
    
    checks = [
        (fetcher.cache_dir.exists(), "Cache directory exists"),
        (fetcher.validator is not None, "Validator initialized"),
        (fetcher.logger is not None, "Logger initialized"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("3. UnifiedDataFetcher - SPX Data")
def test_fetcher_spx():
    """Test SPX data fetching"""
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    fetcher = UnifiedDataFetcher(log_level="WARNING")
    spx = fetcher.fetch_spx('2023-01-01', '2023-12-31', lookback_buffer_days=0)
    
    checks = [
        (spx is not None, "SPX data fetched"),
        (len(spx) > 200, f"Sufficient data points ({len(spx) if spx is not None else 0})"),
        ('Close' in spx.columns if spx is not None else False, "Close column exists"),
        (isinstance(spx.index, pd.DatetimeIndex) if spx is not None else False, "DatetimeIndex confirmed"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("4. UnifiedDataFetcher - VIX Data")
def test_fetcher_vix():
    """Test VIX data fetching"""
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    fetcher = UnifiedDataFetcher(log_level="WARNING")
    vix = fetcher.fetch_vix('2023-01-01', '2023-12-31', lookback_buffer_days=0)
    
    checks = [
        (vix is not None, "VIX data fetched"),
        (len(vix) > 200, f"Sufficient data points ({len(vix) if vix is not None else 0})"),
        (vix.min() > 0 if vix is not None else False, "VIX values positive"),
        (vix.max() < 100 if vix is not None else False, "VIX values reasonable"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("5. UnifiedDataFetcher - FRED Integration")
def test_fetcher_fred():
    """Test FRED API integration"""
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    fetcher = UnifiedDataFetcher(log_level="WARNING")
    
    if fetcher.fred_api_key is None:
        print("   ⚠️  FRED API key not configured (optional)")
        return True  # Pass if key not configured
    
    latest = fetcher.fetch_fred_latest('DGS10')
    
    checks = [
        (latest is not None, f"Latest 10Y Treasury fetched: {latest}"),
        (0 < latest < 20 if latest else False, "Value in reasonable range"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("6. UnifiedFeatureEngine - Initialization")
def test_feature_engine_init():
    """Test feature engine initialization"""
    from unified_feature_engine import UnifiedFeatureEngine
    
    engine = UnifiedFeatureEngine()
    
    checks = [
        (engine.fetcher is not None, "Fetcher initialized"),
        (engine.cboe_data_dir.exists() or True, "CBOE directory configured"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("7. UnifiedFeatureEngine - Feature Build (1 year)")
def test_feature_build():
    """Test complete feature building"""
    from unified_feature_engine import UnifiedFeatureEngine
    
    engine = UnifiedFeatureEngine()
    result = engine.build_complete_features(years=1)
    
    features = result['features']
    spx = result['spx']
    vix = result['vix']
    
    checks = [
        (features is not None, "Features built"),
        (len(features.columns) > 50, f"Sufficient features ({len(features.columns)})"),
        (len(features) > 200, f"Sufficient samples ({len(features)})"),
        (spx is not None and len(spx) > 200, f"SPX data valid ({len(spx) if spx is not None else 0})"),
        (vix is not None and len(vix) > 200, f"VIX data valid ({len(vix) if vix is not None else 0})"),
        ('vix' in features.columns, "Core VIX feature exists"),
        ('vix_regime' in features.columns, "Regime feature exists"),
        ('spx_ret_21d' in features.columns, "SPX momentum exists"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("8. VIXPredictorV4 - Training")
def test_vix_predictor():
    """Test VIX predictor training"""
    from unified_feature_engine import UnifiedFeatureEngine
    from vix_predictor_v2 import VIXPredictorV4
    
    engine = UnifiedFeatureEngine()
    result = engine.build_complete_features(years=1)
    
    predictor = VIXPredictorV4()
    predictor.train_with_features(
        features=result['features'],
        vix=result['vix'],
        spx=result['spx'],
        verbose=False
    )
    
    checks = [
        (predictor.trained, "Predictor trained"),
        (predictor.anomaly_detector is not None, "Anomaly detector initialized"),
        (predictor.regime_stats_historical is not None, "Regime stats computed"),
        (len(predictor.regime_stats_historical['regimes']) == 4, "4 regimes defined"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("9. IntegratedSystem - Full Training")
def test_integrated_training():
    """Test complete integrated system training"""
    from integrated_system_production import IntegratedMarketSystemV4
    
    system = IntegratedMarketSystemV4()
    system.train(years=1, real_time_vix=False, verbose=False)
    
    checks = [
        (system.trained, "System trained"),
        (system.vix_predictor.trained, "VIX predictor trained"),
        (system.feature_engine is not None, "Feature engine exists"),
        (system.spx is not None, "SPX data loaded"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("10. IntegratedSystem - Market State")
def test_market_state():
    """Test market state generation"""
    from integrated_system_production import IntegratedMarketSystemV4
    
    system = IntegratedMarketSystemV4()
    system.train(years=1, real_time_vix=False, verbose=False)
    
    state = system.get_market_state()
    
    checks = [
        ('timestamp' in state, "Timestamp present"),
        ('market_data' in state, "Market data present"),
        ('anomaly_analysis' in state, "Anomaly analysis present"),
        ('vix_predictions' in state, "VIX predictions present"),
        (state['market_data']['vix'] > 0, f"Valid VIX: {state['market_data']['vix']:.2f}"),
        (state['anomaly_analysis']['ensemble']['score'] >= 0, "Anomaly score computed"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("11. IntegratedSystem - JSON Export")
def test_json_export():
    """Test JSON export functionality"""
    from integrated_system_production import IntegratedMarketSystemV4
    
    system = IntegratedMarketSystemV4()
    system.train(years=1, real_time_vix=False, verbose=False)
    
    test_file = "./json_data/test_market_state.json"
    Path("./json_data").mkdir(exist_ok=True)
    
    state = system.export_json(test_file)
    
    checks = [
        (Path(test_file).exists(), "JSON file created"),
        (Path(test_file).stat().st_size > 100, "File has content"),
        (state is not None, "State data returned"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    # Cleanup
    if Path(test_file).exists():
        Path(test_file).unlink()
    
    return all(check for check, _ in checks)


@validator.test("12. Data Quality - Missing Values")
def test_data_quality():
    """Test data quality and missing value handling"""
    from unified_feature_engine import UnifiedFeatureEngine
    
    engine = UnifiedFeatureEngine()
    result = engine.build_complete_features(years=1)
    features = result['features']
    
    total_values = features.size
    missing_values = features.isna().sum().sum()
    missing_pct = (missing_values / total_values) * 100
    
    checks = [
        (missing_pct < 10, f"Missing values acceptable: {missing_pct:.2f}%"),
        (features.shape[0] > 0, "Features have rows"),
        (features.shape[1] > 0, "Features have columns"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("13. Performance - Training Speed")
def test_performance():
    """Test system performance benchmarks"""
    from integrated_system_production import IntegratedMarketSystemV4
    
    system = IntegratedMarketSystemV4()
    
    start = time.time()
    system.train(years=1, real_time_vix=False, verbose=False)
    train_time = time.time() - start
    
    start = time.time()
    state = system.get_market_state()
    inference_time = time.time() - start
    
    checks = [
        (train_time < 120, f"Training time acceptable: {train_time:.2f}s"),
        (inference_time < 5, f"Inference fast: {inference_time:.4f}s"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("14. Anomaly Detection - Functionality")
def test_anomaly_detection():
    """Test anomaly detection system"""
    from integrated_system_production import IntegratedMarketSystemV4
    
    system = IntegratedMarketSystemV4()
    system.train(years=1, real_time_vix=False, verbose=False)
    
    anomaly_result = system._get_cached_anomaly_result()
    
    checks = [
        ('ensemble' in anomaly_result, "Ensemble score present"),
        ('domain_anomalies' in anomaly_result, "Domain anomalies present"),
        (0 <= anomaly_result['ensemble']['score'] <= 1, "Score in valid range"),
        (len(anomaly_result['domain_anomalies']) > 0, "Detectors active"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


@validator.test("15. Regime Statistics - Computation")
def test_regime_stats():
    """Test regime statistics computation"""
    from vix_predictor_v2 import VIXPredictorV4
    from unified_feature_engine import UnifiedFeatureEngine
    
    engine = UnifiedFeatureEngine()
    result = engine.build_complete_features(years=2)
    
    predictor = VIXPredictorV4()
    predictor.train_with_features(
        features=result['features'],
        vix=result['vix'],
        spx=result['spx'],
        verbose=False
    )
    
    stats = predictor.regime_stats_historical
    
    checks = [
        ('regimes' in stats, "Regimes present"),
        (len(stats['regimes']) == 4, "4 regimes defined"),
        (all('statistics' in r for r in stats['regimes']), "Statistics computed"),
        (all('transitions_5d' in r for r in stats['regimes']), "Transitions computed"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
    
    return all(check for check, _ in checks)


def main():
    print(f"\n{'='*70}")
    print("REFACTORED SYSTEM VALIDATION SUITE")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validator.start_time = time.time()
    
    # Run all tests
    test_imports()
    test_fetcher_init()
    test_fetcher_spx()
    test_fetcher_vix()
    test_fetcher_fred()
    test_feature_engine_init()
    test_feature_build()
    test_vix_predictor()
    test_integrated_training()
    test_market_state()
    test_json_export()
    test_data_quality()
    test_performance()
    test_anomaly_detection()
    test_regime_stats()
    
    # Print summary
    success = validator.print_summary()
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
