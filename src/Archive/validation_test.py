"""
Quick Validation Test Script
Run this to check your current system for temporal leakage risks.
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.temporal_validator import TemporalSafetyValidator, run_full_validation
from config import PUBLICATION_LAGS

def main():
    print("=" * 80)
    print("TEMPORAL SAFETY VALIDATION TEST")
    print("=" * 80)
    
    # Initialize validator
    print(f"\n📋 Loaded {len(PUBLICATION_LAGS)} publication lags from config")
    print("\nSample lags:")
    for source, lag in list(PUBLICATION_LAGS.items())[:5]:
        print(f"  {source:<15} → {lag} days")
    
    # Run full validation
    validator = run_full_validation(
        feature_engine_path="core/feature_engine.py",
        publication_lags=PUBLICATION_LAGS
    )
    
    # Additional checks
    print("\n" + "=" * 80)
    print("🔍 ADDITIONAL CHECKS")
    print("=" * 80)
    
    # Check 1: Verify data fetcher integration
    print("\n[Check 1] Verifying data fetcher temporal safety...")
    try:
        from core.data_fetcher import UnifiedDataFetcher
        from datetime import datetime, timedelta
        
        fetcher = UnifiedDataFetcher()
        
        # Test VIX (T+0)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        vix = fetcher.fetch_yahoo(
            '^VIX', 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if vix is not None:
            expected_lag = PUBLICATION_LAGS.get('^VIX', 0)
            latest_date = vix.index[-1]
            expected_latest = end_date - timedelta(days=expected_lag)
            
            print(f"  VIX last date: {latest_date.date()}")
            print(f"  Expected (with {expected_lag}d lag): {expected_latest.date()}")
            
            if latest_date.date() > expected_latest.date():
                print("  ⚠️  WARNING: VIX data may not respect publication lag!")
            else:
                print("  ✅ VIX data respects publication lag")
        
        # Test Treasury (T+1)
        dgs10 = fetcher.fetch_fred_series(
            'DGS10',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if dgs10 is not None:
            expected_lag = PUBLICATION_LAGS.get('DGS10', 1)
            latest_date = dgs10.index[-1]
            expected_latest = end_date - timedelta(days=expected_lag)
            
            print(f"\n  DGS10 last date: {latest_date.date()}")
            print(f"  Expected (with {expected_lag}d lag): {expected_latest.date()}")
            
            if latest_date.date() > expected_latest.date():
                print("  ⚠️  WARNING: DGS10 data may not respect publication lag!")
            else:
                print("  ✅ DGS10 data respects publication lag")
                
    except Exception as e:
        print(f"  ⚠️  Could not verify data fetcher: {e}")
    
    # Check 2: Look for common leakage patterns in feature names
    print("\n[Check 2] Scanning for suspicious feature names...")
    try:
        from core.feature_engine import UnifiedFeatureEngine
        from core.data_fetcher import UnifiedDataFetcher
        
        # Build a small feature set to check names
        fetcher = UnifiedDataFetcher()
        engine = UnifiedFeatureEngine(fetcher)
        
        # We won't build full features (too slow), just check the method signatures
        import inspect
        
        suspicious_names = []
        for name, method in inspect.getmembers(engine, predicate=inspect.ismethod):
            if 'future' in name.lower() or 'forward' in name.lower():
                suspicious_names.append(name)
        
        if suspicious_names:
            print(f"  ⚠️  Found {len(suspicious_names)} methods with 'future/forward' in name:")
            for name in suspicious_names:
                print(f"     → {name}")
        else:
            print("  ✅ No suspicious method names found")
            
    except Exception as e:
        print(f"  ⚠️  Could not check feature names: {e}")
    
    # Check 3: Verify config completeness
    print("\n[Check 3] Checking publication lag coverage...")
    
    # List of data sources that should have lags
    critical_sources = [
        '^VIX', '^GSPC', 'SKEW', 'VIX3M', 'VXTLT',
        'DGS1MO', 'DGS3MO', 'DGS2', 'DGS10', 'DGS30',
        'DTWEXBGS', 'CL=F', 'DX-Y.NYB'
    ]
    
    missing_lags = [src for src in critical_sources if src not in PUBLICATION_LAGS]
    
    if missing_lags:
        print(f"  ⚠️  {len(missing_lags)} critical sources missing lag definitions:")
        for src in missing_lags:
            print(f"     → {src}")
    else:
        print(f"  ✅ All {len(critical_sources)} critical sources have lag definitions")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    audit = validator.audit_results
    if audit.get('issues'):
        critical = sum(1 for i in audit['issues'] if i['severity'] == 'CRITICAL')
        high = sum(1 for i in audit['issues'] if i['severity'] == 'HIGH')
        medium = sum(1 for i in audit['issues'] if i['severity'] == 'MEDIUM')
        
        print(f"\nCode Audit: {audit['total_issues']} issues found")
        print(f"  🔴 Critical: {critical}")
        print(f"  🟡 High: {high}")
        print(f"  🟢 Medium: {medium}")
        
        if critical > 0:
            print("\n❌ CRITICAL ISSUES MUST BE FIXED BEFORE TRAINING")
            print("   Review the issues above and fix forward-looking operations")
            return False
        elif high > 0:
            print("\n⚠️  HIGH PRIORITY ISSUES SHOULD BE REVIEWED")
            print("   Consider fixing before production use")
        else:
            print("\n✅ No critical issues - safe to proceed with caution")
    else:
        print("\n✅ Code audit passed - no suspicious patterns detected")
    
    print(f"\nPublication Lags: {len(PUBLICATION_LAGS)} sources configured")
    print(f"Validation Report: ./models/temporal_safety_report.json")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    if audit.get('total_issues', 0) == 0:
        print("1. ✅ System appears temporally safe")
        print("2. ➡️  Ready to implement Optuna optimization")
        print("3. 💡 Validation will run automatically during training")
    else:
        print("1. 📝 Review issues listed above")
        print("2. 🔧 Fix any critical/high priority issues")
        print("3. 🔄 Re-run this validation script")
        print("4. ➡️  Then proceed with optimization implementation")
    
    return audit.get('total_issues', 0) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)