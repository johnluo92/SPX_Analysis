"""
Quick diagnostic to verify treasury yield scaling is fixed.
Run this BEFORE running the full v3.4 model.
"""

import yfinance as yf
from datetime import datetime, timedelta

print("="*70)
print("TREASURY YIELD DIAGNOSTIC - v3.4 FIX VERIFICATION")
print("="*70)

# Fetch recent data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print(f"\n📊 Fetching last 30 days of treasury data...")

# Fetch 10Y
print("\n1️⃣  10-Year Treasury (^TNX):")
tnx = yf.download('^TNX', start=start_date, end=end_date, progress=False, auto_adjust=True)
if not tnx.empty:
    # Handle both Series and scalar cases
    if 'Close' in tnx.columns:
        close_series = tnx['Close']
    else:
        close_series = tnx
    
    raw_value = float(close_series.iloc[-1])
    fixed_value = raw_value / 10
    mean_value = float(close_series.mean())
    
    print(f"   Raw value from Yahoo: {raw_value:.2f}")
    print(f"   After /10 fix: {fixed_value:.2f}%")
    print(f"   Mean (30d): {(mean_value / 10):.2f}%")
    
    if raw_value > 10:
        print(f"   ✅ Confirmed: Needs /10 scaling (currently showing {raw_value:.1f} instead of {fixed_value:.2f}%)")
    else:
        print(f"   ⚠️  Unexpected: Raw value already in % range")

# Fetch 5Y
print("\n2️⃣  5-Year Treasury (^FVX):")
fvx = yf.download('^FVX', start=start_date, end=end_date, progress=False, auto_adjust=True)
if not fvx.empty:
    # Handle both Series and scalar cases
    if 'Close' in fvx.columns:
        close_series = fvx['Close']
    else:
        close_series = fvx
    
    raw_value = float(close_series.iloc[-1])
    fixed_value = raw_value / 10
    mean_value = float(close_series.mean())
    
    print(f"   Raw value from Yahoo: {raw_value:.2f}")
    print(f"   After /10 fix: {fixed_value:.2f}%")
    print(f"   Mean (30d): {(mean_value / 10):.2f}%")
    
    if raw_value > 10:
        print(f"   ✅ Confirmed: Needs /10 scaling")
    else:
        print(f"   ⚠️  Unexpected: Raw value already in % range")

# Reality check
print("\n" + "="*70)
print("REALITY CHECK (as of October 2025):")
print("="*70)
print("Expected ranges:")
print("  • 10Y Treasury: ~4.0% to 4.5%")
print("  • 5Y Treasury: ~3.8% to 4.3%")
print("  • Yield Curve Slope (10Y-5Y): ~0.1% to 0.3%")
print("\nIf your values are 10x these (40-45), the /10 fix is critical!")

print("\n" + "="*70)
print("✅ DIAGNOSTIC COMPLETE")
print("="*70)
print("\nNext steps:")
print("  1. Delete old macro cache: rm -rf .cache_sector_data/macro_*")
print("  2. Run v3.4 model: python v34.py")
print("  3. Watch for '(scaled /10)' in fetch output")
print("  4. Check 'TREASURY YIELD VALIDATION' section")