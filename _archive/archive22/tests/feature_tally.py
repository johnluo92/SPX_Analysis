"""
Complete Feature Tally - Fetched → Engineered → Exported
Analyzes the entire data pipeline from raw sources to final usage
"""

from collections import defaultdict
from data_lineage import FEATURE_LINEAGE, RAW_SOURCES, ANOMALY_FEATURE_GROUPS

print("\n" + "="*80)
print("COMPLETE FEATURE TALLY - DATA PIPELINE ANALYSIS")
print("="*80)

# ============================================================================
# PHASE 1: RAW DATA FETCHED
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: RAW DATA SOURCES (FETCHED)")
print("="*80)

yahoo_sources = []
fred_sources = []
cboe_sources = []

for name, info in RAW_SOURCES.items():
    if 'Yahoo' in info['source']:
        yahoo_sources.append(name)
    elif 'FRED' in info['source']:
        fred_sources.append(name)
    elif 'CBOE' in info['source']:
        cboe_sources.append(name)

print(f"\n📊 Yahoo Finance: {len(yahoo_sources)} series")
for src in yahoo_sources:
    print(f"   • {src}")

print(f"\n📈 FRED: {len(fred_sources)} series")
for src in fred_sources:
    print(f"   • {src}")

print(f"\n📉 CBOE: {len(cboe_sources)} indicators")
for src in cboe_sources:
    print(f"   • {src}")

total_fetched = len(yahoo_sources) + len(fred_sources) + len(cboe_sources)
print(f"\n✅ TOTAL RAW SOURCES FETCHED: {total_fetched}")

# ============================================================================
# PHASE 2: FEATURES ENGINEERED
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: FEATURES ENGINEERED")
print("="*80)

# Group by feature module method
feature_groups = defaultdict(list)
for name, lineage in FEATURE_LINEAGE.items():
    method = lineage.feature_method
    feature_groups[method].append(name)

print(f"\n🔧 UnifiedFeatureEngine Methods:")
for method in sorted(feature_groups.keys()):
    count = len(feature_groups[method])
    print(f"   • {method}: {count} features")

total_engineered = len(FEATURE_LINEAGE)
print(f"\n✅ TOTAL FEATURES ENGINEERED: {total_engineered}")

# Break down by source type
print(f"\n📊 Breakdown by Raw Source:")
yahoo_engineered = [name for name, lineage in FEATURE_LINEAGE.items() 
                    if 'Yahoo' in lineage.raw_source]
fred_engineered = [name for name, lineage in FEATURE_LINEAGE.items() 
                   if 'FRED' in lineage.raw_source]
cboe_engineered = [name for name, lineage in FEATURE_LINEAGE.items() 
                   if 'CBOE' in lineage.raw_source]
computed = [name for name, lineage in FEATURE_LINEAGE.items() 
            if lineage.raw_source == 'Computed']

print(f"   • From Yahoo Finance: {len(yahoo_engineered)} features")
print(f"   • From FRED: {len(fred_engineered)} features")
print(f"   • From CBOE: {len(cboe_engineered)} features")
print(f"   • Computed (derived): {len(computed)} features")

# ============================================================================
# PHASE 3: FEATURES EXPORTED/USED
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: FEATURES EXPORTED/USED")
print("="*80)

# Features in dashboard exports
dashboard_features = []
anomaly_only_features = []
ml_only_features = []

for name, lineage in FEATURE_LINEAGE.items():
    json_locs = lineage.json_location
    
    if any('dashboard_data.json' in loc or 'market_state.json' in loc 
           for loc in json_locs):
        dashboard_features.append(name)
    elif any('anomaly' in loc.lower() for loc in json_locs):
        anomaly_only_features.append(name)
    elif any('N/A' in loc for loc in json_locs):
        ml_only_features.append(name)

print(f"\n📱 Dashboard Display Features: {len(dashboard_features)}")
print(f"   (Exported to dashboard_data.json or market_state.json)")
for feat in dashboard_features[:10]:
    print(f"   • {feat}")
if len(dashboard_features) > 10:
    print(f"   ... and {len(dashboard_features) - 10} more")

print(f"\n🔍 Anomaly Detection Features: {len(anomaly_only_features)}")
print(f"   (Used in anomaly detection, exported to attribution JSON)")

print(f"\n🤖 ML-Only Features: {len(ml_only_features)}")
print(f"   (Used internally for predictions, not exported)")

# Features used in anomaly domains
print(f"\n🎯 Anomaly Domain Coverage:")
total_domain_features = set()
for domain, features in ANOMALY_FEATURE_GROUPS.items():
    total_domain_features.update(features)
    print(f"   • {domain}: {len(features)} features")

print(f"\n✅ UNIQUE FEATURES IN ANOMALY DOMAINS: {len(total_domain_features)}")

# Check coverage
all_engineered_names = set(FEATURE_LINEAGE.keys())
domain_coverage = len(total_domain_features & all_engineered_names)
print(f"   Coverage: {domain_coverage}/{len(total_domain_features)} features exist in lineage")

missing_from_lineage = total_domain_features - all_engineered_names
if missing_from_lineage:
    print(f"\n⚠️  Features in domains but missing from lineage: {len(missing_from_lineage)}")
    for feat in sorted(missing_from_lineage)[:10]:
        print(f"   • {feat}")
    if len(missing_from_lineage) > 10:
        print(f"   ... and {len(missing_from_lineage) - 10} more")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPLETE PIPELINE SUMMARY")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│ DATA PIPELINE: FETCH → ENGINEER → EXPORT                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 1: FETCH RAW DATA                                    │
│  ├─ Yahoo Finance:     {len(yahoo_sources):3d} series                         │
│  ├─ FRED:              {len(fred_sources):3d} series                         │
│  ├─ CBOE:              {len(cboe_sources):3d} indicators                     │
│  └─ TOTAL FETCHED:     {total_fetched:3d} raw sources                    │
│                                                              │
│  PHASE 2: ENGINEER FEATURES                                 │
│  ├─ From Yahoo:        {len(yahoo_engineered):3d} features                      │
│  ├─ From FRED:         {len(fred_engineered):3d} features                      │
│  ├─ From CBOE:         {len(cboe_engineered):3d} features                      │
│  ├─ Computed:          {len(computed):3d} features                      │
│  └─ TOTAL ENGINEERED:  {total_engineered:3d} features                      │
│                                                              │
│  PHASE 3: EXPORT/USE                                        │
│  ├─ Dashboard:         {len(dashboard_features):3d} features                      │
│  ├─ Anomaly Only:      {len(anomaly_only_features):3d} features                      │
│  ├─ ML Internal:       {len(ml_only_features):3d} features                      │
│  └─ Anomaly Domains:   {len(total_domain_features):3d} unique features              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")

# Efficiency metrics
amplification = total_engineered / total_fetched if total_fetched > 0 else 0
export_rate = (len(dashboard_features) + len(anomaly_only_features)) / total_engineered if total_engineered > 0 else 0

print(f"📊 EFFICIENCY METRICS:")
print(f"   • Feature Amplification: {amplification:.1f}x")
print(f"     (Created {amplification:.1f} features per raw data source)")
print(f"   • Export Rate: {export_rate:.1%}")
print(f"     (Exported {export_rate:.1%} of engineered features)")

print("\n" + "="*80)

# ============================================================================
# DETAILED BREAKDOWN BY CATEGORY
# ============================================================================

print("\n" + "="*80)
print("DETAILED BREAKDOWN BY FEATURE CATEGORY")
print("="*80)

categories = {
    'VIX Mean Reversion': [f for f in FEATURE_LINEAGE if f.startswith('vix_vs_ma') or 
                           f.startswith('vix_zscore') or f.startswith('vix_percentile') or
                           f == 'reversion_strength_63d'],
    'VIX Dynamics': [f for f in FEATURE_LINEAGE if f.startswith('vix_velocity') or 
                     f.startswith('vix_accel') or f.startswith('vix_vol') or 
                     f.startswith('vix_momentum') or f == 'vix_term_structure'],
    'VIX Regime': [f for f in FEATURE_LINEAGE if f.startswith('vix_regime') or 
                   f in ['days_in_regime', 'vix_displacement', 'days_since_crisis', 'elevated_flag']],
    'SPX Price Action': [f for f in FEATURE_LINEAGE if f.startswith('spx_') and 
                         ('ret' in f or 'lag' in f or 'vs_ma' in f or 'momentum' in f or 
                          'realized_vol' in f or 'skew' in f or 'kurt' in f or 'trend' in f or 'vol_ratio' in f)],
    'SPX Technical': [f for f in FEATURE_LINEAGE if f in ['ma20_vs_ma50', 'bb_width_20d', 'rsi_14']],
    'VIX vs Realized Vol': [f for f in FEATURE_LINEAGE if 'vix_vs_rv' in f or 'vix_rv_ratio' in f or f == 'vix_vs_avg_rv'],
    'SPX-VIX Correlation': [f for f in FEATURE_LINEAGE if 'spx_vix_corr' in f],
    'Calendar': [f for f in FEATURE_LINEAGE if f in ['month', 'quarter', 'day_of_week', 'day_of_month', 'is_opex_week']],
    'CBOE Options': [f for f in FEATURE_LINEAGE if any(f.startswith(x) for x in 
                     ['SKEW', 'PCCI', 'PCCE', 'PCC', 'COR1M', 'COR3M', 'VXTH']) or 
                     f in ['pc_divergence', 'tail_risk_elevated', 'cor_term_structure']],
    'Treasury Rates': [f for f in FEATURE_LINEAGE if 'Treasury' in f or 'Yield_Curve' in f or 
                       'Real_Yield' in f or 'Inflation_Expectation' in f or 'High_Yield_Spread' in f],
    'Commodities': [f for f in FEATURE_LINEAGE if any(f.startswith(x) for x in 
                    ['Gold_', 'Silver_', 'Crude Oil_', 'Brent Crude_', 'Natural Gas_', 
                     'Dollar_', 'Producer_Price_Index_'])]
}

for category, features in categories.items():
    print(f"\n📊 {category}: {len(features)} features")
    if features:
        # Show first 5
        for feat in sorted(features)[:5]:
            print(f"   • {feat}")
        if len(features) > 5:
            print(f"   ... and {len(features) - 5} more")

# Verify totals
categorized_count = sum(len(f) for f in categories.values())
print(f"\n✅ Total Categorized: {categorized_count}/{total_engineered} features")

if categorized_count < total_engineered:
    uncategorized = set(FEATURE_LINEAGE.keys()) - set(f for feats in categories.values() for f in feats)
    print(f"\n⚠️  Uncategorized features: {len(uncategorized)}")
    for feat in sorted(uncategorized)[:10]:
        print(f"   • {feat}")

print("\n" + "="*80)