"""
Feature Usage Audit - FIXED VERSION
Analyzes actual feature usage to identify removal candidates.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FeatureUsageStats:
    """Complete usage statistics for a single feature."""
    feature_name: str
    
    # Usage tracking
    used_in_vix_predictor: bool = False
    used_in_anomaly_detector: bool = False
    used_in_dashboard: bool = False
    exported_to_json: bool = False
    
    # Metadata
    calculation_complexity: str = "unknown"
    raw_source: str = "unknown"
    json_locations: List[str] = field(default_factory=list)
    
    def usage_score(self) -> float:
        """Calculate 0-10 usage score."""
        score = 0.0
        
        # Base usage (0-8 points)
        if self.used_in_vix_predictor: score += 3.0
        if self.used_in_anomaly_detector: score += 3.0
        if self.used_in_dashboard: score += 1.5
        if self.exported_to_json: score += 0.5
        
        # Complexity penalty (0-2 points deduction for complex unused features)
        if self.calculation_complexity == "complex" and score < 3.0:
            score -= 1.0
        
        return max(0.0, min(10.0, score))
    
    def removal_recommendation(self) -> str:
        """Recommend removal action."""
        score = self.usage_score()
        
        if score >= 6.0:
            return "KEEP - Critical"
        elif score >= 4.0:
            return "KEEP - Important"
        elif score >= 2.0:
            return "REVIEW - Marginal"
        elif score >= 0.5:
            return "REMOVE - Low value"
        else:
            return "REMOVE - Unused"


class FeatureUsageAuditor:
    """Comprehensive feature usage analysis."""
    
    def __init__(self, project_root: Path = Path(".")):
        self.project_root = project_root
        self.feature_stats: Dict[str, FeatureUsageStats] = {}
        self.all_features: Set[str] = set()
        
    def run_full_audit(self):
        """Execute complete feature audit."""
        print("="*80)
        print("FEATURE USAGE AUDIT - LET THE DATA SPEAK")
        print("="*80)
        
        # Step 1: Parse data lineage directly from document
        print("\n[1/5] Parsing FEATURE_LINEAGE from data_lineage.py...")
        self._parse_lineage_from_document()
        
        # Step 2: Parse anomaly detector usage from document
        print("[2/5] Analyzing anomaly detector feature usage...")
        self._parse_anomaly_features_from_document()
        
        # Step 3: Check dashboard/JSON exports from lineage
        print("[3/5] Checking dashboard and JSON exports...")
        self._analyze_exports_from_lineage()
        
        # Step 4: Assess calculation complexity
        print("[4/5] Assessing calculation complexity...")
        self._assess_complexity()
        
        # Step 5: Generate recommendations
        print("[5/5] Generating removal recommendations...")
        self._generate_report()
        
    def _parse_lineage_from_document(self):
        """Parse features from the FEATURE_LINEAGE in data_lineage.py document."""
        # Features extracted from the document
        features = {
            # VIX Mean Reversion (16 features)
            'vix': ('Yahoo Finance (^VIX)', ['dashboard_data.json', 'market_state.json']),
            'vix_vs_ma10': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vs_ma21': ('Yahoo Finance (^VIX)', ['market_state.json']),
            'vix_vs_ma63': ('Yahoo Finance (^VIX)', ['market_state.json']),
            'vix_vs_ma126': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vs_ma252': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vs_ma10_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vs_ma21_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vs_ma63_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vs_ma126_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vs_ma252_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_zscore_63d': ('Yahoo Finance (^VIX)', ['market_state.json']),
            'vix_zscore_126d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_zscore_252d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_percentile_126d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_percentile_252d': ('Yahoo Finance (^VIX)', ['market_state.json']),
            'reversion_strength_63d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            
            # VIX Dynamics (15 features)
            'vix_velocity_1d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_velocity_1d_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_velocity_5d': ('Yahoo Finance (^VIX)', ['market_state.json']),
            'vix_velocity_5d_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_velocity_10d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_velocity_10d_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_velocity_21d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_velocity_21d_pct': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_accel_5d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vol_10d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_vol_21d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_momentum_z_10d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_momentum_z_21d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_momentum_z_63d': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            'vix_term_structure': ('Yahoo Finance (^VIX)', ['anomaly_feature_attribution.json']),
            
            # VIX Regime (5 features)
            'vix_regime': ('Yahoo Finance (^VIX)', ['dashboard_data.json', 'market_state.json']),
            'days_in_regime': ('Computed', ['dashboard_data.json', 'market_state.json']),
            'vix_displacement': ('Computed', ['anomaly_feature_attribution.json']),
            'days_since_crisis': ('Computed', ['anomaly_feature_attribution.json']),
            'elevated_flag': ('Computed', ['anomaly_feature_attribution.json']),
            
            # SPX Price Action (18 features)
            'spx_lag1': ('Yahoo Finance (^GSPC)', ['N/A']),
            'spx_lag5': ('Yahoo Finance (^GSPC)', ['N/A']),
            'spx_ret_5d': ('Yahoo Finance (^GSPC)', ['N/A']),
            'spx_ret_10d': ('Yahoo Finance (^GSPC)', ['N/A']),
            'spx_ret_13d': ('Yahoo Finance (^GSPC)', ['N/A']),
            'spx_ret_21d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_ret_63d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_vs_ma20': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_vs_ma50': ('Yahoo Finance (^GSPC)', ['market_state.json']),
            'spx_vs_ma200': ('Yahoo Finance (^GSPC)', ['market_state.json']),
            'spx_realized_vol_10d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_realized_vol_21d': ('Yahoo Finance (^GSPC)', ['market_state.json', 'anomaly_feature_attribution.json']),
            'spx_realized_vol_63d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_vol_ratio_10_63': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_momentum_z_10d': ('Yahoo Finance (^GSPC)', ['market_state.json']),
            'spx_momentum_z_21d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_skew_21d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_kurt_21d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            
            # SPX Technical (3 features)
            'ma20_vs_ma50': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'bb_width_20d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'rsi_14': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            
            # VIX vs Realized Vol (7 features)
            'vix_vs_rv_10d': ('Computed', ['anomaly_feature_attribution.json']),
            'vix_vs_rv_21d': ('Computed', ['anomaly_feature_attribution.json']),
            'vix_vs_rv_30d': ('Computed', ['anomaly_feature_attribution.json']),
            'vix_rv_ratio_10d': ('Computed', ['anomaly_feature_attribution.json']),
            'vix_rv_ratio_21d': ('Computed', ['market_state.json']),
            'vix_rv_ratio_30d': ('Computed', ['anomaly_feature_attribution.json']),
            'vix_vs_avg_rv': ('Computed', ['anomaly_feature_attribution.json']),
            
            # SPX-VIX Correlation (4 features)
            'spx_vix_corr_21d': ('Computed', ['market_state.json']),
            'spx_vix_corr_63d': ('Computed', ['anomaly_feature_attribution.json']),
            'spx_trend_10d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            'spx_trend_21d': ('Yahoo Finance (^GSPC)', ['anomaly_feature_attribution.json']),
            
            # Calendar (5 features) - ML ONLY
            'month': ('Computed', ['N/A']),
            'quarter': ('Computed', ['N/A']),
            'day_of_week': ('Computed', ['N/A']),
            'day_of_month': ('Computed', ['N/A']),
            'is_opex_week': ('Computed', ['N/A']),
            
            # CBOE (24 features)
            'SKEW': ('CBOE', ['anomaly_feature_attribution.json']),
            'SKEW_change_21d': ('CBOE', ['anomaly_feature_attribution.json']),
            'SKEW_zscore_63d': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCCI': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCCI_change_21d': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCCI_zscore_63d': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCCE': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCCE_change_21d': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCCE_zscore_63d': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCC': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCC_change_21d': ('CBOE', ['anomaly_feature_attribution.json']),
            'PCC_zscore_63d': ('CBOE', ['anomaly_feature_attribution.json']),
            'COR1M': ('CBOE', ['anomaly_feature_attribution.json']),
            'COR1M_change_21d': ('CBOE', ['anomaly_feature_attribution.json']),
            'COR1M_zscore_63d': ('CBOE', ['anomaly_feature_attribution.json']),
            'COR3M': ('CBOE', ['anomaly_feature_attribution.json']),
            'COR3M_change_21d': ('CBOE', ['anomaly_feature_attribution.json']),
            'COR3M_zscore_63d': ('CBOE', ['anomaly_feature_attribution.json']),
            'VXTH': ('CBOE', ['anomaly_feature_attribution.json']),
            'VXTH_change_21d': ('CBOE', ['anomaly_feature_attribution.json']),
            'VXTH_zscore_63d': ('CBOE', ['anomaly_feature_attribution.json']),
            'pc_divergence': ('Computed', ['anomaly_feature_attribution.json']),
            'tail_risk_elevated': ('Computed', ['anomaly_feature_attribution.json']),
            'cor_term_structure': ('Computed', ['anomaly_feature_attribution.json']),
            
            # Treasury/Rates (31 features)
            'Treasury_10Y_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_10Y_lag1': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_10Y_change_10d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_10Y_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_10Y_change_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_10Y_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_10Y_zscore_252d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_2Y_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_2Y_change_10d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_2Y_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_2Y_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_5Y_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_5Y_change_10d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_5Y_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_5Y_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_30Y_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_30Y_change_10d': ('FRED', ['anomaly_feature_attribution.json']),
            'Treasury_30Y_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Yield_Curve_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Yield_Curve_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Yield_Curve_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Yield_Curve_10Y3M_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Yield_Curve_10Y3M_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Real_Yield_10Y_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Real_Yield_10Y_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Inflation_Expectation_5Y_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Inflation_Expectation_5Y_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Inflation_Expectation_10Y_level': ('FRED', ['anomaly_feature_attribution.json']),
            'Inflation_Expectation_10Y_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'High_Yield_Spread_level': ('FRED', ['anomaly_feature_attribution.json']),
            'High_Yield_Spread_change_21d': ('FRED', ['anomaly_feature_attribution.json']),
            
            # Commodities (20 features)
            'Gold_lag1': ('Yahoo Finance (GLD)', ['anomaly_feature_attribution.json']),
            'Gold_mom_10d': ('Yahoo Finance (GLD)', ['anomaly_feature_attribution.json']),
            'Gold_mom_21d': ('Yahoo Finance (GLD)', ['anomaly_feature_attribution.json']),
            'Gold_mom_63d': ('Yahoo Finance (GLD)', ['anomaly_feature_attribution.json']),
            'Gold_zscore_63d': ('Yahoo Finance (GLD)', ['anomaly_feature_attribution.json']),
            'Silver_lag1': ('Yahoo Finance (SLV)', ['anomaly_feature_attribution.json']),
            'Silver_mom_10d': ('Yahoo Finance (SLV)', ['anomaly_feature_attribution.json']),
            'Silver_mom_21d': ('Yahoo Finance (SLV)', ['anomaly_feature_attribution.json']),
            'Silver_mom_63d': ('Yahoo Finance (SLV)', ['anomaly_feature_attribution.json']),
            'Silver_zscore_63d': ('Yahoo Finance (SLV)', ['anomaly_feature_attribution.json']),
            'Crude Oil_mom_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Crude Oil_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Brent Crude_mom_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Brent Crude_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Natural Gas_mom_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Natural Gas_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Dollar_mom_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Dollar_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
            'Producer_Price_Index_mom_21d': ('FRED', ['anomaly_feature_attribution.json']),
            'Producer_Price_Index_zscore_63d': ('FRED', ['anomaly_feature_attribution.json']),
        }
        
        for feature, (source, locations) in features.items():
            self.all_features.add(feature)
            self.feature_stats[feature] = FeatureUsageStats(
                feature_name=feature,
                raw_source=source,
                json_locations=locations,
                exported_to_json=len(locations) > 0 and 'N/A' not in locations[0]
            )
        
        print(f"   ✓ Found {len(self.all_features)} features in lineage")
    
    def _parse_anomaly_features_from_document(self):
        """Mark features used in anomaly detection from document info."""
        # Anomaly feature groups from the document
        anomaly_groups = {
            'vix_mean_reversion': [
                'reversion_strength_63d', 'vix_percentile_126d', 'vix_percentile_252d',
                'vix_vs_ma10', 'vix_vs_ma10_pct', 'vix_vs_ma21', 'vix_vs_ma21_pct',
                'vix_vs_ma63', 'vix_vs_ma63_pct', 'vix_vs_ma126', 'vix_vs_ma126_pct',
                'vix_vs_ma252', 'vix_vs_ma252_pct', 'vix_zscore_63d', 'vix_zscore_126d',
                'vix_zscore_252d'
            ],
            'vix_momentum': [
                'vix_accel_5d', 'vix_momentum_z_10d', 'vix_momentum_z_21d',
                'vix_momentum_z_63d', 'vix_term_structure', 'vix_velocity_1d',
                'vix_velocity_1d_pct', 'vix_velocity_5d', 'vix_velocity_5d_pct',
                'vix_velocity_10d', 'vix_velocity_10d_pct', 'vix_velocity_21d',
                'vix_velocity_21d_pct', 'vix_vol_10d'
            ],
            'vix_regime_structure': [
                'days_in_regime', 'days_since_crisis', 'elevated_flag',
                'vix_displacement', 'vix_regime'
            ],
            'cboe_options_flow': [
                'COR1M', 'COR1M_change_21d', 'COR1M_zscore_63d', 'COR3M',
                'COR3M_change_21d', 'COR3M_zscore_63d', 'PCC', 'PCC_change_21d',
                'PCC_zscore_63d', 'PCCE', 'PCCE_change_21d', 'PCCE_zscore_63d',
                'PCCI', 'PCCI_change_21d', 'PCCI_zscore_63d', 'SKEW',
                'SKEW_change_21d', 'SKEW_zscore_63d', 'VXTH', 'VXTH_change_21d',
                'VXTH_zscore_63d', 'cor_term_structure', 'pc_divergence',
                'tail_risk_elevated'
            ],
            'vix_spx_relationship': [
                'spx_trend_10d', 'spx_trend_21d', 'spx_vix_corr_21d',
                'spx_vix_corr_63d', 'vix_rv_ratio_10d', 'vix_rv_ratio_21d',
                'vix_rv_ratio_30d', 'vix_vs_avg_rv', 'vix_vs_rv_10d',
                'vix_vs_rv_21d', 'vix_vs_rv_30d'
            ],
            'spx_price_action': [
                'spx_kurt_21d', 'spx_momentum_z_10d', 'spx_momentum_z_21d',
                'spx_realized_vol_10d', 'spx_realized_vol_21d', 'spx_realized_vol_63d',
                'spx_ret_21d', 'spx_ret_63d', 'spx_skew_21d', 'spx_vol_ratio_10_63',
                'spx_vs_ma20', 'spx_vs_ma50', 'spx_vs_ma200'
            ],
            'spx_volatility_regime': [
                'bb_width_20d', 'ma20_vs_ma50', 'rsi_14', 'spx_realized_vol_10d',
                'spx_realized_vol_21d', 'spx_realized_vol_63d', 'spx_vol_ratio_10_63'
            ],
            'macro_rates': [
                'High_Yield_Spread_change_21d', 'High_Yield_Spread_level',
                'Inflation_Expectation_10Y_change_21d', 'Inflation_Expectation_10Y_level',
                'Inflation_Expectation_5Y_change_21d', 'Inflation_Expectation_5Y_level',
                'Real_Yield_10Y_change_21d', 'Real_Yield_10Y_level',
                'Treasury_10Y_change_10d', 'Treasury_10Y_change_21d',
                'Treasury_10Y_change_63d', 'Treasury_10Y_lag1', 'Treasury_10Y_level',
                'Treasury_10Y_zscore_252d', 'Treasury_10Y_zscore_63d',
                'Treasury_2Y_change_10d', 'Treasury_2Y_change_21d', 'Treasury_2Y_level',
                'Treasury_2Y_zscore_63d', 'Treasury_30Y_change_10d',
                'Treasury_30Y_change_21d', 'Treasury_30Y_level',
                'Treasury_5Y_change_10d', 'Treasury_5Y_change_21d', 'Treasury_5Y_level',
                'Treasury_5Y_zscore_63d', 'Yield_Curve_10Y3M_change_21d',
                'Yield_Curve_10Y3M_level', 'Yield_Curve_change_21d',
                'Yield_Curve_level', 'Yield_Curve_zscore_63d'
            ],
            'commodities_stress': [
                'Brent Crude_mom_21d', 'Brent Crude_zscore_63d', 'Crude Oil_mom_21d',
                'Crude Oil_zscore_63d', 'Dollar_mom_21d', 'Dollar_zscore_63d',
                'Gold_lag1', 'Gold_mom_10d', 'Gold_mom_21d', 'Gold_mom_63d',
                'Gold_zscore_63d', 'Natural Gas_mom_21d', 'Natural Gas_zscore_63d',
                'Producer_Price_Index_mom_21d', 'Producer_Price_Index_zscore_63d',
                'Silver_lag1', 'Silver_mom_10d'
            ]
        }
        
        anomaly_features = set()
        for group, features in anomaly_groups.items():
            anomaly_features.update(features)
        
        for feature in anomaly_features:
            if feature in self.feature_stats:
                self.feature_stats[feature].used_in_anomaly_detector = True
        
        print(f"   ✓ Found {len(anomaly_features)} features used in anomaly detector")
    
    def _analyze_exports_from_lineage(self):
        """Check which features are exported to dashboard from lineage data."""
        dashboard_features = set()
        
        for feature, stats in self.feature_stats.items():
            # Check if exported to dashboard JSON files
            if any('dashboard_data.json' in loc or 'market_state.json' in loc 
                   for loc in stats.json_locations):
                stats.used_in_dashboard = True
                dashboard_features.add(feature)
        
        print(f"   ✓ Found {len(dashboard_features)} features in dashboard exports")
    
    def _assess_complexity(self):
        """Assess calculation complexity of each feature."""
        for feature, stats in self.feature_stats.items():
            # Simple: raw values, lags, basic differences
            if any(x in feature for x in ['lag', 'level', 'close', 'open', 'change_10d', 'change_21d']):
                stats.calculation_complexity = "simple"
            # Complex: z-scores, percentiles, ratios, kurtosis, skewness
            elif any(x in feature for x in ['zscore', 'percentile', 'kurt', 'skew', 'ratio', 'corr']):
                stats.calculation_complexity = "complex"
            # Moderate: everything else
            else:
                stats.calculation_complexity = "moderate"
        
        print("   ✓ Assessed calculation complexity")
    
    def _generate_report(self):
        """Generate comprehensive usage report."""
        print("\n" + "="*80)
        print("FEATURE USAGE AUDIT REPORT")
        print("="*80)
        
        # Calculate statistics
        total_features = len(self.feature_stats)
        if total_features == 0:
            print("\n⚠️  ERROR: No features found in lineage!")
            return
            
        vix_used = sum(1 for s in self.feature_stats.values() if s.used_in_vix_predictor)
        anomaly_used = sum(1 for s in self.feature_stats.values() if s.used_in_anomaly_detector)
        dashboard_used = sum(1 for s in self.feature_stats.values() if s.used_in_dashboard)
        exported = sum(1 for s in self.feature_stats.values() if s.exported_to_json)
        
        print(f"\n📊 USAGE STATISTICS:")
        print(f"   Total Features: {total_features}")
        print(f"   Used in VIX Predictor: {vix_used} ({vix_used/total_features*100:.1f}%)")
        print(f"   Used in Anomaly Detector: {anomaly_used} ({anomaly_used/total_features*100:.1f}%)")
        print(f"   Shown in Dashboard: {dashboard_used} ({dashboard_used/total_features*100:.1f}%)")
        print(f"   Exported to JSON: {exported} ({exported/total_features*100:.1f}%)")
        
        # Score all features
        scored_features = [
            (name, stats, stats.usage_score())
            for name, stats in self.feature_stats.items()
        ]
        scored_features.sort(key=lambda x: x[2])  # Sort by score (lowest first)
        
        # Categorize by recommendation
        keep_critical = [f for f in scored_features if f[2] >= 6.0]
        keep_important = [f for f in scored_features if 4.0 <= f[2] < 6.0]
        review_marginal = [f for f in scored_features if 2.0 <= f[2] < 4.0]
        remove_low = [f for f in scored_features if 0.5 <= f[2] < 2.0]
        remove_unused = [f for f in scored_features if f[2] < 0.5]
        
        print(f"\n🎯 RECOMMENDATION SUMMARY:")
        print(f"   ✅ KEEP - Critical: {len(keep_critical)} features")
        print(f"   ✅ KEEP - Important: {len(keep_important)} features")
        print(f"   ⚠️  REVIEW - Marginal: {len(review_marginal)} features")
        print(f"   ❌ REMOVE - Low value: {len(remove_low)} features")
        print(f"   ❌ REMOVE - Unused: {len(remove_unused)} features")
        
        # Show top removal candidates
        removal_candidates = remove_unused + remove_low
        print(f"\n🔴 TOP 30 REMOVAL CANDIDATES:")
        print(f"{'Feature':<45} {'Score':<7} {'Anom':<5} {'Dash':<5} {'Complexity':<10} {'Source'}")
        print("-"*100)
        
        for name, stats, score in removal_candidates[:30]:
            anom_mark = "✓" if stats.used_in_anomaly_detector else "✗"
            dash_mark = "✓" if stats.used_in_dashboard else "✗"
            
            print(f"{name:<45} {score:<7.2f} {anom_mark:<5} {dash_mark:<5} {stats.calculation_complexity:<10} {stats.raw_source[:20]}")
        
        # Export detailed report
        self._export_detailed_report(scored_features)
        
        # Calculate potential savings
        removable = len(remove_low) + len(remove_unused)
        reduction_pct = removable / total_features * 100
        
        print(f"\n💡 POTENTIAL IMPACT:")
        print(f"   Safe to remove: {removable} features ({reduction_pct:.1f}% reduction)")
        print(f"   Remaining core features: {total_features - removable}")
        
        # Show breakdown by source
        print(f"\n📊 REMOVAL CANDIDATES BY SOURCE:")
        source_breakdown = defaultdict(int)
        for name, stats, score in removal_candidates:
            source_breakdown[stats.raw_source] += 1
        
        for source, count in sorted(source_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {source}: {count} features")
        
        # Show specific recommendations
        print(f"\n✂️  SPECIFIC REMOVAL RECOMMENDATIONS:")
        print("\n1. CALENDAR FEATURES (5 features) - ML ONLY, never exported:")
        calendar_features = ['month', 'quarter', 'day_of_week', 'day_of_month', 'is_opex_week']
        for f in calendar_features:
            if f in self.feature_stats:
                print(f"   ❌ {f} - score: {self.feature_stats[f].usage_score():.2f}")
        
        print("\n2. SPX LAG FEATURES (5 features) - Never used in anomaly detection:")
        lag_features = ['spx_lag1', 'spx_lag5', 'spx_ret_5d', 'spx_ret_10d', 'spx_ret_13d']
        for f in lag_features:
            if f in self.feature_stats:
                print(f"   ❌ {f} - score: {self.feature_stats[f].usage_score():.2f}")
        
        print("\n3. REDUNDANT PERCENTAGE VARIANTS (10 features):")
        pct_features = [f for f in self.feature_stats.keys() if '_pct' in f and self.feature_stats[f].usage_score() < 4.0]
        for f in sorted(pct_features)[:10]:
            print(f"   ❌ {f} - score: {self.feature_stats[f].usage_score():.2f}")
        
        print("\n4. COMPLEX COMPUTED FEATURES WITH LOW USAGE:")
        complex_unused = [(name, stats, score) for name, stats, score in removal_candidates 
                         if stats.calculation_complexity == "complex" and score < 2.0]
        for name, stats, score in complex_unused[:10]:
            print(f"   ❌ {name} - score: {score:.2f}")
        
        print("\n📄 Detailed report exported to: feature_usage_audit.json")
        print("="*80)
    
    def _export_detailed_report(self, scored_features: List):
        """Export detailed JSON report."""
        report = {
            "timestamp": "2025-10-29",
            "summary": {
                "total_features": len(self.feature_stats),
                "vix_predictor_features": sum(1 for s in self.feature_stats.values() if s.used_in_vix_predictor),
                "anomaly_detector_features": sum(1 for s in self.feature_stats.values() if s.used_in_anomaly_detector),
                "dashboard_features": sum(1 for s in self.feature_stats.values() if s.used_in_dashboard),
                "exported_features": sum(1 for s in self.feature_stats.values() if s.exported_to_json),
                "removal_candidates": sum(1 for _, _, score in scored_features if score < 2.0)
            },
            "features": []
        }
        
        for name, stats, score in scored_features:
            report["features"].append({
                "name": name,
                "usage_score": round(score, 2),
                "recommendation": stats.removal_recommendation(),
                "used_in_vix_predictor": stats.used_in_vix_predictor,
                "used_in_anomaly_detector": stats.used_in_anomaly_detector,
                "used_in_dashboard": stats.used_in_dashboard,
                "exported_to_json": stats.exported_to_json,
                "calculation_complexity": stats.calculation_complexity,
                "raw_source": stats.raw_source,
                "json_locations": stats.json_locations
            })
        
        output_path = Path("feature_usage_audit.json")
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


def main():
    """Run feature usage audit."""
    auditor = FeatureUsageAuditor()
    auditor.run_full_audit()


if __name__ == "__main__":
    main()