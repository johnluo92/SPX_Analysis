"""
Enhanced Data Contract Validator - Comprehensive Validation
Validates every field, nested structure, type, and value range.

Key additions over original validator:
- Deep nested field validation
- NaN/Inf detection for all floats
- Value range constraints
- Enum validation
- Cross-file consistency checks
- Feature attribution completeness
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime


class ComprehensiveValidator:
    """Deep validation of all JSON outputs."""
    
    def __init__(self, json_dir: str = None):
        # Auto-detect json_data directory relative to script location
        if json_dir is None:
            script_dir = Path(__file__).parent
            # If in 'about tools' folder, go up one level then into json_data
            if script_dir.name == 'about tools':
                json_dir = script_dir.parent / 'json_data'
            else:
                json_dir = script_dir / 'json_data'
        
        self.json_dir = Path(json_dir)
        self.errors = []
        self.warnings = []
        self.passed = []
        
        if not self.json_dir.exists():
            print(f"⚠️  Warning: JSON directory not found at {self.json_dir}")
            print(f"   Script location: {Path(__file__).parent}")
        
    def validate_value(self, value: Any, expected_type: str, 
                      field_path: str, constraints: Optional[Dict] = None) -> bool:
        """Validate a single value against type and constraints."""
        
        # Type validation
        type_map = {
            'string': str,
            'float': (int, float),
            'int': int,
            'bool': bool,
            'array': list,
            'object': dict,
            'ISO timestamp': str
        }
        
        if expected_type not in type_map:
            self.warnings.append(f"⚠️  {field_path}: Unknown type '{expected_type}'")
            return True
        
        expected_python_type = type_map[expected_type]
        
        if value is None:
            self.warnings.append(f"⚠️  {field_path}: Value is None")
            return True
        
        if not isinstance(value, expected_python_type):
            self.errors.append(
                f"❌ {field_path}: Wrong type. Expected {expected_type}, "
                f"got {type(value).__name__}"
            )
            return False
        
        # NaN/Inf check for floats
        if expected_type == 'float' and isinstance(value, (int, float)):
            if math.isnan(value):
                self.errors.append(f"❌ {field_path}: Value is NaN")
                return False
            if math.isinf(value):
                self.errors.append(f"❌ {field_path}: Value is Inf")
                return False
        
        # Constraint validation
        if constraints:
            if 'range' in constraints:
                min_val, max_val = constraints['range']
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        self.errors.append(
                            f"❌ {field_path}: Value {value} out of range "
                            f"[{min_val}, {max_val}]"
                        )
                        return False
            
            if 'enum' in constraints:
                if value not in constraints['enum']:
                    self.errors.append(
                        f"❌ {field_path}: Value '{value}' not in allowed values "
                        f"{constraints['enum']}"
                    )
                    return False
            
            if 'min_length' in constraints:
                if isinstance(value, (str, list)):
                    if len(value) < constraints['min_length']:
                        self.errors.append(
                            f"❌ {field_path}: Length {len(value)} below minimum "
                            f"{constraints['min_length']}"
                        )
                        return False
        
        # ISO timestamp validation
        if expected_type == 'ISO timestamp':
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                self.errors.append(f"❌ {field_path}: Invalid ISO timestamp '{value}'")
                return False
        
        self.passed.append(f"✅ {field_path}: Valid {expected_type}")
        return True
    
    def validate_dashboard_data_comprehensive(self) -> bool:
        """Deep validation of dashboard_data.json structure."""
        filepath = self.json_dir / 'dashboard_data.json'
        if not filepath.exists():
            self.errors.append(f"❌ dashboard_data.json not found at {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        all_valid = True
        
        # Root level
        all_valid &= self.validate_value(data.get('version'), 'string', 
                                         'dashboard_data.version')
        all_valid &= self.validate_value(data.get('last_updated'), 'ISO timestamp',
                                         'dashboard_data.last_updated')
        
        # current_state validation
        if 'current_state' not in data:
            self.errors.append("❌ dashboard_data.current_state missing")
            return False
        
        cs = data['current_state']
        all_valid &= self.validate_value(cs.get('timestamp'), 'ISO timestamp',
                                         'current_state.timestamp')
        all_valid &= self.validate_value(cs.get('vix'), 'float',
                                         'current_state.vix',
                                         {'range': (0, 200)})
        all_valid &= self.validate_value(cs.get('vix_regime'), 'int',
                                         'current_state.vix_regime',
                                         {'range': (0, 3)})
        all_valid &= self.validate_value(cs.get('vix_regime_name'), 'string',
                                         'current_state.vix_regime_name',
                                         {'enum': ['Low Vol', 'Normal', 'Elevated', 'Crisis']})
        all_valid &= self.validate_value(cs.get('days_in_regime'), 'int',
                                         'current_state.days_in_regime',
                                         {'range': (0, 10000)})
        all_valid &= self.validate_value(cs.get('spx_close'), 'float',
                                         'current_state.spx_close',
                                         {'range': (0, 100000)})
        all_valid &= self.validate_value(cs.get('anomaly_ensemble_score'), 'float',
                                         'current_state.anomaly_ensemble_score',
                                         {'range': (0, 1)})
        all_valid &= self.validate_value(cs.get('anomaly_severity'), 'string',
                                         'current_state.anomaly_severity',
                                         {'enum': ['NORMAL', 'MODERATE', 'HIGH', 'CRITICAL']})
        
        # regime_analysis validation
        if 'regime_analysis' in data and data['regime_analysis'].get('available'):
            ra = data['regime_analysis']
            all_valid &= self.validate_value(ra.get('current_regime'), 'int',
                                             'regime_analysis.current_regime',
                                             {'range': (0, 3)})
            all_valid &= self.validate_value(ra.get('days_in_regime'), 'int',
                                             'regime_analysis.days_in_regime',
                                             {'range': (0, 10000)})
            
            if 'regimes' in ra and isinstance(ra['regimes'], dict):
                for regime_id, regime_data in ra['regimes'].items():
                    prefix = f'regime_analysis.regimes.{regime_id}'
                    all_valid &= self.validate_value(regime_data.get('persistence_5d'), 'float',
                                                     f'{prefix}.persistence_5d',
                                                     {'range': (0, 1)})
                    all_valid &= self.validate_value(regime_data.get('mean_duration'), 'float',
                                                     f'{prefix}.mean_duration',
                                                     {'range': (0, 1000)})
        
        # anomaly_analysis validation
        if 'anomaly_analysis' in data and data['anomaly_analysis'].get('available'):
            aa = data['anomaly_analysis']
            
            # Ensemble
            if 'ensemble' in aa:
                ens = aa['ensemble']
                all_valid &= self.validate_value(ens.get('score'), 'float',
                                                 'anomaly_analysis.ensemble.score',
                                                 {'range': (0, 1)})
            
            # Domain anomalies
            if 'domain_anomalies' in aa:
                expected_domains = [
                    'vix_mean_reversion', 'vix_momentum', 'vix_regime_structure',
                    'cboe_options_flow', 'vix_spx_relationship', 'spx_price_action',
                    'spx_volatility_regime', 'macro_rates', 'commodities_stress',
                    'cross_asset_divergence'
                ]
                
                for domain in expected_domains:
                    if domain not in aa['domain_anomalies']:
                        self.errors.append(f"❌ Missing anomaly domain: {domain}")
                        all_valid = False
                        continue
                    
                    domain_data = aa['domain_anomalies'][domain]
                    prefix = f'anomaly_analysis.domain_anomalies.{domain}'
                    
                    all_valid &= self.validate_value(domain_data.get('score'), 'float',
                                                     f'{prefix}.score',
                                                     {'range': (0, 1)})
                    all_valid &= self.validate_value(domain_data.get('level'), 'string',
                                                     f'{prefix}.level',
                                                     {'enum': ['NORMAL', 'MODERATE', 'HIGH', 'CRITICAL']})
            
            # Top anomalies
            if 'top_anomalies' in aa:
                all_valid &= self.validate_value(aa['top_anomalies'], 'array',
                                                 'anomaly_analysis.top_anomalies',
                                                 {'min_length': 0})
        
        # Alerts validation
        if 'alerts' in data:
            all_valid &= self.validate_value(data['alerts'], 'array',
                                             'alerts',
                                             {'min_length': 0})
        
        return all_valid
    
    def validate_feature_attribution(self) -> bool:
        """Validate anomaly_feature_attribution.json structure."""
        filepath = self.json_dir / 'anomaly_feature_attribution.json'
        if not filepath.exists():
            self.errors.append(f"❌ anomaly_feature_attribution.json not found at {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        all_valid = True
        
        if 'domains' not in data:
            self.errors.append("❌ anomaly_feature_attribution missing 'domains' key")
            return False
        
        for domain_name, domain_data in data['domains'].items():
            if 'features' not in domain_data:
                self.errors.append(f"❌ {domain_name} missing 'features' array")
                all_valid = False
                continue
            
            if not isinstance(domain_data['features'], list):
                self.errors.append(f"❌ {domain_name}.features is not an array")
                all_valid = False
                continue
            
            for i, feature in enumerate(domain_data['features']):
                prefix = f'{domain_name}.features[{i}]'
                all_valid &= self.validate_value(feature.get('feature'), 'string',
                                                 f'{prefix}.feature')
                all_valid &= self.validate_value(feature.get('value'), 'float',
                                                 f'{prefix}.value')
                all_valid &= self.validate_value(feature.get('importance'), 'float',
                                                 f'{prefix}.importance',
                                                 {'range': (0, 1)})
        
        return all_valid
    
    def validate_cross_file_consistency(self) -> bool:
        """Validate consistency across multiple JSON files."""
        dashboard_file = self.json_dir / 'dashboard_data.json'
        market_state_file = self.json_dir / 'market_state.json'
        
        if not (dashboard_file.exists() and market_state_file.exists()):
            self.warnings.append("⚠️  Cannot validate cross-file consistency (files missing)")
            return True
        
        with open(dashboard_file, 'r') as f:
            dashboard = json.load(f)
        with open(market_state_file, 'r') as f:
            market_state = json.load(f)
        
        all_valid = True
        
        # Check VIX consistency
        dash_vix = dashboard.get('current_state', {}).get('vix')
        market_vix = market_state.get('market_data', {}).get('vix')
        
        if dash_vix is not None and market_vix is not None:
            if abs(dash_vix - market_vix) > 0.01:
                self.errors.append(
                    f"❌ VIX mismatch: dashboard={dash_vix:.2f}, "
                    f"market_state={market_vix:.2f}"
                )
                all_valid = False
            else:
                self.passed.append("✅ VIX consistent across files")
        
        # Check timestamps are recent (within 7 days)
        dash_timestamp = dashboard.get('last_updated')
        if dash_timestamp:
            try:
                ts = datetime.fromisoformat(dash_timestamp.replace('Z', '+00:00'))
                age_days = (datetime.now() - ts.replace(tzinfo=None)).days
                if age_days > 7:
                    self.warnings.append(
                        f"⚠️  Data is {age_days} days old (last_updated: {dash_timestamp})"
                    )
                else:
                    self.passed.append(f"✅ Data is fresh ({age_days} days old)")
            except Exception as e:
                self.warnings.append(f"⚠️  Could not parse timestamp: {e}")
        
        return all_valid
    
    def run_comprehensive_validation(self) -> bool:
        """Run all comprehensive validation checks."""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA CONTRACT VALIDATION")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"JSON Directory: {self.json_dir.absolute()}")
        print("="*80)
        
        if not self.json_dir.exists():
            print(f"\n❌ ERROR: JSON directory does not exist: {self.json_dir}")
            return False
        
        all_valid = True
        
        print("\n[1/3] Deep validating dashboard_data.json...")
        if not self.validate_dashboard_data_comprehensive():
            all_valid = False
        
        print("\n[2/3] Validating feature attribution structure...")
        if not self.validate_feature_attribution():
            all_valid = False
        
        print("\n[3/3] Checking cross-file consistency...")
        if not self.validate_cross_file_consistency():
            all_valid = False
        
        self._print_summary()
        
        return all_valid
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("="*80)
        
        total_checks = len(self.passed) + len(self.warnings) + len(self.errors)
        pass_rate = (len(self.passed) / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\n📊 Total Checks: {total_checks}")
        print(f"✅ Passed: {len(self.passed)} ({pass_rate:.1f}%)")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for msg in self.errors[:20]:
                print(f"   {msg}")
            if len(self.errors) > 20:
                print(f"   ... and {len(self.errors) - 20} more")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for msg in self.warnings[:10]:
                print(f"   {msg}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more")
        
        print(f"\n{'='*80}")
        if not self.errors:
            print("✅ COMPREHENSIVE VALIDATION PASSED")
        else:
            print("❌ COMPREHENSIVE VALIDATION FAILED")
        print("="*80)


if __name__ == "__main__":
    validator = ComprehensiveValidator()
    passed = validator.run_comprehensive_validation()
    exit(0 if passed else 1)