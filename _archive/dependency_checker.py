"""
Configuration & Dependency Checker
Validates that all imports and config values are consistent across codebase.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set
import re


class DependencyChecker:
    """Check imports and config consistency."""
    
    def __init__(self, src_dir: str = './'):
        self.src_dir = Path(src_dir)
        self.issues = []
        self.warnings = []
    
    def check_all(self) -> bool:
        """Run all checks."""
        print("\n" + "="*80)
        print("DEPENDENCY & CONFIGURATION CHECKER")
        print("="*80)
        
        checks = [
            ("Config file", self._check_config_file),
            ("Import consistency", self._check_imports),
            ("JSON data files", self._check_json_files),
            ("Dead code", self._check_dead_code),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                passed = check_func()
                if passed:
                    print(f"   ✅ {check_name} OK")
                else:
                    print(f"   ❌ {check_name} FAILED")
                    all_passed = False
            except Exception as e:
                print(f"   ❌ {check_name} ERROR: {e}")
                self.issues.append(f"{check_name}: {e}")
                all_passed = False
        
        # Print summary
        self._print_summary(all_passed)
        
        return all_passed
    
    def _check_config_file(self) -> bool:
        """Check config.py exists and has required values."""
        config_path = self.src_dir / 'config.py'
        
        if not config_path.exists():
            self.issues.append("config.py not found")
            return False
        
        # Read config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check for required constants
        required = [
            'REGIME_BOUNDARIES',
            'REGIME_NAMES',
            'RANDOM_STATE',
            'MODEL_PARAMS',
            'SPX_FORWARD_WINDOWS',
            'LOOKBACK_YEARS'
        ]
        
        missing = []
        for const in required:
            if const not in config_content:
                missing.append(const)
        
        if missing:
            self.issues.append(f"Config missing: {', '.join(missing)}")
            return False
        
        return True
    
    def _check_imports(self) -> bool:
        """Check import consistency across files."""
        python_files = list(self.src_dir.glob('*.py'))
        
        import_patterns = {
            'config': re.compile(r'from config import (.+)'),
            'vix_predictor': re.compile(r'from vix_predictor_v2 import (.+)'),
            'spx_predictor': re.compile(r'from spx_predictor_v2 import (.+)'),
        }
        
        import_map = {}
        
        for py_file in python_files:
            if py_file.name.startswith('_'):
                continue
            
            with open(py_file, 'r') as f:
                content = f.read()
            
            for module, pattern in import_patterns.items():
                matches = pattern.findall(content)
                if matches:
                    if module not in import_map:
                        import_map[module] = set()
                    for match in matches:
                        import_map[module].update([m.strip() for m in match.split(',')])
        
        # Check for common issues
        if 'config' in import_map:
            config_imports = import_map['config']
            print(f"   Config imports: {len(config_imports)} unique constants")
            
            # Warn if REGIME_BOUNDARIES hardcoded anywhere
            for py_file in python_files:
                with open(py_file, 'r') as f:
                    content = f.read()
                if re.search(r'bins=\[.*?12.*?18.*?25.*?\]', content):
                    self.warnings.append(
                        f"{py_file.name}: Hardcoded REGIME_BOUNDARIES found"
                    )
        
        return True
    
    def _check_json_files(self) -> bool:
        """Check JSON data directory structure."""
        json_dir = self.src_dir / 'json_data'
        
        if not json_dir.exists():
            self.warnings.append("json_data directory not found")
            return False
        
        required_files = [
            'regime_statistics.json',
            'vix_history.json',
            'market_state.json',
            'anomaly_report.json'
        ]
        
        missing = []
        for filename in required_files:
            if not (json_dir / filename).exists():
                missing.append(filename)
        
        if missing:
            self.warnings.append(f"Missing JSON files: {', '.join(missing)}")
            print(f"   ⚠️  Missing: {', '.join(missing)} (run training to generate)")
        
        return True
    
    def _check_dead_code(self) -> bool:
        """Check for potential dead code."""
        python_files = list(self.src_dir.glob('*.py'))
        
        # Check for unused imports
        for py_file in python_files:
            if py_file.name.startswith('_'):
                continue
            
            with open(py_file, 'r') as f:
                lines = f.readlines()
            
            imports = []
            for line in lines[:50]:  # Check first 50 lines
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    imports.append(line.strip())
            
            # Simple heuristic: if import count > 20, might have unused imports
            if len(imports) > 20:
                self.warnings.append(
                    f"{py_file.name}: {len(imports)} imports (check for unused)"
                )
        
        return True
    
    def _print_summary(self, all_passed: bool):
        """Print summary of issues and warnings."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if self.issues:
            print("\n❌ Issues Found:")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ No issues found")
        
        if all_passed:
            print("\n✅ All checks passed")
        else:
            print("\n❌ Some checks failed - review issues above")


class ConfigReference:
    """Document expected config structure."""
    
    EXPECTED_CONFIG = """
# config.py - Expected Structure

import numpy as np

# Regime boundaries (from GMM clustering)
REGIME_BOUNDARIES = [0, 12, 18, 25, 100]
REGIME_NAMES = {
    0: "Low Volatility",
    1: "Normal Volatility", 
    2: "Elevated Volatility",
    3: "High Volatility"
}

# Model parameters
RANDOM_STATE = 42
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Training windows
LOOKBACK_YEARS = 10
LOOKBACK_YEARS_ML = 10

# SPX prediction windows
SPX_FORWARD_WINDOWS = [5, 13, 21]
SPX_RANGE_THRESHOLDS = [2, 3, 5]
"""
    
    @classmethod
    def print_reference(cls):
        """Print expected config structure."""
        print(cls.EXPECTED_CONFIG)


def main():
    """Run dependency checker."""
    checker = DependencyChecker()
    passed = checker.check_all()
    
    if not passed:
        print("\n💡 To see expected config structure:")
        print("   python dependency_checker.py --show-config")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    import sys
    
    if '--show-config' in sys.argv:
        ConfigReference.print_reference()
    else:
        main()
