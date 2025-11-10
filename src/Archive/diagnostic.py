"""
System Diagnostic Tool
======================
Analyzes code structure, identifies issues, and provides actionable fixes.
"""

import sys
from pathlib import Path
import importlib.util
import ast

def color_print(msg, color='green'):
    """Print with color."""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{msg}{colors['end']}")

def check_file_syntax(filepath):
    """Check if file has valid Python syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except TabError as e:
        return False, f"TabError: {e}"

def analyze_imports(filepath):
    """Extract all imports from a file."""
    imports = {'stdlib': [], 'third_party': [], 'local': []}
    
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split('.')[0]
                    if name.startswith('core.') or name in ['config', 'integrated_system_production']:
                        imports['local'].append(alias.name)
                    elif name in ['sys', 'os', 'pathlib', 'datetime', 'json', 'warnings']:
                        imports['stdlib'].append(alias.name)
                    else:
                        imports['third_party'].append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    name = node.module.split('.')[0]
                    imported_items = [alias.name for alias in node.names]
                    
                    if name.startswith('core') or name in ['config', 'integrated_system_production']:
                        imports['local'].extend([f"{node.module}.{item}" for item in imported_items])
                    elif name in ['sys', 'os', 'pathlib', 'datetime', 'json', 'warnings']:
                        imports['stdlib'].extend([f"{node.module}.{item}" for item in imported_items])
                    else:
                        imports['third_party'].extend([f"{node.module}.{item}" for item in imported_items])
        
        return imports
    except Exception as e:
        return None

def check_module_availability(module_name):
    """Check if a Python module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

def main():
    print("="*80)
    color_print("SYSTEM DIAGNOSTIC TOOL", 'blue')
    print("="*80)
    
    # Define files to check
    files_to_check = {
        'config.py': 'Configuration',
        'integrated_system_production.py': 'Main System',
        'xgboost_integration.py': 'Integration Script',
        'core/anomaly_detector.py': 'Anomaly Detector',
        'core/data_fetcher.py': 'Data Fetcher',
        'core/feature_engine.py': 'Feature Engine',
        'core/xgboost_trainer_v2.py': 'XGBoost Trainer',
        'core/xgboost_feature_selector_v2.py': 'Feature Selector',
    }
    
    print(f"\n{'='*80}")
    color_print("1. FILE STRUCTURE CHECK", 'blue')
    print("="*80)
    
    existing_files = []
    missing_files = []
    
    for filepath, description in files_to_check.items():
        path = Path(filepath)
        if path.exists():
            color_print(f"✓ {filepath:45s} ({description})", 'green')
            existing_files.append(filepath)
        else:
            color_print(f"✗ {filepath:45s} (MISSING)", 'red')
            missing_files.append(filepath)
    
    print(f"\nSummary: {len(existing_files)}/{len(files_to_check)} files found")
    
    if missing_files:
        print("\nMissing files:")
        for f in missing_files:
            print(f"  • {f}")
    
    # Check syntax
    print(f"\n{'='*80}")
    color_print("2. SYNTAX CHECK", 'blue')
    print("="*80)
    
    syntax_issues = []
    for filepath in existing_files:
        valid, error = check_file_syntax(filepath)
        if valid:
            color_print(f"✓ {filepath}", 'green')
        else:
            color_print(f"✗ {filepath}", 'red')
            color_print(f"  Error: {error}", 'red')
            syntax_issues.append((filepath, error))
    
    if syntax_issues:
        print("\n⚠️  Syntax issues found:")
        for filepath, error in syntax_issues:
            print(f"\n  File: {filepath}")
            print(f"  Error: {error}")
    
    # Check imports
    print(f"\n{'='*80}")
    color_print("3. IMPORT ANALYSIS", 'blue')
    print("="*80)
    
    all_third_party = set()
    import_issues = []
    
    for filepath in existing_files:
        imports = analyze_imports(filepath)
        if imports is None:
            color_print(f"✗ {filepath}: Could not analyze imports", 'red')
            continue
        
        print(f"\n{filepath}:")
        
        # Local imports
        if imports['local']:
            print(f"  Local imports ({len(imports['local'])}):")
            for imp in imports['local'][:5]:
                print(f"    • {imp}")
            if len(imports['local']) > 5:
                print(f"    ... and {len(imports['local'])-5} more")
        
        # Third party
        if imports['third_party']:
            all_third_party.update([imp.split('.')[0] for imp in imports['third_party']])
            print(f"  Third-party imports ({len(imports['third_party'])}):")
            for imp in imports['third_party'][:5]:
                print(f"    • {imp}")
            if len(imports['third_party']) > 5:
                print(f"    ... and {len(imports['third_party'])-5} more")
    
    # Check third-party availability
    print(f"\n{'='*80}")
    color_print("4. THIRD-PARTY PACKAGE CHECK", 'blue')
    print("="*80)
    
    required_packages = list(all_third_party)
    available = []
    unavailable = []
    
    for pkg in sorted(required_packages):
        if check_module_availability(pkg):
            color_print(f"✓ {pkg}", 'green')
            available.append(pkg)
        else:
            color_print(f"✗ {pkg} (NOT INSTALLED)", 'red')
            unavailable.append(pkg)
    
    if unavailable:
        print(f"\n⚠️  Missing packages ({len(unavailable)}):")
        print("Install with:")
        print(f"  pip install --break-system-packages {' '.join(unavailable)}")
    
    # Check import chains
    print(f"\n{'='*80}")
    color_print("5. IMPORT DEPENDENCY CHAIN", 'blue')
    print("="*80)
    
    print("\nExpected import order (bottom-up):")
    print("  1. config.py                          (no dependencies)")
    print("  2. core/data_fetcher.py               (needs: yfinance, fredapi, config)")
    print("  3. core/feature_engine.py             (needs: data_fetcher, pandas_ta, config)")
    print("  4. core/anomaly_detector.py           (needs: sklearn, config)")
    print("  5. core/xgboost_trainer_v2.py         (needs: xgboost, sklearn, config)")
    print("  6. core/xgboost_feature_selector_v2.py (needs: xgboost_trainer_v2)")
    print("  7. integrated_system_production.py    (needs: all core modules)")
    print("  8. xgboost_integration.py             (needs: integrated_system_production)")
    
    # Check specific import issues
    print(f"\n{'='*80}")
    color_print("6. KNOWN ISSUES & FIXES", 'blue')
    print("="*80)
    
    issues_found = []
    
    # Issue 1: Tab/space mixing
    selector_path = Path('core/xgboost_feature_selector_v2.py')
    if selector_path.exists():
        with open(selector_path, 'r') as f:
            content = f.read()
            if '\t' in content:
                issues_found.append({
                    'file': 'core/xgboost_feature_selector_v2.py',
                    'issue': 'Contains tab characters (should be spaces)',
                    'fix': 'Replace all tabs with 4 spaces'
                })
    
    # Issue 2: Network access
    issues_found.append({
        'file': 'core/data_fetcher.py',
        'issue': 'Requires network access for Yahoo/FRED data',
        'fix': 'Need cached data or mock mode for offline operation'
    })
    
    # Issue 3: Export dependency
    integrated_path = Path('integrated_system_production.py')
    if integrated_path.exists():
        with open(integrated_path, 'r') as f:
            if 'from export.unified_exporter' in f.read():
                issues_found.append({
                    'file': 'integrated_system_production.py',
                    'issue': 'Imports non-existent export.unified_exporter',
                    'fix': 'Comment out or make optional: try/except ImportError'
                })
    
    if issues_found:
        for i, issue in enumerate(issues_found, 1):
            print(f"\nIssue {i}:")
            print(f"  File: {issue['file']}")
            print(f"  Problem: {issue['issue']}")
            print(f"  Fix: {issue['fix']}")
    else:
        color_print("\n✓ No obvious issues found!", 'green')
    
    # Summary
    print(f"\n{'='*80}")
    color_print("7. SUMMARY & RECOMMENDATIONS", 'blue')
    print("="*80)
    
    print(f"\nFiles: {len(existing_files)}/{len(files_to_check)} present")
    print(f"Syntax errors: {len(syntax_issues)}")
    print(f"Missing packages: {len(unavailable)}")
    print(f"Known issues: {len(issues_found)}")
    
    if syntax_issues:
        print("\n⚠️  CRITICAL: Fix syntax errors first")
        
    if unavailable and not syntax_issues:
        print("\n⚠️  Install missing packages:")
        print(f"   pip install --break-system-packages {' '.join(unavailable)}")
    
    if not syntax_issues and not unavailable:
        print("\n✓ No critical blocking issues")
        print("\n⚠️  Network access issue:")
        print("   - System requires cached data for offline operation")
        print("   - Yahoo Finance and FRED API calls will fail without network")
        print("   - Need to either:")
        print("     1. Enable network access for yahoo/FRED domains")
        print("     2. Provide pre-cached parquet files in ./data_cache/")
        print("     3. Create mock/test mode with synthetic data")
    
    print(f"\n{'='*80}")
    
    return len(syntax_issues) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
