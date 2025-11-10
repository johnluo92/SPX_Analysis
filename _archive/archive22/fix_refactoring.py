#!/usr/bin/env python3
"""
Fix the refactoring by moving files and updating imports
"""
import os
import re
import shutil

# Change this to your actual path
SRC_DIR = "/Users/johnluo/Desktop/GitHub/SPX_Analysis/src"

def move_files():
    """Move files to their new locations"""
    moves = [
        ("UnifiedDataFetcher.py", "core/data_fetcher.py"),
        ("unified_feature_engine.py", "core/feature_engine.py"),
        ("anomaly_system.py", "core/anomaly_detector.py"),
        ("vix_predictor_v2.py", "core/predictor.py"),
    ]
    
    for old_path, new_path in moves:
        old_full = os.path.join(SRC_DIR, old_path)
        new_full = os.path.join(SRC_DIR, new_path)
        
        if os.path.exists(old_full):
            print(f"Moving {old_path} -> {new_path}")
            shutil.move(old_full, new_full)
        else:
            print(f"Skipping {old_path} (already moved or doesn't exist)")

def fix_imports_in_file(filepath, replacements):
    """Fix imports in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    for old_import, new_import in replacements:
        content = content.replace(old_import, new_import)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def fix_all_imports():
    """Fix imports in all affected files"""
    
    # Imports to fix in core modules (use relative imports)
    core_replacements = [
        ("from UnifiedDataFetcher import UnifiedDataFetcher", "from .data_fetcher import UnifiedDataFetcher"),
        ("from unified_feature_engine import UnifiedFeatureEngine", "from .feature_engine import UnifiedFeatureEngine"),
        ("from anomaly_system import AnomalyDetector", "from .anomaly_detector import AnomalyDetector"),
        ("from vix_predictor_v2 import", "from .predictor import"),
    ]
    
    # Imports to fix in root-level files (use absolute imports)
    root_replacements = [
        ("from UnifiedDataFetcher import UnifiedDataFetcher", "from core.data_fetcher import UnifiedDataFetcher"),
        ("from unified_feature_engine import UnifiedFeatureEngine", "from core.feature_engine import UnifiedFeatureEngine"),
        ("from anomaly_system import AnomalyDetector", "from core.anomaly_detector import AnomalyDetector"),
        ("from vix_predictor_v2 import", "from core.predictor import"),
    ]
    
    # Fix core files
    core_files = [
        "core/data_fetcher.py",
        "core/feature_engine.py",
        "core/anomaly_detector.py",
        "core/predictor.py",
    ]
    
    for filename in core_files:
        filepath = os.path.join(SRC_DIR, filename)
        if os.path.exists(filepath):
            if fix_imports_in_file(filepath, core_replacements):
                print(f"Fixed imports in {filename}")
    
    # Fix root-level files
    root_files = [
        "integrated_system_production.py",
        "dashboard_orchestrator.py",
        "generate_claude_package.py",
    ]
    
    for filename in root_files:
        filepath = os.path.join(SRC_DIR, filename)
        if os.path.exists(filepath):
            if fix_imports_in_file(filepath, root_replacements):
                print(f"Fixed imports in {filename}")
    
    # Fix export module
    export_file = os.path.join(SRC_DIR, "export/unified_exporter.py")
    if os.path.exists(export_file):
        if fix_imports_in_file(export_file, root_replacements):
            print(f"Fixed imports in export/unified_exporter.py")

if __name__ == "__main__":
    print("Step 1: Moving files...")
    move_files()
    
    print("\nStep 2: Fixing imports...")
    fix_all_imports()
    
    print("\nDone! Try running your integrated_system_production.py now.")