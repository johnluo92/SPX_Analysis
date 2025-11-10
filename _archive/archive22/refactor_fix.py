#!/usr/bin/env python3
"""Step-by-step refactor fix - no explanation, just actions"""
import subprocess
import sys
from pathlib import Path


def run(cmd: str):
    """Run command and exit on failure"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    if result.stdout:
        print(result.stdout)


def main():
    ROOT = Path.cwd()
    
    print("="*80)
    print("REFACTOR FIX")
    print("="*80)
    
    # Step 1: Create directory structure
    print("\n[1/10] Create directories")
    (ROOT / "core").mkdir(exist_ok=True)
    (ROOT / "export").mkdir(exist_ok=True)
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "core" / "__init__.py").touch()
    (ROOT / "export" / "__init__.py").touch()
    (ROOT / "validation" / "__init__.py").touch()
    
    # Step 2: Move unified_exporter if not already in export/
    print("\n[2/10] Move unified_exporter")
    src = ROOT / "unified_exporter.py"
    dst = ROOT / "export" / "unified_exporter.py"
    if src.exists() and not dst.exists():
        src.rename(dst)
    
    # Step 3: Move UnifiedDataFetcher to core
    print("\n[3/10] Move UnifiedDataFetcher")
    src = ROOT / "UnifiedDataFetcher.py"
    dst = ROOT / "core" / "data_fetcher.py"
    if src.exists() and not dst.exists():
        src.rename(dst)
    
    # Step 4: Move unified_feature_engine to core
    print("\n[4/10] Move unified_feature_engine")
    src = ROOT / "unified_feature_engine.py"
    dst = ROOT / "core" / "feature_engine.py"
    if src.exists() and not dst.exists():
        src.rename(dst)
    
    # Step 5: Move anomaly_system to core
    print("\n[5/10] Move anomaly_system")
    src = ROOT / "anomaly_system.py"
    dst = ROOT / "core" / "anomaly_detector.py"
    if src.exists() and not dst.exists():
        src.rename(dst)
    
    # Step 6: Move vix_predictor_v2 to core
    print("\n[6/10] Move vix_predictor_v2")
    src = ROOT / "vix_predictor_v2.py"
    dst = ROOT / "core" / "predictor.py"
    if src.exists() and not dst.exists():
        src.rename(dst)
    
    # Step 7: Update integrated_system_production.py imports
    print("\n[7/10] Update integrated_system_production.py")
    integrated = ROOT / "integrated_system_production.py"
    content = integrated.read_text()
    
    # Update imports
    content = content.replace(
        "from unified_feature_engine import UnifiedFeatureEngine",
        "from core.feature_engine import UnifiedFeatureEngine"
    )
    content = content.replace(
        "from anomaly_system import MultiDimensionalAnomalyDetector",
        "from core.anomaly_detector import MultiDimensionalAnomalyDetector"
    )
    content = content.replace(
        "from vix_predictor_v2 import VIXPredictorV4",
        "from core.predictor import VIXPredictorV4"
    )
    content = content.replace(
        "from UnifiedDataFetcher import UnifiedDataFetcher",
        "from core.data_fetcher import UnifiedDataFetcher"
    )
    
    # Remove legacy commented export calls (lines 573-583)
    lines = content.split('\n')
    new_lines = []
    skip_block = False
    for i, line in enumerate(lines):
        if i >= 572 and i <= 582:  # Lines 573-583 (0-indexed)
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)
    
    integrated.write_text(content)
    
    # Step 8: Update dashboard_orchestrator.py
    print("\n[8/10] Update dashboard_orchestrator.py")
    dashboard = ROOT / "dashboard_orchestrator.py"
    if dashboard.exists():
        content = dashboard.read_text()
        content = content.replace(
            "from export.unified_exporter import UnifiedExporter",
            "from export.unified_exporter import UnifiedExporter"  # Already correct
        )
        dashboard.write_text(content)
    
    # Step 9: Update core/__init__.py
    print("\n[9/10] Update core/__init__.py")
    core_init = ROOT / "core" / "__init__.py"
    core_init.write_text('''"""Core components"""
from .data_fetcher import UnifiedDataFetcher
from .feature_engine import UnifiedFeatureEngine
from .anomaly_detector import MultiDimensionalAnomalyDetector
from .predictor import VIXPredictorV4

__all__ = [
    'UnifiedDataFetcher',
    'UnifiedFeatureEngine',
    'MultiDimensionalAnomalyDetector',
    'VIXPredictorV4',
]
''')
    
    # Step 10: Update export/__init__.py
    print("\n[10/10] Update export/__init__.py")
    export_init = ROOT / "export" / "__init__.py"
    export_init.write_text('''"""Export system"""
from .unified_exporter import UnifiedExporter

__all__ = ['UnifiedExporter']
''')
    
    # Step 11: Update all moved files' internal imports
    print("\n[11/11] Update internal imports")
    
    # Update core/predictor.py
    predictor = ROOT / "core" / "predictor.py"
    if predictor.exists():
        content = predictor.read_text()
        content = content.replace(
            "from anomaly_system import MultiDimensionalAnomalyDetector",
            "from .anomaly_detector import MultiDimensionalAnomalyDetector"
        )
        content = content.replace(
            "from UnifiedDataFetcher import UnifiedDataFetcher",
            "from .data_fetcher import UnifiedDataFetcher"
        )
        predictor.write_text(content)
    
    print("\n" + "="*80)
    print("REFACTOR COMPLETE")
    print("="*80)
    print("\nNext: python integrated_system_production.py")


if __name__ == "__main__":
    main()
