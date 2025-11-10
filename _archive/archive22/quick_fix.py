#!/usr/bin/env python3
"""Quick fix for import errors"""
from pathlib import Path

ROOT = Path.cwd()

# Fix unified_feature_engine.py
print("Fixing unified_feature_engine.py")
file = ROOT / "unified_feature_engine.py"
if file.exists():
    content = file.read_text()
    content = content.replace(
        "from UnifiedDataFetcher import UnifiedDataFetcher",
        "from core.data_fetcher import UnifiedDataFetcher"
    )
    file.write_text(content)
    print("✅ Fixed")

# Fix anomaly_system.py
print("Fixing anomaly_system.py")
file = ROOT / "anomaly_system.py"
if file.exists():
    content = file.read_text()
    if "from UnifiedDataFetcher import" in content:
        content = content.replace(
            "from UnifiedDataFetcher import UnifiedDataFetcher",
            "from core.data_fetcher import UnifiedDataFetcher"
        )
        file.write_text(content)
        print("✅ Fixed")

# Fix vix_predictor_v2.py
print("Fixing vix_predictor_v2.py")
file = ROOT / "vix_predictor_v2.py"
if file.exists():
    content = file.read_text()
    content = content.replace(
        "from anomaly_system import MultiDimensionalAnomalyDetector",
        "from core.anomaly_detector import MultiDimensionalAnomalyDetector"
    )
    content = content.replace(
        "from UnifiedDataFetcher import UnifiedDataFetcher",
        "from core.data_fetcher import UnifiedDataFetcher"
    )
    file.write_text(content)
    print("✅ Fixed")

print("\nDone. Now move files:")
print("mv UnifiedDataFetcher.py core/data_fetcher.py")
print("mv unified_feature_engine.py core/feature_engine.py")
print("mv anomaly_system.py core/anomaly_detector.py")
print("mv vix_predictor_v2.py core/predictor.py")
