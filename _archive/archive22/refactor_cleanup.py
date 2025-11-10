#!/usr/bin/env python3
"""Refactor Cleanup - Archive old files and fix structure"""
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
ARCHIVE_DIR = ROOT / "archive" / f"pre_refactor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Files to archive (remove from root)
OLD_FILES = [
    "dashboard_data_contract.py",
    "data_lineage.py",
]

# JSON files to archive
OLD_JSON = [
    "anomaly_report.json",
    "anomaly_metadata.json",
    "anomaly_feature_attribution.json",
    "market_state.json",
    "regime_statistics.json",
    "vix_history.json",
    "historical_anomaly_scores.json",
    "dashboard_data.json",
    "refresh_state.pkl",
    "firing_detector_analysis.json",
]

# Directories to remove from json_data
REMOVE_DIRS = ["core", "export", "validation"]


def archive_file(src: Path, category: str):
    """Archive a single file"""
    if not src.exists():
        return False
    
    dest_dir = ARCHIVE_DIR / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    
    shutil.move(str(src), str(dest))
    print(f"✅ Archived: {src.name} → {category}/")
    return True


def remove_dir(path: Path):
    """Remove directory"""
    if not path.exists():
        return False
    
    shutil.rmtree(path)
    print(f"🗑️  Removed: {path}")
    return True


def main():
    print("\n" + "="*80)
    print("REFACTOR CLEANUP")
    print("="*80)
    
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Archive old Python files
    print("\n📦 Archiving old Python files...")
    archived_py = 0
    for filename in OLD_FILES:
        if archive_file(ROOT / filename, "python"):
            archived_py += 1
    
    # Archive old JSON files
    print("\n📦 Archiving old JSON files...")
    json_dir = ROOT / "json_data"
    archived_json = 0
    for filename in OLD_JSON:
        if archive_file(json_dir / filename, "json"):
            archived_json += 1
    
    # Remove empty directories from json_data
    print("\n🗑️  Removing empty directories...")
    removed_dirs = 0
    for dirname in REMOVE_DIRS:
        dir_path = json_dir / dirname
        if remove_dir(dir_path):
            removed_dirs += 1
    
    # Summary
    print("\n" + "="*80)
    print("CLEANUP COMPLETE")
    print("="*80)
    print(f"Archived Python files: {archived_py}")
    print(f"Archived JSON files: {archived_json}")
    print(f"Removed directories: {removed_dirs}")
    print(f"\n📁 Archive location: {ARCHIVE_DIR}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
