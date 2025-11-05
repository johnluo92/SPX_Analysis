#!/usr/bin/env python3
"""
Automated Migration Script for Export Consolidation
====================================================
Helps migrate from 9-file system to 3-file unified system.

Usage:
    python migration_script.py --check      # Dry run, show what would change
    python migration_script.py --migrate    # Perform migration
    python migration_script.py --rollback   # Restore from backup
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse


class MigrationTool:
    def __init__(self):
        self.root = Path.cwd()
        self.json_dir = self.root / 'json_data'
        self.backup_dir = self.json_dir / 'legacy_backup'
        
        # Old files to migrate
        self.old_files = [
            'anomaly_report.json',
            'historical_anomaly_scores.json',
            'market_state.json',
            'regime_statistics.json',
            'vix_history.json',
            'anomaly_metadata.json',
            'anomaly_feature_attribution.json'
        ]
        
        # New unified files
        self.new_files = [
            'live_state.json',
            'historical.json',
            'model_cache.pkl'
        ]
    
    def check_prerequisites(self) -> bool:
        """Verify environment is ready for migration."""
        print("\n" + "="*70)
        print("MIGRATION PREREQUISITE CHECK")
        print("="*70)
        
        checks = []
        
        # Check if unified_exporter.py exists
        exporter_path = self.root / 'unified_exporter.py'
        if exporter_path.exists():
            checks.append(("✅", "unified_exporter.py found"))
        else:
            checks.append(("❌", "unified_exporter.py MISSING"))
        
        # Check if old files exist
        old_count = sum(1 for f in self.old_files if (self.json_dir / f).exists())
        checks.append(("✅" if old_count > 0 else "⚠️", 
                      f"{old_count}/{len(self.old_files)} old files present"))
        
        # Check if backup dir exists
        if self.backup_dir.exists():
            checks.append(("⚠️", "Backup directory already exists (previous migration?)"))
        else:
            checks.append(("✅", "No previous backup found"))
        
        # Check if new files already exist
        new_count = sum(1 for f in self.new_files if (self.json_dir / f).exists())
        if new_count > 0:
            checks.append(("⚠️", f"{new_count}/{len(self.new_files)} new files already exist"))
        else:
            checks.append(("✅", "No new files found (clean slate)"))
        
        # Display results
        for status, msg in checks:
            print(f"{status} {msg}")
        
        all_passed = all(status == "✅" for status, _ in checks[:2])
        
        print("\n" + "-"*70)
        if all_passed:
            print("✅ Prerequisites met - ready to migrate")
        else:
            print("❌ Prerequisites not met - fix issues above")
        print("="*70 + "\n")
        
        return all_passed
    
    def create_backup(self) -> bool:
        """Backup old files before migration."""
        print("\n" + "="*70)
        print("CREATING BACKUP")
        print("="*70)
        
        self.backup_dir.mkdir(exist_ok=True)
        
        backed_up = []
        for filename in self.old_files:
            src = self.json_dir / filename
            if src.exists():
                dst = self.backup_dir / filename
                try:
                    shutil.copy2(src, dst)
                    size_kb = src.stat().st_size / 1024
                    backed_up.append((filename, size_kb))
                    print(f"✅ {filename} → backup ({size_kb:.1f} KB)")
                except Exception as e:
                    print(f"❌ {filename}: {e}")
                    return False
        
        # Create backup manifest
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'files': [f for f, _ in backed_up],
            'total_size_kb': sum(s for _, s in backed_up)
        }
        
        manifest_path = self.backup_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Backed up {len(backed_up)} files ({manifest['total_size_kb']:.1f} KB)")
        print(f"   Location: {self.backup_dir}")
        print("="*70 + "\n")
        
        return True
    
    def verify_new_system(self) -> bool:
        """Verify new unified files exist and are valid."""
        print("\n" + "="*70)
        print("VERIFYING NEW SYSTEM")
        print("="*70)
        
        all_valid = True
        
        for filename in self.new_files:
            filepath = self.json_dir / filename
            
            if not filepath.exists():
                print(f"❌ {filename}: NOT FOUND")
                all_valid = False
                continue
            
            size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Validate JSON files
            if filename.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Check schema version
                    schema = data.get('schema_version', 'unknown')
                    print(f"✅ {filename} ({size_mb:.1f} MB) - schema v{schema}")
                    
                    # Specific validations
                    if filename == 'live_state.json':
                        if 'market' not in data or 'anomaly' not in data:
                            print(f"   ⚠️  Missing required sections")
                            all_valid = False
                    elif filename == 'historical.json':
                        if 'historical' not in data:
                            print(f"   ⚠️  Missing historical section")
                            all_valid = False
                
                except json.JSONDecodeError as e:
                    print(f"❌ {filename}: INVALID JSON - {e}")
                    all_valid = False
                except Exception as e:
                    print(f"❌ {filename}: {e}")
                    all_valid = False
            else:
                # Just check size for pickle files
                print(f"✅ {filename} ({size_mb:.1f} MB)")
        
        print("\n" + "-"*70)
        if all_valid:
            print("✅ New system validated")
        else:
            print("❌ Validation failed")
        print("="*70 + "\n")
        
        return all_valid
    
    def compare_data(self):
        """Compare old vs new data for accuracy."""
        print("\n" + "="*70)
        print("DATA COMPARISON (Old → New)")
        print("="*70)
        
        try:
            # Load old anomaly report
            old_anomaly = self.json_dir / 'anomaly_report.json'
            if old_anomaly.exists():
                with open(old_anomaly, 'r') as f:
                    old_data = json.load(f)
            else:
                old_data = None
            
            # Load new live state
            new_live = self.json_dir / 'live_state.json'
            if new_live.exists():
                with open(new_live, 'r') as f:
                    new_data = json.load(f)
            else:
                new_data = None
            
            if old_data and new_data:
                # Compare ensemble scores
                old_score = old_data['ensemble']['score']
                new_score = new_data['anomaly']['ensemble_score']
                diff = abs(old_score - new_score)
                
                status = "✅" if diff < 0.01 else "⚠️"
                print(f"{status} Ensemble Score: {old_score:.4f} → {new_score:.4f} (Δ {diff:.4f})")
                
                # Compare classification
                old_class = old_data.get('classification', {}).get('level', 'N/A')
                new_class = new_data['anomaly']['classification']
                match = "✅" if old_class == new_class else "⚠️"
                print(f"{match} Classification: {old_class} → {new_class}")
                
                # Compare persistence
                old_persist = old_data.get('persistence', {})
                new_persist = new_data.get('persistence', {})
                
                old_streak = old_persist.get('historical_stats', {}).get('current_streak', 0)
                new_streak = new_persist.get('current_streak', 0)
                match = "✅" if old_streak == new_streak else "⚠️"
                print(f"{match} Current Streak: {old_streak}d → {new_streak}d")
                
            else:
                print("⚠️  Cannot compare (missing files)")
        
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
        
        print("="*70 + "\n")
    
    def rollback(self) -> bool:
        """Restore from backup."""
        print("\n" + "="*70)
        print("ROLLING BACK")
        print("="*70)
        
        if not self.backup_dir.exists():
            print("❌ No backup found")
            return False
        
        # Load manifest
        manifest_path = self.backup_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            print(f"📦 Backup from: {manifest['timestamp']}")
            print(f"   Files: {len(manifest['files'])}")
        
        # Restore files
        restored = []
        for filename in self.old_files:
            src = self.backup_dir / filename
            if src.exists():
                dst = self.json_dir / filename
                try:
                    shutil.copy2(src, dst)
                    restored.append(filename)
                    print(f"✅ Restored: {filename}")
                except Exception as e:
                    print(f"❌ {filename}: {e}")
        
        print(f"\n✅ Restored {len(restored)} files")
        print("="*70 + "\n")
        
        return len(restored) > 0
    
    def migrate(self):
        """Run full migration."""
        print("\n" + "="*70)
        print("EXPORT CONSOLIDATION MIGRATION")
        print("="*70)
        print("This will migrate from 9-file system to 3-file unified system")
        print("="*70 + "\n")
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites not met. Aborting.")
            return False
        
        # Step 2: Create backup
        if not self.create_backup():
            print("❌ Backup failed. Aborting.")
            return False
        
        print("\n" + "="*70)
        print("MIGRATION INSTRUCTIONS")
        print("="*70)
        print("""
Next steps:

1. Run training to generate new files:
   python integrated_system_production.py

2. Run this script again to verify:
   python migration_script.py --verify

3. If verification passes, old files can be removed:
   python migration_script.py --cleanup

4. If issues occur, rollback:
   python migration_script.py --rollback
""")
        print("="*70 + "\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Export Consolidation Migration Tool'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check prerequisites (dry run)'
    )
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Perform migration (creates backup)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify new system after training'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare old vs new data'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Restore from backup'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove old files (after successful migration)'
    )
    
    args = parser.parse_args()
    
    tool = MigrationTool()
    
    if args.check:
        tool.check_prerequisites()
    
    elif args.migrate:
        tool.migrate()
    
    elif args.verify:
        if tool.verify_new_system():
            print("✅ Verification passed")
            tool.compare_data()
        else:
            print("❌ Verification failed")
            sys.exit(1)
    
    elif args.compare:
        tool.compare_data()
    
    elif args.rollback:
        if tool.rollback():
            print("✅ Rollback complete")
        else:
            print("❌ Rollback failed")
            sys.exit(1)
    
    elif args.cleanup:
        response = input("\n⚠️  This will DELETE old files. Continue? (yes/no): ")
        if response.lower() == 'yes':
            for filename in tool.old_files:
                filepath = tool.json_dir / filename
                if filepath.exists():
                    filepath.unlink()
                    print(f"🗑️  Deleted: {filename}")
            print("\n✅ Cleanup complete")
        else:
            print("❌ Cleanup cancelled")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
