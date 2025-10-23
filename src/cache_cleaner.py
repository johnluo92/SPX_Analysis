"""
Cache Cleanup Utility
Manages parquet files in .cache_sector_data directory
"""

import os
from pathlib import Path
from datetime import datetime, timedelta


class CacheCleaner:
    """Clean up old parquet cache files."""
    
    def __init__(self, cache_dir='.cache_sector_data'):
        self.cache_dir = Path(cache_dir)
    
    def clean_old_files(self, max_age_days=30, verbose=True):
        """
        Remove cache files older than max_age_days.
        
        Args:
            max_age_days: Maximum age in days to keep files
            verbose: Print cleaning progress
        """
        if not self.cache_dir.exists():
            if verbose:
                print(f"✓ Cache directory doesn't exist: {self.cache_dir}")
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        files_removed = 0
        bytes_freed = 0
        
        if verbose:
            print(f"\n🧹 Cleaning cache files older than {max_age_days} days...")
            print(f"   Cache directory: {self.cache_dir}")
        
        for file_path in self.cache_dir.glob('*.parquet'):
            # Get file modification time
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if mtime < cutoff_time:
                file_size = file_path.stat().st_size
                try:
                    file_path.unlink()
                    files_removed += 1
                    bytes_freed += file_size
                    if verbose:
                        age_days = (datetime.now() - mtime).days
                        print(f"   ✗ Removed: {file_path.name} (age: {age_days} days, size: {file_size/1024:.1f}KB)")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Failed to remove {file_path.name}: {e}")
        
        if verbose:
            print(f"\n✅ Cleanup complete:")
            print(f"   Files removed: {files_removed}")
            print(f"   Space freed: {bytes_freed/1024:.1f}KB")
        
        return files_removed
    
    def get_cache_info(self):
        """Get info about current cache."""
        if not self.cache_dir.exists():
            return {
                'exists': False,
                'file_count': 0,
                'total_size_kb': 0
            }
        
        files = list(self.cache_dir.glob('*.parquet'))
        total_size = sum(f.stat().st_size for f in files)
        
        # Get age of files
        file_ages = []
        for f in files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            file_ages.append({
                'name': f.name,
                'age_days': age_days,
                'size_kb': f.stat().st_size / 1024
            })
        
        return {
            'exists': True,
            'file_count': len(files),
            'total_size_kb': total_size / 1024,
            'files': sorted(file_ages, key=lambda x: x['age_days'], reverse=True)
        }
    
    def print_cache_info(self):
        """Print cache information."""
        info = self.get_cache_info()
        
        if not info['exists']:
            print(f"\n📁 Cache directory doesn't exist: {self.cache_dir}")
            return
        
        print(f"\n📁 Cache Info:")
        print(f"   Directory: {self.cache_dir}")
        print(f"   Files: {info['file_count']}")
        print(f"   Total size: {info['total_size_kb']:.1f}KB")
        
        if info['files']:
            print(f"\n   Oldest files:")
            for f in info['files'][:5]:  # Show top 5 oldest
                print(f"      • {f['name']}: {f['age_days']} days old ({f['size_kb']:.1f}KB)")


def main():
    """Command-line interface for cache cleaning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up SPX predictor cache files')
    parser.add_argument('--max-age', type=int, default=30,
                       help='Maximum age in days to keep files (default: 30)')
    parser.add_argument('--info', action='store_true',
                       help='Show cache info without cleaning')
    parser.add_argument('--cache-dir', type=str, default='.cache_sector_data',
                       help='Cache directory path (default: .cache_sector_data)')
    
    args = parser.parse_args()
    
    cleaner = CacheCleaner(cache_dir=args.cache_dir)
    
    if args.info:
        cleaner.print_cache_info()
    else:
        cleaner.print_cache_info()
        removed = cleaner.clean_old_files(max_age_days=args.max_age, verbose=True)
        
        if removed == 0:
            print("\n✓ No old files to remove")


if __name__ == '__main__':
    main()