"""
Quick script to inspect your cache files and understand their structure
"""

import pandas as pd
import os

def inspect_cache_file(filename):
    """Inspect a single cache file."""
    print("\n" + "="*70)
    print(f"INSPECTING: {filename}")
    print("="*70)
    
    if not os.path.exists(filename):
        print("❌ File not found")
        return None
    
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"📦 File size: {file_size:.2f} MB")
    
    try:
        data = pd.read_pickle(filename)
        
        print(f"\n📊 Data Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"\n🗂️  Dictionary with {len(data)} keys:")
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    print(f"   • {key}: DataFrame ({len(value)} rows, {len(value.columns)} cols)")
                    print(f"      Columns: {list(value.columns)[:5]}{'...' if len(value.columns) > 5 else ''}")
                elif isinstance(value, pd.Series):
                    print(f"   • {key}: Series ({len(value)} values)")
                else:
                    print(f"   • {key}: {type(value)}")
        
        elif isinstance(data, pd.DataFrame):
            print(f"\n📊 DataFrame:")
            print(f"   Rows: {len(data)}")
            print(f"   Columns: {len(data.columns)}")
            print(f"\n   Column names:")
            for i, col in enumerate(data.columns, 1):
                print(f"      {i:2d}. {col}")
            
            print(f"\n   Index type: {type(data.index)}")
            if hasattr(data.index, 'min'):
                print(f"   Date range: {data.index.min()} to {data.index.max()}")
            
            print(f"\n   First few rows:")
            print(data.head(3))
            
            print(f"\n   Data types:")
            print(data.dtypes)
        
        elif isinstance(data, pd.Series):
            print(f"\n📊 Series:")
            print(f"   Length: {len(data)}")
            print(f"   Name: {data.name}")
            if hasattr(data.index, 'min'):
                print(f"   Date range: {data.index.min()} to {data.index.max()}")
            print(f"\n   First few values:")
            print(data.head())
        
        else:
            print(f"\n⚠️  Unexpected type: {type(data)}")
            print(f"   Content preview: {str(data)[:200]}")
        
        return data
    
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def main():
    """Inspect all cache files."""
    print("\n" + "="*70)
    print("CACHE FILE INSPECTOR")
    print("="*70)
    
    # Check multiple possible cache locations
    cache_dirs = [
        'cache',
        '.cache_sector_data',
        '.'
    ]
    
    cache_files = [
        'fred_2018-10-22_2025-10-20.pkl',
        'fred_2015-10-23_2025-10-20.pkl',
        'sectors_cache.pkl',
        'macro_cache.pkl',
        'vix_cache.pkl'
    ]
    
    found_files = []
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            print(f"\n📁 Checking: {cache_dir}/")
            for filename in os.listdir(cache_dir):
                if filename.endswith('.pkl'):
                    full_path = os.path.join(cache_dir, filename)
                    found_files.append(full_path)
                    print(f"   ✅ Found: {filename}")
    
    if not found_files:
        print("\n❌ No cache files found")
        print(f"\n📁 Current directory: {os.getcwd()}")
        return
    
    print(f"\n" + "="*70)
    print(f"FOUND {len(found_files)} CACHE FILE(S)")
    print("="*70)
    
    # Inspect each file
    results = {}
    for filepath in found_files:
        data = inspect_cache_file(filepath)
        results[filepath] = data
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    # Check which files are actually usable
    usable = []
    for filepath, data in results.items():
        if data is not None:
            usable.append(filepath)
            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"\n✅ {filename} ({file_size:.2f} MB)")
            if isinstance(data, pd.DataFrame):
                print(f"   • {len(data)} rows, {len(data.columns)} columns")
                if hasattr(data.index, 'min'):
                    print(f"   • Date range: {data.index.min().date()} to {data.index.max().date()}")
            elif isinstance(data, dict):
                print(f"   • Dictionary with {len(data)} keys")
    
    if usable:
        print("\n" + "="*70)
        print("💡 RECOMMENDED: Use the most recent cache")
        print("="*70)
        # Recommend the 2018 file (7 years, as per v3.2 spec)
        recommended = [f for f in usable if '2018-10-22' in f]
        if recommended:
            print(f"\n   Best choice: {os.path.basename(recommended[0])}")
            print(f"   Path: {recommended[0]}")
        else:
            print(f"\n   Use: {os.path.basename(usable[0])}")
            print(f"   Path: {usable[0]}")
    
    return results


if __name__ == "__main__":
    results = main()