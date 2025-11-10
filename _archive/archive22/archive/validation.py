"""Data Validation & Enhanced Caching System
Ensures data quality, implements smart caching, and provides diagnostics"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import hashlib
import warnings


class DataQualityValidator:
    """
    Comprehensive data quality validation.
    Catches issues before they impact model training.
    """
    
    @staticmethod
    def validate_series(series: pd.Series, name: str, 
                       min_samples: int = 100,
                       max_null_pct: float = 10.0,
                       check_stationarity: bool = False) -> Dict:
        """
        Comprehensive validation of a single series.
        
        Returns dict with:
        - is_valid: bool
        - issues: List[str]
        - stats: Dict[str, float]
        """
        issues = []
        stats = {}
        
        # Basic checks
        if series is None or len(series) == 0:
            return {'is_valid': False, 'issues': ['Series is empty'], 'stats': {}}
        
        # Sample count
        stats['n_samples'] = len(series)
        if len(series) < min_samples:
            issues.append(f"Insufficient samples: {len(series)} < {min_samples}")
        
        # Null handling
        n_nulls = series.isna().sum()
        null_pct = (n_nulls / len(series)) * 100
        stats['null_count'] = n_nulls
        stats['null_pct'] = null_pct
        
        if null_pct > max_null_pct:
            issues.append(f"Excessive nulls: {null_pct:.1f}% > {max_null_pct}%")
        
        # Value range checks
        valid_values = series.dropna()
        if len(valid_values) > 0:
            stats['min'] = float(valid_values.min())
            stats['max'] = float(valid_values.max())
            stats['mean'] = float(valid_values.mean())
            stats['std'] = float(valid_values.std())
            
            # Check for infinite values
            n_inf = np.isinf(valid_values).sum()
            if n_inf > 0:
                issues.append(f"Contains {n_inf} infinite values")
            
            # Check for constant series
            if stats['std'] == 0:
                issues.append("Series is constant (std=0)")
            
            # Check for extreme outliers (>10 sigma)
            if stats['std'] > 0:
                z_scores = np.abs((valid_values - stats['mean']) / stats['std'])
                extreme_outliers = (z_scores > 10).sum()
                if extreme_outliers > 0:
                    issues.append(f"Contains {extreme_outliers} extreme outliers (>10σ)")
                    stats['max_zscore'] = float(z_scores.max())
        
        # Index validation
        if not isinstance(series.index, pd.DatetimeIndex):
            issues.append("Index is not DatetimeIndex")
        else:
            # Check for duplicates
            if series.index.duplicated().any():
                issues.append("Index contains duplicates")
            
            # Check for proper sorting
            if not series.index.is_monotonic_increasing:
                issues.append("Index is not sorted")
            
            # Check for gaps
            if len(series) > 1:
                time_diffs = series.index.to_series().diff()
                max_gap = time_diffs.max().days if len(time_diffs) > 0 else 0
                stats['max_gap_days'] = max_gap
                if max_gap > 7:  # More than 1 week gap
                    issues.append(f"Large time gap detected: {max_gap} days")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str,
                          min_samples: int = 100,
                          max_null_pct: float = 10.0) -> Dict:
        """
        Validate entire dataframe.
        Returns comprehensive diagnostics.
        """
        if df is None or df.empty:
            return {
                'is_valid': False,
                'issues': ['DataFrame is empty'],
                'column_stats': {}
            }
        
        overall_issues = []
        column_stats = {}
        
        # Overall stats
        total_nulls = df.isna().sum().sum()
        total_cells = df.size
        overall_null_pct = (total_nulls / total_cells) * 100 if total_cells > 0 else 0
        
        # Validate each column
        for col in df.columns:
            col_result = DataQualityValidator.validate_series(
                df[col], col, min_samples, max_null_pct, check_stationarity=False
            )
            column_stats[col] = col_result
            
            if not col_result['is_valid']:
                overall_issues.append(f"Column '{col}': {', '.join(col_result['issues'])}")
        
        return {
            'is_valid': len(overall_issues) == 0,
            'issues': overall_issues,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'overall_null_pct': overall_null_pct,
            'column_stats': column_stats
        }
    
    @staticmethod
    def generate_quality_report(validation_result: Dict, name: str) -> str:
        """Generate human-readable quality report"""
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"DATA QUALITY REPORT: {name}")
        report.append(f"{'='*80}")
        
        if validation_result['is_valid']:
            report.append("✅ PASSED - Data quality is acceptable")
        else:
            report.append("❌ FAILED - Data quality issues detected")
        
        # Overall stats
        if 'n_rows' in validation_result:
            report.append(f"\n📊 Overall Statistics:")
            report.append(f"   Rows: {validation_result['n_rows']}")
            report.append(f"   Columns: {validation_result['n_cols']}")
            report.append(f"   Null %: {validation_result['overall_null_pct']:.2f}%")
        
        # Issues
        if validation_result['issues']:
            report.append(f"\n⚠️  Issues ({len(validation_result['issues'])}):")
            for issue in validation_result['issues'][:10]:  # Show first 10
                report.append(f"   - {issue}")
            if len(validation_result['issues']) > 10:
                report.append(f"   ... and {len(validation_result['issues']) - 10} more")
        
        report.append(f"{'='*80}")
        return '\n'.join(report)


class SmartCache:
    """
    Enhanced caching system with versioning, invalidation, and compression.
    """
    
    def __init__(self, cache_dir: str = './data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_file = self.cache_dir / '_cache_metadata_v2.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save cache metadata: {e}")
    
    def _compute_hash(self, data: Dict) -> str:
        """Compute hash of cache key components"""
        key_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get_cache_key(self, source: str, symbol: str, start: str, end: str, 
                     params: Dict = None) -> str:
        """
        Generate unique cache key.
        
        Args:
            source: Data source (fred, yahoo, cboe)
            symbol: Instrument symbol
            start: Start date
            end: End date
            params: Additional parameters that affect data
        """
        key_data = {
            'source': source,
            'symbol': symbol,
            'start': start,
            'end': end,
            'params': params or {}
        }
        hash_suffix = self._compute_hash(key_data)
        safe_symbol = symbol.replace('^', '_').replace('=', '_').replace('/', '_')
        return f"{source}_{safe_symbol}_{start}_{end}_{hash_suffix}"
    
    def get(self, cache_key: str, max_age_days: int = None) -> Optional[pd.DataFrame]:
        """
        Retrieve from cache if valid.
        
        Args:
            cache_key: Unique cache identifier
            max_age_days: Maximum age in days (None = no limit)
        
        Returns:
            DataFrame if valid cache exists, None otherwise
        """
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        
        if not cache_path.exists():
            return None
        
        # Check age if specified
        if max_age_days is not None:
            if cache_key in self.metadata:
                created = datetime.fromisoformat(self.metadata[cache_key]['created'])
                age_days = (datetime.now() - created).days
                if age_days > max_age_days:
                    return None
        
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            warnings.warn(f"Cache read failed for {cache_key}: {e}")
            return None
    
    def put(self, cache_key: str, df: pd.DataFrame, metadata: Dict = None):
        """
        Store DataFrame in cache with metadata.
        
        Args:
            cache_key: Unique cache identifier
            df: DataFrame to cache
            metadata: Additional metadata to store
        """
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        
        try:
            # Write data
            df.to_parquet(cache_path, compression='snappy')
            
            # Update metadata
            self.metadata[cache_key] = {
                'created': datetime.now().isoformat(),
                'rows': len(df),
                'cols': len(df.columns),
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None,
                'size_bytes': cache_path.stat().st_size,
                **(metadata or {})
            }
            self._save_metadata()
            
        except Exception as e:
            warnings.warn(f"Cache write failed for {cache_key}: {e}")
    
    def invalidate(self, cache_key: str):
        """Remove specific cache entry"""
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            cache_path.unlink()
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
    
    def clear_old_cache(self, days: int = 90):
        """Remove cache entries older than specified days"""
        now = datetime.now()
        removed_count = 0
        
        for cache_key, meta in list(self.metadata.items()):
            try:
                created = datetime.fromisoformat(meta['created'])
                age_days = (now - created).days
                if age_days > days:
                    self.invalidate(cache_key)
                    removed_count += 1
            except:
                continue
        
        return removed_count
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_size = sum(
            (self.cache_dir / f"{key}.parquet").stat().st_size
            for key in self.metadata
            if (self.cache_dir / f"{key}.parquet").exists()
        )
        
        return {
            'n_entries': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'oldest_entry': min(
                (meta['created'] for meta in self.metadata.values()),
                default=None
            ),
            'newest_entry': max(
                (meta['created'] for meta in self.metadata.values()),
                default=None
            )
        }
    
    def print_summary(self):
        """Print cache summary"""
        stats = self.get_stats()
        print(f"\n{'='*80}")
        print("CACHE SUMMARY")
        print(f"{'='*80}")
        print(f"Entries: {stats['n_entries']}")
        print(f"Total Size: {stats['total_size_mb']:.1f} MB")
        print(f"Location: {stats['cache_dir']}")
        if stats['oldest_entry']:
            print(f"Oldest: {stats['oldest_entry'][:10]}")
        if stats['newest_entry']:
            print(f"Newest: {stats['newest_entry'][:10]}")
        print(f"{'='*80}")


class FeatureIntegrityChecker:
    """
    Verify feature integrity after engineering.
    Ensures features meet requirements for modeling.
    """
    
    @staticmethod
    def check_feature_coverage(features_df: pd.DataFrame, 
                               required_groups: Dict[str, List[str]]) -> Dict:
        """
        Check if required feature groups are present.
        
        Args:
            features_df: DataFrame with engineered features
            required_groups: Dict mapping group names to required features
        
        Returns:
            Dict with coverage statistics
        """
        results = {}
        available_features = set(features_df.columns)
        
        for group_name, required_features in required_groups.items():
            required_set = set(required_features)
            present = required_set & available_features
            missing = required_set - available_features
            
            results[group_name] = {
                'required': len(required_set),
                'present': len(present),
                'missing': len(missing),
                'coverage_pct': (len(present) / len(required_set) * 100) if required_set else 100,
                'missing_features': list(missing)
            }
        
        return results
    
    @staticmethod
    def check_feature_correlations(features_df: pd.DataFrame, 
                                   threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated feature pairs.
        These may be redundant for modeling.
        
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        numeric_df = features_df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr = []
        for column in upper.columns:
            high_corr_rows = upper[column][upper[column] > threshold]
            for idx, corr_val in high_corr_rows.items():
                high_corr.append((column, idx, corr_val))
        
        return sorted(high_corr, key=lambda x: x[2], reverse=True)
    
    @staticmethod
    def detect_leakage_features(features_df: pd.DataFrame, 
                                target_series: pd.Series,
                                threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Detect features with suspiciously high correlation to target.
        May indicate data leakage.
        
        Returns:
            List of (feature_name, correlation) tuples
        """
        leakage_candidates = []
        
        for col in features_df.columns:
            if features_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    corr = features_df[col].corr(target_series)
                    if abs(corr) > threshold:
                        leakage_candidates.append((col, corr))
                except:
                    continue
        
        return sorted(leakage_candidates, key=lambda x: abs(x[1]), reverse=True)
    
    @staticmethod
    def generate_integrity_report(features_df: pd.DataFrame,
                                  required_groups: Dict[str, List[str]] = None,
                                  target_series: pd.Series = None) -> str:
        """Generate comprehensive feature integrity report"""
        report = []
        report.append(f"\n{'='*80}")
        report.append("FEATURE INTEGRITY REPORT")
        report.append(f"{'='*80}")
        
        # Basic stats
        report.append(f"\n📊 Dataset Statistics:")
        report.append(f"   Samples: {len(features_df)}")
        report.append(f"   Features: {len(features_df.columns)}")
        report.append(f"   Memory: {features_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        null_counts = features_df.isna().sum()
        features_with_nulls = (null_counts > 0).sum()
        total_nulls = null_counts.sum()
        report.append(f"   Features with nulls: {features_with_nulls}")
        report.append(f"   Total null values: {total_nulls}")
        
        # Feature coverage
        if required_groups:
            coverage_results = FeatureIntegrityChecker.check_feature_coverage(
                features_df, required_groups
            )
            report.append(f"\n📋 Feature Group Coverage:")
            for group_name, result in coverage_results.items():
                status = "✅" if result['coverage_pct'] == 100 else "⚠️"
                report.append(f"   {status} {group_name}: {result['coverage_pct']:.1f}% "
                            f"({result['present']}/{result['required']})")
                if result['missing_features']:
                    report.append(f"      Missing: {', '.join(result['missing_features'][:5])}")
        
        # High correlations
        high_corr = FeatureIntegrityChecker.check_feature_correlations(features_df)
        if high_corr:
            report.append(f"\n🔗 Highly Correlated Features ({len(high_corr)}):")
            for feat1, feat2, corr in high_corr[:5]:
                report.append(f"   {feat1} ↔ {feat2}: {corr:.3f}")
            if len(high_corr) > 5:
                report.append(f"   ... and {len(high_corr) - 5} more pairs")
        
        # Leakage detection
        if target_series is not None:
            leakage = FeatureIntegrityChecker.detect_leakage_features(
                features_df, target_series
            )
            if leakage:
                report.append(f"\n⚠️  Potential Data Leakage ({len(leakage)}):")
                for feat, corr in leakage[:5]:
                    report.append(f"   {feat}: {corr:.3f}")
        
        report.append(f"\n{'='*80}")
        return '\n'.join(report)


# ==================== USAGE EXAMPLE ====================

def example_usage():
    """Example of how to use the validation and caching systems"""
    
    # Initialize systems
    validator = DataQualityValidator()
    cache = SmartCache(cache_dir='./data_cache')
    integrity = FeatureIntegrityChecker()
    
    print("\n" + "="*80)
    print("DATA VALIDATION & CACHING SYSTEM - EXAMPLE")
    print("="*80)
    
    # Example: Validate a series
    example_series = pd.Series(
        np.random.randn(1000),
        index=pd.date_range('2020-01-01', periods=1000, freq='D'),
        name='example_vix'
    )
    
    validation_result = validator.validate_series(
        example_series, 
        'VIX', 
        min_samples=100
    )
    
    print(validator.generate_quality_report(validation_result, 'VIX'))
    
    # Example: Cache operations
    cache_key = cache.get_cache_key(
        source='yahoo',
        symbol='VIX',
        start='2020-01-01',
        end='2023-12-31'
    )
    
    # Store in cache
    cache.put(cache_key, pd.DataFrame(example_series))
    
    # Retrieve from cache
    cached_data = cache.get(cache_key, max_age_days=30)
    print(f"\nCache retrieval: {'Success' if cached_data is not None else 'Failed'}")
    
    # Cache stats
    cache.print_summary()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    example_usage()