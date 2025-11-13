#!/usr/bin/env python3
"""
Feature Quality Diagnostics

Run when feature_quality score looks wrong to identify which
features are degraded/missing/stale.

USAGE:
  python diagnostics/feature_quality_diagnostic.py

  OR with saved features:
  python diagnostics/feature_quality_diagnostic.py --features-csv data_cache/features.csv
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_feature_quality(
    features_df: pd.DataFrame, metadata_df: pd.DataFrame = None
):
    """
    Analyze feature quality and identify issues.

    Args:
        features_df: DataFrame with features (from feature engine)
        metadata_df: Optional metadata DataFrame with feature info
    """
    logger.info("=" * 80)
    logger.info("FEATURE QUALITY DIAGNOSTIC")
    logger.info("=" * 80)

    # Get latest row
    latest_date = features_df.index[-1]
    latest_features = features_df.iloc[-1]

    logger.info(f"\nAnalyzing features for: {latest_date}")
    logger.info(f"   Total features: {len(latest_features)}")

    # 1. Missing features (NaN)
    missing = latest_features.isna()
    missing_count = missing.sum()
    missing_pct = missing_count / len(latest_features) * 100

    logger.info(f"\nMissing Features (NaN):")
    logger.info(
        f"   Count: {missing_count} / {len(latest_features)} ({missing_pct:.1f}%)"
    )

    if missing_count > 0:
        logger.info(f"\n   Missing features:")
        for feat in latest_features[missing].index[:20]:  # Limit to 20
            logger.info(f"      - {feat}")
        if missing_count > 20:
            logger.info(f"      ... and {missing_count - 20} more")

    # 2. Stale features (haven't updated recently)
    logger.info(f"\n Feature Staleness:")

    if len(features_df) >= 10:
        recent_window = features_df.iloc[-10:]

        stale_features = []
        for col in features_df.columns:
            # Skip metadata columns
            if col in ["calendar_cohort", "cohort_weight", "feature_quality"]:
                continue

            if features_df[col].dtype in [np.float64, np.int64]:
                # Check if feature has any variance in last 10 days
                variance = recent_window[col].var()
                if variance == 0 or np.isnan(variance):
                    stale_features.append(col)

        logger.info(f"   Stale features (no change in 10 days): {len(stale_features)}")
        if len(stale_features) > 0:
            logger.info(f"\n   Stale features:")
            for feat in stale_features[:20]:  # Limit to 20
                logger.info(f"      - {feat}")
            if len(stale_features) > 20:
                logger.info(f"      ... and {len(stale_features) - 20} more")

    # 3. Feature source analysis
    logger.info(f"\nFeature Sources:")

    # Categorize by source
    sources = {
        "SPX": [
            col
            for col in features_df.columns
            if "SPX" in col.upper() or "SP500" in col.upper()
        ],
        "VIX": [col for col in features_df.columns if "VIX" in col.upper()],
        "CBOE": [
            col
            for col in features_df.columns
            if any(
                x in col.upper()
                for x in ["SKEW", "PCCE", "PCCI", "PCC", "COR", "VXTH", "VXTLT"]
            )
        ],
        "FRED": [
            col
            for col in features_df.columns
            if any(x in col.upper() for x in ["DGS", "DTWEXBGS", "CPI"])
        ],
        "TREASURY": [
            col
            for col in features_df.columns
            if "TREASURY" in col.upper() or "YIELD" in col.upper()
        ],
        "MACRO": [
            col
            for col in features_df.columns
            if any(x in col.upper() for x in ["GOLD", "OIL", "DOLLAR", "CPI"])
        ],
    }

    for source, cols in sources.items():
        if len(cols) > 0:
            source_missing = latest_features[cols].isna().sum()
            logger.info(
                f"   {source:12} : {len(cols):3} features, {source_missing:3} missing"
            )

    # 4. Feature quality score calculation
    logger.info(f"\nFeature Quality Score Breakdown:")

    # Calculate components
    completeness = 1 - (missing_count / len(latest_features))
    staleness = (
        1 - (len(stale_features) / len(features_df.columns))
        if len(features_df) >= 10
        else 1.0
    )

    # Overall quality (weighted average)
    quality_score = completeness * 0.7 + staleness * 0.3

    logger.info(
        f"   Completeness: {completeness:.3f} ({100 * (1 - completeness):.1f}% missing)"
    )
    logger.info(
        f"   Staleness:    {staleness:.3f} ({len(stale_features)} stale features)"
    )
    logger.info(f"   Overall:      {quality_score:.3f}")

    # 5. Recent data availability
    logger.info(f"\nRecent Data Availability:")

    # Check how recent each source is
    check_features = {
        "VIX (spot)": "vix",
        "SPX (spot)": "spx_lag1",
        "SKEW": "SKEW",
        "VIX3M": "vix_term_structure",
        "10Y Treasury": "yield_10y2y",
    }

    for name, col_pattern in check_features.items():
        matching_cols = [
            c for c in features_df.columns if col_pattern.lower() in c.lower()
        ]
        if matching_cols:
            col = matching_cols[0]
            # Find last non-NaN value
            last_valid_idx = features_df[col].last_valid_index()
            if last_valid_idx:
                days_stale = (latest_date - last_valid_idx).days
                status = (
                    "âœ…" if days_stale == 0 else "âš ï¸" if days_stale <= 3 else "âŒ"
                )
                logger.info(
                    f"   {status} {name:20} : last updated {days_stale} days ago"
                )

    # 6. Recommendations
    logger.info(f"\nðŸ'¡ Recommendations:")

    if missing_pct > 10:
        logger.info(
            f"   âš ï¸  High missingness ({missing_pct:.1f}%) - check data pipeline"
        )

    if len(stale_features) > 20:
        logger.info(
            f"   âš ï¸  Many stale features ({len(stale_features)}) - check data freshness"
        )

    if quality_score < 0.8:
        logger.info(
            f"   âš ï¸  Quality score ({quality_score:.3f}) below 0.8 - forecasts may be degraded"
        )

    if quality_score >= 0.9:
        logger.info(f"   âœ… Quality score ({quality_score:.3f}) is good")

    return {
        "quality_score": quality_score,
        "completeness": completeness,
        "staleness": staleness,
        "missing_features": latest_features[missing].index.tolist(),
        "stale_features": stale_features,
        "total_features": len(latest_features),
    }


def load_features_from_system():
    """Load features from your feature engine."""
    logger.info("\nLoading features from system...")

    try:
        from core.data_fetcher import UnifiedDataFetcher
        from core.feature_engine import UnifiedFeatureEngine

        fetcher = UnifiedDataFetcher()
        engine = UnifiedFeatureEngine(fetcher)

        # Build features
        result = engine.build_complete_features(years=15)
        features_df = result["features"]

        logger.info(
            f"âœ… Loaded {features_df.shape[0]} rows, {features_df.shape[1]} features"
        )
        return features_df

    except Exception as e:
        logger.error(f"âŒ Feature loading failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose feature quality issues")
    parser.add_argument(
        "--features-csv", type=str, help="Path to features CSV (optional)"
    )

    args = parser.parse_args()

    # Load features
    if args.features_csv:
        logger.info(f"Loading features from: {args.features_csv}")
        features_df = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
    else:
        features_df = load_features_from_system()

    if features_df is not None:
        # Run diagnostics
        results = analyze_feature_quality(features_df)

        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTIC COMPLETE")
        logger.info("=" * 80)
    else:
        logger.error("\nâŒ Could not load features")
        logger.info("\nUsage:")
        logger.info("   python diagnostics/feature_quality_diagnostic.py")
        logger.info("   OR")
        logger.info(
            "   python diagnostics/feature_quality_diagnostic.py --features-csv path/to/features.csv"
        )
