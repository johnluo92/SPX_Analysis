"""
VIX Anomaly Detection System - Data Lineage & Validation
==========================================================
Core validation functions and critical operational gaps.
Companion to system.md (architecture) and data_contracts.md (schemas).

Version: 2.0.0
Last Updated: 2025-11-01
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from dataclasses import dataclass

# ============================================================================
# FILESYSTEM PATHS
# ============================================================================

JSON_DATA_DIR = Path("json_data")
CRITICAL_FILES = {
    "anomaly_report": JSON_DATA_DIR / "anomaly_report.json",
    "historical_scores": JSON_DATA_DIR / "historical_anomaly_scores.json",
    "refresh_state": JSON_DATA_DIR / "refresh_state.pkl",
    "dashboard_data": JSON_DATA_DIR / "dashboard_data.json",
}

SUBCHART_PATHS = {
    "hero_section": Path("Chart Modules/subcharts/hero_section.html"),
    "persistence_tracker": Path("Chart Modules/subcharts/persistence_tracker.html"),
    "historical_analysis": Path("Chart Modules/subcharts/historical_analysis.html"),
    "score_distribution": Path("Chart Modules/subcharts/score_distribution.html"),
    "detector_ranking": Path("Chart Modules/subcharts/detector_ranking.html"),
    "forward_returns": Path("Chart Modules/subcharts/forward_returns.html"),
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

@dataclass
class ValidationResult:
    """Validation outcome with specific failure details"""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


def validate_parallel_arrays(data: Dict[str, List]) -> ValidationResult:
    """
    Validate that all arrays in historical_anomaly_scores.json have equal length.
    
    Critical: Parallel array misalignment breaks 5 of 6 dashboard charts.
    Required arrays: dates, ensemble_scores, spx_close, spx_forward_10d
    """
    required = ["dates", "ensemble_scores", "spx_close", "spx_forward_10d"]
    
    missing = [k for k in required if k not in data]
    if missing:
        return ValidationResult(
            passed=False,
            message=f"Missing required arrays: {missing}",
            details={"available_keys": list(data.keys())}
        )
    
    lengths = {k: len(data[k]) for k in required}
    unique_lengths = set(lengths.values())
    
    if len(unique_lengths) > 1:
        return ValidationResult(
            passed=False,
            message="Array length mismatch detected",
            details={"lengths": lengths}
        )
    
    return ValidationResult(
        passed=True,
        message=f"All arrays aligned (n={lengths['dates']})",
        details={"length": lengths['dates']}
    )


def validate_anomaly_report_structure(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate anomaly_report.json contains required fields.
    
    Required top-level keys: timestamp, ensemble, domain_anomalies, 
                             persistence, top_anomalies, classification
    """
    required_top = ["timestamp", "ensemble", "domain_anomalies", "persistence", 
                    "top_anomalies", "classification"]
    
    missing_top = [k for k in required_top if k not in data]
    if missing_top:
        return ValidationResult(
            passed=False,
            message=f"Missing top-level fields: {missing_top}",
            details={"available": list(data.keys())}
        )
    
    # Validate ensemble structure
    ensemble_fields = ["score", "std", "max_anomaly", "min_anomaly", "n_detectors"]
    missing_ensemble = [k for k in ensemble_fields if k not in data.get("ensemble", {})]
    if missing_ensemble:
        return ValidationResult(
            passed=False,
            message=f"Invalid ensemble structure, missing: {missing_ensemble}",
            details={"ensemble_keys": list(data.get("ensemble", {}).keys())}
        )
    
    # Validate classification structure
    if "level" not in data.get("classification", {}):
        return ValidationResult(
            passed=False,
            message="Missing classification.level",
            details={"classification": data.get("classification", {})}
        )
    
    return ValidationResult(
        passed=True,
        message="Anomaly report structure valid",
        details={"n_domains": len(data.get("domain_anomalies", {}))}
    )


def validate_score_ranges(data: Dict[str, Any]) -> ValidationResult:
    """Validate anomaly scores are within [0.0, 1.0] range"""
    ensemble_score = data.get("ensemble", {}).get("score")
    
    if ensemble_score is None:
        return ValidationResult(
            passed=False,
            message="Missing ensemble.score",
            details={"ensemble": data.get("ensemble", {})}
        )
    
    if not (0.0 <= ensemble_score <= 1.0):
        return ValidationResult(
            passed=False,
            message=f"Ensemble score out of range: {ensemble_score}",
            details={"score": ensemble_score}
        )
    
    # Validate domain scores
    domain_anomalies = data.get("domain_anomalies", {})
    invalid_domains = []
    
    for domain, domain_data in domain_anomalies.items():
        score = domain_data.get("score", 0)
        if not (0.0 <= score <= 1.0):
            invalid_domains.append({"domain": domain, "score": score})
    
    if invalid_domains:
        return ValidationResult(
            passed=False,
            message=f"Invalid domain scores detected",
            details={"invalid": invalid_domains}
        )
    
    return ValidationResult(
        passed=True,
        message="All scores within valid range [0.0, 1.0]",
        details={"ensemble_score": ensemble_score, "n_domains": len(domain_anomalies)}
    )


def validate_classification_levels(data: Dict[str, Any]) -> ValidationResult:
    """Validate classification levels match expected values"""
    valid_levels = {"NORMAL", "MODERATE", "HIGH", "CRITICAL"}
    
    classification = data.get("classification", {})
    level = classification.get("level")
    
    if level not in valid_levels:
        return ValidationResult(
            passed=False,
            message=f"Invalid classification level: {level}",
            details={"level": level, "valid_levels": list(valid_levels)}
        )
    
    # Validate domain levels
    domain_anomalies = data.get("domain_anomalies", {})
    invalid_domain_levels = []
    
    for domain, domain_data in domain_anomalies.items():
        domain_level = domain_data.get("level")
        if domain_level not in valid_levels:
            invalid_domain_levels.append({"domain": domain, "level": domain_level})
    
    if invalid_domain_levels:
        return ValidationResult(
            passed=False,
            message="Invalid domain classification levels",
            details={"invalid": invalid_domain_levels}
        )
    
    return ValidationResult(
        passed=True,
        message="All classification levels valid",
        details={"global": level, "n_domains": len(domain_anomalies)}
    )


def validate_threshold_consistency(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate thresholds are monotonically increasing.
    
    Required order: moderate < high < critical
    """
    thresholds = data.get("classification", {}).get("thresholds", {})
    
    required = ["moderate", "high", "critical"]
    missing = [k for k in required if k not in thresholds]
    
    if missing:
        return ValidationResult(
            passed=False,
            message=f"Missing thresholds: {missing}",
            details={"available": list(thresholds.keys())}
        )
    
    moderate = thresholds["moderate"]
    high = thresholds["high"]
    critical = thresholds["critical"]
    
    if not (moderate < high < critical):
        return ValidationResult(
            passed=False,
            message="Thresholds not monotonically increasing",
            details={"moderate": moderate, "high": high, "critical": critical}
        )
    
    return ValidationResult(
        passed=True,
        message="Thresholds properly ordered",
        details={"moderate": moderate, "high": high, "critical": critical}
    )


def validate_json_exports(json_dir: Path = JSON_DATA_DIR) -> Dict[str, ValidationResult]:
    """
    Run all validation checks on exported JSON files.
    
    Returns:
        Dictionary mapping validation name to ValidationResult
    """
    results = {}
    
    # Check file existence
    anomaly_report_path = json_dir / "anomaly_report.json"
    historical_scores_path = json_dir / "historical_anomaly_scores.json"
    
    if not anomaly_report_path.exists():
        results["file_existence"] = ValidationResult(
            passed=False,
            message=f"Missing {anomaly_report_path}",
            details=None
        )
        return results
    
    if not historical_scores_path.exists():
        results["file_existence"] = ValidationResult(
            passed=False,
            message=f"Missing {historical_scores_path}",
            details=None
        )
        return results
    
    results["file_existence"] = ValidationResult(passed=True, message="All critical files exist")
    
    # Load data
    try:
        with open(anomaly_report_path) as f:
            anomaly_data = json.load(f)
        
        with open(historical_scores_path) as f:
            historical_data = json.load(f)
    except json.JSONDecodeError as e:
        results["json_parse"] = ValidationResult(
            passed=False,
            message=f"JSON parse error: {e}",
            details=None
        )
        return results
    
    results["json_parse"] = ValidationResult(passed=True, message="All JSON files parseable")
    
    # Run validation suite
    results["anomaly_structure"] = validate_anomaly_report_structure(anomaly_data)
    results["score_ranges"] = validate_score_ranges(anomaly_data)
    results["classification_levels"] = validate_classification_levels(anomaly_data)
    results["threshold_consistency"] = validate_threshold_consistency(anomaly_data)
    results["parallel_arrays"] = validate_parallel_arrays(historical_data)
    
    return results


def print_validation_report(results: Dict[str, ValidationResult]) -> None:
    """Print formatted validation report"""
    print("\n" + "="*70)
    print("DATA LINEAGE VALIDATION REPORT")
    print("="*70 + "\n")
    
    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} | {name.replace('_', ' ').title()}")
        print(f"     {result.message}")
        if result.details and not result.passed:
            print(f"     Details: {result.details}")
        print()
    
    print("="*70)
    print(f"Results: {passed}/{total} checks passed")
    print("="*70 + "\n")


# ============================================================================
# CRITICAL GAPS DOCUMENTATION
# ============================================================================

# GAP 1: Refresh State Pickle Structure
REFRESH_STATE_PKL = {
    "file": "json_data/refresh_state.pkl",
    "producer": "IntegratedMarketSystemV4.save_state()",
    "required_for": "Cached mode operation (ENABLE_TRAINING=False)",
    "contents": {
        "system": "IntegratedMarketSystemV4 instance",
        "vix_predictor": "VIXPredictorV4 instance with trained detectors",
        "detectors": "List[IsolationForest] (15 trained models)",
        "scalers": "List[RobustScaler] (15 fitted scalers)",
        "statistical_thresholds": "Dict[str, float] (moderate, high, critical)",
        "training_scores": "np.ndarray (historical ensemble scores)",
        "last_252_days": "pd.DataFrame (recent history for persistence calc)",
        "feature_columns": "List[str] (234 feature names)",
        "regime_boundaries": "[0, 16.77, 24.40, 39.67, 100]"
    },
    "size": "5-15 MB (depends on history length)",
    "generation": "Only during training mode (ENABLE_TRAINING=True)",
    "lifespan": "Recommended: regenerate quarterly or after major market events"
}

# GAP 2: Dashboard Data Contract - Unified Export
DASHBOARD_DATA_CONTRACT = {
    "file": "json_data/dashboard_data.json",
    "producer": "dashboard_data_contract.export_unified_dashboard_data()",
    "structure": {
        "vix_state": {
            "current": "float (latest VIX level)",
            "regime": "str (Low Vol, Normal, Elevated, Crisis)",
            "regime_persistence": "int (consecutive days in regime)",
            "percentile_30d": "float (current position in 30-day window)",
            "percentile_historical": "float (position in full history)"
        },
        "anomaly_analysis": {
            "ensemble_score": "float [0.0, 1.0]",
            "classification": "str (NORMAL, MODERATE, HIGH, CRITICAL)",
            "active_detectors": "List[str] (domains with score > threshold)",
            "top_anomalies": "List[{name, score}] (top 5 domains)",
            "persistence_count": "int (number of active domains)"
        },
        "regime_statistics": {
            "current_regime": {
                "duration_days": "int",
                "typical_duration": "float (historical avg)",
                "vix_range": "[float, float] (regime boundaries)"
            },
            "historical_distribution": {
                "low_vol_pct": "float",
                "normal_pct": "float", 
                "elevated_pct": "float",
                "crisis_pct": "float"
            }
        }
    },
    "purpose": "Single unified contract for dashboard overview page",
    "consumers": ["dashboard_unified.html (main dashboard)"],
    "update_frequency": "Every refresh cycle (15s-15min)"
}

# GAP 3: Statistical Thresholds Computation
STATISTICAL_THRESHOLDS_COMPUTATION = {
    "location": "VIXPredictorV4._compute_statistical_thresholds()",
    "called_during": "Training phase after historical score generation",
    "input": "training_scores: np.ndarray (full historical ensemble scores)",
    "algorithm": """
    1. Remove NaN values from training_scores
    2. Calculate percentiles on empirical distribution:
       - moderate: 85th percentile (typical: 0.65-0.75)
       - high: 92nd percentile (typical: 0.75-0.82)
       - critical: 98th percentile (typical: 0.85-0.92)
    3. Store thresholds in detector state
    4. Export to anomaly_report.json['classification']['thresholds']
    """,
    "output": {
        "moderate": "float (85th percentile)",
        "high": "float (92nd percentile)",
        "critical": "float (98th percentile)"
    },
    "critical_property": "Data-driven, adapts to distribution shifts during retraining",
    "stability": "Coefficient of variation <5% across walk-forward folds",
    "known_issue": {
        "problem": "Dashboard HTML uses hardcoded thresholds instead of reading from JSON",
        "affected_files": [
            "persistence_tracker.html",
            "hero_section.html",
            "score_distribution.html"
        ],
        "impact": "Frontend classification diverges when thresholds update",
        "fix_estimate": "2-3 hours to update JavaScript fetch logic"
    }
}

# GAP 4: Persistence Calculation from Historical Scores
PERSISTENCE_CALCULATION = {
    "location": "VIXPredictorV4.export_anomaly_report()",
    "data_source": "ensemble_scores from export_historical_anomaly_scores()",
    "critical_dependency": "Requires full historical scores, not just last observation",
    "algorithm": """
    1. Get ensemble_scores array (full history, 3000-5000+ observations)
    2. Get high threshold from statistical_thresholds
    3. is_anomalous = (ensemble_scores >= threshold)
    4. Identify consecutive True sequences (episodes)
    5. Calculate:
       - current_streak: ongoing consecutive days (if today is anomalous)
       - mean_duration: np.mean(all_episode_lengths)
       - max_duration: np.max(all_episode_lengths)
       - total_anomaly_days: sum(is_anomalous)
       - anomaly_rate: total_anomaly_days / len(ensemble_scores)
    6. Export to anomaly_report.json['persistence']['historical_stats']
    """,
    "output_fields": {
        "current_streak": "int (consecutive anomaly days, 0 if not currently anomalous)",
        "mean_duration": "float (average episode length in days)",
        "max_duration": "int (longest recorded episode)",
        "total_anomaly_days": "int (cumulative count)",
        "anomaly_rate": "float (proportion of history flagged as anomalous)"
    },
    "historical_statistics": {
        "total_episodes": 168,
        "mean_duration": 2.3,
        "max_duration": 15,
        "anomaly_rate": 0.096,
        "distribution": "Right-skewed (skewness: 3.1)"
    },
    "interpretation": {
        "episodes_gt_5d": "10.7% of episodes, correlated with NBER recessions (r=0.68)",
        "transient_anomalies": "Median duration: 1 day",
        "sustained_crises": "Long episodes (>5d) account for 31% of total anomaly days"
    }
}

# GAP 5: Data Flow in Cached Refresh Mode
CACHED_REFRESH_DATA_FLOW = {
    "trigger": "ENABLE_TRAINING=False in config.py",
    "initialization": {
        "duration": "3-8 seconds",
        "steps": [
            "1. Load refresh_state.pkl (5-15 MB)",
            "2. Restore detector state (15 forests + scalers)",
            "3. Load last 252 days of features",
            "4. Initialize UnifiedDataFetcher with cache enabled"
        ]
    },
    "refresh_cycle": {
        "interval": "Configurable (default: 900s/15min, min: 15s)",
        "duration": "3-8 seconds per cycle",
        "steps": [
            "1. Fetch live prices (VIX, SPX) via yfinance",
            "2. Update features.iloc[-1] with new prices",
            "3. Detect anomalies on updated row (15 detector calls)",
            "4. Export 3 critical files:",
            "   - anomaly_report.json (current state)",
            "   - market_state.json (VIX/SPX/regime)",
            "   - dashboard_data.json (unified contract)",
            "5. Log update timestamp"
        ]
    },
    "files_updated": {
        "anomaly_report.json": "Full current state with ensemble + persistence",
        "market_state.json": "Latest VIX/SPX prices + regime",
        "dashboard_data.json": "Unified dashboard contract"
    },
    "files_not_updated": {
        "historical_anomaly_scores.json": "Frozen at training time (static)",
        "regime_statistics.json": "Frozen at training time (static)",
        "anomaly_metadata.json": "Frozen at training time (static)",
        "anomaly_feature_attribution.json": "Frozen at training time (static)",
        "vix_history.json": "Frozen at training time (static)",
        "refresh_state.pkl": "Never modified (only read)"
    },
    "critical_constraint": "Cannot recalculate full historical scores without retraining",
    "performance": {
        "cold_start": "3-8s (pickle load + initialization)",
        "warm_cycle": "1-3s (live fetch + detection + export)",
        "bottleneck": "yfinance API latency (500-2000ms)"
    }
}

# GAP 6: Live Price Fetching
LIVE_PRICE_FETCH = {
    "method": "UnifiedDataFetcher.fetch_price()",
    "implementation": "yfinance.Ticker(ticker).fast_info['last_price']",
    "supported_tickers": {
        "^VIX": "CBOE Volatility Index",
        "^GSPC": "S&P 500 Index",
        "^TNX": "10-Year Treasury Yield",
        "GLD": "Gold ETF",
        "TLT": "Long-Term Treasury ETF"
    },
    "error_handling": {
        "network_error": "Return None, log error",
        "market_closed": "Return last known price",
        "invalid_ticker": "Return None",
        "timeout": "5 second timeout, return None"
    },
    "market_hours": {
        "trading": "9:30 AM - 4:00 PM ET (Mon-Fri)",
        "pre_market": "4:00 AM - 9:30 AM ET",
        "after_hours": "4:00 PM - 8:00 PM ET",
        "behavior": "Returns last traded price if market closed"
    },
    "refresh_interval": {
        "default": "900 seconds (15 minutes)",
        "minimum": "15 seconds (API rate limits)",
        "maximum": "3600 seconds (1 hour)"
    }
}

# GAP 7: Detector Coverage Validation
DETECTOR_COVERAGE_VALIDATION = {
    "location": "IntegratedMarketSystemV4._verify_feature_coverage()",
    "purpose": "Ensure detectors have sufficient features to operate",
    "called": "After training completes in train() method",
    "validation_logic": """
    For each domain:
        1. expected_features = config.ANOMALY_FEATURE_GROUPS[domain]
        2. available_features = features.columns ∩ expected
        3. coverage_pct = (available / expected) * 100
        4. Assign status based on threshold
    """,
    "status_thresholds": {
        "> 80%": "✅ OK - Detector fully operational",
        "50-80%": "⚠️ WARNING - Detector limited capability",
        "< 50%": "❌ CRITICAL - Detector unreliable, should be disabled"
    },
    "common_missing_features": {
        "CBOE_indicators": ["VXTH", "COR1M", "COR3M - Reason: CSV files not in CBOE_Data_Archive"],
        "FRED_series": ["BAMLH0A0HYM2 (High Yield Spread) - Reason: FRED API key invalid/missing"],
        "Commodities": ["DHHNGSP (Natural Gas) - Reason: FRED series discontinued"]
    },
    "fallback_behavior": {
        "missing_features": "Filled with 0 (neutral signal)",
        "missing_domain": "Excluded from ensemble score calculation",
        "all_missing": "Training fails with ValueError"
    }
}

# GAP 8: HTML Subchart Data Loading
HTML_SUBCHART_DATA_LOADING = {
    "standard_pattern": """
    fetch('../../json_data/anomaly_report.json', {
        cache: 'no-store',
        headers: {'Cache-Control': 'no-cache'}
    })
    .then(response => response.json())
    .then(data => renderChart(data))
    .catch(error => {
        console.error('Load failed:', error);
        showErrorState();
    });
    """,
    "path_resolution": {
        "from_subchart": "../../json_data/file.json",
        "from_dashboard": "./json_data/file.json",
        "rationale": "Subcharts are 2 levels deep (Chart Modules/subcharts/)"
    },
    "cache_control": {
        "critical_headers": {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        },
        "rationale": "Ensure fresh data on every request (critical for live updates)"
    },
    "consumption_map": {
        "hero_section.html": [
            "anomaly_report.json: ensemble.score, classification",
            "historical_anomaly_scores.json: for percentile calc"
        ],
        "persistence_tracker.html": [
            "anomaly_report.json: persistence stats",
            "historical_anomaly_scores.json: last 30 days"
        ],
        "historical_analysis.html": [
            "historical_anomaly_scores.json: dates, scores, spx"
        ],
        "score_distribution.html": [
            "anomaly_report.json: current score, thresholds",
            "historical_anomaly_scores.json: all scores for histogram"
        ],
        "detector_ranking.html": [
            "anomaly_report.json: domain_anomalies"
        ],
        "forward_returns.html": [
            "anomaly_report.json: current score",
            "historical_anomaly_scores.json: scores, spx_forward_10d"
        ]
    }
}

# ============================================================================
# EXPORT DEPENDENCY CHAIN
# ============================================================================

EXPORT_DEPENDENCY_CHAIN = {
    "training_mode": {
        "sequence": [
            "1. IntegratedMarketSystemV4.train()",
            "   ├─ Fetch 15yr historical data",
            "   ├─ Engineer 234 features",
            "   ├─ Train 15 Isolation Forests",
            "   └─ Generate full historical scores",
            "",
            "2. dashboard_data_contract.export_historical_anomaly_scores()",
            "   ├─ Loop through features (detect on each row)",
            "   ├─ Build parallel arrays (dates, scores, spx, forward_10d)",
            "   └─ Export to historical_anomaly_scores.json",
            "",
            "3. VIXPredictorV4.export_anomaly_report()",
            "   ├─ Accept ensemble_scores from step 2",
            "   ├─ Calculate persistence statistics (full history)",
            "   ├─ Compute statistical thresholds (85th, 92nd, 98th percentile)",
            "   └─ Export to anomaly_report.json",
            "",
            "4. IntegratedMarketSystemV4.save_state()",
            "   └─ Serialize detectors + scalers + thresholds to refresh_state.pkl",
            "",
            "5. dashboard_orchestrator._export_all_for_refresh()",
            "   └─ Generate remaining 6 JSON files"
        ],
        "critical_flow": "historical_anomaly_scores → export_anomaly_report",
        "reason": "Persistence calculation requires full ensemble_scores array"
    },
    "cached_mode": {
        "sequence": [
            "1. IntegratedMarketSystemV4.load_state()",
            "   └─ Deserialize refresh_state.pkl",
            "",
            "2. dashboard_orchestrator._export_data()",
            "   ├─ Fetch live VIX/SPX prices",
            "   ├─ Update features.iloc[-1]",
            "   ├─ Detect anomalies on current row",
            "   └─ Export 3 files:",
            "       - anomaly_report.json (updated)",
            "       - market_state.json (updated)",
            "       - dashboard_data.json (updated)"
        ],
        "files_frozen": [
            "historical_anomaly_scores.json (static)",
            "regime_statistics.json (static)",
            "anomaly_metadata.json (static)"
        ]
    }
}


# ============================================================================
# QUICK REFERENCE
# ============================================================================

DOMAIN_NAMES = [
    "vix_mean_reversion",
    "vix_momentum", 
    "vix_regime_structure",
    "cboe_options_flow",
    "vix_spx_relationship",
    "spx_price_action",
    "spx_volatility_regime",
    "macro_rates",
    "commodities_stress",
    "cross_asset_divergence"
]

CLASSIFICATION_LEVELS = ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]

REGIME_BOUNDARIES = [0, 16.77, 24.40, 39.67, 100]
REGIME_NAMES = {0: "Low Vol", 1: "Normal", 2: "Elevated", 3: "Crisis"}


if __name__ == "__main__":
    """Run validation suite on current exports"""
    results = validate_json_exports()
    print_validation_report(results)
