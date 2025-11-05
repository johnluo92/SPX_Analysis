"""

Complete Data Lineage & Export Documentation
===============================================
Documents EXACT field mappings, filesystem structure, and data flow paths.
Last Updated: 2025-10-31

FILESYSTEM STRUCTURE:
=====================
src/
├── json_data/                          # ALL JSON exports go here
│   ├── anomaly_feature_attribution.json
│   ├── anomaly_metadata.json
│   ├── anomaly_report.json             # CRITICAL - Current anomaly state
│   ├── dashboard_data.json             # Unified data contract
│   ├── historical_anomaly_scores.json  # CRITICAL - Time series data
│   ├── market_state.json
│   ├── regime_statistics.json
│   └── vix_history.json
│
├── Chart Modules/
│   ├── anomaly.html   # Master aggregator
│   └── subcharts/                      # Individual chart components
│       ├── hero_section.html           # Reads ../../json_data/
│       ├── persistence_tracker.html    # Reads ../../json_data/
│       ├── historical_analysis.html    # Reads ../../json_data/
│       ├── score_distribution.html     # Reads ../../json_data/
│       ├── detector_ranking.html       # Reads ../../json_data/
│       └── forward_returns.html        # Reads ../../json_data/
│
├── dashboard_unified.html              # Main dashboard (reads ./json_data/)
└── dashboard_orchestrator.py           # Orchestrates exports

"""

from typing import Dict, List, Any
from dataclasses import dataclass

# ============================================================================
# SECTION 1: JSON FILE STRUCTURES WITH EXACT FIELD MAPPINGS
# ============================================================================

ANOMALY_REPORT_STRUCTURE = {
    "file": "json_data/anomaly_report.json",
    "producer": "VIXPredictorV4.export_anomaly_report()",
    "called_by": [
        "integrated_system_production.py:main()",
        "dashboard_orchestrator.py:_export_all_for_refresh()"
    ],
    "structure": {
        "timestamp": {
            "type": "string (ISO 8601)",
            "source": "datetime.now().isoformat()",
            "example": "2025-10-31T10:30:00.123456"
        },
        "ensemble": {
            "type": "object",
            "fields": {
                "score": {"type": "float", "range": "[0.0, 1.0]", "calculation": "Mean of 15 detector scores"},
                "std": {"type": "float", "source": "np.std(scores) from 15 detectors"},
                "max_anomaly": {"type": "float", "source": "np.max(scores)"},
                "min_anomaly": {"type": "float", "source": "np.min(scores)"},
                "n_detectors": {"type": "int", "value": 15}
            }
        },
        "domain_anomalies": {
            "type": "object",
            "structure": "Dict[domain_name, domain_data]",
            "domains": [
                "vix_mean_reversion", "vix_momentum", "vix_regime_structure",
                "cboe_options_flow", "vix_spx_relationship", "spx_price_action",
                "spx_volatility_regime", "macro_rates", "commodities_stress",
                "cross_asset_divergence"
            ],
            "per_domain_fields": {
                "score": {"type": "float", "range": "[0.0, 1.0]"},
                "percentile": {"type": "float", "range": "[0.0, 100.0]"},
                "level": {
                    "type": "string",
                    "values": ["NORMAL", "MODERATE", "HIGH", "CRITICAL"],
                    "classification": "Uses statistical_thresholds (data-driven)"
                }
            }
        },
        "persistence": {
            "type": "object",
            "fields": {
                "current_count": {
                    "type": "int",
                    "source": "Count of domains with score > statistical_thresholds['high']",
                    "note": "Uses data-driven threshold (not hardcoded 0.70)"
                },
                "max_possible": {"type": "int", "value": 10},
                "percentage": {"type": "float", "calculation": "current_count / max_possible"},
                "active_detectors": {"type": "array[string]", "source": "List of active domain names"},
                "historical_stats": {
                    "current_streak": {"type": "int", "meaning": "Consecutive anomaly days"},
                    "mean_duration": {"type": "float", "meaning": "Average anomaly episode length"},
                    "max_duration": {"type": "int", "meaning": "Longest anomaly episode"},
                    "total_anomaly_days": {"type": "int"},
                    "anomaly_rate": {"type": "float", "range": "[0.0, 1.0]"}
                }
            }
        },
        "top_anomalies": {
            "type": "array[object]",
            "structure": [{"name": "string", "score": "float"}],
            "sorted_by": "score (descending)",
            "length": 5
        },
        "classification": {
            "type": "object",
            "added": "2025-10-30",
            "purpose": "Data-driven thresholds computed from training distribution",
            "fields": {
                "level": {
                    "type": "string",
                    "values": ["NORMAL", "MODERATE", "HIGH", "CRITICAL"],
                    "classification": """
                    if score >= thresholds['critical']: CRITICAL
                    elif score >= thresholds['high']: HIGH
                    elif score >= thresholds['moderate']: MODERATE
                    else: NORMAL
                    """
                },
                "thresholds": {
                    "moderate": {"type": "float", "source": "85th percentile", "typical": "0.65-0.75"},
                    "high": {"type": "float", "source": "92nd percentile", "typical": "0.75-0.82"},
                    "critical": {"type": "float", "source": "98th percentile", "typical": "0.85-0.92"}
                }
            }
        }
    },
    "consumers": [
        "hero_section.html → ensemble.score, classification",
        "persistence_tracker.html → persistence stats",
        "detector_ranking.html → domain_anomalies",
        "score_distribution.html → ensemble.score, classification.thresholds"
    ]
}


HISTORICAL_ANOMALY_SCORES_STRUCTURE = {
    "file": "json_data/historical_anomaly_scores.json",
    "producer": "dashboard_data_contract.export_historical_anomaly_scores()",
    "called_by": [
        "dashboard_orchestrator.py:_export_data()",
        "dashboard_orchestrator.py:_export_all_for_refresh()"
    ],
    "critical_importance": "PRIMARY data source for time series analysis AND persistence calculation",
    "structure": {
        "dates": {
            "type": "array[string]",
            "format": "YYYY-MM-DD",
            "source": "features.index",
            "length": "3000-5000+ observations (full history)"
        },
        "ensemble_scores": {
            "type": "array[float]",
            "range": "[0.0, 1.0]",
            "calculation": "Loop through features, call detector.detect() for each row",
            "parallel_to": "dates",
            "use": "Time series charts + persistence stats calculation"
        },
        "spx_close": {
            "type": "array[float]",
            "source": "spx.values.tolist()",
            "parallel_to": "dates"
        },
        "spx_forward_10d": {
            "type": "array[float]",
            "calculation": "spx.pct_change(10).shift(-10) * 100",
            "meaning": "10-day forward return (%)",
            "parallel_to": "dates",
            "note": "Last 10 values are NaN (no future data)"
        }
    },
    "data_generation": """
    1. Load features (complete historical set)
    2. For each row: result = detector.detect(features.iloc[[i]])
    3. Collect ensemble_scores.append(result['ensemble']['score'])
    4. Calculate spx_forward_10d
    5. Export parallel arrays
    6. Pass ensemble_scores to export_anomaly_report() for persistence calculation
    """,
    "consumers": [
        "historical_analysis.html → Main time series chart",
        "score_distribution.html → Histogram & KDE",
        "forward_returns.html → Forward return analysis",
        "persistence_tracker.html → Last 30 days timeline",
        "hero_section.html → Percentile calculation",
        "export_anomaly_report() → Full history persistence stats"
    ],
    "critical_notes": [
        "MUST exist before dashboard launch",
        "Arrays MUST be parallel (same length, aligned by index)",
        "Takes 30-60 seconds to generate for full history",
        "Regenerated on every system run"
    ]
}


DASHBOARD_DATA_STRUCTURE = {
    "file": "json_data/dashboard_data.json",
    "producer": "dashboard_data_contract.export_dashboard_data()",
    "purpose": "Unified data contract for dashboard",
    "structure": {
        "version": {"type": "string", "value": "3.0"},
        "last_updated": {"type": "string (ISO 8601)"},
        "current_state": {
            "timestamp": "string (ISO 8601)",
            "vix": "float",
            "vix_regime": {"type": "int", "range": "[0-3]", "mapping": "0=Low, 1=Normal, 2=Elevated, 3=Crisis"},
            "vix_regime_name": "string",
            "days_in_regime": "int",
            "spx_close": "float",
            "anomaly_ensemble_score": {"type": "float", "range": "[0.0, 1.0]"},
            "anomaly_severity": {"type": "string", "values": ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]}
        },
        "regime_analysis": {
            "available": "bool",
            "current_regime": "int",
            "regime_name": "string",
            "days_in_regime": "int",
            "regimes": "Dict[regime_name, {id, vix_range, statistics, transitions}]"
        },
        "anomaly_analysis": {
            "available": "bool",
            "ensemble": {
                "score": "float",
                "std": "float",
                "n_detectors": "int",
                "classification": {
                    "statistical": {
                        "level": "string",
                        "p_value": "float",
                        "confidence": "float",
                        "thresholds": {"moderate": "float", "high": "float", "critical": "float"}
                    },
                    "legacy": {"level": "string", "thresholds": {}},
                    "agreement": "bool"
                }
            },
            "persistence": {"current_streak": "int", "mean_duration": "float", "max_duration": "int"},
            "domain_anomalies": "Dict[domain_name, {score, percentile, level}]",
            "top_anomalies": "array[{detector, score}]",
            "feature_attribution": "Dict[domain_name, array[{feature, value, importance}]]"
        }
    },
    "consumers": ["dashboard_unified.html", "All subchart modules (fallback)"]
}


ANOMALY_FEATURE_ATTRIBUTION_STRUCTURE = {
    "file": "json_data/anomaly_feature_attribution.json",
    "producer": "dashboard_data_contract.export_feature_attribution()",
    "purpose": "Detailed feature-level attribution for each domain",
    "structure": {
        "timestamp": "string (ISO 8601)",
        "domains": {
            "structure": "Dict[domain_name, domain_attribution]",
            "per_domain": {
                "score": "float",
                "level": "string",
                "features": {
                    "type": "array[object]",
                    "structure": [
                        {
                            "feature": "string (feature name)",
                            "value": "float (current value)",
                            "importance": "float [0.0, 1.0] (normalized contribution)"
                        }
                    ],
                    "sorted_by": "importance (descending)",
                    "length": 5
                }
            }
        }
    },
    "feature_importance_methods": {
        "primary": "SHAP (TreeExplainer)",
        "fallback": "Permutation importance"
    }
}


MARKET_STATE_STRUCTURE = {
    "file": "json_data/market_state.json",
    "producer": "IntegratedMarketSystemV4.export_json()",
    "purpose": "Complete market state snapshot",
    "structure": {
        "timestamp": "string (ISO 8601)",
        "market_data": {
            "spx": "float",
            "spx_change_today": "float (%)",
            "vix": "float",
            "vix_regime": {"id": "int", "name": "string", "range": "array[float, float]"}
        },
        "vix_structure": {
            "current_level": "float",
            "vs_ma21": "float",
            "vs_ma63": "float",
            "zscore_63d": "float",
            "percentile_252d": "float",
            "regime": "int",
            "days_in_regime": "int",
            "velocity_5d": "float"
        },
        "spx_structure": {
            "price_action": {"vs_ma50": "float", "vs_ma200": "float", "momentum_10d": "float"},
            "vix_relationship": {"corr_21d": "float", "vix_rv_ratio_21d": "float"}
        },
        "vix_predictions": {
            "regime_persistence": {"probability": "float", "expected_duration": "float"},
            "transition_risk": {"elevated": "bool", "direction": "string", "confidence": "string"}
        },
        "anomaly_analysis": {
            "ensemble": {"score": "float", "severity": "string", "interpretation": "string"},
            "domain_anomalies": "object",
            "persistence": "object",
            "data_availability": {"cboe_indicators": "bool", "overall_confidence": "string"}
        }
    }
}


REGIME_STATISTICS_STRUCTURE = {
    "file": "json_data/regime_statistics.json",
    "producer": "VIXPredictorV4.export_regime_statistics()",
    "structure": {
        "metadata": {"total_trading_days": "int", "start_date": "string", "end_date": "string"},
        "regime_boundaries": {"value": "[0, 16.77, 24.40, 39.67, 100]"},
        "regimes": {
            "type": "array[object]",
            "length": 4,
            "per_regime": {
                "id": "int [0-3]",
                "name": "string",
                "vix_range": "array[float, float]",
                "statistics": {
                    "total_days": "int",
                    "frequency": "float",
                    "mean_duration": "float",
                    "median_duration": "float",
                    "max_duration": "int"
                },
                "transitions_5d": {
                    "persistence": {"probability": "float", "observations": "int"},
                    "to_other_regimes": "Dict[regime_id, {probability, observations}]"
                }
            }
        }
    },
    "calculation_method": """
    1. Classify VIX into regimes using pd.cut()
    2. Identify regime episodes
    3. Calculate duration statistics
    4. Look ahead 5 days for transition probabilities
    """
}


VIX_HISTORY_STRUCTURE = {
    "file": "json_data/vix_history.json",
    "producer": "VIXPredictorV4.export_vix_history()",
    "structure": {
        "metadata": {"total_days": "int", "start_date": "string", "end_date": "string"},
        "data": {
            "type": "array[object]",
            "structure": [{"date": "string (YYYY-MM-DD)", "vix": "float"}],
            "source": "vix_history_all Series (1990-01-02 to present)"
        }
    }
}


ANOMALY_METADATA_STRUCTURE = {
    "file": "json_data/anomaly_metadata.json",
    "producer": "dashboard_orchestrator._export_anomaly_metadata()",
    "purpose": "Human-readable descriptions of anomaly domains",
    "structure": {
        "timestamp": "string (ISO 8601)",
        "version": "string",
        "domains": {
            "structure": "Dict[domain_name, domain_description]",
            "per_domain": {
                "name": "string (human-readable)",
                "description": "string",
                "key_signals": "array[string]"
            }
        }
    }
}


# ============================================================================
# SECTION 2: SUBCHART DATA CONSUMPTION PATTERNS
# ============================================================================

SUBCHART_CONSUMPTION = {
    "hero_section.html": {
        "reads_from": ["anomaly_report.json", "historical_anomaly_scores.json"],
        "data_paths": {
            "ensemble_score": "anomaly_report.json:ensemble.score",
            "detector_count": "anomaly_report.json:persistence.current_count",
            "current_streak": "anomaly_report.json:persistence.historical_stats.current_streak",
            "historical_percentile": "calculated from historical_anomaly_scores.json:ensemble_scores"
        },
        "visualization": "Gauge with ensemble score + key metrics"
    },
    
    "persistence_tracker.html": {
        "reads_from": ["anomaly_report.json", "historical_anomaly_scores.json"],
        "data_paths": {
            "last_30_days": "historical_anomaly_scores.json:ensemble_scores[-30:]",
            "last_30_dates": "historical_anomaly_scores.json:dates[-30:]",
            "persistence_stats": "anomaly_report.json:persistence.historical_stats"
        },
        "visualization": "30-day timeline with anomaly classification"
    },
    
    "historical_analysis.html": {
        "reads_from": ["historical_anomaly_scores.json"],
        "data_paths": {
            "dates": "historical_anomaly_scores.json:dates",
            "ensemble_scores": "historical_anomaly_scores.json:ensemble_scores",
            "spx_close": "historical_anomaly_scores.json:spx_close"
        },
        "visualization": "Dual synchronized charts (SPX + Anomaly Score)",
        "critical": "PRIMARY consumer of historical_anomaly_scores.json"
    },
    
    "score_distribution.html": {
        "reads_from": ["anomaly_report.json", "historical_anomaly_scores.json"],
        "data_paths": {
            "current_score": "anomaly_report.json:ensemble.score",
            "historical_scores": "historical_anomaly_scores.json:ensemble_scores",
            "thresholds": "anomaly_report.json:classification.thresholds"
        },
        "visualization": "Histogram with KDE overlay, threshold lines, current score marker"
    },
    
    "detector_ranking.html": {
        "reads_from": ["anomaly_report.json"],
        "data_paths": {
            "domain_anomalies": "anomaly_report.json:domain_anomalies"
        },
        "visualization": "Horizontal bar chart of domain scores (top 10)"
    },
    
    "forward_returns.html": {
        "reads_from": ["anomaly_report.json", "historical_anomaly_scores.json"],
        "data_paths": {
            "current_score": "anomaly_report.json:ensemble.score",
            "historical_scores": "historical_anomaly_scores.json:ensemble_scores",
            "spx_forward_returns": "historical_anomaly_scores.json:spx_forward_10d"
        },
        "analysis_method": """
        1. Find historical periods with similar scores (current ± 0.10)
        2. Get forward returns for those periods
        3. Calculate median, mean, std, win rate, tail risk
        4. Display histogram of forward returns distribution
        """
    }
}


# ============================================================================
# SECTION 3: DATA FLOW PATHS
# ============================================================================

STATISTICAL_THRESHOLDS_FLOW = {
    "name": "Statistical Anomaly Thresholds (Data-Driven)",
    "added": "2025-10-30",
    "purpose": "Replace hardcoded thresholds with percentile-based thresholds",
    "flow": [
        {
            "stage": "1. Training - Compute Thresholds",
            "location": "MultiDimensionalAnomalyDetector.train()",
            "processing": """
            training_scores = [detector.detect(features.iloc[[i]]) for i in range(len(features))]
            self.statistical_thresholds = {
                'moderate': np.percentile(training_scores, 85),  # 85th percentile
                'high': np.percentile(training_scores, 92),      # 92nd percentile
                'critical': np.percentile(training_scores, 98)   # 98th percentile
            }
            """,
            "typical_values": {"moderate": "0.65-0.75", "high": "0.75-0.82", "critical": "0.85-0.92"}
        },
        {
            "stage": "2. Retrieve in Export",
            "location": "VIXPredictorV4.export_anomaly_report()",
            "processing": """
            stat_thresholds = (detector.statistical_thresholds 
                             if hasattr(detector, 'statistical_thresholds') 
                             else {'moderate': 0.70, 'high': 0.78, 'critical': 0.88})
            """
        },
        {
            "stage": "3. Classify Current Score",
            "processing": """
            if score >= stat_thresholds['critical']: level = 'CRITICAL'
            elif score >= stat_thresholds['high']: level = 'HIGH'
            elif score >= stat_thresholds['moderate']: level = 'MODERATE'
            else: level = 'NORMAL'
            """
        },
        {
            "stage": "4. Export to JSON",
            "outputs": {
                "anomaly_report.json": {
                    "classification.level": "string",
                    "classification.thresholds": "Dict[moderate, high, critical]"
                }
            }
        },
        {
            "stage": "5. Dashboard Consumption",
            "consumers": {
                "hero_section.html": "Display classification badge",
                "score_distribution.html": "Draw threshold lines on histogram",
                "persistence_tracker.html": "Color-code historical scores"
            }
        }
    ],
    "advantages_over_legacy": [
        "Adapts to actual data distribution",
        "Maintains consistent percentile-based classification",
        "Updates when retrained with new data",
        "More robust to distribution shifts"
    ]
}


PERSISTENCE_CALCULATION_FLOW = {
    "name": "Anomaly Persistence Statistics",
    "critical_update": "Now uses FULL historical data instead of 100-observation buffer",
    "flow": [
        {
            "stage": "1. Historical Score Generation",
            "location": "export_historical_anomaly_scores()",
            "processing": """
            ensemble_scores = []
            for i in range(len(features)):
                result = detector.detect(features.iloc[[i]], verbose=False)
                ensemble_scores.append(result['ensemble']['score'])
            """,
            "output": "ensemble_scores array (3000-5000+ observations)"
        },
        {
            "stage": "2. Persistence Stats Calculation",
            "location": "MultiDimensionalAnomalyDetector.calculate_historical_persistence_stats()",
            "inputs": ["ensemble_scores: np.ndarray", "threshold: float (defaults to statistical_thresholds['high'])"],
            "processing": """
            1. Identify anomalous days: is_anomalous = (ensemble_scores >= threshold)
            2. Track consecutive True values as streaks
            3. Calculate: current_streak, mean_duration, max_duration, 
                         total_anomaly_days, anomaly_rate, num_episodes
            """,
            "outputs": {
                "current_streak": "int (ongoing anomaly days)",
                "mean_duration": "float (average episode length)",
                "max_duration": "int (longest episode)",
                "total_anomaly_days": "int",
                "anomaly_rate": "float (% of days anomalous)",
                "num_episodes": "int"
            }
        },
        {
            "stage": "3. Export to JSON",
            "location": "VIXPredictorV4.export_anomaly_report()",
            "signature": "export_anomaly_report(filepath, anomaly_result=None, historical_ensemble_scores=None)",
            "logic": """
            if historical_ensemble_scores is not None:
                persistence_stats = detector.calculate_historical_persistence_stats(
                    historical_ensemble_scores
                )
                # Uses FULL history
            else:
                persistence_stats = detector.get_persistence_stats()
                # Fallback to 100-observation buffer
            """
        }
    ],
    "before_vs_after": {
        "before": {
            "data_source": "self.anomaly_history (last 100 observations)",
            "typical_window": "~3-4 months",
            "issue": "Statistics change as old data drops from buffer"
        },
        "after": {
            "data_source": "historical_ensemble_scores (all observations)",
            "typical_window": "Full history (3000-5000+ observations)",
            "benefit": "Stable, accurate long-term statistics"
        }
    }
}


ANOMALY_SCORE_FLOW = {
    "name": "Ensemble Anomaly Score - Complete Pipeline",
    "stages": [
        {"stage": "1. Raw Data", "location": "UnifiedDataFetcher", 
         "outputs": ["SPX OHLCV", "VIX Close", "FRED data", "CBOE indicators"]},
        
        {"stage": "2. Feature Engineering", "location": "UnifiedFeatureEngine.build_complete_features()", 
         "outputs": "features DataFrame (100+ columns)"},
        
        {"stage": "3. Detector Training", "location": "MultiDimensionalAnomalyDetector.train()", 
         "outputs": ["15 trained Isolation Forests", "Feature importances", "Statistical thresholds"]},
        
        {"stage": "4. Current Detection", "location": "detector.detect(features.iloc[[-1]])", 
         "outputs": {"ensemble": {"score": "float", "std": "float"}, "domain_anomalies": "dict"}},
        
        {"stage": "5. Historical Generation", "location": "export_historical_anomaly_scores()", 
         "outputs": "ensemble_scores array (all historical)"},
        
        {"stage": "6. JSON Export", "outputs": [
             "anomaly_report.json (current state)",
             "historical_anomaly_scores.json (time series)",
             "dashboard_data.json (unified)"
         ]},
        
        {"stage": "7. Dashboard Display", "consumers": [
             "hero_section.html", "historical_analysis.html", 
             "score_distribution.html", "dashboard_unified.html"
         ]}
    ]
}


# ============================================================================
# SECTION 4: AUTO-REFRESH DATA FLOW
# ============================================================================

AUTO_REFRESH_FLOW = {
    "trigger": "dashboard_orchestrator.py --auto-refresh",
    "interval": "15 seconds (configurable with --refresh-interval)",
    "process": [
        {"step": "1. Fetch Live Prices", "actions": ["Fetch live VIX", "Fetch live SPX", "Update latest feature row"]},
        {"step": "2. Recalculate Scores", "action": "detector.detect(features.iloc[[-1]])"},
        {"step": "3. Export Updated Data", "files": ["market_state.json", "anomaly_report.json", "dashboard_data.json"]},
        {"step": "4. Dashboard Auto-Reload", "mechanism": "Subcharts poll JSON files (no-cache headers)"}
    ]
}


# ============================================================================
# SECTION 5: CRITICAL DEPENDENCIES
# ============================================================================

CRITICAL_DEPENDENCIES = {
    "historical_anomaly_scores.json": {
        "status": "CRITICAL",
        "reason": "Required by 5 of 6 subcharts + persistence calculation",
        "dependents": [
            "historical_analysis.html (BREAKS without this)",
            "score_distribution.html",
            "forward_returns.html",
            "persistence_tracker.html",
            "hero_section.html",
            "export_anomaly_report() persistence stats"
        ],
        "generation_time": "30-60 seconds for full history",
        "must_exist_before": "Dashboard launch"
    },
    
    "anomaly_report.json": {
        "status": "CRITICAL",
        "reason": "Current state required by all subcharts",
        "dependents": "All 6 subcharts",
        "update_frequency": "Every 15 seconds during auto-refresh"
    },
    
    "parallel_arrays": {
        "status": "CRITICAL",
        "requirement": "Arrays in historical_anomaly_scores.json MUST be parallel",
        "arrays": ["dates", "ensemble_scores", "spx_close", "spx_forward_10d"],
        "validation": "len(dates) == len(ensemble_scores) == len(spx_close) == len(spx_forward_10d)",
        "breaking_condition": "Mismatched lengths will crash charts"
    },
    
    "statistical_thresholds": {
        "status": "IMPORTANT",
        "reason": "Data-driven classification",
        "location": "detector.statistical_thresholds",
        "exported_to": ["anomaly_report.json:classification.thresholds"],
        "dashboard_usage": ["hero_section.html", "score_distribution.html", "persistence_tracker.html"],
        "fallback": "Uses legacy thresholds {0.70, 0.78, 0.88} if not available"
    },
    
    "filesystem_structure": {
        "status": "CRITICAL",
        "requirement": "json_data/ must be 2 levels up from subcharts/",
        "structure": "src/json_data/ ← src/Chart Modules/subcharts/ reads ../../json_data/",
        "breaking_change": "Moving json_data/ breaks all subchart data loading"
    }
}


# ============================================================================
# SECTION 6: FILE SYSTEM MAP
# ============================================================================

FILE_SYSTEM_MAP = {
    "json_data/": {
        "purpose": "All JSON exports",
        "accessed_by": {
            "dashboard_unified.html": "./json_data/",
            "Chart Modules/subcharts/*.html": "../../json_data/"
        },
        "files": {
            "anomaly_report.json": {
                "producer": "VIXPredictorV4.export_anomaly_report()",
                "size": "~5-10 KB",
                "update_frequency": "Every system run + auto-refresh",
                "critical": True
            },
            "historical_anomaly_scores.json": {
                "producer": "export_historical_anomaly_scores()",
                "size": "~50-100 KB",
                "update_frequency": "Every system run + auto-refresh",
                "generation_time": "30-60 seconds",
                "critical": True
            },
            "dashboard_data.json": {
                "producer": "export_dashboard_data()",
                "size": "~15-30 KB",
                "update_frequency": "Every system run + auto-refresh"
            },
            "anomaly_feature_attribution.json": {
                "producer": "export_feature_attribution()",
                "size": "~5-10 KB",
                "update_frequency": "Every system run"
            },
            "market_state.json": {
                "producer": "IntegratedMarketSystemV4.export_json()",
                "size": "~10-15 KB",
                "update_frequency": "Every system run + auto-refresh"
            },
            "regime_statistics.json": {
                "producer": "VIXPredictorV4.export_regime_statistics()",
                "size": "~3-5 KB",
                "update_frequency": "Every training run (static)"
            },
            "vix_history.json": {
                "producer": "VIXPredictorV4.export_vix_history()",
                "size": "~200-500 KB",
                "update_frequency": "Every training run (static)"
            },
            "anomaly_metadata.json": {
                "producer": "dashboard_orchestrator._export_anomaly_metadata()",
                "size": "~2-3 KB",
                "update_frequency": "Every system run (static)"
            }
        }
    },
    
    "Chart Modules/subcharts/": {
        "purpose": "Individual chart components (loaded as iframes)",
        "data_path": "../../json_data/",
        "files": {
            "hero_section.html": "Ensemble score gauge + key metrics",
            "persistence_tracker.html": "30-day anomaly timeline",
            "historical_analysis.html": "SPX + Anomaly score time series (PRIMARY consumer)",
            "score_distribution.html": "Histogram + KDE of scores",
            "detector_ranking.html": "Bar chart of domain scores",
            "forward_returns.html": "Forward return analysis"
        }
    },
    
    "Python Modules": {
        "anomaly_system.py": {
            "class": "MultiDimensionalAnomalyDetector",
            "key_methods": [
                "train() → Trains 15 Isolation Forests + computes statistical thresholds",
                "detect() → Returns anomaly scores for current state",
                "calculate_historical_persistence_stats() → Full history persistence stats",
                "get_persistence_stats() → Legacy 100-observation buffer method",
                "_calculate_feature_importance() → SHAP/permutation"
            ]
        },
        "vix_predictor_v2.py": {
            "class": "VIXPredictorV4",
            "key_methods": [
                "train_with_features() → Main training entry point",
                "export_regime_statistics() → regime_statistics.json",
                "export_vix_history() → vix_history.json",
                "export_anomaly_report() → anomaly_report.json (with persistence stats)"
            ]
        },
        "dashboard_data_contract.py": {
            "key_functions": [
                "export_dashboard_data() → dashboard_data.json",
                "export_historical_anomaly_scores() → historical_anomaly_scores.json",
                "export_feature_attribution() → anomaly_feature_attribution.json"
            ]
        },
        "integrated_system_production.py": {
            "class": "IntegratedMarketSystemV4",
            "key_methods": [
                "train() → Complete system training",
                "export_json() → market_state.json",
                "get_market_state() → Current state dict"
            ]
        },
        "dashboard_orchestrator.py": {
            "class": "DashboardOrchestrator",
            "key_methods": [
                "run() → Full training + export + launch dashboard",
                "_export_data() → Calls all export functions",
                "_export_anomaly_metadata() → anomaly_metadata.json",
                "_start_refresh() → Auto-refresh loop (15s interval)"
            ]
        }
    }
}


# ============================================================================
# SECTION 7: INTEGRATION CHECKLIST
# ============================================================================

INTEGRATION_CHECKLIST = {
    "statistical_thresholds": {
        "status": "✅ IMPLEMENTED",
        "steps": [
            "✅ Compute thresholds during training (85th, 92nd, 98th percentiles)",
            "✅ Store in detector.statistical_thresholds",
            "✅ Export to anomaly_report.json:classification",
            "⚠️  TODO: Update dashboard to read classification.thresholds",
            "⚠️  TODO: Use thresholds for threshold lines in charts"
        ]
    },
    
    "persistence_calculation": {
        "status": "✅ IMPLEMENTED",
        "steps": [
            "✅ New method: calculate_historical_persistence_stats()",
            "✅ Pass historical_ensemble_scores to export_anomaly_report()",
            "✅ Export full history stats to anomaly_report.json",
            "✅ Dashboard reads persistence.historical_stats"
        ]
    },
    
    "dashboard_updates_needed": {
        "status": "⚠️  PARTIALLY COMPLETE",
        "tasks": [
            "⚠️  hero_section.html: Read classification.level for badge color",
            "⚠️  score_distribution.html: Draw threshold lines from classification.thresholds",
            "⚠️  persistence_tracker.html: Use classification.thresholds for color coding",
            "✅ All charts: Already reading ensemble scores correctly"
        ]
    }
}


# ============================================================================


# ============================================================================
# SECTION: HISTORICAL ENSEMBLE SCORES ARCHITECTURE (Added 2025-10-31)
# ============================================================================

HISTORICAL_ENSEMBLE_SCORES_ARCHITECTURE = {
    "status": "âœ… IMPLEMENTED",
    "added": "2025-10-31",
    "storage": "VIXPredictorV4.historical_ensemble_scores",
    "type": "np.ndarray",
    "size": "3000-5000+ observations (full training history)",
    "generated_by": "VIXPredictorV4._generate_historical_ensemble_scores()",
    "when": "During train_with_features() - after detector training",
    "generation_time": "30-60 seconds for full history",
    
    "processing": """
    1. Loop through ALL features in batches (batch_size=100)
    2. Call detector.detect() for each observation
    3. Extract ensemble score from result['ensemble']['score']
    4. Store as numpy array: self.historical_ensemble_scores = np.array(scores)
    5. Validate: Must have same length as features
    """,
    
    "usage": {
        "export_anomaly_report()": {
            "purpose": "Calculate persistence stats from full history",
            "call": "detector.calculate_historical_persistence_stats(self.historical_ensemble_scores)"
        },
        "export_historical_anomaly_scores()": {
            "purpose": "Export time series data for dashboard",
            "output": "historical_anomaly_scores.json"
        },
        "dashboard_data_contract": {
            "purpose": "Unified dashboard data with persistence",
            "call": "Passed via export_dashboard_data(anomaly_result=...)"
        }
    },
    
    "critical_validation": {
        "checkpoint": "After _generate_historical_ensemble_scores()",
        "check": """
        if self.historical_ensemble_scores is None:
            raise RuntimeError("Historical scores generation failed!")
        """,
        "ensures": "No exports proceed without historical data"
    },
    
    "data_flow": [
        "1. train_with_features() trains detectors",
        "2. _generate_historical_ensemble_scores() creates full history",
        "3. historical_ensemble_scores stored as instance variable",
        "4. All exports use this cached array (not regenerated)",
        "5. Persistence calculation uses full array for accuracy"
    ],
    
    "advantages_over_buffer": [
        "Full history (not limited to last 100 observations)",
        "Stable statistics that don't drift with new data",
        "One-time calculation (cached for all exports)",
        "Accurate long-term persistence patterns",
        "No data loss from buffer rotation"
    ]
}



# ============================================================================
# SECTION: PERSISTENCE CALCULATION V2 (Updated 2025-10-31)
# ============================================================================

PERSISTENCE_CALCULATION_V2 = {
    "method": "calculate_historical_persistence_stats()",
    "location": "MultiDimensionalAnomalyDetector (anomaly_system.py)",
    "signature": "calculate_historical_persistence_stats(ensemble_scores: np.ndarray, threshold: float = None)",
    "status": "âœ… ACTIVE METHOD",
    
    "replaces": {
        "old_method": "get_persistence_stats()",
        "old_data_source": "self.anomaly_history (100-observation circular buffer)",
        "migration_date": "2025-10-30",
        "reason": "Buffer was too short for accurate long-term statistics"
    },
    
    "inputs": {
        "ensemble_scores": {
            "type": "np.ndarray",
            "source": "VIXPredictorV4.historical_ensemble_scores",
            "length": "Full training history (3000-5000+ observations)",
            "range": "[0.0, 1.0]"
        },
        "threshold": {
            "type": "float",
            "default": "detector.statistical_thresholds['high']",
            "fallback": "0.78 if thresholds not available",
            "purpose": "Define anomaly boundary for streak calculation"
        }
    },
    
    "algorithm": """
    1. Identify anomalous periods: is_anomalous = (ensemble_scores >= threshold)
    2. Track consecutive True values as streaks:
       - current_streak = 0
       - for each day:
           if anomalous: current_streak += 1
           else: 
               if current_streak > 0: streaks.append(current_streak)
               current_streak = 0
    3. Calculate statistics:
       - mean_duration = mean(streaks)
       - max_duration = max(streaks)
       - total_anomaly_days = sum(is_anomalous)
       - anomaly_rate = total_anomaly_days / len(ensemble_scores)
       - num_episodes = len(streaks)
    4. Return dict with all stats
    """,
    
    "outputs": {
        "current_streak": {
            "type": "int",
            "meaning": "Ongoing consecutive anomaly days (0 if not currently anomalous)",
            "usage": "Hero section - persistence indicator"
        },
        "mean_duration": {
            "type": "float",
            "meaning": "Average length of anomaly episodes",
            "usage": "Expected duration when anomaly starts"
        },
        "max_duration": {
            "type": "int",
            "meaning": "Longest anomaly episode in history",
            "usage": "Risk assessment - worst case scenario"
        },
        "total_anomaly_days": {
            "type": "int",
            "meaning": "Total days above threshold in full history",
            "usage": "Historical context"
        },
        "anomaly_rate": {
            "type": "float",
            "range": "[0.0, 1.0]",
            "meaning": "Percentage of days that were anomalous",
            "usage": "Base rate for probability assessment"
        },
        "num_episodes": {
            "type": "int",
            "meaning": "Number of distinct anomaly episodes",
            "usage": "Frequency of anomaly events"
        },
        "threshold_used": {
            "type": "float",
            "meaning": "Threshold value used for calculation",
            "usage": "Reproducibility and validation"
        }
    },
    
    "call_sites": [
        {
            "file": "vix_predictor_v2.py",
            "function": "export_anomaly_report()",
            "code": """
            persistence_stats = self.anomaly_detector.calculate_historical_persistence_stats(
                self.historical_ensemble_scores
            )
            """
        },
        {
            "file": "dashboard_data_contract.py",
            "function": "export_dashboard_data()",
            "code": """
            if hasattr(vix_predictor, 'historical_ensemble_scores'):
                persistence_stats = vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
                    vix_predictor.historical_ensemble_scores
                )
            """
        }
    ],
    
    "comparison_with_legacy": {
        "before": {
            "method": "get_persistence_stats()",
            "data_source": "Last 100 observations in circular buffer",
            "window": "~3-4 months",
            "issue": "Statistics drift as old data rotates out"
        },
        "after": {
            "method": "calculate_historical_persistence_stats()",
            "data_source": "Full historical ensemble_scores array",
            "window": "Complete training history (15+ years possible)",
            "benefit": "Stable, accurate long-term statistics"
        }
    }
}



# ============================================================================
# SECTION: EXPORT DEPENDENCY CHAIN (Critical Ordering)
# ============================================================================

EXPORT_DEPENDENCY_CHAIN = {
    "purpose": "Documents critical ordering requirements for data exports",
    "importance": "âœ… CRITICAL - Breaking this order causes export failures",
    
    "ordered_sequence": [
        {
            "step": 1,
            "action": "train_with_features()",
            "location": "VIXPredictorV4",
            "produces": [
                "self.features (complete feature set)",
                "self.anomaly_detector (trained 15 detectors)",
                "self.historical_ensemble_scores (full history array)"
            ],
            "critical_checkpoint": """
            if self.historical_ensemble_scores is None:
                raise RuntimeError("Historical scores generation failed!")
            """,
            "failure_mode": "All subsequent exports will fail without this"
        },
        {
            "step": 2,
            "action": "detector.detect(features.iloc[[-1]])",
            "produces": "anomaly_result (current state)",
            "cached_as": "cached_anomaly_result in system",
            "usage": "Passed to all export functions to avoid recalculation"
        },
        {
            "step": 3,
            "action": "export_anomaly_report()",
            "location": "VIXPredictorV4",
            "requires": [
                "self.historical_ensemble_scores (must exist)",
                "anomaly_result (current detection)"
            ],
            "produces": "anomaly_report.json",
            "includes": [
                "Current ensemble score + domain scores",
                "Persistence stats from FULL history",
                "Statistical thresholds",
                "Top anomalies"
            ]
        },
        {
            "step": 4,
            "action": "export_historical_anomaly_scores()",
            "location": "dashboard_data_contract.py",
            "requires": [
                "vix_predictor.features (full history)",
                "vix_predictor.anomaly_detector (trained)"
            ],
            "produces": "historical_anomaly_scores.json",
            "processing": "Loops through features, generates ensemble_scores array",
            "note": "Independent of step 3 (doesn't need anomaly_report.json)"
        },
        {
            "step": 5,
            "action": "export_dashboard_data()",
            "location": "dashboard_data_contract.py",
            "requires": [
                "vix_predictor (trained system)",
                "anomaly_result (from step 2)",
                "persistence_stats (calculated in this function)"
            ],
            "produces": "dashboard_data.json",
            "processing": """
            - Calculates persistence from historical_ensemble_scores
            - Builds unified data contract
            - Includes current state + regime analysis + anomaly analysis
            """
        }
    ],
    
    "parallel_exports": {
        "can_run_simultaneously": [
            "export_regime_statistics()",
            "export_vix_history()",
            "export_market_state()",
            "export_anomaly_metadata()"
        ],
        "note": "These don't depend on persistence calculation"
    },
    
    "auto_refresh_optimization": {
        "description": "During 15-second refresh cycles",
        "sequence": [
            "1. Fetch live VIX/SPX",
            "2. Update features.iloc[-1]",
            "3. Recalculate detector.detect() for current state",
            "4. Export with CACHED persistence stats (not recalculated)",
            "5. Only anomaly_report.json + dashboard_data.json updated"
        ],
        "optimization": "Persistence stats from training are reused (not recalculated every 15s)"
    },
    
    "error_handling": {
        "missing_historical_scores": {
            "error": "ValueError: Historical ensemble scores not generated",
            "solution": "Re-run training with train_with_features()",
            "prevention": "Checkpoint after _generate_historical_ensemble_scores()"
        },
        "length_mismatch": {
            "error": "ValueError: Score/feature length mismatch",
            "cause": "Batch processing error in _generate_historical_ensemble_scores()",
            "solution": "Regenerate scores with fixed batch logic"
        }
    }
}



# ============================================================================
# SECTION: STATISTICAL THRESHOLDS USAGE PATTERN
# ============================================================================

STATISTICAL_THRESHOLDS_USAGE = {
    "storage_location": "MultiDimensionalAnomalyDetector.statistical_thresholds",
    "type": "Dict[str, float]",
    "computed_during": "detector.train() - after training all 15 detectors",
    "structure": {
        "moderate": "float - 85th percentile of training scores",
        "high": "float - 92nd percentile of training scores",
        "critical": "float - 98th percentile of training scores"
    },
    
    "computation_method": """
    1. Collect training_ensemble_scores during detector.train()
    2. For each training observation:
       - result = detector.detect(features.iloc[[i]])
       - training_scores.append(result['ensemble']['score'])
    3. Calculate percentiles:
       - moderate: np.percentile(training_scores, 85)
       - high: np.percentile(training_scores, 92)
       - critical: np.percentile(training_scores, 98)
    4. Store in detector.statistical_thresholds
    """,
    
    "typical_values": {
        "moderate": "0.65 - 0.75",
        "high": "0.75 - 0.82",
        "critical": "0.85 - 0.92",
        "note": "Values adapt to actual data distribution"
    },
    
    "standardized_access_pattern": """
    # âœ… CORRECT PATTERN (use everywhere):
    if hasattr(detector, 'statistical_thresholds') and detector.statistical_thresholds:
        thresholds = detector.statistical_thresholds
    else:
        # Fallback for backward compatibility
        thresholds = {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}
    """,
    
    "usage_locations": {
        "export_anomaly_report()": {
            "purpose": "Classify current ensemble score",
            "logic": """
            if score >= thresholds['critical']: level = 'CRITICAL'
            elif score >= thresholds['high']: level = 'HIGH'
            elif score >= thresholds['moderate']: level = 'MODERATE'
            else: level = 'NORMAL'
            """
        },
        "export_dashboard_data()": {
            "purpose": "Count active detectors",
            "logic": """
            active_detectors = [
                name for name, data in domain_anomalies.items()
                if data['score'] > thresholds['high']
            ]
            """
        },
        "_build_current_state()": {
            "purpose": "Determine anomaly severity",
            "output": "current_state.anomaly_severity"
        },
        "calculate_historical_persistence_stats()": {
            "purpose": "Define anomaly boundary for streak calculation",
            "default": "thresholds['high']"
        },
        "dashboard_orchestrator._refresh_loop()": {
            "purpose": "Real-time severity classification",
            "frequency": "Every 15 seconds during auto-refresh"
        }
    },
    
    "deprecated_patterns": {
        "config.ANOMALY_THRESHOLDS": {
            "status": "âš ï¸  DEPRECATED",
            "reason": "Hardcoded values don't adapt to data distribution",
            "migration": "Use detector.statistical_thresholds instead"
        },
        "config.LEGACY_ANOMALY_THRESHOLDS": {
            "status": "âš ï¸  DEPRECATED",
            "use": "Fallback only when detector unavailable"
        }
    },
    
    "advantages": [
        "Data-driven (adapts to actual score distribution)",
        "Maintains consistent percentile-based classification",
        "Updates automatically when retrained with new data",
        "More robust to distribution shifts over time",
        "Reduces false positives from hardcoded thresholds"
    ]
}



# ============================================================================
# SECTION: TRAINING CONTROL FLAG (Production Optimization)
# ============================================================================

TRAINING_CONTROL_FLAG = {
    "flag": "config.ENABLE_TRAINING",
    "type": "bool",
    "default": "False (training disabled by default)",
    "purpose": "Skip expensive training in production/refresh scenarios",
    
    "behavior": {
        "True": {
            "action": "Full training pipeline",
            "includes": [
                "Fetch historical data (15 years)",
                "Train 15 Isolation Forest detectors",
                "Generate feature importances (SHAP/permutation)",
                "Calculate statistical thresholds",
                "Generate historical ensemble scores"
            ],
            "duration": "60-120 seconds",
            "use_case": "Initial setup, model updates, retraining"
        },
        "False": {
            "action": "Skip training, use cached state",
            "includes": [
                "Load existing features/detector state",
                "Fetch only latest data",
                "Export JSONs with current data"
            ],
            "duration": "5-10 seconds",
            "use_case": "Auto-refresh, dashboard updates, quick exports"
        }
    },
    
    "implementation": {
        "location": "dashboard_orchestrator.py:_train_system()",
        "logic": """
        if not ENABLE_TRAINING:
            print("Training DISABLED (config.ENABLE_TRAINING = False)")
            print("Using cached/existing model state")
            system.trained = True  # Mark as trained without training
        else:
            print("Training complete system...")
            system.train(years=years, real_time_vix=True, verbose=False)
        """,
        "override": "Command line --skip-training flag takes precedence"
    },
    
    "command_line_interaction": {
        "flag": "--skip-training",
        "precedence": "Command line > config.ENABLE_TRAINING",
        "logic": """
        # Command line --skip-training=True forces skip
        # Else use config.ENABLE_TRAINING value
        skip_training = args.skip_training or not ENABLE_TRAINING
        """
    },
    
    "important_note": {
        "warning": "Even with training disabled, system still needs to:",
        "requirements": [
            "Load/fetch latest market data",
            "Update features with live prices",
            "Run detector.detect() for current state",
            "Export all JSON files"
        ],
        "does_not_skip": "Data fetching, feature updates, JSON exports"
    },
    
    "use_cases": {
        "development": {
            "enable_training": "True",
            "reason": "Test full pipeline, validate changes"
        },
        "production_initial": {
            "enable_training": "True",
            "reason": "First-time model training"
        },
        "production_refresh": {
            "enable_training": "False",
            "reason": "15-second updates don't need retraining"
        },
        "debugging": {
            "enable_training": "False",
            "reason": "Faster iteration on export logic"
        }
    }
}



# ============================================================================
# SECTION: VALIDATION CHECKPOINTS (Error Prevention)
# ============================================================================

VALIDATION_CHECKPOINTS = {
    "purpose": "Critical validation points to prevent export failures",
    
    "checkpoint_1_after_training": {
        "location": "VIXPredictorV4.train_with_features()",
        "validation": """
        if self.historical_ensemble_scores is None:
            raise RuntimeError("âŒ CRITICAL: Historical scores generation failed!")
        """,
        "ensures": "historical_ensemble_scores exists before any exports",
        "failure_consequence": "All persistence calculations will fail"
    },
    
    "checkpoint_2_before_export": {
        "location": "VIXPredictorV4.export_anomaly_report()",
        "validation": """
        if self.historical_ensemble_scores is None:
            raise ValueError("Historical ensemble scores not generated. Re-run training.")
        """,
        "ensures": "Cannot export without historical data",
        "user_action": "Re-run train_with_features()"
    },
    
    "checkpoint_3_array_lengths": {
        "location": "VIXPredictorV4._generate_historical_ensemble_scores()",
        "validation": """
        if len(ensemble_scores) != len(self.features):
            raise ValueError(f"Score/feature mismatch: {len(ensemble_scores)} vs {len(self.features)}")
        """,
        "ensures": "Parallel arrays match in length",
        "failure_mode": "Batch processing error"
    },
    
    "checkpoint_4_threshold_availability": {
        "location": "calculate_historical_persistence_stats()",
        "validation": """
        if threshold is None:
            if hasattr(self, 'statistical_thresholds') and self.statistical_thresholds:
                threshold = self.statistical_thresholds['high']
            else:
                threshold = 0.78  # Absolute fallback
        """,
        "ensures": "Always have valid threshold for persistence calculation",
        "graceful_degradation": "Falls back to legacy threshold"
    },
    
    "checkpoint_5_parallel_arrays": {
        "location": "export_historical_anomaly_scores()",
        "validation": """
        assert len(dates) == len(ensemble_scores) == len(spx_close) == len(spx_forward_10d)
        """,
        "ensures": "All arrays in historical_anomaly_scores.json are parallel",
        "dashboard_requirement": "Charts will crash if arrays misaligned"
    },
    
    "checkpoint_6_json_validity": {
        "location": "dashboard_orchestrator._verify_exports()",
        "validation": """
        for filename in REQUIRED_FILES:
            filepath = json_dir / filename
            if not filepath.exists():
                print(f"âŒ {filename} MISSING")
            else:
                with open(filepath, 'r') as f:
                    json.load(f)  # Validate JSON syntax
        """,
        "ensures": "All JSON files exist and are valid",
        "user_prompt": "Continue anyway? (y/n)"
    },
    
    "error_recovery_strategies": {
        "missing_historical_scores": {
            "error": "ValueError: Historical ensemble scores not generated",
            "recovery": [
                "1. Check _generate_historical_ensemble_scores() completed",
                "2. Re-run train_with_features()",
                "3. Verify self.historical_ensemble_scores is not None"
            ]
        },
        "length_mismatch": {
            "error": "ValueError: Score/feature length mismatch",
            "recovery": [
                "1. Check batch processing logic in _generate_historical_ensemble_scores()",
                "2. Verify all batches processed correctly",
                "3. Regenerate scores with verbose=True to see progress"
            ]
        },
        "threshold_missing": {
            "error": "KeyError: 'statistical_thresholds'",
            "recovery": [
                "1. Check detector.train() completed successfully",
                "2. Verify calculate_statistical_thresholds() was called",
                "3. Fall back to legacy thresholds as last resort"
            ]
        }
    }
}

# SECTION 8: HELPER FUNCTIONS FOR VALIDATION
# ============================================================================

def validate_json_exports(json_dir: str = "./json_data") -> Dict[str, bool]:
    """Validate that all required JSON files exist and have valid structure."""
    import json
    from pathlib import Path
    
    required_files = [
        "anomaly_report.json",
        "historical_anomaly_scores.json",
        "dashboard_data.json",
        "market_state.json",
        "regime_statistics.json"
    ]
    
    validation_results = {}
    json_path = Path(json_dir)
    
    for filename in required_files:
        filepath = json_path / filename
        
        if not filepath.exists():
            validation_results[filename] = False
            print(f"❌ MISSING: {filename}")
            continue
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Specific validation for historical_anomaly_scores.json
            if filename == "historical_anomaly_scores.json":
                required_keys = ["dates", "ensemble_scores", "spx_close", "spx_forward_10d"]
                if all(key in data for key in required_keys):
                    lengths = {key: len(data[key]) for key in required_keys}
                    if len(set(lengths.values())) == 1:
                        validation_results[filename] = True
                        print(f"✅ VALID: {filename} (parallel arrays: {list(lengths.values())[0]} elements)")
                    else:
                        validation_results[filename] = False
                        print(f"❌ INVALID: {filename} - Array length mismatch: {lengths}")
                else:
                    validation_results[filename] = False
                    print(f"❌ INVALID: {filename} - Missing required keys")
            
            # Validation for anomaly_report.json
            elif filename == "anomaly_report.json":
                required_keys = ["ensemble", "domain_anomalies", "persistence", "classification"]
                if all(key in data for key in required_keys):
                    # Check for statistical thresholds
                    if "thresholds" in data.get("classification", {}):
                        print(f"✅ VALID: {filename} (includes statistical thresholds)")
                    else:
                        print(f"⚠️  VALID: {filename} (missing statistical thresholds - using fallback)")
                    validation_results[filename] = True
                else:
                    validation_results[filename] = False
                    print(f"❌ INVALID: {filename} - Missing required keys: {required_keys}")
            
            else:
                validation_results[filename] = True
                print(f"✅ VALID: {filename}")
        
        except json.JSONDecodeError:
            validation_results[filename] = False
            print(f"❌ INVALID JSON: {filename}")
        except Exception as e:
            validation_results[filename] = False
            print(f"❌ ERROR reading {filename}: {e}")
    
    return validation_results


def validate_parallel_arrays(json_dir: str = "./json_data") -> bool:
    """Validate that parallel arrays in historical_anomaly_scores.json are aligned."""
    import json
    from pathlib import Path
    
    filepath = Path(json_dir) / "historical_anomaly_scores.json"
    
    if not filepath.exists():
        print("❌ historical_anomaly_scores.json not found")
        return False
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        arrays = ["dates", "ensemble_scores", "spx_close", "spx_forward_10d"]
        lengths = {key: len(data.get(key, [])) for key in arrays}
        
        if len(set(lengths.values())) == 1:
            print(f"✅ Parallel arrays VALID: All arrays have {list(lengths.values())[0]} elements")
            return True
        else:
            print(f"❌ Parallel arrays INVALID: Length mismatch")
            for key, length in lengths.items():
                print(f"   {key}: {length}")
            return False
    
    except Exception as e:
        print(f"❌ Error validating parallel arrays: {e}")
        return False


def print_data_lineage_summary():
    """Print summary of complete data lineage."""
    print("\n" + "="*80)
    print("COMPLETE DATA LINEAGE SUMMARY")
    print("="*80)
    
    print("\n📁 FILESYSTEM STRUCTURE:")
    print("   src/json_data/                    ← All JSON exports")
    print("   src/Chart Modules/subcharts/      ← Read ../../json_data/")
    print("   src/dashboard_unified.html        ← Reads ./json_data/")
    
    print("\n📊 JSON FILES (8 total):")
    for filename, info in FILE_SYSTEM_MAP["json_data/"]["files"].items():
        critical_marker = "🔴 CRITICAL" if info.get("critical") else ""
        print(f"   • {filename:<40} {info['size']:>15} {critical_marker}")
    
    print("\n🎯 CRITICAL PATHS:")
    print("   1. historical_anomaly_scores.json → 5 subcharts + persistence stats")
    print("   2. anomaly_report.json → All 6 subcharts")
    print("   3. Parallel arrays MUST match in length")
    print("   4. Statistical thresholds adapt to data distribution")
    
    print("\n🔄 AUTO-REFRESH (15s interval):")
    print("   • Fetch live VIX + SPX prices")
    print("   • Recalculate anomaly scores")
    print("   • Update 4 JSON files")
    print("   • Subcharts auto-reload via polling")
    
    print("\n🆕 RECENT UPDATES:")
    print("   ✅ Statistical thresholds (data-driven)")
    print("   ✅ Full history persistence calculation")
    print("   ⚠️  Dashboard integration for thresholds (TODO)")
    
    print("\n" + "="*80)


def print_integration_status():
    """Print current integration status of new features."""
    print("\n" + "="*80)
    print("INTEGRATION STATUS")
    print("="*80)
    
    for feature, details in INTEGRATION_CHECKLIST.items():
        status_icon = "✅" if details["status"] == "✅ IMPLEMENTED" else "⚠️ "
        print(f"\n{status_icon} {feature.upper().replace('_', ' ')}: {details['status']}")
        for step in details.get("steps", details.get("tasks", [])):
            print(f"   {step}")
    
    print("\n" + "="*80)


# ============================================================================
# QUICK REFERENCE: DATA PATHS FOR DASHBOARD DEVELOPERS
# ============================================================================

QUICK_REFERENCE = {
    "Current Anomaly Score": {
        "path": "anomaly_report.json:ensemble.score",
        "type": "float [0.0, 1.0]",
        "usage": "Main anomaly indicator"
    },
    
    "Classification Level": {
        "path": "anomaly_report.json:classification.level",
        "type": "string (NORMAL/MODERATE/HIGH/CRITICAL)",
        "usage": "Color-coded severity badge"
    },
    
    "Statistical Thresholds": {
        "path": "anomaly_report.json:classification.thresholds",
        "type": "Dict[moderate, high, critical]",
        "usage": "Draw threshold lines on charts",
        "fallback": "Use legacy thresholds {0.70, 0.78, 0.88}"
    },
    
    "Historical Scores": {
        "path": "historical_anomaly_scores.json:ensemble_scores",
        "type": "array[float]",
        "usage": "Time series analysis, histogram, percentile calculation"
    },
    
    "Persistence Stats": {
        "path": "anomaly_report.json:persistence.historical_stats",
        "fields": {
            "current_streak": "Consecutive anomaly days",
            "mean_duration": "Average episode length",
            "max_duration": "Longest episode",
            "anomaly_rate": "% of days anomalous"
        }
    },
    
    "Domain Scores": {
        "path": "anomaly_report.json:domain_anomalies",
        "type": "Dict[domain_name, {score, percentile, level}]",
        "usage": "Detector ranking chart"
    },
    
    "Forward Returns": {
        "path": "historical_anomaly_scores.json:spx_forward_10d",
        "type": "array[float]",
        "usage": "Forward return analysis"
    }
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_data_lineage_summary()
    print_integration_status()
    
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)
    
    validation_results = validate_json_exports()
    parallel_arrays_valid = validate_parallel_arrays()
    
    all_valid = all(validation_results.values()) and parallel_arrays_valid
    
    if all_valid:
        print("\n✅ ALL VALIDATIONS PASSED")
    else:
        print("\n❌ VALIDATION FAILURES DETECTED")
        print("   Review errors above and regenerate missing/invalid files")
    
    print("\n" + "="*80)