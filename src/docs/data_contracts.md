# VIX Anomaly Detection System - Data Contracts
## JSON Export Schemas & Field Specifications

**Version**: 2.0.0  
**Last Updated**: 2025-11-01  
**Companion Documents**: system.md (architecture), data_lineage.py (validation)

---

## File Locations

All JSON exports are written to `src/json_data/` directory.

```
src/json_data/
├── anomaly_report.json              # Current anomaly state (CRITICAL)
├── historical_anomaly_scores.json   # Time series data (CRITICAL)
├── refresh_state.pkl                # Model cache (CRITICAL)
├── dashboard_data.json              # Unified contract
├── market_state.json                # Current prices + regime
├── regime_statistics.json           # Regime distribution
├── anomaly_metadata.json            # Detector metadata
├── anomaly_feature_attribution.json # SHAP values
└── vix_history.json                 # Historical VIX levels
```

---

## anomaly_report.json

**Producer**: `VIXPredictorV4.export_anomaly_report()`  
**Update Frequency**: Every refresh cycle  
**Size**: 5-10 KB  
**Consumers**: hero_section.html, persistence_tracker.html, detector_ranking.html, score_distribution.html, forward_returns.html

### Schema

```json
{
  "timestamp": "2025-10-31T10:30:00.123456",
  "ensemble": {
    "score": 0.782,
    "std": 0.134,
    "max_anomaly": 0.915,
    "min_anomaly": 0.621,
    "n_detectors": 15
  },
  "domain_anomalies": {
    "vix_mean_reversion": {
      "score": 0.823,
      "percentile": 91.2,
      "level": "HIGH"
    },
    "vix_momentum": {
      "score": 0.651,
      "percentile": 78.4,
      "level": "MODERATE"
    }
  },
  "persistence": {
    "current_count": 7,
    "max_possible": 10,
    "percentage": 0.70,
    "active_detectors": [
      "vix_mean_reversion",
      "cboe_options_flow",
      "spx_price_action"
    ],
    "historical_stats": {
      "current_streak": 3,
      "mean_duration": 2.3,
      "max_duration": 15,
      "total_anomaly_days": 387,
      "anomaly_rate": 0.096
    }
  },
  "top_anomalies": [
    {"name": "vix_mean_reversion", "score": 0.823},
    {"name": "vix_momentum", "score": 0.651}
  ],
  "classification": {
    "level": "HIGH",
    "thresholds": {
      "moderate": 0.682,
      "high": 0.759,
      "critical": 0.873
    }
  }
}
```

### Field Definitions

| Field | Type | Range/Values | Description |
|-------|------|--------------|-------------|
| `timestamp` | string (ISO 8601) | - | Export generation time |
| `ensemble.score` | float | [0.0, 1.0] | Mean of 15 detector scores |
| `ensemble.std` | float | ≥0.0 | Standard deviation across detectors |
| `ensemble.n_detectors` | int | 15 | Number of active detectors |
| `domain_anomalies[domain].score` | float | [0.0, 1.0] | Domain-specific anomaly score |
| `domain_anomalies[domain].percentile` | float | [0.0, 100.0] | Position in training distribution |
| `domain_anomalies[domain].level` | string | NORMAL, MODERATE, HIGH, CRITICAL | Statistical threshold classification |
| `persistence.current_count` | int | [0, 10] | Domains exceeding high threshold |
| `persistence.active_detectors` | array[string] | - | List of active domain names |
| `persistence.historical_stats.current_streak` | int | ≥0 | Consecutive anomaly days |
| `persistence.historical_stats.mean_duration` | float | ≥0.0 | Average episode length |
| `persistence.historical_stats.max_duration` | int | ≥0 | Longest recorded episode |
| `persistence.historical_stats.anomaly_rate` | float | [0.0, 1.0] | Proportion of history flagged |
| `classification.level` | string | NORMAL, MODERATE, HIGH, CRITICAL | Global anomaly classification |
| `classification.thresholds` | object | - | Data-driven percentile thresholds |

**Critical Properties**:
- `classification.thresholds` are computed during training (85th, 92nd, 98th percentile)
- `persistence.historical_stats` calculated on full history (not windowed)
- All scores normalized to [0.0, 1.0] via percentile inversion

---

## historical_anomaly_scores.json

**Producer**: `dashboard_data_contract.export_historical_anomaly_scores()`  
**Update Frequency**: Training only (static in cached mode)  
**Size**: 50-100 KB  
**Consumers**: ALL dashboard subcharts (primary data source)

### Schema

```json
{
  "dates": [
    "2009-01-05",
    "2009-01-06",
    "2025-10-31"
  ],
  "ensemble_scores": [
    0.234,
    0.198,
    0.782
  ],
  "spx_close": [
    903.25,
    934.70,
    5841.47
  ],
  "spx_forward_10d": [
    -5.23,
    3.47,
    null
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `dates` | array[string] | Trading dates (YYYY-MM-DD format) |
| `ensemble_scores` | array[float] | Historical ensemble anomaly scores |
| `spx_close` | array[float] | SPX closing prices |
| `spx_forward_10d` | array[float] | 10-day forward returns (%) |

**Critical Constraint**: All arrays **must** be parallel (same length, aligned indices). Violation breaks 5 of 6 dashboard charts.

**Forward Returns Calculation**:
```python
spx_forward_10d = spx.pct_change(10).shift(-10) * 100
```
Last 10 values are `null` (no future data available).

**Generation Process**:
1. Load features (complete historical set, 3000-5000+ observations)
2. For each row: `result = detector.detect(features.iloc[[i]])`
3. Collect `ensemble_scores.append(result['ensemble']['score'])`
4. Calculate forward returns
5. Export parallel arrays

---

## dashboard_data.json

**Producer**: `dashboard_data_contract.export_unified_dashboard_data()`  
**Update Frequency**: Every refresh cycle  
**Size**: 15-30 KB  
**Consumers**: dashboard_unified.html (main dashboard)

### Schema

```json
{
  "vix_state": {
    "current": 18.23,
    "regime": "Normal",
    "regime_persistence": 14,
    "percentile_30d": 67.3,
    "percentile_historical": 52.1
  },
  "anomaly_analysis": {
    "ensemble_score": 0.782,
    "classification": "HIGH",
    "active_detectors": [
      "vix_mean_reversion",
      "cboe_options_flow"
    ],
    "top_anomalies": [
      {"name": "vix_mean_reversion", "score": 0.823}
    ],
    "persistence_count": 7
  },
  "regime_statistics": {
    "current_regime": {
      "duration_days": 14,
      "typical_duration": 28.4,
      "vix_range": [16.77, 24.40]
    },
    "historical_distribution": {
      "low_vol_pct": 18.3,
      "normal_pct": 52.7,
      "elevated_pct": 23.1,
      "crisis_pct": 5.9
    }
  }
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `vix_state.current` | float | Latest VIX level |
| `vix_state.regime` | string | Current regime (Low Vol, Normal, Elevated, Crisis) |
| `vix_state.regime_persistence` | int | Consecutive days in current regime |
| `vix_state.percentile_30d` | float | Position in rolling 30-day window |
| `vix_state.percentile_historical` | float | Position in full historical distribution |
| `anomaly_analysis.ensemble_score` | float | Current ensemble anomaly score [0.0, 1.0] |
| `anomaly_analysis.classification` | string | NORMAL, MODERATE, HIGH, CRITICAL |
| `anomaly_analysis.persistence_count` | int | Number of active domains [0, 10] |
| `regime_statistics.current_regime.duration_days` | int | Days in current regime |
| `regime_statistics.current_regime.typical_duration` | float | Historical average for this regime |

---

## market_state.json

**Producer**: `dashboard_data_contract.export_market_state()`  
**Update Frequency**: Every refresh cycle  
**Size**: 2-5 KB  

### Schema

```json
{
  "timestamp": "2025-10-31T10:30:00.123456",
  "vix": {
    "current": 18.23,
    "change": 1.45,
    "change_pct": 8.64,
    "regime": "Normal"
  },
  "spx": {
    "current": 5841.47,
    "change": 32.15,
    "change_pct": 0.55
  },
  "regime_info": {
    "name": "Normal",
    "boundaries": [16.77, 24.40],
    "persistence_days": 14,
    "typical_duration": 28.4
  }
}
```

---

## regime_statistics.json

**Producer**: `dashboard_data_contract.export_regime_statistics()`  
**Update Frequency**: Training only (static in cached mode)  
**Size**: 5-10 KB  

### Schema

```json
{
  "regime_distribution": {
    "Low Vol": {"count": 732, "percentage": 18.3},
    "Normal": {"count": 2118, "percentage": 52.7},
    "Elevated": {"count": 929, "percentage": 23.1},
    "Crisis": {"count": 243, "percentage": 5.9}
  },
  "regime_transitions": {
    "Low Vol": {"mean_duration": 15.2, "max_duration": 87},
    "Normal": {"mean_duration": 28.4, "max_duration": 142},
    "Elevated": {"mean_duration": 12.7, "max_duration": 58},
    "Crisis": {"mean_duration": 5.1, "max_duration": 23}
  },
  "current_regime": {
    "name": "Normal",
    "duration": 14,
    "start_date": "2025-10-15"
  }
}
```

---

## anomaly_metadata.json

**Producer**: `dashboard_data_contract.export_anomaly_metadata()`  
**Update Frequency**: Training only  
**Size**: 10-20 KB  

### Schema

```json
{
  "detector_coverage": {
    "vix_mean_reversion": {
      "expected_features": 17,
      "available_features": 17,
      "coverage_pct": 100.0,
      "status": "OK"
    },
    "cboe_options_flow": {
      "expected_features": 24,
      "available_features": 21,
      "coverage_pct": 87.5,
      "status": "OK"
    }
  },
  "training_info": {
    "training_date": "2025-10-31",
    "n_observations": 4022,
    "n_features": 234,
    "contamination": 0.05
  },
  "statistical_thresholds": {
    "moderate": 0.682,
    "high": 0.759,
    "critical": 0.873,
    "percentiles": [85, 92, 98]
  }
}
```

---

## anomaly_feature_attribution.json

**Producer**: `dashboard_data_contract.export_feature_attribution()`  
**Update Frequency**: Training only  
**Size**: 20-50 KB  

### Schema

```json
{
  "shap_values": {
    "vix_mean_reversion": {
      "vix_vs_ma21": 0.067,
      "vix_zscore_63d": 0.061
    }
  },
  "permutation_importance": {
    "vix_vs_ma21": 0.082,
    "spx_realized_vol_21d": 0.058
  },
  "top_features_by_domain": {
    "vix_mean_reversion": [
      {"feature": "vix_vs_ma21", "shap": 0.067},
      {"feature": "vix_zscore_63d", "shap": 0.061}
    ]
  }
}
```

---

## vix_history.json

**Producer**: `dashboard_data_contract.export_vix_history()`  
**Update Frequency**: Training only  
**Size**: 20-40 KB  

### Schema

```json
{
  "dates": ["2009-01-05", "2025-10-31"],
  "vix": [40.32, 18.23],
  "regimes": [3, 1]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `dates` | array[string] | Trading dates (YYYY-MM-DD) |
| `vix` | array[float] | Historical VIX levels |
| `regimes` | array[int] | Regime codes (0=Low Vol, 1=Normal, 2=Elevated, 3=Crisis) |

---

## Regime Boundaries

All exports use consistent regime definitions:

| Regime Code | Name | VIX Range | Typical Duration |
|-------------|------|-----------|------------------|
| 0 | Low Vol | [0, 16.77) | 15.2 days |
| 1 | Normal | [16.77, 24.40) | 28.4 days |
| 2 | Elevated | [24.40, 39.67) | 12.7 days |
| 3 | Crisis | [39.67, 100] | 5.1 days |

**Derivation**: Historical quartiles from 1990-2025 VIX distribution.

---

## Classification Thresholds

**Data-Driven Approach**: Thresholds computed from training distribution percentiles.

| Level | Percentile | Typical Range | Interpretation |
|-------|-----------|---------------|----------------|
| NORMAL | <85th | <0.65-0.75 | Baseline market conditions |
| MODERATE | 85-92nd | 0.65-0.75 | Elevated but manageable |
| HIGH | 92-98th | 0.75-0.82 | Significant stress |
| CRITICAL | ≥98th | ≥0.85-0.92 | Systemic risk event |

**Adaptive Property**: Thresholds automatically recalibrate during retraining to reflect distribution shifts.

---

## Validation Rules

### Parallel Array Alignment

**Critical Constraint**: Arrays in `historical_anomaly_scores.json` must have identical length.

```python
assert len(dates) == len(ensemble_scores) == len(spx_close) == len(spx_forward_10d)
```

**Impact if Violated**: Breaks 5 of 6 dashboard charts.

### Score Range Validation

All anomaly scores must satisfy: `0.0 ≤ score ≤ 1.0`

**Applies to**:
- `ensemble.score`
- `domain_anomalies[*].score`
- `ensemble_scores` array

### Threshold Monotonicity

Thresholds must be strictly increasing:

```python
assert moderate < high < critical
```

### Classification Consistency

Domain and ensemble classification levels must use identical threshold logic.

---

## Known Issues

### 1. Dashboard Hardcoded Thresholds

**Problem**: Frontend HTML files use hardcoded values (0.725, 0.805, 0.914) instead of reading from `anomaly_report.json['classification']['thresholds']`.

**Affected Files**:
- `persistence_tracker.html`
- `hero_section.html`
- `score_distribution.html`

**Impact**: Frontend classification diverges from backend when statistical thresholds update during retraining.

**Fix**: Update JavaScript to fetch thresholds dynamically (2-3 hour effort).

### 2. Fixed Contamination Rate

**Problem**: All detectors use 5% contamination regardless of regime, but Crisis regime exhibits 15-20% actual anomaly rate.

**Impact**: Reduced sensitivity during high-volatility periods when signals are most valuable.

**Solution**: Regime-adaptive contamination mapping:
```python
contamination_map = {0: 0.02, 1: 0.05, 2: 0.10, 3: 0.20}
```

---

## Usage Examples

### Python Validation

```python
from data_lineage import validate_json_exports, print_validation_report

results = validate_json_exports()
print_validation_report(results)
```

### JavaScript Consumption

```javascript
// Dashboard subchart pattern
fetch('../../json_data/anomaly_report.json', {
    cache: 'no-store',
    headers: {'Cache-Control': 'no-cache'}
})
.then(response => response.json())
.then(data => {
    const score = data.ensemble.score;
    const thresholds = data.classification.thresholds;
    renderChart(score, thresholds);
});
```

---

## Contact & Support

**System Owner**: VIX Trading Team  
**Documentation**: This file + system.md (architecture) + data_lineage.py (validation)  
**Last Review**: 2025-11-01  
**Next Review Due**: 2025-12-01
