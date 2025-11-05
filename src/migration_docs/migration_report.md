# JSON Structure Analysis & Migration Guide

**Generated:** 2025-11-03T18:34:01.990429

## Overview

This system consolidates 6+ legacy JSON files into 2 unified files:

1. **live_state.json** (15 KB) - Updates every 15 minutes
2. **historical.json** (300 KB) - Static training data

## File Comparison

| Legacy File | New Location | Access Method |
|------------|-------------|---------------|
| market_state.json | live_state.json | `getMarketState()` |
| anomaly_report.json | live_state.json | `getAnomalyState()` |
| historical_anomaly_scores.json | historical.json | `getHistoricalData()` |
| anomaly_feature_attribution.json | historical.json | `getFeatureAttribution()` |
| regime_statistics.json | historical.json | `getRegimeStats()` |
| anomaly_metadata.json | historical.json | `getDetectorMetadata()` |

## Data Service API

### Initialization
```javascript
// Call ONCE on page load
await window.dataService.init();
```

### Live Data Methods

```javascript
// Get current anomaly state
const anomaly = window.dataService.getAnomalyState();
// Returns: { score, classification, activeDetectors, detectorScores, persistence, diagnostics }

// Get current market state
const market = window.dataService.getMarketState();
// Returns: { timestamp, vix_close, spx_close, regime, ... }
```

### Historical Data Methods

```javascript
// Get time series data
const hist = window.dataService.getHistoricalData();
// Returns: { dates, scores, spx, forwardReturns, regimeStats, thresholds }

// Get thresholds
const thresholds = window.dataService.getThresholds();
// Returns: { base, withCI, hasConfidenceIntervals }

// Get feature attribution for a detector
const features = window.dataService.getFeatureAttribution('vix_mean_reversion');
// Returns: array of feature importance data
```

## Migration Checklist

- [ ] Add `<script src="../../data_service.js"></script>` to HTML
- [ ] Replace all `fetch('../../json_data/xxx.json')` calls
- [ ] Use `await window.dataService.init()` once on page load
- [ ] Use appropriate getter methods instead of direct fetch
- [ ] Add error handling for failed data loads
- [ ] Test chart with refreshed data
- [ ] Remove old JSON file references

## Benefits

✅ **Performance**: 2 fetches instead of 6+
✅ **Caching**: Data loaded once and shared
✅ **Consistency**: Single source of truth
✅ **Auto-refresh**: Built-in refresh mechanism
✅ **Error handling**: Centralized error management
✅ **Events**: Listen for data updates across all charts
