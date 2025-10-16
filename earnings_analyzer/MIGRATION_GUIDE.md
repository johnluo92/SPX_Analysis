# Migration Guide: V2 Refactoring

## Overview

The codebase has been refactored to improve maintainability while preserving all existing functionality. Both old and new versions coexist, so you can migrate gradually.

## What Changed

### Architecture

**Before (V1):**
```
earnings_analyzer/
├── analysis/
│   ├── single.py (200+ lines, returns dicts)
│   └── batch.py (400+ lines, does EVERYTHING)
├── output/
│   └── formatters.py (formatting helpers)
└── calculations/
    └── strategy.py (strategy logic)
```

**After (V2):**
```
earnings_analyzer/
├── core/                    # NEW: Data models
│   └── models.py
├── analysis/
│   ├── single.py           # OLD: unchanged
│   ├── single_v2.py        # NEW: uses typed models
│   ├── batch.py            # OLD: unchanged
│   ├── batch_v2.py         # NEW: thin orchestrator (~50 lines)
│   └── enrichment.py       # NEW: IV enrichment separated
├── presentation/           # NEW: All display logic
│   ├── formatters.py
│   ├── tables.py
│   └── insights.py
└── visualization/
    ├── quality_matrix.py   # OLD: unchanged
    └── quality_matrix_v2.py # NEW: uses cached strategies
```

### Key Improvements

1. **Typed Data Models** - `AnalysisResult` instead of dictionaries
2. **Separation of Concerns** - Presentation extracted from business logic
3. **Cached Calculations** - Strategies computed once, reused everywhere
4. **Smaller Files** - batch.py: 400 lines → batch_v2.py: 50 lines
5. **Easier Testing** - Each module has single responsibility

## Migration Path

### Option 1: Drop-in Replacement (Easiest)

Your existing code works without changes:

```python
# This still works exactly as before
from earnings_analyzer import batch_analyze
from earnings_analyzer.visualization import plot_quality_matrix

results = batch_analyze(tickers, fetch_iv=True, parallel=True)
plot_quality_matrix(results)
```

### Option 2: Use V2 with Compatibility Mode

Get benefits of new architecture with zero code changes:

```python
# Just change the import
from earnings_analyzer import batch_analyze_v2_compat as batch_analyze
from earnings_analyzer.visualization import plot_quality_matrix_v2 as plot_quality_matrix

results = batch_analyze(tickers, fetch_iv=True, parallel=True)  # Returns DataFrame
plot_quality_matrix(results)  # Works with DataFrame
```

### Option 3: Full V2 (Recommended)

Use typed objects throughout:

```python
from earnings_analyzer import batch_analyze_v2, AnalysisResult
from earnings_analyzer.visualization import plot_quality_matrix_v2

# Returns DataFrame for compatibility
df = batch_analyze_v2(tickers, fetch_iv=True, parallel=True)

# But you can also work with typed objects
from earnings_analyzer.analysis.single_v2 import analyze_ticker_v2

result, status = analyze_ticker_v2("AAPL")
if result:
    print(f"Ticker: {result.ticker}")
    print(f"HVol: {result.hvol}%")
    pattern_45, edges_45 = result.strategy_45  # Cached property
    pattern_90, edges_90 = result.strategy_90  # Cached property
    print(f"45d: {pattern_45} ({edges_45} edges)")
    print(f"90d: {pattern_90} ({edges_90} edges)")
```

## Benefits of Migration

### Before (V1)
```python
# Strategy calculated in batch.py
# Then RECALCULATED in quality_matrix.py
# No type safety, errors caught at runtime
```

### After (V2)
```python
# Strategy calculated ONCE in AnalysisResult
# Cached property reused everywhere
# Type checking catches errors before running
# IDE autocomplete works perfectly
```

## Testing

Run the comparison test to verify outputs match:

```bash
python test_migration.py
```

This runs both V1 and V2 on the same data and compares all outputs.

## Rollback

If anything breaks:
1. All old files unchanged (single.py, batch.py, quality_matrix.py)
2. Just don't import the v2 versions
3. Delete new files: core/, presentation/, *_v2.py

## Next Steps

1. ✅ Test with `test_migration.py`
2. ✅ Run your normal workflow with v2_compat versions
3. ✅ Gradually adopt typed AnalysisResult in your code
4. ✅ Eventually deprecate V1 files

## File Checklist

### New Files to Create
```
earnings_analyzer/
├── core/
│   ├── __init__.py
│   └── models.py
├── analysis/
│   ├── single_v2.py
│   ├── batch_v2.py
│   └── enrichment.py
├── presentation/
│   ├── __init__.py
│   ├── formatters.py
│   ├── tables.py
│   └── insights.py
└── visualization/
    └── quality_matrix_v2.py
```

### Updated Files
```
earnings_analyzer/
├── __init__.py (add v2 exports)
├── analysis/__init__.py (add v2 exports)
└── visualization/__init__.py (add v2 exports)
```

### Unchanged Files (Keep as-is)
```
earnings_analyzer/
├── analysis/
│   ├── single.py
│   └── batch.py
├── calculations/
│   ├── volatility.py
│   ├── statistics.py
│   └── strategy.py
├── data_sources/
│   ├── alpha_vantage.py
│   └── yahoo_finance.py
├── output/
│   ├── formatters.py
│   └── reports.py
└── visualization/
    └── quality_matrix.py
```