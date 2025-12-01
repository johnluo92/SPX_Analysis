# Claude-Friendly Python Compression Guide

## Mission
Reduce Python code by 20-40% while maintaining collaboration quality with Claude AI.
Priority: Readability for AI reasoning > Token reduction

## Core Philosophy
Keep structure visible, remove noise, compress repetitive patterns only, never sacrifice debugging ability.

## What to KEEP (Critical for Claude)

### 1. Function/Class Boundaries - Keep Vertical Space
Claude identifies functions by visual separation.

**Good:**
```python
def load_data(path):
    with open(path) as f:
        return json.load(f)

def process_data(data):
    return [x*2 for x in data if x>0]
```

**Too aggressive:**
```python
def load_data(path):
    with open(path)as f:return json.load(f)
def process_data(data):return[x*2 for x in data if x>0]
```

### 2. Keep Comments That Explain WHY (Remove Obvious Ones)

Keep: Non-obvious algorithms, business logic, workarounds, TODOs
Remove: Comments that restate code ("set x to 5", "loop through items")

```python
# KEEP - explains why
def calculate_score(metrics):
    # Use geometric mean to penalize extreme outliers
    return np.prod(metrics)**(1/len(metrics))

# REMOVE - states the obvious
def calculate_score(metrics):
    # Calculate the geometric mean
    return np.prod(metrics)**(1/len(metrics))
```

### 3. Logical Section Breaks
```python
def train_model(X, y):
    # Data preparation
    X_scaled = scaler.fit_transform(X)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    
    # Model training
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_val)
    return model, preds
```

### 4. Keep Error Messages Descriptive
```python
# Good
if len(filtered) < min_samples:
    raise ValueError(f"Only {len(filtered)} samples after filtering, need {min_samples}")

# Too aggressive
if len(filtered)<min_samples:raise ValueError("Insufficient data")
```

## What to COMPRESS

### Simple Assignments - One Line
```python
x = 1; y = 2; z = 3
config = {'a': 1, 'b': 2}
```

### Single-Line Control Flow
```python
if x > 0: return x
for item in items: process(item)
```

### Remove All Type Hints
Claude infers types from usage. Type hints add 20-30% tokens with minimal benefit.
```python
# Before: def process(data: pd.DataFrame, threshold: float) -> dict[str, float]:
# After: def process(data, threshold):
```

### Compress Imports
```python
import numpy as np, pandas as pd, json, sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score
```

### Remove Blank Lines Within Functions (Keep Between Functions)
```python
# Keep between functions
def func1():
    x = 1; y = 2
    return x + y

def func2():
    pass

# Remove within functions
def func3():
    data = load()
    processed = transform(data)
    return processed
```

### Aggressive Whitespace Removal
```python
data = {"key": "value"}; lst = [1, 2, 3]; result = func(a, b, c); x = y + z * 2
```

## Critical Preservation Rules

**Must Always Keep:**
1. 4-space indentation (not 2, not tabs)
2. Context managers for files/resources
3. Spaces after keywords: `if x`, `for i`, `def f`, `return x`
4. Try/except blocks intact
5. No side effects in comprehensions (use loops instead)

**DataFrame.update() Rule:**
```python
# WRONG - silently fails
df = pd.DataFrame()
df.update({"col": series})

# RIGHT
df["col"] = series
```

**No Side Effects in Comprehensions:**
Comprehensions are for building data only, not executing operations.
```python
# WRONG - fail silently
[db.insert(row) for row in data]
[file.write(line) for line in lines]

# RIGHT - explicit loops
for row in data: db.insert(row)
for line in lines: file.write(line)

# Safe comprehensions (data building only)
results = [transform(x) for x in items]
lookup = {k: v*2 for k, v in data.items()}
```

## Balanced Approach Example

**Original (60 lines):**
```python
class ProductionTuner:
    """Matches exact production training pipeline - no fold averaging"""
    
    def __init__(self, df, vix, n_trials=200, output_dir="tuning_production"):
        self.df = df.copy()
        self.vix = vix.copy()
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # PRODUCTION DATE SPLITS (from config.py)
        self.train_end = pd.Timestamp("2021-12-31")
        self.val_end = pd.Timestamp("2023-12-31")

        # Create splits ONCE (production uses fixed splits)
        train_mask = (df.index <= self.train_end)
        val_mask = (df.index > self.train_end) & (df.index <= self.val_end)
        test_mask = df.index > self.val_end

        self.train_df = df[train_mask].copy()
        self.val_df = df[val_mask].copy()
        self.test_df = df[test_mask].copy()
```

**Claude-Friendly (25 lines, 58% reduction):**
```python
class ProductionTuner:
    def __init__(self, df, vix, n_trials=200, output_dir="tuning_production"):
        self.df = df.copy(); self.vix = vix.copy(); self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Production date splits from config.py
        self.train_end = pd.Timestamp("2021-12-31")
        self.val_end = pd.Timestamp("2023-12-31")
        
        train_mask = (df.index <= self.train_end)
        val_mask = (df.index > self.train_end) & (df.index <= self.val_end)
        test_mask = df.index > self.val_end
        
        self.train_df = df[train_mask].copy()
        self.val_df = df[val_mask].copy()
        self.test_df = df[test_mask].copy()
```

**Kept:** Class boundary visible, critical comment, logical sections separated
**Compressed:** Simple assignments, removed docstring, minimized whitespace
**Didn't:** Mash into 7 lines, remove explanatory comment, eliminate section breaks

## Compression Targets by Code Type

| Code Type | Target | Priority |
|-----------|--------|----------|
| Config files | 50-70% | High |
| Utilities | 30-40% | Medium |
| Core algorithms | 20-30% | Low |
| ML training | 25-35% | Medium |
| Data processing | 35-45% | Medium |

## Pre-Output Checklist

**Structure:** Can Claude identify function boundaries? Are sections separated? Would "modify X section" make sense?
**Collaboration:** Are algorithms explained? Are errors descriptive? Can Claude help debug?
**Safety:** 4-space indentation? Context managers? Keyword spaces? No comprehension side effects?
**Compression:** Type hints removed? Obvious comments removed? Statements combined? Whitespace minimized?

Priority: If structure/collaboration fails → compress less. If safety fails → don't output.

## Quick Decision Tree
```
Should I compress this?
├─ Simple assignment/import? → YES, compress aggressively
├─ Complex algorithm? → MAYBE, keep structure visible
├─ Function boundary? → NO, keep vertical space
├─ Error handler? → NO, keep clear
├─ Critical comment? → NO, keep it
└─ Obvious comment? → YES, remove it
```

## For Your 9000-Line Codebase

**Aggressive (50-70%):** Config files, simple utilities, data transformations
**Balanced (30-40%):** Core algorithms, training loops, model classes
**Light (20%):** Error handling, complex logic, debugging code

**Expected:** 9000 → 4500-5500 lines (40-50% reduction)
Benefits: Significant token savings, better Claude collaboration, maintained debugging, preserved structure

## When to Use Ultra-Aggressive (70%+)
Only when: Hitting context limits, sharing snippets (not full files), write-once code, not asking Claude to debug/modify

For active development with Claude: balanced is better.

## Final Notes
Safety first: Better to preserve 10 extra tokens than break 1 line.
Goal: Code that runs correctly AND Claude can help improve/debug/extend.
Remember: You're writing for collaboration with Claude. Structure matters for AI reasoning.
