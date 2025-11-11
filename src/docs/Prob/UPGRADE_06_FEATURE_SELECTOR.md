# UPGRADE GUIDE: xgboost_feature_selector_v2.py
## Switch from Classification to Regression Target

---

## SYSTEM CONTEXT

### What We're Changing
**Minimal changes** - The feature selector's core logic (stability scoring, multicollinearity removal, recursive addition) stays the same. Only the target variable and evaluation metric change.

**Old:** Binary classification target (0/1 for VIX expansion)
**New:** Continuous regression target (VIX % change)

**Why This Works:**
- Feature importance still measured by gain/split metrics
- Stability still computed via cross-validation variance
- Multicollinearity still detected via correlation
- Only difference: Optimize for RMSE instead of accuracy

---

## FILE ROLE: xgboost_feature_selector_v2.py

**Purpose:** Select optimal feature subset from 232 candidates using:
1. Stability scoring (cross-validation variance)
2. Multicollinearity removal (correlation >0.95)
3. Recursive feature addition (find performance cliff)
4. Preserve critical features (VX1-VX2, SKEW, yield curves)

**What's Changing:**
- Target creation: Binary → Continuous
- Evaluation metric: Accuracy → RMSE
- Model type: XGBClassifier → XGBRegressor
- Keep: All feature selection logic

---

## DETAILED CHANGES

### CHANGE 1: Update Target Creation

**LOCATION:** In `run_selection()` or `_create_target()` method

**FIND:**
```python
def _create_target(self, df, horizon, threshold):
    """Create binary expansion target."""
    future_vix = df['vix'].shift(-horizon)
    target = (future_vix > df['vix'] * (1 + threshold)).astype(int)
    return target
```

**REPLACE WITH:**
```python
def _create_target(self, df, horizon):
    """
    Create continuous VIX % change target.
    
    Args:
        df: DataFrame with 'vix' column
        horizon: Forward-looking days
        
    Returns:
        pd.Series: VIX % change [-50 to +200]
    """
    future_vix = df['vix'].shift(-horizon)
    target = (future_vix / df['vix'] - 1) * 100
    
    # Clip extremes (consistent with trainer)
    target = target.clip(-50, 200)
    
    return target
```

**Key Changes:**
- Remove `threshold` parameter (no longer needed)
- Return continuous values instead of 0/1
- Clip to reasonable range (-50% to +200%)

---

### CHANGE 2: Update Model Type and Metrics

**LOCATION:** In `_train_baseline()` or wherever XGBoost is trained

**FIND:**
```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

model = XGBClassifier(**params)
accuracy = accuracy_score(y_true, y_pred)
```

**REPLACE WITH:**
```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

model = XGBRegressor(**params)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
```

---

### CHANGE 3: Update Cross-Validation Scoring

**LOCATION:** In feature evaluation loop

**FIND:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    XGBClassifier(**params),
    X, y,
    cv=5,
    scoring='accuracy'  # OLD
)
mean_performance = scores.mean()
```

**REPLACE WITH:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    XGBRegressor(**params),
    X, y,
    cv=5,
    scoring='neg_root_mean_squared_error'  # NEW
)
mean_performance = -scores.mean()  # Negate to get RMSE
```

**Note:** `neg_root_mean_squared_error` returns negative RMSE (scikit-learn convention: higher is better). Negate to get actual RMSE.

---

### CHANGE 4: Update Logging/Reporting

**LOCATION:** Wherever performance is logged

**FIND:**
```python
logger.info(f"   Size {size:3d}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")
```

**REPLACE WITH:**
```python
logger.info(f"   Size {size:3d}: RMSE = {mean_rmse:.2f}% ± {std_rmse:.2f}%")
```

---

### CHANGE 5: Update Optimal Feature Set Selection

**LOCATION:** In recursive feature addition loop

**FIND:**
```python
# Select feature set with highest accuracy
optimal_size = feature_sizes[np.argmax(accuracies)]
```

**REPLACE WITH:**
```python
# Select feature set with lowest RMSE
optimal_size = feature_sizes[np.argmin(rmses)]  # MIN not MAX!
```

**Critical:** For regression, lower RMSE = better. For classification, higher accuracy = better.

---

## COMPLETE METHOD EXAMPLE

Here's how a typical method changes:

### BEFORE (Classification)
```python
def _evaluate_feature_set(self, X, y):
    """Evaluate feature set using accuracy."""
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=100,
        scale_pos_weight=4.5  # For imbalanced classes
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)
```

### AFTER (Regression)
```python
def _evaluate_feature_set(self, X, y):
    """Evaluate feature set using RMSE."""
    model = XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=100,
        objective='reg:squarederror'  # No scale_pos_weight for regression
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)
    
    return np.mean(rmses), np.std(rmses)
```

---

## INTEGRATION POINTS

### Called By: xgboost_integration.py (your main script)
```python
from xgboost_feature_selector_v2 import IntelligentFeatureSelector

selector = IntelligentFeatureSelector()
selected_features = selector.run_selection(
    df=features,
    horizon=5,
    # threshold removed - no longer needed
    min_stability=0.3,
    max_correlation=0.95
)
```

### Provides To: xgboost_trainer_v2.py
```python
# In trainer
selected_features = pd.read_csv('models/selected_features_v2.txt', header=None)[0].tolist()
X = df[selected_features]  # Use selected subset
```

---

## TESTING

### Unit Test
```python
def test_feature_selector_regression():
    """Test feature selector with regression target."""
    from feature_engine import FeatureEngineV5
    from xgboost_feature_selector_v2 import IntelligentFeatureSelector
    
    # Build features
    engine = FeatureEngineV5()
    df = engine.build_features(window='1y')
    
    # Run selection
    selector = IntelligentFeatureSelector()
    selected = selector.run_selection(
        df=df,
        horizon=5,
        min_stability=0.3,
        max_correlation=0.95
    )
    
    # Check output
    assert len(selected) > 0
    assert len(selected) < 232  # Should reduce from 232
    assert 'vix' not in selected  # vix used for target creation, not as feature
    
    print(f"✅ Selected {len(selected)} features")
    print(f"   Top 10: {selected[:10]}")

test_feature_selector_regression()
```

---

## COMMON PITFALLS

### 1. Forgetting to Remove threshold Parameter
```python
# WRONG: Still passing threshold
selected = selector.run_selection(df, horizon=5, threshold=0.05)

# CORRECT: No threshold needed
selected = selector.run_selection(df, horizon=5)
```

### 2. Using Wrong Metric Direction
```python
# WRONG: Maximize RMSE
optimal = feature_sizes[np.argmax(rmses)]  # Higher RMSE = worse!

# CORRECT: Minimize RMSE
optimal = feature_sizes[np.argmin(rmses)]
```

### 3. Not Removing Classification-Specific Params
```python
# WRONG: Keep scale_pos_weight
model = XGBRegressor(scale_pos_weight=4.5)  # Not valid for regressor

# CORRECT: Remove it
model = XGBRegressor(objective='reg:squarederror')
```

### 4. Wrong Scoring Name
```python
# WRONG: Old sklearn name
scoring='neg_mean_squared_error'  # Deprecated

# CORRECT: Use new name
scoring='neg_root_mean_squared_error'
```

---

## VALIDATION CHECKLIST

After making changes:

- [ ] Target is continuous (not binary)
- [ ] Target clipped to [-50, 200] range
- [ ] Using XGBRegressor (not XGBClassifier)
- [ ] Scoring uses RMSE (not accuracy)
- [ ] Cross-validation uses TimeSeriesSplit (same as before)
- [ ] Optimal selection minimizes RMSE (not maximizes accuracy)
- [ ] Removed scale_pos_weight parameter
- [ ] Removed threshold parameter from API
- [ ] Logging shows RMSE values (in %)
- [ ] Output file still saves to `models/selected_features_v2.txt`

---

## PERFORMANCE EXPECTATIONS

### Before (Classification)
```
Size  30: Accuracy = 0.536 ± 0.033
Size  40: Accuracy = 0.553 ± 0.036  ← Optimal
Size  50: Accuracy = 0.519 ± 0.058
```

### After (Regression)
```
Size  30: RMSE = 12.8% ± 2.1%
Size  40: RMSE = 11.5% ± 1.8%  ← Optimal
Size  50: RMSE = 12.2% ± 2.3%
```

**Interpretation:**
- RMSE ~11-12% means average forecast error of 11-12 percentage points
- For VIX at 18, this means ±2-2.5 VIX points error
- Lower RMSE = tighter distribution predictions

---

## SUMMARY

**Changes Required:** Minimal (~20 lines modified, 0 lines added)

**Modified Methods:**
- `_create_target()` - Binary → Continuous
- `_train_baseline()` - Classifier → Regressor
- `_evaluate_feature_set()` - Accuracy → RMSE
- Logging statements - Update metric names

**Unchanged Methods:**
- `_compute_stability()` - Works same for regression
- `_remove_multicollinearity()` - Works same
- `_recursive_feature_addition()` - Works same
- `_preserve_critical_features()` - Works same

**Output:** Still saves to `models/selected_features_v2.txt` (48 features)

**Next Steps:**
1. Update target creation to continuous
2. Switch to XGBRegressor
3. Update all metrics to RMSE
4. Test with 1-year data
5. Run full selection on 15 years
