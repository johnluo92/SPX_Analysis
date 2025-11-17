# Comprehensive Feature Cleanup & Enhancement Strategy

## Executive Summary

This document provides a complete implementation plan for:
1. **Removing 58 redundant features** from both `feature_engine.py` and `config.py`
2. **Fixing data generation bugs** (is_opex_week, GOLDSILVER_zscore_63d)
3. **Handling sparse data properly** (GAMMA with 3yr history)
4. **Expanding futures term structure features** (VX, CL, DX spreads/ratios)
5. **Adding comprehensive Treasury yield curve features** (NEW - currently missing)

---

## PART 1: Feature Removal Implementation

### 1.1 VIX/VIXCLS Duplicate Removal

**Problem:** VIX and VIXCLS are the same data source (both fetch ^VIX from Yahoo Finance)

**Location:** `config.py` - these are NOT generated in feature_engine.py

**Action Required:**

```python
# In config.py, REMOVE these lines from META_FEATURES['percentile_rankings']:

# REMOVE:
'VIXCLS_zscore_63d'           # Duplicate of vix_zscore_63d
'VIXCLS_change_10d'            # Duplicate of vix_velocity_10d  
'VIXCLS_change_21d'            # Duplicate of vix_velocity_21d
```

**Verification:** Search config.py for "VIXCLS" - should find ZERO matches after cleanup

---

### 1.2 Perfect Duplicates (corr = 1.000) - Feature Engine

**Location:** `feature_engine.py` - generated in various methods

#### Group A: Vol-of-Vol Duplicates
```python
# In MetaFeatureEngine.extract_cross_asset_relationships()
# REMOVE these two lines (lines ~300-301):

meta['vol_of_vol_10d'] = vix_returns.rolling(10).std() * np.sqrt(252) * 100  # ❌ DELETE
meta['vol_of_vol_21d'] = vix_returns.rolling(21).std() * np.sqrt(252) * 100  # ❌ DELETE

# KEEP: These are already generated in _vix_dynamics():
# ✅ vix_vol_10d
# ✅ vix_vol_21d
```

#### Group B: Risk Premium Duplicate
```python
# In MetaFeatureEngine.extract_cross_asset_relationships()
# REMOVE this line (~line 313):

meta['risk_premium'] = risk_prem  # ❌ DELETE

# KEEP: This is already generated and more descriptive:
# ✅ vix_vs_rv_21d (from _spx_volatility_regime)
```

#### Group C: CBOE Interaction Duplicates
```python
# In UnifiedFeatureEngine._cboe_interactions()
# REMOVE these lines:

features['vxth_vs_vix'] = vxth - vix                              # ❌ DELETE (keep VXTH raw)
features['vxth_premium'] = (vxth - vix) / vix.replace(0, np.nan) * 100  # ❌ DELETE
features['cor_term_slope'] = cor1m - cor3m                        # ❌ DELETE
features['cor_avg'] = cor_avg                                     # ❌ DELETE
features['pc_equity_inst_spread'] = pcce - pcci                   # ❌ DELETE

# KEEP:
# ✅ vxth_vix_ratio (more useful than premium)
# ✅ cor_term_structure (already generated, same as cor_term_slope)
# ✅ pc_divergence (already generated, same calculation)
```

#### Group D: Futures Spread Duplicates
```python
# In FuturesFeatureEngine.extract_vix_futures_features()
# REMOVE these lines:

features['vx_spread'] = spread           # ❌ DELETE (keep VX1-VX2 raw)
features['vx_ratio'] = ratio             # ❌ DELETE (keep VX2-VX1_RATIO raw)
features['vx_spread_velocity_21d'] = spread.diff(21)  # ❌ DELETE

# KEEP:
# ✅ VX1-VX2 (raw from CBOE)
# ✅ VX1-VX2_change_21d (already generated in _cboe_features)
# ✅ VX2-VX1_RATIO (raw from CBOE)
```

```python
# In FuturesFeatureEngine.extract_commodity_futures_features()
# REMOVE:

features['cl_spread'] = cl_spread        # ❌ DELETE (keep CL1-CL2 raw)

# In FuturesFeatureEngine.extract_dollar_futures_features()
# REMOVE:

features['dx_spread'] = dx_spread        # ❌ DELETE (keep DX1-DX2 raw)
```

#### Group E: Macro Momentum Duplicates
```python
# In UnifiedFeatureEngine._build_macro_features()
# MODIFY the loop to use shorter names:

for col in macro.columns:
    features[f'{col}_lag1'] = macro[col].shift(1)
    
    for w in [10, 21, 63]:
        # CHANGE from _mom_ to _ret_:
        features[f'{col.lower()}_ret_{w}d'] = macro[col].pct_change(w) * 100  # ✅ KEEP
        # Don't create features[f'{col}_mom_{w}d']  # ❌ DON'T CREATE
    
    features[f'{col.lower()}_zscore_63d'] = calculate_robust_zscore(macro[col].shift(1), 63)

# Result: crude_oil_ret_10d instead of Crude_Oil_mom_10d
# Result: dollar_index_ret_10d instead of Dollar_Index_mom_10d
```

#### Group F: Moving Average Smoothing
```python
# In FuturesFeatureEngine.extract_vix_futures_features()
# REMOVE:
features['vx_spread_ma10'] = spread.rolling(10).mean()   # ❌ DELETE
features['vx_spread_ma21'] = spread.rolling(21).mean()   # ❌ DELETE

# In FuturesFeatureEngine.extract_commodity_futures_features()
# REMOVE:
features['cl_spread_ma10'] = cl_spread.rolling(10).mean()  # ❌ DELETE

# In FuturesFeatureEngine.extract_dollar_futures_features()
# REMOVE:
features['dx_spread_ma10'] = dx_spread.rolling(10).mean()  # ❌ DELETE
```

#### Group H: Dollar-Crude Correlation Redundancy
```python
# In FuturesFeatureEngine.extract_futures_cross_relationships()
# REMOVE:
features['dollar_crude_corr_breakdown'] = (
    features['dollar_crude_corr_21d'] + 0.5
).abs()  # ❌ DELETE (this is just a transform of the correlation)

# KEEP:
# ✅ dollar_crude_corr_21d
```

---

### 1.3 Velocity/Change Name Consolidation - Feature Engine

**Problem:** Same calculation with different names (velocity vs change vs percentage)

**Strategy:** Keep ONLY percentage versions for normalization across regimes

```python
# In MetaFeatureEngine.extract_rate_of_change_features()
# CURRENT CODE generates both absolute and percentage:

for window in [3, 5, 10, 21]:
    meta[f'{name}_velocity_{window}d'] = series.diff(window)          # ❌ DELETE
    meta[f'{name}_velocity_{window}d_pct'] = series.pct_change(window) * 100  # ✅ KEEP

# CHANGE TO: Only generate percentage
for window in [3, 5, 10, 21]:
    meta[f'{name}_velocity_{window}d_pct'] = series.pct_change(window) * 100  # ✅ KEEP
    # Don't create absolute version
```

**Affected features to REMOVE:**
- `SKEW_velocity_3d`, `SKEW_velocity_5d`, `SKEW_velocity_10d`, `SKEW_velocity_21d`
- `PCC_velocity_3d`, `PCC_velocity_5d`, `PCC_velocity_10d`
- `PCCE_velocity_3d`, `PCCE_velocity_5d`, `PCCE_velocity_10d`
- `PCCI_velocity_3d`, `PCCI_velocity_5d`, `PCCI_velocity_10d`

**Keep percentage versions:**
- `SKEW_velocity_3d_pct`, `SKEW_velocity_5d_pct`, etc.
- `PCC_velocity_3d_pct`, `PCC_velocity_5d_pct`, etc.

---

### 1.4 CBOE _change_21d vs _velocity_21d Cleanup

**Problem:** In `_cboe_features()`, we generate both `_change_21d` AND later generate `_velocity_21d_pct` in rate_of_change

**Location:** `feature_engine.py` in `UnifiedFeatureEngine._cboe_features()`

```python
# CURRENT CODE (lines ~830-835):
for col in cboe_data.columns:
    features[col] = cboe_data[col]
    features[f'{col}_change_21d'] = cboe_data[col].diff(21)  # ❌ REMOVE THIS LINE
    features[f'{col}_zscore_63d'] = calculate_robust_zscore(cboe_data[col].shift(1), 63)

# CHANGE TO:
for col in cboe_data.columns:
    features[col] = cboe_data[col]
    # Don't generate _change_21d here - it will be generated as _velocity_21d_pct in meta features
    features[f'{col}_zscore_63d'] = calculate_robust_zscore(cboe_data[col].shift(1), 63)
```

**Result:** This removes duplicate features like:
- `SKEW_change_21d` (use `SKEW_velocity_21d_pct` instead)
- `PCCI_change_21d` (use `PCCI_velocity_21d_pct` instead)
- `PCCE_change_21d` (use `PCCE_velocity_21d_pct` instead)
- `PCC_change_21d` (use `PCC_velocity_21d_pct` instead)

---

### 1.5 Acceleration Consolidation

**Problem:** `vix_acceleration`, `vix_acceleration_5d`, and `vix_accel_5d` are all the same

**Location:** Multiple places in feature_engine.py

```python
# In _vix_dynamics() - KEEP THIS ONE:
features['vix_accel_5d'] = features['vix_velocity_5d'].diff(5)  # ✅ KEEP

# In MetaFeatureEngine.extract_cross_asset_relationships() - REMOVE:
meta['vix_acceleration'] = df['vix_velocity_5d'].diff(5)  # ❌ DELETE THIS LINE

# In MetaFeatureEngine.extract_rate_of_change_features() - REMOVE:
vel_5d = series.diff(5)
meta[f'{name}_acceleration_5d'] = vel_5d.diff(5)  # ❌ DELETE THIS LINE
# (this creates vix_acceleration_5d, SKEW_acceleration_5d, etc.)
```

**Consolidation:**
- Keep ONLY: `vix_accel_5d` (from _vix_dynamics)
- Remove: `vix_acceleration`, `vix_acceleration_5d`

---

### 1.6 Config.py Removals

**These features need to be removed from config.py because they're being deleted from feature_engine.py:**

```python
# In config.py, remove from VIX_BASE_FEATURES['dynamics']:
'vix_vol_10d',  # ❌ REMOVE (use from _vix_dynamics, not meta)
'vix_vol_21d',  # ❌ REMOVE (use from _vix_dynamics, not meta)

# In CROSS_ASSET_BASE_FEATURES['spx_vix_relationship']:
# No changes needed

# In CBOE_BASE_FEATURES - remove:
'skew_velocity_5d',     # ❌ REMOVE (use SKEW_velocity_5d_pct)
'skew_velocity_21d',    # ❌ REMOVE (use SKEW_velocity_21d_pct)
'pcc_velocity_10d',     # ❌ REMOVE (use PCC_velocity_10d_pct)
'pcci_velocity_10d',    # ❌ REMOVE (use PCCI_velocity_10d_pct)

# In META_FEATURES['cross_asset_relationships']:
'risk_premium',         # ❌ REMOVE (duplicate of vix_vs_rv_21d)

# In META_FEATURES['rate_of_change']:
'vix_velocity_3d',      # ❌ REMOVE (use vix_velocity_3d_pct)
'SKEW_acceleration_5d', # ❌ REMOVE (redundant with vix_accel_5d pattern)
'SKEW_jerk_5d',         # ❌ REMOVE (not useful)
```

---

## PART 2: Bug Fixes

### 2.1 Fix is_opex_week Generation

**Location:** `feature_engine.py` in `_calendar_features()`

**Current Broken Code:**
```python
def _calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    
    features['month'] = index.month
    features['day_of_week'] = index.dayofweek
    features['day_of_month'] = index.day
    
    # ❌ BROKEN: This loop logic is flawed
    for date in index:
        try:
            third_fridays = pd.date_range(
                start=date.replace(day=1),
                end=date.replace(day=1)+pd.offsets.MonthEnd(1),
                freq='W-FRI'
            )
            if len(third_fridays) >= 3:
                third_friday = third_fridays[2]
                if third_friday - pd.Timedelta(days=4) <= date <= third_friday:
                    features.loc[date, 'is_opex_week'] = 1
        except:
            pass
    
    return features
```

**Fixed Code:**
```python
def _calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar effects with proper OPEX detection"""
    features = pd.DataFrame(index=index)
    
    features['month'] = index.month
    features['day_of_week'] = index.dayofweek
    features['day_of_month'] = index.day
    
    # === FIX: Vectorized OPEX detection ===
    features['is_opex_week'] = 0
    features['is_opex_day'] = 0
    features['days_to_opex'] = 0
    
    for date in index:
        # Find all Fridays in the month
        month_start = date.replace(day=1)
        month_end = month_start + pd.offsets.MonthEnd(1)
        fridays = pd.date_range(start=month_start, end=month_end, freq='W-FRI')
        
        if len(fridays) >= 3:
            third_friday = fridays[2]  # 0-indexed: [0]=1st, [1]=2nd, [2]=3rd
            
            # OPEX week is Monday through Friday of that week
            week_start = third_friday - pd.Timedelta(days=4)  # Monday
            
            # Check if current date is in OPEX week
            if week_start <= date <= third_friday:
                features.loc[date, 'is_opex_week'] = 1
                
            # Check if current date IS the OPEX Friday
            if date == third_friday:
                features.loc[date, 'is_opex_day'] = 1
            
            # Calculate days to/from OPEX (useful for cycle analysis)
            if date < third_friday:
                features.loc[date, 'days_to_opex'] = (third_friday - date).days
            else:
                # After OPEX, count negative (days since)
                features.loc[date, 'days_to_opex'] = -((date - third_friday).days)
    
    # Add OPEX cycle phase (0-1 normalized through the month)
    features['opex_cycle_phase'] = (features['day_of_month'] / 21.0).clip(0, 1)
    
    return features
```

**Verification:**
- `is_opex_week` should have values for ~20% of trading days (1 week per month = ~4-5 weeks per year)
- `is_opex_day` should have values for ~1 day per month (12 days per year)
- No NaN values except where month doesn't have 3 Fridays (rare edge case)

---

### 2.2 Fix GOLDSILVER_zscore_63d Computation

**Location:** `feature_engine.py` in `_cboe_features()`

**Problem:** GOLDSILVER data exists (11 years) but zscore computation is failing

**Investigation Needed:**
```python
# In _cboe_features(), check if GOLDSILVER is being processed correctly
# The issue is likely in how it's being z-scored

# CURRENT CODE:
for col in cboe_data.columns:
    features[col] = cboe_data[col]
    features[f'{col}_zscore_63d'] = calculate_robust_zscore(cboe_data[col].shift(1), 63)

# If GOLDSILVER has data but zscore is 100% missing, likely causes:
# 1. GOLDSILVER values are constant (no variance)
# 2. GOLDSILVER has too many NaN values in rolling windows
# 3. Data alignment issue (wrong index)
```

**Debug Steps:**
```python
# Add after fetching CBOE data in build_complete_features():
if 'GOLDSILVER' in cboe_data.columns:
    gs = cboe_data['GOLDSILVER']
    print(f"GOLDSILVER debug:")
    print(f"  Total values: {len(gs)}")
    print(f"  Non-null: {gs.notna().sum()}")
    print(f"  Unique values: {gs.nunique()}")
    print(f"  Range: {gs.min():.2f} to {gs.max():.2f}")
    print(f"  Std: {gs.std():.4f}")
    print(f"  Sample values:\n{gs.dropna().head(10)}")
```

**Expected Fix:**
```python
# If issue is constant values, modify calculate_robust_zscore to handle:
def calculate_robust_zscore(series, window, min_std=1e-8):
    """
    FIXED: Prevent inf values from division by zero
    When std=0 (constant values), return 0 instead of inf
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    
    # If std is too small, the data is essentially constant - return 0
    rolling_std = rolling_std.replace(0, np.nan)
    rolling_std = rolling_std.clip(lower=min_std)
    
    zscore = (series - rolling_mean) / rolling_std
    
    # Replace any remaining inf/nan from division issues
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    
    return zscore
```

---

### 2.3 Handle GAMMA Sparsity (3yr history)

**Location:** `feature_engine.py` in `_cboe_features()`

**Current Approach:** GAMMA is treated like any other CBOE indicator, but it only has 727 rows (vs 2743 for SKEW)

**Enhancement Strategy:**

```python
# Add to _cboe_features() after the main loop:

# === Special handling for sparse GAMMA data ===
if 'GAMMA' in cboe_data.columns:
    gamma = cboe_data['GAMMA']
    
    # Create availability indicator (this missingness itself is a signal)
    features['GAMMA_available'] = gamma.notna().astype(int)
    
    # Days since last GAMMA update (staleness)
    gamma_available_dates = gamma.dropna().index
    features['days_since_gamma_update'] = 0
    
    for date in features.index:
        past_gamma = gamma_available_dates[gamma_available_dates <= date]
        if len(past_gamma) > 0:
            days_since = (date - past_gamma[-1]).days
            features.loc[date, 'days_since_gamma_update'] = days_since
        else:
            features.loc[date, 'days_since_gamma_update'] = 9999
    
    # Forward-fill GAMMA with decay (recent values more reliable)
    features['GAMMA_filled'] = gamma.fillna(method='ffill', limit=5)
    features['GAMMA_stale'] = (features['days_since_gamma_update'] > 5).astype(int)
    
    # Only calculate zscore when GAMMA is actually available
    features['GAMMA_zscore_when_available'] = (
        calculate_robust_zscore(gamma.shift(1), 63) * features['GAMMA_available']
    )
```

**Config.py Update:**
```python
# In CBOE_BASE_FEATURES['other_cboe'], CHANGE:
'GAMMA', 'GAMMA_change_21d', 'GAMMA_zscore_63d',  # ❌ OLD

# TO:
'GAMMA', 'GAMMA_available', 'GAMMA_filled', 'GAMMA_stale', 
'days_since_gamma_update', 'GAMMA_zscore_when_available',  # ✅ NEW
```

---

## PART 3: Futures Term Structure Expansion

### 3.1 Current State Analysis

**VX Futures (VX1-VX2, VX2-VX1_RATIO):**
- ✅ Has: spread, ratio, zscore, percentile, regime
- ❌ Missing: curvature, acceleration, steep/flat flags, historical context

**CL Futures (CL1-CL2):**
- ✅ Has: spread, zscore, regime
- ❌ Missing: backwardation/contango intensity, curve shape, volatility

**DX Futures (DX1-DX2):**
- ✅ Has: spread, zscore
- ❌ Missing: carry signals, curve dynamics

### 3.2 Enhanced VX Term Structure Features

**Add to `FuturesFeatureEngine.extract_vix_futures_features()`:**

```python
def extract_vix_futures_features(vx_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    ENHANCED: VIX futures with full term structure analysis
    """
    features = pd.DataFrame()
    
    if 'VX1-VX2' not in vx_data:
        return features
    
    spread = vx_data['VX1-VX2']
    
    # === EXISTING FEATURES (keep all) ===
    features['vx_spread_velocity_5d'] = spread.diff(5)
    features['vx_spread_regime'] = calculate_regime_with_validation(
        spread, bins=[-10, -1, 0, 1, 10], labels=[0,1,2,3], feature_name='vx_spread'
    )
    features['vx_spread_zscore_63d'] = calculate_robust_zscore(spread, 63)
    features['vx_spread_percentile_63d'] = calculate_percentile_with_validation(spread, 63)
    
    # === NEW FEATURES ===
    
    # 1. CURVE SHAPE DYNAMICS
    features['vx_curve_curvature'] = spread.diff(5).diff(5)  # 2nd derivative
    features['vx_curve_jerk'] = spread.diff(3).diff(3).diff(3)  # 3rd derivative
    
    # 2. CONTANGO/BACKWARDATION INTENSITY
    features['vx_contango_strength'] = spread.clip(upper=0).abs()  # Magnitude when contango
    features['vx_backwardation_strength'] = spread.clip(lower=0)    # Magnitude when backwardation
    
    # Persistent contango/backwardation (21-day count)
    features['vx_contango_days_21d'] = (spread < 0).rolling(21).sum()
    features['vx_backwardation_days_21d'] = (spread > 0).rolling(21).sum()
    
    # 3. SPREAD VOLATILITY
    features['vx_spread_volatility_10d'] = spread.rolling(10).std()
    features['vx_spread_volatility_21d'] = spread.rolling(21).std()
    
    # 4. SPREAD MOMENTUM REGIMES
    spread_mom_5d = spread.diff(5)
    spread_mom_21d = spread.diff(21)
    
    features['vx_spread_momentum_regime'] = np.where(
        spread_mom_5d > 0, 1,   # Steepening
        np.where(spread_mom_5d < 0, -1, 0)  # Flattening
    )
    
    features['vx_spread_acceleration_regime'] = np.where(
        spread_mom_5d.diff(5) > 0, 1,   # Accelerating change
        np.where(spread_mom_5d.diff(5) < 0, -1, 0)
    )
    
    # 5. EXTREME CURVE STATES
    spread_pct_63d = calculate_percentile_with_validation(spread, 63)
    features['vx_extreme_contango'] = (spread_pct_63d < 10).astype(int)
    features['vx_extreme_backwardation'] = (spread_pct_63d > 90).astype(int)
    
    # 6. HISTORICAL CONTEXT
    # Compare current spread to different lookback periods
    features['vx_spread_vs_ma21'] = spread - spread.rolling(21).mean()
    features['vx_spread_vs_ma63'] = spread - spread.rolling(63).mean()
    features['vx_spread_vs_ma126'] = spread - spread.rolling(126).mean()
    
    # === RATIO FEATURES ===
    if 'VX2-VX1_RATIO' in vx_data:
        ratio = vx_data['VX2-VX1_RATIO']
        
        # EXISTING (keep)
        features['vx_ratio_velocity_10d'] = ratio.diff(10)
        features['vx_term_structure_regime'] = calculate_regime_with_validation(
            ratio, bins=[-1, -0.05, 0, 0.05, 1], labels=[0,1,2,3], feature_name='vx_ratio'
        )
        
        # NEW RATIO FEATURES
        features['vx_ratio_momentum_5d'] = ratio.diff(5)
        features['vx_ratio_acceleration'] = ratio.diff(5).diff(5)
        
        # Ratio volatility (curve stability)
        features['vx_ratio_volatility_21d'] = ratio.rolling(21).std()
        
        # Ratio extremes
        ratio_pct_63d = calculate_percentile_with_validation(ratio, 63)
        features['vx_ratio_extreme_low'] = (ratio_pct_63d < 10).astype(int)
        features['vx_ratio_extreme_high'] = (ratio_pct_63d > 90).astype(int)
        
        # === SPREAD-RATIO DIVERGENCE ===
        # When spread and ratio disagree, it signals curve instability
        spread_rank = spread.rolling(63).rank(pct=True)
        ratio_rank = ratio.rolling(63).rank(pct=True)
        features['vx_spread_ratio_divergence'] = (spread_rank - ratio_rank).abs()
        
        # Carry signal: ratio * spread velocity
        features['vx_carry_signal'] = ratio * spread.diff(10)
    
    return features
```

### 3.3 Enhanced CL Term Structure Features

**Add to `FuturesFeatureEngine.extract_commodity_futures_features()`:**

```python
def extract_commodity_futures_features(futures_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    ENHANCED: Commodity futures with full contango/backwardation analysis
    """
    features = pd.DataFrame()
    
    if 'CL1-CL2' not in futures_data:
        return features
    
    cl_spread = futures_data['CL1-CL2']
    
    # === EXISTING FEATURES (keep) ===
    features['CL1-CL2_velocity_5d'] = cl_spread.diff(5)
    features['CL1-CL2_zscore_63d'] = calculate_robust_zscore(cl_spread, 63)
    features['oil_term_regime'] = calculate_regime_with_validation(
        cl_spread, bins=[-10, -1, 0, 2, 20], labels=[0,1,2,3], feature_name='cl_spread'
    )
    
    # === NEW FEATURES ===
    
    # 1. CONTANGO/BACKWARDATION METRICS
    # Crude oil: backwardation (CL1 > CL2) = bullish, contango = bearish
    features['cl_backwardation_intensity'] = cl_spread.clip(lower=0)
    features['cl_contango_intensity'] = cl_spread.clip(upper=0).abs()
    
    # Persistent states
    features['cl_backwardation_days_21d'] = (cl_spread > 0).rolling(21).sum()
    features['cl_contango_days_21d'] = (cl_spread < 0).rolling(21).sum()
    
    # 2. CURVE SHAPE
    features['cl_curve_steepness'] = cl_spread / cl_spread.rolling(63).std()
    features['cl_curve_acceleration'] = cl_spread.diff(5).diff(5)
    
    # 3. SPREAD MOMENTUM
    features['cl_spread_momentum_5d'] = cl_spread.diff(5)
    features['cl_spread_momentum_21d'] = cl_spread.diff(21)
    features['cl_spread_acceleration_5d'] = cl_spread.diff(5).diff(5)
    
    # Momentum regime
    features['cl_spread_momentum_regime'] = np.where(
        features['cl_spread_momentum_5d'] > 0, 1,
        np.where(features['cl_spread_momentum_5d'] < 0, -1, 0)
    )
    
    # 4. SPREAD VOLATILITY
    features['cl_spread_volatility_10d'] = cl_spread.rolling(10).std()
    features['cl_spread_volatility_21d'] = cl_spread.rolling(21).std()
    
    # 5. HISTORICAL CONTEXT
    features['cl_spread_vs_ma21'] = cl_spread - cl_spread.rolling(21).mean()
    features['cl_spread_vs_ma63'] = cl_spread - cl_spread.rolling(63).mean()
    
    # 6. EXTREME STATES
    cl_pct_63d = calculate_percentile_with_validation(cl_spread, 63)
    features['cl_extreme_contango'] = (cl_pct_63d < 10).astype(int)
    features['cl_extreme_backwardation'] = (cl_pct_63d > 90).astype(int)
    
    # 7. CARRY TRADE SIGNALS
    # Strong backwardation = bullish carry (long front month)
    # Strong contango = bearish carry (short front month)
    features['cl_carry_strength'] = cl_spread / cl_spread.rolling(126).std()
    
    # === CRUDE OIL SPOT PRICE FEATURES ===
    if 'Crude_Oil' in futures_data:
        price = futures_data['Crude_Oil']
        
        # EXISTING (keep)
        for window in [10, 21, 63]:
            features[f'crude_oil_ret_{window}d'] = price.pct_change(window) * 100
        
        features['crude_oil_vol_21d'] = (
            price.pct_change().rolling(21).std() * np.sqrt(252) * 100
        )
        features['crude_oil_zscore_63d'] = calculate_robust_zscore(price, 63)
        
        # NEW: Spot-Futures Relationship
        # When spot rallies but curve stays in contango = temporary move
        # When spot rallies AND curve goes to backwardation = structural move
        price_mom = price.pct_change(10)
        spread_mom = cl_spread.diff(10)
        
        features['cl_spot_curve_alignment'] = (
            (price_mom > 0) & (spread_mom > 0)  # Both bullish
        ).astype(int) - (
            (price_mom < 0) & (spread_mom < 0)  # Both bearish
        ).astype(int)
        
        # Spot momentum vs curve momentum divergence
        features['cl_spot_curve_divergence'] = (
            price_mom.rank(pct=True) - spread_mom.rank(pct=True)
        ).abs()
    
    return features
```

### 3.4 Enhanced DX Term Structure Features

**Add to `FuturesFeatureEngine.extract_dollar_futures_features()`:**

```python
def extract_dollar_futures_features(dollar_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    ENHANCED: Dollar index futures with carry and curve dynamics
    """
    features = pd.DataFrame()
    
    if 'DX1-DX2' not in dollar_data:
        return features
    
    dx_spread = dollar_data['DX1-DX2']
    
    # === EXISTING FEATURES (keep) ===
    features['DX1-DX2_velocity_5d'] = dx_spread.diff(5)
    features['DX1-DX2_zscore_63d'] = calculate_robust_zscore(dx_spread, 63)
    
    # === NEW FEATURES ===
    
    # 1. CARRY DYNAMICS
    # Dollar carry: positive spread = dollar strength expected
    features['dx_carry_positive'] = (dx_spread > 0).astype(int)
    features['dx_carry_strength'] = dx_spread / dx_spread.rolling(63).std()
    
    # 2. SPREAD MOMENTUM
    features['dx_spread_momentum_5d'] = dx_spread.diff(5)
    features['dx_spread_momentum_21d'] = dx_spread.diff(21)
    features['dx_spread_acceleration'] = dx_spread.diff(5).diff(5)
    
    features['dx_spread_momentum_regime'] = np.where(
        features['dx_spread_momentum_5d'] > 0, 1,
        np.where(features['dx_spread_momentum_5d'] < 0, -1, 0)
    )
    
    # 3. SPREAD VOLATILITY
    features['dx_spread_volatility_10d'] = dx_spread.rolling(10).std()
    features['dx_spread_volatility_21d'] = dx_spread.rolling(21).std()
    
    # 4. HISTORICAL CONTEXT
    features['dx_spread_vs_ma21'] = dx_spread - dx_spread.rolling(21).mean()
    features['dx_spread_vs_ma63'] = dx_spread - dx_spread.rolling(63).mean()
    
    # 5. EXTREME STATES
    dx_pct_63d = calculate_percentile_with_validation(dx_spread, 63)
    features['dx_extreme_carry_negative'] = (dx_pct_63d < 10).astype(int)
    features['dx_extreme_carry_positive'] = (dx_pct_63d > 90).astype(int)
    
    # === DOLLAR INDEX SPOT FEATURES ===
    if 'Dollar_Index' in dollar_data:
        dxy = dollar_data['Dollar_Index']
        
        # EXISTING (keep)
        for window in [10, 21, 63]:
            features[f'dxy_ret_{window}d'] = dxy.pct_change(window) * 100
        
        for window in [50, 200]:
            ma = dxy.rolling(window).mean()
            features[f'dxy_vs_ma{window}'] = ((dxy - ma) / ma.replace(0, np.nan)) * 100
        
        features['dxy_vol_21d'] = (
            dxy.pct_change().rolling(21).std() * np.sqrt(252) * 100
        )
        
        dxy_ma = dxy.rolling(200).mean()
        features['dxy_regime'] = (dxy > dxy_ma).astype(int)
        
        # NEW: Spot-Futures Alignment
        dxy_mom = dxy.pct_change(10)
        spread_mom = dx_spread.diff(10)
        
        features['dx_spot_curve_alignment'] = (
            (dxy_mom > 0) & (spread_mom > 0)  # Both bullish dollar
        ).astype(int) - (
            (dxy_mom < 0) & (spread_mom < 0)  # Both bearish dollar
        ).astype(int)
        
        features['dx_spot_curve_divergence'] = (
            dxy_mom.rank(pct=True) - spread_mom.rank(pct=True)
        ).abs()
        
        # Dollar strength vs carry expectations
        features['dx_carry_surprise'] = (
            features['dxy_ret_21d'] - dx_spread * 100
        )
    
    return features
```

---

## PART 4: Treasury Yield Curve Features (NEW)

### 4.1 Overview

**Currently Missing:** Treasury yield curve features are COMPLETELY absent from the system

**Why Critical:**
- Yield curve inversion predicts recessions (and VIX spikes)
- Term spread compression = stress indicator
- Rate volatility = market uncertainty
- Fed policy expectations embedded in curve shape

### 4.2 Data Sources

**FRED Series to Fetch:**

```python
TREASURY_YIELD_SERIES = {
    'rates': {
        'DGS1MO': '1-Month',     # 1-month Treasury
        'DGS3MO': '3-Month',     # 3-month Treasury
        'DGS6MO': '6-Month',     # 6-month Treasury
        'DGS1': '1-Year',        # 1-year Treasury
        'DGS2': '2-Year',        # 2-year Treasury
        'DGS5': '5-Year',        # 5-year Treasury
        'DGS10': '10-Year',      # 10-year Treasury
        'DGS30': '30-Year',      # 30-year Treasury
    },
    'policy': {
        'DFEDTARU': 'Fed_Funds_Upper',  # Fed Funds Target Upper
        'DFEDTARL': 'Fed_Funds_Lower',  # Fed Funds Target Lower
    },
    'credit': {
        'BAMLH0A0HYM2': 'HY_Spread',    # High Yield OAS
        'BAMLC0A0CM': 'IG_Spread',      # Investment Grade OAS
    }
}
```

### 4.3 Yield Curve Feature Engine

**Create new class in `feature_engine.py`:**

```python
class TreasuryYieldFeatureEngine:
    """
    Extract comprehensive Treasury yield curve features.
    Focus: Term spreads, curve shape, inversions, rate volatility.
    """
    
    @staticmethod
    def extract_term_spreads(yields: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all meaningful term spreads.
        Critical spreads for recession/volatility prediction.
        """
        features = pd.DataFrame(index=yields.index)
        
        # Ensure we have required yields
        required = ['DGS3MO', 'DGS2', 'DGS10', 'DGS30']
        if not all(col in yields.columns for col in required):
            return features
        
        # === KEY TERM SPREADS ===
        
        # 1. 10Y-2Y Spread (most famous recession indicator)
        features['yield_10y2y'] = yields['DGS10'] - yields['DGS2']
        features['yield_10y2y_inverted'] = (features['yield_10y2y'] < 0).astype(int)
        features['yield_10y2y_zscore'] = calculate_robust_zscore(features['yield_10y2y'], 252)
        
        # 2. 10Y-3M Spread (NY Fed recession model uses this)
        features['yield_10y3m'] = yields['DGS10'] - yields['DGS3MO']
        features['yield_10y3m_inverted'] = (features['yield_10y3m'] < 0).astype(int)
        
        # 3. 2Y-3M Spread (front-end curve)
        features['yield_2y3m'] = yields['DGS2'] - yields['DGS3MO']
        features['yield_2y3m_inverted'] = (features['yield_2y3m'] < 0).astype(int)
        
        # 4. 30Y-10Y Spread (long-end steepness)
        if 'DGS30' in yields.columns:
            features['yield_30y10y'] = yields['DGS30'] - yields['DGS10']
        
        # 5. 5Y-2Y Spread (belly of curve)
        if 'DGS5' in yields.columns:
            features['yield_5y2y'] = yields['DGS5'] - yields['DGS2']
        
        # === SPREAD DYNAMICS ===
        
        for spread_name in ['yield_10y2y', 'yield_10y3m', 'yield_2y3m']:
            if spread_name not in features.columns:
                continue
            
            spread = features[spread_name]
            
            # Momentum
            features[f'{spread_name}_velocity_10d'] = spread.diff(10)
            features[f'{spread_name}_velocity_21d'] = spread.diff(21)
            features[f'{spread_name}_velocity_63d'] = spread.diff(63)
            
            # Acceleration
            features[f'{spread_name}_acceleration'] = spread.diff(10).diff(10)
            
            # Historical context
            features[f'{spread_name}_vs_ma63'] = spread - spread.rolling(63).mean()
            features[f'{spread_name}_percentile_252d'] = calculate_percentile_with_validation(spread, 252)
            
            # Extreme flatness/steepness
            pct = calculate_percentile_with_validation(spread, 252)
            features[f'{spread_name}_extreme_flat'] = (pct < 10).astype(int)
            features[f'{spread_name}_extreme_steep'] = (pct > 90).astype(int)
        
        # === INVERSION INDICATORS ===
        
        # Count of inverted spreads (0-3)
        features['yield_inversion_count'] = (
            features['yield_10y2y_inverted'] +
            features['yield_10y3m_inverted'] +
            features['yield_2y3m_inverted']
        )
        
        # Days inverted (persistence indicator)
        features['yield_10y2y_inversion_days'] = (
            features['yield_10y2y_inverted'].rolling(252).sum()
        )
        
        # Time since last inversion
        features['days_since_inversion'] = 0
        inversion_dates = features[features['yield_10y2y_inverted'] == 1].index
        
        for date in features.index:
            past_inversions = inversion_dates[inversion_dates < date]
            if len(past_inversions) > 0:
                features.loc[date, 'days_since_inversion'] = (date - past_inversions[-1]).days
            else:
                features.loc[date, 'days_since_inversion'] = 9999
        
        return features
    
    @staticmethod
    def extract_curve_shape(yields: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze overall yield curve shape using PCA-like metrics.
        """
        features = pd.DataFrame(index=yields.index)
        
        # Need at least 4 points on the curve
        yield_cols = ['DGS3MO', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
        available_cols = [c for c in yield_cols if c in yields.columns]
        
        if len(available_cols) < 4:
            return features
        
        # === LEVEL, SLOPE, CURVATURE (3-factor model) ===
        
        # Level: Average of all yields
        features['yield_level'] = yields[available_cols].mean(axis=1)
        features['yield_level_velocity_21d'] = features['yield_level'].diff(21)
        features['yield_level_zscore_63d'] = calculate_robust_zscore(features['yield_level'], 63)
        
        # Slope: Long minus short
        if 'DGS10' in yields.columns and 'DGS3MO' in yields.columns:
            features['yield_slope'] = yields['DGS10'] - yields['DGS3MO']
            features['yield_slope_velocity_21d'] = features['yield_slope'].diff(21)
            features['yield_slope_regime'] = calculate_regime_with_validation(
                features['yield_slope'],
                bins=[-5, 0, 1, 2, 10],
                labels=[0, 1, 2, 3],
                feature_name='yield_slope'
            )
        
        # Curvature: Middle minus average of short and long
        if all(c in yields.columns for c in ['DGS2', 'DGS5', 'DGS10']):
            features['yield_curvature'] = (
                2 * yields['DGS5'] - yields['DGS2'] - yields['DGS10']
            )
            features['yield_curvature_velocity_21d'] = features['yield_curvature'].diff(21)
        
        # === CURVE STEEPNESS INDEX ===
        # Variance across the curve (high variance = steep, low = flat)
        features['yield_curve_variance'] = yields[available_cols].std(axis=1)
        features['yield_curve_flatness'] = 1 / (features['yield_curve_variance'] + 0.01)
        
        # === BUTTERFLY SPREAD ===
        if all(c in yields.columns for c in ['DGS2', 'DGS5', 'DGS10']):
            # Butterfly = (2 * belly) - (short + long)
            features['yield_butterfly'] = (
                2 * yields['DGS5'] - yields['DGS2'] - yields['DGS10']
            )
            features['yield_butterfly_velocity_10d'] = features['yield_butterfly'].diff(10)
        
        return features
    
    @staticmethod
    def extract_rate_volatility(yields: pd.DataFrame) -> pd.DataFrame:
        """
        Rate volatility and MOVE index relationships.
        """
        features = pd.DataFrame(index=yields.index)
        
        # === INDIVIDUAL YIELD VOLATILITY ===
        for col in ['DGS2', 'DGS10', 'DGS30']:
            if col not in yields.columns:
                continue
            
            rate = yields[col]
            rate_chg = rate.diff()
            
            # Realized vol of rates
            features[f'{col}_vol_10d'] = rate_chg.rolling(10).std() * np.sqrt(252)
            features[f'{col}_vol_21d'] = rate_chg.rolling(21).std() * np.sqrt(252)
            
            # Rate momentum
            features[f'{col}_momentum_10d'] = rate.diff(10)
            features[f'{col}_momentum_21d'] = rate.diff(21)
            
            # Z-score
            features[f'{col}_zscore_63d'] = calculate_robust_zscore(rate, 63)
        
        # === CROSS-RATE VOLATILITY ===
        if all(c in yields.columns for c in ['DGS2', 'DGS10']):
            # 2Y-10Y spread volatility
            spread = yields['DGS10'] - yields['DGS2']
            features['yield_spread_volatility_21d'] = spread.diff().rolling(21).std() * np.sqrt(252)
            
            # Correlation between 2Y and 10Y (breaks down in stress)
            features['yield_2y10y_corr_21d'] = (
                yields['DGS2'].pct_change().rolling(21).corr(yields['DGS10'].pct_change())
            )
        
        return features
    
    @staticmethod
    def extract_policy_expectations(yields: pd.DataFrame, fed_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract Fed policy expectations from yield curve.
        """
        features = pd.DataFrame(index=yields.index)
        
        # === IMPLIED FED POLICY FROM FRONT END ===
        if all(c in yields.columns for c in ['DGS3MO', 'DGS2']):
            # Front-end curve implies near-term policy expectations
            features['policy_expectations_2y'] = yields['DGS2']
            features['policy_expectations_velocity'] = yields['DGS2'].diff(21)
            
            # When 2Y moves a lot, policy uncertainty is high
            features['policy_uncertainty'] = yields['DGS2'].diff().rolling(21).std() * np.sqrt(252)
        
        # === FED FUNDS vs MARKET RATES ===
        if fed_data is not None and 'Fed_Funds_Upper' in fed_data.columns:
            fed_rate = fed_data['Fed_Funds_Upper']
            
            if 'DGS3MO' in yields.columns:
                # Spread between 3M and Fed Funds (credit risk + expectations)
                features['fed_3m_spread'] = yields['DGS3MO'] - fed_rate
                features['fed_3m_spread_velocity'] = features['fed_3m_spread'].diff(21)
            
            if 'DGS2' in yields.columns:
                # How many cuts/hikes priced in?
                features['implied_rate_change_2y'] = yields['DGS2'] - fed_rate
        
        # === RATE CHANGE REGIMES ===
        if 'DGS2' in yields.columns:
            rate_change_21d = yields['DGS2'].diff(21)
            features['rate_regime'] = calculate_regime_with_validation(
                rate_change_21d,
                bins=[-10, -0.5, 0, 0.5, 10],
                labels=[0, 1, 2, 3],  # cutting, stable, hiking
                feature_name='rate_regime'
            )
        
        return features
    
    @staticmethod
    def extract_credit_spreads(credit_data: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
        """
        Credit spread features (HY, IG) vs Treasuries.
        """
        features = pd.DataFrame(index=yields.index)
        
        if 'HY_Spread' not in credit_data.columns:
            return features
        
        hy_spread = credit_data['HY_Spread']
        
        # === HIGH YIELD SPREAD FEATURES ===
        features['hy_spread'] = hy_spread
        features['hy_spread_velocity_21d'] = hy_spread.diff(21)
        features['hy_spread_zscore_63d'] = calculate_robust_zscore(hy_spread, 63)
        features['hy_spread_percentile_252d'] = calculate_percentile_with_validation(hy_spread, 252)
        
        # Extreme stress
        hy_pct = calculate_percentile_with_validation(hy_spread, 252)
        features['hy_stress_elevated'] = (hy_pct > 75).astype(int)
        features['hy_stress_extreme'] = (hy_pct > 90).astype(int)
        
        # HY spread volatility
        features['hy_spread_volatility_21d'] = hy_spread.diff().rolling(21).std()
        
        # === INVESTMENT GRADE SPREAD ===
        if 'IG_Spread' in credit_data.columns:
            ig_spread = credit_data['IG_Spread']
            
            features['ig_spread'] = ig_spread
            features['ig_spread_velocity_21d'] = ig_spread.diff(21)
            features['ig_spread_zscore_63d'] = calculate_robust_zscore(ig_spread, 63)
            
            # HY-IG spread (credit quality flight)
            features['hy_ig_spread_diff'] = hy_spread - ig_spread
            features['hy_ig_ratio'] = hy_spread / ig_spread.replace(0, np.nan)
            
            # When HY/IG ratio expands, credit stress is rising
            features['credit_quality_flight'] = features['hy_ig_ratio'].diff(21)
        
        # === CREDIT-RATES RELATIONSHIP ===
        if 'DGS10' in yields.columns:
            # HY spread vs 10Y yield correlation
            features['hy_10y_corr_63d'] = (
                hy_spread.pct_change().rolling(63).corr(yields['DGS10'].pct_change())
            )
            
            # When correlation breaks (usually negative), stress mode
            features['hy_10y_corr_breakdown'] = (features['hy_10y_corr_63d'] > -0.3).astype(int)
        
        return features
```

### 4.4 Integration into UnifiedFeatureEngine

**Add to `UnifiedFeatureEngine.build_complete_features()`:**

```python
# After STAGE 7 (FRED features), add STAGE 7.5:

# === STAGE 7.5: Treasury Yield Curve Features ===
print("\n[7.5/8] Treasury yield curve features...")
treasury_features = self._build_treasury_features(start_str, end_str, spx.index)
print(f"   ✅ {len(treasury_features.columns)} treasury features")

# Then in STAGE 8 consolidation:
all_features = [
    base_features,
    cboe_features,
    futures_features,
    macro_features,
    meta_features,
    fred_features,
    treasury_features,  # ADD THIS
]
```

**Add new method to UnifiedFeatureEngine:**

```python
def _build_treasury_features(self, start_str: str, end_str: str, 
                             target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build comprehensive Treasury yield curve features.
    """
    try:
        from feature_engine import TreasuryYieldFeatureEngine
        treasury_engine = TreasuryYieldFeatureEngine()
        
        # Fetch Treasury yields from FRED
        yield_series = {
            'DGS1MO': '1M_Treasury',
            'DGS3MO': '3M_Treasury',
            'DGS6MO': '6M_Treasury',
            'DGS1': '1Y_Treasury',
            'DGS2': '2Y_Treasury',
            'DGS5': '5Y_Treasury',
            'DGS10': '10Y_Treasury',
            'DGS30': '30Y_Treasury',
        }
        
        yields_data = []
        for symbol, name in yield_series.items():
            series = self.fetcher.fetch_fred(symbol, start_str, end_str, incremental=True)
            if series is not None:
                series.name = symbol
                yields_data.append(series)
        
        if not yields_data:
            return pd.DataFrame(index=target_index)
        
        yields = pd.DataFrame(yields_data).T.reindex(target_index, method='ffill')
        
        # Fetch Fed Funds and credit spreads
        fed_data = None
        credit_data = pd.DataFrame(index=target_index)
        
        # Fed Funds
        fed_upper = self.fetcher.fetch_fred('DFEDTARU', start_str, end_str, incremental=True)
        if fed_upper is not None:
            fed_data = pd.DataFrame({'Fed_Funds_Upper': fed_upper}).reindex(target_index, method='ffill')
        
        # Credit spreads
        hy_spread = self.fetcher.fetch_fred('BAMLH0A0HYM2', start_str, end_str, incremental=True)
        if hy_spread is not None:
            credit_data['HY_Spread'] = hy_spread.reindex(target_index, method='ffill')
        
        ig_spread = self.fetcher.fetch_fred('BAMLC0A0CM', start_str, end_str, incremental=True)
        if ig_spread is not None:
            credit_data['IG_Spread'] = ig_spread.reindex(target_index, method='ffill')
        
        # Extract all yield curve features
        feature_groups = [
            treasury_engine.extract_term_spreads(yields),
            treasury_engine.extract_curve_shape(yields),
            treasury_engine.extract_rate_volatility(yields),
            treasury_engine.extract_policy_expectations(yields, fed_data),
            treasury_engine.extract_credit_spreads(credit_data, yields),
        ]
        
        return pd.concat(feature_groups, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
        
    except Exception as e:
        print(f"   ⚠️ Treasury features error: {e}")
        return pd.DataFrame(index=target_index)
```

### 4.5 Config.py Updates for Treasury Features

**Add new section to config.py:**

```python
# Treasury Yield Curve Features
TREASURY_FEATURES = {
    'term_spreads': [
        'yield_10y2y', 'yield_10y2y_inverted', 'yield_10y2y_zscore',
        'yield_10y3m', 'yield_10y3m_inverted',
        'yield_2y3m', 'yield_2y3m_inverted',
        'yield_30y10y', 'yield_5y2y',
        'yield_10y2y_velocity_21d', 'yield_10y2y_acceleration',
        'yield_10y2y_vs_ma63', 'yield_10y2y_percentile_252d',
        'yield_inversion_count', 'yield_10y2y_inversion_days',
        'days_since_inversion'
    ],
    'curve_shape': [
        'yield_level', 'yield_level_velocity_21d', 'yield_level_zscore_63d',
        'yield_slope', 'yield_slope_velocity_21d', 'yield_slope_regime',
        'yield_curvature', 'yield_curvature_velocity_21d',
        'yield_curve_variance', 'yield_curve_flatness',
        'yield_butterfly', 'yield_butterfly_velocity_10d'
    ],
    'rate_volatility': [
        'DGS2_vol_21d', 'DGS10_vol_21d', 'DGS30_vol_21d',
        'DGS2_momentum_21d', 'DGS10_momentum_21d',
        'DGS2_zscore_63d', 'DGS10_zscore_63d',
        'yield_spread_volatility_21d', 'yield_2y10y_corr_21d'
    ],
    'policy_expectations': [
        'policy_expectations_2y', 'policy_expectations_velocity',
        'policy_uncertainty', 'fed_3m_spread', 'fed_3m_spread_velocity',
        'implied_rate_change_2y', 'rate_regime'
    ],
    'credit_spreads': [
        'hy_spread', 'hy_spread_velocity_21d', 'hy_spread_zscore_63d',
        'hy_spread_percentile_252d', 'hy_stress_elevated', 'hy_stress_extreme',
        'hy_spread_volatility_21d',
        'ig_spread', 'ig_spread_velocity_21d', 'ig_spread_zscore_63d',
        'hy_ig_spread_diff', 'hy_ig_ratio', 'credit_quality_flight',
        'hy_10y_corr_63d', 'hy_10y_corr_breakdown'
    ]
}
```

**Add to anomaly detection groups:**

```python
# Add to ANOMALY_FEATURE_GROUPS:
ANOMALY_FEATURE_GROUPS = {
    # ... existing groups ...
    
    'yield_curve_stress': (
        TREASURY_FEATURES['term_spreads'][:10] +
        TREASURY_FEATURES['curve_shape'][:5] +
        ['hy_stress_elevated', 'yield_inversion_count']
    ),
    
    'credit_stress': (
        TREASURY_FEATURES['credit_spreads'] +
        ['vix', 'vix_velocity_21d', 'spx_realized_vol_21d']
    ),
    
    'rate_volatility_regime': (
        TREASURY_FEATURES['rate_volatility'] +
        TREASURY_FEATURES['policy_expectations'] +
        ['vix', 'spx_realized_vol_21d', 'Bond_Vol_mom_21d']
    ),
}
```

---

## PART 5: Implementation Checklist

### Phase 1: Feature Removal (Priority: HIGH)

**Week 1 - Remove Perfect Duplicates**

1. **VIX/VIXCLS Cleanup** (5 min)
   - [ ] Search config.py for "VIXCLS"
   - [ ] Remove all 3 VIXCLS features
   - [ ] Verify no other references exist

2. **Feature Engine Duplicates** (30 min)
   - [ ] Edit `MetaFeatureEngine.extract_cross_asset_relationships()`
     - Remove `vol_of_vol_10d`, `vol_of_vol_21d` lines (~300-301)
     - Remove `risk_premium` line (~313)
   - [ ] Edit `UnifiedFeatureEngine._cboe_interactions()`
     - Remove 5 duplicate lines (vxth_vs_vix, vxth_premium, cor_term_slope, cor_avg, pc_equity_inst_spread)
   - [ ] Edit `FuturesFeatureEngine.extract_vix_futures_features()`
     - Remove vx_spread, vx_ratio, vx_spread_velocity_21d lines
   - [ ] Edit `FuturesFeatureEngine.extract_commodity_futures_features()`
     - Remove cl_spread line
   - [ ] Edit `FuturesFeatureEngine.extract_dollar_futures_features()`
     - Remove dx_spread line
   - [ ] Edit `UnifiedFeatureEngine._build_macro_features()`
     - Change loop to use `_ret_` instead of `_mom_` naming

3. **Moving Average Smoothing Removal** (10 min)
   - [ ] Remove vx_spread_ma10, vx_spread_ma21 from VX features
   - [ ] Remove cl_spread_ma10 from CL features
   - [ ] Remove dx_spread_ma10 from DX features
   - [ ] Remove dollar_crude_corr_breakdown from cross futures

4. **Velocity Consolidation** (20 min)
   - [ ] Edit `MetaFeatureEngine.extract_rate_of_change_features()`
     - Remove absolute velocity calculations, keep only _pct versions
   - [ ] Edit `UnifiedFeatureEngine._cboe_features()`
     - Remove `_change_21d` generation line for all CBOE indicators

5. **Acceleration Consolidation** (10 min)
   - [ ] Remove `vix_acceleration` line from MetaFeatureEngine
   - [ ] Remove `{name}_acceleration_5d` line from rate_of_change loop
   - [ ] Keep only `vix_accel_5d` from _vix_dynamics()

6. **Config.py Cleanup** (15 min)
   - [ ] Remove velocity features from CBOE_BASE_FEATURES
   - [ ] Remove risk_premium from META_FEATURES
   - [ ] Remove obsolete acceleration/jerk features
   - [ ] Remove VIXCLS references

**Verification Steps:**
```bash
# After edits, run these commands to verify cleanup:
grep -r "VIXCLS" config.py  # Should return nothing
grep -r "vol_of_vol" feature_engine.py  # Should return nothing
grep -r "_mom_" feature_engine.py  # Should be minimal
python integrated_system_production.py  # Should run without errors
```

---

### Phase 2: Bug Fixes (Priority: HIGH)

**Week 1 - Fix Broken Features**

1. **Fix is_opex_week** (20 min)
   - [ ] Replace `_calendar_features()` method with fixed version
   - [ ] Add `is_opex_day`, `days_to_opex`, `opex_cycle_phase` features
   - [ ] Test on known OPEX dates (e.g., Sept 15, 2023 = 3rd Friday)

2. **Debug GOLDSILVER_zscore_63d** (15 min)
   - [ ] Add debug print statements in `_cboe_features()`
   - [ ] Check if GOLDSILVER data exists and has variance
   - [ ] Update `calculate_robust_zscore()` if needed
   - [ ] Verify output is no longer 100% missing

3. **Handle GAMMA Sparsity** (30 min)
   - [ ] Add special GAMMA handling section to `_cboe_features()`
   - [ ] Create GAMMA_available, GAMMA_filled, GAMMA_stale features
   - [ ] Create days_since_gamma_update feature
   - [ ] Update config.py with new GAMMA features

**Verification Steps:**
```python
# Test OPEX detection
df = engine._calendar_features(pd.date_range('2023-01-01', '2024-01-01', freq='B'))
print(f"OPEX weeks: {df['is_opex_week'].sum()} / 252 days")  # Should be ~48-52
print(f"OPEX days: {df['is_opex_day'].sum()}")  # Should be 12

# Test GAMMA features
print(f"GAMMA available: {(features['GAMMA_available'] == 1).sum()} days")  # Should be ~727
```

---

### Phase 3: Futures Enhancement (Priority: MEDIUM)

**Week 2 - Expand Futures Features**

1. **VX Futures Enhancement** (1 hour)
   - [ ] Add full code from Section 3.2 to `extract_vix_futures_features()`
   - [ ] Test with actual VX1-VX2 data
   - [ ] Verify all new features generate correctly
   - [ ] Update config.py FUTURES_FEATURES['vix_futures']

2. **CL Futures Enhancement** (45 min)
   - [ ] Add full code from Section 3.3 to `extract_commodity_futures_features()`
   - [ ] Test contango/backwardation detection
   - [ ] Update config.py FUTURES_FEATURES['commodity_futures']

3. **DX Futures Enhancement** (45 min)
   - [ ] Add full code from Section 3.4 to `extract_dollar_futures_features()`
   - [ ] Test carry signal generation
   - [ ] Update config.py FUTURES_FEATURES['dollar_futures']

4. **Config.py Updates** (20 min)
   - [ ] Add all new VX features to vix_futures list
   - [ ] Add all new CL features to commodity_futures list
   - [ ] Add all new DX features to dollar_futures list
   - [ ] Create new anomaly group: 'futures_carry_signals'

**Verification Steps:**
```python
# Check new features exist
vx_features = [c for c in features.columns if 'vx_' in c]
print(f"VX features: {len(vx_features)}")  # Should be 25-30

cl_features = [c for c in features.columns if 'cl_' in c.lower()]
print(f"CL features: {len(cl_features)}")  # Should be 15-20

# Verify contango/backwardation detection
print(features[['vx_contango_strength', 'vx_backwardation_strength']].describe())
```

---

### Phase 4: Treasury Yield Curve (Priority: HIGH - NEW CAPABILITY)

**Week 2-3 - Add Complete Yield Curve Analysis**

1. **Create TreasuryYieldFeatureEngine Class** (2 hours)
   - [ ] Add entire class from Section 4.3 to feature_engine.py
   - [ ] Implement all 5 methods:
     - extract_term_spreads()
     - extract_curve_shape()
     - extract_rate_volatility()
     - extract_policy_expectations()
     - extract_credit_spreads()
   - [ ] Test each method individually

2. **Update Data Fetcher** (30 min)
   - [ ] Verify FRED fetcher can get Treasury yields (DGS2, DGS10, etc.)
   - [ ] Verify credit spread access (BAMLH0A0HYM2, BAMLC0A0CM)
   - [ ] Test incremental=True for all yield series

3. **Integrate into UnifiedFeatureEngine** (1 hour)
   - [ ] Add `_build_treasury_features()` method to UnifiedFeatureEngine
   - [ ] Add STAGE 7.5 to `build_complete_features()`
   - [ ] Add treasury_features to consolidation step
   - [ ] Update feature_breakdown dict

4. **Config.py Treasury Section** (30 min)
   - [ ] Add TREASURY_FEATURES dictionary with all 5 categories
   - [ ] Update ANOMALY_FEATURE_GROUPS with 3 new groups:
     - yield_curve_stress
     - credit_stress
     - rate_volatility_regime
   - [ ] Add to REGIME_CLASSIFICATION_FEATURE_GROUPS

5. **Testing & Validation** (1 hour)
   - [ ] Run full feature build with treasury features
   - [ ] Verify 10Y-2Y inversion detection during 2019, 2022-2023
   - [ ] Check credit spread elevation during March 2020
   - [ ] Verify HY spread percentiles align with known stress periods

**Verification Steps:**
```python
# Check yield curve features
treasury_cols = [c for c in features.columns if 'yield_' in c or 'hy_' in c or 'ig_' in c]
print(f"Treasury features: {len(treasury_cols)}")  # Should be 60-80

# Verify inversion detection
inversions = features[features['yield_10y2y_inverted'] == 1]
print(f"Inversion periods detected: {len(inversions)} days")
print(inversions.index[[0, -1]])  # Should show 2019-2020 and 2022-2023 periods

# Check credit stress alignment with COVID
covid_period = features.loc['2020-03-01':'2020-04-01']
print(f"HY stress during COVID: {covid_period['hy_stress_extreme'].mean():.1%}")  # Should be high
```

---

### Phase 5: Final Integration & Testing (Priority: HIGH)

**Week 3 - Complete System Validation**

1. **Run Full Feature Build** (30 min)
   - [ ] Execute: `python integrated_system_production.py`
   - [ ] Confirm feature count: ~420 - 58 removed + 80 treasury = ~442 features
   - [ ] Check memory usage is acceptable (<200MB)
   - [ ] Verify no errors or warnings

2. **Feature Quality Check** (30 min)
   - [ ] Run diagnostic analysis again
   - [ ] Verify removed features no longer appear
   - [ ] Check for new issues with treasury/futures features
   - [ ] Ensure missing data % is reasonable (<15% overall)

3. **Anomaly Detection Testing** (1 hour)
   - [ ] Verify all 15+ anomaly detectors still work
   - [ ] Test new detectors (yield_curve_stress, credit_stress, rate_volatility_regime)
   - [ ] Check detector coverage for all feature groups
   - [ ] Validate crisis period alignment

4. **Historical Backtesting** (1 hour)
   - [ ] Test on 2008 financial crisis (yield inversions should trigger)
   - [ ] Test on 2020 COVID (credit spreads should spike)
   - [ ] Test on 2022 inflation/rate hikes (curve should flatten)
   - [ ] Verify new features add predictive value

5. **Documentation** (30 min)
   - [ ] Update README with new feature counts
   - [ ] Document treasury yield curve capability
   - [ ] Document enhanced futures features
   - [ ] Note removed redundant features

**Final Verification Script:**
```python
# Complete system check
result = engine.build_complete_features(years=15)
features = result['features']

# Feature count check
print(f"Total features: {len(features.columns)}")
print(f"Target range: 440-450")

# Category breakdown
breakdown = result['feature_breakdown']
for cat, cols in breakdown.items():
    print(f"{cat}: {len(cols)} features")

# Check new capabilities
treasury_features = [c for c in features.columns if any(x in c for x in ['yield_', 'hy_', 'ig_', 'credit_', 'policy_'])]
print(f"\nTreasury/Credit features: {len(treasury_features)}")

enhanced_futures = [c for c in features.columns if any(x in c for x in ['vx_curve', 'cl_carry', 'dx_carry', 'contango', 'backwardation'])]
print(f"Enhanced futures features: {len(enhanced_futures)}")

# Check for removed duplicates
removed_features = ['VIXCLS_zscore_63d', 'vol_of_vol_10d', 'risk_premium', 'vx_spread', 'cl_spread_ma10']
for feat in removed_features:
    if feat in features.columns:
        print(f"❌ ERROR: {feat} should be removed but still exists!")
    else:
        print(f"✅ {feat} successfully removed")

# Missing data check
missing_pct = features.isna().sum().sum() / features.size * 100
print(f"\nMissing data: {missing_pct:.1f}% (target: <15%)")

# Crisis period validation
covid = features.loc['2020-03-01':'2020-03-30']
if 'hy_stress_extreme' in covid.columns:
    print(f"COVID credit stress: {covid['hy_stress_extreme'].mean():.0%} (should be high)")
if 'yield_10y2y_inverted' in features.columns:
    inversions_2022 = features.loc['2022-07-01':'2023-06-01', 'yield_10y2y_inverted'].sum()
    print(f"2022-23 inversions: {inversions_2022} days (should be ~200)")

print("\n✅ System validation complete!")
```

---

## PART 6: Expected Outcomes

### Feature Count Evolution

**Before Cleanup:**
- Total: 420 features
- Redundant: 58 features
- Missing capabilities: Treasury yields, enhanced futures

**After Cleanup:**
- Total: ~442 features
- Removed: 58 redundant
- Added: ~80 treasury/credit features
- Enhanced: ~30 additional futures features

### Feature Breakdown (Target)

```
Base Features:        81  (unchanged)
CBOE Features:        75  (cleaned from 80)
Futures Features:     60  (enhanced from 42)
Macro Features:       25  (unchanged)
Meta Features:        115 (cleaned from 124)
FRED Features:        75  (unchanged)
Treasury Features:    80  (NEW)
Calendar Features:    6   (enhanced from 3)
─────────────────────────
TOTAL:               517  (vs 420 before)
```

### Quality Improvements

1. **No More Duplicates**: All corr=1.000 features removed
2. **Consistent Naming**: All velocity features use _pct suffix
3. **Proper Sparse Handling**: GAMMA features acknowledge sparsity
4. **Fixed Bugs**: OPEX detection works correctly
5. **New Capabilities**: 
   - Full yield curve analysis
   - Recession prediction signals
   - Credit stress indicators
   - Enhanced futures carry signals

### Predictive Power Gains

**Expected improvements in specific scenarios:**

1. **Recession Prediction**: Yield curve inversions provide 6-12 month lead time
2. **Credit Stress**: HY spreads spike before VIX in many crises
3. **Rate Volatility**: Treasury vol predicts equity vol regime changes
4. **Futures Carry**: Enhanced signals improve term structure anomaly detection

---

## PART 7: Maintenance & Future Enhancements

### Ongoing Monitoring

**Weekly:**
- [ ] Check feature generation success rate (should be >95%)
- [ ] Monitor missing data percentages per category
- [ ] Verify no new duplicates introduced

**Monthly:**
- [ ] Re-run diagnostics to catch feature drift
- [ ] Validate treasury data is updating correctly
- [ ] Check for new FRED series that should be added

**Quarterly:**
- [ ] Review feature importance from model
- [ ] Consider adding new interactions
- [ ] Evaluate if any features can be pruned

### Future Enhancements (Backlog)

**Priority 2 - Enhanced Interactions:**
1. VIX-Yield curve cross-asset features
2. Futures-Credit spread interactions
3. Multi-timeframe regime detection

**Priority 3 - Alternative Data:**
1. Equity market breadth (advance/decline)
2. Sector rotation signals
3. International yield curves (bunds, JGBs)

**Priority 4 - Machine Learning Features:**
1. PCA components of yield curve
2. Clustering-based regime labels
3. Auto-encoded meta-features

---

## PART 8: Quick Reference

### File Modification Summary

| File | Lines to Modify | Estimated Time |
|------|----------------|----------------|
| `feature_engine.py` | ~200 lines edited | 3 hours |
| `config.py` | ~100 lines edited | 1 hour |
| `data_fetcher_v7.py` | Verify FRED access | 15 min |
| `integrated_system_production.py` | No changes needed | 0 min |

### Critical Code Sections

**feature_engine.py:**
- Line ~300-315: MetaFeatureEngine.extract_cross_asset_relationships()
- Line ~400-450: MetaFeatureEngine.extract_rate_of_change_features()
- Line ~600-650: FuturesFeatureEngine classes
- Line ~750-800: UnifiedFeatureEngine._cboe_interactions()
- Line ~900-950: UnifiedFeatureEngine._calendar_features()
- Line ~1200+: NEW TreasuryYieldFeatureEngine class

**config.py:**
- Line ~100-150: VIX_BASE_FEATURES
- Line ~200-250: CBOE_BASE_FEATURES
- Line ~300-350: FUTURES_FEATURES
- Line ~400+: NEW TREASURY_FEATURES section

### Testing Commands

```bash
# Quick test (2yr window)
python -c "from feature_engine import UnifiedFeatureEngine; from data_fetcher_v7 import UnifiedDataFetcher; engine = UnifiedFeatureEngine(UnifiedDataFetcher()); result = engine.build_complete_features(years=2); print(f'Features: {len(result[\"features\"].columns)}')"

# Full test (15yr window)
python integrated_system_production.py

# Feature diagnostic
python -c "from feature_diagnostic import FeatureDiagnostic; diag = FeatureDiagnostic(); diag.run_full_diagnostic()"
```

### Success Criteria

✅ **Feature Removal:**
- [ ] 58 redundant features removed
- [ ] No features with corr > 0.99 remaining
- [ ] Config.py has no references to removed features

✅ **Bug Fixes:**
- [ ] is_opex_week has ~20% coverage (not 0%)
- [ ] GOLDSILVER_zscore_63d is not 100% missing
- [ ] GAMMA features handle sparsity gracefully

✅ **Enhancements:**
- [ ] 25+ new VX futures features
- [ ] 15+ new CL futures features
- [ ] 10+ new DX futures features
- [ ] 80+ new Treasury/credit features

✅ **System Health:**
- [ ] Total features: 440-520 range
- [ ] Memory usage: <200MB growth
- [ ] Missing data: <15% overall
- [ ] No errors during feature generation

✅ **Validation:**
- [ ] 2019-2020 yield inversions detected
- [ ] 2020 credit stress peaks detected
- [ ] 2022 rate volatility regime captured
- [ ] Futures carry signals validated

---

## Summary for Next Claude Instance

**Context:** This document provides a complete plan to:
1. Remove 58 redundant features (15% reduction)
2. Fix 3 broken features (OPEX, GOLDSILVER, GAMMA)
3. Add 80 treasury yield curve features (NEW capability)
4. Enhance futures features by 30 features (better term structure)

**Start Here:**
1. Phase 1: Remove duplicates (1-2 hours)
2. Phase 2: Fix bugs (1 hour)
3. Phase 4: Add treasury features (3-4 hours) ← HIGHEST VALUE
4. Phase 3: Enhance futures (2 hours)
5. Phase 5: Test everything (2 hours)

**Critical Success Factors:**
- Don't break existing anomaly detectors
- Treasury features are the biggest win (recession prediction)
- Test on historical crises (2008, 2020, 2022)
- Verify memory usage stays reasonable

**Key Files:**
- `feature_engine.py` - Most changes here
- `config.py` - Feature definitions
- `data_fetcher_v7.py` - Already supports FRED (verify Treasury access)

**Total Estimated Time:** 10-12 hours of focused work