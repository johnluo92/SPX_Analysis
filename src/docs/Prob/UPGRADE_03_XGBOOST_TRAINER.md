# UPGRADE GUIDE: xgboost_trainer_v2.py
## Complete Rewrite: Binary Classifier → Probabilistic Multi-Output Forecaster

---

## ⚠️ CRITICAL: THIS IS A COMPLETE REWRITE

**DO NOT** try to modify existing code. This file must be rebuilt from scratch.

**Reason:** Current file trains binary classifier (`XGBClassifier`) with `scale_pos_weight`. New file trains multi-output regression system with custom loss functions. These are fundamentally different architectures with zero code reuse.

**Action Plan:**
1. Save current file as `xgboost_trainer_v2.backup.py`
2. Create new file from template below
3. Test with small dataset first (1 year)
4. Validate quantile ordering and regime probabilities
5. Run full 15-year training

---

## SYSTEM CONTEXT (READ THIS FIRST)

### The Transformation
**Old System (Binary Classifier):**
```python
Input: Features (232) + Binary target (0/1)
Model: XGBClassifier
Output: P(VIX expansion > 5%) = 0.65
Problem: 39% precision, arbitrary threshold, no uncertainty quantification
```

**New System (Probabilistic Forecaster):**
```python
Input: Features (232) + Cohort context + Continuous target (VIX % change)
Model: 8 XGBoost regressors per cohort
  ├── Point estimate (mean)
  ├── 5 Quantiles (10th, 25th, 50th, 75th, 90th)
  ├── 4 Regime probabilities (Low/Normal/Elevated/Crisis)
  └── 1 Confidence score (forecast quality)
Output: Full distribution object
  - Point: +8.5% VIX change expected
  - q10: -2.3% | q25: +3.1% | q50: +7.9% | q75: +14.2% | q90: +23.8%
  - Regimes: Low(2%) | Normal(38%) | Elevated(52%) | Crisis(8%)
  - Confidence: 0.87 (high quality features)
```

### Why This Works
- **Quantiles** capture distribution shape (symmetric vs skewed)
- **Regimes** classify future state (for risk management)
- **Confidence** adjusts for data quality (missing CBOE features? Lower confidence)
- **Cohorts** learn context-specific dynamics (OpEx compression vs FOMC spikes)

---

## FILE ROLE: xgboost_trainer_v2.py

**Purpose:** Train probabilistic VIX forecasting models and calibrate probabilities.

**Input:** DataFrame from `feature_engine.py` with:
- 232 feature columns
- `calendar_cohort` column (str)
- `cohort_weight` column (float)
- `feature_quality` column (float)
- `vix` column (for target creation)

**Output:** Dictionary of trained models:
```python
models = {
    'monthly_opex_minus_5': {
        'point': XGBRegressor (trained),
        'quantiles': {0.10: XGBRegressor, 0.25: XGBRegressor, ...},
        'regime': XGBClassifier (4-class),
        'confidence': XGBRegressor
    },
    'fomc_week': { ... },
    'mid_cycle': { ... },
    # ... one model set per cohort
}
```

**Saved Artifacts:**
- `models/probabilistic_forecaster_{cohort}.pkl` (one file per cohort)
- `models/calibration_curves.png` (visual check)
- `models/quantile_diagnostics.json` (coverage stats)

---

## REQUIRED CONTEXT FROM OTHER FILES

### From config.py
```python
from config import (
    TARGET_CONFIG,           # Quantile levels, regime boundaries
    XGBOOST_CONFIG,          # Hyperparameters
    CALENDAR_COHORTS,        # Cohort definitions
    PREDICTION_DB_CONFIG     # Not used here, but good to know schema
)
```

### From feature_engine.py (what you'll receive)
```python
# DataFrame structure from feature_engine.build_features()
df = pd.DataFrame({
    # 232 features
    'vix': [...],
    'spx': [...],
    'yield_10y2y': [...],
    # ... 229 more features ...
    
    # Metadata columns (DO NOT use as features)
    'calendar_cohort': ['mid_cycle', 'opex_minus_5', ...],
    'cohort_weight': [1.0, 1.2, 1.5, ...],
    'feature_quality': [0.97, 0.95, 0.82, ...]
})
```

### From temporal_validator.py
```python
from temporal_validator import TemporalValidator
# Used to validate feature alignment (already done in feature_engine, but double-check)
```

---

## COMPLETE NEW FILE STRUCTURE

### PART 1: Imports and Class Definition

```python
"""
Probabilistic VIX Forecasting System
Trains multi-output XGBoost models for distribution forecasting
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET_CONFIG,
    XGBOOST_CONFIG,
    CALENDAR_COHORTS
)

logger = logging.getLogger(__name__)


class ProbabilisticVIXForecaster:
    """
    Multi-output forecaster producing full VIX distribution.
    
    Trains 8 models per calendar cohort:
      - 1 point estimate (mean VIX % change)
      - 5 quantiles (10th, 25th, 50th, 75th, 90th percentiles)
      - 1 regime classifier (4 classes: Low/Normal/Elevated/Crisis)
      - 1 confidence scorer (forecast quality)
    
    Example:
        >>> forecaster = ProbabilisticVIXForecaster()
        >>> forecaster.train(df, horizon=5)
        >>> distribution = forecaster.predict(X, cohort='mid_cycle')
        >>> print(distribution['quantiles']['q50'])  # Median forecast
    """
    
    def __init__(self):
        self.horizon = TARGET_CONFIG['horizon_days']
        self.quantiles = TARGET_CONFIG['quantiles']['levels']
        self.regime_boundaries = TARGET_CONFIG['regimes']['boundaries']
        self.regime_labels = TARGET_CONFIG['regimes']['labels']
        
        self.models = {}  # {cohort: {model_type: model}}
        self.calibrators = {}  # {cohort: {model_type: IsotonicRegression}}
        self.feature_names = None
        
        logger.info("🎯 Probabilistic VIX Forecaster initialized")
        logger.info(f"   Horizon: {self.horizon} days")
        logger.info(f"   Quantiles: {self.quantiles}")
        logger.info(f"   Regimes: {self.regime_labels}")
```

---

### PART 2: Target Creation

```python
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all target variables from raw VIX data.
        
        Args:
            df: DataFrame with 'vix' column and index as dates
            
        Returns:
            DataFrame with original features + target columns:
                - target_point: VIX % change
                - target_q10, target_q25, ..., target_q90: Quantile labels
                - target_regime: Regime class (0-3)
                - target_confidence: Confidence label (feature quality × regime stability)
        """
        df = df.copy()
        
        # 1. Point Estimate: Future VIX % change
        future_vix = df['vix'].shift(-self.horizon)
        df['target_point'] = (future_vix / df['vix'] - 1) * 100
        
        # Clip extremes (VIX can't go to infinity or negative)
        point_min, point_max = TARGET_CONFIG['point_estimate']['range']
        df['target_point'] = df['target_point'].clip(point_min, point_max)
        
        logger.info(f"   Point target range: {df['target_point'].min():.1f}% to {df['target_point'].max():.1f}%")
        
        # 2. Quantiles: Use rolling historical quantiles as pseudo-targets
        # (In production, you'd use ensemble forecasts or simulation, but this works)
        window = 252  # 1 year rolling window
        
        for q in self.quantiles:
            col_name = f'target_q{int(q*100)}'
            # Rolling quantile of forward returns
            df[col_name] = df['target_point'].shift(1).rolling(window).quantile(q)
        
        # Forward-fill NaNs from rolling window startup
        for q in self.quantiles:
            col_name = f'target_q{int(q*100)}'
            df[col_name] = df[col_name].ffill()
        
        logger.info(f"   Quantile targets created (using {window}-day rolling window)")
        
        # 3. Regime: Classify future VIX level
        regime_bins = [0] + self.regime_boundaries + [np.inf]
        df['target_regime'] = pd.cut(
            future_vix,
            bins=regime_bins,
            labels=list(range(len(self.regime_labels))),
            include_lowest=True
        ).astype(int)
        
        regime_counts = df['target_regime'].value_counts().sort_index()
        logger.info("   Regime distribution:")
        for regime_id, label in enumerate(self.regime_labels):
            count = regime_counts.get(regime_id, 0)
            pct = count / len(df) * 100
            logger.info(f"      {label:10s} (class {regime_id}): {count:4d} ({pct:5.1f}%)")
        
        # 4. Confidence: Use existing feature_quality as base
        # Add regime stability component (low volatility = high stability)
        regime_volatility = df['vix'].rolling(21).std()
        regime_stability = 1 / (1 + regime_volatility / df['vix'].rolling(21).mean())
        
        # Combine: 50% feature quality + 50% regime stability
        df['target_confidence'] = (
            0.5 * df['feature_quality'] +
            0.5 * regime_stability
        ).clip(0, 1)
        
        logger.info(f"   Confidence labels: mean={df['target_confidence'].mean():.2f}, "
                   f"std={df['target_confidence'].std():.2f}")
        
        return df
```

---

### PART 3: Cohort-Wise Training

```python
    def train(self, df: pd.DataFrame, save_dir: str = 'models') -> Dict:
        """
        Train separate model sets for each calendar cohort.
        
        Args:
            df: DataFrame from feature_engine with cohort column
            save_dir: Directory to save trained models
            
        Returns:
            Dict with training metrics per cohort
        """
        logger.info("=" * 80)
        logger.info("PROBABILISTIC VIX FORECASTER - TRAINING")
        logger.info("=" * 80)
        
        # Create targets
        df = self._create_targets(df)
        
        # Remove rows where targets are NaN (edge effects from shifts)
        df_clean = df.dropna(subset=['target_point', 'target_regime', 'target_confidence'])
        logger.info(f"Training samples: {len(df_clean)} (dropped {len(df) - len(df_clean)} edge rows)")
        
        # Extract feature names (exclude metadata and targets)
        exclude_cols = ['calendar_cohort', 'cohort_weight', 'feature_quality']
        exclude_cols += [col for col in df_clean.columns if col.startswith('target_')]
        self.feature_names = [col for col in df_clean.columns if col not in exclude_cols]
        
        logger.info(f"Features used: {len(self.feature_names)}")
        
        # Train per cohort
        cohort_metrics = {}
        for cohort in df_clean['calendar_cohort'].unique():
            logger.info(f"\n{'─'*80}")
            logger.info(f"TRAINING COHORT: {cohort}")
            logger.info(f"{'─'*80}")
            
            cohort_df = df_clean[df_clean['calendar_cohort'] == cohort].copy()
            logger.info(f"Cohort samples: {len(cohort_df)}")
            
            if len(cohort_df) < 100:
                logger.warning(f"⚠️  Too few samples ({len(cohort_df)}) for {cohort}, skipping")
                continue
            
            # Train all model types for this cohort
            metrics = self._train_cohort_models(cohort, cohort_df)
            cohort_metrics[cohort] = metrics
            
            # Save models
            self._save_cohort_models(cohort, save_dir)
        
        # Generate diagnostics
        self._generate_diagnostics(cohort_metrics, save_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return cohort_metrics
    
    
    def _train_cohort_models(self, cohort: str, df: pd.DataFrame) -> Dict:
        """Train all 8 models for a single cohort."""
        X = df[self.feature_names]
        
        # Initialize model dictionary for this cohort
        self.models[cohort] = {}
        self.calibrators[cohort] = {}
        
        metrics = {}
        
        # 1. Point Estimate
        logger.info("\n[1/4] Training point estimate model...")
        y_point = df['target_point']
        model_point, metric_point = self._train_regressor(
            X, y_point, 
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        self.models[cohort]['point'] = model_point
        metrics['point'] = metric_point
        logger.info(f"   ✅ Point RMSE: {metric_point['rmse']:.2f}%")
        
        # 2. Quantiles (5 models)
        logger.info("\n[2/4] Training quantile models...")
        self.models[cohort]['quantiles'] = {}
        metrics['quantiles'] = {}
        
        for q in self.quantiles:
            q_label = f'q{int(q*100)}'
            y_quantile = df[f'target_{q_label}']
            
            model_q, metric_q = self._train_regressor(
                X, y_quantile,
                objective='reg:quantileerror',
                quantile_alpha=q,
                eval_metric='mae'
            )
            self.models[cohort]['quantiles'][q] = model_q
            metrics['quantiles'][q_label] = metric_q
            logger.info(f"   ✅ {q_label:3s} MAE: {metric_q['mae']:.2f}%")
        
        # 3. Regime Classifier
        logger.info("\n[3/4] Training regime classifier...")
        y_regime = df['target_regime']
        model_regime, metric_regime = self._train_classifier(
            X, y_regime,
            num_classes=len(self.regime_labels)
        )
        self.models[cohort]['regime'] = model_regime
        self.calibrators[cohort]['regime'] = self._calibrate_probabilities(
            model_regime, X, y_regime
        )
        metrics['regime'] = metric_regime
        logger.info(f"   ✅ Regime Accuracy: {metric_regime['accuracy']:.3f}")
        logger.info(f"   ✅ Log Loss: {metric_regime['log_loss']:.3f}")
        
        # 4. Confidence Scorer
        logger.info("\n[4/4] Training confidence model...")
        y_confidence = df['target_confidence']
        model_conf, metric_conf = self._train_regressor(
            X, y_confidence,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        self.models[cohort]['confidence'] = model_conf
        metrics['confidence'] = metric_conf
        logger.info(f"   ✅ Confidence RMSE: {metric_conf['rmse']:.3f}")
        
        return metrics
```

---

### PART 4: Individual Model Training

```python
    def _train_regressor(self, X, y, objective, eval_metric, quantile_alpha=None):
        """Train single XGBoost regressor with CV."""
        params = XGBOOST_CONFIG['shared_params'].copy()
        params['objective'] = objective
        params['eval_metric'] = eval_metric
        
        if quantile_alpha:
            params['quantile_alpha'] = quantile_alpha
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            
            if eval_metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            elif eval_metric == 'mae':
                score = mean_absolute_error(y_val, y_pred)
            
            cv_scores.append(score)
        
        # Train final model on full data
        final_model = XGBRegressor(**params)
        final_model.fit(X, y, verbose=False)
        
        metrics = {
            eval_metric: np.mean(cv_scores),
            f'{eval_metric}_std': np.std(cv_scores)
        }
        
        return final_model, metrics
    
    
    def _train_classifier(self, X, y, num_classes):
        """Train XGBoost classifier for regime prediction."""
        params = XGBOOST_CONFIG['shared_params'].copy()
        params['objective'] = 'multi:softprob'
        params['num_class'] = num_classes
        params['eval_metric'] = 'mlogloss'
        
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
        
        cv_accuracy = []
        cv_logloss = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            
            accuracy = (y_pred == y_val).mean()
            logloss = log_loss(y_val, y_proba)
            
            cv_accuracy.append(accuracy)
            cv_logloss.append(logloss)
        
        # Train final model
        final_model = XGBClassifier(**params)
        final_model.fit(X, y, verbose=False)
        
        metrics = {
            'accuracy': np.mean(cv_accuracy),
            'accuracy_std': np.std(cv_accuracy),
            'log_loss': np.mean(cv_logloss),
            'log_loss_std': np.std(cv_logloss)
        }
        
        return final_model, metrics
    
    
    def _calibrate_probabilities(self, model, X, y):
        """Calibrate classifier probabilities using isotonic regression."""
        y_proba = model.predict_proba(X)
        
        # One isotonic calibrator per class
        calibrators = []
        for class_idx in range(y_proba.shape[1]):
            y_binary = (y == class_idx).astype(int)
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_proba[:, class_idx], y_binary)
            calibrators.append(calibrator)
        
        return calibrators
```

---

### PART 5: Prediction Interface

```python
    def predict(self, X: pd.DataFrame, cohort: str) -> Dict:
        """
        Generate probabilistic forecast for new data.
        
        Args:
            X: Feature DataFrame (single row or multiple rows)
            cohort: Which cohort model to use
            
        Returns:
            Dict with keys:
                - point_estimate: float
                - quantiles: dict {q10: float, q25: float, ...}
                - regime_probs: dict {low: float, normal: float, ...}
                - confidence_score: float
        """
        if cohort not in self.models:
            raise ValueError(f"Cohort {cohort} not trained. Available: {list(self.models.keys())}")
        
        X_features = X[self.feature_names]
        
        # Get predictions from all models
        point = self.models[cohort]['point'].predict(X_features)[0]
        
        quantiles = {}
        for q in self.quantiles:
            q_label = f'q{int(q*100)}'
            quantiles[q_label] = self.models[cohort]['quantiles'][q].predict(X_features)[0]
        
        # Enforce quantile monotonicity
        quantiles = self._enforce_quantile_order(quantiles)
        
        regime_probs_raw = self.models[cohort]['regime'].predict_proba(X_features)[0]
        
        # Calibrate regime probabilities
        if cohort in self.calibrators and 'regime' in self.calibrators[cohort]:
            regime_probs = []
            for class_idx, calibrator in enumerate(self.calibrators[cohort]['regime']):
                prob_calibrated = calibrator.predict([regime_probs_raw[class_idx]])[0]
                regime_probs.append(prob_calibrated)
            
            # Renormalize to sum to 1
            regime_probs = np.array(regime_probs)
            regime_probs = regime_probs / regime_probs.sum()
        else:
            regime_probs = regime_probs_raw
        
        confidence = self.models[cohort]['confidence'].predict(X_features)[0]
        confidence = np.clip(confidence, 0, 1)  # Ensure [0, 1]
        
        return {
            'point_estimate': float(point),
            'quantiles': {k: float(v) for k, v in quantiles.items()},
            'regime_probabilities': {
                self.regime_labels[i].lower(): float(regime_probs[i])
                for i in range(len(self.regime_labels))
            },
            'confidence_score': float(confidence),
            'cohort': cohort
        }
    
    
    def _enforce_quantile_order(self, quantiles: Dict[str, float]) -> Dict[str, float]:
        """Ensure q10 <= q25 <= q50 <= q75 <= q90."""
        sorted_q = sorted(quantiles.items(), key=lambda x: int(x[0][1:]))
        
        # Forward pass: ensure increasing
        for i in range(1, len(sorted_q)):
            prev_val = sorted_q[i-1][1]
            curr_val = sorted_q[i][1]
            if curr_val < prev_val:
                sorted_q[i] = (sorted_q[i][0], prev_val)
        
        return dict(sorted_q)
```

---

### PART 6: Model Persistence

```python
    def _save_cohort_models(self, cohort: str, save_dir: str):
        """Save models for one cohort to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        cohort_file = save_path / f'probabilistic_forecaster_{cohort}.pkl'
        
        with open(cohort_file, 'wb') as f:
            pickle.dump({
                'models': self.models[cohort],
                'calibrators': self.calibrators.get(cohort, {}),
                'feature_names': self.feature_names,
                'config': {
                    'horizon': self.horizon,
                    'quantiles': self.quantiles,
                    'regime_boundaries': self.regime_boundaries,
                    'regime_labels': self.regime_labels
                }
            }, f)
        
        logger.info(f"💾 Saved: {cohort_file}")
    
    
    def load(self, cohort: str, load_dir: str = 'models'):
        """Load trained models for a specific cohort."""
        load_path = Path(load_dir) / f'probabilistic_forecaster_{cohort}.pkl'
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.models[cohort] = data['models']
        self.calibrators[cohort] = data['calibrators']
        self.feature_names = data['feature_names']
        
        # Restore config
        config = data['config']
        self.horizon = config['horizon']
        self.quantiles = config['quantiles']
        self.regime_boundaries = config['regime_boundaries']
        self.regime_labels = config['regime_labels']
        
        logger.info(f"✅ Loaded cohort: {cohort}")
```

---

### PART 7: Diagnostics

```python
    def _generate_diagnostics(self, cohort_metrics: Dict, save_dir: str):
        """Generate diagnostic plots and JSON summaries."""
        save_path = Path(save_dir)
        
        # 1. Export metrics as JSON
        metrics_file = save_path / 'probabilistic_model_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(cohort_metrics, f, indent=2)
        logger.info(f"📊 Metrics saved: {metrics_file}")
        
        # 2. Plot regime classification performance
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        cohorts = list(cohort_metrics.keys())
        accuracies = [cohort_metrics[c]['regime']['accuracy'] for c in cohorts]
        log_losses = [cohort_metrics[c]['regime']['log_loss'] for c in cohorts]
        
        axes[0].bar(range(len(cohorts)), accuracies)
        axes[0].set_xticks(range(len(cohorts)))
        axes[0].set_xticklabels(cohorts, rotation=45, ha='right')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Regime Classification Accuracy by Cohort')
        axes[0].axhline(0.25, color='r', linestyle='--', label='Random (4 classes)')
        axes[0].legend()
        
        axes[1].bar(range(len(cohorts)), log_losses)
        axes[1].set_xticks(range(len(cohorts)))
        axes[1].set_xticklabels(cohorts, rotation=45, ha='right')
        axes[1].set_ylabel('Log Loss')
        axes[1].set_title('Regime Classification Log Loss by Cohort')
        axes[1].legend()
        
        plt.tight_layout()
        plot_file = save_path / 'regime_performance.png'
        plt.savefig(plot_file, dpi=150)
        logger.info(f"📈 Plot saved: {plot_file}")
        
        # 3. Summary table
        logger.info("\n" + "="*80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"{'Cohort':<30} | {'Point RMSE':>10} | {'Regime Acc':>10} | {'Conf RMSE':>10}")
        logger.info("-"*80)
        
        for cohort in sorted(cohorts):
            m = cohort_metrics[cohort]
            logger.info(
                f"{cohort:<30} | "
                f"{m['point']['rmse']:>9.2f}% | "
                f"{m['regime']['accuracy']:>9.3f} | "
                f"{m['confidence']['rmse']:>9.3f}"
            )
        
        logger.info("="*80)
```

---

## TESTING THE NEW FILE

### Basic Functionality Test
```python
def test_probabilistic_forecaster():
    """Test basic training and prediction."""
    from feature_engine import FeatureEngineV5
    
    # Build features
    engine = FeatureEngineV5()
    df = engine.build_features(window='1y')  # Small dataset for testing
    
    # Train
    forecaster = ProbabilisticVIXForecaster()
    metrics = forecaster.train(df, save_dir='models_test')
    
    # Check all cohorts trained
    assert len(forecaster.models) > 0
    print(f"✅ Trained {len(forecaster.models)} cohorts")
    
    # Test prediction
    test_cohort = list(forecaster.models.keys())[0]
    X_test = df[forecaster.feature_names].iloc[-1:]
    distribution = forecaster.predict(X_test, cohort=test_cohort)
    
    # Validate output structure
    assert 'point_estimate' in distribution
    assert 'quantiles' in distribution
    assert len(distribution['quantiles']) == 5
    assert 'regime_probabilities' in distribution
    assert sum(distribution['regime_probabilities'].values()) == pytest.approx(1.0, abs=0.01)
    assert 0 <= distribution['confidence_score'] <= 1
    
    # Check quantile ordering
    q_values = [distribution['quantiles'][f'q{q}'] for q in [10, 25, 50, 75, 90]]
    assert q_values == sorted(q_values), "Quantiles not monotonic!"
    
    print("✅ All tests passed")

test_probabilistic_forecaster()
```

---

## INTEGRATION WITH OTHER FILES

### Called By: integrated_system_production.py
```python
from xgboost_trainer_v2 import ProbabilisticVIXForecaster

# Training
forecaster = ProbabilisticVIXForecaster()
forecaster.train(df_features)

# Prediction
distribution = forecaster.predict(today_features, cohort='mid_cycle')
```

### Uses From: config.py
- TARGET_CONFIG: Quantiles, regimes, horizon
- XGBOOST_CONFIG: Hyperparameters

### Provides To: integrated_system_production.py
- Distribution object for storage in prediction DB

---

## COMMON PITFALLS

### 1. Quantile Non-Monotonicity
```python
# WRONG: Quantiles can cross
q50 = 5.2%
q75 = 4.8%  # Lower than median!

# FIX: Use _enforce_quantile_order()
```

### 2. Regime Probabilities Don't Sum to 1
```python
# WRONG: After calibration, [0.3, 0.5, 0.1, 0.05] = 0.95
# FIX: Renormalize after calibration
probs = probs / probs.sum()
```

### 3. Too Few Samples per Cohort
```python
# WRONG: Train on cohort with 20 samples → overfitting
# FIX: Skip cohorts with <100 samples
if len(cohort_df) < 100:
    logger.warning(f"Skipping {cohort}")
    continue
```

---

## SUMMARY

**New File:** Complete rewrite (~600 lines)

**Key Classes:**
- ProbabilisticVIXForecaster (main class)

**Key Methods:**
- train() - Cohort-wise training loop
- predict() - Generate distribution
- _create_targets() - Target engineering
- _train_regressor() - Individual model training
- _train_classifier() - Regime model
- _calibrate_probabilities() - Isotonic calibration

**Output:**
- 8 models per cohort (point + 5 quantiles + regime + confidence)
- Saved as pickle files: `probabilistic_forecaster_{cohort}.pkl`
- Diagnostics: metrics JSON + performance plots

**Next Steps:**
1. Test with 1-year data first
2. Validate quantile coverage and regime accuracy
3. Run full 15-year training
4. Update integrated_system_production.py to use new forecaster
