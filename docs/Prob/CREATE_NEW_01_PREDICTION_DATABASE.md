# CREATE NEW FILE: prediction_database.py
## Prediction Storage & Backtesting Infrastructure

---

## SYSTEM CONTEXT

### Why This File
**Problem:** Current system has no memory. Each prediction is ephemeral - generated, displayed, forgotten. Can't evaluate model performance over time.

**Solution:** SQLite database that stores every prediction with:
- Full distribution (point + quantiles + regimes)
- Feature provenance (which features used, their values)
- Actual outcomes (filled post-hoc)
- Metadata (cohort, quality, confidence)

**Use Cases:**
1. **Backtesting**: "How accurate were quantiles over 2024?"
2. **Calibration**: "Do 90% of actuals fall below q90?"
3. **Confidence Validation**: "Does confidence correlate with error?"
4. **Cohort Analysis**: "Does FOMC model outperform mid_cycle?"

---

## FILE PURPOSE

**New file location:** `src/prediction_database.py`

**Core Responsibilities:**
- Create/manage SQLite database schema
- Store predictions with full metadata
- Backfill actual outcomes (async process)
- Query predictions for evaluation
- Compute calibration metrics

---

## REQUIRED CONTEXT

### From config.py
```python
from config import PREDICTION_DB_CONFIG

# Schema definition, table name, file path
```

### From integrated_system_production.py (caller)
```python
from prediction_database import PredictionDatabase

db = PredictionDatabase()
db.store_prediction(record)  # Store forecast
db.backfill_actuals()  # Fill in realized outcomes
```

---

## COMPLETE FILE IMPLEMENTATION

```python
"""
Prediction Database for Probabilistic Forecasting System

Stores all forecasts with full distribution + metadata for backtesting.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from config import PREDICTION_DB_CONFIG

logger = logging.getLogger(__name__)


class PredictionDatabase:
    """
    SQLite database for storing and retrieving probabilistic forecasts.
    
    Schema:
        - predictions: Main table with forecasts
        - actuals: Realized VIX values (joined by forecast_date)
    
    Example:
        >>> db = PredictionDatabase()
        >>> db.store_prediction(forecast_record)
        >>> db.backfill_actuals(vix_data)
        >>> metrics = db.compute_quantile_coverage()
    """
    
    def __init__(self, db_path=None):
        """
        Initialize prediction database.
        
        Args:
            db_path: Path to SQLite database (defaults to config)
        """
        if db_path is None:
            db_path = PREDICTION_DB_CONFIG['db_path']
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Dict-like access
        
        self._create_schema()
        logger.info(f"📂 Prediction database: {self.db_path}")
    
    
    def _create_schema(self):
        """Create database tables if they don't exist."""
        schema = PREDICTION_DB_CONFIG['schema']
        
        # Build CREATE TABLE statement from schema
        columns = []
        for col_name, col_type in schema.items():
            columns.append(f"{col_name} {col_type}")
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_DB_CONFIG['table_name']} (
            {', '.join(columns)}
        )
        """
        
        self.conn.execute(create_sql)
        
        # Create indexes for fast queries
        for index_sql in PREDICTION_DB_CONFIG['indexes']:
            try:
                self.conn.execute(index_sql)
            except sqlite3.OperationalError:
                pass  # Index already exists
        
        self.conn.commit()
        logger.info("✅ Database schema initialized")
    
    
    def store_prediction(self, record: Dict):
        """
        Store a single prediction.
        
        Args:
            record: Dict with keys matching schema
                Required: prediction_id, timestamp, forecast_date, horizon,
                          calendar_cohort, point_estimate, q10-q90,
                          prob_low/normal/elevated/crisis, confidence_score
                Optional: features_used, model_version
        
        Example:
            >>> record = {
            >>>     'prediction_id': 'abc123',
            >>>     'timestamp': pd.Timestamp.now(),
            >>>     'point_estimate': 8.5,
            >>>     'q50': 7.9,
            >>>     # ... other fields
            >>> }
            >>> db.store_prediction(record)
        """
        # Convert timestamps to ISO strings
        record = record.copy()
        for key in ['timestamp', 'forecast_date', 'created_at']:
            if key in record and isinstance(record[key], pd.Timestamp):
                record[key] = record[key].isoformat()
        
        # Ensure created_at is set
        if 'created_at' not in record:
            record['created_at'] = datetime.now().isoformat()
        
        # Build INSERT statement
        columns = list(record.keys())
        placeholders = ', '.join(['?' for _ in columns])
        
        insert_sql = f"""
        INSERT INTO {PREDICTION_DB_CONFIG['table_name']} 
        ({', '.join(columns)})
        VALUES ({placeholders})
        """
        
        values = [record[col] for col in columns]
        
        try:
            self.conn.execute(insert_sql, values)
            self.conn.commit()
            logger.debug(f"💾 Stored prediction: {record['prediction_id']}")
        except sqlite3.IntegrityError as e:
            logger.warning(f"⚠️  Duplicate prediction: {record['prediction_id']}")
    
    
    def backfill_actuals(self, vix_data: pd.Series = None):
        """
        Fill in actual outcomes for past predictions.
        
        Args:
            vix_data: Series of VIX values indexed by date
                      If None, fetches from Yahoo Finance
        
        Process:
            1. Find predictions where actual_vix_change IS NULL
            2. Look up VIX value at forecast_date
            3. Compute realized % change
            4. Update actuals
        """
        logger.info("🔄 Backfilling actual outcomes...")
        
        # Fetch VIX data if not provided
        if vix_data is None:
            import yfinance as yf
            vix_data = yf.download('^VIX', progress=False)['Close']
        
        # Get predictions needing actuals
        query = f"""
        SELECT prediction_id, forecast_date, current_vix, point_estimate,
               q10, q25, q50, q75, q90
        FROM {PREDICTION_DB_CONFIG['table_name']}
        WHERE actual_vix_change IS NULL
          AND forecast_date <= date('now')
        """
        
        cursor = self.conn.execute(query)
        predictions = cursor.fetchall()
        
        logger.info(f"   Found {len(predictions)} predictions to backfill")
        
        updated = 0
        for pred in predictions:
            forecast_date = pd.Timestamp(pred['forecast_date'])
            
            # Get actual VIX at forecast date
            if forecast_date not in vix_data.index:
                # Try next business day (in case of holiday)
                next_dates = vix_data.index[vix_data.index > forecast_date]
                if len(next_dates) == 0:
                    continue  # Future date
                forecast_date = next_dates[0]
            
            actual_vix = vix_data.loc[forecast_date]
            current_vix = pred['current_vix']
            
            # Compute actual % change
            actual_change = (actual_vix - current_vix) / current_vix * 100
            
            # Determine which regime
            from config import TARGET_CONFIG
            boundaries = TARGET_CONFIG['regimes']['boundaries']
            labels = TARGET_CONFIG['regimes']['labels']
            
            if actual_vix < boundaries[0]:
                actual_regime = labels[0]  # Low
            elif actual_vix < boundaries[1]:
                actual_regime = labels[1]  # Normal
            elif actual_vix < boundaries[2]:
                actual_regime = labels[2]  # Elevated
            else:
                actual_regime = labels[3]  # Crisis
            
            # Compute point error
            point_error = abs(actual_change - pred['point_estimate'])
            
            # Check quantile coverage
            coverage = {
                'q10': int(actual_change <= pred['q10']),
                'q25': int(actual_change <= pred['q25']),
                'q50': int(actual_change <= pred['q50']),
                'q75': int(actual_change <= pred['q75']),
                'q90': int(actual_change <= pred['q90'])
            }
            
            # Update database
            update_sql = f"""
            UPDATE {PREDICTION_DB_CONFIG['table_name']}
            SET actual_vix_change = ?,
                actual_regime = ?,
                point_error = ?,
                quantile_coverage = ?
            WHERE prediction_id = ?
            """
            
            self.conn.execute(update_sql, (
                actual_change,
                actual_regime,
                point_error,
                json.dumps(coverage),
                pred['prediction_id']
            ))
            
            updated += 1
        
        self.conn.commit()
        logger.info(f"✅ Backfilled {updated} predictions")
    
    
    def get_predictions(self, 
                       start_date: str = None,
                       end_date: str = None,
                       cohort: str = None,
                       with_actuals: bool = False) -> pd.DataFrame:
        """
        Query predictions from database.
        
        Args:
            start_date: Filter by forecast_date >= start_date
            end_date: Filter by forecast_date <= end_date
            cohort: Filter by calendar_cohort
            with_actuals: If True, only return predictions with actuals
            
        Returns:
            DataFrame with predictions
        """
        query = f"SELECT * FROM {PREDICTION_DB_CONFIG['table_name']} WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND forecast_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND forecast_date <= ?"
            params.append(end_date)
        
        if cohort:
            query += " AND calendar_cohort = ?"
            params.append(cohort)
        
        if with_actuals:
            query += " AND actual_vix_change IS NOT NULL"
        
        query += " ORDER BY forecast_date"
        
        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['forecast_date', 'timestamp'])
        
        return df
    
    
    def compute_quantile_coverage(self, cohort: str = None) -> Dict:
        """
        Compute empirical quantile coverage rates.
        
        For well-calibrated forecasts:
            - 10% of actuals should be <= q10
            - 25% of actuals should be <= q25
            - etc.
        
        Args:
            cohort: Compute for specific cohort (None = all)
            
        Returns:
            dict: {q10: 0.11, q25: 0.27, q50: 0.48, q75: 0.76, q90: 0.89}
                  (ideal would be {q10: 0.10, q25: 0.25, ...})
        """
        df = self.get_predictions(cohort=cohort, with_actuals=True)
        
        if len(df) == 0:
            logger.warning("No predictions with actuals")
            return {}
        
        coverage = {}
        for q in [10, 25, 50, 75, 90]:
            col = f'q{q}'
            covered = (df['actual_vix_change'] <= df[col]).mean()
            coverage[col] = float(covered)
        
        return coverage
    
    
    def compute_regime_brier_score(self, cohort: str = None) -> float:
        """
        Compute Brier score for regime probabilities.
        
        Brier score = mean((predicted_prob - actual_indicator)^2)
        Lower is better. Perfect score = 0.
        
        Args:
            cohort: Compute for specific cohort (None = all)
            
        Returns:
            float: Brier score
        """
        df = self.get_predictions(cohort=cohort, with_actuals=True)
        
        if len(df) == 0:
            return np.nan
        
        brier_scores = []
        
        for _, row in df.iterrows():
            # One-hot encode actual regime
            actual_onehot = {
                'low': 0, 'normal': 0, 'elevated': 0, 'crisis': 0
            }
            if row['actual_regime']:
                actual_onehot[row['actual_regime'].lower()] = 1
            
            # Compute squared errors
            brier = (
                (row['prob_low'] - actual_onehot['low']) ** 2 +
                (row['prob_normal'] - actual_onehot['normal']) ** 2 +
                (row['prob_elevated'] - actual_onehot['elevated']) ** 2 +
                (row['prob_crisis'] - actual_onehot['crisis']) ** 2
            )
            
            brier_scores.append(brier)
        
        return float(np.mean(brier_scores))
    
    
    def get_performance_summary(self) -> Dict:
        """
        Generate comprehensive performance metrics.
        
        Returns:
            dict: Summary statistics
        """
        df = self.get_predictions(with_actuals=True)
        
        if len(df) == 0:
            return {'error': 'No predictions with actuals'}
        
        summary = {
            'n_predictions': len(df),
            'date_range': {
                'start': df['forecast_date'].min().isoformat(),
                'end': df['forecast_date'].max().isoformat()
            },
            'point_estimate': {
                'mae': float(df['point_error'].mean()),
                'rmse': float(np.sqrt((df['actual_vix_change'] - df['point_estimate']) ** 2).mean())
            },
            'quantile_coverage': self.compute_quantile_coverage(),
            'regime_brier_score': self.compute_regime_brier_score(),
            'confidence_correlation': float(df[['confidence_score', 'point_error']].corr().iloc[0, 1])
        }
        
        # Per-cohort breakdown
        summary['by_cohort'] = {}
        for cohort in df['calendar_cohort'].unique():
            summary['by_cohort'][cohort] = {
                'n': int((df['calendar_cohort'] == cohort).sum()),
                'mae': float(df[df['calendar_cohort'] == cohort]['point_error'].mean()),
                'quantile_coverage': self.compute_quantile_coverage(cohort),
                'brier_score': self.compute_regime_brier_score(cohort)
            }
        
        return summary
    
    
    def export_to_csv(self, filename: str = 'predictions_export.csv'):
        """Export all predictions to CSV for external analysis."""
        df = self.get_predictions()
        df.to_csv(filename, index=False)
        logger.info(f"📄 Exported {len(df)} predictions to {filename}")
    
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("🔒 Database connection closed")
```

---

## USAGE EXAMPLES

### Example 1: Store Prediction
```python
from prediction_database import PredictionDatabase
import uuid

db = PredictionDatabase()

record = {
    'prediction_id': str(uuid.uuid4()),
    'timestamp': pd.Timestamp.now(),
    'forecast_date': pd.Timestamp('2025-11-15'),
    'horizon': 5,
    'calendar_cohort': 'mid_cycle',
    'cohort_weight': 1.0,
    'point_estimate': 8.5,
    'q10': -2.3, 'q25': 3.1, 'q50': 7.9, 'q75': 14.2, 'q90': 23.8,
    'prob_low': 0.02, 'prob_normal': 0.38, 'prob_elevated': 0.52, 'prob_crisis': 0.08,
    'confidence_score': 0.87,
    'feature_quality': 0.95,
    'current_vix': 18.5
}

db.store_prediction(record)
```

### Example 2: Backfill Actuals
```python
# After 5 days pass, fill in what actually happened
db.backfill_actuals()

# Check one prediction
df = db.get_predictions(start_date='2025-11-15', end_date='2025-11-15')
print(df[['point_estimate', 'actual_vix_change', 'point_error']])
```

### Example 3: Evaluate Calibration
```python
# Check quantile coverage
coverage = db.compute_quantile_coverage()
print(coverage)
# Expected: {'q10': 0.10, 'q25': 0.25, 'q50': 0.50, 'q75': 0.75, 'q90': 0.90}
# Actual might be: {'q10': 0.11, 'q25': 0.27, 'q50': 0.48, 'q75': 0.76, 'q90': 0.89}

# Check regime accuracy
brier = db.compute_regime_brier_score()
print(f"Brier score: {brier:.3f}")  # Lower is better, <0.2 is good
```

### Example 4: Performance Report
```python
summary = db.get_performance_summary()
print(json.dumps(summary, indent=2))
```

---

## TESTING

```python
def test_prediction_database():
    """Test database functionality."""
    import tempfile
    import uuid
    
    # Use temp database for testing
    with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
        db = PredictionDatabase(db_path=tmp.name)
        
        # Test 1: Store prediction
        record = {
            'prediction_id': str(uuid.uuid4()),
            'timestamp': pd.Timestamp.now(),
            'forecast_date': pd.Timestamp('2024-01-15'),
            'horizon': 5,
            'calendar_cohort': 'mid_cycle',
            'point_estimate': 5.0,
            'q10': -3.0, 'q25': 0.0, 'q50': 4.5, 'q75': 9.0, 'q90': 15.0,
            'prob_low': 0.1, 'prob_normal': 0.5, 'prob_elevated': 0.3, 'prob_crisis': 0.1,
            'confidence_score': 0.85,
            'current_vix': 18.0
        }
        db.store_prediction(record)
        
        # Test 2: Retrieve
        df = db.get_predictions()
        assert len(df) == 1
        assert df.iloc[0]['prediction_id'] == record['prediction_id']
        
        # Test 3: Backfill (with fake VIX data)
        vix_data = pd.Series([19.5], index=[pd.Timestamp('2024-01-15')])
        db.backfill_actuals(vix_data)
        
        df = db.get_predictions(with_actuals=True)
        assert len(df) == 1
        assert df.iloc[0]['actual_vix_change'] is not None
        
        print("✅ Database tests passed")

test_prediction_database()
```

---

## INTEGRATION CHECKLIST

- [ ] Create file: `src/prediction_database.py`
- [ ] Import in `integrated_system_production.py`
- [ ] Call `store_prediction()` after each forecast
- [ ] Schedule daily `backfill_actuals()` job (cron or similar)
- [ ] Create dashboard showing calibration metrics
- [ ] Export CSV for external analysis

---

## SUMMARY

**File Name:** `prediction_database.py`

**Lines of Code:** ~400

**Dependencies:**
- sqlite3 (built-in)
- pandas, numpy
- config.py (PREDICTION_DB_CONFIG)

**Key Methods:**
- `store_prediction()` - Save forecast
- `backfill_actuals()` - Fill outcomes
- `get_predictions()` - Query
- `compute_quantile_coverage()` - Calibration check
- `compute_regime_brier_score()` - Regime accuracy
- `get_performance_summary()` - Full report

**Database Location:** `data_cache/predictions.db` (configurable)

**Next Steps:**
1. Create this file
2. Test with dummy predictions
3. Integrate into integrated_system
4. Run batch backtest to fill database
5. Analyze calibration metrics
