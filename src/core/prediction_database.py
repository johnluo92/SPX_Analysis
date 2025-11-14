"""
Prediction Database Module - Log-Transformed Realized Volatility System V3

CRITICAL CHANGES FROM V2:
1. Schema updated: median_forecast is now PRIMARY forecast metric
2. point_estimate maintained for BACKWARD COMPATIBILITY (populated with median value)
3. New error metrics: median_error (primary), point_error (backward compat)
4. Quantile storage: q10, q25, q50, q75, q90
5. All storage operations handle median_forecast correctly

Author: VIX Forecasting System
Last Updated: 2025-11-13
Version: 3.0 (Log-RV Quantile Regression)
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionDatabase:
    """
    Manages storage and retrieval of VIX forecasts and actuals.

    SCHEMA PHILOSOPHY V3:
    - median_forecast: Primary forecast from q50 quantile model
    - point_estimate: Backward compatibility (equals median_forecast)
    - q10-q90: Full quantile distribution
    - median_error: Primary error metric |actual - median_forecast|
    - point_error: Backward compatibility (equals median_error)

    This dual approach ensures:
    1. New code uses median_forecast (better statistics)
    2. Old code continues to work (point_estimate exists)
    3. Easy migration path (both fields identical)
    """

    def __init__(self, db_path: str = "data_cache/predictions.db"):
        """Initialize database connection and ensure schema exists."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection with optimized settings
        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._create_schema()
        logger.info(f"Database initialized: {self.db_path}")

    def _create_schema(self):
        """
        Create database schema optimized for log-RV quantile forecasting.

        KEY DESIGN DECISIONS:
        1. median_forecast NOT NULL - this is the primary forecast
        2. point_estimate nullable - backward compat, auto-populated with median
        3. UNIQUE(forecast_date, horizon) - prevents duplicate forecasts
        4. created_at DEFAULT CURRENT_TIMESTAMP - audit trail
        """

        create_forecasts_table = """
        CREATE TABLE IF NOT EXISTS forecasts (
            -- Primary Key
            prediction_id TEXT PRIMARY KEY,

            -- Temporal Context
            timestamp DATETIME NOT NULL,
            forecast_date DATE NOT NULL,
            horizon INTEGER NOT NULL,

            -- Calendar Context
            calendar_cohort TEXT,
            cohort_weight REAL,

            -- PRIMARY FORECAST (NEW IN V3)
            median_forecast REAL NOT NULL,  -- From q50 quantile model

            -- Quantile Distribution (TRUE quantile regression)
            q10 REAL,
            q25 REAL,
            q50 REAL,  -- Should equal median_forecast
            q75 REAL,
            q90 REAL,

            -- Backward Compatibility (AUTO-POPULATED)
            point_estimate REAL,  -- Populated with median_forecast value

            -- Regime Probabilities
            prob_low REAL,
            prob_normal REAL,
            prob_elevated REAL,
            prob_crisis REAL,

            -- Directional Forecast
            direction_probability REAL,  -- P(VIX increases)
            confidence_score REAL,

            -- Feature Metadata
            feature_quality REAL,
            regime_stability REAL,
            num_features_used INTEGER,
            missing_features TEXT,

            -- Current Market State
            current_vix REAL,
            features_used TEXT,

            -- System Metadata
            model_version TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            -- Actuals (filled after forecast period)
            actual_vix_change REAL,
            actual_regime TEXT,

            -- ERROR METRICS (NEW IN V3)
            median_error REAL,  -- |actual - median_forecast|
            point_error REAL,   -- Backward compat (equals median_error)
            quantile_coverage TEXT,  -- JSON of which quantiles covered actual

            -- Prevent duplicate forecasts for same date/horizon
            UNIQUE(forecast_date, horizon)
        )
        """

        # Create indices for common queries
        create_indices = [
            "CREATE INDEX IF NOT EXISTS idx_forecast_date ON forecasts(forecast_date)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON forecasts(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_cohort ON forecasts(calendar_cohort)",
            "CREATE INDEX IF NOT EXISTS idx_actuals ON forecasts(actual_vix_change)",
        ]

        try:
            self.conn.execute(create_forecasts_table)
            for idx_sql in create_indices:
                self.conn.execute(idx_sql)
            self.conn.commit()
            logger.info("✅ Database schema created/verified")

        except sqlite3.Error as e:
            logger.error(f"❌ Schema creation failed: {e}")
            raise

    @contextmanager
    def transaction(self):
        """Context manager for atomic transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise

    def store_prediction(self, record: Dict) -> Optional[str]:
        """
        Store a single prediction with proper median_forecast handling.

        CRITICAL V3 CHANGES:
        1. Extracts median_forecast from quantiles if not present
        2. Auto-populates point_estimate = median_forecast
        3. Ensures q50 = median_forecast (consistency check)
        4. Validates quantile ordering

        Args:
            record: Dictionary with prediction data

        Returns:
            prediction_id if successful, None if failed

        Example record structure:
            {
                'prediction_id': 'pred_20250113_21d',
                'timestamp': pd.Timestamp('2025-01-13'),
                'forecast_date': pd.Timestamp('2025-01-13'),
                'horizon': 21,
                'median_forecast': 2.5,  # PRIMARY
                'quantiles': {'q10': -1.0, 'q25': 1.0, 'q50': 2.5, ...},
                'direction_probability': 0.65,
                ...
            }
        """

        # Deep copy to avoid mutating input
        record = record.copy()

        # ============================================================
        # STEP 1: Extract and validate median_forecast
        # ============================================================

        if "median_forecast" not in record:
            # Try to extract from quantiles
            if "quantiles" in record and isinstance(record["quantiles"], dict):
                if "q50" in record["quantiles"]:
                    record["median_forecast"] = record["quantiles"]["q50"]
                    logger.debug("Extracted median_forecast from quantiles['q50']")
                else:
                    logger.error("❌ No median_forecast or q50 in quantiles")
                    return None
            else:
                logger.error("❌ No median_forecast found and no quantiles dict")
                return None

        # Validate median_forecast is numeric
        try:
            median_value = float(record["median_forecast"])
            record["median_forecast"] = median_value
        except (ValueError, TypeError):
            logger.error(f"❌ Invalid median_forecast: {record['median_forecast']}")
            return None

        # ============================================================
        # STEP 2: Populate point_estimate for backward compatibility
        # ============================================================

        record["point_estimate"] = record["median_forecast"]

        # ============================================================
        # STEP 3: Extract quantiles from nested dict if present
        # ============================================================

        if "quantiles" in record and isinstance(record["quantiles"], dict):
            quantiles = record["quantiles"]

            # Extract each quantile
            for q in ["q10", "q25", "q50", "q75", "q90"]:
                if q in quantiles:
                    record[q] = quantiles[q]

            # Remove nested dict (we've extracted values to top level)
            del record["quantiles"]

        # ============================================================
        # STEP 4: Consistency check - q50 should equal median_forecast
        # ============================================================

        if "q50" in record:
            if abs(record["q50"] - record["median_forecast"]) > 0.01:
                logger.warning(
                    f"⚠️  q50 ({record['q50']:.4f}) != median_forecast "
                    f"({record['median_forecast']:.4f})"
                )
                # Use median_forecast as source of truth
                record["q50"] = record["median_forecast"]
        else:
            # If q50 wasn't provided, set it equal to median_forecast
            record["q50"] = record["median_forecast"]

        # ============================================================
        # STEP 5: Validate quantile ordering (if we have multiple quantiles)
        # ============================================================

        quantile_keys = ["q10", "q25", "q50", "q75", "q90"]
        present_quantiles = {q: record[q] for q in quantile_keys if q in record}

        if len(present_quantiles) >= 2:
            values = list(present_quantiles.values())
            if values != sorted(values):
                logger.error(f"❌ Quantiles not properly ordered: {present_quantiles}")
                return None

        # ============================================================
        # STEP 6: Convert timestamps to ISO strings
        # ============================================================

        for key in ["timestamp", "forecast_date", "created_at"]:
            if key in record and isinstance(record[key], pd.Timestamp):
                record[key] = record[key].isoformat()

        # Ensure created_at is set
        if "created_at" not in record:
            record["created_at"] = datetime.now().isoformat()

        # ============================================================
        # STEP 7: Convert lists/dicts to JSON strings
        # ============================================================

        if "missing_features" in record and isinstance(
            record["missing_features"], list
        ):
            record["missing_features"] = ",".join(record["missing_features"])

        if "features_used" in record and isinstance(record["features_used"], list):
            record["features_used"] = ",".join(record["features_used"])

        # ============================================================
        # STEP 8: Build INSERT statement dynamically
        # ============================================================

        # Filter to only include columns that exist in schema
        schema_columns = self._get_schema_columns()
        filtered_record = {k: v for k, v in record.items() if k in schema_columns}

        columns = list(filtered_record.keys())
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)

        insert_sql = f"""
        INSERT OR REPLACE INTO forecasts ({columns_str})
        VALUES ({placeholders})
        """

        # ============================================================
        # STEP 9: Execute with transaction
        # ============================================================

        try:
            with self.transaction():
                self.conn.execute(insert_sql, list(filtered_record.values()))

            logger.debug(
                f"✅ Stored prediction: {record['prediction_id']} "
                f"(median={record['median_forecast']:.2f}%)"
            )
            return record["prediction_id"]

        except sqlite3.IntegrityError as e:
            logger.error(f"❌ Duplicate or constraint violation: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to store prediction: {e}")
            return None

    def _get_schema_columns(self) -> set:
        """Get set of column names from database schema."""
        cursor = self.conn.execute("PRAGMA table_info(forecasts)")
        columns = {row[1] for row in cursor.fetchall()}
        return columns

    def get_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        with_actuals: bool = False,
        cohort: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve predictions with optional filtering.

        Args:
            start_date: Filter to predictions on/after this date
            end_date: Filter to predictions on/before this date
            with_actuals: If True, only return predictions with actual values
            cohort: Filter to specific calendar cohort

        Returns:
            DataFrame with predictions, sorted by forecast_date descending
        """

        # Build WHERE clauses
        conditions = []
        params = []

        if start_date:
            conditions.append("forecast_date >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("forecast_date <= ?")
            params.append(end_date)

        if with_actuals:
            conditions.append("actual_vix_change IS NOT NULL")

        if cohort:
            conditions.append("calendar_cohort = ?")
            params.append(cohort)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
        SELECT * FROM forecasts
        WHERE {where_clause}
        ORDER BY forecast_date DESC
        """

        try:
            df = pd.read_sql_query(query, self.conn, params=params)

            # Convert date columns
            if len(df) > 0:
                df["forecast_date"] = pd.to_datetime(df["forecast_date"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                if "created_at" in df.columns:
                    df["created_at"] = pd.to_datetime(df["created_at"])

            return df

        except Exception as e:
            logger.error(f"❌ Failed to retrieve predictions: {e}")
            return pd.DataFrame()

    def backfill_actuals(self, vix_series: pd.Series, horizon: int = 21):
        """
        Backfill actual outcomes and compute errors for past predictions.

        CRITICAL V3 CHANGES:
        1. Computes median_error as |actual - median_forecast|
        2. Sets point_error = median_error for backward compatibility
        3. Computes quantile_coverage to validate quantile models

        The quantile coverage tells us if our quantile models are well-calibrated:
        - q10 should be exceeded by ~10% of actuals
        - q25 should be exceeded by ~25% of actuals
        - q50 should be exceeded by ~50% of actuals (median)
        - etc.

        Args:
            vix_series: Series with VIX values indexed by date
            horizon: Forecast horizon in days (default 21)
        """

        logger.info("Starting actuals backfill...")

        # Get all predictions without actuals
        query = """
        SELECT prediction_id, forecast_date, horizon,
               median_forecast, point_estimate,
               q10, q25, q50, q75, q90,
               current_vix
        FROM forecasts
        WHERE actual_vix_change IS NULL
        ORDER BY forecast_date
        """

        predictions = pd.read_sql_query(query, self.conn)

        if len(predictions) == 0:
            logger.info("No predictions need backfilling")
            return

        predictions["forecast_date"] = pd.to_datetime(predictions["forecast_date"])

        updated_count = 0
        errors = []

        for idx, pred in predictions.iterrows():
            forecast_date = pred["forecast_date"]
            pred_horizon = pred["horizon"]

            # Calculate target date
            target_date = forecast_date + pd.Timedelta(days=pred_horizon)

            # Get actual VIX at target date
            if target_date not in vix_series.index:
                continue

            actual_vix = vix_series[target_date]
            current_vix = pred["current_vix"]

            if pd.isna(actual_vix) or pd.isna(current_vix):
                continue

            # ============================================================
            # COMPUTE ACTUAL VIX CHANGE
            # ============================================================

            actual_change = ((actual_vix - current_vix) / current_vix) * 100

            # ============================================================
            # COMPUTE MEDIAN ERROR (PRIMARY METRIC)
            # ============================================================

            median_forecast = pred["median_forecast"]
            median_error = abs(actual_change - median_forecast)

            # ============================================================
            # BACKWARD COMPATIBILITY: point_error = median_error
            # ============================================================

            point_error = median_error

            # ============================================================
            # COMPUTE QUANTILE COVERAGE
            # ============================================================

            coverage = {}
            for q in ["q10", "q25", "q50", "q75", "q90"]:
                if pd.notna(pred[q]):
                    # Check if actual is below this quantile
                    coverage[q] = actual_change <= pred[q]

            # Convert to JSON string
            import json

            coverage_str = json.dumps(coverage) if coverage else None

            # ============================================================
            # DETERMINE REGIME
            # ============================================================

            if actual_vix < 15:
                regime = "low"
            elif actual_vix < 25:
                regime = "normal"
            elif actual_vix < 40:
                regime = "elevated"
            else:
                regime = "crisis"

            # ============================================================
            # UPDATE DATABASE
            # ============================================================

            update_sql = """
            UPDATE forecasts
            SET actual_vix_change = ?,
                actual_regime = ?,
                median_error = ?,
                point_error = ?,
                quantile_coverage = ?
            WHERE prediction_id = ?
            """

            try:
                self.conn.execute(
                    update_sql,
                    (
                        actual_change,
                        regime,
                        median_error,
                        point_error,
                        coverage_str,
                        pred["prediction_id"],
                    ),
                )
                updated_count += 1
                errors.append(median_error)

            except Exception as e:
                logger.error(f"Failed to update {pred['prediction_id']}: {e}")

        self.conn.commit()

        # ============================================================
        # REPORT STATISTICS
        # ============================================================

        if updated_count > 0:
            logger.info(f"✅ Backfilled {updated_count} predictions")
            logger.info(f"   Mean median error: {np.mean(errors):.2f}%")
            logger.info(f"   Median error: {np.median(errors):.2f}%")
            logger.info(f"   Std error: {np.std(errors):.2f}%")

            # Check quantile coverage
            self._report_quantile_coverage()
        else:
            logger.info("No predictions were updated")

    def _report_quantile_coverage(self):
        """
        Report how well quantile forecasts are calibrated.

        Expected coverage:
        - q10 should be exceeded by ~90% of actuals
        - q25 should be exceeded by ~75% of actuals
        - q50 should be exceeded by ~50% of actuals
        - q75 should be exceeded by ~25% of actuals
        - q90 should be exceeded by ~10% of actuals
        """

        query = """
        SELECT q10, q25, q50, q75, q90, actual_vix_change
        FROM forecasts
        WHERE actual_vix_change IS NOT NULL
        AND q10 IS NOT NULL
        AND q90 IS NOT NULL
        """

        df = pd.read_sql_query(query, self.conn)

        if len(df) == 0:
            logger.info("Not enough data for quantile coverage analysis")
            return

        logger.info("\nQuantile Coverage Analysis:")
        logger.info("-" * 50)

        for q_name, target_pct in [
            ("q10", 10),
            ("q25", 25),
            ("q50", 50),
            ("q75", 75),
            ("q90", 90),
        ]:
            if q_name in df.columns:
                # What percentage of actuals are below this quantile?
                actual_coverage = (df["actual_vix_change"] <= df[q_name]).mean() * 100
                error = actual_coverage - target_pct

                logger.info(
                    f"   {q_name}: {actual_coverage:.1f}% "
                    f"(target: {target_pct}%, error: {error:+.1f}%)"
                )

    def get_performance_summary(self) -> Dict:
        """
        Generate comprehensive performance summary for V3 system.

        Returns dict with:
        - Overall statistics
        - Quantile coverage
        - Direction accuracy
        - Performance by cohort
        """

        query = """
        SELECT
            forecast_date,
            calendar_cohort,
            median_forecast,
            median_error,
            q10, q25, q50, q75, q90,
            actual_vix_change,
            direction_probability,
            confidence_score
        FROM forecasts
        WHERE actual_vix_change IS NOT NULL
        ORDER BY forecast_date
        """

        df = pd.read_sql_query(query, self.conn)

        if len(df) == 0:
            return {"error": "No predictions with actuals available"}

        # Overall statistics
        summary = {
            "total_predictions": len(df),
            "median_error": {
                "mean": float(df["median_error"].mean()),
                "median": float(df["median_error"].median()),
                "std": float(df["median_error"].std()),
                "min": float(df["median_error"].min()),
                "max": float(df["median_error"].max()),
            },
        }

        # Quantile coverage
        coverage = {}
        for q_name, target in [
            ("q10", 0.10),
            ("q25", 0.25),
            ("q50", 0.50),
            ("q75", 0.75),
            ("q90", 0.90),
        ]:
            if q_name in df.columns and df[q_name].notna().any():
                actual_cov = (df["actual_vix_change"] <= df[q_name]).mean()
                coverage[q_name] = {
                    "actual": float(actual_cov),
                    "target": float(target),
                    "error": float(actual_cov - target),
                }

        summary["quantile_coverage"] = coverage

        # Direction accuracy
        if "direction_probability" in df.columns:
            df_dir = df.dropna(subset=["direction_probability"])
            if len(df_dir) > 0:
                predicted_up = df_dir["direction_probability"] > 0.5
                actual_up = df_dir["actual_vix_change"] > 0
                accuracy = (predicted_up == actual_up).mean()
                summary["direction_accuracy"] = float(accuracy)

        # Performance by cohort
        if "calendar_cohort" in df.columns:
            cohort_stats = {}
            for cohort in df["calendar_cohort"].unique():
                if pd.isna(cohort):
                    continue
                cohort_df = df[df["calendar_cohort"] == cohort]
                cohort_stats[cohort] = {
                    "count": len(cohort_df),
                    "mean_error": float(cohort_df["median_error"].mean()),
                    "median_error": float(cohort_df["median_error"].median()),
                }
            summary["by_cohort"] = cohort_stats

        return summary

    def close(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()
            logger.info("Database connection closed")

    def __del__(self):
        """Ensure connection is closed on deletion."""
        self.close()


# ============================================================
# TESTING AND VALIDATION
# ============================================================


def validate_schema():
    """Validate that database schema matches V3 requirements."""
    db = PredictionDatabase()

    required_columns = [
        "median_forecast",
        "median_error",
        "point_estimate",
        "point_error",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
    ]

    schema_columns = db._get_schema_columns()

    missing = [col for col in required_columns if col not in schema_columns]

    if missing:
        logger.error(f"❌ Missing required columns: {missing}")
        return False
    else:
        logger.info("✅ Schema validation passed")
        return True


if __name__ == "__main__":
    """
    Test database operations and schema validation.
    """

    print("=" * 80)
    print("PREDICTION DATABASE V3 - VALIDATION")
    print("=" * 80)

    # Test 1: Schema validation
    print("\n1. Validating schema...")
    if validate_schema():
        print("   ✅ Schema correct")
    else:
        print("   ❌ Schema validation failed")
        exit(1)

    # Test 2: Store test prediction
    print("\n2. Testing prediction storage...")
    db = PredictionDatabase()

    test_record = {
        "prediction_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": pd.Timestamp.now(),
        "forecast_date": pd.Timestamp.now(),
        "horizon": 21,
        "calendar_cohort": "mid_month",
        "cohort_weight": 1.0,
        "median_forecast": 2.5,
        "quantiles": {
            "q10": -1.0,
            "q25": 1.0,
            "q50": 2.5,
            "q75": 4.0,
            "q90": 6.0,
        },
        "direction_probability": 0.65,
        "confidence_score": 0.75,
        "current_vix": 15.0,
        "model_version": "v3.0",
    }

    pred_id = db.store_prediction(test_record)

    if pred_id:
        print(f"   ✅ Test prediction stored: {pred_id}")

        # Verify median = point
        query = "SELECT median_forecast, point_estimate FROM forecasts WHERE prediction_id = ?"
        result = pd.read_sql_query(query, db.conn, params=[pred_id])

        if len(result) > 0:
            med = result.iloc[0]["median_forecast"]
            pt = result.iloc[0]["point_estimate"]
            if abs(med - pt) < 0.001:
                print(f"   ✅ Backward compatibility: median={med:.2f}, point={pt:.2f}")
            else:
                print(f"   ❌ Mismatch: median={med:.2f}, point={pt:.2f}")
    else:
        print("   ❌ Failed to store test prediction")

    # Test 3: Retrieve predictions
    print("\n3. Testing prediction retrieval...")
    df = db.get_predictions()
    print(f"   Retrieved {len(df)} predictions")

    if len(df) > 0:
        print(f"   Columns: {list(df.columns)}")
        print(f"   ✅ Retrieval successful")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
