"""
Prediction Database for Probabilistic Forecasting System

Stores all forecasts with full distribution + metadata for backtesting.
"""

import atexit
import json
import logging
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import PREDICTION_DB_CONFIG

logger = logging.getLogger(__name__)


class CommitTracker:
    """Tracks uncommitted writes and screams if data isn't committed."""

    def __init__(self):
        self.pending_writes = 0
        self.last_commit_time = None
        self.writes_log = []

    def track_write(self, operation: str):
        """Log a write operation."""
        self.pending_writes += 1
        self.writes_log.append(f"{datetime.now():%H:%M:%S} - {operation}")

        # Warn every 10 writes
        if self.pending_writes % 10 == 0:
            logger.warning(
                f"⚠️  {self.pending_writes} uncommitted writes! Call commit() soon!"
            )

    def verify_clean_exit(self):
        """Called on exit - SCREAMS if uncommitted data exists."""
        if self.pending_writes > 0:
            logger.error("=" * 80)
            logger.error("🚨 CRITICAL: UNCOMMITTED DATA DETECTED!")
            logger.error("=" * 80)
            logger.error(f"   Pending writes: {self.pending_writes}")
            logger.error(f"   Last 5 operations:")
            for op in self.writes_log[-5:]:
                logger.error(f"      • {op}")
            logger.error("")
            logger.error("   DATA WAS NOT SAVED!")
            logger.error("   You must call commit() before exit")
            logger.error("=" * 80)


class PredictionDatabase:
    """
    SQLite database for storing and retrieving probabilistic forecasts.

    Schema:
        - predictions: Main table with forecasts
        - actuals: Realized VIX values (joined by forecast_date)

    Example:
        >>> db = PredictionDatabase()
        >>> db.store_prediction(forecast_record)
        >>> db.commit()  # REQUIRED!
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
            db_path = PREDICTION_DB_CONFIG["db_path"]

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Dict-like access

        # Create/migrate schema (handles old schemas automatically)
        self._create_schema()

        # CRITICAL: Track in-memory keys during batch operations
        self._pending_keys = set()

        # SAFETY: Track uncommitted writes
        self._commit_tracker = CommitTracker()
        atexit.register(self._commit_tracker.verify_clean_exit)

        logger.info(f"📂 Prediction database: {self.db_path}")
        logger.info("⚠️  Safety mode enabled - must call commit() to persist writes")

    def _create_schema(self):
        """Create database tables with UNIQUE constraint to prevent duplicates."""

        # Check if table exists and has wrong schema
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='forecasts'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check if direction_probability column exists (wrong schema)
            cursor = self.conn.execute("PRAGMA table_info(forecasts)")
            columns = [row[1] for row in cursor.fetchall()]

            if "direction_probability" in columns:
                logger.warning(
                    "⚠️  Detected old schema with direction_probability, recreating table..."
                )

                # Backup data if any exists
                cursor = self.conn.execute("SELECT COUNT(*) FROM forecasts")
                count = cursor.fetchone()[0]

                if count > 0:
                    logger.info(f"   Backing up {count} existing records...")
                    self.conn.execute("""
                        CREATE TABLE forecasts_backup AS
                        SELECT * FROM forecasts
                    """)

                # Drop old table
                self.conn.execute("DROP TABLE forecasts")
                logger.info("   Dropped old table")

        # Create table with correct schema (no direction_probability)
        create_sql = """
        CREATE TABLE IF NOT EXISTS forecasts (
            prediction_id TEXT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            forecast_date DATE NOT NULL,
            horizon INTEGER NOT NULL,

            -- Context
            calendar_cohort TEXT,
            cohort_weight REAL,

            -- Predictions
            point_estimate REAL NOT NULL,
            q10 REAL,
            q25 REAL,
            q50 REAL,
            q75 REAL,
            q90 REAL,
            prob_low REAL,
            prob_normal REAL,
            prob_elevated REAL,
            prob_crisis REAL,
            confidence_score REAL,

            -- Metadata
            feature_quality REAL,
            num_features_used INTEGER,
            current_vix REAL,
            features_used TEXT,
            model_version TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            -- Actuals (filled later)
            actual_vix_change REAL,
            actual_regime TEXT,
            point_error REAL,
            quantile_coverage TEXT,

            -- UNIQUE constraint prevents duplicates
            UNIQUE(forecast_date, horizon)
        )
        """

        self.conn.execute(create_sql)

        # Restore data if backup exists
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='forecasts_backup'"
        )
        if cursor.fetchone() is not None:
            logger.info("   Restoring backed up data...")

            # Get all columns except direction_probability
            cursor = self.conn.execute("PRAGMA table_info(forecasts)")
            new_columns = [row[1] for row in cursor.fetchall()]

            # Insert data back (SQLite will ignore missing columns)
            try:
                self.conn.execute(f"""
                    INSERT OR IGNORE INTO forecasts ({", ".join(new_columns)})
                    SELECT {", ".join(new_columns)}
                    FROM forecasts_backup
                """)

                cursor = self.conn.execute("SELECT COUNT(*) FROM forecasts")
                restored = cursor.fetchone()[0]
                logger.info(f"   Restored {restored} records")

            except sqlite3.OperationalError as e:
                logger.warning(f"   Could not restore all data: {e}")

            # Drop backup table
            self.conn.execute("DROP TABLE forecasts_backup")

        # Create indexes for fast queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_forecast_date ON forecasts(forecast_date)",
            "CREATE INDEX IF NOT EXISTS idx_cohort ON forecasts(calendar_cohort)",
            "CREATE INDEX IF NOT EXISTS idx_has_actual ON forecasts(actual_vix_change) WHERE actual_vix_change IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_created_at ON forecasts(created_at)",
        ]

        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
            except sqlite3.OperationalError:
                pass  # Index already exists

        self.conn.commit()
        logger.info("✅ Database schema initialized")

    def store_prediction(self, record: Dict) -> Optional[str]:
        """
        Store a single prediction with atomic duplicate prevention.

        Args:
            record: Dict with prediction data

        Returns:
            prediction_id if stored successfully, None if duplicate/error
        """
        # Convert timestamps to ISO strings
        record = record.copy()
        for key in ["timestamp", "forecast_date", "created_at"]:
            if key in record and isinstance(record[key], pd.Timestamp):
                record[key] = record[key].isoformat()

        # Ensure created_at is set
        if "created_at" not in record:
            record["created_at"] = datetime.now().isoformat()

        # CRITICAL FIX: Remove deprecated fields that don't exist in schema
        deprecated_fields = ["direction_probability"]
        for field in deprecated_fields:
            if field in record:
                logger.debug(f"   Removing deprecated field: {field}")
                record.pop(field)

        # CRITICAL FIX: Add to pending keys FIRST (atomic operation)
        key = (record["forecast_date"], record["horizon"])

        if key in self._pending_keys:
            logger.warning(
                f"⚠️  Prediction already pending for {record['forecast_date']} "
                f"(horizon={record['horizon']}). Skipping."
            )
            return None

        # Claim this key immediately to prevent race conditions
        self._pending_keys.add(key)

        # Check database for existing records
        try:
            cursor = self.conn.execute(
                """
                SELECT prediction_id FROM forecasts
                WHERE forecast_date = ? AND horizon = ?
                """,
                (record["forecast_date"], record["horizon"]),
            )

            existing = cursor.fetchone()

            if existing:
                logger.warning(
                    f"⚠️  Prediction already exists for {record['forecast_date']} "
                    f"(horizon={record['horizon']}). Skipping."
                )
                # Clean up pending key since we're not inserting
                self._pending_keys.discard(key)
                return None

            # Build INSERT statement
            columns = list(record.keys())
            placeholders = ", ".join(["?" for _ in columns])

            insert_sql = f"""
            INSERT INTO forecasts
            ({", ".join(columns)})
            VALUES ({placeholders})
            """

            values = [record[col] for col in columns]

            # Execute insert
            self.conn.execute(insert_sql, values)

            # SAFETY: Track write
            self._commit_tracker.track_write(
                f"INSERT forecast_date={record['forecast_date']}"
            )

            logger.debug(f"💾 Stored prediction: {record['prediction_id']}")
            return record["prediction_id"]

        except sqlite3.IntegrityError as e:
            logger.warning(f"⚠️  Database integrity error (duplicate): {e}")
            self.conn.rollback()
            # Clean up pending key on error
            self._pending_keys.discard(key)
            return None

        except Exception as e:
            logger.error(f"❌ Failed to store prediction: {e}")
            logger.error(
                f"   Record: {record.get('forecast_date')}, horizon={record.get('horizon')}"
            )
            logger.error(f"   Available keys: {list(record.keys())}")
            self.conn.rollback()
            # Clean up pending key on error
            self._pending_keys.discard(key)
            raise

    def commit(self):
        """
        Commit pending transactions with verification.

        CRITICAL: Call this after batch operations to persist data and reset tracking.
        """
        if self._commit_tracker.pending_writes == 0:
            logger.info("ℹ️  No pending writes to commit")
            return

        writes_to_commit = self._commit_tracker.pending_writes

        try:
            # Commit
            self.conn.commit()
            self._pending_keys.clear()

            # Verify commit succeeded
            cursor = self.conn.execute("SELECT COUNT(*) FROM forecasts")
            total = cursor.fetchone()[0]

            logger.info("=" * 80)
            logger.info("✅ COMMIT SUCCESSFUL")
            logger.info("=" * 80)
            logger.info(f"   Writes committed: {writes_to_commit}")
            logger.info(f"   Total forecasts: {total}")
            logger.info(f"   Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")
            logger.info("=" * 80)

            # Reset tracker
            self._commit_tracker.pending_writes = 0
            self._commit_tracker.writes_log = []
            self._commit_tracker.last_commit_time = datetime.now()

        except Exception as e:
            logger.error("=" * 80)
            logger.error("🚨 COMMIT FAILED!")
            logger.error("=" * 80)
            logger.error(f"   Error: {e}")
            logger.error(f"   Attempted writes: {writes_to_commit}")
            logger.error("   Rolling back...")

            self.conn.rollback()
            self._commit_tracker.pending_writes = 0
            self._commit_tracker.writes_log = []

            logger.error("=" * 80)

            raise RuntimeError(f"Database commit failed: {e}")

    def get_commit_status(self) -> dict:
        """Get current commit status."""
        return {
            "pending_writes": self._commit_tracker.pending_writes,
            "last_commit": self._commit_tracker.last_commit_time.isoformat()
            if self._commit_tracker.last_commit_time
            else None,
            "recent_operations": self._commit_tracker.writes_log[-10:],
        }

    def get_predictions(
        self,
        start_date: str = None,
        end_date: str = None,
        cohort: str = None,
        with_actuals: bool = False,
    ) -> pd.DataFrame:
        """
        Query predictions from database.

        Args:
            start_date: Filter by forecast_date >= start_date
            end_date: Filter by forecast_date <= end_date
            cohort: Filter by calendar_cohort
            with_actuals: If True, only return predictions with actuals

        Returns:
            DataFrame with predictions (deduplicated)
        """
        query = "SELECT DISTINCT * FROM forecasts WHERE 1=1"
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

        try:
            df = pd.read_sql_query(
                query,
                self.conn,
                params=params,
                parse_dates=["forecast_date", "timestamp"],
            )

            # Safety: drop any duplicates that made it through
            before_count = len(df)
            df = df.drop_duplicates(subset=["forecast_date", "horizon"], keep="first")
            after_count = len(df)

            if before_count != after_count:
                logger.warning(
                    f"⚠️  Removed {before_count - after_count} duplicate rows from query results"
                )

            return df

        except Exception as e:
            logger.error(f"❌ Failed to query predictions: {e}")
            raise

    def migrate_schema(self):
        """
        Migrate existing database to remove direction_probability column.
        Safe to run multiple times.
        """
        logger.info("🔧 Checking database schema...")

        try:
            # Check if direction_probability exists
            cursor = self.conn.execute("PRAGMA table_info(forecasts)")
            columns = [row[1] for row in cursor.fetchall()]

            if "direction_probability" in columns:
                logger.info(
                    "⚠️  Found deprecated 'direction_probability' column, migrating..."
                )

                # Get all data
                cursor = self.conn.execute("SELECT * FROM forecasts")
                rows = cursor.fetchall()

                if len(rows) > 0:
                    logger.info(f"   Backing up {len(rows)} records...")

                    # Get column names (excluding direction_probability)
                    old_columns = [desc[0] for desc in cursor.description]
                    new_columns = [
                        col for col in old_columns if col != "direction_probability"
                    ]

                    # Drop old table
                    self.conn.execute("DROP TABLE forecasts")

                    # Recreate with new schema
                    self._create_schema()

                    # Reinsert data (excluding direction_probability)
                    placeholders = ", ".join(["?" for _ in new_columns])
                    insert_sql = f"INSERT INTO forecasts ({', '.join(new_columns)}) VALUES ({placeholders})"

                    for row in rows:
                        # Convert row to dict
                        row_dict = dict(zip(old_columns, row))
                        # Extract values for new columns only
                        values = [row_dict[col] for col in new_columns]
                        self.conn.execute(insert_sql, values)

                    self.conn.commit()
                    logger.info(f"✅ Migration complete: {len(rows)} records preserved")
                else:
                    # Empty table, just recreate
                    self.conn.execute("DROP TABLE forecasts")
                    self._create_schema()
                    logger.info("✅ Migration complete: recreated empty table")
            else:
                logger.info("✅ Schema is up to date")

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            self.conn.rollback()
            raise

    def remove_all_duplicates(self):
        """
        One-time cleanup: Remove all duplicate predictions, keeping the earliest.
        Run this once to clean existing data.
        """
        logger.info("🔍 Scanning for duplicate predictions...")

        # Find duplicates
        cursor = self.conn.execute("""
            SELECT forecast_date, horizon, COUNT(*) as count
            FROM forecasts
            GROUP BY forecast_date, horizon
            HAVING count > 1
        """)

        duplicates = cursor.fetchall()

        if len(duplicates) == 0:
            logger.info("✅ No duplicates found")
            return 0

        logger.info(f"⚠️  Found {len(duplicates)} duplicate date-horizon pairs")

        total_removed = 0
        for dup in duplicates:
            forecast_date, horizon, count = (
                dup["forecast_date"],
                dup["horizon"],
                dup["count"],
            )

            # Keep the earliest prediction (by timestamp), delete rest
            cursor = self.conn.execute(
                """
                DELETE FROM forecasts
                WHERE forecast_date = ? AND horizon = ?
                AND prediction_id NOT IN (
                    SELECT prediction_id FROM forecasts
                    WHERE forecast_date = ? AND horizon = ?
                    ORDER BY timestamp ASC
                    LIMIT 1
                )
            """,
                (forecast_date, horizon, forecast_date, horizon),
            )

            removed = cursor.rowcount
            total_removed += removed
            logger.debug(
                f"   Removed {removed} duplicates for {forecast_date} (horizon={horizon})"
            )

        self.conn.commit()
        logger.info(f"✅ Removed {total_removed} duplicate predictions")

        return total_removed

    def get_database_stats(self) -> Dict:
        """Get statistics about the database state."""
        try:
            cursor = self.conn.execute("SELECT COUNT(*) as total FROM forecasts")
            total = cursor.fetchone()["total"]

            cursor = self.conn.execute("""
                SELECT COUNT(*) as with_actuals
                FROM forecasts
                WHERE actual_vix_change IS NOT NULL
            """)
            with_actuals = cursor.fetchone()["with_actuals"]

            cursor = self.conn.execute("""
                SELECT forecast_date, horizon, COUNT(*) as count
                FROM forecasts
                GROUP BY forecast_date, horizon
                HAVING count > 1
            """)
            duplicates = len(cursor.fetchall())

            cursor = self.conn.execute("""
                SELECT MIN(forecast_date) as earliest, MAX(forecast_date) as latest
                FROM forecasts
            """)
            dates = cursor.fetchone()

            return {
                "total_predictions": total,
                "with_actuals": with_actuals,
                "pending_actuals": total - with_actuals,
                "duplicate_pairs": duplicates,
                "date_range": {
                    "earliest": dates["earliest"],
                    "latest": dates["latest"],
                },
            }
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {"error": str(e)}

    def backfill_actuals(self, fetcher=None):
        """Populate actual outcomes for forecasts whose target dates have passed."""
        if fetcher is None:
            from core.data_fetcher import UnifiedDataFetcher

            fetcher = UnifiedDataFetcher()

        logger.info("🔄 Backfilling actual outcomes...")

        # Get VIX data
        vix_data = fetcher.fetch_yahoo("^VIX", start_date="2009-01-01")["Close"]

        # Convert index to date strings for matching
        vix_dates = set(vix_data.index.strftime("%Y-%m-%d"))

        # Find predictions needing actuals
        query = """
            SELECT prediction_id, forecast_date, horizon, current_vix,
                   q10, q25, q50, q75, q90
            FROM forecasts
            WHERE actual_vix_change IS NULL
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        logger.info(f"   Found {len(rows)} predictions to backfill")

        updated = 0
        skipped = 0
        for row in rows:
            pred_id, forecast_date, horizon, current_vix, q10, q25, q50, q75, q90 = row

            # Calculate target date using BUSINESS DAYS
            try:
                target_date_attempt = pd.bdate_range(
                    start=pd.Timestamp(forecast_date), periods=horizon + 1
                )[-1]
            except Exception as e:
                logger.warning(
                    f"   Failed to calculate business days for {forecast_date}: {e}"
                )
                skipped += 1
                continue

            # Find nearest actual VIX date (within 3 days forward to handle holidays)
            found_date = None
            for offset in range(4):  # Check up to 3 days forward
                check_date = (target_date_attempt + pd.Timedelta(days=offset)).strftime(
                    "%Y-%m-%d"
                )
                if check_date in vix_dates:
                    found_date = check_date
                    break

            if found_date is None:
                skipped += 1
                continue

            target_date = found_date

            # Get actual VIX value at target date
            actual_vix = vix_data.loc[pd.Timestamp(target_date)]
            actual_change = ((actual_vix - current_vix) / current_vix) * 100

            # Classify regime
            if actual_change < -5:
                regime = "Low"
            elif actual_change < 10:
                regime = "Normal"
            elif actual_change < 25:
                regime = "Elevated"
            else:
                regime = "Crisis"

            # Get point estimate
            cursor.execute(
                "SELECT point_estimate FROM forecasts WHERE prediction_id = ?",
                (pred_id,),
            )
            point_est = cursor.fetchone()[0]
            point_error = abs(actual_change - point_est)

            # **CRITICAL FIX: Compute quantile coverage**
            quantile_coverage = {
                "q10": 1 if actual_change <= q10 else 0,
                "q25": 1 if actual_change <= q25 else 0,
                "q50": 1 if actual_change <= q50 else 0,
                "q75": 1 if actual_change <= q75 else 0,
                "q90": 1 if actual_change <= q90 else 0,
            }
            coverage_json = json.dumps(quantile_coverage)

            # Update database with ALL fields
            cursor.execute(
                """
                UPDATE forecasts
                SET actual_vix_change = ?,
                    actual_regime = ?,
                    point_error = ?,
                    quantile_coverage = ?
                WHERE prediction_id = ?
                """,
                (actual_change, regime, point_error, coverage_json, pred_id),
            )
            updated += 1

        self.conn.commit()
        logger.info(f"✅ Backfilled {updated} predictions")
        if skipped > 0:
            logger.info(
                f"⚠️  Skipped {skipped} predictions (target date not yet available)"
            )

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
            col = f"q{q}"
            covered = (df["actual_vix_change"] <= df[col]).mean()
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
            actual_onehot = {"low": 0, "normal": 0, "elevated": 0, "crisis": 0}
            if row["actual_regime"]:
                actual_onehot[row["actual_regime"].lower()] = 1

            # Compute squared errors
            brier = (
                (row["prob_low"] - actual_onehot["low"]) ** 2
                + (row["prob_normal"] - actual_onehot["normal"]) ** 2
                + (row["prob_elevated"] - actual_onehot["elevated"]) ** 2
                + (row["prob_crisis"] - actual_onehot["crisis"]) ** 2
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
            return {"error": "No predictions with actuals"}

        summary = {
            "n_predictions": len(df),
            "date_range": {
                "start": df["forecast_date"].min().isoformat(),
                "end": df["forecast_date"].max().isoformat(),
            },
            "point_estimate": {
                "mae": float(df["point_error"].mean()),
                "rmse": float(
                    np.sqrt(
                        (df["actual_vix_change"] - df["point_estimate"]) ** 2
                    ).mean()
                ),
            },
            "quantile_coverage": self.compute_quantile_coverage(),
            "regime_brier_score": self.compute_regime_brier_score(),
            "confidence_correlation": float(
                df[["confidence_score", "point_error"]].corr().iloc[0, 1]
            ),
        }

        # Per-cohort breakdown
        summary["by_cohort"] = {}
        for cohort in df["calendar_cohort"].unique():
            summary["by_cohort"][cohort] = {
                "n": int((df["calendar_cohort"] == cohort).sum()),
                "mae": float(df[df["calendar_cohort"] == cohort]["point_error"].mean()),
                "quantile_coverage": self.compute_quantile_coverage(cohort),
                "brier_score": self.compute_regime_brier_score(cohort),
            }

        return summary

    def export_to_csv(self, filename: str = "predictions_export.csv"):
        """Export all predictions to CSV for external analysis."""
        df = self.get_predictions()
        df.to_csv(filename, index=False)
        logger.info(f"📄 Exported {len(df)} predictions to {filename}")

    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("🔒 Database connection closed")
