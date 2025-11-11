"""
Prediction Database for Probabilistic Forecasting System

Stores all forecasts with full distribution + metadata for backtesting.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
            db_path = PREDICTION_DB_CONFIG["db_path"]

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Dict-like access

        self._create_schema()

        # CRITICAL: Track in-memory keys during batch operations
        self._pending_keys = set()

        logger.info(f"📂 Prediction database: {self.db_path}")

    def _create_schema(self):
        """Create database tables with UNIQUE constraint to prevent duplicates."""
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
            q10 REAL, q25 REAL, q50 REAL, q75 REAL, q90 REAL,
            prob_low REAL, prob_normal REAL, prob_elevated REAL, prob_crisis REAL,
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
            self.conn.rollback()
            # Clean up pending key on error
            self._pending_keys.discard(key)
            raise

    def commit(self):
        """
        Commit pending transactions and clear in-memory tracking.

        CRITICAL: Call this after batch operations to persist data and reset tracking.
        """
        self.conn.commit()
        self._pending_keys.clear()
        logger.debug("✅ Committed batch and cleared pending keys")

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

    def backfill_actuals(self, vix_data: pd.Series = None):
        """
        Fill in actual outcomes for past predictions.

        Args:
            vix_data: Series of VIX values indexed by date
                      If None, fetches from Yahoo Finance
        """
        logger.info("🔄 Backfilling actual outcomes...")

        # Fetch VIX data if not provided
        if vix_data is None:
            import yfinance as yf

            vix_df = yf.download("^VIX", progress=False)
            if isinstance(vix_df, pd.DataFrame):
                vix_data = vix_df["Close"]
            else:
                vix_data = vix_df

            # Ensure it's a Series
            if isinstance(vix_data, pd.DataFrame):
                vix_data = vix_data.squeeze()

            logger.debug(
                f"Fetched VIX data: {len(vix_data)} rows, type={type(vix_data)}"
            )

        # Get predictions needing actuals
        query = f"""
        SELECT prediction_id, forecast_date, horizon, current_vix, point_estimate,
               q10, q25, q50, q75, q90
        FROM {PREDICTION_DB_CONFIG["table_name"]}
        WHERE actual_vix_change IS NULL
          AND forecast_date <= date('now', '-' || horizon || ' days')
        """

        cursor = self.conn.execute(query)
        predictions = cursor.fetchall()

        logger.info(f"   Found {len(predictions)} predictions to backfill")

        updated = 0
        for pred in predictions:
            forecast_date = pd.Timestamp(pred["forecast_date"])
            horizon = pred["horizon"]
            target_date = forecast_date + pd.Timedelta(days=horizon)

            # Get actual VIX at target date
            try:
                # Try exact date first
                if target_date in vix_data.index:
                    actual_vix = vix_data.loc[target_date]
                else:
                    # Try next business day (handle holidays)
                    actual_vix = vix_data.asof(target_date)

                # CRITICAL: Convert to Python float immediately
                if isinstance(actual_vix, pd.Series):
                    if len(actual_vix) == 1:
                        actual_vix = float(actual_vix.iloc[0])
                    else:
                        logger.warning(
                            f"   Multiple VIX values for {target_date}, skipping"
                        )
                        continue
                else:
                    actual_vix = float(actual_vix)

                # Skip if we couldn't find data
                if pd.isna(actual_vix):
                    continue

                current_vix = float(pred["current_vix"])

                # Compute actual % change
                actual_change = (actual_vix - current_vix) / current_vix * 100

                # Determine actual regime
                from config import TARGET_CONFIG

                boundaries = TARGET_CONFIG["regimes"]["boundaries"]
                labels = TARGET_CONFIG["regimes"]["labels"]

                # Use scalar comparisons
                if actual_vix < boundaries[0]:
                    actual_regime = labels[0]  # Low
                elif actual_vix < boundaries[1]:
                    actual_regime = labels[1]  # Normal
                elif actual_vix < boundaries[2]:
                    actual_regime = labels[2]  # Elevated
                else:
                    actual_regime = labels[3]  # Crisis

                # Compute point error
                point_error = abs(actual_change - pred["point_estimate"])

                # Check quantile coverage
                coverage = {
                    "q10": int(actual_change <= float(pred["q10"])),
                    "q25": int(actual_change <= float(pred["q25"])),
                    "q50": int(actual_change <= float(pred["q50"])),
                    "q75": int(actual_change <= float(pred["q75"])),
                    "q90": int(actual_change <= float(pred["q90"])),
                }

                # Update database
                update_sql = f"""
                UPDATE {PREDICTION_DB_CONFIG["table_name"]}
                SET actual_vix_change = ?,
                    actual_regime = ?,
                    point_error = ?,
                    quantile_coverage = ?
                WHERE prediction_id = ?
                """

                self.conn.execute(
                    update_sql,
                    (
                        actual_change,
                        actual_regime,
                        point_error,
                        json.dumps(coverage),
                        pred["prediction_id"],
                    ),
                )

                updated += 1

            except Exception as e:
                logger.debug(f"   Failed to backfill {pred['prediction_id']}: {e}")
                logger.debug(
                    f"      target_date={target_date}, forecast_date={forecast_date}"
                )
                logger.debug(
                    f"      vix_data type={type(vix_data)}, index type={type(vix_data.index)}"
                )
                continue

        self.conn.commit()
        logger.info(f"✅ Backfilled {updated} predictions")

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
