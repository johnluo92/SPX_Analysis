#!/usr/bin/env python3
"""
Database Investigation - Find out why quantile_coverage is inconsistent
"""

import sqlite3
import pandas as pd
from datetime import datetime

db_path = "data_cache/predictions.db"
conn = sqlite3.connect(db_path)

print("=" * 80)
print("DATABASE INVESTIGATION: quantile_coverage Inconsistency")
print("=" * 80)

# 1. Check schema
print("\n[1] DATABASE SCHEMA:")
cursor = conn.execute("PRAGMA table_info(forecasts)")
columns = cursor.fetchall()
for col in columns:
    print(f"   {col[1]:25s} {col[2]:15s} {'NOT NULL' if col[3] else ''}")

# 2. Get all predictions with actuals
query = """
SELECT 
    forecast_date,
    timestamp,
    created_at,
    quantile_coverage,
    actual_vix_change,
    q10, q25, q50, q75, q90
FROM forecasts
WHERE actual_vix_change IS NOT NULL
ORDER BY created_at
"""

df = pd.read_sql_query(query, conn)

print(f"\n[2] TOTAL PREDICTIONS WITH ACTUALS: {len(df)}")

# 3. Analyze quantile_coverage patterns
has_coverage = df["quantile_coverage"].notna()
print(f"\n[3] QUANTILE_COVERAGE BREAKDOWN:")
print(f"   Has coverage:     {has_coverage.sum()} ({has_coverage.sum()/len(df)*100:.1f}%)")
print(f"   Missing coverage: {(~has_coverage).sum()} ({(~has_coverage).sum()/len(df)*100:.1f}%)")

# 4. Check if there's a time pattern
df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
df['has_coverage'] = df['quantile_coverage'].notna()

print(f"\n[4] TIME PATTERN ANALYSIS:")
print(f"   Earliest forecast: {df['created_at_parsed'].min()}")
print(f"   Latest forecast:   {df['created_at_parsed'].max()}")

# Group by month
df['month'] = df['created_at_parsed'].dt.to_period('M')
coverage_by_month = df.groupby('month')['has_coverage'].agg(['sum', 'count'])
print(f"\n   Coverage by month:")
for month, row in coverage_by_month.iterrows():
    pct = row['sum']/row['count']*100
    print(f"   {month}: {row['sum']}/{row['count']} ({pct:.0f}%)")

# 5. Sample some records
print(f"\n[5] SAMPLE RECORDS (with coverage):")
with_coverage = df[df['has_coverage']].head(3)
for idx, row in with_coverage.iterrows():
    print(f"\n   Date: {row['forecast_date']}")
    print(f"   Created: {row['created_at']}")
    print(f"   Coverage: {row['quantile_coverage'][:80]}...")

print(f"\n[6] SAMPLE RECORDS (without coverage):")
without_coverage = df[~df['has_coverage']].head(3)
for idx, row in without_coverage.iterrows():
    print(f"\n   Date: {row['forecast_date']}")
    print(f"   Created: {row['created_at']}")
    print(f"   Coverage: {row['quantile_coverage']}")
    print(f"   Has quantiles: q10={row['q10']}, q50={row['q50']}, q90={row['q90']}")

# 7. Check if backfill happened
print(f"\n[7] BACKFILL ANALYSIS:")
query_backfill = """
SELECT 
    COUNT(*) as total,
    COUNT(actual_vix_change) as with_actual,
    COUNT(quantile_coverage) as with_coverage,
    COUNT(CASE WHEN actual_vix_change IS NOT NULL AND quantile_coverage IS NULL THEN 1 END) as backfilled_missing_coverage
FROM forecasts
"""
cursor = conn.execute(query_backfill)
result = cursor.fetchone()
print(f"   Total forecasts: {result[0]}")
print(f"   With actuals: {result[1]}")
print(f"   With coverage: {result[2]}")
print(f"   Backfilled but missing coverage: {result[3]}")

# 8. Check for pattern in who writes coverage
print(f"\n[8] CHECKING FOR STORAGE PATTERN:")
query_check = """
SELECT 
    prediction_id,
    forecast_date,
    timestamp,
    quantile_coverage IS NOT NULL as has_coverage,
    actual_vix_change IS NOT NULL as has_actual
FROM forecasts
WHERE actual_vix_change IS NOT NULL
ORDER BY timestamp
LIMIT 20
"""
pattern_df = pd.read_sql_query(query_check)
print(pattern_df.to_string(index=False))

# 9. Theory check
print(f"\n[9] THEORY: Was coverage added later?")
print("   Checking if all records without coverage are older...")

df_sorted = df.sort_values('created_at_parsed')
first_with_coverage = df_sorted[df_sorted['has_coverage']]['created_at_parsed'].min()
last_without_coverage = df_sorted[~df_sorted['has_coverage']]['created_at_parsed'].max()

print(f"   First WITH coverage:    {first_with_coverage}")
print(f"   Last WITHOUT coverage:  {last_without_coverage}")

if pd.notna(last_without_coverage) and pd.notna(first_with_coverage):
    if last_without_coverage < first_with_coverage:
        print(f"   ✅ CONFIRMED: Coverage was added to the system later!")
        print(f"   📅 Cutoff appears to be around {first_with_coverage.date()}")
    else:
        print(f"   ⚠️  MIXED: Coverage is randomly missing across time periods")

conn.close()

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
