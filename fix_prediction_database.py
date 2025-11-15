#!/usr/bin/env python3
"""
Fix prediction_database.py schema to match V3 code expectations.

This script will update the _create_schema method to include:
- median_forecast column  
- prob_up and prob_down columns (replacing old regime probabilities)
"""

import re
from pathlib import Path
import shutil


UPDATED_CREATE_TABLE = '''            create_sql = """
            CREATE TABLE forecasts (
                prediction_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                observation_date DATE NOT NULL,
                forecast_date DATE NOT NULL,
                horizon INTEGER NOT NULL,

                -- Context
                calendar_cohort TEXT,
                cohort_weight REAL,

                -- Predictions (V3 UPDATED)
                median_forecast REAL NOT NULL,
                point_estimate REAL NOT NULL,
                q10 REAL,
                q25 REAL,
                q50 REAL,
                q75 REAL,
                q90 REAL,
                
                -- Direction (V3 SIMPLIFIED)
                prob_up REAL,
                prob_down REAL,
                direction_probability REAL,
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
            """'''


def fix_schema(file_path="src/core/prediction_database.py"):
    """Update the database schema in prediction_database.py"""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Backup
    backup_path = file_path.with_suffix('.py.bak2')
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Find the CREATE TABLE statement
    # It starts with 'create_sql = """' and ends with '"""'
    pattern = r'create_sql = """.*?"""'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("❌ Could not find CREATE TABLE statement")
        return False
    
    # Replace with updated schema
    new_content = content[:match.start()] + UPDATED_CREATE_TABLE + content[match.end():]
    
    # Write updated file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Schema updated in {file_path}")
    print("\nChanges made:")
    print("  • Added: median_forecast REAL NOT NULL")
    print("  • Added: prob_up REAL")
    print("  • Added: prob_down REAL")
    print("  • Removed: prob_low, prob_normal, prob_elevated, prob_crisis")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SCHEMA FIX FOR PREDICTION_DATABASE.PY")
    print("="*80)
    
    success = fix_schema()
    
    if success:
        print("\n" + "="*80)
        print("✅ FIX COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Delete old database:")
        print("     rm data_cache/predictions.db")
        print("  2. Run your system:")
        print("     python integrated_system_production.py --mode complete")
    else:
        print("\n❌ Fix failed - apply manually using schema_fix_v3.sql")
