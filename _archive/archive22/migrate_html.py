#!/usr/bin/env python3
"""
Dashboard HTML Migration Script
Updates all HTML files to use new unified JSON structure (live_state.json & historical.json)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


class HTMLMigrator:
    """Migrates HTML dashboard files to new JSON structure"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.changes_made: Dict[str, List[str]] = {}
        
    def migrate_all_files(self, html_dir: Path) -> None:
        """Migrate all HTML files in directory"""
        files_to_migrate = [
            'forward_returns.html',
            'historical_analysis.html',
            'persistence_tracker.html',
            'score_distribution.html'
        ]
        
        for filename in files_to_migrate:
            filepath = html_dir / filename
            if filepath.exists():
                print(f"\n{'[DRY RUN] ' if self.dry_run else ''}Migrating {filename}...")
                self.migrate_file(filepath)
            else:
                print(f"⚠️  Warning: {filename} not found at {filepath}")
        
        self.print_summary()
    
    def migrate_file(self, filepath: Path) -> None:
        """Migrate a single HTML file"""
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        filename = filepath.name
        
        # Route to appropriate migration function
        if filename == 'forward_returns.html':
            content = self.migrate_forward_returns(content)
        elif filename == 'historical_analysis.html':
            content = self.migrate_historical_analysis(content)
        elif filename == 'persistence_tracker.html':
            content = self.migrate_persistence_tracker(content)
        elif filename == 'score_distribution.html':
            content = self.migrate_score_distribution(content)
        
        # Write changes if not dry run
        if content != original_content:
            if not self.dry_run:
                filepath.write_text(content, encoding='utf-8')
                print(f"✅ Updated {filename}")
            else:
                print(f"🔍 Would update {filename}")
                if filename in self.changes_made:
                    for change in self.changes_made[filename]:
                        print(f"   - {change}")
        else:
            print(f"ℹ️  No changes needed for {filename}")
    
    def log_change(self, filename: str, change: str) -> None:
        """Log a change for summary"""
        if filename not in self.changes_made:
            self.changes_made[filename] = []
        self.changes_made[filename].append(change)
    
    def migrate_forward_returns(self, content: str) -> str:
        """Migrate forward_returns.html to use historical.json"""
        filename = 'forward_returns.html'
        
        # 1. Update fetch calls to use historical.json
        old_fetch = r"fetch\(`\$\{path\}\?t=\$\{Date\.now\(\)\}`\)\.then\(r => r\.json\(\)\)"
        
        # Replace the Promise.all fetch pattern
        old_pattern = r"const \[anomaly, historical\] = await Promise\.all\(\[\s*" \
                     r"fetch\(`\$\{path\}anomaly_report\.json\?t=\$\{Date\.now\(\)\}`\)\.then\(r => r\.json\(\)\),\s*" \
                     r"fetch\(`\$\{path\}historical_anomaly_scores\.json\?t=\$\{Date\.now\(\)\}`\)\.then\(r => r\.json\(\)\)\s*" \
                     r"\]\);"
        
        new_pattern = r"const [liveState, historical] = await Promise.all([\n" \
                     r"                    fetch(`${path}live_state.json?t=${Date.now()}`).then(r => r.json()),\n" \
                     r"                    fetch(`${path}historical.json?t=${Date.now()}`).then(r => r.json())\n" \
                     r"                ]);"
        
        if re.search(old_pattern, content, re.DOTALL):
            content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)
            self.log_change(filename, "Updated fetch calls to use live_state.json and historical.json")
        
        # 2. Update data access patterns
        # Current score from live_state
        content = re.sub(
            r"const currentScore = anomaly\.ensemble\?\.score \|\| 0;",
            r"const currentScore = liveState?.anomaly?.ensemble_score || 0;",
            content
        )
        self.log_change(filename, "Updated ensemble score path: anomaly.ensemble.score -> liveState.anomaly.ensemble_score")
        
        # Historical scores access
        content = re.sub(
            r"const ensembleScores = historical\.ensemble_scores \|\| \[\];",
            r"const ensembleScores = historical?.historical?.ensemble_scores || [];",
            content
        )
        content = re.sub(
            r"const forwardReturns = historical\.spx_forward_10d \|\| \[\];",
            r"const forwardReturns = historical?.historical?.spx_forward_10d || [];",
            content
        )
        self.log_change(filename, "Updated historical data paths to use historical.historical.* structure")
        
        # 3. Update thresholds access in renderForwardReturns
        # This is for the sample data fallback
        content = re.sub(
            r"ensemble: \{ score: 0\.563 \},\s*classification: \{[^}]*thresholds: \{ moderate: 0\.725, high: 0\.805, critical: 0\.914 \}",
            r"anomaly: { ensemble_score: 0.563 }",
            content
        )
        
        # Add historical thresholds to sample data
        content = re.sub(
            r"(historical: \{\s*ensemble_scores: generateSampleScores\(\)\s*\})",
            r"historical: {\n                    historical: {\n                        ensemble_scores: generateSampleScores(),\n                        thresholds: { moderate: 0.725, high: 0.805, critical: 0.914 }\n                    }\n                }",
            content
        )
        
        return content
    
    def migrate_historical_analysis(self, content: str) -> str:
        """Migrate historical_analysis.html to use historical.json"""
        filename = 'historical_analysis.html'
        
        # 1. Update fetch calls
        old_pattern = r"const \[anomaly, historical\] = await Promise\.all\(\[\s*" \
                     r"fetch\(`\.\./\.\./json_data/anomaly_report\.json\?t=\$\{timestamp\}`\)\.then\(r => r\.json\(\)\),\s*" \
                     r"fetch\(`\.\./\.\./json_data/historical_anomaly_scores\.json\?t=\$\{timestamp\}`\)\.then\(r => r\.json\(\)\)\s*" \
                     r"\]\);"
        
        new_pattern = r"const [liveState, historical] = await Promise.all([\n" \
                     r"                    fetch(`../../json_data/live_state.json?t=${timestamp}`).then(r => r.json()),\n" \
                     r"                    fetch(`../../json_data/historical.json?t=${timestamp}`).then(r => r.json())\n" \
                     r"                ]);"
        
        if re.search(old_pattern, content, re.DOTALL):
            content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)
            self.log_change(filename, "Updated fetch calls to use live_state.json and historical.json")
        
        # 2. Update data validation
        content = re.sub(
            r"if \(!historical \|\| !historical\.dates \|\| !historical\.ensemble_scores \|\| !historical\.spx_close\)",
            r"if (!historical?.historical || !historical.historical.dates || !historical.historical.ensemble_scores || !historical.historical.spx_close)",
            content
        )
        self.log_change(filename, "Updated data validation to check historical.historical.*")
        
        # 3. Update thresholds extraction
        old_threshold_pattern = r"if \(anomaly\.classification\?\.statistical\?\.thresholds\) \{\s*" \
                               r"statisticalThresholds = anomaly\.classification\.statistical\.thresholds;"
        new_threshold_pattern = r"if (historical?.historical?.thresholds) {\n" \
                               r"                    statisticalThresholds = historical.historical.thresholds;"
        
        content = re.sub(old_threshold_pattern, new_threshold_pattern, content)
        self.log_change(filename, "Updated threshold extraction: anomaly.classification.statistical.thresholds -> historical.historical.thresholds")
        
        # 4. Update filterData function
        content = re.sub(
            r"const \{ dates, ensemble_scores, spx_close \} = fullData;",
            r"const { dates, ensemble_scores, spx_close } = fullData.historical;",
            content
        )
        self.log_change(filename, "Updated filterData to access fullData.historical.*")
        
        return content
    
    def migrate_persistence_tracker(self, content: str) -> str:
        """Migrate persistence_tracker.html to use live_state.json"""
        filename = 'persistence_tracker.html'
        
        # 1. Update fetch calls
        old_pattern = r"const \[anomaly, historical\] = await Promise\.all\(\[\s*" \
                     r"fetch\(`\.\./\.\./json_data/anomaly_report\.json\?t=\$\{timestamp\}`\)\.then\(r => r\.json\(\)\),\s*" \
                     r"fetch\(`\.\./\.\./json_data/historical_anomaly_scores\.json\?t=\$\{timestamp\}`\)\.then\(r => r\.json\(\)\)\s*" \
                     r"\]\);"
        
        new_pattern = r"const [liveState, historical] = await Promise.all([\n" \
                     r"                    fetch(`../../json_data/live_state.json?t=${timestamp}`).then(r => r.json()),\n" \
                     r"                    fetch(`../../json_data/historical.json?t=${timestamp}`).then(r => r.json())\n" \
                     r"                ]);"
        
        if re.search(old_pattern, content, re.DOTALL):
            content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)
            self.log_change(filename, "Updated fetch calls to use live_state.json and historical.json")
        
        # 2. Update thresholds extraction
        old_threshold = r"if \(anomaly\.classification\?\.statistical\?\.thresholds\) \{\s*" \
                       r"statisticalThresholds = anomaly\.classification\.statistical\.thresholds;"
        new_threshold = r"if (historical?.historical?.thresholds) {\n" \
                       r"                    statisticalThresholds = historical.historical.thresholds;"
        
        content = re.sub(old_threshold, new_threshold, content)
        self.log_change(filename, "Updated threshold extraction to use historical.historical.thresholds")
        
        # 3. Update renderPersistence function signature and data access
        content = re.sub(
            r"function renderPersistence\(anomaly, historical\) \{",
            r"function renderPersistence(liveState, historical) {",
            content
        )
        
        # Update persistence data access
        content = re.sub(
            r"const persistence = anomaly\.persistence;",
            r"const persistence = liveState.persistence;",
            content
        )
        content = re.sub(
            r"const historicalStats = persistence\.historical_stats;",
            r"const historicalStats = persistence;  // Already at the right level",
            content
        )
        
        # Update stats display
        content = re.sub(
            r"historicalStats\.current_streak",
            r"persistence.current_streak",
            content
        )
        content = re.sub(
            r"historicalStats\.mean_duration",
            r"persistence.mean_duration",
            content
        )
        content = re.sub(
            r"historicalStats\.max_duration",
            r"persistence.max_duration",
            content
        )
        content = re.sub(
            r"historicalStats\.anomaly_rate",
            r"persistence.anomaly_rate",
            content
        )
        content = re.sub(
            r"historicalStats\.total_anomaly_days",
            r"persistence.total_anomaly_days",
            content
        )
        
        self.log_change(filename, "Updated persistence data access paths")
        
        # 4. Update historical scores access
        content = re.sub(
            r"const scores = historical\.ensemble_scores\.slice\(-30\);",
            r"const scores = historical.historical.ensemble_scores.slice(-30);",
            content
        )
        content = re.sub(
            r"const dates = historical\.dates\.slice\(-30\);",
            r"const dates = historical.historical.dates.slice(-30);",
            content
        )
        self.log_change(filename, "Updated historical data access to use historical.historical.*")
        
        # 5. Update function call
        content = re.sub(
            r"renderPersistence\(anomaly, historical\);",
            r"renderPersistence(liveState, historical);",
            content
        )
        
        return content
    
    def migrate_score_distribution(self, content: str) -> str:
        """Migrate score_distribution.html to use live_state.json and historical.json"""
        filename = 'score_distribution.html'
        
        # 1. Update fetch calls
        old_pattern = r"const \[anomaly, historical\] = await Promise\.all\(\[\s*" \
                     r"fetch\(`\.\./\.\./json_data/anomaly_report\.json\?t=\$\{timestamp\}`\)\.then\(r => r\.json\(\)\),\s*" \
                     r"fetch\(`\.\./\.\./json_data/historical_anomaly_scores\.json\?t=\$\{timestamp\}`\)\.then\(r => r\.json\(\)\)\s*" \
                     r"\]\);"
        
        new_pattern = r"const [liveState, historical] = await Promise.all([\n" \
                     r"                    fetch(`../../json_data/live_state.json?t=${timestamp}`).then(r => r.json()),\n" \
                     r"                    fetch(`../../json_data/historical.json?t=${timestamp}`).then(r => r.json())\n" \
                     r"                ]);"
        
        if re.search(old_pattern, content, re.DOTALL):
            content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)
            self.log_change(filename, "Updated fetch calls to use live_state.json and historical.json")
        
        # 2. Update currentData assignment
        content = re.sub(
            r"currentData = \{ anomaly, historical \};",
            r"currentData = { liveState, historical };",
            content
        )
        
        # 3. Update fallback data structure
        old_fallback = r"currentData = \{[\s\S]*?anomaly: \{[\s\S]*?ensemble: \{ score: 0\.563 \}[\s\S]*?\},[\s\S]*?historical: \{[\s\S]*?ensemble_scores: generateSampleScores\(\)[\s\S]*?\}[\s\S]*?\};"
        
        new_fallback = r"""currentData = {
                    liveState: {
                        anomaly: { 
                            ensemble_score: 0.563
                        }
                    },
                    historical: {
                        historical: {
                            ensemble_scores: generateSampleScores(),
                            thresholds: { moderate: 0.725, high: 0.805, critical: 0.914 }
                        }
                    }
                };"""
        
        content = re.sub(old_fallback, new_fallback, content, flags=re.DOTALL)
        self.log_change(filename, "Updated fallback data structure")
        
        # 4. Update renderDistribution function
        content = re.sub(
            r"const \{ anomaly, historical \} = currentData;",
            r"const { liveState, historical } = currentData;",
            content
        )
        
        # 5. Update data access
        content = re.sub(
            r"const scores = historical\.ensemble_scores;",
            r"const scores = historical.historical.ensemble_scores;",
            content
        )
        content = re.sub(
            r"const currentScore = anomaly\.ensemble\.score;",
            r"const currentScore = liveState.anomaly.ensemble_score;",
            content
        )
        
        # 6. Update thresholds access
        content = re.sub(
            r"const thresholds = anomaly\.classification\?\.statistical\?\.thresholds \|\| \s*" \
            r"\{ moderate: 0\.725, high: 0\.805, critical: 0\.914 \};",
            r"const thresholds = historical.historical?.thresholds || \n" \
            r"                { moderate: 0.725, high: 0.805, critical: 0.914 };",
            content
        )
        self.log_change(filename, "Updated thresholds access to use historical.historical.thresholds")
        
        return content
    
    def print_summary(self) -> None:
        """Print summary of all changes"""
        print("\n" + "="*70)
        print("MIGRATION SUMMARY")
        print("="*70)
        
        if not self.changes_made:
            print("✅ No changes needed - all files already migrated")
            return
        
        for filename, changes in self.changes_made.items():
            print(f"\n📄 {filename}:")
            for change in changes:
                print(f"   ✓ {change}")
        
        print("\n" + "="*70)
        if self.dry_run:
            print("🔍 DRY RUN COMPLETE - No files were modified")
            print("   Run without --dry-run to apply changes")
        else:
            print("✅ MIGRATION COMPLETE")
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Migrate HTML dashboard files to new JSON structure'
    )
    parser.add_argument(
        'html_dir',
        type=Path,
        help='Directory containing HTML files to migrate'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    
    args = parser.parse_args()
    
    if not args.html_dir.exists():
        print(f"❌ Error: Directory not found: {args.html_dir}")
        return 1
    
    print("="*70)
    print("HTML DASHBOARD MIGRATION SCRIPT")
    print("="*70)
    print(f"Directory: {args.html_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("="*70)
    
    migrator = HTMLMigrator(dry_run=args.dry_run)
    migrator.migrate_all_files(args.html_dir)
    
    return 0


# HELPER: Fix integrated_system_production.py exporter calls
def generate_exporter_fix():
    """Generate the correct exporter code for integrated_system_production.py"""
    code = '''
# CORRECT EXPORTER USAGE:

from unified_exporter import UnifiedExporter

# In main() function, replace the export section with:
anomaly_result = system._get_cached_anomaly_result()

if anomaly_result and system.vix_predictor.anomaly_detector:
    # Initialize exporter
    exporter = UnifiedExporter(output_dir='./json_data')
    
    # Export new unified files (NO filepath argument - uses output_dir from init)
    exporter.export_live_state(
        vix_predictor=system.vix_predictor,
        anomaly_result=anomaly_result
    )
    
    exporter.export_historical_context(
        vix_predictor=system.vix_predictor
    )
    
    exporter.export_model_cache(
        vix_predictor=system.vix_predictor
    )
    
    print("\\n✅ Exported unified dashboard files:")
    print("   • live_state.json")
    print("   • historical.json")
    print("   • model_cache.pkl")
    
    # OPTIONAL: Keep old exports during transition (can remove after testing)
    # system.vix_predictor.export_anomaly_report(
    #     filepath='./json_data/anomaly_report.json',
    #     anomaly_result=anomaly_result
    # )
'''
    return code


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--show-fix':
        print(generate_exporter_fix())
    else:
        exit(main())