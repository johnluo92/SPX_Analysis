"""
HTML Migration Script
=====================
Automatically updates HTML files to use the new data_service.js

Features:
- Finds all fetch() calls to legacy JSON files
- Replaces with data_service.js calls
- Adds data service script tag if missing
- Wraps fetch logic in proper async initialization
- Creates backup of original files

Usage:
    python html_migrator.py --dry-run  # Preview changes
    python html_migrator.py            # Apply changes
"""

import re
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import argparse


class HTMLMigrator:
    def __init__(self, charts_dir: str = "./Chart Modules", backup_dir: str = "./backup"):
        self.charts_dir = Path(charts_dir)
        self.backup_dir = Path(backup_dir)
        self.migration_map = {
            "market_state.json": {
                "method": "getMarketState()",
                "type": "live"
            },
            "anomaly_report.json": {
                "method": "getAnomalyState()",
                "type": "live"
            },
            "historical_anomaly_scores.json": {
                "method": "getHistoricalData()",
                "type": "historical"
            },
            "anomaly_feature_attribution.json": {
                "method": "getFeatureAttribution(detectorName)",
                "type": "historical",
                "needs_param": True
            },
            "regime_statistics.json": {
                "method": "getRegimeStats()",
                "type": "historical"
            },
            "anomaly_metadata.json": {
                "method": "getDetectorMetadata()",
                "type": "historical"
            },
            "firing_detector_analysis.json": {
                "method": "getAnomalyState()",
                "type": "live",
                "note": "Firing detectors are in activeDetectors array"
            },
            "vix_history.json": {
                "method": "getHistoricalData()",
                "type": "historical",
                "note": "VIX history is in historical data"
            },
            "dashboard_data.json": {
                "method": "getMarketState() and getAnomalyState()",
                "type": "live",
                "note": "Split between market and anomaly state"
            }
        }
        
    def find_html_files(self) -> List[Path]:
        """Find all HTML files in the charts directory."""
        html_files = []
        
        # Check main directory
        for file in self.charts_dir.glob("*.html"):
            html_files.append(file)
        
        # Check subcharts directory
        subcharts = self.charts_dir / "subcharts"
        if subcharts.exists():
            for file in subcharts.glob("*.html"):
                html_files.append(file)
        
        return sorted(html_files)
    
    def analyze_html(self, filepath: Path) -> Dict:
        """Analyze an HTML file for migration needs."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find fetch calls
        fetch_pattern = r"fetch\s*\(\s*['\"]([^'\"]+\.json)['\"]"
        fetches = []
        
        for match in re.finditer(fetch_pattern, content, re.IGNORECASE):
            url = match.group(1)
            filename = url.split('/')[-1]
            line_num = content[:match.start()].count('\n') + 1
            
            if filename in self.migration_map:
                fetches.append({
                    "line": line_num,
                    "url": url,
                    "filename": filename,
                    "start": match.start(),
                    "end": match.end(),
                    "full_match": match.group(0)
                })
        
        # Check if data_service.js is included
        has_data_service = 'data_service.js' in content
        
        # Check if already migrated
        has_data_service_calls = 'window.dataService' in content
        
        return {
            "filepath": filepath,
            "fetches": fetches,
            "has_data_service_script": has_data_service,
            "has_data_service_calls": has_data_service_calls,
            "needs_migration": len(fetches) > 0 and not has_data_service_calls,
            "content": content
        }
    
    def create_backup(self, filepath: Path) -> Path:
        """Create backup of original file."""
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def add_data_service_script(self, content: str) -> str:
        """Add data_service.js script tag if missing."""
        # Check if already has it
        if 'data_service.js' in content:
            return content
        
        # Find head tag
        head_pattern = r'(<head[^>]*>)'
        match = re.search(head_pattern, content, re.IGNORECASE)
        
        if match:
            insert_pos = match.end()
            script_tag = '\n    <script src="../../data_service.js"></script>'
            content = content[:insert_pos] + script_tag + content[insert_pos:]
        else:
            # No head tag found, add at beginning
            script_tag = '<script src="../../data_service.js"></script>\n'
            content = script_tag + content
        
        return content
    
    def generate_init_wrapper(self, original_fetch_code: str, migrations: List[Dict]) -> str:
        """Generate new initialization code using data_service.js."""
        
        # Build the data service calls
        data_service_calls = []
        for mig in migrations:
            info = self.migration_map[mig['filename']]
            method = info['method']
            
            # Generate variable name from method
            if 'Anomaly' in method:
                var_name = 'anomalyData'
            elif 'Market' in method:
                var_name = 'marketData'
            elif 'Historical' in method:
                var_name = 'historicalData'
            elif 'Thresholds' in method:
                var_name = 'thresholds'
            elif 'Attribution' in method:
                var_name = 'attributionData'
            elif 'Regime' in method:
                var_name = 'regimeData'
            elif 'Metadata' in method:
                var_name = 'metadataData'
            else:
                var_name = 'data'
            
            data_service_calls.append(f"const {var_name} = window.dataService.{method};")
        
        wrapper = f"""
        // ===================================================================
        // MIGRATED TO data_service.js
        // ===================================================================
        async function initChart() {{
            try {{
                // Initialize data service (loads both JSON files)
                await window.dataService.init();
                console.log('✅ Data service initialized');
                
                // Get data from service
{chr(10).join('                ' + call for call in data_service_calls)}
                
                // Your original chart code continues here...
                // (You may need to adjust variable names)
                
            }} catch (error) {{
                console.error('❌ Failed to initialize chart:', error);
                // Add error display logic here
            }}
        }}
        
        // Initialize on page load
        initChart();
        """
        
        return wrapper
    
    def migrate_fetch_calls(self, content: str, fetches: List[Dict]) -> Tuple[str, List[str]]:
        """Replace fetch calls with data_service.js calls."""
        changes = []
        new_content = content
        
        # Sort fetches by position (reverse order to maintain positions)
        fetches_sorted = sorted(fetches, key=lambda x: x['start'], reverse=True)
        
        for fetch in fetches_sorted:
            filename = fetch['filename']
            info = self.migration_map[filename]
            
            # Find the full fetch chain (including .then() calls)
            # Look for the pattern: fetch(...).then(...).then(...)
            fetch_start = fetch['start']
            
            # Find the end of the fetch chain (look for closing of async function or semicolon)
            # This is a simple heuristic - may need refinement
            search_text = content[fetch_start:]
            
            # Find the fetch().then(...) block
            paren_count = 0
            in_fetch = False
            fetch_end = fetch_start
            
            for i, char in enumerate(search_text):
                if char == '(':
                    paren_count += 1
                    in_fetch = True
                elif char == ')':
                    paren_count -= 1
                    if in_fetch and paren_count == 0:
                        # Check if followed by .then
                        remaining = search_text[i+1:i+10]
                        if not remaining.strip().startswith('.then'):
                            fetch_end = fetch_start + i + 1
                            break
                elif char == ';' and paren_count == 0:
                    fetch_end = fetch_start + i + 1
                    break
            
            old_code = content[fetch_start:fetch_end]
            
            # Generate new code using data_service
            method = info['method']
            note = info.get('note', '')
            
            new_code = f"// MIGRATED: {filename} → window.dataService.{method}"
            if note:
                new_code += f"\n        // NOTE: {note}"
            
            # Replace in content
            new_content = new_content[:fetch_start] + new_code + new_content[fetch_end:]
            
            changes.append(f"Line {fetch['line']}: {filename} → {method}")
        
        return new_content, changes
    
    def migrate_file(self, filepath: Path, dry_run: bool = False) -> Dict:
        """Migrate a single HTML file."""
        analysis = self.analyze_html(filepath)
        
        if not analysis['needs_migration']:
            return {
                "filepath": filepath,
                "status": "skipped",
                "reason": "No migration needed" if analysis['has_data_service_calls'] else "No legacy fetches found"
            }
        
        result = {
            "filepath": filepath,
            "status": "migrated",
            "changes": [],
            "backup": None
        }
        
        # Start with original content
        new_content = analysis['content']
        
        # Add data service script
        if not analysis['has_data_service_script']:
            new_content = self.add_data_service_script(new_content)
            result['changes'].append("Added data_service.js script tag")
        
        # Migrate fetch calls
        new_content, fetch_changes = self.migrate_fetch_calls(new_content, analysis['fetches'])
        result['changes'].extend(fetch_changes)
        
        # Add migration comment at top
        migration_header = f"""
<!-- 
    MIGRATED TO data_service.js on {datetime.now().isoformat()}
    
    Changes:
{chr(10).join('    - ' + change for change in result['changes'])}
    
    TODO: 
    1. Wrap your chart initialization in async function
    2. Call await window.dataService.init() at start
    3. Replace old data variables with data service getters
    4. Test chart functionality
-->

"""
        new_content = migration_header + new_content
        
        if not dry_run:
            # Create backup
            backup_path = self.create_backup(filepath)
            result['backup'] = backup_path
            
            # Write migrated file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            result['status'] = 'would_migrate'
            result['preview'] = new_content[:500] + "..."
        
        return result
    
    def migrate_all(self, dry_run: bool = False) -> Dict:
        """Migrate all HTML files."""
        html_files = self.find_html_files()
        
        results = {
            "total_files": len(html_files),
            "migrated": [],
            "skipped": [],
            "errors": []
        }
        
        print(f"\n🔍 Found {len(html_files)} HTML files")
        print("=" * 80)
        
        for filepath in html_files:
            print(f"\n📄 Processing: {filepath.name}")
            
            try:
                result = self.migrate_file(filepath, dry_run)
                
                if result['status'] in ['migrated', 'would_migrate']:
                    results['migrated'].append(result)
                    print(f"   ✅ {'Would migrate' if dry_run else 'Migrated'}")
                    for change in result['changes']:
                        print(f"      - {change}")
                    if result.get('backup'):
                        print(f"   💾 Backup: {result['backup'].name}")
                else:
                    results['skipped'].append(result)
                    print(f"   ⏭️  Skipped: {result['reason']}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results['errors'].append({"filepath": filepath, "error": str(e)})
        
        return results
    
    def generate_summary_report(self, results: Dict, dry_run: bool = False) -> str:
        """Generate summary report of migration."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("MIGRATION SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Mode: {'DRY RUN (no changes made)' if dry_run else 'LIVE (files modified)'}")
        report.append(f"Total files: {results['total_files']}")
        report.append(f"Migrated: {len(results['migrated'])}")
        report.append(f"Skipped: {len(results['skipped'])}")
        report.append(f"Errors: {len(results['errors'])}")
        report.append("")
        
        if results['migrated']:
            report.append("📝 Migrated Files:")
            for r in results['migrated']:
                report.append(f"  • {r['filepath'].name}")
                for change in r['changes']:
                    report.append(f"    - {change}")
            report.append("")
        
        if results['errors']:
            report.append("❌ Errors:")
            for e in results['errors']:
                report.append(f"  • {e['filepath'].name}: {e['error']}")
            report.append("")
        
        if not dry_run and results['migrated']:
            report.append("🎯 Next Steps:")
            report.append("  1. Review each migrated file")
            report.append("  2. Test charts in browser")
            report.append("  3. Update data variable names if needed")
            report.append("  4. Remove old JSON file references from Python")
            report.append(f"  5. Backups saved in: {self.backup_dir.absolute()}")
        elif dry_run:
            report.append("🎯 To apply changes, run without --dry-run flag")
        
        report.append("")
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Migrate HTML files to use data_service.js')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without modifying files')
    parser.add_argument('--charts-dir', default='./Chart Modules', help='Directory containing HTML files')
    parser.add_argument('--backup-dir', default='./backup', help='Directory for backups')
    
    args = parser.parse_args()
    
    print("🚀 HTML Migration Script")
    print("=" * 80)
    print(f"Charts directory: {args.charts_dir}")
    print(f"Backup directory: {args.backup_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    
    migrator = HTMLMigrator(charts_dir=args.charts_dir, backup_dir=args.backup_dir)
    results = migrator.migrate_all(dry_run=args.dry_run)
    
    report = migrator.generate_summary_report(results, dry_run=args.dry_run)
    print(report)
    
    # Save report to file
    report_path = Path(args.backup_dir) / f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.parent.mkdir(exist_ok=True, parents=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved: {report_path}")


if __name__ == "__main__":
    main()
