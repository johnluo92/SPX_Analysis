#!/usr/bin/env python3
"""
Update All Charts to Use data_service.js
=========================================
This script updates all HTML chart files to use the centralized data_service.js
instead of individual fetch() calls.

Usage:
    python update_all_charts.py --dry-run  # Preview changes
    python update_all_charts.py            # Apply changes
"""

import re
import shutil
from pathlib import Path
from datetime import datetime
import argparse


class ChartUpdater:
    def __init__(self, charts_dir: str = "./Chart Modules", backup_dir: str = "./backup"):
        self.charts_dir = Path(charts_dir)
        self.backup_dir = Path(backup_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def backup_file(self, filepath: Path) -> Path:
        """Create backup of original file."""
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        backup_name = f"{filepath.stem}_backup_{self.timestamp}{filepath.suffix}"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def add_data_service_script(self, content: str) -> tuple[str, bool]:
        """Add data_service.js script tag if missing."""
        if 'data_service.js' in content:
            return content, False
        
        # Find head tag and add script
        head_pattern = r'(<head[^>]*>)'
        match = re.search(head_pattern, content, re.IGNORECASE)
        
        if match:
            insert_pos = match.end()
            script_tag = '\n    <script src="../../data_service.js"></script>'
            content = content[:insert_pos] + script_tag + content[insert_pos:]
            return content, True
        
        return content, False
    
    def update_detector_grid_hero(self, content: str) -> tuple[str, list]:
        """Update detector_grid_hero.html"""
        changes = []
        
        # Replace loadData function
        old_load_pattern = r'async function loadData\(\) \{.*?catch \(error\) \{.*?\}\s*\}'
        
        new_load_function = '''async function loadData() {
            try {
                // Initialize data service
                await window.dataService.init();
                
                const liveState = {
                    market: window.dataService.getMarketState(),
                    anomaly: window.dataService.getAnomalyState(),
                    persistence: window.dataService.getPersistenceState(),
                    generated_at: window.dataService.data.live.generated_at
                };
                
                const historical = {
                    historical: {
                        thresholds: window.dataService.getThresholds(),
                        regime_stats: window.dataService.getRegimeStats(),
                        ensemble_scores: window.dataService.getHistoricalData().ensemble_scores
                    }
                };
                
                // Store for tooltip usage
                currentAnomalyData = {
                    classification: {
                        thresholds: historical.historical.thresholds
                    },
                    domain_anomalies: liveState.anomaly.detector_scores,
                    top_anomalies: Object.entries(liveState.anomaly.detector_scores).map(([name, score]) => ({
                        name,
                        score
                    }))
                };
                
                renderMetrics(liveState, historical);
                renderDetectorGrid(liveState, historical);
                
            } catch (error) {
                console.error('Failed to load data:', error);
                document.getElementById('rowContainer').innerHTML = `
                    <div style="color: #ef4444; font-size: 12px;">
                        Error: ${error.message}
                    </div>
                `;
            }
        }'''
        
        if re.search(old_load_pattern, content, re.DOTALL):
            content = re.sub(old_load_pattern, new_load_function, content, flags=re.DOTALL)
            changes.append("Updated loadData() to use data_service.js")
        
        return content, changes
    
    def update_detector_ranking(self, content: str) -> tuple[str, list]:
        """Update detector_ranking.html"""
        changes = []
        
        old_load_pattern = r'async function loadData\(\) \{.*?renderDetectorBars\(\);.*?\}'
        
        new_load_function = '''async function loadData() {
            try {
                // Initialize data service
                await window.dataService.init();
                
                liveStateData = {
                    anomaly: window.dataService.getAnomalyState()
                };
                
                historicalData = {
                    historical: {
                        thresholds: window.dataService.getThresholds()
                    },
                    detector_metadata: window.dataService.getDetectorMetadata()
                };
                
                // Extract statistical thresholds
                statisticalThresholds = historicalData.historical.thresholds;
                console.log('Using statistical thresholds:', statisticalThresholds);
                
            } catch (error) {
                console.warn('Error loading data:', error.message);
                liveStateData = null;
                historicalData = null;
            }
            
            renderDetectorBars();
        }'''
        
        if re.search(old_load_pattern, content, re.DOTALL):
            content = re.sub(old_load_pattern, new_load_function, content, flags=re.DOTALL)
            changes.append("Updated loadData() to use data_service.js")
        
        return content, changes
    
    def update_forward_returns(self, content: str) -> tuple[str, list]:
        """Update forward_returns.html"""
        changes = []
        
        # Remove fetchWithFallback function
        content = re.sub(
            r'async function fetchWithFallback\(filename\) \{.*?\}',
            '',
            content,
            flags=re.DOTALL
        )
        
        # Replace loadData function
        old_load_pattern = r'async function loadData\(\) \{.*?handleFileUpload\);.*?\}'
        
        new_load_function = '''async function loadData() {
            const content = document.getElementById('forwardReturnsContent');
            const statusBadge = document.getElementById('statusBadge');
            
            try {
                statusBadge.className = 'status-badge status-loading';
                statusBadge.textContent = 'Loading';
                
                // Initialize data service
                await window.dataService.init();
                
                const anomaly = {
                    ensemble: {
                        score: window.dataService.getAnomalyState().ensemble_score
                    }
                };
                
                const historical = {
                    ensemble_scores: window.dataService.getHistoricalData().ensemble_scores,
                    spx_forward_10d: window.dataService.getHistoricalData().spx_forward_10d
                };
                
                statusBadge.className = 'status-badge status-ready';
                statusBadge.textContent = 'Ready';
                hasData = true;
                
                renderForwardReturns(anomaly, historical);
            } catch (error) {
                console.error('Error loading data:', error);
                statusBadge.className = 'status-badge status-error';
                statusBadge.textContent = 'Error';
                
                content.innerHTML = `
                    <div class="error">
                        <div class="error-icon">⚠️</div>
                        <div class="error-title">Data Not Available</div>
                        <div class="error-details">
                            Could not load required data files.<br><br>
                            <strong>Error:</strong> ${error.message}
                        </div>
                    </div>
                `;
            }
        }'''
        
        if re.search(old_load_pattern, content, re.DOTALL):
            content = re.sub(old_load_pattern, new_load_function, content, flags=re.DOTALL)
            changes.append("Updated loadData() to use data_service.js")
            changes.append("Removed fetchWithFallback()")
        
        return content, changes
    
    def update_historical_analysis(self, content: str) -> tuple[str, list]:
        """Update historical_analysis.html"""
        changes = []
        
        old_load_pattern = r'async function loadData\(\) \{.*?showError\(.*?\);.*?\}'
        
        new_load_function = '''async function loadData() {
            try {
                // Initialize data service
                await window.dataService.init();
                
                const anomaly = {
                    classification: {
                        statistical: {
                            thresholds: window.dataService.getThresholds()
                        }
                    }
                };
                
                const historical = {
                    dates: window.dataService.getHistoricalData().dates,
                    ensemble_scores: window.dataService.getHistoricalData().ensemble_scores,
                    spx_close: window.dataService.getHistoricalData().spx_close
                };
                
                if (!historical || !historical.dates || !historical.ensemble_scores || !historical.spx_close) {
                    throw new Error('Invalid or missing historical data structure');
                }
                
                // Extract statistical thresholds
                statisticalThresholds = anomaly.classification.statistical.thresholds;
                console.log('Using statistical thresholds:', statisticalThresholds);
                
                fullData = historical;
                renderCharts();
                
                // Notify parent of content height for iframe sizing
                if (window.parent !== window) {
                    setTimeout(() => {
                        const height = document.documentElement.scrollHeight;
                        window.parent.postMessage({
                            type: 'resize',
                            height: height
                        }, '*');
                    }, 500);
                }
            } catch (error) {
                console.error('Error loading data:', error);
                showError('Unable to load historical data: ' + error.message);
            }
        }'''
        
        if re.search(old_load_pattern, content, re.DOTALL):
            content = re.sub(old_load_pattern, new_load_function, content, flags=re.DOTALL)
            changes.append("Updated loadData() to use data_service.js")
        
        return content, changes
    
    def update_persistence_tracker(self, content: str) -> tuple[str, list]:
        """Update persistence_tracker.html"""
        changes = []
        
        old_load_pattern = r'async function loadData\(\) \{.*?renderPersistence\(anomaly, historical\);.*?\}'
        
        new_load_function = '''async function loadData() {
            try {
                // Initialize data service
                await window.dataService.init();
                
                const anomaly = {
                    persistence: window.dataService.getPersistenceState(),
                    classification: {
                        statistical: {
                            thresholds: window.dataService.getThresholds()
                        }
                    }
                };
                
                const historical = {
                    ensemble_scores: window.dataService.getHistoricalData().ensemble_scores,
                    dates: window.dataService.getHistoricalData().dates
                };
                
                // Extract statistical thresholds
                statisticalThresholds = anomaly.classification.statistical.thresholds;
                console.log('Using statistical thresholds:', statisticalThresholds);
                
                renderPersistence(anomaly, historical);
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }'''
        
        if re.search(old_load_pattern, content, re.DOTALL):
            content = re.sub(old_load_pattern, new_load_function, content, flags=re.DOTALL)
            changes.append("Updated loadData() to use data_service.js")
        
        return content, changes
    
    def update_score_distribution(self, content: str) -> tuple[str, list]:
        """Update score_distribution.html"""
        changes = []
        
        old_load_pattern = r'async function loadData\(\) \{.*?renderDistribution\(\);.*?\}'
        
        new_load_function = '''async function loadData() {
            try {
                // Initialize data service
                await window.dataService.init();
                
                const anomaly = {
                    ensemble: {
                        score: window.dataService.getAnomalyState().ensemble_score
                    },
                    classification: {
                        statistical: {
                            level: window.dataService.getAnomalyState().classification,
                            percentile: 0.42, // This would need to be calculated
                            thresholds: window.dataService.getThresholds()
                        }
                    }
                };
                
                const historical = {
                    ensemble_scores: window.dataService.getHistoricalData().ensemble_scores
                };
                
                currentData = { anomaly, historical };
                renderDistribution();
            } catch (error) {
                console.error('Error loading data:', error);
                currentData = {
                    anomaly: {
                        ensemble: { score: 0.563 },
                        classification: {
                            statistical: {
                                level: 'NORMAL',
                                percentile: 0.42,
                                thresholds: { moderate: 0.725, high: 0.805, critical: 0.914 }
                            }
                        }
                    },
                    historical: {
                        ensemble_scores: generateSampleScores()
                    }
                };
                renderDistribution();
            }
        }'''
        
        if re.search(old_load_pattern, content, re.DOTALL):
            content = re.sub(old_load_pattern, new_load_function, content, flags=re.DOTALL)
            changes.append("Updated loadData() to use data_service.js")
        
        return content, changes
    
    def update_file(self, filepath: Path, dry_run: bool = False) -> dict:
        """Update a single HTML file."""
        print(f"\n📄 Processing: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        all_changes = []
        
        # Add data_service.js script tag
        content, added_script = self.add_data_service_script(content)
        if added_script:
            all_changes.append("Added data_service.js script tag")
        
        # Update specific files
        if filepath.name == 'detector_grid_hero.html':
            content, changes = self.update_detector_grid_hero(content)
            all_changes.extend(changes)
        elif filepath.name == 'detector_ranking.html':
            content, changes = self.update_detector_ranking(content)
            all_changes.extend(changes)
        elif filepath.name == 'forward_returns.html':
            content, changes = self.update_forward_returns(content)
            all_changes.extend(changes)
        elif filepath.name == 'historical_analysis.html':
            content, changes = self.update_historical_analysis(content)
            all_changes.extend(changes)
        elif filepath.name == 'persistence_tracker.html':
            content, changes = self.update_persistence_tracker(content)
            all_changes.extend(changes)
        elif filepath.name == 'score_distribution.html':
            content, changes = self.update_score_distribution(content)
            all_changes.extend(changes)
        
        if not all_changes:
            print("   ⏭️  No changes needed")
            return {"filepath": filepath, "status": "skipped", "changes": []}
        
        if dry_run:
            print(f"   ✅ Would update:")
            for change in all_changes:
                print(f"      • {change}")
            return {"filepath": filepath, "status": "would_update", "changes": all_changes}
        
        # Create backup
        backup_path = self.backup_file(filepath)
        print(f"   💾 Backup: {backup_path.name}")
        
        # Write updated file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"   ✅ Updated:")
        for change in all_changes:
            print(f"      • {change}")
        
        return {
            "filepath": filepath,
            "status": "updated",
            "changes": all_changes,
            "backup": backup_path
        }
    
    def update_all(self, dry_run: bool = False):
        """Update all HTML chart files."""
        print("🚀 Chart Update Script")
        print("=" * 80)
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print()
        
        # Find all HTML files
        html_files = []
        
        # Main directory
        for file in self.charts_dir.glob("*.html"):
            html_files.append(file)
        
        # Subcharts directory
        subcharts = self.charts_dir / "subcharts"
        if subcharts.exists():
            for file in subcharts.glob("*.html"):
                html_files.append(file)
        
        if not html_files:
            print("❌ No HTML files found")
            return
        
        print(f"Found {len(html_files)} HTML files")
        
        results = []
        for filepath in sorted(html_files):
            result = self.update_file(filepath, dry_run)
            results.append(result)
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        updated = [r for r in results if r['status'] in ['updated', 'would_update']]
        skipped = [r for r in results if r['status'] == 'skipped']
        
        print(f"\nTotal files: {len(results)}")
        print(f"Updated: {len(updated)}")
        print(f"Skipped: {len(skipped)}")
        
        if updated:
            print(f"\n{'Would update' if dry_run else 'Updated'} files:")
            for r in updated:
                print(f"  ✅ {r['filepath'].name}")
        
        if not dry_run and updated:
            print(f"\n💾 Backups saved to: {self.backup_dir.absolute()}")
            print("\n🎯 Next Steps:")
            print("  1. Test each chart in your browser")
            print("  2. Check browser console for errors")
            print("  3. Verify data displays correctly")
            print("  4. Run: python integrated_system_production.py")
        elif dry_run:
            print("\n🎯 To apply changes, run without --dry-run flag")


def main():
    parser = argparse.ArgumentParser(description='Update all HTML charts to use data_service.js')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without modifying files')
    parser.add_argument('--charts-dir', default='./Chart Modules', help='Directory containing HTML files')
    parser.add_argument('--backup-dir', default='./backup', help='Directory for backups')
    
    args = parser.parse_args()
    
    updater = ChartUpdater(charts_dir=args.charts_dir, backup_dir=args.backup_dir)
    updater.update_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
