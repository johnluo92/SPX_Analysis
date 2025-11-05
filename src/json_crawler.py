"""
JSON Structure Crawler
======================
Analyzes live_state.json and historical.json to generate:
1. Complete data structure map
2. Migration guide for HTML files
3. Data access patterns documentation

Usage:
    python json_crawler.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime


class JSONCrawler:
    def __init__(self, json_dir: str = "./json_data"):
        self.json_dir = Path(json_dir)
        self.structures = {}
        self.access_patterns = {}
        
    def crawl_structure(self, obj: Any, path: str = "", max_depth: int = 10) -> Dict:
        """Recursively crawl JSON structure and build schema."""
        if max_depth == 0:
            return {"type": "max_depth_reached"}
        
        if obj is None:
            return {"type": "null"}
        elif isinstance(obj, bool):
            return {"type": "boolean", "value": obj}
        elif isinstance(obj, int):
            return {"type": "integer", "sample": obj}
        elif isinstance(obj, float):
            return {"type": "float", "sample": round(obj, 4)}
        elif isinstance(obj, str):
            return {"type": "string", "sample": obj[:50], "length": len(obj)}
        elif isinstance(obj, list):
            if len(obj) == 0:
                return {"type": "array", "length": 0, "items": None}
            
            # Sample first few items to understand array structure
            samples = [self.crawl_structure(obj[i], f"{path}[{i}]", max_depth-1) 
                      for i in range(min(3, len(obj)))]
            
            # Check if all items have same structure
            if len(set(json.dumps(s, sort_keys=True) for s in samples)) == 1:
                item_schema = samples[0]
            else:
                item_schema = {"varied": samples}
            
            return {
                "type": "array",
                "length": len(obj),
                "items": item_schema
            }
        elif isinstance(obj, dict):
            schema = {"type": "object", "properties": {}}
            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else key
                schema["properties"][key] = self.crawl_structure(value, child_path, max_depth-1)
            return schema
        else:
            return {"type": str(type(obj).__name__)}
    
    def analyze_file(self, filename: str) -> Dict:
        """Analyze a single JSON file."""
        filepath = self.json_dir / filename
        
        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            file_stats = os.stat(filepath)
            
            return {
                "filename": filename,
                "size_kb": round(file_stats.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "structure": self.crawl_structure(data),
                "raw_keys": list(data.keys()) if isinstance(data, dict) else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def generate_access_patterns(self) -> Dict:
        """Generate common data access patterns from structure analysis."""
        patterns = {
            "live_state.json": {
                "description": "Real-time market and anomaly state (updates every 15min)",
                "patterns": [
                    {
                        "name": "Current Anomaly Score",
                        "path": "live.anomaly.ensemble_score",
                        "js_access": "window.dataService.getAnomalyState().score",
                        "type": "float"
                    },
                    {
                        "name": "Anomaly Classification",
                        "path": "live.anomaly.classification",
                        "js_access": "window.dataService.getAnomalyState().classification",
                        "type": "string"
                    },
                    {
                        "name": "Active Detectors",
                        "path": "live.anomaly.active_detectors",
                        "js_access": "window.dataService.getAnomalyState().activeDetectors",
                        "type": "array"
                    },
                    {
                        "name": "Current VIX",
                        "path": "live.market.vix_close",
                        "js_access": "window.dataService.getMarketState().vix_close",
                        "type": "float"
                    },
                    {
                        "name": "SPX Close",
                        "path": "live.market.spx_close",
                        "js_access": "window.dataService.getMarketState().spx_close",
                        "type": "float"
                    },
                    {
                        "name": "Persistence Streak",
                        "path": "live.persistence.current_streak",
                        "js_access": "window.dataService.getAnomalyState().persistence.current_streak",
                        "type": "integer"
                    }
                ]
            },
            "historical.json": {
                "description": "Historical training data and statistics (static)",
                "patterns": [
                    {
                        "name": "Historical Dates",
                        "path": "historical.historical.dates",
                        "js_access": "window.dataService.getHistoricalData().dates",
                        "type": "array[string]"
                    },
                    {
                        "name": "Historical Scores",
                        "path": "historical.historical.ensemble_scores",
                        "js_access": "window.dataService.getHistoricalData().scores",
                        "type": "array[float]"
                    },
                    {
                        "name": "SPX Historical",
                        "path": "historical.historical.spx_close",
                        "js_access": "window.dataService.getHistoricalData().spx",
                        "type": "array[float]"
                    },
                    {
                        "name": "Forward Returns",
                        "path": "historical.historical.spx_forward_10d",
                        "js_access": "window.dataService.getHistoricalData().forwardReturns",
                        "type": "array[float]"
                    },
                    {
                        "name": "Anomaly Thresholds",
                        "path": "historical.historical.thresholds",
                        "js_access": "window.dataService.getThresholds().base",
                        "type": "object"
                    },
                    {
                        "name": "Feature Attribution",
                        "path": "historical.attribution[detector_name]",
                        "js_access": "window.dataService.getFeatureAttribution('vix_mean_reversion')",
                        "type": "array"
                    }
                ]
            }
        }
        return patterns
    
    def find_old_fetch_calls(self, html_content: str) -> List[Dict]:
        """Find old fetch() calls in HTML that need migration."""
        import re
        
        fetch_pattern = r"fetch\(['\"]([^'\"]+)['\"].*?\)"
        matches = re.finditer(fetch_pattern, html_content)
        
        findings = []
        for match in matches:
            url = match.group(1)
            if '.json' in url:
                findings.append({
                    "line": html_content[:match.start()].count('\n') + 1,
                    "url": url,
                    "full_match": match.group(0)
                })
        
        return findings
    
    def generate_migration_guide(self, old_fetches: List[Dict]) -> str:
        """Generate migration instructions for old fetch calls."""
        guide = []
        guide.append("=" * 80)
        guide.append("MIGRATION GUIDE: Old fetch() → data_service.js")
        guide.append("=" * 80)
        guide.append("")
        
        # Map old files to new structure
        migration_map = {
            "market_state.json": {
                "new_file": "live_state.json",
                "method": "getMarketState()",
                "notes": "Use window.dataService.getMarketState() for current market data"
            },
            "anomaly_report.json": {
                "new_file": "live_state.json",
                "method": "getAnomalyState()",
                "notes": "Use window.dataService.getAnomalyState() for anomaly metrics"
            },
            "historical_anomaly_scores.json": {
                "new_file": "historical.json",
                "method": "getHistoricalData()",
                "notes": "Use window.dataService.getHistoricalData() for time series"
            },
            "anomaly_feature_attribution.json": {
                "new_file": "historical.json",
                "method": "getFeatureAttribution(detector_name)",
                "notes": "Use window.dataService.getFeatureAttribution() with detector name"
            },
            "regime_statistics.json": {
                "new_file": "historical.json",
                "method": "getRegimeStats()",
                "notes": "Use window.dataService.getRegimeStats() for regime analysis"
            },
            "anomaly_metadata.json": {
                "new_file": "historical.json",
                "method": "getDetectorMetadata()",
                "notes": "Use window.dataService.getDetectorMetadata() for detector info"
            }
        }
        
        for fetch in old_fetches:
            filename = fetch['url'].split('/')[-1]
            if filename in migration_map:
                info = migration_map[filename]
                guide.append(f"Line {fetch['line']}: {filename}")
                guide.append(f"  OLD: {fetch['full_match']}")
                guide.append(f"  NEW: window.dataService.{info['method']}")
                guide.append(f"  📝 {info['notes']}")
                guide.append("")
        
        return "\n".join(guide)
    
    def generate_html_template(self) -> str:
        """Generate template HTML showing proper data_service.js usage."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Chart Template - Data Service Integration</title>
    <script src="../../data_service.js"></script>
</head>
<body>
    <div id="chart"></div>
    
    <script>
        // ====================================================================
        // STEP 1: Initialize Data Service (ONCE per page)
        // ====================================================================
        async function initChart() {
            try {
                // Initialize data service (loads both JSON files in parallel)
                await window.dataService.init();
                console.log('✅ Data loaded');
                
                // ================================================================
                // STEP 2: Get data for your chart
                // ================================================================
                
                // For live/current data:
                const anomaly = window.dataService.getAnomalyState();
                const market = window.dataService.getMarketState();
                
                console.log('Current Score:', anomaly.score);
                console.log('Current VIX:', market.vix_close);
                console.log('Active Detectors:', anomaly.activeDetectors);
                
                // For historical data:
                const historical = window.dataService.getHistoricalData();
                
                console.log('Dates:', historical.dates.length);
                console.log('Scores:', historical.scores.length);
                
                // For thresholds:
                const thresholds = window.dataService.getThresholds();
                console.log('Moderate Threshold:', thresholds.base.moderate);
                
                // ================================================================
                // STEP 3: Build your chart with the data
                // ================================================================
                buildChart({
                    currentScore: anomaly.score,
                    classification: anomaly.classification,
                    historicalData: historical,
                    thresholds: thresholds.base
                });
                
                // ================================================================
                // STEP 4: Listen for data refreshes (optional)
                // ================================================================
                window.dataService.on('data-refreshed', (data) => {
                    console.log('🔄 Data refreshed, updating chart...');
                    updateChart(data.live);
                });
                
            } catch (error) {
                console.error('❌ Failed to initialize:', error);
                document.getElementById('chart').innerHTML = 
                    '<div class="error">Failed to load data</div>';
            }
        }
        
        function buildChart(data) {
            // Your Plotly/D3/Chart.js code here
            console.log('Building chart with:', data);
        }
        
        function updateChart(liveData) {
            // Update chart with refreshed data
            console.log('Updating chart with:', liveData);
        }
        
        // Initialize on page load
        initChart();
    </script>
</body>
</html>
"""
    
    def run_full_analysis(self, output_dir: str = "./migration_docs"):
        """Run complete analysis and generate all documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🔍 JSON Structure Crawler")
        print("=" * 80)
        
        # Analyze both JSON files
        files_to_analyze = ["live_state.json", "historical.json"]
        
        for filename in files_to_analyze:
            print(f"\n📄 Analyzing {filename}...")
            analysis = self.analyze_file(filename)
            
            if "error" in analysis:
                print(f"   ❌ {analysis['error']}")
                continue
            
            print(f"   ✅ Size: {analysis['size_kb']} KB")
            print(f"   📅 Modified: {analysis['modified']}")
            
            # Save detailed structure
            output_file = output_path / f"{filename.replace('.json', '_structure.json')}"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"   💾 Saved structure: {output_file}")
            
            self.structures[filename] = analysis
        
        # Generate access patterns
        print("\n📊 Generating access patterns...")
        patterns = self.generate_access_patterns()
        patterns_file = output_path / "access_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        print(f"   💾 Saved: {patterns_file}")
        
        # Generate HTML template
        print("\n📝 Generating HTML template...")
        template = self.generate_html_template()
        template_file = output_path / "chart_template.html"
        with open(template_file, 'w') as f:
            f.write(template)
        print(f"   💾 Saved: {template_file}")
        
        # Generate comprehensive report
        print("\n📋 Generating comprehensive report...")
        report = self.generate_comprehensive_report()
        report_file = output_path / "migration_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"   💾 Saved: {report_file}")
        
        print("\n" + "=" * 80)
        print("✅ Analysis complete!")
        print(f"📁 Output directory: {output_path.absolute()}")
        print("\nGenerated files:")
        print("  • live_state_structure.json")
        print("  • historical_structure.json")
        print("  • access_patterns.json")
        print("  • chart_template.html")
        print("  • migration_report.md")
        
    def generate_comprehensive_report(self) -> str:
        """Generate markdown report with all findings."""
        report = []
        report.append("# JSON Structure Analysis & Migration Guide")
        report.append("")
        report.append(f"**Generated:** {datetime.now().isoformat()}")
        report.append("")
        
        report.append("## Overview")
        report.append("")
        report.append("This system consolidates 6+ legacy JSON files into 2 unified files:")
        report.append("")
        report.append("1. **live_state.json** (15 KB) - Updates every 15 minutes")
        report.append("2. **historical.json** (300 KB) - Static training data")
        report.append("")
        
        report.append("## File Comparison")
        report.append("")
        report.append("| Legacy File | New Location | Access Method |")
        report.append("|------------|-------------|---------------|")
        report.append("| market_state.json | live_state.json | `getMarketState()` |")
        report.append("| anomaly_report.json | live_state.json | `getAnomalyState()` |")
        report.append("| historical_anomaly_scores.json | historical.json | `getHistoricalData()` |")
        report.append("| anomaly_feature_attribution.json | historical.json | `getFeatureAttribution()` |")
        report.append("| regime_statistics.json | historical.json | `getRegimeStats()` |")
        report.append("| anomaly_metadata.json | historical.json | `getDetectorMetadata()` |")
        report.append("")
        
        report.append("## Data Service API")
        report.append("")
        report.append("### Initialization")
        report.append("```javascript")
        report.append("// Call ONCE on page load")
        report.append("await window.dataService.init();")
        report.append("```")
        report.append("")
        
        report.append("### Live Data Methods")
        report.append("")
        report.append("```javascript")
        report.append("// Get current anomaly state")
        report.append("const anomaly = window.dataService.getAnomalyState();")
        report.append("// Returns: { score, classification, activeDetectors, detectorScores, persistence, diagnostics }")
        report.append("")
        report.append("// Get current market state")
        report.append("const market = window.dataService.getMarketState();")
        report.append("// Returns: { timestamp, vix_close, spx_close, regime, ... }")
        report.append("```")
        report.append("")
        
        report.append("### Historical Data Methods")
        report.append("")
        report.append("```javascript")
        report.append("// Get time series data")
        report.append("const hist = window.dataService.getHistoricalData();")
        report.append("// Returns: { dates, scores, spx, forwardReturns, regimeStats, thresholds }")
        report.append("")
        report.append("// Get thresholds")
        report.append("const thresholds = window.dataService.getThresholds();")
        report.append("// Returns: { base, withCI, hasConfidenceIntervals }")
        report.append("")
        report.append("// Get feature attribution for a detector")
        report.append("const features = window.dataService.getFeatureAttribution('vix_mean_reversion');")
        report.append("// Returns: array of feature importance data")
        report.append("```")
        report.append("")
        
        report.append("## Migration Checklist")
        report.append("")
        report.append("- [ ] Add `<script src=\"../../data_service.js\"></script>` to HTML")
        report.append("- [ ] Replace all `fetch('../../json_data/xxx.json')` calls")
        report.append("- [ ] Use `await window.dataService.init()` once on page load")
        report.append("- [ ] Use appropriate getter methods instead of direct fetch")
        report.append("- [ ] Add error handling for failed data loads")
        report.append("- [ ] Test chart with refreshed data")
        report.append("- [ ] Remove old JSON file references")
        report.append("")
        
        report.append("## Benefits")
        report.append("")
        report.append("✅ **Performance**: 2 fetches instead of 6+")
        report.append("✅ **Caching**: Data loaded once and shared")
        report.append("✅ **Consistency**: Single source of truth")
        report.append("✅ **Auto-refresh**: Built-in refresh mechanism")
        report.append("✅ **Error handling**: Centralized error management")
        report.append("✅ **Events**: Listen for data updates across all charts")
        report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import sys
    
    # Check if json_data directory is specified
    json_dir = sys.argv[1] if len(sys.argv) > 1 else "./json_data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./migration_docs"
    
    crawler = JSONCrawler(json_dir)
    crawler.run_full_analysis(output_dir)
    
    print("\n🎯 Next Steps:")
    print("1. Review migration_docs/migration_report.md")
    print("2. Run html_migrator.py to update your HTML files")
    print("3. Test each chart with the new data service")


if __name__ == "__main__":
    main()
