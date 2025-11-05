"""
Dashboard Orchestrator - WITH AUTO-REFRESH
Adds background thread to update data every 15 seconds.
Claude's "I'm not in school anymore" edition.
"""

import subprocess
import sys
import webbrowser
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import http.server
import socketserver
import threading
import traceback


def _format_feature_name(raw_name: str) -> str:
    """Convert raw feature names to readable labels."""
    replacements = {
        'vix_vs_ma': 'VIX vs MA', 'vix_zscore': 'VIX Z-Score',
        'vix_percentile': 'VIX Percentile', 'vix_velocity': 'VIX Velocity',
        'vix_momentum_z': 'VIX Momentum Z', 'vix_accel': 'VIX Acceleration',
        'vix_vol': 'VIX Volatility', 'spx_vs_ma': 'SPX vs MA',
        'spx_ret': 'SPX Return', 'spx_momentum_z': 'SPX Momentum Z',
        'spx_realized_vol': 'SPX Realized Vol', 'spx_vix_corr': 'SPX-VIX Correlation',
        'vix_rv_ratio': 'VIX/RV Ratio', 'Treasury_10Y': '10Y Treasury',
        'Treasury_2Y': '2Y Treasury', 'Yield_Curve': 'Yield Curve',
        'High_Yield_Spread': 'HY Spread', '_mom_': ' Momentum ',
        '_zscore_': ' Z-Score ', '_pct': ' %', '_level': '', '_change': ' Change'
    }
    
    formatted = raw_name
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    formatted = formatted.replace('21d', '(21d)').replace('63d', '(63d)').replace('10d', '(10d)')
    formatted = formatted.replace('5d', '(5d)').replace('252d', '(252d)').replace('126d', '(126d)')
    formatted = formatted.replace('_', ' ').strip()
    
    return formatted


class AutoRefreshDataWorker:
    """Background worker that refreshes data on interval."""
    
    def __init__(self, system, interval_seconds: int = 15):
        self.system = system
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self.last_refresh = None
        self.refresh_count = 0
        self.error_count = 0
        
    def start(self):
        """Start the background refresh worker."""
        if self.running:
            print("⚠️  Refresh worker already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.thread.start()
        print(f"✅ Auto-refresh started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop the background worker."""
        self.running = False
        print("\n🛑 Auto-refresh stopped")
    
    def _refresh_loop(self):
        """Main refresh loop - runs in background thread."""
        while self.running:
            try:
                time.sleep(self.interval)
                
                if not self.running:
                    break
                
                # Get live VIX
                try:
                    live_vix = self.system.vix_predictor.fetcher.fetch_price('^VIX')
                    if live_vix:
                        self.system.vix_predictor.vix.iloc[-1] = live_vix
                except:
                    pass
                
                # Re-run anomaly detection with updated VIX
                cached_anomaly_result = self.system._get_cached_anomaly_result(force_refresh=True)
                
                # Export all JSON files
                self._export_all_data(cached_anomaly_result)
                
                self.last_refresh = datetime.now()
                self.refresh_count += 1
                
                print(f"🔄 [{self.last_refresh.strftime('%H:%M:%S')}] Data refreshed (#{self.refresh_count})")
                
            except Exception as e:
                self.error_count += 1
                print(f"❌ Refresh error #{self.error_count}: {e}")
                if self.error_count > 10:
                    print("⚠️  Too many errors, stopping auto-refresh")
                    self.running = False
    
    def _export_all_data(self, cached_anomaly_result):
        """Export all JSON files (called by refresh loop)."""
        try:
            # Export market state
            self.system.export_json("./json_data/market_state.json")
            
            # Export anomaly report
            self.system.vix_predictor.export_anomaly_report(
                filepath="./json_data/anomaly_report.json",
                anomaly_result=cached_anomaly_result
            )
            
            # Export unified dashboard data
            from dashboard_data_contract import export_dashboard_data
            
            export_dashboard_data(
                vix_predictor=self.system.vix_predictor,
                spx_predictor=None,
                output_dir='./json_data',
                anomaly_result=cached_anomaly_result
            )
            
        except Exception as e:
            print(f"   ⚠️  Export error: {e}")


class DashboardOrchestrator:
    """Orchestrate 15-detector system training and dashboard launch with auto-refresh."""
    
    REQUIRED_FILES = [
        'regime_statistics.json',
        'vix_history.json',
        'market_state.json',
        'anomaly_report.json',
        'dashboard_data.json'
    ]
    
    def __init__(self, json_dir: str = './json_data'):
        self.json_dir = Path(json_dir)
        self.json_dir.mkdir(exist_ok=True)
        self.refresh_worker = None
    
    def run(self, years: int = 7, port: int = 8000, skip_training: bool = False, 
            auto_refresh: bool = True, refresh_interval: int = 15):
        """Execute complete workflow with optional auto-refresh."""
        self._print_header("15-DETECTOR ANOMALY SYSTEM", years, port, skip_training, auto_refresh)
        
        # Train or load system
        if not skip_training:
            system = self._train_system(years)
        else:
            print("\n⚡ Skipping training (using existing models)")
            # Load existing system for refresh
            try:
                from integrated_system_production import IntegratedMarketSystemV4
                system = IntegratedMarketSystemV4()
                system.train(years=years, real_time_vix=True, verbose=False)
            except Exception as e:
                print(f"❌ Could not load system: {e}")
                system = None
        
        self._verify_exports()
        
        # Start auto-refresh worker if requested
        if auto_refresh and system:
            self.refresh_worker = AutoRefreshDataWorker(system, refresh_interval)
            self.refresh_worker.start()
        
        # Launch dashboard
        self._launch_dashboard(port)
        
        # Cleanup on exit
        if self.refresh_worker:
            self.refresh_worker.stop()
        
        print("\n" + "="*80)
        print("✅ ORCHESTRATION COMPLETE")
        print("="*80)
    
    def _print_header(self, title: str, years: int, port: int, skip: bool, auto_refresh: bool):
        """Print startup banner."""
        print("\n" + "="*80)
        print(f"{title}")
        print("="*80)
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Training: {years} years")
        print(f"🌐 Port: {port}")
        print(f"⚡ Skip training: {skip}")
        print(f"🔄 Auto-refresh: {'Enabled' if auto_refresh else 'Disabled'}")
    
    def _train_system(self, years: int):
        """Train integrated system."""
        print("\n" + "="*80)
        print("TRAINING INTEGRATED SYSTEM")
        print("="*80)
        
        try:
            from integrated_system_production import IntegratedMarketSystemV4
            
            system = IntegratedMarketSystemV4()
            
            print("\n🔬 Training complete system with unified features...")
            system.train(years=years, real_time_vix=True, verbose=False)
            print("   ✅ VIX + 15 Anomaly Detectors trained")
            
            print("\n📤 Exporting data files...")
            self._export_data(system)
            
            print("\n✅ Training complete")
            
            return system
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _export_data(self, system):
        """Export all required JSON files using cached anomaly detection results."""
        
        # Get cached anomaly result ONCE
        cached_anomaly_result = system._get_cached_anomaly_result()
        
        # Legacy exports
        exports = [
            ('regime_statistics.json', lambda: system.vix_predictor.export_regime_statistics),
            ('vix_history.json', lambda: system.vix_predictor.export_vix_history),
            ('market_state.json', lambda: system.export_json),
        ]
        
        for filename, export_func in exports:
            filepath = self.json_dir / filename
            try:
                export_func()(str(filepath))
                print(f"   ✅ {filename}")
            except Exception as e:
                print(f"   ❌ {filename}: {e}")
        
        # Export anomaly report
        try:
            filepath = self.json_dir / 'anomaly_report.json'
            system.vix_predictor.export_anomaly_report(
                filepath=str(filepath),
                anomaly_result=cached_anomaly_result
            )
            print(f"   ✅ anomaly_report.json")
        except Exception as e:
            print(f"   ❌ anomaly_report.json: {e}")
        
        # Unified export
        print("\n📤 Exporting unified dashboard data...")
        try:
            from dashboard_data_contract import export_dashboard_data
            
            dashboard_data = export_dashboard_data(
                vix_predictor=system.vix_predictor,
                spx_predictor=None,
                output_dir=str(self.json_dir),
                anomaly_result=cached_anomaly_result
            )
            print(f"   ✅ dashboard_data.json")
        except Exception as e:
            print(f"   ⚠️  Unified export failed: {e}")
            traceback.print_exc()
        
        print("\n📊 Exporting anomaly explainers...")
        self._export_anomaly_metadata()
        self._export_feature_attribution(system, cached_anomaly_result)

    def _export_anomaly_metadata(self):
        """Export static domain definitions."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "domains": {
                "vix_mean_reversion": {
                    "name": "VIX Mean Reversion",
                    "description": "Detects when VIX is stretched far from its historical average.",
                    "key_signals": ["VIX vs 21-day MA", "VIX vs 63-day MA", "Z-score", "Distance from regime mean"]
                },
                "vix_momentum": {
                    "name": "VIX Momentum",
                    "description": "Tracks velocity and acceleration of VIX changes.",
                    "key_signals": ["5-day velocity", "10-day rate of change", "Gap vs previous close", "Acceleration"]
                },
                "vix_regime_structure": {
                    "name": "VIX Regime Structure",
                    "description": "Monitors stability of volatility regime.",
                    "key_signals": ["Days in regime", "VIX percentile", "Regime displacement", "Transition probability"]
                },
                "cboe_options_flow": {
                    "name": "CBOE Options Flow",
                    "description": "Analyzes CBOE indicators for hedging stress.",
                    "key_signals": ["SKEW", "Put/Call ratio", "Correlation indices", "VIX term structure"]
                },
                "vix_spx_relationship": {
                    "name": "VIX-SPX Relationship",
                    "description": "Measures correlation breakdown between VIX and SPX.",
                    "key_signals": ["21-day correlation", "VIX vs SPX divergence", "Beta instability"]
                },
                "spx_price_action": {
                    "name": "SPX Price Action",
                    "description": "Detects unusual price patterns and momentum extremes.",
                    "key_signals": ["SPX vs MAs", "Rate of change", "Gap behavior", "Support/resistance"]
                },
                "spx_volatility_regime": {
                    "name": "SPX Volatility Regime",
                    "description": "Tracks realized volatility vs implied (VIX).",
                    "key_signals": ["21-day realized vol", "VIX vs RV spread", "Vol-of-vol", "Historical vol percentile"]
                },
                "macro_rates": {
                    "name": "Macro Rates",
                    "description": "Monitors treasury yields and yield curve for macro stress.",
                    "key_signals": ["10Y yield", "2Y-10Y curve", "Rate volatility", "Real yields"]
                },
                "commodities_stress": {
                    "name": "Commodities Stress",
                    "description": "Tracks oil, gold, and commodity volatility.",
                    "key_signals": ["Oil momentum", "Gold vs SPX correlation", "Commodity vol spikes"]
                },
                "cross_asset_divergence": {
                    "name": "Cross-Asset Divergence",
                    "description": "Detects when equities, bonds, commodities send conflicting signals.",
                    "key_signals": ["Equity-bond correlation", "Dollar strength", "Cross-asset vol dispersion"]
                }
            }
        }
        
        filepath = self.json_dir / 'anomaly_metadata.json'
        with open(filepath, 'w') as f:
            json.dump(metadata, indent=2, fp=f)
        
        print(f"   ✅ anomaly_metadata.json")

    def _export_feature_attribution(self, system, cached_anomaly_result=None):
        """Export feature attributions."""
        try:
            features = system.vix_predictor.features.iloc[-1].to_dict()
            
            if cached_anomaly_result is not None:
                anomaly_result = cached_anomaly_result
            else:
                anomaly_result = system.vix_predictor.anomaly_detector.detect(
                    system.vix_predictor.features.iloc[[-1]], 
                    verbose=False
                )
            
            attribution = {
                "timestamp": datetime.now().isoformat(),
                "ensemble_score": anomaly_result['ensemble']['score'],
                "domains": {}
            }
            
            for domain_name, domain_data in anomaly_result['domain_anomalies'].items():
                importances = system.vix_predictor.anomaly_detector.feature_importances.get(domain_name, {})
                
                if importances:
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:7]
                    top_features = []
                    
                    for feature_name, importance in sorted_features:
                        value = features.get(feature_name, None)
                        
                        if value is not None and not np.isnan(value):
                            top_features.append({
                                "name": _format_feature_name(feature_name),
                                "value": float(value),
                                "importance": float(importance)
                            })
                
                attribution["domains"][domain_name] = {
                    "score": domain_data['score'],
                    "level": domain_data['level'],
                    "features": top_features
                }
            
            filepath = self.json_dir / 'anomaly_feature_attribution.json'
            with open(filepath, 'w') as f:
                json.dump(attribution, indent=2, fp=f)
            
            print(f"   ✅ anomaly_feature_attribution.json")
            
        except Exception as e:
            print(f"   ⚠️  Could not generate feature attribution: {e}")
    
    def _verify_exports(self):
        """Verify all required JSON files exist."""
        print("\n" + "="*80)
        print("VERIFYING EXPORTS")
        print("="*80)
        
        all_valid = True
        
        for filename in self.REQUIRED_FILES:
            filepath = self.json_dir / filename
            
            if not filepath.exists():
                print(f"   ❌ {filename:30s} MISSING")
                all_valid = False
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                size_kb = filepath.stat().st_size / 1024
                print(f"   ✅ {filename:30s} ({size_kb:6.1f} KB)")
                
            except json.JSONDecodeError:
                print(f"   ❌ {filename:30s} INVALID JSON")
                all_valid = False
            except Exception as e:
                print(f"   ❌ {filename:30s} ERROR: {e}")
                all_valid = False
        
        if not all_valid:
            print("\n⚠️  Some files missing/invalid - re-run training")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        else:
            print("\n✅ All required files valid")
    
    def _launch_dashboard(self, port: int):
        """Launch HTTP server and open dashboard."""
        print("\n" + "="*80)
        print("LAUNCHING DASHBOARD")
        print("="*80)
        
        print(f"\n🚀 Starting server on port {port}...")
        
        class CORSHandler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                return super().end_headers()
            
            def do_OPTIONS(self):
                self.send_response(200)
                self.end_headers()
            
            def log_message(self, format, *args):
                pass
        
        def run_server():
            with socketserver.TCPServer(("", port), CORSHandler) as httpd:
                httpd.serve_forever()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(1)
        
        base_url = f"http://localhost:{port}"
        dashboard_url = f"{base_url}/dashboard_unified.html"
        
        print(f"   ✅ Server running at {base_url}")
        
        print(f"\n🌐 Opening dashboard...")
        try:
            webbrowser.open(dashboard_url)
            print(f"   ✅ Browser opened")
        except Exception as e:
            print(f"   ⚠️  Could not open browser: {e}")
            print(f"   Manually open: {dashboard_url}")
        
        print("\n" + "="*80)
        if self.refresh_worker and self.refresh_worker.running:
            print("ℹ️  Press Ctrl+C to stop server and auto-refresh")
        else:
            print("ℹ️  Press Ctrl+C to stop server")
        print("="*80)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            if self.refresh_worker:
                self.refresh_worker.stop()
            print("✅ Server stopped")


def main():
    """Entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Dashboard Orchestrator for 15-Detector Anomaly System (Auto-Refresh Edition)'
    )
    
    parser.add_argument('--years', type=int, default=7, help='Years of data (default: 7)')
    parser.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    parser.add_argument('--skip-training', action='store_true', help='Skip training')
    parser.add_argument('--no-refresh', action='store_true', help='Disable auto-refresh')
    parser.add_argument('--refresh-interval', type=int, default=15, help='Refresh interval in seconds (default: 15)')
    
    args = parser.parse_args()
    
    try:
        orchestrator = DashboardOrchestrator()
        orchestrator.run(
            years=args.years,
            port=args.port,
            skip_training=args.skip_training,
            auto_refresh=not args.no_refresh,
            refresh_interval=args.refresh_interval
        )
    except KeyboardInterrupt:
        print("\n\n✅ Stopped by user")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
