"""Dashboard Orchestrator - Auto-refresh with Exponential Backoff
TRAINING SCHEDULE (Recommended):
- Monday morning:     ENABLE_TRAINING=True  (full retrain, ~3 min)
- Tuesday-Friday:     ENABLE_TRAINING=False (cached models, ~5 sec startup)
- Intraday refresh:   Auto (15 min intervals during market hours)

REFRESH BEHAVIOR:
- Fetches live ^VIX and ^GSPC prices
- Recalculates all derived features (zscore, momentum, percentiles, etc.)
- Re-runs anomaly detection with fresh features
- Updates live_state.json only (historical.json unchanged)
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
from config import TRAINING_YEARS, ENABLE_TRAINING
from export.unified_exporter import UnifiedExporter

REFRESH_INTERVAL = 900


def _format_feature_name(raw_name: str) -> str:
    """Format feature names for display."""
    replacements = {
        'vix_vs_ma': 'VIX vs MA', 'vix_zscore': 'VIX Z-Score', 'vix_percentile': 'VIX Percentile',
        'vix_velocity': 'VIX Velocity', 'vix_momentum_z': 'VIX Momentum Z', 'vix_accel': 'VIX Acceleration',
        'vix_vol': 'VIX Volatility', 'spx_vs_ma': 'SPX vs MA', 'spx_ret': 'SPX Return',
        'spx_momentum_z': 'SPX Momentum Z', 'spx_realized_vol': 'SPX Realized Vol',
        'spx_vix_corr': 'SPX-VIX Correlation', 'vix_rv_ratio': 'VIX/RV Ratio',
        'Treasury_10Y': '10Y Treasury', 'Treasury_2Y': '2Y Treasury', 'Yield_Curve': 'Yield Curve',
        'High_Yield_Spread': 'HY Spread', '_mom_': ' Momentum ', '_zscore_': ' Z-Score ',
        '_pct': ' %', '_level': '', '_change': ' Change'
    }
    
    formatted = raw_name
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    formatted = formatted.replace('21d', '(21d)').replace('63d', '(63d)').replace('10d', '(10d)')
    formatted = formatted.replace('5d', '(5d)').replace('252d', '(252d)').replace('126d', '(126d)')
    return formatted.replace('_', ' ').strip()


class DashboardOrchestrator:
    REQUIRED_FILES = [
        'live_state.json', 'historical.json', 'model_cache.pkl'
    ]
    
    def __init__(self, json_dir: str = './json_data'):
        self.json_dir = Path(json_dir)
        self.json_dir.mkdir(exist_ok=True)
        self.system = None
        self.refresh_running = False
        self.refresh_thread = None
        self.refresh_interval = REFRESH_INTERVAL
        self.refresh_count = 0
        self.error_count = 0
        self.consecutive_failures = 0
        self.base_refresh_interval = REFRESH_INTERVAL
        self.max_refresh_interval = 300
        self.max_failures_before_stop = 10
        self.last_successful_refresh = None
    
    def _calculate_backoff_delay(self) -> int:
        """Calculate exponential backoff delay based on failure count."""
        if self.consecutive_failures == 0:
            return self.base_refresh_interval
        
        delay = min(
            self.base_refresh_interval * (2 ** self.consecutive_failures),
            self.max_refresh_interval
        )
        return int(delay)
    
    def run(self, years: int = TRAINING_YEARS, port: int = 8000, skip_training: bool = False,
            auto_refresh: bool = True, refresh_interval: int = REFRESH_INTERVAL):
        """Main orchestration flow."""
        self.refresh_interval = refresh_interval
        self.base_refresh_interval = refresh_interval
        effective_skip = skip_training or not ENABLE_TRAINING
        
        self._print_header("15-DETECTOR ANOMALY SYSTEM", years, port, effective_skip, auto_refresh)
        
        if not effective_skip:
            self.system = self._train_system(years)
        else:
            print("\n⚡ Skipping training (loading cached model state)")
            self.system = self._load_cached_system()
        
        self._verify_exports()
        
        if auto_refresh and self.system:
            self._start_refresh()
        
        self._launch_dashboard(port)
        
        if self.refresh_running:
            self._stop_refresh()
        
        print("\n" + "="*80 + "\n✅ ORCHESTRATION COMPLETE\n" + "="*80)
    
    def _load_cached_system(self):
        """Load system from cached state without retraining."""
        print(f"\n{'='*80}\nLOADING CACHED SYSTEM STATE\n{'='*80}")
        
        try:
            from integrated_system_production import IntegratedMarketSystemV4
            system = IntegratedMarketSystemV4()
            
            refresh_state_path = self.json_dir / 'refresh_state.pkl'
            if not refresh_state_path.exists():
                print(f"\n❌ CRITICAL: refresh_state.pkl not found at {refresh_state_path}")
                print("   Run with ENABLE_TRAINING=True first to generate cached state")
                sys.exit(1)
            
            system.vix_predictor.load_refresh_state(str(refresh_state_path))
            
            market_state_path = self.json_dir / 'market_state.json'
            if market_state_path.exists():
                with open(market_state_path, 'r') as f:
                    market_state = json.load(f)
                    if 'market_data' in market_state and 'spx_model' in market_state['market_data']:
                        import pandas as pd
                        system.spx = pd.Series([market_state['market_data']['spx_model']], 
                                              index=system.vix_predictor.vix_ml.tail(1).index)
                        print(f"   ✅ Loaded SPX reference from market_state.json")
            
            if system.spx is None:
                import pandas as pd
                system.spx = system.vix_predictor.spx_ml.copy()
                print(f"   ✅ Using SPX from cached state")
            
            system.trained = True
            print(f"\n✅ SYSTEM LOADED - READY FOR LIVE UPDATES\n{'='*80}\n")
            return system
            
        except FileNotFoundError as e:
            print(f"\n❌ Load failed: {e}")
            print("\n   SOLUTION: Run once with ENABLE_TRAINING=True:")
            print("   1. Set ENABLE_TRAINING = True in config.py")
            print("   2. Run: python dashboard_orchestrator.py")
            print("   3. After successful run, set ENABLE_TRAINING = False")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Load failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _start_refresh(self):
        """Start auto-refresh thread."""
        if self.refresh_running:
            print("⚠️ Refresh already running")
            return
        
        self.refresh_running = True
        self.refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.refresh_thread.start()
        print(f"✅ Auto-refresh started ({self.refresh_interval}s = {self.refresh_interval/60:.1f} min)")
        print(f"   Fetching live: ^VIX and ^GSPC")
        print(f"   🔄 Exponential backoff enabled (max: {self.max_failures_before_stop} failures)")
    
    def _stop_refresh(self):
        """Stop auto-refresh thread."""
        self.refresh_running = False
        print("\n🛑 Auto-refresh stopped")
        if self.consecutive_failures > 0:
            print(f"   Final failure count: {self.consecutive_failures}")
        if self.last_successful_refresh:
            print(f"   Last success: {self.last_successful_refresh.strftime('%H:%M:%S')}")
    
    def _refresh_loop(self):
        """Main refresh loop with exponential backoff."""
        while self.refresh_running:
            try:
                current_delay = self._calculate_backoff_delay()
                time.sleep(current_delay)
                
                if not self.refresh_running:
                    break
                
                refresh_success = self._attempt_refresh()
                
                if refresh_success:
                    if self.consecutive_failures > 0:
                        print(f"   ✅ Refresh recovered after {self.consecutive_failures} failures")
                    
                    self.consecutive_failures = 0
                    self.last_successful_refresh = datetime.now()
                    self.refresh_count += 1
                    print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Data refreshed (#{self.refresh_count})")
                else:
                    self.consecutive_failures += 1
                    next_delay = self._calculate_backoff_delay()
                    print(f"⚠️ Refresh failed ({self.consecutive_failures}x) - backing off to {next_delay}s ({next_delay/60:.1f} min)")
                    
                    if self.consecutive_failures >= self.max_failures_before_stop:
                        print(f"\n❌ CRITICAL: {self.max_failures_before_stop} consecutive failures - stopping auto-refresh")
                        if self.last_successful_refresh:
                            print(f"   Last success: {self.last_successful_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Manual refresh still available in dashboard")
                        self.refresh_running = False
                        break
            except KeyboardInterrupt:
                print("\n🛑 Auto-refresh stopped by user")
                break
            except Exception as e:
                self.consecutive_failures += 1
                next_delay = self._calculate_backoff_delay()
                print(f"❌ Refresh exception ({self.consecutive_failures}x): {e}")
                print(f"   Backing off to {next_delay}s")
                if self.consecutive_failures >= self.max_failures_before_stop:
                    print(f"\n❌ CRITICAL: Too many failures - stopping auto-refresh")
                    self.refresh_running = False
                    break
    
    def _attempt_refresh(self) -> bool:
        """Attempt to refresh data."""
        try:
            live_vix = live_spx = None
            
            # Fetch live prices
            try:
                live_vix = self.system.vix_predictor.fetcher.fetch_price('^VIX')
            except Exception as e:
                print(f"   ⚠️ VIX fetch error: {e}")
            
            try:
                live_spx = self.system.vix_predictor.fetcher.fetch_price('^GSPC')
            except Exception as e:
                print(f"   ⚠️ SPX fetch error: {e}")
            
            # If no prices available, use cached anomaly result
            if not live_vix or not live_spx:
                print(f"   ⚠️ No live prices available (market closed?)")
                cached_anomaly_result = self.system._get_cached_anomaly_result(force_refresh=False)
                self._export_all_for_refresh(cached_anomaly_result)
                return False
            
            # Store old prices for logging
            old_vix = self.system.vix_predictor.vix_ml.iloc[-1]
            old_spx = self.system.vix_predictor.spx_ml.iloc[-1]
            
            # ✅ NEW: Atomic update with feature recalculation
            update_result = self.system._recalculate_live_features(live_vix, live_spx)
            
            # Log changes
            print(f"   📊 VIX: {old_vix:.2f} → {live_vix:.2f} ({live_vix - old_vix:+.2f})")
            spx_change = ((live_spx - old_spx) / old_spx) * 100
            print(f"   📈 SPX: {old_spx:.2f} → {live_spx:.2f} ({spx_change:+.2f}%)")
            
            # Recalculate anomaly scores with fresh features
            cached_anomaly_result = self.system._get_cached_anomaly_result(force_refresh=True)
            new_score = cached_anomaly_result['ensemble']['score']
            
            # Classify severity
            if hasattr(self.system.vix_predictor.anomaly_detector, 'classify_anomaly'):
                severity, _, _ = self.system.vix_predictor.anomaly_detector.classify_anomaly(
                    new_score, method='statistical'
                )
            else:
                thresholds = {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}
                if new_score >= thresholds['critical']:
                    severity = 'CRITICAL'
                elif new_score >= thresholds['high']:
                    severity = 'HIGH'
                elif new_score >= thresholds['moderate']:
                    severity = 'MODERATE'
                else:
                    severity = 'NORMAL'
            
            print(f"   🔴 Anomaly Score: {new_score:.3f} ({severity})")
            
            # Export updated state
            self._export_all_for_refresh(cached_anomaly_result)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Refresh attempt failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _export_all_for_refresh(self, cached_anomaly_result):
        """Export live state during refresh (historical/model unchanged)."""
        try:
            
            # Calculate persistence stats
            persistence_stats = None
            if hasattr(self.system.vix_predictor, 'historical_ensemble_scores') and \
               self.system.vix_predictor.historical_ensemble_scores is not None:
                persistence_stats = self.system.vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
                    self.system.vix_predictor.historical_ensemble_scores,
                    dates=self.system.vix_predictor.features.index
                )
            
            # Initialize exporter
            exporter = UnifiedExporter(output_dir=str(self.json_dir))
            
            # ONLY export live state during refresh
            exporter.export_live_state(
                vix_predictor=self.system.vix_predictor,
                anomaly_result=cached_anomaly_result,
                spx=self.system.spx,
                persistence_stats=persistence_stats
            )
            
            print("   ✅ Live state updated")
            
        except Exception as e:
            print(f"   ⚠️ Export error: {e}")
            traceback.print_exc()
            raise
    
    def _print_header(self, title: str, years: int, port: int, skip: bool, auto_refresh: bool):
        """Print orchestrator header."""
        print("\n" + "="*80 + f"\n{title}\n" + "="*80)
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Training: {years} years")
        print(f"🌐 Port: {port}")
        print(f"⚡ Config training: {'DISABLED' if not ENABLE_TRAINING else 'ENABLED'}")
        print(f"⚡ Skip flag: {skip}")
        if auto_refresh:
            print(f"🔄 Auto-refresh: Enabled ({self.refresh_interval}s = {self.refresh_interval/60:.1f} min)")
            print(f"   Exponential backoff: 15s → 30s → 60s → 120s → 300s (max)")
            print(f"   Circuit breaker: {self.max_failures_before_stop} failures")
        else:
            print(f"🔄 Auto-refresh: Disabled")
    
    def _train_system(self, years: int):
        """Train integrated system."""
        print("\n" + "="*80 + "\nTRAINING INTEGRATED SYSTEM\n" + "="*80)
        
        try:
            from integrated_system_production import IntegratedMarketSystemV4
            system = IntegratedMarketSystemV4()
            
            print("\n🔬 Training complete system...")
            system.train(years=years, real_time_vix=True, verbose=False)
            print("   ✅ VIX + 15 Anomaly Detectors trained")
            
            self._export_data(system)
            
            print("\n✅ Training complete")
            return system
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _export_data(self, system):
        """Export all data files (training time)."""
        
        print("\n📤 Exporting unified data contracts...")
        
        # Get cached anomaly result
        cached_anomaly_result = system._get_cached_anomaly_result()
        
        # Calculate persistence stats
        persistence_stats = system.vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
            system.vix_predictor.historical_ensemble_scores,
            dates=system.vix_predictor.features.index
        )
        
        # Initialize exporter
        exporter = UnifiedExporter(output_dir=str(self.json_dir))
        
        # Export all three unified files
        try:
            exporter.export_live_state(
                vix_predictor=system.vix_predictor,
                anomaly_result=cached_anomaly_result,
                spx=system.spx,
                persistence_stats=persistence_stats
            )
            print("   ✅ live_state.json")
        except Exception as e:
            print(f"   ❌ live_state.json: {e}")
        
        try:
            exporter.export_historical_context(
                vix_predictor=system.vix_predictor,
                spx=system.spx,
                historical_scores=system.vix_predictor.historical_ensemble_scores
            )
            print("   ✅ historical.json")
        except Exception as e:
            print(f"   ❌ historical.json: {e}")
        
        try:
            exporter.export_model_cache(
                vix_predictor=system.vix_predictor
            )
            print("   ✅ model_cache.pkl")
        except Exception as e:
            print(f"   ❌ model_cache.pkl: {e}")
        
        print("\n✅ Unified export complete")

    def _export_anomaly_metadata(self):
        """Export anomaly domain metadata."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "domains": {
                "vix_mean_reversion": {
                    "name": "VIX Mean Reversion",
                    "description": "Detects when VIX is stretched far from historical average.",
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
        """Export feature attribution for each anomaly domain."""
        try:
            features = system.vix_predictor.features.iloc[-1].to_dict()
            
            if cached_anomaly_result is not None:
                anomaly_result = cached_anomaly_result
            else:
                anomaly_result = system.vix_predictor.anomaly_detector.detect(
                    system.vix_predictor.features.iloc[[-1]], verbose=False
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
                                "feature": feature_name,
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
            print(f"   ⚠️ Feature attribution failed: {e}")
    
    def _verify_exports(self):
        """Verify all required JSON files exist and are valid."""
        print("\n" + "="*80 + "\nVERIFYING EXPORTS\n" + "="*80)
        all_valid = True
        
        for filename in self.REQUIRED_FILES:
            filepath = self.json_dir / filename
            
            if not filepath.exists():
                print(f"   ❌ {filename:30s} MISSING")
                all_valid = False
                continue
            
            try:
                if filename.endswith('.json'):
                    with open(filepath, 'r') as f:
                        json.load(f)
                size_kb = filepath.stat().st_size / 1024
                print(f"   ✅ {filename:30s} ({size_kb:6.1f} KB)")
            except json.JSONDecodeError:
                print(f"   ❌ {filename:30s} INVALID JSON")
                all_valid = False
            except Exception as e:
                print(f"   ❌ {filename:30s} ERROR: {e}")
                all_valid = False
        
        if not all_valid:
            print("\n⚠️ Some files missing/invalid")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        else:
            print("\n✅ All files valid")
    
    def _launch_dashboard(self, port: int):
        """Launch HTTP server and open dashboard in browser."""
        print("\n" + "="*80 + "\nLAUNCHING DASHBOARD\n" + "="*80)
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
            print(f"   ⚠️ Could not open browser: {e}")
            print(f"   Manually open: {dashboard_url}")
        
        print("\n" + "="*80)
        if self.refresh_running:
            print("ℹ️ Press Ctrl+C to stop server and auto-refresh")
            print(f"📊 Real-time VIX + SPX updating every {self.refresh_interval}s ({self.refresh_interval/60:.1f} min)")
            print("🔴 Anomaly scores recalculating with live data")
            print(f"🔄 Exponential backoff active (failures trigger delay increase)")
        else:
            print("ℹ️ Press Ctrl+C to stop server")
        print("="*80)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            if self.refresh_running:
                self._stop_refresh()
            print("✅ Server stopped")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dashboard Orchestrator for 15-Detector Anomaly System')
    parser.add_argument('--years', type=int, default=TRAINING_YEARS, help=f'Years of data (default: {TRAINING_YEARS})')
    parser.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    parser.add_argument('--skip-training', action='store_true', help='Skip training')
    parser.add_argument('--no-refresh', action='store_true', help='Disable auto-refresh')
    parser.add_argument('--refresh-interval', type=int, default=REFRESH_INTERVAL, 
                       help=f'Refresh interval in seconds (default: {REFRESH_INTERVAL} = {REFRESH_INTERVAL/60:.1f} min)')
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