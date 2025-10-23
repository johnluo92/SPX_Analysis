"""
SPX Live Dashboard - Python Server
Generates predictions and serves HTML dashboard

Usage:
    python spx_dashboard.py
    
Then open browser to: http://localhost:8000
"""

import json
from datetime import datetime
from pathlib import Path
import http.server
import socketserver
import webbrowser
import threading

from spx_predictor import SPXPredictor
from UnifiedDataFetcher import UnifiedDataFetcher
from cache_cleaner import CacheCleaner


class DashboardGenerator:
    """Generate dashboard data from SPX predictor."""
    
    def __init__(self):
        self.predictor = None
        self.spx = None
        self.vix = None
    
    def calculate_calendar_days(self, trading_days: int) -> int:
        """
        Calculate exact calendar days (DTE) for a given number of trading days.
        Uses actual market calendar to account for weekends and holidays.
        
        Args:
            trading_days: Number of trading days forward
            
        Returns:
            Approximate calendar days (DTE for options)
        """
        if self.spx is None:
            # Rough estimate if no data
            return int(trading_days * 1.4)
        
        # Get the last date in our data
        last_date = self.spx.index[-1]
        
        # Count forward 'trading_days' in our actual data
        # This accounts for real weekends and holidays
        if trading_days < len(self.spx):
            # Use historical data to measure actual calendar span
            reference_date = self.spx.index[-trading_days-1]
            calendar_span = (last_date - reference_date).days
            return calendar_span
        else:
            # For longer windows, use average ratio
            # Typically: 21 trading days = ~30 calendar days (1.43x)
            return int(trading_days * 1.43)
        
    def train_model(self):
        """Train the predictor model."""
        print("\n" + "="*70)
        print("TRAINING SPX PREDICTOR FOR DASHBOARD")
        print("="*70)
        
        self.predictor = SPXPredictor()
        self.predictor.train(years=7)
        
        # Get current data
        fetcher = UnifiedDataFetcher()
        end_date = datetime.now()
        start_date = datetime(end_date.year - 7, end_date.month, end_date.day)
        
        spx_df = fetcher.fetch_spx(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        self.spx = spx_df['Close'].squeeze()
        self.spx.index = self.spx.index.tz_localize(None)
        
        vix = fetcher.fetch_vix(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        self.vix = vix
        
        print("\n✅ Model trained and ready!")
        
    def get_confidence_label(self, prob):
        """Get confidence label from probability."""
        if prob >= 0.70:
            return "HIGH"
        elif prob >= 0.55:
            return "MEDIUM"
        else:
            return "NEUTRAL"
    
    def get_top_features(self, n=5):
        """Extract top N features from trained model - from FEATURE SELECTION results."""
        if self.predictor is None or self.predictor.model is None:
            return {}
        
        # CRITICAL FIX: Extract from the feature selection results stored in the model
        # This is what gets printed during training as "TOP 30 FEATURES BY IMPORTANCE"
        if hasattr(self.predictor.model, 'feature_importances_') and self.predictor.model.feature_importances_ is not None:
            # Use the feature selection importances
            feature_names = self.predictor.model.selected_features
            importances = self.predictor.model.feature_importances_
            
            # Create sorted dictionary
            feature_importance = dict(zip(feature_names, importances))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N as dictionary with rounded values
            return {feat: round(imp, 4) for feat, imp in sorted_features[:n]}
        
        # Fallback to 21d model if feature selection results not available
        model_21d = self.predictor.model.directional_models.get('21d')
        if model_21d is None:
            return {}
        
        feature_names = self.predictor.model.selected_features if self.predictor.model.selected_features else self.predictor.features.columns.tolist()
        importances = model_21d.feature_importances_
        
        feature_importance = dict(zip(feature_names, importances))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {feat: round(imp, 4) for feat, imp in sorted_features[:n]}
    
    def get_model_health(self):
        """Extract real model health metrics from training results."""
        if self.predictor is None or self.predictor.model is None:
            return {
                "status": "NOT_TRAINED",
                "message": "Model not trained"
            }
        
        # Get results from all models
        results = self.predictor.model.results
        
        # Calculate aggregate metrics
        test_accs = [r['test_acc'] for r in results.values() if 'test_acc' in r]
        gaps = [r['gap'] for r in results.values() if 'gap' in r]
        
        if not test_accs or not gaps:
            return {
                "status": "INCOMPLETE",
                "message": "No model metrics available"
            }
        
        avg_test_acc = sum(test_accs) / len(test_accs)
        avg_gap = sum(gaps) / len(gaps)
        std_dev = (sum((x - avg_test_acc) ** 2 for x in test_accs) / len(test_accs)) ** 0.5
        
        # Determine status based on realistic thresholds
        if avg_test_acc >= 0.85 and avg_gap <= 0.10:
            status = "STRONG"
        elif avg_test_acc >= 0.75 and avg_gap <= 0.15:
            status = "GOOD"
        elif avg_test_acc >= 0.65:
            status = "FAIR"
        else:
            status = "WEAK"
        
        return {
            "test_accuracy": round(avg_test_acc, 3),
            "std_dev": round(std_dev, 3),
            "gap": round(avg_gap, 3),
            "status": status,
            "message": f"Avg Accuracy: {avg_test_acc:.1%} ± {std_dev:.1%} • Gap: {avg_gap:+.1%}"
        }
    
    def calculate_strikes(self, spx_price, pct_width=0.05):
        """Calculate strike prices for iron condor."""
        lower_short = int(spx_price * (1 - pct_width))
        lower_long = lower_short - 5
        upper_short = int(spx_price * (1 + pct_width))
        upper_long = upper_short + 5
        
        # Round to nearest 5
        lower_short = round(lower_short / 5) * 5
        lower_long = round(lower_long / 5) * 5
        upper_short = round(upper_short / 5) * 5
        upper_long = round(upper_long / 5) * 5
        
        return f"{lower_long}/{lower_short}/{upper_short}/{upper_long}"
    
    def generate_dashboard_data(self):
        """Generate JSON data for dashboard."""
        fetcher = UnifiedDataFetcher()
        if self.predictor is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Import config to get available horizons
        from config import SPX_FORWARD_WINDOWS, SPX_RANGE_THRESHOLDS
        
        # Get current predictions
        predictions = self.predictor.predict_current()
        
        # Current market data
        current_spx_model = float(self.spx.iloc[-1])  # Last historical price used by model
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S ET")  # Add timestamp
        
        # Fetch real-time SPX and VIX prices
        current_spx_realtime = fetcher.fetch_price('^GSPC')
        if current_spx_realtime is None:
            print("⚠️  Could not fetch real-time SPX price, using model price")
            current_spx_realtime = current_spx_model
            
        current_vix = fetcher.fetch_price('^VIX')
        if current_vix is None:
            print("⚠️  Could not fetch real-time VIX price, using model price")
            current_vix = float(self.vix.iloc[-1])
        
        # DYNAMIC: Format directional predictions from config
        directional = {}
        for window in SPX_FORWARD_WINDOWS:
            key = f"{window}d"
            prob = float(predictions.get(f'direction_{key}', 0.5))
            directional[key] = {
                "prob": prob,
                "confidence": self.get_confidence_label(prob)
            }
        
        # DYNAMIC: Format range predictions from config
        range_bound = {}
        for window in SPX_FORWARD_WINDOWS:
            window_key = f"{window}d"
            range_bound[window_key] = {}
            
            for threshold in SPX_RANGE_THRESHOLDS:
                threshold_key = f"{int(threshold*100)}pct"
                pred_key = f"range_{window}d_{int(threshold*100)}pct"
                range_bound[window_key][threshold_key] = float(predictions.get(pred_key, 0.5))
        
        # Available options for dashboard (from config)
        available_horizons = [f"{w}d" for w in SPX_FORWARD_WINDOWS]
        available_ranges = [f"{int(t*100)}pct" for t in SPX_RANGE_THRESHOLDS]
        
        # Calculate exact DTE (calendar days) for each trading day window
        dte_mapping = {}
        for window in SPX_FORWARD_WINDOWS:
            dte_mapping[f"{window}d"] = self.calculate_calendar_days(window)
        
        # Top features (DYNAMIC - from actual trained model)
        top_features = self.get_top_features(n=5)
        
        # Debug: Print feature importances to verify they match training output
        print(f"\n📊 Top Features for Dashboard:")
        for feat, imp in top_features.items():
            print(f"   {feat}: {imp*100:.2f}%")
        
        # Generate trade signals (use longest horizon for trades)
        longest_horizon = f"{max(SPX_FORWARD_WINDOWS)}d"
        prob_long = directional[longest_horizon]["prob"]
        
        # Find highest confidence range (typically ±5% or highest threshold)
        best_range_threshold = f"{int(max(SPX_RANGE_THRESHOLDS)*100)}pct"
        range_long_best = range_bound[longest_horizon].get(best_range_threshold, 0.5)
        
        trade_signals = []
        
        # Iron Condor recommendation
        if range_long_best >= 0.90:
            ic_strikes = self.calculate_strikes(current_spx_model, max(SPX_RANGE_THRESHOLDS))
            trade_signals.append({
                "type": "Iron Condor",
                "confidence": float(range_long_best),
                "action": "SELL",
                "dte": max(SPX_FORWARD_WINDOWS),
                "strikes": ic_strikes,
                "credit": "$2.45",
                "risk": "$252.55",
                "roi": "0.97%",
                "rationale": f"{range_long_best*100:.1f}% prob stays within ±{int(max(SPX_RANGE_THRESHOLDS)*100)}% range"
            })
        
        # Bull Put Spread recommendation
        bps_lower = int(current_spx_model * 0.97)
        bps_upper = bps_lower + 5
        bps_lower = round(bps_lower / 5) * 5
        bps_upper = round(bps_upper / 5) * 5
        
        if prob_long >= 0.65:
            trade_signals.append({
                "type": "Bull Put Spread",
                "confidence": float(prob_long),
                "action": "SELL",
                "dte": max(SPX_FORWARD_WINDOWS),
                "strikes": f"{bps_lower}/{bps_upper}",
                "credit": "$1.85",
                "risk": "$313.15",
                "roi": "0.59%",
                "rationale": f"{prob_long*100:.1f}% bullish probability"
            })
        else:
            trade_signals.append({
                "type": "Bull Put Spread",
                "confidence": float(prob_long),
                "action": "NEUTRAL",
                "dte": max(SPX_FORWARD_WINDOWS),
                "strikes": f"{bps_lower}/{bps_upper}",
                "credit": "$1.85",
                "risk": "$313.15",
                "roi": "0.59%",
                "rationale": f"Only {prob_long*100:.1f}% bullish - wait for 65%+ signal"
            })
        
        # Model health (DYNAMIC - from actual training results)
        model_health = self.get_model_health()
        
        # Assemble final data with DYNAMIC CONFIG
        data = {
            "current_date": current_date,
            "current_time": current_time,  # Add timestamp for display
            "spx_price": current_spx_realtime,  # Current real-time price for display
            "spx_price_model": current_spx_model,  # Price model used for predictions
            "vix": current_vix,
            "available_horizons": available_horizons,  # ← DYNAMIC from config
            "available_ranges": available_ranges,      # ← DYNAMIC from config
            "dte_mapping": dte_mapping,  # ← Trading days to calendar days (DTE)
            "directional": directional,
            "range_bound": range_bound,
            "top_features": top_features,
            "trade_signals": trade_signals,
            "model_health": model_health
        }
        
        return data
    
    def save_dashboard_data(self, filepath="dashboard_data.json"):
        """Save dashboard data to JSON file."""
        data = self.generate_dashboard_data()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Dashboard data saved to: {filepath}")
        return data


def create_dashboard_html():
    """Create standalone HTML file."""
    
    # Write HTML file
    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPX Live Dashboard</title>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <div id="root"></div>
    <script src="dashboard.js"></script>
</body>
</html>''')
    
    print("✅ Dashboard HTML created: dashboard.html")
    print("✅ Make sure dashboard.js is in the same directory")


class DashboardServer(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving dashboard."""
    
    def log_message(self, format, *args):
        """Suppress server logs."""
        pass


def serve_dashboard(port=8000):
    """Start HTTP server for dashboard."""
    handler = DashboardServer
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\n{'='*70}")
        print(f"🚀 DASHBOARD SERVER RUNNING")
        print(f"{'='*70}")
        print(f"\n📊 Open your browser to: http://localhost:{port}/dashboard.html")
        print(f"\n💡 Press Ctrl+C to stop the server\n")
        
        # Auto-open browser
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{port}/dashboard.html')).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Dashboard closed.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SPX Live Dashboard')
    parser.add_argument('--clean-cache', action='store_true',
                       help='Clean old cache files before starting (removes files older than 30 days)')
    parser.add_argument('--cache-days', type=int, default=30,
                       help='Max age of cache files to keep in days (default: 30)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SPX LIVE DASHBOARD - SETUP")
    print("="*70)
    
    # Optional: Clean old cache files
    if args.clean_cache:
        print("\n🧹 Cleaning old cache files...")
        cleaner = CacheCleaner()
        cleaner.clean_old_files(max_age_days=args.cache_days, verbose=True)
    
    # Step 1: Train model
    generator = DashboardGenerator()
    generator.train_model()
    
    # Step 2: Generate dashboard data
    print("\n📊 Generating dashboard data...")
    generator.save_dashboard_data()
    
    # Step 3: Create HTML file
    print("\n🎨 Creating dashboard HTML...")
    create_dashboard_html()
    
    # Step 4: Start server
    print("\n✅ Setup complete!")
    serve_dashboard(port=8000)


if __name__ == "__main__":
    main()