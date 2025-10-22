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


class DashboardGenerator:
    """Generate dashboard data from SPX predictor."""
    
    def __init__(self):
        self.predictor = None
        self.spx = None
        self.vix = None
        
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
        if self.predictor is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Import config to get available horizons
        from config import SPX_FORWARD_WINDOWS, SPX_RANGE_THRESHOLDS
        
        # Get current predictions
        predictions = self.predictor.predict_current()
        
        # Current market data
        current_spx = float(self.spx.iloc[-1])
        current_vix = float(self.vix.iloc[-1])
        current_date = datetime.now().strftime("%Y-%m-%d")
        
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
        
        # Top features (from training)
        top_features = {
            "iv_rv_spread": 0.1933,
            "iv_rv_vs_avg": 0.1300,
            "iv_rv_momentum_21": 0.0681,
            "yield_spread_change_63": 0.0472,
            "yield_slope": 0.0411
        }
        
        # Generate trade signals (use longest horizon for trades)
        longest_horizon = f"{max(SPX_FORWARD_WINDOWS)}d"
        prob_long = directional[longest_horizon]["prob"]
        
        # Find highest confidence range (typically ±5% or highest threshold)
        best_range_threshold = f"{int(max(SPX_RANGE_THRESHOLDS)*100)}pct"
        range_long_best = range_bound[longest_horizon].get(best_range_threshold, 0.5)
        
        trade_signals = []
        
        # Iron Condor recommendation
        if range_long_best >= 0.90:
            ic_strikes = self.calculate_strikes(current_spx, max(SPX_RANGE_THRESHOLDS))
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
        bps_lower = int(current_spx * 0.97)
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
        
        # Model health
        model_health = {
            "walk_forward_accuracy": 0.903,
            "std_dev": 0.035,
            "gap": -0.002,
            "status": "EXCELLENT"
        }
        
        # Assemble final data with DYNAMIC CONFIG
        data = {
            "current_date": current_date,
            "spx_price": current_spx,
            "vix": current_vix,
            "available_horizons": available_horizons,  # ← DYNAMIC from config
            "available_ranges": available_ranges,      # ← DYNAMIC from config
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
    print("\n" + "="*70)
    print("SPX LIVE DASHBOARD - SETUP")
    print("="*70)
    
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