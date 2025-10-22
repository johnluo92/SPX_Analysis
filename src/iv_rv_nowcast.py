"""
IV-RV NOWCAST SYSTEM
====================
Estimates current IV-RV spread using three methods:
1. Regression: ML model trained on historical relationships
2. Historical Analog: Find similar past conditions
3. VIX Term Structure: Use VIX9D/VIX ratio as proxy

Author: Claude + King John
Date: October 21, 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class IVRVNowcast:
    """
    Estimates current IV-RV spread using multiple methods.
    
    The challenge: IV-RV spread = VIX[t] - realized_vol[t:t+30]
    We can't know realized_vol[t:t+30] until 30 days pass.
    
    Solution: Use features available TODAY to predict what that spread will be.
    """
    
    def __init__(self, data_fetcher=None):
        """
        Args:
            data_fetcher: UnifiedDataFetcher instance (optional, will create if None)
        """
        self.data_fetcher = data_fetcher
        self.regression_model = None
        self.scaler = StandardScaler()
        self.historical_data = None
        self.vix_term_data = None
        
    def fetch_vix_term_structure(self, start_date, end_date):
        """Fetch VIX term structure data from Yahoo Finance."""
        tickers = ['^VIX', '^VIX9D', '^VIX3M', '^VIX6M']
        series_list = []
        
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    s = df['Close'].squeeze()
                    s.name = ticker.replace('^', '')
                    series_list.append(s)
            except Exception as e:
                print(f"  Warning: Could not fetch {ticker}: {e}")
        
        if not series_list:
            return pd.DataFrame()
        
        # Use concat like panel_iv_rv.py does
        term_df = pd.concat(series_list, axis=1, join='outer')
        term_df.index = pd.to_datetime(term_df.index).normalize()
        
        return term_df
    
    def calculate_actual_iv_rv_spread(self, vix_series, spx_series, window=30):
        """
        Calculate actual IV-RV spread for training.
        
        For each date t:
            spread[t] = VIX[t] - realized_vol(SPX[t:t+window])
        
        This is BACKWARD-LOOKING for training only.
        Uses log returns like panel_iv_rv.py does.
        """
        spreads = []
        dates = []
        
        # Align indices
        common_idx = vix_series.index.intersection(spx_series.index)
        vix_aligned = vix_series.loc[common_idx]
        spx_aligned = spx_series.loc[common_idx]
        
        for i in range(len(vix_aligned) - window):
            date = vix_aligned.index[i]
            vix_level = vix_aligned.iloc[i]
            
            # Calculate forward realized vol using log returns (matches panel_iv_rv.py)
            future_prices = spx_aligned.iloc[i:i+window]
            if len(future_prices) >= window - 5:  # Allow some missing data
                future_returns = np.log(future_prices / future_prices.shift(1))
                realized_vol = future_returns.std() * np.sqrt(252) * 100  # Annualized %
                spread = vix_level - realized_vol
                spreads.append(spread)
                dates.append(date)
            else:
                spreads.append(np.nan)
                dates.append(date)
        
        return pd.Series(spreads, index=dates, name='iv_rv_spread')
    
    def prepare_nowcast_features(self, df, date):
        """
        Prepare features available at 'date' to predict future IV-RV spread.
        
        Features (all knowable at date t):
        - VIX level, percentile, z-score
        - Realized vol (10d, 21d, 63d)
        - Yield curve slope and changes
        - VIX9D/VIX ratio (mean reversion indicator)
        - Macro momentum (Gold, Dollar, Oil)
        """
        features = {}
        
        # Get data up to date
        hist = df[df.index <= date].tail(100)
        if len(hist) < 63:
            return None
        
        # VIX features
        vix = hist['VIX'].iloc[-1]
        features['vix_level'] = vix
        features['vix_percentile'] = (hist['VIX'].iloc[-1] <= hist['VIX'].tail(252)).sum() / min(252, len(hist))
        features['vix_zscore'] = (vix - hist['VIX'].tail(63).mean()) / hist['VIX'].tail(63).std()
        features['vix_change_5d'] = hist['VIX'].iloc[-1] - hist['VIX'].iloc[-6] if len(hist) >= 6 else 0
        features['vix_change_21d'] = hist['VIX'].iloc[-1] - hist['VIX'].iloc[-22] if len(hist) >= 22 else 0
        
        # Realized vol features
        spx_returns = hist['SPX'].pct_change()
        features['realized_vol_10d'] = spx_returns.tail(10).std() * np.sqrt(252) * 100
        features['realized_vol_21d'] = spx_returns.tail(21).std() * np.sqrt(252) * 100
        features['realized_vol_63d'] = spx_returns.tail(63).std() * np.sqrt(252) * 100
        
        # VIX term structure (if available)
        if 'VIX9D' in hist.columns and not pd.isna(hist['VIX9D'].iloc[-1]):
            features['vix9d_vix_ratio'] = hist['VIX9D'].iloc[-1] / vix if vix > 0 else 1.0
            features['vix_contango'] = hist['VIX3M'].iloc[-1] - vix if 'VIX3M' in hist.columns else 0
        else:
            features['vix9d_vix_ratio'] = 1.0  # Neutral if no data
            features['vix_contango'] = 0
        
        # Yield curve
        if '10Y-2Y Yield Spread_level' in hist.columns:
            features['yield_slope'] = hist['10Y-2Y Yield Spread_level'].iloc[-1]
            features['yield_slope_change_21d'] = (hist['10Y-2Y Yield Spread_level'].iloc[-1] - 
                                                   hist['10Y-2Y Yield Spread_level'].iloc[-22] 
                                                   if len(hist) >= 22 else 0)
        else:
            features['yield_slope'] = 0
            features['yield_slope_change_21d'] = 0
        
        # Macro momentum
        if 'Gold' in hist.columns:
            features['gold_mom_21d'] = (hist['Gold'].iloc[-1] / hist['Gold'].iloc[-22] - 1) if len(hist) >= 22 else 0
        else:
            features['gold_mom_21d'] = 0
            
        if 'Dollar' in hist.columns:
            features['dollar_mom_21d'] = (hist['Dollar'].iloc[-1] / hist['Dollar'].iloc[-22] - 1) if len(hist) >= 22 else 0
        else:
            features['dollar_mom_21d'] = 0
        
        # SPX momentum
        features['spx_ret_21d'] = (hist['SPX'].iloc[-1] / hist['SPX'].iloc[-22] - 1) if len(hist) >= 22 else 0
        features['spx_ret_63d'] = (hist['SPX'].iloc[-1] / hist['SPX'].iloc[-64] - 1) if len(hist) >= 64 else 0
        
        return features
    
    def method_a_regression(self, train_features, train_targets):
        """
        Method A: Train regression model to predict IV-RV spread.
        
        Model learns: Given current market conditions, what will IV-RV spread be?
        """
        print("\n" + "="*60)
        print("METHOD A: REGRESSION NOWCAST")
        print("="*60)
        
        # Convert to arrays
        X = np.array([list(f.values()) for f in train_features])
        y = np.array(train_targets)
        
        # Remove any NaN rows
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            print("❌ Insufficient training data for regression")
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest regressor
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=30,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.regression_model.fit(X_scaled, y)
        
        # Evaluate on training set
        train_pred = self.regression_model.predict(X_scaled)
        train_mae = mean_absolute_error(y, train_pred)
        train_r2 = r2_score(y, train_pred)
        
        print(f"✓ Trained on {len(X)} samples")
        print(f"  MAE: {train_mae:.2f} vol points")
        print(f"  R²: {train_r2:.3f}")
        
        # Feature importance
        feature_names = list(train_features[0].keys())
        importances = self.regression_model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 predictive features:")
        for fname, imp in top_features:
            print(f"    {fname}: {imp*100:.1f}%")
        
        return self.regression_model
    
    def method_b_historical_analog(self, current_features, historical_features, historical_targets, k=10, verbose=False):
        """
        Method B: Find similar historical periods and average their outcomes.
        
        Process:
        1. Measure "distance" between current market state and each historical state
        2. Find K nearest neighbors
        3. Average their actual IV-RV spreads
        """
        if verbose:
            print("\n" + "="*60)
            print("METHOD B: HISTORICAL ANALOG NOWCAST")
            print("="*60)
        
        if not historical_features or not historical_targets:
            if verbose:
                print("❌ No historical data available")
            return None
        
        # Convert to arrays
        hist_X = np.array([list(f.values()) for f in historical_features])
        hist_y = np.array(historical_targets)
        curr_X = np.array(list(current_features.values())).reshape(1, -1)
        
        # Remove NaN rows
        valid_mask = ~np.isnan(hist_X).any(axis=1) & ~np.isnan(hist_y)
        hist_X = hist_X[valid_mask]
        hist_y = hist_y[valid_mask]
        
        if len(hist_X) < k:
            if verbose:
                print(f"❌ Insufficient historical data (need {k}, have {len(hist_X)})")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        hist_X_scaled = scaler.fit_transform(hist_X)
        curr_X_scaled = scaler.transform(curr_X)
        
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((hist_X_scaled - curr_X_scaled)**2, axis=1))
        
        # Find K nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        nearest_spreads = hist_y[nearest_indices]
        
        # Weighted average (inverse distance weighting)
        weights = 1 / (nearest_distances + 1e-10)  # Avoid division by zero
        weights = weights / weights.sum()
        
        predicted_spread = np.sum(weights * nearest_spreads)
        
        if verbose:
            print(f"✓ Found {k} historical analogs")
            print(f"  Distance range: {nearest_distances.min():.2f} - {nearest_distances.max():.2f}")
            print(f"  Analog spreads: {nearest_spreads.min():.1f} to {nearest_spreads.max():.1f}")
            print(f"  Weighted estimate: {predicted_spread:.2f}")
            print(f"  Simple average: {nearest_spreads.mean():.2f}")
        
        return predicted_spread
    
    def method_c_vix_term_structure(self, vix, vix9d, vix3m, realized_vol_21d, verbose=False):
        """
        Method C: Use VIX term structure as proxy for IV-RV spread.
        
        Logic:
        - If VIX9D < VIX: Market expects vol to drop (backwardation) → Positive IV-RV spread likely
        - If VIX9D > VIX: Market expects vol to rise (contango) → Negative IV-RV spread likely
        - Adjust by current realized vol to anchor estimate
        """
        if verbose:
            print("\n" + "="*60)
            print("METHOD C: VIX TERM STRUCTURE NOWCAST")
            print("="*60)
        
        if pd.isna(vix9d):
            if verbose:
                print("❌ VIX9D data not available, using fallback")
            # Fallback: Assume spread = VIX - realized_vol
            estimate = vix - realized_vol_21d
            if verbose:
                print(f"  Fallback estimate: {estimate:.2f}")
            return estimate
        
        # Calculate term structure slope
        vix9d_ratio = vix9d / vix if vix > 0 else 1.0
        
        # Calculate contango/backwardation adjustment
        if vix9d_ratio < 0.95:  # Strong backwardation
            adjustment = 3.0  # Expect VIX to drop → positive spread
        elif vix9d_ratio < 1.0:  # Mild backwardation
            adjustment = 1.5
        elif vix9d_ratio > 1.05:  # Strong contango
            adjustment = -3.0  # Expect VIX to rise → negative spread
        elif vix9d_ratio > 1.0:  # Mild contango
            adjustment = -1.5
        else:  # Flat
            adjustment = 0
        
        # Base estimate: VIX minus realized vol
        base_spread = vix - realized_vol_21d
        
        # Adjust by term structure signal
        estimate = base_spread + adjustment
        
        if verbose:
            print(f"  VIX: {vix:.2f}")
            print(f"  VIX9D: {vix9d:.2f} (ratio: {vix9d_ratio:.3f})")
            if not pd.isna(vix3m):
                print(f"  VIX3M: {vix3m:.2f}")
            print(f"  Realized Vol 21d: {realized_vol_21d:.2f}")
            print(f"  Base spread: {base_spread:.2f}")
            print(f"  Term structure adjustment: {adjustment:+.2f}")
            print(f"  Final estimate: {estimate:.2f}")
        
        return estimate
    
    def ensemble_nowcast(self, estimates, verbose=False):
        """
        Combine all three methods with weighted average.
        
        Weights (can be tuned based on backtested accuracy):
        - Regression: 40%
        - Historical Analog: 35%
        - VIX Term Structure: 25%
        """
        if verbose:
            print("\n" + "="*60)
            print("ENSEMBLE NOWCAST")
            print("="*60)
        
        weights = {'regression': 0.40, 'analog': 0.35, 'term_structure': 0.25}
        
        valid_estimates = {k: v for k, v in estimates.items() if v is not None}
        
        if not valid_estimates:
            if verbose:
                print("❌ No valid estimates available")
            return None
        
        # Normalize weights if some methods failed
        total_weight = sum(weights[k] for k in valid_estimates.keys())
        normalized_weights = {k: weights[k] / total_weight for k in valid_estimates.keys()}
        
        # Calculate weighted average
        ensemble = sum(v * normalized_weights[k] for k, v in valid_estimates.items())
        
        if verbose:
            print("Method contributions:")
            for method, value in valid_estimates.items():
                weight = normalized_weights[method]
                print(f"  {method:15s}: {value:6.2f} (weight: {weight:.1%})")
            print(f"\n✓ Ensemble estimate: {ensemble:.2f}")
        
        return ensemble
    
    def prepare_training_data(self, data_fetcher, start_date, end_date):
        """
        Prepare full dataset with actual IV-RV spreads for training/validation.
        
        Returns:
            features_list: List of feature dicts
            targets_list: List of actual IV-RV spreads
            dates_list: List of dates
        """
        print("\n" + "="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        
        # Fetch all required data
        print("Fetching market data...")
        spx_data = data_fetcher.fetch_spx(start_date, end_date)
        spx = spx_data['Close'] if isinstance(spx_data, pd.DataFrame) else spx_data
        vix = data_fetcher.fetch_vix(start_date, end_date, source='yahoo')
        
        # Normalize timezones - strip all to naive
        spx.index = pd.to_datetime(spx.index).tz_localize(None)
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        
        # Fetch VIX term structure
        print("Fetching VIX term structure...")
        term_data = self.fetch_vix_term_structure(start_date, end_date)
        if not term_data.empty:
            term_data.index = pd.to_datetime(term_data.index).tz_localize(None)
        
        # Fetch macro data
        print("Fetching macro data...")
        try:
            macro = data_fetcher.fetch_macro(start_date, end_date)
            macro.index = pd.to_datetime(macro.index).tz_localize(None)
        except Exception as e:
            print(f"Warning: Could not fetch macro data: {e}")
            macro = pd.DataFrame()
        
        # Fetch FRED data
        print("Fetching FRED data...")
        try:
            fred = data_fetcher.fetch_fred_multiple(start_date, end_date)
            if not fred.empty:
                fred.index = pd.to_datetime(fred.index).tz_localize(None)
        except Exception as e:
            print(f"Warning: Could not fetch FRED data: {e}")
            fred = pd.DataFrame()
        
        # Combine into single DataFrame
        df = pd.DataFrame({
            'SPX': spx,
            'VIX': vix
        })
        
        # Add term structure
        if not term_data.empty:
            for col in term_data.columns:
                df[col] = term_data[col]
        
        # Add macro
        if not macro.empty:
            for col in macro.columns:
                df[col] = macro[col]
        
        # Add FRED
        if not fred.empty:
            for col in fred.columns:
                df[col] = fred[col]
        
        # Forward fill missing values
        df = df.ffill().dropna(subset=['SPX', 'VIX'])
        
        print(f"✓ Loaded {len(df)} days of data")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Columns: {len(df.columns)}")
        
        # Calculate actual IV-RV spreads
        print("\nCalculating actual IV-RV spreads...")
        actual_spreads = self.calculate_actual_iv_rv_spread(df['VIX'], df['SPX'], window=30)
        
        print(f"✓ Calculated {actual_spreads.notna().sum()} valid spreads")
        print(f"  Spread range: {actual_spreads.min():.2f} to {actual_spreads.max():.2f}")
        print(f"  Spread mean: {actual_spreads.mean():.2f}")
        
        # Prepare features for each date
        features_list = []
        targets_list = []
        dates_list = []
        
        for date in actual_spreads.index:
            if pd.notna(actual_spreads[date]):
                features = self.prepare_nowcast_features(df, date)
                if features is not None:
                    features_list.append(features)
                    targets_list.append(actual_spreads[date])
                    dates_list.append(date)
        
        print(f"✓ Prepared {len(features_list)} training samples")
        
        self.historical_data = {
            'features': features_list,
            'targets': targets_list,
            'dates': dates_list,
            'df': df
        }
        
        return features_list, targets_list, dates_list
    
    def validate_methods(self, train_split=0.8):
        """
        Backtest all three methods on historical data.
        
        Returns:
            Dictionary of MAE and R² scores for each method
        """
        if not self.historical_data:
            raise ValueError("Must call prepare_training_data() first")
        
        print("\n" + "="*70)
        print("VALIDATING NOWCAST METHODS")
        print("="*70)
        
        features = self.historical_data['features']
        targets = self.historical_data['targets']
        dates = self.historical_data['dates']
        
        # Time-series split
        split_idx = int(len(features) * train_split)
        
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        test_features = features[split_idx:]
        test_targets = targets[split_idx:]
        test_dates = dates[split_idx:]
        
        print(f"Train: {len(train_features)} samples ({dates[0].date()} to {dates[split_idx-1].date()})")
        print(f"Test:  {len(test_features)} samples ({dates[split_idx].date()} to {dates[-1].date()})")
        
        results = {}
        
        # Train Method A (Regression)
        print("\n" + "-"*70)
        self.method_a_regression(train_features, train_targets)
        
        # Test Method A
        if self.regression_model:
            test_X = np.array([list(f.values()) for f in test_features])
            test_X_scaled = self.scaler.transform(test_X)
            predictions_a = self.regression_model.predict(test_X_scaled)
            
            mae_a = mean_absolute_error(test_targets, predictions_a)
            r2_a = r2_score(test_targets, predictions_a)
            
            results['regression'] = {'MAE': mae_a, 'R2': r2_a, 'predictions': predictions_a}
            print(f"\n✓ Method A Test Results: MAE={mae_a:.2f}, R²={r2_a:.3f}")
        
        # Test Method B (Historical Analog)
        print("\n" + "-"*70)
        predictions_b = []
        for i, curr_feat in enumerate(test_features):
            pred = self.method_b_historical_analog(curr_feat, train_features, train_targets, k=10)
            predictions_b.append(pred if pred is not None else np.mean(train_targets))
            if i == 0:  # Only print first one to avoid spam
                print("")
        
        mae_b = mean_absolute_error(test_targets, predictions_b)
        r2_b = r2_score(test_targets, predictions_b)
        
        results['analog'] = {'MAE': mae_b, 'R2': r2_b, 'predictions': predictions_b}
        print(f"\n✓ Method B Test Results: MAE={mae_b:.2f}, R²={r2_b:.3f}")
        
        # Test Method C (Term Structure)
        print("\n" + "-"*70)
        predictions_c = []
        df = self.historical_data['df']
        
        for date in test_dates:
            hist = df[df.index <= date].tail(30)
            if len(hist) >= 21:
                vix = hist['VIX'].iloc[-1]
                vix9d = hist['VIX9D'].iloc[-1] if 'VIX9D' in hist.columns else np.nan
                vix3m = hist['VIX3M'].iloc[-1] if 'VIX3M' in hist.columns else np.nan
                
                spx_returns = hist['SPX'].pct_change().dropna()
                realized_vol = spx_returns.tail(21).std() * np.sqrt(252) * 100
                
                pred = self.method_c_vix_term_structure(vix, vix9d, vix3m, realized_vol)
                predictions_c.append(pred if pred is not None else 0)
            else:
                predictions_c.append(0)
            
            if len(predictions_c) == 1:  # Only print first one
                print("")
        
        mae_c = mean_absolute_error(test_targets, predictions_c)
        r2_c = r2_score(test_targets, predictions_c)
        
        results['term_structure'] = {'MAE': mae_c, 'R2': r2_c, 'predictions': predictions_c}
        print(f"\n✓ Method C Test Results: MAE={mae_c:.2f}, R²={r2_c:.3f}")
        
        # Test Ensemble
        print("\n" + "-"*70)
        predictions_ensemble = []
        for i in range(len(test_features)):
            estimates = {
                'regression': predictions_a[i] if 'regression' in results else None,
                'analog': predictions_b[i],
                'term_structure': predictions_c[i]
            }
            pred = self.ensemble_nowcast(estimates)
            predictions_ensemble.append(pred if pred is not None else 0)
            if i == 0:  # Only print first one
                print("")
        
        mae_ens = mean_absolute_error(test_targets, predictions_ensemble)
        r2_ens = r2_score(test_targets, predictions_ensemble)
        
        results['ensemble'] = {'MAE': mae_ens, 'R2': r2_ens, 'predictions': predictions_ensemble}
        print(f"\n✓ Ensemble Test Results: MAE={mae_ens:.2f}, R²={r2_ens:.3f}")
        
        # Summary table
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"{'Method':<20} {'MAE (vol pts)':<15} {'R²':<10} {'Winner'}")
        print("-"*70)
        
        best_mae = min(r['MAE'] for r in results.values())
        best_r2 = max(r['R2'] for r in results.values())
        
        for method, metrics in results.items():
            mae_star = "★" if metrics['MAE'] == best_mae else " "
            r2_star = "★" if metrics['R2'] == best_r2 else " "
            print(f"{method:<20} {metrics['MAE']:>6.2f} {mae_star:<8} {metrics['R2']:>6.3f} {r2_star}")
        
        return results
    
    def get_current_estimate(self, data_fetcher, method='ensemble'):
        """
        Get IV-RV spread estimate for TODAY.
        
        Args:
            data_fetcher: UnifiedDataFetcher instance
            method: 'regression', 'analog', 'term_structure', or 'ensemble'
            
        Returns:
            Estimated IV-RV spread
        """
        if not self.historical_data:
            raise ValueError("Must train first by calling prepare_training_data() and validate_methods()")
        
        # Fetch latest data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        
        df = self.historical_data['df'].copy()
        
        # Update with latest data
        try:
            latest_spx_data = data_fetcher.fetch_spx(start_date, end_date)
            latest_spx = latest_spx_data['Close'] if isinstance(latest_spx_data, pd.DataFrame) else latest_spx_data
            latest_vix = data_fetcher.fetch_vix(start_date, end_date, source='yahoo')
            latest_term = self.fetch_vix_term_structure(start_date, end_date)
            
            # Merge latest data
            for col, series in [('SPX', latest_spx), ('VIX', latest_vix)]:
                # Update existing dates and add new ones
                df[col] = df[col].combine_first(series)
            
            if not latest_term.empty:
                for col in latest_term.columns:
                    df[col] = df[col].combine_first(latest_term[col]) if col in df.columns else latest_term[col]
            
            df = df.ffill().sort_index()
            
        except Exception as e:
            print(f"Warning: Could not fetch latest data: {e}")
            print("Using most recent available data from training set")
        
        # Get current features
        today = df.index[-1]
        print(f"\nGenerating estimate for: {today.date()}")
        
        current_features = self.prepare_nowcast_features(df, today)
        
        if current_features is None:
            raise ValueError("Could not prepare current features")
        
        estimates = {}
        
        # Method A: Regression
        if method in ['regression', 'ensemble'] and self.regression_model:
            curr_X = np.array(list(current_features.values())).reshape(1, -1)
            curr_X_scaled = self.scaler.transform(curr_X)
            estimates['regression'] = self.regression_model.predict(curr_X_scaled)[0]
        
        # Method B: Historical Analog
        if method in ['analog', 'ensemble']:
            train_features = self.historical_data['features']
            train_targets = self.historical_data['targets']
            estimates['analog'] = self.method_b_historical_analog(
                current_features, train_features, train_targets, k=10, verbose=False
            )
        
        # Method C: Term Structure
        if method in ['term_structure', 'ensemble']:
            hist = df[df.index <= today].tail(30)
            vix = hist['VIX'].iloc[-1]
            vix9d = hist['VIX9D'].iloc[-1] if 'VIX9D' in hist.columns else np.nan
            vix3m = hist['VIX3M'].iloc[-1] if 'VIX3M' in hist.columns else np.nan
            
            spx_returns = np.log(hist['SPX'] / hist['SPX'].shift(1)).dropna()
            realized_vol = spx_returns.tail(21).std() * np.sqrt(252) * 100
            
            estimates['term_structure'] = self.method_c_vix_term_structure(
                vix, vix9d, vix3m, realized_vol, verbose=False
            )
        
        if method == 'ensemble':
            return self.ensemble_nowcast(estimates, verbose=False)
        else:
            return estimates.get(method)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    from UnifiedDataFetcher import UnifiedDataFetcher
    
    # Initialize
    fetcher = UnifiedDataFetcher()
    nowcast = IVRVNowcast(fetcher)
    
    # Prepare training data (7 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*7)).strftime('%Y-%m-%d')
    
    features, targets, dates = nowcast.prepare_training_data(fetcher, start_date, end_date)
    
    # Validate all methods
    results = nowcast.validate_methods(train_split=0.8)
    
    # Get current estimate
    print("\n" + "="*70)
    print("CURRENT IV-RV SPREAD ESTIMATE")
    print("="*70)
    
    current_estimate = nowcast.get_current_estimate(fetcher, method='ensemble')
    print(f"\n✓ Current IV-RV Spread: {current_estimate:.2f}")
    
    if current_estimate > 2:
        print("  → VIX is OVERPRICED relative to expected realized vol")
        print("  → Strategy: SELL PREMIUM (short vega)")
    elif current_estimate < -2:
        print("  → VIX is UNDERPRICED relative to expected realized vol")
        print("  → Strategy: BUY PROTECTION (long vega)")
    else:
        print("  → VIX is FAIRLY PRICED")
        print("  → Strategy: NEUTRAL / WAIT")