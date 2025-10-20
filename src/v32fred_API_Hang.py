"""
v3.2 Runner - Uses existing cache + Yahoo (FRED is down)
Quick script to run your v3.2 model with cached FRED data + Yahoo for market data
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

# Import your existing model classes
from v32 import SectorRotationFeatures, SectorRotationModel


def fetch_data_hybrid():
    """Fetch data using cache + Yahoo (FRED is down)."""
    print("\n" + "="*70)
    print("HYBRID DATA FETCH - FRED CACHE + YAHOO FINANCE")
    print("="*70)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)  # 7 years
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\n📅 Date range: {start_str} to {end_str}")
    
    # 1. Fetch SECTOR ETFs from Yahoo
    print("\n🔄 Fetching sector ETFs from Yahoo Finance...")
    sector_tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC', 'SPY']
    
    sectors_data = {}
    for ticker in sector_tickers:
        try:
            print(f"   {ticker}...", end=" ", flush=True)
            df = yf.download(ticker, start=start_str, end=end_str, progress=False)
            sectors_data[ticker] = df['Adj Close']
            print(f"✅ {len(df)} rows")
        except Exception as e:
            print(f"❌ {str(e)[:30]}")
    
    sectors = pd.DataFrame(sectors_data)
    print(f"\n✅ Sectors: {sectors.shape}")
    
    # 2. Fetch MACRO factors from Yahoo
    print("\n🔄 Fetching macro factors from Yahoo Finance...")
    macro_tickers = {
        'GLD': 'Gold',
        'UUP': 'Dollar',
        '^TNX': '10Y Treasury',
        '^IRX': '2Y Treasury'
    }
    
    macro_data = {}
    for ticker, name in macro_tickers.items():
        try:
            print(f"   {name} ({ticker})...", end=" ", flush=True)
            df = yf.download(ticker, start=start_str, end=end_str, progress=False)
            
            if ticker.startswith('^'):
                macro_data[name] = df['Close']
            else:
                macro_data[name] = df['Adj Close']
            
            print(f"✅ {len(df)} rows")
        except Exception as e:
            print(f"❌ {str(e)[:30]}")
    
    macro = pd.DataFrame(macro_data)
    print(f"\n✅ Macro: {macro.shape}")
    
    # 3. Fetch VIX from Yahoo
    print("\n🔄 Fetching VIX from Yahoo Finance...")
    try:
        vix_df = yf.download('^VIX', start=start_str, end=end_str, progress=False)
        vix = vix_df['Close']
        print(f"✅ VIX: {len(vix)} rows")
    except Exception as e:
        print(f"❌ VIX error: {e}")
        vix = None
    
    # 4. FRED cache (we have it but v3.2 doesn't use it)
    print("\n💡 Note: FRED cache exists in ./cache/ but v3.2 doesn't use FRED data")
    
    return sectors, macro, vix


def align_data(sectors, macro, vix):
    """Align all data to common dates."""
    print("\n🔧 Aligning data to common dates...")
    
    common_idx = sectors.index
    
    if macro is not None:
        common_idx = common_idx.intersection(macro.index)
    
    if vix is not None:
        common_idx = common_idx.intersection(vix.index)
    
    sectors_aligned = sectors.loc[common_idx]
    macro_aligned = macro.loc[common_idx] if macro is not None else None
    vix_aligned = vix.loc[common_idx] if vix is not None else None
    
    print(f"✅ Aligned to {len(common_idx)} common dates")
    print(f"   Range: {common_idx.min().date()} to {common_idx.max().date()}")
    
    return sectors_aligned, macro_aligned, vix_aligned


def run_v32_model():
    """Run the full v3.2 model pipeline."""
    print("\n" + "="*70)
    print("SECTOR ROTATION MODEL v3.2 - RUNNING WITH HYBRID DATA")
    print("="*70)
    
    # Step 1: Fetch data
    print("\n📊 Step 1: Fetch Data")
    print("-"*70)
    sectors, macro, vix = fetch_data_hybrid()
    
    if sectors is None or sectors.empty:
        print("\n❌ Failed to fetch data. Exiting.")
        return None
    
    # Step 2: Align data
    print("\n📊 Step 2: Align Data")
    print("-"*70)
    sectors_aligned, macro_aligned, vix_aligned = align_data(sectors, macro, vix)
    
    # Step 3: Feature engineering
    print("\n📊 Step 3: Feature Engineering")
    print("-"*70)
    feat_eng = SectorRotationFeatures()
    features = feat_eng.combine_features(
        sectors_aligned,
        macro_aligned,
        vix_aligned
    )
    
    # Step 4: Create targets
    print("\n📊 Step 4: Create Targets")
    print("-"*70)
    model = SectorRotationModel()
    targets = model.create_targets(sectors_aligned, forward_window=21)
    
    # Step 5: Train models
    print("\n📊 Step 5: Train Models")
    print("-"*70)
    results = model.train_models(features, targets, use_feature_selection=True)
    
    # Step 6: Validation
    print("\n📊 Step 6: Walk-Forward Validation")
    print("-"*70)
    validation = model.walk_forward_validate(features, targets, n_splits=5)
    
    # Step 7: Confidence scoring
    print("\n📊 Step 7: Confidence Scoring")
    print("-"*70)
    confidence = model.calculate_confidence_scores()
    
    # Step 8: Summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results).T
    print("\n📊 Performance Metrics:")
    print(results_df[['test_accuracy', 'overfitting_gap']].round(3))
    
    # Gap analysis
    print("\n" + "="*70)
    print("GAP THRESHOLD ANALYSIS (Target: <0.20)")
    print("="*70)
    
    excellent = results_df[results_df['overfitting_gap'] < 0.20]
    good = results_df[(results_df['overfitting_gap'] >= 0.20) & 
                      (results_df['overfitting_gap'] < 0.30)]
    poor = results_df[results_df['overfitting_gap'] >= 0.30]
    
    print(f"\n✅ Excellent (Gap <0.20): {len(excellent)} sectors")
    if len(excellent) > 0:
        print(excellent[['test_accuracy', 'overfitting_gap']].to_string())
    
    print(f"\n🟡 Good (Gap 0.20-0.30): {len(good)} sectors")
    if len(good) > 0:
        print(good[['test_accuracy', 'overfitting_gap']].to_string())
    
    print(f"\n⚠️  Poor (Gap >0.30): {len(poor)} sectors")
    if len(poor) > 0:
        print(poor[['test_accuracy', 'overfitting_gap']].to_string())
    
    # Step 9: Current predictions
    print("\n" + "="*70)
    print("CURRENT ROTATION PROBABILITIES (Next 21 Days)")
    print("="*70)
    
    current_features = features.iloc[[-1]]
    current_probs = model.predict_probabilities(current_features)
    
    prob_series = current_probs.T.iloc[:, 0]
    summary = pd.DataFrame({
        'Probability': prob_series,
        'Test_Acc': results_df['test_accuracy'],
        'Gap': results_df['overfitting_gap'],
        'Confidence': confidence.set_index('Sector')['Tier']
    })
    
    summary = summary.sort_values('Probability', ascending=False)
    
    print("\n📊 Ranked by Rotation Probability:")
    print(summary.to_string())
    
    print("\n" + "="*70)
    print("✅ v3.2 COMPLETE - HYBRID DATA SUCCESSFUL")
    print("="*70)
    print("\n💡 Data sources:")
    print("   ✅ Sector ETFs: Yahoo Finance")
    print("   ✅ Macro factors: Yahoo Finance")
    print("   ✅ VIX: Yahoo Finance")
    print("   ⏸️  FRED: Skipped (service down, not needed for v3.2)")
    
    return model, features, results, validation, confidence


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STARTING v3.2 MODEL RUN")
    print("="*70)
    print("\n💡 Strategy:")
    print("   • Yahoo Finance for all market data (working)")
    print("   • FRED cache available but not used (v3.2 design)")
    print("   • Should complete in ~2-3 minutes")
    print("\n🔄 Beginning fetch...\n")
    
    result = run_v32_model()
    
    if result is not None:
        print("\n" + "="*70)
        print("✅ SUCCESS - Model trained and ready!")
        print("="*70)
    else:
        print("\n❌ Failed to complete model training")