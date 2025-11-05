"""Generate Comprehensive Data Package for Claude LLM Analysis
   
   Usage:
     from generate_claude_package import generate_claude_package
     
     # After training your system:
     system = IntegratedMarketSystemV4()
     system.train(years=10)
     
     # Generate package:
     package = generate_claude_package(system)
     
     # Now copy/paste claude_intelligence_package.json to Claude.ai
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def _format_feature_name(raw_name: str) -> str:
    """Make feature names human-readable."""
    replacements = {
        'vix_vs_ma': 'VIX vs MA',
        'vix_zscore': 'VIX Z-Score',
        'spx_vs_ma': 'SPX vs MA',
        'spx_ret': 'SPX Return',
        'spx_momentum_z': 'SPX Momentum',
        'Treasury_10Y': '10Y Treasury',
        '_mom_': ' Momentum ',
        '_zscore_': ' Z-Score ',
        '_pct': '%',
        '_level': '',
        '_change': ' Change'
    }
    
    formatted = raw_name
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    formatted = formatted.replace('_', ' ').strip()
    return formatted


def _extract_detector_scores(anomaly_result: dict) -> dict:
    """Extract all detector scores from anomaly result."""
    scores = {}
    
    for name, data in anomaly_result.get('domain_anomalies', {}).items():
        scores[name] = float(data['score'])
    
    for name, data in anomaly_result.get('random_anomalies', {}).items():
        scores[name] = float(data['score'])
    
    return scores


def _build_historical_lineage(system, lookback_days: int = 30) -> dict:
    """Build daily detector scores for last N days."""
    features = system.vix_predictor.features
    detector = system.vix_predictor.anomaly_detector
    
    start_idx = max(0, len(features) - lookback_days)
    lineage = {}
    
    print(f"   Building {lookback_days}-day historical lineage...")
    
    for i in range(start_idx, len(features)):
        try:
            date_str = features.index[i].strftime('%Y-%m-%d')
            result = detector.detect(features.iloc[[i]], verbose=False)
            
            day_scores = {
                'ensemble': float(result['ensemble']['score']),
                'ensemble_std': float(result['ensemble']['std'])
            }
            
            # Add all detector scores
            for name, data in result.get('domain_anomalies', {}).items():
                day_scores[name] = float(data['score'])
            for name, data in result.get('random_anomalies', {}).items():
                day_scores[name] = float(data['score'])
            
            lineage[date_str] = day_scores
            
        except Exception as e:
            print(f"   ⚠️  Failed for {date_str}: {e}")
            continue
    
    return lineage


def _find_inflection_points(system, n_points: int = 5) -> List[dict]:
    """Find historical HIGH anomaly episodes and their outcomes."""
    features = system.vix_predictor.features
    ensemble_history = system.vix_predictor.historical_ensemble_scores
    
    if ensemble_history is None or len(ensemble_history) == 0:
        return []
    
    # Get threshold
    if hasattr(system.vix_predictor.anomaly_detector, 'statistical_thresholds'):
        thresholds = system.vix_predictor.anomaly_detector.statistical_thresholds
        high_threshold = thresholds.get('high', 0.78)
    else:
        high_threshold = 0.78
    
    inflection_points = []
    high_mask = ensemble_history >= high_threshold
    
    # Find episode starts
    in_episode = False
    for i in range(len(high_mask)):
        if high_mask[i] and not in_episode:
            date = features.index[i]
            score = ensemble_history[i]
            
            # Calculate outcome (SPX change over next 10 days)
            outcome = "Recent event (no outcome yet)"
            if i + 10 < len(system.spx):
                spx_start = system.spx.iloc[i]
                spx_end = system.spx.iloc[i + 10]
                pct_change = (spx_end / spx_start - 1) * 100
                
                if abs(pct_change) > 0.1:
                    direction = "dropped" if pct_change < 0 else "rose"
                    outcome = f"SPX {direction} {abs(pct_change):.1f}% over 10 days"
            
            inflection_points.append({
                'date': date.strftime('%Y-%m-%d'),
                'ensemble_score': float(score),
                'outcome': outcome,
                'vix_level': float(system.vix_predictor.vix.iloc[i]) if i < len(system.vix_predictor.vix) else None
            })
            
            in_episode = True
        elif not high_mask[i]:
            in_episode = False
    
    # Return last N points
    return inflection_points[-n_points:] if len(inflection_points) > n_points else inflection_points


def _extract_feature_drivers(system, anomaly_result: dict, top_n: int = 5) -> dict:
    """Extract top features driving each detector."""
    detector = system.vix_predictor.anomaly_detector
    current_features = system.vix_predictor.features.iloc[-1].to_dict()
    
    feature_drivers = {}
    detector_scores = _extract_detector_scores(anomaly_result)
    
    for detector_name in detector_scores.keys():
        if detector_name not in detector.feature_importances:
            continue
        
        importances = detector.feature_importances[detector_name]
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        driver_list = []
        for feat_name, importance in sorted_features:
            value = current_features.get(feat_name, 0)
            
            if not np.isnan(value) and np.isfinite(value):
                driver_list.append({
                    'feature': feat_name,
                    'feature_display': _format_feature_name(feat_name),
                    'value': float(value),
                    'importance': float(importance)
                })
        
        if driver_list:
            feature_drivers[detector_name] = driver_list
    
    return feature_drivers


def generate_claude_package(system, output_path: str = './json_data/claude_intelligence_package.json') -> dict:
    """
    Generate comprehensive JSON package for Claude LLM analysis.
    
    Args:
        system: Trained IntegratedMarketSystemV4 instance
        output_path: Where to save the JSON package
    
    Returns:
        dict: The complete package (also saved to file)
    """
    
    if not system.trained:
        raise ValueError("System must be trained before generating package")
    
    print(f"\n{'='*80}")
    print("GENERATING CLAUDE INTELLIGENCE PACKAGE")
    print(f"{'='*80}")
    
    # Get current anomaly result
    anomaly_result = system._get_cached_anomaly_result(force_refresh=False)
    
    # Build package sections
    print("\n[1/6] Current market state...")
    current_vix = float(system.vix_predictor.vix.iloc[-1])
    current_spx = float(system.spx.iloc[-1])
    
    current_state = {
        'market': {
            'vix': current_vix,
            'vix_regime': system._classify_vix_regime(current_vix)['name'],
            'vix_regime_id': system._classify_vix_regime(current_vix)['id'],
            'spx': current_spx,
            'spx_change_1d': float(system.spx.pct_change(1).iloc[-1] * 100),
            'spx_change_5d': float(system.spx.pct_change(5).iloc[-1] * 100),
            'date': system.vix_predictor.features.index[-1].strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat()
        },
        'ensemble': {
            'score': float(anomaly_result['ensemble']['score']),
            'percentile': int(anomaly_result['ensemble']['score'] * 100),
            'severity': system._classify_severity(anomaly_result['ensemble']['score']),
            'std': float(anomaly_result['ensemble']['std']),
            'n_active_detectors': int(anomaly_result['ensemble']['n_detectors'])
        }
    }
    
    print("[2/6] Current detector scores...")
    detector_scores_current = _extract_detector_scores(anomaly_result)
    
    print("[3/6] Historical lineage (30 days)...")
    historical_lineage = _build_historical_lineage(system, lookback_days=30)
    
    print("[4/6] Key inflection points...")
    inflection_points = _find_inflection_points(system, n_points=5)
    
    print("[5/6] Feature drivers...")
    feature_drivers = _extract_feature_drivers(system, anomaly_result, top_n=5)
    
    print("[6/6] Assembling package...")
    
    # Statistical thresholds
    if hasattr(system.vix_predictor.anomaly_detector, 'statistical_thresholds'):
        thresholds = system.vix_predictor.anomaly_detector.statistical_thresholds
        threshold_summary = {
            'moderate': thresholds.get('moderate', 0.70),
            'high': thresholds.get('high', 0.78),
            'critical': thresholds.get('critical', 0.88)
        }
    else:
        threshold_summary = {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}
    
    # Assemble final package
    package = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'system_version': 'v4.0',
            'package_type': 'claude_intelligence',
            'thresholds': threshold_summary
        },
        
        'current_state': current_state,
        
        'detector_scores': {
            'current': detector_scores_current,
            'historical_lineage': historical_lineage,
            'key_inflection_points': inflection_points
        },
        
        'feature_drivers': feature_drivers,
        
        'trading_context': {
            'strategy': 'Bull put spreads + iron condors (short volatility)',
            'typical_positions': {
                'delta_bias': 'short (~80%)',
                'vega_exposure': 'short',
                'theta_collection': 'primary income',
                'dte_preference': '45-90 days'
            },
            'risk_factors': [
                'VIX spikes (short vega)',
                'SPX corrections (short delta)',
                'Gap risk (overnight moves)',
                'Regime transitions (volatility expansion)'
            ]
        }
    }
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(package, f, indent=2)
    
    # Print summary
    file_size_kb = output_path.stat().st_size / 1024
    
    print(f"\n{'='*80}")
    print("✅ PACKAGE GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📦 Output: {output_path}")
    print(f"   Size: {file_size_kb:.1f} KB")
    print(f"   Historical days: {len(historical_lineage)}")
    print(f"   Inflection points: {len(inflection_points)}")
    print(f"   Detectors tracked: {len(detector_scores_current)}")
    print(f"   Feature drivers: {len(feature_drivers)}")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Open: {output_path}")
    print(f"   2. Copy entire JSON contents")
    print(f"   3. Paste to Claude.ai with analysis prompt")
    print(f"   4. Save Claude's report to: ./reports/YYYY-MM-DD_HH-MM.txt")
    print(f"\n{'='*80}\n")
    
    return package


def generate_analysis_prompt() -> str:
    """Generate the prompt to give Claude along with the package."""
    
    prompt = """You are my quantitative market analyst. Analyze the attached JSON data package and produce a concise 1-page intelligence report.

CONTEXT:
- I trade short volatility (bull put spreads, iron condors)
- Typical exposure: 80% short delta, collecting theta daily
- I need actionable intelligence: should I reduce, hold, or add exposure?

REPORT STRUCTURE (keep to 1 page total):

## 1. CURRENT ASSESSMENT (3-4 sentences)
- What's happening right now in the market?
- Interpret the ensemble anomaly score in plain English
- Which detectors are firing and why does it matter?

## 2. HISTORICAL CONTEXT (2-3 sentences)
- How does this setup compare to past patterns?
- Reference the key inflection points provided
- What happened the last time we saw this configuration?

## 3. FEATURE ANALYSIS (2-3 sentences)
- What underlying features are driving the high detector signals?
- Are these signals meaningful or just noise?
- Which feature drivers should I pay attention to?

## 4. TRADING RECOMMENDATION (3-4 sentences)
- Specific action: REDUCE / HOLD / ADD exposure
- Confidence level: HIGH / MEDIUM / LOW
- What specific thresholds or signals should I watch for next?
- Timeline: how urgent is this?

## 5. RISK/REWARD ASSESSMENT (2 sentences)
- For my short volatility positions, what's the risk/reward right now?
- What's my worst-case scenario if I'm wrong?

GUIDELINES:
- Use plain language, not academic jargon
- Be direct and actionable
- Reference specific detector names and scores
- Compare current readings to historical patterns
- Focus on what matters for short vol trading

Total length: ~300-400 words (1 page when printed)
"""
    
    return prompt


# Integration example
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CLAUDE INTELLIGENCE PACKAGE GENERATOR")
    print("="*80)
    print("\nThis script generates a comprehensive JSON package for Claude to analyze.")
    print("\nINTEGRATION:")
    print("  1. Add to your main script after training:")
    print("     from generate_claude_package import generate_claude_package")
    print("     package = generate_claude_package(system)")
    print("\n  2. The JSON will be saved to: ./json_data/claude_intelligence_package.json")
    print("\n  3. Copy/paste the JSON + analysis prompt to Claude.ai")
    print("\n  4. Claude generates your intelligence report")
    print("\n" + "="*80)
    
    # Print the analysis prompt
    print("\n📋 ANALYSIS PROMPT TO USE WITH PACKAGE:\n")
    print(generate_analysis_prompt())
    print("\n" + "="*80 + "\n")
