#!/usr/bin/env python3
"""
Analyze Firing Random Detectors
================================
Standalone script to identify and explain which random detectors are firing.

Usage:
    python analyze_firing_detectors.py
"""

import json
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from integrated_system_production import IntegratedMarketSystemV4

# Import RandomDetectorAnalyzer from the standalone file
try:
    from random_detector_analyzer import RandomDetectorAnalyzer
except ImportError:
    # If not available as separate file, we'll define it inline
    print("⚠️  random_detector_analyzer.py not found in path")
    print("   Please ensure random_detector_analyzer.py is in the same directory")
    sys.exit(1)


def load_latest_anomaly_result():
    """Load the most recent anomaly detection result."""
    json_path = Path('./json_data/anomaly_report.json')
    
    if not json_path.exists():
        print(f"❌ Could not find {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_firing_detectors(threshold=0.85):
    """Main analysis function."""
    
    print("="*80)
    print("🔍 FIRING RANDOM DETECTOR ANALYSIS")
    print("="*80)
    
    # Load trained system
    print("\n📊 Loading integrated system...")
    system = IntegratedMarketSystemV4()
    
    # Load from cache if available
    try:
        system.load_models()
        print("✅ Loaded models from cache")
    except:
        print("⚠️  No cached models found, training new system...")
        system.train(years=15, verbose=True)
    
    # Get anomaly detector
    anomaly_detector = system.vix_predictor.anomaly_detector
    
    # Initialize analyzer
    analyzer = RandomDetectorAnalyzer(anomaly_detector)
    
    # Load latest anomaly result
    anomaly_result = load_latest_anomaly_result()
    
    if not anomaly_result:
        print("❌ No anomaly result found. Run the system first to generate results.")
        return
    
    print(f"\n📅 Analysis Timestamp: {anomaly_result.get('timestamp', 'Unknown')}")
    print(f"🎯 Ensemble Score: {anomaly_result['ensemble']['score']:.3f}")
    
    # Find firing random detectors
    print(f"\n🔥 FIRING RANDOM DETECTORS (threshold > {threshold}):")
    print("-" * 80)
    
    firing_randoms = []
    
    if 'top_anomalies' in anomaly_result:
        for detector in anomaly_result['top_anomalies']:
            if detector['name'].startswith('random_') and detector['score'] > threshold:
                firing_randoms.append(detector)
                print(f"  ⚡ {detector['name'].upper()}: {detector['score']:.3f} ({detector['score']*100:.1f}%)")
    
    if not firing_randoms:
        print(f"  ✅ No random detectors firing above {threshold} threshold")
        print(f"\n💡 Try lowering threshold. Current top random detectors:")
        for detector in anomaly_result.get('top_anomalies', [])[:5]:
            if detector['name'].startswith('random_'):
                print(f"     • {detector['name']}: {detector['score']:.3f}")
        return
    
    # Analyze each firing detector
    print("\n" + "="*80)
    print("📋 DETAILED COMPOSITION ANALYSIS")
    print("="*80)
    
    for detector_info in firing_randoms:
        detector_name = detector_info['name']
        score = detector_info['score']
        
        print(f"\n{'='*80}")
        print(f"🎲 {detector_name.upper()} - Score: {score:.3f} ({score*100:.1f}%)")
        print(f"{'='*80}")
        
        analysis = analyzer.analyze_single_detector(detector_name)
        
        print(f"\n📝 Description:")
        print(f"   {analysis['description']}")
        
        print(f"\n🎯 Interpretation:")
        print(f"   {analysis['interpretation']}")
        
        print(f"\n🏗️  Feature Composition ({analysis['feature_count']} total features):")
        for category, info in sorted(
            analysis['categorization'].items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        ):
            if info['count'] > 0:
                print(f"   • {category:15s}: {info['count']:2d} features ({info['percentage']:5.1f}%)")
        
        print(f"\n⭐ Top 5 Most Important Features:")
        for i, feat in enumerate(analysis['top_features'][:5], 1):
            imp = feat.get('importance')
            imp_str = f"{imp:6.1%}" if imp is not None else "  N/A "
            readable = feat.get('readable_name', feat['feature'])
            category = feat['category']
            print(f"   {i}. [{imp_str}] {readable}")
            print(f"      └─ Category: {category}")
    
    # Cross-detector comparison
    if len(firing_randoms) > 1:
        print("\n" + "="*80)
        print("🔗 CROSS-DETECTOR COMPARISON")
        print("="*80)
        
        # Reconstruct minimal anomaly_result for comparison
        random_anomalies = {
            d['name']: {'score': d['score']} 
            for d in firing_randoms
        }
        
        comparison = analyzer.compare_firing_detectors(
            {'random_anomalies': random_anomalies}, 
            threshold=threshold
        )
        
        if 'common_themes' in comparison:
            themes = comparison['common_themes']
            
            print(f"\n🎨 Dominant Categories Across All Firing Detectors:")
            for category, count in list(themes.get('dominant_categories', {}).items())[:5]:
                print(f"   • {category}: {count} features")
            
            overlapping = themes.get('overlapping_features', {})
            if overlapping:
                print(f"\n🔄 Features Appearing in Multiple Detectors:")
                for feat, count in sorted(overlapping.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"   • {feat}: {count} detectors")
            
            print(f"\n💭 Overall Interpretation:")
            print(f"   {themes.get('interpretation', 'N/A')}")
    
    # Context from domain detectors
    print("\n" + "="*80)
    print("🌐 DOMAIN DETECTOR CONTEXT")
    print("="*80)
    
    if 'persistence' in anomaly_result:
        active = anomaly_result['persistence'].get('active_detectors', [])
        print(f"\n⚡ Currently Active Domain Detectors ({len(active)}):")
        for detector in active:
            if detector in anomaly_result.get('domain_anomalies', {}):
                score = anomaly_result['domain_anomalies'][detector]['score']
                print(f"   • {detector}: {score:.3f} ({score*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    # Export detailed analysis
    output_path = './json_data/firing_detector_analysis.json'
    full_analysis = {
        'timestamp': anomaly_result.get('timestamp'),
        'threshold': threshold,
        'firing_detectors': [d['name'] for d in firing_randoms],
        'analyses': {
            d['name']: analyzer.analyze_single_detector(d['name']) 
            for d in firing_randoms
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    
    print(f"\n💾 Detailed analysis exported to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze firing random detectors in anomaly detection system'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.85,
        help='Anomaly score threshold for "firing" detectors (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_firing_detectors(threshold=args.threshold)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)