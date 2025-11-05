"""Dashboard Data Contract v3.0 - FIXED"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import REGIME_NAMES, ANOMALY_THRESHOLDS

def export_historical_anomaly_scores(vix_predictor, spx, output_dir='./json_data'):
    output_path = Path(output_dir) / 'historical_anomaly_scores.json'
    features = vix_predictor.features
    detector = vix_predictor.anomaly_detector
    ensemble_scores = []
    for i in range(len(features)):
        result = detector.detect(features.iloc[[i]], verbose=False)
        ensemble_scores.append(result['ensemble']['score'])
    spx_forward_10d = spx.pct_change(10).shift(-10) * 100
    historical_data = {'dates': [d.strftime('%Y-%m-%d') for d in features.index], 'ensemble_scores': ensemble_scores,
                      'spx_close': spx.values.tolist(), 'spx_forward_10d': spx_forward_10d.fillna(0).values.tolist()}
    with open(output_path, 'w') as f:
        json.dump(historical_data, f, indent=2, default=_json_serializer)
    print(f"✅ Historical scores: {len(ensemble_scores)} days")
    return historical_data

def export_dashboard_data(vix_predictor, spx_predictor, output_dir='./json_data', anomaly_result=None):
    output_path = Path(output_dir) / 'dashboard_data.json'
    output_path.parent.mkdir(exist_ok=True)
    print(f"\n{'='*70}\nEXPORTING DASHBOARD DATA CONTRACT v3.0\n{'='*70}")
    if anomaly_result is None and vix_predictor and hasattr(vix_predictor, 'anomaly_detector'):
        print("   ⚠️  Running fresh detection")
        try:
            current_features = vix_predictor.features.iloc[[-1]]
            anomaly_result = vix_predictor.anomaly_detector.detect(current_features, verbose=False)
        except Exception as e:
            print(f"   ⚠️  Detection failed: {e}")
            anomaly_result = None
    elif anomaly_result is not None:
        print("   ✅ Using cached result")
    
    # ✅ FIXED: Calculate persistence from FULL historical scores
    persistence_stats = {}
    if vix_predictor and hasattr(vix_predictor, 'anomaly_detector'):
        try:
            if hasattr(vix_predictor, 'historical_ensemble_scores') and vix_predictor.historical_ensemble_scores is not None:
                persistence_stats = vix_predictor.anomaly_detector.calculate_historical_persistence_stats(
                    vix_predictor.historical_ensemble_scores
                )
                print(f"   ✅ Persistence calculated from {len(vix_predictor.historical_ensemble_scores)} observations")
            else:
                print("   ⚠️ Historical scores not available for persistence calculation")
                persistence_stats = {
                    'current_streak': 0,
                    'mean_duration': 0.0,
                    'max_duration': 0,
                    'total_anomaly_days': 0,
                    'anomaly_rate': 0.0,
                    'num_episodes': 0
                }
        except Exception as e:
            print(f"   ⚠️ Persistence stats failed: {e}")
            persistence_stats = {}

    dashboard_data = {'version': '3.0', 'last_updated': datetime.now().isoformat(),
                     'current_state': _build_current_state(vix_predictor, anomaly_result, persistence_stats),
                     'regime_analysis': _build_regime_analysis(vix_predictor),
                     'anomaly_analysis': _build_anomaly_analysis(vix_predictor, anomaly_result, persistence_stats),
                     'alerts': _generate_alerts(vix_predictor, anomaly_result, persistence_stats)}
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=_json_serializer)
    print(f"✅ Exported to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    export_feature_attribution(vix_predictor, anomaly_result, output_dir)
    return dashboard_data

def export_feature_attribution(vix_predictor, anomaly_result=None, output_dir='./json_data'):
    output_path = Path(output_dir) / 'anomaly_feature_attribution.json'
    output_path.parent.mkdir(exist_ok=True)
    if not vix_predictor or not hasattr(vix_predictor, 'anomaly_detector'):
        print("⚠️  No anomaly detector")
        return
    if anomaly_result is None:
        print("⚠️  No anomaly result")
        return
    detector = vix_predictor.anomaly_detector
    attribution = {'timestamp': datetime.now().isoformat(), 'domains': {}}
    if vix_predictor.features is not None:
        current_features = vix_predictor.features.iloc[-1].to_dict()
        for domain_name, domain_data in anomaly_result.get('domain_anomalies', {}).items():
            importances = detector.feature_importances.get(domain_name, {})
            if importances:
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                attribution['domains'][domain_name] = {
                    'score': float(domain_data['score']), 'level': domain_data['level'],
                    'features': [{'feature': fn, 'value': float(current_features.get(fn, 0)), 'importance': float(imp)}
                               for fn, imp in sorted_features if not np.isnan(current_features.get(fn, np.nan))]}
    with open(output_path, 'w') as f:
        json.dump(attribution, f, indent=2, default=_json_serializer)
    print(f"✅ Exported feature attribution")
    return attribution

def _json_serializer(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return str(obj)

def _get_statistical_thresholds(vix_predictor):
    """Get statistical thresholds from detector, with fallback."""
    if vix_predictor and hasattr(vix_predictor, 'anomaly_detector'):
        detector = vix_predictor.anomaly_detector
        if hasattr(detector, 'statistical_thresholds') and detector.statistical_thresholds:
            return detector.statistical_thresholds
    
    # Fallback to config
    return {'moderate': 0.70, 'high': 0.78, 'critical': 0.88}

def _build_current_state(vix_predictor, anomaly_result=None, persistence_stats=None):
    state = {'timestamp': datetime.now().isoformat(), 'vix': None, 'vix_regime': None, 'vix_regime_name': None,
            'days_in_regime': None, 'spx_close': None, 'anomaly_ensemble_score': None, 'anomaly_severity': None,
            'anomaly_persistence': {'current_count': 0, 'max_possible': 10, 'percentage': 0.0, 'active_detectors': []}}
    if vix_predictor and vix_predictor.vix is not None:
        state['vix'] = float(vix_predictor.vix.iloc[-1])
        if vix_predictor.features is not None and 'vix_regime' in vix_predictor.features.columns:
            regime = int(vix_predictor.features['vix_regime'].iloc[-1])
            state['vix_regime'] = regime
            state['vix_regime_name'] = REGIME_NAMES.get(regime, 'Unknown')
            if 'days_in_regime' in vix_predictor.features.columns:
                state['days_in_regime'] = int(vix_predictor.features['days_in_regime'].iloc[-1])
    try:
        live_spx = vix_predictor.fetcher.fetch_price('^GSPC')
        if live_spx:
            state['spx_close'] = float(live_spx)
    except:
        pass
    if anomaly_result:
        ensemble = anomaly_result.get('ensemble', {})
        state['anomaly_ensemble_score'] = float(ensemble.get('score', 0))
        score = ensemble.get('score', 0)
        
        # Get statistical thresholds
        thresholds = _get_statistical_thresholds(vix_predictor)
        
        # Classify using statistical thresholds
        if score >= thresholds['critical']:
            state['anomaly_severity'] = 'CRITICAL'
        elif score >= thresholds['high']:
            state['anomaly_severity'] = 'HIGH'
        elif score >= thresholds['moderate']:
            state['anomaly_severity'] = 'MODERATE'
        else:
            state['anomaly_severity'] = 'NORMAL'
        
        if 'domain_anomalies' in anomaly_result:
            # Use statistical threshold for active detectors
            threshold = thresholds['high']
            active_detectors = [name for name, data in anomaly_result['domain_anomalies'].items()
                              if data.get('score', 0) > threshold]
            state['anomaly_persistence'] = {'current_count': len(active_detectors),
                                           'max_possible': len(anomaly_result['domain_anomalies']),
                                           'percentage': len(active_detectors) / len(anomaly_result['domain_anomalies']) if anomaly_result['domain_anomalies'] else 0.0,
                                           'active_detectors': active_detectors}
    return state

def _build_regime_analysis(vix_predictor):
    if not vix_predictor or not hasattr(vix_predictor, 'regime_stats_historical'):
        return {'available': False}
    stats = vix_predictor.regime_stats_historical
    analysis = {'available': True, 'current_regime': None, 'regime_name': None, 'days_in_regime': None, 'regimes': {}}
    if vix_predictor.features is not None and 'vix_regime' in vix_predictor.features.columns:
        current_regime = int(vix_predictor.features['vix_regime'].iloc[-1])
        analysis['current_regime'] = current_regime
        analysis['regime_name'] = REGIME_NAMES.get(current_regime, 'Unknown')
        if 'days_in_regime' in vix_predictor.features.columns:
            analysis['days_in_regime'] = int(vix_predictor.features['days_in_regime'].iloc[-1])
    for regime_data in stats.get('regimes', []):
        regime_id = regime_data['id']
        regime_name = regime_data['name']
        analysis['regimes'][regime_name] = {'id': regime_id, 'vix_range': regime_data['vix_range'],
                                           'statistics': regime_data['statistics'],
                                           'persistence_5d': regime_data['transitions_5d']['persistence']['probability'],
                                           'mean_duration': regime_data['statistics']['mean_duration']}
    return analysis

def _build_anomaly_analysis(vix_predictor, anomaly_result=None, persistence_stats=None):
    if not vix_predictor or not hasattr(vix_predictor, 'anomaly_detector'):
        return {'available': False}
    detector = vix_predictor.anomaly_detector
    analysis = {'available': True, 'ensemble': {}, 'persistence': {}, 'domain_anomalies': {},
               'top_anomalies': [], 'feature_attribution': {}}
    if anomaly_result:
        score = float(anomaly_result['ensemble']['score'])
        classification = {}
        if hasattr(detector, 'classify_anomaly') and hasattr(detector, 'statistical_thresholds') and detector.statistical_thresholds:
            level, p_value, confidence = detector.classify_anomaly(score, method='statistical')
            classification = {
                'level': level,
                'p_value': float(p_value) if p_value is not None else None,
                'confidence': float(confidence) if confidence is not None else None,
                'thresholds': detector.statistical_thresholds
            }
        else:
            # Fallback classification
            thresholds = _get_statistical_thresholds(vix_predictor)
            if score >= thresholds['critical']:
                level = 'CRITICAL'
            elif score >= thresholds['high']:
                level = 'HIGH'
            elif score >= thresholds['moderate']:
                level = 'MODERATE'
            else:
                level = 'NORMAL'
            classification = {'level': level, 'thresholds': thresholds}
        
        analysis['ensemble'] = {'score': score, 'std': float(anomaly_result['ensemble']['std']),
                               'n_detectors': int(anomaly_result['ensemble']['n_detectors']), 'classification': classification}
        
        # ✅ FIXED: Use persistence_stats passed from export_dashboard_data()
        if persistence_stats:
            analysis['persistence'] = {
                'current_streak': persistence_stats.get('current_streak', 0),
                'mean_duration': persistence_stats.get('mean_duration', 0.0),
                'max_duration': persistence_stats.get('max_duration', 0),
                'total_anomaly_days': persistence_stats.get('total_anomaly_days', 0),
                'anomaly_rate': persistence_stats.get('anomaly_rate', 0.0),
                'num_episodes': persistence_stats.get('num_episodes', 0)
            }
        
        analysis['domain_anomalies'] = {name: {'score': float(data['score']), 'percentile': float(data['percentile']), 'level': data['level']}
                                       for name, data in anomaly_result.get('domain_anomalies', {}).items()}
        top_anomalies = detector.get_top_anomalies(anomaly_result, top_n=3)
        analysis['top_anomalies'] = [{'detector': name, 'score': float(score)} for name, score in top_anomalies]
        if vix_predictor.features is not None:
            current_features = vix_predictor.features.iloc[-1].to_dict()
            for domain_name, domain_data in anomaly_result.get('domain_anomalies', {}).items():
                importances = detector.feature_importances.get(domain_name, {})
                if importances:
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                    analysis['feature_attribution'][domain_name] = [
                        {'feature': fn, 'value': float(current_features.get(fn, 0)), 'importance': float(imp)}
                        for fn, imp in sorted_features if not np.isnan(current_features.get(fn, np.nan))]
    return analysis

def _generate_alerts(vix_predictor, anomaly_result=None, persistence_stats=None):
    alerts = []
    timestamp = datetime.now().isoformat()
    if vix_predictor and vix_predictor.features is not None:
        if 'days_in_regime' in vix_predictor.features.columns:
            days = int(vix_predictor.features['days_in_regime'].iloc[-1])
            if days <= 1:
                regime = int(vix_predictor.features['vix_regime'].iloc[-1])
                regime_name = REGIME_NAMES.get(regime, 'Unknown')
                alerts.append({'timestamp': timestamp, 'priority': 'HIGH', 'category': 'REGIME_TRANSITION',
                             'message': f"VIX regime transitioned to {regime_name}"})
    if anomaly_result:
        ensemble_score = anomaly_result['ensemble']['score']
        # Get statistical thresholds
        thresholds = _get_statistical_thresholds(vix_predictor)
        
        if ensemble_score >= thresholds['critical']:
            alerts.append({'timestamp': timestamp, 'priority': 'CRITICAL', 'category': 'ANOMALY_CRITICAL',
                         'message': f"CRITICAL anomaly detected (score: {ensemble_score:.1%})"})
        elif ensemble_score >= thresholds['high']:
            alerts.append({'timestamp': timestamp, 'priority': 'HIGH', 'category': 'ANOMALY_HIGH',
                         'message': f"High anomaly risk detected (score: {ensemble_score:.1%})"})
        
        if 'domain_anomalies' in anomaly_result:
            active_count = sum(1 for data in anomaly_result['domain_anomalies'].values()
                             if data.get('score', 0) > thresholds['high'])
            if active_count >= 5:
                alerts.append({'timestamp': timestamp, 'priority': 'HIGH', 'category': 'ANOMALY_PERSISTENCE',
                             'message': f"Anomaly persisting across {active_count} detectors (sustained stress)"})
    return alerts