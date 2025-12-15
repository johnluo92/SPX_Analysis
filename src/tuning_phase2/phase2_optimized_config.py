# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - 2025-12-15 02:39:11
# Target accuracy: 65%+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.1340,
    'confidence_weights': {
        'up': {'classifier': 0.7295, 'magnitude': 0.2705},
        'down': {'classifier': 0.7760, 'magnitude': 0.2240}
    },
    'magnitude_scaling': {
        'up': {'small': 4.0802, 'medium': 7.1423, 'large': 13.4034},
        'down': {'small': 2.8607, 'medium': 5.2218, 'large': 11.7516}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6899,
            'medium_magnitude': 0.7348,
            'low_magnitude': 0.7991
        },
        'down': {
            'high_magnitude': 0.7709,
            'medium_magnitude': 0.8461,
            'low_magnitude': 0.8673
        }
    },
    'min_confidence_up': 0.6909,
    'min_confidence_down': 0.6077,
    'boost_threshold_up': 13.2789,
    'boost_threshold_down': 13.1949,
    'boost_amount_up': 0.0398,
    'boost_amount_down': 0.0353,
    'description': 'Phase 2 optimized for 65%+ balanced accuracy'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 79.3% (UP 79.2%, DOWN 79.4%)
# Balance gap: 0.1%
# MAE 12.24% | Signals: 116 (24.2% actionable)
# UP signals: 53 (45.7%) | DOWN signals: 63 (54.3%)
# Trading frequency: ~1.2 signals/week
