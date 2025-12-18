# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - 2025-12-17 22:34:22
# Target accuracy: 68%+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0558,
    'confidence_weights': {
        'up': {'classifier': 0.5432, 'magnitude': 0.4568},
        'down': {'classifier': 0.6884, 'magnitude': 0.3116}
    },
    'magnitude_scaling': {
        'up': {'small': 3.1433, 'medium': 5.7333, 'large': 13.2343},
        'down': {'small': 2.9516, 'medium': 4.4775, 'large': 8.2669}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6476,
            'medium_magnitude': 0.7107,
            'low_magnitude': 0.7561
        },
        'down': {
            'high_magnitude': 0.7355,
            'medium_magnitude': 0.8018,
            'low_magnitude': 0.8387
        }
    },
    'min_confidence_up': 0.5617,
    'min_confidence_down': 0.6372,
    'boost_threshold_up': 14.7273,
    'boost_threshold_down': 13.3753,
    'boost_amount_up': 0.0731,
    'boost_amount_down': 0.0636,
    'description': 'Phase 2 optimized for 68%+ balanced accuracy'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 77.3% (UP 75.8%, DOWN 79.4%)
# Balance gap: 3.5%
# MAE 12.10% | Signals: 154 (32.1% actionable)
# UP signals: 91 (59.1%) | DOWN signals: 63 (40.9%)
# Trading frequency: ~1.5 signals/week
