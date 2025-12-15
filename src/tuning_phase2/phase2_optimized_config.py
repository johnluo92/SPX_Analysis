# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - 2025-12-12 00:29:21
# Target accuracy: 70%+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0646,
    'confidence_weights': {
        'up': {'classifier': 0.5207, 'magnitude': 0.4793},
        'down': {'classifier': 0.7399, 'magnitude': 0.2601}
    },
    'magnitude_scaling': {
        'up': {'small': 4.4380, 'medium': 5.5688, 'large': 12.8342},
        'down': {'small': 3.9065, 'medium': 4.5442, 'large': 10.9927}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6949,
            'medium_magnitude': 0.7685,
            'low_magnitude': 0.8069
        },
        'down': {
            'high_magnitude': 0.7757,
            'medium_magnitude': 0.8539,
            'low_magnitude': 0.8985
        }
    },
    'min_confidence_up': 0.6849,
    'min_confidence_down': 0.7187,
    'boost_threshold_up': 16.3856,
    'boost_threshold_down': 14.3325,
    'boost_amount_up': 0.0790,
    'boost_amount_down': 0.0393,
    'description': 'Phase 2 optimized for 70%+ balanced accuracy'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 86.4% (UP 88.9%, DOWN 84.4%)
# Balance gap: 4.5%
# MAE 12.91% | Signals: 59 (12.3% actionable)
# UP signals: 27 (45.8%) | DOWN signals: 32 (54.2%)
# Trading frequency: ~0.6 signals/week
