# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - 2025-12-06 15:39:54
# Target accuracy: 70%+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0913,
    'confidence_weights': {
        'up': {'classifier': 0.6453, 'magnitude': 0.3547},
        'down': {'classifier': 0.5411, 'magnitude': 0.4589}
    },
    'magnitude_scaling': {
        'up': {'small': 2.7955, 'medium': 5.0043, 'large': 10.3435},
        'down': {'small': 3.4702, 'medium': 4.6143, 'large': 8.9073}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.5981,
            'medium_magnitude': 0.6780,
            'low_magnitude': 0.6992
        },
        'down': {
            'high_magnitude': 0.7843,
            'medium_magnitude': 0.8044,
            'low_magnitude': 0.8333
        }
    },
    'min_confidence_up': 0.5625,
    'min_confidence_down': 0.6675,
    'boost_threshold_up': 10.4039,
    'boost_threshold_down': 12.6627,
    'boost_amount_up': 0.0791,
    'boost_amount_down': 0.0520,
    'description': 'Phase 2 optimized for 70%+ balanced accuracy'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 72.4% (UP 72.3%, DOWN 72.4%)
# Balance gap: 0.1%
# MAE 13.24% | Signals: 333 (68.8% actionable)
# UP signals: 148 (44.4%) | DOWN signals: 185 (55.6%)
# Trading frequency: ~3.3 signals/week
