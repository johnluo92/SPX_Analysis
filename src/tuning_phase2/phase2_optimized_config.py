# PHASE 2 OPTIMIZED CONFIG - 2025-12-05 15:48:32

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0958,
    'confidence_weights': {
        'up': {'classifier': 0.6177, 'magnitude': 0.3823},
        'down': {'classifier': 0.7698, 'magnitude': 0.2302}
    },
    'magnitude_scaling': {
        'up': {'small': 2.5246, 'medium': 6.6956, 'large': 12.1994},
        'down': {'small': 2.3534, 'medium': 5.9531, 'large': 9.1828}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.4808,
            'medium_magnitude': 0.5488,
            'low_magnitude': 0.5714
        },
        'down': {
            'high_magnitude': 0.5764,
            'medium_magnitude': 0.6487,
            'low_magnitude': 0.6703
        }
    },
    'min_confidence_up': 0.5030,
    'min_confidence_down': 0.6081,
    'boost_threshold_up': 11.9205,
    'boost_threshold_down': 14.0452,
    'boost_amount_up': 0.0307,
    'boost_amount_down': 0.0353,
    'description': 'Phase 2 optimized for HIGH PRECISION (85% target accuracy)'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 56.7% (UP 53.0%, DOWN 66.4%)
# MAE 15.06% | Signals: 467 (96.5% actionable)
# UP signals: 336 (71.9%) | DOWN signals: 131 (28.1%)
# Trading frequency: ~4.7 signals/week
