# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - 2025-12-17 00:04:24
# Target accuracy: 68%+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0711,
    'confidence_weights': {
        'up': {'classifier': 0.7432, 'magnitude': 0.2568},
        'down': {'classifier': 0.6130, 'magnitude': 0.3870}
    },
    'magnitude_scaling': {
        'up': {'small': 2.5296, 'medium': 5.0009, 'large': 10.9756},
        'down': {'small': 3.0799, 'medium': 4.6980, 'large': 8.2516}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6975,
            'medium_magnitude': 0.7594,
            'low_magnitude': 0.8179
        },
        'down': {
            'high_magnitude': 0.7716,
            'medium_magnitude': 0.8414,
            'low_magnitude': 0.8788
        }
    },
    'min_confidence_up': 0.5587,
    'min_confidence_down': 0.6604,
    'boost_threshold_up': 11.9459,
    'boost_threshold_down': 12.3533,
    'boost_amount_up': 0.0431,
    'boost_amount_down': 0.0762,
    'description': 'Phase 2 optimized for 68%+ balanced accuracy'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 77.1% (UP 77.4%, DOWN 76.7%)
# Balance gap: 0.6%
# MAE 14.33% | Signals: 170 (35.4% actionable)
# UP signals: 84 (49.4%) | DOWN signals: 86 (50.6%)
# Trading frequency: ~1.7 signals/week
