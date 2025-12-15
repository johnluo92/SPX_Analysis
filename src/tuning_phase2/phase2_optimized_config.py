# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - 2025-12-15 09:21:45
# Target accuracy: 70%+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.1196,
    'confidence_weights': {
        'up': {'classifier': 0.7514, 'magnitude': 0.2486},
        'down': {'classifier': 0.7848, 'magnitude': 0.2152}
    },
    'magnitude_scaling': {
        'up': {'small': 3.4248, 'medium': 6.0714, 'large': 13.4203},
        'down': {'small': 3.0343, 'medium': 6.2854, 'large': 9.1478}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6860,
            'medium_magnitude': 0.7580,
            'low_magnitude': 0.8052
        },
        'down': {
            'high_magnitude': 0.7866,
            'medium_magnitude': 0.8358,
            'low_magnitude': 0.8889
        }
    },
    'min_confidence_up': 0.5766,
    'min_confidence_down': 0.6467,
    'boost_threshold_up': 17.0062,
    'boost_threshold_down': 17.8712,
    'boost_amount_up': 0.0380,
    'boost_amount_down': 0.0499,
    'description': 'Phase 2 optimized for 70%+ balanced accuracy'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 89.4% (UP 89.3%, DOWN 89.5%)
# Balance gap: 0.2%
# MAE 10.19% | Signals: 66 (13.8% actionable)
# UP signals: 28 (42.4%) | DOWN signals: 38 (57.6%)
# Trading frequency: ~0.7 signals/week
