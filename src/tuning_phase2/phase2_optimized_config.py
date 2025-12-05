# PHASE 2 OPTIMIZED CONFIG - 2025-12-05 18:00:07
# Target accuracy: 55%+ for both UP and DOWN

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0535,
    'confidence_weights': {
        'up': {'classifier': 0.7054, 'magnitude': 0.2946},
        'down': {'classifier': 0.6296, 'magnitude': 0.3704}
    },
    'magnitude_scaling': {
        'up': {'small': 3.9697, 'medium': 6.4848, 'large': 12.4908},
        'down': {'small': 2.8488, 'medium': 6.2560, 'large': 8.7506}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6033,
            'medium_magnitude': 0.6561,
            'low_magnitude': 0.6940
        },
        'down': {
            'high_magnitude': 0.5861,
            'medium_magnitude': 0.6580,
            'low_magnitude': 0.7073
        }
    },
    'min_confidence_up': 0.5864,
    'min_confidence_down': 0.6459,
    'boost_threshold_up': 17.2750,
    'boost_threshold_down': 10.6137,
    'boost_amount_up': 0.0771,
    'boost_amount_down': 0.0529,
    'description': 'Phase 2 optimized for 55%+ accuracy (high precision)'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 69.1% (UP 69.3%, DOWN 68.9%)
# MAE 13.12% | Signals: 375 (77.5% actionable)
# UP signals: 166 (44.3%) | DOWN signals: 209 (55.7%)
# Trading frequency: ~3.8 signals/week
