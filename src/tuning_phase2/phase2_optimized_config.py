# PHASE 2 OPTIMIZED CONFIG - 2025-12-04 01:26:30

CALIBRATION_WINDOW_DAYS = 550

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.0875,
    'confidence_weights': {
        'classifier': 0.7232,
        'magnitude': 0.2768
    },
    'magnitude_scaling': {
        'small': 3.7022,
        'medium': 6.7702,
        'large': 10.0823
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6701,
            'medium_magnitude': 0.6664,
            'low_magnitude': 0.6997
        },
        'down': {
            'high_magnitude': 0.5612,
            'medium_magnitude': 0.6112,
            'low_magnitude': 0.6416
        }
    },
    'min_ensemble_confidence': 0.5212,
    'confidence_boost_threshold': 17.8195,
    'confidence_boost_amount': 0.0650,
    'description': 'Phase 2 optimized for HIGH PRECISION (85% target accuracy)'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 67.8% (UP 70.2%, DOWN 66.3%)
# MAE 13.74% | Signals: 416 (86.1% actionable)
# UP signals: 161 (38.7%) | DOWN signals: 255 (61.3%)
# Trading frequency: ~4.2 signals/week
# Calibration window: 550 days
