# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - 2025-12-15 10:58:20
# Target accuracy: 68%+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': 0.1348,
    'confidence_weights': {
        'up': {'classifier': 0.6529, 'magnitude': 0.3471},
        'down': {'classifier': 0.7999, 'magnitude': 0.2001}
    },
    'magnitude_scaling': {
        'up': {'small': 4.2362, 'medium': 6.0049, 'large': 12.5873},
        'down': {'small': 3.5369, 'medium': 6.0982, 'large': 9.8526}
    },
    'dynamic_thresholds': {
        'up': {
            'high_magnitude': 0.6758,
            'medium_magnitude': 0.7510,
            'low_magnitude': 0.7779
        },
        'down': {
            'high_magnitude': 0.7999,
            'medium_magnitude': 0.8439,
            'low_magnitude': 0.8738
        }
    },
    'min_confidence_up': 0.6682,
    'min_confidence_down': 0.6896,
    'boost_threshold_up': 17.2306,
    'boost_threshold_down': 17.1792,
    'boost_amount_up': 0.0590,
    'boost_amount_down': 0.0383,
    'description': 'Phase 2 optimized for 68%+ balanced accuracy'
}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable 82.1% (UP 82.4%, DOWN 81.6%)
# Balance gap: 0.8%
# MAE 11.13% | Signals: 106 (22.1% actionable)
# UP signals: 68 (64.2%) | DOWN signals: 38 (35.8%)
# Trading frequency: ~1.1 signals/week
