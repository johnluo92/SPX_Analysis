# TUNED CONFIG - 2025-11-24 14:52
# Mag MAE: 10.48% | Bias: -0.26% | Max Pred: 0.170
# Dir F1: 0.5722 | Acc: 61.9%

FEATURE_SELECTION_CV_PARAMS = {
    'n_estimators': 132,
    'max_depth': 3,
    'learning_rate': 0.0544,
    'subsample': 0.8044,
    'colsample_bytree': 0.8411
}

FEATURE_SELECTION_CONFIG = {
    'magnitude_top_n': 70,
    'direction_top_n': 96,
    'cv_folds': 5,
    'protected_features': ['is_fomc_period','is_opex_week','is_earnings_heavy'],
    'correlation_threshold': 1
}

XGBOOST_CONFIG['magnitude_params'].update({
    'max_depth': 3,
    'learning_rate': 0.0133,
    'n_estimators': 436,
    'subsample': 0.7409,
    'colsample_bytree': 0.9162,
    'colsample_bylevel': 0.8823,
    'min_child_weight': 8,
    'reg_alpha': 1.1390,
    'reg_lambda': 3.7097,
    'gamma': 0.3023
})

XGBOOST_CONFIG['direction_params'].update({
    'max_depth': 5,
    'learning_rate': 0.0419,
    'n_estimators': 481,
    'subsample': 0.8288,
    'colsample_bytree': 0.7644,
    'min_child_weight': 6,
    'reg_alpha': 1.9222,
    'reg_lambda': 3.1447,
    'gamma': 0.4038,
    'scale_pos_weight': 1.1706
})
