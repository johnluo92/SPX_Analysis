# COMPREHENSIVE TUNED CONFIGURATION v2.0
# Generated: 2025-11-30 01:32:01
# Trial #40 - Score: 65.8394
#
# WALK-FORWARD CV PERFORMANCE (5 folds):
# ════════════════════════════════════════════════════════════════
# Magnitude MAE: 12.63% ± 3.05%
# Magnitude Bias: -2.63%
# Magnitude Calibration Error: 0.098
# Magnitude Train/Val Gap: 2.85%
#
# Direction Accuracy: 64.3% ± 4.6%
# Direction F1: 0.3814
# Direction Precision: 65.6%
# Direction Recall: 28.4%
# Direction ECE: 0.127 (Expected Calibration Error)
# Direction Brier: 0.2418
# Direction Train/Val Gap: 17.7%
#
# Ensemble Confidence: 63.1%
# Actionable Trades: 22.8%
#
# DIVERSITY METRICS:
# ════════════════════════════════════════════════════════════════
# Feature Jaccard: 0.375
# Feature Overlap: 0.583
# Prediction Correlation: 0.822
# Overall Diversity: 0.418
#
# FEATURES:
# Magnitude: 91
# Direction: 80
# Common: 46
# ════════════════════════════════════════════════════════════════

QUALITY_FILTER_CONFIG={'enabled':True,'min_threshold':0.6286,'warn_pct':20.0,'error_pct':50.0,'strategy':'raise'}

CALENDAR_COHORTS={'fomc_period':{'condition':'macro_event_period','range':(-7,2),'weight':1.1569},'opex_week':{'condition':'days_to_monthly_opex','range':(-7,0),'weight':1.4432},'earnings_heavy':{'condition':'spx_earnings_pct','range':(0.15,1.0),'weight':1.0139},'mid_cycle':{'condition':'default','range':None,'weight':1.0}}

FEATURE_SELECTION_CV_PARAMS={'n_estimators':150,'max_depth':4,'learning_rate':0.0420,'subsample':0.8979,'colsample_bytree':0.9349}

FEATURE_SELECTION_CONFIG={'magnitude_top_n':88,'direction_top_n':77,'cv_folds':5,'protected_features':['is_fomc_period','is_opex_week','is_earnings_heavy'],'correlation_threshold':0.9463,'target_overlap':0.4644,'description':'Optimized via nested walk-forward CV'}

MAGNITUDE_PARAMS={'objective':'reg:squarederror','eval_metric':'rmse','max_depth':3,'learning_rate':0.0818,'n_estimators':331,'subsample':0.6634,'colsample_bytree':0.8012,'colsample_bylevel':0.6506,'min_child_weight':5,'reg_alpha':4.5497,'reg_lambda':2.8007,'gamma':0.2722,'early_stopping_rounds':50,'seed':42,'n_jobs':-1}

DIRECTION_PARAMS={'objective':'binary:logistic','eval_metric':'logloss','max_depth':4,'learning_rate':0.0220,'n_estimators':568,'subsample':0.8247,'colsample_bytree':0.7561,'min_child_weight':14,'reg_alpha':1.8721,'reg_lambda':3.1192,'gamma':0.3081,'scale_pos_weight':1.1472,'max_delta_step':1,'early_stopping_rounds':50,'seed':42,'n_jobs':-1}

ENSEMBLE_CONFIG={'enabled':True,'reconciliation_method':'weighted_agreement','confidence_weights':{'magnitude':0.4634,'direction':0.5346,'agreement':0.1609},'magnitude_thresholds':{'small':2.4365,'medium':7.2424,'large':16.5708},'agreement_bonus':{'strong':0.1808,'moderate':0.0931,'weak':0.0},'contradiction_penalty':{'severe':0.2461,'moderate':0.2170,'minor':0.0297},'min_ensemble_confidence':0.50,'actionable_threshold':0.6917,'description':'Optimized via walk-forward CV with diversity constraints'}

DIVERSITY_CONFIG={'enabled':True,'target_feature_jaccard':0.40,'target_feature_overlap':0.4644,'diversity_weight':1.5000,'metrics':{'feature_jaccard':0.375,'feature_overlap':0.583,'pred_correlation':0.822,'overall_diversity':0.418},'description':'Ensures complementary models without excessive overlap'}
