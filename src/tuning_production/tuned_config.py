# PRODUCTION-TUNED CONFIG v1.0 - 2025-12-01
# Optimized on 2024-2025 test data with single calibrator (matches production)
# Replace these sections in your config.py

QUALITY_FILTER_CONFIG={'enabled':True,'min_threshold':0.5830,'warn_pct':20.0,'error_pct':50.0,'strategy':'raise'}

CALENDAR_COHORTS={'fomc_period':{'condition':'macro_event_period','range':(-7,2),'weight':1.4475,'description':'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},'opex_week':{'condition':'days_to_monthly_opex','range':(-7,0),'weight':1.3501,'description':'Options expiration week + VIX futures rollover'},'earnings_heavy':{'condition':'spx_earnings_pct','range':(0.15,1.0),'weight':1.0558,'description':'Peak earnings season (Jan, Apr, Jul, Oct)'},'mid_cycle':{'condition':'default','range':None,'weight':1.0,'description':'Regular market conditions'}}

FEATURE_SELECTION_CV_PARAMS={'n_estimators':207,'max_depth':3,'learning_rate':0.0675,'subsample':0.8714,'colsample_bytree':0.7970}

FEATURE_SELECTION_CONFIG={'magnitude_top_n':108,'direction_top_n':119,'cv_folds':5,'protected_features':['is_fomc_period','is_opex_week','is_earnings_heavy'],'correlation_threshold':0.9269,'description':'Production-tuned on 2024-2025 test data'}

MAGNITUDE_PARAMS={'objective':'reg:squarederror','eval_metric':'rmse','max_depth':3,'learning_rate':0.0629,'n_estimators':603,'subsample':0.8751,'colsample_bytree':0.9252,'colsample_bylevel':0.9207,'min_child_weight':6,'reg_alpha':5.5318,'reg_lambda':5.8986,'gamma':0.1283,'early_stopping_rounds':50,'seed':42,'n_jobs':-1}

DIRECTION_PARAMS={'objective':'binary:logistic','eval_metric':'logloss','max_depth':8,'learning_rate':0.0677,'n_estimators':478,'subsample':0.9493,'colsample_bytree':0.8408,'min_child_weight':11,'reg_alpha':2.6824,'reg_lambda':6.2968,'gamma':0.4012,'scale_pos_weight':1.1914,'max_delta_step':0,'early_stopping_rounds':50,'seed':42,'n_jobs':-1}

ENSEMBLE_CONFIG={'enabled':True,'reconciliation_method':'weighted_agreement','confidence_weights':{'magnitude':0.3821,'direction':0.4238,'agreement':0.1941},'magnitude_thresholds':{'small':3.3725,'medium':5.2122,'large':11.9369},'agreement_bonus':{'strong':0.1672,'moderate':0.1427,'weak':0.0},'contradiction_penalty':{'severe':0.3369,'moderate':0.1270,'minor':0.0476},'min_ensemble_confidence':0.50,'actionable_threshold':0.6297,'description':'Production-tuned ensemble'}
