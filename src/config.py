from pathlib import Path
from datetime import datetime,timedelta
import pandas as pd

TRAINING_YEARS=20
TUNER_TRIALS=200

def get_last_complete_month_end():
    today=datetime.now(); first_of_month=today.replace(day=1); last_month_end=first_of_month-timedelta(days=1)
    return last_month_end.strftime("%Y-%m-%d")

def get_training_start_date(end_date_str,years):
    end=pd.Timestamp(end_date_str); start=end-timedelta(days=years*365+450)
    return start.strftime("%Y-%m-%d")

CACHE_DIR="./data_cache"; CBOE_DATA_DIR="./CBOE_Data_Archive"; ENABLE_TRAINING=True; RANDOM_STATE=42
TRAINING_END_DATE=get_last_complete_month_end(); TRAINING_START_DATE=get_training_start_date(TRAINING_END_DATE,TRAINING_YEARS)

DATA_SPLIT_CONFIG={"train_end_date":"2021-12-31","val_end_date":"2023-12-31","feature_selection_split_date":"2023-12-31","description":{"train":"Training data: up to 2021-12-31","val":"Validation data: 2022-01-01 to 2023-12-31","test":"Test data: 2024-01-01 onwards","feature_selection":"Uses Train+Val (up to 2023-12-31), excludes Test"}}

TRAIN_END_DATE=DATA_SPLIT_CONFIG["train_end_date"]; VAL_END_DATE=DATA_SPLIT_CONFIG["val_end_date"]
CALIBRATION_WINDOW_DAYS=700; CALIBRATION_DECAY_LAMBDA=0.125; MIN_SAMPLES_FOR_CORRECTION=50
MODEL_VALIDATION_MAE_THRESHOLD=0.20

TARGET_CONFIG={"horizon_days":5,"horizon_label":"5d","target_type":"log_vix_change","output_type":"vix_pct_change","log_space":{"enabled":True,"description":"Train on log(future_vix/current_vix), convert to % for output"},"movement_bounds":{"floor":-50.0,"ceiling":100.0,"description":"Percentage change bounds (converted from log-space)"}}

COHORT_PRIORITY=["fomc_period","opex_week","earnings_heavy","mid_cycle"]

MACRO_EVENT_CONFIG={"cpi_release":{"day_of_month_target":12,"window_days":2},"pce_release":{"day_of_month_target":28,"window_days":3},"fomc_minutes":{"days_after_meeting":21,"window_days":2},"fomc_meeting":{"pre_meeting_days":7,"post_meeting_days":2}}

DIRECTION_CALIBRATION_CONFIG={"enabled":False,"skip_up_calibration":True,"method":"isotonic","min_samples":100,"out_of_bounds":"clip","description":"Calibration disabled - up_advantage provides superior decision boundary adjustment"}

XGBOOST_CONFIG={"strategy":"asymmetric_4model","cohort_aware":False,"cv_config":{"method":"time_series_split","n_splits":5,"gap":5}}

MODEL_OBJECTIVES=["expansion_regressor","compression_regressor","up_classifier","down_classifier"]

PREDICTION_DB_CONFIG={"db_path":"data_cache/predictions.db","table_name":"forecasts","min_samples_for_calibration":MIN_SAMPLES_FOR_CORRECTION,"schema":{"prediction_id":"TEXT PRIMARY KEY","timestamp":"DATETIME","observation_date":"DATE","forecast_date":"DATE","horizon":"INTEGER","calendar_cohort":"TEXT","cohort_weight":"REAL","prob_up":"REAL","prob_down":"REAL","magnitude_forecast":"REAL","expected_vix":"REAL","feature_quality":"REAL","num_features_used":"INTEGER","current_vix":"REAL","actual_vix_change":"REAL","actual_direction":"INTEGER","direction_error":"REAL","magnitude_error":"REAL","correction_type":"TEXT","features_used":"TEXT","model_version":"TEXT","created_at":"DATETIME","direction_probability":"REAL","direction_prediction":"TEXT","direction_correct":"INTEGER"},"indexes":["CREATE INDEX idx_timestamp ON forecasts(timestamp)","CREATE INDEX idx_observation_date ON forecasts(observation_date)","CREATE INDEX idx_cohort ON forecasts(calendar_cohort)","CREATE INDEX idx_forecast_date ON forecasts(forecast_date)","CREATE INDEX idx_correction_type ON forecasts(correction_type)"]}

ENABLE_TEMPORAL_SAFETY=True

FORWARD_FILL_LIMITS={"daily":5,"weekly":7,"monthly":35,"quarterly":135}

FRED_SERIES_METADATA={"ICSA":{"frequency":"weekly","category":"labor"},"STLFSI4":{"frequency":"weekly","category":"financial_stress"},"DGS1MO":{"frequency":"daily","category":"treasuries"},"DGS3MO":{"frequency":"daily","category":"treasuries"},"DGS6MO":{"frequency":"daily","category":"treasuries"},"DGS1":{"frequency":"daily","category":"treasuries"},"DGS2":{"frequency":"daily","category":"treasuries"},"DGS5":{"frequency":"daily","category":"treasuries"},"DGS10":{"frequency":"daily","category":"treasuries"},"DGS30":{"frequency":"daily","category":"treasuries"},"BAMLH0A0HYM2":{"frequency":"daily","category":"credit_spreads"},"BAMLH0A1HYBB":{"frequency":"daily","category":"credit_spreads"},"BAMLH0A2HYB":{"frequency":"daily","category":"credit_spreads"},"BAMLH0A3HYC":{"frequency":"daily","category":"credit_spreads"},"BAMLC0A0CM":{"frequency":"daily","category":"credit_spreads"},"SOFR":{"frequency":"daily","category":"funding"},"SOFR90DAYAVG":{"frequency":"daily","category":"funding"},"DFF":{"frequency":"daily","category":"fed_rates"},"T10Y2Y":{"frequency":"daily","category":"treasury_spreads"},"T10Y3M":{"frequency":"daily","category":"treasury_spreads"},"T5YIE":{"frequency":"daily","category":"inflation"},"T10YIE":{"frequency":"daily","category":"inflation"},"VIXCLS":{"frequency":"daily","category":"volatility"},"CPIAUCSL":{"frequency":"monthly","category":"inflation"},"CPILFESL":{"frequency":"monthly","category":"inflation"},"PCEPI":{"frequency":"monthly","category":"inflation"},"PCEPILFE":{"frequency":"monthly","category":"inflation"},"UNRATE":{"frequency":"monthly","category":"labor"},"PAYEMS":{"frequency":"monthly","category":"labor"},"UMCSENT":{"frequency":"monthly","category":"sentiment"},"INDPRO":{"frequency":"monthly","category":"production"},"M1SL":{"frequency":"monthly","category":"monetary"},"M2SL":{"frequency":"monthly","category":"monetary"},"WALCL":{"frequency":"weekly","category":"monetary"},"WTREGEN":{"frequency":"weekly","category":"monetary"},"GDP":{"frequency":"quarterly","category":"growth"},"GDPC1":{"frequency":"quarterly","category":"growth"}}

PUBLICATION_LAGS={"^GSPC":0,"^VIX":0,"^VVIX":0,"CL=F":0,"GC=F":0,"DX-Y.NYB":0,"SKEW":0,"VIX3M":0,"VX1-VX2":0,"VX2-VX1_RATIO":0,"CL1-CL2":0,"DX1-DX2":0,"COR1M":0,"COR3M":0,"VXTH":0,"VXTLT":0,"PCCE":0,"PCCI":0,"PCC":0,"DGS1MO":1,"DGS3MO":1,"DGS6MO":1,"DGS1":1,"DGS2":1,"DGS5":1,"DGS10":1,"DGS30":1,"DTWEXBGS":1,"CPIAUCSL":14,"CPILFESL":14,"PCEPI":28,"PCEPILFE":28,"UNRATE":7,"PAYEMS":7,"ICSA":4,"STLFSI4":7,"BAMLH0A0HYM2":1,"BAMLH0A1HYBB":1,"BAMLH0A2HYB":1,"BAMLH0A3HYC":1,"BAMLC0A0CM":1,"SOFR":0,"SOFR90DAYAVG":0,"DFF":1,"T10Y2Y":1,"T10Y3M":1,"T5YIE":1,"T10YIE":1,"VIXCLS":1,"GDP":90,"GDPC1":90,"UMCSENT":14,"INDPRO":14,"M1SL":7,"M2SL":7,"WALCL":4,"WTREGEN":4}

FEATURE_QUALITY_CONFIG={"staleness_penalty":{"none":1.0,"minor":0.95,"moderate":0.80,"severe":0.50,"critical":0.20},"missingness_penalty":{"critical_features":["vix","spx","vix_percentile_21d","spx_realized_vol_21d"],"important_features":["VX1-VX2","SKEW","yield_10y2y","Dollar_Index"],"optional_features":["GAMMA","VPN","BFLY"]},"quality_thresholds":{"excellent":0.95,"good":0.85,"acceptable":0.70,"poor":0.50,"unusable":0.30}}

REGIME_BOUNDARIES=[0,15.57,23.36,31.16,100]
REGIME_NAMES={0:"Low Vol",1:"Normal",2:"Elevated",3:"Crisis"}

HYPERPARAMETER_TUNING_CONFIG={"enabled":False,"method":"optuna","n_trials":500,"cv_folds":5,"timeout_hours":24,"magnitude_param_space":{"max_depth":(2,8),"learning_rate":(0.005,0.1),"n_estimators":(100,1000),"subsample":(0.6,1.0),"colsample_bytree":(0.6,1.0),"colsample_bylevel":(0.6,1.0),"min_child_weight":(1,15),"reg_alpha":(0.0,5.0),"reg_lambda":(0.0,10.0),"gamma":(0.0,2.0)},"direction_param_space":{"max_depth":(3,10),"learning_rate":(0.01,0.15),"n_estimators":(100,1000),"subsample":(0.6,1.0),"colsample_bytree":(0.6,1.0),"min_child_weight":(1,15),"reg_alpha":(0.0,5.0),"reg_lambda":(0.0,10.0),"gamma":(0.0,2.0),"scale_pos_weight":(0.8,2.0)},"description":"Hyperparameter optimization with Optuna - run after ensemble implementation"}

QUALITY_FILTER_CONFIG={'enabled':True,'min_threshold':0.6286,'warn_pct':20.0,'error_pct':50.0,'strategy':'raise'}
CALENDAR_COHORTS={'fomc_period':{'condition':'macro_event_period','range':(-7,2),'weight':1.2530,'description':'FOMC meetings, CPI releases, PCE releases, FOMC minutes'},'opex_week':{'condition':'days_to_monthly_opex','range':(-7,0),'weight':1.4466,'description':'Options expiration week + VIX futures rollover'},'earnings_heavy':{'condition':'spx_earnings_pct','range':(0.15,1.0),'weight':1.3473,'description':'Peak earnings season (Jan, Apr, Jul, Oct)'},'mid_cycle':{'condition':'default','range':None,'weight':1.0,'description':'Regular market conditions'}}
FEATURE_SELECTION_CV_PARAMS={'n_estimators':105,'max_depth':6,'learning_rate':0.0639,'subsample':0.8266,'colsample_bytree':0.7005}
FEATURE_SELECTION_CONFIG={'expansion_top_n':115,'compression_top_n':116,'up_top_n':136,'down_top_n':101,'cv_folds':5,'protected_features':[],'correlation_threshold':0.8925,'description':'Phase 1 optimized on RAW predictions (no ensemble filtering)'}
EXPANSION_PARAMS={'objective':'reg:squarederror','eval_metric':'rmse','max_depth':4,'learning_rate':0.0114,'n_estimators':603,'subsample':0.7603,'colsample_bytree':0.9497,'colsample_bylevel':0.8720,'min_child_weight':13,'reg_alpha':3.0552,'reg_lambda':7.9579,'gamma':0.0218,'early_stopping_rounds':50,'seed':42,'n_jobs':1}
COMPRESSION_PARAMS={'objective':'reg:squarederror','eval_metric':'rmse','max_depth':3,'learning_rate':0.0846,'n_estimators':541,'subsample':0.7664,'colsample_bytree':0.7703,'colsample_bylevel':0.9295,'min_child_weight':11,'reg_alpha':7.0132,'reg_lambda':3.6590,'gamma':0.5431,'early_stopping_rounds':50,'seed':42,'n_jobs':1}
UP_CLASSIFIER_PARAMS={'objective':'binary:logistic','eval_metric':'aucpr','max_depth':6,'learning_rate':0.1111,'n_estimators':776,'subsample':0.7562,'colsample_bytree':0.9223,'min_child_weight':13,'reg_alpha':4.8640,'reg_lambda':9.2011,'gamma':0.5018,'scale_pos_weight':1.0,'early_stopping_rounds':50,'seed':42,'n_jobs':1}
DOWN_CLASSIFIER_PARAMS={'objective':'binary:logistic','eval_metric':'aucpr','max_depth':8,'learning_rate':0.0883,'n_estimators':766,'subsample':0.7406,'colsample_bytree':0.8341,'min_child_weight':11,'reg_alpha':2.6657,'reg_lambda':7.8673,'gamma':0.1054,'scale_pos_weight':1.0,'early_stopping_rounds':50,'seed':42,'n_jobs':1}

ENSEMBLE_CONFIG={'enabled':True,'reconciliation_method':'winner_takes_all','up_advantage':0.06,'confidence_weights':{'classifier':0.65,'magnitude':0.35},'magnitude_scaling':{'small':3.5,'medium':6.0,'large':12.0},'dynamic_thresholds':{'up':{'high_magnitude':0.47,'medium_magnitude':0.50,'low_magnitude':0.53},'down':{'high_magnitude':0.57,'medium_magnitude':0.60,'low_magnitude':0.63}},'min_ensemble_confidence':0.50,'confidence_boost_threshold':15.0,'confidence_boost_amount':0.05,'description':'Production-tuned: DOWN must beat UP by 7%, stricter DOWN thresholds for 55/45 split'}
