import logging,sqlite3,json
from pathlib import Path
import pandas as pd
import numpy as np
logger=logging.getLogger(__name__)

class AnomalyDatabaseExtension:
    def __init__(self,db_path="data_cache/predictions.db"):
        self.db_path=Path(db_path);self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(self.db_path,check_same_thread=False);self.conn.row_factory=sqlite3.Row

    def extend_schema(self):
        logger.info("🔧 Extending schema...")

        # Ensure forecasts table exists first
        cursor=self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecasts'")
        table_exists=cursor.fetchone()is not None

        if not table_exists:
            # Create the forecasts table if it doesn't exist
            create_sql="""CREATE TABLE forecasts (prediction_id TEXT PRIMARY KEY,timestamp DATETIME NOT NULL,observation_date DATE NOT NULL,forecast_date DATE NOT NULL,horizon INTEGER NOT NULL,calendar_cohort TEXT,cohort_weight REAL,prob_up REAL,prob_down REAL,magnitude_forecast REAL,expected_vix REAL,feature_quality REAL,num_features_used INTEGER,current_vix REAL,actual_vix_change REAL,actual_direction INTEGER,direction_correct INTEGER,magnitude_error REAL,correction_type TEXT,features_used TEXT,model_version TEXT,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,direction_probability REAL,direction_prediction TEXT,UNIQUE(forecast_date,horizon))"""
            self.conn.execute(create_sql);self.conn.commit()
            logger.info("✅ Created forecasts table")

        cursor=self.conn.execute("PRAGMA table_info(forecasts)")
        existing_cols={row[1]for row in cursor.fetchall()}
        new_columns=[('anomaly_score_prior_day','REAL'),('anomaly_alerts_prior_day','INTEGER'),('spike_gate_triggered','BOOLEAN'),('spike_gate_action','TEXT'),('spike_gate_original_direction','TEXT'),('spike_gate_original_magnitude','REAL'),('spike_gate_confidence_boost','REAL')]
        added=0
        for col_name,col_type in new_columns:
            if col_name not in existing_cols:
                try:
                    self.conn.execute(f"ALTER TABLE forecasts ADD COLUMN {col_name} {col_type}")
                    logger.info(f"  ✓ {col_name}");added+=1
                except sqlite3.OperationalError:pass
        if added>0:self.conn.commit();logger.info(f"✅ Added {added} columns")
        else:logger.info("✅ All columns exist")
        self._create_anomaly_scores_table();self._create_indexes()

    def _create_anomaly_scores_table(self):
        create_sql="""CREATE TABLE IF NOT EXISTS anomaly_scores (
            score_id TEXT PRIMARY KEY,
            observation_date DATE NOT NULL UNIQUE,
            timestamp DATETIME NOT NULL,
            anomaly_score REAL NOT NULL,
            score_level TEXT,
            vix_value REAL,
            vix_velocity_5d REAL,
            vix_accel_5d REAL,
            top_contributing_features TEXT,
            detector_scores TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )"""
        self.conn.execute(create_sql);self.conn.commit();logger.info("✅ Created anomaly_scores table")

    def _create_indexes(self):
        indexes=["CREATE INDEX IF NOT EXISTS idx_anomaly_score ON forecasts(anomaly_score_prior_day)","CREATE INDEX IF NOT EXISTS idx_spike_gate ON forecasts(spike_gate_triggered)","CREATE INDEX IF NOT EXISTS idx_anomaly_obs_date ON anomaly_scores(observation_date)","CREATE INDEX IF NOT EXISTS idx_anomaly_level ON anomaly_scores(score_level)"]
        for idx_sql in indexes:
            try:self.conn.execute(idx_sql)
            except sqlite3.OperationalError:pass
        self.conn.commit();logger.info("✅ Created indexes")

    def store_anomaly_score(self,record):
        from datetime import datetime
        score_id=f"score_{record['observation_date'].replace('-','')}"
        record_with_id={'score_id':score_id,'timestamp':datetime.now().isoformat(),**record}
        if 'top_contributing_features'in record_with_id and isinstance(record_with_id['top_contributing_features'],list):
            record_with_id['top_contributing_features']=json.dumps(record_with_id['top_contributing_features'])
        if 'detector_scores'in record_with_id and isinstance(record_with_id['detector_scores'],dict):
            record_with_id['detector_scores']=json.dumps(record_with_id['detector_scores'])
        columns=list(record_with_id.keys());placeholders=','.join(['?'for _ in columns])
        insert_sql=f"INSERT OR REPLACE INTO anomaly_scores ({','.join(columns)}) VALUES ({placeholders})"
        values=[record_with_id[col]for col in columns]
        self.conn.execute(insert_sql,values);self.conn.commit()

    def get_anomaly_scores(self,start_date=None,end_date=None,min_score=None):
        query="SELECT * FROM anomaly_scores WHERE 1=1";params=[]
        if start_date:query+=" AND observation_date>=?";params.append(start_date)
        if end_date:query+=" AND observation_date<=?";params.append(end_date)
        if min_score is not None:query+=" AND anomaly_score>=?";params.append(min_score)
        query+=" ORDER BY observation_date"
        return pd.read_sql_query(query,self.conn,params=params,parse_dates=['observation_date','timestamp'])

    def _safe_scalar_extract(self,df,date,column):
        """Extract scalar value from DataFrame at specific date/column"""
        if column not in df.columns:
            return None
        try:
            val=df.at[date,column]
            if pd.isna(val):
                return None
            if isinstance(val,(pd.Series,np.ndarray)):
                val=val.iloc[0]if isinstance(val,pd.Series)else val.item()
            return float(val)
        except:
            return None

    def backfill_anomaly_scores(self,anomaly_scorer,feature_data):
        logger.info(f"🔄 Backfilling {len(feature_data)} dates...")
        anomaly_results=anomaly_scorer.calculate_scores_batch(feature_data);stored=0
        for date in anomaly_results.index:
            obs_date=date.strftime('%Y-%m-%d')
            vix_value=self._safe_scalar_extract(feature_data,date,'vix')
            vix_vel=self._safe_scalar_extract(feature_data,date,'vix_velocity_5d')
            vix_accel=self._safe_scalar_extract(feature_data,date,'vix_accel_5d')
            scorer_result=anomaly_scorer.calculate_score(feature_data.loc[[date]])
            record={'observation_date':obs_date,'anomaly_score':float(anomaly_results.loc[date,'anomaly_score']),'score_level':anomaly_results.loc[date,'anomaly_level'],'vix_value':vix_value,'vix_velocity_5d':vix_vel,'vix_accel_5d':vix_accel,'top_contributing_features':scorer_result.get('contributing_features',[]),'detector_scores':{}}
            try:self.store_anomaly_score(record);stored+=1
            except Exception as e:logger.error(f"Failed {obs_date}: {e}")
        logger.info(f"✅ Backfilled {stored} scores")

    def update_forecasts_with_anomaly_scores(self):
        logger.info("🔄 Updating forecasts...")
        update_sql="""UPDATE forecasts SET anomaly_score_prior_day=(
            SELECT anomaly_score FROM anomaly_scores WHERE anomaly_scores.observation_date=forecasts.observation_date
        ) WHERE EXISTS (
            SELECT 1 FROM anomaly_scores WHERE anomaly_scores.observation_date=forecasts.observation_date
        )"""
        cursor=self.conn.execute(update_sql);updated=cursor.rowcount;self.conn.commit()
        logger.info(f"✅ Updated {updated} forecasts");return updated

    def get_spike_gate_stats(self):
        query="""SELECT COUNT(*) as total_forecasts,
            SUM(CASE WHEN spike_gate_triggered=1 THEN 1 ELSE 0 END) as triggered_count,
            SUM(CASE WHEN spike_gate_action='OVERRIDE_TO_UP' THEN 1 ELSE 0 END) as override_to_up,
            SUM(CASE WHEN spike_gate_action='OVERRIDE_TO_NEUTRAL' THEN 1 ELSE 0 END) as override_to_neutral,
            SUM(CASE WHEN spike_gate_action='BOOST_CONFIRM' THEN 1 ELSE 0 END) as boost_confirm,
            AVG(CASE WHEN spike_gate_triggered=1 THEN anomaly_score_prior_day END) as avg_trigger_score
        FROM forecasts WHERE anomaly_score_prior_day IS NOT NULL"""
        cursor=self.conn.execute(query);result=cursor.fetchone()
        return {'total_forecasts':result[0],'triggered_count':result[1]or 0,'override_to_up':result[2]or 0,'override_to_neutral':result[3]or 0,'boost_confirm':result[4]or 0,'trigger_rate':(result[1]or 0)/result[0]if result[0]>0 else 0,'avg_trigger_score':result[5]or 0}

    def close(self):self.conn.close()

def extend_prediction_database():
    ext=AnomalyDatabaseExtension();ext.extend_schema();ext.close();print("✅ Schema extended")
