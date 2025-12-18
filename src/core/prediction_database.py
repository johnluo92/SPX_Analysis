import atexit,json,logging,sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from config import PREDICTION_DB_CONFIG
logger=logging.getLogger(__name__)
class CommitTracker:
    def __init__(self):
        self.pending_writes=0;self.last_commit_time=None;self.writes_log=[]
    def track_write(self,operation):
        self.pending_writes+=1;self.writes_log.append(f"{datetime.now():%H:%M:%S} - {operation}")
    def verify_clean_exit(self):
        if self.pending_writes>0:
            logger.error("="*80);logger.error("🚨 CRITICAL: UNCOMMITTED DATA");logger.error("="*80);logger.error(f"   Pending writes: {self.pending_writes}");logger.error(f"   Last 5 operations:");[logger.error(f"      • {op}")for op in self.writes_log[-5:]];logger.error("");logger.error("   DATA WAS NOT SAVED!");logger.error("="*80)
class PredictionDatabase:
    def __init__(self,db_path=None):
        if db_path is None:db_path=PREDICTION_DB_CONFIG["db_path"]
        self.db_path=Path(db_path);self.db_path.parent.mkdir(parents=True,exist_ok=True);self.conn=sqlite3.connect(self.db_path,check_same_thread=False);self.conn.row_factory=sqlite3.Row;self._create_schema();self._migrate_schema();self._pending_keys=set();self._commit_tracker=CommitTracker();atexit.register(self._commit_tracker.verify_clean_exit)
    def _create_schema(self):
        cursor=self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecasts'");table_exists=cursor.fetchone()is not None
        if not table_exists:
            create_sql="""CREATE TABLE forecasts (prediction_id TEXT PRIMARY KEY,timestamp DATETIME NOT NULL,observation_date DATE NOT NULL,forecast_date DATE NOT NULL,horizon INTEGER NOT NULL,calendar_cohort TEXT,cohort_weight REAL,prob_up REAL,prob_down REAL,magnitude_forecast REAL,expected_vix REAL,feature_quality REAL,num_features_used INTEGER,current_vix REAL,actual_vix_change REAL,actual_direction INTEGER,direction_correct INTEGER,magnitude_error REAL,correction_type TEXT,features_used TEXT,model_version TEXT,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,direction_probability REAL,direction_prediction TEXT,direction_confidence REAL,UNIQUE(forecast_date,horizon))"""
            self.conn.execute(create_sql);self.conn.commit();logger.info("✅ Database schema created")
        indexes=["CREATE INDEX IF NOT EXISTS idx_observation_date ON forecasts(observation_date)","CREATE INDEX IF NOT EXISTS idx_forecast_date ON forecasts(forecast_date)","CREATE INDEX IF NOT EXISTS idx_cohort ON forecasts(calendar_cohort)","CREATE INDEX IF NOT EXISTS idx_has_actual ON forecasts(actual_vix_change) WHERE actual_vix_change IS NOT NULL","CREATE INDEX IF NOT EXISTS idx_created_at ON forecasts(created_at)","CREATE INDEX IF NOT EXISTS idx_correction_type ON forecasts(correction_type)"]
        for index_sql in indexes:
            try:self.conn.execute(index_sql)
            except sqlite3.OperationalError:pass
        self.conn.commit()
    def _migrate_schema(self):
        cursor=self.conn.execute("PRAGMA table_info(forecasts)");cols={row[1]for row in cursor.fetchall()}
        migrations_needed=[]
        if "correction_type"not in cols:migrations_needed.append(("correction_type","ALTER TABLE forecasts ADD COLUMN correction_type TEXT"))
        if "direction_probability"not in cols:migrations_needed.append(("direction_probability","ALTER TABLE forecasts ADD COLUMN direction_probability REAL"))
        if "direction_prediction"not in cols:migrations_needed.append(("direction_prediction","ALTER TABLE forecasts ADD COLUMN direction_prediction TEXT"))
        if "direction_confidence"not in cols:migrations_needed.append(("direction_confidence","ALTER TABLE forecasts ADD COLUMN direction_confidence REAL"))
        if migrations_needed:
            logger.info(f"Migrating database: adding {len(migrations_needed)} columns")
            for col_name,sql in migrations_needed:
                try:self.conn.execute(sql);logger.info(f"  Added: {col_name}")
                except sqlite3.OperationalError as e:logger.warning(f"  Already exists: {col_name}")
            self.conn.commit();logger.info("✅ Migration complete")
    def store_prediction(self,record):
        record=record.copy()
        for key in["timestamp","observation_date","forecast_date","created_at"]:
            if key in record and isinstance(record[key],pd.Timestamp):record[key]=record[key].isoformat()
        if "created_at"not in record:record["created_at"]=datetime.now().isoformat()
        if "timestamp"not in record or record["timestamp"]is None:record["timestamp"]=datetime.now().isoformat()
        key=(record["forecast_date"],record["horizon"])
        if key in self._pending_keys:logger.debug(f"Prediction already pending for {record['forecast_date']}");return None
        self._pending_keys.add(key)
        try:
            cursor=self.conn.execute("SELECT prediction_id FROM forecasts WHERE forecast_date=? AND horizon=?",(record["forecast_date"],record["horizon"]));existing=cursor.fetchone()
            if existing:logger.debug(f"Prediction exists for {record['forecast_date']}");self._pending_keys.discard(key);return None
            columns=sorted(list(record.keys()));placeholders=",".join(["?"for _ in columns]);insert_sql=f"INSERT INTO forecasts ({','.join(columns)}) VALUES ({placeholders})";values=[record[col]for col in columns];self.conn.execute(insert_sql,values);self._commit_tracker.track_write(f"INSERT {record['forecast_date']}");return record["prediction_id"]
        except sqlite3.IntegrityError as e:logger.error(f"❌ Duplicate: {e}");self.conn.rollback();self._pending_keys.discard(key);return None
        except Exception as e:logger.error(f"❌ Store failed: {e}");self.conn.rollback();self._pending_keys.discard(key);raise
    def commit(self):
        if self._commit_tracker.pending_writes==0:return
        writes=self._commit_tracker.pending_writes
        try:
            self.conn.commit();self._pending_keys.clear();self._commit_tracker.pending_writes=0;self._commit_tracker.writes_log=[];self._commit_tracker.last_commit_time=datetime.now()
        except Exception as e:logger.error("="*80);logger.error("🚨 COMMIT FAILED!");logger.error("="*80);logger.error(f"   Error: {e}");logger.error(f"   Attempted: {writes}");logger.error("   Rolling back...");self.conn.rollback();self._commit_tracker.pending_writes=0;self._commit_tracker.writes_log=[];logger.error("="*80);raise RuntimeError(f"Commit failed: {e}")
    def get_commit_status(self):
        return{"pending_writes":self._commit_tracker.pending_writes,"last_commit":self._commit_tracker.last_commit_time.isoformat()if self._commit_tracker.last_commit_time else None,"recent_operations":self._commit_tracker.writes_log[-10:]}
    def get_predictions(self,start_date=None,end_date=None,cohort=None,with_actuals=False):
        query="SELECT * FROM forecasts WHERE 1=1";params=[]
        if start_date:query+=" AND forecast_date>=?";params.append(start_date)
        if end_date:query+=" AND forecast_date<=?";params.append(end_date)
        if cohort:query+=" AND calendar_cohort=?";params.append(cohort)
        if with_actuals:query+=" AND actual_vix_change IS NOT NULL"
        query+=" ORDER BY forecast_date, horizon, prediction_id"
        try:
            df=pd.read_sql_query(query,self.conn,params=params,parse_dates=["observation_date","forecast_date","timestamp"])
            df = df.sort_values(['forecast_date', 'horizon', 'prediction_id'])
            before=len(df)
            df=df.drop_duplicates(subset=["forecast_date","horizon"],keep="last")
            after=len(df)
            if before!=after:logger.warning(f"⚠️  Removed {before-after} duplicates")
            df = df.sort_values(['forecast_date', 'horizon']).reset_index(drop=True)
            return df
        except Exception as e:logger.error(f"❌ Query failed: {e}");raise
    def backfill_actuals(self,fetcher=None):
        if fetcher is None:
            from core.data_fetcher import UnifiedDataFetcher
            fetcher=UnifiedDataFetcher()
        vix_data=fetcher.fetch_yahoo("^VIX",start_date="2009-01-01")["Close"]
        vix_dates=set(vix_data.index.strftime("%Y-%m-%d"))
        query="SELECT prediction_id,forecast_date,horizon,current_vix,direction_probability,direction_prediction,magnitude_forecast FROM forecasts WHERE actual_vix_change IS NULL ORDER BY forecast_date, horizon, prediction_id"
        cursor=self.conn.cursor();cursor.execute(query);rows=cursor.fetchall()
        if len(rows)==0:logger.debug("No predictions need backfilling");return
        updated=0;skipped=0
        for row in rows:
            pred_id=row[0];forecast_date=row[1];horizon=row[2];current_vix=row[3];direction_probability=row[4]if len(row)>4 else None;direction_prediction=row[5]if len(row)>5 else None;magnitude_forecast=row[6]if len(row)>6 else row[4]

            if direction_prediction == "NO_DECISION":
                skipped+=1
                continue

            try:target_date_attempt=pd.bdate_range(start=pd.Timestamp(forecast_date),periods=horizon+1)[-1]
            except Exception as e:logger.warning(f"   Business day calc failed for {forecast_date}: {e}");skipped+=1;continue
            found_date=None
            for offset in range(4):
                check_date=(target_date_attempt+pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if check_date in vix_dates:found_date=check_date;break
            if found_date is None:skipped+=1;continue
            actual_vix=vix_data.loc[pd.Timestamp(found_date)];actual_change=((actual_vix-current_vix)/current_vix)*100;actual_direction=1 if actual_change>0 else 0;magnitude_error=abs(actual_change-magnitude_forecast)
            if direction_prediction is not None:dir_correct=1 if(direction_prediction=="UP"and actual_direction==1)or(direction_prediction=="DOWN"and actual_direction==0)else 0;cursor.execute("UPDATE forecasts SET actual_vix_change=?,actual_direction=?,direction_correct=?,magnitude_error=? WHERE prediction_id=?",(actual_change,actual_direction,dir_correct,magnitude_error,pred_id))
            else:predicted_direction=1 if direction_probability>0.5 else 0;dir_correct=1 if actual_direction==predicted_direction else 0;cursor.execute("UPDATE forecasts SET actual_vix_change=?,actual_direction=?,direction_correct=?,magnitude_error=? WHERE prediction_id=?",(actual_change,actual_direction,dir_correct,magnitude_error,pred_id))
            updated+=1
        self.conn.commit()
        if updated>0:logger.info(f"✅ Backfilled {updated} predictions")
        if skipped>0:logger.debug(f"Skipped {skipped} predictions (NO_DECISION or no data)")
    def get_performance_summary(self):
        df=self.get_predictions(with_actuals=True)
        if len(df)==0:return{"error":"No predictions with actuals"}

        df_with_decision = df[df["direction_prediction"].isin(["UP", "DOWN"])].copy()

        if len(df_with_decision)==0:return{"error":"No predictions with directional decisions"}

        if "direction_correct"not in df_with_decision.columns or df_with_decision["direction_correct"].isna().all():
            if "direction_prediction"in df_with_decision.columns:df_with_decision["direction_correct"]=((df_with_decision["direction_prediction"]=="UP")&(df_with_decision["actual_direction"]==1))|((df_with_decision["direction_prediction"]=="DOWN")&(df_with_decision["actual_direction"]==0))
            else:df_with_decision["predicted_direction"]=(df_with_decision["prob_up"]>0.5).astype(int);df_with_decision["direction_correct"]=(df_with_decision["predicted_direction"]==df_with_decision["actual_direction"]).astype(int)
        if "magnitude_error"not in df_with_decision.columns or df_with_decision["magnitude_error"].isna().all():df_with_decision["magnitude_error"]=np.abs(df_with_decision["actual_vix_change"]-df_with_decision["magnitude_forecast"])

        summary={"n_predictions_total":len(df),"n_predictions_with_decision":len(df_with_decision),"n_no_decision":len(df) - len(df_with_decision),"no_decision_rate":float((len(df) - len(df_with_decision)) / len(df)),"date_range":{"start":df["forecast_date"].min().isoformat(),"end":df["forecast_date"].max().isoformat()},"direction":{"accuracy":float(df_with_decision["direction_correct"].mean()),"precision":float(df_with_decision[df_with_decision["actual_direction"]==1]["direction_correct"].mean())if(df_with_decision["actual_direction"]==1).sum()>0 else 0.0,"recall":float(df_with_decision[df_with_decision["actual_direction"]==1]["direction_correct"].mean())if(df_with_decision["actual_direction"]==1).sum()>0 else 0.0},"magnitude":{"mae":float(df_with_decision["magnitude_error"].mean()),"rmse":float(np.sqrt((df_with_decision["magnitude_error"]**2).mean())),"bias":float((df_with_decision["magnitude_forecast"]-df_with_decision["actual_vix_change"]).mean()),"median_error":float(df_with_decision["magnitude_error"].median())}};summary["by_cohort"]={}
        for cohort in sorted(df_with_decision["calendar_cohort"].unique()):
            cohort_df=df_with_decision[df_with_decision["calendar_cohort"]==cohort];summary["by_cohort"][cohort]={"n":int(len(cohort_df)),"direction_accuracy":float(cohort_df["direction_correct"].mean()),"magnitude_mae":float(cohort_df["magnitude_error"].mean()),"magnitude_bias":float((cohort_df["magnitude_forecast"]-cohort_df["actual_vix_change"]).mean())}
        return summary
    def export_to_csv(self,filename="predictions_export.csv"):
        df=self.get_predictions();df.to_csv(filename,index=False);logger.info(f"📄 Exported {len(df)} to {filename}")
    def close(self):
        self.conn.close()
