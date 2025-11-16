import atexit,json,logging,sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from config import PREDICTION_DB_CONFIG
logger=logging.getLogger(__name__)
class CommitTracker:
    def __init__(self):
        self.pending_writes=0;self.last_commit_time=None;self.writes_log=[]
    def track_write(self,operation):
        self.pending_writes+=1;self.writes_log.append(f"{datetime.now():%H:%M:%S} - {operation}")
        if self.pending_writes%10==0:logger.warning(f"⚠️  {self.pending_writes} uncommitted writes! Call commit() soon!")
    def verify_clean_exit(self):
        if self.pending_writes>0:
            logger.error("="*80);logger.error("🚨 CRITICAL: UNCOMMITTED DATA DETECTED!");logger.error("="*80);logger.error(f"   Pending writes: {self.pending_writes}");logger.error(f"   Last 5 operations:")
            for op in self.writes_log[-5:]:logger.error(f"      • {op}")
            logger.error("");logger.error("   DATA WAS NOT SAVED!");logger.error("   You must call commit() before exit");logger.error("="*80)
class PredictionDatabase:
    def __init__(self,db_path=None):
        if db_path is None:db_path=PREDICTION_DB_CONFIG["db_path"]
        self.db_path=Path(db_path);self.db_path.parent.mkdir(parents=True,exist_ok=True);self.conn=sqlite3.connect(self.db_path,check_same_thread=False);self.conn.row_factory=sqlite3.Row;self._create_schema();self._pending_keys=set();self._commit_tracker=CommitTracker();atexit.register(self._commit_tracker.verify_clean_exit);logger.info(f"✅ Database initialized: {self.db_path}")
    def _create_schema(self):
        cursor=self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecasts'");table_exists=cursor.fetchone()is not None
        if not table_exists:
            create_sql="""
                CREATE TABLE forecasts (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    observation_date DATE NOT NULL,
                    forecast_date DATE NOT NULL,
                    horizon INTEGER NOT NULL,
                    calendar_cohort TEXT,
                    cohort_weight REAL,
                    median_forecast REAL NOT NULL,
                    point_estimate REAL NOT NULL,
                    q10 REAL,
                    q25 REAL,
                    q50 REAL,
                    q75 REAL,
                    q90 REAL,
                    prob_up REAL,
                    prob_down REAL,
                    direction_probability REAL,
                    confidence_score REAL,
                    feature_quality REAL,
                    num_features_used INTEGER,
                    current_vix REAL,
                    features_used TEXT,
                    model_version TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    actual_vix_change REAL,
                    actual_regime TEXT,
                    point_error REAL,
                    quantile_coverage TEXT,
                    UNIQUE(forecast_date, horizon)
                )
            """
            self.conn.execute(create_sql);self.conn.commit();logger.info("✅ Database schema created")
        else:logger.info("✅ Database schema verified")
        indexes=["CREATE INDEX IF NOT EXISTS idx_observation_date ON forecasts(observation_date)","CREATE INDEX IF NOT EXISTS idx_forecast_date ON forecasts(forecast_date)","CREATE INDEX IF NOT EXISTS idx_cohort ON forecasts(calendar_cohort)","CREATE INDEX IF NOT EXISTS idx_has_actual ON forecasts(actual_vix_change) WHERE actual_vix_change IS NOT NULL","CREATE INDEX IF NOT EXISTS idx_created_at ON forecasts(created_at)"]
        for index_sql in indexes:
            try:self.conn.execute(index_sql)
            except sqlite3.OperationalError:pass
        self.conn.commit()
    def store_prediction(self,record):
        record=record.copy()
        for key in["timestamp","observation_date","forecast_date","created_at"]:
            if key in record and isinstance(record[key],pd.Timestamp):record[key]=record[key].isoformat()
        if "created_at"not in record:record["created_at"]=datetime.now().isoformat()
        if "timestamp"not in record or record["timestamp"]is None:record["timestamp"]=datetime.now().isoformat()
        key=(record["forecast_date"],record["horizon"])
        if key in self._pending_keys:
            logger.warning(f"⚠️  Prediction already pending for {record['forecast_date']} (horizon={record['horizon']}). Skipping.")
            return None
        self._pending_keys.add(key)
        try:
            cursor=self.conn.execute("SELECT prediction_id FROM forecasts WHERE forecast_date = ? AND horizon = ?",(record["forecast_date"],record["horizon"]));existing=cursor.fetchone()
            if existing:
                logger.warning(f"⚠️  Prediction already exists for {record['forecast_date']} (horizon={record['horizon']}). Skipping.");self._pending_keys.discard(key)
                return None
            columns=list(record.keys());placeholders=", ".join(["?"for _ in columns]);insert_sql=f"INSERT INTO forecasts ({', '.join(columns)}) VALUES ({placeholders})";values=[record[col]for col in columns];self.conn.execute(insert_sql,values);self._commit_tracker.track_write(f"INSERT forecast_date={record['forecast_date']}");logger.debug(f"💾 Stored prediction: {record['prediction_id']}")
            return record["prediction_id"]
        except sqlite3.IntegrityError as e:
            logger.error(f"❌ Duplicate or constraint violation: {e}");self.conn.rollback();self._pending_keys.discard(key)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to store prediction: {e}");logger.error(f"   Record: {record.get('forecast_date')}, horizon={record.get('horizon')}");logger.error(f"   Available keys: {list(record.keys())[:10]}...");self.conn.rollback();self._pending_keys.discard(key)
            raise
    def commit(self):
        if self._commit_tracker.pending_writes==0:
            logger.info("ℹ️  No pending writes to commit")
            return
        writes_to_commit=self._commit_tracker.pending_writes
        try:
            self.conn.commit();self._pending_keys.clear();cursor=self.conn.execute("SELECT COUNT(*) FROM forecasts");total=cursor.fetchone()[0];logger.info("="*80);logger.info("✅ COMMIT SUCCESSFUL");logger.info("="*80);logger.info(f"   Writes committed: {writes_to_commit}");logger.info(f"   Total forecasts: {total}");logger.info(f"   Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}");logger.info("="*80);self._commit_tracker.pending_writes=0;self._commit_tracker.writes_log=[];self._commit_tracker.last_commit_time=datetime.now()
        except Exception as e:
            logger.error("="*80);logger.error("🚨 COMMIT FAILED!");logger.error("="*80);logger.error(f"   Error: {e}");logger.error(f"   Attempted writes: {writes_to_commit}");logger.error("   Rolling back...");self.conn.rollback();self._commit_tracker.pending_writes=0;self._commit_tracker.writes_log=[];logger.error("="*80)
            raise RuntimeError(f"Database commit failed: {e}")
    def get_commit_status(self):
        return{"pending_writes":self._commit_tracker.pending_writes,"last_commit":self._commit_tracker.last_commit_time.isoformat()if self._commit_tracker.last_commit_time else None,"recent_operations":self._commit_tracker.writes_log[-10:]}
    def get_predictions(self,start_date=None,end_date=None,cohort=None,with_actuals=False):
        query="SELECT DISTINCT * FROM forecasts WHERE 1=1";params=[]
        if start_date:
            query+=" AND forecast_date >= ?";params.append(start_date)
        if end_date:
            query+=" AND forecast_date <= ?";params.append(end_date)
        if cohort:
            query+=" AND calendar_cohort = ?";params.append(cohort)
        if with_actuals:query+=" AND actual_vix_change IS NOT NULL"
        query+=" ORDER BY forecast_date"
        try:
            df=pd.read_sql_query(query,self.conn,params=params,parse_dates=["observation_date","forecast_date","timestamp"]);before_count=len(df);df=df.drop_duplicates(subset=["forecast_date","horizon"],keep="first");after_count=len(df)
            if before_count!=after_count:logger.warning(f"⚠️  Removed {before_count - after_count} duplicate rows from query results")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to query predictions: {e}")
            raise
    def backfill_actuals(self,fetcher=None):
        if fetcher is None:
            from core.data_fetcher import UnifiedDataFetcher
            fetcher=UnifiedDataFetcher()
        logger.info("Starting actuals backfill...");vix_data=fetcher.fetch_yahoo("^VIX",start_date="2009-01-01")["Close"];vix_dates=set(vix_data.index.strftime("%Y-%m-%d"));query="""
            SELECT prediction_id, forecast_date, horizon, current_vix,
                   q10, q25, q50, q75, q90, median_forecast
            FROM forecasts
            WHERE actual_vix_change IS NULL
        """
        cursor=self.conn.cursor();cursor.execute(query);rows=cursor.fetchall()
        if len(rows)==0:
            logger.info("No predictions need backfilling")
            return
        logger.info(f"   Found {len(rows)} predictions to backfill");updated=0;skipped=0
        for row in rows:
            pred_id,forecast_date,horizon,current_vix,q10,q25,q50,q75,q90,median_forecast=row
            try:target_date_attempt=pd.bdate_range(start=pd.Timestamp(forecast_date),periods=horizon+1)[-1]
            except Exception as e:
                logger.warning(f"   Failed to calculate business days for {forecast_date}: {e}");skipped+=1
                continue
            found_date=None
            for offset in range(4):
                check_date=(target_date_attempt+pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if check_date in vix_dates:
                    found_date=check_date
                    break
            if found_date is None:
                skipped+=1
                continue
            target_date=found_date;actual_vix=vix_data.loc[pd.Timestamp(target_date)];actual_change=((actual_vix-current_vix)/current_vix)*100;regime='Low'if actual_change<-5 else('Normal'if actual_change<10 else('Elevated'if actual_change<25 else'Crisis'));point_error=abs(actual_change-median_forecast);quantile_coverage={"q10":1 if actual_change<=q10 else 0,"q25":1 if actual_change<=q25 else 0,"q50":1 if actual_change<=q50 else 0,"q75":1 if actual_change<=q75 else 0,"q90":1 if actual_change<=q90 else 0};coverage_json=json.dumps(quantile_coverage);cursor.execute("""
                UPDATE forecasts
                SET actual_vix_change = ?,
                    actual_regime = ?,
                    point_error = ?,
                    quantile_coverage = ?
                WHERE prediction_id = ?
                """,(actual_change,regime,point_error,coverage_json,pred_id));updated+=1
        self.conn.commit();logger.info(f"✅ Backfilled {updated} predictions")
        if skipped>0:logger.info(f"⚠️  Skipped {skipped} predictions (target date not yet available)")
    def compute_quantile_coverage(self,cohort=None):
        df=self.get_predictions(cohort=cohort,with_actuals=True)
        if len(df)==0:
            logger.warning("No predictions with actuals")
            return{}
        coverage={}
        for q in[10,25,50,75,90]:
            col=f"q{q}";covered=(df["actual_vix_change"]<=df[col]).mean();coverage[col]=float(covered)
        return coverage
    def compute_regime_brier_score(self,cohort=None):
        df=self.get_predictions(cohort=cohort,with_actuals=True)
        if len(df)==0:return np.nan
        brier_scores=[]
        for _,row in df.iterrows():
            actual_onehot={"low":0,"normal":0,"elevated":0,"crisis":0}
            if row["actual_regime"]:actual_onehot[row["actual_regime"].lower()]=1
            brier=(row["prob_low"]-actual_onehot["low"])**2+(row["prob_normal"]-actual_onehot["normal"])**2+(row["prob_elevated"]-actual_onehot["elevated"])**2+(row["prob_crisis"]-actual_onehot["crisis"])**2;brier_scores.append(brier)
        return float(np.mean(brier_scores))
    def get_performance_summary(self):
        df=self.get_predictions(with_actuals=True)
        if len(df)==0:return{"error":"No predictions with actuals"}
        summary={"n_predictions":len(df),"date_range":{"start":df["forecast_date"].min().isoformat(),"end":df["forecast_date"].max().isoformat()},"point_estimate":{"mae":float(df["point_error"].mean()),"rmse":float(np.sqrt(((df["actual_vix_change"]-df["point_estimate"])**2).mean()))},"quantile_coverage":self.compute_quantile_coverage(),"regime_brier_score":self.compute_regime_brier_score(),"confidence_correlation":float(df[["confidence_score","point_error"]].corr().iloc[0,1])if"confidence_score"in df.columns and"point_error"in df.columns else None};summary["by_cohort"]={}
        for cohort in df["calendar_cohort"].unique():summary["by_cohort"][cohort]={"n":int((df["calendar_cohort"]==cohort).sum()),"mae":float(df[df["calendar_cohort"]==cohort]["point_error"].mean()),"quantile_coverage":self.compute_quantile_coverage(cohort),"brier_score":self.compute_regime_brier_score(cohort)}
        return summary
    def export_to_csv(self,filename="predictions_export.csv"):
        df=self.get_predictions();df.to_csv(filename,index=False);logger.info(f"📄 Exported {len(df)} predictions to {filename}")
    def close(self):
        self.conn.close();logger.info("🔒 Database connection closed")
