import json,logging,os,warnings
from calendar import monthrange
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np,pandas as pd,requests,yfinance as yf
from dotenv import load_dotenv
load_dotenv();warnings.filterwarnings("ignore")
class DataFetchLogger:
    def __init__(self,name="DataFetcher"):
        self.logger=logging.getLogger(name);self.logger.handlers.clear();self.logger.propagate=False;h=logging.StreamHandler();h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",datefmt="%H:%M:%S"));self.logger.addHandler(h);self.logger.setLevel(logging.WARNING)
    def info(self,msg):self.logger.info(msg)
    def warning(self,msg):self.logger.warning(msg)
    def error(self,msg):self.logger.error(msg)
class UnifiedDataFetcher:
    def __init__(self,cache_dir="./data_cache",cboe_data_dir="./CBOE_Data_Archive"):
        from config import FRED_SERIES_METADATA,FORWARD_FILL_LIMITS,MIN_ROWS_BY_FREQUENCY,PUBLICATION_LAGS
        self.fred_metadata=FRED_SERIES_METADATA;self.forward_fill_limits=FORWARD_FILL_LIMITS;self.min_rows_by_freq=MIN_ROWS_BY_FREQUENCY;self.publication_lags=PUBLICATION_LAGS;self.fred_base_url="https://api.stlouisfed.org/fred/series/observations";self.fred_api_key=os.getenv("FRED_API_KEY");self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(exist_ok=True);self.cboe_data_dir=Path(cboe_data_dir);self.logger=DataFetchLogger()
        if not self.fred_api_key:self.logger.warning("FRED API key missing - FRED fetches will fail")
        self.cboe_files={"SKEW":"SKEW_INDEX_CBOE.csv","PCCI":"PCCI_INDX_CBOE.csv","PCCE":"PCCE_EQUITIES_CBOE.csv","PCC":"PCC_INDX_EQ_TOTAL_CBOE.csv","COR1M":"COR1M_CBOE.csv","COR3M":"COR3M_CBOE.csv","VXTH":"VXTH_TAILHEDGE_CBOE.csv","CNDR":"CNDR_SPX_IRON_CONDOR_CBOE.csv","VX1-VX2":"VX1-VX2.csv","VX2-VX1_RATIO":"(VX2-VX1)OVER(VX1).csv","CL1-CL2":"CL1-CL2.csv","DX1-DX2":"DX1-DX2.csv","BVOL":"BVOL.csv","BFLY":"BFLY.csv","DSPX":"DSPX.csv","VPN":"VPN.csv","GAMMA":"GAMMA.csv","VIX_VIX3M":"VIXoverVIX3M.csv","VXTLT":"VXTLT.csv","GOLDSILVER":"GOLDSILVER_RATIO.csv"}
    def _get_frequency_metadata(self,series_id):
        if series_id in self.fred_metadata:return self.fred_metadata[series_id]["frequency"],self.fred_metadata[series_id]["lag"]
        return "daily",self.publication_lags.get(series_id,0)
    def _get_forward_fill_limit(self,frequency):
        return self.forward_fill_limits.get(frequency,3)
    def _get_min_rows(self,frequency):
        return self.min_rows_by_freq.get(frequency,10)
    def _apply_publication_lag(self,series,lag_days):
        if lag_days==0:return series
        return series.shift(lag_days)
    def _get_last_date_from_cache(self,cache_path):
        try:
            if cache_path.exists():
                df=pd.read_parquet(cache_path)
                if not df.empty:return df.index[-1].strftime("%Y-%m-%d")
        except:pass
        return None
    def _merge_with_cache(self,new_data,cache_path):
        try:
            if cache_path.exists()and new_data is not None and not new_data.empty:
                cached_df=pd.read_parquet(cache_path)
                if not cached_df.empty:combined=pd.concat([cached_df,new_data]);combined=combined[~combined.index.duplicated(keep="last")];return combined.sort_index()
        except Exception as e:self.logger.warning(f"Cache merge failed: {e}")
        return new_data
    def _normalize_data(self,data,name,frequency="daily"):
        if data is None or(hasattr(data,"empty")and data.empty):return None
        if isinstance(data,pd.Series):data=pd.DataFrame(data)
        if not isinstance(data.index,pd.DatetimeIndex):
            try:data.index=pd.to_datetime(data.index)
            except Exception as e:self.logger.error(f"{name}: Index conversion failed - {e}");return None
        if data.index.tz is not None:data.index=data.index.tz_localize(None)
        data.index=pd.DatetimeIndex(data.index.date)
        if data.index.duplicated().any():data=data[~data.index.duplicated(keep="last")]
        min_rows=self._get_min_rows(frequency)
        return None if len(data)<min_rows else data
    def _should_update_fred_series(self,series_id,last_date):
        freq,lag=self._get_frequency_metadata(series_id);last_dt=pd.to_datetime(last_date);now=datetime.now()
        if freq=="quarterly":
            quarter_end_months={3:"Q1",6:"Q2",9:"Q3",12:"Q4"};last_quarter_month=max([m for m in quarter_end_months.keys()if m<=last_dt.month])
            if now.month>last_quarter_month:quarter_end=datetime(last_dt.year if last_quarter_month<12 else last_dt.year+1,last_quarter_month+1 if last_quarter_month<12 else 1,1);release_date=quarter_end+timedelta(days=35);return now>=release_date
            return False
        if freq=="monthly":
            if last_dt.month==now.month and last_dt.year==now.year:return False
            if now.month==last_dt.month+1 or(now.month==1 and last_dt.month==12):return now.day>=lag+7
            return True
        if freq=="weekly":return(now-last_dt).days>=7
        return len(pd.bdate_range(last_dt,now))-1>1
    def fetch_fred(self,series_id,start_date=None,end_date=None,incremental=True):
        if not self.fred_api_key:self.logger.warning(f"FRED:{series_id}: Cannot fetch - API key missing");return None
        freq,lag=self._get_frequency_metadata(series_id);cache_path=self.cache_dir/f"fred_{series_id}.parquet";fetch_start=start_date
        if incremental and cache_path.exists():
            last_date=self._get_last_date_from_cache(cache_path)
            if last_date:
                if not self._should_update_fred_series(series_id,last_date):
                    try:
                        cached_df=pd.read_parquet(cache_path)
                        if series_id in cached_df.columns:return cached_df[series_id]
                    except:pass
                fetch_start=(pd.to_datetime(last_date)+timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            params={"series_id":series_id,"api_key":self.fred_api_key,"file_type":"json","observation_start":fetch_start or"1990-01-01","observation_end":end_date or datetime.now().strftime("%Y-%m-%d")};response=requests.get(self.fred_base_url,params=params,timeout=10);response.raise_for_status();data=response.json()
            if"observations"not in data or not data["observations"]:
                if cache_path.exists():
                    try:
                        cached_df=pd.read_parquet(cache_path)
                        if series_id in cached_df.columns:return cached_df[series_id]
                    except:pass
                return None
            obs=data["observations"];df=pd.DataFrame(obs);df["date"]=pd.to_datetime(df["date"]);df=df[df["value"]!="."];df["value"]=pd.to_numeric(df["value"],errors="coerce");df=df.dropna(subset=["value"]);df=df.set_index("date")[["value"]];df.columns=[series_id];df=self._normalize_data(df,f"FRED:{series_id}",frequency=freq)
            if df is None:
                if cache_path.exists():
                    try:
                        cached_df=pd.read_parquet(cache_path)
                        if series_id in cached_df.columns:return cached_df[series_id]
                    except:pass
                return None
            if freq in["quarterly","monthly","weekly"]:date_range=pd.date_range(df.index[0],df.index[-1],freq="D");df=df.reindex(date_range,method="ffill");df.index=pd.DatetimeIndex(df.index.date)
            if incremental and cache_path.exists():df=self._merge_with_cache(df,cache_path)
            df.to_parquet(cache_path);return df[series_id]if series_id in df.columns else None
        except Exception as e:
            if cache_path.exists():
                try:
                    cached_df=pd.read_parquet(cache_path)
                    if series_id in cached_df.columns:return cached_df[series_id]
                except:pass
            self.logger.warning(f"FRED:{series_id}: Fetch failed - {str(e)[:100]}");return None
    def fetch_fred_with_lag_applied(self,series_id,target_index,start_date=None,end_date=None):
        series=self.fetch_fred(series_id,start_date,end_date,incremental=True)
        if series is None:return None
        freq,lag=self._get_frequency_metadata(series_id);ff_limit=self._get_forward_fill_limit(freq);lagged_series=self._apply_publication_lag(series,lag);reindexed=lagged_series.reindex(target_index,method="ffill",limit=ff_limit);return reindexed
    def fetch_all_fred_labor(self,target_index,start_date=None,end_date=None):
        labor_series={};series_ids=["ICSA","UNRATE","PAYEMS"]
        for sid in series_ids:
            s=self.fetch_fred_with_lag_applied(sid,target_index,start_date,end_date)
            if s is not None:labor_series[sid]=s
        return labor_series
    def fetch_all_fred_stress(self,target_index,start_date=None,end_date=None):
        stress_series={};series_ids=["STLFSI4"]
        for sid in series_ids:
            s=self.fetch_fred_with_lag_applied(sid,target_index,start_date,end_date)
            if s is not None:stress_series[sid]=s
        return stress_series
    def fetch_all_fred_credit(self,target_index,start_date=None,end_date=None):
        credit_series={};series_ids=["BAMLH0A0HYM2","BAMLH0A1HYBB","BAMLH0A2HYB","BAMLH0A3HYC","BAMLC0A0CM"]
        for sid in series_ids:
            s=self.fetch_fred_with_lag_applied(sid,target_index,start_date,end_date)
            if s is not None:credit_series[sid]=s
        return credit_series
    def fetch_all_fred_rates(self,target_index,start_date=None,end_date=None):
        rate_series={};series_ids=["DFF","SOFR","SOFR90DAYAVG"]
        for sid in series_ids:
            s=self.fetch_fred_with_lag_applied(sid,target_index,start_date,end_date)
            if s is not None:rate_series[sid]=s
        return rate_series
    def fetch_all_fred_treasuries(self,target_index,start_date=None,end_date=None):
        treasury_series={};series_ids=["DGS3MO","DGS2","DGS5","DGS10","DGS30"]
        for sid in series_ids:
            s=self.fetch_fred_with_lag_applied(sid,target_index,start_date,end_date)
            if s is not None:treasury_series[sid]=s
        return treasury_series
    def fetch_all_fred_inflation(self,target_index,start_date=None,end_date=None):
        inflation_series={};series_ids=["CPIAUCSL","CPILFESL"]
        for sid in series_ids:
            s=self.fetch_fred_with_lag_applied(sid,target_index,start_date,end_date)
            if s is not None:inflation_series[sid]=s
        return inflation_series
    def fetch_yahoo(self,symbol,start_date=None,end_date=None,incremental=True):
        cache_path=self.cache_dir/f"yahoo_{symbol.replace('^','').replace('=','_')}.parquet"
        if end_date:end_dt=pd.to_datetime(end_date);is_historical_request=end_dt<(datetime.now()-timedelta(days=7))
        else:end_dt=None;is_historical_request=False
        if is_historical_request and cache_path.exists():
            try:
                cached_df=pd.read_parquet(cache_path)
                if not cached_df.empty:
                    start_dt=pd.to_datetime(start_date)if start_date else cached_df.index[0]
                    cache_start,cache_end=cached_df.index[0],cached_df.index[-1]
                    if(cache_start<=start_dt)and(cache_end>=end_dt):
                        filtered_df=cached_df[(cached_df.index>=start_dt)&(cached_df.index<=end_dt)]
                        return filtered_df if not filtered_df.empty else None
            except Exception as e:self.logger.warning(f"Yahoo:{symbol}: Cache read failed - {e}")
        fetch_start=start_date or"2000-01-01";is_incremental_fetch=False
        if is_historical_request:incremental=False
        if incremental and cache_path.exists():
            last_date=self._get_last_date_from_cache(cache_path)
            if last_date:
                last_dt=pd.to_datetime(last_date);now=datetime.now();business_days_diff=len(pd.bdate_range(last_dt,now))-1;cache_mtime=datetime.fromtimestamp(cache_path.stat().st_mtime);hours_since_update=(now-cache_mtime).total_seconds()/3600
                if business_days_diff==0 and hours_since_update<1:
                    try:
                        cached_df=pd.read_parquet(cache_path)
                        if not cached_df.empty:return cached_df
                    except:pass
                elif business_days_diff==1 and now.hour<16:
                    try:
                        cached_df=pd.read_parquet(cache_path)
                        if not cached_df.empty:return cached_df
                    except:pass
                fetch_start=(last_dt-timedelta(days=2)).strftime("%Y-%m-%d");is_incremental_fetch=True
        try:
            ticker=yf.Ticker(symbol)
            if end_date:fetch_end=(pd.Timestamp(end_date)+timedelta(days=1)).strftime("%Y-%m-%d")
            else:fetch_end=(datetime.now()+timedelta(days=2)).strftime("%Y-%m-%d")
            df=ticker.history(start=fetch_start,end=fetch_end,auto_adjust=True)
            if df.empty:
                if cache_path.exists():
                    try:
                        cached_df=pd.read_parquet(cache_path)
                        if not cached_df.empty:
                            if not is_incremental_fetch:self.logger.warning(f"Yahoo:{symbol}: No new data available, using existing cache")
                            return cached_df
                    except:pass
                return None
            min_rows=1 if is_incremental_fetch else self._get_min_rows("daily");df=self._normalize_data(df,f"Yahoo:{symbol}",frequency="daily")
            if df is None:
                if cache_path.exists():
                    try:
                        cached_df=pd.read_parquet(cache_path)
                        if not is_incremental_fetch:self.logger.warning(f"Yahoo:{symbol}: Data processing issue, using existing cache")
                        return cached_df
                    except:pass
                return None
            if incremental and cache_path.exists():df=self._merge_with_cache(df,cache_path)
            df.to_parquet(cache_path)
            if is_historical_request:
                start_dt=pd.to_datetime(start_date)if start_date else df.index[0]
                df=df[(df.index>=start_dt)&(df.index<=end_dt)]
            return df
        except Exception as e:
            if cache_path.exists():
                try:
                    cached_df=pd.read_parquet(cache_path)
                    if not cached_df.empty:
                        self.logger.warning(f"Yahoo:{symbol}: Fetch failed, using existing cache - {str(e)[:100]}")
                        if is_historical_request:
                            start_dt=pd.to_datetime(start_date)if start_date else cached_df.index[0]
                            cached_df=cached_df[(cached_df.index>=start_dt)&(cached_df.index<=end_dt)]
                        return cached_df
                except:pass
            self.logger.error(f"Yahoo:{symbol}: Fetch failed - {str(e)[:100]}");return None
    def fetch_price(self,symbol):
        try:
            ticker=yf.Ticker(symbol);data=ticker.history(period="1d")
            if not data.empty:return float(data["Close"].iloc[-1])
        except Exception as e:self.logger.warning(f"Price fetch failed for {symbol}: {str(e)[:100]}")
        return None
    def fetch_cboe_series(self,symbol):
        if symbol not in self.cboe_files:self.logger.warning(f"CBOE:{symbol}: Unknown symbol");return None
        file_path=self.cboe_data_dir/self.cboe_files[symbol]
        if not file_path.exists():self.logger.warning(f"CBOE:{symbol}: File not found at {file_path}");return None
        try:
            df=pd.read_csv(file_path,parse_dates=[0],index_col=0,converters={1:lambda x:pd.to_numeric(x,errors="coerce")})
            for col in df.columns:df[col]=pd.to_numeric(df[col],errors="coerce")
            df=self._normalize_data(df,f"CBOE:{symbol}",frequency="daily")
            if df is None or df.empty:return None
            series=df.iloc[:,0]if len(df.columns)>0 else None
            if series is not None:series=series.astype(np.float64)
            return series
        except Exception as e:self.logger.error(f"CBOE:{symbol}: Load failed - {str(e)[:100]}");return None
    def fetch_all_cboe(self):
        series_dict={}
        for symbol in self.cboe_files.keys():
            series=self.fetch_cboe_series(symbol)
            if series is not None:series_dict[symbol]=series
        return series_dict
    def fetch_fomc_calendar(self,start_year,end_year):
        cache_path=self.cache_dir/"fomc_calendar.csv"
        if not cache_path.exists():self.logger.error(f"FOMC: Calendar file not found at {cache_path}");return None
        try:
            df=pd.read_csv(cache_path,parse_dates=["date"])
            if"date"not in df.columns or"meeting_type"not in df.columns:self.logger.error("FOMC: Invalid CSV format. Required columns: date, meeting_type");return None
            df=df.set_index("date");df=self._normalize_data(df,"FOMC:calendar",frequency="daily")
            if df is None or df.empty:return None
            df=df[(df.index.year>=start_year)&(df.index.year<=end_year)]
            if df.empty:return None
            return df
        except Exception as e:self.logger.error(f"FOMC: Failed to load calendar - {str(e)[:100]}");return None
