import warnings
from datetime import datetime,timedelta
from pathlib import Path
from typing import Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
import requests
import time
warnings.filterwarnings("ignore")
class VXContinuousContractBuilder:
    EXPIRATION_DATES=[(2013,1,'2013-01-16'),(2013,2,'2013-02-13'),(2013,3,'2013-03-20'),(2013,4,'2013-04-17'),(2013,5,'2013-05-22'),(2013,6,'2013-06-19'),(2013,7,'2013-07-17'),(2013,8,'2013-08-21'),(2013,9,'2013-09-18'),(2013,10,'2013-10-16'),(2013,11,'2013-11-20'),(2013,12,'2013-12-18'),(2014,1,'2014-01-22'),(2014,2,'2014-02-19'),(2014,3,'2014-03-18'),(2014,4,'2014-04-16'),(2014,5,'2014-05-21'),(2014,6,'2014-06-18'),(2014,7,'2014-07-16'),(2014,8,'2014-08-20'),(2014,9,'2014-09-17'),(2014,10,'2014-10-22'),(2014,11,'2014-11-19'),(2014,12,'2014-12-17'),(2015,1,'2015-01-21'),(2015,2,'2015-02-18'),(2015,3,'2015-03-18'),(2015,4,'2015-04-15'),(2015,5,'2015-05-20'),(2015,6,'2015-06-17'),(2015,7,'2015-07-22'),(2015,8,'2015-08-19'),(2015,9,'2015-09-16'),(2015,10,'2015-10-21'),(2015,11,'2015-11-18'),(2015,12,'2015-12-16'),(2016,1,'2016-01-20'),(2016,2,'2016-02-17'),(2016,3,'2016-03-16'),(2016,4,'2016-04-20'),(2016,5,'2016-05-18'),(2016,6,'2016-06-15'),(2016,7,'2016-07-20'),(2016,8,'2016-08-17'),(2016,9,'2016-09-21'),(2016,10,'2016-10-19'),(2016,11,'2016-11-16'),(2016,12,'2016-12-21'),(2017,1,'2017-01-18'),(2017,2,'2017-02-15'),(2017,3,'2017-03-22'),(2017,4,'2017-04-19'),(2017,5,'2017-05-17'),(2017,6,'2017-06-21'),(2017,7,'2017-07-19'),(2017,8,'2017-08-16'),(2017,9,'2017-09-20'),(2017,10,'2017-10-18'),(2017,11,'2017-11-15'),(2017,12,'2017-12-20'),(2018,1,'2018-01-17'),(2018,2,'2018-02-14'),(2018,3,'2018-03-21'),(2018,4,'2018-04-18'),(2018,5,'2018-05-16'),(2018,6,'2018-06-20'),(2018,7,'2018-07-18'),(2018,8,'2018-08-22'),(2018,9,'2018-09-19'),(2018,10,'2018-10-17'),(2018,11,'2018-11-21'),(2018,12,'2018-12-19'),(2019,1,'2019-01-16'),(2019,2,'2019-02-13'),(2019,3,'2019-03-19'),(2019,4,'2019-04-17'),(2019,5,'2019-05-22'),(2019,6,'2019-06-19'),(2019,7,'2019-07-17'),(2019,8,'2019-08-21'),(2019,9,'2019-09-18'),(2019,10,'2019-10-16'),(2019,11,'2019-11-20'),(2019,12,'2019-12-18'),(2020,1,'2020-01-22'),(2020,2,'2020-02-19'),(2020,3,'2020-03-18'),(2020,4,'2020-04-15'),(2020,5,'2020-05-20'),(2020,6,'2020-06-17'),(2020,7,'2020-07-22'),(2020,8,'2020-08-19'),(2020,9,'2020-09-16'),(2020,10,'2020-10-21'),(2020,11,'2020-11-18'),(2020,12,'2020-12-16'),(2021,1,'2021-01-20'),(2021,2,'2021-02-17'),(2021,3,'2021-03-17'),(2021,4,'2021-04-21'),(2021,5,'2021-05-19'),(2021,6,'2021-06-16'),(2021,7,'2021-07-21'),(2021,8,'2021-08-18'),(2021,9,'2021-09-15'),(2021,10,'2021-10-20'),(2021,11,'2021-11-17'),(2021,12,'2021-12-22'),(2022,1,'2022-01-19'),(2022,2,'2022-02-16'),(2022,3,'2022-03-15'),(2022,4,'2022-04-20'),(2022,5,'2022-05-18'),(2022,6,'2022-06-15'),(2022,7,'2022-07-20'),(2022,8,'2022-08-17'),(2022,9,'2022-09-21'),(2022,10,'2022-10-19'),(2022,11,'2022-11-16'),(2022,12,'2022-12-21'),(2023,1,'2023-01-18'),(2023,2,'2023-02-15'),(2023,3,'2023-03-22'),(2023,4,'2023-04-19'),(2023,5,'2023-05-17'),(2023,6,'2023-06-21'),(2023,7,'2023-07-19'),(2023,8,'2023-08-16'),(2023,9,'2023-09-20'),(2023,10,'2023-10-18'),(2023,11,'2023-11-15'),(2023,12,'2023-12-20'),(2024,1,'2024-01-17'),(2024,2,'2024-02-14'),(2024,3,'2024-03-20'),(2024,4,'2024-04-17'),(2024,5,'2024-05-22'),(2024,6,'2024-06-18'),(2024,7,'2024-07-17'),(2024,8,'2024-08-21'),(2024,9,'2024-09-18'),(2024,10,'2024-10-16'),(2024,11,'2024-11-20'),(2024,12,'2024-12-18'),(2025,1,'2025-01-22'),(2025,2,'2025-02-19'),(2025,3,'2025-03-18'),(2025,4,'2025-04-16'),(2025,5,'2025-05-21'),(2025,6,'2025-06-18'),(2025,7,'2025-07-16'),(2025,8,'2025-08-20'),(2025,9,'2025-09-17'),(2025,10,'2025-10-22'),(2025,11,'2025-11-19'),(2025,12,'2025-12-17'),(2026,1,'2026-01-21'),(2026,2,'2026-02-18'),(2026,3,'2026-03-18'),(2026,4,'2026-04-15'),(2026,5,'2026-05-19'),(2026,6,'2026-06-17'),(2026,7,'2026-07-22'),(2026,8,'2026-08-19')]
    MONTH_CODES={1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}
    def __init__(self,cboe_vx_dir="./CBOE_VX_ALL",cache_dir="./data_cache/vx_continuous"):
        self.vx_dir=Path(cboe_vx_dir);self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        if not self.vx_dir.exists():raise ValueError(f"VX data directory not found: {self.vx_dir}")
        self.expiry_lookup=self._build_expiry_lookup()
    def _build_expiry_lookup(self):
        records=[{'contract_code':f"{self.MONTH_CODES[month]}{str(year)[-1]}",'year':year,'month':month,'expiry_date':pd.Timestamp(exp_date),'filename':f"VX_{self.MONTH_CODES[month]}{str(year)[-1]}_{exp_date}.csv"}for year,month,exp_date in self.EXPIRATION_DATES]
        df=pd.DataFrame(records);df=df.sort_values('expiry_date').reset_index(drop=True);return df
    def _get_last_date_from_cache(self,contract_position):
        cache_file=self.cache_dir/f"VX{contract_position}.parquet"
        if not cache_file.exists():return None
        try:
            df=pd.read_parquet(cache_file)
            if not df.empty:return df.index[-1]
        except:pass
        return None
    def _check_raw_contract_staleness(self,date):
        active=self.expiry_lookup[self.expiry_lookup['expiry_date']>date].head(6);missing=[]
        for _,row in active.iterrows():
            filepath=self.vx_dir/row['filename']
            if not filepath.exists():missing.append(row['filename']);continue
            try:
                df=pd.read_csv(filepath,parse_dates=['Trade Date'],index_col='Trade Date')
                if date not in df.index:missing.append(row['filename'])
            except:missing.append(row['filename'])
        return missing
    def _fetch_missing_cboe_data(self,date):
        missing=self._check_raw_contract_staleness(date)
        if not missing:return 0
        print(f"   Fetching {len(missing)} contracts for {date.date()}...");updated=0
        for filename in missing:
            parts=filename.replace('.csv','').split('_');exp_date=parts[2]
            url=f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{exp_date}.csv";filepath=self.vx_dir/filename
            try:
                response=requests.get(url,timeout=15);response.raise_for_status()
                if len(response.text)>100:filepath.write_text(response.text);updated+=1;time.sleep(0.3)
            except Exception as e:warnings.warn(f"Failed to fetch {filename}: {str(e)[:50]}")
        return updated
    def _load_raw_contract(self,filename):
        filepath=self.vx_dir/filename
        if not filepath.exists():return pd.DataFrame()
        try:
            df=pd.read_csv(filepath,parse_dates=['Trade Date'],index_col='Trade Date')
            df=df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Settle':'settle','Volume':'volume','Open Interest':'open_interest'})
            df['close']=df['settle']
            for col in ['open','high','low','close','settle','volume','open_interest']:
                if col in df.columns:df[col]=pd.to_numeric(df[col],errors='coerce')
            return df
        except Exception as e:warnings.warn(f"Failed to load {filename}: {str(e)}");return pd.DataFrame()
    def _build_continuous_contract_incremental(self,position,start_date=None):
        cache_file=self.cache_dir/f"VX{position}.parquet"
        if cache_file.exists()and start_date is not None:
            try:
                cached=pd.read_parquet(cache_file)
                if not cached.empty:
                    last_cached_date=cached.index[-1]
                    if start_date<=last_cached_date:return cached
                    start_date=last_cached_date+timedelta(days=1)
            except:cached=pd.DataFrame()
        else:
            cached=pd.DataFrame()
            if start_date is None:start_date=pd.Timestamp('2013-03-25')
        end_date=pd.Timestamp.now();date_range=pd.bdate_range(start_date,end_date)
        if len(date_range)==0:return cached
        new_data=[]
        for date in date_range:
            active=self.expiry_lookup[self.expiry_lookup['expiry_date']>date]
            if len(active)<position:continue
            contract=active.iloc[position-1];df=self._load_raw_contract(contract['filename'])
            if df.empty or date not in df.index:continue
            row=df.loc[date]
            new_data.append({'date':date,'open':row['open'],'high':row['high'],'low':row['low'],'close':row['close'],'settle':row['settle'],'volume':row['volume'],'open_interest':row['open_interest'],'contract_code':contract['contract_code'],'expiry_date':contract['expiry_date']})
        if not new_data:return cached
        new_df=pd.DataFrame(new_data).set_index('date');new_df['roll_event']=new_df['contract_code']!=new_df['contract_code'].shift(1);adjusted=self._apply_backwards_ratio_adjustment(new_df)
        if not cached.empty:
            combined=pd.concat([cached,adjusted]);combined=combined[~combined.index.duplicated(keep='last')];combined=combined.sort_index();combined=self._apply_backwards_ratio_adjustment(combined)
        else:combined=adjusted
        return combined
    def _apply_backwards_ratio_adjustment(self,df):
        if 'roll_event'not in df.columns:return df
        adjusted=df.copy();roll_dates=adjusted[adjusted['roll_event']].index
        if len(roll_dates)==0:return adjusted.drop(columns=['roll_event','contract_code','expiry_date'],errors='ignore')
        cumulative_ratio=1.0
        for i in range(len(roll_dates)-1,-1,-1):
            roll_date=roll_dates[i];new_settle=adjusted.loc[roll_date,'settle'];prev_dates=adjusted.index[adjusted.index<roll_date]
            if len(prev_dates)==0:continue
            prev_date=prev_dates[-1];old_settle=adjusted.loc[prev_date,'settle']
            if old_settle>0 and new_settle>0:
                ratio=new_settle/old_settle;cumulative_ratio*=ratio;mask=adjusted.index<roll_date
                for col in ['open','high','low','close','settle']:
                    if col in adjusted.columns:adjusted.loc[mask,col]=adjusted.loc[mask,col]*ratio
        adjusted=adjusted.drop(columns=['roll_event','contract_code','expiry_date'],errors='ignore');return adjusted
    def get_continuous_contract(self,position,start_date=None,end_date=None,force_rebuild=False):
        cache_file=self.cache_dir/f"VX{position}.parquet";last_cached=self._get_last_date_from_cache(position);today=pd.Timestamp.now().normalize();needs_update=False
        if force_rebuild:needs_update=True;print(f"   Force rebuilding VX{position}...")
        elif last_cached is None:needs_update=True;print(f"   Building VX{position} (no cache)...")
        else:
            business_days_behind=len(pd.bdate_range(last_cached,today))-1
            if business_days_behind>0:
                needs_update=True;print(f"   Updating VX{position} ({business_days_behind} days stale)...")
                for i in range(business_days_behind):
                    check_date=last_cached+timedelta(days=i+1)
                    if check_date.weekday()<5:self._fetch_missing_cboe_data(check_date)
        if needs_update:
            df=self._build_continuous_contract_incremental(position,None if force_rebuild else last_cached)
            if not df.empty:df.to_parquet(cache_file)
        else:df=pd.read_parquet(cache_file)
        if start_date:df=df[df.index>=pd.Timestamp(start_date)]
        if end_date:df=df[df.index<=pd.Timestamp(end_date)]
        return df
    def get_all_continuous_contracts(self,start_date=None,end_date=None,force_rebuild=False):
        contracts={}
        for position in range(1,7):
            df=self.get_continuous_contract(position,start_date,end_date,force_rebuild)
            if not df.empty:contracts[f'VX{position}']=df
        return contracts
    def check_system_health(self):
        health={'raw_contracts':0,'cached_contracts':0,'staleness_days':{},'missing_dates':[],'issues':[]}
        raw_files=list(self.vx_dir.glob('VX_*.csv'));health['raw_contracts']=len(raw_files)
        for position in range(1,7):
            cache_file=self.cache_dir/f"VX{position}.parquet"
            if cache_file.exists():
                health['cached_contracts']+=1;last_date=self._get_last_date_from_cache(position)
                if last_date:
                    days_stale=(pd.Timestamp.now().normalize()-last_date).days;health['staleness_days'][f'VX{position}']=days_stale
                    if days_stale>5:health['issues'].append(f"VX{position} is {days_stale} days stale")
        return health
def get_vx_continuous_contracts(start_date=None,end_date=None,cboe_vx_dir="./CBOE_VX_ALL",cache_dir="./data_cache/vx_continuous"):
    builder=VXContinuousContractBuilder(cboe_vx_dir,cache_dir);return builder.get_all_continuous_contracts(start_date,end_date)
