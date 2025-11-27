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
    EXPIRATION_DATES = [(2013, 1, '2013-01-16'), (2013, 2, '2013-02-13'), (2013, 3, '2013-03-20'), (2013, 4, '2013-04-17'), (2013, 5, '2013-05-22'), (2013, 6, '2013-06-19'), (2013, 7, '2013-07-17'), (2013, 8, '2013-08-21'), (2013, 9, '2013-09-18'), (2013, 10, '2013-10-16'), (2013, 11, '2013-11-20'), (2013, 12, '2013-12-18'), (2014, 1, '2014-01-22'), (2014, 2, '2014-02-19'), (2014, 3, '2014-03-18'), (2014, 4, '2014-04-16'), (2014, 5, '2014-05-21'), (2014, 6, '2014-06-18'), (2014, 7, '2014-07-16'), (2014, 8, '2014-08-20'), (2014, 9, '2014-09-17'), (2014, 10, '2014-10-22'), (2014, 11, '2014-11-19'), (2014, 12, '2014-12-17'), (2015, 1, '2015-01-21'), (2015, 2, '2015-02-18'), (2015, 3, '2015-03-18'), (2015, 4, '2015-04-15'), (2015, 5, '2015-05-20'), (2015, 6, '2015-06-17'), (2015, 7, '2015-07-22'), (2015, 8, '2015-08-19'), (2015, 9, '2015-09-16'), (2015, 10, '2015-10-21'), (2015, 11, '2015-11-18'), (2015, 12, '2015-12-16'), (2016, 1, '2016-01-20'), (2016, 2, '2016-02-17'), (2016, 3, '2016-03-16'), (2016, 4, '2016-04-20'), (2016, 5, '2016-05-18'), (2016, 6, '2016-06-15'), (2016, 7, '2016-07-20'), (2016, 8, '2016-08-17'), (2016, 9, '2016-09-21'), (2016, 10, '2016-10-19'), (2016, 11, '2016-11-16'), (2016, 12, '2016-12-21'), (2017, 1, '2017-01-18'), (2017, 2, '2017-02-15'), (2017, 3, '2017-03-22'), (2017, 4, '2017-04-19'), (2017, 5, '2017-05-17'), (2017, 6, '2017-06-21'), (2017, 7, '2017-07-19'), (2017, 8, '2017-08-16'), (2017, 9, '2017-09-20'), (2017, 10, '2017-10-18'), (2017, 11, '2017-11-15'), (2017, 12, '2017-12-20'), (2018, 1, '2018-01-17'), (2018, 2, '2018-02-14'), (2018, 3, '2018-03-21'), (2018, 4, '2018-04-18'), (2018, 5, '2018-05-16'), (2018, 6, '2018-06-20'), (2018, 7, '2018-07-18'), (2018, 8, '2018-08-22'), (2018, 9, '2018-09-19'), (2018, 10, '2018-10-17'), (2018, 11, '2018-11-21'), (2018, 12, '2018-12-19'), (2019, 1, '2019-01-16'), (2019, 2, '2019-02-13'), (2019, 3, '2019-03-19'), (2019, 4, '2019-04-17'), (2019, 5, '2019-05-22'), (2019, 6, '2019-06-19'), (2019, 7, '2019-07-17'), (2019, 8, '2019-08-21'), (2019, 9, '2019-09-18'), (2019, 10, '2019-10-16'), (2019, 11, '2019-11-20'), (2019, 12, '2019-12-18'), (2020, 1, '2020-01-22'), (2020, 2, '2020-02-19'), (2020, 3, '2020-03-18'), (2020, 4, '2020-04-15'), (2020, 5, '2020-05-20'), (2020, 6, '2020-06-17'), (2020, 7, '2020-07-22'), (2020, 8, '2020-08-19'), (2020, 9, '2020-09-16'), (2020, 10, '2020-10-21'), (2020, 11, '2020-11-18'), (2020, 12, '2020-12-16'), (2021, 1, '2021-01-20'), (2021, 2, '2021-02-17'), (2021, 3, '2021-03-17'), (2021, 4, '2021-04-21'), (2021, 5, '2021-05-19'), (2021, 6, '2021-06-16'), (2021, 7, '2021-07-21'), (2021, 8, '2021-08-18'), (2021, 9, '2021-09-15'), (2021, 10, '2021-10-20'), (2021, 11, '2021-11-17'), (2021, 12, '2021-12-22'), (2022, 1, '2022-01-19'), (2022, 2, '2022-02-16'), (2022, 3, '2022-03-15'), (2022, 4, '2022-04-20'), (2022, 5, '2022-05-18'), (2022, 6, '2022-06-15'), (2022, 7, '2022-07-20'), (2022, 8, '2022-08-17'), (2022, 9, '2022-09-21'), (2022, 10, '2022-10-19'), (2022, 11, '2022-11-16'), (2022, 12, '2022-12-21'), (2023, 1, '2023-01-18'), (2023, 2, '2023-02-15'), (2023, 3, '2023-03-22'), (2023, 4, '2023-04-19'), (2023, 5, '2023-05-17'), (2023, 6, '2023-06-21'), (2023, 7, '2023-07-19'), (2023, 8, '2023-08-16'), (2023, 9, '2023-09-20'), (2023, 10, '2023-10-18'), (2023, 11, '2023-11-15'), (2023, 12, '2023-12-20'), (2024, 1, '2024-01-17'), (2024, 2, '2024-02-14'), (2024, 3, '2024-03-20'), (2024, 4, '2024-04-17'), (2024, 5, '2024-05-22'), (2024, 6, '2024-06-18'), (2024, 7, '2024-07-17'), (2024, 8, '2024-08-21'), (2024, 9, '2024-09-18'), (2024, 10, '2024-10-16'), (2024, 11, '2024-11-20'), (2024, 12, '2024-12-18'), (2025, 1, '2025-01-22'), (2025, 2, '2025-02-19'), (2025, 3, '2025-03-18'), (2025, 4, '2025-04-16'), (2025, 5, '2025-05-21'), (2025, 6, '2025-06-18'), (2025, 7, '2025-07-16'), (2025, 8, '2025-08-20'), (2025, 9, '2025-09-17'), (2025, 10, '2025-10-22'), (2025, 11, '2025-11-19'), (2025, 12, '2025-12-17'), (2026, 1, '2026-01-21'), (2026, 2, '2026-02-18'), (2026, 3, '2026-03-18'), (2026, 4, '2026-04-15'), (2026, 5, '2026-05-19'), (2026, 6, '2026-06-17'), (2026, 7, '2026-07-22'), (2026, 8, '2026-08-19')]
    MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    CORRUPTED_CONTRACTS=['VX_F3_2013-01-16.csv','VX_G3_2013-02-13.csv','VX_H3_2013-03-20.csv','VX_J3_2013-04-17.csv']
    def __init__(self,cboe_vx_dir="./CBOE_VX_ALL",cache_dir="./data_cache/vx_continuous"):
        self.vx_dir=Path(cboe_vx_dir);self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        if not self.vx_dir.exists():raise ValueError(f"VX dir not found: {self.vx_dir}")
        self.expiry_lookup=self._build_expiry_lookup()
    def _build_expiry_lookup(self):
        records=[{'contract_code':f"{self.MONTH_CODES[m]}{str(y)[-1]}",'year':y,'month':m,'expiry_date':pd.Timestamp(e),'filename':f"VX_{self.MONTH_CODES[m]}{str(y)[-1]}_{e}.csv"}for y,m,e in self.EXPIRATION_DATES]
        df=pd.DataFrame(records)
        return df.sort_values('expiry_date').reset_index(drop=True)
    def _get_cache_metadata(self,pos):
        cache_file=self.cache_dir/f"VX{pos}.parquet"
        if not cache_file.exists():return None,None
        try:
            df=pd.read_parquet(cache_file)
            return(df.index[-1],cache_file.stat().st_mtime)if not df.empty else(None,None)
        except:return None,None
    def _needs_update(self,pos,target_date=None):
        last_cached,_=self._get_cache_metadata(pos)
        if last_cached is None:return True,'no cache'
        target=target_date or pd.Timestamp.now().normalize();bdays=len(pd.bdate_range(last_cached,target))-1
        return(True,f'{bdays}d stale')if bdays>0 else(False,None)
    def _fetch_missing_cboe_data(self,date):
        active=self.expiry_lookup[self.expiry_lookup['expiry_date']>date].head(6);missing=[]
        for _,row in active.iterrows():
            fp=self.vx_dir/row['filename']
            if not fp.exists():missing.append(row['filename']);continue
            try:
                df=pd.read_csv(fp,parse_dates=['Trade Date'],index_col='Trade Date')
                if date not in df.index:missing.append(row['filename'])
            except:missing.append(row['filename'])
        if not missing:return 0
        for fn in missing:
            parts=fn.replace('.csv','').split('_');exp_date=parts[2];url=f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{exp_date}.csv";fp=self.vx_dir/fn
            try:
                r=requests.get(url,timeout=15);r.raise_for_status()
                if len(r.text)>100:fp.write_text(r.text);time.sleep(0.3)
            except:pass
        return len(missing)
    def _load_raw_contract(self,fn):
        fp=self.vx_dir/fn
        if not fp.exists():return pd.DataFrame()
        if fn in self.CORRUPTED_CONTRACTS:return pd.DataFrame()
        try:
            df=pd.read_csv(fp,parse_dates=['Trade Date'],index_col='Trade Date')
            df=df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Settle':'settle','Total Volume':'volume','Open Interest':'open_interest'})
            for c in['open','high','low','close','settle','volume','open_interest']:
                if c in df.columns:df[c]=pd.to_numeric(df[c],errors='coerce')
            if df.empty:return pd.DataFrame()
            if 'settle'not in df.columns:return pd.DataFrame()
            valid_settle=(df['settle']>0)&df['settle'].notna()
            if valid_settle.sum()==0:return pd.DataFrame()
            df=df[valid_settle].copy()
            if 'close'in df.columns:zero_close=(df['close']<=0)|df['close'].isna();df.loc[zero_close,'close']=df.loc[zero_close,'settle']
            if 'open'in df.columns:zero_open=(df['open']<=0)|df['open'].isna();df.loc[zero_open,'open']=df.loc[zero_open,'settle']
            if 'high'in df.columns and 'low'in df.columns:
                invalid_hl=(df['high']<=0)|(df['low']<=0)|df['high'].isna()|df['low'].isna();df.loc[invalid_hl,'high']=df.loc[invalid_hl,'settle'];df.loc[invalid_hl,'low']=df.loc[invalid_hl,'settle']
                inverted=df['high']<df['low'];df.loc[inverted,['high','low']]=df.loc[inverted,['low','high']].values
                settle_above_high=df['settle']>df['high'];df.loc[settle_above_high,'high']=df.loc[settle_above_high,'settle']
                settle_below_low=df['settle']<df['low'];df.loc[settle_below_low,'low']=df.loc[settle_below_low,'settle']
            df['close']=df['settle']
            for c in['volume','open_interest']:
                if c in df.columns:neg_or_nan=(df[c]<0)|df[c].isna();df.loc[neg_or_nan,c]=0
            return df
        except Exception:return pd.DataFrame()
    def _build_continuous_incremental(self,pos,start_date=None):
        cache_file=self.cache_dir/f"VX{pos}.parquet";cached=pd.DataFrame()
        if cache_file.exists()and start_date:
            try:
                cached=pd.read_parquet(cache_file)
                if not cached.empty:
                    last=cached.index[-1]
                    if start_date<=last:return cached
                    start_date=last+timedelta(days=1)
            except:pass
        if cached.empty:start_date=start_date or pd.Timestamp('2013-05-01')
        end=pd.Timestamp.now();dates=pd.bdate_range(start_date,end)
        if len(dates)==0:return cached
        new_data=[]
        for d in dates:
            active=self.expiry_lookup[self.expiry_lookup['expiry_date']>d]
            if len(active)<pos:continue
            contract=active.iloc[pos-1];df=self._load_raw_contract(contract['filename'])
            if df.empty or d not in df.index:continue
            row=df.loc[d]
            if pd.isna(row['settle'])or row['settle']<=0:continue
            dte=(contract['expiry_date']-d).days
            new_data.append({'date':d,'open':row['open'],'high':row['high'],'low':row['low'],'close':row['settle'],'settle':row['settle'],'volume':row['volume'],'open_interest':row['open_interest'],'contract_code':contract['contract_code'],'expiry_date':contract['expiry_date'],'days_to_expiry':dte})
        if not new_data:return cached
        new_df=pd.DataFrame(new_data).set_index('date');new_df['roll_event']=(new_df['contract_code']!=new_df['contract_code'].shift(1)).astype(int)
        combined=pd.concat([cached,new_df])if not cached.empty else new_df;combined=combined[~combined.index.duplicated(keep='last')].sort_index()
        return self._apply_panama_adjustment(combined)
    def _apply_panama_adjustment(self,df):
        if 'roll_event'not in df.columns:return df
        work=df.copy();rolls=work[work['roll_event']==1].index
        if len(rolls)==0:return work.drop(columns=['roll_event','contract_code','expiry_date','days_to_expiry'],errors='ignore')
        price_cols=['open','high','low','close','settle']
        for i,roll_date in enumerate(rolls):
            if i==0:continue
            prev_date=work.index[work.index<roll_date][-1];old_settle=work.loc[prev_date,'settle'];new_settle_raw=work.loc[roll_date,'settle']
            if old_settle<=0 or new_settle_raw<=0:continue
            level_adj=old_settle/new_settle_raw;mask=work.index>=roll_date
            for c in price_cols:
                if c in work.columns:work.loc[mask,c]*=level_adj
        work['roll_indicator']=work['roll_event']
        return work.drop(columns=['roll_event','contract_code','expiry_date'],errors='ignore')
    def get_continuous_contract(self,pos,start_date=None,end_date=None,force_rebuild=False):
        cache_file=self.cache_dir/f"VX{pos}.parquet";needs,reason=self._needs_update(pos,end_date)
        if force_rebuild or needs:
            last_cached,_=self._get_cache_metadata(pos)
            if last_cached and not force_rebuild:
                for i in range(len(pd.bdate_range(last_cached,pd.Timestamp.now()))):
                    check_date=last_cached+timedelta(days=i+1)
                    if check_date.weekday()<5:self._fetch_missing_cboe_data(check_date)
            df=self._build_continuous_incremental(pos,None if force_rebuild else last_cached)
            if not df.empty:df.to_parquet(cache_file,index=True,compression='snappy')
        else:df=pd.read_parquet(cache_file)
        if start_date:df=df[df.index>=pd.Timestamp(start_date)]
        if end_date:df=df[df.index<=pd.Timestamp(end_date)]
        return df
    def get_all_continuous_contracts(self,start_date=None,end_date=None,force_rebuild=False):
        print("   → Building VX futures from cache");contracts={}
        for pos in range(1,7):
            df=self.get_continuous_contract(pos,start_date,end_date,force_rebuild)
            if not df.empty:contracts[f'VX{pos}']=df
        print(f"   ✓ Loaded {len(contracts)} contracts")
        return contracts
def get_vx_continuous_contracts(start_date=None,end_date=None,cboe_vx_dir="./CBOE_VX_ALL",cache_dir="./data_cache/vx_continuous"):
    builder=VXContinuousContractBuilder(cboe_vx_dir,cache_dir)
    return builder.get_all_continuous_contracts(start_date,end_date)
