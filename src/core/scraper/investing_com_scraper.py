import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path
import requests
from io import StringIO
warnings.filterwarnings('ignore')
def scrape_investing_com(url,after_date=None):
    try:
        headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
        response=requests.get(url,headers=headers,timeout=10);response.raise_for_status()
        tables=pd.read_html(StringIO(response.text))
        if not tables:return None
        df=tables[0];df.columns=['Date','Close','Open','High','Low','Vol','Change'];df=df[['Date','Open','High','Low','Close']];df['Date']=pd.to_datetime(df['Date'])
        for col in['Open','High','Low','Close']:df[col]=df[col].astype(str).str.replace(',','').astype(float)
        df.set_index('Date',inplace=True);df=df.sort_index()
        if after_date:df=df[df.index>after_date]
        return df if len(df)>0 else None
    except Exception:return None
def get_last_date_from_csv(csv_path):
    try:
        with open(csv_path,'rb')as f:
            f.seek(0,2);file_size=f.tell();read_size=min(500,file_size);f.seek(file_size-read_size);last_chunk=f.read().decode('utf-8',errors='ignore')
            lines=last_chunk.strip().split('\n')
            for line in reversed(lines):
                line=line.strip()
                if line and','in line and not line.startswith('Date'):return pd.to_datetime(line.split(',')[0])
        return None
    except Exception:return None
def check_staleness(last_date):
    if last_date is None:return True
    now=datetime.now();bdays=pd.bdate_range(last_date,now);days_stale=len(bdays)-1
    if days_stale<=1 and now.hour<16:return False
    return days_stale>0
def clean_csv_trailing_lines(csv_path):
    try:
        with open(csv_path,'r')as f:lines=f.readlines()
        while lines and not lines[-1].strip():lines.pop()
        if lines and not lines[-1].endswith('\n'):lines[-1]+='\n'
        with open(csv_path,'w')as f:f.writelines(lines)
        return True
    except Exception:return False
def update_ticker(csv_path,url,force=False,cboe_data_dir=None):
    csv_path=Path(csv_path)
    if not csv_path.is_absolute():
        base_dir=Path(cboe_data_dir)if cboe_data_dir else(Path("./CBOE_Data_Archive")if Path("./CBOE_Data_Archive").exists()else Path(__file__).parent.parent.parent/"CBOE_Data_Archive")
        csv_path=base_dir/csv_path.name
    if not csv_path.exists():return False
    last_date=get_last_date_from_csv(csv_path)
    if last_date is None:return False
    if not force and not check_staleness(last_date):return True
    new_data=scrape_investing_com(url,after_date=last_date)
    if new_data is None or len(new_data)==0:return False
    clean_csv_trailing_lines(csv_path)
    try:
        with open(csv_path,'a',newline='')as f:
            for date,row in new_data.iterrows():f.write(f"{date.strftime('%Y-%m-%d')},{row['Open']},{row['High']},{row['Low']},{row['Close']}\n")
        return True
    except Exception:return False
def update_vxtlt(csv_path="VXTLT.csv",force=False,cboe_data_dir=None):
    return update_ticker(csv_path,"https://www.investing.com/indices/tlt-vix-historical-data",force,cboe_data_dir)
def update_vxth(csv_path="VXTH_TAILHEDGE_CBOE.csv",force=False,cboe_data_dir=None):
    return update_ticker(csv_path,"https://www.investing.com/indices/cboe-vix-tail-hedge-historical-data",force,cboe_data_dir)
def update_all(cboe_data_dir=None,force=False):
    results={};results['VXTLT']=update_vxtlt(force=force,cboe_data_dir=cboe_data_dir);results['VXTH']=update_vxth(force=force,cboe_data_dir=cboe_data_dir);return results
if __name__=="__main__":
    print("VIX Data Updater");print("="*40);print("[VXTLT] ",end='');success=update_vxtlt();print("✓"if success else"✗");print("[VXTH]  ",end='');success=update_vxth();print("✓"if success else"✗");print("="*40)
