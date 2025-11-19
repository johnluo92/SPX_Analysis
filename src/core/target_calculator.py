import numpy as np
import pandas as pd
from typing import Optional,Dict
from config import TARGET_CONFIG
class TargetCalculator:
    def __init__(self):
        self.horizon=TARGET_CONFIG["horizon_days"];self.target_type=TARGET_CONFIG["target_type"]
    def calculate_log_vix_change(self,vix_series,dates=None,method="shift"):
        if dates is None:dates=vix_series.index
        if method=="shift":
            vix_aligned=vix_series.reindex(dates);future_vix=vix_aligned.shift(-self.horizon)
            target=np.log(future_vix)-np.log(vix_aligned)
        elif method=="manual":
            target=pd.Series(index=dates,dtype=float);vix_series=vix_series.sort_index()
            for date in dates:
                if date not in vix_series.index:continue
                try:
                    date_pos=vix_series.index.get_loc(date)
                    if not isinstance(date_pos,int):continue
                except KeyError:continue
                end_pos=date_pos+self.horizon
                if end_pos>=len(vix_series):continue
                current_vix=vix_series.iloc[date_pos];future_vix=vix_series.iloc[end_pos]
                if pd.isna(current_vix)or pd.isna(future_vix):continue
                if current_vix<=0 or future_vix<=0:continue
                target[date]=np.log(future_vix/current_vix)
        else:raise ValueError(f"Unknown method: {method}")
        valid_mask=(~target.isna()&np.isfinite(target))
        return target[valid_mask]
    def calculate_direction_from_log_change(self,log_change):return(log_change>0).astype(int)
    def calculate_pct_change_from_log(self,log_change):return(np.exp(log_change)-1)*100
    def calculate_all_targets(self,df,vix_col="vix"):
        if vix_col not in df.columns:raise ValueError(f"Column '{vix_col}' not found in DataFrame")
        df=df.copy()
        df["target_log_vix_change"]=self.calculate_log_vix_change(df[vix_col],method="shift")
        df["target_direction"]=self.calculate_direction_from_log_change(df["target_log_vix_change"])
        df["target_vix_pct_change"]=self.calculate_pct_change_from_log(df["target_log_vix_change"])
        df["future_vix"]=df[vix_col].shift(-self.horizon)
        return df
    def get_target_stats(self,target_series):
        return{"count":int(len(target_series)),"mean":float(target_series.mean()),"std":float(target_series.std()),"min":float(target_series.min()),"max":float(target_series.max()),"median":float(target_series.median()),"q25":float(target_series.quantile(0.25)),"q75":float(target_series.quantile(0.75))}
    def validate_targets(self,df):
        results={"valid":True,"warnings":[],"stats":{}}
        if "target_log_vix_change"not in df.columns:results["valid"]=False;results["warnings"].append("target_log_vix_change column missing");return results
        target=df["target_log_vix_change"];valid_targets=target.dropna()
        if len(valid_targets)<100:results["warnings"].append(f"Only {len(valid_targets)} valid targets (expected >100)")
        extreme_low=(valid_targets<-2.0).sum();extreme_high=(valid_targets>2.0).sum()
        if extreme_low>0:results["warnings"].append(f"{extreme_low} targets with extreme negative values (< -2.0)")
        if extreme_high>0:results["warnings"].append(f"{extreme_high} targets with extreme positive values (> 2.0)")
        results["stats"]=self.get_target_stats(valid_targets)
        if "target_direction"in df.columns:
            up_pct=df["target_direction"].mean()
            if up_pct<0.3 or up_pct>0.7:results["warnings"].append(f"Imbalanced direction: {up_pct:.1%} up, {1-up_pct:.1%} down")
        return results
_calculator=TargetCalculator()
def calculate_targets(df,vix_col="vix"):return _calculator.calculate_all_targets(df,vix_col)
def get_horizon():return TARGET_CONFIG["horizon_days"]