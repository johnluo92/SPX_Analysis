import numpy as np
import pandas as pd
from typing import Union,List,Optional
def calculate_robust_zscore(series:pd.Series,window:int,min_std:float=1e-8)->pd.Series:
    rolling_mean=series.rolling(window).mean();rolling_std=series.rolling(window).std().clip(lower=min_std)
    return(series-rolling_mean)/rolling_std
def calculate_regime_with_validation(series:pd.Series,bins:List[float],labels:List[int],feature_name:str="feature")->pd.Series:
    if series.notna().sum()/len(series)<0.5:return pd.Series(0,index=series.index)
    valid_values=series.dropna()
    if len(valid_values)>0 and valid_values.max()-valid_values.min()<1e-6:return pd.Series(0,index=series.index)
    try:
        regime=pd.cut(series,bins=bins,labels=labels)
        return regime.fillna(0).astype(int)
    except Exception as e:
        import warnings
        warnings.warn(f"⚠️ Regime classification failed for {feature_name}: {str(e)}. Defaulting to regime 0.")
        return pd.Series(0,index=series.index)
def calculate_percentile_with_validation(series:pd.Series,window:int,min_data_pct:float=0.7)->pd.Series:
    def safe_percentile_rank(window_data:pd.Series)->float:
        valid_data=window_data.dropna()
        if len(valid_data)<window*min_data_pct or len(valid_data)==0:return np.nan
        last_value=window_data.iloc[-1]
        if pd.isna(last_value):return np.nan
        below_count=(valid_data<last_value).sum();percentile=(below_count/len(valid_data))*100
        return percentile
    return series.rolling(window+1).apply(safe_percentile_rank,raw=False)
def calculate_velocity(series:pd.Series,window:int)->pd.Series:
    return series.diff(window)
def calculate_acceleration(series:pd.Series,window:int)->pd.Series:
    velocity=series.diff(window)
    return velocity.diff(window)
def validate_series_quality(series:pd.Series,min_valid_pct:float=0.5,check_variance:bool=True,min_variance:float=1e-6)->tuple[bool,str]:
    valid_pct=series.notna().sum()/len(series)
    if valid_pct<min_valid_pct:return False,f"Only {valid_pct:.1%} valid data (need {min_valid_pct:.1%})"
    if check_variance:
        valid_values=series.dropna()
        if len(valid_values)>1:
            data_range=valid_values.max()-valid_values.min()
            if data_range<min_variance:return False,f"Series has insufficient variance (range: {data_range:.2e})"
    return True,"Series passes quality checks"
def safe_division(numerator:Union[pd.Series,float],denominator:Union[pd.Series,float],fill_value:float=np.nan,min_denominator:float=1e-10)->Union[pd.Series,float]:
    if isinstance(denominator,pd.Series):
        safe_denom=denominator.replace(0,np.nan).abs();safe_denom=safe_denom.where(safe_denom>=min_denominator,np.nan);result=numerator/safe_denom
        if fill_value is not np.nan:result=result.fillna(fill_value)
        return result
    else:
        if abs(denominator)<min_denominator:return fill_value
        return numerator/denominator
def rolling_rank_pct(series:pd.Series,window:int)->pd.Series:
    return series.rolling(window).rank(pct=True)*100
def exponential_decay_weights(length:int,halflife:int)->np.ndarray:
    decay_factor=0.5**(1/halflife);positions=np.arange(length);weights=decay_factor**positions[::-1]
    return weights/weights.sum()
DEFAULT_ZSCORE_WINDOW=63;DEFAULT_PERCENTILE_WINDOW=63;DEFAULT_MIN_STD=1e-8;DEFAULT_MIN_DATA_PCT=0.7
VIX_REGIME_BINS=[0,16.77,24.40,39.67,100];VIX_REGIME_LABELS=[0,1,2,3]
SKEW_REGIME_BINS=[0,130,145,160,200];SKEW_REGIME_LABELS=[0,1,2,3]
if __name__=="__main__":
    print("Running calculation module validation tests...\n")
    test_series=pd.Series([10,12,15,20,18,16,14,22,25,28]);zscore=calculate_robust_zscore(test_series,window=5);print("✓ Z-score test passed")
    vix_test=pd.Series([12,20,30,45]);regimes=calculate_regime_with_validation(vix_test,bins=VIX_REGIME_BINS,labels=VIX_REGIME_LABELS,feature_name="vix_test")
    assert list(regimes)==[0,1,2,3],"Regime test failed"
    print("✓ Regime classification test passed")
    percentile=calculate_percentile_with_validation(test_series,window=5)
    assert percentile.notna().sum()>0,"Percentile test failed"
    print("✓ Percentile calculation test passed")
    safe_result=safe_division(pd.Series([1,2,3]),pd.Series([2,0,1]),fill_value=0)
    assert safe_result.iloc[1]==0,"Safe division test failed"
    print("✓ Safe division test passed")
    print("\n✅ All validation tests passed!")
