import warnings
import numpy as np
import pandas as pd
from core.vx_continuous_contract_builder import get_vx_continuous_contracts
warnings.filterwarnings("ignore")
class VXFuturesEngineer:
    def __init__(self,cboe_vx_dir="./CBOE_VX_ALL",cache_dir="./data_cache/vx_continuous"):
        self.cboe_vx_dir=cboe_vx_dir;self.cache_dir=cache_dir
    def build_all_vx_features(self,start_date,end_date,target_index=None):
        print("   → Building VX futures features from cached contracts...")
        vx_contracts=get_vx_continuous_contracts(start_date=start_date,end_date=end_date,cboe_vx_dir=self.cboe_vx_dir,cache_dir=self.cache_dir)
        if not vx_contracts:raise ValueError("Failed to load VX continuous contracts")
        print(f"   ✓ Loaded {len(vx_contracts)} continuous contracts")
        term_features=self.extract_term_structure_features(vx_contracts)
        roll_features=self.extract_roll_yield_features(vx_contracts)
        oi_features=self.extract_open_interest_features(vx_contracts)
        volume_features=self.extract_volume_liquidity_features(vx_contracts)
        range_features=self.extract_intraday_range_features(vx_contracts)
        all_features=pd.concat([term_features,roll_features,oi_features,volume_features,range_features],axis=1)
        all_features=all_features.loc[:,~all_features.columns.duplicated()]
        if target_index is not None:all_features=all_features.reindex(target_index,method='ffill',limit=5)
        print(f"   ✓ Generated {len(all_features.columns)} VX features")
        return all_features
    def extract_term_structure_features(self,vx_contracts):
        indices=[df.index for df in vx_contracts.values()if not df.empty]
        if not indices:return pd.DataFrame()
        common_index=indices[0]
        for idx in indices[1:]:common_index=common_index.intersection(idx)
        features=pd.DataFrame(index=common_index)
        vx_prices={}
        for name,df in vx_contracts.items():
            if not df.empty and'settle'in df.columns:vx_prices[name]=df['settle'].reindex(common_index)
        if'VX1'in vx_prices and'VX2'in vx_prices:features['vx1_vx2_spread']=vx_prices['VX1']-vx_prices['VX2']
        if'VX1'in vx_prices and'VX3'in vx_prices:features['vx1_vx3_spread']=vx_prices['VX1']-vx_prices['VX3']
        if'VX1'in vx_prices and'VX6'in vx_prices:features['vx1_vx6_spread']=vx_prices['VX1']-vx_prices['VX6']
        if'VX1'in vx_prices and'VX6'in vx_prices:features['vx_curve_slope']=(vx_prices['VX6']-vx_prices['VX1'])/5
        if all(k in vx_prices for k in['VX1','VX2','VX3']):
            features['vx_curve_curvature']=vx_prices['VX1']-2*vx_prices['VX2']+vx_prices['VX3']
            spread_12=vx_prices['VX1']-vx_prices['VX2'];spread_23=vx_prices['VX2']-vx_prices['VX3']
            features['vx_curve_skew']=spread_12-spread_23
        if'vx1_vx2_spread'in features:
            features['vx_curve_acceleration_5d']=features['vx1_vx2_spread'].diff(5)
            features['vx_spread_zscore_63d']=self._calculate_zscore(features['vx1_vx2_spread'],63)
            for window in[21,63,126]:features[f'vx_spread_percentile_{window}d']=self._calculate_percentile(features['vx1_vx2_spread'],window)
        if'vx1_vx2_spread'in features:
            features['vx_contango_flag']=(features['vx1_vx2_spread']<0).astype(int)
            features['vx_backwardation_flag']=(features['vx1_vx2_spread']>0).astype(int)
            if'VX1'in vx_prices and'VX2'in vx_prices:
                pct_diff=(vx_prices['VX2']-vx_prices['VX1'])/vx_prices['VX1'].replace(0,np.nan)*100
                features['vx_steep_contango']=(pct_diff>5).astype(int)
        return features
    def extract_roll_yield_features(self,vx_contracts):
        indices=[df.index for df in vx_contracts.values()if not df.empty]
        if not indices:return pd.DataFrame()
        common_index=indices[0]
        for idx in indices[1:]:common_index=common_index.intersection(idx)
        features=pd.DataFrame(index=common_index)
        vx_prices={}
        for name,df in vx_contracts.items():
            if not df.empty and'settle'in df.columns:vx_prices[name]=df['settle'].reindex(common_index)
        if'VX1'in vx_prices and'VX2'in vx_prices:
            vx1,vx2=vx_prices['VX1'],vx_prices['VX2']
            spread=vx1-vx2
            features['vx_roll_yield_daily']=spread/30
            features['vx_annualized_roll']=features['vx_roll_yield_daily']*252
            features['vx_contango_intensity']=((vx2-vx1)/vx1.replace(0,np.nan)*100).clip(lower=0)
            features['vx_backwardation_intensity']=((vx1-vx2)/vx2.replace(0,np.nan)*100).clip(lower=0)
            features['vx_carry_zscore_63d']=self._calculate_zscore(features['vx_contango_intensity'],63)
        return features
    def extract_open_interest_features(self,vx_contracts):
        indices=[df.index for df in vx_contracts.values()if not df.empty]
        if not indices:return pd.DataFrame()
        common_index=indices[0]
        for idx in indices[1:]:common_index=common_index.intersection(idx)
        features=pd.DataFrame(index=common_index)
        vx_oi={}
        for name,df in vx_contracts.items():
            if not df.empty and'open_interest'in df.columns:vx_oi[name]=df['open_interest'].reindex(common_index)
        for i in range(1,7):
            vx_name=f'VX{i}'
            if vx_name in vx_oi:features[f'vx{i}_open_interest']=vx_oi[vx_name]
        if vx_oi:
            features['vx_total_oi']=sum(vx_oi.values())
            if'VX1'in vx_oi:features['vx_front_oi_pct']=vx_oi['VX1']/features['vx_total_oi'].replace(0,np.nan)*100
        if'VX1'in vx_oi:
            vx1_oi=vx_oi['VX1']
            for window in[1,5,10]:features[f'vx1_oi_change_{window}d']=vx1_oi.diff(window)
            features['vx1_oi_zscore_63d']=self._calculate_zscore(vx1_oi,63)
        if'VX1'in vx_oi and'VX2'in vx_oi:
            features['vx_oi_ratio_1_2']=vx_oi['VX1']/vx_oi['VX2'].replace(0,np.nan)
            total_oi_12=vx_oi['VX1']+vx_oi['VX2']
            features['vx_oi_imbalance']=(vx_oi['VX1']-vx_oi['VX2'])/total_oi_12.replace(0,np.nan)
        return features
    def extract_volume_liquidity_features(self,vx_contracts):
        indices=[df.index for df in vx_contracts.values()if not df.empty]
        if not indices:return pd.DataFrame()
        common_index=indices[0]
        for idx in indices[1:]:common_index=common_index.intersection(idx)
        features=pd.DataFrame(index=common_index)
        vx_volume={};vx_oi={}
        for name,df in vx_contracts.items():
            if not df.empty:
                if'volume'in df.columns:vx_volume[name]=df['volume'].reindex(common_index)
                if'open_interest'in df.columns:vx_oi[name]=df['open_interest'].reindex(common_index)
        if'VX1'in vx_volume:
            vx1_vol=vx_volume['VX1']
            features['vx1_volume']=vx1_vol
            features['vx1_volume_ma21']=vx1_vol.rolling(21).mean()
            vol_mean=vx1_vol.rolling(63).mean();vol_std=vx1_vol.rolling(63).std()
            features['vx1_volume_surge']=(vx1_vol-vol_mean)/vol_std.replace(0,np.nan)
        if'VX1'in vx_volume and'VX1'in vx_oi:features['vx1_volume_oi_ratio']=vx_volume['VX1']/vx_oi['VX1'].replace(0,np.nan)
        return features
    def extract_intraday_range_features(self,vx_contracts):
        if'VX1'not in vx_contracts or vx_contracts['VX1'].empty:return pd.DataFrame()
        vx1=vx_contracts['VX1']
        features=pd.DataFrame(index=vx1.index)
        required_cols=['open','high','low','close']
        if not all(col in vx1.columns for col in required_cols):return features
        o,h,l,c=vx1['open'],vx1['high'],vx1['low'],vx1['close']
        features['vx1_daily_range']=h-l
        features['vx1_daily_range_pct']=(h-l)/c.replace(0,np.nan)*100
        prev_close=c.shift(1)
        tr=pd.concat([h-l,(h-prev_close).abs(),(l-prev_close).abs()],axis=1).max(axis=1)
        features['vx1_atr_14']=tr.rolling(14).mean()
        avg_range=features['vx1_daily_range'].rolling(21).mean()
        features['vx1_range_expansion']=features['vx1_daily_range']/avg_range.replace(0,np.nan)
        return features
    def _calculate_zscore(self,series,window):
        mean=series.rolling(window).mean();std=series.rolling(window).std()
        return(series-mean)/std.replace(0,np.nan)
    def _calculate_percentile(self,series,window):
        return series.rolling(window).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1]*100,raw=False)
def build_vx_futures_features(start_date,end_date,cboe_vx_dir="./CBOE_VX_ALL",target_index=None):
    engineer=VXFuturesEngineer(cboe_vx_dir)
    return engineer.build_all_vx_features(start_date,end_date,target_index)
