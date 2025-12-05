import warnings
import numpy as np
import pandas as pd
from core.vx_continuous_contract_builder import get_vx_continuous_contracts
warnings.filterwarnings("ignore")

class VXFuturesEngineer:
    def __init__(self,cboe_vx_dir="./CBOE_VX_ALL",cache_dir="./data_cache/vx_continuous"):
        self.cboe_vx_dir=cboe_vx_dir;self.cache_dir=cache_dir

    def build_all_vx_features(self,start_date,end_date,target_index=None,vix_series=None):
        print("   → Building VX futures features from cached contracts...")
        vx_contracts=get_vx_continuous_contracts(start_date=start_date,end_date=end_date,cboe_vx_dir=self.cboe_vx_dir,cache_dir=self.cache_dir)
        if not vx_contracts:raise ValueError("Failed to load VX continuous contracts")
        print(f"   ✓ Loaded {len(vx_contracts)} continuous contracts")
        indices=[df.index for df in vx_contracts.values()if not df.empty]
        if not indices:return pd.DataFrame()
        common_index=indices[0]
        for idx in indices[1:]:common_index=common_index.union(idx)
        common_index=common_index.sort_values()
        features=pd.DataFrame(index=common_index);vx_prices={};vx_dte={};vx_volume={};vx_oi={}
        for name,df in vx_contracts.items():
            if not df.empty:
                if 'settle'in df.columns:vx_prices[name]=df['settle'].reindex(common_index,method='ffill',limit=10)
                if 'days_to_expiry'in df.columns:vx_dte[name]=df['days_to_expiry'].reindex(common_index,method='ffill',limit=10)
                if 'volume'in df.columns:vx_volume[name]=df['volume'].reindex(common_index,method='ffill',limit=5)
                if 'open_interest'in df.columns:vx_oi[name]=df['open_interest'].reindex(common_index,method='ffill',limit=5)
        print("   → Generating legacy spread & ratio features...")
        features=self._add_legacy_spread_features(features,vx_prices)
        print("   → Generating term structure features...")
        features=self._add_term_structure_features(features,vx_prices,vx_dte)
        print("   → Generating positioning features...")
        features=self._add_positioning_features(features,vx_oi,vx_volume)
        print("   → Generating roll characteristics...")
        features=self._add_roll_features(features,vx_prices,vx_dte)
        if vix_series is not None:
            print("   → Generating VIX-VX basis features...")
            features=self._add_vix_basis_features(features,vx_prices,vix_series)
        print("   → Generating regime indicators...")
        features=self._add_regime_features(features,vx_prices)
        features=features.replace([np.inf,-np.inf],np.nan)
        if target_index is not None:features=features.reindex(target_index,method='ffill',limit=5)
        print(f"   ✓ Generated {len(features.columns)} VX features | Coverage: {len(features)} days")
        return features

    def _add_legacy_spread_features(self,features,vx_prices):
        if 'VX1'not in vx_prices or 'VX2'not in vx_prices:return features
        vx1,vx2=vx_prices['VX1'],vx_prices['VX2']
        features['vx1_vx2_spread']=vx1-vx2
        features['VX1-VX2']=features['vx1_vx2_spread']
        features['VX1-VX2_change_21d']=features['vx1_vx2_spread'].diff(21)
        features['VX1-VX2_zscore_63d']=self._zscore(features['vx1_vx2_spread'],63)
        features['VX1-VX2_percentile_63d']=self._percentile(features['vx1_vx2_spread'],63)
        ratio=vx2/vx1.replace(0,np.nan)
        features['VX2-VX1_RATIO']=ratio
        features['VX2-VX1_RATIO_velocity_10d']=ratio.diff(10)
        features['vx_term_structure_regime']=pd.cut(ratio,bins=[-np.inf,-0.05,0,0.05,np.inf],labels=[0,1,2,3]).astype(float)
        features['vx_curve_acceleration']=ratio.diff(5).diff(5)
        sp_rank=features['vx1_vx2_spread'].rolling(63,min_periods=32).rank(pct=True)
        r_rank=ratio.rolling(63,min_periods=32).rank(pct=True)
        features['vx_term_structure_divergence']=(sp_rank-r_rank).abs()
        return features

    def _add_term_structure_features(self,features,vx_prices,vx_dte):
        if 'VX1'in vx_prices and 'VX2'in vx_prices:
            vx1,vx2=vx_prices['VX1'],vx_prices['VX2']
            features['vx_contango_pct']=((vx2/vx1)-1)*100
            features['vx_contango_5d_change']=features['vx_contango_pct'].diff(5)
            features['vx_contango_zscore']=self._zscore(features['vx_contango_pct'],63)
            features['vx1_return_5d']=vx1.pct_change(5)*100
            features['vx2_return_5d']=vx2.pct_change(5)*100
        if all(f'VX{i}'in vx_prices for i in[1,2,3,4]):
            vx1,vx2,vx3,vx4=vx_prices['VX1'],vx_prices['VX2'],vx_prices['VX3'],vx_prices['VX4']
            features['vx_curve_2_1_ratio']=vx2/vx1
            features['vx_curve_3_2_ratio']=vx3/vx2
            features['vx_curve_4_3_ratio']=vx4/vx3
            features['vx_curve_steepness']=(vx4/vx1-1)*100
        return features

    def _add_positioning_features(self,features,vx_oi,vx_volume):
        if 'VX1'in vx_oi and 'VX2'in vx_oi:
            vx1_oi,vx2_oi=vx_oi['VX1'],vx_oi['VX2']
            features['vx_oi_front_concentration']=vx1_oi/(vx1_oi+vx2_oi).replace(0,np.nan)
            features['vx_oi_imbalance']=(vx1_oi-vx2_oi)/(vx1_oi+vx2_oi).replace(0,np.nan)
        if 'VX1'in vx_oi:
            vx1_oi=vx_oi['VX1']
            features['vx_oi_5d_change']=vx1_oi.diff(5)
            features['vx_oi_zscore_63d']=self._zscore(vx1_oi,63)
            oi_change_5d=vx1_oi.diff(5);oi_mean=oi_change_5d.rolling(63).mean();oi_std=oi_change_5d.rolling(63).std()
            features['vx_oi_surge']=((oi_change_5d-oi_mean)/oi_std.replace(0,np.nan)).clip(-3,3)
        if 'VX1'in vx_volume and 'VX1'in vx_oi:
            vol,oi=vx_volume['VX1'],vx_oi['VX1']
            features['vx_churn_ratio']=vol/oi.replace(0,np.nan)
            features['vx_churn_zscore_63d']=self._zscore(features['vx_churn_ratio'],63)
        return features

    def _add_roll_features(self,features,vx_prices,vx_dte):
        if 'VX1'in vx_prices and 'VX2'in vx_prices and 'VX1'in vx_dte:
            vx1,vx2=vx_prices['VX1'],vx_prices['VX2'];dte1=vx_dte['VX1']
            daily_roll_yield=(vx1-vx2)/dte1.replace(0,np.nan)
            features['vx_roll_yield_annualized']=daily_roll_yield*252
            features['vx_roll_cost_5d']=daily_roll_yield*5
            features['vx_roll_yield_zscore_63d']=self._zscore(daily_roll_yield,63)
        return features

    def _add_vix_basis_features(self,features,vx_prices,vix_series):
        if 'VX1'not in vx_prices:return features
        vx1=vx_prices['VX1'];vix_aligned=vix_series.reindex(features.index,method='ffill',limit=5)
        features['vx_vix_basis']=vx1-vix_aligned
        features['vx_vix_basis_pct']=(vx1-vix_aligned)/vix_aligned.replace(0,np.nan)*100
        features['vx_vix_basis_zscore_63d']=self._zscore(features['vx_vix_basis'],63)
        features['vx_vix_premium_flag']=(features['vx_vix_basis']>0).astype(int)
        return features

    def _add_regime_features(self,features,vx_prices):
        if 'vx_contango_pct'in features:
            contango=features['vx_contango_pct']
            features['vx_steep_contango_flag']=(contango>5).astype(int)
            features['vx_backwardation_flag']=(contango<0).astype(int)
            features['vx_extreme_backwardation_flag']=(contango<-10).astype(int)
            contango_5d=contango.diff(5);features['vx_regime_shift_velocity']=contango_5d
        if 'VX1'in vx_prices:
            vx1=vx_prices['VX1'];vx1_ret_5d=vx1.pct_change(5)
            ret_mean=vx1_ret_5d.rolling(126).mean();ret_std=vx1_ret_5d.rolling(126).std()
            features['vx_momentum_5d']=vx1_ret_5d
            features['vx_momentum_zscore']=(vx1_ret_5d-ret_mean)/ret_std.replace(0,np.nan)
        return features

    def _zscore(self,series,window):
        mean=series.rolling(window,min_periods=window//2).mean()
        std=series.rolling(window,min_periods=window//2).std()
        return(series-mean)/std.replace(0,np.nan)

    def _percentile(self,series,window):
        return series.rolling(window,min_periods=window//2).rank(pct=True)*100

def build_vx_futures_features(start_date,end_date,cboe_vx_dir="./CBOE_VX_ALL",target_index=None,vix_series=None):
    engineer=VXFuturesEngineer(cboe_vx_dir)
    return engineer.build_all_vx_features(start_date,end_date,target_index,vix_series)
