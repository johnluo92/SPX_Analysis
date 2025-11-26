import warnings
import numpy as np
import pandas as pd
from core.vx_continuous_contract_builder import get_vx_continuous_contracts
warnings.filterwarnings("ignore")

class VXFuturesEngineer:
    def __init__(self,cboe_vx_dir="./CBOE_VX_ALL",cache_dir="./data_cache/vx_continuous"):
        self.cboe_vx_dir=cboe_vx_dir;self.cache_dir=cache_dir

    def build_all_vx_features(self,start_date,end_date,target_index=None):
        vx=get_vx_continuous_contracts(start_date=start_date,end_date=end_date,cboe_vx_dir=self.cboe_vx_dir,cache_dir=self.cache_dir)
        if not vx:raise ValueError("Failed to load VX contracts")
        features=[self.extract_term_structure_features(vx),self.extract_roll_yield_features(vx),self.extract_cmf_features(vx),self.extract_basis_features(vx),self.extract_open_interest_features(vx),self.extract_volume_liquidity_features(vx),self.extract_intraday_range_features(vx),self.extract_roll_dynamics(vx)]
        all_f=pd.concat([f for f in features if not f.empty],axis=1);all_f=all_f.loc[:,~all_f.columns.duplicated()]
        if target_index is not None:all_f=all_f.reindex(target_index,method='ffill',limit=5)
        print(f"   ✓ Generated {len(all_f.columns)} VX features");return all_f

    def extract_term_structure_features(self,vx):
        idx=self._idx(vx);f=pd.DataFrame(index=idx);p=self._prices(vx,idx)
        if'VX1'in p and'VX2'in p:f['vx1_vx2_spread']=p['VX1']-p['VX2']
        if'VX1'in p and'VX3'in p:f['vx1_vx3_spread']=p['VX1']-p['VX3']
        if'VX1'in p and'VX4'in p:f['vx1_vx4_spread']=p['VX1']-p['VX4']
        if'VX1'in p and'VX6'in p:f['vx1_vx6_spread']=p['VX1']-p['VX6']
        if'VX2'in p and'VX3'in p:f['vx2_vx3_spread']=p['VX2']-p['VX3']
        if'VX2'in p and'VX4'in p:f['vx2_vx4_spread']=p['VX2']-p['VX4']
        if'VX3'in p and'VX6'in p:f['vx3_vx6_spread']=p['VX3']-p['VX6']
        if all(k in p for k in['VX1','VX6']):f['vx_curve_slope']=(p['VX6']-p['VX1'])/5;f['vx_front_back_ratio']=self._safe_div(p['VX1'],p['VX6'])
        if all(k in p for k in['VX1','VX2','VX3']):
            f['vx_curve_curvature']=p['VX1']-2*p['VX2']+p['VX3'];s12=p['VX1']-p['VX2'];s23=p['VX2']-p['VX3'];f['vx_curve_skew']=s12-s23;f['vx_curve_convexity_ratio']=self._safe_div(s12,s23)
        if all(k in p for k in['VX4','VX5','VX6']):f['vx_near_curvature']=p['VX1']-2*p['VX2']+p['VX3'];f['vx_far_curvature']=p['VX4']-2*p['VX5']+p['VX6']
        if'vx1_vx2_spread'in f:
            f['vx_curve_accel_5d']=f['vx1_vx2_spread'].diff(5);f['vx_spread_zscore_63d']=self._zscore(f['vx1_vx2_spread'],63);f['vx_curve_vol_21d']=f['vx1_vx2_spread'].rolling(21).std()
            for w in[21,63,126]:f[f'vx_spread_pct_{w}d']=self._pct(f['vx1_vx2_spread'],w)
            for w in[3,5,10]:f[f'vx_spread_vel_{w}d']=f['vx1_vx2_spread'].diff(w)
        if'vx1_vx2_spread'in f:
            f['vx_contango']=(f['vx1_vx2_spread']<0).astype(int);f['vx_backwardation']=(f['vx1_vx2_spread']>0).astype(int)
            if'VX1'in p and'VX2'in p:
                pct_diff=self._safe_div(p['VX2']-p['VX1'],p['VX1'])*100;f['vx_steep_contango']=(pct_diff>5).astype(int);pct_back=self._safe_div(p['VX1']-p['VX2'],p['VX2'])*100;f['vx_extreme_backward']=(pct_back>10).astype(int)
        if all(f'VX{i}'in p for i in[1,2,3]):f['vx_123_slope']=self._slope([p['VX1'],p['VX2'],p['VX3']])
        if all(f'VX{i}'in p for i in[4,5,6]):f['vx_456_slope']=self._slope([p['VX4'],p['VX5'],p['VX6']])
        if'VX1'in p and'VX6'in p:r1=p['VX1'].pct_change();r6=p['VX6'].pct_change();f['vx_front_back_divergence']=r1-r6
        return f

    def extract_roll_yield_features(self,vx):
        idx=self._idx(vx);f=pd.DataFrame(index=idx);p=self._prices(vx,idx)
        if'VX1'in vx and'VX2'in vx:
            spread=p['VX1']-p['VX2'];f['vx_roll_yield_daily']=spread/30;f['vx_roll_annualized']=f['vx_roll_yield_daily']*252
            f['vx_roll_cost_pct']=(self._safe_div(p['VX2']-p['VX1'],p['VX1'])*100).clip(lower=0);f['vx_roll_benefit_pct']=(self._safe_div(p['VX1']-p['VX2'],p['VX2'])*100).clip(lower=0)
            f['vx_contango_intensity']=f['vx_roll_cost_pct'];f['vx_backwardation_intensity']=f['vx_roll_benefit_pct'];f['vx_carry_zscore_63d']=self._zscore(f['vx_contango_intensity'],63);f['vx_carry_pct_252d']=self._pct(f['vx_contango_intensity'],252)
            f['vx_theoretical_pnl_5d']=f['vx_roll_yield_daily'].rolling(5).sum();f['vx_delta_roll_1d']=f['vx_roll_yield_daily'].diff(1);f['vx_delta_roll_5d']=f['vx_roll_yield_daily'].diff(5)
        return f

    def extract_cmf_features(self,vx):
        idx=self._idx(vx);f=pd.DataFrame(index=idx);p=self._prices(vx,idx)
        if len(p)<2:return f
        weights=[0.5,0.3,0.2,0.0,0.0,0.0];cmf=sum(p[f'VX{i+1}']*weights[i]for i in range(min(len(p),6))if f'VX{i+1}'in p)
        f['vx_cmf_price']=cmf;f['vx_cmf_ret_1d']=cmf.pct_change(1);f['vx_cmf_ret_5d']=cmf.pct_change(5)
        if'VX1'in p and'VX2'in p:slope=self._safe_div(p['VX2']-p['VX1'],p['VX1']);f['vx_mu_t']=slope.diff(1);f['vx_mu_t_ma5']=f['vx_mu_t'].rolling(5).mean()
        return f

    def extract_basis_features(self,vx):
        idx=self._idx(vx);f=pd.DataFrame(index=idx);p=self._prices(vx,idx)
        if'VX1'not in p:return f
        for i in range(2,7):
            if f'VX{i}'in p:basis=p[f'VX{i}']-p['VX1'];f[f'vx{i}_basis']=basis;f[f'vx{i}_basis_pct']=self._safe_div(basis,p['VX1'])*100
        if'VX2'in p and'VX3'in p:f['vx_basis_slope']=(p['VX3']-p['VX1'])/2;f['vx_basis_curvature']=p['VX1']-2*p['VX2']+p['VX3']
        return f

    def extract_open_interest_features(self,vx):
        idx=self._idx(vx);f=pd.DataFrame(index=idx);oi={n:df['open_interest'].reindex(idx)for n,df in vx.items()if not df.empty and'open_interest'in df.columns}
        for i in range(1,7):
            if f'VX{i}'in oi:f[f'vx{i}_oi']=oi[f'VX{i}']
        if oi:
            f['vx_total_oi']=sum(oi.values())
            if'VX1'in oi:f['vx_front_oi_pct']=self._safe_div(oi['VX1'],f['vx_total_oi'])*100
        if'VX1'in oi:
            v1=oi['VX1']
            for w in[1,5,10]:f[f'vx1_oi_chg_{w}d']=v1.diff(w)
            f['vx1_oi_zscore_63d']=self._zscore(v1,63);f['vx1_oi_pct_252d']=self._pct(v1,252);f['vx1_oi_vel_21d']=v1.diff(5).rolling(21).mean()
        if'VX1'in oi and'VX2'in oi:f['vx_oi_ratio_1_2']=self._safe_div(oi['VX1'],oi['VX2']);t12=oi['VX1']+oi['VX2'];f['vx_oi_imbalance']=self._safe_div(oi['VX1']-oi['VX2'],t12)
        if'VX2'in oi and'VX3'in oi:f['vx_oi_ratio_2_3']=self._safe_div(oi['VX2'],oi['VX3'])
        if all(f'VX{i}'in oi for i in[1,2,3])and'vx_total_oi'in f:front3=oi['VX1']+oi['VX2']+oi['VX3'];f['vx_oi_conc_front3']=self._safe_div(front3,f['vx_total_oi'])
        if'VX1'in oi and'VX2'in oi:c1=oi['VX1'].diff(5);c2=oi['VX2'].diff(5);f['vx_oi_spread_div']=c1-c2
        if'vx_total_oi'in f:f['vx_oi_expansion_rate']=f['vx_total_oi'].pct_change(10)*100
        if'VX1'in oi:chg=oi['VX1'].diff(5);z=self._safe_div(chg-chg.rolling(63).mean(),chg.rolling(63).std());f['vx_oi_surge']=(z.abs()>2).astype(int)
        return f

    def extract_volume_liquidity_features(self,vx):
        idx=self._idx(vx);f=pd.DataFrame(index=idx);vol={n:df['volume'].reindex(idx)for n,df in vx.items()if not df.empty and'volume'in df.columns};oi={n:df['open_interest'].reindex(idx)for n,df in vx.items()if not df.empty and'open_interest'in df.columns};p=self._prices(vx,idx)
        if'VX1'in vol:v1=vol['VX1'];f['vx1_volume']=v1;f['vx1_vol_ma21']=v1.rolling(21).mean();f['vx1_vol_surge']=self._safe_div(v1-v1.rolling(63).mean(),v1.rolling(63).std())
        if'VX1'in vol and'VX2'in vol:f['vx_vol_ratio_1_2']=self._safe_div(vol['VX1'],vol['VX2'])
        if vol:
            f['vx_total_volume']=sum(vol.values())
            if'VX1'in vol:f['vx_vol_conc']=self._safe_div(vol['VX1'],f['vx_total_volume'])
        if'VX1'in vol and'VX1'in oi:f['vx1_churn']=self._safe_div(vol['VX1'],oi['VX1']);f['vx1_churn_zscore']=self._zscore(f['vx1_churn'],63)
        if'VX1'in vol and'VX1'in p:pc=p['VX1'].pct_change();f['vx_vol_price_corr_21d']=vol['VX1'].rolling(21).corr(pc)
        if'VX1'in vol and'VX1'in p:
            pc=p['VX1'].diff();up=pc>0;down=pc<0;vol_up=vol['VX1'].where(up).rolling(21).mean();vol_down=vol['VX1'].where(down).rolling(21).mean()
            f['vx_vol_on_up']=vol_up;f['vx_vol_on_down']=vol_down;f['vx_vol_asymmetry']=self._safe_div(vol_up,vol_down)
        return f

    def extract_intraday_range_features(self,vx):
        if'VX1'not in vx or vx['VX1'].empty:return pd.DataFrame()
        v1=vx['VX1'];f=pd.DataFrame(index=v1.index);req=['open','high','low','close']
        if not all(c in v1.columns for c in req):return f
        o,h,l,c=v1['open'],v1['high'],v1['low'],v1['close'];f['vx1_range']=h-l;f['vx1_range_pct']=self._safe_div(h-l,c)*100
        pc=c.shift(1);tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1);f['vx1_atr_14']=tr.rolling(14).mean();avg=f['vx1_range'].rolling(21).mean()
        f['vx1_range_expansion']=self._safe_div(f['vx1_range'],avg);f['vx1_range_zscore_63d']=self._zscore(f['vx1_range'],63);body=(c-o).abs();total=h-l;f['vx1_body_pct']=self._safe_div(body,total)*100
        upper=h-pd.concat([o,c],axis=1).max(axis=1);lower=pd.concat([o,c],axis=1).min(axis=1)-l;f['vx1_upper_shadow']=self._safe_div(upper,body);f['vx1_lower_shadow']=self._safe_div(lower,body);f['vx1_doji']=(f['vx1_body_pct']<20).astype(int)
        f['vx1_gap']=o-pc;f['vx1_gap_pct']=self._safe_div(f['vx1_gap'],pc)*100;gap_up=o>pc;gap_down=o<pc;gap_filled=(gap_up&(l<=pc))|(gap_down&(h>=pc));f['vx1_gap_filled']=gap_filled.astype(int)
        return f

    def extract_roll_dynamics(self,vx):
        idx=self._idx(vx);f=pd.DataFrame(index=idx)
        if'VX1'not in vx or'roll_indicator'not in vx['VX1'].columns:return f
        v1=vx['VX1'];roll=v1['roll_indicator'].reindex(idx);f['vx_roll_event']=roll;rolls=roll[roll==1].index
        if len(rolls)>0:
            ds=pd.Series(np.nan,index=idx);du=pd.Series(np.nan,index=idx)
            for d in idx:
                past=rolls[rolls<=d];future=rolls[rolls>d]
                if len(past)>0:ds.loc[d]=(d-past[-1]).days
                if len(future)>0:du.loc[d]=(future[0]-d).days
            f['vx_days_since_roll']=ds;f['vx_days_to_roll']=du
        f['vx_roll_freq_63d']=roll.rolling(63).sum()
        return f

    def _idx(self,vx):
        indices=[df.index for df in vx.values()if not df.empty]
        if not indices:return pd.DatetimeIndex([])
        common=indices[0]
        for i in indices[1:]:common=common.intersection(i)
        return common

    def _prices(self,vx,idx):
        return{n:df['settle'].reindex(idx)for n,df in vx.items()if not df.empty and'settle'in df.columns}

    def _safe_div(self,num,denom):
        result=num/denom.replace(0,np.nan);return result.replace([np.inf,-np.inf],np.nan)

    def _zscore(self,s,w):
        m=s.rolling(w).mean();std=s.rolling(w).std();z=(s-m)/std.replace(0,np.nan);return z.replace([np.inf,-np.inf],np.nan)

    def _pct(self,s,w):
        return s.rolling(w).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1]*100,raw=False)

    def _slope(self,pl):
        df=pd.concat(pl,axis=1)
        def rs(row):
            x=np.arange(len(row));y=row.values
            if np.any(np.isnan(y)):return np.nan
            return np.polyfit(x,y,1)[0]
        return df.apply(rs,axis=1)

def build_vx_futures_features(start_date,end_date,cboe_vx_dir="./CBOE_VX_ALL",target_index=None):
    eng=VXFuturesEngineer(cboe_vx_dir);return eng.build_all_vx_features(start_date,end_date,target_index)
