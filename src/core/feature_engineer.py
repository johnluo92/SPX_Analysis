import warnings
from datetime import datetime,timedelta
from typing import Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
from config import TRAINING_YEARS,CALENDAR_COHORTS,COHORT_PRIORITY,ENABLE_TEMPORAL_SAFETY,FEATURE_QUALITY_CONFIG,PUBLICATION_LAGS,TARGET_CONFIG,MACRO_EVENT_CONFIG
from core.calculations import calculate_robust_zscore,calculate_regime_with_validation,calculate_percentile_with_validation,VIX_REGIME_BINS,VIX_REGIME_LABELS,SKEW_REGIME_BINS,SKEW_REGIME_LABELS
from core.temporal_validator import TemporalSafetyValidator
from core.regime_classifier import RegimeClassifier
warnings.filterwarnings("ignore")
try:
    from config import CALENDAR_COHORTS,COHORT_PRIORITY,ENABLE_TEMPORAL_SAFETY,FEATURE_QUALITY_CONFIG,PUBLICATION_LAGS,TARGET_CONFIG
except ImportError:
    ENABLE_TEMPORAL_SAFETY=False;PUBLICATION_LAGS={};CALENDAR_COHORTS={};COHORT_PRIORITY=[];TARGET_CONFIG={};FEATURE_QUALITY_CONFIG={}
    warnings.warn("⚠️ Calendar cohort config not found - cohort features disabled")
class MetaFeatureEngine:
    @staticmethod
    def extract_regime_indicators(df:pd.DataFrame,vix:pd.Series,spx:pd.Series)->pd.DataFrame:
        m=pd.DataFrame(index=df.index)
        if "vix" in df.columns:
            v=df["vix"];m["vix_regime_micro"]=calculate_regime_with_validation(v,bins=[0,12,16,20,100],labels=[0,1,2,3],feature_name="vix_micro")
            if "vix_velocity_5d" in df.columns:m["regime_transition_risk"]=(df["vix_velocity_5d"].abs()/v.replace(0,np.nan)*100).clip(0,100)
        if all(c in df.columns for c in ["spx_realized_vol_21d","vix"]):
            rv,v=df["spx_realized_vol_21d"],df["vix"];m["vol_regime"]=calculate_regime_with_validation(rv,bins=[0,10,15,25,100],labels=[0,1,2,3],feature_name="vol");rp=v-rv;m["risk_premium_regime"]=calculate_regime_with_validation(rp,bins=[-100,0,5,10,100],labels=[0,1,2,3],feature_name="risk_premium")
            if "vix_term_structure" in df.columns:m["vol_term_regime"]=calculate_regime_with_validation(df["vix_term_structure"],bins=[-100,-2,0,2,100],labels=[0,1,2,3],feature_name="vol_term")
        if "spx_vs_ma200" in df.columns:
            t=df["spx_vs_ma200"];m["trend_regime"]=calculate_regime_with_validation(t,bins=[-100,-5,0,5,100],labels=[0,1,2,3],feature_name="trend")
            if "spx_vs_ma50" in df.columns:m["trend_strength"]=(df["spx_vs_ma200"].abs()+df["spx_vs_ma50"].abs())/2
        sc=[]
        if "SKEW" in df.columns:sc.append(((df["SKEW"]-130)/30).clip(0,1))
        if "vix" in df.columns:sc.append(((df["vix"]-15)/25).clip(0,1))
        if "spx_realized_vol_21d" in df.columns:sc.append(((df["spx_realized_vol_21d"]-15)/20).clip(0,1))
        if sc:m["liquidity_stress_composite"]=pd.DataFrame(sc).T.mean(axis=1);m["liquidity_regime"]=calculate_regime_with_validation(m["liquidity_stress_composite"],bins=[0,0.25,0.5,0.75,1],labels=[0,1,2,3],feature_name="liquidity")
        if "spx_vix_corr_21d" in df.columns:m["correlation_regime"]=calculate_regime_with_validation(df["spx_vix_corr_21d"],bins=[-1,-0.8,-0.5,0,1],labels=[0,1,2,3],feature_name="correlation")
        return m
    @staticmethod
    def extract_cross_asset_relationships(df:pd.DataFrame,macro:pd.DataFrame=None)->pd.DataFrame:
        m=pd.DataFrame(index=df.index)
        if all(c in df.columns for c in ["spx_ret_21d","vix_velocity_21d"]):
            sr,vc=df["spx_ret_21d"],df["vix_velocity_21d"];m["equity_vol_divergence"]=((sr.rank(pct=True)+vc.rank(pct=True))-1).abs()
            if "spx_vix_corr_21d" in df.columns:cr=df["spx_vix_corr_21d"];m["equity_vol_corr_breakdown"]=(cr-cr.rolling(63).mean()).abs()
        if all(c in df.columns for c in ["vix","spx_realized_vol_21d"]):
            rp=df["vix"]-df["spx_realized_vol_21d"];m["risk_premium_ma21"]=rp.rolling(21).mean();m["risk_premium_velocity"]=rp.diff(10);m["risk_premium_zscore"]=calculate_robust_zscore(rp,63)
        if macro is not None:
            if "Gold" in macro.columns and "spx_ret_21d" in df.columns:gr=macro["Gold"].pct_change(21)*100;m["gold_spx_divergence"]=(gr.rank(pct=True)-df["spx_ret_21d"].rank(pct=True)).abs()
            if "Dollar_Index" in macro.columns and "spx_ret_21d" in df.columns:dr=macro["Dollar_Index"].pct_change(21)*100;m["dollar_spx_correlation"]=dr.rolling(63).corr(df["spx_ret_21d"])
        return m
    @staticmethod
    def extract_rate_of_change_features(df:pd.DataFrame)->pd.DataFrame:
        m=pd.DataFrame(index=df.index);rs={"vix":df.get("vix"),"SKEW":df.get("SKEW"),"spx_realized_vol_21d":df.get("spx_realized_vol_21d")}
        for n,s in rs.items():
            if s is None:continue
            if n=="vix":m["vix_velocity_3d_pct"]=s.pct_change(3)*100
            if n in ["vix","SKEW"]:v5=s.diff(5);m[f"{n}_jerk_5d"]=v5.diff(5).diff(5);m[f"{n}_momentum_regime"]=np.sign(s.diff(5))
            if n=="spx_realized_vol_21d":m["spx_realized_vol_21d_velocity_3d"]=s.diff(3);m["spx_realized_vol_21d_acceleration_5d"]=s.diff(5).diff(5)
        if all(c in df.columns for c in ["vix","SKEW"]):vm,sm=df["vix"].diff(10),df["SKEW"].diff(10);m["vix_skew_momentum_divergence"]=(vm.rank(pct=True)-sm.rank(pct=True)).abs()
        return m
    @staticmethod
    def extract_percentile_rankings(df:pd.DataFrame)->pd.DataFrame:
        m=pd.DataFrame(index=df.index);rs={"vix":df.get("vix"),"SKEW":df.get("SKEW")}
        if all(c in df.columns for c in ["vix","spx_realized_vol_21d"]):rs["risk_premium"]=df["vix"]-df["spx_realized_vol_21d"]
        for n,s in rs.items():
            if s is None:continue
            for w in [21,63,126,252]:m[f"{n}_percentile_{w}d"]=calculate_percentile_with_validation(s,w)
            if f"{n}_percentile_63d" in m.columns:m[f"{n}_percentile_velocity"]=m[f"{n}_percentile_63d"].diff(10)
            for w in [63,252]:
                pc=f"{n}_percentile_{w}d"
                if pc in m.columns:m[f"{n}_extreme_low_{w}d"]=(m[pc]<10).astype(int)
        if "risk_premium_percentile_63d" in m.columns:m["risk_premium_extreme_low_63d"]=(m["risk_premium_percentile_63d"]<10).astype(int)
        return m
class FuturesFeatureEngine:
    @staticmethod
    def extract_vix_futures_features(vx:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        if "VX1-VX2" in vx:sp=vx["VX1-VX2"];f["VX1-VX2"]=sp;f["VX1-VX2_change_21d"]=sp.diff(21);f["VX1-VX2_zscore_63d"]=calculate_robust_zscore(sp,63);f["VX1-VX2_percentile_63d"]=calculate_percentile_with_validation(sp,63)
        if "VX2-VX1_RATIO" in vx:r=vx["VX2-VX1_RATIO"];f["VX2-VX1_RATIO"]=r;f["VX2-VX1_RATIO_velocity_10d"]=r.diff(10);f["vx_term_structure_regime"]=calculate_regime_with_validation(r,bins=[-1,-0.05,0,0.05,1],labels=[0,1,2,3],feature_name="vx_ratio")
        if "VX1-VX2" in vx and "VX2-VX1_RATIO" in vx:sp,r=vx["VX1-VX2"],vx["VX2-VX1_RATIO"];f["vx_curve_acceleration"]=r.diff(5).diff(5);f["vx_term_structure_divergence"]=(sp.rolling(63).rank(pct=True)-r.rolling(63).rank(pct=True)).abs()
        return f
    @staticmethod
    def extract_commodity_futures_features(fd:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        if "CL1-CL2" in fd:cls=fd["CL1-CL2"];f["CL1-CL2"]=cls;f["CL1-CL2_velocity_5d"]=cls.diff(5);f["CL1-CL2_zscore_63d"]=calculate_robust_zscore(cls,63);f["oil_term_regime"]=calculate_regime_with_validation(cls,bins=[-10,-1,0,2,20],labels=[0,1,2,3],feature_name="cl_spread")
        if "Crude_Oil" in fd:
            p=fd["Crude_Oil"]
            for w in [10,21,63]:f[f"crude_oil_ret_{w}d"]=p.pct_change(w)*100
            f["crude_oil_vol_21d"]=p.pct_change().rolling(21).std()*np.sqrt(252)*100;f["crude_oil_zscore_63d"]=calculate_robust_zscore(p,63)
        return f
    @staticmethod
    def extract_dollar_futures_features(dd:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        if "DX1-DX2" in dd:dxs=dd["DX1-DX2"];f["DX1-DX2"]=dxs;f["DX1-DX2_velocity_5d"]=dxs.diff(5);f["DX1-DX2_zscore_63d"]=calculate_robust_zscore(dxs,63)
        if "Dollar_Index" in dd:
            dxy=dd["Dollar_Index"]
            for w in [10,21,63]:f[f"dxy_ret_{w}d"]=dxy.pct_change(w)*100
            for w in [50,200]:ma=dxy.rolling(w).mean();f[f"dxy_vs_ma{w}"]=((dxy-ma)/ma.replace(0,np.nan))*100
            f["dxy_vol_21d"]=dxy.pct_change().rolling(21).std()*np.sqrt(252)*100
        return f
    @staticmethod
    def extract_futures_cross_relationships(vx:Dict[str,pd.Series],cd:Dict[str,pd.Series],dd:Dict[str,pd.Series],sr:pd.Series=None)->pd.DataFrame:
        f=pd.DataFrame()
        if "VX1-VX2" in vx and "CL1-CL2" in cd:vs,cls=vx["VX1-VX2"],cd["CL1-CL2"];f["vx_crude_corr_21d"]=vs.rolling(21).corr(cls);f["vx_crude_divergence"]=(vs.rolling(63).rank(pct=True)-cls.rolling(63).rank(pct=True)).abs()
        if "VX1-VX2" in vx and "Dollar_Index" in dd:f["vx_dollar_corr_21d"]=vx["VX1-VX2"].rolling(21).corr(dd["Dollar_Index"].pct_change(21)*100)
        if "Dollar_Index" in dd and "Crude_Oil" in cd:f["dollar_crude_corr_21d"]=dd["Dollar_Index"].pct_change().rolling(21).corr(cd["Crude_Oil"].pct_change())
        if sr is not None:
            if "VX1-VX2" in vx:f["spx_vx_spread_corr_21d"]=sr.rolling(21).corr(vx["VX1-VX2"])
            if "Dollar_Index" in dd:f["spx_dollar_corr_21d"]=sr.rolling(21).corr(dd["Dollar_Index"].pct_change(21)*100)
        return f
class TreasuryYieldFeatureEngine:
    @staticmethod
    def extract_term_spreads(y:pd.DataFrame)->pd.DataFrame:
        f=pd.DataFrame(index=y.index);req=["3M","2Y","10Y"]
        if not all(c in y.columns for c in req):return f
        f["yield_10y2y"]=y["10Y"]-y["2Y"];f["yield_10y2y_zscore"]=calculate_robust_zscore(f["yield_10y2y"],252);f["yield_10y3m"]=y["10Y"]-y["3M"];f["yield_2y3m"]=y["2Y"]-y["3M"]
        if "5Y" in y.columns:f["yield_5y2y"]=y["5Y"]-y["2Y"]
        if "30Y" in y.columns:f["yield_30y10y"]=y["30Y"]-y["10Y"]
        for sn in ["yield_10y2y","yield_10y3m","yield_2y3m"]:
            if sn not in f.columns:continue
            sp=f[sn];f[f"{sn}_velocity_10d"]=sp.diff(10);f[f"{sn}_velocity_21d"]=sp.diff(21);f[f"{sn}_velocity_63d"]=sp.diff(63);f[f"{sn}_acceleration"]=sp.diff(10).diff(10);f[f"{sn}_vs_ma63"]=sp-sp.rolling(63).mean();f[f"{sn}_percentile_252d"]=calculate_percentile_with_validation(sp,252)
        if "yield_10y2y" in f.columns:f["yield_10y2y_inversion_depth"]=f["yield_10y2y"].clip(upper=0).abs()
        if "yield_10y3m" in f.columns:f["yield_10y3m_inversion_depth"]=f["yield_10y3m"].clip(upper=0).abs()
        return f
    @staticmethod
    def extract_curve_shape(y:pd.DataFrame)->pd.DataFrame:
        f=pd.DataFrame(index=y.index);req=["3M","2Y","10Y"]
        if not all(c in y.columns for c in req):return f
        ay=[c for c in ["3M","2Y","5Y","10Y","30Y"] if c in y.columns]
        if ay:f["yield_curve_level"]=y[ay].mean(axis=1);f["yield_curve_level_zscore"]=calculate_robust_zscore(f["yield_curve_level"],252)
        if "5Y" in y.columns:f["yield_curve_curvature"]=2*y["5Y"]-y["2Y"]-y["10Y"];f["yield_curve_curvature_zscore"]=calculate_robust_zscore(f["yield_curve_curvature"],252)
        return f
    @staticmethod
    def extract_rate_volatility(y:pd.DataFrame)->pd.DataFrame:
        f=pd.DataFrame(index=y.index)
        for c in ["6M","10Y","30Y"]:
            if c in y.columns:f[f"{c.lower()}_vol_63d"]=y[c].diff().rolling(63).std()*np.sqrt(252)
        vc=[c for c in f.columns if "_vol_" in c]
        if vc:f["yield_curve_vol_avg"]=f[vc].mean(axis=1);f["yield_curve_vol_dispersion"]=f[vc].std(axis=1)
        return f
class FeatureEngineer:
    def __init__(self,data_fetcher):
        self.fetcher=data_fetcher;self.meta_engine=MetaFeatureEngine();self.futures_engine=FuturesFeatureEngine();self.treasury_engine=TreasuryYieldFeatureEngine();self.validator=TemporalSafetyValidator();self.regime_classifier=RegimeClassifier();self.fomc_calendar=None;self.opex_calendar=None;self.earnings_calendar=None;self.vix_futures_expiry=None;self._cohort_cache={}
    def _load_calendar_data(self):
        if self.fomc_calendar is None:
            try:sy,ey=self.training_start_date.year,self.training_end_date.year+1;self.fomc_calendar=self.fetcher.fetch_fomc_calendar(start_year=sy,end_year=ey)
            except Exception as e:print(f"⚠️ FOMC calendar unavailable, using stub");self.fomc_calendar=pd.DataFrame()
        if self.opex_calendar is None:self.opex_calendar=self._generate_opex_calendar()
        if self.vix_futures_expiry is None:self.vix_futures_expiry=self._generate_vix_futures_expiry()
        if self.earnings_calendar is None:self.earnings_calendar=pd.DataFrame()
    def _generate_opex_calendar(self,sy=None,ey=None):
        if sy is None:sy=self.training_start_date.year
        if ey is None:ey=self.training_end_date.year+1
        od=[]
        for yr in range(sy,ey+1):
            for mo in range(1,13):
                fp=pd.Timestamp(yr,mo,15);da=(4-fp.weekday())%7
                if da==0 and fp.day>15:da=7
                tf=fp+pd.Timedelta(days=da);od.append({"date":tf,"expiry_type":"monthly_opex"})
        return pd.DataFrame(od).set_index("date").sort_index()
    def _generate_vix_futures_expiry(self):
        if self.opex_calendar is None:self._generate_opex_calendar()
        ve=[]
        for od in self.opex_calendar.index:
            ad=od-pd.Timedelta(days=30);dtw=(2-ad.weekday())%7;vd=ad+pd.Timedelta(days=dtw);ve.append({"date":vd,"expiry_type":"vix_futures"})
        return pd.DataFrame(ve).set_index("date").sort_index()
    def get_calendar_cohort(self,date):
        date=pd.Timestamp(date)
        if date in self._cohort_cache:return self._cohort_cache[date]
        if self.opex_calendar is None:self._load_calendar_data()
        dto=self._days_to_monthly_opex(date);dtf=self._days_to_fomc(date);dtve=self._days_to_vix_futures_expiry(date);ep=self._spx_earnings_intensity(date);is_cpi=self._is_cpi_release_day(date);is_pce=self._is_pce_release_day(date);is_fomc_minutes=self._is_fomc_minutes_day(date)
        for cn in COHORT_PRIORITY:
            cd=CALENDAR_COHORTS[cn];cond=cd["condition"]
            if cond=="macro_event_period":
                if dtf is not None:
                    rmin,rmax=cd["range"]
                    if rmin<=dtf<=rmax:res=(cn,cd["weight"]);self._cohort_cache[date]=res;return res
                if is_cpi or is_pce or is_fomc_minutes:res=(cn,cd["weight"]);self._cohort_cache[date]=res;return res
            elif cond=="days_to_monthly_opex":
                if dto is not None or dtve is not None:
                    rmin,rmax=cd["range"]
                    if(dto is not None and rmin<=dto<=rmax)or(dtve is not None and rmin<=dtve<=rmax):res=(cn,cd["weight"]);self._cohort_cache[date]=res;return res
            elif cond=="spx_earnings_pct":
                if ep is not None:
                    rmin,rmax=cd["range"]
                    if rmin<=ep<=rmax:res=(cn,cd["weight"]);self._cohort_cache[date]=res;return res
            elif cond=="default":res=(cn,cd["weight"]);self._cohort_cache[date]=res;return res
        raise ValueError(f"No cohort matched for date {date}")
    def _days_to_monthly_opex(self,date):
        if self.opex_calendar is None or len(self.opex_calendar)==0:return None
        fo=self.opex_calendar[self.opex_calendar.index>=date]
        if len(fo)==0:return None
        nxt=fo.index[0];dd=(nxt-date).days
        if dd==0:return 0
        return -dd if dd>0 else dd
    def _days_to_fomc(self,date):
        if self.fomc_calendar is None or len(self.fomc_calendar)==0:return None
        ff=self.fomc_calendar[self.fomc_calendar.index>=date]
        if len(ff)==0:return None
        return -(ff.index[0]-date).days
    def _days_to_vix_futures_expiry(self,date):
        if self.vix_futures_expiry is None or len(self.vix_futures_expiry)==0:return None
        fe=self.vix_futures_expiry[self.vix_futures_expiry.index>=date]
        if len(fe)==0:return None
        return -(fe.index[0]-date).days
    def _spx_earnings_intensity(self,date):
        mo=date.month
        if mo in [1,4,7,10]:
            wom=(date.day-1)//7+1
            if wom in [2,3,4]:return 0.25
        return 0.05
    def _is_cpi_release_day(self,date):
        target=MACRO_EVENT_CONFIG["cpi_release"]["day_of_month_target"];window=MACRO_EVENT_CONFIG["cpi_release"]["window_days"]
        return abs(date.day-target)<=window
    def _is_pce_release_day(self,date):
        target=MACRO_EVENT_CONFIG["pce_release"]["day_of_month_target"];window=MACRO_EVENT_CONFIG["pce_release"]["window_days"]
        return abs(date.day-target)<=window
    def _is_fomc_minutes_day(self,date):
        if self.fomc_calendar is None or len(self.fomc_calendar)==0:return False
        days_after=MACRO_EVENT_CONFIG["fomc_minutes"]["days_after_meeting"];window=MACRO_EVENT_CONFIG["fomc_minutes"]["window_days"]
        for fomc_date in self.fomc_calendar.index:
            minutes_date=fomc_date+pd.Timedelta(days=days_after)
            if abs((date-minutes_date).days)<=window:return True
        return False
    def _align_features_for_prediction(self,bf:pd.DataFrame,cf:pd.DataFrame,ff:pd.DataFrame,mf:pd.DataFrame,tf:pd.DataFrame,vvf:pd.DataFrame,mi:pd.DatetimeIndex)->pd.DataFrame:
        tfi=tf.reindex(mi,method="ffill",limit=3);cfi=cf.reindex(mi,method="ffill",limit=5);mfi=mf.reindex(mi,method="ffill",limit=45);ffi=ff.reindex(mi,method="ffill",limit=3);vvfi=vvf.reindex(mi,method="ffill",limit=5);ba=bf.reindex(mi);al=pd.concat([ba,cfi,ffi,mfi,tfi,vvfi],axis=1);return al
    def _validate_term_structure_timing(self,vix:pd.Series,cd:pd.DataFrame,pd_date:datetime=None)->bool:
        if cd is None or "VIX3M" not in cd.columns:return True
        if pd_date is None:pd_date=vix.index[-1]
        v3m=cd["VIX3M"];v3l=PUBLICATION_LAGS.get("VIX3M",0);lv3=(v3m.dropna().index[-1] if len(v3m.dropna())>0 else None)
        if lv3 is not None:
            ad=pd_date-timedelta(days=v3l)
            if lv3>ad:warnings.warn(f"⚠️ VIX3M term structure may have T+{v3l} leakage: Using data from {lv3.date()} but prediction date is {pd_date.date()}");return False
        return True
    def apply_quality_control(self,features:pd.DataFrame):
        if not hasattr(self,"quality_controller"):return features
        cf,rep=self.quality_controller.validate_features(features)
        import os
        os.makedirs("./data_cache",exist_ok=True);ts=datetime.now().strftime("%Y%m%d_%H%M%S");self.quality_controller.save_report(rep,f"./data_cache/quality_report_{ts}.json");return cf
    def build_complete_features(self,years:int=TRAINING_YEARS,end_date:Optional[str]=None,force_fresh:bool=False)->dict:
        print(f"\n{'='*80}");print(f"🏗️  Building {years}y feature set | Temporal Safety: {'ON' if ENABLE_TEMPORAL_SAFETY else 'OFF'}");print(f"{'='*80}")
        if end_date is None:ed=datetime.now();mode="LIVE"
        else:ed=pd.Timestamp(end_date);ir=ed>(datetime.now()-timedelta(days=7));mode="RECENT" if ir else "HISTORICAL"
        sd=ed-timedelta(days=years*365+450);self.training_start_date=sd;self.training_end_date=ed;ss=sd.strftime("%Y-%m-%d");es=ed.strftime("%Y-%m-%d")
        print(f"Mode: {mode}");print(f"Date range: {ss} → {es}");print(f"  Warmup period: {ss} → {(sd+timedelta(days=450)).strftime('%Y-%m-%d')}");print(f"  Usable data: {(sd+timedelta(days=450)).strftime('%Y-%m-%d')} → {es}")
        spx_df=self.fetcher.fetch_yahoo("^GSPC",ss,es);vix=self.fetcher.fetch_yahoo("^VIX",ss,es);vvix=self.fetcher.fetch_yahoo("^VVIX",ss,es)
        if spx_df is None or vix is None:raise ValueError("❌ Core data fetch failed")
        spx=spx_df["Close"].squeeze();vix=vix["Close"].squeeze();vix=vix.reindex(spx.index,method="ffill",limit=5);spx_ohlc=spx_df.reindex(spx.index,method="ffill",limit=5)
        if vvix is not None:vvix_series=vvix["Close"].squeeze().reindex(spx.index,method="ffill",limit=5)
        else:vvix_series=None
        cbd=self.fetcher.fetch_all_cboe()
        if cbd:cb=pd.DataFrame(index=spx.index);[cb.update({s:ser.reindex(spx.index,method="ffill",limit=5)}) for s,ser in cbd.items()]
        else:cb=pd.DataFrame(index=spx.index)
        bf=self._build_base_features(spx,vix,spx_ohlc,cb);cbf=self._build_cboe_features(cb,vix) if not cb.empty else pd.DataFrame(index=spx.index);ff=self._build_futures_features(ss,es,spx.index,spx,cb);md=self._fetch_macro_data(ss,es,spx.index);mf=self._build_macro_features(md) if md is not None else pd.DataFrame(index=spx.index);tf=self._build_treasury_features(ss,es,spx.index);vvf=self._build_vvix_features(vvix_series,vix) if vvix_series is not None else pd.DataFrame(index=spx.index);cmb=pd.concat([bf,cbf],axis=1);mtf=self._build_meta_features(cmb,spx,vix,md);calf=self._build_calendar_features(spx.index)
        af=pd.concat([bf,cbf,ff,mf,tf,vvf,mtf,calf],axis=1);af=af.loc[:,~af.columns.duplicated()];af=self._ensure_numeric_dtypes(af)
        self._load_calendar_data();cohd=[{"calendar_cohort":self.get_calendar_cohort(dt)[0],"cohort_weight":self.get_calendar_cohort(dt)[1]} for dt in af.index];cohdf=pd.DataFrame(cohd,index=af.index);cohdf["is_fomc_period"]=(cohdf["calendar_cohort"]=="fomc_period").astype(int);cohdf["is_opex_week"]=(cohdf["calendar_cohort"]=="opex_week").astype(int);cohdf["is_earnings_heavy"]=(cohdf["calendar_cohort"]=="earnings_heavy").astype(int);af=pd.concat([af,cohdf],axis=1)
        af["feature_quality"]=self.validator.compute_feature_quality_batch(af);af=self.apply_quality_control(af)
        if ENABLE_TEMPORAL_SAFETY and not cb.empty:self._validate_term_structure_timing(vix,cb)
        print(f"\n✅ Complete: {len(af.columns)} features | {len(af)} rows");print(f"   Date range: {af.index[0].date()} → {af.index[-1].date()}");print(f"{'='*80}\n");return {"features":af,"spx":spx,"vix":vix,"cboe_data":cb if cbd else None,"vvix":vvix_series}
    def _build_base_features(self,spx:pd.Series,vix:pd.Series,so:pd.DataFrame,cb:pd.DataFrame=None)->pd.DataFrame:
        f=pd.DataFrame(index=spx.index);f["vix"],f["spx_lag1"]=vix,spx.shift(1)
        for w in [1,5,10,21]:f[f"vix_ret_{w}d"]=vix.pct_change(w)*100
        vr=vix.pct_change()
        for w in [10,21,63]:f[f"vix_vol_{w}d"]=vr.rolling(w).std()*np.sqrt(252)*100
        for w in [5,10,21]:f[f"vix_velocity_{w}d"]=vix.diff(w)
        for w in [10,21,63,252]:ma=vix.rolling(w).mean();f[f"vix_vs_ma{w}"]=((vix-ma)/ma.replace(0,np.nan))*100
        for w in [21,63,252]:f[f"vix_zscore_{w}d"]=calculate_robust_zscore(vix,w)
        for w in [10,21]:mom=vix.diff(w);f[f"vix_momentum_z_{w}d"]=calculate_robust_zscore(mom,63)
        f["vix_accel_5d"]=vix.diff(5).diff(5);vm21,vm63=vix.rolling(21).mean(),vix.rolling(63).mean();f["vix_stretch_ma21"]=(vix-vm21).abs();f["vix_stretch_ma63"]=(vix-vm63).abs()
        for w in [21,63]:ma=vix.rolling(w).mean();f[f"reversion_strength_{w}d"]=(vix-ma).abs()/ma.replace(0,np.nan)
        bw=20;bma,bstd=vix.rolling(bw).mean(),vix.rolling(bw).std();bu,bl=bma+2*bstd,bma-2*bstd;f["vix_bb_position_20d"]=((vix-bl)/(bu-bl).replace(0,np.nan)).clip(0,1);f["vix_extreme_low_21d"]=(vix<vix.rolling(21).quantile(0.1)).astype(int);f["vix_regime"]=self.regime_classifier.classify_vix_series_numeric(vix);rc=f["vix_regime"].diff().fillna(0)!=0;f["days_in_regime"]=(~rc).cumsum()-(~rc).cumsum().where(rc).ffill().fillna(0)
        if cb is not None and "VIX3M" in cb.columns:f["vix_term_structure"]=((vix/cb["VIX3M"].replace(0,np.nan))-1)*100
        else:f["vix_term_structure"]=np.nan
        for w in [1,5,10,21,63]:f[f"spx_ret_{w}d"]=spx.pct_change(w)*100
        for w in [20,50,200]:ma=spx.rolling(w).mean();f[f"spx_vs_ma{w}"]=((spx-ma)/ma.replace(0,np.nan))*100
        for w in [10,21]:mom=spx.pct_change(w)*100;f[f"spx_momentum_z_{w}d"]=calculate_robust_zscore(mom,63)
        sr=spx.pct_change()
        for w in [10,21,63]:f[f"spx_realized_vol_{w}d"]=sr.rolling(w).std()*np.sqrt(252)*100
        if "spx_realized_vol_10d" in f and "spx_realized_vol_21d" in f:f["spx_vol_ratio_10_21"]=f["spx_realized_vol_10d"]/f["spx_realized_vol_21d"].replace(0,np.nan)
        if "spx_realized_vol_10d" in f and "spx_realized_vol_63d" in f:f["spx_vol_ratio_10_63"]=f["spx_realized_vol_10d"]/f["spx_realized_vol_63d"].replace(0,np.nan)
        f["spx_skew_21d"]=sr.rolling(21).skew();f["spx_kurt_21d"]=sr.rolling(21).kurt()
        bma,bstd=spx.rolling(bw).mean(),spx.rolling(bw).std();bu,bl=bma+2*bstd,bma-2*bstd;f["bb_position_20d"]=((spx-bl)/(bu-bl).replace(0,np.nan)).clip(0,1);f["bb_width_20d"]=((bu-bl)/bma.replace(0,np.nan))*100
        dlt=spx.diff();gn,ls=dlt.clip(lower=0).rolling(14).mean(),(-dlt.clip(upper=0)).rolling(14).mean();rs=gn/ls.replace(0,np.nan);f["rsi_14"]=100-(100/(1+rs));f["rsi_regime"]=calculate_regime_with_validation(f["rsi_14"],bins=[0,30,70,100],labels=[0,1,2],feature_name="rsi");f["rsi_divergence"]=f["rsi_14"]-f["rsi_14"].rolling(21).mean()
        e12,e26=spx.ewm(span=12).mean(),spx.ewm(span=26).mean();f["macd"]=e12-e26;f["macd_signal"]=f["macd"].ewm(span=9).mean();f["macd_histogram"]=f["macd"]-f["macd_signal"]
        h,l,c=so["High"],so["Low"],so["Close"];pdm,mdm=h.diff(),-l.diff();pdm[pdm<0],mdm[mdm<0]=0,0;tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);atr=tr.rolling(14).mean();pdi,mdi=100*(pdm.rolling(14).mean()/atr),100*(mdm.rolling(14).mean()/atr);dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan);f["adx_14"]=dx.rolling(14).mean();f["trend_strength"]=f["spx_vs_ma200"].abs()
        op,hp,lp,cp=so["Open"],so["High"],so["Low"],so["Close"];f["spx_body_size"]=(cp-op).abs();f["spx_range"]=hp-lp;f["spx_range_pct"]=((hp-lp)/cp.replace(0,np.nan))*100;f["spx_upper_shadow"]=hp-cp.combine(op,max);f["spx_lower_shadow"]=cp.combine(op,min)-lp;f["spx_close_position"]=(cp-lp)/(hp-lp).replace(0,np.nan);f["spx_body_to_range"]=f["spx_body_size"]/f["spx_range"].replace(0,np.nan);f["spx_gap"]=op-cp.shift(1);f["spx_gap_magnitude"]=f["spx_gap"].abs();f["spx_upper_rejection"]=(hp-cp.combine(op,max))/f["spx_range"].replace(0,np.nan);f["spx_lower_rejection"]=(cp.combine(op,min)-lp)/f["spx_range"].replace(0,np.nan);f["spx_range_expansion"]=f["spx_range"]/f["spx_range"].rolling(21).mean().replace(0,np.nan)
        for w in [21,63,126]:f[f"spx_vix_corr_{w}d"]=spx.pct_change().rolling(w).corr(vix.pct_change())
        for w in [10,21]:
            if f"spx_realized_vol_{w}d" in f:rv=f[f"spx_realized_vol_{w}d"];f[f"vix_vs_rv_{w}d"]=vix-rv;f[f"vix_rv_ratio_{w}d"]=vix/rv.replace(0,np.nan)
        return f
    def _build_vvix_features(self,vvix:pd.Series,vix:pd.Series)->pd.DataFrame:
        if vvix is None or vvix.isna().all():return pd.DataFrame(index=vix.index)
        f=pd.DataFrame(index=vix.index);f["vvix"]=vvix
        for w in [5,10,21]:f[f"vvix_ret_{w}d"]=vvix.pct_change(w)*100
        for w in [10,21]:f[f"vvix_velocity_{w}d"]=vvix.diff(w)
        for w in [21,63]:ma=vvix.rolling(w).mean();f[f"vvix_vs_ma{w}"]=((vvix-ma)/ma.replace(0,np.nan))*100
        for w in [21,63]:f[f"vvix_zscore_{w}d"]=calculate_robust_zscore(vvix,63)
        f["vvix_vix_ratio"]=vvix/vix.replace(0,np.nan);f["vvix_vix_spread"]=vvix-vix
        for w in [21,63]:f[f"vvix_percentile_{w}d"]=calculate_percentile_with_validation(vvix,w)
        f["vol_of_vol_regime"]=calculate_regime_with_validation(vvix,bins=[0,80,110,140,300],labels=[0,1,2,3],feature_name="vvix")
        return f
    def _build_cboe_features(self,cb:pd.DataFrame,vix:pd.Series)->pd.DataFrame:
        f=pd.DataFrame(index=vix.index)
        if "SKEW" in cb.columns:
            sk=cb["SKEW"];f["SKEW"]=sk;f["skew_regime"]=calculate_regime_with_validation(sk,bins=SKEW_REGIME_BINS,labels=SKEW_REGIME_LABELS,feature_name="skew");f["skew_vs_vix"]=sk-vix;f["skew_vix_ratio"]=sk/vix.replace(0,np.nan);sma=sk.rolling(63).mean();f["skew_displacement"]=((sk-sma)/sma.replace(0,np.nan))*100
        if "PCCE" in cb.columns and "PCCI" in cb.columns:f["pc_equity_inst_divergence"]=(cb["PCCE"].rolling(63).rank(pct=True)-cb["PCCI"].rolling(63).rank(pct=True)).abs()
        if "PCC" in cb.columns:f["pcc_accel_10d"]=cb["PCC"].diff(10).diff(10)
        for cn in ["COR1M","COR3M"]:
            if cn in cb.columns:cr=cb[cn];f[cn]=cr;f[f"{cn}_change_21d"]=cr.diff(21);f[f"{cn}_zscore_63d"]=calculate_robust_zscore(cr,63)
        if "COR1M" in cb.columns and "COR3M" in cb.columns:f["cor_term_structure"]=cb["COR1M"]-cb["COR3M"];f["cor_term_slope_change_21d"]=f["cor_term_structure"].diff(21)
        if "VXTH" in cb.columns:vx=cb["VXTH"];f["VXTH"]=vx;f["VXTH_change_21d"]=vx.diff(21);f["VXTH_zscore_63d"]=calculate_robust_zscore(vx,63);f["vxth_vix_ratio"]=vx/vix.replace(0,np.nan)
        if "VXTLT" in cb.columns:
            vt=cb["VXTLT"];f["VXTLT"]=vt;f["VXTLT_change_21d"]=vt.diff(21);f["VXTLT_zscore_63d"]=calculate_robust_zscore(vt,63);f["VXTLT_velocity_10d"]=vt.diff(10);f["VXTLT_acceleration_5d"]=vt.diff(5).diff(5);f["bond_vol_regime"]=calculate_regime_with_validation(vt,bins=[0,5,10,15,100],labels=[0,1,2,3],feature_name="vxtlt");f["vxtlt_vix_ratio"]=vt/vix.replace(0,np.nan);f["vxtlt_vix_spread"]=vt-vix
            for w in [63,126,252]:f[f"VXTLT_percentile_{w}d"]=calculate_percentile_with_validation(vt,w)
            for w in [21,63]:ma=vt.rolling(w).mean();f[f"VXTLT_vs_ma{w}"]=((vt-ma)/ma.replace(0,np.nan))*100
            if "spx_realized_vol_21d" in f.columns:f["bond_equity_vol_divergence"]=(vt.rank(pct=True)-f["spx_realized_vol_21d"].rank(pct=True)).abs()
        sc=[]
        if "SKEW" in f.columns:sc.append(((f["SKEW"]-130)/30).clip(0,1))
        if "VXTH" in f.columns:sc.append(((f["VXTH"]-15)/20).clip(0,1))
        if "VXTLT" in f.columns:sc.append(((f["VXTLT"]-8)/15).clip(0,1))
        if sc:f["cboe_stress_composite"]=pd.DataFrame(sc).T.mean(axis=1);f["cboe_stress_regime"]=calculate_regime_with_validation(f["cboe_stress_composite"],bins=[0,0.33,0.66,1],labels=[0,1,2],feature_name="cboe_stress")
        return f
    def _build_futures_features(self,ss:str,es:str,idx:pd.DatetimeIndex,spx:pd.Series,cb:pd.DataFrame)->pd.DataFrame:
        vxd={}
        if cb is not None and "VX1-VX2" in cb.columns:vxd["VX1-VX2"]=cb["VX1-VX2"]
        if cb is not None and "VX2-VX1_RATIO" in cb.columns:vxd["VX2-VX1_RATIO"]=cb["VX2-VX1_RATIO"]
        vxf=self.futures_engine.extract_vix_futures_features(vxd);vxf=vxf.reindex(idx,method="ffill") if not vxf.empty else pd.DataFrame(index=idx)
        comd={}
        if cb is not None and "CL1-CL2" in cb.columns:comd["CL1-CL2"]=cb["CL1-CL2"]
        crd=self.fetcher.fetch_yahoo("CL=F",ss,es)
        if crd is not None:comd["Crude_Oil"]=crd["Close"].squeeze().reindex(idx,method="ffill")
        comf=self.futures_engine.extract_commodity_futures_features(comd);comf=comf.reindex(idx,method="ffill") if not comf.empty else pd.DataFrame(index=idx)
        dold={}
        if cb is not None and "DX1-DX2" in cb.columns:dold["DX1-DX2"]=cb["DX1-DX2"]
        dxy=self.fetcher.fetch_yahoo("DX-Y.NYB",ss,es)
        if dxy is not None:dold["Dollar_Index"]=dxy["Close"].squeeze().reindex(idx,method="ffill")
        dolf=self.futures_engine.extract_dollar_futures_features(dold);dolf=dolf.reindex(idx,method="ffill") if not dolf.empty else pd.DataFrame(index=idx)
        srt=spx.pct_change(21)*100;crf=self.futures_engine.extract_futures_cross_relationships(vxd,comd,dold,srt);crf=crf.reindex(idx,method="ffill") if not crf.empty else pd.DataFrame(index=idx)
        return pd.concat([vxf,comf,dolf,crf],axis=1)
    def _fetch_macro_data(self,ss:str,es:str,idx:pd.DatetimeIndex)->pd.DataFrame:
        fd={}
        frs={"CPI":"CPIAUCSL"}
        for n,sid in frs.items():
            try:d=self.fetcher.fetch_fred_series(sid,ss,es);(fd.update({n:d.reindex(idx,method="ffill",limit=5)}) if d is not None and not d.empty else None)
            except:continue
        yhs={"Gold":"GC=F"}
        for n,sym in yhs.items():
            try:d=self.fetcher.fetch_yahoo(sym,ss,es);(fd.update({n:d["Close"].reindex(idx,method="ffill",limit=5)}) if d is not None and not d.empty and "Close" in d.columns else None)
            except:continue
        return pd.DataFrame(fd,index=idx) if fd else None
    def _build_macro_features(self,m:pd.DataFrame)->pd.DataFrame:
        f=pd.DataFrame(index=m.index)
        if "Dollar_Index" in m.columns:
            dxy=m["Dollar_Index"];f["Dollar_Index_lag1"]=dxy.shift(1);f["Dollar_Index_zscore_63d"]=calculate_robust_zscore(dxy,63)
            for w in [10,21,63]:f[f"dxy_ret_{w}d"]=dxy.pct_change(w)*100
        if "Bond_Vol" in m.columns:bv=m["Bond_Vol"];f["Bond_Vol_lag1"]=bv.shift(1);f["Bond_Vol_zscore_63d"]=calculate_robust_zscore(bv,63);[f.update({f"Bond_Vol_mom_{w}d":bv.diff(w)}) for w in [10,21,63]]
        if "CPI" in m.columns:[f.update({f"CPI_change_{w}d":m["CPI"].diff(w)}) for w in [10,21,63]]
        return f
    def _build_treasury_features(self,ss:str,es:str,idx:pd.DatetimeIndex)->pd.DataFrame:
        yahoo_yields={}
        yahoo_tickers={"^IRX":"3M","^FVX":"5Y","^TNX":"10Y","^TYX":"30Y","2YY=F":"2Y"}
        for ticker,name in yahoo_tickers.items():
            try:d=self.fetcher.fetch_yahoo(ticker,ss,es);(yahoo_yields.update({name:d["Close"].squeeze()}) if d is not None and "Close" in d.columns else None)
            except:continue
        if len(yahoo_yields)>=3:
            ydf=pd.DataFrame(yahoo_yields,index=idx).ffill(limit=5)
            if "2Y" not in ydf.columns:
                print("  2YY=F not available, falling back to FRED DGS2")
                try:dgs2=self.fetcher.fetch_fred_series("DGS2",ss,es);(ydf.update({"2Y":dgs2.reindex(idx,method="ffill",limit=5)}) if dgs2 is not None else None)
                except:pass
            tsp=self.treasury_engine.extract_term_spreads(ydf);csh=self.treasury_engine.extract_curve_shape(ydf);rvol=self.treasury_engine.extract_rate_volatility(ydf);return pd.concat([tsp,csh,rvol],axis=1)
        else:
            ts={"DGS1MO":"1M","DGS3MO":"3M","DGS6MO":"6M","DGS1":"1Y","DGS2":"2Y","DGS5":"5Y","DGS10":"10Y","DGS30":"30Y"}
            fy={}
            for n,sid in ts.items():
                try:d=self.fetcher.fetch_fred_series(sid.replace("_",""),ss,es);(fy.update({n:d.reindex(idx,method="ffill",limit=5)}) if d is not None and not d.empty else None)
                except:continue
            if not fy:return pd.DataFrame(index=idx)
            ydf=pd.DataFrame(fy,index=idx);ydf.columns=[c.replace("DGS","").replace("MO","M") for c in ydf.columns];tsp=self.treasury_engine.extract_term_spreads(ydf);csh=self.treasury_engine.extract_curve_shape(ydf);rvol=self.treasury_engine.extract_rate_volatility(ydf);return pd.concat([tsp,csh,rvol],axis=1)
    def _build_meta_features(self,cmb:pd.DataFrame,spx:pd.Series,vix:pd.Series,md:pd.DataFrame)->pd.DataFrame:
        return pd.concat([self.meta_engine.extract_regime_indicators(cmb,vix,spx),self.meta_engine.extract_cross_asset_relationships(cmb,md),self.meta_engine.extract_rate_of_change_features(cmb),self.meta_engine.extract_percentile_rankings(cmb)],axis=1)
    def _build_calendar_features(self,idx:pd.DatetimeIndex)->pd.DataFrame:
        return pd.DataFrame({"month":idx.month,"day_of_week":idx.dayofweek,"day_of_month":idx.day},index=idx)
    def _ensure_numeric_dtypes(self,df:pd.DataFrame)->pd.DataFrame:
        mc=["calendar_cohort","cohort_weight","feature_quality"];nc=[c for c in df.columns if c not in mc]
        for c in nc:
            if df[c].dtype==object:df[c]=pd.to_numeric(df[c],errors="coerce")
            if df[c].dtype in [np.int32,np.int64,np.float32]:df[c]=df[c].astype(np.float64)
        return df
