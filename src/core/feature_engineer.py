import warnings
from datetime import datetime,timedelta
from typing import Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
from config import TRAINING_YEARS,CALENDAR_COHORTS,COHORT_PRIORITY,ENABLE_TEMPORAL_SAFETY,PUBLICATION_LAGS,MACRO_EVENT_CONFIG,FORWARD_FILL_LIMITS
from core.calculations import calculate_robust_zscore,calculate_regime_with_validation,calculate_percentile_with_validation,SKEW_REGIME_BINS,SKEW_REGIME_LABELS
from core.temporal_validator import TemporalSafetyValidator
from core.regime_classifier import RegimeClassifier
from core.vx_futures_engineer import VXFuturesEngineer
warnings.filterwarnings("ignore")
class LaborMarketFeatureEngine:
    @staticmethod
    def extract_initial_claims_features(icsa:pd.Series)->pd.DataFrame:
        f=pd.DataFrame(index=icsa.index);f["initial_claims"]=icsa;f["claims_4wk_ma"]=icsa.rolling(20).mean();f["claims_velocity_4w"]=icsa.diff(20);f["claims_zscore_52w"]=calculate_robust_zscore(icsa,252);f["claims_percentile_52w"]=calculate_percentile_with_validation(icsa,252);f["claims_regime"]=calculate_regime_with_validation(icsa,bins=[0,250000,350000,500000,1000000],labels=[0,1,2,3],feature_name="claims");f["claims_velocity_zscore"]=calculate_robust_zscore(f["claims_velocity_4w"],63);return f
class FinancialStressFeatureEngine:
    @staticmethod
    def extract_stress_index_features(stlfsi:pd.Series)->pd.DataFrame:
        f=pd.DataFrame(index=stlfsi.index);f["fin_stress_index"]=stlfsi;f["stress_velocity_4w"]=stlfsi.diff(20);f["stress_zscore_52w"]=calculate_robust_zscore(stlfsi,252);f["stress_extreme"]=(stlfsi>stlfsi.rolling(252).quantile(0.9)).astype(int);f["stress_regime"]=calculate_regime_with_validation(stlfsi,bins=[-5,-1,0,2,10],labels=[0,1,2,3],feature_name="stress");return f
class CreditSpreadFeatureEngine:
    @staticmethod
    def extract_credit_spread_features(spreads:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        if"HY_OAS_All"in spreads:hy=spreads["HY_OAS_All"];f["hy_spread"]=hy;f["hy_velocity_21d"]=hy.diff(21);f["hy_zscore_252d"]=calculate_robust_zscore(hy,252)
        if"HY_OAS_BB"in spreads:bb=spreads["HY_OAS_BB"];f["hy_bb_spread"]=bb;f["hy_bb_velocity_21d"]=bb.diff(21)
        if"HY_OAS_B"in spreads:b=spreads["HY_OAS_B"];f["hy_b_spread"]=b;f["hy_b_velocity_21d"]=b.diff(21)
        if"HY_OAS_CCC"in spreads:ccc=spreads["HY_OAS_CCC"];f["hy_ccc_spread"]=ccc;f["hy_ccc_velocity_21d"]=ccc.diff(21)
        if"HY_OAS_BB"in spreads and"HY_OAS_CCC"in spreads:f["credit_quality_spread"]=spreads["HY_OAS_CCC"]-spreads["HY_OAS_BB"];f["credit_widening_regime"]=(f["credit_quality_spread"]>f["credit_quality_spread"].rolling(63).quantile(0.8)).astype(int)
        if"IG_OAS"in spreads and"HY_OAS_All"in spreads:f["ig_hy_spread"]=spreads["HY_OAS_All"]-spreads["IG_OAS"];f["credit_stress_composite"]=calculate_percentile_with_validation(f["ig_hy_spread"],252)
        return f
class VIXTermStructureEngine:
    @staticmethod
    def extract_vix_term_features(vix:pd.Series,vix3m:pd.Series,vix6m:pd.Series)->pd.DataFrame:
        f=pd.DataFrame(index=vix.index)
        if vix3m is not None and not vix3m.isna().all():
            f["vix1m_vix3m_ratio"]=vix/vix3m.replace(0,np.nan)
            f["vix_term_slope_short"]=(vix3m-vix)/90
            f["vix_term_slope_velocity_21d"]=f["vix_term_slope_short"].diff(21)
            f["vix_term_ratio_zscore_63d"]=calculate_robust_zscore(f["vix1m_vix3m_ratio"],63)
        if vix6m is not None and not vix6m.isna().all():
            if vix3m is not None and not vix3m.isna().all():
                f["vix3m_vix6m_ratio"]=vix3m/vix6m.replace(0,np.nan)
                f["vix_term_slope_long"]=(vix6m-vix3m)/90
                f["vix_term_curvature"]=2*vix3m-vix-vix6m
                f["vix_term_curvature_zscore"]=calculate_robust_zscore(f["vix_term_curvature"],63)
        return f
class InteractionFeatureEngine:
    @staticmethod
    def build_vix_spx_interactions(vix_features:Dict[str,pd.Series],spx_features:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        if"vix_velocity_5d"in vix_features and"spx_realized_vol_63d"in spx_features:
            vv,srv=vix_features["vix_velocity_5d"],spx_features["spx_realized_vol_63d"]
            f["vix_vel_spx_rv_ratio"]=vv/srv.replace(0,np.nan)
            f["vix_spx_velocity_asymmetry"]=(vv.abs().rolling(21).mean()-srv.diff(5).abs().rolling(21).mean()).abs()
        if"vix_velocity_5d"in vix_features and"spx_ret_5d"in spx_features:
            f["vix_vel_spx_ret_ratio"]=vix_features["vix_velocity_5d"].abs()/spx_features["spx_ret_5d"].abs().replace(0,np.nan)
        if"vix_zscore_252d"in vix_features and"spx_realized_vol_63d"in spx_features:
            vz,srv=vix_features["vix_zscore_252d"],spx_features["spx_realized_vol_63d"]
            f["vix_z_spx_rv_product"]=vz*srv
        return f
    @staticmethod
    def build_vol_clustering_features(vix_vol:pd.Series,spx_vol:pd.Series,bond_vol:pd.Series=None)->pd.DataFrame:
        f=pd.DataFrame()
        if vix_vol is not None and spx_vol is not None:
            vv_z,sv_z=calculate_robust_zscore(vix_vol,63),calculate_robust_zscore(spx_vol,63)
            f["cross_asset_vol_cluster"]=(vv_z>1)&(sv_z>1).astype(int)
            f["vol_correlation_breakage"]=(vv_z.rolling(21).corr(sv_z)<-0.3).astype(int)
        if bond_vol is not None and spx_vol is not None:
            bv_z,sv_z=calculate_robust_zscore(bond_vol,63),calculate_robust_zscore(spx_vol,63)
            f["bond_equity_vol_sync"]=(bv_z*sv_z).clip(-3,3)
        return f
    @staticmethod
    def build_credit_vol_stress(credit_spread:pd.Series,vix:pd.Series)->pd.DataFrame:
        f=pd.DataFrame()
        if credit_spread is not None and vix is not None:
            cs_z,vix_z=calculate_robust_zscore(credit_spread,252),calculate_robust_zscore(vix,63)
            f["credit_vol_stress_composite"]=(cs_z*vix_z).clip(-4,4)
            f["credit_vol_divergence"]=(cs_z.rolling(63).rank(pct=True)-vix_z.rolling(63).rank(pct=True)).abs()
        return f
class RegimeConditionalEngine:
    @staticmethod
    def build_regime_conditional_features(features:Dict[str,pd.Series],regime:pd.Series)->pd.DataFrame:
        f=pd.DataFrame()
        regime_dummies=pd.get_dummies(regime,prefix="regime")
        for feat_name,feat_series in features.items():
            if feat_series is None:continue
            for col in regime_dummies.columns:f[f"{feat_name}_x_{col}"]=feat_series*regime_dummies[col]
        return f
    @staticmethod
    def build_regime_adjusted_momentum(mom_features:Dict[str,pd.Series],regime_stats:pd.DataFrame)->pd.DataFrame:
        f=pd.DataFrame()
        for feat_name,mom_series in mom_features.items():
            if mom_series is None or "regime_expected_return_5d"not in regime_stats.columns:continue
            expected_ret=regime_stats["regime_expected_return_5d"]
            f[f"{feat_name}_regime_adj"]=mom_series-expected_ret
        return f
class TransformationEngine:
    @staticmethod
    def add_log_transforms(features:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        log_candidates=["vix_vol_21d","spx_realized_vol_21d","crude_oil_vol_21d","dxy_vol_21d"]
        for name in log_candidates:
            if name in features and features[name] is not None:
                series=features[name];f[f"{name}_log"]=np.log1p(series.clip(lower=0))
        return f
    @staticmethod
    def add_binary_extremes(features:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        extreme_candidates=["vix_velocity_5d","vix_accel_5d","spx_gap","spx_gap_magnitude"]
        for name in extreme_candidates:
            if name in features and features[name] is not None:
                series=features[name];p95=series.rolling(252).quantile(0.95);f[f"{name}_extreme_flag"]=(series.abs()>p95).astype(int)
        return f
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
    def extract_futures_cross_relationships(vxf:pd.DataFrame,cd:Dict[str,pd.Series],dd:Dict[str,pd.Series],sr:pd.Series=None)->pd.DataFrame:
        f=pd.DataFrame()
        if "vx1_vx2_spread" in vxf.columns and "CL1-CL2" in cd:vs,cls=vxf["vx1_vx2_spread"],cd["CL1-CL2"];f["vx_crude_corr_21d"]=vs.rolling(21).corr(cls);f["vx_crude_divergence"]=(vs.rolling(63).rank(pct=True)-cls.rolling(63).rank(pct=True)).abs()
        if "vx1_vx2_spread" in vxf.columns and "Dollar_Index" in dd:f["vx_dollar_corr_21d"]=vxf["vx1_vx2_spread"].rolling(21).corr(dd["Dollar_Index"].pct_change(21)*100)
        if "Dollar_Index" in dd and "Crude_Oil" in cd:f["dollar_crude_corr_21d"]=dd["Dollar_Index"].pct_change().rolling(21).corr(cd["Crude_Oil"].pct_change())
        if sr is not None:
            if "vx1_vx2_spread" in vxf.columns:f["spx_vx_spread_corr_21d"]=sr.rolling(21).corr(vxf["vx1_vx2_spread"])
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
        self.fetcher=data_fetcher;self.meta_engine=MetaFeatureEngine();self.futures_engine=FuturesFeatureEngine();self.treasury_engine=TreasuryYieldFeatureEngine();self.labor_engine=LaborMarketFeatureEngine();self.stress_engine=FinancialStressFeatureEngine();self.credit_engine=CreditSpreadFeatureEngine();self.vx_engineer=VXFuturesEngineer();self.vix_term_engine=VIXTermStructureEngine();self.interaction_engine=InteractionFeatureEngine();self.regime_conditional_engine=RegimeConditionalEngine();self.transformation_engine=TransformationEngine();self.validator=TemporalSafetyValidator();self.regime_classifier=RegimeClassifier();self.fomc_calendar=None;self.opex_calendar=None;self.earnings_calendar=None;self.vix_futures_expiry=None;self._cohort_cache={}
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
    def _apply_publication_lags(self,data:Dict[str,pd.Series],idx:pd.DatetimeIndex)->Dict[str,pd.Series]:
        lagged={}
        for name,series in data.items():
            lag=PUBLICATION_LAGS.get(name,0)
            if lag>0:lagged[name]=series.shift(lag).reindex(idx)
            else:lagged[name]=series.reindex(idx)
        return lagged
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
    def build_complete_features(self,years:int=TRAINING_YEARS,end_date:Optional[str]=None,force_historical:bool=False)->dict:
        print(f"\n{'='*80}");print(f"🏗️  Building {years}y feature set | Temporal Safety: {'ON' if ENABLE_TEMPORAL_SAFETY else 'OFF'}");print(f"{'='*80}")
        if end_date is None:ed=datetime.now();mode="LIVE"
        else:
            ed=pd.Timestamp(end_date)
            if force_historical:mode="HISTORICAL"
            else:ir=ed>(datetime.now()-timedelta(days=7));mode="RECENT" if ir else"HISTORICAL"
        sd=ed-timedelta(days=years*365+450);self.training_start_date=sd;self.training_end_date=ed;ss=sd.strftime("%Y-%m-%d");es=ed.strftime("%Y-%m-%d")
        print(f"Mode: {mode}");print(f"Date Ranges: Warmup Start -> Warmup End (usable period) -> Training End Date: {ss} → {(sd+timedelta(days=450)).strftime('%Y-%m-%d')} → {es}")
        spx_df=self.fetcher.fetch_yahoo("^GSPC",ss,es);vix=self.fetcher.fetch_yahoo("^VIX",ss,es);vvix=self.fetcher.fetch_yahoo("^VVIX",ss,es)
        vix_term_df=self.fetcher.fetch_vix_term(["^VIX","^VIX3M","^VIX6M"],ss,es)
        if spx_df is None or vix is None:raise ValueError("❌ Core data fetch failed")
        spx=spx_df["Close"].squeeze();vix=vix["Close"].squeeze();ff_daily=FORWARD_FILL_LIMITS.get("daily",5);vix=vix.reindex(spx.index,method="ffill",limit=ff_daily);spx_ohlc=spx_df.reindex(spx.index,method="ffill",limit=ff_daily)
        vix3m=vix_term_df["VIX3M"].reindex(spx.index,method="ffill",limit=ff_daily)if vix_term_df is not None and "VIX3M" in vix_term_df.columns else None
        vix6m=vix_term_df["VIX6M"].reindex(spx.index,method="ffill",limit=ff_daily)if vix_term_df is not None and "VIX6M" in vix_term_df.columns else None
        if vvix is not None:vvix_series=vvix["Close"].squeeze().reindex(spx.index,method="ffill",limit=ff_daily)
        else:vvix_series=None
        cbd,cboe_meta=self.fetcher.fetch_all_cboe(return_metadata=True)
        if cbd:
            cb=pd.DataFrame(index=spx.index)
            for s,ser in cbd.items():
                cb[s]=ser.reindex(spx.index,method="ffill",limit=ff_daily)
                if s in cboe_meta:meta=cboe_meta[s];cb[f"{s}_quality_flag"]=1.0 if meta["data_quality"]=="excellent"else(0.8 if meta["data_quality"]=="good"else(0.5 if meta["data_quality"]=="acceptable"else 0.2))
        else:cb=pd.DataFrame(index=spx.index)
        bf=self._build_base_features(spx,vix,spx_ohlc,cb,vix3m,vix6m);cbf=self._build_cboe_features(cb,vix) if not cb.empty else pd.DataFrame(index=spx.index);ff=self._build_futures_features(ss,es,spx.index,spx,cb);md=self._fetch_macro_data(ss,es,spx.index);mf=self._build_macro_features(md) if md is not None else pd.DataFrame(index=spx.index);ef=self._build_economic_features(md);tf=self._build_treasury_features(ss,es,spx.index);vvf=self._build_vvix_features(vvix_series,vix) if vvix_series is not None else pd.DataFrame(index=spx.index);cmb=pd.concat([bf,cbf],axis=1);mtf=self._build_meta_features(cmb,spx,vix,md);calf=self._build_calendar_features(spx.index)
        interaction_vix={"vix_velocity_5d":bf.get("vix_velocity_5d"),"vix_zscore_252d":bf.get("vix_zscore_252d")}
        interaction_spx={"spx_realized_vol_63d":bf.get("spx_realized_vol_63d"),"spx_ret_5d":bf.get("spx_ret_5d"),"spx_vol_ratio_10_63":bf.get("spx_vol_ratio_10_63")}
        intf=self.interaction_engine.build_vix_spx_interactions(interaction_vix,interaction_spx)
        vol_cluster_f=self.interaction_engine.build_vol_clustering_features(bf.get("vix_vol_21d"),bf.get("spx_realized_vol_21d"),vvf.get("VXTLT") if not vvf.empty else None)
        credit_vol_f=self.interaction_engine.build_credit_vol_stress(ef.get("hy_spread") if not ef.empty else None,vix)
        regime_feat_dict={"vix_velocity_5d":bf.get("vix_velocity_5d"),"vix_zscore_252d":bf.get("vix_zscore_252d"),"spx_realized_vol_63d":bf.get("spx_realized_vol_63d")}
        regime_cond_f=self.regime_conditional_engine.build_regime_conditional_features(regime_feat_dict,bf.get("vix_regime"))if"vix_regime"in bf.columns else pd.DataFrame()
        mom_feats={"spx_ret_5d":bf.get("spx_ret_5d"),"vix_velocity_5d":bf.get("vix_velocity_5d")}
        regime_adj_mom=self.regime_conditional_engine.build_regime_adjusted_momentum(mom_feats,bf)if all(c in bf.columns for c in ["regime_expected_return_5d"])else pd.DataFrame()
        transform_feats={"vix_vol_21d":bf.get("vix_vol_21d"),"spx_realized_vol_21d":bf.get("spx_realized_vol_21d"),"crude_oil_vol_21d":ff.get("crude_oil_vol_21d")if not ff.empty else None,"dxy_vol_21d":ff.get("dxy_vol_21d")if not ff.empty else None,"vix_velocity_5d":bf.get("vix_velocity_5d"),"vix_accel_5d":bf.get("vix_accel_5d"),"spx_gap":bf.get("spx_gap"),"spx_gap_magnitude":bf.get("spx_gap_magnitude")}
        log_transf=self.transformation_engine.add_log_transforms(transform_feats)
        binary_transf=self.transformation_engine.add_binary_extremes(transform_feats)
        af=pd.concat([bf,cbf,ff,mf,ef,tf,vvf,mtf,intf,vol_cluster_f,credit_vol_f,regime_cond_f,regime_adj_mom,log_transf,binary_transf,calf],axis=1);af=af.loc[:,~af.columns.duplicated()];af=self._ensure_numeric_dtypes(af)
        self._load_calendar_data();cohd=[{"calendar_cohort":self.get_calendar_cohort(dt)[0],"cohort_weight":self.get_calendar_cohort(dt)[1]} for dt in af.index];cohdf=pd.DataFrame(cohd,index=af.index);cohdf["is_fomc_period"]=(cohdf["calendar_cohort"]=="fomc_period").astype(int);cohdf["is_opex_week"]=(cohdf["calendar_cohort"]=="opex_week").astype(int);cohdf["is_earnings_heavy"]=(cohdf["calendar_cohort"]=="earnings_heavy").astype(int);af=pd.concat([af,cohdf],axis=1)
        af["feature_quality"]=self.validator.compute_feature_quality_batch(af);af=self.apply_quality_control(af)
        if ENABLE_TEMPORAL_SAFETY and not cb.empty:self._validate_term_structure_timing(vix,cb)
        print(f"\n✅ Complete: {len(af.columns)} features | {len(af)} rows");print(f"   Date range: {af.index[0].date()} → {af.index[-1].date()}");print(f"{'='*80}\n");return {"features":af,"spx":spx,"vix":vix,"cboe_data":cb if cbd else None,"vvix":vvix_series}
    def _build_base_features(self,spx:pd.Series,vix:pd.Series,so:pd.DataFrame,cb:pd.DataFrame=None,vix3m:pd.Series=None,vix6m:pd.Series=None)->pd.DataFrame:
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
        regime_features=self.regime_classifier.compute_all_regime_features(vix)
        for col in regime_features.columns:
            if col=="vix_regime_numeric":continue
            if col=="days_in_regime"and col in f.columns:continue
            if col not in f.columns:f[col]=regime_features[col]
        if vix3m is not None or vix6m is not None:
            vix_term_f=self.vix_term_engine.extract_vix_term_features(vix,vix3m,vix6m)
            for col in vix_term_f.columns:f[col]=vix_term_f[col]
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
        ff_daily=FORWARD_FILL_LIMITS.get("daily",5)
        try:vxf=self.vx_engineer.build_all_vx_features(start_date=ss,end_date=es,target_index=idx);vxf=vxf.reindex(idx,method="ffill",limit=ff_daily)
        except Exception as e:warnings.warn(f"VX engineer failed: {e}. Using fallback.");vxd={};(vxd.update({"VX1-VX2":cb["VX1-VX2"]}) if cb is not None and "VX1-VX2" in cb.columns else None);(vxd.update({"VX2-VX1_RATIO":cb["VX2-VX1_RATIO"]}) if cb is not None and "VX2-VX1_RATIO" in cb.columns else None);vxf=self._legacy_vix_futures_features(vxd);vxf=vxf.reindex(idx,method="ffill",limit=ff_daily) if not vxf.empty else pd.DataFrame(index=idx)
        comd={}
        if cb is not None and "CL1-CL2" in cb.columns:comd["CL1-CL2"]=cb["CL1-CL2"]
        crd=self.fetcher.fetch_yahoo("CL=F",ss,es)
        if crd is not None:comd["Crude_Oil"]=crd["Close"].squeeze().reindex(idx,method="ffill",limit=ff_daily)
        comf=self.futures_engine.extract_commodity_futures_features(comd);comf=comf.reindex(idx,method="ffill",limit=ff_daily) if not comf.empty else pd.DataFrame(index=idx)
        dold={}
        if cb is not None and "DX1-DX2" in cb.columns:dold["DX1-DX2"]=cb["DX1-DX2"]
        dxy=self.fetcher.fetch_yahoo("DX-Y.NYB",ss,es)
        if dxy is not None:dold["Dollar_Index"]=dxy["Close"].squeeze().reindex(idx,method="ffill",limit=ff_daily)
        dolf=self.futures_engine.extract_dollar_futures_features(dold);dolf=dolf.reindex(idx,method="ffill",limit=ff_daily) if not dolf.empty else pd.DataFrame(index=idx)
        srt=spx.pct_change(21)*100;crf=self.futures_engine.extract_futures_cross_relationships(vxf,comd,dold,srt);crf=crf.reindex(idx,method="ffill",limit=ff_daily) if not crf.empty else pd.DataFrame(index=idx)
        return pd.concat([vxf,comf,dolf,crf],axis=1)
    def _legacy_vix_futures_features(self,vxd:Dict[str,pd.Series])->pd.DataFrame:
        f=pd.DataFrame()
        if "VX1-VX2" in vxd:sp=vxd["VX1-VX2"];f["VX1-VX2"]=sp;f["VX1-VX2_change_21d"]=sp.diff(21);f["VX1-VX2_zscore_63d"]=calculate_robust_zscore(sp,63);f["VX1-VX2_percentile_63d"]=calculate_percentile_with_validation(sp,63)
        if "VX2-VX1_RATIO" in vxd:r=vxd["VX2-VX1_RATIO"];f["VX2-VX1_RATIO"]=r;f["VX2-VX1_RATIO_velocity_10d"]=r.diff(10);f["vx_term_structure_regime"]=calculate_regime_with_validation(r,bins=[-1,-0.05,0,0.05,1],labels=[0,1,2,3],feature_name="vx_ratio")
        if "VX1-VX2" in vxd and "VX2-VX1_RATIO" in vxd:sp,r=vxd["VX1-VX2"],vxd["VX2-VX1_RATIO"];f["vx_curve_acceleration"]=r.diff(5).diff(5);f["vx_term_structure_divergence"]=(sp.rolling(63).rank(pct=True)-r.rolling(63).rank(pct=True)).abs()
        return f
    def _fetch_macro_data(self,ss:str,es:str,idx:pd.DatetimeIndex)->pd.DataFrame:
        fd={};ff_monthly=FORWARD_FILL_LIMITS.get("monthly",45);ff_weekly=FORWARD_FILL_LIMITS.get("weekly",10);ff_daily=FORWARD_FILL_LIMITS.get("daily",5)
        frs={"CPI":"CPIAUCSL","Initial_Claims":"ICSA","STL_Fin_Stress":"STLFSI4","Fed_Funds":"DFF","HY_OAS_All":"BAMLH0A0HYM2","HY_OAS_BB":"BAMLH0A1HYBB","HY_OAS_B":"BAMLH0A2HYB","HY_OAS_CCC":"BAMLH0A3HYC","IG_OAS":"BAMLC0A0CM"}
        for n,sid in frs.items():
            try:
                d=self.fetcher.fetch_fred_series(sid,ss,es)
                if d is not None and not d.empty:
                    lag=PUBLICATION_LAGS.get(sid,0)
                    if lag>0:d=d.shift(lag)
                    if sid in ["CPIAUCSL"]:ff_limit=ff_monthly
                    elif sid in ["ICSA","STLFSI4"]:ff_limit=ff_weekly
                    else:ff_limit=ff_daily
                    fd[n]=d.reindex(idx,method="ffill",limit=ff_limit)
            except:continue
        yhs={"Gold":"GC=F"}
        for n,sym in yhs.items():
            try:
                d=self.fetcher.fetch_yahoo(sym,ss,es)
                if d is not None and not d.empty and "Close" in d.columns:fd[n]=d["Close"].reindex(idx,method="ffill",limit=ff_daily)
            except:continue
        df=pd.DataFrame(fd,index=idx) if fd else None
        return df
    def _build_macro_features(self,m:pd.DataFrame)->pd.DataFrame:
        f=pd.DataFrame(index=m.index)
        if "Dollar_Index" in m.columns:
            dxy=m["Dollar_Index"];f["Dollar_Index_lag1"]=dxy.shift(1);f["Dollar_Index_zscore_63d"]=calculate_robust_zscore(dxy,63)
            for w in [10,21,63]:f[f"dxy_ret_{w}d"]=dxy.pct_change(w)*100
        if "Bond_Vol" in m.columns:bv=m["Bond_Vol"];f["Bond_Vol_lag1"]=bv.shift(1);f["Bond_Vol_zscore_63d"]=calculate_robust_zscore(bv,63);[f.update({f"Bond_Vol_mom_{w}d":bv.diff(w)}) for w in [10,21,63]]
        if "CPI" in m.columns:[f.update({f"CPI_change_{w}d":m["CPI"].diff(w)}) for w in [10,21,63]]
        return f
    def _build_economic_features(self,m:pd.DataFrame)->pd.DataFrame:
        if m is None:return pd.DataFrame()
        idx=m.index;labor_feats=self.labor_engine.extract_initial_claims_features(m["Initial_Claims"]) if "Initial_Claims" in m.columns else pd.DataFrame(index=idx)
        stress_feats=self.stress_engine.extract_stress_index_features(m["STL_Fin_Stress"]) if "STL_Fin_Stress" in m.columns else pd.DataFrame(index=idx)
        spread_dict={k:m[k] for k in ["HY_OAS_All","HY_OAS_BB","HY_OAS_B","HY_OAS_CCC","IG_OAS"] if k in m.columns}
        credit_feats=self.credit_engine.extract_credit_spread_features(spread_dict) if spread_dict else pd.DataFrame(index=idx)
        return pd.concat([labor_feats,stress_feats,credit_feats],axis=1)
    def _build_treasury_features(self,ss:str,es:str,idx:pd.DatetimeIndex)->pd.DataFrame:
        ff_daily=FORWARD_FILL_LIMITS.get("daily",5);yahoo_yields={};yahoo_tickers={"^IRX":"3M","^FVX":"5Y","^TNX":"10Y","^TYX":"30Y","2YY=F":"2Y"}
        for ticker,name in yahoo_tickers.items():
            try:d=self.fetcher.fetch_yahoo(ticker,ss,es);(yahoo_yields.update({name:d["Close"].squeeze()}) if d is not None and "Close" in d.columns else None)
            except:continue
        fred_yields={};ts={"DGS1MO":"1M","DGS3MO":"3M","DGS6MO":"6M","DGS1":"1Y","DGS2":"2Y","DGS5":"5Y","DGS10":"10Y","DGS30":"30Y"}
        for sid,n in ts.items():
            try:
                d=self.fetcher.fetch_fred_series(sid,ss,es)
                if d is not None and not d.empty:
                    lag=PUBLICATION_LAGS.get(sid,1)
                    if lag>0:d=d.shift(lag)
                    fred_yields[sid]=d
            except:continue
        if len(yahoo_yields)>=3:
            ydf=pd.DataFrame(yahoo_yields,index=idx).ffill(limit=ff_daily)
            if "2Y" in ydf.columns:
                yahoo_2y_coverage = ydf["2Y"].notna().sum() / len(ydf) if len(ydf) > 0 else 0
                if yahoo_2y_coverage < 0.5 and "DGS2" in fred_yields:ydf["2Y"] = fred_yields["DGS2"].reindex(idx, method="ffill", limit=ff_daily);print(f"⚠️  Replaced sparse Yahoo 2Y ({yahoo_2y_coverage:.1%} coverage) with FRED DGS2")
            else:
                if "DGS2" in fred_yields:ydf["2Y"] = fred_yields["DGS2"].reindex(idx, method="ffill", limit=ff_daily);print("ℹ️  Yahoo 2Y unavailable, using FRED DGS2")
            tsp_yahoo=self.treasury_engine.extract_term_spreads(ydf)
            csh_yahoo=self.treasury_engine.extract_curve_shape(ydf);rvol_yahoo=self.treasury_engine.extract_rate_volatility(ydf);yahoo_feats=pd.concat([tsp_yahoo,csh_yahoo,rvol_yahoo],axis=1)
        else:yahoo_feats=pd.DataFrame(index=idx)
        if fred_yields:
            fdf=pd.DataFrame({k.replace("DGS","").replace("MO","M"):v for k,v in fred_yields.items()},index=idx).ffill(limit=ff_daily)
            tsp_fred=self.treasury_engine.extract_term_spreads(fdf);csh_fred=self.treasury_engine.extract_curve_shape(fdf);rvol_fred=self.treasury_engine.extract_rate_volatility(fdf);fred_feats=pd.concat([tsp_fred,csh_fred,rvol_fred],axis=1);fred_feats.columns=[f"{col}_fred" for col in fred_feats.columns]
        else:fred_feats=pd.DataFrame(index=idx)
        return pd.concat([yahoo_feats,fred_feats],axis=1) if not yahoo_feats.empty or not fred_feats.empty else pd.DataFrame(index=idx)
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
