import numpy as np
import pandas as pd
from typing import Union,Dict,Tuple
from config import REGIME_BOUNDARIES,REGIME_NAMES

class RegimeClassifier:
    REGIME_TRANSITION_PROBS={0:{'crisis':0.0002,'elevated':0.0049,'low_vol':0.8726,'normal':0.1222},1:{'crisis':0.0022,'elevated':0.1027,'low_vol':0.1587,'normal':0.7365},2:{'crisis':0.0340,'elevated':0.7320,'low_vol':0.0026,'normal':0.2314},3:{'crisis':0.7209,'elevated':0.2791,'low_vol':0.0000,'normal':0.0000}}
    REGIME_MEAN_REVERSION={0:{'mean_return':2.331,'median_return':0.990,'std_dev':12.391,'prob_up':0.541},1:{'mean_return':-0.813,'median_return':-1.750,'std_dev':13.158,'prob_up':0.435},2:{'mean_return':-3.475,'median_return':-3.787,'std_dev':14.130,'prob_up':0.380},3:{'mean_return':-6.487,'median_return':-7.605,'std_dev':16.984,'prob_up':0.335}}
    REGIME_VOL_OF_VOL={0:{'mean':5.95,'median':5.29},1:{'mean':6.53,'median':5.84},2:{'mean':7.44,'median':6.62},3:{'mean':10.36,'median':8.89}}
    REGIME_PERSISTENCE={0:{'median_duration':3,'mean_duration':16},1:{'median_duration':2,'mean_duration':8},2:{'median_duration':2,'mean_duration':7},3:{'median_duration':2,'mean_duration':6}}
    REGIME_SPIKE_PROB={0:0.0700,1:0.0578,2:0.0437,3:0.0605}
    REGIME_PERCENTILES={0:{'min':9.14,'p25':12.27,'p75':15.19,'max':16.76},1:{'min':16.77,'p25':18.32,'p75':21.98,'max':24.40},2:{'min':24.41,'p25':25.96,'p75':31.30,'max':39.66},3:{'min':39.68,'p25':42.26,'p75':55.20,'max':82.69}}
    SKEW_BOUNDARIES=[0,114.68,119.61,127.27,200]
    SKEW_REGIME_NAMES={0:'Complacent',1:'Normal',2:'Elevated Tail',3:'Extreme Tail'}
    SKEW_MEAN_REVERSION={0:{'mean_5d_return':0.878,'median':0.576,'std':2.662,'pct_positive':60.9,'behavior':'MOMENTUM'},1:{'mean_5d_return':0.521,'median':0.346,'std':2.985,'pct_positive':55.1,'behavior':'MOMENTUM'},2:{'mean_5d_return':-0.255,'median':-0.508,'std':3.820,'pct_positive':44.4,'behavior':'NEUTRAL'},3:{'mean_5d_return':-1.110,'median':-1.101,'std':5.307,'pct_positive':41.6,'behavior':'MEAN_REVERSION'}}
    SKEW_PERSISTENCE={0:{'median_duration':2,'mean_duration':7},1:{'median_duration':2,'mean_duration':3},2:{'median_duration':2,'mean_duration':4},3:{'median_duration':2,'mean_duration':9}}
    SKEW_TRANSITION_PROBS={0:{0:0.7702,1:0.1963,2:0.0281,3:0.0054},1:{0:0.1924,1:0.5310,2:0.2566,3:0.0200},2:{0:0.0307,1:0.2567,2:0.5611,3:0.1515},3:{0:0.0063,1:0.0165,2:0.1523,3:0.8249}}
    JOINT_REGIME_STATS={0:{0:{'mean':-2.13,'std':9.87,'freq':0.0626},1:{'mean':-2.36,'std':11.99,'freq':0.0862},2:{'mean':-2.61,'std':12.51,'freq':0.0983},3:{'mean':-3.41,'std':13.34,'freq':0.1146}},1:{0:{'mean':0.10,'std':10.01,'freq':0.1152},1:{'mean':1.13,'std':11.98,'freq':0.0956},2:{'mean':0.86,'std':12.84,'freq':0.0955},3:{'mean':-0.17,'std':16.86,'freq':0.1024}},2:{0:{'mean':1.74,'std':11.51,'freq':0.0490},1:{'mean':2.61,'std':12.52,'freq':0.0507},2:{'mean':1.07,'std':13.92,'freq':0.0367},3:{'mean':7.64,'std':17.51,'freq':0.0253}},3:{0:{'mean':3.90,'std':14.09,'freq':0.0228},1:{'mean':6.52,'std':13.49,'freq':0.0176},2:{'mean':2.11,'std':17.52,'freq':0.0196},3:{'mean':14.19,'std':20.86,'freq':0.0078}}}
    JOINT_REGIME_CORR={0:{0:-0.069,1:0.089,2:0.073,3:0.077},1:{0:0.007,1:0.097,2:0.146,3:0.092},2:{0:0.269,1:0.172,2:-0.030,3:0.022},3:{0:0.273,1:0.347,2:0.288,3:-0.066}}
    VIX_MEAN=19.82
    VIX_STD=8.39
    SKEW_MEAN=122.65
    SKEW_STD=11.32

    def __init__(self):
        self.vix_boundaries=REGIME_BOUNDARIES;self.vix_labels=REGIME_NAMES
        if len(self.vix_boundaries)!=5:raise ValueError(f"REGIME_BOUNDARIES must have 5 values, got {len(self.vix_boundaries)}")
        self._string_labels={i:label.lower().replace(" ","_")for i,label in self.vix_labels.items()}
        self._skew_string_labels={i:label.lower().replace(" ","_")for i,label in self.SKEW_REGIME_NAMES.items()}

    def classify_vix_regime_numeric(self,vix):
        bins=self.vix_boundaries
        if vix<bins[1]:return 0
        elif vix<bins[2]:return 1
        elif vix<bins[3]:return 2
        else:return 3

    def classify_vix_regime_string(self,vix):
        numeric=self.classify_vix_regime_numeric(vix);return self._string_labels[numeric]

    def classify_vix_series_numeric(self,vix_series):
        return pd.cut(vix_series,bins=self.vix_boundaries,labels=list(range(len(self.vix_boundaries)-1)),include_lowest=True).astype(int)

    def classify_vix_series_string(self,vix_series):
        numeric=self.classify_vix_series_numeric(vix_series);return numeric.map(self._string_labels)

    def classify_skew_regime_numeric(self,skew):
        bins=self.SKEW_BOUNDARIES
        if skew<bins[1]:return 0
        elif skew<bins[2]:return 1
        elif skew<bins[3]:return 2
        else:return 3

    def classify_skew_series_numeric(self,skew_series):
        return pd.cut(skew_series,bins=self.SKEW_BOUNDARIES,labels=list(range(len(self.SKEW_BOUNDARIES)-1)),include_lowest=True).astype(int)

    def compute_regime_distance_to_boundary(self,vix):
        return vix.apply(self._distance_to_boundary_scalar)if isinstance(vix,pd.Series)else self._distance_to_boundary_scalar(vix)

    def _distance_to_boundary_scalar(self,vix):
        distances=[abs(vix-boundary)for boundary in self.vix_boundaries];return min(distances)

    def compute_skew_distance_to_boundary(self,skew):
        return skew.apply(self._skew_distance_to_boundary_scalar)if isinstance(skew,pd.Series)else self._skew_distance_to_boundary_scalar(skew)

    def _skew_distance_to_boundary_scalar(self,skew):
        distances=[abs(skew-boundary)for boundary in self.SKEW_BOUNDARIES];return min(distances)

    def compute_within_regime_percentile(self,vix):
        return vix.apply(self._within_regime_percentile_scalar)if isinstance(vix,pd.Series)else self._within_regime_percentile_scalar(vix)

    def _within_regime_percentile_scalar(self,vix):
        regime=self.classify_vix_regime_numeric(vix);bounds=self.REGIME_PERCENTILES[regime];regime_range=bounds['max']-bounds['min']
        if regime_range==0:return 0.5
        percentile=(vix-bounds['min'])/regime_range;return np.clip(percentile,0.0,1.0)

    def compute_regime_persistence_features(self,vix_series):
        regimes=self.classify_vix_series_numeric(vix_series);regime_changes=regimes.diff().fillna(0)!=0;regime_groups=regime_changes.cumsum();days_in_regime=regime_groups.groupby(regime_groups).cumcount()+1;regime_change_flag=regime_changes.astype(int)
        def normalize_persistence(row):
            regime=int(row['regime']);days=row['days_in_regime'];median_persistence=self.REGIME_PERSISTENCE[regime]['median_duration'];return days/max(median_persistence,1)
        df=pd.DataFrame({'regime':regimes,'days_in_regime':days_in_regime,'regime_change_flag':regime_change_flag},index=vix_series.index);df['time_in_regime_pct']=df.apply(normalize_persistence,axis=1);return df[['days_in_regime','regime_change_flag','time_in_regime_pct']]

    def compute_regime_statistical_features(self,vix):
        if isinstance(vix,pd.Series):
            regimes=self.classify_vix_series_numeric(vix);features=pd.DataFrame(index=vix.index);features['regime_expected_return_5d']=regimes.map(lambda r:self.REGIME_MEAN_REVERSION[r]['mean_return']);features['regime_median_return_5d']=regimes.map(lambda r:self.REGIME_MEAN_REVERSION[r]['median_return']);features['regime_return_std_5d']=regimes.map(lambda r:self.REGIME_MEAN_REVERSION[r]['std_dev']);features['regime_prob_up_5d']=regimes.map(lambda r:self.REGIME_MEAN_REVERSION[r]['prob_up']);features['regime_vol_of_vol']=regimes.map(lambda r:self.REGIME_VOL_OF_VOL[r]['mean']);features['regime_spike_prob']=regimes.map(lambda r:self.REGIME_SPIKE_PROB[r]);return features
        else:
            regime=self.classify_vix_regime_numeric(vix);return {'regime_expected_return_5d':self.REGIME_MEAN_REVERSION[regime]['mean_return'],'regime_median_return_5d':self.REGIME_MEAN_REVERSION[regime]['median_return'],'regime_return_std_5d':self.REGIME_MEAN_REVERSION[regime]['std_dev'],'regime_prob_up_5d':self.REGIME_MEAN_REVERSION[regime]['prob_up'],'regime_vol_of_vol':self.REGIME_VOL_OF_VOL[regime]['mean'],'regime_spike_prob':self.REGIME_SPIKE_PROB[regime]}

    def compute_regime_transition_features(self,vix):
        if isinstance(vix,pd.Series):
            regimes=self.classify_vix_series_numeric(vix);features=pd.DataFrame(index=vix.index);features['regime_prob_stay']=regimes.map(lambda r:self.REGIME_TRANSITION_PROBS[r][self._string_labels[r]]);features['regime_prob_crisis']=regimes.map(lambda r:self.REGIME_TRANSITION_PROBS[r]['crisis']);features['regime_prob_elevated']=regimes.map(lambda r:self.REGIME_TRANSITION_PROBS[r]['elevated']);return features
        else:
            regime=self.classify_vix_regime_numeric(vix);regime_str=self._string_labels[regime];return {'regime_prob_stay':self.REGIME_TRANSITION_PROBS[regime][regime_str],'regime_prob_crisis':self.REGIME_TRANSITION_PROBS[regime]['crisis'],'regime_prob_elevated':self.REGIME_TRANSITION_PROBS[regime]['elevated']}

    def compute_skew_regime_features(self,skew_series,vix_series=None):
        f=pd.DataFrame(index=skew_series.index);sr=self.classify_skew_series_numeric(skew_series);f['skew_regime_numeric']=sr;f['skew_complacency_flag']=(skew_series<114.68).astype(int);f['skew_extreme_tail_flag']=(skew_series>127.27).astype(int);f['skew_expected_5d_return']=sr.map(lambda r:self.SKEW_MEAN_REVERSION[r]['mean_5d_return']);f['skew_regime_behavior']=sr.map(lambda r:self.SKEW_MEAN_REVERSION[r]['behavior']);f['skew_distance_to_boundary']=self.compute_skew_distance_to_boundary(skew_series)
        sc=sr.diff().fillna(0)!=0;sg=sc.cumsum();f['days_in_skew_regime']=sg.groupby(sg).cumcount()+1;f['skew_regime_change_flag']=sc.astype(int)
        sm63=skew_series.rolling(63,min_periods=1).mean();f['skew_regime_displacement']=(skew_series-sm63)/sm63.replace(0,np.nan)*100;f['skew_percentile_63d']=skew_series.rolling(63,min_periods=1).rank(pct=True);f['skew_percentile_252d']=skew_series.rolling(252,min_periods=1).rank(pct=True)
        if vix_series is not None:
            f['skew_vix_ratio']=skew_series/vix_series.replace(0,np.nan);f['skew_vix_ratio_percentile_252d']=f['skew_vix_ratio'].rolling(252,min_periods=1).rank(pct=True);f['skew_vix_ratio_velocity_10d']=f['skew_vix_ratio'].diff(10)
            sv_bins=[0,5.226,8.968,100];f['skew_vix_ratio_regime']=pd.cut(f['skew_vix_ratio'],bins=sv_bins,labels=[0,1,2],include_lowest=True).astype(float)
        for sr_num in range(4):
            mask=sr==sr_num
            if mask.sum()>0:
                sr_skew=skew_series[mask];sr_mean=sr_skew.mean();sr_std=sr_skew.std()
                if sr_std>0:f.loc[mask,'skew_within_regime_zscore']=(skew_series[mask]-sr_mean)/sr_std
                else:f.loc[mask,'skew_within_regime_zscore']=0
        return f

    def compute_joint_regime_features(self,vix_series,skew_series):
        f=pd.DataFrame(index=vix_series.index);vr=self.classify_vix_series_numeric(vix_series);sr=self.classify_skew_series_numeric(skew_series);f['vix_skew_joint_regime']=vr*4+sr
        jc=f['vix_skew_joint_regime'].diff().fillna(0)!=0;jg=jc.cumsum();f['days_in_joint_regime']=jg.groupby(jg).cumcount()+1;f['joint_regime_change_flag']=jc.astype(int)
        def lookup_joint_stats(row):
            vr,sr=int(row['vix_regime']),int(row['skew_regime'])
            if vr in self.JOINT_REGIME_STATS and sr in self.JOINT_REGIME_STATS[vr]:
                stats=self.JOINT_REGIME_STATS[vr][sr];return stats['mean'],stats['std'],stats['freq']
            return 0,0,0
        def lookup_joint_corr(row):
            vr,sr=int(row['vix_regime']),int(row['skew_regime'])
            if vr in self.JOINT_REGIME_CORR and sr in self.JOINT_REGIME_CORR[vr]:return self.JOINT_REGIME_CORR[vr][sr]
            return 0
        temp_df=pd.DataFrame({'vix_regime':vr,'skew_regime':sr});joint_stats=temp_df.apply(lookup_joint_stats,axis=1,result_type='expand');f['joint_regime_expected_return'],f['joint_regime_volatility'],f['joint_regime_frequency']=joint_stats[0],joint_stats[1],joint_stats[2];f['joint_regime_correlation']=temp_df.apply(lookup_joint_corr,axis=1)
        return f

    def compute_divergence_features(self,vix_series,skew_series):
        f=pd.DataFrame(index=vix_series.index);vn=(vix_series-self.VIX_MEAN)/self.VIX_STD;sn=(skew_series-self.SKEW_MEAN)/self.SKEW_STD;f['vix_skew_divergence']=(vn-sn).abs();f['vix_skew_divergence_zscore_63d']=(f['vix_skew_divergence']-f['vix_skew_divergence'].rolling(63,min_periods=1).mean())/f['vix_skew_divergence'].rolling(63,min_periods=1).std().replace(0,np.nan);f['vix_skew_divergence_extreme']=(f['vix_skew_divergence']>f['vix_skew_divergence'].rolling(252,min_periods=1).quantile(0.85)).astype(int);f['vix_skew_synchronized']=(f['vix_skew_divergence']<f['vix_skew_divergence'].rolling(252,min_periods=1).quantile(0.25)).astype(int);return f

    def compute_comovement_features(self,vix_series,skew_series):
        f=pd.DataFrame(index=vix_series.index);vv5=vix_series.diff(5);sv5=skew_series.diff(5);f['vix_skew_velocity_corr_21d']=vv5.rolling(21,min_periods=1).corr(sv5);f['vix_skew_level_corr_21d']=vix_series.rolling(21,min_periods=1).corr(skew_series);f['vix_skew_correlation_breakdown']=(f['vix_skew_velocity_corr_21d']<f['vix_skew_velocity_corr_21d'].rolling(252,min_periods=1).quantile(0.10)).astype(int)
        cr_bins=[-1,-0.5,0.5,1];f['vix_skew_correlation_regime']=pd.cut(f['vix_skew_velocity_corr_21d'],bins=cr_bins,labels=[0,1,2],include_lowest=True).astype(float);return f

    def compute_boundary_proximity_features(self,vix_series,skew_series):
        f=pd.DataFrame(index=vix_series.index);f['vix_boundary_distance']=self.compute_regime_distance_to_boundary(vix_series);f['skew_boundary_distance']=self.compute_skew_distance_to_boundary(skew_series);f['both_near_boundary_flag']=((f['vix_boundary_distance']<2)&(f['skew_boundary_distance']<5)).astype(int);return f

    def compute_regime_transition_direction(self,vix_series,skew_series):
        f=pd.DataFrame(index=vix_series.index);vr=self.classify_vix_series_numeric(vix_series);sr=self.classify_skew_series_numeric(skew_series);f['vix_regime_transition_direction']=vr.diff().fillna(0).apply(lambda x:1 if x>0 else(-1 if x<0 else 0));f['skew_regime_transition_direction']=sr.diff().fillna(0).apply(lambda x:1 if x>0 else(-1 if x<0 else 0));f['both_regimes_changed_flag']=((vr.diff().fillna(0)!=0)&(sr.diff().fillna(0)!=0)).astype(int);return f

    def compute_all_regime_features(self,vix_series):
        features=pd.DataFrame(index=vix_series.index);features['vix_regime_numeric']=self.classify_vix_series_numeric(vix_series);features['regime_distance_to_boundary']=self.compute_regime_distance_to_boundary(vix_series);features['regime_within_regime_percentile']=self.compute_within_regime_percentile(vix_series);persistence_df=self.compute_regime_persistence_features(vix_series);features=pd.concat([features,persistence_df],axis=1);stats_df=self.compute_regime_statistical_features(vix_series);features=pd.concat([features,stats_df],axis=1);transition_df=self.compute_regime_transition_features(vix_series);features=pd.concat([features,transition_df],axis=1);return features

    def compute_all_skew_vix_features(self,vix_series,skew_series):
        f=pd.DataFrame(index=vix_series.index);vr=self.compute_all_regime_features(vix_series);sr=self.compute_skew_regime_features(skew_series,vix_series);jr=self.compute_joint_regime_features(vix_series,skew_series);dv=self.compute_divergence_features(vix_series,skew_series);cm=self.compute_comovement_features(vix_series,skew_series);bp=self.compute_boundary_proximity_features(vix_series,skew_series);td=self.compute_regime_transition_direction(vix_series,skew_series);return pd.concat([vr,sr,jr,dv,cm,bp,td],axis=1)

    def get_regime_name(self,regime):
        if isinstance(regime,str):
            reverse_map={v:k for k,v in self._string_labels.items()}
            if regime not in reverse_map:raise ValueError(f"Unknown regime: {regime}")
            regime=reverse_map[regime]
        return self.vix_labels[regime]

    def get_regime_boundaries_dict(self):
        boundaries={};
        for i in range(len(self.vix_boundaries)-1):boundaries[i]=(self.vix_boundaries[i],self.vix_boundaries[i+1])
        return boundaries

    def describe_regimes(self):
        data=[];boundaries=self.get_regime_boundaries_dict()
        for regime_num in sorted(boundaries.keys()):
            lower,upper=boundaries[regime_num];stats=self.REGIME_MEAN_REVERSION[regime_num];data.append({'regime_numeric':regime_num,'regime_string':self._string_labels[regime_num],'regime_name':self.vix_labels[regime_num],'vix_min':lower,'vix_max':upper,'mean_5d_return':stats['mean_return'],'prob_up':stats['prob_up'],'median_duration_days':self.REGIME_PERSISTENCE[regime_num]['median_duration']})
        return pd.DataFrame(data)

_classifier=RegimeClassifier()
def classify_vix_regime(vix,numeric=False):
    if isinstance(vix,pd.Series):return _classifier.classify_vix_series_numeric(vix)if numeric else _classifier.classify_vix_series_string(vix)
    else:return _classifier.classify_vix_regime_numeric(vix)if numeric else _classifier.classify_vix_regime_string(vix)
def classify_skew_regime(skew,numeric=True):
    if isinstance(skew,pd.Series):return _classifier.classify_skew_series_numeric(skew)
    else:return _classifier.classify_skew_regime_numeric(skew)
def get_regime_info():return _classifier.describe_regimes()
def compute_regime_features(vix_series):return _classifier.compute_all_regime_features(vix_series)
def compute_skew_vix_features(vix_series,skew_series):return _classifier.compute_all_skew_vix_features(vix_series,skew_series)
