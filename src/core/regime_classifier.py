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
    def __init__(self):
        self.vix_boundaries=REGIME_BOUNDARIES;self.vix_labels=REGIME_NAMES
        if len(self.vix_boundaries)!=5:raise ValueError(f"REGIME_BOUNDARIES must have 5 values, got {len(self.vix_boundaries)}")
        self._string_labels={i:label.lower().replace(" ","_")for i,label in self.vix_labels.items()}
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
    def compute_regime_distance_to_boundary(self,vix):
        return vix.apply(self._distance_to_boundary_scalar)if isinstance(vix,pd.Series)else self._distance_to_boundary_scalar(vix)
    def _distance_to_boundary_scalar(self,vix):
        distances=[abs(vix-boundary)for boundary in self.vix_boundaries];return min(distances)
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
    def compute_all_regime_features(self,vix_series):
        features=pd.DataFrame(index=vix_series.index);features['vix_regime_numeric']=self.classify_vix_series_numeric(vix_series);features['regime_distance_to_boundary']=self.compute_regime_distance_to_boundary(vix_series);features['regime_within_regime_percentile']=self.compute_within_regime_percentile(vix_series);persistence_df=self.compute_regime_persistence_features(vix_series);features=pd.concat([features,persistence_df],axis=1);stats_df=self.compute_regime_statistical_features(vix_series);features=pd.concat([features,stats_df],axis=1);transition_df=self.compute_regime_transition_features(vix_series);features=pd.concat([features,transition_df],axis=1);return features
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
def get_regime_info():return _classifier.describe_regimes()
def compute_regime_features(vix_series):return _classifier.compute_all_regime_features(vix_series)
