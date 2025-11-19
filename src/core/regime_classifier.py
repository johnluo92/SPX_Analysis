import numpy as np
import pandas as pd
from typing import Union,List
from config import REGIME_BOUNDARIES,REGIME_NAMES
class RegimeClassifier:
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
        numeric=self.classify_vix_regime_numeric(vix)
        return self._string_labels[numeric]
    def classify_vix_series_numeric(self,vix_series):
        return pd.cut(vix_series,bins=self.vix_boundaries,labels=list(range(len(self.vix_boundaries)-1)),include_lowest=True).astype(int)
    def classify_vix_series_string(self,vix_series):
        numeric=self.classify_vix_series_numeric(vix_series)
        return numeric.map(self._string_labels)
    def get_regime_name(self,regime):
        if isinstance(regime,str):
            reverse_map={v:k for k,v in self._string_labels.items()}
            if regime not in reverse_map:raise ValueError(f"Unknown regime: {regime}")
            regime=reverse_map[regime]
        return self.vix_labels[regime]
    def get_regime_boundaries_dict(self):
        boundaries={}
        for i in range(len(self.vix_boundaries)-1):boundaries[i]=(self.vix_boundaries[i],self.vix_boundaries[i+1])
        return boundaries
    def describe_regimes(self):
        data=[];boundaries=self.get_regime_boundaries_dict()
        for regime_num in sorted(boundaries.keys()):
            lower,upper=boundaries[regime_num]
            data.append({"regime_numeric":regime_num,"regime_string":self._string_labels[regime_num],"regime_name":self.vix_labels[regime_num],"vix_min":lower,"vix_max":upper})
        return pd.DataFrame(data)
_classifier=RegimeClassifier()
def classify_vix_regime(vix,numeric=False):
    if isinstance(vix,pd.Series):
        if numeric:return _classifier.classify_vix_series_numeric(vix)
        else:return _classifier.classify_vix_series_string(vix)
    else:
        if numeric:return _classifier.classify_vix_regime_numeric(vix)
        else:return _classifier.classify_vix_regime_string(vix)
def get_regime_info():return _classifier.describe_regimes()