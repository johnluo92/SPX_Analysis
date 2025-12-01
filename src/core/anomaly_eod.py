import logging,pickle,json
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

logger=logging.getLogger(__name__)

class EODAnomalyScorer:
    def __init__(self,contamination=0.05,n_estimators=200,random_state=42):
        self.contamination=contamination;self.n_estimators=n_estimators;self.random_state=random_state
        self.detector=None;self.scaler=None;self.feature_names=None
        self.training_distribution=None;self.statistical_thresholds=None;self.feature_importance=None;self.trained=False

    def train(self,features,feature_subset=None):
        logger.info("🎯 TRAINING EOD ANOMALY DETECTOR")
        if feature_subset is None:feature_subset=self._auto_select_spike_features(features)
        logger.info(f"Selected {len(feature_subset)} features")
        valid_features=self._validate_features(features,feature_subset)
        logger.info(f"Valid: {len(valid_features)}")
        if len(valid_features)<10:raise ValueError(f"Need ≥10 features, got {len(valid_features)}")
        self.feature_names=valid_features
        X=features[valid_features].fillna(0)
        self.scaler=RobustScaler();X_scaled=self.scaler.fit_transform(X)
        logger.info(f"Training IsolationForest (contamination={self.contamination})...")
        self.detector=IsolationForest(contamination=self.contamination,n_estimators=self.n_estimators,max_samples='auto',random_state=self.random_state,n_jobs=-1)
        self.detector.fit(X_scaled)
        training_scores=self.detector.score_samples(X_scaled)
        self.training_distribution=training_scores
        self.statistical_thresholds=self._calculate_thresholds(training_scores)
        logger.info(f"Thresholds: M={self.statistical_thresholds['moderate']:.4f} H={self.statistical_thresholds['high']:.4f} C={self.statistical_thresholds['critical']:.4f}")
        self.feature_importance=self._calculate_feature_importance(X_scaled,training_scores)
        self.trained=True
        return {'n_samples':len(X),'n_features':len(valid_features),'thresholds':self.statistical_thresholds}

    def calculate_score(self,features):
        if not self.trained:raise ValueError("Not trained")
        X=features[self.feature_names].fillna(0);X_scaled=self.scaler.transform(X)
        raw_score=self.detector.score_samples(X_scaled)[0]
        percentile_score=self._raw_to_percentile(raw_score);level=self._classify_level(percentile_score)
        return {'anomaly_score':float(percentile_score),'raw_score':float(raw_score),'level':level,'percentile':float(percentile_score*100)}

    def calculate_scores_batch(self,features):
        if not self.trained:raise ValueError("Not trained")
        logger.info(f"Calculating scores for {len(features)} observations...")
        X=features[self.feature_names].fillna(0);X_scaled=self.scaler.transform(X)
        raw_scores=self.detector.score_samples(X_scaled)
        percentile_scores=np.array([self._raw_to_percentile(s)for s in raw_scores])
        levels=[self._classify_level(s)for s in percentile_scores]
        result=pd.DataFrame({'anomaly_score':percentile_scores,'anomaly_level':levels},index=features.index)
        logger.info(f"Mean={percentile_scores.mean():.4f} Max={percentile_scores.max():.4f}")
        return result

    def _auto_select_spike_features(self,features):
        keywords=['velocity','accel','jerk','ratio','stress','divergence','transition','regime','percentile_velocity']
        selected=[col for col in features.columns if any(kw in col.lower()for kw in keywords)]
        core=['vix','vix_velocity_5d','vix_accel_5d','vix_zscore_252d','vix_percentile_21d','spx_realized_vol_21d','vix_vs_rv_21d']
        for f in core:
            if f in features.columns and f not in selected:selected.append(f)
        return selected

    def _validate_features(self,features,feature_list):
        valid=[]
        for feat in feature_list:
            if feat not in features.columns:continue
            series=features[feat]
            # Handle potential DataFrame return (duplicate columns)
            if isinstance(series,pd.DataFrame):
                series=series.iloc[:,0]
            # Use explicit float conversion for pandas safety
            na_ratio=float(series.isna().sum())/float(len(series))
            if na_ratio>0.80:continue
            std_val=float(series.std())
            if std_val<1e-6:continue
            zero_ratio=float((series==0).sum())/float(len(series))
            if zero_ratio>0.95:continue
            valid.append(feat)
        return valid

    def _calculate_thresholds(self,scores):
        return {'moderate':float(np.percentile(scores,75)),'high':float(np.percentile(scores,90)),'critical':float(np.percentile(scores,95))}

    def _raw_to_percentile(self,raw_score):
        percentile=(self.training_distribution<=raw_score).sum()/len(self.training_distribution)
        return float(np.clip(1.0-percentile,0.0,0.95))

    def _classify_level(self,score):
        if score>=self.statistical_thresholds['critical']:return 'CRITICAL'
        elif score>=self.statistical_thresholds['high']:return 'HIGH'
        elif score>=self.statistical_thresholds['moderate']:return 'MODERATE'
        else:return 'NORMAL'

    def _calculate_feature_importance(self,X_scaled,baseline_scores):
        logger.info("Calculating feature importance...")
        importances={};baseline_mean=np.mean(baseline_scores)
        n_samples=min(500,len(X_scaled))
        sample_indices=np.random.RandomState(self.random_state).choice(len(X_scaled),n_samples,replace=False)
        X_sample=X_scaled[sample_indices].copy()
        for i,feature_name in enumerate(self.feature_names):
            original_col=X_sample[:,i].copy()
            np.random.RandomState(self.random_state+i).shuffle(X_sample[:,i])
            permuted_scores=self.detector.score_samples(X_sample)
            importances[feature_name]=abs(baseline_mean-np.mean(permuted_scores))
            X_sample[:,i]=original_col
        total=sum(importances.values())
        if total>0:importances={k:v/total for k,v in importances.items()}
        sorted_importance=dict(sorted(importances.items(),key=lambda x:x[1],reverse=True))
        logger.info("Top 10:")
        for i,(feat,imp)in enumerate(list(sorted_importance.items())[:10],1):
            logger.info(f"  {i:2d}. {feat:40s} {imp:.4f}")
        return sorted_importance

    def save(self,save_dir="models"):
        save_path=Path(save_dir);save_path.mkdir(parents=True,exist_ok=True)
        with open(save_path/"anomaly_detector_eod.pkl",'wb')as f:pickle.dump(self.detector,f)
        with open(save_path/"anomaly_scaler_eod.pkl",'wb')as f:pickle.dump(self.scaler,f)
        metadata={'feature_names':self.feature_names,'thresholds':self.statistical_thresholds,'contamination':self.contamination,'n_estimators':self.n_estimators,'trained_on':datetime.now().isoformat(),'feature_importance':{k:float(v)for k,v in self.feature_importance.items()}}
        with open(save_path/"anomaly_metadata_eod.json",'w')as f:json.dump(metadata,f,indent=2)
        logger.info(f"✅ Saved to {save_dir}/")

    def load(self,load_dir="models"):
        load_path=Path(load_dir)
        with open(load_path/"anomaly_detector_eod.pkl",'rb')as f:self.detector=pickle.load(f)
        with open(load_path/"anomaly_scaler_eod.pkl",'rb')as f:self.scaler=pickle.load(f)
        with open(load_path/"anomaly_metadata_eod.json",'r')as f:metadata=json.load(f)
        self.feature_names=metadata['feature_names'];self.statistical_thresholds=metadata['thresholds']
        self.contamination=metadata['contamination'];self.n_estimators=metadata['n_estimators']
        self.feature_importance=metadata['feature_importance']
        self.training_distribution=np.linspace(metadata['thresholds']['moderate'],metadata['thresholds']['critical'],1000)
        self.trained=True;logger.info(f"✅ Loaded from {load_dir}/")
