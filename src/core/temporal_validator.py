import re,warnings
from datetime import datetime,timedelta
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
from config import FEATURE_QUALITY_CONFIG,PUBLICATION_LAGS
class TemporalSafetyValidator:
    def __init__(self,publication_lags:Dict[str,int]=None):
        self.publication_lags=publication_lags or PUBLICATION_LAGS;self.audit_results={};self.last_update_timestamps={};self.quality_config=FEATURE_QUALITY_CONFIG
        print("🔍 Temporal Validator initialized with feature quality scoring")
    def audit_feature_code(self,feature_engine_path:str="core/feature_engine.py")->Dict:
        print(f"\n{'='*80}\n🔍 TIER 1: STATIC CODE AUDIT\n{'='*80}")
        if not Path(feature_engine_path).exists():return{"error":f"File not found: {feature_engine_path}"}
        with open(feature_engine_path,"r")as f:code=f.read()
        issues=[]
        forward_shifts=re.findall(r"\.shift\s*\(\s*-\s*\d+",code)
        if forward_shifts:issues.append({"severity":"CRITICAL","type":"forward_shift","count":len(forward_shifts),"message":f"Found {len(forward_shifts)} forward .shift(-N) operations","examples":forward_shifts[:5]})
        future_indices=re.findall(r'\[\s*["\'].*future.*["\']\s*\]',code,re.IGNORECASE)
        if future_indices:issues.append({"severity":"HIGH","type":"future_indexing","count":len(future_indices),"message":"Found references to 'future' in column names","examples":future_indices[:3]})
        suspicious_rolling=re.findall(r"\.rolling\([^)]+\)\.apply\([^)]*lambda[^)]*\[[^\]]*\+[^\]]*\]",code)
        if suspicious_rolling:issues.append({"severity":"MEDIUM","type":"suspicious_rolling","count":len(suspicious_rolling),"message":"Found rolling().apply() with forward-looking indexing","examples":suspicious_rolling[:3]})
        direct_future=re.findall(r"=\s*\w+\[\w+\s*\+\s*\d+\]",code)
        if direct_future:issues.append({"severity":"HIGH","type":"direct_future_access","count":len(direct_future),"message":"Found direct future indexing (e.g., series[i+5])","examples":direct_future[:3]})
        self.audit_results={"timestamp":datetime.now().isoformat(),"file":feature_engine_path,"total_issues":len(issues),"issues":issues}
        if issues:
            print(f"\n⚠️ FOUND {len(issues)} POTENTIAL ISSUES:\n")
            for issue in issues:
                print(f"[{issue['severity']}] {issue['type']}: {issue['message']}")
                if "examples"in issue:
                    for ex in issue["examples"]:print(f"   → {ex}")
        else:print("✅ No suspicious patterns detected in feature code")
        print(f"\n{'='*80}")
        return self.audit_results
    def validate_split(self,X_train:pd.DataFrame,X_val:pd.DataFrame,feature_names:List[str])->List[str]:
        violations=[];train_end=X_train.index.max();val_start=X_val.index.min()
        if train_end>=val_start:violations.append(f"TEMPORAL LEAK: Train ends {train_end.date()} >= Val starts {val_start.date()}")
        gap_days=(val_start-train_end).days
        if gap_days>10:violations.append(f"Large gap between train/val ({gap_days} days) - may indicate missing data")
        return violations
    def validate_cv_split(self,X_train:pd.DataFrame,X_val:pd.DataFrame,y_train:pd.Series,y_val:pd.Series,fold_num:int)->bool:
        train_end=X_train.index.max();val_start=X_val.index.min()
        if train_end>=val_start:raise ValueError(f"⚠️ FOLD {fold_num} TEMPORAL LEAK: Train ends {train_end.date()} >= Val starts {val_start.date()}")
        gap_days=(val_start-train_end).days
        if gap_days>10:warnings.warn(f"Fold {fold_num}: Large gap between train/val ({gap_days} days). May indicate missing data.")
        suspicious_features=[];features_to_check=X_train.columns[:50]if len(X_train.columns)>50 else X_train.columns
        for col in features_to_check:
            if X_train[col].isna().mean()>0.5 or X_val[col].isna().mean()>0.5:continue
            try:
                train_last=X_train[col].iloc[-20:].dropna();val_first=X_val[col].iloc[:20].dropna()
                if len(train_last)>5 and len(val_first)>5:
                    combined=pd.concat([train_last,val_first]);autocorr=combined.autocorr(lag=1)
                    if autocorr>0.99:suspicious_features.append({"feature":col,"autocorr":autocorr,"reason":"Unrealistically high autocorrelation across split"})
            except:continue
        if suspicious_features:
            warnings.warn(f"Fold {fold_num}: {len(suspicious_features)} features have suspicious train/val continuity (may indicate leakage)")
            for feat in suspicious_features[:3]:warnings.warn(f"  → {feat['feature']}: autocorr={feat['autocorr']:.4f}")
        try:
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score
            if len(X_train)>100:
                X_check=X_train.iloc[-100:].fillna(0);y_check=y_train.iloc[-100:];model=Ridge(alpha=1.0);model.fit(X_check,y_check)
                X_val_check=X_val.iloc[:50].fillna(0);y_val_check=y_val.iloc[:50]
                if len(X_val_check)>10:
                    preds=model.predict(X_val_check);r2=r2_score(y_val_check,preds)
                    if r2>0.8:warnings.warn(f"Fold {fold_num}: Suspiciously high validation R² = {r2:.3f}. May indicate target leakage.")
        except:pass
        return True
    def verify_publication_lags(self,features:pd.DataFrame,prediction_date:datetime,strict:bool=True)->Tuple[bool,List[str]]:
        if not self.publication_lags:
            warnings.warn("No publication lags configured - skipping validation")
            return True,[]
        violations=[];source_mapping=self._infer_feature_sources(features.columns)
        for feature_col,source in source_mapping.items():
            if source not in self.publication_lags:continue
            lag_days=self.publication_lags[source];latest_allowed_date=prediction_date-timedelta(days=lag_days);feature_data=features[feature_col].dropna()
            if len(feature_data)==0:continue
            latest_feature_date=feature_data.index[-1]
            if latest_feature_date>latest_allowed_date:violations.append(f"{feature_col} (source: {source}): Uses data from {latest_feature_date.date()} but only data up to {latest_allowed_date.date()} should be available (lag={lag_days} days)")
        if violations:
            msg=f"\n⚠️ PUBLICATION LAG VIOLATIONS:\n"+"\n".join(f"  • {v}"for v in violations)
            if strict:raise ValueError(msg)
            else:warnings.warn(msg)
            return False,violations
        return True,[]
    def validate_walk_forward_gap(self,features:pd.DataFrame,prediction_date:datetime,feature_metadata:Dict[str,Dict]=None,strict:bool=True)->Tuple[bool,List[str]]:
        violations=[]
        if feature_metadata is None:
            warnings.warn("No feature metadata provided - cannot validate as-of timestamps. Call build_complete_features() with return_metadata=True")
            return True,[]
        if prediction_date not in features.index:
            warnings.warn(f"Prediction date {prediction_date.date()} not in features index")
            return True,[]
        feature_row=features.loc[prediction_date]
        for feat_name in feature_row.dropna().index:
            if feat_name not in feature_metadata:continue
            meta=feature_metadata[feat_name];source=meta.get("source","unknown");last_available=meta.get("last_available_date")
            if last_available is None:continue
            lag_days=self.publication_lags.get(source,0);effective_cutoff=prediction_date-timedelta(days=lag_days)
            if last_available>effective_cutoff:violations.append(f"{feat_name}: Uses data from {last_available.date()} but cutoff for prediction {prediction_date.date()} is {effective_cutoff.date()} (source: {source}, lag: {lag_days}d)")
        if violations:
            msg=f"\n⚠️ WALK-FORWARD GAP VIOLATIONS at {prediction_date.date()}:\n"+"\n".join(f"  • {v}"for v in violations)
            if strict:raise ValueError(msg)
            else:warnings.warn(msg)
            return False,violations
        return True,[]
    def test_feature_availability_at_prediction_time(self,features:pd.DataFrame,feature_metadata:Dict[str,Dict],test_dates:List[datetime]=None,sample_size:int=20)->Dict:
        print(f"\n{'='*80}\n🧪 WALK-FORWARD GAP AUTOMATED TEST\n{'='*80}")
        if test_dates is None:
            all_dates=features.index;mid_point=len(all_dates)//2
            test_dates=np.random.choice(all_dates[mid_point:],size=min(sample_size,len(all_dates)-mid_point),replace=False)
        results={"total_tests":len(test_dates),"passed":0,"failed":0,"violations":[]}
        for test_date in test_dates:
            is_valid,violations=self.validate_walk_forward_gap(features,test_date,feature_metadata,strict=False)
            if is_valid:results["passed"]+=1
            else:
                results["failed"]+=1
                results["violations"].append({"date":test_date.date().isoformat(),"violations":violations})
        print(f"\nTest Results:\n  ✅ Passed: {results['passed']}/{results['total_tests']}\n  ❌ Failed: {results['failed']}/{results['total_tests']}")
        if results["failed"]>0:
            print(f"\n⚠️ Found {results['failed']} dates with temporal violations\nFirst violation example:")
            if results["violations"]:
                first=results["violations"][0]
                print(f"  Date: {first['date']}")
                for v in first["violations"][:3]:print(f"    → {v}")
        else:print("\n✅ All walk-forward gap tests passed!")
        print(f"{'='*80}\n")
        return results
    def compute_feature_quality(self,feature_dict:dict,date:pd.Timestamp=None)->float:
        if date is None:date=pd.Timestamp.now()
        scores=[];critical_features=self.quality_config["missingness_penalty"]["critical_features"]
        for feat in critical_features:
            if feat in feature_dict:
                scores.append(0.0 if pd.isna(feature_dict[feat])or feature_dict[feat]is None else 1.0)
            else:scores.append(0.0)
        important_features=self.quality_config["missingness_penalty"]["important_features"]
        for feat in important_features:
            if feat in feature_dict:
                if pd.isna(feature_dict[feat])or feature_dict[feat]is None:scores.append(0.5)
                else:
                    staleness_score=self._compute_staleness_score(feat,date)
                    scores.append(staleness_score)
            else:scores.append(0.5)
        optional_features=self.quality_config["missingness_penalty"]["optional_features"]
        for feat in optional_features:
            if feat in feature_dict:
                if pd.isna(feature_dict[feat])or feature_dict[feat]is None:scores.append(0.9)
                else:
                    staleness_score=self._compute_staleness_score(feat,date)
                    scores.append(staleness_score)
            else:scores.append(0.9)
        if len(scores)==0:return 1.0
        quality_score=np.mean(scores)
        return np.clip(quality_score,0.0,1.0)
    def _compute_staleness_score(self,feature_name:str,date:pd.Timestamp)->float:
        if feature_name not in self.last_update_timestamps:return 1.0
        last_update=self.last_update_timestamps[feature_name];days_stale=(date-last_update).days;expected_lag=self.publication_lags.get(feature_name,1);staleness_config=self.quality_config["staleness_penalty"]
        if days_stale<=expected_lag:return staleness_config["none"]
        elif days_stale<=expected_lag+3:return staleness_config["minor"]
        elif days_stale<=expected_lag+7:return staleness_config["moderate"]
        elif days_stale<=expected_lag+14:return staleness_config["severe"]
        else:return staleness_config["critical"]
    def compute_feature_quality_batch(self,df:pd.DataFrame)->pd.Series:
        quality_scores=[]
        for date,row in df.iterrows():
            feature_dict=row.to_dict();quality=self.compute_feature_quality(feature_dict,date)
            quality_scores.append(quality)
        return pd.Series(quality_scores,index=df.index)
    def update_feature_timestamp(self,feature_name:str,timestamp:pd.Timestamp=None):
        if timestamp is None:timestamp=pd.Timestamp.now()
        self.last_update_timestamps[feature_name]=timestamp
    def get_feature_age(self,feature_name:str,current_date:pd.Timestamp=None)->int:
        if current_date is None:current_date=pd.Timestamp.now()
        if feature_name not in self.last_update_timestamps:return 0
        last_update=self.last_update_timestamps[feature_name]
        return(current_date-last_update).days
    def check_quality_threshold(self,quality_score:float,strict:bool=False)->tuple:
        thresholds=self.quality_config["quality_thresholds"];min_threshold=thresholds["acceptable"]if not strict else thresholds["good"]
        if quality_score>=thresholds["excellent"]:return(True,"Excellent data quality")
        elif quality_score>=thresholds["good"]:return(True,"Good data quality")
        elif quality_score>=min_threshold:return(True,"Acceptable data quality (degraded forecast)")
        elif quality_score>=thresholds["poor"]:return(False,"Poor data quality - critical features missing or stale")
        else:return(False,"Unusable data quality - refuse to forecast")
    def get_quality_report(self,feature_dict:dict,date:pd.Timestamp=None)->dict:
        if date is None:date=pd.Timestamp.now()
        report={"overall_quality":self.compute_feature_quality(feature_dict,date),"date":str(date),"critical_features":{},"important_features":{},"optional_features":{},"issues":[]}
        for feat in self.quality_config["missingness_penalty"]["critical_features"]:
            if feat in feature_dict:
                is_missing=pd.isna(feature_dict[feat])or feature_dict[feat]is None;age=self.get_feature_age(feat,date)
                report["critical_features"][feat]={"present":not is_missing,"age_days":age,"expected_lag":self.publication_lags.get(feat,1)}
                if is_missing:report["issues"].append(f"CRITICAL: {feat} is missing")
            else:
                report["critical_features"][feat]={"present":False}
                report["issues"].append(f"CRITICAL: {feat} not in feature set")
        for feat in self.quality_config["missingness_penalty"]["important_features"]:
            if feat in feature_dict:
                is_missing=pd.isna(feature_dict[feat])or feature_dict[feat]is None;age=self.get_feature_age(feat,date)
                report["important_features"][feat]={"present":not is_missing,"age_days":age,"expected_lag":self.publication_lags.get(feat,1)}
                if is_missing:report["issues"].append(f"Important: {feat} is missing")
                elif age>self.publication_lags.get(feat,1)+7:report["issues"].append(f"Important: {feat} is stale ({age} days)")
        optional_count=len(self.quality_config["missingness_penalty"]["optional_features"])
        optional_present=sum(1 for feat in self.quality_config["missingness_penalty"]["optional_features"]if feat in feature_dict and not pd.isna(feature_dict.get(feat)))
        report["optional_features"]["coverage"]=f"{optional_present}/{optional_count}"
        return report
    def _infer_feature_sources(self,feature_names:List[str])->Dict[str,str]:
        source_map={};patterns={r"^vix":"^VIX",r"^spx":"^GSPC",r"SKEW":"SKEW",r"VXTLT":"VXTLT",r"VX[12]":"VX1-VX2",r"^yield|^dgs":"DGS10",r"crude|^cl":"CL=F",r"dxy|dollar":"DX-Y.NYB"}
        for feat in feature_names:
            feat_lower=feat.lower()
            for pattern,source in patterns.items():
                if re.search(pattern,feat_lower):
                    source_map[feat]=source;break
        return source_map
    def generate_validation_report(self,output_path:str="./models/temporal_safety_report.json",include_metadata_check:bool=False,feature_metadata:Dict=None):
        import json
        report={"timestamp":datetime.now().isoformat(),"validation_tiers":{"tier1_code_audit":self.audit_results,"tier2_cv_validation":"Run during training","tier3_publication_lags":f"{len(self.publication_lags)} sources configured","tier4_walk_forward_gap":"Automated test available via test_feature_availability_at_prediction_time()"},"publication_lags":self.publication_lags,"feature_quality":{"enabled":True,"tracked_features":len(self.last_update_timestamps),"quality_thresholds":self.quality_config["quality_thresholds"]}}
        if include_metadata_check and feature_metadata:
            report["feature_metadata_check"]={"total_features":len(feature_metadata),"features_with_timestamps":sum(1 for m in feature_metadata.values()if m.get("last_available_date")is not None),"coverage_pct":round(100*sum(1 for m in feature_metadata.values()if m.get("last_available_date")is not None)/max(len(feature_metadata),1),1)}
        Path(output_path).parent.mkdir(exist_ok=True,parents=True)
        with open(output_path,"w")as f:json.dump(report,f,indent=2)
        print(f"\n✅ Validation report saved: {output_path}")
        return report
def run_full_validation(feature_engine_path:str="core/feature_engine.py",publication_lags:Dict[str,int]=None)->TemporalSafetyValidator:
    validator=TemporalSafetyValidator(publication_lags);audit_results=validator.audit_feature_code(feature_engine_path)
    critical_issues=[issue for issue in audit_results.get("issues",[])if issue["severity"]=="CRITICAL"]
    if critical_issues:
        print("\n❌ CRITICAL ISSUES FOUND - Fix before training!")
        return validator
    print("\n✅ Tier 1 validation passed")
    validator.generate_validation_report()
    return validator
