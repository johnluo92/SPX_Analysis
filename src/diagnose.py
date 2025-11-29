#!/usr/bin/env python3
"""
VIX Forecaster Enterprise Diagnostic System
============================================
Comprehensive code analysis, configuration validation, and issue detection
optimized for LLM consumption and automated fixing.

Version: 2.0
Author: Diagnostic Engine
"""
import json,sys,re,ast,subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict,List,Tuple,Optional,Set
from collections import defaultdict
from dataclasses import dataclass,asdict
@dataclass
class Issue:
    severity:str
    check:str
    issue:str
    file:str
    line:Optional[int]=None
    column:Optional[int]=None
    code_snippet:Optional[str]=None
    current_code:Optional[str]=None
    suggested_fix:Optional[str]=None
    explanation:Optional[str]=None
    impact:Optional[str]=None
    related_issues:Optional[List[str]]=None
class CodeAnalyzer:
    @staticmethod
    def find_pattern_with_context(code:str,pattern:str,context_lines:int=3)->List[Dict]:
        results=[];lines=code.split('\n')
        for i,line in enumerate(lines):
            if re.search(pattern,line):
                start=max(0,i-context_lines);end=min(len(lines),i+context_lines+1)
                results.append({'line_num':i+1,'line':line.strip(),'context':'\n'.join(lines[start:end]),'start_line':start+1,'end_line':end})
        return results
    @staticmethod
    def extract_function_body(code:str,func_name:str)->Optional[str]:
        try:
            tree=ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node,ast.FunctionDef)and node.name==func_name:
                    return ast.get_source_segment(code,node)
        except:pass
        return None
    @staticmethod
    def find_variable_usage(code:str,var_name:str)->List[Dict]:
        usages=[]
        try:
            tree=ast.parse(code);lines=code.split('\n')
            for node in ast.walk(tree):
                if isinstance(node,ast.Name)and node.id==var_name:
                    if hasattr(node,'lineno'):
                        usages.append({'line':node.lineno,'context':lines[node.lineno-1].strip(),'type':type(node.ctx).__name__})
        except:pass
        return usages
    @staticmethod
    def get_imports(code:str)->List[str]:
        imports=[]
        try:
            tree=ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node,ast.ImportFrom):
                    imports.append(node.module if node.module else'')
        except:pass
        return imports
class VIXDiagnostic:
    def __init__(self,project_root:str="."):
        self.root=Path(project_root).resolve()
        if self.root.name=="src":self.root=self.root.parent
        self.src=self.root/"src";self.core=self.src/"core";self.models=self.root/"models"
        self.issues:List[Issue]=[];self.stats={'files_analyzed':0,'lines_analyzed':0,'functions_analyzed':0,'critical':0,'high':0,'medium':0,'low':0}
        self.code_cache={};self.git_info=self._get_git_info()
    def _get_git_info(self)->Dict:
        try:
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=self.root).decode().strip()
            commit=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=self.root).decode().strip()
            status=subprocess.check_output(['git','status','--short'],cwd=self.root).decode().strip()
            return{'branch':branch,'commit':commit,'has_changes':bool(status)}
        except:return{'branch':'unknown','commit':'unknown','has_changes':False}
    def _read_file(self,path:Path)->str:
        if path in self.code_cache:return self.code_cache[path]
        try:
            with open(path,'r')as f:
                content=f.read();self.code_cache[path]=content;self.stats['lines_analyzed']+=len(content.split('\n'));self.stats['files_analyzed']+=1
                return content
        except:return""
    def _add_issue(self,issue:Issue):
        self.issues.append(issue);self.stats[issue.severity.lower()]+=1
    def run_diagnostic(self)->Dict:
        print(f"\n# VIX FORECASTER DIAGNOSTIC - {datetime.now().isoformat()}")
        print(f"Project: {self.root}")
        print(f"Git: {self.git_info['branch']}@{self.git_info['commit']}")
        print(f"Status: {'MODIFIED' if self.git_info['has_changes']else'CLEAN'}\n")
        self._check_feature_quality()
        self._check_publication_lags()
        self._check_cohort_weights()
        self._check_model_health()
        self._check_config_consistency()
        self._check_data_flow()
        self._check_error_handling()
        report=self._generate_report()
        self._save_report(report)
        self._print_claude_optimized_output(report)
        return report
    def _check_feature_quality(self):
        print("## CHECK 1: Feature Quality Enforcement (CRITICAL)")
        fe_path=self.core/"feature_engineer.py";trainer_path=self.core/"xgboost_trainer_v3.py"
        if not fe_path.exists()or not trainer_path.exists():
            self._add_issue(Issue('critical','file_missing','Required files not found','N/A',explanation='Cannot proceed without core files'));return
        fe_code=self._read_file(fe_path);trainer_code=self._read_file(trainer_path)
        computed=CodeAnalyzer.find_pattern_with_context(fe_code,r'feature_quality.*compute_feature_quality_batch')
        enforced=CodeAnalyzer.find_pattern_with_context(trainer_code,r'feature_quality.*>=.*\d+\.\d+')
        print(f"- Feature quality computed: {len(computed)>0} (found {len(computed)} locations)")
        print(f"- Feature quality enforced: {len(enforced)>0} (found {len(enforced)} locations)")
        if computed and not enforced:
            prepare_func=CodeAnalyzer.extract_function_body(trainer_code,'_prepare_features')
            self._add_issue(Issue('critical','feature_quality_not_enforced','Feature quality computed but never filtered in training','xgboost_trainer_v3.py',code_snippet=prepare_func[:500]if prepare_func else None,current_code="exclude_cols=[...,'feature_quality',...]  # Excluded from features\nX=df[feature_cols].copy()  # No filtering applied",suggested_fix="# Add BEFORE preparing features:\nif 'feature_quality' in df.columns:\n    quality_threshold=0.7  # From FEATURE_QUALITY_CONFIG\n    quality_mask=df['feature_quality']>=quality_threshold\n    low_quality_pct=(1-quality_mask.mean())*100\n    if low_quality_pct>20:\n        logger.warning(f'Filtering {low_quality_pct:.1f}% low-quality rows')\n    df=df[quality_mask].copy()\n    logger.info(f'Quality filter: {len(df)} rows retained')",explanation="The model trains on ALL data regardless of quality score. This means predictions during market stress (low quality) are trained on unreliable features, causing wild predictions.",impact="HIGH - Model learns from bad data during critical periods. Results in 2.7 VIX predictions when it should refuse or use simple heuristics."))
            usages=CodeAnalyzer.find_variable_usage(trainer_code,'feature_quality')
            print(f"  Found 'feature_quality' referenced {len(usages)} times but never used for filtering")
        if enforced:print(f"  ✓ Quality enforcement found at lines: {[r['line_num']for r in enforced]}")
    def _check_publication_lags(self):
        print("\n## CHECK 2: Publication Lag Consistency (HIGH)")
        config_path=self.src/"config.py";fe_path=self.core/"feature_engineer.py"
        if not config_path.exists()or not fe_path.exists():return
        config_code=self._read_file(config_path);fe_code=self._read_file(fe_path)
        lag_sources=re.findall(r'"([^"]+)":\d+',re.search(r'PUBLICATION_LAGS\s*=\s*\{([^}]+)\}',config_code,re.DOTALL).group(1)if'PUBLICATION_LAGS'in config_code else'')
        print(f"- Publication lags defined: {len(lag_sources)} sources")
        manual_shifts=CodeAnalyzer.find_pattern_with_context(fe_code,r'\.shift\(\d+\)')
        lag_aware_shifts=CodeAnalyzer.find_pattern_with_context(fe_code,r'PUBLICATION_LAGS\.get')
        print(f"- Manual shift() calls: {len(manual_shifts)}")
        print(f"- Lag-aware shifts: {len(lag_aware_shifts)}")
        apply_lags_func=CodeAnalyzer.extract_function_body(fe_code,'_apply_publication_lags')
        if apply_lags_func:
            calls=CodeAnalyzer.find_pattern_with_context(fe_code,r'self\._apply_publication_lags\(')
            print(f"- _apply_publication_lags defined: Yes")
            print(f"- _apply_publication_lags called: {len(calls)} times")
            if len(calls)==0:
                self._add_issue(Issue('high','publication_lags_not_used','_apply_publication_lags defined but NEVER CALLED','feature_engineer.py',code_snippet=apply_lags_func[:300],current_code="def _apply_publication_lags(self,data:Dict,idx):  # Defined but unused\n    ...\n\n# Meanwhile in _fetch_macro_data:\nfd[n]=d.reindex(idx,method='ffill')  # No lag applied!",suggested_fix="# In _fetch_macro_data, AFTER fetching each series:\nlag=PUBLICATION_LAGS.get(sid,0)\nif lag>0:\n    d=d.shift(lag)  # Apply lag BEFORE reindexing\nfd[n]=d.reindex(idx,method='ffill',limit=ff_limit)",explanation="You have a function to apply publication lags but it's never called. Some features manually shift, others don't. This creates temporal leakage for features that should be lagged but aren't.",impact="HIGH - Treasury yields, employment data, CPI available 'today' in training but wouldn't be available in real prediction. Model overfits to future data."))
        if manual_shifts:
            inconsistent=[]
            for shift in manual_shifts:
                if'PUBLICATION_LAGS'not in shift['context']:inconsistent.append(shift)
            if inconsistent:
                examples='\n'.join([f"  Line {s['line_num']}: {s['line']}"for s in inconsistent[:3]])
                self._add_issue(Issue('high','inconsistent_lag_application',f'Found {len(inconsistent)} manual shifts not using PUBLICATION_LAGS','feature_engineer.py',line=inconsistent[0]['line_num'],code_snippet=examples,suggested_fix="# Replace ALL manual shifts with:\nlag=PUBLICATION_LAGS.get(source_name,0)\nif lag>0:\n    series=series.shift(lag)",impact="MEDIUM - Creates inconsistent feature availability across sources"))
    def _check_cohort_weights(self):
        print("\n## CHECK 3: Cohort Weight Usage (MEDIUM)")
        trainer_path=self.core/"xgboost_trainer_v3.py"
        if not trainer_path.exists():return
        trainer_code=self._read_file(trainer_path)
        sample_weight_usage=CodeAnalyzer.find_pattern_with_context(trainer_code,r'sample_weight\s*=')
        cohort_weight_refs=CodeAnalyzer.find_pattern_with_context(trainer_code,r'cohort_weight')
        print(f"- sample_weight parameter used: {len(sample_weight_usage)>0}")
        print(f"- cohort_weight referenced: {len(cohort_weight_refs)} times")
        if cohort_weight_refs and not sample_weight_usage:
            fit_calls=CodeAnalyzer.find_pattern_with_context(trainer_code,r'\.fit\([^)]*\)')
            self._add_issue(Issue('medium','cohort_weights_unused','Cohort weights computed but not passed to XGBoost','xgboost_trainer_v3.py',code_snippet='\n'.join([f['context']for f in fit_calls[:2]]),current_code="model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)\n# Missing: sample_weight parameter",suggested_fix="# In _train_magnitude_model and _train_direction_model:\n# Extract weights from original df\ntrain_weights=df.loc[X_train.index,'cohort_weight'].values\nval_weights=df.loc[X_val.index,'cohort_weight'].values\n\n# Pass to fit\nmodel.fit(\n    X_train,y_train,\n    sample_weight=train_weights,\n    eval_set=[(X_val,y_val)],\n    verbose=False\n)",explanation="Cohort weights tell XGBoost that FOMC/OpEx periods are 1.35x more important. Without sample_weight, the model treats all days equally.",impact="MEDIUM - Model doesn't learn that certain calendar periods are more predictive. Hurts performance during key market events."))
        if sample_weight_usage:print(f"  ✓ sample_weight used at lines: {[r['line_num']for r in sample_weight_usage]}")
    def _check_model_health(self):
        print("\n## CHECK 4: Model Health Metrics")
        metrics_path=self.models/"training_metrics.json"
        if not metrics_path.exists():
            print("- training_metrics.json: NOT FOUND (run training first)");return
        try:
            with open(metrics_path,'r')as f:metrics=json.load(f)
            if'magnitude'in metrics and'test'in metrics['magnitude']:
                m=metrics['magnitude']['test']
                mae,bias,clip=m.get('mae_pct',0),m.get('bias_pct',0),m.get('clipped_pct',0)
                print(f"- Magnitude MAE: {mae:.2f}% (threshold: <15%)")
                print(f"- Magnitude Bias: {bias:+.2f}% (threshold: ±2%)")
                print(f"- Clipped predictions: {clip:.1f}% (threshold: <5%)")
                if mae>15:self._add_issue(Issue('high','high_mae',f'Magnitude MAE {mae:.2f}% exceeds 15% threshold','training_metrics.json',explanation='Model is inaccurate. May need more features, better feature selection, or hyperparameter tuning.',impact='HIGH - Unreliable magnitude predictions'))
                if clip>5:self._add_issue(Issue('high','high_clipping',f'Clipping rate {clip:.1f}% exceeds 5% threshold','training_metrics.json',explanation='Model making extreme predictions requiring safety clipping. Root cause likely not fully fixed.',impact='HIGH - Model unstable, requires manual bounds'))
            if'direction'in metrics and'test'in metrics['direction']:
                d=metrics['direction']['test']
                acc=d.get('accuracy',0)
                print(f"- Direction Accuracy: {acc:.1%} (threshold: >55%)")
                if acc<0.55:self._add_issue(Issue('high','low_direction_accuracy',f'Direction accuracy {acc:.1%} below 55% threshold','training_metrics.json',explanation='Barely better than random. Direction model needs improvement.',impact='HIGH - Cannot reliably predict market direction'))
        except Exception as e:print(f"- Error reading metrics: {e}")
    def _check_config_consistency(self):
        print("\n## CHECK 5: Configuration Consistency")
        config_path=self.src/"config.py"
        if not config_path.exists():return
        config_code=self._read_file(config_path)
        quality_config=re.search(r'FEATURE_QUALITY_CONFIG\s*=\s*\{([^}]+(?:\{[^}]+\}[^}]*)*)\}',config_code,re.DOTALL)
        xgb_config=re.search(r'XGBOOST_CONFIG\s*=\s*\{([^}]+(?:\{[^}]+\}[^}]*)*)\}',config_code,re.DOTALL)
        print(f"- FEATURE_QUALITY_CONFIG defined: {quality_config is not None}")
        print(f"- XGBOOST_CONFIG defined: {xgb_config is not None}")
        if quality_config:
            thresholds=re.findall(r'"(critical|important|optional)_features":\[([^\]]+)\]',quality_config.group(1))
            if thresholds:
                for tier,features in thresholds:
                    feat_list=[f.strip('"')for f in features.split(',')if f.strip()]
                    print(f"  - {tier.capitalize()}: {len(feat_list)} features")
        pub_lags=re.search(r'PUBLICATION_LAGS\s*=\s*\{([^}]+)\}',config_code,re.DOTALL)
        if pub_lags:
            zero_lag=len(re.findall(r':\s*0\s*[,}]',pub_lags.group(1)))
            nonzero_lag=len(re.findall(r':\s*[1-9]\d*\s*[,}]',pub_lags.group(1)))
            print(f"  - Publication lags: {zero_lag} same-day, {nonzero_lag} delayed")
    def _check_data_flow(self):
        print("\n## CHECK 6: Data Flow Analysis")
        fe_path=self.core/"feature_engineer.py";trainer_path=self.core/"xgboost_trainer_v3.py"
        if not fe_path.exists()or not trainer_path.exists():return
        fe_code=self._read_file(fe_path);trainer_code=self._read_file(trainer_path)
        build_func=CodeAnalyzer.extract_function_body(fe_code,'build_complete_features')
        if build_func:
            returns=re.search(r'return\s+\{([^}]+)\}',build_func,re.DOTALL)
            if returns:
                matches=re.findall(r'"([^"]+)"|\'([^\']+)\'',returns.group(1))
                return_keys=[m[0]if m[0]else m[1]for m in matches if m[0]or m[1]]
                print(f"- build_complete_features returns: {len(return_keys)} keys")
                if return_keys:print(f"  Keys: {', '.join(return_keys[:5])}")
        train_func=CodeAnalyzer.extract_function_body(trainer_code,'train')
        if train_func:
            target_calc=re.search(r'calculate_all_targets',train_func)
            feature_prep=re.search(r'_prepare_features',train_func)
            print(f"- Training pipeline:")
            print(f"  - Target calculation: {'✓'if target_calc else'✗'}")
            print(f"  - Feature preparation: {'✓'if feature_prep else'✗'}")
    def _check_error_handling(self):
        print("\n## CHECK 7: Error Handling & Resource Management")
        core_files=['feature_engineer.py','xgboost_trainer_v3.py','temporal_validator.py']
        total_try_blocks=0;total_context_mgrs=0;total_file_opens=0
        for fname in core_files:
            fpath=self.core/fname
            if not fpath.exists():continue
            code=self._read_file(fpath)
            try_blocks=len(re.findall(r'\btry\s*:',code))
            context_mgrs=len(re.findall(r'\bwith\s+.*\bas\s+',code))
            file_opens=len(re.findall(r'\bopen\s*\(',code))
            total_try_blocks+=try_blocks;total_context_mgrs+=context_mgrs;total_file_opens+=file_opens
            print(f"- {fname}: {try_blocks} try/except, {context_mgrs} context managers, {file_opens} file opens")
        if total_file_opens>total_context_mgrs:
            self._add_issue(Issue('medium','resource_leaks',f'Found {total_file_opens} file opens but only {total_context_mgrs} context managers','core/*.py',explanation='Some file operations may not properly close files on error.',suggested_fix='Always use: with open(path,"r") as f: ...'))
    def _generate_report(self)->Dict:
        critical=[i for i in self.issues if i.severity=='critical']
        high=[i for i in self.issues if i.severity=='high']
        medium=[i for i in self.issues if i.severity=='medium']
        low=[i for i in self.issues if i.severity=='low']
        return{'metadata':{'timestamp':datetime.now().isoformat(),'project_root':str(self.root),'git_branch':self.git_info['branch'],'git_commit':self.git_info['commit'],'git_modified':self.git_info['has_changes']},'statistics':self.stats,'summary':{'total_issues':len(self.issues),'by_severity':{'critical':len(critical),'high':len(high),'medium':len(medium),'low':len(low)}},'issues':{'critical':[asdict(i)for i in critical],'high':[asdict(i)for i in high],'medium':[asdict(i)for i in medium],'low':[asdict(i)for i in low]},'recommendations':self._generate_recommendations()}
    def _generate_recommendations(self)->List[Dict]:
        recs=[]
        if self.stats['critical']>0:
            recs.append({'priority':1,'category':'CRITICAL','title':'Feature Quality Enforcement','action':'Implement quality filtering in xgboost_trainer_v3.py::_prepare_features()','estimated_lines':10,'files':['src/core/xgboost_trainer_v3.py']})
        if any(i.check=='publication_lags_not_used'for i in self.issues):
            recs.append({'priority':2,'category':'HIGH','title':'Publication Lag Application','action':'Call _apply_publication_lags() or implement consistent lagging in _fetch_macro_data()','estimated_lines':5,'files':['src/core/feature_engineer.py']})
        if any(i.check=='cohort_weights_unused'for i in self.issues):
            recs.append({'priority':3,'category':'MEDIUM','title':'Cohort Weight Integration','action':'Pass sample_weight parameter in model.fit() calls','estimated_lines':6,'files':['src/core/xgboost_trainer_v3.py']})
        return recs
    def _save_report(self,report:Dict):
        out_path=self.root/"diagnostic_report.json"
        with open(out_path,'w')as f:json.dump(report,f,indent=2)
        print(f"\n✓ Full JSON report: {out_path}")
    def _print_claude_optimized_output(self,report:Dict):
        print("\n"+"="*80)
        print("CLAUDE HANDOFF SUMMARY")
        print("="*80)
        print(f"\nProject: {report['metadata']['project_root']}")
        print(f"Git: {report['metadata']['git_branch']}@{report['metadata']['git_commit']}")
        print(f"Analysis: {self.stats['files_analyzed']} files, {self.stats['lines_analyzed']} lines")
        print(f"\nISSUE BREAKDOWN:")
        print(f"  CRITICAL: {report['summary']['by_severity']['critical']}")
        print(f"  HIGH:     {report['summary']['by_severity']['high']}")
        print(f"  MEDIUM:   {report['summary']['by_severity']['medium']}")
        print(f"  LOW:      {report['summary']['by_severity']['low']}")
        if report['issues']['critical']:
            print("\n"+"="*80)
            print("CRITICAL ISSUES - FIX IMMEDIATELY")
            print("="*80)
            for i,issue in enumerate(report['issues']['critical'],1):
                print(f"\n{i}. {issue['issue']}")
                print(f"   File: {issue['file']}")
                if issue.get('line'):print(f"   Line: {issue['line']}")
                if issue.get('explanation'):print(f"   Why: {issue['explanation']}")
                if issue.get('impact'):print(f"   Impact: {issue['impact']}")
                if issue.get('suggested_fix'):print(f"\n   Suggested Fix:\n{issue['suggested_fix']}")
        if report['issues']['high']:
            print("\n"+"="*80)
            print("HIGH PRIORITY ISSUES")
            print("="*80)
            for i,issue in enumerate(report['issues']['high'],1):
                print(f"\n{i}. {issue['issue']}")
                print(f"   File: {issue['file']}")
                if issue.get('suggested_fix'):print(f"   Fix: {issue['suggested_fix'][:100]}...")
        print("\n"+"="*80)
        print("NEXT ACTIONS FOR CLAUDE")
        print("="*80)
        for rec in report['recommendations']:
            print(f"\n[Priority {rec['priority']}] {rec['title']}")
            print(f"  Action: {rec['action']}")
            print(f"  Files: {', '.join(rec['files'])}")
            print(f"  Estimated changes: ~{rec['estimated_lines']} lines")
        print("\n"+"="*80)
        print("To implement fixes, provide this report + diagnostic_report.json to Claude")
        print("="*80+"\n")
def main():
    diag=VIXDiagnostic()
    try:
        report=diag.run_diagnostic()
        return 2 if report['summary']['by_severity']['critical']>0 else(1 if report['summary']['by_severity']['high']>0 else 0)
    except Exception as e:
        print(f"\nDIAGNOSTIC FAILED: {e}")
        import traceback;traceback.print_exc()
        return 3
if __name__=="__main__":sys.exit(main())
