import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import TARGET_CONFIG,TRAINING_YEARS,FEATURE_SELECTION_CONFIG,XGBOOST_CONFIG,get_last_complete_month_end,DATA_SPLIT_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
from core.feature_correlation_analyzer import FeatureCorrelationAnalyzer
from core.xgboost_trainer_v3 import train_asymmetric_forecaster

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/training.log")])
logger=logging.getLogger(__name__)

def prepare_training_data():
    training_end=get_last_complete_month_end(); data_fetcher=UnifiedDataFetcher()
    feature_engineer=FeatureEngineer(data_fetcher)
    logger.info(f"📅 Training through: {training_end}")
    result=feature_engineer.build_complete_features(years=TRAINING_YEARS,end_date=training_end)
    features_df=result["features"]; spx=result["spx"]; vix=result["vix"]
    complete_df=features_df.copy(); complete_df["vix"]=vix; complete_df["spx"]=spx
    if "calendar_cohort"not in complete_df.columns: raise ValueError("calendar_cohort missing!")
    cohort_counts=complete_df["calendar_cohort"].value_counts()
    logger.info("  Cohort distribution:")
    [logger.info(f"  {cohort}: {count}")for cohort,count in cohort_counts.items()]
    logger.info("  Data preparation complete\n")
    return complete_df,vix,training_end

def run_feature_selection(features_df,vix,target_type='expansion'):
    logger.info(f"\n📊 FEATURE SELECTION - {target_type.upper()}")
    feature_cols=[c for c in features_df.columns if c not in["vix","spx","calendar_cohort","cohort_weight","feature_quality"]]
    feature_selection_split_date=DATA_SPLIT_CONFIG["feature_selection_split_date"]
    logger.info(f"  Using feature selection split date: {feature_selection_split_date}")
    logger.info(f"  Train+Val: data up to {feature_selection_split_date}")
    logger.info(f"  Test (excluded): data after {feature_selection_split_date}")

    split_date_idx=features_df[features_df.index<=pd.Timestamp(feature_selection_split_date)].index[-1]
    test_start_idx=features_df.index.get_loc(split_date_idx)+1
    top_n=FEATURE_SELECTION_CONFIG[f"{target_type}_top_n"]

    selector=SimplifiedFeatureSelector(target_type=target_type,top_n=top_n)
    selected_features,metadata=selector.select_features(features_df[feature_cols],vix,test_start_idx=test_start_idx)
    selector.save_results(output_dir="data_cache",suffix=f"_{target_type}")
    return selected_features

def save_training_report(forecaster,exp_features,comp_features,up_features,down_features,training_end,output_dir="models"):
    output_path=Path(output_dir); output_path.mkdir(parents=True,exist_ok=True)
    report_file=output_path/f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    report={"timestamp":datetime.now().isoformat(),"system_version":"v6.0_asymmetric_4model","training_end":training_end,"data_splits":DATA_SPLIT_CONFIG,"target_types":["expansion_regressor","compression_regressor","up_classifier","down_classifier"],"feature_selection":{"expansion":{"enabled":True,"top_n":FEATURE_SELECTION_CONFIG["expansion_top_n"],"selected_features":len(exp_features),"selected_feature_list":exp_features},"compression":{"enabled":True,"top_n":FEATURE_SELECTION_CONFIG["compression_top_n"],"selected_features":len(comp_features),"selected_feature_list":comp_features},"up":{"enabled":True,"top_n":FEATURE_SELECTION_CONFIG["up_top_n"],"selected_features":len(up_features),"selected_feature_list":up_features},"down":{"enabled":True,"top_n":FEATURE_SELECTION_CONFIG["down_top_n"],"selected_features":len(down_features),"selected_feature_list":down_features}},"training_summary":{"models_trained":4,"model_types":["expansion_regressor","compression_regressor","up_classifier","down_classifier"],"expansion_features":len(forecaster.expansion_features),"compression_features":len(forecaster.compression_features),"up_features":len(forecaster.up_features),"down_features":len(forecaster.down_features)},"metrics":forecaster.metrics}

    with open(report_file,"w")as f: json.dump(report,f,indent=2,default=str)
    logger.info(f"Training report: {report_file}")

    metadata_file=output_path/"training_metadata.json"
    metadata={"training_end":training_end,"timestamp":datetime.now().isoformat(),"data_splits":DATA_SPLIT_CONFIG,"expansion_feature_count":len(forecaster.expansion_features),"compression_feature_count":len(forecaster.compression_features),"up_feature_count":len(forecaster.up_features),"down_feature_count":len(forecaster.down_features),"test_expansion_mae":forecaster.metrics.get("expansion",{}).get("test",{}).get("mae_pct",0),"test_compression_mae":forecaster.metrics.get("compression",{}).get("test",{}).get("mae_pct",0),"test_up_f1":forecaster.metrics.get("up_classifier_calibrated",{}).get("test_f1",0),"test_down_f1":forecaster.metrics.get("down_classifier_calibrated",{}).get("test_f1",0)}

    with open(metadata_file,"w")as f: json.dump(metadata,f,indent=2,default=str)
    logger.info(f"Training metadata: {metadata_file}")

def main():
    logger.info("ASYMMETRIC 4-MODEL VIX FORECASTER")
    logger.info(f"Version: 6.0 (Expansion/Compression + UP/DOWN) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n"+"="*80)
    logger.info("DATA SPLIT CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Train End:     {DATA_SPLIT_CONFIG['train_end_date']}")
    logger.info(f"Val End:       {DATA_SPLIT_CONFIG['val_end_date']}")
    logger.info(f"Test Start:    {pd.Timestamp(DATA_SPLIT_CONFIG['val_end_date'])+pd.Timedelta(days=1)}")
    logger.info(f"Feature Split: {DATA_SPLIT_CONFIG['feature_selection_split_date']}")
    logger.info("="*80+"\n")

    try:
        complete_df,vix,training_end=prepare_training_data()

        # Run 4 separate feature selections
        logger.info("\n"+"="*80)
        logger.info("DOMAIN-SPECIFIC FEATURE SELECTION")
        logger.info("="*80)

        exp_features=run_feature_selection(complete_df,vix,target_type='expansion')
        with open("data_cache/feature_importance_expansion.json")as f: exp_importance=json.load(f)
        exp_analyzer=FeatureCorrelationAnalyzer(threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.95))
        exp_kept,exp_removed=exp_analyzer.analyze_and_remove(features_df=complete_df[exp_features],importance_scores=exp_importance,protected_features=FEATURE_SELECTION_CONFIG["protected_features"])
        exp_analyzer.generate_report(output_dir="diagnostics",suffix="_expansion")
        logger.info(f"Expansion: Removed {len(exp_removed)} correlated features, kept {len(exp_kept)}")

        comp_features=run_feature_selection(complete_df,vix,target_type='compression')
        with open("data_cache/feature_importance_compression.json")as f: comp_importance=json.load(f)
        comp_analyzer=FeatureCorrelationAnalyzer(threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.95))
        comp_kept,comp_removed=comp_analyzer.analyze_and_remove(features_df=complete_df[comp_features],importance_scores=comp_importance,protected_features=FEATURE_SELECTION_CONFIG["protected_features"])
        comp_analyzer.generate_report(output_dir="diagnostics",suffix="_compression")
        logger.info(f"Compression: Removed {len(comp_removed)} correlated features, kept {len(comp_kept)}")

        up_features=run_feature_selection(complete_df,vix,target_type='up')
        with open("data_cache/feature_importance_up.json")as f: up_importance=json.load(f)
        up_analyzer=FeatureCorrelationAnalyzer(threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.95))
        up_kept,up_removed=up_analyzer.analyze_and_remove(features_df=complete_df[up_features],importance_scores=up_importance,protected_features=FEATURE_SELECTION_CONFIG["protected_features"])
        up_analyzer.generate_report(output_dir="diagnostics",suffix="_up")
        logger.info(f"UP: Removed {len(up_removed)} correlated features, kept {len(up_kept)}")

        down_features=run_feature_selection(complete_df,vix,target_type='down')
        with open("data_cache/feature_importance_down.json")as f: down_importance=json.load(f)
        down_analyzer=FeatureCorrelationAnalyzer(threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.95))
        down_kept,down_removed=down_analyzer.analyze_and_remove(features_df=complete_df[down_features],importance_scores=down_importance,protected_features=FEATURE_SELECTION_CONFIG["protected_features"])
        down_analyzer.generate_report(output_dir="diagnostics",suffix="_down")
        logger.info(f"DOWN: Removed {len(down_removed)} correlated features, kept {len(down_kept)}")

        logger.info("="*80+"\n")
        logger.info("🚀 ASYMMETRIC 4-MODEL TRAINING")

        forecaster=train_asymmetric_forecaster(df=complete_df,expansion_features=exp_kept,compression_features=comp_kept,up_features=up_kept,down_features=down_kept,save_dir="models")

        save_training_report(forecaster,exp_kept,comp_kept,up_kept,down_kept,training_end,output_dir="models")

        logger.info("✅ TRAINING COMPLETE")
        logger.info("Models: models/expansion_model.pkl + compression_model.pkl + up_classifier.pkl + down_classifier.pkl")
        logger.info(f"Training data through: {training_end}")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__=="__main__": main()
