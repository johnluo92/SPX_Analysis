import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import TARGET_CONFIG,TRAINING_YEARS,FEATURE_SELECTION_CONFIG,XGBOOST_CONFIG,get_last_complete_month_end,DATA_SPLIT_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.xgboost_feature_selector_v2 import FeatureSelector
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
    logger.info(f"  Feature selection through: {feature_selection_split_date}")

    split_date_idx=features_df[features_df.index<=pd.Timestamp(feature_selection_split_date)].index[-1]
    test_start_idx=features_df.index.get_loc(split_date_idx)+1
    top_n=FEATURE_SELECTION_CONFIG[f"{target_type}_top_n"]
    corr_threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.90)

    selector=FeatureSelector(target_type=target_type,top_n=top_n,correlation_threshold=corr_threshold)
    selected_features,metadata=selector.select_features(features_df[feature_cols],vix,test_start_idx=test_start_idx)
    selector.save_results(output_dir="data_cache",suffix=f"_{target_type}")
    selector.generate_diagnostics(output_dir="diagnostics",suffix=f"_{target_type}")

    return selected_features

def save_training_report(forecaster,exp_features,comp_features,up_features,down_features,training_end,output_dir="models"):
    output_path=Path(output_dir); output_path.mkdir(parents=True,exist_ok=True)
    report_file=output_path/f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    report={"timestamp":datetime.now().isoformat(),"system_version":"v6.1_unified_selector","training_end":training_end,"data_splits":DATA_SPLIT_CONFIG,"feature_selection":{"expansion":{"top_n":FEATURE_SELECTION_CONFIG["expansion_top_n"],"selected":len(exp_features),"features":exp_features},"compression":{"top_n":FEATURE_SELECTION_CONFIG["compression_top_n"],"selected":len(comp_features),"features":comp_features},"up":{"top_n":FEATURE_SELECTION_CONFIG["up_top_n"],"selected":len(up_features),"features":up_features},"down":{"top_n":FEATURE_SELECTION_CONFIG["down_top_n"],"selected":len(down_features),"features":down_features}},"metrics":forecaster.metrics}

    with open(report_file,"w")as f: json.dump(report,f,indent=2,default=str)
    logger.info(f"Training report: {report_file}")

def main():
    logger.info("ASYMMETRIC 4-MODEL VIX FORECASTER")
    logger.info(f"Version: 6.1 (Unified Selector) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("DATA SPLIT CONFIGURATION")
    logger.info(f"Train End:     {DATA_SPLIT_CONFIG['train_end_date']}")
    logger.info(f"Val End:       {DATA_SPLIT_CONFIG['val_end_date']}")
    logger.info(f"Test Start:    {pd.Timestamp(DATA_SPLIT_CONFIG['val_end_date'])+pd.Timedelta(days=1)}")
    logger.info(f"Feature Split: {DATA_SPLIT_CONFIG['feature_selection_split_date']}")

    try:
        complete_df,vix,training_end=prepare_training_data()

        logger.info("\n"+"="*80)
        logger.info("UNIFIED FEATURE SELECTION (importance + correlation)")

        exp_features=run_feature_selection(complete_df,vix,target_type='expansion')
        comp_features=run_feature_selection(complete_df,vix,target_type='compression')
        up_features=run_feature_selection(complete_df,vix,target_type='up')
        down_features=run_feature_selection(complete_df,vix,target_type='down')

        logger.info("\n"+"="*80)
        logger.info("🚀 ASYMMETRIC 4-MODEL TRAINING")

        forecaster=train_asymmetric_forecaster(df=complete_df,expansion_features=exp_features,compression_features=comp_features,up_features=up_features,down_features=down_features,save_dir="models")

        save_training_report(forecaster,exp_features,comp_features,up_features,down_features,training_end,output_dir="models")

        logger.info("✅ TRAINING COMPLETE")
        logger.info("Models: models/expansion_model.pkl + compression_model.pkl + up_classifier.pkl + down_classifier.pkl")
        logger.info(f"Training data through: {training_end}")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__=="__main__": main()
