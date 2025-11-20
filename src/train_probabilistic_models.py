import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import TARGET_CONFIG,TRAINING_END_DATE,TRAINING_YEARS,FEATURE_SELECTION_CONFIG
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
from core.feature_correlation_analyzer import FeatureCorrelationAnalyzer
from core.xgboost_trainer_v3 import train_simplified_forecaster
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/training.log")])
logger=logging.getLogger(__name__)
def prepare_training_data():
    data_fetcher=UnifiedDataFetcher();feature_engineer=FeatureEngineer(data_fetcher)
    result=feature_engineer.build_complete_features(years=TRAINING_YEARS,end_date=TRAINING_END_DATE)
    features_df=result["features"];spx=result["spx"];vix=result["vix"]
    complete_df=features_df.copy();complete_df["vix"]=vix;complete_df["spx"]=spx
    if "calendar_cohort"not in complete_df.columns:raise ValueError("calendar_cohort column missing!")
    cohort_counts=complete_df["calendar_cohort"].value_counts()
    logger.info("  Cohort distribution:")
    for cohort,count in cohort_counts.items():logger.info(f"  {cohort}: {count}")
    logger.info("  Data preparation complete\n")
    return complete_df,vix
def run_feature_selection(features_df,vix):
    logger.info(" FEATURE SELECTION")
    feature_cols=[c for c in features_df.columns if c not in["vix","spx","calendar_cohort","cohort_weight","feature_quality"]]
    selector=SimplifiedFeatureSelector()
    selected_features,metadata=selector.select_features(features_df[feature_cols],vix)
    selector.save_results(output_dir="data_cache")
    return selected_features
def save_training_report(forecaster,selected_features,output_dir="models"):
    output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True)
    report_file=output_path/f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report={"timestamp":datetime.now().isoformat(),"system_version":"v4.0_with_feature_selection","target_type":TARGET_CONFIG.get("target_type"),"feature_selection":{"enabled":True,"top_n":FEATURE_SELECTION_CONFIG["top_n"],"selected_features":len(selected_features),"selected_feature_list":selected_features},"training_summary":{"models_trained":2,"model_types":["direction_classifier","magnitude_regressor"],"features":len(forecaster.feature_names)},"metrics":forecaster.metrics}
    with open(report_file,"w")as f:json.dump(report,f,indent=2,default=str)
    logger.info(f"Training report: {report_file}")
def main():
    logger.info("SIMPLIFIED VIX FORECASTER - TRAINING PIPELINE WITH FEATURE SELECTION")
    logger.info(f"Version: 4.1 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        complete_df,vix=prepare_training_data()
        selected_features=run_feature_selection(complete_df,vix)
        with open("data_cache/feature_importance.json")as f:importance_scores=json.load(f)
        analyzer=FeatureCorrelationAnalyzer(threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.95))
        kept_features,removed_features=analyzer.analyze_and_remove(features_df=complete_df[selected_features],importance_scores=importance_scores,protected_features=FEATURE_SELECTION_CONFIG["protected_features"])
        analyzer.generate_report(output_dir="diagnostics")
        logger.info(f"Removed {len(removed_features)} correlated features")
        logger.info(f"Final feature count: {len(kept_features)}")
        logger.info("="*80+"\n")
        logger.info("MODEL TRAINING")
        forecaster=train_simplified_forecaster(df=complete_df,selected_features=kept_features,save_dir="models")
        save_training_report(forecaster,kept_features,output_dir="models")
        logger.info("TRAINING COMPLETE")
        logger.info("Models: models/direction_5d_model.pkl, models/magnitude_5d_model.pkl")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__=="__main__":main()
