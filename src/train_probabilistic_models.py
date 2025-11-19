import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import TARGET_CONFIG,TRAINING_END_DATE,TRAINING_YEARS
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
from core.xgboost_trainer_v3 import train_simplified_forecaster
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/training.log")])
logger=logging.getLogger(__name__)
def prepare_training_data():
    logger.info("DATA PREPARATION")
    data_fetcher=UnifiedDataFetcher();feature_engineer=FeatureEngineer(data_fetcher)
    logger.info(f"\nBuilding {TRAINING_YEARS}y feature set ending {TRAINING_END_DATE}...")
    result=feature_engineer.build_complete_features(years=TRAINING_YEARS,end_date=TRAINING_END_DATE)
    features_df=result["features"];spx=result["spx"];vix=result["vix"]
    complete_df=features_df.copy();complete_df["vix"]=vix;complete_df["spx"]=spx
    if "calendar_cohort"not in complete_df.columns:raise ValueError("calendar_cohort column missing!")
    cohort_counts=complete_df["calendar_cohort"].value_counts()
    logger.info("Cohort distribution:")
    for cohort,count in cohort_counts.items():logger.info(f"  {cohort}: {count}")
    logger.info("Data preparation complete\n")
    return complete_df,vix
def run_feature_selection(features_df,vix):
    logger.info("\nFEATURE SELECTION")
    logger.info("="*80)
    protected_features=["is_fomc_period","is_opex_week","is_earnings_heavy"]
    exclude_cols=["vix","spx","calendar_cohort","cohort_weight","feature_quality"]+protected_features
    feature_cols=[c for c in features_df.columns if c not in exclude_cols]
    logger.info(f"  Total candidate features: {len(feature_cols)}")
    logger.info(f"  Protected features: {protected_features}")
    selector=SimplifiedFeatureSelector(horizon=TARGET_CONFIG["horizon_days"],top_n=60,cv_folds=3,protected_features=protected_features)
    selected_features,metadata=selector.select_features(features_df[feature_cols],vix)
    selector.save_results(output_dir="data_cache")
    logger.info("\n"+"="*80)
    logger.info(f"FEATURE SELECTION COMPLETE: {len(selected_features)} total features")
    logger.info(f"  - {len(selected_features)-len(protected_features)} selected")
    logger.info(f"  - {len(protected_features)} protected")
    logger.info("="*80+"\n")
    return selected_features
def save_training_report(forecaster,selected_features,output_dir="models"):
    output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True)
    report_file=output_path/f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report={"timestamp":datetime.now().isoformat(),"system_version":"v4.0_with_protected_features","target_type":TARGET_CONFIG.get("target_type"),"feature_selection":{"enabled":True,"top_n":60,"protected_features":["is_fomc_period","is_opex_week","is_earnings_heavy"],"selected_features":len(selected_features),"selected_feature_list":selected_features},"training_summary":{"models_trained":2,"model_types":["direction_classifier","magnitude_regressor"],"features":len(forecaster.feature_names)},"metrics":forecaster.metrics}
    with open(report_file,"w")as f:json.dump(report,f,indent=2,default=str)
    logger.info(f"Training report: {report_file}")
def main():
    logger.info("SIMPLIFIED VIX FORECASTER - TRAINING PIPELINE WITH FEATURE SELECTION")
    logger.info(f"Version: 4.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        complete_df,vix=prepare_training_data()
        selected_features=run_feature_selection(complete_df,vix)
        logger.info("MODEL TRAINING")
        forecaster=train_simplified_forecaster(df=complete_df,selected_features=selected_features,save_dir="models")
        save_training_report(forecaster,selected_features,output_dir="models")
        logger.info("TRAINING COMPLETE")
        logger.info("Models: models/direction_5d_model.pkl, models/magnitude_5d_model.pkl")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc();sys.exit(1)
if __name__=="__main__":main()
