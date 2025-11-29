import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import TARGET_CONFIG,TRAINING_YEARS,FEATURE_SELECTION_CONFIG,XGBOOST_CONFIG,get_last_complete_month_end
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.xgboost_feature_selector_v2 import SimplifiedFeatureSelector
from core.feature_correlation_analyzer import FeatureCorrelationAnalyzer
from core.xgboost_trainer_v3 import train_simplified_forecaster
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/training.log")])
logger=logging.getLogger(__name__)
def prepare_training_data():
    training_end=get_last_complete_month_end();data_fetcher=UnifiedDataFetcher();feature_engineer=FeatureEngineer(data_fetcher)
    logger.info(f"📅 Training through: {training_end}")
    result=feature_engineer.build_complete_features(years=TRAINING_YEARS,end_date=training_end)
    features_df=result["features"];spx=result["spx"];vix=result["vix"]
    complete_df=features_df.copy();complete_df["vix"]=vix;complete_df["spx"]=spx
    if "calendar_cohort"not in complete_df.columns:raise ValueError("calendar_cohort missing!")
    cohort_counts=complete_df["calendar_cohort"].value_counts()
    logger.info("  Cohort distribution:");[logger.info(f"  {cohort}: {count}")for cohort,count in cohort_counts.items()]
    logger.info("  Data preparation complete\n")
    return complete_df,vix,training_end
def run_feature_selection(features_df,vix,target_type='magnitude'):
    from config import SPLIT_DATES
    import pandas as pd
    logger.info(f"\n📊 FEATURE SELECTION - {target_type.upper()}")
    feature_cols=[c for c in features_df.columns if c not in["vix","spx","calendar_cohort","cohort_weight","feature_quality"]]
    test_start_idx=features_df.index.get_loc(features_df[features_df.index>pd.Timestamp(SPLIT_DATES['val_end'])].index[0])
    top_n=FEATURE_SELECTION_CONFIG[f"{target_type}_top_n"]
    selector=SimplifiedFeatureSelector(target_type=target_type,top_n=top_n)
    selected_features,metadata=selector.select_features(features_df[feature_cols],vix,test_start_idx=test_start_idx)
    selector.save_results(output_dir="data_cache")
    return selected_features
def save_training_report(forecaster,mag_features,dir_features,training_end,output_dir="models"):
    output_path=Path(output_dir);output_path.mkdir(parents=True,exist_ok=True)
    report_file=output_path/f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report={"timestamp":datetime.now().isoformat(),"system_version":"v5.3_dual_model","training_end":training_end,"target_types":["magnitude_regressor","direction_classifier"],"feature_selection":{"magnitude":{"enabled":True,"top_n":FEATURE_SELECTION_CONFIG["magnitude_top_n"],"selected_features":len(mag_features),"selected_feature_list":mag_features},"direction":{"enabled":True,"top_n":FEATURE_SELECTION_CONFIG["direction_top_n"],"selected_features":len(dir_features),"selected_feature_list":dir_features}},"training_summary":{"models_trained":2,"model_types":["magnitude_regressor","direction_classifier"],"magnitude_features":len(forecaster.magnitude_feature_names),"direction_features":len(forecaster.direction_feature_names)},"metrics":forecaster.metrics}
    with open(report_file,"w")as f:json.dump(report,f,indent=2,default=str)
    logger.info(f"Training report: {report_file}")
    metadata_file=output_path/"training_metadata.json";metadata={"training_end":training_end,"timestamp":datetime.now().isoformat(),"magnitude_feature_count":len(forecaster.magnitude_feature_names),"direction_feature_count":len(forecaster.direction_feature_names),"test_mae":forecaster.metrics.get("magnitude",{}).get("test",{}).get("mae_pct",0),"test_direction_acc":forecaster.metrics.get("direction",{}).get("test",{}).get("accuracy",0)}
    with open(metadata_file,"w")as f:json.dump(metadata,f,indent=2,default=str)
    logger.info(f"Training metadata: {metadata_file}")
def main():
    logger.info("DUAL MODEL VIX FORECASTER - MAGNITUDE + DIRECTION")
    logger.info(f"Version: 5.3 (Dual Model) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        complete_df,vix,training_end=prepare_training_data()
        mag_features=run_feature_selection(complete_df,vix,target_type='magnitude')
        with open("data_cache/feature_importance_magnitude.json")as f:mag_importance=json.load(f)
        mag_analyzer=FeatureCorrelationAnalyzer(threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.95))
        mag_kept,mag_removed=mag_analyzer.analyze_and_remove(features_df=complete_df[mag_features],importance_scores=mag_importance,protected_features=FEATURE_SELECTION_CONFIG["protected_features"])
        mag_analyzer.generate_report(output_dir="diagnostics",suffix="_magnitude")
        logger.info(f"Magnitude: Removed {len(mag_removed)} correlated features, kept {len(mag_kept)}")
        dir_features=run_feature_selection(complete_df,vix,target_type='direction')
        with open("data_cache/feature_importance_direction.json")as f:dir_importance=json.load(f)
        dir_analyzer=FeatureCorrelationAnalyzer(threshold=FEATURE_SELECTION_CONFIG.get("correlation_threshold",0.95))
        dir_kept,dir_removed=dir_analyzer.analyze_and_remove(features_df=complete_df[dir_features],importance_scores=dir_importance,protected_features=FEATURE_SELECTION_CONFIG["protected_features"])
        dir_analyzer.generate_report(output_dir="diagnostics",suffix="_direction")
        logger.info(f"Direction: Removed {len(dir_removed)} correlated features, kept {len(dir_kept)}")
        logger.info("="*80+"\n")
        logger.info("🚀 MODEL TRAINING")
        forecaster=train_simplified_forecaster(df=complete_df,magnitude_features=mag_kept,direction_features=dir_kept,save_dir="models")
        save_training_report(forecaster,mag_kept,dir_kept,training_end,output_dir="models")
        logger.info("✅ TRAINING COMPLETE")
        logger.info("Models: models/magnitude_5d_model.pkl + models/direction_5d_model.pkl")
        logger.info(f"Training data through: {training_end}")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__=="__main__":main()
