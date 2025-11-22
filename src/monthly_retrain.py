import json,logging,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import MODEL_VALIDATION_MAE_THRESHOLD
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/monthly_retrain.log")])
logger=logging.getLogger(__name__)
def is_first_business_day_of_month():
    today=datetime.now();first_of_month=today.replace(day=1);bus_days=pd.bdate_range(start=first_of_month,end=today);return len(bus_days)==1
def check_bootstrap_needed():
    models_prod=Path("models");magnitude_file=models_prod/"magnitude_5d_model.pkl";calibrator_file=models_prod/"calibrator.pkl"
    return not(magnitude_file.exists()and calibrator_file.exists())
def run_training():
    logger.info("="*80);logger.info("STEP 1: TRAINING NEW MODEL");logger.info("="*80)
    result=subprocess.run([sys.executable,"train_probabilistic_models.py"],capture_output=True,text=True)
    logger.info(result.stdout)
    if result.returncode!=0:logger.error("❌ Training failed");logger.error(result.stderr);return False
    logger.info("✅ Training complete\n");return True
def run_calibration():
    logger.info("="*80);logger.info("STEP 2: CALIBRATING NEW MODEL");logger.info("="*80)
    from core.prediction_database import PredictionDatabase
    from core.forecast_calibrator import ForecastCalibrator
    db=PredictionDatabase();cal=ForecastCalibrator();success=cal.fit_from_database(db)
    if not success:logger.warning("⚠️  Calibration skipped (no data) - using pass-through mode")
    cal.save("models_temp")
    logger.info("✅ Calibration complete\n");return True
def validate_new_model():
    logger.info("="*80);logger.info("STEP 3: VALIDATING NEW MODEL");logger.info("="*80)
    metadata_file=Path("models_temp/training_metadata.json")
    if not metadata_file.exists():logger.error("❌ Metadata not found");return False
    with open(metadata_file)as f:metadata=json.load(f)
    test_mae=metadata.get("test_mae",999)
    logger.info(f"Test MAE: {test_mae:.2f}%")
    logger.info(f"Threshold: {MODEL_VALIDATION_MAE_THRESHOLD*100:.0f}%")
    if test_mae>=MODEL_VALIDATION_MAE_THRESHOLD*100:logger.error(f"❌ Test MAE ({test_mae:.2f}%) exceeds threshold ({MODEL_VALIDATION_MAE_THRESHOLD*100:.0f}%)");logger.error("Model deployment BLOCKED. Investigate before deploying.");return False
    logger.info("✅ Validation passed\n");return True
def deploy_new_model():
    logger.info("="*80);logger.info("STEP 4: DEPLOYING NEW MODEL");logger.info("="*80)
    models_temp=Path("models_temp");models_prod=Path("models")
    if not models_temp.exists():logger.error("❌ Temp models not found");return False
    if models_prod.exists():
        archive_dir=Path("models_archive");archive_dir.mkdir(exist_ok=True);timestamp=datetime.now().strftime("%Y%m%d_%H%M%S");archive_path=archive_dir/f"models_{timestamp}"
        logger.info(f"Archiving old model to: {archive_path}")
        shutil.move(str(models_prod),str(archive_path))
    logger.info("Moving new model to production...")
    shutil.move(str(models_temp),str(models_prod))
    old_cal=models_prod/"forecast_calibrator.pkl"
    if old_cal.exists():
        logger.info("Removing old calibrator file")
        old_cal.unlink()
    logger.info("✅ Deployment complete\n");return True
def bootstrap_deploy():
    logger.info("="*80);logger.info("BOOTSTRAP DEPLOYMENT");logger.info("="*80)
    logger.info("First-time setup detected - deploying without validation")
    models_temp=Path("models_temp");models_prod=Path("models")
    if not models_temp.exists():logger.error("❌ Temp models not found");return False
    if models_prod.exists():
        logger.warning("⚠️  Production models exist, aborting bootstrap")
        return False
    logger.info("Moving new model to production...")
    shutil.move(str(models_temp),str(models_prod))
    logger.info("✅ Bootstrap deployment complete\n");return True
def cleanup():
    logger.info("="*80);logger.info("STEP 5: CLEANUP");logger.info("="*80)
    temp_dir=Path("models_temp")
    if temp_dir.exists():
        logger.info("Removing temp directory")
        shutil.rmtree(temp_dir)
    logger.info("✅ Cleanup complete\n")
def send_alert(message,success=True):
    logger.info("="*80)
    if success:logger.info("✅ MONTHLY RETRAIN SUCCESSFUL")
    else:logger.error("❌ MONTHLY RETRAIN FAILED")
    logger.info("="*80)
    logger.info(message)
    logger.info("="*80)
def main():
    logger.info("\n"+"="*80)
    logger.info("MONTHLY RETRAINING ORCHESTRATOR")
    logger.info("="*80)
    logger.info(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    bootstrap_mode=check_bootstrap_needed()
    if bootstrap_mode:
        logger.info("🔧 BOOTSTRAP MODE: First-time setup detected")
        logger.info("Will deploy without validation (no predictions exist yet)\n")
    elif not is_first_business_day_of_month():
        logger.info("Not first business day of month. Exiting.")
        logger.info("This script should run on first business day after market close.")
        sys.exit(0)
    else:logger.info("✓ First business day of month confirmed\n")
    try:
        if not run_training():send_alert("Training failed. Old model remains in production.",success=False);sys.exit(1)
        if not run_calibration():send_alert("Calibration failed. Old model remains in production.",success=False);sys.exit(1)
        if bootstrap_mode:
            logger.info("🔧 Skipping validation (bootstrap mode)")
            if not bootstrap_deploy():send_alert("Bootstrap deployment failed.",success=False);sys.exit(1)
            send_alert(f"Bootstrap deployment successful on {datetime.now().strftime('%Y-%m-%d')}. Run production forecasts to build calibration data.",success=True)
            logger.info("\nNext steps:")
            logger.info("1. Run: python integrated_system_production.py --mode forecast")
            logger.info("2. Accumulate 252+ predictions with actuals")
            logger.info("3. Future monthly retrains will include calibration")
        else:
            if not validate_new_model():send_alert("Validation failed: Test MAE exceeds threshold. Old model remains in production. INVESTIGATE BEFORE DEPLOYING.",success=False);sys.exit(1)
            if not deploy_new_model():send_alert("Deployment failed. Old model may be in inconsistent state. CHECK MANUALLY.",success=False);sys.exit(1)
            cleanup()
            send_alert(f"New model deployed successfully on {datetime.now().strftime('%Y-%m-%d')}",success=True)
            logger.info("\nNext steps:")
            logger.info("1. Monitor production forecasts for next 5 days")
            logger.info("2. Compare MAE to previous month")
            logger.info("3. Run: python integrated_system_production.py --mode forecast")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}",exc_info=True)
        send_alert(f"Unexpected error: {e}. Old model remains in production.",success=False)
        sys.exit(1)
if __name__=="__main__":main()
