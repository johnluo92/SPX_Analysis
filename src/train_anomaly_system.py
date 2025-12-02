import argparse,json,logging,sys
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from config import TRAINING_YEARS
from core.data_fetcher import UnifiedDataFetcher
from core.feature_engineer import FeatureEngineer
from core.anomaly_eod import EODAnomalyScorer
from core.anomaly_database import AnomalyDatabaseExtension
from core.spike_gate import SpikeGate

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler('logs/anomaly_training.log')])
logger=logging.getLogger(__name__)

def main():
    parser=argparse.ArgumentParser(description="Train anomaly detection system")
    parser.add_argument('--test-only',action='store_true',help="Test on 2yr data")
    parser.add_argument('--enable-spike-gate',action='store_true',help="Enable overrides")
    parser.add_argument('--skip-backfill',action='store_true',help="Skip backfill")
    args=parser.parse_args()
    logger.info("="*80);logger.info("🚀 ANOMALY SYSTEM TRAINING");logger.info("="*80)
    logger.info(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Test Mode: {args.test_only} | Spike Gate: {'ENABLED'if args.enable_spike_gate else'SAFE'}")
    Path("models").mkdir(exist_ok=True);Path("data_cache").mkdir(exist_ok=True)
    
    logger.info("\n"+"="*80);logger.info("STEP 1: EXTENDING DATABASE");logger.info("="*80)
    db_ext=AnomalyDatabaseExtension();db_ext.extend_schema()
    
    logger.info("\n"+"="*80);logger.info("STEP 2: BUILDING FEATURES");logger.info("="*80)
    data_fetcher=UnifiedDataFetcher();feature_engine=FeatureEngineer(data_fetcher=data_fetcher)
    if args.test_only:
        end_date=(datetime.now()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        feature_data=feature_engine.build_complete_features(years=2,end_date=end_date,force_historical=False)
    else:
        feature_data=feature_engine.build_complete_features(years=TRAINING_YEARS,end_date=None,force_historical=False)
    df=feature_data['features']
    logger.info(f"Data: {len(df)} rows, {len(df.columns)} features")
    logger.info(f"Range: {df.index[0].date()} to {df.index[-1].date()}")
    logger.info("\nAdding velocity/acceleration features...")
    spike_features=feature_engine.add_spike_detection_features(df)
    df=pd.concat([df,spike_features],axis=1)
    logger.info(f"Added {len(spike_features.columns)} spike features → Total: {len(df.columns)}")
    
    logger.info("\n"+"="*80);logger.info("STEP 3: TRAINING DETECTOR");logger.info("="*80)
    anomaly_scorer=EODAnomalyScorer()
    keywords=['velocity','accel','jerk','_ratio','stress','divergence','transition','regime','percentile_velocity']
    selected_features=[col for col in df.columns if any(kw in col.lower()for kw in keywords)]
    logger.info(f"Selected {len(selected_features)} velocity features")
    training_stats=anomaly_scorer.train(df,feature_subset=selected_features)
    anomaly_scorer.save("models");logger.info("✅ Saved detector")
    
    logger.info("\n"+"="*80);logger.info("STEP 4: CALCULATING SCORES");logger.info("="*80)
    if not args.skip_backfill:
        anomaly_results=anomaly_scorer.calculate_scores_batch(df)
        db_ext.backfill_anomaly_scores(anomaly_scorer,df)
        db_ext.update_forecasts_with_anomaly_scores()
        high_anomaly=anomaly_results[anomaly_results['anomaly_level'].isin(['HIGH','CRITICAL'])]
        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Mean: {anomaly_results['anomaly_score'].mean():.4f}")
        logger.info(f"  Max: {anomaly_results['anomaly_score'].max():.4f}")
        logger.info(f"  High/Critical: {len(high_anomaly)} days ({len(high_anomaly)/len(anomaly_results)*100:.1f}%)")
    else:logger.info("⚠️  Skipped backfill")
    
    logger.info("\n"+"="*80);logger.info("STEP 5: TESTING SPIKE GATE");logger.info("="*80)
    mode='moderate'if args.enable_spike_gate else'safe'
    spike_gate=SpikeGate(mode=mode)
    anomaly_results=anomaly_scorer.calculate_scores_batch(df)
    high_dates=anomaly_results[anomaly_results['anomaly_score']>0.85].index[:10]
    logger.info(f"\nTesting on {len(high_dates)} high-anomaly dates:")
    for date in high_dates:
        obs=df.loc[date];score=anomaly_results.loc[date,'anomaly_score']
        mock={'direction':'DOWN','magnitude_pct':-5.0,'direction_confidence':0.60,'expected_vix':obs['vix']*0.95}
        result=spike_gate.check_and_override(forecast=mock,anomaly_score=score,current_vix=obs['vix'],regime='Normal')
        if result['spike_gate_triggered']:
            logger.info(f"  {date.date()}: {result['spike_gate_action']} (score={score:.3f}, {mock['direction']}→{result['direction']})")
    stats=spike_gate.get_override_stats()
    logger.info(f"\n📊 Spike Gate: {stats['total_checks']} checks, {stats['overrides_executed']} executed, {stats['overrides_potential']} potential")
    
    logger.info("\n"+"="*80);logger.info("STEP 6: VALIDATION (2024-2025)");logger.info("="*80)
    test_data=df[df.index>='2024-01-01']
    if len(test_data)==0:logger.warning("⚠️  No 2024-2025 data")
    else:
        test_anomaly=anomaly_scorer.calculate_scores_batch(test_data)
        vix_changes=test_data['vix'].pct_change(5)*100;spikes=vix_changes[vix_changes>15]
        logger.info(f"\n📊 Test Period:")
        logger.info(f"  Days: {len(test_data)}")
        logger.info(f"  Spikes (>15%): {len(spikes)}")
        if len(spikes)>0:
            spike_dates=spikes.index;spike_anomalies=test_anomaly.loc[spike_dates,'anomaly_score']
            high_before_spike=(spike_anomalies>0.85).sum()
            logger.info(f"  High anomaly before spike: {high_before_spike} ({high_before_spike/len(spikes)*100:.1f}%)")
            logger.info(f"\n🎯 Improvement: Baseline 3.4% → With anomaly {high_before_spike/len(spikes)*100:.1f}%")
    
    logger.info("\n"+"="*80);logger.info("STEP 7: GENERATING REPORT");logger.info("="*80)
    report={'training_date':datetime.now().isoformat(),'test_mode':args.test_only,'spike_gate_enabled':args.enable_spike_gate,'feature_stats':{'total_features':len(df.columns),'spike_features':len(spike_features.columns),'date_range':{'start':df.index[0].isoformat(),'end':df.index[-1].isoformat(),'n_days':len(df)}},'anomaly_detector':training_stats,'spike_gate_stats':spike_gate.get_override_stats(),'database_stats':db_ext.get_spike_gate_stats()if not args.skip_backfill else{}}
    with open('models/anomaly_integration_report.json','w')as f:json.dump(report,f,indent=2)
    logger.info("✅ Saved report")
    db_ext.close()
    logger.info("\n"+"="*80);logger.info("✅ COMPLETE");logger.info("="*80)
    logger.info(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("\nNext Steps:")
    logger.info("1. Review: models/anomaly_integration_report.json")
    logger.info("2. Validate spike detection improved")
    logger.info("3. Update SPIKE_GATE_CONFIG['current_mode'] when ready")
    return 0

if __name__=="__main__":sys.exit(main())
