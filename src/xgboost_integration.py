"""
XGBoost Integration Script - Seamless Connection to IntegratedMarketSystemV4
============================================================================

This script provides a zero-friction integration path:
1. Feature selection (once during training)
2. Anomaly detector retraining with selected features
3. Regime transition forecaster (hybrid XGB + heuristics)
4. Export unified outputs for dashboards

Usage:
    python xgboost_integration.py --mode full  # Complete pipeline
    python xgboost_integration.py --mode features_only  # Just feature selection
    python xgboost_integration.py --mode forecast_only  # Just forecasting
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from integrated_system_production import IntegratedMarketSystemV4
from xgboost_trainer_v2 import train_enhanced_xgboost
from xgboost_feature_selector_v2 import run_intelligent_feature_selection
from config import ENABLE_TRAINING, TRAINING_YEARS


class XGBoostIntegrationManager:
    """
    Manages XGBoost integration with IntegratedMarketSystemV4.
    
    Workflow:
    1. Train base system (features + anomaly detection)
    2. Run feature selection (identify top 50-75 features)
    3. Retrain anomaly detectors with selected features
    4. Train XGBoost regime classifier
    5. Build hybrid forecaster (XGB + anomaly + heuristics)
    """
    
    def __init__(self, system: IntegratedMarketSystemV4):
        self.system = system
        self.xgb_trainer = None
        self.selected_features = None
        self.forecaster = None
        
        # Paths
        self.models_dir = Path('./models')
        self.models_dir.mkdir(exist_ok=True)
        
    def run_full_pipeline(
        self,
        optimize_hyperparams: bool = False,
        retrain_anomaly: bool = True,
        build_forecaster: bool = True,
        verbose: bool = True
    ):
        """
        Execute complete XGBoost integration pipeline.
        
        Args:
            optimize_hyperparams: Run nested CV (adds 10-15 min, +2-3% accuracy)
            retrain_anomaly: Retrain anomaly detectors with selected features
            build_forecaster: Create hybrid regime transition forecaster
            verbose: Print detailed logs
        """
        if not self.system.trained:
            raise ValueError("Train IntegratedMarketSystemV4 first: system.train()")
        
        if verbose:
            print(f"\n{'='*80}")
            print("🚀 XGBOOST INTEGRATION PIPELINE")
            print(f"{'='*80}")
            print(f"Mode: {'Optimized' if optimize_hyperparams else 'Fast'}")
        
        # PHASE 1: Feature Selection
        if verbose:
            print(f"\n{'='*80}")
            print("PHASE 1: INTELLIGENT FEATURE SELECTION")
            print(f"{'='*80}")
        
        selection_results = run_intelligent_feature_selection(
            self.system,
            min_stability=0.3,
            max_correlation=0.95,
            preserve_forward_indicators=True,
            verbose=verbose
        )
        
        self.selected_features = selection_results['selected_features']
        
        if verbose:
            print(f"\n✅ Selected {len(self.selected_features)} features")
        
        # PHASE 2: Train Enhanced XGBoost
        if verbose:
            print(f"\n{'='*80}")
            print("PHASE 2: TRAIN ENHANCED XGBOOST")
            print(f"{'='*80}")
        
        # Filter system features to selected features
        system_copy = self.system
        original_features = system_copy.orchestrator.features.copy()
        
        # Train on selected features
        filtered_features = original_features[self.selected_features]
        system_copy.orchestrator.features = filtered_features
        
        self.xgb_trainer = train_enhanced_xgboost(
            system_copy,
            optimize_hyperparams=optimize_hyperparams,
            crisis_balanced=True,
            compute_shap=True,
            verbose=verbose
        )
        
        # Restore original features
        system_copy.orchestrator.features = original_features
        
        # PHASE 3: Retrain Anomaly Detectors
        if retrain_anomaly:
            if verbose:
                print(f"\n{'='*80}")
                print("PHASE 3: RETRAIN ANOMALY DETECTORS")
                print(f"{'='*80}")
            
            self._retrain_anomaly_detectors(filtered_features, verbose)
        
        # PHASE 4: Build Hybrid Forecaster
        if build_forecaster:
            if verbose:
                print(f"\n{'='*80}")
                print("PHASE 4: BUILD REGIME TRANSITION FORECASTER")
                print(f"{'='*80}")
            
            self._build_hybrid_forecaster(verbose)
        
        # PHASE 5: Export Results
        if verbose:
            print(f"\n{'='*80}")
            print("PHASE 5: EXPORT INTEGRATED OUTPUTS")
            print(f"{'='*80}")
        
        self._export_integrated_outputs(verbose)
        
        if verbose:
            print(f"\n{'='*80}")
            print("✅ XGBOOST INTEGRATION COMPLETE")
            print(f"{'='*80}")
    
    def _retrain_anomaly_detectors(self, filtered_features: pd.DataFrame, verbose: bool):
        """
        Retrain anomaly detection system with selected features.
        
        This improves signal-to-noise ratio by removing irrelevant features.
        """
        if verbose:
            print(f"   Retraining {len(self.system.orchestrator.anomaly_detector.detectors)} detectors...")
            print(f"   Using {len(filtered_features.columns)} selected features")
        
        # Store original detector
        original_detector = self.system.orchestrator.anomaly_detector
        
        # Create new detector
        from core.anomaly_detector import MultiDimensionalAnomalyDetector
        from config import RANDOM_STATE
        
        new_detector = MultiDimensionalAnomalyDetector(
            contamination=0.05,
            random_state=RANDOM_STATE
        )
        
        # Filter feature groups to only include selected features
        filtered_feature_groups = {}
        for group_name, feature_list in original_detector.feature_groups.items():
            filtered_group = [f for f in feature_list if f in filtered_features.columns]
            if len(filtered_group) >= 3:  # Need minimum 3 features per detector
                filtered_feature_groups[group_name] = filtered_group
        
        new_detector.feature_groups = filtered_feature_groups
        
        # Train new detector
        new_detector.train(filtered_features.fillna(0), verbose=False)
        
        # Generate historical scores
        scores = []
        for i in range(len(filtered_features)):
            result = new_detector.detect(filtered_features.iloc[[i]], verbose=False)
            scores.append(result['ensemble']['score'])
        
        new_detector.training_ensemble_scores = scores
        new_detector.statistical_thresholds = new_detector.calculate_statistical_thresholds()
        
        # Compare performance
        original_scores = self.system.orchestrator.historical_ensemble_scores
        new_scores = np.array(scores)
        
        if verbose:
            print(f"\n   Performance Comparison:")
            print(f"      Original detector: {len(original_detector.detectors)} detectors")
            print(f"      New detector: {len(new_detector.detectors)} detectors")
            print(f"      Original mean score: {original_scores.mean():.3f}")
            print(f"      New mean score: {new_scores.mean():.3f}")
            
            # Check crisis detection
            crisis_threshold = 0.78
            original_crisis_days = (original_scores > crisis_threshold).sum()
            new_crisis_days = (new_scores > crisis_threshold).sum()
            
            print(f"      Original crisis days (>{crisis_threshold}): {original_crisis_days}")
            print(f"      New crisis days (>{crisis_threshold}): {new_crisis_days}")
        
        # Replace system detector
        self.system.orchestrator.anomaly_detector = new_detector
        self.system.orchestrator.historical_ensemble_scores = new_scores
        
        if verbose:
            print(f"\n   ✅ Anomaly detectors retrained with selected features")
    
    def _build_hybrid_forecaster(self, verbose: bool):
        """
        Build hybrid regime transition forecaster.
        
        Combines:
        1. XGBoost regime probabilities (base prediction)
        2. Anomaly ensemble score (stress adjustment)
        3. Sequential features (momentum/streak)
        4. Historical transition matrix (prior probabilities)
        """
        from regime_transition_forecaster import RegimeTransitionForecaster
        
        self.forecaster = RegimeTransitionForecaster(
            xgb_trainer=self.xgb_trainer,
            anomaly_detector=self.system.orchestrator.anomaly_detector,
            regime_stats=self.system.orchestrator.regime_stats,
            selected_features=self.selected_features
        )
        
        if verbose:
            print(f"   ✅ Hybrid forecaster built")
            print(f"      Components: XGBoost + Anomaly + Sequential + Historical")
        
        # Test on current state
        current_features = self.system.orchestrator.features.iloc[[-1]]
        forecast = self.forecaster.forecast_5d_transition(current_features)
        
        if verbose:
            print(f"\n   📊 Current State Forecast:")
            print(f"      Current regime: {forecast['current_regime']}")
            print(f"      Transition probabilities:")
            for regime, prob in forecast['transition_probabilities'].items():
                print(f"         Regime {regime}: {prob:.1%}")
            print(f"      Confidence: {forecast['confidence']:.1%}")
    
    def _export_integrated_outputs(self, verbose: bool):
        """
        Export unified outputs for dashboards and external use.
        
        Files:
        - xgboost_models.json (model metadata)
        - selected_features.txt (feature list)
        - regime_forecast_live.json (current forecast)
        - xgboost_performance.json (validation metrics)
        """
        output_dir = Path('./json_data')
        output_dir.mkdir(exist_ok=True)
        
        # 1. Model metadata
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'xgboost_version': 'v2_academic_grade',
            'selected_features_count': len(self.selected_features),
            'regime_model_path': str(self.models_dir / 'regime_classifier_v2.json'),
            'range_model_path': str(self.models_dir / 'range_predictor_v2.json'),
            'anomaly_retrained': True,
            'forecaster_enabled': self.forecaster is not None,
        }
        
        with open(output_dir / 'xgboost_models.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # 2. Selected features
        with open(output_dir / 'selected_features_v2.txt', 'w') as f:
            f.write('\n'.join(self.selected_features))
        
        # 3. Current regime forecast (if forecaster built)
        if self.forecaster:
            current_features = self.system.orchestrator.features.iloc[[-1]]
            forecast = self.forecaster.forecast_5d_transition(current_features)
            
            with open(output_dir / 'regime_forecast_live.json', 'w') as f:
                json.dump(forecast, f, indent=2)
        
        # 4. Performance metrics
        validation_path = self.models_dir / 'validation_metrics_v2.json'
        if validation_path.exists():
            with open(validation_path, 'r') as f:
                validation_data = json.load(f)
            
            performance = {
                'regime_classification': {
                    'cv_accuracy': validation_data['regime_metrics']['cv_balanced_accuracy_mean'],
                    'cv_std': validation_data['regime_metrics']['cv_balanced_accuracy_std'],
                },
                'range_prediction': {
                    'cv_rmse': validation_data['range_metrics']['cv_rmse_mean'],
                    'cv_std': validation_data['range_metrics']['cv_rmse_std'],
                    'cv_directional_accuracy': validation_data['range_metrics']['cv_directional_mean'],
                },
                'crisis_validation': validation_data.get('crisis_validation', []),
                'multi_horizon_validation': validation_data.get('multi_horizon_validation', []),
            }
            
            with open(output_dir / 'xgboost_performance.json', 'w') as f:
                json.dump(performance, f, indent=2)
        
        if verbose:
            print(f"\n   ✅ Exported outputs to {output_dir}/")
            print(f"      • xgboost_models.json")
            print(f"      • selected_features_v2.txt")
            if self.forecaster:
                print(f"      • regime_forecast_live.json")
            print(f"      • xgboost_performance.json")


def main():
    """Main execution with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='XGBoost Integration for IntegratedMarketSystemV4'
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'features_only', 'forecast_only'],
        default='full',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable nested CV for hyperparameter optimization (slower, +2-3%% accuracy)'
    )
    
    parser.add_argument(
        '--skip-anomaly-retrain',
        action='store_true',
        help='Skip retraining anomaly detectors with selected features'
    )
    
    parser.add_argument(
        '--skip-forecaster',
        action='store_true',
        help='Skip building hybrid forecaster'
    )
    
    args = parser.parse_args()
    
    if not ENABLE_TRAINING:
        print(f"\n{'='*80}")
        print("⚠️ TRAINING DISABLED (config.ENABLE_TRAINING = False)")
        print("⚠️ Set ENABLE_TRAINING = True in config.py")
        print(f"{'='*80}\n")
        return
    
    # Initialize and train base system
    print(f"\n{'='*80}")
    print("STEP 0: TRAINING BASE SYSTEM")
    print(f"{'='*80}")
    
    system = IntegratedMarketSystemV4()
    system.train(years=TRAINING_YEARS, real_time_vix=True, verbose=False)
    
    print(f"\n✅ Base system trained ({len(system.orchestrator.features.columns)} features)")
    
    # Initialize integration manager
    integration_manager = XGBoostIntegrationManager(system)
    
    # Execute based on mode
    if args.mode == 'full':
        integration_manager.run_full_pipeline(
            optimize_hyperparams=args.optimize,
            retrain_anomaly=not args.skip_anomaly_retrain,
            build_forecaster=not args.skip_forecaster,
            verbose=True
        )
    
    elif args.mode == 'features_only':
        print(f"\n{'='*80}")
        print("MODE: FEATURE SELECTION ONLY")
        print(f"{'='*80}")
        
        selection_results = run_intelligent_feature_selection(
            system,
            min_stability=0.3,
            max_correlation=0.95,
            preserve_forward_indicators=True,
            verbose=True
        )
        
        print(f"\n✅ Selected {len(selection_results['selected_features'])} features")
        print(f"   Saved to: ./models/selected_features_v2.txt")
    
    elif args.mode == 'forecast_only':
        print(f"\n{'='*80}")
        print("MODE: FORECASTING ONLY")
        print(f"{'='*80}")
        
        # Load existing XGBoost models
        from xgboost_trainer_v2 import EnhancedXGBoostTrainer
        
        xgb_trainer = EnhancedXGBoostTrainer.load('./models')
        
        # Load selected features
        selected_features_path = Path('./models/selected_features_v2.txt')
        if not selected_features_path.exists():
            print("⚠️ Selected features not found. Run with --mode features_only first.")
            return
        
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f]
        
        integration_manager.xgb_trainer = xgb_trainer
        integration_manager.selected_features = selected_features
        
        integration_manager._build_hybrid_forecaster(verbose=True)
        integration_manager._export_integrated_outputs(verbose=True)
    
    print(f"\n{'='*80}")
    print("✅ INTEGRATION COMPLETE")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Review ./models/feature_importance_v2_overall.csv")
    print("  2. Check ./models/validation_metrics_v2.json for performance")
    print("  3. Use ./json_data/regime_forecast_live.json for dashboard")


if __name__ == "__main__":
    main()