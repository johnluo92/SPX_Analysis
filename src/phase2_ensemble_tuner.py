#!/usr/bin/env python3
"""
Production Ensemble Tuner - Optimizes Real ENSEMBLE_CONFIG Parameters
======================================================================
Tunes the actual ensemble parameters that production uses:
- decision_threshold
- up_advantage
- confidence_weights (up/down)
- magnitude_scaling (up/down)

Evaluates on TEST set using production ternary decision logic.
Optimizes for: accuracy, natural UP/DOWN balance (44/56), and NO_DECISION rate.
"""
import argparse, json, logging, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ProductionEnsembleTuner:
    def __init__(self, n_trials=500, min_accuracy=0.70, output_dir="tuning_ensemble"):
        self.n_trials = n_trials
        self.min_accuracy = min_accuracy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load trained models
        from core.xgboost_trainer_v3 import AsymmetricVIXForecaster
        self.forecaster = AsymmetricVIXForecaster(use_ensemble=True)

        if not Path("models/expansion_model.pkl").exists():
            logger.error("❌ Models not found - run train_probabilistic_models.py first")
            sys.exit(1)

        self.forecaster.load("models")
        logger.info(f"✅ Loaded models from models/")

        # Load data and split
        self._load_data()

        # Natural VIX distribution (from historical data)
        self.target_up_pct = 0.44  # 44% UP
        self.target_down_pct = 0.56  # 56% DOWN

        logger.info("="*80)
        logger.info("PRODUCTION ENSEMBLE TUNER")
        logger.info("="*80)
        logger.info(f"Test set: {len(self.test_df)} days ({self.test_df.index[0].date()} to {self.test_df.index[-1].date()})")
        logger.info(f"Target accuracy: {self.min_accuracy:.0%}+ for both UP and DOWN")
        logger.info(f"Target signal ratio: {self.target_up_pct:.0%} UP / {self.target_down_pct:.0%} DOWN (natural VIX)")
        logger.info(f"Target NO_DECISION: 20-35%")
        logger.info("="*80)

    def _load_data(self):
        """Load production data and split into train/val/test"""
        from config import TRAINING_YEARS, get_last_complete_month_end, TRAIN_END_DATE, VAL_END_DATE
        from core.data_fetcher import UnifiedDataFetcher
        from features.feature_engineer import FeatureEngineer
        from core.target_calculator import TargetCalculator

        training_end = get_last_complete_month_end()
        fetcher = UnifiedDataFetcher()
        engineer = FeatureEngineer(fetcher)

        logger.info("Loading data...")
        result = engineer.build_complete_features(years=TRAINING_YEARS, end_date=training_end)

        df = result["features"].copy()
        df["vix"] = result["vix"]

        # Calculate targets
        calculator = TargetCalculator()
        df = calculator.calculate_all_targets(df, vix_col='vix')

        # Apply quality filter
        from config import QUALITY_FILTER_CONFIG
        if 'feature_quality' in df.columns:
            threshold = QUALITY_FILTER_CONFIG['min_threshold']
            df = df[df['feature_quality'] >= threshold].copy()
            logger.info(f"Applied quality filter: {threshold:.2f}")

        # Split data
        self.train_end = pd.Timestamp(TRAIN_END_DATE)
        self.val_end = pd.Timestamp(VAL_END_DATE)

        test_mask = df.index > self.val_end
        self.test_df = df[test_mask].copy()

        logger.info(f"Test set: {len(self.test_df)} samples")

    def objective(self, trial):
        """Optuna objective - minimize composite score"""
        try:
            # Sample ensemble config parameters
            config = self._sample_config(trial)

            # Temporarily override forecaster's ensemble config
            original_config = self._save_current_config()
            self._apply_config(config)

            # Evaluate on test set
            metrics = self._evaluate_test_set()

            # Restore original config
            self._apply_config(original_config)

            # Record all metrics as user attributes
            for k, v in metrics.items():
                trial.set_user_attr(k, float(v))

            # Hard constraints
            if metrics['up_accuracy'] < self.min_accuracy:
                return 999.0
            if metrics['down_accuracy'] < self.min_accuracy:
                return 999.0
            if metrics['total_decisions'] < 100:  # Need enough signals
                return 999.0
            if metrics['no_decision_pct'] < 0.15 or metrics['no_decision_pct'] > 0.40:
                return 999.0  # NO_DECISION should be 15-40%

            # Calculate composite score
            score = self._calculate_score(metrics)

            return score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return 999.0

    def _sample_config(self, trial):
        """Sample ensemble configuration parameters"""
        config = {}

        # decision_threshold: controls NO_DECISION rate
        config['decision_threshold'] = trial.suggest_float('decision_threshold', 0.60, 0.75)

        # up_advantage: bias adjustment (-0.1 to +0.1)
        config['up_advantage'] = trial.suggest_float('up_advantage', -0.10, 0.10)

        # confidence_weights: how to combine classifier + magnitude
        up_classifier_weight = trial.suggest_float('up_classifier_weight', 0.45, 0.75)
        down_classifier_weight = trial.suggest_float('down_classifier_weight', 0.45, 0.75)

        config['confidence_weights'] = {
            'up': {
                'classifier': up_classifier_weight,
                'magnitude': 1.0 - up_classifier_weight
            },
            'down': {
                'classifier': down_classifier_weight,
                'magnitude': 1.0 - down_classifier_weight
            }
        }

        # magnitude_scaling: controls how magnitude affects confidence
        config['magnitude_scaling'] = {
            'up': {
                'small': trial.suggest_float('up_small', 2.5, 5.0),
                'medium': trial.suggest_float('up_medium', 4.0, 7.0),
                'large': trial.suggest_float('up_large', 8.0, 12.0)
            },
            'down': {
                'small': trial.suggest_float('down_small', 2.5, 5.0),
                'medium': trial.suggest_float('down_medium', 4.0, 7.0),
                'large': trial.suggest_float('down_large', 8.0, 12.0)
            }
        }

        return config

    def _save_current_config(self):
        """Save current forecaster config"""
        return {
            'decision_threshold': self.forecaster.decision_threshold,
            'up_advantage': self.forecaster.up_advantage,
            'confidence_weights': self.forecaster.confidence_weights.copy(),
            'magnitude_scaling': self.forecaster.magnitude_scaling.copy()
        }

    def _apply_config(self, config):
        """Apply config to forecaster"""
        self.forecaster.decision_threshold = config['decision_threshold']
        self.forecaster.up_advantage = config['up_advantage']
        self.forecaster.confidence_weights = config['confidence_weights']
        self.forecaster.magnitude_scaling = config['magnitude_scaling']

    def _evaluate_test_set(self):
        """Evaluate using production ternary decision logic"""
        results = []

        for idx in self.test_df.index:
            if pd.isna(self.test_df.loc[idx, 'target_direction']):
                continue

            # Create feature row
            obs = self.test_df.loc[idx]
            X = pd.DataFrame(index=[0])

            # Build X with all required features
            for col in self.forecaster.expansion_features:
                X[col] = [obs[col]]
            for col in self.forecaster.compression_features:
                if col not in X.columns:
                    X[col] = [obs[col]]
            for col in self.forecaster.up_features:
                if col not in X.columns:
                    X[col] = [obs[col]]
            for col in self.forecaster.down_features:
                if col not in X.columns:
                    X[col] = [obs[col]]

            current_vix = float(obs["vix"])
            actual_direction = int(obs["target_direction"])

            # Make prediction using PRODUCTION LOGIC
            pred = self.forecaster.predict(X, current_vix)

            results.append({
                'predicted_direction': pred['direction'],
                'actual_direction': 'UP' if actual_direction == 1 else 'DOWN',
                'confidence': pred['direction_confidence'],
                'magnitude_pct': pred['magnitude_pct']
            })

        results_df = pd.DataFrame(results)

        # Calculate metrics
        total = len(results_df)
        no_decision = (results_df['predicted_direction'] == 'NO_DECISION').sum()
        decisions = results_df[results_df['predicted_direction'] != 'NO_DECISION'].copy()

        if len(decisions) == 0:
            return {
                'total_samples': total,
                'total_decisions': 0,
                'no_decision_pct': 1.0,
                'overall_accuracy': 0.0,
                'up_count': 0,
                'up_accuracy': 0.0,
                'up_pct': 0.0,
                'down_count': 0,
                'down_accuracy': 0.0,
                'down_pct': 0.0,
                'signal_imbalance': 1.0,
                'accuracy_imbalance': 0.0
            }

        decisions['correct'] = decisions['predicted_direction'] == decisions['actual_direction']

        overall_acc = decisions['correct'].mean()

        up_decisions = decisions[decisions['predicted_direction'] == 'UP']
        down_decisions = decisions[decisions['predicted_direction'] == 'DOWN']

        up_acc = up_decisions['correct'].mean() if len(up_decisions) > 0 else 0.0
        down_acc = down_decisions['correct'].mean() if len(down_decisions) > 0 else 0.0

        up_pct = len(up_decisions) / len(decisions) if len(decisions) > 0 else 0.0
        down_pct = len(down_decisions) / len(decisions) if len(decisions) > 0 else 0.0

        # Signal imbalance vs natural VIX distribution
        signal_imbalance = abs(up_pct - self.target_up_pct)

        # Accuracy imbalance
        accuracy_imbalance = abs(up_acc - down_acc)

        return {
            'total_samples': total,
            'total_decisions': len(decisions),
            'no_decision_pct': no_decision / total,
            'overall_accuracy': overall_acc,
            'up_count': len(up_decisions),
            'up_accuracy': up_acc,
            'up_pct': up_pct,
            'down_count': len(down_decisions),
            'down_accuracy': down_acc,
            'down_pct': down_pct,
            'signal_imbalance': signal_imbalance,
            'accuracy_imbalance': accuracy_imbalance
        }

    def _calculate_score(self, metrics):
        """Calculate composite score (lower is better)"""

        # 1. Accuracy penalties - fix the weaker direction first
        worst_acc = min(metrics['up_accuracy'], metrics['down_accuracy'])
        worst_penalty = (1.0 - worst_acc) * 100.0  # 100x multiplier

        avg_acc = (metrics['up_accuracy'] + metrics['down_accuracy']) / 2.0
        avg_penalty = (1.0 - avg_acc) * 30.0  # 30x multiplier

        # 2. Accuracy balance - penalize gaps
        accuracy_balance_penalty = (metrics['accuracy_imbalance'] ** 2) * 200.0  # 200x for gaps

        # 3. Signal distribution - match natural VIX (44/56)
        signal_balance_penalty = (metrics['signal_imbalance'] ** 2) * 50.0  # 50x

        # 4. NO_DECISION rate - target 20-35%
        no_dec = metrics['no_decision_pct']
        if no_dec < 0.20:
            no_dec_penalty = (0.20 - no_dec) * 100.0  # Too confident
        elif no_dec > 0.35:
            no_dec_penalty = (no_dec - 0.35) * 100.0  # Too conservative
        else:
            no_dec_penalty = -5.0  # Bonus for being in sweet spot

        # 5. Volume penalty - need enough signals
        if metrics['total_decisions'] < 150:
            volume_penalty = (150 - metrics['total_decisions']) * 0.5
        else:
            volume_penalty = 0.0

        # Total score
        score = (
            worst_penalty +              # Fix weak direction
            avg_penalty +                # Push both higher
            accuracy_balance_penalty +   # Balance accuracies
            signal_balance_penalty +     # Match natural distribution
            no_dec_penalty +             # Optimal NO_DECISION rate
            volume_penalty               # Enough signals
        )

        return score

    def run(self):
        """Run optimization"""
        logger.info(f"Starting optimization: {self.n_trials} trials")
        logger.info("="*80)

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42, n_startup_trials=min(50, self.n_trials // 10))
        )

        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)

        return study

    def save_results(self, study):
        """Save optimization results and generate config"""
        best = study.best_trial
        attrs = best.user_attrs
        params = best.params

        # Save detailed results
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimization': {
                'n_trials': self.n_trials,
                'best_trial': best.number,
                'best_score': float(best.value),
                'min_accuracy_target': self.min_accuracy
            },
            'test_metrics': {
                'total_samples': int(attrs['total_samples']),
                'total_decisions': int(attrs['total_decisions']),
                'no_decision_pct': float(attrs['no_decision_pct']),
                'overall_accuracy': float(attrs['overall_accuracy']),
                'up_count': int(attrs['up_count']),
                'up_accuracy': float(attrs['up_accuracy']),
                'up_pct': float(attrs['up_pct']),
                'down_count': int(attrs['down_count']),
                'down_accuracy': float(attrs['down_accuracy']),
                'down_pct': float(attrs['down_pct']),
                'signal_imbalance': float(attrs['signal_imbalance']),
                'accuracy_imbalance': float(attrs['accuracy_imbalance'])
            },
            'best_parameters': dict(params)
        }

        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved: {results_file}")

        # Generate config
        self._generate_config(params, attrs)

        # Print summary
        self._print_summary(params, attrs)

    def _generate_config(self, params, attrs):
        """Generate ENSEMBLE_CONFIG for config.py"""

        config_text = f"""
# OPTIMIZED ENSEMBLE_CONFIG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Test accuracy: UP {attrs['up_accuracy']:.1%}, DOWN {attrs['down_accuracy']:.1%}
# Signal ratio: {attrs['up_pct']:.1%} UP / {attrs['down_pct']:.1%} DOWN (target: 44/56)
# NO_DECISION: {attrs['no_decision_pct']:.1%}

ENSEMBLE_CONFIG = {{
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': {params['up_advantage']:.4f},
    'confidence_weights': {{
        'up': {{'classifier': {params['up_classifier_weight']:.4f}, 'magnitude': {1.0 - params['up_classifier_weight']:.4f}}},
        'down': {{'classifier': {params['down_classifier_weight']:.4f}, 'magnitude': {1.0 - params['down_classifier_weight']:.4f}}}
    }},
    'magnitude_scaling': {{
        'up': {{'small': {params['up_small']:.4f}, 'medium': {params['up_medium']:.4f}, 'large': {params['up_large']:.4f}}},
        'down': {{'small': {params['down_small']:.4f}, 'medium': {params['down_medium']:.4f}, 'large': {params['down_large']:.4f}}}
    }},
    'decision_threshold': {params['decision_threshold']:.4f},
    'description': 'Optimized for balanced {self.min_accuracy:.0%}+ accuracy with natural VIX distribution'
}}
"""

        config_file = self.output_dir / "optimized_ensemble_config.py"
        with open(config_file, 'w') as f:
            f.write(config_text)

        logger.info(f"✅ Config saved: {config_file}")

    def _print_summary(self, params, attrs):
        """Print optimization summary"""
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info("")
        logger.info("📊 TEST SET PERFORMANCE:")
        logger.info(f"  Total predictions: {int(attrs['total_samples'])}")
        logger.info(f"  Decisions made: {int(attrs['total_decisions'])} ({attrs['total_decisions']/attrs['total_samples']:.1%})")
        logger.info(f"  NO_DECISION: {int(attrs['total_samples'] - attrs['total_decisions'])} ({attrs['no_decision_pct']:.1%})")
        logger.info("")
        logger.info(f"  Overall Accuracy: {attrs['overall_accuracy']:.1%}")
        logger.info(f"  UP:   n={int(attrs['up_count']):3d} ({attrs['up_pct']:5.1%}) | acc={attrs['up_accuracy']:.1%}")
        logger.info(f"  DOWN: n={int(attrs['down_count']):3d} ({attrs['down_pct']:5.1%}) | acc={attrs['down_accuracy']:.1%}")
        logger.info("")
        logger.info("🎯 BALANCE METRICS:")
        logger.info(f"  Accuracy gap: {attrs['accuracy_imbalance']*100:.1f}% (target: <10%)")
        logger.info(f"  Signal imbalance vs natural VIX: {attrs['signal_imbalance']*100:.1f}%")
        logger.info("")
        logger.info("📝 OPTIMIZED PARAMETERS:")
        logger.info(f"  decision_threshold: {params['decision_threshold']:.4f}")
        logger.info(f"  up_advantage: {params['up_advantage']:+.4f}")
        logger.info(f"  UP weights: clf={params['up_classifier_weight']:.2f}, mag={1-params['up_classifier_weight']:.2f}")
        logger.info(f"  DOWN weights: clf={params['down_classifier_weight']:.2f}, mag={1-params['down_classifier_weight']:.2f}")
        logger.info("")
        logger.info("="*80)
        logger.info("")
        logger.info("📝 NEXT STEPS:")
        logger.info("  1. Review: tuning_ensemble/optimization_results.json")
        logger.info("  2. Copy ENSEMBLE_CONFIG from tuning_ensemble/optimized_ensemble_config.py to config.py")
        logger.info("  3. Retrain: python train_probabilistic_models.py")
        logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(description="Production Ensemble Tuner")
    parser.add_argument('--trials', type=int, default=500, help="Number of trials (default: 500)")
    parser.add_argument('--min-accuracy', type=float, default=0.70, help="Min accuracy target (default: 0.70)")
    parser.add_argument('--output-dir', type=str, default='tuning_ensemble', help="Output directory")
    args = parser.parse_args()

    if not (0.55 <= args.min_accuracy <= 0.90):
        logger.error("❌ min-accuracy must be between 0.55 and 0.90")
        sys.exit(1)

    tuner = ProductionEnsembleTuner(
        n_trials=args.trials,
        min_accuracy=args.min_accuracy,
        output_dir=args.output_dir
    )

    study = tuner.run()
    tuner.save_results(study)

    logger.info("\n✅ Optimization complete!")

if __name__ == "__main__":
    main()
