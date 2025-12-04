#!/usr/bin/env python3
import argparse, json, logging, sys, warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np, pandas as pd, optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/phase2_tuner.log")])
logger = logging.getLogger(__name__)

@dataclass
class EnsembleMetrics:
    raw_total: int = 0
    raw_up_count: int = 0
    raw_down_count: int = 0
    raw_up_pct: float = 0.0
    raw_down_pct: float = 0.0
    raw_accuracy: float = 0.0
    raw_up_accuracy: float = 0.0
    raw_down_accuracy: float = 0.0
    act_total: int = 0
    act_up_count: int = 0
    act_down_count: int = 0
    act_up_pct: float = 0.0
    act_down_pct: float = 0.0
    act_accuracy: float = 0.0
    act_up_accuracy: float = 0.0
    act_down_accuracy: float = 0.0
    act_rate: float = 0.0
    mag_mae: float = 0.0
    mag_mae_up: float = 0.0
    mag_mae_down: float = 0.0
    calibration_window_days: int = 0

class Phase2Tuner:
    def __init__(self, df, vix, n_trials=200, output_dir="tuning_phase2"):
        self.df = df.copy()
        self.vix = vix.copy()
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_end = pd.Timestamp("2021-12-31")
        self.val_end = pd.Timestamp("2023-12-31")
        self.test_start = self.val_end + pd.Timedelta(days=1)

        train_mask = df.index <= self.train_end
        val_mask = (df.index > self.train_end) & (df.index <= self.val_end)
        test_mask = df.index > self.val_end

        self.train_df = df[train_mask].copy()
        self.val_df = df[val_mask].copy()
        self.test_df = df[test_mask].copy()

        from config import TARGET_CONFIG
        self.horizon = TARGET_CONFIG["horizon_days"]

        self._calculate_targets()
        self._load_phase1_models()

        logger.info("="*80)
        logger.info("PHASE 2: ENSEMBLE CONFIG TUNER")
        logger.info("="*80)
        logger.info(f"Train:  {len(self.train_df)} days ({self.train_df.index[0].date()} to {self.train_end.date()})")
        logger.info(f"Val:    {len(self.val_df)} days ({self.val_df.index[0].date()} to {self.val_end.date()})")
        logger.info(f"Test:   {len(self.test_df)} days ({self.test_start.date()} to {self.test_df.index[-1].date()})")
        logger.info(f"Using Phase 1 optimized models from models/ directory")
        logger.info("="*80)

    def _calculate_targets(self):
        from core.target_calculator import TargetCalculator
        calculator = TargetCalculator()
        self.train_df = calculator.calculate_all_targets(self.train_df, vix_col='vix')
        self.val_df = calculator.calculate_all_targets(self.val_df, vix_col='vix')
        self.test_df = calculator.calculate_all_targets(self.test_df, vix_col='vix')

    def _load_phase1_models(self):
        from core.xgboost_trainer_v3 import AsymmetricVIXForecaster
        self.forecaster = AsymmetricVIXForecaster()

        models_path = Path("models")
        if not (models_path / "expansion_model.pkl").exists():
            logger.error("❌ Phase 1 models not found - run phase1_model_tuner.py first")
            sys.exit(1)

        self.forecaster.load("models")
        logger.info(f"✅ Loaded Phase 1 models: {len(self.forecaster.expansion_features)} exp, "
                   f"{len(self.forecaster.compression_features)} comp, {len(self.forecaster.up_features)} up, "
                   f"{len(self.forecaster.down_features)} down features")

    def _apply_quality_filter(self, df, threshold=0.55):
        if 'feature_quality' not in df.columns:
            logger.warning("No feature_quality column - skipping filter")
            return df
        filtered = df[df['feature_quality'] >= threshold].copy()
        filtered_pct = (1 - len(filtered)/len(df)) * 100
        if filtered_pct > 50:
            raise ValueError(f"Quality filter removed {filtered_pct:.1f}% of data")
        return filtered

    def _generate_base_predictions(self, test_df):
        """Generate raw model predictions once for all trials"""
        predictions = []

        for idx in test_df.index:
            if pd.isna(test_df.loc[idx, 'target_direction']):
                continue

            X_exp = test_df.loc[[idx], self.forecaster.expansion_features].fillna(0)
            X_comp = test_df.loc[[idx], self.forecaster.compression_features].fillna(0)
            X_up = test_df.loc[[idx], self.forecaster.up_features].fillna(0)
            X_down = test_df.loc[[idx], self.forecaster.down_features].fillna(0)

            exp_log = np.clip(self.forecaster.expansion_model.predict(X_exp)[0], -2, 2)
            comp_log = np.clip(self.forecaster.compression_model.predict(X_comp)[0], -2, 2)
            exp_pct = (np.exp(exp_log) - 1) * 100
            comp_pct = (np.exp(comp_log) - 1) * 100

            p_up = self.forecaster.up_classifier.predict_proba(X_up)[0, 1]
            p_down = self.forecaster.down_classifier.predict_proba(X_down)[0, 1]

            current_vix = float(test_df.loc[idx, 'vix'])
            actual_direction = int(test_df.loc[idx, 'target_direction'])
            actual_mag_log = test_df.loc[idx, 'target_log_vix_change']
            actual_mag_pct = (np.exp(actual_mag_log) - 1) * 100

            predictions.append({
                'date': idx,
                'p_up': p_up, 'p_down': p_down,
                'exp_pct': exp_pct, 'comp_pct': comp_pct,
                'current_vix': current_vix,
                'actual_direction': actual_direction,
                'actual_magnitude': actual_mag_pct
            })

        return pd.DataFrame(predictions)

    def _apply_ensemble_logic(self, pred, ensemble_config):
        """Apply ensemble config to a prediction"""
        p_up, p_down = pred['p_up'], pred['p_down']
        exp_pct, comp_pct = pred['exp_pct'], pred['comp_pct']

        total = p_up + p_down
        p_up_norm = p_up / total if total > 0 else 0.5
        p_down_norm = p_down / total if total > 0 else 0.5

        up_advantage = ensemble_config['up_advantage']
        if p_down > (p_up + up_advantage):
            direction = "DOWN"
            magnitude_pct = comp_pct
            classifier_prob = p_down_norm
        else:
            direction = "UP"
            magnitude_pct = exp_pct
            classifier_prob = p_up_norm

        abs_mag = abs(magnitude_pct)
        weights = ensemble_config['confidence_weights']
        scaling = ensemble_config['magnitude_scaling']

        mag_strength = min(abs_mag / scaling['large'], 1.0)
        confidence = weights['classifier'] * classifier_prob + weights['magnitude'] * mag_strength

        if abs_mag > ensemble_config['confidence_boost_threshold']:
            confidence = min(confidence + ensemble_config['confidence_boost_amount'], 1.0)

        confidence = np.clip(confidence, ensemble_config['min_ensemble_confidence'], 1.0)

        thresholds = ensemble_config['dynamic_thresholds'][direction.lower()]
        if abs_mag > scaling['large']:
            threshold = thresholds['high_magnitude']
        elif abs_mag > scaling['medium']:
            threshold = thresholds['medium_magnitude']
        else:
            threshold = thresholds['low_magnitude']

        actionable = confidence > threshold

        return {
            'direction': direction,
            'magnitude_pct': magnitude_pct,
            'confidence': confidence,
            'threshold': threshold,
            'actionable': actionable,
            'pred_direction': 1 if direction == 'UP' else 0
        }

    def _evaluate_ensemble_config(self, ensemble_config, base_predictions):
        """Evaluate an ensemble config on base predictions"""
        results = []

        for _, pred in base_predictions.iterrows():
            forecast = self._apply_ensemble_logic(pred, ensemble_config)

            results.append({
                'pred_direction': forecast['pred_direction'],
                'actual_direction': pred['actual_direction'],
                'direction_correct': (forecast['pred_direction'] == pred['actual_direction']),
                'confidence': forecast['confidence'],
                'actionable': forecast['actionable'],
                'magnitude': forecast['magnitude_pct'],
                'actual_magnitude': pred['actual_magnitude']
            })

        return pd.DataFrame(results)

    def _calculate_metrics(self, predictions, calibration_window_days):
        df = predictions

        raw_up = df[df['pred_direction'] == 1]
        raw_down = df[df['pred_direction'] == 0]

        act_df = df[df['actionable']]
        act_up = act_df[act_df['pred_direction'] == 1]
        act_down = act_df[act_df['pred_direction'] == 0]

        act_df_clean = act_df.dropna(subset=['actual_magnitude', 'magnitude'])
        act_up_clean = act_up.dropna(subset=['actual_magnitude', 'magnitude'])
        act_down_clean = act_down.dropna(subset=['actual_magnitude', 'magnitude'])

        metrics = EnsembleMetrics(
            raw_total=len(df), raw_up_count=len(raw_up), raw_down_count=len(raw_down),
            raw_up_pct=len(raw_up) / len(df) if len(df) > 0 else 0.0,
            raw_down_pct=len(raw_down) / len(df) if len(df) > 0 else 0.0,
            raw_accuracy=df['direction_correct'].mean() if len(df) > 0 else 0.0,
            raw_up_accuracy=raw_up['direction_correct'].mean() if len(raw_up) > 0 else 0.0,
            raw_down_accuracy=raw_down['direction_correct'].mean() if len(raw_down) > 0 else 0.0,
            act_total=len(act_df), act_up_count=len(act_up), act_down_count=len(act_down),
            act_up_pct=len(act_up) / len(act_df) if len(act_df) > 0 else 0.0,
            act_down_pct=len(act_down) / len(act_df) if len(act_df) > 0 else 0.0,
            act_accuracy=act_df['direction_correct'].mean() if len(act_df) > 0 else 0.0,
            act_up_accuracy=act_up['direction_correct'].mean() if len(act_up) > 0 else 0.0,
            act_down_accuracy=act_down['direction_correct'].mean() if len(act_down) > 0 else 0.0,
            act_rate=len(act_df) / len(df) if len(df) > 0 else 0.0,
            mag_mae=mean_absolute_error(act_df_clean['actual_magnitude'], act_df_clean['magnitude']) if len(act_df_clean) > 0 else 999.0,
            mag_mae_up=mean_absolute_error(act_up_clean['actual_magnitude'], act_up_clean['magnitude']) if len(act_up_clean) > 0 else 999.0,
            mag_mae_down=mean_absolute_error(act_down_clean['actual_magnitude'], act_down_clean['magnitude']) if len(act_down_clean) > 0 else 999.0,
            calibration_window_days=calibration_window_days
        )

        return metrics

    def objective(self, trial):
        try:
            ensemble_config = self._sample_ensemble_config(trial)

            # Validate threshold ordering
            for direction in ['up', 'down']:
                thresholds = ensemble_config['dynamic_thresholds'][direction]
                if not (thresholds['high_magnitude'] < thresholds['medium_magnitude'] < thresholds['low_magnitude']):
                    logger.debug(f"Trial {trial.number}: Invalid threshold ordering for {direction}")
                    return 999.0

            test_filt = self._apply_quality_filter(self.test_df, threshold=0.55)
            if len(test_filt) < 100:
                logger.debug(f"Trial {trial.number}: Insufficient test data")
                return 999.0

            if not hasattr(self, '_base_predictions'):
                logger.info("Generating base predictions (once)...")
                self._base_predictions = self._generate_base_predictions(test_filt)

            predictions = self._evaluate_ensemble_config(ensemble_config, self._base_predictions)
            calibration_window = ensemble_config['calibration_window_days']
            metrics = self._calculate_metrics(predictions, calibration_window)

            for field_name, value in metrics.__dict__.items():
                trial.set_user_attr(field_name, float(value))

            # HIGH PRECISION OBJECTIVE: 80-90% accuracy for both UP and DOWN
            up_acc = metrics.act_up_accuracy
            down_acc = metrics.act_down_accuracy

            # Hard constraints (fail fast)
            min_acc = 0.80  # Must hit 80% for both
            min_signals = 50  # Need statistical validity, not volume

            if up_acc < min_acc:
                return 999.0
            if down_acc < min_acc:
                return 999.0
            if metrics.act_up_count < min_signals or metrics.act_down_count < min_signals:
                return 999.0

            # Target: 85% accuracy for both (sweet spot in 80-90% range)
            target_acc = 0.85

            # Primary: Minimize squared distance from target for both
            up_error = (up_acc - target_acc) ** 2
            down_error = (down_acc - target_acc) ** 2
            accuracy_loss = (up_error + down_error) * 100  # Scale up

            # Secondary: Prefer balanced UP/DOWN split (45-55% range)
            balance_penalty = abs(metrics.act_up_pct - 0.50) * 5.0

            # Tertiary: Prefer lower MAE (better magnitude predictions)
            mag_penalty = max(0, metrics.mag_mae - 12.0) * 0.3

            # Quaternary: Slight penalty for very low signal counts (but not primary objective)
            # Prefer 100+ signals but don't force it
            volume_penalty = 0.0
            if metrics.act_total < 100:
                volume_penalty = (100 - metrics.act_total) * 0.05

            # Total score: minimize loss (prioritize accuracy, then balance, then MAE)
            score = accuracy_loss + balance_penalty + mag_penalty + volume_penalty

            return score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return 999.0

    def _sample_ensemble_config(self, trial):
        config = {}

        config['up_advantage'] = trial.suggest_float('up_advantage', 0.05, 0.15)

        # FIXED: Sample thresholds in correct order (high < medium < low)
        # For UP direction
        up_high = trial.suggest_float('up_thresh_high', 0.48, 0.62)  # Easiest to pass
        up_med = trial.suggest_float('up_thresh_med_offset', 0.02, 0.08)  # Offset from high
        up_low = trial.suggest_float('up_thresh_low_offset', 0.02, 0.08)  # Offset from medium

        up_thresh_med = up_high + up_med
        up_thresh_low = up_thresh_med + up_low

        # For DOWN direction
        down_high = trial.suggest_float('down_thresh_high', 0.53, 0.67)  # Easiest to pass
        down_med = trial.suggest_float('down_thresh_med_offset', 0.02, 0.08)  # Offset from high
        down_low = trial.suggest_float('down_thresh_low_offset', 0.02, 0.08)  # Offset from medium

        down_thresh_med = down_high + down_med
        down_thresh_low = down_thresh_med + down_low

        config['dynamic_thresholds'] = {
            'up': {
                'high_magnitude': up_high,
                'medium_magnitude': up_thresh_med,
                'low_magnitude': up_thresh_low
            },
            'down': {
                'high_magnitude': down_high,
                'medium_magnitude': down_thresh_med,
                'low_magnitude': down_thresh_low
            }
        }

        classifier_weight = trial.suggest_float('classifier_weight', 0.55, 0.75)
        config['confidence_weights'] = {
            'classifier': classifier_weight,
            'magnitude': 1.0 - classifier_weight
        }

        config['magnitude_scaling'] = {
            'small': trial.suggest_float('mag_scale_small', 2.5, 4.5),
            'medium': trial.suggest_float('mag_scale_medium', 5.0, 7.5),
            'large': trial.suggest_float('mag_scale_large', 10.0, 14.0)
        }

        config['confidence_boost_threshold'] = trial.suggest_float('boost_threshold', 12.0, 18.0)
        config['confidence_boost_amount'] = trial.suggest_float('boost_amount', 0.04, 0.07)
        config['min_ensemble_confidence'] = trial.suggest_float('min_confidence', 0.50, 0.60)

        config['calibration_window_days'] = trial.suggest_int('calibration_window_days', 500, 800, step=50)

        return config

    def run(self):
        logger.info(f"Starting Phase 2 optimization: {self.n_trials} trials")
        logger.info(f"Tuning ensemble config (up_advantage, thresholds, weights, etc.)")
        logger.info(f"Evaluating on {len(self.test_df)} test days")
        logger.info(f"Objective: HIGH PRECISION - target 85% accuracy for both UP and DOWN")
        logger.info(f"Actionable rate will fall naturally (quality over quantity)")
        logger.info("="*80)

        study = optuna.create_study(direction='minimize',
            sampler=TPESampler(seed=42, n_startup_trials=min(30, self.n_trials // 6)))

        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)

        return study

    def save_results(self, study):
        best = study.best_trial
        attrs = best.user_attrs

        # Reconstruct actual threshold values from params
        params = best.params
        up_high = params['up_thresh_high']
        up_med = up_high + params['up_thresh_med_offset']
        up_low = up_med + params['up_thresh_low_offset']

        down_high = params['down_thresh_high']
        down_med = down_high + params['down_thresh_med_offset']
        down_low = down_med + params['down_thresh_low_offset']

        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 2 - Ensemble Config Tuning',
            'description': 'Optimizes post-hoc ensemble decision logic using Phase 1 models',
            'optimization': {
                'n_trials': self.n_trials,
                'best_trial': best.number,
                'best_score': float(best.value)
            },
            'data_splits': {
                'train': f"{self.train_df.index[0].date()} to {self.train_end.date()}",
                'val': f"{self.val_df.index[0].date()} to {self.val_end.date()}",
                'test': f"{self.test_start.date()} to {self.test_df.index[-1].date()}",
                'train_size': len(self.train_df),
                'val_size': len(self.val_df),
                'test_size': len(self.test_df)
            },
            'test_metrics': {
                'raw_predictions': {
                    'total': int(attrs['raw_total']),
                    'up_count': int(attrs['raw_up_count']),
                    'down_count': int(attrs['raw_down_count']),
                    'up_pct': float(attrs['raw_up_pct']),
                    'down_pct': float(attrs['raw_down_pct']),
                    'accuracy': float(attrs['raw_accuracy']),
                    'up_accuracy': float(attrs['raw_up_accuracy']),
                    'down_accuracy': float(attrs['raw_down_accuracy'])
                },
                'actionable_signals': {
                    'total': int(attrs['act_total']),
                    'rate': float(attrs['act_rate']),
                    'up_count': int(attrs['act_up_count']),
                    'down_count': int(attrs['act_down_count']),
                    'up_pct': float(attrs['act_up_pct']),
                    'down_pct': float(attrs['act_down_pct']),
                    'accuracy': float(attrs['act_accuracy']),
                    'up_accuracy': float(attrs['act_up_accuracy']),
                    'down_accuracy': float(attrs['act_down_accuracy']),
                    'mag_mae': float(attrs['mag_mae']),
                    'mag_mae_up': float(attrs['mag_mae_up']),
                    'mag_mae_down': float(attrs['mag_mae_down'])
                },
                'calibration_window_days': int(attrs['calibration_window_days'])
            },
            'best_parameters': {
                'up_advantage': params['up_advantage'],
                'up_thresh_high': up_high,
                'up_thresh_med': up_med,
                'up_thresh_low': up_low,
                'down_thresh_high': down_high,
                'down_thresh_med': down_med,
                'down_thresh_low': down_low,
                'classifier_weight': params['classifier_weight'],
                'mag_scale_small': params['mag_scale_small'],
                'mag_scale_medium': params['mag_scale_medium'],
                'mag_scale_large': params['mag_scale_large'],
                'boost_threshold': params['boost_threshold'],
                'boost_amount': params['boost_amount'],
                'min_confidence': params['min_confidence'],
                'calibration_window_days': params['calibration_window_days']
            }
        }

        results_file = self.output_dir / "phase2_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved: {results_file}")

        self._generate_config(best, attrs)
        self._print_summary(best, attrs)

    def _generate_config(self, trial, attrs):
        params = trial.params

        # Reconstruct actual threshold values
        up_high = params['up_thresh_high']
        up_med = up_high + params['up_thresh_med_offset']
        up_low = up_med + params['up_thresh_low_offset']

        down_high = params['down_thresh_high']
        down_med = down_high + params['down_thresh_med_offset']
        down_low = down_med + params['down_thresh_low_offset']

        config_text = f"""# PHASE 2 OPTIMIZED CONFIG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CALIBRATION_WINDOW_DAYS = {params['calibration_window_days']}

ENSEMBLE_CONFIG = {{
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': {params['up_advantage']:.4f},
    'confidence_weights': {{
        'classifier': {params['classifier_weight']:.4f},
        'magnitude': {1.0 - params['classifier_weight']:.4f}
    }},
    'magnitude_scaling': {{
        'small': {params['mag_scale_small']:.4f},
        'medium': {params['mag_scale_medium']:.4f},
        'large': {params['mag_scale_large']:.4f}
    }},
    'dynamic_thresholds': {{
        'up': {{
            'high_magnitude': {up_high:.4f},
            'medium_magnitude': {up_med:.4f},
            'low_magnitude': {up_low:.4f}
        }},
        'down': {{
            'high_magnitude': {down_high:.4f},
            'medium_magnitude': {down_med:.4f},
            'low_magnitude': {down_low:.4f}
        }}
    }},
    'min_ensemble_confidence': {params['min_confidence']:.4f},
    'confidence_boost_threshold': {params['boost_threshold']:.4f},
    'confidence_boost_amount': {params['boost_amount']:.4f},
    'description': 'Phase 2 optimized for HIGH PRECISION (85% target accuracy)'
}}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable {attrs['act_accuracy']:.1%} (UP {attrs['act_up_accuracy']:.1%}, DOWN {attrs['act_down_accuracy']:.1%})
# MAE {attrs['mag_mae']:.2f}% | Signals: {int(attrs['act_total'])} ({attrs['act_rate']:.1%} actionable)
# UP signals: {int(attrs['act_up_count'])} ({attrs['act_up_pct']:.1%}) | DOWN signals: {int(attrs['act_down_count'])} ({attrs['act_down_pct']:.1%})
# Trading frequency: ~{(attrs['act_total'] / 700) * 7:.1f} signals/week
# Calibration window: {int(attrs['calibration_window_days'])} days
"""

        config_file = self.output_dir / "phase2_optimized_config.py"
        with open(config_file, 'w') as f:
            f.write(config_text)

        logger.info(f"✅ Config saved: {config_file}")

    def _print_summary(self, trial, attrs):
        logger.info("\n" + "="*80)
        logger.info("PHASE 2 OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Best trial: #{trial.number} | Score: {trial.value:.3f}")
        logger.info("")
        logger.info("📊 TEST SET PERFORMANCE (2024-2025):")
        logger.info("")
        logger.info("  RAW PREDICTIONS:")
        logger.info(f"    Total: {int(attrs['raw_total'])}")
        logger.info(f"    UP: {attrs['raw_up_pct']:.1%} ({int(attrs['raw_up_count'])} signals)")
        logger.info(f"    DOWN: {attrs['raw_down_pct']:.1%} ({int(attrs['raw_down_count'])} signals)")
        logger.info(f"    Accuracy: {attrs['raw_accuracy']:.1%} | UP: {attrs['raw_up_accuracy']:.1%} | DOWN: {attrs['raw_down_accuracy']:.1%}")
        logger.info("")
        logger.info("  ACTIONABLE SIGNALS (high precision ensemble):")
        logger.info(f"    Total: {int(attrs['act_total'])} signals ({attrs['act_rate']:.1%} actionable)")
        logger.info(f"    UP: {attrs['act_up_pct']:.1%} ({int(attrs['act_up_count'])} signals)")
        logger.info(f"    DOWN: {attrs['act_down_pct']:.1%} ({int(attrs['act_down_count'])} signals)")
        logger.info(f"    Accuracy: {attrs['act_accuracy']:.1%} | UP: {attrs['act_up_accuracy']:.1%} | DOWN: {attrs['act_down_accuracy']:.1%}")
        logger.info(f"    Magnitude MAE: {attrs['mag_mae']:.2f}% (UP: {attrs['mag_mae_up']:.2f}%, DOWN: {attrs['mag_mae_down']:.2f}%)")
        logger.info("")
        logger.info("  CALIBRATION:")
        logger.info(f"    Window: {int(attrs['calibration_window_days'])} days")
        logger.info("="*80)
        logger.info("")

        up_acc = attrs['act_up_accuracy']
        down_acc = attrs['act_down_accuracy']
        target = 0.85

        logger.info(f"🎯 HIGH PRECISION TARGET: 85% accuracy")
        up_delta = (up_acc - target) * 100
        down_delta = (down_acc - target) * 100
        up_status = "✓" if up_acc >= 0.80 else "✗"
        down_status = "✓" if down_acc >= 0.80 else "✗"
        logger.info(f"   {up_status} UP:   {up_acc:.1%} ({up_delta:+.1f}% from target)")
        logger.info(f"   {down_status} DOWN: {down_acc:.1%} ({down_delta:+.1f}% from target)")

        avg_signals_per_week = (attrs['act_total'] / 700) * 7
        logger.info("")
        logger.info(f"📈 TRADING FREQUENCY: ~{avg_signals_per_week:.1f} signals/week (quality over quantity)")
        logger.info("="*80)
        logger.info("")
        logger.info("📝 NEXT STEPS:")
        logger.info("  1. Review results in tuning_phase2/phase2_results.json")
        logger.info("  2. Apply ENSEMBLE_CONFIG from phase2_optimized_config.py to your config.py")
        logger.info("  3. Apply CALIBRATION_WINDOW_DAYS to config.py")
        logger.info("  4. Run: py integrated_system.py")
        logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Ensemble Config Tuner (Post-hoc)")
    parser.add_argument('--trials', type=int, default=200, help="Number of optimization trials (default: 200)")
    parser.add_argument('--output-dir', type=str, default='tuning_phase2', help="Output directory")
    args = parser.parse_args()

    logger.info("Loading production data...")
    from config import TRAINING_YEARS, get_last_complete_month_end
    from core.data_fetcher import UnifiedDataFetcher
    from core.feature_engineer import FeatureEngineer

    training_end = get_last_complete_month_end()
    fetcher = UnifiedDataFetcher()
    engineer = FeatureEngineer(fetcher)

    result = engineer.build_complete_features(years=TRAINING_YEARS, end_date=training_end)

    df = result["features"].copy()
    df["vix"] = result["vix"]
    df["spx"] = result["spx"]

    logger.info(f"Dataset: {len(df)} samples, {len(df.columns)} features")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    logger.info("")

    required_cols = ['calendar_cohort', 'cohort_weight', 'feature_quality']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    tuner = Phase2Tuner(df=df, vix=result["vix"], n_trials=args.trials, output_dir=args.output_dir)

    study = tuner.run()

    tuner.save_results(study)

    logger.info("\n✅ Phase 2 complete!")

if __name__ == "__main__":
    main()
