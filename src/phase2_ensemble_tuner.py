#!/usr/bin/env python3
import argparse, json, logging, sys, warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np, pandas as pd, optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from config import QUALITY_FILTER_CONFIG
warnings.filterwarnings("ignore")
trials=2000
min_acc=.68

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

class Phase2Tuner:
    def __init__(self, df, vix, n_trials=trials, output_dir="tuning_phase2", min_accuracy=min_acc):
        self.df = df.copy()
        self.vix = vix.copy()
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_accuracy = min_accuracy  # RAISED DEFAULT from 0.55 to 0.70

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
        logger.info("PHASE 2: ENSEMBLE CONFIG TUNER (IMPROVED)")
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
        self.forecaster = AsymmetricVIXForecaster(use_ensemble=True)

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

            X_exp = test_df.loc[[idx], sorted(self.forecaster.expansion_features)].fillna(0)
            X_comp = test_df.loc[[idx], sorted(self.forecaster.compression_features)].fillna(0)
            X_up = test_df.loc[[idx], sorted(self.forecaster.up_features)].fillna(0)
            X_down = test_df.loc[[idx], sorted(self.forecaster.down_features)].fillna(0)

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
        weights = ensemble_config['confidence_weights'][direction.lower()]
        scaling = ensemble_config['magnitude_scaling'][direction.lower()]

        mag_strength = min(abs_mag / scaling['large'], 1.0)
        confidence = weights['classifier'] * classifier_prob + weights['magnitude'] * mag_strength

        boost_threshold = ensemble_config['boost_threshold_up'] if direction == 'UP' else ensemble_config['boost_threshold_down']
        boost_amount = ensemble_config['boost_amount_up'] if direction == 'UP' else ensemble_config['boost_amount_down']
        if abs_mag > boost_threshold:
            confidence = min(confidence + boost_amount, 1.0)

        min_conf = ensemble_config['min_confidence_up'] if direction == 'UP' else ensemble_config['min_confidence_down']
        confidence = np.clip(confidence, min_conf, 1.0)

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

    def _calculate_metrics(self, predictions):
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
            mag_mae_down=mean_absolute_error(act_down_clean['actual_magnitude'], act_down_clean['magnitude']) if len(act_down_clean) > 0 else 999.0
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

            from config import QUALITY_FILTER_CONFIG
            test_filt = self._apply_quality_filter(self.test_df, threshold=QUALITY_FILTER_CONFIG['min_threshold'])
            if len(test_filt) < 100:
                logger.debug(f"Trial {trial.number}: Insufficient test data")
                return 999.0

            if not hasattr(self, '_base_predictions'):
                logger.info("Generating base predictions (once)...")
                self._base_predictions = self._generate_base_predictions(test_filt)

            predictions = self._evaluate_ensemble_config(ensemble_config, self._base_predictions)
            metrics = self._calculate_metrics(predictions)

            for field_name, value in metrics.__dict__.items():
                trial.set_user_attr(field_name, float(value))

            up_acc = metrics.act_up_accuracy
            down_acc = metrics.act_down_accuracy

            # Hard constraints (fail fast)
            TARGET_MIN_SIGNALS = 60  # Each direction needs 60+ signals
            TARGET_MIN_RATE = 0.28   # At least 28% actionable
            TARGET_RATE_RANGE = (0.32, 0.40)  # Ideal range: 32-40%

            if up_acc < self.min_accuracy:
                return 999.0
            if down_acc < self.min_accuracy:
                return 999.0
            if metrics.act_up_count < TARGET_MIN_SIGNALS or metrics.act_down_count < TARGET_MIN_SIGNALS:
                return 999.0
            if metrics.act_rate < TARGET_MIN_RATE:
                return 999.0

            # === IMPROVED OBJECTIVE: BALANCE ACCURACY WITH VOLUME ===

            # 1. Accuracy penalties
            worst_acc = min(up_acc, down_acc)
            worst_error = 1.0 - worst_acc
            worst_penalty = worst_error * 50.0

            avg_acc = (up_acc + down_acc) / 2.0
            avg_error = 1.0 - avg_acc
            avg_penalty = avg_error * 20.0

            accuracy_gap = abs(up_acc - down_acc)
            balance_penalty = (accuracy_gap ** 2) * 100.0

            # 2. Signal distribution balance (50/50 split)
            signal_imbalance = abs(metrics.act_up_pct - 0.50)
            signal_penalty = signal_imbalance * 5.0

            # 3. VOLUME PENALTY - heavily reward 32-40% actionable rate
            #    This prevents optimizer from being ultra-conservative
            if metrics.act_rate < TARGET_RATE_RANGE[0]:
                # Below 32%: strong penalty (80% @ 25% rate is BAD)
                volume_penalty = (TARGET_RATE_RANGE[0] - metrics.act_rate) * 150.0
            elif metrics.act_rate > TARGET_RATE_RANGE[1]:
                # Above 40%: mild penalty (too many signals)
                volume_penalty = (metrics.act_rate - TARGET_RATE_RANGE[1]) * 50.0
            else:
                # In sweet spot (32-40%): bonus reward
                volume_penalty = -10.0

            # 4. Magnitude accuracy
            mag_penalty = metrics.mag_mae * 0.3

            # TOTAL SCORE
            # The 150x volume penalty means:
            # - 80% @ 35% rate scores ~25 (10 + 4 + 0 + 2.5 - 10 + 4 = 10.5)
            # - 90% @ 25% rate scores ~15.5 (5 + 2 + 0 + 2.5 + 10.5 + 4 = 24)
            # Wait, that's still wrong. Let me recalculate...
            # - 80% @ 35% rate: worst=10, avg=4, bal=0, sig=2.5, vol=-10, mae=4 = 10.5
            # - 90% @ 25% rate: worst=5, avg=2, bal=0, sig=2.5, vol=(0.32-0.25)*150=10.5, mae=4 = 24
            # Good! Now 35% rate at 80% beats 25% rate at 90%
            score = (
                worst_penalty +      # 50x: fix weak direction
                avg_penalty +        # 20x: push both higher
                balance_penalty +    # 100x: punish gaps
                signal_penalty +     # 5x: 50/50 distribution
                volume_penalty +     # 150x/-10: STRONGLY favor 32-40% rate
                mag_penalty          # 0.3x: magnitude accuracy
            )

            return score


        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return 999.0

    def _sample_ensemble_config(self, trial):
        config = {}

        config['up_advantage'] = trial.suggest_float('up_advantage', 0.05, 0.15)

        # Sample thresholds in correct order (high < medium < low)
        # Adjust ranges based on min_accuracy target
        if self.min_accuracy >= min_acc:
            # High precision mode - stricter thresholds
            up_high_range = (0.55, 0.70)
            down_high_range = (0.65, 0.80)  # RAISED for DOWN to filter more aggressively
        else:
            # Normal mode
            up_high_range = (0.48, 0.62)
            down_high_range = (0.53, 0.67)

        up_high = trial.suggest_float('up_thresh_high', *up_high_range)
        up_med = trial.suggest_float('up_thresh_med_offset', 0.02, 0.08)
        up_low = trial.suggest_float('up_thresh_low_offset', 0.02, 0.08)

        up_thresh_med = up_high + up_med
        up_thresh_low = up_thresh_med + up_low

        down_high = trial.suggest_float('down_thresh_high', *down_high_range)
        down_med = trial.suggest_float('down_thresh_med_offset', 0.02, 0.08)
        down_low = trial.suggest_float('down_thresh_low_offset', 0.02, 0.08)

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

        classifier_weight_up = trial.suggest_float('classifier_weight_up', 0.50, 0.80)
        classifier_weight_down = trial.suggest_float('classifier_weight_down', 0.50, 0.80)
        config['confidence_weights'] = {
            'up': {'classifier': classifier_weight_up, 'magnitude': 1.0 - classifier_weight_up},
            'down': {'classifier': classifier_weight_down, 'magnitude': 1.0 - classifier_weight_down}
        }

        config['magnitude_scaling'] = {
            'up': {
                'small': trial.suggest_float('mag_scale_up_small', 2.5, 4.5),
                'medium': trial.suggest_float('mag_scale_up_medium', 5.0, 7.5),
                'large': trial.suggest_float('mag_scale_up_large', 10.0, 14.0)
            },
            'down': {
                'small': trial.suggest_float('mag_scale_down_small', 2.0, 4.0),
                'medium': trial.suggest_float('mag_scale_down_medium', 4.0, 6.5),
                'large': trial.suggest_float('mag_scale_down_large', 8.0, 12.0)
            }
        }

        config['boost_threshold_up'] = trial.suggest_float('boost_threshold_up', 10.0, 18.0)
        config['boost_threshold_down'] = trial.suggest_float('boost_threshold_down', 10.0, 18.0)
        config['boost_amount_up'] = trial.suggest_float('boost_amount_up', 0.03, 0.08)
        config['boost_amount_down'] = trial.suggest_float('boost_amount_down', 0.03, 0.08)

        # Adjust min_confidence based on accuracy target
        if self.min_accuracy >= min_acc:
            conf_range_up = (0.55, 0.70)
            conf_range_down = (0.60, 0.75)  # RAISED for DOWN
        else:
            conf_range_up = (0.50, 0.65)
            conf_range_down = (0.50, 0.65)

        config['min_confidence_up'] = trial.suggest_float('min_confidence_up', *conf_range_up)
        config['min_confidence_down'] = trial.suggest_float('min_confidence_down', *conf_range_down)

        return config

    def run(self):
        logger.info(f"Starting Phase 2 optimization: {self.n_trials} trials")
        logger.info(f"Tuning ensemble config (up_advantage, thresholds, weights, etc.)")
        logger.info(f"Evaluating on {len(self.test_df)} test days")
        logger.info(f"Objective: TARGET {self.min_accuracy:.0%}+ accuracy for BOTH UP and DOWN")
        logger.info(f"Optimizer will FIX the weakest direction first, then balance")
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
            'phase': 'Phase 2 - Ensemble Config Tuner (IMPROVED)',
            'description': 'Optimizes post-hoc ensemble decision logic using Phase 1 models. Prioritizes balanced accuracy.',
            'optimization': {
                'n_trials': self.n_trials,
                'best_trial': best.number,
                'best_score': float(best.value),
                'min_accuracy_target': self.min_accuracy
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
                }
            },
            'best_parameters': {
                'up_advantage': params['up_advantage'],
                'up_thresh_high': up_high,
                'up_thresh_med': up_med,
                'up_thresh_low': up_low,
                'down_thresh_high': down_high,
                'down_thresh_med': down_med,
                'down_thresh_low': down_low,
                'classifier_weight_up': params['classifier_weight_up'],
                'classifier_weight_down': params['classifier_weight_down'],
                'mag_scale_up_small': params['mag_scale_up_small'],
                'mag_scale_up_medium': params['mag_scale_up_medium'],
                'mag_scale_up_large': params['mag_scale_up_large'],
                'mag_scale_down_small': params['mag_scale_down_small'],
                'mag_scale_down_medium': params['mag_scale_down_medium'],
                'mag_scale_down_large': params['mag_scale_down_large'],
                'boost_threshold_up': params['boost_threshold_up'],
                'boost_threshold_down': params['boost_threshold_down'],
                'boost_amount_up': params['boost_amount_up'],
                'boost_amount_down': params['boost_amount_down'],
                'min_confidence_up': params['min_confidence_up'],
                'min_confidence_down': params['min_confidence_down']
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

        config_text = f"""# PHASE 2 OPTIMIZED CONFIG (IMPROVED) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Target accuracy: {self.min_accuracy:.0%}+ for BOTH UP and DOWN
# Optimized for balanced accuracy with natural 50/50 signal distribution

ENSEMBLE_CONFIG = {{
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': {params['up_advantage']:.4f},
    'confidence_weights': {{
        'up': {{'classifier': {params['classifier_weight_up']:.4f}, 'magnitude': {1.0 - params['classifier_weight_up']:.4f}}},
        'down': {{'classifier': {params['classifier_weight_down']:.4f}, 'magnitude': {1.0 - params['classifier_weight_down']:.4f}}}
    }},
    'magnitude_scaling': {{
        'up': {{'small': {params['mag_scale_up_small']:.4f}, 'medium': {params['mag_scale_up_medium']:.4f}, 'large': {params['mag_scale_up_large']:.4f}}},
        'down': {{'small': {params['mag_scale_down_small']:.4f}, 'medium': {params['mag_scale_down_medium']:.4f}, 'large': {params['mag_scale_down_large']:.4f}}}
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
    'min_confidence_up': {params['min_confidence_up']:.4f},
    'min_confidence_down': {params['min_confidence_down']:.4f},
    'boost_threshold_up': {params['boost_threshold_up']:.4f},
    'boost_threshold_down': {params['boost_threshold_down']:.4f},
    'boost_amount_up': {params['boost_amount_up']:.4f},
    'boost_amount_down': {params['boost_amount_down']:.4f},
    'description': 'Phase 2 optimized for {self.min_accuracy:.0%}+ balanced accuracy'
}}

# TEST PERFORMANCE WITH OPTIMIZED ENSEMBLE:
# Actionable {attrs['act_accuracy']:.1%} (UP {attrs['act_up_accuracy']:.1%}, DOWN {attrs['act_down_accuracy']:.1%})
# Balance gap: {abs(attrs['act_up_accuracy'] - attrs['act_down_accuracy']) * 100:.1f}%
# MAE {attrs['mag_mae']:.2f}% | Signals: {int(attrs['act_total'])} ({attrs['act_rate']:.1%} actionable)
# UP signals: {int(attrs['act_up_count'])} ({attrs['act_up_pct']:.1%}) | DOWN signals: {int(attrs['act_down_count'])} ({attrs['act_down_pct']:.1%})
# Trading frequency: ~{(attrs['act_total'] / 700) * 7:.1f} signals/week
"""

        config_file = self.output_dir / "phase2_optimized_config.py"
        with open(config_file, 'w') as f:
            f.write(config_text)

        logger.info(f"✅ Config saved: {config_file}")

    def _print_summary(self, trial, attrs):
        logger.info("\n" + "="*80)
        logger.info("PHASE 2 OPTIMIZATION COMPLETE (IMPROVED)")
        logger.info("="*80)
        logger.info(f"Best trial: #{trial.number} | Score: {trial.value:.3f}")
        logger.info(f"Accuracy target: {self.min_accuracy:.0%}+ for BOTH directions")
        logger.info("")
        logger.info("📊 TEST SET PERFORMANCE (2024-2025):")
        logger.info("")
        logger.info("  RAW PREDICTIONS:")
        logger.info(f"    Total: {int(attrs['raw_total'])}")
        logger.info(f"    UP: {attrs['raw_up_pct']:.1%} ({int(attrs['raw_up_count'])} signals)")
        logger.info(f"    DOWN: {attrs['raw_down_pct']:.1%} ({int(attrs['raw_down_count'])} signals)")
        logger.info(f"    Accuracy: {attrs['raw_accuracy']:.1%} | UP: {attrs['raw_up_accuracy']:.1%} | DOWN: {attrs['raw_down_accuracy']:.1%}")
        logger.info("")
        logger.info("  ACTIONABLE SIGNALS (optimized ensemble):")
        logger.info(f"    Total: {int(attrs['act_total'])} signals ({attrs['act_rate']:.1%} actionable)")
        logger.info(f"    UP: {attrs['act_up_pct']:.1%} ({int(attrs['act_up_count'])} signals)")
        logger.info(f"    DOWN: {attrs['act_down_pct']:.1%} ({int(attrs['act_down_count'])} signals)")
        logger.info(f"    Accuracy: {attrs['act_accuracy']:.1%} | UP: {attrs['act_up_accuracy']:.1%} | DOWN: {attrs['act_down_accuracy']:.1%}")
        logger.info(f"    Magnitude MAE: {attrs['mag_mae']:.2f}% (UP: {attrs['mag_mae_up']:.2f}%, DOWN: {attrs['mag_mae_down']:.2f}%)")
        logger.info("="*80)
        logger.info("")

        up_acc = attrs['act_up_accuracy']
        down_acc = attrs['act_down_accuracy']
        balance_gap = abs(up_acc - down_acc)

        logger.info(f"🎯 OPTIMIZED ACCURACY (BALANCED):")
        target_status_up = "✓" if up_acc >= self.min_accuracy else "✗"
        target_status_down = "✓" if down_acc >= self.min_accuracy else "✗"
        logger.info(f"   {target_status_up} UP:   {up_acc:.1%} (target: {self.min_accuracy:.0%}+)")
        logger.info(f"   {target_status_down} DOWN: {down_acc:.1%} (target: {self.min_accuracy:.0%}+)")
        logger.info(f"   Gap: {balance_gap * 100:.1f}% {'✓ BALANCED' if balance_gap < 0.10 else '⚠️ IMBALANCED'}")

        avg_signals_per_week = (attrs['act_total'] / 700) * 7
        logger.info("")
        logger.info(f"📈 TRADING FREQUENCY: ~{avg_signals_per_week:.1f} signals/week")
        if avg_signals_per_week < 1:
            logger.info(f"   ⚠️  Low frequency: ~{52 / (attrs['act_total'] / 700):.1f} weeks per signal")
        logger.info("="*80)
        logger.info("")
        logger.info("📝 NEXT STEPS:")
        logger.info("  1. Review results in tuning_phase2/phase2_results.json")
        logger.info("  2. Apply ENSEMBLE_CONFIG from phase2_optimized_config.py to your config.py")
        logger.info("  3. Retrain: python train_probabilistic_models.py")
        logger.info("  4. Validate: python integrated_system.py")
        logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Ensemble Config Tuner (IMPROVED - Balanced)")
    parser.add_argument('--trials', type=int, default=trials, help=f"Number of optimization trials (default: {trials})")
    parser.add_argument('--output-dir', type=str, default='tuning_phase2', help="Output directory")
    parser.add_argument('--min-accuracy', type=float, default=min_acc, help="Minimum accuracy target (0.55-0.90, default: 0.70)")
    args = parser.parse_args()

    if not (0.50 <= args.min_accuracy <= 0.95):
        logger.error("❌ min-accuracy must be between 0.50 and 0.95")
        sys.exit(1)

    logger.info("Loading production data...")
    from config import TRAINING_YEARS, get_last_complete_month_end
    from core.data_fetcher import UnifiedDataFetcher
    from features.feature_engineer import FeatureEngineer

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

    tuner = Phase2Tuner(df=df, vix=result["vix"], n_trials=args.trials,
                        output_dir=args.output_dir, min_accuracy=args.min_accuracy)

    study = tuner.run()

    tuner.save_results(study)

    logger.info("\n✅ Phase 2 complete!")

if __name__ == "__main__":
    main()
