#!/usr/bin/env python3
import argparse, json, logging, sys, warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np, pandas as pd, optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from config import PHASE1_TUNER_TRIALS

warnings.filterwarnings("ignore")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/phase1_tuner.log")])
logger = logging.getLogger(__name__)

@dataclass
class RawMetrics:
    total: int = 0
    up_count: int = 0
    down_count: int = 0
    up_pct: float = 0.0
    down_pct: float = 0.0
    accuracy: float = 0.0
    up_accuracy: float = 0.0
    down_accuracy: float = 0.0
    mae: float = 0.0
    mae_up: float = 0.0
    mae_down: float = 0.0
    expansion_val_mae: float = 0.0
    compression_val_mae: float = 0.0
    up_val_acc: float = 0.0
    down_val_acc: float = 0.0
    n_expansion_features: int = 0
    n_compression_features: int = 0
    n_up_features: int = 0
    n_down_features: int = 0

class Phase1Tuner:
    def __init__(self, df, vix, n_trials=PHASE1_TUNER_TRIALS, output_dir="tuning_phase1"):
        self.df = df.copy(); self.vix = vix.copy(); self.n_trials = n_trials
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
        self.base_cols = [c for c in df.columns if c not in
            ["vix", "spx", "calendar_cohort", "cohort_weight", "feature_quality",
             "future_vix", "target_vix_pct_change", "target_log_vix_change", "target_direction"]]
        self._calculate_targets()
        logger.info("="*80)
        logger.info("PHASE 1: MODEL HYPERPARAMETER TUNER (RAW PREDICTIONS)")
        logger.info("="*80)
        logger.info(f"Train:  {len(self.train_df)} days ({self.train_df.index[0].date()} to {self.train_end.date()})")
        logger.info(f"Val:    {len(self.val_df)} days ({self.val_df.index[0].date()} to {self.val_end.date()})")
        logger.info(f"Test:   {len(self.test_df)} days ({self.test_start.date()} to {self.test_df.index[-1].date()})")
        logger.info(f"Base features: {len(self.base_cols)}")
        logger.info(f"Optimizing on RAW predictions (no ensemble filtering)")
        logger.info("="*80)

    def _calculate_targets(self):
        from core.target_calculator import TargetCalculator
        calculator = TargetCalculator()
        self.train_df = calculator.calculate_all_targets(self.train_df, vix_col='vix')
        self.val_df = calculator.calculate_all_targets(self.val_df, vix_col='vix')
        self.test_df = calculator.calculate_all_targets(self.test_df, vix_col='vix')

    def _apply_quality_filter(self, df, threshold):
        if 'feature_quality' not in df.columns:
            logger.warning("No feature_quality column - skipping filter")
            return df
        filtered = df[df['feature_quality'] >= threshold].copy()
        filtered_pct = (1 - len(filtered)/len(df)) * 100
        if filtered_pct > 50:
            raise ValueError(f"Quality filter removed {filtered_pct:.1f}% of data (threshold={threshold:.3f})")
        return filtered

    def _apply_cohort_weights(self, df, fomc_w, opex_w, earnings_w):
        cohort_map = {'fomc_period': fomc_w, 'opex_week': opex_w, 'earnings_heavy': earnings_w, 'mid_cycle': 1.0}
        return df['calendar_cohort'].map(cohort_map).fillna(1.0)

    def _select_features(self, train_val_df, vix, target_type, top_n, corr_threshold, cv_params):
        from core.xgboost_feature_selector_v2 import FeatureSelector
        import config as cfg
        original_cv = cfg.FEATURE_SELECTION_CV_PARAMS.copy()
        cfg.FEATURE_SELECTION_CV_PARAMS.update(cv_params)
        try:
            selector = FeatureSelector(target_type=target_type, top_n=top_n, correlation_threshold=corr_threshold)
            test_start_idx = len(train_val_df)
            selected_features, _ = selector.select_features(train_val_df[self.base_cols], vix, test_start_idx=test_start_idx)
            return selected_features
        finally:
            cfg.FEATURE_SELECTION_CV_PARAMS.update(original_cv)

    def _train_and_evaluate_models(self, trial_params):
        try:
            train_filt = self._apply_quality_filter(self.train_df, trial_params['quality_threshold'])
            val_filt = self._apply_quality_filter(self.val_df, trial_params['quality_threshold'])
            test_filt = self._apply_quality_filter(self.test_df, trial_params['quality_threshold'])
            if len(train_filt) < 200 or len(val_filt) < 50 or len(test_filt) < 100:
                logger.warning(f"Insufficient data after quality filter")
                return None
            train_weights = self._apply_cohort_weights(train_filt, trial_params['fomc_weight'],
                trial_params['opex_weight'], trial_params['earnings_weight'])
            train_val_df = pd.concat([train_filt, val_filt])
            train_val_vix = pd.concat([self.vix.loc[train_filt.index], self.vix.loc[val_filt.index]])
            cv_params = {'n_estimators': trial_params['cv_n_estimators'], 'max_depth': trial_params['cv_max_depth'],
                'learning_rate': trial_params['cv_learning_rate'], 'subsample': trial_params['cv_subsample'],
                'colsample_bytree': trial_params['cv_colsample_bytree']}
            exp_features = self._select_features(train_val_df, train_val_vix, 'expansion',
                trial_params['expansion_top_n'], trial_params['correlation_threshold'], cv_params)
            comp_features = self._select_features(train_val_df, train_val_vix, 'compression',
                trial_params['compression_top_n'], trial_params['correlation_threshold'], cv_params)
            up_features = self._select_features(train_val_df, train_val_vix, 'up',
                trial_params['up_top_n'], trial_params['correlation_threshold'], cv_params)
            down_features = self._select_features(train_val_df, train_val_vix, 'down',
                trial_params['down_top_n'], trial_params['correlation_threshold'], cv_params)
            if any(len(f) < 20 for f in [exp_features, comp_features, up_features, down_features]):
                logger.warning("Insufficient features selected")
                return None
            from xgboost import XGBRegressor, XGBClassifier
            expansion_params = self._build_expansion_params(trial_params)
            compression_params = self._build_compression_params(trial_params)
            up_params = self._build_up_params(trial_params)
            down_params = self._build_down_params(trial_params)
            train_up_mask = (train_filt['target_direction'] == 1) & (train_filt['target_log_vix_change'].notna())
            val_up_mask = (val_filt['target_direction'] == 1) & (val_filt['target_log_vix_change'].notna())
            X_exp_train = train_filt[train_up_mask][exp_features].fillna(0)
            y_exp_train = train_filt[train_up_mask]['target_log_vix_change']
            w_exp_train = train_weights[train_up_mask]
            X_exp_val = val_filt[val_up_mask][exp_features].fillna(0)
            y_exp_val = val_filt[val_up_mask]['target_log_vix_change']
            expansion_model = XGBRegressor(**expansion_params)
            expansion_model.fit(X_exp_train, y_exp_train, sample_weight=w_exp_train,
                eval_set=[(X_exp_val, y_exp_val)], verbose=False)
            y_exp_val_pred = np.clip(expansion_model.predict(X_exp_val), -2, 2)
            exp_val_mae = mean_absolute_error((np.exp(y_exp_val) - 1) * 100, (np.exp(y_exp_val_pred) - 1) * 100)
            train_down_mask = (train_filt['target_direction'] == 0) & (train_filt['target_log_vix_change'].notna())
            val_down_mask = (val_filt['target_direction'] == 0) & (val_filt['target_log_vix_change'].notna())
            X_comp_train = train_filt[train_down_mask][comp_features].fillna(0)
            y_comp_train = train_filt[train_down_mask]['target_log_vix_change']
            w_comp_train = train_weights[train_down_mask]
            X_comp_val = val_filt[val_down_mask][comp_features].fillna(0)
            y_comp_val = val_filt[val_down_mask]['target_log_vix_change']
            compression_model = XGBRegressor(**compression_params)
            compression_model.fit(X_comp_train, y_comp_train, sample_weight=w_comp_train,
                eval_set=[(X_comp_val, y_comp_val)], verbose=False)
            y_comp_val_pred = np.clip(compression_model.predict(X_comp_val), -2, 2)
            comp_val_mae = mean_absolute_error((np.exp(y_comp_val) - 1) * 100, (np.exp(y_comp_val_pred) - 1) * 100)
            X_up_train = train_filt[up_features].fillna(0)
            y_up_train = train_filt['target_direction']
            X_up_val = val_filt[up_features].fillna(0)
            y_up_val = val_filt['target_direction']
            up_model = XGBClassifier(**up_params)
            up_model.fit(X_up_train, y_up_train, sample_weight=train_weights,
                eval_set=[(X_up_val, y_up_val)], verbose=False)
            from sklearn.metrics import accuracy_score
            up_val_acc = accuracy_score(y_up_val, up_model.predict(X_up_val))
            X_down_train = train_filt[down_features].fillna(0)
            y_down_train = 1 - train_filt['target_direction']
            X_down_val = val_filt[down_features].fillna(0)
            y_down_val = 1 - val_filt['target_direction']
            down_model = XGBClassifier(**down_params)
            down_model.fit(X_down_train, y_down_train, sample_weight=train_weights,
                eval_set=[(X_down_val, y_down_val)], verbose=False)
            down_val_acc = accuracy_score(y_down_val, down_model.predict(X_down_val))

            # RAW PREDICTIONS (no ensemble logic)
            test_predictions = []
            for idx in test_filt.index:
                if pd.isna(test_filt.loc[idx, 'target_direction']): continue
                X_exp = test_filt.loc[[idx], exp_features].fillna(0)
                X_comp = test_filt.loc[[idx], comp_features].fillna(0)
                X_up = test_filt.loc[[idx], up_features].fillna(0)
                X_down = test_filt.loc[[idx], down_features].fillna(0)
                exp_log = np.clip(expansion_model.predict(X_exp)[0], -2, 2)
                comp_log = np.clip(compression_model.predict(X_comp)[0], -2, 2)
                exp_pct = (np.exp(exp_log) - 1) * 100
                comp_pct = (np.exp(comp_log) - 1) * 100
                p_up = up_model.predict_proba(X_up)[0, 1]
                p_down = down_model.predict_proba(X_down)[0, 1]

                # PHASE 1: Simplest decision - raw probability comparison (no normalization)
                if p_up > p_down:
                    pred_direction = 1
                    magnitude = exp_pct
                else:
                    pred_direction = 0
                    magnitude = comp_pct

                actual_direction = int(test_filt.loc[idx, 'target_direction'])
                actual_mag_log = test_filt.loc[idx, 'target_log_vix_change']
                actual_mag_pct = (np.exp(actual_mag_log) - 1) * 100

                test_predictions.append({
                    'pred_direction': pred_direction,
                    'actual_direction': actual_direction,
                    'direction_correct': (pred_direction == actual_direction),
                    'magnitude': magnitude,
                    'actual_magnitude': actual_mag_pct
                })

            if len(test_predictions) < 100:
                logger.warning("Insufficient test predictions")
                return None

            metrics = self._calculate_metrics(test_predictions, exp_val_mae, comp_val_mae, up_val_acc, down_val_acc,
                len(exp_features), len(comp_features), len(up_features), len(down_features))
            return metrics
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_metrics(self, predictions, exp_val_mae, comp_val_mae, up_val_acc, down_val_acc,
                          n_exp_feats, n_comp_feats, n_up_feats, n_down_feats):
        df = pd.DataFrame(predictions)
        up_preds = df[df['pred_direction'] == 1]
        down_preds = df[df['pred_direction'] == 0]

        df_clean = df.dropna(subset=['actual_magnitude', 'magnitude'])
        up_clean = up_preds.dropna(subset=['actual_magnitude', 'magnitude'])
        down_clean = down_preds.dropna(subset=['actual_magnitude', 'magnitude'])

        metrics = RawMetrics(
            total=len(df),
            up_count=len(up_preds),
            down_count=len(down_preds),
            up_pct=len(up_preds) / len(df),
            down_pct=len(down_preds) / len(df),
            accuracy=df['direction_correct'].mean(),
            up_accuracy=up_preds['direction_correct'].mean() if len(up_preds) > 0 else 0.0,
            down_accuracy=down_preds['direction_correct'].mean() if len(down_preds) > 0 else 0.0,
            mae=mean_absolute_error(df_clean['actual_magnitude'], df_clean['magnitude']) if len(df_clean) > 0 else 999.0,
            mae_up=mean_absolute_error(up_clean['actual_magnitude'], up_clean['magnitude']) if len(up_clean) > 0 else 999.0,
            mae_down=mean_absolute_error(down_clean['actual_magnitude'], down_clean['magnitude']) if len(down_clean) > 0 else 999.0,
            expansion_val_mae=exp_val_mae,
            compression_val_mae=comp_val_mae,
            up_val_acc=up_val_acc,
            down_val_acc=down_val_acc,
            n_expansion_features=n_exp_feats,
            n_compression_features=n_comp_feats,
            n_up_features=n_up_feats,
            n_down_features=n_down_feats
        )
        return metrics

    def _build_expansion_params(self, trial_params):
        return {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': trial_params['exp_max_depth'], 'learning_rate': trial_params['exp_learning_rate'],
            'n_estimators': trial_params['exp_n_estimators'], 'subsample': trial_params['exp_subsample'],
            'colsample_bytree': trial_params['exp_colsample_bytree'], 'colsample_bylevel': trial_params['exp_colsample_bylevel'],
            'min_child_weight': trial_params['exp_min_child_weight'], 'reg_alpha': trial_params['exp_reg_alpha'],
            'reg_lambda': trial_params['exp_reg_lambda'], 'gamma': trial_params['exp_gamma'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

    def _build_compression_params(self, trial_params):
        return {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': trial_params['comp_max_depth'], 'learning_rate': trial_params['comp_learning_rate'],
            'n_estimators': trial_params['comp_n_estimators'], 'subsample': trial_params['comp_subsample'],
            'colsample_bytree': trial_params['comp_colsample_bytree'], 'colsample_bylevel': trial_params['comp_colsample_bylevel'],
            'min_child_weight': trial_params['comp_min_child_weight'], 'reg_alpha': trial_params['comp_reg_alpha'],
            'reg_lambda': trial_params['comp_reg_lambda'], 'gamma': trial_params['comp_gamma'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

    def _build_up_params(self, trial_params):
        return {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
            'max_depth': trial_params['up_max_depth'], 'learning_rate': trial_params['up_learning_rate'],
            'n_estimators': trial_params['up_n_estimators'], 'subsample': trial_params['up_subsample'],
            'colsample_bytree': trial_params['up_colsample_bytree'], 'min_child_weight': trial_params['up_min_child_weight'],
            'reg_alpha': trial_params['up_reg_alpha'], 'reg_lambda': trial_params['up_reg_lambda'],
            'gamma': trial_params['up_gamma'], 'scale_pos_weight': trial_params['up_scale_pos_weight'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

    def _build_down_params(self, trial_params):
        return {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
            'max_depth': trial_params['down_max_depth'], 'learning_rate': trial_params['down_learning_rate'],
            'n_estimators': trial_params['down_n_estimators'], 'subsample': trial_params['down_subsample'],
            'colsample_bytree': trial_params['down_colsample_bytree'], 'min_child_weight': trial_params['down_min_child_weight'],
            'reg_alpha': trial_params['down_reg_alpha'], 'reg_lambda': trial_params['down_reg_lambda'],
            'gamma': trial_params['down_gamma'], 'scale_pos_weight': trial_params['down_scale_pos_weight'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}

    def objective(self, trial):
        """
        PHASE 1 OBJECTIVE: Optimize RAW predictions (no ensemble filtering)
        Target: 60-70% accuracy for both UP and DOWN
        Accept natural UP/DOWN imbalance (~20/80 expected)
        Low MAE
        """
        trial_params = self._sample_hyperparameters(trial)
        metrics = self._train_and_evaluate_models(trial_params)
        if metrics is None: return 999.0
        for field_name, value in metrics.__dict__.items():
            trial.set_user_attr(field_name, float(value))

        # Accuracy penalty: penalize low accuracy for both directions
        acc_penalty = (1 - metrics.up_accuracy) + (1 - metrics.down_accuracy)

        # NO balance penalty - accept natural distribution (system naturally does ~20/80)

        # Volume penalties: scale with expected natural distribution
        min_up = 50  # Lower minimum since UP is naturally rare
        min_down = 150  # Higher minimum since DOWN is naturally common
        up_volume_penalty = max(0, min_up - metrics.up_count) * 0.5
        down_volume_penalty = max(0, min_down - metrics.down_count) * 0.3

        # MAE penalty
        mae_penalty = metrics.mae * 0.3

        # Validation penalties
        val_penalty = (max(0, metrics.expansion_val_mae - 12.0) * 0.5 +
            max(0, metrics.compression_val_mae - 8.0) * 0.5 +
            max(0, 0.55 - metrics.up_val_acc) * 2.0 +
            max(0, 0.55 - metrics.down_val_acc) * 2.0)

        score = acc_penalty + up_volume_penalty + down_volume_penalty + mae_penalty + val_penalty
        return score

    def _sample_hyperparameters(self, trial):
        params = {}
        params['quality_threshold'] = trial.suggest_float('quality_threshold', 0.50, 0.65)
        params['fomc_weight'] = trial.suggest_float('cohort_fomc', 1.10, 1.60)
        params['opex_weight'] = trial.suggest_float('cohort_opex', 1.00, 1.50)
        params['earnings_weight'] = trial.suggest_float('cohort_earnings', 1.00, 1.40)
        params['cv_n_estimators'] = trial.suggest_int('cv_n_estimators', 100, 300)
        params['cv_max_depth'] = trial.suggest_int('cv_max_depth', 3, 6)
        params['cv_learning_rate'] = trial.suggest_float('cv_learning_rate', 0.03, 0.15, log=True)
        params['cv_subsample'] = trial.suggest_float('cv_subsample', 0.70, 0.95)
        params['cv_colsample_bytree'] = trial.suggest_float('cv_colsample_bytree', 0.70, 0.95)
        params['expansion_top_n'] = trial.suggest_int('expansion_top_n', 70, 140)
        params['compression_top_n'] = trial.suggest_int('compression_top_n', 70, 140)
        params['up_top_n'] = trial.suggest_int('up_top_n', 80, 150)
        params['down_top_n'] = trial.suggest_int('down_top_n', 80, 150)
        params['correlation_threshold'] = trial.suggest_float('correlation_threshold', 0.85, 0.96)
        params['exp_max_depth'] = trial.suggest_int('exp_max_depth', 2, 7)
        params['exp_learning_rate'] = trial.suggest_float('exp_learning_rate', 0.01, 0.12, log=True)
        params['exp_n_estimators'] = trial.suggest_int('exp_n_estimators', 300, 900)
        params['exp_subsample'] = trial.suggest_float('exp_subsample', 0.70, 0.95)
        params['exp_colsample_bytree'] = trial.suggest_float('exp_colsample_bytree', 0.70, 0.95)
        params['exp_colsample_bylevel'] = trial.suggest_float('exp_colsample_bylevel', 0.70, 0.95)
        params['exp_min_child_weight'] = trial.suggest_int('exp_min_child_weight', 3, 15)
        params['exp_reg_alpha'] = trial.suggest_float('exp_reg_alpha', 1.0, 8.0)
        params['exp_reg_lambda'] = trial.suggest_float('exp_reg_lambda', 2.0, 10.0)
        params['exp_gamma'] = trial.suggest_float('exp_gamma', 0.0, 0.8)
        params['comp_max_depth'] = trial.suggest_int('comp_max_depth', 2, 7)
        params['comp_learning_rate'] = trial.suggest_float('comp_learning_rate', 0.01, 0.12, log=True)
        params['comp_n_estimators'] = trial.suggest_int('comp_n_estimators', 300, 900)
        params['comp_subsample'] = trial.suggest_float('comp_subsample', 0.70, 0.95)
        params['comp_colsample_bytree'] = trial.suggest_float('comp_colsample_bytree', 0.70, 0.95)
        params['comp_colsample_bylevel'] = trial.suggest_float('comp_colsample_bylevel', 0.70, 0.95)
        params['comp_min_child_weight'] = trial.suggest_int('comp_min_child_weight', 3, 15)
        params['comp_reg_alpha'] = trial.suggest_float('comp_reg_alpha', 1.0, 8.0)
        params['comp_reg_lambda'] = trial.suggest_float('comp_reg_lambda', 2.0, 10.0)
        params['comp_gamma'] = trial.suggest_float('comp_gamma', 0.0, 0.8)
        params['up_max_depth'] = trial.suggest_int('up_max_depth', 5, 12)
        params['up_learning_rate'] = trial.suggest_float('up_learning_rate', 0.01, 0.12, log=True)
        params['up_n_estimators'] = trial.suggest_int('up_n_estimators', 300, 800)
        params['up_subsample'] = trial.suggest_float('up_subsample', 0.70, 0.95)
        params['up_colsample_bytree'] = trial.suggest_float('up_colsample_bytree', 0.70, 0.95)
        params['up_min_child_weight'] = trial.suggest_int('up_min_child_weight', 5, 18)
        params['up_reg_alpha'] = trial.suggest_float('up_reg_alpha', 1.0, 6.0)
        params['up_reg_lambda'] = trial.suggest_float('up_reg_lambda', 2.0, 10.0)
        params['up_gamma'] = trial.suggest_float('up_gamma', 0.1, 1.2)
        params['down_max_depth'] = trial.suggest_int('down_max_depth', 5, 12)
        params['down_learning_rate'] = trial.suggest_float('down_learning_rate', 0.01, 0.12, log=True)
        params['down_n_estimators'] = trial.suggest_int('down_n_estimators', 300, 800)
        params['down_subsample'] = trial.suggest_float('down_subsample', 0.70, 0.95)
        params['down_colsample_bytree'] = trial.suggest_float('down_colsample_bytree', 0.70, 0.95)
        params['down_min_child_weight'] = trial.suggest_int('down_min_child_weight', 5, 18)
        params['down_reg_alpha'] = trial.suggest_float('down_reg_alpha', 1.0, 6.0)
        params['down_reg_lambda'] = trial.suggest_float('down_reg_lambda', 2.0, 10.0)
        params['down_gamma'] = trial.suggest_float('down_gamma', 0.1, 1.2)
        params['up_scale_pos_weight'] = trial.suggest_float('up_scale_pos_weight', 0.5, 2.0)
        params['down_scale_pos_weight'] = trial.suggest_float('down_scale_pos_weight', 0.5, 2.0)
        return params


    def run(self):
        logger.info(f"Starting Phase 1 optimization: {self.n_trials} trials")
        logger.info(f"Tuning 56 hyperparameters (models + feature selection + scale_pos_weight)")
        logger.info(f"Evaluating RAW predictions on {len(self.test_df)} test days (NO ensemble filtering)")
        logger.info("="*80)
        study = optuna.create_study(direction='minimize',
            sampler=TPESampler(seed=42, n_startup_trials=min(50, self.n_trials // 6)))
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)
        return study

    def save_results(self, study):
        best = study.best_trial; attrs = best.user_attrs
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 1 - Model Hyperparameter Tuning (RAW)',
            'description': 'Optimizes 4-model architecture + feature selection on RAW predictions (no ensemble)',
            'optimization': {'n_trials': self.n_trials, 'best_trial': best.number, 'best_score': float(best.value)},
            'data_splits': {
                'train': f"{self.train_df.index[0].date()} to {self.train_end.date()}",
                'val': f"{self.val_df.index[0].date()} to {self.val_end.date()}",
                'test': f"{self.test_start.date()} to {self.test_df.index[-1].date()}",
                'train_size': len(self.train_df), 'val_size': len(self.val_df), 'test_size': len(self.test_df)},
            'test_metrics': {
                'raw_predictions': {
                    'total': int(attrs['total']), 'up_count': int(attrs['up_count']),
                    'down_count': int(attrs['down_count']), 'up_pct': float(attrs['up_pct']),
                    'down_pct': float(attrs['down_pct']), 'accuracy': float(attrs['accuracy']),
                    'up_accuracy': float(attrs['up_accuracy']), 'down_accuracy': float(attrs['down_accuracy']),
                    'mae': float(attrs['mae']), 'mae_up': float(attrs['mae_up']), 'mae_down': float(attrs['mae_down'])},
                'validation': {'expansion_mae': float(attrs['expansion_val_mae']),
                    'compression_mae': float(attrs['compression_val_mae']), 'up_accuracy': float(attrs['up_val_acc']),
                    'down_accuracy': float(attrs['down_val_acc'])},
                'features': {'expansion': int(attrs['n_expansion_features']),
                    'compression': int(attrs['n_compression_features']), 'up': int(attrs['n_up_features']),
                    'down': int(attrs['n_down_features'])}},
            'best_parameters': best.params}
        results_file = self.output_dir / "phase1_results.json"
        with open(results_file, 'w') as f: json.dump(results, f, indent=2)
        logger.info(f"\n✅ Results saved: {results_file}")
        self._generate_config(best, attrs)
        self._print_summary(best, attrs)

    def _generate_config(self, trial, attrs):
        config_text = f"""# PHASE 1 OPTIMIZED CONFIG (RAW PREDICTIONS) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUALITY_FILTER_CONFIG = {{'enabled': True, 'min_threshold': {trial.params['quality_threshold']:.4f},
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}}

CALENDAR_COHORTS = {{
    'fomc_period': {{'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': {trial.params['cohort_fomc']:.4f}, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'}},
    'opex_week': {{'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': {trial.params['cohort_opex']:.4f}, 'description': 'Options expiration week + VIX futures rollover'}},
    'earnings_heavy': {{'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': {trial.params['cohort_earnings']:.4f}, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'}},
    'mid_cycle': {{'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}}}

FEATURE_SELECTION_CV_PARAMS = {{'n_estimators': {trial.params['cv_n_estimators']},
    'max_depth': {trial.params['cv_max_depth']}, 'learning_rate': {trial.params['cv_learning_rate']:.4f},
    'subsample': {trial.params['cv_subsample']:.4f}, 'colsample_bytree': {trial.params['cv_colsample_bytree']:.4f}}}

FEATURE_SELECTION_CONFIG = {{'expansion_top_n': {trial.params['expansion_top_n']},
    'compression_top_n': {trial.params['compression_top_n']}, 'up_top_n': {trial.params['up_top_n']},
    'down_top_n': {trial.params['down_top_n']}, 'cv_folds': 5,
    'protected_features': [],
    'correlation_threshold': {trial.params['correlation_threshold']:.4f},
    'description': 'Phase 1 optimized on RAW predictions (no ensemble filtering)'}}

EXPANSION_PARAMS = {{'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': {trial.params['exp_max_depth']}, 'learning_rate': {trial.params['exp_learning_rate']:.4f},
    'n_estimators': {trial.params['exp_n_estimators']}, 'subsample': {trial.params['exp_subsample']:.4f},
    'colsample_bytree': {trial.params['exp_colsample_bytree']:.4f}, 'colsample_bylevel': {trial.params['exp_colsample_bylevel']:.4f},
    'min_child_weight': {trial.params['exp_min_child_weight']}, 'reg_alpha': {trial.params['exp_reg_alpha']:.4f},
    'reg_lambda': {trial.params['exp_reg_lambda']:.4f}, 'gamma': {trial.params['exp_gamma']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}}

COMPRESSION_PARAMS = {{'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': {trial.params['comp_max_depth']}, 'learning_rate': {trial.params['comp_learning_rate']:.4f},
    'n_estimators': {trial.params['comp_n_estimators']}, 'subsample': {trial.params['comp_subsample']:.4f},
    'colsample_bytree': {trial.params['comp_colsample_bytree']:.4f}, 'colsample_bylevel': {trial.params['comp_colsample_bylevel']:.4f},
    'min_child_weight': {trial.params['comp_min_child_weight']}, 'reg_alpha': {trial.params['comp_reg_alpha']:.4f},
    'reg_lambda': {trial.params['comp_reg_lambda']:.4f}, 'gamma': {trial.params['comp_gamma']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}}

UP_CLASSIFIER_PARAMS = {{'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': {trial.params['up_max_depth']}, 'learning_rate': {trial.params['up_learning_rate']:.4f},
    'n_estimators': {trial.params['up_n_estimators']}, 'subsample': {trial.params['up_subsample']:.4f},
    'colsample_bytree': {trial.params['up_colsample_bytree']:.4f}, 'min_child_weight': {trial.params['up_min_child_weight']},
    'reg_alpha': {trial.params['up_reg_alpha']:.4f}, 'reg_lambda': {trial.params['up_reg_lambda']:.4f},
    'gamma': {trial.params['up_gamma']:.4f}, 'scale_pos_weight': {trial.params['up_scale_pos_weight']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}}

DOWN_CLASSIFIER_PARAMS = {{'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': {trial.params['down_max_depth']}, 'learning_rate': {trial.params['down_learning_rate']:.4f},
    'n_estimators': {trial.params['down_n_estimators']}, 'subsample': {trial.params['down_subsample']:.4f},
    'colsample_bytree': {trial.params['down_colsample_bytree']:.4f}, 'min_child_weight': {trial.params['down_min_child_weight']},
    'reg_alpha': {trial.params['down_reg_alpha']:.4f}, 'reg_lambda': {trial.params['down_reg_lambda']:.4f},
    'gamma': {trial.params['down_gamma']:.4f}, 'scale_pos_weight': {trial.params['down_scale_pos_weight']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': -1}}

# TEST PERFORMANCE (RAW): {attrs['accuracy']:.1%} (UP {attrs['up_accuracy']:.1%}, DOWN {attrs['down_accuracy']:.1%})
# MAE: {attrs['mae']:.2f}% (UP: {attrs['mae_up']:.2f}%, DOWN: {attrs['mae_down']:.2f}%)
# Validation: Exp {attrs['expansion_val_mae']:.2f}% Comp {attrs['compression_val_mae']:.2f}% UP {attrs['up_val_acc']:.1%} DOWN {attrs['down_val_acc']:.1%}
"""
        config_file = self.output_dir / "phase1_optimized_config.py"
        with open(config_file, 'w') as f: f.write(config_text)
        logger.info(f"✅ Config saved: {config_file}")

    def _print_summary(self, trial, attrs):
        logger.info("\n" + "="*80)
        logger.info("PHASE 1 OPTIMIZATION COMPLETE (RAW PREDICTIONS)")
        logger.info("="*80)
        logger.info(f"Best trial: #{trial.number} | Score: {trial.value:.3f}")
        logger.info("")
        logger.info("📊 RAW TEST SET PERFORMANCE (2024-2025):")
        logger.info(f"    Total: {int(attrs['total'])}")
        logger.info(f"    UP: {attrs['up_pct']:.1%} ({int(attrs['up_count'])} predictions)")
        logger.info(f"    DOWN: {attrs['down_pct']:.1%} ({int(attrs['down_count'])} predictions)")
        logger.info("")
        logger.info("📊 NATURAL DISTRIBUTION:")
        logger.info(f"   UP: {attrs['up_pct']:.1%} | DOWN: {attrs['down_pct']:.1%}")
        if attrs['up_pct'] < 0.30:
            logger.info(f"   ✓ System naturally favors DOWN (normal behavior)")
            logger.info(f"   → Phase 2 will apply up_advantage to boost UP recall")
        logger.info("")
        logger.info(f"    Overall accuracy: {attrs['accuracy']:.1%}")
        logger.info(f"    UP accuracy: {attrs['up_accuracy']:.1%}")
        logger.info(f"    DOWN accuracy: {attrs['down_accuracy']:.1%}")
        logger.info(f"    MAE: {attrs['mae']:.2f}% (UP: {attrs['mae_up']:.2f}%, DOWN: {attrs['mae_down']:.2f}%)")
        logger.info("")
        logger.info("  VALIDATION METRICS:")
        logger.info(f"    Expansion MAE: {attrs['expansion_val_mae']:.2f}%")
        logger.info(f"    Compression MAE: {attrs['compression_val_mae']:.2f}%")
        logger.info(f"    UP accuracy: {attrs['up_val_acc']:.1%}")
        logger.info(f"    DOWN accuracy: {attrs['down_val_acc']:.1%}")
        logger.info("")
        logger.info("  FEATURES SELECTED:")
        logger.info(f"    Expansion: {int(attrs['n_expansion_features'])}")
        logger.info(f"    Compression: {int(attrs['n_compression_features'])}")
        logger.info(f"    UP: {int(attrs['n_up_features'])}")
        logger.info(f"    DOWN: {int(attrs['n_down_features'])}")
        logger.info("="*80)
        logger.info("")
        logger.info("📝 NEXT STEPS:")
        logger.info("  1. Review results in tuning_phase1/phase1_results.json")
        logger.info("  2. Apply parameters from phase1_optimized_config.py to your config.py")
        logger.info("  3. Retrain models with optimized hyperparameters")
        logger.info("  4. Proceed to Phase 2: Ensemble config tuning on these RAW models")
        logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Model Hyperparameter Tuner (RAW predictions)")
    parser.add_argument('--trials', type=int, default=300, help="Number of optimization trials (default: 300)")
    parser.add_argument('--output-dir', type=str, default='tuning_phase1', help="Output directory (default: tuning_phase1)")
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
    df["vix"] = result["vix"]; df["spx"] = result["spx"]
    logger.info(f"Dataset: {len(df)} samples, {len(df.columns)} features")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    logger.info("")
    required_cols = ['calendar_cohort', 'cohort_weight', 'feature_quality']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)
    tuner = Phase1Tuner(df=df, vix=result["vix"], n_trials=args.trials, output_dir=args.output_dir)
    study = tuner.run()
    tuner.save_results(study)
    logger.info("\n✅ Phase 1 complete!")

if __name__ == "__main__": main()
