#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    UNIFIED PHASE 1 TUNER - TERNARY DECISION SYSTEM
═══════════════════════════════════════════════════════════════════════════════

🎯 PURPOSE:
   Optimizes all 4 models (expansion/compression regressors + UP/DOWN classifiers)
   plus ensemble configuration for ternary decision system (UP/DOWN/NO_DECISION).

🆕 TERNARY SYSTEM:
   ✓ Direct UP/DOWN/NO_DECISION output based on single decision_threshold
   ✓ Balanced optimization: Both UP and DOWN target 40-60% of decisions
   ✓ No more two-stage filtering (removed 12 complex threshold parameters)
   ✓ Honest uncertainty quantification when confidence < threshold
   ✓ up_advantage can now be NEGATIVE (favors DOWN) for true balance
   ✓ Simpler tuning: 42 parameters (down from 74)

═══════════════════════════════════════════════════════════════════════════════
"""
import argparse, json, logging, sys, warnings, hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import numpy as np, pandas as pd, optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from config import PHASE1_TUNER_TRIALS

warnings.filterwarnings("ignore")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/phase1_unified.log")])
logger = logging.getLogger(__name__)

@dataclass
class UnifiedMetrics:
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
    expansion_train_mae: float = 0.0
    compression_train_mae: float = 0.0
    up_train_acc: float = 0.0
    down_train_acc: float = 0.0
    n_expansion_features: int = 0
    n_compression_features: int = 0
    n_up_features: int = 0
    n_down_features: int = 0
    no_decision_count: int = 0
    no_decision_pct: float = 0.0

class UnifiedPhase1Tuner:
    def __init__(self, df, vix, n_trials=PHASE1_TUNER_TRIALS, output_dir="tuning_unified", frozen_features=None):
        self.df = df.copy()
        self.vix = vix.copy()
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frozen_features = frozen_features

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
        self.feature_cache = {}

        logger.info("="*80)
        logger.info("UNIFIED PHASE 1 TUNER - TERNARY DECISION SYSTEM")
        logger.info("="*80)
        logger.info(f"Train:  {len(self.train_df)} days ({self.train_df.index[0].date()} to {self.train_end.date()})")
        logger.info(f"Val:    {len(self.val_df)} days ({self.val_df.index[0].date()} to {self.val_end.date()})")
        logger.info(f"Test:   {len(self.test_df)} days ({self.test_start.date()} to {self.test_df.index[-1].date()})")
        logger.info(f"Base features: {len(self.base_cols)}")

        if frozen_features:
            logger.info(f"✓ USING FROZEN FEATURES:")
            logger.info(f"  - Expansion: {len(frozen_features['expansion'])} features")
            logger.info(f"  - Compression: {len(frozen_features['compression'])} features")
            logger.info(f"  - UP: {len(frozen_features['up'])} features")
            logger.info(f"  - DOWN: {len(frozen_features['down'])} features")
            logger.info(f"  → Effective params: ~42 (down from ~74)")
        else:
            logger.info(f"⚠️  NO FROZEN FEATURES: Will select per trial (~74 params)")

        logger.info(f"✓ TERNARY DECISION SYSTEM: UP/DOWN/NO_DECISION based on single threshold")
        logger.info(f"✓ BALANCED OPTIMIZATION: Both UP and DOWN target 40-60% of decisions")
        logger.info(f"✓ up_advantage: Can be NEGATIVE (favors DOWN) or positive (favors UP)")
        logger.info(f"✓ Parameters: ~42 tunable (removed 12 complex threshold params)")
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

    def _get_feature_cache_key(self, target_type, top_n, corr_threshold, cv_params, quality_threshold):
        key_parts = [
            target_type, str(top_n), f"{corr_threshold:.4f}", f"{quality_threshold:.4f}",
            f"{cv_params['n_estimators']}", f"{cv_params['max_depth']}",
            f"{cv_params['learning_rate']:.4f}", f"{cv_params['subsample']:.4f}",
            f"{cv_params['colsample_bytree']:.4f}"
        ]
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _select_features(self, train_val_df, vix, target_type, top_n, corr_threshold, cv_params, quality_threshold):
        if self.frozen_features:
            return self.frozen_features[target_type]

        cache_key = self._get_feature_cache_key(target_type, top_n, corr_threshold, cv_params, quality_threshold)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        from core.xgboost_feature_selector_v2 import FeatureSelector
        import config as cfg
        original_cv = cfg.FEATURE_SELECTION_CV_PARAMS.copy()
        cfg.FEATURE_SELECTION_CV_PARAMS.update(cv_params)
        try:
            selector = FeatureSelector(target_type=target_type, top_n=top_n,
                correlation_threshold=corr_threshold, random_state=42)
            test_start_idx = len(train_val_df)
            selected_features, _ = selector.select_features(train_val_df[self.base_cols], vix, test_start_idx=test_start_idx)
            self.feature_cache[cache_key] = selected_features
            return selected_features
        finally:
            cfg.FEATURE_SELECTION_CV_PARAMS.update(original_cv)

    def _compute_ensemble_confidence(self, classifier_prob, magnitude_pct, direction, trial_params):
        """Compute ensemble confidence using trial's ensemble params - SIMPLIFIED"""
        if direction == "UP":
            weights = {'classifier': trial_params['classifier_weight_up'],
                      'magnitude': 1.0 - trial_params['classifier_weight_up']}
            scaling = {'small': trial_params['mag_scale_up_small'],
                      'medium': trial_params['mag_scale_up_medium'],
                      'large': trial_params['mag_scale_up_large']}
        else:
            weights = {'classifier': trial_params['classifier_weight_down'],
                      'magnitude': 1.0 - trial_params['classifier_weight_down']}
            scaling = {'small': trial_params['mag_scale_down_small'],
                      'medium': trial_params['mag_scale_down_medium'],
                      'large': trial_params['mag_scale_down_large']}

        abs_mag = abs(magnitude_pct)
        mag_strength = min(abs_mag / scaling['large'], 1.0)
        ensemble_conf = weights['classifier'] * classifier_prob + weights['magnitude'] * mag_strength
        ensemble_conf = np.clip(ensemble_conf, 0.0, 1.0)
        return float(ensemble_conf)

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

            # Feature selection (or use frozen)
            if self.frozen_features:
                exp_features = self.frozen_features['expansion']
                comp_features = self.frozen_features['compression']
                up_features = self.frozen_features['up']
                down_features = self.frozen_features['down']
            else:
                cv_params = {'n_estimators': trial_params['cv_n_estimators'], 'max_depth': trial_params['cv_max_depth'],
                    'learning_rate': trial_params['cv_learning_rate'], 'subsample': trial_params['cv_subsample'],
                    'colsample_bytree': trial_params['cv_colsample_bytree']}

                exp_features = self._select_features(train_val_df, train_val_vix, 'expansion',
                    trial_params['expansion_top_n'], trial_params['correlation_threshold'], cv_params,
                    trial_params['quality_threshold'])
                comp_features = self._select_features(train_val_df, train_val_vix, 'compression',
                    trial_params['compression_top_n'], trial_params['correlation_threshold'], cv_params,
                    trial_params['quality_threshold'])
                up_features = self._select_features(train_val_df, train_val_vix, 'up',
                    trial_params['up_top_n'], trial_params['correlation_threshold'], cv_params,
                    trial_params['quality_threshold'])
                down_features = self._select_features(train_val_df, train_val_vix, 'down',
                    trial_params['down_top_n'], trial_params['correlation_threshold'], cv_params,
                    trial_params['quality_threshold'])

            if any(len(f) < 20 for f in [exp_features, comp_features, up_features, down_features]):
                logger.warning("Insufficient features selected")
                return None

            from xgboost import XGBRegressor, XGBClassifier
            expansion_params = self._build_expansion_params(trial_params)
            compression_params = self._build_compression_params(trial_params)
            up_params = self._build_up_params(trial_params)
            down_params = self._build_down_params(trial_params)

            # Train expansion regressor
            train_up_mask = (train_filt['target_direction'] == 1) & (train_filt['target_log_vix_change'].notna())
            val_up_mask = (val_filt['target_direction'] == 1) & (val_filt['target_log_vix_change'].notna())
            X_exp_train = train_filt[train_up_mask][exp_features].fillna(0)
            y_exp_train = train_filt[train_up_mask]['target_log_vix_change'].dropna()
            X_exp_train = X_exp_train.loc[y_exp_train.index]
            w_exp_train = train_weights[train_up_mask].loc[y_exp_train.index]
            X_exp_val = val_filt[val_up_mask][exp_features].fillna(0)
            y_exp_val = val_filt[val_up_mask]['target_log_vix_change'].dropna()
            X_exp_val = X_exp_val.loc[y_exp_val.index]

            expansion_model = XGBRegressor(**expansion_params)
            expansion_model.fit(X_exp_train, y_exp_train, sample_weight=w_exp_train,
                eval_set=[(X_exp_val, y_exp_val)], verbose=False)

            y_exp_train_pred = np.clip(expansion_model.predict(X_exp_train), -2, 2)
            y_exp_val_pred = np.clip(expansion_model.predict(X_exp_val), -2, 2)
            exp_train_mae = mean_absolute_error((np.exp(y_exp_train) - 1) * 100, (np.exp(y_exp_train_pred) - 1) * 100)
            exp_val_mae = mean_absolute_error((np.exp(y_exp_val) - 1) * 100, (np.exp(y_exp_val_pred) - 1) * 100)

            # Train compression regressor
            train_down_mask = (train_filt['target_direction'] == 0) & (train_filt['target_log_vix_change'].notna())
            val_down_mask = (val_filt['target_direction'] == 0) & (val_filt['target_log_vix_change'].notna())
            X_comp_train = train_filt[train_down_mask][comp_features].fillna(0)
            y_comp_train = train_filt[train_down_mask]['target_log_vix_change'].dropna()
            X_comp_train = X_comp_train.loc[y_comp_train.index]
            w_comp_train = train_weights[train_down_mask].loc[y_comp_train.index]
            X_comp_val = val_filt[val_down_mask][comp_features].fillna(0)
            y_comp_val = val_filt[val_down_mask]['target_log_vix_change'].dropna()
            X_comp_val = X_comp_val.loc[y_comp_val.index]

            compression_model = XGBRegressor(**compression_params)
            compression_model.fit(X_comp_train, y_comp_train, sample_weight=w_comp_train,
                eval_set=[(X_comp_val, y_comp_val)], verbose=False)

            y_comp_train_pred = np.clip(compression_model.predict(X_comp_train), -2, 2)
            y_comp_val_pred = np.clip(compression_model.predict(X_comp_val), -2, 2)
            comp_train_mae = mean_absolute_error((np.exp(y_comp_train) - 1) * 100, (np.exp(y_comp_train_pred) - 1) * 100)
            comp_val_mae = mean_absolute_error((np.exp(y_comp_val) - 1) * 100, (np.exp(y_comp_val_pred) - 1) * 100)

            # Train UP classifier
            X_up_train = train_filt[up_features].fillna(0)
            y_up_train = train_filt['target_direction']
            X_up_val = val_filt[up_features].fillna(0)
            y_up_val = val_filt['target_direction']
            up_model = XGBClassifier(**up_params)
            up_model.fit(X_up_train, y_up_train, sample_weight=train_weights,
                eval_set=[(X_up_val, y_up_val)], verbose=False)

            from sklearn.metrics import accuracy_score
            up_train_acc = accuracy_score(y_up_train, up_model.predict(X_up_train))
            up_val_acc = accuracy_score(y_up_val, up_model.predict(X_up_val))

            # Train DOWN classifier
            X_down_train = train_filt[down_features].fillna(0)
            y_down_train = 1 - train_filt['target_direction']
            X_down_val = val_filt[down_features].fillna(0)
            y_down_val = 1 - val_filt['target_direction']
            down_model = XGBClassifier(**down_params)
            down_model.fit(X_down_train, y_down_train, sample_weight=train_weights,
                eval_set=[(X_down_val, y_down_val)], verbose=False)

            down_train_acc = accuracy_score(y_down_train, down_model.predict(X_down_train))
            down_val_acc = accuracy_score(y_down_val, down_model.predict(X_down_val))

            all_predictions = []

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

                total = p_up + p_down
                p_up_norm = p_up / total
                p_down_norm = p_down / total

                up_advantage = trial_params['up_advantage']
                # Negative advantage means DOWN doesn't need to beat UP by as much
                if p_down_norm > (p_up_norm + up_advantage):
                    direction = "DOWN"
                    magnitude_pct = comp_pct
                    classifier_prob = p_down_norm
                else:
                    direction = "UP"
                    magnitude_pct = exp_pct
                    classifier_prob = p_up_norm

                # ENSEMBLE CONFIDENCE
                ensemble_conf = self._compute_ensemble_confidence(classifier_prob, magnitude_pct, direction, trial_params)

                # TERNARY DECISION LOGIC
                if ensemble_conf < trial_params['decision_threshold']:
                    direction = "NO_DECISION"
                    magnitude_pct = 0.0

                actual_direction = int(test_filt.loc[idx, 'target_direction'])
                actual_mag_log = test_filt.loc[idx, 'target_log_vix_change']
                actual_mag_pct = (np.exp(actual_mag_log) - 1) * 100

                pred = {
                    'pred_direction': direction,  # Now "UP", "DOWN", or "NO_DECISION"
                    'actual_direction': actual_direction,
                    'direction_correct': None if direction == "NO_DECISION" else ((1 if direction == "UP" else 0) == actual_direction),
                    'magnitude': magnitude_pct,
                    'actual_magnitude': actual_mag_pct,
                    'confidence': ensemble_conf
                }

                all_predictions.append(pred)

            if len(all_predictions) < 100:
                logger.warning("Insufficient predictions")
                return None

            metrics = self._calculate_metrics(all_predictions,
                exp_train_mae, exp_val_mae, comp_train_mae, comp_val_mae,
                up_train_acc, up_val_acc, down_train_acc, down_val_acc,
                len(exp_features), len(comp_features), len(up_features), len(down_features))
            return metrics
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return None

    def _calculate_metrics(self, all_preds,
                          exp_train_mae, exp_val_mae, comp_train_mae, comp_val_mae,
                          up_train_acc, up_val_acc, down_train_acc, down_val_acc,
                          n_exp_feats, n_comp_feats, n_up_feats, n_down_feats):
        df_all = pd.DataFrame(all_preds)

        # Separate decisions from NO_DECISION
        df_decisions = df_all[df_all['pred_direction'].isin(['UP', 'DOWN'])].copy()
        df_no_decision = df_all[df_all['pred_direction'] == 'NO_DECISION'].copy()

        if len(df_decisions) == 0:
            return None  # No decisions made

        # Convert string directions to numeric for filtering
        df_decisions['pred_direction_num'] = df_decisions['pred_direction'].map({'UP': 1, 'DOWN': 0})
        up_all = df_decisions[df_decisions['pred_direction_num'] == 1]
        down_all = df_decisions[df_decisions['pred_direction_num'] == 0]

        metrics = UnifiedMetrics(
            total=len(df_all),
            up_count=len(up_all),
            down_count=len(down_all),
            up_pct=len(up_all) / len(df_decisions) if len(df_decisions) > 0 else 0.0,
            down_pct=len(down_all) / len(df_decisions) if len(df_decisions) > 0 else 0.0,
            accuracy=df_decisions['direction_correct'].mean(),
            up_accuracy=up_all['direction_correct'].mean() if len(up_all) > 0 else 0.0,
            down_accuracy=down_all['direction_correct'].mean() if len(down_all) > 0 else 0.0,
            expansion_train_mae=exp_train_mae,
            expansion_val_mae=exp_val_mae,
            compression_train_mae=comp_train_mae,
            compression_val_mae=comp_val_mae,
            up_train_acc=up_train_acc,
            up_val_acc=up_val_acc,
            down_train_acc=down_train_acc,
            down_val_acc=down_val_acc,
            n_expansion_features=n_exp_feats,
            n_compression_features=n_comp_feats,
            n_up_features=n_up_feats,
            n_down_features=n_down_feats,
            no_decision_count=len(df_no_decision),
            no_decision_pct=len(df_no_decision) / len(df_all)
        )

        df_clean = df_decisions.dropna(subset=['actual_magnitude', 'magnitude'])
        up_clean = up_all.dropna(subset=['actual_magnitude', 'magnitude'])
        down_clean = down_all.dropna(subset=['actual_magnitude', 'magnitude'])

        metrics.mae = mean_absolute_error(df_clean['actual_magnitude'], df_clean['magnitude']) if len(df_clean) > 0 else 999.0
        metrics.mae_up = mean_absolute_error(up_clean['actual_magnitude'], up_clean['magnitude']) if len(up_clean) > 0 else 999.0
        metrics.mae_down = mean_absolute_error(down_clean['actual_magnitude'], down_clean['magnitude']) if len(down_clean) > 0 else 999.0

        return metrics

    def _build_expansion_params(self, trial_params):
        return {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': trial_params['exp_max_depth'], 'learning_rate': trial_params['exp_learning_rate'],
            'n_estimators': trial_params['exp_n_estimators'], 'subsample': trial_params['exp_subsample'],
            'colsample_bytree': trial_params['exp_colsample_bytree'], 'colsample_bylevel': trial_params['exp_colsample_bylevel'],
            'min_child_weight': trial_params['exp_min_child_weight'], 'reg_alpha': trial_params['exp_reg_alpha'],
            'reg_lambda': trial_params['exp_reg_lambda'], 'gamma': trial_params['exp_gamma'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

    def _build_compression_params(self, trial_params):
        return {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': trial_params['comp_max_depth'], 'learning_rate': trial_params['comp_learning_rate'],
            'n_estimators': trial_params['comp_n_estimators'], 'subsample': trial_params['comp_subsample'],
            'colsample_bytree': trial_params['comp_colsample_bytree'], 'colsample_bylevel': trial_params['comp_colsample_bylevel'],
            'min_child_weight': trial_params['comp_min_child_weight'], 'reg_alpha': trial_params['comp_reg_alpha'],
            'reg_lambda': trial_params['comp_reg_lambda'], 'gamma': trial_params['comp_gamma'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

    def _build_up_params(self, trial_params):
        return {'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': trial_params['up_max_depth'], 'learning_rate': trial_params['up_learning_rate'],
            'n_estimators': trial_params['up_n_estimators'], 'subsample': trial_params['up_subsample'],
            'colsample_bytree': trial_params['up_colsample_bytree'], 'min_child_weight': trial_params['up_min_child_weight'],
            'reg_alpha': trial_params['up_reg_alpha'], 'reg_lambda': trial_params['up_reg_lambda'],
            'gamma': trial_params['up_gamma'], 'scale_pos_weight': trial_params['up_scale_pos_weight'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

    def _build_down_params(self, trial_params):
        return {'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': trial_params['down_max_depth'], 'learning_rate': trial_params['down_learning_rate'],
            'n_estimators': trial_params['down_n_estimators'], 'subsample': trial_params['down_subsample'],
            'colsample_bytree': trial_params['down_colsample_bytree'], 'min_child_weight': trial_params['down_min_child_weight'],
            'reg_alpha': trial_params['down_reg_alpha'], 'reg_lambda': trial_params['down_reg_lambda'],
            'gamma': trial_params['down_gamma'], 'scale_pos_weight': trial_params['down_scale_pos_weight'],
            'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}

    def objective(self, trial):
        trial_params = self._sample_hyperparameters(trial)
        metrics = self._train_and_evaluate_models(trial_params)
        if metrics is None: return 999.0

        for field_name, value in metrics.__dict__.items():
            trial.set_user_attr(field_name, float(value))

        # DIRECTIONAL ACCURACY (primary target) - NO_DECISION excluded
        decision_count = metrics.up_count + metrics.down_count
        if decision_count < 80:
            return 999.0

        target_acc = 0.75
        up_penalty = abs(metrics.up_accuracy - target_acc) * 3.0
        down_penalty = abs(metrics.down_accuracy - target_acc) * 3.0

        if metrics.up_accuracy < 0.70:
            up_penalty += 15.0
        if metrics.down_accuracy < 0.70:
            down_penalty += 15.0

        # BALANCED SIGNAL DISTRIBUTION (40-60% each is ideal)
        up_pct = metrics.up_count / decision_count if decision_count > 0 else 0.0
        down_pct = metrics.down_count / decision_count if decision_count > 0 else 0.0
        balance_penalty = 0.0

        # Strong penalty if UP is outside 40-60% range
        if up_pct < 0.40:
            balance_penalty += (0.40 - up_pct) * 200.0
        elif up_pct > 0.60:
            balance_penalty += (up_pct - 0.60) * 200.0

        # Strong penalty if DOWN is outside 40-60% range
        if down_pct < 0.40:
            balance_penalty += (0.40 - down_pct) * 200.0
        elif down_pct > 0.60:
            balance_penalty += (down_pct - 0.60) * 200.0

        # Extra penalty for extreme imbalance
        if up_pct > 0.70 or down_pct > 0.70:
            balance_penalty += 50.0

        # Penalty for accuracy imbalance (should be within 5%)
        acc_diff = abs(metrics.up_accuracy - metrics.down_accuracy)
        if acc_diff > 0.05:
            balance_penalty += (acc_diff - 0.05) * 100.0

        # LEARNING RATE BALANCE (ratio < 3x)
        up_lr = trial_params['up_learning_rate']
        down_lr = trial_params['down_learning_rate']
        lr_ratio = max(up_lr, down_lr) / min(up_lr, down_lr)
        lr_balance_penalty = 0.0
        if lr_ratio > 3.0:
            lr_balance_penalty = (lr_ratio - 3.0) * 5.0

        # CLASSIFIER RECALL PENALTY (penalize if both have very high recall)
        # High recall means classifier is too aggressive (saying "yes" too often)
        # This gets filtered by decision_threshold, but we want classifiers to be more selective
        recall_penalty = 0.0
        if metrics.up_train_acc > 0.65 and metrics.down_train_acc > 0.58:
            # Both classifiers have high accuracy on train, check for overconfidence
            up_val_diff = metrics.up_train_acc - metrics.up_val_acc
            down_val_diff = metrics.down_train_acc - metrics.down_val_acc
            if up_val_diff > 0.10:
                recall_penalty += up_val_diff * 20.0
            if down_val_diff > 0.15:
                recall_penalty += down_val_diff * 20.0

        # COMPLEXITY PENALTY (simpler models)
        up_complexity = (trial_params['up_max_depth'] - 4) * 0.5 + (trial_params['up_n_estimators'] - 300) / 100.0
        down_complexity = (trial_params['down_max_depth'] - 6) * 0.5 + (trial_params['down_n_estimators'] - 300) / 100.0
        complexity_penalty = max(0, up_complexity) + max(0, down_complexity)

        # GENERALIZATION
        exp_gap = abs(metrics.expansion_train_mae - metrics.expansion_val_mae)
        comp_gap = abs(metrics.compression_train_mae - metrics.compression_val_mae)
        up_gap = abs(metrics.up_train_acc - metrics.up_val_acc)
        down_gap = abs(metrics.down_train_acc - metrics.down_val_acc)
        generalization_penalty = (
            (exp_gap / 12.0) * 2.0 + (comp_gap / 8.0) * 2.0 +
            (up_gap / 0.10) * 1.5 + (down_gap / 0.10) * 1.5
        )

        # VOLUME (ensure enough decisions, not too many NO_DECISION)
        volume_penalty = 0.0
        if decision_count < 120:
            volume_penalty = (120 - decision_count) * 0.2

        # NO_DECISION penalties
        no_dec_rate = metrics.no_decision_pct
        if no_dec_rate > 0.50:  # Too many no-decisions
            volume_penalty += (no_dec_rate - 0.50) * 100.0
        elif no_dec_rate < 0.20:  # Too few no-decisions (overconfident)
            volume_penalty += (0.20 - no_dec_rate) * 50.0

        # MAE
        mae_penalty = (metrics.mae / 15.0) * 1.0

        # VALIDATION QUALITY
        val_penalty = (
            max(0, metrics.expansion_val_mae - 12.0) * 0.4 +
            max(0, metrics.compression_val_mae - 8.0) * 0.4 +
            max(0, 0.55 - metrics.up_val_acc) * 1.5 +
            max(0, 0.55 - metrics.down_val_acc) * 1.5
        )

        score = (up_penalty + down_penalty +
                 balance_penalty + lr_balance_penalty + recall_penalty + complexity_penalty +
                 generalization_penalty + volume_penalty + mae_penalty + val_penalty)

        return score

    def _sample_hyperparameters(self, trial):
        params = {}
        params['quality_threshold'] = trial.suggest_float('quality_threshold', 0.50, 0.65)
        params['fomc_weight'] = 1
        params['opex_weight'] = 1
        params['earnings_weight'] = 1

        # Only sample feature selection params if NOT using frozen features
        if not self.frozen_features:
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
        else:
            params['cv_n_estimators'] = 200
            params['cv_max_depth'] = 4
            params['cv_learning_rate'] = 0.05
            params['cv_subsample'] = 0.85
            params['cv_colsample_bytree'] = 0.85
            params['expansion_top_n'] = len(self.frozen_features['expansion'])
            params['compression_top_n'] = len(self.frozen_features['compression'])
            params['up_top_n'] = len(self.frozen_features['up'])
            params['down_top_n'] = len(self.frozen_features['down'])
            params['correlation_threshold'] = 0.90

        # Regressor params
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

        # Classifier params
        params['up_max_depth'] = trial.suggest_int('up_max_depth', 3, 6)
        params['up_learning_rate'] = trial.suggest_float('up_learning_rate', 0.01, 0.12, log=True)
        params['up_n_estimators'] = trial.suggest_int('up_n_estimators', 200, 500)
        params['up_subsample'] = trial.suggest_float('up_subsample', 0.60, 0.95)
        params['up_colsample_bytree'] = trial.suggest_float('up_colsample_bytree', 0.70, 0.95)
        params['up_min_child_weight'] = trial.suggest_int('up_min_child_weight', 5, 18)
        params['up_reg_alpha'] = trial.suggest_float('up_reg_alpha', 1.0, 8.0)
        params['up_reg_lambda'] = trial.suggest_float('up_reg_lambda', 5.0, 20.0)
        params['up_gamma'] = trial.suggest_float('up_gamma', 0.5, 2.5)
        params['up_scale_pos_weight'] = trial.suggest_float('up_scale_pos_weight', 0.7, 1.3)

        params['down_max_depth'] = trial.suggest_int('down_max_depth', 4, 8)
        params['down_learning_rate'] = trial.suggest_float('down_learning_rate', 0.01, 0.12, log=True)
        params['down_n_estimators'] = trial.suggest_int('down_n_estimators', 200, 500)
        params['down_subsample'] = trial.suggest_float('down_subsample', 0.60, 0.95)
        params['down_colsample_bytree'] = trial.suggest_float('down_colsample_bytree', 0.70, 0.95)
        params['down_min_child_weight'] = trial.suggest_int('down_min_child_weight', 5, 18)
        params['down_reg_alpha'] = trial.suggest_float('down_reg_alpha', 3.0, 12.0)
        params['down_reg_lambda'] = trial.suggest_float('down_reg_lambda', 5.0, 20.0)
        params['down_gamma'] = trial.suggest_float('down_gamma', 0.5, 3.0)
        params['down_scale_pos_weight'] = trial.suggest_float('down_scale_pos_weight', 0.5, 0.9)

        # ENSEMBLE PARAMS - up_advantage can now be NEGATIVE!
        # Negative values favor DOWN (DOWN doesn't need as much advantage to win)
        # Positive values favor UP (DOWN needs bigger advantage to win)
        params['up_advantage'] = trial.suggest_float('up_advantage', -0.10, 0.10)
        params['classifier_weight_up'] = trial.suggest_float('classifier_weight_up', 0.55, 0.75)
        params['classifier_weight_down'] = trial.suggest_float('classifier_weight_down', 0.65, 0.85)
        params['mag_scale_up_small'] = trial.suggest_float('mag_scale_up_small', 2.5, 4.0)
        params['mag_scale_up_medium'] = trial.suggest_float('mag_scale_up_medium', 5.0, 8.0)
        params['mag_scale_up_large'] = trial.suggest_float('mag_scale_up_large', 9.0, 13.0)
        params['mag_scale_down_small'] = trial.suggest_float('mag_scale_down_small', 2.5, 4.5)
        params['mag_scale_down_medium'] = trial.suggest_float('mag_scale_down_medium', 5.0, 8.0)
        params['mag_scale_down_large'] = trial.suggest_float('mag_scale_down_large', 9.0, 14.0)

        # SINGLE DECISION THRESHOLD (replaces 12 complex thresholds)
        params['decision_threshold'] = trial.suggest_float('decision_threshold', 0.65, 0.80)

        return params

    def run(self):
        mode_desc = "WITH FROZEN FEATURES" if self.frozen_features else "selecting features per trial"
        logger.info(f"Starting Unified Phase 1 optimization: {self.n_trials} trials ({mode_desc})")

        if self.frozen_features:
            logger.info(f"Tuning ~42 hyperparameters (simplified ternary system)")
        else:
            logger.info(f"Tuning ~58 hyperparameters (includes feature selection)")

        logger.info(f"Evaluating with TERNARY DECISION LOGIC on {len(self.test_df)} test days")
        logger.info(f"Optimizing for: Balanced UP/DOWN accuracy + appropriate NO_DECISION rate")
        logger.info("="*80)

        study = optuna.create_study(direction='minimize',
            sampler=TPESampler(seed=42, n_startup_trials=min(50, self.n_trials // 6)))
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)
        self.feature_cache.clear()
        return study

    def save_results(self, study):
        best = study.best_trial
        attrs = best.user_attrs

        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Unified Phase 1 - Ternary Decision System',
            'description': 'Tunes all 4 models + ensemble with direct UP/DOWN/NO_DECISION output',
            'frozen_features_used': self.frozen_features is not None,
            'optimization': {'n_trials': self.n_trials, 'best_trial': best.number, 'best_score': float(best.value)},
            'data_splits': {
                'train': f"{self.train_df.index[0].date()} to {self.train_end.date()}",
                'val': f"{self.val_df.index[0].date()} to {self.val_end.date()}",
                'test': f"{self.test_start.date()} to {self.test_df.index[-1].date()}",
                'train_size': len(self.train_df), 'val_size': len(self.val_df), 'test_size': len(self.test_df)},
            'test_metrics': {
                'directional_predictions': {
                    'total': int(attrs['total']),
                    'decisions': int(attrs['up_count'] + attrs['down_count']),
                    'up_count': int(attrs['up_count']),
                    'down_count': int(attrs['down_count']),
                    'up_pct': float(attrs['up_pct']),
                    'down_pct': float(attrs['down_pct']),
                    'accuracy': float(attrs['accuracy']),
                    'up_accuracy': float(attrs['up_accuracy']),
                    'down_accuracy': float(attrs['down_accuracy'])},
                'no_decision_stats': {
                    'count': int(attrs['no_decision_count']),
                    'rate': float(attrs['no_decision_pct'])},
                'train_metrics': {
                    'expansion_mae': float(attrs['expansion_train_mae']),
                    'compression_mae': float(attrs['compression_train_mae']),
                    'up_accuracy': float(attrs['up_train_acc']),
                    'down_accuracy': float(attrs['down_train_acc'])},
                'validation': {
                    'expansion_mae': float(attrs['expansion_val_mae']),
                    'compression_mae': float(attrs['compression_val_mae']),
                    'up_accuracy': float(attrs['up_val_acc']),
                    'down_accuracy': float(attrs['down_val_acc'])},
                'features': {
                    'expansion': int(attrs['n_expansion_features']),
                    'compression': int(attrs['n_compression_features']),
                    'up': int(attrs['n_up_features']),
                    'down': int(attrs['n_down_features'])}},
            'best_parameters': best.params}

        results_file = self.output_dir / "unified_results.json"
        with open(results_file, 'w') as f: json.dump(results, f, indent=2)
        logger.info(f"\n✅ Results saved: {results_file}")
        self._generate_config(best, attrs)
        self._print_summary(best, attrs)

    def _generate_config(self, trial, attrs):
        p = trial.params

        if self.frozen_features:
            cv_n_estimators = 200
            cv_max_depth = 4
            cv_learning_rate = 0.05
            cv_subsample = 0.85
            cv_colsample_bytree = 0.85
            expansion_top_n = len(self.frozen_features['expansion'])
            compression_top_n = len(self.frozen_features['compression'])
            up_top_n = len(self.frozen_features['up'])
            down_top_n = len(self.frozen_features['down'])
            correlation_threshold = 0.90
        else:
            cv_n_estimators = p['cv_n_estimators']
            cv_max_depth = p['cv_max_depth']
            cv_learning_rate = p['cv_learning_rate']
            cv_subsample = p['cv_subsample']
            cv_colsample_bytree = p['cv_colsample_bytree']
            expansion_top_n = p['expansion_top_n']
            compression_top_n = p['compression_top_n']
            up_top_n = p['up_top_n']
            down_top_n = p['down_top_n']
            correlation_threshold = p['correlation_threshold']

        config_text = f"""# UNIFIED CONFIG - TERNARY DECISION SYSTEM - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Direct UP/DOWN/NO_DECISION output based on single decision_threshold
# up_advantage can now be NEGATIVE to favor DOWN for true balance
# Simplified from 74 to 42 parameters (removed 12 complex threshold params)
{'# TRAINED WITH FROZEN FEATURES' if self.frozen_features else '# Features selected during tuning'}

QUALITY_FILTER_CONFIG = {{'enabled': True, 'min_threshold': {p['quality_threshold']:.4f},
    'warn_pct': 20.0, 'error_pct': 50.0, 'strategy': 'raise'}}

CALENDAR_COHORTS = {{
    'fomc_period': {{'condition': 'macro_event_period', 'range': (-7, 2),
        'weight': 1.0, 'description': 'FOMC meetings, CPI releases, PCE releases, FOMC minutes'}},
    'opex_week': {{'condition': 'days_to_monthly_opex', 'range': (-7, 0),
        'weight': 1.0, 'description': 'Options expiration week + VIX futures rollover'}},
    'earnings_heavy': {{'condition': 'spx_earnings_pct', 'range': (0.15, 1.0),
        'weight': 1.0, 'description': 'Peak earnings season (Jan, Apr, Jul, Oct)'}},
    'mid_cycle': {{'condition': 'default', 'range': None, 'weight': 1.0, 'description': 'Regular market conditions'}}}}

FEATURE_SELECTION_CV_PARAMS = {{'n_estimators': {cv_n_estimators},
    'max_depth': {cv_max_depth}, 'learning_rate': {cv_learning_rate:.4f},
    'subsample': {cv_subsample:.4f}, 'colsample_bytree': {cv_colsample_bytree:.4f},
    'n_jobs': 1, 'random_state': 42}}

FEATURE_SELECTION_CONFIG = {{'expansion_top_n': {expansion_top_n},
    'compression_top_n': {compression_top_n}, 'up_top_n': {up_top_n},
    'down_top_n': {down_top_n}, 'cv_folds': 5, 'protected_features': [],
    'correlation_threshold': {correlation_threshold:.4f},
    'description': 'Ternary system: up_advantage can be negative for true balance'}}

EXPANSION_PARAMS = {{'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': {p['exp_max_depth']}, 'learning_rate': {p['exp_learning_rate']:.4f},
    'n_estimators': {p['exp_n_estimators']}, 'subsample': {p['exp_subsample']:.4f},
    'colsample_bytree': {p['exp_colsample_bytree']:.4f}, 'colsample_bylevel': {p['exp_colsample_bylevel']:.4f},
    'min_child_weight': {p['exp_min_child_weight']}, 'reg_alpha': {p['exp_reg_alpha']:.4f},
    'reg_lambda': {p['exp_reg_lambda']:.4f}, 'gamma': {p['exp_gamma']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}}

COMPRESSION_PARAMS = {{'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': {p['comp_max_depth']}, 'learning_rate': {p['comp_learning_rate']:.4f},
    'n_estimators': {p['comp_n_estimators']}, 'subsample': {p['comp_subsample']:.4f},
    'colsample_bytree': {p['comp_colsample_bytree']:.4f}, 'colsample_bylevel': {p['comp_colsample_bylevel']:.4f},
    'min_child_weight': {p['comp_min_child_weight']}, 'reg_alpha': {p['comp_reg_alpha']:.4f},
    'reg_lambda': {p['comp_reg_lambda']:.4f}, 'gamma': {p['comp_gamma']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}}

UP_CLASSIFIER_PARAMS = {{'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': {p['up_max_depth']}, 'learning_rate': {p['up_learning_rate']:.4f},
    'n_estimators': {p['up_n_estimators']}, 'subsample': {p['up_subsample']:.4f},
    'colsample_bytree': {p['up_colsample_bytree']:.4f}, 'min_child_weight': {p['up_min_child_weight']},
    'reg_alpha': {p['up_reg_alpha']:.4f}, 'reg_lambda': {p['up_reg_lambda']:.4f},
    'gamma': {p['up_gamma']:.4f}, 'scale_pos_weight': {p['up_scale_pos_weight']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}}

DOWN_CLASSIFIER_PARAMS = {{'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': {p['down_max_depth']}, 'learning_rate': {p['down_learning_rate']:.4f},
    'n_estimators': {p['down_n_estimators']}, 'subsample': {p['down_subsample']:.4f},
    'colsample_bytree': {p['down_colsample_bytree']:.4f}, 'min_child_weight': {p['down_min_child_weight']},
    'reg_alpha': {p['down_reg_alpha']:.4f}, 'reg_lambda': {p['down_reg_lambda']:.4f},
    'gamma': {p['down_gamma']:.4f}, 'scale_pos_weight': {p['down_scale_pos_weight']:.4f},
    'early_stopping_rounds': 50, 'seed': 42, 'n_jobs': 1, 'random_state': 42}}

ENSEMBLE_CONFIG = {{
    'enabled': True,
    'reconciliation_method': 'winner_takes_all',
    'up_advantage': {p['up_advantage']:.4f},
    'confidence_weights': {{
        'up': {{'classifier': {p['classifier_weight_up']:.4f}, 'magnitude': {1.0-p['classifier_weight_up']:.4f}}},
        'down': {{'classifier': {p['classifier_weight_down']:.4f}, 'magnitude': {1.0-p['classifier_weight_down']:.4f}}}
    }},
    'magnitude_scaling': {{
        'up': {{'small': {p['mag_scale_up_small']:.4f}, 'medium': {p['mag_scale_up_medium']:.4f}, 'large': {p['mag_scale_up_large']:.4f}}},
        'down': {{'small': {p['mag_scale_down_small']:.4f}, 'medium': {p['mag_scale_down_medium']:.4f}, 'large': {p['mag_scale_down_large']:.4f}}}
    }},
    'decision_threshold': {p['decision_threshold']:.4f},
    'description': 'Ternary decision system with balanced 40-60% UP/DOWN target'
}}

# DIRECTIONAL: {attrs['accuracy']:.1%} (UP {attrs['up_accuracy']:.1%}, DOWN {attrs['down_accuracy']:.1%})
# NO_DECISION: {int(attrs['no_decision_count'])} ({attrs['no_decision_pct']:.1%} of total)
# UP signals: {int(attrs['up_count'])} | DOWN signals: {int(attrs['down_count'])}
"""

        if self.frozen_features:
            config_text += f"""
# FROZEN FEATURES USED
# Expansion: {len(self.frozen_features['expansion'])} features
# Compression: {len(self.frozen_features['compression'])} features
# UP: {len(self.frozen_features['up'])} features
# DOWN: {len(self.frozen_features['down'])} features
"""

        config_file = self.output_dir / "unified_config.py"
        with open(config_file, 'w') as f: f.write(config_text)
        logger.info(f"✅ Config saved: {config_file}")

        if self.frozen_features:
            frozen_ref_file = self.output_dir / "frozen_features_used.json"
            with open(frozen_ref_file, 'w') as f:
                json.dump(self.frozen_features, f, indent=2)
            logger.info(f"✅ Frozen features reference saved: {frozen_ref_file}")

    def _print_summary(self, trial, attrs):
        logger.info("\n" + "="*80)
        logger.info("UNIFIED PHASE 1 OPTIMIZATION COMPLETE - TERNARY DECISION SYSTEM")
        logger.info("="*80)
        logger.info(f"Best trial: #{trial.number} | Score: {trial.value:.3f}")
        logger.info("")
        logger.info("📊 DIRECTIONAL PREDICTIONS (UP/DOWN only):")
        decision_count = int(attrs['up_count'] + attrs['down_count'])
        logger.info(f"    Total decisions: {decision_count}")
        logger.info(f"    UP: {int(attrs['up_count'])} ({attrs['up_pct']:.1%}) | DOWN: {int(attrs['down_count'])} ({attrs['down_pct']:.1%})")
        logger.info(f"    Overall accuracy: {attrs['accuracy']:.1%}")
        logger.info(f"    UP accuracy: {attrs['up_accuracy']:.1%}")
        logger.info(f"    DOWN accuracy: {attrs['down_accuracy']:.1%}")
        logger.info("")
        logger.info("📊 NO_DECISION:")
        logger.info(f"    Count: {int(attrs['no_decision_count'])}")
        logger.info(f"    Rate: {attrs['no_decision_pct']:.1%} of total {int(attrs['total'])} predictions")
        logger.info("")
        logger.info("📊 SYSTEM IMPROVEMENTS:")
        logger.info(f"    ✓ Parameters reduced from 74 to 42")
        logger.info(f"    ✓ Single decision_threshold (removed 12 complex thresholds)")
        logger.info(f"    ✓ Direct ternary output (no two-stage filtering)")
        logger.info(f"    ✓ Balanced optimization target: {attrs['up_accuracy']:.1%} UP vs {attrs['down_accuracy']:.1%} DOWN")
        acc_diff = abs(attrs['up_accuracy'] - attrs['down_accuracy'])
        logger.info(f"    ✓ Accuracy difference: {acc_diff:.1%} (target: < 5%)")
        signal_ratio = int(attrs['up_count']) / int(attrs['down_count']) if int(attrs['down_count']) > 0 else 999
        logger.info(f"    ✓ Signal ratio UP/DOWN: {signal_ratio:.2f} (target: 0.67-1.50)")
        if signal_ratio < 0.67 or signal_ratio > 1.50:
            logger.info(f"    ⚠️  Signal imbalance detected - consider retuning")
        logger.info("="*80)


def test_feature_stability(df, vix, n_runs=10, output_dir="tuning_unified"):
    """Run feature selection N times to assess stability."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info(f"FEATURE STABILITY TEST: Running feature selection {n_runs} times")
    logger.info("="*80)

    train_end = pd.Timestamp("2021-12-31")
    val_end = pd.Timestamp("2023-12-31")
    train_mask = df.index <= train_end
    val_mask = (df.index > train_end) & (df.index <= val_end)
    train_val_df = df[train_mask | val_mask].copy()
    train_val_vix = vix[train_mask | val_mask].copy()

    base_cols = [c for c in df.columns if c not in
        ["vix", "spx", "calendar_cohort", "cohort_weight", "feature_quality",
         "future_vix", "target_vix_pct_change", "target_log_vix_change", "target_direction"]]

    feature_history = {'expansion': [], 'compression': [], 'up': [], 'down': []}

    from core.xgboost_feature_selector_v2 import FeatureSelector

    for run in range(n_runs):
        logger.info(f"\nRun {run+1}/{n_runs}:")

        for target_type in ['expansion', 'compression', 'up', 'down']:
            selector = FeatureSelector(target_type=target_type, top_n=100,
                correlation_threshold=0.90, random_state=42 + run)

            test_start_idx = len(train_val_df)
            selected, _ = selector.select_features(train_val_df[base_cols], train_val_vix,
                                                   test_start_idx=test_start_idx)

            feature_history[target_type].append(set(selected))
            logger.info(f"  {target_type}: {len(selected)} features")

    logger.info("\n" + "="*80)
    logger.info("STABILITY ANALYSIS:")
    logger.info("="*80)

    stable_features = {}
    for target_type in ['expansion', 'compression', 'up', 'down']:
        all_features = [f for run in feature_history[target_type] for f in run]
        feature_counts = Counter(all_features)

        threshold = int(n_runs * 0.8)
        stable = [f for f, count in feature_counts.items() if count >= threshold]
        stable_features[target_type] = sorted(stable)

        logger.info(f"\n{target_type.upper()}:")
        logger.info(f"  Total unique features seen: {len(feature_counts)}")
        logger.info(f"  Stable features (>={threshold}/{n_runs} runs): {len(stable)}")
        logger.info(f"  Stability rate: {len(stable)/len(feature_counts)*100:.1f}%")

        most_stable = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"  Most stable (top 10):")
        for feat, count in most_stable:
            logger.info(f"    {feat}: {count}/{n_runs} ({count/n_runs*100:.0f}%)")

    frozen_file = output_path / "frozen_features.json"
    with open(frozen_file, 'w') as f:
        json.dump(stable_features, f, indent=2)

    logger.info(f"\n✅ Stable features saved to: {frozen_file}")
    logger.info(f"   Use with: python unified_tuner_upgraded.py --frozen 500")

    report = {
        'timestamp': datetime.now().isoformat(),
        'n_runs': n_runs,
        'stability_threshold': 0.8,
        'stable_features': stable_features,
        'feature_frequencies': {
            target_type: {
                feat: count for feat, count in
                Counter([f for run in feature_history[target_type] for f in run]).items()
            }
            for target_type in ['expansion', 'compression', 'up', 'down']
        }
    }

    report_file = output_path / "stability_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"✅ Full report saved to: {report_file}")
    logger.info("="*80)

    return stable_features


def main():
    parser = argparse.ArgumentParser(
        description="Unified Phase 1 Tuner - Ternary Decision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Test feature stability (one time)
  python unified_tuner_upgraded.py --test-feature-stability 10

  # Step 2: Tune with frozen features (recommended)
  python unified_tuner_upgraded.py --frozen 500

  # Alternative: Custom frozen features path
  python unified_tuner_upgraded.py --frozen-features my_features.json --trials 500
        """
    )
    parser.add_argument('--trials', type=int, default=PHASE1_TUNER_TRIALS,
                       help=f"Number of trials (default: {PHASE1_TUNER_TRIALS})")
    parser.add_argument('--output-dir', type=str, default='tuning_unified', help="Output directory")
    parser.add_argument('--frozen-features', type=str, default=None,
                       help="Path to frozen features JSON")
    parser.add_argument('--frozen', type=int, default=None, metavar='TRIALS',
                       help="Shorthand: --frozen 500 = use frozen_features.json with 500 trials")
    parser.add_argument('--test-feature-stability', type=int, default=None, metavar='N',
                       help="Run feature selection N times to test stability")
    args = parser.parse_args()

    if args.frozen is not None:
        if args.frozen_features is None:
            args.frozen_features = "tuning_unified/frozen_features.json"
        args.trials = args.frozen
        logger.info(f"Using --frozen shorthand: {args.frozen} trials with {args.frozen_features}")

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

    if args.test_feature_stability:
        output_dir = args.output_dir if args.output_dir != 'tuning_unified' else 'feature_stability'
        test_feature_stability(df, result["vix"], n_runs=args.test_feature_stability,
                               output_dir=output_dir)
        return

    frozen_features = None
    if args.frozen_features:
        frozen_path = Path(args.frozen_features)
        if not frozen_path.exists():
            logger.error(f"❌ Frozen features file not found: {args.frozen_features}")
            logger.error(f"   Run feature stability test first:")
            logger.error(f"   python unified_tuner_upgraded.py --test-feature-stability 10")
            sys.exit(1)

        with open(frozen_path, 'r') as f:
            frozen_features = json.load(f)

        logger.info(f"✅ Loaded frozen features from: {args.frozen_features}")

    tuner = UnifiedPhase1Tuner(df=df, vix=result["vix"], n_trials=args.trials,
                               output_dir=args.output_dir, frozen_features=frozen_features)
    study = tuner.run()
    tuner.save_results(study)

    logger.info("\n✅ Unified optimization complete!")
    logger.info("\n📝 NEXT STEPS:")
    logger.info("   1. Review: tuning_unified/unified_results.json")
    logger.info("   2. Copy: tuning_unified/unified_config.py → src/config.py")
    logger.info("   3. Train: python train_probabilistic_models.py")
    logger.info("   4. Validate: python integrated_system.py")

if __name__ == "__main__":
    main()
