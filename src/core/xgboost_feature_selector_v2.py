"""XGBoost Feature Selector V2 - VIX % Change Forecasting (Regression)"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

CRITICAL_FORWARD_INDICATORS = [
    "VX1-VX2",
    "VX2-VX1_RATIO",
    "yield_10y2y",
    "yield_10y3m",
    "VXTLT",
    "vxtlt_vix_ratio",
    "SKEW",
    "skew_vs_vix",
]

DEFAULT_CANDIDATE_SIZES = [30, 40, 50, 60, 75, 100]
DEFAULT_MIN_STABILITY = 0.3
DEFAULT_MAX_CORRELATION = 0.95
DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 252


class IntelligentFeatureSelector:
    """Stability-based feature selection for VIX % change forecasting."""

    def __init__(self, output_dir: str = "./models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.baseline_trainer = None
        self.feature_importance_df = None
        self.stability_scores = None
        self.validation_results = []
        self.selected_features = None
        self.optimal_size = None

    def run_full_pipeline(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        horizons: List[int] = [5],
        candidate_sizes: List[int] = None,
        min_stability: float = DEFAULT_MIN_STABILITY,
        max_correlation: float = DEFAULT_MAX_CORRELATION,
        preserve_forward_indicators: bool = True,
        verbose: bool = True,
    ) -> Dict:
        if verbose:
            print(f"\n{'=' * 80}")
            print("INTELLIGENT FEATURE SELECTION - VIX % CHANGE FORECASTING")
            print(f"{'=' * 80}")
            print(f"Starting features: {len(features.columns)}")
            print(f"Horizons: {horizons}")
            print(f"Min stability: {min_stability}")
            print(f"Max correlation: {max_correlation}")
            print(f"Target: Forward VIX % Change (continuous regression)")

        if verbose:
            print(f"\n{'=' * 80}")
            print("[STEP 1/5] Training baseline + measuring feature stability...")
            print(f"{'=' * 80}")

        self.baseline_trainer, baseline_metrics, stability_scores = (
            self._train_baseline_with_stability(features, vix, spx, horizons, verbose)
        )
        self.stability_scores = stability_scores

        if verbose:
            print(f"\n{'=' * 80}")
            print("[STEP 2/5] Ranking features by stability-weighted importance...")
            print(f"{'=' * 80}")

        self.feature_importance_df = self._rank_features_with_stability(
            stability_scores, min_stability, verbose
        )

        if verbose:
            print(f"\n{'=' * 80}")
            print("[STEP 3/5] Removing multicollinear features...")
            print(f"{'=' * 80}")

        filtered_features = self._remove_multicollinearity(
            self.feature_importance_df, features, max_correlation, verbose
        )

        if verbose:
            print(f"\n{'=' * 80}")
            print(
                "[STEP 4/5] Recursive feature addition (finding performance cliff)..."
            )
            print(f"{'=' * 80}")

        if candidate_sizes is None:
            candidate_sizes = DEFAULT_CANDIDATE_SIZES

        optimal_features, optimal_size = self._recursive_feature_addition(
            filtered_features, features, vix, spx, horizons, candidate_sizes, verbose
        )

        self.optimal_size = optimal_size
        self.selected_features = optimal_features

        if preserve_forward_indicators:
            if verbose:
                print(f"\n{'=' * 80}")
                print("[STEP 5/5] Preserving critical forward indicators...")
                print(f"{'=' * 80}")

            self.selected_features = self._preserve_forward_indicators(
                self.selected_features, features.columns.tolist(), verbose
            )

        self._save_results(filtered_features, verbose)

        return {
            "selected_features": self.selected_features,
            "optimal_size": self.optimal_size,
            "feature_importance": self.feature_importance_df,
            "stability_scores": self.stability_scores,
            "validation_results": self.validation_results,
        }

    def _train_baseline_with_stability(
        self,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        horizons: List[int],
        verbose: bool,
    ) -> Tuple:
        features_clean = features.copy()
        if "vix_regime" in features_clean.columns:
            features_clean = features_clean.drop(columns=["vix_regime"])
            if verbose:
                print("   WARNING: Removed vix_regime from features (data leakage)")

        default_horizon = horizons[0]
        vix_future = vix.shift(-default_horizon)
        y_pct_change = ((vix_future / vix) - 1) * 100
        y_pct_change = y_pct_change.clip(-50, 200)

        common_idx = features_clean.index.intersection(vix.index).intersection(
            y_pct_change.dropna().index
        )
        X = features_clean.loc[common_idx]
        y_aligned = y_pct_change.loc[common_idx]

        if verbose:
            print(f"   Raw data: {len(features)} rows")
            print(f"   After alignment: {len(X)} samples")
            print(f"   Date range: {X.index.min().date()} → {X.index.max().date()}")
            print(
                f"   Target: Forward VIX % Change ({default_horizon}-day, continuous)"
            )
            print(f"      Mean: {y_aligned.mean():.2f}%")
            print(f"      Std: {y_aligned.std():.2f}%")
            print(f"      Min: {y_aligned.min():.2f}%")
            print(f"      Max: {y_aligned.max():.2f}%")
            print(f"      Median: {y_aligned.median():.2f}%")

        if len(X) < 1000:
            raise ValueError(f"Only {len(X)} samples after cleaning - check your data!")

        if verbose:
            print("   Computing per-fold feature importance for stability...")

        tscv = TimeSeriesSplit(n_splits=DEFAULT_N_SPLITS, test_size=DEFAULT_TEST_SIZE)

        all_features = X.columns.tolist()
        importance_matrix = pd.DataFrame(
            0.0, index=all_features, columns=[f"fold_{i}" for i in range(1, 6)]
        )

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train = X.iloc[train_idx]
            y_train = y_aligned.iloc[train_idx]

            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                max_depth=6,
                learning_rate=0.03,
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                enable_categorical=False,
            )
            model.fit(X_train, y_train, verbose=False)

            importance_dict = model.get_booster().get_score(importance_type="gain")

            importance_dict_mapped = {}
            for key, value in importance_dict.items():
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    if idx < len(all_features):
                        importance_dict_mapped[all_features[idx]] = value
                else:
                    importance_dict_mapped[key] = value

            if importance_dict_mapped:
                total_importance = sum(importance_dict_mapped.values())
                importance_dict_mapped = {
                    k: v / total_importance for k, v in importance_dict_mapped.items()
                }

            for feature, importance in importance_dict_mapped.items():
                if feature in importance_matrix.index:
                    importance_matrix.loc[feature, f"fold_{fold_idx}"] = importance

        mean_importance = importance_matrix.mean(axis=1)
        std_importance = importance_matrix.std(axis=1)

        stability = pd.Series(0.0, index=all_features, name="stability")
        non_zero_mask = mean_importance > 1e-8

        if non_zero_mask.any():
            cv = std_importance[non_zero_mask] / mean_importance[non_zero_mask]
            stability[non_zero_mask] = (1 - cv).clip(0, 1)

        if verbose:
            print(f"   OK: Stability computed for {len(stability)} features")
            print(
                f"      Features with non-zero importance: {(mean_importance > 0).sum()}"
            )
            print(
                f"      Mean stability (non-zero features): {stability[stability > 0].mean():.3f}"
            )
            print(f"      High stability features (>0.7): {(stability > 0.7).sum()}")
            print(
                f"      Medium stability features (0.3-0.7): {((stability >= 0.3) & (stability <= 0.7)).sum()}"
            )
            print(f"      Low stability features (<0.3): {(stability < 0.3).sum()}")

            used_features = mean_importance[mean_importance > 0].sort_values(
                ascending=False
            )
            print(f"\n   Features used by XGBoost (with mean importance):")
            for i, (feat, imp) in enumerate(used_features.head(15).items(), 1):
                stab = stability[feat]
                print(f"      {i:2d}. {feat:40s} | Imp: {imp:.4f} | Stab: {stab:.3f}")

        baseline_metrics = {
            "target_mean": y_aligned.mean(),
            "target_std": y_aligned.std(),
        }

        return None, baseline_metrics, stability

    def _rank_features_with_stability(
        self, stability_scores: pd.Series, min_stability: float, verbose: bool
    ) -> pd.DataFrame:
        importance_df = pd.DataFrame(
            {
                "feature": stability_scores.index,
                "importance": stability_scores.values,
                "stability": stability_scores.values,
            }
        )

        importance_df["weighted_importance"] = (
            importance_df["importance"] * importance_df["stability"]
        )
        importance_df = importance_df.sort_values(
            "weighted_importance", ascending=False
        )

        high_stability = importance_df[importance_df["stability"] >= min_stability]
        top_by_importance = importance_df.nlargest(100, "weighted_importance")

        combined_filtered = pd.concat(
            [high_stability, top_by_importance]
        ).drop_duplicates()
        combined_filtered = combined_filtered.sort_values(
            "weighted_importance", ascending=False
        )

        if verbose:
            print(f"   Total features with importance: {len(importance_df)}")
            print(
                f"   High stability (>= {min_stability}): {len(high_stability)} features"
            )
            print(f"   Top by weighted importance: {len(top_by_importance)} features")
            print(f"   Combined (deduplicated): {len(combined_filtered)} features")

            print(f"\n   Top 20 features by stability-weighted importance:")
            for i, row in enumerate(combined_filtered.head(20).itertuples(), 1):
                print(
                    f"      {i:2d}. {row.feature:30s} | Imp: {row.importance:.4f} | Stab: {row.stability:.3f} | Weight: {row.weighted_importance:.4f}"
                )

        return combined_filtered

    def _remove_multicollinearity(
        self,
        feature_importance_df: pd.DataFrame,
        features: pd.DataFrame,
        max_correlation: float,
        verbose: bool,
    ) -> pd.DataFrame:
        ranked_features = feature_importance_df["feature"].tolist()
        features_subset = features[ranked_features]
        corr_matrix = features_subset.corr().abs()

        to_remove = set()
        removed_pairs = []

        for i, feat1 in enumerate(ranked_features):
            if feat1 in to_remove:
                continue
            for feat2 in ranked_features[i + 1 :]:
                if feat2 in to_remove:
                    continue
                if corr_matrix.loc[feat1, feat2] > max_correlation:
                    to_remove.add(feat2)
                    removed_pairs.append((feat1, feat2, corr_matrix.loc[feat1, feat2]))

        filtered_df = feature_importance_df[
            ~feature_importance_df["feature"].isin(to_remove)
        ]

        if verbose:
            if removed_pairs:
                print(f"   Removing {len(to_remove)} multicollinear features:")
                for feat1, feat2, corr in removed_pairs[:10]:
                    print(
                        f"      {feat1:30s} ↔ {feat2:30s} (r={corr:.3f}) → Remove {feat2}"
                    )
                if len(removed_pairs) > 10:
                    print(f"      ... and {len(removed_pairs) - 10} more pairs")
            else:
                print(
                    f"   No multicollinear features found (threshold={max_correlation})"
                )

            print(f"   After multicollinearity removal: {len(filtered_df)} features")

        return filtered_df

    def _recursive_feature_addition(
        self,
        feature_importance_df: pd.DataFrame,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        horizons: List[int],
        candidate_sizes: List[int],
        verbose: bool,
    ) -> Tuple[List[str], int]:
        ranked_features = feature_importance_df["feature"].tolist()
        default_horizon = horizons[0]
        results = []

        for size in candidate_sizes:
            if size > len(ranked_features):
                if verbose:
                    print(
                        f"   Skipping size={size} (exceeds {len(ranked_features)} available features)"
                    )
                continue

            subset_features = ranked_features[:size]
            X_subset = features[subset_features]

            if "vix_regime" in X_subset.columns:
                X_subset = X_subset.drop(columns=["vix_regime"])
                if verbose and size == candidate_sizes[0]:
                    print("   WARNING: Removed vix_regime from features (data leakage)")

            vix_future = vix.shift(-default_horizon)
            y = ((vix_future / vix) - 1) * 100
            y = y.clip(-50, 200)

            common_idx = X_subset.index.intersection(vix.index).intersection(
                y.dropna().index
            )
            X_aligned = X_subset.loc[common_idx]
            y_aligned = y.loc[common_idx]

            tscv = TimeSeriesSplit(
                n_splits=DEFAULT_N_SPLITS, test_size=DEFAULT_TEST_SIZE
            )
            cv_scores = []

            for train_idx, test_idx in tscv.split(X_aligned):
                X_train, X_test = X_aligned.iloc[train_idx], X_aligned.iloc[test_idx]
                y_train, y_test = y_aligned.iloc[train_idx], y_aligned.iloc[test_idx]

                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    max_depth=6,
                    learning_rate=0.05,
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
                model.fit(X_train, y_train, verbose=False)

                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_scores.append(rmse)

            mean_rmse = np.mean(cv_scores)
            std_rmse = np.std(cv_scores)

            results.append({"size": size, "rmse": mean_rmse, "std": std_rmse})

            if verbose:
                print(f"   Size {size:3d}: RMSE = {mean_rmse:.2f}% ± {std_rmse:.2f}%")

        best_result = min(results, key=lambda x: x["rmse"])
        optimal_size = best_result["size"]
        optimal_features = ranked_features[:optimal_size]

        self.validation_results = results

        if verbose:
            print(f"\n   OK: Optimal size: {optimal_size} features")
            print(f"      RMSE: {best_result['rmse']:.2f}% ± {best_result['std']:.2f}%")

        return optimal_features, optimal_size

    def _preserve_forward_indicators(
        self, selected_features: List[str], all_features: List[str], verbose: bool
    ) -> List[str]:
        missing_indicators = [
            f
            for f in CRITICAL_FORWARD_INDICATORS
            if f in all_features and f not in selected_features
        ]

        if missing_indicators:
            selected_features = selected_features + missing_indicators
            if verbose:
                print(
                    f"   Adding {len(missing_indicators)} critical forward indicators:"
                )
                for indicator in missing_indicators:
                    print(f"      • {indicator}")

        return selected_features

    def _save_results(self, filtered_features: pd.DataFrame, verbose: bool):
        with open(self.output_dir / "selected_features_v2.txt", "w") as f:
            for feat in self.selected_features:
                f.write(f"{feat}\n")

        self.feature_importance_df.to_csv(
            self.output_dir / "feature_importance_with_stability.csv", index=False
        )
        self.stability_scores.to_csv(self.output_dir / "feature_stability_scores.csv")

        if verbose:
            print(f"\n   OK: Saved: {self.output_dir / 'selected_features_v2.txt'}")
            print(
                f"   OK: Saved: {self.output_dir / 'feature_importance_with_stability.csv'}"
            )
            print(f"   OK: Saved: {self.output_dir / 'feature_stability_scores.csv'}")

        print(f"\n{'=' * 80}")
        print("FEATURE SELECTION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Final feature count: {len(self.selected_features)}")
        initial_count = len(filtered_features) + len(
            [
                f
                for f in CRITICAL_FORWARD_INDICATORS
                if f not in filtered_features["feature"].tolist()
            ]
        )
        reduction_pct = (1 - len(self.selected_features) / initial_count) * 100
        print(
            f"Reduction: {initial_count} → {len(self.selected_features)} ({reduction_pct:.1f}% reduction)"
        )


def run_intelligent_feature_selection(
    integrated_system,
    horizons: List[int] = [5],
    min_stability: float = DEFAULT_MIN_STABILITY,
    max_correlation: float = DEFAULT_MAX_CORRELATION,
    preserve_forward_indicators: bool = True,
    verbose: bool = True,
) -> Dict:
    if not integrated_system.trained:
        raise ValueError("Train integrated system first")

    selector = IntelligentFeatureSelector()

    return selector.run_full_pipeline(
        features=integrated_system.orchestrator.features,
        vix=integrated_system.orchestrator.vix_ml,
        spx=integrated_system.orchestrator.spx_ml,
        horizons=horizons,
        min_stability=min_stability,
        max_correlation=max_correlation,
        preserve_forward_indicators=preserve_forward_indicators,
        verbose=verbose,
    )
