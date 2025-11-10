"""
XGBoost Feature Selector V2 - FIXED (Proper Stability Calculation)
===================================================================
Bug Fix: Initialize importance matrix with ALL features, not just those used in splits.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

try:
    from core.xgboost_trainer_v2 import EnhancedXGBoostTrainer
except ImportError:
    from xgboost_trainer_v2 import EnhancedXGBoostTrainer

# Forward indicators to preserve
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


class IntelligentFeatureSelector:
    """Stability-based feature selection with proper initialization."""

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
        horizons: List[int] = [5],  # NEW: support multi-horizon
        candidate_sizes: List[int] = None,
        min_stability: float = 0.3,
        max_correlation: float = 0.95,
        preserve_forward_indicators: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """Run intelligent feature selection pipeline - UPDATED FOR FORWARD VOL."""
        if verbose:
            print(f"\n{'=' * 80}")
            print(
                "🎯 INTELLIGENT FEATURE SELECTION PIPELINE V2 - FORWARD VOL EXPANSION"
            )
            print(f"{'=' * 80}")
            print(f"Starting features: {len(features.columns)}")
            print(f"Horizons: {horizons}")
            print(f"Min stability: {min_stability}")
            print(f"Max correlation: {max_correlation}")
            print(f"Target: Forward VIX Expansion (threshold=15%)")

        # Step 1: Train baseline with ALL features + measure stability
        if verbose:
            print(f"\n{'=' * 80}")
            print("[STEP 1/5] Training baseline + measuring feature stability...")
            print(f"{'=' * 80}")

        self.baseline_trainer, baseline_metrics, stability_scores = (
            self._train_baseline_with_stability(features, vix, spx, horizons, verbose)
        )

        self.stability_scores = stability_scores

        # Step 2: Rank features by stability-weighted importance
        if verbose:
            print(f"\n{'=' * 80}")
            print("[STEP 2/5] Ranking features by stability-weighted importance...")
            print(f"{'=' * 80}")

        self.feature_importance_df = self._rank_features_with_stability(
            stability_scores, min_stability, verbose
        )

        # Step 3: Remove multicollinear features
        if verbose:
            print(f"\n{'=' * 80}")
            print("[STEP 3/5] Removing multicollinear features...")
            print(f"{'=' * 80}")

        filtered_importance_df = self._remove_multicollinearity(
            self.feature_importance_df, features, max_correlation, verbose
        )

        # Step 4: Recursive feature addition
        if verbose:
            print(f"\n{'=' * 80}")
            print(
                "[STEP 4/5] Recursive feature addition (finding performance cliff)..."
            )
            print(f"{'=' * 80}")

        if candidate_sizes is None:
            candidate_sizes = [30, 40, 50, 60, 75, 100]

        optimal_features, optimal_size = self._recursive_feature_addition(
            filtered_importance_df, features, vix, spx, candidate_sizes, verbose
        )

        # Step 5: Preserve critical forward indicators
        if preserve_forward_indicators:
            if verbose:
                print(f"\n{'=' * 80}")
                print("[STEP 5/5] Preserving critical forward indicators...")
                print(f"{'=' * 80}")

            optimal_features = self._preserve_forward_indicators(
                optimal_features, filtered_importance_df, verbose
            )

        self.selected_features = optimal_features
        self.optimal_size = len(optimal_features)

        # Save results
        self._save_results(optimal_features, baseline_metrics, verbose)

        if verbose:
            print(f"\n{'=' * 80}")
            print("✅ FEATURE SELECTION COMPLETE")
            print(f"{'=' * 80}")
            print(f"Final feature count: {len(optimal_features)}")
            print(
                f"Reduction: {len(features.columns)} → {len(optimal_features)} ({100 * (1 - len(optimal_features) / len(features.columns)):.1f}% reduction)"
            )

        return {
            "selected_features": optimal_features,
            "baseline_metrics": baseline_metrics,
            "stability_scores": stability_scores,
            "importance_df": self.feature_importance_df,
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
        """Train baseline model and compute feature stability - UPDATED FOR FORWARD VOL."""

        # Remove vix_regime from features if present (data leakage!)
        if "vix_regime" in features.columns:
            features = features.drop(columns=["vix_regime"])
            if verbose:
                print("   ⚠️  Removed vix_regime from features (data leakage)")

        # === Properly align data ===
        if verbose:
            print(f"   Raw data: {len(features)} rows")

        # Clean features
        features_clean = features.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.fillna(features_clean.median())

        # === NEW: Forward VIX expansion target ===
        expansion_threshold = 0.15  # 15% VIX increase
        horizon = 5  # 5-day forward

        forward_max_change = vix.rolling(horizon).max().shift(-horizon) / vix - 1
        y_regime = (forward_max_change > expansion_threshold).astype(int)

        # Range target stays the same
        spx_ret_5d = spx.pct_change(5).shift(-5)
        y_range = spx_ret_5d.abs()

        # Get common index
        common_idx = (
            features_clean.notna().any(axis=1) & y_regime.notna() & y_range.notna()
        )
        common_idx = features_clean.index[common_idx]

        # Align everything
        X = features_clean.loc[common_idx]
        y_regime_aligned = y_regime.loc[common_idx]
        y_range_aligned = y_range.loc[common_idx]

        if verbose:
            print(f"   After alignment: {len(X)} samples")
            print(f"   Date range: {X.index.min().date()} → {X.index.max().date()}")
            print(
                f"   Target: Forward VIX Expansion (5-day, threshold={expansion_threshold:.1%})"
            )
            print(
                f"      Class 0 (No Expansion): {(y_regime_aligned == 0).sum():>4} ({(y_regime_aligned == 0).sum() / len(y_regime_aligned) * 100:>5.1f}%)"
            )
            print(
                f"      Class 1 (Expansion): {(y_regime_aligned == 1).sum():>4} ({(y_regime_aligned == 1).sum() / len(y_regime_aligned) * 100:>5.1f}%)"
            )

        if len(X) < 1000:
            raise ValueError(f"Only {len(X)} samples after cleaning - check your data!")

        # === Train baseline trainer (fast, no SHAP) ===
        trainer = EnhancedXGBoostTrainer(output_dir=str(self.output_dir / "baseline"))

        results = trainer.train(
            features=X,
            vix=vix.loc[X.index],
            spx=spx.loc[X.index],
            horizons=horizons,
            n_splits=5,
            optimize_hyperparams=False,
            crisis_balanced=True,
            compute_shap=False,
            verbose=False,
        )

        # === FIX: Handle trainer results - extract metrics safely ===
        if verbose:
            print("   Processing trainer results...")
            print(f"   Results keys: {list(results.keys())}")

        # Extract baseline metrics from new multi-horizon structure
        baseline_metrics = {}

        # Results now has structure: {horizon: {regime_metrics, range_metrics}}
        # Get metrics from the first horizon (or average across all)
        if "all_results" in results:
            # New structure: results['all_results'][horizon]['regime_metrics']
            first_horizon = list(results["all_results"].keys())[0]

            if "regime_metrics" in results["all_results"][first_horizon]:
                baseline_metrics["regime_metrics"] = results["all_results"][
                    first_horizon
                ]["regime_metrics"]
            else:
                baseline_metrics["regime_metrics"] = {"balanced_accuracy": 0.5}

            if "range_metrics" in results["all_results"][first_horizon]:
                baseline_metrics["range_metrics"] = results["all_results"][
                    first_horizon
                ]["range_metrics"]
            else:
                baseline_metrics["range_metrics"] = {"r2": 0.0}
        else:
            # Fallback: create dummy metrics
            baseline_metrics["regime_metrics"] = {"balanced_accuracy": 0.5}
            baseline_metrics["range_metrics"] = {"r2": 0.0}
            if verbose:
                print("   ⚠️  Warning: Could not find metrics in expected structure")

        # === Compute per-fold importance with proper initialization ===
        if verbose:
            print("   Computing per-fold feature importance for stability...")

        tscv = TimeSeriesSplit(n_splits=5, test_size=252)

        # CRITICAL FIX: Initialize importance matrix with ALL features
        all_features = X.columns.tolist()
        importance_matrix = pd.DataFrame(
            0.0, index=all_features, columns=[f"fold_{i}" for i in range(1, 6)]
        )

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train = X.iloc[train_idx]
            y_train = y_regime_aligned.iloc[train_idx]

            # Train simple BINARY model (changed from multiclass)
            model = xgb.XGBClassifier(
                objective="binary:logistic",  # Changed from 'multi:softprob'
                max_depth=6,
                learning_rate=0.03,
                n_estimators=300,
                scale_pos_weight=2,  # Handle class imbalance
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_train, y_train, verbose=False)

            # Get gain-based importance (only for features used in splits)
            importance_dict = model.get_booster().get_score(importance_type="gain")

            # Normalize to sum to 1
            if importance_dict:
                total_importance = sum(importance_dict.values())
                importance_dict = {
                    k: v / total_importance for k, v in importance_dict.items()
                }

            # CRITICAL FIX: Update matrix for features that were used
            # Features not in importance_dict remain at 0.0
            for feature, importance in importance_dict.items():
                if feature in importance_matrix.index:
                    importance_matrix.loc[feature, f"fold_{fold_idx}"] = importance

        # Calculate stability from properly initialized matrix
        mean_importance = importance_matrix.mean(axis=1)
        std_importance = importance_matrix.std(axis=1)

        # Stability = 1 - coefficient of variation
        # For features with zero importance (mean=0), stability should be 0
        stability = pd.Series(0.0, index=all_features, name="stability")
        non_zero_mask = mean_importance > 1e-8

        if non_zero_mask.any():
            cv = std_importance[non_zero_mask] / mean_importance[non_zero_mask]
            stability[non_zero_mask] = (1 - cv).clip(0, 1)

        if verbose:
            print(f"   ✅ Stability computed for {len(stability)} features")
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

            # NEW: Show which features are actually being used
            used_features = mean_importance[mean_importance > 0].sort_values(
                ascending=False
            )
            print(f"\n   📊 Features used by XGBoost (with mean importance):")
            for i, (feat, imp) in enumerate(used_features.head(15).items(), 1):
                stab = stability[feat]
                print(f"      {i:2d}. {feat:40s} | Imp: {imp:.4f} | Stab: {stab:.3f}")

            if len(used_features) < 20:
                print(
                    f"\n   ⚠️  WARNING: Only {len(used_features)} features used by XGBoost!"
                )
                print(
                    f"   This suggests high redundancy or conservative regularization."
                )
                print(f"   Consider:")
                print(f"      • Removing highly correlated features first")
                print(
                    f"      • Reducing regularization (lower min_child_weight, gamma)"
                )
                print(f"      • Using gain + weight importance (not just gain)")

        return trainer, baseline_metrics, stability

    def _rank_features_with_stability(
        self, stability_scores: pd.Series, min_stability: float, verbose: bool
    ) -> pd.DataFrame:
        """Rank features by stability-weighted importance."""

        # Get importance from the baseline trainer's feature_importance dict
        if (
            not hasattr(self.baseline_trainer, "feature_importance")
            or not self.baseline_trainer.feature_importance
        ):
            raise ValueError(
                "Baseline trainer does not have feature_importance computed"
            )

        # Extract from the stored dictionaries
        regime_importance_list = self.baseline_trainer.feature_importance.get(
            "regime", []
        )
        range_importance_list = self.baseline_trainer.feature_importance.get(
            "range", []
        )

        if not regime_importance_list or not range_importance_list:
            raise ValueError("Feature importance lists are empty")

        # Convert to DataFrames
        regime_importance = pd.DataFrame(regime_importance_list).set_index("feature")
        range_importance = pd.DataFrame(range_importance_list).set_index("feature")

        # The column name is 'regime_shap' or 'range_shap' (even for gain-based importance)
        regime_col = (
            "regime_shap"
            if "regime_shap" in regime_importance.columns
            else regime_importance.columns[0]
        )
        range_col = (
            "range_shap"
            if "range_shap" in range_importance.columns
            else range_importance.columns[0]
        )

        # Combine (50/50 weight)
        combined = pd.DataFrame(
            {
                "importance_regime": regime_importance[regime_col],
                "importance_range": range_importance[range_col],
            }
        )
        combined["importance_avg"] = combined.mean(axis=1)

        # Add stability (now includes all features)
        combined = combined.join(stability_scores, how="left")
        combined["stability"] = combined["stability"].fillna(0)

        # Filter by minimum stability
        stable_features = combined[combined["stability"] >= min_stability]

        if verbose:
            print(f"   Total features with importance: {len(combined)}")
            print(
                f"   Filtered by stability >= {min_stability}: {len(combined)} → {len(stable_features)} features"
            )

            if len(stable_features) < 20:
                print(
                    f"\n   ⚠️  WARNING: Only {len(stable_features)} features passed stability filter!"
                )
                print(f"   Consider lowering min_stability (currently {min_stability})")
                print(f"\n   Distribution of stability scores:")
                print(f"      >0.7:  {(combined['stability'] > 0.7).sum()} features")
                print(
                    f"      0.5-0.7: {((combined['stability'] >= 0.5) & (combined['stability'] <= 0.7)).sum()} features"
                )
                print(
                    f"      0.3-0.5: {((combined['stability'] >= 0.3) & (combined['stability'] < 0.5)).sum()} features"
                )
                print(
                    f"      0.2-0.3: {((combined['stability'] >= 0.2) & (combined['stability'] < 0.3)).sum()} features"
                )
                print(
                    f"      0.1-0.2: {((combined['stability'] >= 0.1) & (combined['stability'] < 0.2)).sum()} features"
                )
                print(f"      <0.1:  {(combined['stability'] < 0.1).sum()} features")

                # If too few features, suggest alternatives
                if len(stable_features) < 10:
                    print(f"\n   💡 RECOMMENDATION: Try one of these:")
                    print(f"      1. Lower min_stability to 0.1 or 0.15")
                    print(
                        f"      2. Use top N features by importance (ignore stability)"
                    )
                    print(
                        f"      3. Pre-filter correlated features before stability check"
                    )

        # If we have very few stable features, be more lenient
        if len(stable_features) < 10 and min_stability > 0.0:
            if verbose:
                print(
                    f"\n   🔄 Auto-adjusting: Using importance-only ranking (stability too restrictive)"
                )
            # Fall back to importance-only ranking
            stable_features = combined
            stable_features["weighted_importance"] = stable_features["importance_avg"]
        else:
            # Compute stability-weighted importance
            stable_features["weighted_importance"] = (
                stable_features["importance_avg"] * stable_features["stability"]
            )

        # Sort
        result = stable_features.sort_values("weighted_importance", ascending=False)

        if verbose:
            print(f"\n   Top 20 features by stability-weighted importance:")
            for i, (feat, row) in enumerate(result.head(20).iterrows(), 1):
                print(
                    f"      {i:2d}. {feat:35s} | Imp: {row['importance_avg']:.4f} | Stab: {row['stability']:.3f} | Weight: {row['weighted_importance']:.4f}"
                )

        return result

    def _remove_multicollinearity(
        self,
        importance_df: pd.DataFrame,
        features: pd.DataFrame,
        max_correlation: float,
        verbose: bool,
    ) -> pd.DataFrame:
        """Remove highly correlated features, keeping most important."""
        features_subset = features[importance_df.index]
        corr_matrix = features_subset.corr().abs()

        # Find pairs above threshold
        upper_triangle = np.triu(corr_matrix, k=1)
        high_corr_pairs = []

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if upper_triangle[i, j] > max_correlation:
                    feat_i = corr_matrix.index[i]
                    feat_j = corr_matrix.index[j]

                    # Keep the one with higher weighted importance
                    imp_i = importance_df.loc[feat_i, "weighted_importance"]
                    imp_j = importance_df.loc[feat_j, "weighted_importance"]

                    to_remove = feat_j if imp_i > imp_j else feat_i
                    high_corr_pairs.append(
                        (feat_i, feat_j, upper_triangle[i, j], to_remove)
                    )

        # Remove duplicates
        features_to_remove = set([pair[3] for pair in high_corr_pairs])

        if verbose and features_to_remove:
            print(f"   Removing {len(features_to_remove)} multicollinear features:")
            for feat_i, feat_j, corr, removed in high_corr_pairs[:10]:
                print(
                    f"      {feat_i:25s} ↔ {feat_j:25s} (r={corr:.3f}) → Remove {removed}"
                )
            if len(high_corr_pairs) > 10:
                print(f"      ... and {len(high_corr_pairs) - 10} more")

        result = importance_df[~importance_df.index.isin(features_to_remove)]

        if verbose:
            print(f"   After multicollinearity removal: {len(result)} features")

        return result

    def _recursive_feature_addition(
        self,
        importance_df: pd.DataFrame,
        features: pd.DataFrame,
        vix: pd.Series,
        spx: pd.Series,
        candidate_sizes: List[int],
        verbose: bool,
    ) -> Tuple[List[str], int]:
        """Try different feature set sizes and find optimal - UPDATED FOR FORWARD VOL."""

        # Remove vix_regime from features if present (data leakage!)
        if "vix_regime" in features.columns:
            features = features.drop(columns=["vix_regime"])
            if verbose:
                print("   ⚠️  Removed vix_regime from features (data leakage)")

        # Prepare data once
        features_clean = features.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.fillna(features_clean.median())

        # === NEW: Forward VIX expansion target ===
        expansion_threshold = 0.15
        horizon = 5

        forward_max_change = vix.rolling(horizon).max().shift(-horizon) / vix - 1
        y_regime = (forward_max_change > expansion_threshold).astype(int)

        # Range target
        spx_ret_5d = spx.pct_change(5).shift(-5)
        y_range = spx_ret_5d.abs()

        common_idx = (
            features_clean.notna().any(axis=1) & y_regime.notna() & y_range.notna()
        )
        common_idx = features_clean.index[common_idx]

        X_full = features_clean.loc[common_idx]
        y_regime_aligned = y_regime.loc[common_idx]
        y_range_aligned = y_range.loc[common_idx]

        # Try each size
        results = []
        available_features = len(importance_df)

        for size in candidate_sizes:
            if size > available_features:
                if verbose:
                    print(
                        f"   Skipping size={size} (exceeds {available_features} available features)"
                    )
                continue

            top_features = importance_df.head(size).index.tolist()
            X_subset = X_full[top_features]

            # Quick cross-validation
            tscv = TimeSeriesSplit(n_splits=3, test_size=252)
            fold_scores = []

            for train_idx, val_idx in tscv.split(X_subset):
                X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
                y_train, y_val = (
                    y_regime_aligned.iloc[train_idx],
                    y_regime_aligned.iloc[val_idx],
                )

                # BINARY classifier (changed from multiclass)
                model = xgb.XGBClassifier(
                    objective="binary:logistic",  # Changed from 'multi:softprob'
                    max_depth=6,
                    learning_rate=0.03,
                    n_estimators=200,
                    scale_pos_weight=2,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
                model.fit(X_train, y_train, verbose=False)

                from sklearn.metrics import balanced_accuracy_score

                y_pred = model.predict(X_val)
                score = balanced_accuracy_score(y_val, y_pred)
                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            results.append(
                {
                    "size": size,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "features": top_features,
                }
            )

            if verbose:
                print(
                    f"   Size {size:3d}: Accuracy = {mean_score:.4f} ± {std_score:.4f}"
                )

        # Handle edge case: if no sizes were tested
        if not results:
            if verbose:
                print(
                    f"\n   ⚠️  No valid sizes tested - using all {available_features} features"
                )
            return importance_df.index.tolist(), available_features

        # Find optimal size
        results_df = pd.DataFrame(results)
        results_df["robust_score"] = results_df["mean_score"] - results_df["std_score"]

        optimal_idx = results_df["robust_score"].idxmax()
        optimal_result = results_df.loc[optimal_idx]

        if verbose:
            print(f"\n   ✅ Optimal size: {optimal_result['size']} features")
            print(
                f"      Accuracy: {optimal_result['mean_score']:.4f} ± {optimal_result['std_score']:.4f}"
            )

        self.validation_results = results

        return optimal_result["features"], optimal_result["size"]

    def _preserve_forward_indicators(
        self, selected_features: List[str], importance_df: pd.DataFrame, verbose: bool
    ) -> List[str]:
        """Add critical forward indicators if not already selected."""
        available_indicators = [
            f for f in CRITICAL_FORWARD_INDICATORS if f in importance_df.index
        ]
        missing_indicators = [
            f for f in available_indicators if f not in selected_features
        ]

        if missing_indicators:
            if verbose:
                print(
                    f"   Adding {len(missing_indicators)} critical forward indicators:"
                )
                for feat in missing_indicators:
                    print(f"      • {feat}")

            selected_features = selected_features + missing_indicators
        else:
            if verbose:
                print(f"   All critical forward indicators already selected ✅")

        return selected_features

    def _save_results(
        self, selected_features: List[str], baseline_metrics: Dict, verbose: bool
    ):
        """Save selection results."""
        # Save selected features list
        features_path = self.output_dir / "selected_features_v2.txt"
        with open(features_path, "w") as f:
            f.write("\n".join(selected_features))

        if verbose:
            print(f"\n   ✅ Saved: {features_path}")

        # Save importance rankings
        importance_path = self.output_dir / "feature_importance_with_stability.csv"
        self.feature_importance_df.to_csv(importance_path)

        if verbose:
            print(f"   ✅ Saved: {importance_path}")

        # Save stability scores
        stability_path = self.output_dir / "feature_stability_scores.csv"
        self.stability_scores.to_frame("stability").to_csv(stability_path)

        if verbose:
            print(f"   ✅ Saved: {stability_path}")


def run_intelligent_feature_selection(
    integrated_system,
    horizons: List[int] = [5],  # NEW: support multi-horizon
    min_stability: float = 0.3,
    max_correlation: float = 0.95,
    preserve_forward_indicators: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run intelligent feature selection on trained system."""
    if not integrated_system.trained:
        raise ValueError("Train integrated system first: system.train(years=15)")

    selector = IntelligentFeatureSelector()

    results = selector.run_full_pipeline(
        features=integrated_system.orchestrator.features,
        vix=integrated_system.orchestrator.vix_ml,
        spx=integrated_system.orchestrator.spx_ml,
        horizons=horizons,  # NEW: pass through
        candidate_sizes=None,
        min_stability=min_stability,
        max_correlation=max_correlation,
        preserve_forward_indicators=preserve_forward_indicators,
        verbose=verbose,
    )

    return results
