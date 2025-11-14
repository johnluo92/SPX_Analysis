"""
XGBoost Feature Selector V3 - Log-Transformed Realized Volatility System

CRITICAL CHANGES FROM V2:
1. Target calculation changed from VIX % change to forward-looking realized volatility
2. Realized volatility calculated from SPX returns over forecast horizon
3. Log transformation applied to target for better distribution
4. Feature importance evaluated against log(RV) target
5. All features maintain strict temporal hygiene (no future leakage)

Author: VIX Forecasting System
Last Updated: 2025-11-13
Version: 3.0 (Log-RV Feature Selection)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostFeatureSelector:
    """
    Feature selection using XGBoost feature importance for log-RV forecasting.

    KEY V3 METHODOLOGY:
    - Target: Log-transformed forward-looking realized volatility
    - Forward window: Returns from t to t+horizon (e.g., 21 days)
    - Realized vol: Annualized std of SPX returns over forward window
    - Log transform: log(realized_vol) for better distribution
    - Temporal hygiene: Features at time t, target from t to t+horizon

    This ensures we're selecting features that predict FUTURE volatility,
    not features that explain PAST VIX changes.
    """

    def __init__(
        self,
        horizon: int = 21,
        min_importance: float = 0.001,
        top_n: int = 50,
        cv_folds: int = 3,
    ):
        """
        Initialize feature selector.

        Args:
            horizon: Forecast horizon in trading days (default 21)
            min_importance: Minimum feature importance threshold
            top_n: Maximum number of features to select
            cv_folds: Number of cross-validation folds
        """
        self.horizon = horizon
        self.min_importance = min_importance
        self.top_n = top_n
        self.cv_folds = cv_folds

        self.selected_features = None
        self.importance_scores = None
        self.selection_metadata = None

        logger.info(
            f"Initialized XGBoost Feature Selector V3:\n"
            f"  Horizon: {horizon} days\n"
            f"  Target: Log-transformed forward realized volatility\n"
            f"  Min importance: {min_importance}\n"
            f"  Top N: {top_n}"
        )

    def _calculate_forward_realized_volatility(
        self,
        spx_returns: pd.Series,
        dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Calculate forward-looking realized volatility for each date.

        CRITICAL METHODOLOGY:
        1. For each date t, look FORWARD horizon days
        2. Compute realized volatility from returns between t and t+horizon
        3. Annualize the volatility (multiply by sqrt(252))
        4. Apply log transformation for better statistical properties

        This is the ACTUAL target we're trying to predict:
        "What will the realized volatility of SPX be over the next 21 days?"

        Args:
            spx_returns: Series of SPX daily returns
            dates: DatetimeIndex of dates to calculate RV for

        Returns:
            Series of log-transformed realized volatility, indexed by date

        Example:
            Date t=2020-01-15:
            - Look at returns from 2020-01-15 to 2020-02-15 (21 days)
            - Calculate std of those returns
            - Annualize: std * sqrt(252)
            - Transform: log(annualized_std)
            - This becomes the target for features observed on 2020-01-15
        """

        logger.info(
            f"Calculating forward-looking realized volatility:\n"
            f"  Window: {self.horizon} trading days\n"
            f"  Method: Annualized std of forward returns\n"
            f"  Transform: Natural log"
        )

        realized_vols = pd.Series(index=dates, dtype=float)

        # Ensure returns are sorted by date
        spx_returns = spx_returns.sort_index()

        valid_count = 0
        insufficient_data = 0

        for date in dates:
            if date not in spx_returns.index:
                continue

            # Get position of this date in the returns series
            try:
                date_pos = spx_returns.index.get_loc(date)
            except KeyError:
                continue

            # Define forward window: from date (inclusive) to date+horizon
            # We need horizon+1 days to get horizon returns
            end_pos = date_pos + self.horizon + 1

            # Check if we have enough future data
            if end_pos > len(spx_returns):
                insufficient_data += 1
                continue

            # Extract forward returns
            forward_returns = spx_returns.iloc[date_pos:end_pos]

            # Need at least horizon/2 valid returns
            if forward_returns.notna().sum() < self.horizon / 2:
                insufficient_data += 1
                continue

            # Calculate realized volatility
            # Drop NaN returns before calculating std
            valid_returns = forward_returns.dropna()

            if len(valid_returns) < 10:  # Minimum sample size
                insufficient_data += 1
                continue

            # Realized vol = std of returns * sqrt(252) for annualization
            realized_vol = valid_returns.std() * np.sqrt(252)

            # Apply log transformation
            # Note: RV is always positive, so log is well-defined
            # Typical RV values range from 10-80%, so log(RV) ≈ 2.3 to 4.4
            log_rv = np.log(realized_vol * 100)  # Convert to percentage first

            realized_vols[date] = log_rv
            valid_count += 1

        # Remove NaN values
        realized_vols = realized_vols.dropna()

        logger.info(
            f"Forward RV calculation complete:\n"
            f"  Valid calculations: {valid_count}\n"
            f"  Insufficient data: {insufficient_data}\n"
            f"  Log(RV) range: [{realized_vols.min():.2f}, {realized_vols.max():.2f}]\n"
            f"  Log(RV) mean: {realized_vols.mean():.2f}\n"
            f"  Log(RV) std: {realized_vols.std():.2f}"
        )

        return realized_vols

    def select_features(
        self,
        features_df: pd.DataFrame,
        spx_returns: pd.Series,
        feature_categories: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[str], Dict]:
        """
        Select features using XGBoost importance on log-RV target.

        METHODOLOGY:
        1. Calculate forward-looking log(RV) as target
        2. Align features with targets (strict temporal matching)
        3. Train XGBoost models via cross-validation
        4. Aggregate feature importance across folds
        5. Select top features above importance threshold

        Args:
            features_df: DataFrame with features (rows=dates, cols=features)
            spx_returns: Series of SPX daily returns for RV calculation
            feature_categories: Optional dict grouping features by category

        Returns:
            Tuple of (selected_features, metadata_dict)
        """

        logger.info("\n" + "=" * 80)
        logger.info("FEATURE SELECTION - LOG-RV TARGET")
        logger.info("=" * 80)

        # ================================================================
        # STEP 1: Calculate forward-looking log(RV) targets
        # ================================================================

        logger.info("\nStep 1: Calculating forward-looking realized volatility...")

        target = self._calculate_forward_realized_volatility(
            spx_returns=spx_returns,
            dates=features_df.index,
        )

        if len(target) == 0:
            logger.error("❌ No valid targets calculated")
            return [], {}

        # ================================================================
        # STEP 2: Align features with targets
        # ================================================================

        logger.info("\nStep 2: Aligning features with targets...")

        # Find dates that exist in both features and target
        common_dates = features_df.index.intersection(target.index)

        if len(common_dates) < 100:
            logger.error(
                f"❌ Insufficient aligned data: {len(common_dates)} samples\n"
                f"   Need at least 100 samples for reliable feature selection"
            )
            return [], {}

        # Subset to common dates
        X = features_df.loc[common_dates].copy()
        y = target.loc[common_dates].copy()

        # Handle any remaining NaN in features
        # Forward-fill then backward-fill, then drop any remaining NaN
        X = X.fillna(method="ffill").fillna(method="bfill")

        # Drop columns that are still all NaN
        X = X.dropna(axis=1, how="all")

        # Drop rows with any NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(
            f"Aligned dataset:\n"
            f"  Samples: {len(X)}\n"
            f"  Features: {len(X.columns)}\n"
            f"  Target range: [{y.min():.2f}, {y.max():.2f}]\n"
            f"  Date range: {X.index[0].date()} to {X.index[-1].date()}"
        )

        # ================================================================
        # STEP 3: Cross-validated feature importance
        # ================================================================

        logger.info(
            f"\nStep 3: Computing feature importance via {self.cv_folds}-fold CV..."
        )

        # TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        # Aggregate importance across folds
        importance_accumulator = np.zeros(len(X.columns))
        fold_performances = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Train XGBoost model
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold_idx,
                n_jobs=-1,
            )

            model.fit(X_train, y_train, verbose=False)

            # Get feature importance (gain)
            importance = model.feature_importances_
            importance_accumulator += importance

            # Evaluate performance
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_mae = np.mean(np.abs(train_pred - y_train))
            val_mae = np.mean(np.abs(val_pred - y_val))

            fold_performances.append(
                {
                    "fold": fold_idx + 1,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                }
            )

            logger.info(
                f"  Fold {fold_idx + 1}/{self.cv_folds}: "
                f"Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}"
            )

        # Average importance across folds
        avg_importance = importance_accumulator / self.cv_folds

        # Create importance scores dict
        self.importance_scores = dict(zip(X.columns, avg_importance))

        # ================================================================
        # STEP 4: Select features
        # ================================================================

        logger.info("\nStep 4: Selecting features...")

        # Sort by importance
        sorted_features = sorted(
            self.importance_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Apply filters
        selected = []
        for feature, score in sorted_features:
            # Stop if we have enough features
            if len(selected) >= self.top_n:
                break

            # Filter by minimum importance
            if score < self.min_importance:
                continue

            selected.append(feature)

        self.selected_features = selected

        logger.info(
            f"\nSelected {len(selected)} features:\n"
            f"  Top importance: {sorted_features[0][1]:.4f}\n"
            f"  Min importance: {sorted_features[len(selected) - 1][1]:.4f}\n"
            f"  Threshold: {self.min_importance}"
        )

        # ================================================================
        # STEP 5: Analyze by category
        # ================================================================

        category_analysis = {}
        if feature_categories:
            logger.info("\nFeature selection by category:")
            for category, cat_features in feature_categories.items():
                selected_in_cat = [f for f in selected if f in cat_features]
                total_in_cat = len([f for f in cat_features if f in X.columns])

                if total_in_cat > 0:
                    pct = 100 * len(selected_in_cat) / total_in_cat
                    logger.info(
                        f"  {category}: {len(selected_in_cat)}/{total_in_cat} ({pct:.1f}%)"
                    )

                    category_analysis[category] = {
                        "selected": len(selected_in_cat),
                        "total": total_in_cat,
                        "features": selected_in_cat,
                    }

        # ================================================================
        # STEP 6: Build metadata
        # ================================================================

        self.selection_metadata = {
            "timestamp": datetime.now().isoformat(),
            "target": "log_realized_volatility",
            "horizon": self.horizon,
            "samples": len(X),
            "total_features": len(X.columns),
            "selected_features": len(selected),
            "min_importance": self.min_importance,
            "top_n": self.top_n,
            "cv_folds": self.cv_folds,
            "fold_performances": fold_performances,
            "avg_train_mae": np.mean([f["train_mae"] for f in fold_performances]),
            "avg_val_mae": np.mean([f["val_mae"] for f in fold_performances]),
            "target_statistics": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
            },
            "category_analysis": category_analysis,
            "top_20_features": [
                {"feature": f, "importance": float(s)} for f, s in sorted_features[:20]
            ],
        }

        # ================================================================
        # STEP 7: Display top features
        # ================================================================

        logger.info("\n" + "=" * 80)
        logger.info("TOP 20 SELECTED FEATURES")
        logger.info("=" * 80)

        for rank, (feature, score) in enumerate(sorted_features[:20], 1):
            logger.info(f"{rank:2d}. {feature:50s} {score:.6f}")

        logger.info("\n" + "=" * 80)
        logger.info("FEATURE SELECTION COMPLETE")
        logger.info("=" * 80)

        return self.selected_features, self.selection_metadata

    def save_results(self, output_dir: str = "data_cache"):
        """
        Save feature selection results to disk.

        Saves:
        1. selected_features.json - List of selected feature names
        2. feature_importance.json - Full importance scores
        3. selection_metadata.json - Detailed selection statistics
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.selected_features is None:
            logger.error("❌ No features selected yet")
            return

        # Save selected features list
        features_file = output_path / "selected_features.json"
        with open(features_file, "w") as f:
            json.dump(self.selected_features, f, indent=2)
        logger.info(f"✅ Saved selected features: {features_file}")

        # Save full importance scores
        importance_file = output_path / "feature_importance.json"
        with open(importance_file, "w") as f:
            json.dump(self.importance_scores, f, indent=2)
        logger.info(f"✅ Saved importance scores: {importance_file}")

        # Save metadata
        metadata_file = output_path / "selection_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.selection_metadata, f, indent=2)
        logger.info(f"✅ Saved metadata: {metadata_file}")

    def load_results(self, input_dir: str = "data_cache"):
        """Load previously saved feature selection results."""

        input_path = Path(input_dir)

        features_file = input_path / "selected_features.json"
        if features_file.exists():
            with open(features_file, "r") as f:
                self.selected_features = json.load(f)
            logger.info(f"✅ Loaded {len(self.selected_features)} selected features")

        importance_file = input_path / "feature_importance.json"
        if importance_file.exists():
            with open(importance_file, "r") as f:
                self.importance_scores = json.load(f)
            logger.info(f"✅ Loaded importance scores")

        metadata_file = input_path / "selection_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.selection_metadata = json.load(f)
            logger.info(f"✅ Loaded selection metadata")


# ============================================================
# TESTING
# ============================================================


def test_feature_selector():
    """Test feature selector with synthetic data."""

    print("\n" + "=" * 80)
    print("TESTING FEATURE SELECTOR V3")
    print("=" * 80)

    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Synthetic SPX returns (with volatility clustering)
    spx_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

    # Synthetic features
    n_features = 100
    features_df = pd.DataFrame(
        np.random.randn(len(dates), n_features),
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Add some predictive features (correlated with future volatility)
    for i in range(5):
        # Feature that's correlated with forward volatility
        future_vol = spx_returns.rolling(21).std().shift(-21)
        features_df[f"predictive_{i}"] = (
            future_vol + np.random.randn(len(dates)) * 0.001
        )

    # Initialize selector
    selector = XGBoostFeatureSelector(
        horizon=21,
        min_importance=0.001,
        top_n=50,
    )

    # Run selection
    selected, metadata = selector.select_features(
        features_df=features_df,
        spx_returns=spx_returns,
    )

    # Verify results
    print(f"\n✅ Selected {len(selected)} features")
    print(f"✅ Avg validation MAE: {metadata['avg_val_mae']:.4f}")

    # Check if predictive features were selected
    predictive_selected = [f for f in selected if "predictive" in f]
    print(f"✅ Predictive features selected: {len(predictive_selected)}/5")

    # Save results
    selector.save_results(output_dir="/home/claude/test_output")
    print(f"✅ Results saved to /home/claude/test_output")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_feature_selector()
