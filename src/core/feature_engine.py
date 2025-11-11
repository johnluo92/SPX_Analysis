"""Enhanced Feature Engine V5 - Streamlined, No Duplicates
WITH CALENDAR COHORT INTEGRATION
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import REGIME_BOUNDARIES, TRAINING_YEARS

try:
    from config import (
        CALENDAR_COHORTS,
        COHORT_PRIORITY,
        ENABLE_TEMPORAL_SAFETY,
        FEATURE_QUALITY_CONFIG,
        PUBLICATION_LAGS,
        TARGET_CONFIG,
    )
except ImportError:
    ENABLE_TEMPORAL_SAFETY = False
    PUBLICATION_LAGS = {}
    CALENDAR_COHORTS = {}
    COHORT_PRIORITY = []
    TARGET_CONFIG = {}
    FEATURE_QUALITY_CONFIG = {}
    warnings.warn("⚠️ Calendar cohort config not found - cohort features disabled")


# ==================== ROBUST HELPER FUNCTIONS ====================


def calculate_robust_zscore(series, window, min_std=1e-8):
    """Prevent inf values from division by zero"""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    rolling_std = rolling_std.clip(lower=min_std)
    zscore = (series - rolling_mean) / rolling_std
    return zscore


def calculate_regime_with_validation(series, bins, labels, feature_name="feature"):
    """Validate data availability before pd.cut()"""
    valid_pct = series.notna().sum() / len(series)

    if valid_pct < 0.5:
        return pd.Series(0, index=series.index)

    valid_values = series.dropna()
    if len(valid_values) > 0:
        value_range = valid_values.max() - valid_values.min()
        if value_range < 1e-6:
            return pd.Series(0, index=series.index)

    try:
        regime = pd.cut(series, bins=bins, labels=labels)
        return regime.fillna(0).astype(int)
    except Exception:
        return pd.Series(0, index=series.index)


def calculate_percentile_with_validation(series, window, min_data_pct=0.7):
    """Percentile calculation with sparse data handling"""

    def safe_percentile_rank(x):
        valid = x.dropna()
        if len(valid) < window * min_data_pct:
            return np.nan
        if len(valid) == 0:
            return np.nan
        last_val = x.iloc[-1]
        if pd.isna(last_val):
            return np.nan
        return (valid < last_val).sum() / len(valid) * 100

    percentile = series.rolling(window + 1).apply(safe_percentile_rank, raw=False)
    return percentile


# ==================== META FEATURE ENGINE ====================
# [Keep all existing MetaFeatureEngine code unchanged]


class MetaFeatureEngine:
    """Advanced meta-feature extraction from base features."""

    @staticmethod
    def extract_regime_indicators(
        df: pd.DataFrame, vix: pd.Series, spx: pd.Series
    ) -> pd.DataFrame:
        """Extract comprehensive regime indicators."""
        meta = pd.DataFrame(index=df.index)

        # VIX Regime
        if "vix" in df.columns:
            v = df["vix"]
            meta["vix_regime_micro"] = calculate_regime_with_validation(
                v,
                bins=[0, 12, 16, 20, 100],
                labels=[0, 1, 2, 3],
                feature_name="vix_micro",
            )
            if "vix_velocity_5d" in df.columns:
                meta["regime_transition_risk"] = (
                    df["vix_velocity_5d"].abs() / v.replace(0, np.nan) * 100
                ).clip(0, 100)

        # Volatility Regime
        if all(col in df.columns for col in ["spx_realized_vol_21d", "vix"]):
            rv = df["spx_realized_vol_21d"]
            v = df["vix"]
            meta["vol_regime"] = calculate_regime_with_validation(
                rv, bins=[0, 10, 15, 25, 100], labels=[0, 1, 2, 3], feature_name="vol"
            )
            risk_prem = v - rv
            meta["risk_premium_regime"] = calculate_regime_with_validation(
                risk_prem,
                bins=[-100, 0, 5, 10, 100],
                labels=[0, 1, 2, 3],
                feature_name="risk_premium",
            )
            if "vix_term_structure" in df.columns:
                ts = df["vix_term_structure"]
                meta["vol_term_regime"] = calculate_regime_with_validation(
                    ts,
                    bins=[-100, -2, 0, 2, 100],
                    labels=[0, 1, 2, 3],
                    feature_name="vol_term",
                )

        # Trend Regime
        if "spx_vs_ma200" in df.columns:
            trend = df["spx_vs_ma200"]
            meta["trend_regime"] = calculate_regime_with_validation(
                trend,
                bins=[-100, -5, 0, 5, 100],
                labels=[0, 1, 2, 3],
                feature_name="trend",
            )
            if "spx_vs_ma50" in df.columns:
                meta["trend_strength"] = (
                    df["spx_vs_ma200"].abs() + df["spx_vs_ma50"].abs()
                ) / 2

        # Liquidity/Stress Regime
        stress_components = []
        if "SKEW" in df.columns:
            stress_components.append(((df["SKEW"] - 130) / 30).clip(0, 1))
        if "vix" in df.columns:
            stress_components.append(((df["vix"] - 15) / 25).clip(0, 1))
        if "spx_realized_vol_21d" in df.columns:
            stress_components.append(
                ((df["spx_realized_vol_21d"] - 15) / 20).clip(0, 1)
            )

        if stress_components:
            meta["liquidity_stress_composite"] = pd.DataFrame(stress_components).T.mean(
                axis=1
            )
            meta["liquidity_regime"] = calculate_regime_with_validation(
                meta["liquidity_stress_composite"],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=[0, 1, 2, 3],
                feature_name="liquidity",
            )

        # Correlation Regime
        if "spx_vix_corr_21d" in df.columns:
            corr = df["spx_vix_corr_21d"]
            meta["correlation_regime"] = calculate_regime_with_validation(
                corr,
                bins=[-1, -0.8, -0.5, 0, 1],
                labels=[0, 1, 2, 3],
                feature_name="correlation",
            )

        return meta

    @staticmethod
    def extract_cross_asset_relationships(
        df: pd.DataFrame, macro: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Extract cross-asset correlation and divergence features."""
        meta = pd.DataFrame(index=df.index)

        # Equity-Vol Relationship
        if all(col in df.columns for col in ["spx_ret_21d", "vix_velocity_21d"]):
            spx_ret = df["spx_ret_21d"]
            vix_chg = df["vix_velocity_21d"]
            meta["equity_vol_divergence"] = (
                (spx_ret.rank(pct=True) + vix_chg.rank(pct=True)) - 1
            ).abs()
            if "spx_vix_corr_21d" in df.columns:
                corr = df["spx_vix_corr_21d"]
                corr_ma = corr.rolling(63).mean()
                meta["equity_vol_corr_breakdown"] = (corr - corr_ma).abs()

        # Risk Premium Dynamics
        if all(col in df.columns for col in ["vix", "spx_realized_vol_21d"]):
            risk_prem = df["vix"] - df["spx_realized_vol_21d"]
            meta["risk_premium_ma21"] = risk_prem.rolling(21).mean()
            meta["risk_premium_velocity"] = risk_prem.diff(10)
            meta["risk_premium_zscore"] = calculate_robust_zscore(risk_prem, 63)

        # Macro Asset Integration
        if macro is not None:
            if "Gold" in macro.columns and "spx_ret_21d" in df.columns:
                gold_ret = macro["Gold"].pct_change(21) * 100
                meta["gold_spx_divergence"] = (
                    gold_ret.rank(pct=True) - df["spx_ret_21d"].rank(pct=True)
                ).abs()
            if "Dollar_Index" in macro.columns and "spx_ret_21d" in df.columns:
                dxy_ret = macro["Dollar_Index"].pct_change(21) * 100
                meta["dollar_spx_correlation"] = dxy_ret.rolling(63).corr(
                    df["spx_ret_21d"]
                )

        return meta

    @staticmethod
    def extract_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract velocity and acceleration features."""
        meta = pd.DataFrame(index=df.index)

        roc_series = {
            "vix": df.get("vix"),
            "SKEW": df.get("SKEW"),
            "spx_realized_vol_21d": df.get("spx_realized_vol_21d"),
        }

        for name, series in roc_series.items():
            if series is None:
                continue

            # Percentage velocity only for vix
            if name == "vix":
                meta["vix_velocity_3d_pct"] = series.pct_change(3) * 100

            # Jerk for VIX and SKEW only
            if name in ["vix", "SKEW"]:
                vel_5d = series.diff(5)
                accel_5d = vel_5d.diff(5)
                meta[f"{name}_jerk_5d"] = accel_5d.diff(5)

            # Momentum regime for VIX and SKEW only
            if name in ["vix", "SKEW"]:
                meta[f"{name}_momentum_regime"] = np.sign(series.diff(5))

            # SPX realized vol specific features
            if name == "spx_realized_vol_21d":
                meta["spx_realized_vol_21d_velocity_3d"] = series.diff(3)
                meta["spx_realized_vol_21d_acceleration_5d"] = series.diff(5).diff(5)

        if all(col in df.columns for col in ["vix", "SKEW"]):
            vix_mom = df["vix"].diff(10)
            skew_mom = df["SKEW"].diff(10)
            meta["vix_skew_momentum_divergence"] = (
                vix_mom.rank(pct=True) - skew_mom.rank(pct=True)
            ).abs()

        return meta

    @staticmethod
    def extract_percentile_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Percentile rankings for key indicators."""
        meta = pd.DataFrame(index=df.index)

        ranking_series = {
            "vix": df.get("vix"),
            "SKEW": df.get("SKEW"),
        }

        if all(col in df.columns for col in ["vix", "spx_realized_vol_21d"]):
            ranking_series["risk_premium"] = df["vix"] - df["spx_realized_vol_21d"]

        for name, series in ranking_series.items():
            if series is None:
                continue

            # Percentiles for multiple windows
            for window in [21, 63, 126, 252]:
                meta[f"{name}_percentile_{window}d"] = (
                    calculate_percentile_with_validation(series, window)
                )

            if f"{name}_percentile_63d" in meta.columns:
                meta[f"{name}_percentile_velocity"] = meta[
                    f"{name}_percentile_63d"
                ].diff(10)

            # Extreme indicators - only low extremes
            for window in [63, 252]:
                pct_col = f"{name}_percentile_{window}d"
                if pct_col in meta.columns:
                    meta[f"{name}_extreme_low_{window}d"] = (meta[pct_col] < 10).astype(
                        int
                    )

        # Risk premium specific - only low extreme
        if "risk_premium_percentile_63d" in meta.columns:
            meta["risk_premium_extreme_low_63d"] = (
                meta["risk_premium_percentile_63d"] < 10
            ).astype(int)

        return meta


# ==================== FUTURES FEATURE ENGINE ====================
# [Keep all existing FuturesFeatureEngine code - unchanged]


class FuturesFeatureEngine:
    """Specialized feature extraction for futures contracts."""

    @staticmethod
    def extract_vix_futures_features(vx_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """VIX futures features - no duplicates."""
        features = pd.DataFrame()

        if "VX1-VX2" in vx_data:
            spread = vx_data["VX1-VX2"]
            features["VX1-VX2"] = spread
            features["VX1-VX2_change_21d"] = spread.diff(21)
            features["VX1-VX2_zscore_63d"] = calculate_robust_zscore(spread, 63)
            features["VX1-VX2_percentile_63d"] = calculate_percentile_with_validation(
                spread, 63
            )

        if "VX2-VX1_RATIO" in vx_data:
            ratio = vx_data["VX2-VX1_RATIO"]
            features["VX2-VX1_RATIO"] = ratio
            features["VX2-VX1_RATIO_velocity_10d"] = ratio.diff(10)
            features["vx_term_structure_regime"] = calculate_regime_with_validation(
                ratio,
                bins=[-1, -0.05, 0, 0.05, 1],
                labels=[0, 1, 2, 3],
                feature_name="vx_ratio",
            )

        if "VX1-VX2" in vx_data and "VX2-VX1_RATIO" in vx_data:
            spread = vx_data["VX1-VX2"]
            ratio = vx_data["VX2-VX1_RATIO"]
            features["vx_curve_acceleration"] = ratio.diff(5).diff(5)
            spread_rank = spread.rolling(63).rank(pct=True)
            ratio_rank = ratio.rolling(63).rank(pct=True)
            features["vx_term_structure_divergence"] = (spread_rank - ratio_rank).abs()

        return features

    @staticmethod
    def extract_commodity_futures_features(
        futures_data: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Commodity futures features."""
        features = pd.DataFrame()

        if "CL1-CL2" in futures_data:
            cl_spread = futures_data["CL1-CL2"]
            features["CL1-CL2"] = cl_spread
            features["CL1-CL2_velocity_5d"] = cl_spread.diff(5)
            features["CL1-CL2_zscore_63d"] = calculate_robust_zscore(cl_spread, 63)
            features["oil_term_regime"] = calculate_regime_with_validation(
                cl_spread,
                bins=[-10, -1, 0, 2, 20],
                labels=[0, 1, 2, 3],
                feature_name="cl_spread",
            )

        if "Crude_Oil" in futures_data:
            price = futures_data["Crude_Oil"]
            for window in [10, 21, 63]:
                features[f"crude_oil_ret_{window}d"] = price.pct_change(window) * 100
            features["crude_oil_vol_21d"] = (
                price.pct_change().rolling(21).std() * np.sqrt(252) * 100
            )
            features["crude_oil_zscore_63d"] = calculate_robust_zscore(price, 63)

        return features

    @staticmethod
    def extract_dollar_futures_features(
        dollar_data: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Dollar futures features - spread only, no dxy_ret_* (handled in macro)."""
        features = pd.DataFrame()

        if "DX1-DX2" in dollar_data:
            dx_spread = dollar_data["DX1-DX2"]
            features["DX1-DX2"] = dx_spread
            features["DX1-DX2_velocity_5d"] = dx_spread.diff(5)
            features["DX1-DX2_zscore_63d"] = calculate_robust_zscore(dx_spread, 63)

        if "Dollar_Index" in dollar_data:
            dxy = dollar_data["Dollar_Index"]
            for window in [50, 200]:
                ma = dxy.rolling(window).mean()
                features[f"dxy_vs_ma{window}"] = (
                    (dxy - ma) / ma.replace(0, np.nan)
                ) * 100
            features["dxy_vol_21d"] = (
                dxy.pct_change().rolling(21).std() * np.sqrt(252) * 100
            )

        return features

    @staticmethod
    def extract_futures_cross_relationships(
        vx_data: Dict[str, pd.Series],
        commodity_data: Dict[str, pd.Series],
        dollar_data: Dict[str, pd.Series],
        spx_ret: pd.Series = None,
    ) -> pd.DataFrame:
        """Cross-futures relationships."""
        features = pd.DataFrame()

        if "VX1-VX2" in vx_data and "CL1-CL2" in commodity_data:
            vx_spread = vx_data["VX1-VX2"]
            cl_spread = commodity_data["CL1-CL2"]
            features["vx_crude_corr_21d"] = vx_spread.rolling(21).corr(cl_spread)
            vx_rank = vx_spread.rolling(63).rank(pct=True)
            cl_rank = cl_spread.rolling(63).rank(pct=True)
            features["vx_crude_divergence"] = (vx_rank - cl_rank).abs()

        if "VX1-VX2" in vx_data and "Dollar_Index" in dollar_data:
            vx_spread = vx_data["VX1-VX2"]
            dxy_ret = dollar_data["Dollar_Index"].pct_change(21) * 100
            features["vx_dollar_corr_21d"] = vx_spread.rolling(21).corr(dxy_ret)

        if "Dollar_Index" in dollar_data and "Crude_Oil" in commodity_data:
            dxy = dollar_data["Dollar_Index"]
            crude = commodity_data["Crude_Oil"]
            features["dollar_crude_corr_21d"] = (
                dxy.pct_change().rolling(21).corr(crude.pct_change())
            )

        if spx_ret is not None:
            if "VX1-VX2" in vx_data:
                features["spx_vx_spread_corr_21d"] = spx_ret.rolling(21).corr(
                    vx_data["VX1-VX2"]
                )
            if "Dollar_Index" in dollar_data:
                dxy_ret = dollar_data["Dollar_Index"].pct_change(21) * 100
                features["spx_dollar_corr_21d"] = spx_ret.rolling(21).corr(dxy_ret)

        return features


# ==================== TREASURY YIELD FEATURE ENGINE ====================
# [Keep all existing TreasuryYieldFeatureEngine code - unchanged]


class TreasuryYieldFeatureEngine:
    """Treasury yield curve features."""

    @staticmethod
    def extract_term_spreads(yields: pd.DataFrame) -> pd.DataFrame:
        """Calculate term spreads."""
        features = pd.DataFrame(index=yields.index)

        required = ["DGS3MO", "DGS2", "DGS10"]
        if not all(col in yields.columns for col in required):
            return features

        # Key spreads
        features["yield_10y2y"] = yields["DGS10"] - yields["DGS2"]
        features["yield_10y2y_zscore"] = calculate_robust_zscore(
            features["yield_10y2y"], 252
        )

        features["yield_10y3m"] = yields["DGS10"] - yields["DGS3MO"]
        features["yield_2y3m"] = yields["DGS2"] - yields["DGS3MO"]

        if "DGS5" in yields.columns:
            features["yield_5y2y"] = yields["DGS5"] - yields["DGS2"]

        if "DGS30" in yields.columns:
            features["yield_30y10y"] = yields["DGS30"] - yields["DGS10"]

        # Spread dynamics
        for spread_name in ["yield_10y2y", "yield_10y3m", "yield_2y3m"]:
            if spread_name not in features.columns:
                continue
            spread = features[spread_name]
            features[f"{spread_name}_velocity_10d"] = spread.diff(10)
            features[f"{spread_name}_velocity_21d"] = spread.diff(21)
            features[f"{spread_name}_velocity_63d"] = spread.diff(63)
            features[f"{spread_name}_acceleration"] = spread.diff(10).diff(10)
            features[f"{spread_name}_vs_ma63"] = spread - spread.rolling(63).mean()
            features[f"{spread_name}_percentile_252d"] = (
                calculate_percentile_with_validation(spread, 252)
            )

        if "yield_10y2y" in features.columns:
            features["yield_10y2y_inversion_depth"] = (
                features["yield_10y2y"].clip(upper=0).abs()
            )
        if "yield_10y3m" in features.columns:
            features["yield_10y3m_inversion_depth"] = (
                features["yield_10y3m"].clip(upper=0).abs()
            )

        return features

    @staticmethod
    def extract_curve_shape(yields: pd.DataFrame) -> pd.DataFrame:
        """Yield curve shape features."""
        features = pd.DataFrame(index=yields.index)

        required = ["DGS3MO", "DGS2", "DGS10"]
        if not all(col in yields.columns for col in required):
            return features

        # Curve level
        available_yields = [
            col
            for col in ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]
            if col in yields.columns
        ]
        if available_yields:
            features["yield_curve_level"] = yields[available_yields].mean(axis=1)
            features["yield_curve_level_zscore"] = calculate_robust_zscore(
                features["yield_curve_level"], 252
            )

        # Curve curvature
        if "DGS5" in yields.columns:
            features["yield_curve_curvature"] = (
                2 * yields["DGS5"] - yields["DGS2"] - yields["DGS10"]
            )
            features["yield_curve_curvature_zscore"] = calculate_robust_zscore(
                features["yield_curve_curvature"], 252
            )

        return features

    @staticmethod
    def extract_rate_volatility(yields: pd.DataFrame) -> pd.DataFrame:
        """Interest rate volatility features."""
        features = pd.DataFrame(index=yields.index)

        for col in ["DGS6MO", "DGS10", "DGS30"]:
            if col in yields.columns:
                daily_change = yields[col].diff()
                features[f"{col.lower()}_vol_63d"] = daily_change.rolling(
                    63
                ).std() * np.sqrt(252)

        # Aggregate rate vol
        vol_cols = [col for col in features.columns if "_vol_" in col]
        if vol_cols:
            features["yield_curve_vol_avg"] = features[vol_cols].mean(axis=1)
            features["yield_curve_vol_dispersion"] = features[vol_cols].std(axis=1)

        return features


# ==================== UNIFIED FEATURE ENGINE WITH CALENDAR COHORTS ====================


class UnifiedFeatureEngine:
    """Enhanced unified feature engine - WITH CALENDAR COHORT INTEGRATION"""

    def __init__(self, data_fetcher):
        self.fetcher = data_fetcher
        self.meta_engine = MetaFeatureEngine()
        self.futures_engine = FuturesFeatureEngine()
        self.treasury_engine = TreasuryYieldFeatureEngine()

        # Calendar data for cohort classification
        self.fomc_calendar = None
        self.opex_calendar = None
        self.earnings_calendar = None
        self.vix_futures_expiry = None

        # Cache for performance
        self._cohort_cache = {}  # {date: (cohort, weight)}

    # ==================== CALENDAR DATA LOADING ====================

    def _load_calendar_data(self):
        """Load all calendar sources once at startup."""
        if self.fomc_calendar is None:
            try:
                # Load from data_fetcher
                self.fomc_calendar = self.fetcher.fetch_fomc_calendar()
                print(f"✅ FOMC calendar loaded: {len(self.fomc_calendar)} meetings")
            except Exception as e:
                print(f"⚠️  FOMC calendar load failed: {e}, using stub")
                self.fomc_calendar = pd.DataFrame()  # Empty fallback

        # Generate OpEx calendar (always 3rd Friday of month)
        if self.opex_calendar is None:
            self.opex_calendar = self._generate_opex_calendar()
            print(f"✅ OpEx calendar generated: {len(self.opex_calendar)} dates")

        # VIX futures expiry (Wednesday 30 days before S&P OpEx)
        if self.vix_futures_expiry is None:
            self.vix_futures_expiry = self._generate_vix_futures_expiry()
            print(
                f"✅ VIX futures expiry calendar: {len(self.vix_futures_expiry)} dates"
            )

        # Earnings calendar (stub - implement later or use API)
        if self.earnings_calendar is None:
            self.earnings_calendar = pd.DataFrame()
            print("⚠️  Earnings calendar not implemented (will use default cohort)")

    def _generate_opex_calendar(self, start_year=2009, end_year=2030):
        """
        Generate monthly options expiration dates (3rd Friday of each month).

        Returns:
            DataFrame with columns: [date, expiry_type]
        """
        opex_dates = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Find 3rd Friday of month
                # Start from 15th (earliest 3rd Friday), find next Friday
                first_possible = pd.Timestamp(year, month, 15)

                # Find next Friday (weekday 4)
                days_ahead = (4 - first_possible.weekday()) % 7
                if days_ahead == 0 and first_possible.day > 15:
                    days_ahead = 7  # Move to next week if already past

                third_friday = first_possible + pd.Timedelta(days=days_ahead)
                opex_dates.append({"date": third_friday, "expiry_type": "monthly_opex"})

        df = pd.DataFrame(opex_dates)
        df = df.set_index("date").sort_index()
        return df

    def _generate_vix_futures_expiry(self):
        """
        VIX futures expire on Wednesday, 30 days before 3rd Friday.

        Returns:
            DataFrame with columns: [date, expiry_type]
        """
        if self.opex_calendar is None:
            self._generate_opex_calendar()

        vix_expiry = []
        for opex_date in self.opex_calendar.index:
            # 30 days before, find Wednesday
            approx_date = opex_date - pd.Timedelta(days=30)

            # Find next Wednesday (weekday 2)
            days_to_wed = (2 - approx_date.weekday()) % 7
            vix_date = approx_date + pd.Timedelta(days=days_to_wed)

            vix_expiry.append({"date": vix_date, "expiry_type": "vix_futures"})

        df = pd.DataFrame(vix_expiry)
        df = df.set_index("date").sort_index()
        return df

    # ==================== COHORT CLASSIFICATION ====================

    def get_calendar_cohort(self, date):
        """
        Determine which calendar cohort a date belongs to.

        Args:
            date: pd.Timestamp or datetime

        Returns:
            tuple: (cohort_name: str, weight: float)

        Example:
            >>> get_calendar_cohort(pd.Timestamp('2025-01-15'))
            ('monthly_opex_minus_1', 1.5)
        """
        date = pd.Timestamp(date)

        # Check cache first (performance optimization)
        if date in self._cohort_cache:
            return self._cohort_cache[date]

        # Ensure calendars are loaded
        if self.opex_calendar is None:
            self._load_calendar_data()

        # Calculate days to various events
        days_to_opex = self._days_to_monthly_opex(date)
        days_to_fomc = self._days_to_fomc(date)
        days_to_vix_expiry = self._days_to_vix_futures_expiry(date)
        earnings_pct = self._spx_earnings_intensity(date)

        # Match against cohorts in priority order
        for cohort_name in COHORT_PRIORITY:
            cohort_def = CALENDAR_COHORTS[cohort_name]
            condition = cohort_def["condition"]

            if condition == "days_to_monthly_opex":
                range_min, range_max = cohort_def["range"]
                if range_min <= days_to_opex <= range_max:
                    result = (cohort_name, cohort_def["weight"])
                    self._cohort_cache[date] = result
                    return result

            elif condition == "days_to_fomc":
                if days_to_fomc is not None:
                    range_min, range_max = cohort_def["range"]
                    if range_min <= days_to_fomc <= range_max:
                        result = (cohort_name, cohort_def["weight"])
                        self._cohort_cache[date] = result
                        return result

            elif condition == "days_to_futures_expiry":
                if days_to_vix_expiry is not None:
                    range_min, range_max = cohort_def["range"]
                    if range_min <= days_to_vix_expiry <= range_max:
                        result = (cohort_name, cohort_def["weight"])
                        self._cohort_cache[date] = result
                        return result

            elif condition == "spx_earnings_pct":
                if earnings_pct is not None:
                    range_min, range_max = cohort_def["range"]
                    if range_min <= earnings_pct <= range_max:
                        result = (cohort_name, cohort_def["weight"])
                        self._cohort_cache[date] = result
                        return result

            elif condition == "default":
                # Catch-all for mid_cycle
                result = (cohort_name, cohort_def["weight"])
                self._cohort_cache[date] = result
                return result

        # Should never reach here if 'mid_cycle' is last in priority
        raise ValueError(f"No cohort matched for date {date}")

    def _days_to_monthly_opex(self, date):
        """
        Calculate days until next monthly OpEx (3rd Friday).

        Returns:
            int: Negative if before OpEx, 0 on OpEx, positive after
            Example: -5 means "5 days until OpEx"
        """
        # Find next OpEx date
        future_opex = self.opex_calendar[self.opex_calendar.index >= date]

        if len(future_opex) == 0:
            return None  # No future OpEx (end of calendar)

        next_opex = future_opex.index[0]
        days_diff = (next_opex - date).days

        return -days_diff  # Negative before, positive after

    def _days_to_fomc(self, date):
        """
        Calculate days until next FOMC meeting.

        Returns:
            int: Days to next meeting (negative before, positive after)
            None: If FOMC calendar unavailable
        """
        if self.fomc_calendar is None or len(self.fomc_calendar) == 0:
            return None

        future_fomc = self.fomc_calendar[self.fomc_calendar.index >= date]

        if len(future_fomc) == 0:
            return None

        next_fomc = future_fomc.index[0]
        days_diff = (next_fomc - date).days

        return -days_diff

    def _days_to_vix_futures_expiry(self, date):
        """Calculate days until next VIX futures expiration."""
        if self.vix_futures_expiry is None or len(self.vix_futures_expiry) == 0:
            return None

        future_expiry = self.vix_futures_expiry[self.vix_futures_expiry.index >= date]

        if len(future_expiry) == 0:
            return None

        next_expiry = future_expiry.index[0]
        days_diff = (next_expiry - date).days

        return -days_diff

    def _spx_earnings_intensity(self, date):
        """
        Calculate % of SPX components reporting earnings this week.

        Returns:
            float: Percentage [0.0, 1.0] of SPX reporting
            None: If earnings calendar unavailable (stub implementation)
        """
        # STUB IMPLEMENTATION
        # TODO: Integrate with earnings calendar API or manual CSV

        # For now, use heuristic: Peak earnings months are Jan, Apr, Jul, Oct
        month = date.month
        if month in [1, 4, 7, 10]:
            # Check if in earnings window (typically 2nd-4th week of month)
            week_of_month = (date.day - 1) // 7 + 1
            if week_of_month in [2, 3, 4]:
                return 0.25  # Assume 25% of SPX reporting

        return 0.05  # Low intensity otherwise

    # ==================== FEATURE QUALITY COMPUTATION ====================

    def _compute_feature_quality_vectorized(self, df):
        """
        Compute feature quality score for each row.
        Based on missingness and staleness of features.

        Returns:
            pd.Series: Quality scores [0, 1] where 1 = perfect
        """
        if not FEATURE_QUALITY_CONFIG:
            return pd.Series(1.0, index=df.index)

        quality_scores = []

        for idx, row in df.iterrows():
            score_components = []

            # Check critical features (must be present)
            for feat in FEATURE_QUALITY_CONFIG.get("missingness_penalty", {}).get(
                "critical_features", []
            ):
                if feat in df.columns:
                    if pd.isna(row[feat]):
                        score_components.append(0.0)  # Critical missing = fail
                    else:
                        score_components.append(1.0)

            # Check important features (0.5 if missing)
            for feat in FEATURE_QUALITY_CONFIG.get("missingness_penalty", {}).get(
                "important_features", []
            ):
                if feat in df.columns:
                    if pd.isna(row[feat]):
                        score_components.append(0.5)
                    else:
                        score_components.append(1.0)

            # Check optional features (0.9 if missing)
            for feat in FEATURE_QUALITY_CONFIG.get("missingness_penalty", {}).get(
                "optional_features", []
            ):
                if feat in df.columns:
                    if pd.isna(row[feat]):
                        score_components.append(0.9)
                    else:
                        score_components.append(1.0)

            # Average all components
            if len(score_components) > 0:
                quality_scores.append(np.mean(score_components))
            else:
                quality_scores.append(1.0)  # Default if no tracked features

        return pd.Series(quality_scores, index=df.index)

    # ==================== FEATURE METADATA GENERATION ====================

    def _generate_feature_metadata(
        self,
        features: pd.DataFrame,
        spx: pd.Series,
        vix: pd.Series,
        cboe_data: pd.DataFrame = None,
        macro_df: pd.DataFrame = None,
    ) -> Dict[str, Dict]:
        """
        Generate metadata tracking 'as-of' timestamps for each feature.

        Returns:
            Dict mapping feature_name -> {
                'source': str,
                'last_available_date': datetime,
                'publication_lag': int,
                'feature_type': str
            }
        """
        metadata = {}

        # Helper to get last valid date
        def get_last_valid_date(series):
            if series is None or len(series) == 0:
                return None
            valid = series.dropna()
            return valid.index[-1] if len(valid) > 0 else None

        # Map feature patterns to sources
        source_patterns = {
            "vix": ("^VIX", 0),
            "spx": ("^GSPC", 0),
            "SKEW": ("SKEW", 0),
            "VXTLT": ("VXTLT", 0),
            "VX1": ("VX1-VX2", 0),
            "VX2": ("VX2-VX1_RATIO", 0),
            "dgs": ("DGS10", 1),  # Generic treasury
            "yield": ("DGS10", 1),
            "crude": ("CL=F", 0),
            "cl_": ("CL1-CL2", 0),
            "dxy": ("DX-Y.NYB", 0),
            "dollar": ("DTWEXBGS", 1),
            "CPI": ("CPIAUCSL", 14),
        }

        for col in features.columns:
            col_lower = col.lower()

            # Determine source and lag
            source = "derived"
            lag = 0
            feature_type = "computed"
            last_date = None

            # Check if it's a base data feature
            for pattern, (src, pub_lag) in source_patterns.items():
                if pattern in col_lower:
                    source = src
                    lag = pub_lag
                    feature_type = (
                        "base"
                        if pattern in col_lower[: len(pattern) + 5]
                        else "derived"
                    )

                    # Get last available date from source
                    if pattern == "vix" and vix is not None:
                        last_date = get_last_valid_date(vix)
                    elif pattern == "spx" and spx is not None:
                        last_date = get_last_valid_date(spx)
                    elif cboe_data is not None and source in cboe_data.columns:
                        last_date = get_last_valid_date(cboe_data[source])
                    elif macro_df is not None:
                        # Try to find in macro_df
                        for mcol in macro_df.columns:
                            if pattern in mcol.lower():
                                last_date = get_last_valid_date(macro_df[mcol])
                                break
                    break

            # If no source match, get from feature itself
            if last_date is None:
                last_date = get_last_valid_date(features[col])

            metadata[col] = {
                "source": source,
                "last_available_date": last_date,
                "publication_lag": lag,
                "feature_type": feature_type,
            }

        return metadata

    def _validate_term_structure_timing(
        self, vix: pd.Series, cboe_data: pd.DataFrame, prediction_date: datetime = None
    ) -> bool:
        """
        Validate VIX term structure calculation doesn't use future data.

        Addresses Gap 2: VIX3M edge case where forward-fill might create T+1 leakage.

        Args:
            vix: VIX spot series
            cboe_data: CBOE data with VIX3M
            prediction_date: Date to validate (if None, uses latest)

        Returns:
            True if valid, raises warning if potential leakage detected
        """
        if cboe_data is None or "VIX3M" not in cboe_data.columns:
            return True

        if prediction_date is None:
            prediction_date = vix.index[-1]

        # Check VIX3M availability at prediction date
        vix3m = cboe_data["VIX3M"]

        # Get publication lag for VIX3M (should be T+0 like VIX)
        vix3m_lag = PUBLICATION_LAGS.get("VIX3M", 0)

        # Validate VIX3M is not from the future
        latest_vix3m = vix3m.dropna().index[-1] if len(vix3m.dropna()) > 0 else None

        if latest_vix3m is not None:
            allowed_date = prediction_date - timedelta(days=vix3m_lag)

            if latest_vix3m > allowed_date:
                warnings.warn(
                    f"⚠️ VIX3M term structure may have T+{vix3m_lag} leakage: "
                    f"Using data from {latest_vix3m.date()} "
                    f"but prediction date is {prediction_date.date()}"
                )
                return False

        return True

    # ==================== QUALITY CONTROL ====================

    def apply_quality_control(self, features: pd.DataFrame):
        """Apply quality control if available."""
        if not hasattr(self, "quality_controller"):
            return features

        print("\n" + "=" * 80)
        print("🛡️ QUALITY CONTROL")
        print("=" * 80)

        clean_features, report = self.quality_controller.validate_features(features)

        import os

        os.makedirs("./data_cache", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.quality_controller.save_report(
            report, f"./data_cache/quality_report_{timestamp}.json"
        )

        return clean_features

    # ==================== MAIN BUILD METHOD ====================

    def build_complete_features(self, years: int = TRAINING_YEARS) -> dict:
        """Build complete feature set - WITH TEMPORAL SAFETY AND CALENDAR COHORTS."""
        print(
            f"\n{'=' * 80}\nENHANCED FEATURE ENGINE V5 - WITH CALENDAR COHORTS\n{'=' * 80}\nWindow: {years}y"
        )

        if ENABLE_TEMPORAL_SAFETY:
            print(
                f"🔒 TEMPORAL SAFETY ENABLED - {len(PUBLICATION_LAGS)} lags configured"
            )
        else:
            print(
                "⚠️  WARNING: TEMPORAL SAFETY DISABLED - predictions may have look-ahead bias!"
            )

        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=years * 365 + 450)
        start_str, end_str = (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        # Core Data (temporal safety handled in data_fetcher)
        print("\n[1/8] Core market data (SPX, VIX)...")
        spx_df = self.fetcher.fetch_yahoo("^GSPC", start_str, end_str)
        vix = self.fetcher.fetch_yahoo("^VIX", start_str, end_str)

        if spx_df is None or vix is None:
            raise ValueError("❌ Core data fetch failed")

        spx = spx_df["Close"].squeeze()
        vix = vix["Close"].squeeze()

        # CRITICAL: Use 'ffill' ONLY for alignment, NOT for filling missing future data
        vix = vix.reindex(spx.index, method="ffill", limit=5)
        spx_ohlc = spx_df.reindex(spx.index, method="ffill", limit=5)

        print(f"   ✅ SPX: {len(spx)} | VIX: {len(vix)}")

        # CBOE Data
        print("\n[2/8] CBOE data...")
        cboe_dict = self.fetcher.fetch_all_cboe()
        if cboe_dict:
            cboe_data = pd.DataFrame(index=spx.index)
            for symbol, series in cboe_dict.items():
                cboe_data[symbol] = series.reindex(spx.index, method="ffill", limit=5)
            print(f"   ✅ {len(cboe_data.columns)} CBOE series loaded")
        else:
            cboe_data = pd.DataFrame(index=spx.index)
            print("   ⚠️ CBOE data not available")

        # Base Features
        print("\n[3/8] Base features...")
        base_features = self._build_base_features(spx, vix, spx_ohlc, cboe_data)
        print(f"   ✅ {len(base_features.columns)} base features")

        # CBOE Features
        cboe_features = (
            self._build_cboe_features(cboe_data, vix)
            if not cboe_data.empty
            else pd.DataFrame(index=spx.index)
        )
        if not cboe_features.empty:
            print(f"   ✅ {len(cboe_features.columns)} CBOE features")

        # Futures Data
        print("\n[4/8] Futures data...")
        futures_features = self._build_futures_features(
            start_str, end_str, spx.index, spx, cboe_data
        )
        print(f"   ✅ {len(futures_features.columns)} futures features")

        # Macro Data
        print("\n[5/8] Macro data...")
        macro_df = self._fetch_macro_data(start_str, end_str, spx.index)
        macro_features = (
            self._build_macro_features(macro_df)
            if macro_df is not None
            else pd.DataFrame(index=spx.index)
        )
        print(f"   ✅ {len(macro_features.columns)} macro features")

        # Treasury Features
        print("\n[6/8] Treasury yield curve...")
        treasury_features = self._build_treasury_features(start_str, end_str, spx.index)
        print(f"   ✅ {len(treasury_features.columns)} treasury features")

        # Meta Features
        print("\n[7/8] Meta features...")
        combined_base = pd.concat(
            [
                base_features,
                cboe_features,
                futures_features,
                macro_features,
                treasury_features,
            ],
            axis=1,
        )
        meta_features = self._build_meta_features(combined_base, spx, vix, macro_df)
        print(f"   ✅ {len(meta_features.columns)} meta features")

        # Calendar Features
        print("\n[8/8] Calendar features...")
        calendar_features = self._build_calendar_features(spx.index)
        print(f"   ✅ {len(calendar_features.columns)} calendar features")

        # Combine All Features
        print("\n" + "=" * 80)
        print("📊 CONSOLIDATING FEATURES")
        print("=" * 80)

        all_features = pd.concat(
            [
                base_features,
                cboe_features,
                futures_features,
                macro_features,
                treasury_features,
                meta_features,
                calendar_features,
            ],
            axis=1,
        )

        # Remove any remaining duplicates
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = self._ensure_numeric_dtypes(all_features)
        print(f"\nTotal features before cohorts: {len(all_features.columns)}")

        # ==================== ADD CALENDAR COHORTS ====================
        print("\n📅 ADDING CALENDAR COHORTS")
        self._load_calendar_data()  # Ensure calendars loaded

        cohort_data = []
        for date in all_features.index:
            cohort_name, cohort_weight = self.get_calendar_cohort(date)
            cohort_data.append(
                {"calendar_cohort": cohort_name, "cohort_weight": cohort_weight}
            )

        cohort_df = pd.DataFrame(cohort_data, index=all_features.index)
        all_features = pd.concat([all_features, cohort_df], axis=1)

        # Log cohort distribution
        cohort_counts = all_features["calendar_cohort"].value_counts()
        print("📊 Cohort Distribution:")
        for cohort, count in cohort_counts.items():
            pct = count / len(all_features) * 100
            print(f"   {cohort:30s} | {count:4d} rows ({pct:5.1f}%)")

        # Add feature quality tracking
        print("\n🔍 COMPUTING FEATURE QUALITY SCORES")
        all_features["feature_quality"] = self._compute_feature_quality_vectorized(
            all_features
        )

        print(
            f"\n✅ Final feature count: {len(all_features.columns)} (includes 3 metadata cols)"
        )
        print(f"   Features: {len(all_features.columns) - 3}")
        print(f"   Metadata: calendar_cohort, cohort_weight, feature_quality")

        # Apply Quality Control
        all_features = self.apply_quality_control(all_features)

        if ENABLE_TEMPORAL_SAFETY:
            print("🔒 All features respect publication delays")

        # Generate feature metadata
        print("\n[9/9] Generating feature metadata...")
        feature_metadata = self._generate_feature_metadata(
            all_features, spx, vix, cboe_data, macro_df
        )

        # Validate VIX term structure timing
        if ENABLE_TEMPORAL_SAFETY and not cboe_data.empty:
            self._validate_term_structure_timing(vix, cboe_data)

        print(f"✅ Feature metadata generated for {len(feature_metadata)} features")

        # Calculate metadata statistics
        features_with_timestamps = sum(
            1
            for m in feature_metadata.values()
            if m.get("last_available_date") is not None
        )
        coverage_pct = round(100 * features_with_timestamps / len(feature_metadata), 1)

        print(
            f"📊 Metadata coverage: {features_with_timestamps}/{len(feature_metadata)} ({coverage_pct}%)"
        )

        print("=" * 80)

        return {
            "features": all_features,
            "spx": spx,
            "vix": vix,
            "cboe_data": cboe_data if cboe_dict else None,
            "metadata": feature_metadata,
            "temporal_validation": {
                "enabled": ENABLE_TEMPORAL_SAFETY,
                "feature_count": len(all_features.columns),
                "date_range": (all_features.index[0], all_features.index[-1]),
                "metadata_coverage_pct": coverage_pct,
            },
        }

    # ==================== FEATURE BUILDING METHODS (KEEP EXISTING) ====================

    def _build_base_features(
        self,
        spx: pd.Series,
        vix: pd.Series,
        spx_ohlc: pd.DataFrame,
        cboe_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Build base VIX and SPX features."""
        features = pd.DataFrame(index=spx.index)

        # VIX Core
        features["vix"] = vix
        features["spx_lag1"] = spx.shift(1)

        # VIX Returns
        for window in [1, 5, 10, 21]:
            features[f"vix_ret_{window}d"] = vix.pct_change(window) * 100

        # VIX Volatility
        vix_ret = vix.pct_change()
        for window in [10, 21, 63]:
            features[f"vix_vol_{window}d"] = (
                vix_ret.rolling(window).std() * np.sqrt(252) * 100
            )

        # VIX Velocity (absolute change)
        for window in [5, 10, 21]:
            features[f"vix_velocity_{window}d"] = vix.diff(window)

        # VIX vs Moving Averages
        for window in [10, 21, 63, 252]:
            ma = vix.rolling(window).mean()
            features[f"vix_vs_ma{window}"] = ((vix - ma) / ma.replace(0, np.nan)) * 100

        # VIX Z-Scores
        for window in [21, 63, 252]:
            features[f"vix_zscore_{window}d"] = calculate_robust_zscore(vix, window)

        # VIX Momentum Z-Scores
        for window in [10, 21]:
            mom = vix.diff(window)
            features[f"vix_momentum_z_{window}d"] = calculate_robust_zscore(mom, 63)

        # VIX Acceleration
        features["vix_accel_5d"] = vix.diff(5).diff(5)

        # VIX Mean Reversion
        vix_ma21 = vix.rolling(21).mean()
        vix_ma63 = vix.rolling(63).mean()
        features["vix_stretch_ma21"] = (vix - vix_ma21).abs()
        features["vix_stretch_ma63"] = (vix - vix_ma63).abs()

        # Reversion Strength
        for window in [21, 63]:
            ma = vix.rolling(window).mean()
            deviation = (vix - ma).abs()
            features[f"reversion_strength_{window}d"] = deviation / ma.replace(
                0, np.nan
            )

        # VIX Bollinger Bands
        bb_window = 20
        bb_ma = vix.rolling(bb_window).mean()
        bb_std = vix.rolling(bb_window).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        features["vix_bb_position_20d"] = (
            (vix - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        ).clip(0, 1)

        # VIX Extreme Indicators
        features["vix_extreme_low_21d"] = (vix < vix.rolling(21).quantile(0.1)).astype(
            int
        )

        # VIX Regime
        features["vix_regime"] = calculate_regime_with_validation(
            vix, bins=REGIME_BOUNDARIES, labels=[0, 1, 2, 3], feature_name="vix"
        )

        # VIX Regime Duration
        regime_change = features["vix_regime"].diff().fillna(0) != 0
        features["days_in_regime"] = (~regime_change).cumsum() - (
            ~regime_change
        ).cumsum().where(regime_change).ffill().fillna(0)

        # VIX Term Structure (VIX vs VIX3M)
        if cboe_data is not None and "VIX3M" in cboe_data.columns:
            vix3m = cboe_data["VIX3M"]
            features["vix_term_structure"] = (
                (vix / vix3m.replace(0, np.nan)) - 1
            ) * 100
        else:
            features["vix_term_structure"] = np.nan

        # SPX Returns
        for window in [1, 5, 10, 21, 63]:
            features[f"spx_ret_{window}d"] = spx.pct_change(window) * 100

        # SPX vs Moving Averages
        for window in [20, 50, 200]:
            ma = spx.rolling(window).mean()
            features[f"spx_vs_ma{window}"] = ((spx - ma) / ma.replace(0, np.nan)) * 100

        # SPX Momentum Z-Scores
        for window in [10, 21]:
            mom = spx.pct_change(window) * 100
            features[f"spx_momentum_z_{window}d"] = calculate_robust_zscore(mom, 63)

        # SPX Realized Volatility
        spx_ret = spx.pct_change()
        for window in [10, 21, 63]:
            features[f"spx_realized_vol_{window}d"] = (
                spx_ret.rolling(window).std() * np.sqrt(252) * 100
            )

        # SPX Vol Ratios
        if "spx_realized_vol_10d" in features and "spx_realized_vol_21d" in features:
            features["spx_vol_ratio_10_21"] = features[
                "spx_realized_vol_10d"
            ] / features["spx_realized_vol_21d"].replace(0, np.nan)
        if "spx_realized_vol_10d" in features and "spx_realized_vol_63d" in features:
            features["spx_vol_ratio_10_63"] = features[
                "spx_realized_vol_10d"
            ] / features["spx_realized_vol_63d"].replace(0, np.nan)

        # SPX Skew and Kurtosis
        features["spx_skew_21d"] = spx_ret.rolling(21).skew()
        features["spx_kurt_21d"] = spx_ret.rolling(21).kurt()

        # SPX Bollinger Bands
        bb_window = 20
        bb_ma = spx.rolling(bb_window).mean()
        bb_std = spx.rolling(bb_window).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        features["bb_position_20d"] = (
            (spx - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        ).clip(0, 1)
        features["bb_width_20d"] = (
            (bb_upper - bb_lower) / bb_ma.replace(0, np.nan)
        ) * 100

        # SPX Technical Indicators
        # RSI
        delta = spx.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # RSI Regime
        features["rsi_regime"] = calculate_regime_with_validation(
            features["rsi_14"],
            bins=[0, 30, 70, 100],
            labels=[0, 1, 2],
            feature_name="rsi",
        )

        # RSI Divergence
        rsi_ma = features["rsi_14"].rolling(21).mean()
        features["rsi_divergence"] = features["rsi_14"] - rsi_ma

        # MACD
        ema12 = spx.ewm(span=12).mean()
        ema26 = spx.ewm(span=26).mean()
        features["macd"] = ema12 - ema26
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # ADX
        high = spx_ohlc["High"]
        low = spx_ohlc["Low"]
        close = spx_ohlc["Close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        features["adx_14"] = dx.rolling(14).mean()

        # Trend Strength
        features["trend_strength"] = features["spx_vs_ma200"].abs()

        # OHLC Microstructure
        open_price = spx_ohlc["Open"]
        high_price = spx_ohlc["High"]
        low_price = spx_ohlc["Low"]
        close_price = spx_ohlc["Close"]

        features["spx_body_size"] = (close_price - open_price).abs()
        features["spx_range"] = high_price - low_price
        features["spx_range_pct"] = (
            (high_price - low_price) / close_price.replace(0, np.nan) * 100
        )

        features["spx_upper_shadow"] = high_price - close_price.combine(open_price, max)
        features["spx_lower_shadow"] = close_price.combine(open_price, min) - low_price

        features["spx_close_position"] = (close_price - low_price) / (
            high_price - low_price
        ).replace(0, np.nan)
        features["spx_body_to_range"] = features["spx_body_size"] / features[
            "spx_range"
        ].replace(0, np.nan)

        features["spx_gap"] = open_price - close_price.shift(1)
        features["spx_gap_magnitude"] = features["spx_gap"].abs()

        features["spx_upper_rejection"] = (
            high_price - close_price.combine(open_price, max)
        ) / features["spx_range"].replace(0, np.nan)
        features["spx_lower_rejection"] = (
            close_price.combine(open_price, min) - low_price
        ) / features["spx_range"].replace(0, np.nan)

        range_ma = features["spx_range"].rolling(21).mean()
        features["spx_range_expansion"] = features["spx_range"] / range_ma.replace(
            0, np.nan
        )

        # SPX-VIX Correlation
        for window in [21, 63, 126]:
            features[f"spx_vix_corr_{window}d"] = (
                spx.pct_change().rolling(window).corr(vix.pct_change())
            )

        # VIX vs Realized Vol
        for window in [10, 21]:
            if f"spx_realized_vol_{window}d" in features:
                rv = features[f"spx_realized_vol_{window}d"]
                features[f"vix_vs_rv_{window}d"] = vix - rv
                features[f"vix_rv_ratio_{window}d"] = vix / rv.replace(0, np.nan)

        return features

    def _build_cboe_features(self, cboe: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
        """Build CBOE-specific features."""
        features = pd.DataFrame(index=vix.index)

        # SKEW Features
        if "SKEW" in cboe.columns:
            skew = cboe["SKEW"]
            features["SKEW"] = skew

            # SKEW Regime
            features["skew_regime"] = calculate_regime_with_validation(
                skew,
                bins=[0, 130, 145, 160, 200],
                labels=[0, 1, 2, 3],
                feature_name="skew",
            )

            # SKEW vs VIX
            features["skew_vs_vix"] = skew - vix
            features["skew_vix_ratio"] = skew / vix.replace(0, np.nan)

            # SKEW Displacement
            skew_ma = skew.rolling(63).mean()
            features["skew_displacement"] = (
                (skew - skew_ma) / skew_ma.replace(0, np.nan)
            ) * 100

        # Put/Call Ratios - only equity institutional divergence and pcc acceleration
        if "PCCE" in cboe.columns and "PCCI" in cboe.columns:
            pcce = cboe["PCCE"]
            pcci = cboe["PCCI"]
            pcce_rank = pcce.rolling(63).rank(pct=True)
            pcci_rank = pcci.rolling(63).rank(pct=True)
            features["pc_equity_inst_divergence"] = (pcce_rank - pcci_rank).abs()

        if "PCC" in cboe.columns:
            pcc = cboe["PCC"]
            features["pcc_accel_10d"] = pcc.diff(10).diff(10)

        # Correlation Indices
        for cor_name in ["COR1M", "COR3M"]:
            if cor_name in cboe.columns:
                cor = cboe[cor_name]
                features[cor_name] = cor
                features[f"{cor_name}_change_21d"] = cor.diff(21)
                features[f"{cor_name}_zscore_63d"] = calculate_robust_zscore(cor, 63)

        if "COR1M" in cboe.columns and "COR3M" in cboe.columns:
            features["cor_term_structure"] = cboe["COR1M"] - cboe["COR3M"]
            features["cor_term_slope_change_21d"] = features["cor_term_structure"].diff(
                21
            )

        # VXTH Features
        if "VXTH" in cboe.columns:
            vxth = cboe["VXTH"]
            features["VXTH"] = vxth
            features["VXTH_change_21d"] = vxth.diff(21)
            features["VXTH_zscore_63d"] = calculate_robust_zscore(vxth, 63)
            features["vxth_vix_ratio"] = vxth / vix.replace(0, np.nan)

        # VXTLT Features (Bond Volatility)
        if "VXTLT" in cboe.columns:
            vxtlt = cboe["VXTLT"]
            features["VXTLT"] = vxtlt

            # Bond vol dynamics
            features["VXTLT_change_21d"] = vxtlt.diff(21)
            features["VXTLT_zscore_63d"] = calculate_robust_zscore(vxtlt, 63)
            features["VXTLT_velocity_10d"] = vxtlt.diff(10)
            features["VXTLT_acceleration_5d"] = vxtlt.diff(5).diff(5)

            # Bond vol regime
            features["bond_vol_regime"] = calculate_regime_with_validation(
                vxtlt,
                bins=[0, 5, 10, 15, 100],
                labels=[0, 1, 2, 3],
                feature_name="vxtlt",
            )

            # Bond-Equity vol relationship
            features["vxtlt_vix_ratio"] = vxtlt / vix.replace(0, np.nan)
            features["vxtlt_vix_spread"] = vxtlt - vix

            # Bond vol percentiles
            for window in [63, 126, 252]:
                features[f"VXTLT_percentile_{window}d"] = (
                    calculate_percentile_with_validation(vxtlt, window)
                )

            # Bond vol vs moving averages
            for window in [21, 63]:
                ma = vxtlt.rolling(window).mean()
                features[f"VXTLT_vs_ma{window}"] = (
                    (vxtlt - ma) / ma.replace(0, np.nan)
                ) * 100

            # Cross-asset vol divergence (bond vs equity)
            if "spx_realized_vol_21d" in features.columns:
                features["bond_equity_vol_divergence"] = (
                    vxtlt.rank(pct=True)
                    - features["spx_realized_vol_21d"].rank(pct=True)
                ).abs()

        # CBOE Stress Composite (updated to include VXTLT)
        stress_components = []
        if "SKEW" in features.columns:
            stress_components.append(((features["SKEW"] - 130) / 30).clip(0, 1))
        if "VXTH" in features.columns:
            stress_components.append(((features["VXTH"] - 15) / 20).clip(0, 1))
        if "VXTLT" in features.columns:
            stress_components.append(((features["VXTLT"] - 8) / 15).clip(0, 1))

        if stress_components:
            features["cboe_stress_composite"] = pd.DataFrame(stress_components).T.mean(
                axis=1
            )
            features["cboe_stress_regime"] = calculate_regime_with_validation(
                features["cboe_stress_composite"],
                bins=[0, 0.33, 0.66, 1],
                labels=[0, 1, 2],
                feature_name="cboe_stress",
            )

        return features

    def _build_futures_features(
        self,
        start_str: str,
        end_str: str,
        index: pd.DatetimeIndex,
        spx: pd.Series,
        cboe_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build futures-based features - using CBOE data instead of Yahoo futures."""

        # VIX Futures - Use CBOE data instead
        vx_data = {}

        # Check if CBOE has VX spreads already
        if cboe_data is not None and "VX1-VX2" in cboe_data.columns:
            vx_data["VX1-VX2"] = cboe_data["VX1-VX2"]
            print("   ✅ Using CBOE VX1-VX2 spread")

        if cboe_data is not None and "VX2-VX1_RATIO" in cboe_data.columns:
            vx_data["VX2-VX1_RATIO"] = cboe_data["VX2-VX1_RATIO"]
            print("   ✅ Using CBOE VX2-VX1 ratio")

        vx_features = self.futures_engine.extract_vix_futures_features(vx_data)
        if not vx_features.empty:
            vx_features = vx_features.reindex(index, method="ffill")
        else:
            vx_features = pd.DataFrame(index=index)

        # Commodity Futures - Use continuous contracts or CBOE
        commodity_data = {}

        # Check CBOE first
        if cboe_data is not None and "CL1-CL2" in cboe_data.columns:
            commodity_data["CL1-CL2"] = cboe_data["CL1-CL2"]
            print("   ✅ Using CBOE CL1-CL2 spread")

        # Try continuous crude contract
        crude = self.fetcher.fetch_yahoo("CL=F", start_str, end_str)
        if crude is not None:
            commodity_data["Crude_Oil"] = (
                crude["Close"].squeeze().reindex(index, method="ffill")
            )
            print("   ✅ Got Crude Oil continuous")

        commodity_features = self.futures_engine.extract_commodity_futures_features(
            commodity_data
        )
        if not commodity_features.empty:
            commodity_features = commodity_features.reindex(index, method="ffill")
        else:
            commodity_features = pd.DataFrame(index=index)

        # Dollar Futures - Use CBOE or spot index
        dollar_data = {}

        # Check CBOE first
        if cboe_data is not None and "DX1-DX2" in cboe_data.columns:
            dollar_data["DX1-DX2"] = cboe_data["DX1-DX2"]
            print("   ✅ Using CBOE DX1-DX2 spread")

        # Try Dollar Index spot
        dxy = self.fetcher.fetch_yahoo("DX-Y.NYB", start_str, end_str)
        if dxy is not None:
            dollar_data["Dollar_Index"] = (
                dxy["Close"].squeeze().reindex(index, method="ffill")
            )
            print("   ✅ Got Dollar Index spot")

        dollar_features = self.futures_engine.extract_dollar_futures_features(
            dollar_data
        )
        if not dollar_features.empty:
            dollar_features = dollar_features.reindex(index, method="ffill")
        else:
            dollar_features = pd.DataFrame(index=index)

        # Cross-Futures Relationships
        spx_ret = spx.pct_change(21) * 100
        cross_features = self.futures_engine.extract_futures_cross_relationships(
            vx_data, commodity_data, dollar_data, spx_ret
        )
        if not cross_features.empty:
            cross_features = cross_features.reindex(index, method="ffill")
        else:
            cross_features = pd.DataFrame(index=index)

        # Combine
        all_futures = pd.concat(
            [vx_features, commodity_features, dollar_features, cross_features], axis=1
        )
        print(f"   📊 Total futures features generated: {len(all_futures.columns)}")
        return all_futures

    def _fetch_macro_data(
        self, start_str: str, end_str: str, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Fetch macro data from FRED and Yahoo Finance - WITH TEMPORAL SAFETY."""

        fetched_data = {}

        # FRED series (temporal safety handled in data_fetcher)
        fred_series = {
            "Dollar_Index": "DTWEXBGS",
            "CPI": "CPIAUCSL",
        }

        for name, series_id in fred_series.items():
            try:
                print(f"   Fetching {name} ({series_id})...", end=" ")
                data = self.fetcher.fetch_fred_series(series_id, start_str, end_str)
                if data is not None and not data.empty:
                    # Limited forward-fill to prevent excessive propagation
                    fetched_data[name] = data.reindex(index, method="ffill", limit=5)
                    print(f"✅ {len(data)} rows")
                else:
                    print(f"❌ No data")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

        # Yahoo Finance series (temporal safety handled in data_fetcher)
        yahoo_series = {
            "Gold": "GC=F",  # Gold Futures
        }

        for name, symbol in yahoo_series.items():
            try:
                print(f"   Fetching {name} ({symbol})...", end=" ")
                data = self.fetcher.fetch_yahoo(symbol, start_str, end_str)
                if data is not None and not data.empty:
                    if "Close" in data.columns:
                        # Limited forward-fill
                        fetched_data[name] = data["Close"].reindex(
                            index, method="ffill", limit=5
                        )
                        print(f"✅ {len(data)} rows")
                    else:
                        print(f"❌ No Close column")
                else:
                    print(f"❌ No data")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

        if fetched_data:
            macro_df = pd.DataFrame(fetched_data, index=index)
            return macro_df

        print("   ⚠️  No macro data available")
        return None

    def _build_macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Build macro features."""
        features = pd.DataFrame(index=macro.index)

        # Dollar Index
        if "Dollar_Index" in macro.columns:
            dxy = macro["Dollar_Index"]
            features["Dollar_Index_lag1"] = dxy.shift(1)
            features["Dollar_Index_zscore_63d"] = calculate_robust_zscore(dxy, 63)

            # Dollar returns (kept here, removed from futures to avoid duplication)
            for window in [10, 21, 63]:
                features[f"dxy_ret_{window}d"] = dxy.pct_change(window) * 100

        # Bond Volatility
        if "Bond_Vol" in macro.columns:
            bv = macro["Bond_Vol"]
            features["Bond_Vol_lag1"] = bv.shift(1)
            features["Bond_Vol_zscore_63d"] = calculate_robust_zscore(bv, 63)

            for window in [10, 21, 63]:
                features[f"Bond_Vol_mom_{window}d"] = bv.diff(window)

        # CPI (keep, PCE removed)
        if "CPI" in macro.columns:
            cpi = macro["CPI"]
            for window in [10, 21, 63]:
                features[f"CPI_change_{window}d"] = cpi.diff(window)

        return features

    def _build_treasury_features(
        self, start_str: str, end_str: str, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build treasury yield features - WITH TEMPORAL SAFETY."""

        treasury_series = {
            "DGS1MO": "DGS1MO",
            "DGS3MO": "DGS3MO",
            "DGS6MO": "DGS6MO",
            "DGS1": "DGS1",
            "DGS2": "DGS2",
            "DGS5": "DGS5",
            "DGS10": "DGS10",
            "DGS30": "DGS30",
        }

        fetched_yields = {}
        for name, series_id in treasury_series.items():
            try:
                print(f"   Fetching {name}...", end=" ")
                data = self.fetcher.fetch_fred_series(series_id, start_str, end_str)
                if data is not None and not data.empty:
                    # Limited forward-fill
                    fetched_yields[name] = data.reindex(index, method="ffill", limit=5)
                    print(f"✅ {len(data)} rows")
                else:
                    print(f"❌ No data")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

        if not fetched_yields:
            print("   ⚠️  No treasury data available")
            return pd.DataFrame(index=index)

        print(f"   📊 Got {len(fetched_yields)} yield series")
        yields_df = pd.DataFrame(fetched_yields, index=index)

        # Extract treasury features
        term_spreads = self.treasury_engine.extract_term_spreads(yields_df)
        curve_shape = self.treasury_engine.extract_curve_shape(yields_df)
        rate_vol = self.treasury_engine.extract_rate_volatility(yields_df)

        treasury_features = pd.concat([term_spreads, curve_shape, rate_vol], axis=1)
        print(f"   ✅ Generated {len(treasury_features.columns)} treasury features")
        return treasury_features

    def _build_meta_features(
        self,
        combined_base: pd.DataFrame,
        spx: pd.Series,
        vix: pd.Series,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build meta features from combined base features."""

        regime_features = self.meta_engine.extract_regime_indicators(
            combined_base, vix, spx
        )
        cross_asset = self.meta_engine.extract_cross_asset_relationships(
            combined_base, macro_df
        )
        roc_features = self.meta_engine.extract_rate_of_change_features(combined_base)
        percentile_features = self.meta_engine.extract_percentile_rankings(
            combined_base
        )

        meta_features = pd.concat(
            [regime_features, cross_asset, roc_features, percentile_features], axis=1
        )

        return meta_features

    def _build_calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Build calendar features."""
        features = pd.DataFrame(index=index)

        features["month"] = index.month
        features["day_of_week"] = index.dayofweek
        features["day_of_month"] = index.day

        return features

    def _ensure_numeric_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all numeric-looking columns to float64.

        This is critical because:
        1. Pandas concat() can create object dtypes from mixed numeric types
        2. XGBoost requires numeric dtypes
        3. Single-row extraction from object-dtype DataFrame returns strings

        Args:
            df: DataFrame that may have object dtypes

        Returns:
            DataFrame with numeric columns converted to float64
        """
        # Identify columns that should be numeric (exclude known categorical)
        metadata_cols = ["calendar_cohort", "cohort_weight", "feature_quality"]
        numeric_cols = [col for col in df.columns if col not in metadata_cols]

        print(f"🔧 Converting {len(numeric_cols)} columns to numeric dtypes...")

        conversion_stats = {"converted": 0, "failed": 0, "already_numeric": 0}

        for col in numeric_cols:
            if df[col].dtype == object:
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    conversion_stats["converted"] += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to convert {col}: {e}")
                    conversion_stats["failed"] += 1
            else:
                conversion_stats["already_numeric"] += 1

        # Ensure all numeric columns are float64
        for col in numeric_cols:
            if col in df.columns and df[col].dtype in [np.int32, np.int64, np.float32]:
                df[col] = df[col].astype(np.float64)

        print(f"   ✅ Converted: {conversion_stats['converted']}")
        print(f"   ℹ️  Already numeric: {conversion_stats['already_numeric']}")
        if conversion_stats["failed"] > 0:
            print(f"   ❌ Failed: {conversion_stats['failed']}")

        return df
