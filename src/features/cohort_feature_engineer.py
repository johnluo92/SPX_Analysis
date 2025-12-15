import pandas as pd
import numpy as np
from datetime import timedelta
from config import CALENDAR_COHORTS, COHORT_PRIORITY, MACRO_EVENT_CONFIG


# INTEGRATION INSTRUCTIONS:
# Add these lines to feature_engineer.py after the build_cohort_features() call:
#
#   proximity_features = self.cohort_engineer.build_cohort_proximity_features(af.index)
#   af = pd.concat([af, proximity_features], axis=1)
#
#   interaction_features = self.cohort_engineer.build_cohort_interaction_features(af.index, af)
#   af = pd.concat([af, interaction_features], axis=1)


class CohortFeatureEngineer:
    """Handles all cohort-related feature engineering including FOMC, OPEX, and earnings calendar logic."""

    def __init__(self, data_fetcher):
        self.fetcher = data_fetcher
        self.fomc_calendar = None
        self.opex_calendar = None
        self.earnings_calendar = None
        self.vix_futures_expiry = None
        self._cohort_cache = {}
        self.training_start_date = None
        self.training_end_date = None

    def set_date_range(self, start_date, end_date):
        """Set the training date range for calendar generation."""
        self.training_start_date = start_date
        self.training_end_date = end_date

    def _load_calendar_data(self):
        """Load FOMC, OPEX, VIX futures expiry, and earnings calendars."""
        if self.fomc_calendar is None:
            try:
                sy, ey = self.training_start_date.year, self.training_end_date.year + 1
                self.fomc_calendar = self.fetcher.fetch_fomc_calendar(start_year=sy, end_year=ey)
            except Exception as e:
                print(f"⚠️ FOMC calendar unavailable, using stub")
                self.fomc_calendar = pd.DataFrame()

        if self.opex_calendar is None:
            self.opex_calendar = self._generate_opex_calendar()

        if self.vix_futures_expiry is None:
            self.vix_futures_expiry = self._generate_vix_futures_expiry()

        if self.earnings_calendar is None:
            self.earnings_calendar = pd.DataFrame()

    def _generate_opex_calendar(self, sy=None, ey=None):
        """Generate monthly OPEX calendar."""
        if sy is None:
            sy = self.training_start_date.year
        if ey is None:
            ey = self.training_end_date.year + 1

        od = []
        for yr in range(sy, ey + 1):
            for mo in range(1, 13):
                fp = pd.Timestamp(yr, mo, 15)
                da = (4 - fp.weekday()) % 7
                if da == 0 and fp.day > 15:
                    da = 7
                tf = fp + pd.Timedelta(days=da)
                od.append({"date": tf, "expiry_type": "monthly_opex"})

        return pd.DataFrame(od).set_index("date").sort_index()

    def _generate_vix_futures_expiry(self):
        """Generate VIX futures expiry calendar."""
        if self.opex_calendar is None:
            self._generate_opex_calendar()

        ve = []
        for od in self.opex_calendar.index:
            ad = od - pd.Timedelta(days=30)
            dtw = (2 - ad.weekday()) % 7
            vd = ad + pd.Timedelta(days=dtw)
            ve.append({"date": vd, "expiry_type": "vix_futures"})

        return pd.DataFrame(ve).set_index("date").sort_index()

    def get_calendar_cohort(self, date):
        """Determine calendar cohort for a given date."""
        date = pd.Timestamp(date)

        if date in self._cohort_cache:
            return self._cohort_cache[date]

        if self.opex_calendar is None:
            self._load_calendar_data()

        dto = self._days_to_monthly_opex(date)
        dtf = self._days_to_fomc(date)
        dtve = self._days_to_vix_futures_expiry(date)
        ep = self._spx_earnings_intensity(date)
        is_cpi = self._is_cpi_release_day(date)
        is_pce = self._is_pce_release_day(date)
        is_fomc_minutes = self._is_fomc_minutes_day(date)

        for cn in COHORT_PRIORITY:
            cd = CALENDAR_COHORTS[cn]
            cond = cd["condition"]

            if cond == "macro_event_period":
                if dtf is not None:
                    rmin, rmax = cd["range"]
                    if rmin <= dtf <= rmax:
                        res = (cn, cd["weight"])
                        self._cohort_cache[date] = res
                        return res

                if is_cpi or is_pce or is_fomc_minutes:
                    res = (cn, cd["weight"])
                    self._cohort_cache[date] = res
                    return res

            elif cond == "days_to_monthly_opex":
                if dto is not None or dtve is not None:
                    rmin, rmax = cd["range"]
                    if (dto is not None and rmin <= dto <= rmax) or (dtve is not None and rmin <= dtve <= rmax):
                        res = (cn, cd["weight"])
                        self._cohort_cache[date] = res
                        return res

            elif cond == "spx_earnings_pct":
                if ep is not None:
                    rmin, rmax = cd["range"]
                    if rmin <= ep <= rmax:
                        res = (cn, cd["weight"])
                        self._cohort_cache[date] = res
                        return res

            elif cond == "default":
                res = (cn, cd["weight"])
                self._cohort_cache[date] = res
                return res

        raise ValueError(f"No cohort matched for date {date}")

    def _days_to_monthly_opex(self, date):
        """Calculate days to next monthly OPEX."""
        if self.opex_calendar is None or len(self.opex_calendar) == 0:
            return None

        fo = self.opex_calendar[self.opex_calendar.index >= date]
        if len(fo) == 0:
            return None

        nxt = fo.index[0]
        dd = (nxt - date).days

        if dd == 0:
            return 0

        return -dd if dd > 0 else dd

    def _days_to_vix_futures_expiry(self, date):
        """Calculate days to next VIX futures expiry."""
        if self.vix_futures_expiry is None or len(self.vix_futures_expiry) == 0:
            return None

        fe = self.vix_futures_expiry[self.vix_futures_expiry.index >= date]
        if len(fe) == 0:
            return None

        return -(fe.index[0] - date).days

    def _days_to_fomc(self, date):
        """Calculate days to next FOMC meeting."""
        if self.fomc_calendar is None or len(self.fomc_calendar) == 0:
            return None

        ff = self.fomc_calendar[self.fomc_calendar.index >= date]
        if len(ff) == 0:
            return None

        return -(ff.index[0] - date).days

    def _spx_earnings_intensity(self, date):
        """Calculate SPX earnings intensity for a given date."""
        mo = date.month

        if mo in [1, 4, 7, 10]:
            wom = (date.day - 1) // 7 + 1
            if wom in [2, 3, 4]:
                return 0.25

        return 0.05

    def _is_cpi_release_day(self, date):
        """Check if date is a CPI release day."""
        target = MACRO_EVENT_CONFIG["cpi_release"]["day_of_month_target"]
        window = MACRO_EVENT_CONFIG["cpi_release"]["window_days"]
        return abs(date.day - target) <= window

    def _is_pce_release_day(self, date):
        """Check if date is a PCE release day."""
        target = MACRO_EVENT_CONFIG["pce_release"]["day_of_month_target"]
        window = MACRO_EVENT_CONFIG["pce_release"]["window_days"]
        return abs(date.day - target) <= window

    def _is_fomc_minutes_day(self, date):
        """Check if date is an FOMC minutes release day."""
        if self.fomc_calendar is None or len(self.fomc_calendar) == 0:
            return False

        days_after = MACRO_EVENT_CONFIG["fomc_minutes"]["days_after_meeting"]
        window = MACRO_EVENT_CONFIG["fomc_minutes"]["window_days"]

        for fomc_date in self.fomc_calendar.index:
            minutes_date = fomc_date + pd.Timedelta(days=days_after)
            if abs((date - minutes_date).days) <= window:
                return True

        return False

    def build_cohort_features(self, index):
        """Build cohort features for a given DatetimeIndex."""
        self._load_calendar_data()

        cohd = [
            {
                "calendar_cohort": self.get_calendar_cohort(dt)[0],
                "cohort_weight": self.get_calendar_cohort(dt)[1]
            }
            for dt in index
        ]

        cohdf = pd.DataFrame(cohd, index=index)
        cohdf["is_fomc_period"] = (cohdf["calendar_cohort"] == "fomc_period").astype(int)
        cohdf["is_opex_week"] = (cohdf["calendar_cohort"] == "opex_week").astype(int)
        cohdf["is_earnings_heavy"] = (cohdf["calendar_cohort"] == "earnings_heavy").astype(int)

        return cohdf

    def build_cohort_proximity_features(self, index):
        """Build continuous proximity features for cohort events.

        These are continuous rather than binary features that allow models to learn
        the natural relationship between event proximity and market behavior.
        """
        self._load_calendar_data()

        features = pd.DataFrame(index=index)

        # FOMC proximity features (continuous, 0-1 scale)
        fomc_days_raw = [self._days_to_fomc(dt) for dt in index]
        fomc_days = pd.Series(fomc_days_raw, index=index, dtype=float)  # Force float to convert None to NaN

        fomc_days_abs = fomc_days.abs()
        features["fomc_proximity"] = np.where(
            fomc_days.isna(),
            0.0,
            np.clip(1.0 - (fomc_days_abs / 14.0), 0.0, 1.0)  # 2-week decay window
        )

        # FOMC pre/post periods (continuous intensity)
        features["fomc_pre_period"] = np.where(
            fomc_days.isna() | (fomc_days >= 0),
            0.0,
            np.clip(1.0 + (fomc_days / 7.0), 0.0, 1.0)  # 7 days before
        )

        features["fomc_post_period"] = np.where(
            fomc_days.isna() | (fomc_days < 0),
            0.0,
            np.clip(1.0 - (fomc_days / 3.0), 0.0, 1.0)  # 3 days after
        )

        # OPEX proximity features (continuous)
        opex_days_raw = [self._days_to_monthly_opex(dt) for dt in index]
        opex_days = pd.Series(opex_days_raw, index=index, dtype=float)

        opex_days_abs = opex_days.abs()
        features["opex_proximity"] = np.where(
            opex_days.isna(),
            0.0,
            np.clip(1.0 - (opex_days_abs / 10.0), 0.0, 1.0)  # 10-day decay
        )

        features["opex_week_intensity"] = np.where(
            opex_days.isna() | (opex_days < -7) | (opex_days > 0),
            0.0,
            np.clip(1.0 + (opex_days / 7.0), 0.0, 1.0)  # Week before OPEX
        )

        # VIX futures expiry proximity (continuous)
        vix_exp_days_raw = [self._days_to_vix_futures_expiry(dt) for dt in index]
        vix_exp_days = pd.Series(vix_exp_days_raw, index=index, dtype=float)

        vix_exp_days_abs = vix_exp_days.abs()
        features["vix_expiry_proximity"] = np.where(
            vix_exp_days.isna(),
            0.0,
            np.clip(1.0 - (vix_exp_days_abs / 10.0), 0.0, 1.0)
        )

        # Earnings intensity (already continuous from 0.05 to 0.25)
        features["earnings_intensity"] = pd.Series(
            [self._spx_earnings_intensity(dt) for dt in index],
            index=index
        )

        # Normalize earnings intensity to 0-1 scale for interactions
        features["earnings_intensity_norm"] = (features["earnings_intensity"] - 0.05) / 0.20

        # Macro event proximity features
        features["cpi_week"] = pd.Series(
            [1.0 if self._is_cpi_release_day(dt) else 0.0 for dt in index],
            index=index
        )

        features["pce_week"] = pd.Series(
            [1.0 if self._is_pce_release_day(dt) else 0.0 for dt in index],
            index=index
        )

        # Macro event intensity (any macro event nearby)
        features["macro_event_intensity"] = np.clip(
            features["fomc_proximity"] +
            features["cpi_week"] * 0.5 +
            features["pce_week"] * 0.5,
            0.0, 1.0
        )

        # Multi-event stress (multiple events converging)
        features["multi_event_stress"] = (
            (features["fomc_proximity"] > 0.3).astype(float) +
            (features["opex_proximity"] > 0.3).astype(float) +
            (features["earnings_intensity_norm"] > 0.5).astype(float)
        ) / 3.0

        # Final safety: fill any remaining NaN with 0.0
        features = features.fillna(0.0)

        return features

    def build_cohort_interaction_features(self, index, market_features):
        """Build interaction features between cohort timing and market conditions.

        Args:
            index: DatetimeIndex for the features
            market_features: DataFrame with existing market features (vix, spx, etc.)

        Returns:
            DataFrame with cohort × market interaction features
        """
        # Get proximity features first
        proximity = self.build_cohort_proximity_features(index)

        interactions = pd.DataFrame(index=index)

        # VIX velocity × cohort timing interactions
        if "vix_velocity_5d" in market_features.columns:
            vv = market_features["vix_velocity_5d"]

            interactions["vix_vel_x_fomc_prox"] = vv * proximity["fomc_proximity"]
            interactions["vix_vel_x_fomc_pre"] = vv * proximity["fomc_pre_period"]
            interactions["vix_vel_x_fomc_post"] = vv * proximity["fomc_post_period"]
            interactions["vix_vel_x_opex_prox"] = vv * proximity["opex_proximity"]
            interactions["vix_vel_x_earnings"] = vv * proximity["earnings_intensity_norm"]
            interactions["vix_vel_x_macro_event"] = vv * proximity["macro_event_intensity"]

            # Absolute velocity for stress detection
            vv_abs = vv.abs()
            interactions["vix_vel_abs_x_multi_event"] = vv_abs * proximity["multi_event_stress"]

        # VIX level × cohort interactions
        if "vix" in market_features.columns:
            vix = market_features["vix"]

            interactions["vix_x_fomc_prox"] = vix * proximity["fomc_proximity"]
            interactions["vix_x_opex_prox"] = vix * proximity["opex_proximity"]
            interactions["vix_x_earnings"] = vix * proximity["earnings_intensity_norm"]

        # Term structure × OPEX proximity (important for VIX roll dynamics)
        if "VX1-VX2" in market_features.columns:
            ts = market_features["VX1-VX2"]

            interactions["vx_spread_x_opex_prox"] = ts * proximity["opex_proximity"]
            interactions["vx_spread_x_opex_week"] = ts * proximity["opex_week_intensity"]
            interactions["vx_spread_x_vix_expiry"] = ts * proximity["vix_expiry_proximity"]

        # SPX gap × post-FOMC (news reactions)
        if "spx_gap" in market_features.columns:
            gap = market_features["spx_gap"]

            interactions["spx_gap_x_fomc_post"] = gap * proximity["fomc_post_period"]
            interactions["spx_gap_x_macro_event"] = gap * proximity["macro_event_intensity"]

        # Realized vol × earnings (earnings vol tends to cluster)
        if "spx_realized_vol_21d" in market_features.columns:
            rv = market_features["spx_realized_vol_21d"]

            interactions["spx_rv_x_earnings"] = rv * proximity["earnings_intensity_norm"]
            interactions["spx_rv_x_fomc_prox"] = rv * proximity["fomc_proximity"]

        # VIX-RV spread × cohort timing (risk premium dynamics)
        if all(f in market_features.columns for f in ["vix", "spx_realized_vol_21d"]):
            rp = market_features["vix"] - market_features["spx_realized_vol_21d"]

            interactions["risk_premium_x_fomc_pre"] = rp * proximity["fomc_pre_period"]
            interactions["risk_premium_x_fomc_post"] = rp * proximity["fomc_post_period"]
            interactions["risk_premium_x_opex"] = rp * proximity["opex_proximity"]

        # SKEW × FOMC (tail risk around events)
        if "SKEW" in market_features.columns:
            skew = market_features["SKEW"]

            interactions["skew_x_fomc_prox"] = skew * proximity["fomc_proximity"]
            interactions["skew_x_fomc_pre"] = skew * proximity["fomc_pre_period"]
            interactions["skew_x_macro_event"] = skew * proximity["macro_event_intensity"]

        # Momentum × earnings intensity (earnings can amplify trends)
        if "spx_ret_5d" in market_features.columns:
            mom = market_features["spx_ret_5d"]

            interactions["spx_mom_x_earnings"] = mom * proximity["earnings_intensity_norm"]
            interactions["spx_mom_x_fomc_post"] = mom * proximity["fomc_post_period"]

        # VIX acceleration × event proximity (rapid changes around events)
        if "vix_accel_5d" in market_features.columns:
            accel = market_features["vix_accel_5d"]

            interactions["vix_accel_x_fomc_prox"] = accel * proximity["fomc_proximity"]
            interactions["vix_accel_x_opex_prox"] = accel * proximity["opex_proximity"]

        # Vol-of-vol × OPEX (VIX vol tends to spike at expiry)
        if "vix_vol_21d" in market_features.columns:
            vv = market_features["vix_vol_21d"]

            interactions["vix_vol_x_opex_week"] = vv * proximity["opex_week_intensity"]
            interactions["vix_vol_x_vix_expiry"] = vv * proximity["vix_expiry_proximity"]

        # Credit stress × macro events
        if "hy_spread" in market_features.columns:
            hy = market_features["hy_spread"]

            interactions["hy_spread_x_fomc_prox"] = hy * proximity["fomc_proximity"]
            interactions["hy_spread_x_macro_event"] = hy * proximity["macro_event_intensity"]

        # Regime × cohort interactions (different regimes behave differently around events)
        if "vix_regime" in market_features.columns:
            regime = market_features["vix_regime"]

            interactions["regime_x_fomc_prox"] = regime * proximity["fomc_proximity"]
            interactions["regime_x_opex_prox"] = regime * proximity["opex_proximity"]
            interactions["regime_x_earnings"] = regime * proximity["earnings_intensity_norm"]

        # Final safety: replace any NaN or inf with 0.0
        interactions = interactions.replace([np.inf, -np.inf], 0.0)
        interactions = interactions.fillna(0.0)

        return interactions

    def validate_cohort_features(self, features_df, feature_type="cohort"):
        """Validate cohort features for data quality issues.

        Args:
            features_df: DataFrame of features to validate
            feature_type: Type of features for error messages

        Returns:
            Validated DataFrame with issues fixed
        """
        if features_df.empty:
            return features_df

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns

        # Check for NaN
        nan_counts = features_df[numeric_cols].isna().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  {feature_type} features: {nan_counts.sum()} NaN values detected")
            for col in nan_counts[nan_counts > 0].index:
                print(f"     {col}: {nan_counts[col]} NaN values")
            features_df[numeric_cols] = features_df[numeric_cols].fillna(0.0)

        # Check for inf
        inf_mask = np.isinf(features_df[numeric_cols].values)
        if inf_mask.sum() > 0:
            print(f"⚠️  {feature_type} features: {inf_mask.sum()} Inf values detected")
            features_df = features_df.replace([np.inf, -np.inf], 0.0)

        # Check for extreme values (beyond reasonable range)
        for col in numeric_cols:
            col_abs_max = features_df[col].abs().max()
            if col_abs_max > 1e6:
                print(f"⚠️  {feature_type} feature {col} has extreme values (max={col_abs_max:.2e})")
                features_df[col] = features_df[col].clip(-1e6, 1e6)

        return features_df
