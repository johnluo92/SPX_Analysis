import pandas as pd
from datetime import timedelta
from config import CALENDAR_COHORTS, COHORT_PRIORITY, MACRO_EVENT_CONFIG


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
    
    def _days_to_fomc(self, date):
        """Calculate days to next FOMC meeting."""
        if self.fomc_calendar is None or len(self.fomc_calendar) == 0:
            return None
        
        ff = self.fomc_calendar[self.fomc_calendar.index >= date]
        if len(ff) == 0:
            return None
        
        return -(ff.index[0] - date).days
    
    def _days_to_vix_futures_expiry(self, date):
        """Calculate days to next VIX futures expiry."""
        if self.vix_futures_expiry is None or len(self.vix_futures_expiry) == 0:
            return None
        
        fe = self.vix_futures_expiry[self.vix_futures_expiry.index >= date]
        if len(fe) == 0:
            return None
        
        return -(fe.index[0] - date).days
    
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
