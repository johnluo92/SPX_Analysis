"""Core data models for earnings analysis

These dataclasses provide clean contracts for data flowing through the system.
They replace scattered dictionaries with typed, validated structures.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime


@dataclass
class EarningsEvent:
    """Single earnings announcement"""
    date: datetime
    time: str  # 'bmo' or 'amc'
    
    def __post_init__(self):
        """Validate timing"""
        if self.time not in ('bmo', 'amc'):
            raise ValueError(f"Invalid timing: {self.time}. Must be 'bmo' or 'amc'")


@dataclass
class HistoricalDataPoint:
    """Single historical earnings + price movement"""
    earnings_date: str
    ref_date: Optional[str]
    move: float          # % move from entry
    width: float         # Strike width used
    hvol: float          # HVol at time of earnings
    
    @property
    def contained(self) -> bool:
        """Check if move stayed within width"""
        return abs(self.move) <= self.width
    
    @property
    def break_direction(self) -> Optional[str]:
        """Return 'up', 'down', or None if contained"""
        if self.contained:
            return None
        return 'up' if self.move > self.width else 'down'


@dataclass
class TimeframeStats:
    """Statistics for a specific timeframe (45d or 90d)"""
    total: int
    containment: float        # % contained
    breaks_up: int
    breaks_down: int
    trend_pct: float         # % moves that closed above entry
    break_up_pct: float      # % of breaks that went up
    drift_pct: float         # Average move as %
    drift_vs_width: float    # Drift as % of width
    avg_width: float         # Average width used
    
    def __post_init__(self):
        """Validate ranges"""
        if not 0 <= self.containment <= 100:
            raise ValueError(f"Containment must be 0-100%, got {self.containment}")
        if not 0 <= self.trend_pct <= 100:
            raise ValueError(f"Trend % must be 0-100%, got {self.trend_pct}")
        if not 0 <= self.break_up_pct <= 100:
            raise ValueError(f"Break up % must be 0-100%, got {self.break_up_pct}")


@dataclass
class IVData:
    """Current implied volatility snapshot"""
    iv: float                # Current IV as %
    dte: int                 # Days to expiration
    expiration: str          # Expiration date string
    fetched_at: str          # ISO timestamp when fetched
    
    def __post_init__(self):
        """Validate IV"""
        if self.iv < 0 or self.iv > 500:
            raise ValueError(f"IV out of reasonable range: {self.iv}%")
        if self.dte < 0:
            raise ValueError(f"DTE cannot be negative: {self.dte}")


@dataclass
class AnalysisResult:
    """Complete analysis result for a single ticker
    
    This is the primary data structure returned by analyze_ticker()
    and used throughout the system for display, export, and visualization.
    """
    ticker: str
    hvol: float                          # Average HVol across history
    strike_width: float                  # Average 90d width
    rvol_45d: Optional[float]           # Realized vol (45d moves)
    
    # 45-day stats
    stats_45: TimeframeStats
    
    # 90-day stats  
    stats_90: TimeframeStats
    
    # Optional IV enrichment
    current_iv: Optional[float] = None
    iv_dte: Optional[int] = None
    iv_elevation: Optional[float] = None  # IV vs RVol45d
    
    # Historical detail (for audit trail)
    earnings_history: List[Dict] = field(default_factory=list)
    
    # Cached strategy calculations
    _strategy_45: Optional[Tuple[str, int]] = field(default=None, repr=False)
    _strategy_90: Optional[Tuple[str, int]] = field(default=None, repr=False)
    _combined_strategy: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate ticker and percentages"""
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError(f"Invalid ticker: {self.ticker}")
        if self.hvol < 0 or self.hvol > 500:
            raise ValueError(f"HVol out of range: {self.hvol}%")
    
    @property
    def strategy_45(self) -> Tuple[str, int]:
        """Get 45-day strategy (cached)"""
        if self._strategy_45 is None:
            from ..calculations.strategy import determine_strategy_45
            self._strategy_45 = determine_strategy_45({
                'containment': self.stats_45.containment,
                'breaks_up': self.stats_45.breaks_up,
                'breaks_down': self.stats_45.breaks_down,
                'break_up_pct': self.stats_45.break_up_pct,
                'trend_pct': self.stats_45.trend_pct,
                'drift_pct': self.stats_45.drift_pct
            })
        return self._strategy_45
    
    @property
    def strategy_90(self) -> Tuple[str, int]:
        """Get 90-day strategy (cached)"""
        if self._strategy_90 is None:
            from ..calculations.strategy import determine_strategy_90
            self._strategy_90 = determine_strategy_90({
                'containment': self.stats_90.containment,
                'breaks_up': self.stats_90.breaks_up,
                'breaks_down': self.stats_90.breaks_down,
                'break_up_pct': self.stats_90.break_up_pct,
                'trend_pct': self.stats_90.trend_pct,
                'drift_pct': self.stats_90.drift_pct
            })
        return self._strategy_90
    
    @property
    def combined_strategy(self) -> str:
        """Get combined strategy recommendation (cached)"""
        if self._combined_strategy is None:
            from ..calculations.strategy import determine_strategy
            self._combined_strategy = determine_strategy(
                {
                    'containment': self.stats_45.containment,
                    'breaks_up': self.stats_45.breaks_up,
                    'breaks_down': self.stats_45.breaks_down,
                    'break_up_pct': self.stats_45.break_up_pct,
                    'trend_pct': self.stats_45.trend_pct,
                    'drift_pct': self.stats_45.drift_pct
                },
                {
                    'containment': self.stats_90.containment,
                    'breaks_up': self.stats_90.breaks_up,
                    'breaks_down': self.stats_90.breaks_down,
                    'break_up_pct': self.stats_90.break_up_pct,
                    'trend_pct': self.stats_90.trend_pct,
                    'drift_pct': self.stats_90.drift_pct
                }
            )
        return self._combined_strategy
    
    @property
    def has_iv_data(self) -> bool:
        """Check if IV data is available"""
        return self.current_iv is not None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for backward compatibility
        
        This allows gradual migration - old code expecting dicts will still work.
        """
        result = {
            'ticker': self.ticker,
            'hvol': round(self.hvol, 1),
            'strike_width': round(self.strike_width, 1),
            'rvol_45d': round(self.rvol_45d, 1) if self.rvol_45d is not None else None,
            
            # 45d stats
            '45d_contain': round(self.stats_45.containment, 0),
            '45d_breaks_up': self.stats_45.breaks_up,
            '45d_breaks_dn': self.stats_45.breaks_down,
            '45d_trend_pct': round(self.stats_45.trend_pct, 0),
            '45d_break_up_pct': round(self.stats_45.break_up_pct, 0),
            '45d_drift': round(self.stats_45.drift_pct, 1),
            
            # 90d stats
            '90d_contain': round(self.stats_90.containment, 0),
            '90d_breaks_up': self.stats_90.breaks_up,
            '90d_breaks_dn': self.stats_90.breaks_down,
            '90d_trend_pct': round(self.stats_90.trend_pct, 0),
            '90d_break_up_pct': round(self.stats_90.break_up_pct, 0),
            '90d_drift': round(self.stats_90.drift_pct, 1),
            
            # Strategy
            'strategy': self.combined_strategy,
            
            # History
            'earnings_history': self.earnings_history
        }
        
        # Add IV data if available
        if self.has_iv_data:
            result['current_iv'] = self.current_iv
            result['iv_dte'] = self.iv_dte
            result['iv_elevation'] = self.iv_elevation
        else:
            result['current_iv'] = None
            result['iv_dte'] = None
            result['iv_elevation'] = None
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisResult':
        """Create from dictionary (for backward compatibility)"""
        return cls(
            ticker=data['ticker'],
            hvol=data['hvol'],
            strike_width=data['strike_width'],
            rvol_45d=data.get('rvol_45d'),
            stats_45=TimeframeStats(
                total=data.get('45d_total', 24),  # Default if missing
                containment=data['45d_contain'],
                breaks_up=data['45d_breaks_up'],
                breaks_down=data['45d_breaks_dn'],
                trend_pct=data['45d_trend_pct'],
                break_up_pct=data['45d_break_up_pct'],
                drift_pct=data['45d_drift'],
                drift_vs_width=0.0,  # Can calculate if needed
                avg_width=0.0  # Can calculate if needed
            ),
            stats_90=TimeframeStats(
                total=data.get('90d_total', 24),
                containment=data['90d_contain'],
                breaks_up=data['90d_breaks_up'],
                breaks_down=data['90d_breaks_dn'],
                trend_pct=data['90d_trend_pct'],
                break_up_pct=data['90d_break_up_pct'],
                drift_pct=data['90d_drift'],
                drift_vs_width=0.0,
                avg_width=0.0
            ),
            current_iv=data.get('current_iv'),
            iv_dte=data.get('iv_dte'),
            iv_elevation=data.get('iv_elevation'),
            earnings_history=data.get('earnings_history', [])
        )


# Convenience function for batch results
def results_to_dataframe(results: List[AnalysisResult]):
    """Convert list of AnalysisResults to DataFrame
    
    This maintains backward compatibility while allowing
    use of typed AnalysisResult objects internally.
    """
    import pandas as pd
    return pd.DataFrame([r.to_dict() for r in results])