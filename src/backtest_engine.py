"""
VIX-based SPX options backtesting engine - PROFESSIONAL VERSION.

Core philosophy:
- No predictions, just probabilities from historical data
- VIX regime determines structure and sizing
- Mechanical entry/exit rules with PROFIT TARGETS
- Track what actually happened, not what "should" happen

PROFESSIONAL IMPROVEMENTS:
- 14 DTE (not 3) for optimal theta decay
- 50% profit targets (don't hold to expiration)
- Wider spreads ($25) for better credit collection
- Trend filters to avoid selling into crashes
- More realistic credit estimation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Import our modules
from UnifiedDataFetcher import UnifiedDataFetcher
from config_PROFESSIONAL import (
    DATA_CONFIG, VIX_REGIMES, POSITION_CONFIG, ENTRY_RULES, 
    EXIT_RULES, RISK_CONFIG, get_vix_regime, get_trade_structure
)

@dataclass
class Trade:
    """Represents a single options trade."""
    entry_date: datetime
    expiration_date: datetime
    entry_vix: float
    entry_spx: float
    regime: str
    structure: str
    
    # Strike prices
    short_strike: float
    long_strike: float  # For spreads
    
    # Trade details
    credit_received: float
    max_loss: float
    dte: int
    
    # Exit tracking
    exit_date: Optional[datetime] = None
    exit_spx: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # Status
    is_open: bool = True
    unrealized_pnl: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.days_held = 0
        self.max_loss_pct = self.max_loss / abs(self.credit_received) if self.credit_received != 0 else 0


class BacktestEngine:
    """Core backtesting engine for VIX-based SPX options strategies - PROFESSIONAL VERSION."""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize the backtest engine.
        
        Args:
            config_override: Optional dict to override default config values
        """
        self.fetcher = UnifiedDataFetcher()
        self.config = self._merge_config(config_override)
        
        # State tracking
        self.trades: List[Trade] = []
        self.open_positions: List[Trade] = []
        self.daily_pnl: pd.Series = None
        self.equity_curve: pd.Series = None
        
        # Data storage
        self.spx_data: pd.DataFrame = None
        self.vix_data: pd.Series = None
        self.aligned_data: pd.DataFrame = None
        
    def _merge_config(self, override: Optional[Dict]) -> Dict:
        """Merge default config with overrides."""
        config = {
            'data': DATA_CONFIG,
            'position': POSITION_CONFIG,
            'entry': ENTRY_RULES,
            'exit': EXIT_RULES,
            'risk': RISK_CONFIG,
        }
        
        if override:
            for key, value in override.items():
                if key in config:
                    config[key].update(value)
        
        return config
    
    # ==================== DATA LOADING ====================
    
    def load_data(self, start_date: str, end_date: Optional[str] = None) -> None:
        """
        Load SPX and VIX data for backtesting.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date (default: today)
        """
        print(f"📊 Loading data from {start_date}...")
        
        # Fetch SPX OHLC
        self.spx_data = self.fetcher.fetch_spx(start_date, end_date)
        print(f"   SPX: {len(self.spx_data)} days")
        
        # Fetch VIX close
        vix_source = self.config['data']['vix_source']
        self.vix_data = self.fetcher.fetch_vix(start_date, end_date, source=vix_source)
        print(f"   VIX: {len(self.vix_data)} days")
        
        # Normalize indices
        spx_close = self.spx_data['Close'].copy()
        spx_close.index = pd.to_datetime(spx_close.index).tz_localize(None).normalize()
        spx_close.name = 'SPX'
        
        vix_close = self.vix_data.copy()
        vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None).normalize()
        vix_close.name = 'VIX'
        
        # Align data
        self.aligned_data = pd.DataFrame({
            'SPX': spx_close,
            'VIX': vix_close
        }).dropna()
        
        if len(self.aligned_data) == 0:
            raise ValueError("No overlapping dates between SPX and VIX data!")
        
        print(f"   Aligned: {len(self.aligned_data)} trading days")
        print(f"   Date range: {self.aligned_data.index[0].date()} to {self.aligned_data.index[-1].date()}")
        
    # ==================== VIX-BASED CALCULATIONS ====================
    
    def calculate_expected_move(self, vix: float, days: int, spx_price: float) -> float:
        """
        Calculate expected SPX move based on VIX.
        
        Args:
            vix: Current VIX level
            days: Number of days forward
            spx_price: Current SPX price
            
        Returns:
            Expected move in SPX points (1 standard deviation)
        """
        # VIX is annualized, convert to daily
        daily_vol = vix / 100 / np.sqrt(252)
        
        # Scale to desired time horizon
        move_vol = daily_vol * np.sqrt(days)
        
        # Expected move in points
        expected_move = spx_price * move_vol
        
        return expected_move
    
    def calculate_short_strike(self, spx_price: float, vix: float, dte: int, 
                              structure: str = 'put') -> float:
        """
        Calculate short strike based on VIX-implied move.
        PROFESSIONAL: Uses elevated multiplier in high vol regimes.
        
        Args:
            spx_price: Current SPX price
            vix: Current VIX level
            dte: Days to expiration
            structure: 'put' or 'call'
            
        Returns:
            Short strike price
        """
        # Get base std dev
        std_dev = self.config['position']['std_dev_short']
        
        # PROFESSIONAL: Go wider in elevated vol
        regime = get_vix_regime(vix)
        if regime == 'ELEVATED':
            std_dev *= self.config['position']['std_dev_elevated_multiplier']
        
        expected_move = self.calculate_expected_move(vix, dte, spx_price)
        
        # Calculate strike
        if structure == 'put':
            strike = spx_price - (std_dev * expected_move)
        else:  # call
            strike = spx_price + (std_dev * expected_move)
        
        # Round to nearest 5
        strike = round(strike / 5) * 5
        
        return strike
    
    def calculate_credit(self, wing_width: float, vix: float, dte: int) -> float:
        """
        PROFESSIONAL credit calculation based on industry standards.
        
        Key insights:
        - 14-45 DTE typically yields 15-25% of spread width
        - Higher VIX = better credit
        - Longer DTE = better credit (up to 45 days)
        
        Args:
            wing_width: Spread width in dollars
            vix: Current VIX level
            dte: Days to expiration
            
        Returns:
            Estimated credit
        """
        # Base credit percentage based on DTE
        if dte >= 30:
            base_pct = 0.22  # Best credit for 30-45 DTE
        elif dte >= 14:
            base_pct = 0.18  # Good credit for 14-30 DTE
        elif dte >= 7:
            base_pct = 0.14  # Acceptable for 7-14 DTE
        else:
            base_pct = 0.12  # Lower for ultra-short DTE
        
        # VIX multiplier (normalized to VIX=20)
        vix_multiplier = min(vix / 20, 2.0)  # Cap at 2x
        
        # Calculate credit
        credit = wing_width * base_pct * vix_multiplier
        
        # Enforce minimum (10% of width) and target (15% of width)
        min_credit = wing_width * self.config['position']['min_credit_pct']
        credit = max(credit, min_credit)
        
        return credit
    
    # ==================== TRADE LOGIC ====================
    
    def should_enter_trade(self, date: datetime, vix: float, spx_history: pd.Series) -> bool:
        """
        PROFESSIONAL entry logic with trend filters.
        
        Args:
            date: Current date
            vix: Current VIX level
            spx_history: Recent SPX prices for trend analysis
            
        Returns:
            True if we should enter a trade
        """
        # Check VIX regime
        if vix < self.config['entry']['vix_min']:
            return False
        
        # Check day of week
        if date.weekday() in self.config['entry']['avoid_entry_days']:
            return False
        
        # Check position count
        if len(self.open_positions) >= self.config['entry']['max_positions']:
            return False
        
        # PROFESSIONAL: Trend filter
        if self.config['entry']['use_trend_filter'] and len(spx_history) >= 2:
            # Don't sell puts after down days (catching falling knife)
            if self.config['entry']['no_entry_after_down_day']:
                threshold = self.config['entry']['down_day_threshold']
                prev_day = spx_history.iloc[-2]
                current_day = spx_history.iloc[-1]
                
                if current_day < prev_day * (1 - threshold):
                    return False  # Skip entry after big down day
        
        return True
    
    def create_trade(self, entry_date: datetime, spx_price: float, vix: float) -> Trade:
        """
        Create a new trade based on current market conditions.
        
        Args:
            entry_date: Trade entry date
            spx_price: Current SPX price
            vix: Current VIX level
            
        Returns:
            Trade object
        """
        dte = self.config['position']['dte']
        expiration_date = entry_date + timedelta(days=dte)
        
        # Determine regime and structure
        regime = get_vix_regime(vix)
        structure = get_trade_structure(vix)
        
        # Calculate strikes (bull put spread)
        short_strike = self.calculate_short_strike(spx_price, vix, dte, 'put')
        wing_width = self.config['position']['wing_width_dollars']
        long_strike = short_strike - wing_width
        
        # PROFESSIONAL credit calculation
        credit_received = self.calculate_credit(wing_width, vix, dte)
        
        # Max loss = width - credit
        max_loss = wing_width - credit_received
        
        trade = Trade(
            entry_date=entry_date,
            expiration_date=expiration_date,
            entry_vix=vix,
            entry_spx=spx_price,
            regime=regime,
            structure=structure,
            short_strike=short_strike,
            long_strike=long_strike,
            credit_received=credit_received,
            max_loss=max_loss,
            dte=dte,
        )
        
        return trade
    
    def check_exit(self, trade: Trade, current_date: datetime, spx_price: float, 
                   vix: float) -> Optional[str]:
        """
        PROFESSIONAL exit logic - simplified to hold to expiration.
        
        Args:
            trade: Trade to check
            current_date: Current date
            spx_price: Current SPX price
            vix: Current VIX level
            
        Returns:
            Exit reason if should exit, None otherwise
        """
        trade.days_held = (current_date - trade.entry_date).days
        
        # Check if reached expiration
        if current_date >= trade.expiration_date:
            return 'EXPIRATION'
        
        # Stop loss - only if deep ITM
        if self.config['exit']['stop_loss_enabled']:
            if spx_price < trade.long_strike:  # Breached long strike = max loss
                return 'STOP_LOSS'
            elif spx_price < trade.short_strike:
                # Calculate estimated loss
                intrinsic_loss = trade.short_strike - spx_price
                if intrinsic_loss > trade.credit_received * self.config['exit']['stop_loss_multiplier']:
                    return 'STOP_LOSS'
        
        return None
    
    def close_trade(self, trade: Trade, exit_date: datetime, exit_spx: float, 
                   exit_reason: str) -> None:
        """
        Close a trade and calculate P&L.
        Simplified to focus on expiration-based exits.
        
        Args:
            trade: Trade to close
            exit_date: Exit date
            exit_spx: SPX price at exit
            exit_reason: Reason for exit
        """
        trade.exit_date = exit_date
        trade.exit_spx = exit_spx
        trade.exit_reason = exit_reason
        trade.is_open = False
        trade.days_held = (exit_date - trade.entry_date).days
        
        # Calculate P&L based on exit reason
        if exit_reason == 'EXPIRATION':
            # At expiration - check final position
            if exit_spx > trade.short_strike:
                # OTM - both options expire worthless, keep full credit
                trade.pnl = trade.credit_received
            else:
                # ITM - calculate spread value
                short_intrinsic = max(0, trade.short_strike - exit_spx)
                long_intrinsic = max(0, trade.long_strike - exit_spx)
                spread_value = short_intrinsic - long_intrinsic
                
                trade.pnl = trade.credit_received - spread_value
                # Can't lose more than max loss
                trade.pnl = max(trade.pnl, -trade.max_loss)
                
        else:  # STOP_LOSS
            # Exited early - take max loss
            trade.pnl = -trade.max_loss
        
        trade.pnl_pct = trade.pnl / abs(trade.credit_received) if trade.credit_received > 0 else 0
        
        # Remove from open positions
        if trade in self.open_positions:
            self.open_positions.remove(trade)
    
    # ==================== BACKTEST EXECUTION ====================
    
    def run_backtest(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Run the backtest over the specified period.
        
        Args:
            start_date: Start date
            end_date: End date (default: today)
            
        Returns:
            DataFrame with backtest results
        """
        print("\n" + "="*60)
        print("RUNNING PROFESSIONAL BACKTEST")
        print("="*60)
        
        # Load data
        self.load_data(start_date, end_date)
        
        # Initialize tracking
        self.trades = []
        self.open_positions = []
        daily_equity = []
        
        account_value = self.config['position']['account_size']
        
        # Iterate through each trading day
        print(f"\n⚙️ Processing {len(self.aligned_data)} trading days...")
        
        for i, (date, row) in enumerate(self.aligned_data.iterrows()):
            spx = row['SPX']
            vix = row['VIX']
            
            # Get recent SPX history for trend filter
            idx = self.aligned_data.index.get_loc(date)
            lookback = self.config['entry']['trend_lookback_days']
            spx_history = self.aligned_data['SPX'].iloc[max(0, idx-lookback):idx+1]
            
            # Check for exits first
            for trade in self.open_positions[:]:
                exit_reason = self.check_exit(trade, date, spx, vix)
                if exit_reason:
                    self.close_trade(trade, date, spx, exit_reason)
                    account_value += trade.pnl
            
            # Check for new entries
            if self.should_enter_trade(date, vix, spx_history):
                trade = self.create_trade(date, spx, vix)
                self.trades.append(trade)
                self.open_positions.append(trade)
            
            # Track daily equity
            daily_equity.append({
                'date': date,
                'equity': account_value,
                'open_positions': len(self.open_positions),
                'vix': vix,
                'spx': spx,
            })
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(self.aligned_data)} days...", end='\r')
        
        print(f"\n✅ Backtest complete: {len(self.trades)} trades executed")
        
        # Create equity curve
        equity_df = pd.DataFrame(daily_equity).set_index('date')
        self.equity_curve = equity_df['equity']
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics()
        
        return results
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        closed_trades = [t for t in self.trades if not t.is_open]
        
        if not closed_trades:
            return {'error': 'No closed trades'}
        
        # Basic stats
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades
        
        total_pnl = sum(t.pnl for t in closed_trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for reason in ['EXPIRATION', 'STOP_LOSS']:
            reason_trades = [t for t in closed_trades if t.exit_reason == reason]
            if reason_trades:
                exit_reasons[reason] = {
                    'count': len(reason_trades),
                    'win_rate': sum(1 for t in reason_trades if t.pnl > 0) / len(reason_trades),
                    'avg_pnl': np.mean([t.pnl for t in reason_trades]),
                }
        
        # Regime breakdown
        regime_stats = {}
        for regime in VIX_REGIMES.keys():
            regime_trades = [t for t in closed_trades if t.regime == regime]
            if regime_trades:
                regime_stats[regime] = {
                    'count': len(regime_trades),
                    'win_rate': sum(1 for t in regime_trades if t.pnl > 0) / len(regime_trades),
                    'avg_pnl': np.mean([t.pnl for t in regime_trades]),
                    'total_pnl': sum(t.pnl for t in regime_trades),
                }
        
        # Equity curve metrics
        returns = self.equity_curve.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        max_equity = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - max_equity) / max_equity
        max_drawdown = drawdown.min()
        
        results = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_equity': self.equity_curve.iloc[-1],
            'initial_equity': self.equity_curve.iloc[0],
            'total_return_pct': (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100,
            'regime_stats': regime_stats,
            'exit_reasons': exit_reasons,
        }
        
        return results
    
    def print_results(self, results: Dict) -> None:
        """Print backtest results in readable format."""
        print("\n" + "="*60)
        print("PROFESSIONAL BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n📊 Overall Performance:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']*100:.1f}%")
        print(f"  Total P&L: ${results['total_pnl']:,.2f}")
        print(f"  Total Return: {results['total_return_pct']:.1f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']*100:.1f}%")
        
        print(f"\n💰 Trade Statistics:")
        print(f"  Average Win: ${results['avg_win']:.2f}")
        print(f"  Average Loss: ${results['avg_loss']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Risk/Reward: {abs(results['avg_loss']/results['avg_win']) if results['avg_win'] > 0 else 0:.2f}:1")
        
        print(f"\n🎯 Exit Reasons:")
        for reason, stats in results['exit_reasons'].items():
            print(f"  {reason}:")
            print(f"    Count: {stats['count']}")
            print(f"    Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"    Avg P&L: ${stats['avg_pnl']:.2f}")
        
        print(f"\n📈 Performance by VIX Regime:")
        for regime, stats in results['regime_stats'].items():
            print(f"  {regime}:")
            print(f"    Trades: {stats['count']}")
            print(f"    Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"    Avg P&L: ${stats['avg_pnl']:.2f}")
            print(f"    Total P&L: ${stats['total_pnl']:.2f}")
        
        # Success criteria check
        print(f"\n✨ SUCCESS CRITERIA CHECK:")
        print(f"  ✓ Profit Factor > 1.0: {'✅ PASS' if results['profit_factor'] > 1.0 else '❌ FAIL'} ({results['profit_factor']:.2f})")
        print(f"  ✓ Sharpe > 1.0: {'✅ PASS' if results['sharpe_ratio'] > 1.0 else '❌ FAIL'} ({results['sharpe_ratio']:.2f})")
        print(f"  ✓ Max DD < 20%: {'✅ PASS' if results['max_drawdown'] > -0.20 else '❌ FAIL'} ({results['max_drawdown']*100:.1f}%)")
        
        print("="*60)
    
    def debug_trades(self, num_trades: int = 10) -> None:
        """Print detailed info about first N trades for debugging."""
        print("\n" + "="*60)
        print(f"🔍 DEBUG: First {num_trades} Trades")
        print("="*60)
        
        closed_trades = [t for t in self.trades if not t.is_open][:num_trades]
        
        for i, t in enumerate(closed_trades, 1):
            print(f"\nTrade #{i}:")
            print(f"  Entry: {t.entry_date.date()} | SPX: ${t.entry_spx:.2f} | VIX: {t.entry_vix:.2f}")
            print(f"  Exit:  {t.exit_date.date()} | SPX: ${t.exit_spx:.2f} | Days: {t.days_held}")
            print(f"  Regime: {t.regime} | DTE: {t.dte}")
            print(f"  Strikes: Short ${t.short_strike:.0f} / Long ${t.long_strike:.0f}")
            print(f"  Credit: ${t.credit_received:.2f} ({t.credit_received/t.max_loss*100:.1f}% of risk)")
            print(f"  Max Loss: ${t.max_loss:.2f}")
            print(f"  P&L: ${t.pnl:.2f} ({t.pnl_pct*100:.1f}%) | Exit: {t.exit_reason}")
            print(f"  Result: {'✅ WIN' if t.pnl > 0 else '❌ LOSS'}")
        
        print("="*60)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VIX-BASED SPX OPTIONS - PROFESSIONAL BACKTEST")
    print("="*60)
    print("\n📋 Configuration:")
    print(f"  • DTE: 14 days (optimal theta decay)")
    print(f"  • Exit: Hold to expiration (profit targets disabled)")
    print(f"  • Spread Width: $25")
    print(f"  • Strike Placement: 0.85 std dev (~80% PoP)")
    print(f"  • Trend Filter: ENABLED")
    print(f"  • Stop Loss: 2.5x credit")
    print("\n⚠️  NOTE: Profit targets disabled until we have real options pricing")
    print("="*60)
    
    # Initialize engine
    engine = BacktestEngine()
    
    # Run backtest
    results = engine.run_backtest(
        start_date='2020-01-01',
        end_date=None
    )
    
    # Print results
    engine.print_results(results)
    
    # Debug sample trades
    engine.debug_trades(15)
    
    print(f"\n✅ Backtest complete!")
    print(f"📊 Access equity curve: engine.equity_curve")
    print(f"📈 Access trades: engine.trades")