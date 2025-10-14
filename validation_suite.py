"""
Lightweight validation suite for earnings backtest integrity
Run after batch_analyze() to catch regressions immediately
"""

import pandas as pd
from typing import Dict, List, Tuple
import sys


class BacktestValidator:
    """Lean validator - only checks critical invariants"""
    
    def __init__(self, df: pd.DataFrame, ticker_order: List[str] = None):
        """
        Args:
            df: Results dataframe from batch_analyze
            ticker_order: Expected ticker order (optional, from input file)
        """
        self.df = df
        self.ticker_order = ticker_order
        self.errors = []
        self.warnings = []
        
    def validate_all(self, strict: bool = False) -> bool:
        """
        Run all critical validations
        
        Args:
            strict: If True, warnings count as failures
            
        Returns:
            True if all validations pass
        """
        print("\n" + "="*80)
        print("🔍 RUNNING VALIDATION CHECKS")
        print("="*80)
        
        # Critical checks
        self._check_required_columns()
        self._check_tier_sanity()
        self._check_containment_bounds()
        self._check_strategy_logic()
        
        # Important but non-critical
        if self.ticker_order:
            self._check_order_preservation()
        
        # Report results
        self._print_results()
        
        if self.errors:
            return False
        if strict and self.warnings:
            return False
        return True
    
    def _check_required_columns(self):
        """Verify all critical columns exist"""
        required = [
            'ticker', 'hvol',
            '45d_contain', '90d_contain',
            '45d_breaks_up', '45d_breaks_dn',
            '90d_breaks_up', '90d_breaks_dn',
            '45d_drift', '90d_drift',
            'strategy'
        ]
        
        missing = [col for col in required if col not in self.df.columns]
        
        if missing:
            self.errors.append(f"❌ Missing required columns: {', '.join(missing)}")
            # Also show what columns ARE present for debugging
            print(f"   Available columns: {self.df.columns.tolist()}")
        else:
            print("✅ All required columns present")
    
    def _check_tier_sanity(self):
        """Verify tier values are reasonable"""
        
        if 'tier' not in self.df.columns:
            return  # Will be caught by required columns check
        
        # Check tier values are reasonable (0.5 to 3.0 typically)
        invalid_tiers = self.df[(self.df['tier'] < 0.5) | (self.df['tier'] > 3.0)]
        if not invalid_tiers.empty:
            self.warnings.append(
                f"⚠️  Unusual tier values: {invalid_tiers[['ticker', 'tier']].to_dict('records')}"
            )
        
        # Check that tiers aren't all the same (would indicate calculation issue)
        unique_tiers = self.df['tier'].nunique()
        if unique_tiers == 1:
            self.errors.append("❌ All tiers are identical - tier calculation failed")
        elif unique_tiers < 3 and len(self.df) > 10:
            self.warnings.append(f"⚠️  Only {unique_tiers} unique tiers for {len(self.df)} tickers")
        else:
            print("✅ Tier values appear reasonable and varied")
    
    def _check_containment_bounds(self):
        """Verify containment rates are within valid range"""
        
        # Containment should be 0-100%
        invalid_45 = self.df[(self.df['45d_contain'] < 0) | (self.df['45d_contain'] > 100)]
        invalid_90 = self.df[(self.df['90d_contain'] < 0) | (self.df['90d_contain'] > 100)]
        
        if not invalid_45.empty:
            self.errors.append(
                f"❌ Invalid 45d containment: {invalid_45[['ticker', '45d_contain']].to_dict('records')}"
            )
        if not invalid_90.empty:
            self.errors.append(
                f"❌ Invalid 90d containment: {invalid_90[['ticker', '90d_contain']].to_dict('records')}"
            )
        
        if invalid_45.empty and invalid_90.empty:
            print("✅ Containment rates within valid bounds (0-100%)")
    
    def _check_strategy_logic(self):
        """Verify strategy assignments follow rules"""
        
        for idx, row in self.df.iterrows():
            strategy = row['strategy']
            contain_90 = row['90d_contain']
            contain_45 = row['45d_contain']
            
            # IC90 should only appear if 90d containment >= 69.5%
            if 'IC90' in strategy and contain_90 < 69.5:
                self.errors.append(
                    f"❌ {row['ticker']}: IC90 assigned but 90d containment "
                    f"({contain_90:.1f}%) < 69.5%"
                )
            
            # IC45 should only appear if 45d containment >= 69.5%
            if 'IC45' in strategy and contain_45 < 69.5:
                self.errors.append(
                    f"❌ {row['ticker']}: IC45 assigned but 45d containment "
                    f"({contain_45:.1f}%) < 69.5%"
                )
            
            # SKIP should mean no IC assignments
            if strategy == 'SKIP' and (contain_90 >= 69.5 or contain_45 >= 69.5):
                self.warnings.append(
                    f"⚠️  {row['ticker']}: SKIP assigned but has sufficient containment "
                    f"(45d: {contain_45:.1f}%, 90d: {contain_90:.1f}%)"
                )
        
        if not self.errors:
            print("✅ Strategy assignments follow logic rules")
    
    def _check_order_preservation(self):
        """Verify output order matches input order"""
        
        actual_order = self.df['ticker'].tolist()
        
        if actual_order != self.ticker_order:
            # Find what changed
            changes = []
            for i, (expected, actual) in enumerate(zip(self.ticker_order, actual_order)):
                if expected != actual:
                    changes.append(f"Position {i}: expected {expected}, got {actual}")
            
            self.warnings.append(
                f"⚠️  Order not preserved! First differences:\n" +
                "\n".join(changes[:5])  # Show first 5 differences
            )
        else:
            print("✅ Ticker order preserved from input")
    
    def _print_results(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("📋 VALIDATION SUMMARY")
        print("="*80)
        
        if not self.errors and not self.warnings:
            print("✅ ALL CHECKS PASSED - No issues detected")
        else:
            if self.errors:
                print(f"\n❌ CRITICAL ERRORS: {len(self.errors)}")
                for error in self.errors:
                    print(f"  {error}")
            
            if self.warnings:
                print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
                for warning in self.warnings[:10]:  # Limit to 10 warnings
                    print(f"  {warning}")
                if len(self.warnings) > 10:
                    print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        print("="*80 + "\n")


def quick_validate(df: pd.DataFrame, ticker_order: List[str] = None, strict: bool = False) -> bool:
    """
    Quick validation - single function call
    
    Args:
        df: Results dataframe
        ticker_order: Expected ticker order (optional)
        strict: Treat warnings as failures
        
    Returns:
        True if validation passes
        
    Example:
        >>> results_df = batch_analyze(tickers)
        >>> if not quick_validate(results_df, ticker_list):
        >>>     print("Validation failed!")
    """
    validator = BacktestValidator(df, ticker_order)
    return validator.validate_all(strict=strict)
