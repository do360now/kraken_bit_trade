"""
Phase 8 Task 2: Tiered Profit Taking System
Implements multi-tier profit harvesting for capital recycling optimization

Author: Phase 8 Optimization
Date: February 2026
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
import logging

logger = logging.getLogger("trading_bot")


class ProfitTier(Enum):
    """Profit taking tiers with different position reduction percentages"""
    TIER_1 = 5      # First 5% profit target
    TIER_2 = 10     # Second 10% profit target
    TIER_3 = 15     # Third 15% profit target
    TIER_4 = 20     # Fourth 20% profit target
    TIER_5 = 30     # Fifth 30% profit target (major resistance)


class PositionReduction(Enum):
    """Position reduction amount for each tier"""
    SMALL = 0.15    # Sell 15% of position (tiers 1-2)
    MEDIUM = 0.20   # Sell 20% of position (tiers 3-4)
    LARGE = 0.25    # Sell 25% of position (tier 5)


@dataclass
class TierAnalysis:
    """Result of tiered profit analysis"""
    current_profit: float           # Current profit margin (%)
    active_tier: Optional[int]      # Which tier is triggered (1-5 or None)
    tier_price_target: Optional[float]  # Price at which tier triggers
    reduction_percentage: Optional[float]  # % of position to sell
    tier_profit_margin: Optional[float]    # Profit % at tier
    recommendation: str              # Action recommendation
    next_tier_price: Optional[float] # Price at next tier
    capital_recovered: float        # Estimated EUR to recover


@dataclass
class TieredSellAnalysis:
    """Complete tiered sell analysis"""
    should_sell: bool               # Whether to execute tiered sell
    active_tiers: List[int]         # All triggered tiers (for backtesting)
    highest_active_tier: Optional[int]  # Highest tier currently active
    total_position_reduction: float  # Total % of position to sell
    total_capital_recovery: float    # Total EUR to recover
    recommendation: str              # Final recommendation
    tier_details: Dict[int, TierAnalysis]  # Details for each tier


class TieredProfitTakingSystem:
    """
    Implements tiered profit taking to maximize capital recycling.
    
    Strategy:
    - Tier 1 (5%): Sell small amount to lock initial gains
    - Tier 2 (10%): Sell small amount as confirmation of uptrend
    - Tier 3 (15%): Sell medium amount at technical resistance
    - Tier 4 (20%): Sell medium amount at strong resistance
    - Tier 5 (30%): Sell large amount at major resistance
    
    Benefits:
    - Locks in gains progressively
    - Recycled capital available for new buys
    - Reduces drawdown risk
    - Maintains upside participation
    """
    
    # Tier configuration (profit % required)
    TIER_LEVELS = {
        1: 5,      # Tier 1: 5% profit
        2: 10,     # Tier 2: 10% profit
        3: 15,     # Tier 3: 15% profit
        4: 20,     # Tier 4: 20% profit
        5: 30,     # Tier 5: 30% profit
    }
    
    # Position reduction amounts for each tier
    POSITION_REDUCTIONS = {
        1: 0.15,   # Tier 1: Sell 15%
        2: 0.15,   # Tier 2: Sell 15%
        3: 0.20,   # Tier 3: Sell 20%
        4: 0.20,   # Tier 4: Sell 20%
        5: 0.25,   # Tier 5: Sell 25%
    }
    
    def __init__(self):
        """Initialize tiered profit taking system"""
        self.logger = logging.getLogger("trading_bot")
        self.tier_history = {}  # Track which tiers have been hit
    
    def analyze_tiered_profits(
        self,
        current_price: float,
        avg_buy_price: float,
        btc_balance: float,
        tier_history: Optional[Dict[int, bool]] = None
    ) -> TieredSellAnalysis:
        """
        Analyze current position against all profit tiers.
        
        Args:
            current_price: Current BTC price in EUR
            avg_buy_price: Average buy price in EUR
            btc_balance: Current BTC holdings
            tier_history: Previously hit tiers {1: True, 2: True, ...}
        
        Returns:
            TieredSellAnalysis with tier recommendations
        """
        if not tier_history:
            tier_history = {}
        
        # Calculate current profit
        current_profit = ((current_price - avg_buy_price) / avg_buy_price) * 100 if avg_buy_price > 0 else 0
        
        active_tiers = []
        total_reduction = 0.0
        total_recovery = 0.0
        tier_details = {}
        
        # Check each tier
        for tier_num in range(1, 6):
            tier_profit = self.TIER_LEVELS[tier_num]
            tier_reached = current_profit >= tier_profit
            previously_hit = tier_history.get(tier_num, False)
            
            # Calculate tier-specific metrics
            if tier_reached and not previously_hit:
                # This tier is active and hasn't been hit yet
                active_tiers.append(tier_num)
                reduction = self.POSITION_REDUCTIONS[tier_num]
                total_reduction += reduction
                
                # Calculate capital recovery
                btc_to_sell = btc_balance * reduction
                capital_recovery = btc_to_sell * current_price
                total_recovery += capital_recovery
            
            # Always calculate tier analysis for logging/display
            tier_price = self._calculate_tier_price(avg_buy_price, tier_profit)
            next_tier_price = None
            if tier_num < 5:
                next_tier_price = self._calculate_tier_price(
                    avg_buy_price, self.TIER_LEVELS[tier_num + 1]
                )
            
            tier_details[tier_num] = TierAnalysis(
                current_profit=current_profit,
                active_tier=tier_num if tier_reached else None,
                tier_price_target=tier_price,
                reduction_percentage=self.POSITION_REDUCTIONS[tier_num] if tier_reached else None,
                tier_profit_margin=tier_profit,
                recommendation=self._get_tier_recommendation(tier_num, tier_reached, previously_hit),
                next_tier_price=next_tier_price,
                capital_recovered=total_recovery
            )
        
        # Determine highest active tier
        highest_active = max(active_tiers) if active_tiers else None
        
        # Generate recommendation
        should_sell = len(active_tiers) > 0
        recommendation = self._generate_recommendation(
            current_profit, active_tiers, highest_active, total_reduction
        )
        
        return TieredSellAnalysis(
            should_sell=should_sell,
            active_tiers=active_tiers,
            highest_active_tier=highest_active,
            total_position_reduction=total_reduction,
            total_capital_recovery=total_recovery,
            recommendation=recommendation,
            tier_details=tier_details
        )
    
    def _calculate_tier_price(self, avg_buy_price: float, profit_percent: float) -> float:
        """Calculate the price at which a profit tier is reached"""
        return avg_buy_price * (1 + profit_percent / 100.0)
    
    def _get_tier_recommendation(
        self, tier_num: int, tier_reached: bool, previously_hit: bool
    ) -> str:
        """Get recommendation text for a specific tier"""
        tier_names = {
            1: "First gains (scalp)",
            2: "Confirmed uptrend",
            3: "Technical resistance",
            4: "Strong resistance",
            5: "Major resistance",
        }
        
        status = "✓ READY" if tier_reached and not previously_hit else "✗ Pending" if not tier_reached else "✓ Already taken"
        
        return f"Tier {tier_num} ({self.TIER_LEVELS[tier_num]}%): {tier_names[tier_num]} - {status}"
    
    def _generate_recommendation(
        self, current_profit: float, active_tiers: List[int], 
        highest_active: Optional[int], total_reduction: float
    ) -> str:
        """Generate final recommendation based on active tiers"""
        if not active_tiers:
            if current_profit > 0:
                profit_str = f"{current_profit:.1f}% profit"
                next_target = None
                for tier_profit in sorted(self.TIER_LEVELS.values()):
                    if current_profit < tier_profit:
                        next_target = tier_profit
                        break
                
                if next_target:
                    return f"📈 Holding: {profit_str}, next tier at {next_target}%"
                else:
                    return f"📈 All tiers passed: {profit_str} - Consider emergency sell"
            else:
                return "📊 No profit yet - holding"
        
        if highest_active == 1:
            return f"💰 Tier 1 active: Sell {self.POSITION_REDUCTIONS[1]*100:.0f}% to lock initial gains"
        elif highest_active == 2:
            return f"💰 Tier 2 active: Sell {self.POSITION_REDUCTIONS[2]*100:.0f}% - confirmed uptrend"
        elif highest_active == 3:
            return f"💰💰 Tier 3 active: Sell {self.POSITION_REDUCTIONS[3]*100:.0f}% at resistance"
        elif highest_active == 4:
            return f"💰💰 Tier 4 active: Sell {self.POSITION_REDUCTIONS[4]*100:.0f}% at strong resistance"
        elif highest_active == 5:
            return f"💰💰💰 Tier 5 active: Sell {self.POSITION_REDUCTIONS[5]*100:.0f}% - MAJOR PROFITS"
        
        return "Unknown tier configuration"
    
    def calculate_sale_amount(
        self, active_tier: int, btc_balance: float, current_price: float
    ) -> float:
        """
        Calculate exact amount to sell based on active tier.
        
        Args:
            active_tier: Tier number (1-5)
            btc_balance: Total BTC held
            current_price: Current price in EUR
        
        Returns:
            Amount in EUR to sell
        """
        reduction_pct = self.POSITION_REDUCTIONS.get(active_tier, 0.15)
        btc_to_sell = btc_balance * reduction_pct
        eur_amount = btc_to_sell * current_price
        
        return eur_amount
    
    def get_position_after_tier(
        self, tier_num: int, btc_balance: float
    ) -> float:
        """
        Calculate remaining BTC position after tier sale.
        
        Args:
            tier_num: Tier number (1-5)
            btc_balance: Current BTC balance
        
        Returns:
            Remaining BTC after sale
        """
        reduction_pct = self.POSITION_REDUCTIONS.get(tier_num, 0.15)
        return btc_balance * (1 - reduction_pct)
    
    def get_total_reduction_at_tier(self, tier_num: int) -> float:
        """
        Calculate cumulative position reduction up to and including tier.
        
        For example: Tier 3 means tiers 1, 2, 3 have all been executed.
        Cumulative: 15% + 15% + 20% = 50% total reduction.
        
        Args:
            tier_num: Tier number (1-5)
        
        Returns:
            Total % of position sold up to this tier
        """
        total = 0.0
        for tier in range(1, min(tier_num + 1, 6)):
            total += self.POSITION_REDUCTIONS.get(tier, 0.15)
        return total
    
    def get_remaining_position_at_tier(
        self, tier_num: int, btc_balance: float
    ) -> float:
        """
        Calculate remaining position after executing all tiers up to tier_num.
        
        Args:
            tier_num: Tier number (1-5)
            btc_balance: Initial BTC balance
        
        Returns:
            Remaining BTC after all tiers up to tier_num
        """
        total_reduction = self.get_total_reduction_at_tier(tier_num)
        return btc_balance * (1 - total_reduction)
    
    def format_tier_analysis(self, analysis: TieredSellAnalysis) -> str:
        """Format tier analysis for logging"""
        lines = []
        lines.append(f"📊 TIERED PROFIT ANALYSIS")
        lines.append(f"   Current Profit: {analysis.tier_details[1].current_profit:.2f}%")
        lines.append(f"   Highest Active Tier: {analysis.highest_active_tier if analysis.highest_active_tier else 'None'}")
        lines.append(f"   Total Position Reduction: {analysis.total_position_reduction*100:.1f}%")
        lines.append(f"   Capital Recovery: €{analysis.total_capital_recovery:.2f}")
        lines.append(f"   Recommendation: {analysis.recommendation}")
        
        return "\n".join(lines)


def log_tier_details(analysis: TieredSellAnalysis) -> None:
    """Log detailed tier information"""
    logger.info("━━━ TIER BREAKDOWN ━━━")
    
    for tier_num in range(1, 6):
        tier_detail = analysis.tier_details[tier_num]
        status = "✓" if tier_num in analysis.active_tiers else " "
        
        logger.info(
            f"{status} Tier {tier_num}: {tier_detail.tier_profit_margin:2d}% target @ "
            f"€{tier_detail.tier_price_target:,.0f} | "
            f"Sell {TieredProfitTakingSystem.POSITION_REDUCTIONS[tier_num]*100:.0f}% | "
            f"Recover €{tier_detail.capital_recovered:,.0f}"
        )
