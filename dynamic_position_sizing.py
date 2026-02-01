"""
Phase 8 Task 3: Dynamic Position Sizing System
Implements adaptive position sizing based on signal quality, risk, and market conditions

Author: Phase 8 Optimization
Date: February 2026
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger("trading_bot")


class RiskProfile(Enum):
    """Risk classification for position sizing"""
    CONSERVATIVE = 0.6    # 60% of base size (high risk/uncertainty)
    MODERATE = 0.85      # 85% of base size (balanced)
    AGGRESSIVE = 1.2     # 120% of base size (low risk/high signal)
    MAXIMUM = 1.5        # 150% of base size (optimal conditions)


class MarketRegime(Enum):
    """Market regime affecting position sizing"""
    CRASH = 0.3          # Extreme drawdown period
    BEAR = 0.5           # Downtrend with losses
    CONSOLIDATION = 0.8  # Sideways movement
    BULL = 1.1           # Strong uptrend
    SUPER_BULL = 1.3     # Extreme rally


@dataclass
class PositionMetrics:
    """Metrics used for position sizing calculation"""
    signal_quality: float       # 0-100% from enhanced buy signals
    signal_strength: str        # EXTREME, VERY_STRONG, STRONG, etc.
    risk_off_probability: float # 0-1.0
    win_rate: float            # 0-1.0 (recent performance)
    sharpe_ratio: float        # Risk-adjusted returns
    drawdown: float            # Current drawdown %
    volatility: float          # 0-1.0+
    market_regime: str         # BULL, BEAR, CONSOLIDATION, etc.
    trade_frequency: int       # Trades in last N periods
    consecutive_losses: int    # Current loss streak
    confidence_score: float    # 0-100% overall confidence


@dataclass
class PositionSizing:
    """Result of position sizing calculation"""
    base_size_pct: float           # Base allocation %
    adjusted_size_pct: float       # After adjustments %
    risk_adjusted_size_pct: float  # Final recommended size %
    adjustments: Dict[str, float]  # Individual adjustment factors
    explanation: str               # Why this size was chosen
    max_loss_eur: float           # Maximum loss at this size
    capital_at_risk_pct: float    # % of portfolio at risk
    efficiency_rating: float      # 0-100% capital efficiency rating


class DynamicPositionSizer:
    """
    Implements dynamic position sizing based on multiple factors.
    
    Strategy:
    - Base positions scaled from signal quality
    - Risk adjustments for market conditions
    - Win rate confidence factors
    - Volatility-adjusted sizing
    - Drawdown protection
    - Consecutive loss penalties
    
    Benefits:
    - Larger positions during high-confidence signals
    - Smaller positions during uncertain/risky periods
    - Protection during losing streaks
    - Capital preservation during drawdowns
    - Optimized for risk-adjusted returns
    """
    
    # Base position sizes (% of available capital)
    BASE_BUY_SIZE = 0.10     # 10% per buy
    BASE_SELL_SIZE = 0.08    # 8% per sell
    MAX_POSITION_SIZE = 0.25 # Never exceed 25% on single trade
    MIN_POSITION_SIZE = 0.02 # Never go below 2%
    
    # Adjustment limits
    MAX_ADJUSTMENT = 1.5     # Never exceed 1.5x
    MIN_ADJUSTMENT = 0.3     # Never go below 0.3x
    
    # Risk thresholds
    HIGH_RISK_THRESHOLD = 0.7      # >70% risk-off = reduce size
    EXTREME_DRAWDOWN_THRESHOLD = 0.15  # >15% drawdown = conservative
    HIGH_VOLATILITY_THRESHOLD = 0.08   # >8% volatility = reduce
    
    def __init__(self):
        """Initialize dynamic position sizer"""
        self.logger = logging.getLogger("trading_bot")
    
    def calculate_buy_size(
        self,
        available_capital: float,
        metrics: PositionMetrics,
        current_price: float
    ) -> PositionSizing:
        """
        Calculate dynamic buy position size.
        
        Args:
            available_capital: EUR available to invest
            metrics: Position metrics for adjustment
            current_price: Current BTC price in EUR
        
        Returns:
            PositionSizing with recommended allocation
        """
        # Start with base size
        base_size_pct = self.BASE_BUY_SIZE
        
        # Calculate adjustment factors
        adjustments = {}
        
        # 1. Signal quality adjustment (0-1.5x)
        quality_factor = self._calculate_quality_factor(metrics.signal_quality)
        adjustments['quality'] = quality_factor
        
        # 2. Risk-off adjustment (0.3-1.0x)
        risk_factor = self._calculate_risk_factor(metrics.risk_off_probability)
        adjustments['risk_off'] = risk_factor
        
        # 3. Win rate confidence (0.6-1.3x)
        win_rate_factor = self._calculate_win_rate_factor(metrics.win_rate)
        adjustments['win_rate'] = win_rate_factor
        
        # 4. Volatility adjustment (0.5-1.1x)
        volatility_factor = self._calculate_volatility_factor(metrics.volatility)
        adjustments['volatility'] = volatility_factor
        
        # 5. Drawdown protection (0.3-1.0x)
        drawdown_factor = self._calculate_drawdown_factor(metrics.drawdown)
        adjustments['drawdown'] = drawdown_factor
        
        # 6. Loss streak penalty (0.3-1.0x)
        loss_factor = self._calculate_loss_streak_factor(metrics.consecutive_losses)
        adjustments['loss_streak'] = loss_factor
        
        # 7. Market regime (0.5-1.3x)
        regime_factor = self._calculate_regime_factor(metrics.market_regime)
        adjustments['regime'] = regime_factor
        
        # Calculate combined adjustment (geometric mean to avoid over-scaling)
        combined_adjustment = self._combine_adjustments(list(adjustments.values()))
        
        # Apply adjustment to base size
        adjusted_size_pct = base_size_pct * combined_adjustment
        
        # Clamp to limits
        final_size_pct = max(self.MIN_POSITION_SIZE, 
                            min(self.MAX_POSITION_SIZE, adjusted_size_pct))
        
        # Calculate risk metrics
        capital_to_deploy = available_capital * final_size_pct
        btc_amount = capital_to_deploy / current_price if current_price > 0 else 0
        
        # Estimate maximum loss (assume stop loss at 3%)
        max_loss = capital_to_deploy * 0.03
        
        # Generate explanation
        explanation = self._generate_buy_explanation(metrics, adjustments, final_size_pct)
        
        # Calculate efficiency rating
        efficiency = self._calculate_efficiency_rating(metrics, combined_adjustment)
        
        return PositionSizing(
            base_size_pct=base_size_pct,
            adjusted_size_pct=adjusted_size_pct,
            risk_adjusted_size_pct=final_size_pct,
            adjustments=adjustments,
            explanation=explanation,
            max_loss_eur=max_loss,
            capital_at_risk_pct=final_size_pct * 100,
            efficiency_rating=efficiency
        )
    
    def calculate_sell_size(
        self,
        btc_balance: float,
        metrics: PositionMetrics,
        current_price: float,
        profit_margin: float
    ) -> PositionSizing:
        """
        Calculate dynamic sell position size.
        
        Args:
            btc_balance: Current BTC holdings
            metrics: Position metrics for adjustment
            current_price: Current BTC price in EUR
            profit_margin: Current profit % on position
        
        Returns:
            PositionSizing with recommended allocation
        """
        # Start with base size
        base_size_pct = self.BASE_SELL_SIZE
        
        # Calculate adjustment factors
        adjustments = {}
        
        # 1. Profit margin boost (0.8-1.5x) - sell more when profitable
        profit_factor = self._calculate_profit_factor(profit_margin)
        adjustments['profit'] = profit_factor
        
        # 2. Win rate confidence (0.6-1.3x)
        win_rate_factor = self._calculate_win_rate_factor(metrics.win_rate)
        adjustments['win_rate'] = win_rate_factor
        
        # 3. Market regime (0.7-1.1x) - sell less in strong bull
        regime_factor = min(1.1, self._calculate_regime_factor(metrics.market_regime))
        adjustments['regime'] = regime_factor
        
        # 4. Risk-off adjustment (0.5-1.0x) - sell faster if risky
        risk_factor = self._calculate_risk_factor(metrics.risk_off_probability, sell=True)
        adjustments['risk_off'] = risk_factor
        
        # 5. Volatility adjustment (0.8-1.2x) - sell more in high vol
        volatility_factor = self._calculate_volatility_factor(metrics.volatility, sell=True)
        adjustments['volatility'] = volatility_factor
        
        # Combine adjustments
        combined_adjustment = self._combine_adjustments(list(adjustments.values()))
        
        # Apply adjustment
        adjusted_size_pct = base_size_pct * combined_adjustment
        
        # Clamp to limits
        final_size_pct = max(self.MIN_POSITION_SIZE,
                            min(self.MAX_POSITION_SIZE, adjusted_size_pct))
        
        # Calculate actual BTC to sell
        btc_to_sell = btc_balance * final_size_pct
        eur_amount = btc_to_sell * current_price
        
        # Generate explanation
        explanation = self._generate_sell_explanation(metrics, adjustments, final_size_pct, profit_margin)
        
        # Calculate efficiency
        efficiency = self._calculate_efficiency_rating(metrics, combined_adjustment)
        
        return PositionSizing(
            base_size_pct=base_size_pct,
            adjusted_size_pct=adjusted_size_pct,
            risk_adjusted_size_pct=final_size_pct,
            adjustments=adjustments,
            explanation=explanation,
            max_loss_eur=0,  # No loss on sell
            capital_at_risk_pct=eur_amount / (btc_balance * current_price) * 100 if btc_balance > 0 else 0,
            efficiency_rating=efficiency
        )
    
    def _calculate_quality_factor(self, quality: float) -> float:
        """Quality factor: 0-100% → 0.7-1.5x"""
        # Map 0-100 to 0.7-1.5
        return 0.7 + (quality / 100.0) * 0.8
    
    def _calculate_risk_factor(self, risk_off: float, sell: bool = False) -> float:
        """Risk factor: higher risk-off → lower positions"""
        if sell:
            # Sell more aggressively during risk
            return max(0.5, 0.6 + (1.0 - risk_off) * 0.4)
        # Buy less during risk
        return max(0.3, 1.0 - (risk_off * 1.4))
    
    def _calculate_win_rate_factor(self, win_rate: float) -> float:
        """Win rate factor: better performance → larger positions"""
        if win_rate < 0.3:
            return 0.6
        elif win_rate < 0.45:
            return 0.8
        elif win_rate < 0.55:
            return 1.0
        elif win_rate < 0.65:
            return 1.15
        else:
            return 1.3
    
    def _calculate_volatility_factor(self, volatility: float, sell: bool = False) -> float:
        """Volatility adjustment"""
        if sell:
            # Sell more in high volatility
            return min(1.2, 1.0 + volatility * 2)
        # Buy less in high volatility
        if volatility > self.HIGH_VOLATILITY_THRESHOLD:
            return max(0.5, 1.0 - (volatility - self.HIGH_VOLATILITY_THRESHOLD) * 5)
        return max(0.85, 1.0 - volatility * 5)
    
    def _calculate_drawdown_factor(self, drawdown: float) -> float:
        """Drawdown protection: higher drawdown → conservative sizing"""
        if drawdown < 0.05:
            return 1.0
        elif drawdown < 0.10:
            return 0.8
        elif drawdown < self.EXTREME_DRAWDOWN_THRESHOLD:
            return 0.5
        else:
            return 0.3
    
    def _calculate_loss_streak_factor(self, consecutive_losses: int) -> float:
        """Loss streak penalty: protect after losses"""
        if consecutive_losses == 0:
            return 1.0
        elif consecutive_losses == 1:
            return 0.8
        elif consecutive_losses == 2:
            return 0.6
        else:
            return 0.3
    
    def _calculate_regime_factor(self, market_regime: str) -> float:
        """Market regime adjustment"""
        regime_map = {
            'SUPER_BULL': 1.3,
            'BULL': 1.1,
            'CONSOLIDATION': 0.8,
            'BEAR': 0.5,
            'CRASH': 0.3,
        }
        return regime_map.get(market_regime, 0.8)
    
    def _calculate_profit_factor(self, profit_margin: float) -> float:
        """Profit factor: higher profits → more comfortable selling"""
        if profit_margin < 5:
            return 0.8
        elif profit_margin < 10:
            return 0.9
        elif profit_margin < 15:
            return 1.0
        elif profit_margin < 20:
            return 1.1
        else:
            return 1.5
    
    def _combine_adjustments(self, factors: list) -> float:
        """Combine multiple adjustment factors using geometric mean"""
        if not factors:
            return 1.0
        
        product = 1.0
        for factor in factors:
            product *= factor
        
        # Geometric mean
        geometric_mean = product ** (1.0 / len(factors))
        
        # Clamp to limits
        return max(self.MIN_ADJUSTMENT, 
                  min(self.MAX_ADJUSTMENT, geometric_mean))
    
    def _generate_buy_explanation(
        self, metrics: PositionMetrics, adjustments: Dict, size_pct: float
    ) -> str:
        """Generate explanation for buy sizing"""
        quality_level = "excellent" if metrics.signal_quality > 80 else "good" if metrics.signal_quality > 60 else "moderate" if metrics.signal_quality > 40 else "weak"
        risk_level = "low" if metrics.risk_off_probability < 0.3 else "moderate" if metrics.risk_off_probability < 0.6 else "high"
        
        factors = []
        if adjustments.get('quality', 1.0) > 1.1:
            factors.append("high-quality signal")
        if adjustments.get('win_rate', 1.0) > 1.1:
            factors.append("strong win rate")
        if adjustments.get('regime', 1.0) > 1.1:
            factors.append("bull market")
        
        factor_str = ", ".join(factors) if factors else "balanced approach"
        
        return f"📊 BUY {size_pct*100:.1f}%: {quality_level} signal ({metrics.signal_quality:.0f}%), {risk_level} risk, {factor_str}"
    
    def _generate_sell_explanation(
        self, metrics: PositionMetrics, adjustments: Dict, size_pct: float, profit: float
    ) -> str:
        """Generate explanation for sell sizing"""
        profit_text = "excellent" if profit > 20 else "strong" if profit > 10 else "good" if profit > 5 else "minimal"
        
        factors = []
        if adjustments.get('profit', 1.0) > 1.1:
            factors.append(f"{profit:.1f}% profit")
        if adjustments.get('risk_off', 1.0) > 1.0:
            factors.append("risk mitigation")
        if adjustments.get('volatility', 1.0) > 1.1:
            factors.append("high volatility")
        
        factor_str = ", ".join(factors) if factors else "regular rebalancing"
        
        return f"💰 SELL {size_pct*100:.1f}%: {profit_text} gains, {factor_str}"
    
    def _calculate_efficiency_rating(self, metrics: PositionMetrics, adjustment: float) -> float:
        """Calculate capital efficiency rating (0-100%)"""
        base_efficiency = 50.0
        
        # Add for good signal quality
        if metrics.signal_quality > 70:
            base_efficiency += 15
        elif metrics.signal_quality > 50:
            base_efficiency += 10
        
        # Add for low risk
        if metrics.risk_off_probability < 0.3:
            base_efficiency += 15
        elif metrics.risk_off_probability < 0.6:
            base_efficiency += 5
        
        # Add for good win rate
        if metrics.win_rate > 0.55:
            base_efficiency += 10
        
        # Adjust for volatility
        if metrics.volatility < 0.05:
            base_efficiency += 5
        elif metrics.volatility > 0.10:
            base_efficiency -= 5
        
        # Cap at 100
        return min(100.0, base_efficiency)
    
    def format_position_sizing(self, sizing: PositionSizing) -> str:
        """Format position sizing for logging"""
        lines = []
        lines.append(f"📊 DYNAMIC POSITION SIZING")
        lines.append(f"   Base Size: {sizing.base_size_pct*100:.1f}%")
        lines.append(f"   Adjusted Size: {sizing.adjusted_size_pct*100:.1f}%")
        lines.append(f"   Final Size: {sizing.risk_adjusted_size_pct*100:.1f}%")
        lines.append(f"   Max Loss: €{sizing.max_loss_eur:.2f}")
        lines.append(f"   Efficiency: {sizing.efficiency_rating:.0f}%")
        lines.append(f"   {sizing.explanation}")
        
        return "\n".join(lines)


def log_position_sizing_breakdown(sizing: PositionSizing) -> None:
    """Log detailed position sizing adjustments"""
    logger.info("━━━ POSITION SIZING BREAKDOWN ━━━")
    
    for factor_name, factor_value in sizing.adjustments.items():
        pct_change = (factor_value - 1.0) * 100
        direction = "↑" if factor_value > 1.0 else "↓" if factor_value < 1.0 else "→"
        logger.info(
            f"{direction} {factor_name.replace('_', ' ').title()}: {factor_value:.2f}x "
            f"({pct_change:+.0f}%)"
        )
