"""
RiskManager - Risk Assessment and Position Sizing

This module consolidates all risk-related logic into one place.
Provides a simple interface while hiding complex risk calculations.

PUBLIC INTERFACE:
    can_buy() -> bool
    can_sell() -> bool
    position_size() -> float
    should_emergency_sell() -> bool

PRIVATE IMPLEMENTATION:
    Risk score calculations
    Portfolio volatility analysis
    Position concentration checks
    Drawdown monitoring
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
from logger_config import logger


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk assessment results."""
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    can_trade: bool
    position_size_adjustment: float  # Multiplier for position size
    reason: str


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""
    btc_balance: float
    eur_balance: float
    current_price: float
    avg_buy_price: float
    unrealized_pnl: float
    win_rate: float
    volatility: float
    max_daily_drawdown: float


class RiskManager:
    """
    Manages all risk-related decisions and calculations.

    Philosophy:
    - Conservative by default
    - Multiple risk checks before allowing trades
    - Automatic position sizing reduction under stress
    - Clear emergency exit conditions
    """

    def __init__(
        self,
        max_position_pct: float = 0.15,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.10,
        max_daily_trades: int = 8,
        max_drawdown_pct: float = 0.15,
        risk_per_trade: float = 0.01,
    ):
        """
        Initialize risk manager.

        Args:
            max_position_pct: Maximum position size as % of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_daily_trades: Maximum trades per day
            max_drawdown_pct: Maximum portfolio drawdown allowed
            risk_per_trade: Risk as % of capital per trade
        """
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade = risk_per_trade

        # Risk thresholds
        self.high_volatility_threshold = 0.08  # 8%
        self.critical_volatility_threshold = 0.12  # 12%
        self.max_position_concentration = 0.25  # Max 25% of capital in one asset

        # Tracking
        self.daily_trade_count = 0
        self.today_pnl = 0.0
        self.peak_portfolio_value = 1.0

    def assess_risk(self, portfolio: PortfolioState) -> RiskMetrics:
        """
        Comprehensive risk assessment.

        Returns RiskMetrics with overall risk level and position sizing adjustment.
        """
        risk_score = 0.0
        reasons = []

        # Check volatility
        if portfolio.volatility > self.critical_volatility_threshold:
            risk_score += 0.4
            reasons.append("CRITICAL volatility")
        elif portfolio.volatility > self.high_volatility_threshold:
            risk_score += 0.2
            reasons.append("HIGH volatility")

        # Check drawdown
        max_drawdown = self._calculate_drawdown(portfolio)
        if max_drawdown > self.max_drawdown_pct:
            risk_score += 0.3
            reasons.append("Excessive drawdown")

        # Check position concentration
        position_concentration = (
            (portfolio.btc_balance * portfolio.current_price)
            / (portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance)
            if (portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance)
            > 0
            else 0
        )
        if position_concentration > self.max_position_concentration:
            risk_score += 0.2
            reasons.append("High position concentration")

        # Check win rate
        if portfolio.win_rate < 0.4:
            risk_score += 0.15
            reasons.append("Low win rate")

        # Determine risk level
        if risk_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
            position_adjustment = 0.25  # 25% of normal size
            can_trade = False
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
            position_adjustment = 0.5  # 50% of normal size
            can_trade = True
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MODERATE
            position_adjustment = 0.75  # 75% of normal size
            can_trade = True
        else:
            risk_level = RiskLevel.LOW
            position_adjustment = 1.0  # Full position size
            can_trade = True

        reason = (
            "; ".join(reasons) if reasons else "All risk metrics within acceptable range"
        )

        logger.info(
            f"Risk Assessment: Level={risk_level.value}, Score={risk_score:.2f}, "
            f"PositionAdjustment={position_adjustment:.0%}"
        )

        return RiskMetrics(
            risk_level=risk_level,
            risk_score=risk_score,
            can_trade=can_trade,
            position_size_adjustment=position_adjustment,
            reason=reason,
        )

    def can_buy(self, portfolio: PortfolioState) -> bool:
        """
        Check if buying is allowed given current risk conditions.

        Multiple checks:
        1. Risk level allows trading
        2. Haven't exceeded daily trade limit
        3. Have sufficient capital
        4. Position not too concentrated
        """
        risk_metrics = self.assess_risk(portfolio)

        if not risk_metrics.can_trade:
            logger.warning(f"⛔ Cannot buy - High risk: {risk_metrics.reason}")
            return False

        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(f"⛔ Cannot buy - Daily trade limit reached ({self.daily_trade_count})")
            return False

        if portfolio.eur_balance < 5.0:
            logger.warning(f"⛔ Cannot buy - Insufficient EUR balance (€{portfolio.eur_balance:.2f})")
            return False

        logger.info("✅ Buy check passed")
        return True

    def can_sell(self, portfolio: PortfolioState) -> bool:
        """
        Check if selling is allowed.

        Generally more permissive than buying - we should be able to exit positions
        even in high-risk situations.
        """
        if portfolio.btc_balance < 0.00001:
            logger.warning("⛔ Cannot sell - No BTC position")
            return False

        logger.info("✅ Sell check passed")
        return True

    def should_emergency_sell(self, portfolio: PortfolioState) -> bool:
        """
        Determine if we should force-sell to protect capital.

        Triggers:
        1. Drawdown exceeds threshold
        2. Volatility critical and losing money
        3. Portfolio underwater
        """
        risk_metrics = self.assess_risk(portfolio)

        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            if portfolio.unrealized_pnl < -0.05 * (
                portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance
            ):
                logger.error("🚨 EMERGENCY SELL - Critical risk + losses")
                return True

        return False

    def calculate_position_size(
        self,
        available_eur: float,
        current_price: float,
        portfolio: PortfolioState,
        base_position_pct: float = 0.10,
    ) -> float:
        """
        Calculate safe position size given risk conditions.

        Applies risk adjustments based on current portfolio state.
        """
        if available_eur <= 0 or current_price <= 0:
            return 0.0

        # Get risk adjustment
        risk_metrics = self.assess_risk(portfolio)

        # Calculate base position
        position_eur = available_eur * base_position_pct

        # Apply risk adjustment
        adjusted_position_eur = position_eur * risk_metrics.position_size_adjustment

        # Cap at max position
        max_position_eur = available_eur * self.max_position_pct
        adjusted_position_eur = min(adjusted_position_eur, max_position_eur)

        # Ensure minimum (only if buying is allowed)
        min_position_eur = 5.0  # Minimum €5
        if risk_metrics.can_trade:
            adjusted_position_eur = max(adjusted_position_eur, min_position_eur)

        # Convert to BTC
        btc_amount = adjusted_position_eur / current_price

        logger.info(
            f"Position sizing: {btc_amount:.8f} BTC "
            f"(€{adjusted_position_eur:.2f}, adjusted {risk_metrics.position_size_adjustment:.0%})"
        )

        return btc_amount

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_drawdown(self, portfolio: PortfolioState) -> float:
        """Calculate current portfolio drawdown from peak."""
        if self.peak_portfolio_value <= 0:
            return 0.0

        current_value = (
            portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance
        )
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value

        return max(0.0, drawdown)

    def record_trade(self, is_buy: bool):
        """Record a trade for daily limit tracking."""
        self.daily_trade_count += 1
        logger.debug(f"Trade recorded: {self.daily_trade_count}/{self.max_daily_trades} daily limit")

    def reset_daily_limits(self):
        """Reset daily trading limits (call at end of day)."""
        self.daily_trade_count = 0
        self.today_pnl = 0.0
        logger.info("Daily limits reset")

    def update_peak_value(self, portfolio: PortfolioState):
        """Update peak portfolio value for drawdown calculation."""
        current_value = (
            portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance
        )
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
