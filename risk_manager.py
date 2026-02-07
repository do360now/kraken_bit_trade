"""
RiskManager - Cycle-Aware Risk Assessment and Position Sizing

UPDATED: Now integrates CycleAnalyzer to adjust ALL risk parameters
based on where we are in the Bitcoin halving cycle.

PUBLIC INTERFACE:
    assess_risk(portfolio) -> RiskMetrics
    can_buy(portfolio) -> bool
    can_sell(portfolio) -> bool
    calculate_position_size(available_eur, price, portfolio) -> float
    should_emergency_sell(portfolio) -> bool
    get_cycle_adjusted_stops(entry_price, current_price) -> Dict

CYCLE INTEGRATION POINTS:
    - Position sizing scaled by cycle phase multiplier + DCA intensity
    - Stop-loss width adjusted by phase volatility expectations
    - Buy/sell gates shift based on cycle phase
    - Emergency sell respects "golden rule" floor estimate
    - Bear capitulation overrides risk blocks for small DCA buys
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
from logger_config import logger
from cycle_analyzer import CycleAnalyzer, CyclePhase, CycleAdjustments


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk assessment results — now includes cycle context."""
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    can_trade: bool
    position_size_adjustment: float  # Combined risk + cycle multiplier
    reason: str
    cycle_phase: Optional[CyclePhase] = None
    accumulation_score: Optional[float] = None


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
    Manages all risk-related decisions with cycle awareness.

    Philosophy:
    - Conservative by default, but modulated by cycle phase
    - Bear capitulation = be GREEDY (override normal risk blocks)
    - Euphoria/distribution = be CAUTIOUS (tighter everything)
    - Emergency sells respect the "golden rule" floor estimate
    """

    def __init__(
        self,
        max_position_pct: float = 0.15,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.10,
        max_daily_trades: int = 8,
        max_drawdown_pct: float = 0.15,
        risk_per_trade: float = 0.01,
        cycle_analyzer: Optional[CycleAnalyzer] = None,
    ):
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade = risk_per_trade

        # Cycle awareness — optional, falls back to neutral if not provided
        self.cycle = cycle_analyzer or CycleAnalyzer(current_cycle=4)

        # Cache to avoid recalculating on every call within the same tick
        self._cached_adjustments: Optional[CycleAdjustments] = None
        self._cached_price: Optional[float] = None

        # Risk thresholds
        self.high_volatility_threshold = 0.08
        self.critical_volatility_threshold = 0.12
        self.max_position_concentration = 0.25

        # Daily tracking
        self.daily_trade_count = 0
        self.today_pnl = 0.0
        self.peak_portfolio_value = 1.0

    # =========================================================================
    # CYCLE HELPERS
    # =========================================================================

    def _get_cycle_adj(self, price: float) -> CycleAdjustments:
        """Get cached cycle adjustments (refreshes if price moved >1%)."""
        if (
            self._cached_adjustments is None
            or self._cached_price is None
            or abs(price - self._cached_price) / max(self._cached_price, 1) > 0.01
        ):
            self._cached_adjustments = self.cycle.get_cycle_adjustments(price)
            self._cached_price = price
        return self._cached_adjustments

    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================

    def assess_risk(self, portfolio: PortfolioState) -> RiskMetrics:
        """
        Comprehensive risk assessment WITH cycle modulation.

        The cycle phase adjusts:
        - Volatility thresholds (wider in accumulation, tighter in distribution)
        - Drawdown tolerance (more in accumulation, less in euphoria)
        - Concentration limits (higher allowed in accumulation)
        - Final risk score dampened/amplified by phase risk_tolerance
        """
        cycle_adj = self._get_cycle_adj(portfolio.current_price)
        risk_score = 0.0
        reasons = []

        # --- Volatility check (cycle-adjusted thresholds) ---
        vol_high = self.high_volatility_threshold * cycle_adj.stop_loss_width
        vol_critical = self.critical_volatility_threshold * cycle_adj.stop_loss_width

        if portfolio.volatility > vol_critical:
            risk_score += 0.6
            reasons.append("CRITICAL volatility")
        elif portfolio.volatility > vol_high:
            risk_score += 0.2
            reasons.append("HIGH volatility")

        # --- Drawdown check (cycle-adjusted tolerance) ---
        max_drawdown = self._calculate_drawdown(portfolio)
        # In accumulation phases, tolerate more drawdown
        dd_tolerance = self.max_drawdown_pct
        if cycle_adj.risk_tolerance > 0:
            dd_tolerance = self.max_drawdown_pct / cycle_adj.risk_tolerance
        if max_drawdown > dd_tolerance:
            risk_score += 0.3
            reasons.append("Excessive drawdown")

        # --- Position concentration (cycle-adjusted) ---
        total_value = portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance
        concentration = (
            (portfolio.btc_balance * portfolio.current_price) / total_value
            if total_value > 0 else 0
        )
        # Allow higher concentration during strong accumulation phases
        max_conc = self.max_position_concentration * (
            1 + cycle_adj.buy_aggressiveness * 0.15
        )
        if concentration > max_conc:
            risk_score += 0.2
            reasons.append("High position concentration")

        # --- Win rate check ---
        if portfolio.win_rate < 0.4:
            risk_score += 0.15
            reasons.append("Low win rate")

        # --- Apply cycle dampener ---
        # High risk_tolerance (accumulation) reduces effective score
        # Low risk_tolerance (distribution) amplifies it
        dampener = 1.0 - (cycle_adj.risk_tolerance - 0.5) * 0.3
        risk_score *= max(0.5, min(1.5, dampener))

        # --- Classify risk level ---
        if risk_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
            pos_adj = 0.25
            can_trade = False
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
            pos_adj = 0.5
            can_trade = True
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MODERATE
            pos_adj = 0.75
            can_trade = True
        else:
            risk_level = RiskLevel.LOW
            pos_adj = 1.0
            can_trade = True

        # Apply cycle position multiplier ON TOP of risk adjustment
        pos_adj *= cycle_adj.position_size_multiplier

        reason = "; ".join(reasons) if reasons else "All risk metrics within acceptable range"
        reason = f"[{cycle_adj.phase.value}] {reason}"

        accum_score = self.cycle.get_accumulation_score(portfolio.current_price)

        logger.info(
            f"Risk: {risk_level.value} (score={risk_score:.2f}) | "
            f"Phase={cycle_adj.phase.value} | AccumScore={accum_score:.0f} | "
            f"PosAdj={pos_adj:.1%}"
        )

        return RiskMetrics(
            risk_level=risk_level,
            risk_score=risk_score,
            can_trade=can_trade,
            position_size_adjustment=pos_adj,
            reason=reason,
            cycle_phase=cycle_adj.phase,
            accumulation_score=accum_score,
        )

    def can_buy(self, portfolio: PortfolioState) -> bool:
        """
        Check if buying is allowed.

        Cycle-aware: bear capitulation OVERRIDES risk blocks for small DCA buys.
        """
        risk_metrics = self.assess_risk(portfolio)
        cycle_adj = self._get_cycle_adj(portfolio.current_price)

        if not risk_metrics.can_trade:
            # Override: allow small buys during bear capitulation even at high risk
            if cycle_adj.phase == CyclePhase.BEAR_CAPITULATION:
                logger.warning(
                    "⚠️ Risk block OVERRIDDEN — BEAR_CAPITULATION allows small DCA buys"
                )
                return True
            logger.warning(f"⛔ Cannot buy — {risk_metrics.reason}")
            return False

        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(f"⛔ Cannot buy — daily limit ({self.daily_trade_count})")
            return False

        min_eur = 5.0 / max(cycle_adj.buy_aggressiveness, 0.1)
        if portfolio.eur_balance < min_eur:
            logger.warning(f"⛔ Cannot buy — €{portfolio.eur_balance:.2f} < min €{min_eur:.2f}")
            return False

        logger.info(f"✅ Buy allowed (phase: {cycle_adj.phase.value})")
        return True

    def can_sell(self, portfolio: PortfolioState) -> bool:
        """
        Check if selling is allowed.

        Cycle-aware: during strong accumulation phases, require higher
        profit margins before allowing a sell.
        """
        if portfolio.btc_balance < 0.00001:
            logger.warning("⛔ Cannot sell — no BTC position")
            return False

        cycle_adj = self._get_cycle_adj(portfolio.current_price)

        # In strong accumulation, block sells unless significantly profitable
        if cycle_adj.sell_reluctance >= 2.0 and portfolio.avg_buy_price > 0:
            profit_pct = (
                (portfolio.current_price - portfolio.avg_buy_price)
                / portfolio.avg_buy_price
            )
            min_profit = 0.05 * cycle_adj.sell_reluctance
            if profit_pct < min_profit:
                logger.info(
                    f"⛔ Sell blocked by {cycle_adj.phase.value}: "
                    f"profit {profit_pct:.1%} < required {min_profit:.1%}"
                )
                return False

        logger.info("✅ Sell allowed")
        return True

    def should_emergency_sell(self, portfolio: PortfolioState) -> bool:
        """
        Determine if we should force-sell to protect capital.

        CYCLE-AWARE: If price is near or above the estimated "golden rule" floor,
        DON'T emergency sell — historical patterns say it will recover.
        """
        risk_metrics = self.assess_risk(portfolio)
        if risk_metrics.risk_level != RiskLevel.CRITICAL:
            return False

        cycle_adj = self._get_cycle_adj(portfolio.current_price)
        floor_eur = cycle_adj.estimated_floor_eur

        # Golden rule protection: if we're above the estimated floor, hold
        if portfolio.current_price >= floor_eur * 0.90:
            logger.info(
                f"🛡️ Emergency sell BLOCKED: €{portfolio.current_price:,.0f} "
                f"near cycle floor €{floor_eur:,.0f} — golden rule protection"
            )
            return False

        # Standard emergency: critical risk AND significant unrealized loss
        total_value = portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance
        if portfolio.unrealized_pnl < -0.05 * total_value:
            logger.error("🚨 EMERGENCY SELL — below golden rule floor with losses")
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
        Calculate position size with full cycle + risk integration.

        Cycle phase DIRECTLY scales position:
        - Bear capitulation: up to 2x normal * 2x DCA = 4x effective
        - Post-halving: 1.5x * 1.5 DCA = 2.25x
        - Euphoria: 0.5x * 0.5 DCA = 0.25x
        - Distribution: 0.3x * 0.3 DCA = 0.09x
        """
        if available_eur <= 0 or current_price <= 0:
            return 0.0

        risk_metrics = self.assess_risk(portfolio)
        cycle_adj = self._get_cycle_adj(current_price)

        # Base position in EUR
        position_eur = available_eur * base_position_pct

        # Apply combined risk + cycle adjustment
        position_eur *= risk_metrics.position_size_adjustment

        # Apply DCA intensity from cycle phase
        position_eur *= cycle_adj.dca_intensity

        # Cap at max position (also cycle-scaled, hard cap at 30%)
        max_pct = min(self.max_position_pct * cycle_adj.position_size_multiplier, 0.30)
        max_eur = available_eur * max_pct
        position_eur = min(position_eur, max_eur)

        # Minimum trade size
        min_eur = 5.0
        if risk_metrics.can_trade or cycle_adj.phase == CyclePhase.BEAR_CAPITULATION:
            position_eur = max(position_eur, min_eur)

        btc_amount = position_eur / current_price

        logger.info(
            f"Position: {btc_amount:.8f} BTC (€{position_eur:.2f}) | "
            f"Phase={cycle_adj.phase.value} | "
            f"RiskAdj={risk_metrics.position_size_adjustment:.0%} | "
            f"DCA={cycle_adj.dca_intensity:.1f}x"
        )

        return btc_amount

    def get_cycle_adjusted_stops(
        self, entry_price: float, current_price: float
    ) -> Dict[str, float]:
        """
        Calculate cycle-adjusted stop-loss and take-profit levels.

        Returns:
            Dict with stop_loss, take_profit prices and percentages
        """
        cycle_adj = self._get_cycle_adj(current_price)

        sl_pct = self.stop_loss_pct * cycle_adj.stop_loss_width
        tp_pct = self.take_profit_pct * cycle_adj.take_profit_width

        stop_loss = entry_price * (1 - sl_pct)
        take_profit = entry_price * (1 + tp_pct)

        # Floor protection: stop loss never below 95% of estimated cycle floor
        floor = cycle_adj.estimated_floor_eur
        if stop_loss < floor * 0.95:
            stop_loss = floor * 0.95
            logger.debug(f"Stop loss clamped to cycle floor: €{stop_loss:,.0f}")

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "stop_loss_pct": sl_pct,
            "take_profit_pct": tp_pct,
            "phase": cycle_adj.phase.value,
        }

    # =========================================================================
    # TRACKING METHODS
    # =========================================================================

    def _calculate_drawdown(self, portfolio: PortfolioState) -> float:
        if self.peak_portfolio_value <= 0:
            return 0.0
        current = portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance
        dd = (self.peak_portfolio_value - current) / self.peak_portfolio_value
        return max(0.0, dd)

    def record_trade(self, is_buy: bool):
        self.daily_trade_count += 1

    def reset_daily_limits(self):
        self.daily_trade_count = 0
        self.today_pnl = 0.0
        logger.info("Daily limits reset")

    def update_peak_value(self, portfolio: PortfolioState):
        current = portfolio.btc_balance * portfolio.current_price + portfolio.eur_balance
        if current > self.peak_portfolio_value:
            self.peak_portfolio_value = current
