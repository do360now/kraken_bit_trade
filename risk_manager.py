"""
Risk manager — cycle-aware risk gating for the trading bot.

Decides whether a trade is allowed, enforces emergency sell logic with
golden rule floor protection, manages drawdown tracking, and provides
cycle-adjusted stop levels.

Key lessons from previous bot:
- Golden rule floor protection: don't emergency sell above estimated cycle floor.
- Bear capitulation override: allow DCA buys even at elevated risk levels.
- Reserve floor: never go below 20% of starting EUR balance.
- Daily trade count limits with proper reset logic.

Design:
- can_trade() is the main gate: returns (allowed, reason) for every decision.
- emergency_sell() checks if portfolio is in danger requiring immediate exit.
- get_stops() provides cycle-adjusted stop-loss levels.
- All state is persisted for restart recovery.
- Never raises exceptions past the public interface.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import BotConfig, CyclePhase, RiskConfig, PersistenceConfig
from cycle_detector import CycleState
from signal_engine import Action, CompositeSignal

logger = logging.getLogger(__name__)


# ─── Output dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskDecision:
    """Result of a risk check."""
    allowed: bool
    reason: str
    override_active: bool = False  # True if capitulation override engaged


@dataclass(frozen=True)
class StopLevels:
    """Cycle-adjusted stop-loss levels for an open position."""
    stop_price: float         # Hard stop — exit if price drops below this
    warning_price: float      # Soft warning — reduce position if hit
    atr_width: float          # ATR-based distance from entry to stop

    @property
    def valid(self) -> bool:
        """Stop levels are valid if prices are positive and ordered."""
        return 0 < self.stop_price < self.warning_price


@dataclass(frozen=True)
class PortfolioState:
    """Current portfolio snapshot for risk assessment."""
    eur_balance: float
    btc_balance: float
    btc_price: float
    starting_eur: float       # Initial EUR balance for reserve floor calc

    @property
    def total_value_eur(self) -> float:
        """Total portfolio value in EUR."""
        return self.eur_balance + self.btc_balance * self.btc_price

    @property
    def btc_value_eur(self) -> float:
        return self.btc_balance * self.btc_price

    @property
    def eur_allocation(self) -> float:
        """Fraction of portfolio in EUR (0.0–1.0)."""
        total = self.total_value_eur
        if total <= 0:
            return 1.0
        return self.eur_balance / total

    @property
    def btc_allocation(self) -> float:
        """Fraction of portfolio in BTC (0.0–1.0)."""
        return 1.0 - self.eur_allocation


# ─── Risk Manager ────────────────────────────────────────────────────────────

class RiskManager:
    """
    Cycle-aware risk management gate.

    Every trade attempt passes through can_trade() before execution.
    The risk manager tracks drawdowns, enforces reserve floors, limits
    daily trade counts, and provides emergency sell logic.

    Args:
        config: Bot configuration (risk parameters).
    """

    def __init__(self, config: BotConfig) -> None:
        self._cfg = config.risk
        self._persistence = config.persistence

        # Mutable state
        self._daily_trade_count: int = 0
        self._trade_count_date: str = ""  # ISO date for reset tracking
        self._peak_portfolio_eur: float = 0.0
        self._starting_eur: Optional[float] = None
        self._last_buy_time: float = int(time.time())  # Timestamp of last buy for DCA floor

        self._load_state()

    # ─── Public interface ────────────────────────────────────────────────

    def can_trade(
        self,
        signal: CompositeSignal,
        portfolio: PortfolioState,
        cycle: CycleState,
    ) -> RiskDecision:
        """
        Main risk gate — determines if a trade should proceed.

        Checks in order:
        1. Daily trade limit
        2. Reserve floor (for buys)
        3. Drawdown tolerance (cycle-adjusted)
        4. Capitulation override (for buys in bear markets)

        Returns RiskDecision with allowed flag and reason.
        """
        # Initialize starting EUR on first call
        if self._starting_eur is None:
            self._starting_eur = portfolio.eur_balance + portfolio.btc_value_eur
            self._peak_portfolio_eur = self._starting_eur
            self._save_state()

        # Update peak portfolio value
        current_total = portfolio.total_value_eur
        if current_total > self._peak_portfolio_eur:
            self._peak_portfolio_eur = current_total

        # ── Check 1: Daily trade limit ───────────────────────────────
        self._maybe_reset_daily_count()
        if self._daily_trade_count >= self._cfg.max_daily_trades:
            return RiskDecision(
                allowed=False,
                reason=f"Daily trade limit reached ({self._cfg.max_daily_trades})",
            )

        # ── Direction-specific checks ────────────────────────────────
        if signal.is_buy:
            return self._check_buy_risk(portfolio, cycle)
        elif signal.is_sell:
            return self._check_sell_risk(portfolio, cycle)
        else:
            return RiskDecision(allowed=False, reason="Signal is HOLD — no trade")

    def emergency_sell(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
    ) -> RiskDecision:
        """
        Check if an emergency sell is needed.

        Golden rule: don't emergency sell if price is above the estimated
        cycle floor. Emergency sells only trigger when the portfolio is in
        genuine danger AND price is below a level that suggests structural
        breakdown rather than normal volatility.
        """
        if not self._cfg.enable_golden_rule_floor:
            # Emergency sells fully disabled — backtest-proven that selling
            # during crashes destroys accumulation
            return RiskDecision(
                allowed=False,
                reason="Emergency sells disabled (enable_golden_rule_floor=False)",
            )

        # Golden rule floor protection
        floor = cycle.price_structure.position_in_range
        drawdown = self._portfolio_drawdown(portfolio)

        # Only emergency sell if:
        # 1. Drawdown exceeds 1.5x the phase tolerance
        # 2. Price is in the bottom 15% of the estimated range (near floor)
        extreme_drawdown = drawdown > cycle.drawdown_tolerance * 1.5
        near_floor = floor < 0.15

        if extreme_drawdown and near_floor:
            return RiskDecision(
                allowed=True,
                reason=f"Emergency: drawdown {drawdown:.1%} with price near "
                       f"cycle floor (range position={floor:.2f})",
            )

        if extreme_drawdown and not near_floor:
            logger.warning(
                f"Drawdown extreme ({drawdown:.1%}) but price not near floor "
                f"(range={floor:.2f}). Golden rule holding — no emergency sell."
            )
            return RiskDecision(
                allowed=False,
                reason=f"Golden rule: drawdown {drawdown:.1%} but price "
                       f"above floor (range={floor:.2f}). Holding.",
            )

        return RiskDecision(allowed=False, reason="No emergency condition")

    def get_stops(
        self,
        entry_price: float,
        current_price: float,
        cycle: CycleState,
        atr_value: Optional[float] = None,
    ) -> StopLevels:
        """
        Compute cycle-adjusted stop-loss levels.

        Stop width scales with:
        - ATR (volatility-based distance)
        - Cycle phase (wider in accumulation, tighter in distribution)
        - A configurable multiplier

        Args:
            entry_price: Price at which position was entered.
            current_price: Current market price.
            cycle: Current cycle state.
            atr_value: Current ATR value. If None, uses 2% of price as fallback.
        """
        if atr_value is None or atr_value <= 0:
            atr_value = current_price * 0.02  # 2% fallback

        # Phase-adjusted ATR multiplier
        phase_multipliers = {
            CyclePhase.CAPITULATION: 3.5,     # Wide — expect volatility
            CyclePhase.ACCUMULATION: 3.0,
            CyclePhase.EARLY_BULL: 2.5,
            CyclePhase.GROWTH: 2.0,
            CyclePhase.EUPHORIA: 1.5,         # Tight — protect profits
            CyclePhase.DISTRIBUTION: 1.5,
            CyclePhase.EARLY_BEAR: 2.0,
        }
        phase_mult = phase_multipliers.get(cycle.phase, self._cfg.stop_atr_multiplier)

        atr_width = atr_value * phase_mult
        stop_price = max(0.0, entry_price - atr_width)
        warning_price = max(stop_price, entry_price - atr_width * 0.6)

        return StopLevels(
            stop_price=stop_price,
            warning_price=warning_price,
            atr_width=atr_width,
        )

    def record_trade(self) -> None:
        """Record that a trade was executed. Updates daily count."""
        self._maybe_reset_daily_count()
        self._daily_trade_count += 1
        self._save_state()
        logger.info(f"Trade recorded. Daily count: {self._daily_trade_count}")

    def record_buy(self) -> None:
        """Record that a buy was executed. Updates last_buy_time for DCA floor."""
        self._last_buy_time = time.time()
        self._save_state()
        logger.info(f"Buy recorded. Last buy time updated.")

    @property
    def last_buy_time(self) -> float:
        """Timestamp of last buy for DCA floor tracking."""
        return self._last_buy_time

    def get_drawdown(self, portfolio: PortfolioState) -> float:
        """Current portfolio drawdown from peak (0.0–1.0)."""
        return self._portfolio_drawdown(portfolio)

    # ─── Private checks ──────────────────────────────────────────────────

    def _check_buy_risk(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
    ) -> RiskDecision:
        """Risk checks specific to buy orders."""
        # Reserve floor: never go below reserve_floor_pct of starting EUR
        starting = self._starting_eur or portfolio.total_value_eur
        reserve_floor = starting * self._cfg.reserve_floor_pct

        if portfolio.eur_balance <= reserve_floor:
            return RiskDecision(
                allowed=False,
                reason=f"EUR balance €{portfolio.eur_balance:,.0f} at or below "
                       f"reserve floor €{reserve_floor:,.0f} "
                       f"({self._cfg.reserve_floor_pct:.0%} of starting)",
            )

        # Drawdown check
        drawdown = self._portfolio_drawdown(portfolio)
        tolerance = cycle.drawdown_tolerance

        if drawdown > tolerance:
            # Capitulation override: allow buys during capitulation
            # even at elevated drawdowns (DCA into the bottom)
            if (self._cfg.enable_capitulation_override
                    and cycle.phase in (CyclePhase.CAPITULATION, CyclePhase.ACCUMULATION)):
                logger.info(
                    f"Capitulation override: allowing buy despite "
                    f"drawdown {drawdown:.1%} > tolerance {tolerance:.1%} "
                    f"in {cycle.phase.value} phase"
                )
                return RiskDecision(
                    allowed=True,
                    reason=f"Capitulation override in {cycle.phase.value}: "
                           f"DCA buying despite drawdown {drawdown:.1%}",
                    override_active=True,
                )

            return RiskDecision(
                allowed=False,
                reason=f"Drawdown {drawdown:.1%} exceeds phase tolerance "
                       f"{tolerance:.1%} ({cycle.phase.value})",
            )

        return RiskDecision(
            allowed=True,
            reason=f"Buy allowed: drawdown {drawdown:.1%} within "
                   f"tolerance {tolerance:.1%} ({cycle.phase.value})",
        )

    def _check_sell_risk(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
    ) -> RiskDecision:
        """Risk checks specific to sell orders."""
        # Can't sell if no BTC
        if portfolio.btc_balance <= 0:
            return RiskDecision(
                allowed=False,
                reason="No BTC to sell",
            )

        return RiskDecision(
            allowed=True,
            reason=f"Sell allowed in {cycle.phase.value} phase",
        )

    def _portfolio_drawdown(self, portfolio: PortfolioState) -> float:
        """Calculate current drawdown from peak portfolio value."""
        if self._peak_portfolio_eur <= 0:
            return 0.0
        current = portfolio.total_value_eur
        return max(0.0, 1.0 - current / self._peak_portfolio_eur)

    def _maybe_reset_daily_count(self) -> None:
        """Reset daily trade count if the date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._trade_count_date:
            if self._daily_trade_count > 0:
                logger.info(
                    f"Daily trade count reset (was {self._daily_trade_count})"
                )
            self._daily_trade_count = 0
            self._trade_count_date = today

    # ─── State persistence ───────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self._persistence.get_path("risk_state.json")

    def _load_state(self) -> None:
        """Load persisted risk state."""
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._daily_trade_count = int(data.get("daily_trade_count", 0))
            self._trade_count_date = data.get("trade_count_date", "")
            self._peak_portfolio_eur = float(data.get("peak_portfolio_eur", 0.0))
            starting = data.get("starting_eur")
            self._starting_eur = float(starting) if starting is not None else None
            self._last_buy_time = float(data.get("last_buy_time", 0.0))
            # On first run (no prior trades), set last_buy_time to now
            # to avoid immediate DCA floor triggering
            if self._last_buy_time == 0.0:
                self._last_buy_time = time.time()
            logger.info(f"Loaded risk state: peak=€{self._peak_portfolio_eur:,.0f}")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to load risk state: {exc}")

    def _save_state(self) -> None:
        """Persist risk state to disk."""
        try:
            data = {
                "daily_trade_count": self._daily_trade_count,
                "trade_count_date": self._trade_count_date,
                "peak_portfolio_eur": self._peak_portfolio_eur,
                "starting_eur": self._starting_eur,
                "last_buy_time": self._last_buy_time,
            }
            self._state_path().write_text(json.dumps(data, indent=2))
        except OSError as exc:
            logger.error(f"Failed to save risk state: {exc}")
