"""
Position sizer — determines how much to buy or sell.

Key lessons from previous bot:
- The _combine_adjustments function anchored 70% to 1.0, so even with 7
  factors all saying "reduce", maximum reduction was 15%. Useless.
  FIX: Use geometric mean of adjustment factors — no anchoring.
- Tiered profit taking had a cascade bug where all tiers fired at once.
  FIX: Only the single highest unhit tier fires per evaluation.
- Position sizing was neutered by over-dampening.
  FIX: Hard bounds (min/max) but no artificial anchoring within those bounds.

Design:
- compute_buy_size() returns the EUR amount to spend on a buy.
- compute_sell_tiers() returns which profit tier to execute (if any).
- Adjustment factors combine via geometric mean (pull complexity downward).
- All public methods return typed results, never raise.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import BotConfig, CyclePhase, SizingConfig, PersistenceConfig, VolatilityRegime
from cycle_detector import CycleState
from risk_manager import PortfolioState, RiskDecision
from signal_engine import CompositeSignal

logger = logging.getLogger(__name__)


# ─── Output dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuySize:
    """Recommended buy order size."""
    eur_amount: float         # EUR to spend
    btc_amount: float         # Estimated BTC to receive (at current price)
    fraction_of_capital: float  # What fraction of spendable EUR this represents
    adjustments: dict[str, float]  # Individual adjustment factors for logging
    reason: str


@dataclass(frozen=True)
class SellTier:
    """A profit-taking tier that has been triggered."""
    tier_index: int           # Which tier (0-based)
    threshold_pct: float      # Profit threshold that was crossed
    sell_pct: float           # Fraction of BTC position to sell
    btc_amount: float         # Actual BTC to sell
    reason: str


@dataclass(frozen=True)
class SellDecision:
    """Result of profit-taking evaluation."""
    should_sell: bool
    tier: Optional[SellTier]  # None if no tier triggered
    reason: str


# ─── Position Sizer ──────────────────────────────────────────────────────────

class PositionSizer:
    """
    Determines trade sizes using geometric-mean factor combination.

    For buys: base fraction × geometric_mean(adjustments) → EUR amount.
    For sells: tiered profit taking with one-tier-per-cycle guard.

    Args:
        config: Bot configuration (sizing parameters, Kraken limits).
    """

    def __init__(self, config: BotConfig) -> None:
        self._cfg = config.sizing
        self._cfg_risk = config.risk
        self._kraken_cfg = config.kraken
        self._persistence = config.persistence
        self._hit_tiers: set[int] = set()  # Tiers already triggered this cycle
        self._last_cycle_phase: Optional[CyclePhase] = None
        self._load_state()

    # ─── Public interface ────────────────────────────────────────────────

    def compute_buy_size(
        self,
        signal: CompositeSignal,
        portfolio: PortfolioState,
        cycle: CycleState,
        risk: RiskDecision,
    ) -> BuySize:
        """
        Compute how much EUR to spend on a buy order.

        Base: fixed fraction of spendable EUR (eur_balance - reserve).
        Adjustments (geometric mean, NOT anchored):
        - Signal score scaling
        - Volatility adjustment
        - Cycle phase multiplier
        - Drawdown protection
        - Risk override scaling (reduced if capitulation override active)
        - Value averaging: buy more when price is below 200-day MA
        - Acceleration brake: reduce size when price is running up fast
        """
        # Spendable capital: EUR minus reserve.
        # Reserve is based on EUR balance only — using total portfolio
        # (including BTC value) creates a death spiral where accumulated
        # BTC inflates the reserve and prevents further buying.
        reserve = portfolio.eur_balance * self._cfg_risk.reserve_floor_pct
        spendable = max(0.0, portfolio.eur_balance - reserve)

        if spendable <= 0:
            return BuySize(
                eur_amount=0.0, btc_amount=0.0,
                fraction_of_capital=0.0, adjustments={},
                reason="No spendable EUR after reserve",
            )

        # Base size
        base_eur = spendable * self._cfg.base_fraction

        # Compute individual adjustment factors
        adjustments: dict[str, float] = {}

        # 1. Signal score: stronger signal → larger position
        score_factor = self._signal_score_factor(signal.score)
        adjustments["signal_score"] = score_factor

        # 2. Volatility: reduce in high vol, increase in low vol
        vol_factor = self._volatility_factor(cycle)
        adjustments["volatility"] = vol_factor

        # 3. Cycle phase: from CycleState
        phase_factor = cycle.position_size_multiplier
        adjustments["cycle_phase"] = phase_factor

        # 4. Drawdown protection: reduce sizing during drawdowns
        drawdown_factor = self._drawdown_factor(portfolio)
        adjustments["drawdown"] = drawdown_factor

        # 5. Risk override: if capitulation override is active, scale down
        override_factor = 0.6 if risk.override_active else 1.0
        adjustments["risk_override"] = override_factor

        # 6. Value averaging: buy more when price is below 200-day MA
        value_avg_factor = self._value_averaging_factor(cycle)
        adjustments["value_avg"] = value_avg_factor

        # 7. Acceleration brake: reduce when price is running hot
        brake_factor = self._acceleration_brake_factor(cycle)
        adjustments["acceleration_brake"] = brake_factor

        # Combine via geometric mean (no anchoring!)
        combined = self._geometric_mean(list(adjustments.values()))
        adjusted_eur = base_eur * combined

        # Hard bounds
        max_eur = spendable * self._kraken_cfg.max_order_pct
        min_btc = self._kraken_cfg.min_order_btc
        min_eur = min_btc * portfolio.btc_price if portfolio.btc_price > 0 else 10.0

        # Apply factor bounds from config
        lower = base_eur * self._cfg.min_adjustment
        upper = base_eur * self._cfg.max_adjustment
        adjusted_eur = max(lower, min(upper, adjusted_eur))

        # Apply hard bounds
        adjusted_eur = min(adjusted_eur, max_eur)

        if adjusted_eur < min_eur:
            return BuySize(
                eur_amount=0.0, btc_amount=0.0,
                fraction_of_capital=0.0, adjustments=adjustments,
                reason=f"Adjusted size €{adjusted_eur:.0f} below minimum €{min_eur:.0f}",
            )

        btc_amount = adjusted_eur / portfolio.btc_price if portfolio.btc_price > 0 else 0.0

        return BuySize(
            eur_amount=adjusted_eur,
            btc_amount=btc_amount,
            fraction_of_capital=adjusted_eur / spendable if spendable > 0 else 0.0,
            adjustments=adjustments,
            reason=f"Buy €{adjusted_eur:,.0f} (base €{base_eur:,.0f} × "
                   f"combined {combined:.3f})",
        )

    def compute_sell_tiers(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
        avg_entry_price: float,
    ) -> SellDecision:
        """
        Evaluate profit-taking tiers.

        Key fix: only the SINGLE HIGHEST unhit tier fires per evaluation.
        Prevents the cascade bug where a price jump triggers all tiers
        simultaneously, selling 95% of the position.

        Tiers reset when cycle phase changes (new accumulation after
        distribution means fresh tiers).

        Args:
            portfolio: Current portfolio state.
            cycle: Current cycle state.
            avg_entry_price: Volume-weighted average entry price.
        """
        # Reset tiers on cycle phase transition
        if self._last_cycle_phase is not None and cycle.phase != self._last_cycle_phase:
            if self._hit_tiers:
                logger.info(
                    f"Cycle phase changed {self._last_cycle_phase.value} → "
                    f"{cycle.phase.value}: resetting profit tiers "
                    f"(was: {self._hit_tiers})"
                )
                self._hit_tiers.clear()
                self._save_state()
        self._last_cycle_phase = cycle.phase

        # Only take profits in appropriate phases
        if not cycle.profit_taking_active:
            return SellDecision(
                should_sell=False, tier=None,
                reason=f"Profit taking inactive in {cycle.phase.value} phase",
            )

        if portfolio.btc_balance <= 0 or avg_entry_price <= 0:
            return SellDecision(
                should_sell=False, tier=None,
                reason="No BTC position or no entry price",
            )

        current_price = portfolio.btc_price
        profit_pct = (current_price - avg_entry_price) / avg_entry_price

        if profit_pct <= 0:
            return SellDecision(
                should_sell=False, tier=None,
                reason=f"No profit (P&L: {profit_pct:+.1%})",
            )

        # Find the highest unhit tier that has been crossed
        # Use phase-specific tiers if configured, otherwise defaults
        phase_val = cycle.phase.value
        tiers = self._cfg.phase_profit_tiers.get(phase_val, self._cfg.profit_tiers)
        best_tier_idx: Optional[int] = None

        for i, tier in enumerate(tiers):
            threshold = tier["threshold"]
            if profit_pct >= threshold and i not in self._hit_tiers:
                best_tier_idx = i  # Keep iterating to find highest

        if best_tier_idx is None:
            # All crossed tiers already hit, or no tier crossed
            unhit = [i for i in range(len(tiers)) if i not in self._hit_tiers]
            if unhit:
                next_threshold = tiers[unhit[0]]["threshold"]
                return SellDecision(
                    should_sell=False, tier=None,
                    reason=f"Profit {profit_pct:+.1%}, next tier at "
                           f"+{next_threshold:.0%}",
                )
            return SellDecision(
                should_sell=False, tier=None,
                reason=f"All {len(tiers)} profit tiers already taken",
            )

        # Fire ONLY the highest unhit tier
        tier_cfg = tiers[best_tier_idx]
        sell_pct = tier_cfg["sell_pct"]
        btc_to_sell = portfolio.btc_balance * sell_pct

        # Ensure minimum order
        if btc_to_sell < self._kraken_cfg.min_order_btc:
            return SellDecision(
                should_sell=False, tier=None,
                reason=f"Tier {best_tier_idx} sell amount "
                       f"{btc_to_sell:.8f} BTC below minimum",
            )

        tier = SellTier(
            tier_index=best_tier_idx,
            threshold_pct=tier_cfg["threshold"],
            sell_pct=sell_pct,
            btc_amount=btc_to_sell,
            reason=f"Tier {best_tier_idx}: profit {profit_pct:+.1%} crossed "
                   f"+{tier_cfg['threshold']:.0%} → sell {sell_pct:.0%} "
                   f"({btc_to_sell:.8f} BTC)",
        )

        return SellDecision(should_sell=True, tier=tier, reason=tier.reason)

    def mark_tier_hit(self, tier_index: int) -> None:
        """Record that a profit tier has been executed."""
        self._hit_tiers.add(tier_index)
        self._save_state()
        logger.info(f"Profit tier {tier_index} marked as hit. Active: {self._hit_tiers}")

    @property
    def hit_tiers(self) -> frozenset[int]:
        """Currently hit profit-taking tiers."""
        return frozenset(self._hit_tiers)

    # ─── Adjustment factor calculations ──────────────────────────────────

    @staticmethod
    def _signal_score_factor(score: float) -> float:
        """
        Map signal score (-100..+100) to a sizing factor.

        Score 0 → factor 1.0 (neutral)
        Score +100 → factor ~1.8
        Score -50 → factor ~0.5 (but we shouldn't be buying on negative scores)
        Score +20 (min buy threshold) → factor ~1.15
        """
        # Sigmoid-ish mapping centered at 0
        # factor = 1 + score / 150, clamped to [0.3, 2.0]
        factor = 1.0 + score / 150.0
        return max(0.3, min(2.0, factor))

    @staticmethod
    def _volatility_factor(cycle: CycleState) -> float:
        """
        Volatility adjustment: reduce in high vol, increase in low vol.

        Uses the volatility score from cycle detector (-1 to +1, where
        positive = compression/opportunity, negative = extreme/risk).
        """
        # Map vol_score: +1 → factor 1.3, 0 → 1.0, -1 → 0.6
        factor = 1.0 + cycle.volatility_score * 0.3
        return max(0.5, min(1.5, factor))

    def _drawdown_factor(self, portfolio: PortfolioState) -> float:
        """
        Reduce position size during drawdowns.

        Linear reduction: 0% drawdown → 1.0, 30%+ drawdown → 0.4.
        """
        starting = self._starting_eur_or_current(portfolio)
        if starting <= 0:
            return 1.0
        current = portfolio.total_value_eur
        drawdown = max(0.0, 1.0 - current / starting)

        # Linear interpolation: 0% → 1.0, 30% → 0.4
        factor = 1.0 - drawdown * 2.0
        return max(0.4, min(1.0, factor))

    @staticmethod
    def _starting_eur_or_current(portfolio: PortfolioState) -> float:
        """Get starting EUR, falling back to current total."""
        if portfolio.starting_eur > 0:
            return portfolio.starting_eur
        return portfolio.total_value_eur

    def _value_averaging_factor(self, cycle: CycleState) -> float:
        """
        Value averaging: buy more when price is below 200-day MA.

        Uses distance_from_200d_ma from CycleState.price_structure.
        Negative distance = below MA = boost buying.
        Positive distance = above MA = neutral or slight reduction.

        Boost formula: 1 + max_boost * (1 - exp(-sensitivity * distance_below_ma))
        """
        if not self._cfg.value_avg_enabled:
            return 1.0

        price_struct = cycle.price_structure
        dist = price_struct.distance_from_200d_ma  # +0.15 = 15% above, -0.10 = 10% below

        if dist >= 0:
            # Above 200d MA: slight reduction when far above
            # At 50%+ above MA, reduce to 0.7
            if dist > 0.3:
                return max(0.7, 1.0 - (dist - 0.3) * 0.75)
            return 1.0
        else:
            # Below 200d MA: boost buying proportionally
            below = abs(dist)  # e.g. 0.15 for 15% below MA
            boost = self._cfg.value_avg_max_boost * (
                1 - math.exp(-self._cfg.value_avg_sensitivity * below)
            )
            return 1.0 + boost

    def _acceleration_brake_factor(self, cycle: CycleState) -> float:
        """
        FOMO protection: reduce buy size when price is running up fast.

        Triggers when:
        - Volatility regime is elevated or extreme, AND
        - Price is near upper Bollinger band (position_in_range > 0.85), OR
        - Momentum is strongly bullish (chasing a run)

        The idea: if everyone is buying, you don't want to chase.
        Let the run happen, accumulate on the pullback.
        """
        if not self._cfg.acceleration_brake_enabled:
            return 1.0

        vol_regime = cycle.volatility_regime

        # Only brake in elevated/extreme volatility
        if vol_regime not in (VolatilityRegime.ELEVATED, VolatilityRegime.EXTREME):
            return 1.0

        # Check price position
        pos = getattr(cycle.price_structure, 'position_in_range', 0.5)
        momentum_score = cycle.momentum_score

        braking = False

        # Price near top of range in high vol = chasing
        if pos > 0.85:
            braking = True

        # Strong bullish momentum in high vol = FOMO territory
        if momentum_score > 0.6 and vol_regime == VolatilityRegime.EXTREME:
            braking = True

        if braking:
            logger.debug(
                f"Acceleration brake: vol={vol_regime.value} "
                f"pos={pos:.2f} momentum={momentum_score:.2f}"
            )
            return self._cfg.acceleration_brake_factor  # e.g. 0.4

        return 1.0

    @staticmethod
    def _geometric_mean(factors: list[float]) -> float:
        """
        Geometric mean of adjustment factors.

        This is the key fix: no anchoring to 1.0. If all factors say
        "reduce" (< 1.0), the combined factor genuinely reduces.
        If all say "increase" (> 1.0), it genuinely increases.

        Handles zero/negative factors by clamping to 0.01.
        """
        if not factors:
            return 1.0

        # Clamp to avoid log(0) or log(negative)
        clamped = [max(0.01, f) for f in factors]

        # Geometric mean = exp(mean(log(factors)))
        log_sum = sum(math.log(f) for f in clamped)
        return math.exp(log_sum / len(clamped))

    # ─── State persistence ───────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self._persistence.get_path("sizer_state.json")

    def _load_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._hit_tiers = set(data.get("hit_tiers", []))
            phase = data.get("last_cycle_phase")
            if phase:
                self._last_cycle_phase = CyclePhase(phase)
            logger.info(f"Loaded sizer state: hit_tiers={self._hit_tiers}")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to load sizer state: {exc}")

    def _save_state(self) -> None:
        try:
            data = {
                "hit_tiers": list(self._hit_tiers),
                "last_cycle_phase": (
                    self._last_cycle_phase.value
                    if self._last_cycle_phase else None
                ),
            }
            self._state_path().write_text(json.dumps(data, indent=2))
        except OSError as exc:
            logger.error(f"Failed to save sizer state: {exc}")
