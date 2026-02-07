"""
CycleAnalyzer - Bitcoin Halving Cycle Awareness

OUSTERHOUT PRINCIPLE: Deep module with simple interface.

This module provides a single source of truth for where we are in the
Bitcoin 4-year halving cycle, and translates that into actionable trading
parameters that the rest of the bot consumes.

PUBLIC INTERFACE (Simple):
    get_cycle_phase(current_price_eur) -> CyclePhase
    get_cycle_adjustments(current_price_eur) -> CycleAdjustments
    get_accumulation_score(current_price_eur) -> float
    get_cycle_summary(current_price_eur) -> Dict

PRIVATE IMPLEMENTATION (Complex):
    Halving date calculations
    Phase boundary detection with price-aware transitions
    Historical pattern matching across 4 complete cycles
    Diminishing returns modeling
    Drawdown floor estimation (golden rule vs 80% drawdown tension)
    ATH window timing analysis

KEY PATTERNS ENCODED (from EUR-denominated analysis):
    1. Diminishing returns: 88x -> 28x -> 7.4x -> 1.8x per cycle
    2. Bear drawdowns: ~75-84% consistently (avg 80%)
    3. ATL > prev ATH (the "golden rule") — nearly failed in EUR Cycle 3
    4. Days to ATH: 368 -> 525 -> 549 -> 534 (avg ~494)
    5. Pre-halving rally emergence (Cycle 4 ATH before halving — unprecedented)
    6. Critical tension: 80% drawdown vs golden rule — one must break in Cycle 4+
    7. EUR/USD divergence effect amplifies bears, dampens bulls
"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from logger_config import logger


# =============================================================================
# HALVING CYCLE CONSTANTS (Verified historical data, EUR-denominated)
# =============================================================================

HALVINGS = [
    {
        "number": 1,
        "date": datetime(2012, 11, 28),
        "block": 210_000,
        "reward_before": 50.0,
        "reward_after": 25.0,
        "price_eur_at_halving": 9.5,
    },
    {
        "number": 2,
        "date": datetime(2016, 7, 9),
        "block": 420_000,
        "reward_before": 25.0,
        "reward_after": 12.5,
        "price_eur_at_halving": 586.0,
    },
    {
        "number": 3,
        "date": datetime(2020, 5, 11),
        "block": 630_000,
        "reward_before": 12.5,
        "reward_after": 6.25,
        "price_eur_at_halving": 7_860.0,
    },
    {
        "number": 4,
        "date": datetime(2024, 4, 20),
        "block": 840_000,
        "reward_before": 6.25,
        "reward_after": 3.125,
        "price_eur_at_halving": 59_700.0,
    },
]

# Estimated next halving (~April 2028, based on ~210,000 blocks at ~10 min)
NEXT_HALVING_ESTIMATE = datetime(2028, 4, 15)

# Historical cycle peaks and troughs (EUR-denominated)
CYCLE_HISTORY = [
    {
        "cycle": 1,
        "ath_eur": 835,
        "ath_date": datetime(2013, 11, 30),
        "atl_eur": 148,
        "atl_date": datetime(2015, 1, 14),
        "days_halving_to_ath": 368,
        "drawdown_pct": 0.823,
        "multiplier_from_halving": 87.9,
    },
    {
        "cycle": 2,
        "ath_eur": 16_680,
        "ath_date": datetime(2017, 12, 16),
        "atl_eur": 2_720,
        "atl_date": datetime(2018, 12, 15),
        "days_halving_to_ath": 525,
        "drawdown_pct": 0.837,
        "multiplier_from_halving": 28.5,
    },
    {
        "cycle": 3,
        "ath_eur": 58_100,
        "ath_date": datetime(2021, 11, 10),
        "atl_eur": 15_030,
        "atl_date": datetime(2022, 11, 21),
        "days_halving_to_ath": 549,
        "drawdown_pct": 0.741,
        "multiplier_from_halving": 7.4,
    },
    {
        "cycle": 4,
        "ath_eur": 107_662,
        "ath_date": datetime(2025, 10, 6),
        "atl_eur": None,  # TBD — cycle still in progress
        "atl_date": None,
        "days_halving_to_ath": 534,
        "drawdown_pct": None,
        "multiplier_from_halving": 1.8,
    },
]

# Derived pattern constants
AVG_DAYS_TO_ATH = 494  # mean(368, 525, 549, 534)
AVG_BEAR_DRAWDOWN = 0.80  # mean(0.823, 0.837, 0.741)
DIMINISHING_RETURN_RATIO = 0.27  # each cycle ~27% of previous multiplier
PREV_CYCLE_ATH_EUR = 58_100  # Cycle 3 ATH — golden rule floor for Cycle 4


class CyclePhase(Enum):
    """
    Bitcoin halving cycle phases.
    Each phase has distinct risk/reward characteristics and optimal strategies.
    """

    # Bull phases (post-halving)
    POST_HALVING_ACCUMULATION = "post_halving_accumulation"  # 0-180d after halving
    GROWTH = "growth"  # 180-365d after halving
    EUPHORIA = "euphoria"  # 365-550d, approaching ATH zone
    DISTRIBUTION = "distribution"  # 550+d or well past ATH

    # Bear phases
    BEAR_EARLY = "bear_early"  # 0-6 months after cycle ATH
    BEAR_CAPITULATION = "bear_capitulation"  # 6-18 months after ATH, deepest pain

    # Transition
    PRE_HALVING = "pre_halving"  # 6 months before next halving


@dataclass(frozen=True)
class CycleAdjustments:
    """
    Cycle-aware parameter adjustments for the trading bot.

    These are MULTIPLIERS applied to existing strategy parameters.
    A value of 1.0 means no change from default behavior.
    """

    phase: CyclePhase
    phase_description: str

    # Position sizing: 0.0 = don't trade, 2.0 = double normal position
    position_size_multiplier: float

    # Buy aggressiveness: higher = more eager to buy dips
    buy_aggressiveness: float

    # Sell reluctance: higher = more reluctant to sell (accumulation mode)
    sell_reluctance: float

    # Stop loss width: higher = wider stops (more room for volatility)
    stop_loss_width: float

    # Take profit width: higher = wider take-profit targets
    take_profit_width: float

    # Risk tolerance: 0.0 = ultra conservative, 1.0 = maximum risk
    risk_tolerance: float

    # DCA intensity: how aggressively to dollar-cost-average
    dca_intensity: float

    # Estimated floor price (EUR) from cycle patterns
    estimated_floor_eur: float

    # Estimated ceiling (EUR) from diminishing returns
    estimated_ceiling_eur: float

    # Confidence in this phase assessment (0.0 to 1.0)
    confidence: float


class CycleAnalyzer:
    """
    Analyzes current position in Bitcoin's halving cycle and provides
    actionable adjustments for all trading parameters.

    GUARANTEE: All public methods always return valid data, never None.
    """

    def __init__(self, current_cycle: int = 4):
        self._cycle_num = current_cycle
        self._halving = HALVINGS[current_cycle - 1]
        self._halving_date = self._halving["date"]
        self._next_halving = NEXT_HALVING_ESTIMATE

        logger.info(
            f"CycleAnalyzer initialized: Cycle {current_cycle}, "
            f"Halving: {self._halving_date:%Y-%m-%d}, "
            f"Next halving est: {self._next_halving:%Y-%m-%d}"
        )

    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================

    def get_cycle_phase(self, current_price_eur: Optional[float] = None) -> CyclePhase:
        """
        Determine current cycle phase using time + price signals.

        GUARANTEE: Always returns a valid CyclePhase.
        """
        now = datetime.now()
        days_since_halving = (now - self._halving_date).days
        days_until_next = (self._next_halving - now).days

        # Pre-halving takes priority if close to next halving
        if 0 < days_until_next <= 180:
            return CyclePhase.PRE_HALVING

        # Price-aware bear detection: if we have price data and an ATH
        cycle_data = CYCLE_HISTORY[self._cycle_num - 1]
        if current_price_eur and cycle_data["ath_eur"]:
            drawdown = 1 - (current_price_eur / cycle_data["ath_eur"])

            # Past the ATH window AND significantly below ATH = bear market
            if drawdown > 0.30 and days_since_halving > AVG_DAYS_TO_ATH:
                if drawdown > 0.50:
                    return CyclePhase.BEAR_CAPITULATION
                return CyclePhase.BEAR_EARLY

        # Time-based phase detection for bull market
        if days_since_halving <= 180:
            return CyclePhase.POST_HALVING_ACCUMULATION
        elif days_since_halving <= 365:
            return CyclePhase.GROWTH
        elif days_since_halving <= 550:
            return CyclePhase.EUPHORIA
        else:
            return CyclePhase.DISTRIBUTION

    def get_cycle_adjustments(
        self, current_price_eur: Optional[float] = None
    ) -> CycleAdjustments:
        """
        Get cycle-aware parameter adjustments for all trading decisions.

        GUARANTEE: Always returns valid CycleAdjustments with safe defaults.
        """
        phase = self.get_cycle_phase(current_price_eur)
        floor_eur, ceiling_eur = self._estimate_price_bounds()

        adjustments = _PHASE_ADJUSTMENTS[phase]

        result = CycleAdjustments(
            phase=phase,
            phase_description=adjustments["description"],
            position_size_multiplier=adjustments["position_size"],
            buy_aggressiveness=adjustments["buy_aggr"],
            sell_reluctance=adjustments["sell_reluct"],
            stop_loss_width=adjustments["sl_width"],
            take_profit_width=adjustments["tp_width"],
            risk_tolerance=adjustments["risk_tol"],
            dca_intensity=adjustments["dca"],
            estimated_floor_eur=floor_eur,
            estimated_ceiling_eur=ceiling_eur,
            confidence=adjustments["confidence"],
        )

        logger.info(
            f"Cycle: {phase.value} | Pos={result.position_size_multiplier:.1f}x | "
            f"Buy={result.buy_aggressiveness:.1f}x | DCA={result.dca_intensity:.1f}x | "
            f"Floor=€{floor_eur:,.0f} | Ceiling=€{ceiling_eur:,.0f}"
        )

        return result

    def get_accumulation_score(self, current_price_eur: float) -> float:
        """
        Calculate 0-100 accumulation score.

        Higher = better time to accumulate.
        Factors: cycle phase, distance from floor/ceiling, drawdown from ATH.

        GUARANTEE: Always returns a float between 0 and 100.
        """
        adjustments = self.get_cycle_adjustments(current_price_eur)
        floor_eur = adjustments.estimated_floor_eur
        ceiling_eur = adjustments.estimated_ceiling_eur

        # Phase component (0-40 points)
        phase_scores = {
            CyclePhase.BEAR_CAPITULATION: 40,
            CyclePhase.POST_HALVING_ACCUMULATION: 35,
            CyclePhase.PRE_HALVING: 30,
            CyclePhase.BEAR_EARLY: 25,
            CyclePhase.GROWTH: 20,
            CyclePhase.EUPHORIA: 8,
            CyclePhase.DISTRIBUTION: 3,
        }
        phase_score = phase_scores.get(adjustments.phase, 15)

        # Price position component (0-35 points): closer to floor = higher
        if ceiling_eur > floor_eur and current_price_eur > 0:
            price_range = ceiling_eur - floor_eur
            position_in_range = (current_price_eur - floor_eur) / price_range
            position_in_range = max(0.0, min(1.0, position_in_range))
            price_score = (1.0 - position_in_range) * 35
        else:
            price_score = 15

        # Drawdown bonus (0-25 points): deeper drawdown = better buy
        cycle_ath = CYCLE_HISTORY[self._cycle_num - 1]["ath_eur"]
        if cycle_ath and current_price_eur < cycle_ath:
            drawdown = 1 - (current_price_eur / cycle_ath)
            drawdown_score = min(25, drawdown * 50)
        else:
            drawdown_score = 0

        total = max(0, min(100, phase_score + price_score + drawdown_score))

        logger.debug(
            f"Accumulation score: {total:.0f}/100 "
            f"(phase={phase_score}, price={price_score:.0f}, drawdown={drawdown_score:.0f})"
        )
        return total

    def get_cycle_summary(self, current_price_eur: Optional[float] = None) -> Dict:
        """Get comprehensive cycle status for logging and dashboards."""
        now = datetime.now()
        days_since = (now - self._halving_date).days
        days_until = (self._next_halving - now).days
        cycle_length = (self._next_halving - self._halving_date).days
        progress = min(100.0, (days_since / cycle_length) * 100)

        phase = self.get_cycle_phase(current_price_eur)
        adj = self.get_cycle_adjustments(current_price_eur)
        floor_eur, ceiling_eur = self._estimate_price_bounds()

        summary = {
            "cycle_number": self._cycle_num,
            "halving_date": self._halving_date.isoformat(),
            "next_halving_date": self._next_halving.isoformat(),
            "days_since_halving": days_since,
            "days_until_next_halving": days_until,
            "cycle_progress_pct": round(progress, 1),
            "current_phase": phase.value,
            "phase_description": adj.phase_description,
            "block_reward": self._halving["reward_after"],
            "price_at_halving_eur": self._halving["price_eur_at_halving"],
            "estimated_floor_eur": floor_eur,
            "estimated_ceiling_eur": ceiling_eur,
            "avg_days_to_ath": AVG_DAYS_TO_ATH,
            "days_past_avg_ath": max(0, days_since - AVG_DAYS_TO_ATH),
            "historical_drawdown_avg": AVG_BEAR_DRAWDOWN,
            "position_size_multiplier": adj.position_size_multiplier,
            "buy_aggressiveness": adj.buy_aggressiveness,
            "dca_intensity": adj.dca_intensity,
            "confidence": adj.confidence,
        }

        if current_price_eur:
            halving_price = self._halving["price_eur_at_halving"]
            summary["current_price_eur"] = current_price_eur
            summary["accumulation_score"] = self.get_accumulation_score(current_price_eur)
            summary["gain_from_halving_pct"] = round(
                ((current_price_eur - halving_price) / halving_price) * 100, 1
            )
            cycle_ath = CYCLE_HISTORY[self._cycle_num - 1]["ath_eur"]
            if cycle_ath:
                summary["drawdown_from_ath_pct"] = round(
                    (1 - current_price_eur / cycle_ath) * 100, 1
                )

        return summary

    def is_in_ath_window(self) -> bool:
        """Check if we're in the historical ATH timing window (365-550 days)."""
        days = (datetime.now() - self._halving_date).days
        return 365 <= days <= 550

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _estimate_price_bounds(self) -> Tuple[float, float]:
        """
        Estimate floor and ceiling prices based on historical cycle patterns.

        Floor uses the HIGHER of:
            Model A: 80% drawdown from current cycle ATH
            Model B: Previous cycle ATH (golden rule)
        This is conservative — respects the golden rule unless forced to break.

        Ceiling uses current ATH + 15% margin (or 2.5x halving price if no ATH yet).
        """
        current_cycle = CYCLE_HISTORY[self._cycle_num - 1]
        halving_price = self._halving["price_eur_at_halving"]

        # Floor estimation
        if current_cycle["ath_eur"]:
            floor_drawdown = current_cycle["ath_eur"] * (1 - AVG_BEAR_DRAWDOWN)
        else:
            floor_drawdown = halving_price * 0.5

        floor_golden_rule = PREV_CYCLE_ATH_EUR  # €58,100
        floor_eur = max(floor_drawdown, floor_golden_rule)

        # Ceiling estimation
        if current_cycle["ath_eur"]:
            ceiling_eur = current_cycle["ath_eur"] * 1.15
        else:
            ceiling_eur = halving_price * 2.5

        return round(floor_eur, 0), round(ceiling_eur, 0)


# =============================================================================
# PHASE PARAMETER TABLES
# =============================================================================
# Separated from the class for clarity and easy tuning.
# Each phase defines multipliers applied to the bot's base parameters.

_PHASE_ADJUSTMENTS = {
    CyclePhase.POST_HALVING_ACCUMULATION: {
        "description": (
            "Early post-halving: historically the best accumulation window. "
            "Price typically consolidates before a major move. Maximize buying."
        ),
        "position_size": 1.5,
        "buy_aggr": 1.8,
        "sell_reluct": 2.0,
        "sl_width": 1.5,
        "tp_width": 2.0,
        "risk_tol": 0.7,
        "dca": 1.5,
        "confidence": 0.75,
    },
    CyclePhase.GROWTH: {
        "description": (
            "Growth phase: price building momentum toward ATH zone. "
            "Continue accumulating but be more selective on entries."
        ),
        "position_size": 1.2,
        "buy_aggr": 1.3,
        "sell_reluct": 1.5,
        "sl_width": 1.2,
        "tp_width": 1.5,
        "risk_tol": 0.6,
        "dca": 1.2,
        "confidence": 0.70,
    },
    CyclePhase.EUPHORIA: {
        "description": (
            "Euphoria phase: inside the historical ATH window (368-549 days). "
            "Reduce new buys, tighten stops, consider partial profit-taking."
        ),
        "position_size": 0.5,
        "buy_aggr": 0.4,
        "sell_reluct": 0.6,
        "sl_width": 0.7,
        "tp_width": 0.8,
        "risk_tol": 0.3,
        "dca": 0.5,
        "confidence": 0.65,
    },
    CyclePhase.DISTRIBUTION: {
        "description": (
            "Distribution phase: past typical ATH window. Risk of cycle top. "
            "Minimal new buying, protect capital, consider reducing exposure."
        ),
        "position_size": 0.3,
        "buy_aggr": 0.2,
        "sell_reluct": 0.3,
        "sl_width": 0.5,
        "tp_width": 0.5,
        "risk_tol": 0.2,
        "dca": 0.3,
        "confidence": 0.60,
    },
    CyclePhase.BEAR_EARLY: {
        "description": (
            "Early bear: price declining from ATH. Historically drops ~40-60% "
            "in first 6 months. Start accumulating cautiously with small sizes."
        ),
        "position_size": 0.8,
        "buy_aggr": 1.0,
        "sell_reluct": 0.8,
        "sl_width": 1.3,
        "tp_width": 1.0,
        "risk_tol": 0.4,
        "dca": 0.8,
        "confidence": 0.55,
    },
    CyclePhase.BEAR_CAPITULATION: {
        "description": (
            "Bear capitulation: maximum fear, historically -75% to -84% from ATH. "
            "THIS IS THE BEST ACCUMULATION ZONE. Be greedy when others are fearful."
        ),
        "position_size": 2.0,
        "buy_aggr": 2.0,
        "sell_reluct": 3.0,
        "tp_width": 3.0,
        "sl_width": 2.0,
        "risk_tol": 0.8,
        "dca": 2.0,
        "confidence": 0.70,
    },
    CyclePhase.PRE_HALVING: {
        "description": (
            "Pre-halving window: 6 months before supply cut. Historically "
            "strong. Cycle 4 saw a pre-halving ATH (ETF-driven structural change)."
        ),
        "position_size": 1.3,
        "buy_aggr": 1.4,
        "sell_reluct": 1.8,
        "sl_width": 1.3,
        "tp_width": 1.5,
        "risk_tol": 0.6,
        "dca": 1.3,
        "confidence": 0.65,
    },
}
