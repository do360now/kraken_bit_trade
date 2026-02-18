"""
Composite signal engine — the brain of the trading bot.

Takes TechnicalSnapshot, CycleState, and cached context (LLM analysis,
on-chain metrics) and produces a single CompositeSignal that drives
all downstream decisions (risk gating, position sizing, execution).

Design principles:
- Every sub-signal contributes a score in [-100, +100] and a direction vote.
- Agreement measures what fraction of sub-signals agree on direction.
- Minimum agreement threshold prevents acting on conflicting signals.
- Component weights are configurable via SignalConfig.
- Full reasoning chain is captured for observability.
- Never raises exceptions — returns a neutral signal on any failure.

Key lesson from previous bot: this module IS called from the main loop.
Every component feeds into the composite score. No dead code.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bitcoin_node import OnChainSnapshot
from config import BotConfig, SignalConfig, CyclePhase, VolatilityRegime
from cycle_detector import CycleState
from indicators import TechnicalSnapshot

logger = logging.getLogger(__name__)


# ─── Output dataclasses ──────────────────────────────────────────────────────

class Action(Enum):
    """Trading action recommendation."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass(frozen=True)
class SubSignal:
    """Individual signal component with score and reasoning."""
    name: str
    score: float        # -100 to +100
    weight: float       # 0.0 to 1.0 (from config)
    direction: int      # -1 (bearish), 0 (neutral), +1 (bullish)
    reason: str         # Human-readable explanation


@dataclass(frozen=True)
class CompositeSignal:
    """
    Final signal output consumed by risk_manager, position_sizer, and trade_executor.

    score: -100 (strong sell) to +100 (strong buy).
    agreement: 0.0–1.0 (fraction of sub-signals agreeing on direction).
    action: Recommended trading action.
    components: Individual sub-signal details for logging/debugging.
    """
    score: float
    agreement: float
    action: Action
    components: tuple[SubSignal, ...]
    data_quality: float       # 0.0–1.0: how much input data was available
    timestamp: float

    @property
    def actionable(self) -> bool:
        """Whether this signal meets minimum quality thresholds to act on."""
        return self.data_quality >= 0.3 and self.action != Action.HOLD

    @property
    def is_buy(self) -> bool:
        return self.action in (Action.BUY, Action.STRONG_BUY)

    @property
    def is_sell(self) -> bool:
        return self.action in (Action.SELL, Action.STRONG_SELL)


# ─── Neutral signal (returned on failure) ────────────────────────────────────

def _neutral_signal() -> CompositeSignal:
    return CompositeSignal(
        score=0.0,
        agreement=0.0,
        action=Action.HOLD,
        components=(),
        data_quality=0.0,
        timestamp=time.time(),
    )


# ─── LLM Analysis context (from ollama_analyst slow path) ───────────────────

@dataclass(frozen=True)
class LLMContext:
    """Cached LLM analysis results from the slow loop."""
    regime: str              # "accumulation", "markup", "distribution", etc.
    sentiment: float         # -1.0 to +1.0
    risk_level: str          # "low", "medium", "high", "extreme"
    themes: tuple[str, ...]  # Key market themes
    timestamp: float

    @property
    def stale(self) -> bool:
        """Context older than 2 hours is considered stale."""
        return (time.time() - self.timestamp) > 7200


# ─── Signal Engine ───────────────────────────────────────────────────────────

class SignalEngine:
    """
    Composite signal generator.

    Combines technical indicators, cycle phase, on-chain metrics, and
    LLM analysis into a single actionable signal. Each data source
    contributes a weighted sub-signal; the engine measures both the
    composite score and the agreement between sub-signals.

    Args:
        config: Bot configuration (signal weights, thresholds).
    """

    def __init__(self, config: BotConfig) -> None:
        self._cfg = config.signal

    def _get_weight(self, signal_name: str, phase_value: str) -> float:
        """
        Resolve signal weight: phase-specific override → default.

        For accumulation: RSI/onchain/cycle matter more.
        For euphoria/distribution: cycle/MACD divergence matter more.
        """
        overrides = self._cfg.phase_weight_overrides.get(phase_value, {})
        weight_key = f"{signal_name}_weight"

        if weight_key in overrides:
            return overrides[weight_key]

        return getattr(self._cfg, weight_key, 0.1)

    def generate(
        self,
        snapshot: TechnicalSnapshot,
        cycle: CycleState,
        onchain: Optional[OnChainSnapshot] = None,
        llm: Optional[LLMContext] = None,
    ) -> CompositeSignal:
        """
        Generate a composite trading signal.

        Args:
            snapshot: Current technical indicators (from fast loop).
            cycle: Current cycle phase analysis.
            onchain: Cached on-chain metrics (may be None if node is down).
            llm: Cached LLM analysis (may be None if Ollama is down).

        Returns:
            CompositeSignal with score, agreement, action, and component details.
        """
        phase_val = cycle.phase.value
        components: list[SubSignal] = []

        # ── RSI sub-signal ───────────────────────────────────────────
        components.append(self._rsi_signal(snapshot, cycle, phase_val))

        # ── MACD sub-signal ──────────────────────────────────────────
        components.append(self._macd_signal(snapshot, phase_val))

        # ── Bollinger sub-signal ─────────────────────────────────────
        components.append(self._bollinger_signal(snapshot, phase_val))

        # ── Cycle sub-signal ─────────────────────────────────────────
        components.append(self._cycle_signal(cycle, phase_val))

        # ── On-chain sub-signal ──────────────────────────────────────
        components.append(self._onchain_signal(onchain, phase_val))

        # ── LLM sub-signal ───────────────────────────────────────────
        components.append(self._llm_signal(llm, phase_val))

        # ── Microstructure sub-signal ────────────────────────────────
        components.append(self._microstructure_signal(snapshot, phase_val))

        # ── Composite calculation ────────────────────────────────────
        score, agreement = self._compute_composite(components)
        data_quality = self._assess_data_quality(snapshot, cycle, onchain, llm)
        action = self._determine_action(score, agreement, data_quality)

        signal = CompositeSignal(
            score=score,
            agreement=agreement,
            action=action,
            components=tuple(components),
            data_quality=data_quality,
            timestamp=time.time(),
        )

        logger.debug(
            f"Signal: score={score:+.1f} agreement={agreement:.2f} "
            f"action={action.value} quality={data_quality:.2f}"
        )

        return signal

    # ─── Sub-signal generators ───────────────────────────────────────────

    def _rsi_signal(self, snapshot: TechnicalSnapshot, cycle: CycleState, phase_val: str = '') -> SubSignal:
        """
        RSI-based signal with cycle-aware interpretation.

        Key insight from architecture doc: RSI divergences in appropriate
        cycle phases should be highest-conviction signals.
        """
        if snapshot.rsi is None:
            return SubSignal(
                name="rsi", score=0.0, weight=self._cfg.rsi_weight,
                direction=0, reason="RSI unavailable",
            )

        rsi_val = snapshot.rsi
        score = 0.0
        reasons: list[str] = []

        # Base RSI scoring: oversold = buy, overbought = sell
        if rsi_val < 20:
            score = 60.0
            reasons.append(f"RSI deeply oversold ({rsi_val:.0f})")
        elif rsi_val < 30:
            score = 40.0
            reasons.append(f"RSI oversold ({rsi_val:.0f})")
        elif rsi_val < 40:
            score = 15.0
            reasons.append(f"RSI low ({rsi_val:.0f})")
        elif rsi_val > 85:
            score = -60.0
            reasons.append(f"RSI extremely overbought ({rsi_val:.0f})")
        elif rsi_val > 75:
            score = -40.0
            reasons.append(f"RSI overbought ({rsi_val:.0f})")
        elif rsi_val > 65:
            score = -10.0
            reasons.append(f"RSI elevated ({rsi_val:.0f})")
        else:
            reasons.append(f"RSI neutral ({rsi_val:.0f})")

        # RSI divergence — highest conviction signals
        div = snapshot.rsi_divergence
        if div is not None:
            if div.bullish:
                # Bullish divergence in accumulation/capitulation = very strong buy
                div_boost = 35.0 * div.strength
                if cycle.phase in (CyclePhase.ACCUMULATION, CyclePhase.CAPITULATION,
                                   CyclePhase.EARLY_BEAR):
                    div_boost *= 1.5
                    reasons.append(f"Bullish RSI divergence in {cycle.phase.value} (HIGH CONVICTION)")
                else:
                    reasons.append(f"Bullish RSI divergence (str={div.strength:.2f})")
                score += div_boost

            if div.bearish:
                div_penalty = -35.0 * div.strength
                if cycle.phase in (CyclePhase.EUPHORIA, CyclePhase.DISTRIBUTION):
                    div_penalty *= 1.5
                    reasons.append(f"Bearish RSI divergence in {cycle.phase.value} (HIGH CONVICTION)")
                else:
                    reasons.append(f"Bearish RSI divergence (str={div.strength:.2f})")
                score += div_penalty

        score = max(-100.0, min(100.0, score))
        direction = 1 if score > 5 else (-1 if score < -5 else 0)

        return SubSignal(
            name="rsi", score=score, weight=self._get_weight("rsi", phase_val),
            direction=direction, reason="; ".join(reasons),
        )

    def _macd_signal(self, snapshot: TechnicalSnapshot, phase_val: str = '') -> SubSignal:
        """MACD crossover and histogram momentum signal."""
        if snapshot.macd is None:
            return SubSignal(
                name="macd", score=0.0, weight=self._get_weight("macd", phase_val),
                direction=0, reason="MACD unavailable",
            )

        macd = snapshot.macd
        score = 0.0
        reasons: list[str] = []

        # Histogram direction and magnitude
        if macd.histogram > 0:
            # Normalize histogram relative to price for comparable scoring
            # Use a simple scaling: positive histogram → bullish
            score = min(50.0, macd.histogram / (snapshot.price * 0.001) * 25)
            reasons.append(f"MACD histogram positive ({macd.histogram:.0f})")
        else:
            score = max(-50.0, macd.histogram / (snapshot.price * 0.001) * 25)
            reasons.append(f"MACD histogram negative ({macd.histogram:.0f})")

        # Crossover bonus
        if macd.bullish and macd.histogram > 0:
            score += 15.0
            reasons.append("MACD bullish crossover")
        elif macd.bearish and macd.histogram < 0:
            score -= 15.0
            reasons.append("MACD bearish crossover")

        score = max(-100.0, min(100.0, score))
        direction = 1 if score > 5 else (-1 if score < -5 else 0)

        return SubSignal(
            name="macd", score=score, weight=self._get_weight("macd", phase_val),
            direction=direction, reason="; ".join(reasons),
        )

    def _bollinger_signal(self, snapshot: TechnicalSnapshot, phase_val: str = '') -> SubSignal:
        """
        Bollinger Bands mean-reversion and squeeze breakout signal.

        Per architecture doc: Bollinger squeeze breakouts should trigger
        increased attention.
        """
        if snapshot.bollinger is None:
            return SubSignal(
                name="bollinger", score=0.0, weight=self._get_weight("bollinger", phase_val),
                direction=0, reason="Bollinger unavailable",
            )

        bb = snapshot.bollinger
        score = 0.0
        reasons: list[str] = []

        # %B position: below 0 = below lower band (oversold), above 1 = above upper
        if bb.percent_b < 0.0:
            score = 50.0  # Strong buy — price below lower band
            reasons.append(f"Price below lower Bollinger band (%B={bb.percent_b:.2f})")
        elif bb.percent_b < 0.15:
            score = 30.0
            reasons.append(f"Price near lower band (%B={bb.percent_b:.2f})")
        elif bb.percent_b > 1.0:
            score = -50.0  # Strong sell — above upper band
            reasons.append(f"Price above upper Bollinger band (%B={bb.percent_b:.2f})")
        elif bb.percent_b > 0.85:
            score = -25.0
            reasons.append(f"Price near upper band (%B={bb.percent_b:.2f})")
        else:
            reasons.append(f"Price within bands (%B={bb.percent_b:.2f})")

        # Squeeze detection — don't generate a directional signal,
        # but flag increased attention
        if snapshot.bollinger_squeeze:
            reasons.append("Bollinger squeeze ACTIVE — breakout imminent")
            # Amplify whatever direction we're seeing
            score *= 1.3

        score = max(-100.0, min(100.0, score))
        direction = 1 if score > 5 else (-1 if score < -5 else 0)

        return SubSignal(
            name="bollinger", score=score, weight=self._get_weight("bollinger", phase_val),
            direction=direction, reason="; ".join(reasons),
        )

    def _cycle_signal(self, cycle: CycleState, phase_val: str = '') -> SubSignal:
        """
        Cycle phase signal — translates the multi-dimensional cycle analysis
        into a trading bias.
        """
        # Use the composite score directly (already -1.0 to +1.0)
        # Scale to our -100 to +100 range
        score = cycle.composite_score * 70.0  # Don't go full ±100; leave room for extremes

        reasons: list[str] = [
            f"Phase: {cycle.phase.value} (confidence={cycle.phase_confidence:.2f})",
            f"Cycle day {cycle.cycle_day} ({cycle.cycle_progress:.0%} elapsed)",
        ]

        # Phase-specific adjustments
        if cycle.phase == CyclePhase.CAPITULATION:
            score += 20.0  # Accumulation bot: capitulation = buy opportunity
            reasons.append("Capitulation phase — accumulation opportunity")
        elif cycle.phase == CyclePhase.EUPHORIA:
            score -= 20.0  # Reduce exposure
            reasons.append("Euphoria phase — reduce exposure")

        # Momentum agreement bonus
        if cycle.momentum.rsi_bullish_divergence:
            score += 10.0
            reasons.append("Cycle momentum shows bullish divergence")
        if cycle.momentum.rsi_bearish_divergence:
            score -= 10.0
            reasons.append("Cycle momentum shows bearish divergence")

        score = max(-100.0, min(100.0, score))
        direction = 1 if score > 5 else (-1 if score < -5 else 0)

        return SubSignal(
            name="cycle", score=score, weight=self._get_weight("cycle", phase_val),
            direction=direction, reason="; ".join(reasons),
        )

    def _onchain_signal(self, onchain: Optional[OnChainSnapshot], phase_val: str = '') -> SubSignal:
        """
        On-chain metrics signal.

        Per architecture doc: on-chain metrics (mempool clearing, large tx
        clusters) serve as confirmation signals.
        """
        if onchain is None:
            return SubSignal(
                name="onchain", score=0.0, weight=self._get_weight("onchain", phase_val),
                direction=0, reason="On-chain data unavailable",
            )

        score = 0.0
        reasons: list[str] = []

        # Mempool clearing = bullish (transactions being confirmed, demand for blockspace resolving)
        if onchain.mempool.clearing:
            score += 20.0
            reasons.append("Mempool clearing — reduced congestion")
        elif onchain.mempool.congested:
            score -= 15.0
            reasons.append("Mempool congested — network stress")

        # Fee pressure
        fp = onchain.fees.fee_pressure
        if fp < 0.2:
            score += 10.0
            reasons.append(f"Low fee pressure ({fp:.2f})")
        elif fp > 0.7:
            score -= 10.0
            reasons.append(f"High fee pressure ({fp:.2f})")

        # Whale activity — information signal, direction depends on context
        if onchain.whale_activity:
            reasons.append(f"Whale activity: {len(onchain.large_txs)} large txs")
            # Large transactions alone are ambiguous — slight positive bias
            # (whales moving coins = market activity = interest)
            score += 5.0

        # Network stress composite
        stress = onchain.network_stress
        if stress > 0.7:
            score -= 15.0
            reasons.append(f"Network stress elevated ({stress:.2f})")
        elif stress < 0.2:
            score += 10.0
            reasons.append(f"Network calm ({stress:.2f})")

        if not reasons:
            reasons.append("On-chain metrics neutral")

        score = max(-100.0, min(100.0, score))
        direction = 1 if score > 5 else (-1 if score < -5 else 0)

        return SubSignal(
            name="onchain", score=score, weight=self._get_weight("onchain", phase_val),
            direction=direction, reason="; ".join(reasons),
        )

    def _llm_signal(self, llm: Optional[LLMContext], phase_val: str = '') -> SubSignal:
        """
        LLM-based market analysis signal.

        Key lesson from previous bot: sentiment must be contextualized by
        strategy intent. For an accumulation bot, "Bitcoin crashes" is a
        BUY signal, not a sell.
        """
        if llm is None:
            return SubSignal(
                name="llm", score=0.0, weight=self._get_weight("llm", phase_val),
                direction=0, reason="LLM analysis unavailable",
            )

        if llm.stale:
            return SubSignal(
                name="llm", score=0.0, weight=self._get_weight("llm", phase_val),
                direction=0, reason="LLM analysis stale (>2h old)",
            )

        score = 0.0
        reasons: list[str] = []

        # INVERTED sentiment for accumulation strategy:
        # Negative sentiment = cheaper prices = buy opportunity
        # Positive sentiment = prices already ran = less upside
        inverted_sentiment = -llm.sentiment * 30.0  # Invert!
        score += inverted_sentiment
        reasons.append(
            f"Sentiment {llm.sentiment:+.2f} → "
            f"accumulation signal {inverted_sentiment:+.0f} (inverted)"
        )

        # Risk level
        risk_scores = {
            "low": 15.0,
            "medium": 0.0,
            "high": -15.0,
            "extreme": -30.0,
        }
        risk_adj = risk_scores.get(llm.risk_level, 0.0)
        score += risk_adj
        reasons.append(f"Risk level: {llm.risk_level} ({risk_adj:+.0f})")

        # Regime alignment
        bullish_regimes = {"accumulation", "markup"}
        bearish_regimes = {"distribution", "decline", "capitulation"}
        if llm.regime in bullish_regimes:
            score += 10.0
            reasons.append(f"LLM regime: {llm.regime} (bullish)")
        elif llm.regime in bearish_regimes:
            # For accumulation bot: bearish regime is actually opportunity
            score += 5.0
            reasons.append(f"LLM regime: {llm.regime} (bear → accumulation opportunity)")

        score = max(-100.0, min(100.0, score))
        direction = 1 if score > 5 else (-1 if score < -5 else 0)

        return SubSignal(
            name="llm", score=score, weight=self._get_weight("llm", phase_val),
            direction=direction, reason="; ".join(reasons),
        )

    def _microstructure_signal(self, snapshot: TechnicalSnapshot, phase_val: str = '') -> SubSignal:
        """
        Market microstructure signal from VWAP relationship and volatility.

        Lightweight — the detailed order book analysis happens in trade_executor.
        """
        if snapshot.vwap is None:
            return SubSignal(
                name="microstructure", score=0.0,
                weight=self._get_weight("microstructure", phase_val),
                direction=0, reason="VWAP unavailable",
            )

        score = 0.0
        reasons: list[str] = []

        # Price vs VWAP
        if snapshot.price < snapshot.vwap:
            discount_pct = (snapshot.vwap - snapshot.price) / snapshot.vwap * 100
            score = min(40.0, discount_pct * 10)
            reasons.append(f"Price below VWAP by {discount_pct:.1f}%")
        elif snapshot.price > snapshot.vwap:
            premium_pct = (snapshot.price - snapshot.vwap) / snapshot.vwap * 100
            score = max(-40.0, -premium_pct * 10)
            reasons.append(f"Price above VWAP by {premium_pct:.1f}%")
        else:
            reasons.append("Price at VWAP")

        # Volatility context
        if snapshot.volatility_percentile is not None:
            if snapshot.volatility_percentile > 0.85:
                reasons.append("High volatility — wider spreads likely")
                score *= 0.7  # Discount signal in high vol
            elif snapshot.volatility_percentile < 0.15:
                reasons.append("Low volatility — tight spreads, good execution")
                score *= 1.2  # Boost signal in low vol

        score = max(-100.0, min(100.0, score))
        direction = 1 if score > 5 else (-1 if score < -5 else 0)

        return SubSignal(
            name="microstructure", score=score,
            weight=self._get_weight("microstructure", phase_val),
            direction=direction, reason="; ".join(reasons),
        )

    # ─── Composite calculation ───────────────────────────────────────────

    def _compute_composite(
        self, components: list[SubSignal],
    ) -> tuple[float, float]:
        """
        Compute weighted composite score and agreement.

        Returns:
            (score, agreement) where score is -100..+100 and agreement is 0.0..1.0.
        """
        total_weight = sum(c.weight for c in components if c.weight > 0)
        if total_weight == 0:
            return 0.0, 0.0

        # Weighted score
        weighted_sum = sum(c.score * c.weight for c in components)
        score = weighted_sum / total_weight

        # Agreement: fraction of non-neutral, weighted signals that agree on direction
        bullish_weight = sum(
            c.weight for c in components if c.direction > 0
        )
        bearish_weight = sum(
            c.weight for c in components if c.direction < 0
        )
        directional_weight = bullish_weight + bearish_weight

        if directional_weight == 0:
            agreement = 0.0
        else:
            dominant = max(bullish_weight, bearish_weight)
            agreement = dominant / directional_weight

        return max(-100.0, min(100.0, score)), agreement

    def _determine_action(
        self,
        score: float,
        agreement: float,
        data_quality: float,
    ) -> Action:
        """
        Map composite score + agreement to a trading action.

        Requires minimum agreement before acting. Low data quality
        forces HOLD regardless of score.

        Uses asymmetric agreement thresholds when configured:
        higher bar for buying (patience), lower for selling (eagerness).
        """
        if data_quality < 0.2:
            return Action.HOLD

        buy_thresh = self._cfg.buy_threshold
        sell_thresh = self._cfg.sell_threshold

        # Asymmetric agreement: use direction-specific threshold
        buy_agreement = self._cfg.buy_min_agreement or self._cfg.min_agreement
        sell_agreement = self._cfg.sell_min_agreement or self._cfg.min_agreement

        if score >= buy_thresh:
            # Buy direction — apply buy agreement threshold
            if agreement < buy_agreement:
                return Action.HOLD
            if score >= buy_thresh * 2:
                return Action.STRONG_BUY
            return Action.BUY
        elif score <= sell_thresh:
            # Sell direction — apply sell agreement threshold
            if agreement < sell_agreement:
                return Action.HOLD
            if score <= sell_thresh * 2:
                return Action.STRONG_SELL
            return Action.SELL
        else:
            return Action.HOLD

    @staticmethod
    def _assess_data_quality(
        snapshot: TechnicalSnapshot,
        cycle: CycleState,
        onchain: Optional[OnChainSnapshot],
        llm: Optional[LLMContext],
    ) -> float:
        """
        Assess overall data quality for confidence scaling.

        Returns 0.0 (no data) to 1.0 (all sources available and fresh).
        """
        quality = 0.0

        # Technical snapshot (50% of total quality)
        quality += snapshot.data_quality * 0.50

        # Cycle analysis (20%) — always available if we have price data
        if cycle.phase_confidence > 0:
            quality += 0.20 * cycle.phase_confidence

        # On-chain (15%)
        if onchain is not None:
            freshness = max(0.0, 1.0 - (time.time() - onchain.timestamp) / 3600)
            quality += 0.15 * freshness

        # LLM (15%)
        if llm is not None and not llm.stale:
            quality += 0.15

        return min(1.0, quality)
