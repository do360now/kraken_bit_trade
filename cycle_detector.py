"""
Multi-signal Bitcoin halving cycle phase detector.

Addresses the critical flaw from the previous bot: phase detection was 90%
calendar-based ("day 300 = GROWTH regardless of whether price already 3x'd").

This implementation combines four signal dimensions with configurable weights:
- Time component (35%): Position within halving cycle
- Price structure (30%): Position relative to dynamic ATH, floor/ceiling model
- Momentum (25%): RSI structure, trend analysis, higher-highs/lower-lows
- Volatility (10%): ATR regime, Bollinger squeeze detection

No single dimension can determine phase alone. The system requires agreement
across multiple dimensions before declaring a phase transition, preventing
premature classification.

Design:
- CycleState dataclass bundles all outputs for downstream consumers.
- ATHTracker (from config.py) is the single source of truth for ATH.
- All public methods return typed results, never raise.
- Phase transitions are logged with full reasoning chains.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import (
    ATHTracker,
    BotConfig,
    CycleConfig,
    CyclePhase,
    IndicatorConfig,
    PersistenceConfig,
    VolatilityRegime,
)
from indicators import (
    TechnicalSnapshot,
    atr_percentile,
    bollinger_bandwidth_percentile,
    rsi,
    rsi_series,
)

logger = logging.getLogger(__name__)


# ─── Output dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class MomentumState:
    """Momentum analysis results."""
    rsi_zone: str                # "oversold", "neutral", "overbought"
    trend_direction: str         # "up", "down", "sideways"
    higher_highs: bool           # Price making higher highs
    higher_lows: bool            # Price making higher lows
    rsi_bullish_divergence: bool
    rsi_bearish_divergence: bool
    momentum_score: float        # -1.0 (bearish) to +1.0 (bullish)


@dataclass(frozen=True)
class PriceStructure:
    """Price structure analysis relative to cycle model."""
    drawdown_from_ath: float          # 0.0 = at ATH, 1.0 = total loss
    position_in_range: float          # 0.0 = at floor, 1.0 = at ceiling
    distance_from_200d_ma: float      # Fraction above/below 200-day MA
    price_structure_score: float      # -1.0 (deep bear) to +1.0 (euphoric)


@dataclass(frozen=True)
class CycleState:
    """
    Complete cycle analysis output — consumed by signal_engine.py.

    This is the interface between cycle detection and the rest of the system.
    """
    phase: CyclePhase
    phase_confidence: float           # 0.0–1.0: how confident we are in this phase

    # Component scores (each -1.0 to +1.0)
    time_score: float
    price_score: float
    momentum_score: float
    volatility_score: float
    composite_score: float            # Weighted combination

    # Detailed sub-analyses
    momentum: MomentumState
    price_structure: PriceStructure
    volatility_regime: VolatilityRegime

    # Cycle metadata
    cycle_day: int                    # Days since halving
    cycle_progress: float             # 0.0–1.0 fraction of estimated cycle elapsed
    ath_eur: float                    # Current all-time high

    # Phase-specific adjustments for downstream modules
    drawdown_tolerance: float         # How much drawdown to tolerate in this phase
    position_size_multiplier: float   # Scale position sizing (0.5–2.0)
    profit_taking_active: bool        # Whether to run profit-taking logic

    timestamp: float


# ─── Cycle Detector ──────────────────────────────────────────────────────────

class CycleDetector:
    """
    Multi-signal cycle phase detector.

    Combines time, price structure, momentum, and volatility to determine
    the current Bitcoin halving cycle phase. Requires agreement across
    dimensions before transitioning phases.

    Args:
        config: Bot configuration (cycle parameters, indicator settings).
        ath_tracker: Dynamic ATH tracker (single source of truth).
    """

    def __init__(self, config: BotConfig, ath_tracker: ATHTracker) -> None:
        self._cycle_cfg = config.cycle
        self._indicator_cfg = config.indicators
        self._risk_cfg = config.risk
        self._ath = ath_tracker
        self._last_phase: Optional[CyclePhase] = None
        self._phase_hold_cycles: int = 0  # Hysteresis counter

    def analyze(
        self,
        snapshot: TechnicalSnapshot,
        closes_daily: list[float],
        highs_daily: list[float],
        lows_daily: list[float],
    ) -> CycleState:
        """
        Run full cycle analysis.

        Args:
            snapshot: Current technical indicator snapshot (from fast loop).
            closes_daily: Daily close prices, oldest first (need 200+ for MA).
            highs_daily: Daily high prices, oldest first.
            lows_daily: Daily low prices, oldest first.

        Returns:
            CycleState with phase, scores, and downstream adjustments.
        """
        now = datetime.now(timezone.utc)
        price = snapshot.price

        # Update ATH
        self._ath.update(price)

        # ── Time analysis ────────────────────────────────────────────
        cycle_day, cycle_progress = self._compute_cycle_timing(now)
        time_score = self._compute_time_score(cycle_progress)

        # ── Price structure analysis ─────────────────────────────────
        price_struct = self._analyze_price_structure(
            price, closes_daily,
        )
        price_score = price_struct.price_structure_score

        # ── Momentum analysis ────────────────────────────────────────
        momentum = self._analyze_momentum(
            closes_daily, highs_daily, lows_daily, snapshot,
        )
        momentum_score = momentum.momentum_score

        # ── Volatility analysis ──────────────────────────────────────
        vol_regime, vol_score = self._analyze_volatility(
            closes_daily, highs_daily, lows_daily, snapshot,
        )

        # ── Weighted composite ───────────────────────────────────────
        cfg = self._cycle_cfg
        composite = (
            time_score * cfg.time_weight
            + price_score * cfg.price_structure_weight
            + momentum_score * cfg.momentum_weight
            + vol_score * cfg.volatility_weight
        )

        # ── Phase determination ──────────────────────────────────────
        phase, confidence = self._determine_phase(
            composite, time_score, price_score, momentum_score,
            vol_regime, cycle_progress,
        )

        # ── Downstream adjustments ───────────────────────────────────
        drawdown_tol = self._risk_cfg.drawdown_tolerance.get(
            phase.value, 0.20,
        )
        size_mult = self._phase_size_multiplier(phase, confidence)
        profit_active = phase in (
            CyclePhase.GROWTH, CyclePhase.EUPHORIA, CyclePhase.DISTRIBUTION,
        )

        state = CycleState(
            phase=phase,
            phase_confidence=confidence,
            time_score=time_score,
            price_score=price_score,
            momentum_score=momentum_score,
            volatility_score=vol_score,
            composite_score=composite,
            momentum=momentum,
            price_structure=price_struct,
            volatility_regime=vol_regime,
            cycle_day=cycle_day,
            cycle_progress=cycle_progress,
            ath_eur=self._ath.ath_eur,
            drawdown_tolerance=drawdown_tol,
            position_size_multiplier=size_mult,
            profit_taking_active=profit_active,
            timestamp=time.time(),
        )

        # Log phase transitions
        if phase != self._last_phase:
            logger.info(
                f"Phase transition: {self._last_phase} → {phase} "
                f"(confidence={confidence:.2f}, composite={composite:.2f}, "
                f"time={time_score:.2f}, price={price_score:.2f}, "
                f"momentum={momentum_score:.2f}, vol={vol_score:.2f})"
            )
            self._last_phase = phase
            self._phase_hold_cycles = 0
        else:
            self._phase_hold_cycles += 1

        return state

    # ─── Time component ──────────────────────────────────────────────────

    def _compute_cycle_timing(self, now: datetime) -> tuple[int, float]:
        """
        Compute days since halving and cycle progress fraction.

        Returns:
            (cycle_day, cycle_progress) where progress is 0.0–1.0.
        """
        delta = now - self._cycle_cfg.halving_date
        cycle_day = max(0, delta.days)
        cycle_progress = min(1.0, cycle_day / self._cycle_cfg.estimated_cycle_days)
        return cycle_day, cycle_progress

    def _compute_time_score(self, progress: float) -> float:
        """
        Time-based score: -1.0 (early accumulation) to +1.0 (late distribution).

        Maps cycle progress to a score curve that reflects historical patterns:
        early cycle is bearish/accumulation, mid-cycle is bullish, late is distribution.
        """
        # Sigmoid-like mapping centered at 0.5 progress
        # Early (0.0–0.2) → negative, Mid (0.3–0.6) → positive, Late (0.7+) → declining
        if progress < 0.15:
            return -0.8 + progress * 2.0  # -0.8 to -0.5
        elif progress < 0.55:
            # Ramp up through bull phase
            return -0.5 + (progress - 0.15) * 3.75  # -0.5 to +1.0
        elif progress < 0.70:
            # Peak zone
            return 1.0 - (progress - 0.55) * 2.0  # +1.0 to +0.7
        elif progress < 0.85:
            # Distribution decline
            return 0.7 - (progress - 0.70) * 6.0  # +0.7 to -0.2
        else:
            # Bear/capitulation
            return -0.2 - (progress - 0.85) * 4.0  # -0.2 to -0.8

    # ─── Price structure component ───────────────────────────────────────

    def _analyze_price_structure(
        self,
        current_price: float,
        closes_daily: list[float],
    ) -> PriceStructure:
        """Analyze price position relative to cycle model and moving averages."""
        # Drawdown from ATH
        drawdown = self._ath.drawdown_from_ath(current_price)

        # Position in floor-ceiling range
        floor = self._cycle_cfg.cycle_floor_eur
        ceiling = self._cycle_cfg.cycle_ceiling_eur
        range_size = ceiling - floor
        if range_size > 0:
            position_in_range = max(0.0, min(1.0,
                (current_price - floor) / range_size
            ))
        else:
            position_in_range = 0.5

        # 200-day moving average distance
        distance_200d = 0.0
        if len(closes_daily) >= 200:
            ma_200 = sum(closes_daily[-200:]) / 200
            if ma_200 > 0:
                distance_200d = (current_price - ma_200) / ma_200

        # Composite price structure score
        score = self._compute_price_score(drawdown, position_in_range, distance_200d)

        return PriceStructure(
            drawdown_from_ath=drawdown,
            position_in_range=position_in_range,
            distance_from_200d_ma=distance_200d,
            price_structure_score=score,
        )

    def _compute_price_score(
        self,
        drawdown: float,
        position_in_range: float,
        distance_200d: float,
    ) -> float:
        """
        Combine price structure metrics into a single score (-1.0 to +1.0).

        High drawdown + below 200d MA → bearish (negative)
        Low drawdown + above 200d MA → bullish (positive)
        Near ceiling → overbought (declining positive)
        """
        # Drawdown component: 0% drawdown → +0.5, 50%+ drawdown → -1.0
        drawdown_component = 0.5 - drawdown * 2.0
        drawdown_component = max(-1.0, min(0.5, drawdown_component))

        # 200-day MA component: above → bullish, below → bearish
        # Clamp distance to [-0.5, 0.5] range
        ma_component = max(-0.5, min(0.5, distance_200d))

        # Range position adjustment: penalize extremes
        # Near floor (buy zone) → slight positive
        # Near ceiling (sell zone) → negative
        if position_in_range < 0.2:
            range_adj = 0.2
        elif position_in_range > 0.8:
            range_adj = -0.3
        else:
            range_adj = 0.0

        score = drawdown_component + ma_component + range_adj
        return max(-1.0, min(1.0, score))

    # ─── Momentum component ──────────────────────────────────────────────

    def _analyze_momentum(
        self,
        closes_daily: list[float],
        highs_daily: list[float],
        lows_daily: list[float],
        snapshot: TechnicalSnapshot,
    ) -> MomentumState:
        """Analyze momentum via RSI, trend structure, and divergences."""
        # RSI zone
        rsi_val = snapshot.rsi
        if rsi_val is not None:
            if rsi_val < 30:
                rsi_zone = "oversold"
            elif rsi_val > 70:
                rsi_zone = "overbought"
            else:
                rsi_zone = "neutral"
        else:
            rsi_zone = "neutral"

        # Trend direction via price structure (20-day vs 50-day)
        trend = self._detect_trend(closes_daily)

        # Higher highs / higher lows detection
        hh, hl = self._detect_swing_structure(highs_daily, lows_daily)

        # RSI divergences from snapshot
        bull_div = (snapshot.rsi_divergence is not None
                    and snapshot.rsi_divergence.bullish)
        bear_div = (snapshot.rsi_divergence is not None
                    and snapshot.rsi_divergence.bearish)

        # Composite momentum score
        m_score = self._compute_momentum_score(
            rsi_val, trend, hh, hl, bull_div, bear_div,
        )

        return MomentumState(
            rsi_zone=rsi_zone,
            trend_direction=trend,
            higher_highs=hh,
            higher_lows=hl,
            rsi_bullish_divergence=bull_div,
            rsi_bearish_divergence=bear_div,
            momentum_score=m_score,
        )

    def _detect_trend(self, closes: list[float]) -> str:
        """Detect trend using 20-day vs 50-day moving averages."""
        if len(closes) < 50:
            return "sideways"

        ma_20 = sum(closes[-20:]) / 20
        ma_50 = sum(closes[-50:]) / 50

        if ma_50 == 0:
            return "sideways"

        spread = (ma_20 - ma_50) / ma_50

        if spread > 0.02:
            return "up"
        elif spread < -0.02:
            return "down"
        return "sideways"

    def _detect_swing_structure(
        self,
        highs: list[float],
        lows: list[float],
        lookback: int = 30,
        swing_window: int = 5,
    ) -> tuple[bool, bool]:
        """
        Detect higher-highs and higher-lows pattern.

        Scans the last `lookback` bars for swing points (local extremes over
        `swing_window` bars), then checks if the last two swing highs are
        ascending and the last two swing lows are ascending.
        """
        if len(highs) < lookback or len(lows) < lookback:
            return False, False

        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        half = swing_window // 2

        swing_highs: list[float] = []
        swing_lows: list[float] = []

        for i in range(half, len(recent_highs) - half):
            window_h = recent_highs[i - half: i + half + 1]
            if recent_highs[i] == max(window_h):
                swing_highs.append(recent_highs[i])

            window_l = recent_lows[i - half: i + half + 1]
            if recent_lows[i] == min(window_l):
                swing_lows.append(recent_lows[i])

        hh = (len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2])
        hl = (len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2])

        return hh, hl

    def _compute_momentum_score(
        self,
        rsi_val: Optional[float],
        trend: str,
        higher_highs: bool,
        higher_lows: bool,
        bull_divergence: bool,
        bear_divergence: bool,
    ) -> float:
        """Combine momentum signals into a single score (-1.0 to +1.0)."""
        score = 0.0

        # RSI contribution
        if rsi_val is not None:
            # Map RSI 0–100 to -0.4 to +0.4
            score += (rsi_val - 50.0) / 125.0

        # Trend contribution
        if trend == "up":
            score += 0.25
        elif trend == "down":
            score -= 0.25

        # Swing structure
        if higher_highs and higher_lows:
            score += 0.2  # Strong bullish structure
        elif not higher_highs and not higher_lows:
            score -= 0.15  # Weakening structure

        # Divergences (high conviction)
        if bull_divergence:
            score += 0.3
        if bear_divergence:
            score -= 0.3

        return max(-1.0, min(1.0, score))

    # ─── Volatility component ────────────────────────────────────────────

    def _analyze_volatility(
        self,
        closes_daily: list[float],
        highs_daily: list[float],
        lows_daily: list[float],
        snapshot: TechnicalSnapshot,
    ) -> tuple[VolatilityRegime, float]:
        """
        Classify volatility regime and compute volatility score.

        Returns:
            (regime, score) where score is -1.0 (extreme vol) to +1.0 (compression).
        """
        # ATR percentile
        atr_pct = snapshot.volatility_percentile

        # Bollinger bandwidth percentile
        bb_pct = bollinger_bandwidth_percentile(closes_daily) if len(closes_daily) >= 120 else None

        # Use the more reliable of the two, or average if both available
        if atr_pct is not None and bb_pct is not None:
            vol_percentile = (atr_pct + bb_pct) / 2.0
        elif atr_pct is not None:
            vol_percentile = atr_pct
        elif bb_pct is not None:
            vol_percentile = bb_pct
        else:
            vol_percentile = 0.5  # Neutral

        # Classify regime
        regime = self._classify_vol_regime(vol_percentile, snapshot.bollinger_squeeze)

        # Score: compression is positive (breakout potential), extreme is negative (risk)
        # This is intentionally asymmetric — compression is opportunity, extreme vol is risk
        if vol_percentile < 0.15:
            score = 0.6   # Compression — high breakout potential
        elif vol_percentile < 0.30:
            score = 0.3   # Low vol
        elif vol_percentile < 0.70:
            score = 0.0   # Normal
        elif vol_percentile < 0.85:
            score = -0.3  # Elevated
        else:
            score = -0.6  # Extreme

        # Squeeze bonus
        if snapshot.bollinger_squeeze:
            score += 0.2
            score = min(1.0, score)

        return regime, score

    @staticmethod
    def _classify_vol_regime(
        percentile: float, squeeze: bool,
    ) -> VolatilityRegime:
        """Map volatility percentile to regime classification."""
        if squeeze or percentile < 0.10:
            return VolatilityRegime.COMPRESSION
        elif percentile < 0.25:
            return VolatilityRegime.LOW
        elif percentile < 0.75:
            return VolatilityRegime.NORMAL
        elif percentile < 0.90:
            return VolatilityRegime.ELEVATED
        else:
            return VolatilityRegime.EXTREME

    # ─── Phase determination ─────────────────────────────────────────────

    def _determine_phase(
        self,
        composite: float,
        time_score: float,
        price_score: float,
        momentum_score: float,
        vol_regime: VolatilityRegime,
        cycle_progress: float,
    ) -> tuple[CyclePhase, float]:
        """
        Determine cycle phase from composite and component scores.

        Uses a combination of composite score ranges and component agreement.
        Includes hysteresis to prevent rapid phase flipping.

        Returns:
            (phase, confidence) where confidence is 0.0–1.0.
        """
        # Score all phases and pick the best fit
        candidates: list[tuple[CyclePhase, float]] = []

        # Each phase has characteristic score profiles
        candidates.append((
            CyclePhase.CAPITULATION,
            self._score_phase_fit(
                composite, time_score, price_score, momentum_score,
                target_composite=(-1.0, -0.4),
                target_momentum=(-1.0, -0.2),
                target_price=(-1.0, -0.2),
            ),
        ))

        candidates.append((
            CyclePhase.ACCUMULATION,
            self._score_phase_fit(
                composite, time_score, price_score, momentum_score,
                target_composite=(-0.5, 0.0),
                target_momentum=(-0.3, 0.2),
                target_price=(-0.5, 0.1),
            ),
        ))

        candidates.append((
            CyclePhase.EARLY_BULL,
            self._score_phase_fit(
                composite, time_score, price_score, momentum_score,
                target_composite=(-0.1, 0.4),
                target_momentum=(0.0, 0.5),
                target_price=(-0.1, 0.4),
            ),
        ))

        candidates.append((
            CyclePhase.GROWTH,
            self._score_phase_fit(
                composite, time_score, price_score, momentum_score,
                target_composite=(0.2, 0.7),
                target_momentum=(0.1, 0.8),
                target_price=(0.1, 0.7),
            ),
        ))

        candidates.append((
            CyclePhase.EUPHORIA,
            self._score_phase_fit(
                composite, time_score, price_score, momentum_score,
                target_composite=(0.5, 1.0),
                target_momentum=(0.3, 1.0),
                target_price=(0.4, 1.0),
            ),
        ))

        candidates.append((
            CyclePhase.DISTRIBUTION,
            self._score_phase_fit(
                composite, time_score, price_score, momentum_score,
                target_composite=(0.0, 0.5),
                target_momentum=(-0.3, 0.1),
                target_price=(0.2, 0.7),
                # Distribution: price still high but momentum fading
            ),
        ))

        candidates.append((
            CyclePhase.EARLY_BEAR,
            self._score_phase_fit(
                composite, time_score, price_score, momentum_score,
                target_composite=(-0.5, 0.0),
                target_momentum=(-0.6, -0.1),
                target_price=(-0.2, 0.3),
            ),
        ))

        # Sort by fit score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_phase, best_fit = candidates[0]

        # ── Phase stability: prevent rapid flapping ───────────────────
        # Phases represent multi-week market regimes, not minute-level
        # fluctuations. Three guards prevent spurious transitions:
        #
        # 1. Minimum dwell time: must stay in current phase for N cycles
        # 2. Confidence floor: new phase must meet minimum confidence
        # 3. Fit advantage: new phase must convincingly beat current

        if (self._last_phase is not None
                and best_phase != self._last_phase):

            min_dwell = self._cycle_cfg.min_phase_dwell_cycles
            min_confidence = self._cycle_cfg.phase_transition_confidence
            base_advantage = self._cycle_cfg.phase_transition_advantage

            # Guard 1: Minimum dwell time
            if self._phase_hold_cycles < min_dwell:
                best_phase = self._last_phase
                # Recalculate best_fit for the held phase
                for phase, fit in candidates:
                    if phase == self._last_phase:
                        best_fit = fit
                        break
            else:
                # Guard 2 & 3: Confidence and advantage checks
                current_fit = 0.0
                for phase, fit in candidates:
                    if phase == self._last_phase:
                        current_fit = fit
                        break

                # Calculate confidence for candidate phase
                second_fit = candidates[1][1] if len(candidates) > 1 else 0.0
                candidate_confidence = min(1.0, max(0.1, best_fit - second_fit + 0.3))

                # Reject if confidence too low
                if candidate_confidence < min_confidence:
                    best_phase = self._last_phase
                    best_fit = current_fit
                # Reject if advantage too small
                elif best_fit - current_fit < base_advantage:
                    best_phase = self._last_phase
                    best_fit = current_fit

        # Confidence: how much better is the best fit than the second best
        second_fit = candidates[1][1] if len(candidates) > 1 else 0.0
        confidence = min(1.0, max(0.1, best_fit - second_fit + 0.3))

        return best_phase, confidence

    @staticmethod
    def _score_phase_fit(
        composite: float,
        time_score: float,
        price_score: float,
        momentum_score: float,
        target_composite: tuple[float, float],
        target_momentum: tuple[float, float],
        target_price: tuple[float, float],
    ) -> float:
        """
        Score how well current signals fit a target phase profile.

        Each target is a (min, max) range. Score is higher when values
        fall within or near the target range.
        """
        def range_fit(value: float, target: tuple[float, float]) -> float:
            lo, hi = target
            if lo <= value <= hi:
                # Inside range: score based on how centered
                mid = (lo + hi) / 2.0
                half_range = (hi - lo) / 2.0
                if half_range == 0:
                    return 1.0
                return 1.0 - abs(value - mid) / half_range * 0.3
            else:
                # Outside range: penalize by distance
                dist = min(abs(value - lo), abs(value - hi))
                return max(0.0, 1.0 - dist * 2.0)

        fit = (
            range_fit(composite, target_composite) * 0.4
            + range_fit(momentum_score, target_momentum) * 0.35
            + range_fit(price_score, target_price) * 0.25
        )
        return fit

    # ─── Downstream adjustments ──────────────────────────────────────────

    @staticmethod
    def _phase_size_multiplier(phase: CyclePhase, confidence: float) -> float:
        """
        Position size multiplier by phase.

        Accumulation/capitulation: larger (buy the dip).
        Growth: normal.
        Euphoria/distribution: smaller (take profits, don't overextend).
        """
        base_multipliers = {
            CyclePhase.CAPITULATION: 1.5,
            CyclePhase.ACCUMULATION: 1.3,
            CyclePhase.EARLY_BULL: 1.2,
            CyclePhase.GROWTH: 1.0,
            CyclePhase.EUPHORIA: 0.6,
            CyclePhase.DISTRIBUTION: 0.5,
            CyclePhase.EARLY_BEAR: 0.7,
        }
        base = base_multipliers.get(phase, 1.0)

        # Scale toward 1.0 when confidence is low
        return 1.0 + (base - 1.0) * confidence
