"""
Macro volatility event detector — CPI, FOMC, and stagflation regime.

Key insight: CPI day is a volatility event for Bitcoin, not a directional
signal. The initial move reflects algo/leveraged-trader reaction; the
48-hour follow-through reveals what spot buyers believe. For an accumulation
bot this creates two distinct behaviours:

  PRE-EVENT WINDOW (2h before):
    - Suppress directional signals, do not initiate new buys
    - Pre-position spread-aware limit orders deep in the book,
      ready to catch a "hot CPI" dip automatically

  EVENT DAY (CPI/FOMC release day):
    - Reduce position sizes (algo volatility, not accumulation opportunity)
    - Hold all open limit orders; do not chase fills

  FOLLOW-THROUGH WINDOW (0–48h after event):
    - Re-enable directional signals
    - Boost dip-buy signal if price dropped on the event
      (hot CPI dips historically get bought back within 2-3 days)
    - If price held / rallied: treat as macro confirmation signal

  STAGFLATION REGIME (sticky inflation + slowing growth):
    - CoinShares 2026 outlook: stagflation floor ~$70k (~€65k)
    - Apply floor: if price < stagflation_floor_eur, signal dampening inverted
    - Tighten position sizes: stagflation is ambiguous for BTC, not bullish

Design:
  - MacroEventState dataclass: consumed by signal_engine and main.py
  - MacroEventDetector: stateless analysis of current datetime
  - CPI dates: 2026 hardcoded (BLS releases ~2nd/3rd Wed of month at 13:30 UTC)
  - FOMC dates: 2026 hardcoded (8 meetings/year)
  - Stagflation: keyword analysis of LLM themes (passed in)
  - No external dependencies beyond stdlib
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Event types ─────────────────────────────────────────────────────────────

class MacroEventType(Enum):
    """Type of macro volatility event."""
    CPI      = "cpi"
    FOMC     = "fomc"
    PCE      = "pce"       # PCE deflator — Fed's preferred inflation gauge
    NONE     = "none"


class EventPhase(Enum):
    """Where in the event lifecycle are we?"""
    PRE_EVENT        = "pre_event"       # Within pre_event_window_hours before release
    EVENT_DAY        = "event_day"       # Day of the event but outside pre-event window
    FOLLOW_THROUGH   = "follow_through"  # Within follow_through_window_hours after event
    NORMAL           = "normal"          # No nearby event


# ─── Output dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MacroEventState:
    """
    Complete macro event analysis — consumed by signal_engine and main.py.

    This is the interface between macro event detection and the rest of the system.
    """
    event_type: MacroEventType
    event_phase: EventPhase

    # Timing context
    hours_until_next_event: float      # > 0 if event is upcoming
    hours_since_last_event: float      # > 0 if we're in follow-through

    # Action guidance
    suppress_directional_signals: bool  # True in pre-event window
    signal_dampening_factor: float      # 0.0–1.0: multiply signal score by this
    size_multiplier: float              # 0.0–1.0: multiply position size by this
    pre_position_for_dip: bool          # True: place low limit orders pre-event
    dip_buy_boost: float                # Extra score to add if price dipped on event

    # Stagflation regime
    is_stagflation_regime: bool
    stagflation_confidence: float       # 0.0–1.0: how confident we are
    stagflation_floor_eur: float        # Estimated downside floor in this regime
    below_stagflation_floor: bool       # True when current price < floor
    # When below_stagflation_floor: dampening is removed, floor_boost is added to score.
    # The floor IS the accumulation opportunity — limited further downside from here.

    # Human-readable context
    description: str


def _no_event() -> MacroEventState:
    """Default state when no event is nearby."""
    return MacroEventState(
        event_type=MacroEventType.NONE,
        event_phase=EventPhase.NORMAL,
        hours_until_next_event=9999.0,
        hours_since_last_event=9999.0,
        suppress_directional_signals=False,
        signal_dampening_factor=1.0,
        size_multiplier=1.0,
        pre_position_for_dip=False,
        dip_buy_boost=0.0,
        is_stagflation_regime=False,
        stagflation_confidence=0.0,
        stagflation_floor_eur=0.0,
        below_stagflation_floor=False,
        description="No macro event nearby",
    )


# ─── Known macro event dates ──────────────────────────────────────────────────

# US CPI release dates for 2026 (BLS schedule, 8:30 AM ET = 13:30 UTC)
# Source: Bureau of Labor Statistics release calendar
_CPI_DATES_2026: list[tuple[int, int]] = [
    (1, 14), (2, 11), (3, 11), (4, 10), (5, 13), (6, 10),
    (7, 15), (8, 12), (9, 9),  (10, 14), (11, 12), (12, 9),
]

# FOMC meeting end dates for 2026 (rate decisions announced ~14:00 ET = 19:00 UTC)
# Fed press conference same day
_FOMC_DATES_2026: list[tuple[int, int]] = [
    (1, 29), (3, 18), (5, 6), (6, 17), (7, 29), (9, 16), (11, 4), (12, 16),
]

# PCE deflator release dates for 2026 (BEA, 8:30 AM ET = 13:30 UTC)
# Typically released last Friday of the month (for prior month)
_PCE_DATES_2026: list[tuple[int, int]] = [
    (1, 30), (2, 27), (3, 28), (4, 30), (5, 29), (6, 26),
    (7, 31), (8, 28), (9, 26), (10, 30), (11, 25), (12, 23),
]

# Release times in UTC (hour, minute)
_RELEASE_TIME_UTC: dict[MacroEventType, tuple[int, int]] = {
    MacroEventType.CPI:  (13, 30),  # 8:30 AM ET
    MacroEventType.PCE:  (13, 30),  # 8:30 AM ET
    MacroEventType.FOMC: (19, 0),   # 2:00 PM ET
}


# ─── Stagflation keyword sets ─────────────────────────────────────────────────

_STAGFLATION_KEYWORDS = {
    "stagflation", "stagflationary",
    "sticky inflation", "persistent inflation", "elevated inflation",
    "supply shock", "oil shock", "energy shock",
    "slowing growth", "growth slowdown", "recessionary",
    "iran", "geopolitical", "oil price",
    "supply disruption", "iran oil",
}

_ANTI_STAGFLATION_KEYWORDS = {
    "inflation falling", "disinflation", "deflation",
    "strong growth", "robust growth", "soft landing",
    "rate cuts", "qe", "easing",
}


# ─── Macro Event Detector ────────────────────────────────────────────────────

class MacroEventDetector:
    """
    Detects proximity to CPI/FOMC/PCE events and stagflation regime.

    Stateless: call analyze() each loop iteration; no internal state.
    All known event dates are hardcoded for 2026 and fall back to
    a heuristic for dates outside that range.

    Args:
        config: Bot configuration (MacroEventConfig for windows/thresholds).
    """

    def __init__(self, config: "BotConfig") -> None:  # type: ignore[name-defined]
        self._cfg = config.macro_event

    def analyze(
        self,
        now: Optional[datetime] = None,
        llm_themes: Optional[tuple[str, ...]] = None,
        llm_regime: Optional[str] = None,
        price_at_last_event: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> MacroEventState:
        """
        Analyze current macro event context.

        Args:
            now: Current UTC datetime (defaults to datetime.now(UTC)).
            llm_themes: Themes from LLM analysis (used for stagflation detection).
            llm_regime: Regime label from LLM analysis.
            price_at_last_event: BTC price at the time of the last event release.
            current_price: Current BTC price (used to detect post-event dip).

        Returns:
            MacroEventState with guidance for signal engine and execution.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # ── Find nearest events ──────────────────────────────────────
        nearest_upcoming, hours_until = self._find_nearest_upcoming(now)
        nearest_past, hours_since = self._find_nearest_past(now)

        # ── Determine phase ──────────────────────────────────────────
        event_type = MacroEventType.NONE
        phase = EventPhase.NORMAL

        if hours_until <= self._cfg.pre_event_window_hours:
            # We're in the pre-event suppression window
            phase = EventPhase.PRE_EVENT
            event_type = nearest_upcoming[0] if nearest_upcoming else MacroEventType.NONE
        elif hours_until <= 8 and nearest_upcoming:
            # Same day as event (outside the immediate pre-event window)
            phase = EventPhase.EVENT_DAY
            event_type = nearest_upcoming[0]
        elif hours_since <= self._cfg.follow_through_window_hours and nearest_past:
            # We're in the follow-through observation window
            phase = EventPhase.FOLLOW_THROUGH
            event_type = nearest_past[0]

        # ── Compute action parameters by phase ───────────────────────
        suppress = False
        dampening = 1.0
        size_mult = 1.0
        pre_pos_dip = False
        dip_buy_boost = 0.0
        description_parts: list[str] = []

        if phase == EventPhase.PRE_EVENT:
            suppress = True
            dampening = self._cfg.pre_event_dampening
            size_mult = self._cfg.pre_event_size_multiplier
            pre_pos_dip = True
            description_parts.append(
                f"PRE-EVENT: {event_type.value.upper()} in "
                f"{hours_until:.1f}h — signals suppressed, limit orders ready"
            )

        elif phase == EventPhase.EVENT_DAY:
            suppress = False
            dampening = self._cfg.event_day_dampening
            size_mult = self._cfg.event_day_size_multiplier
            pre_pos_dip = True
            description_parts.append(
                f"EVENT DAY: {event_type.value.upper()} releasing in "
                f"{hours_until:.1f}h — reduced sizing, dip orders active"
            )

        elif phase == EventPhase.FOLLOW_THROUGH:
            # Detect post-event dip for boost
            if (price_at_last_event is not None
                    and current_price is not None
                    and current_price < price_at_last_event):
                dip_pct = (price_at_last_event - current_price) / price_at_last_event
                # Boost scales with dip magnitude, capped at configured max
                dip_buy_boost = min(
                    self._cfg.follow_through_dip_boost_max,
                    dip_pct * self._cfg.follow_through_dip_boost_sensitivity * 100,
                )
                description_parts.append(
                    f"FOLLOW-THROUGH: {event_type.value.upper()} was {hours_since:.0f}h ago, "
                    f"price dipped {dip_pct:.1%} — dip boost +{dip_buy_boost:.1f}"
                )
            else:
                description_parts.append(
                    f"FOLLOW-THROUGH: {event_type.value.upper()} was {hours_since:.0f}h ago"
                    f" — no dip detected, observing follow-through"
                )

        # ── Stagflation regime ───────────────────────────────────────
        is_stagflation, stagflation_conf = self._detect_stagflation(
            llm_themes, llm_regime,
        )

        below_floor = False
        floor_eur = self._cfg.stagflation_floor_eur if is_stagflation else 0.0

        if is_stagflation:
            below_floor = (
                current_price is not None
                and current_price < self._cfg.stagflation_floor_eur
            )

            if below_floor:
                # ── BELOW STAGFLATION FLOOR: accumulation territory ───
                # CoinShares thesis: the floor (~€65k) represents limited
                # further downside. Being below it is the accumulation
                # opportunity, not a reason to reduce sizing further.
                #
                # Logic inversion:
                #   ABOVE floor → dampen (uncertainty, downside risk remains)
                #   BELOW floor → floor boost (limited further downside, buy)
                #
                # Size is still cautious (0.85x) — stagflation is ambiguous —
                # but dampening is removed and a floor bonus is added to score.
                dip_below_pct = (
                    (self._cfg.stagflation_floor_eur - current_price)
                    / self._cfg.stagflation_floor_eur
                )
                floor_bonus = min(
                    self._cfg.stagflation_floor_boost,
                    dip_below_pct * 200,  # 1% below floor = +2 pts, caps at floor_boost
                )
                dip_buy_boost += floor_bonus
                # Reduce size slightly but do NOT dampen signals
                size_mult *= self._cfg.stagflation_below_floor_size_multiplier
                description_parts.append(
                    f"BELOW STAGFLATION FLOOR €{self._cfg.stagflation_floor_eur:,.0f} "
                    f"by {dip_below_pct:.1%} — accumulation territory, "
                    f"floor boost +{floor_bonus:.1f}, limited further downside "
                    f"(CoinShares stagflation thesis)"
                )
            else:
                # Above floor: apply dampening — downside risk to floor still exists
                dampening *= self._cfg.stagflation_dampening
                size_mult *= self._cfg.stagflation_size_multiplier
                distance_pct = (
                    (current_price - self._cfg.stagflation_floor_eur)
                    / self._cfg.stagflation_floor_eur
                ) if current_price else 0.0
                description_parts.append(
                    f"STAGFLATION regime (conf={stagflation_conf:.2f}) — "
                    f"{distance_pct:.1%} above floor €{self._cfg.stagflation_floor_eur:,.0f}, "
                    f"downside risk present, signals dampened"
                )

        description = "; ".join(description_parts) if description_parts else "No macro event nearby"

        if phase == EventPhase.NORMAL and not is_stagflation:
            return _no_event()

        return MacroEventState(
            event_type=event_type,
            event_phase=phase,
            hours_until_next_event=hours_until,
            hours_since_last_event=hours_since,
            suppress_directional_signals=suppress,
            signal_dampening_factor=dampening,
            size_multiplier=size_mult,
            pre_position_for_dip=pre_pos_dip,
            dip_buy_boost=dip_buy_boost,
            is_stagflation_regime=is_stagflation,
            stagflation_confidence=stagflation_conf,
            stagflation_floor_eur=floor_eur,
            below_stagflation_floor=below_floor,
            description=description,
        )

    # ─── Event date helpers ──────────────────────────────────────────────

    def _all_events_for_year(
        self, year: int,
    ) -> list[tuple[MacroEventType, datetime]]:
        """Build a sorted list of (event_type, release_datetime_utc) for year."""
        events: list[tuple[MacroEventType, datetime]] = []

        date_maps: list[tuple[MacroEventType, list[tuple[int, int]]]] = [
            (MacroEventType.CPI,  _CPI_DATES_2026  if year == 2026 else self._heuristic_cpi(year)),
            (MacroEventType.FOMC, _FOMC_DATES_2026 if year == 2026 else []),
            (MacroEventType.PCE,  _PCE_DATES_2026  if year == 2026 else []),
        ]

        for event_type, dates in date_maps:
            hour, minute = _RELEASE_TIME_UTC[event_type]
            for month, day in dates:
                try:
                    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                    events.append((event_type, dt))
                except ValueError:
                    pass  # Invalid date, skip

        events.sort(key=lambda x: x[1])
        return events

    @staticmethod
    def _heuristic_cpi(year: int) -> list[tuple[int, int]]:
        """
        Heuristic fallback for years without hardcoded dates.

        CPI is typically released on the 2nd or 3rd Wednesday of each month.
        Returns approximate dates (day 10-15 of each month).
        """
        results = []
        for month in range(1, 13):
            # Find the 2nd Wednesday of the month
            first_day = datetime(year, month, 1, tzinfo=timezone.utc)
            # Weekday 2 = Wednesday
            days_until_wed = (2 - first_day.weekday()) % 7
            first_wed = first_day + timedelta(days=days_until_wed)
            second_wed = first_wed + timedelta(weeks=1)
            results.append((month, second_wed.day))
        return results

    def _find_nearest_upcoming(
        self, now: datetime,
    ) -> tuple[Optional[tuple[MacroEventType, datetime]], float]:
        """Find the next upcoming event and hours until it."""
        years_to_check = {now.year, now.year + 1}
        events: list[tuple[MacroEventType, datetime]] = []
        for year in sorted(years_to_check):
            events.extend(self._all_events_for_year(year))

        upcoming = [(t, dt) for t, dt in events if dt > now]
        if not upcoming:
            return None, 9999.0

        nearest_type, nearest_dt = upcoming[0]
        hours_until = (nearest_dt - now).total_seconds() / 3600
        return (nearest_type, nearest_dt), hours_until

    def _find_nearest_past(
        self, now: datetime,
    ) -> tuple[Optional[tuple[MacroEventType, datetime]], float]:
        """Find the most recent past event and hours since it."""
        years_to_check = {now.year - 1, now.year}
        events: list[tuple[MacroEventType, datetime]] = []
        for year in sorted(years_to_check):
            events.extend(self._all_events_for_year(year))

        past = [(t, dt) for t, dt in events if dt <= now]
        if not past:
            return None, 9999.0

        nearest_type, nearest_dt = past[-1]
        hours_since = (now - nearest_dt).total_seconds() / 3600
        return (nearest_type, nearest_dt), hours_since

    # ─── Stagflation detection ───────────────────────────────────────────

    def _detect_stagflation(
        self,
        themes: Optional[tuple[str, ...]],
        regime: Optional[str],
    ) -> tuple[bool, float]:
        """
        Detect stagflation regime from LLM themes and regime label.

        Stagflation = sticky/persistent inflation + slowing growth.
        This is specifically the bear case for Bitcoin where supply shocks
        (oil, geopolitical) combine with demand weakness.

        Returns:
            (is_stagflation, confidence) where confidence is 0.0–1.0.
        """
        if themes is None and regime is None:
            return False, 0.0

        text = " ".join(t.lower() for t in (themes or ())) + " " + (regime or "").lower()

        # Count keyword hits
        stagflation_hits = sum(1 for kw in _STAGFLATION_KEYWORDS if kw in text)
        anti_hits = sum(1 for kw in _ANTI_STAGFLATION_KEYWORDS if kw in text)

        # Explicit regime labels
        if regime and any(r in regime.lower() for r in ["stagflat", "stagnation"]):
            stagflation_hits += 3

        # Net score
        net_hits = stagflation_hits - anti_hits

        if net_hits <= 0:
            return False, 0.0

        # Confidence scales with hits but caps at 0.90
        confidence = min(0.90, net_hits * 0.25)
        threshold = self._cfg.stagflation_detection_threshold

        return confidence >= threshold, confidence
