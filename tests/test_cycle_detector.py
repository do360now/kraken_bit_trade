"""
Tests for cycle_detector.py

Validates:
- Time score mapping across full cycle
- Price structure analysis (ATH drawdown, 200d MA, range position)
- Momentum analysis (RSI zones, trend, swing structure, divergences)
- Volatility regime classification
- Multi-signal phase determination
- Phase transition hysteresis
- Downstream adjustments (size multiplier, drawdown tolerance)
- Edge cases (no data, early cycle, late cycle)

Run: python -m pytest tests/test_cycle_detector.py -v
"""
from __future__ import annotations

import math
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ATHTracker,
    BotConfig,
    CyclePhase,
    IndicatorConfig,
    PersistenceConfig,
    VolatilityRegime,
)
from indicators import (
    BollingerBands,
    MACDResult,
    RSIDivergence,
    TechnicalSnapshot,
)
from cycle_detector import (
    CycleDetector,
    CycleState,
    MomentumState,
    PriceStructure,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path) -> BotConfig:
    """Create a BotConfig with tmp_path for persistence."""
    config = BotConfig(
        persistence=PersistenceConfig(base_dir=tmp_path),
    )
    return config


def make_ath_tracker(tmp_path: Path, ath_value: float = 100000.0) -> ATHTracker:
    """Create an ATH tracker with a known ATH value."""
    persistence = PersistenceConfig(base_dir=tmp_path)
    persistence.ensure_dirs()
    tracker = ATHTracker(persistence)
    if ath_value > 0:
        tracker.update(ath_value)
    return tracker


def make_snapshot(
    price: float = 50000.0,
    rsi: float = 50.0,
    bollinger_squeeze: bool = False,
    vol_percentile: float = 0.5,
    rsi_divergence: RSIDivergence = None,
) -> TechnicalSnapshot:
    """Create a TechnicalSnapshot with sensible defaults."""
    return TechnicalSnapshot(
        price=price,
        timestamp=time.time(),
        rsi=rsi,
        macd=MACDResult(macd_line=0.0, signal_line=0.0, histogram=0.0),
        bollinger=BollingerBands(
            upper=price * 1.04, middle=price, lower=price * 0.96,
            bandwidth=0.08, percent_b=0.5,
        ),
        atr=price * 0.02,
        vwap=price,
        rsi_divergence=rsi_divergence,
        bollinger_squeeze=bollinger_squeeze,
        volatility_percentile=vol_percentile,
    )


def make_trending_closes(
    n: int, start: float, end: float, noise: float = 0.0,
) -> list[float]:
    """Generate a trending price series."""
    random.seed(42)
    step = (end - start) / max(n - 1, 1)
    return [start + i * step + random.uniform(-noise, noise) for i in range(n)]


def make_daily_data(closes: list[float], spread: float = 500.0):
    """Generate highs and lows from closes."""
    highs = [c + abs(spread) for c in closes]
    lows = [c - abs(spread) for c in closes]
    return closes, highs, lows


# ─── Time score tests ────────────────────────────────────────────────────────

class TestTimeScore:
    def _make_detector(self, tmp_path) -> CycleDetector:
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path)
        return CycleDetector(config, ath)

    def test_early_cycle_negative(self, tmp_path):
        """Very early in cycle → negative time score (accumulation territory)."""
        det = self._make_detector(tmp_path)
        score = det._compute_time_score(0.05)
        assert score < -0.5

    def test_mid_cycle_positive(self, tmp_path):
        """Mid cycle → positive time score (growth/bull territory)."""
        det = self._make_detector(tmp_path)
        score = det._compute_time_score(0.40)
        assert score > 0.3

    def test_peak_cycle_high(self, tmp_path):
        """Around cycle peak → high positive score."""
        det = self._make_detector(tmp_path)
        score = det._compute_time_score(0.55)
        assert score > 0.5

    def test_late_cycle_declining(self, tmp_path):
        """Late cycle → declining/negative score (distribution/bear)."""
        det = self._make_detector(tmp_path)
        score = det._compute_time_score(0.80)
        assert score < 0.3

    def test_very_late_cycle_negative(self, tmp_path):
        """Very late cycle → negative score (bear market)."""
        det = self._make_detector(tmp_path)
        score = det._compute_time_score(0.95)
        assert score < -0.3

    def test_score_bounded(self, tmp_path):
        """Time score should always be within [-1, 1]."""
        det = self._make_detector(tmp_path)
        for progress in [i * 0.05 for i in range(21)]:
            score = det._compute_time_score(progress)
            assert -1.0 <= score <= 1.0, f"Score {score} out of bounds at progress {progress}"

    def test_monotonic_in_bull_phase(self, tmp_path):
        """Score should increase through the bull phase (0.15 to 0.55)."""
        det = self._make_detector(tmp_path)
        prev = det._compute_time_score(0.15)
        for p in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            curr = det._compute_time_score(p)
            assert curr >= prev, f"Score not increasing at {p}: {prev} → {curr}"
            prev = curr


# ─── Price structure tests ───────────────────────────────────────────────────

class TestPriceStructure:
    def _make_detector(self, tmp_path, ath: float = 100000.0) -> CycleDetector:
        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, ath)
        return CycleDetector(config, tracker)

    def test_at_ath_positive_score(self, tmp_path):
        """Price at ATH → positive price structure score."""
        det = self._make_detector(tmp_path, ath=100000.0)
        closes = make_trending_closes(250, 80000.0, 100000.0)
        result = det._analyze_price_structure(100000.0, closes)
        assert result.drawdown_from_ath == pytest.approx(0.0)
        assert result.price_structure_score > 0.0

    def test_deep_drawdown_negative(self, tmp_path):
        """50% drawdown from ATH → negative score."""
        det = self._make_detector(tmp_path, ath=100000.0)
        closes = make_trending_closes(250, 80000.0, 50000.0)
        result = det._analyze_price_structure(50000.0, closes)
        assert result.drawdown_from_ath == pytest.approx(0.5)
        assert result.price_structure_score < 0.0

    def test_above_200d_ma_bullish(self, tmp_path):
        """Price well above 200-day MA → bullish component."""
        det = self._make_detector(tmp_path, ath=120000.0)
        # 200 days of prices averaging ~80k, current price 100k
        closes = make_trending_closes(250, 70000.0, 100000.0)
        result = det._analyze_price_structure(100000.0, closes)
        assert result.distance_from_200d_ma > 0.0

    def test_below_200d_ma_bearish(self, tmp_path):
        """Price well below 200-day MA → bearish component."""
        det = self._make_detector(tmp_path, ath=100000.0)
        closes = make_trending_closes(250, 80000.0, 40000.0)
        result = det._analyze_price_structure(40000.0, closes)
        assert result.distance_from_200d_ma < 0.0

    def test_insufficient_data_for_200d_ma(self, tmp_path):
        """Less than 200 daily closes → distance_200d = 0."""
        det = self._make_detector(tmp_path, ath=100000.0)
        closes = [50000.0] * 100
        result = det._analyze_price_structure(50000.0, closes)
        assert result.distance_from_200d_ma == 0.0

    def test_position_in_range_bounded(self, tmp_path):
        """Position in range should always be 0.0–1.0."""
        det = self._make_detector(tmp_path)
        for price in [10000.0, 50000.0, 100000.0, 200000.0]:
            result = det._analyze_price_structure(price, [price] * 50)
            assert 0.0 <= result.position_in_range <= 1.0


# ─── Momentum tests ─────────────────────────────────────────────────────────

class TestMomentum:
    def _make_detector(self, tmp_path) -> CycleDetector:
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path)
        return CycleDetector(config, ath)

    def test_overbought_rsi(self, tmp_path):
        det = self._make_detector(tmp_path)
        snap = make_snapshot(rsi=80.0)
        closes, highs, lows = make_daily_data(make_trending_closes(50, 40000, 60000))
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.rsi_zone == "overbought"

    def test_oversold_rsi(self, tmp_path):
        det = self._make_detector(tmp_path)
        snap = make_snapshot(rsi=20.0)
        closes, highs, lows = make_daily_data(make_trending_closes(50, 60000, 40000))
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.rsi_zone == "oversold"

    def test_neutral_rsi(self, tmp_path):
        det = self._make_detector(tmp_path)
        snap = make_snapshot(rsi=50.0)
        closes, highs, lows = make_daily_data([50000.0] * 60)
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.rsi_zone == "neutral"

    def test_none_rsi_handled(self, tmp_path):
        det = self._make_detector(tmp_path)
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        closes, highs, lows = make_daily_data([50000.0] * 60)
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.rsi_zone == "neutral"
        assert isinstance(result.momentum_score, float)

    def test_uptrend_detected(self, tmp_path):
        det = self._make_detector(tmp_path)
        snap = make_snapshot(rsi=60.0)
        closes = make_trending_closes(60, 40000, 70000, noise=100)
        _, highs, lows = make_daily_data(closes)
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.trend_direction == "up"

    def test_downtrend_detected(self, tmp_path):
        det = self._make_detector(tmp_path)
        snap = make_snapshot(rsi=35.0)
        closes = make_trending_closes(60, 70000, 40000, noise=100)
        _, highs, lows = make_daily_data(closes)
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.trend_direction == "down"

    def test_bullish_divergence_boosts_score(self, tmp_path):
        det = self._make_detector(tmp_path)
        div = RSIDivergence(bullish=True, bearish=False, strength=0.8)
        snap = make_snapshot(rsi=35.0, rsi_divergence=div)
        closes, highs, lows = make_daily_data([50000.0] * 60)
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.rsi_bullish_divergence is True
        assert result.momentum_score > -0.5  # Divergence should offset low RSI

    def test_bearish_divergence_drags_score(self, tmp_path):
        det = self._make_detector(tmp_path)
        div = RSIDivergence(bullish=False, bearish=True, strength=0.8)
        snap = make_snapshot(rsi=65.0, rsi_divergence=div)
        closes, highs, lows = make_daily_data([50000.0] * 60)
        result = det._analyze_momentum(closes, highs, lows, snap)
        assert result.rsi_bearish_divergence is True

    def test_momentum_score_bounded(self, tmp_path):
        det = self._make_detector(tmp_path)
        for rsi_val in [5.0, 25.0, 50.0, 75.0, 95.0]:
            snap = make_snapshot(rsi=rsi_val)
            closes, highs, lows = make_daily_data(make_trending_closes(60, 40000, 60000))
            result = det._analyze_momentum(closes, highs, lows, snap)
            assert -1.0 <= result.momentum_score <= 1.0


# ─── Trend detection tests ──────────────────────────────────────────────────

class TestTrendDetection:
    def _make_detector(self, tmp_path) -> CycleDetector:
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path)
        return CycleDetector(config, ath)

    def test_strong_uptrend(self, tmp_path):
        det = self._make_detector(tmp_path)
        closes = make_trending_closes(60, 40000, 80000)
        assert det._detect_trend(closes) == "up"

    def test_strong_downtrend(self, tmp_path):
        det = self._make_detector(tmp_path)
        closes = make_trending_closes(60, 80000, 40000)
        assert det._detect_trend(closes) == "down"

    def test_sideways(self, tmp_path):
        det = self._make_detector(tmp_path)
        closes = [50000.0 + (100 if i % 2 == 0 else -100) for i in range(60)]
        assert det._detect_trend(closes) == "sideways"

    def test_insufficient_data(self, tmp_path):
        det = self._make_detector(tmp_path)
        assert det._detect_trend([50000.0] * 30) == "sideways"


# ─── Swing structure tests ──────────────────────────────────────────────────

class TestSwingStructure:
    def _make_detector(self, tmp_path) -> CycleDetector:
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path)
        return CycleDetector(config, ath)

    def test_higher_highs_and_lows(self, tmp_path):
        """Clear ascending structure should be detected."""
        det = self._make_detector(tmp_path)
        n = 30
        # Monotonically increasing base ensures only explicit dips/peaks are swings.
        # Highs: rising base with two distinct peaks
        highs = [100.0 + i * 0.5 for i in range(n)]
        highs[8] = 115.0     # First swing high (clearly above neighbors ~104)
        highs[22] = 125.0    # Second swing high (higher, clearly above neighbors ~111)

        # Lows: rising base with two distinct troughs
        lows = [90.0 + i * 0.5 for i in range(n)]
        lows[14] = 80.0      # First swing low (clearly below neighbors ~97)
        lows[26] = 85.0      # Second swing low (higher, clearly below neighbors ~103)

        hh, hl = det._detect_swing_structure(highs, lows, lookback=n)
        assert hh is True
        assert hl is True

    def test_insufficient_data(self, tmp_path):
        det = self._make_detector(tmp_path)
        hh, hl = det._detect_swing_structure([100.0] * 10, [90.0] * 10)
        assert hh is False
        assert hl is False


# ─── Volatility tests ───────────────────────────────────────────────────────

class TestVolatility:
    def _make_detector(self, tmp_path) -> CycleDetector:
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path)
        return CycleDetector(config, ath)

    def test_compression_regime(self, tmp_path):
        det = self._make_detector(tmp_path)
        regime = det._classify_vol_regime(0.05, squeeze=False)
        assert regime == VolatilityRegime.COMPRESSION

    def test_squeeze_forces_compression(self, tmp_path):
        det = self._make_detector(tmp_path)
        regime = det._classify_vol_regime(0.50, squeeze=True)
        assert regime == VolatilityRegime.COMPRESSION

    def test_normal_regime(self, tmp_path):
        det = self._make_detector(tmp_path)
        regime = det._classify_vol_regime(0.50, squeeze=False)
        assert regime == VolatilityRegime.NORMAL

    def test_elevated_regime(self, tmp_path):
        det = self._make_detector(tmp_path)
        regime = det._classify_vol_regime(0.80, squeeze=False)
        assert regime == VolatilityRegime.ELEVATED

    def test_extreme_regime(self, tmp_path):
        det = self._make_detector(tmp_path)
        regime = det._classify_vol_regime(0.95, squeeze=False)
        assert regime == VolatilityRegime.EXTREME

    def test_volatility_score_compression_positive(self, tmp_path):
        """Compression = breakout potential → positive score."""
        det = self._make_detector(tmp_path)
        snap = make_snapshot(vol_percentile=0.10, bollinger_squeeze=True)
        closes, highs, lows = make_daily_data(make_trending_closes(200, 50000, 55000))
        regime, score = det._analyze_volatility(closes, highs, lows, snap)
        assert score > 0.0

    def test_volatility_score_extreme_negative(self, tmp_path):
        """Extreme volatility → negative score (risk)."""
        det = self._make_detector(tmp_path)
        snap = make_snapshot(vol_percentile=0.95)
        # Use < 120 closes so BB percentile can't compute, only ATR percentile used
        closes, highs, lows = make_daily_data(make_trending_closes(100, 50000, 55000))
        regime, score = det._analyze_volatility(closes, highs, lows, snap)
        assert score < 0.0


# ─── Phase determination tests ───────────────────────────────────────────────

class TestPhaseDetermination:
    def _make_detector(self, tmp_path) -> CycleDetector:
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path)
        return CycleDetector(config, ath)

    def test_strong_bearish_signals_yield_capitulation_or_bear(self, tmp_path):
        """Very negative composite → capitulation or early_bear."""
        det = self._make_detector(tmp_path)
        phase, conf = det._determine_phase(
            composite=-0.7, time_score=-0.5, price_score=-0.8,
            momentum_score=-0.7, vol_regime=VolatilityRegime.ELEVATED,
            cycle_progress=0.9,
        )
        assert phase in (CyclePhase.CAPITULATION, CyclePhase.EARLY_BEAR)

    def test_strong_bullish_signals_yield_growth_or_euphoria(self, tmp_path):
        """Very positive composite → growth or euphoria."""
        det = self._make_detector(tmp_path)
        phase, conf = det._determine_phase(
            composite=0.7, time_score=0.8, price_score=0.6,
            momentum_score=0.7, vol_regime=VolatilityRegime.NORMAL,
            cycle_progress=0.45,
        )
        assert phase in (CyclePhase.GROWTH, CyclePhase.EUPHORIA)

    def test_neutral_signals(self, tmp_path):
        """Neutral signals → accumulation or early_bull."""
        det = self._make_detector(tmp_path)
        phase, conf = det._determine_phase(
            composite=-0.1, time_score=0.0, price_score=-0.1,
            momentum_score=0.0, vol_regime=VolatilityRegime.NORMAL,
            cycle_progress=0.20,
        )
        assert phase in (CyclePhase.ACCUMULATION, CyclePhase.EARLY_BULL,
                         CyclePhase.EARLY_BEAR)

    def test_confidence_bounded(self, tmp_path):
        det = self._make_detector(tmp_path)
        for composite in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            phase, conf = det._determine_phase(
                composite=composite, time_score=composite,
                price_score=composite, momentum_score=composite,
                vol_regime=VolatilityRegime.NORMAL, cycle_progress=0.5,
            )
            assert 0.0 <= conf <= 1.0


# ─── Hysteresis tests ───────────────────────────────────────────────────────

class TestHysteresis:
    def test_phase_holds_on_minor_change(self, tmp_path):
        """Small score changes should not cause phase flipping."""
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path, 100000.0)
        det = CycleDetector(config, ath)

        # Establish a phase
        det._last_phase = CyclePhase.GROWTH
        det._phase_hold_cycles = 3  # Recently transitioned

        # Slightly bearish signals — shouldn't transition yet
        phase, _ = det._determine_phase(
            composite=0.15, time_score=0.3, price_score=0.2,
            momentum_score=0.0, vol_regime=VolatilityRegime.NORMAL,
            cycle_progress=0.45,
        )
        # Should hold GROWTH due to hysteresis
        assert phase == CyclePhase.GROWTH

    def test_strong_signal_overrides_hysteresis(self, tmp_path):
        """Very strong signals should override hysteresis."""
        config = make_config(tmp_path)
        ath = make_ath_tracker(tmp_path, 100000.0)
        det = CycleDetector(config, ath)

        det._last_phase = CyclePhase.GROWTH
        det._phase_hold_cycles = 5

        # Very strong bearish signals
        phase, _ = det._determine_phase(
            composite=-0.8, time_score=-0.6, price_score=-0.9,
            momentum_score=-0.8, vol_regime=VolatilityRegime.EXTREME,
            cycle_progress=0.85,
        )
        # Should transition despite hysteresis
        assert phase != CyclePhase.GROWTH


# ─── Downstream adjustments tests ───────────────────────────────────────────

class TestDownstreamAdjustments:
    def test_size_multiplier_accumulation_larger(self, tmp_path):
        """Accumulation phase should have > 1.0 size multiplier."""
        mult = CycleDetector._phase_size_multiplier(CyclePhase.ACCUMULATION, 0.8)
        assert mult > 1.0

    def test_size_multiplier_distribution_smaller(self, tmp_path):
        """Distribution phase should have < 1.0 size multiplier."""
        mult = CycleDetector._phase_size_multiplier(CyclePhase.DISTRIBUTION, 0.8)
        assert mult < 1.0

    def test_size_multiplier_low_confidence_near_1(self, tmp_path):
        """Low confidence → multiplier should be near 1.0 regardless of phase."""
        mult = CycleDetector._phase_size_multiplier(CyclePhase.CAPITULATION, 0.1)
        assert 0.9 < mult < 1.1

    def test_size_multiplier_always_positive(self, tmp_path):
        for phase in CyclePhase:
            for conf in [0.0, 0.1, 0.5, 0.8, 1.0]:
                mult = CycleDetector._phase_size_multiplier(phase, conf)
                assert mult > 0.0


# ─── Full analyze() integration tests ───────────────────────────────────────

class TestAnalyzeIntegration:
    def _run_analysis(
        self,
        tmp_path: Path,
        price: float,
        ath: float,
        closes: list[float],
        rsi_val: float = 50.0,
        squeeze: bool = False,
        vol_pct: float = 0.5,
    ) -> CycleState:
        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, ath)
        det = CycleDetector(config, tracker)

        snap = make_snapshot(
            price=price, rsi=rsi_val,
            bollinger_squeeze=squeeze, vol_percentile=vol_pct,
        )
        _, highs, lows = make_daily_data(closes)
        return det.analyze(snap, closes, highs, lows)

    def test_bull_market_scenario(self, tmp_path):
        """Simulated bull market conditions should yield bullish phase."""
        closes = make_trending_closes(250, 40000, 90000)
        state = self._run_analysis(
            tmp_path, price=90000.0, ath=92000.0,
            closes=closes, rsi_val=65.0, vol_pct=0.5,
        )
        assert state.phase in (CyclePhase.EARLY_BULL, CyclePhase.GROWTH, CyclePhase.EUPHORIA)
        assert state.composite_score > -0.3
        assert state.profit_taking_active or state.phase == CyclePhase.EARLY_BULL

    def test_bear_market_scenario(self, tmp_path):
        """Simulated bear market should yield bearish phase."""
        closes = make_trending_closes(250, 80000, 35000)
        state = self._run_analysis(
            tmp_path, price=35000.0, ath=100000.0,
            closes=closes, rsi_val=25.0, vol_pct=0.85,
        )
        assert state.phase in (CyclePhase.CAPITULATION, CyclePhase.EARLY_BEAR,
                                CyclePhase.ACCUMULATION)
        assert state.drawdown_tolerance > 0.1

    def test_accumulation_scenario(self, tmp_path):
        """Low price, low vol, sideways → accumulation."""
        closes = [35000.0 + random.uniform(-500, 500) for _ in range(250)]
        random.seed(42)
        state = self._run_analysis(
            tmp_path, price=35000.0, ath=100000.0,
            closes=closes, rsi_val=40.0, vol_pct=0.20,
        )
        assert state.position_size_multiplier >= 0.9  # Should not shrink much

    def test_ath_updates_during_analysis(self, tmp_path):
        """ATH should auto-update when price exceeds it."""
        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, 80000.0)
        det = CycleDetector(config, tracker)

        snap = make_snapshot(price=85000.0, rsi=65.0)
        closes = make_trending_closes(250, 40000, 85000)
        _, highs, lows = make_daily_data(closes)

        state = det.analyze(snap, closes, highs, lows)
        assert state.ath_eur == 85000.0  # Updated
        assert tracker.ath_eur == 85000.0

    def test_output_structure_complete(self, tmp_path):
        """All CycleState fields should be populated."""
        closes = make_trending_closes(250, 50000, 60000)
        state = self._run_analysis(
            tmp_path, price=60000.0, ath=70000.0,
            closes=closes,
        )

        assert isinstance(state.phase, CyclePhase)
        assert 0.0 <= state.phase_confidence <= 1.0
        assert -1.0 <= state.time_score <= 1.0
        assert -1.0 <= state.price_score <= 1.0
        assert -1.0 <= state.momentum_score <= 1.0
        assert -1.0 <= state.volatility_score <= 1.0
        assert -1.0 <= state.composite_score <= 1.0
        assert isinstance(state.momentum, MomentumState)
        assert isinstance(state.price_structure, PriceStructure)
        assert isinstance(state.volatility_regime, VolatilityRegime)
        assert state.cycle_day >= 0
        assert 0.0 <= state.cycle_progress <= 1.0
        assert state.ath_eur > 0
        assert state.drawdown_tolerance > 0
        assert state.position_size_multiplier > 0
        assert isinstance(state.profit_taking_active, bool)
        assert state.timestamp > 0

    def test_frozen_output(self, tmp_path):
        closes = make_trending_closes(250, 50000, 60000)
        state = self._run_analysis(tmp_path, price=60000.0, ath=70000.0, closes=closes)
        with pytest.raises(AttributeError):
            state.phase = CyclePhase.CAPITULATION  # type: ignore

    def test_minimal_data_doesnt_crash(self, tmp_path):
        """Even with minimal data, analysis should complete."""
        closes = [50000.0] * 20
        state = self._run_analysis(
            tmp_path, price=50000.0, ath=100000.0,
            closes=closes, rsi_val=50.0,
        )
        assert isinstance(state.phase, CyclePhase)

    def test_empty_data_doesnt_crash(self, tmp_path):
        """Empty price data should not raise."""
        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, 100000.0)
        det = CycleDetector(config, tracker)

        snap = make_snapshot(price=50000.0)
        state = det.analyze(snap, [], [], [])
        assert isinstance(state.phase, CyclePhase)


# ─── Phase fit scoring tests ────────────────────────────────────────────────

class TestPhaseFitScoring:
    def test_perfect_fit(self):
        """Values at center of target range → high score."""
        score = CycleDetector._score_phase_fit(
            composite=0.5, time_score=0.5, price_score=0.5, momentum_score=0.5,
            target_composite=(0.3, 0.7),
            target_momentum=(0.3, 0.7),
            target_price=(0.3, 0.7),
        )
        assert score > 0.8

    def test_outside_range_penalized(self):
        """Values far outside target → low score."""
        score = CycleDetector._score_phase_fit(
            composite=-0.8, time_score=-0.8, price_score=-0.8, momentum_score=-0.8,
            target_composite=(0.5, 1.0),
            target_momentum=(0.5, 1.0),
            target_price=(0.5, 1.0),
        )
        assert score < 0.3

    def test_score_non_negative(self):
        """Fit score should never go below 0."""
        score = CycleDetector._score_phase_fit(
            composite=-1.0, time_score=-1.0, price_score=-1.0, momentum_score=-1.0,
            target_composite=(0.5, 1.0),
            target_momentum=(0.5, 1.0),
            target_price=(0.5, 1.0),
        )
        assert score >= 0.0
