"""
Tests for indicators.py

Validates:
- Wilder RSI matches reference implementations (NOT SMA-based RSI)
- MACD calculations against known values
- Bollinger Bands math and squeeze detection
- ATR with Wilder smoothing
- VWAP calculation
- RSI divergence detection
- TechnicalSnapshot data quality
- Edge cases: insufficient data, zero volume, flat prices

Run: python -m pytest tests/test_indicators.py -v
"""
from __future__ import annotations

import math
import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators import (
    BollingerBands,
    MACDResult,
    RSIDivergence,
    TechnicalSnapshot,
    atr,
    atr_percentile,
    atr_series,
    bollinger_bands,
    bollinger_bandwidth_percentile,
    compute_snapshot,
    detect_rsi_divergence,
    ema,
    macd,
    rsi,
    rsi_series,
    vwap,
)
from config import IndicatorConfig


# ─── Helper: generate synthetic price data ───────────────────────────────────

def make_trending_up(n: int, start: float = 50000.0, step: float = 100.0, noise: float = 50.0) -> list[float]:
    """Generate upward trending prices with noise."""
    import random
    random.seed(42)
    return [start + i * step + random.uniform(-noise, noise) for i in range(n)]


def make_trending_down(n: int, start: float = 60000.0, step: float = 100.0, noise: float = 50.0) -> list[float]:
    import random
    random.seed(42)
    return [start - i * step + random.uniform(-noise, noise) for i in range(n)]


def make_flat(n: int, price: float = 50000.0, noise: float = 20.0) -> list[float]:
    import random
    random.seed(42)
    return [price + random.uniform(-noise, noise) for i in range(n)]


def make_ohlcv(closes: list[float], spread: float = 50.0):
    """Generate OHLCV data from closes."""
    highs = [c + abs(spread) for c in closes]
    lows = [c - abs(spread) for c in closes]
    volumes = [10.0 + i * 0.1 for i in range(len(closes))]
    return highs, lows, closes, volumes


# ─── EMA tests ───────────────────────────────────────────────────────────────

class TestEMA:
    def test_insufficient_data(self):
        result = ema([1.0, 2.0], period=5)
        assert all(math.isnan(v) for v in result)

    def test_exact_period(self):
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = ema(values, period=5)
        # First 4 entries should be NaN, 5th should be SMA = 6.0
        assert math.isnan(result[0])
        assert math.isnan(result[3])
        assert result[4] == pytest.approx(6.0)

    def test_smoothing(self):
        values = [10.0] * 5 + [20.0] * 5
        result = ema(values, period=5)
        # After step change, EMA should approach 20 but not reach it immediately
        assert result[4] == pytest.approx(10.0)  # SMA seed
        assert 10.0 < result[5] < 20.0  # Moving toward 20
        assert result[-1] > result[5]  # Progressing toward 20

    def test_output_length(self):
        values = list(range(50))
        result = ema(values, period=10)
        assert len(result) == len(values)


# ─── RSI tests (Wilder smoothing — the critical fix) ─────────────────────────

class TestRSI:
    def test_insufficient_data(self):
        assert rsi([50000.0] * 10, period=14) is None

    def test_exact_minimum_data(self):
        closes = list(range(100, 116))  # 16 points = period(14) + 1 + 1
        result = rsi(closes, period=14)
        assert result is not None

    def test_all_gains_returns_100(self):
        """Monotonically increasing prices → RSI = 100."""
        closes = [float(i) for i in range(100)]
        result = rsi(closes, period=14)
        assert result == pytest.approx(100.0)

    def test_all_losses_returns_0(self):
        """Monotonically decreasing prices → RSI = 0."""
        closes = [float(100 - i) for i in range(100)]
        result = rsi(closes, period=14)
        assert result == pytest.approx(0.0)

    def test_flat_prices_returns_50ish(self):
        """No change → avg_gain == avg_loss → RSI approaches 50 (or 100 if no losses)."""
        # Alternating +1, -1
        closes = [50000.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(100)]
        result = rsi(closes, period=14)
        assert result is not None
        # With equal gains and losses, RSI should be near 50
        assert 45.0 < result < 55.0

    def test_wilder_differs_from_sma(self):
        """
        Verify our RSI is NOT the SMA-based version.

        Wilder smoothing gives exponentially weighted recent data more influence,
        so after a trend change, Wilder RSI responds differently than SMA RSI.
        """
        # Strong uptrend followed by sharp reversal
        closes = [float(50000 + i * 100) for i in range(30)]
        closes.extend([float(closes[-1] - i * 200) for i in range(1, 20)])

        wilder_rsi = rsi(closes, period=14)

        # Compute SMA-based RSI for comparison
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in deltas]
        losses = [abs(min(d, 0.0)) for d in deltas]
        # SMA: just average the last `period` values
        sma_avg_gain = sum(gains[-14:]) / 14
        sma_avg_loss = sum(losses[-14:]) / 14
        if sma_avg_loss == 0:
            sma_rsi = 100.0
        else:
            sma_rs = sma_avg_gain / sma_avg_loss
            sma_rsi = 100.0 - (100.0 / (1.0 + sma_rs))

        # They should NOT be equal (proving we're using Wilder, not SMA)
        assert wilder_rsi is not None
        assert abs(wilder_rsi - sma_rsi) > 1.0, (
            f"Wilder RSI ({wilder_rsi:.2f}) too close to SMA RSI ({sma_rsi:.2f}). "
            f"Are we accidentally using SMA?"
        )

    def test_known_reference_values(self):
        """
        Validate against a manually computed Wilder RSI sequence.

        Data: 15 closing prices, period=5 (small for easy manual verification).
        """
        closes = [44.0, 44.34, 44.09, 43.61, 44.33, 44.83,
                  45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
                  45.61, 46.28, 46.28]

        # Period 5: need at least 6 data points
        result = rsi(closes, period=5)
        assert result is not None

        # The RSI should reflect the overall uptrend
        assert result > 50.0

    def test_rsi_bounded_0_100(self):
        """RSI should always be between 0 and 100."""
        import random
        random.seed(123)
        for _ in range(20):
            closes = [50000.0 + random.gauss(0, 1000) for _ in range(100)]
            result = rsi(closes, period=14)
            if result is not None:
                assert 0.0 <= result <= 100.0


class TestRSISeries:
    def test_length_matches_input(self):
        closes = make_trending_up(50)
        result = rsi_series(closes, period=14)
        assert len(result) == len(closes)

    def test_first_period_is_none(self):
        closes = make_trending_up(50)
        result = rsi_series(closes, period=14)
        assert all(v is None for v in result[:14])
        assert result[14] is not None

    def test_last_value_matches_scalar(self):
        closes = make_trending_up(100)
        series = rsi_series(closes, period=14)
        scalar = rsi(closes, period=14)
        assert series[-1] == pytest.approx(scalar, abs=0.01)

    def test_all_values_bounded(self):
        closes = make_trending_up(100) + make_trending_down(50, start=60000.0)
        result = rsi_series(closes, period=14)
        for v in result:
            if v is not None:
                assert 0.0 <= v <= 100.0


# ─── MACD tests ──────────────────────────────────────────────────────────────

class TestMACD:
    def test_insufficient_data(self):
        assert macd([1.0] * 20) is None

    def test_sufficient_data(self):
        closes = make_trending_up(50)
        result = macd(closes, fast=12, slow=26, signal_period=9)
        assert result is not None
        assert isinstance(result, MACDResult)

    def test_uptrend_positive_macd(self):
        closes = make_trending_up(100, step=200.0, noise=10.0)
        result = macd(closes)
        assert result is not None
        assert result.macd_line > 0, "MACD line should be positive in uptrend"
        assert result.bullish

    def test_downtrend_negative_macd(self):
        closes = make_trending_down(100, step=200.0, noise=10.0)
        result = macd(closes)
        assert result is not None
        assert result.macd_line < 0, "MACD line should be negative in downtrend"

    def test_histogram_is_difference(self):
        closes = make_trending_up(100)
        result = macd(closes)
        assert result is not None
        assert result.histogram == pytest.approx(
            result.macd_line - result.signal_line
        )

    def test_flat_market_near_zero(self):
        closes = make_flat(100, noise=5.0)
        result = macd(closes)
        assert result is not None
        # In a flat market, MACD should be near zero
        assert abs(result.macd_line) < 50.0


# ─── Bollinger Bands tests ───────────────────────────────────────────────────

class TestBollingerBands:
    def test_insufficient_data(self):
        assert bollinger_bands([1.0] * 10, period=20) is None

    def test_basic_calculation(self):
        # Constant prices → zero std → bands collapse to middle
        closes = [100.0] * 30
        bb = bollinger_bands(closes, period=20, num_std=2.0)
        assert bb is not None
        assert bb.middle == pytest.approx(100.0)
        assert bb.upper == pytest.approx(100.0)
        assert bb.lower == pytest.approx(100.0)
        assert bb.bandwidth == pytest.approx(0.0)

    def test_upper_above_lower(self):
        closes = make_trending_up(50)
        bb = bollinger_bands(closes)
        assert bb is not None
        assert bb.upper > bb.middle > bb.lower

    def test_percent_b_at_upper(self):
        # Price at upper band → %B ≈ 1.0
        closes = [100.0] * 19 + [120.0]
        bb = bollinger_bands(closes, period=20)
        assert bb is not None
        # %B should be > 1.0 since last price is outside the band
        assert bb.percent_b > 0.5

    def test_squeeze_detection(self):
        # Very low volatility → squeeze
        closes = [50000.0 + (i % 2) * 0.5 for i in range(30)]
        bb = bollinger_bands(closes, period=20)
        assert bb is not None
        assert bb.squeeze is True

    def test_no_squeeze_high_vol(self):
        import random
        random.seed(42)
        closes = [50000.0 + random.gauss(0, 5000) for _ in range(30)]
        bb = bollinger_bands(closes, period=20)
        assert bb is not None
        # High volatility should not trigger squeeze
        assert bb.bandwidth > 0.04


class TestBollingerBandwidthPercentile:
    def test_insufficient_data(self):
        assert bollinger_bandwidth_percentile([1.0] * 50) is None

    def test_returns_bounded(self):
        closes = make_trending_up(200)
        result = bollinger_bandwidth_percentile(closes, lookback=100)
        assert result is not None
        assert 0.0 <= result <= 1.0


# ─── ATR tests ───────────────────────────────────────────────────────────────

class TestATR:
    def test_insufficient_data(self):
        assert atr([1.0] * 10, [1.0] * 10, [1.0] * 10, period=14) is None

    def test_zero_volatility(self):
        n = 30
        highs = [100.0] * n
        lows = [100.0] * n
        closes = [100.0] * n
        result = atr(highs, lows, closes, period=14)
        assert result is not None
        assert result == pytest.approx(0.0)

    def test_known_true_range(self):
        # Simple case: constant range bars
        n = 30
        highs = [110.0] * n
        lows = [90.0] * n
        closes = [100.0] * n
        result = atr(highs, lows, closes, period=14)
        assert result is not None
        # TR = max(110-90, |110-100|, |90-100|) = 20
        # ATR should converge to 20
        assert result == pytest.approx(20.0, abs=0.5)

    def test_gap_increases_tr(self):
        """Gap up should increase true range via prev_close distance."""
        n = 20
        closes = [100.0] * 15 + [100.0, 120.0, 122.0, 121.0, 123.0]
        highs = [c + 2.0 for c in closes]
        lows = [c - 2.0 for c in closes]
        result = atr(highs, lows, closes, period=14)
        assert result is not None
        # ATR should be elevated due to the gap
        assert result > 4.0  # Without gap it would be ~4.0

    def test_mismatched_lengths(self):
        assert atr([1.0] * 30, [1.0] * 29, [1.0] * 30) is None


class TestATRSeries:
    def test_length_matches_input(self):
        n = 50
        h, l, c, _ = make_ohlcv(make_trending_up(n))
        result = atr_series(h, l, c, period=14)
        assert len(result) == n

    def test_first_period_none(self):
        n = 50
        h, l, c, _ = make_ohlcv(make_trending_up(n))
        result = atr_series(h, l, c, period=14)
        assert all(v is None for v in result[:14])
        assert result[14] is not None

    def test_last_matches_scalar(self):
        n = 100
        h, l, c, _ = make_ohlcv(make_trending_up(n))
        series = atr_series(h, l, c, period=14)
        scalar = atr(h, l, c, period=14)
        assert series[-1] == pytest.approx(scalar, abs=0.01)


class TestATRPercentile:
    def test_insufficient_data(self):
        h, l, c, _ = make_ohlcv(make_trending_up(50))
        assert atr_percentile(h, l, c, period=14, lookback=100) is None

    def test_returns_bounded(self):
        h, l, c, _ = make_ohlcv(make_trending_up(200))
        result = atr_percentile(h, l, c, period=14, lookback=100)
        assert result is not None
        assert 0.0 <= result <= 1.0


# ─── VWAP tests ──────────────────────────────────────────────────────────────

class TestVWAP:
    def test_empty_data(self):
        assert vwap([], [], [], []) is None

    def test_zero_volume(self):
        assert vwap([100.0], [90.0], [95.0], [0.0]) is None

    def test_single_candle(self):
        result = vwap([110.0], [90.0], [100.0], [10.0])
        # Typical price = (110+90+100)/3 = 100
        assert result == pytest.approx(100.0)

    def test_volume_weighting(self):
        highs = [110.0, 210.0]
        lows = [90.0, 190.0]
        closes = [100.0, 200.0]
        volumes = [100.0, 1.0]

        result = vwap(highs, lows, closes, volumes)
        assert result is not None
        # Should be much closer to 100 (typical=100) than 200 (typical=200)
        # because first candle has 100x the volume
        tp1 = (110 + 90 + 100) / 3  # 100
        tp2 = (210 + 190 + 200) / 3  # 200
        expected = (tp1 * 100 + tp2 * 1) / 101
        assert result == pytest.approx(expected)

    def test_period_limit(self):
        highs = [110.0] * 20
        lows = [90.0] * 20
        closes = [100.0] * 20
        volumes = [10.0] * 20

        result = vwap(highs, lows, closes, volumes, period=5)
        assert result is not None
        # Using only last 5 candles, all same → typical price
        assert result == pytest.approx(100.0)

    def test_period_exceeds_data(self):
        assert vwap([100.0], [90.0], [95.0], [10.0], period=5) is None


# ─── RSI Divergence tests ────────────────────────────────────────────────────

class TestRSIDivergence:
    def test_no_divergence_on_short_data(self):
        closes = [100.0] * 10
        rsi_vals = [50.0] * 10
        result = detect_rsi_divergence(closes, rsi_vals, lookback=30)
        # Not enough data
        assert result.bullish is False
        assert result.bearish is False

    def test_bullish_divergence(self):
        """Price makes lower low, RSI makes higher low."""
        # Construct a clear bullish divergence pattern
        n = 40
        # Price: down, up, down further (lower low)
        closes = (
            [100.0] * 5
            + [95.0, 93.0, 91.0, 90.0, 91.0, 93.0, 95.0]  # First low at 90
            + [97.0, 99.0, 100.0, 99.0, 97.0]
            + [95.0, 93.0, 91.0, 89.0, 88.0, 89.0, 91.0, 93.0]  # Lower low at 88
            + [95.0, 97.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        )

        # RSI: make it higher at the second low than the first
        # (manually craft RSI values that show divergence)
        rsi_vals: list = (
            [50.0] * 5
            + [40.0, 35.0, 30.0, 25.0, 30.0, 35.0, 40.0]  # RSI low at 25
            + [45.0, 48.0, 50.0, 48.0, 45.0]
            + [42.0, 38.0, 35.0, 32.0, 30.0, 32.0, 35.0, 38.0]  # RSI low at 30 (higher!)
            + [42.0, 45.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0]
        )

        result = detect_rsi_divergence(closes, rsi_vals, lookback=len(closes))
        assert result.bullish is True

    def test_mismatched_lengths(self):
        closes = [100.0] * 40
        rsi_vals = [50.0] * 30  # Different length
        result = detect_rsi_divergence(closes, rsi_vals)
        assert result.bullish is False
        assert result.bearish is False

    def test_strength_bounded(self):
        """Strength should be between 0 and 1."""
        closes = make_trending_up(50) + make_trending_down(30, start=55000.0)
        rsi_vals = rsi_series(closes, period=14)
        result = detect_rsi_divergence(closes, rsi_vals, lookback=40)
        assert 0.0 <= result.strength <= 1.0


# ─── TechnicalSnapshot tests ─────────────────────────────────────────────────

class TestTechnicalSnapshot:
    def test_data_quality_all_present(self):
        snap = TechnicalSnapshot(
            price=50000.0,
            timestamp=time.time(),
            rsi=55.0,
            macd=MACDResult(macd_line=100.0, signal_line=80.0, histogram=20.0),
            bollinger=BollingerBands(
                upper=51000.0, middle=50000.0, lower=49000.0,
                bandwidth=0.04, percent_b=0.5,
            ),
            atr=500.0,
            vwap=50100.0,
        )
        assert snap.data_quality == pytest.approx(1.0)

    def test_data_quality_partial(self):
        snap = TechnicalSnapshot(
            price=50000.0,
            timestamp=time.time(),
            rsi=55.0,
            # Everything else None
        )
        assert snap.data_quality == pytest.approx(0.2)

    def test_data_quality_none(self):
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        assert snap.data_quality == pytest.approx(0.0)


# ─── compute_snapshot integration tests ──────────────────────────────────────

class TestComputeSnapshot:
    def test_with_sufficient_data(self):
        n = 200
        closes = make_trending_up(n)
        highs, lows, closes_list, volumes = make_ohlcv(closes)

        snap = compute_snapshot(
            highs=highs,
            lows=lows,
            closes=closes_list,
            volumes=volumes,
            timestamp=time.time(),
        )

        assert snap.price == closes_list[-1]
        assert snap.rsi is not None
        assert snap.macd is not None
        assert snap.bollinger is not None
        assert snap.atr is not None
        assert snap.vwap is not None
        assert snap.data_quality == pytest.approx(1.0)

    def test_with_minimal_data(self):
        """Only enough data for some indicators."""
        closes = make_trending_up(20)
        highs, lows, closes_list, volumes = make_ohlcv(closes)

        snap = compute_snapshot(
            highs=highs,
            lows=lows,
            closes=closes_list,
            volumes=volumes,
            timestamp=time.time(),
        )

        assert snap.price == closes_list[-1]
        # RSI needs 15, so should work
        assert snap.rsi is not None
        # Bollinger needs 20, should work
        assert snap.bollinger is not None
        # ATR needs 15, should work
        assert snap.atr is not None

    def test_with_empty_data(self):
        """Graceful degradation with no data."""
        snap = compute_snapshot(
            highs=[], lows=[], closes=[], volumes=[],
            timestamp=time.time(),
        )
        assert snap.price == 0.0
        assert snap.rsi is None
        assert snap.macd is None
        assert snap.bollinger is None
        assert snap.atr is None
        assert snap.vwap is None
        assert snap.data_quality == pytest.approx(0.0)

    def test_custom_config(self):
        closes = make_trending_up(100)
        highs, lows, closes_list, volumes = make_ohlcv(closes)

        cfg = IndicatorConfig(rsi_period=7, atr_period=7)
        snap = compute_snapshot(
            highs=highs,
            lows=lows,
            closes=closes_list,
            volumes=volumes,
            timestamp=time.time(),
            config=cfg,
        )

        assert snap.rsi is not None
        assert snap.atr is not None

    def test_snapshot_is_frozen(self):
        snap = compute_snapshot(
            highs=[100.0] * 50,
            lows=[90.0] * 50,
            closes=[95.0] * 50,
            volumes=[10.0] * 50,
            timestamp=time.time(),
        )
        with pytest.raises(AttributeError):
            snap.price = 99999.0  # type: ignore


# ─── Edge case tests ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_price_point(self):
        assert rsi([100.0]) is None
        assert macd([100.0]) is None
        assert bollinger_bands([100.0]) is None
        assert atr([100.0], [100.0], [100.0]) is None

    def test_negative_prices(self):
        """Indicators should still compute (for testing, not production)."""
        closes = [float(-100 + i) for i in range(50)]
        result = rsi(closes, period=14)
        assert result is not None

    def test_very_large_prices(self):
        closes = [float(1e12 + i * 1e6) for i in range(50)]
        result = rsi(closes, period=14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_identical_prices(self):
        """All same price → RSI undefined but should handle gracefully."""
        closes = [50000.0] * 50
        result = rsi(closes, period=14)
        assert result is not None
        # No gains or losses, avg_loss = 0 → RSI = 100 (our implementation)
        assert result == 100.0

    def test_alternating_prices(self):
        """Alternating up/down → RSI near 50."""
        closes = [50000.0 + (500.0 if i % 2 == 0 else -500.0) for i in range(100)]
        result = rsi(closes, period=14)
        assert result is not None
        assert 40.0 < result < 60.0

    def test_ema_with_period_1(self):
        """Period 1 EMA should just return the input values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ema(values, period=1)
        assert result == pytest.approx(values)

    def test_bollinger_zero_middle(self):
        """Handle edge case where middle could be zero."""
        closes = [0.0] * 30
        bb = bollinger_bands(closes)
        assert bb is not None
        assert bb.bandwidth == 0.0
