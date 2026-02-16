"""
Standard technical indicators for trading signals.

All implementations follow canonical definitions used by TradingView,
professional charting software, and institutional quant systems.

Critical: RSI uses Wilder's exponential smoothing — NOT SMA. This was
a production bug in the previous bot that caused systematic signal divergence
from what every other trader and platform sees.

Design:
- All functions take simple lists (closes, highs, lows, volumes) and return
  Optional values (None when insufficient data).
- A TechnicalSnapshot dataclass bundles all indicators for one point in time.
- Pure math — no I/O, no side effects, no logging in hot paths.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from config import IndicatorConfig

logger = logging.getLogger(__name__)


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MACDResult:
    """MACD indicator components."""
    macd_line: float      # MACD line (fast EMA - slow EMA)
    signal_line: float    # Signal line (EMA of MACD line)
    histogram: float      # MACD - Signal

    @property
    def bullish(self) -> bool:
        """MACD line above signal line."""
        return self.macd_line > self.signal_line

    @property
    def bearish(self) -> bool:
        return self.macd_line < self.signal_line


@dataclass(frozen=True)
class BollingerBands:
    """Bollinger Bands indicator components."""
    upper: float          # Upper band (SMA + k * σ)
    middle: float         # Middle band (SMA)
    lower: float          # Lower band (SMA - k * σ)
    bandwidth: float      # (upper - lower) / middle — measures volatility
    percent_b: float      # (price - lower) / (upper - lower) — position within bands

    @property
    def squeeze(self) -> bool:
        """Bandwidth below 4% suggests a squeeze (low volatility, breakout incoming)."""
        return self.bandwidth < 0.04


@dataclass(frozen=True)
class RSIDivergence:
    """Detected RSI divergence signal."""
    bullish: bool         # Price lower low + RSI higher low
    bearish: bool         # Price higher high + RSI lower high
    strength: float       # 0.0–1.0 magnitude of divergence


@dataclass(frozen=True)
class TechnicalSnapshot:
    """
    Bundles all indicator values for a single point in time.

    This is the interface between raw indicator math and the signal engine.
    Every field is Optional because any indicator can fail due to insufficient data.
    """
    price: float
    timestamp: float

    # Core indicators
    rsi: Optional[float] = None
    macd: Optional[MACDResult] = None
    bollinger: Optional[BollingerBands] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None

    # Derived signals
    rsi_divergence: Optional[RSIDivergence] = None
    bollinger_squeeze: bool = False
    volatility_percentile: Optional[float] = None  # ATR percentile over lookback

    @property
    def data_quality(self) -> float:
        """Fraction of indicators that have valid data (0.0–1.0)."""
        fields = [self.rsi, self.macd, self.bollinger, self.atr, self.vwap]
        valid = sum(1 for f in fields if f is not None)
        return valid / len(fields)


# ─── Exponential Moving Average (foundation for MACD) ────────────────────────

def ema(values: list[float], period: int) -> list[float]:
    """
    Exponential Moving Average.

    Seed: SMA of the first `period` values.
    Then: EMA_t = price * k + EMA_{t-1} * (1 - k), where k = 2 / (period + 1).

    Returns a list the same length as input. First `period-1` entries
    are NaN (insufficient data).
    """
    if len(values) < period:
        return [float("nan")] * len(values)

    k = 2.0 / (period + 1)
    result = [float("nan")] * (period - 1)

    # Seed with SMA
    seed = sum(values[:period]) / period
    result.append(seed)

    # EMA from period onwards
    for i in range(period, len(values)):
        result.append(values[i] * k + result[-1] * (1 - k))

    return result


# ─── RSI (Wilder's Smoothing) ────────────────────────────────────────────────

def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """
    Relative Strength Index using Wilder's exponential smoothing.

    NOT SMA-based. Wilder smoothing:
        avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
        avg_loss = (prev_avg_loss * (period - 1) + current_loss) / period

    First value is seeded with SMA of gains/losses over the initial window.

    Returns the most recent RSI value, or None if insufficient data.
    Requires at least period + 1 data points.
    """
    needed = period + 1
    if len(closes) < needed:
        return None

    # Calculate price changes
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    # Seed: SMA over first `period` changes
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for the rest
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(closes: list[float], period: int = 14) -> list[Optional[float]]:
    """
    Full RSI series using Wilder's smoothing.

    Returns a list the same length as `closes`.
    First `period` entries are None (insufficient data).
    """
    if len(closes) < period + 1:
        return [None] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    result: list[Optional[float]] = [None] * period

    # Seed
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0.0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100.0 - (100.0 / (1.0 + rs)))

    # Wilder smoothing for remaining
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - (100.0 / (1.0 + rs)))

    return result


# ─── MACD ─────────────────────────────────────────────────────────────────────

def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[MACDResult]:
    """
    Moving Average Convergence Divergence.

    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(signal_period) of MACD line
    Histogram = MACD - Signal

    Returns the most recent MACD values, or None if insufficient data.
    Requires at least slow + signal_period - 1 data points.
    """
    needed = slow + signal_period - 1
    if len(closes) < needed:
        return None

    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)

    # MACD line starts where slow EMA becomes valid
    macd_line_values: list[float] = []
    for i in range(len(closes)):
        if math.isnan(fast_ema[i]) or math.isnan(slow_ema[i]):
            continue
        macd_line_values.append(fast_ema[i] - slow_ema[i])

    if len(macd_line_values) < signal_period:
        return None

    signal_ema = ema(macd_line_values, signal_period)
    if math.isnan(signal_ema[-1]):
        return None

    macd_val = macd_line_values[-1]
    signal_val = signal_ema[-1]

    return MACDResult(
        macd_line=macd_val,
        signal_line=signal_val,
        histogram=macd_val - signal_val,
    )


# ─── Bollinger Bands ─────────────────────────────────────────────────────────

def bollinger_bands(
    closes: list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> Optional[BollingerBands]:
    """
    Bollinger Bands: SMA ± k * standard deviation.

    Also computes bandwidth (volatility measure) and %B (position within bands).

    Returns most recent values, or None if insufficient data.
    """
    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)

    upper = middle + num_std * std
    lower = middle - num_std * std

    band_width = upper - lower
    bandwidth = band_width / middle if middle > 0 else 0.0

    price = closes[-1]
    percent_b = (price - lower) / band_width if band_width > 0 else 0.5

    return BollingerBands(
        upper=upper,
        middle=middle,
        lower=lower,
        bandwidth=bandwidth,
        percent_b=percent_b,
    )


def bollinger_bandwidth_percentile(
    closes: list[float],
    period: int = 20,
    num_std: float = 2.0,
    lookback: int = 100,
) -> Optional[float]:
    """
    Compute the percentile rank of current Bollinger bandwidth over a lookback window.

    Low percentile (< 0.20) = squeeze / compression.
    High percentile (> 0.80) = expanded volatility.

    Returns 0.0–1.0, or None if insufficient data.
    """
    needed = period + lookback
    if len(closes) < needed:
        return None

    bandwidths: list[float] = []
    for i in range(lookback):
        end = len(closes) - lookback + i + 1
        start = end - period
        window = closes[start:end]
        mid = sum(window) / period
        if mid <= 0:
            continue
        std = math.sqrt(sum((x - mid) ** 2 for x in window) / period)
        bw = (2 * num_std * std) / mid
        bandwidths.append(bw)

    if not bandwidths:
        return None

    current = bandwidths[-1]
    rank = sum(1 for bw in bandwidths if bw <= current)
    return rank / len(bandwidths)


# ─── ATR (Average True Range) ────────────────────────────────────────────────

def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> Optional[float]:
    """
    Average True Range — Wilder's smoothing.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR is the Wilder-smoothed average of TR over `period`.

    Returns the most recent ATR value, or None if insufficient data.
    Requires at least period + 1 data points.
    """
    n = len(closes)
    if n < period + 1 or len(highs) != n or len(lows) != n:
        return None

    # True Range series (starts at index 1)
    trs: list[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    if len(trs) < period:
        return None

    # Seed: SMA of first `period` TRs
    atr_val = sum(trs[:period]) / period

    # Wilder smoothing for the rest
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period

    return atr_val


def atr_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[Optional[float]]:
    """
    Full ATR series using Wilder's smoothing.

    Returns list same length as input. First `period` entries are None.
    """
    n = len(closes)
    if n < period + 1 or len(highs) != n or len(lows) != n:
        return [None] * n

    # True Range series
    trs: list[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    result: list[Optional[float]] = [None] * period

    # Seed
    atr_val = sum(trs[:period]) / period
    result.append(atr_val)

    # Wilder smoothing
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
        result.append(atr_val)

    return result


def atr_percentile(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
    lookback: int = 100,
) -> Optional[float]:
    """
    Percentile rank of current ATR over a lookback window.

    Useful for classifying volatility regime.
    Returns 0.0–1.0, or None if insufficient data.
    """
    series = atr_series(highs, lows, closes, period)
    valid = [v for v in series if v is not None]

    if len(valid) < lookback:
        return None

    window = valid[-lookback:]
    current = window[-1]
    rank = sum(1 for v in window if v <= current)
    return rank / len(window)


# ─── VWAP ─────────────────────────────────────────────────────────────────────

def vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    period: Optional[int] = None,
) -> Optional[float]:
    """
    Volume-Weighted Average Price.

    Uses typical price: (high + low + close) / 3, weighted by volume.

    If period is specified, uses the last `period` candles.
    Otherwise uses all available data.

    Returns VWAP value, or None if insufficient data or zero total volume.
    """
    n = len(closes)
    if n == 0 or len(highs) != n or len(lows) != n or len(volumes) != n:
        return None

    if period is not None:
        if n < period:
            return None
        highs = highs[-period:]
        lows = lows[-period:]
        closes = closes[-period:]
        volumes = volumes[-period:]

    cumulative_tpv = 0.0
    cumulative_vol = 0.0

    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical_price = (h + l + c) / 3.0
        cumulative_tpv += typical_price * v
        cumulative_vol += v

    if cumulative_vol <= 0:
        return None

    return cumulative_tpv / cumulative_vol


# ─── RSI Divergence Detection ────────────────────────────────────────────────

def detect_rsi_divergence(
    closes: list[float],
    rsi_values: list[Optional[float]],
    lookback: int = 30,
    min_swing: float = 0.02,
) -> RSIDivergence:
    """
    Detect bullish and bearish RSI divergences.

    Bullish divergence: Price makes a lower low, but RSI makes a higher low.
    Bearish divergence: Price makes a higher high, but RSI makes a lower high.

    Args:
        closes: Price series.
        rsi_values: RSI series (same length as closes, may contain None).
        lookback: Number of bars to scan for swing points.
        min_swing: Minimum price swing as fraction to qualify as a swing point.

    Returns:
        RSIDivergence with bullish/bearish flags and strength.
    """
    neutral = RSIDivergence(bullish=False, bearish=False, strength=0.0)

    n = len(closes)
    if n < lookback or len(rsi_values) != n:
        return neutral

    # Work with the most recent `lookback` bars
    window_closes = closes[-lookback:]
    window_rsi = rsi_values[-lookback:]

    # Find swing lows and highs (simple: local min/max over 5-bar window)
    swing_lows: list[tuple[int, float, float]] = []   # (index, price, rsi)
    swing_highs: list[tuple[int, float, float]] = []  # (index, price, rsi)

    for i in range(2, len(window_closes) - 2):
        if window_rsi[i] is None:
            continue

        price = window_closes[i]
        rsi_val = window_rsi[i]

        # Swing low: lower than 2 bars on each side
        if (price <= window_closes[i - 1] and price <= window_closes[i - 2]
                and price <= window_closes[i + 1] and price <= window_closes[i + 2]):
            swing_lows.append((i, price, rsi_val))

        # Swing high: higher than 2 bars on each side
        if (price >= window_closes[i - 1] and price >= window_closes[i - 2]
                and price >= window_closes[i + 1] and price >= window_closes[i + 2]):
            swing_highs.append((i, price, rsi_val))

    bullish = False
    bearish = False
    strength = 0.0

    # Check for bullish divergence: compare last two swing lows
    if len(swing_lows) >= 2:
        prev_low = swing_lows[-2]
        curr_low = swing_lows[-1]
        price_change = (curr_low[1] - prev_low[1]) / prev_low[1] if prev_low[1] > 0 else 0

        if price_change < -min_swing and curr_low[2] > prev_low[2]:
            bullish = True
            # Strength: how much RSI diverged relative to price
            rsi_diff = curr_low[2] - prev_low[2]
            strength = max(strength, min(rsi_diff / 20.0, 1.0))

    # Check for bearish divergence: compare last two swing highs
    if len(swing_highs) >= 2:
        prev_high = swing_highs[-2]
        curr_high = swing_highs[-1]
        price_change = (curr_high[1] - prev_high[1]) / prev_high[1] if prev_high[1] > 0 else 0

        if price_change > min_swing and curr_high[2] < prev_high[2]:
            bearish = True
            rsi_diff = prev_high[2] - curr_high[2]
            strength = max(strength, min(rsi_diff / 20.0, 1.0))

    return RSIDivergence(bullish=bullish, bearish=bearish, strength=strength)


# ─── Snapshot builder ─────────────────────────────────────────────────────────

def compute_snapshot(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    timestamp: float,
    config: Optional[IndicatorConfig] = None,
) -> TechnicalSnapshot:
    """
    Compute all technical indicators and bundle into a TechnicalSnapshot.

    This is the single entry point the signal engine calls. Handles all
    indicator failures gracefully — any indicator that lacks sufficient
    data returns None rather than raising.

    Args:
        highs: High prices (oldest first).
        lows: Low prices (oldest first).
        closes: Close prices (oldest first).
        volumes: Volume per candle (oldest first).
        timestamp: Current timestamp.
        config: Indicator parameters (uses defaults if None).
    """
    cfg = config or IndicatorConfig()
    price = closes[-1] if closes else 0.0

    # RSI
    rsi_val = rsi(closes, cfg.rsi_period)

    # MACD
    macd_val = macd(closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    # Bollinger Bands
    bb_val = bollinger_bands(closes, cfg.bollinger_period, cfg.bollinger_std)
    squeeze = bb_val.squeeze if bb_val else False

    # ATR
    atr_val = atr(highs, lows, closes, cfg.atr_period)

    # ATR percentile for volatility classification
    vol_pct = atr_percentile(highs, lows, closes, cfg.atr_period, lookback=100)

    # VWAP
    vwap_val = vwap(highs, lows, closes, volumes, cfg.vwap_period)

    # RSI divergence
    rsi_vals = rsi_series(closes, cfg.rsi_period)
    divergence = detect_rsi_divergence(closes, rsi_vals)

    return TechnicalSnapshot(
        price=price,
        timestamp=timestamp,
        rsi=rsi_val,
        macd=macd_val,
        bollinger=bb_val,
        atr=atr_val,
        vwap=vwap_val,
        rsi_divergence=divergence,
        bollinger_squeeze=squeeze,
        volatility_percentile=vol_pct,
    )
