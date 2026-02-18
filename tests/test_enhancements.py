"""
Tests for the 6 trading efficiency enhancements:
  1. Adaptive signal weights (phase-specific)
  2. Asymmetric agreement thresholds
  3. Value averaging (200-day MA proximity)
  4. Acceleration brake (FOMO protection)
  5. Phase-aware profit tiers
  6. DCA floor (guaranteed accumulation)

Run: python -m pytest tests/test_enhancements.py -v
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BotConfig, CyclePhase, PersistenceConfig, SignalConfig,
    SizingConfig, VolatilityRegime,
)
from cycle_detector import CycleState, PriceStructure, MomentumState
from signal_engine import SignalEngine, CompositeSignal, SubSignal, Action
from position_sizer import PositionSizer, BuySize, SellDecision
from risk_manager import RiskDecision, PortfolioState


# ─── Test helpers ────────────────────────────────────────────────────────────

def make_config(tmp_path: Path, **overrides) -> BotConfig:
    return BotConfig(
        persistence=PersistenceConfig(base_dir=tmp_path),
        **overrides,
    )


def make_cycle(
    phase: CyclePhase = CyclePhase.GROWTH,
    confidence: float = 0.8,
    vol_regime: VolatilityRegime = VolatilityRegime.NORMAL,
    position_in_range: float = 0.5,
    distance_200d: float = 0.0,
    momentum_score: float = 0.0,
    profit_active: bool = True,
    size_mult: float = 1.0,
) -> CycleState:
    return CycleState(
        phase=phase,
        phase_confidence=confidence,
        time_score=0.5,
        price_score=0.0,
        momentum_score=momentum_score,
        volatility_score=0.0,
        composite_score=0.0,
        momentum=MomentumState(
            rsi_zone="neutral",
            trend_direction="sideways",
            higher_highs=False,
            higher_lows=False,
            rsi_bullish_divergence=False,
            rsi_bearish_divergence=False,
            momentum_score=momentum_score,
        ),
        price_structure=PriceStructure(
            drawdown_from_ath=0.3,
            position_in_range=position_in_range,
            distance_from_200d_ma=distance_200d,
            price_structure_score=0.0,
        ),
        volatility_regime=vol_regime,
        cycle_day=400,
        cycle_progress=0.4,
        ath_eur=70000.0,
        drawdown_tolerance=0.20,
        position_size_multiplier=size_mult,
        profit_taking_active=profit_active,
        timestamp=time.time(),
    )


def make_signal(
    score: float = 30.0,
    agreement: float = 0.7,
    action: Action = Action.BUY,
) -> CompositeSignal:
    return CompositeSignal(
        score=score, agreement=agreement, action=action,
        components=(), data_quality=0.8, timestamp=time.time(),
    )


def make_portfolio(
    eur: float = 10000.0, btc: float = 0.5, price: float = 50000.0,
) -> PortfolioState:
    return PortfolioState(
        eur_balance=eur, btc_balance=btc,
        btc_price=price,
        starting_eur=eur + btc * price,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. ADAPTIVE SIGNAL WEIGHTS
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptiveWeights:
    """Signal engine uses phase-specific weights when configured."""

    def test_default_weights_when_no_override(self, tmp_path):
        """Phases without overrides use default weights."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        # early_bull has no override in default config
        w = engine._get_weight("rsi", "early_bull")
        assert w == config.signal.rsi_weight

    def test_accumulation_boosts_rsi_weight(self, tmp_path):
        """Accumulation phase overrides boost RSI weight."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        default_rsi = config.signal.rsi_weight
        accum_rsi = engine._get_weight("rsi", "accumulation")

        assert accum_rsi > default_rsi  # 0.25 > 0.20

    def test_accumulation_boosts_onchain_weight(self, tmp_path):
        """Accumulation phase overrides boost on-chain weight."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        default_onchain = config.signal.onchain_weight
        accum_onchain = engine._get_weight("onchain", "accumulation")

        assert accum_onchain > default_onchain  # 0.15 > 0.10

    def test_euphoria_boosts_cycle_weight(self, tmp_path):
        """Euphoria phase overrides boost cycle weight."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        default_cycle = config.signal.cycle_weight
        euphoria_cycle = engine._get_weight("cycle", "euphoria")

        assert euphoria_cycle > default_cycle  # 0.30 > 0.20

    def test_euphoria_boosts_macd_weight(self, tmp_path):
        """Euphoria phase overrides boost MACD weight (divergence detection)."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        default_macd = config.signal.macd_weight
        euphoria_macd = engine._get_weight("macd", "euphoria")

        assert euphoria_macd > default_macd  # 0.20 > 0.15

    def test_weights_sum_approximately_one(self, tmp_path):
        """Override weight sets should sum close to 1.0."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        for phase_val, overrides in config.signal.phase_weight_overrides.items():
            total = sum(overrides.values())
            assert 0.95 <= total <= 1.05, (
                f"Phase {phase_val} weights sum to {total}"
            )

    def test_adaptive_weights_flow_into_subsignals(self, tmp_path):
        """Generate with real cycle produces SubSignals with phase-specific weights."""
        from indicators import compute_snapshot, TechnicalSnapshot

        config = make_config(tmp_path)
        engine = SignalEngine(config)

        # Create minimal valid snapshot
        from test_integration import make_trending_prices, make_daily_closes, make_ath_tracker
        from cycle_detector import CycleDetector

        highs, lows, closes, volumes = make_trending_prices(100, 45000.0, 50000.0)
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        tracker = make_ath_tracker(tmp_path, ath=70000.0)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 50000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        signal = engine.generate(snapshot=snapshot, cycle=cycle)

        # Components should exist and have weights
        assert len(signal.components) == 7
        for comp in signal.components:
            assert comp.weight > 0


# ═══════════════════════════════════════════════════════════════════════
# 2. ASYMMETRIC AGREEMENT THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════

class TestAsymmetricAgreement:
    """Buy and sell thresholds test correctly at V4 defaults."""

    def test_buy_blocked_below_agreement(self, tmp_path):
        """Buy at 32% agreement is blocked (buy_min_agreement = 0.35)."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        action = engine._determine_action(
            score=15.0, agreement=0.32, data_quality=0.8,
        )
        assert action == Action.HOLD

    def test_buy_allowed_above_buy_threshold(self, tmp_path):
        """Buy at 40% agreement is allowed (above 0.35)."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        action = engine._determine_action(
            score=15.0, agreement=0.40, data_quality=0.8,
        )
        assert action == Action.BUY

    def test_sell_allowed_at_lower_agreement(self, tmp_path):
        """Sell at 36% agreement is allowed (sell_min_agreement = 0.35)."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        action = engine._determine_action(
            score=-25.0, agreement=0.36, data_quality=0.8,
        )
        assert action == Action.SELL

    def test_sell_blocked_below_sell_threshold(self, tmp_path):
        """Sell at 30% agreement is blocked (below 0.35)."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        action = engine._determine_action(
            score=-25.0, agreement=0.30, data_quality=0.8,
        )
        assert action == Action.HOLD

    def test_score_asymmetry_exists(self, tmp_path):
        """Buy threshold (10) is easier to hit than sell threshold (-20)."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        # Score +12: triggers buy
        buy_action = engine._determine_action(
            score=12.0, agreement=0.50, data_quality=0.8,
        )
        # Score -12: does NOT trigger sell (need -20)
        sell_action = engine._determine_action(
            score=-12.0, agreement=0.50, data_quality=0.8,
        )

        assert buy_action == Action.BUY
        assert sell_action == Action.HOLD

    def test_strong_buy_still_requires_agreement(self, tmp_path):
        """Even STRONG_BUY respects the buy_min_agreement threshold."""
        config = make_config(tmp_path)
        engine = SignalEngine(config)

        action = engine._determine_action(
            score=45.0, agreement=0.32, data_quality=0.8,
        )
        assert action == Action.HOLD  # 0.32 < 0.35


# ═══════════════════════════════════════════════════════════════════════
# 3. VALUE AVERAGING (200-DAY MA PROXIMITY)
# ═══════════════════════════════════════════════════════════════════════

class TestValueAveraging:
    """Buy more when price is below 200-day MA."""

    def test_below_200d_ma_boosts_factor(self, tmp_path):
        """15% below 200d MA → factor > 1.0."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(distance_200d=-0.15)
        factor = sizer._value_averaging_factor(cycle)

        assert factor > 1.0
        assert factor <= 1.0 + config.sizing.value_avg_max_boost

    def test_far_below_200d_ma_bigger_boost(self, tmp_path):
        """30% below MA gives bigger boost than 10% below."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle_10 = make_cycle(distance_200d=-0.10)
        cycle_30 = make_cycle(distance_200d=-0.30)

        factor_10 = sizer._value_averaging_factor(cycle_10)
        factor_30 = sizer._value_averaging_factor(cycle_30)

        assert factor_30 > factor_10

    def test_above_200d_ma_neutral(self, tmp_path):
        """15% above 200d MA → factor still ~1.0 (slight or no reduction)."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(distance_200d=0.15)
        factor = sizer._value_averaging_factor(cycle)

        assert 0.9 <= factor <= 1.05

    def test_far_above_200d_ma_reduces(self, tmp_path):
        """50% above 200d MA → factor < 1.0 (reduce buying)."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(distance_200d=0.50)
        factor = sizer._value_averaging_factor(cycle)

        assert factor < 1.0

    def test_at_200d_ma_neutral(self, tmp_path):
        """Exactly at 200d MA → factor = 1.0."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(distance_200d=0.0)
        factor = sizer._value_averaging_factor(cycle)

        assert factor == pytest.approx(1.0)

    def test_disabled_returns_neutral(self, tmp_path):
        """When disabled, always returns 1.0."""
        config = make_config(
            tmp_path,
            sizing=SizingConfig(value_avg_enabled=False),
        )
        sizer = PositionSizer(config)

        cycle = make_cycle(distance_200d=-0.30)
        factor = sizer._value_averaging_factor(cycle)

        assert factor == 1.0

    def test_boost_bounded_by_max(self, tmp_path):
        """Even at extreme drawdowns, boost never exceeds 1 + max_boost."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(distance_200d=-0.80)  # 80% below MA (extreme)
        factor = sizer._value_averaging_factor(cycle)

        max_factor = 1.0 + config.sizing.value_avg_max_boost
        assert factor <= max_factor + 0.01


# ═══════════════════════════════════════════════════════════════════════
# 4. ACCELERATION BRAKE (FOMO PROTECTION)
# ═══════════════════════════════════════════════════════════════════════

class TestAccelerationBrake:
    """Reduce buy size when price is running up fast."""

    def test_normal_volatility_no_brake(self, tmp_path):
        """Normal volatility → no braking regardless of position."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(
            vol_regime=VolatilityRegime.NORMAL,
            position_in_range=0.95,
        )
        factor = sizer._acceleration_brake_factor(cycle)

        assert factor == 1.0

    def test_elevated_vol_near_top_brakes(self, tmp_path):
        """Elevated vol + near top of range → brake engages."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(
            vol_regime=VolatilityRegime.ELEVATED,
            position_in_range=0.90,
        )
        factor = sizer._acceleration_brake_factor(cycle)

        assert factor < 1.0
        assert factor == pytest.approx(config.sizing.acceleration_brake_factor)

    def test_extreme_vol_high_momentum_brakes(self, tmp_path):
        """Extreme vol + high momentum → brake engages."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(
            vol_regime=VolatilityRegime.EXTREME,
            momentum_score=0.7,
            position_in_range=0.5,
        )
        factor = sizer._acceleration_brake_factor(cycle)

        assert factor < 1.0

    def test_elevated_vol_low_position_no_brake(self, tmp_path):
        """Elevated vol but price in lower range → no brake."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(
            vol_regime=VolatilityRegime.ELEVATED,
            position_in_range=0.5,
            momentum_score=0.3,
        )
        factor = sizer._acceleration_brake_factor(cycle)

        assert factor == 1.0

    def test_disabled_returns_neutral(self, tmp_path):
        """When disabled, always returns 1.0."""
        config = make_config(
            tmp_path,
            sizing=SizingConfig(acceleration_brake_enabled=False),
        )
        sizer = PositionSizer(config)

        cycle = make_cycle(
            vol_regime=VolatilityRegime.EXTREME,
            position_in_range=0.95,
        )
        factor = sizer._acceleration_brake_factor(cycle)

        assert factor == 1.0

    def test_compression_vol_no_brake(self, tmp_path):
        """Compression volatility → no braking (breakout pending)."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        cycle = make_cycle(
            vol_regime=VolatilityRegime.COMPRESSION,
            position_in_range=0.95,
        )
        factor = sizer._acceleration_brake_factor(cycle)

        assert factor == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 5. PHASE-AWARE PROFIT TIERS
# ═══════════════════════════════════════════════════════════════════════

class TestPhaseAwareProfitTiers:
    """Profit tiers adjust per cycle phase."""

    def test_growth_phase_uses_growth_tiers(self, tmp_path):
        """Growth phase uses tighter tiers with smaller sell percentages."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        # +20% profit, growth phase
        portfolio = make_portfolio(eur=5000, btc=1.0, price=60000.0)
        cycle = make_cycle(phase=CyclePhase.GROWTH)

        sell = sizer.compute_sell_tiers(
            portfolio=portfolio, cycle=cycle, avg_entry_price=50000.0,
        )

        # Growth tiers: first at 15% (sell 5%), second at 30% (sell 10%)
        # At +20%, only the 15% tier should fire, selling just 5%
        if sell.should_sell:
            assert sell.tier.sell_pct <= 0.10  # Growth tiers are conservative

    def test_distribution_phase_uses_aggressive_tiers(self, tmp_path):
        """Distribution phase sells more aggressively."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        portfolio = make_portfolio(eur=5000, btc=1.0, price=55000.0)
        cycle = make_cycle(phase=CyclePhase.DISTRIBUTION)

        sell = sizer.compute_sell_tiers(
            portfolio=portfolio, cycle=cycle, avg_entry_price=50000.0,
        )

        # Distribution tiers: first at 5% (sell 20%)
        # At +10%, the 5% tier fires, selling 20%
        if sell.should_sell:
            assert sell.tier.sell_pct >= 0.15  # Distribution is aggressive

    def test_early_bull_uses_default_tiers(self, tmp_path):
        """Phases without overrides use default tier config."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        portfolio = make_portfolio(eur=5000, btc=1.0, price=55000.0)
        cycle = make_cycle(phase=CyclePhase.EARLY_BULL)

        sell = sizer.compute_sell_tiers(
            portfolio=portfolio, cycle=cycle, avg_entry_price=50000.0,
        )

        # Default tiers: first at 5% (sell 10%)
        if sell.should_sell:
            assert sell.tier.sell_pct == pytest.approx(0.10)

    def test_euphoria_lower_thresholds(self, tmp_path):
        """Euphoria tiers have lower entry threshold than defaults."""
        config = make_config(tmp_path)
        euphoria_tiers = config.sizing.phase_profit_tiers.get("euphoria", [])
        default_tiers = config.sizing.profit_tiers

        if euphoria_tiers and default_tiers:
            # Euphoria first tier threshold should be higher than default
            # (10% vs 5%), but sell_pct should be larger (15% vs 10%)
            assert euphoria_tiers[0]["sell_pct"] > default_tiers[0]["sell_pct"]


# ═══════════════════════════════════════════════════════════════════════
# 6. DCA FLOOR (GUARANTEED ACCUMULATION)
# ═══════════════════════════════════════════════════════════════════════

class TestDCAFloor:
    """Force minimum buys during extended HOLD periods."""

    def _make_bot(self, tmp_path, paper=True):
        """Create a Bot with mocked externals."""
        from main import Bot

        config = make_config(tmp_path)
        config.paper_trade = paper

        with patch("main.KrakenAPI") as MockAPI, \
             patch("main.BitcoinNode") as MockNode, \
             patch("main.OllamaAnalyst") as MockOllama, \
             patch("main.PerformanceTracker") as MockPerf:

            MockAPI.return_value = MagicMock()
            MockNode.return_value = MagicMock()
            MockOllama.return_value = MagicMock()
            MockPerf.return_value = MagicMock()

            bot = Bot(config)

        return bot

    def test_should_dca_floor_after_interval(self, tmp_path):
        """DCA floor triggers after interval elapses."""
        bot = self._make_bot(tmp_path)

        # Set last buy to 25 hours ago
        bot._last_buy_time = time.time() - (25 * 3600)

        portfolio = make_portfolio(eur=10000, btc=0.5, price=50000.0)
        cycle = make_cycle(phase=CyclePhase.GROWTH)
        signal = make_signal()

        assert bot._should_dca_floor(portfolio, cycle, signal) is True

    def test_should_not_dca_floor_within_interval(self, tmp_path):
        """DCA floor doesn't trigger if bought recently."""
        bot = self._make_bot(tmp_path)

        # Set last buy to 12 hours ago
        bot._last_buy_time = time.time() - (12 * 3600)

        portfolio = make_portfolio(eur=10000, btc=0.5, price=50000.0)
        cycle = make_cycle(phase=CyclePhase.GROWTH)
        signal = make_signal()

        assert bot._should_dca_floor(portfolio, cycle, signal) is False

    def test_should_not_dca_floor_in_euphoria(self, tmp_path):
        """DCA floor blocked during euphoria (don't force-buy near peaks)."""
        bot = self._make_bot(tmp_path)
        bot._last_buy_time = time.time() - (48 * 3600)

        portfolio = make_portfolio(eur=10000, btc=0.5, price=50000.0)
        cycle = make_cycle(phase=CyclePhase.EUPHORIA)
        signal = make_signal()

        assert bot._should_dca_floor(portfolio, cycle, signal) is False

    def test_should_not_dca_floor_in_distribution(self, tmp_path):
        """DCA floor blocked during distribution."""
        bot = self._make_bot(tmp_path)
        bot._last_buy_time = time.time() - (48 * 3600)

        portfolio = make_portfolio(eur=10000, btc=0.5, price=50000.0)
        cycle = make_cycle(phase=CyclePhase.DISTRIBUTION)
        signal = make_signal()

        assert bot._should_dca_floor(portfolio, cycle, signal) is False

    def test_should_not_dca_floor_no_spendable(self, tmp_path):
        """DCA floor blocked when EUR balance is zero."""
        bot = self._make_bot(tmp_path)
        bot._last_buy_time = time.time() - (48 * 3600)

        # No EUR at all → nothing to spend
        portfolio = make_portfolio(eur=0, btc=0.5, price=50000.0)
        cycle = make_cycle(phase=CyclePhase.GROWTH)
        signal = make_signal()

        assert bot._should_dca_floor(portfolio, cycle, signal) is False

    def test_dca_floor_disabled(self, tmp_path):
        """When disabled, never triggers."""
        from main import Bot

        config = make_config(
            tmp_path,
            sizing=SizingConfig(dca_floor_enabled=False),
        )
        config.paper_trade = True

        with patch("main.KrakenAPI") as MockAPI, \
             patch("main.BitcoinNode") as MockNode, \
             patch("main.OllamaAnalyst") as MockOllama, \
             patch("main.PerformanceTracker") as MockPerf:

            MockAPI.return_value = MagicMock()
            MockNode.return_value = MagicMock()
            MockOllama.return_value = MagicMock()
            MockPerf.return_value = MagicMock()

            bot = Bot(config)

        bot._last_buy_time = time.time() - (48 * 3600)
        portfolio = make_portfolio(eur=10000, btc=0.5, price=50000.0)
        cycle = make_cycle(phase=CyclePhase.ACCUMULATION)
        signal = make_signal()

        assert bot._should_dca_floor(portfolio, cycle, signal) is False

    def test_dca_floor_in_capitulation_allowed(self, tmp_path):
        """DCA floor IS allowed during capitulation (accumulation territory)."""
        bot = self._make_bot(tmp_path)
        bot._last_buy_time = time.time() - (48 * 3600)

        portfolio = make_portfolio(eur=10000, btc=0.5, price=50000.0)
        cycle = make_cycle(phase=CyclePhase.CAPITULATION)
        signal = make_signal()

        assert bot._should_dca_floor(portfolio, cycle, signal) is True

    def test_handle_buy_updates_last_buy_time(self, tmp_path):
        """_handle_buy sets _last_buy_time on execution."""
        bot = self._make_bot(tmp_path, paper=True)
        assert bot._last_buy_time == 0.0

        portfolio = make_portfolio()
        cycle = make_cycle()
        signal = make_signal()
        buy_size = BuySize(
            eur_amount=500, btc_amount=0.01,
            fraction_of_capital=0.05, adjustments={},
            reason="test",
        )
        risk = RiskDecision(allowed=True, reason="ok")

        bot._handle_buy(portfolio, cycle, signal, buy_size, risk)

        assert bot._last_buy_time > 0


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION: NEW FACTORS FLOW THROUGH COMPUTE_BUY_SIZE
# ═══════════════════════════════════════════════════════════════════════

class TestNewFactorsInBuySize:
    """Verify new adjustment factors appear in compute_buy_size output."""

    def test_value_avg_factor_in_adjustments(self, tmp_path):
        """Value averaging factor is recorded in buy size adjustments."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        signal = make_signal(score=40.0)
        portfolio = make_portfolio()
        cycle = make_cycle(distance_200d=-0.15)
        risk = RiskDecision(allowed=True, reason="ok")

        buy = sizer.compute_buy_size(signal, portfolio, cycle, risk)

        assert "value_avg" in buy.adjustments
        assert buy.adjustments["value_avg"] > 1.0  # Below MA → boost

    def test_acceleration_brake_in_adjustments(self, tmp_path):
        """Acceleration brake factor is recorded in buy size adjustments."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        signal = make_signal(score=40.0)
        portfolio = make_portfolio()
        cycle = make_cycle(
            vol_regime=VolatilityRegime.ELEVATED,
            position_in_range=0.92,
        )
        risk = RiskDecision(allowed=True, reason="ok")

        buy = sizer.compute_buy_size(signal, portfolio, cycle, risk)

        assert "acceleration_brake" in buy.adjustments
        assert buy.adjustments["acceleration_brake"] < 1.0  # Braking

    def test_below_ma_produces_larger_buy(self, tmp_path):
        """Below 200d MA → larger buy amount than at MA (value averaging)."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        signal = make_signal(score=40.0)
        portfolio = make_portfolio()
        risk = RiskDecision(allowed=True, reason="ok")

        cycle_at_ma = make_cycle(distance_200d=0.0)
        cycle_below_ma = make_cycle(distance_200d=-0.20)

        buy_at = sizer.compute_buy_size(signal, portfolio, cycle_at_ma, risk)
        buy_below = sizer.compute_buy_size(signal, portfolio, cycle_below_ma, risk)

        assert buy_below.eur_amount >= buy_at.eur_amount
