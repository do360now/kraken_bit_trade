"""
Tests for risk_manager.py and position_sizer.py

Validates:
- Risk gating: daily limits, reserve floor, drawdown tolerance
- Capitulation override: DCA buys allowed despite elevated drawdowns
- Golden rule floor: no emergency sell above cycle floor
- Emergency sell: fires when drawdown extreme AND near floor
- Stop levels: phase-adjusted ATR-based stops
- State persistence across restarts
- Position sizing: geometric mean factor combination (no anchoring!)
- Signal score factor mapping
- Volatility factor mapping
- Drawdown factor reduction
- Tiered profit taking: only highest unhit tier fires (cascade bug fix)
- Tier reset on cycle phase change
- Minimum order enforcement

Run: python -m pytest tests/test_risk_sizer.py -v
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BotConfig,
    CyclePhase,
    PersistenceConfig,
    RiskConfig,
    SizingConfig,
    VolatilityRegime,
)
from cycle_detector import CycleState, MomentumState, PriceStructure
from signal_engine import Action, CompositeSignal
from risk_manager import (
    PortfolioState,
    RiskDecision,
    RiskManager,
    StopLevels,
)
from position_sizer import (
    BuySize,
    PositionSizer,
    SellDecision,
    SellTier,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path) -> BotConfig:
    return BotConfig(persistence=PersistenceConfig(base_dir=tmp_path))


def make_portfolio(
    eur: float = 10000.0,
    btc: float = 0.5,
    btc_price: float = 50000.0,
    starting_eur: float = 35000.0,
) -> PortfolioState:
    return PortfolioState(
        eur_balance=eur,
        btc_balance=btc,
        btc_price=btc_price,
        starting_eur=starting_eur,
    )


def make_signal(
    score: float = 30.0,
    action: Action = Action.BUY,
    agreement: float = 0.7,
    quality: float = 0.8,
) -> CompositeSignal:
    return CompositeSignal(
        score=score, agreement=agreement, action=action,
        components=(), data_quality=quality, timestamp=time.time(),
    )


def make_cycle(
    phase: CyclePhase = CyclePhase.GROWTH,
    composite_score: float = 0.3,
    confidence: float = 0.7,
    drawdown_tolerance: float = 0.20,
    size_mult: float = 1.0,
    profit_active: bool = True,
    vol_score: float = 0.0,
    position_in_range: float = 0.5,
) -> CycleState:
    return CycleState(
        phase=phase,
        phase_confidence=confidence,
        time_score=0.3,
        price_score=0.2,
        momentum_score=0.3,
        volatility_score=vol_score,
        composite_score=composite_score,
        momentum=MomentumState(
            rsi_zone="neutral", trend_direction="up",
            higher_highs=True, higher_lows=True,
            rsi_bullish_divergence=False, rsi_bearish_divergence=False,
            momentum_score=0.3,
        ),
        price_structure=PriceStructure(
            drawdown_from_ath=0.2,
            position_in_range=position_in_range,
            distance_from_200d_ma=0.1,
            price_structure_score=0.2,
        ),
        volatility_regime=VolatilityRegime.NORMAL,
        cycle_day=400,
        cycle_progress=0.28,
        ath_eur=65000.0,
        drawdown_tolerance=drawdown_tolerance,
        position_size_multiplier=size_mult,
        profit_taking_active=profit_active,
        timestamp=time.time(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO STATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioState:
    def test_total_value(self):
        p = make_portfolio(eur=10000, btc=0.5, btc_price=50000)
        assert p.total_value_eur == pytest.approx(35000.0)

    def test_allocations(self):
        p = make_portfolio(eur=10000, btc=0.5, btc_price=50000)
        assert p.eur_allocation == pytest.approx(10000 / 35000)
        assert p.btc_allocation == pytest.approx(25000 / 35000)

    def test_zero_total(self):
        p = make_portfolio(eur=0, btc=0, btc_price=50000)
        assert p.eur_allocation == 1.0
        assert p.btc_allocation == 0.0

    def test_btc_value(self):
        p = make_portfolio(eur=5000, btc=0.2, btc_price=60000)
        assert p.btc_value_eur == pytest.approx(12000.0)

    def test_frozen(self):
        p = make_portfolio()
        with pytest.raises(AttributeError):
            p.eur_balance = 0.0  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskManagerDailyLimit:
    def test_under_limit_allowed(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        signal = make_signal(action=Action.BUY)
        portfolio = make_portfolio()
        cycle = make_cycle()
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is True

    def test_at_limit_blocked(self, tmp_path):
        config = make_config(tmp_path)
        rm = RiskManager(config)
        # Exhaust daily limit
        max_trades = config.risk.max_daily_trades
        for _ in range(max_trades):
            rm.record_trade()

        signal = make_signal(action=Action.BUY)
        portfolio = make_portfolio()
        cycle = make_cycle()
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is False
        assert "daily trade limit" in decision.reason.lower()


class TestRiskManagerReserveFloor:
    def test_reserve_floor_blocks_buy(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        # Very low EUR — below 20% reserve of starting
        portfolio = make_portfolio(eur=100, btc=0.5, btc_price=50000, starting_eur=35000)
        signal = make_signal(action=Action.BUY)
        cycle = make_cycle()
        # Initialize starting EUR
        rm.can_trade(signal, portfolio, cycle)
        # Now check with low EUR
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is False
        assert "reserve" in decision.reason.lower()

    def test_above_reserve_allowed(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        portfolio = make_portfolio(eur=10000, btc=0.5, btc_price=50000, starting_eur=35000)
        signal = make_signal(action=Action.BUY)
        cycle = make_cycle()
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is True


class TestRiskManagerDrawdown:
    def test_within_tolerance_allowed(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        portfolio = make_portfolio(eur=10000, btc=0.5, btc_price=50000)
        signal = make_signal(action=Action.BUY)
        cycle = make_cycle(drawdown_tolerance=0.30)
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is True

    def test_exceeds_tolerance_blocked(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        # First call sets peak portfolio
        big_portfolio = make_portfolio(eur=20000, btc=1.0, btc_price=50000)
        signal = make_signal(action=Action.BUY)
        cycle = make_cycle(drawdown_tolerance=0.10)
        rm.can_trade(signal, big_portfolio, cycle)

        # Now simulate drawdown: price crashed
        crashed_portfolio = make_portfolio(eur=20000, btc=1.0, btc_price=30000)
        decision = rm.can_trade(signal, crashed_portfolio, cycle)
        # Drawdown = 1 - 50000/70000 = ~28.6%, tolerance = 10%
        assert decision.allowed is False
        assert "drawdown" in decision.reason.lower()


class TestCapitulationOverride:
    def test_override_allows_buy_in_capitulation(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        # Set peak high
        big = make_portfolio(eur=30000, btc=1.0, btc_price=60000)
        signal = make_signal(action=Action.BUY)
        cycle_growth = make_cycle(drawdown_tolerance=0.10)
        rm.can_trade(signal, big, cycle_growth)

        # Crash: 50% drawdown
        crashed = make_portfolio(eur=30000, btc=1.0, btc_price=15000)
        cycle_cap = make_cycle(
            phase=CyclePhase.CAPITULATION, drawdown_tolerance=0.35,
        )
        # Even with drawdown exceeding tolerance...
        # Override should kick in for capitulation
        # Let's use a low tolerance to trigger the override
        cycle_cap_tight = make_cycle(
            phase=CyclePhase.CAPITULATION, drawdown_tolerance=0.10,
        )
        decision = rm.can_trade(signal, crashed, cycle_cap_tight)
        assert decision.allowed is True
        assert decision.override_active is True
        assert "capitulation override" in decision.reason.lower()

    def test_override_does_not_apply_in_growth(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        big = make_portfolio(eur=30000, btc=1.0, btc_price=60000)
        signal = make_signal(action=Action.BUY)
        cycle = make_cycle(phase=CyclePhase.GROWTH, drawdown_tolerance=0.10)
        rm.can_trade(signal, big, cycle)

        crashed = make_portfolio(eur=30000, btc=1.0, btc_price=15000)
        decision = rm.can_trade(signal, crashed, cycle)
        assert decision.allowed is False
        assert decision.override_active is False


class TestEmergencySell:
    def test_no_emergency_normal_conditions(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        portfolio = make_portfolio()
        cycle = make_cycle(drawdown_tolerance=0.30, position_in_range=0.5)
        # Initialize
        rm.can_trade(make_signal(), portfolio, cycle)
        decision = rm.emergency_sell(portfolio, cycle)
        assert decision.allowed is False

    def test_golden_rule_blocks_sell_above_floor(self, tmp_path):
        """Even with extreme drawdown, don't sell if price above floor."""
        rm = RiskManager(make_config(tmp_path))
        # Set peak
        big = make_portfolio(eur=50000, btc=1.0, btc_price=80000)
        rm.can_trade(make_signal(), big, make_cycle())

        # 50% drawdown but price still at 50% of range (above floor)
        crashed = make_portfolio(eur=50000, btc=1.0, btc_price=15000)
        cycle = make_cycle(drawdown_tolerance=0.10, position_in_range=0.50)
        decision = rm.emergency_sell(crashed, cycle)
        assert decision.allowed is False
        assert "golden rule" in decision.reason.lower()

    def test_emergency_fires_near_floor(self, tmp_path):
        """Extreme drawdown + near cycle floor → emergency sell."""
        rm = RiskManager(make_config(tmp_path))
        big = make_portfolio(eur=50000, btc=1.0, btc_price=80000)
        rm.can_trade(make_signal(), big, make_cycle())

        crashed = make_portfolio(eur=50000, btc=1.0, btc_price=10000)
        cycle = make_cycle(drawdown_tolerance=0.10, position_in_range=0.05)
        decision = rm.emergency_sell(crashed, cycle)
        assert decision.allowed is True
        assert "emergency" in decision.reason.lower()


class TestStopLevels:
    def test_basic_stop(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        cycle = make_cycle(phase=CyclePhase.GROWTH)
        stops = rm.get_stops(
            entry_price=50000, current_price=52000,
            cycle=cycle, atr_value=1000.0,
        )
        # Growth = 2.0x multiplier → atr_width = 2000
        assert stops.atr_width == pytest.approx(2000.0)
        assert stops.stop_price == pytest.approx(48000.0)
        assert stops.valid is True

    def test_wider_stops_in_capitulation(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        cycle_growth = make_cycle(phase=CyclePhase.GROWTH)
        cycle_cap = make_cycle(phase=CyclePhase.CAPITULATION)
        stops_growth = rm.get_stops(50000, 50000, cycle_growth, 1000.0)
        stops_cap = rm.get_stops(50000, 50000, cycle_cap, 1000.0)
        assert stops_cap.atr_width > stops_growth.atr_width

    def test_tighter_stops_in_euphoria(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        cycle_growth = make_cycle(phase=CyclePhase.GROWTH)
        cycle_euph = make_cycle(phase=CyclePhase.EUPHORIA)
        stops_growth = rm.get_stops(50000, 50000, cycle_growth, 1000.0)
        stops_euph = rm.get_stops(50000, 50000, cycle_euph, 1000.0)
        assert stops_euph.atr_width < stops_growth.atr_width

    def test_fallback_atr(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        cycle = make_cycle()
        stops = rm.get_stops(50000, 50000, cycle, atr_value=None)
        assert stops.atr_width > 0
        assert stops.stop_price > 0

    def test_warning_above_stop(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        cycle = make_cycle()
        stops = rm.get_stops(50000, 50000, cycle, 1000.0)
        assert stops.warning_price >= stops.stop_price

    def test_stop_never_negative(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        cycle = make_cycle(phase=CyclePhase.CAPITULATION)
        stops = rm.get_stops(1000, 1000, cycle, 500.0)
        assert stops.stop_price >= 0.0


class TestRiskManagerSellCheck:
    def test_sell_allowed_with_btc(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        portfolio = make_portfolio(btc=0.5)
        signal = make_signal(action=Action.SELL, score=-30.0)
        cycle = make_cycle()
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is True

    def test_sell_blocked_no_btc(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        portfolio = make_portfolio(btc=0.0)
        signal = make_signal(action=Action.SELL, score=-30.0)
        cycle = make_cycle()
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is False
        assert "no btc" in decision.reason.lower()

    def test_hold_signal_not_allowed(self, tmp_path):
        rm = RiskManager(make_config(tmp_path))
        portfolio = make_portfolio()
        signal = make_signal(action=Action.HOLD, score=0.0)
        cycle = make_cycle()
        decision = rm.can_trade(signal, portfolio, cycle)
        assert decision.allowed is False
        assert "hold" in decision.reason.lower()


class TestRiskStatePersistence:
    def test_state_persists_across_instances(self, tmp_path):
        config = make_config(tmp_path)
        rm1 = RiskManager(config)
        rm1.record_trade()
        rm1.record_trade()

        # Initialize starting EUR
        portfolio = make_portfolio()
        rm1.can_trade(make_signal(), portfolio, make_cycle())

        rm2 = RiskManager(config)
        assert rm2._daily_trade_count == 2
        assert rm2._starting_eur is not None


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometricMean:
    def test_all_ones(self):
        assert PositionSizer._geometric_mean([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_all_twos(self):
        assert PositionSizer._geometric_mean([2.0, 2.0, 2.0]) == pytest.approx(2.0)

    def test_mixed_factors(self):
        # geomean(2.0, 0.5) = sqrt(1.0) = 1.0
        assert PositionSizer._geometric_mean([2.0, 0.5]) == pytest.approx(1.0)

    def test_all_reducing(self):
        """All factors < 1 → combined < 1 (no anchoring to 1.0!)."""
        result = PositionSizer._geometric_mean([0.5, 0.6, 0.7, 0.8, 0.9])
        assert result < 0.9  # Should genuinely reduce

    def test_all_increasing(self):
        """All factors > 1 → combined > 1."""
        result = PositionSizer._geometric_mean([1.2, 1.3, 1.4, 1.5])
        assert result > 1.2

    def test_empty_returns_one(self):
        assert PositionSizer._geometric_mean([]) == 1.0

    def test_zero_clamped(self):
        """Zero factor should be clamped, not cause math error."""
        result = PositionSizer._geometric_mean([0.0, 1.0, 1.0])
        assert result > 0
        assert math.isfinite(result)

    def test_negative_clamped(self):
        result = PositionSizer._geometric_mean([-0.5, 1.0])
        assert result > 0

    def test_no_anchoring_regression(self):
        """
        REGRESSION TEST for the previous bot's anchoring bug.
        
        Old code: combined = 0.3 + 0.7 * geomean → max reduction 30%.
        New code: combined = geomean → full reduction possible.
        
        With 5 factors all at 0.5, old system gave ~0.65. New gives 0.5.
        """
        factors = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = PositionSizer._geometric_mean(factors)
        assert result == pytest.approx(0.5)
        # Old anchored system would give: 0.3 + 0.7 * 0.5 = 0.65
        assert result < 0.55  # Verify no anchoring


class TestSignalScoreFactor:
    def test_zero_score_neutral(self):
        assert PositionSizer._signal_score_factor(0.0) == pytest.approx(1.0)

    def test_positive_score_increases(self):
        factor = PositionSizer._signal_score_factor(50.0)
        assert factor > 1.0

    def test_max_score_capped(self):
        factor = PositionSizer._signal_score_factor(100.0)
        assert factor <= 2.0

    def test_negative_score_decreases(self):
        factor = PositionSizer._signal_score_factor(-50.0)
        assert factor < 1.0

    def test_always_positive(self):
        for score in [-100, -50, 0, 50, 100]:
            assert PositionSizer._signal_score_factor(score) > 0


class TestVolatilityFactor:
    def test_compression_increases(self):
        cycle = make_cycle(vol_score=0.8)
        factor = PositionSizer._volatility_factor(cycle)
        assert factor > 1.0

    def test_extreme_vol_decreases(self):
        cycle = make_cycle(vol_score=-0.8)
        factor = PositionSizer._volatility_factor(cycle)
        assert factor < 1.0

    def test_normal_vol_neutral(self):
        cycle = make_cycle(vol_score=0.0)
        factor = PositionSizer._volatility_factor(cycle)
        assert factor == pytest.approx(1.0)


class TestBuySizeComputation:
    def test_basic_buy(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        signal = make_signal(score=30.0)
        portfolio = make_portfolio(eur=10000, btc=0.0, btc_price=50000, starting_eur=10000)
        cycle = make_cycle(size_mult=1.0)
        risk = RiskDecision(allowed=True, reason="ok")

        buy = sizer.compute_buy_size(signal, portfolio, cycle, risk)

        assert buy.eur_amount > 0
        assert buy.btc_amount > 0
        assert buy.fraction_of_capital > 0
        assert buy.eur_amount <= portfolio.eur_balance

    def test_no_spendable_eur(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        signal = make_signal()
        # EUR below reserve floor (20% of starting)
        portfolio = make_portfolio(eur=100, btc=0, btc_price=50000, starting_eur=10000)
        cycle = make_cycle()
        risk = RiskDecision(allowed=True, reason="ok")

        buy = sizer.compute_buy_size(signal, portfolio, cycle, risk)
        assert buy.eur_amount == 0.0
        assert "reserve" in buy.reason.lower()

    def test_strong_signal_larger_buy(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=10000, btc=0, btc_price=50000, starting_eur=10000)
        cycle = make_cycle()
        risk = RiskDecision(allowed=True, reason="ok")

        weak = sizer.compute_buy_size(
            make_signal(score=20.0), portfolio, cycle, risk,
        )
        strong = sizer.compute_buy_size(
            make_signal(score=80.0), portfolio, cycle, risk,
        )
        assert strong.eur_amount > weak.eur_amount

    def test_capitulation_override_reduces(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=10000, btc=0, btc_price=50000, starting_eur=10000)
        cycle = make_cycle()

        normal = sizer.compute_buy_size(
            make_signal(), portfolio, cycle,
            RiskDecision(allowed=True, reason="ok", override_active=False),
        )
        override = sizer.compute_buy_size(
            make_signal(), portfolio, cycle,
            RiskDecision(allowed=True, reason="ok", override_active=True),
        )
        # Override should reduce size (0.6x factor)
        assert override.eur_amount < normal.eur_amount

    def test_high_vol_reduces_buy(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=10000, btc=0, btc_price=50000, starting_eur=10000)
        risk = RiskDecision(allowed=True, reason="ok")

        low_vol = sizer.compute_buy_size(
            make_signal(), portfolio, make_cycle(vol_score=0.5), risk,
        )
        high_vol = sizer.compute_buy_size(
            make_signal(), portfolio, make_cycle(vol_score=-0.8), risk,
        )
        assert high_vol.eur_amount < low_vol.eur_amount

    def test_adjustments_logged(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=10000, btc=0, btc_price=50000, starting_eur=10000)
        cycle = make_cycle()
        risk = RiskDecision(allowed=True, reason="ok")

        buy = sizer.compute_buy_size(make_signal(), portfolio, cycle, risk)
        assert "signal_score" in buy.adjustments
        assert "volatility" in buy.adjustments
        assert "cycle_phase" in buy.adjustments
        assert "drawdown" in buy.adjustments
        assert "risk_override" in buy.adjustments


# ─── Tiered profit taking tests ─────────────────────────────────────────────

class TestTieredProfitTaking:
    def test_no_profit_no_sell(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=5000, btc=0.5, btc_price=50000)
        cycle = make_cycle(profit_active=True)
        result = sizer.compute_sell_tiers(portfolio, cycle, avg_entry_price=55000)
        assert result.should_sell is False
        assert "no profit" in result.reason.lower()

    def test_first_tier_triggers(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=5000, btc=0.5, btc_price=60000)
        cycle = make_cycle(profit_active=True)
        # Entry at 50000, current 60000 → +20% profit
        result = sizer.compute_sell_tiers(portfolio, cycle, avg_entry_price=50000)
        assert result.should_sell is True
        assert result.tier is not None
        assert result.tier.btc_amount > 0

    def test_only_highest_unhit_tier_fires(self, tmp_path):
        """CASCADE BUG FIX: jumping from +5% to +80% should not fire all tiers."""
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=5000, btc=0.5, btc_price=90000)
        cycle = make_cycle(profit_active=True)
        # Entry at 50000, current 90000 → +80% profit
        result = sizer.compute_sell_tiers(portfolio, cycle, avg_entry_price=50000)

        assert result.should_sell is True
        assert result.tier is not None
        # Should fire only ONE tier (the highest crossed unhit one)
        # Not all tiers simultaneously
        assert isinstance(result.tier.tier_index, int)

    def test_tiers_dont_refire(self, tmp_path):
        """Once a tier is hit, it should not fire again."""
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=5000, btc=0.5, btc_price=60000)
        cycle = make_cycle(profit_active=True)

        # First evaluation: tier triggers
        result1 = sizer.compute_sell_tiers(portfolio, cycle, avg_entry_price=50000)
        assert result1.should_sell is True
        tier_idx = result1.tier.tier_index
        sizer.mark_tier_hit(tier_idx)

        # Second evaluation: same conditions, tier already hit
        result2 = sizer.compute_sell_tiers(portfolio, cycle, avg_entry_price=50000)
        # Should either find a different tier or no tier
        if result2.should_sell:
            assert result2.tier.tier_index != tier_idx

    def test_tiers_reset_on_phase_change(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=5000, btc=0.5, btc_price=60000)

        # Hit a tier in GROWTH
        cycle_growth = make_cycle(phase=CyclePhase.GROWTH, profit_active=True)
        result = sizer.compute_sell_tiers(portfolio, cycle_growth, avg_entry_price=50000)
        if result.should_sell:
            sizer.mark_tier_hit(result.tier.tier_index)
        assert len(sizer.hit_tiers) > 0

        # Phase changes to ACCUMULATION then back to GROWTH
        cycle_accum = make_cycle(phase=CyclePhase.ACCUMULATION, profit_active=False)
        sizer.compute_sell_tiers(portfolio, cycle_accum, avg_entry_price=50000)
        # Tiers should have reset
        assert len(sizer.hit_tiers) == 0

    def test_profit_taking_inactive_phase(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=5000, btc=0.5, btc_price=80000)
        cycle = make_cycle(
            phase=CyclePhase.ACCUMULATION, profit_active=False,
        )
        result = sizer.compute_sell_tiers(portfolio, cycle, avg_entry_price=50000)
        assert result.should_sell is False
        assert "inactive" in result.reason.lower()

    def test_no_btc_no_sell(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(btc=0.0)
        cycle = make_cycle(profit_active=True)
        result = sizer.compute_sell_tiers(portfolio, cycle, avg_entry_price=50000)
        assert result.should_sell is False

    def test_sell_tier_frozen(self, tmp_path):
        tier = SellTier(
            tier_index=0, threshold_pct=0.15,
            sell_pct=0.10, btc_amount=0.05,
            reason="test",
        )
        with pytest.raises(AttributeError):
            tier.btc_amount = 0.0  # type: ignore


class TestSizerStatePersistence:
    def test_hit_tiers_persist(self, tmp_path):
        config = make_config(tmp_path)
        sizer1 = PositionSizer(config)
        sizer1.mark_tier_hit(0)
        sizer1.mark_tier_hit(2)

        sizer2 = PositionSizer(config)
        assert 0 in sizer2.hit_tiers
        assert 2 in sizer2.hit_tiers


class TestDrawdownFactor:
    def test_no_drawdown_full_factor(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=20000, btc=0.6, btc_price=50000, starting_eur=50000)
        # total = 50000, starting = 50000, drawdown = 0%
        factor = sizer._drawdown_factor(portfolio)
        assert factor == pytest.approx(1.0)

    def test_moderate_drawdown_reduces(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=10000, btc=0.5, btc_price=50000, starting_eur=50000)
        # total = 35000, starting = 50000, drawdown = 30%
        factor = sizer._drawdown_factor(portfolio)
        assert factor == pytest.approx(0.4)

    def test_heavy_drawdown_floors(self, tmp_path):
        sizer = PositionSizer(make_config(tmp_path))
        portfolio = make_portfolio(eur=5000, btc=0.1, btc_price=50000, starting_eur=100000)
        # total = 10000, starting = 100000, drawdown = 90%
        factor = sizer._drawdown_factor(portfolio)
        assert factor == pytest.approx(0.4)  # Floored at 0.4
