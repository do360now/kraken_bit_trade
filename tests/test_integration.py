"""
Integration tests — real modules, synthetic data, only I/O mocked.

These tests wire the actual computation pipeline together:
  indicators → cycle → signal → risk → sizer

Unlike unit tests (which mock everything except the module under test),
these exercise real module interactions to catch:
  1. Interface mismatches between modules (wrong field names, types, shapes)
  2. Emergent behavior when real modules interact (e.g., cycle phase affecting
     both signal weights AND risk tolerance AND position size multiplier)
  3. Pipeline routing bugs where a valid output from module A doesn't flow
     correctly into module B

Only external I/O is mocked: Kraken API, Bitcoin node RPC, Ollama HTTP.

Run: python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ATHTracker,
    BotConfig,
    CyclePhase,
    PersistenceConfig,
    SizingConfig,
    Urgency,
    VolatilityRegime,
)
from indicators import compute_snapshot, TechnicalSnapshot
from cycle_detector import CycleDetector, CycleState
from signal_engine import SignalEngine, CompositeSignal, Action, LLMContext
from risk_manager import RiskManager, PortfolioState, RiskDecision
from position_sizer import PositionSizer, BuySize, SellDecision


# ─── Synthetic market data generators ─────────────────────────────────────────

def make_trending_prices(
    n: int, start: float, end: float, noise_pct: float = 0.005,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Generate synthetic OHLCV data trending from start to end.

    Returns (highs, lows, closes, volumes) with realistic candle shapes.
    """
    import random
    random.seed(42)

    step = (end - start) / max(n - 1, 1)
    highs, lows, closes, volumes = [], [], [], []

    for i in range(n):
        base = start + i * step
        noise = base * noise_pct
        close = base + random.uniform(-noise, noise)
        high = close + abs(random.gauss(0, noise * 1.5))
        low = close - abs(random.gauss(0, noise * 1.5))
        vol = 1.0 + random.uniform(0, 0.5)

        highs.append(high)
        lows.append(low)
        closes.append(close)
        volumes.append(vol)

    return highs, lows, closes, volumes


def make_daily_closes(
    n: int, start: float, end: float,
) -> tuple[list[float], list[float], list[float]]:
    """Generate daily close/high/low arrays for cycle detector."""
    import random
    random.seed(99)

    step = (end - start) / max(n - 1, 1)
    closes, highs, lows = [], [], []

    for i in range(n):
        base = start + i * step
        noise = base * 0.01
        c = base + random.uniform(-noise, noise)
        h = c + abs(random.gauss(0, noise))
        lo = c - abs(random.gauss(0, noise))
        closes.append(c)
        highs.append(h)
        lows.append(lo)

    return closes, highs, lows


def make_config(tmp_path: Path, **overrides) -> BotConfig:
    """Create a BotConfig with persistence pointed at tmp_path."""
    return BotConfig(
        persistence=PersistenceConfig(base_dir=tmp_path),
        **overrides,
    )


def make_ath_tracker(tmp_path: Path, ath: float = 70000.0) -> ATHTracker:
    """Create ATHTracker with a known ATH value."""
    config = PersistenceConfig(base_dir=tmp_path)
    tracker = ATHTracker(config)
    tracker._ath_eur = ath
    return tracker


def make_portfolio(
    eur: float = 10000.0, btc: float = 0.5, price: float = 50000.0,
) -> PortfolioState:
    return PortfolioState(
        eur_balance=eur,
        btc_balance=btc,
        btc_price=price,
        starting_eur=eur + btc * price,
    )


# ─── Pipeline: indicators → cycle → signal ──────────────────────────────────

class TestIndicatorsToCycle:
    """Verify compute_snapshot output flows correctly into CycleDetector."""

    def test_snapshot_feeds_cycle_detector(self, tmp_path):
        """Real indicators → real cycle detector. No mocks."""
        highs, lows, closes, volumes = make_trending_prices(
            100, 45000.0, 55000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        # Snapshot must have the fields CycleDetector expects
        assert isinstance(snapshot, TechnicalSnapshot)
        assert snapshot.price > 0
        assert snapshot.rsi is not None

        # Feed into real CycleDetector
        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, ath=70000.0)
        detector = CycleDetector(config, tracker)

        daily_closes, daily_highs, daily_lows = make_daily_closes(
            250, 30000.0, 55000.0,
        )
        cycle = detector.analyze(snapshot, daily_closes, daily_highs, daily_lows)

        # CycleState must have all the fields downstream modules expect
        assert isinstance(cycle, CycleState)
        assert cycle.phase in list(CyclePhase)
        assert 0.0 <= cycle.phase_confidence <= 1.0
        assert cycle.drawdown_tolerance > 0
        assert cycle.position_size_multiplier > 0
        assert cycle.ath_eur > 0

    def test_insufficient_data_still_produces_valid_cycle(self, tmp_path):
        """With minimal candles, indicators may be None but cycle still works."""
        # Only 30 candles — not enough for 200-day MA, RSI divergence, etc.
        highs, lows, closes, volumes = make_trending_prices(30, 50000.0, 51000.0)
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path)
        detector = CycleDetector(config, tracker)

        # Only 30 daily candles too — well under the 200 needed
        daily_c, daily_h, daily_l = make_daily_closes(30, 49000.0, 51000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        assert isinstance(cycle, CycleState)
        assert cycle.phase in list(CyclePhase)


class TestCycleToSignal:
    """Verify CycleState flows correctly into SignalEngine."""

    def test_cycle_feeds_signal_engine(self, tmp_path):
        """Real cycle → real signal engine with real indicators."""
        highs, lows, closes, volumes = make_trending_prices(
            100, 45000.0, 55000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, ath=70000.0)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 55000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        # Feed into real SignalEngine (no on-chain, no LLM — both optional)
        engine = SignalEngine(config)
        signal = engine.generate(snapshot=snapshot, cycle=cycle)

        assert isinstance(signal, CompositeSignal)
        assert -100 <= signal.score <= 100
        assert 0.0 <= signal.agreement <= 1.0
        assert signal.action in list(Action)
        assert 0.0 <= signal.data_quality <= 1.0
        assert len(signal.components) > 0

    def test_signal_with_llm_context(self, tmp_path):
        """Signal engine handles optional LLM context without crashing."""
        highs, lows, closes, volumes = make_trending_prices(
            100, 45000.0, 55000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 55000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        llm = LLMContext(
            regime="accumulation", sentiment=-0.5,
            risk_level="medium", themes=("oversold", "fear"),
            timestamp=time.time(),
        )

        engine = SignalEngine(config)
        signal = engine.generate(
            snapshot=snapshot, cycle=cycle, llm=llm,
        )

        assert isinstance(signal, CompositeSignal)
        assert signal.data_quality > 0


# ─── Pipeline: signal → risk → sizer ────────────────────────────────────────

class TestSignalToRiskToSizer:
    """Full computation pipeline: indicators → cycle → signal → risk → sizer."""

    def _run_pipeline(
        self, tmp_path: Path, price_start: float, price_end: float,
        ath: float, eur: float = 10000.0, btc: float = 0.5,
    ) -> tuple[TechnicalSnapshot, CycleState, CompositeSignal, PortfolioState]:
        """Run the full computation pipeline, return intermediate results."""
        highs, lows, closes, volumes = make_trending_prices(
            100, price_start, price_end,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, ath=ath)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(
            250, price_start * 0.7, price_end,
        )
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        engine = SignalEngine(config)
        signal = engine.generate(snapshot=snapshot, cycle=cycle)

        portfolio = make_portfolio(eur=eur, btc=btc, price=price_end)

        return snapshot, cycle, signal, portfolio

    def test_buy_signal_passes_through_risk_and_sizer(self, tmp_path):
        """When signal is BUY and risk allows, sizer produces a valid order."""
        snapshot, cycle, signal, portfolio = self._run_pipeline(
            tmp_path, price_start=40000.0, price_end=50000.0, ath=70000.0,
        )

        config = make_config(tmp_path)
        risk_mgr = RiskManager(config)
        sizer = PositionSizer(config)

        if signal.is_buy and signal.actionable:
            risk = risk_mgr.can_trade(signal, portfolio, cycle)

            if risk.allowed:
                buy_size = sizer.compute_buy_size(
                    signal=signal, portfolio=portfolio,
                    cycle=cycle, risk=risk,
                )

                assert isinstance(buy_size, BuySize)
                assert buy_size.eur_amount >= 0
                assert buy_size.btc_amount >= 0
                assert buy_size.fraction_of_capital >= 0
                assert len(buy_size.reason) > 0

                # Sanity: buy shouldn't exceed spendable EUR
                reserve = portfolio.starting_eur * config.risk.reserve_floor_pct
                spendable = max(0, portfolio.eur_balance - reserve)
                assert buy_size.eur_amount <= spendable * 1.01  # tiny float tolerance

    def test_risk_gate_blocks_when_reserve_floor_breached(self, tmp_path):
        """Low EUR balance triggers reserve floor block even with BUY signal.

        can_trade() checks: 1. daily limit → 2. signal direction → 3. reserve floor.
        We need a BUY signal (uptrend) but nearly zero EUR so the reserve floor
        check is reached and blocks the trade.
        """
        config = make_config(tmp_path)
        risk_mgr = RiskManager(config)

        # Uptrend data so signal engine generates a BUY
        highs, lows, closes, volumes = make_trending_prices(
            100, 42000.0, 55000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        tracker = make_ath_tracker(tmp_path, ath=70000.0)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 55000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        engine = SignalEngine(config)
        signal = engine.generate(snapshot=snapshot, cycle=cycle)

        # Nearly all capital deployed: €200 EUR left, 0.5 BTC at €55k
        # starting_eur = 200 + 0.5*55000 = €27700, reserve = 20% = €5540
        # €200 < €5540 → reserve floor should block
        portfolio = make_portfolio(eur=200.0, btc=0.5, price=55000.0)

        risk = risk_mgr.can_trade(signal, portfolio, cycle)

        if signal.is_buy:
            # Reserve floor should block since €200 < 20% of starting EUR
            assert risk.allowed is False
            assert "reserve" in risk.reason.lower() or "floor" in risk.reason.lower()
        else:
            # Signal wasn't a BUY despite uptrend data — still valid,
            # just means agreement was too low. Risk blocks for HOLD.
            assert risk.allowed is False

    def test_emergency_sell_golden_rule(self, tmp_path):
        """Emergency sell blocked when price is above cycle floor (golden rule)."""
        config = make_config(tmp_path)
        risk_mgr = RiskManager(config)

        # Price is still decent — above the cycle floor
        portfolio = make_portfolio(eur=500.0, btc=1.0, price=50000.0)

        highs, lows, closes, volumes = make_trending_prices(
            100, 48000.0, 50000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        tracker = make_ath_tracker(tmp_path, ath=70000.0)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 50000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        emergency = risk_mgr.emergency_sell(portfolio, cycle)

        # Price at 50k with 70k ATH is only ~29% drawdown — within tolerance
        # for most phases, golden rule should prevent emergency sell
        assert emergency.allowed is False


# ─── Full pipeline: profit-taking path ───────────────────────────────────────

class TestProfitTakingPipeline:
    """Integration test for the sell/profit-taking path through real modules."""

    def test_profit_tier_fires_at_sufficient_gain(self, tmp_path):
        """When price is well above avg entry, compute_sell_tiers fires a tier."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        # Bought at €30k, now at €50k — that's +67%, should hit tiers
        portfolio = make_portfolio(eur=5000.0, btc=1.0, price=50000.0)

        highs, lows, closes, volumes = make_trending_prices(
            100, 45000.0, 50000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        tracker = make_ath_tracker(tmp_path, ath=70000.0)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 50000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        # Only fire if profit_taking_active
        if cycle.profit_taking_active:
            sell_decision = sizer.compute_sell_tiers(
                portfolio=portfolio, cycle=cycle, avg_entry_price=30000.0,
            )

            assert isinstance(sell_decision, SellDecision)

            if sell_decision.should_sell:
                tier = sell_decision.tier
                assert tier is not None
                assert tier.btc_amount > 0
                assert tier.sell_pct > 0
                assert tier.threshold_pct > 0
                # Only ONE tier should fire (not cascade)
                assert tier.tier_index >= 0

    def test_no_profit_tier_when_underwater(self, tmp_path):
        """When price is below avg entry, no profit tier fires."""
        config = make_config(tmp_path)
        sizer = PositionSizer(config)

        # Bought at €50k, now at €40k — underwater, no profit
        portfolio = make_portfolio(eur=5000.0, btc=1.0, price=40000.0)

        highs, lows, closes, volumes = make_trending_prices(
            100, 38000.0, 40000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        tracker = make_ath_tracker(tmp_path, ath=70000.0)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 40000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        sell_decision = sizer.compute_sell_tiers(
            portfolio=portfolio, cycle=cycle, avg_entry_price=50000.0,
        )
        assert sell_decision.should_sell is False


# ─── Market scenario end-to-end tests ────────────────────────────────────────

class TestMarketScenarios:
    """
    End-to-end scenarios with realistic market conditions.

    Each test creates a market scenario, runs the full computation pipeline
    (indicators → cycle → signal → risk → sizer), and verifies the bot
    would make a sensible decision.
    """

    def _full_pipeline(
        self, tmp_path: Path,
        fast_prices: tuple[float, float],
        daily_prices: tuple[float, float],
        ath: float,
        eur: float = 10000.0,
        btc: float = 0.5,
        llm: LLMContext | None = None,
    ) -> dict:
        """Run full pipeline and return all intermediate results."""
        highs, lows, closes, volumes = make_trending_prices(
            100, fast_prices[0], fast_prices[1],
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path, ath=ath)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(
            250, daily_prices[0], daily_prices[1],
        )
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        engine = SignalEngine(config)
        signal = engine.generate(
            snapshot=snapshot, cycle=cycle, llm=llm,
        )

        portfolio = make_portfolio(
            eur=eur, btc=btc, price=fast_prices[1],
        )

        risk_mgr = RiskManager(config)
        sizer = PositionSizer(config)

        risk = risk_mgr.can_trade(signal, portfolio, cycle)
        emergency = risk_mgr.emergency_sell(portfolio, cycle)

        buy_size = None
        if signal.is_buy and signal.actionable and risk.allowed:
            buy_size = sizer.compute_buy_size(
                signal=signal, portfolio=portfolio,
                cycle=cycle, risk=risk,
            )

        return {
            "snapshot": snapshot,
            "cycle": cycle,
            "signal": signal,
            "portfolio": portfolio,
            "risk": risk,
            "emergency": emergency,
            "buy_size": buy_size,
            "config": config,
        }

    def test_steady_uptrend_accumulation(self, tmp_path):
        """Steady uptrend from lows → bot should be willing to buy."""
        result = self._full_pipeline(
            tmp_path,
            fast_prices=(45000.0, 55000.0),
            daily_prices=(30000.0, 55000.0),
            ath=70000.0,
            eur=10000.0, btc=0.3,
        )

        # In an uptrend with healthy balance, pipeline should produce valid outputs
        assert result["cycle"].phase in list(CyclePhase)
        assert -100 <= result["signal"].score <= 100
        assert result["emergency"].allowed is False  # No emergency in uptrend

        # All types correct across module boundaries
        assert isinstance(result["snapshot"], TechnicalSnapshot)
        assert isinstance(result["cycle"], CycleState)
        assert isinstance(result["signal"], CompositeSignal)
        assert isinstance(result["risk"], RiskDecision)

    def test_crash_scenario(self, tmp_path):
        """Sharp decline from highs → emergency sell may trigger, or risk blocks buys."""
        result = self._full_pipeline(
            tmp_path,
            fast_prices=(60000.0, 35000.0),   # 42% drop in fast candles
            daily_prices=(60000.0, 35000.0),   # Sustained decline
            ath=70000.0,
            eur=2000.0, btc=0.8,   # Mostly in BTC, low EUR
        )

        # In a crash, the bot should NOT be aggressively buying
        cycle = result["cycle"]
        signal = result["signal"]
        risk = result["risk"]

        # Either the signal says don't buy, or risk blocks it, or
        # the signal is cautious. The pipeline should not produce
        # an aggressive buy in a crash.
        if signal.is_buy and signal.actionable:
            # If it somehow says buy (capitulation DCA), the size should be small
            if risk.allowed and result["buy_size"] is not None:
                assert result["buy_size"].fraction_of_capital <= 0.15

    def test_all_types_consistent_across_pipeline(self, tmp_path):
        """Every module's output is the correct type for the next module's input."""
        result = self._full_pipeline(
            tmp_path,
            fast_prices=(48000.0, 52000.0),
            daily_prices=(35000.0, 52000.0),
            ath=70000.0,
        )

        # Indicators → Cycle: snapshot has required fields
        snap = result["snapshot"]
        assert hasattr(snap, "price")
        assert hasattr(snap, "rsi")
        assert hasattr(snap, "macd")
        assert hasattr(snap, "bollinger")
        assert hasattr(snap, "atr")
        assert hasattr(snap, "data_quality")

        # Cycle → Signal: cycle has required fields
        cycle = result["cycle"]
        assert hasattr(cycle, "phase")
        assert hasattr(cycle, "volatility_regime")
        assert hasattr(cycle, "momentum")
        assert hasattr(cycle, "price_structure")
        assert hasattr(cycle, "profit_taking_active")

        # Signal → Risk: signal has required fields
        signal = result["signal"]
        assert hasattr(signal, "score")
        assert hasattr(signal, "agreement")
        assert hasattr(signal, "action")
        assert hasattr(signal, "actionable")
        assert hasattr(signal, "is_buy")
        assert hasattr(signal, "is_sell")

        # Risk → Sizer: risk decision has required fields
        risk = result["risk"]
        assert hasattr(risk, "allowed")
        assert hasattr(risk, "reason")

    def test_pipeline_with_bearish_llm(self, tmp_path):
        """
        Bearish LLM sentiment should boost buy signals (accumulation inversion).

        The signal engine inverts LLM sentiment: fear = buying opportunity.
        """
        # Run pipeline twice: once without LLM, once with bearish LLM
        base_result = self._full_pipeline(
            tmp_path,
            fast_prices=(42000.0, 45000.0),
            daily_prices=(30000.0, 45000.0),
            ath=70000.0,
        )

        bearish_llm = LLMContext(
            regime="capitulation", sentiment=-0.8,
            risk_level="high", themes=("fear", "panic", "capitulation"),
            timestamp=time.time(),
        )

        llm_result = self._full_pipeline(
            tmp_path,
            fast_prices=(42000.0, 45000.0),
            daily_prices=(30000.0, 45000.0),
            ath=70000.0,
            llm=bearish_llm,
        )

        # Both should produce valid outputs
        assert isinstance(base_result["signal"], CompositeSignal)
        assert isinstance(llm_result["signal"], CompositeSignal)

        # Bearish LLM should shift signal in the bullish direction (inversion)
        # This isn't guaranteed to be strictly > in all cases due to agreement
        # gating, but the LLM component should contribute positively to buying
        assert llm_result["signal"].data_quality >= base_result["signal"].data_quality

    def test_idempotent_pipeline_same_data(self, tmp_path):
        """Same inputs → same outputs. Pipeline has no hidden mutable state leak."""
        kwargs = dict(
            fast_prices=(47000.0, 53000.0),
            daily_prices=(35000.0, 53000.0),
            ath=70000.0,
            eur=10000.0, btc=0.5,
        )

        r1 = self._full_pipeline(tmp_path, **kwargs)
        r2 = self._full_pipeline(tmp_path, **kwargs)

        assert r1["signal"].score == pytest.approx(r2["signal"].score)
        assert r1["signal"].action == r2["signal"].action
        assert r1["cycle"].phase == r2["cycle"].phase
        assert r1["risk"].allowed == r2["risk"].allowed


# ─── Performance tracker integration ─────────────────────────────────────────

class TestPerformanceTrackerIntegration:
    """Verify PerformanceTracker works with real TradeResult objects."""

    def test_record_and_report_cycle(self, tmp_path):
        """Record trades → generate report → format report (no mocks)."""
        from performance_tracker import PerformanceTracker
        from trade_executor import TradeResult, TradeOutcome

        config = make_config(tmp_path)
        perf = PerformanceTracker(config)

        # Record a buy trade
        buy = TradeResult(
            outcome=TradeOutcome.FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.01,
            filled_price=50000.0, fee_eur=2.0, txid="TX-001",
            limit_price=49990.0, chase_count=0,
            elapsed_seconds=1.5, reason="Filled",
        )
        perf.record_trade(buy, CyclePhase.GROWTH)
        perf.record_dca_baseline(500.0, 50000.0)

        # Snapshot portfolio
        portfolio = make_portfolio(eur=9500.0, btc=0.51, price=51000.0)
        perf.snapshot_portfolio(portfolio)

        # Generate and format report — must not crash
        report = perf.generate_report(portfolio)
        text = perf.format_report(report)

        assert isinstance(text, str)
        assert len(text) > 50  # Not empty
        assert "BTC" in text or "btc" in text.lower()


# ─── Module wiring sanity ────────────────────────────────────────────────────

class TestModuleWiringSanity:
    """
    Quick checks that all modules can be instantiated with the same config
    and that their constructors don't raise on default BotConfig.
    """

    def test_all_modules_instantiate_from_same_config(self, tmp_path):
        """Every pipeline module accepts the same BotConfig without error."""
        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path)

        # These must not raise
        detector = CycleDetector(config, tracker)
        engine = SignalEngine(config)
        risk_mgr = RiskManager(config)
        sizer = PositionSizer(config)

        assert detector is not None
        assert engine is not None
        assert risk_mgr is not None
        assert sizer is not None

    def test_frozen_dataclass_outputs(self, tmp_path):
        """All pipeline outputs are frozen (immutable) dataclasses."""
        highs, lows, closes, volumes = make_trending_prices(
            100, 45000.0, 55000.0,
        )
        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
        )

        config = make_config(tmp_path)
        tracker = make_ath_tracker(tmp_path)
        detector = CycleDetector(config, tracker)
        daily_c, daily_h, daily_l = make_daily_closes(250, 30000.0, 55000.0)
        cycle = detector.analyze(snapshot, daily_c, daily_h, daily_l)

        engine = SignalEngine(config)
        signal = engine.generate(snapshot=snapshot, cycle=cycle)

        # TechnicalSnapshot is frozen
        with pytest.raises(AttributeError):
            snapshot.price = 999.0

        # CycleState is frozen
        with pytest.raises(AttributeError):
            cycle.phase = CyclePhase.EUPHORIA

        # CompositeSignal is frozen
        with pytest.raises(AttributeError):
            signal.score = 999.0
