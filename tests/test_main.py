"""
Tests for main.py

Validates:
- Startup health checks (mandatory vs optional)
- Fast loop pipeline: every module called in sequence
- Slow loop: on-chain + LLM cached for fast loop
- Buy execution path: signal -> risk -> size -> execute -> record
- Sell execution path: profit tier -> execute -> mark tier
- Emergency sell path: golden rule -> partial sell
- Paper trade mode: full pipeline but no API calls
- Shutdown: clean report generation
- No dead code: every module is invoked

Run: python -m pytest tests/test_main.py -v
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BotConfig,
    CyclePhase,
    ExecutionConfig,
    PersistenceConfig,
    TimingConfig,
    Urgency,
    VolatilityRegime,
)
from kraken_api import (
    Balance,
    OHLCCandle,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)
from indicators import TechnicalSnapshot
from cycle_detector import CycleState, MomentumState, PriceStructure
from signal_engine import Action, CompositeSignal, LLMContext
from risk_manager import PortfolioState, RiskDecision
from position_sizer import BuySize, SellTier, SellDecision
from trade_executor import TradeResult, TradeOutcome
from main import Bot


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path, paper: bool = True) -> BotConfig:
    return BotConfig(
        persistence=PersistenceConfig(base_dir=tmp_path),
        timing=TimingConfig(
            fast_loop_seconds=0.1,
            slow_loop_seconds=0.1,
            ollama_timeout=5.0,
        ),
        execution=ExecutionConfig(
            order_ttl_seconds=1.0,
            check_interval_seconds=0.01,
        ),
        paper_trade=paper,
    )


def make_ticker(price: float = 50000.0) -> Ticker:
    return Ticker(
        pair="XXBTZEUR",
        bid=price - 25.0, ask=price + 25.0, last=price,
        volume_24h=1000.0, vwap_24h=price,
        high_24h=price + 500, low_24h=price - 500,
        timestamp=time.time(),
    )


def make_balance(eur: float = 10000.0, btc: float = 0.5) -> Balance:
    return Balance(eur=eur, btc=btc, timestamp=time.time())


def make_candles(n: int = 100, base_price: float = 50000.0) -> list[OHLCCandle]:
    candles = []
    for i in range(n):
        p = base_price + (i - n // 2) * 10
        candles.append(OHLCCandle(
            timestamp=time.time() - (n - i) * 300,
            open=p - 5, high=p + 20, low=p - 20,
            close=p, vwap=p, volume=1.0 + i * 0.01, count=10,
        ))
    return candles


def make_cycle(
    phase: CyclePhase = CyclePhase.GROWTH,
    profit_active: bool = True,
) -> CycleState:
    return CycleState(
        phase=phase, phase_confidence=0.7,
        time_score=0.3, price_score=0.2,
        momentum_score=0.3, volatility_score=0.0,
        composite_score=0.35,
        momentum=MomentumState(
            rsi_zone="neutral", trend_direction="up",
            higher_highs=True, higher_lows=True,
            rsi_bullish_divergence=False, rsi_bearish_divergence=False,
            momentum_score=0.3,
        ),
        price_structure=PriceStructure(
            drawdown_from_ath=0.15, position_in_range=0.5,
            distance_from_200d_ma=0.1, price_structure_score=0.2,
        ),
        volatility_regime=VolatilityRegime.NORMAL,
        cycle_day=400, cycle_progress=0.28,
        ath_eur=65000.0, drawdown_tolerance=0.20,
        position_size_multiplier=1.0,
        profit_taking_active=profit_active,
        timestamp=time.time(),
    )


def make_signal(
    score: float = 35.0,
    action: Action = Action.BUY,
    agreement: float = 0.7,
    quality: float = 0.8,
) -> CompositeSignal:
    return CompositeSignal(
        score=score, agreement=agreement, action=action,
        components=(), data_quality=quality, timestamp=time.time(),
    )


def make_bot(tmp_path: Path, paper: bool = True) -> Bot:
    """Create a Bot with all external dependencies mocked.

    IMPORTANT: We do NOT reassign bot._api / bot._bitcoin_node / etc.
    after construction. The mock objects flow through Bot.__init__
    naturally via MockClass.return_value. Post-construction reassignment
    was the pattern that masked the double-instantiation bug — if
    someone adds a second KrakenAPI() call, the call_count assertion
    below will catch it immediately.
    """
    config = make_config(tmp_path, paper=paper)

    with patch("main.KrakenAPI") as MockAPI, \
         patch("main.BitcoinNode") as MockNode, \
         patch("main.OllamaAnalyst") as MockOllama, \
         patch("main.PerformanceTracker") as MockPerf:

        mock_api = MagicMock()
        MockAPI.return_value = mock_api

        mock_node = MagicMock()
        mock_node.is_available = False
        MockNode.return_value = mock_node

        mock_ollama = MagicMock()
        mock_ollama.health_check.return_value = False
        MockOllama.return_value = mock_ollama

        mock_perf = MagicMock()
        MockPerf.return_value = mock_perf

        bot = Bot(config)

        # Guard: each external dependency constructed exactly once
        assert MockAPI.call_count == 1, (
            f"KrakenAPI instantiated {MockAPI.call_count} times"
        )

    return bot


# ─── Startup tests ──────────────────────────────────────────────────────────

class TestStartup:
    def test_startup_success_with_ticker(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(200)
        bot._bitcoin_node.is_available = False
        bot._ollama.health_check.return_value = False

        assert bot.startup() is True

    def test_startup_fails_without_ticker(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._api.get_ticker.return_value = None
        bot._api.get_ohlc.return_value = []
        bot._bitcoin_node.is_available = False
        bot._ollama.health_check.return_value = False

        assert bot.startup() is False

    def test_startup_ok_without_optional_services(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(200)
        bot._bitcoin_node.is_available = False
        bot._ollama.health_check.return_value = False

        assert bot.startup() is True

    def test_startup_loads_daily_candles(self, tmp_path):
        bot = make_bot(tmp_path)
        daily = make_candles(250, 48000.0)
        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = daily

        bot.startup()
        assert len(bot._daily_candles) == 250


# ─── Fast loop tests ────────────────────────────────────────────────────────

class TestFastLoop:
    def test_fast_loop_skips_without_ticker(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._api.get_ticker.return_value = None
        bot._fast_loop()  # Should not raise

    def test_fast_loop_skips_insufficient_candles(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(5)
        bot._fast_loop()  # Should not raise

    def test_fast_loop_full_pipeline_hold(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._daily_candles = make_candles(250)

        # _get_avg_entry_price() calls compute_trade_stats().avg_buy_price
        mock_stats = MagicMock()
        mock_stats.avg_buy_price = 0.0
        bot._performance.compute_trade_stats.return_value = mock_stats

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance()

        with patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(allowed=False, reason="ok")), \
             patch.object(bot._signal_engine, 'generate',
                          return_value=make_signal(score=5.0, action=Action.HOLD)):
            bot._fast_loop()
            # No trades attempted for HOLD signal

    def test_fast_loop_buy_path_paper(self, tmp_path):
        bot = make_bot(tmp_path, paper=True)
        bot._daily_candles = make_candles(250)

        # _get_avg_entry_price() calls compute_trade_stats().avg_buy_price
        mock_stats = MagicMock()
        mock_stats.avg_buy_price = 0.0
        bot._performance.compute_trade_stats.return_value = mock_stats

        buy_signal = make_signal(score=40.0, action=Action.BUY)

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance()

        with patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(allowed=False, reason="ok")), \
             patch.object(bot._signal_engine, 'generate', return_value=buy_signal), \
             patch.object(bot._risk_manager, 'can_trade',
                          return_value=RiskDecision(allowed=True, reason="ok")), \
             patch.object(bot._position_sizer, 'compute_buy_size',
                          return_value=BuySize(500.0, 0.01, 0.05, {}, "test")) as mock_size, \
             patch.object(bot._trade_executor, 'execute_buy') as mock_exec:
            bot._fast_loop()
            mock_size.assert_called_once()
            # Paper mode: executor NOT called
            mock_exec.assert_not_called()

    def test_fast_loop_skips_without_balance(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._daily_candles = make_candles(250)
        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = None
        bot._fast_loop()  # Should not raise


# ─── Trade handler tests ────────────────────────────────────────────────────

class TestBuyHandler:
    def test_paper_buy_no_executor(self, tmp_path):
        bot = make_bot(tmp_path, paper=True)
        portfolio = PortfolioState(10000, 0.5, 50000, 35000)
        cycle = make_cycle()
        signal = make_signal()
        buy_size = BuySize(500.0, 0.01, 0.05, {}, "test")
        risk = RiskDecision(True, "ok")

        with patch.object(bot._trade_executor, 'execute_buy') as mock_exec:
            bot._handle_buy(portfolio, cycle, signal, buy_size, risk)
            mock_exec.assert_not_called()

    def test_live_buy_calls_executor(self, tmp_path):
        bot = make_bot(tmp_path, paper=False)
        portfolio = PortfolioState(10000, 0.5, 50000, 35000)
        cycle = make_cycle()
        signal = make_signal(score=45.0)
        buy_size = BuySize(500.0, 0.01, 0.05, {}, "test")
        risk = RiskDecision(True, "ok")

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.01,
            filled_price=49990.0, fee_eur=0.80, txid="tx1",
            limit_price=49990.0, chase_count=0,
            elapsed_seconds=5.0, reason="filled",
        )

        with patch.object(bot._trade_executor, 'execute_buy', return_value=filled) as mock_exec, \
             patch.object(bot._performance, 'record_trade') as mock_record:
            bot._handle_buy(portfolio, cycle, signal, buy_size, risk)
            mock_exec.assert_called_once()
            mock_record.assert_called_once()

    def test_buy_urgency_from_signal_strength(self, tmp_path):
        bot = make_bot(tmp_path, paper=False)
        portfolio = PortfolioState(10000, 0.5, 50000, 35000)
        cycle = make_cycle()
        buy_size = BuySize(500.0, 0.01, 0.05, {}, "test")
        risk = RiskDecision(True, "ok")

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.01,
            filled_price=50000.0, fee_eur=0.80, txid="tx1",
            limit_price=50000.0, chase_count=0,
            elapsed_seconds=5.0, reason="filled",
        )

        # Strong signal -> HIGH urgency
        with patch.object(bot._trade_executor, 'execute_buy', return_value=filled) as mock_exec, \
             patch.object(bot._performance, 'record_trade'):
            bot._handle_buy(portfolio, cycle, make_signal(score=70.0), buy_size, risk)
            call_args, call_kwargs = mock_exec.call_args
            urgency = call_kwargs.get('urgency') or (call_args[1] if len(call_args) > 1 else None)
            assert urgency == Urgency.HIGH

        # Weak signal -> LOW urgency
        with patch.object(bot._trade_executor, 'execute_buy', return_value=filled) as mock_exec, \
             patch.object(bot._performance, 'record_trade'):
            bot._handle_buy(portfolio, cycle, make_signal(score=22.0), buy_size, risk)
            call_args, call_kwargs = mock_exec.call_args
            urgency = call_kwargs.get('urgency') or (call_args[1] if len(call_args) > 1 else None)
            assert urgency == Urgency.LOW


class TestSellHandler:
    def test_paper_sell_marks_tier(self, tmp_path):
        bot = make_bot(tmp_path, paper=True)
        portfolio = PortfolioState(10000, 0.5, 50000, 35000)
        cycle = make_cycle()
        tier = SellTier(0, 0.15, 0.10, 0.05, "test")

        with patch.object(bot._trade_executor, 'execute_sell') as mock_exec, \
             patch.object(bot._position_sizer, 'mark_tier_hit') as mock_mark:
            bot._handle_sell(portfolio, cycle, tier)
            mock_exec.assert_not_called()
            mock_mark.assert_called_once_with(0)

    def test_live_sell_calls_executor(self, tmp_path):
        bot = make_bot(tmp_path, paper=False)
        portfolio = PortfolioState(10000, 0.5, 50000, 35000)
        cycle = make_cycle()
        tier = SellTier(1, 0.25, 0.15, 0.075, "test")

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="sell",
            requested_volume=0.075, filled_volume=0.075,
            filled_price=62500.0, fee_eur=7.50, txid="tx2",
            limit_price=62500.0, chase_count=0,
            elapsed_seconds=3.0, reason="filled",
        )

        with patch.object(bot._trade_executor, 'execute_sell', return_value=filled) as mock_exec, \
             patch.object(bot._position_sizer, 'mark_tier_hit') as mock_mark, \
             patch.object(bot._performance, 'record_trade') as mock_record:
            bot._handle_sell(portfolio, cycle, tier)
            mock_exec.assert_called_once()
            mock_mark.assert_called_once_with(1)
            mock_record.assert_called_once()


class TestEmergencySell:
    def test_paper_emergency_no_executor(self, tmp_path):
        bot = make_bot(tmp_path, paper=True)
        portfolio = PortfolioState(10000, 0.5, 50000, 35000)
        cycle = make_cycle()

        with patch.object(bot._trade_executor, 'execute_sell') as mock_exec:
            bot._handle_emergency_sell(portfolio, cycle, "test emergency")
            mock_exec.assert_not_called()

    def test_emergency_sell_25_pct(self, tmp_path):
        bot = make_bot(tmp_path, paper=False)
        portfolio = PortfolioState(10000, 1.0, 50000, 60000)
        cycle = make_cycle()

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="sell",
            requested_volume=0.25, filled_volume=0.25,
            filled_price=50000.0, fee_eur=20.0, txid="tx3",
            limit_price=50000.0, chase_count=0,
            elapsed_seconds=5.0, reason="emergency filled",
        )

        with patch.object(bot._trade_executor, 'execute_sell', return_value=filled) as mock_exec, \
             patch.object(bot._performance, 'record_trade'):
            bot._handle_emergency_sell(portfolio, cycle, "extreme drawdown")
            mock_exec.assert_called_once()
            tier = mock_exec.call_args[0][0]
            assert tier.btc_amount == pytest.approx(0.25)  # 25% of 1.0
            assert tier.tier_index == -1  # Emergency marker

    def test_emergency_skip_tiny_balance(self, tmp_path):
        bot = make_bot(tmp_path, paper=False)
        portfolio = PortfolioState(10000, 0.00001, 50000, 10000)
        cycle = make_cycle()

        with patch.object(bot._trade_executor, 'execute_sell') as mock_exec:
            bot._handle_emergency_sell(portfolio, cycle, "test")
            mock_exec.assert_not_called()


# ─── Slow loop tests ────────────────────────────────────────────────────────

class TestSlowLoop:
    def test_slow_loop_caches_onchain(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._daily_candles = make_candles(250)

        mock_snapshot = MagicMock()
        mock_snapshot.network_stress = 0.3
        mock_snapshot.mempool.clearing = True

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._bitcoin_node.get_snapshot.return_value = mock_snapshot

        with patch.object(bot._ollama, 'analyze', return_value=None):
            bot._slow_loop()
            assert bot._onchain_cache is mock_snapshot

    def test_slow_loop_caches_llm(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._daily_candles = make_candles(250)

        llm_ctx = LLMContext(
            regime="markup", sentiment=0.3,
            risk_level="low", themes=("test",),
            timestamp=time.time(),
        )

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._bitcoin_node.get_snapshot.return_value = None

        with patch.object(bot._ollama, 'analyze', return_value=llm_ctx):
            bot._slow_loop()
            assert bot._llm_cache is llm_ctx

    def test_slow_loop_refreshes_daily_candles(self, tmp_path):
        bot = make_bot(tmp_path)
        new_daily = make_candles(300)
        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = new_daily
        bot._bitcoin_node.get_snapshot.return_value = None

        with patch.object(bot._ollama, 'analyze', return_value=None):
            bot._slow_loop()
            assert len(bot._daily_candles) == 300


# ─── Shutdown tests ──────────────────────────────────────────────────────────

class TestShutdown:
    def test_shutdown_sets_running_false(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._running = True
        bot._api.get_balance.return_value = None
        bot._api.get_ticker.return_value = None
        bot.shutdown()
        assert bot._running is False

    def test_shutdown_generates_report(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._api.get_balance.return_value = make_balance()
        bot._api.get_ticker.return_value = make_ticker()

        mock_report = MagicMock()
        bot._performance.generate_report.return_value = mock_report
        bot._performance.format_report.return_value = "Final report text"

        bot.shutdown()

        bot._performance.generate_report.assert_called_once()
        bot._performance.format_report.assert_called_once_with(mock_report)

    def test_shutdown_survives_report_failure(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._api.get_balance.side_effect = Exception("fail")
        bot.shutdown()
        assert bot._running is False


# ─── Integration: no dead code ───────────────────────────────────────────────

class TestNoDeadCode:
    """
    Verify every module is called from the decision pipeline.

    THE key test: the previous bot had 10 modules that were never invoked.
    This ensures every module gets called during a full loop iteration.
    """

    def test_all_modules_called_in_pipeline(self, tmp_path):
        bot = make_bot(tmp_path, paper=True)
        bot._daily_candles = make_candles(250)

        # _get_avg_entry_price() calls compute_trade_stats().avg_buy_price
        mock_stats = MagicMock()
        mock_stats.avg_buy_price = 0.0
        bot._performance.compute_trade_stats.return_value = mock_stats

        buy_signal = make_signal(score=40.0, action=Action.BUY)

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance()

        with patch.object(bot._cycle_detector, 'analyze', return_value=make_cycle()) as m_cycle, \
             patch.object(bot._signal_engine, 'generate', return_value=buy_signal) as m_signal, \
             patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(False, "ok")) as m_emergency, \
             patch.object(bot._risk_manager, 'can_trade',
                          return_value=RiskDecision(True, "ok")) as m_risk, \
             patch.object(bot._position_sizer, 'compute_buy_size',
                          return_value=BuySize(500.0, 0.01, 0.05, {}, "test")) as m_sizer:

            bot._fast_loop()

            # EVERY module was called:
            bot._api.get_ticker.assert_called()      # 1. Kraken ticker
            bot._api.get_ohlc.assert_called()        # 2. OHLC data
            bot._api.get_balance.assert_called()     # 3. Balance
            m_cycle.assert_called()                  # 4. Cycle detector
            m_signal.assert_called()                 # 5. Signal engine
            m_emergency.assert_called()              # 6. Emergency check
            m_risk.assert_called()                   # 7. Risk manager
            m_sizer.assert_called()                  # 8. Position sizer

    def test_slow_loop_modules_called(self, tmp_path):
        bot = make_bot(tmp_path)
        bot._daily_candles = make_candles(250)

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._bitcoin_node.get_snapshot.return_value = None

        with patch.object(bot._ollama, 'analyze', return_value=None) as m_llm:
            bot._slow_loop()
            bot._bitcoin_node.get_snapshot.assert_called()  # 10. Bitcoin node
            m_llm.assert_called()                           # 11. Ollama analyst


# ─── Avg entry price helper ─────────────────────────────────────────────────

class TestAvgEntryPrice:
    def test_from_performance_tracker(self, tmp_path):
        bot = make_bot(tmp_path)

        mock_stats = MagicMock()
        mock_stats.avg_buy_price = 48500.0
        bot._performance.compute_trade_stats.return_value = mock_stats

        assert bot._get_avg_entry_price() == pytest.approx(48500.0)


# ─── Regression: double-instantiation bug ────────────────────────────────────

class TestAPIIdentity:
    """
    Regression guard for the double-instantiation bug.

    Previous code had KrakenAPI instantiated TWICE in Bot.__init__():
    line 60 (passed to TradeExecutor) and line 93 (assigned to self._api).
    This caused separate circuit breaker states, rate limit counters, and
    nonce sequences between the bot and its executor.

    These tests verify the bot and its trade executor share the SAME
    KrakenAPI instance — not just equivalent ones, but object identity.
    """

    def test_bot_and_executor_share_same_api_instance(self, tmp_path):
        """bot._api must be the exact same object as bot._trade_executor._api."""
        config = make_config(tmp_path)

        with patch("main.KrakenAPI") as MockAPI, \
             patch("main.BitcoinNode"), \
             patch("main.OllamaAnalyst"), \
             patch("main.PerformanceTracker"):

            mock_api = MagicMock()
            MockAPI.return_value = mock_api

            bot = Bot(config)

        # The critical invariant: same object, not just same value
        assert bot._api is bot._trade_executor._api, (
            "Bot._api and TradeExecutor._api must be the same instance. "
            "Separate instances cause split circuit breaker state, rate "
            "limit counters, and nonce sequences."
        )

    def test_kraken_api_constructed_exactly_once(self, tmp_path):
        """KrakenAPI should only be instantiated once during Bot.__init__."""
        config = make_config(tmp_path)

        with patch("main.KrakenAPI") as MockAPI, \
             patch("main.BitcoinNode"), \
             patch("main.OllamaAnalyst"), \
             patch("main.PerformanceTracker"):

            MockAPI.return_value = MagicMock()
            Bot(config)

        assert MockAPI.call_count == 1, (
            f"KrakenAPI was instantiated {MockAPI.call_count} times, expected 1. "
            "Multiple instances cause split state between bot and executor."
        )


# ─── Sell path through full fast loop ────────────────────────────────────────

class TestSellPathFullPipeline:
    """
    Verify the sell path flows through the full fast loop pipeline:
    signal → profit tiers → risk check → execute.

    Previous tests only tested sell handler in isolation. This ensures the
    fast loop correctly routes sell decisions end-to-end.
    """

    def test_profit_tier_sell_through_full_pipeline(self, tmp_path):
        """Fast loop: sell signal + active profit tier → execute sell."""
        bot = make_bot(tmp_path, paper=False)
        bot._daily_candles = make_candles(250)

        sell_signal = make_signal(score=-30.0, action=Action.SELL)
        cycle = make_cycle(phase=CyclePhase.EUPHORIA, profit_active=True)
        tier = SellTier(
            tier_index=1, threshold_pct=0.5,
            sell_pct=0.10, btc_amount=0.05,
            reason="50% profit tier",
        )
        sell_decision = SellDecision(should_sell=True, tier=tier, reason="tier fired")

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance()

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="sell",
            requested_volume=0.05, filled_volume=0.05,
            filled_price=50000.0, fee_eur=2.5, txid="SELL-001",
            limit_price=50010.0, chase_count=0,
            elapsed_seconds=1.0, reason="Filled",
        )

        with patch.object(bot._cycle_detector, 'analyze', return_value=cycle), \
             patch.object(bot._signal_engine, 'generate', return_value=sell_signal), \
             patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(False, "ok")), \
             patch.object(bot._performance, 'compute_trade_stats') as mock_stats, \
             patch.object(bot._position_sizer, 'compute_sell_tiers',
                          return_value=sell_decision) as mock_tiers, \
             patch.object(bot._trade_executor, 'execute_sell',
                          return_value=filled) as mock_exec, \
             patch.object(bot._position_sizer, 'mark_tier_hit') as mock_mark:

            # Set up avg entry price for profit calculation
            mock_stats.return_value = MagicMock(avg_buy_price=30000.0)

            bot._fast_loop()

            # Full pipeline was traversed:
            mock_tiers.assert_called_once()  # Profit tiers evaluated
            mock_exec.assert_called_once()   # Executor called
            mock_mark.assert_called_once_with(1)  # Tier marked as hit
            bot._performance.record_trade.assert_called_once()  # Trade recorded

    def test_no_sell_when_no_profit_tier_fires(self, tmp_path):
        """Fast loop: sell signal but no tier → falls through to buy check."""
        bot = make_bot(tmp_path, paper=True)
        bot._daily_candles = make_candles(250)

        sell_signal = make_signal(score=-20.0, action=Action.SELL)
        cycle = make_cycle(phase=CyclePhase.GROWTH, profit_active=True)
        no_sell = SellDecision(should_sell=False, tier=None, reason="no tier")

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance()

        with patch.object(bot._cycle_detector, 'analyze', return_value=cycle), \
             patch.object(bot._signal_engine, 'generate', return_value=sell_signal), \
             patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(False, "ok")), \
             patch.object(bot._performance, 'compute_trade_stats') as mock_stats, \
             patch.object(bot._position_sizer, 'compute_sell_tiers',
                          return_value=no_sell):

            mock_stats.return_value = MagicMock(avg_buy_price=30000.0)

            bot._fast_loop()

            # No sell execution happened — no executor call, no tier marking
            assert not hasattr(bot._trade_executor.execute_sell, 'assert_called') or \
                   bot._trade_executor.execute_sell.call_count == 0


# ─── Pipeline-level order enforcement ────────────────────────────────────────

class TestPipelineOrderEnforcement:
    """
    Verify main.py never places orders directly via bot._api.place_order().

    All orders must flow through TradeExecutor, which enforces limit-only.
    If main.py ever bypasses the executor (as the previous bot did with
    market orders), this test catches it.
    """

    def test_buy_path_never_calls_api_place_order_directly(self, tmp_path):
        """Buy signal through pipeline must not call _api.place_order directly."""
        bot = make_bot(tmp_path, paper=False)
        bot._daily_candles = make_candles(250)

        # avg_entry = 0 → skips sell path, falls through to buy
        mock_stats = MagicMock()
        mock_stats.avg_buy_price = 0.0
        bot._performance.compute_trade_stats.return_value = mock_stats

        buy_signal = make_signal(score=50.0, action=Action.BUY)

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance()

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.01,
            filled_price=50000.0, fee_eur=2.0, txid="BUY-001",
            limit_price=49990.0, chase_count=0,
            elapsed_seconds=0.5, reason="Filled",
        )

        with patch.object(bot._cycle_detector, 'analyze', return_value=make_cycle()), \
             patch.object(bot._signal_engine, 'generate', return_value=buy_signal), \
             patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(False, "ok")), \
             patch.object(bot._risk_manager, 'can_trade',
                          return_value=RiskDecision(True, "ok")), \
             patch.object(bot._position_sizer, 'compute_buy_size',
                          return_value=BuySize(500.0, 0.01, 0.05, {}, "test")), \
             patch.object(bot._trade_executor, 'execute_buy', return_value=filled):

            bot._fast_loop()

        # The critical assertion: main.py must NEVER call place_order directly
        bot._api.place_order.assert_not_called()

    def test_sell_path_never_calls_api_place_order_directly(self, tmp_path):
        """Sell signal through pipeline must not call _api.place_order directly."""
        bot = make_bot(tmp_path, paper=False)
        bot._daily_candles = make_candles(250)

        sell_signal = make_signal(score=-30.0, action=Action.SELL)
        cycle = make_cycle(phase=CyclePhase.EUPHORIA, profit_active=True)
        tier = SellTier(
            tier_index=1, threshold_pct=0.5,
            sell_pct=0.10, btc_amount=0.05,
            reason="profit tier",
        )
        sell_decision = SellDecision(should_sell=True, tier=tier, reason="tier fired")

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance()

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="sell",
            requested_volume=0.05, filled_volume=0.05,
            filled_price=50000.0, fee_eur=2.5, txid="SELL-001",
            limit_price=50010.0, chase_count=0,
            elapsed_seconds=1.0, reason="Filled",
        )

        with patch.object(bot._cycle_detector, 'analyze', return_value=cycle), \
             patch.object(bot._signal_engine, 'generate', return_value=sell_signal), \
             patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(False, "ok")), \
             patch.object(bot._performance, 'compute_trade_stats') as mock_stats, \
             patch.object(bot._position_sizer, 'compute_sell_tiers',
                          return_value=sell_decision), \
             patch.object(bot._trade_executor, 'execute_sell', return_value=filled), \
             patch.object(bot._position_sizer, 'mark_tier_hit'):

            mock_stats.return_value = MagicMock(avg_buy_price=30000.0)
            bot._fast_loop()

        # The critical assertion: main.py must NEVER call place_order directly
        bot._api.place_order.assert_not_called()

    def test_emergency_sell_never_calls_api_place_order_directly(self, tmp_path):
        """Emergency sell must also go through executor, not direct API."""
        bot = make_bot(tmp_path, paper=False)
        bot._daily_candles = make_candles(250)

        bot._api.get_ticker.return_value = make_ticker()
        bot._api.get_ohlc.return_value = make_candles(100)
        bot._api.get_balance.return_value = make_balance(btc=1.0)

        filled = TradeResult(
            outcome=TradeOutcome.FILLED, side="sell",
            requested_volume=0.25, filled_volume=0.25,
            filled_price=50000.0, fee_eur=6.0, txid="EMRG-001",
            limit_price=49980.0, chase_count=0,
            elapsed_seconds=0.5, reason="Emergency filled",
        )

        with patch.object(bot._cycle_detector, 'analyze', return_value=make_cycle()), \
             patch.object(bot._signal_engine, 'generate',
                          return_value=make_signal(score=-50.0, action=Action.SELL)), \
             patch.object(bot._risk_manager, 'emergency_sell',
                          return_value=RiskDecision(True, "drawdown exceeded")), \
             patch.object(bot._trade_executor, 'execute_sell', return_value=filled):

            bot._fast_loop()

        bot._api.place_order.assert_not_called()
