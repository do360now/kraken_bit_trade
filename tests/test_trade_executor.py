"""
Tests for trade_executor.py

Validates:
- Spread-aware pricing (buy: offset from ask, sell: offset from bid)
- Urgency levels (LOW/MEDIUM/HIGH affect price aggressiveness)
- Tick size rounding
- Chase logic (re-pricing toward market on TTL expiry)
- Slippage guard (rejects orders too far from mid)
- Order lifecycle (filled, partially filled, cancelled, failed)
- Post-trade bookkeeping (risk manager record, trade log)
- Pre-flight checks (zero size, no ticker, below minimum)
- TradeResult properties (eur_value, success)
- Trade log persistence (JSONL format)

Run: python -m pytest tests/test_trade_executor.py -v
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BotConfig,
    ExecutionConfig,
    PersistenceConfig,
    Urgency,
)
from kraken_api import (
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)
from risk_manager import RiskManager
from position_sizer import BuySize, SellTier
from trade_executor import (
    TradeExecutor,
    TradeOutcome,
    TradeResult,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        persistence=PersistenceConfig(base_dir=tmp_path),
        execution=ExecutionConfig(
            order_ttl_seconds=1.0,       # Short TTL for tests
            check_interval_seconds=0.01,  # Fast polling for tests
            max_chase_attempts=2,
        ),
    )


def make_ticker(
    bid: float = 49950.0,
    ask: float = 50050.0,
) -> Ticker:
    return Ticker(
        pair="XXBTZEUR",
        bid=bid, ask=ask, last=50000.0,
        volume_24h=1000.0, vwap_24h=50000.0,
        high_24h=51000.0, low_24h=49000.0,
        timestamp=time.time(),
    )


def make_buy_size(
    eur: float = 500.0,
    btc: float = 0.01,
) -> BuySize:
    return BuySize(
        eur_amount=eur, btc_amount=btc,
        fraction_of_capital=0.05,
        adjustments={"test": 1.0},
        reason="test buy",
    )


def make_sell_tier(btc: float = 0.05) -> SellTier:
    return SellTier(
        tier_index=0, threshold_pct=0.15,
        sell_pct=0.10, btc_amount=btc,
        reason="test sell",
    )


def make_executor(tmp_path: Path) -> tuple[TradeExecutor, MagicMock, MagicMock]:
    """Create executor with mocked API and risk manager."""
    config = make_config(tmp_path)
    mock_api = MagicMock()
    mock_risk = MagicMock()

    executor = TradeExecutor(
        api=mock_api,
        risk_manager=mock_risk,
        config=config,
    )
    return executor, mock_api, mock_risk


# ─── TradeResult property tests ──────────────────────────────────────────────

class TestTradeResult:
    def test_eur_value(self):
        r = TradeResult(
            outcome=TradeOutcome.FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.01,
            filled_price=50000.0, fee_eur=0.80, txid="abc",
            limit_price=49990.0, chase_count=0,
            elapsed_seconds=5.0, reason="test",
        )
        assert r.eur_value == pytest.approx(500.0)

    def test_success_filled(self):
        r = TradeResult(
            outcome=TradeOutcome.FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.01,
            filled_price=50000.0, fee_eur=0.80, txid="abc",
            limit_price=49990.0, chase_count=0,
            elapsed_seconds=5.0, reason="test",
        )
        assert r.success is True

    def test_success_partial(self):
        r = TradeResult(
            outcome=TradeOutcome.PARTIALLY_FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.005,
            filled_price=50000.0, fee_eur=0.40, txid="abc",
            limit_price=49990.0, chase_count=1,
            elapsed_seconds=10.0, reason="test",
        )
        assert r.success is True

    def test_not_success_cancelled(self):
        r = TradeResult(
            outcome=TradeOutcome.CANCELLED, side="buy",
            requested_volume=0.01, filled_volume=0.0,
            filled_price=0.0, fee_eur=0.0, txid=None,
            limit_price=49990.0, chase_count=2,
            elapsed_seconds=15.0, reason="test",
        )
        assert r.success is False

    def test_frozen(self):
        r = TradeResult(
            outcome=TradeOutcome.FILLED, side="buy",
            requested_volume=0.01, filled_volume=0.01,
            filled_price=50000.0, fee_eur=0.80, txid="abc",
            limit_price=49990.0, chase_count=0,
            elapsed_seconds=5.0, reason="test",
        )
        with pytest.raises(AttributeError):
            r.filled_volume = 0.0  # type: ignore


# ─── Spread-aware pricing tests ─────────────────────────────────────────────

class TestSpreadPricing:
    def test_buy_price_below_ask(self, tmp_path):
        """Buy limit should be below ask (we want a better price)."""
        executor, _, _ = make_executor(tmp_path)
        ticker = make_ticker(bid=49950.0, ask=50050.0)
        price = executor._compute_buy_price(ticker, Urgency.MEDIUM)
        assert price < ticker.ask
        assert price > ticker.bid

    def test_sell_price_above_bid(self, tmp_path):
        """Sell limit should be above bid (we want a better price)."""
        executor, _, _ = make_executor(tmp_path)
        ticker = make_ticker(bid=49950.0, ask=50050.0)
        price = executor._compute_sell_price(ticker, Urgency.MEDIUM)
        assert price > ticker.bid
        assert price < ticker.ask

    def test_low_urgency_better_price(self, tmp_path):
        """LOW urgency should give a better (lower buy / higher sell) price."""
        executor, _, _ = make_executor(tmp_path)
        ticker = make_ticker(bid=49950.0, ask=50050.0)

        buy_low = executor._compute_buy_price(ticker, Urgency.LOW)
        buy_high = executor._compute_buy_price(ticker, Urgency.HIGH)
        assert buy_low < buy_high  # Lower buy = better for us

        sell_low = executor._compute_sell_price(ticker, Urgency.LOW)
        sell_high = executor._compute_sell_price(ticker, Urgency.HIGH)
        assert sell_low > sell_high  # Higher sell = better for us

    def test_high_urgency_near_market(self, tmp_path):
        """HIGH urgency should be near the market price."""
        executor, _, _ = make_executor(tmp_path)
        ticker = make_ticker(bid=49950.0, ask=50050.0)

        buy = executor._compute_buy_price(ticker, Urgency.HIGH)
        sell = executor._compute_sell_price(ticker, Urgency.HIGH)

        assert abs(buy - ticker.ask) < 10.0
        assert abs(sell - ticker.bid) < 10.0

    def test_tick_rounding(self, tmp_path):
        executor, _, _ = make_executor(tmp_path)
        assert executor._round_to_tick(50000.03) == 50000.0
        assert executor._round_to_tick(50000.07) == 50000.1
        assert executor._round_to_tick(50000.15) == 50000.2


# ─── Chase price tests ──────────────────────────────────────────────────────

class TestChasePrice:
    def test_buy_chase_moves_up(self, tmp_path):
        executor, _, _ = make_executor(tmp_path)
        ticker = make_ticker(bid=49950.0, ask=50050.0)
        initial = 49980.0
        chased = executor._chase_price(initial, OrderSide.BUY, ticker, 1)
        assert chased > initial

    def test_sell_chase_moves_down(self, tmp_path):
        executor, _, _ = make_executor(tmp_path)
        ticker = make_ticker(bid=49950.0, ask=50050.0)
        initial = 50020.0
        chased = executor._chase_price(initial, OrderSide.SELL, ticker, 1)
        assert chased < initial

    def test_chase_converges_toward_market(self, tmp_path):
        executor, _, _ = make_executor(tmp_path)
        ticker = make_ticker(bid=49950.0, ask=50050.0)
        price = 49970.0
        for i in range(5):
            price = executor._chase_price(price, OrderSide.BUY, ticker, i + 1)
        assert price > 49970.0
        assert price <= ticker.ask


# ─── Pre-flight check tests ─────────────────────────────────────────────────

class TestPreFlightChecks:
    def test_zero_buy_size_skipped(self, tmp_path):
        executor, _, _ = make_executor(tmp_path)
        result = executor.execute_buy(
            BuySize(eur_amount=0.0, btc_amount=0.0,
                    fraction_of_capital=0.0, adjustments={},
                    reason="no capital"),
        )
        assert result.outcome == TradeOutcome.SKIPPED

    def test_zero_sell_size_skipped(self, tmp_path):
        executor, _, _ = make_executor(tmp_path)
        result = executor.execute_sell(
            SellTier(tier_index=0, threshold_pct=0.15,
                     sell_pct=0.10, btc_amount=0.0, reason="test"),
        )
        assert result.outcome == TradeOutcome.SKIPPED

    def test_no_ticker_fails(self, tmp_path):
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = None
        result = executor.execute_buy(make_buy_size())
        assert result.outcome == TradeOutcome.FAILED
        assert "ticker" in result.reason.lower()


# ─── Buy execution tests (mocked API) ───────────────────────────────────────

class TestBuyExecution:
    @patch("trade_executor.time.sleep")
    def test_immediate_fill(self, mock_sleep, tmp_path):
        executor, mock_api, mock_risk = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.FILLED,
            filled_volume=0.01, filled_price=49990.0, fee=0.80,
        )

        result = executor.execute_buy(make_buy_size())

        assert result.outcome == TradeOutcome.FILLED
        assert result.filled_volume == pytest.approx(0.01)
        assert result.filled_price == pytest.approx(49990.0)
        assert result.success is True
        mock_risk.record_trade.assert_called_once()

    @patch("trade_executor.time.sleep")
    def test_placement_failure(self, mock_sleep, tmp_path):
        executor, mock_api, mock_risk = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=False, status=OrderStatus.FAILED,
            error="Insufficient funds",
        )

        result = executor.execute_buy(make_buy_size())

        assert result.outcome == TradeOutcome.FAILED
        assert "insufficient" in result.reason.lower()
        mock_risk.record_trade.assert_not_called()

    @patch("trade_executor.time.sleep")
    def test_below_minimum_volume(self, mock_sleep, tmp_path):
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        tiny_buy = BuySize(
            eur_amount=1.0, btc_amount=0.00001,
            fraction_of_capital=0.001, adjustments={},
            reason="tiny",
        )
        result = executor.execute_buy(tiny_buy)
        assert result.outcome == TradeOutcome.FAILED
        assert "minimum" in result.reason.lower()


# ─── Sell execution tests ────────────────────────────────────────────────────

class TestSellExecution:
    @patch("trade_executor.time.sleep")
    def test_immediate_fill(self, mock_sleep, tmp_path):
        executor, mock_api, mock_risk = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.FILLED,
            filled_volume=0.05, filled_price=50010.0, fee=4.0,
        )

        result = executor.execute_sell(make_sell_tier())

        assert result.outcome == TradeOutcome.FILLED
        assert result.side == "sell"
        assert result.filled_volume == pytest.approx(0.05)
        mock_risk.record_trade.assert_called_once()


# ─── Chase logic tests ──────────────────────────────────────────────────────

class TestChaseLogic:
    @patch("trade_executor.time.sleep")
    @patch("trade_executor.time.time")
    def test_chase_after_ttl_expiry(self, mock_time, mock_sleep, tmp_path):
        """If order doesn't fill within TTL, cancel and re-price."""
        executor, mock_api, mock_risk = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()

        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )

        # Simulate time advancing: each call to time.time() adds 0.5s
        # TTL is 1.0s, so after 2-3 queries the order will expire
        time_counter = [1000.0]
        def advancing_time():
            time_counter[0] += 0.5
            return time_counter[0]
        mock_time.side_effect = advancing_time

        # First order: always PENDING (expires). Second order: fills.
        place_count = [0]
        def mock_place(**kwargs):
            place_count[0] += 1
            return OrderResult(
                success=True, txid=f"tx{place_count[0]}",
                status=OrderStatus.PENDING,
            )
        mock_api.place_order.side_effect = mock_place

        query_count = [0]
        def mock_query(txid):
            query_count[0] += 1
            # First order queries: always pending
            if txid == "tx1":
                return OrderResult(
                    success=True, txid=txid, status=OrderStatus.PENDING,
                )
            # Second+ order: fill immediately
            return OrderResult(
                success=True, txid=txid, status=OrderStatus.FILLED,
                filled_volume=0.01, filled_price=50010.0, fee=0.80,
            )

        mock_api.query_order.side_effect = mock_query
        mock_api.cancel_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.CANCELLED,
        )

        result = executor.execute_buy(make_buy_size())

        # Cancel should have been called for the expired first order
        assert mock_api.cancel_order.called
        # Should have placed more than one order (initial + chase)
        assert place_count[0] >= 2

        result = executor.execute_buy(make_buy_size())

        assert mock_api.cancel_order.called

    @patch("trade_executor.time.sleep")
    def test_all_chases_exhausted(self, mock_sleep, tmp_path):
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.cancel_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.CANCELLED,
        )

        result = executor.execute_buy(make_buy_size())

        assert result.outcome == TradeOutcome.CANCELLED
        assert result.success is False


# ─── Slippage guard tests ────────────────────────────────────────────────────

class TestSlippageGuard:
    def test_extreme_spread_blocked(self, tmp_path):
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker(bid=40000.0, ask=60000.0)

        result = executor.execute_buy(make_buy_size())

        assert result.outcome == TradeOutcome.FAILED
        assert "slippage" in result.reason.lower()


# ─── Post-trade bookkeeping tests ────────────────────────────────────────────

class TestBookkeeping:
    @patch("trade_executor.time.sleep")
    def test_trade_logged(self, mock_sleep, tmp_path):
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.FILLED,
            filled_volume=0.01, filled_price=50000.0, fee=0.80,
        )

        executor.execute_buy(make_buy_size())

        log = executor.get_trade_log()
        assert len(log) == 1
        assert log[0]["side"] == "buy"
        assert log[0]["volume"] == pytest.approx(0.01)

    @patch("trade_executor.time.sleep")
    def test_trade_persisted_jsonl(self, mock_sleep, tmp_path):
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.FILLED,
            filled_volume=0.01, filled_price=50000.0, fee=0.80,
        )

        executor.execute_buy(make_buy_size())

        log_path = tmp_path / "trade_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["side"] == "buy"
        assert entry["outcome"] == "filled"

    @patch("trade_executor.time.sleep")
    def test_load_trade_history(self, mock_sleep, tmp_path):
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.FILLED,
            filled_volume=0.01, filled_price=50000.0, fee=0.80,
        )

        executor.execute_buy(make_buy_size())
        executor.execute_buy(make_buy_size())

        config = make_config(tmp_path)
        executor2 = TradeExecutor(
            api=mock_api, risk_manager=MagicMock(), config=config,
        )
        history = executor2.load_trade_history()
        assert len(history) == 2

    @patch("trade_executor.time.sleep")
    def test_load_trade_history_chronological_order(self, mock_sleep, tmp_path):
        """Trade history must be in chronological order (oldest first).

        JSONL append-order should naturally preserve chronology, but this
        explicitly asserts it. A future bulk-import or log rotation must
        not break ordering, since performance_tracker relies on it for
        DCA baseline and drawdown tracking.
        """
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.FILLED,
            filled_volume=0.01, filled_price=50000.0, fee=0.80,
        )

        # Execute 3 trades — timestamps should be monotonically increasing
        for _ in range(3):
            executor.execute_buy(make_buy_size())

        config = make_config(tmp_path)
        executor2 = TradeExecutor(
            api=mock_api, risk_manager=MagicMock(), config=config,
        )
        history = executor2.load_trade_history()
        assert len(history) == 3

        timestamps = [t["timestamp"] for t in history]
        assert timestamps == sorted(timestamps), (
            f"Trade history not in chronological order: {timestamps}"
        )

    @patch("trade_executor.time.sleep")
    def test_failed_trade_not_logged(self, mock_sleep, tmp_path):
        executor, mock_api, mock_risk = make_executor(tmp_path)
        mock_api.get_ticker.return_value = None

        executor.execute_buy(make_buy_size())

        assert len(executor.get_trade_log()) == 0
        mock_risk.record_trade.assert_not_called()


# ─── Limit order enforcement tests ──────────────────────────────────────────

class TestLimitOrderEnforcement:
    @patch("trade_executor.time.sleep")
    def test_only_limit_orders_placed(self, mock_sleep, tmp_path):
        """Verify no market orders are ever sent."""
        executor, mock_api, _ = make_executor(tmp_path)
        mock_api.get_ticker.return_value = make_ticker()
        mock_api.place_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.PENDING,
        )
        mock_api.query_order.return_value = OrderResult(
            success=True, txid="tx1", status=OrderStatus.FILLED,
            filled_volume=0.01, filled_price=50000.0, fee=0.80,
        )

        executor.execute_buy(make_buy_size())

        for call in mock_api.place_order.call_args_list:
            _, kwargs = call
            assert kwargs.get("order_type") == OrderType.LIMIT
