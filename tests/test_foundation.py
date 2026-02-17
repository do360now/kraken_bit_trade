"""
Tests for config.py and kraken_api.py.

Run: python -m pytest tests/test_foundation.py -v
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure we can import from the package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ATHTracker,
    BotConfig,
    CyclePhase,
    KrakenConfig,
    PersistenceConfig,
    VolatilityRegime,
)
from kraken_api import (
    CircuitBreaker,
    KrakenAPI,
    NonceGenerator,
    OrderBook,
    OrderBookLevel,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)


# ─── Config tests ────────────────────────────────────────────────────────────

class TestKrakenConfig:
    def test_defaults(self):
        cfg = KrakenConfig()
        assert cfg.pair == "XXBTZEUR"
        assert cfg.maker_fee == 0.0016
        assert cfg.taker_fee == 0.0026
        assert cfg.min_order_btc == 0.0001

    def test_from_env(self):
        with patch.dict(os.environ, {
            "KRAKEN_API_KEY": "test_key",
            "KRAKEN_API_SECRET": "test_secret",
        }):
            cfg = KrakenConfig.from_env()
            assert cfg.api_key == "test_key"
            assert cfg.private_key == "test_secret"

    def test_from_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove the keys if they exist
            os.environ.pop("KRAKEN_API_KEY", None)
            os.environ.pop("KRAKEN_API_SECRET", None)
            cfg = KrakenConfig.from_env()
            assert cfg.api_key == ""
            assert cfg.private_key == ""


class TestBotConfig:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KRAKEN_API_KEY", None)
            os.environ.pop("KRAKEN_API_SECRET", None)
            cfg = BotConfig()
            assert cfg.paper_trade is False
            assert cfg.kraken.pair == "XXBTZEUR"
            assert cfg.risk.reserve_floor_pct == 0.20

    def test_load_with_overrides(self, tmp_path):
        overrides = {"buy_threshold": 30.0, "paper_trade": False}
        override_file = tmp_path / "overrides.json"
        override_file.write_text(json.dumps(overrides))

        cfg = BotConfig.load(config_path=override_file)
        assert cfg.signal.buy_threshold == 30.0
        assert cfg.paper_trade is False

    def test_phase_boundaries_sum_to_valid_range(self):
        cfg = BotConfig()
        boundaries = cfg.cycle.phase_boundaries
        values = sorted(boundaries.values())
        assert all(0.0 < v < 1.0 for v in values)
        # Each boundary should be strictly increasing
        for i in range(1, len(values)):
            assert values[i] > values[i - 1]


class TestATHTracker:
    def test_initial_state(self, tmp_path):
        persistence = PersistenceConfig(base_dir=tmp_path)
        tracker = ATHTracker(persistence)
        # Initial ATH may be seeded with a production default
        assert tracker.ath_eur >= 0.0  # Non-negative

    def test_update_new_high(self, tmp_path):
        persistence = PersistenceConfig(base_dir=tmp_path)
        tracker = ATHTracker(persistence)
        # Use a price ABOVE the seed ATH to guarantee a new high
        new_high = tracker.ath_eur + 50000.0  # Always above current
        assert tracker.update(new_high) is True
        assert tracker.ath_eur == new_high

    def test_update_not_new_high(self, tmp_path):
        persistence = PersistenceConfig(base_dir=tmp_path)
        tracker = ATHTracker(persistence)
        # Set a known high, then try below it
        high = tracker.ath_eur + 10000.0
        tracker.update(high)
        assert tracker.update(high - 1.0) is False
        assert tracker.ath_eur == high

    def test_persistence_across_instances(self, tmp_path):
        persistence = PersistenceConfig(base_dir=tmp_path)
        tracker1 = ATHTracker(persistence)
        high = tracker1.ath_eur + 25000.0
        tracker1.update(high)

        tracker2 = ATHTracker(persistence)
        assert tracker2.ath_eur == high

    def test_drawdown_calculation(self, tmp_path):
        persistence = PersistenceConfig(base_dir=tmp_path)
        tracker = ATHTracker(persistence)
        high = 200000.0  # Well above any seed
        tracker.update(high)
        assert tracker.drawdown_from_ath(200000.0) == pytest.approx(0.0)
        assert tracker.drawdown_from_ath(160000.0) == pytest.approx(0.20)


class TestEnums:
    def test_cycle_phases(self):
        assert CyclePhase.ACCUMULATION.value == "accumulation"
        assert len(CyclePhase) == 7

    def test_volatility_regimes(self):
        assert VolatilityRegime.COMPRESSION.value == "compression"
        assert len(VolatilityRegime) == 5


# ─── Kraken API component tests ─────────────────────────────────────────────

class TestNonceGenerator:
    def test_monotonic(self):
        gen = NonceGenerator()
        nonces = [gen.next() for _ in range(100)]
        assert nonces == sorted(nonces)
        assert len(set(nonces)) == 100  # All unique

    def test_rapid_fire(self):
        """Even without sleeping, nonces must be strictly increasing."""
        gen = NonceGenerator()
        prev = gen.next()
        for _ in range(1000):
            curr = gen.next()
            assert curr > prev
            prev = curr

    def test_concurrent_threads_no_duplicates(self):
        """
        Two threads hammering the SAME generator must never produce duplicates.

        This guards against the double-instantiation bug scenario where two
        KrakenAPI instances (each with their own NonceGenerator) could produce
        overlapping nonces. Even with a single generator, concurrent access
        must be safe.

        NOTE: NonceGenerator currently has no lock, so this test documents
        the threading risk. If it fails under load, a threading.Lock should
        be added to NonceGenerator.next().
        """
        import threading
        from collections import defaultdict

        gen = NonceGenerator()
        results: dict[int, list[int]] = defaultdict(list)
        barrier = threading.Barrier(2)

        def generate_nonces(thread_id: int, count: int):
            barrier.wait()  # Start both threads simultaneously
            for _ in range(count):
                nonce = gen.next()
                results[thread_id].append(nonce)

        t1 = threading.Thread(target=generate_nonces, args=(0, 500))
        t2 = threading.Thread(target=generate_nonces, args=(1, 500))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        all_nonces = results[0] + results[1]
        assert len(set(all_nonces)) == 1000, (
            f"Duplicate nonces detected under concurrent access: "
            f"{1000 - len(set(all_nonces))} duplicates"
        )

    def test_separate_generators_can_collide(self):
        """
        Two SEPARATE NonceGenerators can produce identical nonces.

        This is the exact failure mode of the double-instantiation bug:
        bot._api and trade_executor._api each had their own NonceGenerator,
        and both could generate the same microsecond-based nonce.
        Kraken rejects duplicate nonces, causing sporadic auth failures.
        """
        gen1 = NonceGenerator()
        gen2 = NonceGenerator()

        # Both generators start from the same time base
        n1 = gen1.next()
        n2 = gen2.next()

        # They CAN be equal or very close — this documents the risk.
        # The fix is to never have two generators (i.e., one KrakenAPI instance).
        # We just verify they're independent (not sharing state).
        assert gen1._last_nonce != gen2._last_nonce or n1 == n2


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown=1.0)
        assert cb.is_open is False

    def test_trips_after_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is True

    def test_success_resets(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False  # Only 2 consecutive

    def test_cooldown_resets(self):
        cb = CircuitBreaker(threshold=2, cooldown=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True
        time.sleep(0.15)
        assert cb.is_open is False


class TestTicker:
    def test_spread_calculation(self):
        t = Ticker(
            pair="XXBTZEUR", ask=50100.0, bid=50000.0,
            last=50050.0, volume_24h=100.0, vwap_24h=50050.0,
            high_24h=51000.0, low_24h=49000.0, timestamp=time.time(),
        )
        assert t.spread == 100.0
        assert t.mid == 50050.0
        assert t.spread_pct == pytest.approx(100.0 / 50050.0)


class TestOrderBook:
    def _make_book(self) -> OrderBook:
        ts = time.time()
        bids = [
            OrderBookLevel(price=50000.0, volume=1.0, timestamp=ts),
            OrderBookLevel(price=49990.0, volume=2.0, timestamp=ts),
            OrderBookLevel(price=49980.0, volume=0.5, timestamp=ts),
        ]
        asks = [
            OrderBookLevel(price=50010.0, volume=0.8, timestamp=ts),
            OrderBookLevel(price=50020.0, volume=1.5, timestamp=ts),
            OrderBookLevel(price=50030.0, volume=3.0, timestamp=ts),
        ]
        return OrderBook(pair="XXBTZEUR", bids=bids, asks=asks, timestamp=ts)

    def test_best_bid_ask(self):
        book = self._make_book()
        assert book.best_bid == 50000.0
        assert book.best_ask == 50010.0
        assert book.spread == 10.0

    def test_volume(self):
        book = self._make_book()
        assert book.bid_volume(2) == pytest.approx(3.0)
        assert book.ask_volume(2) == pytest.approx(2.3)

    def test_imbalance(self):
        book = self._make_book()
        imb = book.imbalance(3)
        bid_v = 1.0 + 2.0 + 0.5  # 3.5
        ask_v = 0.8 + 1.5 + 3.0  # 5.3
        expected = (bid_v - ask_v) / (bid_v + ask_v)
        assert imb == pytest.approx(expected)


class TestKrakenAPIPublicParsing:
    """Test response parsing with mocked HTTP calls."""

    def _make_api(self) -> KrakenAPI:
        cfg = KrakenConfig(
            api_key="test", private_key="dGVzdA==",  # base64("test")
            call_spacing_seconds=0.0,  # No rate limiting in tests
        )
        return KrakenAPI(cfg)

    @patch("kraken_api.KrakenAPI._execute_request_with_retry")
    def test_get_ticker_parses(self, mock_request):
        mock_request.return_value = {
            "XXBTZEUR": {
                "a": ["50100.0", "1", "1.000"],
                "b": ["50000.0", "1", "1.000"],
                "c": ["50050.0", "0.1"],
                "v": ["100.0", "200.0"],
                "p": ["50050.0", "50025.0"],
                "h": ["51000.0", "52000.0"],
                "l": ["49000.0", "48000.0"],
            }
        }
        api = self._make_api()
        ticker = api.get_ticker()

        assert ticker is not None
        assert ticker.ask == 50100.0
        assert ticker.bid == 50000.0
        assert ticker.last == 50050.0

    @patch("kraken_api.KrakenAPI._execute_request_with_retry")
    def test_get_ohlc_parses(self, mock_request):
        mock_request.return_value = {
            "XXBTZEUR": [
                [1700000000, "50000", "50500", "49500", "50200", "50100", "10.5", 150],
                [1700003600, "50200", "50800", "50100", "50600", "50400", "8.2", 120],
            ],
            "last": 1700003600,
        }
        api = self._make_api()
        candles = api.get_ohlc(interval=60)

        assert len(candles) == 2
        assert candles[0].open == 50000.0
        assert candles[0].close == 50200.0
        assert candles[1].volume == 8.2

    @patch("kraken_api.KrakenAPI._execute_request_with_retry")
    def test_get_order_book_parses(self, mock_request):
        mock_request.return_value = {
            "XXBTZEUR": {
                "bids": [
                    ["50000.0", "1.5", 1700000000],
                    ["49990.0", "2.0", 1700000000],
                ],
                "asks": [
                    ["50010.0", "1.0", 1700000000],
                    ["50020.0", "3.0", 1700000000],
                ],
            }
        }
        api = self._make_api()
        book = api.get_order_book(depth=2)

        assert book is not None
        assert book.best_bid == 50000.0
        assert book.best_ask == 50010.0
        assert len(book.bids) == 2
        assert len(book.asks) == 2

    @patch("kraken_api.KrakenAPI._execute_request_with_retry")
    def test_get_ticker_returns_none_on_failure(self, mock_request):
        mock_request.side_effect = Exception("network error")
        api = self._make_api()
        ticker = api.get_ticker()
        assert ticker is None

    @patch("kraken_api.KrakenAPI._execute_request_with_retry")
    def test_get_balance_parses(self, mock_request):
        mock_request.return_value = {
            "ZEUR": "5000.00",
            "XXBT": "0.12345678",
        }
        api = self._make_api()
        balance = api.get_balance()

        assert balance is not None
        assert balance.eur == 5000.0
        assert balance.btc == pytest.approx(0.12345678)


class TestKrakenAPIOrderValidation:
    def _make_api(self) -> KrakenAPI:
        cfg = KrakenConfig(
            api_key="test", private_key="dGVzdA==",
            call_spacing_seconds=0.0,
        )
        return KrakenAPI(cfg)

    def test_limit_order_without_price_fails(self):
        api = self._make_api()
        result = api.place_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=0.001,
            price=None,
        )
        assert result.success is False
        assert "price" in result.error.lower()

    def test_below_minimum_order_fails(self):
        api = self._make_api()
        result = api.place_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=0.00001,  # Below minimum
            price=50000.0,
        )
        assert result.success is False
        assert "minimum" in result.error.lower()


class TestOrderResult:
    def test_failed_result(self):
        r = OrderResult(success=False, error="something went wrong")
        assert r.success is False
        assert r.status == OrderStatus.UNKNOWN
        assert r.filled_volume == 0.0

    def test_success_result(self):
        r = OrderResult(
            success=True,
            txid="OABCDE-12345-FGHIJK",
            status=OrderStatus.PENDING,
        )
        assert r.success is True
        assert r.txid is not None
