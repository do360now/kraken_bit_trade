"""
Tests for performance_tracker.py

Validates:
- Trade statistics computation (counts, volumes, averages)
- Fill rate and fee percentage calculations
- Net BTC accumulation tracking
- DCA baseline comparison and efficiency ratio
- Portfolio snapshot and drawdown tracking
- Phase breakdown aggregation
- Report generation and formatting
- State persistence across restarts
- Edge cases (no trades, zero prices, single trade)

Run: python -m pytest tests/test_performance_tracker.py -v
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BotConfig, CyclePhase, PersistenceConfig
from risk_manager import PortfolioState
from trade_executor import TradeResult, TradeOutcome
from performance_tracker import (
    AccumulationMetrics,
    PerformanceReport,
    PerformanceTracker,
    TradeStats,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path) -> BotConfig:
    return BotConfig(persistence=PersistenceConfig(base_dir=tmp_path))


def make_trade(
    side: str = "buy",
    outcome: TradeOutcome = TradeOutcome.FILLED,
    volume: float = 0.01,
    price: float = 50000.0,
    fee: float = 0.80,
    chase: int = 0,
    elapsed: float = 5.0,
) -> TradeResult:
    return TradeResult(
        outcome=outcome, side=side,
        requested_volume=volume, filled_volume=volume,
        filled_price=price, fee_eur=fee,
        txid="test_tx", limit_price=price,
        chase_count=chase, elapsed_seconds=elapsed,
        reason="test",
    )


def make_portfolio(
    eur: float = 10000.0,
    btc: float = 0.5,
    btc_price: float = 50000.0,
) -> PortfolioState:
    return PortfolioState(
        eur_balance=eur, btc_balance=btc,
        btc_price=btc_price, starting_eur=35000.0,
    )


# ─── TradeStats property tests ──────────────────────────────────────────────

class TestTradeStats:
    def test_net_btc_accumulated(self):
        stats = TradeStats(
            total_trades=3, buy_count=2, sell_count=1,
            filled_count=3, partial_count=0,
            cancelled_count=0, failed_count=0,
            total_btc_bought=0.05, total_btc_sold=0.01,
            total_eur_spent=2500.0, total_eur_received=600.0,
            total_fees_eur=3.0, avg_buy_price=50000.0,
            avg_sell_price=60000.0, avg_chase_count=0.5,
            avg_fill_time_seconds=8.0,
        )
        assert stats.net_btc_accumulated == pytest.approx(0.04)

    def test_fill_rate(self):
        stats = TradeStats(
            total_trades=10, buy_count=8, sell_count=2,
            filled_count=7, partial_count=1,
            cancelled_count=1, failed_count=1,
            total_btc_bought=0.0, total_btc_sold=0.0,
            total_eur_spent=0.0, total_eur_received=0.0,
            total_fees_eur=0.0, avg_buy_price=0.0,
            avg_sell_price=0.0, avg_chase_count=0.0,
            avg_fill_time_seconds=0.0,
        )
        assert stats.fill_rate == pytest.approx(0.8)

    def test_fill_rate_zero_trades(self):
        stats = TradeStats(
            total_trades=0, buy_count=0, sell_count=0,
            filled_count=0, partial_count=0,
            cancelled_count=0, failed_count=0,
            total_btc_bought=0.0, total_btc_sold=0.0,
            total_eur_spent=0.0, total_eur_received=0.0,
            total_fees_eur=0.0, avg_buy_price=0.0,
            avg_sell_price=0.0, avg_chase_count=0.0,
            avg_fill_time_seconds=0.0,
        )
        assert stats.fill_rate == 0.0

    def test_fee_pct(self):
        stats = TradeStats(
            total_trades=1, buy_count=1, sell_count=0,
            filled_count=1, partial_count=0,
            cancelled_count=0, failed_count=0,
            total_btc_bought=0.01, total_btc_sold=0.0,
            total_eur_spent=1000.0, total_eur_received=0.0,
            total_fees_eur=1.60, avg_buy_price=50000.0,
            avg_sell_price=0.0, avg_chase_count=0.0,
            avg_fill_time_seconds=5.0,
        )
        assert stats.fee_pct == pytest.approx(0.0016)  # 0.16%


# ─── Trade recording and stats computation ──────────────────────────────────

class TestTradeRecording:
    def test_record_and_compute(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker.record_trade(make_trade(side="buy", volume=0.01, price=50000.0), CyclePhase.GROWTH)
        tracker.record_trade(make_trade(side="buy", volume=0.02, price=48000.0), CyclePhase.GROWTH)
        tracker.record_trade(make_trade(side="sell", volume=0.005, price=55000.0), CyclePhase.EUPHORIA)

        stats = tracker.compute_trade_stats()
        assert stats.total_trades == 3
        assert stats.buy_count == 2
        assert stats.sell_count == 1
        assert stats.total_btc_bought == pytest.approx(0.03)
        assert stats.total_btc_sold == pytest.approx(0.005)
        assert stats.net_btc_accumulated == pytest.approx(0.025)

    def test_avg_buy_price(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        # Buy 0.01 @ 50000 (EUR 500) and 0.02 @ 48000 (EUR 960)
        tracker.record_trade(make_trade(side="buy", volume=0.01, price=50000.0, fee=0.80), CyclePhase.GROWTH)
        tracker.record_trade(make_trade(side="buy", volume=0.02, price=48000.0, fee=1.54), CyclePhase.GROWTH)

        stats = tracker.compute_trade_stats()
        expected_avg = (500.0 + 960.0) / 0.03  # ≈ 48666.67
        assert stats.avg_buy_price == pytest.approx(expected_avg, rel=0.01)

    def test_chase_and_time_averages(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker.record_trade(
            make_trade(chase=0, elapsed=5.0), CyclePhase.GROWTH,
        )
        tracker.record_trade(
            make_trade(chase=2, elapsed=15.0), CyclePhase.GROWTH,
        )

        stats = tracker.compute_trade_stats()
        assert stats.avg_chase_count == pytest.approx(1.0)
        assert stats.avg_fill_time_seconds == pytest.approx(10.0)

    def test_cancelled_counted(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        cancelled = TradeResult(
            outcome=TradeOutcome.CANCELLED, side="buy",
            requested_volume=0.01, filled_volume=0.0,
            filled_price=0.0, fee_eur=0.0, txid=None,
            limit_price=50000.0, chase_count=2,
            elapsed_seconds=30.0, reason="exhausted",
        )
        tracker.record_trade(cancelled, CyclePhase.GROWTH)

        stats = tracker.compute_trade_stats()
        assert stats.cancelled_count == 1
        assert stats.total_btc_bought == 0.0

    def test_empty_stats(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        stats = tracker.compute_trade_stats()
        assert stats.total_trades == 0
        assert stats.fill_rate == 0.0
        assert stats.fee_pct == 0.0


# ─── DCA baseline and accumulation metrics ──────────────────────────────────

class TestAccumulationMetrics:
    def test_basic_efficiency(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker._tracking_start = time.time() - 86400 * 10  # 10 days ago

        # We bought 0.03 BTC
        tracker.record_trade(make_trade(volume=0.02, price=48000.0), CyclePhase.ACCUMULATION)
        tracker.record_trade(make_trade(volume=0.01, price=50000.0), CyclePhase.GROWTH)

        # DCA baseline: same EUR at same times
        tracker.record_dca_baseline(960.0, 48000.0)  # 0.02 BTC
        tracker.record_dca_baseline(500.0, 50000.0)  # 0.01 BTC

        metrics = tracker.compute_accumulation_metrics(52000.0)

        assert metrics.total_btc_accumulated == pytest.approx(0.03)
        assert metrics.dca_baseline_btc == pytest.approx(0.03)
        assert metrics.efficiency_ratio == pytest.approx(1.0)
        assert metrics.btc_per_day > 0
        assert metrics.tracking_days == pytest.approx(10.0, abs=0.1)

    def test_outperforming_dca(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker._tracking_start = time.time() - 86400

        # We bought cheaper than DCA
        tracker.record_trade(make_trade(volume=0.03, price=45000.0), CyclePhase.ACCUMULATION)
        # DCA would have bought at higher price
        tracker.record_dca_baseline(1350.0, 50000.0)  # Only 0.027 BTC

        metrics = tracker.compute_accumulation_metrics(50000.0)
        assert metrics.efficiency_ratio > 1.0

    def test_unrealized_pnl(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker._tracking_start = time.time() - 86400

        tracker.record_trade(make_trade(volume=0.01, price=50000.0), CyclePhase.GROWTH)

        # Price went up to 55000
        metrics = tracker.compute_accumulation_metrics(55000.0)
        assert metrics.unrealized_pnl_pct == pytest.approx(0.10)

    def test_no_trades_metrics(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        metrics = tracker.compute_accumulation_metrics(50000.0)
        assert metrics.total_btc_accumulated == 0.0
        assert metrics.efficiency_ratio == 1.0  # Default
        assert metrics.tracking_days == 0.0

    def test_zero_dca_baseline(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker._tracking_start = time.time()
        tracker.record_trade(make_trade(volume=0.01), CyclePhase.GROWTH)

        metrics = tracker.compute_accumulation_metrics(50000.0)
        assert metrics.efficiency_ratio == float("inf")

    def test_dca_ignores_bad_input(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker.record_dca_baseline(0.0, 50000.0)
        tracker.record_dca_baseline(100.0, 0.0)
        tracker.record_dca_baseline(-50.0, 50000.0)
        assert len(tracker._dca_deposits) == 0


# ─── Drawdown tracking ──────────────────────────────────────────────────────

class TestDrawdownTracking:
    def test_max_drawdown_tracked(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))

        # Portfolio at peak
        tracker.snapshot_portfolio(make_portfolio(eur=10000, btc=1.0, btc_price=50000))
        # Portfolio drops
        tracker.snapshot_portfolio(make_portfolio(eur=10000, btc=1.0, btc_price=35000))
        # Peak was 60000, now 45000 → 25% drawdown
        assert tracker._max_drawdown == pytest.approx(0.25)

    def test_drawdown_updates_on_new_low(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker.snapshot_portfolio(make_portfolio(btc_price=60000))
        tracker.snapshot_portfolio(make_portfolio(btc_price=50000))
        dd1 = tracker._max_drawdown
        tracker.snapshot_portfolio(make_portfolio(btc_price=40000))
        assert tracker._max_drawdown > dd1

    def test_snapshots_bounded(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        for i in range(15000):
            tracker._snapshots.append({"timestamp": time.time(), "total_eur": 50000})
        tracker.snapshot_portfolio(make_portfolio())
        assert len(tracker._snapshots) <= 10001


# ─── Phase breakdown ────────────────────────────────────────────────────────

class TestPhaseBreakdown:
    def test_breakdown_by_phase(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker.record_trade(make_trade(volume=0.02), CyclePhase.ACCUMULATION)
        tracker.record_trade(make_trade(volume=0.01), CyclePhase.GROWTH)
        tracker.record_trade(make_trade(volume=0.01), CyclePhase.GROWTH)
        tracker.record_trade(
            make_trade(side="sell", volume=0.005, price=55000.0),
            CyclePhase.EUPHORIA,
        )

        breakdown = tracker.compute_phase_breakdown()

        assert "accumulation" in breakdown
        assert breakdown["accumulation"]["trades"] == 1
        assert breakdown["accumulation"]["btc_bought"] == pytest.approx(0.02)

        assert "growth" in breakdown
        assert breakdown["growth"]["trades"] == 2

        assert "euphoria" in breakdown
        assert breakdown["euphoria"]["btc_sold"] == pytest.approx(0.005)

    def test_empty_breakdown(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        assert tracker.compute_phase_breakdown() == {}


# ─── Report generation ──────────────────────────────────────────────────────

class TestReporting:
    def test_generate_report_structure(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker.record_trade(make_trade(volume=0.01), CyclePhase.GROWTH)
        tracker.snapshot_portfolio(make_portfolio())

        report = tracker.generate_report(make_portfolio())

        assert isinstance(report, PerformanceReport)
        assert isinstance(report.trade_stats, TradeStats)
        assert isinstance(report.accumulation, AccumulationMetrics)
        assert isinstance(report.portfolio, PortfolioState)
        assert report.generated_at
        assert report.tracking_since

    def test_format_report_readable(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        tracker.record_trade(make_trade(volume=0.01), CyclePhase.GROWTH)
        tracker.record_trade(
            make_trade(side="sell", volume=0.003, price=55000.0),
            CyclePhase.EUPHORIA,
        )
        tracker.snapshot_portfolio(make_portfolio())

        report = tracker.generate_report(make_portfolio())
        text = tracker.format_report(report)

        assert "PERFORMANCE REPORT" in text
        assert "PORTFOLIO" in text
        assert "ACCUMULATION" in text
        assert "TRADES" in text
        assert "RISK" in text
        assert "PHASE BREAKDOWN" in text

    def test_report_with_no_trades(self, tmp_path):
        tracker = PerformanceTracker(make_config(tmp_path))
        report = tracker.generate_report(make_portfolio())
        text = tracker.format_report(report)
        assert "PERFORMANCE REPORT" in text
        assert report.trade_stats.total_trades == 0


# ─── State persistence ──────────────────────────────────────────────────────

class TestPersistence:
    def test_trades_persist(self, tmp_path):
        config = make_config(tmp_path)
        tracker1 = PerformanceTracker(config)
        tracker1.record_trade(make_trade(volume=0.01), CyclePhase.GROWTH)
        tracker1.record_trade(make_trade(volume=0.02), CyclePhase.ACCUMULATION)

        tracker2 = PerformanceTracker(config)
        stats = tracker2.compute_trade_stats()
        assert stats.total_trades == 2
        assert stats.total_btc_bought == pytest.approx(0.03)

    def test_max_drawdown_persists(self, tmp_path):
        config = make_config(tmp_path)
        tracker1 = PerformanceTracker(config)
        tracker1.snapshot_portfolio(make_portfolio(btc_price=60000))
        tracker1.snapshot_portfolio(make_portfolio(btc_price=40000))
        dd1 = tracker1._max_drawdown
        tracker1._save_state()

        tracker2 = PerformanceTracker(config)
        assert tracker2._max_drawdown == pytest.approx(dd1)

    def test_dca_deposits_persist(self, tmp_path):
        config = make_config(tmp_path)
        tracker1 = PerformanceTracker(config)
        tracker1.record_dca_baseline(500.0, 50000.0)
        tracker1._save_state()

        tracker2 = PerformanceTracker(config)
        assert len(tracker2._dca_deposits) == 1
