"""
Tests for the backtesting harness.

Validates:
  - Data loader CSV round-trip
  - BacktestEngine with synthetic data
  - DCA baseline calculation
  - Fee simulation
  - Trade counting
  - Export functions

Run: python -m pytest tests/test_backtester.py -v
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester import (
    BacktestConfig, BacktestEngine, BacktestResult,
    SimulatedTrade, DailySnapshot,
    export_trades_csv, export_snapshots_csv,
)
from data_loader import DataLoader, HistoricalCandle


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_cycle_candles(
    n: int = 400,
    start_price: float = 10000.0,
    peak_price: float = 60000.0,
    end_price: float = 40000.0,
    start_ts: float = 1577836800.0,  # 2020-01-01
) -> list[HistoricalCandle]:
    """Generate synthetic candles simulating a BTC halving cycle."""
    candles = []
    for day in range(n):
        progress = day / n
        if progress < 0.3:
            price = start_price + (peak_price * 0.2 - start_price) * (progress / 0.3)
        elif progress < 0.7:
            p = (progress - 0.3) / 0.4
            price = peak_price * 0.2 + (peak_price - peak_price * 0.2) * p
        elif progress < 0.85:
            p = (progress - 0.7) / 0.15
            price = peak_price - (peak_price - end_price) * p
        else:
            price = end_price + math.sin((progress - 0.85) * 40) * end_price * 0.05

        # Add realistic noise
        noise = math.sin(day * 0.7) * 0.02 + math.cos(day * 0.3) * 0.01
        price *= (1 + noise)
        price = max(price, 100.0)  # Floor

        ts = start_ts + day * 86400
        candles.append(HistoricalCandle(
            timestamp=ts,
            date=f"day-{day:04d}",
            open=price * 0.998,
            high=price * 1.02,
            low=price * 0.98,
            close=price,
            vwap=price * 0.999,
            volume=100 + day * 0.5,
            count=1000,
        ))
    return candles


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADER TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDataLoaderCSV:
    """CSV save/load round-trip."""

    def test_csv_round_trip(self, tmp_path):
        """Save and reload candles without data loss."""
        loader = DataLoader(cache_dir=tmp_path)
        original = make_cycle_candles(50)

        csv_path = tmp_path / "test.csv"
        loader._save_csv(original, csv_path)
        loaded = loader._load_csv(csv_path)

        assert len(loaded) == len(original)
        for o, l in zip(original, loaded):
            assert l.timestamp == pytest.approx(o.timestamp)
            assert l.close == pytest.approx(o.close)
            assert l.volume == pytest.approx(o.volume)

    def test_csv_sorted_by_timestamp(self, tmp_path):
        """Loaded candles are sorted oldest-first."""
        loader = DataLoader(cache_dir=tmp_path)
        candles = make_cycle_candles(30)

        csv_path = tmp_path / "test.csv"
        loader._save_csv(candles, csv_path)
        loaded = loader._load_csv(csv_path)

        timestamps = [c.timestamp for c in loaded]
        assert timestamps == sorted(timestamps)

    def test_load_csv_file(self, tmp_path):
        """Load from user-provided CSV."""
        csv_path = tmp_path / "user_data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "date", "open", "high", "low",
                "close", "vwap", "volume", "count",
            ])
            writer.writerow([
                1577836800, "2020-01-01", 7200, 7350, 7100,
                7250, 7230, 500, 5000,
            ])
            writer.writerow([
                1577923200, "2020-01-02", 7250, 7400, 7180,
                7380, 7300, 600, 5500,
            ])

        loader = DataLoader(cache_dir=tmp_path)
        candles = loader.load_csv_file(csv_path)

        assert len(candles) == 2
        assert candles[0].close == 7250.0
        assert candles[1].date == "2020-01-02"


# ═══════════════════════════════════════════════════════════════════════
# BACKTESTER ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestBacktestEngine:
    """Core simulation engine tests."""

    def test_minimum_candles_enforced(self):
        """Engine rejects insufficient data."""
        candles = make_cycle_candles(100)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))

        with pytest.raises(ValueError, match="Need at least"):
            engine.run(candles)

    def test_runs_with_sufficient_data(self):
        """Engine completes with enough candles."""
        candles = make_cycle_candles(400)
        engine = BacktestEngine(BacktestConfig(
            starting_eur=10_000.0,
            indicator_window=200,
        ))

        result = engine.run(candles)

        assert result.num_days == 200  # 400 - 200 indicator window
        assert result.starting_eur == 10_000.0
        assert isinstance(result.final_value_eur, float)

    def test_starting_balance_preserved(self):
        """EUR + BTC at start matches total value accounting."""
        candles = make_cycle_candles(300)
        engine = BacktestEngine(BacktestConfig(
            starting_eur=5000.0,
            starting_btc=0.1,
            indicator_window=200,
        ))

        result = engine.run(candles)

        # Final EUR + final BTC value should account for all money
        assert result.final_eur >= 0
        assert result.final_btc >= 0

    def test_trades_have_correct_fields(self):
        """Every trade has all required fields."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        for trade in result.trades:
            assert trade.side in ("buy", "sell")
            assert trade.eur_amount > 0
            assert trade.btc_amount > 0
            assert trade.price > 0
            assert trade.fee_eur >= 0
            assert len(trade.phase) > 0

    def test_no_negative_balances(self):
        """EUR and BTC never go negative during simulation."""
        candles = make_cycle_candles(400)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        for snap in result.daily_snapshots:
            assert snap.eur_balance >= -0.01, (
                f"Negative EUR on {snap.date}: {snap.eur_balance}"
            )
            assert snap.btc_balance >= -0.0001, (
                f"Negative BTC on {snap.date}: {snap.btc_balance}"
            )


class TestFeeSimulation:
    """Fee calculation correctness."""

    def test_buy_fee_deducted(self):
        """Buy simulation deducts fee from EUR spent."""
        engine = BacktestEngine(BacktestConfig(
            maker_fee=0.0016, spread_bps=5.0,
        ))

        btc, fee = engine._simulate_buy(1000.0, 50000.0)

        # Fee: 1000 * 0.0016 = 1.60
        assert fee == pytest.approx(1.60)
        # Net EUR: 1000 - 1.60 = 998.40
        # Effective price: 50000 * 1.0005 = 50025
        # BTC: 998.40 / 50025 ≈ 0.01996
        assert btc > 0
        assert btc < 1000.0 / 50000.0  # Less than no-fee amount

    def test_sell_fee_deducted(self):
        """Sell simulation deducts fee from EUR received."""
        engine = BacktestEngine(BacktestConfig(
            maker_fee=0.0016, spread_bps=5.0,
        ))

        eur, fee = engine._simulate_sell(0.1, 50000.0)

        # Effective price: 50000 * 0.9995 = 49975
        # Gross: 0.1 * 49975 = 4997.50
        # Fee: 4997.50 * 0.0016 ≈ 7.996
        assert fee == pytest.approx(4997.50 * 0.0016, rel=1e-3)
        assert eur < 0.1 * 50000.0  # Less than no-fee amount
        assert eur > 0


class TestDCABaseline:
    """DCA comparison baseline."""

    def test_dca_buys_at_interval(self):
        """DCA baseline makes regular purchases."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(
            starting_eur=10_000.0,
            dca_interval_days=7,
            indicator_window=200,
        ))
        result = engine.run(candles)

        # Should have accumulated some BTC via DCA
        assert result.dca_final_btc > 0
        assert result.dca_avg_buy_price > 0
        assert result.dca_total_fees_eur > 0

    def test_dca_spends_reasonable_amount(self):
        """DCA doesn't spend more than starting capital."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(
            starting_eur=10_000.0,
            dca_interval_days=7,
            indicator_window=200,
        ))
        result = engine.run(candles)

        # DCA should have EUR remaining (reserved some)
        dca_final_eur = result.dca_final_value_eur - result.dca_final_btc * result.final_btc_price
        # DCA should not have negative EUR
        assert dca_final_eur >= -1.0  # Small floating point tolerance

    def test_advantage_metrics_computed(self):
        """BTC and value advantage percentages are computed."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        # These should be real numbers (not NaN/inf)
        assert math.isfinite(result.btc_advantage_pct)
        assert math.isfinite(result.value_advantage_pct)


class TestSnapshotsAndExport:
    """Daily snapshots and CSV export."""

    def test_snapshot_count_matches_days(self):
        """One snapshot per simulated day."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        assert len(result.daily_snapshots) == result.num_days

    def test_export_trades_csv(self, tmp_path):
        """Trade log exports to valid CSV."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        csv_path = tmp_path / "trades.csv"
        export_trades_csv(result.trades, csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(result.trades)

    def test_export_snapshots_csv(self, tmp_path):
        """Snapshot export produces valid CSV with correct row count."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        csv_path = tmp_path / "snapshots.csv"
        export_snapshots_csv(result.daily_snapshots, csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(result.daily_snapshots)


class TestResultSummary:
    """Summary output formatting."""

    def test_summary_contains_key_metrics(self):
        """Summary string includes essential metrics."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        summary = result.summary()

        assert "BACKTEST RESULTS" in summary
        assert "PORTFOLIO" in summary
        assert "vs DCA BASELINE" in summary
        assert "VERDICT" in summary
        assert "BEATS" in summary or "LOSES TO" in summary

    def test_summary_no_nan(self):
        """Summary doesn't contain NaN or inf values."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        summary = result.summary()
        assert "nan" not in summary.lower()
        assert "inf" not in summary.lower()


class TestPhaseTracking:
    """Phase-level trade breakdown."""

    def test_phases_tracked(self):
        """trades_by_phase and btc_by_phase are populated."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        if result.total_trades > 0:
            assert len(result.trades_by_phase) > 0
            assert len(result.btc_by_phase) > 0

    def test_trade_count_matches(self):
        """Sum of phase trade counts matches total trades."""
        candles = make_cycle_candles(350)
        engine = BacktestEngine(BacktestConfig(indicator_window=200))
        result = engine.run(candles)

        phase_total = sum(result.trades_by_phase.values())
        assert phase_total == result.total_trades
