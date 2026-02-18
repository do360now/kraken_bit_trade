#!/usr/bin/env python3
"""
Backtest runner — download data, run simulation, compare vs DCA.

Usage:
    # Full cycle 3+4 backtest (needs internet for first run):
    python run_backtest.py

    # Custom date range:
    python run_backtest.py --start 2022-01-01 --end 2024-12-31

    # With custom starting capital:
    python run_backtest.py --capital 50000

    # Skip data download (use cached CSV):
    python run_backtest.py --cached

    # Use a local CSV file:
    python run_backtest.py --csv data/my_btc_data.csv

    # Export results to CSV:
    python run_backtest.py --export
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from backtester import (
    BacktestConfig, BacktestEngine, BacktestResult,
    export_trades_csv, export_snapshots_csv,
)
from data_loader import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bitcoin accumulation bot backtester",
    )
    parser.add_argument(
        "--start", default="2024-04-20",
        help="Start date (YYYY-MM-DD). Default: 2024-04-20",
    )
    parser.add_argument(
        "--end", default="2027-02-18",
        help="End date (YYYY-MM-DD). Default: 2027-02-18",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Starting EUR capital. Default: 10000",
    )
    parser.add_argument(
        "--btc", type=float, default=0.0,
        help="Starting BTC balance. Default: 0.0",
    )
    parser.add_argument(
        "--dca-interval", type=int, default=7,
        help="DCA baseline buy interval in days. Default: 7",
    )
    parser.add_argument(
        "--dca-amount", type=float, default=0.0,
        help="DCA amount per buy (0 = auto-calculate). Default: 0",
    )
    parser.add_argument(
        "--cached", action="store_true",
        help="Only use cached data, don't fetch from Kraken",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a local CSV file with OHLCV data",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export trades and snapshots to CSV files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="backtest_results",
        help="Output directory for exported files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    logger = logging.getLogger("run_backtest")

    # ── Load data ─────────────────────────────────────────────────────
    loader = DataLoader(cache_dir=Path("data"))

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            sys.exit(1)
        logger.info(f"Loading data from {csv_path}")
        candles = loader.load_csv_file(csv_path)
    else:
        logger.info(f"Loading BTC/EUR data: {args.start} → {args.end}")
        candles = loader.load_daily(
            args.start, args.end, use_cache=True,
        )

    if not candles:
        logger.error("No data loaded. Use --csv to provide a local file.")
        sys.exit(1)

    logger.info(
        f"Loaded {len(candles)} daily candles: "
        f"{candles[0].date} → {candles[-1].date} "
        f"(€{candles[0].close:,.0f} → €{candles[-1].close:,.0f})"
    )

    # ── Configure backtest ────────────────────────────────────────────
    bt_config = BacktestConfig(
        starting_eur=args.capital,
        starting_btc=args.btc,
        dca_interval_days=args.dca_interval,
        dca_amount_eur=args.dca_amount,
        output_dir=Path(args.output_dir),
    )

    # ── Run backtest ──────────────────────────────────────────────────
    engine = BacktestEngine(bt_config)

    try:
        result = engine.run(candles)
    except ValueError as exc:
        logger.error(f"Backtest failed: {exc}")
        sys.exit(1)

    # ── Print results ─────────────────────────────────────────────────
    print(result.summary())

    # ── Export CSVs ───────────────────────────────────────────────────
    if args.export:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        trades_path = output / "trades.csv"
        export_trades_csv(result.trades, trades_path)
        logger.info(f"Exported {len(result.trades)} trades to {trades_path}")

        snaps_path = output / "daily_snapshots.csv"
        export_snapshots_csv(result.daily_snapshots, snaps_path)
        logger.info(
            f"Exported {len(result.daily_snapshots)} snapshots to {snaps_path}"
        )

        # Also export the summary text
        summary_path = output / "summary.txt"
        summary_path.write_text(result.summary())
        logger.info(f"Exported summary to {summary_path}")

        print(f"\nResults exported to {output}/")


if __name__ == "__main__":
    main()
