"""
Historical OHLCV data loader for backtesting.

Fetches BTC/EUR candles from Kraken's public API, caches locally as CSV.
No API key required — uses only public endpoints.

Usage:
    loader = DataLoader(cache_dir=Path("data/"))
    candles = loader.load_daily("2020-01-01", "2024-12-31")
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class HistoricalCandle:
    """Single OHLCV candle for backtesting."""
    timestamp: float
    date: str            # YYYY-MM-DD for readability
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    count: int           # Number of trades


class DataLoader:
    """
    Fetch and cache historical BTC/EUR OHLCV data from Kraken.

    Kraken's public OHLC endpoint returns at most 720 candles per call.
    For daily candles, that's ~2 years per request. The loader
    automatically paginates for longer date ranges.

    Args:
        cache_dir: Directory for CSV cache files.
        pair: Kraken pair string (default XXBTZEUR).
    """

    BASE_URL = "https://api.kraken.com/0/public/OHLC"

    def __init__(
        self,
        cache_dir: Path = Path("data"),
        pair: str = "XXBTZEUR",
    ) -> None:
        self._cache_dir = cache_dir
        self._pair = pair
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load_daily(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> list[HistoricalCandle]:
        """
        Load daily OHLCV candles for a date range.

        Args:
            start_date: "YYYY-MM-DD" format.
            end_date: "YYYY-MM-DD" format.
            use_cache: If True, check cache before fetching.

        Returns:
            List of HistoricalCandle sorted oldest-first.
        """
        cache_path = self._cache_path(start_date, end_date, interval=1440)

        if use_cache and cache_path.exists():
            candles = self._load_csv(cache_path)
            logger.info(f"Loaded {len(candles)} cached candles from {cache_path}")
            return candles

        candles = self._fetch_range(start_date, end_date, interval=1440)

        if candles:
            self._save_csv(candles, cache_path)
            logger.info(f"Saved {len(candles)} candles to {cache_path}")

        return candles

    def load_csv_file(self, csv_path: Path) -> list[HistoricalCandle]:
        """Load candles from an existing CSV file (user-provided data)."""
        return self._load_csv(csv_path)

    # ─── Fetching ────────────────────────────────────────────────────────

    def _fetch_range(
        self,
        start_date: str,
        end_date: str,
        interval: int = 1440,
    ) -> list[HistoricalCandle]:
        """
        Fetch candles from Kraken, paginating if needed.

        Kraken returns max 720 candles per request. For daily candles
        that's ~2 years. We paginate using the 'since' parameter.
        """
        start_ts = self._date_to_timestamp(start_date)
        end_ts = self._date_to_timestamp(end_date) + 86400  # Include end date

        all_candles: list[HistoricalCandle] = []
        since = start_ts

        while since < end_ts:
            logger.info(
                f"Fetching {self._pair} from "
                f"{datetime.fromtimestamp(since, tz=timezone.utc).date()}"
            )

            batch = self._fetch_ohlc(since=since, interval=interval)
            if not batch:
                logger.warning("No more data returned from Kraken")
                break

            # Filter to our date range
            for candle in batch:
                if start_ts <= candle.timestamp < end_ts:
                    all_candles.append(candle)

            # Advance 'since' to last candle timestamp
            last_ts = batch[-1].timestamp
            if last_ts <= since:
                break  # No progress, avoid infinite loop
            since = int(last_ts)

            # Respect rate limits
            time.sleep(2.0)

        # Deduplicate by timestamp
        seen = set()
        unique = []
        for c in all_candles:
            key = int(c.timestamp)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda c: c.timestamp)
        logger.info(f"Fetched {len(unique)} unique daily candles")
        return unique

    def _fetch_ohlc(
        self, since: int, interval: int = 1440,
    ) -> list[HistoricalCandle]:
        """Single Kraken OHLC API call."""
        params = {
            "pair": self._pair,
            "interval": interval,
            "since": since,
        }

        try:
            resp = requests.get(
                self.BASE_URL, params=params, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("error"):
                logger.error(f"Kraken API error: {data['error']}")
                return []

            # Result is keyed by pair name (which varies)
            result = data.get("result", {})
            pair_key = [k for k in result if k != "last"]
            if not pair_key:
                return []

            raw_candles = result[pair_key[0]]
            candles = []
            for row in raw_candles:
                ts = float(row[0])
                candles.append(HistoricalCandle(
                    timestamp=ts,
                    date=datetime.fromtimestamp(
                        ts, tz=timezone.utc,
                    ).strftime("%Y-%m-%d"),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    vwap=float(row[5]),
                    volume=float(row[6]),
                    count=int(row[7]),
                ))

            return candles

        except requests.RequestException as exc:
            logger.error(f"Kraken fetch failed: {exc}")
            return []

    # ─── CSV caching ─────────────────────────────────────────────────────

    def _cache_path(
        self, start: str, end: str, interval: int,
    ) -> Path:
        return self._cache_dir / f"{self._pair}_{start}_{end}_{interval}m.csv"

    def _save_csv(
        self, candles: list[HistoricalCandle], path: Path,
    ) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "date", "open", "high", "low",
                "close", "vwap", "volume", "count",
            ])
            for c in candles:
                writer.writerow([
                    c.timestamp, c.date, c.open, c.high,
                    c.low, c.close, c.vwap, c.volume, c.count,
                ])

    def _load_csv(self, path: Path) -> list[HistoricalCandle]:
        candles = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append(HistoricalCandle(
                    timestamp=float(row["timestamp"]),
                    date=row["date"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    vwap=float(row["vwap"]),
                    volume=float(row["volume"]),
                    count=int(row["count"]),
                ))
        candles.sort(key=lambda c: c.timestamp)
        return candles

    @staticmethod
    def _date_to_timestamp(date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc,
        )
        return int(dt.timestamp())
