"""
Kraken exchange API client.

Production-grade with circuit breaker, exponential backoff retries,
monotonic nonce synchronization, and typed dataclass responses.
Callers never see raw API dicts or handle HTTP errors.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import KrakenConfig

logger = logging.getLogger(__name__)

# ─── Response dataclasses ─────────────────────────────────────────────────────

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Ticker:
    """Current market ticker data."""
    pair: str
    ask: float          # Best ask price
    bid: float          # Best bid price
    last: float         # Last trade price
    volume_24h: float   # 24h volume in BTC
    vwap_24h: float     # 24h VWAP
    high_24h: float     # 24h high
    low_24h: float      # 24h low
    timestamp: float    # Unix timestamp when fetched

    @property
    def spread(self) -> float:
        """Absolute spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = (self.ask + self.bid) / 2.0
        if mid <= 0:
            return 0.0
        return self.spread / mid

    @property
    def mid(self) -> float:
        """Mid-market price."""
        return (self.ask + self.bid) / 2.0


@dataclass(frozen=True)
class OHLCCandle:
    """Single OHLC candlestick."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    count: int  # Number of trades


@dataclass(frozen=True)
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    volume: float
    timestamp: float


@dataclass(frozen=True)
class OrderBook:
    """Order book snapshot."""
    pair: str
    bids: list[OrderBookLevel]  # Sorted descending by price (best bid first)
    asks: list[OrderBookLevel]  # Sorted ascending by price (best ask first)
    timestamp: float

    @property
    def best_bid(self) -> float:
        """Best bid price, or 0.0 if empty."""
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Best ask price, or 0.0 if empty."""
        return self.asks[0].price if self.asks else 0.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def mid(self) -> float:
        return (self.best_ask + self.best_bid) / 2.0

    def bid_volume(self, depth: int = 5) -> float:
        """Total volume in top N bid levels."""
        return sum(b.volume for b in self.bids[:depth])

    def ask_volume(self, depth: int = 5) -> float:
        """Total volume in top N ask levels."""
        return sum(a.volume for a in self.asks[:depth])

    def imbalance(self, depth: int = 5) -> float:
        """
        Bid/ask volume imbalance.

        Returns value in [-1, 1]. Positive = more bid volume (bullish).
        """
        bv = self.bid_volume(depth)
        av = self.ask_volume(depth)
        total = bv + av
        if total <= 0:
            return 0.0
        return (bv - av) / total


@dataclass(frozen=True)
class Balance:
    """Account balance snapshot."""
    eur: float
    btc: float
    timestamp: float


@dataclass(frozen=True)
class OrderResult:
    """Result of placing or querying an order."""
    success: bool
    txid: Optional[str] = None
    status: OrderStatus = OrderStatus.UNKNOWN
    filled_volume: float = 0.0
    filled_price: float = 0.0  # Average fill price
    fee: float = 0.0
    error: str = ""


# ─── Exceptions (internal only — never leak past public interface) ────────────

class KrakenAPIError(Exception):
    """Kraken returned an error response."""
    pass


class CircuitBreakerOpen(Exception):
    """Circuit breaker is tripped — API calls temporarily blocked."""
    pass


class NonRetryableError(Exception):
    """Error that should not be retried (auth failure, invalid params, etc.)."""
    pass


# ─── Circuit Breaker ─────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Simple circuit breaker: trips after N consecutive failures,
    resets after a cooldown period.
    """

    def __init__(self, threshold: int, cooldown: float) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._consecutive_failures = 0
        self._tripped_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is currently open (blocking calls)."""
        if self._tripped_at is None:
            return False
        elapsed = time.monotonic() - self._tripped_at
        if elapsed >= self._cooldown:
            # Cooldown expired, allow a probe
            self._reset()
            return False
        return True

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_success(self) -> None:
        """Record a successful call — resets failure counter."""
        self._reset()

    def record_failure(self) -> None:
        """Record a failed call — may trip the breaker."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self._tripped_at = time.monotonic()
            logger.warning(
                f"Circuit breaker TRIPPED after {self._consecutive_failures} "
                f"consecutive failures. Cooling down for {self._cooldown}s"
            )

    def _reset(self) -> None:
        self._consecutive_failures = 0
        self._tripped_at = None


# ─── Nonce Generator ─────────────────────────────────────────────────────────

class NonceGenerator:
    """
    Monotonic nonce generator for Kraken's API authentication.

    Uses microsecond timestamps, guaranteeing strict monotonic increase
    even under rapid successive calls. Thread-safe via a lock — critical
    when the bot's fast and slow loops run concurrently.
    """

    def __init__(self) -> None:
        self._last_nonce = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        """Generate the next nonce value (thread-safe)."""
        with self._lock:
            nonce = int(time.time() * 1_000_000)
            if nonce <= self._last_nonce:
                nonce = self._last_nonce + 1
            self._last_nonce = nonce
            return nonce


# ─── Kraken API Client ───────────────────────────────────────────────────────

class KrakenAPI:
    """
    Production Kraken REST API client.

    Features:
    - Circuit breaker pattern (trips after consecutive failures)
    - Exponential backoff retries via tenacity
    - Monotonic nonce synchronization
    - Typed dataclass responses — callers never see raw dicts
    - Rate limiting via minimum call spacing
    """

    def __init__(self, config: KrakenConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "KrakenBot/1.0"})
        self._nonce = NonceGenerator()
        self._circuit = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            cooldown=config.circuit_breaker_cooldown,
        )
        self._last_call_time = 0.0

    # ─── Public market data (no auth required) ───────────────────────────

    def get_ticker(self, pair: Optional[str] = None) -> Optional[Ticker]:
        """Fetch current ticker for the configured pair."""
        pair = pair or self._config.pair
        data = self._public_request("Ticker", {"pair": pair})
        if data is None:
            return None

        try:
            # Kraken returns data keyed by pair name (sometimes different from input)
            pair_key = next(iter(data))
            t = data[pair_key]
            return Ticker(
                pair=pair,
                ask=float(t["a"][0]),
                bid=float(t["b"][0]),
                last=float(t["c"][0]),
                volume_24h=float(t["v"][1]),
                vwap_24h=float(t["p"][1]),
                high_24h=float(t["h"][1]),
                low_24h=float(t["l"][1]),
                timestamp=time.time(),
            )
        except (KeyError, IndexError, ValueError, StopIteration) as exc:
            logger.error(f"Failed to parse ticker response: {exc}")
            return None

    def get_ohlc(
        self,
        interval: int = 60,
        since: Optional[int] = None,
        pair: Optional[str] = None,
    ) -> list[OHLCCandle]:
        """
        Fetch OHLC candles.

        Args:
            interval: Candle interval in minutes (1, 5, 15, 60, 240, 1440).
            since: Unix timestamp to fetch candles after.
            pair: Trading pair (defaults to configured pair).

        Returns:
            List of OHLCCandle, oldest first. Empty list on failure.
        """
        pair = pair or self._config.pair
        params: dict = {"pair": pair, "interval": interval}
        if since is not None:
            params["since"] = since

        data = self._public_request("OHLC", params)
        if data is None:
            return []

        try:
            pair_key = [k for k in data if k != "last"][0]
            candles = []
            for row in data[pair_key]:
                candles.append(OHLCCandle(
                    timestamp=float(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    vwap=float(row[5]),
                    volume=float(row[6]),
                    count=int(row[7]),
                ))
            return candles
        except (KeyError, IndexError, ValueError) as exc:
            logger.error(f"Failed to parse OHLC response: {exc}")
            return []

    def get_order_book(
        self, depth: int = 20, pair: Optional[str] = None
    ) -> Optional[OrderBook]:
        """Fetch order book with specified depth."""
        pair = pair or self._config.pair
        data = self._public_request("Depth", {"pair": pair, "count": depth})
        if data is None:
            return None

        try:
            pair_key = next(iter(data))
            book = data[pair_key]
            bids = [
                OrderBookLevel(
                    price=float(b[0]), volume=float(b[1]), timestamp=float(b[2])
                )
                for b in book["bids"]
            ]
            asks = [
                OrderBookLevel(
                    price=float(a[0]), volume=float(a[1]), timestamp=float(a[2])
                )
                for a in book["asks"]
            ]
            return OrderBook(
                pair=pair, bids=bids, asks=asks, timestamp=time.time()
            )
        except (KeyError, IndexError, ValueError, StopIteration) as exc:
            logger.error(f"Failed to parse order book response: {exc}")
            return None

    # ─── Private authenticated endpoints ─────────────────────────────────

    def get_balance(self) -> Optional[Balance]:
        """Fetch account balances (EUR and BTC)."""
        data = self._private_request("Balance")
        if data is None:
            return None

        try:
            # Kraken balance keys: ZEUR for EUR, XXBT for BTC
            eur = float(data.get("ZEUR", data.get("EUR", 0.0)))
            btc = float(data.get("XXBT", data.get("XBT", 0.0)))
            return Balance(eur=eur, btc=btc, timestamp=time.time())
        except (ValueError, TypeError) as exc:
            logger.error(f"Failed to parse balance response: {exc}")
            return None

    def place_order(
        self,
        side: OrderSide,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        pair: Optional[str] = None,
    ) -> OrderResult:
        """
        Place an order on Kraken.

        For limit orders, price is required.
        For market orders, price is ignored (but we should never use market orders).
        """
        pair = pair or self._config.pair

        if order_type == OrderType.LIMIT and price is None:
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                error="Limit order requires a price",
            )

        if volume < self._config.min_order_btc:
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                error=f"Volume {volume} below minimum {self._config.min_order_btc}",
            )

        params: dict = {
            "pair": pair,
            "type": side.value,
            "ordertype": order_type.value,
            "volume": f"{volume:.8f}",
        }
        if price is not None:
            params["price"] = f"{price:.1f}"

        data = self._private_request("AddOrder", params)
        if data is None:
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                error="API request failed",
            )

        try:
            txids = data.get("txid", [])
            txid = txids[0] if txids else None
            logger.info(
                f"Order placed: {side.value} {volume:.8f} BTC "
                f"@ {'market' if price is None else f'€{price:,.1f}'} "
                f"→ txid={txid}"
            )
            return OrderResult(
                success=True,
                txid=txid,
                status=OrderStatus.PENDING,
            )
        except (IndexError, KeyError) as exc:
            logger.error(f"Failed to parse order response: {exc}")
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                error=f"Parse error: {exc}",
            )

    def query_order(self, txid: str) -> OrderResult:
        """Query the status of an existing order."""
        data = self._private_request("QueryOrders", {"txid": txid})
        if data is None:
            return OrderResult(
                success=False,
                txid=txid,
                status=OrderStatus.UNKNOWN,
                error="API request failed",
            )

        try:
            order = data.get(txid, {})
            raw_status = order.get("status", "unknown")
            status = self._parse_order_status(raw_status)

            vol_exec = float(order.get("vol_exec", 0.0))
            avg_price = float(order.get("price", 0.0))
            fee = float(order.get("fee", 0.0))

            return OrderResult(
                success=True,
                txid=txid,
                status=status,
                filled_volume=vol_exec,
                filled_price=avg_price,
                fee=fee,
            )
        except (ValueError, KeyError) as exc:
            logger.error(f"Failed to parse order query response: {exc}")
            return OrderResult(
                success=False,
                txid=txid,
                status=OrderStatus.UNKNOWN,
                error=f"Parse error: {exc}",
            )

    def cancel_order(self, txid: str) -> OrderResult:
        """Cancel an open order."""
        data = self._private_request("CancelOrder", {"txid": txid})
        if data is None:
            return OrderResult(
                success=False,
                txid=txid,
                status=OrderStatus.UNKNOWN,
                error="API request failed",
            )

        logger.info(f"Order cancelled: txid={txid}")
        return OrderResult(success=True, txid=txid, status=OrderStatus.CANCELLED)

    # ─── Internal HTTP mechanics ─────────────────────────────────────────

    def _public_request(self, method: str, params: Optional[dict] = None) -> Optional[dict]:
        """Execute a public API request with circuit breaker and retries."""
        url = f"{self._config.base_url}/0/public/{method}"
        try:
            self._check_circuit_breaker()
            self._rate_limit()
            result = self._execute_request_with_retry(url, params or {}, authenticated=False)
            self._circuit.record_success()
            return result
        except CircuitBreakerOpen:
            logger.warning(f"Circuit breaker open — skipping {method}")
            return None
        except NonRetryableError as exc:
            logger.error(f"Non-retryable error in {method}: {exc}")
            return None
        except Exception as exc:
            self._circuit.record_failure()
            logger.error(f"Public request {method} failed after retries: {exc}")
            return None

    def _private_request(self, method: str, params: Optional[dict] = None) -> Optional[dict]:
        """Execute an authenticated private API request."""
        if not self._config.api_key or not self._config.private_key:
            logger.error("Cannot make private request: API credentials not configured")
            return None

        url_path = f"/0/private/{method}"
        url = f"{self._config.base_url}{url_path}"
        post_data = params or {}

        try:
            self._check_circuit_breaker()
            self._rate_limit()

            nonce = self._nonce.next()
            post_data["nonce"] = nonce
            headers = self._sign_request(url_path, post_data, nonce)

            result = self._execute_request_with_retry(
                url, post_data, authenticated=True, extra_headers=headers
            )
            self._circuit.record_success()
            return result
        except CircuitBreakerOpen:
            logger.warning(f"Circuit breaker open — skipping {method}")
            return None
        except NonRetryableError as exc:
            logger.error(f"Non-retryable error in {method}: {exc}")
            return None
        except Exception as exc:
            self._circuit.record_failure()
            logger.error(f"Private request {method} failed after retries: {exc}")
            return None

    @retry(
        retry=retry_if_exception_type(KrakenAPIError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _execute_request_with_retry(
        self,
        url: str,
        data: dict,
        authenticated: bool,
        extra_headers: Optional[dict] = None,
    ) -> dict:
        """
        Execute HTTP request with tenacity retry logic.

        Retries on transient KrakenAPIError. Raises NonRetryableError
        for auth failures and invalid parameters.
        """
        try:
            headers = extra_headers or {}
            response = self._session.post(url, data=data, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise KrakenAPIError("Request timed out")
        except requests.exceptions.ConnectionError as exc:
            raise KrakenAPIError(f"Connection error: {exc}")
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            if status in (401, 403):
                raise NonRetryableError(f"Authentication failed: {status}")
            if status == 429:
                raise KrakenAPIError("Rate limited")
            raise KrakenAPIError(f"HTTP {status}: {exc}")

        try:
            body = response.json()
        except ValueError:
            raise KrakenAPIError("Invalid JSON response")

        errors = body.get("error", [])
        if errors:
            error_str = "; ".join(str(e) for e in errors)
            # Classify errors
            if any("EAPI:Invalid nonce" in str(e) for e in errors):
                # Nonce errors are retryable (our generator will produce a higher one)
                raise KrakenAPIError(f"Nonce error: {error_str}")
            if any("EGeneral:Internal error" in str(e) for e in errors):
                raise KrakenAPIError(f"Kraken internal error: {error_str}")
            if any("EAPI:Rate limit" in str(e) for e in errors):
                raise KrakenAPIError(f"Rate limited: {error_str}")
            if any("EAPI:Invalid key" in str(e) for e in errors):
                raise NonRetryableError(f"Invalid API key: {error_str}")
            if any("EOrder:" in str(e) for e in errors):
                raise NonRetryableError(f"Order error: {error_str}")
            # Default: treat unknown errors as retryable
            raise KrakenAPIError(f"API error: {error_str}")

        return body.get("result", {})

    def _sign_request(self, url_path: str, data: dict, nonce: int) -> dict:
        """Generate Kraken API signature headers."""
        post_data = urllib.parse.urlencode(data)
        encoded = (str(nonce) + post_data).encode("utf-8")
        message = url_path.encode("utf-8") + hashlib.sha256(encoded).digest()
        secret = base64.b64decode(self._config.private_key)
        signature = hmac.new(secret, message, hashlib.sha512)
        sig_b64 = base64.b64encode(signature.digest()).decode("utf-8")

        return {
            "API-Key": self._config.api_key,
            "API-Sign": sig_b64,
        }

    def _check_circuit_breaker(self) -> None:
        """Raise if circuit breaker is open."""
        if self._circuit.is_open:
            raise CircuitBreakerOpen(
                f"Circuit breaker open — {self._circuit.consecutive_failures} "
                f"consecutive failures"
            )

    def _rate_limit(self) -> None:
        """Enforce minimum spacing between API calls."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < self._config.call_spacing_seconds:
            sleep_time = self._config.call_spacing_seconds - elapsed
            time.sleep(sleep_time)
        self._last_call_time = time.monotonic()

    @staticmethod
    def _parse_order_status(raw: str) -> OrderStatus:
        """Map Kraken's order status strings to our enum."""
        mapping = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
        }
        return mapping.get(raw, OrderStatus.UNKNOWN)

    def close(self) -> None:
        """Clean up the HTTP session."""
        self._session.close()
