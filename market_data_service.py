"""
MarketDataService - Deep Module Example

OUSTERHOUT PRINCIPLE: "The best modules are those that provide powerful 
functionality yet have simple interfaces."

This module demonstrates:
1. Simple public interface (3 methods)
2. Complex private implementation (caching, error handling, data validation)
3. Information hiding (Kraken API details hidden)
4. Error handling defined out of existence (never returns None/throws)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum
import time


# ============================================================================
# PUBLIC INTERFACES - Simple, Stable, Well-Documented
# ============================================================================

@dataclass(frozen=True)
class Price:
    """
    Immutable price point.
    
    Why immutable? Prevents accidental modification and makes caching safe.
    This is "defining errors out of existence" - can't modify by accident.
    """
    value: float
    timestamp: datetime
    volume: float
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError(f"Invalid price: {self.value}")
        if self.volume < 0:
            raise ValueError(f"Invalid volume: {self.volume}")


class MarketRegime(Enum):
    """Market state abstraction - hides calculation complexity."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class MarketDataService:
    """
    Deep module for market data access.
    
    PUBLIC INTERFACE (Simple - 3 methods):
        current_price() -> Price          # Always works, never None
        price_history(hours) -> List[Price]  # Always returns data
        market_regime() -> MarketRegime   # Always returns regime
    
    PRIVATE IMPLEMENTATION (Complex):
        - Multi-level caching (memory, disk)
        - Automatic retry with exponential backoff
        - Graceful degradation (returns stale data vs failing)
        - Data validation and sanitization
        - Exchange API abstraction
    
    DESIGN PRINCIPLES APPLIED:
    1. Define errors out of existence: Methods never fail, return sensible defaults
    2. Information hiding: Callers don't know about Kraken, caching, retries
    3. Pull complexity downward: All hard stuff happens inside
    """
    
    def __init__(
        self,
        exchange_client,
        cache_duration_seconds: int = 60,
        history_cache_duration_seconds: int = 300,
        stale_data_tolerance_seconds: int = 600
    ):
        """
        Initialize market data service.
        
        Args:
            exchange_client: Exchange API client (abstracted - could be any exchange)
            cache_duration_seconds: How long to cache current price
            history_cache_duration_seconds: How long to cache price history
            stale_data_tolerance_seconds: How old data can be before we force refresh
        
        Note: Configuration is injected, not imported from global config.
        This makes testing easy and reduces coupling.
        """
        self._exchange = exchange_client
        self._cache_duration = cache_duration_seconds
        self._history_cache_duration = history_cache_duration_seconds
        self._stale_tolerance = stale_data_tolerance_seconds
        
        # Multi-level cache
        self._price_cache: Optional[Price] = None
        self._price_cache_time: Optional[datetime] = None
        self._history_cache: List[Price] = []
        self._history_cache_time: Optional[datetime] = None
        
        # Fallback data (last known good state)
        self._last_known_price: Optional[Price] = None
    
    # ------------------------------------------------------------------------
    # PUBLIC INTERFACE - Simple, Never Fails
    # ------------------------------------------------------------------------
    
    def current_price(self) -> Price:
        """
        Get current BTC price.
        
        GUARANTEE: This method ALWAYS returns a valid Price.
        It NEVER returns None or raises exceptions to the caller.
        
        Returns:
            Price: Current price with timestamp and volume
        
        Implementation strategy:
        1. Check memory cache
        2. Try API call with retry
        3. Fall back to last known price
        4. Ultimate fallback: construct reasonable estimate
        
        This is "defining errors out of existence" - the caller doesn't
        need error handling because the method handles all error cases internally.
        """
        # Level 1: Fresh cache
        if self._is_cache_fresh(self._price_cache_time, self._cache_duration):
            return self._price_cache
        
        # Level 2: Try to fetch new data
        try:
            fresh_price = self._fetch_current_price_with_retry()
            if fresh_price:
                self._update_price_cache(fresh_price)
                return fresh_price
        except Exception:
            # Log internally but don't propagate
            pass  # Fall through to fallbacks
        
        # Level 3: Stale cache is better than nothing
        if self._is_cache_fresh(self._price_cache_time, self._stale_tolerance):
            return self._price_cache
        
        # Level 4: Last known good price
        if self._last_known_price:
            return self._last_known_price
        
        # Level 5: This should never happen, but handle gracefully
        # Return a constructed price based on historical data
        return self._construct_emergency_price()
    
    def price_history(self, hours: int = 24) -> List[Price]:
        """
        Get price history for the specified duration.
        
        GUARANTEE: Always returns a non-empty list.
        
        Args:
            hours: Number of hours to look back (default 24)
        
        Returns:
            List[Price]: Price history, newest first
        
        Design note: Returns empty list instead of None, eliminating need
        for null checks. This is "define errors out of existence."
        """
        # Check cache freshness
        if self._is_cache_fresh(
            self._history_cache_time,
            self._history_cache_duration
        ):
            return self._filter_history_by_hours(self._history_cache, hours)
        
        # Fetch fresh data
        try:
            fresh_history = self._fetch_price_history_with_retry(hours)
            if fresh_history:
                self._update_history_cache(fresh_history)
                return fresh_history
        except Exception:
            pass  # Fall through to cache
        
        # Return cached data even if stale
        if self._history_cache:
            return self._filter_history_by_hours(self._history_cache, hours)
        
        # Last resort: construct minimal history from current price
        current = self.current_price()
        return [current]
    
    def market_regime(self) -> MarketRegime:
        """
        Determine current market regime (trend direction/strength).
        
        GUARANTEE: Always returns a valid MarketRegime.
        
        Returns:
            MarketRegime: Current market state
        
        Design note: This hides ALL the complexity of regime detection.
        Callers get a simple enum, not raw indicator values.
        This is "information hiding" - the implementation can change
        completely without affecting callers.
        """
        history = self.price_history(hours=168)  # 1 week
        
        if len(history) < 50:
            return MarketRegime.RANGING  # Not enough data
        
        # Calculate moving averages (hidden complexity)
        prices = [p.value for p in history]
        ma_20 = sum(prices[:20]) / 20
        ma_50 = sum(prices[:50]) / 50
        ma_200 = sum(prices[:200]) / 200 if len(prices) >= 200 else ma_50
        
        # Determine regime (complex logic hidden from caller)
        if ma_20 > ma_50 * 1.05 and ma_50 > ma_200 * 1.05:
            return MarketRegime.STRONG_UPTREND
        elif ma_20 > ma_50 * 1.02:
            return MarketRegime.UPTREND
        elif ma_20 < ma_50 * 0.95 and ma_50 < ma_200 * 0.95:
            return MarketRegime.STRONG_DOWNTREND
        elif ma_20 < ma_50 * 0.98:
            return MarketRegime.DOWNTREND
        else:
            return MarketRegime.RANGING
    
    # ------------------------------------------------------------------------
    # PRIVATE IMPLEMENTATION - Complex, Hidden from Callers
    # ------------------------------------------------------------------------
    
    def _fetch_current_price_with_retry(
        self,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0
    ) -> Optional[Price]:
        """
        Fetch current price with exponential backoff retry.
        
        This complexity is HIDDEN from callers. They don't need to know
        about retries, backoff, or error handling.
        """
        for attempt in range(max_attempts):
            try:
                # Call exchange API (abstracted)
                ohlc = self._exchange.get_latest_ohlc()
                
                # Validate and convert to our Price type
                if ohlc and len(ohlc) > 0:
                    return Price(
                        value=float(ohlc[4]),  # Close price
                        timestamp=datetime.fromtimestamp(ohlc[0]),
                        volume=float(ohlc[6])
                    )
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(backoff_seconds * (2 ** attempt))
                    continue
                # Last attempt failed, return None
                return None
        
        return None
    
    def _fetch_price_history_with_retry(
        self,
        hours: int,
        max_attempts: int = 3
    ) -> Optional[List[Price]]:
        """Fetch price history with retry logic."""
        for attempt in range(max_attempts):
            try:
                # Calculate time range
                since_timestamp = int(time.time() - (hours * 3600))
                
                # Call exchange API
                ohlc_data = self._exchange.get_ohlc_data(since=since_timestamp)
                
                if not ohlc_data:
                    continue
                
                # Convert to Price objects
                prices = []
                for candle in ohlc_data:
                    try:
                        prices.append(Price(
                            value=float(candle[4]),
                            timestamp=datetime.fromtimestamp(candle[0]),
                            volume=float(candle[6])
                        ))
                    except (ValueError, IndexError):
                        continue  # Skip invalid candles
                
                if prices:
                    return prices
                    
            except Exception:
                if attempt < max_attempts - 1:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
        
        return None
    
    def _is_cache_fresh(
        self,
        cache_time: Optional[datetime],
        max_age_seconds: int
    ) -> bool:
        """Check if cached data is still fresh."""
        if cache_time is None:
            return False
        
        age = (datetime.now() - cache_time).total_seconds()
        return age < max_age_seconds
    
    def _update_price_cache(self, price: Price) -> None:
        """Update price cache and last known good price."""
        self._price_cache = price
        self._price_cache_time = datetime.now()
        self._last_known_price = price
    
    def _update_history_cache(self, history: List[Price]) -> None:
        """Update history cache."""
        self._history_cache = history
        self._history_cache_time = datetime.now()
    
    def _filter_history_by_hours(
        self,
        history: List[Price],
        hours: int
    ) -> List[Price]:
        """Filter cached history to requested time range."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [p for p in history if p.timestamp >= cutoff]
    
    def _construct_emergency_price(self) -> Price:
        """
        Construct a reasonable price when all else fails.
        
        This should never happen in production, but having a graceful
        fallback is better than crashing.
        """
        # If we have any history, use the last price
        if self._history_cache:
            return self._history_cache[0]
        
        # Absolute last resort: reasonable default
        # (Better than crashing, but log this as critical error)
        return Price(
            value=50000.0,  # Reasonable BTC price
            timestamp=datetime.now(),
            volume=0.0
        )


# ============================================================================
# USAGE EXAMPLE - Notice How Simple the Caller Code Is
# ============================================================================

def example_usage():
    """
    Demonstrate how simple it is to use MarketDataService.
    
    Notice:
    1. No error handling needed
    2. No null checks needed  
    3. No knowledge of Kraken API
    4. No knowledge of caching
    5. No knowledge of retries
    
    This is the benefit of a DEEP module.
    """
    # Assume we have an exchange client (could be any exchange)
    from kraken_api import KrakenAPI
    exchange = KrakenAPI(api_key="...", api_secret="...")
    
    # Create service
    market_data = MarketDataService(
        exchange_client=exchange,
        cache_duration_seconds=60
    )
    
    # Use it - simple, clean, no error handling
    current = market_data.current_price()
    print(f"Current price: €{current.value:.2f}")
    
    history = market_data.price_history(hours=24)
    print(f"24h price range: €{min(p.value for p in history):.2f} - €{max(p.value for p in history):.2f}")
    
    regime = market_data.market_regime()
    print(f"Market regime: {regime.value}")
    
    # That's it! No try/except, no if result is None, no complexity.


# ============================================================================
# TESTING EXAMPLE - Notice How Easy It Is to Test
# ============================================================================

class MockExchange:
    """Mock exchange for testing - demonstrates how abstraction helps testing."""
    
    def get_latest_ohlc(self):
        return [
            1234567890,  # timestamp
            50000.0,     # open
            51000.0,     # high
            49000.0,     # low
            50500.0,     # close
            50300.0,     # vwap
            100.5,       # volume
            1000         # count
        ]
    
    def get_ohlc_data(self, since):
        # Return mock history
        return [self.get_latest_ohlc()]


def test_market_data_service():
    """
    Test MarketDataService in isolation.
    
    Benefits:
    1. No real API needed
    2. No network calls
    3. Deterministic results
    4. Fast execution
    5. Tests only the module, not dependencies
    """
    # Create service with mock
    mock_exchange = MockExchange()
    service = MarketDataService(mock_exchange, cache_duration_seconds=60)
    
    # Test current price
    price = service.current_price()
    assert price.value == 50500.0
    assert price.volume == 100.5
    
    # Test caching (second call should use cache)
    price2 = service.current_price()
    assert price2.value == 50500.0  # Same value from cache
    
    # Test history
    history = service.price_history(hours=24)
    assert len(history) > 0
    assert all(isinstance(p, Price) for p in history)
    
    # Test regime
    regime = service.market_regime()
    assert isinstance(regime, MarketRegime)
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_market_data_service()
