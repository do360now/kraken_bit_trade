"""
TradingStrategy - Deep Module with Strategy Pattern

OUSTERHOUT PRINCIPLE: "Different layers should have different abstractions"

This module demonstrates:
1. Strategy pattern (swap strategies without changing client code)
2. Pull complexity downward (all indicator logic hidden)
3. Information hiding (TradingBot doesn't know about RSI/MACD)
4. Simple interface, complex implementation (deep module)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol
from enum import Enum
import numpy as np


# ============================================================================
# PUBLIC INTERFACES - Protocol-Based Design
# ============================================================================

class TradingSignal(Enum):
    """Simple enum for trade signals - no complexity exposed."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass(frozen=True)
class PositionSize:
    """
    Immutable position size recommendation.
    
    Includes both amount and confidence to allow for dynamic position sizing.
    """
    btc_amount: float
    confidence: float  # 0.0 to 1.0
    reason: str
    
    def __post_init__(self):
        if self.btc_amount < 0:
            raise ValueError("Position size cannot be negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


class MarketData(Protocol):
    """
    Protocol (interface) for market data.
    
    Using Protocol instead of concrete class allows:
    1. Easy testing (any compatible object works)
    2. No coupling to specific implementation
    3. Duck typing benefits
    """
    def current_price(self): ...
    def price_history(self, hours: int): ...
    def market_regime(self): ...


class TradingStrategy(ABC):
    """
    Abstract base for trading strategies.
    
    PUBLIC INTERFACE (Simple):
        should_buy() -> bool
        should_sell() -> bool
        get_signal() -> TradingSignal
        position_size() -> PositionSize
    
    DESIGN BENEFITS:
    1. Client code (TradingBot) is decoupled from strategy details
    2. Can swap strategies at runtime
    3. Each strategy can have different complexity
    4. Easy to A/B test strategies
    5. Strategy changes don't affect TradingBot
    """
    
    @abstractmethod
    def should_buy(self) -> bool:
        """Should we buy right now? Simple yes/no."""
        pass
    
    @abstractmethod
    def should_sell(self) -> bool:
        """Should we sell right now? Simple yes/no."""
        pass
    
    @abstractmethod
    def get_signal(self) -> TradingSignal:
        """Get current trading signal with strength."""
        pass
    
    @abstractmethod
    def position_size(self, available_capital: float) -> PositionSize:
        """Calculate recommended position size."""
        pass


# ============================================================================
# CONCRETE STRATEGY - Accumulation Strategy
# ============================================================================

class AccumulationStrategy(TradingStrategy):
    """
    Bitcoin accumulation strategy focused on buying dips.
    
    DEEP MODULE CHARACTERISTICS:
    - Simple interface (inherited from TradingStrategy)
    - Complex implementation (RSI, MACD, sentiment, on-chain, etc.)
    - All indicator complexity hidden
    - Self-contained (gets its own data)
    
    CONFIGURATION:
    All parameters injected via config object (no global imports).
    This makes testing easy and reduces coupling.
    """
    
    @dataclass
    class Config:
        """
        Strategy configuration.
        
        Grouped together to avoid parameter explosion.
        Default values make it easy to use.
        """
        # RSI thresholds
        rsi_oversold: float = 30.0
        rsi_overbought: float = 70.0
        rsi_strong_oversold: float = 20.0
        
        # Position sizing
        base_position_pct: float = 0.10  # 10% of capital
        max_position_pct: float = 0.20   # 20% max
        min_position_pct: float = 0.02   # 2% min
        
        # Risk management  
        max_drawdown_threshold: float = 0.15  # 15%
        profit_target: float = 0.08  # 8% profit to consider selling
        
        # Strategy weights
        technical_weight: float = 0.4
        sentiment_weight: float = 0.3
        onchain_weight: float = 0.3
    
    def __init__(
        self,
        market_data: MarketData,
        config: Optional[Config] = None
    ):
        """
        Initialize accumulation strategy.
        
        Args:
            market_data: Market data service (injected dependency)
            config: Strategy configuration (injected, defaults provided)
        """
        self._market = market_data
        self._config = config or self.Config()
        
        # Internal state (hidden from callers)
        self._last_signal = TradingSignal.HOLD
        self._signal_confidence = 0.0
    
    # ------------------------------------------------------------------------
    # PUBLIC INTERFACE - Simple, Clean
    # ------------------------------------------------------------------------
    
    def should_buy(self) -> bool:
        """
        Should we buy right now?
        
        SIMPLE INTERFACE: Returns bool. That's it.
        
        COMPLEX IMPLEMENTATION: Considers RSI, MACD, sentiment, on-chain,
        market regime, volatility, etc. Caller doesn't need to know.
        """
        signal = self.get_signal()
        return signal in (TradingSignal.BUY, TradingSignal.STRONG_BUY)
    
    def should_sell(self) -> bool:
        """
        Should we sell right now?
        
        For accumulation strategy, we rarely sell (HODL bias).
        Only sell on:
        1. Strong overbought conditions
        2. Profit target reached
        3. Risk management (stop loss)
        """
        signal = self.get_signal()
        return signal in (TradingSignal.SELL, TradingSignal.STRONG_SELL)
    
    def get_signal(self) -> TradingSignal:
        """
        Get current trading signal.
        
        This is where all the complexity lives. Multiple indicators
        are combined into a single, simple signal.
        """
        # Gather all inputs (hidden complexity)
        technical_score = self._calculate_technical_score()
        sentiment_score = self._calculate_sentiment_score()
        onchain_score = self._calculate_onchain_score()
        
        # Weighted combination
        combined_score = (
            technical_score * self._config.technical_weight +
            sentiment_score * self._config.sentiment_weight +
            onchain_score * self._config.onchain_weight
        )
        
        # Map score to signal
        if combined_score >= 0.7:
            signal = TradingSignal.STRONG_BUY
        elif combined_score >= 0.4:
            signal = TradingSignal.BUY
        elif combined_score <= -0.7:
            signal = TradingSignal.STRONG_SELL
        elif combined_score <= -0.4:
            signal = TradingSignal.SELL
        else:
            signal = TradingSignal.HOLD
        
        # Update internal state
        self._last_signal = signal
        self._signal_confidence = abs(combined_score)
        
        return signal
    
    def position_size(self, available_capital: float) -> PositionSize:
        """
        Calculate position size based on signal strength and available capital.
        
        Returns:
            PositionSize: Recommended position with confidence and reason
        """
        # Get current signal
        signal = self.get_signal()
        
        # No position for hold/sell signals
        if signal not in (TradingSignal.BUY, TradingSignal.STRONG_BUY):
            return PositionSize(
                btc_amount=0.0,
                confidence=0.0,
                reason="No buy signal"
            )
        
        # Calculate base position
        current_price = self._market.current_price().value
        
        # Scale position by signal strength
        if signal == TradingSignal.STRONG_BUY:
            position_pct = self._config.max_position_pct
            reason = "Strong buy signal - max position"
        else:
            position_pct = self._config.base_position_pct
            reason = "Buy signal - base position"
        
        # Adjust for confidence
        adjusted_pct = position_pct * self._signal_confidence
        
        # Enforce limits
        adjusted_pct = max(self._config.min_position_pct, adjusted_pct)
        adjusted_pct = min(self._config.max_position_pct, adjusted_pct)
        
        # Calculate BTC amount
        eur_amount = available_capital * adjusted_pct
        btc_amount = eur_amount / current_price
        
        return PositionSize(
            btc_amount=btc_amount,
            confidence=self._signal_confidence,
            reason=reason
        )
    
    # ------------------------------------------------------------------------
    # PRIVATE IMPLEMENTATION - Complex, Hidden
    # ------------------------------------------------------------------------
    
    def _calculate_technical_score(self) -> float:
        """
        Calculate technical analysis score.
        
        Returns:
            float: Score from -1.0 (bearish) to +1.0 (bullish)
        
        HIDDEN COMPLEXITY:
        - RSI calculation
        - MACD calculation
        - Bollinger Bands
        - Moving averages
        - Divergence detection
        
        Caller just gets a score.
        """
        history = self._market.price_history(hours=168)  # 1 week
        prices = [p.value for p in history]
        
        if len(prices) < 50:
            return 0.0  # Not enough data
        
        # Calculate indicators
        rsi = self._calculate_rsi(prices, window=14)
        macd, signal = self._calculate_macd(prices)
        
        # Convert to score
        score = 0.0
        
        # RSI scoring
        if rsi < self._config.rsi_strong_oversold:
            score += 0.8  # Strong buy signal
        elif rsi < self._config.rsi_oversold:
            score += 0.4  # Buy signal
        elif rsi > self._config.rsi_overbought:
            score -= 0.4  # Sell signal
        
        # MACD scoring
        if macd > signal:
            score += 0.3
        else:
            score -= 0.3
        
        # Normalize to -1 to +1
        return max(-1.0, min(1.0, score))
    
    def _calculate_sentiment_score(self) -> float:
        """
        Calculate sentiment score.
        
        In real implementation, would fetch news and analyze sentiment.
        Hidden from caller.
        """
        # Simplified for example
        # Real implementation would:
        # 1. Fetch news from multiple sources
        # 2. Analyze sentiment with NLP
        # 3. Weight by source reliability
        # 4. Detect risk-off events
        
        return 0.0  # Neutral for example
    
    def _calculate_onchain_score(self) -> float:
        """
        Calculate on-chain metrics score.
        
        Hidden complexity:
        - Exchange netflow
        - UTXO age distribution
        - Mining metrics
        - Whale movements
        """
        # Simplified for example
        return 0.0  # Neutral for example
    
    def _calculate_rsi(self, prices: list, window: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < window + 1:
            return 50.0  # Neutral if not enough data
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(
        self,
        prices: list,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9
    ) -> tuple[float, float]:
        """Calculate MACD and signal line."""
        if len(prices) < slow:
            return 0.0, 0.0
        
        prices_array = np.array(prices)
        
        # Calculate EMAs
        ema_fast = self._ema(prices_array, fast)
        ema_slow = self._ema(prices_array, slow)
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal = self._ema(np.array([macd]), signal_period)
        
        return float(macd), float(signal)
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate exponential moving average."""
        if len(data) < period:
            return float(np.mean(data))
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])  # Start with SMA
        
        for value in data[period:]:
            ema = (value - ema) * multiplier + ema
        
        return float(ema)


# ============================================================================
# ALTERNATIVE STRATEGY - Shows Strategy Pattern Benefit
# ============================================================================

class MomentumStrategy(TradingStrategy):
    """
    Momentum-based strategy - different approach, same interface.
    
    This demonstrates the POWER of the strategy pattern:
    - TradingBot doesn't need to change
    - Can swap strategies at runtime
    - Can A/B test strategies
    - Each strategy has its own complexity
    """
    
    @dataclass
    class Config:
        lookback_period: int = 20
        momentum_threshold: float = 0.05
        base_position_pct: float = 0.10
    
    def __init__(self, market_data: MarketData, config: Optional[Config] = None):
        self._market = market_data
        self._config = config or self.Config()
    
    def should_buy(self) -> bool:
        """Buy on positive momentum."""
        history = self._market.price_history(hours=self._config.lookback_period)
        if len(history) < 2:
            return False
        
        momentum = (history[0].value - history[-1].value) / history[-1].value
        return momentum > self._config.momentum_threshold
    
    def should_sell(self) -> bool:
        """Sell on negative momentum."""
        history = self._market.price_history(hours=self._config.lookback_period)
        if len(history) < 2:
            return False
        
        momentum = (history[0].value - history[-1].value) / history[-1].value
        return momentum < -self._config.momentum_threshold
    
    def get_signal(self) -> TradingSignal:
        """Get signal based on momentum."""
        if self.should_buy():
            return TradingSignal.BUY
        elif self.should_sell():
            return TradingSignal.SELL
        else:
            return TradingSignal.HOLD
    
    def position_size(self, available_capital: float) -> PositionSize:
        """Fixed percentage position sizing."""
        if not self.should_buy():
            return PositionSize(0.0, 0.0, "No buy signal")
        
        current_price = self._market.current_price().value
        eur_amount = available_capital * self._config.base_position_pct
        btc_amount = eur_amount / current_price
        
        return PositionSize(btc_amount, 0.8, "Momentum buy signal")


# ============================================================================
# USAGE EXAMPLE - Notice How Clean the Client Code Is
# ============================================================================

def example_trading_bot_usage():
    """
    Demonstrate how TradingBot uses strategy.
    
    BENEFITS:
    1. No knowledge of indicators
    2. No knowledge of calculation complexity
    3. Can swap strategies easily
    4. Simple, readable code
    """
    # Setup (dependency injection)
    from market_data_service import MarketDataService
    
    market_data = MarketDataService(...)
    
    # Choose strategy (could be config-driven)
    strategy = AccumulationStrategy(
        market_data=market_data,
        config=AccumulationStrategy.Config(
            rsi_oversold=25.0,  # More aggressive
            base_position_pct=0.15
        )
    )
    
    # OR swap to different strategy - same interface!
    # strategy = MomentumStrategy(market_data=market_data)
    
    # Use strategy - simple, clean
    if strategy.should_buy():
        position = strategy.position_size(available_capital=1000.0)
        print(f"Buy {position.btc_amount:.8f} BTC")
        print(f"Confidence: {position.confidence:.2%}")
        print(f"Reason: {position.reason}")
    
    elif strategy.should_sell():
        print("Sell signal - exit position")
    
    else:
        print("Hold - no action")


# ============================================================================
# TESTING EXAMPLE - Easy to Test in Isolation
# ============================================================================

class MockMarketData:
    """Mock for testing - no real API needed."""
    
    def __init__(self, price: float = 50000.0):
        self._price = price
    
    def current_price(self):
        from market_data_service import Price
        from datetime import datetime
        return Price(self._price, datetime.now(), 100.0)
    
    def price_history(self, hours: int):
        # Return mock history with slight upward trend
        from market_data_service import Price
        from datetime import datetime, timedelta
        
        prices = []
        for i in range(hours):
            price_value = self._price - (hours - i) * 10  # Upward trend
            prices.append(Price(
                price_value,
                datetime.now() - timedelta(hours=i),
                100.0
            ))
        return prices
    
    def market_regime(self):
        from market_data_service import MarketRegime
        return MarketRegime.UPTREND


def test_accumulation_strategy():
    """Test strategy in isolation - no dependencies needed."""
    # Create mock data
    mock_market = MockMarketData(price=50000.0)
    
    # Create strategy with test config
    config = AccumulationStrategy.Config(
        rsi_oversold=30.0,
        base_position_pct=0.10
    )
    strategy = AccumulationStrategy(mock_market, config)
    
    # Test signal generation
    signal = strategy.get_signal()
    assert isinstance(signal, TradingSignal)
    print(f"✅ Signal test passed: {signal.value}")
    
    # Test position sizing
    position = strategy.position_size(available_capital=10000.0)
    assert position.btc_amount > 0
    assert 0 <= position.confidence <= 1
    print(f"✅ Position size test passed: {position.btc_amount:.8f} BTC")
    
    # Test buy/sell decisions
    should_buy = strategy.should_buy()
    should_sell = strategy.should_sell()
    assert not (should_buy and should_sell)  # Can't be both
    print(f"✅ Decision test passed: buy={should_buy}, sell={should_sell}")


if __name__ == "__main__":
    test_accumulation_strategy()
