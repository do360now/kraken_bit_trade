"""
Simplified Strategy Manager - Performance Optimized
Addresses the complexity and performance issues in the original strategies.py
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class TradingAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class StrategyType(Enum):
    DCA = "dca"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BALANCED = "balanced"

@dataclass
class SimpleSignal:
    """Simplified trading signal"""
    action: TradingAction
    confidence: float
    reasoning: str
    strategy: str
    volume_pct: float = 0.0
    priority: int = 0  # Higher number = higher priority

@dataclass
class MarketConditions:
    """Simplified market conditions"""
    price: float
    rsi: float = 50.0
    trend: str = "neutral"  # "up", "down", "neutral"
    volatility: str = "normal"  # "low", "normal", "high"
    volume_surge: bool = False

class SimpleStrategy:
    """Base class for simplified strategies"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.enabled = True
        self.last_signal_time = 0
        self.cooldown_seconds = 300  # 5 minutes
        
    def can_signal(self) -> bool:
        """Check if strategy can generate signal (cooldown)"""
        return self.enabled and (time.time() - self.last_signal_time) > self.cooldown_seconds
    
    def generate_signal(self, conditions: MarketConditions) -> SimpleSignal:
        """Generate trading signal - to be overridden"""
        if not self.can_signal():
            return SimpleSignal(TradingAction.HOLD, 0.0, "Cooldown active", self.name)
        
        return SimpleSignal(TradingAction.HOLD, 0.0, "No signal", self.name)
    
    def update_signal_time(self):
        """Update last signal time"""
        self.last_signal_time = time.time()

class DCAStrategy(SimpleStrategy):
    """Simple Dollar Cost Averaging"""
    
    def __init__(self, interval_hours: int = 48):
        super().__init__("DCA", weight=0.3)
        self.interval_hours = interval_hours
        self.last_buy_time = 0
        
    def generate_signal(self, conditions: MarketConditions) -> SimpleSignal:
        if not self.can_signal():
            return SimpleSignal(TradingAction.HOLD, 0.0, "Cooldown active", self.name)
        
        # Simple time-based DCA
        hours_since_last = (time.time() - self.last_buy_time) / 3600
        
        if hours_since_last >= self.interval_hours:
            confidence = 0.7
            
            # Boost confidence in oversold conditions
            if conditions.rsi < 35:
                confidence += 0.2
            elif conditions.rsi > 70:
                confidence -= 0.3
            
            if confidence > 0.4:
                self.last_buy_time = time.time()
                self.update_signal_time()
                return SimpleSignal(
                    TradingAction.BUY, 
                    min(0.9, confidence),
                    f"DCA buy after {hours_since_last:.1f}h",
                    self.name,
                    volume_pct=0.02,  # 2% of portfolio
                    priority=2
                )
        
        return SimpleSignal(TradingAction.HOLD, 0.0, "DCA timing not met", self.name)

class MomentumStrategy(SimpleStrategy):
    """Simple momentum strategy"""
    
    def __init__(self):
        super().__init__("Momentum", weight=0.4)
        self.price_history = deque(maxlen=20)
        
    def generate_signal(self, conditions: MarketConditions) -> SimpleSignal:
        if not self.can_signal():
            return SimpleSignal(TradingAction.HOLD, 0.0, "Cooldown active", self.name)
        
        self.price_history.append(conditions.price)
        
        if len(self.price_history) < 10:
            return SimpleSignal(TradingAction.HOLD, 0.0, "Insufficient data", self.name)
        
        # Calculate simple momentum
        recent_avg = np.mean(list(self.price_history)[-5:])
        older_avg = np.mean(list(self.price_history)[-15:-5])
        
        if older_avg == 0:
            return SimpleSignal(TradingAction.HOLD, 0.0, "Invalid price data", self.name)
        
        momentum = (recent_avg - older_avg) / older_avg
        
        # Strong momentum signals
        if momentum > 0.02 and conditions.trend == "up":  # 2% momentum + uptrend
            confidence = min(0.9, 0.6 + abs(momentum) * 10)
            
            # Volume confirmation
            if conditions.volume_surge:
                confidence += 0.1
            
            # Avoid buying if RSI too high
            if conditions.rsi > 75:
                confidence *= 0.5
            
            if confidence > 0.6:
                self.update_signal_time()
                return SimpleSignal(
                    TradingAction.BUY,
                    confidence,
                    f"Momentum: {momentum:.1%}, trend: {conditions.trend}",
                    self.name,
                    volume_pct=0.03,
                    priority=3
                )
        
        elif momentum < -0.02 and conditions.trend == "down":  # Bearish momentum
            confidence = min(0.9, 0.6 + abs(momentum) * 10)
            
            if conditions.rsi < 25:  # Don't sell if extremely oversold
                confidence *= 0.3
            
            if confidence > 0.6:
                self.update_signal_time()
                return SimpleSignal(
                    TradingAction.SELL,
                    confidence,
                    f"Bearish momentum: {momentum:.1%}",
                    self.name,
                    volume_pct=0.03,
                    priority=3
                )
        
        return SimpleSignal(TradingAction.HOLD, 0.0, f"Momentum {momentum:.1%} not strong enough", self.name)

class MeanReversionStrategy(SimpleStrategy):
    """Simple mean reversion"""
    
    def __init__(self):
        super().__init__("MeanReversion", weight=0.3)
        
    def generate_signal(self, conditions: MarketConditions) -> SimpleSignal:
        if not self.can_signal():
            return SimpleSignal(TradingAction.HOLD, 0.0, "Cooldown active", self.name)
        
        # RSI-based mean reversion
        if conditions.rsi < 25:  # Extremely oversold
            confidence = 0.8
            
            # Boost if in ranging market
            if conditions.trend == "neutral":
                confidence += 0.1
                
            self.update_signal_time()
            return SimpleSignal(
                TradingAction.BUY,
                min(0.9, confidence),
                f"Extreme oversold RSI: {conditions.rsi:.1f}",
                self.name,
                volume_pct=0.025,
                priority=4  # High priority for extreme conditions
            )
        
        elif conditions.rsi > 80:  # Extremely overbought
            confidence = 0.7
            
            if conditions.trend == "neutral":
                confidence += 0.1
                
            self.update_signal_time()
            return SimpleSignal(
                TradingAction.SELL,
                min(0.9, confidence),
                f"Extreme overbought RSI: {conditions.rsi:.1f}",
                self.name,
                volume_pct=0.025,
                priority=4
            )
        
        return SimpleSignal(TradingAction.HOLD, 0.0, f"RSI {conditions.rsi:.1f} in normal range", self.name)

class BalancedStrategy(SimpleStrategy):
    """Balanced portfolio rebalancing strategy"""
    
    def __init__(self, target_btc_pct: float = 0.7):
        super().__init__("Balanced", weight=0.5)
        self.target_btc_pct = target_btc_pct
        self.cooldown_seconds = 3600  # 1 hour cooldown for rebalancing
        
    def generate_signal(self, conditions: MarketConditions, 
                       btc_balance: float = 0, eur_balance: float = 0) -> SimpleSignal:
        if not self.can_signal():
            return SimpleSignal(TradingAction.HOLD, 0.0, "Cooldown active", self.name)
        
        # Calculate current allocation
        total_value = eur_balance + (btc_balance * conditions.price)
        if total_value <= 0:
            return SimpleSignal(TradingAction.HOLD, 0.0, "No portfolio value", self.name)
        
        btc_value = btc_balance * conditions.price
        current_btc_pct = btc_value / total_value
        
        deviation = current_btc_pct - self.target_btc_pct
        
        # Rebalance if deviation > 10%
        if abs(deviation) > 0.1:
            if deviation > 0.1:  # Too much BTC
                confidence = min(0.9, 0.6 + abs(deviation))
                self.update_signal_time()
                return SimpleSignal(
                    TradingAction.SELL,
                    confidence,
                    f"Rebalance: {current_btc_pct:.1%} BTC, target {self.target_btc_pct:.1%}",
                    self.name,
                    volume_pct=abs(deviation) * 0.5,  # Rebalance half the deviation
                    priority=5  # Highest priority for portfolio balance
                )
            
            elif deviation < -0.1:  # Too little BTC
                confidence = min(0.9, 0.6 + abs(deviation))
                self.update_signal_time()
                return SimpleSignal(
                    TradingAction.BUY,
                    confidence,
                    f"Rebalance: {current_btc_pct:.1%} BTC, target {self.target_btc_pct:.1%}",
                    self.name,
                    volume_pct=abs(deviation) * 0.5,
                    priority=5
                )
        
        return SimpleSignal(TradingAction.HOLD, 0.0, f"Portfolio balanced: {current_btc_pct:.1%} BTC", self.name)

class SimplifiedStrategyManager:
    """High-performance simplified strategy manager"""
    
    def __init__(self):
        # Initialize only essential strategies
        self.strategies = {
            'dca': DCAStrategy(interval_hours=48),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'balanced': BalancedStrategy(target_btc_pct=0.65)  # Conservative 65% BTC target
        }
        
        # Performance tracking
        self.total_signals = 0
        self.successful_trades = 0
        self.last_analysis_time = 0
        self.analysis_cache = {}
        self.cache_duration = 60  # 1 minute cache
        
        # Emergency controls
        self.emergency_stop = False
        self.max_signals_per_hour = 8
        self.signal_history = deque(maxlen=50)
        
        logger.info(f"Simplified Strategy Manager initialized with {len(self.strategies)} strategies")
    
    def analyze_market_conditions(self, price: float, rsi: float = 50.0, 
                                 volume_ratio: float = 1.0) -> MarketConditions:
        """Quick market condition analysis"""
        current_time = time.time()
        
        # Check cache
        cache_key = f"{price}_{rsi}_{volume_ratio}"
        if (cache_key in self.analysis_cache and 
            current_time - self.last_analysis_time < self.cache_duration):
            return self.analysis_cache[cache_key]
        
        # Determine trend (simplified)
        trend = "neutral"
        if rsi > 60:
            trend = "up"
        elif rsi < 40:
            trend = "down"
        
        # Determine volatility (based on RSI extremes)
        volatility = "normal"
        if rsi > 75 or rsi < 25:
            volatility = "high"
        elif 45 <= rsi <= 55:
            volatility = "low"
        
        # Volume surge detection
        volume_surge = volume_ratio > 2.0
        
        conditions = MarketConditions(
            price=price,
            rsi=rsi,
            trend=trend,
            volatility=volatility,
            volume_surge=volume_surge
        )
        
        # Cache result
        self.analysis_cache[cache_key] = conditions
        self.last_analysis_time = current_time
        
        return conditions
    
    def generate_consensus_signal(self, price: float, rsi: float = 50.0, 
                                 volume_ratio: float = 1.0,
                                 btc_balance: float = 0, eur_balance: float = 0) -> SimpleSignal:
        """Generate consensus signal from all strategies"""
        try:
            # Check emergency stop
            if self.emergency_stop:
                return SimpleSignal(TradingAction.HOLD, 0.0, "Emergency stop active", "Manager")
            
            # Check rate limiting
            if not self._check_rate_limit():
                return SimpleSignal(TradingAction.HOLD, 0.0, "Rate limit exceeded", "Manager")
            
            # Analyze market conditions
            conditions = self.analyze_market_conditions(price, rsi, volume_ratio)
            
            # Collect signals from strategies
            signals = []
            
            for name, strategy in self.strategies.items():
                try:
                    if name == 'balanced':
                        signal = strategy.generate_signal(conditions, btc_balance, eur_balance)
                    else:
                        signal = strategy.generate_signal(conditions)
                    
                    if signal.action != TradingAction.HOLD:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.warning(f"Strategy {name} failed: {e}")
                    continue
            
            if not signals:
                return SimpleSignal(TradingAction.HOLD, 0.0, "No active signals", "Manager")
            
            # Sort by priority and confidence
            signals.sort(key=lambda s: (s.priority, s.confidence), reverse=True)
            
            # Take the highest priority signal
            best_signal = signals[0]
            
            # Apply consensus logic for similar priority signals
            if len(signals) > 1:
                top_signals = [s for s in signals if s.priority == best_signal.priority]
                
                if len(top_signals) > 1:
                    # Check for consensus
                    buy_signals = [s for s in top_signals if s.action == TradingAction.BUY]
                    sell_signals = [s for s in top_signals if s.action == TradingAction.SELL]
                    
                    if len(buy_signals) >= len(sell_signals) and buy_signals:
                        # Buy consensus
                        avg_confidence = np.mean([s.confidence for s in buy_signals])
                        avg_volume = np.mean([s.volume_pct for s in buy_signals])
                        strategies = [s.strategy for s in buy_signals]
                        
                        best_signal = SimpleSignal(
                            TradingAction.BUY,
                            min(0.9, avg_confidence + 0.1),  # Consensus boost
                            f"Buy consensus: {', '.join(strategies)}",
                            "Consensus",
                            volume_pct=min(0.05, avg_volume),  # Cap at 5%
                            priority=max(s.priority for s in buy_signals)
                        )
                    
                    elif sell_signals:
                        # Sell consensus
                        avg_confidence = np.mean([s.confidence for s in sell_signals])
                        avg_volume = np.mean([s.volume_pct for s in sell_signals])
                        strategies = [s.strategy for s in sell_signals]
                        
                        best_signal = SimpleSignal(
                            TradingAction.SELL,
                            min(0.9, avg_confidence + 0.1),
                            f"Sell consensus: {', '.join(strategies)}",
                            "Consensus",
                            volume_pct=min(0.05, avg_volume),
                            priority=max(s.priority for s in sell_signals)
                        )
            
            # Apply final safety checks
            best_signal = self._apply_safety_filters(best_signal, conditions)
            
            # Record signal
            self.signal_history.append({
                'timestamp': time.time(),
                'action': best_signal.action.value,
                'confidence': best_signal.confidence,
                'strategy': best_signal.strategy
            })
            self.total_signals += 1
            
            logger.info(f"Generated signal: {best_signal.action.value} ({best_signal.confidence:.2f}) - {best_signal.reasoning}")
            return best_signal
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return SimpleSignal(TradingAction.HOLD, 0.0, f"Error: {str(e)[:50]}", "Manager")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        recent_signals = [s for s in self.signal_history if s['timestamp'] > hour_ago]
        return len(recent_signals) < self.max_signals_per_hour
    
    def _apply_safety_filters(self, signal: SimpleSignal, conditions: MarketConditions) -> SimpleSignal:
        """Apply final safety filters"""
        # Don't trade in extreme volatility
        if conditions.volatility == "high" and signal.action != TradingAction.HOLD:
            signal.confidence *= 0.7
            signal.volume_pct *= 0.5
            signal.reasoning += " (reduced for high volatility)"
        
        # Confidence threshold
        if signal.confidence < 0.5:
            return SimpleSignal(TradingAction.HOLD, 0.0, "Below confidence threshold", "Safety")
        
        # Volume limits
        signal.volume_pct = min(0.08, signal.volume_pct)  # Max 8% position
        
        return signal
    
    def update_strategy_weights(self, performance_data: Dict[str, float]):
        """Simple weight adjustment based on performance"""
        try:
            for strategy_name, strategy in self.strategies.items():
                if strategy_name in performance_data:
                    perf = performance_data[strategy_name]
                    
                    # Adjust weight based on performance
                    if perf > 0.6:  # Good performance
                        strategy.weight = min(1.0, strategy.weight * 1.1)
                    elif perf < 0.4:  # Poor performance
                        strategy.weight = max(0.1, strategy.weight * 0.9)
            
            logger.info("Strategy weights updated based on performance")
            
        except Exception as e:
            logger.error(f"Weight update failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            "strategies": {name: {
                "enabled": s.enabled,
                "weight": s.weight,
                "last_signal": s.last_signal_time
            } for name, s in self.strategies.items()},
            "performance": {
                "total_signals": self.total_signals,
                "successful_trades": self.successful_trades,
                "success_rate": self.successful_trades / max(1, self.total_signals),
                "recent_signals": len([s for s in self.signal_history if time.time() - s['timestamp'] < 3600])
            },
            "controls": {
                "emergency_stop": self.emergency_stop,
                "cache_size": len(self.analysis_cache),
                "rate_limit_ok": self._check_rate_limit()
            }
        }
    
    def set_emergency_stop(self, active: bool):
        """Set emergency stop"""
        self.emergency_stop = active
        logger.warning(f"Emergency stop {'ACTIVATED' if active else 'DEACTIVATED'}")
    
    def record_trade_outcome(self, success: bool):
        """Record trade outcome for performance tracking"""
        if success:
            self.successful_trades += 1
    
    def cleanup(self):
        """Cleanup resources"""
        self.analysis_cache.clear()
        self.signal_history.clear()
        logger.info("Simplified Strategy Manager cleaned up")


# Usage example
def demo_simplified_manager():
    """Demonstrate the simplified strategy manager"""
    print("Testing Simplified Strategy Manager...")
    
    manager = SimplifiedStrategyManager()
    
    # Test signal generation
    test_conditions = [
        (45000, 30, 1.0),   # Oversold
        (46000, 75, 2.5),   # Overbought with volume
        (45500, 50, 1.0),   # Neutral
    ]
    
    for price, rsi, volume_ratio in test_conditions:
        signal = manager.generate_consensus_signal(
            price=price,
            rsi=rsi,
            volume_ratio=volume_ratio,
            btc_balance=0.1,
            eur_balance=5000
        )
        
        print(f"Price: €{price}, RSI: {rsi}")
        print(f"Signal: {signal.action.value} ({signal.confidence:.2f}) - {signal.reasoning}")
        print(f"Volume: {signal.volume_pct:.1%}\n")
    
    # Test status
    status = manager.get_status()
    print(f"Status: {status}")
    
    manager.cleanup()
    print("✅ Simplified Strategy Manager test completed")


if __name__ == "__main__":
    demo_simplified_manager()