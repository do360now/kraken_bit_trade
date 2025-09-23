"""
Unified Trading Bot Core - Consolidates duplicate implementations
Production-ready with proper separation of concerns
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TradingAction(Enum):
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class MarketIndicators:
    """Comprehensive market indicators"""
    current_price: float
    current_volume: float
    rsi: float = 50.0
    macd: float = 0.0
    signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0 
    bollinger_lower: float = 0.0
    vwap: float = 0.0
    volatility: float = 0.02
    sentiment: float = 0.0
    risk_off_probability: float = 0.0
    ml_success_probability: Optional[float] = None
    market_regime: str = "ranging"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TradingSignal:
    """Enhanced trading signal with risk management"""
    action: TradingAction
    confidence: float
    volume: float
    price: float
    reasoning: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    expected_return: float = 0.0
    max_risk: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class CircuitBreaker:
    """Circuit breaker for emergency stops"""
    
    def __init__(self, daily_loss_threshold: float = 0.10, 
                 consecutive_loss_threshold: int = 5):
        self.daily_loss_threshold = daily_loss_threshold
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        self.is_tripped = False
        
    def check_conditions(self, trade_pnl: float, portfolio_value: float) -> bool:
        """Check if circuit breaker should trip"""
        current_date = datetime.now().date()
        
        # Reset daily counters
        if current_date > self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = current_date
            
        # Update daily P&L
        self.daily_pnl += trade_pnl
        
        # Check daily loss threshold
        daily_loss_pct = abs(self.daily_pnl) / portfolio_value if portfolio_value > 0 else 0
        if daily_loss_pct > self.daily_loss_threshold and self.daily_pnl < 0:
            logger.critical(f"CIRCUIT BREAKER: Daily loss {daily_loss_pct:.1%} exceeds threshold")
            self.is_tripped = True
            return True
            
        # Check consecutive losses
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        if self.consecutive_losses >= self.consecutive_loss_threshold:
            logger.critical(f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses")
            self.is_tripped = True
            return True
            
        return False
    
    def reset(self):
        """Manual reset of circuit breaker"""
        self.is_tripped = False
        self.consecutive_losses = 0
        logger.info("Circuit breaker manually reset")

class PositionManager:
    """Advanced position sizing and risk management"""
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.daily_trades = 0
        self.last_trade_time = 0
        
    def calculate_position_size(self, signal: TradingSignal, 
                              portfolio_value: float, 
                              current_exposure: float) -> Tuple[float, str]:
        """Calculate optimal position size with risk controls"""
        
        if signal.action == TradingAction.HOLD:
            return 0.0, "Hold signal"
            
        # Base position size
        base_size = self.config.trading.base_position_size_pct
        
        # Confidence adjustment (0.5x to 1.5x)
        confidence_mult = 0.5 + signal.confidence
        
        # Risk level adjustment
        risk_adjustments = {
            RiskLevel.LOW: 1.2,
            RiskLevel.MEDIUM: 1.0, 
            RiskLevel.HIGH: 0.6,
            RiskLevel.EXTREME: 0.3
        }
        risk_mult = risk_adjustments[signal.risk_level]
        
        # Volatility adjustment
        volatility_mult = max(0.5, min(1.5, 1.0 - (signal.max_risk - 0.02) * 10))
        
        # Portfolio concentration check
        concentration_mult = 1.0
        if current_exposure > 0.5:  # Already 50%+ exposed
            concentration_mult = 0.3
        elif current_exposure > 0.3:  # 30%+ exposed
            concentration_mult = 0.6
            
        # Calculate final size
        position_pct = base_size * confidence_mult * risk_mult * volatility_mult * concentration_mult
        
        # Apply bounds
        position_pct = max(
            self.config.trading.min_position_size_pct,
            min(self.config.trading.max_position_size_pct, position_pct)
        )
        
        # Convert to actual volume
        position_value = portfolio_value * position_pct
        volume = position_value / signal.price
        
        reasoning = f"Base: {base_size:.1%}, Conf: {confidence_mult:.2f}x, Risk: {risk_mult:.2f}x, Vol: {volatility_mult:.2f}x"
        
        return volume, reasoning
    
    def validate_trade(self, signal: TradingSignal, portfolio_value: float) -> Tuple[bool, str]:
        """Validate trade against all risk parameters"""
        
        # Check daily trade limit
        if self.daily_trades >= self.config.security.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades}"
            
        # Check cooldown period
        if time.time() - self.last_trade_time < self.config.trading.trade_cooldown_minutes * 60:
            remaining = self.config.trading.trade_cooldown_minutes * 60 - (time.time() - self.last_trade_time)
            return False, f"Cooldown active: {remaining:.0f}s remaining"
            
        # Check minimum trade value
        trade_value = signal.volume * signal.price
        min_trade_eur = 15.0  # Minimum viable trade
        if trade_value < min_trade_eur:
            return False, f"Trade value €{trade_value:.2f} below minimum €{min_trade_eur:.2f}"
            
        # Check position size vs portfolio
        position_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
        if position_pct > self.config.risk.position_concentration_limit:
            return False, f"Position size {position_pct:.1%} exceeds limit {self.config.risk.position_concentration_limit:.1%}"
            
        return True, "Trade validated"
    
    def record_trade(self, signal: TradingSignal):
        """Record executed trade"""
        self.daily_trades += 1
        self.last_trade_time = time.time()
        
        trade_record = {
            'timestamp': signal.timestamp,
            'action': signal.action.value,
            'volume': signal.volume,
            'price': signal.price,
            'confidence': signal.confidence
        }
        
        # Store in positions tracking
        position_key = f"{signal.action.value}_{int(signal.timestamp.timestamp())}"
        self.positions[position_key] = trade_record

class UnifiedTradingBot:
    """
    Unified trading bot that consolidates all previous implementations
    Production-ready with proper async support and risk management
    """
    
    def __init__(self, api_client, data_manager, config):
        self.api_client = api_client
        self.data_manager = data_manager
        self.config = config
        
        # Core components
        self.position_manager = PositionManager(config)
        self.circuit_breaker = CircuitBreaker(
            daily_loss_threshold=config.risk.max_drawdown_pct,
            consecutive_loss_threshold=5
        )
        
        # State tracking
        self.session_start = datetime.now()
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Price history cache
        self.price_history = []
        self.volume_history = []
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Optional components (lazy loaded)
        self._ml_engine = None
        self._peak_detector = None
        
        logger.info("Unified Trading Bot initialized")
    
    async def analyze_market(self) -> MarketIndicators:
        """
        Comprehensive market analysis with async data fetching
        """
        try:
            # Fetch current market data asynchronously
            current_price = await self.api_client.get_btc_price()
            current_volume = await self.api_client.get_market_volume()
            
            if not current_price:
                raise ValueError("Cannot fetch current market price")
                
            # Update price history
            await self._update_price_history()
            
            # Ensure we have enough data
            if len(self.price_history) < 50:
                raise ValueError(f"Insufficient price history: {len(self.price_history)} points")
            
            # Calculate technical indicators in thread pool
            indicators_task = self.executor.submit(
                self._calculate_technical_indicators,
                self.price_history, self.volume_history, current_price
            )
            
            # Calculate base indicators
            base_indicators = await asyncio.get_event_loop().run_in_executor(
                None, indicators_task.result
            )
            
            # Create market indicators object
            indicators = MarketIndicators(
                current_price=current_price,
                current_volume=current_volume,
                **base_indicators
            )
            
            # Add advanced analysis if available
            await self._enhance_indicators(indicators)
            
            logger.debug(f"Market analysis complete: Price={current_price:.2f}, RSI={indicators.rsi:.1f}")
            return indicators
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise
    
    async def _update_price_history(self):
        """Update price history with latest OHLC data"""
        try:
            # Get recent candles
            ohlc_data = await self.api_client.get_ohlc_data(
                pair="XXBTZEUR",
                interval=15,
                limit=100
            )
            
            if ohlc_data:
                # Append to data manager
                added_count = self.data_manager.append_ohlc_data(ohlc_data)
                
                if added_count > 0:
                    # Reload price history
                    prices, volumes = self.data_manager.load_price_history()
                    self.price_history = prices[-1000:]  # Keep last 1000 points
                    self.volume_history = volumes[-1000:]
                    logger.debug(f"Updated price history: {added_count} new candles")
                    
        except Exception as e:
            logger.warning(f"Failed to update price history: {e}")
    
    def _calculate_technical_indicators(self, prices: List[float], 
                                      volumes: List[float], 
                                      current_price: float) -> Dict[str, float]:
        """Calculate technical indicators - CPU intensive"""
        try:
            # Import here to avoid startup dependencies
            from core.indicators import (
                calculate_rsi, calculate_macd, calculate_bollinger_bands,
                calculate_moving_average, calculate_vwap
            )
            
            # Calculate indicators
            rsi = calculate_rsi(prices) or 50.0
            macd, signal = calculate_macd(prices) or (0.0, 0.0)
            upper, middle, lower = calculate_bollinger_bands(prices) or (
                current_price * 1.02, current_price, current_price * 0.98
            )
            vwap = calculate_vwap(prices, volumes) or current_price
            
            # Calculate volatility safely
            volatility = self._calculate_volatility(prices)
            
            # Determine market regime
            ma_20 = calculate_moving_average(prices, 20) or current_price
            ma_50 = calculate_moving_average(prices, 50) or current_price
            
            if ma_20 > ma_50 * 1.02:
                market_regime = "uptrend"
            elif ma_20 < ma_50 * 0.98:
                market_regime = "downtrend"
            else:
                market_regime = "ranging"
            
            return {
                'rsi': rsi,
                'macd': macd,
                'signal': signal,
                'bollinger_upper': upper,
                'bollinger_middle': middle,
                'bollinger_lower': lower,
                'vwap': vwap,
                'volatility': volatility,
                'market_regime': market_regime
            }
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            return {
                'rsi': 50.0, 'macd': 0.0, 'signal': 0.0,
                'bollinger_upper': current_price * 1.02,
                'bollinger_middle': current_price,
                'bollinger_lower': current_price * 0.98,
                'vwap': current_price, 'volatility': 0.02,
                'market_regime': 'unknown'
            }
    
    def _calculate_volatility(self, prices: List[float], window: int = 20) -> float:
        """Safe volatility calculation"""
        try:
            if len(prices) < window + 1:
                return 0.02
                
            recent_prices = np.array(prices[-window-1:])
            returns = np.diff(recent_prices) / recent_prices[:-1]
            
            # Remove any infinite or NaN values
            returns = returns[np.isfinite(returns)]
            
            if len(returns) == 0:
                return 0.02
                
            volatility = float(np.std(returns))
            return max(0.001, min(1.0, volatility))  # Bounded between 0.1% and 100%
            
        except Exception:
            return 0.02
    
    async def _enhance_indicators(self, indicators: MarketIndicators):
        """Add ML and advanced analysis to indicators"""
        try:
            # Add sentiment analysis
            if self.config.enable_news_sentiment:
                await self._add_sentiment_analysis(indicators)
            
            # Add ML predictions
            if self.config.enable_ml and self._ml_engine:
                await self._add_ml_predictions(indicators)
                
            # Add peak detection
            if self.config.enable_peak_detection and self._peak_detector:
                await self._add_peak_analysis(indicators)
                
        except Exception as e:
            logger.warning(f"Enhanced indicators failed: {e}")
    
    async def _add_sentiment_analysis(self, indicators: MarketIndicators):
        """Add news sentiment analysis"""
        try:
            # Placeholder for sentiment analysis
            # Would integrate with news APIs
            indicators.sentiment = 0.0
            indicators.risk_off_probability = 0.0
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
    
    async def _add_ml_predictions(self, indicators: MarketIndicators):
        """Add ML predictions"""
        try:
            # Placeholder for ML predictions
            indicators.ml_success_probability = 0.5
        except Exception as e:
            logger.warning(f"ML predictions failed: {e}")
    
    async def _add_peak_analysis(self, indicators: MarketIndicators):
        """Add peak detection analysis"""
        try:
            # Placeholder for peak detection
            pass
        except Exception as e:
            logger.warning(f"Peak analysis failed: {e}")
    
    async def generate_signal(self, indicators: MarketIndicators) -> TradingSignal:
        """
        Generate trading signal with comprehensive analysis
        """
        try:
            # Check circuit breaker
            if self.circuit_breaker.is_tripped:
                return TradingSignal(
                    action=TradingAction.HOLD,
                    confidence=0.0,
                    volume=0.0,
                    price=indicators.current_price,
                    reasoning=["Circuit breaker active"],
                    risk_level=RiskLevel.EXTREME
                )
            
            # Calculate signal scores
            buy_score, buy_reasons = self._calculate_buy_score(indicators)
            sell_score, sell_reasons = self._calculate_sell_score(indicators)
            
            # Determine action
            action = TradingAction.HOLD
            confidence = 0.5
            reasoning = []
            
            # Conservative thresholds for live trading
            if buy_score >= 4.0 and buy_score > sell_score + 1.0:
                action = TradingAction.BUY
                confidence = min(0.9, buy_score / 7.0)
                reasoning = buy_reasons[:3]
                
            elif sell_score >= 3.5 and sell_score > buy_score + 0.5:
                action = TradingAction.SELL
                confidence = min(0.9, sell_score / 6.0)
                reasoning = sell_reasons[:3]
                
            else:
                reasoning = [f"Insufficient signals: Buy={buy_score:.1f}, Sell={sell_score:.1f}"]
            
            # Assess risk level
            risk_level = self._assess_risk_level(indicators)
            
            # Calculate position size
            portfolio_value = await self._get_portfolio_value()
            current_exposure = await self._get_current_exposure()
            
            volume, size_reasoning = self.position_manager.calculate_position_size(
                TradingSignal(action, confidence, 0, indicators.current_price, reasoning, risk_level),
                portfolio_value,
                current_exposure
            )
            
            # Create final signal
            signal = TradingSignal(
                action=action,
                confidence=confidence,
                volume=volume,
                price=indicators.current_price,
                reasoning=reasoning,
                risk_level=risk_level,
                stop_loss=self._calculate_stop_loss(action, indicators),
                take_profit=self._calculate_take_profit(action, indicators)
            )
            
            logger.info(f"Signal: {action.value.upper()} | Confidence: {confidence:.1%} | Volume: {volume:.6f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return TradingSignal(
                action=TradingAction.HOLD,
                confidence=0.0,
                volume=0.0,
                price=indicators.current_price,
                reasoning=[f"Signal generation error: {str(e)[:50]}"],
                risk_level=RiskLevel.EXTREME
            )
    
    def _calculate_buy_score(self, indicators: MarketIndicators) -> Tuple[float, List[str]]:
        """Calculate buy signal strength"""
        score = 0.0
        reasons = []
        
        # RSI oversold conditions
        if indicators.rsi < 30:
            score += 1.5
            reasons.append(f"Strong oversold RSI: {indicators.rsi:.1f}")
        elif indicators.rsi < 40:
            score += 0.8
            reasons.append(f"Mild oversold RSI: {indicators.rsi:.1f}")
        
        # Price vs VWAP
        if indicators.current_price < indicators.vwap * 0.98:
            score += 1.0
            reasons.append("Price below VWAP")
        
        # Bollinger Bands
        if indicators.current_price < indicators.bollinger_lower:
            score += 1.2
            reasons.append("Price below Bollinger lower band")
        
        # MACD
        if indicators.macd > indicators.signal:
            score += 0.7
            reasons.append("MACD bullish crossover")
        
        # Market regime
        if indicators.market_regime == "uptrend":
            score += 0.5
            reasons.append("Uptrend regime")
        
        # Sentiment
        if indicators.sentiment > 0.1:
            score += 0.8
            reasons.append(f"Positive sentiment: {indicators.sentiment:.2f}")
        
        # ML prediction
        if indicators.ml_success_probability and indicators.ml_success_probability > 0.7:
            score += 1.0
            reasons.append(f"High ML confidence: {indicators.ml_success_probability:.1%}")
        
        return score, reasons
    
    def _calculate_sell_score(self, indicators: MarketIndicators) -> Tuple[float, List[str]]:
        """Calculate sell signal strength"""
        score = 0.0
        reasons = []
        
        # RSI overbought conditions
        if indicators.rsi > 70:
            score += 1.5
            reasons.append(f"Overbought RSI: {indicators.rsi:.1f}")
        elif indicators.rsi > 65:
            score += 0.8
            reasons.append(f"High RSI: {indicators.rsi:.1f}")
        
        # Price vs VWAP
        if indicators.current_price > indicators.vwap * 1.02:
            score += 1.0
            reasons.append("Price above VWAP")
        
        # Bollinger Bands
        if indicators.current_price > indicators.bollinger_upper:
            score += 1.2
            reasons.append("Price above Bollinger upper band")
        
        # Risk-off conditions
        if indicators.risk_off_probability > 0.6:
            score += 2.0
            reasons.append(f"High risk-off probability: {indicators.risk_off_probability:.1%}")
        
        # High volatility
        if indicators.volatility > 0.08:
            score += 1.0
            reasons.append(f"High volatility: {indicators.volatility:.1%}")
        
        # Negative sentiment
        if indicators.sentiment < -0.2:
            score += 1.0
            reasons.append(f"Negative sentiment: {indicators.sentiment:.2f}")
        
        return score, reasons
    
    def _assess_risk_level(self, indicators: MarketIndicators) -> RiskLevel:
        """Assess current risk level"""
        risk_score = 0
        
        if indicators.volatility > 0.08:
            risk_score += 2
        elif indicators.volatility > 0.05:
            risk_score += 1
            
        if indicators.risk_off_probability > 0.7:
            risk_score += 3
        elif indicators.risk_off_probability > 0.4:
            risk_score += 1
            
        if indicators.sentiment < -0.3:
            risk_score += 2
            
        if risk_score >= 5:
            return RiskLevel.EXTREME
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _get_portfolio_value(self) -> float:
        """Get total portfolio value in EUR"""
        try:
            btc_balance = await self.api_client.get_btc_balance() or 0
            eur_balance = await self.api_client.get_eur_balance() or 0
            current_price = await self.api_client.get_btc_price() or 0
            
            return eur_balance + (btc_balance * current_price)
        except Exception:
            return 1000.0  # Fallback value
    
    async def _get_current_exposure(self) -> float:
        """Get current BTC exposure as percentage of portfolio"""
        try:
            btc_balance = await self.api_client.get_btc_balance() or 0
            portfolio_value = await self._get_portfolio_value()
            current_price = await self.api_client.get_btc_price() or 0
            
            btc_value = btc_balance * current_price
            return btc_value / portfolio_value if portfolio_value > 0 else 0
        except Exception:
            return 0.5  # Assume 50% exposure if unknown
    
    def _calculate_stop_loss(self, action: TradingAction, indicators: MarketIndicators) -> Optional[float]:
        """Calculate stop loss price"""
        if action == TradingAction.HOLD:
            return None
            
        base_stop = self.config.trading.stop_loss_pct
        
        # Adjust for volatility
        if indicators.volatility > 0.05:
            base_stop *= 1.5
            
        if action == TradingAction.BUY:
            return indicators.current_price * (1 - base_stop)
        else:
            return indicators.current_price * (1 + base_stop)
    
    def _calculate_take_profit(self, action: TradingAction, indicators: MarketIndicators) -> Optional[float]:
        """Calculate take profit price"""
        if action == TradingAction.HOLD:
            return None
            
        base_profit = self.config.trading.take_profit_pct
        
        if action == TradingAction.BUY:
            return indicators.current_price * (1 + base_profit)
        else:
            return indicators.current_price * (1 - base_profit)
    
    async def execute_strategy(self):
        """
        Main strategy execution method
        """
        try:
            logger.info("Executing trading strategy...")
            
            # Analyze market
            indicators = await self.analyze_market()
            
            # Generate signal
            signal = await self.generate_signal(indicators)
            
            # Validate trade
            if signal.action != TradingAction.HOLD:
                portfolio_value = await self._get_portfolio_value()
                
                is_valid, validation_msg = self.position_manager.validate_trade(signal, portfolio_value)
                
                if not is_valid:
                    logger.info(f"Trade validation failed: {validation_msg}")
                    return
                
                # Execute trade
                success = await self._execute_trade(signal)
                
                if success:
                    self.position_manager.record_trade(signal)
                    self.total_trades += 1
                    logger.info(f"Trade executed successfully: {signal.action.value.upper()}")
                else:
                    logger.warning("Trade execution failed")
            else:
                logger.info(f"Holding: {signal.reasoning[0] if signal.reasoning else 'Low confidence'}")
                
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
    
    async def _execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade with proper error handling"""
        try:
            # Get order book for optimal pricing
            order_book = await self.api_client.get_order_book("XXBTZEUR")
            if not order_book:
                logger.error("Cannot get order book")
                return False
            
            # Calculate optimal price
            optimal_price = self.api_client.get_optimal_price(order_book, signal.action.value)
            if not optimal_price:
                logger.error("Cannot calculate optimal price")
                return False
            
            # Place order
            order_id = await self.api_client.place_order(
                pair="XXBTZEUR",
                order_type="limit",
                side=signal.action.value,
                volume=signal.volume,
                price=optimal_price
            )
            
            if order_id:
                logger.info(f"Order placed: {order_id} - {signal.action.value.upper()} {signal.volume:.6f} BTC @ €{optimal_price:.2f}")
                return True
            else:
                logger.error("Failed to place order")
                return False
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        try:
            # Get current balances
            btc_balance = await self.api_client.get_btc_balance() or 0
            eur_balance = await self.api_client.get_eur_balance() or 0
            current_price = await self.api_client.get_btc_price() or 0
            
            portfolio_value = eur_balance + (btc_balance * current_price)
            btc_exposure = (btc_balance * current_price) / portfolio_value if portfolio_value > 0 else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "session": {
                    "start_time": self.session_start.isoformat(),
                    "duration_hours": (datetime.now() - self.session_start).total_seconds() / 3600,
                    "total_trades": self.total_trades,
                    "daily_trades": self.position_manager.daily_trades
                },
                "portfolio": {
                    "btc_balance": btc_balance,
                    "eur_balance": eur_balance,
                    "total_value_eur": portfolio_value,
                    "btc_exposure_pct": btc_exposure,
                    "current_price": current_price
                },
                "risk_management": {
                    "circuit_breaker_active": self.circuit_breaker.is_tripped,
                    "daily_pnl": self.circuit_breaker.daily_pnl,
                    "consecutive_losses": self.circuit_breaker.consecutive_losses
                },
                "market": {
                    "price_history_size": len(self.price_history),
                    "last_update": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Status generation failed: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down unified trading bot...")
        
        try:
            # Cancel any pending orders
            open_orders = await self.api_client.get_open_orders()
            for order_id in open_orders:
                await self.api_client.cancel_order(order_id)
                logger.info(f"Cancelled order: {order_id}")
            
            # Close API client
            await self.api_client.close()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            logger.info("Unified trading bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")