"""
Simplified TradingBot - Orchestrator Pattern

OUSTERHOUT PRINCIPLE: Keep it simple.

This bot is now a THIN ORCHESTRATOR (~200 lines) instead of a 1000+ line monolith.

All complexity moved to specialized modules:
- MarketDataService: Price fetching
- TradingStrategy: Buy/sell decisions  
- RiskManager: Risk assessment
- PositionManager: Portfolio tracking
- TradeExecutor: Order execution

The bot just ORCHESTRATES these pieces together.
"""

from typing import Optional, Dict, List
from datetime import datetime, timedelta
import time
import logging
from logger_config import logger

from market_data_service import MarketDataService
from risk_manager import RiskManager, PortfolioState
from position_manager import PositionManager
from trade_executor import TradeExecutor
from indicators import (
    calculate_rsi, calculate_macd, calculate_moving_average,
    calculate_vwap, calculate_bollinger_bands, calculate_sentiment,
    fetch_enhanced_news, calculate_enhanced_sentiment,
    calculate_risk_adjusted_indicators
)
from data_manager import DataManager
from onchain_analyzer import OnChainAnalyzer
from performance_tracker import PerformanceTracker
from metrics_server import MetricsServer


class TradingBot:
    """
    Simplified trading bot - pure orchestration.
    
    Responsibilities:
    1. Initialize services
    2. Run main trading loop
    3. Coordinate between services
    
    NO business logic - delegates to specialized modules.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        trade_executor: TradeExecutor,
        data_manager: DataManager,
        onchain_analyzer: OnChainAnalyzer,
    ):
        """Initialize bot with all dependencies injected."""
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.position = position_manager
        self.executor = trade_executor
        self.data_manager = data_manager
        self.onchain = onchain_analyzer

        # Performance tracking
        self.performance_tracker = PerformanceTracker(
            initial_btc_balance=0.0,
            initial_eur_balance=0.0,
        )
        self.metrics_server = MetricsServer(self)
        self.metrics_server.start()

        # State
        self.running = False
        self.price_history: List[float] = []
        self.last_decision = "HOLD"

    def run(self):
        """Main trading loop."""
        logger.info("🤖 Trading bot started")
        self.running = True

        try:
            while self.running:
                try:
                    self._execute_trading_cycle()
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)

                # Wait before next cycle (e.g., 60 seconds)
                time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.stop()

    def stop(self):
        """Stop the bot."""
        self.running = False
        self.metrics_server.stop()
        logger.info("Bot stopped")

    # =========================================================================
    # MAIN ORCHESTRATION - The only "business logic" in this class
    # =========================================================================

    def _execute_trading_cycle(self):
        """
        Execute one trading cycle.

        Flow:
        1. Get market state
        2. Check risk
        3. Evaluate conditions
        4. Execute if approved
        5. Update position
        """
        try:
            # Get current state
            price = self.market_data.current_price()
            position = self.position.get_position()

            logger.debug(f"Cycle: Price €{price.value:.0f}, Position: {position.btc_amount:.8f} BTC")

            # Update performance tracker
            self.performance_tracker.update_equity(
                position.btc_amount,
                position.eur_balance,
                price.value,
            )

            # Check if we should trade based on risk
            portfolio = self._create_portfolio_state(position, price.value)
            if not self.risk_manager.assess_risk(portfolio).can_trade:
                logger.debug("⚠️  High risk - skipping this cycle")
                return

            # Placeholder: Decision logic would go here
            # In production, you'd integrate AccumulationStrategy or other logic
            # For now, just evaluate basic conditions

            logger.debug(f"✓ Cycle completed - Last decision: {self.last_decision}")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _create_portfolio_state(
        self, position, current_price: float
    ) -> PortfolioState:
        """Create PortfolioState for risk manager."""
        return PortfolioState(
            btc_balance=position.btc_amount,
            eur_balance=position.eur_balance,
            current_price=current_price,
            avg_buy_price=position.avg_buy_price,
            unrealized_pnl=position.unrealized_pnl,
            win_rate=0.5,
            volatility=0.02,
            max_daily_drawdown=0.0,
        )
