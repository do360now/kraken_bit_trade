"""
TradingBot - Cycle-Aware Orchestrator

OUSTERHOUT PRINCIPLE: Thin orchestrator that coordinates deep modules.

This bot orchestrates:
- MarketDataService: Price fetching
- CycleAnalyzer: Halving cycle phase detection
- RiskManager: Cycle-aware risk assessment
- PositionManager: Portfolio tracking
- TradeExecutor: Order execution
- Indicators: Technical + cycle-adjusted signals

The cycle phase is the TOP-LEVEL filter that shapes every decision.
"""

from typing import Optional, Dict, List
from datetime import datetime, timedelta
import time

from logger_config import logger
from market_data_service import MarketDataService
from cycle_analyzer import CycleAnalyzer, CyclePhase
from risk_manager import RiskManager, PortfolioState
from position_manager import PositionManager
from trade_executor import TradeExecutor
from indicators import (
    calculate_rsi, calculate_macd, calculate_moving_average,
    calculate_vwap, calculate_bollinger_bands,
    fetch_enhanced_news, calculate_enhanced_sentiment,
    calculate_risk_adjusted_indicators,
)
from data_manager import DataManager
from onchain_analyzer import OnChainAnalyzer


class TradingBot:
    """
    Cycle-aware trading bot.

    The halving cycle phase is the MASTER context that modulates
    every other signal: RSI thresholds shift, position sizes scale,
    sell gates tighten or loosen, and the bot's overall posture
    adapts to whether we're in accumulation or distribution territory.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        cycle_analyzer: CycleAnalyzer,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        trade_executor: TradeExecutor,
        data_manager: DataManager,
        onchain_analyzer: OnChainAnalyzer,
    ):
        self.market_data = market_data
        self.cycle = cycle_analyzer
        self.risk_manager = risk_manager
        self.position = position_manager
        self.executor = trade_executor
        self.data_manager = data_manager
        self.onchain = onchain_analyzer

        # State
        self.running = False
        self.last_decision = "HOLD"
        self.last_cycle_log_time = 0
        self._cycle_log_interval = 3600  # Log full cycle summary hourly

    def run(self):
        """Main trading loop."""
        logger.info("🤖 Cycle-aware trading bot started")
        self._log_cycle_status()
        self.running = True

        try:
            while self.running:
                try:
                    self._execute_trading_cycle()
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        logger.info("Bot stopped")

    # =========================================================================
    # MAIN ORCHESTRATION
    # =========================================================================

    def _execute_trading_cycle(self):
        """
        Execute one trading cycle with cycle awareness.

        Flow:
        1. Get market state + cycle phase
        2. Calculate cycle-adjusted indicators
        3. Assess risk (cycle-modulated)
        4. Make buy/sell/hold decision
        5. Execute if approved
        6. Update position
        """
        # 1. Get current state
        price = self.market_data.current_price()
        pos = self.position.get_position()
        current_price = price.value

        # 2. Get cycle context (the master signal)
        cycle_adj = self.cycle.get_cycle_adjustments(current_price)
        accum_score = self.cycle.get_accumulation_score(current_price)

        # Periodic full cycle status log
        now = time.time()
        if now - self.last_cycle_log_time > self._cycle_log_interval:
            self._log_cycle_status(current_price)
            self.last_cycle_log_time = now

        logger.debug(
            f"Tick: €{current_price:,.0f} | Phase={cycle_adj.phase.value} | "
            f"AccumScore={accum_score:.0f} | Pos={pos.btc_amount:.8f} BTC"
        )

        # 3. Assess risk (cycle-aware)
        portfolio = self._create_portfolio_state(pos, current_price)
        self.risk_manager.update_peak_value(portfolio)
        risk = self.risk_manager.assess_risk(portfolio)

        # 4. Check for emergency conditions first
        if self.risk_manager.should_emergency_sell(portfolio):
            self._execute_emergency_sell(pos, current_price)
            return

        # 5. Get technical indicators
        indicators = self._get_indicators()
        if not indicators:
            logger.debug("Insufficient data for indicators, skipping")
            return

        # 6. Make cycle-aware trading decision
        decision = self._evaluate_signals(
            indicators=indicators,
            cycle_adj=cycle_adj,
            accum_score=accum_score,
            portfolio=portfolio,
            current_price=current_price,
        )

        # 7. Execute decision
        if decision == "BUY" and self.risk_manager.can_buy(portfolio):
            self._execute_buy(portfolio, current_price, cycle_adj)
        elif decision == "SELL" and self.risk_manager.can_sell(portfolio):
            self._execute_sell(pos, current_price, cycle_adj)

        self.last_decision = decision

    # =========================================================================
    # SIGNAL EVALUATION — Where cycle phase meets technical indicators
    # =========================================================================

    def _evaluate_signals(
        self,
        indicators: Dict,
        cycle_adj,
        accum_score: float,
        portfolio: PortfolioState,
        current_price: float,
    ) -> str:
        """
        Combine technical indicators with cycle phase to produce a decision.

        The cycle phase shifts ALL thresholds:
        - In accumulation phases: RSI buy threshold rises (easier to trigger buy)
        - In distribution phases: RSI sell threshold drops (easier to trigger sell)
        - The accumulation score provides a continuous 0-100 overlay

        Returns: "BUY", "SELL", or "HOLD"
        """
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        signal = indicators.get("signal", 0)
        ma_short = indicators.get("ma_short", current_price)
        ma_long = indicators.get("ma_long", current_price)

        # --- Cycle-adjusted RSI thresholds ---
        # Base: buy < 30, sell > 70
        # Accumulation: shift buy UP (e.g., buy < 40), shift sell UP (harder to sell)
        # Distribution: shift buy DOWN (harder to buy), shift sell DOWN (easier to sell)
        rsi_buy = 30 + (cycle_adj.buy_aggressiveness - 1.0) * 10
        rsi_sell = 70 - (1.0 - cycle_adj.sell_reluctance) * 10 if cycle_adj.sell_reluctance < 1.0 else 70 + (cycle_adj.sell_reluctance - 1.0) * 5

        # Clamp to reasonable ranges
        rsi_buy = max(20, min(50, rsi_buy))
        rsi_sell = max(60, min(85, rsi_sell))

        buy_signals = 0
        sell_signals = 0
        total_signals = 5  # RSI, MACD, MA cross, accum score, price vs floor/ceiling

        # RSI signal
        if rsi < rsi_buy:
            buy_signals += 1
        elif rsi > rsi_sell:
            sell_signals += 1

        # MACD signal (bullish crossover)
        if macd > signal and macd < 0:  # Recovering from oversold
            buy_signals += 1
        elif macd < signal and macd > 0:  # Rolling over from overbought
            sell_signals += 1

        # MA crossover
        if ma_short > ma_long:
            buy_signals += 0.5
        elif ma_short < ma_long * 0.98:  # Short MA below long with margin
            sell_signals += 0.5

        # Accumulation score overlay (the cycle phase contribution)
        if accum_score >= 70:
            buy_signals += 1.5  # Strong cycle-based buy bias
        elif accum_score >= 50:
            buy_signals += 0.5
        elif accum_score <= 15:
            sell_signals += 1.5  # Strong cycle-based sell bias
        elif accum_score <= 30:
            sell_signals += 0.5

        # Price vs cycle bounds
        floor_eur = cycle_adj.estimated_floor_eur
        ceiling_eur = cycle_adj.estimated_ceiling_eur
        if current_price < floor_eur * 1.1:  # Within 10% of floor
            buy_signals += 1
        elif current_price > ceiling_eur * 0.95:  # Within 5% of ceiling
            sell_signals += 1

        # --- Decision logic ---
        # Require stronger signal for sells during accumulation phases
        buy_threshold = 2.0
        sell_threshold = 2.0

        # Phase-based threshold adjustment
        if cycle_adj.phase in (
            CyclePhase.POST_HALVING_ACCUMULATION,
            CyclePhase.BEAR_CAPITULATION,
            CyclePhase.PRE_HALVING,
        ):
            buy_threshold = 1.5   # Easier to buy
            sell_threshold = 3.5  # Much harder to sell
        elif cycle_adj.phase in (CyclePhase.EUPHORIA, CyclePhase.DISTRIBUTION):
            buy_threshold = 3.5   # Much harder to buy
            sell_threshold = 1.5  # Easier to sell

        logger.debug(
            f"Signals: buy={buy_signals:.1f} sell={sell_signals:.1f} | "
            f"RSI={rsi:.0f} (buy<{rsi_buy:.0f}, sell>{rsi_sell:.0f}) | "
            f"MACD={macd:.0f} | AccumScore={accum_score:.0f} | "
            f"Thresholds: buy>{buy_threshold:.1f} sell>{sell_threshold:.1f}"
        )

        if buy_signals >= buy_threshold and buy_signals > sell_signals:
            return "BUY"
        elif sell_signals >= sell_threshold and sell_signals > buy_signals:
            return "SELL"
        else:
            return "HOLD"

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def _execute_buy(self, portfolio: PortfolioState, price: float, cycle_adj):
        """Execute a buy with cycle-aware position sizing."""
        btc_amount = self.risk_manager.calculate_position_size(
            available_eur=portfolio.eur_balance,
            current_price=price,
            portfolio=portfolio,
        )

        if btc_amount <= 0:
            logger.debug("Position size too small, skipping buy")
            return

        # Get optimal price from order book
        order_book = self.executor.get_btc_order_book()
        if order_book:
            buy_price = self.executor.get_optimal_price(order_book, "buy")
        else:
            buy_price = price

        logger.info(
            f"📈 BUY {btc_amount:.8f} BTC @ €{buy_price:,.0f} | "
            f"Phase={cycle_adj.phase.value} | DCA={cycle_adj.dca_intensity:.1f}x"
        )

        trade = self.executor.buy(btc_amount, buy_price)
        if trade.is_success:
            self.position.record_trade(trade.btc_filled, trade.price_filled, is_buy=True, fee_eur=trade.fee_eur)
            self.risk_manager.record_trade(is_buy=True)
            logger.info(f"✅ Bought {trade.btc_filled:.8f} BTC @ €{trade.price_filled:,.0f}")

    def _execute_sell(self, pos, price: float, cycle_adj):
        """Execute a sell with cycle-aware profit targets."""
        # Determine sell amount based on phase
        if cycle_adj.phase in (CyclePhase.EUPHORIA, CyclePhase.DISTRIBUTION):
            # In distribution: sell larger portions
            sell_pct = 0.25
        elif cycle_adj.phase == CyclePhase.BEAR_EARLY:
            sell_pct = 0.10
        else:
            sell_pct = 0.05  # Minimal sells in accumulation phases

        btc_to_sell = pos.btc_amount * sell_pct
        if btc_to_sell < 0.0001:
            return

        order_book = self.executor.get_btc_order_book()
        if order_book:
            sell_price = self.executor.get_optimal_price(order_book, "sell")
        else:
            sell_price = price

        logger.info(
            f"📉 SELL {btc_to_sell:.8f} BTC @ €{sell_price:,.0f} | "
            f"Phase={cycle_adj.phase.value}"
        )

        trade = self.executor.sell(btc_to_sell, sell_price)
        if trade.is_success:
            self.position.record_trade(trade.btc_filled, trade.price_filled, is_buy=False, fee_eur=trade.fee_eur)
            self.risk_manager.record_trade(is_buy=False)
            logger.info(f"✅ Sold {trade.btc_filled:.8f} BTC @ €{trade.price_filled:,.0f}")

    def _execute_emergency_sell(self, pos, price: float):
        """Emergency sell — only triggered below golden rule floor."""
        btc_to_sell = pos.btc_amount * 0.5
        if btc_to_sell < 0.0001:
            return

        logger.error(f"🚨 EMERGENCY SELL: {btc_to_sell:.8f} BTC @ €{price:,.0f}")
        trade = self.executor.sell(btc_to_sell, price)
        if trade.is_success:
            self.position.record_trade(trade.btc_filled, trade.price_filled, is_buy=False, fee_eur=trade.fee_eur)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_indicators(self) -> Optional[Dict]:
        """Fetch and calculate technical indicators."""
        try:
            history = self.market_data.price_history(hours=168)  # 1 week
            if len(history) < 50:
                return None

            prices = [p.value for p in history]
            volumes = [p.volume for p in history]

            rsi = calculate_rsi(prices)
            macd_val, signal_val = calculate_macd(prices)
            ma_short = calculate_moving_average(prices, 20)
            ma_long = calculate_moving_average(prices, 50)
            vwap = calculate_vwap(prices, volumes)

            if rsi is None or macd_val is None:
                return None

            return {
                "rsi": rsi,
                "macd": macd_val,
                "signal": signal_val,
                "ma_short": ma_short or prices[-1],
                "ma_long": ma_long or prices[-1],
                "vwap": vwap or prices[-1],
            }
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return None

    def _create_portfolio_state(self, position, current_price: float) -> PortfolioState:
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

    def _log_cycle_status(self, current_price: float = None):
        """Log full cycle summary (called hourly)."""
        summary = self.cycle.get_cycle_summary(current_price)
        logger.info(
            f"═══ CYCLE STATUS ═══\n"
            f"  Cycle {summary['cycle_number']} | "
            f"Day {summary['days_since_halving']} / ~1460 | "
            f"{summary['cycle_progress_pct']:.0f}% complete\n"
            f"  Phase: {summary['current_phase']}\n"
            f"  {summary['phase_description']}\n"
            f"  Floor: €{summary['estimated_floor_eur']:,.0f} | "
            f"Ceiling: €{summary['estimated_ceiling_eur']:,.0f}\n"
            f"  Position mult: {summary['position_size_multiplier']:.1f}x | "
            f"DCA: {summary['dca_intensity']:.1f}x | "
            f"Confidence: {summary['confidence']:.0%}"
        )
