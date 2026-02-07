import time
import signal
import sys

from trading_bot import TradingBot
from data_manager import DataManager
from trade_executor import TradeExecutor
from onchain_analyzer import OnChainAnalyzer
from kraken_api import KrakenAPI
from cycle_analyzer import CycleAnalyzer
from risk_manager import RiskManager
from position_manager import PositionManager
from market_data_service import MarketDataService
from order_manager import OrderManager
from config import API_KEY, API_SECRET, API_DOMAIN, PRICE_HISTORY_FILE, BOT_LOGS_FILE
from logger_config import logger


def main():
    logger.info("Starting Bitcoin accumulation bot (cycle-aware)...")

    # Initialize components
    kraken_api = KrakenAPI(API_KEY, API_SECRET, API_DOMAIN)
    order_manager = OrderManager(kraken_api)
    data_manager = DataManager(PRICE_HISTORY_FILE, BOT_LOGS_FILE)
    trade_executor = TradeExecutor(kraken_api)
    onchain_analyzer = OnChainAnalyzer()
    market_data_service = MarketDataService(exchange_client=kraken_api)
    position_manager = PositionManager()
    eur = kraken_api.get_available_balance('EUR') or 0.0
    btc = kraken_api.get_available_balance('XXBT') or 0.0
    price = kraken_api.get_btc_price() or 50000.0  # Fallback price
    position_manager.update_balance(btc, eur, price)

    # ── NEW: Cycle-aware components ──
    cycle_analyzer = CycleAnalyzer(current_cycle=4)
    risk_manager = RiskManager(cycle_analyzer=cycle_analyzer)

    # Log initial cycle status
    summary = cycle_analyzer.get_cycle_summary()
    logger.info(
        f"Cycle {summary['cycle_number']} | "
        f"Day {summary['days_since_halving']} | "
        f"Phase: {summary['current_phase']} | "
        f"Floor: €{summary['estimated_floor_eur']:,.0f} | "
        f"Ceiling: €{summary['estimated_ceiling_eur']:,.0f}"
    )

    bot = TradingBot(
        market_data=market_data_service,
        cycle_analyzer=cycle_analyzer,
        risk_manager=risk_manager,
        position_manager=position_manager,
        trade_executor=trade_executor,
        data_manager=data_manager,
        onchain_analyzer=onchain_analyzer,
    )

    # Graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, saving state...")
        order_manager._save_order_history()
        logger.info("Order history saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run bot
    while True:
        try:
            bot._execute_trading_cycle()
            current_time = time.time()
            next_run = ((current_time // 900) + 1) * 900
            sleep_time = next_run - current_time
            logger.debug(f"Sleeping {sleep_time:.0f}s until {time.ctime(next_run)}")
            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            order_manager._save_order_history()
            time.sleep(30)


if __name__ == "__main__":
    main()
