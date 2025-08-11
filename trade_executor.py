import time
from typing import Optional, Dict, List
from logger_config import logger

class TradeExecutor:
    def __init__(self, kraken_api):
        self.kraken_api = kraken_api

    def fetch_current_price(self) -> tuple[Optional[float], float]:
        since = int(time.time() - 7200)
        ohlc = self.kraken_api.get_ohlc_data(pair="XXBTZEUR", interval=15, since=since)
        logger.debug(f"Raw OHLC response: {ohlc}")
        if ohlc and len(ohlc) > 0:
            try:
                logger.debug(f"Fetched {len(ohlc)} OHLC candles: {ohlc[-1]}")
                return float(ohlc[-1][4]), float(ohlc[-1][6])
            except (IndexError, TypeError, ValueError) as e:
                logger.error(f"Failed to parse OHLC data: {e}")
                return None, 0.0
        logger.warning("No OHLC data available")
        return None, 0.0

    def get_btc_order_book(self) -> Optional[Dict]:
        return self.kraken_api.get_btc_order_book()

    def get_optimal_price(self, order_book: Dict, side: str) -> Optional[float]:
        optimal_price = self.kraken_api.get_optimal_price(order_book, side)
        logger.debug(f"Optimal {side} price: {optimal_price}")
        return optimal_price

    def execute_trade(self, volume: float, side: str, price: float) -> bool:
        try:
            if side not in ["buy", "sell"]:
                logger.error(f"Invalid trade side: {side}")
                return False
            order_data = {
                "pair": "XXBTZEUR",
                "type": side,
                "ordertype": "limit",
                "price": price,
                "volume": volume
            }
            logger.debug(f"Executing {side} order: {order_data}")
            response = self.kraken_api.query_private("AddOrder", order_data)
            if response.get("error"):
                logger.error(f"Failed to execute {side} order: {response['error']}")
                return False
            logger.info(f"Executed {side} order for {volume} BTC at {price}. Order response: {response['result']}")
            return True
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def get_total_btc_balance(self) -> Optional[float]:
        return self.kraken_api.get_total_btc_balance()

    def get_available_balance(self, currency: str) -> Optional[float]:
        return self.kraken_api.get_available_balance(currency)