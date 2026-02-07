"""
TradeExecutor - Deep Module for Order Execution and Management

OUSTERHOUT PRINCIPLE: Pull complexity downward into modules.

This module handles ALL order complexity:
- Order placement
- Fill tracking
- Partial fills
- Timeouts and retries
- Cancellation
- Fee calculation

PUBLIC INTERFACE (Simple):
    buy(amount, price) -> Trade
    sell(amount, price) -> Trade
    get_order_status(order_id) -> Trade
    cancel_order(order_id) -> bool

PRIVATE IMPLEMENTATION (Complex):
    Order monitoring
    Retry logic
    Status polling
    Fill tracking
    History management
"""

import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from logger_config import logger
from trade import Trade, TradeStatus, TradeType


class TradeExecutor:
    """
    Handles all trade execution and order management.
    
    Philosophy:
    - Public methods (buy/sell) NEVER fail - they ALWAYS return a Trade object
    - Handles all complexity internally (retries, timeouts, monitoring)
    - Returns simple Trade objects with clear status
    - Caller doesn't need error handling
    """

    def __init__(
        self,
        kraken_api,
        order_timeout_seconds: int = 300,
        poll_interval_seconds: float = 2.0,
        max_retry_attempts: int = 3,
    ):
        """
        Initialize trade executor.

        Args:
            kraken_api: Exchange API client
            order_timeout_seconds: How long to monitor orders (default 5 min)
            poll_interval_seconds: How often to check order status
            max_retry_attempts: How many times to retry failed operations
        """
        self.kraken_api = kraken_api
        self.order_timeout_seconds = order_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.max_retry_attempts = max_retry_attempts

        # Order tracking
        self.pending_orders: Dict[str, Trade] = {}
        self.filled_orders: Dict[str, Trade] = {}
        self.cancelled_orders: Dict[str, Trade] = {}
        self.failed_orders: Dict[str, Trade] = {}
        
        self.order_history_file = "./order_history.json"
        self._load_order_history()

    def buy(self, btc_amount: float, limit_price: float) -> Trade:
        """
        Execute a buy order.

        GUARANTEE: This method ALWAYS returns a valid Trade object.
        It NEVER fails or raises exceptions to the caller.

        Args:
            btc_amount: Amount of BTC to buy
            limit_price: Maximum price to pay

        Returns:
            Trade object with execution status
        """
        return self._place_order_with_monitoring(
            trade_type=TradeType.BUY,
            btc_amount=btc_amount,
            price=limit_price,
        )

    def sell(self, btc_amount: float, limit_price: float) -> Trade:
        """
        Execute a sell order.

        GUARANTEE: This method ALWAYS returns a valid Trade object.

        Args:
            btc_amount: Amount of BTC to sell
            limit_price: Minimum price to accept

        Returns:
            Trade object with execution status
        """
        return self._place_order_with_monitoring(
            trade_type=TradeType.SELL,
            btc_amount=btc_amount,
            price=limit_price,
        )

    def get_order_status(self, order_id: str) -> Optional[Trade]:
        """Get status of a specific order."""
        # Check all tracking dictionaries
        if order_id in self.filled_orders:
            return self.filled_orders[order_id]
        if order_id in self.cancelled_orders:
            return self.cancelled_orders[order_id]
        if order_id in self.failed_orders:
            return self.failed_orders[order_id]
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            response = self.kraken_api.query_private("CancelOrder", {"txid": order_id})
            
            if response.get("error"):
                logger.error(f"Failed to cancel order {order_id}: {response['error']}")
                return False
            
            logger.info(f"Successfully cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    # =========================================================================
    # PRIVATE METHODS - Hidden Complexity
    # =========================================================================

    def _place_order_with_monitoring(
        self,
        trade_type: TradeType,
        btc_amount: float,
        price: float,
    ) -> Trade:
        """
        Place an order and monitor until filled or timeout.

        This method handles all the complexity:
        - Retry logic
        - Status polling
        - Timeout handling
        - Partial fills
        - Always returns a Trade object
        """
        if btc_amount <= 0 or price <= 0:
            return self._create_failed_trade(
                trade_type=trade_type,
                btc_amount=btc_amount,
                price=price,
                reason="Invalid amount or price",
            )

        # Place order with retries
        order_id = self._place_order_with_retry(
            trade_type=trade_type,
            btc_amount=btc_amount,
            price=price,
        )

        if not order_id:
            return self._create_failed_trade(
                trade_type=trade_type,
                btc_amount=btc_amount,
                price=price,
                reason="Failed to place order after retries",
            )

        # Monitor order until filled or timeout
        trade = self._monitor_order(
            order_id=order_id,
            trade_type=trade_type,
            btc_amount=btc_amount,
            price=price,
        )

        # Save to history
        self._save_order_to_history(trade)

        return trade

    def _place_order_with_retry(
        self,
        trade_type: TradeType,
        btc_amount: float,
        price: float,
    ) -> Optional[str]:
        """Place order with exponential backoff retry."""
        for attempt in range(self.max_retry_attempts):
            try:
                order_data = {
                    "pair": "XXBTZEUR",
                    "type": trade_type.value,
                    "ordertype": "limit",
                    "price": price,
                    "volume": btc_amount,
                }

                response = self.kraken_api.query_private("AddOrder", order_data)

                if response.get("error"):
                    logger.warning(
                        f"Order placement attempt {attempt + 1} failed: {response['error']}"
                    )
                    if attempt < self.max_retry_attempts - 1:
                        backoff = 2 ** attempt
                        time.sleep(backoff)
                    continue

                # Extract order ID
                order_ids = response.get("result", {}).get("txid", [])
                if order_ids:
                    order_id = order_ids[0]
                    logger.info(
                        f"Order placed successfully: {order_id} "
                        f"({btc_amount} BTC @ €{price})"
                    )
                    return order_id

            except Exception as e:
                logger.error(f"Error placing order (attempt {attempt + 1}): {e}")
                if attempt < self.max_retry_attempts - 1:
                    backoff = 2 ** attempt
                    time.sleep(backoff)

        return None

    def _monitor_order(
        self,
        order_id: str,
        trade_type: TradeType,
        btc_amount: float,
        price: float,
    ) -> Trade:
        """Monitor order status until filled or timeout."""
        start_time = datetime.now()
        timeout = timedelta(seconds=self.order_timeout_seconds)

        while True:
            # Check timeout
            if datetime.now() - start_time > timeout:
                logger.warning(f"Order {order_id} timed out")
                
                # Try to cancel
                self.cancel_order(order_id)
                
                return self._create_cancelled_trade(
                    order_id=order_id,
                    trade_type=trade_type,
                    btc_amount=btc_amount,
                    price=price,
                    reason="Order timeout",
                )

            try:
                # Query order status
                query_response = self.kraken_api.query_private(
                    "QueryOrders", {"txid": order_id}
                )

                if query_response.get("error"):
                    logger.warning(f"Error querying order {order_id}: {query_response['error']}")
                    time.sleep(self.poll_interval_seconds)
                    continue

                order_info = query_response.get("result", {}).get(order_id, {})

                if not order_info:
                    logger.debug(f"No order info for {order_id}, waiting...")
                    time.sleep(self.poll_interval_seconds)
                    continue

                status = order_info.get("status", "unknown")
                filled_amount = float(order_info.get("vol_exec", 0))
                fee = float(order_info.get("fee", 0))
                
                logger.debug(
                    f"Order {order_id} status: {status}, "
                    f"filled: {filled_amount}/{btc_amount} BTC"
                )

                # Check if filled
                if status == "closed":
                    logger.info(f"Order {order_id} filled: {filled_amount} BTC")
                    
                    return self._create_filled_trade(
                        order_id=order_id,
                        trade_type=trade_type,
                        btc_amount=btc_amount,
                        btc_filled=filled_amount,
                        price=price,
                        fee_eur=fee,
                    )

                # Check if partially filled
                if filled_amount > 0:
                    logger.info(
                        f"Order {order_id} partially filled: {filled_amount}/{btc_amount} BTC"
                    )

                # Wait before next poll
                time.sleep(self.poll_interval_seconds)

            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                time.sleep(self.poll_interval_seconds)

    def _create_failed_trade(
        self,
        trade_type: TradeType,
        btc_amount: float,
        price: float,
        reason: str,
    ) -> Trade:
        """Create a failed trade object."""
        # Normalize invalid values to 0
        btc_amount = max(0, btc_amount)
        price = max(0, price)
        
        trade = Trade(
            trade_id="FAILED-" + str(int(time.time() * 1000)),
            trade_type=trade_type,
            status=TradeStatus.FAILED,
            btc_amount=btc_amount,
            btc_filled=0.0,
            price_limit=price,
            price_filled=0.0,
            total_cost=0.0,
            fee_eur=0.0,
            created_at=datetime.now(),
            filled_at=None,
            reason=reason,
        )
        self.failed_orders[trade.trade_id] = trade
        logger.error(f"Trade failed: {reason}")
        return trade

    def _create_cancelled_trade(
        self,
        order_id: str,
        trade_type: TradeType,
        btc_amount: float,
        price: float,
        reason: str,
    ) -> Trade:
        """Create a cancelled trade object."""
        trade = Trade(
            trade_id=order_id,
            trade_type=trade_type,
            status=TradeStatus.CANCELLED,
            btc_amount=btc_amount,
            btc_filled=0.0,
            price_limit=price,
            price_filled=0.0,
            total_cost=0.0,
            fee_eur=0.0,
            created_at=datetime.now(),
            filled_at=None,
            reason=reason,
        )
        self.cancelled_orders[order_id] = trade
        logger.warning(f"Trade cancelled: {reason}")
        return trade

    def _create_filled_trade(
        self,
        order_id: str,
        trade_type: TradeType,
        btc_amount: float,
        btc_filled: float,
        price: float,
        fee_eur: float,
    ) -> Trade:
        """Create a filled trade object."""
        status = (
            TradeStatus.FILLED
            if btc_filled >= btc_amount * 0.999
            else TradeStatus.PARTIALLY_FILLED
        )
        total_cost = btc_filled * price

        trade = Trade(
            trade_id=order_id,
            trade_type=trade_type,
            status=status,
            btc_amount=btc_amount,
            btc_filled=btc_filled,
            price_limit=price,
            price_filled=price if btc_filled > 0 else 0.0,
            total_cost=total_cost,
            fee_eur=fee_eur,
            created_at=datetime.now(),
            filled_at=datetime.now(),
            reason="Order filled",
        )
        self.filled_orders[order_id] = trade  # ← Changed from trade_id to order_id
        logger.info(f"Trade executed: {btc_filled} BTC @ €{price}")
        return trade

    # =========================================================================
    # History Management
    # =========================================================================

    def _load_order_history(self):
        """Load order history from file."""
        try:
            if Path(self.order_history_file).exists():
                with open(self.order_history_file, 'r') as f:
                    data = json.load(f)
                    logger.info(
                        f"Loaded {len(data.get('filled_orders', {}))} filled orders, "
                        f"{len(data.get('cancelled_orders', {}))} cancelled orders"
                    )
        except Exception as e:
            logger.error(f"Failed to load order history: {e}")

    def _save_order_to_history(self, trade: Trade):
        """Save trade to order history."""
        try:
            history = {}
            if Path(self.order_history_file).exists():
                with open(self.order_history_file, 'r') as f:
                    history = json.load(f)

            # Add trade to appropriate section
            section = "filled_orders" if trade.status == TradeStatus.FILLED else "cancelled_orders"
            if section not in history:
                history[section] = {}

            history[section][trade.trade_id] = {
                "type": trade.trade_type.value,
                "btc_amount": trade.btc_filled,
                "price": trade.price_filled,
                "cost": trade.total_cost,
                "fee": trade.fee_eur,
                "timestamp": trade.filled_at.isoformat() if trade.filled_at else None,
            }

            with open(self.order_history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save order to history: {e}")

    # =========================================================================
    # Legacy Methods (for backward compatibility)
    # =========================================================================

    def fetch_current_price(self) -> tuple[Optional[float], float]:
        """Legacy method - fetch current price from OHLC."""
        since = int(time.time() - 7200)
        ohlc = self.kraken_api.get_ohlc_data(pair="XXBTZEUR", interval=15, since=since)
        if ohlc and len(ohlc) > 0:
            try:
                return float(ohlc[-1][4]), float(ohlc[-1][6])
            except (IndexError, TypeError, ValueError) as e:
                logger.error(f"Failed to parse OHLC data: {e}")
                return None, 0.0
        return None, 0.0

    def get_btc_order_book(self) -> Optional[Dict]:
        """Legacy method - get order book."""
        return self.kraken_api.get_order_book()

    def get_optimal_price(self, order_book: Dict, side: str) -> Optional[float]:
        """Legacy method - get optimal price."""
        optimal_price = self.kraken_api.get_optimal_price(order_book, side)
        logger.debug(f"Optimal {side} price: {optimal_price}")
        return optimal_price

    def execute_trade(self, volume: float, side: str, price: float) -> bool:
        """Legacy method - execute trade."""
        if side.lower() == "buy":
            trade = self.buy(volume, price)
        else:
            trade = self.sell(volume, price)
        return trade.is_success

    def get_total_btc_balance(self) -> Optional[float]:
        """Legacy method - get total BTC balance."""
        return self.kraken_api.get_total_btc_balance()

    def get_available_balance(self, currency: str) -> Optional[float]:
        """Legacy method - get available balance."""
        return self.kraken_api.get_available_balance(currency)