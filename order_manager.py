import time
import json
from typing import Dict, Optional, List
from datetime import datetime
from logger_config import logger


class OrderManager:
    """Manages order lifecycle including placement, monitoring, and cancellation"""
    
    def __init__(self, kraken_api):
        self.kraken_api = kraken_api
        self.pending_orders = {}
        self.filled_orders = {}
        self.cancelled_orders = {}
        self.order_history_file = "./order_history.json"
        self._load_order_history()
    
    def _load_order_history(self):
        """Load order history from file"""
        try:
            with open(self.order_history_file, 'r') as f:
                data = json.load(f)
                self.filled_orders = data.get('filled_orders', {})
                self.cancelled_orders = data.get('cancelled_orders', {})
                logger.info(f"Loaded {len(self.filled_orders)} filled orders and {len(self.cancelled_orders)} cancelled orders")
        except FileNotFoundError:
            logger.info("No order history file found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load order history: {e}")
    
    def _save_order_history(self):
        """Save order history to file"""
        try:
            data = {
                'filled_orders': self.filled_orders,
                'cancelled_orders': self.cancelled_orders,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.order_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save order history: {e}")
    
    def place_limit_order_with_timeout(self, volume: float, side: str, price: float, 
                                     timeout: int = 300, post_only: bool = True) -> Optional[str]:
        """
        Place a limit order with automatic cancellation after timeout
        
        Args:
            volume: Amount of BTC to trade
            side: 'buy' or 'sell'
            price: Limit price
            timeout: Seconds before auto-cancellation (default 5 minutes)
            post_only: If True, order will only make liquidity (lower fees)
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_data = {
                "pair": "XXBTZEUR",
                "type": side,
                "ordertype": "limit",
                "price": str(price),
                "volume": str(volume),
                "validate": False
            }
            
            if post_only:
                order_data["oflags"] = "post"
            
            logger.info(f"Placing {side} order: {volume} BTC at €{price}")
            response = self.kraken_api.query_private("AddOrder", order_data)
            
            if response.get("error"):
                logger.error(f"Failed to place order: {response['error']}")
                return None
            
            order_result = response.get("result", {})
            order_ids = order_result.get("txid", [])
            
            if not order_ids:
                logger.error("No order ID returned")
                return None
            
            order_id = order_ids[0]
            
            # Store order details
            self.pending_orders[order_id] = {
                'timestamp': time.time(),
                'timeout': timeout,
                'volume': volume,
                'side': side,
                'price': price,
                'status': 'pending',
                'post_only': post_only
            }
            
            logger.info(f"Order placed successfully: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def check_order_status(self, order_id: str) -> Optional[Dict]:
        """Check the status of a specific order"""
        try:
            response = self.kraken_api.query_private("QueryOrders", {"txid": order_id})
            
            if response.get("error"):
                logger.error(f"Failed to query order {order_id}: {response['error']}")
                return None
            
            order_info = response.get("result", {}).get(order_id)
            if not order_info:
                logger.warning(f"No info found for order {order_id}")
                return None
            
            return {
                'status': order_info.get('status'),
                'volume_executed': float(order_info.get('vol_exec', 0)),
                'price': float(order_info.get('price', 0)),
                'average_price': float(order_info.get('avg_price', 0)) if order_info.get('avg_price') else None,
                'fee': float(order_info.get('fee', 0)),
                'time': order_info.get('opentm', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to check order status: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            logger.info(f"Attempting to cancel order {order_id}")
            response = self.kraken_api.query_private("CancelOrder", {"txid": order_id})
            
            if response.get("error"):
                if 'EOrder:Unknown order' in response['error']:
                    logger.warning(f"Order {order_id} already canceled or unknown on exchange")
                    if order_id in self.pending_orders:
                        order_info = self.pending_orders.pop(order_id)
                        order_info['cancelled_at'] = time.time()
                        order_info['status'] = 'cancelled'
                        self.cancelled_orders[order_id] = order_info
                        self._save_order_history()
                    return True
                logger.error(f"Failed to cancel order {order_id}: {response['error']}")
                return False
            
            # Move to cancelled orders
            if order_id in self.pending_orders:
                order_info = self.pending_orders.pop(order_id)
                order_info['cancelled_at'] = time.time()
                order_info['status'] = 'cancelled'
                self.cancelled_orders[order_id] = order_info
                self._save_order_history()
            
            logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def check_and_update_orders(self) -> Dict[str, List[str]]:
        """
        Check all pending orders and update their status
        Cancel orders that have exceeded their timeout
        
        Returns:
            Dict with lists of filled and cancelled order IDs
        """
        results = {
            'filled': [],
            'cancelled': [],
            'partial': []
        }
        
        try:
            # Get current open orders
            open_orders_response = self.kraken_api.query_private("OpenOrders", {})
            logger.debug(f"OpenOrders response: {open_orders_response}")
            
            if open_orders_response.get("error"):
                logger.error(f"Failed to fetch OpenOrders: {open_orders_response['error']}")
                return results
            
            open_orders = open_orders_response.get("result", {}).get("open", {})
            
            # Check for recent trades
            recent_trades = {}
            try:
                trades_response = self.kraken_api.query_private("TradesHistory", {
                    "start": int(time.time() - 3600)
                })
                
                if not trades_response.get("error") and "result" in trades_response:
                    trades = trades_response["result"].get("trades", {})
                    for trade_id, trade_info in trades.items():
                        order_id = trade_info.get("ordertxid")
                        if order_id in self.pending_orders:
                            if order_id not in recent_trades:
                                recent_trades[order_id] = []
                            recent_trades[order_id].append(trade_info)
                            
                    logger.debug(f"Found {len(recent_trades)} orders with recent trades")
            except Exception as e:
                logger.error(f"Failed to fetch recent trades: {e}")
            
            # Process each pending order
            orders_to_remove = []
            
            for order_id in list(self.pending_orders.keys()):
                order_info = self.pending_orders[order_id]
                current_time = time.time()
                order_age = current_time - order_info['timestamp']
                
                # Check if executed via recent trades
                if order_id in recent_trades:
                    trades = recent_trades[order_id]
                    total_volume = sum(float(trade.get("vol", 0)) for trade in trades)
                    total_cost = sum(float(trade.get("vol", 0)) * float(trade.get("price", 0)) for trade in trades)
                    total_fee = sum(float(trade.get("fee", 0)) for trade in trades)
                    avg_price = total_cost / total_volume if total_volume > 0 else order_info['price']
                    
                    order_info.update({
                        'filled_at': current_time,
                        'status': 'filled',
                        'executed_volume': total_volume,
                        'average_price': avg_price,
                        'fee': total_fee
                    })
                    
                    self.filled_orders[order_id] = order_info
                    orders_to_remove.append(order_id)
                    results['filled'].append(order_id)
                    logger.info(f"✅ Order {order_id} FILLED via TradesHistory: {total_volume:.8f} BTC @ €{avg_price:.2f}")
                    self._save_order_history()
                    continue
                
                # Check timeout
                if order_age > order_info['timeout']:
                    logger.info(f"Order {order_id} timed out after {order_age:.0f}s, cancelling...")
                    if self.cancel_order(order_id):
                        results['cancelled'].append(order_id)
                    continue
                
                # Check open orders
                if order_id in open_orders:
                    kraken_order = open_orders[order_id]
                    executed_volume = float(kraken_order.get('vol_exec', 0))
                    
                    if executed_volume > 0:
                        order_info['executed_volume'] = executed_volume
                        self.pending_orders[order_id] = order_info
                        results['partial'].append(order_id)
                        logger.info(f"📊 Order {order_id} partially filled: {executed_volume:.8f}/{order_info['volume']:.8f} BTC")
                    continue
                
                # Targeted status query
                try:
                    status_response = self.kraken_api.query_private("QueryOrders", {"txid": order_id})
                    
                    if not status_response.get("error") and "result" in status_response:
                        order_details = status_response["result"].get(order_id, {})
                        order_status = order_details.get("status")
                        
                        if order_status == "closed":
                            executed_vol = float(order_details.get("vol_exec", order_info['volume']))
                            avg_price = float(order_details.get("price", order_info['price']))
                            fee = float(order_details.get("fee", 0))
                            
                            order_info.update({
                                'filled_at': current_time,
                                'status': 'filled',
                                'executed_volume': executed_vol,
                                'average_price': avg_price,
                                'fee': fee
                            })
                            
                            self.filled_orders[order_id] = order_info
                            orders_to_remove.append(order_id)
                            results['filled'].append(order_id)
                            logger.info(f"✅ Order {order_id} FILLED via QueryOrders: {executed_vol:.8f} BTC @ €{avg_price:.2f}")
                            self._save_order_history()
                            continue
                            
                        elif order_status == "canceled":
                            order_info.update({
                                'cancelled_at': current_time,
                                'status': 'cancelled'
                            })
                            self.cancelled_orders[order_id] = order_info
                            orders_to_remove.append(order_id)
                            results['cancelled'].append(order_id)
                            logger.warning(f"❌ Order {order_id} was cancelled on exchange")
                            self._save_order_history()
                            continue
                    
                    if order_age > 600:
                        logger.warning(f"Order {order_id} not found, marking as cancelled after {order_age:.0f}s")
                        order_info.update({
                            'cancelled_at': current_time,
                            'status': 'cancelled'
                        })
                        self.cancelled_orders[order_id] = order_info
                        orders_to_remove.append(order_id)
                        results['cancelled'].append(order_id)
                        self._save_order_history()
                    else:
                        logger.debug(f"Order {order_id} not found but recent (age: {order_age:.0f}s)")
                        
                except Exception as e:
                    logger.error(f"Error querying order {order_id}: {e}")
            
            for order_id in orders_to_remove:
                self.pending_orders.pop(order_id, None)
                
        except Exception as e:
            logger.error(f"Error in check_and_update_orders: {e}", exc_info=True)
        
        return results
    
    def get_pending_orders(self) -> Dict[str, Dict]:
        return self.pending_orders.copy()
    
    def get_filled_orders(self, hours: int = 24) -> Dict[str, Dict]:
        cutoff_time = time.time() - (hours * 3600)
        return {
            order_id: order_info 
            for order_id, order_info in self.filled_orders.items()
            if order_info.get('filled_at', 0) > cutoff_time
        }
    
    def calculate_average_fill_price(self, side: str, hours: int = 24) -> Optional[float]:
        recent_fills = self.get_filled_orders(hours)
        
        side_fills = [
            order for order in recent_fills.values()
            if order.get('side') == side and order.get('average_price')
        ]
        
        if not side_fills:
            return None
        
        total_volume = sum(order.get('executed_volume', 0) for order in side_fills)
        if total_volume == 0:
            return None
        
        weighted_sum = sum(
            order.get('average_price', 0) * order.get('executed_volume', 0)
            for order in side_fills
        )
        
        return weighted_sum / total_volume
    
    def get_order_statistics(self) -> Dict:
        total_filled = len(self.filled_orders)
        total_cancelled = len(self.cancelled_orders)
        total_orders = total_filled + total_cancelled
        
        if total_orders == 0:
            return {
                'fill_rate': 0.0,
                'avg_time_to_fill': 0,
                'total_fees_paid': 0.0,
                'total_filled_orders': 0,
                'total_cancelled_orders': 0
            }
        
        fill_rate = total_filled / total_orders if total_orders > 0 else 0.0
        
        fill_times = []
        for order in self.filled_orders.values():
            if 'filled_at' in order and 'timestamp' in order:
                fill_times.append(order['filled_at'] - order['timestamp'])
        
        avg_time_to_fill = sum(fill_times) / len(fill_times) if fill_times else 0
        
        total_fees = sum(
            order.get('fee', 0) for order in self.filled_orders.values()
        )
        
        return {
            'fill_rate': fill_rate,
            'avg_time_to_fill': avg_time_to_fill,
            'total_fees_paid': total_fees,
            'total_filled_orders': total_filled,
            'total_cancelled_orders': total_cancelled
        }
    
    def should_use_market_order(self, side: str, spread_percentage: float) -> bool:
        max_spread_for_market = 0.1
        
        stats = self.get_order_statistics()
        fill_rate = stats.get('fill_rate', 0)
        
        if fill_rate < 0.5 and spread_percentage < max_spread_for_market:
            logger.info(f"Low fill rate ({fill_rate:.1%}) and tight spread ({spread_percentage:.2%}%), suggesting market order")
            return True
        
        return False