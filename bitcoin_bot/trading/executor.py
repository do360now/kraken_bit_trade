"""
Unified Trade Executor - Fixed API Integration
Resolves the critical API mismatch between Kraken and Bitvavo
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    KRAKEN = "kraken"
    BITVAVO = "bitvavo"  # For future support

@dataclass
class OrderResult:
    """Standardized order result"""
    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class MarketData:
    """Standardized market data"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

class UnifiedTradeExecutor:
    """
    Unified trade executor that works with Kraken API
    Fixes the critical API integration issues
    """
    
    def __init__(self, api_client, exchange_type: ExchangeType = ExchangeType.KRAKEN):
        self.api_client = api_client
        self.exchange_type = exchange_type
        
        # Exchange-specific configurations
        self.exchange_config = self._get_exchange_config()
        
        # Performance optimizations
        self.market_data_cache = {}
        self.cache_duration = 30  # seconds
        self.last_order_book_fetch = 0
        self.order_book_cache = None
        
        # Order tracking
        self.pending_orders = {}
        self.order_history = deque(maxlen=1000)
        
        # Rate limiting
        self.last_api_call = 0
        self.min_api_interval = 1.0  # seconds between API calls
        
        logger.info(f"Unified Trade Executor initialized for {exchange_type.value}")
    
    def _get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange-specific configuration"""
        configs = {
            ExchangeType.KRAKEN: {
                "pair": "XXBTZEUR",
                "base_currency": "XXBT",  # Kraken's BTC symbol
                "quote_currency": "ZEUR", # Kraken's EUR symbol
                "min_order_size": 0.0001,
                "price_precision": 1,
                "volume_precision": 8,
                "maker_fee": 0.0026,
                "taker_fee": 0.0026,
                "order_types": ["market", "limit"],
                "api_rate_limit": 1.0  # seconds between calls
            },
            ExchangeType.BITVAVO: {
                "pair": "BTC-EUR",
                "base_currency": "BTC",
                "quote_currency": "EUR",
                "min_order_size": 0.0001,
                "price_precision": 2,
                "volume_precision": 8,
                "maker_fee": 0.0015,
                "taker_fee": 0.0025,
                "order_types": ["market", "limit"],
                "api_rate_limit": 0.5
            }
        }
        
        return configs[self.exchange_type]
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.min_api_interval:
            sleep_time = self.min_api_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    async def fetch_current_price(self) -> Tuple[Optional[float], Optional[float]]:
        """Fetch current price and volume with caching"""
        try:
            cache_key = f"ticker_{self.exchange_config['pair']}"
            current_time = time.time()
            
            # Check cache
            if (cache_key in self.market_data_cache and 
                current_time - self.market_data_cache[cache_key]["timestamp"] < self.cache_duration):
                cached_data = self.market_data_cache[cache_key]
                return cached_data["price"], cached_data["volume"]
            
            await self._rate_limit()
            
            if self.exchange_type == ExchangeType.KRAKEN:
                price = await self._fetch_kraken_price()
                volume = await self._fetch_kraken_volume()
            else:
                # Future: Bitvavo implementation
                raise NotImplementedError("Bitvavo support not yet implemented")
            
            if price is not None:
                # Update cache
                self.market_data_cache[cache_key] = {
                    "price": price,
                    "volume": volume,
                    "timestamp": current_time
                }
                
                logger.debug(f"Fetched current price: €{price:.2f}, volume: {volume:.2f}")
                return price, volume
            
            return None, None
            
        except Exception as e:
            logger.error(f"Failed to fetch current price: {e}")
            return None, None
    
    async def _fetch_kraken_price(self) -> Optional[float]:
        """Fetch price from Kraken API"""
        try:
            if hasattr(self.api_client, 'get_btc_price'):
                # Async API
                if asyncio.iscoroutinefunction(self.api_client.get_btc_price):
                    return await self.api_client.get_btc_price()
                else:
                    # Sync API - run in executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self.api_client.get_btc_price)
            else:
                # Fallback to ticker method
                ticker = await self._get_ticker()
                return float(ticker['c'][0]) if ticker and 'c' in ticker else None
                
        except Exception as e:
            logger.error(f"Kraken price fetch failed: {e}")
            return None
    
    async def _fetch_kraken_volume(self) -> Optional[float]:
        """Fetch volume from Kraken API"""
        try:
            if hasattr(self.api_client, 'get_market_volume'):
                if asyncio.iscoroutinefunction(self.api_client.get_market_volume):
                    return await self.api_client.get_market_volume()
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self.api_client.get_market_volume)
            else:
                ticker = await self._get_ticker()
                return float(ticker['v'][1]) if ticker and 'v' in ticker else None
                
        except Exception as e:
            logger.error(f"Kraken volume fetch failed: {e}")
            return None
    
    async def _get_ticker(self) -> Optional[Dict]:
        """Get ticker data from exchange"""
        try:
            if hasattr(self.api_client, 'query_public'):
                # Sync Kraken API
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self.api_client.query_public, 
                    "Ticker", 
                    {"pair": self.exchange_config["pair"]}
                )
                return result.get('result', {}).get(self.exchange_config["pair"])
            else:
                logger.error("Unsupported API client")
                return None
                
        except Exception as e:
            logger.error(f"Ticker fetch failed: {e}")
            return None
    
    async def get_order_book(self, depth: int = 10) -> Optional[Dict]:
        """Get order book with caching"""
        try:
            current_time = time.time()
            
            # Use cached order book if recent
            if (self.order_book_cache and 
                current_time - self.last_order_book_fetch < 15):  # 15 second cache
                return self.order_book_cache
            
            await self._rate_limit()
            
            if self.exchange_type == ExchangeType.KRAKEN:
                order_book = await self._get_kraken_order_book(depth)
            else:
                raise NotImplementedError("Bitvavo support not yet implemented")
            
            if order_book:
                self.order_book_cache = order_book
                self.last_order_book_fetch = current_time
                
                logger.debug(f"Order book fetched: {len(order_book.get('bids', []))} bids, {len(order_book.get('asks', []))} asks")
            
            return order_book
            
        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")
            return None
    
    async def _get_kraken_order_book(self, depth: int) -> Optional[Dict]:
        """Get order book from Kraken"""
        try:
            if hasattr(self.api_client, 'get_btc_order_book'):
                if asyncio.iscoroutinefunction(self.api_client.get_btc_order_book):
                    return await self.api_client.get_btc_order_book()
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self.api_client.get_btc_order_book)
            else:
                # Fallback to depth query
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.api_client.query_public,
                    "Depth",
                    {"pair": self.exchange_config["pair"], "count": depth}
                )
                return result.get('result', {}).get(self.exchange_config["pair"])
                
        except Exception as e:
            logger.error(f"Kraken order book failed: {e}")
            return None
    
    def calculate_optimal_price(self, order_book: Dict, side: str, 
                              aggression: float = 0.5) -> Optional[float]:
        """Calculate optimal price with configurable aggression"""
        try:
            if side.lower() == "buy":
                asks = order_book.get("asks", [])
                if not asks:
                    return None
                
                best_ask = float(asks[0][0])
                
                if aggression <= 0.3:  # Conservative - place well below best ask
                    optimal_price = best_ask * (1 - 0.001)  # 0.1% below
                elif aggression <= 0.7:  # Moderate - slightly below best ask
                    optimal_price = best_ask * (1 - 0.0005)  # 0.05% below
                else:  # Aggressive - at or above best ask
                    optimal_price = best_ask
                
            else:  # sell
                bids = order_book.get("bids", [])
                if not bids:
                    return None
                
                best_bid = float(bids[0][0])
                
                if aggression <= 0.3:  # Conservative - place well above best bid
                    optimal_price = best_bid * (1 + 0.001)  # 0.1% above
                elif aggression <= 0.7:  # Moderate - slightly above best bid
                    optimal_price = best_bid * (1 + 0.0005)  # 0.05% above
                else:  # Aggressive - at or below best bid
                    optimal_price = best_bid
            
            # Round to exchange precision
            precision = self.exchange_config["price_precision"]
            optimal_price = round(optimal_price, precision)
            
            logger.debug(f"Optimal {side} price: €{optimal_price:.2f} (aggression: {aggression})")
            return optimal_price
            
        except Exception as e:
            logger.error(f"Optimal price calculation failed: {e}")
            return None
    
    async def get_balances(self) -> Dict[str, float]:
        """Get account balances"""
        try:
            await self._rate_limit()
            
            if self.exchange_type == ExchangeType.KRAKEN:
                balances = await self._get_kraken_balances()
            else:
                raise NotImplementedError("Bitvavo support not yet implemented")
            
            return balances
            
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return {}
    
    async def _get_kraken_balances(self) -> Dict[str, float]:
        """Get balances from Kraken"""
        try:
            # Try direct methods first
            btc_balance = None
            eur_balance = None
            
            if hasattr(self.api_client, 'get_total_btc_balance'):
                if asyncio.iscoroutinefunction(self.api_client.get_total_btc_balance):
                    btc_balance = await self.api_client.get_total_btc_balance()
                else:
                    loop = asyncio.get_event_loop()
                    btc_balance = await loop.run_in_executor(None, self.api_client.get_total_btc_balance)
            
            if hasattr(self.api_client, 'get_available_balance'):
                if asyncio.iscoroutinefunction(self.api_client.get_available_balance):
                    eur_balance = await self.api_client.get_available_balance("EUR")
                else:
                    loop = asyncio.get_event_loop()
                    eur_balance = await loop.run_in_executor(None, self.api_client.get_available_balance, "EUR")
            
            # Fallback to balance query
            if btc_balance is None or eur_balance is None:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.api_client.query_private, "Balance", {}
                )
                
                if result.get('error'):
                    logger.error(f"Balance query error: {result['error']}")
                    return {}
                
                balance_data = result.get('result', {})
                btc_balance = float(balance_data.get('XXBT', 0)) if btc_balance is None else btc_balance
                eur_balance = float(balance_data.get('ZEUR', 0)) if eur_balance is None else eur_balance
            
            return {
                "BTC": btc_balance or 0.0,
                "EUR": eur_balance or 0.0
            }
            
        except Exception as e:
            logger.error(f"Kraken balance fetch failed: {e}")
            return {"BTC": 0.0, "EUR": 0.0}
    
    def get_total_btc_balance(self) -> Optional[float]:
        """Sync wrapper for BTC balance"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_get_btc_balance())
                    return future.result(timeout=10)
            else:
                return asyncio.run(self._async_get_btc_balance())
        except Exception as e:
            logger.error(f"Sync BTC balance failed: {e}")
            return None
    
    async def _async_get_btc_balance(self) -> Optional[float]:
        """Async helper for BTC balance"""
        balances = await self.get_balances()
        return balances.get("BTC", 0.0)
    
    def get_available_balance(self, currency: str) -> Optional[float]:
        """Sync wrapper for available balance"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_get_balance(currency))
                    return future.result(timeout=10)
            else:
                return asyncio.run(self._async_get_balance(currency))
        except Exception as e:
            logger.error(f"Sync {currency} balance failed: {e}")
            return None
    
    async def _async_get_balance(self, currency: str) -> Optional[float]:
        """Async helper for specific currency balance"""
        balances = await self.get_balances()
        return balances.get(currency, 0.0)
    
    async def place_order(self, side: str, order_type: str, volume: float, 
                         price: Optional[float] = None, **kwargs) -> OrderResult:
        """Place order with comprehensive error handling"""
        try:
            # Validate inputs
            validation_result = self._validate_order_params(side, order_type, volume, price)
            if not validation_result.success:
                return validation_result
            
            await self._rate_limit()
            
            if self.exchange_type == ExchangeType.KRAKEN:
                result = await self._place_kraken_order(side, order_type, volume, price, **kwargs)
            else:
                raise NotImplementedError("Bitvavo support not yet implemented")
            
            # Track order
            if result.success and result.order_id:
                self.pending_orders[result.order_id] = {
                    "side": side,
                    "type": order_type,
                    "volume": volume,
                    "price": price,
                    "timestamp": datetime.now(),
                    "status": "pending"
                }
                
                self.order_history.append({
                    "order_id": result.order_id,
                    "side": side,
                    "volume": volume,
                    "price": price,
                    "timestamp": datetime.now(),
                    "success": True
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def _validate_order_params(self, side: str, order_type: str, volume: float, 
                              price: Optional[float]) -> OrderResult:
        """Validate order parameters"""
        # Check side
        if side.lower() not in ["buy", "sell"]:
            return OrderResult(success=False, error_message=f"Invalid side: {side}")
        
        # Check order type
        if order_type.lower() not in self.exchange_config["order_types"]:
            return OrderResult(success=False, error_message=f"Unsupported order type: {order_type}")
        
        # Check volume
        min_size = self.exchange_config["min_order_size"]
        if volume < min_size:
            return OrderResult(success=False, error_message=f"Volume {volume} below minimum {min_size}")
        
        # Check price for limit orders
        if order_type.lower() == "limit" and price is None:
            return OrderResult(success=False, error_message="Price required for limit orders")
        
        if price is not None and price <= 0:
            return OrderResult(success=False, error_message=f"Invalid price: {price}")
        
        return OrderResult(success=True)
    
    async def _place_kraken_order(self, side: str, order_type: str, volume: float, 
                                 price: Optional[float], **kwargs) -> OrderResult:
        """Place order on Kraken"""
        try:
            order_data = {
                'pair': self.exchange_config["pair"],
                'type': side.lower(),
                'ordertype': order_type.lower(),
                'volume': str(volume)
            }
            
            if price is not None:
                order_data['price'] = str(price)
            
            # Add additional parameters
            order_data.update(kwargs)
            
            # Execute order
            if hasattr(self.api_client, 'place_order'):
                if asyncio.iscoroutinefunction(self.api_client.place_order):
                    result = await self.api_client.place_order(**order_data)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.api_client.place_order, **order_data)
            else:
                # Fallback to direct API call
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.api_client.query_private, "AddOrder", order_data
                )
            
            if result.get('error'):
                return OrderResult(
                    success=False,
                    error_message=f"Kraken error: {result['error']}"
                )
            
            # Extract order ID
            order_ids = result.get('result', {}).get('txid', [])
            order_id = order_ids[0] if order_ids else None
            
            if order_id:
                logger.info(f"Order placed successfully: {order_id}")
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    price=price,
                    volume=volume,
                    timestamp=datetime.now()
                )
            else:
                return OrderResult(
                    success=False,
                    error_message="No order ID returned"
                )
                
        except Exception as e:
            logger.error(f"Kraken order placement failed: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def execute_trade(self, volume: float, side: str, price: float) -> bool:
        """Sync wrapper for trade execution"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_execute_trade(volume, side, price))
                    return future.result(timeout=30)
            else:
                return asyncio.run(self._async_execute_trade(volume, side, price))
        except Exception as e:
            logger.error(f"Sync trade execution failed: {e}")
            return False
    
    async def _async_execute_trade(self, volume: float, side: str, price: float) -> bool:
        """Async trade execution"""
        try:
            result = await self.place_order(side, "limit", volume, price)
            return result.success
        except Exception as e:
            logger.error(f"Async trade execution failed: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            await self._rate_limit()
            
            if self.exchange_type == ExchangeType.KRAKEN:
                success = await self._cancel_kraken_order(order_id)
            else:
                raise NotImplementedError("Bitvavo support not yet implemented")
            
            if success and order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = "cancelled"
            
            return success
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    async def _cancel_kraken_order(self, order_id: str) -> bool:
        """Cancel order on Kraken"""
        try:
            if hasattr(self.api_client, 'cancel_order'):
                if asyncio.iscoroutinefunction(self.api_client.cancel_order):
                    result = await self.api_client.cancel_order(order_id)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.api_client.cancel_order, order_id)
                
                return result if isinstance(result, bool) else not result.get('error')
            else:
                # Fallback to direct API
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.api_client.query_private, "CancelOrder", {'txid': order_id}
                )
                return not result.get('error')
                
        except Exception as e:
            logger.error(f"Kraken order cancellation failed: {e}")
            return False
    
    async def get_ohlc_data(self, pair: Optional[str] = None, interval: str = "15m", 
                           since: Optional[int] = None, limit: int = 100) -> List[List[float]]:
        """Get OHLC data with proper error handling"""
        try:
            if pair is None:
                pair = self.exchange_config["pair"]
            
            await self._rate_limit()
            
            if self.exchange_type == ExchangeType.KRAKEN:
                return await self._get_kraken_ohlc(pair, interval, since, limit)
            else:
                raise NotImplementedError("Bitvavo support not yet implemented")
                
        except Exception as e:
            logger.error(f"OHLC data fetch failed: {e}")
            return []
    
    async def _get_kraken_ohlc(self, pair: str, interval: str, since: Optional[int], 
                              limit: int) -> List[List[float]]:
        """Get OHLC from Kraken with proper interval mapping"""
        try:
            # Map interval to Kraken format
            interval_map = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "4h": 240, "1d": 1440
            }
            kraken_interval = interval_map.get(interval, 15)
            
            params = {
                "pair": pair,
                "interval": kraken_interval
            }
            
            if since:
                params["since"] = since
            
            if hasattr(self.api_client, 'get_ohlc_data'):
                if asyncio.iscoroutinefunction(self.api_client.get_ohlc_data):
                    return await self.api_client.get_ohlc_data(pair, kraken_interval, since or 0)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, self.api_client.get_ohlc_data, pair, kraken_interval, since or 0
                    )
            else:
                # Fallback to direct API
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.api_client.query_public, "OHLC", params
                )
                
                if result.get('error'):
                    logger.error(f"OHLC error: {result['error']}")
                    return []
                
                ohlc_data = result.get('result', {}).get(pair, [])
                
                # Convert to float and validate
                valid_candles = []
                for candle in ohlc_data[:limit]:
                    try:
                        if len(candle) >= 6:
                            float_candle = [float(x) for x in candle[:6]]
                            valid_candles.append(float_candle)
                    except (ValueError, TypeError):
                        continue
                
                logger.debug(f"Retrieved {len(valid_candles)} OHLC candles")
                return valid_candles
                
        except Exception as e:
            logger.error(f"Kraken OHLC failed: {e}")
            return []
    
    # Backward compatibility methods
    def get_btc_order_book(self) -> Optional[Dict]:
        """Sync wrapper for order book"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.get_order_book())
                    return future.result(timeout=10)
            else:
                return asyncio.run(self.get_order_book())
        except Exception as e:
            logger.error(f"Sync order book failed: {e}")
            return None
    
    def get_optimal_price(self, order_book: Dict, side: str) -> Optional[float]:
        """Get optimal price with default aggression"""
        return self.calculate_optimal_price(order_book, side, aggression=0.5)
    
    def get_market_volume(self) -> Optional[float]:
        """Sync wrapper for market volume"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_get_volume())
                    return future.result(timeout=10)
            else:
                return asyncio.run(self._async_get_volume())
        except Exception as e:
            logger.error(f"Sync volume failed: {e}")
            return None
    
    async def _async_get_volume(self) -> Optional[float]:
        """Async helper for volume"""
        _, volume = await self.fetch_current_price()
        return volume
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return {
            "exchange": self.exchange_type.value,
            "pair": self.exchange_config["pair"],
            "pending_orders": len(self.pending_orders),
            "order_history": len(self.order_history),
            "cache_entries": len(self.market_data_cache),
            "last_api_call": self.last_api_call,
            "api_rate_limit": self.exchange_config["api_rate_limit"]
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel any pending orders if needed
            for order_id in list(self.pending_orders.keys()):
                if self.pending_orders[order_id]["status"] == "pending":
                    await self.cancel_order(order_id)
            
            # Clear caches
            self.market_data_cache.clear()
            self.order_book_cache = None
            
            logger.info("Unified Trade Executor cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Factory function for creating the appropriate executor
def create_trade_executor(api_client, exchange_type: str = "kraken") -> UnifiedTradeExecutor:
    """Create trade executor for specified exchange"""
    try:
        exchange_enum = ExchangeType(exchange_type.lower())
        return UnifiedTradeExecutor(api_client, exchange_enum)
    except ValueError:
        raise ValueError(f"Unsupported exchange: {exchange_type}")


# Example usage
async def test_unified_executor():
    """Test the unified executor"""
    print("Testing Unified Trade Executor...")
    
    # Mock API client for testing
    class MockKrakenAPI:
        def get_btc_price(self):
            return 45000.0
        
        def get_market_volume(self):
            return 1500.0
        
        def get_total_btc_balance(self):
            return 0.1
        
        def get_available_balance(self, currency):
            return 5000.0 if currency == "EUR" else 0.0
    
    mock_api = MockKrakenAPI()
    executor = UnifiedTradeExecutor(mock_api, ExchangeType.KRAKEN)
    
    # Test price fetch
    price, volume = await executor.fetch_current_price()
    print(f"Price: €{price}, Volume: {volume}")
    
    # Test balance fetch
    balances = await executor.get_balances()
    print(f"Balances: {balances}")
    
    # Test statistics
    stats = executor.get_statistics()
    print(f"Statistics: {stats}")
    
    await executor.cleanup()
    print("✅ Unified Trade Executor test completed")


if __name__ == "__main__":
    asyncio.run(test_unified_executor())