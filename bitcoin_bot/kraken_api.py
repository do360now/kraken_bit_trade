"""
Production-Ready Async Kraken API Client
Optimized for high-frequency trading with proper error handling and security
"""

import asyncio
import aiohttp
import time
import base64
import hashlib
import hmac
import urllib.parse
from typing import Optional, List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom API error with detailed context"""
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 response_data: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.response_data = response_data

class RateLimitError(APIError):
    """Rate limit exceeded error"""
    pass

@dataclass
class RateLimiter:
    """Advanced rate limiting with burst capability"""
    calls_per_minute: int = 60
    burst_limit: int = 10
    _call_times: List[float] = None
    _burst_count: int = 0
    
    def __post_init__(self):
        if self._call_times is None:
            self._call_times = []
    
    async def acquire(self):
        """Acquire rate limit permission"""
        now = time.time()
        
        # Clean old call times (older than 1 minute)
        self._call_times = [t for t in self._call_times if now - t < 60]
        
        # Check burst limit
        if self._burst_count >= self.burst_limit:
            sleep_time = 60 / self.calls_per_minute
            logger.debug(f"Burst limit reached, sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            self._burst_count = 0
        
        # Check per-minute limit
        if len(self._call_times) >= self.calls_per_minute:
            oldest_call = min(self._call_times)
            sleep_time = 60 - (now - oldest_call) + 0.1  # Small buffer
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self._call_times.append(now)
        self._burst_count += 1

class AsyncKrakenAPI:
    """Production-grade async Kraken API client"""
    
    def __init__(self, api_key: str, api_secret: str, 
                 api_domain: str = "https://api.kraken.com",
                 rate_limiter: Optional[RateLimiter] = None):
        self.api_key = api_key
        self.api_secret = api_secret  
        self.api_domain = api_domain
        self.api_version = "0"
        self.rate_limiter = rate_limiter or RateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Connection settings for high performance
        self.connector_settings = {
            'limit': 100,  # Connection pool size
            'limit_per_host': 10,
            'ttl_dns_cache': 300,
            'use_dns_cache': True,
        }
        
    @asynccontextmanager
    async def get_session(self):
        """Get or create aiohttp session with proper cleanup"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(**self.connector_settings)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'BitcoinTradingBot/1.0'}
            )
        
        try:
            yield self._session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
    
    def _generate_signature(self, urlpath: str, data: Dict) -> str:
        """Generate Kraken API signature"""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    async def _request(self, endpoint: str, data: Optional[Dict] = None, 
                      is_private: bool = False, retries: int = 3) -> Dict:
        """Make API request with comprehensive error handling"""
        
        for attempt in range(retries + 1):
            try:
                await self.rate_limiter.acquire()
                
                async with self.get_session() as session:
                    if is_private:
                        data = data or {}
                        data['nonce'] = str(int(time.time() * 1000000))
                        urlpath = f'/{self.api_version}/private/{endpoint}'
                        
                        headers = {
                            'API-Key': self.api_key,
                            'API-Sign': self._generate_signature(urlpath, data)
                        }
                        
                        url = f'{self.api_domain}{urlpath}'
                        async with session.post(url, data=data, headers=headers) as response:
                            return await self._handle_response(response, attempt, retries)
                    else:
                        url = f'{self.api_domain}/{self.api_version}/public/{endpoint}'
                        async with session.get(url, params=data) as response:
                            return await self._handle_response(response, attempt, retries)
                            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == retries:
                    raise APIError(f"Network error after {retries + 1} attempts: {e}")
                
                backoff = (2 ** attempt) + (time.time() % 1)  # Jittered exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {backoff:.2f}s: {e}")
                await asyncio.sleep(backoff)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == retries:
                    raise APIError(f"Unexpected error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_response(self, response: aiohttp.ClientResponse, 
                              attempt: int, max_retries: int) -> Dict:
        """Handle API response with proper error checking"""
        try:
            if response.status == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                if attempt < max_retries:
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    raise RateLimitError("Rate limit exceeded")
                else:
                    raise RateLimitError("Rate limit exceeded, max retries reached")
            
            response.raise_for_status()
            result = await response.json()
            
            # Check for Kraken-specific errors
            if result.get('error'):
                error_msg = ', '.join(result['error'])
                error_code = result['error'][0] if result['error'] else None
                
                # Handle specific error types
                if 'API:Rate limit exceeded' in error_msg:
                    raise RateLimitError(error_msg)
                elif 'Service:Unavailable' in error_msg:
                    raise APIError(f"Kraken service unavailable: {error_msg}")
                else:
                    raise APIError(f"Kraken API error: {error_msg}", error_code, result)
            
            return result
            
        except aiohttp.ContentTypeError:
            text = await response.text()
            raise APIError(f"Invalid JSON response: {text[:200]}")
        except json.JSONDecodeError as e:
            text = await response.text()
            raise APIError(f"JSON decode error: {e}, response: {text[:200]}")

    # Public API Methods
    async def get_ticker(self, pair: str = "XXBTZEUR") -> Dict:
        """Get ticker information"""
        result = await self._request("Ticker", {"pair": pair})
        return result.get('result', {}).get(pair, {})
    
    async def get_btc_price(self) -> Optional[float]:
        """Get current BTC price"""
        try:
            ticker = await self.get_ticker("XXBTZEUR")
            return float(ticker['c'][0]) if ticker.get('c') else None
        except Exception as e:
            logger.error(f"Failed to get BTC price: {e}")
            return None
    
    async def get_market_volume(self, pair: str = "XXBTZEUR") -> Optional[float]:
        """Get 24h volume"""
        try:
            ticker = await self.get_ticker(pair)
            return float(ticker['v'][1]) if ticker.get('v') else None
        except Exception as e:
            logger.error(f"Failed to get market volume: {e}")
            return None
    
    async def get_order_book(self, pair: str = "XXBTZEUR", count: int = 10) -> Optional[Dict]:
        """Get order book"""
        try:
            result = await self._request("Depth", {"pair": pair, "count": count})
            return result.get('result', {}).get(pair)
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return None
    
    async def get_ohlc_data(self, pair: str = "XXBTZEUR", interval: int = 15, 
                           since: int = 0, limit: Optional[int] = None) -> List[List[float]]:
        """Get OHLC data with improved error handling"""
        try:
            params = {"pair": pair, "interval": interval}
            if since > 0:
                params["since"] = since
                
            result = await self._request("OHLC", params)
            ohlc_data = result.get('result', {}).get(pair, [])
            
            # Validate and convert data
            valid_candles = []
            for candle in ohlc_data:
                if isinstance(candle, list) and len(candle) >= 7:
                    try:
                        converted = [float(x) for x in candle[:7]]
                        valid_candles.append(converted)
                    except (ValueError, TypeError):
                        continue
            
            # Apply limit if specified
            if limit and len(valid_candles) > limit:
                valid_candles = valid_candles[-limit:]
                
            logger.debug(f"Retrieved {len(valid_candles)} OHLC candles for {pair}")
            return valid_candles
            
        except Exception as e:
            logger.error(f"Failed to get OHLC data: {e}")
            return []

    # Private API Methods
    async def get_account_balance(self) -> Optional[Dict]:
        """Get account balance"""
        try:
            result = await self._request("Balance", is_private=True)
            return result.get('result', {})
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return None
    
    async def get_btc_balance(self) -> Optional[float]:
        """Get BTC balance"""
        try:
            balance = await self.get_account_balance()
            if balance:
                return float(balance.get('XXBT', 0))
            return None
        except Exception as e:
            logger.error(f"Failed to get BTC balance: {e}")
            return None
    
    async def get_eur_balance(self) -> Optional[float]:
        """Get EUR balance"""
        try:
            balance = await self.get_account_balance()
            if balance:
                return float(balance.get('ZEUR', 0))
            return None
        except Exception as e:
            logger.error(f"Failed to get EUR balance: {e}")
            return None
    
    async def place_order(self, pair: str, order_type: str, side: str, 
                         volume: float, price: Optional[float] = None,
                         **kwargs) -> Optional[str]:
        """Place order with enhanced error handling"""
        try:
            data = {
                'pair': pair,
                'type': side,  # buy/sell
                'ordertype': order_type,  # limit/market
                'volume': str(volume)
            }
            
            if price is not None:
                data['price'] = str(price)
                
            # Add additional parameters
            data.update(kwargs)
            
            result = await self._request("AddOrder", data, is_private=True)
            
            # Extract order ID
            order_ids = result.get('result', {}).get('txid', [])
            return order_ids[0] if order_ids else None
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            result = await self._request("CancelOrder", {'txid': order_id}, is_private=True)
            return 'result' in result and not result.get('error')
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_open_orders(self) -> Dict:
        """Get open orders"""
        try:
            result = await self._request("OpenOrders", is_private=True)
            return result.get('result', {}).get('open', {})
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return {}
    
    async def get_order_info(self, order_id: str) -> Optional[Dict]:
        """Get order information"""
        try:
            result = await self._request("QueryOrders", {'txid': order_id}, is_private=True)
            orders = result.get('result', {})
            return orders.get(order_id) if orders else None
        except Exception as e:
            logger.error(f"Failed to get order info for {order_id}: {e}")
            return None

    def get_optimal_price(self, order_book: Dict, side: str, 
                         buffer_pct: float = 0.01) -> Optional[float]:
        """Calculate optimal price from order book"""
        try:
            if side.lower() == "buy":
                asks = order_book.get('asks', [])
                if not asks:
                    return None
                best_ask = float(asks[0][0])
                return round(best_ask * (1 - buffer_pct), 2)
                
            elif side.lower() == "sell":
                bids = order_book.get('bids', [])
                if not bids:
                    return None
                best_bid = float(bids[0][0])
                return round(best_bid * (1 + buffer_pct), 2)
                
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Failed to calculate optimal price: {e}")
            return None
    
    async def close(self):
        """Close the API client and cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Kraken API client closed")

    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Factory function for easy integration
async def create_kraken_client(api_key: str, api_secret: str) -> AsyncKrakenAPI:
    """Create and test Kraken API client"""
    client = AsyncKrakenAPI(api_key, api_secret)
    
    # Test connection
    try:
        price = await client.get_btc_price()
        if price is None:
            raise APIError("Failed to fetch BTC price - connection test failed")
        
        logger.info(f"Kraken API client ready - Current BTC price: €{price:.2f}")
        return client
        
    except Exception as e:
        await client.close()
        raise APIError(f"Kraken API connection test failed: {e}")


# Synchronous wrapper for backward compatibility
class KrakenAPISync:
    """Synchronous wrapper for async Kraken API"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._loop = None
        self._client = None
    
    def _run_async(self, coro):
        """Run async function in sync context"""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        if self._client is None:
            self._client = AsyncKrakenAPI(self.api_key, self.api_secret)
        
        return self._loop.run_until_complete(coro)
    
    def get_btc_price(self) -> Optional[float]:
        return self._run_async(self._client.get_btc_price())
    
    def get_btc_balance(self) -> Optional[float]:
        return self._run_async(self._client.get_btc_balance())
    
    def get_eur_balance(self) -> Optional[float]:
        return self._run_async(self._client.get_eur_balance())
    
    def place_order(self, pair: str, order_type: str, side: str, 
                   volume: float, price: Optional[float] = None) -> Optional[str]:
        return self._run_async(self._client.place_order(pair, order_type, side, volume, price))
    
    def __del__(self):
        if self._client:
            try:
                self._loop.run_until_complete(self._client.close())
            except:
                pass