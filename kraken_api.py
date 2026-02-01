"""
Improved Kraken API client with fixed balance calculation and better error handling.
"""
import requests
import time
import base64
import hashlib
import hmac
import urllib.parse
from typing import Optional, List, Dict, Any
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log
)
import logging

from core.constants import (
    KRAKEN_API_VERSION,
    KRAKEN_BTC_KEYS,
    KRAKEN_EUR_KEYS,
    DEFAULT_REQUEST_INTERVAL,
    MAX_RETRY_ATTEMPTS,
    OHLCField,
    MIN_OHLC_LENGTH
)
from core.exceptions import (
    APIError,
    NetworkError,
    AuthenticationError,
    RateLimitError
)
from circuit_breaker import circuit_breaker
from logger_config import logger


class RetryableAPIError(APIError):
    """API errors that should trigger retry"""
    pass


def is_retryable_error(exception) -> bool:
    """Determine if an error should trigger retry"""
    if isinstance(exception, (RetryableAPIError, NetworkError, RateLimitError)):
        return True
    
    if isinstance(exception, requests.exceptions.Timeout):
        return True
    
    if isinstance(exception, requests.exceptions.ConnectionError):
        return True
    
    if isinstance(exception, requests.exceptions.HTTPError):
        if exception.response is not None:
            # Retry on 5xx errors, not 4xx
            return 500 <= exception.response.status_code < 600
    
    return False


class KrakenAPI:
    """
    Improved Kraken API client with:
    - Fixed balance calculation considering locked funds
    - Better retry logic with exponential backoff
    - Proper error classification
    - Thread-safe request rate limiting
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_domain: str = "https://api.kraken.com"
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_domain = api_domain
        self.api_version = KRAKEN_API_VERSION
        self.request_interval = DEFAULT_REQUEST_INTERVAL
        
        # Thread-safe rate limiting
        self._last_request_time = 0
        self._request_lock = __import__('threading').Lock()
    
    def _get_kraken_signature(self, urlpath: str, data: Dict, secret: str) -> str:
        """Generate Kraken API signature"""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        with self._request_lock:
            time_since_last = time.time() - self._last_request_time
            if time_since_last < self.request_interval:
                time.sleep(self.request_interval - time_since_last)
            self._last_request_time = time.time()
    
    def _handle_kraken_error(self, error_list: List[str], endpoint: str) -> None:
        """
        Parse Kraken error and raise appropriate exception.
        
        Args:
            error_list: List of error strings from Kraken
            endpoint: API endpoint that was called
        
        Raises:
            Appropriate exception based on error type
        """
        error_str = ', '.join(error_list)
        
        # Retryable errors
        retryable_patterns = [
            'Rate limit exceeded',
            'Service unavailable',
            'Service:Unavailable',
            'Service:Busy',
            'EAPI:Rate limit exceeded'
        ]
        
        if any(pattern in error_str for pattern in retryable_patterns):
            logger.warning(
                f"Kraken API rate limit/service error on {endpoint}",
                error=error_str
            )
            raise RetryableAPIError(f"Kraken API (retryable): {error_str}")
        
        # Authentication errors (not retryable)
        auth_patterns = [
            'Invalid key',
            'Invalid signature',
            'Permission denied',
            'EAPI:Invalid key',
            'EAPI:Invalid signature'
        ]
        
        if any(pattern in error_str for pattern in auth_patterns):
            logger.error(
                f"Kraken API authentication error on {endpoint}",
                error=error_str
            )
            raise AuthenticationError(f"Kraken authentication failed: {error_str}")
        
        # Other non-retryable errors
        logger.error(
            f"Kraken API error on {endpoint}",
            error=error_str
        )
        raise APIError(f"Kraken API error: {error_str}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        retry=retry_if_exception_type(RetryableAPIError),
        before_sleep=before_sleep_log(logger.logger, logging.WARNING),
        reraise=True
    )
    @circuit_breaker(failure_threshold=5, recovery_timeout=300, expected_exception=(APIError,))
    def query_private(self, endpoint: str, data: Dict[str, Any]) -> Dict:
        """
        Execute private API request with retry and circuit breaker.
        
        Args:
            endpoint: API endpoint
            data: Request data
        
        Returns:
            Response dict from Kraken
        
        Raises:
            APIError: For API errors
            NetworkError: For network errors
        """
        try:
            self._rate_limit()
            
            data['nonce'] = str(int(time.time() * 1000))
            urlpath = f'/{self.api_version}/private/{endpoint}'
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._get_kraken_signature(urlpath, data, self.api_secret)
            }
            
            url = f'{self.api_domain}{urlpath}'
            
            response = requests.post(url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            # Check for Kraken API errors
            if result.get('error'):
                self._handle_kraken_error(result['error'], endpoint)
            
            logger.debug(f"Kraken private API success: {endpoint}")
            return result
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Kraken API timeout on {endpoint}", error=str(e))
            raise NetworkError(f"Request timeout: {e}")
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Kraken API connection error on {endpoint}", error=str(e))
            raise NetworkError(f"Connection error: {e}")
        
        except requests.exceptions.HTTPError as e:
            if e.response is not None and 500 <= e.response.status_code < 600:
                logger.warning(f"Kraken API server error on {endpoint}", status_code=e.response.status_code)
                raise RetryableAPIError(f"HTTP {e.response.status_code}: {e}")
            else:
                logger.error(f"Kraken API HTTP error on {endpoint}", error=str(e))
                raise APIError(f"HTTP error: {e}")
        
        except (RetryableAPIError, AuthenticationError, APIError):
            # Re-raise our custom exceptions
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error in Kraken API {endpoint}", error=str(e), exc_info=True)
            raise APIError(f"Unexpected error: {e}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        retry=retry_if_exception_type((RetryableAPIError, NetworkError)),
        before_sleep=before_sleep_log(logger.logger, logging.WARNING),
        reraise=True
    )
    @circuit_breaker(failure_threshold=5, recovery_timeout=300, expected_exception=(APIError,))
    def query_public(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        Execute public API request with retry and circuit breaker.
        
        Args:
            endpoint: API endpoint
            data: Request parameters
        
        Returns:
            Response dict from Kraken
        """
        try:
            self._rate_limit()
            
            url = f'{self.api_domain}/{self.api_version}/public/{endpoint}'
            
            response = requests.get(url, params=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('error'):
                self._handle_kraken_error(result['error'], endpoint)
            
            return result
            
        except requests.exceptions.Timeout as e:
            raise NetworkError(f"Request timeout: {e}")
        
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Connection error: {e}")
        
        except requests.exceptions.HTTPError as e:
            if e.response is not None and 500 <= e.response.status_code < 600:
                raise RetryableAPIError(f"HTTP {e.response.status_code}: {e}")
            raise APIError(f"HTTP error: {e}")
        
        except (RetryableAPIError, APIError):
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error in public API {endpoint}", error=str(e), exc_info=True)
            raise APIError(f"Unexpected error: {e}")
    
    def get_ohlc_data(
        self,
        pair: str = "XXBTZEUR",
        interval: int = 15,
        since: int = 0
    ) -> Optional[List[List]]:
        """
        Fetch OHLC data with validation.
        
        Args:
            pair: Trading pair
            interval: Candle interval in minutes
            since: Unix timestamp to fetch from
        
        Returns:
            List of OHLC candles or empty list
        """
        try:
            result = self.query_public("OHLC", {"pair": pair, "interval": interval, "since": since})
            
            ohlc_data = result.get('result', {}).get(pair, [])
            if not ohlc_data:
                logger.warning(f"No OHLC data returned for {pair}")
                return []
            
            # Validate and convert candles
            valid_candles = []
            for candle in ohlc_data:
                if not isinstance(candle, list) or len(candle) < MIN_OHLC_LENGTH:
                    logger.debug(f"Skipping invalid candle: {candle}")
                    continue
                
                try:
                    # Convert to float and validate
                    converted = [float(c) for c in candle[:MIN_OHLC_LENGTH]]
                    
                    # Basic sanity checks
                    if converted[OHLCField.CLOSE] <= 0 or converted[OHLCField.VOLUME] < 0:
                        logger.debug(f"Skipping candle with invalid values: {converted}")
                        continue
                    
                    valid_candles.append(converted)
                    
                except (ValueError, TypeError) as e:
                    logger.debug(f"Invalid candle data: {candle}, error: {e}")
                    continue
            
            logger.debug(f"Fetched {len(valid_candles)} valid OHLC candles for {pair}")
            return valid_candles
            
        except APIError as e:
            logger.error(f"Failed to fetch OHLC data for {pair}", error=str(e))
            return []
    
    def get_latest_ohlc(self, pair: str = "XXBTZEUR", interval: int = 15) -> Optional[List]:
        """
        Fetch the latest OHLC candle for a pair.
        
        This is a convenience method used by MarketDataService for getting
        the most recent price data.
        
        Args:
            pair: Trading pair (default BTC/EUR)
            interval: Candle interval in minutes (default 15)
        
        Returns:
            Latest OHLC candle (list) or None if error
        """
        try:
            # Fetch last 2 hours of data to ensure we get the latest candle
            since = int(time.time() - 7200)
            ohlc_data = self.get_ohlc_data(pair=pair, interval=interval, since=since)
            
            if ohlc_data and len(ohlc_data) > 0:
                return ohlc_data[-1]  # Return the most recent candle
            
            logger.warning(f"No OHLC data available for {pair}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch latest OHLC for {pair}: {e}")
            return None
    
    def get_available_balance(self, asset: str) -> Optional[float]:
        """
        Get available balance for an asset, accounting for locked funds.
        
        This is the FIXED version that properly handles:
        - Both EUR and BTC assets
        - Partially filled orders
        - Remaining volume instead of total volume
        
        Args:
            asset: Asset symbol (e.g., 'EUR', 'XXBT')
        
        Returns:
            Available balance or None if error
        """
        try:
            # Get base balance
            balance_result = self.query_private("Balance", {})
            
            # Check for API errors first
            if balance_result.get('error'):
                logger.error(
                    f"API error fetching balance for {asset}",
                    error=balance_result['error']
                )
                return None
            
            # Try different asset key variations
            if asset == 'EUR':
                possible_keys = KRAKEN_EUR_KEYS
            elif asset in ['BTC', 'XXBT', 'XBT']:
                possible_keys = KRAKEN_BTC_KEYS
            else:
                possible_keys = [f"Z{asset}", asset, f"{asset}.F", f"X{asset}"]
            
            balance = None
            balance_key_used = None
            for key in possible_keys:
                if key in balance_result.get('result', {}):
                    balance = float(balance_result['result'][key])
                    balance_key_used = key
                    break
            
            if balance is None:
                available_keys = list(balance_result.get('result', {}).keys())
                logger.warning(
                    f"Asset {asset} not found in balance",
                    available_keys=available_keys
                )
                return None
            
            logger.debug(f"Base balance for {asset} ({balance_key_used}): {balance:.8f}")
            
            # Get open orders
            open_orders_result = self.query_private("OpenOrders", {})
            
            if open_orders_result.get('error'):
                logger.warning("Failed to fetch open orders, returning base balance")
                return balance
            
            # Calculate locked amounts
            locked_balance = 0.0
            open_orders = open_orders_result.get('result', {}).get('open', {})
            
            for order_id, order in open_orders.items():
                if order['descr']['pair'] != 'XXBTZEUR':
                    continue
                
                # Use REMAINING volume, not total volume
                total_vol = float(order['vol'])
                executed_vol = float(order.get('vol_exec', 0))
                remaining_vol = total_vol - executed_vol
                
                order_type = order['descr']['type']
                
                if asset == 'EUR' and order_type == 'buy':
                    # EUR locked in buy orders
                    price = float(order['descr']['price'])
                    locked_amount = remaining_vol * price
                    locked_balance += locked_amount
                    
                    logger.debug(
                        f"EUR locked in buy order {order_id}",
                        remaining_vol=remaining_vol,
                        price=price,
                        locked=locked_amount
                    )
                
                elif asset in ['XXBT', 'XBT', 'BTC'] and order_type == 'sell':
                    # BTC locked in sell orders
                    locked_balance += remaining_vol
                    
                    logger.debug(
                        f"BTC locked in sell order {order_id}",
                        remaining_vol=remaining_vol
                    )
            
            available = max(balance - locked_balance, 0.0)
            
            logger.info(
                f"Available {asset} balance calculated",
                total=balance,
                locked=locked_balance,
                available=available
            )
            
            return available
            
        except APIError as e:
            logger.error(f"Failed to get available balance for {asset}", error=str(e))
            return None
    
    def get_total_btc_balance(self) -> Optional[float]:
        """
        Get total BTC balance (including locked funds).
        
        Returns:
            Total BTC balance, 0.0 if no BTC, or None if API error
        """
        try:
            result = self.query_private("Balance", {})
            
            # Check for API errors first
            if result.get('error'):
                logger.warning(
                    "API returned error when fetching balance",
                    error=result['error']
                )
                return None
            
            # Try different BTC asset keys
            for key in KRAKEN_BTC_KEYS:
                if key in result.get('result', {}):
                    balance = float(result['result'][key])
                    logger.debug(f"Total BTC balance ({key}): {balance:.8f}")
                    return balance
            
            # BTC not found - means you have 0 BTC (API succeeded but no balance)
            logger.warning(
                "BTC asset not found in balance",
                available_keys=list(result.get('result', {}).keys())
            )
            return 0.0  # No BTC in account (valid state)
            
        except APIError as e:
            logger.error("Failed to fetch BTC balance", error=str(e))
            return None
    
    def get_btc_price(self) -> Optional[float]:
        """Get current BTC price in EUR"""
        try:
            result = self.query_public("Ticker", {"pair": "XXBTZEUR"})
            price = float(result['result']['XXBTZEUR']['c'][0])
            return price
        except APIError as e:
            logger.error("Failed to fetch BTC price", error=str(e))
            return None
    
    def get_market_volume(self, pair: str = "XXBTZEUR") -> Optional[float]:
        """Get 24h market volume"""
        try:
            result = self.query_public("Ticker", {"pair": pair})
            volume = float(result['result'][pair]['v'][1])
            return volume
        except APIError as e:
            logger.error(f"Failed to fetch market volume for {pair}", error=str(e))
            return None
    
    def get_order_book(self, pair: str = "XXBTZEUR", count: int = 10) -> Optional[Dict]:
        """Get order book depth"""
        try:
            result = self.query_public("Depth", {"pair": pair, "count": count})
            return result.get('result', {}).get(pair)
        except APIError as e:
            logger.error(f"Failed to fetch order book for {pair}", error=str(e))
            return None
        
    def get_optimal_price(self, order_book: Dict, side: str) -> float:
        """
        Get optimal price from order book.
        
        Args:
            order_book: Order book with 'asks' and 'bids'
            side: 'buy' or 'sell'
        
        Returns:
            Optimal price for the order
        """
        try:
            if side == "buy":
                # For buying, use best ask price (what sellers are asking)
                asks = order_book.get('asks', [])
                if asks and len(asks) > 0:
                    # asks format: [[price, volume, timestamp], ...]
                    best_ask = float(asks[0][0])
                    logger.debug(f"Best ask price: €{best_ask:.2f}")
                    return best_ask
                else:
                    # Fallback to current market price
                    logger.warning("No asks in order book, using market price")
                    return self.get_btc_price()
            
            else:  # sell
                # For selling, use best bid price (what buyers are offering)
                bids = order_book.get('bids', [])
                if bids and len(bids) > 0:
                    # bids format: [[price, volume, timestamp], ...]
                    best_bid = float(bids[0][0])
                    logger.debug(f"Best bid price: €{best_bid:.2f}")
                    return best_bid
                else:
                    # Fallback to current market price
                    logger.warning("No bids in order book, using market price")
                    return self.get_btc_price()
        
        except Exception as e:
            logger.error(f"Error getting optimal price: {e}")
            # Ultimate fallback
            return self.get_btc_price()

