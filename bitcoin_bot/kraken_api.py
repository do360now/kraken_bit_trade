import requests
import time
import base64
import hashlib
import hmac
import urllib.parse
from typing import Optional, List, Dict, Any
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

class KrakenAPI:
    def __init__(self, api_key: str, api_secret: str, api_domain: str = "https://api.kraken.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_domain = api_domain
        self.api_version = "0"
        self.last_request_time = 0
        self.request_interval = 1

    def _get_kraken_signature(self, urlpath: str, data: Dict, secret: str) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def query_private(self, endpoint: str, data: Dict[str, Any]) -> Dict:
        try:
            data['nonce'] = str(int(time.time() * 1000))
            urlpath = f'/{self.api_version}/private/{endpoint}'
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._get_kraken_signature(urlpath, data, self.api_secret)
            }
            url = f'{self.api_domain}{urlpath}'
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.request_interval:
                time.sleep(self.request_interval - time_since_last)
            self.last_request_time = time.time()
            response = requests.post(url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get('error'):
                logger.error(f"Kraken private API error: {result['error']}")
            return result
        except Exception as e:
            logger.error(f"Kraken private API request failed: {e}")
            return {'error': str(e)}

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def query_public(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        try:
            url = f'{self.api_domain}/{self.api_version}/public/{endpoint}'
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.request_interval:
                time.sleep(self.request_interval - time_since_last)
            self.last_request_time = time.time()
            response = requests.get(url, params=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get('error'):
                logger.error(f"Kraken public API error: {result['error']}")
            return result
        except Exception as e:
            logger.error(f"Kraken public API request failed: {e}")
            return {'error': str(e)}

    def get_ohlc_data(self, pair: str = "XXBTZEUR", interval: int = 15, since: int = 0) -> Optional[List[List]]:
        try:
            result = self.query_public("OHLC", {"pair": pair, "interval": interval, "since": since})
            if result.get('error'):
                logger.error(f"Failed to fetch OHLC: {result['error']}")
                return []
            ohlc_data = result.get('result', {}).get(pair, [])
            if not ohlc_data:
                logger.warning(f"No OHLC data returned for {pair}")
                return []
            valid_candles = []
            for candle in ohlc_data:
                if isinstance(candle, list) and len(candle) >= 7:
                    try:
                        valid_candles.append([float(c) for c in candle[:7]])
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Invalid candle data: {candle}, error: {e}")
                        continue
            logger.debug(f"Fetched {len(valid_candles)} valid OHLC candles for {pair}")
            return valid_candles
        except Exception as e:
            logger.error(f"Failed to fetch OHLC data for {pair}: {e}")
            return []

    def get_btc_order_book(self) -> Optional[Dict]:
        result = self.query_public("Depth", {"pair": "XXBTZEUR"})
        if result.get('error'):
            logger.error(f"Failed to fetch order book: {result['error']}")
            return None
        return result.get('result', {}).get("XXBTZEUR")

    def get_optimal_price(self, order_book: Dict, side: str, buffer: float = 0.05, decimals: int = 1) -> Optional[float]:
        if not order_book:
            return None
        try:
            best_bid = float(order_book['bids'][0][0]) if order_book.get('bids') else None
            best_ask = float(order_book['asks'][0][0]) if order_book.get('asks') else None
            if side == "buy":
                if best_ask is None:
                    return None
                if best_bid is not None and best_ask * 0.999 > best_bid:
                    optimal_price = best_ask - buffer
                else:
                    optimal_price = best_ask
            elif side == "sell":
                if best_bid is None:
                    return None
                optimal_price = best_bid + buffer
            else:
                return None
            return round(optimal_price, decimals)
        except Exception as e:
            logger.error(f"Failed to calculate optimal price: {e}")
            return None

    def get_btc_price(self) -> Optional[float]:
        result = self.query_public("Ticker", {"pair": "XXBTZEUR"})
        if result.get('error'):
            logger.error(f"Failed to fetch ticker: {result['error']}")
            return None
        return float(result['result']['XXBTZEUR']['c'][0])

    def get_market_volume(self, pair: str = "XXBTZEUR") -> Optional[float]:
        result = self.query_public("Ticker", {"pair": pair})
        if result.get('error'):
            logger.error(f"Failed to fetch ticker: {result['error']}")
            return None
        try:
            return float(result['result'][pair]['v'][1])
        except (KeyError, ValueError) as e:
            logger.error(f"Error retrieving market volume: {e}")
            return None

    def get_total_btc_balance(self) -> Optional[float]:
        result = self.query_private("Balance", {})
        if result.get('error'):
            logger.error(f"Failed to fetch balance: {result['error']}")
            return None
        balance = result.get('result', {}).get('XBT.F', 0)  # Use XXBT for BTC balance on Kraken
        return float(balance)

    def get_available_balance(self, asset: str) -> Optional[float]:
        result = self.query_private("Balance", {})
        if result.get('error'):
            logger.error(f"Failed to fetch balance: {result['error']}")
            return None
        
        # Map asset names to Kraken format
        asset_map = {
            'EUR': 'ZEUR',
            'BTC': 'XXBT',
            'USD': 'ZUSD'
        }
        
        kraken_asset = asset_map.get(asset, asset)
        balance = result.get('result', {}).get(kraken_asset, 0)
        
        if balance is None:
            logger.warning(f"Asset {asset} not found in balance response")
            return 0.0
            
        return float(balance)

    def place_order(self, pair: str, type_: str, ordertype: str, volume: float, price: float = None, **kwargs) -> Dict:
        """Place order on Kraken"""
        data = {
            'pair': pair,
            'type': type_,
            'ordertype': ordertype,
            'volume': str(volume)
        }
        
        if price is not None:
            data['price'] = str(price)
            
        # Add any additional parameters
        data.update(kwargs)
        
        return self.query_private('AddOrder', data)

    def cancel_order(self, txid: str) -> Dict:
        """Cancel order on Kraken"""
        return self.query_private('CancelOrder', {'txid': txid})

    def get_open_orders(self) -> Dict:
        """Get open orders"""
        return self.query_private('OpenOrders', {})

    def get_order_info(self, txid: str) -> Dict:
        """Get order information"""
        return self.query_private('QueryOrders', {'txid': txid})


def authenticate_kraken():
    """Authenticate with Kraken exchange"""
    try:
        import os
        from dotenv import load_dotenv
        # Load environment variables from the .env file
        load_dotenv()
        api_key = os.getenv("KRAKEN_API_KEY")
        api_secret = os.getenv("KRAKEN_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in environment variables")
        
        kraken = KrakenAPI(api_key, api_secret)
        
        # Test the connection
        balance = kraken.get_total_btc_balance()
        if balance is not None:
            logger.info("Kraken authentication successful")
            return kraken
        else:
            raise Exception("Failed to fetch balance - authentication may have failed")
            
    except Exception as e:
        logger.error(f"Failed to authenticate with Kraken: {e}")
        raise

def test_connection(kraken_api):
    """Test Kraken connection"""
    try:
        # Test market data
        price = kraken_api.get_btc_price()
        if price:
            logger.info(f"Market data test: BTC/EUR = €{price:.2f}")
            
        # Test authenticated endpoints
        btc_balance = kraken_api.get_total_btc_balance()
        eur_balance = kraken_api.get_available_balance("EUR")
        
        if btc_balance is not None and eur_balance is not None:
            logger.info("Authenticated API access confirmed")
            return True
        else:
            logger.warning("Failed to fetch balances")
            return False
            
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False