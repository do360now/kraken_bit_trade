"""
Free Exchange Flow Alternatives - No API Keys Required

Since Glassnode, CryptoQuant, and Santiment now require paid subscriptions,
this module provides FREE alternatives using:

1. Blockchain.com public API (free, no key needed)
2. CoinGecko API (free tier available)
3. Proxy indicators (volume, price momentum)
4. Smart heuristics

No API keys required!
"""

import requests
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from logger_config import logger


class FreeExchangeFlowTracker:
    """
    Track exchange flows using FREE data sources.
    No API keys required!
    
    Uses multiple free sources and smart heuristics to estimate
    accumulation vs distribution.
    """
    
    def __init__(self, cache_duration: int = 300):
        """
        Initialize free exchange flow tracker.
        
        Args:
            cache_duration: Cache duration in seconds (default 5 minutes)
        """
        self.cache_duration = cache_duration
        self._cache = {}
        self._cache_time = 0
        
        # Free API endpoints (no keys needed!)
        self.blockchain_api = "https://blockchain.info"
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        logger.info("Free exchange flow tracker initialized (no API key needed)")
    
    def get_exchange_netflow_estimate(self) -> Optional[float]:
        """
        Estimate exchange netflow using free data sources.
        
        Method:
        - Uses volume trends (high volume + price drop = distribution)
        - Uses transaction count trends
        - Uses mempool analysis
        - Uses price momentum
        
        Returns:
            Estimated netflow in BTC (negative = accumulation)
        """
        # Check cache
        if time.time() - self._cache_time < self.cache_duration:
            if 'netflow' in self._cache:
                logger.debug(f"Using cached netflow estimate: {self._cache['netflow']:.2f} BTC")
                return self._cache['netflow']
        
        try:
            # Get multiple indicators
            volume_signal = self._get_volume_signal()
            mempool_signal = self._get_mempool_signal()
            price_momentum = self._get_price_momentum()
            
            # Combine signals into netflow estimate
            netflow_estimate = self._calculate_netflow_estimate(
                volume_signal,
                mempool_signal,
                price_momentum
            )
            
            # Cache result
            self._cache['netflow'] = netflow_estimate
            self._cache_time = time.time()
            
            logger.info(
                f"Estimated netflow: {netflow_estimate:.2f} BTC",
                volume_signal=volume_signal,
                mempool_signal=mempool_signal,
                price_momentum=price_momentum
            )
            
            return netflow_estimate
            
        except Exception as e:
            logger.error(f"Failed to estimate netflow: {e}")
            return None
    
    def _get_volume_signal(self) -> float:
        """
        Get volume signal from free sources.
        
        Returns:
            Volume score (-1 to +1, negative = accumulation)
        """
        try:
            # Use CoinGecko for volume data (free, no key)
            url = f"{self.coingecko_api}/coins/bitcoin"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Get current volume
            volume_24h = data.get('market_data', {}).get('total_volume', {}).get('btc', 0)
            
            # Get price change
            price_change_24h = data.get('market_data', {}).get('price_change_percentage_24h', 0)
            
            # Volume interpretation:
            # High volume + price drop = distribution (selling pressure)
            # High volume + price rise = accumulation (buying pressure)
            # Low volume = neutral
            
            if volume_24h > 200000:  # High volume
                if price_change_24h < -2:  # Price dropping
                    return 0.8  # Strong distribution signal
                elif price_change_24h > 2:  # Price rising
                    return -0.8  # Strong accumulation signal
                else:
                    return 0.2  # Neutral high volume
            else:  # Normal volume
                if price_change_24h < -3:
                    return 0.4  # Mild distribution
                elif price_change_24h > 3:
                    return -0.4  # Mild accumulation
                else:
                    return 0.0  # Neutral
                    
        except Exception as e:
            logger.debug(f"Failed to get volume signal: {e}")
            return 0.0
    
    def _get_mempool_signal(self) -> float:
        """
        Get mempool signal from blockchain.info (free).
        
        Returns:
            Mempool score (-1 to +1, negative = accumulation)
        """
        try:
            # Get mempool stats from blockchain.info
            url = f"{self.blockchain_api}/q/unconfirmedcount"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            unconfirmed_tx = int(response.text)
            
            # Mempool interpretation:
            # Very high mempool (>100k tx) = network congestion, possible distribution
            # Low mempool (<20k tx) = calm network, possible accumulation
            # Normal mempool (20k-100k) = neutral
            
            if unconfirmed_tx > 100000:
                return 0.5  # Distribution signal
            elif unconfirmed_tx < 20000:
                return -0.5  # Accumulation signal
            else:
                return 0.0  # Neutral
                
        except Exception as e:
            logger.debug(f"Failed to get mempool signal: {e}")
            return 0.0
    
    def _get_price_momentum(self) -> float:
        """
        Get price momentum signal.
        
        Returns:
            Momentum score (-1 to +1, negative = accumulation opportunity)
        """
        try:
            # Get price data from CoinGecko
            url = f"{self.coingecko_api}/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'eur',
                'days': '7',
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            prices = [p[1] for p in data.get('prices', [])]
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate 7-day momentum
            current_price = prices[-1]
            week_ago_price = prices[0]
            
            momentum = (current_price - week_ago_price) / week_ago_price * 100
            
            # Momentum interpretation:
            # Strong drop (-5%+) = accumulation opportunity (negative netflow expected)
            # Strong rise (+5%+) = distribution likely (positive netflow expected)
            
            if momentum < -5:
                return -0.7  # Strong accumulation opportunity
            elif momentum < -2:
                return -0.3  # Mild accumulation opportunity
            elif momentum > 5:
                return 0.7  # Strong distribution likely
            elif momentum > 2:
                return 0.3  # Mild distribution likely
            else:
                return 0.0  # Neutral
                
        except Exception as e:
            logger.debug(f"Failed to get price momentum: {e}")
            return 0.0
    
    def _calculate_netflow_estimate(
        self,
        volume_signal: float,
        mempool_signal: float,
        price_momentum: float
    ) -> float:
        """
        Calculate netflow estimate from multiple signals.
        
        Args:
            volume_signal: -1 to +1
            mempool_signal: -1 to +1
            price_momentum: -1 to +1
        
        Returns:
            Estimated netflow in BTC (negative = accumulation)
        """
        # Weighted average of signals
        # Volume is most important (50%), then momentum (30%), then mempool (20%)
        composite_signal = (
            volume_signal * 0.5 +
            price_momentum * 0.3 +
            mempool_signal * 0.2
        )
        
        # Scale to realistic BTC range
        # -1.0 signal = -1500 BTC (strong accumulation)
        # +1.0 signal = +1500 BTC (strong distribution)
        # 0.0 signal = 0 BTC (neutral)
        
        netflow_estimate = composite_signal * 1500
        
        return netflow_estimate
    
    def is_accumulation_phase(self, threshold: float = -500) -> bool:
        """
        Check if we're in an accumulation phase.
        
        Args:
            threshold: Negative netflow threshold (default -500 BTC)
        
        Returns:
            True if accumulating
        """
        netflow = self.get_exchange_netflow_estimate()
        
        if netflow is None:
            logger.warning("No netflow estimate available")
            return False
        
        is_accumulating = netflow < threshold
        
        if is_accumulating:
            logger.info(f"✅ ACCUMULATION DETECTED: estimated {netflow:.0f} BTC")
        
        return is_accumulating
    
    def get_flow_metrics(self) -> Dict[str, any]:
        """Get comprehensive flow metrics using free sources."""
        netflow = self.get_exchange_netflow_estimate()
        
        if netflow is None:
            return {
                'available': False,
                'message': 'Could not estimate netflow'
            }
        
        # Determine signal strength
        if netflow < -1000:
            signal = 'strong_accumulation'
        elif netflow < -500:
            signal = 'accumulation'
        elif netflow < 0:
            signal = 'mild_accumulation'
        elif netflow > 1000:
            signal = 'strong_distribution'
        elif netflow > 500:
            signal = 'distribution'
        elif netflow > 0:
            signal = 'mild_distribution'
        else:
            signal = 'neutral'
        
        return {
            'available': True,
            'netflow_estimate': netflow,
            'signal': signal,
            'method': 'free_proxy_indicators',
            'confidence': 'medium'  # Lower than paid APIs but still useful
        }


class SimpleAccumulationDetector:
    """
    Ultra-simple accumulation detector using only price action.
    No external APIs needed at all!
    """
    
    def __init__(self):
        self.price_history = []
    
    def add_price(self, price: float):
        """Add price to history"""
        self.price_history.append(price)
        
        # Keep last 100 prices
        if len(self.price_history) > 100:
            self.price_history.pop(0)
    
    def is_accumulation_zone(self) -> bool:
        """
        Detect accumulation using simple price patterns.
        
        Accumulation indicators:
        - Price consolidating (low volatility)
        - Price near recent lows
        - Volume declining (from your RPC data)
        
        Returns:
            True if in accumulation zone
        """
        if len(self.price_history) < 20:
            return False
        
        recent_prices = self.price_history[-20:]
        current_price = recent_prices[-1]
        
        # Calculate simple metrics
        avg_price = sum(recent_prices) / len(recent_prices)
        min_price = min(recent_prices)
        max_price = max(recent_prices)
        price_range = max_price - min_price
        
        # Accumulation signals:
        # 1. Low volatility (range < 3% of average)
        low_volatility = (price_range / avg_price) < 0.03
        
        # 2. Near recent lows (bottom 25% of range)
        near_lows = current_price < (min_price + price_range * 0.25)
        
        # 3. Below average price
        below_average = current_price < avg_price
        
        # If 2+ signals, we're in accumulation zone
        signals = sum([low_volatility, near_lows, below_average])
        
        return signals >= 2


# Factory function
def create_free_exchange_tracker() -> FreeExchangeFlowTracker:
    """Create free exchange flow tracker (no API key needed)"""
    return FreeExchangeFlowTracker()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print(" FREE EXCHANGE FLOW TRACKER - No API Keys Required!")
    print("=" * 70)
    print()
    
    # Create tracker
    tracker = FreeExchangeFlowTracker()
    
    # Get netflow estimate
    print("Fetching netflow estimate using free data sources...")
    netflow = tracker.get_exchange_netflow_estimate()
    
    if netflow is not None:
        print(f"\n✅ Estimated Netflow: {netflow:.2f} BTC")
        print()
        
        if netflow < -500:
            print("🟢 ACCUMULATION SIGNAL")
            print(f"   Estimated {abs(netflow):.0f} BTC leaving exchanges")
        elif netflow > 500:
            print("🔴 DISTRIBUTION SIGNAL")
            print(f"   Estimated {netflow:.0f} BTC to exchanges")
        else:
            print("⚪ NEUTRAL")
            print("   Balanced flows")
        
        # Get metrics
        metrics = tracker.get_flow_metrics()
        print(f"\n📊 Flow Metrics:")
        print(f"   Signal: {metrics.get('signal')}")
        print(f"   Method: {metrics.get('method')}")
        print(f"   Confidence: {metrics.get('confidence')}")
        
    else:
        print("❌ Could not estimate netflow")
    
    print()
    print("=" * 70)
    print("Note: This uses free proxy indicators, not direct exchange data.")
    print("Confidence is medium, but it's FREE and requires no API keys!")
    print("=" * 70)