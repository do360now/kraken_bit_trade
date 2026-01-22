"""
Trade History Management - Fetch from Exchange Instead of Local Tracking

This approach is MORE RELIABLE because:
1. Exchange is the source of truth
2. No risk of missing trades due to bot crashes
3. No need to maintain local state
4. Can reconstruct history at any time
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time

from logger_config import logger


class TradeHistoryManager:
    """
    Manages trade history by querying the exchange.
    Better than local tracking because exchange is source of truth.
    """
    
    def __init__(self, kraken_api):
        self.api = kraken_api
        self._cache = {}
        self._cache_time = 0
        self._cache_duration = 300  # 5 minutes
    
    def get_all_trades(
        self,
        pair: str = "XXBTZEUR",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get all trades from Kraken.
        
        Args:
            pair: Trading pair (default: XXBTZEUR)
            start_time: Unix timestamp to start from
            end_time: Unix timestamp to end at
            use_cache: Whether to use cached results
        
        Returns:
            List of trade dicts with keys:
            - trade_id: Trade ID
            - order_id: Order that generated this trade
            - pair: Trading pair
            - time: Unix timestamp
            - type: 'buy' or 'sell'
            - price: Execution price
            - volume: Trade volume in BTC
            - cost: Total cost in EUR
            - fee: Fee paid
            - margin: Margin amount (0 for spot)
        """
        # Check cache
        cache_key = f"{pair}_{start_time}_{end_time}"
        if use_cache and cache_key in self._cache:
            if time.time() - self._cache_time < self._cache_duration:
                logger.debug("Returning cached trade history")
                return self._cache[cache_key]
        
        try:
            # Fetch from Kraken
            data = {}
            if start_time:
                data['start'] = start_time
            if end_time:
                data['end'] = end_time
            
            result = self.api.query_private("TradesHistory", data)
            
            if result.get('error'):
                logger.error(f"Failed to fetch trade history: {result['error']}")
                return []
            
            trades_data = result.get('result', {}).get('trades', {})
            
            # Convert to list and filter by pair
            trades = []
            for trade_id, trade_info in trades_data.items():
                if trade_info['pair'] != pair:
                    continue
                
                trades.append({
                    'trade_id': trade_id,
                    'order_id': trade_info.get('ordertxid', ''),
                    'pair': trade_info['pair'],
                    'time': float(trade_info['time']),
                    'type': trade_info['type'],  # 'buy' or 'sell'
                    'price': float(trade_info['price']),
                    'volume': float(trade_info['vol']),
                    'cost': float(trade_info['cost']),
                    'fee': float(trade_info['fee']),
                    'margin': float(trade_info.get('margin', 0))
                })
            
            # Sort by time
            trades.sort(key=lambda t: t['time'])
            
            # Cache results
            self._cache[cache_key] = trades
            self._cache_time = time.time()
            
            logger.info(f"Fetched {len(trades)} trades from Kraken")
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trade history: {e}", exc_info=True)
            return []
    
    def get_recent_trades(
        self,
        pair: str = "XXBTZEUR",
        hours: int = 24
    ) -> List[Dict]:
        """Get trades from the last N hours"""
        start_time = int(time.time() - (hours * 3600))
        return self.get_all_trades(pair=pair, start_time=start_time)
    
    def get_trades_by_date_range(
        self,
        pair: str = "XXBTZEUR",
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Dict]:
        """
        Get trades within a date range.
        
        Args:
            pair: Trading pair
            start_date: Start date (datetime object)
            end_date: End date (datetime object)
        """
        start_time = int(start_date.timestamp()) if start_date else None
        end_time = int(end_date.timestamp()) if end_date else None
        
        return self.get_all_trades(
            pair=pair,
            start_time=start_time,
            end_time=end_time
        )
    
    def calculate_average_buy_price(
        self,
        pair: str = "XXBTZEUR",
        hours: int = None
    ) -> Optional[float]:
        """
        Calculate average buy price from exchange history.
        
        This is the SOURCE OF TRUTH, not local tracking!
        
        Args:
            pair: Trading pair
            hours: Look back this many hours (None = all time)
        
        Returns:
            Average buy price or None if no buys
        """
        if hours:
            trades = self.get_recent_trades(pair=pair, hours=hours)
        else:
            trades = self.get_all_trades(pair=pair)
        
        buy_trades = [t for t in trades if t['type'] == 'buy']
        
        if not buy_trades:
            return None
        
        total_cost = sum(t['cost'] for t in buy_trades)
        total_volume = sum(t['volume'] for t in buy_trades)
        
        if total_volume == 0:
            return None
        
        avg_price = total_cost / total_volume
        
        logger.info(
            f"Calculated average buy price from {len(buy_trades)} trades",
            avg_price=avg_price,
            total_volume=total_volume,
            total_cost=total_cost
        )
        
        return avg_price
    
    def calculate_current_position(
        self,
        pair: str = "XXBTZEUR",
        hours: int = None
    ) -> Dict[str, float]:
        """
        Calculate current position from trade history.
        
        Returns:
            - net_volume: Net BTC position (buys - sells)
            - total_cost: Total EUR spent
            - avg_buy_price: Average buy price
            - realized_pnl: Realized profit/loss
        """
        if hours:
            trades = self.get_recent_trades(pair=pair, hours=hours)
        else:
            trades = self.get_all_trades(pair=pair)
        
        buy_volume = 0
        buy_cost = 0
        sell_volume = 0
        sell_revenue = 0
        total_fees = 0
        
        for trade in trades:
            total_fees += trade['fee']
            
            if trade['type'] == 'buy':
                buy_volume += trade['volume']
                buy_cost += trade['cost']
            else:  # sell
                sell_volume += trade['volume']
                sell_revenue += trade['cost']
        
        net_volume = buy_volume - sell_volume
        net_cost = buy_cost - sell_revenue
        avg_buy_price = buy_cost / buy_volume if buy_volume > 0 else 0
        
        # Realized P&L: revenue from sells minus cost basis of those sells
        # Simplified: sell_revenue - (avg_buy_price * sell_volume)
        realized_pnl = sell_revenue - (avg_buy_price * sell_volume) if avg_buy_price > 0 else 0
        
        return {
            'net_volume': net_volume,
            'total_cost': net_cost,
            'avg_buy_price': avg_buy_price,
            'realized_pnl': realized_pnl - total_fees,
            'total_fees': total_fees,
            'buy_count': len([t for t in trades if t['type'] == 'buy']),
            'sell_count': len([t for t in trades if t['type'] == 'sell'])
        }
    
    def get_trades_summary(
        self,
        pair: str = "XXBTZEUR",
        hours: int = 24
    ) -> Dict:
        """Get summary statistics of recent trades"""
        trades = self.get_recent_trades(pair=pair, hours=hours)
        
        if not trades:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_volume': 0,
                'total_fees': 0
            }
        
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        return {
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_volume': sum(t['volume'] for t in trades),
            'buy_volume': sum(t['volume'] for t in buy_trades),
            'sell_volume': sum(t['volume'] for t in sell_trades),
            'total_fees': sum(t['fee'] for t in trades),
            'avg_buy_price': sum(t['price'] for t in buy_trades) / len(buy_trades) if buy_trades else 0,
            'avg_sell_price': sum(t['price'] for t in sell_trades) / len(sell_trades) if sell_trades else 0,
            'first_trade_time': datetime.fromtimestamp(trades[0]['time']),
            'last_trade_time': datetime.fromtimestamp(trades[-1]['time'])
        }
    
    def find_matching_buy_for_sell(
        self,
        sell_trade: Dict,
        all_trades: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        """
        Find the buy trade that corresponds to a sell (for P&L calculation).
        Uses FIFO (First In, First Out) matching.
        
        Args:
            sell_trade: The sell trade to match
            all_trades: All trades sorted by time
        
        Returns:
            (matched_buy_trade, profit_or_loss)
        """
        if sell_trade['type'] != 'sell':
            return None, 0.0
        
        # Find all buys before this sell
        buy_trades = [
            t for t in all_trades
            if t['type'] == 'buy' and t['time'] < sell_trade['time']
        ]
        
        if not buy_trades:
            return None, 0.0
        
        # Use FIFO - oldest buy
        matched_buy = buy_trades[0]
        
        # Calculate P&L
        buy_cost = matched_buy['volume'] * matched_buy['price']
        sell_revenue = sell_trade['volume'] * sell_trade['price']
        pnl = sell_revenue - buy_cost - sell_trade['fee']
        
        return matched_buy, pnl
    
    def export_to_csv(
        self,
        filename: str = "trade_history.csv",
        pair: str = "XXBTZEUR",
        hours: int = None
    ):
        """Export trade history to CSV for analysis"""
        import csv
        
        if hours:
            trades = self.get_recent_trades(pair=pair, hours=hours)
        else:
            trades = self.get_all_trades(pair=pair)
        
        if not trades:
            logger.warning("No trades to export")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        
        logger.info(f"Exported {len(trades)} trades to {filename}")


# Example usage in your bot
def example_usage():
    """How to use TradeHistoryManager in your trading bot"""
    from kraken_api import KrakenAPI
    from config import API_KEY, API_SECRET
    
    # Initialize
    api = KrakenAPI(API_KEY, API_SECRET)
    trade_history = TradeHistoryManager(api)
    
    # Get recent trades
    recent_trades = trade_history.get_recent_trades(hours=24)
    print(f"Found {len(recent_trades)} trades in last 24 hours")
    
    # Calculate average buy price (SOURCE OF TRUTH!)
    avg_buy_price = trade_history.calculate_average_buy_price()
    print(f"Average buy price: €{avg_buy_price:.2f}")
    
    # Get current position
    position = trade_history.calculate_current_position()
    print(f"Net BTC position: {position['net_volume']:.8f}")
    print(f"Average buy price: €{position['avg_buy_price']:.2f}")
    print(f"Realized P&L: €{position['realized_pnl']:.2f}")
    
    # Get summary
    summary = trade_history.get_trades_summary(hours=24)
    print(f"24h Summary: {summary['total_trades']} trades")
    print(f"  Buys: {summary['buy_trades']}, Sells: {summary['sell_trades']}")
    print(f"  Total fees: €{summary['total_fees']:.2f}")
    
    # Export to CSV for analysis
    trade_history.export_to_csv("my_trades.csv", hours=168)  # Last week


if __name__ == "__main__":
    example_usage()