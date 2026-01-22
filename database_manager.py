"""
SQLite database manager to replace JSON file storage.
Provides better concurrency, querying, and data integrity.
"""
import sqlite3
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from logger_config import logger
from core.exceptions import DataError


class DatabaseManager:
    """
    SQLite database manager for trading bot data.
    
    Tables:
    - price_history: OHLC candle data
    - trades: Executed trades
    - orders: Order history
    - strategy_logs: Strategy execution logs
    - bot_metrics: Performance metrics snapshots
    """
    
    def __init__(self, db_path: str = './trading_bot.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            # Price history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    timestamp REAL PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    vwap REAL,
                    volume REAL NOT NULL,
                    count INTEGER
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_history(timestamp)')
            
            # Trades table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    order_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    side TEXT NOT NULL,
                    volume REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    value REAL NOT NULL,
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_side (side)
                )
            ''')
            
            # Orders table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    side TEXT NOT NULL,
                    volume REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    executed_volume REAL DEFAULT 0,
                    average_price REAL,
                    fee REAL DEFAULT 0,
                    timeout INTEGER,
                    post_only INTEGER DEFAULT 0,
                    cancelled_at REAL,
                    filled_at REAL,
                    metadata TEXT,
                    INDEX idx_status (status),
                    INDEX idx_created (created_at)
                )
            ''')
            
            # Strategy logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS strategy_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    price REAL,
                    trade_volume REAL,
                    side TEXT,
                    reason TEXT,
                    rsi REAL,
                    macd REAL,
                    signal TEXT,
                    ma_short REAL,
                    ma_long REAL,
                    upper_band REAL,
                    lower_band REAL,
                    sentiment REAL,
                    fee_rate REAL,
                    netflow REAL,
                    volume REAL,
                    old_utxos INTEGER,
                    buy_decision INTEGER DEFAULT 0,
                    sell_decision INTEGER DEFAULT 0,
                    btc_balance REAL,
                    eur_balance REAL,
                    avg_buy_price REAL,
                    profit_margin REAL,
                    INDEX idx_log_timestamp (timestamp)
                )
            ''')
            
            # Bot metrics snapshots
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_metrics (
                    timestamp REAL PRIMARY KEY,
                    btc_balance REAL NOT NULL,
                    eur_balance REAL NOT NULL,
                    btc_price REAL NOT NULL,
                    total_equity_eur REAL NOT NULL,
                    daily_trades INTEGER DEFAULT 0,
                    pending_orders INTEGER DEFAULT 0,
                    INDEX idx_metrics_timestamp (timestamp)
                )
            ''')
        
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise DataError(f"Database operation failed: {e}")
        finally:
            conn.close()
    
    # Price History Methods
    
    def insert_ohlc_candles(self, candles: List[List[float]]) -> int:
        """
        Insert OHLC candles, ignoring duplicates.
        
        Args:
            candles: List of [timestamp, open, high, low, close, vwap, volume, count]
        
        Returns:
            Number of candles inserted
        """
        inserted = 0
        with self.get_connection() as conn:
            for candle in candles:
                try:
                    conn.execute('''
                        INSERT OR IGNORE INTO price_history
                        (timestamp, open, high, low, close, vwap, volume, count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', candle)
                    if conn.total_changes > 0:
                        inserted += 1
                except sqlite3.Error as e:
                    logger.debug(f"Failed to insert candle: {e}")
                    continue
        
        logger.debug(f"Inserted {inserted} new OHLC candles")
        return inserted
    
    def get_recent_prices(
        self,
        hours: int = 96,
        limit: Optional[int] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Get recent price and volume data.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of candles to return
        
        Returns:
            Tuple of (prices, volumes)
        """
        cutoff = time.time() - (hours * 3600)
        
        with self.get_connection() as conn:
            query = '''
                SELECT close, volume
                FROM price_history
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            if limit:
                query += f' LIMIT {limit}'
            
            cursor = conn.execute(query, (cutoff,))
            rows = cursor.fetchall()
        
        prices = [row['close'] for row in rows]
        volumes = [row['volume'] for row in rows]
        
        return prices, volumes
    
    def get_price_range(
        self,
        start_time: float,
        end_time: float
    ) -> Tuple[List[float], List[float]]:
        """Get prices in a specific time range"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT close, volume
                FROM price_history
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (start_time, end_time))
            rows = cursor.fetchall()
        
        prices = [row['close'] for row in rows]
        volumes = [row['volume'] for row in rows]
        
        return prices, volumes
    
    # Trade Methods
    
    def insert_trade(
        self,
        order_id: str,
        side: str,
        volume: float,
        price: float,
        fee: float,
        timestamp: Optional[float] = None
    ) -> bool:
        """Insert a completed trade"""
        if timestamp is None:
            timestamp = time.time()
        
        value = volume * price
        
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO trades
                    (order_id, timestamp, side, volume, price, fee, value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (order_id, timestamp, side, volume, price, fee, value))
            
            logger.debug(f"Recorded trade: {order_id} {side} {volume:.8f} BTC @ €{price:.2f}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to insert trade: {e}")
            return False
    
    def get_recent_trades(
        self,
        hours: int = 24,
        side: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent trades"""
        cutoff = time.time() - (hours * 3600)
        
        with self.get_connection() as conn:
            if side:
                cursor = conn.execute('''
                    SELECT * FROM trades
                    WHERE timestamp >= ? AND side = ?
                    ORDER BY timestamp DESC
                ''', (cutoff, side))
            else:
                cursor = conn.execute('''
                    SELECT * FROM trades
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (cutoff,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # Order Methods
    
    def upsert_order(self, order_data: Dict[str, Any]) -> bool:
        """Insert or update order"""
        try:
            with self.get_connection() as conn:
                # Serialize metadata if present
                metadata = order_data.get('metadata')
                if metadata and not isinstance(metadata, str):
                    metadata = json.dumps(metadata)
                
                conn.execute('''
                    INSERT OR REPLACE INTO orders
                    (order_id, created_at, updated_at, side, volume, price, status,
                     executed_volume, average_price, fee, timeout, post_only,
                     cancelled_at, filled_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order_data['order_id'],
                    order_data.get('created_at', order_data.get('timestamp', time.time())),
                    time.time(),
                    order_data['side'],
                    order_data['volume'],
                    order_data['price'],
                    order_data['status'],
                    order_data.get('executed_volume', 0),
                    order_data.get('average_price'),
                    order_data.get('fee', 0),
                    order_data.get('timeout'),
                    1 if order_data.get('post_only') else 0,
                    order_data.get('cancelled_at'),
                    order_data.get('filled_at'),
                    metadata
                ))
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to upsert order: {e}")
            return False
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM orders
                WHERE status IN ('pending', 'open', 'partially_filled')
                ORDER BY created_at ASC
            ''')
            
            orders = []
            for row in cursor.fetchall():
                order = dict(row)
                # Deserialize metadata
                if order['metadata']:
                    try:
                        order['metadata'] = json.loads(order['metadata'])
                    except:
                        pass
                orders.append(order)
            
            return orders
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics"""
        with self.get_connection() as conn:
            # Count by status
            cursor = conn.execute('''
                SELECT status, COUNT(*) as count
                FROM orders
                GROUP BY status
            ''')
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Average time to fill
            cursor = conn.execute('''
                SELECT AVG(filled_at - created_at) as avg_fill_time
                FROM orders
                WHERE filled_at IS NOT NULL
            ''')
            avg_fill_time = cursor.fetchone()['avg_fill_time'] or 0
            
            # Total fees
            cursor = conn.execute('''
                SELECT SUM(fee) as total_fees
                FROM orders
                WHERE status = 'filled'
            ''')
            total_fees = cursor.fetchone()['total_fees'] or 0
            
            # Fill rate
            total_orders = sum(status_counts.values())
            filled_orders = status_counts.get('filled', 0)
            fill_rate = filled_orders / total_orders if total_orders > 0 else 0
            
            return {
                'status_counts': status_counts,
                'avg_fill_time': avg_fill_time,
                'total_fees': total_fees,
                'fill_rate': fill_rate,
                'total_orders': total_orders
            }
    
    # Strategy Logs
    
    def log_strategy(self, **kwargs) -> bool:
        """Log strategy execution"""
        try:
            if 'timestamp' not in kwargs:
                kwargs['timestamp'] = datetime.now().isoformat()
            
            # Build SQL dynamically based on provided kwargs
            columns = list(kwargs.keys())
            placeholders = ','.join(['?' for _ in columns])
            column_names = ','.join(columns)
            values = [kwargs[col] for col in columns]
            
            with self.get_connection() as conn:
                conn.execute(f'''
                    INSERT INTO strategy_logs ({column_names})
                    VALUES ({placeholders})
                ''', values)
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to log strategy: {e}")
            return False
    
    # Metrics
    
    def record_metrics_snapshot(
        self,
        btc_balance: float,
        eur_balance: float,
        btc_price: float,
        daily_trades: int = 0,
        pending_orders: int = 0
    ) -> bool:
        """Record bot metrics snapshot"""
        try:
            timestamp = time.time()
            total_equity = eur_balance + (btc_balance * btc_price)
            
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO bot_metrics
                    (timestamp, btc_balance, eur_balance, btc_price, total_equity_eur,
                     daily_trades, pending_orders)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, btc_balance, eur_balance, btc_price, total_equity,
                      daily_trades, pending_orders))
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to record metrics: {e}")
            return False
    
    def get_equity_curve(self, hours: int = 168) -> List[Dict[str, Any]]:
        """Get equity curve data"""
        cutoff = time.time() - (hours * 3600)
        
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM bot_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            ''', (cutoff,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # Cleanup
    
    def cleanup_old_data(self, days: int = 90):
        """Remove data older than specified days"""
        cutoff = time.time() - (days * 86400)
        
        with self.get_connection() as conn:
            # Keep price history longer (1 year)
            conn.execute('DELETE FROM price_history WHERE timestamp < ?', (time.time() - 365*86400,))
            
            # Strategy logs
            conn.execute('DELETE FROM strategy_logs WHERE timestamp < ?',
                        (datetime.fromtimestamp(cutoff).isoformat(),))
            
            # Metrics snapshots
            conn.execute('DELETE FROM bot_metrics WHERE timestamp < ?', (cutoff,))
            
            # Old completed orders
            conn.execute('''
                DELETE FROM orders
                WHERE status IN ('filled', 'cancelled')
                AND updated_at < ?
            ''', (cutoff,))
        
        logger.info(f"Cleaned up data older than {days} days")
    
    def vacuum(self):
        """Reclaim disk space"""
        with self.get_connection() as conn:
            conn.execute('VACUUM')
        logger.info("Database vacuumed")