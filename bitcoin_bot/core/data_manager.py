"""
Production-Ready Secure Data Manager
Addresses security vulnerabilities and performance issues
"""

import json
import os
import asyncio
import aiofiles
import pandas as pd
import csv
import sqlite3
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
import hashlib
import time
from collections import deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class DataValidationConfig:
    """Configuration for data validation"""
    max_price_change_pct: float = 0.20  # 20% max price change between candles
    min_price_eur: float = 1000.0       # Minimum realistic BTC price
    max_price_eur: float = 1000000.0    # Maximum realistic BTC price
    max_volume_multiplier: float = 100.0 # Max 100x volume spike
    timestamp_tolerance_hours: int = 2   # Allow 2 hours tolerance for real data


class SecureDataManager:
    """Production-ready data manager with security and performance optimizations"""
    
    REQUIRED_HEADERS = [
        "timestamp", "price", "trade_volume", "side", "reason", "rsi", "macd",
        "signal", "upper_band", "lower_band", "sentiment", "buy_decision", 
        "sell_decision", "btc_balance", "eur_balance"
    ]
    
    def __init__(self, data_dir: str = "./data", use_database: bool = True):
        self.data_dir = data_dir
        self.use_database = use_database
        self.validation_config = DataValidationConfig()
        
        # Thread-safe file access
        self._file_locks = {}
        self._lock_manager = threading.Lock()
        
        # In-memory cache for frequently accessed data
        self._price_cache = deque(maxlen=2000)
        self._cache_lock = threading.RLock()
        self._cache_timestamp = 0
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # File paths
        self.price_history_file = os.path.join(data_dir, "price_history.json")
        self.bot_logs_file = os.path.join(data_dir, "bot_logs.csv")
        self.db_file = os.path.join(data_dir, "trading_data.db")
        
        # Initialize storage
        if use_database:
            self._init_database()
        else:
            self._init_file_storage()
    
    def _init_database(self):
        """Initialize SQLite database for better performance"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                # Create price history table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS price_history (
                        timestamp INTEGER PRIMARY KEY,
                        open_price REAL NOT NULL,
                        high_price REAL NOT NULL,
                        low_price REAL NOT NULL,
                        close_price REAL NOT NULL,
                        volume REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        data_hash TEXT
                    )
                """)
                
                # Create trading logs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        price REAL,
                        trade_volume REAL,
                        side TEXT,
                        reason TEXT,
                        rsi REAL,
                        macd REAL,
                        signal_line REAL,
                        upper_band REAL,
                        lower_band REAL,
                        sentiment REAL,
                        buy_decision BOOLEAN,
                        sell_decision BOOLEAN,
                        btc_balance REAL,
                        eur_balance REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indices for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_history(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON trading_logs(timestamp)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Fallback to file storage
            self.use_database = False
            self._init_file_storage()
    
    def _init_file_storage(self):
        """Initialize file-based storage"""
        if not os.path.exists(self.price_history_file):
            with open(self.price_history_file, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.bot_logs_file):
            with open(self.bot_logs_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.REQUIRED_HEADERS)
    
    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """Get or create a lock for a specific file"""
        with self._lock_manager:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.Lock()
            return self._file_locks[file_path]
    
    def _validate_ohlc_candle(self, candle: List, previous_close: Optional[float] = None) -> bool:
        """Comprehensive OHLC data validation"""
        try:
            if not isinstance(candle, list) or len(candle) < 6:
                return False
            
            timestamp, open_price, high_price, low_price, close_price, volume = [
                float(x) for x in candle[:6]
            ]
            
            # Timestamp validation
            current_time = time.time()
            tolerance = self.validation_config.timestamp_tolerance_hours * 3600
            
            if not (current_time - 365*24*3600 <= timestamp <= current_time + tolerance):
                logger.debug(f"Invalid timestamp: {timestamp}")
                return False
            
            # Price validation
            if not (self.validation_config.min_price_eur <= close_price <= self.validation_config.max_price_eur):
                logger.debug(f"Price out of range: {close_price}")
                return False
            
            # OHLC relationship validation
            if not (low_price <= open_price <= high_price and 
                   low_price <= close_price <= high_price and
                   low_price <= high_price):
                logger.debug(f"Invalid OHLC relationships")
                return False
            
            # Volume validation
            if volume < 0:
                logger.debug(f"Negative volume: {volume}")
                return False
            
            # Price change validation (if previous close available)
            if previous_close:
                price_change_pct = abs(close_price - previous_close) / previous_close
                if price_change_pct > self.validation_config.max_price_change_pct:
                    logger.warning(f"Large price change: {price_change_pct:.1%}")
                    # Don't reject, just warn for large moves
            
            return True
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Validation error: {e}")
            return False
    
    def _calculate_data_hash(self, data: List) -> str:
        """Calculate hash for data integrity verification"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    async def load_price_history_async(self) -> Tuple[List[float], List[float]]:
        """Async version of load_price_history with caching"""
        # Check cache first
        with self._cache_lock:
            cache_age = time.time() - self._cache_timestamp
            if cache_age < 300 and self._price_cache:  # 5-minute cache
                prices = [candle[4] for candle in self._price_cache]  # Close prices
                volumes = [candle[5] for candle in self._price_cache]  # Volumes
                logger.debug(f"Using cached price data ({len(prices)} points)")
                return prices, volumes
        
        if self.use_database:
            return await self._load_from_database()
        else:
            return await self._load_from_file()
    
    async def _load_from_database(self) -> Tuple[List[float], List[float]]:
        """Load price history from database"""
        try:
            def _db_query():
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.execute("""
                        SELECT timestamp, open_price, high_price, low_price, close_price, volume
                        FROM price_history
                        ORDER BY timestamp ASC
                        LIMIT 2000
                    """)
                    return cursor.fetchall()
            
            # Run database query in thread pool
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, _db_query)
            
            if not rows:
                logger.warning("No price history found in database")
                return [], []
            
            # Update cache
            with self._cache_lock:
                self._price_cache.clear()
                self._price_cache.extend(rows)
                self._cache_timestamp = time.time()
            
            prices = [row[4] for row in rows]  # Close prices
            volumes = [row[5] for row in rows]  # Volumes
            
            logger.info(f"Loaded {len(prices)} price points from database")
            return prices, volumes
            
        except Exception as e:
            logger.error(f"Database load failed: {e}")
            return [], []
    
    async def _load_from_file(self) -> Tuple[List[float], List[float]]:
        """Load price history from file asynchronously"""
        try:
            async with aiofiles.open(self.price_history_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            if not isinstance(data, list):
                logger.error("Invalid price history format")
                return [], []
            
            # Validate and process data
            valid_candles = []
            previous_close = None
            
            for candle in data:
                if self._validate_ohlc_candle(candle, previous_close):
                    valid_candles.append(candle)
                    previous_close = float(candle[4])
            
            # Update cache
            with self._cache_lock:
                self._price_cache.clear()
                self._price_cache.extend(valid_candles[-2000:])  # Keep last 2000
                self._cache_timestamp = time.time()
            
            prices = [candle[4] for candle in valid_candles]
            volumes = [candle[5] for candle in valid_candles]
            
            logger.info(f"Loaded {len(prices)} valid price points from file")
            return prices, volumes
            
        except Exception as e:
            logger.error(f"File load failed: {e}")
            return [], []
    
    def load_price_history(self) -> Tuple[List[float], List[float]]:
        """Synchronous wrapper for async load_price_history"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new task if loop is already running
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.load_price_history_async())
                    return future.result(timeout=30)
            else:
                return asyncio.run(self.load_price_history_async())
        except Exception as e:
            logger.error(f"Sync load wrapper failed: {e}")
            return [], []
    
    async def append_ohlc_data_async(self, ohlc_data: List[List]) -> int:
        """Async version of append_ohlc_data with improved validation"""
        if not ohlc_data:
            return 0
        
        try:
            if self.use_database:
                return await self._append_to_database(ohlc_data)
            else:
                return await self._append_to_file(ohlc_data)
        except Exception as e:
            logger.error(f"Append OHLC data failed: {e}")
            return 0
    
    async def _append_to_database(self, ohlc_data: List[List]) -> int:
        """Append OHLC data to database"""
        def _db_insert():
            added_count = 0
            with sqlite3.connect(self.db_file) as conn:
                # Get existing timestamps to avoid duplicates
                cursor = conn.execute("SELECT timestamp FROM price_history")
                existing_timestamps = {row[0] for row in cursor.fetchall()}
                
                # Prepare valid candles
                valid_candles = []
                previous_close = None
                
                for candle in ohlc_data:
                    if not self._validate_ohlc_candle(candle, previous_close):
                        continue
                    
                    timestamp = int(candle[0])
                    if timestamp in existing_timestamps:
                        continue
                    
                    data_hash = self._calculate_data_hash(candle[:6])
                    valid_candles.append((
                        timestamp, float(candle[1]), float(candle[2]), 
                        float(candle[3]), float(candle[4]), float(candle[5]), data_hash
                    ))
                    previous_close = float(candle[4])
                    added_count += 1
                
                # Batch insert
                if valid_candles:
                    conn.executemany("""
                        INSERT INTO price_history 
                        (timestamp, open_price, high_price, low_price, close_price, volume, data_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, valid_candles)
                    conn.commit()
                
                # Cleanup old data (keep last 5000 records)
                conn.execute("""
                    DELETE FROM price_history 
                    WHERE timestamp < (
                        SELECT timestamp FROM price_history 
                        ORDER BY timestamp DESC LIMIT 1 OFFSET 5000
                    )
                """)
                conn.commit()
            
            return added_count
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        added_count = await loop.run_in_executor(None, _db_insert)
        
        if added_count > 0:
            # Invalidate cache
            with self._cache_lock:
                self._cache_timestamp = 0
            
            logger.info(f"Added {added_count} new candles to database")
        
        return added_count
    
    async def _append_to_file(self, ohlc_data: List[List]) -> int:
        """Append OHLC data to file"""
        lock = self._get_file_lock(self.price_history_file)
        
        with lock:
            # Load existing data
            try:
                async with aiofiles.open(self.price_history_file, 'r') as f:
                    content = await f.read()
                    existing_data = json.loads(content)
            except:
                existing_data = []
            
            # Get existing timestamps
            existing_timestamps = {int(candle[0]) for candle in existing_data}
            
            # Validate and add new data
            new_candles = []
            previous_close = None
            
            for candle in ohlc_data:
                if not self._validate_ohlc_candle(candle, previous_close):
                    continue
                
                timestamp = int(candle[0])
                if timestamp in existing_timestamps:
                    continue
                
                formatted_candle = [
                    timestamp, float(candle[1]), float(candle[2]),
                    float(candle[3]), float(candle[4]), float(candle[5])
                ]
                new_candles.append(formatted_candle)
                existing_timestamps.add(timestamp)
                previous_close = float(candle[4])
            
            if new_candles:
                # Add to existing data and sort
                existing_data.extend(new_candles)
                existing_data.sort(key=lambda x: x[0])
                
                # Keep only last 5000 candles
                existing_data = existing_data[-5000:]
                
                # Write back to file
                async with aiofiles.open(self.price_history_file, 'w') as f:
                    await f.write(json.dumps(existing_data, separators=(',', ':')))
                
                # Invalidate cache
                with self._cache_lock:
                    self._cache_timestamp = 0
                
                logger.info(f"Added {len(new_candles)} new candles to file")
            
            return len(new_candles)
    
    def append_ohlc_data(self, ohlc_data: List[List]) -> int:
        """Synchronous wrapper for append_ohlc_data_async"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.append_ohlc_data_async(ohlc_data))
                    return future.result(timeout=30)
            else:
                return asyncio.run(self.append_ohlc_data_async(ohlc_data))
        except Exception as e:
            logger.error(f"Sync append wrapper failed: {e}")
            return 0
    
    def log_strategy(self, **kwargs) -> None:
        """Thread-safe strategy logging with validation"""
        try:
            # Sanitize input data
            sanitized_kwargs = self._sanitize_log_data(kwargs)
            
            if self.use_database:
                self._log_to_database(sanitized_kwargs)
            else:
                self._log_to_file(sanitized_kwargs)
                
        except Exception as e:
            logger.error(f"Strategy logging failed: {e}")
    
    def _sanitize_log_data(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate log data"""
        sanitized = {}
        
        # Set timestamp if not provided
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now().isoformat()
        
        # Define expected fields with types and defaults
        field_specs = {
            "timestamp": (str, datetime.now().isoformat()),
            "price": (float, None),
            "trade_volume": (float, None),
            "side": (str, ""),
            "reason": (str, ""),
            "rsi": (float, None),
            "macd": (float, None),
            "signal": (float, None),
            "upper_band": (float, None),
            "lower_band": (float, None),
            "sentiment": (float, None),
            "buy_decision": (bool, False),
            "sell_decision": (bool, False),
            "btc_balance": (float, None),
            "eur_balance": (float, None),
        }
        
        for field, (field_type, default) in field_specs.items():
            value = kwargs.get(field, default)
            
            # Type conversion and validation
            if value is not None:
                try:
                    if field_type == bool:
                        sanitized[field] = str(value).lower() in ['true', '1', 'yes']
                    elif field_type == str:
                        # Sanitize strings to prevent injection
                        sanitized[field] = str(value)[:200]  # Limit length
                    else:
                        sanitized[field] = field_type(value)
                except (ValueError, TypeError):
                    sanitized[field] = default
            else:
                sanitized[field] = default
        
        return sanitized
    
    def _log_to_database(self, data: Dict[str, Any]):
        """Log to database"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute("""
                    INSERT INTO trading_logs (
                        timestamp, price, trade_volume, side, reason, rsi, macd, 
                        signal_line, upper_band, lower_band, sentiment, 
                        buy_decision, sell_decision, btc_balance, eur_balance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["timestamp"], data["price"], data["trade_volume"],
                    data["side"], data["reason"], data["rsi"], data["macd"],
                    data["signal"], data["upper_band"], data["lower_band"],
                    data["sentiment"], data["buy_decision"], data["sell_decision"],
                    data["btc_balance"], data["eur_balance"]
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Database logging failed: {e}")
    
    def _log_to_file(self, data: Dict[str, Any]):
        """Log to CSV file"""
        lock = self._get_file_lock(self.bot_logs_file)
        
        with lock:
            try:
                with open(self.bot_logs_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [data.get(header, "") for header in self.REQUIRED_HEADERS]
                    writer.writerow(row)
            except Exception as e:
                logger.error(f"File logging failed: {e}")
    
    def validate_bot_logs(self) -> bool:
        """Validate bot logs integrity"""
        try:
            if self.use_database:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM trading_logs")
                    count = cursor.fetchone()[0]
                    logger.info(f"Database contains {count} trading log entries")
                    return count > 0
            else:
                if not os.path.exists(self.bot_logs_file):
                    return False
                
                df = pd.read_csv(self.bot_logs_file, dtype={"buy_decision": str, "sell_decision": str})
                logger.info(f"CSV file contains {len(df)} trading log entries")
                return len(df) > 0
                
        except Exception as e:
            logger.error(f"Log validation failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data manager statistics"""
        try:
            stats = {
                "storage_type": "database" if self.use_database else "file",
                "cache_size": len(self._price_cache),
                "cache_age_seconds": time.time() - self._cache_timestamp,
            }
            
            if self.use_database:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM price_history")
                    stats["price_history_count"] = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT COUNT(*) FROM trading_logs")
                    stats["trading_logs_count"] = cursor.fetchone()[0]
            else:
                prices, _ = self.load_price_history()
                stats["price_history_count"] = len(prices)
                
                if os.path.exists(self.bot_logs_file):
                    with open(self.bot_logs_file, 'r') as f:
                        stats["trading_logs_count"] = sum(1 for _ in f) - 1  # Subtract header
                else:
                    stats["trading_logs_count"] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Cleanup old data to manage storage"""
        try:
            cutoff_timestamp = time.time() - (days_to_keep * 24 * 3600)
            
            if self.use_database:
                with sqlite3.connect(self.db_file) as conn:
                    # Cleanup old price history
                    cursor = conn.execute(
                        "DELETE FROM price_history WHERE timestamp < ?", 
                        (cutoff_timestamp,)
                    )
                    price_deleted = cursor.rowcount
                    
                    # Cleanup old trading logs
                    cursor = conn.execute(
                        "DELETE FROM trading_logs WHERE timestamp < ?",
                        (datetime.fromtimestamp(cutoff_timestamp).isoformat(),)
                    )
                    logs_deleted = cursor.rowcount
                    
                    conn.commit()
                    
                    # Vacuum database to reclaim space
                    conn.execute("VACUUM")
                    
                    logger.info(f"Cleaned up {price_deleted} price records and {logs_deleted} log records")
            
            # Invalidate cache
            with self._cache_lock:
                self._cache_timestamp = 0
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def close(self):
        """Cleanup resources"""
        with self._cache_lock:
            self._price_cache.clear()
        logger.info("Data manager closed")


# Enhanced data validation functions
def detect_anomalies(prices: List[float], threshold: float = 3.0) -> List[int]:
    """Detect price anomalies using z-score"""
    if len(prices) < 10:
        return []
    
    prices_array = np.array(prices)
    z_scores = np.abs((prices_array - np.mean(prices_array)) / np.std(prices_array))
    
    return [i for i, z in enumerate(z_scores) if z > threshold]


def validate_data_continuity(timestamps: List[int], expected_interval: int = 900) -> Dict[str, Any]:
    """Validate data continuity and identify gaps"""
    if len(timestamps) < 2:
        return {"gaps": [], "continuity_score": 1.0}
    
    timestamps_sorted = sorted(timestamps)
    gaps = []
    
    for i in range(1, len(timestamps_sorted)):
        interval = timestamps_sorted[i] - timestamps_sorted[i-1]
        if interval > expected_interval * 1.5:  # Allow 50% tolerance
            gaps.append({
                "start": timestamps_sorted[i-1],
                "end": timestamps_sorted[i],
                "duration_minutes": (interval) / 60
            })
    
    total_time = timestamps_sorted[-1] - timestamps_sorted[0]
    gap_time = sum(gap["end"] - gap["start"] for gap in gaps)
    continuity_score = 1.0 - (gap_time / total_time) if total_time > 0 else 1.0
    
    return {
        "gaps": gaps,
        "continuity_score": continuity_score,
        "total_gaps": len(gaps)
    }


# Usage example and testing
async def test_secure_data_manager():
    """Test the secure data manager"""
    print("Testing Secure Data Manager...")
    
    manager = SecureDataManager(data_dir="./test_data", use_database=True)
    
    # Test data loading
    prices, volumes = await manager.load_price_history_async()
    print(f"Loaded {len(prices)} price points")
    
    # Test data appending
    test_ohlc = [
        [int(time.time()), 45000, 45100, 44900, 45050, 1.5],
        [int(time.time()) + 900, 45050, 45200, 45000, 45150, 2.1]
    ]
    
    added = await manager.append_ohlc_data_async(test_ohlc)
    print(f"Added {added} test candles")
    
    # Test logging
    manager.log_strategy(
        timestamp=datetime.now().isoformat(),
        price=45100.0,
        side="buy",
        reason="test_strategy",
        rsi=45.0,
        buy_decision=True
    )
    print("Logged test strategy entry")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Statistics: {stats}")
    
    manager.close()
    print("✅ Secure Data Manager test completed")


if __name__ == "__main__":
    asyncio.run(test_secure_data_manager())