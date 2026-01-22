"""
Constants and enums for the trading bot.
Replaces magic numbers and strings with named constants.
"""
from enum import Enum, IntEnum, auto


class OHLCField(IntEnum):
    """OHLC candle data field indices"""
    TIMESTAMP = 0
    OPEN = 1
    HIGH = 2
    LOW = 3
    CLOSE = 4
    VWAP = 5
    VOLUME = 6
    COUNT = 7


MIN_OHLC_LENGTH = 7


class OrderState(Enum):
    """Order lifecycle states"""
    PENDING = auto()           # Order created, not yet submitted
    SUBMITTED = auto()         # Submitted to exchange
    OPEN = auto()              # Confirmed by exchange
    PARTIALLY_FILLED = auto()  # Partial execution
    FILLED = auto()            # Fully executed
    CANCELLED = auto()         # Cancelled
    FAILED = auto()            # Failed to place
    EXPIRED = auto()           # Timed out


class OrderSide(Enum):
    """Order side types"""
    BUY = "buy"
    SELL = "sell"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class EventType(Enum):
    """Event types for event-driven architecture"""
    ORDER_PLACED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_FAILED = auto()
    PRICE_UPDATED = auto()
    BALANCE_CHANGED = auto()
    SIGNAL_TRIGGERED = auto()
    ERROR_OCCURRED = auto()


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class HealthStatus(Enum):
    """Health check statuses"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Bitcoin constants
SATOSHIS_PER_BTC = 100_000_000

# Time constants (seconds)
MINUTE = 60
HOUR = 3600
DAY = 86400
WEEK = 604800

# Cache durations (seconds)
DEFAULT_CACHE_TTL = HOUR
BALANCE_CACHE_DURATION = 15 * MINUTE
ONCHAIN_CACHE_DURATION = MINUTE
PRICE_CACHE_DURATION = MINUTE

# API rate limits
DEFAULT_REQUEST_INTERVAL = 1  # seconds between requests
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 2  # exponential backoff base

# File paths
DEFAULT_LOG_FILE = "trading_bot.log"
DEFAULT_PRICE_HISTORY_FILE = "./price_history.json"
DEFAULT_BOT_LOGS_FILE = "./bot_logs.csv"
DEFAULT_ORDER_HISTORY_FILE = "./order_history.json"
DEFAULT_PERFORMANCE_FILE = "./performance_history.json"
DEFAULT_DB_FILE = "./trading_bot.db"

# Trading pair
DEFAULT_TRADING_PAIR = "XXBTZEUR"

# Validation constants
MAX_FUTURE_TIMESTAMP_TOLERANCE = HOUR  # 1 hour in future
MAX_PAST_TIMESTAMP_TOLERANCE = 365 * DAY  # 1 year in past

# Order timeouts
DEFAULT_ORDER_TIMEOUT = 300  # 5 minutes
MAX_ORDER_AGE_BEFORE_CLEANUP = 10 * MINUTE

# Cache sizes
MAX_BLOCK_CACHE_SIZE = 20
MAX_PRICE_HISTORY_MEMORY = 10000  # Keep last 10k prices in memory

# Metrics server
DEFAULT_METRICS_PORT = 8080
METRICS_BIND_ADDRESS = "127.0.0.1"  # Localhost only for security

# Transaction sampling
MEMPOOL_SAMPLE_SIZE = 30
BLOCK_TX_SAMPLE_SIZE = 100

# Kraken-specific
KRAKEN_API_VERSION = "0"
KRAKEN_BTC_KEYS = ['XXBT', 'XBT', 'XBT.F', 'XBTC']
KRAKEN_EUR_KEYS = ['ZEUR', 'EUR', 'EUR.F']