"""
Configuration module with validation.
Loads settings from environment variables and validates them at startup.
"""
import os
import sys
from typing import Dict, List, Any
from dotenv import load_dotenv
from core.exceptions import ConfigurationError

# Load environment variables
load_dotenv()


class Config:
    """Configuration with validation"""
    
    # API credentials
    API_KEY = os.getenv("KRAKEN_API_KEY") or os.getenv("API_KEY")
    API_SECRET = os.getenv("KRAKEN_API_SECRET") or os.getenv("API_SECRET")
    API_DOMAIN = os.getenv("API_DOMAIN", "https://api.kraken.com")
    
    # Allocation strategy
    ALLOC_HODL = float(os.getenv("ALLOC_HODL", "0.9"))
    ALLOC_YIELD = float(os.getenv("ALLOC_YIELD", "0.0"))
    ALLOC_TRADING = float(os.getenv("ALLOC_TRADING", "0.1"))
    
    ALLOCATIONS = {
        'HODL': ALLOC_HODL,
        'YIELD': ALLOC_YIELD,
        'TRADING': ALLOC_TRADING,
    }
    
    # Bitcoin node RPC
    RPC_USER = os.getenv("RPC_USER")
    RPC_PASSWORD = os.getenv("RPC_PASSWORD")
    RPC_HOST = os.getenv("RPC_HOST", "localhost")
    RPC_PORT = os.getenv("RPC_PORT", "8332")
    
    # Trading parameters
    TOTAL_BTC = float(os.getenv("TOTAL_BTC", "0.01"))
    MIN_TRADE_VOLUME = float(os.getenv("MIN_TRADE_VOLUME", "0.00005"))
    MIN_EUR_FOR_TRADE = float(os.getenv("MIN_EUR_FOR_TRADE", "15.0"))
    MIN_BTC = float(os.getenv("MIN_BTC", "10"))
    
    # Timing
    GLOBAL_TRADE_COOLDOWN = int(os.getenv("GLOBAL_TRADE_COOLDOWN", "180"))
    SLEEP_DURATION = int(os.getenv("SLEEP_DURATION", "900"))
    
    # Cache durations
    BALANCE_CACHE_DURATION = int(os.getenv("BALANCE_CACHE_DURATION", "900"))
    ONCHAIN_CACHE_DURATION = int(os.getenv("ONCHAIN_CACHE_DURATION", "60"))
    
    # Risk management
    MAX_CASH_ALLOCATION = float(os.getenv("MAX_CASH_ALLOCATION", "0.8"))
    MAX_SELL_ALLOCATION = float(os.getenv("MAX_SELL_ALLOCATION", "0.5"))
    MIN_PROFIT_MARGIN = float(os.getenv("MIN_PROFIT_MARGIN", "0.05"))
    
    # Stop loss and take profit
    USE_STOP_LOSS = os.getenv("USE_STOP_LOSS", "true").lower() == "true"
    STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.03"))
    USE_TAKE_PROFIT = os.getenv("USE_TAKE_PROFIT", "true").lower() == "true"
    TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.08"))
    
    # Enhanced risk parameters
    ENABLE_DYNAMIC_STOPS = os.getenv("ENABLE_DYNAMIC_STOPS", "true").lower() == "true"
    BASE_STOP_LOSS_PCT = float(os.getenv("BASE_STOP_LOSS_PCT", "0.03"))
    MAX_RISK_OFF_THRESHOLD = float(os.getenv("MAX_RISK_OFF_THRESHOLD", "0.6"))
    HIGH_VOLATILITY_THRESHOLD = float(os.getenv("HIGH_VOLATILITY_THRESHOLD", "0.05"))
    LIQUIDATION_CASCADE_THRESHOLD = float(os.getenv("LIQUIDATION_CASCADE_THRESHOLD", "0.5"))
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "60.0"))
    
    # News monitoring
    ENHANCED_NEWS_ENABLED = os.getenv("ENHANCED_NEWS_ENABLED", "true").lower() == "true"
    NEWS_CACHE_MINUTES = int(os.getenv("NEWS_CACHE_MINUTES", "30"))
    MAX_NEWS_ARTICLES = int(os.getenv("MAX_NEWS_ARTICLES", "20"))
    RISK_OFF_WEIGHT = float(os.getenv("RISK_OFF_WEIGHT", "2.0"))
    MACRO_NEWS_WEIGHT = float(os.getenv("MACRO_NEWS_WEIGHT", "2.0"))
    
    # Correlation monitoring
    ENABLE_CORRELATION_MONITORING = os.getenv("ENABLE_CORRELATION_MONITORING", "true").lower() == "true"
    CORRELATION_LOOKBACK_DAYS = int(os.getenv("CORRELATION_LOOKBACK_DAYS", "30"))
    HIGH_CORRELATION_THRESHOLD = float(os.getenv("HIGH_CORRELATION_THRESHOLD", "0.7"))
    CORRELATION_CACHE_MINUTES = int(os.getenv("CORRELATION_CACHE_MINUTES", "15"))
    
    # Position sizing
    BASE_POSITION_PCT = float(os.getenv("BASE_POSITION_PCT", "0.1"))
    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.15"))
    MIN_POSITION_PCT = float(os.getenv("MIN_POSITION_PCT", "0.02"))
    RISK_REDUCTION_FACTOR = float(os.getenv("RISK_REDUCTION_FACTOR", "0.5"))
    
    # LLM configuration
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:4b")
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    FALLBACK_TO_SIMPLE_LOGIC = os.getenv("FALLBACK_TO_SIMPLE_LOGIC", "true").lower() == "true"
    
    # Market data
    YAHOO_FINANCE_ENABLED = True
    BITCOIN_PAIR = 'BTC-USD'
    SPY_TICKER = 'SPY'
    DXY_TICKER = 'DXY=X'
    GOLD_TICKER = 'GC=F'
    
    # File paths
    PRICE_HISTORY_FILE = os.getenv("PRICE_HISTORY_FILE", "./price_history.json")
    BOT_LOGS_FILE = os.getenv("BOT_LOGS_FILE", "./bot_logs.csv")
    ORDER_HISTORY_FILE = os.getenv("ORDER_HISTORY_FILE", "./order_history.json")
    PERFORMANCE_FILE = os.getenv("PERFORMANCE_FILE", "./performance_history.json")
    DB_FILE = os.getenv("DB_FILE", "./trading_bot.db")
    LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Metrics server
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8080"))
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    METRICS_AUTH_ENABLED = os.getenv("METRICS_AUTH_ENABLED", "false").lower() == "true"
    METRICS_USER = os.getenv("METRICS_USER", "admin")
    METRICS_PASSWORD = os.getenv("METRICS_PASSWORD", "changeme")
    
    # Alerting
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")
    ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "true").lower() == "true"
    
    # Exchange addresses for on-chain analysis
    EXCHANGE_ADDRESSES = {
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa": "Coinbase",
        "3EktnHQD7RiAE6uzMj2ZifT9YgRrkSgzQX": "Binance1",
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo": "Binance2",
        "3FrSzikNqBgikWgTHixywhXcx57q6H6rHC": "Binance3",
        "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r": "Bitfinex",
        "1AnwDVbwsLBVwRfqN2x9Eo4YEJSPXo2cwG": "Kraken"
    }
    
    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate all configuration values.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required API credentials
        if not cls.API_KEY or not cls.API_SECRET:
            errors.append("API_KEY and API_SECRET must be set")
        
        # Validate allocations sum to 1.0
        alloc_sum = sum(cls.ALLOCATIONS.values())
        if not (0.99 <= alloc_sum <= 1.01):  # Allow small float error
            errors.append(f"ALLOCATIONS must sum to 1.0, got {alloc_sum:.4f}")
        
        # Validate individual allocations are between 0 and 1
        for name, value in cls.ALLOCATIONS.items():
            if not (0 <= value <= 1):
                errors.append(f"{name} allocation must be between 0 and 1, got {value}")
        
        # Validate percentage ranges (0-1)
        percentage_configs = [
            ("STOP_LOSS_PERCENT", cls.STOP_LOSS_PERCENT),
            ("TAKE_PROFIT_PERCENT", cls.TAKE_PROFIT_PERCENT),
            ("BASE_STOP_LOSS_PCT", cls.BASE_STOP_LOSS_PCT),
            ("MAX_RISK_OFF_THRESHOLD", cls.MAX_RISK_OFF_THRESHOLD),
            ("HIGH_VOLATILITY_THRESHOLD", cls.HIGH_VOLATILITY_THRESHOLD),
            ("LIQUIDATION_CASCADE_THRESHOLD", cls.LIQUIDATION_CASCADE_THRESHOLD),
            ("MAX_CASH_ALLOCATION", cls.MAX_CASH_ALLOCATION),
            ("MAX_SELL_ALLOCATION", cls.MAX_SELL_ALLOCATION),
            ("MIN_PROFIT_MARGIN", cls.MIN_PROFIT_MARGIN),
            ("BASE_POSITION_PCT", cls.BASE_POSITION_PCT),
            ("MAX_POSITION_PCT", cls.MAX_POSITION_PCT),
            ("MIN_POSITION_PCT", cls.MIN_POSITION_PCT),
            ("RISK_REDUCTION_FACTOR", cls.RISK_REDUCTION_FACTOR),
            ("HIGH_CORRELATION_THRESHOLD", cls.HIGH_CORRELATION_THRESHOLD),
        ]
        
        for name, value in percentage_configs:
            if not (0 <= value <= 1):
                errors.append(f"{name} must be between 0 and 1, got {value}")
        
        # Validate positive values
        positive_configs = [
            ("MIN_TRADE_VOLUME", cls.MIN_TRADE_VOLUME),
            ("MIN_EUR_FOR_TRADE", cls.MIN_EUR_FOR_TRADE),
            ("MIN_BTC", cls.MIN_BTC),
            ("GLOBAL_TRADE_COOLDOWN", cls.GLOBAL_TRADE_COOLDOWN),
            ("SLEEP_DURATION", cls.SLEEP_DURATION),
        ]
        
        for name, value in positive_configs:
            if value <= 0:
                errors.append(f"{name} must be positive, got {value}")
        
        # Validate RPC settings if on-chain analysis is needed
        if not all([cls.RPC_USER, cls.RPC_PASSWORD]):
            errors.append("RPC_USER and RPC_PASSWORD must be set for on-chain analysis")
        
        # Validate position sizing makes sense
        if cls.MIN_POSITION_PCT > cls.BASE_POSITION_PCT:
            errors.append(f"MIN_POSITION_PCT ({cls.MIN_POSITION_PCT}) cannot be greater than BASE_POSITION_PCT ({cls.BASE_POSITION_PCT})")
        
        if cls.BASE_POSITION_PCT > cls.MAX_POSITION_PCT:
            errors.append(f"BASE_POSITION_PCT ({cls.BASE_POSITION_PCT}) cannot be greater than MAX_POSITION_PCT ({cls.MAX_POSITION_PCT})")
        
        # Validate thresholds
        if cls.MIN_CONFIDENCE_THRESHOLD < 0 or cls.MIN_CONFIDENCE_THRESHOLD > 100:
            errors.append(f"MIN_CONFIDENCE_THRESHOLD must be between 0 and 100, got {cls.MIN_CONFIDENCE_THRESHOLD}")
        
        return errors
    
    @classmethod
    def validate_or_exit(cls):
        """Validate configuration and exit if invalid"""
        errors = cls.validate()
        
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("✅ Configuration validation passed")


# Validate on import
try:
    Config.validate_or_exit()
except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)


# Export commonly used values
API_KEY = Config.API_KEY
API_SECRET = Config.API_SECRET
API_DOMAIN = Config.API_DOMAIN
ALLOCATIONS = Config.ALLOCATIONS
TOTAL_BTC = Config.TOTAL_BTC
MIN_TRADE_VOLUME = Config.MIN_TRADE_VOLUME
GLOBAL_TRADE_COOLDOWN = Config.GLOBAL_TRADE_COOLDOWN
SLEEP_DURATION = Config.SLEEP_DURATION
RPC_USER = Config.RPC_USER
RPC_PASSWORD = Config.RPC_PASSWORD
RPC_HOST = Config.RPC_HOST
RPC_PORT = Config.RPC_PORT
EXCHANGE_ADDRESSES = Config.EXCHANGE_ADDRESSES
PRICE_HISTORY_FILE = Config.PRICE_HISTORY_FILE
BOT_LOGS_FILE = Config.BOT_LOGS_FILE
MIN_BTC = Config.MIN_BTC
BALANCE_CACHE_DURATION = Config.BALANCE_CACHE_DURATION
ONCHAIN_CACHE_DURATION = Config.ONCHAIN_CACHE_DURATION
MIN_EUR_FOR_TRADE = Config.MIN_EUR_FOR_TRADE
MAX_CASH_ALLOCATION = Config.MAX_CASH_ALLOCATION
MAX_SELL_ALLOCATION = Config.MAX_SELL_ALLOCATION
MIN_PROFIT_MARGIN = Config.MIN_PROFIT_MARGIN