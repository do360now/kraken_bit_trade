"""
Secure Configuration Manager - Replaces config.py
Implements proper security practices for production trading
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security-focused configuration"""
    max_daily_trades: int = 12
    max_position_pct: float = 0.15  # Reduced from unsafe levels
    emergency_stop_threshold: float = 0.10  # 10% daily loss triggers stop
    api_timeout_seconds: int = 10
    max_retries: int = 3
    rate_limit_per_minute: int = 60

@dataclass 
class TradingConfig:
    """Safe trading parameters"""
    base_position_size_pct: float = 0.08  # Conservative 8%
    stop_loss_pct: float = 0.025  # 2.5% stop loss
    take_profit_pct: float = 0.08  # 8% take profit
    min_confidence_threshold: float = 0.35
    trade_cooldown_minutes: int = 15
    max_position_pct: float = 0.20  # Max 20% of portfolio in one position
    
@dataclass
class RiskConfig:
    """Comprehensive risk management"""
    max_drawdown_pct: float = 0.15  # 15% max portfolio drawdown
    position_concentration_limit: float = 0.25  # Max 25% in single position
    volatility_circuit_breaker: float = 0.10  # Stop trading if vol > 10%
    correlation_risk_threshold: float = 0.80  # Reduce size if high correlation


class SecureConfigManager:
    """Secure configuration management with validation"""
    
    def __init__(self, env_file: str = ".env"):
        load_dotenv(env_file)
        self.security = SecurityConfig()
        self.trading = TradingConfig() 
        self.risk = RiskConfig()
        self._validate_environment()
        
    def _validate_environment(self) -> None:
        """Validate critical environment variables exist"""
        required_vars = ["KRAKEN_API_KEY", "KRAKEN_API_SECRET"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
            
        # Validate API key format (basic check)
        api_key = os.getenv("KRAKEN_API_KEY", "")
        if len(api_key) < 20:  # Kraken keys are typically longer
            logger.warning("API key appears to be invalid length")
            
    def get_api_credentials(self) -> Dict[str, str]:
        """Safely retrieve API credentials"""
        return {
            "api_key": os.getenv("KRAKEN_API_KEY"),
            "api_secret": os.getenv("KRAKEN_API_SECRET"),
            "api_domain": os.getenv("KRAKEN_API_DOMAIN", "https://api.kraken.com")
        }
    
    def sanitize_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data before logging"""
        sensitive_keys = {
            'api_key', 'api_secret', 'signature', 'nonce', 
            'balance', 'volume', 'price', 'password', 'secret'
        }
        
        def _clean_dict(obj):
            if isinstance(obj, dict):
                return {
                    k: "***REDACTED***" if any(sens in k.lower() for sens in sensitive_keys) 
                    else _clean_dict(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [_clean_dict(item) for item in obj]
            else:
                return obj
                
        return _clean_dict(data)
    
    def get_file_paths(self) -> Dict[str, str]:
        """Get standardized file paths"""
        base_dir = os.getenv("BOT_DATA_DIR", "./")
        return {
            "price_history": os.path.join(base_dir, "price_history.json"),
            "bot_logs": os.path.join(base_dir, "bot_logs.csv"), 
            "bot_state": os.path.join(base_dir, "bot_state.json"),
            "performance": os.path.join(base_dir, "performance.json"),
            "ml_model": os.path.join(base_dir, "ml_model.pkl")
        }
    
    def validate_trade_parameters(self, volume: float, price: float, 
                                 portfolio_value: float) -> bool:
        """Validate trade parameters against risk limits"""
        trade_value = volume * price
        position_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
        
        # Check position size limits
        if position_pct > self.trading.max_position_pct:
            logger.warning(f"Trade size {position_pct:.1%} exceeds limit {self.trading.max_position_pct:.1%}")
            return False
            
        # Check minimum trade size
        min_trade_eur = float(os.getenv("MIN_TRADE_EUR", "15.0"))
        if trade_value < min_trade_eur:
            logger.warning(f"Trade value €{trade_value:.2f} below minimum €{min_trade_eur}")
            return False
            
        return True
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange-specific configuration"""
        return {
            "name": "kraken",
            "pair": "XXBTZEUR", 
            "min_order_size": 0.0001,  # BTC
            "maker_fee": 0.0026,  # 0.26%
            "taker_fee": 0.0026,  # 0.26%
            "api_rate_limit": 60,  # calls per minute
            "supported_order_types": ["market", "limit", "stop_loss"]
        }
        
    def export_sanitized_config(self) -> Dict[str, Any]:
        """Export configuration for logging/debugging without sensitive data"""
        return {
            "security": {
                "max_daily_trades": self.security.max_daily_trades,
                "max_position_pct": self.security.max_position_pct,
                "api_timeout": self.security.api_timeout_seconds
            },
            "trading": {
                "base_position_size_pct": self.trading.base_position_size_pct,
                "stop_loss_pct": self.trading.stop_loss_pct,
                "take_profit_pct": self.trading.take_profit_pct
            },
            "risk": {
                "max_drawdown_pct": self.risk.max_drawdown_pct,
                "volatility_circuit_breaker": self.risk.volatility_circuit_breaker
            },
            "exchange": self.get_exchange_config()
        }


# Global instance
config_manager = SecureConfigManager()

# Backwards compatibility exports
KRAKEN_CONFIG = config_manager.get_api_credentials()
TRADING_PARAMS = config_manager.trading
RISK_PARAMS = config_manager.risk
FILE_PATHS = config_manager.get_file_paths()