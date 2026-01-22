"""
Custom exceptions for the trading bot.
Provides specific error types for better error handling and debugging.
"""


class TradingBotError(Exception):
    """Base exception for all trading bot errors"""
    pass


class APIError(TradingBotError):
    """Raised when API communication fails"""
    pass


class InsufficientBalanceError(TradingBotError):
    """Raised when attempting to trade with insufficient balance"""
    pass


class OrderError(TradingBotError):
    """Raised when order operations fail"""
    pass


class ConfigurationError(TradingBotError):
    """Raised when configuration is invalid"""
    pass


class DataError(TradingBotError):
    """Raised when data operations fail"""
    pass


class NetworkError(APIError):
    """Raised for network-related errors"""
    pass


class AuthenticationError(APIError):
    """Raised for authentication failures"""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""
    pass


class ValidationError(TradingBotError):
    """Raised when data validation fails"""
    pass