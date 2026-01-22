"""
Enhanced logging configuration with structured logging support.
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Formatter that adds structured data to log records"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Start with the basic formatted message
        message = super().format(record)
        
        # Add structured data if available
        if hasattr(record, 'extra_data') and record.extra_data:
            structured = json.dumps(record.extra_data, default=str)
            message = f"{message} | data={structured}"
        
        return message


class TradingBotLogger:
    """Enhanced logger for the trading bot"""
    
    def __init__(
        self,
        name: str = "trading_bot",
        log_file: str = "trading_bot.log",
        log_level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.propagate = False
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = StructuredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Rotating file handler
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = StructuredFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def _log_with_data(self, level: int, msg: str, extra_data: Dict[str, Any] = None, exc_info=False):
        """Internal method to log with structured data"""
        extra = {'extra_data': extra_data} if extra_data else {}
        self.logger.log(level, msg, extra=extra, exc_info=exc_info)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with optional structured data"""
        self._log_with_data(logging.DEBUG, msg, kwargs if kwargs else None)
    
    def info(self, msg: str, **kwargs):
        """Log info message with optional structured data"""
        self._log_with_data(logging.INFO, msg, kwargs if kwargs else None)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message with optional structured data"""
        self._log_with_data(logging.WARNING, msg, kwargs if kwargs else None)
    
    def error(self, msg: str, exc_info=False, **kwargs):
        """Log error message with optional structured data and exception info"""
        self._log_with_data(logging.ERROR, msg, kwargs if kwargs else None, exc_info=exc_info)
    
    def critical(self, msg: str, exc_info=False, **kwargs):
        """Log critical message with optional structured data and exception info"""
        self._log_with_data(logging.CRITICAL, msg, kwargs if kwargs else None, exc_info=exc_info)
    
    def set_level(self, level: str):
        """Change logging level at runtime"""
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# Create default logger instance
_default_logger = None


def get_logger(
    name: str = "trading_bot",
    log_file: str = None,
    log_level: str = None
) -> TradingBotLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level
    
    Returns:
        TradingBotLogger instance
    """
    global _default_logger
    
    # Use default logger if no specific config provided
    if name == "trading_bot" and log_file is None and log_level is None:
        if _default_logger is None:
            # Import here to avoid circular dependency
            try:
                from config import Config
                _default_logger = TradingBotLogger(
                    name=name,
                    log_file=Config.LOG_FILE,
                    log_level=Config.LOG_LEVEL
                )
            except ImportError:
                _default_logger = TradingBotLogger(name=name)
        return _default_logger
    
    # Create new logger with specific config
    return TradingBotLogger(
        name=name,
        log_file=log_file or "trading_bot.log",
        log_level=log_level or "INFO"
    )


# Default logger for backward compatibility
logger = get_logger()


# Convenience functions that use the default logger
def debug(msg: str, **kwargs):
    logger.debug(msg, **kwargs)


def info(msg: str, **kwargs):
    logger.info(msg, **kwargs)


def warning(msg: str, **kwargs):
    logger.warning(msg, **kwargs)


def error(msg: str, exc_info=False, **kwargs):
    logger.error(msg, exc_info=exc_info, **kwargs)


def critical(msg: str, exc_info=False, **kwargs):
    logger.critical(msg, exc_info=exc_info, **kwargs)