"""
Trade - Represents a single trade execution result

This is an immutable dataclass that encapsulates all trade information.
Returned by TradeExecutor.buy() and TradeExecutor.sell() methods.
"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional


class TradeStatus(Enum):
    """Status of a trade."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TradeType(Enum):
    """Type of trade."""
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Trade:
    """
    Immutable representation of a trade execution.
    
    Either fully encapsulates the result of a buy/sell attempt.
    Used as return value from TradeExecutor.buy() and TradeExecutor.sell().
    """
    trade_id: str  # Unique order ID from exchange
    trade_type: TradeType
    status: TradeStatus
    btc_amount: float  # Amount requested
    btc_filled: float  # Amount actually filled
    price_limit: float  # Limit price set
    price_filled: float  # Average fill price
    total_cost: float  # Total cost in EUR (filled * price)
    fee_eur: float  # Trading fee in EUR
    created_at: datetime
    filled_at: Optional[datetime]
    reason: str  # Description of status
    
    def __post_init__(self):
        """Validate trade data."""
        if self.btc_amount < 0:
            raise ValueError("BTC amount cannot be negative")
        if self.btc_filled < 0 or self.btc_filled > self.btc_amount:
            raise ValueError(f"Filled amount {self.btc_filled} invalid for {self.btc_amount}")
        if self.price_limit < 0 or self.price_filled < 0:
            raise ValueError("Prices cannot be negative")
        if self.fee_eur < 0:
            raise ValueError("Fee cannot be negative")
    
    @property
    def is_success(self) -> bool:
        """Check if trade was successful (at least partially filled)."""
        return self.status in [TradeStatus.FILLED, TradeStatus.PARTIALLY_FILLED]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.btc_amount <= 0:
            return 0.0
        return (self.btc_filled / self.btc_amount) * 100
    
    @property
    def net_proceeds(self) -> float:
        """Calculate net proceeds after fees."""
        return self.total_cost - self.fee_eur if self.trade_type == TradeType.BUY else self.total_cost - self.fee_eur
