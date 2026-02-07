"""
PositionManager - Portfolio State and Position Tracking

This module provides a single source of truth for current portfolio state.
Tracks BTC and EUR balances, position values, and calculates portfolio metrics.

PUBLIC INTERFACE:
    get_position() -> Position
    get_available_eur() -> float
    get_btc_amount() -> float
    get_portfolio_value() -> float
    update_balance() -> None

PRIVATE IMPLEMENTATION:
    Balance caching
    Fee calculations
    Profit/loss tracking
    Portfolio composition
"""

from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime
from logger_config import logger


@dataclass(frozen=True)
class Position:
    """Current portfolio position snapshot."""
    btc_amount: float
    eur_balance: float
    current_price: float
    btc_value: float  # BTC amount * current price
    total_value: float  # BTC value + EUR balance
    avg_buy_price: float
    unrealized_pnl: float  # Current - Cost basis
    unrealized_pnl_pct: float  # PnL %
    position_concentration: float  # % of portfolio in BTC


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_invested: float
    total_value: float
    unrealized_gain: float
    unrealized_gain_pct: float
    fees_paid: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int


class PositionManager:
    """
    Manages current portfolio position and calculates portfolio metrics.

    Provides consistent interface for querying portfolio state.
    Hides complexity of balance tracking, fee calculations, etc.
    """

    def __init__(
        self,
        initial_btc: float = 0.0,
        initial_eur: float = 0.0,
        initial_price: float = 50000.0,
    ):
        """
        Initialize position manager.

        Args:
            initial_btc: Initial BTC balance
            initial_eur: Initial EUR balance
            initial_price: Initial BTC price for calculations
        """
        self._btc_balance = initial_btc
        self._eur_balance = initial_eur
        self._current_price = initial_price
        self._avg_buy_price = initial_price if initial_btc > 0 else 0.0

        # History tracking
        self._buy_price_history: list = []
        self._portfolio_value_history: list = []
        self._buy_trades_count = 0  # Count only buy trades
        self._sell_trades_count = 0  # Count only sell trades
        self._winning_trades = 0
        self._fees_paid = 0.0

        # If started with BTC, record it
        if initial_btc > 0:
            self._buy_price_history.extend([initial_price] * int(initial_btc * 1e8))

    def update_balance(self, btc_amount: float, eur_amount: float, current_price: float):
        """
        Update portfolio balances after a trade.

        Args:
            btc_amount: New BTC balance
            eur_amount: New EUR balance
            current_price: Current BTC price
        """
        self._btc_balance = btc_amount
        self._eur_balance = eur_amount
        self._current_price = current_price

        # Update history
        self._portfolio_value_history.append({
            'timestamp': datetime.now(),
            'value': self.get_portfolio_value(),
            'price': current_price
        })

        logger.debug(
            f"Position updated: {btc_amount:.8f} BTC, €{eur_amount:.2f} EUR @ €{current_price:.0f}"
        )

    def record_trade(self, btc_amount: float, price: float, is_buy: bool, fee_eur: float = 0.0):
        """
        Record a trade for tracking purposes.

        Args:
            btc_amount: BTC amount traded
            price: Price per BTC
            is_buy: True if buy, False if sell
            fee_eur: Trading fee in EUR
        """
        self._fees_paid += fee_eur

        if is_buy:
            cost = btc_amount * price + fee_eur
            self._eur_balance -= cost  # Subtract from EUR
            total_cost = (self._avg_buy_price * self._btc_balance) + (btc_amount * price)
            self._btc_balance += btc_amount
            if self._btc_balance > 0:
                self._avg_buy_price = total_cost / self._btc_balance
            
        else:
            proceeds = btc_amount * price - fee_eur
            self._eur_balance += proceeds  # Add to EUR
            self._sell_trades_count += 1
            # Selling - check if profitable
            if self._avg_buy_price > 0:
                pnl = (price - self._avg_buy_price) * btc_amount
                if pnl > 0:
                    self._winning_trades += 1
            self._btc_balance -= btc_amount

        logger.debug(f"Trade recorded: {btc_amount:.8f} BTC @ €{price:.0f} ({'BUY' if is_buy else 'SELL'})")

    def get_position(self) -> Position:
        """
        Get current portfolio position snapshot.

        Returns comprehensive position data with all metrics.
        """
        btc_value = self._btc_balance * self._current_price
        total_value = btc_value + self._eur_balance

        # Calculate unrealized P&L
        if self._avg_buy_price > 0 and self._btc_balance > 0:
            cost_basis = self._avg_buy_price * self._btc_balance
            unrealized_pnl = btc_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
        else:
            unrealized_pnl = 0.0
            unrealized_pnl_pct = 0.0

        # Position concentration
        position_concentration = (btc_value / total_value * 100) if total_value > 0 else 0

        return Position(
            btc_amount=self._btc_balance,
            eur_balance=self._eur_balance,
            current_price=self._current_price,
            btc_value=btc_value,
            total_value=total_value,
            avg_buy_price=self._avg_buy_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            position_concentration=position_concentration,
        )

    def get_available_eur(self) -> float:
        """Get available EUR for trading."""
        return self._eur_balance

    def get_btc_amount(self) -> float:
        """Get current BTC holdings."""
        return self._btc_balance

    def get_portfolio_value(self) -> float:
        """Get total portfolio value in EUR."""
        return self._btc_balance * self._current_price + self._eur_balance

    def get_avg_buy_price(self) -> float:
        """Get average buy price."""
        return self._avg_buy_price

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.

        Returns performance statistics for analysis.
        """
        # Calculate returns
        cost_basis = self._avg_buy_price * self._btc_balance
        btc_value = self._btc_balance * self._current_price
        unrealized_gain = btc_value - cost_basis
        unrealized_gain_pct = (unrealized_gain / cost_basis * 100) if cost_basis > 0 else 0

        # Win rate (only based on sell trades, since each sell is either a win or loss)
        win_rate = (
            self._winning_trades / self._sell_trades_count * 100
            if self._sell_trades_count > 0
            else 0
        )

        # Calculate drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Sharpe ratio (simplified - daily data)
        sharpe_ratio = self._calculate_sharpe_ratio()

        return PortfolioMetrics(
            total_invested=cost_basis,
            total_value=self.get_portfolio_value(),
            unrealized_gain=unrealized_gain,
            unrealized_gain_pct=unrealized_gain_pct,
            fees_paid=self._fees_paid,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=self._buy_trades_count + self._sell_trades_count,
            winning_trades=self._winning_trades,
            losing_trades=self._sell_trades_count - self._winning_trades,
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history."""
        if not self._portfolio_value_history:
            return 0.0

        peak = self._portfolio_value_history[0]['value']
        max_dd = 0.0

        for entry in self._portfolio_value_history:
            if entry['value'] > peak:
                peak = entry['value']
            drawdown = (peak - entry['value']) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (simplified).

        Assumes daily returns from portfolio value history.
        """
        if len(self._portfolio_value_history) < 2:
            return 0.0

        returns = []
        prev_value = self._portfolio_value_history[0]['value']

        for entry in self._portfolio_value_history[1:]:
            current_value = entry['value']
            ret = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            returns.append(ret)
            prev_value = current_value

        if not returns:
            return 0.0

        # Calculate mean and std dev
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return 0.0

        # Annualize (assuming 365 trading days)
        annual_return = mean_return * 365
        annual_std = std_dev * (365 ** 0.5)

        sharpe = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0

        return sharpe
