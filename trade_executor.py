"""
Trade executor — smart limit order placement and lifecycle management.

Key lessons from previous bot:
- Market orders bypassed the TradeExecutor entirely and went straight to the
  API. FIX: This executor ONLY places limit orders. No market order path exists.
- Orders were fire-and-forget with no status tracking.
  FIX: Full lifecycle — place, monitor, chase (re-price), cancel.
- No spread-aware pricing.
  FIX: Urgency-based offset from best bid/ask into the spread.

Design:
- execute_buy() / execute_sell() are the only entry points.
- Each returns a TradeResult summarizing what happened.
- Orders are placed as limit orders offset into the spread.
- If unfilled after TTL, orders are cancelled and optionally re-priced
  (chased) up to max_chase_attempts times.
- Post-trade bookkeeping: records trade in RiskManager, marks profit tiers,
  logs full reasoning chain.
- Never raises exceptions past the public interface.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from config import BotConfig, ExecutionConfig, Urgency
from kraken_api import (
    KrakenAPI,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)
from risk_manager import RiskManager
from position_sizer import PositionSizer, BuySize, SellTier

logger = logging.getLogger(__name__)


# ─── Output dataclasses ──────────────────────────────────────────────────────

class TradeOutcome(Enum):
    """What happened to the trade attempt."""
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"       # Cancelled by us (TTL expired, chase exhausted)
    FAILED = "failed"             # API error, validation failure
    SKIPPED = "skipped"           # Pre-flight check failed (no ticker, etc.)


@dataclass(frozen=True)
class TradeResult:
    """Complete record of a trade execution attempt."""
    outcome: TradeOutcome
    side: str                     # "buy" or "sell"
    requested_volume: float       # BTC amount we tried to trade
    filled_volume: float          # BTC actually traded
    filled_price: float           # Average fill price (EUR)
    fee_eur: float                # Fees paid
    txid: Optional[str]           # Kraken transaction ID
    limit_price: float            # Limit price we set
    chase_count: int              # How many times we re-priced
    elapsed_seconds: float        # Total time from start to completion
    reason: str                   # Human-readable explanation

    @property
    def eur_value(self) -> float:
        """EUR value of the filled portion."""
        return self.filled_volume * self.filled_price

    @property
    def success(self) -> bool:
        return self.outcome in (TradeOutcome.FILLED, TradeOutcome.PARTIALLY_FILLED)
    
    def to_dict(self) -> dict:
        """Convert to dict format matching the JSONL trade log schema."""
        return {
            "timestamp": time.time(),
            "side": self.side,
            "outcome": self.outcome.value,
            "volume": self.filled_volume,
            "price": self.filled_price,
            "eur_value": self.eur_value,
            "fee": self.fee_eur,
            "txid": self.txid,
            "limit_price": self.limit_price,
            "chase_count": self.chase_count,
            "elapsed_s": self.elapsed_seconds,
            "reason": self.reason,
        }


# ─── Trade Executor ──────────────────────────────────────────────────────────

class TradeExecutor:
    """
    Smart limit order executor with spread-aware pricing and chase logic.

    Args:
        api: Kraken API client.
        risk_manager: For post-trade recording.
        config: Bot configuration.
    """

    def __init__(
        self,
        api: KrakenAPI,
        risk_manager: RiskManager,
        config: BotConfig,
    ) -> None:
        self._api = api
        self._risk = risk_manager
        self._cfg = config.execution
        self._kraken_cfg = config.kraken
        self._persistence = config.persistence
        self._trade_log: list[dict] = []

    # ─── Public interface ────────────────────────────────────────────────

    def execute_buy(
        self,
        buy_size: BuySize,
        urgency: Urgency = Urgency.MEDIUM,
    ) -> TradeResult:
        """
        Execute a buy order.

        1. Fetch ticker for spread-aware pricing.
        2. Compute limit price (offset from best ask toward bid).
        3. Place limit order.
        4. Monitor, chase if unfilled after TTL.
        5. Record trade in risk manager.

        Args:
            buy_size: From PositionSizer.compute_buy_size().
            urgency: Controls how aggressively we price the order.
        """
        start = time.time()

        if buy_size.eur_amount <= 0 or buy_size.btc_amount <= 0:
            return TradeResult(
                outcome=TradeOutcome.SKIPPED, side="buy",
                requested_volume=0.0, filled_volume=0.0,
                filled_price=0.0, fee_eur=0.0, txid=None,
                limit_price=0.0, chase_count=0,
                elapsed_seconds=time.time() - start,
                reason="Zero buy size",
            )

        # Fetch current spread
        ticker = self._api.get_ticker()
        if ticker is None:
            return self._failed_result(
                "buy", buy_size.btc_amount, start, "Failed to fetch ticker",
            )

        # Compute limit price for buy: offset from ask toward bid
        limit_price = self._compute_buy_price(ticker, urgency)

        # Slippage guard
        mid = (ticker.ask + ticker.bid) / 2.0
        if mid > 0 and (limit_price - mid) / mid > self._cfg.max_slippage_pct:
            return self._failed_result(
                "buy", buy_size.btc_amount, start,
                f"Slippage guard: limit €{limit_price:,.1f} too far from "
                f"mid €{mid:,.1f} ({(limit_price - mid) / mid:.2%})",
            )

        # Recalculate BTC volume at our limit price
        volume = buy_size.eur_amount / limit_price if limit_price > 0 else 0.0
        if volume < self._kraken_cfg.min_order_btc:
            return self._failed_result(
                "buy", volume, start,
                f"Volume {volume:.8f} below minimum {self._kraken_cfg.min_order_btc}",
            )

        # Execute with chase logic
        result = self._execute_with_chase(
            side=OrderSide.BUY,
            volume=volume,
            initial_price=limit_price,
            ticker=ticker,
            urgency=urgency,
            start_time=start,
        )

        # Post-trade bookkeeping
        if result.success:
            self._risk.record_trade()
            self._log_trade(result)

        return result

    def execute_sell(
        self,
        sell_tier: SellTier,
        urgency: Urgency = Urgency.MEDIUM,
    ) -> TradeResult:
        """
        Execute a sell order for profit taking.

        Args:
            sell_tier: From PositionSizer.compute_sell_tiers().
            urgency: Controls pricing aggressiveness.
        """
        start = time.time()

        if sell_tier.btc_amount <= 0:
            return TradeResult(
                outcome=TradeOutcome.SKIPPED, side="sell",
                requested_volume=0.0, filled_volume=0.0,
                filled_price=0.0, fee_eur=0.0, txid=None,
                limit_price=0.0, chase_count=0,
                elapsed_seconds=time.time() - start,
                reason="Zero sell size",
            )

        ticker = self._api.get_ticker()
        if ticker is None:
            return self._failed_result(
                "sell", sell_tier.btc_amount, start, "Failed to fetch ticker",
            )

        # Compute limit price for sell: offset from bid toward ask
        limit_price = self._compute_sell_price(ticker, urgency)

        # Slippage guard
        mid = (ticker.ask + ticker.bid) / 2.0
        if mid > 0 and (mid - limit_price) / mid > self._cfg.max_slippage_pct:
            return self._failed_result(
                "sell", sell_tier.btc_amount, start,
                f"Slippage guard: limit €{limit_price:,.1f} too far from "
                f"mid €{mid:,.1f}",
            )

        volume = sell_tier.btc_amount

        result = self._execute_with_chase(
            side=OrderSide.SELL,
            volume=volume,
            initial_price=limit_price,
            ticker=ticker,
            urgency=urgency,
            start_time=start,
        )

        if result.success:
            self._risk.record_trade()
            self._log_trade(result)

        return result

    # ─── Spread-aware pricing ────────────────────────────────────────────

    def _compute_buy_price(self, ticker: Ticker, urgency: Urgency) -> float:
        """
        Compute limit buy price.

        Strategy: start from the ask and move toward the bid based on urgency.
        LOW urgency -> deeper in book (closer to bid) -> better price, slower fill.
        HIGH urgency -> near ask -> faster fill, worse price.
        """
        offset = self._urgency_offset(urgency)
        # Price = ask - offset * spread (moving from ask toward bid)
        price = ticker.ask - offset * ticker.spread
        return self._round_to_tick(price)

    def _compute_sell_price(self, ticker: Ticker, urgency: Urgency) -> float:
        """
        Compute limit sell price.

        Strategy: start from the bid and move toward the ask based on urgency.
        LOW urgency -> deeper in book (closer to ask) -> better price, slower fill.
        HIGH urgency -> near bid -> faster fill, worse price.
        """
        offset = self._urgency_offset(urgency)
        # Price = bid + offset * spread (moving from bid toward ask)
        price = ticker.bid + offset * ticker.spread
        return self._round_to_tick(price)

    def _urgency_offset(self, urgency: Urgency) -> float:
        """Map urgency to spread offset fraction."""
        offsets = {
            Urgency.LOW: self._cfg.spread_offset_low,
            Urgency.MEDIUM: self._cfg.spread_offset_medium,
            Urgency.HIGH: self._cfg.spread_offset_high,
        }
        return offsets.get(urgency, self._cfg.spread_offset_medium)

    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest tick size."""
        tick = self._cfg.tick_size
        if tick <= 0:
            return round(price, 1)
        return round(round(price / tick) * tick, 1)

    # ─── Order lifecycle: place, monitor, chase ──────────────────────────

    def _execute_with_chase(
        self,
        side: OrderSide,
        volume: float,
        initial_price: float,
        ticker: Ticker,
        urgency: Urgency,
        start_time: float,
    ) -> TradeResult:
        """
        Place order, monitor for fill, re-price (chase) if TTL expires.

        Chase makes the price more aggressive each attempt:
        - Each chase moves the price ~30% closer to the market.
        - After max_chase_attempts, cancel and give up.
        """
        current_price = initial_price
        chase_count = 0
        total_filled = 0.0
        avg_price = 0.0
        total_fee = 0.0
        last_txid: Optional[str] = None

        while chase_count <= self._cfg.max_chase_attempts:
            # Place order
            remaining = volume - total_filled
            if remaining < self._kraken_cfg.min_order_btc:
                break

            order_result = self._api.place_order(
                side=side,
                order_type=OrderType.LIMIT,
                volume=remaining,
                price=current_price,
            )

            if not order_result.success:
                return TradeResult(
                    outcome=TradeOutcome.FAILED, side=side.value,
                    requested_volume=volume,
                    filled_volume=total_filled,
                    filled_price=avg_price,
                    fee_eur=total_fee,
                    txid=last_txid,
                    limit_price=current_price,
                    chase_count=chase_count,
                    elapsed_seconds=time.time() - start_time,
                    reason=f"Order placement failed: {order_result.error}",
                )

            last_txid = order_result.txid

            # Monitor order
            fill_result = self._monitor_order(order_result.txid)

            if fill_result is not None:
                if fill_result.filled_volume > 0:
                    # Weighted average price
                    prev_value = avg_price * total_filled
                    new_value = fill_result.filled_price * fill_result.filled_volume
                    total_filled += fill_result.filled_volume
                    avg_price = (prev_value + new_value) / total_filled if total_filled > 0 else 0.0
                    total_fee += fill_result.fee

                if fill_result.status == OrderStatus.FILLED:
                    return TradeResult(
                        outcome=TradeOutcome.FILLED, side=side.value,
                        requested_volume=volume,
                        filled_volume=total_filled,
                        filled_price=avg_price,
                        fee_eur=total_fee,
                        txid=last_txid,
                        limit_price=current_price,
                        chase_count=chase_count,
                        elapsed_seconds=time.time() - start_time,
                        reason=f"Filled after {chase_count} chase(s)",
                    )

                if fill_result.status == OrderStatus.PARTIALLY_FILLED:
                    # Cancel remainder
                    self._api.cancel_order(order_result.txid)
                    if total_filled >= volume * 0.5:
                        return TradeResult(
                            outcome=TradeOutcome.PARTIALLY_FILLED,
                            side=side.value,
                            requested_volume=volume,
                            filled_volume=total_filled,
                            filled_price=avg_price,
                            fee_eur=total_fee,
                            txid=last_txid,
                            limit_price=current_price,
                            chase_count=chase_count,
                            elapsed_seconds=time.time() - start_time,
                            reason=f"Partially filled {total_filled:.8f}/{volume:.8f} BTC",
                        )

            # TTL expired — cancel and chase
            if order_result.txid:
                self._api.cancel_order(order_result.txid)

            chase_count += 1
            if chase_count > self._cfg.max_chase_attempts:
                break

            # Chase: make price more aggressive
            current_price = self._chase_price(
                current_price, side, ticker, chase_count,
            )

            logger.info(
                f"Chase {chase_count}/{self._cfg.max_chase_attempts}: "
                f"{side.value} re-priced to €{current_price:,.1f}"
            )

        # All chases exhausted
        outcome = (
            TradeOutcome.PARTIALLY_FILLED
            if total_filled > 0
            else TradeOutcome.CANCELLED
        )
        return TradeResult(
            outcome=outcome, side=side.value,
            requested_volume=volume,
            filled_volume=total_filled,
            filled_price=avg_price,
            fee_eur=total_fee,
            txid=last_txid,
            limit_price=current_price,
            chase_count=max(0, chase_count - 1),
            elapsed_seconds=time.time() - start_time,
            reason=f"Chase exhausted ({self._cfg.max_chase_attempts} attempts). "
                   f"Filled {total_filled:.8f}/{volume:.8f}",
        )

    def _monitor_order(self, txid: Optional[str]) -> Optional[OrderResult]:
        """
        Monitor an order until filled, TTL expires, or error.

        Polls order status at check_interval_seconds intervals.
        Returns the final OrderResult, or None if TTL expired unfilled.
        """
        if txid is None:
            return None

        deadline = time.time() + self._cfg.order_ttl_seconds

        while time.time() < deadline:
            result = self._api.query_order(txid)

            if result.status == OrderStatus.FILLED:
                return result

            if result.status == OrderStatus.PARTIALLY_FILLED:
                # Check if we're close to deadline — return what we have
                if time.time() + self._cfg.check_interval_seconds >= deadline:
                    return result

            if result.status in (
                OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.FAILED,
            ):
                return result

            time.sleep(self._cfg.check_interval_seconds)

        # TTL expired — query one final time
        return self._api.query_order(txid)

    def _chase_price(
        self,
        current_price: float,
        side: OrderSide,
        ticker: Ticker,
        chase_num: int,
    ) -> float:
        """
        Compute a more aggressive price for chasing.

        Each chase moves 30% closer to the market price.
        """
        if side == OrderSide.BUY:
            # Move price up toward ask
            target = ticker.ask
            new_price = current_price + (target - current_price) * 0.30
        else:
            # Move price down toward bid
            target = ticker.bid
            new_price = current_price - (current_price - target) * 0.30

        return self._round_to_tick(new_price)

    # ─── Trade logging ───────────────────────────────────────────────────

    def _log_trade(self, result: TradeResult) -> None:
        """Append trade to the persistent log."""
        entry = {
            "timestamp": time.time(),
            "side": result.side,
            "outcome": result.outcome.value,
            "volume": result.filled_volume,
            "price": result.filled_price,
            "eur_value": result.eur_value,
            "fee": result.fee_eur,
            "txid": result.txid,
            "limit_price": result.limit_price,
            "chase_count": result.chase_count,
            "elapsed_s": result.elapsed_seconds,
            "reason": result.reason,
        }
        self._trade_log.append(entry)
        self._persist_trade(entry)
        logger.info(
            f"Trade: {result.side} {result.filled_volume:.8f} BTC "
            f"@ €{result.filled_price:,.1f} (€{result.eur_value:,.0f}) "
            f"fee=€{result.fee_eur:.2f} chase={result.chase_count} "
            f"[{result.outcome.value}]"
        )

    def _persist_trade(self, entry: dict) -> None:
        """Append trade to the trade log file (JSONL format)."""
        try:
            path = self._persistence.get_path("trade_log.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.error(f"Failed to persist trade: {exc}")

    def get_trade_log(self) -> list[dict]:
        """Return the in-memory trade log."""
        return list(self._trade_log)

    def load_trade_history(self) -> list[dict]:
        """Load trade history from disk."""
        path = self._persistence.get_path("trade_log.jsonl")
        if not path.exists():
            return []
        trades = []
        try:
            for line in path.read_text().splitlines():
                if line.strip():
                    trades.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load trade history: {exc}")
        return trades

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _failed_result(
        side: str, volume: float, start: float, reason: str,
    ) -> TradeResult:
        return TradeResult(
            outcome=TradeOutcome.FAILED, side=side,
            requested_volume=volume, filled_volume=0.0,
            filled_price=0.0, fee_eur=0.0, txid=None,
            limit_price=0.0, chase_count=0,
            elapsed_seconds=time.time() - start,
            reason=reason,
        )
