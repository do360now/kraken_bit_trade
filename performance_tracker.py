"""
Performance tracker — measures what matters for an accumulation bot.

Key insight: the primary metric is NOT EUR profit. It's the BTC accumulation
rate — how much Bitcoin we're acquiring per unit time, and whether our
strategy is acquiring more BTC than a simple DCA baseline would.

Tracked metrics:
- BTC accumulated (total and per period)
- Accumulation efficiency vs DCA baseline
- Trade statistics (win rate, avg fill quality, chase frequency)
- Fee analysis (total fees, fee as % of volume)
- Capital efficiency (EUR deployed vs EUR available)
- Drawdown tracking (max drawdown, recovery time)
- Per-phase performance breakdown

Design:
- Ingests TradeResult objects after each trade.
- Periodically snapshots portfolio state.
- Persists all data for long-term analysis.
- Generates human-readable reports.
- Never raises exceptions past the public interface.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import BotConfig, CyclePhase, PersistenceConfig
from risk_manager import PortfolioState
from trade_executor import TradeResult, TradeOutcome

logger = logging.getLogger(__name__)


# ─── Metric dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TradeStats:
    """Aggregate trade statistics."""
    total_trades: int
    buy_count: int
    sell_count: int
    filled_count: int
    partial_count: int
    cancelled_count: int
    failed_count: int
    total_btc_bought: float
    total_btc_sold: float
    total_eur_spent: float
    total_eur_received: float
    total_fees_eur: float
    avg_buy_price: float
    avg_sell_price: float
    avg_chase_count: float
    avg_fill_time_seconds: float

    @property
    def net_btc_accumulated(self) -> float:
        return self.total_btc_bought - self.total_btc_sold

    @property
    def fill_rate(self) -> float:
        """Fraction of trades that filled (fully or partially)."""
        if self.total_trades == 0:
            return 0.0
        return (self.filled_count + self.partial_count) / self.total_trades

    @property
    def fee_pct(self) -> float:
        """Fees as percentage of total volume."""
        total_volume = self.total_eur_spent + self.total_eur_received
        if total_volume == 0:
            return 0.0
        return self.total_fees_eur / total_volume


@dataclass(frozen=True)
class AccumulationMetrics:
    """How well we're accumulating BTC vs baseline."""
    total_btc_accumulated: float
    dca_baseline_btc: float        # What simple DCA would have accumulated
    efficiency_ratio: float         # actual / baseline (>1.0 = outperforming)
    btc_per_day: float              # Average daily accumulation rate
    avg_cost_basis: float           # Weighted avg price paid per BTC
    current_price: float
    unrealized_pnl_pct: float      # (current - avg_cost) / avg_cost
    tracking_days: float            # How many days we've been tracking


@dataclass(frozen=True)
class PerformanceReport:
    """Complete performance snapshot."""
    trade_stats: TradeStats
    accumulation: AccumulationMetrics
    portfolio: PortfolioState
    max_drawdown_pct: float
    current_drawdown_pct: float
    phase_breakdown: dict[str, dict]  # phase -> {trades, btc_bought, ...}
    generated_at: str
    tracking_since: str


# ─── Performance Tracker ─────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Tracks and reports on trading performance.

    Ingests trade results and portfolio snapshots, computes metrics,
    and generates reports focused on BTC accumulation effectiveness.

    Args:
        config: Bot configuration.
    """

    def __init__(self, config: BotConfig) -> None:
        self._persistence = config.persistence

        # Trade history (in memory + persisted)
        self._trades: list[dict] = []

        # Portfolio snapshots for drawdown tracking
        self._snapshots: list[dict] = []

        # DCA baseline tracking
        self._dca_deposits: list[dict] = []  # {timestamp, eur_amount, btc_price}

        # State
        self._peak_value_eur: float = 0.0
        self._max_drawdown: float = 0.0
        self._tracking_start: Optional[float] = None

        self._load_state()

    # ─── Ingestion ───────────────────────────────────────────────────────

    def record_trade(self, result: TradeResult, phase: CyclePhase) -> None:
        """Record a completed trade."""
        if self._tracking_start is None:
            self._tracking_start = time.time()

        entry = {
            "timestamp": time.time(),
            "side": result.side,
            "outcome": result.outcome.value,
            "filled_volume": result.filled_volume,
            "filled_price": result.filled_price,
            "eur_value": result.eur_value,
            "fee_eur": result.fee_eur,
            "chase_count": result.chase_count,
            "elapsed_seconds": result.elapsed_seconds,
            "phase": phase.value,
        }
        self._trades.append(entry)
        self._save_state()

        logger.debug(
            f"Performance: recorded {result.side} "
            f"{result.filled_volume:.8f} BTC in {phase.value}"
        )

    def record_dca_baseline(self, eur_amount: float, btc_price: float) -> None:
        """
        Record what a simple DCA would have done at this moment.

        Called whenever we make a buy, recording the same EUR amount
        at the current price — this is what a simple "buy X EUR every
        time" strategy would have achieved.
        """
        if btc_price <= 0 or eur_amount <= 0:
            return
        self._dca_deposits.append({
            "timestamp": time.time(),
            "eur_amount": eur_amount,
            "btc_price": btc_price,
            "btc_amount": eur_amount / btc_price,
        })

    def snapshot_portfolio(self, portfolio: PortfolioState) -> None:
        """Record a portfolio snapshot for drawdown tracking."""
        if self._tracking_start is None:
            self._tracking_start = time.time()

        total = portfolio.total_value_eur
        self._snapshots.append({
            "timestamp": time.time(),
            "total_eur": total,
            "eur_balance": portfolio.eur_balance,
            "btc_balance": portfolio.btc_balance,
            "btc_price": portfolio.btc_price,
        })

        # Update peak and max drawdown
        if total > self._peak_value_eur:
            self._peak_value_eur = total

        if self._peak_value_eur > 0:
            drawdown = 1.0 - total / self._peak_value_eur
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown

        # Keep snapshots bounded (last 10000)
        if len(self._snapshots) > 10000:
            self._snapshots = self._snapshots[-10000:]

    # ─── Metrics computation ─────────────────────────────────────────────

    def compute_trade_stats(self) -> TradeStats:
        """Compute aggregate trade statistics."""
        if not self._trades:
            return TradeStats(
                total_trades=0, buy_count=0, sell_count=0,
                filled_count=0, partial_count=0,
                cancelled_count=0, failed_count=0,
                total_btc_bought=0.0, total_btc_sold=0.0,
                total_eur_spent=0.0, total_eur_received=0.0,
                total_fees_eur=0.0, avg_buy_price=0.0,
                avg_sell_price=0.0, avg_chase_count=0.0,
                avg_fill_time_seconds=0.0,
            )

        buy_count = 0
        sell_count = 0
        filled = 0
        partial = 0
        cancelled = 0
        failed = 0
        btc_bought = 0.0
        btc_sold = 0.0
        eur_spent = 0.0
        eur_received = 0.0
        fees = 0.0
        chase_sum = 0.0
        fill_time_sum = 0.0
        successful_count = 0

        for t in self._trades:
            if t["side"] == "buy":
                buy_count += 1
            else:
                sell_count += 1

            outcome = t["outcome"]
            if outcome == "filled":
                filled += 1
            elif outcome == "partially_filled":
                partial += 1
            elif outcome == "cancelled":
                cancelled += 1
            elif outcome == "failed":
                failed += 1

            if t["filled_volume"] > 0:
                successful_count += 1
                chase_sum += t["chase_count"]
                fill_time_sum += t["elapsed_seconds"]

                if t["side"] == "buy":
                    btc_bought += t["filled_volume"]
                    eur_spent += t["eur_value"]
                else:
                    btc_sold += t["filled_volume"]
                    eur_received += t["eur_value"]

            fees += t.get("fee_eur", 0.0)

        avg_buy = eur_spent / btc_bought if btc_bought > 0 else 0.0
        avg_sell = eur_received / btc_sold if btc_sold > 0 else 0.0
        avg_chase = chase_sum / successful_count if successful_count > 0 else 0.0
        avg_time = fill_time_sum / successful_count if successful_count > 0 else 0.0

        return TradeStats(
            total_trades=len(self._trades),
            buy_count=buy_count,
            sell_count=sell_count,
            filled_count=filled,
            partial_count=partial,
            cancelled_count=cancelled,
            failed_count=failed,
            total_btc_bought=btc_bought,
            total_btc_sold=btc_sold,
            total_eur_spent=eur_spent,
            total_eur_received=eur_received,
            total_fees_eur=fees,
            avg_buy_price=avg_buy,
            avg_sell_price=avg_sell,
            avg_chase_count=avg_chase,
            avg_fill_time_seconds=avg_time,
        )

    def compute_accumulation_metrics(
        self, current_price: float,
    ) -> AccumulationMetrics:
        """Compute BTC accumulation effectiveness vs DCA baseline."""
        stats = self.compute_trade_stats()
        total_accumulated = stats.net_btc_accumulated
        avg_cost = stats.avg_buy_price

        # DCA baseline
        dca_btc = sum(d["btc_amount"] for d in self._dca_deposits)

        # Efficiency ratio
        if dca_btc > 0:
            efficiency = total_accumulated / dca_btc
        elif total_accumulated > 0:
            efficiency = float("inf")
        else:
            efficiency = 1.0

        # Tracking duration
        if self._tracking_start:
            days = (time.time() - self._tracking_start) / 86400.0
        else:
            days = 0.0

        btc_per_day = total_accumulated / days if days > 0 else 0.0

        # Unrealized P&L
        if avg_cost > 0 and current_price > 0:
            unrealized_pnl = (current_price - avg_cost) / avg_cost
        else:
            unrealized_pnl = 0.0

        return AccumulationMetrics(
            total_btc_accumulated=total_accumulated,
            dca_baseline_btc=dca_btc,
            efficiency_ratio=efficiency,
            btc_per_day=btc_per_day,
            avg_cost_basis=avg_cost,
            current_price=current_price,
            unrealized_pnl_pct=unrealized_pnl,
            tracking_days=days,
        )

    def compute_phase_breakdown(self) -> dict[str, dict]:
        """Performance breakdown by cycle phase."""
        breakdown: dict[str, dict] = {}

        for t in self._trades:
            phase = t.get("phase", "unknown")
            if phase not in breakdown:
                breakdown[phase] = {
                    "trades": 0,
                    "btc_bought": 0.0,
                    "btc_sold": 0.0,
                    "eur_spent": 0.0,
                    "eur_received": 0.0,
                    "fees": 0.0,
                }

            b = breakdown[phase]
            b["trades"] += 1

            if t["filled_volume"] > 0:
                if t["side"] == "buy":
                    b["btc_bought"] += t["filled_volume"]
                    b["eur_spent"] += t["eur_value"]
                else:
                    b["btc_sold"] += t["filled_volume"]
                    b["eur_received"] += t["eur_value"]
            b["fees"] += t.get("fee_eur", 0.0)

        return breakdown
    
   


    # ─── Reporting ───────────────────────────────────────────────────────

    def generate_report(
        self, portfolio: PortfolioState,
    ) -> PerformanceReport:
        """Generate a complete performance report."""
        stats = self.compute_trade_stats()
        accum = self.compute_accumulation_metrics(portfolio.btc_price)
        phase_breakdown = self.compute_phase_breakdown()

        current_dd = 0.0
        if self._peak_value_eur > 0:
            current_dd = max(
                0.0,
                1.0 - portfolio.total_value_eur / self._peak_value_eur,
            )

        now = datetime.now(timezone.utc).isoformat()
        start = (
            datetime.fromtimestamp(
                self._tracking_start, tz=timezone.utc,
            ).isoformat()
            if self._tracking_start
            else now
        )

        return PerformanceReport(
            trade_stats=stats,
            accumulation=accum,
            portfolio=portfolio,
            max_drawdown_pct=self._max_drawdown,
            current_drawdown_pct=current_dd,
            phase_breakdown=phase_breakdown,
            generated_at=now,
            tracking_since=start,
        )

    def format_report(self, report: PerformanceReport) -> str:
        """Format a performance report as human-readable text."""
        lines: list[str] = []
        s = report.trade_stats
        a = report.accumulation
        p = report.portfolio

        lines.append("=" * 60)
        lines.append("  BITCOIN ACCUMULATION BOT — PERFORMANCE REPORT")
        lines.append("=" * 60)
        lines.append(f"  Generated: {report.generated_at}")
        lines.append(f"  Tracking since: {report.tracking_since}")
        lines.append("")

        # Portfolio
        lines.append("── PORTFOLIO ──────────────────────────────────")
        lines.append(f"  EUR balance:   €{p.eur_balance:>12,.2f}")
        lines.append(f"  BTC balance:   {p.btc_balance:>12.8f} BTC")
        lines.append(f"  BTC value:     €{p.btc_value_eur:>12,.2f}")
        lines.append(f"  Total value:   €{p.total_value_eur:>12,.2f}")
        lines.append(f"  BTC price:     €{p.btc_price:>12,.2f}")
        lines.append("")

        # Accumulation
        lines.append("── ACCUMULATION ───────────────────────────────")
        lines.append(f"  BTC accumulated:    {a.total_btc_accumulated:>12.8f}")
        lines.append(f"  DCA baseline:       {a.dca_baseline_btc:>12.8f}")
        eff_str = f"{a.efficiency_ratio:.2f}x" if a.efficiency_ratio != float("inf") else "∞"
        lines.append(f"  Efficiency vs DCA:  {eff_str:>12}")
        lines.append(f"  BTC per day:        {a.btc_per_day:>12.8f}")
        lines.append(f"  Avg cost basis:     €{a.avg_cost_basis:>12,.2f}")
        lines.append(f"  Unrealized P&L:     {a.unrealized_pnl_pct:>12.1%}")
        lines.append(f"  Tracking days:      {a.tracking_days:>12.1f}")
        lines.append("")

        # Trade stats
        lines.append("── TRADES ─────────────────────────────────────")
        lines.append(f"  Total trades:       {s.total_trades:>8}")
        lines.append(f"  Buys / Sells:       {s.buy_count:>4} / {s.sell_count:<4}")
        lines.append(f"  Fill rate:          {s.fill_rate:>8.1%}")
        lines.append(f"  Avg chase count:    {s.avg_chase_count:>8.1f}")
        lines.append(f"  Avg fill time:      {s.avg_fill_time_seconds:>8.1f}s")
        lines.append(f"  Total fees:         €{s.total_fees_eur:>8,.2f} ({s.fee_pct:.3%})")
        lines.append("")

        # Risk
        lines.append("── RISK ───────────────────────────────────────")
        lines.append(f"  Max drawdown:       {report.max_drawdown_pct:>8.1%}")
        lines.append(f"  Current drawdown:   {report.current_drawdown_pct:>8.1%}")
        lines.append("")

        # Phase breakdown
        if report.phase_breakdown:
            lines.append("── PHASE BREAKDOWN ────────────────────────────")
            for phase, data in sorted(report.phase_breakdown.items()):
                net = data["btc_bought"] - data["btc_sold"]
                lines.append(
                    f"  {phase:<16} "
                    f"trades={data['trades']:>3}  "
                    f"net_btc={net:>+.6f}  "
                    f"fees=€{data['fees']:>6.2f}"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ─── State persistence ───────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self._persistence.get_path("performance_state.json")

    def _load_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._trades = data.get("trades", [])
            self._dca_deposits = data.get("dca_deposits", [])
            self._peak_value_eur = float(data.get("peak_value_eur", 0.0))
            self._max_drawdown = float(data.get("max_drawdown", 0.0))
            start = data.get("tracking_start")
            self._tracking_start = float(start) if start is not None else None
            logger.info(
                f"Loaded performance state: {len(self._trades)} trades, "
                f"max_dd={self._max_drawdown:.1%}"
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to load performance state: {exc}")

    def _save_state(self) -> None:
        try:
            data = {
                "trades": self._trades,
                "dca_deposits": self._dca_deposits,
                "peak_value_eur": self._peak_value_eur,
                "max_drawdown": self._max_drawdown,
                "tracking_start": self._tracking_start,
            }
            self._state_path().write_text(json.dumps(data, indent=2))
        except OSError as exc:
            logger.error(f"Failed to save performance state: {exc}")

 