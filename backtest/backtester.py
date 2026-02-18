"""
Backtesting engine for the Bitcoin accumulation bot.

Replays historical daily candles through the REAL pipeline modules
(indicators → cycle → signal → risk → sizer). Only trade execution
is simulated (instant fills at close price + Kraken fees).

This is the truth test: does the bot beat naive DCA?

Usage:
    from backtester import BacktestEngine, BacktestConfig
    from data_loader import DataLoader

    loader = DataLoader()
    candles = loader.load_daily("2020-01-01", "2024-12-31")

    engine = BacktestEngine(BacktestConfig(starting_eur=10_000.0))
    result = engine.run(candles)
    print(result.summary())
"""
from __future__ import annotations

import copy
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ATHTracker, BotConfig, CyclePhase, PersistenceConfig,
    SizingConfig, VolatilityRegime,
)
from cycle_detector import CycleDetector, CycleState
from data_loader import HistoricalCandle
from indicators import compute_snapshot, TechnicalSnapshot
from kraken_api import OHLCCandle
from position_sizer import PositionSizer, BuySize, SellDecision
from risk_manager import RiskManager, PortfolioState, RiskDecision
from signal_engine import SignalEngine, CompositeSignal, Action

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Parameters for a backtest run."""
    starting_eur: float = 10_000.0       # Initial EUR balance
    starting_btc: float = 0.0            # Initial BTC balance

    # Fee model (Kraken maker/taker)
    maker_fee: float = 0.0016
    taker_fee: float = 0.0026
    use_maker_fee: bool = True           # Assume limit orders (maker)

    # Spread simulation
    spread_bps: float = 5.0              # Simulated spread in basis points

    # DCA baseline comparison
    dca_interval_days: int = 7           # Buy every N days for DCA baseline
    dca_amount_eur: float = 0.0          # Auto-calculated if 0

    # Bot config overrides (applied to BotConfig)
    bot_config: Optional[BotConfig] = None

    # Indicator window: how many daily candles to feed the pipeline
    indicator_window: int = 200          # Needs 200 for 200d MA

    # Output directory for results
    output_dir: Path = field(default_factory=lambda: Path("backtest_results"))

    @property
    def fee_rate(self) -> float:
        return self.maker_fee if self.use_maker_fee else self.taker_fee


# ─── Trade log ───────────────────────────────────────────────────────────────

@dataclass
class SimulatedTrade:
    """Record of a simulated trade."""
    date: str
    side: str                # "buy" or "sell"
    eur_amount: float
    btc_amount: float
    price: float
    fee_eur: float
    reason: str
    phase: str               # Cycle phase at trade time
    signal_score: float
    portfolio_value_eur: float


@dataclass
class DailySnapshot:
    """End-of-day portfolio state."""
    date: str
    close_price: float
    eur_balance: float
    btc_balance: float
    total_value_eur: float
    btc_accumulated: float   # Total BTC ever bought
    phase: str
    signal_score: float
    action: str


# ─── Backtest result ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Complete backtest output with performance metrics."""
    # Config
    start_date: str
    end_date: str
    starting_eur: float
    starting_btc: float
    num_days: int

    # Bot performance
    final_eur: float
    final_btc: float
    final_btc_price: float
    final_value_eur: float
    total_return_pct: float

    # Trade stats
    total_trades: int
    buy_trades: int
    sell_trades: int
    emergency_sells: int
    dca_floor_buys: int
    total_fees_eur: float
    avg_buy_price: float

    # BTC accumulation
    total_btc_bought: float
    total_btc_sold: float
    net_btc_accumulated: float

    # DCA comparison
    dca_final_btc: float
    dca_final_value_eur: float
    dca_avg_buy_price: float
    dca_total_fees_eur: float
    btc_advantage_pct: float       # Bot BTC vs DCA BTC
    value_advantage_pct: float     # Bot value vs DCA value

    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_date: str
    peak_value_eur: float

    # Phase breakdown
    trades_by_phase: dict[str, int]
    btc_by_phase: dict[str, float]

    # Raw data
    trades: list[SimulatedTrade]
    daily_snapshots: list[DailySnapshot]

    def summary(self) -> str:
        """Human-readable performance summary."""
        lines = [
            "",
            "═" * 70,
            "  BACKTEST RESULTS",
            "═" * 70,
            f"  Period:          {self.start_date} → {self.end_date} ({self.num_days} days)",
            f"  Starting:        €{self.starting_eur:,.0f} + {self.starting_btc:.4f} BTC",
            "",
            "─── PORTFOLIO ───────────────────────────────────────────────",
            f"  Final EUR:       €{self.final_eur:,.0f}",
            f"  Final BTC:       {self.final_btc:.8f}",
            f"  Final value:     €{self.final_value_eur:,.0f}",
            f"  Total return:    {self.total_return_pct:+.1f}%",
            f"  Max drawdown:    {self.max_drawdown_pct:.1f}% (on {self.max_drawdown_date})",
            "",
            "─── ACCUMULATION ────────────────────────────────────────────",
            f"  BTC bought:      {self.total_btc_bought:.8f}",
            f"  BTC sold:        {self.total_btc_sold:.8f}",
            f"  Net accumulated: {self.net_btc_accumulated:.8f}",
            f"  Avg buy price:   €{self.avg_buy_price:,.0f}",
            f"  Total fees:      €{self.total_fees_eur:,.2f}",
            "",
            "─── TRADES ──────────────────────────────────────────────────",
            f"  Total:           {self.total_trades}",
            f"  Buys:            {self.buy_trades}",
            f"  Sells:           {self.sell_trades}",
            f"  Emergency sells: {self.emergency_sells}",
            f"  DCA floor buys:  {self.dca_floor_buys}",
            "",
            "─── vs DCA BASELINE ─────────────────────────────────────────",
            f"  DCA BTC:         {self.dca_final_btc:.8f}",
            f"  DCA value:       €{self.dca_final_value_eur:,.0f}",
            f"  DCA avg price:   €{self.dca_avg_buy_price:,.0f}",
            f"  DCA fees:        €{self.dca_total_fees_eur:,.2f}",
            f"  BTC advantage:   {self.btc_advantage_pct:+.1f}%",
            f"  Value advantage: {self.value_advantage_pct:+.1f}%",
            "",
            "─── PHASE BREAKDOWN ─────────────────────────────────────────",
        ]
        for phase, count in sorted(self.trades_by_phase.items()):
            btc = self.btc_by_phase.get(phase, 0.0)
            lines.append(f"  {phase:20s}  {count:3d} trades  {btc:+.8f} BTC")

        lines.extend([
            "",
            "═" * 70,
            f"  VERDICT: Bot {'BEATS' if self.btc_advantage_pct > 0 else 'LOSES TO'} "
            f"DCA by {abs(self.btc_advantage_pct):.1f}% in BTC accumulated",
            "═" * 70,
            "",
        ])

        return "\n".join(lines)


# ─── Engine ──────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Replay historical candles through the real trading pipeline.

    Uses the actual indicator, cycle, signal, risk, and sizing modules.
    Only trade execution is simulated (instant fills at close + fees).
    """

    def __init__(
        self,
        config: BacktestConfig = BacktestConfig(),
    ) -> None:
        self._cfg = config
        self._tmp_dir: Optional[Path] = None

    def run(self, candles: list[HistoricalCandle]) -> BacktestResult:
        """
        Run the full backtest.

        Args:
            candles: Daily OHLCV candles, oldest first. Need at least
                     indicator_window + 50 candles for meaningful results.

        Returns:
            BacktestResult with full performance breakdown.
        """
        if len(candles) < self._cfg.indicator_window + 10:
            raise ValueError(
                f"Need at least {self._cfg.indicator_window + 10} candles, "
                f"got {len(candles)}"
            )

        # ── Setup ─────────────────────────────────────────────────────
        self._setup_tmp_dir()
        bot_cfg = self._make_bot_config()

        # Initialize real pipeline modules
        ath_tracker = ATHTracker(bot_cfg.persistence)
        cycle_detector = CycleDetector(bot_cfg, ath_tracker)
        signal_engine = SignalEngine(bot_cfg)
        risk_manager = RiskManager(bot_cfg)
        position_sizer = PositionSizer(bot_cfg)

        # Portfolio state
        eur = self._cfg.starting_eur
        btc = self._cfg.starting_btc
        starting_value = eur + btc * candles[0].close

        # Tracking
        trades: list[SimulatedTrade] = []
        snapshots: list[DailySnapshot] = []
        peak_value = starting_value
        max_drawdown = 0.0
        max_drawdown_date = candles[0].date
        total_btc_bought = 0.0
        total_btc_sold = 0.0
        total_fees = 0.0
        emergency_sells = 0
        dca_floor_buys = 0
        trades_by_phase: dict[str, int] = {}
        btc_by_phase: dict[str, float] = {}
        buy_eur_spent = 0.0
        last_buy_time = 0.0

        # DCA baseline
        dca_eur = self._cfg.starting_eur
        dca_btc = self._cfg.starting_btc
        dca_total_btc_bought = 0.0
        dca_total_fees = 0.0
        dca_eur_spent = 0.0
        dca_interval = self._cfg.dca_interval_days

        # Auto-calculate DCA amount: spread total EUR evenly
        num_dca_buys = len(candles) // dca_interval
        dca_amount = self._cfg.dca_amount_eur
        if dca_amount <= 0 and num_dca_buys > 0:
            dca_amount = (self._cfg.starting_eur * 0.80) / num_dca_buys

        # ── Main simulation loop ──────────────────────────────────────
        start_idx = self._cfg.indicator_window
        logger.info(
            f"Starting backtest: {candles[start_idx].date} → "
            f"{candles[-1].date} ({len(candles) - start_idx} trading days)"
        )

        for i in range(start_idx, len(candles)):
            candle = candles[i]
            price = candle.close
            day_idx = i - start_idx

            # ── Build indicator window ────────────────────────────────
            window = candles[max(0, i - self._cfg.indicator_window):i + 1]
            highs = [c.high for c in window]
            lows = [c.low for c in window]
            closes = [c.close for c in window]
            volumes = [c.volume for c in window]

            # ── Technical snapshot ────────────────────────────────────
            snapshot = compute_snapshot(
                highs=highs, lows=lows, closes=closes,
                volumes=volumes, timestamp=candle.timestamp,
                config=bot_cfg.indicators,
            )

            # ── Cycle analysis ────────────────────────────────────────
            # Use full available daily history for cycle detector
            all_closes = [c.close for c in candles[:i + 1]]
            all_highs = [c.high for c in candles[:i + 1]]
            all_lows = [c.low for c in candles[:i + 1]]

            cycle = cycle_detector.analyze(
                snapshot, all_closes, all_highs, all_lows,
            )

            # ── Composite signal ──────────────────────────────────────
            # No on-chain or LLM in backtesting (not available historically)
            composite = signal_engine.generate(
                snapshot=snapshot, cycle=cycle,
                onchain=None, llm=None,
            )

            # ── Portfolio state ───────────────────────────────────────
            portfolio = PortfolioState(
                eur_balance=eur,
                btc_balance=btc,
                btc_price=price,
                starting_eur=self._cfg.starting_eur,
            )

            # ── Emergency sell check ──────────────────────────────────
            emergency = risk_manager.emergency_sell(portfolio, cycle)
            if emergency.allowed and btc > bot_cfg.kraken.min_order_btc:
                sell_btc = btc * 0.25
                sell_eur, fee = self._simulate_sell(sell_btc, price)
                btc -= sell_btc
                eur += sell_eur
                total_btc_sold += sell_btc
                total_fees += fee
                emergency_sells += 1

                trade = SimulatedTrade(
                    date=candle.date, side="sell",
                    eur_amount=sell_eur, btc_amount=sell_btc,
                    price=price, fee_eur=fee,
                    reason=f"EMERGENCY: {emergency.reason}",
                    phase=cycle.phase.value,
                    signal_score=composite.score,
                    portfolio_value_eur=eur + btc * price,
                )
                trades.append(trade)
                self._track_phase(trades_by_phase, btc_by_phase, cycle.phase, -sell_btc)

            # ── Profit-taking check ───────────────────────────────────
            elif (composite.is_sell or cycle.profit_taking_active) and btc > 0:
                avg_entry = buy_eur_spent / total_btc_bought if total_btc_bought > 0 else 0.0
                if avg_entry > 0:
                    sell_decision = position_sizer.compute_sell_tiers(
                        portfolio=portfolio,
                        cycle=cycle,
                        avg_entry_price=avg_entry,
                    )
                    if sell_decision.should_sell and sell_decision.tier is not None:
                        tier = sell_decision.tier
                        sell_btc = min(tier.btc_amount, btc)
                        sell_eur, fee = self._simulate_sell(sell_btc, price)
                        btc -= sell_btc
                        eur += sell_eur
                        total_btc_sold += sell_btc
                        total_fees += fee
                        position_sizer.mark_tier_hit(tier.tier_index)

                        trade = SimulatedTrade(
                            date=candle.date, side="sell",
                            eur_amount=sell_eur, btc_amount=sell_btc,
                            price=price, fee_eur=fee,
                            reason=f"Tier {tier.tier_index}: {tier.threshold_pct:+.0%}",
                            phase=cycle.phase.value,
                            signal_score=composite.score,
                            portfolio_value_eur=eur + btc * price,
                        )
                        trades.append(trade)
                        self._track_phase(trades_by_phase, btc_by_phase, cycle.phase, -sell_btc)

            # ── Buy evaluation ────────────────────────────────────────
            elif composite.is_buy and composite.actionable:
                risk_decision = risk_manager.can_trade(
                    composite, portfolio, cycle,
                )

                if risk_decision.allowed:
                    buy_size = position_sizer.compute_buy_size(
                        signal=composite,
                        portfolio=portfolio,
                        cycle=cycle,
                        risk=risk_decision,
                    )
                    if buy_size.eur_amount > 0 and buy_size.eur_amount <= eur:
                        buy_btc, fee = self._simulate_buy(
                            buy_size.eur_amount, price,
                        )
                        eur -= buy_size.eur_amount
                        btc += buy_btc
                        total_btc_bought += buy_btc
                        buy_eur_spent += buy_size.eur_amount
                        total_fees += fee
                        last_buy_time = candle.timestamp

                        trade = SimulatedTrade(
                            date=candle.date, side="buy",
                            eur_amount=buy_size.eur_amount,
                            btc_amount=buy_btc,
                            price=price, fee_eur=fee,
                            reason=buy_size.reason,
                            phase=cycle.phase.value,
                            signal_score=composite.score,
                            portfolio_value_eur=eur + btc * price,
                        )
                        trades.append(trade)
                        self._track_phase(trades_by_phase, btc_by_phase, cycle.phase, buy_btc)

            # ── DCA floor check ───────────────────────────────────────
            else:
                hours_since = (candle.timestamp - last_buy_time) / 3600 if last_buy_time > 0 else 999
                dca_floor_cfg = bot_cfg.sizing
                if (
                    dca_floor_cfg.dca_floor_enabled
                    and hours_since >= dca_floor_cfg.dca_floor_interval_hours
                    and cycle.phase not in (CyclePhase.EUPHORIA, CyclePhase.DISTRIBUTION)
                ):
                    # Reserve based on current EUR balance, not starting capital
                    # (matches live fix: prevents death spiral)
                    reserve = eur * bot_cfg.risk.reserve_floor_pct
                    spendable = max(0.0, eur - reserve)
                    floor_eur = spendable * dca_floor_cfg.dca_floor_fraction
                    min_eur = bot_cfg.kraken.min_order_btc * price

                    # Bump to minimum if below but spendable allows
                    if floor_eur < min_eur and spendable >= min_eur:
                        floor_eur = min_eur

                    if floor_eur >= min_eur and floor_eur <= eur:
                        buy_btc, fee = self._simulate_buy(floor_eur, price)
                        eur -= floor_eur
                        btc += buy_btc
                        total_btc_bought += buy_btc
                        buy_eur_spent += floor_eur
                        total_fees += fee
                        last_buy_time = candle.timestamp
                        dca_floor_buys += 1

                        trade = SimulatedTrade(
                            date=candle.date, side="buy",
                            eur_amount=floor_eur, btc_amount=buy_btc,
                            price=price, fee_eur=fee,
                            reason=f"DCA floor ({hours_since:.0f}h gap)",
                            phase=cycle.phase.value,
                            signal_score=composite.score,
                            portfolio_value_eur=eur + btc * price,
                        )
                        trades.append(trade)
                        self._track_phase(trades_by_phase, btc_by_phase, cycle.phase, buy_btc)

            # ── DCA baseline ──────────────────────────────────────────
            if day_idx > 0 and day_idx % dca_interval == 0:
                if dca_eur >= dca_amount and dca_amount > 0:
                    dca_buy_btc, dca_fee = self._simulate_buy(
                        dca_amount, price,
                    )
                    dca_eur -= dca_amount
                    dca_btc += dca_buy_btc
                    dca_total_btc_bought += dca_buy_btc
                    dca_total_fees += dca_fee
                    dca_eur_spent += dca_amount

            # ── End-of-day tracking ───────────────────────────────────
            total_value = eur + btc * price
            if total_value > peak_value:
                peak_value = total_value
            dd = (peak_value - total_value) / peak_value if peak_value > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
                max_drawdown_date = candle.date

            snapshots.append(DailySnapshot(
                date=candle.date,
                close_price=price,
                eur_balance=eur,
                btc_balance=btc,
                total_value_eur=total_value,
                btc_accumulated=total_btc_bought - total_btc_sold,
                phase=cycle.phase.value,
                signal_score=composite.score,
                action=composite.action.value,
            ))

        # ── Compile results ───────────────────────────────────────────
        final_price = candles[-1].close
        final_value = eur + btc * final_price
        dca_final_value = dca_eur + dca_btc * final_price

        avg_buy = buy_eur_spent / total_btc_bought if total_btc_bought > 0 else 0.0
        dca_avg_buy = dca_eur_spent / dca_total_btc_bought if dca_total_btc_bought > 0 else 0.0

        net_btc = total_btc_bought - total_btc_sold
        btc_advantage = (
            (net_btc - dca_total_btc_bought) / dca_total_btc_bought * 100
            if dca_total_btc_bought > 0 else 0.0
        )
        value_advantage = (
            (final_value - dca_final_value) / dca_final_value * 100
            if dca_final_value > 0 else 0.0
        )

        buy_trades = sum(1 for t in trades if t.side == "buy")
        sell_trades = sum(1 for t in trades if t.side == "sell")

        result = BacktestResult(
            start_date=candles[start_idx].date,
            end_date=candles[-1].date,
            starting_eur=self._cfg.starting_eur,
            starting_btc=self._cfg.starting_btc,
            num_days=len(candles) - start_idx,
            final_eur=eur,
            final_btc=btc,
            final_btc_price=final_price,
            final_value_eur=final_value,
            total_return_pct=(final_value - starting_value) / starting_value * 100,
            total_trades=len(trades),
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            emergency_sells=emergency_sells,
            dca_floor_buys=dca_floor_buys,
            total_fees_eur=total_fees,
            avg_buy_price=avg_buy,
            total_btc_bought=total_btc_bought,
            total_btc_sold=total_btc_sold,
            net_btc_accumulated=net_btc,
            dca_final_btc=dca_btc,
            dca_final_value_eur=dca_final_value,
            dca_avg_buy_price=dca_avg_buy,
            dca_total_fees_eur=dca_total_fees,
            btc_advantage_pct=btc_advantage,
            value_advantage_pct=value_advantage,
            max_drawdown_pct=max_drawdown * 100,
            max_drawdown_date=max_drawdown_date,
            peak_value_eur=peak_value,
            trades_by_phase=trades_by_phase,
            btc_by_phase=btc_by_phase,
            trades=trades,
            daily_snapshots=snapshots,
        )

        logger.info(f"Backtest complete: {len(trades)} trades over {result.num_days} days")
        return result

    # ─── Trade simulation ────────────────────────────────────────────────

    def _simulate_buy(
        self, eur_amount: float, price: float,
    ) -> tuple[float, float]:
        """
        Simulate a buy trade.

        Returns: (btc_received, fee_eur)
        """
        # Apply spread (buy at slightly above mid)
        effective_price = price * (1 + self._cfg.spread_bps / 10_000)
        fee = eur_amount * self._cfg.fee_rate
        net_eur = eur_amount - fee
        btc = net_eur / effective_price
        return btc, fee

    def _simulate_sell(
        self, btc_amount: float, price: float,
    ) -> tuple[float, float]:
        """
        Simulate a sell trade.

        Returns: (eur_received, fee_eur)
        """
        # Apply spread (sell at slightly below mid)
        effective_price = price * (1 - self._cfg.spread_bps / 10_000)
        gross_eur = btc_amount * effective_price
        fee = gross_eur * self._cfg.fee_rate
        net_eur = gross_eur - fee
        return net_eur, fee

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _setup_tmp_dir(self) -> None:
        """Create temp directory for module state files."""
        import tempfile
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="backtest_"))

    def _make_bot_config(self) -> BotConfig:
        """Create bot config for backtesting."""
        if self._cfg.bot_config:
            cfg = self._cfg.bot_config
        else:
            cfg = BotConfig(
                persistence=PersistenceConfig(
                    base_dir=self._tmp_dir or Path("/tmp/backtest"),
                ),
            )

        # Ensure persistence dir uses our tmp dir
        if self._tmp_dir:
            cfg.persistence = PersistenceConfig(base_dir=self._tmp_dir)
            cfg.persistence.ensure_dirs()

        return cfg

    @staticmethod
    def _track_phase(
        trades_dict: dict[str, int],
        btc_dict: dict[str, float],
        phase: CyclePhase,
        btc_delta: float,
    ) -> None:
        """Track trade count and BTC flow per phase."""
        name = phase.value
        trades_dict[name] = trades_dict.get(name, 0) + 1
        btc_dict[name] = btc_dict.get(name, 0.0) + btc_delta


# ─── CSV export ──────────────────────────────────────────────────────────────

def export_trades_csv(trades: list[SimulatedTrade], path: Path) -> None:
    """Export trade log to CSV."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "side", "eur_amount", "btc_amount", "price",
            "fee_eur", "reason", "phase", "signal_score", "portfolio_value",
        ])
        for t in trades:
            writer.writerow([
                t.date, t.side, f"{t.eur_amount:.2f}",
                f"{t.btc_amount:.8f}", f"{t.price:.2f}",
                f"{t.fee_eur:.4f}", t.reason, t.phase,
                f"{t.signal_score:.1f}", f"{t.portfolio_value_eur:.2f}",
            ])


def export_snapshots_csv(
    snapshots: list[DailySnapshot], path: Path,
) -> None:
    """Export daily snapshots to CSV."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "close_price", "eur_balance", "btc_balance",
            "total_value_eur", "btc_accumulated", "phase",
            "signal_score", "action",
        ])
        for s in snapshots:
            writer.writerow([
                s.date, f"{s.close_price:.2f}",
                f"{s.eur_balance:.2f}", f"{s.btc_balance:.8f}",
                f"{s.total_value_eur:.2f}", f"{s.btc_accumulated:.8f}",
                s.phase, f"{s.signal_score:.1f}", s.action,
            ])
