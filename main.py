"""
Main bot orchestrator — wires every module into the decision pipeline.

Architecture:
  Fast loop (every 2 min): ticker → indicators → cycle → signal → risk → size → execute
  Slow loop (every 30 min): on-chain metrics + LLM analysis (cached for fast loop)

Key lesson from previous bot: the decision pipeline was disconnected —
10 modules existed but were never called. Here, every module is called
from exactly one place in the loop, and every call's output feeds the
next stage. No dead code.

Lifecycle:
  1. startup(): Initialize all components, health checks, load state.
  2. run(): Main loop — alternates fast/slow based on timing.
  3. shutdown(): Clean up, persist state, generate final report.

Paper trade mode:
  When config.paper_trade is True, the bot runs the full pipeline but
  skips the actual API call in execute_buy/execute_sell. Instead it
  logs what it WOULD have done. This lets you validate the strategy
  with real market data before committing real capital.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional

from config import ATHTracker, BotConfig, CyclePhase, Urgency
from kraken_api import KrakenAPI, OHLCCandle
from indicators import compute_snapshot, TechnicalSnapshot
from bitcoin_node import BitcoinNode, OnChainSnapshot
from cycle_detector import CycleDetector, CycleState
from signal_engine import SignalEngine, CompositeSignal, LLMContext, Action
from risk_manager import RiskManager, PortfolioState, RiskDecision
from position_sizer import PositionSizer, BuySize
from ollama_analyst import OllamaAnalyst
from trade_executor import TradeExecutor, TradeResult, TradeOutcome
from performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class Bot:
    """
    Bitcoin accumulation bot — main orchestrator.

    Connects all modules into a single decision pipeline.
    Every module is called on every loop iteration.
    """

    def __init__(self, config: Optional[BotConfig] = None) -> None:
        self._config = config or BotConfig()
        self._running = False

        # ── Component initialization ─────────────────────────────────
        self._api = KrakenAPI(self._config.kraken)
        self._ath_tracker = ATHTracker(self._config.persistence)
        self._cycle_detector = CycleDetector(self._config, self._ath_tracker)
        self._signal_engine = SignalEngine(self._config)
        self._risk_manager = RiskManager(self._config)
        self._position_sizer = PositionSizer(self._config)
        self._trade_executor = TradeExecutor(
            self._api, self._risk_manager, self._config,
        )
        self._performance = PerformanceTracker(self._config)

        self._bitcoin_node = BitcoinNode(
            rpc_url=self._config.timing.bitcoin_rpc_url,
            rpc_user=self._config.timing.bitcoin_rpc_user,
            rpc_password=self._config.timing.bitcoin_rpc_password,
            cache_ttl=self._config.timing.onchain_cache_ttl,
        )
        self._ollama = OllamaAnalyst(
            config=self._config,
            base_url=self._config.timing.ollama_url,
            model=self._config.timing.ollama_model,
            timeout=self._config.timing.ollama_timeout,
        )

        # ── Cached slow-loop data ────────────────────────────────────
        self._onchain_cache: Optional[OnChainSnapshot] = None
        self._llm_cache: Optional[LLMContext] = None
        self._daily_candles: list[OHLCCandle] = []
        self._last_slow_loop: float = 0.0

        # ── Accumulated OHLCV from fast loop ─────────────────────────
        self._fast_candles: list[OHLCCandle] = []

        # ── DCA floor tracking via risk_manager ───────────────────────


    # ─── Lifecycle ───────────────────────────────────────────────────────

    def startup(self) -> bool:
        """
        Initialize components and run health checks.

        Returns True if minimum requirements are met to start trading.
        """
        logger.info("=" * 60)
        logger.info("  BITCOIN ACCUMULATION BOT — STARTING")
        logger.info("=" * 60)
        logger.info(f"  Mode: {'PAPER TRADE' if self._config.paper_trade else 'LIVE'}")
        logger.info(f"  Pair: {self._config.kraken.pair}")
        logger.info(f"  Fast loop: {self._config.timing.fast_loop_seconds}s")
        logger.info(f"  Slow loop: {self._config.timing.slow_loop_seconds}s")

        # Health checks
        checks: dict[str, bool] = {}

        # Kraken API — mandatory
        ticker = self._api.get_ticker()
        checks["kraken_api"] = ticker is not None
        if ticker:
            logger.info(f"  Kraken: connected (BTC/EUR = €{ticker.last:,.0f})")
        else:
            logger.error("  Kraken: FAILED — cannot start without exchange access")

        # Balance check (unless paper trading)
        if not self._config.paper_trade:
            balance = self._api.get_balance()
            checks["balance"] = balance is not None
            if balance:
                logger.info(f"  Balance: €{balance.eur:,.2f} + {balance.btc:.8f} BTC")
            else:
                logger.warning("  Balance: unavailable")
        else:
            checks["balance"] = True

        # Bitcoin node — optional
        node_ok = self._bitcoin_node.is_available
        checks["bitcoin_node"] = node_ok
        logger.info(f"  Bitcoin node: {'connected' if node_ok else 'unavailable (optional)'}")

        # Ollama — optional
        ollama_ok = self._ollama.health_check()
        checks["ollama"] = ollama_ok
        logger.info(f"  Ollama: {'connected' if ollama_ok else 'unavailable (optional)'}")

        # ATH tracker
        logger.info(f"  ATH: €{self._ath_tracker.ath_eur:,.0f}")

        # Load daily candles for cycle detector
        self._load_daily_candles()
        logger.info(f"  Daily candles loaded: {len(self._daily_candles)}")

        # Mandatory: Kraken API must be reachable
        can_start = checks["kraken_api"]

        if can_start:
            logger.info("  Status: READY")
        else:
            logger.error("  Status: CANNOT START — mandatory checks failed")

        logger.info("=" * 60)
        return can_start

    def run(self) -> None:
        """
        Main event loop.

        Alternates between fast and slow loops based on timing config.
        Runs until shutdown() is called or a signal is received.
        """
        self._running = True

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Entering main loop")

        while self._running:
            try:
                now = time.time()

                # Slow loop check
                slow_interval = self._config.timing.slow_loop_seconds
                if now - self._last_slow_loop >= slow_interval:
                    self._slow_loop()
                    self._last_slow_loop = now

                # Fast loop — the core decision pipeline
                self._fast_loop()

                # Sleep until next fast loop
                elapsed = time.time() - now
                sleep_time = max(
                    1.0,
                    self._config.timing.fast_loop_seconds - elapsed,
                )
                # Sleep in short intervals so we can respond to shutdown quickly
                sleep_end = time.time() + sleep_time
                while self._running and time.time() < sleep_end:
                    time.sleep(min(5.0, sleep_end - time.time()))

            except Exception:
                logger.exception("Unhandled exception in main loop")
                # Sleep before retrying to avoid tight error loops
                time.sleep(30.0)

    def shutdown(self) -> None:
        """Clean shutdown: persist state, generate final report, close connections."""
        logger.info("Shutting down...")
        self._running = False

        # Generate final performance report
        try:
            balance = self._api.get_balance()
            ticker = self._api.get_ticker()
            if balance and ticker:
                portfolio = PortfolioState(
                    eur_balance=balance.eur,
                    btc_balance=balance.btc,
                    btc_price=ticker.last,
                    starting_eur=balance.eur + balance.btc * ticker.last,
                )
                report = self._performance.generate_report(portfolio)
                text = self._performance.format_report(report)
                logger.info(f"\n{text}")
        except Exception:
            logger.exception("Failed to generate final report")

        self._ollama.close()
        logger.info("Shutdown complete")

    # ─── Fast loop — core decision pipeline ──────────────────────────────

    def _fast_loop(self) -> None:
        """
        Fast loop: ticker → indicators → cycle → signal → risk → size → execute.

        This is the beating heart. Every module is called, every output
        feeds the next stage. No dead code.
        """
        # 1. Fetch current market data
        ticker = self._api.get_ticker()
        if ticker is None:
            logger.warning("Fast loop: no ticker — skipping cycle")
            return

        candles = self._api.get_ohlc(interval=5)  # 5-min candles
        if len(candles) < 30:
            logger.warning(f"Fast loop: only {len(candles)} candles — need 30+")
            return

        # 2. Compute technical indicators
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        snapshot = compute_snapshot(
            highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamp=time.time(),
            config=self._config.indicators,
        )

        # 3. Cycle analysis (uses daily candles from slow loop cache)
        daily_closes = [c.close for c in self._daily_candles]
        daily_highs = [c.high for c in self._daily_candles]
        daily_lows = [c.low for c in self._daily_candles]

        cycle = self._cycle_detector.analyze(
            snapshot=snapshot,
            closes_daily=daily_closes,
            highs_daily=daily_highs,
            lows_daily=daily_lows,
        )

        # 4. Generate composite signal
        composite = self._signal_engine.generate(
            snapshot=snapshot,
            cycle=cycle,
            onchain=self._onchain_cache,
            llm=self._llm_cache,
        )

        # 5. Build portfolio state
        balance = self._api.get_balance()
        if balance is None:
            logger.warning("Fast loop: no balance — skipping execution")
            return

        portfolio = PortfolioState(
            eur_balance=balance.eur,
            btc_balance=balance.btc,
            btc_price=ticker.last,
            starting_eur=balance.eur + balance.btc * ticker.last,
        )

        # Snapshot for drawdown tracking
        self._performance.snapshot_portfolio(portfolio)

        # 6. Check for emergency conditions
        emergency = self._risk_manager.emergency_sell(portfolio, cycle)
        if emergency.allowed:
            logger.warning(f"EMERGENCY SELL: {emergency.reason}")
            self._handle_emergency_sell(portfolio, cycle, emergency.reason)
            return

        # 7. Check profit-taking tiers
        if composite.is_sell or cycle.profit_taking_active:
            avg_entry = self._get_avg_entry_price()
            if avg_entry > 0:
                sell_decision = self._position_sizer.compute_sell_tiers(
                    portfolio=portfolio,
                    cycle=cycle,
                    avg_entry_price=avg_entry,
                )
                if sell_decision.should_sell and sell_decision.tier is not None:
                    self._handle_sell(
                        portfolio, cycle, sell_decision.tier,
                    )
                    return
            else:
                logger.debug(
                    "Profit-taking skipped: no avg entry price "
                    "(no filled buy trades yet)"
                )

        # 8. Evaluate buy signal
        if composite.is_buy and composite.actionable:
            risk_decision = self._risk_manager.can_trade(
                composite, portfolio, cycle,
            )

            if risk_decision.allowed:
                buy_size = self._position_sizer.compute_buy_size(
                    signal=composite,
                    portfolio=portfolio,
                    cycle=cycle,
                    risk=risk_decision,
                )
                if buy_size.eur_amount > 0:
                    self._handle_buy(
                        portfolio, cycle, composite, buy_size, risk_decision,
                    )
                    return

        # 9. DCA floor: if no buy occurred and it's been too long, force minimum buy
        if self._should_dca_floor(portfolio, cycle, composite):
            bought = self._handle_dca_floor_buy(portfolio, cycle, composite)
            if bought:
                return

        # 10. Log current state (no action taken)
        logger.info(
            f"Loop: price=€{ticker.last:,.0f} "
            f"phase={cycle.phase.value} "
            f"signal={composite.score:+.1f} ({composite.action.value}) "
            f"quality={composite.data_quality:.2f} "
            f"agreement={composite.agreement:.2f}"
        )

    # ─── DCA floor ─────────────────────────────────────────────────────

    def _should_dca_floor(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
        signal: CompositeSignal,
    ) -> bool:
        """
        Check if a DCA floor buy should trigger.

        Triggers when:
        - DCA floor is enabled in config
        - It's been longer than dca_floor_interval_hours since last buy
        - We have spendable EUR above reserve
        - We're not in distribution/euphoria (don't force-buy near peaks)
        """
        cfg = self._config.sizing
        if not cfg.dca_floor_enabled:
            return False

        # Don't force buys near cycle peaks
        if cycle.phase in (CyclePhase.EUPHORIA, CyclePhase.DISTRIBUTION):
            return False

        # Check time since last buy
        hours_since_buy = int((time.time()) - self._risk_manager.last_buy_time) / 3600
        if hours_since_buy < cfg.dca_floor_interval_hours:
            return False

        # Reserve is based on EUR balance only, not total portfolio.
        # Using total portfolio creates a death spiral: more BTC → higher
        # reserve → less spendable EUR → can't accumulate more BTC.
        reserve = portfolio.eur_balance * self._config.risk.reserve_floor_pct
        spendable = max(0.0, portfolio.eur_balance - reserve)
        if spendable <= 0:
            return False

        return True

    def _handle_dca_floor_buy(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
        signal: CompositeSignal,
    ) -> bool:
        """
        Execute a minimum-size DCA floor buy.

        Returns True if a buy order was placed, False if skipped.
        """
        cfg = self._config.sizing
        # Reserve based on EUR balance only (not total portfolio)
        reserve = portfolio.eur_balance * self._config.risk.reserve_floor_pct
        spendable = max(0.0, portfolio.eur_balance - reserve)

        eur_amount = spendable * cfg.dca_floor_fraction
        min_eur = self._config.kraken.min_order_btc * portfolio.btc_price

        # If calculated amount is below minimum but we have enough spendable
        # EUR, bump up to minimum. The floor is a safety net — it must buy.
        if eur_amount < min_eur:
            if spendable >= min_eur:
                eur_amount = min_eur
            else:
                logger.info(
                    f"DCA floor: €{spendable:.0f} spendable below "
                    f"minimum order €{min_eur:.0f} — skipping"
                )
                return False

        hours_since = (time.time() - self._risk_manager.last_buy_time) / 3600

        logger.info(
            f"DCA FLOOR: €{eur_amount:,.0f} "
            f"({hours_since:.0f}h since last buy, "
            f"phase={cycle.phase.value})"
        )

        buy_size = BuySize(
            eur_amount=eur_amount,
            # Ensure at least min_order_btc to avoid "below minimum" errors
            btc_amount=max(eur_amount / portfolio.btc_price, self._config.kraken.min_order_btc),
            fraction_of_capital=eur_amount / spendable if spendable > 0 else 0.0,
            adjustments={"dca_floor": 1.0},
            reason=f"DCA floor: {hours_since:.0f}h without buy",
        )

        # Use LOW urgency — no rush, just maintaining accumulation
        risk = RiskDecision(allowed=True, reason="DCA floor override")
        self._handle_buy(portfolio, cycle, signal, buy_size, risk)
        return True

    # ─── Slow loop — background data refresh ─────────────────────────────

    def _slow_loop(self) -> None:
        """
        Slow loop: refresh on-chain metrics, LLM analysis, daily candles.

        These are expensive operations that don't need to run every 2 minutes.
        Results are cached and consumed by the fast loop.
        """
        logger.info("Slow loop: refreshing background data")

        # 1. Refresh daily candles
        self._load_daily_candles()

        # 2. On-chain metrics
        try:
            self._onchain_cache = self._bitcoin_node.get_snapshot()
            if self._onchain_cache:
                logger.info(
                    f"On-chain: stress={self._onchain_cache.network_stress:.2f} "
                    f"mempool_clearing={self._onchain_cache.mempool.clearing}"
                )
        except Exception:
            logger.exception("Slow loop: on-chain fetch failed")

        # 3. LLM analysis (needs a recent snapshot for context)
        ticker = self._api.get_ticker()
        if ticker is None:
            logger.warning("Slow loop: no ticker — skipping LLM analysis")
        else:
            candles = self._api.get_ohlc(interval=5)
            if len(candles) >= 30:
                closes = [c.close for c in candles]
                highs = [c.high for c in candles]
                lows = [c.low for c in candles]
                volumes = [c.volume for c in candles]

                snapshot = compute_snapshot(
                    highs=highs, lows=lows, closes=closes,
                    volumes=volumes, timestamp=time.time(),
                )

                daily_closes = [c.close for c in self._daily_candles]
                daily_highs = [c.high for c in self._daily_candles]
                daily_lows = [c.low for c in self._daily_candles]

                cycle = self._cycle_detector.analyze(
                    snapshot, daily_closes, daily_highs, daily_lows,
                )

                llm_result = self._ollama.analyze(snapshot, cycle)
                if llm_result:
                    self._llm_cache = llm_result
                    logger.info(
                        f"LLM: regime={llm_result.regime} "
                        f"sentiment={llm_result.sentiment:+.2f} "
                        f"risk={llm_result.risk_level}"
                    )
                else:
                    logger.info("LLM: analysis unavailable")
            else:
                logger.warning(
                    f"Slow loop: only {len(candles)} candles — "
                    f"need 30+ for LLM analysis"
                )

        logger.info("Slow loop: complete")

    # ─── Trade handlers ──────────────────────────────────────────────────

    def _handle_buy(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
        signal: CompositeSignal,
        buy_size: "PositionSizer.BuySize",
        risk: "RiskManager.RiskDecision",
    ) -> None:
        """Execute a buy order and record results."""
        # Determine urgency from signal strength
        if signal.score >= 60:
            urgency = Urgency.HIGH
        elif signal.score >= 35:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW

        logger.info(
            f"BUY: €{buy_size.eur_amount:,.0f} "
            f"(signal={signal.score:+.1f}, phase={cycle.phase.value}, "
            f"urgency={urgency.value})"
        )

        if self._config.paper_trade:
            logger.info(f"[PAPER] Would buy €{buy_size.eur_amount:,.0f} of BTC")
            self._risk_manager.record_buy()
            return

        result = self._trade_executor.execute_buy(buy_size, urgency)

        if result.success:
            self._risk_manager.record_buy()
            self._performance.record_trade(result, cycle.phase)
            self._performance.record_dca_baseline(
                buy_size.eur_amount, portfolio.btc_price,
            )
            logger.info(
                f"BUY FILLED: {result.filled_volume:.8f} BTC "
                f"@ €{result.filled_price:,.1f} "
                f"(fee=€{result.fee_eur:.2f})"
            )
        else:
            logger.warning(f"BUY FAILED: {result.reason}")

    def _handle_sell(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
        tier: "PositionSizer.SellTier",
    ) -> None:
        """Execute a profit-taking sell and mark the tier."""
        logger.info(
            f"SELL: tier {tier.tier_index} — "
            f"{tier.sell_pct:.0%} of position ({tier.btc_amount:.8f} BTC)"
        )

        if self._config.paper_trade:
            logger.info(f"[PAPER] Would sell {tier.btc_amount:.8f} BTC (tier {tier.tier_index})")
            self._position_sizer.mark_tier_hit(tier.tier_index)
            return

        result = self._trade_executor.execute_sell(tier, Urgency.MEDIUM)

        if result.success:
            self._position_sizer.mark_tier_hit(tier.tier_index)
            self._performance.record_trade(result, cycle.phase)
            logger.info(
                f"SELL FILLED: {result.filled_volume:.8f} BTC "
                f"@ €{result.filled_price:,.1f}"
            )
        else:
            logger.warning(f"SELL FAILED: {result.reason}")

    def _handle_emergency_sell(
        self,
        portfolio: PortfolioState,
        cycle: CycleState,
        reason: str,
    ) -> None:
        """Handle emergency sell — sell a portion of BTC to reduce exposure."""
        sell_pct = 0.25  # Sell 25% of BTC in emergencies
        btc_amount = portfolio.btc_balance * sell_pct

        if btc_amount < self._config.kraken.min_order_btc:
            logger.warning("Emergency sell: BTC balance too low to sell")
            return

        from position_sizer import SellTier
        emergency_tier = SellTier(
            tier_index=-1,
            threshold_pct=0.0,
            sell_pct=sell_pct,
            btc_amount=btc_amount,
            reason=f"EMERGENCY: {reason}",
        )

        if self._config.paper_trade:
            logger.info(f"[PAPER] EMERGENCY would sell {btc_amount:.8f} BTC")
            return

        result = self._trade_executor.execute_sell(
            emergency_tier, Urgency.HIGH,
        )

        if result.success:
            self._performance.record_trade(result, cycle.phase)
            logger.warning(
                f"EMERGENCY SELL FILLED: {result.filled_volume:.8f} BTC "
                f"@ €{result.filled_price:,.1f}"
            )

    # ─── Data loading ────────────────────────────────────────────────────

    def _load_daily_candles(self) -> None:
        """Load daily candles for cycle detector (200+ needed for 200-day MA)."""
        candles = self._api.get_ohlc(interval=1440)  # Daily candles
        if candles:
            self._daily_candles = candles
            logger.debug(f"Loaded {len(candles)} daily candles")

    def _get_avg_entry_price(self) -> float:
        """Compute average entry price from trade history."""
        stats = self._performance.compute_trade_stats()
        return stats.avg_buy_price

    # ─── Signal handlers ─────────────────────────────────────────────────

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle SIGINT/SIGTERM for clean shutdown."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name} — initiating shutdown")
        self.shutdown()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the bot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = BotConfig.load()
    logging.getLogger().setLevel(config.log_level)

    bot = Bot(config)

    if not bot.startup():
        logger.error("Startup failed — exiting")
        sys.exit(1)

    try:
        bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()