"""
Configuration module — single source of truth for all bot parameters.

All thresholds, credentials, and tuning knobs live here. No magic numbers
buried in logic elsewhere. Dynamic state (ATH, cycle floor) is tracked and
persisted automatically.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Enums ────────────────────────────────────────────────────────────────────

class CyclePhase(Enum):
    """Bitcoin halving cycle phases."""
    ACCUMULATION = "accumulation"
    EARLY_BULL = "early_bull"
    GROWTH = "growth"
    EUPHORIA = "euphoria"
    DISTRIBUTION = "distribution"
    EARLY_BEAR = "early_bear"
    CAPITULATION = "capitulation"


class VolatilityRegime(Enum):
    """Volatility classification."""
    COMPRESSION = "compression"
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


class Urgency(Enum):
    """Order urgency for spread-aware pricing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ─── Kraken config ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KrakenConfig:
    """Kraken exchange connection and trading parameters."""
    api_key: str = os.environ.get("KRAKEN_API_KEY", "")
    private_key: str = os.environ.get("KRAKEN_API_SECRET", "")
    pair: str = "XXBTZEUR"
    base_url: str = "https://api.kraken.com"

    # Fees
    maker_fee: float = 0.0016
    taker_fee: float = 0.0026

    # Order constraints
    min_order_btc: float = 0.0001
    max_order_pct: float = 0.30  # Never exceed 30% of spendable capital

    # Rate limiting
    max_calls_per_minute: int = 15
    call_spacing_seconds: float = 4.0  # Conservative: 60/15

    # Circuit breaker
    circuit_breaker_threshold: int = 5  # Consecutive failures before cooldown
    circuit_breaker_cooldown: float = 60.0  # Seconds to wait after tripping

    # Retry policy (tenacity parameters)
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
    retry_max_attempts: int = 5

    # Order execution
    fill_timeout_seconds: float = 300.0  # 5 minutes
    execution_retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> KrakenConfig:
        """Load credentials from environment, keep defaults for everything else."""
        api_key = os.environ.get("KRAKEN_API_KEY", "")
        private_key = os.environ.get("KRAKEN_API_SECRET", "")
        if not api_key or not private_key:
            logger.warning("Kraken API credentials not found in environment")
        return cls(api_key=api_key, private_key=private_key)


# ─── Cycle config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CycleConfig:
    """Bitcoin Cycle 4 parameters (halving April 2024)."""
    halving_date: datetime = field(
        default_factory=lambda: datetime(2024, 4, 20, tzinfo=timezone.utc)
    )
    # Estimated cycle duration in days (historically ~1400-1460)
    estimated_cycle_days: int = 1440

    # Phase boundaries as fraction of cycle elapsed
    # Based on historical patterns: accumulation → early bull → growth → euphoria → distribution → bear
    phase_boundaries: dict[str, float] = field(default_factory=lambda: {
        "accumulation_end": 0.15,     # ~216 days post-halving
        "early_bull_end": 0.30,       # ~432 days
        "growth_end": 0.55,           # ~792 days
        "euphoria_end": 0.70,         # ~1008 days
        "distribution_end": 0.85,     # ~1224 days
        # After distribution_end → early_bear → capitulation until next halving
    })

    # Weight allocation for multi-signal phase detection
    time_weight: float = 0.35
    price_structure_weight: float = 0.30
    momentum_weight: float = 0.25
    volatility_weight: float = 0.10

    # Diminishing returns model for cycle ceiling estimation
    # Each cycle peak is roughly this fraction of the previous cycle's multiplier
    diminishing_returns_factor: float = 0.40
    # Cycle 3 peak was ~69k USD (~60k EUR at the time)
    # Cycle 4 conservative ceiling estimate in EUR
    cycle_ceiling_eur: float = 180_000.0
    cycle_floor_eur: float = 20_000.0  # Estimated absolute floor

    # Phase stability: prevent rapid phase flapping
    min_phase_dwell_cycles: int = 30     # Minimum cycles before allowing transition (~60 min at 2-min loop)
    phase_transition_confidence: float = 0.40  # Minimum confidence to accept a new phase
    phase_transition_advantage: float = 0.20   # New phase must beat current by this margin


# ─── Indicator config ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorConfig:
    """Technical indicator parameters."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    vwap_period: int = 14  # Number of candles for VWAP


# ─── Signal engine config ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalConfig:
    """Composite signal generation parameters."""
    # Score range is -100 to +100
    # Minimum agreement among sub-signals before acting
    min_agreement: float = 0.30
    # Buy threshold: score must exceed this to trigger a buy
    buy_threshold: float = 15.3
    # Sell threshold: score must drop below this to trigger a sell
    sell_threshold: float = -20.0

    # Sub-signal weights (should sum to ~1.0)
    rsi_weight: float = 0.20
    macd_weight: float = 0.15
    bollinger_weight: float = 0.15
    cycle_weight: float = 0.20
    onchain_weight: float = 0.10
    llm_weight: float = 0.10
    microstructure_weight: float = 0.10

    # ── Asymmetric agreement thresholds ──────────────────────────
    # For accumulation: be patient buying, eager selling at peaks.
    # When set, these override min_agreement for the respective direction.
    # None = use min_agreement for both (backward compatible).
    buy_min_agreement: Optional[float] = 0.35
    sell_min_agreement: Optional[float] = 0.35

    # ── Adaptive weights by cycle phase ──────────────────────────
    # Keys are CyclePhase.value strings. Each dict overrides the
    # default weights above. Missing keys fall back to defaults.
    # Rationale:
    #   Accumulation/capitulation: RSI oversold + on-chain + cycle dominate
    #   Growth/early_bull: balanced — technicals matter
    #   Euphoria/distribution: cycle + MACD divergence dominate
    phase_weight_overrides: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "accumulation": {
                "rsi_weight": 0.25, "cycle_weight": 0.25,
                "onchain_weight": 0.15, "macd_weight": 0.10,
                "bollinger_weight": 0.10, "llm_weight": 0.10,
                "microstructure_weight": 0.05,
            },
            "capitulation": {
                "rsi_weight": 0.25, "cycle_weight": 0.25,
                "onchain_weight": 0.20, "macd_weight": 0.10,
                "bollinger_weight": 0.05, "llm_weight": 0.10,
                "microstructure_weight": 0.05,
            },
            "euphoria": {
                "cycle_weight": 0.30, "macd_weight": 0.20,
                "rsi_weight": 0.15, "bollinger_weight": 0.10,
                "onchain_weight": 0.10, "llm_weight": 0.10,
                "microstructure_weight": 0.05,
            },
            "distribution": {
                "cycle_weight": 0.30, "macd_weight": 0.20,
                "rsi_weight": 0.15, "bollinger_weight": 0.10,
                "onchain_weight": 0.10, "llm_weight": 0.10,
                "microstructure_weight": 0.05,
            },
        }
    )


# ─── Risk management config ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""
    # Reserve floor: never go below this fraction of starting EUR balance
    reserve_floor_pct: float = 0.12  # Keep 12% of starting EUR as reserve

    # Daily trade limits
    max_daily_trades: int = 10

    # Drawdown tolerance by phase (from portfolio peak)
    drawdown_tolerance: dict[str, float] = field(default_factory=lambda: {
        "accumulation": 0.45,
        "early_bull": 0.35,
        "growth": 0.30,
        "euphoria": 0.20,
        "distribution": 0.15,
        "early_bear": 0.40,
        "capitulation": 0.55,  # Max tolerance — prime DCA territory
    })

    # Emergency sell: only if below estimated cycle floor (golden rule)
    enable_golden_rule_floor: bool = False  # Disabled: emergency sells hurt accumulation

    # Bear capitulation override: allow buys even at elevated risk
    enable_capitulation_override: bool = True

    # Stop loss: ATR multiplier for cycle-adjusted stops
    stop_atr_multiplier: float = 2.5


# ─── Position sizing config ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SizingConfig:
    """Position sizing parameters."""
    # Base method
    base_fraction: float = 0.035  # 3.5% of spendable capital per trade

    # Kelly criterion parameters
    use_kelly: bool = False  # Start with fixed fractional, graduate to Kelly
    kelly_fraction: float = 0.25  # Quarter Kelly for safety

    # Adjustment bounds
    min_adjustment: float = 0.25  # Floor: never size below 25% of base
    max_adjustment: float = 3.0   # Ceiling: never size above 3x base

    # Tiered profit taking (default — used when no phase override matches)
    profit_tiers: list[dict[str, float]] = field(default_factory=lambda: [
        {"threshold": 0.20, "sell_pct": 0.05},   # +20% → sell 5%
        {"threshold": 0.50, "sell_pct": 0.08},   # +50% → sell 8%
        {"threshold": 1.00, "sell_pct": 0.12},   # +100% → sell 12%
        {"threshold": 2.00, "sell_pct": 0.15},   # +200% → sell 15%
    ])

    # ── Value averaging ──────────────────────────────────────────
    # Buy more when price is below 200-day MA (accumulate at better prices).
    # Uses distance_from_200d_ma from CycleState.price_structure.
    # Boost formula: 1 + max_boost * (1 - exp(-sensitivity * distance_below_ma))
    value_avg_enabled: bool = True
    value_avg_max_boost: float = 2.0    # Max extra multiplier when deeply below MA
    value_avg_sensitivity: float = 1.5  # How quickly boost ramps (higher = faster)

    # ── Price acceleration brake (FOMO protection) ───────────────
    # Reduces buy size when price is running up fast (near Bollinger upper band).
    # Uses CycleState.volatility_regime + price_structure.position_in_range.
    acceleration_brake_enabled: bool = True
    acceleration_brake_factor: float = 0.4  # Reduce to 40% when braking

    # ── Phase-aware profit tiers ─────────────────────────────────
    # Override default profit_tiers per cycle phase. Missing phases use default.
    # Growth: hold longer for bigger gains. Euphoria: take profit faster.
    # Distribution: aggressive exit before bear market.
    phase_profit_tiers: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: {
            "growth": [
                {"threshold": 0.30, "sell_pct": 0.03},   # +30% → sell 3%
                {"threshold": 0.60, "sell_pct": 0.05},   # +60% → sell 5%
                {"threshold": 1.00, "sell_pct": 0.08},   # +100% → sell 8%
                {"threshold": 2.00, "sell_pct": 0.12},   # +200% → sell 12%
            ],
            "euphoria": [
                {"threshold": 0.20, "sell_pct": 0.08},   # +20% → sell 8%
                {"threshold": 0.50, "sell_pct": 0.12},   # +50% → sell 12%
                {"threshold": 1.00, "sell_pct": 0.18},   # +100% → sell 18%
            ],
            "distribution": [
                {"threshold": 0.15, "sell_pct": 0.10},   # +15% → sell 10%
                {"threshold": 0.40, "sell_pct": 0.15},   # +40% → sell 15%
                {"threshold": 0.80, "sell_pct": 0.20},   # +80% → sell 20%
            ],
        }
    )

    # ── DCA floor ────────────────────────────────────────────────
    # Ensures accumulation happens even during extended HOLD periods.
    # If no buy occurs within dca_floor_interval_hours, a minimum-size
    # buy is executed regardless of signal state.
    dca_floor_enabled: bool = True
    dca_floor_interval_hours: float = 24.0   # Max hours without a buy
    dca_floor_fraction: float = 0.015        # 1.5% of spendable for floor buys


# ─── Timing config ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimingConfig:
    """Loop timing parameters."""
    # Fast loop: price, indicators, signals, execution
    fast_loop_seconds: float = 120.0  # 2 minutes

    # Slow loop: LLM analysis, on-chain metrics, news
    slow_loop_seconds: float = 1800.0  # 30 minutes

    # On-chain cache TTL
    onchain_cache_ttl: float = 300.0  # 5 minutes

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_timeout: float = 60.0  # Seconds before giving up on LLM call

    # Bitcoin node RPC
    bitcoin_rpc_url: str = os.environ.get("RPC_URL", "http://192.168.1.78:8332")
    bitcoin_rpc_user: str = os.environ.get("RPC_USER", "")
    bitcoin_rpc_password: str = os.environ.get("RPC_PASSWORD", "")


# ─── Persistence config ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionConfig:
    """Trade execution parameters."""
    # Limit order pricing: offset from best bid/ask
    # LOW urgency: deeper into book (better price, slower fill)
    # HIGH urgency: tighter to market (worse price, faster fill)
    spread_offset_low: float = 0.6    # 60% into spread from favorable side
    spread_offset_medium: float = 0.3  # 30% into spread
    spread_offset_high: float = 0.05   # 5% into spread (near market)

    # Order lifecycle
    order_ttl_seconds: float = 300.0   # Cancel unfilled orders after 5 min
    check_interval_seconds: float = 15.0  # Poll order status every 15s
    max_chase_attempts: int = 3         # Re-price up to 3 times before giving up

    # Price improvement: tick size for Kraken BTC/EUR
    tick_size: float = 0.1  # Kraken XXBTZEUR tick size

    # Slippage guard: max distance from mid-price
    max_slippage_pct: float = 0.005  # 0.5% max slippage from mid


@dataclass(frozen=True)
class PersistenceConfig:
    """File paths for state persistence."""
    base_dir: Path = field(default_factory=lambda: Path.home() / ".kraken_bot")
    state_file: str = "bot_state.json"
    trade_history_file: str = "trade_history.json"
    ath_file: str = "ath_tracker.json"
    reflexion_file: str = "reflexion_memory.json"
    parameter_evolution_file: str = "param_evolution.json"
    performance_file: str = "performance.json"

    def ensure_dirs(self) -> None:
        """Create persistence directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, filename: str) -> Path:
        """Get full path for a persistence file."""
        return self.base_dir / filename


# ─── Dynamic ATH Tracker ─────────────────────────────────────────────────────

class ATHTracker:
    """
    Single source of truth for All-Time High tracking.

    Persists to disk. Updates automatically. Never hardcoded.
    This exists as a class (not a frozen dataclass) because it's mutable state.
    """

    def __init__(self, persistence: PersistenceConfig) -> None:
        self._path = persistence.get_path(persistence.ath_file)
        self._ath_eur: float = 114000.0
        self._ath_timestamp: Optional[datetime] = None
        self._load()

    @property
    def ath_eur(self) -> float:
        """Current all-time high in EUR."""
        return self._ath_eur

    @property
    def ath_timestamp(self) -> Optional[datetime]:
        """When the ATH was recorded."""
        return self._ath_timestamp

    def update(self, price_eur: float) -> bool:
        """
        Update ATH if price exceeds current record.

        Returns True if a new ATH was set.
        """
        if price_eur > self._ath_eur:
            self._ath_eur = price_eur
            self._ath_timestamp = datetime.now(timezone.utc)
            self._save()
            logger.info(f"New ATH recorded: €{price_eur:,.2f}")
            return True
        return False

    def drawdown_from_ath(self, current_price: float) -> float:
        """Calculate current drawdown as a fraction (0.0 = at ATH, 1.0 = total loss)."""
        if self._ath_eur <= 0:
            return 0.0
        return max(0.0, 1.0 - (current_price / self._ath_eur))

    def _load(self) -> None:
        """Load ATH from persisted file."""
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._ath_eur = float(data.get("ath_eur", 0.0))
                ts = data.get("ath_timestamp")
                if ts:
                    self._ath_timestamp = datetime.fromisoformat(ts)
                logger.info(f"Loaded ATH: €{self._ath_eur:,.2f}")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(f"Failed to load ATH file, starting fresh: {exc}")
            self._ath_eur = 0.0
            self._ath_timestamp = None

    def _save(self) -> None:
        """Persist ATH to file."""
        try:
            data = {
                "ath_eur": self._ath_eur,
                "ath_timestamp": self._ath_timestamp.isoformat() if self._ath_timestamp else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._path.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            logger.error(f"Failed to persist ATH: {exc}")


# ─── Master config ───────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    """
    Master configuration — aggregates all sub-configs.

    Usage:
        config = BotConfig.load()
        # Access any parameter:
        config.kraken.pair
        config.cycle.halving_date
        config.risk.reserve_floor_pct
    """
    kraken: KrakenConfig = field(default_factory=KrakenConfig.from_env)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    # Runtime mode
    paper_trade: bool = False  # Start in paper mode, switch to live explicitly
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.persistence.ensure_dirs()

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> BotConfig:
        """
        Load config with optional JSON override file.

        Priority: JSON file overrides → env vars → defaults.
        """
        config = cls()

        if config_path and config_path.exists():
            try:
                overrides = json.loads(config_path.read_text())
                config = cls._apply_overrides(config, overrides)
                logger.info(f"Loaded config overrides from {config_path}")
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(f"Failed to load config overrides: {exc}")

        # Environment overrides for runtime settings
        if os.environ.get("BOT_PAPER_TRADE", "").lower() in ("false", "0", "no"):
            config.paper_trade = False

        log_level = os.environ.get("BOT_LOG_LEVEL", "")
        if log_level:
            config.log_level = log_level.upper()

        return config

    @staticmethod
    def _apply_overrides(config: BotConfig, overrides: dict) -> BotConfig:
        """Apply JSON overrides to config. Only overrides known fields."""
        # Flat override mapping for common tuning knobs
        field_map = {
            "fast_loop_seconds": ("timing", "fast_loop_seconds"),
            "slow_loop_seconds": ("timing", "slow_loop_seconds"),
            "paper_trade": (None, "paper_trade"),
            "log_level": (None, "log_level"),
            "buy_threshold": ("signal", "buy_threshold"),
            "sell_threshold": ("signal", "sell_threshold"),
            "min_agreement": ("signal", "min_agreement"),
            "buy_min_agreement": ("signal", "buy_min_agreement"),
            "sell_min_agreement": ("signal", "sell_min_agreement"),
            "reserve_floor_pct": ("risk", "reserve_floor_pct"),
            "max_daily_trades": ("risk", "max_daily_trades"),
            "base_fraction": ("sizing", "base_fraction"),
            "dca_floor_enabled": ("sizing", "dca_floor_enabled"),
            "dca_floor_interval_hours": ("sizing", "dca_floor_interval_hours"),
            "dca_floor_fraction": ("sizing", "dca_floor_fraction"),
            "value_avg_enabled": ("sizing", "value_avg_enabled"),
            "value_avg_max_boost": ("sizing", "value_avg_max_boost"),
            "acceleration_brake_enabled": ("sizing", "acceleration_brake_enabled"),
            "min_phase_dwell_cycles": ("cycle", "min_phase_dwell_cycles"),
            "phase_transition_confidence": ("cycle", "phase_transition_confidence"),
            "phase_transition_advantage": ("cycle", "phase_transition_advantage"),
        }

        for key, value in overrides.items():
            if key in field_map:
                section, attr = field_map[key]
                if section is None:
                    setattr(config, attr, value)
                else:
                    # Frozen dataclasses require reconstruction
                    sub = getattr(config, section)
                    sub_dict = asdict(sub)
                    sub_dict[attr] = value
                    setattr(config, section, type(sub)(**sub_dict))
                logger.info(f"Config override: {key} = {value}")
            else:
                logger.warning(f"Unknown config override key: {key}")

        return config

    def setup_logging(self) -> None:
        """Configure structured logging for the bot."""
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Quiet noisy libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
