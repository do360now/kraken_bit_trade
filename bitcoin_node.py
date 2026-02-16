"""
Local Bitcoin node RPC client.

Wraps bitcoin-cli / JSON-RPC to extract on-chain metrics that most retail
traders don't have access to: mempool pressure, fee rate trends, block
intervals, large transaction detection, and network hashrate.

Design:
- Results cached with configurable TTL (on-chain data doesn't change fast).
- Graceful degradation: if the node is unreachable, return neutral defaults,
  log a warning, and let the bot continue trading on price data alone.
- All public methods return typed dataclasses, never raw dicts.
- Never raises exceptions past the public interface.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ─── Response dataclasses ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MempoolInfo:
    """Bitcoin mempool state."""
    tx_count: int              # Number of unconfirmed transactions
    size_bytes: int            # Total size in bytes
    total_fee_btc: float       # Total fees in BTC
    min_fee_rate: float        # Minimum fee rate in sat/vB to enter mempool
    memory_usage_bytes: int    # RAM usage of mempool

    @property
    def congested(self) -> bool:
        """Mempool is congested if >50k transactions or >100MB."""
        return self.tx_count > 50_000 or self.size_bytes > 100_000_000

    @property
    def clearing(self) -> bool:
        """Mempool is clearing if <5k transactions."""
        return self.tx_count < 5_000


@dataclass(frozen=True)
class FeeEstimate:
    """Fee rate estimates for different confirmation targets."""
    fast_sat_vb: float         # ~1-2 blocks (high priority)
    medium_sat_vb: float       # ~6 blocks
    slow_sat_vb: float         # ~25 blocks
    economy_sat_vb: float      # ~144 blocks (next day)

    @property
    def fee_pressure(self) -> float:
        """
        Fee pressure indicator: 0.0 (no pressure) to 1.0 (extreme).

        Based on ratio of fast to economy fee rate. When they converge,
        pressure is low. When fast is many multiples of economy, pressure
        is high.
        """
        if self.economy_sat_vb <= 0:
            return 0.5
        ratio = self.fast_sat_vb / self.economy_sat_vb
        # ratio of 1 = no pressure, ratio of 10+ = extreme
        return min(1.0, max(0.0, (ratio - 1.0) / 9.0))


@dataclass(frozen=True)
class BlockInfo:
    """Recent block statistics."""
    height: int
    avg_interval_seconds: float   # Average interval over last N blocks
    avg_tx_count: int             # Average transactions per block
    avg_block_size_bytes: int     # Average block size
    blocks_since_difficulty: int  # Blocks since last difficulty adjustment

    @property
    def blocks_fast(self) -> bool:
        """Blocks coming faster than expected 600s average."""
        return self.avg_interval_seconds < 540  # <9 min

    @property
    def blocks_slow(self) -> bool:
        """Blocks coming slower than expected."""
        return self.avg_interval_seconds > 720  # >12 min


@dataclass(frozen=True)
class LargeTransaction:
    """A large on-chain transaction detected in mempool or recent blocks."""
    txid: str
    value_btc: float
    fee_btc: float
    size_bytes: int
    is_mempool: bool    # True if in mempool, False if in recent block


@dataclass(frozen=True)
class NetworkInfo:
    """Bitcoin network summary."""
    hashrate_eh: float         # Network hashrate in EH/s
    difficulty: float          # Current difficulty
    connection_count: int      # Number of peer connections


@dataclass(frozen=True)
class OnChainSnapshot:
    """
    Bundles all on-chain metrics for the signal engine's context cache.

    This is the interface between the Bitcoin node and the rest of the system.
    """
    mempool: MempoolInfo
    fees: FeeEstimate
    blocks: BlockInfo
    network: NetworkInfo
    large_txs: list[LargeTransaction]
    timestamp: float

    @property
    def whale_activity(self) -> bool:
        """Significant large transaction activity detected."""
        return len(self.large_txs) >= 3

    @property
    def network_stress(self) -> float:
        """
        Combined network stress indicator: 0.0 (calm) to 1.0 (stressed).

        Factors: mempool congestion, fee pressure, slow blocks.
        """
        stress = 0.0
        if self.mempool.congested:
            stress += 0.4
        stress += self.fees.fee_pressure * 0.4
        if self.blocks.blocks_slow:
            stress += 0.2
        return min(1.0, stress)


# ─── Neutral defaults (returned when node is unreachable) ────────────────────

_NEUTRAL_MEMPOOL = MempoolInfo(
    tx_count=10_000, size_bytes=5_000_000, total_fee_btc=0.5,
    min_fee_rate=1.0, memory_usage_bytes=50_000_000,
)

_NEUTRAL_FEES = FeeEstimate(
    fast_sat_vb=10.0, medium_sat_vb=6.0,
    slow_sat_vb=4.0, economy_sat_vb=3.0,
)

_NEUTRAL_BLOCKS = BlockInfo(
    height=0, avg_interval_seconds=600.0,
    avg_tx_count=2500, avg_block_size_bytes=1_500_000,
    blocks_since_difficulty=0,
)

_NEUTRAL_NETWORK = NetworkInfo(
    hashrate_eh=500.0, difficulty=0.0, connection_count=0,
)

_NEUTRAL_SNAPSHOT = OnChainSnapshot(
    mempool=_NEUTRAL_MEMPOOL,
    fees=_NEUTRAL_FEES,
    blocks=_NEUTRAL_BLOCKS,
    network=_NEUTRAL_NETWORK,
    large_txs=[],
    timestamp=0.0,
)


# ─── TTL Cache ────────────────────────────────────────────────────────────────

class _TTLCache:
    """Simple per-key TTL cache for RPC results."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str) -> Optional[object]:
        """Return cached value if not expired, else None."""
        entry = self._store.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if time.monotonic() - cached_at > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: object) -> None:
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


# ─── Bitcoin Node Client ─────────────────────────────────────────────────────

class BitcoinNode:
    """
    Local Bitcoin full node RPC client.

    Connects via JSON-RPC to extract on-chain metrics for the trading bot.
    All results are cached and all failures degrade gracefully.

    Args:
        rpc_url: JSON-RPC endpoint (default: http://127.0.0.1:8332).
        rpc_user: RPC username.
        rpc_password: RPC password.
        cache_ttl: Cache TTL in seconds (default: 300 = 5 minutes).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8332",
        rpc_user: str = "",
        rpc_password: str = "",
        cache_ttl: float = 300.0,
        timeout: float = 30.0,
    ) -> None:
        self._rpc_url = rpc_url
        self._auth = (rpc_user, rpc_password) if rpc_user else None
        self._timeout = timeout
        self._cache = _TTLCache(ttl=cache_ttl)
        self._session = requests.Session()
        self._request_id = 0
        self._available: Optional[bool] = None  # Tri-state: None=untested

    @property
    def is_available(self) -> bool:
        """Whether the node was reachable on the last attempt."""
        return self._available is True

    # ─── Public interface ────────────────────────────────────────────────

    def get_mempool_info(self) -> MempoolInfo:
        """Fetch current mempool state."""
        cached = self._cache.get("mempool")
        if cached is not None:
            return cached  # type: ignore

        data = self._rpc_call("getmempoolinfo")
        if data is None:
            return _NEUTRAL_MEMPOOL

        try:
            result = MempoolInfo(
                tx_count=int(data.get("size", 0)),
                size_bytes=int(data.get("bytes", 0)),
                total_fee_btc=float(data.get("total_fee", 0.0)),
                min_fee_rate=float(data.get("mempoolminfee", 0.0)) * 100_000,  # BTC/kvB → sat/vB
                memory_usage_bytes=int(data.get("usage", 0)),
            )
            self._cache.set("mempool", result)
            return result
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to parse mempool info: {exc}")
            return _NEUTRAL_MEMPOOL

    def get_fee_estimates(self) -> FeeEstimate:
        """Fetch fee rate estimates for various confirmation targets."""
        cached = self._cache.get("fees")
        if cached is not None:
            return cached  # type: ignore

        # estimatesmartfee returns BTC/kvB — convert to sat/vB
        fast = self._estimate_fee(2)
        medium = self._estimate_fee(6)
        slow = self._estimate_fee(25)
        economy = self._estimate_fee(144)

        result = FeeEstimate(
            fast_sat_vb=fast,
            medium_sat_vb=medium,
            slow_sat_vb=slow,
            economy_sat_vb=economy,
        )
        self._cache.set("fees", result)
        return result

    def get_block_info(self, num_blocks: int = 6) -> BlockInfo:
        """
        Fetch recent block statistics.

        Computes average block interval, tx count, and size over the
        last `num_blocks` blocks.
        """
        cached = self._cache.get("blocks")
        if cached is not None:
            return cached  # type: ignore

        chain_info = self._rpc_call("getblockchaininfo")
        if chain_info is None:
            return _NEUTRAL_BLOCKS

        try:
            height = int(chain_info.get("blocks", 0))
        except (ValueError, TypeError):
            return _NEUTRAL_BLOCKS

        # Fetch the last N+1 block headers to compute N intervals
        timestamps: list[int] = []
        tx_counts: list[int] = []
        block_sizes: list[int] = []

        current_hash = self._rpc_call("getbestblockhash")
        if not isinstance(current_hash, str):
            return _NEUTRAL_BLOCKS

        for _ in range(num_blocks + 1):
            block = self._rpc_call("getblock", [current_hash])
            if block is None:
                break
            try:
                timestamps.append(int(block["time"]))
                tx_counts.append(int(block.get("nTx", 0)))
                block_sizes.append(int(block.get("size", 0)))
                prev_hash = block.get("previousblockhash")
                if prev_hash is None:
                    break
                current_hash = prev_hash
            except (KeyError, ValueError, TypeError):
                break

        if len(timestamps) < 2:
            return _NEUTRAL_BLOCKS

        # Timestamps are newest-first, reverse for interval calculation
        timestamps.reverse()
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

        # Difficulty epoch: blocks since last retarget (every 2016 blocks)
        blocks_since_difficulty = height % 2016

        result = BlockInfo(
            height=height,
            avg_interval_seconds=sum(intervals) / len(intervals) if intervals else 600.0,
            avg_tx_count=int(sum(tx_counts) / len(tx_counts)) if tx_counts else 2500,
            avg_block_size_bytes=int(sum(block_sizes) / len(block_sizes)) if block_sizes else 1_500_000,
            blocks_since_difficulty=blocks_since_difficulty,
        )
        self._cache.set("blocks", result)
        return result

    def get_network_info(self) -> NetworkInfo:
        """Fetch network hashrate, difficulty, and peer count."""
        cached = self._cache.get("network")
        if cached is not None:
            return cached  # type: ignore

        mining_info = self._rpc_call("getmininginfo")
        net_info = self._rpc_call("getnetworkinfo")

        hashrate_hps = 0.0
        difficulty = 0.0
        connections = 0

        if mining_info is not None:
            try:
                hashrate_hps = float(mining_info.get("networkhashps", 0))
                difficulty = float(mining_info.get("difficulty", 0))
            except (ValueError, TypeError):
                pass

        if net_info is not None:
            try:
                connections = int(net_info.get("connections", 0))
            except (ValueError, TypeError):
                pass

        # Convert H/s to EH/s
        hashrate_eh = hashrate_hps / 1e18

        result = NetworkInfo(
            hashrate_eh=hashrate_eh,
            difficulty=difficulty,
            connection_count=connections,
        )
        self._cache.set("network", result)
        return result

    def get_large_transactions(self, min_btc: float = 10.0, max_results: int = 20) -> list[LargeTransaction]:
        """
        Detect large transactions in the mempool.

        Scans the mempool for transactions with total output value >= min_btc.
        This is computationally expensive, so results are cached aggressively.

        Note: We scan raw mempool entries. For very large mempools (>100k txs),
        this may be slow. The cache ensures we only do it once per TTL.
        """
        cache_key = f"large_txs_{min_btc}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore

        # Get raw mempool with verbose=True for fee/size data
        raw_mempool = self._rpc_call("getrawmempool", [True])
        if raw_mempool is None or not isinstance(raw_mempool, dict):
            return []

        large_txs: list[LargeTransaction] = []

        # Sort by fee descending to find whale transactions
        entries = sorted(
            raw_mempool.items(),
            key=lambda x: float(x[1].get("fees", {}).get("base", 0)) if isinstance(x[1], dict) else 0,
            reverse=True,
        )

        # Limit scanning to top entries by fee (whales pay high fees)
        scan_limit = min(len(entries), 500)

        for txid, info in entries[:scan_limit]:
            if not isinstance(info, dict):
                continue

            try:
                fee_btc = float(info.get("fees", {}).get("base", 0))
                size_bytes = int(info.get("vsize", info.get("size", 0)))

                # Use the raw transaction to get actual value
                # This is expensive, so we use a heuristic first:
                # large fee → likely large value transaction
                # Minimum fee for a whale tx: ~0.001 BTC
                if fee_btc < 0.0005:
                    continue

                raw_tx = self._rpc_call("getrawtransaction", [txid, True])
                if raw_tx is None:
                    continue

                # Sum outputs
                total_value = 0.0
                for vout in raw_tx.get("vout", []):
                    total_value += float(vout.get("value", 0))

                if total_value >= min_btc:
                    large_txs.append(LargeTransaction(
                        txid=txid,
                        value_btc=total_value,
                        fee_btc=fee_btc,
                        size_bytes=size_bytes,
                        is_mempool=True,
                    ))

                    if len(large_txs) >= max_results:
                        break

            except (KeyError, ValueError, TypeError):
                continue

        self._cache.set(cache_key, large_txs)
        logger.info(f"Found {len(large_txs)} large mempool transactions (>={min_btc} BTC)")
        return large_txs

    def get_snapshot(self) -> OnChainSnapshot:
        """
        Build a complete on-chain snapshot for the signal engine.

        This is the main entry point — call this from the slow loop.
        Individual components that fail return neutral defaults.
        """
        return OnChainSnapshot(
            mempool=self.get_mempool_info(),
            fees=self.get_fee_estimates(),
            blocks=self.get_block_info(),
            network=self.get_network_info(),
            large_txs=self.get_large_transactions(),
            timestamp=time.time(),
        )

    # ─── Internal RPC mechanics ──────────────────────────────────────────

    def _rpc_call(self, method: str, params: Optional[list] = None) -> Optional[object]:
        """
        Execute a JSON-RPC call to the Bitcoin node.

        Returns the result field on success, None on any failure.
        Never raises exceptions.
        """
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or [],
        }

        try:
            response = self._session.post(
                self._rpc_url,
                json=payload,
                auth=self._auth,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            if self._available is not False:
                logger.warning(f"Bitcoin node unreachable at {self._rpc_url}")
            self._available = False
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Bitcoin node RPC timeout: {method}")
            self._available = False
            return None
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            if status == 401:
                logger.error("Bitcoin node RPC authentication failed")
            else:
                logger.warning(f"Bitcoin node RPC HTTP error {status}: {method}")
            self._available = False
            return None

        try:
            body = response.json()
        except ValueError:
            logger.warning(f"Bitcoin node RPC invalid JSON response: {method}")
            self._available = False
            return None

        error = body.get("error")
        if error is not None:
            code = error.get("code", "?")
            msg = error.get("message", "unknown")
            logger.warning(f"Bitcoin node RPC error [{code}]: {msg} (method: {method})")
            # Node is reachable but method failed — still mark as available
            self._available = True
            return None

        self._available = True
        return body.get("result")

    def _estimate_fee(self, conf_target: int) -> float:
        """
        Get fee estimate for a confirmation target.

        Returns fee rate in sat/vB. Falls back to a default if the
        node can't estimate (e.g., not enough data).
        """
        data = self._rpc_call("estimatesmartfee", [conf_target])
        if data is None:
            return self._default_fee_rate(conf_target)

        fee_rate_btc_kvb = data.get("feerate")
        if fee_rate_btc_kvb is None or fee_rate_btc_kvb <= 0:
            # Node returned an error or negative fee (insufficient data)
            errors = data.get("errors", [])
            if errors:
                logger.debug(f"Fee estimation for {conf_target} blocks: {errors}")
            return self._default_fee_rate(conf_target)

        # Convert BTC/kvB to sat/vB: multiply by 1e8 (sat/BTC), divide by 1000 (vB/kvB)
        return float(fee_rate_btc_kvb) * 100_000  # 1e8 / 1e3

    @staticmethod
    def _default_fee_rate(conf_target: int) -> float:
        """Reasonable default fee rates when node can't estimate."""
        if conf_target <= 2:
            return 20.0
        if conf_target <= 6:
            return 10.0
        if conf_target <= 25:
            return 5.0
        return 2.0

    def close(self) -> None:
        """Clean up the HTTP session."""
        self._session.close()
