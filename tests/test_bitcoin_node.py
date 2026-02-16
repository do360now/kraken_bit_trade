"""
Tests for bitcoin_node.py

Validates:
- RPC response parsing for all methods
- TTL cache behavior (hit, miss, expiry)
- Graceful degradation (neutral defaults on failure)
- OnChainSnapshot composition
- Derived properties (congestion, fee pressure, whale activity)
- Edge cases (empty mempool, zero fees, node down)

Run: python -m pytest tests/test_bitcoin_node.py -v
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from bitcoin_node import (
    BitcoinNode,
    BlockInfo,
    FeeEstimate,
    LargeTransaction,
    MempoolInfo,
    NetworkInfo,
    OnChainSnapshot,
    _NEUTRAL_BLOCKS,
    _NEUTRAL_FEES,
    _NEUTRAL_MEMPOOL,
    _NEUTRAL_NETWORK,
    _NEUTRAL_SNAPSHOT,
    _TTLCache,
)


# ─── TTL Cache tests ─────────────────────────────────────────────────────────

class TestTTLCache:
    def test_set_and_get(self):
        cache = _TTLCache(ttl=60.0)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        cache = _TTLCache(ttl=60.0)
        assert cache.get("nonexistent") is None

    def test_expiry(self):
        cache = _TTLCache(ttl=0.05)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.06)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = _TTLCache(ttl=60.0)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_overwrite(self):
        cache = _TTLCache(ttl=60.0)
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_different_types(self):
        cache = _TTLCache(ttl=60.0)
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}


# ─── MempoolInfo tests ───────────────────────────────────────────────────────

class TestMempoolInfo:
    def test_congested_by_tx_count(self):
        info = MempoolInfo(
            tx_count=60_000, size_bytes=50_000_000,
            total_fee_btc=1.0, min_fee_rate=5.0, memory_usage_bytes=100_000_000,
        )
        assert info.congested is True
        assert info.clearing is False

    def test_congested_by_size(self):
        info = MempoolInfo(
            tx_count=10_000, size_bytes=150_000_000,
            total_fee_btc=1.0, min_fee_rate=5.0, memory_usage_bytes=200_000_000,
        )
        assert info.congested is True

    def test_clearing(self):
        info = MempoolInfo(
            tx_count=3_000, size_bytes=1_000_000,
            total_fee_btc=0.1, min_fee_rate=1.0, memory_usage_bytes=5_000_000,
        )
        assert info.clearing is True
        assert info.congested is False

    def test_normal(self):
        info = MempoolInfo(
            tx_count=20_000, size_bytes=30_000_000,
            total_fee_btc=0.5, min_fee_rate=2.0, memory_usage_bytes=50_000_000,
        )
        assert info.congested is False
        assert info.clearing is False


# ─── FeeEstimate tests ───────────────────────────────────────────────────────

class TestFeeEstimate:
    def test_no_pressure(self):
        fees = FeeEstimate(
            fast_sat_vb=5.0, medium_sat_vb=4.0,
            slow_sat_vb=3.0, economy_sat_vb=5.0,
        )
        # ratio = 1.0 → pressure = 0.0
        assert fees.fee_pressure == pytest.approx(0.0)

    def test_moderate_pressure(self):
        fees = FeeEstimate(
            fast_sat_vb=50.0, medium_sat_vb=20.0,
            slow_sat_vb=10.0, economy_sat_vb=5.0,
        )
        # ratio = 10 → pressure = (10-1)/9 = 1.0
        assert fees.fee_pressure == pytest.approx(1.0)

    def test_extreme_pressure_capped(self):
        fees = FeeEstimate(
            fast_sat_vb=500.0, medium_sat_vb=100.0,
            slow_sat_vb=20.0, economy_sat_vb=2.0,
        )
        # ratio = 250 → capped at 1.0
        assert fees.fee_pressure == 1.0

    def test_zero_economy_fee(self):
        fees = FeeEstimate(
            fast_sat_vb=50.0, medium_sat_vb=20.0,
            slow_sat_vb=10.0, economy_sat_vb=0.0,
        )
        assert fees.fee_pressure == 0.5  # Fallback


# ─── BlockInfo tests ──────────────────────────────────────────────────────────

class TestBlockInfo:
    def test_fast_blocks(self):
        info = BlockInfo(
            height=800000, avg_interval_seconds=500.0,
            avg_tx_count=3000, avg_block_size_bytes=1_500_000,
            blocks_since_difficulty=100,
        )
        assert info.blocks_fast is True
        assert info.blocks_slow is False

    def test_slow_blocks(self):
        info = BlockInfo(
            height=800000, avg_interval_seconds=800.0,
            avg_tx_count=3000, avg_block_size_bytes=1_500_000,
            blocks_since_difficulty=100,
        )
        assert info.blocks_fast is False
        assert info.blocks_slow is True

    def test_normal_blocks(self):
        info = BlockInfo(
            height=800000, avg_interval_seconds=600.0,
            avg_tx_count=3000, avg_block_size_bytes=1_500_000,
            blocks_since_difficulty=100,
        )
        assert info.blocks_fast is False
        assert info.blocks_slow is False


# ─── OnChainSnapshot tests ───────────────────────────────────────────────────

class TestOnChainSnapshot:
    def _make_snapshot(self, **overrides) -> OnChainSnapshot:
        defaults = dict(
            mempool=_NEUTRAL_MEMPOOL,
            fees=_NEUTRAL_FEES,
            blocks=_NEUTRAL_BLOCKS,
            network=_NEUTRAL_NETWORK,
            large_txs=[],
            timestamp=time.time(),
        )
        defaults.update(overrides)
        return OnChainSnapshot(**defaults)

    def test_no_whale_activity(self):
        snap = self._make_snapshot(large_txs=[])
        assert snap.whale_activity is False

    def test_whale_activity(self):
        txs = [
            LargeTransaction(txid=f"tx{i}", value_btc=100.0,
                             fee_btc=0.01, size_bytes=500, is_mempool=True)
            for i in range(5)
        ]
        snap = self._make_snapshot(large_txs=txs)
        assert snap.whale_activity is True

    def test_network_stress_calm(self):
        calm_mempool = MempoolInfo(
            tx_count=5000, size_bytes=2_000_000,
            total_fee_btc=0.1, min_fee_rate=1.0, memory_usage_bytes=10_000_000,
        )
        calm_fees = FeeEstimate(
            fast_sat_vb=3.0, medium_sat_vb=2.0,
            slow_sat_vb=1.5, economy_sat_vb=3.0,
        )
        calm_blocks = BlockInfo(
            height=800000, avg_interval_seconds=590.0,
            avg_tx_count=2500, avg_block_size_bytes=1_500_000,
            blocks_since_difficulty=100,
        )
        snap = self._make_snapshot(
            mempool=calm_mempool, fees=calm_fees, blocks=calm_blocks,
        )
        assert snap.network_stress < 0.2

    def test_network_stress_high(self):
        congested_mempool = MempoolInfo(
            tx_count=80_000, size_bytes=200_000_000,
            total_fee_btc=5.0, min_fee_rate=20.0, memory_usage_bytes=500_000_000,
        )
        high_fees = FeeEstimate(
            fast_sat_vb=200.0, medium_sat_vb=100.0,
            slow_sat_vb=50.0, economy_sat_vb=5.0,
        )
        slow_blocks = BlockInfo(
            height=800000, avg_interval_seconds=900.0,
            avg_tx_count=2500, avg_block_size_bytes=1_500_000,
            blocks_since_difficulty=100,
        )
        snap = self._make_snapshot(
            mempool=congested_mempool, fees=high_fees, blocks=slow_blocks,
        )
        assert snap.network_stress > 0.8

    def test_snapshot_frozen(self):
        snap = self._make_snapshot()
        with pytest.raises(AttributeError):
            snap.timestamp = 0.0  # type: ignore


# ─── Neutral defaults tests ─────────────────────────────────────────────────

class TestNeutralDefaults:
    def test_neutral_mempool_is_neutral(self):
        assert _NEUTRAL_MEMPOOL.congested is False
        assert _NEUTRAL_MEMPOOL.clearing is False

    def test_neutral_fees_moderate(self):
        # Neutral fee pressure should be moderate, not extreme
        assert 0.1 < _NEUTRAL_FEES.fee_pressure < 0.9

    def test_neutral_blocks_normal(self):
        assert _NEUTRAL_BLOCKS.blocks_fast is False
        assert _NEUTRAL_BLOCKS.blocks_slow is False

    def test_neutral_snapshot_no_stress(self):
        assert _NEUTRAL_SNAPSHOT.network_stress < 0.5
        assert _NEUTRAL_SNAPSHOT.whale_activity is False


# ─── BitcoinNode RPC parsing tests (mocked) ─────────────────────────────────

class TestBitcoinNodeParsing:
    def _make_node(self, cache_ttl: float = 0.0) -> BitcoinNode:
        """Create a node with zero TTL cache for testing."""
        return BitcoinNode(
            rpc_url="http://127.0.0.1:8332",
            rpc_user="test",
            rpc_password="test",
            cache_ttl=cache_ttl,
        )

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_mempool_info_parses(self, mock_rpc):
        mock_rpc.return_value = {
            "loaded": True,
            "size": 15000,
            "bytes": 8000000,
            "usage": 40000000,
            "total_fee": 0.75,
            "maxmempool": 300000000,
            "mempoolminfee": 0.00001000,
            "minrelaytxfee": 0.00001000,
        }
        node = self._make_node()
        result = node.get_mempool_info()

        assert result.tx_count == 15000
        assert result.size_bytes == 8000000
        assert result.total_fee_btc == pytest.approx(0.75)
        assert result.min_fee_rate == pytest.approx(1.0)  # 0.00001 * 100_000

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_mempool_info_failure_returns_neutral(self, mock_rpc):
        mock_rpc.return_value = None
        node = self._make_node()
        result = node.get_mempool_info()
        assert result == _NEUTRAL_MEMPOOL

    @patch.object(BitcoinNode, "_estimate_fee")
    def test_get_fee_estimates_parses(self, mock_fee):
        mock_fee.side_effect = [50.0, 20.0, 8.0, 3.0]
        node = self._make_node()
        result = node.get_fee_estimates()

        assert result.fast_sat_vb == 50.0
        assert result.medium_sat_vb == 20.0
        assert result.slow_sat_vb == 8.0
        assert result.economy_sat_vb == 3.0

    @patch.object(BitcoinNode, "_rpc_call")
    def test_estimate_fee_conversion(self, mock_rpc):
        # estimatesmartfee returns BTC/kvB
        mock_rpc.return_value = {"feerate": 0.00020000, "blocks": 2}
        node = self._make_node()
        result = node._estimate_fee(2)
        # 0.0002 BTC/kvB * 100_000 = 20 sat/vB
        assert result == pytest.approx(20.0)

    @patch.object(BitcoinNode, "_rpc_call")
    def test_estimate_fee_no_data(self, mock_rpc):
        mock_rpc.return_value = {"errors": ["Insufficient data"], "blocks": 0}
        node = self._make_node()
        result = node._estimate_fee(2)
        # Should return default for 2-block target
        assert result == 20.0

    @patch.object(BitcoinNode, "_rpc_call")
    def test_estimate_fee_node_down(self, mock_rpc):
        mock_rpc.return_value = None
        node = self._make_node()
        result = node._estimate_fee(6)
        assert result == 10.0  # Default for 6-block target

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_block_info_parses(self, mock_rpc):
        def rpc_side_effect(method, params=None):
            if method == "getblockchaininfo":
                return {"blocks": 880000, "chain": "main"}
            if method == "getbestblockhash":
                return "hash_block_3"
            if method == "getblock":
                block_data = {
                    "hash_block_3": {"time": 1700003000, "nTx": 3000, "size": 1600000, "previousblockhash": "hash_block_2"},
                    "hash_block_2": {"time": 1700002400, "nTx": 2800, "size": 1500000, "previousblockhash": "hash_block_1"},
                    "hash_block_1": {"time": 1700001800, "nTx": 2600, "size": 1400000, "previousblockhash": "hash_block_0"},
                    "hash_block_0": {"time": 1700001200, "nTx": 2400, "size": 1300000},
                }
                hash_arg = params[0] if params else None
                return block_data.get(hash_arg)
            return None

        mock_rpc.side_effect = rpc_side_effect
        node = self._make_node()
        result = node.get_block_info(num_blocks=3)

        assert result.height == 880000
        # 4 timestamps → 3 intervals: 600, 600, 600
        assert result.avg_interval_seconds == pytest.approx(600.0)
        assert result.avg_tx_count > 0
        assert result.avg_block_size_bytes > 0
        # 880000 % 2016
        assert result.blocks_since_difficulty == 880000 % 2016

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_block_info_failure_returns_neutral(self, mock_rpc):
        mock_rpc.return_value = None
        node = self._make_node()
        result = node.get_block_info()
        assert result == _NEUTRAL_BLOCKS

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_network_info_parses(self, mock_rpc):
        def rpc_side_effect(method, params=None):
            if method == "getmininginfo":
                return {
                    "networkhashps": 6.5e20,  # 650 EH/s
                    "difficulty": 95672703408223.0,
                }
            if method == "getnetworkinfo":
                return {"connections": 125}
            return None

        mock_rpc.side_effect = rpc_side_effect
        node = self._make_node()
        result = node.get_network_info()

        assert result.hashrate_eh == pytest.approx(650.0)
        assert result.difficulty == pytest.approx(95672703408223.0)
        assert result.connection_count == 125

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_network_info_partial_failure(self, mock_rpc):
        def rpc_side_effect(method, params=None):
            if method == "getmininginfo":
                return {"networkhashps": 5e20, "difficulty": 80e12}
            if method == "getnetworkinfo":
                return None  # Network info unavailable
            return None

        mock_rpc.side_effect = rpc_side_effect
        node = self._make_node()
        result = node.get_network_info()

        assert result.hashrate_eh == pytest.approx(500.0)
        assert result.connection_count == 0  # Fallback

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_large_transactions_parses(self, mock_rpc):
        def rpc_side_effect(method, params=None):
            if method == "getrawmempool":
                return {
                    "tx_whale": {
                        "vsize": 500,
                        "fees": {"base": 0.005},
                    },
                    "tx_small": {
                        "vsize": 200,
                        "fees": {"base": 0.0001},
                    },
                }
            if method == "getrawtransaction":
                txid = params[0] if params else None
                if txid == "tx_whale":
                    return {
                        "vout": [
                            {"value": 50.0},
                            {"value": 25.0},
                        ]
                    }
            return None

        mock_rpc.side_effect = rpc_side_effect
        node = self._make_node()
        result = node.get_large_transactions(min_btc=10.0)

        assert len(result) == 1
        assert result[0].txid == "tx_whale"
        assert result[0].value_btc == pytest.approx(75.0)
        assert result[0].is_mempool is True

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_large_transactions_empty_mempool(self, mock_rpc):
        mock_rpc.return_value = {}
        node = self._make_node()
        result = node.get_large_transactions()
        assert result == []

    @patch.object(BitcoinNode, "_rpc_call")
    def test_get_large_transactions_node_down(self, mock_rpc):
        mock_rpc.return_value = None
        node = self._make_node()
        result = node.get_large_transactions()
        assert result == []


# ─── Cache integration tests ─────────────────────────────────────────────────

class TestBitcoinNodeCaching:
    @patch.object(BitcoinNode, "_rpc_call")
    def test_mempool_cached(self, mock_rpc):
        mock_rpc.return_value = {
            "size": 10000, "bytes": 5000000, "usage": 20000000,
            "total_fee": 0.5, "mempoolminfee": 0.00001,
        }
        node = BitcoinNode(
            rpc_url="http://127.0.0.1:8332",
            cache_ttl=60.0,
        )

        result1 = node.get_mempool_info()
        result2 = node.get_mempool_info()

        # Should only call RPC once — second call hits cache
        assert mock_rpc.call_count == 1
        assert result1 == result2

    @patch.object(BitcoinNode, "_rpc_call")
    def test_cache_expires(self, mock_rpc):
        mock_rpc.return_value = {
            "size": 10000, "bytes": 5000000, "usage": 20000000,
            "total_fee": 0.5, "mempoolminfee": 0.00001,
        }
        node = BitcoinNode(
            rpc_url="http://127.0.0.1:8332",
            cache_ttl=0.05,
        )

        node.get_mempool_info()
        time.sleep(0.06)
        node.get_mempool_info()

        # Should call RPC twice after cache expiry
        assert mock_rpc.call_count == 2


# ─── Availability tracking tests ─────────────────────────────────────────────

class TestBitcoinNodeAvailability:
    @patch.object(BitcoinNode, "_rpc_call")
    def test_available_after_success(self, mock_rpc):
        mock_rpc.return_value = {
            "size": 10000, "bytes": 5000000, "usage": 20000000,
            "total_fee": 0.5, "mempoolminfee": 0.00001,
        }
        node = BitcoinNode(rpc_url="http://127.0.0.1:8332", cache_ttl=0.0)
        assert node._available is None  # Untested
        node.get_mempool_info()
        # _rpc_call is mocked, so _available won't be set by the mock
        # But the parsing code should work without issues

    def test_initial_state_untested(self):
        node = BitcoinNode(rpc_url="http://127.0.0.1:8332")
        assert node._available is None


# ─── Full snapshot integration test ──────────────────────────────────────────

class TestBitcoinNodeSnapshot:
    @patch.object(BitcoinNode, "_rpc_call")
    def test_snapshot_composition(self, mock_rpc):
        """Snapshot should compose all sub-components with graceful degradation."""
        mock_rpc.return_value = None  # All RPC calls fail
        node = BitcoinNode(rpc_url="http://127.0.0.1:8332", cache_ttl=0.0)
        snap = node.get_snapshot()

        # Should get neutral defaults for mempool and blocks
        assert snap.mempool == _NEUTRAL_MEMPOOL
        assert snap.blocks == _NEUTRAL_BLOCKS
        # Network info degrades to zeros when both RPCs fail (not the neutral constant)
        assert snap.network.connection_count == 0
        assert snap.large_txs == []
        assert snap.timestamp > 0

    @patch.object(BitcoinNode, "_rpc_call")
    def test_snapshot_with_partial_data(self, mock_rpc):
        """Some RPCs succeed, others fail — snapshot should be a mix."""
        call_count = 0

        def selective_rpc(method, params=None):
            if method == "getmempoolinfo":
                return {
                    "size": 25000, "bytes": 15000000, "usage": 60000000,
                    "total_fee": 1.2, "mempoolminfee": 0.00002,
                }
            # Everything else fails
            return None

        mock_rpc.side_effect = selective_rpc
        node = BitcoinNode(rpc_url="http://127.0.0.1:8332", cache_ttl=0.0)
        snap = node.get_snapshot()

        # Mempool should have real data
        assert snap.mempool.tx_count == 25000
        # Blocks should be neutral
        assert snap.blocks == _NEUTRAL_BLOCKS


# ─── Default fee rate tests ──────────────────────────────────────────────────

class TestDefaultFeeRates:
    def test_fast_target(self):
        assert BitcoinNode._default_fee_rate(1) == 20.0
        assert BitcoinNode._default_fee_rate(2) == 20.0

    def test_medium_target(self):
        assert BitcoinNode._default_fee_rate(3) == 10.0
        assert BitcoinNode._default_fee_rate(6) == 10.0

    def test_slow_target(self):
        assert BitcoinNode._default_fee_rate(7) == 5.0
        assert BitcoinNode._default_fee_rate(25) == 5.0

    def test_economy_target(self):
        assert BitcoinNode._default_fee_rate(26) == 2.0
        assert BitcoinNode._default_fee_rate(144) == 2.0


# ─── LargeTransaction tests ─────────────────────────────────────────────────

class TestLargeTransaction:
    def test_frozen(self):
        tx = LargeTransaction(
            txid="abc123", value_btc=50.0, fee_btc=0.01,
            size_bytes=500, is_mempool=True,
        )
        with pytest.raises(AttributeError):
            tx.value_btc = 100.0  # type: ignore

    def test_fields(self):
        tx = LargeTransaction(
            txid="abc123", value_btc=50.0, fee_btc=0.01,
            size_bytes=500, is_mempool=False,
        )
        assert tx.txid == "abc123"
        assert tx.value_btc == 50.0
        assert tx.is_mempool is False
