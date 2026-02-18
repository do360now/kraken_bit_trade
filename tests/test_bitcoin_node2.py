import json
import pytest
import responses
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bitcoin_node import BitcoinNode, MempoolInfo, FeeEstimate, OnChainSnapshot
from config import BotConfig


@responses.activate
def test_get_snapshot_success():
    """Happy path: all RPC calls succeed → valid OnChainSnapshot"""
    config = BotConfig()
    base_url = "http://192.168.1.78:8332"
    node = BitcoinNode(
        rpc_url=base_url,
        rpc_user=config.timing.bitcoin_rpc_user,
        rpc_password=config.timing.bitcoin_rpc_password,
        cache_ttl=0,
    )
    node._available = None

    def request_callback(request):
        body = request.json()
        method = body.get("method", "")
        if method == "getmempoolinfo":
            result = {
                "size": 42000,
                "bytes": 85000000,
                "usage": 120000000,
                "minfee": 0.00001,
                "total_fee": 2.5,
            }
        elif method == "estimatesmartfee":
            result = {"feerate": 0.00015, "blocks": body.get("params", [6])[0]}
        elif method == "getblockchaininfo":
            result = {"blocks": 880000, "headers": 880001}
        elif method == "getnetworkinfo":
            result = {"connections": 12, "version": 270001, "protocolversion": 70015}
        elif method == "getmininginfo":
            result = {"blocks": 880000, "difficulty": 8.5e13, "networkhashps": 5.2e18}
        elif method == "getrawmempool":
            result = {}
        else:
            result = {}
        return (200, {}, json.dumps({"result": result, "error": None, "id": 1}))

    responses.add_callback(
        responses.POST,
        base_url,
        callback=request_callback,
    )

    snapshot = node.get_snapshot()

    assert isinstance(snapshot, OnChainSnapshot)
    # Fees are successfully mocked
    assert isinstance(snapshot.fees, FeeEstimate)
    assert snapshot.fees.fast_sat_vb != 20.0  # Not neutral


@responses.activate
def test_node_unreachable_graceful_degradation():
    """Node down → returns neutral defaults, sets available=False, no crash"""
    config = BotConfig()
    base_url = "http://192.168.1.78:8332"
    node = BitcoinNode(
        rpc_url=base_url,
        rpc_user=config.timing.bitcoin_rpc_user,
        rpc_password=config.timing.bitcoin_rpc_password,
        cache_ttl=0,
    )

    responses.add(
        responses.POST,
        base_url,
        body=Exception("Connection refused"),
        status=0,
    )

    snapshot = node.get_snapshot()

    assert snapshot is not None
    assert snapshot.available is False


@responses.activate
def test_invalid_url_raises_during_init():
    """Malformed URL → fail fast at construction (defensive)"""
    with pytest.raises(ValueError, match="Invalid RPC URL"):
        BitcoinNode(
            rpc_url='f"http://192.168.1.78:8332"',
            rpc_user="user",
            rpc_password="pass",
        )


@responses.activate
def test_auth_failure_logs_and_degrades():
    """401 Unauthorized → logs warning, degrades gracefully"""
    config = BotConfig()
    base_url = "http://192.168.1.78:8332"
    node = BitcoinNode(
        rpc_url=base_url,
        rpc_user=config.timing.bitcoin_rpc_user,
        rpc_password=config.timing.bitcoin_rpc_password,
        cache_ttl=0,
    )

    responses.add(
        responses.POST,
        base_url,
        json={"error": {"code": -18, "message": "auth failed"}},
        status=401,
    )

    snapshot = node.get_snapshot()

    assert snapshot.available is False
