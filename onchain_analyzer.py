# onchain_analyzer.py
import time
from typing import Dict
from bitcoinrpc.authproxy import AuthServiceProxy, JSONRPCException
from config import (
    RPC_HOST, RPC_PORT, RPC_USER, RPC_PASSWORD,
    MIN_BTC, EXCHANGE_ADDRESSES, ONCHAIN_CACHE_DURATION
)
from logger_config import logger

class OnChainAnalyzer:
    def __init__(self):
        self.rpc = None
        self.connect_rpc()
        self.onchain_cache = {"fee_rate": 0, "netflow": 0, "volume": 0, "old_utxos": 0}
        self.onchain_cache_time = 0
        self.onchain_cache_duration = ONCHAIN_CACHE_DURATION
        self.block_cache = {}
        self.mempool_cache = None
        self.mempool_cache_time = 0

    def connect_rpc(self):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                rpc_url = f"http://{RPC_USER}:{RPC_PASSWORD}@{RPC_HOST}:{RPC_PORT}"
                logger.debug(f"Connecting to node: {RPC_HOST}:{RPC_PORT}, attempt {attempt + 1}")
                self.rpc = AuthServiceProxy(rpc_url, timeout=30)
                self.rpc.getblockcount()  # Health check
                logger.debug("RPC connection established")
                return
            except Exception as e:
                logger.error(f"Node connection failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.warning("Max RPC connection attempts reached")
                    self.rpc = None

    def check_rpc_health(self):
        try:
            if not self.rpc:
                self.connect_rpc()
            self.rpc.getblockcount()
            return True
        except Exception as e:
            logger.debug(f"RPC health check failed, reconnecting: {e}")
            self.connect_rpc()
            return bool(self.rpc)

    def satoshis_to_btc(self, value):
        """
        Convert satoshis to BTC.
        
        Bitcoin uses satoshis as the base unit where:
        1 BTC = 100,000,000 satoshis
        
        Therefore: BTC = satoshis / 100,000,000
        
        Args:
            value: Satoshi amount (int, float, or string)
            
        Returns:
            float: BTC amount
            
        Examples:
            100000000 satoshis → 1.0 BTC
            50000000 satoshis → 0.5 BTC
            1 satoshi → 0.00000001 BTC
        """
        if value is None or value == "":
            return 0.0
        try:
            satoshis = float(value)
            # CRITICAL: Convert satoshis to BTC by dividing by 100 million
            btc = satoshis / 100_000_000.0
            return max(btc, 0.0)
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid value for BTC conversion: {value} (type: {type(value)}), error: {e}")
            return 0.0

    def is_exchange_address(self, address):
        return EXCHANGE_ADDRESSES.get(address, "Unknown")

    def get_utxo_age(self, txid, vout):
        try:
            if not self.check_rpc_health():
                return None
            tx = self.rpc.getrawtransaction(txid, True)
            block_height = tx.get("blockheight", 0)
            current_height = self.rpc.getblockcount()
            return (current_height - block_height) * 10 / 1440 if block_height else None
        except Exception as e:
            logger.debug(f"Failed to get UTXO age for {txid[:8]}: {e}")
            return None

    def get_onchain_signals(self) -> Dict[str, float]:
        """
        Fast on-chain analysis using getblockstats API (10-50x faster than full block parsing).
        Only fetches individual transactions for exchange flow analysis.
        """
        if not self.check_rpc_health():
            logger.warning("No node connection; using cached on-chain signals")
            return self.onchain_cache

        current_time = time.time()
        if (current_time - self.onchain_cache_time) < self.onchain_cache_duration:
            return self.onchain_cache

        retries = 3
        for attempt in range(retries):
            try:
                # Mempool fee rate
                if not self.mempool_cache or (current_time - self.mempool_cache_time) > 60:
                    mempool = self.rpc.getmempoolinfo()
                    fee_rate = float(mempool["total_fee"]) / mempool["size"] * 1e8 if mempool["size"] > 0 else 0
                    self.mempool_cache = {"fee_rate": fee_rate}
                    self.mempool_cache_time = current_time
                else:
                    fee_rate = self.mempool_cache["fee_rate"]

                # Use fast getblockstats for volume metrics (analyze last 6 blocks = ~1 hour)
                current_height = self.rpc.getblockcount()
                total_volume = 0
                
                # Use getblockstats which is MUCH faster than fetching full blocks
                for height in range(max(current_height - 5, 0), current_height + 1):
                    stats_key = f"stats_{height}"
                    if stats_key in self.block_cache:
                        stats = self.block_cache[stats_key]
                    else:
                        # getblockstats returns aggregate data without fetching all transactions
                        stats = self.rpc.getblockstats(height, ['total_out', 'txs'])
                        self.block_cache[stats_key] = stats
                        # Keep cache size manageable
                        if len(self.block_cache) > 12:
                            oldest_key = min(self.block_cache.keys())
                            self.block_cache.pop(oldest_key)
                    
                    # total_out is in satoshis, convert to BTC
                    block_volume = stats['total_out'] / 1e8
                    total_volume += block_volume
                    logger.debug(f"Block {height}: {block_volume:.2f} BTC volume, {stats['txs']} txs")

                # For exchange netflow, sample recent blocks more efficiently
                # Only fetch individual transactions for large volume monitoring
                netflow = 0
                old_utxos = 0
                
                # Sample only the most recent block for exchange flow (not all blocks)
                recent_block_hash = self.rpc.getblockhash(current_height)
                
                # Use verbosity=1 to get only txids (much faster than verbosity=2)
                if recent_block_hash not in self.block_cache:
                    recent_block = self.rpc.getblock(recent_block_hash, 1)
                    self.block_cache[recent_block_hash] = recent_block
                else:
                    recent_block = self.block_cache[recent_block_hash]
                
                # Sample only first 100 transactions for exchange flow analysis
                tx_sample = recent_block.get("tx", [])[1:101]  # Skip coinbase, sample 100 txs
                
                for txid in tx_sample:
                    try:
                        tx = self.rpc.getrawtransaction(txid, True)
                        
                        # Only check outputs for exchange addresses
                        for vout in tx.get("vout", []):
                            amount = self.satoshis_to_btc(vout.get("value", 0))
                            if amount >= MIN_BTC:
                                address = vout.get("scriptPubKey", {}).get("address", "")
                                exchange = self.is_exchange_address(address)
                                if exchange != "Unknown":
                                    # Inflow to exchange (potential selling pressure)
                                    netflow += amount
                        
                        # Check for old coin movements (only if netflow is significantly negative)
                        if netflow < -50:
                            for vin in tx.get("vin", [])[:5]:  # Check first 5 inputs only
                                prev_txid = vin.get("txid")
                                prev_vout = vin.get("vout")
                                if prev_txid and prev_vout is not None:
                                    age = self.get_utxo_age(prev_txid, prev_vout)
                                    if age and age > 365:
                                        old_utxos += 1
                    except Exception as tx_error:
                        logger.debug(f"Failed to process tx {txid[:8]}: {tx_error}")
                        continue

                # Mempool sampling (only if volume is low)
                if total_volume < 5000:
                    logger.debug("Low block volume; sampling mempool")
                    try:
                        mempool_txs = self.rpc.getrawmempool(False)  # Just get txids (fast)
                        
                        # Sample only 30 random transactions from mempool
                        import random
                        sample_size = min(30, len(mempool_txs))
                        sampled_txids = random.sample(list(mempool_txs), sample_size) if mempool_txs else []
                        
                        mempool_volume = 0
                        for txid in sampled_txids:
                            try:
                                tx = self.rpc.getrawtransaction(txid, True)
                                for vout in tx.get("vout", []):
                                    amount = self.satoshis_to_btc(vout.get("value", 0))
                                    mempool_volume += amount
                                    
                                    if amount >= MIN_BTC:
                                        address = vout.get("scriptPubKey", {}).get("address", "")
                                        exchange = self.is_exchange_address(address)
                                        if exchange != "Unknown":
                                            netflow += amount
                            except Exception as e:
                                logger.debug(f"Error processing mempool tx: {e}")
                                continue
                        
                        # Extrapolate mempool volume from sample
                        if sampled_txids:
                            total_mempool_txs = len(mempool_txs)
                            estimated_mempool_volume = (mempool_volume / sample_size) * total_mempool_txs
                            total_volume += estimated_mempool_volume
                            logger.debug(f"Mempool: sampled {sample_size} txs, estimated {estimated_mempool_volume:.2f} BTC")
                    except Exception as mempool_error:
                        logger.debug(f"Mempool sampling failed: {mempool_error}")

                signals = {
                    "fee_rate": fee_rate,
                    "netflow": netflow,
                    "volume": total_volume,
                    "old_utxos": old_utxos
                }
                self.onchain_cache = signals
                self.onchain_cache_time = current_time
                logger.info(f"On-chain signals: Fee={fee_rate:.2f} sat/vB, Netflow={netflow:.2f} BTC, "
                           f"Volume={total_volume:.2f} BTC, Old_UTXOs={old_utxos}")
                return signals
                
            except Exception as e:
                logger.error(f"On-chain query failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    self.connect_rpc()
                    time.sleep(2 ** attempt)
                else:
                    logger.warning("Max retries reached; using cached on-chain signals")
                    return self.onchain_cache