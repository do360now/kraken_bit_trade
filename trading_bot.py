import time
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_moving_average, calculate_vwap, fetch_latest_news, calculate_sentiment, fetch_enhanced_news, calculate_enhanced_sentiment, calculate_risk_adjusted_indicators
from data_manager import DataManager
from order_manager import OrderManager
from trade_executor import TradeExecutor
from onchain_analyzer import OnChainAnalyzer
from config import GLOBAL_TRADE_COOLDOWN, BOT_LOGS_FILE
from logger_config import logger
from performance_tracker import PerformanceTracker
from metrics_server import MetricsServer
import requests
from market_data_service import MarketDataService

class TradingBot:
    def __init__(self, data_manager: DataManager, trade_executor: TradeExecutor, onchain_analyzer: OnChainAnalyzer, order_manager: OrderManager = None, market_data_service: MarketDataService = None):
        self.max_position_size = 0.15
        self.stop_loss_percentage = 0.03
        self.take_profit_percentage = 0.10
        self.max_daily_trades = 8
        self.daily_trade_count = 0
        self.data_manager = data_manager
        self.trade_executor = trade_executor
        self.onchain_analyzer = onchain_analyzer
        self.order_manager = order_manager
        self.market_data_service = market_data_service
        self.last_trade_time = 0
        self.max_cash_allocation = 0.9
        self.min_eur_for_trade = 5.0
        self.min_trade_volume = 0.00005
        self.recent_buys = []
        self._load_recent_buys()
        self.performance_tracker = PerformanceTracker(
            initial_btc_balance=self.trade_executor.get_total_btc_balance() or 0,
            initial_eur_balance=self.trade_executor.get_available_balance("EUR") or 0
        )
        self.lookback_period = 96  # hours
        self.price_history = []
        self._initialize_price_history()
        self.start_time = time.time()
        self.metrics_server = MetricsServer(self)
        self.metrics_server.start()
        self.ollama_url = "http://localhost:11434/api/generate"  # Assuming Ollama runs locally
        self.model_name = "gemma3:4b"
        self.last_rsi = 0
        self.last_sentiment = 0

    def _initialize_price_history(self):
        prices, _ = self.data_manager.load_price_history()
        if not prices:
            logger.warning("No price history available. Fetching initial data...")
            self._fetch_historical_data()
            prices, _ = self.data_manager.load_price_history()

        required_candles = self.lookback_period * 4  # 15-min candles
        current_time = int(time.time())
        required_time = current_time - (self.lookback_period * 3600)

        if prices:
            with open(self.data_manager.price_history_file, 'r') as f:
                data = json.load(f)
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and len(data[0]) > 0:
                earliest_timestamp = data[0][0]
                if earliest_timestamp > required_time:
                    logger.info(f"Price history only goes back to {datetime.fromtimestamp(earliest_timestamp).isoformat()}. Fetching more data...")
                    self._fetch_historical_data(since=earliest_timestamp - (self.lookback_period * 3600))
                    prices, _ = self.data_manager.load_price_history()

        self._load_price_history_for_lookback()

    def _fetch_historical_data(self, since: Optional[int] = None):
        if since is None:
            since = int(time.time()) - (self.lookback_period * 3600 * 2)  # 2x lookback period as buffer
        logger.info(f"Fetching historical OHLC data from {datetime.fromtimestamp(since).isoformat()}")
        ohlc = self.trade_executor.kraken_api.get_ohlc_data(pair="XXBTZEUR", interval=15, since=since)
        if ohlc:
            self.data_manager.append_ohlc_data(ohlc)

    def _load_price_history_for_lookback(self):
        try:
            prices, _ = self.data_manager.load_price_history()
            if prices:
                num_candles = self.lookback_period * 4
                self.price_history = [float(p) for p in prices[-num_candles:]]
                logger.info(f"Loaded {len(self.price_history)} prices into price_history for lookback")
                if self.price_history:
                    logger.info(f"Price history range: min={min(self.price_history):.2f}, max={max(self.price_history):.2f}")
                else:
                    logger.warning("Price history is empty after loading")
        except Exception as e:
            logger.error(f"Failed to load price history for lookback: {e}")

    def check_pending_orders(self):
        if not self.order_manager:
            return
        try:
            logger.debug("Checking pending orders...")
            results = self.order_manager.check_and_update_orders()
            if results['filled']:
                logger.info(f"🎯 Orders FILLED: {results['filled']}")
                for order_id in results['filled']:
                    order_info = self.order_manager.filled_orders.get(order_id)
                    if order_info:
                        self.performance_tracker.record_trade(
                            order_id=order_id,
                            side=order_info['side'],
                            volume=order_info['executed_volume'],
                            price=order_info['average_price'],
                            fee=order_info.get('fee', 0)
                        )
                        self.daily_trade_count += 1
                    if order_info and order_info['side'] == 'buy':
                        self.recent_buys.append((
                            order_info['average_price'],
                            order_info['executed_volume']
                        ))
                        self.recent_buys = self.recent_buys[-10:]
                        self._save_recent_buys()
                        logger.info(f"✅ Updated recent_buys with filled buy order {order_id}: {order_info['executed_volume']:.8f} BTC @ €{order_info['average_price']:.2f}")
                    elif order_info and order_info['side'] == 'sell':
                        avg_buy_price = self._estimate_avg_buy_price()
                        if avg_buy_price:
                            profit = (order_info['average_price'] - avg_buy_price) / avg_buy_price * 100
                            logger.info(f"💰 Sell order filled with {profit:.2f}% profit")
            if results['cancelled']:
                logger.info(f"❌ Orders CANCELLED (timeout): {results['cancelled']}")
            if results['partial']:
                logger.info(f"📊 Orders PARTIALLY filled: {results['partial']}")
                for order_id in results['partial']:
                    if order_id in self.order_manager.pending_orders:
                        order_info = self.order_manager.pending_orders[order_id]
                        logger.info(f"  {order_id}: {order_info.get('executed_volume', 0):.8f}/{order_info['volume']:.8f} BTC filled")
        except Exception as e:
            logger.error(f"Error checking pending orders: {e}", exc_info=True)

    def _detect_market_regime(self, prices: List[float]) -> str:
        if len(prices) < 50:
            return "unknown"
        ma_50 = calculate_moving_average(prices, 50)
        ma_200 = calculate_moving_average(prices, 200) if len(prices) >= 200 else ma_50
        if ma_50 > ma_200 * 1.02:
            return "strong_uptrend"
        elif ma_50 < ma_200 * 0.98:
            return "strong_downtrend"
        else:
            return "ranging"

    def _load_recent_buys(self):
        try:
            recent_buys_file = "recent_buys.json"
            if os.path.exists(recent_buys_file):
                with open(recent_buys_file, 'r') as f:
                    self.recent_buys = json.load(f)
                logger.debug(f"Loaded {len(self.recent_buys)} recent buys from {recent_buys_file}")
        except Exception as e:
            logger.error(f"Failed to load recent_buys: {e}")

    def _save_recent_buys(self):
        try:
            recent_buys_file = "recent_buys.json"
            with open(recent_buys_file, 'w') as f:
                json.dump(self.recent_buys, f)
            logger.debug(f"Saved {len(self.recent_buys)} recent buys to {recent_buys_file}")
        except Exception as e:
            logger.error(f"Failed to save recent_buys: {e}")

    def _calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns[-20:])) if len(returns) >= 20 else 0.01

    def _estimate_avg_buy_price(self) -> Optional[float]:
        try:
            if not os.path.exists(BOT_LOGS_FILE):
                logger.error(f"Bot logs file not found at {BOT_LOGS_FILE}")
                return self._fallback_avg_buy_price()
            logger.debug(f"Reading bot_logs.csv from {BOT_LOGS_FILE}")
            df = pd.read_csv(BOT_LOGS_FILE, dtype={"buy_decision": str, "sell_decision": str}, encoding='utf-8', on_bad_lines='warn')
            logger.debug(f"CSV shape: {df.shape}")
            expected_columns = ["timestamp", "price", "trade_volume", "side", "buy_decision"]
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"Missing required columns in bot_logs.csv: {set(expected_columns) - set(df.columns)}")
                return self._fallback_avg_buy_price()
            df['timestamp'] = df['timestamp'].astype(str)
            df['side'] = df['side'].astype(str)
            df['buy_decision'] = df['buy_decision'].astype(str)
            valid_rows = df[df['timestamp'].notna()]
            buy_trades = df[
                (df["buy_decision"].str.lower().isin(["true", "1"])) & 
                (df["side"].str.lower() == "buy")
            ]
            if buy_trades.empty:
                logger.debug("No valid buy trades found in logs")
                return self._fallback_avg_buy_price()
            buy_trades = buy_trades.copy()
            buy_trades["price"] = pd.to_numeric(buy_trades["price"], errors="coerce")
            buy_trades["trade_volume"] = pd.to_numeric(buy_trades["trade_volume"], errors="coerce")
            buy_trades = buy_trades.dropna(subset=["price", "trade_volume"])
            if buy_trades.empty:
                logger.debug("No valid buy trades after numeric conversion")
                return self._fallback_avg_buy_price()
            total_cost = (buy_trades["price"] * buy_trades["trade_volume"]).sum()
            total_volume = buy_trades["trade_volume"].sum()
            avg_price = total_cost / total_volume if total_volume > 0 else None
            logger.debug(f"Estimated avg buy price from logs: {avg_price}")
            return avg_price
        except Exception as e:
            logger.error(f"Failed to estimate avg buy price: {e}", exc_info=True)
            return self._fallback_avg_buy_price()

    def _fallback_avg_buy_price(self) -> Optional[float]:
        if not self.recent_buys:
            logger.debug("No recent buys available")
            return None
        total_cost = sum(price * volume for price, volume in self.recent_buys)
        total_volume = sum(volume for _, volume in self.recent_buys)
        avg_price = total_cost / total_volume if total_volume > 0 else None
        logger.debug(f"Fallback avg buy price from recent_buys: {avg_price}")
        return avg_price

    def enhanced_decide_action_with_risk_override(self, indicators_data: Dict) -> str:
        """
        IMPROVED: Bitcoin accumulation strategy with strict sell protection.
        Only sells when TRULY necessary or at significant profit.
        """
        # Extract all key indicators
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        sentiment = indicators_data.get('sentiment', 0)
        current_price = indicators_data.get('current_price', 0)
        avg_buy_price = indicators_data.get('avg_buy_price', 0)
        market_trend = indicators_data.get('market_trend', 'ranging')
        netflow = indicators_data.get('netflow', 0)
        rsi = indicators_data.get('rsi', 50)
        macd = indicators_data.get('macd', 0)
        signal = indicators_data.get('signal', 0)
        vwap = indicators_data.get('vwap', current_price)
        volatility = indicators_data.get('volatility', 0)
        ma_long = indicators_data.get('ma_long', current_price)
        
        # Performance metrics
        performance = indicators_data.get('performance_report', {})
        win_rate = float(performance.get('risk_metrics', {}).get('win_rate', '0%').rstrip('%')) / 100
        
        # Calculate profit/loss position
        if avg_buy_price and avg_buy_price > 0:
            profit_margin = (current_price - avg_buy_price) / avg_buy_price * 100
        else:
            profit_margin = 0
        
        # Calculate position relative to VWAP and long-term MA
        vwap_distance = (current_price - vwap) / vwap * 100
        ma_distance = (current_price - ma_long) / ma_long * 100 if ma_long > 0 else 0
        
        # BULL MARKET DETECTION (crucial for accumulation strategy)
        is_bull_market = (
            ma_distance > 5 or  # Price 5%+ above 50-day MA
            market_trend == "strong_uptrend"
        )
        
        logger.info(f"🔍 ANALYSIS - Price: €{current_price:.0f}, P&L: {profit_margin:.1f}%, "
                    f"Bull Market: {is_bull_market}, MA Distance: {ma_distance:+.1f}%, "
                    f"Risk-off: {risk_off_prob*100:.0f}%, RSI: {rsi:.1f}")
        
        # =============================================================================
        # EMERGENCY CONDITIONS ONLY - Very high bar for selling
        # =============================================================================
        
        # 1. EXTREME SYSTEMIC RISK (nearly certain crash)
        if risk_off_prob > 0.9 and sentiment < -0.3:  # Raised threshold to 90%
            logger.error(f"🚨 EXTREME SYSTEMIC RISK: Risk-off {risk_off_prob*100:.0f}% - EMERGENCY SELL")
            return 'sell'
        
        # 2. CONFIRMED LIQUIDATION CASCADE (rare)
        liquidation_cascade = (
            volatility > 0.12 and  # Very high volatility
            sentiment < -0.2 and
            current_price < vwap * 0.92 and  # 8%+ below VWAP
            rsi < 25  # Extremely oversold = panic
        )
        if liquidation_cascade:
            logger.error(f"🚨 LIQUIDATION CASCADE DETECTED - EMERGENCY SELL")
            return 'sell'
        
        # =============================================================================
        # HIGH PROFIT CONDITIONS - Take significant profits
        # =============================================================================
        
        # 3. MAJOR PROFIT TARGET (25%+ gains)
        if profit_margin > 25:
            # Even in bull markets, 25% is worth taking SOME profit
            logger.info(f"💰💰 MAJOR PROFITS: {profit_margin:.1f}% - PARTIAL PROFIT TAKING")
            return 'sell'
        
        # 4. STRONG PROFIT + EXTREME OVERBOUGHT (15%+ gains)
        if profit_margin > 15 and rsi > 80:
            # Price is extremely overbought AND we have good profits
            logger.info(f"💰 STRONG PROFITS + EXTREME OVERBOUGHT: {profit_margin:.1f}% - TAKE PROFITS")
            return 'sell'
        
        # 5. GOOD PROFIT + BEARISH SIGNALS (in non-bull markets)
        if not is_bull_market:  # DON'T sell in bull markets unless extreme
            if profit_margin > 12 and rsi > 75:
                logger.info(f"💰 GOOD PROFITS + OVERBOUGHT (no bull trend): {profit_margin:.1f}% - TAKE PROFITS")
                return 'sell'
        
        # =============================================================================
        # RISK MANAGEMENT - Be more cautious with high risk
        # =============================================================================
        
        # 6. VERY HIGH RISK + ALREADY PROFITABLE
        if risk_off_prob > 0.7 and profit_margin > 8:
            # Only sell if we have SOME profit and risk is VERY high
            logger.warning(f"⚠️ VERY HIGH RISK: {risk_off_prob*100:.0f}% + profitable - DEFENSIVE SELL")
            return 'sell'
        
        # 7. HIGH RISK - NO NEW POSITIONS (but don't sell existing)
        if risk_off_prob > 0.6:
            logger.warning(f"⚠️ HIGH RISK: {risk_off_prob*100:.0f}% - NO NEW POSITIONS")
            return 'hold'
        
        # =============================================================================
        # BUY CONDITIONS - Aggressive accumulation in dips
        # =============================================================================
        
        # 8. STRONG BUY SIGNALS (multiple confirmations needed)
        buy_signals = [
            rsi < 45,
            current_price < vwap * 0.98,  # Below VWAP
            netflow < -3000,  # Exchange outflow = accumulation
            sentiment > -0.1,  # Not extremely negative
            macd > signal or abs(macd - signal) < 5,
        ]
        
        buy_score = sum(buy_signals)
        
        # Bonus signals for aggressive buying
        if rsi < 30:  # Very oversold
            buy_score += 1
            logger.info(f"🔥 VERY OVERSOLD BOOST: RSI {rsi:.1f}")
        
        if netflow < -8000:  # Strong accumulation
            buy_score += 1
            logger.info(f"🐋 STRONG ACCUMULATION BOOST: Netflow {netflow:.0f}")
        
        # LOWER threshold in bull markets (accumulate more aggressively)
        required_buy_signals = 3 if is_bull_market else 4
        
        if buy_score >= required_buy_signals:
            logger.info(f"✅ BUY CONDITIONS MET: {buy_score}/{required_buy_signals} signals")
            return 'buy'
        
        # =============================================================================
        # TECHNICAL SELL CONDITIONS - Very strict
        # =============================================================================
        
        # 9. TECHNICAL SELL (need extreme conditions AND profitability)
        sell_signals = [
            rsi > 80,  # VERY overbought (not just 75)
            current_price > vwap * 1.08,  # 8% above VWAP (not just 5%)
            market_trend == 'strong_downtrend',
            sentiment < -0.15,  # Very negative
            macd < signal - 10,  # Strong bearish divergence
        ]
        
        sell_score = sum(sell_signals)
        
        # Need MORE signals AND decent profit AND not in bull market
        if sell_score >= 4 and profit_margin > 5 and not is_bull_market:
            logger.info(f"📉 TECHNICAL SELL: {sell_score}/5 signals + profitable")
            return 'sell'
        
        # =============================================================================
        # DEFAULT: HOLD (accumulation bias)
        # =============================================================================
        
        hold_reasons = []
        if buy_score < required_buy_signals:
            hold_reasons.append(f"insufficient buy signals ({buy_score}/{required_buy_signals})")
        if profit_margin < 12:
            hold_reasons.append(f"profits below threshold ({profit_margin:.1f}% < 12%)")
        if is_bull_market:
            hold_reasons.append("in bull market - accumulating")
        
        logger.info(f"⏸️ HOLD: {', '.join(hold_reasons) if hold_reasons else 'market neutral'}")
        return 'hold'

    def calculate_risk_adjusted_position_size(self, action: str, indicators_data: Dict, btc_balance: float, eur_balance: float) -> float:
        """
        IMPROVED: Much more conservative position sizing for Bitcoin accumulation.
        
        Key changes:
        - SELL: Max 25% per trade (not 80%!)
        - BUY: Larger positions in dips
        - Risk adjustment more conservative
        """
        if action == 'hold':
            return 0.0
        
        # BASE POSITION SIZES (more conservative for sells)
        base_buy_pct = 0.10  # 10% of EUR balance (slightly more aggressive for accumulation)
        base_sell_pct = 0.08  # 8% of BTC balance (much less than current 12%)
        
        # Risk adjustments
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        volatility = indicators_data.get('volatility', 0.02)
        profit_margin = ((indicators_data.get('current_price', 0) - indicators_data.get('avg_buy_price', 1)) 
                        / indicators_data.get('avg_buy_price', 1) * 100)
        
        # Performance adjustment
        performance = indicators_data.get('performance_report', {})
        win_rate = float(performance.get('risk_metrics', {}).get('win_rate', '0%').rstrip('%')) / 100
        
        # Calculate risk multiplier
        risk_multiplier = 1.0
        
        # Reduce size for high risk
        if risk_off_prob > 0.5:
            risk_multiplier *= (1 - risk_off_prob * 0.8)  # More conservative
        
        # Reduce size for high volatility
        if volatility > 0.06:
            risk_multiplier *= 0.6
        
        # Reduce size for poor performance
        if win_rate < 0.35:
            risk_multiplier *= 0.7
        
        # BUYING: Aggressive in good conditions
        if action == 'buy':
            rsi = indicators_data.get('rsi', 50)
            netflow = indicators_data.get('netflow', 0)
            
            # Increase buy size in great conditions
            if rsi < 30 and netflow < -8000 and risk_off_prob < 0.2:
                risk_multiplier *= 1.8  # Very aggressive in extreme dips
            elif rsi < 35 and netflow < -5000:
                risk_multiplier *= 1.4  # Aggressive in dips
            
            position_eur = eur_balance * base_buy_pct * risk_multiplier
            current_price = indicators_data.get('current_price', 1)
            position_btc = position_eur / current_price
            
            # Ensure we have enough EUR
            max_affordable = eur_balance * 0.9 / current_price
            position_btc = min(position_btc, max_affordable)
            position_btc = max(position_btc, self.min_trade_volume)
        
        # SELLING: Very conservative
        elif action == 'sell':
            position_btc = btc_balance * base_sell_pct * risk_multiplier
            
            # CRITICAL: Hard cap at 25% of holdings (NEVER sell 80%)
            max_sell_btc = btc_balance * 0.25
            position_btc = min(position_btc, max_sell_btc)
            
            # Scale up slightly if massive profits
            if profit_margin > 30:
                position_btc = min(position_btc * 1.5, max_sell_btc)
            
            # Ensure minimum trade size
            position_btc = max(position_btc, self.min_trade_volume)
            
            logger.info(f"🛡️ SELL SIZE PROTECTION: Selling max {position_btc:.8f} BTC "
                    f"({position_btc/btc_balance*100:.1f}% of holdings)")
        
        else:
            position_btc = 0.0
        
        logger.info(f"📏 POSITION SIZING: Action={action}, Risk multiplier={risk_multiplier:.2f}, "
                    f"Size={position_btc:.8f} BTC")
        
        return position_btc

    def decide_amount(self, action: str, indicators_data: Dict, btc_balance: float, eur_balance: float) -> float:
        """
        Calculate position size using the enhanced risk-adjusted method.
        """
        return self.calculate_risk_adjusted_position_size(action, indicators_data, btc_balance, eur_balance)

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ''
    
    def _calculate_technical_indicators(self, prices: List[float], volumes: List[float]) -> Dict:
        """Calculate all technical indicators from price/volume data."""
        rsi = calculate_rsi(prices) or 0
        macd, signal = calculate_macd(prices) or (0, 0)
        upper_band, ma_short, lower_band = calculate_bollinger_bands(prices) or (0, 0, 0)
        ma_long = calculate_moving_average(prices, 50) or 0
        vwap = calculate_vwap(prices, volumes) or (prices[-1] if prices else 0)
        volatility = self._calculate_volatility(prices)
        market_trend = self._detect_market_regime(prices)
        
        return {
            'rsi': rsi, 'macd': macd, 'signal': signal, 'ma_short': ma_short,
            'ma_long': ma_long, 'vwap': vwap, 'volatility': volatility,
            'market_trend': market_trend, 'upper_band': upper_band, 'lower_band': lower_band
        }
    
    def _fetch_news_and_sentiment(self) -> tuple:
        """Fetch news articles and calculate sentiment with error handling."""
        try:
            articles = fetch_enhanced_news(top_n=20)
            news_analysis = calculate_enhanced_sentiment(articles)
            sentiment = calculate_sentiment(articles)
            return articles, news_analysis, sentiment
        except Exception as e:
            logger.warning(f"News fetching failed, using neutral sentiment: {e}")
            return [], {"sentiment": 0.0, "risk_off_probability": 0.0, "macro_weight": 1.0}, 0.0
    
    def _get_balances_and_performance(self) -> tuple:
        """Get current balances and performance metrics."""
        btc_balance = self.trade_executor.get_total_btc_balance() or 0
        eur_balance = self.trade_executor.get_available_balance("EUR") or 0
        performance_report = self.performance_tracker.generate_performance_report()
        avg_buy_price = self._estimate_avg_buy_price()
        return btc_balance, eur_balance, performance_report, avg_buy_price
    
    def _build_indicators_dict(self, indicators: Dict, onchain: Dict, news: Dict,
                               current_price: float, sentiment: float, market_trend: str,
                               performance: Dict, avg_buy_price: float, price_history: List) -> Dict:
        """Build comprehensive indicators dictionary from all data sources."""
        dip_percentage = (indicators['ma_short'] - current_price) / indicators['ma_short'] if indicators['ma_short'] else 0
        peak_price = max(price_history) if price_history else current_price
        peak_dip_percentage = (peak_price - current_price) / peak_price if peak_price else 0
        
        return {
            'current_price': current_price,
            'news_analysis': news,
            'rsi': indicators.get('rsi', 50),
            'macd': indicators.get('macd', 0),
            'signal': indicators.get('signal', 0),
            'ma_short': indicators.get('ma_short', 0),
            'ma_long': indicators.get('ma_long', 0),
            'vwap': indicators.get('vwap', current_price),
            'volatility': indicators.get('volatility', 0),
            'sentiment': sentiment,
            'market_trend': market_trend,
            'fee_rate': onchain.get('fee_rate', 0),
            'netflow': onchain.get('netflow', 0),
            'onchain_volume': onchain.get('volume', 0),
            'old_utxos': onchain.get('old_utxos', 0),
            'dip_percentage': dip_percentage,
            'peak_dip_percentage': peak_dip_percentage,
            'performance_report': performance,
            'avg_buy_price': avg_buy_price
        }
    
    def execute_strategy(self):
        try:
            self.get_order_summary()
            prices, volumes = self.data_manager.load_price_history()
            if not prices or len(prices) < 10:
                logger.warning("Insufficient price data")
                return

            # Get current price from market data service
            price_obj = self.market_data_service.current_price()
            current_price = price_obj.value
            self.price_history.append(current_price)
            self.price_history = self.price_history[-self.lookback_period * 4:]

            ohlc = self.trade_executor.kraken_api.get_ohlc_data(pair="XXBTZEUR", interval=15, since=int(time.time() - 7200))
            self.data_manager.append_ohlc_data(ohlc)

            prices = [float(p) for p in prices]
            volumes = [float(v) for v in volumes]
            if len(prices) != len(volumes):
                logger.error("Mismatched price and volume lengths")
                return

            # Calculate basic technical indicators
            indicators = self._calculate_technical_indicators(prices, volumes)
            rsi, macd, signal = indicators['rsi'], indicators['macd'], indicators['signal']
            ma_short, ma_long = indicators['ma_short'], indicators['ma_long']
            vwap = indicators['vwap']
            volatility = indicators['volatility']
            market_trend = indicators['market_trend']

            # Enhanced news analysis (with timeout and fallback)
            articles, news_analysis, sentiment = self._fetch_news_and_sentiment()

            # Get on-chain signals
            onchain_signals = self.onchain_analyzer.get_onchain_signals()

            # Get balances and performance
            btc_balance, eur_balance, performance_report, avg_buy_price = self._get_balances_and_performance()
            self.performance_tracker.update_equity(btc_balance, eur_balance, current_price)

            # Combine all indicator data
            indicators_data = self._build_indicators_dict(
                indicators, onchain_signals, news_analysis, current_price, sentiment,
                market_trend, performance_report, avg_buy_price, self.price_history
            )

            logger.info(f"Indicators Data: Risk-off: {news_analysis.get('risk_off_probability', 0)*100:.0f}%, "
                        f"RSI: {indicators_data.get('rsi', 50):.1f}, "
                        f"Sentiment: {sentiment:.3f}, "
                        f"Netflow: {indicators_data.get('netflow', 0):.0f}")

            # Enhanced decision making with risk override
            action = self.enhanced_decide_action_with_risk_override(indicators_data)
            reason = f"{action.upper()} decided by enhanced risk system"
            logger.info(f"Decision: {action} - {reason}")

            # Calculate position size
            trade_volume = 0
            if action in ['buy', 'sell']:
                trade_volume = self.decide_amount(action, indicators_data, btc_balance, eur_balance)
                if trade_volume <= 0:
                    action = 'hold'
                    reason = "Zero volume calculated; holding"

            buy_decision = action == 'buy'
            sell_decision = action == 'sell'

            # Execute trades
            if buy_decision:
                if self.should_wait_for_pending_orders('buy'):
                    buy_decision = False
                    reason = "Waiting for pending buy orders to fill"
                    logger.info(reason)
                else:
                    order_book = self.trade_executor.get_btc_order_book()
                    optimal_price = self.trade_executor.get_optimal_price(order_book, "buy")
                    if optimal_price:
                        if self.order_manager:
                            order_id = self.order_manager.place_limit_order_with_timeout(
                                volume=trade_volume,
                                side="buy",
                                price=optimal_price,
                                timeout=300,
                                post_only=False
                            )
                            if order_id:
                                self.last_trade_time = time.time()
                                self._log_trade(datetime.now().isoformat(), optimal_price, trade_volume, "buy", reason)
                                logger.info(f"\033[32mBuy order placed: {trade_volume:.8f} BTC at €{optimal_price}, Order ID: {order_id}\033[0m")
                            else:
                                reason = "Buy failed: Could not place order"
                                buy_decision = False
                                logger.info(reason)
                        else:
                            if self.trade_executor.execute_trade(trade_volume, "buy", optimal_price):
                                self.last_trade_time = time.time()
                                self.recent_buys.append((optimal_price, trade_volume))
                                self.recent_buys = self.recent_buys[-10:]
                                self._save_recent_buys()
                                self._log_trade(datetime.now().isoformat(), optimal_price, trade_volume, "buy", reason)
                                logger.info(f"\033[32mBuy executed: {trade_volume:.8f} BTC at €{optimal_price}\033[0m")
                            else:
                                reason = "Buy failed: No response"
                                buy_decision = False
                                logger.info(reason)
                    else:
                        reason = "Buy failed: No optimal price"
                        buy_decision = False
                        logger.info(reason)

            elif sell_decision:
                if self.should_wait_for_pending_orders('sell'):
                    sell_decision = False
                    reason = "Waiting for pending sell orders to fill"
                    logger.info(reason)
                else:
                    order_book = self.trade_executor.get_btc_order_book()
                    optimal_price = self.trade_executor.get_optimal_price(order_book, "sell")
                    if optimal_price:
                        if self.order_manager:
                            order_id = self.order_manager.place_limit_order_with_timeout(
                                volume=trade_volume,
                                side="sell",
                                price=optimal_price,
                                timeout=300,
                                post_only=False
                            )
                            if order_id:
                                self.last_trade_time = time.time()
                                self._log_trade(datetime.now().isoformat(), optimal_price, trade_volume, "sell", reason)
                                logger.info(f"\033[31mSell order placed: {trade_volume:.8f} BTC at €{optimal_price}, Order ID: {order_id}\033[0m")
                            else:
                                reason = "Sell failed: Could not place order"
                                sell_decision = False
                                logger.info(reason)
                        else:
                            if self.trade_executor.execute_trade(trade_volume, "sell", optimal_price):
                                self.last_trade_time = time.time()
                                self._log_trade(datetime.now().isoformat(), optimal_price, trade_volume, "sell", reason)
                                logger.info(f"\033[31mSell executed: {trade_volume:.8f} BTC at €{optimal_price}\033[0m")
                            else:
                                reason = "Sell failed: No response"
                                sell_decision = False
                                logger.info(reason)
                    else:
                        reason = "Sell failed: No optimal price"
                        sell_decision = False
                        logger.info(reason)

            # Log risk decision for monitoring
            self.log_risk_decision(action, indicators_data, reason)

            # Log all strategy data
            profit_margin = (current_price - avg_buy_price) / avg_buy_price if avg_buy_price else None
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "price": current_price,
                "trade_volume": trade_volume,
                "side": "buy" if buy_decision else "sell" if sell_decision else "",
                "reason": reason,
                "dip": indicators_data.get('dip_percentage', 0),
                "rsi": indicators_data.get('rsi', 0),
                "macd": indicators_data.get('macd', 0),
                "signal": indicators_data.get('signal', 0),
                "ma_short": indicators_data.get('ma_short', 0),
                "ma_long": indicators_data.get('ma_long', 0),
                "upper_band": indicators_data.get('vwap', 0),  # Using vwap as proxy for bands
                "lower_band": indicators_data.get('vwap', 0),
                "sentiment": indicators_data.get('sentiment', 0),
                "fee_rate": indicators_data.get('fee_rate', 0),
                "netflow": indicators_data.get('netflow', 0),
                "volume": indicators_data.get('onchain_volume', 0),
                "old_utxos": indicators_data.get('old_utxos', 0),
                "buy_decision": buy_decision,
                "sell_decision": sell_decision,
                "btc_balance": btc_balance,
                "eur_balance": eur_balance,
                "avg_buy_price": indicators_data.get('avg_buy_price', 0),
                "profit_margin": profit_margin
            }
            self.data_manager.log_strategy(**log_data)
            self.last_rsi = rsi
            self.last_sentiment = sentiment

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}", exc_info=True)

    def _log_trade(self, timestamp: str, price: float, volume: float, side: str, reason: str):
        self.data_manager.log_strategy(
            timestamp=timestamp,
            price=price,
            trade_volume=volume,
            side=side,
            reason=reason,
            buy_decision=side == "buy",
            sell_decision=side == "sell"
        )

    def should_wait_for_pending_orders(self, side: str) -> bool:
        if not self.order_manager:
            return False
        pending = self.order_manager.get_pending_orders()
        same_side_pending = [o for o in pending.values() if o['side'] == side]
        if same_side_pending:
            logger.info(f"⏳ Waiting for {len(same_side_pending)} pending {side} order(s) to fill before placing new ones")
            return True
        return False

    def get_order_summary(self):
        if not self.order_manager:
            return
        try:
            pending = self.order_manager.get_pending_orders()
            if pending:
                logger.info(f"=== PENDING ORDERS ({len(pending)}) ===")
                total_pending_buy_volume = 0
                total_pending_sell_volume = 0
                for order_id, order_info in pending.items():
                    age = time.time() - order_info['timestamp']
                    remaining = order_info['timeout'] - age
                    logger.info(f"  {order_id}: {order_info['side'].upper()} {order_info['volume']:.8f} BTC @ €{order_info['price']:.2f} (expires in {remaining:.0f}s)")
                    if order_info['side'] == 'buy':
                        total_pending_buy_volume += order_info['volume']
                    else:
                        total_pending_sell_volume += order_info['volume']
                logger.info(f"  Total pending: BUY {total_pending_buy_volume:.8f} BTC, SELL {total_pending_sell_volume:.8f} BTC")
            recent_fills = self.order_manager.get_filled_orders(hours=24)
            if recent_fills:
                logger.info(f"=== RECENT FILLS (Last 24h: {len(recent_fills)}) ===")
                for order_id, order_info in list(recent_fills.items())[-5:]:
                    fill_time = datetime.fromtimestamp(order_info.get('filled_at', 0))
                    logger.info(f"  {order_id}: {order_info['side'].upper()} {order_info.get('executed_volume', 0):.8f} BTC @ €{order_info.get('average_price', 0):.2f} at {fill_time.strftime('%H:%M:%S')}")
            stats = self.order_manager.get_order_statistics()
            logger.info(f"=== ORDER STATISTICS ===")
            logger.info(f"  Fill rate: {stats['fill_rate']:.1%}")
            logger.info(f"  Avg time to fill: {stats['avg_time_to_fill']:.0f}s")
            logger.info(f"  Total fees paid: €{stats['total_fees_paid']:.2f}")
            logger.info(f"  Total orders: {stats['total_filled_orders']} filled, {stats['total_cancelled_orders']} cancelled")
        except Exception as e:
            logger.error(f"Error getting order summary: {e}")

    def log_risk_decision(self, action: str, indicators_data: Dict, reasoning: str):
        """
        Log detailed risk analysis for monitoring and backtesting.
        """
        import json
        from datetime import datetime
        
        risk_log_file = "./risk_decisions.json"
        
        # Extract key risk factors
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        sentiment = indicators_data.get('sentiment', 0)
        current_price = indicators_data.get('current_price', 0)
        avg_buy_price = indicators_data.get('avg_buy_price', 0)
        profit_margin = ((current_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price and avg_buy_price > 0 else 0
        
        risk_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "reasoning": reasoning,
            "market_data": {
                "price": current_price,
                "profit_margin": profit_margin,
                "rsi": indicators_data.get('rsi', 50),
                "market_trend": indicators_data.get('market_trend', 'unknown'),
                "vwap_distance": ((current_price - indicators_data.get('vwap', current_price)) 
                                / indicators_data.get('vwap', current_price) * 100)
            },
            "risk_factors": {
                "risk_off_probability": risk_off_prob,
                "sentiment": sentiment,
                "netflow": indicators_data.get('netflow', 0),
                "volatility": indicators_data.get('volatility', 0),
                "macro_articles": news_analysis.get('macro_articles', 0),
                "total_articles": news_analysis.get('total_articles', 0)
            },
            "performance": {
                "win_rate": indicators_data.get('performance_report', {}).get('risk_metrics', {}).get('win_rate', '0%'),
                "total_return": indicators_data.get('performance_report', {}).get('returns', {}).get('total', '0%')
            }
        }
        
        # Load existing log
        try:
            with open(risk_log_file, 'r') as f:
                risk_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            risk_log = []
        
        # Add new entry
        risk_log.append(risk_entry)
        
        # Keep only last 1000 entries
        risk_log = risk_log[-1000:]
        
        # Save updated log
        try:
            with open(risk_log_file, 'w') as f:
                json.dump(risk_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk log: {e}")

    def get_risk_summary(self) -> Dict:
        """
        Get summary of recent risk decisions for monitoring.
        """
        import json
        from datetime import datetime, timedelta
        
        risk_log_file = "./risk_decisions.json"
        
        try:
            with open(risk_log_file, 'r') as f:
                risk_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "No risk log available"}
        
        if not risk_log:
            return {"error": "Empty risk log"}
        
        # Analyze last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_decisions = [
            entry for entry in risk_log 
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        if not recent_decisions:
            return {"error": "No recent decisions"}
        
        # Calculate statistics
        total_decisions = len(recent_decisions)
        action_counts = {}
        risk_levels = []
        
        for decision in recent_decisions:
            action = decision['action']
            action_counts[action] = action_counts.get(action, 0) + 1
            risk_levels.append(decision['risk_factors']['risk_off_probability'])
        
        avg_risk_level = sum(risk_levels) / len(risk_levels) if risk_levels else 0
        max_risk_level = max(risk_levels) if risk_levels else 0
        
        return {
            "total_decisions_24h": total_decisions,
            "action_breakdown": action_counts,
            "avg_risk_off_probability": avg_risk_level,
            "max_risk_off_probability": max_risk_level,
            "latest_decision": recent_decisions[-1] if recent_decisions else None,
            "risk_status": (
                "HIGH RISK" if avg_risk_level > 0.6 else
                "MODERATE RISK" if avg_risk_level > 0.4 else
                "LOW RISK"
            )
        }

    def implement_dynamic_stop_loss(self, entry_price: float, side: str, indicators_data: Dict) -> float:
        """
        Implement dynamic stop-loss based on market conditions.
        """
        base_stop_pct = 0.03  # 3% base stop loss
        
        # Get risk factors
        news_analysis = indicators_data.get('news_analysis', {})
        liquidation_signals = indicators_data.get('liquidation_signals', {})
        volatility = indicators_data.get('volatility', 0.02)
        
        # Tighten stops during high risk periods
        risk_off_prob = news_analysis.get('risk_off_probability', 0)
        if risk_off_prob > 0.6:
            base_stop_pct = 0.02  # 2% stop loss
        elif risk_off_prob > 0.3:
            base_stop_pct = 0.025  # 2.5% stop loss
        
        # Adjust for liquidation risk
        cascade_prob = liquidation_signals.get('cascade_probability', 0)
        if cascade_prob > 0.5:
            base_stop_pct = 0.015  # Very tight 1.5% stop
        
        # Adjust for volatility to avoid whipsaws
        volatility_adjustment = min(2.0, 1 + volatility * 5)  # Up to 2x adjustment
        final_stop_pct = base_stop_pct * volatility_adjustment
        
        # Calculate stop price
        if side == 'buy':
            stop_price = entry_price * (1 - final_stop_pct)
        else:  # sell
            stop_price = entry_price * (1 + final_stop_pct)
        
        logger.info(f"Dynamic stop-loss - Entry: €{entry_price:.2f}, Stop: €{stop_price:.2f}, "
                    f"Distance: {final_stop_pct*100:.1f}%")
        
        return stop_price

    def check_risk_status(self):
        """
        Quick CLI command to check current risk status.
        Usage: Add this as a method and call it manually when needed.
        """
        risk_summary = self.get_risk_summary()
        
        print("\n=== RISK STATUS SUMMARY ===")
        print(f"Status: {risk_summary.get('risk_status', 'UNKNOWN')}")
        print(f"24h Decisions: {risk_summary.get('total_decisions_24h', 0)}")
        print(f"Actions: {risk_summary.get('action_breakdown', {})}")
        print(f"Avg Risk Level: {risk_summary.get('avg_risk_off_probability', 0)*100:.1f}%")
        print(f"Max Risk Level: {risk_summary.get('max_risk_off_probability', 0)*100:.1f}%")
        
        latest = risk_summary.get('latest_decision')
        if latest:
            print(f"\nLatest Decision: {latest['action'].upper()} at {latest['timestamp']}")
            print(f"Reasoning: {latest['reasoning']}")
        print("===========================\n")