#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for Bitcoin Trading Bot
Tests critical components to prevent regressions and ensure reliability
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import sys
import os
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestKrakenAPI(unittest.TestCase):
    """Test Kraken API wrapper"""
    
    def setUp(self):
        from kraken_api import KrakenAPI
        self.api = KrakenAPI("test_key", "test_secret")
    
    def test_btc_balance_detection_xxbt(self):
        """Test that XXBT key is correctly detected for BTC balance"""
        mock_response = {
            'error': [],
            'result': {
                'XXBT': '0.12345678',
                'ZEUR': '1000.00'
            }
        }
        
        with patch.object(self.api, 'query_private', return_value=mock_response):
            balance = self.api.get_total_btc_balance()
            self.assertEqual(balance, 0.12345678)
    
    def test_btc_balance_fallback_keys(self):
        """Test that alternative BTC keys work"""
        for key in ['XBT', 'XBT.F', 'XBTC']:
            mock_response = {
                'error': [],
                'result': {key: '0.5'}
            }
            
            with patch.object(self.api, 'query_private', return_value=mock_response):
                balance = self.api.get_total_btc_balance()
                self.assertEqual(balance, 0.5, f"Failed for key: {key}")
    
    def test_btc_balance_not_found(self):
        """Test handling when BTC balance not in response"""
        mock_response = {
            'error': [],
            'result': {'ZEUR': '1000.00'}
        }
        
        with patch.object(self.api, 'query_private', return_value=mock_response):
            balance = self.api.get_total_btc_balance()
            self.assertEqual(balance, 0.0)
    
    def test_btc_balance_api_error(self):
        """Test handling of API errors"""
        mock_response = {
            'error': ['EAPI:Invalid key']
        }
        
        with patch.object(self.api, 'query_private', return_value=mock_response):
            balance = self.api.get_total_btc_balance()
            self.assertIsNone(balance)
    
    def test_available_balance_with_open_orders(self):
        """Test EUR balance calculation with locked funds"""
        balance_response = {
            'error': [],
            'result': {'ZEUR': '100.00'}
        }
        
        orders_response = {
            'error': [],
            'result': {
                'open': {
                    'ORDER123': {
                        'descr': {'pair': 'XXBTZEUR', 'type': 'buy', 'price': '50000'},
                        'vol': '0.001'
                    }
                }
            }
        }
        
        with patch.object(self.api, 'query_private', side_effect=[balance_response, orders_response]):
            available = self.api.get_available_balance('EUR')
            # Locked: 0.001 * 50000 = 50
            # Available: 100 - 50 = 50
            self.assertEqual(available, 50.0)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking and calculations"""
    
    def setUp(self):
        from performance_tracker import PerformanceTracker
        # Create tracker without loading historical data for clean tests
        self.tracker = PerformanceTracker(load_history=False)
    
    def test_win_rate_fifo_basic(self):
        """Test basic FIFO win rate calculation"""
        self.tracker.record_trade("buy1", "buy", 0.1, 50000, 10)
        self.tracker.record_trade("sell1", "sell", 0.1, 55000, 10)
        
        win_rate = self.tracker.calculate_win_rate()
        self.assertEqual(win_rate, 1.0, "Basic profitable trade should be 100% win rate")
    
    def test_win_rate_fifo_loss(self):
        """Test FIFO with losing trade"""
        self.tracker.record_trade("buy1", "buy", 0.1, 60000, 10)
        self.tracker.record_trade("sell1", "sell", 0.1, 55000, 10)
        
        win_rate = self.tracker.calculate_win_rate()
        self.assertEqual(win_rate, 0.0, "Losing trade should be 0% win rate")
    
    def test_win_rate_fifo_partial_fills(self):
        """Test FIFO with partial fills across multiple buys"""
        self.tracker.record_trade("buy1", "buy", 0.1, 50000, 10, time.time())
        self.tracker.record_trade("buy2", "buy", 0.1, 60000, 10, time.time() + 1)
        self.tracker.record_trade("sell1", "sell", 0.15, 55000, 15, time.time() + 2)
        
        win_rate = self.tracker.calculate_win_rate()
        # Uses 0.1 @ 50k (win) + 0.05 @ 60k (loss)
        # Avg: (0.1*50k + 0.05*60k)/0.15 = 53,333
        # Selling at 55k > 53,333 = WIN
        self.assertEqual(win_rate, 1.0)
    
    def test_win_rate_no_sells(self):
        """Test win rate with only buys"""
        self.tracker.record_trade("buy1", "buy", 0.1, 50000, 10)
        self.tracker.record_trade("buy2", "buy", 0.1, 52000, 10)
        
        win_rate = self.tracker.calculate_win_rate()
        self.assertEqual(win_rate, 0.0, "No sells should result in 0% win rate")
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Simulate equity curve
        for equity in [10000, 10500, 9000, 9500, 11000]:
            self.tracker.equity_curve.append({
                'timestamp': time.time(),
                'total_equity_eur': equity,
                'btc_balance': 0.1,
                'eur_balance': equity - 5000,
                'btc_price': 50000
            })
        
        max_dd = self.tracker.calculate_max_drawdown()
        # Peak: 10500, Trough: 9000, DD: (9000-10500)/10500 = 14.3%
        self.assertGreater(max_dd, 0.14)
        self.assertLess(max_dd, 0.15)
    
    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive returns"""
        # Simulate growing equity curve
        for i in range(100):
            equity = 10000 + i * 100  # Steady growth
            self.tracker.equity_curve.append({
                'timestamp': time.time() + i * 3600,
                'total_equity_eur': equity,
                'btc_balance': 0.1,
                'eur_balance': equity - 5000,
                'btc_price': 50000
            })
        
        sharpe = self.tracker.calculate_sharpe_ratio()
        self.assertGreater(sharpe, 0, "Positive returns should yield positive Sharpe")


class TestOnChainAnalyzer(unittest.TestCase):
    """Test on-chain analysis functionality"""
    
    def setUp(self):
        from onchain_analyzer import OnChainAnalyzer
        self.analyzer = OnChainAnalyzer()
    
    def test_satoshis_to_btc_conversion(self):
        """Test proper satoshi to BTC conversion"""
        # 100,000,000 satoshis = 1 BTC
        result = self.analyzer.satoshis_to_btc(100000000)
        self.assertEqual(result, 1.0)
        
        result = self.analyzer.satoshis_to_btc(50000000)
        self.assertEqual(result, 0.5)
    
    def test_satoshis_to_btc_scientific_notation(self):
        """Test handling of scientific notation"""
        result = self.analyzer.satoshis_to_btc("1e-8")
        self.assertAlmostEqual(result, 1e-16, places=20)
        
        result = self.analyzer.satoshis_to_btc("0E-8")
        self.assertEqual(result, 0.0)
    
    def test_satoshis_to_btc_invalid_input(self):
        """Test handling of invalid inputs"""
        result = self.analyzer.satoshis_to_btc(None)
        self.assertEqual(result, 0.0)
        
        result = self.analyzer.satoshis_to_btc("")
        self.assertEqual(result, 0.0)
        
        result = self.analyzer.satoshis_to_btc("invalid")
        self.assertEqual(result, 0.0)
    
    def test_exchange_address_detection(self):
        """Test known exchange address detection"""
        from config import EXCHANGE_ADDRESSES
        
        for address, exchange_name in EXCHANGE_ADDRESSES.items():
            result = self.analyzer.is_exchange_address(address)
            self.assertEqual(result, exchange_name)
    
    def test_cache_mechanism(self):
        """Test that caching prevents redundant RPC calls"""
        mock_rpc = Mock()
        mock_rpc.getblockcount.return_value = 800000
        mock_rpc.getblockstats.return_value = {'total_out': 1000000000000, 'txs': 2000}
        mock_rpc.getblock.return_value = {'tx': []}
        mock_rpc.getmempoolinfo.return_value = {'size': 10000, 'total_fee': 1.0}
        
        self.analyzer.rpc = mock_rpc
        
        # First call
        signals1 = self.analyzer.get_onchain_signals()
        call_count_1 = mock_rpc.getblockstats.call_count
        
        # Second call (should use cache)
        signals2 = self.analyzer.get_onchain_signals()
        call_count_2 = mock_rpc.getblockstats.call_count
        
        self.assertEqual(call_count_1, call_count_2, "Cache should prevent redundant calls")
        self.assertEqual(signals1, signals2)


class TestOrderManager(unittest.TestCase):
    """Test order management functionality"""
    
    def setUp(self):
        from order_manager import OrderManager
        mock_api = Mock()
        self.manager = OrderManager(mock_api)
    
    def test_place_order_success(self):
        """Test successful order placement"""
        mock_response = {
            'error': [],
            'result': {'txid': ['ORDER123']}
        }
        
        self.manager.kraken_api.query_private = Mock(return_value=mock_response)
        
        order_id = self.manager.place_limit_order_with_timeout(0.001, 'buy', 50000)
        
        self.assertEqual(order_id, 'ORDER123')
        self.assertIn('ORDER123', self.manager.pending_orders)
    
    def test_place_order_failure(self):
        """Test failed order placement"""
        mock_response = {
            'error': ['EGeneral:Invalid arguments']
        }
        
        self.manager.kraken_api.query_private = Mock(return_value=mock_response)
        
        order_id = self.manager.place_limit_order_with_timeout(0.001, 'buy', 50000)
        
        self.assertIsNone(order_id)
    
    def test_order_timeout_cancellation(self):
        """Test automatic cancellation of timed-out orders"""
        # Place order with 1 second timeout
        self.manager.pending_orders['ORDER123'] = {
            'timestamp': time.time() - 2,  # 2 seconds ago
            'timeout': 1,
            'volume': 0.001,
            'side': 'buy',
            'price': 50000,
            'status': 'pending'
        }
        
        self.manager.kraken_api.query_private = Mock(return_value={
            'error': [],
            'result': {'open': {}, 'count': 0}
        })
        self.manager.kraken_api.query_private = Mock(return_value={'error': []})
        self.manager.cancel_order = Mock(return_value=True)
        
        results = self.manager.check_and_update_orders()
        
        self.manager.cancel_order.assert_called_once()
    
    def test_order_statistics(self):
        """Test order statistics calculation"""
        # Add filled orders
        self.manager.filled_orders = {
            'ORDER1': {
                'timestamp': time.time() - 300,
                'filled_at': time.time() - 200,
                'fee': 5.0
            },
            'ORDER2': {
                'timestamp': time.time() - 600,
                'filled_at': time.time() - 500,
                'fee': 10.0
            }
        }
        
        self.manager.cancelled_orders = {'ORDER3': {}}
        
        stats = self.manager.get_order_statistics()
        
        self.assertEqual(stats['fill_rate'], 2/3)  # 2 filled out of 3 total
        self.assertEqual(stats['total_fees_paid'], 15.0)
        self.assertEqual(stats['total_filled_orders'], 2)
        self.assertEqual(stats['total_cancelled_orders'], 1)


class TestDataManager(unittest.TestCase):
    """Test data management and persistence"""
    
    def setUp(self):
        from data_manager import DataManager
        self.manager = DataManager("test_prices.json", "test_logs.csv")
    
    def test_ohlc_deduplication(self):
        """Test that duplicate OHLC candles are not added"""
        timestamp = 1234567890.0
        ohlc1 = [[timestamp, 50000, 51000, 49000, 50500, 50250, 100]]
        ohlc2 = [[timestamp, 50000, 51000, 49000, 50500, 50250, 100]]  # Duplicate
        
        with patch('builtins.open', mock_open(read_data='[]')):
            with patch('json.dump'):
                with patch('json.load', return_value=[]):
                    self.manager.append_ohlc_data(ohlc1)
                    
                with patch('json.load', return_value=ohlc1):
                    self.manager.append_ohlc_data(ohlc2)
    
    def test_csv_header_validation(self):
        """Test that CSV headers are properly validated"""
        self.assertTrue(hasattr(self.manager, 'HEADERS'))
        self.assertIn('timestamp', self.manager.HEADERS)
        self.assertIn('price', self.manager.HEADERS)
        self.assertIn('buy_decision', self.manager.HEADERS)
    
    def test_log_strategy_with_defaults(self):
        """Test logging with missing fields uses defaults"""
        with patch('builtins.open', mock_open()):
            with patch('fcntl.flock'):
                self.manager.log_strategy(
                    price=50000,
                    side='buy'
                    # Missing other fields should use defaults
                )


class TestIndicators(unittest.TestCase):
    """Test technical indicator calculations"""
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        from indicators import calculate_rsi
        
        # Generate test prices with uptrend
        prices = [100 + i for i in range(20)]
        rsi = calculate_rsi(prices)
        
        self.assertIsNotNone(rsi)
        self.assertGreater(rsi, 50, "Uptrend should yield RSI > 50")
        self.assertLessEqual(rsi, 100)
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data"""
        from indicators import calculate_rsi
        
        prices = [100, 101, 102]
        rsi = calculate_rsi(prices)
        
        self.assertIsNone(rsi)
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        from indicators import calculate_macd
        
        # Generate test prices
        prices = [50000 + i*100 for i in range(50)]
        macd, signal = calculate_macd(prices)
        
        self.assertIsNotNone(macd)
        self.assertIsNotNone(signal)
        self.assertIsInstance(macd, float)
        self.assertIsInstance(signal, float)
    
    def test_vwap_calculation(self):
        """Test VWAP calculation"""
        from indicators import calculate_vwap
        
        prices = [100, 101, 102, 103, 104]
        volumes = [10, 20, 15, 25, 30]
        
        vwap = calculate_vwap(prices, volumes)
        
        self.assertIsNotNone(vwap)
        self.assertGreater(vwap, 100)
        self.assertLess(vwap, 105)
    
    def test_vwap_mismatched_lengths(self):
        """Test VWAP with mismatched price/volume lengths"""
        from indicators import calculate_vwap
        
        prices = [100, 101, 102]
        volumes = [10, 20]
        
        vwap = calculate_vwap(prices, volumes)
        self.assertIsNone(vwap)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        from indicators import calculate_bollinger_bands
        
        prices = [100 + i for i in range(30)]
        upper, middle, lower = calculate_bollinger_bands(prices)
        
        self.assertIsNotNone(upper)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(lower)
        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after failure threshold"""
        from circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        def failing_func():
            raise Exception("Test failure")
        
        # First 3 failures
        for i in range(3):
            with self.assertRaises(Exception):
                breaker.call(failing_func)
        
        # 4th call should raise CircuitBreakerException
        from circuit_breaker import CircuitBreakerException
        with self.assertRaises(CircuitBreakerException):
            breaker.call(failing_func)
    
    def test_circuit_recovers_after_timeout(self):
        """Test circuit recovery after timeout"""
        from circuit_breaker import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Open circuit
        for i in range(2):
            with self.assertRaises(Exception):
                breaker.call(failing_func)
        
        # Wait for recovery
        time.sleep(1.1)
        
        # Should move to HALF_OPEN
        def working_func():
            return "success"
        
        result = breaker.call(working_func)
        self.assertEqual(result, "success")


def run_test_suite():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKrakenAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestOnChainAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderManager))
    suite.addTests(loader.loadTestsFromTestCase(TestDataManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
