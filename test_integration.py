#!/usr/bin/env python3
"""
Integration Tests for Bitcoin Trading Bot
Tests full workflows and component interactions
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestFullTradingCycle(unittest.TestCase):
    """Test complete trading workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_api = Mock()
        self.mock_api.query_private = Mock()
        self.mock_api.query_public = Mock()
    
    def test_full_buy_order_lifecycle(self):
        """Test complete buy order: place → pending → filled"""
        from order_manager import OrderManager
        
        manager = OrderManager(self.mock_api)
        
        # 1. Place order
        self.mock_api.query_private.return_value = {
            'error': [],
            'result': {'txid': ['ORDER123']}
        }
        
        order_id = manager.place_limit_order_with_timeout(0.001, 'buy', 50000, timeout=300)
        self.assertEqual(order_id, 'ORDER123')
        self.assertIn('ORDER123', manager.pending_orders)
        
        # 2. Check status - order is open
        self.mock_api.query_private.return_value = {
            'error': [],
            'result': {
                'open': {
                    'ORDER123': {
                        'status': 'open',
                        'vol_exec': '0',
                        'descr': {'pair': 'XXBTZEUR', 'type': 'buy'}
                    }
                }
            }
        }
        
        results = manager.check_and_update_orders()
        self.assertEqual(len(results['filled']), 0)
        self.assertIn('ORDER123', manager.pending_orders)
        
        # 3. Order fills
        self.mock_api.query_private.side_effect = [
            {  # OpenOrders
                'error': [],
                'result': {'open': {}}
            },
            {  # TradesHistory
                'error': [],
                'result': {
                    'trades': {
                        'TRADE123': {
                            'ordertxid': 'ORDER123',
                            'vol': '0.001',
                            'price': '50000',
                            'fee': '25',
                            'time': time.time()
                        }
                    }
                }
            }
        ]
        
        results = manager.check_and_update_orders()
        self.assertIn('ORDER123', results['filled'])
        self.assertIn('ORDER123', manager.filled_orders)
        self.assertNotIn('ORDER123', manager.pending_orders)
    
    def test_full_sell_order_with_profit(self):
        """Test selling at profit after buy"""
        from order_manager import OrderManager
        from performance_tracker import PerformanceTracker
        
        manager = OrderManager(self.mock_api)
        tracker = PerformanceTracker()
        
        # 1. Record buy trade
        tracker.record_trade("BUY1", "buy", 0.1, 50000, 10, time.time())
        
        # 2. Place sell order
        self.mock_api.query_private.return_value = {
            'error': [],
            'result': {'txid': ['SELL1']}
        }
        
        order_id = manager.place_limit_order_with_timeout(0.1, 'sell', 55000, timeout=300)
        self.assertEqual(order_id, 'SELL1')
        
        # 3. Sell fills
        tracker.record_trade("SELL1", "sell", 0.1, 55000, 10, time.time() + 1)
        
        # 4. Verify profit
        win_rate = tracker.calculate_win_rate()
        self.assertEqual(win_rate, 1.0, "Sell at 55k after buy at 50k should be profitable")
    
    def test_order_timeout_and_retry(self):
        """Test that timed-out orders are cancelled and can be retried"""
        from order_manager import OrderManager
        
        manager = OrderManager(self.mock_api)
        
        # Place order with 1 second timeout
        self.mock_api.query_private.return_value = {
            'error': [],
            'result': {'txid': ['ORDER123']}
        }
        
        order_id = manager.place_limit_order_with_timeout(0.001, 'buy', 50000, timeout=1)
        
        # Simulate time passing
        manager.pending_orders[order_id]['timestamp'] = time.time() - 2
        
        # Mock responses for check_and_update
        self.mock_api.query_private.side_effect = [
            {'error': [], 'result': {'open': {}}},  # OpenOrders
            {'error': [], 'result': {'trades': {}}},  # TradesHistory
            {'error': []}  # CancelOrder
        ]
        
        results = manager.check_and_update_orders()
        
        # Verify order was cancelled
        self.assertIn(order_id, results['cancelled'])
        
        # Can place new order - clear side_effect and set new return_value
        self.mock_api.query_private.side_effect = None
        self.mock_api.query_private.return_value = {
            'error': [],
            'result': {'txid': ['ORDER456']}
        }
        new_order_id = manager.place_limit_order_with_timeout(0.001, 'buy', 49000, timeout=300)
        self.assertEqual(new_order_id, 'ORDER456')


class TestAPIResilience(unittest.TestCase):
    """Test handling of API failures and edge cases"""
    
    def test_balance_fetch_with_api_down(self):
        """Test balance fetching when API is down"""
        from kraken_api import KrakenAPI
        
        api = KrakenAPI("test", "test")
        api.query_private = Mock(return_value={'error': ['EService:Unavailable']})
        
        balance = api.get_total_btc_balance()
        self.assertIsNone(balance)
    
    def test_balance_fetch_with_timeout(self):
        """Test balance fetching with timeout"""
        from kraken_api import KrakenAPI
        import requests
        
        api = KrakenAPI("test", "test")
        
        with patch('kraken_api.requests.post', side_effect=requests.Timeout):
            try:
                # This should retry and eventually fail
                balance = api.get_total_btc_balance()
                # If it doesn't raise, it should return None or 0.0 (acceptable fallback)
                self.assertIn(balance, [None, 0.0], "Should return None or 0.0 on timeout")
            except requests.Timeout:
                pass  # Expected if retry doesn't catch it
    
    def test_order_placement_with_network_error(self):
        """Test order placement with network errors"""
        from order_manager import OrderManager
        
        api = Mock()
        api.query_private = Mock(side_effect=Exception("Network error"))
        
        manager = OrderManager(api)
        order_id = manager.place_limit_order_with_timeout(0.001, 'buy', 50000)
        
        self.assertIsNone(order_id)
        self.assertEqual(len(manager.pending_orders), 0)


class TestDataPersistence(unittest.TestCase):
    """Test data saving and loading"""
    
    def test_order_history_persistence(self):
        """Test that order history is properly saved and loaded"""
        from order_manager import OrderManager
        
        # Create manager and add orders
        api = Mock()
        manager = OrderManager(api)
        manager.order_history_file = "test_orders.json"
        
        manager.filled_orders = {
            'ORDER1': {
                'timestamp': time.time(),
                'filled_at': time.time(),
                'volume': 0.001,
                'side': 'buy',
                'price': 50000
            }
        }
        
        # Save
        manager._save_order_history()
        
        # Create new manager and load
        manager2 = OrderManager(api)
        manager2.order_history_file = "test_orders.json"
        manager2._load_order_history()
        
        self.assertIn('ORDER1', manager2.filled_orders)
        self.assertEqual(manager2.filled_orders['ORDER1']['price'], 50000)
        
        # Cleanup
        if os.path.exists("test_orders.json"):
            os.remove("test_orders.json")
    
    def test_performance_history_persistence(self):
        """Test that performance history is saved correctly"""
        from performance_tracker import PerformanceTracker
        
        # Use clean tracker without loading history to avoid test contamination
        tracker = PerformanceTracker(load_history=False)
        tracker.performance_file = "test_performance.json"
        
        # Add trades
        tracker.record_trade("TRADE1", "buy", 0.1, 50000, 10)
        tracker.record_trade("TRADE2", "sell", 0.1, 55000, 10)
        
        # Verify saved
        self.assertTrue(os.path.exists(tracker.performance_file))
        
        with open(tracker.performance_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('trades', data)
        self.assertEqual(len(data['trades']), 2)
        
        # Cleanup
        if os.path.exists(tracker.performance_file):
            os.remove(tracker.performance_file)


class TestIndicatorAccuracy(unittest.TestCase):
    """Test that technical indicators produce accurate results"""
    
    def test_rsi_extreme_values(self):
        """Test RSI calculation for extreme conditions"""
        from indicators import calculate_rsi
        
        # All gains
        uptrend = [100 + i for i in range(50)]
        rsi_up = calculate_rsi(uptrend)
        self.assertGreater(rsi_up, 70, "Strong uptrend should yield RSI > 70")
        
        # All losses
        downtrend = [100 - i for i in range(50)]
        rsi_down = calculate_rsi(downtrend)
        self.assertLess(rsi_down, 30, "Strong downtrend should yield RSI < 30")
    
    def test_vwap_weighted_correctly(self):
        """Test that VWAP correctly weights by volume"""
        from indicators import calculate_vwap
        
        # Price 100 with volume 1, price 200 with volume 9
        # VWAP should be close to 200
        prices = [100, 200]
        volumes = [1, 9]
        
        vwap = calculate_vwap(prices, volumes)
        
        expected = (100*1 + 200*9) / (1 + 9)  # 190
        self.assertAlmostEqual(vwap, expected, places=2)
    
    def test_bollinger_bands_structure(self):
        """Test Bollinger Bands maintain proper structure"""
        from indicators import calculate_bollinger_bands
        
        # Stable prices
        prices = [100] * 50
        upper, middle, lower = calculate_bollinger_bands(prices)
        
        # With no volatility, bands should be very close
        self.assertAlmostEqual(upper, middle, delta=1)
        self.assertAlmostEqual(lower, middle, delta=1)
        
        # Volatile prices
        volatile_prices = [100 + (i % 2) * 10 for i in range(50)]
        upper_v, middle_v, lower_v = calculate_bollinger_bands(volatile_prices)
        
        # With volatility, bands should be spread
        self.assertGreater(upper_v - middle_v, 5)
        self.assertGreater(middle_v - lower_v, 5)


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFullTradingCycle))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIResilience))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestIndicatorAccuracy))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
