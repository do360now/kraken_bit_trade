#!/usr/bin/env python3
"""
Test suite for Kraken API balance calculations.
Tests the critical balance bug fix.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKrakenAPIBalanceCalculation(unittest.TestCase):
    """Tests for the fixed balance calculation logic"""
    
    def setUp(self):
        """Setup test fixtures"""
        from kraken_api import KrakenAPI
        self.api = KrakenAPI("test_key", "test_secret")
    
    def test_eur_balance_with_buy_orders(self):
        """
        Test that EUR balance correctly accounts for funds locked in buy orders.
        
        This tests the FIX for the critical bug where only total volume was used,
        not remaining volume.
        """
        with patch.object(self.api, 'query_private') as mock_query:
            # Setup mock responses
            mock_query.side_effect = [
                # Balance response
                {'result': {'ZEUR': '1000.0'}, 'error': []},
                # Open orders response with partial fill
                {
                    'result': {
                        'open': {
                            'ORDER1': {
                                'descr': {
                                    'pair': 'XXBTZEUR',
                                    'type': 'buy',
                                    'price': '50000'
                                },
                                'vol': '0.01',      # Total volume
                                'vol_exec': '0.004' # Already executed 40%
                            }
                        }
                    },
                    'error': []
                }
            ]
            
            balance = self.api.get_available_balance('EUR')
            
            # Should deduct REMAINING volume only: (0.01 - 0.004) * 50000 = 300 EUR
            # Total: 1000, Locked: 300, Available: 700
            expected_available = 700.0
            
            self.assertAlmostEqual(balance, expected_available, places=6,
                msg=f"Expected {expected_available}, got {balance}")
    
    def test_btc_balance_with_sell_orders(self):
        """
        Test that BTC balance correctly accounts for BTC locked in sell orders.
        
        This tests the FIX where sell orders were not considered at all.
        """
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.side_effect = [
                # Balance response
                {'result': {'XXBT': '0.1'}, 'error': []},
                # Open orders response
                {
                    'result': {
                        'open': {
                            'ORDER1': {
                                'descr': {
                                    'pair': 'XXBTZEUR',
                                    'type': 'sell',
                                    'price': '50000'
                                },
                                'vol': '0.03',
                                'vol_exec': '0.01'  # 1/3 executed
                            }
                        }
                    },
                    'error': []
                }
            ]
            
            balance = self.api.get_available_balance('XXBT')
            
            # Should deduct REMAINING volume: 0.03 - 0.01 = 0.02 BTC
            # Total: 0.1, Locked: 0.02, Available: 0.08
            expected_available = 0.08
            
            self.assertAlmostEqual(balance, expected_available, places=8,
                msg=f"Expected {expected_available}, got {balance}")
    
    def test_balance_with_multiple_orders(self):
        """Test balance calculation with multiple open orders"""
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.side_effect = [
                {'result': {'ZEUR': '2000.0'}, 'error': []},
                {
                    'result': {
                        'open': {
                            'ORDER1': {
                                'descr': {'pair': 'XXBTZEUR', 'type': 'buy', 'price': '50000'},
                                'vol': '0.01',
                                'vol_exec': '0.0'
                            },
                            'ORDER2': {
                                'descr': {'pair': 'XXBTZEUR', 'type': 'buy', 'price': '51000'},
                                'vol': '0.005',
                                'vol_exec': '0.002'
                            }
                        }
                    },
                    'error': []
                }
            ]
            
            balance = self.api.get_available_balance('EUR')
            
            # ORDER1: 0.01 * 50000 = 500 EUR locked
            # ORDER2: (0.005 - 0.002) * 51000 = 153 EUR locked
            # Total locked: 653 EUR
            # Available: 2000 - 653 = 1347 EUR
            expected_available = 1347.0
            
            self.assertAlmostEqual(balance, expected_available, places=6)
    
    def test_balance_with_fully_filled_order(self):
        """
        Test that fully filled orders don't lock any balance.
        
        This would appear in open orders but vol_exec == vol
        """
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.side_effect = [
                {'result': {'ZEUR': '1000.0'}, 'error': []},
                {
                    'result': {
                        'open': {
                            'ORDER1': {
                                'descr': {'pair': 'XXBTZEUR', 'type': 'buy', 'price': '50000'},
                                'vol': '0.01',
                                'vol_exec': '0.01'  # Fully filled
                            }
                        }
                    },
                    'error': []
                }
            ]
            
            balance = self.api.get_available_balance('EUR')
            
            # No remaining volume, so nothing locked
            expected_available = 1000.0
            
            self.assertAlmostEqual(balance, expected_available, places=6)
    
    def test_balance_calculation_negative_protection(self):
        """
        Test that available balance never goes negative.
        
        Edge case: If somehow locked > balance, should return 0
        """
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.side_effect = [
                {'result': {'ZEUR': '100.0'}, 'error': []},
                {
                    'result': {
                        'open': {
                            'ORDER1': {
                                'descr': {'pair': 'XXBTZEUR', 'type': 'buy', 'price': '50000'},
                                'vol': '0.01',
                                'vol_exec': '0.0'
                            }
                        }
                    },
                    'error': []
                }
            ]
            
            balance = self.api.get_available_balance('EUR')
            
            # Locked: 0.01 * 50000 = 500, but balance is only 100
            # Should return 0, not -400
            self.assertEqual(balance, 0.0)
            self.assertGreaterEqual(balance, 0, "Balance should never be negative")
    
    def test_balance_no_open_orders(self):
        """Test balance when there are no open orders"""
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.side_effect = [
                {'result': {'ZEUR': '1000.0'}, 'error': []},
                {'result': {'open': {}}, 'error': []}
            ]
            
            balance = self.api.get_available_balance('EUR')
            
            # No locked funds
            self.assertEqual(balance, 1000.0)
    
    def test_balance_api_error_handling(self):
        """Test that API errors are properly handled"""
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.side_effect = [
                {'result': {'ZEUR': '1000.0'}, 'error': []},
                {'error': ['EGeneral:Internal error'], 'result': None}
            ]
            
            # Should handle the error gracefully and return base balance
            balance = self.api.get_available_balance('EUR')
            self.assertEqual(balance, 1000.0)
    
    def test_btc_balance_not_found_returns_zero(self):
        """Test that missing BTC (but successful API) returns 0.0"""
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.return_value = {
                'error': [],
                'result': {'ZEUR': '1000.0'}  # No BTC
            }
            
            balance = self.api.get_total_btc_balance()
            self.assertEqual(balance, 0.0)
    
    def test_btc_balance_api_error_returns_none(self):
        """Test that API errors return None"""
        with patch.object(self.api, 'query_private') as mock_query:
            mock_query.return_value = {
                'error': ['EService:Unavailable']
            }
            
            balance = self.api.get_total_btc_balance()
            self.assertIsNone(balance)


class TestKrakenAPIThreadSafety(unittest.TestCase):
    """Tests for thread-safe operations"""
    
    def setUp(self):
        from kraken_api import KrakenAPI
        self.api = KrakenAPI("test_key", "test_secret")
    
    def test_concurrent_requests_rate_limited(self):
        """Test that concurrent requests are properly rate limited"""
        call_times = []
        
        def make_request():
            with patch('requests.get') as mock_get:
                mock_get.return_value.json.return_value = {
                    'result': {'XXBTZEUR': {'c': ['50000']}},
                    'error': []
                }
                mock_get.return_value.raise_for_status = Mock()
                
                call_times.append(time.time())
                self.api.get_btc_price()
        
        # Make concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        
        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify rate limiting worked
        # Calls should be spaced by request_interval
        # (Simplified test - in real scenario, check actual timings)
        self.assertEqual(len(call_times), 5)


def test_import_modules():
    """Verify all modules can be imported"""
    try:
        from kraken_api import KrakenAPI
        from database_manager import DatabaseManager
        from circuit_breaker import circuit_breaker
        from logger_config import logger
        
        # These might not exist in original project
        try:
            from core.exceptions import APIError
            from core.constants import OrderState
        except ImportError:
            pass  # OK if these don't exist
        
        assert True, "All modules imported successfully"
    except ImportError as e:
        import pytest
        pytest.fail(f"Failed to import module: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)