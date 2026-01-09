#!/usr/bin/env python3
"""
Unit tests for the fixed win rate calculation (Issue #8)
Tests FIFO trade matching and profit/loss calculations
"""

import sys
import time
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, '/home/claude')

from performance_tracker import PerformanceTracker

def test_basic_fifo():
    """Test basic FIFO matching: buy low, sell high"""
    print("\n=== Test 1: Basic FIFO - Buy Low, Sell High ===")
    
    tracker = PerformanceTracker()
    
    # Buy at 50k
    tracker.record_trade("buy1", "buy", 0.1, 50000, 10, 
                        timestamp=time.time())
    
    # Sell at 55k (should be profitable)
    tracker.record_trade("sell1", "sell", 0.1, 55000, 10, 
                        timestamp=time.time() + 1)
    
    win_rate = tracker.calculate_win_rate()
    print(f"Win rate: {win_rate:.1%}")
    assert win_rate == 1.0, f"Expected 100% win rate, got {win_rate:.1%}"
    print("✅ PASS: Basic profitable trade")

def test_partial_fills():
    """Test partial fills: sell less than you bought"""
    print("\n=== Test 2: Partial Fills ===")
    
    tracker = PerformanceTracker()
    
    # Buy 0.2 BTC at 50k
    tracker.record_trade("buy1", "buy", 0.2, 50000, 20, 
                        timestamp=time.time())
    
    # Sell 0.1 BTC at 55k (half the position, profitable)
    tracker.record_trade("sell1", "sell", 0.1, 55000, 10, 
                        timestamp=time.time() + 1)
    
    # Sell remaining 0.1 BTC at 48k (loss)
    tracker.record_trade("sell2", "sell", 0.1, 48000, 10, 
                        timestamp=time.time() + 2)
    
    win_rate = tracker.calculate_win_rate()
    print(f"Win rate: {win_rate:.1%}")
    assert win_rate == 0.5, f"Expected 50% win rate, got {win_rate:.1%}"
    print("✅ PASS: Partial fills handled correctly")

def test_multiple_buys_one_sell():
    """Test selling across multiple buy lots"""
    print("\n=== Test 3: Multiple Buys, One Large Sell ===")
    
    tracker = PerformanceTracker()
    
    # Buy 0.1 @ 50k
    tracker.record_trade("buy1", "buy", 0.1, 50000, 10, 
                        timestamp=time.time())
    
    # Buy 0.1 @ 60k  
    tracker.record_trade("buy2", "buy", 0.1, 60000, 10, 
                        timestamp=time.time() + 1)
    
    # Sell 0.15 @ 55k (uses 0.1 from first buy @ 50k + 0.05 from second buy @ 60k)
    # First 0.1: 55k - 50k = +5k profit (win)
    # Next 0.05: 55k - 60k = -5k loss (but avg is 52.5k, so should be win overall)
    tracker.record_trade("sell1", "sell", 0.15, 55000, 15, 
                        timestamp=time.time() + 2)
    
    win_rate = tracker.calculate_win_rate()
    
    # Avg buy price for 0.15: (0.1*50k + 0.05*60k) / 0.15 = 53,333
    # Selling at 55k > 53,333 = WIN
    print(f"Win rate: {win_rate:.1%}")
    assert win_rate == 1.0, f"Expected 100% win rate, got {win_rate:.1%}"
    print("✅ PASS: Multiple buy lots handled correctly")

def test_dca_strategy():
    """Test dollar-cost averaging (DCA) strategy"""
    print("\n=== Test 4: DCA Strategy ===")
    
    tracker = PerformanceTracker()
    base_time = time.time()
    
    # Buy regularly over time (DCA)
    prices = [50000, 48000, 52000, 49000, 51000]
    for i, price in enumerate(prices):
        tracker.record_trade(f"buy{i}", "buy", 0.02, price, 5, 
                           timestamp=base_time + i)
    
    # Average cost: (50k + 48k + 52k + 49k + 51k) / 5 = 50k
    
    # Sell all at 52k (should be profitable)
    tracker.record_trade("sell1", "sell", 0.1, 52000, 10, 
                        timestamp=base_time + 10)
    
    win_rate = tracker.calculate_win_rate()
    print(f"Win rate: {win_rate:.1%}")
    assert win_rate == 1.0, f"Expected 100% win rate for DCA, got {win_rate:.1%}"
    print("✅ PASS: DCA strategy tracked correctly")

def test_losing_trades():
    """Test that losses are properly counted"""
    print("\n=== Test 5: Losing Trades ===")
    
    tracker = PerformanceTracker()
    base_time = time.time()
    
    # Buy high
    tracker.record_trade("buy1", "buy", 0.1, 60000, 10, 
                        timestamp=base_time)
    
    # Sell low (loss)
    tracker.record_trade("sell1", "sell", 0.1, 55000, 10, 
                        timestamp=base_time + 1)
    
    win_rate = tracker.calculate_win_rate()
    print(f"Win rate: {win_rate:.1%}")
    assert win_rate == 0.0, f"Expected 0% win rate, got {win_rate:.1%}"
    print("✅ PASS: Losing trades counted correctly")

def test_mixed_trades():
    """Test realistic mix of wins and losses"""
    print("\n=== Test 6: Mixed Win/Loss Scenario ===")
    
    tracker = PerformanceTracker()
    base_time = time.time()
    
    # Series of trades
    trades = [
        ("buy", 0.1, 50000),   # Buy @ 50k
        ("sell", 0.1, 52000),  # Sell @ 52k = WIN
        ("buy", 0.1, 53000),   # Buy @ 53k
        ("sell", 0.1, 51000),  # Sell @ 51k = LOSS
        ("buy", 0.1, 49000),   # Buy @ 49k
        ("sell", 0.1, 50000),  # Sell @ 50k = WIN
        ("buy", 0.1, 55000),   # Buy @ 55k
        ("sell", 0.1, 54000),  # Sell @ 54k = LOSS
    ]
    
    for i, (side, volume, price) in enumerate(trades):
        tracker.record_trade(f"{side}{i}", side, volume, price, 10,
                           timestamp=base_time + i)
    
    win_rate = tracker.calculate_win_rate()
    print(f"Win rate: {win_rate:.1%}")
    # 4 sells total: 2 wins, 2 losses = 50%
    assert win_rate == 0.5, f"Expected 50% win rate, got {win_rate:.1%}"
    print("✅ PASS: Mixed trades calculated correctly")

def test_no_sells():
    """Test that win rate is 0 when no sells have occurred"""
    print("\n=== Test 7: No Sells Yet ===")
    
    tracker = PerformanceTracker()
    
    # Only buys, no sells
    tracker.record_trade("buy1", "buy", 0.1, 50000, 10)
    tracker.record_trade("buy2", "buy", 0.1, 52000, 10)
    
    win_rate = tracker.calculate_win_rate()
    print(f"Win rate: {win_rate:.1%}")
    assert win_rate == 0.0, f"Expected 0% (no sells), got {win_rate:.1%}"
    print("✅ PASS: No sells handled correctly")

def test_comparison_old_vs_new():
    """Compare old buggy logic vs new FIFO logic"""
    print("\n=== Test 8: Old vs New Logic Comparison ===")
    
    # Scenario where old logic gives WRONG result
    tracker = PerformanceTracker()
    base_time = time.time()
    
    tracker.record_trade("buy1", "buy", 0.1, 50000, 10, timestamp=base_time)
    tracker.record_trade("buy2", "buy", 0.1, 60000, 10, timestamp=base_time + 1)
    tracker.record_trade("sell1", "sell", 0.1, 55000, 10, timestamp=base_time + 2)
    
    # OLD LOGIC: Would use BOTH buys for avg = 55k = breakeven
    # NEW LOGIC: Uses FIRST buy only = 50k = PROFIT
    
    win_rate = tracker.calculate_win_rate()
    print(f"New (FIFO) win rate: {win_rate:.1%}")
    assert win_rate == 1.0, "New FIFO logic should show 100% win"
    
    # Simulate old logic
    old_avg = (50000 + 60000) / 2  # 55k
    old_profitable = 55000 > old_avg  # False (breakeven)
    print(f"Old (buggy) would say: {'WIN' if old_profitable else 'BREAKEVEN/LOSS'}")
    print("✅ PASS: New logic correctly identifies this as profitable")

def run_all_tests():
    """Run all test cases"""
    print("="*60)
    print("Testing Win Rate Fix (Issue #8)")
    print("="*60)
    
    tests = [
        test_basic_fifo,
        test_partial_fills,
        test_multiple_buys_one_sell,
        test_dca_strategy,
        test_losing_trades,
        test_mixed_trades,
        test_no_sells,
        test_comparison_old_vs_new,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)