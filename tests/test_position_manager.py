"""
Tests for PositionManager - Position tracking and portfolio metrics
"""
import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from position_manager import PositionManager, Position

class TestPositionManager:
    """Test position tracking"""
    
    def test_initial_position(self):
        """Test initial position state"""
        pm = PositionManager(
            initial_btc=0.5,
            initial_eur=10000.0,
            initial_price=50000.0
        )
        
        position = pm.get_position()
        
        assert position.btc_amount == 0.5
        assert position.eur_balance == 10000.0
        assert position.current_price == 50000.0
        assert position.btc_value == 25000.0  # 0.5 * 50000
        assert position.total_value == 35000.0  # 25000 + 10000
    
    def test_record_buy_updates_avg_price(self):
        """Test that buying updates average buy price correctly"""
        pm = PositionManager(initial_btc=0.0, initial_eur=10000.0)
        
        # Buy 0.1 BTC at 50000
        pm.record_trade(0.1, 50000.0, is_buy=True, fee_eur=10.0)
        
        assert pm.get_avg_buy_price() == 50000.0
        assert pm.get_btc_amount() == 0.1
        
        # Buy another 0.1 BTC at 60000
        pm.record_trade(0.1, 60000.0, is_buy=True, fee_eur=10.0)
        
        # Average should be 55000
        assert pm.get_avg_buy_price() == 55000.0
        assert pm.get_btc_amount() == 0.2
    
    def test_record_sell_calculates_profit(self):
        """Test that selling tracks wins/losses correctly"""
        pm = PositionManager(initial_btc=0.0, initial_eur=10000.0)
        
        # Buy at 50000
        pm.record_trade(0.1, 50000.0, is_buy=True)
        
        # Sell at 55000 (profit)
        pm.record_trade(0.1, 55000.0, is_buy=False)
        
        metrics = pm.get_portfolio_metrics()
        assert metrics.winning_trades == 1
        assert metrics.win_rate == 100.0
    
    def test_position_immutability(self):
        """Test that Position dataclass is immutable"""
        pm = PositionManager(initial_btc=0.5, initial_eur=10000.0, initial_price=50000.0)
        position = pm.get_position()
        
        with pytest.raises(AttributeError):
            position.btc_amount = 1.0  # Should fail - frozen dataclass
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation"""
        pm = PositionManager(initial_btc=0.0, initial_eur=10000.0)
        
        # Buy at 50000
        pm.record_trade(0.1, 50000.0, is_buy=True)
        
        # Update with new price
        pm.update_balance(0.1, 5000.0, 55000.0)
        
        position = pm.get_position()
        
        # Bought at 50k, now worth 55k
        # Unrealized P&L = (55000 - 50000) * 0.1 = 500
        assert position.unrealized_pnl == pytest.approx(500.0, rel=0.01)
        assert position.unrealized_pnl_pct == pytest.approx(10.0, rel=0.01)