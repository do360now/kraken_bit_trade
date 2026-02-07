"""
Tests for Trade dataclass - Immutable trade representation
"""
import os
import sys
import pytest
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade import Trade, TradeStatus, TradeType

class TestTrade:
    """Test Trade dataclass"""
    
    def test_trade_creation(self):
        """Test creating a trade"""
        trade = Trade(
            trade_id="ORDER123",
            trade_type=TradeType.BUY,
            status=TradeStatus.FILLED,
            btc_amount=0.1,
            btc_filled=0.1,
            price_limit=50000.0,
            price_filled=50000.0,
            total_cost=5000.0,
            fee_eur=10.0,
            created_at=datetime.now(),
            filled_at=datetime.now(),
            reason="Test trade"
        )
        
        assert trade.trade_id == "ORDER123"
        assert trade.is_success == True
        assert trade.fill_percentage == 100.0
    
    def test_trade_immutability(self):
        """Test that Trade is immutable"""
        trade = Trade(
            trade_id="ORDER123",
            trade_type=TradeType.BUY,
            status=TradeStatus.FILLED,
            btc_amount=0.1,
            btc_filled=0.1,
            price_limit=50000.0,
            price_filled=50000.0,
            total_cost=5000.0,
            fee_eur=10.0,
            created_at=datetime.now(),
            filled_at=datetime.now(),
            reason="Test"
        )
        
        with pytest.raises(AttributeError):
            trade.status = TradeStatus.CANCELLED  # Should fail
    
    def test_trade_validation(self):
        """Test that invalid trades are rejected"""
        with pytest.raises(ValueError):
            Trade(
                trade_id="BAD",
                trade_type=TradeType.BUY,
                status=TradeStatus.FILLED,
                btc_amount=0.1,
                btc_filled=0.2,  # ❌ Can't fill more than requested!
                price_limit=50000.0,
                price_filled=50000.0,
                total_cost=5000.0,
                fee_eur=10.0,
                created_at=datetime.now(),
                filled_at=datetime.now(),
                reason="Invalid"
            )