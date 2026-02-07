"""
Tests for RiskManager - Risk assessment and position sizing
"""
import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_manager import RiskManager, RiskLevel, PortfolioState

class TestRiskManager:
    """Test risk assessment"""
    
    def test_can_buy_low_risk(self):
        """Test buy approval in low-risk conditions"""
        rm = RiskManager(max_daily_trades=10)
        
        portfolio = PortfolioState(
            btc_balance=0.5,
            eur_balance=10000.0,
            current_price=50000.0,
            avg_buy_price=48000.0,
            unrealized_pnl=1000.0,
            win_rate=0.6,
            volatility=0.02,  # Low volatility
            max_daily_drawdown=0.01
        )
        
        assert rm.can_buy(portfolio) == True
    
    def test_cannot_buy_high_volatility(self):
        """Test buy rejection in high volatility"""
        rm = RiskManager()
        
        portfolio = PortfolioState(
            btc_balance=0.5,
            eur_balance=10000.0,
            current_price=50000.0,
            avg_buy_price=48000.0,
            unrealized_pnl=1000.0,
            win_rate=0.6,
            volatility=0.15,  # CRITICAL volatility
            max_daily_drawdown=0.01
        )
        
        assert rm.can_buy(portfolio) == False
    
    def test_daily_trade_limit(self):
        """Test daily trade limit enforcement"""
        rm = RiskManager(max_daily_trades=2)
        
        portfolio = PortfolioState(
            btc_balance=0.5,
            eur_balance=10000.0,
            current_price=50000.0,
            avg_buy_price=48000.0,
            unrealized_pnl=1000.0,
            win_rate=0.6,
            volatility=0.02,
            max_daily_drawdown=0.01
        )
        
        # First trade OK
        assert rm.can_buy(portfolio) == True
        rm.record_trade(is_buy=True)
        
        # Second trade OK
        assert rm.can_buy(portfolio) == True
        rm.record_trade(is_buy=True)
        
        # Third trade blocked
        assert rm.can_buy(portfolio) == False
    
    def test_position_sizing_adjusts_for_risk(self):
        """Test that position size reduces under high risk"""
        rm = RiskManager()
        
        # Low risk portfolio
        low_risk = PortfolioState(
            btc_balance=0.5,
            eur_balance=10000.0,
            current_price=50000.0,
            avg_buy_price=48000.0,
            unrealized_pnl=1000.0,
            win_rate=0.6,
            volatility=0.02,
            max_daily_drawdown=0.01
        )
        
        # High risk portfolio
        high_risk = PortfolioState(
            btc_balance=0.5,
            eur_balance=10000.0,
            current_price=50000.0,
            avg_buy_price=48000.0,
            unrealized_pnl=-2000.0,
            win_rate=0.3,
            volatility=0.10,
            max_daily_drawdown=0.10
        )
        
        low_risk_size = rm.calculate_position_size(10000.0, 50000.0, low_risk)
        high_risk_size = rm.calculate_position_size(10000.0, 50000.0, high_risk)
        
        # High risk should result in smaller position
        assert high_risk_size < low_risk_size
    
    def test_emergency_sell_triggered(self):
        """Test emergency sell triggers correctly"""
        rm = RiskManager()
        
        # Critical losses
        portfolio = PortfolioState(
            btc_balance=0.5,
            eur_balance=2000.0,
            current_price=40000.0,  # Down from 50k
            avg_buy_price=50000.0,
            unrealized_pnl=-5000.0,  # Big loss
            win_rate=0.3,
            volatility=0.15,  # Critical
            max_daily_drawdown=0.20
        )
        
        assert rm.should_emergency_sell(portfolio) == True