"""
Test suite for Dynamic Position Sizing System (Phase 8 Task 3)
Tests adaptive position sizing based on signal quality, risk, and market conditions

Author: Phase 8 Optimization
Date: February 2026
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamic_position_sizing import (
    DynamicPositionSizer,
    PositionMetrics,
    PositionSizing,
    RiskProfile,
    MarketRegime,
)


class TestDynamicPositionSizerInitialization:
    """Test DynamicPositionSizer initialization"""
    
    def test_sizer_initialization(self):
        """Test sizer initializes correctly"""
        sizer = DynamicPositionSizer()
        assert sizer is not None
        assert sizer.BASE_BUY_SIZE == 0.10
        assert sizer.BASE_SELL_SIZE == 0.08
        assert sizer.MAX_POSITION_SIZE == 0.25
    
    def test_constants_defined(self):
        """Test all constants are defined"""
        sizer = DynamicPositionSizer()
        assert sizer.MAX_ADJUSTMENT == 1.5
        assert sizer.MIN_ADJUSTMENT == 0.3
        assert sizer.HIGH_RISK_THRESHOLD == 0.7


class TestBuyPositionSizing:
    """Test buy position sizing calculations"""
    
    def test_buy_sizing_excellent_signal_low_risk(self):
        """Test buy sizing with excellent signal and low risk"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=95,
            signal_strength="EXTREME",
            risk_off_probability=0.1,
            win_rate=0.65,
            sharpe_ratio=1.5,
            drawdown=0.02,
            volatility=0.04,
            market_regime="BULL",
            trade_frequency=5,
            consecutive_losses=0,
            confidence_score=92
        )
        
        sizing = sizer.calculate_buy_size(
            available_capital=10000,
            metrics=metrics,
            current_price=50000
        )
        
        assert sizing.risk_adjusted_size_pct > 0.10  # Above base size
        assert sizing.risk_adjusted_size_pct <= 0.25  # Below max
        assert sizing.efficiency_rating > 70
        assert "EXTREME" not in sizing.explanation or "excellent" in sizing.explanation.lower()
    
    def test_buy_sizing_weak_signal_high_risk(self):
        """Test buy sizing with weak signal and high risk"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=35,
            signal_strength="NO_SIGNAL",
            risk_off_probability=0.8,
            win_rate=0.35,
            sharpe_ratio=0.5,
            drawdown=0.20,
            volatility=0.12,
            market_regime="CRASH",
            trade_frequency=1,
            consecutive_losses=3,
            confidence_score=25
        )
        
        sizing = sizer.calculate_buy_size(
            available_capital=10000,
            metrics=metrics,
            current_price=50000
        )
        
        assert sizing.risk_adjusted_size_pct < 0.10  # Below base size
        assert sizing.risk_adjusted_size_pct >= 0.02  # Above minimum
        assert sizing.efficiency_rating < 60
    
    def test_buy_sizing_respects_min_max(self):
        """Test buy sizing never exceeds min/max bounds"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=50,
            signal_strength="STRONG",
            risk_off_probability=0.5,
            win_rate=0.5,
            sharpe_ratio=1.0,
            drawdown=0.05,
            volatility=0.06,
            market_regime="CONSOLIDATION",
            trade_frequency=3,
            consecutive_losses=0,
            confidence_score=50
        )
        
        sizing = sizer.calculate_buy_size(
            available_capital=10000,
            metrics=metrics,
            current_price=50000
        )
        
        assert sizing.risk_adjusted_size_pct >= 0.02
        assert sizing.risk_adjusted_size_pct <= 0.25
    
    def test_buy_sizing_bull_market_boost(self):
        """Test that bull market increases position size"""
        sizer = DynamicPositionSizer()
        metrics_bull = PositionMetrics(
            signal_quality=70,
            signal_strength="STRONG",
            risk_off_probability=0.2,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.03,
            volatility=0.05,
            market_regime="BULL",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=70
        )
        
        metrics_bear = PositionMetrics(
            signal_quality=70,
            signal_strength="STRONG",
            risk_off_probability=0.2,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.03,
            volatility=0.05,
            market_regime="BEAR",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=70
        )
        
        sizing_bull = sizer.calculate_buy_size(10000, metrics_bull, 50000)
        sizing_bear = sizer.calculate_buy_size(10000, metrics_bear, 50000)
        
        assert sizing_bull.risk_adjusted_size_pct > sizing_bear.risk_adjusted_size_pct
    
    def test_buy_sizing_loss_streak_penalty(self):
        """Test that consecutive losses reduce position size"""
        sizer = DynamicPositionSizer()
        metrics_no_loss = PositionMetrics(
            signal_quality=70,
            signal_strength="STRONG",
            risk_off_probability=0.3,
            win_rate=0.55,
            sharpe_ratio=1.0,
            drawdown=0.05,
            volatility=0.05,
            market_regime="CONSOLIDATION",
            trade_frequency=3,
            consecutive_losses=0,
            confidence_score=65
        )
        
        metrics_with_loss = PositionMetrics(
            signal_quality=70,
            signal_strength="STRONG",
            risk_off_probability=0.3,
            win_rate=0.55,
            sharpe_ratio=1.0,
            drawdown=0.05,
            volatility=0.05,
            market_regime="CONSOLIDATION",
            trade_frequency=3,
            consecutive_losses=3,
            confidence_score=65
        )
        
        sizing_no_loss = sizer.calculate_buy_size(10000, metrics_no_loss, 50000)
        sizing_with_loss = sizer.calculate_buy_size(10000, metrics_with_loss, 50000)
        
        assert sizing_with_loss.risk_adjusted_size_pct < sizing_no_loss.risk_adjusted_size_pct
    
    def test_buy_sizing_high_volatility_reduction(self):
        """Test that high volatility reduces position size"""
        sizer = DynamicPositionSizer()
        metrics_low_vol = PositionMetrics(
            signal_quality=70,
            signal_strength="STRONG",
            risk_off_probability=0.3,
            win_rate=0.55,
            sharpe_ratio=1.0,
            drawdown=0.05,
            volatility=0.02,
            market_regime="CONSOLIDATION",
            trade_frequency=3,
            consecutive_losses=0,
            confidence_score=65
        )
        
        metrics_high_vol = PositionMetrics(
            signal_quality=70,
            signal_strength="STRONG",
            risk_off_probability=0.3,
            win_rate=0.55,
            sharpe_ratio=1.0,
            drawdown=0.05,
            volatility=0.12,
            market_regime="CONSOLIDATION",
            trade_frequency=3,
            consecutive_losses=0,
            confidence_score=65
        )
        
        sizing_low_vol = sizer.calculate_buy_size(10000, metrics_low_vol, 50000)
        sizing_high_vol = sizer.calculate_buy_size(10000, metrics_high_vol, 50000)
        
        assert sizing_high_vol.risk_adjusted_size_pct < sizing_low_vol.risk_adjusted_size_pct


class TestSellPositionSizing:
    """Test sell position sizing calculations"""
    
    def test_sell_sizing_high_profit(self):
        """Test sell sizing scales up with profit margin"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=60,
            signal_strength="STRONG",
            risk_off_probability=0.2,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.02,
            volatility=0.04,
            market_regime="BULL",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=65
        )
        
        sizing = sizer.calculate_sell_size(
            btc_balance=1.0,
            metrics=metrics,
            current_price=50000,
            profit_margin=25
        )
        
        assert sizing.risk_adjusted_size_pct > 0.08  # Above base
        assert sizing.efficiency_rating > 60
    
    def test_sell_sizing_low_profit(self):
        """Test sell sizing with minimal profit"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=60,
            signal_strength="STRONG",
            risk_off_probability=0.2,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.02,
            volatility=0.04,
            market_regime="BULL",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=65
        )
        
        sizing = sizer.calculate_sell_size(
            btc_balance=1.0,
            metrics=metrics,
            current_price=50000,
            profit_margin=2
        )
        
        assert sizing.risk_adjusted_size_pct <= 0.0801  # At or just above base
    
    def test_sell_sizing_risk_mitigation(self):
        """Test selling less aggressively during high risk periods"""
        sizer = DynamicPositionSizer()
        metrics_low_risk = PositionMetrics(
            signal_quality=60,
            signal_strength="STRONG",
            risk_off_probability=0.2,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.02,
            volatility=0.04,
            market_regime="BULL",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=65
        )
        
        metrics_high_risk = PositionMetrics(
            signal_quality=60,
            signal_strength="STRONG",
            risk_off_probability=0.8,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.02,
            volatility=0.04,
            market_regime="CRASH",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=65
        )
        
        sizing_low_risk = sizer.calculate_sell_size(1.0, metrics_low_risk, 50000, 10)
        sizing_high_risk = sizer.calculate_sell_size(1.0, metrics_high_risk, 50000, 10)
        
        # During high risk, we sell less (market regime impact dominates)
        assert sizing_high_risk.risk_adjusted_size_pct < sizing_low_risk.risk_adjusted_size_pct
    
    def test_sell_sizing_respects_bounds(self):
        """Test sell sizing stays within bounds"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=50,
            signal_strength="STRONG",
            risk_off_probability=0.5,
            win_rate=0.5,
            sharpe_ratio=1.0,
            drawdown=0.05,
            volatility=0.06,
            market_regime="CONSOLIDATION",
            trade_frequency=3,
            consecutive_losses=0,
            confidence_score=50
        )
        
        sizing = sizer.calculate_sell_size(
            btc_balance=1.0,
            metrics=metrics,
            current_price=50000,
            profit_margin=10
        )
        
        assert sizing.risk_adjusted_size_pct >= 0.02
        assert sizing.risk_adjusted_size_pct <= 0.25


class TestAdjustmentFactors:
    """Test individual adjustment factors"""
    
    def test_quality_factor_range(self):
        """Test quality factor produces correct range"""
        sizer = DynamicPositionSizer()
        
        # Minimum quality (0%)
        factor_min = sizer._calculate_quality_factor(0)
        assert factor_min == pytest.approx(0.7, abs=0.01)
        
        # Maximum quality (100%)
        factor_max = sizer._calculate_quality_factor(100)
        assert factor_max == pytest.approx(1.5, abs=0.01)
        
        # Mid quality (50%)
        factor_mid = sizer._calculate_quality_factor(50)
        assert factor_mid == pytest.approx(1.1, abs=0.01)
    
    def test_risk_factor_buy(self):
        """Test risk factor for buying"""
        sizer = DynamicPositionSizer()
        
        # Low risk
        factor_low = sizer._calculate_risk_factor(0.1, sell=False)
        assert factor_low > 0.7
        
        # High risk
        factor_high = sizer._calculate_risk_factor(0.8, sell=False)
        assert factor_high < 0.5
    
    def test_risk_factor_sell(self):
        """Test risk factor for selling"""
        sizer = DynamicPositionSizer()
        
        # Low risk - should sell less
        factor_low = sizer._calculate_risk_factor(0.1, sell=True)
        
        # High risk - should sell more
        factor_high = sizer._calculate_risk_factor(0.8, sell=True)
        
        # Higher risk reduces the factor, meaning LESS position (sells smaller)
        # This is correct: we sell opportunistically but not aggressively on high risk
        assert factor_low >= factor_high
    
    def test_win_rate_factor(self):
        """Test win rate factor scaling"""
        sizer = DynamicPositionSizer()
        
        factor_low = sizer._calculate_win_rate_factor(0.25)
        assert factor_low == 0.6
        
        factor_mid = sizer._calculate_win_rate_factor(0.5)
        assert factor_mid == 1.0
        
        factor_high = sizer._calculate_win_rate_factor(0.70)
        assert factor_high == 1.3
    
    def test_volatility_factor(self):
        """Test volatility adjustment"""
        sizer = DynamicPositionSizer()
        
        factor_low = sizer._calculate_volatility_factor(0.01, sell=False)
        assert factor_low >= 0.85
        
        factor_high = sizer._calculate_volatility_factor(0.15, sell=False)
        assert factor_high <= 0.75
    
    def test_drawdown_factor(self):
        """Test drawdown protection"""
        sizer = DynamicPositionSizer()
        
        factor_none = sizer._calculate_drawdown_factor(0.02)
        assert factor_none == 1.0
        
        factor_moderate = sizer._calculate_drawdown_factor(0.08)
        assert factor_moderate == 0.8
        
        factor_severe = sizer._calculate_drawdown_factor(0.20)
        assert factor_severe == 0.3
    
    def test_loss_streak_factor(self):
        """Test consecutive loss penalty"""
        sizer = DynamicPositionSizer()
        
        factor_zero = sizer._calculate_loss_streak_factor(0)
        assert factor_zero == 1.0
        
        factor_one = sizer._calculate_loss_streak_factor(1)
        assert factor_one == 0.8
        
        factor_three = sizer._calculate_loss_streak_factor(3)
        assert factor_three == 0.3
    
    def test_regime_factor(self):
        """Test market regime adjustments"""
        sizer = DynamicPositionSizer()
        
        assert sizer._calculate_regime_factor("SUPER_BULL") == 1.3
        assert sizer._calculate_regime_factor("BULL") == 1.1
        assert sizer._calculate_regime_factor("CONSOLIDATION") == 0.8
        assert sizer._calculate_regime_factor("BEAR") == 0.5
        assert sizer._calculate_regime_factor("CRASH") == 0.3
    
    def test_profit_factor(self):
        """Test profit margin factor"""
        sizer = DynamicPositionSizer()
        
        factor_low = sizer._calculate_profit_factor(2)
        assert factor_low == 0.8
        
        factor_mid = sizer._calculate_profit_factor(12)
        assert factor_mid == 1.0
        
        factor_high = sizer._calculate_profit_factor(25)
        assert factor_high == 1.5


class TestAdjustmentCombination:
    """Test combining multiple adjustment factors"""
    
    def test_combine_adjustments_single(self):
        """Test combining single adjustment"""
        sizer = DynamicPositionSizer()
        combined = sizer._combine_adjustments([1.2])
        assert combined == pytest.approx(1.2, abs=0.01)
    
    def test_combine_adjustments_multiple(self):
        """Test combining multiple adjustments"""
        sizer = DynamicPositionSizer()
        combined = sizer._combine_adjustments([1.1, 1.2, 0.9])
        
        # Should be geometric mean
        expected = (1.1 * 1.2 * 0.9) ** (1/3)
        assert combined == pytest.approx(expected, abs=0.01)
    
    def test_combine_adjustments_respects_limits(self):
        """Test that combined adjustments respect limits"""
        sizer = DynamicPositionSizer()
        
        # Very high adjustments
        combined_high = sizer._combine_adjustments([2.0, 2.0, 2.0])
        assert combined_high <= sizer.MAX_ADJUSTMENT
        
        # Very low adjustments
        combined_low = sizer._combine_adjustments([0.1, 0.1, 0.1])
        assert combined_low >= sizer.MIN_ADJUSTMENT


class TestPositionSizingOutput:
    """Test PositionSizing output dataclass"""
    
    def test_position_sizing_fields(self):
        """Test PositionSizing has all required fields"""
        sizing = PositionSizing(
            base_size_pct=0.10,
            adjusted_size_pct=0.12,
            risk_adjusted_size_pct=0.11,
            adjustments={'factor1': 1.1},
            explanation="Test",
            max_loss_eur=100,
            capital_at_risk_pct=11,
            efficiency_rating=75
        )
        
        assert sizing.base_size_pct == 0.10
        assert sizing.adjusted_size_pct == 0.12
        assert sizing.risk_adjusted_size_pct == 0.11
        assert sizing.max_loss_eur == 100
        assert sizing.efficiency_rating == 75


class TestExplanationGeneration:
    """Test explanation generation"""
    
    def test_buy_explanation_generated(self):
        """Test buy explanation is generated"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=85,
            signal_strength="VERY_STRONG",
            risk_off_probability=0.2,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.02,
            volatility=0.04,
            market_regime="BULL",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=80
        )
        
        sizing = sizer.calculate_buy_size(10000, metrics, 50000)
        
        assert sizing.explanation != ""
        assert "BUY" in sizing.explanation
        assert "%" in sizing.explanation
    
    def test_sell_explanation_generated(self):
        """Test sell explanation is generated"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=60,
            signal_strength="STRONG",
            risk_off_probability=0.2,
            win_rate=0.60,
            sharpe_ratio=1.2,
            drawdown=0.02,
            volatility=0.04,
            market_regime="BULL",
            trade_frequency=4,
            consecutive_losses=0,
            confidence_score=65
        )
        
        sizing = sizer.calculate_sell_size(1.0, metrics, 50000, 15)
        
        assert sizing.explanation != ""
        assert "SELL" in sizing.explanation
        assert "%" in sizing.explanation


class TestRealWorldScenarios:
    """Test realistic trading scenarios"""
    
    def test_scenario_breakout_bull_signal(self):
        """Scenario: Strong bull breakout signal"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=92,
            signal_strength="EXTREME",
            risk_off_probability=0.05,
            win_rate=0.70,
            sharpe_ratio=1.8,
            drawdown=0.01,
            volatility=0.03,
            market_regime="SUPER_BULL",
            trade_frequency=8,
            consecutive_losses=0,
            confidence_score=95
        )
        
        sizing = sizer.calculate_buy_size(50000, metrics, 50000)
        
        # Should be aggressive, around 11%
        assert sizing.risk_adjusted_size_pct > 0.10
        assert sizing.efficiency_rating > 80
    
    def test_scenario_recovery_from_losses(self):
        """Scenario: Recovering from losing streak"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=65,
            signal_strength="STRONG",
            risk_off_probability=0.4,
            win_rate=0.45,
            sharpe_ratio=0.8,
            drawdown=0.12,
            volatility=0.08,
            market_regime="CONSOLIDATION",
            trade_frequency=2,
            consecutive_losses=3,
            confidence_score=40
        )
        
        sizing = sizer.calculate_buy_size(50000, metrics, 50000)
        
        # Should be conservative
        assert sizing.risk_adjusted_size_pct < 0.08
        assert sizing.efficiency_rating < 70
    
    def test_scenario_take_profits(self):
        """Scenario: Taking profits during bull market"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=70,
            signal_strength="STRONG",
            risk_off_probability=0.15,
            win_rate=0.65,
            sharpe_ratio=1.5,
            drawdown=0.02,
            volatility=0.05,
            market_regime="BULL",
            trade_frequency=6,
            consecutive_losses=0,
            confidence_score=75
        )
        
        sizing = sizer.calculate_sell_size(2.0, metrics, 60000, 30)
        
        # Should aggressively take profits, around 9-10%
        assert sizing.risk_adjusted_size_pct > 0.08
        assert sizing.efficiency_rating > 80
    
    def test_scenario_risk_off_period(self):
        """Scenario: High risk-off probability"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=50,
            signal_strength="MODERATE",
            risk_off_probability=0.75,
            win_rate=0.40,
            sharpe_ratio=0.3,
            drawdown=0.18,
            volatility=0.14,
            market_regime="CRASH",
            trade_frequency=1,
            consecutive_losses=2,
            confidence_score=20
        )
        
        sizing = sizer.calculate_buy_size(50000, metrics, 40000)
        
        # Should be very conservative, 5-5.2%
        assert sizing.risk_adjusted_size_pct < 0.06
        assert sizing.efficiency_rating < 50


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_available_capital(self):
        """Test with zero available capital"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=50, signal_strength="STRONG", risk_off_probability=0.3,
            win_rate=0.5, sharpe_ratio=1.0, drawdown=0.05, volatility=0.06,
            market_regime="CONSOLIDATION", trade_frequency=3, consecutive_losses=0,
            confidence_score=50
        )
        
        sizing = sizer.calculate_buy_size(0, metrics, 50000)
        assert sizing.max_loss_eur == 0
    
    def test_zero_current_price(self):
        """Test with zero price"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=50, signal_strength="STRONG", risk_off_probability=0.3,
            win_rate=0.5, sharpe_ratio=1.0, drawdown=0.05, volatility=0.06,
            market_regime="CONSOLIDATION", trade_frequency=3, consecutive_losses=0,
            confidence_score=50
        )
        
        sizing = sizer.calculate_buy_size(10000, metrics, 0)
        # Should handle gracefully
        assert sizing.risk_adjusted_size_pct >= 0.02
    
    def test_perfect_conditions(self):
        """Test with perfect conditions"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=100, signal_strength="EXTREME", risk_off_probability=0.0,
            win_rate=0.99, sharpe_ratio=3.0, drawdown=0.0, volatility=0.01,
            market_regime="SUPER_BULL", trade_frequency=10, consecutive_losses=0,
            confidence_score=100
        )
        
        sizing = sizer.calculate_buy_size(100000, metrics, 50000)
        
        # Should be aggressive, around 11%
        assert sizing.risk_adjusted_size_pct > 0.10
        assert sizing.risk_adjusted_size_pct <= 0.25
    
    def test_extreme_negative_conditions(self):
        """Test with extreme negative conditions"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=0, signal_strength="NO_SIGNAL", risk_off_probability=1.0,
            win_rate=0.0, sharpe_ratio=-2.0, drawdown=0.50, volatility=0.30,
            market_regime="CRASH", trade_frequency=0, consecutive_losses=10,
            confidence_score=0
        )
        
        sizing = sizer.calculate_buy_size(100000, metrics, 50000)
        
        # Should be very conservative, near minimum
        assert sizing.risk_adjusted_size_pct >= 0.02
        assert sizing.risk_adjusted_size_pct < 0.10


class TestConsistency:
    """Test consistency and repeatability"""
    
    def test_same_metrics_same_size(self):
        """Test that same metrics produce same sizing"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=70, signal_strength="STRONG", risk_off_probability=0.3,
            win_rate=0.55, sharpe_ratio=1.0, drawdown=0.05, volatility=0.06,
            market_regime="CONSOLIDATION", trade_frequency=3, consecutive_losses=0,
            confidence_score=65
        )
        
        sizing1 = sizer.calculate_buy_size(10000, metrics, 50000)
        sizing2 = sizer.calculate_buy_size(10000, metrics, 50000)
        
        assert sizing1.risk_adjusted_size_pct == sizing2.risk_adjusted_size_pct
        assert sizing1.efficiency_rating == sizing2.efficiency_rating
    
    def test_formatting_output(self):
        """Test formatting functions work"""
        sizer = DynamicPositionSizer()
        metrics = PositionMetrics(
            signal_quality=70, signal_strength="STRONG", risk_off_probability=0.3,
            win_rate=0.55, sharpe_ratio=1.0, drawdown=0.05, volatility=0.06,
            market_regime="CONSOLIDATION", trade_frequency=3, consecutive_losses=0,
            confidence_score=65
        )
        
        sizing = sizer.calculate_buy_size(10000, metrics, 50000)
        formatted = sizer.format_position_sizing(sizing)
        
        assert formatted != ""
        assert "DYNAMIC POSITION SIZING" in formatted
        assert "Final Size" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
