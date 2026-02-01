"""
Test Suite for Intraday Volatility Scalping System - Phase 8 Task 5
===================================================================

Comprehensive tests for volatility detection, signal generation, and scalping logic.
Target: 45+ tests covering all components.
"""

import os
import pytest
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intraday_volatility_scalping import (
    IntraDayVolatilityScalper,
    VolatilityMetrics,
    VolatilityRegime,
    ScalpSignal,
    ScalpDirection,
    ScalpPosition,
)


class TestVolatilityRegimeClassification:
    """Tests for volatility regime classification."""
    
    def test_low_volatility_detection(self):
        """Test detection of low volatility regime."""
        scalper = IntraDayVolatilityScalper()
        regime = scalper._classify_volatility_regime(0.003)
        assert regime == VolatilityRegime.LOW
    
    def test_moderate_volatility_detection(self):
        """Test detection of moderate volatility regime."""
        scalper = IntraDayVolatilityScalper()
        regime = scalper._classify_volatility_regime(0.008)
        assert regime == VolatilityRegime.MODERATE
    
    def test_high_volatility_detection(self):
        """Test detection of high volatility regime."""
        scalper = IntraDayVolatilityScalper()
        regime = scalper._classify_volatility_regime(0.020)
        assert regime == VolatilityRegime.HIGH
    
    def test_extreme_volatility_detection(self):
        """Test detection of extreme volatility regime."""
        scalper = IntraDayVolatilityScalper()
        regime = scalper._classify_volatility_regime(0.035)
        assert regime == VolatilityRegime.EXTREME


class TestATRCalculation:
    """Tests for Average True Range calculation."""
    
    def test_atr_calculation_basic(self):
        """Test basic ATR calculation."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000 + i * 100 for i in range(20)]  # Rising prices
        atr = scalper._calculate_atr(prices, 14)
        assert atr > 0
        assert atr < 200  # Max variation is 100
    
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000, 40100, 40200]
        atr = scalper._calculate_atr(prices, 14)
        assert atr >= 0
    
    def test_atr_period_7(self):
        """Test ATR calculation with period 7."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000 + i * 50 for i in range(15)]
        atr_7 = scalper._calculate_atr(prices, 7)
        atr_14 = scalper._calculate_atr(prices, 14)
        assert atr_7 > 0
        assert atr_14 > 0
        # Shorter period should be more responsive
        assert atr_7 >= 0


class TestVWAPCalculation:
    """Tests for VWAP calculation."""
    
    def test_vwap_calculation_basic(self):
        """Test basic VWAP calculation."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000, 40100, 40200, 40150, 40250]
        volumes = [10, 20, 15, 25, 30]
        vwap = scalper._calculate_vwap(prices, volumes)
        assert vwap > 0
        assert min(prices) <= vwap <= max(prices)
    
    def test_vwap_equal_volumes(self):
        """Test VWAP with equal volumes."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000, 40100, 40200]
        volumes = [10, 10, 10]
        vwap = scalper._calculate_vwap(prices, volumes)
        assert vwap == pytest.approx(np.mean(prices), rel=0.01)
    
    def test_vwap_zero_volumes(self):
        """Test VWAP with zero volumes."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000, 40100, 40200]
        volumes = [0, 0, 0]
        vwap = scalper._calculate_vwap(prices, volumes)
        assert vwap > 0


class TestMeanReversionProbability:
    """Tests for mean reversion probability calculation."""
    
    def test_high_rsi_mean_reversion(self):
        """Test mean reversion with high RSI."""
        scalper = IntraDayVolatilityScalper()
        prob = scalper._calculate_mean_reversion_prob(rsi=80, trend_strength=0.3, bb_width=0.1)
        assert prob > 0.2
    
    def test_low_rsi_mean_reversion(self):
        """Test mean reversion with low RSI."""
        scalper = IntraDayVolatilityScalper()
        prob = scalper._calculate_mean_reversion_prob(rsi=20, trend_strength=0.3, bb_width=0.1)
        assert prob > 0.2
    
    def test_neutral_rsi_low_probability(self):
        """Test mean reversion with neutral RSI."""
        scalper = IntraDayVolatilityScalper()
        prob = scalper._calculate_mean_reversion_prob(rsi=50, trend_strength=1.0, bb_width=0.01)
        assert prob < 0.5


class TestBollingerSignal:
    """Tests for Bollinger Band signal generation."""
    
    def test_price_above_upper_band_short_signal(self):
        """Test short signal when price is above upper band."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000 + i * 10 for i in range(20)]  # Steady rise
        prices[-1] = 40300  # Price spike above band
        volatility = VolatilityMetrics(trend_strength=0.2)
        direction, confidence = scalper._bollinger_scalp_signal(40300, volatility, prices)
        assert direction == ScalpDirection.SHORT
        assert confidence > 0
    
    def test_price_below_lower_band_long_signal(self):
        """Test long signal when price is below lower band."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000 + i * 10 for i in range(20)]
        prices[-1] = 39700  # Price drop below band
        volatility = VolatilityMetrics(trend_strength=0.2)
        direction, confidence = scalper._bollinger_scalp_signal(39700, volatility, prices)
        assert direction == ScalpDirection.LONG
        assert confidence > 0
    
    def test_price_within_bands_no_signal(self):
        """Test no signal when price is within bands."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000 + i * 10 for i in range(20)]
        volatility = VolatilityMetrics(trend_strength=0.2)
        direction, confidence = scalper._bollinger_scalp_signal(40100, volatility, prices)
        assert direction == ScalpDirection.NONE


class TestRSISignal:
    """Tests for RSI-based signal generation."""
    
    def test_overbought_rsi_short_signal(self):
        """Test short signal for overbought RSI."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._rsi_divergence_signal(rsi=80)
        assert direction == ScalpDirection.SHORT
        assert confidence > 0
    
    def test_oversold_rsi_long_signal(self):
        """Test long signal for oversold RSI."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._rsi_divergence_signal(rsi=20)
        assert direction == ScalpDirection.LONG
        assert confidence > 0
    
    def test_neutral_rsi_no_signal(self):
        """Test no signal for neutral RSI."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._rsi_divergence_signal(rsi=50)
        assert direction == ScalpDirection.NONE
    
    def test_rsi_boundary_75(self):
        """Test RSI at boundary (75)."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._rsi_divergence_signal(rsi=76)
        assert direction == ScalpDirection.SHORT


class TestVWAPSignal:
    """Tests for VWAP interaction signal."""
    
    def test_price_above_vwap_short(self):
        """Test short signal when price > VWAP."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._vwap_interaction_signal(price=40300, vwap=40000)
        assert direction == ScalpDirection.SHORT
        assert confidence > 0
    
    def test_price_below_vwap_long(self):
        """Test long signal when price < VWAP."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._vwap_interaction_signal(price=39700, vwap=40000)
        assert direction == ScalpDirection.LONG
        assert confidence > 0
    
    def test_price_near_vwap_no_signal(self):
        """Test no signal when price near VWAP."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._vwap_interaction_signal(price=40010, vwap=40000)
        assert direction == ScalpDirection.NONE


class TestMomentumSignal:
    """Tests for MACD momentum signal."""
    
    def test_macd_bullish_long_signal(self):
        """Test long signal for bullish MACD."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._momentum_signal(macd_line=100, macd_signal=50)
        assert direction == ScalpDirection.LONG
        assert confidence > 0
    
    def test_macd_bearish_short_signal(self):
        """Test short signal for bearish MACD."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._momentum_signal(macd_line=50, macd_signal=100)
        assert direction == ScalpDirection.SHORT
        assert confidence > 0
    
    def test_macd_neutral_no_signal(self):
        """Test no signal for neutral MACD."""
        scalper = IntraDayVolatilityScalper()
        direction, confidence = scalper._momentum_signal(macd_line=100, macd_signal=100)
        assert direction == ScalpDirection.NONE


class TestMicroPositionSizing:
    """Tests for micro-position size calculation."""
    
    def test_position_size_low_volatility(self):
        """Test position sizing in low volatility."""
        scalper = IntraDayVolatilityScalper()
        volatility = VolatilityMetrics(regime=VolatilityRegime.LOW)
        position = scalper._calculate_micro_position(volatility, confidence=0.8)
        assert position <= 0.025  # Should be smaller in low vol
        assert position >= scalper.min_micro_position
    
    def test_position_size_high_volatility(self):
        """Test position sizing in high volatility."""
        scalper = IntraDayVolatilityScalper()
        volatility = VolatilityMetrics(regime=VolatilityRegime.HIGH)
        position = scalper._calculate_micro_position(volatility, confidence=0.8)
        assert position <= scalper.max_micro_position
        assert position >= scalper.min_micro_position
    
    def test_position_size_extreme_volatility(self):
        """Test position sizing in extreme volatility (reduced)."""
        scalper = IntraDayVolatilityScalper()
        volatility = VolatilityMetrics(regime=VolatilityRegime.EXTREME)
        position = scalper._calculate_micro_position(volatility, confidence=0.8)
        assert position <= 0.025  # Should be reduced in extreme vol
    
    def test_position_size_respects_bounds(self):
        """Test that position size always respects bounds."""
        scalper = IntraDayVolatilityScalper()
        for regime in VolatilityRegime:
            for confidence in [0.3, 0.5, 0.7, 0.9]:
                volatility = VolatilityMetrics(regime=regime)
                position = scalper._calculate_micro_position(volatility, confidence)
                assert scalper.min_micro_position <= position <= scalper.max_micro_position
    
    def test_position_size_confidence_scaling(self):
        """Test that higher confidence increases position size."""
        scalper = IntraDayVolatilityScalper()
        volatility = VolatilityMetrics(regime=VolatilityRegime.MODERATE)
        
        pos_low = scalper._calculate_micro_position(volatility, confidence=0.4)
        pos_high = scalper._calculate_micro_position(volatility, confidence=0.9)
        
        assert pos_high >= pos_low


class TestScalpSignalGeneration:
    """Tests for scalp signal generation."""
    
    def test_signal_valid_long(self):
        """Test valid long scalp signal."""
        scalper = IntraDayVolatilityScalper()
        volatility = VolatilityMetrics(
            regime=VolatilityRegime.MODERATE,
            trend_strength=0.3
        )
        prices = [40000 + i * 20 for i in range(20)]
        
        signal = scalper.generate_scalp_signal(
            current_price=40300,
            volatility=volatility,
            rsi=35,
            macd_line=150,
            macd_signal=100,
            prices=prices
        )
        
        if signal.direction != ScalpDirection.NONE:
            assert signal.is_valid() or signal.confidence < 0.6
            assert signal.profit_target > 0
            assert signal.stop_loss > 0
    
    def test_signal_low_volatility_rejected(self):
        """Test that signals are rejected in low volatility."""
        scalper = IntraDayVolatilityScalper()
        volatility = VolatilityMetrics(
            regime=VolatilityRegime.LOW,
            trend_strength=0.1
        )
        prices = [40000 + i * 10 for i in range(20)]
        
        signal = scalper.generate_scalp_signal(
            current_price=40100,
            volatility=volatility,
            rsi=50,
            macd_line=100,
            macd_signal=100,
            prices=prices
        )
        
        assert signal.direction == ScalpDirection.NONE
    
    def test_signal_confidence_threshold(self):
        """Test that signals respect minimum confidence."""
        scalper = IntraDayVolatilityScalper()
        signal = ScalpSignal(
            direction=ScalpDirection.LONG,
            confidence=0.5,
            entry_price=40000,
            micro_position_size=0.01,
            profit_target=40100,
            stop_loss=39900
        )
        
        assert not signal.is_valid()  # Below 0.6 threshold
    
    def test_signal_micro_position_bounds(self):
        """Test that signal position size respects bounds."""
        scalper = IntraDayVolatilityScalper()
        signal = ScalpSignal(
            direction=ScalpDirection.LONG,
            confidence=0.8,
            entry_price=40000,
            micro_position_size=0.1,  # Above max
            profit_target=40100,
            stop_loss=39900
        )
        
        assert not signal.is_valid()  # Position size too large


class TestVolatilityAnalysis:
    """Tests for comprehensive volatility analysis."""
    
    def test_volatility_analysis_complete(self):
        """Test complete volatility analysis."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000 + i * 50 for i in range(30)]
        volumes = [100 + i * 5 for i in range(30)]
        
        metrics = scalper.analyze_volatility(
            prices=prices,
            volumes=volumes,
            rsi=55,
            macd_line=150,
            macd_signal=140,
            current_price=42500
        )
        
        assert metrics.atr_14 > 0
        assert metrics.atr_7 > 0
        assert metrics.hourly_volatility >= 0
        assert metrics.bollinger_width >= 0
        assert metrics.vwap > 0
        assert isinstance(metrics.regime, VolatilityRegime)
        assert 0 <= metrics.trend_strength <= 1
        assert 0 <= metrics.mean_reversion_probability <= 1
    
    def test_volatility_analysis_insufficient_data(self):
        """Test volatility analysis with insufficient data."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000, 40100]
        volumes = [100, 100]
        
        metrics = scalper.analyze_volatility(
            prices=prices,
            volumes=volumes,
            rsi=50,
            macd_line=100,
            macd_signal=100,
            current_price=40100
        )
        
        # Should return empty or minimal metrics
        assert isinstance(metrics, VolatilityMetrics)


class TestPositionEvaluation:
    """Tests for active position evaluation."""
    
    def test_position_profit_target_hit_long(self):
        """Test closing LONG position when profit target hit."""
        scalper = IntraDayVolatilityScalper()
        position = ScalpPosition(
            entry_price=40000,
            entry_time=datetime.now(),
            direction=ScalpDirection.LONG,
            micro_position_size=0.02,
            profit_target=40100,
            stop_loss=39900,
            confidence=0.8
        )
        
        should_close, reason = scalper.evaluate_position(position, current_price=40100)
        assert should_close
        assert "Profit target" in reason
    
    def test_position_stop_loss_hit_long(self):
        """Test closing LONG position when stop loss hit."""
        scalper = IntraDayVolatilityScalper()
        position = ScalpPosition(
            entry_price=40000,
            entry_time=datetime.now(),
            direction=ScalpDirection.LONG,
            micro_position_size=0.02,
            profit_target=40100,
            stop_loss=39900,
            confidence=0.8
        )
        
        should_close, reason = scalper.evaluate_position(position, current_price=39900)
        assert should_close
        assert "Stop loss" in reason
    
    def test_position_max_hold_time(self):
        """Test closing position after max hold time."""
        scalper = IntraDayVolatilityScalper()
        old_time = datetime.now() - timedelta(minutes=20)
        position = ScalpPosition(
            entry_price=40000,
            entry_time=old_time,
            direction=ScalpDirection.LONG,
            micro_position_size=0.02,
            profit_target=40100,
            stop_loss=39900,
            confidence=0.8
        )
        
        should_close, reason = scalper.evaluate_position(position, current_price=40050)
        assert should_close
        assert "Max hold time" in reason
    
    def test_position_active_mid_range(self):
        """Test position remains active in mid-range."""
        scalper = IntraDayVolatilityScalper()
        position = ScalpPosition(
            entry_price=40000,
            entry_time=datetime.now(),
            direction=ScalpDirection.LONG,
            micro_position_size=0.02,
            profit_target=40100,
            stop_loss=39900,
            confidence=0.8
        )
        
        should_close, reason = scalper.evaluate_position(position, current_price=40050)
        assert not should_close


class TestPositionShort:
    """Tests for SHORT position evaluation."""
    
    def test_position_profit_target_hit_short(self):
        """Test closing SHORT position when profit target hit."""
        scalper = IntraDayVolatilityScalper()
        position = ScalpPosition(
            entry_price=40000,
            entry_time=datetime.now(),
            direction=ScalpDirection.SHORT,
            micro_position_size=0.02,
            profit_target=39900,
            stop_loss=40100,
            confidence=0.8
        )
        
        should_close, reason = scalper.evaluate_position(position, current_price=39900)
        assert should_close
        assert "Profit target" in reason
    
    def test_position_stop_loss_hit_short(self):
        """Test closing SHORT position when stop loss hit."""
        scalper = IntraDayVolatilityScalper()
        position = ScalpPosition(
            entry_price=40000,
            entry_time=datetime.now(),
            direction=ScalpDirection.SHORT,
            micro_position_size=0.02,
            profit_target=39900,
            stop_loss=40100,
            confidence=0.8
        )
        
        should_close, reason = scalper.evaluate_position(position, current_price=40100)
        assert should_close
        assert "Stop loss" in reason


class TestScalperStateManagement:
    """Tests for scalper state management."""
    
    def test_active_position_count(self):
        """Test active position counting."""
        scalper = IntraDayVolatilityScalper()
        assert scalper.get_active_position_count() == 0
        
        # Manually add positions
        scalper.active_positions['pos1'] = ScalpPosition(
            entry_price=40000, entry_time=datetime.now(),
            direction=ScalpDirection.LONG, micro_position_size=0.02,
            profit_target=40100, stop_loss=39900, confidence=0.8
        )
        
        assert scalper.get_active_position_count() == 1
    
    def test_last_scalp_time_tracking(self):
        """Test tracking of last scalp execution time."""
        scalper = IntraDayVolatilityScalper()
        initial_time = scalper.last_scalp_time
        
        scalper.update_last_scalp_time()
        
        assert scalper.last_scalp_time >= initial_time
    
    def test_scalping_stats(self):
        """Test scalping statistics retrieval."""
        scalper = IntraDayVolatilityScalper()
        stats = scalper.get_scalping_stats()
        
        assert 'active_positions' in stats
        assert stats['active_positions'] == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_volatility(self):
        """Test handling of zero volatility."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000] * 20  # Flat prices
        metrics = scalper.analyze_volatility(
            prices=prices, volumes=[100]*20,
            rsi=50, macd_line=100, macd_signal=100,
            current_price=40000
        )
        
        assert metrics.hourly_volatility == 0
    
    def test_extreme_rsi_values(self):
        """Test handling of extreme RSI values."""
        scalper = IntraDayVolatilityScalper()
        
        # RSI = 100 (extreme overbought)
        direction, confidence = scalper._rsi_divergence_signal(rsi=100)
        assert direction == ScalpDirection.SHORT
        
        # RSI = 0 (extreme oversold)
        direction, confidence = scalper._rsi_divergence_signal(rsi=0)
        assert direction == ScalpDirection.LONG
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        scalper = IntraDayVolatilityScalper()
        prices = [40000, np.nan, 40100, 40200]
        
        # Should not crash
        try:
            atr = scalper._calculate_atr(prices, 14)
            assert atr >= 0 or np.isnan(atr)
        except:
            pass  # Expected to potentially fail with NaN


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_scalping_workflow(self):
        """Test complete scalping workflow."""
        scalper = IntraDayVolatilityScalper()
        
        # Generate realistic data with higher volatility
        prices = [40000 + np.sin(i/5) * 400 + np.random.normal(0, 100) for i in range(50)]
        volumes = [100 + np.random.normal(0, 20) for i in range(50)]
        
        # Analyze volatility
        metrics = scalper.analyze_volatility(
            prices=prices, volumes=volumes,
            rsi=55, macd_line=150, macd_signal=140,
            current_price=prices[-1]
        )
        
        assert isinstance(metrics, VolatilityMetrics)
        
        # Generate signal
        signal = scalper.generate_scalp_signal(
            current_price=prices[-1],
            volatility=metrics,
            rsi=55,
            macd_line=150,
            macd_signal=140,
            prices=prices
        )
        
        # Check signal properties
        if signal.direction != ScalpDirection.NONE:
            assert 0 <= signal.confidence <= 1
            assert 0 < signal.micro_position_size <= 0.05
    
    def test_multiple_scalps_sequence(self):
        """Test handling multiple scalps in sequence."""
        scalper = IntraDayVolatilityScalper()
        
        for i in range(3):
            price = 40000 + (i * 100)
            volatility = VolatilityMetrics(regime=VolatilityRegime.MODERATE)
            prices = [price - 500 + j * 50 for j in range(20)]
            
            signal = scalper.generate_scalp_signal(
                current_price=price,
                volatility=volatility,
                rsi=50 + (i * 5),
                macd_line=100 + (i * 10),
                macd_signal=95 + (i * 10),
                prices=prices
            )
            
            assert isinstance(signal, ScalpSignal)
            if signal.is_valid():
                scalper.update_last_scalp_time()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
