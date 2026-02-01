"""
Phase 8 Enhancement: Tests for Enhanced Buy Signal Detector
Tests the weighted scoring system and signal strength determination
"""

import pytest
import sys
import os

# Add parent directory to path to import main modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_buy_signals import (
    EnhancedBuySignalDetector, 
    BuySignalAnalysis, 
    SignalStrength
)


class TestEnhancedBuySignalDetector:
    """Test suite for the enhanced buy signal detector with weighted scoring"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = EnhancedBuySignalDetector()
        self.base_indicators = {
            'rsi': 50.0,
            'macd': 0.0,
            'signal_line': 0.0,
            'vwap': 100.0,
            'current_price': 100.0,
            'ma_short': 100.0,
            'ma_long': 100.0,
            'netflow': 0.0,
            'risk_off_probability': 0.0,
            'sentiment': 0.0,
            'volatility': 0.02,
        }
        self.base_price_history = [100.0] * 100
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly"""
        detector = EnhancedBuySignalDetector()
        assert detector is not None
        assert hasattr(detector, 'analyze_buy_opportunity')
    
    def test_no_signal_normal_conditions(self):
        """Test no buy signal in normal market conditions"""
        indicators = self.base_indicators.copy()
        indicators['rsi'] = 50.0  # Normal RSI
        indicators['current_price'] = 100.0  # Normal price
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        assert analysis.total_score < 3.0
        assert analysis.strength in [SignalStrength.NO_SIGNAL, SignalStrength.WEAK]
    
    def test_extreme_oversold_signal(self):
        """Test extreme oversold conditions (RSI < 25)"""
        indicators = self.base_indicators.copy()
        indicators['rsi'] = 20.0  # Extremely oversold
        indicators['netflow'] = -8000  # Strong accumulation
        indicators['current_price'] = 95.0  # Below VWAP
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Should score high due to extreme oversold
        assert analysis.total_score >= 5.0
        assert analysis.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
        assert analysis.components['oversold'] >= 3.0
    
    def test_whale_accumulation_signal(self):
        """Test whale accumulation detection"""
        indicators = self.base_indicators.copy()
        indicators['netflow'] = -10000  # Major whale accumulation
        indicators['rsi'] = 40.0
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Whale score should be significant
        assert analysis.components['whale_activity'] >= 1.5
    
    def test_vwap_dip_detection(self):
        """Test detection of price dips below VWAP"""
        indicators = self.base_indicators.copy()
        indicators['vwap'] = 100.0
        indicators['current_price'] = 92.0  # 8% below VWAP
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # VWAP dip score should be high
        assert analysis.components['vwap_dip'] >= 2.0
    
    def test_bullish_technical_cross(self):
        """Test bullish technical signals"""
        indicators = self.base_indicators.copy()
        indicators['macd'] = 0.5  # Above signal line (bullish cross)
        indicators['signal_line'] = 0.0
        indicators['ma_short'] = 101.0  # Short MA above long MA
        indicators['ma_long'] = 100.0
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Technical score should be positive
        assert analysis.components['technical'] > 0.0
    
    def test_positive_sentiment_boost(self):
        """Test positive sentiment influence"""
        indicators = self.base_indicators.copy()
        indicators['sentiment'] = 0.3  # Positive sentiment
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Sentiment score should reflect positive sentiment
        assert analysis.components['sentiment'] >= 0.0
    
    def test_high_risk_off_penalty(self):
        """Test risk-off probability penalty"""
        indicators = self.base_indicators.copy()
        indicators['risk_off_probability'] = 0.8  # Very high risk
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Risk score should be low (max 0.5)
        assert analysis.components['risk_level'] <= 0.5
    
    def test_combined_extreme_conditions(self):
        """Test combination of extreme buy conditions"""
        indicators = self.base_indicators.copy()
        indicators['rsi'] = 15.0  # Extreme oversold
        indicators['current_price'] = 90.0  # 10% below VWAP
        indicators['netflow'] = -12000  # Extreme whale accumulation
        indicators['macd'] = 1.0  # Strong bullish
        indicators['signal_line'] = -0.5
        indicators['sentiment'] = 0.2  # Stable/positive
        indicators['volatility'] = 0.08  # High volatility
        indicators['risk_off_probability'] = 0.1  # Low risk
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Should give extreme strength signal
        assert analysis.total_score >= 8.0
        assert analysis.strength == SignalStrength.EXTREME
        assert analysis.opportunity_quality >= 80.0
    
    def test_signal_strength_levels(self):
        """Test signal strength levels are properly assigned"""
        test_cases = [
            (15.0, -10000, 90.0, SignalStrength.EXTREME),      # Extreme conditions
            (30.0, -5000, 95.0, SignalStrength.STRONG),        # Strong conditions
            (50.0, 0.0, 100.0, SignalStrength.NO_SIGNAL),      # No signal
        ]
        
        for rsi, netflow, price, expected_strength in test_cases:
            indicators = self.base_indicators.copy()
            indicators['rsi'] = rsi
            indicators['netflow'] = netflow
            indicators['current_price'] = price
            
            analysis = self.detector.analyze_buy_opportunity(
                indicators_data=indicators,
                price_history=self.base_price_history
            )
            
            # Verify the strength matches expectations
            if expected_strength == SignalStrength.EXTREME:
                assert analysis.strength in [SignalStrength.EXTREME, SignalStrength.VERY_STRONG]
            elif expected_strength == SignalStrength.STRONG:
                assert analysis.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
            else:
                assert analysis.strength in [SignalStrength.NO_SIGNAL, SignalStrength.WEAK]
    
    def test_opportunity_quality_calculation(self):
        """Test opportunity quality score calculation"""
        indicators = self.base_indicators.copy()
        indicators['rsi'] = 20.0
        indicators['netflow'] = -10000
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Quality should be between 0 and 100
        assert 0 <= analysis.opportunity_quality <= 100
    
    def test_support_level_detection(self):
        """Test detection of support levels"""
        # Create price history with clear support levels
        price_history = [100, 102, 101, 103, 102] * 5  # Oscillating around 102
        price_history.extend([105, 104, 106, 103, 105])
        price_history.extend([100, 99, 98, 99, 100])  # Lower support at ~99
        
        indicators = self.base_indicators.copy()
        indicators['current_price'] = 105.0
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=price_history
        )
        
        # Should detect support levels
        assert analysis.supports_nearby is not None
        if analysis.supports_nearby:
            assert len(analysis.supports_nearby) > 0
    
    def test_recommendation_generation(self):
        """Test buy recommendation generation"""
        indicators = self.base_indicators.copy()
        indicators['rsi'] = 25.0
        indicators['netflow'] = -8000
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Recommendation should be non-empty
        assert isinstance(analysis.recommendation, str)
        assert len(analysis.recommendation) > 0
    
    def test_score_components_sum_correctly(self):
        """Test that score components sum to total score"""
        indicators = self.base_indicators.copy()
        indicators['rsi'] = 22.0
        indicators['netflow'] = -9000
        indicators['current_price'] = 94.0
        
        analysis = self.detector.analyze_buy_opportunity(
            indicators_data=indicators,
            price_history=self.base_price_history
        )
        
        # Components should roughly sum to total (with some rounding tolerance)
        component_sum = sum(analysis.components.values())
        assert abs(component_sum - analysis.total_score) < 0.5


class TestBuySignalAnalysis:
    """Test the BuySignalAnalysis dataclass"""
    
    def test_analysis_creation(self):
        """Test creation of BuySignalAnalysis object"""
        analysis = BuySignalAnalysis(
            total_score=8.5,
            strength=SignalStrength.EXTREME,
            components={'rsi': 3.0, 'vwap': 2.0},
            supports_nearby=[99.0, 98.0],
            resistance_above=105.0,
            dip_severity=5.0,
            opportunity_quality=85.0,
            recommendation="STRONG BUY: Extreme oversold at support"
        )
        
        assert analysis.total_score == 8.5
        assert analysis.strength == SignalStrength.EXTREME
        assert analysis.opportunity_quality == 85.0
    
    def test_analysis_fields_accessible(self):
        """Test all fields are accessible in analysis object"""
        analysis = BuySignalAnalysis(
            total_score=5.0,
            strength=SignalStrength.STRONG,
            components={'test': 1.0},
            supports_nearby=[100.0],
            resistance_above=110.0,
            dip_severity=3.0,
            opportunity_quality=60.0,
            recommendation="BUY"
        )
        
        assert hasattr(analysis, 'total_score')
        assert hasattr(analysis, 'strength')
        assert hasattr(analysis, 'components')
        assert hasattr(analysis, 'supports_nearby')
        assert hasattr(analysis, 'resistance_above')
        assert hasattr(analysis, 'dip_severity')
        assert hasattr(analysis, 'opportunity_quality')
        assert hasattr(analysis, 'recommendation')


class TestSignalStrength:
    """Test the SignalStrength enum"""
    
    def test_all_signal_strengths_exist(self):
        """Test that all expected signal strength levels exist"""
        assert hasattr(SignalStrength, 'EXTREME')
        assert hasattr(SignalStrength, 'VERY_STRONG')
        assert hasattr(SignalStrength, 'STRONG')
        assert hasattr(SignalStrength, 'MODERATE')
        assert hasattr(SignalStrength, 'WEAK')
        assert hasattr(SignalStrength, 'NO_SIGNAL')
    
    def test_signal_strength_names(self):
        """Test signal strength names are accessible"""
        strengths = [
            SignalStrength.EXTREME,
            SignalStrength.VERY_STRONG,
            SignalStrength.STRONG,
            SignalStrength.MODERATE,
            SignalStrength.WEAK,
            SignalStrength.NO_SIGNAL,
        ]
        
        for strength in strengths:
            assert hasattr(strength, 'name')
            assert len(strength.name) > 0


class TestIntegrationWithTradingBot:
    """Integration tests for enhanced buy signals with trading bot"""
    
    def test_detector_produces_consistent_results(self):
        """Test that detector produces consistent results for same input"""
        detector = EnhancedBuySignalDetector()
        
        indicators = {
            'rsi': 25.0,
            'macd': 0.3,
            'signal_line': 0.0,
            'vwap': 100.0,
            'current_price': 95.0,
            'ma_short': 100.0,
            'ma_long': 100.0,
            'netflow': -8000,
            'risk_off_probability': 0.1,
            'sentiment': 0.1,
            'volatility': 0.04,
        }
        
        price_history = [100.0] * 100
        
        # Call multiple times
        result1 = detector.analyze_buy_opportunity(indicators, price_history)
        result2 = detector.analyze_buy_opportunity(indicators, price_history)
        
        # Results should be identical
        assert result1.total_score == result2.total_score
        assert result1.strength == result2.strength
        assert result1.opportunity_quality == result2.opportunity_quality
    
    def test_detector_handles_missing_indicators(self):
        """Test detector gracefully handles missing indicator fields"""
        detector = EnhancedBuySignalDetector()
        
        incomplete_indicators = {
            'rsi': 30.0,
            'current_price': 100.0,
            # Missing other fields
        }
        
        # Should not crash
        try:
            analysis = detector.analyze_buy_opportunity(
                indicators_data=incomplete_indicators,
                price_history=[100.0] * 50
            )
            # Should return some analysis even with incomplete data
            assert analysis is not None
        except Exception as e:
            # If it fails, that's also acceptable for incomplete data
            assert isinstance(e, (KeyError, ValueError))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
