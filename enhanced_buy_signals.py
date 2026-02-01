"""
Enhanced Buy Signal Detection - Weighted Scoring System

Implements intelligent dip detection with:
- Weighted signal scoring (not binary)
- Support/resistance level detection
- Opportunity quality metrics
- Dynamic threshold adjustment

PRINCIPLE: "Buy more BTC by detecting more high-quality dips"
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np
from logger_config import logger


class SignalStrength(Enum):
    """Signal strength classification."""
    EXTREME = "extreme"        # 9+ points (rare, maximum deployment)
    VERY_STRONG = "very_strong"  # 7-9 points (strong dip, large position)
    STRONG = "strong"          # 5-7 points (good dip, normal position)
    MODERATE = "moderate"      # 3-5 points (weak signal, small position)
    WEAK = "weak"              # 1-3 points (very weak, minimal position)
    NO_SIGNAL = "no_signal"    # < 1 point (don't buy)


@dataclass
class BuySignalAnalysis:
    """Detailed buy signal analysis with scoring breakdown."""
    total_score: float
    strength: SignalStrength
    components: dict  # Individual signal scores
    supports_nearby: List[float]  # Support levels within 5%
    resistance_above: float  # Nearest resistance
    dip_severity: float  # How deep the dip (0-100%)
    opportunity_quality: float  # 0-100%, quality metric
    recommendation: str  # Action recommendation
    
    def __str__(self) -> str:
        return (f"BuySignal(score={self.total_score:.1f}, "
                f"strength={self.strength.value}, "
                f"quality={self.opportunity_quality:.0f}%)")


class EnhancedBuySignalDetector:
    """
    Detects high-quality buy opportunities using weighted signal scoring.
    
    PUBLIC INTERFACE:
        analyze_buy_opportunity(indicators_data) -> BuySignalAnalysis
        
    PRIVATE IMPLEMENTATION:
        - Technical signal scoring
        - Sentiment analysis
        - On-chain metrics
        - Support/resistance detection
        - Opportunity quality ranking
    """
    
    def __init__(self):
        """Initialize signal detector with thresholds."""
        # Signal weights (sum to 10 for easy scaling)
        self.weights = {
            'extreme_oversold': 3.0,      # RSI < 25 = very strong
            'oversold': 2.0,              # RSI 25-35 = strong
            'dip_to_support': 2.5,        # Price near support
            'below_vwap': 1.5,            # Below volume-weighted avg
            'whale_accumulation': 2.0,    # Exchange netflow
            'bullish_cross': 1.5,         # MACD/MA crossovers
            'sentiment_stable': 1.0,      # Not panicking
            'no_crisis': 0.5,             # Risk-off not extreme
        }
        
        # Thresholds
        self.rsi_extreme_oversold = 25.0
        self.rsi_oversold = 35.0
        self.rsi_moderately_low = 45.0
        self.vwap_dip_threshold = 0.97  # 3% below VWAP
        self.netflow_whale_threshold = -5000
        self.netflow_strong_threshold = -8000
        self.risk_off_threshold_weak = 0.3
        self.risk_off_threshold_moderate = 0.6
        self.support_proximity = 0.05  # 5% distance
        
        # Signal quality bins
        self.quality_bins = {
            SignalStrength.EXTREME: (9.0, 100),
            SignalStrength.VERY_STRONG: (7.0, 99),
            SignalStrength.STRONG: (5.0, 85),
            SignalStrength.MODERATE: (3.0, 60),
            SignalStrength.WEAK: (1.0, 30),
            SignalStrength.NO_SIGNAL: (0.0, 0),
        }
    
    def analyze_buy_opportunity(
        self,
        indicators_data: dict,
        price_history: List[float]
    ) -> BuySignalAnalysis:
        """
        Comprehensive buy signal analysis with weighted scoring.
        
        Args:
            indicators_data: Technical indicators (RSI, MACD, etc)
            price_history: Historical prices for support detection
            
        Returns:
            BuySignalAnalysis with detailed breakdown
        """
        current_price = indicators_data.get('current_price', 0)
        
        # Calculate individual signal scores
        components = {
            'oversold': self._score_oversold_condition(indicators_data),
            'vwap_dip': self._score_vwap_dip(indicators_data),
            'whale_activity': self._score_whale_accumulation(indicators_data),
            'technical': self._score_technical_signals(indicators_data),
            'sentiment': self._score_sentiment(indicators_data),
            'risk_level': self._score_risk_level(indicators_data),
        }
        
        # Detect support/resistance
        supports = self._detect_support_levels(price_history, current_price)
        resistance = self._detect_resistance_level(price_history, current_price)
        nearby_supports = [s for s in supports if s > current_price * (1 - self.support_proximity)]
        
        # Calculate dip severity (how far below recent peak)
        peak_price = max(price_history[-100:]) if len(price_history) >= 100 else max(price_history)
        dip_severity = ((peak_price - current_price) / peak_price * 100) if peak_price > 0 else 0
        
        # Calculate total weighted score
        total_score = sum(components.values())
        
        # Determine signal strength
        strength = self._determine_strength(total_score)
        
        # Calculate opportunity quality (0-100)
        opportunity_quality = self._calculate_quality(components, dip_severity, nearby_supports)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(strength, opportunity_quality, dip_severity)
        
        analysis = BuySignalAnalysis(
            total_score=total_score,
            strength=strength,
            components=components,
            supports_nearby=nearby_supports,
            resistance_above=resistance,
            dip_severity=dip_severity,
            opportunity_quality=opportunity_quality,
            recommendation=recommendation,
        )
        
        logger.info(f"🎯 Buy Signal Analysis: {analysis}")
        logger.debug(f"   Components: {components}")
        
        return analysis
    
    def _score_oversold_condition(self, indicators_data: dict) -> float:
        """Score RSI oversold condition (0-3 points)."""
        rsi = indicators_data.get('rsi', 50)
        
        if rsi < self.rsi_extreme_oversold:
            score = self.weights['extreme_oversold']
            logger.debug(f"   🔥 Extreme oversold RSI {rsi:.1f} → {score} pts")
        elif rsi < self.rsi_oversold:
            score = self.weights['oversold']
            logger.debug(f"   📉 Oversold RSI {rsi:.1f} → {score} pts")
        elif rsi < self.rsi_moderately_low:
            score = 1.0  # Weak signal
            logger.debug(f"   ↘️ Moderately low RSI {rsi:.1f} → 1.0 pts")
        else:
            score = 0.0
            
        return score
    
    def _score_vwap_dip(self, indicators_data: dict) -> float:
        """Score price position relative to VWAP (0-2.5 points)."""
        current_price = indicators_data.get('current_price', 0)
        vwap = indicators_data.get('vwap', current_price)
        
        if vwap == 0:
            return 0.0
        
        price_to_vwap_ratio = current_price / vwap
        
        if price_to_vwap_ratio < 0.93:  # 7%+ below VWAP
            score = self.weights['dip_to_support']
            logger.debug(f"   📍 Deep dip {price_to_vwap_ratio:.3f} vs VWAP → {score} pts")
        elif price_to_vwap_ratio < self.vwap_dip_threshold:  # 3% below
            score = self.weights['dip_to_support'] * 0.7
            logger.debug(f"   📍 Moderate dip {price_to_vwap_ratio:.3f} vs VWAP → {score:.1f} pts")
        else:
            score = 0.0
            
        return score
    
    def _score_whale_accumulation(self, indicators_data: dict) -> float:
        """Score whale/exchange netflow (0-2 points)."""
        netflow = indicators_data.get('netflow', 0)
        
        if netflow < self.netflow_strong_threshold:  # Very strong accumulation
            score = self.weights['whale_accumulation']
            logger.debug(f"   🐋 Strong whale accumulation {netflow:.0f} → {score} pts")
        elif netflow < self.netflow_whale_threshold:  # Moderate accumulation
            score = self.weights['whale_accumulation'] * 0.7
            logger.debug(f"   🐳 Moderate whale activity {netflow:.0f} → {score:.1f} pts")
        else:
            score = 0.0
            
        return score
    
    def _score_technical_signals(self, indicators_data: dict) -> float:
        """Score technical indicators (0-1.5 points)."""
        rsi = indicators_data.get('rsi', 50)
        macd = indicators_data.get('macd', 0)
        signal = indicators_data.get('signal', 0)
        ma_short = indicators_data.get('ma_short', 0)
        ma_long = indicators_data.get('ma_long', 0)
        
        score = 0.0
        
        # MACD bullish
        if macd > signal and abs(macd - signal) < 10:
            score += 0.7
            logger.debug(f"   📊 MACD bullish → +0.7 pts")
        
        # MA alignment (price above short MA)
        if ma_short > 0 and indicators_data.get('current_price', 0) > ma_short:
            score += 0.5
            logger.debug(f"   📈 Price above short MA → +0.5 pts")
        
        # Potential reversal (oversold but MACD recovering)
        if rsi < 40 and macd > signal:
            score += 0.3
            logger.debug(f"   🔄 Reversal setup → +0.3 pts")
        
        return min(score, self.weights['bullish_cross'])
    
    def _score_sentiment(self, indicators_data: dict) -> float:
        """Score sentiment and news (0-1 point)."""
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off = news_analysis.get('risk_off_probability', 0)
        sentiment = indicators_data.get('sentiment', 0)
        
        # Positive if not in panic mode
        if risk_off < self.risk_off_threshold_weak and sentiment > -0.2:
            score = self.weights['sentiment_stable']
            logger.debug(f"   😊 Sentiment stable → {score} pts")
        else:
            score = 0.0
            
        return score
    
    def _score_risk_level(self, indicators_data: dict) -> float:
        """Score systemic risk level (0-0.5 point)."""
        news_analysis = indicators_data.get('news_analysis', {})
        risk_off = news_analysis.get('risk_off_probability', 0)
        
        # Small bonus if no crisis
        if risk_off < self.risk_off_threshold_weak:
            score = self.weights['no_crisis']
            logger.debug(f"   ✅ Low systemic risk → {score} pts")
        else:
            score = 0.0
            
        return score
    
    def _detect_support_levels(
        self,
        price_history: List[float],
        current_price: float
    ) -> List[float]:
        """Detect support levels from price history."""
        if len(price_history) < 20:
            return []
        
        supports = []
        
        # Find local minima in last 100 candles
        history = price_history[-100:] if len(price_history) >= 100 else price_history
        
        for i in range(1, len(history) - 1):
            if history[i] < history[i-1] and history[i] < history[i+1]:
                # Local minimum found
                support = history[i]
                # Only add if below current price
                if support < current_price * 0.99:
                    supports.append(support)
        
        # Also add moving average as support
        ma_50 = np.mean(history[-50:]) if len(history) >= 50 else np.mean(history)
        if ma_50 < current_price:
            supports.append(ma_50)
        
        # Remove duplicates and sort
        supports = sorted(list(set([round(s, 2) for s in supports])), reverse=True)
        
        return supports[:3]  # Return top 3
    
    def _detect_resistance_level(
        self,
        price_history: List[float],
        current_price: float
    ) -> float:
        """Detect nearest resistance level above current price."""
        if len(price_history) < 20:
            return current_price * 1.10  # Default: 10% above
        
        history = price_history[-100:] if len(price_history) >= 100 else price_history
        
        # Find local maxima
        resistances = []
        for i in range(1, len(history) - 1):
            if history[i] > history[i-1] and history[i] > history[i+1]:
                resistance = history[i]
                if resistance > current_price * 1.01:
                    resistances.append(resistance)
        
        # Also add 50-period MA + 2% as resistance
        ma_50 = np.mean(history[-50:]) if len(history) >= 50 else np.mean(history)
        resistances.append(ma_50 * 1.02)
        
        # Return closest resistance above price
        resistances = sorted([r for r in resistances if r > current_price])
        return resistances[0] if resistances else current_price * 1.10
    
    def _determine_strength(self, score: float) -> SignalStrength:
        """Determine signal strength from score."""
        for strength, (threshold, _) in self.quality_bins.items():
            if score >= threshold:
                return strength
        return SignalStrength.NO_SIGNAL
    
    def _calculate_quality(
        self,
        components: dict,
        dip_severity: float,
        nearby_supports: List[float]
    ) -> float:
        """Calculate opportunity quality score (0-100)."""
        quality = 0.0
        
        # Component quality
        quality += min(components['oversold'] / 3.0, 1.0) * 30  # Up to 30%
        quality += min(components['vwap_dip'] / 2.5, 1.0) * 20  # Up to 20%
        quality += min(components['whale_activity'] / 2.0, 1.0) * 15  # Up to 15%
        quality += min(components['technical'] / 1.5, 1.0) * 15  # Up to 15%
        quality += min(components['sentiment'] / 1.0, 1.0) * 10  # Up to 10%
        quality += min(components['risk_level'] / 0.5, 1.0) * 5  # Up to 5%
        
        # Dip severity bonus (deeper = better opportunity)
        quality += min(dip_severity / 5.0, 1.0) * 5  # Up to 5%
        
        # Support proximity bonus (closer to support = safer)
        if nearby_supports:
            closest_support = nearby_supports[0]
            distance_to_support = (closest_support - 0) / closest_support  # Lower = closer
            quality += (1 - min(distance_to_support, 1.0)) * 5  # Up to 5%
        
        return min(quality, 100.0)
    
    def _generate_recommendation(
        self,
        strength: SignalStrength,
        quality: float,
        dip_severity: float
    ) -> str:
        """Generate action recommendation."""
        if strength == SignalStrength.EXTREME:
            return (f"🚀 EXTREME DIP ({dip_severity:.1f}%) - "
                    f"Maximum position (All-in on this level!)")
        elif strength == SignalStrength.VERY_STRONG:
            return f"💪 Very strong dip - Large position (80-100%)"
        elif strength == SignalStrength.STRONG:
            return f"✅ Strong dip - Normal position (50-80%)"
        elif strength == SignalStrength.MODERATE:
            return f"⏸️ Moderate signal - Small position (20-50%)"
        elif strength == SignalStrength.WEAK:
            return f"🤔 Weak signal - Minimal position (5-20%)"
        else:
            return "❌ No buy signal - Wait for better opportunity"
