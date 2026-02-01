"""
Phase 8 Task 4: Support/Resistance Framework
Detects, tracks, and optimizes trading around support/resistance levels

Author: Phase 8 Optimization
Date: February 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
import logging
from statistics import mean, stdev

logger = logging.getLogger("trading_bot")


class LevelType(Enum):
    """Classification of support/resistance levels"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    BOTH = "both"  # Both support and resistance


class LevelStrength(Enum):
    """Reliability rating of a level"""
    WEAK = 1        # Touched 1-2 times
    MODERATE = 2    # Touched 3-4 times
    STRONG = 3      # Touched 5+ times
    CRITICAL = 4    # Tested 8+ times, highly reliable


class DetectionMethod(Enum):
    """How the level was detected"""
    PIVOT_POINT = "pivot_point"
    ROUND_NUMBER = "round_number"
    TREND_LINE = "trend_line"
    CONSOLIDATION = "consolidation"
    HISTORICAL_SWING = "historical_swing"
    FIBONACCI = "fibonacci"


@dataclass
class SupportResistanceLevel:
    """Individual support/resistance level"""
    price: float
    level_type: LevelType
    strength: LevelStrength
    detection_method: DetectionMethod
    touches: int = 1  # Number of times price has touched this level
    last_touched_index: int = 0  # Candle index when last touched
    formation_index: int = 0  # Candle index when first detected
    breakout_count: int = 0  # Times price broke through
    hold_count: int = 0  # Times price held at this level
    confidence_score: float = 0.0  # 0-100% reliability
    
    def __post_init__(self):
        """Calculate confidence score based on strength and touches"""
        base_score = self.strength.value * 15  # 15, 30, 45, 60
        touch_bonus = min(self.touches * 5, 30)  # Max +30
        self.confidence_score = min(100.0, base_score + touch_bonus)


@dataclass
class LevelAnalysis:
    """Analysis results for current price vs support/resistance"""
    current_price: float
    nearest_support: Optional[SupportResistanceLevel]
    nearest_resistance: Optional[SupportResistanceLevel]
    distance_to_support_pct: float  # How far (%) from support
    distance_to_resistance_pct: float  # How far (%) from resistance
    support_strength: int = 0  # 1-4 rating
    resistance_strength: int = 0  # 1-4 rating
    reward_risk_ratio: float = 1.0  # Resistance distance / Support distance
    is_near_support: bool = False  # Within 2% of support
    is_near_resistance: bool = False  # Within 2% of resistance
    breakout_probability: float = 0.0  # 0-1.0 estimated breakout chance


@dataclass
class PivotLevels:
    """Daily/weekly pivot point calculations"""
    pivot_point: float
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    high: float
    low: float
    close: float


class SupportResistanceDetector:
    """
    Detects and tracks support/resistance levels using multiple methods.
    
    Strategy:
    - Pivot point analysis (classic technical method)
    - Round number detection (psychological levels)
    - Trend line calculation (linear regression)
    - Consolidation zones (sideways movement detection)
    - Historical swing analysis (past highs/lows)
    - Fibonacci retracements (golden ratio levels)
    
    Benefits:
    - Identifies optimal entry/exit points
    - Improves win rate through level-based trading
    - Provides reward:risk ratio calculation
    - Tracks level reliability over time
    - Detects breakout opportunities
    """
    
    # Detection sensitivity
    TOLERANCE_PCT = 0.5  # Consider prices within 0.5% as "touching" level
    NEAR_LEVEL_PCT = 2.0  # Consider prices within 2% as "near" level
    CONSOLIDATION_LOOKBACK = 5  # Candles to check for consolidation
    SIGNIFICANT_MOVE_PCT = 2.0  # Move of 2%+ is significant
    
    def __init__(self):
        """Initialize support/resistance detector"""
        self.logger = logging.getLogger("trading_bot")
        self.levels: Dict[float, SupportResistanceLevel] = {}
        self.level_history: List[SupportResistanceLevel] = []
    
    def detect_levels(
        self,
        price_history: List[float],
        current_price: float,
        period: int = 50
    ) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """
        Detect support and resistance levels from price history.
        
        Args:
            price_history: List of recent prices (oldest to newest)
            current_price: Current BTC price
            period: Number of candles to analyze
        
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if not price_history or len(price_history) < 5:
            return [], []
        
        support_levels = []
        resistance_levels = []
        
        # 1. Detect pivot points
        pivot_levels = self._calculate_pivot_points(price_history[-1:], price_history)
        if pivot_levels:
            support_levels.extend(self._levels_from_pivot(pivot_levels, period))
            resistance_levels.extend(self._levels_from_pivot(pivot_levels, period, is_resistance=True))
        
        # 2. Detect round numbers
        round_levels = self._detect_round_numbers(price_history, period)
        for level in round_levels:
            if level.price < current_price:
                support_levels.append(level)
            elif level.price > current_price:
                resistance_levels.append(level)
        
        # 3. Detect trend lines
        trend_levels = self._detect_trend_lines(price_history, period)
        support_levels.extend([l for l in trend_levels if l.price < current_price])
        resistance_levels.extend([l for l in trend_levels if l.price > current_price])
        
        # 4. Detect consolidation zones
        consolidation_levels = self._detect_consolidation_zones(price_history)
        for level in consolidation_levels:
            if level.price < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
        
        # 5. Detect historical swings
        swing_levels = self._detect_historical_swings(price_history, period)
        support_levels.extend([l for l in swing_levels if l.price < current_price])
        resistance_levels.extend([l for l in swing_levels if l.price > current_price])
        
        # 6. Detect fibonacci levels
        fib_levels = self._detect_fibonacci_levels(price_history, period)
        for level in fib_levels:
            if level.price < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
        
        # Remove duplicates and clean up
        support_levels = self._deduplicate_and_rank(support_levels)
        resistance_levels = self._deduplicate_and_rank(resistance_levels)
        
        return support_levels, resistance_levels
    
    def analyze_current_position(
        self,
        current_price: float,
        support_levels: List[SupportResistanceLevel],
        resistance_levels: List[SupportResistanceLevel]
    ) -> LevelAnalysis:
        """
        Analyze current price position relative to levels.
        
        Args:
            current_price: Current BTC price
            support_levels: Detected support levels
            resistance_levels: Detected resistance levels
        
        Returns:
            LevelAnalysis with detailed position information
        """
        nearest_support = None
        nearest_resistance = None
        min_support_distance = float('inf')
        min_resistance_distance = float('inf')
        
        # Find nearest support (below current price)
        for level in support_levels:
            distance = current_price - level.price
            if distance > 0 and distance < min_support_distance:
                min_support_distance = distance
                nearest_support = level
        
        # Find nearest resistance (above current price)
        for level in resistance_levels:
            distance = level.price - current_price
            if distance > 0 and distance < min_resistance_distance:
                min_resistance_distance = distance
                nearest_resistance = level
        
        # Calculate distances as percentages
        distance_to_support_pct = (min_support_distance / current_price * 100) if nearest_support else 0
        distance_to_resistance_pct = (min_resistance_distance / current_price * 100) if nearest_resistance else 0
        
        # Calculate reward:risk ratio
        if nearest_support and nearest_resistance:
            risk = current_price - nearest_support.price
            reward = nearest_resistance.price - current_price
            reward_risk_ratio = reward / risk if risk > 0 else 1.0
        else:
            reward_risk_ratio = 1.0
        
        # Check if near levels
        is_near_support = distance_to_support_pct <= self.NEAR_LEVEL_PCT if nearest_support else False
        is_near_resistance = distance_to_resistance_pct <= self.NEAR_LEVEL_PCT if nearest_resistance else False
        
        # Estimate breakout probability
        breakout_prob = self._estimate_breakout_probability(
            current_price,
            nearest_support,
            nearest_resistance,
            distance_to_support_pct,
            distance_to_resistance_pct
        )
        
        return LevelAnalysis(
            current_price=current_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            distance_to_support_pct=distance_to_support_pct,
            distance_to_resistance_pct=distance_to_resistance_pct,
            support_strength=nearest_support.strength.value if nearest_support else 0,
            resistance_strength=nearest_resistance.strength.value if nearest_resistance else 0,
            reward_risk_ratio=reward_risk_ratio,
            is_near_support=is_near_support,
            is_near_resistance=is_near_resistance,
            breakout_probability=breakout_prob
        )
    
    def _calculate_pivot_points(
        self,
        recent_close: List[float],
        full_history: List[float]
    ) -> Optional[PivotLevels]:
        """Calculate classic pivot points from price data"""
        if len(full_history) < 1:
            return None
        
        high = max(full_history[-20:]) if len(full_history) >= 20 else max(full_history)
        low = min(full_history[-20:]) if len(full_history) >= 20 else min(full_history)
        close = recent_close[0] if recent_close else full_history[-1]
        
        pivot = (high + low + close) / 3.0
        resistance_1 = pivot * 2 - low
        resistance_2 = pivot + (high - low)
        support_1 = pivot * 2 - high
        support_2 = pivot - (high - low)
        
        return PivotLevels(
            pivot_point=pivot,
            support_1=support_1,
            support_2=support_2,
            resistance_1=resistance_1,
            resistance_2=resistance_2,
            high=high,
            low=low,
            close=close
        )
    
    def _levels_from_pivot(
        self,
        pivot: PivotLevels,
        period: int,
        is_resistance: bool = False
    ) -> List[SupportResistanceLevel]:
        """Extract support/resistance levels from pivot calculation"""
        levels = []
        
        if is_resistance:
            for price, method in [
                (pivot.resistance_1, DetectionMethod.PIVOT_POINT),
                (pivot.resistance_2, DetectionMethod.PIVOT_POINT)
            ]:
                level = SupportResistanceLevel(
                    price=price,
                    level_type=LevelType.RESISTANCE,
                    strength=LevelStrength.MODERATE,
                    detection_method=method,
                    touches=1,
                    formation_index=period - 1
                )
                levels.append(level)
        else:
            for price, method in [
                (pivot.support_1, DetectionMethod.PIVOT_POINT),
                (pivot.support_2, DetectionMethod.PIVOT_POINT)
            ]:
                level = SupportResistanceLevel(
                    price=price,
                    level_type=LevelType.SUPPORT,
                    strength=LevelStrength.MODERATE,
                    detection_method=method,
                    touches=1,
                    formation_index=period - 1
                )
                levels.append(level)
        
        return levels
    
    def _detect_round_numbers(
        self,
        price_history: List[float],
        period: int
    ) -> List[SupportResistanceLevel]:
        """Detect psychologically significant round numbers"""
        levels = []
        
        if not price_history:
            return levels
        
        # Average price determines scale of round numbers
        avg_price = mean(price_history[-period:] if len(price_history) >= period else price_history)
        
        # Determine rounding scale
        if avg_price < 100:
            scale = 10
        elif avg_price < 1000:
            scale = 100
        elif avg_price < 10000:
            scale = 1000
        else:
            scale = 5000
        
        # Find round numbers nearby
        for price in price_history[-period:]:
            rounded = round(price / scale) * scale
            
            # Check if round number is used in actual price history
            for p in price_history[-period:]:
                if abs(p - rounded) / rounded < (self.TOLERANCE_PCT / 100):
                    level = SupportResistanceLevel(
                        price=rounded,
                        level_type=LevelType.BOTH,
                        strength=LevelStrength.WEAK,
                        detection_method=DetectionMethod.ROUND_NUMBER,
                        touches=1,
                        formation_index=period - 1
                    )
                    levels.append(level)
                    break
        
        return levels
    
    def _detect_trend_lines(
        self,
        price_history: List[float],
        period: int
    ) -> List[SupportResistanceLevel]:
        """Detect support/resistance using linear regression trend lines"""
        levels = []
        
        if len(price_history) < 10:
            return levels
        
        # Use recent data
        data = price_history[-period:] if len(price_history) >= period else price_history
        
        # Linear regression for uptrend support
        x_vals = list(range(len(data)))
        y_vals = data
        
        if len(x_vals) >= 2:
            n = len(x_vals)
            x_mean = mean(x_vals)
            y_mean = mean(y_vals)
            
            numerator = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n))
            denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
            
            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                # Calculate trend line values
                trend_start = intercept
                trend_end = slope * (len(data) - 1) + intercept
                
                # Use as support/resistance
                if slope > 0:  # Uptrend
                    level = SupportResistanceLevel(
                        price=trend_start,
                        level_type=LevelType.SUPPORT,
                        strength=LevelStrength.MODERATE,
                        detection_method=DetectionMethod.TREND_LINE,
                        touches=1,
                        formation_index=period - 1
                    )
                else:  # Downtrend
                    level = SupportResistanceLevel(
                        price=trend_end,
                        level_type=LevelType.RESISTANCE,
                        strength=LevelStrength.MODERATE,
                        detection_method=DetectionMethod.TREND_LINE,
                        touches=1,
                        formation_index=period - 1
                    )
                
                levels.append(level)
        
        return levels
    
    def _detect_consolidation_zones(
        self,
        price_history: List[float]
    ) -> List[SupportResistanceLevel]:
        """Detect sideways consolidation zones"""
        levels = []
        
        if len(price_history) < self.CONSOLIDATION_LOOKBACK:
            return levels
        
        # Check recent prices for consolidation
        recent = price_history[-self.CONSOLIDATION_LOOKBACK:]
        high = max(recent)
        low = min(recent)
        price_range = high - low
        avg_price = mean(recent)
        
        # If range is small relative to price, it's consolidation
        if price_range / avg_price < 0.02:  # Less than 2% range
            level_high = SupportResistanceLevel(
                price=high,
                level_type=LevelType.RESISTANCE,
                strength=LevelStrength.STRONG,
                detection_method=DetectionMethod.CONSOLIDATION,
                touches=1,
                formation_index=len(price_history) - 1
            )
            level_low = SupportResistanceLevel(
                price=low,
                level_type=LevelType.SUPPORT,
                strength=LevelStrength.STRONG,
                detection_method=DetectionMethod.CONSOLIDATION,
                touches=1,
                formation_index=len(price_history) - 1
            )
            levels.extend([level_high, level_low])
        
        return levels
    
    def _detect_historical_swings(
        self,
        price_history: List[float],
        period: int
    ) -> List[SupportResistanceLevel]:
        """Detect levels from historical swing highs/lows"""
        levels = []
        
        if len(price_history) < 5:
            return levels
        
        data = price_history[-period:] if len(price_history) >= period else price_history
        
        # Find local highs and lows
        for i in range(1, len(data) - 1):
            # Local high
            if data[i] > data[i-1] and data[i] > data[i+1]:
                level = SupportResistanceLevel(
                    price=data[i],
                    level_type=LevelType.RESISTANCE,
                    strength=LevelStrength.MODERATE,
                    detection_method=DetectionMethod.HISTORICAL_SWING,
                    touches=1,
                    formation_index=period - (len(data) - i)
                )
                levels.append(level)
            
            # Local low
            elif data[i] < data[i-1] and data[i] < data[i+1]:
                level = SupportResistanceLevel(
                    price=data[i],
                    level_type=LevelType.SUPPORT,
                    strength=LevelStrength.MODERATE,
                    detection_method=DetectionMethod.HISTORICAL_SWING,
                    touches=1,
                    formation_index=period - (len(data) - i)
                )
                levels.append(level)
        
        return levels
    
    def _detect_fibonacci_levels(
        self,
        price_history: List[float],
        period: int
    ) -> List[SupportResistanceLevel]:
        """Detect Fibonacci retracement levels"""
        levels = []
        
        if len(price_history) < 10:
            return levels
        
        data = price_history[-period:] if len(price_history) >= period else price_history
        
        high = max(data)
        low = min(data)
        diff = high - low
        
        # Standard Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            fib_price = high - (diff * ratio)
            
            level = SupportResistanceLevel(
                price=fib_price,
                level_type=LevelType.BOTH,
                strength=LevelStrength.WEAK,
                detection_method=DetectionMethod.FIBONACCI,
                touches=1,
                formation_index=period - 1
            )
            levels.append(level)
        
        return levels
    
    def _deduplicate_and_rank(
        self,
        levels: List[SupportResistanceLevel]
    ) -> List[SupportResistanceLevel]:
        """Remove duplicate levels and rank by strength"""
        if not levels:
            return []
        
        # Group nearby levels
        unique_levels = {}
        for level in levels:
            # Find existing level within tolerance
            found = False
            for existing_price, existing_level in unique_levels.items():
                if abs(level.price - existing_price) / level.price < (self.TOLERANCE_PCT / 100):
                    # Merge: increase touches and use stronger one
                    existing_level.touches += 1
                    if level.strength.value > existing_level.strength.value:
                        existing_level.strength = level.strength
                    found = True
                    break
            
            if not found:
                unique_levels[level.price] = level
        
        # Sort by strength
        result = sorted(unique_levels.values(), key=lambda x: x.confidence_score, reverse=True)
        
        return result
    
    def _estimate_breakout_probability(
        self,
        current_price: float,
        support: Optional[SupportResistanceLevel],
        resistance: Optional[SupportResistanceLevel],
        distance_to_support_pct: float,
        distance_to_resistance_pct: float
    ) -> float:
        """Estimate probability of breakout from current levels"""
        prob = 0.5  # Base 50% probability
        
        # Adjust based on distance and strength
        if support:
            if distance_to_support_pct < 1.0:
                prob += 0.15  # Very close to support = likely bounce
            elif distance_to_support_pct > 10.0:
                prob += 0.1   # Far from support = possible breakup
        
        if resistance:
            if distance_to_resistance_pct < 1.0:
                prob -= 0.15  # Very close to resistance = potential reversal
            elif distance_to_resistance_pct > 10.0:
                prob -= 0.1   # Far from resistance = heading up
        
        # Clamp to 0-1 range
        return max(0.0, min(1.0, prob))
    
    def format_level_analysis(self, analysis: LevelAnalysis) -> str:
        """Format level analysis for logging"""
        lines = []
        lines.append("📊 SUPPORT/RESISTANCE ANALYSIS")
        lines.append(f"   Price: €{analysis.current_price:,.2f}")
        
        if analysis.nearest_support:
            lines.append(f"   Support: €{analysis.nearest_support.price:,.2f} "
                        f"(↓{analysis.distance_to_support_pct:.1f}%, "
                        f"strength: {analysis.support_strength}/4)")
        
        if analysis.nearest_resistance:
            lines.append(f"   Resistance: €{analysis.nearest_resistance.price:,.2f} "
                        f"(↑{analysis.distance_to_resistance_pct:.1f}%, "
                        f"strength: {analysis.resistance_strength}/4)")
        
        lines.append(f"   Reward:Risk Ratio: {analysis.reward_risk_ratio:.2f}x")
        lines.append(f"   Breakout Probability: {analysis.breakout_probability*100:.0f}%")
        
        if analysis.is_near_support:
            lines.append("   ⚠️ NEAR SUPPORT - Watch for bounce")
        if analysis.is_near_resistance:
            lines.append("   ⚠️ NEAR RESISTANCE - Watch for rejection")
        
        return "\n".join(lines)
