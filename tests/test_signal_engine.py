"""
Tests for signal_engine.py

Validates:
- Each sub-signal generator (RSI, MACD, Bollinger, cycle, on-chain, LLM, microstructure)
- Composite score calculation and weighting
- Agreement measurement
- Action determination with thresholds and minimum agreement
- Data quality assessment
- Accumulation strategy sentiment inversion
- RSI divergence conviction boosting in appropriate phases
- Edge cases (all None inputs, stale LLM, partial data)
- CompositeSignal properties (actionable, is_buy, is_sell)

Run: python -m pytest tests/test_signal_engine.py -v
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ATHTracker,
    BotConfig,
    CyclePhase,
    PersistenceConfig,
    SignalConfig,
    VolatilityRegime,
)
from indicators import (
    BollingerBands,
    MACDResult,
    RSIDivergence,
    TechnicalSnapshot,
)
from cycle_detector import CycleState, MomentumState, PriceStructure
from bitcoin_node import (
    FeeEstimate,
    MempoolInfo,
    BlockInfo,
    NetworkInfo,
    OnChainSnapshot,
)
from signal_engine import (
    Action,
    CompositeSignal,
    LLMContext,
    SignalEngine,
    SubSignal,
    _neutral_signal,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path) -> BotConfig:
    return BotConfig(persistence=PersistenceConfig(base_dir=tmp_path))


def make_engine(tmp_path: Path) -> SignalEngine:
    return SignalEngine(make_config(tmp_path))


def make_snapshot(
    price: float = 50000.0,
    rsi: float = 50.0,
    macd_hist: float = 0.0,
    bb_pct_b: float = 0.5,
    bb_squeeze: bool = False,
    vwap: float = 50000.0,
    vol_pct: float = 0.5,
    rsi_divergence: RSIDivergence = None,
) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        price=price,
        timestamp=time.time(),
        rsi=rsi,
        macd=MACDResult(
            macd_line=macd_hist + 10.0,
            signal_line=10.0,
            histogram=macd_hist,
        ),
        bollinger=BollingerBands(
            upper=price * 1.04, middle=price,
            lower=price * 0.96,
            bandwidth=0.08, percent_b=bb_pct_b,
        ),
        atr=price * 0.02,
        vwap=vwap,
        rsi_divergence=rsi_divergence,
        bollinger_squeeze=bb_squeeze,
        volatility_percentile=vol_pct,
    )


def make_cycle(
    phase: CyclePhase = CyclePhase.GROWTH,
    composite_score: float = 0.3,
    confidence: float = 0.7,
    bull_div: bool = False,
    bear_div: bool = False,
) -> CycleState:
    return CycleState(
        phase=phase,
        phase_confidence=confidence,
        time_score=0.3,
        price_score=0.2,
        momentum_score=0.3,
        volatility_score=0.0,
        composite_score=composite_score,
        momentum=MomentumState(
            rsi_zone="neutral",
            trend_direction="up",
            higher_highs=True,
            higher_lows=True,
            rsi_bullish_divergence=bull_div,
            rsi_bearish_divergence=bear_div,
            momentum_score=0.3,
        ),
        price_structure=PriceStructure(
            drawdown_from_ath=0.2,
            position_in_range=0.5,
            distance_from_200d_ma=0.1,
            price_structure_score=0.2,
        ),
        volatility_regime=VolatilityRegime.NORMAL,
        cycle_day=400,
        cycle_progress=0.28,
        ath_eur=65000.0,
        drawdown_tolerance=0.20,
        position_size_multiplier=1.0,
        profit_taking_active=True,
        timestamp=time.time(),
    )


def make_onchain(
    mempool_clearing: bool = False,
    mempool_congested: bool = False,
    fee_pressure: float = 0.3,
    whale_count: int = 0,
    network_stress: float = 0.2,
) -> OnChainSnapshot:
    from bitcoin_node import LargeTransaction
    mempool = MempoolInfo(
        tx_count=3000 if mempool_clearing else (60000 if mempool_congested else 20000),
        size_bytes=2000000 if mempool_clearing else (150000000 if mempool_congested else 30000000),
        total_fee_btc=0.5,
        min_fee_rate=2.0,
        memory_usage_bytes=50000000,
    )
    # Construct fee rates to produce desired pressure
    # pressure = (fast/economy - 1) / 9
    # fast = economy * (1 + pressure * 9)
    economy = 3.0
    fast = economy * (1 + fee_pressure * 9)
    fees = FeeEstimate(
        fast_sat_vb=fast, medium_sat_vb=fast * 0.5,
        slow_sat_vb=economy * 1.5, economy_sat_vb=economy,
    )
    blocks = BlockInfo(
        height=880000, avg_interval_seconds=600.0,
        avg_tx_count=2500, avg_block_size_bytes=1500000,
        blocks_since_difficulty=100,
    )
    network = NetworkInfo(hashrate_eh=600.0, difficulty=80e12, connection_count=100)
    large_txs = [
        LargeTransaction(
            txid=f"whale_{i}", value_btc=100.0,
            fee_btc=0.01, size_bytes=500, is_mempool=True,
        )
        for i in range(whale_count)
    ]
    return OnChainSnapshot(
        mempool=mempool, fees=fees, blocks=blocks,
        network=network, large_txs=large_txs, timestamp=time.time(),
    )


def make_llm(
    sentiment: float = 0.0,
    risk_level: str = "medium",
    regime: str = "accumulation",
    stale: bool = False,
) -> LLMContext:
    ts = time.time() - (10000 if stale else 100)
    return LLMContext(
        regime=regime, sentiment=sentiment, risk_level=risk_level,
        themes=("test",), timestamp=ts,
    )


# ─── SubSignal tests ────────────────────────────────────────────────────────

class TestSubSignal:
    def test_frozen(self):
        s = SubSignal(name="test", score=50.0, weight=0.2, direction=1, reason="test")
        with pytest.raises(AttributeError):
            s.score = 0.0  # type: ignore

    def test_fields(self):
        s = SubSignal(name="rsi", score=-30.0, weight=0.2, direction=-1, reason="oversold")
        assert s.name == "rsi"
        assert s.direction == -1


# ─── RSI sub-signal tests ───────────────────────────────────────────────────

class TestRSISignal:
    def test_deeply_oversold_bullish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(rsi=15.0)
        cycle = make_cycle()
        sig = engine._rsi_signal(snap, cycle)
        assert sig.score > 40
        assert sig.direction == 1

    def test_overbought_bearish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(rsi=88.0)
        cycle = make_cycle()
        sig = engine._rsi_signal(snap, cycle)
        assert sig.score < -40
        assert sig.direction == -1

    def test_neutral_rsi(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(rsi=50.0)
        cycle = make_cycle()
        sig = engine._rsi_signal(snap, cycle)
        assert -10 < sig.score < 10

    def test_none_rsi_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        cycle = make_cycle()
        sig = engine._rsi_signal(snap, cycle)
        assert sig.score == 0.0
        assert sig.direction == 0

    def test_bullish_divergence_in_capitulation_high_conviction(self, tmp_path):
        """Bullish divergence during capitulation should be highest conviction."""
        engine = make_engine(tmp_path)
        div = RSIDivergence(bullish=True, bearish=False, strength=0.8)
        snap = make_snapshot(rsi=28.0, rsi_divergence=div)
        cycle_cap = make_cycle(phase=CyclePhase.CAPITULATION)
        cycle_growth = make_cycle(phase=CyclePhase.GROWTH)

        sig_cap = engine._rsi_signal(snap, cycle_cap)
        sig_growth = engine._rsi_signal(snap, cycle_growth)

        # Capitulation divergence should score higher than growth divergence
        assert sig_cap.score > sig_growth.score
        assert sig_cap.direction == 1

    def test_bearish_divergence_in_euphoria(self, tmp_path):
        """Bearish divergence during euphoria = high conviction sell."""
        engine = make_engine(tmp_path)
        div = RSIDivergence(bullish=False, bearish=True, strength=0.9)
        snap = make_snapshot(rsi=72.0, rsi_divergence=div)
        cycle = make_cycle(phase=CyclePhase.EUPHORIA)
        sig = engine._rsi_signal(snap, cycle)
        assert sig.score < -30
        assert sig.direction == -1

    def test_score_bounded(self, tmp_path):
        engine = make_engine(tmp_path)
        div = RSIDivergence(bullish=True, bearish=True, strength=1.0)
        snap = make_snapshot(rsi=10.0, rsi_divergence=div)
        cycle = make_cycle(phase=CyclePhase.CAPITULATION)
        sig = engine._rsi_signal(snap, cycle)
        assert -100.0 <= sig.score <= 100.0


# ─── MACD sub-signal tests ──────────────────────────────────────────────────

class TestMACDSignal:
    def test_positive_histogram_bullish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(macd_hist=200.0)
        sig = engine._macd_signal(snap)
        assert sig.score > 0
        assert sig.direction == 1

    def test_negative_histogram_bearish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(macd_hist=-200.0)
        sig = engine._macd_signal(snap)
        assert sig.score < 0
        assert sig.direction == -1

    def test_none_macd_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        sig = engine._macd_signal(snap)
        assert sig.score == 0.0

    def test_score_bounded(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(macd_hist=50000.0)  # Extreme
        sig = engine._macd_signal(snap)
        assert -100.0 <= sig.score <= 100.0


# ─── Bollinger sub-signal tests ─────────────────────────────────────────────

class TestBollingerSignal:
    def test_below_lower_band_bullish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(bb_pct_b=-0.1)
        sig = engine._bollinger_signal(snap)
        assert sig.score > 30
        assert sig.direction == 1

    def test_above_upper_band_bearish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(bb_pct_b=1.1)
        sig = engine._bollinger_signal(snap)
        assert sig.score < -30
        assert sig.direction == -1

    def test_middle_of_bands_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(bb_pct_b=0.5)
        sig = engine._bollinger_signal(snap)
        assert -10 < sig.score < 10

    def test_squeeze_amplifies_signal(self, tmp_path):
        engine = make_engine(tmp_path)
        snap_no_squeeze = make_snapshot(bb_pct_b=0.1, bb_squeeze=False)
        snap_squeeze = make_snapshot(bb_pct_b=0.1, bb_squeeze=True)

        sig_no = engine._bollinger_signal(snap_no_squeeze)
        sig_yes = engine._bollinger_signal(snap_squeeze)

        # Squeeze should amplify the buy signal
        assert abs(sig_yes.score) > abs(sig_no.score)

    def test_none_bollinger_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        sig = engine._bollinger_signal(snap)
        assert sig.score == 0.0


# ─── Cycle sub-signal tests ─────────────────────────────────────────────────

class TestCycleSignal:
    def test_bullish_cycle_positive(self, tmp_path):
        engine = make_engine(tmp_path)
        cycle = make_cycle(phase=CyclePhase.GROWTH, composite_score=0.6)
        sig = engine._cycle_signal(cycle)
        assert sig.score > 0
        assert sig.direction == 1

    def test_capitulation_gets_accumulation_bonus(self, tmp_path):
        engine = make_engine(tmp_path)
        cycle = make_cycle(phase=CyclePhase.CAPITULATION, composite_score=-0.5)
        sig = engine._cycle_signal(cycle)
        # Capitulation should get a +20 bonus for accumulation strategy
        # Even though composite is -0.5, the bonus should help
        assert "accumulation opportunity" in sig.reason.lower()

    def test_euphoria_gets_penalty(self, tmp_path):
        engine = make_engine(tmp_path)
        cycle = make_cycle(phase=CyclePhase.EUPHORIA, composite_score=0.8)
        sig = engine._cycle_signal(cycle)
        # Should still be positive but reduced
        assert "reduce exposure" in sig.reason.lower()

    def test_score_bounded(self, tmp_path):
        engine = make_engine(tmp_path)
        for comp in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for phase in CyclePhase:
                cycle = make_cycle(phase=phase, composite_score=comp)
                sig = engine._cycle_signal(cycle)
                assert -100.0 <= sig.score <= 100.0


# ─── On-chain sub-signal tests ──────────────────────────────────────────────

class TestOnchainSignal:
    def test_none_onchain_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        sig = engine._onchain_signal(None)
        assert sig.score == 0.0
        assert sig.direction == 0

    def test_mempool_clearing_bullish(self, tmp_path):
        engine = make_engine(tmp_path)
        onchain = make_onchain(mempool_clearing=True, fee_pressure=0.1)
        sig = engine._onchain_signal(onchain)
        assert sig.score > 0

    def test_mempool_congested_bearish(self, tmp_path):
        engine = make_engine(tmp_path)
        onchain = make_onchain(mempool_congested=True, fee_pressure=0.8)
        sig = engine._onchain_signal(onchain)
        assert sig.score < 0

    def test_whale_activity_noted(self, tmp_path):
        engine = make_engine(tmp_path)
        onchain = make_onchain(whale_count=5)
        sig = engine._onchain_signal(onchain)
        assert "whale" in sig.reason.lower()

    def test_score_bounded(self, tmp_path):
        engine = make_engine(tmp_path)
        for clearing in [True, False]:
            for congested in [True, False]:
                onchain = make_onchain(
                    mempool_clearing=clearing,
                    mempool_congested=congested,
                    whale_count=10,
                )
                sig = engine._onchain_signal(onchain)
                assert -100.0 <= sig.score <= 100.0


# ─── LLM sub-signal tests ───────────────────────────────────────────────────

class TestLLMSignal:
    def test_none_llm_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        sig = engine._llm_signal(None)
        assert sig.score == 0.0

    def test_stale_llm_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        llm = make_llm(sentiment=0.8, stale=True)
        sig = engine._llm_signal(llm)
        assert sig.score == 0.0
        assert "stale" in sig.reason.lower()

    def test_sentiment_inverted_for_accumulation(self, tmp_path):
        """
        Critical: negative sentiment should produce POSITIVE signal
        for an accumulation bot.
        """
        engine = make_engine(tmp_path)
        negative_llm = make_llm(sentiment=-0.8, risk_level="medium")
        positive_llm = make_llm(sentiment=0.8, risk_level="medium")

        sig_neg = engine._llm_signal(negative_llm)
        sig_pos = engine._llm_signal(positive_llm)

        # Negative sentiment → higher score (inverted for accumulation)
        assert sig_neg.score > sig_pos.score
        assert "inverted" in sig_neg.reason.lower()

    def test_extreme_risk_bearish(self, tmp_path):
        engine = make_engine(tmp_path)
        llm = make_llm(sentiment=0.0, risk_level="extreme")
        sig = engine._llm_signal(llm)
        assert sig.score < 0

    def test_low_risk_bullish(self, tmp_path):
        engine = make_engine(tmp_path)
        llm = make_llm(sentiment=0.0, risk_level="low")
        sig = engine._llm_signal(llm)
        assert sig.score > 0

    def test_bearish_regime_is_opportunity(self, tmp_path):
        """For accumulation bot, bearish LLM regime = buy opportunity."""
        engine = make_engine(tmp_path)
        llm = make_llm(sentiment=0.0, risk_level="medium", regime="capitulation")
        sig = engine._llm_signal(llm)
        assert "opportunity" in sig.reason.lower()


# ─── Microstructure sub-signal tests ─────────────────────────────────────────

class TestMicrostructureSignal:
    def test_below_vwap_bullish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(price=49000.0, vwap=50000.0)
        sig = engine._microstructure_signal(snap)
        assert sig.score > 0
        assert sig.direction == 1

    def test_above_vwap_bearish(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(price=51000.0, vwap=50000.0)
        sig = engine._microstructure_signal(snap)
        assert sig.score < 0
        assert sig.direction == -1

    def test_at_vwap_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot(price=50000.0, vwap=50000.0)
        sig = engine._microstructure_signal(snap)
        assert sig.score == pytest.approx(0.0)

    def test_none_vwap_neutral(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        sig = engine._microstructure_signal(snap)
        assert sig.score == 0.0

    def test_high_vol_discounts_signal(self, tmp_path):
        engine = make_engine(tmp_path)
        snap_normal = make_snapshot(price=49000.0, vwap=50000.0, vol_pct=0.5)
        snap_highvol = make_snapshot(price=49000.0, vwap=50000.0, vol_pct=0.90)

        sig_normal = engine._microstructure_signal(snap_normal)
        sig_highvol = engine._microstructure_signal(snap_highvol)

        assert abs(sig_normal.score) > abs(sig_highvol.score)


# ─── Composite calculation tests ────────────────────────────────────────────

class TestCompositeCalculation:
    def test_all_bullish_positive_score(self, tmp_path):
        engine = make_engine(tmp_path)
        components = [
            SubSignal(name="a", score=50.0, weight=0.3, direction=1, reason=""),
            SubSignal(name="b", score=40.0, weight=0.3, direction=1, reason=""),
            SubSignal(name="c", score=60.0, weight=0.4, direction=1, reason=""),
        ]
        score, agreement = engine._compute_composite(components)
        assert score > 0
        assert agreement == 1.0  # All bullish

    def test_all_bearish_negative_score(self, tmp_path):
        engine = make_engine(tmp_path)
        components = [
            SubSignal(name="a", score=-50.0, weight=0.5, direction=-1, reason=""),
            SubSignal(name="b", score=-30.0, weight=0.5, direction=-1, reason=""),
        ]
        score, agreement = engine._compute_composite(components)
        assert score < 0
        assert agreement == 1.0

    def test_mixed_signals_low_agreement(self, tmp_path):
        engine = make_engine(tmp_path)
        components = [
            SubSignal(name="a", score=50.0, weight=0.5, direction=1, reason=""),
            SubSignal(name="b", score=-50.0, weight=0.5, direction=-1, reason=""),
        ]
        score, agreement = engine._compute_composite(components)
        assert agreement == 0.5  # 50/50 split

    def test_neutral_signals_zero_agreement(self, tmp_path):
        engine = make_engine(tmp_path)
        components = [
            SubSignal(name="a", score=0.0, weight=0.5, direction=0, reason=""),
            SubSignal(name="b", score=0.0, weight=0.5, direction=0, reason=""),
        ]
        score, agreement = engine._compute_composite(components)
        assert agreement == 0.0

    def test_weighted_correctly(self, tmp_path):
        engine = make_engine(tmp_path)
        # Signal A: +100, weight 0.8
        # Signal B: -100, weight 0.2
        # Weighted: (100*0.8 + (-100)*0.2) / 1.0 = 60
        components = [
            SubSignal(name="a", score=100.0, weight=0.8, direction=1, reason=""),
            SubSignal(name="b", score=-100.0, weight=0.2, direction=-1, reason=""),
        ]
        score, agreement = engine._compute_composite(components)
        assert score == pytest.approx(60.0)

    def test_empty_components(self, tmp_path):
        engine = make_engine(tmp_path)
        score, agreement = engine._compute_composite([])
        assert score == 0.0
        assert agreement == 0.0

    def test_score_bounded(self, tmp_path):
        engine = make_engine(tmp_path)
        components = [
            SubSignal(name="a", score=100.0, weight=1.0, direction=1, reason=""),
        ]
        score, _ = engine._compute_composite(components)
        assert -100.0 <= score <= 100.0


# ─── Action determination tests ─────────────────────────────────────────────

class TestActionDetermination:
    def test_strong_buy(self, tmp_path):
        engine = make_engine(tmp_path)
        # Default buy_threshold=10, so strong_buy at 20+
        action = engine._determine_action(score=50.0, agreement=0.8, data_quality=0.8)
        assert action == Action.STRONG_BUY

    def test_buy(self, tmp_path):
        engine = make_engine(tmp_path)
        # Score 15: above buy_threshold (10) but below strong_buy (20)
        action = engine._determine_action(score=18.0, agreement=0.7, data_quality=0.8)
        assert action == Action.BUY

    def test_hold_neutral_score(self, tmp_path):
        engine = make_engine(tmp_path)
        action = engine._determine_action(score=5.0, agreement=0.8, data_quality=0.8)
        assert action == Action.HOLD

    def test_sell(self, tmp_path):
        engine = make_engine(tmp_path)
        action = engine._determine_action(score=-25.0, agreement=0.7, data_quality=0.8)
        assert action == Action.SELL

    def test_strong_sell(self, tmp_path):
        engine = make_engine(tmp_path)
        action = engine._determine_action(score=-50.0, agreement=0.8, data_quality=0.8)
        assert action == Action.STRONG_SELL

    def test_low_agreement_forces_hold(self, tmp_path):
        """Below buy_min_agreement (0.35 default) → HOLD regardless of score."""
        engine = make_engine(tmp_path)
        action = engine._determine_action(score=80.0, agreement=0.3, data_quality=0.8)
        assert action == Action.HOLD

    def test_low_data_quality_forces_hold(self, tmp_path):
        """Below data quality threshold → HOLD."""
        engine = make_engine(tmp_path)
        action = engine._determine_action(score=80.0, agreement=0.9, data_quality=0.1)
        assert action == Action.HOLD


# ─── Data quality tests ─────────────────────────────────────────────────────

class TestDataQuality:
    def test_full_data_high_quality(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot()  # All indicators present
        cycle = make_cycle(confidence=0.8)
        onchain = make_onchain()
        llm = make_llm()
        quality = engine._assess_data_quality(snap, cycle, onchain, llm)
        assert quality > 0.8

    def test_no_optional_data_partial_quality(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot()
        cycle = make_cycle()
        quality = engine._assess_data_quality(snap, cycle, None, None)
        # Should have tech + cycle but no onchain/llm → ~0.5-0.7
        assert 0.4 < quality < 0.8

    def test_empty_snapshot_low_quality(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        cycle = make_cycle(confidence=0.1)
        quality = engine._assess_data_quality(snap, cycle, None, None)
        assert quality < 0.3

    def test_stale_llm_not_counted(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot()
        cycle = make_cycle()
        stale_llm = make_llm(stale=True)
        fresh_llm = make_llm(stale=False)

        q_stale = engine._assess_data_quality(snap, cycle, None, stale_llm)
        q_fresh = engine._assess_data_quality(snap, cycle, None, fresh_llm)
        assert q_fresh > q_stale

    def test_quality_bounded(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot()
        cycle = make_cycle()
        onchain = make_onchain()
        llm = make_llm()
        quality = engine._assess_data_quality(snap, cycle, onchain, llm)
        assert 0.0 <= quality <= 1.0


# ─── CompositeSignal property tests ─────────────────────────────────────────

class TestCompositeSignalProperties:
    def test_actionable_buy(self):
        sig = CompositeSignal(
            score=30.0, agreement=0.7, action=Action.BUY,
            components=(), data_quality=0.8, timestamp=time.time(),
        )
        assert sig.actionable is True
        assert sig.is_buy is True
        assert sig.is_sell is False

    def test_actionable_sell(self):
        sig = CompositeSignal(
            score=-30.0, agreement=0.7, action=Action.SELL,
            components=(), data_quality=0.8, timestamp=time.time(),
        )
        assert sig.actionable is True
        assert sig.is_buy is False
        assert sig.is_sell is True

    def test_hold_not_actionable(self):
        sig = CompositeSignal(
            score=0.0, agreement=0.0, action=Action.HOLD,
            components=(), data_quality=0.8, timestamp=time.time(),
        )
        assert sig.actionable is False

    def test_low_quality_not_actionable(self):
        sig = CompositeSignal(
            score=80.0, agreement=0.9, action=Action.STRONG_BUY,
            components=(), data_quality=0.1, timestamp=time.time(),
        )
        assert sig.actionable is False

    def test_strong_buy_is_buy(self):
        sig = CompositeSignal(
            score=80.0, agreement=0.9, action=Action.STRONG_BUY,
            components=(), data_quality=0.8, timestamp=time.time(),
        )
        assert sig.is_buy is True

    def test_strong_sell_is_sell(self):
        sig = CompositeSignal(
            score=-80.0, agreement=0.9, action=Action.STRONG_SELL,
            components=(), data_quality=0.8, timestamp=time.time(),
        )
        assert sig.is_sell is True

    def test_neutral_signal(self):
        sig = _neutral_signal()
        assert sig.score == 0.0
        assert sig.agreement == 0.0
        assert sig.action == Action.HOLD
        assert sig.actionable is False


# ─── Full generate() integration tests ──────────────────────────────────────

class TestGenerateIntegration:
    def test_bullish_scenario(self, tmp_path):
        """All indicators bullish → buy signal."""
        engine = make_engine(tmp_path)
        snap = make_snapshot(
            price=48000.0, rsi=28.0, macd_hist=300.0,
            bb_pct_b=0.05, vwap=50000.0,
        )
        cycle = make_cycle(
            phase=CyclePhase.ACCUMULATION, composite_score=0.3,
        )
        onchain = make_onchain(mempool_clearing=True, fee_pressure=0.1)
        llm = make_llm(sentiment=-0.5, risk_level="low", regime="accumulation")

        signal = engine.generate(snap, cycle, onchain, llm)

        assert signal.score > 0
        assert signal.action in (Action.BUY, Action.STRONG_BUY)
        assert signal.agreement > 0.5
        assert len(signal.components) == 7

    def test_bearish_scenario(self, tmp_path):
        """All indicators bearish → sell signal."""
        engine = make_engine(tmp_path)
        snap = make_snapshot(
            price=52000.0, rsi=85.0, macd_hist=-300.0,
            bb_pct_b=1.05, vwap=50000.0,
        )
        cycle = make_cycle(
            phase=CyclePhase.DISTRIBUTION, composite_score=-0.4,
        )
        onchain = make_onchain(mempool_congested=True, fee_pressure=0.9)
        llm = make_llm(sentiment=0.7, risk_level="high", regime="distribution")

        signal = engine.generate(snap, cycle, onchain, llm)

        assert signal.score < 0
        assert signal.action in (Action.SELL, Action.STRONG_SELL)

    def test_mixed_scenario_hold(self, tmp_path):
        """Conflicting signals → HOLD."""
        engine = make_engine(tmp_path)
        snap = make_snapshot(
            price=50000.0, rsi=50.0, macd_hist=0.0,
            bb_pct_b=0.5, vwap=50000.0,
        )
        cycle = make_cycle(phase=CyclePhase.GROWTH, composite_score=0.0)

        signal = engine.generate(snap, cycle)

        assert signal.action == Action.HOLD

    def test_minimal_data(self, tmp_path):
        """With minimal data, should still produce a signal."""
        engine = make_engine(tmp_path)
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        cycle = make_cycle()
        signal = engine.generate(snap, cycle)

        assert isinstance(signal, CompositeSignal)
        assert signal.data_quality < 0.5

    def test_all_components_present(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot()
        cycle = make_cycle()
        onchain = make_onchain()
        llm = make_llm()

        signal = engine.generate(snap, cycle, onchain, llm)

        names = {c.name for c in signal.components}
        assert names == {"rsi", "macd", "bollinger", "cycle", "onchain", "llm", "microstructure"}

    def test_signal_frozen(self, tmp_path):
        engine = make_engine(tmp_path)
        snap = make_snapshot()
        cycle = make_cycle()
        signal = engine.generate(snap, cycle)
        with pytest.raises(AttributeError):
            signal.score = 999.0  # type: ignore

    def test_observability(self, tmp_path):
        """Every component should have a non-empty reason string."""
        engine = make_engine(tmp_path)
        snap = make_snapshot()
        cycle = make_cycle()
        onchain = make_onchain()
        llm = make_llm()

        signal = engine.generate(snap, cycle, onchain, llm)

        for comp in signal.components:
            assert comp.reason, f"Component {comp.name} has empty reason"
            assert comp.name, "Component has empty name"
            assert -100.0 <= comp.score <= 100.0
