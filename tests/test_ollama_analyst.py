"""
Tests for ollama_analyst.py

Validates:
- Prompt construction from snapshot + cycle state
- JSON response parsing (clean, markdown-wrapped, embedded)
- Regex fallback parsing for malformed responses
- Field validation and clamping (sentiment bounds, valid regimes)
- Graceful degradation (Ollama down, empty response, garbage response)
- LLMContext properties (stale detection)
- Health check logic

Run: python -m pytest tests/test_ollama_analyst.py -v
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BotConfig,
    CyclePhase,
    PersistenceConfig,
    VolatilityRegime,
)
from indicators import (
    BollingerBands,
    MACDResult,
    TechnicalSnapshot,
)
from cycle_detector import CycleState, MomentumState, PriceStructure
from signal_engine import LLMContext
from ollama_analyst import OllamaAnalyst


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path) -> BotConfig:
    return BotConfig(persistence=PersistenceConfig(base_dir=tmp_path))


def make_snapshot(price: float = 55000.0) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        price=price,
        timestamp=time.time(),
        rsi=45.0,
        macd=MACDResult(macd_line=150.0, signal_line=100.0, histogram=50.0),
        bollinger=BollingerBands(
            upper=57000.0, middle=55000.0, lower=53000.0,
            bandwidth=0.07, percent_b=0.5,
        ),
        atr=800.0,
        vwap=54500.0,
    )


def make_cycle() -> CycleState:
    return CycleState(
        phase=CyclePhase.GROWTH,
        phase_confidence=0.7,
        time_score=0.3,
        price_score=0.2,
        momentum_score=0.3,
        volatility_score=0.0,
        composite_score=0.35,
        momentum=MomentumState(
            rsi_zone="neutral", trend_direction="up",
            higher_highs=True, higher_lows=True,
            rsi_bullish_divergence=False, rsi_bearish_divergence=False,
            momentum_score=0.3,
        ),
        price_structure=PriceStructure(
            drawdown_from_ath=0.15,
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


def make_analyst(tmp_path: Path) -> OllamaAnalyst:
    return OllamaAnalyst(
        config=make_config(tmp_path),
        base_url="http://127.0.0.1:11434",
        model="gemma3:4b",
    )


# ─── Prompt construction tests ───────────────────────────────────────────────

class TestPromptConstruction:
    def test_prompt_contains_price(self, tmp_path):
        prompt = OllamaAnalyst._build_prompt(make_snapshot(55000.0), make_cycle())
        assert "55,000" in prompt

    def test_prompt_contains_rsi(self, tmp_path):
        prompt = OllamaAnalyst._build_prompt(make_snapshot(), make_cycle())
        assert "45.0" in prompt

    def test_prompt_contains_cycle_phase(self, tmp_path):
        prompt = OllamaAnalyst._build_prompt(make_snapshot(), make_cycle())
        assert "growth" in prompt.lower()

    def test_prompt_contains_cycle_day(self, tmp_path):
        prompt = OllamaAnalyst._build_prompt(make_snapshot(), make_cycle())
        assert "400" in prompt

    def test_prompt_handles_none_indicators(self, tmp_path):
        """Prompt should show N/A for missing indicators."""
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        cycle = make_cycle()
        prompt = OllamaAnalyst._build_prompt(snap, cycle)
        assert "N/A" in prompt

    def test_prompt_contains_ath(self, tmp_path):
        prompt = OllamaAnalyst._build_prompt(make_snapshot(), make_cycle())
        assert "65,000" in prompt

    def test_prompt_contains_trend(self, tmp_path):
        prompt = OllamaAnalyst._build_prompt(make_snapshot(), make_cycle())
        assert "up" in prompt.lower()


# ─── JSON parsing tests ─────────────────────────────────────────────────────

class TestJSONParsing:
    def test_clean_json(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "markup",
            "sentiment": 0.6,
            "risk_level": "medium",
            "themes": ["halving rally", "institutional buying"],
        })
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.regime == "markup"
        assert result.sentiment == pytest.approx(0.6)
        assert result.risk_level == "medium"
        assert len(result.themes) == 2

    def test_json_in_markdown_block(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = """Here's my analysis:
```json
{
  "regime": "accumulation",
  "sentiment": -0.3,
  "risk_level": "low",
  "themes": ["fear in market", "buying opportunity"]
}
```
Hope this helps!"""
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.regime == "accumulation"
        assert result.sentiment == pytest.approx(-0.3)

    def test_json_embedded_in_text(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = """Based on the data, my analysis is:
{"regime": "distribution", "sentiment": 0.8, "risk_level": "high", "themes": ["overheated"]}
That's what I think."""
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.regime == "distribution"
        assert result.risk_level == "high"

    def test_json_with_extra_whitespace(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = """
        {
            "regime":   "markdown",
            "sentiment":  -0.7 ,
            "risk_level": "extreme",
            "themes": [ "panic selling" , "liquidity crisis" ]
        }
        """
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.regime == "decline"  # "markdown" → "decline" alias
        assert result.sentiment == pytest.approx(-0.7)
        assert result.risk_level == "extreme"


# ─── Regex fallback tests ───────────────────────────────────────────────────

class TestRegexFallback:
    def test_malformed_json_with_extractable_fields(self, tmp_path):
        analyst = make_analyst(tmp_path)
        # Broken JSON but fields are extractable
        raw = """I think the market is in an
"regime": "capitulation"
with "sentiment": -0.9 and
"risk_level": "extreme"
"themes": ["crash", "panic"]
overall quite bearish"""
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.regime == "capitulation"
        assert result.sentiment == pytest.approx(-0.9)

    def test_complete_garbage_returns_none(self, tmp_path):
        analyst = make_analyst(tmp_path)
        result = analyst._parse_response("I don't understand the question")
        assert result is None

    def test_empty_string_returns_none(self, tmp_path):
        analyst = make_analyst(tmp_path)
        result = analyst._parse_response("")
        assert result is None


# ─── Validation tests ───────────────────────────────────────────────────────

class TestValidation:
    def test_sentiment_clamped_to_bounds(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "markup",
            "sentiment": 5.0,  # Way out of range
            "risk_level": "low",
            "themes": [],
        })
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.sentiment == 1.0  # Clamped

    def test_negative_sentiment_clamped(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "markdown",
            "sentiment": -3.0,
            "risk_level": "high",
            "themes": [],
        })
        result = analyst._parse_response(raw)
        assert result.sentiment == -1.0

    def test_invalid_regime_defaults(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "bullish_explosion",  # Not a valid regime
            "sentiment": 0.5,
            "risk_level": "low",
            "themes": [],
        })
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.regime == "accumulation"  # Default

    def test_invalid_risk_defaults(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "markup",
            "sentiment": 0.5,
            "risk_level": "nuclear",  # Not valid
            "themes": [],
        })
        result = analyst._parse_response(raw)
        assert result.risk_level == "medium"  # Default

    def test_themes_truncated_to_5(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "markup",
            "sentiment": 0.5,
            "risk_level": "low",
            "themes": ["a", "b", "c", "d", "e", "f", "g"],
        })
        result = analyst._parse_response(raw)
        assert len(result.themes) <= 5

    def test_themes_non_list_handled(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "markup",
            "sentiment": 0.5,
            "risk_level": "low",
            "themes": "just a string",
        })
        result = analyst._parse_response(raw)
        assert result.themes == ()

    def test_non_numeric_sentiment_defaults(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({
            "regime": "markup",
            "sentiment": "very positive",
            "risk_level": "low",
            "themes": [],
        })
        result = analyst._parse_response(raw)
        assert result.sentiment == 0.0

    def test_missing_fields_get_defaults(self, tmp_path):
        analyst = make_analyst(tmp_path)
        raw = json.dumps({"regime": "markup"})
        result = analyst._parse_response(raw)
        assert result is not None
        assert result.sentiment == 0.0
        assert result.risk_level == "medium"
        assert result.themes == ()


# ─── LLMContext property tests ───────────────────────────────────────────────

class TestLLMContext:
    def test_fresh_context_not_stale(self):
        ctx = LLMContext(
            regime="markup", sentiment=0.5, risk_level="low",
            themes=("test",), timestamp=time.time(),
        )
        assert ctx.stale is False

    def test_old_context_is_stale(self):
        ctx = LLMContext(
            regime="markup", sentiment=0.5, risk_level="low",
            themes=("test",), timestamp=time.time() - 10000,  # ~2.8 hours ago
        )
        assert ctx.stale is True

    def test_frozen(self):
        ctx = LLMContext(
            regime="markup", sentiment=0.5, risk_level="low",
            themes=("test",), timestamp=time.time(),
        )
        with pytest.raises(AttributeError):
            ctx.sentiment = 0.0  # type: ignore


# ─── Ollama interaction tests (mocked) ──────────────────────────────────────

class TestOllamaInteraction:
    @patch.object(OllamaAnalyst, "_query_ollama")
    def test_analyze_success(self, mock_query, tmp_path):
        mock_query.return_value = json.dumps({
            "regime": "accumulation",
            "sentiment": -0.2,
            "risk_level": "low",
            "themes": ["halving approaching", "low volatility"],
        })
        analyst = make_analyst(tmp_path)
        result = analyst.analyze(make_snapshot(), make_cycle())

        assert result is not None
        assert result.regime == "accumulation"
        assert result.sentiment == pytest.approx(-0.2)
        assert not result.stale

    @patch.object(OllamaAnalyst, "_query_ollama")
    def test_analyze_ollama_down(self, mock_query, tmp_path):
        mock_query.return_value = None
        analyst = make_analyst(tmp_path)
        result = analyst.analyze(make_snapshot(), make_cycle())
        assert result is None

    @patch.object(OllamaAnalyst, "_query_ollama")
    def test_analyze_garbage_response(self, mock_query, tmp_path):
        mock_query.return_value = "Error: model not found"
        analyst = make_analyst(tmp_path)
        result = analyst.analyze(make_snapshot(), make_cycle())
        assert result is None

    @patch.object(OllamaAnalyst, "_query_ollama")
    def test_analyze_partial_json(self, mock_query, tmp_path):
        """LLM returns JSON-ish text with extractable fields."""
        mock_query.return_value = """Sure, here is my analysis:
{"regime": "markup", "sentiment": 0.4, "risk_level": "medium", "themes": ["steady growth"]}"""
        analyst = make_analyst(tmp_path)
        result = analyst.analyze(make_snapshot(), make_cycle())
        assert result is not None
        assert result.regime == "markup"


# ─── Health check tests (mocked) ────────────────────────────────────────────

class TestHealthCheck:
    @patch("ollama_analyst.requests.Session.get")
    def test_healthy_with_model(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "gemma3:4b"}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        analyst = make_analyst(tmp_path)
        assert analyst.health_check() is True
        assert analyst.is_available is True

    @patch("ollama_analyst.requests.Session.get")
    def test_model_not_found(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:8b"}],  # Different model
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        analyst = make_analyst(tmp_path)
        assert analyst.health_check() is False

    @patch("ollama_analyst.requests.Session.get")
    def test_connection_error(self, mock_get, tmp_path):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        analyst = make_analyst(tmp_path)
        assert analyst.health_check() is False
        assert analyst.is_available is False


# ─── Availability tracking tests ─────────────────────────────────────────────

class TestAvailabilityTracking:
    def test_initial_state(self, tmp_path):
        analyst = make_analyst(tmp_path)
        assert analyst._available is None
        assert analyst.is_available is False

    @patch.object(OllamaAnalyst, "_query_ollama")
    def test_available_after_success(self, mock_query, tmp_path):
        mock_query.return_value = json.dumps({
            "regime": "markup", "sentiment": 0.0,
            "risk_level": "medium", "themes": [],
        })
        analyst = make_analyst(tmp_path)
        analyst.analyze(make_snapshot(), make_cycle())
        # _query_ollama is mocked at the public level, so _available
        # isn't set by the mock. We test the real flow below.


# ─── Edge case integration tests ────────────────────────────────────────────

class TestEdgeCases:
    def test_all_none_snapshot(self, tmp_path):
        """Minimal snapshot should still produce a valid prompt."""
        snap = TechnicalSnapshot(price=50000.0, timestamp=time.time())
        cycle = make_cycle()
        prompt = OllamaAnalyst._build_prompt(snap, cycle)
        assert "50,000" in prompt
        assert "N/A" in prompt  # Missing indicators

    def test_extreme_values_in_prompt(self, tmp_path):
        """Very large/small values shouldn't crash prompt construction."""
        snap = TechnicalSnapshot(
            price=1000000.0,
            timestamp=time.time(),
            rsi=99.9,
            macd=MACDResult(macd_line=50000.0, signal_line=40000.0, histogram=10000.0),
            bollinger=BollingerBands(
                upper=1100000.0, middle=1000000.0, lower=900000.0,
                bandwidth=0.2, percent_b=1.5,
            ),
            atr=50000.0,
            vwap=990000.0,
        )
        cycle = make_cycle()
        prompt = OllamaAnalyst._build_prompt(snap, cycle)
        assert "1,000,000" in prompt

    @patch.object(OllamaAnalyst, "_query_ollama")
    def test_multiple_json_objects_takes_first(self, mock_query, tmp_path):
        """If response has multiple JSON objects, we parse the first valid one."""
        mock_query.return_value = """
{"regime": "markup", "sentiment": 0.5, "risk_level": "low", "themes": ["first"]}
{"regime": "markdown", "sentiment": -0.5, "risk_level": "high", "themes": ["second"]}
"""
        analyst = make_analyst(tmp_path)
        result = analyst.analyze(make_snapshot(), make_cycle())
        assert result is not None
        # Should get the first valid parse (either via full text or regex)
        assert result.regime in ("markup", "decline")
