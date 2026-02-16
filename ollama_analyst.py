"""
Local LLM market analyst via Ollama.

Feeds structured market context to a local gemma3:4b model and extracts
regime classification, sentiment, risk level, and key themes. This runs
on the slow loop (every 15–60 minutes) since LLM inference is expensive.

Design:
- Structured prompts with explicit JSON output format.
- Robust parsing: regex fallback if JSON parsing fails.
- Timeout and retry logic — Ollama can be slow on CPU.
- Graceful degradation: if Ollama is down, return None and let the
  signal engine continue without the LLM sub-signal.
- Prompt designed for accumulation strategy context.
- All public methods return typed results, never raise.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import requests

from config import BotConfig
from cycle_detector import CycleState
from indicators import TechnicalSnapshot
from signal_engine import LLMContext

logger = logging.getLogger(__name__)


# ─── Prompt templates ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a Bitcoin market analyst for an automated accumulation trading bot.
Your job is to assess current market conditions and provide structured analysis.

IMPORTANT CONTEXT:
- This is a Bitcoin ACCUMULATION bot. It buys BTC with EUR.
- The goal is to accumulate more Bitcoin over time, not to maximize EUR returns.
- Negative sentiment / fear / crashes = BUYING OPPORTUNITY (not a risk).
- The bot operates on the Bitcoin 4-year halving cycle model.

You must respond with ONLY a JSON object, no other text. The JSON must have exactly these fields:
{
  "regime": "<one of: accumulation, markup, distribution, markdown, capitulation>",
  "sentiment": <float from -1.0 (extreme fear) to 1.0 (extreme greed)>,
  "risk_level": "<one of: low, medium, high, extreme>",
  "themes": ["<key theme 1>", "<key theme 2>", "<key theme 3>"]
}"""

_ANALYSIS_TEMPLATE = """Analyze the current Bitcoin market conditions:

PRICE DATA:
- Current price: €{price:,.0f}
- RSI (14): {rsi}
- MACD histogram: {macd_hist}
- Bollinger %B: {bb_pct_b}
- ATR: €{atr}
- Volatility regime: {vol_regime}

CYCLE CONTEXT:
- Cycle phase: {cycle_phase}
- Cycle day: {cycle_day} ({cycle_progress:.0%} elapsed)
- Composite cycle score: {cycle_composite:+.2f}
- ATH: €{ath:,.0f}
- Drawdown from ATH: {drawdown:.1%}

MOMENTUM:
- Trend: {trend}
- RSI zone: {rsi_zone}
- Higher highs: {higher_highs}
- Higher lows: {higher_lows}
- Bullish divergence: {bull_div}
- Bearish divergence: {bear_div}

Provide your analysis as a JSON object."""


# ─── Ollama Analyst ──────────────────────────────────────────────────────────

class OllamaAnalyst:
    """
    Local LLM market analyst using Ollama.

    Runs structured analysis prompts against a local model and parses
    the response into an LLMContext for the signal engine.

    Args:
        config: Bot configuration.
        base_url: Ollama API endpoint.
        model: Model name to use.
        timeout: HTTP timeout in seconds (LLM inference can be slow).
    """

    def __init__(
        self,
        config: BotConfig,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "gemma3:4b",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._session = requests.Session()
        self._available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """Whether Ollama was reachable on the last attempt."""
        return self._available is True

    # ─── Public interface ────────────────────────────────────────────────

    def analyze(
        self,
        snapshot: TechnicalSnapshot,
        cycle: CycleState,
    ) -> Optional[LLMContext]:
        """
        Run market analysis via the local LLM.

        Returns LLMContext on success, None on any failure.
        This is the slow-loop entry point — call every 15–60 minutes.
        """
        prompt = self._build_prompt(snapshot, cycle)
        raw_response = self._query_ollama(prompt)

        if raw_response is None:
            return None

        parsed = self._parse_response(raw_response)
        if parsed is None:
            return None

        return parsed

    def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check if our model is loaded (name may include :tag)
            available = any(
                self._model in m or m.startswith(self._model.split(":")[0])
                for m in models
            )
            if not available:
                logger.warning(
                    f"Model '{self._model}' not found in Ollama. "
                    f"Available: {models}"
                )
            self._available = available
            return available
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning(f"Ollama health check failed: {exc}")
            self._available = False
            return False

    # ─── Prompt construction ─────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        snapshot: TechnicalSnapshot,
        cycle: CycleState,
    ) -> str:
        """Build the analysis prompt from current market state."""
        rsi_str = f"{snapshot.rsi:.1f}" if snapshot.rsi is not None else "N/A"
        macd_str = (
            f"{snapshot.macd.histogram:+.0f}"
            if snapshot.macd is not None else "N/A"
        )
        bb_str = (
            f"{snapshot.bollinger.percent_b:.2f}"
            if snapshot.bollinger is not None else "N/A"
        )
        atr_str = f"{snapshot.atr:,.0f}" if snapshot.atr is not None else "N/A"

        return _ANALYSIS_TEMPLATE.format(
            price=snapshot.price,
            rsi=rsi_str,
            macd_hist=macd_str,
            bb_pct_b=bb_str,
            atr=atr_str,
            vol_regime=cycle.volatility_regime.value,
            cycle_phase=cycle.phase.value,
            cycle_day=cycle.cycle_day,
            cycle_progress=cycle.cycle_progress,
            cycle_composite=cycle.composite_score,
            ath=cycle.ath_eur,
            drawdown=cycle.price_structure.drawdown_from_ath,
            trend=cycle.momentum.trend_direction,
            rsi_zone=cycle.momentum.rsi_zone,
            higher_highs=cycle.momentum.higher_highs,
            higher_lows=cycle.momentum.higher_lows,
            bull_div=cycle.momentum.rsi_bullish_divergence,
            bear_div=cycle.momentum.rsi_bearish_divergence,
        )

    # ─── Ollama API interaction ──────────────────────────────────────────

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Ollama and return the response text.

        Uses the /api/generate endpoint with streaming disabled.
        """
        payload = {
            "model": self._model,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,      # Low temp for consistent analysis
                "num_predict": 512,       # Limit response length
                "top_p": 0.9,
            },
        }

        try:
            resp = self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            if self._available is not False:
                logger.warning(f"Ollama unreachable at {self._base_url}")
            self._available = False
            return None
        except requests.exceptions.Timeout:
            logger.warning(
                f"Ollama timeout after {self._timeout}s — "
                f"model may be loading or underpowered"
            )
            self._available = False
            return None
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else 0
            logger.warning(f"Ollama HTTP error {status}")
            self._available = False
            return None

        self._available = True

        try:
            data = resp.json()
            response_text = data.get("response", "")
            if not response_text:
                logger.warning("Ollama returned empty response")
                return None
            return response_text
        except (ValueError, KeyError) as exc:
            logger.warning(f"Failed to parse Ollama response: {exc}")
            return None

    # ─── Response parsing ────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Optional[LLMContext]:
        """
        Parse the LLM's JSON response into an LLMContext.

        Strategy:
        1. Try direct JSON parse.
        2. Try extracting JSON from markdown code blocks.
        3. Regex fallback for individual fields.
        """
        # Attempt 1: direct JSON parse
        parsed = self._try_json_parse(raw)
        if parsed is not None:
            return parsed

        # Attempt 2: extract from markdown code block
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block:
            parsed = self._try_json_parse(code_block.group(1))
            if parsed is not None:
                return parsed

        # Attempt 3: find any JSON-like object in the response
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            parsed = self._try_json_parse(json_match.group(0))
            if parsed is not None:
                return parsed

        # Attempt 4: regex extraction as last resort
        parsed = self._regex_fallback(raw)
        if parsed is not None:
            logger.info("Parsed LLM response via regex fallback")
            return parsed

        logger.warning(f"Failed to parse LLM response: {raw[:200]}...")
        return None

    def _try_json_parse(self, text: str) -> Optional[LLMContext]:
        """Attempt to parse text as JSON and validate the structure."""
        try:
            data = json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            return None

        return self._validate_and_build(data)

    def _validate_and_build(self, data: dict) -> Optional[LLMContext]:
        """Validate parsed data and build LLMContext."""
        if not isinstance(data, dict):
            return None

        # Extract and validate regime
        regime = str(data.get("regime", "")).lower().strip()
        valid_regimes = {
            "accumulation", "markup", "distribution",
            "markdown", "capitulation",
        }
        if regime not in valid_regimes:
            logger.debug(f"Invalid regime '{regime}', defaulting to 'accumulation'")
            regime = "accumulation"

        # Extract and validate sentiment
        try:
            sentiment = float(data.get("sentiment", 0.0))
            sentiment = max(-1.0, min(1.0, sentiment))
        except (ValueError, TypeError):
            sentiment = 0.0

        # Extract and validate risk level
        risk_level = str(data.get("risk_level", "medium")).lower().strip()
        valid_risks = {"low", "medium", "high", "extreme"}
        if risk_level not in valid_risks:
            risk_level = "medium"

        # Extract themes
        raw_themes = data.get("themes", [])
        if isinstance(raw_themes, list):
            themes = tuple(str(t) for t in raw_themes[:5])
        else:
            themes = ()

        return LLMContext(
            regime=regime,
            sentiment=sentiment,
            risk_level=risk_level,
            themes=themes,
            timestamp=time.time(),
        )

    def _regex_fallback(self, raw: str) -> Optional[LLMContext]:
        """
        Extract fields via regex when JSON parsing fails entirely.

        LLMs sometimes wrap JSON in explanatory text or malform it.
        """
        regime_match = re.search(
            r'"regime"\s*:\s*"(\w+)"', raw, re.IGNORECASE,
        )
        sentiment_match = re.search(
            r'"sentiment"\s*:\s*(-?[\d.]+)', raw, re.IGNORECASE,
        )
        risk_match = re.search(
            r'"risk_level"\s*:\s*"(\w+)"', raw, re.IGNORECASE,
        )

        if not (regime_match and sentiment_match):
            return None

        data = {
            "regime": regime_match.group(1),
            "sentiment": float(sentiment_match.group(1)),
            "risk_level": risk_match.group(1) if risk_match else "medium",
            "themes": [],
        }

        # Try to extract themes
        themes_match = re.search(
            r'"themes"\s*:\s*\[(.*?)\]', raw, re.DOTALL,
        )
        if themes_match:
            theme_strs = re.findall(r'"([^"]+)"', themes_match.group(1))
            data["themes"] = theme_strs[:5]

        return self._validate_and_build(data)

    def close(self) -> None:
        """Clean up the HTTP session."""
        self._session.close()
