"""
consensus.py — ML + LLM Consensus Layer (Phase 4)

Combines LightGBM's P(win) score with Groq's verdict on the same signal
to produce a STRONG_BUY / BUY / CONFLICT / SKIP decision and a position
size multiplier.

Decision table:
  LightGBM   Groq       Consensus    Size multiplier
  ─────────  ─────────  ──────────   ───────────────
  BUY        bullish    STRONG_BUY   1.00
  BUY        neutral    BUY          0.75
  BUY        bearish    CONFLICT     0.25
  SKIP       bullish    CONFLICT     0.25
  SKIP       neutral    SKIP         0.00
  SKIP       bearish    SKIP         0.00

Priority rules:
  - ML wins in normal regime (no strong catalyst)
  - LLM wins when catalyst_present=True or when regime is transitioning
  - CONFLICT is always logged with Groq's full reasoning for audit

Conflict logging:
  All CONFLICT signals are written to logs/consensus_conflicts.jsonl with the
  LightGBM score, Groq verdict, reasoning, symbol, and timestamp. This feeds
  the monthly feedback loop analysis.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

CONFLICT_LOG_PATH = Path("logs/consensus_conflicts.jsonl")

# Consensus labels
STRONG_BUY = "STRONG_BUY"
BUY = "BUY"
CONFLICT = "CONFLICT"
SKIP = "SKIP"

# Size multipliers per consensus decision
SIZE_MULTIPLIERS = {
    STRONG_BUY: 1.00,
    BUY:        0.75,
    CONFLICT:   0.25,
    SKIP:       0.00,
}


class ConsensusEngine:
    """Combines LightGBM + Groq signals into a single consensus decision.

    Designed to reduce bad trades when LightGBM's walk-forward accuracy is
    imperfect (~57%) by requiring alignment with Groq's qualitative reasoning
    before committing full position size.
    """

    def __init__(self, config: dict):
        self._cfg = config.get("model", config)
        self._high_conf = self._cfg.get("meta_model_confidence_high", 0.65)
        self._low_conf = self._cfg.get("meta_model_confidence_low", 0.52)
        self._groq_client = None
        self._groq_warned = False
        CONFLICT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self,
        symbol: str,
        lgbm_score: float,
        signal_data: dict,
        catalyst_present: bool = False,
        regime: str = "NEUTRAL",
    ) -> dict:
        """Run consensus evaluation for one symbol.

        Parameters
        ----------
        symbol        : ticker
        lgbm_score    : P(win) from LightGBM [0.0, 1.0]
        signal_data   : dict with technical + news context (passed to Groq)
        catalyst_present : True if a material catalyst was detected (LLM takes priority)
        regime        : current HMM regime name

        Returns
        -------
        dict with:
          lgbm_score, lgbm_verdict, groq_verdict, groq_reasoning,
          consensus, size_multiplier, catalyst_present, regime
        """
        # Determine LightGBM verdict
        if lgbm_score >= self._high_conf:
            lgbm_verdict = "bullish"
        elif lgbm_score >= self._low_conf:
            lgbm_verdict = "neutral"
        else:
            lgbm_verdict = "bearish"

        lgbm_is_buy = lgbm_verdict in ("bullish", "neutral")

        # Get Groq verdict
        groq_verdict, groq_reasoning = await self._get_groq_verdict(
            symbol=symbol,
            lgbm_score=lgbm_score,
            lgbm_verdict=lgbm_verdict,
            signal_data=signal_data,
            regime=regime,
        )

        # Apply priority rules
        # LLM takes precedence when a catalyst is present or regime is transitioning
        llm_priority = catalyst_present or regime in ("EARLY_BULL", "CRASH", "EUPHORIA")

        consensus = self._compute_consensus(
            lgbm_is_buy=lgbm_is_buy,
            groq_verdict=groq_verdict,
            lgbm_verdict=lgbm_verdict,
            llm_priority=llm_priority,
        )

        result = {
            "symbol":           symbol,
            "lgbm_score":       round(lgbm_score, 4),
            "lgbm_verdict":     lgbm_verdict,
            "groq_verdict":     groq_verdict,
            "groq_reasoning":   groq_reasoning,
            "consensus":        consensus,
            "size_multiplier":  SIZE_MULTIPLIERS[consensus],
            "catalyst_present": catalyst_present,
            "regime":           regime,
            "evaluated_at":     datetime.utcnow().isoformat(),
        }

        if consensus == CONFLICT:
            self._log_conflict(result)

        logger.info(
            "Consensus %s: %s | LGBM=%.3f(%s) Groq=%s → %s × %.2f",
            symbol, consensus, lgbm_score, lgbm_verdict,
            groq_verdict, consensus, SIZE_MULTIPLIERS[consensus],
        )
        return result

    def _compute_consensus(
        self,
        lgbm_is_buy: bool,
        groq_verdict: str,
        lgbm_verdict: str,
        llm_priority: bool,
    ) -> str:
        """Apply decision table with priority rules."""
        groq_bullish = groq_verdict == "bullish"
        groq_bearish = groq_verdict == "bearish"

        if llm_priority:
            # LLM takes priority: override when catalyst/regime transition
            if groq_bullish:
                return STRONG_BUY if lgbm_is_buy else BUY
            elif groq_bearish:
                return SKIP
            else:
                return BUY if lgbm_is_buy else SKIP

        # Normal regime: ML leads
        if lgbm_is_buy and groq_bullish:
            return STRONG_BUY
        elif lgbm_is_buy and groq_verdict == "neutral":
            return BUY
        elif lgbm_is_buy and groq_bearish:
            return CONFLICT
        elif not lgbm_is_buy and groq_bullish:
            return CONFLICT
        else:
            return SKIP

    async def _get_groq_verdict(
        self,
        symbol: str,
        lgbm_score: float,
        lgbm_verdict: str,
        signal_data: dict,
        regime: str,
    ) -> tuple[str, str]:
        """Ask Groq for a bullish/neutral/bearish verdict on the signal.

        Returns (verdict: str, reasoning: str).
        Falls back to 'neutral' with empty reasoning if Groq is unavailable.
        """
        client = self._get_groq_client()
        if client is None:
            return "neutral", ""

        # Build a compact context for the prompt
        context_lines = []
        for key in ("rsi", "price_vs_50ma", "volume_ratio", "adx", "finbert_score",
                     "news_volume_7d", "threat_level_encoded", "regime_encoded"):
            val = signal_data.get(key)
            if val is not None:
                context_lines.append(f"  {key}: {val}")

        prompt = f"""You are a quantitative trading analyst making a single trade decision.

Symbol: {symbol}
Market regime: {regime}
LightGBM P(win): {lgbm_score:.4f} ({lgbm_verdict})

Signal context:
{chr(10).join(context_lines) if context_lines else '  (no additional context)'}

Based on the technical setup and market regime, give your verdict:
- Is this trade bullish, neutral, or bearish?
- What is your key reason in one sentence?

Reply in JSON:
{{"verdict": "bullish|neutral|bearish", "reasoning": "one sentence explanation"}}

Return only valid JSON, no markdown."""

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                ),
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            verdict = parsed.get("verdict", "neutral").lower()
            if verdict not in ("bullish", "neutral", "bearish"):
                verdict = "neutral"
            reasoning = parsed.get("reasoning", "")
            return verdict, reasoning
        except json.JSONDecodeError:
            logger.debug("Groq returned non-JSON verdict for %s", symbol)
            return "neutral", ""
        except Exception:
            logger.debug("Groq verdict failed for %s", symbol, exc_info=True)
            return "neutral", ""

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            if not self._groq_warned:
                logger.info(
                    "GROQ_API_KEY not set — consensus uses LGBM-only mode. "
                    "All signals default to LightGBM verdict with no LLM check."
                )
                self._groq_warned = True
            return None
        try:
            from groq import Groq  # type: ignore
            self._groq_client = Groq(api_key=api_key)
            return self._groq_client
        except ImportError:
            if not self._groq_warned:
                logger.warning("groq package not installed — run: pip install groq>=0.4.0")
                self._groq_warned = True
            return None
        except Exception:
            logger.exception("Failed to initialize Groq client for consensus")
            return None

    def _log_conflict(self, result: dict) -> None:
        """Append CONFLICT signals to the conflict log for monthly review."""
        try:
            with open(CONFLICT_LOG_PATH, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception:
            logger.debug("Failed to write conflict log: %s", CONFLICT_LOG_PATH)


# ── Synchronous convenience wrapper ──────────────────────────────────────────

def lgbm_only_consensus(lgbm_score: float, high_conf: float = 0.65, low_conf: float = 0.52) -> dict:
    """Fast synchronous consensus when no Groq key is available.

    Used by the scoring cycle when GROQ_API_KEY is not set or when
    Groq is rate-limited.
    """
    if lgbm_score >= high_conf:
        consensus, verdict = STRONG_BUY, "bullish"
    elif lgbm_score >= low_conf:
        consensus, verdict = BUY, "neutral"
    else:
        consensus, verdict = SKIP, "bearish"

    return {
        "lgbm_score":     round(lgbm_score, 4),
        "lgbm_verdict":   verdict,
        "groq_verdict":   "n/a",
        "groq_reasoning": "",
        "consensus":      consensus,
        "size_multiplier": SIZE_MULTIPLIERS[consensus],
        "catalyst_present": False,
        "regime":         "NEUTRAL",
    }
