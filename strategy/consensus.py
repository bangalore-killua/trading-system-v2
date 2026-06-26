"""
consensus.py — ML + LLM Consensus Layer (Phase 4 + v13 upgrade)

Combines LightGBM/AutoGluon P(win) score with LLM verdict to produce a
STRONG_BUY / BUY / CONFLICT / SKIP decision and a position size multiplier.

v13 additions (Fix 1 from audit + new verdicts):
  - RAG: 3 past cases from ChromaDB, filtered by current regime (regime-matched)
  - system/user prompt split (not single-string prompt)
  - company name + headline context passed to LLM
  - exit_verdict()   — Claude Haiku, decides hold/sell_partial/sell_all
  - dip_verdict()    — Claude Haiku, decides buy_dip/accumulate/avoid
  - tranche_verdict()— Groq (routine/high-volume), decides add/hold/reduce
  - Claude routing: Groq (routine), Haiku (exit/dip), Sonnet (final synthesis)

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
  - CONFLICT is always logged with LLM's full reasoning for audit

AI cost routing:
  Groq  (routine, free):  tranche_verdict, quick consensus checks
  Haiku (~$0.72/mo):      exit_verdict, dip_verdict, research
  Sonnet (~$2/mo):        final orchestrator synthesis only

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

GROQ_MODEL   = "llama-3.3-70b-versatile"   # free — all verdicts + routine tasks
SONNET_MODEL = "claude-sonnet-4-6"          # final synthesis only (~$2/mo)

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

    # ── v13: Exit / Dip / Tranche verdicts ───────────────────────────────────

    async def exit_verdict(self, symbol: str, position_data: dict) -> dict:
        """Groq — decide whether to exit an open position.

        Parameters
        ----------
        position_data : {unrealized_pnl_pct, days_held, regime, news_since_entry,
                         peak_drawdown_pct, entry_composite_score}

        Returns
        -------
        {"action": "hold|sell_partial|sell_all", "urgency": "immediate|next_open|no_rush",
         "reasoning": str}
        """
        client = self._get_groq_client()
        if client is None:
            return {"action": "hold", "urgency": "no_rush", "reasoning": "Groq unavailable"}

        # Enrich prompt with symbol profile context if available
        profile_context = self._get_symbol_context(symbol)
        prompt = (
            f"Symbol: {symbol}{profile_context}\n"
            f"Position: {json.dumps(position_data, default=str)}\n\n"
            f"Should we exit this position? Consider unrealized P&L, days held, "
            f"regime, peak drawdown, entry composite score, and any news since entry.\n"
            f'Reply JSON only: {{"action": "hold|sell_partial|sell_all", '
            f'"urgency": "immediate|next_open|no_rush", "reasoning": "one sentence"}}'
        )
        try:
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a position exit advisor. Reply JSON only."},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=150,
                ),
            )
            raw = response.choices[0].message.content.strip().lstrip("```json").lstrip("```").rstrip("```")
            result = json.loads(raw)
            logger.info("exit_verdict %s: %s", symbol, result.get("action"))
            return result
        except Exception as e:
            logger.debug("exit_verdict Groq call failed: %s", e)
            return {"action": "hold", "urgency": "no_rush", "reasoning": "unavailable"}

    async def dip_verdict(self, symbol: str, dip_data: dict) -> dict:
        """Groq — evaluate whether a price dip is a buying opportunity.

        Parameters
        ----------
        dip_data : {drop_pct, drop_speed_days, cause, regime, quality_score,
                    insider_activity, eps_beat_streak}

        Returns
        -------
        {"verdict": "buy_dip|accumulate|avoid", "conviction": "high|medium|low",
         "reasoning": str}
        """
        client = self._get_groq_client()
        if client is None:
            return {"verdict": "avoid", "conviction": "low", "reasoning": "Groq unavailable"}

        profile_context = self._get_symbol_context(symbol)
        prompt = (
            f"Symbol: {symbol}{profile_context}\n"
            f"Dip data: {json.dumps(dip_data, default=str)}\n\n"
            f"Is this dip a buying opportunity or a falling knife? Consider drop magnitude, "
            f"speed, cause, regime, quality score, insider activity, and earnings streak.\n"
            f'Reply JSON only: {{"verdict": "buy_dip|accumulate|avoid", '
            f'"conviction": "high|medium|low", "reasoning": "one sentence"}}'
        )
        try:
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a dip-buying advisor. Reply JSON only."},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=150,
                ),
            )
            raw = response.choices[0].message.content.strip().lstrip("```json").lstrip("```").rstrip("```")
            result = json.loads(raw)
            logger.info("dip_verdict %s: %s (%s)", symbol, result.get("verdict"), result.get("conviction"))
            return result
        except Exception as e:
            logger.debug("dip_verdict Groq call failed: %s", e)
            return {"verdict": "avoid", "conviction": "low", "reasoning": "unavailable"}

    async def tranche_verdict(self, symbol: str, position_data: dict) -> dict:
        """Groq (routine, free) — decide tranche add/hold/reduce.

        Parameters
        ----------
        position_data : {current_pnl, composite_score, tranche_level, available_cash_pct,
                         momentum_confirming, atr_threshold_hit}

        Returns
        -------
        {"action": "add_tranche|hold|reduce", "reasoning": str}
        """
        client = self._get_groq_client()
        if client is None:
            return {"action": "hold", "reasoning": "Groq unavailable"}

        prompt = (
            f"Symbol: {symbol}\n"
            f"Position: {json.dumps(position_data, default=str)}\n\n"
            f"Should we add a tranche (scale up), hold, or reduce? "
            f"Reply JSON only: {{\"action\": \"add_tranche|hold|reduce\", \"reasoning\": \"one sentence\"}}"
        )
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a position sizing assistant. Reply JSON only."},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=100,
                ),
            )
            raw = response.choices[0].message.content.strip().lstrip("```json").lstrip("```").rstrip("```")
            result = json.loads(raw)
            logger.info("tranche_verdict %s: %s", symbol, result.get("action"))
            return result
        except Exception as e:
            logger.debug("tranche_verdict Groq call failed: %s", e)
            return {"action": "hold", "reasoning": "unavailable"}

    # ── v13: RAG-enhanced Groq prompt (Fix 1 from audit) ─────────────────────

    async def _get_groq_verdict(
        self,
        symbol: str,
        lgbm_score: float,
        lgbm_verdict: str,
        signal_data: dict,
        regime: str,
    ) -> tuple[str, str]:
        """Ask Groq for a bullish/neutral/bearish verdict on the signal.

        v13 fix: system/user split + company name + headlines + regime-matched RAG.
        Returns (verdict: str, reasoning: str).
        """
        client = self._get_groq_client()
        if client is None:
            return "neutral", ""

        # ── RAG: 3 regime-matched past cases from ChromaDB ────────────────────
        rag_context = self._fetch_rag_context(signal_data, regime, n=3)

        # ── Build context ─────────────────────────────────────────────────────
        company_name = signal_data.get("company_name", symbol)
        headlines    = signal_data.get("headlines", [])
        headline_str = "\n".join(f"  - {h}" for h in headlines[:5]) if headlines else "  (none)"

        context_lines = []
        for key in ("rsi", "price_vs_50ma", "volume_ratio", "adx", "finbert_score",
                     "news_volume_7d", "threat_level_encoded", "regime_encoded",
                     "options_score", "macro_score", "4h_trend", "insider_score"):
            val = signal_data.get(key)
            if val is not None:
                context_lines.append(f"  {key}: {val}")

        system_prompt = (
            "You are a quantitative trading analyst. You will be given a signal and recent cases. "
            "Output JSON only. No markdown, no explanation outside JSON."
        )
        user_prompt = (
            f"Company: {company_name} ({symbol})\n"
            f"Regime: {regime}\n"
            f"Model P(win): {lgbm_score:.4f} ({lgbm_verdict})\n\n"
            f"Signal context:\n" + ("\n".join(context_lines) or "  (none)") + "\n\n"
            f"Recent headlines:\n{headline_str}\n\n"
            f"Similar past cases (same regime):\n{rag_context}\n\n"
            f"Verdict: {{\"verdict\": \"bullish|neutral|bearish\", \"reasoning\": \"one sentence\"}}"
        )

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=150,
                ),
            )
            content = response.choices[0].message.content.strip()
            content = content.lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed  = json.loads(content)
            verdict = parsed.get("verdict", "neutral").lower()
            if verdict not in ("bullish", "neutral", "bearish"):
                verdict = "neutral"
            return verdict, parsed.get("reasoning", "")
        except json.JSONDecodeError:
            logger.debug("Groq returned non-JSON for %s", symbol)
            return "neutral", ""
        except Exception:
            logger.debug("Groq verdict failed for %s", symbol, exc_info=True)
            return "neutral", ""

    def _fetch_rag_context(self, signal_data: dict, regime: str, n: int = 3) -> str:
        """Fetch n regime-matched past cases from ChromaDB for RAG prompt injection."""
        try:
            import chromadb  # type: ignore
            client = chromadb.PersistentClient(path="data/chromadb")
            coll   = client.get_collection("feedback_records")

            composite = signal_data.get("composite_score", 0.5)
            results   = coll.query(
                query_texts=[str(composite)],
                where={"regime": {"$eq": regime}},
                n_results=n,
            )
            docs = results.get("documents", [[]])[0]
            if not docs:
                return "  (no past cases in this regime)"
            return "\n".join(f"  Case {i+1}: {d[:150]}" for i, d in enumerate(docs))
        except Exception as e:
            logger.debug("RAG fetch failed: %s", e)
            return "  (RAG unavailable)"

    def _get_symbol_context(self, symbol: str) -> str:
        """Return a short context string from symbol_profiler for use in Groq prompts."""
        try:
            from data.sources.symbol_profiler import load_profile_cache
            profile = load_profile_cache().get(symbol)
            if not profile:
                return ""
            parts = []
            if profile.get("sector_bucket"):
                parts.append(f"sector={profile['sector_bucket']}")
            if profile.get("beta"):
                parts.append(f"beta={profile['beta']}")
            if profile.get("market_cap_tier"):
                parts.append(f"cap={profile['market_cap_tier']}")
            if profile.get("business_summary"):
                parts.append(f"business={profile['business_summary'][:150]}")
            return f" ({', '.join(parts)})" if parts else ""
        except Exception:
            return ""

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
