"""
research_agent.py — Real-time web research agent (Claude Haiku + Playwright).

Extends BaseAgent to add:
  1. Claude Haiku (claude-haiku-4-5-20251001) for qualitative research calls.
     Groq handles routine tasks; Haiku handles research + exit/dip verdicts.
     Cost estimate: ~$0.72/month at 15 research calls/day.
  2. Playwright browser for live web navigation (SEC, MarketBeat, Finviz).
  3. Multi-source research: browses 3-4 sources, synthesises to structured output.

Usage in signal orchestrator:
    agent   = ResearchAgent()
    summary = agent.research("NVDA", context_needed="latest analyst ratings and risks")
    # Returns: {"headlines": [...], "analyst_summary": str, "risk_flags": [...], "sentiment": str}

Usage for exit verdicts:
    verdict = agent.exit_verdict("NVDA", position_data)
    # Returns: {"action": "hold|sell_partial|sell_all", "urgency": str, "reasoning": str}

Usage for dip verdicts:
    verdict = agent.dip_verdict("NVDA", dip_data)
    # Returns: {"verdict": "buy_dip|accumulate|avoid", "conviction": str, "reasoning": str}
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from data.fincept.agents.base import BaseAgent  # type: ignore

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
GROQ_MODEL  = "llama-3.3-70b-versatile"  # fallback for research if Haiku unavailable

_RESEARCH_SYSTEM = """You are a financial research analyst. Given a stock symbol and research question, you:
1. Summarise what you know about current analyst sentiment, recent news, and key risks.
2. Identify any qualitative factors the numerical signals might be missing.
3. Flag anything anomalous — regulatory risk, executive departure, narrative shift.
Be concise (max 300 words). Return JSON with keys: headlines, analyst_summary, risk_flags, sentiment.
sentiment must be exactly one of: "bullish", "bearish", "neutral".
risk_flags must be a list of strings. headlines must be a list of short strings."""

_EXIT_SYSTEM = """You are a trading position manager evaluating whether to exit a position.
Given position data (P&L, days held, regime, news), decide the exit action.
Be concise. Return JSON with exactly these keys:
  action: "hold" | "sell_partial" | "sell_all"
  urgency: "immediate" | "next_open" | "no_rush"
  reasoning: one sentence max"""

_DIP_SYSTEM = """You are a fundamental analyst evaluating whether a price dip is a buying opportunity.
Given dip data (drop magnitude, cause, fundamentals), give your verdict.
Be concise. Return JSON with exactly these keys:
  verdict: "buy_dip" | "accumulate" | "avoid"
  conviction: "high" | "medium" | "low"
  reasoning: one sentence max"""


class ResearchAgent(BaseAgent):
    """Claude Haiku-powered research agent with optional Playwright web browsing.

    Falls back to Groq if Haiku/Anthropic key not configured.
    """

    def __init__(self):
        super().__init__(
            name="research_agent",
            system_prompt=_RESEARCH_SYSTEM,
        )
        self._anthropic_client = None

    # ── Public interface ──────────────────────────────────────────────────────

    def research(self, symbol: str, context_needed: str, browse: bool = True) -> dict:
        """Research a symbol. Optionally uses Playwright to browse live web sources.

        Parameters
        ----------
        symbol         : ticker, e.g. "NVDA"
        context_needed : research question, e.g. "latest analyst ratings and risks"
        browse         : if True, browse 1-2 web sources before calling Haiku

        Returns
        -------
        {"headlines": list, "analyst_summary": str, "risk_flags": list, "sentiment": str}
        """
        web_context = ""
        if browse:
            web_context = self._browse_for_context(symbol)

        prompt = (
            f"Symbol: {symbol}\n"
            f"Research question: {context_needed}\n\n"
            f"Live web context:\n{web_context if web_context else '(no live context available)'}\n\n"
            f"Return JSON only."
        )
        raw = self._call_haiku(system=_RESEARCH_SYSTEM, user=prompt)
        return self._parse_json(raw, _research_neutral(symbol))

    def exit_verdict(self, symbol: str, position_data: dict) -> dict:
        """Evaluate whether to exit an open position.

        Parameters
        ----------
        position_data: dict with keys like unrealized_pnl_pct, days_held, regime,
                       news_since_entry, peak_drawdown_pct, entry_composite_score

        Returns
        -------
        {"action": str, "urgency": str, "reasoning": str}
        """
        prompt = (
            f"Symbol: {symbol}\n"
            f"Position: {json.dumps(position_data, default=str)}\n\n"
            f"Evaluate exit. Return JSON only."
        )
        raw = self._call_haiku(system=_EXIT_SYSTEM, user=prompt)
        return self._parse_json(raw, {"action": "hold", "urgency": "no_rush", "reasoning": "unavailable"})

    def dip_verdict(self, symbol: str, dip_data: dict) -> dict:
        """Evaluate whether a price dip is a buying opportunity.

        Parameters
        ----------
        dip_data: dict with keys like drop_pct, drop_speed_days, cause, regime,
                  quality_score, insider_activity, eps_beat_streak

        Returns
        -------
        {"verdict": str, "conviction": str, "reasoning": str}
        """
        prompt = (
            f"Symbol: {symbol}\n"
            f"Dip data: {json.dumps(dip_data, default=str)}\n\n"
            f"Evaluate dip. Return JSON only."
        )
        raw = self._call_haiku(system=_DIP_SYSTEM, user=prompt)
        return self._parse_json(raw, {"verdict": "avoid", "conviction": "low", "reasoning": "unavailable"})

    # ── BaseAgent required method ─────────────────────────────────────────────

    def _format_user_message(self, context: Dict[str, Any]) -> str:
        symbol = context.get("symbol", "UNKNOWN")
        return f"Research symbol: {symbol}\n{json.dumps(context, default=str)}"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_haiku(self, system: str, user: str) -> str:
        """Call Claude Haiku. Falls back to Groq if Anthropic key not set."""
        client = self._get_anthropic_client()
        if client:
            try:
                msg = client.messages.create(
                    model=HAIKU_MODEL,
                    max_tokens=512,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text.strip()
            except Exception as e:
                logger.warning("ResearchAgent: Haiku call failed — %s. Falling back to Groq.", e)

        # Fallback: Groq
        groq = self._get_groq_client()
        if groq:
            try:
                resp = groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("ResearchAgent: Groq fallback also failed — %s", e)

        return "{}"

    def _get_anthropic_client(self):
        if self._anthropic_client is not None:
            return self._anthropic_client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("ANTHROPIC_API_KEY not set — ResearchAgent using Groq fallback")
            return None
        try:
            import anthropic  # type: ignore
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            return self._anthropic_client
        except ImportError:
            logger.warning("anthropic package not installed — run: pip install anthropic")
            return None
        except Exception as e:
            logger.warning("ResearchAgent: failed to init Anthropic client — %s", e)
            return None

    def _browse_for_context(self, symbol: str) -> str:
        """Browse Finviz news for the symbol (lightweight, reliable)."""
        try:
            from data.sources.browser_scraper import BrowserScraper
            from bs4 import BeautifulSoup

            url = f"https://finviz.com/quote.ashx?t={symbol}"
            with BrowserScraper(stealth=True) as s:
                html = s.fetch(url, wait_for="#news-table, .news-table", delay=(2.0, 4.0), timeout=12_000)

            soup  = BeautifulSoup(html, "html.parser")
            rows  = soup.select("#news-table tr, .news-table tr")[:8]
            lines = []
            for row in rows:
                link = row.select_one("a")
                if link:
                    lines.append(link.get_text(strip=True)[:120])
            return "\n".join(lines) if lines else ""
        except Exception as e:
            logger.debug("ResearchAgent: browse failed — %s", e)
            return ""

    def _parse_json(self, raw: str, default: dict) -> dict:
        """Parse JSON from model output. Returns default on failure."""
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.debug("ResearchAgent: failed to parse JSON — %s", raw[:100])
            return default


# ── Helpers ────────────────────────────────────────────────────────────────────

def _research_neutral(symbol: str) -> dict:
    return {
        "headlines":      [],
        "analyst_summary": f"Research unavailable for {symbol}",
        "risk_flags":     [],
        "sentiment":      "neutral",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sym   = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    agent = ResearchAgent()
    result = agent.research(sym, "latest analyst ratings and key risks", browse=False)
    print(json.dumps(result, indent=2))
