"""
base.py — AgentFactory base class for Fincept-style AI agents.

All trading agents inherit from BaseAgent. The model is Groq Llama 3.3 70B
(free tier, 14,400 requests/day).

Each agent:
  - Has a system prompt defining its role and output format
  - Accepts a context dict (portfolio, market, signals)
  - Returns a structured string report (markdown formatted)
  - Caches the last response for the dashboard (no re-run if context unchanged)
"""

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"


class BaseAgent(ABC):
    """Abstract base class for all trading AI agents."""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self._system_prompt = system_prompt
        self._groq_client = None
        self._last_context_hash: Optional[str] = None
        self._last_response: Optional[str] = None

    def run(self, context: Dict[str, Any]) -> str:
        """Run the agent with given context. Returns markdown report string.

        Uses cached response if context has not changed.
        """
        ctx_hash = hashlib.md5(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
        if ctx_hash == self._last_context_hash and self._last_response:
            logger.debug("Agent %s: returning cached response", self.name)
            return self._last_response

        client = self._get_groq_client()
        if client is None:
            return f"**{self.name}** — Groq API key not configured. Set GROQ_API_KEY in .env."

        user_message = self._format_user_message(context)
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            result = response.choices[0].message.content.strip()
            self._last_context_hash = ctx_hash
            self._last_response = result
            return result
        except Exception:
            logger.exception("Agent %s: Groq call failed", self.name)
            return f"**{self.name}** — Analysis unavailable (API error). Check logs."

    @abstractmethod
    def _format_user_message(self, context: Dict[str, Any]) -> str:
        """Format the user-facing message from context. Override per agent."""

    def _get_groq_client(self):
        if self._groq_client is not None:
            return self._groq_client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.info("GROQ_API_KEY not set — agent %s unavailable", self.name)
            return None
        try:
            from groq import Groq  # type: ignore
            self._groq_client = Groq(api_key=api_key)
            return self._groq_client
        except ImportError:
            logger.warning("groq package not installed — run: pip install groq>=0.4.0")
            return None
        except Exception:
            logger.exception("Failed to init Groq client for agent %s", self.name)
            return None


def get_agent_factory():
    """Returns a dict of available agent instances (lazy-initialized)."""
    from data.fincept.agents.portfolio_manager import PortfolioManagerAgent
    from data.fincept.agents.risk_quant import RiskQuantAgent
    from data.fincept.agents.signal_scientist import SignalScientistAgent
    return {
        "portfolio_manager": PortfolioManagerAgent(),
        "risk_quant":        RiskQuantAgent(),
        "signal_scientist":  SignalScientistAgent(),
    }
