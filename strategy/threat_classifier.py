"""
threat_classifier.py — Word-List NLP Threat Classifier

Classifies news headlines into CRITICAL / HIGH / MEDIUM / LOW threat levels.
Uses a weighted keyword approach (fast, no GPU needed) rather than a model,
so it can run in real-time on every article before the expensive FinBERT pass.

Threat levels control two decisions:
  CRITICAL  → block ALL new entries + trigger immediate exit of open positions
  HIGH      → block new entries; existing positions monitored more aggressively
  MEDIUM    → reduce position size on new entries; no exit pressure
  LOW       → no action change; informational

Source credibility modifiers:
  STATE_MEDIA → downgrade threat by one level (propaganda risk)
  CAUTION     → treat as unverified until corroborated
  NONE        → accept at face value

Adapted from blm_terminal_windy/backend/app/services/nlp_service.py
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Keyword lists (each term carries a score; higher = more severe) ────────────

CRITICAL_TERMS: Dict[str, float] = {
    "market crash": 4.0,
    "circuit breaker": 4.0,
    "trading halted": 4.0,
    "trading suspended": 4.0,
    "stock exchange closed": 4.0,
    "federal reserve emergency": 4.0,
    "bank failure": 3.5,
    "bank run": 3.5,
    "systemic risk": 3.5,
    "nuclear": 3.5,
    "world war": 3.5,
    "pandemic": 3.0,
    "emergency rate cut": 3.0,
    "emergency meeting fed": 3.0,
    "flash crash": 3.0,
    "going concern": 3.0,
    "bankruptcy filing": 3.0,
    "chapter 11": 3.0,
    "chapter 7": 3.0,
    "insolvency": 3.0,
    "default on debt": 3.0,
    "sovereign default": 3.5,
    "financial contagion": 3.5,
    "recession declared": 3.0,
    "depression": 3.0,
    "hyperinflation": 3.0,
}

HIGH_TERMS: Dict[str, float] = {
    "recession": 2.0,
    "stagflation": 2.0,
    "rate hike": 1.5,
    "rate cut": 1.0,
    "fed raises": 1.5,
    "fed cuts": 1.0,
    "interest rate rise": 1.5,
    "inflation surge": 2.0,
    "cpi above": 1.5,
    "yield spike": 1.5,
    "inverted yield": 2.0,
    "credit crunch": 2.0,
    "liquidity crisis": 2.5,
    "war": 2.0,
    "sanctions": 1.5,
    "trade war": 1.5,
    "tariff": 1.0,
    "geopolitical": 1.5,
    "oil shock": 2.0,
    "energy crisis": 1.5,
    "supply chain crisis": 1.5,
    "fraud": 2.0,
    "accounting irregularity": 2.5,
    "sec investigation": 2.5,
    "doj investigation": 2.5,
    "class action": 1.5,
    "executive arrested": 2.5,
    "ceo ousted": 1.5,
    "mass layoffs": 1.5,
    "guidance cut": 2.0,
    "profit warning": 2.0,
    "earnings miss": 1.5,
    "revenue miss": 1.5,
    "write-down": 1.5,
    "impairment": 1.5,
    "goodwill write": 1.5,
    "restatement": 2.5,
    "going private": 1.0,
    "delisted": 2.0,
    "forced selling": 2.0,
    "margin call": 2.5,
}

MEDIUM_TERMS: Dict[str, float] = {
    "uncertainty": 0.8,
    "volatile": 0.8,
    "volatility": 0.8,
    "headwinds": 0.8,
    "challenges": 0.5,
    "slowdown": 1.0,
    "contraction": 1.0,
    "decline": 0.8,
    "miss estimates": 1.0,
    "below expectations": 1.0,
    "caution": 0.8,
    "warning": 0.8,
    "concern": 0.5,
    "risk": 0.5,
    "downgrade": 1.0,
    "cut rating": 1.0,
    "sell rating": 1.0,
    "lower target": 0.8,
    "weak guidance": 1.0,
    "disappointing": 0.8,
}

# Sources that systematically exaggerate threats (reduce threat level by 1)
STATE_MEDIA_SOURCES = {
    "xinhua",
    "cgtn",
    "rt ",
    "russia today",
    "sputnik",
    "tass",
    "people's daily",
    "global times",
    "press tv",
    "al-manar",
}

# Sources that may have editorial bias (require corroboration before acting on CRITICAL)
CAUTION_SOURCES = {
    "zerohedge",
    "zero hedge",
    "infowars",
    "breitbart",
}


@dataclass
class ThreatResult:
    level: str              # CRITICAL / HIGH / MEDIUM / LOW
    score: float            # raw weighted score
    matched_terms: List[str] = field(default_factory=list)
    source_modifier: str = "NONE"  # STATE_MEDIA / CAUTION / NONE


class ThreatClassifier:
    """Fast keyword-based threat classifier for news headlines.

    Designed to run synchronously before FinBERT so expensive sentiment
    inference is skipped entirely on CRITICAL/HIGH articles.
    """

    LEVEL_THRESHOLDS = {
        "CRITICAL": 3.0,
        "HIGH": 1.5,
        "MEDIUM": 0.5,
        "LOW": 0.0,
    }

    def classify(self, text: str, source_name: str = "") -> str:
        """Classify text. Returns the threat level string."""
        result = self.classify_full(text, source_name)
        return result.level

    def classify_full(self, text: str, source_name: str = "") -> ThreatResult:
        """Full classification returning score and matched terms."""
        lower_text = text.lower()

        # Check source credibility modifier
        source_lower = source_name.lower()
        if any(s in source_lower for s in STATE_MEDIA_SOURCES):
            source_modifier = "STATE_MEDIA"
        elif any(s in source_lower for s in CAUTION_SOURCES):
            source_modifier = "CAUTION"
        else:
            source_modifier = "NONE"

        # Accumulate score from keyword matches
        score = 0.0
        matched = []

        # Check critical terms first (short-circuit if above CRITICAL threshold)
        for term, weight in CRITICAL_TERMS.items():
            if term in lower_text:
                score += weight
                matched.append(f"CRITICAL:{term}")
                if score >= self.LEVEL_THRESHOLDS["CRITICAL"]:
                    break

        # Only check lower-tier terms if still below CRITICAL
        if score < self.LEVEL_THRESHOLDS["CRITICAL"]:
            for term, weight in HIGH_TERMS.items():
                if term in lower_text:
                    score += weight
                    matched.append(f"HIGH:{term}")

        if score < self.LEVEL_THRESHOLDS["HIGH"]:
            for term, weight in MEDIUM_TERMS.items():
                if term in lower_text:
                    score += weight
                    matched.append(f"MEDIUM:{term}")

        # Apply source downgrade: STATE_MEDIA reduces score (propaganda risk)
        if source_modifier == "STATE_MEDIA":
            score *= 0.6  # 40% score reduction

        # Map score to level
        level = self._score_to_level(score)

        return ThreatResult(
            level=level,
            score=round(score, 2),
            matched_terms=matched[:10],  # cap for logging
            source_modifier=source_modifier,
        )

    def _score_to_level(self, score: float) -> str:
        if score >= self.LEVEL_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        if score >= self.LEVEL_THRESHOLDS["HIGH"]:
            return "HIGH"
        if score >= self.LEVEL_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def is_blocking(self, threat_level: str) -> bool:
        """Returns True if this threat level should block new position entry."""
        return threat_level in ("CRITICAL", "HIGH")

    def is_exit_trigger(self, threat_level: str) -> bool:
        """Returns True if this threat level should trigger immediate exit."""
        return threat_level == "CRITICAL"
