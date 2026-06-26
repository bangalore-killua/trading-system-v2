"""
symbol_profiler.py — Auto-classify new watchlist additions.

When a symbol enters the watchlist that is not in SECTOR_BUCKETS (scorer.py),
this module:
  1. Fetches background via yfinance.Ticker.info  (1 free API call, cached forever)
  2. Auto-classifies to sector_bucket via deterministic GICS mapping (~95% hit rate)
  3. Falls back to Groq (free) when GICS data is missing or ambiguous
  4. Stores the full profile to data/symbol_profiles.json

The stored profile is loaded by scorer.py at startup and merged into
SECTOR_BUCKETS so all subsequent predictions immediately use the correct sector.

The profile also provides business_summary, beta, and market_cap_tier to
ConsensusEngine prompts — giving Groq context like "semiconductor equipment
monopoly, beta=0.9" vs "meme stock, beta=3.1, short_ratio=18".
"""

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROFILES_PATH = Path("data/symbol_profiles.json")

# Valid sector_bucket values (must match SECTOR_BUCKETS in scorer.py)
_VALID_BUCKETS = {
    "TECH_GROWTH", "SEMICONDUCTOR", "BIOTECH", "HEALTHCARE",
    "FINANCIALS", "FINTECH", "CRYPTO_PROXY", "ENERGY",
    "CONSUMER_DISC", "DEFENSIVE", "CYCLICAL", "DEFENSE", "COMMODITY",
}

# Deterministic GICS sector + industry → sector_bucket
# Exact industry match tried first; sector-level fallback used when no match.
_INDUSTRY_MAP = {
    "Semiconductors":                     "SEMICONDUCTOR",
    "Semiconductor Equipment & Materials": "SEMICONDUCTOR",
    "Biotechnology":                       "BIOTECH",
    "Aerospace & Defense":                 "DEFENSE",
    "Banks—Diversified":                  "FINANCIALS",
    "Banks—Regional":                     "FINANCIALS",
    "Capital Markets":                    "FINANCIALS",
    "Asset Management":                   "FINANCIALS",
    "Credit Services":                    "FINTECH",
    "Insurance—Life":                     "FINANCIALS",
}

_SECTOR_MAP = {
    "Technology":            "TECH_GROWTH",
    "Healthcare":            "HEALTHCARE",
    "Financial Services":    "FINANCIALS",
    "Energy":                "ENERGY",
    "Consumer Cyclical":     "CONSUMER_DISC",
    "Consumer Defensive":    "DEFENSIVE",
    "Industrials":           "CYCLICAL",
    "Basic Materials":       "COMMODITY",
    "Utilities":             "DEFENSIVE",
    "Real Estate":           "DEFENSIVE",
    "Communication Services": "TECH_GROWTH",
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_profile_cache() -> dict:
    """Load all stored symbol profiles from disk. Returns {} on any error."""
    if not PROFILES_PATH.exists():
        return {}
    try:
        with open(PROFILES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def get_sector_bucket(symbol: str, auto_fetch: bool = True) -> str:
    """
    Resolve sector_bucket for a symbol.

    Resolution order:
      1. Hardcoded SECTOR_BUCKETS in scorer.py   (instant, known symbols)
      2. data/symbol_profiles.json cache          (instant, previously profiled)
      3. classify_symbol() fetch + classify       (async-safe blocking call)
      4. "OTHER"                                  (if auto_fetch=False or all fail)
    """
    # Avoid circular import — import inline
    try:
        from strategy.scorer import SECTOR_BUCKETS
        if symbol in SECTOR_BUCKETS:
            return SECTOR_BUCKETS[symbol]
    except ImportError:
        pass

    cache = load_profile_cache()
    if symbol in cache:
        return cache[symbol].get("sector_bucket", "OTHER")

    if auto_fetch:
        try:
            profile = classify_symbol(symbol)
            return profile.get("sector_bucket", "OTHER")
        except Exception as e:
            logger.warning("classify_symbol failed for %s: %s", symbol, e)

    return "OTHER"


def classify_symbol(symbol: str) -> dict:
    """
    Fetch background and classify a new symbol. Idempotent — safe to call repeatedly.

    Returns the stored profile dict (always contains 'sector_bucket').
    Uses cache if the symbol was already profiled.
    """
    cache = load_profile_cache()
    if symbol in cache:
        return cache[symbol]

    logger.info("Profiling new symbol: %s", symbol)

    # Step 1: yfinance background fetch
    info = _fetch_yfinance(symbol)

    gics_sector  = info.get("sector", "")  or ""
    gics_industry = info.get("industry", "") or ""
    beta         = float(info.get("beta") or 1.0)
    market_cap   = int(info.get("marketCap") or 0)
    summary      = info.get("longBusinessSummary", "") or ""
    div_yield    = float(info.get("dividendYield") or 0.0)
    short_ratio  = float(info.get("shortRatio") or 0.0)

    cap_tier = (
        "MEGA"  if market_cap >= 200_000_000_000 else
        "LARGE" if market_cap >= 10_000_000_000  else
        "MID"   if market_cap >= 2_000_000_000   else
        "SMALL"
    )

    # Step 2: deterministic GICS classification
    sector_bucket = _auto_classify(gics_sector, gics_industry)
    source = "yfinance_auto"

    # Step 3: Groq fallback if uncertain
    if sector_bucket is None:
        logger.info(
            "%s: GICS auto-classify uncertain (%s / %s) — using Groq",
            symbol, gics_sector, gics_industry,
        )
        sector_bucket = _groq_classify(symbol, summary, gics_sector, gics_industry)
        source = "groq_fallback" if sector_bucket != "OTHER" else "unknown"

    profile = {
        "sector_bucket":          sector_bucket,
        "gics_sector":            gics_sector,
        "gics_industry":          gics_industry,
        "beta":                   round(beta, 2),
        "market_cap_tier":        cap_tier,
        "dividend_yield":         round(div_yield, 4),
        "short_ratio":            round(short_ratio, 1),
        "business_summary":       summary[:800],
        "classified_at":          date.today().isoformat(),
        "classification_source":  source,
    }

    _save_profile(symbol, profile)
    logger.info("Profiled %s → %s (via %s)", symbol, sector_bucket, source)
    return profile


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str) -> dict:
    try:
        import yfinance as yf
        return yf.Ticker(symbol).info or {}
    except Exception as e:
        logger.warning("yfinance.Ticker(%s).info failed: %s", symbol, e)
        return {}


def _auto_classify(sector: str, industry: str) -> Optional[str]:
    """Deterministic GICS → sector_bucket. Returns None if too uncertain."""
    # Exact industry match
    if industry in _INDUSTRY_MAP:
        return _INDUSTRY_MAP[industry]

    # Keyword scan on industry string
    il = industry.lower()
    if "semiconductor" in il:
        return "SEMICONDUCTOR"
    if "biotech" in il:
        return "BIOTECH"
    if "defense" in il or "aerospace" in il:
        return "DEFENSE"
    if "bank" in il:
        return "FINANCIALS"
    if "capital market" in il or "asset management" in il:
        return "FINANCIALS"
    if "credit" in il or "payment" in il or "fintech" in il:
        return "FINTECH"
    if "crypto" in il or "bitcoin" in il or "digital asset" in il:
        return "CRYPTO_PROXY"
    if "software" in il or "internet" in il or "cloud" in il:
        return "TECH_GROWTH"
    if "oil" in il or "gas" in il or "petroleum" in il:
        return "ENERGY"
    if "gold" in il or "silver" in il or "mining" in il or "copper" in il:
        return "COMMODITY"
    if "pharmaceutical" in il or "drug" in il:
        return "HEALTHCARE"

    # Sector-level fallback
    return _SECTOR_MAP.get(sector)


def _groq_classify(symbol: str, business_summary: str, sector: str, industry: str) -> str:
    """Groq fallback — free, fires only when GICS data is missing or ambiguous."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — cannot classify %s via Groq", symbol)
        return "OTHER"

    try:
        import groq as groq_lib
        client = groq_lib.Groq(api_key=api_key)

        buckets = ", ".join(sorted(_VALID_BUCKETS))
        prompt = (
            f"Symbol: {symbol}\n"
            f"GICS Sector: {sector or 'unknown'}\n"
            f"GICS Industry: {industry or 'unknown'}\n"
            f"Business description: {business_summary[:500] or 'not available'}\n\n"
            f"Classify this stock into exactly one of these trading behaviour buckets:\n"
            f"{buckets}\n\n"
            f"Key distinctions:\n"
            f"  SEMICONDUCTOR = chip designers/makers/equipment (NVDA, AMD, ASML)\n"
            f"  TECH_GROWTH   = software, internet, cloud, SaaS (MSFT, CRM, NFLX)\n"
            f"  BIOTECH       = event-driven FDA pipeline (MRNA, BIIB)\n"
            f"  HEALTHCARE    = steady pharma/managed care (JNJ, UNH)\n"
            f"  DEFENSE       = defense contractors with congressional trades (LMT, RTX)\n"
            f"  CRYPTO_PROXY  = price moves with Bitcoin (COIN, MSTR)\n\n"
            f"Reply with just the label, nothing else."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You classify stocks into trading behaviour buckets. Reply with exactly one label from the list, nothing else."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        label = resp.choices[0].message.content.strip().upper()
        if label in _VALID_BUCKETS:
            return label
        logger.warning("Groq returned unrecognised bucket '%s' for %s — using OTHER", label, symbol)
        return "OTHER"
    except Exception as e:
        logger.warning("Groq classify failed for %s: %s", symbol, e)
        return "OTHER"


def _save_profile(symbol: str, profile: dict) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = load_profile_cache()
    cache[symbol] = profile
    with open(PROFILES_PATH, "w") as f:
        json.dump(cache, f, indent=2)
