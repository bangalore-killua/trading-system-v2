"""
finra_dark_pool.py — FINRA ATS Dark Pool Volume Signal

Fetches weekly ATS (Alternative Trading System) transparency data from FINRA's
public API. No API key required. Published every week covering the prior week.

Dark pool interpretation for swing trading:
  High dark_pool_pct (>30% of total volume) on a stock with positive signals
  = institutions are accumulating quietly before a public move.
  Low dark_pool_pct = retail-driven price action, less reliable.

Data source: https://ats.finra.org / https://api.finra.org/data/group/otcMarket
Cadence: weekly (FINRA publishes each Monday for the prior week)

Output: dark_pool_score 0-1
  - Score computed as universe percentile rank of dark_pool_pct
  - A stock in the 90th percentile for dark pool activity scores 0.90

Usage:
    from data.sources.finra_dark_pool import get_dark_pool_score, refresh_dark_pool_cache
    refresh_dark_pool_cache()               # call once per week / on startup
    score, detail = get_dark_pool_score("NVDA")
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# FINRA OTC Transparency — ATS weekly summary endpoint
_FINRA_ATS_URL = (
    "https://api.finra.org/data/group/otcMarket/name/weeklyDownloadDetails"
    "?limit=5000&offset=0"
)
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "bangalore.killua@gmail.com (personal trading research)",
}

_dark_pool_data: Dict[str, float] = {}   # symbol → dark_pool_pct (0.0–1.0)
_universe_pct: list = []                  # sorted pcts for percentile ranking
_last_refresh: Optional[datetime] = None
_REFRESH_INTERVAL_DAYS = 7

_cache: Dict[str, Tuple[float, dict]] = {}  # symbol → (score, detail)


def refresh_dark_pool_cache(force: bool = False) -> bool:
    """
    Download the latest FINRA ATS weekly summary and build the dark_pool lookup.
    Returns True on success, False on failure (stale cache retained).
    """
    global _dark_pool_data, _universe_pct, _last_refresh

    if not force and _last_refresh:
        age = (datetime.now() - _last_refresh).days
        if age < _REFRESH_INTERVAL_DAYS:
            logger.debug("FINRA dark pool cache is %d days old — skip refresh", age)
            return True

    try:
        resp = requests.get(_FINRA_ATS_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        records = resp.json()

        if not records:
            logger.warning("FINRA ATS returned empty response")
            return False

        raw: Dict[str, dict] = {}
        for rec in records:
            sym   = rec.get("issueSymbolIdentifier", "").strip().upper()
            total = float(rec.get("totalWeeklyShareQuantity", 0) or 0)
            ats   = float(rec.get("atsWeeklyShareQuantity",   0) or 0)
            if sym and total > 0:
                pct = ats / total
                # Keep the highest-volume week if symbol appears multiple times
                if sym not in raw or raw[sym]["total"] < total:
                    raw[sym] = {"pct": pct, "total": total, "ats": ats}

        _dark_pool_data = {sym: v["pct"] for sym, v in raw.items()}
        _universe_pct   = sorted(_dark_pool_data.values())
        _last_refresh   = datetime.now()
        _cache.clear()

        logger.info("FINRA dark pool cache refreshed: %d symbols", len(_dark_pool_data))
        return True

    except requests.exceptions.RequestException as e:
        logger.warning("FINRA ATS fetch failed: %s — using cached data", e)
        return False
    except Exception:
        logger.exception("FINRA dark pool cache refresh failed")
        return False


def _percentile_rank(pct: float) -> float:
    """Position of pct in the universe distribution → 0.0–1.0 percentile score."""
    if not _universe_pct:
        return 0.5
    below = sum(1 for v in _universe_pct if v < pct)
    return round(below / len(_universe_pct), 3)


def get_dark_pool_score(symbol: str) -> Tuple[float, dict]:
    """
    Returns (score 0-1, detail dict).
    Returns (0.5, {}) if symbol not found — neutral default.

    Score = percentile rank of symbol's dark pool % vs all tracked symbols.
    A score of 0.85 means 85% of stocks have LESS dark pool activity → high institutional interest.
    """
    if symbol in _cache:
        return _cache[symbol]

    if not _dark_pool_data:
        _try_auto_refresh()

    raw_pct = _dark_pool_data.get(symbol)
    if raw_pct is None:
        return 0.5, {"note": "symbol_not_in_finra_ats", "symbol": symbol}

    score = _percentile_rank(raw_pct)
    detail = {
        "dark_pool_pct":       round(raw_pct * 100, 1),   # e.g. 34.2 (%)
        "percentile_rank":     score,
        "data_age_days":       (datetime.now() - _last_refresh).days if _last_refresh else None,
    }
    _cache[symbol] = (score, detail)
    return score, detail


def _try_auto_refresh() -> None:
    """Attempt a refresh if cache is empty or very stale."""
    global _last_refresh
    if _last_refresh is None or (datetime.now() - _last_refresh).days >= _REFRESH_INTERVAL_DAYS:
        refresh_dark_pool_cache()
