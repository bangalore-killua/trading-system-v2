"""
hot_universe.py — Event-Driven Symbol TTL Cache (Layer 2 of 3-layer universe)

Symbols enter HOT_UNIVERSE when triggered by:
  - Alpaca NewsDataStream: bullish FinBERT score + not CRITICAL/HIGH threat
  - SEC EDGAR RSS: material 8-K filing or Form 4 insider buy
  - TradingView Screener: unusual volume + price breakout

On entry the symbol is immediately queued for signal basket evaluation,
bypassing the 60-minute static watchlist cycle.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TriggerSource:
    NEWS_EVENT = "NEWS_EVENT"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    EDGAR_8K = "EDGAR_8K"
    EDGAR_FORM4 = "EDGAR_FORM4"
    TV_SCREENER = "TV_SCREENER"


@dataclass
class HotEntry:
    symbol: str
    source: str
    added_at: datetime
    expires_at: datetime
    headline: Optional[str] = None
    sentiment_score: Optional[float] = None
    trigger_metadata: Dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at

    def ttl_remaining_seconds(self) -> float:
        delta = self.expires_at - datetime.utcnow()
        return max(0.0, delta.total_seconds())


class HotUniverse:
    """Thread-safe TTL cache for event-triggered symbols.

    Downstream consumers poll `get_active_symbols()` or register a callback
    via `on_symbol_added()` to be notified immediately on new entries.
    """

    # TTL per trigger source (hours) — overridden by config.json at runtime.
    # News-driven entries use catalyst-based durations (weeks) so that discovered
    # stocks stay in scope long enough for the weekly scoring cycle to act on them.
    # TV_SCREENER / VOLUME_SPIKE use short TTLs (intraday breakout signals only).
    DEFAULT_TTL_HOURS: Dict[str, float] = {
        TriggerSource.NEWS_EVENT: 336.0,   # 2 weeks  (earnings / FDA catalyst default)
        TriggerSource.VOLUME_SPIKE: 2.0,   # 2 hours  (intraday only)
        TriggerSource.EDGAR_8K: 1008.0,    # 6 weeks  (material agreements / partnerships)
        TriggerSource.EDGAR_FORM4: 336.0,  # 2 weeks  (insider buy signal)
        TriggerSource.TV_SCREENER: 2.0,    # 2 hours  (momentum breakout, intraday)
    }

    # Groq-assigned catalyst durations override the defaults above when provided.
    # Keys map catalyst_type strings returned by Groq to TTL in hours.
    CATALYST_TTL_HOURS: Dict[str, float] = {
        "earnings": 336.0,       # 2 weeks
        "fda": 336.0,            # 2 weeks
        "partnership": 1008.0,   # 6 weeks
        "merger": 2160.0,        # 3 months
        "acquisition": 2160.0,   # 3 months
        "default": 672.0,        # 4 weeks
    }

    def __init__(self, ttl_config: Optional[Dict[str, float]] = None):
        # symbol → most-recent HotEntry (one entry per symbol regardless of how many triggers)
        self._entries: Dict[str, HotEntry] = {}
        self._lock = threading.RLock()
        self._callbacks: list = []

        if ttl_config:
            self.ttl_hours = {**self.DEFAULT_TTL_HOURS, **ttl_config}
        else:
            self.ttl_hours = dict(self.DEFAULT_TTL_HOURS)

    def add(
        self,
        symbol: str,
        source: str,
        headline: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        metadata: Optional[Dict] = None,
        catalyst_type: Optional[str] = None,
    ) -> bool:
        """Add or refresh a symbol. Returns True if this is a brand-new entry.

        catalyst_type overrides the source-based TTL when provided. Use the
        CATALYST_TTL_HOURS mapping (earnings, fda, partnership, merger, etc.)
        for Groq-assigned durations.
        """
        if catalyst_type and catalyst_type in self.CATALYST_TTL_HOURS:
            ttl_h = self.CATALYST_TTL_HOURS[catalyst_type]
        else:
            ttl_h = self.ttl_hours.get(source, 2.0)
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=ttl_h)

        with self._lock:
            is_new = symbol not in self._entries or self._entries[symbol].is_expired()
            self._entries[symbol] = HotEntry(
                symbol=symbol,
                source=source,
                added_at=now,
                expires_at=expires_at,
                headline=headline,
                sentiment_score=sentiment_score,
                trigger_metadata=metadata or {},
            )

        if is_new:
            logger.info(
                "HOT_UNIVERSE +%s via %s (TTL %.1fh) — %s",
                symbol,
                source,
                ttl_h,
                headline or "",
            )
            self._fire_callbacks(symbol, source)

        return is_new

    def remove(self, symbol: str) -> None:
        with self._lock:
            self._entries.pop(symbol, None)

    def get_active_symbols(self) -> list[str]:
        """Return all non-expired symbols."""
        with self._lock:
            self._purge_expired()
            return list(self._entries.keys())

    def get_entry(self, symbol: str) -> Optional[HotEntry]:
        with self._lock:
            entry = self._entries.get(symbol)
            return entry if entry and not entry.is_expired() else None

    def is_active(self, symbol: str) -> bool:
        entry = self.get_entry(symbol)
        return entry is not None

    def size(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._entries)

    def on_symbol_added(self, callback) -> None:
        """Register callback(symbol, source) fired whenever a new symbol enters."""
        self._callbacks.append(callback)

    def _fire_callbacks(self, symbol: str, source: str) -> None:
        for cb in self._callbacks:
            try:
                cb(symbol, source)
            except Exception:
                logger.exception("HOT_UNIVERSE callback error for %s", symbol)

    def _purge_expired(self) -> None:
        # Must be called with self._lock held
        expired = [s for s, e in self._entries.items() if e.is_expired()]
        for s in expired:
            del self._entries[s]
            logger.debug("HOT_UNIVERSE -%s (TTL expired)", s)

    def summary(self) -> str:
        with self._lock:
            self._purge_expired()
            if not self._entries:
                return "HOT_UNIVERSE: empty"
            lines = [f"HOT_UNIVERSE ({len(self._entries)} active):"]
            for sym, e in sorted(self._entries.items()):
                lines.append(
                    f"  {sym:8s} via {e.source:15s} "
                    f"expires in {e.ttl_remaining_seconds()/3600:.1f}h"
                )
            return "\n".join(lines)


# Module-level singleton shared across all data sources
_hot_universe: Optional[HotUniverse] = None


def get_hot_universe(ttl_config: Optional[Dict] = None) -> HotUniverse:
    """Return the module-level singleton, creating it on first call."""
    global _hot_universe
    if _hot_universe is None:
        _hot_universe = HotUniverse(ttl_config)
    return _hot_universe
