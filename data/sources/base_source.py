"""
base_source.py — SourceWithFallback base class for all data sources.

Every Playwright-based scraper inherits from SourceWithFallback.
Provides:
  - Automatic fallback chain: try primary → fallback[0] → fallback[1] → ...
  - Telegram notification on degradation (rate-limited, 1h cooldown per source)
  - Telegram critical alert when all sources fail
  - Standard return tuple: (data, source_name_used, is_fallback)

Usage pattern:

    class UnusualWhales(SourceWithFallback):
        name = "unusual_whales"
        fallbacks = [BarchartFallback(), YFinanceFallback()]

        def _fetch(self, symbol: str) -> dict:
            # Playwright scraping logic
            ...

    uw = UnusualWhales()
    data, source, is_fallback = uw.get("NVDA")
    options_score = compute_score(data) if data is not None else 0.5  # neutral default
"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias for the return tuple from get()
SourceResult = Tuple[Optional[Any], Optional[str], bool]


class SourceWithFallback:
    """Abstract base class for data sources with graceful degradation.

    Subclasses must set:
        name      : str  — human-readable source name (used in Telegram alerts)
        fallbacks : list — ordered list of fallback source objects,
                          each with a .fetch(symbol) method and .name attribute

    Subclasses must implement:
        _fetch(self, symbol: str) -> Any  — primary data fetch logic
    """

    name: str = "unnamed_source"
    fallbacks: list = []

    def get(self, symbol: str) -> SourceResult:
        """Fetch data with automatic fallback and Telegram notifications.

        Returns
        -------
        (data, source_name, is_fallback)
          data        : fetched data, or None if everything failed
          source_name : name of the source that provided data, or None
          is_fallback : True if a fallback was used (primary failed)
        """
        try:
            data = self._fetch(symbol)
            return data, self.name, False

        except Exception as primary_err:
            logger.warning("%s: primary fetch failed for %s — %s", self.name, symbol, primary_err)

            for fb in self.fallbacks:
                try:
                    data = fb.fetch(symbol)
                    _notify_degraded(self.name, symbol, fb.name)
                    return data, fb.name, True
                except Exception as fb_err:
                    logger.warning("%s: fallback %s also failed for %s — %s",
                                   self.name, fb.name, symbol, fb_err)
                    continue

            # All sources failed
            _notify_critical(self.name, symbol, str(primary_err))
            return None, None, True

    def _fetch(self, symbol: str) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} must implement _fetch()")


class SimpleFallback:
    """Adapter for plain functions used as fallbacks.

    Usage:
        SimpleFallback(name="yfinance_options", fn=lambda s: yf.Ticker(s).option_chain())
    """

    def __init__(self, name: str, fn):
        self.name = name
        self._fn  = fn

    def fetch(self, symbol: str):
        return self._fn(symbol)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _notify_degraded(source_name: str, symbol: str, fallback_name: str) -> None:
    try:
        from data.notifications.telegram import notify_source_degraded  # type: ignore
        notify_source_degraded(source_name, symbol, fallback_name)
    except Exception as e:
        logger.debug("Telegram notify_degraded failed: %s", e)


def _notify_critical(source_name: str, symbol: str, error: str) -> None:
    try:
        from data.notifications.telegram import notify_source_down  # type: ignore
        notify_source_down(source_name, symbol, error)
    except Exception as e:
        logger.debug("Telegram notify_critical failed: %s", e)
