"""
earnings_whispers.py — Whisper EPS numbers via EarningsWhispers (Playwright + fallback).

Information asymmetry: market prices the whisper, not official consensus.
  Official EPS beat + whisper miss → stock SELLS OFF (common trap)
  Whisper beat > 5% → strong bullish catalyst

Fallback chain:
  1. earningswhispers.com (Playwright, primary)
  2. EarningsWhispers RSS feed (if RSS exists)
  3. yfinance analyst estimate (weakest — just the official consensus)

Use case: before earnings, compute whisper_spread = (whisper - consensus) / abs(consensus).
  whisper_spread > 0.10 → market expects a beat (bar is high)
  whisper_spread < -0.05 → market expects a miss (bar is low → good setup)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

EW_RSS_URL  = "https://www.earningswhispers.com/rss"
SOURCE_NAME = "earnings_whispers"


class EarningsWhispers:
    """Whisper numbers with Playwright primary and RSS/yfinance fallback."""

    name = "earnings_whispers"

    def get(self, symbol: str) -> tuple:
        from data.sources.base_source import _notify_degraded, _notify_critical

        # Try Playwright primary
        try:
            data = self._fetch_playwright(symbol)
            return data, self.name, False
        except Exception as e:
            logger.warning("earnings_whispers: Playwright failed for %s — %s", symbol, e)

        # Fallback 1: RSS
        try:
            data = self._fetch_rss(symbol)
            _notify_degraded(self.name, symbol, "earnings_whispers_rss")
            return data, "earnings_whispers_rss", True
        except Exception as e:
            logger.warning("earnings_whispers: RSS fallback failed for %s — %s", symbol, e)

        # Fallback 2: yfinance official estimate
        try:
            data = self._fetch_yfinance(symbol)
            _notify_degraded(self.name, symbol, "yfinance_estimate")
            return data, "yfinance_estimate", True
        except Exception as e:
            logger.warning("earnings_whispers: yfinance fallback failed for %s — %s", symbol, e)

        _notify_critical(self.name, symbol, "all sources failed")
        return _neutral_whisper(), None, True

    def _fetch_playwright(self, symbol: str) -> dict:
        from data.sources.browser_scraper import BrowserScraper
        from bs4 import BeautifulSoup

        url = f"https://www.earningswhispers.com/stocks/{symbol.lower()}"
        with BrowserScraper(stealth=True) as s:
            html = s.fetch(url, wait_for="#eps, .eps-item, .whisper", delay=(3.0, 6.0))

        return _parse_ew_html(html, symbol)

    def _fetch_rss(self, symbol: str) -> dict:
        try:
            import feedparser
        except ImportError:
            raise ImportError("feedparser not installed")

        feed = feedparser.parse(EW_RSS_URL)
        for entry in feed.entries:
            if symbol.lower() in entry.get("title", "").lower():
                return {
                    "symbol":          symbol,
                    "whisper_eps":     None,
                    "consensus_eps":   None,
                    "whisper_spread":  None,
                    "earnings_date":   entry.get("published", ""),
                    "source":          "earnings_whispers_rss",
                    "data_quality":    "low",
                }
        raise ValueError(f"No RSS entry found for {symbol}")

    def _fetch_yfinance(self, symbol: str) -> dict:
        import yfinance as yf
        info      = yf.Ticker(symbol).info
        consensus = info.get("epsCurrentYear") or info.get("epsForward")
        if consensus is None:
            raise ValueError(f"yfinance: no EPS estimate for {symbol}")
        return {
            "symbol":         symbol,
            "whisper_eps":    None,
            "consensus_eps":  round(float(consensus), 4),
            "whisper_spread": None,
            "earnings_date":  None,
            "source":         "yfinance_estimate",
            "data_quality":   "low",
        }


def _parse_ew_html(html: str, symbol: str) -> dict:
    """Parse EarningsWhispers HTML for whisper and consensus EPS."""
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(html, "html.parser")

    def _extract_number(selector):
        el = soup.select_one(selector)
        if not el:
            return None
        text = re.sub(r"[^\d.\-]", "", el.get_text())
        try:
            return float(text)
        except ValueError:
            return None

    whisper   = _extract_number("#eps, .whisper-eps, [id*='whisper']")
    consensus = _extract_number(".consensus-eps, [id*='consensus']")

    spread = None
    if whisper is not None and consensus is not None and consensus != 0:
        spread = round((whisper - consensus) / abs(consensus), 4)

    return {
        "symbol":         symbol,
        "whisper_eps":    whisper,
        "consensus_eps":  consensus,
        "whisper_spread": spread,
        "earnings_date":  None,
        "source":         SOURCE_NAME,
        "data_quality":   "high" if whisper is not None else "low",
    }


def compute_whisper_score(data: Optional[dict]) -> float:
    """Convert whisper data to a 0-1 signal score for earnings events.

    0.5 = in-line with consensus (neutral)
    >0.5 = whisper above consensus (bar is high, careful)
    <0.5 = whisper below consensus (bar is low, upside surprise likely)
    """
    if not data or data.get("whisper_spread") is None:
        return 0.5

    spread = data["whisper_spread"]
    # spread +0.20 → score 0.40 (high bar, risk of miss)
    # spread -0.10 → score 0.60 (low bar, upside surprise possible)
    score = 0.5 - spread * 0.5
    return round(max(0.0, min(1.0, score)), 3)


def _neutral_whisper() -> dict:
    return {
        "symbol":         None,
        "whisper_eps":    None,
        "consensus_eps":  None,
        "whisper_spread": None,
        "earnings_date":  None,
        "source":         "neutral_default",
        "data_quality":   "none",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    ew  = EarningsWhispers()
    d, source, is_fb = ew.get(sym)
    print(f"source={source}, fallback={is_fb}")
    print(d)
    print(f"whisper_score={compute_whisper_score(d)}")
