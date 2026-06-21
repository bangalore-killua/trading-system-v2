"""
edgar_monitor.py — SEC EDGAR Real-Time RSS Monitor

Polls two EDGAR RSS feeds every 60 seconds (free, no API key required):
  1. 8-K filings  → detects material corporate events
  2. Form 4 filings → detects insider buying/selling

On material event detection, the affected symbol is added to HOT_UNIVERSE
for immediate signal basket evaluation.

8-K item codes of interest:
  Bullish: 1.01 (material agreement), 7.01 (Reg FD), 8.01 (other events)
  Bearish: 1.03 (bankruptcy), 2.04 (debt default), 5.02 (executive departure)
           Note: bearish items trigger HOT_UNIVERSE so we can evaluate EXIT

SEC EDGAR RSS endpoints (no authentication, completely free):
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&...
  https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&...
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Set

import feedparser  # type: ignore
import httpx

logger = logging.getLogger(__name__)

# EDGAR CIK → ticker mapping lives in SEC's company_tickers.json (free endpoint)
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# 8-K items that indicate material bullish events
BULLISH_8K_ITEMS = {"1.01", "7.01", "8.01"}

# 8-K items that indicate material bearish events (still add to HOT_UNIVERSE for exit eval)
BEARISH_8K_ITEMS = {"1.03", "2.04", "5.02", "8.06"}


class EdgarMonitor:
    """Polls EDGAR RSS feeds for 8-K and Form 4 filings.

    CIK→ticker mapping is loaded once at startup and cached in memory.
    The mapping is ~300KB JSON and refreshes weekly.
    """

    EDGAR_8K_RSS = (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        "?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom"
    )
    EDGAR_FORM4_RSS = (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        "?action=getcurrent&type=4&dateb=&owner=include&count=40&search_text=&output=atom"
    )

    def __init__(
        self,
        hot_universe,
        poll_interval_seconds: int = 60,
        bullish_items: Optional[Set[str]] = None,
        bearish_items: Optional[Set[str]] = None,
    ):
        self._hot_universe = hot_universe
        self._interval = poll_interval_seconds
        self._bullish_items = bullish_items or BULLISH_8K_ITEMS
        self._bearish_items = bearish_items or BEARISH_8K_ITEMS

        # CIK (as zero-padded str) → ticker symbol
        self._cik_to_ticker: dict[str, str] = {}

        # Dedup: track entry IDs we've already processed so reruns don't re-trigger
        self._seen_8k: Set[str] = set()
        self._seen_form4: Set[str] = set()
        self._running = False

    async def start(self) -> None:
        """Load CIK mapping then poll in a loop."""
        await self._load_cik_mapping()
        self._running = True
        logger.info("EDGAR monitor started (poll every %ds)", self._interval)

        while self._running:
            try:
                await self._poll_8k()
                await self._poll_form4()
            except Exception:
                logger.exception("EDGAR monitor poll error")

            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False

    async def _load_cik_mapping(self) -> None:
        """Download SEC's official CIK→ticker JSON (~300KB, weekly refresh)."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(COMPANY_TICKERS_URL)
                data = resp.json()

            # Data format: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
            self._cik_to_ticker = {
                str(v["cik_str"]).zfill(10): v["ticker"].upper()
                for v in data.values()
                if "cik_str" in v and "ticker" in v
            }
            logger.info(
                "EDGAR CIK mapping loaded: %d companies", len(self._cik_to_ticker)
            )
        except Exception:
            logger.exception("Failed to load EDGAR CIK mapping — tickers unavailable")

    async def _poll_8k(self) -> None:
        entries = await self._fetch_feed(self.EDGAR_8K_RSS)
        for entry in entries:
            entry_id = entry.get("id", "")
            if entry_id in self._seen_8k:
                continue
            self._seen_8k.add(entry_id)

            # Keep seen set bounded to last 2000 entries
            if len(self._seen_8k) > 2000:
                self._seen_8k = set(list(self._seen_8k)[-1000:])

            title = entry.get("title", "")
            cik = self._extract_cik(entry)
            ticker = self._cik_to_ticker.get(cik, "") if cik else ""

            if not ticker:
                continue

            item_codes = self._extract_8k_items(title)
            is_bullish = bool(item_codes & self._bullish_items)
            is_bearish = bool(item_codes & self._bearish_items)

            if is_bullish or is_bearish:
                polarity = "bullish" if is_bullish else "bearish"
                logger.info(
                    "EDGAR 8-K %s: %s — items %s",
                    polarity.upper(), ticker, item_codes,
                )
                self._hot_universe.add(
                    symbol=ticker,
                    source="EDGAR_8K",
                    headline=title,
                    metadata={
                        "items": list(item_codes),
                        "is_bullish": is_bullish,
                        "is_bearish": is_bearish,
                        "cik": cik,
                        "filing_url": entry.get("link", ""),
                    },
                )

    async def _poll_form4(self) -> None:
        entries = await self._fetch_feed(self.EDGAR_FORM4_RSS)
        for entry in entries:
            entry_id = entry.get("id", "")
            if entry_id in self._seen_form4:
                continue
            self._seen_form4.add(entry_id)

            if len(self._seen_form4) > 2000:
                self._seen_form4 = set(list(self._seen_form4)[-1000:])

            title = entry.get("title", "")
            cik = self._extract_cik(entry)
            ticker = self._cik_to_ticker.get(cik, "") if cik else ""

            if not ticker:
                continue

            # Only process insider BUYS — sells are noise (diversification, taxes, etc.)
            if not self._is_insider_buy(title):
                continue

            logger.info("EDGAR Form 4 insider BUY: %s — %s", ticker, title[:80])
            self._hot_universe.add(
                symbol=ticker,
                source="EDGAR_FORM4",
                headline=title,
                metadata={
                    "is_insider_buy": True,
                    "cik": cik,
                    "filing_url": entry.get("link", ""),
                },
            )

    @staticmethod
    async def _fetch_feed(url: str) -> list:
        """Fetch and parse an RSS/Atom feed, return entries list."""
        try:
            # Use httpx for async HTTP; feedparser handles the parsing
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers={"User-Agent": "trading-system-v2/1.0"})
            parsed = feedparser.parse(resp.text)
            return parsed.get("entries", [])
        except Exception:
            logger.exception("EDGAR feed fetch error: %s", url)
            return []

    @staticmethod
    def _extract_cik(entry: dict) -> Optional[str]:
        """Extract zero-padded 10-digit CIK from EDGAR entry links/ids."""
        # EDGAR IDs look like: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&...
        for field in ("id", "link"):
            val = entry.get(field, "")
            match = re.search(r"CIK=(\d+)", val, re.IGNORECASE)
            if match:
                return match.group(1).zfill(10)
        return None

    @staticmethod
    def _extract_8k_items(title: str) -> Set[str]:
        """Parse item codes like '1.01', '5.02' from 8-K filing title."""
        # Typical title: "8-K for ACME Corp: Items 1.01, 7.01"
        return set(re.findall(r"\b(\d+\.\d+)\b", title))

    @staticmethod
    def _is_insider_buy(title: str) -> bool:
        """Heuristic: Form 4 title often says '(Purchase)' or 'Acquisition'."""
        lower = title.lower()
        buy_keywords = {"purchase", "acquisition", "acquired", "bought", "buy"}
        sell_keywords = {"sale", "sold", "dispose", "disposed"}
        has_buy = any(k in lower for k in buy_keywords)
        has_sell = any(k in lower for k in sell_keywords)
        # Only flag as buy if we see buy keywords and no sell keywords
        return has_buy and not has_sell
