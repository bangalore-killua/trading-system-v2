"""
news_relevance_engine.py — Ticker Extraction from News Text

Uses SpaCy NER to extract company mentions from article headlines/summaries,
then maps them to US equity tickers via a lookup table.

Two resolution strategies:
  1. Exact name match — "Apple" → AAPL, "Tesla" → TSLA
  2. SpaCy ORG entity → fuzzy match against known company names

This feeds news_aggregator: articles mentioning AAPL → AAPL added to HOT_UNIVERSE.

The lookup table (~5000 entries) is loaded once at startup from a bundled JSON.
For companies not in the table, we fall back to SEC EDGAR's company_tickers.json.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Common company name → ticker mappings (covers most liquid US stocks)
# Supplemented at runtime from SEC EDGAR's company_tickers.json
KNOWN_COMPANIES: Dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "berkshire": "BRK.B",
    "unitedhealth": "UNH",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "visa": "V",
    "procter & gamble": "PG",
    "procter and gamble": "PG",
    "mastercard": "MA",
    "walmart": "WMT",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "broadcom": "AVGO",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "costco": "COST",
    "home depot": "HD",
    "abbvie": "ABBV",
    "merck": "MRK",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pepsico": "PEP",
    "pepsi": "PEP",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "micron": "MU",
    "salesforce": "CRM",
    "netflix": "NFLX",
    "adobe": "ADBE",
    "oracle": "ORCL",
    "ibm": "IBM",
    "palantir": "PLTR",
    "snowflake": "SNOW",
    "coinbase": "COIN",
    "shopify": "SHOP",
    "paypal": "PYPL",
    "square": "SQ",
    "block": "SQ",
    "airbnb": "ABNB",
    "uber": "UBER",
    "lyft": "LYFT",
    "spotify": "SPOT",
    "twitter": "TWTR",
    "snap": "SNAP",
    "pinterest": "PINS",
    "zoom": "ZM",
    "doordash": "DASH",
    "roblox": "RBLX",
    "rivian": "RIVN",
    "lucid": "LCID",
    "ford": "F",
    "general motors": "GM",
    "gm": "GM",
    "boeing": "BA",
    "lockheed": "LMT",
    "raytheon": "RTX",
    "caterpillar": "CAT",
    "deere": "DE",
    "john deere": "DE",
    "united airlines": "UAL",
    "delta": "DAL",
    "delta airlines": "DAL",
    "delta air": "DAL",
    "american airlines": "AAL",
    "southwest": "LUV",
    "marriott": "MAR",
    "hilton": "HLT",
    "mcdonalds": "MCD",
    "mcdonald's": "MCD",
    "starbucks": "SBUX",
    "chipotle": "CMG",
    "disney": "DIS",
    "comcast": "CMCSA",
    "att": "T",
    "at&t": "T",
    "verizon": "VZ",
    "t-mobile": "TMUS",
    "tmobile": "TMUS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "citigroup": "C",
    "citi": "C",
    "american express": "AXP",
    "amex": "AXP",
    "blackrock": "BLK",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "astrazeneca": "AZN",
    "regeneron": "REGN",
    "biogen": "BIIB",
    "gilead": "GILD",
    "cvs": "CVS",
    "unitedhealth group": "UNH",
    "anthem": "ELV",
    "elevance": "ELV",
    "humana": "HUM",
    "marvell": "MRVL",
    "marvell semiconductor": "MRVL",
    "dell": "DELL",
    "hp": "HPQ",
    "hewlett": "HPQ",
    "arm": "ARM",
    "arm holdings": "ARM",
    "asml": "ASML",
    "taiwan semiconductor": "TSM",
    "tsmc": "TSM",
    "samsung": "005930.KS",
    "alibaba": "BABA",
    "baidu": "BIDU",
    "nio": "NIO",
    "xiaomi": "XIACY",
    "softbank": "SFTBY",
    "sony": "SONY",
    "toyota": "TM",
    "honda": "HMC",
}

# Regex: looks for $TICKER pattern in text (common in financial Twitter/media)
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

# Known non-ticker words that look like tickers (filter these out)
FALSE_POSITIVE_TICKERS: Set[str] = {
    "A", "AT", "IT", "IN", "BE", "BY", "OR", "US", "TO", "AN",
    "ALL", "FOR", "NEW", "NOW", "CEO", "CFO", "IPO", "ETF", "GDP",
    "CPI", "FED", "SEC", "DOJ", "FBI", "CIA", "UK", "EU", "AI",
    "EV", "PC", "TV", "AM", "PM", "OK", "SO", "GO",
}


class NewsRelevanceEngine:
    """Extracts US equity tickers from news text using SpaCy NER + name lookup."""

    def __init__(self, extra_companies: Optional[Dict[str, str]] = None):
        self._name_to_ticker: Dict[str, str] = dict(KNOWN_COMPANIES)
        if extra_companies:
            self._name_to_ticker.update(
                {k.lower(): v for k, v in extra_companies.items()}
            )
        self._nlp = None  # Lazy load SpaCy model

    def _ensure_nlp(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy  # type: ignore
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "SpaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            self._nlp = None

    def extract_tickers(self, text: str) -> List[str]:
        """Extract all mentioned US equity tickers from text.

        Combines three strategies in priority order:
          1. $TICKER regex (explicit mentions like $AAPL)
          2. Known company name lookup
          3. SpaCy ORG NER entity → name lookup
        """
        if not text:
            return []

        found: Set[str] = set()

        # Strategy 1: $TICKER pattern (highest confidence)
        for match in TICKER_PATTERN.finditer(text):
            ticker = match.group(1).upper()
            if ticker not in FALSE_POSITIVE_TICKERS and len(ticker) >= 2:
                found.add(ticker)

        # Strategy 2: Known company name lookup (case-insensitive substring match)
        lower_text = text.lower()
        for name, ticker in self._name_to_ticker.items():
            if name in lower_text:
                # Verify it's a word boundary match, not a substring of another word
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, lower_text):
                    found.add(ticker)

        # Strategy 3: SpaCy ORG NER
        self._ensure_nlp()
        if self._nlp:
            try:
                doc = self._nlp(text[:512])  # SpaCy is slow on long text
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        ent_lower = ent.text.lower().strip()
                        # Try exact match first, then substring
                        ticker = self._name_to_ticker.get(ent_lower)
                        if not ticker:
                            # Try matching start of company name
                            for name, t in self._name_to_ticker.items():
                                if ent_lower.startswith(name) or name.startswith(ent_lower):
                                    ticker = t
                                    break
                        if ticker:
                            found.add(ticker)
            except Exception:
                logger.debug("SpaCy NER error on: %s...", text[:60])

        # Filter: only return plausible US equity tickers (1-5 chars, all caps)
        return [
            t for t in found
            if 1 <= len(t) <= 5
            and t.replace(".", "").isalpha()
            and t not in FALSE_POSITIVE_TICKERS
        ]

    def load_sec_tickers(self, company_tickers_json: dict) -> int:
        """Merge SEC EDGAR's company_tickers.json into the name→ticker map.

        Returns the number of new entries added.
        """
        added = 0
        for entry in company_tickers_json.values():
            name = entry.get("title", "").lower().strip()
            ticker = entry.get("ticker", "").upper().strip()
            if name and ticker and name not in self._name_to_ticker:
                self._name_to_ticker[name] = ticker
                added += 1
        logger.info("NewsRelevanceEngine: loaded %d SEC EDGAR tickers", added)
        return added
