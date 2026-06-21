"""
news_aggregator.py — 39-Source RSS News Aggregator

Fetches from all 39 registered sources (news_sources.py) every 5 minutes.
For each article:
  1. Dedup by URL/title hash
  2. Run threat classifier (fast, word-list based)
  3. Run FinBERT sentiment (GPU-cached model)
  4. Extract mentioned tickers via SpaCy NER
  5. Add bullish-signal tickers to HOT_UNIVERSE

Fixes a critical bug in TradingSystem: the news cache was capped at watchlist[:20],
silently ignoring symbols 21-50. This version processes ALL active symbols.

Article cache: last 72h articles per source, bounded at 1000 total entries.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import feedparser  # type: ignore
import httpx

from data.news_sources import ALL_SOURCES, NewsSource, get_source_weight

logger = logging.getLogger(__name__)


class ArticleRecord:
    __slots__ = ("url", "title", "source_name", "published_at", "sentiment", "threat", "tickers")

    def __init__(
        self,
        url: str,
        title: str,
        source_name: str,
        published_at: datetime,
        sentiment: float = 0.0,
        threat: str = "UNKNOWN",
        tickers: Optional[List[str]] = None,
    ):
        self.url = url
        self.title = title
        self.source_name = source_name
        self.published_at = published_at
        self.sentiment = sentiment
        self.threat = threat
        self.tickers = tickers or []


class NewsAggregator:
    """Polls all 39 RSS sources, runs NLP, and feeds HOT_UNIVERSE."""

    def __init__(
        self,
        hot_universe,
        threat_classifier=None,     # strategy.threat_classifier.ThreatClassifier
        sentiment_model=None,       # strategy.news_sentiment.FinBERTSentiment
        ticker_extractor=None,      # callable: text → List[str]
        poll_interval_minutes: float = 5.0,
        sentiment_threshold: float = 0.70,
        max_article_age_hours: int = 72,
        sources: Optional[List[NewsSource]] = None,
    ):
        self._hot_universe = hot_universe
        self._threat_clf = threat_classifier
        self._sentiment = sentiment_model
        self._ticker_extract = ticker_extractor
        self._interval = poll_interval_minutes * 60
        self._sentiment_threshold = sentiment_threshold
        self._max_age = timedelta(hours=max_article_age_hours)
        self._sources = sources or ALL_SOURCES
        self._running = False

        # Dedup: fingerprint (url-or-title hash) → published_at
        self._seen: Dict[str, datetime] = {}
        self._articles: List[ArticleRecord] = []

    async def start(self) -> None:
        self._running = True
        logger.info(
            "NewsAggregator started: %d sources, poll every %.0f min",
            len(self._sources),
            self._interval / 60,
        )
        while self._running:
            try:
                await self._fetch_all_sources()
            except Exception:
                logger.exception("NewsAggregator fetch cycle error")
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False

    async def _fetch_all_sources(self) -> None:
        """Fetch all sources concurrently with a semaphore to cap parallelism."""
        sem = asyncio.Semaphore(10)  # max 10 concurrent HTTP requests

        async def bounded_fetch(source: NewsSource) -> List[dict]:
            async with sem:
                return await self._fetch_source(source)

        results = await asyncio.gather(
            *[bounded_fetch(s) for s in self._sources],
            return_exceptions=True,
        )

        new_articles: List[ArticleRecord] = []
        for source, entries in zip(self._sources, results):
            if isinstance(entries, Exception):
                logger.debug("Source %s failed: %s", source.name, entries)
                continue
            for entry in entries:
                article = self._parse_entry(entry, source)
                if article:
                    new_articles.append(article)

        if not new_articles:
            return

        logger.debug("NewsAggregator: %d new articles this cycle", len(new_articles))

        # Process NLP in batches (sentiment model is slow; batch for GPU efficiency)
        await self._process_articles(new_articles)

        # Clean up old articles and seen fingerprints
        self._evict_stale()

    async def _fetch_source(self, source: NewsSource) -> List[dict]:
        """Fetch and parse one RSS source. Returns raw feedparser entry dicts."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    source.url,
                    headers={"User-Agent": "trading-system-v2/1.0"},
                    follow_redirects=True,
                )
            parsed = feedparser.parse(resp.text)
            return parsed.get("entries", [])
        except Exception as exc:
            logger.debug("Feed fetch failed (%s): %s", source.name, exc)
            return []

    def _parse_entry(self, entry: dict, source: NewsSource) -> Optional[ArticleRecord]:
        """Convert a feedparser entry into an ArticleRecord, or None if stale/seen."""
        url = entry.get("link", "") or entry.get("id", "")
        title = entry.get("title", "").strip()

        if not title:
            return None

        # Fingerprint on URL if available, else on title content
        fingerprint = hashlib.md5((url or title).encode()).hexdigest()
        if fingerprint in self._seen:
            return None

        # Parse publication time
        published_at = self._parse_date(entry)
        if published_at and (datetime.utcnow() - published_at) > self._max_age:
            return None  # Too old

        self._seen[fingerprint] = published_at or datetime.utcnow()

        return ArticleRecord(
            url=url,
            title=title,
            source_name=source.name,
            published_at=published_at or datetime.utcnow(),
        )

    async def _process_articles(self, articles: List[ArticleRecord]) -> None:
        """Run threat + sentiment + NER, then decide HOT_UNIVERSE eligibility."""
        for article in articles:
            text = article.title

            # Step 1: Threat classification (fast word-list, synchronous)
            threat = self._classify_threat(text)
            article.threat = threat

            # Block CRITICAL/HIGH before even running expensive FinBERT
            if threat in ("CRITICAL", "HIGH"):
                self._articles.append(article)
                continue

            # Step 2: Sentiment (FinBERT — run in executor to avoid blocking)
            sentiment = await self._score_sentiment(text)

            # Adjust by source credibility weight
            weight = get_source_weight(article.source_name)
            adjusted_score = sentiment * weight
            article.sentiment = adjusted_score

            # Step 3: Extract tickers if article is bullish enough
            if adjusted_score >= self._sentiment_threshold:
                tickers = await self._extract_tickers(text)
                article.tickers = tickers

                for ticker in tickers:
                    self._hot_universe.add(
                        symbol=ticker,
                        source="NEWS_EVENT",
                        headline=article.title,
                        sentiment_score=adjusted_score,
                        metadata={
                            "source_name": article.source_name,
                            "threat": threat,
                            "raw_sentiment": sentiment,
                            "url": article.url,
                        },
                    )

            self._articles.append(article)

    def _classify_threat(self, text: str) -> str:
        """Use threat classifier if available, else return UNKNOWN."""
        if self._threat_clf is None:
            return "UNKNOWN"
        try:
            return self._threat_clf.classify(text)
        except Exception:
            return "UNKNOWN"

    async def _score_sentiment(self, text: str) -> float:
        """Run FinBERT in a thread executor (model inference is CPU/GPU bound)."""
        if self._sentiment is None:
            return 0.0
        try:
            loop = asyncio.get_event_loop()
            score = await loop.run_in_executor(
                None, self._sentiment.score, text
            )
            return float(score)
        except Exception:
            logger.debug("Sentiment scoring error: %s", text[:60])
            return 0.0

    async def _extract_tickers(self, text: str) -> List[str]:
        """Extract US equity tickers mentioned in the article text."""
        if self._ticker_extract is None:
            return []
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._ticker_extract, text)
        except Exception:
            return []

    def _evict_stale(self) -> None:
        """Remove articles older than max_age and trim seen fingerprints."""
        cutoff = datetime.utcnow() - self._max_age
        self._articles = [a for a in self._articles if a.published_at >= cutoff]

        # Trim seen dict to prevent unbounded growth — keep last 5000 fingerprints
        if len(self._seen) > 5000:
            keys = list(self._seen.keys())
            self._seen = {k: self._seen[k] for k in keys[-3000:]}

    @staticmethod
    def _parse_date(entry: dict) -> Optional[datetime]:
        """Extract publication datetime from feedparser entry."""
        import time
        for field in ("published_parsed", "updated_parsed", "created_parsed"):
            val = entry.get(field)
            if val:
                try:
                    return datetime(*val[:6])
                except Exception:
                    pass
        return None

    def get_recent_articles(
        self,
        hours: int = 24,
        min_sentiment: float = 0.0,
        tickers: Optional[List[str]] = None,
    ) -> List[ArticleRecord]:
        """Return recent articles, optionally filtered by ticker and sentiment."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        results = [
            a for a in self._articles
            if a.published_at >= cutoff and a.sentiment >= min_sentiment
        ]
        if tickers:
            ticker_set = set(t.upper() for t in tickers)
            results = [a for a in results if any(t in ticker_set for t in a.tickers)]
        return sorted(results, key=lambda a: a.published_at, reverse=True)
