"""
news_stream.py — Alpaca NewsDataStream WebSocket consumer

Subscribes to Alpaca's real-time news feed for ALL symbols using a single
persistent WebSocket connection. This is the fastest (sub-second) and cheapest
(zero per-event API calls) way to detect market-moving news.

Flow:
  1. Message received → symbol extracted → NLP pipeline called
  2. If FinBERT score > threshold AND threat NOT CRITICAL/HIGH:
       → symbol added to HOT_UNIVERSE with TTL=4h
  3. HOT_UNIVERSE fires callback → position_manager evaluates entry immediately

This runs in a background asyncio task started by the scheduler.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AlpacaNewsStream:
    """Wraps alpaca-py's NewsDataStream for the 'news for all symbols' use case.

    The stream pushes news events as they happen — one WebSocket connection
    covers every US symbol (no per-symbol subscription limit on the free tier).
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        hot_universe,          # HotUniverse instance
        nlp_pipeline=None,     # optional NLP callable for sentiment + threat
        sentiment_threshold: float = 0.70,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._hot_universe = hot_universe
        self._nlp = nlp_pipeline
        self._sentiment_threshold = sentiment_threshold
        self._stream = None
        self._running = False

    async def start(self) -> None:
        """Start the WebSocket stream. Runs until stop() is called."""
        # Import here so the module loads even without alpaca-py installed
        from alpaca.data.live import NewsDataStream  # type: ignore

        self._stream = NewsDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        # Subscribe to ALL symbols with the wildcard
        self._stream.subscribe_news(self._on_news, "*")

        logger.info("Alpaca NewsDataStream started — subscribed to ALL symbols")
        self._running = True

        try:
            await self._stream._run_forever()
        except Exception as exc:
            logger.exception("Alpaca NewsDataStream error — will retry via scheduler")
            self._running = False
            from server.telegram_alerts import notify_failure
            notify_failure(
                source="AlpacaNewsStream.start",
                error_type=type(exc).__name__,
                cause=str(exc),
                affects="Real-time Alpaca news feed — RSS aggregator still active, no real-time news until restart",
                degraded=False,
            )

    async def stop(self) -> None:
        if self._stream:
            await self._stream.close()
        self._running = False
        logger.info("Alpaca NewsDataStream stopped")

    async def _on_news(self, news) -> None:
        """Called by alpaca-py for each incoming news item."""
        # news.symbols is a list; some articles cover multiple symbols
        symbols = getattr(news, "symbols", []) or []
        headline = getattr(news, "headline", "") or ""
        summary = getattr(news, "summary", "") or ""

        if not symbols:
            return

        # NLP: get sentiment + threat level for this headline+summary
        text = f"{headline}. {summary}".strip()
        sentiment_score, threat_level = await self._run_nlp(text)

        # Block if threat is CRITICAL or HIGH — these indicate panic/crisis events
        if threat_level in ("CRITICAL", "HIGH"):
            logger.debug(
                "News blocked (threat=%s): %s → %s",
                threat_level, symbols, headline[:80],
            )
            return

        # Only forward bullish articles above the threshold
        if sentiment_score < self._sentiment_threshold:
            return

        for symbol in symbols:
            # Skip non-equity symbols (options chains, crypto, etc.)
            if "/" in symbol or len(symbol) > 5:
                continue

            self._hot_universe.add(
                symbol=symbol,
                source="NEWS_EVENT",
                headline=headline,
                sentiment_score=sentiment_score,
                metadata={
                    "threat_level": threat_level,
                    "url": getattr(news, "url", ""),
                    "author": getattr(news, "author", ""),
                    "published_at": str(getattr(news, "created_at", "")),
                },
            )

    async def _run_nlp(self, text: str):
        """Run NLP pipeline if available; return (sentiment, threat) defaults otherwise."""
        if self._nlp is None:
            return 0.0, "UNKNOWN"

        try:
            if asyncio.iscoroutinefunction(self._nlp):
                return await self._nlp(text)
            return self._nlp(text)
        except Exception:
            logger.exception("NLP error in news stream")
            return 0.0, "UNKNOWN"

    @property
    def is_running(self) -> bool:
        return self._running
