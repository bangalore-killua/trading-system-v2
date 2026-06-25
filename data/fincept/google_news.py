"""
google_news.py — Google News RSS feed for symbol-specific news.

Free, no API key. Credibility: 0.60 (aggregated).
Used as:
  1. Fallback for SeekingAlpha scraper
  2. Additional news source in /api/news endpoint
  3. Headline velocity signal (rate of change more informative than volume)

Deduplication: articles from PRNewswire/BusinessWire take priority.
If same story appears in Google News AND PRNewswire → keep PRNewswire only.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

CREDIBILITY  = 0.60
SOURCE_NAME  = "google_news"
BASE_RSS     = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def get_google_news(symbol: str, company_name: Optional[str] = None, limit: int = 20) -> list[dict]:
    """Fetch Google News RSS articles for a symbol.

    Parameters
    ----------
    symbol       : ticker, e.g. "NVDA"
    company_name : optional company name for better matching, e.g. "NVIDIA"
    limit        : max articles to return

    Returns
    -------
    List of dicts:
      {"title": str, "url": str, "published": str (ISO 8601),
       "summary": str, "source": "google_news", "credibility": 0.60}
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        raise ImportError("feedparser not installed — run: pip install feedparser")

    # Build query: ticker + optional company name for better coverage
    if company_name:
        query = f"{symbol} OR \"{company_name}\" stock"
    else:
        query = f"{symbol} stock"

    url  = BASE_RSS.format(query=query.replace(" ", "+"))
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:limit]:
        try:
            published = _parse_date(entry)
            articles.append({
                "title":       entry.get("title", ""),
                "url":         entry.get("link", ""),
                "published":   published,
                "summary":     _clean_summary(entry.get("summary", "")),
                "source":      SOURCE_NAME,
                "credibility": CREDIBILITY,
            })
        except Exception:
            continue

    return articles


def get_market_news(topics: list[str] = None, limit: int = 30) -> list[dict]:
    """Fetch general market/macro news from Google News.

    Parameters
    ----------
    topics : list of search terms, e.g. ["Federal Reserve", "inflation", "S&P 500"]
    """
    try:
        import feedparser
    except ImportError:
        return []

    if not topics:
        topics = ["stock market", "Federal Reserve", "earnings"]

    all_articles = []
    for topic in topics[:3]:
        url  = BASE_RSS.format(query=topic.replace(" ", "+"))
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit // len(topics)]:
            try:
                published = _parse_date(entry)
                all_articles.append({
                    "title":       entry.get("title", ""),
                    "url":         entry.get("link", ""),
                    "published":   published,
                    "summary":     _clean_summary(entry.get("summary", "")),
                    "source":      SOURCE_NAME,
                    "credibility": CREDIBILITY,
                    "topic":       topic,
                })
            except Exception:
                continue

    return all_articles


def compute_headline_velocity(articles: list[dict], window_hours: int = 6) -> float:
    """Compute headline velocity: rate of change of article count per hour.

    More informative than total count. A spike in velocity = news event.
    Returns: articles_per_hour over the window (normalized 0-1 for scoring).
    """
    if not articles:
        return 0.0

    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    recent = 0
    for a in articles:
        try:
            pub = datetime.fromisoformat(a["published"].replace("Z", "+00:00"))
            if pub >= cutoff:
                recent += 1
        except Exception:
            continue

    velocity = recent / window_hours
    # Normalize: 0 articles/hr → 0, 3+ articles/hr → 1.0
    return round(min(1.0, velocity / 3.0), 3)


def _parse_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        import calendar
        ts = calendar.timegm(entry.published_parsed)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _clean_summary(text: str) -> str:
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    articles = get_google_news(sym, limit=5)
    velocity = compute_headline_velocity(articles)
    print(f"Velocity: {velocity:.2f} articles/hr")
    for a in articles:
        print(f"  [{a['published'][:10]}] {a['title'][:70]}")
