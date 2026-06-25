"""
prnewswire.py — PRNewswire RSS press release feed.

PRIMARY news source — direct from companies, 22 minutes earlier than aggregators.
Credibility: 1.0 (highest possible — source is the company itself).

No Playwright needed — plain RSS via feedparser.
Falls back to retry-3 if RSS fails; no Telegram alert for RSS failure
(transient network issue, not a structural data gap).

RSS URL: https://www.prnewswire.com/rss/news-releases-list.rss
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

PR_NEWSWIRE_RSS = "https://www.prnewswire.com/rss/news-releases-list.rss"
CREDIBILITY     = 1.0
SOURCE_NAME     = "prnewswire"


def get_prnewswire(limit: int = 50, retries: int = 3) -> list[dict]:
    """Fetch latest PRNewswire press releases.

    Returns
    -------
    List of dicts, each:
      {"title": str, "url": str, "published": str (ISO 8601),
       "summary": str, "source": "prnewswire", "credibility": 1.0}
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        raise ImportError("feedparser not installed — run: pip install feedparser")

    for attempt in range(retries):
        try:
            feed = feedparser.parse(PR_NEWSWIRE_RSS)
            if feed.bozo and not feed.entries:
                raise ValueError(f"feedparser error: {feed.bozo_exception}")
            return _parse_feed(feed, limit)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                logger.warning("PRNewswire RSS attempt %d/%d failed: %s", attempt + 1, retries, e)
            else:
                logger.error("PRNewswire RSS failed after %d attempts: %s", retries, e)
                return []
    return []


def get_prnewswire_for_symbol(symbol: str, company_name: Optional[str] = None, limit: int = 50) -> list[dict]:
    """Filter PRNewswire feed for articles relevant to a specific symbol/company.

    Parameters
    ----------
    symbol       : ticker, e.g. "NVDA"
    company_name : company name for text matching (e.g. "NVIDIA")
    limit        : max articles to fetch from feed before filtering
    """
    all_articles = get_prnewswire(limit=limit)
    search_terms = {symbol.lower()}
    if company_name:
        search_terms.update(w.lower() for w in company_name.split() if len(w) > 3)

    return [
        a for a in all_articles
        if any(term in a["title"].lower() or term in a["summary"].lower()
               for term in search_terms)
    ]


def _parse_feed(feed, limit: int) -> list[dict]:
    articles = []
    for entry in feed.entries[:limit]:
        try:
            published = _parse_date(entry)
            articles.append({
                "title":       entry.get("title", ""),
                "url":         entry.get("link", ""),
                "published":   published,
                "summary":     _clean_text(entry.get("summary", "")),
                "source":      SOURCE_NAME,
                "credibility": CREDIBILITY,
            })
        except Exception as e:
            logger.debug("PRNewswire: skipping entry — %s", e)
    return articles


def _parse_date(entry) -> str:
    """Return ISO 8601 UTC string from a feedparser entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        import calendar
        ts = calendar.timegm(entry.published_parsed)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _clean_text(text: str) -> str:
    """Strip HTML tags and excessive whitespace."""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = get_prnewswire(limit=5)
    for a in articles:
        print(f"[{a['published']}] {a['title'][:80]}")
