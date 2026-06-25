"""
seeking_alpha.py — Analysis headlines via Playwright + Google News RSS fallback.

Credibility: 0.70 (aggregated analysis, not primary source)
Use: headline sentiment velocity + analyst opinion divergence.

Fallback chain:
  1. seekingalpha.com (Playwright, primary)
  2. Google News RSS for "{symbol} stock" (no API key required)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

CREDIBILITY = 0.70
SOURCE_NAME = "seeking_alpha"


class SeekingAlpha:
    """SA headlines with Playwright primary and Google News fallback."""

    name = "seeking_alpha"

    def get(self, symbol: str, limit: int = 10) -> tuple:
        from data.sources.base_source import _notify_degraded, _notify_critical

        try:
            data = self._fetch_playwright(symbol, limit)
            return data, self.name, False
        except Exception as e:
            logger.warning("seeking_alpha: Playwright failed for %s — %s", symbol, e)

        try:
            data = _fetch_google_news(symbol, limit)
            _notify_degraded(self.name, symbol, "google_news_rss")
            return data, "google_news_rss", True
        except Exception as e:
            logger.warning("seeking_alpha: Google News fallback also failed for %s — %s", symbol, e)

        _notify_critical(self.name, symbol, "SA and Google News both failed")
        return [], None, True

    def _fetch_playwright(self, symbol: str, limit: int) -> list[dict]:
        from data.sources.browser_scraper import BrowserScraper
        from bs4 import BeautifulSoup

        url = f"https://seekingalpha.com/symbol/{symbol}/news"
        with BrowserScraper(stealth=True) as s:
            html = s.fetch(
                url,
                wait_for="[data-test-id='post-list'] article, .sa-article",
                delay=(4.0, 8.0),
                timeout=20_000,
            )

        return _parse_sa_html(html, limit)


def _parse_sa_html(html: str, limit: int) -> list[dict]:
    from bs4 import BeautifulSoup

    soup     = BeautifulSoup(html, "html.parser")
    articles = []

    # SA uses data-test-id attributes — try multiple selectors
    candidates = (
        soup.select("[data-test-id='post-list'] article")
        or soup.select(".sa-article")
        or soup.select("article.article")
    )

    for el in candidates[:limit]:
        try:
            title_el = el.select_one("h3, h4, [data-test-id='post-title']")
            link_el  = el.select_one("a[href*='/article/'], a[href*='/news/']")
            time_el  = el.select_one("time, [data-test-id='post-date']")

            title = title_el.get_text(strip=True) if title_el else ""
            url   = link_el["href"] if link_el else ""
            if url.startswith("/"):
                url = "https://seekingalpha.com" + url

            published = datetime.now(timezone.utc).isoformat()
            if time_el:
                ts = time_el.get("datetime") or time_el.get("title")
                if ts:
                    published = ts

            if title:
                articles.append({
                    "title":       title,
                    "url":         url,
                    "published":   published,
                    "summary":     "",
                    "source":      SOURCE_NAME,
                    "credibility": CREDIBILITY,
                })
        except Exception:
            continue

    return articles


def _fetch_google_news(symbol: str, limit: int) -> list[dict]:
    """RSS fallback via Google News search."""
    import feedparser

    query    = f"{symbol} stock"
    rss_url  = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed     = feedparser.parse(rss_url)

    if not feed.entries:
        raise ValueError(f"Google News RSS: no results for {query}")

    articles = []
    for entry in feed.entries[:limit]:
        try:
            import calendar
            published = datetime.now(timezone.utc).isoformat()
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                ts = calendar.timegm(entry.published_parsed)
                published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            articles.append({
                "title":       entry.get("title", ""),
                "url":         entry.get("link", ""),
                "published":   published,
                "summary":     entry.get("summary", "")[:300],
                "source":      "google_news_rss",
                "credibility": 0.60,
            })
        except Exception:
            continue

    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    sa  = SeekingAlpha()
    articles, source, is_fb = sa.get(sym, limit=5)
    print(f"source={source}, fallback={is_fb}, count={len(articles)}")
    for a in articles:
        print(f"  [{a['published'][:10]}] {a['title'][:70]}")
