"""
news_sources.py — 39-Source RSS Feed Registry

Three-tier classification:
  Tier 1 — Wire services + regulators: highest credibility, fastest, broadest reach
  Tier 2 — Financial media + international: general financial coverage, some delay
  Tier 3 — Blogs + supplementary: informational, use lower weight in sentiment scoring

Each source has a `weight` (1.0 = full, 0.7 = tier2, 0.5 = tier3) used to adjust
FinBERT scores before they flow into the HOT_UNIVERSE threshold check.

Adapted from blm_terminal_windy/backend/app/services/news_sources.py
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str
    tier: int          # 1, 2, or 3
    weight: float      # credibility multiplier for sentiment scoring
    is_us_market: bool # True if primarily US equity focused


# ── Tier 1: Wire services, regulators, central banks ─────────────────────────
TIER_1: List[NewsSource] = [
    NewsSource("Reuters Business",         "https://feeds.reuters.com/reuters/businessNews", 1, 1.0, True),
    NewsSource("Reuters Finance",          "https://feeds.reuters.com/reuters/financialNewsUS", 1, 1.0, True),
    NewsSource("AP Business",              "https://feeds.apnews.com/rss/business", 1, 1.0, True),
    NewsSource("Bloomberg Markets",        "https://feeds.bloomberg.com/markets/news.rss", 1, 1.0, True),
    NewsSource("CNBC Top News",            "https://www.cnbc.com/id/100003114/device/rss/rss.html", 1, 1.0, True),
    NewsSource("Wall Street Journal",      "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", 1, 1.0, True),
    NewsSource("Financial Times",          "https://www.ft.com/rss/home/us", 1, 1.0, False),
    NewsSource("SEC Press Releases",       "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=10&search_text=&output=atom", 1, 1.0, True),
    NewsSource("Federal Reserve",          "https://www.federalreserve.gov/feeds/press_all.xml", 1, 1.0, True),
]

# ── Tier 2: Financial media + international ───────────────────────────────────
TIER_2: List[NewsSource] = [
    NewsSource("MarketWatch",              "https://feeds.marketwatch.com/marketwatch/topstories/", 2, 0.8, True),
    NewsSource("Seeking Alpha",            "https://seekingalpha.com/feed.xml", 2, 0.7, True),
    NewsSource("Barron's",                 "https://www.barrons.com/xml/rss/3_7014.xml", 2, 0.8, True),
    NewsSource("Investor's Business Daily","https://www.investors.com/ibd-videos/rss-feed/", 2, 0.7, True),
    NewsSource("Yahoo Finance",            "https://finance.yahoo.com/news/rssindex", 2, 0.7, True),
    NewsSource("The Economist",            "https://www.economist.com/finance-and-economics/rss.xml", 2, 0.8, False),
    NewsSource("Guardian Business",        "https://www.theguardian.com/uk/business/rss", 2, 0.7, False),
    NewsSource("BBC Business",             "https://feeds.bbci.co.uk/news/business/rss.xml", 2, 0.7, False),
    NewsSource("South China Morning Post", "https://www.scmp.com/rss/5/feed", 2, 0.6, False),
    NewsSource("Nikkei Asia",              "https://asia.nikkei.com/rss/feed/nar", 2, 0.7, False),
    NewsSource("Economic Times India",     "https://economictimes.indiatimes.com/rssfeedstopstories.cms", 2, 0.6, False),
    NewsSource("Moneycontrol",             "https://www.moneycontrol.com/rss/MCtopnews.xml", 2, 0.6, False),
    NewsSource("LiveMint",                 "https://www.livemint.com/rss/markets", 2, 0.6, False),
    NewsSource("Business Standard India",  "https://www.business-standard.com/rss/home_page_top_stories.rss", 2, 0.6, False),
    NewsSource("NDTV Business",            "https://feeds.feedburner.com/ndtvprofit-latest", 2, 0.5, False),
    NewsSource("Forbes",                   "https://www.forbes.com/money/feed/", 2, 0.7, True),
    NewsSource("Fortune",                  "https://fortune.com/feed/", 2, 0.7, True),
    NewsSource("Business Insider",         "https://feeds.businessinsider.com/custom/all", 2, 0.6, True),
    NewsSource("CNBC World",               "https://www.cnbc.com/id/100727362/device/rss/rss.html", 2, 0.8, False),
    NewsSource("Bloomberg Technology",     "https://feeds.bloomberg.com/technology/news.rss", 2, 0.8, False),
]

# ── Tier 3: Supplementary / blogs ────────────────────────────────────────────
TIER_3: List[NewsSource] = [
    NewsSource("Zero Hedge",               "https://feeds.feedburner.com/zerohedge/feed", 3, 0.4, True),
    NewsSource("The Street",               "https://www.thestreet.com/.rss/full/", 3, 0.5, True),
    NewsSource("Motley Fool",              "https://www.fool.com/feeds/news/", 3, 0.5, True),
    NewsSource("Kiplinger",                "https://www.kiplinger.com/feeds/everything.rss", 3, 0.5, True),
    NewsSource("ETF Trends",               "https://www.etftrends.com/feed/", 3, 0.5, True),
    NewsSource("ETF.com",                  "https://www.etf.com/sections/features-and-news?format=rss", 3, 0.5, True),
    NewsSource("Alpha Architect",          "https://alphaarchitect.com/feed/", 3, 0.4, True),
    NewsSource("AQR Insights",             "https://www.aqr.com/Insights/feed", 3, 0.5, True),
    NewsSource("Macro Musings",            "https://macromusings.libsyn.com/rss", 3, 0.4, False),
    NewsSource("Calculated Risk",          "https://www.calculatedriskblog.com/feeds/posts/default", 3, 0.4, True),
]

# ── Combined registry ─────────────────────────────────────────────────────────
ALL_SOURCES: List[NewsSource] = TIER_1 + TIER_2 + TIER_3

# US-market-focused sources only (used for HOT_UNIVERSE symbol detection)
US_SOURCES: List[NewsSource] = [s for s in ALL_SOURCES if s.is_us_market]


def get_sources_by_tier(tier: int) -> List[NewsSource]:
    return [s for s in ALL_SOURCES if s.tier == tier]


def get_source_weight(name: str) -> float:
    for s in ALL_SOURCES:
        if s.name == name:
            return s.weight
    return 0.5  # unknown source → conservative weight
