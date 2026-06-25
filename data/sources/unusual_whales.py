"""
unusual_whales.py — Options flow via Unusual Whales (Playwright + fallback).

Large OTM call buying 3–5 days before a stock move = institutional signal.
Call/put ratio spike = directional bias.

Fallback chain:
  1. unusualwhales.com (Playwright, primary)
  2. Barchart options summary (Playwright, simpler)
  3. yfinance option chain (API, always available — less filtered)

Returns neutral score (0.5) if all sources fail.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Neutral score used when data is unavailable — neither bullish nor bearish
NEUTRAL_SCORE = 0.5


class _BarchartFallback:
    name = "barchart_options"

    def fetch(self, symbol: str) -> dict:
        from data.sources.browser_scraper import BrowserScraper
        from bs4 import BeautifulSoup  # type: ignore

        url = f"https://www.barchart.com/stocks/quotes/{symbol}/put-call-ratios"
        with BrowserScraper(stealth=True) as s:
            html = s.fetch(url, wait_for=".bc-table", delay=(3.0, 6.0))

        soup = BeautifulSoup(html, "html.parser")
        # Extract put/call ratio from the summary table
        rows = soup.select(".bc-table tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) >= 2 and "put/call" in cells[0].text.lower():
                try:
                    ratio = float(cells[1].text.strip().replace(",", ""))
                    return {
                        "call_put_ratio": round(1 / ratio, 3) if ratio > 0 else 1.0,
                        "put_call_ratio": ratio,
                        "unusual_activity": [],
                        "data_quality": "medium",
                        "source": "barchart",
                    }
                except (ValueError, ZeroDivisionError):
                    pass
        raise ValueError(f"Barchart: could not parse put/call ratio for {symbol}")


class _YFinanceFallback:
    name = "yfinance_options"

    def fetch(self, symbol: str) -> dict:
        import yfinance as yf  # type: ignore

        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            raise ValueError(f"yfinance: no option expirations for {symbol}")

        # Use nearest expiry
        chain      = ticker.option_chain(expirations[0])
        calls_vol  = int(chain.calls["volume"].fillna(0).sum())
        puts_vol   = int(chain.puts["volume"].fillna(0).sum())
        total      = calls_vol + puts_vol

        return {
            "call_put_ratio":  round(calls_vol / max(puts_vol, 1), 3),
            "put_call_ratio":  round(puts_vol / max(calls_vol, 1), 3),
            "calls_volume":    calls_vol,
            "puts_volume":     puts_vol,
            "unusual_activity": [],  # yfinance has no unusual activity filter
            "data_quality":    "low",
            "source":          "yfinance",
        }


class UnusualWhales:
    """Options flow from Unusual Whales with automatic fallback.

    Usage:
        uw = UnusualWhales()
        data, source, is_fallback = uw.get("NVDA")
        score = compute_options_score(data) if data else 0.5
    """

    name = "unusual_whales"
    fallbacks = [_BarchartFallback(), _YFinanceFallback()]

    def get(self, symbol: str) -> tuple:
        from data.sources.base_source import SourceWithFallback, _notify_degraded, _notify_critical

        try:
            data = self._fetch(symbol)
            return data, self.name, False
        except Exception as primary_err:
            logger.warning("unusual_whales: primary failed for %s — %s", symbol, primary_err)
            for fb in self.fallbacks:
                try:
                    data = fb.fetch(symbol)
                    _notify_degraded(self.name, symbol, fb.name)
                    return data, fb.name, True
                except Exception as fb_err:
                    logger.warning("unusual_whales: fallback %s failed for %s — %s", fb.name, symbol, fb_err)
            _notify_critical(self.name, symbol, str(primary_err))
            return None, None, True

    def _fetch(self, symbol: str) -> dict:
        from data.sources.browser_scraper import BrowserScraper
        from bs4 import BeautifulSoup  # type: ignore

        url = f"https://unusualwhales.com/stock/{symbol}/options-flow"
        with BrowserScraper(stealth=True) as s:
            html = s.fetch(url, wait_for=".flow-table, [data-testid='flow-row']",
                           delay=(3.0, 6.0), timeout=20_000)

        return _parse_unusual_whales(html, symbol)


def _parse_unusual_whales(html: str, symbol: str) -> dict:
    """Parse options flow table from Unusual Whales HTML."""
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(html, "html.parser")

    # Try to find flow rows
    rows = soup.select("[data-testid='flow-row'], .flow-table tr")
    unusual = []
    calls_premium = 0.0
    puts_premium  = 0.0

    for row in rows[:50]:  # cap at 50 most recent
        cells = row.select("td")
        if len(cells) < 5:
            continue
        texts = [c.get_text(strip=True) for c in cells]
        try:
            option_type = "call" if "C" in texts[2].upper() else "put"
            # Extract premium (e.g. "$2.4M" → 2_400_000)
            premium_text = texts[-1] if texts else "0"
            premium = _parse_dollar(premium_text)

            if option_type == "call":
                calls_premium += premium
            else:
                puts_premium += premium

            if premium > 500_000:  # unusual = >$500k single flow
                unusual.append({
                    "type":    option_type,
                    "premium": premium,
                    "raw":     texts,
                })
        except Exception:
            continue

    total = calls_premium + puts_premium
    call_put_ratio = round(calls_premium / max(puts_premium, 1), 3)

    return {
        "call_put_ratio":   call_put_ratio,
        "calls_premium":    calls_premium,
        "puts_premium":     puts_premium,
        "total_premium":    total,
        "unusual_activity": unusual,
        "data_quality":     "high",
        "source":           "unusual_whales",
    }


def compute_options_score(data: Optional[dict]) -> float:
    """Convert options flow data to a 0-1 signal score.

    0.5 = neutral, >0.5 = bullish bias, <0.5 = bearish bias.
    """
    if data is None:
        return NEUTRAL_SCORE

    ratio = data.get("call_put_ratio", 1.0)
    # ratio 2.0+ → bullish (score ~1.0), ratio 0.5 → bearish (score ~0.0)
    # Normalize: log scale around 1.0
    import math
    log_ratio = math.log(max(ratio, 0.01)) / math.log(2)  # log2 scale
    score = max(0.0, min(1.0, 0.5 + log_ratio * 0.25))
    return round(score, 3)


def _parse_dollar(text: str) -> float:
    """Parse dollar strings like '$2.4M', '$500K', '$1.2B' → float."""
    import re
    text = text.replace("$", "").replace(",", "").strip().upper()
    multipliers = {"B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            try:
                return float(text[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0
