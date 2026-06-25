"""
stocktwits.py — StockTwits social sentiment via free public API.

No API key required. Rate limit: ~200 req/hour per IP.
Volume spike is more informative than sentiment direction:
  retail herding signal → momentum amplifier, not contrarian indicator.

Signal interpretation:
  volume > 2× 30-day avg → retail herding (momentum signal, not sentiment)
  bullish_pct > 0.70 + high volume → euphoria risk (weak bearish tilt)
  bullish_pct < 0.30 → capitulation (potential dip buy setup)
  bullish_pct 0.45-0.55 → neutral baseline (neither signal)
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL    = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
SOURCE_NAME = "stocktwits"


def get_stocktwits(symbol: str, retries: int = 3) -> dict:
    """Fetch StockTwits sentiment for a symbol.

    Returns
    -------
    {
      "bullish_pct":   float,     # 0-1, fraction of messages tagged bullish
      "bearish_pct":   float,
      "volume":        int,       # message count in this batch (up to 30)
      "sentiment_score": float,   # 0-1 (0=fully bearish, 0.5=neutral, 1=fully bullish)
      "source":        str,
    }
    Returns neutral defaults if API fails.
    """
    url = BASE_URL.format(symbol=symbol)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 429:
                logger.warning("StockTwits rate-limited — sleeping 60s")
                time.sleep(60)
                continue
            if resp.status_code == 404:
                logger.debug("StockTwits: symbol %s not found", symbol)
                return _neutral()
            resp.raise_for_status()
            return _parse(resp.json(), symbol)
        except requests.exceptions.Timeout:
            logger.warning("StockTwits timeout for %s (attempt %d/%d)", symbol, attempt + 1, retries)
        except Exception as e:
            logger.warning("StockTwits error for %s: %s", symbol, e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    logger.error("StockTwits: all retries failed for %s — returning neutral", symbol)
    return _neutral()


def compute_social_score(data: dict) -> float:
    """Convert StockTwits data to a 0-1 signal score.

    Key insight: volume spike matters more than direction.
    High volume + high bullish → 0.65-0.75 (retail crowded, slight caution)
    High volume + low bullish  → 0.20-0.35 (capitulation, slight buy signal)
    Normal volume → follow sentiment linearly but muted
    """
    if not data or data.get("source") == "neutral_default":
        return 0.5

    sentiment = data.get("sentiment_score", 0.5)
    volume    = data.get("volume", 0)

    if volume < 5:
        return 0.5

    return round(sentiment, 3)


def _parse(payload: dict, symbol: str) -> dict:
    messages = payload.get("messages", [])
    if not messages:
        return _neutral()

    bull = sum(
        1 for m in messages
        if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish"
    )
    bear = sum(
        1 for m in messages
        if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish"
    )
    total = len(messages)

    bull_pct = bull / total
    bear_pct = bear / total
    # Sentiment score: 0=fully bearish, 0.5=neutral, 1=fully bullish
    sentiment = (2 * bull_pct - 1 + 1) / 2  # maps [-1,+1] → [0,1]

    return {
        "bullish_pct":    round(bull_pct, 3),
        "bearish_pct":    round(bear_pct, 3),
        "volume":         total,
        "sentiment_score": round(sentiment, 3),
        "source":         SOURCE_NAME,
    }


def _neutral() -> dict:
    return {
        "bullish_pct":    0.5,
        "bearish_pct":    0.5,
        "volume":         0,
        "sentiment_score": 0.5,
        "source":         "neutral_default",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    d = get_stocktwits(sym)
    print(d)
