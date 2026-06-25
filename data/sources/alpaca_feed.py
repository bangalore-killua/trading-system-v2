"""
alpaca_feed.py — Real-time and historical US equity data via Alpaca.

We already use Alpaca as our broker — this module exposes a clean interface
to the same account's market data API (free tier: real-time SIP data).

Provides:
  - Real-time quotes (bid/ask/last) for signal polling every 5 minutes
  - Historical bars (1m–1D) as pandas DataFrame
  - Latest bar snapshot for quick price checks

Environment variables (same as broker):
  ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER (true/false)

CLI:
    python -m data.sources.alpaca_feed --symbol NVDA --bars 20
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Alpaca bar timeframe → timedelta for date range calculation
_BAR_MINUTES = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "1Day": 1440}


def _get_clients():
    """Return (StockHistoricalDataClient, StockDataStream) lazily."""
    try:
        from alpaca.data import StockHistoricalDataClient  # type: ignore
    except ImportError:
        raise ImportError("alpaca-py not installed — run: pip install alpaca-py")

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def get_bars(
    symbol: str,
    timeframe: str = "1Day",
    n_bars: int = 500,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV bars from Alpaca.

    Parameters
    ----------
    symbol    : ticker, e.g. "NVDA"
    timeframe : "1Min", "5Min", "15Min", "1Hour", "1Day"
    n_bars    : approximate number of bars (converted to date range)
    end       : end datetime (UTC, defaults to now)

    Returns
    -------
    DataFrame with columns: open, high, low, close, volume, vwap
    Index: datetime (UTC)
    """
    from alpaca.data.requests import StockBarsRequest  # type: ignore
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore

    tf_map = {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {list(tf_map)}")

    client = _get_clients()
    end_dt = end or datetime.now(timezone.utc)
    minutes_per_bar = _BAR_MINUTES.get(timeframe, 1440)
    start_dt = end_dt - timedelta(minutes=minutes_per_bar * n_bars * 1.5)  # 50% buffer for market hours

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe],
        start=start_dt,
        end=end_dt,
        limit=n_bars,
        adjustment="split",
    )
    try:
        bars = client.get_stock_bars(req)
        df = bars.df
    except Exception as e:
        logger.error("Alpaca get_bars failed for %s %s: %s", symbol, timeframe, e)
        raise

    if df is None or df.empty:
        raise ValueError(f"No Alpaca data returned for {symbol} {timeframe}")

    # Alpaca returns MultiIndex (symbol, timestamp) when requesting one symbol
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df = df.tail(n_bars)

    logger.debug("Alpaca: %s %s — %d bars (%s → %s)", symbol, timeframe, len(df),
                 df.index[0].date(), df.index[-1].date())
    return df


def get_latest_quote(symbol: str) -> dict:
    """Fetch latest bid/ask/last price for a symbol.

    Returns {"bid": float, "ask": float, "last": float, "timestamp": str}
    Used by signal poller every 5 minutes during market hours.
    """
    from alpaca.data.requests import StockLatestQuoteRequest  # type: ignore

    client = _get_clients()
    req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    try:
        quotes = client.get_stock_latest_quote(req)
        q = quotes[symbol]
        return {
            "bid":       float(q.bid_price),
            "ask":       float(q.ask_price),
            "mid":       round((float(q.bid_price) + float(q.ask_price)) / 2, 4),
            "timestamp": str(q.timestamp),
        }
    except Exception as e:
        logger.error("Alpaca latest quote failed for %s: %s", symbol, e)
        raise


def get_latest_bar(symbol: str) -> dict:
    """Fetch the most recent completed daily bar for a symbol.

    Returns {"open": float, "high": float, "low": float, "close": float,
             "volume": int, "vwap": float, "date": str}
    """
    from alpaca.data.requests import StockLatestBarRequest  # type: ignore

    client = _get_clients()
    req = StockLatestBarRequest(symbol_or_symbols=symbol)
    try:
        bars = client.get_stock_latest_bar(req)
        b = bars[symbol]
        return {
            "open":      float(b.open),
            "high":      float(b.high),
            "low":       float(b.low),
            "close":     float(b.close),
            "volume":    int(b.volume),
            "vwap":      float(b.vwap) if b.vwap else None,
            "timestamp": str(b.timestamp),
        }
    except Exception as e:
        logger.error("Alpaca latest bar failed for %s: %s", symbol, e)
        raise


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Fetch bars via Alpaca")
    parser.add_argument("--symbol",    default="NVDA")
    parser.add_argument("--timeframe", default="1Day", choices=list(_BAR_MINUTES))
    parser.add_argument("--bars",      type=int, default=20)
    args = parser.parse_args()

    df = get_bars(args.symbol, timeframe=args.timeframe, n_bars=args.bars)
    print(df.tail(10).to_string())
    q  = get_latest_quote(args.symbol)
    print(f"\nLatest quote: bid={q['bid']} ask={q['ask']} mid={q['mid']}")
