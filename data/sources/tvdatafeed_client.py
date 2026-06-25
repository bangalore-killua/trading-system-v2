"""
tvdatafeed_client.py — TradingView OHLCV via tvdatafeed.

Replaces yfinance for historical OHLCV. Provides:
  - Native 4H candles (no server-side resample needed)
  - All timeframes: 1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W, 1M
  - All markets: US stocks, international, crypto, forex, futures, indices

Uses free TradingView account. Credentials optional for most US equity data.
Set TV_USERNAME and TV_PASSWORD in .env for authenticated access (more data).

CLI:
    python -m data.sources.tvdatafeed_client --symbol NVDA --interval 4H --bars 500
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Map our interval strings → tvdatafeed Interval enum values
INTERVAL_MAP = {
    "1m":  "in_1_minute",
    "5m":  "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "1H":  "in_1_hour",
    "4H":  "in_4_hour",
    "1D":  "in_daily",
    "1W":  "in_weekly",
    "1M":  "in_monthly",
}

# Default exchange per symbol prefix (override via exchange= param)
EXCHANGE_DEFAULTS = {
    "BTC": "BINANCE",
    "ETH": "BINANCE",
}

_tv_instance = None


def _get_tv():
    global _tv_instance
    if _tv_instance is not None:
        return _tv_instance
    try:
        from tvdatafeed import TvDatafeed  # type: ignore
    except ImportError:
        raise ImportError("tvdatafeed not installed — run: pip install tvdatafeed")

    username = os.getenv("TV_USERNAME")
    password = os.getenv("TV_PASSWORD")
    if username and password:
        _tv_instance = TvDatafeed(username=username, password=password)
        logger.info("tvdatafeed: authenticated as %s", username)
    else:
        _tv_instance = TvDatafeed()
        logger.info("tvdatafeed: anonymous session (limited data)")
    return _tv_instance


def get_ohlcv(
    symbol: str,
    interval: str = "1D",
    n_bars: int = 500,
    exchange: Optional[str] = None,
    fut_contract: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars from TradingView.

    Parameters
    ----------
    symbol      : ticker, e.g. "NVDA", "AAPL", "BTCUSDT"
    interval    : one of "1m","5m","15m","30m","1H","4H","1D","1W","1M"
    n_bars      : number of bars to fetch (max ~5000 for most symbols)
    exchange    : TradingView exchange string (auto-detected for common symbols)
    fut_contract: for futures, the continuous contract number (1 = front month)

    Returns
    -------
    DataFrame with columns: open, high, low, close, volume
    Index: datetime (UTC)
    """
    interval_key = INTERVAL_MAP.get(interval)
    if interval_key is None:
        raise ValueError(f"Unknown interval '{interval}'. Valid: {list(INTERVAL_MAP)}")

    if exchange is None:
        prefix = symbol[:3].upper()
        exchange = EXCHANGE_DEFAULTS.get(prefix, "NASDAQ")

    try:
        from tvdatafeed import Interval  # type: ignore
        tv_interval = getattr(Interval, interval_key)
    except AttributeError:
        raise ValueError(f"tvdatafeed does not support interval '{interval}'")

    tv = _get_tv()
    try:
        kwargs = dict(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
        if fut_contract is not None:
            kwargs["fut_contract"] = fut_contract
        df = tv.get_hist(**kwargs)
    except Exception as e:
        logger.error("tvdatafeed fetch failed for %s/%s %s: %s", symbol, exchange, interval, e)
        raise

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} {interval} on {exchange}")

    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.sort_index(inplace=True)
    logger.debug("tvdatafeed: %s %s — %d bars (%s → %s)", symbol, interval, len(df),
                 df.index[0].date(), df.index[-1].date())
    return df


def get_multi_timeframe(symbol: str, exchange: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """Convenience: fetch 1D + 4H + 1H for a symbol in one call.

    Used by entry model feature builder and signal orchestrator.
    Returns {"1D": df, "4H": df, "1H": df}
    """
    result = {}
    for interval, bars in [("1D", 500), ("4H", 1000), ("1H", 500)]:
        try:
            result[interval] = get_ohlcv(symbol, interval=interval, n_bars=bars, exchange=exchange)
        except Exception as e:
            logger.warning("get_multi_timeframe: skipping %s %s — %s", symbol, interval, e)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(description="Fetch OHLCV via tvdatafeed")
    parser.add_argument("--symbol",   default="NVDA")
    parser.add_argument("--interval", default="1D", choices=list(INTERVAL_MAP))
    parser.add_argument("--bars",     type=int, default=20)
    parser.add_argument("--exchange", default=None)
    args = parser.parse_args()

    df = get_ohlcv(args.symbol, interval=args.interval, n_bars=args.bars, exchange=args.exchange)
    print(df.tail(10).to_string())
    print(f"\nShape: {df.shape} | Latest: {df.index[-1].date()}")
