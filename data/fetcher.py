"""
fetcher.py — OHLCV Data Fetcher

Primary source: yfinance (5-year history, free, no API key)
Fallback: alpaca-py historical bars (for intraday or when yfinance is slow)

Returns a pandas DataFrame with lowercase columns: open, high, low, close, volume.
Minimum 250 bars is requested so all MAs (including 200MA) are valid.

Used by position_manager._fetch_ohlcv() for both entry and exit evaluation.
"""

import asyncio
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class OHLCVFetcher:
    """Fetches historical OHLCV bars for a symbol.

    yfinance is the primary source — it's free, covers 5 years, and needs
    no API key. Alpaca historical bars are used as fallback when yfinance
    returns empty data (e.g. crypto symbols not listed on Yahoo).
    """

    def __init__(
        self,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        alpaca_paper: bool = True,
        default_bars: int = 250,
        timeframe: str = "1Day",
    ):
        self._alpaca_key = alpaca_api_key
        self._alpaca_secret = alpaca_secret_key
        self._alpaca_paper = alpaca_paper
        self._default_bars = default_bars
        self._timeframe = timeframe

        # Simple in-memory cache: symbol → (DataFrame, fetched_at timestamp)
        self._cache: dict = {}
        self._cache_ttl_seconds = 300  # 5-minute cache for daily bars

    async def get_bars(self, symbol: str, bars: int = 250) -> Optional[pd.DataFrame]:
        """Return OHLCV DataFrame for the given symbol.

        Tries yfinance first, falls back to alpaca-py historical bars.
        Returns None if both sources fail.
        """
        # Check cache
        import time
        cached = self._cache.get(symbol)
        if cached:
            df, fetched_at = cached
            if time.monotonic() - fetched_at < self._cache_ttl_seconds:
                return df

        # Run in executor — yfinance is synchronous
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, self._fetch_yfinance, symbol, bars)

        if df is None or len(df) < 50:
            logger.debug("%s: yfinance returned insufficient data, trying Alpaca", symbol)
            df = await loop.run_in_executor(None, self._fetch_alpaca, symbol, bars)

        if df is not None and not df.empty:
            self._cache[symbol] = (df, time.monotonic())

        return df

    def _fetch_yfinance(self, symbol: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch via yfinance. Returns daily OHLCV or None on failure."""
        try:
            import yfinance as yf  # type: ignore

            # Request more than needed so we always have bars after NA drops
            period = "2y" if bars <= 500 else "5y"
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                return None

            # Standardise column names
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()

            # Return only the last `bars` rows
            return df.tail(bars).copy()

        except Exception as exc:
            logger.debug("yfinance error for %s: %s", symbol, exc)
            return None

    def _fetch_alpaca(self, symbol: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch via alpaca-py historical bars API. Used as yfinance fallback."""
        if not self._alpaca_key or not self._alpaca_secret:
            return None

        try:
            from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
            from alpaca.data.requests import StockBarsRequest              # type: ignore
            from alpaca.data.timeframe import TimeFrame                    # type: ignore
            from datetime import datetime, timedelta

            client = StockHistoricalDataClient(
                api_key=self._alpaca_key,
                secret_key=self._alpaca_secret,
            )

            end = datetime.utcnow()
            # Request ~1.5x bars to account for weekends/holidays
            start = end - timedelta(days=int(bars * 1.5))

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars_df = client.get_stock_bars(request).df

            if bars_df is None or bars_df.empty:
                return None

            # alpaca-py returns MultiIndex (symbol, timestamp) — drop symbol level
            if isinstance(bars_df.index, pd.MultiIndex):
                bars_df = bars_df.xs(symbol, level=0)

            bars_df.columns = [c.lower() for c in bars_df.columns]
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in bars_df.columns]
            return bars_df[cols].dropna().tail(bars).copy()

        except Exception as exc:
            logger.debug("Alpaca bars error for %s: %s", symbol, exc)
            return None

    def get_bars_sync(self, symbol: str, bars: int = 250) -> Optional[pd.DataFrame]:
        """Synchronous wrapper — for use outside async context (e.g. tests)."""
        return self._fetch_yfinance(symbol, bars) or self._fetch_alpaca(symbol, bars)

    def invalidate(self, symbol: str) -> None:
        """Clear cache entry for a symbol (call after a known price-moving event)."""
        self._cache.pop(symbol, None)
