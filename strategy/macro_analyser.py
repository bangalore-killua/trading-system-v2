"""
macro_analyser.py — Macro Context for Regime Detection

Provides macro indicators consumed by the regime detector:
  - VIX (CBOE Volatility Index) — proxy for market fear
  - Yield spread (10Y - 2Y Treasury) — recession predictor
  - Fear & Greed Index (CNN Money approximation via market data)
  - SPY 50/200 MA trend — primary regime signal

Data sources (all free):
  - yfinance: VIX (^VIX), SPY, TLT, IEI, ^TNX, ^IRX
  - pandas-datareader: FRED series (FEDFUNDS, UNRATE, CPIAUCSL) — optional

This module is called by the regime detector every 4 hours and on startup.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MacroAnalyser:
    """Fetches and computes macro indicators for regime classification."""

    # FRED series IDs for optional macro enrichment
    FRED_SERIES = {
        "fed_funds_rate": "FEDFUNDS",
        "unemployment_rate": "UNRATE",
        "cpi_yoy": "CPIAUCSL",
    }

    def __init__(self, use_fred: bool = False):
        # FRED requires pandas-datareader; disable if not installed
        self._use_fred = use_fred
        self._cache: Dict[str, dict] = {}
        self._cache_ttl_minutes = 60

    def get_macro_snapshot(self) -> Dict:
        """Return a dict of current macro indicators.

        All values are floats or None if unavailable.
        Cached for 60 minutes to avoid repeated yfinance calls.
        """
        cached = self._cache.get("snapshot")
        if cached:
            age = (datetime.utcnow() - cached["_fetched_at"]).total_seconds() / 60
            if age < self._cache_ttl_minutes:
                return cached

        snapshot = {}

        # ── VIX ────────────────────────────────────────────────────────────
        snapshot["vix"] = self._fetch_last_close("^VIX")

        # ── Yield Spread (10Y - 2Y) ─────────────────────────────────────────
        tnx = self._fetch_last_close("^TNX")  # 10-year yield * 10 (Yahoo format)
        irx = self._fetch_last_close("^IRX")  # 3-month yield * 10
        tyx = self._fetch_last_close("^TYX")  # 30-year yield * 10

        # ^TNX is reported as yield * 10 on some Yahoo endpoints — normalise
        if tnx and tnx > 20:
            tnx = tnx / 10
        if irx and irx > 20:
            irx = irx / 10
        if tyx and tyx > 20:
            tyx = tyx / 10

        snapshot["yield_10y"] = tnx
        snapshot["yield_3m"] = irx
        snapshot["yield_30y"] = tyx

        # 10Y-2Y spread approximation (use 3M as proxy when 2Y not available)
        if tnx and irx:
            snapshot["yield_spread_10y2y"] = round(tnx - irx, 3)
            snapshot["inverted_yield_curve"] = int(tnx < irx)
        else:
            snapshot["yield_spread_10y2y"] = None
            snapshot["inverted_yield_curve"] = 0

        # ── SPY Trend (50/200 MA alignment) ────────────────────────────────
        spy_data = self._fetch_spy_trend()
        snapshot.update(spy_data)

        # ── Fear & Greed approximation ──────────────────────────────────────
        # Crude proxy: high VIX + negative SPY momentum = Fear
        vix = snapshot.get("vix", 20)
        spy_vs_200 = snapshot.get("spy_vs_200ma_pct", 0) or 0
        if vix and vix > 30 and spy_vs_200 < -5:
            snapshot["fear_greed_proxy"] = "EXTREME_FEAR"
        elif vix and vix > 20 and spy_vs_200 < 0:
            snapshot["fear_greed_proxy"] = "FEAR"
        elif vix and vix < 15 and spy_vs_200 > 5:
            snapshot["fear_greed_proxy"] = "GREED"
        elif vix and vix < 12:
            snapshot["fear_greed_proxy"] = "EXTREME_GREED"
        else:
            snapshot["fear_greed_proxy"] = "NEUTRAL"

        # ── Optional FRED data ──────────────────────────────────────────────
        if self._use_fred:
            fred_data = self._fetch_fred()
            snapshot.update(fred_data)

        snapshot["_fetched_at"] = datetime.utcnow()
        self._cache["snapshot"] = snapshot
        logger.debug("Macro snapshot: VIX=%.1f spread=%.2f SPY_vs_200=%.1f%%",
                     vix or 0, snapshot.get("yield_spread_10y2y") or 0, spy_vs_200)
        return snapshot

    def _fetch_spy_trend(self) -> Dict:
        """Compute SPY 50MA / 200MA position and trend alignment."""
        try:
            import yfinance as yf  # type: ignore

            spy = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
            if spy.empty:
                return {}

            close = spy["Close"]
            last = float(close.iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

            result = {
                "spy_price": round(last, 2),
                "spy_sma50": round(sma50, 2),
                "spy_sma200": round(sma200, 2) if sma200 else None,
                "spy_vs_50ma_pct": round((last - sma50) / sma50 * 100, 2),
                "spy_vs_200ma_pct": round((last - sma200) / sma200 * 100, 2) if sma200 else None,
                "spy_above_50ma": int(last > sma50),
                "spy_above_200ma": int(last > sma200) if sma200 else None,
                # Golden cross: 50MA above 200MA
                "spy_golden_cross": int(sma50 > sma200) if sma200 else None,
            }

            # 20-day return as momentum proxy
            if len(close) >= 20:
                price_20d_ago = float(close.iloc[-20])
                result["spy_momentum_20d_pct"] = round(
                    (last - price_20d_ago) / price_20d_ago * 100, 2
                )

            return result

        except Exception as exc:
            logger.debug("SPY trend fetch error: %s", exc)
            return {}

    def _fetch_last_close(self, ticker: str) -> Optional[float]:
        """Fetch the latest close price for a Yahoo Finance ticker."""
        try:
            import yfinance as yf  # type: ignore

            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if hist.empty:
                return None
            return round(float(hist["Close"].iloc[-1]), 4)
        except Exception:
            return None

    def _fetch_fred(self) -> Dict:
        """Fetch FRED macroeconomic series via pandas-datareader."""
        result = {}
        try:
            import pandas_datareader.data as web  # type: ignore

            end = datetime.today()
            start = end - timedelta(days=60)

            for label, series_id in self.FRED_SERIES.items():
                try:
                    df = web.DataReader(series_id, "fred", start, end)
                    result[label] = float(df.iloc[-1, 0])
                except Exception:
                    result[label] = None

        except ImportError:
            logger.debug("pandas-datareader not installed; FRED data unavailable")

        return result

    def vix_is_elevated(self, threshold: float = 20.0) -> bool:
        snap = self.get_macro_snapshot()
        vix = snap.get("vix")
        return vix is not None and vix > threshold

    def yield_curve_inverted(self) -> bool:
        snap = self.get_macro_snapshot()
        return bool(snap.get("inverted_yield_curve", 0))
