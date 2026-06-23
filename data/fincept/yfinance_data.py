"""
yfinance_data.py — Stock quotes, OHLCV, financials, ratios, news via yfinance.

Replaces OpenBB equity endpoints with free yfinance data.
All functions are synchronous and return plain dicts or DataFrames.

OpenBB equivalents replaced:
  obb.equity.price.quote       → get_quote()
  obb.equity.price.historical  → get_ohlcv()
  obb.equity.fundamental.*     → get_financials(), get_info()
  obb.news.*                   → get_news()
"""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def get_quote(symbol: str) -> dict:
    """Return current quote for a ticker."""
    try:
        import yfinance as yf  # type: ignore
        info = yf.Ticker(symbol).info or {}
        return {
            "symbol":          symbol.upper(),
            "price":           info.get("regularMarketPrice") or info.get("currentPrice"),
            "change":          info.get("regularMarketChange"),
            "change_pct":      info.get("regularMarketChangePercent"),
            "volume":          info.get("regularMarketVolume"),
            "market_cap":      info.get("marketCap"),
            "52w_high":        info.get("fiftyTwoWeekHigh"),
            "52w_low":         info.get("fiftyTwoWeekLow"),
            "avg_volume_10d":  info.get("averageVolume10days"),
            "beta":            info.get("beta"),
            "short_name":      info.get("shortName", symbol),
        }
    except Exception:
        logger.exception("get_quote failed for %s", symbol)
        return {"symbol": symbol.upper()}


def get_ohlcv(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Return OHLCV DataFrame for a ticker.

    period  : 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    try:
        import yfinance as yf  # type: ignore
        return yf.Ticker(symbol).history(period=period, interval=interval)
    except Exception:
        logger.exception("get_ohlcv failed for %s", symbol)
        return pd.DataFrame()


def get_info(symbol: str) -> dict:
    """Return fundamental info for a ticker (P/E, EPS, margins, etc.)."""
    try:
        import yfinance as yf  # type: ignore
        info = yf.Ticker(symbol).info or {}
        return {
            "symbol":          symbol.upper(),
            "pe_trailing":     info.get("trailingPE"),
            "pe_forward":      info.get("forwardPE"),
            "eps_trailing":    info.get("trailingEps"),
            "eps_forward":     info.get("forwardEps"),
            "price_to_book":   info.get("priceToBook"),
            "price_to_sales":  info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda":    info.get("enterpriseToEbitda"),
            "debt_to_equity":  info.get("debtToEquity"),
            "roe":             info.get("returnOnEquity"),
            "roa":             info.get("returnOnAssets"),
            "profit_margin":   info.get("profitMargins"),
            "gross_margin":    info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "revenue_ttm":     info.get("totalRevenue"),
            "fcf":             info.get("freeCashflow"),
            "sector":          info.get("sector"),
            "industry":        info.get("industry"),
            "employees":       info.get("fullTimeEmployees"),
            "description":     (info.get("longBusinessSummary") or "")[:500],
        }
    except Exception:
        logger.exception("get_info failed for %s", symbol)
        return {"symbol": symbol.upper()}


def get_financials(symbol: str) -> dict:
    """Return annual income statement, balance sheet, and cash flow data."""
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(symbol)
        return {
            "income_statement": t.financials.to_dict() if t.financials is not None else {},
            "balance_sheet":    t.balance_sheet.to_dict() if t.balance_sheet is not None else {},
            "cash_flow":        t.cashflow.to_dict() if t.cashflow is not None else {},
        }
    except Exception:
        logger.exception("get_financials failed for %s", symbol)
        return {"income_statement": {}, "balance_sheet": {}, "cash_flow": {}}


def get_news(symbol: str, max_items: int = 10) -> List[dict]:
    """Return recent news articles for a ticker."""
    try:
        import yfinance as yf  # type: ignore
        news = yf.Ticker(symbol).news or []
        result = []
        for item in news[:max_items]:
            result.append({
                "title":        item.get("title", ""),
                "url":          item.get("link", ""),
                "publisher":    item.get("publisher", ""),
                "published_at": item.get("providerPublishTime"),
                "thumbnail":    (item.get("thumbnail") or {}).get("resolutions", [{}])[:1],
            })
        return result
    except Exception:
        logger.exception("get_news failed for %s", symbol)
        return []


def get_batch_quotes(symbols: List[str]) -> Dict[str, dict]:
    """Fetch quotes for multiple tickers in one yfinance download call."""
    try:
        import yfinance as yf  # type: ignore
        data = yf.download(
            symbols, period="2d", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False,
        )
        result = {}
        for sym in symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    sym_data = data[sym]
                else:
                    sym_data = data
                if sym_data is not None and not sym_data.empty:
                    last = sym_data.iloc[-1]
                    prev = sym_data.iloc[-2] if len(sym_data) > 1 else last
                    pct = float((last["Close"] - prev["Close"]) / prev["Close"] * 100) if prev["Close"] else 0
                    result[sym] = {
                        "symbol":      sym,
                        "price":       float(last["Close"]),
                        "change_pct":  round(pct, 2),
                        "volume":      int(last.get("Volume", 0)),
                    }
            except Exception:
                result[sym] = {"symbol": sym}
        return result
    except Exception:
        logger.exception("get_batch_quotes failed")
        return {sym: {"symbol": sym} for sym in symbols}


if __name__ == "__main__":
    q = get_quote("AAPL")
    print("Quote:", q)
    info = get_info("NVDA")
    print("P/E:", info.get("pe_trailing"))
