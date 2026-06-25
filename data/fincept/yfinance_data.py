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


def get_analysis(symbol: str) -> dict:
    """Analyst ratings, price targets, and recommendation trend.

    Returns
    -------
    {
      "recommendations": [{"date", "firm", "to_grade", "from_grade", "action"}],
      "price_target": {"mean", "high", "low", "median", "analyst_count"},
      "recommendation_mean": float,  # 1=Strong Buy, 5=Sell
    }
    """
    try:
        import yfinance as yf  # type: ignore
        t    = yf.Ticker(symbol)
        info = t.info or {}

        # Analyst price target
        price_target = {
            "mean":           info.get("targetMeanPrice"),
            "high":           info.get("targetHighPrice"),
            "low":            info.get("targetLowPrice"),
            "median":         info.get("targetMedianPrice"),
            "analyst_count":  info.get("numberOfAnalystOpinions"),
        }

        # Recent upgrades/downgrades
        recs = []
        try:
            rec_df = t.recommendations
            if rec_df is not None and not rec_df.empty:
                rec_df = rec_df.tail(20)
                for ts, row in rec_df.iterrows():
                    recs.append({
                        "date":       str(ts)[:10],
                        "firm":       row.get("Firm", ""),
                        "to_grade":   row.get("To Grade", ""),
                        "from_grade": row.get("From Grade", ""),
                        "action":     row.get("Action", ""),
                    })
        except Exception:
            pass

        return {
            "symbol":              symbol.upper(),
            "recommendations":     recs,
            "price_target":        price_target,
            "recommendation_mean": info.get("recommendationMean"),
            "recommendation_key":  info.get("recommendationKey", ""),  # "buy","hold","sell"
        }
    except Exception:
        logger.exception("get_analysis failed for %s", symbol)
        return {"symbol": symbol.upper(), "recommendations": [], "price_target": {}}


def get_earnings_data(symbol: str) -> dict:
    """Earnings history, upcoming calendar, and quarterly results.

    Returns
    -------
    {
      "quarterly": [{"date", "actual_eps", "estimate_eps", "actual_revenue", "estimate_revenue", "surprise_pct"}],
      "next_earnings_date": str | None,
      "trailing_eps": float,
      "forward_eps": float,
    }
    """
    try:
        import yfinance as yf  # type: ignore
        t    = yf.Ticker(symbol)
        info = t.info or {}

        quarterly = []
        try:
            # Earnings history (actual vs estimate)
            hist = t.earnings_history
            if hist is not None and not hist.empty:
                for ts, row in hist.tail(8).iterrows():
                    actual   = row.get("epsActual")
                    estimate = row.get("epsEstimate")
                    surprise = None
                    if actual is not None and estimate is not None and estimate != 0:
                        surprise = round((actual - estimate) / abs(estimate) * 100, 1)
                    quarterly.append({
                        "date":     str(ts)[:10],
                        "actual_eps":    actual,
                        "estimate_eps":  estimate,
                        "surprise_pct":  surprise,
                    })
        except Exception:
            pass

        # Revenue quarterly (separate from EPS)
        revenue_rows = []
        try:
            fin = t.quarterly_financials
            if fin is not None and not fin.empty and "Total Revenue" in fin.index:
                revenue_series = fin.loc["Total Revenue"]
                for col, val in revenue_series.items():
                    revenue_rows.append({"date": str(col)[:10], "revenue": val})
        except Exception:
            pass

        # Merge EPS and revenue by date
        rev_by_date = {r["date"]: r["revenue"] for r in revenue_rows}
        for q in quarterly:
            q["revenue"] = rev_by_date.get(q["date"])

        return {
            "symbol":            symbol.upper(),
            "quarterly":         quarterly,
            "next_earnings_date": info.get("earningsDate"),
            "trailing_eps":      info.get("trailingEps"),
            "forward_eps":       info.get("forwardEps"),
            "trailing_pe":       info.get("trailingPE"),
            "forward_pe":        info.get("forwardPE"),
        }
    except Exception:
        logger.exception("get_earnings_data failed for %s", symbol)
        return {"symbol": symbol.upper(), "quarterly": []}


def get_financials_detailed(symbol: str, period: str = "quarterly") -> dict:
    """Detailed financials: income statement, balance sheet, cash flow.

    Parameters
    ----------
    period : "quarterly" | "annual"

    Returns
    -------
    {
      "income_statement": [{date, total_revenue, gross_profit, net_income, ebitda, ...}],
      "balance_sheet":    [{date, total_assets, total_liabilities, total_equity, cash, debt}],
      "cash_flow":        [{date, operating_cashflow, capex, free_cashflow, ...}],
    }
    """
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(symbol)

        def df_to_rows(df) -> list:
            if df is None or df.empty:
                return []
            rows = []
            for col in df.columns:
                row = {"date": str(col)[:10]}
                for idx in df.index:
                    val = df.loc[idx, col]
                    row[str(idx).lower().replace(" ", "_")] = (
                        float(val) if pd.notna(val) else None
                    )
                rows.append(row)
            return rows

        if period == "annual":
            income_df  = t.financials
            balance_df = t.balance_sheet
            cashflow_df = t.cashflow
        else:
            income_df  = t.quarterly_financials
            balance_df = t.quarterly_balance_sheet
            cashflow_df = t.quarterly_cashflow

        income_rows  = df_to_rows(income_df)
        balance_rows = df_to_rows(balance_df)
        cf_rows      = df_to_rows(cashflow_df)

        # Compute FCF = operating_cash_flow - capex
        for row in cf_rows:
            ocf   = row.get("operating_cash_flow") or row.get("total_cash_from_operating_activities")
            capex = row.get("capital_expenditure") or row.get("capital_expenditures")
            if ocf is not None and capex is not None:
                row["free_cash_flow"] = ocf + capex  # capex is usually negative

        return {
            "symbol":           symbol.upper(),
            "period":           period,
            "income_statement": income_rows[:8],
            "balance_sheet":    balance_rows[:8],
            "cash_flow":        cf_rows[:8],
        }
    except Exception:
        logger.exception("get_financials_detailed failed for %s", symbol)
        return {
            "symbol": symbol.upper(), "period": period,
            "income_statement": [], "balance_sheet": [], "cash_flow": [],
        }


def get_holdings_data(symbol: str) -> dict:
    """Institutional holders, mutual funds, insiders, and options activity.

    Returns
    -------
    {
      "institutional": [{holder, shares, value, pct_held, date_reported}],
      "mutual_funds":  [{holder, shares, value, pct_held, date_reported}],
      "insiders":      [{name, title, transaction, shares, value, date}],
    }
    """
    try:
        import yfinance as yf  # type: ignore
        t = yf.Ticker(symbol)

        def parse_holders(df) -> list:
            if df is None or df.empty:
                return []
            rows = []
            for _, row in df.head(10).iterrows():
                rows.append({
                    "holder":         str(row.get("Holder",        row.get("Name", ""))),
                    "shares":         _safe_int(row.get("Shares")),
                    "value":          _safe_float(row.get("Value")),
                    "pct_held":       _safe_float(row.get("% Out", row.get("Pct Held"))),
                    "date_reported":  str(row.get("Date Reported", ""))[:10],
                })
            return rows

        def parse_insiders(df) -> list:
            if df is None or df.empty:
                return []
            rows = []
            for _, row in df.head(20).iterrows():
                rows.append({
                    "name":        str(row.get("Name",         "")),
                    "title":       str(row.get("Title",        "")),
                    "transaction": str(row.get("Transaction",  "")),
                    "shares":      _safe_int(row.get("Shares")),
                    "value":       _safe_float(row.get("Value")),
                    "date":        str(row.get("Start Date",   ""))[:10],
                })
            return rows

        institutional = parse_holders(t.institutional_holders)
        mutual_funds  = parse_holders(t.mutualfund_holders)
        insiders      = parse_insiders(t.insider_transactions)

        return {
            "symbol":       symbol.upper(),
            "institutional": institutional,
            "mutual_funds":  mutual_funds,
            "insiders":      insiders,
        }
    except Exception:
        logger.exception("get_holdings_data failed for %s", symbol)
        return {
            "symbol": symbol.upper(),
            "institutional": [], "mutual_funds": [], "insiders": [],
        }


def _safe_int(val) -> Optional[int]:
    try:
        return int(val) if val is not None and str(val) != "nan" else None
    except Exception:
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return round(float(val), 4) if val is not None and str(val) != "nan" else None
    except Exception:
        return None


if __name__ == "__main__":
    q = get_quote("AAPL")
    print("Quote:", q)
    info = get_info("NVDA")
    print("P/E:", info.get("pe_trailing"))
