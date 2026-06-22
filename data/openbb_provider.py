"""
openbb_provider.py — OpenBB 4.x Data Provider

Thin async wrapper around the OpenBB Python SDK (openbb==4.3.2).
Acts as a unified second-tier source behind EDGAR/Alpaca and in front of FMP:

    Fundamentals : EDGAR  →  OpenBB/FMP  →  direct FMP
    Macro (FRED) : OpenBB →  yfinance fallback
    Company news : OpenBB →  RSS aggregator

All methods return None / [] on any error — callers fall through to their
next source without crashing.  Every degradation fires notify_failure().

PAT (Personal Access Token) unlocks premium providers like Benzinga.
Without a PAT the yfinance + FMP providers still work.
Get a free PAT at: https://my.openbb.co
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OpenBBProvider:
    """Lazy-initialised OpenBB 4.x wrapper.

    Pass the instance to FundamentalsFetcher, MacroAnalyser, and NewsAggregator
    at construction time in run_server.py.

    All public methods are async and safe to call before init completes —
    they will attempt init on first call.
    """

    def __init__(self, pat: str = "", fmp_api_key: str = ""):
        self._pat = pat or os.environ.get("OPENBB_PAT", "")
        self._fmp_key = fmp_api_key or os.environ.get("FMP_API_KEY", "")
        self._obb = None
        self._init_attempted = False
        self._available = False  # set True only if init succeeds

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _ensure_init(self) -> bool:
        """Lazy-initialise the OpenBB SDK. Returns True if ready."""
        if self._init_attempted:
            return self._available
        self._init_attempted = True
        try:
            from openbb import obb  # noqa: F401  (may trigger HF/provider downloads)
            if self._pat:
                obb.account.login(pat=self._pat)
                logger.info("OpenBB: authenticated with PAT")
            else:
                logger.info("OpenBB: no PAT — using free providers (yfinance, fmp)")
            # Set FMP key so obb.equity.fundamental.* can use it
            if self._fmp_key:
                try:
                    obb.user.credentials.fmp_api_key = self._fmp_key
                except Exception:
                    pass  # credentials object varies by version
            self._obb = obb
            self._available = True
        except Exception as exc:
            logger.warning("OpenBB init failed — OpenBB data source disabled: %s", exc)
            self._available = False
            from server.telegram_alerts import notify_failure
            notify_failure(
                source="OpenBBProvider.__init__",
                error_type=type(exc).__name__,
                cause=str(exc),
                affects=(
                    "OpenBB data tier disabled — fundamentals fall back to direct FMP, "
                    "macro falls back to yfinance, company news uses RSS only"
                ),
            )
        return self._available

    # ── Fundamentals ──────────────────────────────────────────────────────────

    async def fundamentals(self, symbol: str) -> Optional[Dict]:
        """Fetch key fundamentals via OpenBB (FMP provider).

        Returns a dict compatible with FundamentalsFetcher's cache format,
        or None if unavailable.

        Fields returned: pe_ratio, roe, revenue_growth_yoy, debt_equity,
        eps_ttm, mktcap, mktcap_bucket, sector, industry,
        earnings_revision_3m.
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._sync_fundamentals, symbol)
        except Exception as exc:
            logger.debug("OpenBB fundamentals failed for %s: %s", symbol, exc)
            from server.telegram_alerts import notify_failure
            notify_failure(
                source=f"OpenBBProvider.fundamentals({symbol})",
                error_type=type(exc).__name__,
                cause=str(exc),
                affects=f"Fundamentals for {symbol} — falling back to direct FMP",
            )
            return None

    def _sync_fundamentals(self, symbol: str) -> Optional[Dict]:
        if not self._ensure_init():
            return None
        obb = self._obb

        try:
            # Key metrics: PE, ROE, revenue growth, D/E ratio
            metrics_res = obb.equity.fundamental.metrics(symbol, provider="fmp")
            metrics = metrics_res.results[0] if metrics_res and metrics_res.results else None
        except Exception:
            metrics = None

        try:
            # Profile: sector, industry, market cap
            profile_res = obb.equity.fundamental.overview(symbol, provider="fmp")
            profile = profile_res.results[0] if profile_res and profile_res.results else None
        except Exception:
            profile = None

        try:
            # Analyst EPS estimates for revision signal
            est_res = obb.equity.estimates.consensus(symbol, provider="fmp")
            estimates = est_res.results[0] if est_res and est_res.results else None
        except Exception:
            estimates = None

        if not metrics and not profile:
            return None

        from data.fundamentals_fetcher import _mktcap_to_bucket

        mktcap = getattr(profile, "market_cap", None) if profile else None
        pe = getattr(metrics, "pe", None) if metrics else None
        roe = getattr(metrics, "roe", None) if metrics else None
        revenue_growth = getattr(metrics, "revenue_growth", None) if metrics else None
        debt_equity = getattr(metrics, "debt_to_equity", None) if metrics else None
        eps = getattr(metrics, "eps", None) if metrics else None
        sector = getattr(profile, "sector", "Unknown") if profile else "Unknown"
        industry = getattr(profile, "industry", "") if profile else ""

        # Revenue growth from FMP is a decimal (0.12 = 12%) — convert to %
        if revenue_growth is not None:
            try:
                revenue_growth = round(float(revenue_growth) * 100, 2)
            except (TypeError, ValueError):
                revenue_growth = None

        # EPS revision: compare current vs prior year consensus EPS
        earnings_revision = None
        if estimates:
            try:
                current_eps = getattr(estimates, "estimated_eps_avg", None)
                last_eps = getattr(estimates, "last_year_actual_eps", None)
                if current_eps and last_eps and last_eps != 0:
                    earnings_revision = round((current_eps - last_eps) / abs(last_eps) * 100, 2)
            except Exception:
                pass

        return {
            "pe_ratio": round(float(pe), 1) if pe else None,
            "roe": round(float(roe) * 100, 2) if roe else None,
            "revenue_growth_yoy": revenue_growth,
            "earnings_revision_3m": earnings_revision,
            "debt_equity": round(float(debt_equity), 2) if debt_equity else None,
            "eps_ttm": float(eps) if eps else None,
            "mktcap": float(mktcap) if mktcap else None,
            "mktcap_bucket": _mktcap_to_bucket(float(mktcap) if mktcap else None),
            "sector": str(sector),
            "industry": str(industry),
            "insider_buying_30d": False,
        }

    # ── Company News ──────────────────────────────────────────────────────────

    async def company_news(
        self,
        symbols: List[str],
        limit_per_symbol: int = 10,
        hours_back: int = 24,
    ) -> List[Dict]:
        """Fetch recent company news via OpenBB (FMP provider).

        Returns a list of dicts with keys: symbol, title, url, published_utc, text.
        Articles older than hours_back are filtered out.
        """
        if not symbols:
            return []
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._sync_company_news, symbols, limit_per_symbol, hours_back
            )
        except Exception as exc:
            logger.debug("OpenBB company_news failed: %s", exc)
            from server.telegram_alerts import notify_failure
            notify_failure(
                source="OpenBBProvider.company_news",
                error_type=type(exc).__name__,
                cause=str(exc),
                affects="Company-specific news for active symbols — RSS feeds still active",
            )
            return []

    def _sync_company_news(
        self,
        symbols: List[str],
        limit_per_symbol: int,
        hours_back: int,
    ) -> List[Dict]:
        if not self._ensure_init():
            return []
        obb = self._obb
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        articles = []

        for symbol in symbols:
            try:
                result = obb.news.company(symbol, limit=limit_per_symbol, provider="fmp")
                if not result or not result.results:
                    continue
                for item in result.results:
                    pub = getattr(item, "published_utc", None) or getattr(item, "date", None)
                    if pub:
                        try:
                            if isinstance(pub, str):
                                from dateutil import parser as dateparser
                                pub = dateparser.parse(pub).replace(tzinfo=None)
                            elif hasattr(pub, "replace"):
                                pub = pub.replace(tzinfo=None)
                        except Exception:
                            pub = None
                    if pub and pub < cutoff:
                        continue  # too old
                    articles.append({
                        "symbol": symbol,
                        "title": str(getattr(item, "title", "") or ""),
                        "url": str(getattr(item, "url", "") or ""),
                        "published_utc": pub,
                        "text": str(getattr(item, "text", "") or ""),
                    })
            except Exception as exc:
                logger.debug("OpenBB news for %s failed: %s", symbol, exc)

        return articles

    # ── FRED Macro Series ──────────────────────────────────────────────────────

    async def fred_series(self, series_id: str, days_back: int = 30) -> Optional[float]:
        """Return the most-recent value for a FRED series via OpenBB.

        Useful series: T10Y2Y (10Y-2Y spread), VIXCLS (VIX), FEDFUNDS, UNRATE.
        Returns a float (latest value) or None if unavailable.
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._sync_fred_series, series_id, days_back
            )
        except Exception as exc:
            logger.debug("OpenBB FRED %s failed: %s", series_id, exc)
            from server.telegram_alerts import notify_failure
            notify_failure(
                source=f"OpenBBProvider.fred_series({series_id})",
                error_type=type(exc).__name__,
                cause=str(exc),
                affects=f"FRED macro series {series_id} — using yfinance fallback",
            )
            return None

    def _sync_fred_series(self, series_id: str, days_back: int) -> Optional[float]:
        if not self._ensure_init():
            return None
        obb = self._obb
        start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            result = obb.economy.fred_series(series_id, start_date=start)
            if not result or not result.results:
                return None
            values = [
                getattr(r, "value", None)
                for r in result.results
                if getattr(r, "value", None) is not None
            ]
            return float(values[-1]) if values else None
        except Exception as exc:
            logger.debug("OpenBB FRED sync error (%s): %s", series_id, exc)
            return None

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if OpenBB initialised successfully."""
        return self._available

    def __repr__(self) -> str:
        state = "ready" if self._available else ("not attempted" if not self._init_attempted else "failed")
        pat = "PAT" if self._pat else "no PAT"
        return f"<OpenBBProvider {state}, {pat}>"
