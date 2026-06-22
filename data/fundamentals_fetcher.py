"""
fundamentals_fetcher.py — Fundamental Data Fetcher

Primary source: sec-edgar-api (free, direct from SEC XBRL, unlimited)
Fallback: FinancialModelingPrep (250 req/day free tier)

Data fetched and cached weekly (fundamentals don't change daily):
  - P/E ratio
  - Return on equity (ROE)
  - Revenue growth YoY
  - Earnings revision (3-month change in analyst consensus EPS)
  - Debt/equity ratio
  - Insider buying in last 30 days (from Form 4 via EDGAR)
  - Market cap bucket (nano/micro/small/mid/large/mega)

Refreshed every Sunday at 2am (configured in scheduler.py / config.json).
All results cached in memory; symbol → dict.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Market cap buckets in USD
MKTCAP_BUCKETS = [
    (50e6,   "nano"),
    (300e6,  "micro"),
    (2e9,    "small"),
    (10e9,   "mid"),
    (200e9,  "large"),
    (float("inf"), "mega"),
]


def _mktcap_to_bucket(mktcap: Optional[float]) -> str:
    if mktcap is None:
        return "mid"
    for threshold, label in MKTCAP_BUCKETS:
        if mktcap < threshold:
            return label
    return "mega"


class FundamentalsFetcher:
    """Fetches fundamental data. Source priority: EDGAR → OpenBB/FMP → direct FMP.

    EDGAR   — free, direct from SEC XBRL, unlimited, covers only US-listed companies
    OpenBB  — unified FMP wrapper; adds analyst estimates + EPS revisions EDGAR lacks
    FMP     — direct fallback when OpenBB is unavailable (250 req/day free tier)
    """

    EDGAR_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={key}"
    FMP_RATIOS_URL = "https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={key}"

    def __init__(
        self,
        fmp_api_key: Optional[str] = None,
        cache_ttl_hours: int = 168,  # 1 week
        openbb_provider=None,        # data.openbb_provider.OpenBBProvider
    ):
        self._fmp_key = fmp_api_key or os.environ.get("FMP_API_KEY", "")
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._openbb = openbb_provider
        # symbol → {data dict, fetched_at}
        self._cache: Dict[str, dict] = {}
        # CIK lookup (populated from EDGAR company_tickers.json)
        self._ticker_to_cik: Dict[str, str] = {}

    async def prefetch_watchlist(self, symbols: list[str]) -> None:
        """Batch-fetch fundamentals for all watchlist symbols (Sunday refresh)."""
        logger.info("Prefetching fundamentals for %d symbols", len(symbols))
        sem = asyncio.Semaphore(5)

        async def bounded(sym):
            async with sem:
                await self.fetch(sym)

        await asyncio.gather(*[bounded(s) for s in symbols], return_exceptions=True)

    def get_cached(self, symbol: str) -> Optional[Dict]:
        """Return cached fundamentals for a symbol, or None if stale/missing."""
        entry = self._cache.get(symbol)
        if not entry:
            return None
        if datetime.utcnow() - entry["fetched_at"] > self._cache_ttl:
            return None
        return entry["data"]

    async def fetch(self, symbol: str) -> Optional[Dict]:
        """Fetch fundamentals for one symbol.

        Priority: EDGAR → OpenBB/FMP → direct FMP.
        Each tier fires a notify_failure() Telegram alert if it degrades.
        """
        cached = self.get_cached(symbol)
        if cached:
            return cached

        # Tier 1: SEC EDGAR (free, unlimited, US-listed only)
        data = await self._fetch_edgar(symbol)
        if data is None:
            logger.debug("Fundamentals: EDGAR miss for %s — trying OpenBB", symbol)

        # Tier 2: OpenBB (FMP via unified SDK, adds analyst estimates EDGAR lacks)
        if data is None and self._openbb:
            try:
                data = await self._openbb.fundamentals(symbol)
            except Exception as exc:
                logger.debug("OpenBB fundamentals error for %s: %s", symbol, exc)
                from server.telegram_alerts import notify_failure
                notify_failure(
                    source=f"FundamentalsFetcher._fetch_openbb({symbol})",
                    error_type=type(exc).__name__,
                    cause=str(exc),
                    affects=f"Fundamentals tier-2 for {symbol} — falling back to direct FMP",
                )

        # Tier 3: direct FMP (250 req/day free tier)
        if data is None and self._fmp_key:
            data = await self._fetch_fmp(symbol)

        if data:
            self._cache[symbol] = {"data": data, "fetched_at": datetime.utcnow()}
            logger.debug(
                "Fundamentals for %s: P/E=%s ROE=%s rev_growth=%s",
                symbol,
                data.get("pe_ratio"),
                data.get("roe"),
                data.get("revenue_growth_yoy"),
            )
        else:
            logger.warning(
                "Fundamentals: all sources failed for %s — no data cached", symbol
            )
            from server.telegram_alerts import notify_failure
            notify_failure(
                source=f"FundamentalsFetcher.fetch({symbol})",
                error_type="AllSourcesFailed",
                cause="EDGAR, OpenBB, and FMP all returned no data",
                affects=f"P/E, ROE, debt/equity, analyst estimates for {symbol} — features will be None",
            )

        return data

    async def _fetch_edgar(self, symbol: str) -> Optional[Dict]:
        """Fetch from SEC EDGAR XBRL company facts (free, no API key)."""
        cik = await self._resolve_cik(symbol)
        if not cik:
            return None

        url = self.EDGAR_COMPANY_FACTS_URL.format(cik=cik.zfill(10))
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    url, headers={"User-Agent": "trading-system-v2 bangalore.killua@gmail.com"}
                )
            if resp.status_code != 200:
                return None

            facts = resp.json()
            return self._parse_edgar_facts(facts)

        except Exception as exc:
            logger.debug("EDGAR fundamentals error for %s: %s", symbol, exc)
            from server.telegram_alerts import notify_failure
            notify_failure(
                source=f"FundamentalsFetcher._fetch_edgar({symbol})",
                error_type=type(exc).__name__,
                cause=str(exc),
                affects=f"SEC EDGAR fundamentals for {symbol} — trying OpenBB/FMP next",
            )
            return None

    def _parse_edgar_facts(self, facts: dict) -> Dict:
        """Extract key metrics from SEC EDGAR XBRL company facts JSON."""
        data = {}
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        def _latest_annual(concept: str) -> Optional[float]:
            """Get the most recent annual (10-K) value for a GAAP concept."""
            entries = us_gaap.get(concept, {}).get("units", {})
            # Units might be USD, shares, pure — try each
            for unit_vals in entries.values():
                annual = [
                    e for e in unit_vals
                    if e.get("form") in ("10-K", "10-K/A") and e.get("val") is not None
                ]
                if annual:
                    latest = sorted(annual, key=lambda x: x.get("end", ""))[-1]
                    return float(latest["val"])
            return None

        # Revenue
        rev = _latest_annual("Revenues") or _latest_annual("RevenueFromContractWithCustomerExcludingAssessedTax")
        rev_prev = None
        # Try to get prior year revenue for YoY growth
        rev_concept = "Revenues" if _latest_annual("Revenues") else "RevenueFromContractWithCustomerExcludingAssessedTax"
        all_rev = us_gaap.get(rev_concept, {}).get("units", {})
        for unit_vals in all_rev.values():
            annual = sorted(
                [e for e in unit_vals if e.get("form") in ("10-K", "10-K/A")],
                key=lambda x: x.get("end", "")
            )
            if len(annual) >= 2:
                rev = float(annual[-1]["val"])
                rev_prev = float(annual[-2]["val"])

        if rev and rev_prev and rev_prev != 0:
            data["revenue_growth_yoy"] = round((rev - rev_prev) / abs(rev_prev) * 100, 2)
        else:
            data["revenue_growth_yoy"] = None

        # Net income for ROE
        net_income = _latest_annual("NetIncomeLoss")
        equity = _latest_annual("StockholdersEquity") or _latest_annual("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
        if net_income and equity and equity != 0:
            data["roe"] = round(net_income / equity * 100, 2)
        else:
            data["roe"] = None

        # Shares outstanding for market cap (price needed from price fetcher)
        shares = _latest_annual("CommonStockSharesOutstanding")
        data["shares_outstanding"] = shares

        # Debt / Equity
        total_debt = _latest_annual("LongTermDebt") or _latest_annual("LongTermDebtNoncurrent")
        if total_debt and equity and equity != 0:
            data["debt_equity"] = round(total_debt / equity, 2)
        else:
            data["debt_equity"] = None

        # EPS for P/E (price needed from price fetcher)
        eps = _latest_annual("EarningsPerShareBasic") or _latest_annual("EarningsPerShareDiluted")
        data["eps_ttm"] = eps
        data["pe_ratio"] = None  # Computed by caller using current price / eps_ttm
        data["mktcap"] = None    # Computed by caller using shares * current price
        data["mktcap_bucket"] = "mid"  # Will be updated by caller

        # Insider buying proxy: look at EntityRegistrantName for SEC Form 4 count
        # (Real Form 4 count comes from edgar_monitor; this is just a flag slot)
        data["insider_buying_30d"] = False
        data["earnings_revision_3m"] = None  # Requires analyst data — use FMP if available

        return data

    async def _fetch_fmp(self, symbol: str) -> Optional[Dict]:
        """Fetch from FinancialModelingPrep as fallback (250 req/day free tier)."""
        if not self._fmp_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                profile_resp, ratios_resp = await asyncio.gather(
                    client.get(self.FMP_PROFILE_URL.format(symbol=symbol, key=self._fmp_key)),
                    client.get(self.FMP_RATIOS_URL.format(symbol=symbol, key=self._fmp_key)),
                )

            profile = profile_resp.json()
            ratios = ratios_resp.json()

            if not profile or isinstance(profile, dict) and "Error Message" in profile:
                return None

            p = profile[0] if isinstance(profile, list) and profile else {}
            r = ratios[0] if isinstance(ratios, list) and ratios else {}

            mktcap = p.get("mktCap") or p.get("marketCap")

            return {
                "pe_ratio": r.get("peRatioTTM") or p.get("pe"),
                "roe": r.get("returnOnEquityTTM"),
                "revenue_growth_yoy": None,  # not in FMP profile/ratios-ttm
                "earnings_revision_3m": None,
                "debt_equity": r.get("debtEquityRatioTTM"),
                "eps_ttm": r.get("epsTTM") or p.get("eps"),
                "mktcap": mktcap,
                "mktcap_bucket": _mktcap_to_bucket(mktcap),
                "shares_outstanding": p.get("sharesOutstanding"),
                "insider_buying_30d": False,
                "sector": p.get("sector", "Unknown"),
                "industry": p.get("industry", ""),
            }

        except Exception as exc:
            logger.debug("FMP fundamentals error for %s: %s", symbol, exc)
            return None

    def enrich_with_price(self, symbol: str, current_price: float) -> None:
        """Compute price-dependent fields (P/E, market cap) using latest price."""
        entry = self._cache.get(symbol)
        if not entry:
            return
        data = entry["data"]
        eps = data.get("eps_ttm")
        if eps and eps != 0 and current_price > 0:
            data["pe_ratio"] = round(current_price / eps, 1)
        shares = data.get("shares_outstanding")
        if shares and current_price > 0:
            mktcap = shares * current_price
            data["mktcap"] = mktcap
            data["mktcap_bucket"] = _mktcap_to_bucket(mktcap)

    async def _resolve_cik(self, symbol: str) -> Optional[str]:
        """Get SEC CIK for a ticker symbol."""
        if symbol in self._ticker_to_cik:
            return self._ticker_to_cik[symbol]

        # Download company tickers JSON if cache is empty
        if not self._ticker_to_cik:
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(
                        "https://www.sec.gov/files/company_tickers.json",
                        headers={"User-Agent": "trading-system-v2 bangalore.killua@gmail.com"},
                    )
                data = resp.json()
                self._ticker_to_cik = {
                    v["ticker"].upper(): str(v["cik_str"])
                    for v in data.values()
                    if "ticker" in v and "cik_str" in v
                }
            except Exception as exc:
                logger.debug("CIK ticker mapping failed: %s", exc)

        return self._ticker_to_cik.get(symbol.upper())
