"""
sec_data.py — SEC filings, insider Form 4 trades, institutional 13F holdings.

Replaces obb.equity.ownership.insider_trading with richer free SEC EDGAR data.

Free endpoints used:
  EDGAR full-text search: https://efts.sec.gov/LATEST/search-index?q=...
  EDGAR company filings:  https://data.sec.gov/submissions/CIK{cik}.json
  EDGAR SEC API:          https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  Insider 4 RSS:          https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "trading-system-v2 bangalore.killua@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={forms}"


def get_filings(symbol: str, forms: str = "10-K,10-Q,8-K", limit: int = 10) -> List[dict]:
    """Return recent SEC filings for a ticker.

    Parameters
    ----------
    symbol : ticker
    forms  : comma-separated form types (e.g. "10-K,10-Q,8-K,4")
    limit  : max filings to return

    Returns list of dicts with keys: form, date, description, accession_number, url
    """
    cik = _get_cik(symbol)
    if not cik:
        return []
    try:
        url = SEC_SUBMISSIONS_URL.format(cik=str(cik).zfill(10))
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        wanted_forms = {f.strip() for f in forms.split(",")}
        filings = data.get("filings", {}).get("recent", {})

        form_types = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        descriptions = filings.get("primaryDocument", [])

        results = []
        for i, form in enumerate(form_types):
            if form in wanted_forms:
                acc = accessions[i].replace("-", "")
                results.append({
                    "form":             form,
                    "date":             dates[i],
                    "description":      descriptions[i],
                    "accession_number": accessions[i],
                    "url":              f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{descriptions[i]}",
                })
                if len(results) >= limit:
                    break
        return results
    except Exception:
        logger.exception("get_filings failed for %s", symbol)
        return []


def get_insider_trading(symbol: str, days_back: int = 90) -> List[dict]:
    """Return Form 4 insider transactions for a ticker.

    Returns list of dicts with: insider_name, role, transaction_date,
    transaction_type (buy/sell), shares, price, value_usd
    """
    cik = _get_cik(symbol)
    if not cik:
        return []

    try:
        url = SEC_SUBMISSIONS_URL.format(cik=str(cik).zfill(10))
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        form_types = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        cutoff = (datetime.today() - timedelta(days=days_back)).date().isoformat()
        form4_accessions = [
            (accessions[i], dates[i])
            for i, f in enumerate(form_types)
            if f == "4" and dates[i] >= cutoff
        ][:20]

        trades = []
        for acc, filing_date in form4_accessions:
            trade_data = _parse_form4(cik, acc, filing_date)
            if trade_data:
                trades.append(trade_data)

        return trades
    except Exception:
        logger.exception("get_insider_trading failed for %s", symbol)
        return []


def get_institutional_holdings(symbol: str) -> List[dict]:
    """Return top institutional 13F holders.

    Uses yfinance as a proxy since parsing 13F directly requires more effort.
    Returns list of {"holder", "shares", "date_reported", "pct_held"}.
    """
    try:
        import yfinance as yf  # type: ignore
        holders = yf.Ticker(symbol).institutional_holders
        if holders is None or holders.empty:
            return []
        result = []
        for _, row in holders.iterrows():
            result.append({
                "holder":        row.get("Holder", ""),
                "shares":        int(row.get("Shares", 0)),
                "date_reported": str(row.get("Date Reported", "")),
                "pct_held":      float(row.get("% Out", 0)),
                "value_usd":     int(row.get("Value", 0)),
            })
        return result[:15]
    except Exception:
        logger.exception("get_institutional_holdings failed for %s", symbol)
        return []


# ── Internal helpers ──────────────────────────────────────────────────────────

_cik_cache: Dict[str, Optional[int]] = {}


def _get_cik(symbol: str) -> Optional[int]:
    """Resolve ticker → CIK via SEC EDGAR company search."""
    symbol = symbol.upper()
    if symbol in _cik_cache:
        return _cik_cache[symbol]

    try:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={symbol}&type=10-K&dateb=&owner=include&count=1&search_text=&action=getcompany&output=atom"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        # Parse CIK from response — look for CIK number
        match = re.search(r"CIK=(\d+)", resp.text)
        if match:
            cik = int(match.group(1))
            _cik_cache[symbol] = cik
            return cik
    except Exception:
        pass

    try:
        # Try EDGAR company facts API directly
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index?q=%22ticker%22&dateRange=custom",
            headers=SEC_HEADERS, timeout=5,
        )
    except Exception:
        pass

    _cik_cache[symbol] = None
    return None


def _parse_form4(cik: int, accession_number: str, filing_date: str) -> Optional[dict]:
    """Parse a Form 4 filing to extract the core trade data."""
    try:
        acc = accession_number.replace("-", "")
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{accession_number}-index.htm"
        resp = requests.get(idx_url, headers=SEC_HEADERS, timeout=5)
        # Find the .xml file in the index
        xml_match = re.search(r'href="([^"]+\.xml)"', resp.text)
        if not xml_match:
            return None

        xml_path = xml_match.group(1)
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
        xml_url = base + xml_path.split("/")[-1]
        xml_resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=5)
        xml = xml_resp.text

        def extract(tag: str) -> Optional[str]:
            m = re.search(f"<{tag}>(.*?)</{tag}>", xml, re.DOTALL)
            return m.group(1).strip() if m else None

        trans_type = extract("transactionCode")
        shares_str = extract("transactionShares") or extract("sharesOwnedFollowingTransaction")
        price_str = extract("transactionPricePerShare")
        name_str = extract("rptOwnerName")
        role_str = extract("officerTitle") or extract("isOfficer") or "Unknown"

        shares = float(shares_str) if shares_str else None
        price = float(price_str) if price_str else None

        return {
            "insider_name":      name_str or "Unknown",
            "role":              role_str,
            "transaction_date":  filing_date,
            "transaction_type":  "buy" if trans_type == "P" else "sell" if trans_type == "S" else trans_type,
            "shares":            shares,
            "price":             price,
            "value_usd":         round(shares * price, 0) if shares and price else None,
            "filing_url":        f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/",
        }
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Recent filings for {sym}:")
    for f in get_filings(sym, limit=5):
        print(f"  {f['form']} {f['date']}: {f['description']}")
