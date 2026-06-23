"""
cftc_data.py — Commitment of Traders (COT) reports for futures contracts.

Free data source: CFTC publishes weekly COT reports every Friday at ~3:30 PM ET.
URL: https://www.cftc.gov/dea/futures/deafut.htm (TFF reports, disaggregated)

For simplicity we use the quandl/nasdaq-data-link free endpoint or direct CSV
download from cftc.gov. Falls back gracefully if neither is available.

Key contracts tracked (commodity code → readable name):
  ES / SP500 futures → S&P 500 positioning
  NQ / Nasdaq futures → Nasdaq 100 positioning
  GC / Gold futures → Gold positioning
  CL / WTI crude → Energy positioning
  ZN / 10-year note → Rate positioning
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CFTC_CSV_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
CFTC_CURRENT_URL = "https://www.cftc.gov/dea/futures/financial_lf.htm"

# Map human-readable names to CFTC market names (fuzzy match used)
KNOWN_CONTRACTS: Dict[str, str] = {
    "SP500":   "S&P 500 STOCK INDEX",
    "NASDAQ":  "NASDAQ MINI",
    "DOW":     "DJIA",
    "GOLD":    "GOLD",
    "WTI":     "CRUDE OIL, LIGHT SWEET",
    "NATURAL_GAS": "NATURAL GAS",
    "SILVER":  "SILVER",
    "COPPER":  "COPPER",
    "BONDS_10Y": "U.S. TREASURY BONDS",
    "BONDS_2Y": "2-YEAR U.S. TREASURY NOTES",
    "EUR_USD": "EURO FX",
    "JPY_USD": "JAPANESE YEN",
}


def get_cot_snapshot() -> List[dict]:
    """Return the latest COT net positioning for key contracts.

    Returns list of dicts: {contract, commercial_net, non_commercial_net,
                             commercial_pct_change, date, sentiment}
    """
    df = _fetch_cot_data()
    if df is None or df.empty:
        return []

    results = []
    for readable_name, cftc_name in KNOWN_CONTRACTS.items():
        row = _find_contract(df, cftc_name)
        if row is not None:
            results.append(_format_cot_row(readable_name, row))

    return [r for r in results if r]


def get_cot_history(contract: str, periods: int = 52) -> List[dict]:
    """Return historical COT positioning for a contract.

    Parameters
    ----------
    contract : readable name from KNOWN_CONTRACTS (e.g. "GOLD", "SP500")
    periods  : number of weekly records to return
    """
    cftc_name = KNOWN_CONTRACTS.get(contract.upper(), contract)
    df = _fetch_cot_data(years_back=2)
    if df is None or df.empty:
        return []

    # Filter to this contract
    mask = df["Market_and_Exchange_Names"].str.upper().str.contains(cftc_name[:15].upper(), na=False)
    contract_df = df[mask].sort_values("As_of_Date_In_Form_YYMMDD", ascending=False).head(periods)

    results = []
    for _, row in contract_df.iterrows():
        results.append({
            "date": str(row.get("As_of_Date_In_Form_YYMMDD", "")),
            "non_commercial_long":  int(row.get("NonComm_Positions_Long_All", 0) or 0),
            "non_commercial_short": int(row.get("NonComm_Positions_Short_All", 0) or 0),
            "commercial_long":      int(row.get("Comm_Positions_Long_All", 0) or 0),
            "commercial_short":     int(row.get("Comm_Positions_Short_All", 0) or 0),
            "non_commercial_net":   (
                int(row.get("NonComm_Positions_Long_All", 0) or 0) -
                int(row.get("NonComm_Positions_Short_All", 0) or 0)
            ),
        })
    return results


# ── Internal ──────────────────────────────────────────────────────────────────

def _fetch_cot_data(years_back: int = 1) -> Optional[pd.DataFrame]:
    """Download and parse CFTC Traders in Financial Futures (TFF) CSV."""
    year = datetime.today().year
    frames = []
    for y in range(year, year - years_back - 1, -1):
        try:
            url = CFTC_CSV_URL.format(year=y)
            resp = requests.get(url, timeout=20, headers={"User-Agent": "trading-system-v2"})
            if resp.status_code == 200:
                import zipfile
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                for name in z.namelist():
                    if name.endswith(".txt") or name.endswith(".csv"):
                        df = pd.read_csv(io.StringIO(z.read(name).decode("latin-1")))
                        frames.append(df)
                        break
        except Exception:
            logger.debug("CFTC data fetch failed for year %d", y)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _find_contract(df: pd.DataFrame, cftc_name: str) -> Optional[pd.Series]:
    """Find the most recent row for a contract name."""
    if "Market_and_Exchange_Names" not in df.columns:
        return None
    mask = df["Market_and_Exchange_Names"].str.upper().str.contains(cftc_name[:15].upper(), na=False)
    subset = df[mask]
    if subset.empty:
        return None
    return subset.sort_values("As_of_Date_In_Form_YYMMDD", ascending=False).iloc[0]


def _format_cot_row(readable_name: str, row: pd.Series) -> Optional[dict]:
    """Format a COT row into a clean dict with sentiment classification."""
    try:
        nc_long  = int(row.get("NonComm_Positions_Long_All", 0) or 0)
        nc_short = int(row.get("NonComm_Positions_Short_All", 0) or 0)
        c_long   = int(row.get("Comm_Positions_Long_All", 0) or 0)
        c_short  = int(row.get("Comm_Positions_Short_All", 0) or 0)
        nc_net = nc_long - nc_short
        c_net  = c_long  - c_short
        total_oi = nc_long + nc_short + c_long + c_short

        # Simple sentiment: non-commercial net as % of total OI
        sentiment_pct = (nc_net / total_oi * 100) if total_oi > 0 else 0
        if sentiment_pct > 15:
            sentiment = "bullish"
        elif sentiment_pct < -15:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "contract":             readable_name,
            "date":                 str(row.get("As_of_Date_In_Form_YYMMDD", "")),
            "non_commercial_long":  nc_long,
            "non_commercial_short": nc_short,
            "non_commercial_net":   nc_net,
            "commercial_long":      c_long,
            "commercial_short":     c_short,
            "commercial_net":       c_net,
            "sentiment":            sentiment,
            "sentiment_pct":        round(sentiment_pct, 1),
        }
    except Exception:
        return None


if __name__ == "__main__":
    snap = get_cot_snapshot()
    for row in snap[:5]:
        print(f"{row['contract']}: NC net={row.get('non_commercial_net', 'N/A')} ({row.get('sentiment', '')})")
