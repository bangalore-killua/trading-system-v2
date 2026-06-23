"""
fred_data.py — FRED macro economic series via fredapi.

Replaces obb.economy.fred_series with free FRED data.

FRED_API_KEY can be set in .env (free at fred.stlouisfed.org/docs/api/api_key.html).
Falls back to pandas-datareader (no API key needed) if fredapi not installed.

Key series used in macro strip:
  GDP       — Real GDP (quarterly, YoY%)
  CPIAUCSL  — CPI All Urban (monthly, YoY%)
  UNRATE    — Unemployment rate (monthly)
  DFF       — Fed funds effective rate (daily)
  GS10      — 10-Year Treasury yield (daily)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Common FRED series with human-readable labels
SERIES_LABELS: Dict[str, str] = {
    "GDP":       "Real GDP (QoQ%)",
    "CPIAUCSL":  "CPI All Urban",
    "UNRATE":    "Unemployment Rate",
    "DFF":       "Fed Funds Rate",
    "GS10":      "10-Year Treasury",
    "GS2":       "2-Year Treasury",
    "MORTGAGE30US": "30Y Mortgage Rate",
    "VIXCLS":    "VIX",
    "T10Y2Y":    "10Y-2Y Yield Spread",
    "BAMLH0A0HYM2": "HY Credit Spread",
}


def get_series(
    series_id: str,
    limit: int = 24,
    frequency: Optional[str] = None,
) -> List[dict]:
    """Fetch a FRED series and return the last N observations.

    Parameters
    ----------
    series_id : FRED series identifier (e.g. "CPIAUCSL")
    limit     : number of most recent observations to return
    frequency : optional aggregation override (e.g. "m" for monthly)

    Returns list of {"date": "YYYY-MM-DD", "value": float, "series_id": str}
    """
    data = _fetch_via_fredapi(series_id, limit, frequency)
    if data is None:
        data = _fetch_via_pandas_datareader(series_id, limit)
    if data is None:
        return []
    return data


def get_macro_snapshot() -> dict:
    """Return the latest value for the 5 key macro indicators.

    Returns dict with one entry per indicator:
      {"GDP": {"value": 2.8, "label": "Real GDP (QoQ%)", "date": "2026-01-01"}, ...}
    """
    SNAPSHOT_SERIES = ["GDP", "CPIAUCSL", "UNRATE", "DFF", "GS10"]
    snapshot = {}
    for series_id in SNAPSHOT_SERIES:
        obs = get_series(series_id, limit=2)
        if obs:
            latest = obs[-1]
            snapshot[series_id] = {
                "value": latest["value"],
                "label": SERIES_LABELS.get(series_id, series_id),
                "date":  latest["date"],
                "series_id": series_id,
            }
    return snapshot


def get_yield_curve() -> dict:
    """Return current 2Y, 5Y, 10Y, 30Y Treasury yields and spread.

    Returns {"2y": 4.5, "10y": 4.8, "spread_10_2": 0.3, "inverted": False}
    """
    yields = {}
    for series, key in [("GS2", "2y"), ("GS5", "5y"), ("GS10", "10y"), ("GS30", "30y")]:
        obs = get_series(series, limit=1)
        if obs:
            yields[key] = obs[-1]["value"]

    spread = yields.get("10y", 0) - yields.get("2y", 0) if "10y" in yields and "2y" in yields else None
    return {
        **yields,
        "spread_10_2": round(spread, 3) if spread is not None else None,
        "inverted": spread < 0 if spread is not None else None,
    }


# ── Internal fetchers ─────────────────────────────────────────────────────────

def _fetch_via_fredapi(series_id: str, limit: int, frequency: Optional[str]) -> Optional[List[dict]]:
    """Fetch via fredapi library (requires FRED_API_KEY in env)."""
    try:
        from fredapi import Fred  # type: ignore
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            return None
        fred = Fred(api_key=api_key)
        kwargs = {}
        if frequency:
            kwargs["frequency"] = frequency
        series = fred.get_series(series_id, **kwargs)
        series = series.dropna().tail(limit)
        return [
            {"date": str(d.date()), "value": float(v), "series_id": series_id}
            for d, v in series.items()
        ]
    except ImportError:
        return None
    except Exception:
        logger.debug("fredapi fetch failed for %s", series_id, exc_info=True)
        return None


def _fetch_via_pandas_datareader(series_id: str, limit: int) -> Optional[List[dict]]:
    """Fetch via pandas-datareader (no API key needed, uses FRED public endpoint)."""
    try:
        import pandas_datareader as pdr  # type: ignore
        end = datetime.today()
        start = end - timedelta(days=limit * 40)  # generous lookback
        df = pdr.get_data_fred(series_id, start=start, end=end)
        df = df.dropna().tail(limit)
        return [
            {"date": str(idx.date()), "value": float(row.iloc[0]), "series_id": series_id}
            for idx, row in df.iterrows()
        ]
    except ImportError:
        logger.info("pandas-datareader not installed — FRED data unavailable. Run: pip install pandas-datareader")
        return None
    except Exception:
        logger.debug("pandas-datareader fetch failed for %s", series_id, exc_info=True)
        return None


if __name__ == "__main__":
    snap = get_macro_snapshot()
    for k, v in snap.items():
        print(f"{k}: {v['value']} ({v['date']})")
