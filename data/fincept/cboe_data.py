"""
cboe_data.py — Options chains, Greeks, IV, VIX futures via yfinance + CBOE.

Replaces OpenBB options endpoints with free data.

Options Greeks (delta/gamma/theta/vega) are estimated using Black-Scholes
when not available directly — yfinance provides IV and bid/ask, which is enough.
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_options_chain(symbol: str, expiry: Optional[str] = None) -> dict:
    """Return options chain for the nearest expiry (or specific expiry).

    Parameters
    ----------
    symbol : equity ticker
    expiry : "YYYY-MM-DD" — if None, uses nearest available expiry

    Returns dict with keys:
      expiry, underlying_price, calls (list of dicts), puts (list of dicts)
    """
    try:
        import yfinance as yf  # type: ignore
        ticker = yf.Ticker(symbol)

        # Pick expiry
        expiries = ticker.options
        if not expiries:
            return {"symbol": symbol, "calls": [], "puts": []}

        if expiry is None:
            expiry = expiries[0]
        elif expiry not in expiries:
            # Use nearest available
            expiry = expiries[0]

        chain = ticker.option_chain(expiry)
        spot = ticker.info.get("regularMarketPrice") or ticker.info.get("currentPrice") or 0

        def format_leg(df: pd.DataFrame, option_type: str) -> List[dict]:
            rows = []
            if df is None or df.empty:
                return rows
            for _, row in df.iterrows():
                strike = float(row.get("strike", 0))
                last_price = float(row.get("lastPrice", 0))
                iv = float(row.get("impliedVolatility", 0))
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                oi = int(row.get("openInterest", 0) or 0)
                vol = int(row.get("volume", 0) or 0)

                # Estimate Greeks using Black-Scholes
                greeks = _black_scholes_greeks(
                    S=spot, K=strike, T=_days_to_expiry(expiry) / 365,
                    r=0.05, sigma=iv, option_type=option_type[0].lower(),
                )
                rows.append({
                    "strike":          strike,
                    "last_price":      last_price,
                    "bid":             bid,
                    "ask":             ask,
                    "iv":              round(iv * 100, 1),
                    "volume":          vol,
                    "open_interest":   oi,
                    "delta":           greeks.get("delta"),
                    "gamma":           greeks.get("gamma"),
                    "theta":           greeks.get("theta"),
                    "vega":            greeks.get("vega"),
                    "in_the_money":    bool(row.get("inTheMoney", False)),
                })
            return rows

        return {
            "symbol":           symbol.upper(),
            "expiry":           expiry,
            "underlying_price": spot,
            "calls":            format_leg(chain.calls, "call"),
            "puts":             format_leg(chain.puts, "put"),
        }
    except Exception:
        logger.exception("get_options_chain failed for %s", symbol)
        return {"symbol": symbol, "calls": [], "puts": []}


def get_vix() -> dict:
    """Return current VIX level and recent history."""
    try:
        import yfinance as yf  # type: ignore
        ticker = yf.Ticker("^VIX")
        info = ticker.info or {}
        hist = ticker.history(period="1mo")
        vix_history = hist["Close"].round(2).tolist() if not hist.empty else []
        return {
            "current": info.get("regularMarketPrice") or (vix_history[-1] if vix_history else None),
            "history": vix_history[-20:],
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low":  info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        logger.exception("get_vix failed")
        return {"current": None, "history": []}


def get_iv_skew(symbol: str, expiry: Optional[str] = None) -> List[dict]:
    """Return IV at each strike for calls and puts — useful for skew chart.

    Returns list of {"strike", "call_iv", "put_iv"} sorted by strike.
    """
    chain = get_options_chain(symbol, expiry)
    spot = chain.get("underlying_price", 0)
    if not spot:
        return []

    call_ivs = {c["strike"]: c["iv"] for c in chain.get("calls", [])}
    put_ivs  = {p["strike"]: p["iv"] for p in chain.get("puts", [])}
    strikes = sorted(set(call_ivs) | set(put_ivs))

    # Only return strikes within ±25% of spot for relevance
    filtered = [
        {
            "strike":   s,
            "call_iv":  call_ivs.get(s),
            "put_iv":   put_ivs.get(s),
            "moneyness": round((s / spot - 1) * 100, 1),
        }
        for s in strikes
        if 0.75 * spot <= s <= 1.25 * spot
    ]
    return filtered


# ── Black-Scholes Greeks ──────────────────────────────────────────────────────

def _black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "c") -> dict:
    """Estimate Black-Scholes Greeks. Returns dict with delta/gamma/theta/vega."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    try:
        from scipy.stats import norm  # type: ignore
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        nd1 = norm.pdf(d1)

        gamma = nd1 / (S * sigma * math.sqrt(T))
        vega  = S * nd1 * math.sqrt(T) / 100  # per 1% change in IV

        if option_type == "c":
            delta = norm.cdf(d1)
            theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

        return {
            "delta": round(delta, 3),
            "gamma": round(gamma, 4),
            "theta": round(theta, 3),
            "vega":  round(vega, 3),
        }
    except ImportError:
        # scipy not installed — return raw values without Greeks
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    except Exception:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}


def _days_to_expiry(expiry_str: str) -> float:
    """Return calendar days between today and expiry date."""
    try:
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return max(1.0, float((expiry - datetime.today().date()).days))
    except Exception:
        return 30.0


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    chain = get_options_chain(sym)
    print(f"{sym} options expiry {chain.get('expiry')} — {len(chain.get('calls', []))} calls, {len(chain.get('puts', []))} puts")
    print("VIX:", get_vix()["current"])
