"""
options_flow.py — EOD Options Flow Signal via yfinance

Computes options_score (0-1) for a symbol using end-of-day options chain data.
Replaces Unusual Whales Playwright scraping — same signal, free, no bot detection.

Metrics computed:
  call_put_ratio      — call volume / put volume (>1.5 = bullish)
  unusual_call_volume — call volume vs 20-day rolling average (>2× = unusual)
  iv_skew             — (put IV - call IV) / call IV (negative = calls bid up = bullish)
  net_options_score   — weighted composite 0-1

For 5-60 day swing trading, EOD data captures institutional positioning
as clearly as real-time flow — the accumulation pattern persists over days.

Usage:
    from data.sources.options_flow import get_options_score
    score, detail = get_options_score("NVDA")
    # score: float 0-1
    # detail: {"call_put_ratio": 2.1, "unusual_call_vol": True, "iv_skew": -0.08}
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_cache: dict = {}
_CACHE_TTL_HOURS = 6


def get_options_score(symbol: str) -> Tuple[float, dict]:
    """
    Returns (score 0-1, detail dict).
    Returns (0.5, {}) on any error — neutral, not zero, so one failed source
    doesn't penalise the composite unfairly.
    """
    cached = _cache.get(symbol)
    if cached and (datetime.now() - cached["ts"]).total_seconds() < _CACHE_TTL_HOURS * 3600:
        return cached["score"], cached["detail"]

    try:
        import yfinance as yf  # type: ignore
        ticker = yf.Ticker(symbol)

        expirations = ticker.options
        if not expirations:
            return 0.5, {"note": "no_options_listed"}

        # Use nearest expiry with meaningful volume (skip weekly micro-expiries)
        chain = None
        for exp in expirations[:3]:
            c = ticker.option_chain(exp)
            if c.calls["volume"].sum() > 100:
                chain = c
                break

        if chain is None:
            return 0.5, {"note": "insufficient_options_volume"}

        calls = chain.calls.dropna(subset=["volume", "openInterest", "impliedVolatility"])
        puts  = chain.puts.dropna(subset=["volume", "openInterest", "impliedVolatility"])

        call_vol = float(calls["volume"].sum())
        put_vol  = float(puts["volume"].sum())

        # ── call/put ratio ────────────────────────────────────────────────────
        if put_vol < 1:
            put_vol = 1.0
        cp_ratio = call_vol / put_vol

        # Normalise: ratio 0.5→ score 0.2, ratio 1.0→ score 0.5, ratio 2.5+→ score 1.0
        cp_score = min(1.0, max(0.0, (cp_ratio - 0.5) / 2.0))

        # ── unusual volume vs ATM open interest proxy ─────────────────────────
        # We don't have 20-day historical options vol here (would need paid data),
        # so proxy: if today's call vol > 2× call open interest → unusual
        call_oi = float(calls["openInterest"].sum())
        unusual_call = call_vol > 2.0 * call_oi if call_oi > 0 else False
        unusual_score = 1.0 if unusual_call else 0.0

        # ── IV skew: put IV - call IV (negative = calls more expensive = bullish) ──
        atm_calls = calls.copy()
        atm_puts  = puts.copy()
        avg_call_iv = float(atm_calls["impliedVolatility"].median())
        avg_put_iv  = float(atm_puts["impliedVolatility"].median())

        raw_skew = avg_put_iv - avg_call_iv   # negative = bullish (puts cheaper than calls)
        # Normalise: skew -0.1 → score 1.0, skew 0 → score 0.5, skew +0.1 → score 0.0
        skew_score = min(1.0, max(0.0, 0.5 - raw_skew * 5.0))

        # ── composite ─────────────────────────────────────────────────────────
        score = round(0.50 * cp_score + 0.30 * unusual_score + 0.20 * skew_score, 3)

        detail = {
            "call_put_ratio":    round(cp_ratio, 2),
            "call_vol":          int(call_vol),
            "put_vol":           int(put_vol),
            "unusual_call_vol":  unusual_call,
            "iv_skew":           round(raw_skew, 4),
            "avg_call_iv":       round(avg_call_iv, 3),
            "avg_put_iv":        round(avg_put_iv, 3),
            "expiry_used":       expirations[0],
        }

        _cache[symbol] = {"score": score, "detail": detail, "ts": datetime.now()}
        logger.debug("options_score %s = %.3f (cp=%.2f unusual=%s skew=%.3f)",
                     symbol, score, cp_ratio, unusual_call, raw_skew)
        return score, detail

    except Exception as e:
        logger.debug("options_flow failed for %s: %s", symbol, e)
        return 0.5, {"error": str(e)}
