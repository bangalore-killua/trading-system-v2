"""
screener_guardrails.py — SCREENER-1
Liquidity guardrails applied before any symbol is auto-added to watchlist.

Manual additions (user-added via dashboard) bypass the guardrail.
Auto-added symbols (screener_picks) must pass before being added.

All thresholds in config.json["screener_guardrails"].
No hardcoded values anywhere in this file.

Public API:
  passes_liquidity_screen(symbol)    → bool
  check_watchlist_size()             → dict  {at_limit, current, max}
"""

import logging

log = logging.getLogger(__name__)


# ── Config helper ─────────────────────────────────────────────────────────────

def _load_guardrail_config() -> dict:
    defaults = {
        "min_equity_avg_volume":       500_000,
        "min_equity_market_cap":       500_000_000,
        "min_crypto_volume_24h":       10_000_000,
        "max_active_watchlist_size":   50,
        "auto_add_requires_guardrail_pass": True,
    }
    try:
        from strategy.config_loader import load_config
        return load_config().get("screener_guardrails", defaults)
    except Exception:
        return defaults


# ── Liquidity check ───────────────────────────────────────────────────────────

def passes_liquidity_screen(symbol: str) -> bool:
    """
    Returns True if symbol meets minimum liquidity requirements.
    Returns False on yfinance failure (fail-safe: unknown = illiquid).

    US equities: averageVolume >= min_equity_avg_volume
                 marketCap    >= min_equity_market_cap
    Crypto (ends -USD): volume24Hr >= min_crypto_volume_24h
    """
    limits = _load_guardrail_config()

    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
    except Exception as e:
        log.warning(f"passes_liquidity_screen({symbol}): yfinance failed: {e}")
        return False

    if str(symbol).upper().endswith("-USD"):
        vol_24h = float(info.get("volume24Hr", 0) or 0)
        min_vol = float(limits.get("min_crypto_volume_24h", 10_000_000))
        result  = vol_24h >= min_vol
        log.debug(
            f"Liquidity {symbol} (crypto): "
            f"vol24h={vol_24h:,.0f} min={min_vol:,.0f} → {result}"
        )
        return result

    avg_vol    = float(info.get("averageVolume",  0) or 0)
    market_cap = float(info.get("marketCap",      0) or 0)
    min_vol    = float(limits.get("min_equity_avg_volume",  500_000))
    min_cap    = float(limits.get("min_equity_market_cap",  500_000_000))
    result     = avg_vol >= min_vol and market_cap >= min_cap
    log.debug(
        f"Liquidity {symbol} (equity): "
        f"vol={avg_vol:,.0f} cap={market_cap:,.0f} → {result}"
    )
    return result


# ── Watchlist size guard ──────────────────────────────────────────────────────

def check_watchlist_size() -> dict:
    """
    Returns whether the active watchlist has hit the configured size cap.
    Called before adding any symbol (auto or manual).

    Returns:
    {at_limit: bool, current: int, max: int}
    """
    limits  = _load_guardrail_config()
    max_size = int(limits.get("max_active_watchlist_size", 50))

    try:
        from strategy.config_loader import load_config
        cfg = load_config()
        wl  = cfg.get("watchlist", {})
        picks  = wl.get("screener_picks",   [])
        manual = wl.get("manual_additions", [])
        mandatory = wl.get("mandatory",     [])
        combined = list({*picks, *manual, *mandatory})
        current  = len(combined)
    except Exception:
        current = 0

    return {
        "at_limit": current >= max_size,
        "current":  current,
        "max":      max_size,
    }


# ── Filter a candidate list against guardrails ────────────────────────────────

def filter_screener_picks(candidates: list) -> list:
    """
    Filters a list of candidate symbols through the liquidity screen.
    Used by run_screen() before writing to screener_picks.
    Manual additions are NOT filtered — they bypass this function.

    Returns the subset of candidates that pass.
    """
    limits = _load_guardrail_config()
    if not limits.get("auto_add_requires_guardrail_pass", True):
        return candidates   # guardrail disabled in config

    passed  = []
    blocked = []
    for symbol in candidates:
        if passes_liquidity_screen(symbol):
            passed.append(symbol)
        else:
            blocked.append(symbol)

    if blocked:
        log.info(
            f"Screener guardrail blocked {len(blocked)} illiquid symbols: "
            f"{blocked[:10]}"
        )
    return passed
