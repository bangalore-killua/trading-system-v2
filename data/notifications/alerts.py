"""
alerts.py — Alert type constants and formatting helpers.

Central registry for all alert types used across the trading system.
Import from here so alert names are consistent across all modules.
"""

# ── Alert levels ──────────────────────────────────────────────────────────────
INFO     = "INFO"
WARNING  = "WARNING"
ERROR    = "ERROR"
CRITICAL = "CRITICAL"
CIRCUIT  = "CIRCUIT"
TRADE    = "TRADE"
SYSTEM   = "SYSTEM"

# ── Circuit breaker reasons ───────────────────────────────────────────────────
CB_VIX_HIGH          = "VIX > 40 — extreme fear"
CB_DRAWDOWN          = "Portfolio drawdown > 12%"
CB_CONSECUTIVE_LOSS  = "5 consecutive losses"
CB_SECTOR_CONC       = "Sector concentration > 40%"
CB_SYMBOL_EXPOSURE   = "Combined symbol exposure > 22%"
CB_CASH_LOW          = "Cash reserve < 10%"
CB_CRASH_REGIME      = "Regime = CRASH"

# ── Trade actions ─────────────────────────────────────────────────────────────
ENTRY_T1   = "ENTRY T1 (33%)"
ENTRY_T2   = "ENTRY T2 (+33%)"
ENTRY_T3   = "ENTRY T3 (+34%)"
ENTRY_DIP  = "ENTRY DIP TRANCHE"
EXIT_E1    = "EXIT E1 (partial)"
EXIT_E2    = "EXIT E2 (partial)"
EXIT_E3    = "EXIT E3 (full)"
EXIT_STOP  = "EXIT STOP HIT"

# ── Source degradation keys (used as rate-limit keys in telegram.notify) ──────
SOURCE_UNUSUAL_WHALES    = "unusual_whales"
SOURCE_EARNINGS_WHISPERS = "earnings_whispers"
SOURCE_SEEKING_ALPHA     = "seeking_alpha"
SOURCE_RESEARCH_AGENT    = "research_agent"
SOURCE_TVDATAFEED        = "tvdatafeed"
SOURCE_ALPACA            = "alpaca_feed"
SOURCE_FRED              = "fred_macro"
SOURCE_STOCKTWITS        = "stocktwits"
SOURCE_PRNEWSWIRE        = "prnewswire"
SOURCE_BUSINESSWIRE      = "businesswire"


def fmt_pct(value: float) -> str:
    """Format a decimal as percentage string, e.g. 0.114 → '+11.4%'."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


def fmt_alpha(alpha: float) -> str:
    """Format alpha for trade notification, e.g. 0.083 → 'alpha=+8.3%'."""
    return f"alpha={fmt_pct(alpha)}"


def fmt_dollars(value: float) -> str:
    """Format large dollar amounts, e.g. 1_234_567 → '$1.23M'."""
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.2f}"
