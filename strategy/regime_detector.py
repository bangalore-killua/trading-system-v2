"""
regime_detector.py — REGIME-1
8-state market regime detector + time-of-day size modifiers.

Regime detection uses yfinance free data:
  SPY  — market proxy (SMA200 distance)
  ^VIX — implied volatility (current + 10-day trend)
  ^TNX — 10yr Treasury yield
  ^IRX — 3-month Treasury yield
  (yield spread = ^TNX - ^IRX)

All thresholds configurable in config.json["regime_config"].
Defaults embedded here only as fallback when config missing.
"""

import logging
import time
from datetime import datetime, time as dtime
from typing import Optional

import pytz

log = logging.getLogger(__name__)

# ── Default REGIME_CONFIG (overridden by config.json) ────────────────────────

REGIME_CONFIG: dict = {
    "EARLY_BULL": {"size_mult": 1.10, "entry_min_pct": 0.52},
    "MID_BULL":   {"size_mult": 1.00, "entry_min_pct": 0.52},
    "LATE_BULL":  {"size_mult": 0.70, "entry_min_pct": 0.72},
    "EUPHORIA":   {"size_mult": 0.40, "entry_min_pct": 0.80},
    "NEUTRAL":    {"size_mult": 0.90, "entry_min_pct": 0.52},
    "EARLY_BEAR": {"size_mult": 0.60, "entry_min_pct": 0.60},
    "MID_BEAR":   {"size_mult": 0.50, "entry_min_pct": 0.65},
    "CRASH":      {"size_mult": 0.20, "entry_min_pct": 0.80},
}

# Maps 8-state → legacy 3-state for backwards compat with check_market_regime()
_LEGACY_MAP: dict = {
    "EARLY_BULL": "bull",
    "MID_BULL":   "bull",
    "LATE_BULL":  "bull",
    "EUPHORIA":   "bull",
    "NEUTRAL":    "sideways",
    "EARLY_BEAR": "bear",
    "MID_BEAR":   "bear",
    "CRASH":      "bear",
}

# ── In-process cache (4-hour TTL matches scheduler) ──────────────────────────

_CACHE: dict = {"data": None, "expires": 0.0}
_CACHE_TTL   = 4 * 3600   # 4 hours

# ── Eastern timezone ──────────────────────────────────────────────────────────

_ET = pytz.timezone("America/New_York")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _fetch_close(ticker: str, period: str = "1y") -> Optional[object]:
    """Returns a pandas Series of close prices or None on failure."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False,
                         auto_adjust=True)
        if df.empty:
            return None
        col = "Close" if "Close" in df.columns else df.columns[0]
        return df[col].dropna()
    except Exception as e:
        log.debug(f"_fetch_close({ticker}): {e}")
        return None


def _get_spy_pct_above_200ma() -> float:
    """SPY % above(+) or below(-) its 200-day SMA. Returns 0.0 on failure."""
    try:
        closes = _fetch_close("SPY", period="1y")
        if closes is None or len(closes) < 200:
            return 0.0
        close  = float(closes.iloc[-1])
        sma200 = float(closes.rolling(200).mean().iloc[-1])
        return round((close - sma200) / sma200 * 100, 2)
    except Exception:
        return 0.0


def _get_vix_current() -> float:
    """Current VIX close. Returns 20.0 (neutral) on failure."""
    try:
        closes = _fetch_close("^VIX", period="3mo")
        if closes is None or closes.empty:
            return 20.0
        return float(closes.iloc[-1])
    except Exception:
        return 20.0


def _get_vix_10d_trend() -> str:
    """'rising' | 'falling' | 'stable' based on last 10 VIX closes."""
    try:
        closes = _fetch_close("^VIX", period="3mo")
        if closes is None or len(closes) < 11:
            return "stable"
        current = float(closes.iloc[-1])
        ten_ago = float(closes.iloc[-11])
        change  = (current - ten_ago) / max(ten_ago, 0.01) * 100
        if change >  10:
            return "rising"
        if change < -10:
            return "falling"
        return "stable"
    except Exception:
        return "stable"


def _get_10yr_minus_3mo() -> float:
    """Yield spread = 10yr(^TNX) minus 3mo(^IRX). Negative = inverted."""
    try:
        t10 = _fetch_close("^TNX", period="5d")
        t3m = _fetch_close("^IRX", period="5d")
        if t10 is None or t3m is None or t10.empty or t3m.empty:
            return 1.0   # neutral assumption
        return round(float(t10.iloc[-1]) - float(t3m.iloc[-1]), 3)
    except Exception:
        return 1.0


def _load_regime_config() -> dict:
    """Loads regime thresholds from config.json; falls back to module defaults."""
    try:
        from strategy.config_loader import load_config
        return load_config().get("regime_config", REGIME_CONFIG)
    except Exception:
        return REGIME_CONFIG


# ── 8-State regime detection ──────────────────────────────────────────────────

def detect_regime(force_refresh: bool = False) -> dict:
    """
    Detects one of 8 market states using free yfinance data.

    Returns:
    {
      state:          str   — one of the 8 states above
      regime:         str   — legacy compat: "bull"|"bear"|"sideways"
      size_mult:      float — position size multiplier from REGIME_CONFIG
      entry_min_pct:  float — minimum scorecard pct to open new position
      vix:            float
      vix_trend:      str
      spy_pct_vs_200: float
      yield_spread:   float
      block_new_entries: bool  — True only for EUPHORIA
      block_non_tier1:   bool  — True only for CRASH
      from_cache:     bool
    }
    """
    now = time.time()
    if not force_refresh and _CACHE["data"] and now < _CACHE["expires"]:
        cached = dict(_CACHE["data"])
        cached["from_cache"] = True
        return cached

    try:
        spy_pct      = _get_spy_pct_above_200ma()
        vix          = _get_vix_current()
        vix_trend    = _get_vix_10d_trend()
        yield_spread = _get_10yr_minus_3mo()

        # ── Decision tree (order matters) ────────────────────────────────────
        if vix > 40 and spy_pct < -10:
            state = "CRASH"
        elif vix < 11 and spy_pct > 20:
            state = "EUPHORIA"
        elif spy_pct < -5 and vix_trend == "rising":
            state = "MID_BEAR"
        elif spy_pct < 0 and vix > 20:
            state = "EARLY_BEAR"
        elif spy_pct > 15 and vix < 14 and yield_spread < 0.3:
            state = "LATE_BULL"
        elif spy_pct > 0 and vix_trend == "falling" and vix > 20:
            state = "EARLY_BULL"
        elif spy_pct > 5 and vix < 20:
            state = "MID_BULL"
        else:
            state = "NEUTRAL"

        cfg = _load_regime_config()
        rcfg = cfg.get(state, REGIME_CONFIG.get(state, {}))

        result = {
            "state":             state,
            "regime":            _LEGACY_MAP.get(state, "sideways"),
            "size_mult":         rcfg.get("size_mult",      0.90),
            "entry_min_pct":     rcfg.get("entry_min_pct",  0.52),
            "vix":               round(vix, 2),
            "vix_trend":         vix_trend,
            "spy_pct_vs_200":    spy_pct,
            "yield_spread":      yield_spread,
            "block_new_entries": state == "EUPHORIA",
            "block_non_tier1":   state == "CRASH",
            "from_cache":        False,
        }
        _CACHE["data"]    = result
        _CACHE["expires"] = now + _CACHE_TTL
        log.info(
            f"Regime: {state} | SPY vs 200MA: {spy_pct:+.1f}% | "
            f"VIX: {vix:.1f} ({vix_trend}) | Spread: {yield_spread:.2f}"
        )
        return result

    except Exception as e:
        log.warning(f"detect_regime failed: {e}")
        return {
            "state":             "NEUTRAL",
            "regime":            "sideways",
            "size_mult":         0.90,
            "entry_min_pct":     0.52,
            "vix":               20.0,
            "vix_trend":         "stable",
            "spy_pct_vs_200":    0.0,
            "yield_spread":      1.0,
            "block_new_entries": False,
            "block_non_tier1":   False,
            "from_cache":        False,
        }


def invalidate_regime_cache() -> None:
    """Forces next detect_regime() call to re-fetch from yfinance."""
    _CACHE["expires"] = 0.0
    _CACHE["data"]    = None


# ── Time-of-day size modifiers ────────────────────────────────────────────────

# US equity session windows in Eastern time (hour, minute) tuples
_US_SESSIONS = [
    (dtime(9,  30), dtime(10, 30), "OPENING",    0.70),
    (dtime(10, 30), dtime(12,  0), "LATE_MORN",  1.00),
    (dtime(12,  0), dtime(14,  0), "MIDDAY",     0.85),
    (dtime(14,  0), dtime(15, 30), "AFTERNOON",  1.00),
    (dtime(15, 30), dtime(16,  0), "CLOSING",    0.70),
]


def get_time_of_day_modifier(symbol: str,
                              is_crypto: bool,
                              now_et: Optional[datetime] = None) -> dict:
    """
    Returns position-size modifier based on current market session.

    US equities:
      OPENING  09:30-10:30 → 0.70 (false moves, volatile open)
      LATE_MORN 10:30-12:00 → 1.00
      MIDDAY   12:00-14:00 → 0.85 (low volume, drift)
      AFTERNOON 14:00-15:30 → 1.00 (most reliable)
      CLOSING  15:30-16:00 → 0.70 (rebalancing noise)
      CLOSED   outside hours → 0.00 (no new entries)

    Crypto (24/7):
      Weekend          → 0.80 (thin liquidity)
      Asian session    → 0.85 (00:00-09:00 ET Mon-Fri)
      Other            → 1.00

    Returns:
    {modifier: float, session: str, entry_allowed: bool}
    """
    if now_et is None:
        now_et = datetime.now(_ET)

    if is_crypto or str(symbol).upper().endswith("-USD"):
        weekday  = now_et.weekday()   # Mon=0 ... Sun=6
        hour     = now_et.hour
        if weekday >= 5:
            return {"modifier": 0.80, "session": "CRYPTO_WEEKEND",
                    "entry_allowed": True}
        if 0 <= hour < 9:
            return {"modifier": 0.85, "session": "CRYPTO_ASIAN",
                    "entry_allowed": True}
        return {"modifier": 1.00, "session": "CRYPTO_NORMAL",
                "entry_allowed": True}

    # US equity
    t = now_et.time()
    for start, end, name, mod in _US_SESSIONS:
        if start <= t < end:
            return {"modifier": mod, "session": name,
                    "entry_allowed": mod > 0.0}

    return {"modifier": 0.00, "session": "CLOSED", "entry_allowed": False}
