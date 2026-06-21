"""
signal_basket.py — SETUP-1
Decision layer that sits on top of the scorecard (LAYER 2).

LAYER 1 (scorecard) is unchanged — produces raw data.
LAYER 2 (this file) reads scorecard output and makes entry/exit decisions.

All signal weights and thresholds live in config.json.
No hardcoded weights anywhere in this file.

Public API:
  detect_primary_setup(data)         → str   setup type or "NONE"
  compute_entry_basket(data, setup)  → dict  {basket_score, signals_hit, threshold, qualifies}
  compute_exit_basket(data, setup)   → dict  {basket_score, signals_hit, threshold, qualifies}
  HARD_EXIT_CONDITIONS               → list  conditions that bypass basket entirely
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── Hard exits bypass the basket entirely ─────────────────────────────────────

HARD_EXIT_CONDITIONS = [
    "hard_stop_triggered",
    "kill_switch_active",
    "going_concern_8k",
    "trading_halt",
]

# ── Config helpers ────────────────────────────────────────────────────────────

def _load_basket_config() -> dict:
    try:
        from strategy.config_loader import load_config
        cfg = load_config()
        return {
            "entry_weights":       cfg.get("basket_signal_weights", {}),
            "exit_weights":        cfg.get("exit_basket_signal_weights", {}),
            "entry_thresholds":    cfg.get("basket_thresholds", {}),
            "exit_thresholds":     cfg.get("exit_basket_thresholds", {}),
        }
    except Exception:
        return {
            "entry_weights": {}, "exit_weights": {},
            "entry_thresholds": {}, "exit_thresholds": {},
        }


# ── Individual entry signal evaluators ───────────────────────────────────────

def _eval_entry_signal(name: str, data: dict) -> bool:
    """Returns True if the named entry signal is active, given data context."""
    rsi          = data.get("rsi", 50.0)
    vix          = data.get("vix", 20.0)
    score_pct    = data.get("score_pct", 0.0)
    fund_pct     = data.get("fundamental_score_pct", 0.0)
    regime       = data.get("regime_state", "NEUTRAL")
    stab         = data.get("stabilisation_score", 0)
    vol_ratio    = data.get("volume_ratio", 1.0)
    atr_pct      = data.get("atr_pct", 2.0)

    # Negative news presence
    has_major_neg   = data.get("major_negative_event_48h", False)
    has_struct_sec  = data.get("no_structural_sec", False)  # True = problem present
    neg_news_48h    = data.get("negative_news_48h", False)

    # Map of signal name → evaluation logic
    _evals = {
        "rsi_momentum_zone":    50 <= rsi <= 68,
        "rsi_oversold":         rsi < 35,
        "price_above_20ma":     data.get("price_above_20ma", False),
        "price_above_50ma":     data.get("price_above_50ma", False),
        "macd_bullish":         data.get("macd_bullish", False),
        "volume_confirming":    vol_ratio >= 1.2,
        "atr_not_extreme":      atr_pct < 3.0,
        "score_above_52pct":    score_pct >= 0.52,
        "score_above_60pct":    score_pct >= 0.60,
        "insider_buying_30d":   data.get("insider_buying_30d", False),
        "estimates_rising":     data.get("estimates_revised_up", False),
        "revenue_inflecting":   data.get("revenue_inflecting", False),
        "no_value_trap":        not (data.get("pe_ratio", 20) < 0 and fund_pct < 0.3),
        "strong_fundamentals":  fund_pct >= 0.60,
        "positive_news_6h":     data.get("positive_news_6h", False),
        "major_positive_event": data.get("major_positive_event_2h", False),
        "no_major_negative_48h":not has_major_neg,
        "no_structural_sec":    not has_struct_sec,
        "sector_positive_news": data.get("sector_positive_news", False),
        "regime_favourable":    regime in ("EARLY_BULL", "MID_BULL", "NEUTRAL"),
        "sector_inflow":        data.get("sector_inflow", False),
        "daily_trend_bullish":  data.get("daily_trend", "NEUTRAL") == "BULLISH",
        "vix_not_inverted":     vix < 30,
        "stabilisation_4of7":   stab >= 4,
        "stabilisation_6of7":   stab >= 6,
        "peers_not_singled_out":not data.get("sector_wide_negative", False),
        "no_negative_news_48h": not neg_news_48h,
    }
    return _evals.get(name, False)


# ── Individual exit signal evaluators ────────────────────────────────────────

def _eval_exit_signal(name: str, data: dict) -> bool:
    """Returns True if the named exit signal is active (bearish)."""
    rsi              = data.get("rsi", 50.0)
    regime           = data.get("regime_state", "NEUTRAL")
    regime_at_entry  = data.get("regime_at_entry", "MID_BULL")
    score_pct        = data.get("score_pct", 0.0)
    setup            = data.get("setup_type_at_entry", "SETUP_MOMENTUM")

    # Exit thresholds per setup
    _exit_thresholds = {
        "SETUP_MOMENTUM":    0.40,
        "SETUP_BOUNCE":      0.35,
        "SETUP_INFLECTION":  0.40,
        "SETUP_NEWS_DRIVEN": 0.30,
    }
    exit_score_threshold = _exit_thresholds.get(setup, 0.40)

    _evals = {
        "rsi_momentum_lost_2bars":    data.get("rsi_below_45_streak", 0) >= 2,
        "price_below_20ma_2bars":     data.get("price_below_20ma_streak", 0) >= 2,
        "volume_distribution":        data.get("volume_distribution", False),
        "macd_bearish_cross":         data.get("macd_bearish_cross", False),
        "score_below_exit_threshold": score_pct < exit_score_threshold,
        "revenue_accelerating_down":  data.get("revenue_declining_accel", False),
        "insider_selling_cluster":    data.get("insider_selling_count_30d", 0) >= 3,
        "estimates_cut_twice":        data.get("estimates_cut_count", 0) >= 2,
        "major_negative_event":       data.get("major_negative_event_48h", False),
        "negative_guidance":          data.get("negative_guidance", False),
        "ceo_departure":              data.get("ceo_departure", False),
        "regime_crash":               regime == "CRASH",
        "regime_deteriorated": (
            regime in ("EARLY_BEAR", "MID_BEAR", "CRASH") and
            regime_at_entry in ("EARLY_BULL", "MID_BULL", "LATE_BULL")
        ),
    }
    return _evals.get(name, False)


# ── Setup detection ───────────────────────────────────────────────────────────

def detect_primary_setup(data: dict) -> str:
    """
    Determines which setup type applies given current data context.
    Priority: NEWS_DRIVEN > INFLECTION > BOUNCE > MOMENTUM > NONE

    NONE is correct behaviour — not a failure.
    """
    rsi              = data.get("rsi", 50.0)
    score_pct        = data.get("score_pct", 0.0)
    stabilisation    = data.get("stabilisation_score", 0)
    signal_type      = data.get("signal_type", "")
    news_dir         = data.get("news_direction", "neutral")
    news_hours_ago   = data.get("news_event_hours_ago", 99.0)
    major_news       = data.get("recent_major_news", False)
    daily_trend      = data.get("daily_trend", "NEUTRAL")

    # Priority 1: NEWS_DRIVEN — major positive event within last 2 hours
    if major_news and news_hours_ago <= 2 and news_dir.upper() == "POSITIVE":
        return "SETUP_NEWS_DRIVEN"

    # Priority 2: INFLECTION — cyclical turn signal from scorer
    if signal_type == "CYCLICAL_INFLECTION":
        return "SETUP_INFLECTION"

    # Priority 3: BOUNCE — oversold + stabilisation
    if rsi < 35 and stabilisation >= 4:
        return "SETUP_BOUNCE"

    # Priority 4: MOMENTUM — above-threshold score + RSI in zone + bullish trend
    if score_pct >= 0.52 and 50 <= rsi <= 72 and daily_trend == "BULLISH":
        return "SETUP_MOMENTUM"

    return "NONE"


# ── Entry basket ──────────────────────────────────────────────────────────────

def compute_entry_basket(data: dict, setup_type: str) -> dict:
    """
    Evaluates all entry signals and returns basket result.
    All weights loaded from config.json — no hardcoded values.

    Returns:
    {
      basket_score:  int    total points accumulated
      max_possible:  int    maximum achievable given the data
      signals_hit:   list   signal names that evaluated True
      signals_miss:  list   signal names that evaluated False
      threshold:     int    minimum score to qualify
      qualifies:     bool
      setup_type:    str
    }
    """
    cfg       = _load_basket_config()
    weights   = cfg["entry_weights"]
    threshold = cfg["entry_thresholds"].get(setup_type, 50)

    basket_score = 0
    signals_hit  = []
    signals_miss = []

    for signal_name, weight in weights.items():
        if _eval_entry_signal(signal_name, data):
            basket_score += weight
            signals_hit.append(f"{signal_name}(+{weight})")
        else:
            signals_miss.append(signal_name)

    max_possible = sum(weights.values())
    qualifies    = basket_score >= threshold

    log.debug(
        f"Entry basket [{setup_type}]: {basket_score}/{threshold} "
        f"{'QUALIFIES' if qualifies else 'FAILS'} | "
        f"hits={len(signals_hit)}"
    )
    return {
        "basket_score":  basket_score,
        "max_possible":  max_possible,
        "signals_hit":   signals_hit,
        "signals_miss":  signals_miss,
        "threshold":     threshold,
        "qualifies":     qualifies,
        "setup_type":    setup_type,
    }


# ── Exit basket ───────────────────────────────────────────────────────────────

def compute_exit_basket(data: dict, setup_type: str) -> dict:
    """
    Evaluates exit signals. qualifies=True means EXIT conditions met.
    All weights loaded from config.json — no hardcoded values.
    """
    cfg       = _load_basket_config()
    weights   = cfg["exit_weights"]
    threshold = cfg["exit_thresholds"].get(setup_type, 45)

    basket_score = 0
    signals_hit  = []
    signals_miss = []

    for signal_name, weight in weights.items():
        if _eval_exit_signal(signal_name, data):
            basket_score += weight
            signals_hit.append(f"{signal_name}(+{weight})")
        else:
            signals_miss.append(signal_name)

    max_possible = sum(weights.values())
    qualifies    = basket_score >= threshold

    log.debug(
        f"Exit basket [{setup_type}]: {basket_score}/{threshold} "
        f"{'EXIT' if qualifies else 'HOLD'} | hits={len(signals_hit)}"
    )
    return {
        "basket_score":  basket_score,
        "max_possible":  max_possible,
        "signals_hit":   signals_hit,
        "signals_miss":  signals_miss,
        "threshold":     threshold,
        "qualifies":     qualifies,
        "setup_type":    setup_type,
    }


# ── Hard-exit check ───────────────────────────────────────────────────────────

def check_hard_exit(data: dict) -> Optional[str]:
    """
    Returns the triggered hard-exit condition name, or None.
    Hard exits bypass the basket entirely — immediate action.
    """
    for condition in HARD_EXIT_CONDITIONS:
        if data.get(condition, False):
            return condition
    return None
