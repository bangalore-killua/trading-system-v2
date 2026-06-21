"""
signal_types.py
Fine-grained signal taxonomy for the trading system.

Replaces the coarse BUY / HOLD / SELL trichotomy with 10 named
signal types that carry conviction, exit urgency, and entry style
information. The legacy "signal" field (BUY/HOLD/SELL) is preserved
for backward compatibility via SIGNAL_TYPE_TO_LEGACY.

All classification thresholds are read from config.json at call time
via strategy.config_loader — no hardcoded numbers in this file.
"""

from enum import Enum
from strategy.config_loader import get_signal_thresholds, get_signal_labels


class SignalType(str, Enum):
    """Signal classification — 12 types including V12 additions."""
    BUY_MOMENTUM         = "BUY_MOMENTUM"
    BUY_FUNDAMENTAL      = "BUY_FUNDAMENTAL"
    BUY_WEAK             = "BUY_WEAK"
    WATCH                = "WATCH"
    HOLD                 = "HOLD"
    HOLD_TREND           = "HOLD_TREND"
    DEGRADE              = "DEGRADE"
    REDUCE               = "REDUCE"
    SELL                 = "SELL"
    EXIT_FULL            = "EXIT_FULL"
    # V12-6: cyclical bottom — revenue declining but rate slowing
    CYCLICAL_INFLECTION  = "CYCLICAL_INFLECTION"
    # V12-7: tactical exit — predicted dip, intend to rebuy lower
    TACTICAL_EXIT_PREDICTED_DIP = "TACTICAL_EXIT_PREDICTED_DIP"


# ── Legacy compatibility maps ─────────────────────────────────────────────────
# All existing code that checks signal == "BUY" / "SELL" / "HOLD" continues
# to work unchanged — add signal_type as a NEW field, never remove signal.

SIGNAL_TYPE_TO_LEGACY: dict[SignalType, str] = {
    SignalType.BUY_MOMENTUM:                  "BUY",
    SignalType.BUY_FUNDAMENTAL:               "BUY",
    SignalType.BUY_WEAK:                      "BUY",
    SignalType.CYCLICAL_INFLECTION:           "BUY",
    SignalType.WATCH:                         "HOLD",
    SignalType.HOLD:                          "HOLD",
    SignalType.HOLD_TREND:                    "HOLD",
    SignalType.DEGRADE:                       "HOLD",
    SignalType.REDUCE:                        "SELL",
    SignalType.SELL:                          "SELL",
    SignalType.EXIT_FULL:                     "SELL",
    SignalType.TACTICAL_EXIT_PREDICTED_DIP:   "SELL",
}

SIGNAL_TYPE_TO_INT: dict[SignalType, int] = {
    SignalType.BUY_MOMENTUM:                  1,
    SignalType.BUY_FUNDAMENTAL:               1,
    SignalType.BUY_WEAK:                      1,
    SignalType.CYCLICAL_INFLECTION:           1,
    SignalType.WATCH:                         0,
    SignalType.HOLD:                          0,
    SignalType.HOLD_TREND:                    0,
    SignalType.DEGRADE:                       0,
    SignalType.REDUCE:                        -1,
    SignalType.SELL:                          -1,
    SignalType.EXIT_FULL:                     -1,
    SignalType.TACTICAL_EXIT_PREDICTED_DIP:   -1,
}

SIGNAL_TYPE_EMOJI: dict[SignalType, str] = {
    SignalType.BUY_MOMENTUM:                  "🚀",
    SignalType.BUY_FUNDAMENTAL:               "🟢",
    SignalType.BUY_WEAK:                      "🟡",
    SignalType.CYCLICAL_INFLECTION:           "⚡",
    SignalType.WATCH:                         "👀",
    SignalType.HOLD:                          "⚪",
    SignalType.HOLD_TREND:                    "🔵",
    SignalType.DEGRADE:                       "⚠️",
    SignalType.REDUCE:                        "🟠",
    SignalType.SELL:                          "🔴",
    SignalType.EXIT_FULL:                     "💀",
    SignalType.TACTICAL_EXIT_PREDICTED_DIP:   "🎯",
}


# ── Classification ────────────────────────────────────────────────────────────

def classify_signal(
    normalised_score:  float,
    tech_score:        float,
    tech_max:          float,
    fund_score:        float,
    fund_max:          float,
    trend_intact:      bool,
    score_velocity:    float,
    has_open_position: bool,
) -> SignalType:
    """
    Maps a normalised score (-100 to +100) to a SignalType.

    All thresholds are read from config.json at call time so they can
    be tuned without a code change.

    Args:
        normalised_score:  weighted composite score (-100 to +100)
        tech_score:        raw points earned in the technical category
        tech_max:          maximum possible technical points
        fund_score:        raw points earned in the fundamental category
        fund_max:          maximum possible fundamental points
        trend_intact:      True when price is above SMA200 / EMA bullish
        score_velocity:    score delta vs previous cycle (negative = falling)
        has_open_position: True when there is a live position in this symbol

    Returns:
        SignalType enum value
    """
    t = get_signal_thresholds()

    buy_momentum_min   = float(t["buy_momentum_min_score"])
    buy_fund_min       = float(t["buy_fundamental_min_score"])
    buy_weak_min       = float(t["buy_weak_min_score"])
    watch_min          = float(t["watch_min_score"])
    hold_trend_floor   = float(t["hold_trend_min_score"])
    trend_filter_floor = float(t["trend_filter_hold_floor"])
    exit_full_thresh   = float(t["exit_full_below_score"])
    reduce_thresh      = float(t["reduce_min_score"])
    velocity_thresh    = float(t["score_velocity_degrade_threshold"])
    tech_ratio         = float(t["tech_dominance_ratio"])

    # DEGRADE — score collapsing fast while holding a position
    if has_open_position and score_velocity < velocity_thresh:
        return SignalType.DEGRADE

    # EXIT_FULL — deeply negative score
    if normalised_score < exit_full_thresh:
        return SignalType.EXIT_FULL

    # REDUCE — moderately negative
    if normalised_score < reduce_thresh:
        return SignalType.REDUCE

    # SELL / HOLD_TREND — below hold_trend_floor
    if normalised_score < hold_trend_floor:
        if trend_intact and normalised_score > trend_filter_floor:
            return SignalType.HOLD_TREND
        return SignalType.SELL

    # HOLD_TREND — low-negative score but trend structure still intact
    if hold_trend_floor <= normalised_score < 0 and trend_intact:
        return SignalType.HOLD_TREND

    # BUY range — distinguish by which factor category dominates
    if normalised_score >= buy_momentum_min or normalised_score >= buy_fund_min:
        tech_pct = tech_score / tech_max if tech_max > 0 else 0.0
        fund_pct = fund_score / fund_max if fund_max > 0 else 0.0
        if tech_pct > fund_pct * tech_ratio:
            return SignalType.BUY_MOMENTUM
        return SignalType.BUY_FUNDAMENTAL

    if normalised_score >= buy_weak_min:
        return SignalType.BUY_WEAK

    if normalised_score >= watch_min:
        return SignalType.WATCH

    return SignalType.HOLD


def get_label(signal_type: SignalType) -> str:
    """Returns the human-readable label from config.json signal_labels."""
    return get_signal_labels().get(signal_type.value, signal_type.value)
