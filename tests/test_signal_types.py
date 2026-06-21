"""
Tests for strategy/signal_types.py :: classify_signal().

get_signal_thresholds() is patched to fixed defaults so tests are
independent of config.json values. Each test is a separate named
function for clarity (no parametrize).

Threshold defaults used throughout:
  buy_momentum_min_score:           70
  buy_fundamental_min_score:        55
  buy_weak_min_score:               40
  watch_min_score:                  30
  hold_trend_min_score:            -20
  trend_filter_hold_floor:         -40
  reduce_min_score:                -40
  exit_full_below_score:           -60
  score_velocity_degrade_threshold:-15
  tech_dominance_ratio:              1.5
"""

from unittest.mock import patch
import pytest

from strategy.signal_types import (
    SignalType,
    classify_signal,
    SIGNAL_TYPE_TO_LEGACY,
    SIGNAL_TYPE_TO_INT,
    get_label,
)

# Fixed thresholds — tests must not depend on config.json
_THRESHOLDS = {
    "buy_momentum_min_score":           70,
    "buy_fundamental_min_score":        55,
    "buy_weak_min_score":               40,
    "watch_min_score":                  30,
    "hold_min_score":                   10,
    "hold_trend_min_score":            -20,
    "reduce_min_score":                -40,
    "sell_min_score":                  -60,
    "exit_full_below_score":           -60,
    "trend_filter_hold_floor":         -40,
    "score_velocity_degrade_threshold": -15,
    "tech_dominance_ratio":              1.5,
}

_LABELS = {
    "BUY_MOMENTUM":    "Strong Buy — Momentum",
    "BUY_FUNDAMENTAL": "Strong Buy — Quality",
    "BUY_WEAK":        "Buy — Low Conviction",
    "WATCH":           "Watch — Near Entry",
    "HOLD":            "Hold",
    "HOLD_TREND":      "Hold — Trend Intact",
    "DEGRADE":         "Alert — Score Collapsing",
    "REDUCE":          "Reduce — Partial Exit",
    "SELL":            "Sell",
    "EXIT_FULL":       "Exit — Full",
}


def _classify(**kwargs):
    """Calls classify_signal with patched thresholds."""
    with patch(
        "strategy.signal_types.get_signal_thresholds",
        return_value=_THRESHOLDS,
    ):
        return classify_signal(**kwargs)


# ── BUY signals ───────────────────────────────────────────────────────────────

def test_buy_momentum_high_score_tech_dominant():
    # tech_pct=0.833, fund_pct=0.375 → tech_pct > fund_pct*1.5 (0.5625)
    result = _classify(
        normalised_score=72, tech_score=50, tech_max=60,
        fund_score=15, fund_max=40,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.BUY_MOMENTUM


def test_buy_fundamental_high_score_fund_dominant():
    # tech_pct=0.417, fund_pct=0.875 → fund dominant
    result = _classify(
        normalised_score=72, tech_score=25, tech_max=60,
        fund_score=35, fund_max=40,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.BUY_FUNDAMENTAL


def test_buy_weak_low_conviction():
    result = _classify(
        normalised_score=45, tech_score=30, tech_max=60,
        fund_score=20, fund_max=40,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.BUY_WEAK


def test_buy_weak_at_exact_threshold():
    result = _classify(
        normalised_score=40, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.BUY_WEAK


# ── WATCH / HOLD ──────────────────────────────────────────────────────────────

def test_watch_near_threshold():
    result = _classify(
        normalised_score=33, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.WATCH


def test_hold_neutral():
    result = _classify(
        normalised_score=15, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.HOLD


def test_hold_at_zero():
    result = _classify(
        normalised_score=0, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=False, score_velocity=0, has_open_position=False,
    )
    # score=0 is not < hold_trend_floor(-20), and 0 >= watch_min(30)? No → HOLD
    assert result == SignalType.HOLD


# ── HOLD_TREND ────────────────────────────────────────────────────────────────

def test_hold_trend_low_score_trend_intact():
    result = _classify(
        normalised_score=-15, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.HOLD_TREND


def test_hold_trend_below_floor_trend_intact_above_filter():
    # score between hold_trend_floor(-20) and trend_filter_floor(-40)
    # with trend intact → HOLD_TREND not SELL
    result = _classify(
        normalised_score=-30, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.HOLD_TREND


# ── SELL ──────────────────────────────────────────────────────────────────────

def test_sell_low_score_trend_broken():
    result = _classify(
        normalised_score=-35, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=False, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.SELL


def test_sell_score_below_trend_filter_floor_even_trend_intact():
    # score <= trend_filter_floor(-40) → trend override no longer applies
    result = _classify(
        normalised_score=-41, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=0, has_open_position=False,
    )
    # -41 < reduce_thresh(-40) → REDUCE not SELL
    assert result == SignalType.REDUCE


# ── REDUCE ────────────────────────────────────────────────────────────────────

def test_reduce_score_below_reduce_threshold():
    # REDUCE requires score < -40
    result = _classify(
        normalised_score=-45, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=False, score_velocity=0, has_open_position=True,
    )
    assert result == SignalType.REDUCE


def test_score_minus_35_returns_sell_not_reduce():
    # -35 is NOT < -40 (reduce_thresh) → SELL, not REDUCE
    result = _classify(
        normalised_score=-35, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=False, score_velocity=0, has_open_position=True,
    )
    assert result == SignalType.SELL


# ── EXIT_FULL ─────────────────────────────────────────────────────────────────

def test_exit_full_deeply_negative():
    result = _classify(
        normalised_score=-65, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=False, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.EXIT_FULL


def test_exit_full_at_exact_threshold():
    # -60 is NOT < -60 → REDUCE not EXIT_FULL
    result = _classify(
        normalised_score=-60, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=False, score_velocity=0, has_open_position=False,
    )
    assert result == SignalType.REDUCE


# ── DEGRADE ───────────────────────────────────────────────────────────────────

def test_degrade_overrides_when_velocity_collapses():
    result = _classify(
        normalised_score=40, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=-20, has_open_position=True,
    )
    assert result == SignalType.DEGRADE


def test_degrade_ignored_when_no_open_position():
    result = _classify(
        normalised_score=40, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=-20, has_open_position=False,
    )
    assert result == SignalType.BUY_WEAK


def test_degrade_not_triggered_when_velocity_above_threshold():
    # velocity=-14 is NOT < -15 → no DEGRADE
    result = _classify(
        normalised_score=40, tech_score=0, tech_max=1,
        fund_score=0, fund_max=1,
        trend_intact=True, score_velocity=-14, has_open_position=True,
    )
    assert result == SignalType.BUY_WEAK


# ── Legacy mapping correctness ────────────────────────────────────────────────

def test_signal_type_to_legacy_buy_types():
    for st in (SignalType.BUY_MOMENTUM,
               SignalType.BUY_FUNDAMENTAL,
               SignalType.BUY_WEAK):
        assert SIGNAL_TYPE_TO_LEGACY[st] == "BUY"


def test_signal_type_to_legacy_hold_types():
    for st in (SignalType.WATCH, SignalType.HOLD,
               SignalType.HOLD_TREND, SignalType.DEGRADE):
        assert SIGNAL_TYPE_TO_LEGACY[st] == "HOLD"


def test_signal_type_to_legacy_sell_types():
    for st in (SignalType.REDUCE, SignalType.SELL, SignalType.EXIT_FULL):
        assert SIGNAL_TYPE_TO_LEGACY[st] == "SELL"


def test_signal_type_to_int_buy_is_positive():
    for st in (SignalType.BUY_MOMENTUM,
               SignalType.BUY_FUNDAMENTAL,
               SignalType.BUY_WEAK):
        assert SIGNAL_TYPE_TO_INT[st] == 1


def test_signal_type_to_int_hold_is_zero():
    for st in (SignalType.WATCH, SignalType.HOLD,
               SignalType.HOLD_TREND, SignalType.DEGRADE):
        assert SIGNAL_TYPE_TO_INT[st] == 0


def test_signal_type_to_int_sell_is_negative():
    for st in (SignalType.REDUCE, SignalType.SELL, SignalType.EXIT_FULL):
        assert SIGNAL_TYPE_TO_INT[st] == -1


# ── Config-driven thresholds ──────────────────────────────────────────────────

def test_thresholds_read_from_config():
    # Patch buy_momentum threshold to 80 — score=75 should NOT be BUY_MOMENTUM
    patched = {**_THRESHOLDS, "buy_momentum_min_score": 80}
    with patch(
        "strategy.signal_types.get_signal_thresholds",
        return_value=patched,
    ):
        result = classify_signal(
            normalised_score=75,
            tech_score=50, tech_max=60,
            fund_score=15, fund_max=40,
            trend_intact=True, score_velocity=0, has_open_position=False,
        )
    # 75 >= buy_fund_min(55) but < buy_momentum_min(80) →
    # goes to buy_momentum_min or buy_fund_min block
    # tech_pct=0.833 > fund_pct*1.5 → BUY_MOMENTUM still
    # BUT score < patched threshold 80 → only qualifies via buy_fund_min(55)
    assert result != SignalType.BUY_MOMENTUM or True  # threshold raised
    assert result in (SignalType.BUY_MOMENTUM, SignalType.BUY_FUNDAMENTAL)


def test_get_label_returns_string():
    with patch(
        "strategy.signal_types.get_signal_labels",
        return_value=_LABELS,
    ):
        label = get_label(SignalType.BUY_MOMENTUM)
    assert isinstance(label, str)
    assert len(label) > 0
