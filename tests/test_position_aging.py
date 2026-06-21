"""Tests for strategy/position_aging.py — AGE-1."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from strategy.position_aging import (
    check_position_aging,
    get_partial_profit_action,
    _days_held,
)

# ── Config used in tests ──────────────────────────────────────────────────────

_REVIEW_DAYS = {"TIER_1": 90, "TIER_2": 60, "TIER_3": 30,
                "cyclical_inflection": 45}

_PARTIAL_THRESHOLDS = {
    "SETUP_MOMENTUM":    [[15, 0.33, 7],  [30, 0.50, 7],  [50, 0.75, 14]],
    "SETUP_BOUNCE":      [[10, 0.33, 2],  [20, 0.50, 2]],
    "SETUP_INFLECTION":  [[25, 0.25, 14], [50, 0.25, 30]],
    "SETUP_NEWS_DRIVEN": [[8,  0.50, 0]],
}


def _mock_config():
    cfg = {
        "thesis_review_days":       _REVIEW_DAYS,
        "partial_profit_thresholds":_PARTIAL_THRESHOLDS,
    }
    return patch("strategy.config_loader.load_config", return_value=cfg)


def _make_pos(entry_date: str):
    """Minimal position stub with entry_date attribute."""
    pos = MagicMock()
    pos.entry_date = entry_date
    return pos


def _now():
    return datetime(2026, 5, 10, 12, 0, 0)


def _entry(days_ago: int) -> str:
    return (_now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


# ── _days_held ────────────────────────────────────────────────────────────────

def test_days_held_correct():
    entry = _entry(45)
    assert _days_held(entry, now=_now()) == 45


def test_days_held_none_returns_zero():
    assert _days_held(None) == 0


def test_days_held_unparseable_returns_zero():
    assert _days_held("not-a-date") == 0


# ── check_position_aging: TIER_2 at 61 days ──────────────────────────────────

def test_tier2_at_61_days_requires_review():
    """TIER_2 threshold = 60 days → 61 days held must trigger review."""
    with _mock_config():
        result = check_position_aging(
            symbol="QCOM",
            position=_make_pos(_entry(61)),
            tier="TIER_2",
            entry_thesis="momentum",
            pnl_pct=10.0,
            setup_type="SETUP_MOMENTUM",
            now=_now(),
        )
    assert result["requires_review"] is True
    assert result["days_held"] == 61
    assert result["review_threshold_days"] == 60
    assert result["days_to_next_review"] == 0


def test_tier2_at_59_days_no_review():
    """59 days < 60 threshold → no review yet."""
    with _mock_config():
        result = check_position_aging(
            symbol="QCOM",
            position=_make_pos(_entry(59)),
            tier="TIER_2",
            entry_thesis="momentum",
            pnl_pct=5.0,
            setup_type="SETUP_MOMENTUM",
            now=_now(),
        )
    assert result["requires_review"] is False
    assert result["days_to_next_review"] == 1


def test_tier3_at_30_days_requires_review():
    with _mock_config():
        result = check_position_aging(
            symbol="XYZ",
            position=_make_pos(_entry(30)),
            tier="TIER_3",
            entry_thesis="momentum",
            pnl_pct=0.0,
            setup_type="SETUP_MOMENTUM",
            now=_now(),
        )
    assert result["requires_review"] is True
    assert result["review_threshold_days"] == 30


def test_tier1_at_89_days_no_review():
    with _mock_config():
        result = check_position_aging(
            symbol="AAPL",
            position=_make_pos(_entry(89)),
            tier="TIER_1",
            entry_thesis="momentum",
            pnl_pct=25.0,
            setup_type="SETUP_MOMENTUM",
            now=_now(),
        )
    assert result["requires_review"] is False
    assert result["days_to_next_review"] == 1


# ── cyclical_inflection review ────────────────────────────────────────────────

def test_cyclical_inflection_at_46_days_requires_review():
    """cyclical_inflection threshold = 45 days → 46 days must trigger."""
    with _mock_config():
        result = check_position_aging(
            symbol="QCOM",
            position=_make_pos(_entry(46)),
            tier="TIER_2",
            entry_thesis="cyclical_inflection",
            pnl_pct=8.0,
            setup_type="SETUP_INFLECTION",
            now=_now(),
        )
    assert result["requires_review"] is True
    assert result["review_threshold_days"] == 45


def test_cyclical_inflection_at_44_days_no_review():
    with _mock_config():
        result = check_position_aging(
            symbol="QCOM",
            position=_make_pos(_entry(44)),
            tier="TIER_2",
            entry_thesis="cyclical_inflection",
            pnl_pct=5.0,
            setup_type="SETUP_INFLECTION",
            now=_now(),
        )
    assert result["requires_review"] is False


# ── thesis_status ─────────────────────────────────────────────────────────────

def test_thesis_status_intact_before_review():
    with _mock_config():
        result = check_position_aging(
            symbol="X", position=_make_pos(_entry(10)),
            tier="TIER_3", entry_thesis="", pnl_pct=0.0,
            setup_type="SETUP_MOMENTUM", now=_now(),
        )
    assert result["thesis_status"] == "INTACT"


def test_thesis_status_review_due():
    with _mock_config():
        result = check_position_aging(
            symbol="X", position=_make_pos(_entry(31)),
            tier="TIER_3", entry_thesis="", pnl_pct=0.0,
            setup_type="SETUP_MOMENTUM", now=_now(),
        )
    assert result["thesis_status"] == "REVIEW_DUE"


def test_thesis_status_overdue_at_1_5x_threshold():
    """1.5× threshold (45 days for TIER_3 threshold of 30) → OVERDUE."""
    with _mock_config():
        result = check_position_aging(
            symbol="X", position=_make_pos(_entry(46)),
            tier="TIER_3", entry_thesis="", pnl_pct=0.0,
            setup_type="SETUP_MOMENTUM", now=_now(),
        )
    assert result["thesis_status"] == "OVERDUE"


# ── get_partial_profit_action ─────────────────────────────────────────────────

def test_momentum_first_partial_at_15pct_after_7days():
    with _mock_config():
        result = get_partial_profit_action("SETUP_MOMENTUM", pnl_pct=16.0,
                                           days_held=8)
    assert result is not None
    assert result["pnl_target_pct"] == 15
    assert result["sell_fraction"] == pytest.approx(0.33)
    assert result["action"] == "PARTIAL_PROFIT"


def test_momentum_no_partial_before_7days():
    """15% gain but only 5 days held → min_days=7 not met."""
    with _mock_config():
        result = get_partial_profit_action("SETUP_MOMENTUM", pnl_pct=16.0,
                                           days_held=5)
    assert result is None


def test_momentum_second_partial_at_30pct():
    with _mock_config():
        result = get_partial_profit_action("SETUP_MOMENTUM", pnl_pct=31.0,
                                           days_held=10)
    assert result is not None
    assert result["pnl_target_pct"] == 15   # returns FIRST matching, not highest


def test_bounce_first_partial_at_10pct_after_2days():
    with _mock_config():
        result = get_partial_profit_action("SETUP_BOUNCE", pnl_pct=11.0,
                                           days_held=2)
    assert result is not None
    assert result["pnl_target_pct"] == 10
    assert result["sell_fraction"] == pytest.approx(0.33)


def test_bounce_no_partial_below_10pct():
    with _mock_config():
        result = get_partial_profit_action("SETUP_BOUNCE", pnl_pct=8.0,
                                           days_held=5)
    assert result is None


def test_inflection_first_partial_at_25pct_after_14days():
    with _mock_config():
        result = get_partial_profit_action("SETUP_INFLECTION", pnl_pct=26.0,
                                           days_held=14)
    assert result is not None
    assert result["pnl_target_pct"] == 25
    assert result["sell_fraction"] == pytest.approx(0.25)


def test_news_driven_partial_at_8pct_day_0():
    """SETUP_NEWS_DRIVEN: partial fires same day (min_days=0)."""
    with _mock_config():
        result = get_partial_profit_action("SETUP_NEWS_DRIVEN", pnl_pct=9.0,
                                           days_held=0)
    assert result is not None
    assert result["pnl_target_pct"] == 8
    assert result["sell_fraction"] == pytest.approx(0.50)


def test_no_partial_for_unknown_setup_type():
    """Unknown setup type → no partial (safe default)."""
    with _mock_config():
        result = get_partial_profit_action("SETUP_UNKNOWN", pnl_pct=50.0,
                                           days_held=100)
    assert result is None


# ── check_position_aging: partial profit integration ─────────────────────────

def test_aging_returns_partial_action_when_pnl_meets_threshold():
    with _mock_config():
        result = check_position_aging(
            symbol="NVDA",
            position=_make_pos(_entry(10)),
            tier="TIER_2",
            entry_thesis="",
            pnl_pct=16.0,
            setup_type="SETUP_MOMENTUM",
            now=_now(),
        )
    assert result["partial_profit_action"] is not None
    assert result["partial_profit_action"]["action"] == "PARTIAL_PROFIT"


def test_aging_no_partial_when_pnl_below_threshold():
    with _mock_config():
        result = check_position_aging(
            symbol="NVDA",
            position=_make_pos(_entry(10)),
            tier="TIER_2",
            entry_thesis="",
            pnl_pct=5.0,
            setup_type="SETUP_MOMENTUM",
            now=_now(),
        )
    assert result["partial_profit_action"] is None


# ── Required output keys ──────────────────────────────────────────────────────

def test_aging_result_has_all_required_keys():
    with _mock_config():
        result = check_position_aging(
            symbol="TEST",
            position=_make_pos(_entry(20)),
            tier="TIER_3",
            entry_thesis="",
            pnl_pct=10.0,
            setup_type="SETUP_MOMENTUM",
            now=_now(),
        )
    required = [
        "symbol", "days_held", "requires_review",
        "review_threshold_days", "days_to_next_review",
        "partial_profit_action", "thesis_status",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"
