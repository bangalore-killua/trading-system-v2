"""Tests for strategy/regime_detector.py — REGIME-1."""

from datetime import datetime, time as dtime
from unittest.mock import patch

import pytz
import pytest

from strategy.regime_detector import (
    detect_regime,
    get_time_of_day_modifier,
    invalidate_regime_cache,
    REGIME_CONFIG,
    _LEGACY_MAP,
)

_ET = pytz.timezone("America/New_York")


# ── Helper: force detect_regime without network calls ────────────────────────

def _mock_regime_inputs(spy_pct: float, vix: float,
                         vix_trend: str, yield_spread: float):
    """Context manager that injects synthetic market data."""
    return (
        patch("strategy.regime_detector._get_spy_pct_above_200ma",
              return_value=spy_pct),
        patch("strategy.regime_detector._get_vix_current",
              return_value=vix),
        patch("strategy.regime_detector._get_vix_10d_trend",
              return_value=vix_trend),
        patch("strategy.regime_detector._get_10yr_minus_3mo",
              return_value=yield_spread),
    )


def _run_detect(spy_pct, vix, vix_trend="stable", yield_spread=1.0):
    invalidate_regime_cache()
    with _mock_regime_inputs(spy_pct, vix, vix_trend, yield_spread)[0], \
         _mock_regime_inputs(spy_pct, vix, vix_trend, yield_spread)[1], \
         _mock_regime_inputs(spy_pct, vix, vix_trend, yield_spread)[2], \
         _mock_regime_inputs(spy_pct, vix, vix_trend, yield_spread)[3]:
        return detect_regime(force_refresh=True)


# ── CRASH detection ───────────────────────────────────────────────────────────

def test_crash_detected_vix40_spy_minus10():
    result = _run_detect(spy_pct=-12.0, vix=42.0, vix_trend="rising")
    assert result["state"] == "CRASH"
    assert result["regime"] == "bear"
    assert result["size_mult"] == pytest.approx(0.20)
    assert result["block_non_tier1"] is True


def test_crash_not_triggered_without_both_conditions():
    """VIX > 40 alone (SPY ok) should NOT be CRASH."""
    result = _run_detect(spy_pct=5.0, vix=42.0, vix_trend="rising")
    assert result["state"] != "CRASH"


# ── EUPHORIA detection ────────────────────────────────────────────────────────

def test_euphoria_detected_low_vix_high_spy():
    result = _run_detect(spy_pct=22.0, vix=10.5, vix_trend="falling")
    assert result["state"] == "EUPHORIA"
    assert result["block_new_entries"] is True
    assert result["size_mult"] == pytest.approx(0.40)


def test_euphoria_not_triggered_without_both_conditions():
    """SPY > 20% alone (VIX 15) should NOT be EUPHORIA."""
    result = _run_detect(spy_pct=22.0, vix=15.0)
    assert result["state"] != "EUPHORIA"


# ── MID_BEAR detection ────────────────────────────────────────────────────────

def test_mid_bear_detected():
    result = _run_detect(spy_pct=-7.0, vix=25.0, vix_trend="rising")
    assert result["state"] == "MID_BEAR"
    assert result["regime"] == "bear"


# ── EARLY_BEAR detection ──────────────────────────────────────────────────────

def test_early_bear_detected():
    result = _run_detect(spy_pct=-2.0, vix=22.0, vix_trend="stable")
    assert result["state"] == "EARLY_BEAR"
    assert result["regime"] == "bear"


# ── MID_BULL detection (2023-2024 SPY-like conditions) ───────────────────────

def test_mid_bull_detected_normal_conditions():
    """2023-2024 SPY: ~10% above 200MA, VIX ~15."""
    result = _run_detect(spy_pct=10.0, vix=15.0, vix_trend="stable")
    assert result["state"] == "MID_BULL"
    assert result["regime"] == "bull"
    assert result["size_mult"] == pytest.approx(1.00)
    assert result["block_new_entries"] is False
    assert result["block_non_tier1"] is False


# ── LATE_BULL detection ───────────────────────────────────────────────────────

def test_late_bull_detected():
    result = _run_detect(spy_pct=18.0, vix=13.0, yield_spread=0.2)
    assert result["state"] == "LATE_BULL"
    assert result["size_mult"] == pytest.approx(0.70)


# ── EARLY_BULL detection ──────────────────────────────────────────────────────

def test_early_bull_detected():
    """SPY slightly above 200MA, VIX was high but now falling."""
    result = _run_detect(spy_pct=2.0, vix=22.0, vix_trend="falling")
    assert result["state"] == "EARLY_BULL"
    assert result["regime"] == "bull"
    assert result["size_mult"] == pytest.approx(1.10)


# ── NEUTRAL fallback ──────────────────────────────────────────────────────────

def test_neutral_when_no_clear_signal():
    result = _run_detect(spy_pct=3.0, vix=19.0, vix_trend="stable")
    assert result["state"] == "NEUTRAL"
    assert result["regime"] == "sideways"


# ── Cache behaviour ───────────────────────────────────────────────────────────

def test_regime_cache_returns_same_result():
    invalidate_regime_cache()
    with _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[0], \
         _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[1], \
         _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[2], \
         _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[3]:
        r1 = detect_regime(force_refresh=True)
    # second call should use cache (no mocks needed)
    r2 = detect_regime()
    assert r2["from_cache"] is True
    assert r2["state"] == r1["state"]


def test_invalidate_clears_cache():
    invalidate_regime_cache()
    with _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[0], \
         _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[1], \
         _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[2], \
         _mock_regime_inputs(10.0, 15.0, "stable", 1.0)[3]:
        detect_regime(force_refresh=True)
    invalidate_regime_cache()
    with _mock_regime_inputs(-8.0, 28.0, "rising", 0.5)[0], \
         _mock_regime_inputs(-8.0, 28.0, "rising", 0.5)[1], \
         _mock_regime_inputs(-8.0, 28.0, "rising", 0.5)[2], \
         _mock_regime_inputs(-8.0, 28.0, "rising", 0.5)[3]:
        r = detect_regime(force_refresh=True)
    assert r["state"] == "MID_BEAR"


# ── REGIME_CONFIG completeness ────────────────────────────────────────────────

def test_all_8_states_have_size_mult():
    states = ["EARLY_BULL", "MID_BULL", "LATE_BULL", "EUPHORIA",
              "NEUTRAL", "EARLY_BEAR", "MID_BEAR", "CRASH"]
    for s in states:
        assert s in REGIME_CONFIG
        assert "size_mult" in REGIME_CONFIG[s]
        assert "entry_min_pct" in REGIME_CONFIG[s]


def test_legacy_map_covers_all_states():
    for state in REGIME_CONFIG:
        assert state in _LEGACY_MAP
        assert _LEGACY_MAP[state] in ("bull", "bear", "sideways")


# ── Time-of-day modifiers: US equities ────────────────────────────────────────

def _et(h, m, weekday_offset=0):
    """Creates an ET-aware datetime for a weekday at h:m."""
    # Use a known Monday: 2024-01-08 (Monday)
    from datetime import timedelta
    base = datetime(2024, 1, 8, h, m, 0, tzinfo=_ET)
    return base + timedelta(days=weekday_offset)


def test_opening_session_modifier():
    result = get_time_of_day_modifier("AAPL", is_crypto=False,
                                      now_et=_et(9, 45))
    assert result["session"] == "OPENING"
    assert result["modifier"] == pytest.approx(0.70)
    assert result["entry_allowed"] is True


def test_late_morning_modifier():
    result = get_time_of_day_modifier("AAPL", is_crypto=False,
                                      now_et=_et(11, 0))
    assert result["session"] == "LATE_MORN"
    assert result["modifier"] == pytest.approx(1.00)


def test_midday_modifier():
    result = get_time_of_day_modifier("AAPL", is_crypto=False,
                                      now_et=_et(13, 0))
    assert result["session"] == "MIDDAY"
    assert result["modifier"] == pytest.approx(0.85)


def test_afternoon_modifier():
    result = get_time_of_day_modifier("AAPL", is_crypto=False,
                                      now_et=_et(14, 30))
    assert result["session"] == "AFTERNOON"
    assert result["modifier"] == pytest.approx(1.00)


def test_closing_session_modifier():
    result = get_time_of_day_modifier("AAPL", is_crypto=False,
                                      now_et=_et(15, 45))
    assert result["session"] == "CLOSING"
    assert result["modifier"] == pytest.approx(0.70)


def test_closed_market_returns_zero():
    """After 16:00 ET — market is closed, no new entries."""
    result = get_time_of_day_modifier("AAPL", is_crypto=False,
                                      now_et=_et(18, 0))
    assert result["session"] == "CLOSED"
    assert result["modifier"] == pytest.approx(0.00)
    assert result["entry_allowed"] is False


def test_pre_market_returns_closed():
    """Before 09:30 ET — no entries."""
    result = get_time_of_day_modifier("AAPL", is_crypto=False,
                                      now_et=_et(8, 0))
    assert result["session"] == "CLOSED"
    assert result["entry_allowed"] is False


# ── Time-of-day modifiers: Crypto ────────────────────────────────────────────

def test_crypto_weekend_modifier():
    saturday = _et(14, 0, weekday_offset=5)   # Monday + 5 = Saturday
    result = get_time_of_day_modifier("BTC-USD", is_crypto=True,
                                       now_et=saturday)
    assert result["session"] == "CRYPTO_WEEKEND"
    assert result["modifier"] == pytest.approx(0.80)
    assert result["entry_allowed"] is True


def test_crypto_asian_session_modifier():
    """Mon-Fri 00:00-09:00 ET is Asian session."""
    result = get_time_of_day_modifier("BTC-USD", is_crypto=True,
                                       now_et=_et(3, 0))
    assert result["session"] == "CRYPTO_ASIAN"
    assert result["modifier"] == pytest.approx(0.85)


def test_crypto_normal_session_modifier():
    result = get_time_of_day_modifier("BTC-USD", is_crypto=True,
                                       now_et=_et(14, 0))
    assert result["session"] == "CRYPTO_NORMAL"
    assert result["modifier"] == pytest.approx(1.00)


def test_crypto_detected_by_symbol_suffix():
    """Symbol ending in -USD should be treated as crypto even if is_crypto=False."""
    result = get_time_of_day_modifier("ETH-USD", is_crypto=False,
                                       now_et=_et(14, 0))
    assert result["session"].startswith("CRYPTO")


# ── Scheduler interval ────────────────────────────────────────────────────────

def test_scheduler_uses_60_minute_interval():
    """Verify the scheduler source registers main jobs at 60 minutes."""
    import inspect
    import server.scheduler as sched_mod
    src = inspect.getsource(sched_mod)
    # find the block that registers score_and_execute
    assert "every(60).minutes" in src, (
        "Main scoring cycle must be 60 minutes per REGIME-1"
    )
    assert 'every(30).minutes.do(\n        _guarded(_score_and_execute' not in src, (
        "score_and_execute still at 30 minutes — should be 60"
    )
