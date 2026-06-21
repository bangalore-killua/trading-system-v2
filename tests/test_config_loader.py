"""
Tests for strategy/config_loader.py.
Verifies TTL caching, new accessor functions, and settings defaults.
"""

import time
from unittest.mock import patch, mock_open
import pytest

import strategy.config_loader as cl


def _reset_cache():
    cl._config_cache["data"]    = None
    cl._config_cache["expires"] = 0.0


# ── TTL cache ─────────────────────────────────────────────────────────────────

def test_load_config_caches_result():
    """open() must only be called once when called twice within TTL."""
    _reset_cache()
    fake_json = '{"settings": {"max_positions": 10}}'
    with patch("builtins.open", mock_open(read_data=fake_json)) as m:
        cl.load_config()
        cl.load_config()   # second call — should hit cache
    assert m.call_count == 1


def test_cache_expires_after_ttl():
    """Expired cache must trigger a fresh disk read."""
    _reset_cache()
    fake_json = '{"settings": {"max_positions": 10}}'
    with patch("builtins.open", mock_open(read_data=fake_json)) as m:
        cl.load_config()
        # manually expire the cache
        cl._config_cache["expires"] = time.time() - 1
        cl.load_config()   # must re-read
    assert m.call_count == 2


def test_load_config_returns_dict_on_file_error():
    """load_config() must never raise — returns {} or last cached value."""
    _reset_cache()
    with patch("builtins.open", side_effect=FileNotFoundError):
        result = cl.load_config()
    assert isinstance(result, dict)


# ── get_factor_weights ────────────────────────────────────────────────────────

def test_get_factor_weights_returns_all_base_keys():
    """Must include all 22 base factor keys at minimum."""
    base_keys = [
        "rsi", "vfi", "macd", "ema_alignment", "bollinger",
        "breakout", "volume_surge", "ath_proximity", "adx_dmi",
        "roe", "roce", "debt_equity", "pe_ratio", "pb_ratio",
        "revenue_growth", "earnings_surprise",
        "market_regime", "fear_greed", "news_sentiment",
        "sector_rs", "insider_activity",
        "geopolitical_risk", "event_risk",
    ]
    with patch.object(cl, "load_config", return_value={}):
        weights = cl.get_factor_weights()
    for key in base_keys:
        assert key in weights, f"Missing weight key: {key}"


def test_get_factor_weights_all_values_numeric():
    with patch.object(cl, "load_config", return_value={}):
        weights = cl.get_factor_weights()
    for key, val in weights.items():
        assert isinstance(val, (int, float)), \
            f"Non-numeric weight for {key}: {val!r}"


def test_get_factor_weights_config_overrides_defaults():
    """A key in config.json factor_weights must override the default."""
    with patch.object(cl, "load_config",
                      return_value={"factor_weights": {"rsi": 99}}):
        weights = cl.get_factor_weights()
    assert weights["rsi"] == 99


# ── get_settings ──────────────────────────────────────────────────────────────

def test_get_settings_returns_initial_capital():
    with patch.object(cl, "load_config",
                      return_value={"settings": {"initial_capital_usd": 50000}}):
        settings = cl.get_settings()
    assert settings["initial_capital_usd"] == 50000


def test_get_settings_defaults_when_empty_config():
    with patch.object(cl, "load_config", return_value={}):
        settings = cl.get_settings()
    assert settings["max_positions"] == 10
    assert settings["daily_loss_limit_pct"] == pytest.approx(0.03)


# ── get_signal_thresholds ─────────────────────────────────────────────────────

def test_get_signal_thresholds_has_all_keys():
    required = [
        "buy_momentum_min_score", "buy_fundamental_min_score",
        "buy_weak_min_score", "watch_min_score",
        "hold_trend_min_score", "reduce_min_score",
        "exit_full_below_score", "trend_filter_hold_floor",
        "score_velocity_degrade_threshold", "tech_dominance_ratio",
    ]
    with patch.object(cl, "load_config", return_value={}):
        t = cl.get_signal_thresholds()
    for key in required:
        assert key in t, f"Missing threshold key: {key}"


# ── get_execution_config ──────────────────────────────────────────────────────

def test_get_execution_config_has_all_keys():
    required = [
        "max_single_order_notional", "gap_up_skip_pct",
        "gap_down_skip_pct", "market_open_buffer_minutes",
        "market_close_buffer_minutes",
    ]
    with patch.object(cl, "load_config", return_value={}):
        cfg = cl.get_execution_config()
    for key in required:
        assert key in cfg, f"Missing execution config key: {key}"
