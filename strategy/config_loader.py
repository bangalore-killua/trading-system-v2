"""
config_loader.py
Centralised config.json reader with typed accessors and safe defaults.
Never imports from any strategy module — no circular imports possible.
"""

import json
import os
import time as _time

# 60-second TTL cache — eliminates 60+ disk reads per 30-min cycle
_config_cache: dict = {"data": None, "expires": 0.0}


def load_config() -> dict:
    """Loads config.json with 60-second TTL cache. Never raises."""
    if (
        _time.time() < _config_cache["expires"]
        and _config_cache["data"] is not None
    ):
        return _config_cache["data"]
    path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        _config_cache["data"]    = data
        _config_cache["expires"] = _time.time() + 60.0
        return data
    except Exception:
        return _config_cache["data"] or {}


def get_signal_thresholds() -> dict:
    """Returns signal_thresholds from config with safe defaults."""
    defaults = {
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
    return {**defaults, **load_config().get("signal_thresholds", {})}


def get_signal_labels() -> dict:
    """Returns signal_labels from config with safe defaults."""
    defaults = {
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
    return {**defaults, **load_config().get("signal_labels", {})}


def get_execution_config() -> dict:
    """Returns execution settings from config with safe defaults."""
    defaults = {
        "max_single_order_notional":   15000,
        "gap_up_skip_pct":               3.0,
        "gap_down_skip_pct":            -3.0,
        "gap_up_reduce_pct":             2.5,
        "gap_reduce_size_mult":          0.5,
        "market_open_buffer_minutes":     30,
        "market_close_buffer_minutes":    30,
        "post_stop_reentry_pct":         0.95,
        "post_trail_reentry_pct":        0.97,
    }
    return {**defaults, **load_config().get("execution", {})}


def get_trailing_stop_config() -> dict:
    """Returns trailing stop settings from config with safe defaults."""
    defaults = {
        "activation_pnl_pct": 5.0,
        "stock_trail_pct":    8.0,
        "crypto_trail_pct":  12.0,
    }
    return {**defaults, **load_config().get("trailing_stop", {})}


def get_settings() -> dict:
    """Returns the top-level settings block with safe defaults."""
    defaults = {
        "max_positions":        10,
        "daily_loss_limit_pct": 0.03,
        "cash_reserve_pct":     20,
        "max_position_pct":     10,
    }
    return {**defaults, **load_config().get("settings", {})}


def get_factor_weights() -> dict:
    """Returns scoring factor weights from config with safe defaults."""
    defaults = {
        "rsi": 8, "vfi": 10, "macd": 8,
        "ema_alignment": 6, "bollinger": 6,
        "breakout": 7, "volume_surge": 6,
        "ath_proximity": 5, "adx_dmi": 4,
        "roe": 8, "roce": 8, "debt_equity": 6,
        "pe_ratio": 5, "pb_ratio": 4,
        "revenue_growth": 5, "earnings_surprise": 4,
        "market_regime": 8, "fear_greed": 5,
        "news_sentiment": 5, "sector_rs": 4,
        "insider_activity": 2,
        "geopolitical_risk": 6, "event_risk": 4,
        "analyst_ratings": 5,
    }
    return {**defaults, **load_config().get("factor_weights", {})}
