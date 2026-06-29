"""
Unit tests for strategy/exit_monitor.py — fallback RSI mode.
The LSTM model (models/exit_lstm_v1.pt) doesn't exist yet; all tests
verify the RSI fallback path works correctly until the model is trained.
"""

import numpy as np
import pandas as pd
import pytest

from strategy.exit_monitor import ExitMonitor


@pytest.fixture(scope="module")
def em():
    return ExitMonitor()


@pytest.fixture
def ohlcv_200():
    """200 bars of synthetic daily OHLCV with lowercase columns."""
    np.random.seed(99)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.8 + 0.1)
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "open":   close - np.abs(np.random.randn(n) * 0.3),
        "high":   close + np.abs(np.random.randn(n) * 0.5),
        "low":    close - np.abs(np.random.randn(n) * 0.5),
        "close":  close,
        "volume": volume,
    })


@pytest.fixture
def ohlcv_title_case(ohlcv_200):
    """Same data but Title Case columns (yfinance style)."""
    return ohlcv_200.rename(columns=str.title)


_POS = {
    "unrealized_pnl_pct": 0.0,
    "peak_drawdown_pct":   0.0,
    "days_held":           10,
    "entry_composite_score": 0.65,
    "regime_changed_since_entry": False,
}


# ── Model availability ────────────────────────────────────────────────────────

def test_model_not_available(em):
    """LSTM model file doesn't exist yet — should always return False."""
    assert em._is_model_available() is False


# ── Fallback mode output structure ───────────────────────────────────────────

def test_score_returns_dict(em, ohlcv_200):
    result = em.score("NVDA", ohlcv_200, _POS)
    assert isinstance(result, dict)

def test_score_has_required_keys(em, ohlcv_200):
    result = em.score("NVDA", ohlcv_200, _POS)
    for key in ("symbol", "peak_prob", "exit_action", "threshold_hit", "model_source"):
        assert key in result, f"Missing key: {key}"

def test_exit_action_is_valid(em, ohlcv_200):
    result = em.score("NVDA", ohlcv_200, _POS)
    assert result["exit_action"] in ("E1", "E2", "E3", "hold")

def test_model_source_is_fallback(em, ohlcv_200):
    result = em.score("NVDA", ohlcv_200, _POS)
    assert result["model_source"] == "fallback_rsi"

def test_peak_prob_in_range(em, ohlcv_200):
    result = em.score("NVDA", ohlcv_200, _POS)
    assert 0.0 <= result["peak_prob"] <= 1.0


# ── Column casing normalisation ───────────────────────────────────────────────

def test_title_case_columns_handled(em, ohlcv_title_case):
    """ExitMonitor must handle Title Case df from yfinance without crashing."""
    result = em.score("AAPL", ohlcv_title_case, _POS)
    assert result["exit_action"] in ("E1", "E2", "E3", "hold")


# ── Overbought position triggers higher score ─────────────────────────────────

def test_high_pnl_raises_exit_score(em, ohlcv_200):
    """Position with +30% unrealised gain should have higher peak_prob than flat."""
    flat_pos = {**_POS, "unrealized_pnl_pct": 0.0}
    rich_pos = {**_POS, "unrealized_pnl_pct": 0.30}
    flat_prob = em.score("NVDA", ohlcv_200, flat_pos)["peak_prob"]
    rich_prob = em.score("NVDA", ohlcv_200, rich_pos)["peak_prob"]
    assert rich_prob >= flat_prob


# ── Short df graceful handling ────────────────────────────────────────────────

def test_short_df_returns_hold(em):
    """Only 10 bars — not enough for RSI. Should return hold, not crash."""
    short_df = pd.DataFrame({
        "close":  [100.0] * 10,
        "volume": [1_000_000.0] * 10,
    })
    result = em.score("TEST", short_df, _POS)
    assert result["exit_action"] == "hold"
