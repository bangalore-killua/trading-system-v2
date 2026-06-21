"""
conftest.py — pytest fixtures shared across all test modules.

Tests run against real logic but with mocked broker (no real orders)
and synthetic price data (no real API calls).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Config fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config():
    """Load the actual config.json from the repo root."""
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


# ── Synthetic price data ─────────────────────────────────────────────────────

@pytest.fixture
def ohlcv_trending_up():
    """250 bars of a steadily trending-up stock with normal volume."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.3)
    open_ = close - np.random.uniform(0, 1, n)
    high = close + np.random.uniform(0.5, 2, n)
    low = open_ - np.random.uniform(0.5, 2, n)
    volume = np.random.randint(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv_oversold():
    """250 bars with a sharp recent drop — RSI will be oversold at the end."""
    np.random.seed(7)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Gradual uptrend then crash in last 20 bars
    close_base = 100 + np.cumsum(np.random.randn(n) * 0.3 + 0.1)
    close_base[-20:] -= np.linspace(0, 30, 20)
    close = np.maximum(close_base, 5)
    open_ = close - np.random.uniform(0, 1, n)
    high = np.maximum(open_, close) + np.random.uniform(0.1, 1, n)
    low = np.minimum(open_, close) - np.random.uniform(0.1, 1, n)
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv_short():
    """Only 30 bars — too short for most indicators."""
    np.random.seed(1)
    n = 30
    dates = pd.date_range("2024-11-01", periods=n, freq="D")
    close = 50 + np.cumsum(np.random.randn(n))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


# ── Mock broker ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_trader():
    """AlpacaTrader with all methods mocked — no real network calls."""
    trader = MagicMock()
    trader.place_market_buy.return_value = MagicMock(
        order_id="test-order-123",
        symbol="AAPL",
        side="buy",
        qty=10,
        filled_qty=10.0,
        filled_avg_price=150.0,
        status="filled",
        is_filled=True,
        is_failed=False,
    )
    trader.place_market_sell.return_value = MagicMock(
        order_id="test-order-456",
        symbol="AAPL",
        side="sell",
        qty=10,
        filled_qty=10.0,
        filled_avg_price=160.0,
        status="filled",
        is_filled=True,
        is_failed=False,
    )
    mock_account = MagicMock()
    mock_account.portfolio_value = 100_000.0
    mock_account.cash = 60_000.0
    trader.get_account.return_value = mock_account
    trader.calculate_position_size.return_value = 10
    return trader


# ── Hot universe fixture ─────────────────────────────────────────────────────

@pytest.fixture
def hot_universe():
    from data.hot_universe import HotUniverse
    return HotUniverse()
