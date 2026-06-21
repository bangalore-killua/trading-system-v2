"""Tests for strategy/screener_guardrails.py — SCREENER-1."""

from unittest.mock import patch, MagicMock

import pytest

from strategy.screener_guardrails import (
    passes_liquidity_screen,
    filter_screener_picks,
    check_watchlist_size,
)

# ── Config mock ───────────────────────────────────────────────────────────────

_GUARDRAILS = {
    "min_equity_avg_volume":       500_000,
    "min_equity_market_cap":       500_000_000,
    "min_crypto_volume_24h":       10_000_000,
    "max_active_watchlist_size":   50,
    "auto_add_requires_guardrail_pass": True,
}


def _mock_guardrails():
    return patch(
        "strategy.screener_guardrails._load_guardrail_config",
        return_value=_GUARDRAILS,
    )


def _mock_yf(info: dict):
    mock_ticker = MagicMock()
    mock_ticker.info = info
    return patch("yfinance.Ticker", return_value=mock_ticker)


# ── passes_liquidity_screen: equities ─────────────────────────────────────────

def test_liquid_equity_passes():
    """NVDA-like: high volume + large cap → passes."""
    info = {"averageVolume": 50_000_000, "marketCap": 3_000_000_000_000}
    with _mock_guardrails(), _mock_yf(info):
        assert passes_liquidity_screen("NVDA") is True


def test_illiquid_equity_fails_low_volume():
    """Low avg volume → fails even if market cap is ok."""
    info = {"averageVolume": 100_000, "marketCap": 1_000_000_000}
    with _mock_guardrails(), _mock_yf(info):
        assert passes_liquidity_screen("ILLIQ") is False


def test_illiquid_equity_fails_low_market_cap():
    """Micro-cap → fails even if volume is ok."""
    info = {"averageVolume": 1_000_000, "marketCap": 10_000_000}
    with _mock_guardrails(), _mock_yf(info):
        assert passes_liquidity_screen("MICRO") is False


def test_equity_fails_when_both_below_threshold():
    info = {"averageVolume": 50_000, "marketCap": 50_000_000}
    with _mock_guardrails(), _mock_yf(info):
        assert passes_liquidity_screen("TINY") is False


def test_yfinance_failure_returns_false():
    """Network error → fail-safe: unknown = illiquid, no exception."""
    with _mock_guardrails(), patch("yfinance.Ticker",
                                    side_effect=Exception("timeout")):
        result = passes_liquidity_screen("AAPL")
    assert result is False


# ── passes_liquidity_screen: crypto ──────────────────────────────────────────

def test_liquid_crypto_passes():
    """BTC-USD with high 24h volume → passes."""
    info = {"volume24Hr": 50_000_000_000}   # $50B/day
    with _mock_guardrails(), _mock_yf(info):
        assert passes_liquidity_screen("BTC-USD") is True


def test_illiquid_crypto_fails():
    """Tiny altcoin with low volume → fails."""
    info = {"volume24Hr": 500_000}   # $500K/day — far below 10M threshold
    with _mock_guardrails(), _mock_yf(info):
        assert passes_liquidity_screen("SHITCOIN-USD") is False


def test_crypto_at_exact_threshold_passes():
    """volume24Hr == min_crypto_volume_24h → passes (>= not >)."""
    info = {"volume24Hr": 10_000_000}
    with _mock_guardrails(), _mock_yf(info):
        assert passes_liquidity_screen("ETH-USD") is True


# ── filter_screener_picks ─────────────────────────────────────────────────────

def test_filter_removes_illiquid_symbols():
    """Illiquid symbols are removed from auto screener picks."""
    def _mock_passes(symbol):
        return symbol in ("NVDA", "AAPL")   # only these pass

    candidates = ["NVDA", "AAPL", "ILLIQ1", "ILLIQ2"]
    with _mock_guardrails(), \
         patch("strategy.screener_guardrails.passes_liquidity_screen",
               side_effect=_mock_passes):
        result = filter_screener_picks(candidates)

    assert "NVDA" in result
    assert "AAPL" in result
    assert "ILLIQ1" not in result
    assert "ILLIQ2" not in result


def test_filter_passes_all_liquid_symbols():
    candidates = ["NVDA", "MSFT", "GOOGL"]
    with _mock_guardrails(), \
         patch("strategy.screener_guardrails.passes_liquidity_screen",
               return_value=True):
        result = filter_screener_picks(candidates)
    assert result == candidates


def test_filter_empty_list():
    with _mock_guardrails():
        assert filter_screener_picks([]) == []


def test_filter_disabled_by_config():
    """When auto_add_requires_guardrail_pass=False, all candidates pass."""
    disabled_cfg = {**_GUARDRAILS, "auto_add_requires_guardrail_pass": False}
    candidates = ["NVDA", "ILLIQ1"]
    with patch("strategy.screener_guardrails._load_guardrail_config",
               return_value=disabled_cfg):
        result = filter_screener_picks(candidates)
    assert result == candidates   # no filtering applied


# ── check_watchlist_size ──────────────────────────────────────────────────────

def test_watchlist_size_below_limit():
    cfg = {"watchlist": {
        "screener_picks":   ["NVDA", "AAPL"],
        "manual_additions": ["MSFT"],
        "mandatory":        ["SPY"],
    }}
    with _mock_guardrails(), \
         patch("strategy.config_loader.load_config", return_value=cfg):
        result = check_watchlist_size()
    assert result["at_limit"] is False
    assert result["current"] == 4
    assert result["max"] == 50


def test_watchlist_size_at_limit():
    """Exactly at max → at_limit=True."""
    symbols = [f"SYM{i}" for i in range(50)]
    cfg = {"watchlist": {
        "screener_picks":   symbols,
        "manual_additions": [],
        "mandatory":        [],
    }}
    with _mock_guardrails(), \
         patch("strategy.config_loader.load_config", return_value=cfg):
        result = check_watchlist_size()
    assert result["at_limit"] is True
    assert result["current"] == 50


def test_watchlist_deduplicates_across_buckets():
    """Same symbol in multiple buckets counted once."""
    cfg = {"watchlist": {
        "screener_picks":   ["NVDA", "SPY"],
        "manual_additions": ["NVDA"],   # duplicate
        "mandatory":        ["SPY"],    # duplicate
    }}
    with _mock_guardrails(), \
         patch("strategy.config_loader.load_config", return_value=cfg):
        result = check_watchlist_size()
    assert result["current"] == 2   # NVDA + SPY only


# ── Manual additions bypass guardrail (spec requirement) ──────────────────────

def test_manual_additions_bypass_guardrail_in_screener():
    """
    add_manual_symbol() in fundamental_screener.py adds to manual_additions
    without calling filter_screener_picks — manual bypasses guardrail.
    """
    import inspect
    import strategy.fundamental_screener as screener_mod
    src = inspect.getsource(screener_mod.FundamentalScreener.add_manual_symbol)
    # filter_screener_picks must NOT be called in add_manual_symbol
    assert "filter_screener_picks" not in src, (
        "Manual additions must bypass the liquidity guardrail"
    )


def test_screener_picks_use_guardrail():
    """run_screen() must call filter_screener_picks on auto picks."""
    import inspect
    import strategy.fundamental_screener as screener_mod
    src = inspect.getsource(screener_mod.FundamentalScreener.run_screen)
    assert "filter_screener_picks" in src, (
        "run_screen must apply guardrail filter to auto screener picks"
    )
