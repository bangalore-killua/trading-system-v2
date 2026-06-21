"""Tests for data/options_chain.py — OptionsChainFetcher and OptionContract."""

import time
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from data.options_chain import OptionContract, OptionsChainFetcher


# ── Fixtures ─────────────────────────────────────────────────────────────────

MINIMAL_CONFIG = {
    "options": {
        "long_call_min_dte": 25,
        "long_call_max_dte": 50,
        "min_contract_oi": 100,
        "max_contract_spread_pct": 0.20,
    }
}


def make_contract(
    symbol="AAPL240119C00185000",
    strike=185.0,
    delta=0.50,
    bid=3.0,
    ask=3.20,
    oi=1000,
    volume=200,
    dte=35,
) -> OptionContract:
    today = date.today()
    return OptionContract(
        symbol=symbol,
        underlying="AAPL",
        option_type="call",
        strike=strike,
        expiry=today + timedelta(days=dte),
        days_to_expiry=dte,
        bid=bid,
        ask=ask,
        mid=round((bid + ask) / 2, 4),
        open_interest=oi,
        volume=volume,
        implied_volatility=0.30,
        delta=delta,
    )


def make_fetcher(config=None) -> OptionsChainFetcher:
    return OptionsChainFetcher(
        api_key="test_key",
        secret_key="test_secret",
        config=config or MINIMAL_CONFIG,
    )


# ── OptionContract unit tests ─────────────────────────────────────────────────

class TestOptionContract:

    def test_mid_price(self):
        c = make_contract(bid=3.0, ask=3.40)
        assert c.mid == pytest.approx(3.20, abs=0.01)

    def test_spread_pct(self):
        c = make_contract(bid=3.0, ask=3.20)
        spread = (3.20 - 3.0) / 3.10
        assert c.spread_pct == pytest.approx(spread, rel=0.01)

    def test_spread_pct_zero_mid_returns_one(self):
        c = make_contract(bid=0, ask=0)
        assert c.spread_pct == 1.0

    def test_contract_value_is_mid_times_100(self):
        c = make_contract(bid=3.0, ask=3.20)
        assert c.contract_value == pytest.approx(c.mid * 100, rel=0.001)

    def test_negative_delta_allowed_for_puts(self):
        c = make_contract(delta=-0.30)
        assert c.delta == -0.30


# ── OptionsChainFetcher filter tests ─────────────────────────────────────────

class TestOptionsChainFetcherFilters:

    def test_filter_by_strike_removes_far_strikes(self):
        fetcher = make_fetcher()
        contracts = [
            make_contract(strike=100.0),
            make_contract(strike=185.0),
            make_contract(strike=300.0),
        ]
        result = fetcher._filter_by_strike(contracts, current_price=185.0, strike_range_pct=10.0)
        strikes = [c.strike for c in result]
        assert 185.0 in strikes
        assert 100.0 not in strikes
        assert 300.0 not in strikes

    def test_filter_by_strike_zero_price_returns_all(self):
        fetcher = make_fetcher()
        contracts = [make_contract(strike=100), make_contract(strike=500)]
        result = fetcher._filter_by_strike(contracts, current_price=0, strike_range_pct=10)
        assert len(result) == 2

    def test_filter_by_strike_exactly_at_boundary(self):
        fetcher = make_fetcher()
        contracts = [make_contract(strike=90.0), make_contract(strike=110.0)]
        # 10% range around 100: 90–110
        result = fetcher._filter_by_strike(contracts, current_price=100.0, strike_range_pct=10.0)
        assert len(result) == 2  # both at boundary, inclusive


# ── get_atm_contract selection logic tests ────────────────────────────────────

class TestGetAtmContract:

    def test_returns_none_when_chain_empty(self):
        fetcher = make_fetcher()
        with patch.object(fetcher, "get_chain", return_value=[]):
            result = fetcher.get_atm_contract("AAPL", "call", 185.0)
        assert result is None

    def test_selects_closest_delta(self):
        fetcher = make_fetcher()
        c1 = make_contract(symbol="C1", delta=0.30)
        c2 = make_contract(symbol="C2", delta=0.52)
        c3 = make_contract(symbol="C3", delta=0.70)
        with patch.object(fetcher, "get_chain", return_value=[c1, c2, c3]):
            result = fetcher.get_atm_contract("AAPL", "call", 185.0, target_delta=0.50)
        assert result is not None
        assert result.symbol == "C2"

    def test_excludes_wide_spread_contracts(self):
        fetcher = make_fetcher()
        # spread_pct = (5-1)/3 = 1.33 >> 0.20 threshold
        wide = make_contract(bid=1.0, ask=5.0)
        tight = make_contract(symbol="AAPL240119C00190000", strike=190.0, bid=3.0, ask=3.10, delta=0.48)
        with patch.object(fetcher, "get_chain", return_value=[wide, tight]):
            result = fetcher.get_atm_contract("AAPL", "call", 185.0)
        assert result is not None
        assert result.symbol == "AAPL240119C00190000"

    def test_excludes_low_oi_contracts(self):
        fetcher = make_fetcher()
        low_oi = make_contract(oi=10)
        high_oi = make_contract(symbol="AAPL240119C00190000", strike=190.0, oi=1000, delta=0.48)
        with patch.object(fetcher, "get_chain", return_value=[low_oi, high_oi]):
            result = fetcher.get_atm_contract("AAPL", "call", 185.0)
        assert result is not None
        assert result.symbol == "AAPL240119C00190000"

    def test_returns_none_when_all_contracts_filtered(self):
        fetcher = make_fetcher()
        no_delta = make_contract(delta=None)
        with patch.object(fetcher, "get_chain", return_value=[no_delta]):
            result = fetcher.get_atm_contract("AAPL", "call", 185.0)
        assert result is None


# ── Cache behaviour ───────────────────────────────────────────────────────────

class TestChainCache:

    def test_cache_stores_and_returns_result(self):
        fetcher = make_fetcher()
        contracts = [make_contract()]
        fetcher._cache[("AAPL", "call")] = (time.monotonic(), contracts)

        with patch.object(fetcher, "_fetch_from_alpaca") as mock_fetch:
            result = fetcher.get_chain("AAPL", "call", current_price=185.0)
            mock_fetch.assert_not_called()

        assert len(result) == 1

    def test_cache_miss_when_expired(self):
        fetcher = make_fetcher()
        contracts = [make_contract()]
        # Put an expired timestamp in the cache
        fetcher._cache[("AAPL", "call")] = (time.monotonic() - 400, contracts)

        with patch.object(fetcher, "_fetch_from_alpaca", return_value=[]) as mock_fetch:
            fetcher.get_chain("AAPL", "call", current_price=185.0)
            mock_fetch.assert_called_once()


# ── Alpaca import graceful degradation ───────────────────────────────────────

class TestGracefulDegradation:

    def test_returns_empty_list_when_alpaca_not_installed(self):
        fetcher = make_fetcher()
        with patch.dict("sys.modules", {"alpaca": None, "alpaca.data": None}):
            with patch.object(fetcher, "_fetch_from_alpaca", return_value=[]):
                result = fetcher.get_chain("AAPL", "call", 185.0)
        assert result == []
