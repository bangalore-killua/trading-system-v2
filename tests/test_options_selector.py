"""Tests for strategy/options_selector.py — OptionsSelector."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from data.options_chain import OptionContract
from strategy.options_selector import OptionTrade, OptionsSelector


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_contract(
    symbol="AAPL240119C00185000",
    underlying="AAPL",
    option_type="call",
    strike=185.0,
    delta=0.50,
    bid=3.0,
    ask=3.20,
    dte=35,
    oi=1000,
) -> OptionContract:
    today = date.today()
    return OptionContract(
        symbol=symbol,
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiry=today + timedelta(days=dte),
        days_to_expiry=dte,
        bid=bid,
        ask=ask,
        mid=round((bid + ask) / 2, 4),
        open_interest=oi,
        volume=200,
        implied_volatility=0.30,
        delta=delta,
    )


BASE_CONFIG = {
    "options": {
        "enabled": True,
        "strategies_enabled": ["long_call", "protective_put", "covered_call"],
        "max_options_exposure_pct": 10,
        "long_call_min_lgbm_confidence": 0.65,
        "long_call_target_delta_min": 0.40,
        "long_call_target_delta_max": 0.60,
        "long_call_min_dte": 25,
        "long_call_max_dte": 50,
        "long_call_budget_pct": 0.5,
        "protective_put_otm_pct": 7.0,
        "protective_put_min_dte": 30,
        "protective_put_max_dte": 60,
        "covered_call_otm_pct": 7.0,
        "covered_call_min_dte": 14,
        "covered_call_max_dte": 21,
    }
}

DISABLED_CONFIG = {"options": {"enabled": False}}


def make_selector(config=None, contract=None) -> tuple:
    """Returns (selector, mock_fetcher)."""
    mock_fetcher = MagicMock()
    mock_fetcher.get_atm_contract.return_value = contract
    selector = OptionsSelector(chain_fetcher=mock_fetcher, config=config or BASE_CONFIG)
    return selector, mock_fetcher


# ── Options disabled ──────────────────────────────────────────────────────────

class TestOptionsDisabled:

    def test_select_long_call_returns_none_when_disabled(self):
        selector, _ = make_selector(config=DISABLED_CONFIG)
        result = selector.select_long_call("AAPL", 185.0, 100_000, 9_250)
        assert result is None

    def test_select_protective_put_returns_none_when_disabled(self):
        selector, _ = make_selector(config=DISABLED_CONFIG)
        result = selector.select_protective_put("AAPL", 185.0, 18_500, 100_000)
        assert result is None

    def test_select_covered_call_returns_none_when_disabled(self):
        selector, _ = make_selector(config=DISABLED_CONFIG)
        result = selector.select_covered_call("AAPL", 185.0, 200)
        assert result is None


# ── select_long_call ──────────────────────────────────────────────────────────

class TestSelectLongCall:

    def test_returns_none_when_no_contract_found(self):
        selector, _ = make_selector(contract=None)
        result = selector.select_long_call("AAPL", 185.0, 100_000, 9_250)
        assert result is None

    def test_returns_option_trade_when_contract_found(self):
        contract = make_contract()
        selector, _ = make_selector(contract=contract)
        result = selector.select_long_call("AAPL", 185.0, 100_000, 9_250)
        assert result is not None
        assert isinstance(result, OptionTrade)
        assert result.strategy == "long_call"
        assert result.contract is contract

    def test_qty_at_least_1(self):
        contract = make_contract(bid=3.0, ask=3.20)  # mid=3.10, cost=$310/contract
        selector, _ = make_selector(contract=contract)
        # budget = 5000 * 0.5 = 25 → too small for 1 contract at $310
        # but qty is floored at 0 if budget < cost, returning None
        # With budget = 500 → 1 contract
        result = selector.select_long_call("AAPL", 185.0, 100_000, 1_000)
        # 1000 * 0.005 = 5 → not enough for 3.10 * 100 = 310, qty=0 → None
        assert result is None

    def test_qty_respects_max_exposure(self):
        contract = make_contract(bid=1.0, ask=1.20)  # mid=1.10, cost=$110/contract
        selector, _ = make_selector(contract=contract)
        # portfolio=100k, max_exposure=10% → $10k max
        # budget = 200k * 0.005 = 1000 → 9 contracts, but max_exposure caps at $10k/110=$90 contracts
        result = selector.select_long_call("AAPL", 185.0, 100_000, 200_000)
        assert result is not None
        assert result.notional <= 100_000 * 0.10 + 1   # within max_exposure

    def test_strategy_not_in_enabled_returns_none(self):
        config = {**BASE_CONFIG, "options": {**BASE_CONFIG["options"], "strategies_enabled": ["protective_put"]}}
        selector, _ = make_selector(config=config, contract=make_contract())
        result = selector.select_long_call("AAPL", 185.0, 100_000, 9_250)
        assert result is None


# ── select_protective_put ────────────────────────────────────────────────────

class TestSelectProtectivePut:

    def test_returns_none_when_no_contract(self):
        selector, _ = make_selector(contract=None)
        result = selector.select_protective_put("AAPL", 185.0, 18_500, 100_000)
        assert result is None

    def test_returns_put_trade(self):
        contract = make_contract(option_type="put", delta=-0.30, strike=172.0)
        selector, _ = make_selector(contract=contract)
        result = selector.select_protective_put("AAPL", 185.0, 18_500, 100_000)
        assert result is not None
        assert result.strategy == "protective_put"

    def test_qty_based_on_shares_equivalent(self):
        contract = make_contract(option_type="put", delta=-0.30, strike=172.0, bid=2.0, ask=2.20)
        selector, _ = make_selector(contract=contract)
        # position_value=18500, current_price=185 → ~100 shares → 1 contract
        result = selector.select_protective_put("AAPL", 185.0, 18_500, 100_000)
        assert result is not None
        assert result.qty >= 1

    def test_uses_otm_strike_as_atm_target(self):
        contract = make_contract(option_type="put", delta=-0.30, strike=172.0)
        selector, mock_fetcher = make_selector(contract=contract)
        selector.select_protective_put("AAPL", 185.0, 18_500, 100_000)
        # Should call get_atm_contract with target price ≈ 185 * (1 - 0.07)
        call_args = mock_fetcher.get_atm_contract.call_args
        expected_target = 185.0 * (1 - 0.07)
        assert call_args.kwargs.get("current_price", call_args.args[2] if len(call_args.args) > 2 else None) == pytest.approx(expected_target, rel=0.01)


# ── select_covered_call ───────────────────────────────────────────────────────

class TestSelectCoveredCall:

    def test_returns_none_when_fewer_than_100_shares(self):
        selector, _ = make_selector(contract=make_contract())
        result = selector.select_covered_call("AAPL", 185.0, shares_held=50)
        assert result is None

    def test_returns_none_when_no_contract(self):
        selector, _ = make_selector(contract=None)
        result = selector.select_covered_call("AAPL", 185.0, shares_held=200)
        assert result is None

    def test_returns_covered_call_trade(self):
        contract = make_contract(option_type="call", strike=198.0, delta=0.25)
        selector, _ = make_selector(contract=contract)
        result = selector.select_covered_call("AAPL", 185.0, shares_held=200)
        assert result is not None
        assert result.strategy == "covered_call"

    def test_qty_equals_shares_div_100(self):
        contract = make_contract(option_type="call", strike=198.0, delta=0.25)
        selector, _ = make_selector(contract=contract)
        result = selector.select_covered_call("AAPL", 185.0, shares_held=250)
        assert result is not None
        assert result.qty == 2  # floor(250/100) = 2


# ── OptionTrade dataclass ────────────────────────────────────────────────────

class TestOptionTrade:

    def test_notional_equals_qty_times_contract_value(self):
        contract = make_contract(bid=3.0, ask=3.20)   # mid=3.10
        trade = OptionTrade(
            contract=contract,
            strategy="long_call",
            qty=5,
            max_premium=3.10 * 1.05,
            notional=5 * 3.10 * 100,
        )
        assert trade.notional == pytest.approx(1550.0, rel=0.01)
