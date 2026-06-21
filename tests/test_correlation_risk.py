"""Tests for strategy/correlation_risk.py — CORR-1."""

from unittest.mock import patch

import pytest

from strategy.correlation_risk import get_sector_concentration, get_size_penalty

# ── Shared config ─────────────────────────────────────────────────────────────

_CFG = {
    "max_same":       3,
    "size_penalties": {"1": 1.00, "2": 0.80, "3": 0.60},
    "sector_assignments": {
        "NVDA": "semiconductor",
        "AMD":  "semiconductor",
        "INTC": "semiconductor",
        "MU":   "semiconductor",
        "AAPL": "big_tech",
        "MSFT": "big_tech",
        "BTC-USD": "crypto_ecosystem",
    },
}


def _mock_cfg():
    return patch(
        "strategy.correlation_risk._load_corr_config",
        return_value=_CFG,
    )


# ── get_sector_concentration ──────────────────────────────────────────────────

def test_three_existing_semiconductor_blocks_fourth():
    """3 existing semiconductor positions → entry blocked for 4th."""
    open_positions = {"NVDA": True, "AMD": True, "INTC": True}
    with _mock_cfg():
        result = get_sector_concentration("MU", open_positions)
    assert result["entry_blocked"] is True
    assert result["existing_count"] == 3
    assert result["sector"] == "semiconductor"
    assert "MU" not in result["existing_same"]


def test_two_existing_semiconductor_allows_third():
    """2 existing same-sector → entry allowed (3rd is the max)."""
    open_positions = {"NVDA": True, "AMD": True}
    with _mock_cfg():
        result = get_sector_concentration("INTC", open_positions)
    assert result["entry_blocked"] is False
    assert result["existing_count"] == 2


def test_two_existing_applies_0_80_size_penalty():
    """2 same-sector existing positions → 0.80 size modifier."""
    open_positions = {"NVDA": True, "AMD": True}
    with _mock_cfg():
        result = get_sector_concentration("INTC", open_positions)
    assert result["size_penalty"] == pytest.approx(0.80)


def test_one_existing_applies_1_00_size_penalty():
    """1 same-sector existing position → no size penalty (1.00)."""
    open_positions = {"NVDA": True}
    with _mock_cfg():
        result = get_sector_concentration("AMD", open_positions)
    assert result["size_penalty"] == pytest.approx(1.00)
    assert result["entry_blocked"] is False


def test_zero_existing_same_sector():
    """First position in a sector → no existing, full size, not blocked."""
    open_positions = {"AAPL": True, "MSFT": True}   # big_tech only
    with _mock_cfg():
        result = get_sector_concentration("NVDA", open_positions)
    assert result["entry_blocked"] is False
    assert result["existing_count"] == 0
    assert result["existing_same"] == []


def test_different_sector_not_counted():
    """Positions in other sectors don't affect concentration check."""
    open_positions = {"AAPL": True, "MSFT": True, "BTC-USD": True}
    with _mock_cfg():
        result = get_sector_concentration("NVDA", open_positions)
    # NVDA is semiconductor; AAPL/MSFT are big_tech; BTC is crypto
    assert result["existing_count"] == 0
    assert result["entry_blocked"] is False


def test_symbol_not_counted_against_itself():
    """The candidate symbol itself must not appear in existing_same."""
    open_positions = {"NVDA": True, "AMD": True, "INTC": True}
    with _mock_cfg():
        result = get_sector_concentration("NVDA", open_positions)
    # NVDA is in open_positions but should not count against itself
    assert "NVDA" not in result["existing_same"]
    # AMD + INTC = 2 (not 3)
    assert result["existing_count"] == 2
    assert result["entry_blocked"] is False


def test_unknown_symbol_sector_is_unknown():
    """Symbol not in sector_assignments → sector='unknown', never blocked."""
    open_positions = {"NVDA": True}
    with _mock_cfg():
        result = get_sector_concentration("UNKNOWN_SYM", open_positions)
    assert result["sector"] == "unknown"
    # Only other 'unknown' sector symbols would count — none here
    assert result["entry_blocked"] is False


def test_uses_config_sector_assignments_not_hardcoded():
    """Result must change when sector_assignments config changes."""
    # Put NVDA in big_tech in this variant
    alt_cfg = {**_CFG, "sector_assignments": {**_CFG["sector_assignments"],
                                               "NVDA": "big_tech"}}
    open_positions = {"AAPL": True, "MSFT": True}  # 2 big_tech

    with patch("strategy.correlation_risk._load_corr_config",
               return_value=alt_cfg):
        result = get_sector_concentration("NVDA", open_positions)

    # Now NVDA is big_tech → should see 2 existing big_tech
    assert result["sector"] == "big_tech"
    assert result["existing_count"] == 2


# ── get_size_penalty ──────────────────────────────────────────────────────────

def test_size_penalty_0_existing():
    with _mock_cfg():
        # 0 existing → key "0" not in penalties → defaults to 1.00
        # The function clamps to max key (3), but 0 < 1 so it returns penalties["1"]
        # Actually we want 1.00 for 0 existing — check the implementation
        penalty = get_size_penalty(0)
    # 0 existing = first time in this sector → full size
    assert penalty == pytest.approx(1.00)


def test_size_penalty_1_existing():
    with _mock_cfg():
        penalty = get_size_penalty(1)
    assert penalty == pytest.approx(1.00)


def test_size_penalty_2_existing():
    with _mock_cfg():
        penalty = get_size_penalty(2)
    assert penalty == pytest.approx(0.80)


def test_size_penalty_3_existing():
    with _mock_cfg():
        penalty = get_size_penalty(3)
    assert penalty == pytest.approx(0.60)


def test_size_penalty_clamped_at_max():
    """More than max positions → clamped to lowest penalty, not error."""
    with _mock_cfg():
        penalty = get_size_penalty(10)
    assert penalty == pytest.approx(0.60)   # clamped at key "3"


# ── Integration: evaluate_entry blocked by concentration ─────────────────────

def test_evaluate_entry_blocked_by_sector_concentration():
    """
    evaluate_entry() must return should_enter=False when 3 same-sector
    positions already exist.
    """
    from strategy.position_manager import PositionManager
    from unittest.mock import MagicMock

    mock_trade_state = MagicMock()
    mock_trade_state.can_reenter.return_value = {
        "can_enter": True, "reason": "ok", "last_trade": None
    }
    mock_trade_state.get_symbols_to_monitor.return_value = [
        "NVDA", "AMD", "INTC"
    ]

    conc_result = {
        "entry_blocked":  True,
        "existing_count": 3,
        "existing_same":  ["NVDA", "AMD", "INTC"],
        "sector":         "semiconductor",
        "max_allowed":    3,
        "size_penalty":   0.60,
        "symbol":         "MU",
    }

    with patch("strategy.position_manager.fetcher") as mock_fetcher, \
         patch("strategy.stock_profiles.check_binary_event",
               return_value={"binary_pending": False, "event": None}), \
         patch("strategy.correlation_risk.get_sector_concentration",
               return_value=conc_result), \
         patch("strategy.trade_state.TradeStateManager",
               return_value=mock_trade_state):
        mock_fetcher.get_current_price.return_value = {"price": 100.0}
        pm = PositionManager()
        result = pm.evaluate_entry("MU", scorecard={"signal": "BUY"})

    assert result["should_enter"] is False
    assert result.get("entry_type") == "sector_concentration_block"
