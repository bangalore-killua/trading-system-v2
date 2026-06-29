"""
Data tests — verify config.json has the correct schema after each PR.
These act as a guardrail: any config key rename or removal breaks here first.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cfg():
    path = Path(__file__).parent.parent / "config.json"
    with open(path) as f:
        return json.load(f)


# ── Risk section ──────────────────────────────────────────────────────────────

def test_emergency_floor_exists(cfg):
    assert "emergency_cash_floor_pct" in cfg["risk"], \
        "emergency_cash_floor_pct missing from risk — was it renamed?"

def test_emergency_floor_is_numeric(cfg):
    v = cfg["risk"]["emergency_cash_floor_pct"]
    assert isinstance(v, (int, float))
    assert 1 <= v <= 20, f"emergency_cash_floor_pct={v} looks wrong (expect 1–20%)"

def test_cash_reserve_removed(cfg):
    assert "cash_reserve_pct" not in cfg["risk"], \
        "cash_reserve_pct still in config — should have been replaced by emergency_cash_floor_pct"

def test_max_portfolio_positions_removed(cfg):
    assert "max_portfolio_positions" not in cfg["risk"], \
        "max_portfolio_positions still in config — replaced by max_positions_safety_cap"

def test_safety_cap_exists(cfg):
    assert "max_positions_safety_cap" in cfg["risk"]
    assert cfg["risk"]["max_positions_safety_cap"] >= 20


# ── Position sizing section ───────────────────────────────────────────────────

def test_base_size_pct(cfg):
    assert cfg["position_sizing"]["base_size_pct"] == 5, \
        "base_size_pct should be 5 (was 2 — updated in PR #41)"

def test_min_position_usd_exists(cfg):
    assert "min_position_usd" in cfg["position_sizing"]
    assert cfg["position_sizing"]["min_position_usd"] >= 100

def test_news_size_multiplier_exists(cfg):
    assert "news_size_multiplier" in cfg["position_sizing"]
    assert cfg["position_sizing"]["news_size_multiplier"] > 1.0


# ── Backtest section ──────────────────────────────────────────────────────────

def test_backtest_section_exists(cfg):
    required = ["entry_thresh", "exit_thresh", "max_hold_days", "stop_loss_pct"]
    for key in required:
        assert key in cfg.get("backtest", {}), f"backtest.{key} missing"


# ── OOS / training overlap ────────────────────────────────────────────────────

def test_oos_symbols_not_in_training(cfg):
    oos = set(cfg.get("oos_test_watchlist", {}).get("symbols", []))
    training = set(cfg.get("watchlist", {}).get("symbols", []))
    overlap = oos & training
    assert not overlap, f"OOS symbols found in training watchlist: {overlap}"
