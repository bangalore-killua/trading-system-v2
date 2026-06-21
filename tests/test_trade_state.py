"""
Tests for TradeStateManager — cooldown logic and position tracking.

State files are redirected to temp directories so tests never
touch the real logs/trade_state.json or logs/open_positions.json.
"""

import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest


def _make_ts(tmp_path):
    """Return a TradeStateManager backed by files in tmp_path."""
    import strategy.trade_state as ts_mod

    orig_state = ts_mod._STATE_PATH
    orig_pos   = ts_mod._POS_PATH

    ts_mod._STATE_PATH = str(tmp_path / "trade_state.json")
    ts_mod._POS_PATH   = str(tmp_path / "open_positions.json")

    from strategy.trade_state import TradeStateManager
    ts = TradeStateManager()
    return ts, ts_mod, orig_state, orig_pos


def _restore(ts_mod, orig_state, orig_pos):
    ts_mod._STATE_PATH = orig_state
    ts_mod._POS_PATH   = orig_pos


class TestCanReenter:

    def test_fresh_symbol_can_enter(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            result = ts.can_reenter("AAPL", 150.0)
            assert result["can_enter"] is True
            assert "no_prior" in result["reason"]
        finally:
            _restore(mod, os_, op)

    def test_after_buy_can_enter(self, tmp_path):
        """A BUY trade leaves no cooldown — only SELL starts cooldown."""
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            ts.save_trade(
                symbol="MSFT", action="BUY",
                price=300.0, trade_id="T-001"
            )
            result = ts.can_reenter("MSFT", 300.0)
            assert result["can_enter"] is True
        finally:
            _restore(mod, os_, op)

    def test_cooldown_blocks_reentry(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            # sell sets a cooldown_until in the future
            future = (
                datetime.now() + timedelta(hours=24)
            ).strftime("%Y-%m-%d %H:%M:%S")
            ts._state["GOOGL"] = {
                "last_action":    "SELL",
                "cooldown_until": future,
                "last_price":     100.0,
            }
            ts._save()

            result = ts.can_reenter("GOOGL", 100.0)
            assert result["can_enter"] is False
            assert "cooldown" in result["reason"]
        finally:
            _restore(mod, os_, op)

    def test_expired_cooldown_allows_reentry(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            past = (
                datetime.now() - timedelta(hours=1)
            ).strftime("%Y-%m-%d %H:%M:%S")
            ts._state["TSLA"] = {
                "last_action":    "SELL",
                "cooldown_until": past,
                "last_price":     200.0,
            }
            ts._save()

            result = ts.can_reenter("TSLA", 200.0)
            assert result["can_enter"] is True
        finally:
            _restore(mod, os_, op)

    def test_no_cooldown_tier12(self, tmp_path):
        """cooldown_until=None means Tier 1/2 — no cooldown."""
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            ts._state["AAPL"] = {
                "last_action":    "SELL",
                "cooldown_until": None,
                "last_price":     170.0,
            }
            ts._save()

            result = ts.can_reenter("AAPL", 170.0)
            assert result["can_enter"] is True
            assert result["reason"] == "no_cooldown_tier12"
        finally:
            _restore(mod, os_, op)


class TestGetCooldownRemaining:

    def test_zero_when_no_state(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            assert ts.get_cooldown_remaining("UNKNOWN") == 0
        finally:
            _restore(mod, os_, op)

    def test_positive_when_in_cooldown(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            future = (
                datetime.now() + timedelta(hours=12)
            ).strftime("%Y-%m-%d %H:%M:%S")
            ts._state["BTC-USD"] = {
                "last_action":    "SELL",
                "cooldown_until": future,
            }
            ts._save()
            remaining = ts.get_cooldown_remaining("BTC-USD")
            assert remaining > 0
        finally:
            _restore(mod, os_, op)


class TestSaveTrade:

    def test_save_and_retrieve(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            ts.save_trade(
                symbol="SPY", action="BUY",
                price=450.0, trade_id="T-SPY-001"
            )
            state = ts._state.get("SPY", {})
            assert state.get("last_action") == "BUY"
            assert state.get("last_price") == pytest.approx(450.0)
        finally:
            _restore(mod, os_, op)

    def test_state_written_to_disk(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            ts.save_trade(
                symbol="QQQ", action="SELL",
                price=380.0, trade_id="T-QQQ-001"
            )
            with open(str(tmp_path / "trade_state.json")) as f:
                disk = json.load(f)
            assert "QQQ" in disk
        finally:
            _restore(mod, os_, op)


class TestGetSymbolsToMonitor:

    def test_empty_when_no_positions(self, tmp_path):
        ts, mod, os_, op = _make_ts(tmp_path)
        try:
            assert ts.get_symbols_to_monitor() == []
        finally:
            _restore(mod, os_, op)
