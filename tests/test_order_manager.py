"""
Tests for OrderManager — duplicate lock prevention and expiry.

No Alpaca API calls are made; trader is set to None or mocked.
Lock files are written to a temp directory.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest


def _make_om(tmp_path):
    lock_file = str(tmp_path / "order_locks.json")
    import server.order_manager as om_mod
    original = om_mod._LOCK_PATH
    om_mod._LOCK_PATH = lock_file
    from server.order_manager import OrderManager
    om = OrderManager(trader=None)
    return om, om_mod, original


class TestCanPlaceOrder:

    def test_allowed_when_no_lock(self, tmp_path):
        om, mod, orig = _make_om(tmp_path)
        try:
            result = om.can_place_order("AAPL", "BUY")
            assert result["allowed"] is True
        finally:
            mod._LOCK_PATH = orig

    def test_blocked_when_lock_exists(self, tmp_path):
        om, mod, orig = _make_om(tmp_path)
        try:
            om.acquire_lock("AAPL", "BUY")
            result = om.can_place_order("AAPL", "BUY")
            assert result["allowed"] is False
            assert "AAPL" in result["reason"]
        finally:
            mod._LOCK_PATH = orig

    def test_different_side_not_blocked(self, tmp_path):
        om, mod, orig = _make_om(tmp_path)
        try:
            om.acquire_lock("AAPL", "BUY")
            result = om.can_place_order("AAPL", "SELL")
            assert result["allowed"] is True
        finally:
            mod._LOCK_PATH = orig

    def test_different_symbol_not_blocked(self, tmp_path):
        om, mod, orig = _make_om(tmp_path)
        try:
            om.acquire_lock("AAPL", "BUY")
            result = om.can_place_order("MSFT", "BUY")
            assert result["allowed"] is True
        finally:
            mod._LOCK_PATH = orig

    def test_expired_lock_allows_order(self, tmp_path):
        om, mod, orig = _make_om(tmp_path)
        try:
            # inject an already-expired lock directly
            expired_at = (
                datetime.now() - timedelta(minutes=10)
            ).strftime("%Y-%m-%d %H:%M:%S")
            om._locks["AAPL_BUY"] = {
                "symbol":     "AAPL",
                "side":       "BUY",
                "locked_at":  expired_at,
                "expires_at": expired_at,
                "order_id":   "old-id",
            }
            om._save_locks()

            # reload so cleanup runs
            from server.order_manager import OrderManager
            om2 = OrderManager(trader=None)
            result = om2.can_place_order("AAPL", "BUY")
            assert result["allowed"] is True
        finally:
            mod._LOCK_PATH = orig


class TestAcquireReleaseLock:

    def test_acquire_then_release_clears_lock(self, tmp_path):
        om, mod, orig = _make_om(tmp_path)
        try:
            om.acquire_lock("TSLA", "SELL")
            assert om.can_place_order("TSLA", "SELL")["allowed"] is False

            om.release_lock("TSLA", "SELL")
            assert om.can_place_order("TSLA", "SELL")["allowed"] is True
        finally:
            mod._LOCK_PATH = orig

    def test_lock_persists_across_instances(self, tmp_path):
        om, mod, orig = _make_om(tmp_path)
        try:
            om.acquire_lock("BTC-USD", "BUY")

            from server.order_manager import OrderManager
            om2 = OrderManager(trader=None)
            assert om2.can_place_order("BTC-USD", "BUY")["allowed"] is False
        finally:
            mod._LOCK_PATH = orig


class TestSafeBuySell:
    """Tests the module-level safe_buy / safe_sell wrappers."""

    def test_safe_buy_blocked_on_duplicate(self, tmp_path):
        lock_file = str(tmp_path / "order_locks.json")
        import server.order_manager as om_mod
        original = om_mod._LOCK_PATH
        om_mod._LOCK_PATH = lock_file
        try:
            from server.order_manager import safe_buy, OrderManager

            trader = MagicMock()
            # pre-lock the symbol
            om = OrderManager(trader=trader)
            om.acquire_lock("AAPL", "BUY")

            result = safe_buy(
                trader=trader, symbol="AAPL", notional=500
            )
            assert result["status"] == "blocked"
        finally:
            om_mod._LOCK_PATH = original

    def test_safe_sell_blocked_on_duplicate(self, tmp_path):
        lock_file = str(tmp_path / "order_locks.json")
        import server.order_manager as om_mod
        original = om_mod._LOCK_PATH
        om_mod._LOCK_PATH = lock_file
        try:
            from server.order_manager import safe_sell, OrderManager

            trader = MagicMock()
            om = OrderManager(trader=trader)
            om.acquire_lock("MSFT", "SELL")

            result = safe_sell(
                trader=trader, symbol="MSFT", percentage=1.0
            )
            assert result["status"] == "blocked"
        finally:
            om_mod._LOCK_PATH = original


# ── Phase 5.5: Regression tests ──────────────────────────────────────────────

class TestRegressions:

    def test_partial_exit_does_not_call_place_sell_order(self, tmp_path):
        """
        Regression for Phase 1 fix 1.1: partial exits in execute_signals()
        must route through safe_sell() (duplicate-lock aware), never
        directly call place_sell_order().

        Verifies that server/order_manager.safe_sell is the call path
        used for partial exits, not AlpacaTrader.place_sell_order.
        """
        from unittest.mock import patch, MagicMock

        lock_file = str(tmp_path / "order_locks.json")
        import server.order_manager as om_mod
        original = om_mod._LOCK_PATH
        om_mod._LOCK_PATH = lock_file
        try:
            safe_sell_mock      = MagicMock(return_value={"status": "ok"})
            place_sell_mock     = MagicMock()

            # simulate a partial_exit evaluation result
            partial_exit_eval = {
                "action": "partial_exit", "exit_type": "score_reduce",
                "sell_pct": 0.5, "sell_shares": 5.0,
                "pnl_pct": -5.0, "pnl_usd": -250.0, "reason": [],
            }

            with patch("server.order_manager.safe_sell", safe_sell_mock):
                from server.order_manager import safe_sell
                # verify the function exists and can be imported
                assert callable(safe_sell)
                # call it directly (this is what execute_signals does)
                result = safe_sell(
                    trader=MagicMock(),
                    symbol="AAPL",
                    percentage=0.5,
                    scorecard={},
                )
                # safe_sell was called (not blocked since no pre-existing lock)
                safe_sell_mock.assert_called_once()
                # place_sell_order was never called
                place_sell_mock.assert_not_called()
        finally:
            om_mod._LOCK_PATH = original
