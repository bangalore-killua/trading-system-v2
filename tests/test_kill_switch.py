"""
Tests for KillSwitch and DailyLossGuard.

All file I/O is redirected to a temp directory so tests never
touch the real logs/kill_switch.json.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────

def _make_ks(tmp_path):
    """Return a KillSwitch whose state file lives in tmp_path."""
    state_file = os.path.join(tmp_path, "kill_switch.json")
    with patch("server.kill_switch._STATE_PATH", state_file):
        from server.kill_switch import KillSwitch
        ks = KillSwitch.__new__(KillSwitch)
        # re-run __init__ with patched path
        import importlib
        import server.kill_switch as ks_mod
        ks_mod._STATE_PATH = state_file
        ks.__init__()
        return ks, ks_mod, state_file


# ── KillSwitch tests ──────────────────────────────────────────

class TestKillSwitch:

    def test_inactive_by_default(self, tmp_path):
        state_file = str(tmp_path / "kill_switch.json")
        import server.kill_switch as ks_mod
        original = ks_mod._STATE_PATH
        try:
            ks_mod._STATE_PATH = state_file
            from server.kill_switch import KillSwitch
            ks = KillSwitch()
            assert ks.is_active() is False
        finally:
            ks_mod._STATE_PATH = original

    def test_activate_sets_active_true(self, tmp_path):
        state_file = str(tmp_path / "kill_switch.json")
        import server.kill_switch as ks_mod
        original = ks_mod._STATE_PATH
        try:
            ks_mod._STATE_PATH = state_file
            from server.kill_switch import KillSwitch
            ks = KillSwitch()
            with patch.object(ks, "_close_all_positions",
                              return_value=([], [])), \
                 patch("server.telegram_alerts.send_message",
                       side_effect=Exception("no telegram")):
                ks.activate(reason="test", close_positions=True)
            assert ks.is_active() is True
        finally:
            ks_mod._STATE_PATH = original

    def test_deactivate_sets_active_false(self, tmp_path):
        state_file = str(tmp_path / "kill_switch.json")
        import server.kill_switch as ks_mod
        original = ks_mod._STATE_PATH
        try:
            ks_mod._STATE_PATH = state_file
            from server.kill_switch import KillSwitch
            ks = KillSwitch()
            # force active
            ks._state["active"] = True
            ks._save_state()
            with patch("server.telegram_alerts.send_message",
                       side_effect=Exception("no telegram")):
                ks.deactivate(reason="test")
            assert ks.is_active() is False
        finally:
            ks_mod._STATE_PATH = original

    def test_state_persists_to_disk(self, tmp_path):
        state_file = str(tmp_path / "kill_switch.json")
        import server.kill_switch as ks_mod
        original = ks_mod._STATE_PATH
        try:
            ks_mod._STATE_PATH = state_file
            from server.kill_switch import KillSwitch
            ks = KillSwitch()
            with patch.object(ks, "_close_all_positions",
                              return_value=([], [])), \
                 patch("server.telegram_alerts.send_message",
                       side_effect=Exception("no telegram")):
                ks.activate(reason="persist_test",
                            close_positions=False)

            # reload from disk — should still be active
            ks2 = KillSwitch()
            assert ks2.is_active() is True
        finally:
            ks_mod._STATE_PATH = original

    def test_check_kill_switch_module_fn(self, tmp_path):
        state_file = str(tmp_path / "kill_switch.json")
        import server.kill_switch as ks_mod
        original = ks_mod._STATE_PATH
        try:
            ks_mod._STATE_PATH = state_file
            from server.kill_switch import check_kill_switch
            assert check_kill_switch() is False
        finally:
            ks_mod._STATE_PATH = original


# ── DailyLossGuard tests ──────────────────────────────────────

class TestDailyLossGuard:

    def _make_guard(self, tmp_path, limit_pct=0.03):
        """Creates a DailyLossGuard backed by a temp state file."""
        state_file = str(tmp_path / "kill_switch.json")
        import server.kill_switch as ks_mod
        original   = ks_mod._STATE_PATH
        ks_mod._STATE_PATH = state_file

        cfg_file   = str(tmp_path / "config.json")
        with open(cfg_file, "w") as f:
            json.dump(
                {"settings": {"daily_loss_limit_pct": limit_pct}},
                f
            )

        original_cfg = ks_mod._CONFIG_PATH
        ks_mod._CONFIG_PATH = cfg_file

        from server.kill_switch import DailyLossGuard
        guard = DailyLossGuard()
        return guard, ks_mod, original, original_cfg

    def test_no_breach_within_limit(self, tmp_path):
        guard, ks_mod, orig, orig_cfg = self._make_guard(
            tmp_path, limit_pct=0.03
        )
        try:
            with patch("server.telegram_alerts.send_message",
                       side_effect=Exception("no telegram")):
                # Set opening value snapshot
                result1 = guard.check(100_000.0)
                # 1% drop — within 3% limit
                result2 = guard.check(99_000.0)
            assert result2["breached"] is False
            assert result2["drawdown_pct"] == pytest.approx(1.0, abs=0.01)
        finally:
            ks_mod._STATE_PATH  = orig
            ks_mod._CONFIG_PATH = orig_cfg

    def test_breach_at_limit(self, tmp_path):
        guard, ks_mod, orig, orig_cfg = self._make_guard(
            tmp_path, limit_pct=0.03
        )
        try:
            with patch("server.telegram_alerts.send_message",
                       side_effect=Exception("no telegram")):
                guard.check(100_000.0)          # set opening
                result = guard.check(96_000.0)  # 4% drop
            assert result["breached"] is True
            assert result["drawdown_pct"] >= 3.0
        finally:
            ks_mod._STATE_PATH  = orig
            ks_mod._CONFIG_PATH = orig_cfg

    def test_breach_activates_kill_switch(self, tmp_path):
        guard, ks_mod, orig, orig_cfg = self._make_guard(
            tmp_path, limit_pct=0.03
        )
        try:
            with patch("server.telegram_alerts.send_message",
                       side_effect=Exception("no telegram")), \
                 patch.object(guard._ks, "_close_all_positions",
                              return_value=([], [])):
                guard.check(100_000.0)
                guard.check(96_500.0)  # ~3.5% drop
            assert guard._ks.is_active() is True
        finally:
            ks_mod._STATE_PATH  = orig
            ks_mod._CONFIG_PATH = orig_cfg

    def test_zero_value_skipped(self, tmp_path):
        guard, ks_mod, orig, orig_cfg = self._make_guard(
            tmp_path, limit_pct=0.03
        )
        try:
            result = guard.check(0.0)
            assert result["breached"] is False
        finally:
            ks_mod._STATE_PATH  = orig
            ks_mod._CONFIG_PATH = orig_cfg

    def test_opening_value_reset_new_day(self, tmp_path):
        """Simulates crossing midnight — opening_value resets."""
        guard, ks_mod, orig, orig_cfg = self._make_guard(
            tmp_path, limit_pct=0.03
        )
        try:
            with patch("server.telegram_alerts.send_message",
                       side_effect=Exception("no telegram")):
                guard.check(100_000.0)

                # artificially age the snapshot to yesterday
                guard._ks._state["daily_loss"]["date"] = "2000-01-01"
                guard._ks._save_state()

                # next check should reset opening_value to new amount
                result = guard.check(110_000.0)
            # no breach — it's a fresh day with higher value
            assert result["breached"] is False
            assert result["opening_value"] == pytest.approx(110_000.0)
        finally:
            ks_mod._STATE_PATH  = orig
            ks_mod._CONFIG_PATH = orig_cfg
