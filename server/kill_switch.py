# kill_switch.py
# v7.0 — Emergency kill switch + daily loss circuit breaker
#
# Kill Switch (dashboard button):
#   Immediately halts all new orders
#   Closes all open Alpaca positions
#   Sends Telegram confirmation
#   State persists across restarts via kill_switch.json
#
# Daily Loss Circuit Breaker (v7.0 new):
#   Snapshots portfolio value at first check each trading day
#   Auto-activates kill switch if daily drawdown exceeds limit
#   Default limit: 3% (configurable via config.json settings.daily_loss_limit_pct)
#   Resets automatically at midnight UTC on next check
#   Use check_daily_loss(portfolio_value) before any execute_signals()
#
# Live/Paper Toggle:
#   Option A: Manual .env edit (always available)
#   Option B: Dashboard toggle with double confirmation
#   Prevents accidental live trading
#   Logs every mode change to audit trail
#
# Usage:
#   from server.kill_switch import KillSwitch, check_daily_loss
#   ks = KillSwitch()
#   ks.activate()          # emergency stop
#   ks.deactivate()        # resume trading
#   ks.is_active()         # check if halted
#   check_daily_loss(100000.0)  # auto-halts if -3% today

import os
import sys
import json
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv, set_key

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'
))

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(__file__), '..', '.env'
))

log = logging.getLogger(__name__)

# ── State file ────────────────────────────────────────────────
_STATE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'logs', 'kill_switch.json'
)

# ── Live trading confirmation code ────────────────────────────
# Must be typed exactly to switch to live trading
LIVE_CONFIRM_CODE = "CONFIRM_LIVE_TRADING"


class KillSwitch:
    """
    Emergency kill switch and trading mode manager.

    Kill switch state persists across restarts.
    If system crashes while kill switch is active,
    it stays active on restart — must be manually deactivated.
    """

    def __init__(self):
        os.makedirs(
            os.path.dirname(_STATE_PATH), exist_ok=True
        )
        self._state = self._load_state()

    # ── State persistence ──────────────────────────────────────

    def _load_state(self) -> dict:
        """Loads kill switch state from disk."""
        try:
            if os.path.exists(_STATE_PATH):
                with open(_STATE_PATH) as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "active":       False,
            "activated_at": None,
            "activated_by": None,
            "reason":       None,
            "trading_mode": os.getenv(
                "TRADING_MODE", "paper"
            ),
        }

    def _save_state(self) -> None:
        """Saves kill switch state to disk."""
        try:
            with open(_STATE_PATH, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            log.error(f"Kill switch state save failed: {e}")

    # ── Kill Switch ────────────────────────────────────────────

    def is_active(self) -> bool:
        """Returns True if kill switch is active."""
        return self._state.get("active", False)

    def activate(self, reason: str = "manual",
                  close_positions: bool = True) -> dict:
        """
        Activates the kill switch.
        Halts all new orders immediately.
        Optionally closes all open Alpaca positions.

        Args:
            reason:          why kill switch was activated
            close_positions: if True, closes all Alpaca positions

        Returns:
            dict with activation result
        """
        log.warning(
            f"KILL SWITCH ACTIVATED — reason: {reason}"
        )

        self._state["active"]       = True
        self._state["activated_at"] = datetime.now()\
            .strftime("%Y-%m-%d %H:%M:%S")
        self._state["activated_by"] = "dashboard"
        self._state["reason"]       = reason
        self._save_state()

        result = {
            "status":       "activated",
            "activated_at": self._state["activated_at"],
            "reason":       reason,
            "positions_closed": [],
            "positions_failed": [],
        }

        # log to database
        try:
            from database import get_db
            get_db().log_event(
                event_type = "kill_switch_activated",
                message    = f"Kill switch activated: {reason}",
                severity   = "critical",
                details    = result,
            )
        except Exception:
            pass

        # close all Alpaca positions
        if close_positions:
            closed, failed = self._close_all_positions()
            result["positions_closed"] = closed
            result["positions_failed"] = failed

        # send Telegram alert
        try:
            from server.telegram_alerts import send_message
            closed_str = (
                ", ".join(result["positions_closed"])
                if result["positions_closed"]
                else "none"
            )
            failed_str = (
                ", ".join(result["positions_failed"])
                if result["positions_failed"]
                else "none"
            )
            send_message(
                f"🛑 <b>KILL SWITCH ACTIVATED</b>\n"
                f"<code>Reason:  {reason}\n"
                f"Time:    {self._state['activated_at']}\n"
                f"Closed:  {closed_str}\n"
                f"Failed:  {failed_str}\n"
                f"Status:  ALL TRADING HALTED</code>"
            )
        except Exception as e:
            log.error(f"Kill switch Telegram failed: {e}")

        return result

    def deactivate(self, reason: str = "manual") -> dict:
        """
        Deactivates the kill switch.
        Resumes normal trading operations.

        Args:
            reason: why kill switch was deactivated

        Returns:
            dict with deactivation result
        """
        log.info(
            f"Kill switch deactivated — reason: {reason}"
        )

        was_active = self._state.get("active", False)

        self._state["active"]         = False
        self._state["deactivated_at"] = datetime.now()\
            .strftime("%Y-%m-%d %H:%M:%S")
        self._state["deactivated_by"] = "dashboard"
        self._state["deactivate_reason"] = reason
        self._save_state()

        result = {
            "status":         "deactivated",
            "deactivated_at": self._state["deactivated_at"],
            "was_active":     was_active,
            "reason":         reason,
        }

        # log to database
        try:
            from database import get_db
            get_db().log_event(
                event_type = "kill_switch_deactivated",
                message    = (
                    f"Kill switch deactivated: {reason}"
                ),
                severity   = "info",
                details    = result,
            )
        except Exception:
            pass

        # send Telegram alert
        try:
            from server.telegram_alerts import send_message
            send_message(
                f"✅ <b>KILL SWITCH DEACTIVATED</b>\n"
                f"<code>Reason: {reason}\n"
                f"Time:   {self._state['deactivated_at']}\n"
                f"Status: TRADING RESUMED</code>"
            )
        except Exception:
            pass

        return result

    def get_status(self) -> dict:
        """Returns full kill switch status for dashboard."""
        return {
            "active":         self._state.get(
                "active", False
            ),
            "activated_at":   self._state.get(
                "activated_at"
            ),
            "activated_by":   self._state.get(
                "activated_by"
            ),
            "reason":         self._state.get("reason"),
            "trading_mode":   self.get_trading_mode(),
            "is_paper":       self.is_paper_trading(),
        }

    def _close_all_positions(self) -> tuple:
        """
        Closes all open positions in Alpaca.
        Returns (closed_symbols, failed_symbols).
        """
        closed = []
        failed = []

        try:
            from server.alpaca_trader import AlpacaTrader
            trader    = AlpacaTrader()
            positions = trader.get_positions()

            if not positions:
                log.info(
                    "  Kill switch: no open positions "
                    "to close"
                )
                return closed, failed

            log.warning(
                f"  Kill switch: closing "
                f"{len(positions)} positions..."
            )

            for pos in positions:
                symbol = pos["symbol"]
                try:
                    trader.close_position(symbol)
                    closed.append(symbol)
                    log.warning(
                        f"  Closed: {symbol}"
                    )

                    # update DB position
                    try:
                        from database import get_db
                        get_db().close_position_db(symbol)
                        get_db().save_trade({
                            "symbol":      symbol,
                            "action":      "SELL",
                            "price":       pos.get(
                                "current_price", 0
                            ),
                            "exit_reason": "kill_switch",
                            "shares":      pos.get("qty"),
                            "entry_price": pos.get(
                                "avg_entry"
                            ),
                        })
                    except Exception:
                        pass

                except Exception as e:
                    failed.append(symbol)
                    log.error(
                        f"  Failed to close {symbol}: {e}"
                    )

        except Exception as e:
            log.error(
                f"  Kill switch position close failed: {e}"
            )

        return closed, failed

    # ── Trading Mode Toggle ────────────────────────────────────

    def get_trading_mode(self) -> str:
        """Returns current trading mode: paper or live."""
        return os.getenv("TRADING_MODE", "paper").lower()

    def is_paper_trading(self) -> bool:
        """Returns True if in paper trading mode."""
        return self.get_trading_mode() == "paper"

    def set_trading_mode(self, mode: str,
                          confirm_code: str = None) -> dict:
        """
        Switches trading mode between paper and live.

        Switching to LIVE requires:
          1. confirm_code == LIVE_CONFIRM_CODE
          2. Kill switch must be INACTIVE
          3. Logs the change with full audit trail

        Switching to PAPER:
          No confirmation required
          Always safe

        Args:
            mode:         "paper" or "live"
            confirm_code: must equal LIVE_CONFIRM_CODE
                          when switching to live

        Returns:
            dict with result and new mode
        """
        mode = mode.lower().strip()

        if mode not in ["paper", "live"]:
            return {
                "status":  "error",
                "message": "Mode must be 'paper' or 'live'",
            }

        current_mode = self.get_trading_mode()

        # switching to paper — always allowed
        if mode == "paper":
            return self._apply_mode_change(
                "paper", current_mode
            )

        # switching to live — requires confirmation
        if mode == "live":

            # check confirmation code
            if confirm_code != LIVE_CONFIRM_CODE:
                return {
                    "status":  "error",
                    "message": (
                        "Incorrect confirmation code. "
                        "To switch to live trading, provide "
                        f"confirm_code='{LIVE_CONFIRM_CODE}'"
                    ),
                    "current_mode": current_mode,
                }

            # kill switch must be inactive
            if self.is_active():
                return {
                    "status":  "error",
                    "message": (
                        "Kill switch is active. "
                        "Deactivate kill switch before "
                        "switching to live trading."
                    ),
                    "current_mode": current_mode,
                }

            # check Alpaca base URL has 'paper'
            alpaca_url = os.getenv(
                "ALPACA_BASE_URL", ""
            ).lower()
            if "paper" not in alpaca_url:
                return {
                    "status":  "error",
                    "message": (
                        "ALPACA_BASE_URL does not contain "
                        "'paper'. Update .env with live "
                        "Alpaca credentials before switching."
                    ),
                }

            return self._apply_mode_change(
                "live", current_mode
            )

        return {
            "status":  "error",
            "message": f"Unknown mode: {mode}",
        }

    def _apply_mode_change(self, new_mode: str,
                            old_mode: str) -> dict:
        """Applies the trading mode change."""
        if new_mode == old_mode:
            return {
                "status":  "ok",
                "message": f"Already in {new_mode} mode",
                "mode":    new_mode,
            }

        # update .env file
        env_path = os.path.join(
            os.path.dirname(__file__), '..', '.env'
        )
        try:
            set_key(env_path, "TRADING_MODE", new_mode)
        except Exception as e:
            log.error(f"Failed to update .env: {e}")
            return {
                "status":  "error",
                "message": f"Failed to update .env: {e}",
            }

        # update state
        self._state["trading_mode"] = new_mode
        self._save_state()

        # log to database
        try:
            from database import get_db
            get_db().log_event(
                event_type = "trading_mode_change",
                message    = (
                    f"Trading mode: {old_mode} → {new_mode}"
                ),
                severity   = (
                    "warn" if new_mode == "live"
                    else "info"
                ),
                details    = {
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                    "changed_at": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
        except Exception:
            pass

        # Telegram alert
        try:
            from server.telegram_alerts import send_message
            emoji = "🔴" if new_mode == "live" else "📄"
            send_message(
                f"{emoji} <b>TRADING MODE CHANGED</b>\n"
                f"<code>{old_mode.upper()} → "
                f"{new_mode.upper()}\n"
                f"Time: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                f"</code>"
            )
        except Exception:
            pass

        log.warning(
            f"Trading mode changed: {old_mode} → {new_mode}"
        )

        if new_mode == "live":
            mode_msg = (
                f"Trading mode label changed to live. "
                f"IMPORTANT: This does NOT connect to a live Alpaca "
                f"account. To trade real money you must: "
                f"(1) Stop the server, "
                f"(2) Edit .env with your live Alpaca API key and "
                f"live base URL (remove 'paper' from the URL), "
                f"(3) Restart the server. "
                f"The dashboard toggle is a label only."
            )
        else:
            mode_msg = f"Trading mode changed to {new_mode}"

        return {
            "status":   "ok",
            "message":  mode_msg,
            "old_mode": old_mode,
            "new_mode": new_mode,
        }


# ── Daily Loss Circuit Breaker ────────────────────────────────

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'config.json'
)


class DailyLossGuard:
    """
    Monitors intraday portfolio drawdown and auto-activates
    the kill switch when daily loss exceeds the configured limit.

    How it works:
      - First call each UTC day snapshots the opening portfolio value
      - Each subsequent call computes drawdown vs that snapshot
      - If drawdown >= limit → kill switch activates, new orders halt
      - State stored inside kill_switch.json (daily_loss key)
      - Resets automatically on first call of the next UTC day

    Default limit: 3% (config.json settings.daily_loss_limit_pct)
    """

    def __init__(self):
        self._ks = KillSwitch()

    def _load_limit(self) -> float:
        try:
            with open(_CONFIG_PATH) as f:
                config = json.load(f)
            return float(
                config.get("settings", {})
                      .get("daily_loss_limit_pct", 0.03)
            )
        except Exception:
            return 0.03

    def check(self, current_value: float) -> dict:
        """
        Checks current portfolio value against today's opening snapshot.
        Auto-activates kill switch if limit breached.

        Args:
            current_value: live portfolio value from Alpaca

        Returns:
            {
              "breached":       bool,
              "drawdown_pct":   float,   # e.g. 2.5
              "limit_pct":      float,   # e.g. 3.0
              "opening_value":  float,
              "current_value":  float,
            }
        """
        if current_value <= 0:
            return {"breached": False, "drawdown_pct": 0.0,
                    "limit_pct": 0.0}

        limit_pct = self._load_limit()
        today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state     = self._ks._state
        daily     = state.get("daily_loss", {})

        # new day or first run — snapshot opening value
        if daily.get("date") != today or not daily.get("opening_value"):
            state["daily_loss"] = {
                "date":          today,
                "opening_value": current_value,
                "limit_pct":     limit_pct,
                "last_check":    datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            self._ks._save_state()
            log.info(
                f"Daily loss guard reset — "
                f"opening value: ${current_value:,.2f} "
                f"limit: {limit_pct*100:.1f}%"
            )
            return {
                "breached":      False,
                "drawdown_pct":  0.0,
                "limit_pct":     round(limit_pct * 100, 1),
                "opening_value": current_value,
                "current_value": current_value,
            }

        opening  = daily["opening_value"]
        drawdown = (opening - current_value) / opening

        # update last_check timestamp
        state["daily_loss"]["last_check"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self._ks._save_state()

        result = {
            "breached":      drawdown >= limit_pct,
            "drawdown_pct":  round(drawdown * 100, 2),
            "limit_pct":     round(limit_pct * 100, 1),
            "opening_value": opening,
            "current_value": current_value,
        }

        if result["breached"]:
            log.warning(
                f"DAILY LOSS LIMIT BREACHED: "
                f"-{result['drawdown_pct']}% "
                f"(limit -{result['limit_pct']}%) "
                f"opening=${opening:,.2f} "
                f"current=${current_value:,.2f}"
            )
            if not self._ks.is_active():
                self._ks.activate(
                    reason=(
                        f"daily_loss_limit_"
                        f"{result['drawdown_pct']}pct"
                    ),
                    close_positions=False,
                )
                # extra Telegram with detail
                try:
                    from server.telegram_alerts import send_message
                    send_message(
                        f"🛑 <b>DAILY LOSS LIMIT BREACHED</b>\n"
                        f"<code>"
                        f"Loss:    -{result['drawdown_pct']}%\n"
                        f"Limit:   -{result['limit_pct']}%\n"
                        f"Opening: ${opening:,.2f}\n"
                        f"Current: ${current_value:,.2f}\n"
                        f"Action:  ALL NEW ORDERS HALTED\n"
                        f"Note:    Existing positions kept open\n"
                        f"         Deactivate kill switch to resume"
                        f"</code>"
                    )
                except Exception:
                    pass
        else:
            log.info(
                f"Daily loss check: "
                f"-{result['drawdown_pct']}% "
                f"(limit -{result['limit_pct']}%) — OK"
            )

        return result

    def get_status(self) -> dict:
        """Returns current daily loss status for dashboard."""
        state = self._ks._state
        daily = state.get("daily_loss", {})
        limit_pct = self._load_limit()
        opening   = daily.get("opening_value", 0)
        return {
            "date":          daily.get("date"),
            "opening_value": opening,
            "limit_pct":     round(limit_pct * 100, 1),
            "last_check":    daily.get("last_check"),
        }


# ── Guard function for scheduler ─────────────────────────────

def check_kill_switch() -> bool:
    """
    Call this at the start of any order placement.
    Returns True if trading is halted (kill switch active).
    """
    ks = KillSwitch()
    if ks.is_active():
        log.warning(
            "Kill switch active — order placement blocked"
        )
        return True
    return False


def check_daily_loss(current_portfolio_value: float) -> dict:
    """
    Call this at the start of execute_signals() after check_kill_switch().
    Returns dict with breached bool and drawdown details.
    Auto-activates kill switch if daily loss limit exceeded.

    Usage:
        from server.kill_switch import check_daily_loss
        result = check_daily_loss(portfolio_value)
        if result["breached"]:
            return [{"action": "halted", "reason": "daily_loss_limit"}]
    """
    guard = DailyLossGuard()
    return guard.check(current_portfolio_value)


# ── API endpoints (imported by api.py) ───────────────────────

def get_kill_switch_api():
    """Returns kill switch status for dashboard API."""
    ks = KillSwitch()
    return ks.get_status()


def activate_kill_switch_api(reason: str = "dashboard") -> dict:
    """Activates kill switch from dashboard."""
    ks = KillSwitch()
    return ks.activate(reason=reason)


def deactivate_kill_switch_api(reason: str = "dashboard") -> dict:
    """Deactivates kill switch from dashboard."""
    ks = KillSwitch()
    return ks.deactivate(reason=reason)


def set_trading_mode_api(mode: str,
                          confirm_code: str = None) -> dict:
    """Sets trading mode from dashboard."""
    ks = KillSwitch()
    return ks.set_trading_mode(mode, confirm_code)


# ── Test runner ───────────────────────────────────────────────

if __name__ == "__main__":
    from colorama import Fore, Style, init
    init(autoreset=True)

    print(Fore.WHITE + Style.BRIGHT +
          "\n  Testing KillSwitch...\n")

    ks = KillSwitch()

    # test status
    print(Fore.WHITE + Style.BRIGHT +
          "=== Initial Status ===")
    status = ks.get_status()
    print(Fore.WHITE +
          f"  Active:       {status['active']}")
    print(Fore.WHITE +
          f"  Trading mode: {status['trading_mode']}")
    print(Fore.WHITE +
          f"  Is paper:     {status['is_paper']}")

    # test kill switch activate/deactivate
    print(Fore.WHITE + Style.BRIGHT +
          "\n=== Kill Switch Test ===")
    result = ks.activate(
        reason="test", close_positions=False
    )
    print(Fore.RED +
          f"  Activated: {result['status']} "
          f"@ {result['activated_at']}")
    print(Fore.WHITE +
          f"  Is active: {ks.is_active()}")

    result = ks.deactivate(reason="test_complete")
    print(Fore.GREEN +
          f"  Deactivated: {result['status']}")
    print(Fore.WHITE +
          f"  Is active: {ks.is_active()}")

    # test mode switch — paper (no confirmation needed)
    print(Fore.WHITE + Style.BRIGHT +
          "\n=== Trading Mode Test ===")
    result = ks.set_trading_mode("paper")
    print(Fore.WHITE +
          f"  Switch to paper: {result['status']} — "
          f"{result['message']}")

    # test mode switch — live without confirmation
    result = ks.set_trading_mode("live")
    print(Fore.YELLOW +
          f"  Switch to live (no code): "
          f"{result['status']} — {result['message']}")

    # test mode switch — live with wrong confirmation
    result = ks.set_trading_mode(
        "live", confirm_code="WRONG_CODE"
    )
    print(Fore.YELLOW +
          f"  Switch to live (wrong code): "
          f"{result['status']}")

    # test check_kill_switch guard
    print(Fore.WHITE + Style.BRIGHT +
          "\n=== Guard Function Test ===")
    halted = check_kill_switch()
    print(Fore.WHITE +
          f"  check_kill_switch(): {halted} "
          f"(should be False)")

    ks.activate(reason="guard_test", close_positions=False)
    halted = check_kill_switch()
    print(Fore.WHITE +
          f"  check_kill_switch() while active: {halted} "
          f"(should be True)")
    ks.deactivate(reason="guard_test_complete")

    print(Fore.GREEN + Style.BRIGHT +
          "\n  KillSwitch working!\n")