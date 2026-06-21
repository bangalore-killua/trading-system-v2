# order_manager.py
# v5.0 — Order reconciliation + duplicate prevention
#
# Two responsibilities:
#
# 1. Duplicate Prevention
#    Before placing any order, checks an execution lock.
#    If a symbol is already being processed, blocks the
#    duplicate order. Lock expires after 5 minutes.
#    Prevents double orders from:
#      - Scheduler firing twice on restart
#      - Race condition between run_strategy() and
#        score_open_positions() running simultaneously
#      - Manual API trigger while scheduler is running
#
# 2. Order Reconciliation
#    After placing an order, polls Alpaca to verify fill.
#    If filled: records actual fill price in DB
#    If rejected: logs error, sends Telegram alert
#    If partial: logs partial fill, adjusts position
#    Runs reconciliation check every 30 minutes
#    comparing our DB positions vs Alpaca positions
#
# Usage:
#   from server.order_manager import OrderManager
#   om = OrderManager(trader)
#   if om.can_place_order(symbol, "BUY"):
#       order = trader.place_buy_order(symbol, notional=500)
#       om.track_order(order, symbol, "BUY", notional=500)

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'
))

log = logging.getLogger(__name__)

# ── Lock settings — read from config.json at runtime via self.xxx ──────────

# ── Lock file path ────────────────────────────────────────────
_LOCK_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'logs', 'order_locks.json'
)

# ── Module-level threading lock ───────────────────────────────
# Makes can_place_order + acquire_lock atomic across threads.
# Without this, scheduler thread and API thread can both pass
# can_place_order() before either writes the lock to disk.
_THREAD_LOCK = threading.Lock()


class OrderManager:
    """
    Manages order safety for the trading system.

    Wraps AlpacaTrader to add:
      - Duplicate order prevention via execution locks
      - Post-order fill verification
      - Position reconciliation vs Alpaca account
      - Full audit trail in PostgreSQL
    """

    def __init__(self, trader=None):
        """
        trader — AlpacaTrader instance (optional)
                 if None, reconciliation methods still work
                 but order placement methods will fail
        """
        self.trader = trader
        self._locks = self._load_locks()
        from strategy.config_loader import get_settings as _gs3
        _s = _gs3()
        self.lock_expiry_minutes = int(_s.get("lock_expiry_minutes",  5))
        self.fill_poll_seconds   = int(_s.get("fill_poll_seconds",   30))
        self.fill_max_wait       = int(_s.get("fill_max_wait",       120))
        self.reconcile_tolerance = float(_s.get("reconcile_tolerance", 0.001))

    # ── Lock Management ───────────────────────────────────────

    def _load_locks(self) -> dict:
        """Loads execution locks from disk."""
        try:
            if os.path.exists(_LOCK_PATH):
                with open(_LOCK_PATH) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_locks(self) -> None:
        """Saves execution locks to disk."""
        try:
            os.makedirs(
                os.path.dirname(_LOCK_PATH), exist_ok=True
            )
            with open(_LOCK_PATH, "w") as f:
                json.dump(self._locks, f, indent=2)
        except Exception as e:
            log.error(f"Lock save failed: {e}")

    def _cleanup_expired_locks(self) -> None:
        """Removes locks that have expired."""
        now     = datetime.now()
        expired = []
        for key, lock in self._locks.items():
            try:
                expires = datetime.strptime(
                    lock["expires_at"],
                    "%Y-%m-%d %H:%M:%S"
                )
                if now > expires:
                    expired.append(key)
            except Exception:
                expired.append(key)

        for key in expired:
            del self._locks[key]

        if expired:
            self._save_locks()
            log.info(
                f"  Cleaned {len(expired)} expired locks"
            )

    def can_place_order(self, symbol: str,
                         side: str) -> dict:
        """
        Checks if an order can be placed for a symbol.
        Returns dict with allowed bool and reason.

        Blocks if:
          - Same symbol+side already has an active lock
          - Lock was set within LOCK_EXPIRY_MINUTES

        Args:
            symbol: ticker e.g. "AAPL"
            side:   "BUY" or "SELL"

        Returns:
            {"allowed": True, "reason": "clear"}
            {"allowed": False, "reason": "duplicate_lock_..."}
        """
        self._cleanup_expired_locks()

        lock_key = f"{symbol}_{side.upper()}"

        if lock_key in self._locks:
            lock    = self._locks[lock_key]
            expires = lock.get("expires_at", "")
            return {
                "allowed": False,
                "reason":  f"duplicate_lock_{lock_key}",
                "locked_at": lock.get("locked_at"),
                "expires_at": expires,
                "order_id":  lock.get("order_id"),
            }

        return {"allowed": True, "reason": "clear"}

    def acquire_lock(self, symbol: str,
                      side: str,
                      order_id: str = None) -> None:
        """
        Acquires an execution lock for a symbol+side.
        Call immediately before placing an order.
        Lock expires automatically after LOCK_EXPIRY_MINUTES.
        """
        lock_key   = f"{symbol}_{side.upper()}"
        expires_at = (
            datetime.now() +
            timedelta(minutes=self.lock_expiry_minutes)
        ).strftime("%Y-%m-%d %H:%M:%S")

        self._locks[lock_key] = {
            "symbol":     symbol,
            "side":       side.upper(),
            "locked_at":  datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "expires_at": expires_at,
            "order_id":   order_id,
        }
        self._save_locks()
        log.info(
            f"  Lock acquired: {lock_key} "
            f"(expires {expires_at})"
        )

    def release_lock(self, symbol: str,
                      side: str) -> None:
        """
        Releases an execution lock after order is confirmed.
        Call after fill verification completes.
        """
        lock_key = f"{symbol}_{side.upper()}"
        if lock_key in self._locks:
            del self._locks[lock_key]
            self._save_locks()
            log.info(f"  Lock released: {lock_key}")

    def release_all_locks(self) -> None:
        """
        Emergency release of all locks.
        Use if system crashed mid-execution.
        """
        count = len(self._locks)
        self._locks = {}
        self._save_locks()
        log.warning(
            f"  All {count} locks released (emergency)"
        )

    # ── Order Tracking ─────────────────────────────────────────

    def track_order(self, order: dict,
                     symbol: str,
                     side: str,
                     notional: float = None,
                     qty: float      = None,
                     scorecard: dict = None) -> dict:
        """
        Tracks an order after submission.
        Verifies fill status and records to database.

        Call immediately after place_buy_order or
        place_sell_order returns.

        Args:
            order:     return value from Alpaca trader
            symbol:    ticker
            side:      "BUY" or "SELL"
            notional:  dollar amount (for buys)
            qty:       share count (for sells)
            scorecard: full scorecard at time of order

        Returns:
            Updated order dict with fill information
        """
        if "error" in order:
            log.error(
                f"  Order failed for {symbol}: "
                f"{order['error']}"
            )
            self.release_lock(symbol, side)
            self._log_order_event(
                symbol    = symbol,
                side      = side,
                status    = "failed",
                reason    = order["error"],
                scorecard = scorecard,
            )
            return order

        order_id = order.get("order_id")
        if not order_id:
            self.release_lock(symbol, side)
            return order

        # update lock with order ID
        self.acquire_lock(symbol, side, order_id)

        # Run fill verification in a background thread so the
        # scheduler is not blocked for 30 seconds per order.
        # The lock is released and DB is written when fill resolves.
        def _verify_async():
            try:
                fill_result = self._verify_fill(order_id, symbol, side)
                self._log_order_event(
                    symbol     = symbol,
                    side       = side,
                    order_id   = order_id,
                    status     = fill_result.get("fill_status", "unknown"),
                    fill_price = fill_result.get("fill_price"),
                    fill_qty   = fill_result.get("fill_qty"),
                    notional   = notional,
                    scorecard  = scorecard,
                )
            finally:
                self.release_lock(symbol, side)

        t = threading.Thread(target=_verify_async, daemon=True,
                             name=f"fill-verify-{symbol}-{side}")
        t.start()

        # Return immediately — fill result will be logged async
        order["fill_status"] = "pending_verification"
        return order

    def _verify_fill(self, order_id: str,
                      symbol: str,
                      side: str) -> dict:
        """
        Polls Alpaca to verify order fill status.
        Waits up to FILL_MAX_WAIT seconds for fill.

        Returns dict with fill_status, fill_price, fill_qty.
        """
        if not self.trader:
            return {
                "fill_status": "unverified",
                "reason":      "no_trader"
            }

        log.info(
            f"  Verifying fill: {order_id} "
            f"({symbol} {side})"
        )

        # poll until filled/cancelled/rejected or timeout
        _poll_interval = self.fill_poll_seconds
        _polls_done    = 0
        _max_polls     = self.fill_max_wait // _poll_interval
        while _polls_done < _max_polls:
            time.sleep(_poll_interval)
            _polls_done += 1
            try:
                _orders = self.trader.get_orders(limit=20)
                _order  = next(
                    (o for o in _orders if o.get("id") == order_id),
                    None,
                )
                if _order:
                    _status = _order.get("status", "")
                    if _status in ["filled", "cancelled", "rejected"]:
                        break
            except Exception:
                pass

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            # fetch specific order
            orders = self.trader.get_orders(limit=20)
            order  = next(
                (o for o in orders
                 if o.get("id") == order_id),
                None
            )

            if not order:
                return {
                    "fill_status": "not_found",
                    "order_id":    order_id,
                }

            status     = order.get("status", "unknown")
            fill_price = order.get("filled_price", 0)
            fill_qty   = order.get("filled_qty", 0)

            if status == "filled":
                log.info(
                    f"  ✅ Filled: {symbol} {side} "
                    f"{fill_qty} @ ${fill_price}"
                )
                return {
                    "fill_status": "filled",
                    "fill_price":  fill_price,
                    "fill_qty":    fill_qty,
                    "order_id":    order_id,
                }

            elif status == "partially_filled":
                log.warning(
                    f"  ⚠️ Partial fill: {symbol} {side} "
                    f"{fill_qty} @ ${fill_price}"
                )
                return {
                    "fill_status": "partial",
                    "fill_price":  fill_price,
                    "fill_qty":    fill_qty,
                    "order_id":    order_id,
                }

            elif status in ["cancelled", "rejected"]:
                log.error(
                    f"  ❌ Order {status}: {symbol} {side}"
                )
                # send Telegram alert
                try:
                    from server.telegram_alerts import (
                        send_message
                    )
                    send_message(
                        f"❌ <b>ORDER {status.upper()}</b>\n"
                        f"<code>Symbol: {symbol}\n"
                        f"Side:   {side}\n"
                        f"ID:     {order_id}</code>"
                    )
                except Exception:
                    pass
                return {
                    "fill_status": status,
                    "order_id":    order_id,
                }

            else:
                # pending — not filled yet
                return {
                    "fill_status": "pending",
                    "status":      status,
                    "order_id":    order_id,
                }

        except Exception as e:
            log.error(f"  Fill verification failed: {e}")
            return {
                "fill_status": "verify_error",
                "error":       str(e),
                "order_id":    order_id,
            }

    def _log_order_event(self, symbol: str,
                          side: str,
                          status: str,
                          order_id: str   = None,
                          fill_price: float = None,
                          fill_qty: float   = None,
                          notional: float   = None,
                          reason: str       = None,
                          scorecard: dict   = None) -> None:
        """Logs order event to database audit trail."""
        try:
            from database import get_db
            get_db().log_event(
                event_type = f"order_{side.lower()}",
                message    = (
                    f"{symbol} {side} {status}"
                    + (f" @ ${fill_price}" if fill_price
                       else "")
                ),
                severity   = (
                    "error" if status in
                    ["failed", "rejected", "cancelled"]
                    else "info"
                ),
                symbol     = symbol,
                details    = {
                    "order_id":   order_id,
                    "side":       side,
                    "status":     status,
                    "fill_price": fill_price,
                    "fill_qty":   fill_qty,
                    "notional":   notional,
                    "reason":     reason,
                    "scorecard_conviction": scorecard.get(
                        "conviction"
                    ) if scorecard else None,
                    "scorecard_score": scorecard.get(
                        "normalised_score"
                    ) if scorecard else None,
                }
            )
        except Exception as e:
            log.error(f"Order event log failed: {e}")

    # ── Position Reconciliation ───────────────────────────────

    def reconcile_positions(self) -> dict:
        """
        Compares our database positions vs Alpaca account.
        Detects and reports discrepancies.

        Called every 30 minutes by scheduler.

        Returns dict with reconciliation results:
          matched   — positions in sync
          missing   — in DB but not in Alpaca
          extra     — in Alpaca but not in DB
          mismatch  — in both but qty/price differs
        """
        if not self.trader:
            return {"error": "no_trader"}

        log.info("  Running position reconciliation...")

        try:
            from database import get_db

            # get our DB positions
            db_positions = {
                p["symbol"]: p
                for p in get_db().get_open_positions()
            }

            # get Alpaca positions
            alpaca_positions = {
                p["symbol"]: p
                for p in self.trader.get_positions()
            }

            results = {
                "matched":   [],
                "missing":   [],   # in DB not Alpaca
                "extra":     [],   # in Alpaca not DB
                "mismatch":  [],   # qty/price differs
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }

            # check DB positions against Alpaca
            for symbol, db_pos in db_positions.items():
                if symbol not in alpaca_positions:
                    results["missing"].append({
                        "symbol":    symbol,
                        "db_shares": db_pos.get("shares"),
                        "db_entry":  db_pos.get("avg_entry"),
                        "issue":     "in_db_not_alpaca",
                    })
                    log.warning(
                        f"  ⚠️ {symbol}: in DB but "
                        f"not in Alpaca"
                    )
                else:
                    # compare quantities
                    al_pos    = alpaca_positions[symbol]
                    db_shares = float(
                        db_pos.get("shares", 0)
                    )
                    al_shares = float(al_pos.get("qty", 0))
                    diff      = abs(
                        db_shares - al_shares
                    ) / max(db_shares, 0.001)

                    if diff > self.reconcile_tolerance:
                        results["mismatch"].append({
                            "symbol":    symbol,
                            "db_shares": db_shares,
                            "al_shares": al_shares,
                            "diff_pct":  round(diff * 100, 2),
                            "issue":     "qty_mismatch",
                        })
                        if diff < 0.05:
                            # CHANGE 12 / HOT-2: < 5% mismatch —
                            # auto-correct BOTH DB and trade_state so they
                            # stay in sync. Previously only DB was updated.
                            log.info(
                                f"  Auto-correcting {symbol}: "
                                f"DB {db_shares:.4f} → "
                                f"Alpaca {al_shares:.4f} "
                                f"({diff*100:.2f}% diff)"
                            )
                            try:
                                from database import get_db
                                get_db().log_event(
                                    event_type = "reconciliation_autocorrect",
                                    message    = (
                                        f"{symbol}: DB {db_shares:.4f} → "
                                        f"{al_shares:.4f}"
                                    ),
                                    severity   = "info",
                                    details    = {
                                        "symbol": symbol,
                                        "db_shares": db_shares,
                                        "al_shares": al_shares,
                                        "diff_pct": round(diff * 100, 2),
                                    },
                                )
                            except Exception:
                                pass
                            # HOT-2: sync trade_state open_positions.json
                            try:
                                from strategy.trade_state import (
                                    TradeStateManager as _TSTM
                                )
                                _ts_r = _TSTM()
                                _pos  = _ts_r.get_open_position(symbol)
                                if _pos:
                                    _pos.shares    = al_shares
                                    _pos.avg_entry = float(
                                        al_pos.get("avg_entry",
                                                   _pos.avg_entry)
                                    )
                                    _ts_r._save_positions()
                                    log.info(
                                        f"  trade_state synced: "
                                        f"{symbol} shares → {al_shares:.4f}"
                                    )
                            except Exception as _te:
                                log.warning(
                                    f"  trade_state sync failed for "
                                    f"{symbol}: {_te}"
                                )
                        else:
                            # > 5%: Telegram alert, await manual review
                            log.warning(
                                f"  ⚠️ {symbol}: qty mismatch > 5% "
                                f"DB={db_shares:.4f} "
                                f"Alpaca={al_shares:.4f} — "
                                f"manual review required"
                            )
                            try:
                                from server.telegram_alerts import send_message
                                send_message(
                                    f"⚠️ <b>RECONCILIATION MISMATCH > 5%"
                                    f" — {symbol}</b>\n"
                                    f"<code>DB:     {db_shares:.4f}\n"
                                    f"Alpaca: {al_shares:.4f}\n"
                                    f"Diff:   {diff*100:.2f}%\n"
                                    f"Action: Manual review required</code>"
                                )
                            except Exception:
                                pass
                    else:
                        results["matched"].append(symbol)

            # check Alpaca positions against DB
            for symbol, al_pos in alpaca_positions.items():
                if symbol not in db_positions:
                    results["extra"].append({
                        "symbol":    symbol,
                        "al_shares": al_pos.get("qty"),
                        "al_entry":  al_pos.get("avg_entry"),
                        "issue":     "in_alpaca_not_db",
                    })
                    log.warning(
                        f"  ⚠️ {symbol}: in Alpaca but "
                        f"not in DB"
                    )

            # log reconciliation result
            issues = (
                len(results["missing"]) +
                len(results["extra"]) +
                len(results["mismatch"])
            )

            if issues > 0:
                try:
                    from database import get_db
                    get_db().log_event(
                        event_type = "reconciliation_mismatch",
                        message    = (
                            f"{issues} position discrepancies "
                            f"found"
                        ),
                        severity   = "warn",
                        details    = results,
                    )
                except Exception:
                    pass

                # send Telegram if discrepancies
                try:
                    from server.telegram_alerts import (
                        send_message
                    )
                    lines = [
                        "⚠️ <b>RECONCILIATION ALERT</b>\n"
                    ]
                    for m in results["missing"]:
                        lines.append(
                            f"<code>MISSING: "
                            f"{m['symbol']} — "
                            f"in DB not Alpaca</code>"
                        )
                    for e in results["extra"]:
                        lines.append(
                            f"<code>EXTRA: "
                            f"{e['symbol']} — "
                            f"in Alpaca not DB</code>"
                        )
                    for mm in results["mismatch"]:
                        lines.append(
                            f"<code>MISMATCH: "
                            f"{mm['symbol']} "
                            f"DB={mm['db_shares']:.2f} "
                            f"vs Alpaca="
                            f"{mm['al_shares']:.2f}</code>"
                        )
                    send_message("\n".join(lines))
                except Exception:
                    pass
            else:
                log.info(
                    f"  ✅ Reconciliation clean: "
                    f"{len(results['matched'])} positions "
                    f"in sync"
                )

            return results

        except Exception as e:
            log.error(f"Reconciliation failed: {e}")
            return {"error": str(e)}

    def get_lock_status(self) -> dict:
        """Returns current lock status for dashboard."""
        self._cleanup_expired_locks()
        return {
            "active_locks": len(self._locks),
            "locks":        list(self._locks.values()),
        }


# ── Safe Order Execution ──────────────────────────────────────

def safe_buy(trader,
             symbol:    str,
             notional:  float,
             scorecard: dict = None) -> dict:
    """
    Safe wrapper for buy orders.
    Checks duplicate lock, places order, verifies fill.

    Usage:
        from server.order_manager import safe_buy
        result = safe_buy(trader, "AAPL", 500, scorecard)
    """
    om = OrderManager(trader)

    # Atomic check-then-acquire under module-level thread lock.
    # Prevents two threads from both passing can_place_order()
    # before either writes the lock file.
    with _THREAD_LOCK:
        check = om.can_place_order(symbol, "BUY")
        if not check["allowed"]:
            log.warning(f"  {symbol} BUY blocked: {check['reason']}")
            return {"status": "blocked", "reason": check["reason"],
                    "symbol": symbol}
        om.acquire_lock(symbol, "BUY")

    try:
        order = trader.place_buy_order(
            symbol=symbol, notional=notional
        )
        return om.track_order(
            order     = order,
            symbol    = symbol,
            side      = "BUY",
            notional  = notional,
            scorecard = scorecard,
        )
    except Exception as e:
        om.release_lock(symbol, "BUY")
        log.error(f"safe_buy failed for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}


def safe_sell(trader,
              symbol:    str,
              qty:       float = None,
              percentage: float = None,
              scorecard: dict  = None) -> dict:
    """
    Safe wrapper for sell orders.
    Checks duplicate lock, places order, verifies fill.
    """
    om = OrderManager(trader)

    with _THREAD_LOCK:
        check = om.can_place_order(symbol, "SELL")
        if not check["allowed"]:
            log.warning(f"  {symbol} SELL blocked: {check['reason']}")
            return {"status": "blocked", "reason": check["reason"],
                    "symbol": symbol}
        om.acquire_lock(symbol, "SELL")

    try:
        order = trader.place_sell_order(
            symbol     = symbol,
            qty        = qty,
            percentage = percentage,
        )
        return om.track_order(
            order     = order,
            symbol    = symbol,
            side      = "SELL",
            qty       = qty,
            scorecard = scorecard,
        )
    except Exception as e:
        om.release_lock(symbol, "SELL")
        log.error(f"safe_sell failed for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}


# ── Test runner ───────────────────────────────────────────────

if __name__ == "__main__":
    from colorama import Fore, Style, init
    init(autoreset=True)

    print(Fore.WHITE + Style.BRIGHT +
          "\n  Testing OrderManager...\n")

    om = OrderManager(trader=None)

    # test lock system
    print(Fore.WHITE + Style.BRIGHT +
          "=== Lock System ===")

    result = om.can_place_order("AAPL", "BUY")
    print(Fore.GREEN +
          f"  AAPL BUY allowed: {result['allowed']} "
          f"({result['reason']})")

    om.acquire_lock("AAPL", "BUY", "ORDER-001")
    result = om.can_place_order("AAPL", "BUY")
    print(Fore.YELLOW +
          f"  AAPL BUY after lock: {result['allowed']} "
          f"({result['reason']})")

    # different side — should be allowed
    result = om.can_place_order("AAPL", "SELL")
    print(Fore.GREEN +
          f"  AAPL SELL (different side): "
          f"{result['allowed']} ({result['reason']})")

    om.release_lock("AAPL", "BUY")
    result = om.can_place_order("AAPL", "BUY")
    print(Fore.GREEN +
          f"  AAPL BUY after release: {result['allowed']} "
          f"({result['reason']})")

    # test lock status
    print(Fore.WHITE + Style.BRIGHT +
          "\n=== Lock Status ===")
    status = om.get_lock_status()
    print(Fore.WHITE +
          f"  Active locks: {status['active_locks']}")

    # test reconciliation without trader
    print(Fore.WHITE + Style.BRIGHT +
          "\n=== Reconciliation (no trader) ===")
    result = om.reconcile_positions()
    print(Fore.YELLOW +
          f"  Result: {result}")

    print(Fore.GREEN + Style.BRIGHT +
          "\n  OrderManager working!\n")