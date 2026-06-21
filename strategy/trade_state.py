# trade_state.py
# v4.1 — Trade state manager with open positions tracking
#
# Two distinct responsibilities:
#
# 1. Open Positions State Machine (new in v4.1)
#    Tracks every open position independently with:
#      - Entry tier (locked at entry, can only degrade)
#      - Tier-specific exit parameters
#      - Accumulation tranches
#      - Monitoring flag (persists beyond watchlist removal)
#      - Weeks below threshold (for sustained exit detection)
#    Positions are scored even when removed from watchlist.
#
# 2. Last Trade Tracker (existing)
#    Records most recent trade per symbol for:
#      - Cooldown tracking
#      - Re-entry gate
#      - Dashboard trade state display

import os
import sys
import json
import math
from datetime import datetime, timedelta
from colorama import Fore, Style, init

init(autoreset=True)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'
))

# ── State file paths ──────────────────────────────────────────
_STATE_DIR  = os.path.join(
    os.path.dirname(__file__), '..', 'logs'
)
_STATE_PATH = os.path.join(_STATE_DIR, 'trade_state.json')
_POS_PATH   = os.path.join(_STATE_DIR, 'open_positions.json')

# ── Position ID counter ───────────────────────────────────────
_POS_COUNTER_PATH = os.path.join(
    _STATE_DIR, 'position_counter.json'
)

# ── Cooldown settings ─────────────────────────────────────────
DEFAULT_COOLDOWN_HOURS = 48
DEFAULT_COOLDOWN_DROP  = 0.03  # 3% drop required for re-entry


class OpenPosition:
    """
    Represents a single open position with full state.
    Each position has its own tier-based exit parameters
    that are locked at entry and can only tighten over time.
    """

    def __init__(self,
                 symbol:          str,
                 position_id:     str,
                 entry_date:      str,
                 entry_price:     float,
                 shares:          float,
                 entry_score:     float,
                 entry_tier_id:   str,
                 sell_threshold:  float,
                 stop_style:      str,
                 bundle_id:       str,
                 entry_reason:    str   = "signal_buy",
                 price_stop_pct:  float = None):

        self.symbol          = symbol
        self.position_id     = position_id
        self.entry_date      = entry_date
        self.entry_price     = entry_price
        self.shares          = shares
        self.entry_score     = entry_score
        self.entry_tier_id   = entry_tier_id
        self.current_tier_id = entry_tier_id  # can degrade
        self.sell_threshold  = sell_threshold
        self.stop_style      = stop_style
        self.price_stop_pct  = price_stop_pct
        self.bundle_id       = bundle_id
        self.entry_reason    = entry_reason

        # accumulation tracking
        self.tranches = [{
            "date":   entry_date,
            "price":  entry_price,
            "shares": shares,
        }]
        self.avg_entry = entry_price

        # trailing stop — peak price seen since entry
        self.peak_price    = entry_price

        # monitoring — survives watchlist removal
        self.monitoring    = True
        self.in_watchlist  = True
        self.last_scored   = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # sustained deterioration tracking
        self.weeks_below_threshold = 0
        self.consecutive_low_score = 0

        # tier degradation history
        self.tier_history = [{
            "date":    entry_date,
            "tier_id": entry_tier_id,
            "reason":  "entry",
        }]

    def add_tranche(self, price: float, shares: float,
                     date: str = None) -> None:
        """Adds an accumulation tranche to this position."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self.tranches.append({
            "date":   date,
            "price":  price,
            "shares": shares,
        })
        self.shares    += shares

        # recalculate average entry
        total_cost      = sum(
            t["price"] * t["shares"] for t in self.tranches
        )
        total_shares    = sum(t["shares"] for t in self.tranches)
        self.avg_entry  = round(
            total_cost / total_shares, 4
        ) if total_shares > 0 else price

    def degrade_tier(self, new_tier_id: str,
                      reason: str,
                      new_sell_threshold: float,
                      new_stop_style: str,
                      new_price_stop_pct: float = None) -> None:
        """
        Degrades the tier for this position.
        Can only go from T1→T2, T2→T3.
        Cannot upgrade.
        Updates exit parameters to match new tier.
        """
        self.current_tier_id  = new_tier_id
        self.sell_threshold   = new_sell_threshold
        self.stop_style       = new_stop_style
        self.price_stop_pct   = new_price_stop_pct

        self.tier_history.append({
            "date":    datetime.now().strftime("%Y-%m-%d"),
            "tier_id": new_tier_id,
            "reason":  reason,
        })

    def update_score_tracking(self,
                               current_score: float,
                               deterioration_threshold: float) -> None:
        """
        Updates sustained deterioration tracking.
        Called every time position is scored.
        """
        if current_score < deterioration_threshold:
            self.consecutive_low_score += 1
            # approximate weeks (score runs every 30min, ~336/week)
            # use scoring cycles not actual weeks for simplicity
            self.weeks_below_threshold = round(
                self.consecutive_low_score / 336, 1
            )
        else:
            self.consecutive_low_score = max(
                0, self.consecutive_low_score - 1
            )
            self.weeks_below_threshold = round(
                self.consecutive_low_score / 336, 1
            )

        self.last_scored = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def should_exit_by_deterioration(self,
                                      deterioration_weeks: float,
                                      deterioration_threshold: float
                                      ) -> bool:
        """
        Returns True if position should exit due to sustained
        score deterioration (used for Tier 1 fundamental exit).
        """
        return (self.weeks_below_threshold >= deterioration_weeks
                and self.consecutive_low_score > 0)

    def to_dict(self) -> dict:
        """Serialises position to dict for JSON storage."""
        return {
            "symbol":                 self.symbol,
            "position_id":            self.position_id,
            "entry_date":             self.entry_date,
            "entry_price":            self.entry_price,
            "shares":                 round(self.shares, 6),
            "entry_score":            self.entry_score,
            "entry_tier_id":          self.entry_tier_id,
            "current_tier_id":        self.current_tier_id,
            "sell_threshold":         self.sell_threshold,
            "stop_style":             self.stop_style,
            "price_stop_pct":         self.price_stop_pct,
            "bundle_id":              self.bundle_id,
            "entry_reason":           self.entry_reason,
            "tranches":               self.tranches,
            "avg_entry":              self.avg_entry,
            "peak_price":             round(self.peak_price, 4),
            "monitoring":             self.monitoring,
            "in_watchlist":           self.in_watchlist,
            "last_scored":            self.last_scored,
            "weeks_below_threshold":  self.weeks_below_threshold,
            "consecutive_low_score":  self.consecutive_low_score,
            "tier_history":           self.tier_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpenPosition":
        """Deserialises position from stored dict."""
        pos = cls(
            symbol         = data["symbol"],
            position_id    = data["position_id"],
            entry_date     = data["entry_date"],
            entry_price    = data["entry_price"],
            shares         = data["shares"],
            entry_score    = data["entry_score"],
            entry_tier_id  = data["entry_tier_id"],
            sell_threshold = data["sell_threshold"],
            stop_style     = data["stop_style"],
            bundle_id      = data["bundle_id"],
            entry_reason   = data.get("entry_reason", "signal_buy"),
            price_stop_pct = data.get("price_stop_pct"),
        )
        pos.current_tier_id       = data.get(
            "current_tier_id", data["entry_tier_id"]
        )
        pos.tranches               = data.get("tranches", [])
        pos.avg_entry              = data.get(
            "avg_entry", data["entry_price"]
        )
        pos.peak_price             = float(
            data.get("peak_price", data["entry_price"])
        )
        pos.monitoring             = data.get("monitoring", True)
        pos.in_watchlist           = data.get("in_watchlist", True)
        pos.last_scored            = data.get("last_scored", "")
        pos.weeks_below_threshold  = data.get(
            "weeks_below_threshold", 0
        )
        pos.consecutive_low_score  = data.get(
            "consecutive_low_score", 0
        )
        pos.tier_history           = data.get("tier_history", [])
        return pos


class TradeStateManager:
    """
    Manages two types of state:

    1. open_positions: OpenPosition objects for all live positions
       Used by scheduler to score positions even after they
       leave the watchlist. Tier-aware exit logic.

    2. _state: last trade per symbol (legacy)
       Used for dashboard display and cooldown checks.
    """

    def __init__(self):
        os.makedirs(_STATE_DIR, exist_ok=True)
        self._state     = self._load_state()
        self._positions = self._load_positions()

    # ── State persistence ──────────────────────────────────────

    def _load_state(self) -> dict:
        """Loads last-trade state from JSON."""
        try:
            if os.path.exists(_STATE_PATH):
                with open(_STATE_PATH) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save(self) -> None:
        """Saves last-trade state to JSON."""
        try:
            with open(_STATE_PATH, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            print(Fore.YELLOW + f"  State save failed: {e}")

    def _load_positions(self) -> dict:
        """Loads open positions from JSON."""
        try:
            if os.path.exists(_POS_PATH):
                with open(_POS_PATH) as f:
                    data = json.load(f)
                    return {
                        sym: OpenPosition.from_dict(pos)
                        for sym, pos in data.items()
                    }
        except Exception:
            pass
        return {}

    def _save_positions(self) -> None:
        """Saves open positions to JSON."""
        try:
            with open(_POS_PATH, "w") as f:
                json.dump(
                    {sym: pos.to_dict()
                     for sym, pos in self._positions.items()},
                    f, indent=2
                )
        except Exception as e:
            print(Fore.YELLOW +
                  f"  Positions save failed: {e}")

    def _next_position_id(self) -> str:
        """Generates next position ID: POS-XXXX"""
        try:
            counter = 0
            if os.path.exists(_POS_COUNTER_PATH):
                with open(_POS_COUNTER_PATH) as f:
                    counter = json.load(f).get("counter", 0)
            counter += 1
            with open(_POS_COUNTER_PATH, "w") as f:
                json.dump({"counter": counter}, f)
            return f"POS-{counter:04d}"
        except Exception:
            import random
            return f"POS-{random.randint(1000, 9999)}"

    # ── Open Position Management ───────────────────────────────

    def open_position(self, symbol: str,
                       entry_price: float,
                       shares: float,
                       entry_score: float,
                       scorecard: dict,
                       entry_reason: str = "signal_buy") -> OpenPosition:
        """
        Opens a new position with tier-based exit parameters.
        Called by alpaca_trader after a BUY order fills.

        Automatically determines tier from scorecard.
        Stores entry-locked parameters in position state.
        """
        from strategy.holding_tier import (
            get_holding_tier, TIER_CONFIG
        )

        bundle_id  = scorecard.get("bundle_id", "unknown")
        tier_cfg   = get_holding_tier(symbol, scorecard)
        tier_id    = tier_cfg["tier_id"]

        position = OpenPosition(
            symbol         = symbol,
            position_id    = self._next_position_id(),
            entry_date     = datetime.now().strftime("%Y-%m-%d"),
            entry_price    = entry_price,
            shares         = shares,
            entry_score    = entry_score,
            entry_tier_id  = tier_id,
            sell_threshold = tier_cfg["sell_threshold"],
            stop_style     = tier_cfg["stop_style"],
            bundle_id      = bundle_id,
            entry_reason   = entry_reason,
            price_stop_pct = tier_cfg.get("price_stop_pct"),
        )

        self._positions[symbol] = position
        self._save_positions()

        print(Fore.GREEN +
              f"  Position opened: {symbol} "
              f"{tier_cfg['tier_emoji']} {tier_cfg['tier_name']} "
              f"@ ${entry_price:.2f} "
              f"sell_threshold={tier_cfg['sell_threshold']}")

        return position

    def close_position(self, symbol: str,
                        exit_price: float,
                        exit_reason: str,
                        exit_score: float) -> dict:
        """
        Closes a position and moves it to trade history.
        Called after a SELL order fills.
        """
        if symbol not in self._positions:
            return {"status": "not_found", "symbol": symbol}

        pos = self._positions[symbol]
        pnl = round(
            (exit_price - pos.avg_entry) * pos.shares, 2
        )
        pnl_pct = round(
            (exit_price - pos.avg_entry) / pos.avg_entry * 100, 2
        )

        closed = {
            **pos.to_dict(),
            "exit_date":   datetime.now().strftime("%Y-%m-%d"),
            "exit_price":  exit_price,
            "exit_reason": exit_reason,
            "exit_score":  exit_score,
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
        }

        del self._positions[symbol]
        self._save_positions()

        # also update last-trade state for dashboard
        self.save_trade(
            symbol      = symbol,
            action      = "SELL",
            price       = exit_price,
            trade_id    = pos.position_id,
            exit_type   = exit_reason,
            entry_price = pos.avg_entry,
        )

        color = Fore.GREEN if pnl > 0 else Fore.RED
        print(color +
              f"  Position closed: {symbol} "
              f"@ ${exit_price:.2f} "
              f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) "
              f"[{exit_reason}]")

        return closed

    def add_tranche(self, symbol: str,
                     price: float, shares: float) -> dict:
        """
        Adds an accumulation tranche to an existing position.
        Only valid for Tier 1 and Tier 2 positions.
        """
        if symbol not in self._positions:
            return {
                "status": "not_found",
                "symbol": symbol
            }

        pos     = self._positions[symbol]
        max_t   = 5 if pos.current_tier_id == "tier1" else 2
        curr_t  = len(pos.tranches)

        if curr_t >= max_t:
            return {
                "status":  "max_tranches",
                "symbol":  symbol,
                "message": f"Already at max {max_t} tranches",
            }

        pos.add_tranche(price, shares)
        self._save_positions()

        print(Fore.GREEN +
              f"  Tranche added: {symbol} "
              f"tranche {curr_t + 1}/{max_t} "
              f"@ ${price:.2f} "
              f"avg_entry now ${pos.avg_entry:.2f}")

        return {
            "status":    "ok",
            "symbol":    symbol,
            "tranche":   curr_t + 1,
            "max":       max_t,
            "avg_entry": pos.avg_entry,
        }

    def update_peak_price(self, symbol: str,
                          current_price: float,
                          save: bool = True) -> None:
        """Updates peak price for trailing stop calculation.
        Pass save=False in batch loops; call save_positions_if_dirty() once after."""
        pos = self._positions.get(symbol)
        if pos and current_price > pos.peak_price:
            pos.peak_price = current_price
            self._dirty    = True
            if save:
                self._save_positions()

    def save_positions_if_dirty(self) -> None:
        """Writes open_positions.json only if peak prices changed — avoids
        up to 30 separate disk writes per cycle in a bull market."""
        if getattr(self, "_dirty", False):
            self._save_positions()
            self._dirty = False

    def check_tier_degradation(self,
                                symbol: str,
                                scorecard: dict) -> dict:
        """
        Checks if an open position's tier should degrade.
        Called weekly during bundle reclassification.

        Tiers can only degrade, never upgrade.
        Updates position parameters if degradation detected.
        """
        if symbol not in self._positions:
            return {"degraded": False, "symbol": symbol}

        from strategy.holding_tier import (
            check_tier_degradation as _check,
            TIER_CONFIG
        )

        pos    = self._positions[symbol]
        result = _check(
            symbol, pos.current_tier_id, scorecard
        )

        if result["degraded"]:
            new_cfg = result["config"]
            pos.degrade_tier(
                new_tier_id        = result["new_tier"],
                reason             = result["reason"],
                new_sell_threshold = new_cfg["sell_threshold"],
                new_stop_style     = new_cfg["stop_style"],
                new_price_stop_pct = new_cfg.get("price_stop_pct"),
            )
            self._save_positions()

            print(Fore.YELLOW +
                  f"  Tier degraded: {symbol} "
                  f"{result['old_tier']} → {result['new_tier']} "
                  f"({result['reason']})")

        return result

    def update_watchlist_flags(self,
                                current_watchlist: list) -> None:
        """
        Updates in_watchlist flag for all open positions.
        Called after watchlist updates.
        Positions with in_watchlist=False are still monitored
        by score_open_positions() in scheduler.
        """
        watchlist_set = set(current_watchlist)
        for symbol, pos in self._positions.items():
            was_in = pos.in_watchlist
            pos.in_watchlist = symbol in watchlist_set
            if was_in and not pos.in_watchlist:
                print(Fore.YELLOW +
                      f"  {symbol} left watchlist "
                      f"but position still monitored")
        self._save_positions()

    def get_all_open_positions(self) -> dict:
        """Returns all open positions as dict."""
        return dict(self._positions)

    def get_open_position(self, symbol: str):
        """Returns open position for symbol or None."""
        return self._positions.get(symbol)

    def has_open_position(self, symbol: str) -> bool:
        """Returns True if there is an open position."""
        return symbol in self._positions

    def get_symbols_to_monitor(self) -> list:
        """
        Returns all symbols with open positions.
        Includes symbols no longer in watchlist.
        Used by scheduler's score_open_positions() loop.
        """
        return list(self._positions.keys())

    def get_position_exit_params(self,
                                  symbol: str) -> dict:
        """
        Returns tier-based exit parameters for a position.
        Used by scheduler and alpaca_trader to determine
        when to exit.
        """
        pos = self._positions.get(symbol)
        if not pos:
            return {
                "sell_threshold": -20,
                "stop_style":     "atr",
                "price_stop_pct": None,
                "avg_entry":      0,
                "tier_id":        "tier3",
            }

        return {
            "sell_threshold":         pos.sell_threshold,
            "stop_style":             pos.stop_style,
            "price_stop_pct":         pos.price_stop_pct,
            "avg_entry":              pos.avg_entry,
            "tier_id":                pos.current_tier_id,
            "weeks_below_threshold":  pos.weeks_below_threshold,
        }

    def update_score_tracking(self, symbol: str,
                               current_score: float) -> None:
        """Updates sustained deterioration counter."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        from strategy.holding_tier import TIER_CONFIG
        tier_cfg = TIER_CONFIG.get(
            pos.current_tier_id, TIER_CONFIG["tier3"]
        )
        pos.update_score_tracking(
            current_score,
            tier_cfg["deterioration_threshold"]
        )
        self._save_positions()

    # ── Legacy last-trade tracking ─────────────────────────────

    def save_trade(self, symbol: str, action: str,
                    price: float, trade_id: str,
                    exit_type: str   = None,
                    entry_price: float = None) -> dict:
        """
        Records the most recent trade for a symbol.
        Used by dashboard and cooldown checks.
        """
        record = {
            "symbol":      symbol,
            "last_action": action.upper(),
            "last_price":  round(price, 4),
            "trade_id":    trade_id,
            "timestamp":   datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        if exit_type:
            record["exit_type"] = exit_type
        if entry_price:
            pnl_pct = round(
                (price - entry_price) / entry_price * 100, 2
            ) if action.upper() == "SELL" else 0
            record["pnl_pct"]    = pnl_pct
            record["entry_price"] = round(entry_price, 4)

        if action.upper() == "SELL":
            # tier-aware cooldown:
            # Tier 1/2: no cooldown — re-enter on valid signal
            # Tier 3:   48h cooldown — speculative protection
            # determine tier from open position if available
            pos = self._positions.get(symbol)
            tier_id = pos.current_tier_id if pos else "tier3"
            from strategy.config_loader import get_settings as _gs2
            _s = _gs2()
            cooldown_hours = {
                "tier1": int(_s.get("cooldown_hours_tier1", 0)),
                "tier2": int(_s.get("cooldown_hours_tier2", 0)),
                "tier3": int(_s.get("cooldown_hours_tier3", 48)),
            }.get(tier_id, int(_s.get("cooldown_hours_tier3", 48)))

            if cooldown_hours > 0:
                record["cooldown_until"] = (
                    datetime.now() + timedelta(hours=cooldown_hours)
                ).strftime("%Y-%m-%d %H:%M:%S")
            else:
                record["cooldown_until"] = None
            record["tier_id"] = tier_id

        self._state[symbol] = record
        self._save()
        # also persist to PostgreSQL
        try:
            from database import get_db
            get_db().save_trade({
                "symbol":        symbol,
                "action":        action.upper(),
                "price":         round(price, 4),
                "trade_id":      trade_id,
                "exit_reason":   exit_type,
                "entry_price":   entry_price,
                "tier_id":       self._positions.get(
                    symbol, type('obj', (object,), 
                    {'current_tier_id': None})()
                ).current_tier_id if symbol in self._positions 
                else None,
            })
        except Exception:
            pass
        return record

    def get_last_trade(self, symbol: str) -> dict:
        """Returns last trade record for a symbol."""
        return self._state.get(symbol)

    def get_all_last_trades(self) -> dict:
        """Returns all last trade records."""
        return dict(self._state)

    def can_reenter(self, symbol: str,
                     current_price: float) -> dict:
        """
        Checks if we can re-enter a symbol.
        Returns dict with can_enter bool and reason.
        """
        state = self._state.get(symbol)
        if not state:
            return {"can_enter": True, "reason": "no_prior_trade"}

        if state.get("last_action") != "SELL":
            return {"can_enter": True, "reason": "not_after_sell"}

        # cooldown check — None means no cooldown (Tier 1/2)
        cooldown_until = state.get("cooldown_until")
        if cooldown_until is None:
            return {"can_enter": True, "reason": "no_cooldown_tier12"}

        if cooldown_until:
            try:
                until = datetime.strptime(
                    cooldown_until, "%Y-%m-%d %H:%M:%S"
                )
                if datetime.now() < until:
                    remaining = round(
                        (until - datetime.now()).seconds / 3600, 1
                    )
                    return {
                        "can_enter": False,
                        "reason":    f"cooldown_{remaining}h",
                    }
            except Exception:
                pass

        # ── Price structure gate (CHANGE 9: Tier 1/2 bypass) ────
        # Tier 1/2: remove gate entirely — re-enter at any price if
        #           score is positive (quality positions worth holding).
        # Tier 3:   keep the gate to prevent buying broken structures.
        tier_id   = state.get("tier_id", "tier3")
        if tier_id in ("tier1", "tier2"):
            return {"can_enter": True, "reason": "clear_tier12_no_gate"}

        from strategy.config_loader import get_execution_config
        exec_cfg   = get_execution_config()
        exit_type  = state.get("exit_type", "")
        last_price = float(state.get("last_price", 0))

        if (exit_type == "hard_stop"
                and last_price > 0
                and current_price > 0
                and current_price >= last_price):
            threshold = last_price * float(
                exec_cfg["post_stop_reentry_pct"]
            )
            return {
                "can_enter": False,
                "reason": (
                    f"post_stop_structure_gate: "
                    f"price ${current_price:.2f} has not "
                    f"retraced below ${threshold:.2f} "
                    f"(stop-out was ${last_price:.2f})"
                ),
            }

        if (exit_type == "trailing_stop"
                and last_price > 0
                and current_price > 0):
            threshold = last_price * float(
                exec_cfg["post_trail_reentry_pct"]
            )
            if current_price >= threshold:
                return {
                    "can_enter": False,
                    "reason": (
                        f"post_trailing_stop_gate: "
                        f"price ${current_price:.2f} has not "
                        f"retraced below ${threshold:.2f}"
                    ),
                }

        return {"can_enter": True, "reason": "clear"}

    def is_in_cooldown(self, symbol: str) -> bool:
        """Returns True if symbol is in cooldown."""
        result = self.can_reenter(symbol, 0)
        return not result["can_enter"]

    def get_cooldown_remaining(self, symbol: str) -> float:
        """Returns hours remaining in cooldown (0 if none)."""
        state = self._state.get(symbol)
        if not state:
            return 0

        cooldown_until = state.get("cooldown_until")
        if cooldown_until:
            try:
                until = datetime.strptime(
                    cooldown_until, "%Y-%m-%d %H:%M:%S"
                )
                remaining = (
                    until - datetime.now()
                ).total_seconds() / 3600
                return max(0, round(remaining, 1))
            except Exception:
                pass
        return 0

    def get_summary(self) -> dict:
        """Returns summary for dashboard display."""
        summary = {}
        for symbol, state in self._state.items():
            pos = self._positions.get(symbol)
            summary[symbol] = {
                **state,
                "has_open_position": pos is not None,
                "tier_id": pos.current_tier_id if pos else None,
                "avg_entry": pos.avg_entry if pos else None,
                "tranches": len(pos.tranches) if pos else 0,
                "in_watchlist": pos.in_watchlist if pos else True,
            }
        return summary


# ── Test runner ───────────────────────────────────────────────

if __name__ == "__main__":
    print(Fore.WHITE + Style.BRIGHT +
          "\n  Testing TradeStateManager...\n")

    ts = TradeStateManager()

    # test open positions
    print(Fore.WHITE + Style.BRIGHT +
          "=== Open Positions ===")

    all_pos = ts.get_all_open_positions()
    if all_pos:
        for sym, pos in all_pos.items():
            print(Fore.WHITE +
                  f"  {sym:<12} "
                  f"{pos.current_tier_id:<8} "
                  f"avg_entry:${pos.avg_entry:.2f} "
                  f"tranches:{len(pos.tranches)} "
                  f"monitoring:{pos.monitoring} "
                  f"in_watchlist:{pos.in_watchlist}")
    else:
        print(Fore.WHITE + "  No open positions")

    # test last trades
    print(Fore.WHITE + Style.BRIGHT +
          "\n=== Last Trades ===")
    summary = ts.get_summary()
    if summary:
        for sym, state in list(summary.items())[:5]:
            cooldown = ts.get_cooldown_remaining(sym)
            print(Fore.WHITE +
                  f"  {sym:<12} "
                  f"{state.get('last_action','?'):<5} "
                  f"${state.get('last_price', 0):.2f} "
                  f"cooldown:{cooldown:.1f}h "
                  f"tier:{state.get('tier_id','?')}")
    else:
        print(Fore.WHITE + "  No trade history")

    # test monitoring list
    to_monitor = ts.get_symbols_to_monitor()
    print(Fore.WHITE + Style.BRIGHT +
          f"\n=== Symbols to Monitor: {len(to_monitor)} ===")
    for sym in to_monitor:
        params = ts.get_position_exit_params(sym)
        print(Fore.WHITE +
              f"  {sym:<12} "
              f"sell_threshold:{params['sell_threshold']} "
              f"stop:{params['stop_style']} "
              f"tier:{params['tier_id']}")

    print(Fore.GREEN + Style.BRIGHT +
          "\n  TradeStateManager working!\n")