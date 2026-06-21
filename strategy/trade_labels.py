"""
trade_labels.py
Typed labels for every order placed by the system.
Stored in DB on every order, shown in Telegram alerts and dashboard.
"""

from enum import Enum


class TradeLabel(str, Enum):
    # ── Entry labels ──────────────────────────────────────────────
    INITIAL_BUY           = "INITIAL_BUY"
    DIP_BUY_T1            = "DIP_BUY_T1"       # Tier 1 accumulation
    DIP_BUY_T2            = "DIP_BUY_T2"       # Tier 2 accumulation
    SIGNAL_REENTRY        = "SIGNAL_REENTRY"   # re-enter after cooldown
    PREDICTED_DIP_SCALE   = "PREDICTED_DIP_SCALE"  # staggered bear entry

    # ── Exit labels ───────────────────────────────────────────────
    TAKE_PROFIT           = "TAKE_PROFIT"
    TRAILING_STOP         = "TRAILING_STOP"
    HARD_STOP_T3          = "HARD_STOP_T3"     # only Tier 3 flat stop
    SCORE_EXIT_FULL       = "SCORE_EXIT_FULL"
    SCORE_REDUCE          = "SCORE_REDUCE"
    EMERGENCY_STOP        = "EMERGENCY_STOP"   # catastrophic gap ≥ -50%


LABEL_EMOJI: dict[TradeLabel, str] = {
    TradeLabel.INITIAL_BUY:         "🟢",
    TradeLabel.DIP_BUY_T1:          "📈",
    TradeLabel.DIP_BUY_T2:          "📈",
    TradeLabel.SIGNAL_REENTRY:      "🔄",
    TradeLabel.PREDICTED_DIP_SCALE: "📊",
    TradeLabel.TAKE_PROFIT:         "✅",
    TradeLabel.TRAILING_STOP:       "🛑",
    TradeLabel.HARD_STOP_T3:        "🛑",
    TradeLabel.SCORE_EXIT_FULL:     "💀",
    TradeLabel.SCORE_REDUCE:        "🟠",
    TradeLabel.EMERGENCY_STOP:      "🚨",
}
