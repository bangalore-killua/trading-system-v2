"""
tranche_manager.py — Pyramid up/down tranche position management.

Tranche strategy (from plan):
  T1 (33%): composite_score ≥ 0.35 → initial entry
  T2 (+33%): price confirms by 1.5×ATR above entry AND score still ≥ 0.35
  T3 (+34%): new high AND composite_score ≥ regime_threshold
  DIP tranche (0.8×-1.5×): dip_score ≥ 0.65 + fundamentals intact

Key asymmetry:
  Add-up tranches (T2, T3): tighter stop (1.0×ATR), follow the momentum
  Add-down tranches (dip): larger size, wider stop (3.0×ATR), held longer

Combined exposure check:
  Before any add: total_exposure[symbol] = short_pct + long_pct < 22%

Tranche state persists in trade_ledger (loaded on startup).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Regime thresholds for T3 entry (from plan)
T3_THRESHOLDS = {
    "EARLY_BULL":   0.58,
    "MID_BULL":     0.65,
    "LATE_BULL":    0.74,
    "RECOVERY":     0.60,
    "NEUTRAL":      0.65,
    "SIDEWAYS":     0.70,
    "CRASH":        9.99,  # effectively impossible — no T3 in crash
}

# ATR multipliers for stop placement
STOP_ATR_MULT = {
    "T1":  2.5,   # initial entry — standard stop
    "T2":  1.5,   # momentum add — tighter stop
    "T3":  1.0,   # high conviction — tightest stop
    "DIP": 3.0,   # dip tranche — wider stop, held longer
}

# Size ratios (of full Kelly position)
TRANCHE_SIZES = {
    "T1":  0.33,
    "T2":  0.33,
    "T3":  0.34,
    "DIP_SMALL":   0.80,   # 0.8× T1 — lower confidence dip
    "DIP_NORMAL":  1.00,   # 1.0× T1 — standard dip
    "DIP_LARGE":   1.50,   # 1.5× T1 — high confidence dip
}


@dataclass
class TrancheState:
    """State of one position's tranche progression."""
    symbol:         str
    entry_price:    float
    atr_14:         float
    entry_composite: float
    regime:         str
    tranches:       list = field(default_factory=list)
    dip_tranches:   list = field(default_factory=list)
    opened_at:      str  = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def current_tranche_level(self) -> int:
        return len(self.tranches)

    @property
    def total_shares(self) -> float:
        return sum(t.get("shares", 0) for t in self.tranches + self.dip_tranches)

    @property
    def avg_cost(self) -> float:
        total_cost   = sum(t.get("shares", 0) * t.get("price", 0) for t in self.tranches + self.dip_tranches)
        total_shares = self.total_shares
        return total_cost / total_shares if total_shares > 0 else 0.0


class TrancheManager:
    """Manages tranche entry and position scaling for a single position.

    Usage:
        tm = TrancheManager(portfolio_state)

        # Check T1
        action = tm.should_enter_t1(symbol, composite_score, atr, entry_price, regime)

        # Check T2 after T1 is open
        action = tm.should_add_t2(state, current_price, composite_score)

        # Check DIP tranche
        action = tm.should_add_dip(symbol, dip_score, dip_features, portfolio_pct)
    """

    def __init__(self, portfolio_state: Optional[dict] = None):
        self._portfolio = portfolio_state or {}
        self._positions: dict[str, TrancheState] = {}

    def should_enter_t1(
        self,
        symbol:          str,
        composite_score: float,
        atr:             float,
        entry_price:     float,
        regime:          str,
        kelly_pct:       float = 0.05,
    ) -> dict:
        """Evaluate T1 entry.

        Returns
        -------
        {"action": "enter_t1"|"skip", "size_pct": float, "stop_price": float, "reason": str}
        """
        # Hard gates
        if regime == "CRASH":
            return {"action": "skip", "size_pct": 0, "stop_price": 0, "reason": "crash_regime"}
        if self._portfolio.get("cash_pct", 1.0) < 0.10:
            return {"action": "skip", "size_pct": 0, "stop_price": 0, "reason": "cash_below_10pct"}
        if self._combined_exposure(symbol) >= 0.22:
            return {"action": "skip", "size_pct": 0, "stop_price": 0, "reason": "exposure_cap_22pct"}

        if composite_score < 0.35:
            return {"action": "skip", "size_pct": 0, "stop_price": 0, "reason": f"score_{composite_score:.3f}<0.35"}

        size_pct   = round(kelly_pct * TRANCHE_SIZES["T1"] * composite_score, 4)
        stop_price = round(entry_price - STOP_ATR_MULT["T1"] * atr, 4)
        return {
            "action":      "enter_t1",
            "size_pct":    size_pct,
            "stop_price":  stop_price,
            "reason":      f"composite={composite_score:.3f}≥0.35",
        }

    def should_add_t2(
        self,
        state:           TrancheState,
        current_price:   float,
        composite_score: float,
    ) -> dict:
        """Evaluate T2 add (requires T1 open, price confirmed above threshold)."""
        if state.current_tranche_level < 1:
            return {"action": "skip", "reason": "t1_not_open"}
        if state.current_tranche_level >= 2:
            return {"action": "skip", "reason": "t2_already_open"}

        t2_threshold = state.entry_price + 1.5 * state.atr_14
        if current_price < t2_threshold:
            return {"action": "skip", "reason": f"price_{current_price:.2f}<t2_threshold_{t2_threshold:.2f}"}
        if composite_score < 0.35:
            return {"action": "skip", "reason": f"score_{composite_score:.3f}<0.35"}
        if self._combined_exposure(state.symbol) >= 0.22:
            return {"action": "skip", "reason": "exposure_cap_22pct"}

        stop_price = round(current_price - STOP_ATR_MULT["T2"] * state.atr_14, 4)
        return {
            "action":     "add_t2",
            "size_pct":   round(state.entry_composite * TRANCHE_SIZES["T2"] * 0.05, 4),
            "stop_price": stop_price,
            "reason":     f"price confirmed +{1.5:.1f}×ATR",
        }

    def should_add_t3(
        self,
        state:           TrancheState,
        current_price:   float,
        composite_score: float,
        is_new_high:     bool,
    ) -> dict:
        """Evaluate T3 add (requires T2 open, new high, composite ≥ regime threshold)."""
        if state.current_tranche_level < 2:
            return {"action": "skip", "reason": "t2_not_open"}
        if state.current_tranche_level >= 3:
            return {"action": "skip", "reason": "t3_already_open"}
        if not is_new_high:
            return {"action": "skip", "reason": "no_new_high"}

        threshold = T3_THRESHOLDS.get(state.regime, 0.65)
        if composite_score < threshold:
            return {"action": "skip", "reason": f"score_{composite_score:.3f}<regime_threshold_{threshold}"}
        if self._combined_exposure(state.symbol) >= 0.22:
            return {"action": "skip", "reason": "exposure_cap_22pct"}

        stop_price = round(current_price - STOP_ATR_MULT["T3"] * state.atr_14, 4)
        return {
            "action":     "add_t3",
            "size_pct":   round(composite_score * TRANCHE_SIZES["T3"] * 0.05, 4),
            "stop_price": stop_price,
            "reason":     f"new_high + composite={composite_score:.3f}≥{threshold}",
        }

    def should_add_dip(
        self,
        symbol:          str,
        dip_score:       float,
        recommended_size: str,
        fundamentals_intact: bool,
        regime:          str,
    ) -> dict:
        """Evaluate dip tranche add (separate from T1/T2/T3 pyramid)."""
        if regime == "CRASH":
            return {"action": "skip", "reason": "crash_regime"}
        if not fundamentals_intact:
            return {"action": "skip", "reason": "fundamentals_broken"}
        if dip_score < 0.65:
            return {"action": "skip", "reason": f"dip_score_{dip_score:.3f}<0.65"}
        if self._combined_exposure(symbol) >= 0.22:
            return {"action": "skip", "reason": "exposure_cap_22pct"}

        size_key = f"DIP_{recommended_size.upper()}" if recommended_size != "skip" else "DIP_NORMAL"
        size_ratio = TRANCHE_SIZES.get(size_key, 1.0)

        return {
            "action":      "add_dip",
            "size_ratio":  size_ratio,
            "dip_score":   dip_score,
            "reason":      f"dip_score={dip_score:.3f}, size={size_key}",
            "notes":       "Dip tranches held longer, wider ATR stop",
        }

    def _combined_exposure(self, symbol: str) -> float:
        """Get combined short+long term exposure for symbol."""
        positions = self._portfolio.get("positions", {})
        return float(positions.get(symbol, {}).get("total_pct", 0.0))
