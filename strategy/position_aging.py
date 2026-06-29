"""
position_aging.py — AGE-1
Position aging + thesis review + partial profit system.

Addresses:
  - No systematic partial profit mechanism
  - Positions held indefinitely on stale thesis
  - No forced review at intervals

All thresholds in config.json — no hardcoded values.

Public API:
  check_position_aging(symbol, position, tier, entry_thesis) → dict
  get_partial_profit_action(setup_type, pnl_pct, days_held) → dict | None
"""

import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# Minimum hold before ANY exit signal is acted on — prevents Day-1 churn.
# This is live-system business logic, not a backtester parameter.
MIN_HOLD_DAYS: int = 5

# ── Config helpers ────────────────────────────────────────────────────────────

def _load_review_days() -> dict:
    """Returns thesis_review_days from config; falls back to safe defaults."""
    defaults = {"TIER_1": 90, "TIER_2": 60, "TIER_3": 30,
                "cyclical_inflection": 45}
    try:
        from strategy.config_loader import load_config
        return load_config().get("thesis_review_days", defaults)
    except Exception:
        return defaults


def _load_partial_thresholds() -> dict:
    """Returns partial_profit_thresholds from config."""
    defaults = {
        "SETUP_MOMENTUM":    [[15, 0.33, 7],  [30, 0.50, 7],  [50, 0.75, 14]],
        "SETUP_BOUNCE":      [[10, 0.33, 2],  [20, 0.50, 2]],
        "SETUP_INFLECTION":  [[25, 0.25, 14], [50, 0.25, 30]],
        "SETUP_NEWS_DRIVEN": [[8,  0.50, 0]],
    }
    try:
        from strategy.config_loader import load_config
        return load_config().get("partial_profit_thresholds", defaults)
    except Exception:
        return defaults


# ── Days-held calculation ─────────────────────────────────────────────────────

def _days_held(entry_date_str: Optional[str],
               now: Optional[datetime] = None) -> int:
    """Returns days since entry. Returns 0 if entry_date_str is None/unparseable."""
    if not entry_date_str:
        return 0
    if now is None:
        now = datetime.now()
    try:
        entry = datetime.strptime(str(entry_date_str)[:10], "%Y-%m-%d")
        return max(0, (now - entry).days)
    except Exception:
        return 0


# ── Partial profit logic ──────────────────────────────────────────────────────

def get_partial_profit_action(setup_type: str,
                              pnl_pct: float,
                              days_held: int) -> Optional[dict]:
    """
    Checks whether a partial profit should be taken based on current P&L,
    days held, and setup type.

    Threshold format per setup: [[pnl_target_pct, sell_fraction, min_days_held], ...]

    Returns the FIRST matching threshold action, or None if no threshold met.

    Returns:
    {
      pnl_target_pct:  float   — the threshold that triggered
      sell_fraction:   float   — portion of position to sell (0.33 = 33%)
      min_days_held:   int     — minimum days required
      action:          str     — "PARTIAL_PROFIT"
    }
    """
    thresholds = _load_partial_thresholds()
    tiers      = thresholds.get(setup_type, [])

    for (pnl_target, sell_fraction, min_days) in tiers:
        if pnl_pct >= pnl_target and days_held >= min_days:
            return {
                "pnl_target_pct": pnl_target,
                "sell_fraction":  sell_fraction,
                "min_days_held":  min_days,
                "action":         "PARTIAL_PROFIT",
            }
    return None


# ── Main aging check ──────────────────────────────────────────────────────────

def check_position_aging(symbol: str,
                         position,
                         tier: str,
                         entry_thesis: str,
                         pnl_pct: float = 0.0,
                         setup_type: str = "SETUP_MOMENTUM",
                         now: Optional[datetime] = None) -> dict:
    """
    Called in score_open_positions() for every open position.

    Args:
      symbol:        trading symbol
      position:      position object (has .entry_date, .shares, etc.)
      tier:          "TIER_1" | "TIER_2" | "TIER_3"
      entry_thesis:  "cyclical_inflection" or setup type string
      pnl_pct:       current unrealised P&L %
      setup_type:    "SETUP_MOMENTUM" | "SETUP_BOUNCE" | etc.
      now:           override for unit tests (default: datetime.now())

    Returns:
    {
      symbol:               str
      days_held:            int
      requires_review:      bool
      review_threshold_days:int
      days_to_next_review:  int   — 0 if review overdue
      partial_profit_action:dict|None
      thesis_status:        str   — "INTACT"|"REVIEW_DUE"|"OVERDUE"
    }
    """
    review_cfg = _load_review_days()

    # Normalise tier key (e.g. "tier1" → "TIER_1", "TIER_1" stays "TIER_1")
    tier_key = tier.upper().replace("TIER", "TIER_").replace("__", "_")
    if not tier_key.startswith("TIER_"):
        tier_key = f"TIER_{tier_key.replace('TIER', '').lstrip('_')}"

    # Special: cyclical inflection has its own review period
    if "inflection" in str(entry_thesis).lower():
        review_days = review_cfg.get("cyclical_inflection",
                                      review_cfg.get("TIER_2", 60))
    else:
        review_days = review_cfg.get(tier_key, review_cfg.get("TIER_3", 30))

    entry_date = getattr(position, "entry_date", None)
    held       = _days_held(entry_date, now=now)

    requires_review = held >= review_days
    days_to_next    = max(0, review_days - held)

    if requires_review:
        thesis_status = "OVERDUE" if held >= review_days * 1.5 else "REVIEW_DUE"
    else:
        thesis_status = "INTACT"

    partial_action = get_partial_profit_action(setup_type, pnl_pct, held)

    result = {
        "symbol":                symbol,
        "days_held":             held,
        "min_hold_passed":       held >= MIN_HOLD_DAYS,
        "requires_review":       requires_review,
        "review_threshold_days": review_days,
        "days_to_next_review":   days_to_next,
        "partial_profit_action": partial_action,
        "thesis_status":         thesis_status,
        "setup_type":            setup_type,
        "pnl_pct":               round(pnl_pct, 2),
    }
    log.debug(
        f"Aging check {symbol}: held={held}d "
        f"review={review_days}d "
        f"requires_review={requires_review} "
        f"partial={partial_action is not None}"
    )
    return result


# ── Class interface for run_server.py / PositionManager ──────────────────────

class PositionAging:
    """Class wrapper for PositionManager dependency injection."""

    def __init__(self, cfg: dict = None):
        self._cfg = cfg or {}

    def check_position_aging(self, symbol: str, pos_info, indicators: dict = None):
        """Returns (should_partial: bool, partial_pct: float)."""
        indicators = indicators or {}
        tier        = getattr(pos_info, "tier", None) or indicators.get("tier", "TIER_1")
        thesis      = getattr(pos_info, "entry_thesis", None) or indicators.get("entry_thesis", "")
        pnl_pct     = getattr(pos_info, "pnl_pct", None) or indicators.get("pnl_pct", 0.0)
        setup_type  = getattr(pos_info, "setup_type", None) or indicators.get("setup_type", "SETUP_MOMENTUM")

        result = check_position_aging(
            symbol=symbol,
            position=pos_info,
            tier=tier,
            entry_thesis=thesis,
            pnl_pct=float(pnl_pct),
            setup_type=setup_type,
        )
        action = result.get("partial_profit_action")
        if action and action.get("action") == "partial_sell":
            return True, float(action.get("sell_pct", 50))
        return False, 0.0
