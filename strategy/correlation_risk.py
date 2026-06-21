"""
correlation_risk.py — CORR-1
Portfolio sector concentration check.

Prevents silent sector bets: NVDA + AMD + INTC + MU = 4 positions
but really just 1 semiconductor bet.

All sector assignments come from config.json["sector_assignments"].
No hardcoded symbols anywhere in this file.

Public API:
  get_sector_concentration(symbol, open_positions) → dict
  get_size_penalty(existing_count)                 → float
"""

import logging

log = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_corr_config() -> dict:
    defaults = {
        "max_same_sector_positions":           3,
        "sector_concentration_size_penalties": {"1": 1.00, "2": 0.80, "3": 0.60},
        "sector_assignments":                  {},
    }
    try:
        from strategy.config_loader import load_config
        cfg = load_config()
        return {
            "max_same":          cfg.get("max_same_sector_positions", 3),
            "size_penalties":    cfg.get("sector_concentration_size_penalties",
                                         defaults["sector_concentration_size_penalties"]),
            "sector_assignments":cfg.get("sector_assignments", {}),
        }
    except Exception:
        return {
            "max_same":       3,
            "size_penalties": defaults["sector_concentration_size_penalties"],
            "sector_assignments": {},
        }


# ── Core functions ────────────────────────────────────────────────────────────

def get_sector_concentration(symbol: str,
                              open_positions: dict) -> dict:
    """
    Checks how many existing open positions are in the same sector as `symbol`.

    Uses config["sector_assignments"] exclusively — no hardcoded symbols.
    Adding a new symbol: add one line to sector_assignments in config.json.

    Args:
      symbol:         candidate symbol (the one we want to enter)
      open_positions: dict of currently open positions {symbol: position_obj}
                      (or any iterable of symbol strings)

    Returns:
    {
      symbol:          str
      sector:          str    — sector name from sector_assignments ("unknown" if unset)
      existing_same:   list   — symbols in open_positions with the same sector
      existing_count:  int    — len(existing_same)
      max_allowed:     int    — from config (default 3)
      entry_blocked:   bool   — True when existing_count >= max_allowed
      size_penalty:    float  — multiplier to apply to position size (1.0 / 0.8 / 0.6)
    }
    """
    cfg        = _load_corr_config()
    assignments = cfg["sector_assignments"]
    max_same    = cfg["max_same"]
    penalties   = cfg["size_penalties"]

    target_sector = assignments.get(str(symbol).upper(), "unknown")

    position_keys = (
        open_positions.keys()
        if isinstance(open_positions, dict)
        else list(open_positions)
    )

    existing_same = [
        s for s in position_keys
        if s != symbol and assignments.get(str(s).upper()) == target_sector
    ]
    existing_count = len(existing_same)
    entry_blocked  = existing_count >= max_same

    # Size penalty: key is existing_count as string
    # 0 existing → 1.00, 1 existing → 0.80, 2 existing → 0.60, 3+ → blocked
    penalty_key  = str(min(existing_count, max(int(k) for k in penalties)))
    size_penalty = float(penalties.get(penalty_key, 1.0))

    log.debug(
        f"Sector concentration {symbol} [{target_sector}]: "
        f"{existing_count}/{max_same} existing {existing_same} "
        f"blocked={entry_blocked} penalty={size_penalty}"
    )

    return {
        "symbol":         symbol,
        "sector":         target_sector,
        "existing_same":  existing_same,
        "existing_count": existing_count,
        "max_allowed":    max_same,
        "entry_blocked":  entry_blocked,
        "size_penalty":   size_penalty,
    }


def get_size_penalty(existing_count: int) -> float:
    """
    Returns size multiplier for a given number of same-sector positions already held.
    Reads from config — no hardcoded values.
    """
    cfg      = _load_corr_config()
    penalties = cfg["size_penalties"]
    max_key   = max(int(k) for k in penalties)
    key       = str(min(existing_count, max_key))
    return float(penalties.get(key, 1.0))
