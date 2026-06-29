# holding_tier.py
# v4.1 — Position holding tier classification
#
# Tiers determine EXIT behaviour only.
# Bundles determine ENTRY analysis and position sizing.
# They are completely independent axes:
#
#   Bundle answers: HOW does this asset move?
#   Tier answers:   HOW certain is this company's future?
#
# Three tiers:
#   Tier 1 — Institutional Grade
#     Companies so embedded in the global economy that
#     their disappearance requires civilisational events.
#     AAPL, MSFT, GOOGL, NVDA, SPY, QQQ, GLD
#     Exit: fundamental breakdown only, no price stop
#     Accumulate: aggressively on dips (up to 5 tranches)
#
#   Tier 2 — Growth Quality
#     Strong companies dependent on continued execution.
#     BTC, ETH, PLTR, COIN, SHOP, ARM
#     Exit: -30 score threshold + 15% price stop
#     Accumulate: conservatively (1 extra tranche)
#
#   Tier 3 — Speculative
#     High potential but uncertain future.
#     Small caps, altcoins, recent IPOs, declining revenue
#     Exit: current behaviour (-20 + ATR stop)
#     Accumulate: never — capital preservation first
#
# Tier is:
#   Locked at entry time in position state
#   Can degrade over time (T1→T2, T2→T3) if company deteriorates
#   Cannot upgrade (T2→T1) for existing position
#   New positions get fresh classification
#
# This prevents selling on pullbacks for Tier 1 companies
# while maintaining protection for speculative positions.

import os
import sys
from colorama import Fore, Style, init

init(autoreset=True)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'
))

# ── Tier Configuration ────────────────────────────────────────

TIER_CONFIG = {
    "tier1": {
        "tier_id":            "tier1",
        "tier_name":          "Institutional Grade",
        "tier_emoji":         "⭐",
        "tier_description":   "Hold forever — exit on fundamental breakdown only",

        # exit parameters
        "sell_threshold":     -80,    # requires catastrophic multi-factor breakdown
        "stop_style":         "fundamental",  # no price stop
        "emergency_stop_pct": 0.50,   # only on -50% catastrophic gap (fraud etc)
        "take_profit_pct":    None,   # no take profit — let winners run indefinitely

        # sustained deterioration exit
        # if score stays below -60 for this many weeks → force exit
        "deterioration_weeks": 8,
        "deterioration_threshold": -60,

        # accumulation
        "accumulate_on_dips":  True,
        "accumulate_min_dip":  0.05,  # price must be 5%+ below avg entry
        "max_tranches":        5,

        # trend filter — very wide for Tier 1
        # only sell if score is catastrophically negative
        "trend_filter_active": True,
        "trend_sell_override": -80,   # score below this → sell regardless of trend
    },

    "tier2": {
        "tier_id":            "tier2",
        "tier_name":          "Growth Quality",
        "tier_emoji":         "🔷",
        "tier_description":   "Hold with protection — 15% stop, accumulate cautiously",

        # exit parameters
        "sell_threshold":     -30,
        "stop_style":         "price_pct",  # percentage from avg entry
        "price_stop_pct":     0.15,         # 15% from average entry
        "take_profit_pct":    0.30,         # 30% take profit

        # sustained deterioration exit
        "deterioration_weeks": 4,
        "deterioration_threshold": -40,

        # accumulation
        "accumulate_on_dips":  True,
        "accumulate_min_dip":  0.08,  # price must be 8%+ below avg entry
        "max_tranches":        2,

        # trend filter
        "trend_filter_active": True,
        "trend_sell_override": -50,   # score below this → sell regardless of trend
    },

    "tier3": {
        "tier_id":            "tier3",
        "tier_name":          "Speculative",
        "tier_emoji":         "⚡",
        "tier_description":   "Active management — ATR stops, no averaging down",

        # exit parameters — mirrors current system behaviour
        "sell_threshold":     -20,
        "stop_style":         "atr",     # ATR-based per bundle (existing logic)
        "take_profit_pct":    0.20,      # 20% take profit (bundle-specific override)

        # sustained deterioration exit
        "deterioration_weeks": 2,
        "deterioration_threshold": -30,

        # accumulation
        "accumulate_on_dips":  False,   # never average down on speculative
        "max_tranches":        1,

        # trend filter
        "trend_filter_active": True,
        "trend_sell_override": -20,     # sell at standard threshold
    },
}

# ── Always-fixed tier overrides ───────────────────────────────
# These symbols always get a specific tier regardless of
# fundamental scoring. Immutable.

ALWAYS_TIER1 = {
    # Index ETFs — market itself, cannot go to zero
    "SPY", "QQQ", "IWM", "DIA", "VOO", "IVV",
    # Sector ETFs — broad diversification
    "XLK", "XLF", "XLV", "XLE", "XLY", "XLI",
    "XLRE", "XLB", "SMH",
    # Metals — store of value, institutional adoption
    "GLD", "SLV", "IAU",
}

ALWAYS_TIER2 = {
    # Bitcoin and Ethereum — institutional adoption growing
    # but regulatory risk means not full Tier 1
    "BTC-USD", "ETH-USD",
}

# ── Mega cap threshold ────────────────────────────────────────
# Approximate market cap proxy via bundle + fundamental score
# True market cap would require an API call — we approximate
# B3 Structured Mega Cap + strong fundamentals → likely Tier 1

TIER1_BUNDLES = {
    "B3_STRUCTURED_MEGA_CAP",
    "B4_DEFENSIVE_INCOME",
}

TIER1_ROE_MIN     = 0.6   # ROE score must be >= 0.6 (good vs sector)
TIER1_REV_MIN     = 0.0   # Revenue growth score must be >= 0 (not declining)
TIER1_FUND_PCT    = 0.55  # Fundamental score as % of max must be >= 55%


def get_holding_tier(symbol: str,
                     scorecard: dict = None) -> dict:
    """
    Determines the holding tier for a symbol.

    Called at BUY time to set position-level exit parameters.
    Also called weekly to check for tier degradation.

    Priority order:
    1. Always-fixed overrides (SPY, GLD, BTC etc)
    2. Crypto (non BTC/ETH) → always Tier 3
    3. Dynamic from scorecard fundamentals

    Args:
        symbol:    ticker symbol e.g. "AAPL", "BTC-USD"
        scorecard: output of scorer.score() — used for
                   fundamental quality assessment.
                   If None, defaults to Tier 3 (conservative)

    Returns:
        TIER_CONFIG dict for the assigned tier
    """

    # ── Always-fixed overrides ────────────────────────────────
    if symbol in ALWAYS_TIER1:
        return _with_symbol(TIER_CONFIG["tier1"], symbol)

    if symbol in ALWAYS_TIER2:
        return _with_symbol(TIER_CONFIG["tier2"], symbol)

    # ── Crypto (non BTC/ETH) always Tier 3 ───────────────────
    if _is_crypto(symbol):
        return _with_symbol(TIER_CONFIG["tier3"], symbol)

    # ── No scorecard available → conservative default ─────────
    if scorecard is None:
        return _with_symbol(TIER_CONFIG["tier3"], symbol)

    # ── Dynamic classification from scorecard ─────────────────
    bundle_id  = scorecard.get("bundle_id", "")
    factors    = scorecard.get("all_factors", {})
    fund_data  = scorecard.get("fundamental", {})

    roe_score  = factors.get("roe",  {}).get("score", 0)
    rev_score  = factors.get("revenue_growth", {}).get("score", 0)
    fund_score = fund_data.get("score", 0)
    fund_max   = fund_data.get("max", 40)
    fund_pct   = fund_score / fund_max if fund_max > 0 else 0

    is_tier1_bundle = bundle_id in TIER1_BUNDLES

    # Tier 1: mega cap bundle + strong fundamentals
    if (is_tier1_bundle and
            roe_score >= TIER1_ROE_MIN and
            rev_score >= TIER1_REV_MIN and
            fund_pct  >= TIER1_FUND_PCT):
        return _with_symbol(TIER_CONFIG["tier1"], symbol)

    # Tier 1: exceptional fundamentals regardless of bundle
    # (small company with outstanding metrics can qualify)
    if (roe_score >= 0.9 and
            rev_score >= 0.5 and
            fund_pct >= 0.70):
        return _with_symbol(TIER_CONFIG["tier1"], symbol)

    # Tier 3: poor fundamentals or declining
    if (rev_score <= -0.4 or
            roe_score <= -0.6 or
            fund_pct < 0.30):
        return _with_symbol(TIER_CONFIG["tier3"], symbol)

    # Tier 3: no fundamental data at all
    if fund_max == 0 or fund_score == 0:
        return _with_symbol(TIER_CONFIG["tier3"], symbol)

    # Everything else: Tier 2
    return _with_symbol(TIER_CONFIG["tier2"], symbol)


def check_tier_degradation(symbol: str,
                             current_tier_id: str,
                             scorecard: dict) -> dict:
    """
    Checks if an existing position's tier should degrade.
    Called weekly on all open positions.

    Tiers can only degrade (T1→T2, T2→T3), never upgrade.
    This protects against holding deteriorating companies
    with the wrong (too loose) exit parameters.

    Returns:
        {
          "degraded": bool,
          "old_tier": "tier1",
          "new_tier": "tier2",
          "reason":   "ROE dropped below sector",
          "config":   TIER_CONFIG dict for new tier,
        }
    """
    new_tier_config = get_holding_tier(symbol, scorecard)
    new_tier_id     = new_tier_config["tier_id"]

    tier_rank = {"tier1": 1, "tier2": 2, "tier3": 3}
    current_rank = tier_rank.get(current_tier_id, 3)
    new_rank     = tier_rank.get(new_tier_id, 3)

    if new_rank > current_rank:
        # tier has degraded
        factors   = scorecard.get("all_factors", {})
        roe_score = factors.get("roe",  {}).get("score", 0)
        rev_score = factors.get("revenue_growth", {}).get("score", 0)

        reasons = []
        if roe_score < TIER1_ROE_MIN:
            reasons.append(f"ROE score {roe_score:.2f} below threshold")
        if rev_score < TIER1_REV_MIN:
            reasons.append(f"Revenue score {rev_score:.2f} — declining")
        if not reasons:
            reasons.append("Fundamental quality declined")

        return {
            "degraded": True,
            "old_tier": current_tier_id,
            "new_tier": new_tier_id,
            "reason":   "; ".join(reasons),
            "config":   new_tier_config,
        }

    return {
        "degraded": False,
        "old_tier": current_tier_id,
        "new_tier": current_tier_id,
        "reason":   "No degradation",
        "config":   TIER_CONFIG[current_tier_id],
    }


def get_tier_label(tier_id: str) -> str:
    """Returns human-readable tier label."""
    cfg = TIER_CONFIG.get(tier_id, TIER_CONFIG["tier3"])
    return f"{cfg['tier_emoji']} {cfg['tier_name']}"


# ── Helpers ───────────────────────────────────────────────────

def _is_crypto(symbol: str) -> bool:
    """Returns True if symbol is a cryptocurrency."""
    return any(symbol.endswith(s)
               for s in ["-USD", "-USDT", "-BTC", "-ETH"])


def _with_symbol(config: dict, symbol: str) -> dict:
    """Returns tier config with symbol added for context."""
    return {**config, "symbol": symbol}


# ── Test runner ───────────────────────────────────────────────

if __name__ == "__main__":
    print(Fore.WHITE + Style.BRIGHT +
          "\n  Testing HoldingTier...\n")

    # test without scorecard (conservative defaults)
    test_symbols = [
        "AAPL", "MSFT", "BTC-USD", "ETH-USD",
        "SPY", "GLD", "COIN", "PLTR",
        "XRP-USD", "DOGE-USD",
    ]

    print(Fore.WHITE + Style.BRIGHT +
          "=== Tier Classification (no scorecard) ===")
    for sym in test_symbols:
        tier = get_holding_tier(sym)
        print(Fore.WHITE +
              f"  {sym:<12} "
              f"{tier['tier_emoji']} {tier['tier_name']:<22} "
              f"sell:{tier['sell_threshold']:>4}  "
              f"stop:{tier['stop_style']}")

    print(Fore.WHITE + Style.BRIGHT +
          "\n=== Tier Config Detail ===")
    for tier_id, cfg in TIER_CONFIG.items():
        print(Fore.WHITE + Style.BRIGHT +
              f"\n  {cfg['tier_emoji']} {cfg['tier_name']} ({tier_id})")
        print(Fore.WHITE +
              f"    Sell threshold:    {cfg['sell_threshold']}")
        print(Fore.WHITE +
              f"    Stop style:        {cfg['stop_style']}")
        print(Fore.WHITE +
              f"    Accumulate:        {cfg['accumulate_on_dips']}")
        print(Fore.WHITE +
              f"    Max tranches:      {cfg['max_tranches']}")
        print(Fore.WHITE +
              f"    Deterioration:     "
              f"{cfg['deterioration_weeks']} weeks below "
              f"{cfg['deterioration_threshold']}")

    print(Fore.GREEN + Style.BRIGHT +
          "\n  HoldingTier working!\n")