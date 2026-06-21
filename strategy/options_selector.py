"""
options_selector.py — Option Contract Selection Logic

Given a trade signal and portfolio context, selects the optimal contract
for each of the three Phase 4 strategies:

  LONG CALL (bullish entry with P(win) ≥ 0.65):
    - ATM call (target delta 0.40–0.60), DTE 25–50 days
    - Budget: options.long_call_budget_pct × position_notional
    - Only fires when LightGBM meta-model is in high-confidence mode

  PROTECTIVE PUT (risk hedge):
    - OTM put 5–10% below current price, DTE 30–60 days
    - Sized to cover options.protective_put_cover_pct of position value
    - Fires automatically when regime moves to LATE_BULL or EARLY_BEAR

  COVERED CALL (income after AGE-1 partial exit):
    - OTM call 5–10% above current price, DTE 14–21 days (fast theta decay)
    - Quantity = shares held (1 contract per 100 shares, rounded down)
    - Only fires when we hold equity and AGE-1 has already triggered

All three strategies check options.enabled before proceeding and gracefully
return None when no suitable contract is found.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from data.options_chain import OptionContract, OptionsChainFetcher

logger = logging.getLogger(__name__)


@dataclass
class OptionTrade:
    """Everything needed to place and track an option position."""
    contract: OptionContract
    strategy: str           # "long_call" | "protective_put" | "covered_call"
    qty: int                # number of contracts
    max_premium: float      # per-contract cost cap (mid price ceiling)
    notional: float         # total cost = qty × contract.mid × 100


class OptionsSelector:
    """Selects optimal option contracts for Phase 4 strategies.

    Wraps OptionsChainFetcher with strategy-specific logic.
    All threshold values come from config.json options section.
    """

    def __init__(self, chain_fetcher: OptionsChainFetcher, config: dict):
        self._fetcher = chain_fetcher
        self._cfg = config.get("options", {})

    # ── Public API ───────────────────────────────────────────────────────────

    def select_long_call(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        position_notional: float,
    ) -> Optional[OptionTrade]:
        """Select ATM call for a bullish entry with high LightGBM confidence.

        Returns None if options are disabled, no suitable contract exists,
        or budget is too small for even 1 contract.
        """
        if not self._cfg.get("enabled", False):
            return None
        if "long_call" not in self._cfg.get("strategies_enabled", []):
            return None

        target_delta = (
            self._cfg.get("long_call_target_delta_min", 0.40) +
            self._cfg.get("long_call_target_delta_max", 0.60)
        ) / 2.0
        min_dte = self._cfg.get("long_call_min_dte", 25)
        max_dte = self._cfg.get("long_call_max_dte", 50)
        target_dte = (min_dte + max_dte) // 2

        contract = self._fetcher.get_atm_contract(
            symbol=symbol,
            option_type="call",
            current_price=current_price,
            target_delta=target_delta,
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if contract is None:
            return None

        # Budget: long_call_budget_pct of the equity position notional
        budget = position_notional * self._cfg.get("long_call_budget_pct", 0.005)
        qty = int(budget / (contract.mid * 100))
        if qty < 1:
            logger.debug(
                "SKIP long_call %s — budget $%.0f < 1 contract cost $%.0f",
                symbol, budget, contract.mid * 100,
            )
            return None

        # Max exposure guard: options can't exceed options.max_options_exposure_pct
        max_options_dollars = portfolio_value * self._cfg.get("max_options_exposure_pct", 10) / 100
        notional = qty * contract.mid * 100
        if notional > max_options_dollars:
            qty = max(1, int(max_options_dollars / (contract.mid * 100)))
            notional = qty * contract.mid * 100

        logger.info(
            "LONG CALL selected: %s strike=%.2f DTE=%d delta=%.2f "
            "qty=%d cost=$%.0f",
            contract.symbol, contract.strike, contract.days_to_expiry,
            contract.delta or 0, qty, notional,
        )
        return OptionTrade(
            contract=contract,
            strategy="long_call",
            qty=qty,
            max_premium=contract.mid * 1.05,  # allow 5% slippage from mid
            notional=notional,
        )

    def select_protective_put(
        self,
        symbol: str,
        current_price: float,
        position_value: float,
        portfolio_value: float,
    ) -> Optional[OptionTrade]:
        """Select OTM put to hedge an existing equity position.

        Strike is placed protective_put_otm_pct below current price.
        DTE window comes from config (default 30–60 days).
        """
        if not self._cfg.get("enabled", False):
            return None
        if "protective_put" not in self._cfg.get("strategies_enabled", []):
            return None

        otm_pct = self._cfg.get("protective_put_otm_pct", 7.0)
        target_strike = current_price * (1 - otm_pct / 100)
        min_dte = self._cfg.get("protective_put_min_dte", 30)
        max_dte = self._cfg.get("protective_put_max_dte", 60)
        target_dte = (min_dte + max_dte) // 2

        contract = self._fetcher.get_atm_contract(
            symbol=symbol,
            option_type="put",
            current_price=target_strike,   # treat target strike as "ATM" for selection
            target_delta=-0.30,            # OTM put ~30 delta
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if contract is None:
            return None

        # Buy enough puts to cover the position value (1 contract = 100 shares notional)
        shares_equivalent = int(position_value / current_price)
        qty = max(1, shares_equivalent // 100)

        notional = qty * contract.mid * 100
        max_options_dollars = portfolio_value * self._cfg.get("max_options_exposure_pct", 10) / 100
        if notional > max_options_dollars:
            qty = max(1, int(max_options_dollars / (contract.mid * 100)))
            notional = qty * contract.mid * 100

        logger.info(
            "PROTECTIVE PUT selected: %s strike=%.2f DTE=%d qty=%d cost=$%.0f",
            contract.symbol, contract.strike, contract.days_to_expiry, qty, notional,
        )
        return OptionTrade(
            contract=contract,
            strategy="protective_put",
            qty=qty,
            max_premium=contract.mid * 1.05,
            notional=notional,
        )

    def select_covered_call(
        self,
        symbol: str,
        current_price: float,
        shares_held: int,
    ) -> Optional[OptionTrade]:
        """Select OTM call to write against an existing equity position (income).

        Quantity limited by: floor(shares_held / 100) contracts.
        Strike placed covered_call_otm_pct above current price.
        Short DTE (14–21 days) for faster theta capture.
        """
        if not self._cfg.get("enabled", False):
            return None
        if "covered_call" not in self._cfg.get("strategies_enabled", []):
            return None

        max_contracts = shares_held // 100
        if max_contracts < 1:
            logger.debug("SKIP covered_call %s — fewer than 100 shares held", symbol)
            return None

        otm_pct = self._cfg.get("covered_call_otm_pct", 7.0)
        target_strike = current_price * (1 + otm_pct / 100)
        min_dte = self._cfg.get("covered_call_min_dte", 14)
        max_dte = self._cfg.get("covered_call_max_dte", 21)

        contract = self._fetcher.get_atm_contract(
            symbol=symbol,
            option_type="call",
            current_price=target_strike,
            target_delta=0.25,   # OTM call, low delta = lower assignment risk
            target_dte=(min_dte + max_dte) // 2,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if contract is None:
            return None

        notional = max_contracts * contract.mid * 100
        logger.info(
            "COVERED CALL selected: %s strike=%.2f DTE=%d qty=%d premium_collected=$%.0f",
            contract.symbol, contract.strike, contract.days_to_expiry,
            max_contracts, notional,
        )
        return OptionTrade(
            contract=contract,
            strategy="covered_call",
            qty=max_contracts,
            max_premium=contract.mid * 0.95,  # sell at 5% below mid (get filled quickly)
            notional=notional,
        )
