"""
options_chain.py — Alpaca Options Chain Fetcher

Fetches live option chain snapshots via alpaca-py's OptionHistoricalDataClient.
Filters contracts by type, DTE window, strike proximity, min OI, and min volume.
Results are cached for 5 minutes per (symbol, option_type) pair.

Snapshots include bid/ask, Greeks (delta, gamma, theta, vega), and IV when
the underlying has listed options on Alpaca (US equities only).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache TTL — option quotes don't need sub-second freshness for entry decisions
_CACHE_TTL_SECONDS = 300


@dataclass
class OptionContract:
    """A single option contract with market data."""
    symbol: str               # OCC symbol e.g. "AAPL240119C00185000"
    underlying: str           # "AAPL"
    option_type: str          # "call" or "put"
    strike: float
    expiry: date
    days_to_expiry: int
    bid: float
    ask: float
    mid: float
    open_interest: int
    volume: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as a percentage of mid price."""
        if self.mid <= 0:
            return 1.0
        return (self.ask - self.bid) / self.mid

    @property
    def contract_value(self) -> float:
        """Mid price × 100 (standard US option multiplier)."""
        return self.mid * 100


class OptionsChainFetcher:
    """Fetches and filters option chains from Alpaca.

    All filtering params come from config.json options section so nothing
    is hardcoded here. Gracefully returns [] on any API failure.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        config: dict,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._cfg = config.get("options", {})

        # Cache: (symbol, option_type) → (timestamp, list[OptionContract])
        self._cache: Dict[Tuple[str, str], Tuple[float, List[OptionContract]]] = {}

        self._min_dte = self._cfg.get("long_call_min_dte", 25)
        self._max_dte = self._cfg.get("long_call_max_dte", 50)
        self._min_oi = self._cfg.get("min_contract_oi", 500)
        self._max_spread_pct = self._cfg.get("max_contract_spread_pct", 0.15)

    def get_chain(
        self,
        symbol: str,
        option_type: str,
        current_price: float,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
        strike_range_pct: float = 15.0,
    ) -> List[OptionContract]:
        """Fetch and filter option contracts for one symbol.

        Args:
            symbol: underlying ticker
            option_type: "call" or "put"
            current_price: current underlying price (for strike filter)
            min_dte: override config min_dte
            max_dte: override config max_dte
            strike_range_pct: only return strikes within ±pct of current_price

        Returns:
            Filtered list of OptionContract, sorted by days_to_expiry ASC then
            strike ASC. Empty list on failure or no data.
        """
        cache_key = (symbol, option_type)
        now = time.monotonic()
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if now - ts < _CACHE_TTL_SECONDS:
                return self._filter_by_strike(cached, current_price, strike_range_pct)

        contracts = self._fetch_from_alpaca(
            symbol, option_type,
            min_dte=min_dte or self._min_dte,
            max_dte=max_dte or self._max_dte,
        )
        self._cache[cache_key] = (now, contracts)
        return self._filter_by_strike(contracts, current_price, strike_range_pct)

    def get_atm_contract(
        self,
        symbol: str,
        option_type: str,
        current_price: float,
        target_delta: float = 0.50,
        target_dte: int = 35,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
    ) -> Optional[OptionContract]:
        """Select the single best contract targeting a delta and DTE.

        Selection order:
          1. Filter chain to in-range DTE and strike window
          2. Score each contract: weight delta proximity (70%) + DTE proximity (30%)
          3. Return lowest-score (best) contract with acceptable spread

        Returns None if no contract meets filters.
        """
        contracts = self.get_chain(
            symbol, option_type, current_price,
            min_dte=min_dte, max_dte=max_dte,
        )
        if not contracts:
            return None

        # Filter out contracts with no delta (Alpaca doesn't always return Greeks
        # for illiquid strikes) or unacceptably wide spreads
        tradeable = [
            c for c in contracts
            if c.delta is not None
            and c.spread_pct <= self._max_spread_pct
            and c.open_interest >= self._min_oi
            and c.mid > 0.05  # no sub-nickel options
        ]
        if not tradeable:
            logger.debug("No tradeable contracts for %s %s after filter", symbol, option_type)
            return None

        def score(c: OptionContract) -> float:
            delta_dist = abs(abs(c.delta) - target_delta)
            dte_dist = abs(c.days_to_expiry - target_dte) / max(target_dte, 1)
            return 0.70 * delta_dist + 0.30 * dte_dist

        best = min(tradeable, key=score)
        logger.debug(
            "Selected %s %s: strike=%.2f delta=%.2f DTE=%d mid=%.2f",
            symbol, option_type, best.strike, best.delta or 0,
            best.days_to_expiry, best.mid,
        )
        return best

    def _fetch_from_alpaca(
        self,
        symbol: str,
        option_type: str,
        min_dte: int,
        max_dte: int,
    ) -> List[OptionContract]:
        """Call Alpaca OptionHistoricalDataClient and parse snapshots."""
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient  # type: ignore
            from alpaca.data.requests import OptionChainRequest  # type: ignore

            client = OptionHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )

            today = date.today()
            req = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=today + timedelta(days=min_dte),
                expiration_date_lte=today + timedelta(days=max_dte),
                type=option_type,
            )
            snapshots = client.get_option_chain(req)

            if not snapshots:
                return []

            contracts = []
            for occ_symbol, snap in snapshots.items():
                try:
                    contract = self._parse_snapshot(occ_symbol, symbol, option_type, snap, today)
                    if contract:
                        contracts.append(contract)
                except Exception:
                    logger.debug("Failed to parse snapshot for %s", occ_symbol)

            contracts.sort(key=lambda c: (c.days_to_expiry, c.strike))
            logger.debug("Fetched %d %s contracts for %s", len(contracts), option_type, symbol)
            return contracts

        except ImportError:
            logger.warning("alpaca-py not installed — options chain unavailable")
            return []
        except Exception:
            logger.exception("Options chain fetch failed for %s %s", symbol, option_type)
            return []

    def _parse_snapshot(
        self,
        occ_symbol: str,
        underlying: str,
        option_type: str,
        snap,
        today: date,
    ) -> Optional[OptionContract]:
        """Parse an Alpaca OptionSnapshot into an OptionContract."""
        quote = getattr(snap, "latest_quote", None)
        greeks = getattr(snap, "greeks", None)

        bid = float(getattr(quote, "bid_price", 0) or 0) if quote else 0.0
        ask = float(getattr(quote, "ask_price", 0) or 0) if quote else 0.0
        mid = round((bid + ask) / 2, 4) if (bid > 0 and ask > 0) else 0.0

        # OCC symbol format: AAPL240119C00185000
        # Strike encoded as last 8 chars: SSSSS.CCC → divide by 1000
        try:
            strike_str = occ_symbol[-8:]
            strike = int(strike_str) / 1000.0
            expiry_str = occ_symbol[-15:-9]  # YYMMDD
            expiry = date(2000 + int(expiry_str[:2]), int(expiry_str[2:4]), int(expiry_str[4:6]))
        except (ValueError, IndexError):
            return None

        dte = (expiry - today).days
        if dte < 0:
            return None

        day = getattr(snap, "day", None)
        volume = int(getattr(day, "volume", 0) or 0) if day else 0
        oi = int(getattr(snap, "open_interest", 0) or 0)
        iv = float(getattr(snap, "implied_volatility", 0) or 0) or None

        delta = gamma = theta = vega = None
        if greeks:
            delta = float(getattr(greeks, "delta", 0) or 0) or None
            gamma = float(getattr(greeks, "gamma", 0) or 0) or None
            theta = float(getattr(greeks, "theta", 0) or 0) or None
            vega = float(getattr(greeks, "vega", 0) or 0) or None

        return OptionContract(
            symbol=occ_symbol,
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            days_to_expiry=dte,
            bid=bid,
            ask=ask,
            mid=mid,
            open_interest=oi,
            volume=volume,
            implied_volatility=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
        )

    @staticmethod
    def _filter_by_strike(
        contracts: List[OptionContract],
        current_price: float,
        strike_range_pct: float,
    ) -> List[OptionContract]:
        if not current_price:
            return contracts
        lo = current_price * (1 - strike_range_pct / 100)
        hi = current_price * (1 + strike_range_pct / 100)
        return [c for c in contracts if lo <= c.strike <= hi]
