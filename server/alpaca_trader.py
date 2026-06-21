"""
alpaca_trader.py — Alpaca Broker Client (alpaca-py SDK)

Replaces TradingSystem's archived `alpaca-trade-api` with the official `alpaca-py` SDK.

Critical fix over TradingSystem: fill verification.
Old code submitted an order and returned immediately — if the order was rejected
or partially filled, the system never knew. This version polls `get_order_by_id()`
until the order reaches a terminal state (filled, cancelled, expired, rejected).

Usage:
    trader = AlpacaTrader.from_env()
    fill = trader.place_market_buy("AAPL", qty=10)
    fill = trader.place_market_sell("AAPL", qty=10)
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Terminal states — stop polling when order reaches one of these
TERMINAL_STATES = {"filled", "cancelled", "expired", "rejected", "done_for_day", "held"}


@dataclass
class OrderFill:
    order_id: str
    symbol: str
    side: str          # "buy" or "sell"
    qty: float
    filled_qty: float
    filled_avg_price: float
    status: str
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def is_failed(self) -> bool:
        return self.status in ("cancelled", "expired", "rejected")


class AlpacaTrader:
    """Broker client wrapping alpaca-py for order submission and fill verification.

    Paper vs live is controlled by the `paper` flag (default True).
    The Alpaca base URL difference is handled by alpaca-py automatically.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        fill_timeout_seconds: int = 30,
        max_slippage_pct: float = 0.5,
    ):
        from alpaca.trading.client import TradingClient  # type: ignore
        from alpaca.trading.requests import MarketOrderRequest  # type: ignore
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore

        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self._fill_timeout = fill_timeout_seconds
        self._max_slippage = max_slippage_pct
        self._paper = paper

        # Keep these imported references so we don't re-import on every order
        self._MarketOrderRequest = MarketOrderRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce

        logger.info(
            "AlpacaTrader initialized (mode=%s, fill_timeout=%ds)",
            "PAPER" if paper else "LIVE",
            fill_timeout_seconds,
        )

    @classmethod
    def from_env(cls) -> "AlpacaTrader":
        """Construct from environment variables (loaded via .env)."""
        return cls(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
        )

    def place_market_buy(
        self,
        symbol: str,
        qty: float,
        notional: Optional[float] = None,
    ) -> OrderFill:
        """Submit a market buy and block until filled or timeout.

        Args:
            symbol: equity ticker
            qty: number of shares (fractional OK on Alpaca)
            notional: dollar amount to buy (alternative to qty)

        Returns:
            OrderFill with final status. Check fill.is_filled before recording.
        """
        return self._place_order(symbol, "buy", qty=qty, notional=notional)

    def place_market_sell(
        self,
        symbol: str,
        qty: float,
    ) -> OrderFill:
        """Submit a market sell and block until filled or timeout."""
        return self._place_order(symbol, "sell", qty=qty)

    def _place_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
    ) -> OrderFill:
        """Submit order then poll until terminal state."""
        order_side = self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL

        request_kwargs = dict(
            symbol=symbol,
            side=order_side,
            time_in_force=self._TimeInForce.DAY,
            type="market",
        )
        if notional is not None:
            request_kwargs["notional"] = str(round(notional, 2))
        elif qty is not None:
            request_kwargs["qty"] = str(qty)
        else:
            raise ValueError("Must provide qty or notional")

        request = self._MarketOrderRequest(**request_kwargs)

        logger.info("ORDER SUBMIT: %s %s qty=%s notional=%s", side.upper(), symbol, qty, notional)

        try:
            order = self._client.submit_order(order_data=request)
        except Exception as exc:
            logger.error("ORDER FAILED (submit): %s %s — %s", side.upper(), symbol, exc)
            return OrderFill(
                order_id="failed",
                symbol=symbol,
                side=side,
                qty=qty or 0,
                filled_qty=0,
                filled_avg_price=0,
                status="rejected",
            )

        return self._wait_for_fill(order.id, symbol, side, qty or 0)

    def _wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
    ) -> OrderFill:
        """Poll get_order_by_id() until terminal state or timeout.

        Polls every 0.5 seconds. Alpaca market orders typically fill in < 1s
        during market hours. The timeout is a safety net for edge cases
        (pre/post market, halted stock, weekend).
        """
        deadline = time.monotonic() + self._fill_timeout
        poll_interval = 0.5

        while time.monotonic() < deadline:
            try:
                order = self._client.get_order_by_id(order_id)
            except Exception as exc:
                logger.warning("Fill poll error for %s: %s — retrying", order_id, exc)
                time.sleep(poll_interval)
                continue

            status = order.status.value if hasattr(order.status, "value") else str(order.status)

            if status in TERMINAL_STATES:
                filled_qty = float(order.filled_qty or 0)
                filled_price = float(order.filled_avg_price or 0)

                if status == "filled":
                    logger.info(
                        "ORDER FILLED: %s %s qty=%.2f @ $%.4f",
                        side.upper(), symbol, filled_qty, filled_price,
                    )
                else:
                    logger.warning(
                        "ORDER %s: %s %s — filled=%.2f/%s",
                        status.upper(), side.upper(), symbol, filled_qty, qty,
                    )

                return OrderFill(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    filled_qty=filled_qty,
                    filled_avg_price=filled_price,
                    status=status,
                    submitted_at=str(getattr(order, "submitted_at", "")),
                    filled_at=str(getattr(order, "filled_at", "")),
                )

            time.sleep(poll_interval)

        # Timeout reached — cancel the order and return partial fill info
        logger.error(
            "ORDER TIMEOUT (%ds): %s %s — attempting cancel",
            self._fill_timeout, side.upper(), symbol,
        )
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception:
            logger.exception("Cancel failed for order %s", order_id)

        return OrderFill(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            filled_qty=0,
            filled_avg_price=0,
            status="timeout",
        )

    def get_account(self):
        """Return Alpaca account object with cash, equity, buying_power, etc."""
        return self._client.get_account()

    def get_all_positions(self) -> list:
        """Return all open positions."""
        return self._client.get_all_positions()

    def get_position(self, symbol: str):
        """Return a single open position, or None if not held."""
        try:
            return self._client.get_open_position(symbol)
        except Exception:
            return None

    def cancel_all_orders(self) -> None:
        """Emergency: cancel all open orders (used by kill_switch)."""
        self._client.cancel_orders()
        logger.warning("ALL ORDERS CANCELLED via kill_switch")

    def close_all_positions(self) -> None:
        """Emergency: close all open positions at market (used by kill_switch)."""
        self._client.close_all_positions(cancel_orders=True)
        logger.warning("ALL POSITIONS CLOSED via kill_switch")

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        target_pct: float,
        atr: Optional[float] = None,
        stop_atr_mult: float = 1.5,
    ) -> int:
        """Calculate share count for a given target portfolio allocation %.

        Uses ATR-based stop to compute risk-adjusted size:
          risk_per_share = ATR × stop_multiplier
          max_risk_dollars = portfolio_value × target_pct / 100
          qty = max_risk_dollars / risk_per_share

        Falls back to simple dollar allocation if ATR not available.
        """
        if current_price <= 0:
            return 0

        target_dollars = portfolio_value * (target_pct / 100)

        if atr and atr > 0:
            risk_per_share = atr * stop_atr_mult
            qty = target_dollars / risk_per_share
        else:
            qty = target_dollars / current_price

        qty = max(1, int(qty))

        # Sanity check: single position shouldn't exceed 10% of portfolio
        max_shares = int(portfolio_value * 0.10 / current_price)
        qty = min(qty, max_shares)

        logger.debug(
            "Position size: %s @ $%.2f → %d shares (%.1f%% allocation, ATR=%.2f)",
            symbol, current_price, qty, target_pct, atr or 0,
        )
        return qty
