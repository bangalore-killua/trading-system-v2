"""
option_stream.py — Alpaca OptionDataStream WebSocket Consumer

Subscribes to real-time quote updates for open option positions.
Used by the options monitoring job to track P&L and fire exit rules:

  - Stop loss:    current mid drops 50% below entry premium
  - Take profit:  current mid doubles vs entry premium (100% gain)
  - DTE close:    7 days to expiry — close to avoid gamma/pin risk

Subscriptions are managed dynamically:
  - subscribe(contract_symbol)   → called when a new option position opens
  - unsubscribe(contract_symbol) → called when a position closes

The stream runs as a background asyncio task, started by TradingScheduler.
"""

import asyncio
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class OptionDataStreamManager:
    """Manages an Alpaca OptionDataStream WebSocket connection.

    The underlying stream object (alpaca.data.live.option.OptionDataStream)
    is lazy-initialized on first subscription so startup isn't blocked if
    no option positions are open.
    """

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key
        self._stream = None
        self._subscriptions: Dict[str, float] = {}   # contract_symbol → entry_premium
        self._quote_callbacks: Dict[str, Callable] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket stream as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="option_data_stream")
        logger.info("OptionDataStream manager started")

    async def stop(self) -> None:
        """Gracefully shut down the stream."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OptionDataStream manager stopped")

    # ── Subscription management ──────────────────────────────────────────────

    def subscribe(
        self,
        contract_symbol: str,
        entry_premium: float,
        on_quote: Optional[Callable] = None,
    ) -> None:
        """Track a new open option position.

        Args:
            contract_symbol: OCC symbol e.g. "AAPL240119C00185000"
            entry_premium: per-share cost paid (× 100 = contract value)
            on_quote: optional callback(contract_symbol, mid_price) on quote update
        """
        self._subscriptions[contract_symbol] = entry_premium
        if on_quote:
            self._quote_callbacks[contract_symbol] = on_quote
        logger.info("Option stream: subscribed to %s (entry_premium=%.2f)", contract_symbol, entry_premium)

        # If stream is running, add this symbol dynamically
        if self._stream is not None:
            try:
                self._stream.subscribe_quotes(self._handle_quote, contract_symbol)
            except Exception:
                logger.exception("Failed to subscribe stream to %s", contract_symbol)

    def unsubscribe(self, contract_symbol: str) -> None:
        """Remove a closed position from tracking."""
        self._subscriptions.pop(contract_symbol, None)
        self._quote_callbacks.pop(contract_symbol, None)
        logger.debug("Option stream: unsubscribed from %s", contract_symbol)

    def get_subscriptions(self) -> Dict[str, float]:
        """Return all tracked contracts and their entry premiums."""
        return dict(self._subscriptions)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        """Main loop: initialise stream and reconnect on failure."""
        while self._running:
            try:
                await self._connect_and_run()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("OptionDataStream error — reconnecting in 30s")
                await asyncio.sleep(30)

    async def _connect_and_run(self) -> None:
        """Establish connection and subscribe to current positions."""
        if not self._subscriptions:
            # No positions open — idle wait, check every 60 seconds
            await asyncio.sleep(60)
            return

        try:
            from alpaca.data.live.option import OptionDataStream  # type: ignore

            self._stream = OptionDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )

            # Subscribe to all currently open positions
            symbols = list(self._subscriptions.keys())
            self._stream.subscribe_quotes(self._handle_quote, *symbols)
            logger.info("OptionDataStream connected — subscribed to %d contracts", len(symbols))

            await self._stream._run_forever()

        except ImportError:
            logger.warning("alpaca-py OptionDataStream not available — option quotes will use polling")
            await asyncio.sleep(300)  # Retry after 5 minutes
        finally:
            self._stream = None

    async def _handle_quote(self, data) -> None:
        """Process an incoming option quote update."""
        try:
            contract_symbol = str(getattr(data, "symbol", ""))
            bid = float(getattr(data, "bid_price", 0) or 0)
            ask = float(getattr(data, "ask_price", 0) or 0)
            mid = round((bid + ask) / 2, 4) if (bid > 0 and ask > 0) else 0.0

            if not contract_symbol or mid <= 0:
                return

            entry_premium = self._subscriptions.get(contract_symbol)
            if entry_premium is None:
                return

            # Fire user callback if registered
            cb = self._quote_callbacks.get(contract_symbol)
            if cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(contract_symbol, mid)
                else:
                    cb(contract_symbol, mid)

        except Exception:
            logger.debug("Option quote parse error", exc_info=True)
