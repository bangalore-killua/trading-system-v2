"""
price_stream.py — Alpaca StockDataStream WebSocket (Real-Time Bars)

Subscribes to real-time minute bars for all watchlist symbols via a single
persistent WebSocket connection. Bars are used to:
  - Update the regime detector with fresh SPY/QQQ price data
  - Detect intraday volume spikes → feed HOT_UNIVERSE
  - Provide real-time price for trailing stop evaluation

Free tier (IEX feed): slight volume undercount vs SIP tape, but price is accurate.
Paid upgrade: Alpaca SIP feed ($9/month) for full consolidated tape.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PriceStream:
    """Streams real-time 1-minute bars for a dynamic watchlist via alpaca-py.

    The subscription list is rebuilt whenever `update_symbols()` is called.
    This lets us add HOT_UNIVERSE symbols without restarting the stream.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        on_bar: Optional[Callable] = None,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._on_bar = on_bar  # callback(symbol, bar_dict) → called on each bar
        self._stream = None
        self._subscribed: Set[str] = set()
        self._running = False

        # Latest bar per symbol — read by other components without waiting for stream
        self._latest: Dict[str, dict] = {}

    async def start(self, symbols: List[str]) -> None:
        """Start the stream and subscribe to the given symbol list."""
        from alpaca.data.live import StockDataStream  # type: ignore

        self._stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed="iex",  # free feed; change to "sip" for paid consolidated tape
        )

        if symbols:
            self._subscribed = set(symbols)
            self._stream.subscribe_bars(self._handle_bar, *symbols)
            logger.info(
                "PriceStream started — subscribed to %d symbols: %s...",
                len(symbols),
                list(symbols)[:5],
            )

        self._running = True

        try:
            await self._stream._run_forever()
        except Exception:
            logger.exception("PriceStream error — will reconnect via scheduler")
            self._running = False

    async def stop(self) -> None:
        if self._stream:
            await self._stream.close()
        self._running = False

    async def update_symbols(self, new_symbols: List[str]) -> None:
        """Add new symbols to the live subscription (no restart needed)."""
        if not self._stream:
            return
        to_add = set(new_symbols) - self._subscribed
        if to_add:
            self._stream.subscribe_bars(self._handle_bar, *to_add)
            self._subscribed.update(to_add)
            logger.debug("PriceStream: added %s to subscription", list(to_add))

    async def _handle_bar(self, bar) -> None:
        """Called by alpaca-py for each incoming 1-minute bar."""
        symbol = bar.symbol
        bar_dict = {
            "symbol": symbol,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "timestamp": str(bar.timestamp),
        }
        self._latest[symbol] = bar_dict

        if self._on_bar:
            try:
                if asyncio.iscoroutinefunction(self._on_bar):
                    await self._on_bar(symbol, bar_dict)
                else:
                    self._on_bar(symbol, bar_dict)
            except Exception:
                logger.exception("PriceStream on_bar callback error for %s", symbol)

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Return the most recently received bar for a symbol."""
        return self._latest.get(symbol)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return the latest close price for a symbol, or None if not yet received."""
        bar = self._latest.get(symbol)
        return bar["close"] if bar else None

    @property
    def is_running(self) -> bool:
        return self._running
