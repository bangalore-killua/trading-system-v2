"""
tv_screener.py — TradingView Screener (Layer 3 of 3-layer universe)

Uses the `tradingview-screener` MIT library to scan all US stocks in a single
REST call every 5 minutes. No API key required.

Symbols with unusual volume + price momentum are added to HOT_UNIVERSE.
This catches breakouts in stocks not yet on the watchlist — before the next
60-minute static scoring cycle would see them.

Criteria (all configurable via config.json):
  - Volume ratio (today vs 20-day avg) > 3.0   ← 3x unusual volume
  - Price change % > 1.5%                       ← clear directional move
  - Price > $5                                  ← exclude penny stocks
  - Average daily volume > 500K                 ← minimum liquidity
"""

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TradingViewScreener:
    """Runs the TradingView Screener on all US stocks and feeds HOT_UNIVERSE.

    The `tradingview_screener` library makes a single POST to TradingView's
    undocumented (but stable) screener endpoint. No login, no API key.
    """

    def __init__(
        self,
        hot_universe,
        poll_interval_minutes: float = 5.0,
        min_volume_ratio: float = 3.0,
        min_price_change_pct: float = 1.5,
        min_price_usd: float = 5.0,
        min_avg_volume: int = 500_000,
        top_n: int = 25,
    ):
        self._hot_universe = hot_universe
        self._interval = poll_interval_minutes * 60
        self._min_volume_ratio = min_volume_ratio
        self._min_price_change_pct = min_price_change_pct
        self._min_price = min_price_usd
        self._min_avg_vol = min_avg_volume
        self._top_n = top_n
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info(
            "TradingView Screener started (poll every %.0f min, top %d symbols)",
            self._interval / 60,
            self._top_n,
        )
        while self._running:
            try:
                symbols = await asyncio.get_event_loop().run_in_executor(
                    None, self._scan
                )
                for sym in symbols:
                    self._hot_universe.add(
                        symbol=sym["symbol"],
                        source="TV_SCREENER",
                        headline=(
                            f"TV: {sym['change_pct']:+.1f}% on {sym['volume_ratio']:.1f}x volume"
                        ),
                        metadata={
                            "change_pct": sym["change_pct"],
                            "volume_ratio": sym["volume_ratio"],
                            "price": sym["price"],
                        },
                    )
            except Exception:
                logger.exception("TradingView Screener error")

            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._running = False

    def _scan(self) -> List[dict]:
        """Blocking screener call — run via executor to avoid blocking the event loop."""
        from tradingview_screener import Query, Column  # type: ignore

        df = (
            Query()
            .select(
                "name",
                "close",
                "change",           # % change today
                "volume",
                "average_volume_10d_calc",
                "relative_volume_10d_calc",  # volume / 10d avg
            )
            .where(
                Column("close") > self._min_price,
                Column("average_volume_10d_calc") > self._min_avg_vol,
                Column("relative_volume_10d_calc") > self._min_volume_ratio,
                Column("change") > self._min_price_change_pct,
                Column("exchange").isin(["NASDAQ", "NYSE", "AMEX"]),
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(self._top_n)
            .get_scanner_data()
        )

        # get_scanner_data() returns (count, DataFrame)
        _, results_df = df if isinstance(df, tuple) else (0, df)

        if results_df is None or results_df.empty:
            return []

        output = []
        for _, row in results_df.iterrows():
            output.append({
                "symbol": str(row.get("name", "")).replace("NASDAQ:", "").replace("NYSE:", "").replace("AMEX:", ""),
                "price": float(row.get("close", 0)),
                "change_pct": float(row.get("change", 0)),
                "volume_ratio": float(row.get("relative_volume_10d_calc", 0)),
            })

        if output:
            logger.info(
                "TradingView Screener: %d symbols with unusual volume — top: %s",
                len(output),
                [s["symbol"] for s in output[:5]],
            )

        return output

    def run_once_sync(self) -> List[dict]:
        """Convenience method for testing — runs a single scan synchronously."""
        return self._scan()
