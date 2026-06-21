"""
position_manager.py — Entry & Exit Logic (Signal Basket Properly Wired)

This is the critical fix over TradingSystem where signal_basket.py was fully
implemented but compute_entry_basket() / detect_primary_setup() were NEVER
called in evaluate_entry(). The old code used a simple score-threshold bypass.

This version:
  - Calls detect_primary_setup() to classify the setup type
  - Calls compute_entry_basket() to get the weighted entry score
  - Calls compute_exit_basket() on every monitoring cycle (was also missing)
  - Enforces all risk checks before submitting any order
  - Never auto-executes SETUP_NEWS_DRIVEN — always sends Telegram confirm

Entry flow per symbol:
  fetch OHLCV → compute indicators → detect_primary_setup() →
  compute_entry_basket() → risk_checks() → [Telegram if NEWS_DRIVEN] → place_market_buy()

Exit flow (5-min cycle):
  for each open position:
    hard_exit_check() → regime_exit_check() → compute_exit_basket() →
    age1_partial_check() → trailing_stop_check()
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PositionManager:
    """Orchestrates entry and exit decisions using the signal basket."""

    def __init__(
        self,
        config: dict,
        alpaca_trader,           # AlpacaTrader
        signal_basket,           # from strategy.signal_basket
        regime_detector,         # from strategy.regime_detector
        technical_analysis,      # module reference (strategy.technical_analysis)
        threat_classifier,       # ThreatClassifier
        news_sentiment,          # FinBERTSentiment
        trade_ledger,            # TradeLedger
        correlation_risk,        # CorrelationRisk
        position_aging,          # PositionAging
        trade_state,             # TradeState
        hot_universe,            # HotUniverse
        telegram=None,           # TelegramAlerts
        data_fetcher=None,       # fetches OHLCV
        fundamentals_fetcher=None,
    ):
        self._cfg = config
        self._trader = alpaca_trader
        self._basket = signal_basket
        self._regime = regime_detector
        self._ta = technical_analysis
        self._threat = threat_classifier
        self._sentiment = news_sentiment
        self._ledger = trade_ledger
        self._corr = correlation_risk
        self._aging = position_aging
        self._state = trade_state
        self._hot = hot_universe
        self._telegram = telegram
        self._fetcher = data_fetcher
        self._fundamentals = fundamentals_fetcher

        self._risk_cfg = config.get("risk", {})
        self._news_cfg = config.get("news", {})
        self._basket_cfg = config.get("signal_basket", {})
        self._sizing_cfg = config.get("position_sizing", {})

        # Pending news-driven confirms: symbol → asyncio.Event
        self._pending_news_confirms: Dict[str, asyncio.Event] = {}
        self._news_confirm_approvals: Dict[str, bool] = {}

    # ── Entry path ──────────────────────────────────────────────────────────

    async def evaluate_entry(self, symbol: str, source: str = "WATCHLIST") -> bool:
        """Full entry evaluation for one symbol. Returns True if order was placed."""
        # Skip if already holding this symbol
        if self._state.has_position(symbol):
            logger.debug("SKIP %s — already in position", symbol)
            return False

        # Skip if at max portfolio positions
        max_pos = self._risk_cfg.get("max_portfolio_positions", 15)
        if len(self._state.get_all_positions()) >= max_pos:
            logger.info("SKIP %s — at max positions (%d)", symbol, max_pos)
            return False

        # Fetch OHLCV (last 250 bars for indicator warm-up)
        df = await self._fetch_ohlcv(symbol, bars=250)
        if df is None or len(df) < 50:
            logger.warning("SKIP %s — insufficient price data", symbol)
            return False

        # Compute all indicators and candlestick patterns
        indicators = self._ta.compute_indicators(df)
        if not indicators:
            logger.warning("SKIP %s — indicator computation failed", symbol)
            return False

        current_price = indicators.get("price", 0)
        if current_price < self._risk_cfg.get("min_price_usd", 5.0):
            logger.debug("SKIP %s — price $%.2f below minimum", symbol, current_price)
            return False

        # ── Step 1: Detect primary setup type ──────────────────────────────
        # This is the call that was MISSING in TradingSystem's position_manager
        setup_type = self._basket.detect_primary_setup(symbol, indicators)
        if setup_type is None:
            logger.debug("SKIP %s — no setup detected", symbol)
            return False

        logger.info("SETUP DETECTED: %s → %s", symbol, setup_type)

        # ── Step 2: Compute entry basket score ─────────────────────────────
        # Also missing in old code — old code used raw scorecard sum
        entry_score, fired_signals = self._basket.compute_entry_basket(
            symbol, setup_type, indicators
        )
        threshold = self._basket_cfg.get("entry_thresholds", {}).get(str(setup_type), 50)

        logger.info(
            "ENTRY BASKET: %s %s score=%d threshold=%d signals=%s",
            symbol, setup_type, entry_score, threshold,
            [s for s in fired_signals if fired_signals[s]][:5],
        )

        if entry_score < threshold:
            logger.debug(
                "SKIP %s — entry score %d < threshold %d", symbol, entry_score, threshold
            )
            return False

        # ── Step 3: Risk checks (all must pass) ────────────────────────────
        risk_ok, risk_reason = await self._check_risk(symbol, current_price, indicators)
        if not risk_ok:
            logger.info("SKIP %s — risk check failed: %s", symbol, risk_reason)
            return False

        # ── Step 4: SETUP_NEWS_DRIVEN requires Telegram confirm ────────────
        from strategy.signal_types import SetupType  # type: ignore
        if hasattr(SetupType, "NEWS_DRIVEN") and setup_type == SetupType.NEWS_DRIVEN:
            approved = await self._request_news_confirm(symbol, entry_score, indicators)
            if not approved:
                logger.info("SKIP %s — Telegram confirm rejected or timed out", symbol)
                return False

        # ── Step 5: Calculate position size ────────────────────────────────
        atr = indicators.get("atr")
        portfolio_value = await self._get_portfolio_value()
        target_pct = self._sizing_cfg.get("base_size_pct", 5.0)

        # Reduce size in NEUTRAL regime
        regime = self._regime.get_current_regime() if self._regime else None
        if regime and str(regime) in self._cfg.get("regime", {}).get("regimes_reduced_size", []):
            target_pct = self._sizing_cfg.get("reduced_size_pct", 2.5)
            logger.info("Reduced position size due to NEUTRAL regime")

        qty = self._trader.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            target_pct=target_pct,
            atr=atr,
            stop_atr_mult=self._sizing_cfg.get("atr_multiplier_stop", 1.5),
        )

        if qty <= 0:
            logger.warning("SKIP %s — calculated qty=0 (insufficient buying power?)", symbol)
            return False

        # ── Step 6: Place order ────────────────────────────────────────────
        fill = self._trader.place_market_buy(symbol, qty=qty)

        if not fill.is_filled:
            logger.error("ORDER FAILED: %s — status=%s", symbol, fill.status)
            return False

        # ── Step 7: Record the trade ───────────────────────────────────────
        stop_price = fill.filled_avg_price - (atr * self._sizing_cfg.get("atr_multiplier_stop", 1.5)) if atr else None
        target_price = fill.filled_avg_price + (atr * self._sizing_cfg.get("atr_multiplier_target", 3.0)) if atr else None

        self._state.record_entry(
            symbol=symbol,
            qty=fill.filled_qty,
            entry_price=fill.filled_avg_price,
            setup_type=str(setup_type),
            entry_score=entry_score,
            stop_price=stop_price,
            target_price=target_price,
        )

        if self._ledger:
            self._ledger.log_entry(
                symbol=symbol,
                qty=fill.filled_qty,
                price=fill.filled_avg_price,
                setup_type=str(setup_type),
                score=entry_score,
            )

        if self._telegram:
            await self._telegram.send_entry_alert(
                symbol=symbol,
                qty=fill.filled_qty,
                price=fill.filled_avg_price,
                setup_type=str(setup_type),
                score=entry_score,
                source=source,
            )

        logger.info(
            "ENTERED: %s %d shares @ $%.4f (setup=%s score=%d)",
            symbol, int(fill.filled_qty), fill.filled_avg_price, setup_type, entry_score,
        )
        return True

    async def score_watchlist(self) -> None:
        """Score all static watchlist symbols. Called every 60 minutes."""
        from strategy.config_loader import get_watchlist  # type: ignore

        watchlist = get_watchlist(self._cfg)
        logger.info("Scoring watchlist: %d symbols", len(watchlist))

        # Process symbols concurrently with a concurrency cap
        sem = asyncio.Semaphore(5)

        async def bounded_eval(sym: str) -> None:
            async with sem:
                await self.evaluate_entry(sym, source="WATCHLIST")

        await asyncio.gather(*[bounded_eval(s) for s in watchlist], return_exceptions=True)

    # ── Exit path ───────────────────────────────────────────────────────────

    async def monitor_exits(self) -> None:
        """Score all open positions for exit. Called every 5 minutes."""
        positions = self._state.get_all_positions()
        if not positions:
            return

        logger.debug("Monitoring %d open positions", len(positions))

        for symbol, pos_info in positions.items():
            try:
                await self._evaluate_exit(symbol, pos_info)
            except Exception:
                logger.exception("Exit eval error for %s", symbol)

    async def _evaluate_exit(self, symbol: str, pos_info: dict) -> None:
        """Check all exit conditions for one open position in priority order."""

        # Priority 1: Hard exits — immediate, no basket needed
        if await self._check_hard_exit(symbol, pos_info):
            return

        # Priority 2: Regime deterioration exit
        if await self._check_regime_exit(symbol, pos_info):
            return

        # Fetch fresh indicators for basket evaluation
        df = await self._fetch_ohlcv(symbol, bars=60)
        if df is None or len(df) < 20:
            return

        indicators = self._ta.compute_indicators(df)
        if not indicators:
            return

        # Priority 3: Exit basket
        setup_type = pos_info.get("setup_type", "SETUP_MOMENTUM")
        exit_score, _ = self._basket.compute_exit_basket(symbol, setup_type, indicators)
        exit_threshold = self._basket_cfg.get("exit_thresholds", {}).get(setup_type, 40)

        if exit_score >= exit_threshold:
            logger.info(
                "EXIT BASKET: %s score=%d >= threshold=%d",
                symbol, exit_score, exit_threshold,
            )
            await self._execute_exit(symbol, pos_info, reason=f"exit_basket:{exit_score}")
            return

        # Priority 4: AGE-1 partial profit
        if self._aging:
            should_partial, partial_pct = self._aging.check_position_aging(
                symbol, pos_info, indicators
            )
            if should_partial:
                await self._execute_partial_exit(symbol, pos_info, partial_pct)
                return

        # Priority 5: Trailing stop
        await self._check_trailing_stop(symbol, pos_info, indicators)

    async def _check_hard_exit(self, symbol: str, pos_info: dict) -> bool:
        """Hard exits: kill switch, CRITICAL threat, going concern 8-K."""
        from server.kill_switch import is_kill_switch_active  # type: ignore

        if is_kill_switch_active():
            await self._execute_exit(symbol, pos_info, reason="kill_switch")
            return True

        # Check if CRITICAL threat news arrived for this symbol
        hot_entry = self._hot.get_entry(symbol)
        if hot_entry and hot_entry.trigger_metadata.get("is_bearish"):
            threat = hot_entry.trigger_metadata.get("threat_level", "")
            if self._threat and self._threat.is_exit_trigger(threat):
                await self._execute_exit(symbol, pos_info, reason=f"CRITICAL_threat:{threat}")
                return True

        # Going concern 8-K (item 1.03 = bankruptcy, 2.04 = debt default)
        if hot_entry and hot_entry.source == "EDGAR_8K":
            items = hot_entry.trigger_metadata.get("items", [])
            going_concern_items = {"1.03", "2.04", "8.06"}
            if any(item in going_concern_items for item in items):
                await self._execute_exit(symbol, pos_info, reason=f"edgar_hard_exit:{items}")
                return True

        return False

    async def _check_regime_exit(self, symbol: str, pos_info: dict) -> bool:
        """Exit based on regime deterioration."""
        if not self._regime:
            return False

        regime = str(self._regime.get_current_regime())
        regime_cfg = self._cfg.get("regime", {})

        if regime in regime_cfg.get("regimes_exit_all", ["CRASH"]):
            await self._execute_exit(symbol, pos_info, reason=f"regime:{regime}")
            return True

        setup_type = pos_info.get("setup_type", "")
        # EARLY_BEAR: exit everything except INFLECTION setups (they go up in bear markets)
        if regime == "EARLY_BEAR" and "INFLECTION" not in setup_type:
            await self._execute_exit(symbol, pos_info, reason="regime:EARLY_BEAR")
            return True

        return False

    async def _check_trailing_stop(self, symbol: str, pos_info: dict, indicators: dict) -> None:
        """ATR-based trailing stop — activates only after position gains activation_pct."""
        entry_price = pos_info.get("entry_price", 0)
        current_price = indicators.get("price", 0)
        atr = indicators.get("atr")
        peak_price = pos_info.get("peak_price", entry_price)

        if not entry_price or not current_price or not atr:
            return

        gain_pct = (current_price - entry_price) / entry_price * 100
        activation = self._sizing_cfg.get("trailing_stop_activation_pct", 20.0)

        if gain_pct < activation:
            return  # Not yet profitable enough to trail

        # Update peak price if current is higher
        if current_price > peak_price:
            self._state.update_peak_price(symbol, current_price)
            peak_price = current_price

        trail_distance = atr * self._sizing_cfg.get("trailing_stop_atr_multiplier", 2.0)
        trail_stop = peak_price - trail_distance

        if current_price <= trail_stop:
            logger.info(
                "TRAILING STOP: %s price=%.2f trail_stop=%.2f (peak=%.2f)",
                symbol, current_price, trail_stop, peak_price,
            )
            await self._execute_exit(symbol, pos_info, reason=f"trailing_stop:peak={peak_price:.2f}")

    async def _execute_exit(self, symbol: str, pos_info: dict, reason: str) -> None:
        """Place market sell for full position."""
        qty = pos_info.get("qty", 0)
        if qty <= 0:
            return

        fill = self._trader.place_market_sell(symbol, qty=qty)

        if fill.is_filled:
            pnl = (fill.filled_avg_price - pos_info.get("entry_price", 0)) * fill.filled_qty
            self._state.remove_position(symbol)

            if self._ledger:
                self._ledger.log_exit(
                    symbol=symbol,
                    qty=fill.filled_qty,
                    price=fill.filled_avg_price,
                    reason=reason,
                    pnl=pnl,
                )

            if self._telegram:
                await self._telegram.send_exit_alert(
                    symbol=symbol,
                    qty=fill.filled_qty,
                    price=fill.filled_avg_price,
                    reason=reason,
                    pnl=pnl,
                )

            logger.info(
                "EXITED: %s %d shares @ $%.4f reason=%s PnL=$%.2f",
                symbol, int(fill.filled_qty), fill.filled_avg_price, reason, pnl,
            )

    async def _execute_partial_exit(self, symbol: str, pos_info: dict, sell_pct: float) -> None:
        """Partial exit for AGE-1 profit taking (default: sell 50%)."""
        qty = pos_info.get("qty", 0)
        sell_qty = max(1, int(qty * sell_pct / 100))

        if sell_qty >= qty:
            await self._execute_exit(symbol, pos_info, reason="age1_full")
            return

        fill = self._trader.place_market_sell(symbol, qty=sell_qty)
        if fill.is_filled:
            self._state.reduce_position(symbol, sell_qty)

            if self._telegram:
                await self._telegram.send_partial_exit_alert(
                    symbol=symbol,
                    sold_qty=fill.filled_qty,
                    price=fill.filled_avg_price,
                    remaining_qty=qty - sell_qty,
                )

            logger.info(
                "PARTIAL EXIT (AGE-1): %s sold %d of %d shares @ $%.4f",
                symbol, int(fill.filled_qty), qty, fill.filled_avg_price,
            )

    # ── Risk checks ─────────────────────────────────────────────────────────

    async def _check_risk(
        self, symbol: str, price: float, indicators: dict
    ) -> tuple[bool, str]:
        """Run all pre-entry risk checks. Returns (passed, reason_if_failed)."""

        # 1. Regime gate
        if self._regime:
            regime = str(self._regime.get_current_regime())
            no_new_longs = self._cfg.get("regime", {}).get("regimes_no_new_longs", [])
            if regime in no_new_longs:
                return False, f"regime={regime}"

        # 2. Cash reserve
        account = self._trader.get_account()
        portfolio_value = float(getattr(account, "portfolio_value", 0) or 0)
        cash = float(getattr(account, "cash", 0) or 0)
        cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0
        min_cash = self._risk_cfg.get("cash_reserve_pct", 20)
        if cash_pct < min_cash:
            return False, f"cash_reserve={cash_pct:.1f}%<{min_cash}%"

        # 3. Sector concentration
        if self._corr:
            sector_ok, sector_reason = self._corr.check_sector_concentration(symbol)
            if not sector_ok:
                return False, f"sector_concentration:{sector_reason}"

        # 4. Correlation check
        if self._corr:
            corr_ok, corr_reason = self._corr.check_correlation(symbol)
            if not corr_ok:
                return False, f"correlation:{corr_reason}"

        # 5. Volume filter (min liquidity)
        vol = indicators.get("volume_sma20", 0)
        if vol < self._risk_cfg.get("min_avg_volume", 500_000):
            return False, f"low_volume={vol:.0f}"

        return True, ""

    # ── Telegram news-driven confirm ─────────────────────────────────────────

    async def _request_news_confirm(
        self, symbol: str, score: int, indicators: dict
    ) -> bool:
        """Send Telegram alert and wait for human y/n reply (30-min timeout)."""
        if not self._telegram:
            return False  # No Telegram = no way to confirm = block

        timeout = self._cfg.get("news", {}).get("news_driven_confirm_timeout_minutes", 30) * 60

        event = asyncio.Event()
        self._pending_news_confirms[symbol] = event
        self._news_confirm_approvals[symbol] = False

        await self._telegram.send_news_confirm_request(
            symbol=symbol,
            score=score,
            price=indicators.get("price", 0),
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.info("NEWS_CONFIRM timeout for %s", symbol)

        approved = self._news_confirm_approvals.pop(symbol, False)
        self._pending_news_confirms.pop(symbol, None)
        return approved

    def approve_news_confirm(self, symbol: str) -> None:
        """Called by Telegram webhook handler when user replies 'y'."""
        self._news_confirm_approvals[symbol] = True
        event = self._pending_news_confirms.get(symbol)
        if event:
            event.set()

    # ── Helpers ─────────────────────────────────────────────────────────────

    async def _fetch_ohlcv(self, symbol: str, bars: int = 250):
        """Fetch OHLCV bars. Returns None on failure."""
        if self._fetcher is None:
            logger.warning("No data fetcher configured")
            return None
        try:
            return await self._fetcher.get_bars(symbol, bars=bars)
        except Exception:
            logger.exception("OHLCV fetch error for %s", symbol)
            return None

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value from Alpaca."""
        try:
            account = self._trader.get_account()
            return float(getattr(account, "portfolio_value", 100_000) or 100_000)
        except Exception:
            return 100_000.0  # Safe fallback for paper trading
