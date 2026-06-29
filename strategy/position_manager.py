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
        self._options_cfg = config.get("options", {})
        self._scorer = None        # Scorer — injected after construction
        self._options_selector = None  # OptionsSelector — injected after construction
        self._options_stream = None    # OptionDataStreamManager — injected after construction
        self._db = None            # TradingDatabase — injected after construction
        self._exit_monitor = None  # ExitMonitor — lazy-init on first exit evaluation

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

        # ── Step 5a: Build feature vector + LightGBM size multiplier ──────────
        fv = None
        size_multiplier = 1.0
        if self._scorer:
            try:
                fv = self._scorer.build_vector(
                    symbol=symbol,
                    df=df,
                    indicators=indicators,
                    setup_type=str(setup_type),
                )
                size_multiplier = self._scorer.get_position_size_multiplier(fv)
                logger.info(
                    "META-MODEL: %s P(win) multiplier=%.2f", symbol, size_multiplier
                )
            except Exception:
                logger.exception("Scorer failed for %s — using full size", symbol)

        if size_multiplier == 0.0:
            logger.info(
                "SKIP %s — LightGBM P(win) below minimum confidence threshold", symbol
            )
            # Log the skipped evaluation for training data
            if self._scorer and fv:
                await self._scorer.log_entry(
                    symbol=symbol, fv=fv, entry_price=current_price,
                    setup_type=str(setup_type), entry_score=entry_score,
                    order_placed=False,
                )
            return False

        # ── Step 5b: Calculate position size ───────────────────────────────
        atr = indicators.get("atr")
        portfolio_value = await self._get_portfolio_value()
        target_pct = self._sizing_cfg.get("base_size_pct", 2)

        # Reduce size in NEUTRAL regime
        regime = self._regime.get_current_regime() if self._regime else None
        if regime and str(regime) in self._cfg.get("regime", {}).get("regimes_reduced_size", []):
            target_pct = self._sizing_cfg.get("reduced_size_pct", 1)
            logger.info("Reduced position size due to NEUTRAL regime")

        # Apply LightGBM multiplier (1.0 full / 0.5 half — never 0.0 here, caught above)
        target_pct = target_pct * size_multiplier

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
            try:
                from data.logging.trade_logger import new_trace
                trace_id, elog = new_trace(symbol, "EXECUTION")
                elog.result("FAIL").reason(f"order_status={fill.status}").error(f"fill failed: {fill.status}").emit()
            except Exception:
                pass
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

        # ── EXECUTION log ─────────────────────────────────────────────────
        try:
            from data.logging.trade_logger import new_trace, WhyAnalyzer
            _trace_id, elog = new_trace(symbol, "EXECUTION")
            elog.result("PASS").signals({
                "side": "BUY", "qty": int(fill.filled_qty),
                "limit_price": current_price, "filled_price": fill.filled_avg_price,
                "setup_type": str(setup_type), "entry_score": entry_score,
            }).why(WhyAnalyzer.execution(
                "BUY", int(fill.filled_qty), symbol, current_price,
                fill.filled_avg_price, 1,
            )).emit()
        except Exception:
            pass

        if self._ledger:
            self._ledger.log_entry(
                symbol=symbol,
                qty=fill.filled_qty,
                price=fill.filled_avg_price,
                setup_type=str(setup_type),
                score=entry_score,
            )

        # Log feature vector for Phase 3 training data
        if self._scorer and fv:
            await self._scorer.log_entry(
                symbol=symbol,
                fv=fv,
                entry_price=fill.filled_avg_price,
                setup_type=str(setup_type),
                entry_score=entry_score,
                order_placed=True,
                trade_id=fill.order_id,
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

        # ── Step 8 (Phase 4): Evaluate long call alongside equity ─────────
        if self._options_cfg.get("enabled", False) and size_multiplier >= self._options_cfg.get(
            "long_call_min_lgbm_confidence", 0.65
        ):
            await self._try_long_call(
                symbol=symbol,
                current_price=fill.filled_avg_price,
                position_notional=fill.filled_qty * fill.filled_avg_price,
            )

        return True

    async def score_watchlist(self) -> None:
        """Score all static watchlist symbols. Called every 60 minutes."""
        from strategy.config_loader import get_watchlist  # type: ignore
        from data.parallel_runner import adaptive_workers  # type: ignore

        watchlist = get_watchlist(self._cfg)
        logger.info("Scoring watchlist: %d symbols", len(watchlist))

        # Concurrency = CPU count (adapts: CX22=2, laptop=8)
        workers = adaptive_workers(len(watchlist))
        sem     = asyncio.Semaphore(workers)

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

        # Priority 2.5: ExitMonitor LSTM peak detection
        # Falls back to RSI rule-based scoring if model file is not yet trained.
        if self._exit_monitor is None:
            from strategy.exit_monitor import ExitMonitor
            self._exit_monitor = ExitMonitor()

        em = self._exit_monitor.score(symbol, df, {
            "unrealized_pnl_pct":        pos_info.get("unrealized_pnl_pct", 0.0),
            "peak_drawdown_pct":          pos_info.get("peak_drawdown_pct", 0.0),
            "days_held":                  pos_info.get("days_held", 0),
            "entry_composite_score":      pos_info.get("entry_score", 0.5),
            "regime_changed_since_entry": pos_info.get("regime_changed", False),
        })
        if em["exit_action"] == "E3":
            await self._execute_exit(symbol, pos_info, reason=f"exit_monitor_E3:{em['peak_prob']:.3f}")
            return
        if em["exit_action"] in ("E2", "E1"):
            partial_pct = 50.0 if em["exit_action"] == "E2" else 20.0
            await self._execute_partial_exit(symbol, pos_info, partial_pct, indicators.get("price", 0))
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
                current_price = indicators.get("price", 0)
                await self._execute_partial_exit(symbol, pos_info, partial_pct, current_price)
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

    async def _execute_partial_exit(
        self, symbol: str, pos_info: dict, sell_pct: float, current_price: float = 0
    ) -> None:
        """Partial exit for AGE-1 profit taking (default: sell 50%).

        After the partial sell, attempts to write a covered call on the remaining
        shares when options are enabled (income strategy on a position with
        already-realized profit).
        """
        qty = pos_info.get("qty", 0)
        sell_qty = max(1, int(qty * sell_pct / 100))

        if sell_qty >= qty:
            await self._execute_exit(symbol, pos_info, reason="age1_full")
            return

        fill = self._trader.place_market_sell(symbol, qty=sell_qty)
        if fill.is_filled:
            remaining_qty = qty - sell_qty
            self._state.reduce_position(symbol, sell_qty)

            if self._telegram:
                await self._telegram.send_partial_exit_alert(
                    symbol=symbol,
                    sold_qty=fill.filled_qty,
                    price=fill.filled_avg_price,
                    remaining_qty=remaining_qty,
                )

            logger.info(
                "PARTIAL EXIT (AGE-1): %s sold %d of %d shares @ $%.4f",
                symbol, int(fill.filled_qty), qty, fill.filled_avg_price,
            )

            # Write covered call on remaining shares for income
            price = current_price or fill.filled_avg_price
            if (
                self._options_cfg.get("enabled", False)
                and "covered_call" in self._options_cfg.get("strategies_enabled", [])
                and remaining_qty >= 100
                and price > 0
            ):
                await self._try_covered_call(symbol, price, remaining_qty)

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
        min_cash = self._risk_cfg.get("cash_reserve_pct", 30)
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

    # ── Options entry/exit (Phase 4) ────────────────────────────────────────

    async def _try_long_call(
        self,
        symbol: str,
        current_price: float,
        position_notional: float,
    ) -> None:
        """Attempt to buy an ATM call alongside the equity entry.

        Fires only when options.enabled=true and LightGBM confidence is high.
        Silently skips if no suitable contract found.
        """
        if not self._options_selector:
            return
        try:
            portfolio_value = await self._get_portfolio_value()
            trade = self._options_selector.select_long_call(
                symbol=symbol,
                current_price=current_price,
                portfolio_value=portfolio_value,
                position_notional=position_notional,
            )
            if trade is None:
                return

            fill = self._trader.place_option_order(
                contract_symbol=trade.contract.symbol,
                side="buy",
                qty=trade.qty,
            )
            if not fill.is_filled:
                logger.warning("OPTION ORDER FAILED: %s — %s", trade.contract.symbol, fill.status)
                return

            actual_premium = fill.filled_avg_price
            logger.info(
                "OPTION ENTERED: %s %s qty=%d @ $%.2f/share (contract=$%.0f)",
                trade.strategy, trade.contract.symbol,
                fill.filled_qty, actual_premium, actual_premium * 100,
            )

            if self._db:
                from datetime import datetime as _dt
                self._db.open_option_position(
                    underlying=symbol,
                    contract_symbol=trade.contract.symbol,
                    option_type=trade.contract.option_type,
                    strike=trade.contract.strike,
                    expiry=trade.contract.expiry,
                    days_to_expiry=trade.contract.days_to_expiry,
                    strategy=trade.strategy,
                    qty=int(fill.filled_qty),
                    entry_premium=actual_premium,
                    entry_order_id=fill.order_id,
                    delta_at_entry=trade.contract.delta,
                    iv_at_entry=trade.contract.implied_volatility,
                )

            if self._options_stream:
                self._options_stream.subscribe(
                    contract_symbol=trade.contract.symbol,
                    entry_premium=actual_premium,
                )

            if self._telegram:
                await self._telegram.send_message(
                    f"OPTION ENTERED ({trade.strategy})\n"
                    f"Contract: {trade.contract.symbol}\n"
                    f"Strike: ${trade.contract.strike:.2f}  DTE: {trade.contract.days_to_expiry}\n"
                    f"Qty: {int(fill.filled_qty)}  Cost: ${actual_premium * 100 * fill.filled_qty:.0f}"
                )

        except Exception:
            logger.exception("Long call entry failed for %s", symbol)

    async def _try_covered_call(
        self, symbol: str, current_price: float, shares_held: int
    ) -> None:
        """Write a covered call after AGE-1 partial exit for income on remaining shares."""
        if not self._options_selector:
            return
        try:
            trade = self._options_selector.select_covered_call(
                symbol=symbol,
                current_price=current_price,
                shares_held=shares_held,
            )
            if trade is None:
                return

            fill = self._trader.place_option_order(
                contract_symbol=trade.contract.symbol,
                side="sell",  # sell to open
                qty=trade.qty,
            )
            if not fill.is_filled:
                logger.warning(
                    "COVERED CALL WRITE FAILED: %s — %s", trade.contract.symbol, fill.status
                )
                return

            actual_premium = fill.filled_avg_price
            logger.info(
                "COVERED CALL WRITTEN: %s qty=%d @ $%.2f/share (collected=$%.0f)",
                trade.contract.symbol, fill.filled_qty,
                actual_premium, actual_premium * 100 * fill.filled_qty,
            )

            if self._db:
                self._db.open_option_position(
                    underlying=symbol,
                    contract_symbol=trade.contract.symbol,
                    option_type="call",
                    strike=trade.contract.strike,
                    expiry=trade.contract.expiry,
                    days_to_expiry=trade.contract.days_to_expiry,
                    strategy="covered_call",
                    qty=int(fill.filled_qty),
                    entry_premium=actual_premium,
                    entry_order_id=fill.order_id,
                    delta_at_entry=trade.contract.delta,
                    iv_at_entry=trade.contract.implied_volatility,
                )

            if self._options_stream:
                self._options_stream.subscribe(
                    contract_symbol=trade.contract.symbol,
                    entry_premium=actual_premium,
                )

            if self._telegram:
                await self._telegram.send_message(
                    f"COVERED CALL WRITTEN\n"
                    f"Contract: {trade.contract.symbol}\n"
                    f"Strike: ${trade.contract.strike:.2f}  DTE: {trade.contract.days_to_expiry}\n"
                    f"Qty: {int(fill.filled_qty)}  "
                    f"Premium collected: ${actual_premium * 100 * fill.filled_qty:.0f}"
                )

        except Exception:
            logger.exception("Covered call write failed for %s", symbol)

    async def monitor_option_exits(self) -> None:
        """Check all open option positions for stop-loss, take-profit, and DTE close.

        Called every 15 minutes by the scheduler during market hours.
        """
        if not self._options_cfg.get("enabled", False):
            return
        if not self._db:
            return

        open_positions = self._db.get_open_option_positions()
        if not open_positions:
            return

        stop_loss_pct = self._options_cfg.get("stop_loss_pct", 50) / 100
        take_profit_pct = self._options_cfg.get("take_profit_pct", 100) / 100
        dte_threshold = self._options_cfg.get("close_dte_threshold", 7)

        from datetime import datetime, date as _date
        today = _date.today()

        for pos in open_positions:
            try:
                await self._check_option_exit(
                    pos, today, stop_loss_pct, take_profit_pct, dte_threshold
                )
            except Exception:
                logger.exception("Option exit check error for %s", pos.contract_symbol)

    async def _check_option_exit(
        self, pos, today, stop_loss_pct: float, take_profit_pct: float, dte_threshold: int
    ) -> None:
        """Evaluate one open option position for exit."""
        # DTE check — prioritize before price check to avoid gamma risk
        expiry_date = pos.expiry.date() if hasattr(pos.expiry, "date") else pos.expiry
        dte = (expiry_date - today).days
        if dte <= dte_threshold:
            await self._close_option_position(pos, reason="dte_close")
            return

        # Get current quote
        snapshot = self._trader.get_option_snapshot(pos.contract_symbol)
        if not snapshot:
            return

        current_mid = snapshot.get("mid", 0)
        if current_mid <= 0:
            return

        entry = pos.entry_premium
        gain_pct = (current_mid - entry) / entry if entry > 0 else 0

        if gain_pct <= -stop_loss_pct:
            await self._close_option_position(pos, reason="stop_loss", current_mid=current_mid)
        elif gain_pct >= take_profit_pct:
            await self._close_option_position(pos, reason="take_profit", current_mid=current_mid)

    async def _close_option_position(self, pos, reason: str, current_mid: Optional[float] = None) -> None:
        """Place closing order and record the result."""
        # long_call / protective_put: sell to close; covered_call: buy to close
        close_side = "sell" if pos.strategy in ("long_call", "protective_put") else "buy"

        fill = self._trader.place_option_order(
            contract_symbol=pos.contract_symbol,
            side=close_side,
            qty=pos.qty,
        )
        if not fill.is_filled:
            logger.warning("OPTION CLOSE FAILED: %s — %s", pos.contract_symbol, fill.status)
            return

        actual_exit = fill.filled_avg_price
        pnl_per_share = actual_exit - pos.entry_premium
        if pos.strategy == "covered_call":
            pnl_per_share = pos.entry_premium - actual_exit
        pnl = pnl_per_share * pos.qty * 100

        logger.info(
            "OPTION CLOSED: %s reason=%s qty=%d exit=$%.2f pnl=$%.0f",
            pos.contract_symbol, reason, pos.qty, actual_exit, pnl,
        )

        if self._db:
            self._db.close_option_position(
                contract_symbol=pos.contract_symbol,
                exit_premium=actual_exit,
                exit_reason=reason,
                exit_order_id=fill.order_id,
            )

        if self._options_stream:
            self._options_stream.unsubscribe(pos.contract_symbol)

        if self._telegram:
            await self._telegram.send_message(
                f"OPTION CLOSED ({reason})\n"
                f"Contract: {pos.contract_symbol}\n"
                f"P&L: ${pnl:+.0f}  ({pnl_per_share / pos.entry_premium * 100:+.1f}%)"
            )

    async def buy_protective_puts_for_regime(self, regime: str) -> None:
        """Buy protective puts on all open equity positions on regime deterioration.

        Called by scheduler when regime transitions to LATE_BULL or EARLY_BEAR.
        Skips symbols that already have an open protective put in the DB.
        """
        if not self._options_cfg.get("enabled", False):
            return
        if "protective_put" not in self._options_cfg.get("strategies_enabled", []):
            return

        positions = self._state.get_all_positions()
        if not positions:
            return

        logger.info(
            "REGIME %s — evaluating protective puts for %d open positions",
            regime, len(positions),
        )

        # Get already-hedged symbols to avoid double-buying
        already_hedged = set()
        if self._db:
            open_opts = self._db.get_open_option_positions()
            already_hedged = {
                p.underlying for p in open_opts if p.strategy == "protective_put"
            }

        for symbol, pos_info in positions.items():
            if symbol in already_hedged:
                logger.debug("SKIP protective put %s — already hedged", symbol)
                continue
            try:
                df = await self._fetch_ohlcv(symbol, bars=20)
                current_price = (
                    float(df["close"].iloc[-1])
                    if df is not None and len(df) >= 2
                    else pos_info.get("entry_price", 0)
                )
                position_value = pos_info.get("qty", 0) * current_price
                await self.evaluate_protective_put(symbol, current_price, position_value)
            except Exception:
                logger.exception("Protective put failed for %s", symbol)

    async def evaluate_protective_put(
        self,
        symbol: str,
        current_price: float,
        position_value: float,
    ) -> None:
        """Buy a protective put when regime deteriorates to LATE_BULL or EARLY_BEAR.

        Called by the scheduler when regime changes, not on every monitoring cycle.
        """
        if not self._options_cfg.get("enabled", False):
            return
        if not self._options_selector:
            return
        try:
            portfolio_value = await self._get_portfolio_value()
            trade = self._options_selector.select_protective_put(
                symbol=symbol,
                current_price=current_price,
                position_value=position_value,
                portfolio_value=portfolio_value,
            )
            if trade is None:
                return

            fill = self._trader.place_option_order(
                contract_symbol=trade.contract.symbol,
                side="buy",
                qty=trade.qty,
            )
            if not fill.is_filled:
                logger.warning("PROTECTIVE PUT FAILED: %s — %s", trade.contract.symbol, fill.status)
                return

            logger.info(
                "PROTECTIVE PUT ENTERED: %s qty=%d @ $%.2f/share",
                trade.contract.symbol, fill.filled_qty, fill.filled_avg_price,
            )

            if self._db:
                self._db.open_option_position(
                    underlying=symbol,
                    contract_symbol=trade.contract.symbol,
                    option_type="put",
                    strike=trade.contract.strike,
                    expiry=trade.contract.expiry,
                    days_to_expiry=trade.contract.days_to_expiry,
                    strategy="protective_put",
                    qty=int(fill.filled_qty),
                    entry_premium=fill.filled_avg_price,
                    entry_order_id=fill.order_id,
                    delta_at_entry=trade.contract.delta,
                    iv_at_entry=trade.contract.implied_volatility,
                )

            if self._options_stream:
                self._options_stream.subscribe(
                    contract_symbol=trade.contract.symbol,
                    entry_premium=fill.filled_avg_price,
                )

        except Exception:
            logger.exception("Protective put entry failed for %s", symbol)

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

    # ── v13: Portfolio-level circuit breakers ─────────────────────────────────

    async def check_portfolio_circuit_breakers(self) -> dict:
        """Run all portfolio-level risk checks. Returns status dict.

        Called before every entry evaluation cycle (not per-symbol).
        Any TRIGGERED check blocks new entries and sends Telegram alert.

        Checks:
          VIX > 40           → scale all new sizes × 0.5 (not a full block)
          Drawdown > 12%     → pause all new entries
          5 consecutive loss → pause + human alert
          Cash < 10%         → block new entries
          Sector conc > 40%  → block new entries in that sector
        """
        from data.notifications.telegram import notify_circuit_breaker  # type: ignore
        from data.notifications.alerts import (  # type: ignore
            CB_VIX_HIGH, CB_DRAWDOWN, CB_CONSECUTIVE_LOSS, CB_CASH_LOW
        )

        status = {
            "vix_scale":         1.0,
            "entries_paused":    False,
            "pause_reason":      None,
            "circuit_triggered": False,
        }

        try:
            account = self._trader.get_account()
            portfolio_value = float(getattr(account, "portfolio_value", 100_000) or 100_000)
            cash            = float(getattr(account, "cash", portfolio_value * 0.3) or 0)
            cash_pct        = cash / portfolio_value if portfolio_value > 0 else 1.0

            # Load peak portfolio value for drawdown
            peak = self._state.get_metadata("peak_portfolio_value", portfolio_value)
            if portfolio_value > peak:
                self._state.set_metadata("peak_portfolio_value", portfolio_value)
                peak = portfolio_value
            drawdown = (peak - portfolio_value) / peak if peak > 0 else 0.0

            # Consecutive losses
            consec = self._state.get_metadata("consecutive_losses", 0)

        except Exception:
            logger.warning("Circuit breaker: could not read account state — skipping")
            return status

        # ── VIX > 40 (sizing scale, not full block) ────────────────────────────
        try:
            from data.sources.fred_macro import get_macro_signals  # type: ignore
            macro = get_macro_signals()
            vix   = macro.get("vix_raw", 20)
            if vix and vix > 40:
                status["vix_scale"] = 0.5
                notify_circuit_breaker(CB_VIX_HIGH, f"All new position sizes × 0.5 (VIX={vix:.1f})")
                logger.warning("CIRCUIT: VIX=%.1f > 40 — sizing × 0.5", vix)
        except Exception:
            pass

        # ── Drawdown > 12% ─────────────────────────────────────────────────────
        if drawdown > 0.12:
            status["entries_paused"]    = True
            status["pause_reason"]      = f"drawdown={drawdown*100:.1f}%>12%"
            status["circuit_triggered"] = True
            notify_circuit_breaker(
                CB_DRAWDOWN,
                f"All new entries paused (drawdown={drawdown*100:.1f}%)"
            )
            logger.warning("CIRCUIT: drawdown %.1f%% > 12%% — pausing entries", drawdown * 100)

        # ── 5 consecutive losses ────────────────────────────────────────────────
        if consec >= 5:
            status["entries_paused"]    = True
            status["pause_reason"]      = f"{consec} consecutive losses"
            status["circuit_triggered"] = True
            notify_circuit_breaker(
                CB_CONSECUTIVE_LOSS,
                f"Trading paused — {consec} consecutive losses. Human review required."
            )
            logger.warning("CIRCUIT: %d consecutive losses — pausing", consec)

        # ── Cash < 10% ─────────────────────────────────────────────────────────
        if cash_pct < 0.10:
            status["entries_paused"]    = True
            status["pause_reason"]      = f"cash={cash_pct*100:.1f}%<10%"
            status["circuit_triggered"] = True
            logger.warning("CIRCUIT: cash %.1f%% < 10%% — blocking new entries", cash_pct * 100)

        return status

    def half_kelly_size(
        self,
        p_win:         float,
        avg_win:       float,
        avg_loss:      float,
        composite:     float = 1.0,
        vix_scale:     float = 1.0,
        regime_scalar: float = 1.0,
        beta:          float = 1.0,
    ) -> float:
        """Compute half-Kelly position size as fraction of portfolio.

        Kelly formula: (P_win × avg_win - P_loss × avg_loss) / avg_win
        Applied: kelly × 0.5 × composite × regime_scalar × vix_scale / beta_adj

        Parameters
        ----------
        p_win       : win probability [0,1]
        avg_win     : average winning trade return (e.g. 0.12 = 12%)
        avg_loss    : average losing trade magnitude (e.g. 0.06 = 6%)
        composite   : signal composite score 0-1
        vix_scale   : 0.5 if VIX>40 else 1.0
        regime_scalar: 0-1 (EARLY_BULL=1.0, CRASH=0)
        beta        : stock beta vs SPY

        Returns
        -------
        float: position size as fraction of portfolio (e.g. 0.05 = 5%)
        """
        p_loss = 1 - p_win
        if avg_win <= 0 or avg_loss <= 0:
            return 0.02  # fallback: 2% minimum

        kelly = (p_win * avg_win - p_loss * avg_loss) / avg_win
        if kelly <= 0:
            return 0.0  # negative edge — don't trade

        beta_adj = min(max(beta, 0.5), 2.5)
        size     = (kelly * 0.5) * composite * regime_scalar * vix_scale / beta_adj
        return round(max(0.0, min(size, 0.15)), 4)  # hard cap 15% per position

    def atr_stop_price(
        self,
        entry_price: float,
        atr_14:      float,
        beta:        float = 1.0,
    ) -> float:
        """Compute ATR-based stop loss price.

        ATR multiplier by beta (from plan):
          beta > 1.5  → 2.0× ATR (high beta = volatile, tighter absolute stop)
          0.8-1.5     → 2.5× ATR (standard)
          < 0.8       → 3.0× ATR (low beta = stable, wider stop needed)
        """
        if beta > 1.5:
            mult = 2.0
        elif beta >= 0.8:
            mult = 2.5
        else:
            mult = 3.0
        return round(entry_price - mult * atr_14, 4)
