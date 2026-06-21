"""
scheduler.py — APScheduler Orchestrator

Replaces TradingSystem's blocking `schedule` library with APScheduler backed by
asyncio. Key advantages:
  - Non-blocking: all jobs run as asyncio coroutines
  - PostgreSQL job store: jobs survive process restart
  - Cron expressions: exact day/hour targeting (e.g. Sunday 2am screener refresh)
  - Startup gate: async tasks (WebSocket streams) start first; scheduling begins
    only after a startup_done Event is set — mirrors blm_terminal_windy's pattern

Startup sequence (all parallel):
  1. Load config
  2. Connect to PostgreSQL
  3. Start AlpacaNewsStream WebSocket
  4. Start EdgarMonitor polling loop
  5. Start TradingViewScreener polling loop
  6. Set startup_done → scheduler begins processing regular jobs

Regular jobs (all non-overlapping — misfire_grace_time prevents pileup):
  Every 5 min:   fetch_news_rss, monitor_open_positions
  Every 60 min:  score_watchlist_and_enter
  Every 4 h:     update_regime
  Sunday 2am:    refresh_fundamentals
  Sunday 4am:    refresh_screener_picks
  Daily 4:30pm:  generate_performance_report
"""

import asyncio
import logging
import os
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
from apscheduler.triggers.cron import CronTrigger  # type: ignore
from apscheduler.triggers.interval import IntervalTrigger  # type: ignore

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Central orchestrator: starts all data streams and schedules all periodic jobs."""

    def __init__(self, config: dict):
        self._cfg = config
        self._scheduler = AsyncIOScheduler()
        self._startup_done = asyncio.Event()

        # These are injected after construction (dependency injection)
        self.hot_universe = None
        self.news_stream = None          # AlpacaNewsStream
        self.edgar_monitor = None        # EdgarMonitor
        self.tv_screener = None          # TradingViewScreener
        self.news_aggregator = None      # NewsAggregator
        self.position_manager = None     # PositionManager
        self.alpaca_trader = None        # AlpacaTrader
        self.regime_detector = None      # RegimeDetector
        self.telegram = None             # TelegramAlerts
        self.training_pipeline = None    # TrainingPipeline (Phase 3)
        self.options_stream = None       # OptionDataStreamManager (Phase 4)

    async def start(self) -> None:
        """Run the full startup sequence then hand off to the scheduler loop."""
        logger.info("=== Trading System Startup ===")

        # Phase 1: Start always-on background data streams (parallel)
        stream_tasks = []
        if self.news_stream:
            stream_tasks.append(asyncio.create_task(
                self.news_stream.start(), name="alpaca_news_stream"
            ))
            logger.info("[1/3] Alpaca NewsDataStream started")

        if self.edgar_monitor:
            stream_tasks.append(asyncio.create_task(
                self.edgar_monitor.start(), name="edgar_monitor"
            ))
            logger.info("[2/3] EDGAR monitor started")

        if self.tv_screener:
            stream_tasks.append(asyncio.create_task(
                self.tv_screener.start(), name="tv_screener"
            ))
            logger.info("[3/3] TradingView Screener started")

        if self.options_stream and self._cfg.get("options", {}).get("enabled", False):
            stream_tasks.append(asyncio.create_task(
                self.options_stream.start(), name="option_data_stream"
            ))
            logger.info("[4/4] Option data stream started")

        # Phase 2: Register HOT_UNIVERSE callback → triggers immediate entry eval
        if self.hot_universe and self.position_manager:
            self.hot_universe.on_symbol_added(self._on_hot_symbol)

        # Phase 3: Run initial regime update before scheduling begins
        if self.regime_detector:
            await self._update_regime()

        # Phase 4: Schedule all periodic jobs
        self._register_jobs()
        self._scheduler.start()

        # Signal that startup is complete
        self._startup_done.set()
        logger.info("=== Startup complete — scheduler running ===")

        # Keep the event loop alive; stream_tasks run until stop() is called
        if stream_tasks:
            await asyncio.gather(*stream_tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        if self.news_stream:
            await self.news_stream.stop()
        if self.edgar_monitor:
            await self.edgar_monitor.stop()
        if self.tv_screener:
            await self.tv_screener.stop()
        if self.options_stream:
            await self.options_stream.stop()
        logger.info("Trading System stopped")

    def _register_jobs(self) -> None:
        """Register all APScheduler jobs."""
        sched_cfg = self._cfg.get("scheduler", {})

        # ── Every N minutes jobs ────────────────────────────────────────────
        self._scheduler.add_job(
            self._fetch_news_rss,
            trigger=IntervalTrigger(minutes=sched_cfg.get("news_rss_interval_minutes", 5)),
            id="fetch_news_rss",
            max_instances=1,
            misfire_grace_time=60,
            name="Fetch 39-source RSS news",
        )

        self._scheduler.add_job(
            self._monitor_positions,
            trigger=IntervalTrigger(minutes=sched_cfg.get("position_monitor_interval_minutes", 5)),
            id="monitor_positions",
            max_instances=1,
            misfire_grace_time=60,
            name="Monitor open positions for exits",
        )

        self._scheduler.add_job(
            self._score_watchlist,
            trigger=IntervalTrigger(minutes=sched_cfg.get("scoring_interval_minutes", 60)),
            id="score_watchlist",
            max_instances=1,
            misfire_grace_time=300,
            name="Score static watchlist and evaluate entries",
        )

        self._scheduler.add_job(
            self._update_regime,
            trigger=IntervalTrigger(hours=sched_cfg.get("regime_update_interval_hours", 4)),
            id="update_regime",
            max_instances=1,
            misfire_grace_time=600,
            name="Update 8-state market regime",
        )

        # ── Weekly maintenance jobs ─────────────────────────────────────────
        self._scheduler.add_job(
            self._refresh_fundamentals,
            trigger=CronTrigger(
                day_of_week=sched_cfg.get("fundamentals_refresh_day", "sun"),
                hour=sched_cfg.get("fundamentals_refresh_hour", 2),
            ),
            id="refresh_fundamentals",
            max_instances=1,
            misfire_grace_time=3600,
            name="Refresh fundamentals from EDGAR + FMP",
        )

        self._scheduler.add_job(
            self._refresh_screener,
            trigger=CronTrigger(
                day_of_week=sched_cfg.get("screener_refresh_day", "sun"),
                hour=sched_cfg.get("screener_refresh_hour", 4),
            ),
            id="refresh_screener",
            max_instances=1,
            misfire_grace_time=3600,
            name="Refresh static watchlist from screener picks",
        )

        # ── Daily post-market report ────────────────────────────────────────
        self._scheduler.add_job(
            self._generate_performance_report,
            trigger=CronTrigger(
                hour=sched_cfg.get("performance_report_hour", 16),
                minute=sched_cfg.get("performance_report_minute", 30),
            ),
            id="performance_report",
            max_instances=1,
            misfire_grace_time=1800,
            name="Generate daily quantstats performance report",
        )

        # ── Phase 3: LightGBM meta-model retrain (Sunday 6am) ──────────────
        self._scheduler.add_job(
            self._retrain_meta_model,
            trigger=CronTrigger(
                day_of_week=sched_cfg.get("model_retrain_day", "sun"),
                hour=sched_cfg.get("model_retrain_hour", 6),
            ),
            id="retrain_meta_model",
            max_instances=1,
            misfire_grace_time=7200,
            name="Retrain LightGBM meta-model on live trade labels",
        )

        # ── Phase 4: Option position monitoring (every 15 min, only if enabled) ──
        options_cfg = self._cfg.get("options", {})
        if options_cfg.get("enabled", False):
            self._scheduler.add_job(
                self._monitor_options,
                trigger=IntervalTrigger(
                    minutes=options_cfg.get("monitor_interval_minutes", 15)
                ),
                id="monitor_options",
                max_instances=1,
                misfire_grace_time=120,
                name="Monitor open option positions for exit signals",
            )

        logger.info("Registered %d scheduler jobs", len(self._scheduler.get_jobs()))

    # ── Job implementations ─────────────────────────────────────────────────

    async def _fetch_news_rss(self) -> None:
        """Run the 39-source RSS aggregator cycle."""
        if self.news_aggregator:
            try:
                await self.news_aggregator._fetch_all_sources()
            except Exception:
                logger.exception("news_rss job error")

    async def _monitor_positions(self) -> None:
        """Score all open positions for exit signals."""
        if not self.position_manager:
            return
        try:
            await self.position_manager.monitor_exits()
        except Exception:
            logger.exception("monitor_positions job error")

    async def _score_watchlist(self) -> None:
        """Score all static watchlist symbols and evaluate entries."""
        if not self.position_manager:
            return
        try:
            await self.position_manager.score_watchlist()
        except Exception:
            logger.exception("score_watchlist job error")

    async def _update_regime(self) -> None:
        """Detect current market regime (SPY trend + VIX + breadth)."""
        if not self.regime_detector:
            return
        try:
            # regime_detector.detect() is typically synchronous CPU work
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.regime_detector.detect)
        except Exception:
            logger.exception("update_regime job error")

    async def _refresh_fundamentals(self) -> None:
        """Weekly fundamental data refresh for all watchlist symbols."""
        logger.info("Weekly fundamentals refresh starting...")
        # Placeholder — fundamentals_fetcher will implement this
        # Called Sunday at 2am so FMP quota resets before market open Monday

    async def _retrain_meta_model(self) -> None:
        """Sunday 6am: retrain LightGBM on live paper trade labels.

        Only retrains if enough new labeled samples have accumulated since
        last retrain (threshold: config.json model.retrain_min_samples).
        """
        if not self.training_pipeline:
            logger.debug("training_pipeline not wired — skipping retrain")
            return
        try:
            logger.info("Starting LightGBM meta-model retrain...")
            result = await self.training_pipeline.run_live_update()
            if result:
                logger.info(
                    "Meta-model retrain complete: samples=%d accuracy=%.3f",
                    result.get("n_samples", 0),
                    result.get("train_accuracy", 0),
                )
                if self.telegram:
                    await self.telegram.send_message(
                        f"🤖 Meta-model retrained\n"
                        f"Samples: {result.get('n_samples')}\n"
                        f"Train acc: {result.get('train_accuracy', 0):.1%}\n"
                        f"WF P(win) acc: {result.get('avg_high_conf_accuracy', 'N/A')}"
                    )
            else:
                logger.info("Retrain skipped — insufficient labeled samples")
        except Exception:
            logger.exception("Meta-model retrain job failed")

    async def _monitor_options(self) -> None:
        """Check open option positions for stop-loss, take-profit, and DTE close."""
        if not self.position_manager:
            return
        try:
            await self.position_manager.monitor_option_exits()
        except Exception:
            logger.exception("monitor_options job error")

    async def _refresh_screener(self) -> None:
        """Weekly screener picks refresh — updates config watchlist."""
        logger.info("Weekly screener picks refresh starting...")
        # Placeholder — tv_screener batch run for Sunday maintenance

    async def _generate_performance_report(self) -> None:
        """Generate quantstats HTML report after market close."""
        logger.info("Generating post-market performance report...")
        # Placeholder — performance_calculator.py will implement this

    def _on_hot_symbol(self, symbol: str, source: str) -> None:
        """Callback fired by HOT_UNIVERSE when a new symbol enters.

        Schedules immediate entry evaluation for this symbol without waiting
        for the next 60-minute watchlist scoring cycle.
        """
        logger.info("HOT_UNIVERSE callback: %s via %s → immediate entry eval", symbol, source)

        # Schedule as a one-shot job that runs ASAP (not in the callback thread)
        self._scheduler.add_job(
            self._evaluate_hot_entry,
            args=[symbol, source],
            id=f"hot_{symbol}_{int(datetime.utcnow().timestamp())}",
            max_instances=1,
            misfire_grace_time=30,
        )

    async def _evaluate_hot_entry(self, symbol: str, source: str) -> None:
        """Evaluate a HOT_UNIVERSE symbol for entry immediately."""
        if not self.position_manager:
            return
        try:
            await self.position_manager.evaluate_entry(symbol, source=source)
        except Exception:
            logger.exception("Hot entry eval error for %s", symbol)
