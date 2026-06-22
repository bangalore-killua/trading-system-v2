"""
run_server.py — Entry Point

Wires all components together and starts the system:
  1. Load config.json and .env
  2. Instantiate all services (dependency injection — no global singletons in modules)
  3. Start FastAPI in background
  4. Hand off to TradingScheduler (blocks, manages all async tasks)

Usage:
    python run_server.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Bootstrap ───────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/system.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


async def main() -> None:
    cfg = load_config()
    logger.info("Config loaded — paper=%s", cfg.get("alpaca", {}).get("paper", True))

    # ── Instantiate data layer ───────────────────────────────────────────────
    from data.hot_universe import get_hot_universe
    from data.news_stream import AlpacaNewsStream
    from data.edgar_monitor import EdgarMonitor
    from data.tv_screener import TradingViewScreener
    from data.news_aggregator import NewsAggregator
    from data.openbb_provider import OpenBBProvider

    hot_universe = get_hot_universe(cfg.get("universe", {}).get("hot_universe_ttl_hours"))

    # OpenBB provider — lazy init, all components degrade gracefully if unavailable
    openbb = OpenBBProvider(
        pat=os.environ.get("OPENBB_PAT", ""),
        fmp_api_key=os.environ.get("FMP_API_KEY", ""),
    )

    # ── Instantiate NLP layer ────────────────────────────────────────────────
    from strategy.threat_classifier import ThreatClassifier
    from strategy.news_sentiment import FinBERTSentiment

    threat_clf = ThreatClassifier()

    # FinBERT loads ~400MB on first inference call — lazy, not at startup
    sentiment_model = FinBERTSentiment(device="cpu")

    # ── Instantiate broker ───────────────────────────────────────────────────
    from server.alpaca_trader import AlpacaTrader

    trader = AlpacaTrader.from_env()

    # ── Instantiate strategy layer ───────────────────────────────────────────
    from strategy import technical_analysis as ta_module
    from strategy.regime_detector import RegimeDetector  # type: ignore
    from strategy.signal_basket import SignalBasket      # type: ignore
    from strategy.trade_ledger import TradeLedger        # type: ignore
    from strategy.position_aging import PositionAging    # type: ignore
    from strategy.correlation_risk import CorrelationRisk  # type: ignore
    from strategy.trade_state import TradeState          # type: ignore
    from data.fetcher import OHLCVFetcher
    from data.news_relevance_engine import NewsRelevanceEngine
    from data.fundamentals_fetcher import FundamentalsFetcher
    from strategy.macro_analyser import MacroAnalyser
    from strategy.chart_pattern_detector import ChartPatternDetector
    from strategy.scorer import Scorer
    from strategy.position_manager import PositionManager

    # Data fetcher — primary OHLCV source for all entry/exit evaluations
    ohlcv_fetcher = OHLCVFetcher(
        alpaca_api_key=os.environ.get("ALPACA_API_KEY"),
        alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY"),
        alpaca_paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
    )

    # Ticker extractor — maps article text to equity symbols for HOT_UNIVERSE
    relevance_engine = NewsRelevanceEngine()

    macro_analyser = MacroAnalyser(use_fred=True, openbb_provider=openbb)
    regime_detector = RegimeDetector(cfg)

    fundamentals_fetcher = FundamentalsFetcher(
        fmp_api_key=os.environ.get("FMP_API_KEY", ""),
        openbb_provider=openbb,
    )

    # YOLOv8 chart pattern detector — returns all-zero until model is trained
    chart_detector = ChartPatternDetector(
        model_path=cfg.get("model", {}).get(
            "chart_pattern_model_path", "models/chart_patterns_yolov8n.pt"
        ),
        confidence_threshold=0.40,
        lookback_bars=60,
    )

    # LightGBM meta-model — loads from disk if pkl exists, else graceful no-op
    from strategy.meta_model import MetaModel
    from strategy.training_pipeline import TrainingPipeline

    meta_model = MetaModel(
        model_path=cfg.get("model", {}).get(
            "meta_model_path", "models/meta_model_lgbm.pkl"
        ),
        config=cfg,
    )
    if meta_model.is_ready:
        logger.info("LightGBM meta-model loaded (%d features)", len(meta_model._feature_names))
    else:
        logger.info("LightGBM meta-model not found — using full-size for all trades (Phase 1/2 mode)")

    from database import get_db
    db = get_db()

    scorer = Scorer(
        config=cfg,
        chart_detector=chart_detector,
        fundamentals_fetcher=fundamentals_fetcher,
        macro_analyser=macro_analyser,
        regime_detector=regime_detector,
        db_session=db,
        meta_model=meta_model if meta_model.is_ready else None,
    )

    training_pipeline = TrainingPipeline(
        config=cfg,
        db_url=os.environ.get("DATABASE_URL"),
    )

    signal_basket = SignalBasket(cfg)
    trade_ledger = TradeLedger(cfg)
    position_aging = PositionAging(cfg)
    correlation_risk = CorrelationRisk(cfg)
    trade_state = TradeState()

    from server.telegram_alerts import TelegramAlerts  # type: ignore

    telegram = TelegramAlerts(
        token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
    )

    position_manager = PositionManager(
        config=cfg,
        alpaca_trader=trader,
        signal_basket=signal_basket,
        regime_detector=regime_detector,
        technical_analysis=ta_module,
        threat_classifier=threat_clf,
        news_sentiment=sentiment_model,
        trade_ledger=trade_ledger,
        correlation_risk=correlation_risk,
        position_aging=position_aging,
        trade_state=trade_state,
        hot_universe=hot_universe,
        telegram=telegram,
        data_fetcher=ohlcv_fetcher,
    )
    position_manager._scorer = scorer  # Phase 3: feature vector + P(win) multiplier
    position_manager._db = db

    # ── Phase 4: Options layer ───────────────────────────────────────────────
    from data.options_chain import OptionsChainFetcher
    from strategy.options_selector import OptionsSelector
    from server.option_stream import OptionDataStreamManager

    options_cfg = cfg.get("options", {})
    if options_cfg.get("enabled", False):
        options_chain_fetcher = OptionsChainFetcher(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            config=cfg,
        )
        options_selector = OptionsSelector(chain_fetcher=options_chain_fetcher, config=cfg)
        options_stream = OptionDataStreamManager(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
        )
        position_manager._options_selector = options_selector
        position_manager._options_stream = options_stream
        logger.info("Phase 4 options layer enabled — strategies: %s", options_cfg.get("strategies_enabled", []))
    else:
        options_stream = None
        logger.info("Phase 4 options disabled (config options.enabled=false)")

    # ── Wire up streams ──────────────────────────────────────────────────────
    news_stream = AlpacaNewsStream(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        hot_universe=hot_universe,
        nlp_pipeline=lambda text: (
            sentiment_model.score(text),
            threat_clf.classify(text),
        ),
        sentiment_threshold=cfg.get("universe", {}).get("news_sentiment_threshold_hot", 0.70),
    )

    edgar_monitor = EdgarMonitor(
        hot_universe=hot_universe,
        poll_interval_seconds=cfg.get("universe", {}).get("edgar_poll_interval_seconds", 60),
    )

    universe_cfg = cfg.get("universe", {})
    tv_screener = TradingViewScreener(
        hot_universe=hot_universe,
        poll_interval_minutes=universe_cfg.get("tv_screener_interval_minutes", 5),
        min_volume_ratio=universe_cfg.get("tv_screener_min_volume_ratio", 3.0),
        min_price_change_pct=universe_cfg.get("tv_screener_min_price_change_pct", 1.5),
        min_price_usd=cfg.get("risk", {}).get("min_price_usd", 5.0),
        min_avg_volume=universe_cfg.get("tv_screener_min_volume", 500_000),
        top_n=universe_cfg.get("tv_screener_top_n", 25),
    )

    wl_cfg = cfg.get("watchlist", {})
    _static_symbols = (
        wl_cfg.get("mandatory", []) + wl_cfg.get("screener_picks", []) + wl_cfg.get("manual_additions", [])
        if isinstance(wl_cfg, dict) else list(wl_cfg)
    )

    news_aggregator = NewsAggregator(
        hot_universe=hot_universe,
        threat_classifier=threat_clf,
        sentiment_model=sentiment_model,
        poll_interval_minutes=cfg.get("news", {}).get("rss_fetch_interval_minutes", 5),
        sentiment_threshold=cfg.get("news", {}).get("sentiment_threshold_buy", 0.70),
        ticker_extractor=relevance_engine.extract_tickers,
        openbb_provider=openbb,
        watchlist=_static_symbols,
    )

    # ── Start scheduler (blocks until stop()) ───────────────────────────────
    from server.scheduler import TradingScheduler

    scheduler = TradingScheduler(cfg)
    scheduler.hot_universe = hot_universe
    scheduler.news_stream = news_stream
    scheduler.edgar_monitor = edgar_monitor
    scheduler.tv_screener = tv_screener
    scheduler.news_aggregator = news_aggregator
    scheduler.position_manager = position_manager
    scheduler.alpaca_trader = trader
    scheduler.regime_detector = regime_detector
    scheduler.telegram = telegram
    scheduler.training_pipeline = training_pipeline      # Phase 3: Sunday retrain job
    scheduler.options_stream = options_stream            # Phase 4: option quote stream
    scheduler.fundamentals_fetcher = fundamentals_fetcher  # weekly Sunday refresh
    scheduler.db = db                                    # daily performance report

    logger.info("Starting trading system...")
    await scheduler.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt")
