"""
scorer.py — Feature Vector Builder + Trade Logger

Assembles the complete feature vector for every trade evaluated by the signal basket.
Two roles:

  1. TRAINING DATA COLLECTOR: logs the full feature vector per trade entry to
     PostgreSQL. This is the raw material for Phase 3 LightGBM training.

  2. INFERENCE INPUT (Phase 3+): at prediction time, the same build_vector() method
     is called and the vector is passed directly to the LightGBM model.

In Phase 1 and 2: the signal basket still makes the final buy/sell decision.
The scorer just records what the market looked like at entry so we can train later.

Feature vector dimensions:
  ~11 continuous technical   (RSI, MACD, price vs MAs, volume ratio, ATR, ADX...)
  ~14 binary technical       (price above MA, oversold, surge flags...)
  ~8  OHLC geometry          (body size, wicks, gap flags...)
  ~60 candlestick patterns   (pandas-ta CDL flags, all binary)
  ~16 chart patterns         (YOLOv8, binary) — Phase 2+
  ~5  fundamental            (P/E, ROE, revenue growth, EPS revision, D/E)
  ~3  news/NLP               (FinBERT score, news volume, threat level encoded)
  ~4  macro                  (regime, VIX, yield spread, fear/greed)
  ~4  context                (sector, setup type, mkt cap bucket, days to earnings)
  ───────────────────────────
  ~125 total features
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_dynamic_sector_cache() -> dict:
    """Load sector_buckets for dynamically profiled symbols from disk.
    Returns {} silently on any error — static SECTOR_BUCKETS always covers known symbols.
    """
    try:
        p = Path("data/symbol_profiles.json")
        if p.exists():
            with open(p) as f:
                profiles = json.load(f)
            return {sym: v["sector_bucket"] for sym, v in profiles.items() if "sector_bucket" in v}
    except Exception:
        pass
    return {}


# Merged at import time: static known symbols + any previously profiled new symbols.
# Re-merged each time scorer.py is imported (i.e. once per process startup).
_DYNAMIC_SECTOR_CACHE: dict = _load_dynamic_sector_cache()

# Encode categorical fields to integers for LightGBM
REGIME_ENCODING = {
    "EARLY_BULL": 0, "MID_BULL": 1, "LATE_BULL": 2, "EUPHORIA": 3,
    "NEUTRAL": 4,
    "EARLY_BEAR": 5, "MID_BEAR": 6, "CRASH": 7,
}
SETUP_TYPE_ENCODING = {
    "SETUP_MOMENTUM": 0, "SETUP_BOUNCE": 1,
    "SETUP_INFLECTION": 2, "SETUP_NEWS_DRIVEN": 3,
}
THREAT_ENCODING = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "UNKNOWN": 1}
FEAR_GREED_ENCODING = {
    "EXTREME_FEAR": 0, "FEAR": 1, "NEUTRAL": 2, "GREED": 3, "EXTREME_GREED": 4,
}
MKTCAP_ENCODING = {"nano": 0, "micro": 1, "small": 2, "mid": 3, "large": 4, "mega": 5}

# Behavioural sector buckets — finer-grained than GICS standard sectors.
# Key splits that matter for signal IC:
#   SEMICONDUCTOR ≠ TECH_GROWTH  (GPU momentum vs SaaS subscription cycles)
#   BIOTECH ≠ HEALTHCARE          (FDA event spikes vs steady pharma earnings)
#   DEFENSE ≠ CYCLICAL            (congressional trade IC is unique to defense)
#   CRYPTO_PROXY ≠ FINTECH        (BTC correlation vs payment volume cycles)
# AutoGluon reads these as native categoricals — no integer mapping needed.
SECTOR_BUCKETS: dict = {
    "AAPL": "TECH_GROWTH",  "MSFT": "TECH_GROWTH",  "GOOGL": "TECH_GROWTH",
    "META": "TECH_GROWTH",  "AMZN": "TECH_GROWTH",  "TSLA": "TECH_GROWTH",
    "CRM":  "TECH_GROWTH",  "NFLX": "TECH_GROWTH",  "SNOW": "TECH_GROWTH",
    "PLTR": "TECH_GROWTH",  "ORCL": "TECH_GROWTH",
    "CRWD": "TECH_GROWTH",  "PANW": "TECH_GROWTH",
    "NVDA": "SEMICONDUCTOR", "AMD":  "SEMICONDUCTOR", "QCOM": "SEMICONDUCTOR",
    "AVGO": "SEMICONDUCTOR", "INTC": "SEMICONDUCTOR", "TSM":  "SEMICONDUCTOR",
    "MRVL": "SEMICONDUCTOR", "ASML": "SEMICONDUCTOR", "ARM":  "SEMICONDUCTOR",
    "SMCI": "SEMICONDUCTOR",
    "V":    "FINTECH",   "PYPL": "FINTECH",  "HOOD": "FINTECH",
    "COIN": "CRYPTO_PROXY",  "MSTR": "CRYPTO_PROXY",
    "JPM": "FINANCIALS", "GS":  "FINANCIALS", "BAC": "FINANCIALS",
    "WFC": "FINANCIALS", "MS":  "FINANCIALS",
    "LLY": "HEALTHCARE", "UNH": "HEALTHCARE", "JNJ": "HEALTHCARE",
    "PFE": "HEALTHCARE", "ABBV": "HEALTHCARE", "NVO": "HEALTHCARE",
    "GILD": "BIOTECH", "MRNA": "BIOTECH", "BIIB": "BIOTECH",
    "REGN": "BIOTECH",  "VRTX": "BIOTECH",
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY",
    "SLB": "ENERGY", "ENPH": "ENERGY", "OXY": "ENERGY",
    "NKE": "CONSUMER_DISC", "MCD": "CONSUMER_DISC", "HD":   "CONSUMER_DISC",
    "SBUX": "CONSUMER_DISC", "UBER": "CONSUMER_DISC", "COST": "CONSUMER_DISC",
    "PG": "DEFENSIVE", "KO": "DEFENSIVE", "WMT": "DEFENSIVE",
    "NEE": "DEFENSIVE", "AMT": "DEFENSIVE",
    "CAT": "CYCLICAL", "HON": "CYCLICAL", "DE": "CYCLICAL", "GE": "CYCLICAL",
    "LMT": "DEFENSE", "RTX": "DEFENSE", "NOC": "DEFENSE",
    "FCX": "COMMODITY", "NEM": "COMMODITY",
}


class FeatureVector:
    """Builds and validates the complete feature vector for one trade evaluation."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def add_technical(self, indicators: Dict) -> "FeatureVector":
        """Add all values from technical_analysis.compute_indicators() output."""
        # Continuous technical indicators
        for key in (
            "rsi", "macd_hist", "price_vs_20ma", "price_vs_50ma", "price_vs_200ma",
            "bb_position", "bb_width", "volume_ratio", "atr_pct", "adx",
            "stoch_k", "ema9", "ema21",
        ):
            self._data[key] = indicators.get(key)

        # Binary technical signals
        for key in (
            "price_above_20ma", "price_above_50ma", "price_above_200ma",
            "rsi_oversold", "rsi_overbought", "rsi_momentum_zone",
            "macd_bullish", "macd_bullish_cross", "macd_bearish_cross",
            "volume_surge", "volume_confirming", "volume_distribution",
            "bb_squeeze", "adx_strong_trend", "daily_trend_bullish",
            "stoch_oversold", "stoch_overbought", "gap_up", "gap_down",
        ):
            self._data[key] = indicators.get(key, 0)

        # OHLC geometry
        for key in (
            "body_size", "body_direction", "upper_wick", "lower_wick",
            "wick_ratio", "candle_range_vs_atr",
        ):
            self._data[key] = indicators.get(key)

        # Candlestick patterns (all cdl_ prefixed keys)
        for key, val in indicators.items():
            if key.startswith("cdl_") or key in (
                "hammer_at_200ma", "hammer_at_50ma", "engulfing_at_support",
                "marubozu_on_volume_surge", "morning_star_at_support",
                "evening_star_near_highs",
            ):
                self._data[key] = val

        return self

    def add_chart_patterns(self, patterns: Dict) -> "FeatureVector":
        """Add YOLOv8 chart pattern binary flags (Phase 2)."""
        for key, val in patterns.items():
            if key.startswith("chart_"):
                self._data[key] = int(val)
        return self

    def add_fundamentals(self, fundamentals: Optional[Dict]) -> "FeatureVector":
        """Add fundamental data for the symbol."""
        if not fundamentals:
            for key in ("pe_ratio", "roe", "revenue_growth_yoy",
                        "earnings_revision_3m", "debt_equity"):
                self._data[key] = None
            return self

        self._data["pe_ratio"] = fundamentals.get("pe_ratio")
        self._data["roe"] = fundamentals.get("roe")
        self._data["revenue_growth_yoy"] = fundamentals.get("revenue_growth_yoy")
        self._data["earnings_revision_3m"] = fundamentals.get("earnings_revision_3m")
        self._data["debt_equity"] = fundamentals.get("debt_equity")
        self._data["insider_buying_30d"] = int(fundamentals.get("insider_buying_30d", False))
        return self

    def add_news(
        self,
        finbert_score: float = 0.0,
        news_volume_7d: int = 0,
        threat_level: str = "UNKNOWN",
    ) -> "FeatureVector":
        """Add NLP-derived news features."""
        self._data["finbert_score"] = round(finbert_score, 4)
        self._data["news_volume_7d"] = news_volume_7d
        self._data["threat_level_encoded"] = THREAT_ENCODING.get(threat_level, 1)
        return self

    def add_macro(self, macro_snapshot: Optional[Dict]) -> "FeatureVector":
        """Add macro context from macro_analyser.get_macro_snapshot()."""
        if not macro_snapshot:
            self._data.update({
                "vix": None, "yield_spread_10y2y": None,
                "fear_greed_encoded": 2, "spy_above_200ma": None,
            })
            return self

        self._data["vix"] = macro_snapshot.get("vix")
        self._data["yield_spread_10y2y"] = macro_snapshot.get("yield_spread_10y2y")
        self._data["fear_greed_encoded"] = FEAR_GREED_ENCODING.get(
            macro_snapshot.get("fear_greed_proxy", "NEUTRAL"), 2
        )
        self._data["spy_above_200ma"] = macro_snapshot.get("spy_above_200ma")
        self._data["inverted_yield_curve"] = macro_snapshot.get("inverted_yield_curve", 0)
        return self

    def add_context(
        self,
        regime: str = "NEUTRAL",
        setup_type: str = "SETUP_MOMENTUM",
        sector_bucket: str = "OTHER",
        mktcap_bucket: str = "mid",
        days_since_earnings: Optional[int] = None,
    ) -> "FeatureVector":
        """Add categorical context features."""
        self._data["regime_encoded"] = REGIME_ENCODING.get(regime, 4)
        self._data["setup_type_encoded"] = SETUP_TYPE_ENCODING.get(setup_type, 0)
        self._data["sector_bucket"] = sector_bucket  # string — AutoGluon handles natively
        self._data["mktcap_bucket_encoded"] = MKTCAP_ENCODING.get(mktcap_bucket, 3)
        self._data["days_since_earnings"] = days_since_earnings
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete feature dict."""
        return dict(self._data)

    def to_list(self, feature_names: List[str]) -> List[Optional[float]]:
        """Return values as an ordered list matching feature_names (for LightGBM)."""
        return [self._data.get(name) for name in feature_names]

    def feature_count(self) -> int:
        return len(self._data)


class Scorer:
    """Builds feature vectors and logs them to PostgreSQL for Phase 3 training.

    In Phase 1/2: logs every entry evaluation, even if no trade was placed.
    In Phase 3: also calls LightGBM for position sizing.
    """

    def __init__(
        self,
        config: dict,
        chart_detector=None,       # ChartPatternDetector (Phase 2)
        fundamentals_fetcher=None, # FundamentalsFetcher
        macro_analyser=None,       # MacroAnalyser
        regime_detector=None,      # RegimeDetector
        db_session=None,           # SQLAlchemy async session (for logging)
        meta_model=None,           # LightGBM wrapper (Phase 3+)
    ):
        self._cfg = config
        self._chart = chart_detector
        self._fundamentals = fundamentals_fetcher
        self._macro = macro_analyser
        self._regime = regime_detector
        self._db = db_session
        self._meta_model = meta_model

        self._model_enabled = config.get("model", {}).get("meta_model_enabled", False)

    def build_vector(
        self,
        symbol: str,
        df,
        indicators: Dict,
        setup_type: str = "SETUP_MOMENTUM",
        news_data: Optional[Dict] = None,
        sector: str = "Unknown",
    ) -> FeatureVector:
        """Build the complete feature vector for one symbol at one point in time."""
        fv = FeatureVector()

        # Technical indicators + candlestick patterns (from technical_analysis.py)
        fv.add_technical(indicators)

        # YOLOv8 chart patterns (Phase 2 — zeros if model not trained yet)
        if self._chart and df is not None:
            chart_patterns = self._chart.detect(df)
        else:
            from strategy.chart_pattern_detector import _empty_patterns
            chart_patterns = _empty_patterns()
        fv.add_chart_patterns(chart_patterns)

        # Fundamentals (weekly cached)
        fundamentals = None
        if self._fundamentals:
            try:
                fundamentals = self._fundamentals.get_cached(symbol)
            except Exception:
                pass
        fv.add_fundamentals(fundamentals)

        # News sentiment + threat
        if news_data:
            fv.add_news(
                finbert_score=news_data.get("finbert_score", 0.0),
                news_volume_7d=news_data.get("news_volume_7d", 0),
                threat_level=news_data.get("threat_level", "UNKNOWN"),
            )
        else:
            fv.add_news()

        # Macro context
        macro_snapshot = None
        if self._macro:
            try:
                macro_snapshot = self._macro.get_macro_snapshot()
            except Exception:
                pass
        fv.add_macro(macro_snapshot)

        # Categorical context
        regime = "NEUTRAL"
        if self._regime:
            try:
                regime = str(self._regime.get_current_regime())
            except Exception:
                pass

        mktcap = "mid"
        if fundamentals:
            mktcap = fundamentals.get("mktcap_bucket", "mid")

        # Resolve sector: static dict first, then dynamic profile cache, then "OTHER"
        sector_bucket = SECTOR_BUCKETS.get(symbol) or _DYNAMIC_SECTOR_CACHE.get(symbol)
        if sector_bucket is None:
            # New symbol not seen before — trigger background profiling, use OTHER now
            sector_bucket = "OTHER"
            try:
                import threading
                from data.sources.symbol_profiler import classify_symbol
                threading.Thread(target=classify_symbol, args=(symbol,), daemon=True).start()
                logger.info("Triggered background profiling for unknown symbol: %s", symbol)
            except Exception:
                pass

        fv.add_context(
            regime=regime,
            setup_type=setup_type,
            sector_bucket=sector_bucket,
            mktcap_bucket=mktcap,
        )

        return fv

    def get_position_size_multiplier(self, fv: FeatureVector) -> float:
        """Return a size multiplier from LightGBM P(win) prediction.

        In Phase 1/2 (model disabled): always returns 1.0.
        In Phase 3+: returns 1.0 for high confidence, 0.5 for medium, 0.0 for low.
        """
        if not self._model_enabled or self._meta_model is None:
            return 1.0

        model_cfg = self._cfg.get("model", {})
        high_conf = model_cfg.get("meta_model_confidence_high", 0.65)
        low_conf = model_cfg.get("meta_model_confidence_low", 0.52)

        try:
            prob = self._meta_model.predict_proba(fv)
            if prob >= high_conf:
                return 1.0
            elif prob >= low_conf:
                return 0.5
            else:
                return 0.0  # Signal says yes but model says low confidence — skip
        except Exception:
            logger.exception("LightGBM prediction error — defaulting to full size")
            return 1.0

    async def log_entry(
        self,
        symbol: str,
        fv: FeatureVector,
        entry_price: float,
        setup_type: str,
        entry_score: int,
        order_placed: bool,
        trade_id: Optional[str] = None,
    ) -> None:
        """Persist the feature vector to PostgreSQL (trade_features table).

        The triple_barrier_label column stays NULL until the trade closes or
        the retroactive labeling script fills it in.
        """
        if self._db is None:
            return

        try:
            self._db.save_trade_features(
                symbol=symbol,
                evaluated_at=datetime.utcnow(),
                setup_type=setup_type,
                entry_score=entry_score,
                entry_price=entry_price,
                order_placed=order_placed,
                features_json=json.dumps(fv.to_dict(), default=str),
                trade_id=trade_id,
            )
            logger.debug(
                "Feature vector logged for %s (trade=%s, %d features)",
                symbol, trade_id, fv.feature_count(),
            )
        except Exception:
            logger.exception("Failed to log feature vector for %s", symbol)


class StockScorer:
    """High-level scoring facade used by server/api.py.

    Reads pre-computed signals from the database (written by the scoring cycle
    in position_manager.py) and exposes them in a format the dashboard expects.
    Falls back to a stub dict when the database is unavailable or the symbol
    has no stored signal yet.
    """

    def score(self, symbol: str) -> dict:
        """Return the latest stored signal dict for *symbol*, or a stub."""
        try:
            from database import get_db
            db = get_db()
            signals = db.get_latest_signals()
            for s in signals:
                if s.get("symbol", "").upper() == symbol.upper():
                    return s
        except Exception:
            logger.debug("StockScorer: DB unavailable for %s", symbol)

        # Stub — returned when DB is down or symbol has no history
        return {
            "symbol": symbol.upper(),
            "signal": "HOLD",
            "signal_label": "HOLD",
            "normalised_score": 0,
            "overall_display": "N/A",
            "conviction": "INSUFFICIENT",
            "bundle_name": "Unknown",
            "bundle_emoji": "",
            "bundle_id": "",
            "position_pct": 0,
            "price": 0,
            "trade_id": None,
            "timestamp": None,
        }

    def format_scorecard_dashboard(self, sc: dict) -> dict:
        """Format a signal dict for the dashboard scorecard display."""
        return {
            "symbol":           sc.get("symbol", ""),
            "signal":           sc.get("signal", sc.get("signal_label", "HOLD")),
            "score":            sc.get("normalised_score", 0),
            "overall_display":  sc.get("overall_display", "N/A"),
            "conviction":       sc.get("conviction", "INSUFFICIENT"),
            "bundle_name":      sc.get("bundle_name", ""),
            "bundle_emoji":     sc.get("bundle_emoji", ""),
            "asset_class":      sc.get("bundle_id", sc.get("asset_class", "")),
            "position_pct":     sc.get("position_pct", 0),
            "price":            sc.get("price", 0),
            "trade_id":         sc.get("trade_id"),
            "timestamp":        sc.get("timestamp"),
        }
