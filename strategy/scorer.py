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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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

SECTOR_ENCODING = {
    "Technology": 0, "Healthcare": 1, "Financials": 2, "Consumer Discretionary": 3,
    "Industrials": 4, "Communication Services": 5, "Consumer Staples": 6,
    "Energy": 7, "Materials": 8, "Real Estate": 9, "Utilities": 10, "Unknown": 11,
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
        sector: str = "Unknown",
        mktcap_bucket: str = "mid",
        days_since_earnings: Optional[int] = None,
    ) -> "FeatureVector":
        """Add categorical context features (encoded as integers for LightGBM)."""
        self._data["regime_encoded"] = REGIME_ENCODING.get(regime, 4)
        self._data["setup_type_encoded"] = SETUP_TYPE_ENCODING.get(setup_type, 0)
        self._data["sector_encoded"] = SECTOR_ENCODING.get(sector, 11)
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

        fv.add_context(
            regime=regime,
            setup_type=setup_type,
            sector=sector,
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
