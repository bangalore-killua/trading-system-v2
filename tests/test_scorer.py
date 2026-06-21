"""Tests for strategy.scorer — feature vector builder."""

import pytest

from strategy.scorer import (
    FeatureVector,
    Scorer,
    REGIME_ENCODING,
    SETUP_TYPE_ENCODING,
    THREAT_ENCODING,
)
from strategy.chart_pattern_detector import _empty_patterns


@pytest.fixture
def basic_indicators():
    return {
        "rsi": 55.0,
        "macd_hist": 0.3,
        "price_vs_20ma": 2.5,
        "price_vs_50ma": 5.1,
        "price_vs_200ma": 12.0,
        "bb_position": 0.65,
        "bb_width": 8.0,
        "volume_ratio": 1.8,
        "atr_pct": 1.5,
        "adx": 28.0,
        "price_above_20ma": 1,
        "price_above_50ma": 1,
        "price_above_200ma": 1,
        "rsi_oversold": 0,
        "rsi_overbought": 0,
        "rsi_momentum_zone": 1,
        "macd_bullish": 1,
        "macd_bullish_cross": 0,
        "macd_bearish_cross": 0,
        "volume_surge": 0,
        "volume_confirming": 1,
        "bb_squeeze": 0,
        "daily_trend_bullish": 1,
        "body_size": 0.7,
        "body_direction": 1,
        "upper_wick": 0.1,
        "lower_wick": 0.2,
        "wick_ratio": 0.5,
        "candle_range_vs_atr": 0.9,
        "gap_up": 0,
        "gap_down": 0,
        "cdl_hammer": 0,
        "cdl_doji": 0,
        "cdl_engulfing": 0,
        "hammer_at_200ma": 0,
        "hammer_at_50ma": 0,
        "engulfing_at_support": 0,
        "marubozu_on_volume_surge": 0,
    }


# ── FeatureVector tests ──────────────────────────────────────────────────────

def test_add_technical_populates_rsi(basic_indicators):
    fv = FeatureVector()
    fv.add_technical(basic_indicators)
    d = fv.to_dict()
    assert d["rsi"] == 55.0
    assert d["price_above_20ma"] == 1
    assert d["cdl_hammer"] == 0


def test_add_chart_patterns_prefix_filter():
    fv = FeatureVector()
    patterns = _empty_patterns()
    patterns["chart_bull_flag"] = 1
    fv.add_chart_patterns(patterns)
    d = fv.to_dict()
    assert d["chart_bull_flag"] == 1
    assert d.get("chart_head_and_shoulders") == 0


def test_add_news_encoding():
    fv = FeatureVector()
    fv.add_news(finbert_score=0.82, news_volume_7d=5, threat_level="LOW")
    d = fv.to_dict()
    assert d["finbert_score"] == 0.82
    assert d["threat_level_encoded"] == THREAT_ENCODING["LOW"]


def test_add_macro_encoding():
    fv = FeatureVector()
    fv.add_macro({
        "vix": 18.5,
        "yield_spread_10y2y": 0.45,
        "fear_greed_proxy": "NEUTRAL",
        "spy_above_200ma": 1,
        "inverted_yield_curve": 0,
    })
    d = fv.to_dict()
    assert d["vix"] == 18.5
    assert d["fear_greed_encoded"] == 2  # NEUTRAL


def test_add_context_regime_encoded():
    fv = FeatureVector()
    fv.add_context(regime="MID_BULL", setup_type="SETUP_MOMENTUM", sector="Technology")
    d = fv.to_dict()
    assert d["regime_encoded"] == REGIME_ENCODING["MID_BULL"]
    assert d["setup_type_encoded"] == SETUP_TYPE_ENCODING["SETUP_MOMENTUM"]
    assert d["sector_encoded"] == 0  # Technology


def test_feature_count_reasonable(basic_indicators):
    fv = FeatureVector()
    fv.add_technical(basic_indicators)
    fv.add_chart_patterns(_empty_patterns())
    fv.add_fundamentals(None)
    fv.add_news()
    fv.add_macro(None)
    fv.add_context()
    # Should have at least 80 features
    assert fv.feature_count() >= 80


def test_to_list_ordered(basic_indicators):
    fv = FeatureVector()
    fv.add_technical(basic_indicators)
    feature_names = ["rsi", "macd_hist", "price_above_20ma"]
    values = fv.to_list(feature_names)
    assert values[0] == 55.0
    assert values[1] == 0.3
    assert values[2] == 1


def test_to_list_none_for_missing_key(basic_indicators):
    fv = FeatureVector()
    fv.add_technical(basic_indicators)
    values = fv.to_list(["rsi", "nonexistent_feature"])
    assert values[0] == 55.0
    assert values[1] is None


# ── Scorer tests ─────────────────────────────────────────────────────────────

def test_scorer_position_multiplier_default_1():
    """Without a meta-model, size multiplier must always be 1.0."""
    scorer = Scorer(config={"model": {"meta_model_enabled": False}})
    fv = FeatureVector()
    assert scorer.get_position_size_multiplier(fv) == 1.0


def test_scorer_build_vector_returns_feature_vector(basic_indicators, ohlcv_trending_up):
    scorer = Scorer(config={})
    fv = scorer.build_vector(
        symbol="AAPL",
        df=ohlcv_trending_up,
        indicators=basic_indicators,
        setup_type="SETUP_MOMENTUM",
        sector="Technology",
    )
    assert isinstance(fv, FeatureVector)
    assert fv.feature_count() > 50


def test_all_threat_levels_have_encoding():
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"):
        assert level in THREAT_ENCODING


def test_all_regime_states_have_encoding():
    for regime in ("EARLY_BULL", "MID_BULL", "LATE_BULL", "EUPHORIA",
                   "NEUTRAL", "EARLY_BEAR", "MID_BEAR", "CRASH"):
        assert regime in REGIME_ENCODING
