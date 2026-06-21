"""Tests for strategy.technical_analysis — pandas-ta indicators + candlestick patterns."""

import numpy as np
import pandas as pd
import pytest

from strategy.technical_analysis import compute_indicators, _pct_diff


def test_returns_empty_dict_for_short_data():
    """DataFrame with < 50 rows should return {} safely."""
    df = pd.DataFrame({
        "open": [10] * 20, "high": [11] * 20,
        "low": [9] * 20, "close": [10.5] * 20, "volume": [1e6] * 20,
    })
    result = compute_indicators(df)
    assert result == {}


def test_rsi_present_for_valid_data(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    assert "rsi" in result
    assert result["rsi"] is not None
    assert 0 < result["rsi"] < 100


def test_macd_keys_present(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    for key in ("macd", "macd_signal", "macd_hist", "macd_bullish"):
        assert key in result


def test_bollinger_keys_present(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    for key in ("bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_position"):
        assert key in result


def test_price_above_20ma_binary(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    assert result["price_above_20ma"] in (0, 1)


def test_volume_ratio_positive(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    assert result["volume_ratio"] > 0


def test_atr_reasonable_magnitude(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    # ATR % should be between 0.1% and 20% for a normal stock
    assert 0.1 < result["atr_pct"] < 20.0


def test_ohlc_geometry_keys_present(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    for key in ("body_size", "body_direction", "upper_wick", "lower_wick", "wick_ratio"):
        assert key in result


def test_body_direction_sign(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    assert result["body_direction"] in (-1, 1)


def test_candlestick_patterns_are_binary(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    # Verify at least one pattern key exists and is in {-1, 0, 1}
    pattern_keys = [k for k in result if k.startswith("cdl_")]
    assert len(pattern_keys) > 0
    for k in pattern_keys:
        assert result[k] in (-1, 0, 1), f"{k}={result[k]} not in (-1, 0, 1)"


def test_rsi_oversold_for_crashed_data(ohlcv_oversold):
    result = compute_indicators(ohlcv_oversold)
    # After a sharp 30-point drop the RSI should be oversold
    assert result["rsi_oversold"] == 1


def test_context_combos_present(ohlcv_trending_up):
    result = compute_indicators(ohlcv_trending_up)
    for key in ("hammer_at_200ma", "hammer_at_50ma", "engulfing_at_support",
                "marubozu_on_volume_surge"):
        assert key in result
        assert result[key] in (0, 1)


def test_pct_diff_none_for_zero_reference():
    assert _pct_diff(100, 0) is None
    assert _pct_diff(100, None) is None


def test_pct_diff_positive_when_price_above():
    assert _pct_diff(110, 100) == pytest.approx(10.0, abs=0.01)


def test_pct_diff_negative_when_price_below():
    assert _pct_diff(90, 100) == pytest.approx(-10.0, abs=0.01)
