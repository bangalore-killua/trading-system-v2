"""Tests for strategy.chart_pattern_detector — YOLOv8 chart pattern detection."""

import pytest

from strategy.chart_pattern_detector import (
    ChartPatternDetector,
    PATTERN_CLASSES,
    BULLISH_PATTERNS,
    BEARISH_PATTERNS,
    _empty_patterns,
)


def test_empty_patterns_has_all_classes():
    empty = _empty_patterns()
    assert len(empty) == len(PATTERN_CLASSES)
    for cls in PATTERN_CLASSES:
        assert f"chart_{cls}" in empty


def test_empty_patterns_all_zero():
    empty = _empty_patterns()
    assert all(v == 0 for v in empty.values())


def test_detect_returns_all_zero_when_model_missing(ohlcv_trending_up):
    """Without a model file, detect() must return all zeros — not crash."""
    detector = ChartPatternDetector(
        model_path="models/nonexistent_model.pt",
        confidence_threshold=0.40,
    )
    result = detector.detect(ohlcv_trending_up)
    assert isinstance(result, dict)
    assert len(result) == len(PATTERN_CLASSES)
    assert all(v == 0 for v in result.values())


def test_detect_returns_empty_for_short_data():
    """DataFrames with < 30 bars should return all zeros without crashing."""
    import pandas as pd
    import numpy as np

    short_df = pd.DataFrame({
        "open":   [100.0] * 10,
        "high":   [102.0] * 10,
        "low":    [98.0]  * 10,
        "close":  [101.0] * 10,
        "volume": [1e6]   * 10,
    })
    detector = ChartPatternDetector(model_path="models/nonexistent.pt")
    result = detector.detect(short_df)
    assert all(v == 0 for v in result.values())


def test_pattern_classes_cover_expected_patterns():
    expected = {
        "head_and_shoulders", "inv_head_and_shoulders",
        "double_top", "double_bottom",
        "ascending_triangle", "descending_triangle",
        "bull_flag", "bear_flag",
        "cup_and_handle", "falling_wedge", "rising_wedge",
        "rounding_bottom",
    }
    assert expected.issubset(set(PATTERN_CLASSES))


def test_bullish_and_bearish_sets_disjoint():
    assert BULLISH_PATTERNS.isdisjoint(BEARISH_PATTERNS)


def test_bullish_patterns_subset_of_all_classes():
    assert BULLISH_PATTERNS.issubset(set(PATTERN_CLASSES))


def test_bearish_patterns_subset_of_all_classes():
    assert BEARISH_PATTERNS.issubset(set(PATTERN_CLASSES))


def test_detect_bullish_false_when_no_model(ohlcv_trending_up):
    """detect_bullish() should return False (not crash) when model missing."""
    detector = ChartPatternDetector(model_path="models/nonexistent.pt")
    assert detector.detect_bullish(ohlcv_trending_up) is False


def test_detect_bearish_false_when_no_model(ohlcv_trending_up):
    detector = ChartPatternDetector(model_path="models/nonexistent.pt")
    assert detector.detect_bearish(ohlcv_trending_up) is False


def test_model_warned_only_once(ohlcv_trending_up, caplog):
    """Missing model warning should only be emitted once regardless of call count."""
    import logging
    detector = ChartPatternDetector(model_path="models/nonexistent.pt")
    with caplog.at_level(logging.WARNING):
        detector.detect(ohlcv_trending_up)
        detector.detect(ohlcv_trending_up)
        detector.detect(ohlcv_trending_up)
    warning_count = sum(1 for r in caplog.records if "not found" in r.message)
    assert warning_count == 1
