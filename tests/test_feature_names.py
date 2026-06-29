"""
Data tests — verify feature_builder.py canonical feature list.
These catch renames, additions, or deletions that would silently break
model inference (training vs live feature mismatch).
"""

import pytest

from backtester.feature_builder import FEATURE_NAMES


def test_exactly_65_features():
    assert len(FEATURE_NAMES) == 65, \
        f"Expected 65 canonical features, got {len(FEATURE_NAMES)}. " \
        f"If you added/removed features, retrain AutoGluon models."


def test_all_names_unique():
    dupes = [n for n in FEATURE_NAMES if FEATURE_NAMES.count(n) > 1]
    assert not dupes, f"Duplicate feature names: {set(dupes)}"


def test_canonical_names_not_old_aliases():
    """Ensure old aliased names were replaced — these caused avg_hold=1 bug."""
    banned = {"rsi_14", "vol_ratio", "macd"}
    present = set(FEATURE_NAMES) & banned
    assert not present, \
        f"Old feature aliases still in FEATURE_NAMES: {present}. " \
        f"Use canonical names: rsi, volume_ratio, macd_hist"


def test_key_features_present():
    required = ["rsi", "macd_hist", "volume_ratio", "atr_pct",
                "ema9", "ema21", "adx", "vix"]
    missing = [f for f in required if f not in FEATURE_NAMES]
    assert not missing, f"Key features missing from FEATURE_NAMES: {missing}"


def test_candlestick_pattern_features_present():
    """These 6 contextual features are always 0 in backtester but must exist in schema."""
    patterns = [
        "hammer_at_200ma", "hammer_at_50ma", "engulfing_at_support",
        "marubozu_on_volume_surge", "morning_star_at_support", "evening_star_near_highs",
    ]
    missing = [p for p in patterns if p not in FEATURE_NAMES]
    assert not missing, f"Candlestick pattern features missing: {missing}"
